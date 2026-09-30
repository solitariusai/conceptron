from typing import Any

import jax
import jax.numpy as jnp
from taktiny import nn
from taktiny.utils.typing import QuantConfig

from conceptron._parts import ConceptronMLP, product_attention, rotate_half
from conceptron.arachne.config import ArachneConfig


class Attention(nn.Module):
    def __init__(self, config: ArachneConfig, *, rngs: nn.Rngs, quant: QuantConfig = None, is_final: bool = False):
        if config.num_key_value_heads < 1 or config.num_heads % config.num_key_value_heads:
            raise ValueError('num_heads must be a multiple of num_key_value_heads')

        self.num_heads = config.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.qkv_proj = nn.Linear(
            config.hidden_size, 
            (config.num_heads + config.num_key_value_heads * 2, config.head_dims),
            rngs=rngs,
            bias=False,
            quant=quant,
            axis_names=('hidden', 'num_heads', 'head_dim')
        )
        self.o_proj = nn.Linear(
            (config.num_heads, config.head_dims),
            config.hidden_size, 
            rngs=rngs,
            bias=False,
            quant=quant,
            axis_names=('num_heads', 'head_dim', 'hidden')
        )
        self.r_proj = None
        if not is_final:
            self.r_proj = nn.Linear(
                (config.num_heads, config.head_dims),
                config.hidden_size,
                rngs=rngs,
                bias=False,
                quant=quant,
                axis_names=('num_heads', 'head_dim', 'hidden')
            )

    def __call__(self, x: jax.Array, ctx: dict[str, Any] | None = None):
        """Apply attention using optional rotary embeddings and a KV cache.

        ``ctx['position_embedding']`` is a ``(cos, sin)`` pair for the current
        tokens, each shaped [sequence, head_dim] or [batch, sequence, head_dim].
        ``ctx['kv_cache']`` is a dict with ``key`` and ``value`` arrays
        shaped [batch, cached_sequence, kv_heads, head_dim]. An empty dict
        starts a new cache. Returns ``(o, r, new_ctx)`` with the updated cache
        and all other context entries preserved. The input ``ctx`` is unchanged.
        """
        if ctx is None:
            ctx = {}
        qkv = self.qkv_proj(x)
        q, k, v = jnp.split(
            qkv,
            (self.num_heads, self.num_heads + self.num_key_value_heads),
            axis=-2,
        )

        if 'position_embedding' in ctx:
            cos, sin = (jnp.asarray(part) for part in ctx['position_embedding'])
            if cos.shape != sin.shape or cos.shape[-1] != q.shape[-1]:
                raise ValueError('position_embedding cos and sin must match the head dimension')
            if q.shape[-1] % 2:
                raise ValueError('rotary position embeddings require an even head dimension')
            if cos.ndim == 2:
                cos, sin = cos[None, :, None, :], sin[None, :, None, :]
            elif cos.ndim == 3:
                cos, sin = cos[:, :, None, :], sin[:, :, None, :]
            elif cos.ndim != 4 or cos.shape[-2] != 1:
                raise ValueError('position_embedding must have shape [T, D], [B, T, D], or [B, T, 1, D]')

            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin

        query_start = 0
        if 'kv_cache' in ctx:
            cache = ctx['kv_cache']
            if not isinstance(cache, dict):
                raise TypeError('kv_cache must be a dict')
            if ('key' in cache) != ('value' in cache):
                raise ValueError('kv_cache must contain both key and value')
            if 'key' in cache:
                previous_key, previous_value = cache['key'], cache['value']
                if (previous_key.ndim != 4 or previous_key.shape != previous_value.shape
                        or previous_key.shape[0] != k.shape[0]
                        or previous_key.shape[2:] != k.shape[2:]):
                    raise ValueError('cached key and value must match the current key shape')
                query_start = previous_key.shape[1]
                k = jnp.concatenate((previous_key, k), axis=1)
                v = jnp.concatenate((previous_value, v), axis=1)

        attended = product_attention(q, k, v, query_start=query_start)
        new_cache = {'key': k, 'value': v}
        new_ctx = {**ctx, 'kv_cache': new_cache}
        return self.o_proj(attended), self.r_proj(attended) if self.r_proj is not None else None, new_ctx


class Router(nn.Module):
    def __init__(self, config: ArachneConfig, *, rngs: nn.Rngs, quant: QuantConfig = None):
        if config.num_layers < 1:
            raise ValueError('num_layers must be at least 1')

        self.p_heads = nn.Linear(
            config.hidden_size, 
            config.num_layers,
            rngs=rngs,
            bias=False,
            quant=quant,
            axis_names=('hidden', 'num_layers')
        )
        self.used_mask = jnp.zeros(config.num_layers, dtype=jnp.bool_)

    def __call__(self, x: jax.Array):
        logits = jnp.mean(self.p_heads(x), axis=(0, 1))  # [num_layers]
        logits = jnp.where(self.used_mask, -jnp.inf, logits)
        probs = jax.nn.softmax(logits)
        index = jnp.argmax(probs)
        self.used_mask = self.used_mask | (jnp.arange(logits.shape[0]) == index)
        return index

class Decoder(nn.Module):
    def __init__(self, config: ArachneConfig, *, rngs: nn.Rngs, quant: QuantConfig = None, is_final: bool = False):
        self.norm1 = nn.RMSNorm(config.hidden_size, 1e-7, dtype='float32', axis_names=('hidden',))
        self.norm2 = nn.RMSNorm(config.hidden_size, 1e-7, dtype='float32', axis_names=('hidden',))
        self.attn = Attention(config, rngs=rngs, quant=quant, is_final=is_final)
        self.mlp = ConceptronMLP(config, rngs=rngs, quant=quant)

    def __call__(self, x: jax.Array, ctx: dict | None = None):
        res = x
        x, r, ctx = self.attn(self.norm1(x), ctx)
        x = x + res
        x = self.mlp(self.norm2(x)) + x
        return x, r, ctx


class ArachneModel(nn.Module):
    def __init__(self, config: ArachneConfig, *, rngs: nn.Rngs, quant: QuantConfig = None):
        self.wte = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            rngs=rngs,
            quant=quant,
            axis_names=('vocab', 'hidden')
        )
        self.router = Router(config, rngs=rngs, quant=quant)
        self.init_layer = Decoder(config, rngs=rngs, quant=quant)
        self.final_layer = Decoder(config, rngs=rngs, quant=quant, is_final=True)
        self.layers = nn.List([Decoder(config, rngs=rngs, quant=quant) for _ in range(config.num_layers)])
        self.norm = nn.RMSNorm(config.hidden_size, 1e-7, dtype='float32', axis_names=('hidden',))

    def __call__(self, ids: jax.Array, ctx: dict | None = None):
        x = self.wte(ids)
        x, r, ctx = self.init_layer(x, ctx)
        for _ in range(len(self.layers)):
            i = self.router(r)
            x, r, ctx = self.layers[i](x, ctx)

        x, _, ctx = self.final_layer(x, ctx)
        x = self.norm(x)
        return x, ctx