from collections.abc import Callable

import jax
import jax.numpy as jnp
from taktiny import nn

from conceptron._parts import (
    ConceptronCache,
    ConceptronRMSNorm,
    ConceptronRoPE,
    ConceptronTokenEmbedding,
    rotate_half,
)
from conceptron.exp.gu1gkjy4.conf import ControlConfig


class Attention_gu1gkjy4(nn.Module):
    def __init__(self, config: ControlConfig, *, rngs: nn.Rngs, shared: bool = True):
        kernel_size = 3
        self.qkv_proj = None
        if shared:
            self.qkv_proj = nn.Conv(
                config.hidden_size, 
                (config.num_heads + config.num_key_value_heads * 2, config.head_dim), 
                kernel_size,
                rngs=rngs, 
                bias=False, 
                padding=(kernel_size - 1, 0),
                dtype=config.dtype
            )
        else:
            self.q_proj = nn.Conv(
                config.hidden_size, 
                (config.num_heads, config.head_dim), 
                kernel_size,
                rngs=rngs, 
                bias=False, 
                padding=(kernel_size - 1, 0),
                dtype=config.dtype
            )
            self.k_proj = nn.Conv(
                config.hidden_size, 
                (config.num_key_value_heads, config.head_dim), 
                kernel_size,
                rngs=rngs, 
                bias=False, 
                padding=(kernel_size - 1, 0),
                dtype=config.dtype
            )
            self.v_proj = nn.Conv(
                config.hidden_size, 
                (config.num_key_value_heads, config.head_dim), 
                kernel_size,
                rngs=rngs, 
                bias=False, 
                padding=(kernel_size - 1, 0),
                dtype=config.dtype
            )
        self.o_proj = nn.Conv(
            (config.num_heads, config.head_dim), 
            config.hidden_size, 
            kernel_size,
            rngs=rngs, 
            bias=False, 
            padding=(kernel_size - 1, 0),
            dtype=config.dtype
        )
        self.num_key_value_heads = config.num_key_value_heads

    def __call__(
        self, 
        x: jax.Array, 
        mask: jax.Array | None = None,
        position_embedding: tuple[jax.Array, jax.Array] | None = None, 
        cache: ConceptronCache | None = None,
        layer_idx: jax.Array | int | None = None
    ):
        if self.qkv_proj is not None:
            x = self.qkv_proj(x)
            k, v, q = jnp.split(x, [self.num_key_value_heads, 2 * self.num_key_value_heads], axis=-2)
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

        dtype = x.dtype
        if position_embedding is not None:
            cos, sin = position_embedding # shape [T, H]
            if cos.ndim == 3:
                cos = cos[:, :, None, :]
                sin = sin[:, :, None, :]

            elif cos.ndim == 2:
                cos = cos[None, :, None, :]
                sin = sin[None, :, None, :]

            q = (q * cos + rotate_half(q) * sin).astype(dtype)
            k = (k * cos + rotate_half(k) * sin).astype(dtype)

        if cache is not None:
            assert layer_idx is not None
            cache.update(k, v, layer_idx)
            k, v= cache.get(layer_idx)
            q_pos = cache.position_idx[...] + jnp.arange(q.shape[1])
            cache_mask = (
                jnp.arange(k.shape[1])[None, :] <= q_pos[:, None]
            )
            mask = cache_mask if mask is None else mask & cache_mask
        
        o = jax.nn.dot_product_attention(
            q, k, v, mask=mask
        )
        return self.o_proj(o).astype(dtype)
        
class MLP_gu1gkjy4(nn.Module):
    def __init__(self, config: ControlConfig, *, rngs: nn.Rngs, shared: bool = True):
        kernel_size = 3
        self.gate_proj = None
        if shared:
            self.up_proj = nn.Conv(
                config.hidden_size, 
                config.inter_size * 2, 
                kernel_size,
                rngs=rngs, 
                bias=False, 
                padding=(kernel_size - 1, 0),
                dtype=config.dtype
            )
        else:
            self.gate_proj = nn.Conv(
                config.hidden_size, 
                config.inter_size, 
                kernel_size,
                rngs=rngs, 
                bias=False, 
                padding=(kernel_size - 1, 0),
                dtype=config.dtype
            )
            self.up_proj = nn.Conv(
                config.hidden_size, 
                config.inter_size, 
                kernel_size,
                rngs=rngs, 
                bias=False, 
                padding=(kernel_size - 1, 0),
                dtype=config.dtype
            )

        self.down_proj = nn.Conv(
            config.inter_size, 
            config.hidden_size, 
            kernel_size,
            rngs=rngs, 
            bias=False, 
            padding=(kernel_size - 1, 0),
            dtype=config.dtype
        )

    def __call__(self, x: jax.Array):
        if self.gate_proj is None:
            x = self.up_proj(x)
            x1, x2 = jnp.split(x, 2, -1)
        else:
            x1 = self.up_proj(x)
            x2 = self.gate_proj(x)

        return self.down_proj(jax.nn.silu(x1) * x2)

class Decoder(nn.Module):
    def __init__(
        self, 
        config: ControlConfig, 
        *, 
        rngs: nn.Rngs, 
        shared_attn: bool = True, 
        shared_mlp: bool = True
    ):
        self.norm1 = ConceptronRMSNorm(config)
        self.norm2 = ConceptronRMSNorm(config)
        self.attn = Attention_gu1gkjy4(config, rngs=rngs, shared=shared_attn)
        self.mlp = MLP_gu1gkjy4(config, rngs=rngs, shared=shared_mlp)

    def __call__(
        self, 
        x: jax.Array, 
        mask: jax.Array | None = None,
        position_embedding: tuple[jax.Array, jax.Array] | None = None, 
        cache: ConceptronCache | None = None,
        layer_idx: jax.Array | int | None = None,
    ):
        res = x
        x = self.attn(self.norm1(x), mask, position_embedding, cache, layer_idx)
        x = x + res

        res = x
        x = self.mlp(self.norm2(x))
        x = x + res

        return x

class Exp_gu1gkjy4_shared_attn(nn.Module):
    def __init__(self, config: ControlConfig, *, rngs: nn.Rngs):
        self.wte = ConceptronTokenEmbedding(config, rngs=rngs)
        self.layers = nn.SeqStack([Decoder(config, rngs=rngs, shared_attn=True, shared_mlp=False) for _ in range(config.num_layers)])
        self.norm = ConceptronRMSNorm(config)
        self.lm_head = jax.new_ref(self.wte.embedding.value.T)
        self.rope = ConceptronRoPE(config.head_dim, config.rope_theta)

    def __call__(
        self, 
        ids: jax.Array, 
        mask: jax.Array | None = None,
        position_ids: jax.Array | None = None, 
        cache: ConceptronCache | None = None,
        loss_fn: Callable | None = None,
    ) -> jax.Array:
        x = jax.checkpoint(self.wte)(ids)
        if position_ids is None:
            if cache is not None:
                start_idx = cache.position_idx[...]
            else:
                start_idx = 0

            position_ids = start_idx + jnp.arange(x.shape[1])

        position_embedding = self.rope(position_ids)
        layer_idx = jax.new_ref(jnp.asarray(0, dtype='uint32'))
        def fwd_layer(layer, x, layer_idx):
            x = jax.checkpoint(layer)(x, mask, position_embedding, cache, layer_idx[...])
            layer_idx[...] += 1
            return x, None
            
        x, _ = self.layers(fwd_layer, x, layer_idx)
        x = jax.checkpoint(self.norm)(x)
        if loss_fn is not None:
            return loss_fn(x, self.lm_head)
        else:
            logits = jax.checkpoint(jnp.dot)(x, self.lm_head[...])

        if cache is not None:
            cache.advance(logits.shape[1])
            
        return logits

class Exp_gu1gkjy4_shared_mlp(nn.Module):
    def __init__(self, config: ControlConfig, *, rngs: nn.Rngs):
        self.wte = ConceptronTokenEmbedding(config, rngs=rngs)
        self.layers = nn.SeqStack([Decoder(config, rngs=rngs, shared_attn=False, shared_mlp=True) for _ in range(config.num_layers)])
        self.norm = ConceptronRMSNorm(config)
        self.lm_head = jax.new_ref(self.wte.embedding.value.T)
        self.rope = ConceptronRoPE(config.head_dim, config.rope_theta)

    def __call__(
        self, 
        ids: jax.Array, 
        mask: jax.Array | None = None,
        position_ids: jax.Array | None = None, 
        cache: ConceptronCache | None = None,
        loss_fn: Callable | None = None,
    ) -> jax.Array:
        x = jax.checkpoint(self.wte)(ids)
        if position_ids is None:
            if cache is not None:
                start_idx = cache.position_idx[...]
            else:
                start_idx = 0

            position_ids = start_idx + jnp.arange(x.shape[1])

        position_embedding = self.rope(position_ids)
        layer_idx = jax.new_ref(jnp.asarray(0, dtype='uint32'))
        def fwd_layer(layer, x, layer_idx):
            remat_layer = jax.checkpoint(layer)
            x = remat_layer(x, mask, position_embedding, cache, layer_idx[...])
            layer_idx[...] += 1
            return x, None
            
        x, _ = self.layers(fwd_layer, x, layer_idx)
        x = jax.checkpoint(self.norm)(x)
        
        if loss_fn is not None:
            return loss_fn(x, self.lm_head)
        else:
            logits = jax.checkpoint(jnp.dot)(x, self.lm_head[...])

        if cache is not None:
            cache.advance(logits.shape[1])
            
        return logits

class Exp_gu1gkjy4_independent(nn.Module):
    def __init__(self, config: ControlConfig, *, rngs: nn.Rngs):
        self.wte = ConceptronTokenEmbedding(config, rngs=rngs)
        self.layers = nn.SeqStack([Decoder(config, rngs=rngs, shared_attn=False, shared_mlp=False) for _ in range(config.num_layers)])
        self.norm = ConceptronRMSNorm(config)
        self.lm_head = jax.new_ref(self.wte.embedding.value.T)
        self.rope = ConceptronRoPE(config.head_dim, config.rope_theta)

    def __call__(
        self, 
        ids: jax.Array, 
        mask: jax.Array | None = None,
        position_ids: jax.Array | None = None, 
        cache: ConceptronCache | None = None,
        loss_fn: Callable | None = None,
    ) -> jax.Array:
        x = jax.checkpoint(self.wte)(ids)
        if position_ids is None:
            if cache is not None:
                start_idx = cache.position_idx[...]
            else:
                start_idx = 0

            position_ids = start_idx + jnp.arange(x.shape[1])

        position_embedding = self.rope(position_ids)
        layer_idx = jax.new_ref(jnp.asarray(0, dtype='uint32'))
        def fwd_layer(layer, x, layer_idx):
            remat_layer = jax.checkpoint(layer)
            x = remat_layer(x, mask, position_embedding, cache, layer_idx[...])
            layer_idx[...] += 1
            return x, None
            
        x, _ = self.layers(fwd_layer, x, layer_idx)
        x = jax.checkpoint(self.norm)(x)
        
        if loss_fn is not None:
            return loss_fn(x, self.lm_head)
        else:
            logits = jax.checkpoint(jnp.dot)(x, self.lm_head[...])
            
        if cache is not None:
            cache.advance(logits.shape[1])
            
        return logits

__all__ = [
    'Exp_gu1gkjy4_independent',
    'Exp_gu1gkjy4_shared_attn',
    'Exp_gu1gkjy4_shared_mlp',
]