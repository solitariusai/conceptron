import jax
import jax.numpy as jnp
from taktiny import nn

from conceptron._parts import (
    ConceptronAttention,
    ConceptronCache,
    ConceptronMLP,
    ConceptronRMSNorm,
    ConceptronRoPE,
    ConceptronTokenEmbedding,
    rotate_half,
)
from conceptron.exp.uztkqi93.conf import ControlConfig


class Attention_uztkqi93(ConceptronAttention):
    def __init__(self, config: ControlConfig, *, rngs: nn.Rngs):
        super().__init__(config, rngs=rngs, quant=None)

    def __call__(
        self, 
        x: jax.Array, 
        mask: jax.Array | None = None,
        position_embedding: tuple[jax.Array, jax.Array] | None = None, 
        cache: ConceptronCache | None = None,
        layer_idx: jax.Array | int | None = None
    ):  # ty: ignore[invalid-method-override]
        x = self.qkv_proj(x)
        dtype = x.dtype
        k, v, q = jnp.split(x, [self.num_key_value_heads, 2 * self.num_key_value_heads], axis=-2)
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
        
class Decoder(nn.Module):
    def __init__(self, config: ControlConfig, *, rngs: nn.Rngs):
        self.norm1 = ConceptronRMSNorm(config)
        self.norm2 = ConceptronRMSNorm(config)
        self.attn = Attention_uztkqi93(config, rngs=rngs)
        self.mlp = ConceptronMLP(config, rngs=rngs, quant=None)

    def __call__(
        self, 
        x: jax.Array, 
        mask: jax.Array | None = None,
        position_embedding: tuple[jax.Array, jax.Array] | None = None, 
        cache: ConceptronCache | None = None,
        layer_idx: jax.Array | int | None = None,
        add_residual: bool = True
    ):
        res = x
        x = self.attn(self.norm1(x), mask, position_embedding, cache, layer_idx)
        if add_residual:
            x = x + res

        res = x
        x = self.mlp(self.norm2(x))
        if add_residual:
            x = x + res

        return x

class Exp_uztkqi93(nn.Module):
    def __init__(self, config: ControlConfig, *, rngs: nn.Rngs):
        self.wte = ConceptronTokenEmbedding(config, rngs=rngs)
        self.layer = Decoder(config, rngs=rngs)
        self.layers = nn.SeqStack([Decoder(config, rngs=rngs) for _ in range(config.num_layers - 1)])
        self.norm = ConceptronRMSNorm(config)
        self.lm_head = jax.new_ref(self.wte.embedding.value.T)
        self.rope = ConceptronRoPE(config.head_dims, config.rope_theta)

    def __call__(
        self, 
        ids: jax.Array, 
        mask: jax.Array | None = None,
        position_ids: jax.Array | None = None, 
        cache: ConceptronCache | None = None,
        base: bool = True,
    ) -> jax.Array:
        x = jax.checkpoint(self.wte)(ids)
        if position_ids is None:
            if cache is not None:
                start_idx = cache.position_idx[...]
            else:
                start_idx = 0

            position_ids = start_idx + jnp.arange(x.shape[1])

        position_embedding = self.rope(position_ids)
        x = jax.checkpoint(self.layer)(x, mask, position_embedding, cache, 0)

        if not base:
            position_embedding = None

        layer_idx = jax.new_ref(jnp.asarray(1, dtype='uint32'))
        def fwd_layer(layer, x, layer_idx):
            x = jax.checkpoint(layer)(x, mask, position_embedding, cache, layer_idx[...])
            layer_idx[...] += 1
            return x, None
            
        x, _ = self.layers(fwd_layer, x, layer_idx)
        x = jax.checkpoint(self.norm)(x)
        logits = jax.checkpoint(jnp.dot)(x, self.lm_head[...])
        if cache is not None:
            cache.advance(logits.shape[1])
            
        return logits

class Exp_uztkqi93_2(nn.Module):
    def __init__(self, config: ControlConfig, *, rngs: nn.Rngs):
        self.wte = ConceptronTokenEmbedding(config, rngs=rngs)
        self.layer = Decoder(config, rngs=rngs)
        self.layers = nn.SeqStack([Decoder(config, rngs=rngs) for _ in range(config.num_layers - 1)])
        self.norm = ConceptronRMSNorm(config)
        self.lm_head = jax.new_ref(self.wte.embedding.value.T)
        self.rope = ConceptronRoPE(config.head_dims, config.rope_theta)

    def __call__(
        self, 
        ids: jax.Array, 
        mask: jax.Array | None = None,
        position_ids: jax.Array | None = None, 
        cache: ConceptronCache | None = None,
        base: bool = False, # different from `Exp_uztkqi93.__call__` which default is `True`
    ) -> jax.Array:
        x = jax.checkpoint(self.wte)(ids)
        if position_ids is None:
            if cache is not None:
                start_idx = cache.position_idx[...]
            else:
                start_idx = 0

            position_ids = start_idx + jnp.arange(x.shape[1])

        position_embedding = self.rope(position_ids)
        x = jax.checkpoint(self.layer, static_argnums=5)(x, mask, position_embedding, cache, 0, False) # not add residual in the first layer

        if not base:
            position_embedding = None

        layer_idx = jax.new_ref(jnp.asarray(1, dtype='uint32'))
        def fwd_layer(layer, x, layer_idx):
            x = jax.checkpoint(layer)(x, mask, position_embedding, cache, layer_idx[...])
            layer_idx[...] += 1
            return x, None
            
        x, _ = self.layers(fwd_layer, x, layer_idx)
        x = jax.checkpoint(self.norm)(x)
        logits = jax.checkpoint(jnp.dot)(x, self.lm_head[...])
        if cache is not None:
            cache.advance(logits.shape[1])
            
        return logits

class Exp_uztkqi93_3(nn.Module):
    def __init__(self, config: ControlConfig, *, rngs: nn.Rngs):
        self.wte = ConceptronTokenEmbedding(config, rngs=rngs)
        self.layers = nn.SeqStack([Decoder(config, rngs=rngs) for _ in range(config.num_layers)])
        self.norm = ConceptronRMSNorm(config)
        self.lm_head = jax.new_ref(self.wte.embedding.value.T)
        self.rope = ConceptronRoPE(config.head_dims, config.rope_theta)

    def __call__(
        self, 
        ids: jax.Array, 
        mask: jax.Array | None = None,
        position_ids: jax.Array | None = None, 
        cache: ConceptronCache | None = None,
    ) -> jax.Array:
        x = jax.checkpoint(self.wte)(ids)
        if position_ids is None:
            if cache is not None:
                start_idx = cache.position_idx[...]
            else:
                start_idx = 0

            position_ids = start_idx + jnp.arange(x.shape[1])

        position_embedding = self.rope(position_ids)
        z = x
        layer_idx = jax.new_ref(jnp.asarray(0, dtype='uint32'))
        def fwd_layer(layer, z, layer_idx):
            z = jax.checkpoint(layer, static_argnums=5)(x, mask, position_embedding, cache, layer_idx[...], False) + z
            layer_idx[...] += 1
            return z, None
            
        z, _ = self.layers(fwd_layer, z, layer_idx)
        x = x + z
        x = jax.checkpoint(self.norm)(x)
        logits = jax.checkpoint(jnp.dot)(x, self.lm_head[...])
        if cache is not None:
            cache.advance(logits.shape[1])
            
        return logits

class Exp_uztkqi93_4(nn.Module):
    def __init__(self, config: ControlConfig, *, rngs: nn.Rngs):
        self.wte = ConceptronTokenEmbedding(config, rngs=rngs)
        num_layers_1 = config.num_layers // 2
        num_layers_2 = config.num_layers - num_layers_1
        self.layers_1 = nn.SeqStack([Decoder(config, rngs=rngs) for _ in range(num_layers_1)])
        self.layers_2 = nn.SeqStack([Decoder(config, rngs=rngs) for _ in range(num_layers_2)])
        self.norm = ConceptronRMSNorm(config)
        self.lm_head = jax.new_ref(self.wte.embedding.value.T)
        self.rope = ConceptronRoPE(config.head_dims, config.rope_theta)

    def __call__(
        self, 
        ids: jax.Array, 
        mask: jax.Array | None = None,
        position_ids: jax.Array | None = None, 
        cache: ConceptronCache | None = None,
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
        def fwd_layer(layer, z, layer_idx):
            z = jax.checkpoint(layer, static_argnums=5)(x, mask, position_embedding, cache, layer_idx[...], False) + z
            layer_idx[...] += 1
            return z, None
            
        z = x
        z, _ = self.layers_1(fwd_layer, z, layer_idx)
        x = x + z
        z = x
        z, _ = self.layers_2(fwd_layer, z, layer_idx)
        x = x + z
        x = jax.checkpoint(self.norm)(x)
        logits = jax.checkpoint(jnp.dot)(x, self.lm_head[...])
        if cache is not None:
            cache.advance(logits.shape[1])
            
        return logits

class Exp_uztkqi93_5(nn.Module):
    def __init__(self, config: ControlConfig, *, rngs: nn.Rngs):
        self.wte = ConceptronTokenEmbedding(config, rngs=rngs)
        num_layers_quater = config.num_layers // 4
        num_layers_remain = config.num_layers - num_layers_quater * 3
        self.layers_1 = nn.SeqStack([Decoder(config, rngs=rngs) for _ in range(num_layers_quater)])
        self.layers_2 = nn.SeqStack([Decoder(config, rngs=rngs) for _ in range(num_layers_quater)])
        self.layers_3 = nn.SeqStack([Decoder(config, rngs=rngs) for _ in range(num_layers_quater)])
        self.layers_4 = nn.SeqStack([Decoder(config, rngs=rngs) for _ in range(num_layers_remain)])
        self.norm = ConceptronRMSNorm(config)
        self.lm_head = jax.new_ref(self.wte.embedding.value.T)
        self.rope = ConceptronRoPE(config.head_dims, config.rope_theta)

    def __call__(
        self, 
        ids: jax.Array, 
        mask: jax.Array | None = None,
        position_ids: jax.Array | None = None, 
        cache: ConceptronCache | None = None,
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
        def fwd_layer(layer, z, x, layer_idx):
            z = jax.checkpoint(layer, static_argnums=5)(x, mask, position_embedding, cache, layer_idx[...], False) + z
            layer_idx[...] += 1
            return z, None
            
        z = x
        z, _ = self.layers_1(fwd_layer, z, x, layer_idx)
        x = x + z

        z = x
        z, _ = self.layers_2(fwd_layer, z, x, layer_idx)
        x = x + z

        z = x
        z, _ = self.layers_3(fwd_layer, z, x, layer_idx)
        x = x + z

        z = x
        z, _ = self.layers_4(fwd_layer, z, x, layer_idx)
        x = x + z

        x = jax.checkpoint(self.norm)(x)
        logits = jax.checkpoint(jnp.dot)(x, self.lm_head[...])
        if cache is not None:
            cache.advance(logits.shape[1])
            
        return logits

__all__ = ['Exp_uztkqi93', 'Exp_uztkqi93_2', 'Exp_uztkqi93_3', 'Exp_uztkqi93_4', 'Exp_uztkqi93_5']