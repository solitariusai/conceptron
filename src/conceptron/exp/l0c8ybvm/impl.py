from jax.sharding import PartitionSpec
from collections.abc import Callable

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
from conceptron.exp.l0c8ybvm.conf import ControlConfig


class Attention_l0c8ybvm(ConceptronAttention):
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
        self.attn = Attention_l0c8ybvm(config, rngs=rngs)
        self.mlp = ConceptronMLP(config, rngs=rngs, quant=None)

    def __call__(
        self, 
        x: jax.Array, 
        mask: jax.Array | None = None,
        position_embedding: tuple[jax.Array, jax.Array] | None = None, 
        cache: ConceptronCache | None = None,
        layer_idx: jax.Array | int | None = None,
        add_attn_residual: bool = True,
        add_mlp_residual: bool = False,
    ):
        res = x
        x = self.attn(self.norm1(x), mask, position_embedding, cache, layer_idx)
        if add_attn_residual:
            x = x + res

        res = x
        x = self.mlp(self.norm2(x))
        if add_mlp_residual:
            x = x + res

        return x

class Exp_l0c8ybvm_sum(nn.Module):
    def __init__(self, config: ControlConfig, *, rngs: nn.Rngs):
        self.wte = ConceptronTokenEmbedding(config, rngs=rngs)
        k = 4
        assert config.num_layers % k == 0, 'num_layers should divisble by 4'
        num_layers_quater = config.num_layers // k
        self.layers = [nn.SeqStack([Decoder(config, rngs=rngs) for _ in range(num_layers_quater)]) for _ in range(k)]
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
            z = jax.checkpoint(layer)(x, mask, position_embedding, cache, layer_idx[...]) + z
            layer_idx[...] += 1
            return z, None

        for layer in self.layers:
            z = x
            z, _ = layer(fwd_layer, z, x, layer_idx)
            x = z

        x = jax.checkpoint(self.norm)(x)
        logits = jax.checkpoint(jnp.dot)(x, self.lm_head[...])
        if cache is not None:
            cache.advance(logits.shape[1])
            
        return logits

class Exp_l0c8ybvm_mean(nn.Module):
    def __init__(self, config: ControlConfig, *, rngs: nn.Rngs):
        self.wte = ConceptronTokenEmbedding(config, rngs=rngs)
        k = 4
        assert config.num_layers % k == 0, 'num_layers should divisble by 4'
        num_layers_quater = config.num_layers // k
        self.layers = [nn.SeqStack([Decoder(config, rngs=rngs) for _ in range(num_layers_quater)]) for _ in range(k)]
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
            z = jax.checkpoint(layer)(x, mask, position_embedding, cache, layer_idx[...]) + z
            layer_idx[...] += 1
            return z, None

        k = len(self.layers)
        for layer in self.layers:
            z = x
            z, _ = layer(fwd_layer, z, x, layer_idx)
            x = z / k

        x = jax.checkpoint(self.norm)(x)
        logits = jax.checkpoint(jnp.dot)(x, self.lm_head[...])
        if cache is not None:
            cache.advance(logits.shape[1])
            
        return logits

class Exp_l0c8ybvm_weight_sum(nn.Module):
    def __init__(self, config: ControlConfig, *, rngs: nn.Rngs):
        self.wte = ConceptronTokenEmbedding(config, rngs=rngs)
        k = 4
        assert config.num_layers % k == 0, 'num_layers should divisble by 4'
        num_layers_quater = config.num_layers // k
        self.layers = [nn.SeqStack([Decoder(config, rngs=rngs) for _ in range(num_layers_quater)]) for _ in range(k)]
        self.norm = ConceptronRMSNorm(config)
        self.lm_head = jax.new_ref(self.wte.embedding.value.T)
        self.rope = ConceptronRoPE(config.head_dims, config.rope_theta)
        self.w = nn.Parameter((1 / num_layers_quater) * jnp.ones((k, num_layers_quater), 'float32'), partition_spec=PartitionSpec())

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
        def fwd_layer(layer, z, x, layer_idx, w, w_idx):
            w = w[w_idx[...]]
            z = (jax.checkpoint(layer)(x, mask, position_embedding, cache, layer_idx[...]) * w).astype(dtype) + z
            layer_idx[...] += 1
            w_idx[...] += 1
            return z, None

        dtype = x.dtype
        w_idx = jax.new_ref(jnp.asarray(0, dtype='uint32'))
        for idx, layer in enumerate(self.layers):
            w = self.w[idx]
            z = x
            z, _ = layer(fwd_layer, z, x, layer_idx, w, w_idx)

            w_idx[...] = jnp.asarray(0, dtype='uint32')
            x = z

        x = jax.checkpoint(self.norm)(x)
        if loss_fn is not None:
            return loss_fn(x, self.lm_head)
        else:
            logits = jax.checkpoint(jnp.dot)(x, self.lm_head[...])

        if cache is not None:
            cache.advance(logits.shape[1])
            
        return logits

class Exp_l0c8ybvm_weight_sum_bias(nn.Module):
    def __init__(self, config: ControlConfig, *, rngs: nn.Rngs):
        self.wte = ConceptronTokenEmbedding(config, rngs=rngs)
        k = 4
        assert config.num_layers % k == 0, 'num_layers should divisble by 4'
        num_layers_quater = config.num_layers // k
        self.layers = [nn.SeqStack([Decoder(config, rngs=rngs) for _ in range(num_layers_quater)]) for _ in range(k)]
        self.norm = ConceptronRMSNorm(config)
        self.lm_head = jax.new_ref(self.wte.embedding.value.T)
        self.rope = ConceptronRoPE(config.head_dims, config.rope_theta)
        self.w = nn.Parameter((1 / num_layers_quater) * jnp.ones((k, num_layers_quater), 'float32'), partition_spec=PartitionSpec())
        self.b = nn.Parameter(jnp.zeros((k,), 'float32'), partition_spec=PartitionSpec())

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
        def fwd_layer(layer, z, x, layer_idx, w, w_idx):
            w = w[w_idx[...]]
            z = (jax.checkpoint(layer)(x, mask, position_embedding, cache, layer_idx[...]) * w).astype(dtype) + z
            layer_idx[...] += 1
            w_idx[...] += 1
            return z, None

        dtype = x.dtype
        w_idx = jax.new_ref(jnp.asarray(0, dtype='uint32'))
        for idx, layer in enumerate(self.layers):
            w = self.w[idx]
            z = x
            z, _ = layer(fwd_layer, z, x, layer_idx, w, w_idx)
            
            w_idx[...] = jnp.asarray(0, dtype='uint32')
            x = z + self.b[idx]

        x = jax.checkpoint(self.norm)(x)
        if loss_fn is not None:
            return loss_fn(x, self.lm_head)
        else:
            logits = jax.checkpoint(jnp.dot)(x, self.lm_head[...])

        if cache is not None:
            cache.advance(logits.shape[1])
            
        return logits

class Exp_l0c8ybvm_rmsnorm(nn.Module):
    def __init__(self, config: ControlConfig, *, rngs: nn.Rngs):
        self.wte = ConceptronTokenEmbedding(config, rngs=rngs)
        k = 4
        assert config.num_layers % k == 0, 'num_layers should divisble by 4'
        num_layers_quater = config.num_layers // k
        self.layers = [nn.SeqStack([Decoder(config, rngs=rngs) for _ in range(num_layers_quater)]) for _ in range(k)]
        self.norm = ConceptronRMSNorm(config)
        self.lm_head = jax.new_ref(self.wte.embedding.value.T)
        self.rope = ConceptronRoPE(config.head_dims, config.rope_theta)
        self.inter_norm = ConceptronRMSNorm(config)

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
            z = jax.checkpoint(layer)(x, mask, position_embedding, cache, layer_idx[...]) + z
            layer_idx[...] += 1
            return z, None

        for layer in self.layers:
            z = x
            z, _ = layer(fwd_layer, z, x, layer_idx)
            z = jax.checkpoint(self.inter_norm)(z)
            x = z

        x = jax.checkpoint(self.norm)(x)
        logits = jax.checkpoint(jnp.dot)(x, self.lm_head[...])
        if cache is not None:
            cache.advance(logits.shape[1])
            
        return logits

class Exp_l0c8ybvm_rmsnorm_no_affine(nn.Module):
    def __init__(self, config: ControlConfig, *, rngs: nn.Rngs):
        self.wte = ConceptronTokenEmbedding(config, rngs=rngs)
        k = 4
        assert config.num_layers % k == 0, 'num_layers should divisble by 4'
        num_layers_quater = config.num_layers // k
        self.layers = [nn.SeqStack([Decoder(config, rngs=rngs) for _ in range(num_layers_quater)]) for _ in range(k)]
        self.norm = ConceptronRMSNorm(config)
        self.lm_head = jax.new_ref(self.wte.embedding.value.T)
        self.rope = ConceptronRoPE(config.head_dims, config.rope_theta)
        self.inter_norm = nn.RMSNorm(config.hidden_size, config.epsilon, dtype='float32', elementwise_affine=False)

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
            z = jax.checkpoint(layer)(x, mask, position_embedding, cache, layer_idx[...]) + z
            layer_idx[...] += 1
            return z, None

        for layer in self.layers:
            z = x
            z, _ = layer(fwd_layer, z, x, layer_idx)
            z = jax.checkpoint(self.inter_norm)(z)
            x = z

        x = jax.checkpoint(self.norm)(x)
        logits = jax.checkpoint(jnp.dot)(x, self.lm_head[...])
        if cache is not None:
            cache.advance(logits.shape[1])
            
        return logits


__all__ = [
    'Exp_l0c8ybvm_mean',
    'Exp_l0c8ybvm_rmsnorm',
    'Exp_l0c8ybvm_rmsnorm_no_affine',
    'Exp_l0c8ybvm_sum',
    'Exp_l0c8ybvm_weight_sum',
    'Exp_l0c8ybvm_weight_sum_bias',
]