import jax
import jax.numpy as jnp
from taktiny.nn import Pytree

from conceptron._parts.conf import ConceptronTextConfig


class ConceptronCache(Pytree):
    def __init__(self, config: ConceptronTextConfig, num_batches: int, max_sequences: int = 1024, dtype: str | None = None):
        num_key_value_heads = config.num_key_value_heads
        head_dims = config.head_dims
        num_layers = config.num_layers
        if dtype is None:
            self.dtype = config.dtype

        self.key_cache = jax.new_ref(jnp.zeros((num_batches, num_layers, max_sequences, num_key_value_heads, head_dims), dtype=self.dtype))
        self.value_cache = jax.new_ref(jnp.zeros((num_batches, num_layers, max_sequences, num_key_value_heads, head_dims), dtype=self.dtype))
        self.position_idx = jax.new_ref(jnp.asarray(0, dtype=jnp.int32))
        self.cache_length = max_sequences

    def get(self, layer_idx: int | jax.Array):
        return self.key_cache[:, layer_idx], self.value_cache[:, layer_idx]

    def update(self, key: jax.Array, value: jax.Array, layer_idx: jax.Array | int):
        if key.shape != value.shape or key.ndim != 4:
            raise ValueError('key and value must have matching [batch, sequence, heads, dim] shapes')
        if key.shape[0] != self.key_cache.shape[0] or key.shape[2:] != self.key_cache.shape[3:]:
            raise ValueError('key and value shapes do not match the cache')
        if key.shape[1] > self.key_cache.shape[2]:
            raise ValueError('sequence is longer than the cache capacity')

        length = key.shape[1]
        idx = self.position_idx[...]
        self.key_cache[:, layer_idx, jax.ds(idx, length)] = key.astype(self.dtype)
        self.value_cache[:, layer_idx, jax.ds(idx, length)] = value.astype(self.dtype)
        
    def advance(self, length):
        self.position_idx[...] += length

class AxisNames:
    Embedding = ('vocab', 'hidden')
    RMSNorm = ('hidden',)


__all__ = ['AxisNames', 'ConceptronCache']
