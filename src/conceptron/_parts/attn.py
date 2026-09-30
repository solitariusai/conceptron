import jax
import jax.numpy as jnp
from taktiny import nn
from taktiny.utils.typing import QuantConfig

from conceptron._parts.conf import ConceptronTextConfig
from conceptron._parts.fn import rotate_half
from conceptron._parts.utils import ConceptronCache


def product_attention(
    query: jax.Array, key: jax.Array, value: jax.Array, *,
    query_start: int | jax.Array = 0, key_valid_length: int | jax.Array | None = None
) -> jax.Array:
    """Causal grouped-query attention with an online softmax over keys.

    Inputs have shapes [batch, sequence, heads, head_dim]. Key and value
    may have a longer sequence and fewer heads than query. ``query_start``
    gives the absolute position of the first query within the key sequence.
    """
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError('query, key, and value must be rank-4; key and value must match')

    batch, length, num_heads, head_dim = query.shape
    if (key.shape[0] != batch or key.shape[-1] != head_dim
            or key.shape[2] < 1 or num_heads % key.shape[2]):
        raise ValueError(
            'query and key must share batch and head dimensions; '
            'query heads must divide by key heads'
        )

    num_key_value_heads = key.shape[2]
    dtype = jnp.result_type(query.dtype, key.dtype, value.dtype, jnp.float32)
    grouped_query = query.astype(dtype).reshape(
        batch, length, num_key_value_heads, num_heads // num_key_value_heads, head_dim
    )
    key = key.astype(dtype)
    value = value.astype(dtype)

    scores_shape = grouped_query.shape[:-1]
    running_max = jnp.full(scores_shape, jnp.finfo(dtype).min, dtype=dtype)
    running_sum = jnp.zeros(scores_shape, dtype=dtype)
    running_output = jnp.zeros(grouped_query.shape, dtype=dtype)
    query_positions = (query_start + jnp.arange(length))[None, :, None, None]

    def step(carry, inputs):
        maximum, normalizer, output = carry
        position, current_key, current_value = inputs
        score = jnp.einsum('btkgh,bkh->btkg', grouped_query, current_key)
        score = score * (head_dim ** -0.5)
        valid = query_positions >= position
        if key_valid_length is not None:
            valid = valid & (position < key_valid_length)
        next_maximum = jnp.maximum(
            maximum, jnp.where(valid, score, jnp.finfo(dtype).min)
        )
        old_scale = jnp.exp(maximum - next_maximum)
        safe_score = jnp.where(valid, score, next_maximum)
        new_scale = jnp.where(valid, jnp.exp(safe_score - next_maximum), 0)
        normalizer = normalizer * old_scale + new_scale
        output = output * old_scale[..., None] + (
            new_scale[..., None] * current_value[:, None, :, None, :]
        )
        return (next_maximum, normalizer, output), None

    (_, normalizer, output), _ = jax.lax.scan(
        step,
        (running_max, running_sum, running_output),
        (jnp.arange(key.shape[1]), jnp.moveaxis(key, 1, 0), jnp.moveaxis(value, 1, 0)),
    )
    output = output / jnp.where(normalizer[..., None] > 0, normalizer[..., None], 1)
    return output.reshape(batch, length, num_heads, head_dim).astype(query.dtype)

class ConceptronAttention(nn.Module):
    def __init__(self, config: ConceptronTextConfig, *, rngs: nn.Rngs, quant: QuantConfig = None):
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
            axis_names=('hidden', 'num_heads', 'head_dim'),
            dtype=config.dtype,
        )
        self.o_proj = nn.Linear(
            (config.num_heads, config.head_dims),
            config.hidden_size, 
            rngs=rngs,
            bias=False,
            quant=quant,
            axis_names=('num_heads', 'head_dim', 'hidden'),
            dtype=config.dtype,
        )

    def __call__(
        self, 
        x: jax.Array, 
        position_embedding: tuple[jax.Array, jax.Array] | None = None, 
        cache: ConceptronCache | None = None
    ):
        """Apply attention using optional rotary embeddings and a KV cache.

        ``position_embedding`` is a ``(cos, sin)`` pair for the current tokens.
        When supplied, ``cache`` holds past keys and values for the pinned layer.
        Returns the attention output and updated cache.
        """
        qkv = self.qkv_proj(x)
        q, k, v = jnp.split(
            qkv,
            (self.num_heads, self.num_heads + self.num_key_value_heads),
            axis=-2,
        )

        if position_embedding is not None:
            cos, sin = position_embedding
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
        key_valid_length = None
        if cache is not None:
            query_start = cache.position_idx
            cache.update(k, v)
            k, v = cache.get()
            key_valid_length = query_start + q.shape[1]

        attended = product_attention(
            q, k, v, query_start=query_start, key_valid_length=key_valid_length
        )
        return self.o_proj(attended), cache


__all__ = ['ConceptronAttention', 'product_attention']
