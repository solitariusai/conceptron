import functools

import jax
import jax.numpy as jnp
from taktiny import nn
from taktiny.utils.ops import linear
from taktiny.utils.typing import QuantConfig

from conceptron._parts import (
    ConceptronCache,
    ConceptronRMSNorm,
    ConceptronRoPE,
    ConceptronTokenEmbedding,
    Decoder,
)
from conceptron.spy import SpyConfig


class SpyModel(nn.Module):
    def __init__(self, config: SpyConfig, *, rngs: nn.Rngs, quant: QuantConfig = None):
        self.wte = ConceptronTokenEmbedding(config, rngs=rngs)
        self.rope = ConceptronRoPE(config.head_dims)
        self.layers = nn.SeqStack([Decoder(config, rngs=rngs, quant=quant) for _ in range(config.num_layers)])
        self.norm = ConceptronRMSNorm(config)

    def __call__(self, ids: jax.Array, cache: ConceptronCache | None = None):
        if cache is not None:
            capacity = cache.key_cache.shape[2]
            if ids.shape[1] > capacity:
                raise ValueError('sequence is longer than the cache capacity')
            if not isinstance(cache.position_idx, jax._src.core.Tracer) and int(cache.position_idx) + ids.shape[1] > capacity:
                raise ValueError('sequence exceeds the remaining cache capacity')
        x = self.wte(ids)
        start = cache.position_idx if cache is not None else 0
        positions = start + jnp.arange(ids.shape[1])
        position_embedding = self.rope(positions, dtype=x.dtype)

        def layer_fwd(layer, carry, posemb):
            x, cache, layer_idx = carry
            if cache is not None:
                cache.pin(layer_idx)
            x, cache = layer(x, posemb, cache)
            if cache is not None:
                cache.remove_pin()
            return (x, cache, layer_idx + 1), None

        (x, cache, _), _ = self.layers(
            layer_fwd, (x, cache, jnp.asarray(0, dtype=jnp.int32)), position_embedding
        )
        if cache is not None:
            cache.position_idx = cache.position_idx + ids.shape[1]
        return self.norm(x), cache

class Spy(nn.Module):
    def __init__(self, config: SpyConfig, *, rngs: nn.Rngs, quant: QuantConfig = None):
        self.model = SpyModel(config, rngs=rngs, quant=quant)
        self.quant = quant

    def __call__(self, ids: jax.Array, cache: ConceptronCache | None = None):
        x, cache = self.model(ids, cache)
        logits = linear(
            x, self.model.wte.embedding.value.T,
            quant=self.quant
        )
        return logits, cache

    @classmethod
    def init(cls, config: SpyConfig | None = None, rngs: nn.Rngs | None = None, quant: QuantConfig = None, debug: bool = True):
        if config is None:
            config = SpyConfig()
        
        if rngs is None:
            rngs = nn.Rngs(0)

        init_fn = lambda: cls(config, rngs=rngs, quant=quant)
        if debug:
            init_fn = functools.partial(jax.eval_shape, fun=init_fn)

        return init_fn()
    

__all__ = ['Spy', 'SpyModel']
