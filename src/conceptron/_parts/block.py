import jax
from taktiny import nn
from taktiny.utils.typing import QuantConfig

from conceptron._parts.attn import ConceptronAttention
from conceptron._parts.conf import ConceptronTextConfig
from conceptron._parts.ff import ConceptronMLP
from conceptron._parts.norm import ConceptronRMSNorm
from conceptron._parts.utils import ConceptronCache


class Decoder(nn.Module):
    def __init__(self, config: ConceptronTextConfig, *, rngs: nn.Rngs, quant: QuantConfig):
        self.norm1 = ConceptronRMSNorm(config)
        self.norm2 = ConceptronRMSNorm(config)
        self.attn = ConceptronAttention(config, rngs=rngs, quant=quant)
        self.mlp = ConceptronMLP(config, rngs=rngs, quant=quant)

    def __call__(
        self, 
        x: jax.Array, 
        position_embedding: tuple[jax.Array, jax.Array] | None, 
        cache: ConceptronCache | None = None
    ) -> tuple[jax.Array, ConceptronCache]:
        res = x
        x, cache = self.attn(self.norm1(x), position_embedding, cache)
        x = x + res

        x = self.mlp(self.norm2(x)) + x
        return x, cache


__all__ = ['Decoder']