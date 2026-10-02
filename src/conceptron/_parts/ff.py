import jax
import jax.numpy as jnp
from taktiny import nn
from taktiny.utils.typing import QuantConfig

from conceptron._parts.conf import ConceptronTextConfig
from conceptron._parts.utils import AxisNames


class ConceptronMLP(nn.Module):
    def __init__(self, config: ConceptronTextConfig, *, rngs: nn.Rngs, quant: QuantConfig = None):
        self.w1 = nn.Linear(
            config.hidden_size, 
            config.inter_size * 2,
            rngs=rngs,
            bias=False,
            quant=quant,
            axis_names=AxisNames.MLPW1,
            dtype=config.dtype,
        )
        self.w2 = nn.Linear(
            config.inter_size,
            config.hidden_size, 
            rngs=rngs,
            bias=False,
            quant=quant,
            axis_names=AxisNames.MLPW2,
            dtype=config.dtype,
        )

    def __call__(self, x: jax.Array):
        x = self.w1(x)
        x1, x2 = jnp.split(x, 2, axis=-1)
        return self.w2(jax.nn.silu(x1) * x2)


__all__ = ['ConceptronMLP']