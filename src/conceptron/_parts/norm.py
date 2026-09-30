from taktiny import nn

from conceptron._parts.conf import ConceptronTextConfig
from conceptron._parts.utils import AxisNames


class ConceptronRMSNorm(nn.RMSNorm):
    def __init__(self, config: ConceptronTextConfig):
        super().__init__(
            config.hidden_size,
            config.epsilon,
            dtype='float32',
            axis_names=AxisNames.RMSNorm
        )


__all__ = ['ConceptronRMSNorm']