from taktiny import nn

from conceptron._parts.conf import ConceptronTextConfig
from conceptron._parts.utils import AxisNames


class ConceptronTokenEmbedding(nn.Embedding):
    def __init__(self, config: ConceptronTextConfig, *, rngs: nn.Rngs):
        super().__init__(
            config.vocab_size,
            config.hidden_size,
            dtype=config.dtype,
            rngs=rngs,
            axis_names=AxisNames.Embedding
        )


__all__ = ['ConceptronTokenEmbedding']