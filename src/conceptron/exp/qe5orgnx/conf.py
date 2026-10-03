from dataclasses import dataclass

from conceptron._parts.conf import ConceptronTextConfig


@dataclass
class ControlConfig(ConceptronTextConfig):
    vocab_size: int = 49152
    hidden_size: int = 512
    inter_size: int = 1536
    num_heads: int = 12
    num_key_value_heads: int = 12
    head_dims: int = 64
    num_layers: int = 24
    dtype: str = 'bfloat16'


__all__ = ['ControlConfig']