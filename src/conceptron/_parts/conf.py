from dataclasses import dataclass


@dataclass
class ConceptronTextConfig:
    vocab_size: int = 102016
    hidden_size: int = 512
    inter_size: int = 3072
    num_heads: int = 12
    num_key_value_heads: int = 4
    head_dims: int = 128
    num_layers: int = 22
    epsilon: float = 1e-7
    rope_theta: float = 10_000
    dtype: str = 'bfloat16'


__all__ = ['ConceptronTextConfig']