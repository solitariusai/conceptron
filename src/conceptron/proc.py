from pathlib import Path

from transformers import AutoTokenizer

current_dir = Path(__file__).resolve().parent

class _Tokenizer:
    def __init__(self):
        path = "_tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(current_dir / path, trust_remote_code=True)

    def __call__(self) -> AutoTokenizer:
        return self.tokenizer

class _TokenizerExp:
    def __init__(self):
        path = "_smol_tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(current_dir / path, trust_remote_code=True)

    def __call__(self) -> AutoTokenizer:
        return self.tokenizer

Tokenizer = _Tokenizer()
TokenizerExp = _TokenizerExp()

__all__ = ['Tokenizer', 'TokenizerExp']