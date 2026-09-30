from transformers import PreTrainedTokenizerFast


class PratchyaTokenizer(PreTrainedTokenizerFast):

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *init_inputs,
        **kwargs,
    ):
        kwargs.setdefault("bos_token", "<|BOS|>")
        kwargs.setdefault("eos_token", "<|EOS|>")
        kwargs.setdefault("pad_token", "<|PAD|>")

        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path,
            *init_inputs,
            **kwargs,
        )

        tokenizer.add_bos_token = True
        tokenizer.add_eos_token = False

        return tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self)