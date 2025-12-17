from typing import Dict, List, Optional

from gigacheck.train.src.data.utils import Input, replace_repeated_symbols
from gigacheck.train.src.data.data_format import Labels, TextSample
from gigacheck.train.src.data.base_dataset import BaseDataset

from transformers import PreTrainedTokenizer


class EncodedDataset(BaseDataset):

    def __init__(
        self,
        texts: List[TextSample],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_sequence_length: int = None,
        min_sequence_length: int = None,
        random: bool = False,
        id2label: Dict[int, str] = {0: "ai", 1: "human"},
        seed: Optional[int] = None,
        is_eval: bool = False,
    ):
        super().__init__(
            texts,
            tokenizer,
            max_sequence_length,
            min_sequence_length,
            random,
            id2label,
            seed=seed,
            is_eval=is_eval,
        )

    def _preprocess_once(self, sample: TextSample) -> None:
        if not hasattr(sample, "modified"):
            setattr(sample, "modified", False)

        if sample.modified:
            return

        sample.text = replace_repeated_symbols(sample.text).strip()
        sample.modified = True

    def _augment_text(self, tokens: List[int], max_len: int):
        max_sequence_length = max_len - 2
        output_length = min(len(tokens), max_sequence_length)
        if self.min_sequence_length and self.random is not None and self.random.rand() < 0.5:
            output_length = self.random.randint(min(self.min_sequence_length, len(tokens)), output_length + 1)

        start_index = (
            0
            if self.random is None or len(tokens) <= output_length
            else self.random.randint(0, len(tokens) - output_length + 1)
        )
        end_index = start_index + output_length
        tokens = tokens[start_index:end_index]

        return tokens

    def _get_encoded_text_without_pad(self, sample: TextSample, max_len: Optional[int]) -> List[int]:
        tokens: List[int] = self.tokenizer.encode(sample.text, add_special_tokens=False)

        if self.is_eval or max_len is None:
            tokens = tokens[: self.tokenizer.max_len - 2]
        else:
            tokens = self._augment_text(tokens, max_len)

        return tokens

    def __getitem__(self, index: int) -> Input:

        sample = self.texts[index]
        label = self._get_label(Labels(sample.label))

        self._preprocess_once(sample)
        tokens = self._get_encoded_text_without_pad(sample, self.max_sequence_length)

        # NOTE: None during validation
        max_len = self.max_sequence_length - 2 if (self.max_sequence_length is not None and not self.is_eval) else None
        mask, tokens = self._pad_tokens(tokens, max_len)

        return Input(
            tokens=tokens,
            mask=mask,
            label=label,
            sample=sample,
        )
