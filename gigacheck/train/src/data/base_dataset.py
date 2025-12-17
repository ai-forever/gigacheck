from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from gigacheck.train.src.data.data_format import Labels, TextSample
from gigacheck.train.src.data.utils import Input


class BaseDataset(Dataset):
    def __init__(
        self,
        texts: List[TextSample],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_sequence_length: int = None,
        min_sequence_length: int = None,
        random: bool = False,
        id2label: Dict[int, str] = {0: "ai", 1: "human", 2: "mixed"},
        seed: int = None,
        is_eval: bool = False,
    ):
        self.texts: List[TextSample] = texts

        assert len(self.texts), "Empty dataset"
        self.tokenizer = tokenizer

        # Used for FalconMamba tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

        # length of input including <bos> and <eos> (does not include instruction + answer in case of EncodedLMDataset)
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.random = np.random.RandomState(seed) if random else None
        self.id2label = id2label
        self.is_eval = is_eval

    def __len__(self):
        return len(self.texts)

    def _pad_tokens(self, tokens: List[int], max_sequence_length: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        # max_sequence_length here is expected len without <bos>, <eos>
        if max_sequence_length is None or len(tokens) == max_sequence_length:
            mask = torch.ones(len(tokens) + 2)
            out_tokens = torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id])
        else:
            padding = [self.tokenizer.pad_token_id] * (max_sequence_length - len(tokens))
            out_tokens = torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id] + padding)
            mask = torch.ones(out_tokens.shape[0])
            if len(padding):
                mask[-len(padding) :] = 0
        return mask, out_tokens

    def _get_label(self, label: Labels) -> int:
        # Return index of the label based on self.id2label (e.g. { 0: "ai", 1: "human"} )
        for key, value in self.id2label.items():
            if label is Labels(value):
                return key
        raise ValueError(f"Unknown label={label}. self.id2label={self.id2label}")

    def __getitem__(self, index: int) -> Input:
        raise NotImplementedError
