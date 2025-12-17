import re
from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from gigacheck.train.src.data.data_format import TextSample


@dataclass
class Meta:
    len: Optional[int] = None  # length of the processing text (in chars or tokens), for ai intervals normalization
    index: Optional[int] = None  # index of the text in the dataset
    tokens_len: Optional[int] = None  # number of tokens which will be processed by the model


@dataclass
class Input:
    tokens: Union[torch.Tensor, str, List[int]]
    mask: Optional[torch.Tensor]
    label: int
    sample: TextSample
    # gt for detr: Tensor of shape (#windows, 2), each row is [center, width] normalized by text length (meta.len)
    span_labels: Optional[torch.Tensor] = None
    # additional info about the text for detr
    meta: Optional[Meta] = None

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    @staticmethod
    def collate_fn(batch: List["Input"]) -> dict:
        tokens, masks, label = [], [], []
        metas, samples = [], []
        span_labels = []

        for i, text in enumerate(batch):
            tokens.append(text.tokens)
            masks.append(text.mask)
            label.append(text.label)
            samples.append(text.sample)

            # for detr
            span_labels.append(text.span_labels)
            metas.append(text.meta)

        collated_batch = {
            "tokens": torch.stack(tokens, 0) if not isinstance(tokens[0], str) else list(tokens),
            "masks": torch.stack(masks, 0) if masks[0] is not None else None,
            "labels": torch.tensor(label),
            "samples": list(samples),
        }
        if all(meta is not None for meta in metas):
            collated_batch["metas"] = metas
        if all(spans is not None for spans in span_labels):
            collated_batch["span_labels"] = [dict(spans=e) for e in span_labels]

        return collated_batch


def replace_repeated_symbols(text: str) -> str:
    """
    Replace repeated '\n', ' ' and '-' symbols.
    """
    text = re.sub(r"(\r\n)+", r"\n", text)
    text = re.sub(r"(\r )+", r" ", text)
    text = re.sub(r"(\n)+", r"\n", text)
    text = re.sub(r"( )+", r" ", text)
    text = re.sub(r"(-)+", r"-", text)

    return text
