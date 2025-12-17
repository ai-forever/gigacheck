from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from transformers import PreTrainedTokenizer
from transformers.models.llama.tokenization_llama import SPIECE_UNDERLINE

from gigacheck.train.src.data.data_format import Labels, TextSample
from gigacheck.train.src.data.base_dataset import BaseDataset
from gigacheck.train.src.data.utils import Input, Meta
from gigacheck.model.src.interval_detector.span_utils import span_xx_to_cxw


class IntervalEncodedDataset(BaseDataset):
    def __init__(
        self,
        texts: List[TextSample],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_sequence_length: int = None,
        min_sequence_length: int = None,
        random: bool = False,
        id2label: Dict[int, str] = {0: "ai", 1: "human", 2: "mixed"},
        seed: int = None,
        span_loss_type="l1",
        is_eval: bool = False,
        max_tokens_to_augment: int = 10000,
    ):
        super().__init__(
            texts,
            tokenizer,
            max_sequence_length,
            min_sequence_length,
            seed=seed,
            random=random,
            id2label=id2label,
            is_eval=is_eval,
        )

        self.span_loss_type = span_loss_type
        self.max_tokens_to_augment = max_tokens_to_augment

    def _get_encoded_text_without_pad(
        self,
        tokens: List[int],
        sample: TextSample,
        max_len: Optional[int],
        augment=True,
        start_token: int = 0,
    ) -> Tuple[List[int], List[list], int, int]:
        """
        :param sample: The TextSample object containing the text to be encoded.
        :param max_len: The maximum length of the encoded text in tokens.
                        If None (during validation), self.tokenizer.max_len to be used.
        :param augment: Determines whether data augmentation should be applied to the text.
        :param start_token: the token number from which to start examining the text

        :return: a tuple containing:
                 - the sliced encoded tokens
                 - AI intervals
                 - the length of the text (in chars or tokens)
                 - number of all gt intervals for full text
        """

        ai_char_intervals = sample.ai_char_intervals
        if Labels(sample.label) is Labels.AI:
            ai_char_intervals = [[0, len(sample.text)]]

        should_augment = len(tokens) > self.min_sequence_length if self.min_sequence_length is not None else True
        if max_len is None or not augment or not should_augment:
            # get ai intervals for the cropped text
            ai_intervals, text_len = get_intervals(
                tokens,
                ai_char_intervals,
                start_token=start_token,
                stop_token=start_token + self.tokenizer.max_len - 2,
                tokenizer=self.tokenizer,
            )
            tokens_slice = tokens[start_token : start_token + self.tokenizer.max_len - 2]
        else:
            assert start_token == 0, "Augmentations are done for all tokens together"

            len_tokens = len(tokens)
            if self.max_tokens_to_augment is not None:
                # this number(max_tokens_to_augment) of first tokens will be considered for augmentations
                # this restriction was done because of very slow augmentation of long sequences
                len_tokens = min(len_tokens, self.max_tokens_to_augment)

            max_sequence_length = max_len - 2
            output_length = min(len_tokens, max_sequence_length)
            aug_prob = 0.5 if Labels(sample.label) is not Labels.MIXED else 0.2
            if self.min_sequence_length and self.random is not None and self.random.rand() < aug_prob:
                output_length = self.random.randint(min(self.min_sequence_length, len_tokens), output_length + 1)

            start_index = (
                0
                if self.random is None or len_tokens <= output_length
                else self.random.randint(0, len_tokens - output_length + 1)
            )
            end_index = start_index + output_length

            # get ai intervals for augmented part of the text
            ai_intervals, text_len = get_intervals(
                tokens,
                ai_char_intervals,
                start_token=start_index,
                stop_token=end_index,
                tokenizer=self.tokenizer,
            )
            tokens_slice = tokens[start_index:end_index]

        n_orig_intervals = len(ai_char_intervals) if ai_char_intervals is not None else 0
        return tokens_slice, ai_intervals, text_len, n_orig_intervals

    def _get_span_labels(self, windows_list: Union[List[list], np.ndarray], text_len: int) -> torch.Tensor:
        """
        :param windows_list: list([st, ed]) in chars. E.g. [[26, 36]]
        :param text_len: length of text in chars
        :return: a Tensor of shape (#windows, 2), each row is [center, width] normalized by text length (in tokens)
        """

        if len(windows_list) == 0:
            return torch.empty((0, 2), dtype=torch.float32)

        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows_list) / text_len  # normalized windows in xx
            assert (windows[:, 1] >= windows[:, 0]).all()
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        else:
            raise NotImplementedError
        return windows

    def _get_input(self, all_tokens: List[int], sample: TextSample, index: Optional[int]) -> Input:
        """
        Prepare sample for model input.
        Args:
            all_tokens: encoded text into inout_ids
            sample: info about text sample
            index: index of element in dataset

        Returns: Input
        """
        tokens, ai_intervals, text_len, n_all_intervals = self._get_encoded_text_without_pad(
            all_tokens,
            sample,
            self.max_sequence_length,
            augment=True if self.random is not None else False,
        )

        label = self._get_label(Labels(sample.label))
        if len(ai_intervals) == 0 and Labels(sample.label) is Labels.MIXED:
            label = self._get_label(Labels.HUMAN)

        # NOTE: None during validation
        max_len = self.max_sequence_length - 2 if (self.max_sequence_length is not None and not self.is_eval) else None
        tokens_len = len(tokens)

        # (torch.Tensor, torch.Tensor)
        mask, tokens = self._pad_tokens(tokens, max_len)

        # xx -> normalized cxw
        span_labels: torch.Tensor = self._get_span_labels(ai_intervals, text_len)

        return Input(
            tokens=tokens,
            mask=mask,
            label=label,
            sample=sample,
            span_labels=span_labels,
            meta=Meta(len=text_len, index=index, tokens_len=tokens_len),
        )

    def __getitem__(self, index: int) -> Input:
        sample = self.texts[index]

        all_tokens: List[int] = self.tokenizer.encode(sample.text, add_special_tokens=False)
        input_data: Input = self._get_input(all_tokens, sample, index)
        return input_data


def get_intervals(
    input_ids,
    ai_chars_intervals,
    start_token,
    stop_token,
    tokenizer,
) -> Tuple[List[list], int]:
    """
    :param input_ids: The input IDs.
    :param ai_chars_intervals: The AI intervals in chars.
    :param start_token: The start token index.
    :param stop_token: The stop token index.
    :param tokenizer: The tokenizer object.

    :return: The intervals and the length of input (in tokens or chars).
    """

    if ai_chars_intervals is None:
        ai_chars_intervals = []

    tokens: List[str] = tokenizer.convert_ids_to_tokens(input_ids)
    startCh = SPIECE_UNDERLINE

    now = 0
    cur_interval = None
    cur_interval_start_token, cur_interval_end_token = 0, 0
    tokens_intervals = []

    char_begin, char_end = None, None
    while True:
        if now >= len(tokens):
            break
        tail = now
        if tokens[now][0] == startCh:
            while tail < len(tokens) - 1:
                if tokens[tail + 1][0].isalpha() and tokens[tail + 1][0] != startCh:
                    tail += 1
                else:
                    break

        cur_text = tokenizer.convert_tokens_to_string(tokens[0 : tail + 1])

        if cur_interval is not None:
            st, end = ai_chars_intervals[cur_interval]
            if end == len(cur_text):
                # stop interval ind=cur_interval
                cur_interval_end_token = tail + 1
                tokens_intervals.append((cur_interval_start_token, cur_interval_end_token))
                cur_interval = None

        if cur_interval is None:
            for interval_ind, (st, end) in enumerate(ai_chars_intervals):
                if (st == len(cur_text) or st - 1 == len(cur_text)) or (st == 0 and now == 0):
                    cur_interval = interval_ind
                    # start interval ind=cur_interval
                    cur_interval_start_token = tail + 1
                    break

        if now <= start_token <= tail:
            char_begin = len(tokenizer.convert_tokens_to_string(tokens[0:start_token]))

        if now <= stop_token <= tail:
            if cur_interval is not None:
                cur_interval_end_token = stop_token
                tokens_intervals.append((cur_interval_start_token, cur_interval_end_token))
            char_end = len(tokenizer.convert_tokens_to_string(tokens[0:stop_token]))
            break

        now = tail + 1

    if char_end is None:
        char_end = len(tokenizer.convert_tokens_to_string(tokens[0:now]))

    if cur_interval is not None:
        tokens_intervals.append((cur_interval_start_token, now))

    text_chars_len = char_end - char_begin

    final_char_intervals = []
    final_token_intervals = []
    for char_interval, token_intereval in zip(ai_chars_intervals, tokens_intervals):
        interval_token_st, interval_token_end = token_intereval
        char_st, char_end = char_interval

        if start_token >= interval_token_end:
            continue

        if interval_token_st >= interval_token_end:
            continue

        if interval_token_st <= start_token <= interval_token_end:
            char_st = len(tokenizer.convert_tokens_to_string(tokens[0:start_token]))
        if interval_token_st <= stop_token <= interval_token_end:
            char_end = len(tokenizer.convert_tokens_to_string(tokens[0:stop_token]))

        char_st, char_end = char_st - char_begin, min(char_end - char_begin, text_chars_len)
        if char_end < char_st or abs(char_st - char_end) < 2:
            continue
        final_char_intervals.append([char_st, char_end])
        final_token_intervals.append([interval_token_st - start_token, interval_token_end - start_token])

    return final_char_intervals, text_chars_len
