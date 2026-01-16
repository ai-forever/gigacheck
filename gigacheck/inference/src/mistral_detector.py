from typing import Any, Dict, Tuple, List
import numpy as np

import torch
from loguru import logger
from intervaltree import Interval, IntervalTree

from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
)

from gigacheck.model.mistral_ai_detector import MistralAIDetectorForSequenceClassification
from gigacheck.model.src.interval_detector.span_utils import span_cxw_to_xx


class MistralDetector:
    def __init__(
        self,
        max_seq_len: int,
        with_detr: bool,
        id2label: Dict[int, str],
        device: str = "cuda:0",
        verbose: bool = True,
        debug: bool = False,
        conf_interval_thresh: float = 0.8,
    ):
        self.verbose = verbose
        self.debug = debug
        self.device = device
        self.with_detr = with_detr
        self._max_len = max_seq_len
        self.conf_interval_thresh = conf_interval_thresh

        self.tokenizer: PreTrainedTokenizer = None
        self.model: MistralAIDetectorForSequenceClassification = None
        self.id2label: Dict[int, str] = id2label
        self.trained_classification_head = True

        if self.with_detr:
            assert len(self.id2label) == 3

    @property
    def max_len(self) -> int:
        return self._max_len

    def _from_pretrained_multitask(self, base_model_path, device_map, **kwargs):
        if self.with_detr:
            model = self._from_pretrained_detr(base_model_path, device_map, **kwargs)
        else:
            model = self._from_pretrained_classifier(base_model_path, device_map, **kwargs)

        return model

    def _from_pretrained_detr(self, base_model_path, device_map, **kwargs):
        pretrain_conf = PretrainedConfig.from_pretrained(base_model_path)
        detr_config = pretrain_conf.detr_config
        num_labels = pretrain_conf.num_labels

        if pretrain_conf.to_dict().get("trained_classification_head", True) is False:
            self.trained_classification_head = False

        additional_kwargs = {
            "num_labels": num_labels,
            "max_length": self._max_len,
            "with_detr": self.with_detr,
            "detr_config": detr_config,
            "device_map": device_map,
            "torch_dtype": torch.float32,
        }
        additional_kwargs.update(kwargs)

        model = MistralAIDetectorForSequenceClassification.from_pretrained(base_model_path, **additional_kwargs)

        extractor_dtype = getattr(torch, pretrain_conf.detr_config["extractor_dtype"])
        logger.info(f"Using dtype={extractor_dtype} for {type(model.model)}")
        if extractor_dtype == torch.bfloat16:
            model.model.to(torch.bfloat16)
            model.classification_head.to(torch.bfloat16)

        return model

    def _from_pretrained_classifier(self, base_model_path, device_map, **kwargs):
        pretrain_conf = PretrainedConfig.from_pretrained(base_model_path)
        num_labels = pretrain_conf.num_labels

        assert num_labels, "Number of labels must be not 0."

        additional_kwargs = {
            "num_labels": num_labels,
            "max_length": self._max_len,
            "with_detr": self.with_detr,
            "device_map": device_map,
            "torch_dtype": "auto",
        }
        additional_kwargs.update(kwargs)
        model = MistralAIDetectorForSequenceClassification.from_pretrained(base_model_path, **additional_kwargs)

        return model

    def from_pretrained(self, base_model_path, model: MistralAIDetectorForSequenceClassification = None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        if model is None:
            device_map = self.device
            self.model = self._from_pretrained_multitask(base_model_path, device_map, **kwargs)

            logger.info(f"{self.model.dtype=} max_len: {self._max_len}")
        else:
            self.model = model

        self.model.eval()

        self.model.config.max_length = self._max_len
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.unk_token_id = self.tokenizer.unk_token_id

        return self

    def _get_tokens(self, text: str) -> Tuple[torch.tensor, torch.tensor, int, int]:
        assert self._max_len is not None and self.tokenizer is not None, "Model must be initialized"

        tokens = self.tokenizer(text, add_special_tokens=False).input_ids
        n_tokens = len(tokens)

        if len(tokens) > self._max_len - 2:
            tokens = tokens[: self._max_len - 2]
            cur_text = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(tokens))
            text_len = len(cur_text)
        else:
            text_len = len(text)

        tokens = torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id])
        used_tokens = len(tokens)

        tokens = tokens.unsqueeze(0)
        mask = torch.ones_like(tokens)

        if n_tokens > self._max_len - 2:
            logger.debug(f"Use only {used_tokens} tokens from {n_tokens} ({len(text.split(' '))} all words)")

        logger.debug(f"Use tokens: {used_tokens}. Max tokens: {self._max_len}")

        return tokens.to(self.device), mask.to(self.device), text_len, n_tokens

    def _get_ai_intervals(self, detr_out: Dict[str, torch.Tensor], text_lens: List[int]) -> List[np.ndarray]:
        pred_spans = detr_out["pred_spans"].detach()
        src_logits = detr_out["pred_logits"].detach()  # (batch_size, #queries, #classes=2)
        assert len(text_lens) == pred_spans.shape[0]

        # take probs for foreground objects only (ind = 0)
        pred_probs = torch.softmax(src_logits, -1)[:, :, 0:1]
        spans = torch.stack([to_absolete(pred_spans[i], text_lens[i]) for i in range(len(text_lens))], dim=0)

        # concat spans and scores
        preds = torch.cat([spans, pred_probs], dim=2)  # xx, prob
        final_preds = [text_preds[text_preds[:, 2] > self.conf_interval_thresh].cpu().numpy() for text_preds in preds]

        return final_preds

    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, Any]:
        assert self.model is not None, "Model must be initialized, call from_pretrained() method."

        tokens, mask, text_len, n_tokens = self._get_tokens(text)

        output = self.model(
            tokens,
            attention_mask=mask,
            return_dict=True,
            return_detr_output=self.with_detr,
        )
        if self.debug:
            logger.info(f"Raw output: {output.logits}; id2label: {self.id2label} ")

        if not self.with_detr:
            logits = output.logits[0]
            probs = logits.to(torch.float32).softmax(dim=-1)
            probs = probs.detach().cpu().numpy()
            cls_ind = int(np.argmax(probs))
            pred_label = self.id2label[cls_ind]

            return {"pred_label": pred_label, "classification_head_probs": probs}
        else:
            # ai / human / mixed classification
            main_logits, detr_out = output.logits
            ai_intervals: List[np.ndarray] = self._get_ai_intervals(detr_out, [text_len])
            ai_intervals = ai_intervals[0]

            if self.trained_classification_head:
                main_probs = main_logits.to(torch.float32).softmax(dim=-1)
                main_probs = main_probs.detach().cpu().numpy()
                main_probs = main_probs[0]
                cls_ind = int(np.argmax(main_probs))
                pred_label = self.id2label[cls_ind]
                return {"pred_label": pred_label,
                        "classification_head_probs": main_probs,
                        "ai_intervals": ai_intervals}
            else:
                return {"ai_intervals": ai_intervals}


def to_absolete(pred_spans: torch.Tensor, text_len: int) -> torch.Tensor:
    spans = span_cxw_to_xx(pred_spans) * text_len
    return torch.clamp(spans, 0, text_len)


def merge_intervals(ai_intervals: np.ndarray) -> List[Tuple[int, int]]:
    intervals = IntervalTree(Interval(begin, end, prob) for (begin, end, prob) in ai_intervals)
    intervals.merge_overlaps(strict=False)
    ai_intervals = [(int(i.begin), int(i.end)) for i in sorted(intervals)]
    return ai_intervals
