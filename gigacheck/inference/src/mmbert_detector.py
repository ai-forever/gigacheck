import torch
from loguru import logger
from intervaltree import Interval, IntervalTree

from transformers import PretrainedConfig, ModernBertForSequenceClassification

from gigacheck.model.mmbert_ai_detector import ModernBertAIDetectorForSequenceClassification
from gigacheck.inference.src.ai_detector import AIDetector


class ModernBertDetector(AIDetector):

    def _from_pretrained_detr(self, base_model_path, device_map):
        pretrain_conf = PretrainedConfig.from_pretrained(base_model_path)
        detr_config = pretrain_conf.detr_config
        num_labels = pretrain_conf.num_labels

        if pretrain_conf.to_dict().get("trained_classification_head", True) is False:
            self.trained_classification_head = False

        kwargs = {
            "num_labels": num_labels,
            "max_sequence_length": self._max_len,
            "with_detr": self.with_detr,
            "detr_config": detr_config,
        }

        model = ModernBertAIDetectorForSequenceClassification.from_pretrained(
            base_model_path,
            device_map=device_map,
            torch_dtype=torch.float32,
            **kwargs,
            key_mapping={"decoder.weight": "model.embeddings.tok_embeddings.weight"},
        )

        extractor_dtype = getattr(torch, pretrain_conf.detr_config["extractor_dtype"])
        logger.info(f"Using dtype={extractor_dtype} for {type(model.model)}")
        if extractor_dtype == torch.bfloat16:
            model.model.to(torch.bfloat16)
            if model.classification_head is not None:
                model.classification_head.to(torch.bfloat16)

        logger.info(f"DETR dtype: {model.detr.dtype=}")

        return model

    def _from_pretrained_classifier(self, base_model_path, device_map):
        pretrain_conf = PretrainedConfig.from_pretrained(base_model_path)
        num_labels = pretrain_conf.num_labels

        assert num_labels, "Number of labels must be not 0."
        model = ModernBertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=base_model_path,
            device_map=device_map,
            torch_dtype="auto",
            config=pretrain_conf,
            key_mapping={"decoder.weight": "model.embeddings.tok_embeddings.weight"},
        )

        return model
