from typing import List
from loguru import logger
from functools import partial
import torch

import click
from transformers import AutoTokenizer, PretrainedConfig

from gigacheck.train.src.detection.arguments_parsing import TrainingArguments
from gigacheck.train.src.detection.interval_dataset import IntervalEncodedDataset
from gigacheck.train.src.detection.detr_trainer import DetrTrainer

from gigacheck.train.src.data.corpus import Corpus
from gigacheck.train.src.data.data_format import TextSample
from gigacheck.train.src.data.utils import Input

from gigacheck.model.mistral_ai_detector import MistralAIDetectorForSequenceClassification
from gigacheck.train.scripts.train_detr_model import compute_metrics, update_tokenizer


@click.command()
@click.option("--pretrained_model_path", type=str, required=True, help="Path to pretrained model dir")
@click.option("--eval_data_path", type=str, required=True, help="Path to .jsonl file.")
def main(pretrained_model_path: str, eval_data_path: str):

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    pretrain_conf = PretrainedConfig.from_pretrained(pretrained_model_path)
    update_tokenizer(tokenizer, pretrain_conf.max_length)
    logger.info(f"Model max_length: {pretrain_conf.max_length}")

    val_texts: List[TextSample] = Corpus(eval_data_path).data
    eval_dataset = IntervalEncodedDataset(
        val_texts,
        tokenizer,
        id2label=pretrain_conf.id2label,
        span_loss_type=pretrain_conf.detr_config["span_loss_type"],
        is_eval=True,
    )

    model = MistralAIDetectorForSequenceClassification.from_pretrained(
        pretrained_model_path,
        torch_dtype=torch.float32,
        num_labels=pretrain_conf.num_labels,
        max_length=pretrain_conf.max_length,
        with_detr=True,
        detr_config=pretrain_conf.detr_config,
        id2label=pretrain_conf.id2label,
    )

    extractor_dtype = getattr(torch, pretrain_conf.detr_config["extractor_dtype"])
    logger.info(f"Using dtype={extractor_dtype} for {type(model.model)}")
    if extractor_dtype == torch.bfloat16:
        model.model.to(torch.bfloat16)
        model.classification_head.to(torch.bfloat16)

    model_detr_dtype = next(model.detr.parameters()).dtype
    logger.info(f"model device: {model.device}; mistral dtype: {model.model.dtype}; detr dtype: {model_detr_dtype}")

    model.eval()

    logger.info(f"***** Running DETR validation *****")

    args = TrainingArguments(
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        do_eval=True,
        output_dir="validation",
        report_to="tensorboard",
    )

    trainer = DetrTrainer(
        model=model,
        args=args,
        eval_dataset=eval_dataset,
        data_collator=Input.collate_fn,
        custom_id2label=pretrain_conf.id2label,
        compute_metrics=partial(compute_metrics, eval_dataset=eval_dataset),
        tokenizer_to_save=tokenizer,
    )

    metrics = trainer.evaluate()
    logger.info(f"[data] {eval_data_path}")
    logger.info(f"[model] {pretrained_model_path}")
    for k, v in metrics.items():
        logger.info(f"{k}:\t{v}")


if __name__ == "__main__":
    main()
