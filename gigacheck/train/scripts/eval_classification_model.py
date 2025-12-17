from typing import List
from loguru import logger
from functools import partial

import click
import torch
from transformers import AutoTokenizer, PretrainedConfig, TrainingArguments

from gigacheck.train.src.classification.classification_dataset import EncodedDataset
from gigacheck.train.src.classification.custom_trainer import CustomTrainer

from gigacheck.train.src.data.corpus import Corpus
from gigacheck.train.src.data.data_format import TextSample
from gigacheck.train.src.data.utils import Input

from gigacheck.model.mistral_ai_detector import MistralAIDetectorForSequenceClassification
from gigacheck.train.scripts.train_classification_model import compute_metrics, update_tokenizer


@click.command()
@click.option("--pretrained_model_path", type=str, required=True, help="Path to pretrained model dir")
@click.option("--eval_data_path", type=str, required=True, help="Path to .jsonl file.")
def main(pretrained_model_path: str, eval_data_path: str):

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    pretrain_conf = PretrainedConfig.from_pretrained(pretrained_model_path)
    update_tokenizer(tokenizer, pretrain_conf.max_length)

    val_texts: List[TextSample] = Corpus(eval_data_path).data
    eval_dataset = EncodedDataset(
        val_texts,
        tokenizer,
        id2label=pretrain_conf.id2label,
        is_eval=True,
    )

    model = MistralAIDetectorForSequenceClassification.from_pretrained(
        pretrained_model_path,
        torch_dtype=torch.bfloat16,
        num_labels=pretrain_conf.num_labels,
        max_length=pretrain_conf.max_length,
    )
    model.eval()

    logger.info(f"***** Running validation *****")

    args = TrainingArguments(
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        do_eval=True,
        output_dir="validation",
        bf16=True,
    )

    trainer = CustomTrainer(
        model=model,
        args=args,
        eval_dataset=eval_dataset,
        data_collator=Input.collate_fn,
        custom_id2label=pretrain_conf.id2label,
        compute_metrics=partial(compute_metrics, id2label=pretrain_conf.id2label),
    )

    metrics = trainer.evaluate()
    logger.info(f"[data] {eval_data_path}")
    logger.info(f"[model] {pretrained_model_path}")
    for k, v in metrics.items():
        logger.info(f"{k}:\t{v}")


if __name__ == "__main__":
    main()
