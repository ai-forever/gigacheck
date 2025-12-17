import pathlib
from typing import Dict, List, Tuple
from loguru import logger
from functools import partial

import torch
import transformers
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.trainer_utils import EvalPrediction

from peft import TaskType, LoraConfig, get_peft_model
from torchmetrics.classification import MulticlassAccuracy

from gigacheck.train.src.classification.arguments_parsing import DataArguments, ModelArguments, TrainingArguments
from gigacheck.train.src.classification.classification_dataset import EncodedDataset
from gigacheck.train.src.classification.custom_trainer import CustomTrainer

from gigacheck.train.src.data.corpus import Corpus
from gigacheck.train.src.data.data_format import TextSample
from gigacheck.train.src.data.utils import Input

from gigacheck.model.mistral_ai_detector import MistralAIDetectorForSequenceClassification
from gigacheck.train.src.utils import save_config
from gigacheck.model.src.model_load_utils import custom_prepare_model_for_training


def compute_metrics(eval_pred: EvalPrediction, id2label: Dict) -> Dict[str, float]:

    all_pred_logits = torch.tensor(eval_pred.predictions)
    all_labels = torch.tensor(eval_pred.label_ids)

    metric = MulticlassAccuracy(num_classes=len(id2label), average=None)
    preds = torch.argmax(all_pred_logits, dim=-1)
    assert len(all_labels) == len(preds)

    accuracy_by_class = metric(preds, all_labels)
    mean_accuracy = torch.mean(accuracy_by_class)

    metrics = {"eval/mean_cls_accuracy": mean_accuracy}
    for class_id, class_acc in enumerate(accuracy_by_class):
        metrics[f"eval/{id2label[class_id]}_accuracy"] = class_acc
    return metrics


def load_datasets(
    data_args: DataArguments,
    tokenizer: transformers.PreTrainedTokenizer,
    id2label: Dict[int, str],
    random_seed: int = 0,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:

    train_texts: List[TextSample] = Corpus(data_args.train_data_path).data
    val_texts: List[TextSample] = Corpus(data_args.eval_data_path).data

    train_dataset = EncodedDataset(
        train_texts,
        tokenizer,
        data_args.max_sequence_length,
        data_args.min_sequence_length,
        random=data_args.random_sequence_length,
        id2label=id2label,
        seed=random_seed,
    )

    eval_dataset = EncodedDataset(
        val_texts,
        tokenizer,
        id2label=id2label,
        is_eval=True,
    )
    return train_dataset, eval_dataset


def update_tokenizer(tokenizer, max_sequence_length):
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.max_len = max_sequence_length
    tokenizer.padding_side = "right"


def load_model(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
):
    attn_implementation = training_args.attn_implementation
    compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float32

    kwargs = {"token": model_args.hf_token} if model_args.hf_token else {}
    model = MistralAIDetectorForSequenceClassification.from_pretrained(  # type: ignore
        pretrained_model_name_or_path=model_args.pretrained_model_name,
        attn_implementation=attn_implementation,
        torch_dtype=compute_dtype,
        num_labels=len(model_args.id2label),
        id2label=model_args.id2label,
        ce_weights=training_args.ce_weights,
        **kwargs,
    )

    # update model config
    model.config.max_length = tokenizer.max_len
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.unk_token_id = tokenizer.unk_token_id

    output_embedding_names = ["classification_head.dense", "classification_head.out_proj"]
    model = custom_prepare_model_for_training(model, output_embedding_names)

    if training_args.lora_enable:
        kwargs = {
            "target_modules": training_args.lora_target_modules,
            "modules_to_save": ["classification_head",],
            "task_type": TaskType.SEQ_CLS,
            "r": training_args.lora_r,
            "bias": training_args.lora_bias,
            "lora_alpha": training_args.lora_alpha,
            "lora_dropout": training_args.lora_dropout,
            "use_rslora": training_args.use_rslora,
            "use_dora": training_args.use_dora,
        }
        peft_config = LoraConfig(**kwargs)
        logger.info(f"peft_config: {peft_config}")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    kwargs = {"token": model_args.hf_token} if model_args.hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_model_name, **kwargs)
    update_tokenizer(tokenizer, data_args.max_sequence_length)

    train_dataset, eval_dataset = load_datasets(data_args, tokenizer, model_args.id2label, random_seed=training_args.seed)
    model = load_model(model_args, training_args, tokenizer)
    save_config(model.config, model_args.pretrained_model_name, model_args.id2label, training_args.output_dir)

    logger.info(f"***** Running training for {training_args.num_train_epochs} epochs *****")

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Input.collate_fn,
        custom_id2label=model_args.id2label,
        compute_metrics=partial(compute_metrics, id2label=model_args.id2label),
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    train()
