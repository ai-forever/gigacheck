import click
from loguru import logger

import torch
from transformers import AutoConfig, AutoTokenizer
from peft import PeftModel

from gigacheck.model.mistral_ai_detector import MistralAIDetectorForSequenceClassification


def join_adapter(adapter_path, config_path, output_path):
    compute_dtype = torch.bfloat16
    config = AutoConfig.from_pretrained(config_path)
    base_model_path = config.pretrained_model_name_or_path

    logger.info(f"Loaded model config: {config}. \nPretrained model: {base_model_path}")
    model = MistralAIDetectorForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=base_model_path,
        config=config,
        torch_dtype=compute_dtype,
        id2label=config.id2label,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    model = PeftModel.from_pretrained(model, adapter_path, torch_dtype=compute_dtype)
    model = model.merge_and_unload()
    model.eval()

    logger.info("Model loaded, saving...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info(f"Saved to {output_path}")


@click.command()
@click.option("--lora_ckpt_path", type=str, required=True)
@click.option("--config_path", type=str, required=True)
@click.option("--output_path", type=str, required=True)
def main(lora_ckpt_path: str, config_path: str, output_path: str):
    join_adapter(lora_ckpt_path, config_path, output_path)


if __name__ == "__main__":
    main()
