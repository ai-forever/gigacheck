import click
from loguru import logger
from packaging import version

import transformers
from transformers import AutoConfig
from gigacheck.inference.src.mistral_detector import MistralDetector
from gigacheck.inference.src.mmbert_detector import ModernBertDetector

TRANSFORMERS_VERSION = version.parse(version.parse(transformers.__version__).base_version)


def load_model(model_path, device):
    config = AutoConfig.from_pretrained(model_path)
    arch_name = config.architectures[0]
    max_length = config.max_length if hasattr(config, "max_length") else config.max_sequence_length

    if "ModernBert" in arch_name:
        assert TRANSFORMERS_VERSION >= version.parse("5.0.0rc0")
        model = ModernBertDetector(
            max_seq_len=max_length,
            with_detr=config.with_detr,
            id2label=config.id2label,
            device=device,
        ).from_pretrained(model_path)
    else:
        assert TRANSFORMERS_VERSION < version.parse("4.58.0")
        model = MistralDetector(
            max_seq_len=max_length,
            with_detr=config.with_detr,
            id2label=config.id2label,
            device=device,
        ).from_pretrained(model_path)

    return model


@click.command()
@click.option("--model_path", type=str, required=True)
@click.option("--text", type=str, required=True)
@click.option("--device", type=str, default="cuda:0")
def main(model_path: str, text: str, device: str):
    model = load_model(model_path, device)
    output = model.predict(text)
    logger.info(f"[model={model_path}] {output}")


if __name__ == "__main__":
    main()
