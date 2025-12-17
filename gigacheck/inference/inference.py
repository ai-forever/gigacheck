import click
from loguru import logger

from transformers import AutoConfig
from gigacheck.inference.src.mistral_detector import MistralDetector


def load_model(model_path, device):
    config = AutoConfig.from_pretrained(model_path)

    model = MistralDetector(
        max_seq_len=config.max_length,
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
