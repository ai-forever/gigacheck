from pathlib import Path
from typing import List

import click
from gigacheck.train.src.data.data_format import (
    Labels,
    TextSample,
    create_sample_from_dict,
    save_samples_jsonl,
)
from loguru import logger

DIRS = ["essay", "news", "story"]
LABELS = ["gpt", "human"]
SOURCE_MAPPING = {"essay": "IvyPanda", "news": "Reuters", "story": "Reddit"}


def read_from_data_type_dir(dir_path: Path, label: str, original_split: str):
    label_dir = dir_path / label
    label = "ai" if label != "human" else label  # Changing label to our standard notation
    # Because 'news' subfolder has different structure
    directory_to_iter = (
        [item for item in label_dir.iterdir() if item.is_dir()] if dir_path.name == "news" else [label_dir]
    )

    for text_dir in directory_to_iter:
        for file in text_dir.glob("*.txt"):
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()
                if not text:
                    continue

            if Labels(label) == Labels.AI:
                if dir_path.name == "news":
                    prompt_file = text_dir / "headlines" / file.name
                else:
                    prompt_file = dir_path / "prompts" / file.name

                with open(prompt_file, "r", encoding="utf-8") as f:
                    prompt = f.read()
                model = "gpt3"
            else:
                prompt = None
                model = "human"

            if dir_path.name == "news":
                data_type = text_dir.parent.parent.name
            else:
                data_type = text_dir.parent.name

            yield {
                "label": label,
                "model": model,
                "text": text,
                "source": SOURCE_MAPPING[data_type],
                "source_dataset": "Ghostbuster",
                "data_type": data_type,
                "original_split": original_split,
                "prompt": prompt,
            }


def get_ghostbuster_data(dir_path: str, data_type: str, original_split: str):
    if data_type not in DIRS:
        logger.info("Loading all data types.")
        for dir in DIRS:
            for label in LABELS:
                yield from read_from_data_type_dir(Path(dir_path) / dir, label, original_split)
    else:
        logger.info(f"Loading {data_type} data type.")
        for label in LABELS:
            yield from read_from_data_type_dir(Path(dir_path) / data_type, label, original_split)


def load_ghostbuster_data(
    dataset_folder_path: str, data_type: str, original_split: str, json_out_dir: str, images_out_dir="/data/roft/images"
) -> None:
    samples: List[TextSample] = []
    file_name = data_type

    for data in get_ghostbuster_data(dataset_folder_path, data_type, original_split):
        src, model_name = data["source"], data["model"]
        samples.append(create_sample_from_dict(data))

    save_samples_jsonl(samples, f"{file_name}", json_out_dir)


@click.command()
@click.option(
    "--dataset_folder_path",
    type=str,
    default="/data/data/en/ghostbuster/orig",
    help="Folder with all the folders",
)
@click.option("--data_type", type=str, default="story", help="news/essay/story/all")
@click.option("--original_split", type=str, default="all")
@click.option("--json_out_dir", type=str, default="/data/data/en/ghostbuster/jsonl/splits/story")
def main(dataset_folder_path: str, data_type: str, original_split: str, json_out_dir: str):
    """
    This script is for loading Ghostbuster data from their data-repository:
    github.com/vivek3141/ghostbuster-data.
    The data does not have originall train/val/test split.
    """
    load_ghostbuster_data(dataset_folder_path, data_type, original_split, json_out_dir)


if __name__ == "__main__":
    main()
