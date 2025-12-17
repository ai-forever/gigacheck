import json
import os
from typing import List

import click
from gigacheck.train.src.data.data_format import TextSample, create_sample_from_dict, save_samples_jsonl


def get_ai_intervals(original_text, revised_text, mode="char"):
    original_seq = original_text if mode == "char" else original_text.split()
    revised_seq = revised_text if mode == "char" else revised_text.split()
    intervals = []
    start = None

    for i, (o_char, r_char) in enumerate(zip(original_seq, revised_seq)):
        if o_char != r_char:
            start = i
            break

    if start is not None:
        intervals.append([start, len(revised_seq)])

    return intervals


def read_mixset_json(filepath: str, original_split: str, num_classes: int):
    with open(filepath) as f:
        data = json.load(f)

    type_mapping = {
        "email_content": "story",
        "blog": "article",
        "speech": "story",
        "SQuAD1_LLMs": "knowledge",
        "news": "news",
        "TruthfulQA_LLMs": "story",
        "game_review": "review",
        "paper_abstract": "paper_abstract",
        "NarrativeQA_LLMs": "story",
    }
    file_name = os.path.splitext(os.path.basename(filepath))[0]

    for entry in data:
        orig_text_is_ai = entry.get("model", None)
        category = entry["mixset_category"]
        # NOTE: uncomment for loading dataset for DETR (so that mixed texts without intervals are not loaded)
        # if (orig_text_is_ai and "MGT" in category) or \
        #     (not orig_text_is_ai and "complete" not in category):
        #     continue

        if num_classes == 2:
            label = "ai" if entry["binary"] == "MGT" else "human"
        else:
            label = (
                "human" if entry["binary"] == "HWT" else "ai" if orig_text_is_ai and "MGT" not in category else "mixed"
            )

        yield {
            "label": label,
            "model": entry["mixset_category"].split("_")[0],
            "text": entry["revised"],
            "source": entry["category"],
            "source_dataset": file_name,
            "data_type": type_mapping[entry["category"]],
            "original_split": original_split,
            "prompt_type": (
                "adapt_token"
                if category == "MGT_polish_token"
                else "adapt_sentence"
                if category == "MGT_polish_sentence"
                else "_".join(category.split("_")[1:])
            ),
            "ai_char_intervals": (
                get_ai_intervals(entry["original"], entry["revised"])
                # Because we also have mixed texts without specified intervals
                if label == "mixed" and "complete" in category
                else None
            ),
            "ai_words_intervals": (
                get_ai_intervals(entry["original"], entry["revised"], mode="words")
                if label == "mixed" and "complete" in category
                else None
            ),
        }


def load_mixset_data(
    data_path: str, original_split: str, num_classes: int, json_out_dir: str, images_out_dir="/data/roft/images"
) -> None:
    samples: List[TextSample] = []
    file_name = os.path.splitext(os.path.basename(data_path))[0]

    for data in read_mixset_json(data_path, original_split, num_classes):
        src, model_name = data["source"], data["model"]
        samples.append(create_sample_from_dict(data))

    save_samples_jsonl(samples, f"{file_name}", json_out_dir)


@click.command()
@click.option("--dataset_path", type=str, default="/data/data/en/mixset/orig/Mixset_test.json")
@click.option("--original_split", type=str, default="test")
@click.option("--num_classes", type=int, default=2)
@click.option("--json_out_dir", type=str, default="/data/data/en/mixset/jsonl_orig_binary")
def main(dataset_path, original_split, json_out_dir, num_classes):
    assert 1 < num_classes < 4, "For MixSet dataset only 2 or 3 classes are available"
    load_mixset_data(dataset_path, original_split, num_classes, json_out_dir)


if __name__ == "__main__":
    main()
