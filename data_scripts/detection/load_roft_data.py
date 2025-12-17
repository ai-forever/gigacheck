import os
from typing import List, Tuple
import re

import click
import pandas as pd
from gigacheck.train.src.data.data_format import TextSample, create_sample_from_dict, save_samples_jsonl


def clean_string(input_string):
    input_string = re.sub(r"\n", " ", input_string)
    input_string = re.sub(r"[^A-Za-z0-9 !\"$%&\'()\*+,-./:;?@^_`~]", "", input_string)
    input_string = re.sub(r"[ ]+", " ", input_string)
    input_string = input_string.strip()

    return input_string


def read_csv(filepath: str, original_split: str, separator="_SEP_", data_type: str = "roft"):
    assert data_type in ["roft", "roft-gpt"]
    df = pd.read_csv(filepath, sep=",")

    type_mapping = {
        "Recipes": "article",
        "Presidential Speeches": "speech",
        "Short Stories": "story",
        "New York Times": "news",
    }
    file_name = os.path.splitext(os.path.basename(filepath))[0]

    # Dropping duplicates that vary only in human-predicted index
    # df.drop_duplicates(subset="prompt_body", keep="first", inplace=True)

    for id_, row in df.iterrows():
        # Removing unnecessary symbols from text
        row["prompt_body"] = clean_string(row["prompt_body"])
        row["gen_body"] = clean_string(row["gen_body"]) if isinstance(row["gen_body"], str) else row["gen_body"]

        if row["model"] == "finetuned":
            model = "gpt2-xl-finetuned"
        elif data_type == "roft":
            model = (
                "human" if (row["model"] in ("baseline", "human") or row["true_boundary_index"] == 9) else row["model"]
            )
        elif data_type == "roft-gpt":
            model = "human" if (row["model"] in ("baseline", "human") or row["label"] == 9) else row["model"]

        splitted_prompt: List[str] = row["prompt_body"].split(separator)
        combined_text, last_human_sent_idx = combine_text(splitted_prompt, row["gen_body"], separator)

        if model == "human":
            ai_char_intervals = None
        else:
            ai_char_intervals = get_intervals(splitted_prompt, combined_text, char=True)

        yield id_, {
            "label": "human" if model == "human" else "mixed",
            "text": combined_text,
            "model": model,
            "source": row["dataset"],
            "data_type": type_mapping[row["dataset"]],
            "prompt": None,
            "prompt_type": "machine_continuation",
            "topic_id": None,
            "original_split": original_split,
            "source_dataset": f"{data_type}_{file_name}",
            "ended": True,
            "temperature": None,
            "ai_char_intervals": ai_char_intervals,
            "sep_indices": get_sep_indices(row["prompt_body"], row["gen_body"], last_human_sent_idx),
        }


def load_roft_data(
    data_path: str,
    original_split: str,
    json_out_dir="/data/roft/out",
    data_type: str = "roft",
) -> None:
    samples: List[TextSample] = []
    file_name = os.path.splitext(os.path.basename(data_path))[0]

    for id, data in read_csv(data_path, original_split, data_type=data_type):
        src, model_name = data["source"], data["model"]
        samples.append(create_sample_from_dict(data))

    save_samples_jsonl(samples, f"{file_name}", json_out_dir)


def combine_text(prompt_sentences: List[str], generated_text: str, separator) -> Tuple[str, int]:
    # Calculate how many more sentences we need from gen_body to make it 10
    num_prompt_sentences = len(prompt_sentences)
    if num_prompt_sentences >= 10 or not isinstance(generated_text, str):
        return " ".join(prompt_sentences[:10]), 9

    gen_sentences = generated_text.split(separator)

    num_gen_sentences_needed = 10 - num_prompt_sentences
    combined_sentences = prompt_sentences + gen_sentences[:num_gen_sentences_needed]

    return " ".join(combined_sentences), num_prompt_sentences - 1


def get_intervals(prompt_sentences: List[str], combined_text: str, char: bool) -> List[List[int]]:
    prompt_without_sep = " ".join(prompt_sentences)
    if char:
        return [[len(prompt_without_sep) + 1, len(combined_text)]]
    return [[len(prompt_without_sep.split()), len((combined_text).split())]]


def get_sep_indices(prompt: str, generated_text: str, last_human_sent_idx: int, separator="_SEP_"):
    human_part = prompt.split(separator)[: last_human_sent_idx + 1]
    num_gen_sentences_needed = 10 - (last_human_sent_idx + 1)

    if num_gen_sentences_needed == 0 or not generated_text:
        gen_part = []
    else:
        gen_part = generated_text.split(separator)[:num_gen_sentences_needed]

    sentences = human_part + gen_part
    indices = []
    current_index = 0

    for sentence in sentences[:-1]:
        current_index += len(sentence)
        indices.append(current_index)
        current_index += 1

    return indices


@click.command()
@click.option("--dataset_path", type=str, default="/data/data/en/tda/csv/roft_gpt_tda_test.csv")
@click.option("--original_split", type=str, default="all")
@click.option("--json_out_dir", type=str, default="/data/roft/out")
@click.option("--data_type", type=str, default="roft-gpt")
def main(dataset_path, original_split, json_out_dir, data_type):
    assert data_type in ["roft", "roft-gpt"]

    if not os.path.exists(json_out_dir):
        os.mkdir(json_out_dir)

    load_roft_data(
        dataset_path,
        original_split,
        json_out_dir=json_out_dir,
        data_type=data_type,
    )


if __name__ == "__main__":
    main()
