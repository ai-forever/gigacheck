import os
from collections import defaultdict
from typing import List

import pandas as pd
from gigacheck.train.src.data.data_format import TextSample, create_sample_from_dict, save_samples_jsonl

DATASET_NAME = "tweepfake"


def read_csv(filepath):
    df = pd.read_csv(filepath, sep=";")

    for id_, row in df.iterrows():
        label = row["account.type"]
        label = "human" if label == "human" else "ai"
        yield id_, {
            "text": row["text"],
            "label": label,
            "source": "tweeter",
            "model": row["class_type"],
            "data_type": "tweet",
            "prompt": None,
            "prompt_type": None,
            "topic_id": None,
            "source_dataset": DATASET_NAME,
        }


def load_tweepfake_data(data_path, json_out_dir="converted", save_by_splits=False):

    samples: List[TextSample] = []
    samples_by_split = defaultdict(list)
    for split in ["train", "validation", "test"]:
        file_path = os.path.join(data_path, split + ".csv")

        for id, data in read_csv(file_path):

            data["original_split"] = split if split != "validation" else "valid"
            src, model_name = data["source"], data["model"]

            samples.append(create_sample_from_dict(data))
            samples_by_split[data["original_split"]].append(samples[-1])

    if not save_by_splits:
        save_samples_jsonl(samples, f"{DATASET_NAME}", json_out_dir)
    else:
        for split, samples in samples_by_split.items():
            save_samples_jsonl(samples, split, json_out_dir)


if __name__ == "__main__":
    load_tweepfake_data(
        data_path="/data/tweepfake/splits",
        json_out_dir="/data/tweepfake/converted",
        save_by_splits=True,
    )
