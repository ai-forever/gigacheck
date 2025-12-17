import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import List

from gigacheck.train.src.data.data_format import TextSample, create_sample_from_dict, save_samples_jsonl

DATASET_NAME = "TuringBench"


def read_csv(filepath):
    with open(filepath, encoding="utf-8") as f:
        reader = csv.reader(f)
        for id_, row in enumerate(reader):
            if id_ == 0:
                continue
            label: str = row[1]
            yield id_ - 1, {
                "text": row[0],
                "label": "human" if label == "human" else "ai",
                "model": label,
                "prompt": None,
                "source_dataset": DATASET_NAME,
            }


def load_turing_bench_data(
    data_path, name="TT_gpt3", json_out_dir="converted", save_by_splits=False
):

    for p in [json_out_dir,]:
        if not Path(p).exists():
            Path(p).mkdir(parents=True)

    samples: List[TextSample] = []
    samples_by_split = defaultdict(list)
    for split in ["valid", "test", "train"]:

        file_path = os.path.join(data_path, name, split + ".csv") if Path(data_path).is_dir() else data_path
        for id, data in read_csv(file_path):
            # print(data)
            data["original_split"] = split
            data["source"] = "unknown"
            data["data_type"] = "unknown"

            samples.append(create_sample_from_dict(data))
            samples_by_split[data["original_split"]].append(samples[-1])

    if not save_by_splits:
        save_samples_jsonl(samples, f"{DATASET_NAME}", json_out_dir)
    else:
        for split, samples in samples_by_split.items():
            save_samples_jsonl(samples, split, json_out_dir)


if __name__ == "__main__":
    # NOTE: before running this script download data(TuringBench.zip)
    # from huggingface.co/datasets/turingbench/TuringBench/tree/main

    # the TT task of human vs gpt3
    load_turing_bench_data(
        data_path="/data/TuringBench",
        name="TT_gpt3",
        json_out_dir="/data/TuringBench/gpt3_converted",
        save_by_splits=True,
    )

    # the TT task of human vs fair_wmt20
    load_turing_bench_data(
        data_path="/data/TuringBench",
        name="TT_fair_wmt20",
        json_out_dir="/data/TuringBench/fair_wmt20_converted",
        save_by_splits=True,
    )
