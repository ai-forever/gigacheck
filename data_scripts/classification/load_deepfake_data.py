import csv
import os
from pathlib import Path
from typing import List

from gigacheck.train.src.data.data_format import TextSample, create_sample_from_dict, save_samples_jsonl


DATASET_NAME = "deepfake"

SET_NAMES = {
    "cmv": "review",
    "yelp": "review",
    "xsum": "news",
    "tldr": "news",
    "eli5": "question",
    "wp": "story",
    "roct": "story",
    "hswag": "knowledge",
    "squad": "knowledge",
    "sci_gen": "paper_abstract",
    # additional test domains
    "cnn": "news",
    "dialogsum": "story",
    "imdb": "review",
    "pubmed": "question",
}

MODEL_SETS = {
    "oai": [
        # openai
        "gpt-3.5-trubo",
        "text-davinci-003",
        "text-davinci-002",
        "gpt4",  # additional model for test
    ],
    "llama": ["7B", "13B", "30B", "65B"],
    "glm": ["GLM130B"],
    "flan": [
        # flan_t5,
        "flan_t5_small",
        "flan_t5_base",
        "flan_t5_large",
        "flan_t5_xl",
        "flan_t5_xxl",
    ],
    "opt": [
        # opt,
        "opt_125m",
        "opt_350m",
        "opt_1.3b",
        "opt_2.7b",
        "opt_6.7b",
        "opt_13b",
        "opt_30b",
        "opt_iml_30b",
        "opt_iml_max_1.3b",
    ],
    "bigscience": [
        "bloom_7b",
        "t0_3b",
        "t0_11b",
    ],
    "eleuther": [
        "gpt_j",
        "gpt_neox",
    ],
}


def read_csv(filepath):
    with open(filepath, encoding="utf-8") as f:
        reader = csv.reader(f)
        for id_, row in enumerate(reader):
            if id_ == 0:
                continue
            label = int(row[1])
            label = "human" if label == 1 else "ai"
            yield id_ - 1, {
                "text": row[0],
                "label": label,
                "src": row[2] if len(row) > 2 else None,
                "prompt": None,
                "source_dataset": DATASET_NAME,
            }


def load_deepfake_data(
    data_path, images_out_dir="images", json_out_dir="converted", save_by_splits=False, test_name="test"
):
    assert test_name in ["test", "test_ood"], "test_name must be 'test' or 'test_ood'"
    for p in [images_out_dir, json_out_dir]:
        if not Path(p).exists():
            Path(p).mkdir()

    samples: List[TextSample] = []

    splits = ["train", "valid", test_name] if Path(data_path).is_dir() else ["test"]
    for split in splits:
        file_path = os.path.join(data_path, split + ".csv") if Path(data_path).is_dir() else data_path

        for id, data in read_csv(file_path):

            # {set}_{prompt_type}_{model}
            src = data["src"]
            if src is None:
                # loading data from testbeds
                data_type = "unknown"
                set_name = DATASET_NAME
                prompt_type = None

            if data["label"] != "human":
                if src is not None:
                    model_name, model_var = get_model_name(src)
                    set_name, data_type = get_source_name(src)
                    assert model_name is not None
                    assert set_name is not None
                    if model_var is not None:
                        prompt_type = src.replace(set_name + "_", "").replace("_" + model_var, "")
                    elif "human_para" in src:
                        prompt_type = "human_para"
                    elif "_para" in src:
                        prompt_type = "ai_para"
                    # model_name = model_name + "_" + prompt_type
                else:
                    model_name = "ai"
            else:
                model_name = "human"
                if src is not None:
                    set_name, data_type = get_source_name(src)
                    prompt_type = None
                    assert set_name is not None

            data["original_split"] = split if split != "test_ood" else "test"
            data["data_type"] = data_type
            data["prompt_type"] = prompt_type
            data["model"] = model_name
            data["source"] = set_name
            samples.append(create_sample_from_dict(data))

        if save_by_splits:
            save_samples_jsonl(samples, split, json_out_dir)
            samples = []

    if not save_by_splits:
        save_samples_jsonl(samples, f"{DATASET_NAME}", json_out_dir)


def get_model_name(source_str):
    # source_str is like {set}_{prompt_type}_{model}
    if "human_para" in source_str or "_para" in source_str:
        return "gpt-3.5-turbo", None
    for name, model_variations in MODEL_SETS.items():
        for model_var in model_variations:
            if "_" + model_var in source_str:
                return name, model_var
    return None, None


def get_source_name(source_str):
    # source_str is like {set}_{prompt_type}_{model}
    for name, data_type in SET_NAMES.items():
        if name + "_" in source_str:
            return name, data_type
    return None, None


if __name__ == "__main__":
    # NOTE: you need to run this script after the official one
    # github.com/yafuly/MAGE/blob/main/deployment/prepare_testbeds.py

    DATA_PATH = "/data/mage"
    # all domain, all models
    load_deepfake_data(
        data_path=f"{DATA_PATH}/cross_domains_cross_models/",
        images_out_dir=f"{DATA_PATH}/cross_domains_cross_models_converted",
        json_out_dir=f"{DATA_PATH}/cross_domains_cross_models_converted",
        save_by_splits=True,
    )

    # additional test sets
    load_deepfake_data(
        data_path=f"{DATA_PATH}/test_ood_gpt.csv",
        images_out_dir=f"{DATA_PATH}/test_ood_gpt_converted/",
        json_out_dir=f"{DATA_PATH}/test_ood_gpt_converted/",
        save_by_splits=True,
    )
    load_deepfake_data(
        data_path=f"{DATA_PATH}/test_ood_gpt_para.csv",
        images_out_dir=f"{DATA_PATH}/test_ood_gpt_para_converted",
        json_out_dir=f"{DATA_PATH}/test_ood_gpt_para_converted",
        save_by_splits=True,
    )

    # out of models
    for model_name in ["_7B", "bloom_7b", "flan_t5_small", "GLM130B", "gpt-3.5-trubo", "gpt_j", "opt_125m"]:
        load_deepfake_data(
            data_path=f"{DATA_PATH}/unseen_models/unseen_model_{model_name}",
            images_out_dir=f"{DATA_PATH}/unseen_model_{model_name}_converted",
            json_out_dir=f"{DATA_PATH}/unseen_model_{model_name}_converted",
            save_by_splits=True,
            test_name="test_ood",
        )

    # out of domains
    for domain_name in ["cmv", "eli5", "hswag", "roct", "sci_gen", "squad", "tldr", "wp", "xsum", "yelp"]:
        load_deepfake_data(
            data_path=f"{DATA_PATH}/unseen_domains/unseen_domain_{domain_name}",
            images_out_dir=f"{DATA_PATH}/unseen_domain_{domain_name}_converted",
            json_out_dir=f"{DATA_PATH}/unseen_domain_{domain_name}_converted",
            save_by_splits=True,
            test_name="test_ood",
        )
