import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dacite import from_dict
from loguru import logger


class Labels(Enum):
    AI = "ai"
    HUMAN = "human"
    MIXED = "mixed"

    def __str__(self):
        return str(self.value)

    @classmethod
    def values(cls):
        return [str(el) for el in cls]


@dataclass
class TextSample:
    label: str
    model: str
    text: str
    data_type: str
    topic_id: str = None
    prompt_type: Optional[str] = None
    prompt: Optional[str] = None
    lang: str = None

    ai_char_intervals: Optional[List[Any]] = None
    sep_indices: Optional[List[int]] = None

    def __post_init__(self):
        assert self.label in Labels.values()
        assert self.text

        if self.label == str(Labels.MIXED):
            if self.ai_char_intervals is None or not len(self.ai_char_intervals):
                logger.warning("ai_char_intervals are None or empty")

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v}


def create_sample_from_dict(data) -> TextSample:

    def _check_required_keys(data: Dict[str, Any], required_keys: List[str]):
        if not all(key in data for key in required_keys):
            missed = [key for key in required_keys if key not in data]
            raise ValueError(f"Missed keys: {missed}")

    required_keys = ["label", "model", "text", "data_type"]
    _check_required_keys(data, required_keys)
    return from_dict(data_class=TextSample, data=data)


def save_samples_jsonl(samples: List[TextSample], filename: str, out_dir: Union[Path, str]) -> None:
    out_json_file = Path(out_dir) / f"{filename}.jsonl"

    out_file = open(str(out_json_file), "w", encoding="utf8")
    for text_sample in samples:
        out_file.write(json.dumps(text_sample.to_dict(), ensure_ascii=False) + "\n")
    out_file.close()
    print(f">> Saved {len(samples)} texts to {out_json_file}")


def read_jsonl(filepath: Union[Path, str]):
    with open(str(filepath), "r", encoding="utf8") as f:
        for line in f:
            item = json.loads(line)
            yield item


def read_jsonl_dataset(filepath: Union[Path, str]):
    for item in read_jsonl(str(filepath)):
        text_sample: TextSample = create_sample_from_dict(item)
        yield text_sample

