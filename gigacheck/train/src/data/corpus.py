from pathlib import Path
from typing import List
from loguru import logger

from gigacheck.train.src.data.data_format import TextSample, read_jsonl_dataset


def load_texts(data_file: str) -> List[TextSample]:
    """
    :return: A list of TextSample objects.
    """
    logger.info(f"Loading data from: {data_file}")

    texts: List[TextSample] = []
    for sample in read_jsonl_dataset(data_file):
        texts.append(sample)

    return texts


class Corpus:
    """
    Corpus class for storing and handling text samples.
    """

    def __init__(self, data_path: str):

        if not Path(data_path).exists():
            raise FileNotFoundError(f"File {data_path} does not exist")

        if Path(data_path).expanduser().is_file():
            data: List[TextSample] = load_texts(data_path)
        else:
            files = list(Path(data_path).expanduser().glob("*.jsonl"))
            logger.info(f"Found {len(files)} data files")
            assert len(files) > 0, "No data files have been found"
            data = sum([load_texts(str(fpath)) for fpath in files], [])

        self.data = data
        self.name = Path(data_path).stem

        logger.info(f"[{self.name}] Loaded text samples: {len(data)}")

    def __len__(self) -> int:
        return len(self.data)
