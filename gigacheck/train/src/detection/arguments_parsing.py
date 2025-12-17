from dataclasses import asdict, dataclass, field
from typing import Optional, Dict

import transformers

from gigacheck.model.src.interval_detector.config import DetrModelConfig


@dataclass
class ModelArguments(DetrModelConfig):
    """Model configuration arguments. Extends DetrModelConfig.
    """

    pretrained_model_name: str = field(default="mistralai/Mistral-7B-v0.3", metadata={"help": "Name or path to the base model."})
    id2label: Dict[int, str] = field(default_factory=lambda: ({ 0: "ai", 1: "human", 2: "mixed"}))
    hf_token: Optional[str] = field(default=None, metadata={"help": "Auth token for HF model."})


@dataclass
class DataArguments:
    """Data configuration arguments.
    """

    train_data_path: str = field(metadata={"help": "Path to the training .jsonl data."})
    eval_data_path: Optional[str] = field(default=None, metadata={"help": "Path to the training .jsonl data."})
    random_sequence_length: bool = field(default=False, metadata={"help": "Whether to apply augmentations to texts."})
    max_sequence_length: int = field(default=512, metadata={"help": "Sequence length to use during training."})
    min_sequence_length: int = field(default=15, metadata={"help": "Minimum text len to use during augmentations."})

    def to_dict(self):
        return {k: str(v) for k, v in asdict(self).items()}


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Additional training arguments for advanced configurations.

    Extends the Hugging Face `transformers.TrainingArguments` with extra parameters.
    """

    clearml_dotenv_path: str = field(default=None, metadata={"help": "ClearML configuration"})
    clearml_project_name: str = field(default=None, metadata={"help": "ClearML configuration"})
    clearml_task_name: str = field(default=None, metadata={"help": "ClearML configuration"})

