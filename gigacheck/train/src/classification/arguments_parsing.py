from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, List, Any

import transformers


@dataclass
class ModelArguments:
    """Model configuration arguments.
    """

    pretrained_model_name: str = field(default="mistralai/Mistral-7B-v0.3", metadata={"help": "Name or path to the base model."})
    id2label: Dict[int, str] = field(default_factory=lambda: ({ 0: "ai", 1: "human"}))
    hf_token: Optional[str] = field(default=None, metadata={"help": "Auth token for HF model."})


@dataclass
class DataArguments:
    """Data configuration arguments.
    """

    train_data_path: str = field(metadata={"help": "Path to the training .jsonl data."})
    eval_data_path: Optional[str] = field(default=None, metadata={"help": "Path to the training .jsonl data."})
    random_sequence_length: bool = field(default=True, metadata={"help": "Whether to apply augmentations to texts."})
    max_sequence_length: int = field(default=512, metadata={"help": "Sequence length to use during training."})
    min_sequence_length: int = field(default=15, metadata={"help": "Minimum text len to use during augmentations."})

    def to_dict(self):
        return {k: str(v) for k, v in asdict(self).items()}


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Additional training arguments for advanced configurations.

    Extends the Hugging Face `transformers.TrainingArguments` with extra parameters.
    """

    attn_implementation: str = field(default=None, metadata={"help": "May be flash_attention_2 or None."})
    metric_for_best_model: str = field(
        default="eval/accuracy", metadata={"help": "Metric to use for best model selection."},
    )

    # LoRa settings
    lora_enable: bool = field(default=True, metadata={"help": "Enable LoRa-based PEFT."})
    lora_target_modules: List[str] = field(default_factory=lambda: (["q_proj", "v_proj"]))
    lora_r: int = field(default=8, metadata={"help": "Rank for LoRa matrices."})
    lora_alpha: int = field(default=16, metadata={"help": "Scaling factor for LoRa updates."})
    lora_dropout: float = field(default=0.1, metadata={"help": "Dropout probability for LoRa layers."})
    lora_bias: str = field(default="none", metadata={"help": "Type of LoRa bias parameters to train"})
    use_dora: bool = field(default=False, metadata={"help": "Enable DORA extension for PEFT."})
    use_rslora: bool = field(default=False, metadata={"help": "Enable RSLora extension for PEFT."})
    ce_weights: List[float] = field(default=None, metadata={"help": "Class weights for CE loss"})

    clearml_dotenv_path: str = field(default=None, metadata={"help": "ClearML configuration"})
    clearml_project_name: str = field(default=None, metadata={"help": "ClearML configuration"})
    clearml_task_name: str = field(default=None, metadata={"help": "ClearML configuration"})
