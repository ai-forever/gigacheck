import os
from typing import Dict, List, Optional, Union, Any

import torch
from torch import nn

from transformers import Trainer
from transformers.integrations import ClearMLCallback
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import logging
from transformers.trainer_callback import TrainerCallback

from accelerate.state import AcceleratorState
from dotenv import load_dotenv
from clearml import Task

from gigacheck.train.src.classification.arguments_parsing import TrainingArguments

logger = logging.get_logger(__name__)


def setup_clearml_task(config: Dict) -> Task:
    """
    Initialize a logger task using given configuration.

    Args:
        config (Dict): The clearml configuration object with project and experiment names.

    Returns:
        Task: An initialized Task object.
    """
    tags = [t for t in config.get("tags", [])]
    task = Task.init(
        project_name=config.get("project_name", "TrainProject"),
        task_name=config.get("task_name", "train"),
        reuse_last_task_id=False,
        tags=tags,
        auto_connect_frameworks={
            "tensorboard": {"report_hparams": True},
            "pytorch": "*.pt",
            "detect_repository": True,
            "jsonargparse": True,
        },
    )
    task.set_comment(config.get("description", "train"))
    return task


class CustomTrainer(Trainer):
    def __init__(
        self,
        args: TrainingArguments = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        **kwargs,
    ):
        self.args = args
        self.id2label = kwargs.pop("id2label", {0: "ai", 1: "human", 2: "mixed"})

        self.custom_id2label = kwargs.pop("custom_id2label", {0: "ai", 1: "human", 2: "mixed"})
        self._tokenizer = kwargs.get("tokenizer", None)

        if self.is_world_process_zero():
            if args.clearml_dotenv_path:
                load_dotenv(args.clearml_dotenv_path)
                task = setup_clearml_task({"project_name": args.clearml_project_name, "task_name": args.clearml_task_name})
                Task.__main_task = task
                if callbacks:
                    callbacks.append(ClearMLCallback)
                else:
                    callbacks = [ClearMLCallback]

        super().__init__(args=args, callbacks=callbacks, **kwargs)

        self.model_accepts_loss_kwargs = False

        state = AcceleratorState()
        print(f"rank={os.environ.get('LOCAL_RANK', -1)} AcceleratorState: {state=}")

    def _get_model_output(self, model, inputs: dict):
        texts, masks, labels, samples = (
            inputs["tokens"],
            inputs["masks"],
            inputs["labels"],
            inputs["samples"],
        )
        outputs = model(
            input_ids=texts,
            attention_mask=masks,
            labels=labels,
            output_hidden_states=False,
        )

        return outputs

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        labels = inputs["labels"]
        with torch.no_grad():
            model_output = self._get_model_output(model, inputs)

            loss = (
                model_output.loss
                if model_output.loss is not None and isinstance(model_output.loss, torch.Tensor)
                else torch.tensor(0, device=labels.device)
            )
            logits = model_output.logits if not isinstance(model_output.logits, tuple) else model_output.logits[0]
            loss = loss.detach()

        if prediction_loss_only:
            return (loss, None, None)

        out_labels = nested_detach(labels)
        out_logits = nested_detach(logits)

        return (loss, out_logits, out_labels)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = self._get_model_output(model, inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
