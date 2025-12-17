from typing import Dict, List, Optional, Union, Any

import torch

from torch import nn
from transformers import (
    Trainer,
    TrainerControl,
    TrainerState,
    is_torch_xla_available,
)
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import SaveStrategy
from transformers.integrations import ClearMLCallback
from transformers.trainer_callback import TrainerCallback
from transformers.trainer import logger

from accelerate.state import AcceleratorState
from clearml import Task
from dotenv import load_dotenv

from gigacheck.train.src.detection.arguments_parsing import TrainingArguments

# from transformers.utils import logging
# logger.setLevel(logging.DEBUG)
PREFIX_LOGS_DIR = "logs"

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


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


class AddExtraLossesToTrainerState(TrainerCallback):
    def __init__(self, extra_losses: List[str]):
        self.extra_losses = extra_losses

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.extra_losses = {k: torch.tensor(0.0).to(args.device) for k in self.extra_losses}
        return control


class DetrTrainer(Trainer):
    def __init__(
        self,
        args: TrainingArguments = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        log_extra_losses: List[str] = None,
        sampling_field: str = "label",
        **kwargs,
    ):
        self._tokenizer = kwargs.pop("tokenizer_to_save", None)
        self.args = args
        self.sampling_field = sampling_field
        self.custom_id2label = kwargs.pop("custom_id2label", {0: "ai", 1: "human", 2: "mixed"})

        if self.is_world_process_zero():
            if args.clearml_dotenv_path:
                load_dotenv(args.clearml_dotenv_path)
                task = setup_clearml_task(
                    {"project_name": args.clearml_project_name, "task_name": args.clearml_task_name})
                Task.__main_task = task
                if callbacks:
                    callbacks.append(ClearMLCallback)
                else:
                    callbacks = [ClearMLCallback]

        super().__init__(args=args, callbacks=callbacks, **kwargs)

        if log_extra_losses is not None:
            self.add_callback(AddExtraLossesToTrainerState(log_extra_losses))
        self.model_accepts_loss_kwargs = False

        state = AcceleratorState()
        print(f"Process={self.accelerator.process_index} "
              f"\nmixed_precision: {state.mixed_precision=} "
              f"\ndistributed_type: {self.accelerator.distributed_type}")

    def _get_model_output(self, model, inputs: dict, return_detr_output: bool = False):
        texts, masks, labels, samples = (
            inputs["tokens"],
            inputs["masks"],
            inputs["labels"],
            inputs["samples"],
        )

        targets = {
            "text_labels": labels,
            "span_labels": inputs["span_labels"],
            "input_ids": texts,
            "samples": samples,
            "metas": inputs["metas"],
        }

        outputs = model(
            texts,
            attention_mask=masks,
            targets=targets,
            return_detr_output=return_detr_output,
        )

        return outputs

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        text_lengths = [s.len for s in inputs["metas"]]
        text_inds = [s.index for s in inputs["metas"]]

        with torch.no_grad():
            out = model(
                input_ids=inputs["tokens"],
                attention_mask=inputs["masks"],
                targets={"span_labels": inputs["span_labels"]},
                return_detr_output=True,
            )
            loss = out.loss[0]
            detr_output = out.logits[-1]
            loss = loss.detach()

        if prediction_loss_only:
            return (loss, None, None)

        device = detr_output["pred_logits"].device

        # batch_size = detr_output["pred_logits"].shape[0]
        # gt_spans = []
        # if len(inputs["span_labels"]):
        #     # iterate over batch
        #     for s in inputs["span_labels"]:
        #         if len(s["spans"]):
        #             gt_spans.extend([span for span in s["spans"]])

        out_logits = (
            detr_output["pred_spans"],
            detr_output["pred_logits"],
        )
        out_logits = nested_detach(out_logits)

        # all spans for all samples in batch
        out_labels = (
            torch.tensor(text_lengths, device=device),
            torch.tensor(text_inds, device=device),
        )
        out_labels = nested_detach(out_labels)

        return (loss, out_logits, out_labels)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = self._get_model_output(model, inputs)

        loss, weight_losses_dict = outputs.loss
        for k, v in weight_losses_dict.items():
            if k in self.control.extra_losses:
                if v is not None:
                    if self.args.n_gpu > 1:
                        v = v.mean()
                    self.control.extra_losses[k] += v.detach() / self.args.gradient_accumulation_steps

        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(
            self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        """adapted from Trainer._maybe_log_save_evaluate to support logging extra losses"""
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)

            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            if hasattr(self.control, "extra_losses"):
                for k, v in self.control.extra_losses.items():
                    logs[k] = self._nested_gather(v).mean().item()
                    # reset the loss
                    self.control.extra_losses[k] -= self.control.extra_losses[k]

                    logs[k] = round(logs[k] / (self.state.global_step - self._globalstep_last_logged), 4)

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

            if grad_norm is not None and logs["grad_norm"] > 1e+6 and self.state.global_step > 2:
                raise RuntimeError("Gradient explosion")

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(output_dir)
            logger.info(f"Saving tokenizer to {output_dir}")
