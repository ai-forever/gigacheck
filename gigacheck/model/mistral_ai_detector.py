from typing import Dict, Optional, Tuple, Union, Any
import contextlib

from gigacheck.model.src.interval_detector.config import DetrModelConfig
from gigacheck.model.src.interval_detector.build import build_detr_model
from gigacheck.model.src.interval_detector.utils import get_ref_points

import torch
from torch import nn
from transformers import MistralModel, MistralPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


class MistralAIDetectorForSequenceClassification(MistralPreTrainedModel):
    _no_split_modules = [
        "MistralDecoderLayer",
        "TransformerEncoderLayer",
        "TransformerDecoderLayer",
    ]

    def __init__(
        self,
        config,
        with_detr: bool = False,
        detr_config: Optional[Dict[str, Any]] = None,
        ce_weights=None,
        freeze_backbone: bool = False,
        id2label: Dict[int, str] = None,
    ):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.model = MistralModel(config)

        self.config.classifier_dropout = 0.1
        self.config.id2label = id2label

        self.id2label = id2label

        if not hasattr(self.config, "with_detr"):
            self.config.with_detr = with_detr

        self.classification_head = ClassificationHead(self.config, self.num_labels)

        self.config.architectures.append("MistralAIDetectorForSequenceClassification")
        self.ce_weights = ce_weights
        self.freeze_backbone = freeze_backbone

        if self.config.with_detr:

            if detr_config is not None:
                detr_config = DetrModelConfig.from_dict(detr_config)
            elif detr_config is None and hasattr(self.config, "detr_config"):
                detr_config = DetrModelConfig.from_dict(self.config.detr_config)
            else:
                detr_config = DetrModelConfig()

            self.detr, self.criterion = build_detr_model(
                config=detr_config,
                hidden_size=self.config.hidden_size,
                max_seq_len=self.config.max_length,
                with_loss=True,
            )
            self.config.detr_config = detr_config.to_dict()

        # Initialize weights and apply final processing
        self.post_init()

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        if self.classification_head is not None:
            for param in self.classification_head.parameters():
                param.requires_grad = False
            self.classification_head.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self._freeze_backbone()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward_backbone(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        model_output = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return model_output

    def get_output(self, loss, logits, model_output):
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=model_output.past_key_values if model_output is not None else None,
            hidden_states=model_output.hidden_states if model_output is not None else None,
            attentions=model_output.attentions if model_output is not None else None,
        )

    def inference_detr(self, input_ids, attention_mask, targets, hidden_states) -> Tuple[Optional[tuple], Dict[str, Any]]:
        model_detr_dtype = next(self.detr.parameters()).dtype
        if hidden_states.dtype is not model_detr_dtype:
            hidden_states = hidden_states.to(model_detr_dtype)

        out = self.detr(input_ids, attention_mask, hidden_states, targets)

        loss = None
        if targets is not None:
            loss_dict = self.criterion(
                out,
                targets,
                ref_points=get_ref_points(self.detr, "pt"),
            )
            weight_dict = self.criterion.weight_dict
            weighted_losses = {k: loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict}

            loss = (sum(weighted_losses.values()), weighted_losses)  # type: ignore
        return loss, out

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_detr_output: bool = False,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        # **_,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:

        if not self.config.with_detr:
            return_detr_output = False

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        context = torch.no_grad if self.freeze_backbone else contextlib.nullcontext
        with context():
            model_output = self.forward_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = model_output[0]

            if self.classification_head is not None:
                tokens = get_eos_token(input_ids, hidden_states, self.config, inputs_embeds=inputs_embeds)  # (B, dim)
                pooled_logits = self.classification_head(tokens)
            else:
                pooled_logits = None

        all_outputs = (pooled_logits,)
        loss = 0
        if self.config.with_detr:
            loss, out = self.inference_detr(input_ids, attention_mask, targets, hidden_states)
            all_outputs = all_outputs + (out,)  # type: ignore
        else:
            if labels is not None and pooled_logits is not None:
                labels = labels.to(pooled_logits.device)
                loss = calculate_cross_entropy(pooled_logits, labels, self.num_labels, self.ce_weights)

        if not return_dict:
            output = (pooled_logits,) + model_output[1:] if not return_detr_output else all_outputs + model_output[1:]
            return ((loss,) + output) if loss is not None else output

        return self.get_output(
            loss=loss,
            logits=pooled_logits if not return_detr_output else all_outputs,
            model_output=model_output,
        )


def get_eos_token(input_ids, hidden_states, config, inputs_embeds=None):
    if input_ids is not None:
        batch_size = input_ids.shape[0]
    else:
        batch_size = inputs_embeds.shape[0]

    if config.pad_token_id is None and batch_size != 1:
        raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
    if config.pad_token_id is None:
        sequence_lengths = -1
    else:
        sequence_lengths = (torch.eq(input_ids, config.pad_token_id).long().argmax(-1) - 1).to(hidden_states.device)
    # take </s> token
    tokens = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
    return tokens


def calculate_cross_entropy(logits, labels, num_labels, weights=None):
    if weights is not None:
        weights = torch.tensor(weights, device=logits.device, dtype=logits.dtype)
    else:
        weights = None
    loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
    loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
    return loss


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features  # take </s> token
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
