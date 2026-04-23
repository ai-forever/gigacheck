from typing import Dict, Optional, Tuple, Union, Any
import contextlib

from gigacheck.model.src.interval_detector.config import DetrModelConfig
from gigacheck.model.src.interval_detector.build import build_detr_model
from gigacheck.model.src.interval_detector.utils import get_ref_points

import torch
from transformers import ModernBertModel, ModernBertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from packaging import version
import transformers
TRANSFORMERS_VERSION = version.parse(version.parse(transformers.__version__).base_version)


class ModernBertAIDetectorForSequenceClassification(ModernBertPreTrainedModel):
    _no_split_modules = [
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
        max_sequence_length: int = None,
    ):
        assert TRANSFORMERS_VERSION >= version.parse("5.0.0rc0")
        super().__init__(config)

        self.num_labels = config.num_labels
        self.model = ModernBertModel(config)

        self.config.classifier_dropout = 0.1
        self.config.id2label = id2label

        self.id2label = id2label

        if not hasattr(self.config, "with_detr"):
            self.config.with_detr = with_detr

        # only for detr training now
        self.classification_head = None
        assert self.config.with_detr

        self.config.architectures.append("ModernBertAIDetectorForSequenceClassification")
        self.ce_weights = ce_weights
        self.freeze_backbone = freeze_backbone

        if self.config.with_detr:

            if detr_config is not None:
                detr_config = DetrModelConfig.from_dict(detr_config)
            elif detr_config is None and hasattr(self.config, "detr_config"):
                detr_config = DetrModelConfig.from_dict(self.config.detr_config)
            else:
                detr_config = DetrModelConfig()

            if hasattr(self.config, "max_sequence_length"):
                max_sequence_length = self.config.max_sequence_length

            self.detr, self.criterion = build_detr_model(
                config=detr_config,
                hidden_size=self.config.hidden_size,
                max_seq_len=max_sequence_length,
                with_loss=True,
            )
            self.config.detr_config = detr_config.to_dict()
            self.config.max_sequence_length = max_sequence_length

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
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)

        if batch_size is None and seq_len is None:
            if inputs_embeds is not None:
                batch_size, seq_len = inputs_embeds.shape[:2]
            else:
                batch_size, seq_len = input_ids.shape[:2]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

        model_output = self.model(
            input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return model_output

    def get_output(self, loss, logits, model_output):
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
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
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # labels: Optional[torch.LongTensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
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
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                indices=indices,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                batch_size=batch_size,
                seq_len=seq_len,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = model_output[0]

            if self.classification_head is not None:
                raise NotImplementedError
            else:
                pooled_logits = None

        all_outputs = (pooled_logits,)

        if self.config.with_detr:
            loss, out = self.inference_detr(input_ids, attention_mask, targets, hidden_states)
            all_outputs = all_outputs + (out,)  # type: ignore
        else:
            raise NotImplementedError

        if not return_dict:
            output = (pooled_logits,) + model_output[1:] if not return_detr_output else all_outputs + model_output[1:]
            return ((loss,) + output) if loss is not None else output

        return self.get_output(
            loss=loss,
            logits=pooled_logits if not return_detr_output else all_outputs,
            model_output=model_output,
        )
