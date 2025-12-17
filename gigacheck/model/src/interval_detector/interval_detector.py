import math
from typing import Tuple

import torch
from loguru import logger
from torch import nn

from gigacheck.model.src.interval_detector.dn_detr.dn_components import dn_post_process
from gigacheck.model.src.interval_detector.modules.layers import MLP, SlimMLP, inverse_sigmoid
from gigacheck.model.src.interval_detector.modules.position_encoding import PositionEmbeddingSine
from gigacheck.model.src.interval_detector.modules.transformer import DETRTransformer

from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig


class IntervalDETRConfig(PretrainedConfig):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_input_proj_layers: int,
        proj_dropout: float,
        aux_loss=True,
        max_s_l=75,
        span_loss_type="l1",
        n_input_proj=2,
        dn_detr: bool = False,
        use_focal_loss=False,
        **kwargs,
    ):
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_input_proj_layers = num_input_proj_layers
        self.proj_dropout = proj_dropout
        self.aux_loss = aux_loss
        self.max_s_l = max_s_l
        self.span_loss_type = span_loss_type
        self.n_input_proj = n_input_proj
        self.dn_detr = dn_detr
        self.use_focal_loss = use_focal_loss
        self.dn_detr = dn_detr
        super().__init__(**kwargs)


class IntervalDETR(PreTrainedModel):
    """This is the Interval-DETR module that performs ai intervals localization."""

    def __init__(
        self,
        config: IntervalDETRConfig,
        transformer: DETRTransformer,
        position_embed: PositionEmbeddingSine,
    ):
        """Initializes the model.
        Parameters:
            input_dim (int): input dimension
            model_dim (int): hidden dimension of the model
            num_input_proj_layers (int): number of input projection layers
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            max_s_l: int, maximum seq len
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            dn_detr: bool, if True, apply denoising
        """
        super().__init__(config=config)
        logger.info("Building IntervalDETR")

        self.input_proj = MLP(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.model_dim,
            output_dim=self.config.model_dim,
            num_layers=self.config.num_input_proj_layers,
            dropout=self.config.proj_dropout,
        )

        self.dn_detr = self.config.dn_detr
        self.transformer = transformer
        self.position_embed: nn.Module = position_embed

        self.span_loss_type = self.config.span_loss_type
        self.max_s_l = self.config.max_s_l
        span_embed = SlimMLP(input_dim=self.config.model_dim, hidden_dim=self.config.model_dim, output_dim=2, num_layers=3)
        self.transformer.decoder.bbox_embed = span_embed

        num_classes = 2  # 0: foreground, 1: background
        self.class_embed = nn.Linear(self.config.model_dim, num_classes)
        self.n_input_proj = self.config.n_input_proj

        self.aux_loss = self.config.aux_loss
        self.use_focal_loss = self.config.use_focal_loss
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, MLP) or isinstance(module, nn.LayerNorm) or isinstance(module, SlimMLP):
                module.reset_parameters()
        self.transformer.reset_parameters()

        # init bbox_embed
        nn.init.constant_(self.transformer.decoder.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.transformer.decoder.bbox_embed.layers[-1].bias.data, 0)

        std = 1.0 / math.sqrt(self.class_embed.weight.size(1))
        torch.nn.init.normal_(self.class_embed.weight, std=std)
        prior_prob = 0.3
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        num_classes = 2
        device = next(self.class_embed.parameters()).device
        init_value = torch.ones(num_classes, dtype=self.class_embed.bias.data.dtype, device=device) * bias_value
        self.class_embed.bias.data = init_value

    def _predict_spans(self, output: torch.Tensor, reference: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        offset = self.transformer.decoder.bbox_embed(output)  # (#layers, bsz, #queries, 2)
        reference_before_sigmoid = inverse_sigmoid(reference)
        outputs_coord = offset + reference_before_sigmoid
        return outputs_coord.sigmoid(), offset

    def forward(self, tokens, attention_mask, memory, targets):
        """It returns a dict with the following elements:
        - "pred_spans": The normalized boxes coordinates for all queries, represented as
                        (center_x, width). These values are normalized in [0, 1],
                        relative to the size of each individual text (disregarding possible padding).
                        See PostProcess for information on how to retrieve the unnormalized interval.
        - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                         dictionnaries containing the two above keys for each decoder layer.
        """

        # project input features to the desired dimension
        memory = self.input_proj(memory)  # [bsz, L, d]

        pos = self.position_embed(memory, attention_mask)  # (bsz, L, d)
        # (#layers, bsz, #queries, d), (#layers, bsz, #queries, d), (bsz, L, d), dict
        hs, reference, mask_dict = self.transformer(memory, attention_mask, pos, targets)

        outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)
        outputs_coord, offsets = self._predict_spans(hs, reference)

        # probs = torch.softmax(outputs_class[-1], -1)
        # shifts = outputs_coord - reference

        if self.dn_detr:
            # denoise postprocessing
            outputs_class, outputs_coord, offsets = dn_post_process(
                outputs_class,
                outputs_coord,
                offsets,
                mask_dict,
            )

        out = {
            "pred_logits": outputs_class[-1].to(torch.float32),
            "pred_spans": outputs_coord[-1].to(torch.float32),
            "hs": hs,
            "mask_dict": mask_dict,
            "offset": offsets[-1].to(torch.float32),
        }

        if self.aux_loss:
            out["aux_outputs"] = [
                {
                    "pred_logits": logits.to(torch.float32),
                    "pred_spans": outputs_coord[lvl].to(torch.float32),
                    "offset": offsets[lvl].to(torch.float32),
                    "mask_dict": mask_dict,
                }
                for lvl, logits in enumerate(outputs_class[:-1])
            ]
        return out
