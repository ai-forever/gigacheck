from typing import Any, Dict, Optional

import torch
from torch import nn

from gigacheck.model.src.interval_detector.dn_detr.dn_components import prepare_for_denoise
from gigacheck.model.src.interval_detector.layer_anchor import LayerAnchor

from gigacheck.model.src.interval_detector.modules.layers import inverse_sigmoid
from gigacheck.model.src.interval_detector.modules.decoder import TransformerDecoder, TransformerDecoderLayer
from gigacheck.model.src.interval_detector.modules.encoder import TransformerEncoder, TransformerEncoderLayer

from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig


class DETRTransformerConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size,
        nhead=8,
        num_decoder_layers=6,
        num_encoder_layers=6,
        num_queries=10,
        num_groups=5,
        dropout=0.1,
        keep_query_pos=False,
        return_intermediate_dec=False,
        temperature: int = 10000,
        query_scale_type: str = "cond_elewise",
        bbox_embed_diff_each_layer: bool = False,
        center_noise_scale: float = 0.7,
        width_noise_scale: float = 1.15,
        noise_offset: float = 0.05,
        dn_detr: bool = False,
        query_initialization_method: str = "default",
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers
        self.num_encoder_layers = num_encoder_layers
        self.num_queries = num_queries
        self.num_groups = num_groups
        self.dropout = dropout
        self.keep_query_pos = keep_query_pos
        self.return_intermediate_dec = return_intermediate_dec
        self.temperature = temperature
        self.query_scale_type = query_scale_type
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        self.center_noise_scale = center_noise_scale
        self.width_noise_scale = width_noise_scale
        self.noise_offset = noise_offset
        self.dn_detr = dn_detr
        self.query_initialization_method = query_initialization_method
        super().__init__(**kwargs,)


class DETRTransformer(PreTrainedModel):
    def __init__(self, config: DETRTransformerConfig):
        super().__init__(config=config)

        d_model = self.config.hidden_size

        encoder_layer = TransformerEncoderLayer(d_model, dropout=self.config.dropout)
        self.encoder = TransformerEncoder(encoder_layer, self.config.num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(
            d_model,
            self.config.nhead,
            d_model * 4,
            self.config.dropout,
            "relu",
            keep_query_pos=self.config.keep_query_pos,
        )

        self.decoder = TransformerDecoder(
            decoder_layer,
            self.config.num_decoder_layers,
            return_intermediate=self.config.return_intermediate_dec,
            d_model=d_model,
            temperature=self.config.temperature,
            keep_query_pos=self.config.keep_query_pos,
            query_scale_type=self.config.query_scale_type,
            bbox_embed_diff_each_layer=self.config.bbox_embed_diff_each_layer,
        )
        # define query tokens for decoder layers
        self.query_embed = QueryEmbed(
            self.config.num_queries,
            d_model,
            self.config.query_initialization_method,
        )
        self.reset_parameters()

        self.d_model = d_model
        self.nhead = self.config.nhead

        self.num_queries = self.config.num_queries
        self.dn_detr = self.config.dn_detr

        if self.dn_detr:
            # target encoder
            num_classes = 2  # 0: foreground, 1: background
            self.label_enc = nn.Embedding(num_classes, d_model - 1)

            self.num_groups = self.config.num_groups
            self.center_noise_scale = self.config.center_noise_scale
            self.width_noise_scale = self.config.width_noise_scale
            self.noise_offset = self.config.noise_offset

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.query_embed.reset_parameters()

    def init_special_ref_points(self):
        self.query_embed.init_special_ref_points()

    def forward(
        self,
        src: torch.Tensor,
        attention_mask: torch.Tensor,
        pos_embed: torch.Tensor,
        targets: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            src: (batch_size, L, d)
            attention_mask: (batch_size, L)
            pos_embed: (batch_size, L, d) the same as src
            targets (Optional[Dict[str, Any]]): target meta information

        Returns:

        """
        bs, l, dim = src.shape
        src = src.permute(1, 0, 2)  # (L, batch_size, d)

        pos_embed = pos_embed.permute(1, 0, 2)  # (L, batch_size, d)
        mask = ~attention_mask.bool()

        memory = self.encoder(src, pos_embed, src_key_padding_mask=mask)

        if self.dn_detr:
            input_query_label, input_query_bbox, attn_mask, mask_dict = prepare_for_denoise(
                label_enc=self.label_enc,
                targets=targets,
                refpoint_emb=self.query_embed.get_query_embed(),
                num_queries=self.num_queries,
                num_groups=self.num_groups,
                center_noise_scale=self.center_noise_scale,
                width_noise_scale=self.width_noise_scale,
                noise_offset=self.noise_offset,
                batch_size=src.size(1),
                training=self.training,
            )

        else:
            # (num_queries, bs, 2), (num_queries, bs, dim)
            input_query_bbox, input_query_label = self.query_embed.get_refpoint_for_detector(bs)
            attn_mask, mask_dict = None, None

        hs, references = self.decoder(
            input_query_label,
            memory,
            tgt_mask=attn_mask,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            refpoints_unsigmoid=input_query_bbox,
        )  # (#layers, #queries, batch_size, dim)

        return hs, references, mask_dict


class QueryEmbed(nn.Module):
    def __init__(self, num_queries: int, d_model: int, init_method: str = "default", init_map: Optional[dict] = None):
        super().__init__()
        if init_method == "custom":
            assert num_queries == len(init_map), "Number of queries must equal number of entries in init map"
        self.num_queries: int = num_queries
        self.d_model: int = d_model
        self.init_method: str = init_method
        self.init_map: Dict = init_map or {}

        if self.init_method == "pyramid":
            ratios = (0.5, 0.35, 0.11)
            self.query_embed = LayerAnchor(self.num_queries, ratios, special_anchor=True)
        else:
            self.query_embed = nn.Embedding(num_queries, 2)
            self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.query_embed, LayerAnchor):
            self.query_embed.reset_parameters()
        else:
            self._initialize_query_embed(self.init_method)
            self.query_embed.weight.data[:, :1] = inverse_sigmoid(self.query_embed.weight.data[:, :1])

    def init_special_ref_points(self):
        if isinstance(self.query_embed, LayerAnchor):
            return
        # last ref point is for the full text (center: 0.5, width: 1)
        special_cxw = torch.tensor([[0.5, 1]], device=self.query_embed.weight.device)
        self.query_embed.weight.data[-1:, :] = inverse_sigmoid(special_cxw)

    def _initialize_query_embed(self, method):
        if method == "default":
            self.query_embed.weight.data[:, :1].uniform_(0, 1)
        elif method == "second_half":
            self.query_embed.weight.data[:, :1].uniform_(0.5, 1)
        elif method == "custom":
            self._apply_initialization_map()
        else:
            raise ValueError(f"Unsupported method '{method}'. Use 'default', 'second_half' or 'custom'.")

    def _apply_initialization_map(self):
        for query_index, init_values in self.init_map.items():
            if query_index < 0 or query_index >= self.query_embed.weight.size(0):
                raise IndexError(
                    f"Query index {query_index} is out of bounds for num_queries {self.query_embed.weight.size(0)}"
                )

            if isinstance(init_values, list):
                # (low, high) range init
                self.query_embed.weight.data[query_index].uniform_(*init_values)
            elif isinstance(init_values, float):
                # Fixed value init
                self.query_embed.weight.data[query_index].fill_(init_values)
            elif callable(init_values):
                # Custom init function
                init_values(self.query_embed.weight.data[query_index])
            else:
                raise ValueError(f"Unsupported initialization value for query {query_index}: {init_values}")

    def get_query_embed(self):
        if isinstance(self.query_embed, LayerAnchor):
            return self.query_embed.get_reference_points()
        else:
            return self.query_embed.weight

    def get_refpoint_for_detector(self, batch_size):
        if isinstance(self.query_embed, LayerAnchor):
            refpoint_embed = self.query_embed.get_reference_points()
        else:
            refpoint_embed = self.query_embed.weight
        refpoint_embed = refpoint_embed.unsqueeze(1).repeat(1, batch_size, 1)
        target = torch.zeros(self.num_queries, batch_size, self.d_model).to(refpoint_embed.device)
        return refpoint_embed, target
