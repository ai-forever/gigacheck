from dataclasses import asdict, dataclass, field, fields
from typing import List
import copy

import torch
from dacite import from_dict
from omegaconf import DictConfig, OmegaConf
from loguru import logger


@dataclass
class DetrModelConfig:
    extractor_dtype: str = field(default="float32")
    model_dim: int = field(default=512)
    num_input_proj_layers: int = field(default=2)
    proj_dropout: float = field(default=0.5)

    # loss
    use_focal_loss: bool = field(default=False)
    span_loss_coef: float = field(default=10.0)
    giou_loss_coef: float = field(default=1.0)
    label_loss_coef: float = field(default=4.0)

    eos_coef: float = field(default=0.1)
    temperature: int = field(default=10000)
    span_loss_type: str = field(default="l1")
    n_input_proj: int = field(default=2)
    aux_loss: bool = field(default=False)
    num_queries: int = field(default=35)
    dec_layers: int = field(default=2)
    enc_layers: int = field(default=2)
    nheads: int = field(default=8)
    position_embedding: str = field(default="sine")
    query_initialization_method: str = field(default="default")

    dab_detr: bool = field(default=True)
    # dn detr
    dn_detr: bool = field(default=False)
    num_groups: int = field(default=5)
    center_noise_scale: float = field(default=0.6)
    width_noise_scale: float = field(default=0.7)
    noise_offset: float = field(default=0.05)

    # dn detr loss
    tgt_label_loss_coef: float = field(default=0.0)
    tgt_loss_span_coef: float = field(default=9.0)
    tgt_loss_giou_coef: float = field(default=3.0)

    # matcher
    set_cost_span: int = field(default=5)
    set_cost_giou: int = field(default=1)
    set_cost_class: int = field(default=2)
    set_cost_iou: int = field(default=2)
    set_cost_reference: int = field(default=1)

    losses: List[str] = field(default_factory=list)
    special_ref_points: bool = field(default=False)

    def to_dict(self):
        return {k: v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d):
        data: dict = OmegaConf.to_container(d) if isinstance(d, DictConfig) else d
        fields_names = [f.name for f in fields(cls)]

        for k in copy.copy(data):
            if k not in fields_names:
                data.pop(k)
        logger.info(f"DETR Model config: {data}")
        return from_dict(data_class=cls, data=data)
