from typing import Optional, Tuple
from loguru import logger

from gigacheck.model.src.interval_detector.config import DetrModelConfig
from gigacheck.model.src.interval_detector.interval_detector import IntervalDETR, IntervalDETRConfig
from gigacheck.model.src.interval_detector.losses import SetCriterion
from gigacheck.model.src.interval_detector.modules.matcher import HungarianMatcher
from gigacheck.model.src.interval_detector.modules.transformer import DETRTransformer, DETRTransformerConfig
from gigacheck.model.src.interval_detector.modules.position_encoding import PositionEmbeddingSine


def build_matcher(config: DetrModelConfig, max_seq_len: int):
    return HungarianMatcher(
        cost_span=config.set_cost_span,
        cost_giou=config.set_cost_giou,
        cost_class=config.set_cost_class,
        cost_iou=config.set_cost_iou,
        cost_reference=config.set_cost_reference,
        span_loss_type=config.span_loss_type,
        max_s_l=max_seq_len,
        use_focal=config.use_focal_loss,
    )


def build_loss(max_seq_len: int, matcher: HungarianMatcher, config: DetrModelConfig) -> SetCriterion:
    if config.losses:
        losses = list(config.losses)
    else:
        losses = ["spans", "span_labels"]
        if config.dn_detr:
            losses.append("denoise")

    # NOTE: if there is no specific key in the weight_dict => loss with such name will be skipped
    weight_dict = {
        "loss_span": config.span_loss_coef if "spans" in losses else 0,
        "loss_giou": config.giou_loss_coef if "spans" in losses else 0,
        "loss_label": config.label_loss_coef if "span_labels" in losses else 0,
        # dn detr
        "tgt_loss_span": config.tgt_loss_span_coef if "denoise" in losses else 0,
        "tgt_loss_giou": config.tgt_loss_giou_coef if "denoise" in losses else 0,
        "tgt_loss_label": config.tgt_label_loss_coef if "denoise" in losses else 0,
    }

    # Choose losses
    if config.aux_loss:
        aux_weight_dict = {}
        for i in range(config.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    for k, v in list(weight_dict.items())[:]:
        if v == 0:
            # loss will be skipped from the final sum(losses)
            del weight_dict[k]

    # Build losses
    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        eos_coef=config.eos_coef,
        span_loss_type=config.span_loss_type,
        max_s_l=max_seq_len,
        use_focal_loss=config.use_focal_loss,
    )
    return criterion


def build_detr_model(
    config: DetrModelConfig,
    hidden_size: int,
    max_seq_len: int,
    with_loss=True,
) -> Tuple[IntervalDETR, Optional[SetCriterion]]:

    if not config.dab_detr:
        raise NotImplementedError("Not implemented without dab")

    transformer_config = DETRTransformerConfig(
        hidden_size=config.model_dim,
        num_decoder_layers=config.dec_layers,
        num_encoder_layers=config.enc_layers,
        nhead=config.nheads,
        return_intermediate_dec=True,
        num_queries=config.num_queries,
        num_groups=config.num_groups,
        center_noise_scale=config.center_noise_scale,
        width_noise_scale=config.width_noise_scale,
        noise_offset=config.noise_offset,
        dn_detr=config.dn_detr,
        temperature=config.temperature,
        query_initialization_method=config.query_initialization_method,
    )
    transformer = DETRTransformer(transformer_config)

    position_embedding = None
    if config.position_embedding == "sine":
        # use MistralRotaryEmbedding for Q, K rotation instead of Absolute ?
        position_embedding = PositionEmbeddingSine(config.model_dim, normalize=True)

    # Build final DETR model
    model_config = IntervalDETRConfig(
        input_dim=hidden_size,
        model_dim=config.model_dim,
        num_input_proj_layers=config.num_input_proj_layers,
        proj_dropout=config.proj_dropout,
        aux_loss=config.aux_loss,
        max_s_l=max_seq_len,
        span_loss_type=config.span_loss_type,
        n_input_proj=config.n_input_proj,
        dn_detr=config.dn_detr,
        use_focal_loss=config.use_focal_loss,
    )
    model = IntervalDETR(
        config=model_config,
        transformer=transformer,
        position_embed=position_embedding,
    )
    logger.info("IntervalDETR building model complete")

    if not with_loss:
        return model, None

    matcher: HungarianMatcher = build_matcher(config, max_seq_len)
    criterion: SetCriterion = build_loss(max_seq_len, matcher, config)
    return model, criterion
