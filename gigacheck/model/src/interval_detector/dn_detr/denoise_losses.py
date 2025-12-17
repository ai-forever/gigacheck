"""Module for computing losses."""

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gigacheck.model.src.interval_detector.focal_loss import sigmoid_focal_loss
from gigacheck.model.src.interval_detector.span_utils import generalized_temporal_iou, span_cxw_to_xx


class DenoiseLosses(nn.Module):
    """Compute the Denoise losses for DETR."""

    def __init__(self, *args, **kwargs):
        self.use_focal_loss = kwargs.pop("use_focal_loss", False)
        self.focal_alpha = 0.25
        self.focal_gamma = 2

        super().__init__(*args, **kwargs)

    def loss_spans(self, src_spans: torch.Tensor, tgt_spans: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the losses related to the noised spans (the L1 regression loss and the GIoU loss).

        Args:
            src_spans (torch.Tensor): predicted spans
            tgt_spans (torch.Tensor): target spans

        Returns:
            Dict[str, torch.Tensor]: A dict containing the L1 regression and gIoU losses.
        """
        loss_span = F.l1_loss(src_spans, tgt_spans, reduction="none")
        loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))

        losses = {"tgt_loss_span": loss_span.mean(), "tgt_loss_giou": loss_giou.mean()}
        return losses

    def loss_labels(self, src_logits: Tensor, tgt_labels: Tensor, num_tgt: int) -> Dict[str, Tensor]:
        """Classification loss.

        Args:
            src_logits (Tensor): predicted logits
            tgt_labels (Tensor): target labels

        Returns:
            Dict[str, torch.Tensor]: A dict containing the classification loss and the classification error.q
        """
        src_logits, tgt_labels = src_logits.unsqueeze(0), tgt_labels.unsqueeze(0)

        if self.use_focal_loss:
            target_classes_onehot = torch.zeros(
                [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]],
                dtype=src_logits.dtype,
                layout=src_logits.layout,
                device=src_logits.device,
            )
            target_classes_onehot.scatter_(2, tgt_labels.unsqueeze(-1), 1)

            loss_ce = (
                sigmoid_focal_loss(
                    src_logits, target_classes_onehot, num_tgt, alpha=self.focal_alpha, gamma=self.focal_gamma
                )
                * src_logits.shape[1]
            )
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), tgt_labels, reduction="none").mean()

        return {"tgt_loss_label": loss_ce}

    @staticmethod
    def prepare_for_loss(mask_dict: Dict[str, Any]) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
        """
        Prepare dn components to calculate loss.

        Args:
            mask_dict: a dict that contains dn information

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, int]: Prepared components.
        """
        # (#layers, bs, pad_size, 2), (#layers, bs, pad_size, 2)
        output_known_class, output_known_coord = mask_dict["output_known_lbs_bboxes"]
        known_labels, known_bboxes = mask_dict["known_lbs_bboxes"]
        map_known_indices = mask_dict["map_known_indices"]
        known_indices = mask_dict["known_indices"]
        batch_idx = mask_dict["batch_idx"]
        bid = batch_idx[known_indices]

        if len(output_known_class) > 0:
            output_known_class = output_known_class.permute(1, 2, 0, 3)
            output_known_class = output_known_class[(bid, map_known_indices)].permute(1, 0, 2)
            output_known_coord = output_known_coord.permute(1, 2, 0, 3)
            output_known_coord = output_known_coord[(bid, map_known_indices)].permute(1, 0, 2)

        num_tgt = known_indices.numel()  # #all_gt_bboxes * num_groups
        return known_labels, known_bboxes, output_known_class, output_known_coord, num_tgt

    def forward(self, mask_dict: Dict[str, Any], aux_num: int) -> Dict[str, Tensor]:
        """
        Compute dn loss in criterion.

        Args:
            mask_dict (Dict[str, Any]): a dict for dn information
            aux_num (int): aux loss number

        Returns:
            Dict[str, Tensor]: computed losses.
        """
        losses = {}
        known_labels, known_bboxes, output_known_class, output_known_coord, num_tgt = self.prepare_for_loss(mask_dict)
        if num_tgt > 0:
            losses.update(self.loss_labels(output_known_class[aux_num], known_labels, num_tgt))
            losses.update(self.loss_spans(output_known_coord[aux_num], known_bboxes))
        return losses
