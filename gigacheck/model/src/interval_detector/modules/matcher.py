# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from gigacheck.model.src.interval_detector.span_utils import generalized_temporal_iou, span_cxw_to_xx
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_span: float = 1,
        cost_giou: float = 1,
        cost_iou: float = 1,
        cost_reference: float = 1,
        span_loss_type: str = "l1",
        max_s_l: int = 1024,
        use_focal: bool = False,
    ):
        """Creates the matcher

        Params:
            cost_class (float): Weight of the classification error in the matching cost
            cost_span: This is the relative weight of the L1 error of the span coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the spans in the matching cost
            cost_iou (float): Weight of the iou error in the matching cost
            cost_reference (float): Weight of the reference distance cost.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.cost_giou = cost_giou
        self.cost_iou = cost_iou
        self.cost_reference = cost_reference

        self.span_loss_type = span_loss_type
        self.max_s_l = max_s_l
        self.foreground_label = 0
        assert cost_class != 0 or cost_span != 0 or cost_giou != 0 or cost_reference != 0, "all costs cant be 0"

        self.use_focal = use_focal
        self.focal_alpha = 0.25
        self.focal_gamma = 2

    @torch.no_grad()
    def forward(self, outputs, targets, ref_points):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_spans": Tensor of dim [batch_size, num_queries, 2] with the predicted span coordinates,
                    in normalized (cx, w) format
                 ""pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "spans": Tensor of dim [num_target_spans, 2] containing the target span coordinates. The spans are
                    in normalized (cx, w) format

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_spans)
        """

        bs, num_queries = outputs["pred_spans"].shape[:2]
        targets = targets["span_labels"]

        # Also concat the target labels and spans
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        tgt_spans = torch.cat([v["spans"] for v in targets])  # [num_target_spans in batch, 2]
        tgt_spans = tgt_spans.to(out_prob.device)
        tgt_ids = torch.full([len(tgt_spans)], self.foreground_label)  # [total #spans in the batch]

        # Compute the classification cost.
        if self.use_focal:
            alpha = self.focal_alpha
            gamma = self.focal_gamma
            neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        else:
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - prob[target class].
            # The 1 is a constant that doesn't change the matching, it can be omitted.
            cost_class = -out_prob[:, tgt_ids]  # [batch_size * num_queries, total #spans in the batch]

        if self.span_loss_type == "l1":
            # We flatten to compute the cost matrices in a batch
            out_spans = outputs["pred_spans"].flatten(0, 1)  # [batch_size * num_queries, 2]

            # Compute the L1 cost between spans
            cost_span = torch.cdist(out_spans, tgt_spans, p=1)  # [batch_size * num_queries, total #spans in the batch]

            # Compute the giou cost between spans
            out_spans = span_cxw_to_xx(out_spans)
            tgt_spans = span_cxw_to_xx(tgt_spans)

            # [batch_size * num_queries, total #spans in the batch]
            cost_giou = -generalized_temporal_iou(out_spans, tgt_spans)
        else:
            pred_spans = outputs["pred_spans"]  # (bsz, #queries, max_s_l * 2)
            pred_spans = pred_spans.view(bs * num_queries, 2, self.max_s_l).softmax(-1)  # (bsz * #queries, 2, max_s_l)
            cost_span = (
                -pred_spans[:, 0][:, tgt_spans[:, 0]] - pred_spans[:, 1][:, tgt_spans[:, 1]]
            )  # (bsz * #queries, #spans)
            # giou
            cost_giou = 0

        # We flatten to compute the cost matrices in a batch
        if ref_points is not None:
            pred_diffs = torch.abs(
                span_cxw_to_xx(outputs["pred_spans"]).detach() - ref_points.to(outputs["pred_spans"].device)
            )
            pred_diffs = torch.sqrt(pred_diffs[:, :, 0] ** 2 + pred_diffs[:, :, 1] ** 2)  # noqa: WPS221
            pred_diffs = pred_diffs.flatten()
            cost_reference = pred_diffs.unsqueeze(1).repeat(1, len(tgt_spans))
        else:
            cost_reference = torch.zeros_like(outputs["pred_spans"][..., 0].flatten())
            cost_reference = cost_reference.unsqueeze(1).repeat(1, len(tgt_spans))

        cost_matrix = self.cost_span * cost_span
        cost_matrix += self.cost_giou * cost_giou
        cost_matrix += self.cost_class * cost_class
        cost_matrix += self.cost_reference * cost_reference

        cost_matrix = cost_matrix.view(bs, num_queries, -1).cpu()
        # targets[0], torch.softmax(outputs['pred_logits'][0][21], -1), outputs['pred_spans'][0][21]
        sizes = [len(v["spans"]) for v in targets]

        # selected pred, selected target
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
