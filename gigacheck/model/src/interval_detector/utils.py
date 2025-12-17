from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch

from gigacheck.model.src.interval_detector.interval_detector import IntervalDETR
from gigacheck.model.src.interval_detector.modules.matcher import HungarianMatcher
from gigacheck.model.src.interval_detector.span_utils import span_cxw_to_xx


@dataclass
class TextPredictionWithGt:
    gt_spans: torch.Tensor  # xx (#gt_spans, 2)
    pred_probs: torch.Tensor  # (#queries,)
    out_spans: torch.Tensor  # xx (#queries, 2)
    matching_pred_indxs: torch.Tensor  # (#gt_spans, )
    matching_tgt_indxs: torch.Tensor  # (#gt_spans, )


class PredictionsWithGt:
    def __init__(
        self,
        target_spans: List[dict],
        pred: Tuple[torch.Tensor, ...],
        matcher: HungarianMatcher,
        ref_points: Optional[torch.Tensor],
    ):
        # pred_spans: predicted spans (batch, #queries, 2) in format cxw
        # src_logits: predicted logits for each of #spans (batch, #queries, 2)
        src_logits, pred_spans = pred

        # Retrieve the matching between the outputs of the last layer and the targets
        # each tuple is (pred_span_indices, tgt_span_indices)
        self.indices: list = matcher.forward(
            outputs={
                "pred_spans": pred_spans,
                "pred_logits": src_logits,
            },
            targets={"span_labels": target_spans},
            ref_points=ref_points,
        )

        self.predictions: List[TextPredictionWithGt] = []
        batch_size = len(target_spans)
        for batch_idx in range(batch_size):
            gt_spans = span_cxw_to_xx(target_spans[batch_idx]["spans"])
            # get scores of foreground label
            pred_probs = torch.softmax(src_logits[batch_idx], -1)[:, 0]
            out_spans = span_cxw_to_xx(pred_spans[batch_idx])
            matching_pred_indxs = self.indices[batch_idx][0]
            matching_tgt_indxs = self.indices[batch_idx][1]
            self.predictions.append(
                TextPredictionWithGt(
                    gt_spans=gt_spans,
                    pred_probs=pred_probs,
                    out_spans=out_spans,
                    matching_pred_indxs=matching_pred_indxs,
                    matching_tgt_indxs=matching_tgt_indxs,
                )
            )

    def __iter__(self):
        for prediction in self.predictions:
            yield prediction


@dataclass
class TextPrediction:
    pred_probs: torch.Tensor  # (#queries,)
    out_spans: torch.Tensor  # xx (#queries, 2)


class Predictions:
    def __init__(
        self,
        pred_spans: torch.Tensor,
        src_logits: torch.Tensor,
    ):
        """
        :param pred_spans: predicted spans (batch, #queries, 2)
        :param src_logits: predicted logits for each of #spans (batch, #queries, 2)
        """
        self.predictions: List[TextPrediction] = []
        batch_size = len(pred_spans)
        for batch_idx in range(batch_size):
            # get score of foreground label
            pred_probs = torch.softmax(src_logits[batch_idx], -1)[:, 0]
            out_spans = span_cxw_to_xx(pred_spans[batch_idx])

            self.predictions.append(
                TextPrediction(
                    pred_probs=pred_probs,
                    out_spans=out_spans,
                )
            )


def get_ref_points(detr_model: IntervalDETR, return_type="np") -> Optional[np.ndarray]:
    """

    :param detr_model: IntervalDETR model
    :param return_type: "np" or "pt"
    :return: points in format (start, end)
    """
    ref_points = detr_model.transformer.query_embed.get_query_embed()

    if ref_points.shape[-1] != 2:
        return None

    if return_type == "np":
        anchors = span_cxw_to_xx(ref_points.sigmoid().detach())
        anchors = anchors.to(torch.float32)
        return anchors.cpu().numpy()
    else:
        return span_cxw_to_xx(ref_points.sigmoid())


def calculate_iou_1d(span_a, span_b):
    """
    Calculate the Intersection over Union (IoU) of two spans.

    Parameters:
        span_a (list, tuple): Start and end of the first span.
        span_b (list, tuple): Start and end of the second span.

    Returns:
        float: The IoU between span_a and span_b.
    """

    start_a, end_a = span_a
    start_b, end_b = span_b

    if isinstance(span_a, torch.Tensor) and isinstance(span_b, torch.Tensor):
        start_a, end_a = start_a.item(), end_a.item()
        start_b, end_b = start_b.item(), end_b.item()

    # Find the intersection of the spans
    intersection_start = max(start_a, start_b)
    intersection_end = min(end_a, end_b)
    intersection = max(intersection_end - intersection_start, 0)

    # Find the union of the spans
    union = max(end_a, end_b) - min(start_a, start_b)

    # Calculate the IoU
    iou = intersection / union if union > 0 else 0

    return iou


def general_nms(
    items: torch.Tensor,
    score_function: Callable[[Any], float],
    iou_function: Callable[[Any, Any], float],
    threshold: float,
) -> np.ndarray:
    """
    Apply Non-Maximum Suppression (NMS) to a tensor.

    :param items: Tensor with items to apply NMS.
    :param score_function: Function to compute the score for each item.
    :param iou_function: Function to compute IoU (Intersection over Union) between two items.
    :param threshold: IoU threshold to consider two items as overlapping.
    :return: Indices of items selected by NMS.
    """

    # Compute scores for each item
    scores = np.array([score_function(item) for item in items])

    # Sort items by their scores in descending order
    sorted_indices = np.argsort(-scores)

    # List to keep track of selected indices
    selected_indices = []

    while sorted_indices.size > 0:
        # Always select the item with the highest score
        current_index = sorted_indices[0]
        selected_indices.append(current_index)

        # Compute IoU between the selected item and the rest; keep those with IoU less than threshold
        remaining_indices = sorted_indices[1:]
        remaining_indices = [
            idx
            for idx in remaining_indices
            if iou_function(items[current_index], items[idx]) < threshold  # type: ignore
        ]

        # Update the indices for the next iteration
        sorted_indices = np.array(remaining_indices)

    return np.array(selected_indices)
