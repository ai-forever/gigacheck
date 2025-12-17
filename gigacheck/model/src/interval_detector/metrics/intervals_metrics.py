from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from gigacheck.model.src.interval_detector.utils import calculate_iou_1d, general_nms


class AveragePrecision:

    def __init__(
        self,
        with_nms=False,
    ):
        self.iou_thresholds = torch.linspace(0.5, 0.95, 10)
        self.avg_precision: List[torch.Tensor] = []
        self.with_nms = with_nms
        self.nms_iou_thresh = 0.5

    def update(self, submissions: List[dict], targets: List[dict], text_len: int = None):
        if self.with_nms:
            submissions = apply_nms(submissions, iou_threshold=self.nms_iou_thresh)

        gt_qid2data, pred_qid2data = prepare_data(
            submissions,
            targets,
            len_range=(0, float("+inf")),
            text_len=text_len,
        )

        ap_per_text: Dict[int, np.ndarray] = {}
        for tid in pred_qid2data:
            ground_truth, prediction = gt_qid2data[tid], pred_qid2data[tid]
            scores = compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=self.iou_thresholds)
            ap_per_text[tid] = scores

        if len(ap_per_text):
            ap_array = np.array(list(ap_per_text.values()))  # (#queries, #thd)
            self.avg_precision.append(torch.tensor(ap_array))

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute the average precision per IoU threshold.

        Returns:
            Dict[str, Tensor]: The Average Precision Score computed per each IoU threshold.
        """
        ap_per_query_per_iou = (
            torch.cat(self.avg_precision, dim=0)
            if len(self.avg_precision)
            else torch.empty((0, len(self.iou_thresholds)))
        )
        if ap_per_query_per_iou.nelement() == 0:
            ap_per_query_per_iou = torch.zeros(1, len(self.iou_thresholds))
        ap_per_iou = ap_per_query_per_iou.mean(0)  # mAP at different IoU thresholds.
        str_ious = [str(float(f"{iou:.2f}")) for iou in self.iou_thresholds]
        str_ious[str_ious.index("0.5")] = "mAP@0.5"
        iou_thd2ap = dict(zip(str_ious, ap_per_iou))
        name = f"mAP@{str_ious[0]}-{str_ious[-1]}" if "mAP@" not in str_ious[0] else f"{str_ious[0]}-{str_ious[-1]}"
        iou_thd2ap[name] = torch.mean(ap_per_iou)
        return iou_thd2ap


def compute_average_precision_detection(
    ground_truth: List[dict],
    prediction: List[dict],
    tiou_thresholds,
) -> np.ndarray:
    """Compute average precision (detection task) between ground truth and predictions.

    If multiple predictions occurs for the same predicted segment, only the one with highest score is matches as true
    positive. This code is greatly inspired by Pascal VOC devkit.

    Args:
        ground_truth (List[dict]): List containing the ground truth instances dict['text-id', 't-start', 't-end']
        prediction (List[dict]): List containing the prediction instances dict['text-id', 't-start', 't-end', 'score']
        tiou_thresholds: indicates the temporal intersection over union threshold, which is optional.

    Returns:
        np.ndarray: ap, Average precision score for each IoU threshold.
    """
    num_thresholds = len(tiou_thresholds)
    num_gts = len(ground_truth)
    num_preds = len(prediction)
    ap = np.zeros(num_thresholds)
    if len(prediction) == 0:
        return ap

    num_positive = float(num_gts)
    lock_gt = np.ones((num_thresholds, num_gts)) * -1
    # Sort predictions by decreasing score order.
    prediction.sort(key=lambda x: -x["score"])
    # Initialize true positive and false positive vectors.
    tp = np.zeros((num_thresholds, num_preds))
    fp = np.zeros((num_thresholds, num_preds))

    # Adaptation to query faster
    ground_truth_by_textid: dict = {}
    for i, item in enumerate(ground_truth):
        item["index"] = i
        ground_truth_by_textid.setdefault(item["text_id"], []).append(item)

    # Assigning true positive to truly grount truth instances.
    for idx, pred in enumerate(prediction):
        if pred["text_id"] in ground_truth_by_textid:
            gts = ground_truth_by_textid[pred["text_id"]]
        else:
            fp[:, idx] = 1
            continue

        _pred = np.array([[pred["t-start"], pred["t-end"]]])
        _gt = np.array([[gt["t-start"], gt["t-end"]] for gt in gts])
        tiou_arr = compute_temporal_iou_batch_cross(_pred, _gt)[0]

        tiou_arr = tiou_arr.reshape(-1)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for t_idx, tiou_threshold in enumerate(tiou_thresholds):
            for j_idx in tiou_sorted_idx:
                if tiou_arr[j_idx] < tiou_threshold:
                    fp[t_idx, idx] = 1
                    break
                if lock_gt[t_idx, gts[j_idx]["index"]] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[t_idx, idx] = 1
                lock_gt[t_idx, gts[j_idx]["index"]] = idx
                break

            if fp[t_idx, idx] == 0 and tp[t_idx, idx] == 0:
                fp[t_idx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum = tp_cumsum / num_positive
    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for t_idx, _ in enumerate(tiou_thresholds):
        ap[t_idx] = interpolated_precision_recall(precision_cumsum[t_idx, :], recall_cumsum[t_idx, :])
    return ap


def interpolated_precision_recall(precision: np.ndarray, recall: np.ndarray) -> float:
    """Interpolated AP - VOCdevkit from VOC 2011.

    Args:
        precision (np.ndarray): The precision of different thresholds.
        recall (np.ndarray): The recall of different thresholds.

    Returns:
        float: Average precision score.
    """
    mprecision = np.hstack([[0], precision, [0]])
    mrecall = np.hstack([[0], recall, [1]])
    for i in range(len(mprecision) - 1)[::-1]:
        mprecision[i] = max(mprecision[i], mprecision[i + 1])
    idx = np.where(mrecall[1::] != mrecall[0:-1])[0] + 1
    return np.sum((mrecall[idx] - mrecall[idx - 1]) * mprecision[idx])


def compute_temporal_iou_batch_cross(spans1: np.ndarray, spans2: np.ndarray):
    """
    Compute temporal intersection over union between spans1 and spans2.

    Args:
        spans1 (np.ndarray): each row defines a span [st, ed] (N, 2)
        spans2 (np.ndarray): each row defines a span [st, ed] (M, 2)

    Returns:
        iou: (N, M) np.ndarray
        union: (N, M) np.ndarray
    """
    areas1 = spans1[:, 1] - spans1[:, 0]  # (N, )
    areas2 = spans2[:, 1] - spans2[:, 0]  # (M, )

    left = np.maximum(spans1[:, None, 0], spans2[None, :, 0])  # (N, M)
    right = np.minimum(spans1[:, None, 1], spans2[None, :, 1])  # (N, M)

    inter = np.clip(right - left, 0, None)  # (N, M)
    union = areas1[:, None] + areas2[None, :] - inter  # (N, M)

    iou = inter / union
    return iou, union


def check_interval_len(
    window: List[int],
    min_l: int,
    max_l: int,
    text_len: int = None,
) -> int:
    if text_len and isinstance(min_l, float) and isinstance(max_l, float):
        min_l *= text_len
        max_l *= text_len
    return min_l < (window[1] - window[0]) <= max_l


def apply_nms(predictions: List[dict], iou_threshold: float = 0.5) -> List[dict]:
    # since the data is transmitted by link, a full copy of it is made
    nms_predictions = []

    # NMS applies to all predictions
    for item in predictions:
        pred: torch.Tensor = item["pred_relevant_windows"]
        idxes = general_nms(
            pred,
            score_function=lambda x: x[2],  # prob
            iou_function=lambda x, y: calculate_iou_1d(x[:2], y[:2]),
            threshold=iou_threshold,
        )
        item["pred_relevant_windows"] = pred[idxes]
        nms_predictions.append(item)
    return nms_predictions


def prepare_data(
    submissions: List[dict],
    targets: List[dict],
    len_range: Tuple[int, int],
    text_len: int,
) -> Tuple[Dict[int, List[dict]], Dict[int, List[dict]]]:
    """Keep queries with ground truth window length in the specified length range.

    Args:
        submissions (List[dict]): submissions to be filtered
        targets (List[dict]): Targets to be filtered
        len_range (Tuple[int, int]): Target window length range

    Returns:
        filtered submissions and ground truth
    """

    def format_span(text_id: int, window) -> dict:
        data = {
            "text_id": text_id,
            "t-start": float(window[0]),
            "t-end": float(window[1]),
        }
        if len(window) > 2:
            data["score"] = window[2]  # take prob of zero class
        return data

    gt_qid2data = defaultdict(list)
    target_qids_in_range = set()
    for target in targets:
        rel_windows_in_range = [
            window
            for window in target["relevant_windows"]
            if check_interval_len(window, *len_range, text_len)
        ]
        tid = target["text_id"]
        if rel_windows_in_range:
            for window in rel_windows_in_range:
                gt_qid2data[tid].append(format_span(tid, window))
            target_qids_in_range.add(tid)

    # keep only submissions for ground_truth_in_range
    pred_qid2data = defaultdict(list)
    for submission in submissions:
        tid = submission["text_id"]
        if tid in target_qids_in_range:
            for window in submission["pred_relevant_windows"]:
                pred_qid2data[tid].append(format_span(tid, window))

    assert len(gt_qid2data) == len(pred_qid2data), f"{len(gt_qid2data)} != {len(pred_qid2data)}"
    return gt_qid2data, pred_qid2data
