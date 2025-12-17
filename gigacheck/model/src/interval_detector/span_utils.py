import math
from typing import Any, List, Tuple

import torch
from torch import Tensor


def span_xx_to_cxw(xx_spans):
    """
    Args:
        xx_spans: tensor, (#windows, 2) or (..., 2), each row is a window of format (st, ed)

    Returns:
        cxw_spans: tensor, (#windows, 2), each row is a window of format (center=(st+ed)/2, width=(ed-st))
    >>> spans = torch.Tensor([[0, 1], [0.2, 0.4]])
    >>> span_xx_to_cxw(spans)
    tensor([[0.5000, 1.0000],
        [0.3000, 0.2000]])
    >>> spans = torch.Tensor([[[0, 1], [0.2, 0.4]]])
    >>> span_xx_to_cxw(spans)
    tensor([[[0.5000, 1.0000],
         [0.3000, 0.2000]]])
    """
    center = xx_spans.sum(-1) * 0.5
    width = xx_spans[..., 1] - xx_spans[..., 0]
    return torch.stack([center, width], dim=-1)


def span_cxw_to_xx(cxw_spans):
    """
    Args:
        cxw_spans: tensor, (#windows, 2) or (..., 2), the last dim is a row denoting a window of format (center, width)

    >>> spans = torch.Tensor([[0.5000, 1.0000], [0.3000, 0.2000]])
    >>> span_cxw_to_xx(spans)
    tensor([[0.0000, 1.0000],
        [0.2000, 0.4000]])
    >>> spans = torch.Tensor([[[0.5000, 1.0000], [0.3000, 0.2000]]])
    >>> span_cxw_to_xx(spans)
    tensor([[[0.0000, 1.0000],
        [0.2000, 0.4000]]])
    """
    x1 = cxw_spans[..., 0] - 0.5 * cxw_spans[..., 1]
    x2 = cxw_spans[..., 0] + 0.5 * cxw_spans[..., 1]
    return torch.stack([x1, x2], dim=-1)


def temporal_iou(spans1, spans2):
    """
    Args:
        spans1: (N, 2) torch.Tensor, each row defines a span [st, ed]
        spans2: (M, 2) torch.Tensor, ...

    Returns:
        iou: (N, M) torch.Tensor
        union: (N, M) torch.Tensor
    >>> test_spans1 = torch.Tensor([[0, 0.2], [0.5, 1.0]])
    >>> test_spans2 = torch.Tensor([[0, 0.3], [0., 1.0]])
    >>> temporal_iou(test_spans1, test_spans2)
    (tensor([[0.6667, 0.2000],
         [0.0000, 0.5000]]),
     tensor([[0.3000, 1.0000],
             [0.8000, 1.0000]]))
    """
    areas1 = spans1[:, 1] - spans1[:, 0]  # (N, )
    areas2 = spans2[:, 1] - spans2[:, 0]  # (M, )

    left = torch.max(spans1[:, None, 0], spans2[:, 0])  # (N, M)
    right = torch.min(spans1[:, None, 1], spans2[:, 1])  # (N, M)

    inter = (right - left).clamp(min=0)  # (N, M)
    union = areas1[:, None] + areas2 - inter  # (N, M)

    iou = inter / union
    return iou, union


def temporal_intersection_over_pred(gt_spans, pred_spans):
    """intersection over the second input spans
    Args:
        gt_spans: (N, 2),
        pred_spans: (M, 2)

    Returns:

    """
    left = torch.max(gt_spans[:, None, 0], pred_spans[:, 0])
    right = torch.min(gt_spans[:, None, 1], pred_spans[:, 1])

    inter = (right - left).clamp(min=0)  # (N, M)
    inter_over_pred = inter / (pred_spans[:, 1] - pred_spans[:, 0])
    return inter_over_pred


def generalized_temporal_iou(spans1, spans2):
    """
    Generalized IoU from giou.stanford.edu/
    Also reference to DETR implementation of generalized_box_iou
    github.com/facebookresearch/detr/blob/master/util/box_ops.py#L40

    Args:
        spans1: (N, 2) torch.Tensor, each row defines a span in xx format [st, ed]
        spans2: (M, 2) torch.Tensor, ...

    Returns:
        giou: (N, M) torch.Tensor

    >>> test_spans1 = torch.Tensor([[0, 0.2], [0.5, 1.0]])
    >>> test_spans2 = torch.Tensor([[0, 0.3], [0., 1.0]])
    >>> generalized_temporal_iou(test_spans1, test_spans2)
    tensor([[ 0.6667,  0.2000],
        [-0.2000,  0.5000]])
    """
    spans1 = spans1.float()
    spans2 = spans2.float()
    assert (spans1[:, 1] >= spans1[:, 0]).all()
    assert (spans2[:, 1] >= spans2[:, 0]).all()
    iou, union = temporal_iou(spans1, spans2)

    left = torch.min(spans1[:, None, 0], spans2[:, 0])  # (N, M)
    right = torch.max(spans1[:, None, 1], spans2[:, 1])  # (N, M)
    enclosing_area = (right - left).clamp(min=0)  # (N, M)

    return iou - (enclosing_area - union) / enclosing_area


class SpanList:
    """
    This class represents a set of spans.

    The spans are represented as a Nx2 Tensor.
    In order to uniquely determine the spans with respect to an text, we also store the corresponding text sizes.
    """

    def __init__(self, spans: Tensor, size: int, mode="xx"):
        """Initialize the SpanList.

        Args:
            spans (Tensor): A Nx2 Tensor representing the spans.
            size (int): The size of the text.
            mode (str): Format of the span. Defaults to "xx".

        Raises:
            ValueError: If the mode is not "xx" or "cxw".
        """
        if mode not in {"xx", "cxw"}:
            raise ValueError("mode should be 'xx' or 'cxw'")

        self.spans = spans
        self.size = size
        self.mode = mode
        self.extra_fields: dict = {}

    def add_field(self, field: str, field_data: Any):
        """Add a field to the SpanList.

        Args:
            field (str): The name of the field.
            field_data (Any): The data to be stored in the field.
        """
        self.extra_fields[field] = field_data

    def get_field(self, field: str) -> Any:
        """Get the data stored in a field.

        Args:
            field (str): The name of the field.

        Returns:
            Any: The data stored in the field.
        """
        return self.extra_fields[field]

    def fields(self) -> List[str]:
        """Get the names of the fields stored in the SpanList.

        Returns:
            List[str]: The names of the fields stored in the SpanList.
        """
        return list(self.extra_fields.keys())

    def copy_extra_fields(self, spans: "SpanList") -> None:
        """Copy the extra fields from another SpanList.

        Args:
            spans (SpanList): The SpanList to copy the extra fields from.
        """
        for key, value in spans.extra_fields.items():
            self.extra_fields[key] = value

    def _split_into_xx(self) -> Tuple[Tensor, Tensor]:
        """Split the spans into the format (xmin, xmax).

        Returns:
            Tuple[Tensor, Tensor]: The spans in the format (xmin, xmax).
        """
        if self.mode == "xx":
            xmin, xmax = self.spans.split(1, dim=-1)  # type: ignore
            return xmin, xmax  # noqa: WPS331
        xmin, width = self.spans.split(1, dim=-1)  # type: ignore
        return xmin, xmin + (width - 1).clamp(min=0)

    def convert(self, mode: str) -> "SpanList":
        """Convert the spans to a different format.

        Args:
            mode (str): The format to convert the spans to.

        Returns:
            "SpanList": The spans in the new format.

        Raises:
            ValueError: If the mode is not "xx" or "cxw".
        """
        if mode not in {"xx", "cxw"}:
            raise ValueError("mode should be 'xx' or 'cxw'")
        if mode == self.mode:
            return self

        xmin, xmax = self._split_into_xx()
        if mode == "xx":
            spans = torch.cat((xmin, xmax), dim=-1)
            spans_list = SpanList(spans, self.size, mode=mode)
        else:
            spans = torch.cat((xmin, xmax - xmin + 1), dim=-1)
            spans_list = SpanList(spans, self.size, mode=mode)
        spans_list.copy_extra_fields(self)
        return spans_list

    def __getitem__(self, item: int) -> "SpanList":
        """Get a subset of the spans.

        Args:
            item (int): The index of the span to get.

        Returns:
            SpanList: The subset of spans.
        """
        spans = SpanList(self.spans[item], self.size, self.mode)
        for key, value in self.extra_fields.items():
            spans.add_field(key, value[item])
        return spans

    def __len__(self) -> int:
        """Get the number of spans.

        Returns:
            int: The number of spans.
        """
        return self.spans.shape[0]

    def __repr__(self):
        """Get a string representation of the SpanList.

        Returns:
            str: A string representation of the SpanList.
        """
        string = self.__class__.__name__ + "("  # noqa: WPS336
        string += f"num_boxes={len(self)}, "  # noqa: WPS336,WPS237
        string += f"text_length={self.size}, "  # noqa: WPS336
        string += f"mode={self.mode})"  # noqa: WPS336
        return string


def cat_boxlist(spans: List[SpanList]):
    """
    Concatenate a list of SpanList (having the same embed size) into a single SpanList.

    Args:
        spans (List["SpanList"]): A list of SpanList to concatenate.

    Returns:
        List["SpanList"]: The concatenated SpanList.
    """
    assert isinstance(spans, (list, tuple))
    assert all(isinstance(span, SpanList) for span in spans)

    size = spans[0].size
    assert all(span.size == size for span in spans)

    mode = spans[0].mode
    assert all(span.mode == mode for span in spans)

    fields = set(spans[0].fields())
    assert all(set(span.fields()) == fields for span in spans)

    cat_boxes = SpanList(torch.cat([span.spans for span in spans], dim=0), size, mode)  # noqa: WPS221

    for field in fields:
        data = torch.cat([bbox.get_field(field) for bbox in spans], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes


def encode_spans(gt_boxes: Tensor, anchors: Tensor) -> Tensor:
    """Encode spans into deltas between anchors and ground truth boxes.

    Args:
        gt_boxes (Tensor): Ground truth boxes in the format (start, end).
        anchors (Tensor): Anchors in the format (start, end).

    Returns:
        Tensor: Encoded deltas between anchors and ground truth boxes.
    """
    ex_widths = anchors[:, 1] - anchors[:, 0] + 1  # TODO: WHY DO WE ADD 1?
    ex_ctr_x = (anchors[:, 0] + anchors[:, 1]) / 2

    gt_widths = gt_boxes[:, 1] - gt_boxes[:, 0] + 1  # TODO: WHY DO WE ADD 1?
    gt_ctr_x = (gt_boxes[:, 0] + gt_boxes[:, 1]) / 2

    wx, ww = (10.0, 5.0)  # noqa: WPS111
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dw = ww * torch.log(gt_widths / ex_widths)

    return torch.stack((targets_dx, targets_dw), dim=1)  # type: ignore


def decode_spans(preds: Tensor, anchors: Tensor) -> Tensor:
    """Decode deltas into spans.

    Args:
        preds (Tensor): Predictions in the format (dx, dw).
        anchors (Tensor): Anchors in the format (start, end).

    Returns:
        Tensor: Decoded spans.
    """
    anchors = anchors.to(preds.dtype)

    widths = anchors[:, 1] - anchors[:, 0] + 1  # TODO: WHY DO WE ADD 1?
    ctr_x = (anchors[:, 0] + anchors[:, 1]) / 2

    wx, ww = (10.0, 5.0)  # noqa: WPS111
    delta_w = preds[:, 1::2] / ww
    delta_x = preds[:, 0::2] / wx

    # Prevent sending too large values into torch.exp()
    delta_w = torch.clamp(delta_w, max=math.log(1000.0 / 16))  # noqa: WPS432

    pred_ctr_x = delta_x * widths[:, None] + ctr_x[:, None]
    pred_w = torch.exp(delta_w) * widths[:, None]

    pred_boxes = torch.zeros_like(preds)
    pred_boxes[:, 0::2] = pred_ctr_x - 0.5 * (pred_w - 1)
    pred_boxes[:, 1::2] = pred_ctr_x + 0.5 * (pred_w - 1)
    return pred_boxes


def gt_to_absolete(spans: torch.Tensor, text_len: int) -> torch.Tensor:
    gt_spans = span_cxw_to_xx(spans) * text_len
    gt_spans = torch.clamp(gt_spans, 0, text_len)
    return gt_spans


def filter_by_thresh(preds, thresh=0.1):
    preds = preds[:, 2] > thresh
    return preds


def convert_outputs(logits: torch.Tensor, spans: torch.Tensor, text_len: int) -> torch.Tensor:
    """
    logits: (B, 2)
    spans: (B, 2)
    Returns: all predictions in format (x_absolute, w_absolute, ai_conf, not_ai_conf)

    """
    pred_probs = torch.softmax(logits, -1)

    # converts spans to chars
    spans = span_cxw_to_xx(spans) * text_len
    spans = torch.clamp(spans, 0, text_len)
    # concat spans and scores
    preds = torch.cat([spans, pred_probs], dim=1)
    return preds
