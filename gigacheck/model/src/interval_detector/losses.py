from loguru import logger
from torch import nn
import torch
import torch.nn.functional as F

from gigacheck.model.src.interval_detector.dn_detr.denoise_losses import DenoiseLosses
from gigacheck.model.src.interval_detector.focal_loss import sigmoid_focal_loss
from gigacheck.model.src.interval_detector.modules.matcher import HungarianMatcher
from gigacheck.model.src.interval_detector.span_utils import generalized_temporal_iou, span_cxw_to_xx


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        matcher: HungarianMatcher,
        weight_dict,
        eos_coef,
        losses,
        span_loss_type,
        max_s_l,
        use_focal_loss,
    ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            span_loss_type: str, [l1, ce]
            max_s_l: int,
        """
        super().__init__()
        logger.info("Building SetCriterion for losses calculation.")
        self.matcher: HungarianMatcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.span_loss_type = span_loss_type
        self.max_s_l = max_s_l

        self.denoise_losses = DenoiseLosses(use_focal_loss=use_focal_loss)
        self.use_focal_loss = use_focal_loss

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef

        self.focal_alpha = 0.75
        self.focal_gamma = 2
        empty_weight = torch.ones(2)
        # 0: foreground, 1: background
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer("empty_weight", empty_weight)

    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
        The target spans are expected in format (center_x, w), normalized by the text size.
        """
        assert "pred_spans" in outputs  # (bs, #queries, 2)
        targets = targets["span_labels"]

        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs["pred_spans"][idx]  # (#spans, max_s_l * 2)
        tgt_spans = torch.cat([t["spans"][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)
        tgt_spans = tgt_spans.to(src_spans.device)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction="none")
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_s_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction="none")
            loss_giou = loss_span.new_zeros([1])

        losses = {}
        losses["loss_span"] = loss_span.mean() if len(loss_span) else torch.tensor(0)
        losses["loss_giou"] = loss_giou.mean() if len(loss_giou) else torch.tensor(0)
        return losses

    def loss_span_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)"""
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]  # (batch_size, #queries, #classes=2)
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(
            src_logits.shape[:2], self.background_label, dtype=torch.int64, device=src_logits.device
        )  # (batch_size, #queries)
        target_classes[idx] = self.foreground_label

        if self.use_focal_loss:
            target_classes_onehot = torch.zeros(
                [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]],
                dtype=src_logits.dtype,
                layout=src_logits.layout,
                device=src_logits.device,
            )
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)  # (batch_size, #queries, #classes=2)
            num_bboxes = sum([len(span["spans"]) for span in targets["span_labels"]])
            loss_ce = (
                sigmoid_focal_loss(
                    src_logits, target_classes_onehot, num_bboxes, alpha=self.focal_alpha, gamma=self.focal_gamma
                )
                * src_logits.shape[1]
            )
        else:
            loss_ce = F.cross_entropy(
                src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none"
            ).mean()

        losses = {"loss_label": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        return losses

    def loss_denoise(self, outputs, targets, indices, aux_num: int = -1):
        return self.denoise_losses(outputs["mask_dict"], aux_num=aux_num)

    def _get_src_permutation_idx_by_samples(self, indices):
        # permute predictions following indices, but keep by-sample structure
        batch_idx = [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        src_idx = [src for (src, _) in indices]
        return batch_idx, src_idx

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "span_labels": self.loss_span_labels,
            "denoise": self.loss_denoise,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs=outputs, targets=targets, indices=indices, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        # Positive indicies
        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)
        indices = self.matcher(outputs_without_aux, targets, kwargs.get("ref_points", None))

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if loss == "denoise" and ("mask_dict" not in outputs or outputs["mask_dict"] is None):
                continue

            new_losses = self.get_loss(loss, outputs, targets, indices)
            losses.update(new_losses)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # Positive aux idices. (src_ind, tgt_ind)
                indices = self.matcher(aux_outputs, targets, kwargs.get("ref_points", None))
                for loss in self.losses:

                    kwargs = {}
                    if loss == "denoise":
                        if "mask_dict" not in outputs or outputs["mask_dict"] is None:
                            continue
                        kwargs = {"aux_num": i}

                    if loss == "span_labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    output: (#items, #classes)
    target: int,
    """
    maxk = max(topk)
    num_items = output.size(0)

    if num_items == 0:
        # to get zero loss
        return [100]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / num_items))
    return res
