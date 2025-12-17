from typing import Any, Dict, Optional, Tuple
import torch

from gigacheck.model.src.interval_detector.modules.layers import inverse_sigmoid
from gigacheck.model.src.interval_detector.span_utils import span_cxw_to_xx, span_xx_to_cxw


def prepare_for_denoise(
    label_enc: torch.nn.Embedding,
    targets: Optional[Dict[str, Any]],
    refpoint_emb: torch.Tensor,
    num_queries: int = 10,
    num_groups: int = 5,
    center_noise_scale: float = 0.4,
    width_noise_scale: float = 0.4,
    noise_offset: float = 0.1,
    batch_size: int = 512,
    training: int = True,
    num_classes: int = 1,
    label_noise_scale: float = 0,
    tgt_weight: Optional[torch.Tensor] = None,
):
    """
    Prepare for dn components in forward function.

    Args:
        label_enc: target embeddings.
        targets: target dict contains "span_labels"
        refpoint_emb: positional queries as anchor points
        num_queries: number of queries
        num_groups: number of denoise groups
        center_noise_scale: noise scale for bbox center
        width_noise_scale: noise scale for bbox width
        noise_offset: min diff value
        batch_size: batch size
        training: whether it is training or inference
        tgt_weight: learnable tgt in dab deformable detr

    Returns:
        tuple: input_query_bbox, attn_mask, mask_dict
    """
    device = refpoint_emb.device
    indicator0 = torch.zeros([num_queries, 1], device=device)
    if tgt_weight is not None:
        # sometimes the target is empty, add a zero part of label_enc to avoid unused parameters
        tgt = torch.cat([tgt_weight, indicator0], dim=1) + label_enc.weight[0][0] * torch.tensor(0).to(device)
    else:
        # 1 is background labels (no object), get embedding for no object
        tgt = label_enc(torch.tensor(num_classes, device=device)).repeat(num_queries, 1)
        tgt = torch.cat([tgt, indicator0], dim=1)

    tgt = tgt.to(refpoint_emb.dtype)

    if training and num_groups > 0 and targets is not None:
        spans = targets["span_labels"]
        known = [torch.ones(len(span["spans"])) for span in spans]  # replace bboxes with ones in batch
        know_idx = [torch.nonzero(idxs) for idxs in known]  # enumerate bboxes of each object in the batch
        known_num = [sum(idxs) for idxs in known]  # count bboxes in the batch

        # prepare for the dn part
        tmp = torch.cat(known)  # flatten the indices
        known_indices = torch.nonzero(tmp).view(-1)  # enumerate all bboxes

        # get the batch index for each bbox
        boxes = torch.cat([span["spans"] for span in spans])  # flatten the batch bboxes
        if num_classes == 1:
            labels = torch.cat(known).long()
        else:
            labels = torch.cat([t["label"] for t in targets["labels"]])
        known_indices = known_indices.to(device)
        labels = labels.to(device)
        batch_idx = torch.cat([torch.full_like(ones, idx) for idx, ones in enumerate(known)])  # noqa: WPS221
        batch_idx = batch_idx.to(device)

        # add noise
        known_labels = labels.repeat(num_groups, 1).view(-1)
        known_indices = known_indices.repeat(num_groups, 1).view(-1)
        known_bid = batch_idx.repeat(num_groups, 1).view(-1)
        known_bboxes = boxes.repeat(num_groups, 1)
        known_labels_expanded = known_labels.clone()
        known_bbox_expand = known_bboxes.clone()

        # noise on the label
        if label_noise_scale > 0 and num_classes > 1:
            p = torch.rand_like(known_labels_expanded.float())
            chosen_indice = torch.nonzero(p < (label_noise_scale)).view(-1)  # usually half of bbox noise
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            known_labels_expanded.scatter_(0, chosen_indice, new_label)

        # apply noise on the box
        if center_noise_scale * width_noise_scale > 0:
            diff = torch.zeros_like(known_bbox_expand, device=device)
            # diff_value = known_bbox_expand[:, 1:]
            diff_value = torch.log(known_bbox_expand[:, 1:] + 1) + noise_offset  # from 0 to 1
            diff[:, :1] = diff_value / 2  # center diff
            diff[:, 1:] = diff_value  # width diff
            modulation = torch.rand_like(known_bbox_expand, device=device) * 2 - 1.0  # from -1 to 1
            modulated_diff = torch.mul(modulation, diff).to(known_bbox_expand.device)

            # make diff scale dependent
            known_bbox_expand[:, 0] += modulated_diff[:, 0] * center_noise_scale
            known_bbox_expand[:, 1] += modulated_diff[:, 1] * width_noise_scale
            known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)
            known_bbox_expand_xx = span_cxw_to_xx(known_bbox_expand).clamp(min=0.0, max=1.0)
            known_bbox_expand = span_xx_to_cxw(known_bbox_expand_xx)

        # padding shapes
        single_pad = int(max(known_num))
        pad_size = int(single_pad * num_groups)

        # prepare labels
        if num_classes == 1:
            # 0 is foreground, get embedding for an object
            input_label_embed = label_enc(known_labels_expanded - 1)
        else:
            input_label_embed = label_enc(known_labels_expanded)
        indicator1 = torch.ones([input_label_embed.shape[0], 1]).to(device)
        input_label_embed = torch.cat([input_label_embed, indicator1], dim=1)
        padding_label = torch.zeros(pad_size, tgt.size(1)).to(device)
        input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)

        # prepare bboxes
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)
        padding_bbox = torch.zeros(pad_size, 2, device=device)
        input_query_bbox = (
            torch.cat([padding_bbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1).to(known_bbox_expand.device)
        )

        # map in order
        map_known_indices = torch.tensor([]).to(known_bid.device)
        if len(known_num):
            map_known_indices = torch.cat([torch.tensor(range(int(num))) for num in known_num])  # [0,1, 0,1,2,3]
            map_known_indices = torch.cat([map_known_indices + single_pad * idx for idx in range(num_groups)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indices)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indices)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size, device=device) < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for idx in range(num_groups):
            if idx == 0:
                attn_mask[single_pad * idx : single_pad * (idx + 1), single_pad * (idx + 1) : pad_size] = True
            if idx == num_groups - 1:
                attn_mask[single_pad * idx : single_pad * (idx + 1), : single_pad * idx] = True
            else:
                attn_mask[single_pad * idx : single_pad * (idx + 1), single_pad * (idx + 1) : pad_size] = True
                attn_mask[single_pad * idx : single_pad * (idx + 1), : single_pad * idx] = True

        mask_dict = {
            "known_indices": torch.as_tensor(known_indices).long(),
            "batch_idx": torch.as_tensor(batch_idx).long(),
            "map_known_indices": torch.as_tensor(map_known_indices).long(),
            "known_lbs_bboxes": (known_labels, known_bboxes),
            "know_idx": know_idx,
            "pad_size": pad_size,
        }
    else:
        # no dn for inference
        input_query_label = tgt.repeat(batch_size, 1, 1)
        input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
        attn_mask = None
        mask_dict = None

    if tgt_weight is None:
        input_query_label = input_query_label.transpose(0, 1)  # (pad_size + num_queries, bs, 4096)
        input_query_bbox = input_query_bbox.transpose(0, 1)  # (pad_size + num_queries, bs, 2)

    return input_query_label, input_query_bbox, attn_mask, mask_dict


def dn_post_process(
    outputs_class: torch.Tensor,
    outputs_coord: torch.Tensor,
    offsets: torch.Tensor,
    mask_dict: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Post process of dn after output from the transformer.
    Separate denoise part from the output and put it in the mask_dict.

    Args:
        outputs_class (torch.Tensor): known and predicted classes.
        outputs_coord (torch.Tensor): known and predicted spans.
        offsets (torch.Tensor): calculated offsets.
        mask_dict (Dict[str, Any]): denoise paddings.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Predicted classes, spans, offsets

    """
    if mask_dict and mask_dict["pad_size"] > 0:
        tgt_pad_size = mask_dict["pad_size"]
        output_known_class = outputs_class[:, :, :tgt_pad_size, :]
        output_known_coord = outputs_coord[:, :, :tgt_pad_size, :]

        outputs_class = outputs_class[:, :, tgt_pad_size:, :]
        outputs_coord = outputs_coord[:, :, tgt_pad_size:, :]
        if offsets is not None:
            offsets = offsets[:, :, tgt_pad_size:, :]

        mask_dict["output_known_lbs_bboxes"] = (output_known_class, output_known_coord)
    elif mask_dict:
        mask_dict["output_known_lbs_bboxes"] = ([], [])

    return outputs_class, outputs_coord, offsets
