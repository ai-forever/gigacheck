# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math

import torch
from loguru import logger
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images. (To 1D sequences)
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        logger.info("Building PositionEmbeddingSine")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mem, mask):
        """
        Args:
            mem: torch.tensor, (batch_size, L, d)
            mask: torch.tensor, (batch_size, L), with 1 as valid

        Returns:

        """
        assert mask is not None
        x_embed = mask.cumsum(1, dtype=torch.float32)  # (bsz, L)
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mem.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t  # (bsz, L, num_pos_feats)
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(
            2
        )  # (bsz, L, num_pos_feats)
        pos_x = pos_x.to(mem.dtype)
        return pos_x


def gen_sineembed_for_position(pos_tensor: torch.Tensor, d_model: int, temperature: int = 10000) -> torch.Tensor:
    """Generate sine embeddings for position.

    Args:
        pos_tensor (Tensor): position tensor (anchor points)
        d_model (int): dimension of the model
        temperature (int): temperature of the pos emb.

    Returns:
        Tensor: sine embeddings for position
    """
    scale = 2 * math.pi
    dim_t = torch.arange(d_model // 2, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / (d_model // 2))

    # prepare PE embedding for center value
    center_embed = pos_tensor[:, :, 0] * scale
    pos_x = center_embed[:, :, None] / dim_t
    pos_x = torch.stack(  # noqa: WPS317
        (
            pos_x[:, :, 0::2].sin(),
            pos_x[:, :, 1::2].cos(),
        ),
        dim=3,
    ).flatten(2)

    # prepare PE embedding for width value
    span_embed = pos_tensor[:, :, 1] * scale
    pos_w = span_embed[:, :, None] / dim_t
    pos_w = torch.stack(  # noqa: WPS317
        (
            pos_w[:, :, 0::2].sin(),
            pos_w[:, :, 1::2].cos(),
        ),
        dim=3,
    ).flatten(2)

    sineembed = torch.cat((pos_x, pos_w), dim=2).to(pos_tensor.dtype)
    return sineembed
