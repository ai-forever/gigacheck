from typing import List, Tuple

import numpy as np
import torch
from gigacheck.model.src.interval_detector.modules.layers import inverse_sigmoid
from torch import Tensor, nn


class LayerAnchor(nn.Module):
    """
    The embedding class for anchors.

    Separates the centers and widths of the anchors, which allows to better adjust them during training.
    In addition, there is a special initialization for a video width token
    """

    def __init__(self, num_queries: int, ratios: Tuple[float, ...] = (0.5, 0.35), special_anchor: bool = True) -> None:
        """
        Initialize the AnchorEmbedding module.

        Args:
            num_queries (int): Total number of anchors
            ratios (Tuple[float, ...]): Defines anchor levels, each level consists of ratio * num_queries of anchors.
            special_anchor (bool): If True, a special anchor with parameters (0.5, 1) will be added.
        """
        super().__init__()
        self.num_queries = num_queries
        self.special_anchor = special_anchor
        self.ratios = ratios
        # Create two separate embeddings: one for the centers, the other for the widths
        self.layers_num_queries = num_queries - 1 if special_anchor else num_queries  # If there is a special anchor,
        # then remove one of the anchor from the distribution
        self.center = nn.Embedding(self.num_queries, 1)
        self.width = nn.Embedding(self.num_queries, 1)

        self.reset_parameters()

    def reset_parameters(self):
        # Determine the number of anchors for each level
        ratios_n = [int(ratio * self.layers_num_queries) for ratio in self.ratios]
        # Determine the number of anchors for the last level
        ratios_n.append(self.layers_num_queries - sum(ratios_n))
        centers: List[float] = []
        widths: List[float] = []
        for ratio_n in ratios_n:
            # distribute the anchors evenly, their width is the same
            centers.extend(np.linspace(0, 1, ratio_n + 2)[1:-1])
            widths.extend([1 / ratio_n] * ratio_n)
        # Add a special anchor
        if self.special_anchor:
            centers.append(0.5)
            widths.append(1)

        # Initialize the weights
        if self.center.weight.device.type != "meta":
            self.center.weight.data = inverse_sigmoid(torch.Tensor(centers))[:, None]
            self.width.weight.data = inverse_sigmoid(torch.Tensor(widths))[:, None]

    def get_reference_points(self) -> Tensor:
        """
        Get reference points as tensor [n_points, 2].

        Returns:
            Tensor: reference points as tensor [n_points, 2]
        """
        return torch.cat([self.center.weight, self.width.weight], dim=-1)

    def forward(self, idx: int) -> Tensor:
        """
        Forward pass for the AnchorEmbedding module.

        Args:
            idx (Tensor): Index tensor for the anchors.

        Returns:
            Tensor: Concatenated embeddings of centers and widths.
        """
        centers = self.center_embedding(idx)  # type: ignore
        widths = self.width_embedding(idx)  # type: ignore
        return torch.cat([centers, widths], dim=-1)
