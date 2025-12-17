from typing import List, Optional, Union

import torch
from torch import nn

from gigacheck.model.src.interval_detector.modules.layers import FeedForwardNetwork, get_clones


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        expansion_ratio: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize the TransformerEncoderLayer.

        Args:
            d_model (int): The dimension of the input feature.
            nhead (int): The number of heads in the multihead attention.
            expansion_ratio (int): The expansion ratio for the hidden layer dimension of FFN. Defaults to 4.
            dropout (float): Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.ffn = FeedForwardNetwork(d_model, expansion_ratio, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        embedding: torch.Tensor,
        position_embedding: torch.Tensor,
        embedding_mask: Optional[torch.Tensor] = None,
        embedding_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass of the TransformerEncoderLayer.

        Args:
            embedding (Tensor): The input tensor.
            position_embedding (Tensor): The positional embedding for the input tensor.
            embedding_mask (Optional[Tensor]): The mask for the input tensor. Defaults to None.
            embedding_key_padding_mask (Optional[Tensor]):  The key padding mask for the input tensor. Defaults to None.

        Returns:
            _type_: _description_
        """
        query = key = self.with_pos_embed(embedding, position_embedding)

        # Attention part
        att_out, _ = self.self_attn(
            query=query,
            key=key,
            value=embedding,
            attn_mask=embedding_mask,
            key_padding_mask=embedding_key_padding_mask,
        )
        embedding = embedding + self.dropout1(att_out)
        embedding = self.norm1(embedding)

        # FFN part
        ffn_out = self.ffn(embedding)
        embedding = embedding + self.dropout2(ffn_out)
        embedding = self.norm2(embedding)
        return embedding


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder class that stacks multiple encoder layers.

    Attributes:
        num_layers (int): Number of encoder layers.
        layers (nn.ModuleList): List of duplicated encoder layers.
        return_intermediate (bool): Whether to return intermediate outputs from each layer.
    """

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        return_intermediate: bool = False,
    ) -> None:
        """Initialize TransformerEncoder.

        Args:
            encoder_layer (TransformerEncoderLayer): An instance of the TransformerEncoderLayer to be duplicated.
            num_layers (int): Number of layers to be stacked.
            return_intermediate (bool): If set to True, the encoder will return all intermediate representations.
        """
        super().__init__()
        self.num_layers = num_layers
        self.layers = get_clones(encoder_layer, num_layers)
        self.return_intermediate = return_intermediate

    def forward(
        self,
        src: torch.Tensor,
        src_pos: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Pass the input through the encoder layers in turn.

        Args:
            src (Tensor): The sequence to the encoder (required).
            src_pos (Tensor): The position of the sequence (required).
            mask (Optional[Tensor]): The mask for the src sequence (optional).
            src_key_padding_mask (Optional[Tensor]): The mask for the src keys per batch (optional).

        Returns:
            Union[Tensor, List[Tensor]]: Output of the last layer or intermediate outputs from all layers.
        """
        output = src

        intermediate = []

        for i, layer in enumerate(self.layers):
            output = layer(
                output,
                position_embedding=src_pos,
                embedding_mask=mask,
                embedding_key_padding_mask=src_key_padding_mask,
            )

            if self.return_intermediate:
                intermediate.append(output)

            if torch.any(torch.isnan(output)):
                raise ValueError(f"Encoder {i} layer's output contains nan")

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output
