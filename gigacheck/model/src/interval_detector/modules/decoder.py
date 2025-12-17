"""Transformer decoder based on DAB DETR implementation."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gigacheck.model.src.interval_detector.modules.attention import MultiheadAttention
from gigacheck.model.src.interval_detector.modules.layers import MLP, get_clones, inverse_sigmoid
from gigacheck.model.src.interval_detector.modules.position_encoding import gen_sineembed_for_position


class TransformerDecoder(nn.Module):
    """Transformer decoder consisting of *args.decoder_layers* layers."""

    def __init__(
        self,
        decoder_layer,
        num_layers: int,
        return_intermediate: bool = False,
        d_model: int = 2048,
        query_dim: int = 2,
        temperature: int = 10000,
        keep_query_pos: bool = False,
        query_scale_type: str = "cond_elewise",
        bbox_embed_diff_each_layer: bool = False,
    ) -> None:
        """
        Initialize a Transformer decoder.

        Args:
            decoder_layer (TransformerDecoderLayer): an instance of the TransformerDecoderLayer() class
            num_layers (int): number of decoder layers
            return_intermediate (bool): whether to return intermediate results
            d_model (int): dimension of the model
            query_dim (int): dimension of the query vector
            temperature (int): temperature of the pos emb.
            keep_query_pos (bool): whether to keep the query position
            query_scale_type (str): type of query scale
            bbox_embed_diff_each_layer (bool): whether to use different bbox_embed for each layer
        Raises:
            NotImplementedError: unknown query_scale_type
        """
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.query_dim = query_dim
        self.temperature = temperature

        assert query_scale_type in {"cond_elewise", "cond_scalar", "fix_elewise"}
        self.query_scale_type = query_scale_type
        if query_scale_type == "cond_elewise":
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == "cond_scalar":
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == "fix_elewise":
            self.query_scale = nn.Embedding(num_layers, d_model)  # type: ignore
        else:
            raise NotImplementedError(f"Unknown query_scale_type: {query_scale_type}")

        # mlp for query PE embeddings
        self.ref_point_head = MLP(input_dim=d_model, hidden_dim=d_model, output_dim=d_model, num_layers=2)

        self.bbox_embed: MLP = None

        self.d_model = d_model
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        self.norm = nn.LayerNorm(d_model)
        self.ref_anchor_head = MLP(input_dim=d_model, hidden_dim=d_model, output_dim=1, num_layers=2)

        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None

    def reset_parameters(self):
        pass

    def apply_cond_spatial_query(self, query_sine_embed: Tensor, output: Tensor, layer_id: int) -> Tensor:
        """Rescale the postional embs leverage conditional spatial query.

        Based on DAB-Implementation.

        Args:
            query_sine_embed (Tensor): PE embedding generated from query embs. Shape: [#Queries, batch_size, dim]
            output (Tensor): Content vector used as query. Shape: [#Queries, batch_size, dim]
            layer_id (int): Decoder layer id.

        Returns:
            Tensor: Rescaled PE Embs.
        """
        # For the first decoder layer, we do not apply transformation over p_s
        if self.query_scale_type == "fix_elewise":
            pos_transformation = self.query_scale.weight[layer_id]
        else:
            pos_transformation = 1 if layer_id == 0 else self.query_scale(output)

        # apply transformation
        return query_sine_embed * pos_transformation

    def update_reference_points(self, output: Tensor, reference_points: Tensor, layer_id: int) -> Tensor:
        """Update reference points based on DAB implementation.

        Args:
            output (Tensor): content vector from CA block
            reference_points (Tensor): anchor points
            layer_id (int): Decoder layer id

        Returns:
            Tensor: updated reference points
        """
        if self.bbox_embed_diff_each_layer:
            box_offsets = self.bbox_embed[layer_id](output)
        else:
            box_offsets = self.bbox_embed(output)
        new_boxes = box_offsets[..., : self.query_dim] + inverse_sigmoid(reference_points)  # noqa: WPS221
        return new_boxes.sigmoid()

    def forward(  # noqa: WPS210
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
    ) -> Tuple[Tensor, Tensor]:
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()

        device = next(self.ref_point_head.parameters()).device
        if reference_points.device != device:
            reference_points = reference_points.to(device)

        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(reference_points, self.d_model, temperature=self.temperature)

            # construct PE emb for self attention layer
            query_pos = self.ref_point_head(query_sine_embed)

            query_sine_embed = self.apply_cond_spatial_query(query_sine_embed, output, layer_id)
            # modulated HW attentions
            reft_cond = self.ref_anchor_head(output).sigmoid().squeeze(2)  # nq, bs, 1
            obj_width = reference_points[..., 1]
            if reft_cond.device != obj_width.device:
                reft_cond = reft_cond.to(obj_width.device)
            modulation_value = (obj_width / reft_cond).unsqueeze(-1)
            query_sine_embed = query_sine_embed * modulation_value

            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                query_sine_embed=query_sine_embed,
                is_first=(layer_id == 0),
            )

            # update anchor
            new_reference_points = self.update_reference_points(output, reference_points, layer_id)

            if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)
            reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        output = self.norm(output)
        if self.return_intermediate:
            intermediate.pop()
            intermediate.append(output)
            stacked_decoder_outputs = torch.stack(intermediate).transpose(1, 2)
            stacked_reference_points = torch.stack(ref_points).transpose(1, 2)
            return stacked_decoder_outputs, stacked_reference_points

        return output.unsqueeze(0), new_reference_points.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        keep_query_pos=False,
        rm_self_attn_decoder=False,
    ):
        super().__init__()
        # Decoder Self-Attention
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.keep_query_pos = keep_query_pos

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        query_sine_embed=None,
        is_first=False,
    ):

        # ========== Begin of Self-Attention =============
        if not self.rm_self_attn_decoder:
            # Apply projections here
            # shape: num_queries x batch_size x 256
            q_content = self.sa_qcontent_proj(tgt)  # target is the input of the first decoder layer. zero by default.
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)

            # num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            q = q_content + q_pos
            k = k_content + k_pos

            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
            # ========== End of Self-Attention =============

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(
            query=q, key=k, value=v, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask
        )[0]
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
