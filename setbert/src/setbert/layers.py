"""Transformer layers used by the vendored DNABERT backbone.

Only the subset of layers that ``DnaBert.__init__`` actually constructs is
kept here: ``MultiHeadAttention`` → ``RelativeMultiHeadAttention`` →
``MultiHeadAttentionBlock`` → ``TransformerEncoderBlock`` →
``TransformerEncoder``. The flex-attention path, induced-set / decoder
blocks, and the ``nn.TransformerEncoderLayer`` override from the upstream
deepbio-toolkit are unused at inference / fine-tuning time on the released
checkpoints and have been removed.

All deprecated classes are left flagged with ``@deprecated("")`` for parity
with the upstream module; the decorator is a no-op stub (see
:mod:`setbert.utils`).
"""

from __future__ import annotations

import abc
import copy
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import deprecated, export


# Multi-head Attention Mechanisms ------------------------------------------------------------------


@export
@deprecated("")
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        head_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias
        if head_embed_dim is None:
            assert embed_dim % num_heads == 0, (
                "embed_dim must be divisible by num_heads if head_embed_dim is not provided"
            )
            head_embed_dim = embed_dim // num_heads
        self.head_embed_dim = head_embed_dim

        self.w_query = nn.Linear(embed_dim, self.head_embed_dim * num_heads, bias=bias)
        self.w_key = nn.Linear(embed_dim, self.head_embed_dim * num_heads, bias=bias)
        self.w_value = nn.Linear(embed_dim, self.head_embed_dim * num_heads, bias=bias)
        self.w_output = nn.Linear(self.head_embed_dim * num_heads, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attention_head_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # NOTE: this forward dispatches to torch.nn.functional.scaled_dot_product_attention
        # (SDPA) in the common case. SDPA fuses the QK^T -> softmax -> @V chain into a
        # single tiled kernel (FlashAttention or mem-efficient attention on Ampere+),
        # avoiding the [..., h, n_q, n_k] attention matrix materialisation that the
        # original compute_attention_weights() path did in HBM. The original path is
        # preserved as _forward_classic and is still used when the caller asks for the
        # attention weights themselves (SDPA does not expose them).
        if return_attention_weights:
            return self._forward_classic(
                query, key, value,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
                attention_head_mask=attention_head_mask,
                average_attention_weights=average_attention_weights,
                return_attention_weights=return_attention_weights,
            )

        *extra_dims_q, n_q, _ = query.size()
        *extra_dims_k, n_k, _ = key.size()
        *extra_dims_v, _, _ = value.size()

        q = self.w_query(query).view((*extra_dims_q, n_q, self.num_heads, self.head_embed_dim)).transpose(-2, -3)
        k = self.w_key(key).view((*extra_dims_k, n_k, self.num_heads, self.head_embed_dim)).transpose(-2, -3)
        v = self.w_value(value).view((*extra_dims_v, n_k, self.num_heads, self.head_embed_dim)).transpose(-2, -3)

        # Build SDPA attn_mask. SDPA computes softmax(Q K^T / sqrt(d) + attn_mask) V,
        # so any subclass-supplied additive bias (e.g. relative-position) must already
        # be divided by sqrt(d_k) -- see _additive_bias docstring.
        mask = self.merge_mask(attention_mask, key_padding_mask)  # True = mask out
        bias = self._additive_bias(q, k)  # [..., h, n_q, n_k] or None
        if bias is not None:
            attn_mask = bias
            if mask is not None:
                attn_mask = attn_mask.masked_fill(mask.unsqueeze(-3), float("-inf"))
        elif mask is not None:
            # Float -inf bias, broadcast over heads. Avoids allocating a
            # [..., h, n_q, n_k] tensor when only padding masking is needed.
            pad_bias = torch.zeros(mask.shape, dtype=q.dtype, device=q.device)
            pad_bias = pad_bias.masked_fill(mask, float("-inf"))
            attn_mask = pad_bias.unsqueeze(-3)
        else:
            attn_mask = None

        dropout_p = float(self.dropout.p) if self.training else 0.0
        attention = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False,
        )

        # head_mask is applied to softmax(...) in _forward_classic; matmul is linear so
        # multiplying the attention output by the per-head scalar gives the same result.
        if attention_head_mask is not None:
            attention = attention * attention_head_mask.view(-1, 1, 1)

        attention = attention.transpose(-2, -3).reshape(
            (*extra_dims_v, n_q, self.num_heads * self.head_embed_dim)
        )
        return self.w_output(attention)

    def _additive_bias(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Subclass hook: return additive attention bias ``[..., h, n_q, n_k]`` or None.

        The bias is added to ``QK^T / sqrt(d_k)`` before softmax (SDPA's
        attn_mask convention). Subclasses adding relative-position or other
        learned biases must therefore return the bias already divided by
        ``sqrt(d_k)``.
        """
        return None

    def _forward_classic(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attention_head_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Materialises the attention matrix via compute_attention_weights. Kept
        # for callers that need attention weights back, or subclasses that only
        # override compute_attention_weights (not _additive_bias).
        *extra_dims_q, n_q, _ = query.size()
        *extra_dims_k, n_k, _ = key.size()
        *extra_dims_v, _, _ = value.size()

        q = self.w_query(query).view((*extra_dims_q, n_q, self.num_heads, self.head_embed_dim)).transpose(-2, -3)
        k = self.w_key(key).view((*extra_dims_k, n_k, self.num_heads, self.head_embed_dim)).transpose(-2, -3)
        v = self.w_value(value).view((*extra_dims_v, n_k, self.num_heads, self.head_embed_dim)).transpose(-2, -3)

        attention_weights = self.compute_attention_weights(
            q, k,
            self.merge_mask(attention_mask, key_padding_mask),
            attention_head_mask,
        )
        attention_weights = self.dropout(attention_weights)

        attention = torch.matmul(attention_weights, v)
        attention = attention.transpose(-2, -3).reshape(
            (*extra_dims_v, n_q, self.num_heads * self.head_embed_dim)
        )
        output = self.w_output(attention)

        if return_attention_weights:
            if average_attention_weights:
                n = (
                    attention_head_mask.sum()
                    if attention_head_mask is not None
                    else attention_weights.size(-1)
                )
                attention_weights = attention_weights.sum(dim=-3) / n
            return output, attention_weights
        return output

    def merge_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> Union[torch.Tensor, None]:
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(-2)
        if attention_mask is None:
            return key_padding_mask
        if key_padding_mask is None:
            return attention_mask
        return attention_mask | key_padding_mask

    def compute_attention_weights(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        attention_head_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attention_weights = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.head_embed_dim)
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(
                attention_mask.unsqueeze(-3), float("-inf")
            )
        attention_weights = F.softmax(attention_weights, dim=-1)
        if attention_head_mask is not None:
            attention_weights = attention_weights * attention_head_mask.view((-1, 1, 1))
        return attention_weights

    def __len__(self):
        return self.num_heads


@export
@deprecated("")
class RelativeMultiHeadAttention(MultiHeadAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_length: int,
        dropout: float = 0.0,
        bias: bool = True,
        head_embed_dim: Optional[int] = None,
    ):
        super().__init__(embed_dim, num_heads, dropout, bias, head_embed_dim)
        self.max_length = max_length
        self.pos_embeddings = nn.Parameter(
            torch.randn(self.head_embed_dim, 2 * max_length - 1)
        )

    def _skew(self, x: torch.Tensor):
        """Memory-efficient skew operation."""
        n = x.shape[-1] - x.shape[-1] // 2
        x = F.pad(x, (0, 1))
        skewed = x.flatten(-2).narrow(-1, 0, x.shape[-2] * (x.shape[-1] - 1)).view(
            (*x.shape[:-1], -1)
        )
        rel = skewed.narrow(-1, n - 1, n)
        return rel

    def _additive_bias(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Return the relative-position bias pre-scaled by ``1/sqrt(d_k)``.

        SDPA's ``attn_mask`` is added AFTER scaling, so the bias must already
        be divided by ``sqrt(d_k)``.
        """
        n = key.shape[-2]
        pos_embeddings = F.pad(
            self.pos_embeddings, (n - self.max_length, n - self.max_length), mode="replicate"
        )
        att_qrel = self._skew(torch.matmul(query, pos_embeddings))
        return att_qrel / np.sqrt(self.head_embed_dim)

    def compute_attention_weights(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        attention_head_mask: Optional[torch.Tensor],
    ):
        # Kept for the _forward_classic path (return_attention_weights=True). The
        # fast SDPA path uses _additive_bias above and never calls this method.
        n = key.shape[-2]
        pos_embeddings = F.pad(
            self.pos_embeddings, (n - self.max_length, n - self.max_length), mode="replicate"
        )
        att_qk = torch.matmul(query, key.transpose(-2, -1))
        att_qrel = self._skew(torch.matmul(query, pos_embeddings))
        attention_weights = (att_qk + att_qrel) / np.sqrt(self.head_embed_dim)
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(
                attention_mask.unsqueeze(-3), float("-inf")
            )
        attention_weights = F.softmax(attention_weights, dim=-1)
        if attention_head_mask is not None:
            attention_weights = attention_weights * attention_head_mask.view((-1, 1, 1))
        return attention_weights


# Transformer Generics -----------------------------------------------------------------------------


@export
@deprecated("")
class MultiHeadAttentionBlock(nn.Module):
    def __init__(
        self,
        mha: MultiHeadAttention,
        feedforward_dim: int,
        feedforward_activation: Callable = F.gelu,
        norm_first: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mha = mha
        self.feedforward_dim = feedforward_dim
        self.feedforward_activation = feedforward_activation
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(mha.embed_dim)
        self.norm2 = nn.LayerNorm(mha.embed_dim)
        self.feedforward_linear1 = nn.Linear(mha.embed_dim, feedforward_dim)
        self.feedforward_linear2 = nn.Linear(feedforward_dim, mha.embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _cross_attention_block(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        attention_head_mask: Optional[torch.Tensor],
        average_attention_weights: bool,
        return_attention_weights: bool,
    ):
        attention_output = self.mha(
            x, y, y,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            attention_head_mask=attention_head_mask,
            average_attention_weights=average_attention_weights,
            return_attention_weights=return_attention_weights,
        )
        if isinstance(attention_output, tuple):
            attention_output, *extra = attention_output
        else:
            extra = None
        return self.dropout1(attention_output), extra

    def _feedforward_block(self, x: torch.Tensor):
        return self.dropout2(
            self.feedforward_linear2(
                self.feedforward_activation(self.feedforward_linear1(x))
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attention_head_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False,
    ):
        if self.norm_first:
            if x is y:
                x_norm = y_norm = self.norm1(x)
            else:
                x_norm = self.norm1(x)
                y_norm = self.norm1(y)
            attention_output, extra_output = self._cross_attention_block(
                x_norm, y_norm,
                attention_mask, key_padding_mask, attention_head_mask,
                average_attention_weights, return_attention_weights,
            )
            x = x + attention_output
            x = x + self._feedforward_block(self.norm2(x))
        else:
            attention_output, extra_output = self._cross_attention_block(
                x, y,
                attention_mask, key_padding_mask, attention_head_mask,
                average_attention_weights, return_attention_weights,
            )
            x = self.norm1(x + attention_output)
            x = self.norm2(x + self._feedforward_block(x))
            return x, extra_output
        if extra_output is not None:
            return x, *extra_output
        return x

    @property
    def embed_dim(self):
        return self.mha.embed_dim


# Transformer Encoders -----------------------------------------------------------------------------


@deprecated("")
class ITransformerEncoder(abc.ABC, nn.Module):
    """Abstract base class for transformer encoder blocks / stacks."""

    @abc.abstractmethod
    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        attention_head_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False,
    ) -> Any:
        return NotImplemented

    @property
    @abc.abstractmethod
    def embed_dim(self) -> int:
        return NotImplemented


@export
@deprecated("")
class TransformerEncoderBlock(ITransformerEncoder):
    def __init__(
        self,
        mha: MultiHeadAttention,
        feedforward_dim: int,
        feedforward_activation: Callable = F.gelu,
        norm_first: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mab = MultiHeadAttentionBlock(
            mha=mha,
            feedforward_dim=feedforward_dim,
            feedforward_activation=feedforward_activation,
            norm_first=norm_first,
            dropout=dropout,
        )
        self.attention_head_mask = nn.Parameter(
            torch.ones(mha.num_heads), requires_grad=False
        )

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        attention_head_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False,
    ):
        if attention_head_mask is None:
            attention_head_mask = self.attention_head_mask
        return self.mab.forward(
            x=src, y=src,
            attention_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            attention_head_mask=attention_head_mask,
            average_attention_weights=average_attention_weights,
            return_attention_weights=return_attention_weights,
        )

    @property
    def embed_dim(self) -> int:
        return self.mab.embed_dim

    @property
    def num_heads(self) -> int:
        return self.mab.mha.num_heads


@export
@deprecated("")
class TransformerEncoder(ITransformerEncoder):
    def __init__(self, encoder_layer: ITransformerEncoder, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        attention_head_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False,
    ):
        extra_outputs: List[Any] = []
        output = src
        head_mask = None
        for i, layer in enumerate(cast(Sequence[ITransformerEncoder], self.layers)):
            if attention_head_mask is not None:
                head_mask = attention_head_mask[i]
            output = layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                attention_head_mask=head_mask,
                average_attention_weights=average_attention_weights,
                return_attention_weights=return_attention_weights,
            )
            if isinstance(output, tuple):
                output, *extra_output = output
                if len(extra_output) > 0:
                    extra_outputs.append(extra_output)
        if len(extra_outputs) > 0:
            return output, *zip(*extra_outputs)
        return output

    @property
    def attention_head_mask(self):
        return torch.stack([layer.attention_head_mask for layer in self.layers])

    @attention_head_mask.setter
    def attention_head_mask(self, attention_head_mask):
        for layer, mask in zip(self.layers, attention_head_mask):
            layer.attention_head_mask[:] = mask

    @property
    def embed_dim(self):
        return self.layers[0].embed_dim

    @property
    def num_heads(self):
        return self.layers[0].num_heads

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, index):
        return self.layers[index]
