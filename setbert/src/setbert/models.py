"""SetBERT backbone used by the BreCol pipeline.

Only the base :class:`SetBert` model is kept here; the upstream
``SetBertForPretraining`` / ``SetBertForSequenceEmbedding`` /
``SetBertForSampleEmbedding`` variants are not exercised by
``scripts/train_setbert.py`` or ``scripts/build_setbert_embeddings.py`` and
have been removed along with the Lightning pretraining hooks.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel

from .base import BaseModelClassType, BaseModelType, DbtkModel


# Legacy class-path aliases. The published ``sirdavidludwig/setbert``
# checkpoint hard-codes ``dnabert.models.DnaBert(ForEmbedding)`` in its
# ``config.json``; rewrite those at load time so they resolve into this
# package's flat :mod:`setbert.dnabert` module.
_LEGACY_CLASS_PATHS = {
    "dnabert.models.DnaBert": "setbert.dnabert.DnaBert",
    "dnabert.models.DnaBertForEmbedding": "setbert.dnabert.DnaBertForEmbedding",
}


def _remap_class_path(value):
    if isinstance(value, str):
        return _LEGACY_CLASS_PATHS.get(value, value)
    return value


def _remap_nested_class_paths(obj):
    """Recursively rewrite legacy class-path strings in nested config dicts."""
    if isinstance(obj, dict):
        return {
            k: _remap_class_path(v) if k.endswith("_class") else _remap_nested_class_paths(v)
            for k, v in obj.items()
        }
    return obj


class SetBertConfig(PretrainedConfig):
    """Configuration class for SetBERT."""

    model_type = "setbert"
    is_composition = True

    def __init__(
        self,
        sequence_encoder: BaseModelType = None,
        sequence_encoder_class: Optional[BaseModelClassType] = None,
        sequence_encoder_chunk_size: int = 0,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 8,
        num_induce_points: int = 0,
        feedforward_dim: int = 2048,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "gelu",
        pad_token_id: int = 0,
        dropout: float = 0.1,
        num_rep_taxa: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sequence_encoder = _remap_nested_class_paths(sequence_encoder)
        self.sequence_encoder_class = _remap_class_path(sequence_encoder_class)
        self.sequence_encoder_chunk_size = sequence_encoder_chunk_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_induce_points = num_induce_points
        self.feedforward_dim = feedforward_dim
        self.activation = activation
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.num_rep_taxa = num_rep_taxa


class SetBert(DbtkModel):
    """The base SetBERT model."""

    config_class = SetBertConfig

    sub_models = ["sequence_encoder"]

    sequence_encoder: PreTrainedModel

    def __init__(self, config: Optional[Union[SetBertConfig, dict]], **kwargs):
        super().__init__(config, **kwargs)

        self.class_token = nn.Parameter(torch.randn(1, 1, self.config.embed_dim))

        if self.config.num_induce_points > 0:
            raise ValueError("Induced Set Attention Block is not currently supported.")

        # Set Attention Block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.embed_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.feedforward_dim,
            dropout=self.config.dropout,
            activation=self.config.activation,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_layers,
            enable_nested_tensor=False,
        )

    def compute_padding_mask(
        self,
        *,
        sequence_tokens: Optional[torch.Tensor] = None,
        sequence_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the padding mask from token IDs or zero-vector embeddings.

        Args:
            sequence_tokens: Optional sequence tokens.
                Shape: ``[batch_size, sample_size, num_tokens]``
            sequence_embeddings: Optional sequence embeddings.
                Shape: ``[batch_size, sample_size, embed_dim]``

        Returns:
            The padding mask.
                Shape: ``[batch_size, sample_size]``
        """
        if sequence_tokens is None and sequence_embeddings is None:
            raise ValueError(
                "At least one of sequence_tokens or sequence_embeddings must be provided."
            )
        if sequence_tokens is not None and sequence_embeddings is not None:
            raise ValueError(
                "sequence_tokens and sequence_embeddings cannot both be provided."
            )
        if sequence_tokens is not None:
            if self.config.pad_token_id is None:
                raise ValueError("Pad token ID must be specified in the configuration.")
            return torch.all(sequence_tokens == self.config.pad_token_id, -1)
        return torch.all(sequence_embeddings == 0.0, -1)

    def embed_sequences(
        self,
        sequence_tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Embed individual DNA sequences.

        Args:
            sequence_tokens: The sequence tokens to embed.
                Shape: ``[batch_size, sample_size, num_tokens]``
            padding_mask: The padding mask.
                Shape: ``[batch_size, sample_size]``
            chunk_size: Optional chunk size.

        Returns:
            The embedded sequence tokens.
                Shape: ``[batch_size, sample_size, embed_dim]``
        """
        if sequence_tokens.ndim not in (2, 3):
            raise ValueError("Sequence tokens must be a 2D or 3D tensor.")

        non_padding_mask = ~padding_mask

        to_encode = sequence_tokens[non_padding_mask]

        if chunk_size is None or chunk_size == 0:
            chunk_size = self.config.sequence_encoder_chunk_size
        if chunk_size is None or chunk_size == 0:
            chunk_size = to_encode.shape[0]

        # Encode the non-pad sequences through DNABERT in fixed-size chunks,
        # wrapping each chunk in torch.utils.checkpoint to bound activation
        # memory during training. use_reentrant must be False: because the
        # wrapped call only receives integer token IDs (which cannot carry
        # gradients), the reentrant checkpoint silently drops the backward
        # path through the encoder, so DNABERT parameters never receive
        # gradients even when they are nominally trainable.
        pieces = [
            torch.utils.checkpoint.checkpoint(
                self.sequence_encoder,
                to_encode[i:i + chunk_size],
                use_reentrant=False,
            )
            for i in range(0, len(to_encode), chunk_size)
        ]

        # Scatter the encoded chunks back into a dense [batch, set_size, embed_dim]
        # tensor, leaving padded positions as zero. We allocate that destination in
        # the encoder's output dtype so under AMP the assignment is a same-dtype
        # copy; with the default float32 the bf16 encoder output would be up-cast
        # on assignment and then immediately down-cast again by the next autocast
        # region in the SAB stack.
        if pieces:
            encoded = pieces[0] if len(pieces) == 1 else torch.cat(pieces)
            out_dtype = encoded.dtype
        else:
            encoded = None
            out_dtype = torch.float32
        embeddings = torch.zeros(
            (*sequence_tokens.shape[:-1], self.config.embed_dim),
            device=sequence_tokens.device,
            dtype=out_dtype,
        )
        if encoded is not None:
            embeddings[non_padding_mask] = encoded

        return embeddings

    def validate_input_sequences(
        self,
        sequences: Optional[torch.Tensor] = None,
        sequence_tokens: Optional[torch.Tensor] = None,
        sequence_embeddings: Optional[torch.Tensor] = None,
    ):
        """Resolve ``sequences`` (tokens vs. embeddings) into the correct slot.

        Args:
            sequences: Optional sequence tensor. Floating-point tensors are
                interpreted as embeddings; integer tensors as token ids.
            sequence_embeddings: Optional sequence embeddings.
                Shape: ``[batch_size, sample_size, embed_dim]``
            sequence_tokens: Optional sequence tokens.
                Shape: ``[batch_size, sample_size, num_tokens]``

        Returns:
            ``(sequence_tokens, sequence_embeddings)`` with exactly one populated.
        """
        if sequences is not None:
            if sequence_embeddings is not None or sequence_tokens is not None:
                raise ValueError(
                    "Cannot provide both sequences and sequence embeddings/tokens."
                )
            if torch.is_floating_point(sequences):
                sequence_embeddings = sequences
            else:
                sequence_tokens = sequences
        elif sequence_embeddings is None and sequence_tokens is None:
            raise ValueError(
                "At least one of sequence_embeddings or sequence_tokens must be provided."
            )
        elif sequence_tokens is not None and sequence_embeddings is not None:
            raise ValueError(
                "Sequence tokens and sequence embeddings cannot both be provided."
            )
        return sequence_tokens, sequence_embeddings

    def forward(
        self,
        sequences: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        *,
        sequence_embeddings: Optional[torch.Tensor] = None,
        sequence_tokens: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ):
        """Forward pass for the SetBERT model.

        Args:
            sequences: Optional sequence tensor; integer tensors are treated
                as tokens, floating-point tensors as embeddings.
                Shape: ``[batch_size, sample_size, num_tokens]`` or
                ``[batch_size, sample_size, embed_dim]``
            sequence_embeddings: Optional sequence embeddings.
                Shape: ``[batch_size, sample_size, embed_dim]``
            sequence_tokens: Optional sequence tokens.
                Shape: ``[batch_size, sample_size, num_tokens]``
            padding_mask: Optional sequence padding mask.
                Shape: ``[batch_size, sample_size]``

        Returns:
            Dict with:
              - ``class``: transformed ``[CLS]`` token.
                Shape: ``[batch_size, embed_dim]``
              - ``sequences``: transformed per-sequence tokens.
                Shape: ``[batch_size, sample_size, embed_dim]``
        """
        sequence_tokens, sequence_embeddings = self.validate_input_sequences(
            sequences, sequence_tokens, sequence_embeddings,
        )

        if padding_mask is None:
            padding_mask = self.compute_padding_mask(
                sequence_tokens=sequence_tokens,
                sequence_embeddings=sequence_embeddings,
            )

        if sequence_tokens is not None:
            sequence_embeddings = self.embed_sequences(
                sequence_tokens, padding_mask, chunk_size=chunk_size,
            )

        batch_dim_added = False
        if sequence_embeddings.ndim == 2:
            batch_dim_added = True
            sequence_embeddings = sequence_embeddings.unsqueeze(0)
            padding_mask = padding_mask.unsqueeze(0)

        batch_size = sequence_embeddings.shape[0]

        class_tokens = self.class_token.expand(batch_size, 1, -1)
        sequence_embeddings = torch.cat((class_tokens, sequence_embeddings), -2)
        padding_mask = F.pad(padding_mask, (1, 0), value=self.config.pad_token_id)

        output = self.transformer(
            sequence_embeddings,
            src_key_padding_mask=padding_mask,
        )

        transformed_class_embedding = output[:, 0]
        transformed_sequence_embeddings = output[:, 1:]

        if batch_dim_added:
            transformed_class_embedding = transformed_class_embedding.squeeze(0)
            transformed_sequence_embeddings = transformed_sequence_embeddings.squeeze(0)

        return {
            "class": transformed_class_embedding,
            "sequences": transformed_sequence_embeddings,
        }
