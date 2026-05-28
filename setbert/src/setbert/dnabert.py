"""Vendored DNABERT model used as SetBERT's per-sequence encoder.

Only the inference / fine-tuning surface required by SetBERT is kept:
``DnaBert`` (the BERT-style encoder) and ``DnaBertForEmbedding`` (the wrapper
the SetBERT checkpoint nests under ``sequence_encoder``). The masked-LM
``DnaBertForPretraining`` and its Lightning training hooks are gone.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from . import layers
from .base import BaseModelClassType, BaseModelType, DbtkModel
from .tokenizers import DnaTokenizer
from .utils import export


@export
class DnaBert(DbtkModel):
    class Config(PretrainedConfig):
        model_type = "dnabert"

        def __init__(
            self,
            kmer: int = 6,
            kmer_stride: int = 1,
            normalize_sequences: bool = True,
            embed_dim: int = 768,
            num_heads: int = 12,
            num_layers: int = 6,
            feedforward_dim: int = 2048,
            activation: str = "gelu",
            max_length: int = 250,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.kmer = kmer
            self.kmer_stride = kmer_stride
            self.normalize_sequences = normalize_sequences
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.feedforward_dim = feedforward_dim
            self.activation = activation
            self.max_length = max_length

    config_class = Config

    def __init__(self, config: Optional[Union[Config, dict]] = None):
        super().__init__(config)

        self.tokenizer = DnaTokenizer(
            kmer=self.config.kmer,
            kmer_stride=self.config.kmer_stride,
            normalize_sequences=self.config.normalize_sequences,
        )

        if isinstance(self.config.activation, str):
            activation = getattr(F, self.config.activation)
        else:
            activation = self.config.activation

        self.transformer = layers.TransformerEncoder(
            layers.TransformerEncoderBlock(
                mha=layers.RelativeMultiHeadAttention(
                    embed_dim=self.config.embed_dim,
                    num_heads=self.config.num_heads,
                    max_length=self.config.max_length,
                ),
                feedforward_dim=self.config.feedforward_dim,
                feedforward_activation=activation,
            ),
            num_layers=self.config.num_layers,
        )

        self.embeddings = nn.Embedding(
            len(self.tokenizer),
            self.config.embed_dim,
            padding_idx=self.tokenizer.vocab["[PAD]"],
        )

    def forward(self, kmers: torch.Tensor):
        kmers = F.pad(kmers, (1, 0), mode="constant", value=self.tokenizer.vocab["[CLS]"])
        tokens = self.embeddings(kmers)

        output = self.transformer(tokens)

        transformed_class_tokens = output[:, 0]
        transformed_kmers = output[:, 1:]

        return {
            "class": transformed_class_tokens,
            "tokens": transformed_kmers,
        }


@export
class DnaBertForEmbedding(DbtkModel):
    class Config(PretrainedConfig):
        is_composition = True

        base: Optional[BaseModelType[DnaBert]] = None
        base_class: Optional[BaseModelClassType[DnaBert]] = "setbert.dnabert.DnaBert"

    config_class = Config
    base_model_prefix = "base"
    sub_models = ["base"]

    base: DnaBert

    def __init__(self, config: Optional[Union[Config, dict]] = None):
        super().__init__(config)

    def forward(self, kmers: torch.Tensor):
        return self.base(kmers)["class"]

    @property
    def tokenizer(self):
        return self.base.tokenizer

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        base = DnaBert.from_pretrained(*args, **kwargs)
        return cls(cls.Config(base=base))

    def save_pretrained(self, *args, **kwargs):
        return self.base.save_pretrained(*args, **kwargs)
