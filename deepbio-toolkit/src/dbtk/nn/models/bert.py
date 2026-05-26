import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional

from .. import layers
from ...data.vocabularies import Vocabulary
from ..._utils import export

# Vocabularies -------------------------------------------------------------------------------------

@export
class BertVocabulary(Vocabulary):
    def __init__(self, words):
        super().__init__(["[CLS]", "[SEP]", "[MASK]", *words])

# Models -------------------------------------------------------------------------------------------

@export
class BertModel(L.LightningModule):
    def __init__(
        self,
        transformer_encoder: layers.TransformerEncoder,
        tokenizer: Callable,
        vocabulary: Vocabulary
    ):
        super().__init__()
        self.transformer_encoder = transformer_encoder
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.segment_embeddings = nn.Parameter(torch.randn(2, self.transformer_encoder.embed_dim))
        self.token_embeddings = nn.Embedding(
            num_embeddings=len(self.vocabulary),
            embedding_dim=self.embed_dim,
            padding_idx=0)
        self.return_class_embeddings = True
        self.return_item_embeddings = True

    def forward(
        self,
        src_a: torch.Tensor,
        src_b: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False,
        **kwargs
    ):
        # Construct input
        src_a = F.pad(src_a, (1, 0), mode="constant", value=self.vocabulary["[CLS]"])
        src_a = F.pad(src_a, (0, 1), mode="constant", value=self.vocabulary["[SEP]"])
        src = self.token_embeddings(src_a) + self.segment_embeddings[0]
        mask = (src_a == self.vocabulary["[PAD]"])
        if src_b is not None:
            src = torch.cat((src, self.token_embeddings(src_b) + self.segment_embeddings[1]), -2)
            mask = torch.cat((mask, src_b == self.vocabulary["[PAD]"]), -1)

        # Pass through transformer encoder
        output = self.transformer_encoder(
            src,
            src_key_padding_mask=mask,
            average_attention_weights=average_attention_weights,
            return_attention_weights=return_attention_weights,
            **kwargs)

        # Construct outputs
        if isinstance(output, tuple):
            output, *extra = output
        else:
            extra = ()
        # (class_tokens, output_tokens, extra)
        # (class_tokens, (output_tokens_a, output_tokens_b), extra)
        result = ()
        if self.return_class_embeddings:
            result += (output.select(-2, 0),)
        if self.return_item_embeddings:
            output_tokens = (output.narrow(-2, 1, src_a.shape[-1] - 2),)
            if src_b is not None:
                output_tokens = (output_tokens[0], output.narrow(-2, src_a.shape[-1], src_b.shape[-1]))
            result += output_tokens
        result += extra
        if len(result) == 1:
            return result[0]
        return result

    @property
    def embed_dim(self):
        return self.transformer_encoder.embed_dim

    def outputs(self, class_embeddings: Optional[bool] = None, item_embeddings: Optional[bool] = None) -> "BertModel":
        """
        Configure what the model should return.
        """
        if class_embeddings is not None:
            self.return_class_embeddings = class_embeddings
        if item_embeddings is not None:
            self.return_item_embeddings = item_embeddings
        return self


@export
class BertPretrainingModel(L.LightningModule):
    def __init__(self, base: BertModel, num_nsp_classes: Optional[int] = None):
        super().__init__()
        self.base = base
        self.num_nsp_classes = num_nsp_classes
        if self.num_nsp_classes is not None:
            self.predict_nsp = nn.Linear(
                self.embed_dim,
                self.num_nsp_classes)
        self.predict_tokens = nn.Linear(
            self.embed_dim,
            len(self.base.vocabulary))

    def forward(
        self,
        src_a: torch.Tensor,
        src_b: Optional[torch.Tensor] = None
    ):
        if src_b is None:
            return (*self.base(src_a), None)
        return self.base(src_a, src_b)

    def _predict_masked(self, src, masked_tokens, output):
        indices = torch.where(src.flatten() == self.base.vocabulary["[MASK]"])
        predicted = self.predict_tokens(output.flatten(0, -2)[indices])
        loss = F.cross_entropy(predicted, masked_tokens)
        num_correct = torch.sum(torch.argmax(predicted, dim=-1) == masked_tokens)
        return loss, num_correct

    def _step(self, mode, batch):
        (src_a, masked_tokens_a), src_b, nsp = batch
        if src_b is not None:
            src_b, masked_tokens_b = src_b
        class_tokens, output_a, output_b = self(src_a, src_b)
        loss = 0.0
        if nsp is not None:
            predicted = self.predict_nsp(class_tokens)
            nsp_loss = F.cross_entropy(predicted, nsp)
            loss += nsp_loss
            self.log(f"{mode}/nsp_loss", nsp_loss)
        src_a_loss, num_correct = self._predict_masked(src_a, masked_tokens_a, output_a)
        loss += src_a_loss
        n = masked_tokens_a.shape[-1]
        if src_b is not None:
            src_b_loss, num_correct_b = self._predict_masked(src_b, masked_tokens_b, output_b)
            loss += src_b_loss
            num_correct += num_correct_b
            n += masked_tokens_b.shape[-1]
            self.log(f"{mode}/segment_a_loss", src_a_loss)
            self.log(f"{mode}/segment_b_loss", src_b_loss)
        else:
            self.log(f"{mode}/segment_loss", src_a_loss)
        self.log(f"{mode}/loss", loss, prog_bar=True)
        self.log(f"{mode}/reconstruction_accuracy", num_correct.float() / n, prog_bar=True)
        return loss

    def training_step(self, batch):
        return self._step("train", batch)

    def validation_step(self, batch):
        return self._step("val", batch)

    def test_step(self, batch):
        return self._step("test", batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4) # type: ignore

    @property
    def embed_dim(self):
        return self.base.embed_dim
