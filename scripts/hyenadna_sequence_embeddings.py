"""Tokenize FASTA sequences and reduce backbone hidden states to per-sequence embeddings."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch

from hyenadna_fasta_data import PAD_TOKEN_ID, make_character_tokenizer, pad_left


def tokenize_dna_sequence(
    seq: str,
    tokenizer,
    *,
    max_tokens: int,
    run: str,
    sequence_index_1based: int,
) -> List[int]:
    """Return DNA token ids; fail if empty or longer than the model context."""
    tokens = tokenizer(seq)["input_ids"]
    dna_tokens = [int(t) for t in tokens if int(t) >= 7]
    if not dna_tokens:
        raise ValueError(
            f"{run} sequence_index={sequence_index_1based}: no tokenizable DNA content."
        )
    if len(dna_tokens) > max_tokens:
        raise ValueError(
            f"{run} sequence_index={sequence_index_1based}: token length {len(dna_tokens)} "
            f"exceeds model context window ({max_tokens}). "
            "Use a longer-context HyenaDNA checkpoint or exclude long sequences."
        )
    return dna_tokens


def batch_token_ids(
    token_rows: Sequence[Sequence[int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Left-pad token rows to a batch; return input_ids and attention_mask (1 = valid)."""
    if not token_rows:
        raise ValueError("batch_token_ids requires at least one sequence")
    target_len = max(len(row) for row in token_rows)
    ids_batch: List[List[int]] = []
    mask_batch: List[List[int]] = []
    for row in token_rows:
        padded, mask = pad_left(list(row), pad_id=PAD_TOKEN_ID, target_len=target_len)
        ids_batch.append(padded)
        mask_batch.append(mask)
    input_ids = torch.tensor(ids_batch, dtype=torch.long)
    attention_mask = torch.tensor(mask_batch, dtype=torch.long)
    return input_ids, attention_mask


def reduce_hidden_states(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    """Map (batch, seq_len, dim) backbone states to (batch, dim)."""
    if mode == "masked_mean":
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denom
    if mode == "last":
        if not torch.all(attention_mask[:, -1].bool()):
            raise RuntimeError(
                "last-token reduction requires valid tokens at the final position "
                "(left-padded batches)."
            )
        return hidden[:, -1, :]
    raise ValueError(f"Unknown sequence_reduce mode {mode!r}; use masked_mean or last.")


def embeddings_from_sequences(
    model: torch.nn.Module,
    sequences: Sequence[str],
    tokenizer,
    *,
    max_tokens: int,
    batch_size: int,
    device: torch.device,
    sequence_reduce: str,
    run: str,
    sequence_indices_1based: Sequence[int],
) -> np.ndarray:
    """Run backbone forward passes and return float32 array (n_sequences, embed_dim)."""
    if len(sequences) != len(sequence_indices_1based):
        raise ValueError("sequences and sequence_indices_1based length mismatch")
    if not sequences:
        return np.empty((0, 0), dtype=np.float32)

    rows: List[np.ndarray] = []
    n = len(sequences)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        token_rows = [
            tokenize_dna_sequence(
                sequences[i],
                tokenizer,
                max_tokens=max_tokens,
                run=run,
                sequence_index_1based=sequence_indices_1based[i],
            )
            for i in range(start, end)
        ]
        input_ids, attention_mask = batch_token_ids(token_rows)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.inference_mode():
            hidden = model(input_ids)
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError(
                f"Expected backbone hidden states (batch, seq, dim); got {type(hidden)}"
            )
        reduced = reduce_hidden_states(hidden, attention_mask, mode=sequence_reduce)
        rows.append(reduced.to(dtype=torch.float32).cpu().numpy())
    return np.vstack(rows)
