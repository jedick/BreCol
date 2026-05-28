"""Shared SetBERT helpers: config validation, model load, per-Run set selection and tokenization.

Used by:
- scripts/build_setbert_run_tensors.py (token tensor cache for train_setbert.py)
- scripts/train_setbert.py (model construction + batch token padding)

The DNABERT backbone inside SetBERT wraps its sequence-encoder calls in
``torch.utils.checkpoint`` (see SetBert.embed_sequences). Because the inputs are
int64 DNABERT token IDs, none of the checkpoint inputs require grad, so PyTorch
emits the warning::

    "None of the inputs have requires_grad=True. Gradients will be None"

This warning is genuinely correct under ``use_reentrant=True`` (where the encoder
parameters silently fail to receive gradients) but is *misleading* under
``use_reentrant=False``, which is what dbtk-setbert now uses: parameter gradients
do flow correctly via saved_tensors_hooks during the backward recompute.

Callers that import this module silence the (now-misleading) warning at import
time so it does not spam the log on every training/inference call.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from cache_operations import (
    count_fasta_records,
    iter_selected_fasta_sequences,
    run_sampling_seed,
    select_row_indices_0based,
    skip_reason,
)

warnings.filterwarnings(
    "ignore",
    message="None of the inputs have requires_grad=True. Gradients will be None",
    category=UserWarning,
    module="torch.utils.checkpoint",
)


# ----- Config -----


def load_setbert_model_section(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the shared ``setbert_model`` block; return a typed dict.

    Holds only the pretrained checkpoint identity and the device default - the two
    things both ``build_setbert_run_tensors.py`` and ``train_setbert.py`` need.
    Training-only knobs (``amp``, ``amp_dtype``, ``weight_init``, etc.) live under
    ``train_setbert``.
    """
    section = cfg.get("setbert_model")
    if not isinstance(section, dict):
        raise SystemExit("defaults.yaml must define a `setbert_model` mapping.")
    try:
        pretrained_repo = str(section["pretrained_repo"]).strip()
        pretrained_revision = str(section["pretrained_revision"]).strip()
    except KeyError as exc:
        raise SystemExit(
            f"setbert_model missing required key: {exc.args[0]!r}"
        ) from exc
    if not pretrained_repo or not pretrained_revision:
        raise SystemExit(
            "setbert_model.pretrained_repo and pretrained_revision must be non-empty."
        )
    device_raw = str(section.get("device") or "").strip().lower()
    return {
        "pretrained_repo": pretrained_repo,
        "pretrained_revision": pretrained_revision,
        "device_raw": device_raw,
    }


def load_setbert_run_tensors_section(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the ``setbert_run_tensors`` block; return a typed dict."""
    section = cfg.get("setbert_run_tensors")
    if not isinstance(section, dict):
        raise SystemExit("defaults.yaml must define a `setbert_run_tensors` mapping.")
    try:
        set_size = int(section["set_size"])
        max_sequence_length = int(section["max_sequence_length"])
        truncation_seed = int(section["truncation_seed"])
    except KeyError as exc:
        raise SystemExit(
            f"setbert_run_tensors missing required key: {exc.args[0]!r}"
        ) from exc
    if set_size <= 0:
        raise SystemExit("setbert_run_tensors.set_size must be a positive integer.")
    if max_sequence_length <= 0:
        raise SystemExit(
            "setbert_run_tensors.max_sequence_length must be a positive integer."
        )
    return {
        "set_size": set_size,
        "max_sequence_length": max_sequence_length,
        "truncation_seed": truncation_seed,
    }


def resolve_device(device_raw: str) -> torch.device:
    """Resolve a YAML device string ('cuda' / 'cpu' / null / 'auto') to a torch.device."""
    s = (device_raw or "").strip().lower()
    if s in ("", "auto", "null", "none"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


# ----- Model / tokenizer loading -----


def load_setbert_model(
    *,
    pretrained_repo: str,
    pretrained_revision: str,
    device: torch.device,
    sequence_encoder_chunk_size: Optional[int] = None,
    eval_mode: bool = True,
    weight_init: str = "pretrained",
) -> Tuple[torch.nn.Module, Any, int, int, int]:
    """Load SetBERT from HF Hub. Return (model, tokenizer, embed_dim, pad_token_id, kmer).

    ``sequence_encoder_chunk_size`` controls how many sequences are pushed through
    the DNABERT encoder per forward chunk inside ``SetBert.embed_sequences`` (also
    the activation-checkpoint granularity during training). When ``None`` (the
    default), the value baked into the released checkpoint config is preserved -
    callers that only need the tokenizer (e.g. ``build_setbert_run_tensors.py``)
    can skip this argument. Pass an explicit int from training callers to control
    memory.

    ``weight_init`` controls backbone weights:

    - ``"pretrained"`` (default): load both the architecture and weights from
      ``pretrained_repo @ pretrained_revision`` via ``SetBert.from_pretrained``.
    - ``"random"``: load only the architecture (``SetBertConfig.from_pretrained``)
      and construct ``SetBert(config)``, which leaves both the DNABERT sequence
      encoder and the SAB transformer (including ``class_token``) with freshly
      randomized weights. Tokenizer / k-mer / padding metadata still come from
      the released config. Callers that need reproducibility should seed
      ``torch.manual_seed`` before calling this function.
    """
    from setbert import SetBert, SetBertConfig

    init_mode = (weight_init or "pretrained").strip().lower()
    if init_mode == "pretrained":
        model = SetBert.from_pretrained(pretrained_repo, revision=pretrained_revision)
    elif init_mode == "random":
        config = SetBertConfig.from_pretrained(
            pretrained_repo, revision=pretrained_revision
        )
        model = SetBert(config)
    else:
        raise ValueError(
            f"load_setbert_model: weight_init must be 'pretrained' or 'random'; "
            f"got {weight_init!r}."
        )
    if sequence_encoder_chunk_size is not None:
        model.config.sequence_encoder_chunk_size = int(sequence_encoder_chunk_size)
    model = model.to(device)
    if eval_mode:
        model.eval()
    tokenizer = model.sequence_encoder.tokenizer
    embed_dim = int(model.config.embed_dim)
    pad_token_id = int(model.config.pad_token_id)
    if pad_token_id != int(tokenizer.vocab["[PAD]"]):
        raise SystemExit(
            "Model pad_token_id and tokenizer [PAD] id disagree; cannot build padding mask."
        )
    kmer = int(getattr(tokenizer, "kmer", 0))
    return model, tokenizer, embed_dim, pad_token_id, kmer


# ----- Per-Run trimming RNG -----


def run_trim_rng(truncation_seed: int, run: str) -> "np.random.Generator":
    """Deterministic per-Run NumPy Generator for random trim-window offsets."""
    return np.random.default_rng(run_sampling_seed(truncation_seed, run))


def trim_sequence(seq: str, *, target_len: int, rng: "np.random.Generator") -> str:
    """Return a window of ``target_len`` chars from ``seq`` (random offset). Shorter seqs unchanged."""
    n = len(seq)
    if n <= target_len:
        return seq
    offset = int(rng.integers(0, n - target_len + 1))
    return seq[offset : offset + target_len]


# ----- Per-Run selection and trimming -----


def select_trimmed_set_for_run(
    fasta_gz: Path,
    *,
    seq_offset: int,
    min_seqs: int,
    set_size: int,
    sample_mode: str,
    sampling_seed: int,
    truncation_seed: int,
    max_sequence_length: int,
    run: str,
) -> Tuple[Optional[np.ndarray], Optional[List[str]], int]:
    """Return (1-based sequence indices, trimmed sequences, n_raw_records) or (None, None, n_raw).

    Sampling matches the without-replacement convention shared by tetramer/embedding caches.
    Sequences longer than ``max_sequence_length`` are randomly cut to ``max_sequence_length``
    (random offset chosen by the per-Run RNG seeded by ``truncation_seed``), matching the
    SetBERT paper's "trim either end to a fixed length" recipe (Suppl. §3.1-3.2). Shorter
    sequences are passed through unchanged. Returns ``n_raw_records`` so callers can record
    provenance even on skip.
    """
    n_raw = count_fasta_records(fasta_gz)
    if skip_reason(n_raw, seq_offset=seq_offset, min_seqs=min_seqs) is not None:
        return None, None, n_raw
    pool_after_offset = n_raw - seq_offset
    if pool_after_offset < set_size:
        return None, None, n_raw

    indices_0 = select_row_indices_0based(
        n_raw,
        seq_offset=seq_offset,
        min_seqs=min_seqs,
        n_max=set_size,
        sample_mode=sample_mode,
        sampling_seed=sampling_seed,
        run=run,
    )
    if indices_0 is None or int(indices_0.size) != set_size:
        return None, None, n_raw

    wanted = {int(i) for i in indices_0.tolist()}
    max_index = max(wanted)

    seq_by_index: Dict[int, str] = {}
    for seq_index, seq in iter_selected_fasta_sequences(fasta_gz, wanted, max_index):
        seq_by_index[seq_index] = seq

    if len(seq_by_index) != set_size:
        return None, None, n_raw

    rng = run_trim_rng(truncation_seed, run)
    index_rows: List[int] = []
    trimmed: List[str] = []
    for idx0 in indices_0.tolist():
        seq = seq_by_index[int(idx0)]
        trimmed.append(trim_sequence(seq, target_len=max_sequence_length, rng=rng))
        index_rows.append(int(idx0) + 1)
    return np.asarray(index_rows, dtype=np.int32), trimmed, n_raw


# ----- Tokenization + padding -----


def tokenize_sequences(
    sequences: Sequence[str],
    tokenizer,
) -> List[List[int]]:
    """Tokenize each DNA string to a list of int token ids via the DNABERT tokenizer."""
    out: List[List[int]] = []
    for seq in sequences:
        tokens = tokenizer(seq)
        if not tokens:
            raise ValueError("DNABERT tokenizer returned an empty token list.")
        out.append([int(t) for t in tokens])
    return out


def pad_set_to_token_len(
    token_rows: Sequence[Sequence[int]],
    *,
    target_token_len: int,
    pad_token_id: int,
) -> np.ndarray:
    """Right-pad each token row to ``target_token_len``. Returns int64 array (n, target_token_len)."""
    n = len(token_rows)
    out = np.full((n, target_token_len), pad_token_id, dtype=np.int64)
    for i, row in enumerate(token_rows):
        L = len(row)
        if L > target_token_len:
            raise ValueError(
                f"Token row length {L} exceeds target {target_token_len}; "
                "check setbert_run_tensors.max_sequence_length / tokenizer."
            )
        out[i, :L] = row
    return out


def build_batch_tokens(
    batch_token_rows: Sequence[Sequence[Sequence[int]]],
    *,
    pad_token_id: int,
) -> np.ndarray:
    """Stack a list of (set_size,) lists of token rows into a (B, set_size, T_max) int array."""
    batch_size = len(batch_token_rows)
    set_sizes = {len(rows) for rows in batch_token_rows}
    if len(set_sizes) != 1:
        raise ValueError("All runs in a batch must share the same set_size.")
    set_size = set_sizes.pop()
    target_token_len = max(
        len(tok) for rows in batch_token_rows for tok in rows
    )
    out = np.full(
        (batch_size, set_size, target_token_len), pad_token_id, dtype=np.int64
    )
    for b, rows in enumerate(batch_token_rows):
        out[b] = pad_set_to_token_len(
            rows, target_token_len=target_token_len, pad_token_id=pad_token_id
        )
    return out
