#!/usr/bin/env python3
"""
Fine-tune SetBERT with a single linear binary classification head on the transformed
``[CLS]`` token (per the SetBERT paper, Suppl. §3.5).

Reads a feature-only per-Run token tensor cache from paths.setbert_run_tensors_dir/
(scripts/build_setbert_run_tensors.py), joins labels/splits from shared metadata via
shared_utilities.build_run_task_table(task), and reports run-level AUC and F1 on the
val/test/holdout splits.

Single-task only: ``train_setbert.task`` is ``cancer_diagnosis`` or ``cancer_type``.
Loss = ``BCEWithLogits`` on a single output neuron predicting the positive class.

Two-knob backbone freezing (the linear classifier is always trainable):
  - ``train_setbert.freeze_sequence_encoder_epochs``  (DNABERT sequence encoder)
  - ``train_setbert.freeze_sab_epochs``               (SAB transformer + class_token)

When the DNABERT sequence encoder is frozen during an epoch, this script bypasses the
checkpointed encoder path inside ``SetBert.embed_sequences`` (which expects float/
``requires_grad=True`` inputs) by computing token embeddings under ``torch.no_grad()``
once and feeding the result to SAB via the ``sequence_embeddings=`` keyword. That also
saves the wasted forward recomputation that would otherwise occur in the backward pass.

While the encoder is frozen, per-Run DNABERT outputs for the val/test/holdout splits do
not change between epochs, so we encode each scoring-split Run once at the start of the
frozen phase, keep the result as ``[set_size, embed_dim]`` on CPU (~1.5 MB at bf16 for
set_size=1000), and feed those cached embeddings into SAB via ``sequence_embeddings=``
on every subsequent scoring pass. The cache is dropped as soon as the encoder unfreezes
and is never used for the train split. Scoring loops also show per-batch ``tqdm``
progress bars (Val / Test / Holdout) alongside the existing Train bar.

Config: defaults.yaml (train_setbert + setbert + paths) with optional experiments.yaml
``train_setbert.experiments`` overrides selected by ``--expt N`` (1-based). When the
selected experiment row sets ``random_seed`` to a YAML list, the parent process forks
one ``--child-run`` subprocess per seed (results JSON path templated by
``train_setbert.results_json_template``); each subprocess runs a single training.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from setbert_data import load_setbert_model, load_setbert_section, resolve_device
from shared_utilities import (
    binary_auc_from_scores,
    build_run_task_table,
    resolve_repo_path,
)

VALID_TASKS = frozenset(("cancer_diagnosis", "cancer_type"))
VALID_TUNING = frozenset(("auc", "f1"))
VALID_AMP_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}
VALID_CLASS_WEIGHTS = frozenset(("none", "null", "", "balanced", "balanced_sqrt"))
VALID_TRAIN_SAMPLERS = frozenset(("random", "study_balanced"))
VALID_HEAD_TYPES = frozenset(
    (
        "linear",
        "layernorm",
        "cosine",
        "spectral",
        "prototype",
        "pooled",
        "pooled_cosine",
    )
)


# ---------------------------------------------------------------------------
# Config merge (defaults.yaml + experiments.yaml row overrides)
# ---------------------------------------------------------------------------


def _load_train_setbert_section(defaults_path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise SystemExit(f"{defaults_path} must contain a YAML mapping.")
    sec = cfg.get("train_setbert")
    if not isinstance(sec, dict):
        raise SystemExit(f"{defaults_path} must define a train_setbert mapping.")
    return dict(sec)


def merge_train_setbert_config(
    defaults_path: Path,
    experiments_path: Path,
    *,
    expt: int,
) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
    """Return (merged config, optional experiment name, optional results_json template).

    experiments.yaml ``train_setbert`` may set ``results_json_template`` and
    ``sigsegv_retries`` at the section level; those merge into the baseline before
    per-experiment ``overrides``.
    """
    defaults = _load_train_setbert_section(defaults_path)
    experiment_name: Optional[str] = None
    template: Optional[str] = None
    experiments_cfg: Dict[str, Any] = {}
    if experiments_path.is_file():
        experiments_cfg = yaml.safe_load(experiments_path.read_text(encoding="utf-8")) or {}
    section = experiments_cfg.get("train_setbert") or {}
    if section and not isinstance(section, dict):
        raise SystemExit("experiments.yaml train_setbert must be a mapping when present.")
    if isinstance(section, dict):
        raw_tpl = section.get("results_json_template")
        template = str(raw_tpl).strip() if isinstance(raw_tpl, str) else None
        experiments = section.get("experiments") or []
        if not isinstance(experiments, list):
            raise SystemExit("train_setbert.experiments must be a list when present.")
        if expt == 0:
            selected: Dict[str, Any] = {}
        else:
            if not experiments:
                raise SystemExit("No train_setbert.experiments in experiments.yaml.")
            if expt > len(experiments):
                raise SystemExit(
                    f"--expt {expt} out of range; experiments.yaml has "
                    f"{len(experiments)} train_setbert rows."
                )
            row = experiments[expt - 1]
            if not isinstance(row, dict):
                raise SystemExit("train_setbert experiment entry must be a mapping.")
            experiment_name = str(row.get("name") or "").strip() or None
            overrides = row.get("overrides") or {}
            if not isinstance(overrides, dict):
                raise SystemExit("train_setbert experiment overrides must be a mapping.")
            selected = dict(overrides)
        if "sigsegv_retries" in section:
            defaults["sigsegv_retries"] = section["sigsegv_retries"]
        defaults = {**defaults, **selected}
    elif expt != 0:
        raise SystemExit(
            f"--expt {expt} requires experiments.yaml with a train_setbert.experiments list."
        )
    return defaults, experiment_name, template


def _parse_seed_grid(raw: object) -> Optional[List[int]]:
    if raw is None:
        return None
    if isinstance(raw, int):
        return [int(raw)]
    if isinstance(raw, (list, tuple)):
        if not raw:
            raise SystemExit("random_seed grid must not be empty.")
        try:
            return [int(x) for x in raw]
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"random_seed grid must be a list of integers; got {raw!r}."
            ) from exc
    raise SystemExit(
        f"random_seed must be an integer or a YAML list of integers; got {type(raw).__name__}."
    )


def _task_abbrv(task: str) -> str:
    if task == "cancer_diagnosis":
        return "cd"
    if task == "cancer_type":
        return "ct"
    raise SystemExit(f"Unknown task {task!r} (expected cancer_diagnosis or cancer_type).")


def _format_results_template(
    template: str, *, task: str, name: str, seed: int
) -> str:
    return str(template).format(
        task=str(task).strip(),
        task_abbrv=_task_abbrv(task),
        name=str(name),
        seed=int(seed),
    )


# ---------------------------------------------------------------------------
# Grid orchestration (parent forks one child per seed)
# ---------------------------------------------------------------------------


def _clamp_sigsegv_retries(raw: object) -> int:
    try:
        n = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return max(0, min(int(n), 8))


def _grid_child_subprocess_env() -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONFAULTHANDLER"] = "1"
    return env


def _run_grid_child(
    cmd: Sequence[str],
    repo_root: Path,
    *,
    sigsegv_retries: object,
) -> None:
    """Run one grid child process; retry on SIGSEGV up to ``sigsegv_retries`` extra times."""
    retries = _clamp_sigsegv_retries(sigsegv_retries)
    max_attempts = 1 + retries
    env = _grid_child_subprocess_env()
    for attempt in range(1, max_attempts + 1):
        try:
            subprocess.run(list(cmd), check=True, cwd=str(repo_root), env=env)
            return
        except subprocess.CalledProcessError as exc:
            sigsegv = getattr(signal, "SIGSEGV", None)
            is_sigsegv = sigsegv is not None and exc.returncode == -int(sigsegv)
            if is_sigsegv and attempt < max_attempts:
                print(
                    "train_setbert: grid child died with SIGSEGV "
                    f"(attempt {attempt}/{max_attempts}); retrying after short delay.",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(min(float(attempt), 5.0))
                continue
            if is_sigsegv:
                raise SystemExit(
                    "SetBERT training subprocess crashed with SIGSEGV (signal 11). "
                    "This normally comes from the CUDA stack or GPU driver, not from "
                    "Python logic. Reruns often succeed; raise "
                    "train_setbert.sigsegv_retries in experiments.yaml to retry more."
                ) from exc
            raise


# ---------------------------------------------------------------------------
# Per-Run records and dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunRecord:
    run: str
    split: str
    label: int  # 0 (negative) or 1 (positive)
    task_label: str
    study_name: str
    file: Path


def _peek_cache_metadata(pt_path: Path) -> Dict[str, Any]:
    try:
        blob = torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(pt_path, map_location="cpu")
    keys = (
        "set_size",
        "token_len",
        "pad_token_id",
        "kmer",
        "pretrained_repo",
        "pretrained_revision",
    )
    return {k: blob[k] for k in keys if k in blob}


def _pos_neg_for_task(task: str) -> Tuple[str, str]:
    """Return (pos_label, neg_label) using the same convention as hyenadna_multitask."""
    if task == "cancer_diagnosis":
        return ("cancer", "healthy")
    if task == "cancer_type":
        return ("breast_cancer", "colorectal_cancer")
    raise SystemExit(f"Unknown task {task!r}.")


def _build_run_records(
    task: str,
    cache_root: Path,
    *,
    pos_label: str,
    neg_label: str,
    config_path: Path,
) -> Tuple[List[RunRecord], int, Tuple[str, ...]]:
    df = build_run_task_table(task, config_path=config_path)
    df = df.loc[:, ["Run", "split", "task_label", "study_name"]].drop_duplicates(
        subset=["Run"]
    ).reset_index(drop=True)
    out: List[RunRecord] = []
    missing: List[str] = []
    for _, row in df.iterrows():
        run = str(row["Run"]).strip()
        pt_path = cache_root / f"{run}.pt"
        if not pt_path.is_file():
            missing.append(run)
            continue
        task_label = str(row["task_label"]).strip()
        if task_label == pos_label:
            y = 1
        elif task_label == neg_label:
            y = 0
        else:
            raise SystemExit(
                f"Unexpected task_label {task_label!r} for run {run!r}; "
                f"expected {pos_label!r} or {neg_label!r}."
            )
        out.append(
            RunRecord(
                run=run,
                split=str(row["split"]).strip(),
                label=int(y),
                task_label=task_label,
                study_name=str(row["study_name"]).strip(),
                file=pt_path,
            )
        )
    return out, len(missing), tuple(sorted(missing)[:5])


class SetBertRunDataset(Dataset):
    def __init__(self, entries: Sequence[RunRecord], *, requested_set_size: int):
        self.entries = list(entries)
        self.requested_set_size = int(requested_set_size)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        e = self.entries[idx]
        try:
            blob = torch.load(e.file, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(e.file, map_location="cpu")
        cache_set_size = int(blob["set_size"])
        if cache_set_size < self.requested_set_size:
            raise SystemExit(
                f"Cache {e.file.name} has set_size={cache_set_size} < requested "
                f"{self.requested_set_size}; rebuild build_setbert_run_tensors."
            )
        input_ids: torch.Tensor = blob["input_ids"][: self.requested_set_size, :].long()
        return {
            "input_ids": input_ids,
            "label": int(e.label),
            "run": e.run,
        }


def _make_collate(pad_token_id: int) -> Callable[[List[Dict[str, object]]], Dict[str, object]]:
    """Right-pad each Run's token rows to the batch's max token length."""

    def collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
        token_lens = [b["input_ids"].shape[1] for b in batch]
        set_size = batch[0]["input_ids"].shape[0]
        T_max = max(token_lens)
        out_ids = torch.full(
            (len(batch), set_size, T_max), pad_token_id, dtype=torch.long
        )
        for i, b in enumerate(batch):
            t = b["input_ids"].shape[1]
            out_ids[i, :, :t] = b["input_ids"]
        labels = torch.tensor(
            [float(b["label"]) for b in batch], dtype=torch.float32
        )
        runs = [str(b["run"]) for b in batch]
        return {"input_ids": out_ids, "label": labels, "run": runs}

    return collate


def _study_sampler_weights(entries: Sequence[RunRecord]) -> torch.Tensor:
    counts: Dict[str, int] = defaultdict(int)
    for e in entries:
        counts[e.study_name] += 1
    w = [1.0 / counts[e.study_name] for e in entries]
    return torch.tensor(w, dtype=torch.double)


# ---------------------------------------------------------------------------
# Model wrapper: SetBERT + configurable binary classification head (BCE)
# ---------------------------------------------------------------------------


class SetBertHead(nn.Module):
    """Binary classification head over a transformed SetBERT batch.

    Selected via ``head_type`` (see :data:`VALID_HEAD_TYPES`). All variants
    accept the three tensors emitted by ``SetBert`` for a batch:

    - ``cls``: transformed ``[CLS]`` token, shape ``[B, embed_dim]``
    - ``set_tokens``: transformed per-sequence tokens, shape ``[B, set_size, embed_dim]``
    - ``pad_mask``: boolean padding mask for ``set_tokens``, shape ``[B, set_size]``
      (``True`` at padded positions)

    Non-pooled heads consume only ``cls``; the ``pooled`` head additionally uses
    masked mean and max over ``set_tokens``. ``dropout`` is applied to the head
    input (after ``LayerNorm`` when present); for the ``linear`` head it matches
    the original SetBERT-paper recipe (dropout on the raw ``[CLS]``).
    """

    def __init__(self, embed_dim: int, *, kind: str, dropout: float = 0.0):
        super().__init__()
        kind = str(kind).strip().lower()
        if kind not in VALID_HEAD_TYPES:
            raise SystemExit(
                f"Unknown head_type {kind!r} "
                f"(use one of {sorted(VALID_HEAD_TYPES)})."
            )
        self.kind = kind
        self.dropout = nn.Dropout(float(dropout))
        d = int(embed_dim)
        if kind == "linear":
            self.classifier = nn.Linear(d, 1)
        elif kind == "layernorm":
            self.norm = nn.LayerNorm(d)
            self.classifier = nn.Linear(d, 1)
        elif kind == "cosine":
            self.norm = nn.LayerNorm(d)
            self.weight = nn.Parameter(torch.randn(d) * (d ** -0.5))
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(10.0)))
        elif kind == "spectral":
            self.norm = nn.LayerNorm(d)
            self.classifier = nn.utils.parametrizations.spectral_norm(
                nn.Linear(d, 1)
            )
        elif kind == "prototype":
            self.norm = nn.LayerNorm(d)
            self.prototypes = nn.Parameter(torch.randn(2, d) * (d ** -0.5))
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(10.0)))
        elif kind == "pooled":
            self.norm = nn.LayerNorm(3 * d)
            self.classifier = nn.Linear(3 * d, 1)
        elif kind == "pooled_cosine":
            self.norm = nn.LayerNorm(3 * d)
            self.weight = nn.Parameter(
                torch.randn(3 * d) * ((3 * d) ** -0.5)
            )
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(10.0)))
        else:  # pragma: no cover - guarded above
            raise SystemExit(f"Unknown head_type {kind!r}.")

    def _pool_set(
        self, set_tokens: torch.Tensor, pad_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = (~pad_mask).unsqueeze(-1).to(set_tokens.dtype)
        denom = valid.sum(dim=1).clamp_min(1.0)
        mean_pool = (set_tokens * valid).sum(dim=1) / denom
        neg_inf = torch.finfo(set_tokens.dtype).min
        masked = set_tokens.masked_fill(pad_mask.unsqueeze(-1), neg_inf)
        max_pool = masked.max(dim=1).values
        return mean_pool, max_pool

    def forward(
        self,
        cls: torch.Tensor,
        set_tokens: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.kind == "linear":
            return self.classifier(self.dropout(cls)).squeeze(-1)
        if self.kind == "layernorm":
            return self.classifier(self.dropout(self.norm(cls))).squeeze(-1)
        if self.kind == "spectral":
            return self.classifier(self.dropout(self.norm(cls))).squeeze(-1)
        if self.kind == "cosine":
            x = F.normalize(self.dropout(self.norm(cls)), dim=-1)
            w = F.normalize(self.weight, dim=-1)
            return self.log_temperature.exp() * (x @ w)
        if self.kind == "prototype":
            x = F.normalize(self.dropout(self.norm(cls)), dim=-1)
            p = F.normalize(self.prototypes, dim=-1)
            sims = self.log_temperature.exp() * (x @ p.T)
            return sims[:, 0] - sims[:, 1]
        if self.kind == "pooled":
            mean_pool, max_pool = self._pool_set(set_tokens, pad_mask)
            feat = torch.cat([cls, mean_pool, max_pool], dim=-1)
            return self.classifier(self.dropout(self.norm(feat))).squeeze(-1)
        if self.kind == "pooled_cosine":
            mean_pool, max_pool = self._pool_set(set_tokens, pad_mask)
            feat = torch.cat([cls, mean_pool, max_pool], dim=-1)
            x = F.normalize(self.dropout(self.norm(feat)), dim=-1)
            w = F.normalize(self.weight, dim=-1)
            return self.log_temperature.exp() * (x @ w)
        raise SystemExit(f"Unknown head_type {self.kind!r}.")


class SetBertBinaryClassifier(nn.Module):
    """Wrap ``SetBert`` and predict via a configurable :class:`SetBertHead`.

    When the DNABERT sequence encoder is frozen during training, the forward pass
    computes per-sequence embeddings under ``torch.no_grad()`` (skipping the
    activation-checkpointed encoder path inside ``SetBert.embed_sequences``) and
    routes them into ``SetBert`` via ``sequence_embeddings=...``. This avoids the
    PyTorch checkpoint warning about non-grad integer inputs and removes the
    wasted forward recomputation that would otherwise happen during backward.
    """

    def __init__(
        self,
        base_model: nn.Module,
        embed_dim: int,
        *,
        head_type: str = "linear",
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.head = SetBertHead(
            int(embed_dim), kind=str(head_type), dropout=float(head_dropout)
        )

    def _encoder_trainable(self) -> bool:
        return any(
            p.requires_grad for p in self.base_model.sequence_encoder.parameters()
        )

    def _embed_under_no_grad(
        self, sequence_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_mask = self.base_model.compute_padding_mask(sequence_tokens=sequence_tokens)
        non_pad = ~pad_mask
        to_encode = sequence_tokens[non_pad]
        embed_dim = int(self.base_model.config.embed_dim)
        chunk = int(self.base_model.config.sequence_encoder_chunk_size)
        if chunk <= 0:
            chunk = max(1, int(to_encode.shape[0]) or 1)
        if int(to_encode.shape[0]) == 0:
            embeddings = torch.zeros(
                (*sequence_tokens.shape[:-1], embed_dim),
                device=sequence_tokens.device,
                dtype=torch.float32,
            )
            return embeddings, pad_mask
        with torch.no_grad():
            pieces: List[torch.Tensor] = []
            for i in range(0, int(to_encode.shape[0]), chunk):
                pieces.append(self.base_model.sequence_encoder(to_encode[i : i + chunk]))
            cat = torch.cat(pieces)
        embeddings = torch.zeros(
            (*sequence_tokens.shape[:-1], embed_dim),
            device=sequence_tokens.device,
            dtype=cat.dtype,
        )
        embeddings[non_pad] = cat
        return embeddings, pad_mask

    def forward(self, sequence_tokens: torch.Tensor) -> torch.Tensor:
        if self.training and not self._encoder_trainable():
            embeddings, pad_mask = self._embed_under_no_grad(sequence_tokens)
            out = self.base_model(
                sequence_embeddings=embeddings, padding_mask=pad_mask
            )
        else:
            pad_mask = self.base_model.compute_padding_mask(
                sequence_tokens=sequence_tokens
            )
            out = self.base_model(
                sequence_tokens=sequence_tokens, padding_mask=pad_mask
            )
        return self.head(out["class"], out["sequences"], pad_mask)


def _set_backbone_requires_grad(
    model: SetBertBinaryClassifier,
    *,
    train_encoder: bool,
    train_sab: bool,
) -> None:
    for p in model.base_model.sequence_encoder.parameters():
        p.requires_grad = bool(train_encoder)
    for p in model.base_model.transformer.parameters():
        p.requires_grad = bool(train_sab)
    model.base_model.class_token.requires_grad = bool(train_sab)
    for p in model.head.parameters():
        p.requires_grad = True


def _make_optimizer(
    model: SetBertBinaryClassifier,
    *,
    lr: float,
    weight_decay: float,
    backbone_lr_mult: float,
) -> torch.optim.Optimizer:
    backbone_params: List[nn.Parameter] = []
    for p in model.base_model.sequence_encoder.parameters():
        if p.requires_grad:
            backbone_params.append(p)
    for p in model.base_model.transformer.parameters():
        if p.requires_grad:
            backbone_params.append(p)
    if model.base_model.class_token.requires_grad:
        backbone_params.append(model.base_model.class_token)
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    groups: List[Dict[str, object]] = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr * float(backbone_lr_mult)})
    if head_params:
        groups.append({"params": head_params, "lr": lr})
    if not groups:
        raise SystemExit("No trainable parameters for optimizer.")
    return torch.optim.AdamW(groups, weight_decay=float(weight_decay))


# ---------------------------------------------------------------------------
# AMP, loss, scoring
# ---------------------------------------------------------------------------


def _resolve_amp_config(
    merged: Mapping[str, object], device: torch.device
) -> Tuple[bool, torch.dtype, str, bool]:
    amp_requested = bool(merged.get("amp", False))
    amp_dtype_raw = str(merged.get("amp_dtype", "bfloat16")).strip().lower()
    if amp_dtype_raw not in VALID_AMP_DTYPES:
        raise SystemExit("train_setbert.amp_dtype must be 'float16' or 'bfloat16'.")
    amp_dtype = VALID_AMP_DTYPES[amp_dtype_raw]
    use_grad_scaler = amp_dtype is torch.float16
    if not amp_requested:
        return False, amp_dtype, amp_dtype_raw, use_grad_scaler
    if device.type != "cuda":
        print("AMP requested but CUDA is unavailable; running in float32.", flush=True)
        return False, amp_dtype, amp_dtype_raw, use_grad_scaler
    return True, amp_dtype, amp_dtype_raw, use_grad_scaler


def _amp_autocast(
    device: torch.device, *, amp_enabled: bool, amp_dtype: torch.dtype
) -> ContextManager[object]:
    if amp_enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def _compute_pos_weight(
    train_entries: Sequence[RunRecord], mode: str, device: torch.device
) -> Optional[torch.Tensor]:
    m = (mode or "none").strip().lower()
    if m in ("none", "null", ""):
        return None
    n_pos = sum(1 for e in train_entries if e.label == 1)
    n_neg = sum(1 for e in train_entries if e.label == 0)
    if n_pos == 0 or n_neg == 0:
        return None
    bal = float(n_neg) / float(n_pos)
    if m == "balanced":
        val = bal
    elif m == "balanced_sqrt":
        val = bal ** 0.5
    else:
        raise SystemExit(
            f"Unknown class_weight {mode!r} (use none, balanced, or balanced_sqrt)."
        )
    return torch.tensor([val], dtype=torch.float32, device=device)


def _train_one_epoch(
    model: SetBertBinaryClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    scaler: Optional[torch.amp.GradScaler],
    pos_weight: Optional[torch.Tensor],
) -> float:
    model.train()
    total = 0.0
    n_batches = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with _amp_autocast(device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
            logits = model(input_ids)
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight
            )
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += float(loss.item())
        n_batches += 1
    return total / max(n_batches, 1)


@torch.no_grad()
def _populate_encoder_cache(
    model: SetBertBinaryClassifier,
    loader: Optional[DataLoader],
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    desc: str,
) -> Dict[str, torch.Tensor]:
    """One-time DNABERT pass; return ``{Run -> [set_size, embed_dim] CPU tensor}``.

    The cache is only valid while the sequence encoder is frozen; the caller is
    responsible for discarding it the moment the encoder is unfrozen. Stored in
    the encoder's output dtype (bf16 under AMP, fp32 otherwise), on CPU to keep
    GPU memory free for SAB activations.
    """
    if loader is None:
        return {}
    model.eval()
    cache: Dict[str, torch.Tensor] = {}
    for batch in tqdm(loader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        runs = list(batch["run"])
        with _amp_autocast(device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
            embeddings, _ = model._embed_under_no_grad(input_ids)
        for i, run in enumerate(runs):
            cache[str(run)] = embeddings[i].detach().to("cpu").clone()
    return cache


def _forward_from_cache(
    model: SetBertBinaryClassifier,
    runs: Sequence[str],
    cache: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Stack cached encoder embeddings and run SAB + classification head only."""
    stack = torch.stack([cache[str(r)] for r in runs], dim=0).to(
        device, non_blocking=True
    )
    pad_mask = torch.zeros(stack.shape[:2], dtype=torch.bool, device=device)
    out = model.base_model(sequence_embeddings=stack, padding_mask=pad_mask)
    return model.head(out["class"], out["sequences"], pad_mask)


@torch.no_grad()
def _score_loader(
    model: SetBertBinaryClassifier,
    loader: Optional[DataLoader],
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    desc: str = "Score",
    encoder_cache: Optional[Dict[str, torch.Tensor]] = None,
) -> np.ndarray:
    """Forward-only scoring pass over a pre-built loader (shuffle=False).

    Returns positive-class probabilities in loader iteration order. The caller is
    responsible for keeping the entries list in the same order as the loader.
    When ``encoder_cache`` is provided, the DNABERT pass is skipped and the
    cached per-Run embeddings are fed straight into SAB.
    """
    if loader is None:
        return np.empty(0, dtype=np.float64)
    model.eval()
    scores: List[float] = []
    for batch in tqdm(loader, desc=desc, leave=False):
        with _amp_autocast(device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
            if encoder_cache is not None:
                logits = _forward_from_cache(
                    model, list(batch["run"]), encoder_cache, device
                )
            else:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                logits = model(input_ids)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        scores.extend(float(p) for p in probs.tolist())
    return np.asarray(scores, dtype=np.float64)


def _binary_metrics(
    entries: Sequence[RunRecord],
    scores: np.ndarray,
    *,
    pos_label: str,
    neg_label: str,
) -> Tuple[float, float]:
    if len(entries) == 0:
        return float("nan"), float("nan")
    y_true = np.array([e.task_label for e in entries], dtype=object)
    auc = binary_auc_from_scores(y_true, scores, positive_label=pos_label)
    y_pred = np.where(scores >= 0.5, pos_label, neg_label)
    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=pos_label,
        average="binary",
        zero_division=0,
    )
    return float(auc), float(f1)


@torch.no_grad()
def _eval_val_full(
    model: SetBertBinaryClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    pos_weight: Optional[torch.Tensor],
    desc: str = "Val",
    encoder_cache: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[float, np.ndarray]:
    """Fused val pass: returns (val_loss, val_scores) from a single forward pass.

    Previously the script did two complete passes over the val set per epoch (one
    for the loss and one for the scores). The forward pass dominates inference
    cost, so collapsing them halves the val-side work per epoch.
    When ``encoder_cache`` is provided, the DNABERT pass is skipped and the
    cached per-Run embeddings are fed straight into SAB.
    """
    model.eval()
    total = 0.0
    n_batches = 0
    scores: List[float] = []
    for batch in tqdm(loader, desc=desc, leave=False):
        labels = batch["label"].to(device, non_blocking=True)
        with _amp_autocast(device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
            if encoder_cache is not None:
                logits = _forward_from_cache(
                    model, list(batch["run"]), encoder_cache, device
                )
            else:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                logits = model(input_ids)
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight
            )
        total += float(loss.item())
        n_batches += 1
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        scores.extend(float(p) for p in probs.tolist())
    return total / max(n_batches, 1), np.asarray(scores, dtype=np.float64)


# ---------------------------------------------------------------------------
# Output writers (results.json, _training.json, _predictions.csv)
# ---------------------------------------------------------------------------


def _float_or_none(x: float) -> Optional[float]:
    return float(x) if x == x else None


def _results_json_out_path(
    repo_root: Path, raw: Optional[object], *, task: str
) -> Optional[Path]:
    if raw is None:
        return None
    if raw in ("", "null"):
        defaults_path = repo_root / "defaults.yaml"
        try:
            paths_cfg = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))["paths"]
            scratch_key = paths_cfg["results_scratch_dir"]
        except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
            raise SystemExit(
                f"Cannot read paths.results_scratch_dir from {defaults_path}: {exc}"
            ) from exc
        scratch_base = resolve_repo_path(repo_root, scratch_key)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return scratch_base / f"train_setbert_{task}_{ts}.json"
    p = Path(str(raw).strip()).expanduser()
    return p if p.is_absolute() else repo_root / p


def _training_log_path_from_results(path: Path) -> Path:
    return path.with_name(f"{path.stem}_training.json")


def _write_training_log(path: Path, epoch_rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: List[Dict[str, object]] = []
    for row in epoch_rows:
        payload.append(
            {
                "epoch": int(row["epoch"]),
                "learning_rate": float(row["learning_rate"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": _float_or_none(float(row["val_loss"])),
                "val_f1": _float_or_none(float(row["val_f1"])),
                "test_f1": _float_or_none(float(row["test_f1"])),
                "holdout_f1": _float_or_none(float(row["holdout_f1"])),
                "val_auc": _float_or_none(float(row["val_auc"])),
                "test_auc": _float_or_none(float(row["test_auc"])),
                "holdout_auc": _float_or_none(float(row["holdout_auc"])),
            }
        )
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_predictions_csv(
    path: Path,
    entries: Sequence[RunRecord],
    scores: np.ndarray,
    *,
    pos_label: str,
    neg_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Run", "task_label", "predicted_label", "positive_score"],
            lineterminator="\n",
        )
        writer.writeheader()
        for e, s in zip(entries, scores):
            pred = pos_label if float(s) >= 0.5 else neg_label
            writer.writerow(
                {
                    "Run": str(e.run),
                    "task_label": str(e.task_label),
                    "predicted_label": str(pred),
                    "positive_score": f"{float(s):.10f}",
                }
            )


def _write_results_json(
    path: Path,
    *,
    merged: Mapping[str, Any],
    cache_info: Mapping[str, Any],
    task: str,
    pos_label: str,
    neg_label: str,
    best_epoch: int,
    tuning_metric: str,
    best_tuning_score: float,
    best_val_auc: float,
    best_val_f1: float,
    test_auc: float,
    test_f1: float,
    holdout_auc: float,
    holdout_f1: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": Path(__file__).name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": dict(merged),
        "data": {"cache": dict(cache_info)},
        "task": {
            "task": str(task),
            "pos_label": str(pos_label),
            "neg_label": str(neg_label),
        },
        "tuning": {
            "split": "validation",
            "metric": str(tuning_metric),
            "best_epoch": int(best_epoch),
            "best_score": _float_or_none(best_tuning_score),
            "val_auc": _float_or_none(best_val_auc),
            "val_f1": _float_or_none(best_val_f1),
        },
        "metrics": {
            "test": {
                "auc": _float_or_none(test_auc),
                "f1": _float_or_none(test_f1),
            },
            "holdout": {
                "auc": _float_or_none(holdout_auc),
                "f1": _float_or_none(holdout_f1),
            },
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def _parse_argv(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expt",
        type=int,
        default=None,
        help="Optional train_setbert experiment index from experiments.yaml (1-based). "
        "Omit or use 0 for defaults.yaml only.",
    )
    parser.add_argument(
        "--results-json",
        type=str,
        nargs="?",
        const="",
        default=argparse.SUPPRESS,
        help=(
            "Override results JSON path. With no path, writes under results/scratch/ "
            "as train_setbert_<task>_<utc>.json. Omit entirely to use YAML only."
        ),
    )
    parser.add_argument(
        "--override-random-seed",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--child-run",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.expt is not None and args.expt < 0:
        raise SystemExit("--expt must be >= 0.")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    cli = _parse_argv(argv)
    expt = int(cli.expt) if cli.expt is not None else 0
    repo_root = Path(__file__).resolve().parent.parent
    defaults_path = repo_root / "defaults.yaml"
    experiments_path = repo_root / "experiments.yaml"

    merged, exp_name, template = merge_train_setbert_config(
        defaults_path, experiments_path, expt=expt
    )
    if hasattr(cli, "results_json"):
        rj = cli.results_json
        if rj == "" and expt > 0 and not cli.child_run:
            # Parent of a grid; let the template/expansion logic decide per child.
            pass
        elif rj == "":
            merged = {**merged, "results_json": ""}
        else:
            merged = {**merged, "results_json": rj}
    if cli.override_random_seed is not None:
        merged = {**merged, "random_seed": int(cli.override_random_seed)}

    # ----- Grid orchestration (parent only) ---------------------------------
    if expt > 0 and not cli.child_run:
        experiments_cfg = yaml.safe_load(experiments_path.read_text(encoding="utf-8")) or {}
        section = experiments_cfg.get("train_setbert") or {}
        expts = section.get("experiments") or []
        if not isinstance(expts, list) or expt > len(expts):
            raise SystemExit(
                f"--expt {expt} out of range; experiments.yaml has "
                f"{len(expts) if isinstance(expts, list) else 0} train_setbert rows."
            )
        row = expts[expt - 1]
        if not isinstance(row, dict):
            raise SystemExit("train_setbert experiment entry must be a mapping.")
        overrides = row.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise SystemExit("train_setbert experiment overrides must be a mapping.")
        seed_grid = _parse_seed_grid(overrides.get("random_seed")) or [int(merged["random_seed"])]
        task_v = str(merged["task"]).strip()
        if task_v not in VALID_TASKS:
            raise SystemExit(
                f"Unknown train_setbert.task {task_v!r} "
                "(use cancer_diagnosis or cancer_type)."
            )

        if len(seed_grid) > 1:
            if not (isinstance(template, str) and template.strip()):
                raise SystemExit(
                    "Grid experiment requires train_setbert.results_json_template in experiments.yaml."
                )
            if not exp_name:
                raise SystemExit("Grid experiment requires a non-empty train_setbert experiment name.")
            if hasattr(cli, "results_json") and str(cli.results_json) != "":
                raise SystemExit(
                    "--results-json cannot be used with multi-run train_setbert grid experiments."
                )
            total = len(seed_grid)
            completed = 0
            skipped = 0
            print(
                f"Running EXPT={expt} grid: {total} seed(s) for task={task_v}.",
                flush=True,
            )
            for seed in seed_grid:
                out_rel = _format_results_template(
                    template, task=task_v, name=str(exp_name), seed=int(seed)
                )
                out_path = resolve_repo_path(repo_root, out_rel)
                if out_path.is_file():
                    skipped += 1
                    print(
                        f"Skipping EXPT={expt} seed={seed}: "
                        f"{out_path} already exists.",
                        flush=True,
                    )
                    continue
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--expt",
                    str(expt),
                    "--child-run",
                    "--override-random-seed",
                    str(seed),
                    "--results-json",
                    str(out_path),
                ]
                print(
                    f"Launching EXPT={expt} seed={seed} -> {out_path}",
                    flush=True,
                )
                _run_grid_child(
                    cmd,
                    repo_root,
                    sigsegv_retries=merged.get("sigsegv_retries"),
                )
                completed += 1
            print(
                f"Finished EXPT={expt} grid: launched={completed}, "
                f"skipped_existing={skipped}, total={total}.",
                flush=True,
            )
            return 0

        # Single-seed expt: maybe still apply template to choose the output path.
        if (
            isinstance(template, str)
            and template.strip()
            and exp_name
            and merged.get("results_json") in (None, "null")
        ):
            out_rel = _format_results_template(
                template, task=task_v, name=str(exp_name), seed=int(merged["random_seed"])
            )
            merged = {**merged, "results_json": out_rel}

    # ----- Single-run training ----------------------------------------------
    task = str(merged["task"]).strip()
    if task not in VALID_TASKS:
        raise SystemExit(
            f"Unknown train_setbert.task {task!r} "
            "(use cancer_diagnosis or cancer_type)."
        )
    tuning_metric = str(merged.get("tuning_metric") or "auc").strip().lower()
    if tuning_metric not in VALID_TUNING:
        raise SystemExit(
            f"train_setbert.tuning_metric must be one of {sorted(VALID_TUNING)}; "
            f"got {tuning_metric!r}."
        )
    train_sampler = str(merged.get("train_sampler") or "random").strip().lower()
    if train_sampler not in VALID_TRAIN_SAMPLERS:
        raise SystemExit(
            f"Unknown train_sampler {train_sampler!r} (use random or study_balanced)."
        )
    class_weight_mode = str(merged.get("class_weight") or "none").strip().lower()
    if class_weight_mode not in VALID_CLASS_WEIGHTS:
        raise SystemExit(
            f"Unknown class_weight {class_weight_mode!r} "
            "(use none, balanced, or balanced_sqrt)."
        )

    defaults_cfg = yaml.safe_load(defaults_path.read_text(encoding="utf-8")) or {}
    if not isinstance(defaults_cfg, dict):
        raise SystemExit(f"{defaults_path} must contain a YAML mapping.")
    paths_cfg = defaults_cfg.get("paths")
    if not isinstance(paths_cfg, dict):
        raise SystemExit(f"{defaults_path} must define paths as a mapping.")
    setbert_section = load_setbert_section(defaults_cfg)
    run_tensors_root = resolve_repo_path(
        repo_root, str(paths_cfg["setbert_run_tensors_dir"]).strip()
    )
    if not run_tensors_root.is_dir():
        raise SystemExit(
            f"Missing SetBERT run tensors directory: {run_tensors_root}. "
            "Run: python scripts/build_setbert_run_tensors.py"
        )

    requested_set_size = int(merged["set_size"])
    if requested_set_size <= 0:
        raise SystemExit("train_setbert.set_size must be > 0.")
    if requested_set_size > int(setbert_section["set_size"]):
        raise SystemExit(
            f"train_setbert.set_size ({requested_set_size}) exceeds "
            f"setbert.set_size ({setbert_section['set_size']}); rebuild the run tensor "
            "cache with a larger setbert.set_size."
        )

    pos_label, neg_label = _pos_neg_for_task(task)
    all_records, n_missing_cache, missing_cache_examples = _build_run_records(
        task,
        run_tensors_root,
        pos_label=pos_label,
        neg_label=neg_label,
        config_path=defaults_path,
    )
    if n_missing_cache:
        ex = ", ".join(missing_cache_examples)
        print(
            f"train_setbert: skipping {n_missing_cache} metadata run(s) with no "
            f"{run_tensors_root.name}/*.pt tensor (examples: {ex}).",
            flush=True,
        )
    by_split: Dict[str, List[RunRecord]] = defaultdict(list)
    for r in all_records:
        by_split[r.split].append(r)
    train_entries = by_split["train"]
    val_entries = by_split["val"]
    test_entries = by_split["test"]
    holdout_entries = by_split["holdout"]
    if not train_entries:
        raise SystemExit("No training runs in cache (split=train). Build the run tensor cache first.")
    if not val_entries:
        raise SystemExit("No validation runs in cache.")
    if not test_entries:
        raise SystemExit("No test runs in cache.")

    # Verify cache compatibility (one peek; assume all runs share the same checkpoint).
    sample_meta = _peek_cache_metadata(train_entries[0].file)
    if int(sample_meta.get("set_size", 0)) < requested_set_size:
        raise SystemExit(
            f"Cache set_size ({sample_meta.get('set_size')}) < train_setbert.set_size "
            f"({requested_set_size}); rebuild run tensors."
        )
    sample_repo = str(sample_meta.get("pretrained_repo", "")).strip()
    sample_rev = str(sample_meta.get("pretrained_revision", "")).strip()
    if sample_repo and sample_repo != setbert_section["pretrained_repo"]:
        raise SystemExit(
            f"Cache pretrained_repo {sample_repo!r} does not match "
            f"setbert.pretrained_repo {setbert_section['pretrained_repo']!r}."
        )
    if sample_rev and sample_rev != setbert_section["pretrained_revision"]:
        raise SystemExit(
            f"Cache pretrained_revision {sample_rev!r} does not match "
            f"setbert.pretrained_revision {setbert_section['pretrained_revision']!r}."
        )

    seed = int(merged["random_seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_s = str(merged.get("device") or "").strip().lower()
    if not device_s or device_s == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_s)

    # Free speedup on Ampere+ for any fp32 matmuls outside the bf16 autocast
    # region (does not affect bf16 numerics).
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    amp_enabled, amp_dtype, amp_dtype_name, amp_use_scaler = _resolve_amp_config(
        merged, device
    )

    exp_label = str(exp_name).strip() if exp_name is not None else ""
    if expt > 0 and exp_label:
        print(f"\nExperiment: EXPT={expt} name={exp_label}", flush=True)
    elif expt > 0:
        print(f"\nExperiment: EXPT={expt}", flush=True)
    else:
        print("\nExperiment: EXPT=0 (defaults.yaml config)", flush=True)
    print(
        f"\nSetBERT train | task={task} pos={pos_label!r} neg={neg_label!r} | "
        f"set_size={requested_set_size} batch_size={int(merged['batch_size'])} | "
        f"precision={'amp(' + amp_dtype_name + ')' if amp_enabled else 'fp32'} "
        f"device={device}",
        flush=True,
    )

    chunk_size = int(merged["sequence_encoder_chunk_size"])
    print(
        f"Loading SetBERT {setbert_section['pretrained_repo']}@"
        f"{setbert_section['pretrained_revision']} on {device} "
        f"(sequence_encoder_chunk_size={chunk_size})",
        flush=True,
    )
    try:
        base_model, _tokenizer, embed_dim, pad_token_id, kmer = load_setbert_model(
            pretrained_repo=setbert_section["pretrained_repo"],
            pretrained_revision=setbert_section["pretrained_revision"],
            sequence_encoder_chunk_size=chunk_size,
            device=device,
            eval_mode=False,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Failed to load SetBERT model: {exc}", file=sys.stderr)
        return 1
    print(
        f"  embed_dim={embed_dim}, kmer={kmer}, pad_token_id={pad_token_id}",
        flush=True,
    )

    head_dropout = float(merged.get("head_dropout") or 0.0)
    if head_dropout < 0.0:
        raise SystemExit("train_setbert.head_dropout must be >= 0.")
    head_type = str(merged.get("head_type") or "linear").strip().lower()
    if head_type not in VALID_HEAD_TYPES:
        raise SystemExit(
            f"train_setbert.head_type must be one of {sorted(VALID_HEAD_TYPES)}; "
            f"got {head_type!r}."
        )
    merged = {**merged, "head_type": head_type}
    model = SetBertBinaryClassifier(
        base_model,
        embed_dim=embed_dim,
        head_type=head_type,
        head_dropout=head_dropout,
    ).to(device)

    pos_weight = _compute_pos_weight(train_entries, class_weight_mode, device)
    if pos_weight is not None:
        print(
            f"  pos_weight={float(pos_weight.item()):.4f} (mode={class_weight_mode})",
            flush=True,
        )

    # ----- Dataloaders ------------------------------------------------------
    loader_batch_size = int(merged["batch_size"])
    if loader_batch_size < 1:
        raise SystemExit("train_setbert.batch_size must be >= 1.")
    raw_inf_bs = merged.get("inference_batch_size")
    inference_batch_size = (
        int(raw_inf_bs) if raw_inf_bs not in (None, "", "null") else loader_batch_size
    )
    if inference_batch_size < 1:
        raise SystemExit("train_setbert.inference_batch_size must be >= 1.")
    num_workers = int(merged["num_workers"])
    collate = _make_collate(pad_token_id)
    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0
    train_ds = SetBertRunDataset(train_entries, requested_set_size=requested_set_size)
    val_ds = SetBertRunDataset(val_entries, requested_set_size=requested_set_size)
    test_ds = SetBertRunDataset(test_entries, requested_set_size=requested_set_size)
    holdout_ds = (
        SetBertRunDataset(holdout_entries, requested_set_size=requested_set_size)
        if holdout_entries
        else None
    )

    sampler: Optional[WeightedRandomSampler] = None
    shuffle = True
    if train_sampler == "study_balanced":
        sampler = WeightedRandomSampler(
            _study_sampler_weights(train_entries),
            num_samples=len(train_entries),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=loader_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    # Inference loaders are reused across epochs so workers stay warm and
    # .pt -> tensor decode overlaps with GPU forward. Earlier versions built
    # these from scratch each epoch with num_workers=0, which serialized
    # CPU load and GPU compute and was the dominant source of the 15-min
    # post-train inference stall.
    val_loader = DataLoader(
        val_ds,
        batch_size=inference_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=inference_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    holdout_loader: Optional[DataLoader] = None
    if holdout_ds is not None:
        holdout_loader = DataLoader(
            holdout_ds,
            batch_size=inference_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    # ----- Optimization knobs ----------------------------------------------
    lr = float(merged["learning_rate"])
    wd = float(merged["weight_decay"])
    raw_blm = merged.get("backbone_lr_mult")
    backbone_lr_mult = 1.0 if raw_blm is None else float(raw_blm)
    freeze_enc_n = int(merged.get("freeze_sequence_encoder_epochs") or 0)
    freeze_sab_n = int(merged.get("freeze_sab_epochs") or 0)
    if freeze_enc_n < 0 or freeze_sab_n < 0:
        raise SystemExit("freeze_sequence_encoder_epochs and freeze_sab_epochs must be >= 0.")
    epochs = int(merged["epochs"])
    if epochs < 1:
        raise SystemExit("train_setbert.epochs must be >= 1.")

    transition_epochs: set[int] = set()
    if freeze_enc_n > 0:
        transition_epochs.add(freeze_enc_n + 1)
    if freeze_sab_n > 0:
        transition_epochs.add(freeze_sab_n + 1)

    scaler: Optional[torch.amp.GradScaler] = None
    if amp_enabled and amp_use_scaler:
        scaler = torch.amp.GradScaler("cuda")

    epoch_log: List[Dict[str, object]] = []
    best_tuning_score = float("-inf")
    best_epoch = 0
    best_val_auc = float("nan")
    best_val_f1 = float("nan")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_test_scores: Optional[np.ndarray] = None
    best_holdout_scores: Optional[np.ndarray] = None

    results_path = _results_json_out_path(
        repo_root, merged.get("results_json"), task=task
    )
    training_log_path: Optional[Path] = None
    if results_path is not None:
        training_log_path = _training_log_path_from_results(results_path)

    opt: Optional[torch.optim.Optimizer] = None
    # Per-Run DNABERT outputs for val/test/holdout while the sequence encoder is
    # frozen; cleared the moment the encoder transitions to trainable so we
    # never feed stale embeddings into SAB.
    val_encoder_cache: Optional[Dict[str, torch.Tensor]] = None
    test_encoder_cache: Optional[Dict[str, torch.Tensor]] = None
    holdout_encoder_cache: Optional[Dict[str, torch.Tensor]] = None
    for ep in range(1, epochs + 1):
        train_encoder = ep > freeze_enc_n if freeze_enc_n > 0 else True
        train_sab = ep > freeze_sab_n if freeze_sab_n > 0 else True
        _set_backbone_requires_grad(
            model, train_encoder=train_encoder, train_sab=train_sab
        )
        if ep == 1 or ep in transition_epochs:
            opt = _make_optimizer(
                model, lr=lr, weight_decay=wd, backbone_lr_mult=backbone_lr_mult
            )
        assert opt is not None

        print(
            f"\n--- Epoch {ep}/{epochs} "
            f"(train_encoder={train_encoder}, train_sab={train_sab}) ---",
            flush=True,
        )

        if train_encoder:
            val_encoder_cache = None
            test_encoder_cache = None
            holdout_encoder_cache = None
        else:
            if val_encoder_cache is None:
                val_encoder_cache = _populate_encoder_cache(
                    model, val_loader, device,
                    amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                    desc="Cache(val)",
                )
            if test_encoder_cache is None:
                test_encoder_cache = _populate_encoder_cache(
                    model, test_loader, device,
                    amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                    desc="Cache(test)",
                )
            if holdout_loader is not None and holdout_encoder_cache is None:
                holdout_encoder_cache = _populate_encoder_cache(
                    model, holdout_loader, device,
                    amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                    desc="Cache(holdout)",
                )
            if ep == 1 or ep in transition_epochs:
                n_holdout = (
                    len(holdout_encoder_cache)
                    if holdout_encoder_cache is not None
                    else 0
                )
                print(
                    f"  cached encoder embeddings: "
                    f"val={len(val_encoder_cache)} "
                    f"test={len(test_encoder_cache)} "
                    f"holdout={n_holdout} Runs",
                    flush=True,
                )

        train_loss = _train_one_epoch(
            model,
            train_loader,
            opt,
            device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
            pos_weight=pos_weight,
        )
        val_loss, val_scores = _eval_val_full(
            model,
            val_loader,
            device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            pos_weight=pos_weight,
            desc="Val",
            encoder_cache=val_encoder_cache,
        )
        test_scores = _score_loader(
            model, test_loader, device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            desc="Test",
            encoder_cache=test_encoder_cache,
        )
        holdout_scores = _score_loader(
            model, holdout_loader, device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            desc="Holdout",
            encoder_cache=holdout_encoder_cache,
        )
        val_auc, val_f1 = _binary_metrics(
            val_entries, val_scores, pos_label=pos_label, neg_label=neg_label
        )
        test_auc, test_f1 = _binary_metrics(
            test_entries, test_scores, pos_label=pos_label, neg_label=neg_label
        )
        holdout_auc, holdout_f1 = _binary_metrics(
            holdout_entries, holdout_scores, pos_label=pos_label, neg_label=neg_label
        )

        tuning_score = val_auc if tuning_metric == "auc" else val_f1
        if tuning_score == tuning_score and tuning_score > best_tuning_score:
            best_tuning_score = float(tuning_score)
            best_epoch = int(ep)
            best_val_auc = val_auc
            best_val_f1 = val_f1
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            # Test/holdout scores at the best epoch are reused for the final
            # results JSON (no extra inference passes after training).
            best_test_scores = test_scores
            best_holdout_scores = holdout_scores

        epoch_log.append(
            {
                "epoch": int(ep),
                "learning_rate": float(opt.param_groups[0]["lr"]),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_f1": float(val_f1),
                "test_f1": float(test_f1),
                "holdout_f1": float(holdout_f1),
                "val_auc": float(val_auc),
                "test_auc": float(test_auc),
                "holdout_auc": float(holdout_auc),
            }
        )
        if training_log_path is not None:
            _write_training_log(training_log_path, epoch_log)

        print(
            f"  loss(train)={train_loss:.4f} loss(val)={val_loss:.4f}  "
            f"val_auc={val_auc:.4f} val_f1={val_f1:.4f}  "
            f"test_auc={test_auc:.4f} test_f1={test_f1:.4f}  "
            f"holdout_auc={holdout_auc:.4f} holdout_f1={holdout_f1:.4f}  "
            f"best_epoch={best_epoch}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    if results_path is not None:
        # Reuse the test/holdout scores captured at the best epoch instead of
        # running two more full inference passes after restoring weights. The
        # cache reuse below only kicks in for the (rare) edge case where the
        # tuning score never improved AND the encoder was frozen for the whole
        # run (so the cache was never invalidated and still matches the
        # restored encoder weights).
        if best_test_scores is None:
            final_test_scores = _score_loader(
                model, test_loader, device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                desc="Test (final)",
                encoder_cache=test_encoder_cache,
            )
        else:
            final_test_scores = best_test_scores
        if best_holdout_scores is None:
            final_holdout_scores = _score_loader(
                model, holdout_loader, device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                desc="Holdout (final)",
                encoder_cache=holdout_encoder_cache,
            )
        else:
            final_holdout_scores = best_holdout_scores
        final_test_auc, final_test_f1 = _binary_metrics(
            test_entries, final_test_scores, pos_label=pos_label, neg_label=neg_label
        )
        final_holdout_auc, final_holdout_f1 = _binary_metrics(
            holdout_entries, final_holdout_scores, pos_label=pos_label, neg_label=neg_label
        )
        _write_results_json(
            results_path,
            merged=merged,
            cache_info={
                "dir": str(run_tensors_root),
                "cache_set_size": int(sample_meta.get("set_size", 0)),
                "pretrained_repo": setbert_section["pretrained_repo"],
                "pretrained_revision": setbert_section["pretrained_revision"],
                "n_cached_runs": len(all_records),
            },
            task=task,
            pos_label=pos_label,
            neg_label=neg_label,
            best_epoch=best_epoch,
            tuning_metric=tuning_metric,
            best_tuning_score=best_tuning_score,
            best_val_auc=best_val_auc,
            best_val_f1=best_val_f1,
            test_auc=final_test_auc,
            test_f1=final_test_f1,
            holdout_auc=final_holdout_auc,
            holdout_f1=final_holdout_f1,
        )
        _write_predictions_csv(
            results_path.with_name(f"{results_path.stem}_test_predictions.csv"),
            test_entries,
            final_test_scores,
            pos_label=pos_label,
            neg_label=neg_label,
        )
        _write_predictions_csv(
            results_path.with_name(f"{results_path.stem}_holdout_predictions.csv"),
            holdout_entries,
            final_holdout_scores,
            pos_label=pos_label,
            neg_label=neg_label,
        )
        print(f"\nWrote results -> {results_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
