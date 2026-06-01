#!/usr/bin/env python3
"""
Fine-tune HyenaDNA with a single-task binary classification head.

Uses a pre-built feature-only per-run tensor cache under paths.hyenadna_run_tensors_dir/ from
scripts/build_hyenadna_run_tensors.py, joins labels/splits from shared metadata at runtime, and
reports run-level AUC on the test and holdout splits (one score per Run).

Training logs (`*_training.json`) record per-epoch loss, binary F1, and AUC metrics aligned
with console output. Checkpoints are selected by ``train_hyenadna.tuning_metric`` (default
``auc``).

Config: defaults.yaml (train_hyenadna + hyenadna_run_tensors + paths) with optional
experiments.yaml train_hyenadna overrides (--expt).

Intermittent ``Signals.SIGSEGV`` during grid runs usually indicates a GPU driver or
PyTorch CUDA native crash rather than bad Python tensor logic. We use
``PYTHONFAULTHANDLER=1`` for child processes to get more info.
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ContextManager, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from hyenadna import HyenaDNAPreTrainedModel
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from hyenadna_fasta_data import (
    merge_train_hyenadna_config,
    model_max_length,
)
from shared_utilities import (
    binary_auc_from_scores,
    build_run_task_table,
    resolve_repo_path,
)
from cache_operations import load_sequence_row_selection

HEAD_MODES = ("last", "first", "pool", "sum")

_HYENADNA_TASK_VALUES = frozenset({"cancer_diagnosis", "cancer_type"})


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunRecord:
    run: str
    split: str
    label: int
    task_label: str
    study_name: str
    n_sets: int
    file: Path


@dataclass(frozen=True)
class ScoreContext:
    requested_num_sets: int
    requested_max_len: int
    cache_num_sets: int
    cache_max_len: int


@dataclass(frozen=True)
class HeadScores:
    """Metrics use ``entries``/``y_true``/``y_score``; CSV uses ``all_*`` (one row per input run)."""

    entries: Tuple[RunRecord, ...]
    y_true: np.ndarray
    y_score: np.ndarray
    auc: float
    f1: float
    all_entries: Tuple[RunRecord, ...] = ()
    all_scores: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.float64))


@dataclass(frozen=True)
class SplitScores:
    val: HeadScores
    test: HeadScores
    holdout: HeadScores


@dataclass(frozen=True)
class TaskConfig:
    pos_class_index: int
    neg_label: str
    pos_label: str
    class_names: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Grid-mode subprocess helpers
# ---------------------------------------------------------------------------


def _grid_child_subprocess_env() -> Dict[str, str]:
    """Copy of the environment for ``--child-run`` workers; enables faulthandler (see module doc)."""
    env = dict(os.environ)
    env["PYTHONFAULTHANDLER"] = "1"
    return env


def _clamp_sigsegv_retries(raw: object) -> int:
    """Clamp train_hyenadna.sigsegv_retries for grid-mode subprocesses only (0..8 extra runs after SIGSEGV)."""
    try:
        n = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return max(0, min(int(n), 8))


def _run_grid_child_check(
    cmd: Sequence[str],
    repo_root: Path,
    *,
    sigsegv_retries: object,
) -> None:
    """Run one grid child process; retry on SIGSEGV up to train_hyenadna.sigsegv_retries extra times."""
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
                    "train_hyenadna: grid child died with SIGSEGV "
                    f"(attempt {attempt}/{max_attempts}); retrying after short delay.",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(min(float(attempt), 5.0))
                continue
            if is_sigsegv:
                raise SystemExit(
                    "HyenaDNA training subprocess crashed with SIGSEGV (signal 11). "
                    "This normally comes from the CUDA stack or GPU driver, not from "
                    "Python exceptions—reruns often succeed. Next steps: run once with "
                    "CUDA_LAUNCH_BLOCKING=1; confirm train_hyenadna.num_workers is 0; "
                    "upgrade driver/PyTorch; or raise "
                    "train_hyenadna.sigsegv_retries in experiments.yaml (already retried)."
                ) from exc
            raise


# ---------------------------------------------------------------------------
# Results path helpers
# ---------------------------------------------------------------------------


def _task_abbrv(task: str) -> str:
    t = str(task).strip()
    if t == "cancer_diagnosis":
        return "cd"
    if t == "cancer_type":
        return "ct"
    raise SystemExit(
        f"Unknown task {task!r} for results path "
        "(expected cancer_diagnosis or cancer_type)."
    )


def _results_json_out_path(
    repo_root: Path,
    raw: Optional[object],
    *,
    task: str,
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
        return scratch_base / f"train_hyenadna_{task}_{ts}.json"
    p = Path(str(raw).strip()).expanduser()
    return p if p.is_absolute() else repo_root / p


# ---------------------------------------------------------------------------
# Run record construction
# ---------------------------------------------------------------------------


def prepare_task_config(
    task: str,
    defaults_path: Path,
) -> Tuple[Any, LabelEncoder, TaskConfig]:
    """Build the run metadata frame, fit a LabelEncoder on the training rows, and return the task config."""
    run_df = build_run_task_table(task, config_path=defaults_path)
    enc = LabelEncoder()
    y_train = run_df.loc[run_df["split"] == "train", "task_label"].to_numpy(dtype=object)
    if y_train.size == 0:
        raise SystemExit("No training runs found after shared split assignment.")
    enc.fit(y_train)
    class_names = [str(x) for x in enc.classes_.tolist()]
    if len(class_names) < 2:
        raise SystemExit("HyenaDNA training expects at least 2 task classes.")
    # Lexicographic LabelEncoder convention: index 0 is the negative class, index 1 the positive class.
    cfg = TaskConfig(
        pos_class_index=1,
        neg_label=class_names[0],
        pos_label=class_names[1],
        class_names=tuple(class_names),
    )
    return run_df, enc, cfg


def build_run_records(
    run_task_df,
    cache_root: Path,
    *,
    label_encoder: LabelEncoder,
    requested_num_sets: int,
) -> Tuple[List[RunRecord], int, Tuple[str, ...]]:
    """Build RunRecords from a per-task metadata frame and the per-run tensor cache."""
    cols = ["Run", "split", "task_label", "study_name"]
    rows = run_task_df.loc[:, cols].drop_duplicates(subset=["Run"]).reset_index(drop=True)
    out: List[RunRecord] = []
    missing_runs: List[str] = []

    for _, row in rows.iterrows():
        run = str(row["Run"]).strip()
        pt_path = cache_root / f"{run}.pt"
        if not pt_path.is_file():
            missing_runs.append(run)
            continue
        try:
            blob = torch.load(pt_path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(pt_path, map_location="cpu")
        n_sets = int(blob.get("n_sets", 0))
        if n_sets <= 0:
            continue
        primary_label = str(row["task_label"]).strip()
        out.append(
            RunRecord(
                run=run,
                split=str(row["split"]),
                label=int(label_encoder.transform([primary_label])[0]),
                task_label=primary_label,
                study_name=str(row["study_name"]).strip(),
                n_sets=min(n_sets, requested_num_sets),
                file=pt_path,
            )
        )

    return out, len(missing_runs), tuple(sorted(missing_runs)[:5])


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------


class RunTensorDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[RunRecord],
        *,
        requested_num_sets: int,
        requested_max_len: int,
        cache_num_sets: int,
        cache_max_len: int,
    ):
        self.entries = list(entries)
        self.requested_num_sets = requested_num_sets
        self.requested_max_len = requested_max_len
        self.cache_num_sets = cache_num_sets
        self.cache_max_len = cache_max_len

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        e = self.entries[idx]
        path = e.file
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(path, map_location="cpu")
        n_sets = min(int(blob["n_sets"]), self.requested_num_sets, self.cache_num_sets)
        start = self.cache_max_len - self.requested_max_len
        if start < 0:
            raise SystemExit(
                "train_hyenadna.max_length exceeds hyenadna_run_tensors.max_length. "
                "Increase hyenadna_run_tensors.max_length and rebuild run tensors."
            )
        input_ids = blob["input_ids"][: self.requested_num_sets, start:self.cache_max_len]
        attention_mask = blob["attention_mask"][: self.requested_num_sets, start:self.cache_max_len]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "n_sets": n_sets,
            "label": e.label,
            "run": e.run,
        }


def collate_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    n_sets = torch.tensor([b["n_sets"] for b in batch], dtype=torch.long)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    runs = [str(b["run"]) for b in batch]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "n_sets": n_sets,
        "label": labels,
        "run": runs,
    }


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def _resolve_train_batch_size(raw: object) -> int:
    value = float(raw)
    if not value.is_integer() or value < 1:
        raise SystemExit("train_hyenadna.batch_size must be a positive integer.")
    return int(value)


def _resolve_amp_config(
    merged: Mapping[str, object],
    device: torch.device,
) -> Tuple[bool, torch.dtype, str, bool]:
    amp_requested = bool(merged.get("amp", False))
    amp_dtype_raw = str(merged.get("amp_dtype", "float16")).strip().lower()
    if amp_dtype_raw == "float16":
        amp_dtype = torch.float16
        use_grad_scaler = True
    elif amp_dtype_raw == "bfloat16":
        amp_dtype = torch.bfloat16
        use_grad_scaler = False
    else:
        raise SystemExit("train_hyenadna.amp_dtype must be 'float16' or 'bfloat16'.")

    if not amp_requested:
        return False, amp_dtype, amp_dtype_raw, use_grad_scaler
    if device.type != "cuda":
        print("AMP requested but CUDA is unavailable; running in float32.", flush=True)
        return False, amp_dtype, amp_dtype_raw, use_grad_scaler
    return True, amp_dtype, amp_dtype_raw, use_grad_scaler


def _amp_autocast(
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> ContextManager[object]:
    if amp_enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def _study_sampler_weights(entries: Sequence[RunRecord]) -> torch.Tensor:
    counts: Dict[str, int] = defaultdict(int)
    for e in entries:
        counts[e.study_name] += 1
    w = [1.0 / counts[e.study_name] for e in entries]
    return torch.tensor(w, dtype=torch.double)


def _compute_ce_weight_tensor(
    train_entries: Sequence[RunRecord],
    *,
    mode: str,
    n_classes: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    m = str(mode or "none").strip().lower()
    if m in ("none", "null", ""):
        return None
    if m in ("balanced", "balanced_sqrt"):
        y = np.array([int(e.label) for e in train_entries], dtype=np.int64)
        if y.size == 0:
            return None
        present = np.unique(y)
        if present.size < 2:
            return None
        cw = compute_class_weight(class_weight="balanced", classes=present, y=y)
        if m == "balanced_sqrt":
            cw = np.sqrt(cw)
            cw = cw / np.mean(cw)
        t = torch.ones(n_classes, dtype=torch.float32, device=device)
        for cls, w in zip(present, cw):
            t[int(cls)] = float(w)
        return t
    raise SystemExit(
        f"Unknown class_weight {mode!r} (use none, balanced, or balanced_sqrt)."
    )


def training_loss_sum_and_count(
    model: torch.nn.Module,
    batch: Mapping[str, object],
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    ce_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int]:
    input_ids = batch["input_ids"].to(device)
    bsz, n_set, seq_len = input_ids.shape
    flat_in = input_ids.view(bsz * n_set, seq_len)
    nv = batch["n_sets"]
    labels = batch["label"].to(device)
    mask = torch.arange(n_set, device=device).unsqueeze(0) < nv.to(device).unsqueeze(1)

    with _amp_autocast(device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
        logits = model(flat_in)
        logits = logits.view(bsz, n_set, -1)
        flat_logits = logits[mask]
        flat_y = labels.unsqueeze(1).expand(bsz, n_set)[mask]
    if flat_logits.numel() == 0:
        return torch.zeros((), device=device), 0
    ce_kw: Dict[str, object] = {"reduction": "sum"}
    if ce_weight is not None:
        # AMP can produce bf16/fp16 logits; class-weight tensor must match.
        ce_kw["weight"] = ce_weight.to(
            device=flat_logits.device,
            dtype=flat_logits.dtype,
        )
    task_ce = F.cross_entropy(flat_logits, flat_y, **ce_kw)
    return task_ce, int(flat_y.shape[0])


def _eval_mean_ce_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    ce_weight: Optional[torch.Tensor],
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            loss_sum, n = training_loss_sum_and_count(
                model,
                batch,
                device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                ce_weight=ce_weight,
            )
            total += float(loss_sum.item())
            count += int(n)
    return total / max(count, 1)


def _classification_head_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    if model.head is None:
        return []
    return [p for p in model.head.parameters() if p.requires_grad]


def _make_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    backbone_lr_mult: float,
) -> torch.optim.Optimizer:
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = _classification_head_params(model)
    groups: List[Dict[str, object]] = []
    if backbone_params:
        blr = lr * float(backbone_lr_mult)
        groups.append({"params": backbone_params, "lr": blr})
    if head_params:
        groups.append({"params": head_params, "lr": lr})
    if not groups:
        raise SystemExit("No trainable parameters for optimizer.")
    return torch.optim.AdamW(groups, weight_decay=float(weight_decay))


def _set_backbone_requires_grad(model: torch.nn.Module, trainable: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = trainable
    for p in _classification_head_params(model):
        p.requires_grad = True


def _optimizer_step(
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
) -> None:
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    scaler: Optional[torch.amp.GradScaler],
    ce_weight: Optional[torch.Tensor],
) -> float:
    model.train()
    total = 0.0
    n_batches = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        loss_sum, denom = training_loss_sum_and_count(
            model,
            batch,
            device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            ce_weight=ce_weight,
        )
        if denom <= 0:
            raise SystemExit("Training batch has zero valid sets after masking.")
        loss = loss_sum / float(denom)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        _optimizer_step(optimizer, scaler)
        total += float(loss.item())
        n_batches += 1
    return total / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _load_run_tensor(
    path: Path,
    *,
    requested_num_sets: int,
    requested_max_len: int,
    cache_num_sets: int,
    cache_max_len: int,
) -> Tuple[Optional[torch.Tensor], int]:
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    n_sets = int(blob.get("n_sets", 0))
    if n_sets <= 0:
        return None, 0
    start = cache_max_len - requested_max_len
    if start < 0:
        raise SystemExit(
            "train_hyenadna.max_length exceeds hyenadna_run_tensors.max_length. "
            "Increase hyenadna_run_tensors.max_length and rebuild run tensors."
        )
    x = blob["input_ids"][:requested_num_sets, start:cache_max_len]
    nv = min(n_sets, requested_num_sets, cache_num_sets)
    return x, nv


def _positive_class_score(logits: torch.Tensor, pos_class_index: int) -> float:
    """Run-level score: mean logits over sequence sets, then softmax positive class."""
    agg = logits.mean(dim=0)
    return float(torch.softmax(agg, dim=-1)[pos_class_index].item())


def _binary_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    pos_label: str,
    neg_label: str,
) -> float:
    """Positive-class F1 at threshold 0.5 on the positive-class score."""
    if y_true.size == 0:
        return float("nan")
    y_pred = np.where(y_score >= 0.5, pos_label, neg_label)
    return float(
        f1_score(
            y_true,
            y_pred,
            pos_label=pos_label,
            average="binary",
            zero_division=0,
        )
    )


def _head_metrics(
    entries: Sequence[RunRecord],
    scores: np.ndarray,
    task_cfg: TaskConfig,
) -> HeadScores:
    scores = np.asarray(scores, dtype=np.float64)
    valid = scores == scores
    pairs = [(e, float(s)) for e, s, ok in zip(entries, scores, valid) if ok]
    all_entries = tuple(entries)
    all_scores = np.asarray(scores, dtype=np.float64)
    if not pairs:
        return HeadScores(
            (),
            np.asarray([], dtype=object),
            np.asarray([], dtype=np.float64),
            float("nan"),
            float("nan"),
            all_entries=all_entries,
            all_scores=all_scores,
        )
    sub_entries, sub_scores = zip(*pairs)
    sub_entries_t = tuple(sub_entries)
    y_true = np.array([e.task_label for e in sub_entries_t], dtype=object)
    y_score = np.asarray(sub_scores, dtype=np.float64)
    auc = binary_auc_from_scores(y_true, y_score, positive_label=task_cfg.pos_label)
    f1 = _binary_f1(y_true, y_score, pos_label=task_cfg.pos_label, neg_label=task_cfg.neg_label)
    return HeadScores(
        sub_entries_t,
        y_true,
        y_score,
        float(auc),
        f1,
        all_entries=all_entries,
        all_scores=all_scores,
    )


def _forward_entry_score(
    model: torch.nn.Module,
    entry: RunRecord,
    pos_class_index: int,
    ctx: ScoreContext,
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> float:
    x, nv = _load_run_tensor(
        entry.file,
        requested_num_sets=ctx.requested_num_sets,
        requested_max_len=ctx.requested_max_len,
        cache_num_sets=ctx.cache_num_sets,
        cache_max_len=ctx.cache_max_len,
    )
    if x is None or nv <= 0:
        return float("nan")
    x = x.to(device)
    with _amp_autocast(device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
        logits = model(x[:nv])
    return _positive_class_score(logits, pos_class_index)


def score_entries(
    model: torch.nn.Module,
    entries: Sequence[RunRecord],
    device: torch.device,
    task_cfg: TaskConfig,
    ctx: ScoreContext,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> HeadScores:
    """Score all entries with one backbone forward per run."""
    if not entries:
        return HeadScores(
            (),
            np.asarray([], dtype=object),
            np.asarray([], dtype=np.float64),
            float("nan"),
            float("nan"),
            all_entries=(),
            all_scores=np.asarray([], dtype=np.float64),
        )
    model.eval()
    raw: List[float] = []
    with torch.no_grad():
        for e in entries:
            raw.append(
                _forward_entry_score(
                    model,
                    e,
                    task_cfg.pos_class_index,
                    ctx,
                    device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
            )
    return _head_metrics(entries, np.asarray(raw, dtype=np.float64), task_cfg)


def eval_splits(
    model: torch.nn.Module,
    *,
    val_entries: Sequence[RunRecord],
    test_entries: Sequence[RunRecord],
    holdout_entries: Sequence[RunRecord],
    task_cfg: TaskConfig,
    ctx: ScoreContext,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> SplitScores:
    return SplitScores(
        val=score_entries(
            model, val_entries, device, task_cfg, ctx,
            amp_enabled=amp_enabled, amp_dtype=amp_dtype,
        ),
        test=score_entries(
            model, test_entries, device, task_cfg, ctx,
            amp_enabled=amp_enabled, amp_dtype=amp_dtype,
        ),
        holdout=score_entries(
            model, holdout_entries, device, task_cfg, ctx,
            amp_enabled=amp_enabled, amp_dtype=amp_dtype,
        ),
    )


def tuning_score(f1: float, auc: float, metric: str) -> float:
    m = str(metric).strip()
    if m == "auc":
        v = float(auc)
    elif m == "f1":
        v = float(f1)
    else:
        raise SystemExit(
            f"Unknown tuning_metric {metric!r} (use auc or f1)."
        )
    return v if v == v else float("-inf")


def epoch_progress_fields(
    *,
    split_eval: SplitScores,
    train_loss: float,
    val_loss: float,
    best_epoch: int,
) -> List[str]:
    fields = [f"train_loss={train_loss:.4f}", f"val_loss={val_loss:.4f}"]
    fields.extend(
        [
            f"val_auc={split_eval.val.auc:.4f}",
            f"test_auc={split_eval.test.auc:.4f}",
            f"holdout_auc={split_eval.holdout.auc:.4f}",
        ]
    )
    fields.append(f"best_epoch={int(best_epoch)}")
    return fields


# ---------------------------------------------------------------------------
# Argument parsing and per-experiment grid driver
# ---------------------------------------------------------------------------


def _parse_argv(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expt",
        type=int,
        default=None,
        help="Optional train_hyenadna experiment index from experiments.yaml (1-based). "
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
            "as train_hyenadna_<task>_<utc>.json. Omit entirely to use YAML only."
        ),
    )
    parser.add_argument(
        "--override-random-seed",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--override-task",
        type=str,
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


def _parse_task_grid(raw: object) -> Optional[List[str]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        task = raw.strip()
        if not task:
            return None
        if task not in _HYENADNA_TASK_VALUES:
            raise SystemExit(
                f"Unknown train_hyenadna.task {task!r} "
                "(use cancer_diagnosis or cancer_type)."
            )
        return [task]
    if isinstance(raw, (list, tuple)):
        if not raw:
            raise SystemExit("task grid must not be empty.")
        tasks = [str(t).strip() for t in raw]
        bad = [p for p in tasks if p not in _HYENADNA_TASK_VALUES]
        if bad:
            raise SystemExit(
                "train_hyenadna task grid must list only cancer_diagnosis and/or "
                f"cancer_type; unknown value(s): {bad!r}."
            )
        return tasks
    raise SystemExit(
        f"task must be a string or a YAML list of task names; got {type(raw).__name__}."
    )


def _format_hyenadna_results_template(
    template: str,
    *,
    task: str,
    name: str,
    seed: int,
    max_length: int,
) -> str:
    max_length_k = int(max_length) // 1024
    text = str(template).replace("{max_length/1024}", "{max_length_k}")
    return text.format(
        task=str(task).strip(),
        task_abbrv=_task_abbrv(task),
        name=name,
        seed=int(seed),
        max_length=int(max_length),
        max_length_k=int(max_length_k),
    )


# ---------------------------------------------------------------------------
# Results / prediction writing
# ---------------------------------------------------------------------------


_PREDICTION_CSV_FIELDS = ("Run", "task_label", "predicted_label", "positive_score")


def _prediction_rows(
    entries: Sequence[RunRecord],
    scores: HeadScores,
    task_cfg: TaskConfig,
) -> List[Dict[str, object]]:
    use_entries = scores.all_entries if scores.all_entries else tuple(entries)
    use_scores = scores.all_scores if scores.all_scores.size else scores.y_score
    if len(use_entries) != len(use_scores):
        raise SystemExit("Prediction row count mismatch.")
    rows: List[Dict[str, object]] = []
    for e, score in zip(use_entries, use_scores):
        pred = task_cfg.pos_label if float(score) >= 0.5 else task_cfg.neg_label
        rows.append(
            {
                "Run": str(e.run),
                "task_label": str(e.task_label),
                "predicted_label": str(pred),
                "positive_score": float(score),
            }
        )
    return rows


def _write_predictions_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_PREDICTION_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out_row: Dict[str, str] = {}
            for key in _PREDICTION_CSV_FIELDS:
                if key == "positive_score":
                    out_row[key] = f"{float(row[key]):.10f}"
                else:
                    out_row[key] = str(row[key])
            writer.writerow(out_row)


def _float_or_none(x: float) -> Optional[float]:
    return float(x) if x == x else None


def _write_results_json(
    path: Path,
    *,
    merged: Mapping[str, Any],
    cache_info: Mapping[str, Any],
    test_auc: float,
    holdout_auc: float,
    best_epoch: int,
    tuning_metric: str,
    best_tuning_score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": Path(__file__).name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": dict(merged),
        "data": {"cache": dict(cache_info)},
        "tuning": {
            "split": "validation",
            "metric": str(tuning_metric),
            "score": _float_or_none(float(best_tuning_score)),
            "best_epoch": int(best_epoch),
        },
        "metrics": {
            "test": {"auc": _float_or_none(test_auc)},
            "holdout": {"auc": _float_or_none(holdout_auc)},
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _training_log_path_from_results(path: Path) -> Path:
    return path.with_name(f"{path.stem}_training.json")


def _write_training_log(
    path: Path,
    epoch_rows: Sequence[Mapping[str, object]],
) -> None:
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


def write_run_artifacts(
    results_path: Path,
    *,
    merged: Mapping[str, Any],
    cache_info: Mapping[str, Any],
    task_cfg: TaskConfig,
    test_entries: Sequence[RunRecord],
    holdout_entries: Sequence[RunRecord],
    test_scores: HeadScores,
    holdout_scores: HeadScores,
    best_epoch: int,
    tuning_metric: str,
    best_tuning_score: float,
) -> None:
    """Write results JSON and split prediction CSVs."""
    pred_rows_test = _prediction_rows(test_entries, test_scores, task_cfg)
    pred_rows_hold = _prediction_rows(holdout_entries, holdout_scores, task_cfg)

    test_pred_path = results_path.with_name(f"{results_path.stem}_test.csv")
    holdout_pred_path = results_path.with_name(f"{results_path.stem}_holdout.csv")
    _write_predictions_csv(test_pred_path, pred_rows_test)
    _write_predictions_csv(holdout_pred_path, pred_rows_hold)

    _write_results_json(
        results_path,
        merged=merged,
        cache_info=cache_info,
        test_auc=test_scores.auc,
        holdout_auc=holdout_scores.auc,
        best_epoch=best_epoch,
        tuning_metric=tuning_metric,
        best_tuning_score=best_tuning_score,
    )

    print(
        f"Final eval (best epoch by {tuning_metric}): best_epoch={best_epoch} "
        f"test_auc={test_scores.auc:.4f} "
        f"holdout_auc={holdout_scores.auc:.4f}\n",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    cli = _parse_argv(argv)
    expt = int(cli.expt) if cli.expt is not None else 0

    repo_root = Path(__file__).resolve().parent.parent

    defaults_path = repo_root / "defaults.yaml"
    experiments_path = repo_root / "experiments.yaml"

    merged, exp_name, _tpl = merge_train_hyenadna_config(
        defaults_path,
        experiments_path,
        expt=expt,
    )

    if hasattr(cli, "results_json"):
        rj = cli.results_json
        if rj == "" and expt > 0 and not cli.child_run:
            # Makefile passes --results-json for the default no-EXPT workflow.
            # For experiment runs, leave output path selection to experiment/template logic.
            pass
        elif rj == "":
            merged = {**merged, "results_json": ""}
        else:
            merged = {**merged, "results_json": rj}
    if cli.override_task is not None:
        merged = {**merged, "task": str(cli.override_task).strip()}
    if cli.override_random_seed is not None:
        merged = {**merged, "random_seed": int(cli.override_random_seed)}

    if expt > 0 and not cli.child_run:
        experiments_cfg = yaml.safe_load(experiments_path.read_text(encoding="utf-8")) or {}
        train_section = experiments_cfg.get("train_hyenadna") or {}
        if not isinstance(train_section, dict):
            raise SystemExit("experiments.yaml train_hyenadna must be a mapping when present.")
        expts = train_section.get("experiments") or []
        if not isinstance(expts, list) or expt > len(expts):
            raise SystemExit(
                f"--expt {expt} out of range; experiments.yaml has {len(expts) if isinstance(expts, list) else 0} rows."
            )
        row = expts[expt - 1]
        if not isinstance(row, dict):
            raise SystemExit("train_hyenadna experiment entry must be a mapping.")
        overrides = row.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise SystemExit("train_hyenadna experiment overrides must be a mapping.")

        task_grid = _parse_task_grid(overrides.get("task")) or [str(merged["task"]).strip()]
        seed_grid = _parse_seed_grid(overrides.get("random_seed")) or [int(merged["random_seed"])]
        grid_max_length = int(
            model_max_length(str(merged["model"]).strip(), merged.get("max_length"))
        )

        if len(task_grid) > 1 or len(seed_grid) > 1:
            template = _tpl if isinstance(_tpl, str) and _tpl.strip() else None
            if template is None:
                raise SystemExit(
                    "Grid experiment requires train_hyenadna.results_json_template in experiments.yaml."
                )
            if not exp_name:
                raise SystemExit("Grid experiment requires a non-empty train_hyenadna experiment name.")
            if hasattr(cli, "results_json") and str(cli.results_json) != "":
                raise SystemExit(
                    "--results-json cannot be used with multi-run train_hyenadna grid experiments."
                )

            total = int(len(task_grid) * len(seed_grid))
            completed = 0
            skipped = 0
            print(
                f"Running EXPT={expt} grid: {len(task_grid)} task(s) x {len(seed_grid)} seed(s) "
                f"= {total} run(s) (max_length={grid_max_length}; outer: task, inner: seed).",
                flush=True,
            )
            for task_v in task_grid:
                for seed in seed_grid:
                    out_rel = _format_hyenadna_results_template(
                        template,
                        task=str(task_v),
                        name=str(exp_name),
                        seed=int(seed),
                        max_length=grid_max_length,
                    )
                    out_path = resolve_repo_path(repo_root, out_rel)
                    if out_path.is_file():
                        skipped += 1
                        print(
                            f"Skipping EXPT={expt} task={task_v} seed={seed}: "
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
                        "--override-task",
                        str(task_v),
                        "--override-random-seed",
                        str(seed),
                        "--results-json",
                        str(out_path),
                    ]
                    print(
                        f"Launching EXPT={expt} task={task_v} seed={seed} -> {out_path}",
                        flush=True,
                    )
                    _run_grid_child_check(
                        cmd,
                        repo_root,
                        sigsegv_retries=merged.get("sigsegv_retries"),
                    )
                    completed += 1
            print(
                f"Finished EXPT={expt} grid: launched={completed}, skipped_existing={skipped}, total={total}.",
                flush=True,
            )
            return 0

    defaults_cfg = yaml.safe_load(defaults_path.read_text(encoding="utf-8")) or {}
    if not isinstance(defaults_cfg, dict):
        raise SystemExit(f"{defaults_path} must contain a YAML mapping.")
    paths_cfg = defaults_cfg.get("paths")
    if not isinstance(paths_cfg, dict):
        raise SystemExit(f"{defaults_path} must define paths as a mapping.")
    hyenadna_run_tensors_cfg = defaults_cfg.get("hyenadna_run_tensors")
    if not isinstance(hyenadna_run_tensors_cfg, dict):
        raise SystemExit(
            f"{defaults_path} must define hyenadna_run_tensors as a mapping."
        )
    run_tensors_root = resolve_repo_path(
        repo_root,
        str(
            paths_cfg.get("hyenadna_run_tensors_dir", "outputs/hyenadna_run_tensors")
        ).strip(),
    )

    cache_num_sets = int(hyenadna_run_tensors_cfg["num_sets"])
    cache_max_len = int(hyenadna_run_tensors_cfg["max_length"])
    selection = load_sequence_row_selection(defaults_cfg)
    cache_seq_offset = selection["seq_offset"]
    cache_min_seqs = selection["min_seqs"]

    task = str(merged["task"]).strip()
    if task not in _HYENADNA_TASK_VALUES:
        raise SystemExit(
            f"Unknown train_hyenadna.task {task!r} "
            "(use cancer_diagnosis or cancer_type)."
        )
    model_name = str(merged["model"]).strip()
    num_sets = int(merged["num_sets"])
    loader_batch_size = _resolve_train_batch_size(merged["batch_size"])
    max_len = model_max_length(model_name, merged.get("max_length"))
    head_mode = str(merged["head_pooling_mode"]).strip()
    if head_mode not in HEAD_MODES:
        raise SystemExit(f"head_pooling_mode must be one of {HEAD_MODES}; got {head_mode!r}.")
    if num_sets > cache_num_sets:
        raise SystemExit(
            f"train_hyenadna.num_sets ({num_sets}) exceeds hyenadna_run_tensors.num_sets "
            f"({cache_num_sets}). Rebuild run tensors with a larger cache."
        )
    if max_len > cache_max_len:
        raise SystemExit(
            f"Resolved max_length ({max_len}) exceeds hyenadna_run_tensors.max_length "
            f"({cache_max_len}). Rebuild run tensors with a larger cache."
        )
    if not run_tensors_root.is_dir():
        raise SystemExit(
            f"Missing HyenaDNA run tensors directory: {run_tensors_root}. "
            "Run: python scripts/build_hyenadna_run_tensors.py"
        )

    run_task_df, label_encoder, task_cfg = prepare_task_config(task, defaults_path)

    all_records, n_skipped_missing_cache, missing_cache_examples = build_run_records(
        run_task_df,
        run_tensors_root,
        label_encoder=label_encoder,
        requested_num_sets=num_sets,
    )
    if n_skipped_missing_cache:
        ex = ", ".join(missing_cache_examples)
        print(
            f"train_hyenadna: skipping {n_skipped_missing_cache} metadata run(s) with no "
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
        raise SystemExit("No training runs in cache (split=train). Build dataset first.")
    if not val_entries:
        raise SystemExit("No validation runs in cache.")
    if not test_entries:
        raise SystemExit("No test runs in cache.")

    seed = int(merged["random_seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    if (
        expt > 0
        and exp_name
        and isinstance(_tpl, str)
        and _tpl.strip()
        and merged.get("results_json") in (None, "null")
    ):
        merged = {
            **merged,
            "results_json": _format_hyenadna_results_template(
                _tpl,
                task=str(task).strip(),
                name=str(exp_name),
                seed=seed,
                max_length=max_len,
            ),
        }

    device_s = str(merged.get("device") or "").strip().lower()
    if not device_s or device_s == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_s)
    amp_enabled, amp_dtype, amp_dtype_name, amp_use_grad_scaler = _resolve_amp_config(
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
        f"\nHyenaDNA train | task={task} model={model_name} | "
        f"positive_class={task_cfg.pos_label!r} (index {task_cfg.pos_class_index}) | "
        f"precision={'amp(' + amp_dtype_name + ')' if amp_enabled else 'fp32'}",
        flush=True,
    )

    lr = float(merged["learning_rate"])
    wd = float(merged["weight_decay"])
    raw_blm = merged.get("backbone_lr_mult")
    backbone_lr_mult = 1.0 if raw_blm is None else float(raw_blm)
    train_sampler = str(merged.get("train_sampler") or "random").strip().lower()
    tuning_metric = str(merged.get("tuning_metric") or "auc").strip()
    class_weight_mode = str(merged.get("class_weight") or "none")
    ce_weight = _compute_ce_weight_tensor(
        train_entries,
        mode=class_weight_mode,
        n_classes=len(task_cfg.class_names),
        device=device,
    )
    transition_n = int(merged.get("freeze_backbone_epochs") or 0)
    if "head_hidden" not in merged:
        raise SystemExit("train_hyenadna.head_hidden is required in defaults.yaml.")
    try:
        head_hidden = int(merged["head_hidden"])
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            "train_hyenadna.head_hidden must be an integer (0 = linear, >0 = one-layer MLP)."
        ) from exc
    if head_hidden < 0:
        raise SystemExit("train_hyenadna.head_hidden must be >= 0.")
    head_dropout = float(merged.get("head_dropout") or 0.0)

    train_ds = RunTensorDataset(
        train_entries,
        requested_num_sets=num_sets,
        requested_max_len=max_len,
        cache_num_sets=cache_num_sets,
        cache_max_len=cache_max_len,
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
    elif train_sampler != "random":
        raise SystemExit(
            f"Unknown train_sampler {train_sampler!r} (use random or study_balanced)."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=loader_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(merged["num_workers"]),
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    val_ds = RunTensorDataset(
        val_entries,
        requested_num_sets=num_sets,
        requested_max_len=max_len,
        cache_num_sets=cache_num_sets,
        cache_max_len=cache_max_len,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=loader_batch_size,
        shuffle=False,
        num_workers=int(merged["num_workers"]),
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    ckpt_dir = resolve_repo_path(
        repo_root, str(paths_cfg["checkpoint_dir"]).strip()
    )
    model = HyenaDNAPreTrainedModel.from_pretrained(
        str(ckpt_dir),
        model_name,
        config=None,
        device=str(device),
        use_head=True,
        n_classes=len(task_cfg.class_names),
        head_pooling_mode=head_mode,
        head_hidden=head_hidden,
        head_dropout=head_dropout,
    )
    model = model.to(device)

    scaler: Optional[torch.amp.GradScaler] = None
    if amp_enabled and amp_use_grad_scaler:
        scaler = torch.amp.GradScaler("cuda")

    epochs = int(merged["epochs"])
    epoch_log: List[Dict[str, object]] = []
    best_tuning_score = float("-inf")
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    score_ctx = ScoreContext(
        requested_num_sets=num_sets,
        requested_max_len=max_len,
        cache_num_sets=cache_num_sets,
        cache_max_len=cache_max_len,
    )

    opt: Optional[torch.optim.Optimizer] = None

    results_path = _results_json_out_path(
        repo_root,
        merged.get("results_json"),
        task=task,
    )
    training_log_path: Optional[Path] = None
    if results_path is not None:
        training_log_path = _training_log_path_from_results(results_path)

    for ep in range(1, epochs + 1):
        print(f"\n--- Epoch {ep}/{epochs} ---", flush=True)

        if transition_n > 0:
            _set_backbone_requires_grad(model, ep > transition_n)
        else:
            _set_backbone_requires_grad(model, True)

        if ep == 1 or (transition_n > 0 and ep == transition_n + 1):
            opt = _make_optimizer(
                model,
                lr=lr,
                weight_decay=wd,
                backbone_lr_mult=backbone_lr_mult,
            )

        assert opt is not None

        last_loss = train_epoch(
            model,
            train_loader,
            opt,
            device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
            ce_weight=ce_weight,
        )

        val_loss = _eval_mean_ce_loss(
            model,
            val_loader,
            device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            ce_weight=ce_weight,
        )

        split_eval = eval_splits(
            model,
            val_entries=val_entries,
            test_entries=test_entries,
            holdout_entries=holdout_entries,
            task_cfg=task_cfg,
            ctx=score_ctx,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )

        lr_log = float(opt.param_groups[0]["lr"])
        ts = tuning_score(split_eval.val.f1, split_eval.val.auc, tuning_metric)
        epoch_row: Dict[str, object] = {
            "epoch": int(ep),
            "learning_rate": lr_log,
            "train_loss": float(last_loss),
            "val_loss": float(val_loss),
            "val_f1": float(split_eval.val.f1),
            "test_f1": float(split_eval.test.f1),
            "holdout_f1": float(split_eval.holdout.f1),
            "val_auc": float(split_eval.val.auc),
            "test_auc": float(split_eval.test.auc),
            "holdout_auc": float(split_eval.holdout.auc),
        }
        epoch_log.append(epoch_row)
        if training_log_path is not None:
            _write_training_log(training_log_path, epoch_log)

        if ts > best_tuning_score:
            best_tuning_score = float(ts)
            best_epoch = int(ep)
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        print(
            "  ".join(
                epoch_progress_fields(
                    split_eval=split_eval,
                    train_loss=last_loss,
                    val_loss=val_loss,
                    best_epoch=best_epoch,
                )
            ),
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    if results_path is not None:
        if best_epoch == 0:
            raise SystemExit("No best validation epoch recorded; cannot write results.")
        final_test = score_entries(
            model,
            test_entries,
            device,
            task_cfg,
            score_ctx,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        final_holdout = score_entries(
            model,
            holdout_entries,
            device,
            task_cfg,
            score_ctx,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        write_run_artifacts(
            results_path,
            merged=merged,
            cache_info={
                "dir": str(run_tensors_root),
                "hyenadna_run_tensors_num_sets": cache_num_sets,
                "hyenadna_run_tensors_max_length": cache_max_len,
                "sequence_cache_seq_offset": cache_seq_offset,
                "sequence_cache_min_seqs": cache_min_seqs,
                "n_cached_runs": len(all_records),
            },
            task_cfg=task_cfg,
            test_entries=test_entries,
            holdout_entries=holdout_entries,
            test_scores=final_test,
            holdout_scores=final_holdout,
            best_epoch=best_epoch,
            tuning_metric=tuning_metric,
            best_tuning_score=best_tuning_score,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
