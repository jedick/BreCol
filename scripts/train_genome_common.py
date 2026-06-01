"""Shared training helpers for the HyenaDNA and SetBERT fine-tuning scripts.

Both ``scripts/train_hyenadna.py`` and ``scripts/train_setbert.py`` consume the
shared ``genome_models`` block in ``defaults.yaml`` and follow the same
end-to-end recipe (build per-Run records, optionally study-balanced sampler,
AMP autocast, BCEWithLogits with class-balancing pos_weight, val/test/holdout
scoring at every epoch with best-epoch checkpoint selection, results JSON +
per-epoch training JSON + predictions CSVs). This module owns the helpers
common to that recipe plus the shared :class:`BinaryClassificationHead`, so
each training script keeps only its model-specific construction logic
(datasets, freezing, optimizer parameter groups).
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
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    ContextManager,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score

from shared_utilities import binary_auc_from_scores, resolve_repo_path

# ---------------------------------------------------------------------------
# Binary classification head (shared between HyenaDNA and SetBERT training)
# ---------------------------------------------------------------------------
#
# Both training scripts pool their backbone into a per-Run feature vector of
# shape ``[B, embed_dim]`` (HyenaDNA via ``head_pooling_mode``, SetBERT via the
# SAB ``[CLS]`` token) and then apply this head to obtain a single
# positive-class logit (BCEWithLogits-compatible).
#
# Variants (selected by ``kind``):
#   - ``linear``  Dropout -> Linear(d, 1)                          (BCE baseline)
#   - ``mlp``     Linear(d, h) -> GELU -> Dropout -> Linear(h, 1)  (one-hidden-layer MLP)
#   - ``cosine``  learnable_temperature * cosine(x, w)             (one normalized direction)
#
# ``head_dropout`` is applied to the input for ``linear`` and between GELU and
# the final ``Linear`` for ``mlp``. The ``cosine`` head ignores
# ``head_dropout``: ``F.normalize`` discards the standard ``1/(1-p)`` magnitude
# rescaling, so dropout would degrade into coordinate-dropping direction noise
# on the unit sphere - a different regularizer than the one users expect from
# the knob.

VALID_HEAD_TYPES = frozenset({"linear", "mlp", "cosine"})


class BinaryClassificationHead(nn.Module):
    """Map a per-Run feature ``[..., embed_dim]`` to a single positive-class logit ``[...]``."""

    def __init__(
        self,
        embed_dim: int,
        *,
        kind: str,
        head_dropout: float = 0.0,
        head_hidden: int = 0,
    ):
        super().__init__()
        kind = str(kind).strip().lower()
        if kind not in VALID_HEAD_TYPES:
            raise SystemExit(
                f"Unknown head_type {kind!r} (use one of {sorted(VALID_HEAD_TYPES)})."
            )
        self.kind = kind
        d = int(embed_dim)
        if d <= 0:
            raise SystemExit(f"BinaryClassificationHead embed_dim must be > 0; got {d}.")
        if kind == "linear":
            self.dropout = nn.Dropout(float(head_dropout))
            self.classifier = nn.Linear(d, 1)
        elif kind == "mlp":
            h = int(head_hidden)
            if h <= 0:
                raise SystemExit(
                    f"head_type='mlp' requires head_hidden > 0; got {h}."
                )
            self.classifier = nn.Sequential(
                nn.Linear(d, h),
                nn.GELU(),
                nn.Dropout(float(head_dropout)),
                nn.Linear(h, 1),
            )
        elif kind == "cosine":
            self.weight = nn.Parameter(torch.randn(d) * (d ** -0.5))
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(10.0)))
        else:  # pragma: no cover - guarded above
            raise SystemExit(f"Unknown head_type {kind!r}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "linear":
            return self.classifier(self.dropout(x)).squeeze(-1)
        if self.kind == "mlp":
            return self.classifier(x).squeeze(-1)
        if self.kind == "cosine":
            xn = F.normalize(x, dim=-1)
            wn = F.normalize(self.weight, dim=-1)
            return self.log_temperature.exp() * (xn @ wn)
        raise SystemExit(f"Unknown head_type {self.kind!r}.")


def validate_head_config(
    *,
    head_type: object,
    head_hidden: object,
) -> Tuple[str, int]:
    """Centralized validation of ``head_type`` / ``head_hidden`` from a merged config.

    Returns ``(head_type, head_hidden)`` as the normalized strings/ints both
    training scripts pass to ``BinaryClassificationHead``. Raises ``SystemExit``
    on bad values.
    """
    kind = str(head_type or "linear").strip().lower()
    if kind not in VALID_HEAD_TYPES:
        raise SystemExit(
            f"head_type must be one of {sorted(VALID_HEAD_TYPES)}; got {kind!r}."
        )
    raw = head_hidden
    try:
        h = int(raw) if raw is not None else 0
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"head_hidden must be an integer; got {raw!r}.") from exc
    if h < 0:
        raise SystemExit("head_hidden must be >= 0.")
    if kind == "mlp" and h <= 0:
        raise SystemExit(f"head_type='mlp' requires head_hidden > 0; got {h}.")
    return kind, h


# ---------------------------------------------------------------------------
# Task labels
# ---------------------------------------------------------------------------

VALID_TASKS = frozenset({"cancer_diagnosis", "cancer_type"})


def pos_neg_for_task(task: str) -> Tuple[str, str]:
    """Return ``(pos_label, neg_label)`` for a binary task."""
    if task == "cancer_diagnosis":
        return ("cancer", "healthy")
    if task == "cancer_type":
        return ("breast_cancer", "colorectal_cancer")
    raise SystemExit(f"Unknown task {task!r}.")


def task_abbrv(task: str) -> str:
    """Two-letter abbreviation used in templated results paths."""
    t = str(task).strip()
    if t == "cancer_diagnosis":
        return "cd"
    if t == "cancer_type":
        return "ct"
    raise SystemExit(
        f"Unknown task {task!r} (expected cancer_diagnosis or cancer_type)."
    )


# ---------------------------------------------------------------------------
# Class weighting and study-balanced sampling
# ---------------------------------------------------------------------------


def compute_pos_weight(
    train_entries: Sequence[Any],
    *,
    mode: str,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Return ``pos_weight`` for ``BCEWithLogitsLoss`` from ``mode``.

    ``mode``: ``none`` / ``null`` / ``""`` (no weighting), ``balanced``
    (``n_neg / n_pos``), or ``balanced_sqrt`` (``sqrt(n_neg / n_pos)``).
    Each entry must expose an integer ``.label`` of 0 or 1.
    """
    m = str(mode or "none").strip().lower()
    if m in ("none", "null", ""):
        return None
    n_pos = sum(1 for e in train_entries if int(e.label) == 1)
    n_neg = sum(1 for e in train_entries if int(e.label) == 0)
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


def study_sampler_weights(entries: Sequence[Any]) -> torch.Tensor:
    """Per-Run sampler weights for approximate uniform sampling over ``study_name``."""
    counts: Dict[str, int] = defaultdict(int)
    for e in entries:
        counts[str(e.study_name)] += 1
    w = [1.0 / counts[str(e.study_name)] for e in entries]
    return torch.tensor(w, dtype=torch.double)


# ---------------------------------------------------------------------------
# AMP
# ---------------------------------------------------------------------------

_AMP_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def resolve_amp_config(
    merged: Mapping[str, object],
    device: torch.device,
) -> Tuple[bool, torch.dtype, str, bool]:
    """Resolve ``(amp_enabled, amp_dtype, amp_dtype_name, use_grad_scaler)``.

    AMP is silently disabled (returns ``False`` for ``amp_enabled``) when CUDA
    is unavailable, so callers can build a single configuration that works on
    both CPU and GPU. ``use_grad_scaler`` is true only for ``float16``.
    """
    amp_requested = bool(merged.get("amp", False))
    amp_dtype_raw = str(merged.get("amp_dtype", "bfloat16")).strip().lower()
    if amp_dtype_raw not in _AMP_DTYPES:
        raise SystemExit("amp_dtype must be 'float16' or 'bfloat16'.")
    amp_dtype = _AMP_DTYPES[amp_dtype_raw]
    use_grad_scaler = amp_dtype is torch.float16

    if not amp_requested:
        return False, amp_dtype, amp_dtype_raw, use_grad_scaler
    if device.type != "cuda":
        print("AMP requested but CUDA is unavailable; running in float32.", flush=True)
        return False, amp_dtype, amp_dtype_raw, use_grad_scaler
    return True, amp_dtype, amp_dtype_raw, use_grad_scaler


def amp_autocast(
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> ContextManager[object]:
    if amp_enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


# ---------------------------------------------------------------------------
# Binary metrics
# ---------------------------------------------------------------------------


def binary_auc_and_f1(
    y_true_obj: np.ndarray,
    y_score: np.ndarray,
    *,
    pos_label: str,
    neg_label: str,
) -> Tuple[float, float]:
    """AUC + positive-class F1 (threshold 0.5) given task-label strings.

    ``y_true_obj`` is an object array of label strings; ``y_score`` is the
    positive-class probability. Empty inputs return ``(nan, nan)`` to mirror
    sklearn's behaviour. F1 uses ``zero_division=0`` so a single-class split
    returns 0 instead of warning.
    """
    y_true_obj = np.asarray(y_true_obj, dtype=object)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true_obj.size == 0:
        return float("nan"), float("nan")
    auc = binary_auc_from_scores(y_true_obj, y_score, positive_label=pos_label)
    y_pred = np.where(y_score >= 0.5, pos_label, neg_label)
    f1 = f1_score(
        y_true_obj,
        y_pred,
        pos_label=pos_label,
        average="binary",
        zero_division=0,
    )
    return float(auc), float(f1)


# ---------------------------------------------------------------------------
# Output writers (results.json, _training.json, predictions CSV)
# ---------------------------------------------------------------------------


def float_or_none(x: float) -> Optional[float]:
    """``float(x)`` for finite values; ``None`` for NaN (so JSON gets ``null``)."""
    return float(x) if x == x else None


def results_json_out_path(
    repo_root: Path,
    raw: Optional[object],
    *,
    task: str,
    script_stem: str,
) -> Optional[Path]:
    """Resolve a YAML ``results_json`` value to an absolute Path.

    ``None`` -> disabled. Empty string / ``"null"`` -> auto-named
    ``<script_stem>_<task>_<utc>.json`` under ``paths.results_scratch_dir``.
    Anything else is treated as a path (relative to ``repo_root`` if not
    absolute).
    """
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
        return scratch_base / f"{script_stem}_{task}_{ts}.json"
    p = Path(str(raw).strip()).expanduser()
    return p if p.is_absolute() else repo_root / p


def training_log_path_from_results(path: Path) -> Path:
    return path.with_name(f"{path.stem}_training.json")


_TRAINING_LOG_FIELDS = (
    "epoch",
    "learning_rate",
    "train_loss",
    "val_loss",
    "val_f1",
    "test_f1",
    "holdout_f1",
    "val_auc",
    "test_auc",
    "holdout_auc",
)


def write_training_log(path: Path, epoch_rows: Sequence[Mapping[str, object]]) -> None:
    """Write the per-epoch metrics log alongside the results JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: List[Dict[str, object]] = []
    for row in epoch_rows:
        out: Dict[str, object] = {}
        for key in _TRAINING_LOG_FIELDS:
            v = row[key]
            if key in ("epoch",):
                out[key] = int(v)
            elif key in ("learning_rate", "train_loss"):
                out[key] = float(v)
            else:
                out[key] = float_or_none(float(v))
        payload.append(out)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


_PREDICTION_CSV_FIELDS = ("Run", "task_label", "predicted_label", "positive_score")


def write_predictions_csv(
    path: Path,
    entries: Sequence[Any],
    scores: np.ndarray,
    *,
    pos_label: str,
    neg_label: str,
) -> None:
    """Write Run,task_label,predicted_label,positive_score CSV.

    Each entry must expose ``.run`` and ``.task_label``. ``scores`` is the
    positive-class probability in the same order as ``entries``. NaN scores
    are emitted as the literal ``nan`` and predicted_label defaults to
    ``neg_label`` (matches the threshold-0.5 convention).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(entries) != len(scores):
        raise SystemExit(
            f"write_predictions_csv: entries ({len(entries)}) and scores "
            f"({len(scores)}) length mismatch."
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=_PREDICTION_CSV_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        for e, s in zip(entries, scores):
            score = float(s)
            pred = pos_label if score >= 0.5 else neg_label
            writer.writerow(
                {
                    "Run": str(e.run),
                    "task_label": str(e.task_label),
                    "predicted_label": str(pred),
                    "positive_score": f"{score:.10f}",
                }
            )


def write_results_json(
    path: Path,
    *,
    script_filename: str,
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
    """Write the canonical results JSON.

    ``cache_info`` is freeform (each script writes its own keys: HyenaDNA
    records ``hyenadna_run_tensors_*`` shape; SetBERT records ``pretrained_*``
    identity).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": str(script_filename),
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
            "best_score": float_or_none(float(best_tuning_score)),
            "val_auc": float_or_none(best_val_auc),
            "val_f1": float_or_none(best_val_f1),
        },
        "metrics": {
            "test_auc": float_or_none(test_auc),
            "test_f1": float_or_none(test_f1),
            "holdout_auc": float_or_none(holdout_auc),
            "holdout_f1": float_or_none(holdout_f1),
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Grid orchestration (parent forks one child per (task, seed) combination)
# ---------------------------------------------------------------------------


def parse_seed_grid(raw: object) -> Optional[List[int]]:
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


def parse_task_grid(raw: object) -> Optional[List[str]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        task = raw.strip()
        if not task:
            return None
        if task not in VALID_TASKS:
            raise SystemExit(
                f"Unknown task {task!r} (use cancer_diagnosis or cancer_type)."
            )
        return [task]
    if isinstance(raw, (list, tuple)):
        if not raw:
            raise SystemExit("task grid must not be empty.")
        tasks = [str(t).strip() for t in raw]
        bad = [t for t in tasks if t not in VALID_TASKS]
        if bad:
            raise SystemExit(
                "task grid must list only cancer_diagnosis and/or "
                f"cancer_type; unknown value(s): {bad!r}."
            )
        return tasks
    raise SystemExit(
        f"task must be a string or a YAML list of task names; got {type(raw).__name__}."
    )


def clamp_sigsegv_retries(raw: object) -> int:
    """Clamp grid-mode subprocess SIGSEGV retries to the [0, 8] range."""
    try:
        n = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return max(0, min(int(n), 8))


def grid_child_subprocess_env() -> Dict[str, str]:
    """Environment for ``--child-run`` subprocesses; enables faulthandler."""
    env = dict(os.environ)
    env["PYTHONFAULTHANDLER"] = "1"
    return env


def run_grid_child(
    cmd: Sequence[str],
    repo_root: Path,
    *,
    sigsegv_retries: object,
    model_label: str,
) -> None:
    """Run one grid child process; retry on SIGSEGV up to ``sigsegv_retries`` extra times.

    SIGSEGV (signal 11) crashes during fine-tuning typically come from the CUDA
    stack / GPU driver rather than Python logic, so a bounded retry is a useful
    safety net for long grid runs. ``model_label`` ("HyenaDNA" / "SetBERT") only
    affects the user-facing error message on the final failure.
    """
    retries = clamp_sigsegv_retries(sigsegv_retries)
    max_attempts = 1 + retries
    env = grid_child_subprocess_env()
    for attempt in range(1, max_attempts + 1):
        try:
            subprocess.run(list(cmd), check=True, cwd=str(repo_root), env=env)
            return
        except subprocess.CalledProcessError as exc:
            sigsegv = getattr(signal, "SIGSEGV", None)
            is_sigsegv = sigsegv is not None and exc.returncode == -int(sigsegv)
            if is_sigsegv and attempt < max_attempts:
                print(
                    f"{model_label}: grid child died with SIGSEGV "
                    f"(attempt {attempt}/{max_attempts}); retrying after short delay.",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(min(float(attempt), 5.0))
                continue
            if is_sigsegv:
                raise SystemExit(
                    f"{model_label} training subprocess crashed with SIGSEGV "
                    "(signal 11). This normally comes from the CUDA stack or GPU "
                    "driver, not from Python logic; reruns often succeed. Raise "
                    "the section's sigsegv_retries in experiments.yaml to retry more."
                ) from exc
            raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_common_train_args(parser: argparse.ArgumentParser, *, model_label: str) -> None:
    """Add the five flags both training scripts share to ``parser``.

    ``model_label`` ("train_hyenadna" / "train_setbert") only customises the
    ``--expt`` and ``--results-json`` help text.
    """
    parser.add_argument(
        "--expt",
        type=int,
        default=None,
        help=f"Optional {model_label} experiment index from experiments.yaml (1-based). "
        "Omit or use 0 for defaults.yaml only.",
    )
    parser.add_argument(
        "--results-json",
        type=str,
        nargs="?",
        const="",
        default=argparse.SUPPRESS,
        help=(
            f"Override results JSON path. With no path, writes under results/scratch/ "
            f"as {model_label}_<task>_<utc>.json. Omit entirely to use YAML only."
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


def apply_cli_overrides(
    cli: argparse.Namespace,
    merged: Mapping[str, Any],
    *,
    expt: int,
    child_run: bool,
) -> Dict[str, Any]:
    """Apply ``--results-json`` / ``--override-task`` / ``--override-random-seed``.

    The ``--results-json`` rule mirrors the behaviour both scripts had inline:
    on the parent of a multi-cell grid (``--results-json ""`` and ``EXPT > 0``
    and not ``--child-run``), leave the value unset so the template/expansion
    logic decides per child. Otherwise the explicit value wins.
    """
    out = dict(merged)
    if hasattr(cli, "results_json"):
        rj = cli.results_json
        if rj == "" and expt > 0 and not child_run:
            pass
        elif rj == "":
            out["results_json"] = ""
        else:
            out["results_json"] = rj
    if cli.override_task is not None:
        out["task"] = str(cli.override_task).strip()
    if cli.override_random_seed is not None:
        out["random_seed"] = int(cli.override_random_seed)
    return out
