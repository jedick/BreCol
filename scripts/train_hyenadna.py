#!/usr/bin/env python3
"""
Fine-tune HyenaDNA with a single-task binary classification head (BCEWithLogits on a
single positive-class logit; matches scripts/train_setbert.py).

Uses a pre-built feature-only per-run tensor cache under paths.hyenadna_run_tensors_dir/ from
scripts/build_hyenadna_run_tensors.py, joins labels/splits from shared metadata at runtime, and
reports run-level AUC and F1 on the val/test/holdout splits (one score per Run).

Training logs (`*_training.json`) record per-epoch loss, binary F1, and AUC metrics aligned
with console output. Checkpoints are selected by ``genome_models.tuning_metric`` (default
``auc``).

Config: defaults.yaml (``genome_models`` + ``train_hyenadna`` + ``hyenadna_run_tensors`` +
``paths``) with optional experiments.yaml ``train_hyenadna`` overrides (--expt). The shared
``genome_models`` block holds settings common to HyenaDNA and SetBERT (checkpoint identity,
device/AMP, head, task/optimization); ``train_hyenadna`` holds HyenaDNA-specific knobs
(``num_sets``, ``max_length``, ``batch_size``, ``freeze_backbone_epochs``,
``head_pooling_mode``).

Intermittent ``Signals.SIGSEGV`` during grid runs usually indicates a GPU driver or
PyTorch CUDA native crash rather than bad Python tensor logic. We use
``PYTHONFAULTHANDLER=1`` for child processes to get more info.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from hyenadna import HyenaDNAPreTrainedModel
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from cache_operations import load_sequence_row_selection
from hyenadna_fasta_data import (
    merge_train_hyenadna_config,
    model_max_length,
)
from shared_utilities import (
    build_run_task_table,
    resolve_repo_path,
)
from train_genome_common import (
    VALID_TASKS,
    BinaryClassificationHead,
    add_common_train_args,
    amp_autocast,
    apply_cli_overrides,
    binary_auc_and_f1,
    compute_pos_weight,
    parse_seed_grid,
    parse_task_grid,
    pos_neg_for_task,
    resolve_amp_config,
    results_json_out_path,
    run_grid_child,
    study_sampler_weights,
    task_abbrv,
    training_log_path_from_results,
    validate_head_config,
    write_predictions_csv,
    write_results_json,
    write_training_log,
)

HEAD_MODES = ("last", "first", "pool", "sum")


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
    """Per-split scoring result.

    ``entries`` and ``y_score`` cover *every* input run (one row per Run, NaN
    for missing-cache runs); the prediction CSV iterates this list. ``auc`` and
    ``f1`` are computed over the NaN-filtered subset.
    """

    entries: Tuple[RunRecord, ...]
    y_score: np.ndarray
    auc: float
    f1: float


@dataclass(frozen=True)
class SplitScores:
    val: HeadScores
    test: HeadScores
    holdout: HeadScores


@dataclass(frozen=True)
class TaskConfig:
    pos_label: str
    neg_label: str


# ---------------------------------------------------------------------------
# Run record construction
# ---------------------------------------------------------------------------


def prepare_task_config(
    task: str,
    defaults_path: Path,
) -> Tuple[Any, TaskConfig]:
    """Build the run metadata frame and resolve the (pos/neg) task labels."""
    run_df = build_run_task_table(task, config_path=defaults_path)
    y_train = run_df.loc[run_df["split"] == "train", "task_label"].to_numpy(dtype=object)
    if y_train.size == 0:
        raise SystemExit("No training runs found after shared split assignment.")
    pos_label, neg_label = pos_neg_for_task(task)
    return run_df, TaskConfig(pos_label=pos_label, neg_label=neg_label)


def build_run_records(
    run_task_df,
    cache_root: Path,
    *,
    task_cfg: TaskConfig,
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
        if primary_label == task_cfg.pos_label:
            label = 1
        elif primary_label == task_cfg.neg_label:
            label = 0
        else:
            raise SystemExit(
                f"Unexpected task_label {primary_label!r} for run {run!r}; "
                f"expected {task_cfg.pos_label!r} or {task_cfg.neg_label!r}."
            )
        out.append(
            RunRecord(
                run=run,
                split=str(row["split"]),
                label=int(label),
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


# ---------------------------------------------------------------------------
# Model wrapper: HyenaDNA backbone + shared binary classification head
# ---------------------------------------------------------------------------


class HyenaDNABinaryClassifier(nn.Module):
    """Wrap a pooled-feature HyenaDNA backbone with a ``BinaryClassificationHead``.

    The backbone (``HyenaDNAPreTrainedModel.from_pretrained(use_head=True, ...)``)
    returns ``[B, d_model]`` after the configured ``head_pooling_mode``; this
    wrapper applies the shared head and returns a single positive-class logit
    per sample. Mirrors ``SetBertBinaryClassifier`` in train_setbert.py.

    The head's Linear layers are re-initialized with ``normal_(std=0.02)`` /
    ``zeros_`` to match the upstream HyenaDNA ``_init_weights`` scheme that
    used to apply when the head lived inside ``HyenaDNAModel``. The cosine
    head's raw ``nn.Parameter`` is unaffected (matches old behaviour, where
    ``_init_weights`` also skipped non-Linear parameters).
    """

    def __init__(
        self,
        base_model: nn.Module,
        *,
        embed_dim: int,
        head_type: str,
        head_dropout: float = 0.0,
        head_hidden: int = 0,
        init_normal_std: float = 0.02,
    ):
        super().__init__()
        self.base_model = base_model
        self.head = BinaryClassificationHead(
            int(embed_dim),
            kind=head_type,
            head_dropout=float(head_dropout),
            head_hidden=int(head_hidden),
        )
        if init_normal_std is not None:
            std = float(init_normal_std)
            for module in self.head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.base_model(input_ids))


def training_loss_sum_and_count(
    model: torch.nn.Module,
    batch: Mapping[str, object],
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    pos_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int]:
    input_ids = batch["input_ids"].to(device)
    bsz, n_set, seq_len = input_ids.shape
    flat_in = input_ids.view(bsz * n_set, seq_len)
    nv = batch["n_sets"]
    labels = batch["label"].to(device)
    mask = torch.arange(n_set, device=device).unsqueeze(0) < nv.to(device).unsqueeze(1)

    with amp_autocast(device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
        logits = model(flat_in)
        # Wrapper head emits [B*N]; reshape to (B, N) per-set positive-class logits.
        logits = logits.view(bsz, n_set, -1).squeeze(-1)
        flat_logits = logits[mask]
        flat_y = labels.unsqueeze(1).expand(bsz, n_set)[mask].float()
    if flat_logits.numel() == 0:
        return torch.zeros((), device=device), 0
    bce_kw: Dict[str, object] = {"reduction": "sum"}
    if pos_weight is not None:
        # AMP can produce bf16/fp16 logits; pos_weight tensor must match.
        bce_kw["pos_weight"] = pos_weight.to(
            device=flat_logits.device,
            dtype=flat_logits.dtype,
        )
    bce = F.binary_cross_entropy_with_logits(flat_logits, flat_y, **bce_kw)
    return bce, int(flat_y.shape[0])


def _eval_mean_bce_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    pos_weight: Optional[torch.Tensor],
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
                pos_weight=pos_weight,
            )
            total += float(loss_sum.item())
            count += int(n)
    return total / max(count, 1)


def _make_optimizer(
    model: HyenaDNABinaryClassifier,
    *,
    lr: float,
    weight_decay: float,
    backbone_lr_mult: float,
) -> torch.optim.Optimizer:
    backbone_params = [p for p in model.base_model.parameters() if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    groups: List[Dict[str, object]] = []
    if backbone_params:
        blr = lr * float(backbone_lr_mult)
        groups.append({"params": backbone_params, "lr": blr})
    if head_params:
        groups.append({"params": head_params, "lr": lr})
    if not groups:
        raise SystemExit("No trainable parameters for optimizer.")
    return torch.optim.AdamW(groups, weight_decay=float(weight_decay))


def _set_backbone_requires_grad(model: HyenaDNABinaryClassifier, trainable: bool) -> None:
    for p in model.base_model.parameters():
        p.requires_grad = trainable
    for p in model.head.parameters():
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
    pos_weight: Optional[torch.Tensor],
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
            pos_weight=pos_weight,
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


def _positive_class_score(logits: torch.Tensor) -> float:
    """Run-level score: mean positive-class logit over sequence sets, then sigmoid."""
    agg = logits.view(-1).mean(dim=0)
    return float(torch.sigmoid(agg).item())


def _head_metrics(
    entries: Sequence[RunRecord],
    scores: np.ndarray,
    task_cfg: TaskConfig,
) -> HeadScores:
    """Build a HeadScores from entries + scores (NaN-tolerant).

    Metric computation filters NaN scores; the stored ``entries`` and
    ``y_score`` retain every input row so prediction CSVs cover all of them.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return HeadScores(tuple(entries), scores, float("nan"), float("nan"))
    valid = scores == scores
    if not valid.any():
        return HeadScores(tuple(entries), scores, float("nan"), float("nan"))
    sub_entries = [e for e, ok in zip(entries, valid) if ok]
    y_true = np.array([e.task_label for e in sub_entries], dtype=object)
    sub_scores = scores[valid]
    auc, f1 = binary_auc_and_f1(
        y_true, sub_scores, pos_label=task_cfg.pos_label, neg_label=task_cfg.neg_label
    )
    return HeadScores(tuple(entries), scores, auc, f1)


def _forward_entry_score(
    model: torch.nn.Module,
    entry: RunRecord,
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
    with amp_autocast(device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
        logits = model(x[:nv])
    return _positive_class_score(logits)


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
        return HeadScores((), np.asarray([], dtype=np.float64), float("nan"), float("nan"))
    model.eval()
    raw: List[float] = []
    with torch.no_grad():
        for e in entries:
            raw.append(
                _forward_entry_score(
                    model,
                    e,
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


def epoch_progress_line(
    *,
    split_eval: SplitScores,
    train_loss: float,
    val_loss: float,
    best_epoch: int,
) -> str:
    """One-line console log; mirrors SetBERT (3 decimals, paired AUC/F1 per split)."""
    return (
        f"train_loss={train_loss:.3f} val_loss={val_loss:.3f}  "
        f"val_auc={split_eval.val.auc:.3f} val_f1={split_eval.val.f1:.3f}  "
        f"test_auc={split_eval.test.auc:.3f} test_f1={split_eval.test.f1:.3f}  "
        f"holdout_auc={split_eval.holdout.auc:.3f} "
        f"holdout_f1={split_eval.holdout.f1:.3f}  "
        f"best_epoch={int(best_epoch)}"
    )


# ---------------------------------------------------------------------------
# Argument parsing and per-experiment grid driver
# ---------------------------------------------------------------------------


def _parse_argv(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_train_args(parser, model_label="train_hyenadna")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.expt is not None and args.expt < 0:
        raise SystemExit("--expt must be >= 0.")
    return args


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
        task_abbrv=task_abbrv(task),
        name=name,
        seed=int(seed),
        max_length=int(max_length),
        max_length_k=int(max_length_k),
    )


# ---------------------------------------------------------------------------
# Results / prediction writing
# ---------------------------------------------------------------------------


def write_run_artifacts(
    results_path: Path,
    *,
    merged: Mapping[str, Any],
    cache_info: Mapping[str, Any],
    task_cfg: TaskConfig,
    task: str,
    test_entries: Sequence[RunRecord],
    holdout_entries: Sequence[RunRecord],
    test_scores: HeadScores,
    holdout_scores: HeadScores,
    best_epoch: int,
    tuning_metric: str,
    best_tuning_score: float,
    best_val_auc: float,
    best_val_f1: float,
) -> None:
    """Write results JSON and split prediction CSVs."""
    test_pred_path = results_path.with_name(f"{results_path.stem}_test.csv")
    holdout_pred_path = results_path.with_name(f"{results_path.stem}_holdout.csv")
    write_predictions_csv(
        test_pred_path,
        test_scores.entries if test_scores.entries else test_entries,
        test_scores.y_score,
        pos_label=task_cfg.pos_label,
        neg_label=task_cfg.neg_label,
    )
    write_predictions_csv(
        holdout_pred_path,
        holdout_scores.entries if holdout_scores.entries else holdout_entries,
        holdout_scores.y_score,
        pos_label=task_cfg.pos_label,
        neg_label=task_cfg.neg_label,
    )

    write_results_json(
        results_path,
        script_filename=Path(__file__).name,
        merged=merged,
        cache_info=cache_info,
        task=task,
        pos_label=task_cfg.pos_label,
        neg_label=task_cfg.neg_label,
        best_epoch=best_epoch,
        tuning_metric=tuning_metric,
        best_tuning_score=best_tuning_score,
        best_val_auc=best_val_auc,
        best_val_f1=best_val_f1,
        test_auc=test_scores.auc,
        test_f1=test_scores.f1,
        holdout_auc=holdout_scores.auc,
        holdout_f1=holdout_scores.f1,
    )

    print(
        f"Final eval (best epoch by {tuning_metric}): best_epoch={best_epoch}  "
        f"test_auc={test_scores.auc:.3f} test_f1={test_scores.f1:.3f}  "
        f"holdout_auc={holdout_scores.auc:.3f} holdout_f1={holdout_scores.f1:.3f}\n",
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

    merged = apply_cli_overrides(cli, merged, expt=expt, child_run=cli.child_run)

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

        task_grid = parse_task_grid(overrides.get("task")) or [str(merged["task"]).strip()]
        seed_grid = parse_seed_grid(overrides.get("random_seed")) or [int(merged["random_seed"])]
        grid_max_length = int(
            model_max_length(str(merged["hyenadna_model"]).strip(), merged.get("max_length"))
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
                    run_grid_child(
                        cmd,
                        repo_root,
                        sigsegv_retries=merged.get("sigsegv_retries"),
                        model_label="HyenaDNA",
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
    if task not in VALID_TASKS:
        raise SystemExit(
            f"Unknown train_hyenadna.task {task!r} "
            "(use cancer_diagnosis or cancer_type)."
        )
    model_name = str(merged["hyenadna_model"]).strip()
    num_sets = int(merged["num_sets"])
    loader_batch_size = _resolve_train_batch_size(merged["batch_size"])
    max_len = model_max_length(model_name, merged.get("max_length"))
    head_mode = str(merged["head_pooling_mode"]).strip()
    if head_mode not in HEAD_MODES:
        raise SystemExit(f"head_pooling_mode must be one of {HEAD_MODES}; got {head_mode!r}.")
    head_type, head_hidden = validate_head_config(
        head_type=merged.get("head_type"),
        head_hidden=merged.get("head_hidden"),
    )
    head_dropout = float(merged.get("head_dropout") or 0.0)
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

    run_task_df, task_cfg = prepare_task_config(task, defaults_path)

    all_records, n_skipped_missing_cache, missing_cache_examples = build_run_records(
        run_task_df,
        run_tensors_root,
        task_cfg=task_cfg,
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
    amp_enabled, amp_dtype, amp_dtype_name, amp_use_grad_scaler = resolve_amp_config(
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
        f"pos={task_cfg.pos_label!r} neg={task_cfg.neg_label!r} | "
        f"head_type={head_type} head_pooling_mode={head_mode} | "
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
    pos_weight = compute_pos_weight(
        train_entries,
        mode=class_weight_mode,
        device=device,
    )
    if pos_weight is not None:
        print(
            f"  pos_weight={float(pos_weight.item()):.3f} (mode={class_weight_mode})",
            flush=True,
        )
    transition_n = int(merged.get("freeze_backbone_epochs") or 0)

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
            study_sampler_weights(train_entries),
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
    base_model = HyenaDNAPreTrainedModel.from_pretrained(
        str(ckpt_dir),
        model_name,
        config=None,
        device=str(device),
        use_head=True,
        head_pooling_mode=head_mode,
    )
    embed_dim = int(base_model.d_model)
    model = HyenaDNABinaryClassifier(
        base_model,
        embed_dim=embed_dim,
        head_type=head_type,
        head_dropout=head_dropout,
        head_hidden=head_hidden,
    )
    model = model.to(device)

    scaler: Optional[torch.amp.GradScaler] = None
    if amp_enabled and amp_use_grad_scaler:
        scaler = torch.amp.GradScaler("cuda")

    epochs = int(merged["epochs"])
    epoch_log: List[Dict[str, object]] = []
    best_tuning_score = float("-inf")
    best_epoch = 0
    best_val_auc = float("nan")
    best_val_f1 = float("nan")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    score_ctx = ScoreContext(
        requested_num_sets=num_sets,
        requested_max_len=max_len,
        cache_num_sets=cache_num_sets,
        cache_max_len=cache_max_len,
    )

    opt: Optional[torch.optim.Optimizer] = None

    results_path = results_json_out_path(
        repo_root,
        merged.get("results_json"),
        task=task,
        script_stem="train_hyenadna",
    )
    training_log_path: Optional[Path] = None
    if results_path is not None:
        training_log_path = training_log_path_from_results(results_path)

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
            pos_weight=pos_weight,
        )

        val_loss = _eval_mean_bce_loss(
            model,
            val_loader,
            device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            pos_weight=pos_weight,
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
            write_training_log(training_log_path, epoch_log)

        if ts > best_tuning_score:
            best_tuning_score = float(ts)
            best_epoch = int(ep)
            best_val_auc = float(split_eval.val.auc)
            best_val_f1 = float(split_eval.val.f1)
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        print(
            epoch_progress_line(
                split_eval=split_eval,
                train_loss=last_loss,
                val_loss=val_loss,
                best_epoch=best_epoch,
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
            task=task,
            test_entries=test_entries,
            holdout_entries=holdout_entries,
            test_scores=final_test,
            holdout_scores=final_holdout,
            best_epoch=best_epoch,
            tuning_metric=tuning_metric,
            best_tuning_score=best_tuning_score,
            best_val_auc=best_val_auc,
            best_val_f1=best_val_f1,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
