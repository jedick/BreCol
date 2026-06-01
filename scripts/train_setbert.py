#!/usr/bin/env python3
"""
Fine-tune SetBERT with a binary classification head on the transformed ``[CLS]``
token (per the SetBERT paper, Suppl. §3.5). Loss is ``BCEWithLogits`` on a single
positive-class logit (matches scripts/train_hyenadna.py).

Reads a feature-only per-Run token tensor cache from paths.setbert_run_tensors_dir/
(scripts/build_setbert_run_tensors.py), joins labels/splits from shared metadata via
shared_utilities.build_run_task_table(task), and reports run-level AUC and F1 on the
val/test/holdout splits.

Single-task per run: ``genome_models.task`` is ``cancer_diagnosis`` or ``cancer_type``.

Two-knob backbone freezing (the classification head is always trainable):
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

Config: defaults.yaml (``genome_models`` + ``train_setbert`` + ``paths``) with optional
experiments.yaml ``train_setbert.experiments`` overrides selected by ``--expt N``
(1-based). The baseline merge combines ``genome_models`` (checkpoint identity, device,
AMP, head, task / optimization knobs - shared with HyenaDNA training) with
``train_setbert`` (set construction, batch sizes, backbone freezing); the two blocks
must not share keys. An experiment row may override any of those keys (e.g. ``head_type:
cosine``, ``device: cpu``). When the selected experiment row sets ``random_seed`` and/or
``task`` to a YAML list, the parent process forks one ``--child-run`` subprocess per
combination (results JSON path templated by ``train_setbert.results_json_template``);
each subprocess runs a single training.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from setbert_data import load_genome_models_section, load_setbert_model
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

VALID_TUNING = frozenset(("auc", "f1"))
VALID_CLASS_WEIGHTS = frozenset(("none", "null", "", "balanced", "balanced_sqrt"))
VALID_TRAIN_SAMPLERS = frozenset(("random", "study_balanced"))


# ---------------------------------------------------------------------------
# Config merge (defaults.yaml + experiments.yaml row overrides)
# ---------------------------------------------------------------------------


def _load_train_setbert_section(defaults_path: Path) -> Dict[str, Any]:
    """Merged baseline = ``genome_models`` block + ``train_setbert`` block.

    Shared settings (checkpoint identity ``setbert_repo`` / ``setbert_revision``,
    ``device``, ``amp`` / ``amp_dtype``, ``task``, ``head_type`` and the
    task / optimization knobs common to both HyenaDNA and SetBERT training)
    live in ``genome_models``; everything SetBERT-specific (set construction,
    batch sizes, backbone freezing) lives in ``train_setbert``. Experiment
    overrides in experiments.yaml may target keys from either block (e.g.
    ``head_type: cosine``, ``device: cpu``).
    """
    cfg = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise SystemExit(f"{defaults_path} must contain a YAML mapping.")
    shared = cfg.get("genome_models")
    if not isinstance(shared, dict):
        raise SystemExit(f"{defaults_path} must define a genome_models mapping.")
    train_sec = cfg.get("train_setbert")
    if not isinstance(train_sec, dict):
        raise SystemExit(f"{defaults_path} must define a train_setbert mapping.")
    overlap = set(shared) & set(train_sec)
    if overlap:
        raise SystemExit(
            "genome_models and train_setbert must not share keys; conflict: "
            f"{sorted(overlap)!r}."
        )
    return {**shared, **train_sec}


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


def _format_results_template(
    template: str, *, task: str, name: str, seed: int
) -> str:
    return str(template).format(
        task=str(task).strip(),
        task_abbrv=task_abbrv(task),
        name=str(name),
        seed=int(seed),
    )


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


# ---------------------------------------------------------------------------
# Model wrapper: SetBERT + shared binary classification head (BCE)
# ---------------------------------------------------------------------------


class SetBertBinaryClassifier(nn.Module):
    """Wrap ``SetBert`` and predict via a shared :class:`BinaryClassificationHead`.

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
        head_hidden: int = 0,
    ):
        super().__init__()
        self.base_model = base_model
        self.head = BinaryClassificationHead(
            int(embed_dim),
            kind=str(head_type),
            head_dropout=float(head_dropout),
            head_hidden=int(head_hidden),
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
        return self.head(out["class"])


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
# Loss, scoring
# ---------------------------------------------------------------------------


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
        with amp_autocast(device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
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
        with amp_autocast(device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
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
    return model.head(out["class"])


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
        with amp_autocast(device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
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
    """Per-Run binary AUC + F1; thin SetBERT-side wrapper around the shared metric."""
    y_true = np.array([e.task_label for e in entries], dtype=object)
    return binary_auc_and_f1(y_true, scores, pos_label=pos_label, neg_label=neg_label)


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
        with amp_autocast(device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
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


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def _parse_argv(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_train_args(parser, model_label="train_setbert")
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
    merged = apply_cli_overrides(cli, merged, expt=expt, child_run=cli.child_run)

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
        task_grid = parse_task_grid(overrides.get("task")) or [str(merged["task"]).strip()]
        seed_grid = parse_seed_grid(overrides.get("random_seed")) or [int(merged["random_seed"])]

        if len(task_grid) > 1 or len(seed_grid) > 1:
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
            total = int(len(task_grid) * len(seed_grid))
            completed = 0
            skipped = 0
            print(
                f"Running EXPT={expt} grid: {len(task_grid)} task(s) x {len(seed_grid)} seed(s) "
                f"= {total} run(s) (outer: task, inner: seed).",
                flush=True,
            )
            for task_v in task_grid:
                for seed in seed_grid:
                    out_rel = _format_results_template(
                        template, task=str(task_v), name=str(exp_name), seed=int(seed)
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
                        model_label="SetBERT",
                    )
                    completed += 1
            print(
                f"Finished EXPT={expt} grid: launched={completed}, "
                f"skipped_existing={skipped}, total={total}.",
                flush=True,
            )
            return 0

        # Single-cell expt: maybe still apply template to choose the output path.
        if (
            isinstance(template, str)
            and template.strip()
            and exp_name
            and merged.get("results_json") in (None, "null")
        ):
            out_rel = _format_results_template(
                template,
                task=str(merged["task"]).strip(),
                name=str(exp_name),
                seed=int(merged["random_seed"]),
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
    # Validate that file-level ``genome_models`` defaults exist and are well-typed; the
    # actual values used come from ``merged`` so experiments.yaml overrides (e.g.
    # ``device: cpu``) still take effect.
    load_genome_models_section(defaults_cfg)
    pretrained_repo = str(merged["setbert_repo"]).strip()
    pretrained_revision = str(merged["setbert_revision"]).strip()
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

    pos_label, neg_label = pos_neg_for_task(task)
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
    if sample_repo and sample_repo != pretrained_repo:
        raise SystemExit(
            f"Cache pretrained_repo {sample_repo!r} does not match "
            f"genome_models.setbert_repo {pretrained_repo!r}."
        )
    if sample_rev and sample_rev != pretrained_revision:
        raise SystemExit(
            f"Cache pretrained_revision {sample_rev!r} does not match "
            f"genome_models.setbert_revision {pretrained_revision!r}."
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

    amp_enabled, amp_dtype, amp_dtype_name, amp_use_scaler = resolve_amp_config(
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
        f"Loading SetBERT {pretrained_repo}@{pretrained_revision} on {device} "
        f"(sequence_encoder_chunk_size={chunk_size})",
        flush=True,
    )
    try:
        base_model, _tokenizer, embed_dim, pad_token_id, kmer = load_setbert_model(
            pretrained_repo=pretrained_repo,
            pretrained_revision=pretrained_revision,
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
        raise SystemExit("head_dropout must be >= 0.")
    head_type, head_hidden = validate_head_config(
        head_type=merged.get("head_type"),
        head_hidden=merged.get("head_hidden"),
    )
    merged = {
        **merged,
        "head_type": head_type,
        "head_hidden": head_hidden,
    }
    model = SetBertBinaryClassifier(
        base_model,
        embed_dim=embed_dim,
        head_type=head_type,
        head_dropout=head_dropout,
        head_hidden=head_hidden,
    ).to(device)

    pos_weight = compute_pos_weight(
        train_entries, mode=class_weight_mode, device=device
    )
    if pos_weight is not None:
        print(
            f"  pos_weight={float(pos_weight.item()):.3f} (mode={class_weight_mode})",
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
            study_sampler_weights(train_entries),
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

    results_path = results_json_out_path(
        repo_root,
        merged.get("results_json"),
        task=task,
        script_stem="train_setbert",
    )
    training_log_path: Optional[Path] = None
    if results_path is not None:
        training_log_path = training_log_path_from_results(results_path)

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
            write_training_log(training_log_path, epoch_log)

        print(
            f"  train_loss={train_loss:.3f} val_loss={val_loss:.3f}  "
            f"val_auc={val_auc:.3f} val_f1={val_f1:.3f}  "
            f"test_auc={test_auc:.3f} test_f1={test_f1:.3f}  "
            f"holdout_auc={holdout_auc:.3f} holdout_f1={holdout_f1:.3f}  "
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
        write_results_json(
            results_path,
            script_filename=Path(__file__).name,
            merged=merged,
            cache_info={
                "dir": str(run_tensors_root),
                "cache_set_size": int(sample_meta.get("set_size", 0)),
                "pretrained_repo": pretrained_repo,
                "pretrained_revision": pretrained_revision,
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
        write_predictions_csv(
            results_path.with_name(f"{results_path.stem}_test_predictions.csv"),
            test_entries,
            final_test_scores,
            pos_label=pos_label,
            neg_label=neg_label,
        )
        write_predictions_csv(
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
