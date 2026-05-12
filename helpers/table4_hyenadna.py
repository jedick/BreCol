#!/usr/bin/env python3
"""
Build Table 4 (HyenaDNA cancer-type ablations) from JSON metrics under
``results/hyenadna/``.

Reads experiments 1..12 from ``experiments.yaml`` ``train_hyenadna.experiments``
(0-based indices 0..11) and, for each one, aggregates over all available random
seeds (files named ``{name}_<L>k_s<seed>.json``).

Output is an HTML table with one row per ablation, in experiment order:

  | Ablation (plain English) | Best epoch (per seed) | Test AUC | Holdout AUC |

* ``Best epoch`` lists the validation-selected best epoch per seed (from
  ``tuning.best_epoch``).
* ``Test AUC`` and ``Holdout AUC`` show ``mean ± standard deviation`` of
  ``metrics.test.roc_auc`` / ``metrics.holdout.roc_auc`` across seeds.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


# Plain-English row labels for experiments 1..12 in experiments.yaml.
# Descriptions may contain HTML (e.g. <sup>) and are intentionally not escaped
# when rendered into the <td>. Update this mapping whenever experiments.yaml
# rows 1..12 are reordered or renamed.
ABLATION_DESCRIPTIONS: Dict[str, str] = {
    "ct_best_recipe": "Best recipe (baseline)",
    "ct_amp_float16": "Mixed-precision training in float16 (instead of bfloat16)",
    "ct_no_amp": "Full-precision training (no mixed precision)",
    "ct_high_lr": "Higher learning rate (10<sup>\u22124</sup> instead of 10<sup>\u22125</sup>)",
    "ct_no_grad_clip": "No gradient clipping",
    "ct_no_study_balanced": "Random training sampler (no study-balanced sampling)",
    "ct_no_class_weight": "No class weighting",
    "ct_head_arch_mlp": "Multilayer perceptron classification head (instead of linear)",
    "ct_tuning_val_f1": "Tune by validation F1 (instead of validation AUC)",
    "ct_no_adv_delay": "No domain adversarial delay (study head active from epoch 1)",
    "ct_no_disc_warmup": "No study discriminator warm-up",
    "ct_no_dann": "No domain adversarial training",
}
ABLATION_ORDER: Tuple[str, ...] = tuple(ABLATION_DESCRIPTIONS.keys())
N_ABLATIONS: int = len(ABLATION_ORDER)


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _train_hyenadna_base(repo_root: Path) -> dict:
    cfg = _load_yaml(repo_root / "defaults.yaml")
    sec = cfg.get("train_hyenadna")
    if not isinstance(sec, dict):
        raise SystemExit("defaults.yaml must define train_hyenadna as a mapping.")
    return dict(sec)


def _experiments_1_to_12(repo_root: Path) -> List[Tuple[str, int]]:
    """Return (name, max_length) for experiments 1..12 (1-based) in order."""
    base = _train_hyenadna_base(repo_root)
    exp_cfg = _load_yaml(repo_root / "experiments.yaml").get("train_hyenadna") or {}
    rows = exp_cfg.get("experiments") or []
    if not isinstance(rows, list) or len(rows) < N_ABLATIONS:
        n_got = len(rows) if isinstance(rows, list) else 0
        raise SystemExit(
            "experiments.yaml train_hyenadna.experiments must list at least "
            f"{N_ABLATIONS} entries; got {n_got}."
        )
    out: List[Tuple[str, int]] = []
    for idx in range(N_ABLATIONS):
        row = rows[idx]
        if not isinstance(row, dict):
            raise SystemExit(f"experiments.yaml row {idx + 1} is not a mapping.")
        name = str(row.get("name") or "").strip()
        expected = ABLATION_ORDER[idx]
        if name != expected:
            raise SystemExit(
                f"experiments.yaml row {idx + 1}: expected name {expected!r}, "
                f"got {name!r}. Update ABLATION_DESCRIPTIONS to match."
            )
        overrides = row.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise SystemExit(f"{name}: overrides must be a mapping.")
        merged = {**base, **overrides}
        try:
            max_length = int(merged["max_length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(
                f"{name}: cannot resolve max_length from defaults+overrides ({exc})."
            )
        out.append((name, max_length))
    return out


def _len_token(max_length: int) -> str:
    if max_length <= 0 or max_length % 1024 != 0:
        raise SystemExit(
            f"max_length {max_length} is not a positive multiple of 1024; "
            "cannot reconstruct results JSON filename."
        )
    return f"{max_length // 1024}k"


_SEED_RE = re.compile(r"_s(\d+)\.json$")


def _seed_files(
    hyenadna_dir: Path, name: str, max_length: int
) -> List[Tuple[int, Path]]:
    tok = _len_token(max_length)
    candidates = sorted(hyenadna_dir.glob(f"{name}_{tok}_s*.json"))
    out: List[Tuple[int, Path]] = []
    for p in candidates:
        m = _SEED_RE.search(p.name)
        if m is None:
            continue
        out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def _read_metrics(path: Path) -> Tuple[int, float, float]:
    """Return (best_epoch, test_roc_auc, holdout_roc_auc) from one JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a JSON object.")

    tuning = data.get("tuning") or {}
    best_epoch = tuning.get("best_epoch")
    if not isinstance(best_epoch, int):
        raise SystemExit(f"{path}: missing or non-integer tuning.best_epoch.")

    metrics = data.get("metrics") or {}
    for split in ("test", "holdout"):
        blob = metrics.get(split)
        if not isinstance(blob, dict):
            raise SystemExit(
                f"{path}: expected metrics['{split}'] to be an object with 'roc_auc'."
            )
    test_v = metrics["test"].get("roc_auc")
    hold_v = metrics["holdout"].get("roc_auc")
    if test_v is None or hold_v is None:
        raise SystemExit(f"{path}: missing test or holdout roc_auc.")
    t = float(test_v)
    h = float(hold_v)
    if not math.isfinite(t) or not math.isfinite(h):
        raise SystemExit(f"{path}: non-finite test/holdout roc_auc ({t}, {h}).")
    return int(best_epoch), t, h


def _mean_std(values: List[float]) -> Tuple[float, Optional[float]]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) >= 2 else None
    return mean, std


def _fmt_mean_std(values: List[float], *, decimals: int) -> str:
    if not values:
        return "\u2014"  # em dash
    mean, std = _mean_std(values)
    if std is None:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} \u00b1 {std:.{decimals}f}"


def _fmt_epochs(values: List[int]) -> str:
    if not values:
        return "\u2014"
    return ", ".join(str(v) for v in values)


def collect_rows(
    hyenadna_dir: Path,
    experiments: List[Tuple[str, int]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    missing: List[str] = []
    for name, max_length in experiments:
        files = _seed_files(hyenadna_dir, name, max_length)
        if not files:
            missing.append(f"{name} ({_len_token(max_length)})")
            continue
        epochs: List[int] = []
        test_aucs: List[float] = []
        hold_aucs: List[float] = []
        for _seed, path in files:
            ep, t, h = _read_metrics(path)
            epochs.append(ep)
            test_aucs.append(t)
            hold_aucs.append(h)
        rows.append(
            {
                "name": name,
                "description": ABLATION_DESCRIPTIONS[name],
                "epochs": epochs,
                "test_aucs": test_aucs,
                "hold_aucs": hold_aucs,
                "n_seeds": len(files),
            }
        )
    if missing:
        miss = "\n".join(f"  - {m}" for m in missing)
        raise SystemExit(
            "Missing HyenaDNA results for ablations:\n"
            f"{miss}\n"
            f"Searched under: {hyenadna_dir}"
        )
    return rows


def format_table_html(
    rows: List[Dict[str, object]],
    *,
    decimals: int,
) -> str:
    thead = (
        "<thead>\n"
        "<tr>\n"
        "<th>Ablation</th>"
        "<th>Best epoch (per seed)</th>"
        "<th>Test AUC</th>"
        "<th>Holdout AUC</th>\n"
        "</tr>\n"
        "</thead>\n"
    )
    body_rows: List[str] = []
    for row in rows:
        desc = row["description"]  # raw HTML allowed (e.g. <sup>)
        ep_cell = html.escape(_fmt_epochs(row["epochs"]))
        test_cell = html.escape(_fmt_mean_std(row["test_aucs"], decimals=decimals))
        hold_cell = html.escape(_fmt_mean_std(row["hold_aucs"], decimals=decimals))
        body_rows.append(
            f"<tr>\n<td>{desc}</td>"
            f"<td>{ep_cell}</td>"
            f"<td>{test_cell}</td>"
            f"<td>{hold_cell}</td>\n</tr>"
        )
    tbody = "<tbody>\n" + "\n".join(body_rows) + "\n</tbody>\n"
    return f"<table>\n{thead}{tbody}</table>\n"


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hyenadna-dir",
        type=Path,
        default=repo_root / "results" / "hyenadna",
        help="Directory with {name}_<L>k_s<seed>.json files (default: results/hyenadna).",
    )
    p.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Decimal places for AUC mean/std (default: 3).",
    )
    args = p.parse_args()
    hyenadna_dir = args.hyenadna_dir.expanduser()
    if not hyenadna_dir.is_dir():
        raise SystemExit(f"Not a directory: {hyenadna_dir}")

    experiments = _experiments_1_to_12(repo_root)
    rows = collect_rows(hyenadna_dir, experiments)
    print(format_table_html(rows, decimals=args.decimals), end="", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
