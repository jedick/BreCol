---
name: manuscript-tables
description: >-
  Refreshes Table 1, 3, or 4 in manuscript.md from classifier JSON.
  User picks the table with a digit after the skill (e.g. /manuscript-tables 1):
  1 = tetramer (table1_tetramer.py), 3 = UC/CAP selected triple
  (table3_uc_cap.py), 4 = HyenaDNA cancer-type ablations
  (table4_hyenadna.py). Replace HTML only between the matching
  <!-- classifier-table-N --> markers.
---

# Manuscript tables

## How to invoke

The user selects **one** table with a digit **1**, **3**, or **4** after the skill (for example `/manuscript-tables 1`, `/manuscript-tables 3`, or “manuscript-tables **4**”). Follow **only** the matching section below for that run. Do not refresh other tables unless the user asks.

| Digit | Table | Command | HTML markers in `manuscript.md` |
| ---: | --- | --- | --- |
| **1** | Tetramer classifiers | `python helpers/table1_tetramer.py` | `<!-- classifier-table-1 -->` … `<!-- /classifier-table-1 -->` |
| **3** | UC/CAP selected triple (test + holdout) | `python helpers/table3_uc_cap.py` | `<!-- classifier-table-3 -->` … `<!-- /classifier-table-3 -->` |
| **4** | HyenaDNA cancer-type ablations (test + holdout, mean ± std over seeds) | `python helpers/table4_hyenadna.py` | `<!-- classifier-table-4 -->` … `<!-- /classifier-table-4 -->` |

If no digit is given, ask which table (1, 3, or 4) to refresh, or whether to run all three in order (1 then 3 then 4).

All commands assume the **repository root** as the current working directory.

---

# Manuscript Table 1 (classifier AUC)

## Goal

Keep **Table 1** in `manuscript.md` in sync with **eight JSON files** under `results/tetramer/`:

- `cancer_diagnosis_{baseline,knn,svm,random_forest}.json`
- `cancer_type_{baseline,knn,svm,random_forest}.json`

(produced by `make fit_tetramer` / `scripts/fit_classifier.py --tetramer` via `experiments.yaml`).

## Steps

1. From the **repository root**, run:

   ```bash
   python helpers/table1_tetramer.py
   ```

   Optional overrides:
   - `--tetramer-dir PATH` (default: `results/tetramer`)
   - `--decimals N` (default: 3)
   - `--markdown` — pipe table (two-line header) instead of default **HTML** nested header table

2. Open **`manuscript.md`**. Find the block between these markers:

   - `<!-- classifier-table-1 -->`
   - `<!-- /classifier-table-1 -->`

3. **Replace only the lines between those two markers** (not the markers themselves) with the script’s stdout. Default output is an HTML `<table>` (nested headers: task × test/holdout). Preserve one newline after the closing marker if the surrounding prose expects it.

4. Save the manuscript. Follow **`.cursor/rules/manuscript-prose.mdc`** for any prose edits nearby (for example update surrounding sentences if the table now includes holdout and random forest).

## Notes

- Rows: **Majority class** (baseline), **KNN**, **SVM**, **Random Forest**. Columns: per task, **Test** and **Holdout** ROC AUC.
- If the script exits with “Missing expected JSON”, run `make fit_tetramer` (or add the missing `{task}_{model}` experiment) until all eight files exist.

---

# Manuscript Table 3 (selected UC/CAP triple, test + holdout)

## Goal

Sync **Table 3** with KNN, SVM, and random forest JSON for one feature triple. Defaults match the manuscript prose: *n*<sub>UC</sub> = 2000, *K* = 5000, *n*<sub>CAP</sub> = 10000 (resolved to `results/uc_cap/<feat>/` via `experiments.yaml` + `defaults.yaml`).

## Steps

1. From the repository root:

   ```bash
   python helpers/table3_uc_cap.py
   ```

   Optional: `--n-uc`, `--n-clusters`, `--n-cap`, `--uc-cap-dir`, `--decimals`.

2. Replace lines between `<!-- classifier-table-3 -->` and `<!-- /classifier-table-3 -->` in `manuscript.md` with stdout.

---

# Manuscript Table 4 (HyenaDNA cancer-type ablations, test + holdout)

## Goal

Sync **Table 4** in the *Improving stability with domain adversarial training* subsection with HyenaDNA cancer-type ablation JSON under `results/hyenadna/`.

Rows correspond to experiments **1–12** in `experiments.yaml` `train_hyenadna.experiments` (in that order):

1. `ct_best_recipe` — Best recipe (baseline)
2. `ct_amp_float16`
3. `ct_no_amp`
4. `ct_high_lr`
5. `ct_no_grad_clip`
6. `ct_no_study_balanced`
7. `ct_no_class_weight`
8. `ct_head_arch_mlp`
9. `ct_tuning_val_f1`
10. `ct_no_adv_delay`
11. `ct_no_disc_warmup`
12. `ct_no_dann`

The helper resolves each row’s plain-English label, gathers every seed file matching `{name}_<L>k_s<seed>.json` under `results/hyenadna/`, and emits an HTML table with four columns: **Ablation**, **Best epoch (per seed)**, **Test AUC**, **Holdout AUC**. Test/holdout cells show `mean ± std` (sample standard deviation; omitted when only one seed is available).

## Steps

1. From the repository root:

   ```bash
   python helpers/table4_hyenadna.py
   ```

   Optional: `--hyenadna-dir` (default `results/hyenadna`), `--decimals` (default `3`).

2. Replace lines between `<!-- classifier-table-4 -->` and `<!-- /classifier-table-4 -->` in `manuscript.md` with stdout. Preserve the markers and surrounding blank lines.

## Notes

- The script reads `tuning.best_epoch`, `metrics.test.roc_auc`, and `metrics.holdout.roc_auc` from each JSON, so the result files must come from `scripts/train_hyenadna.py` (via `make train_hyenadna EXPT=<n>`).
- Plain-English row labels live in `ABLATION_DESCRIPTIONS` at the top of `helpers/table4_hyenadna.py`. If `experiments.yaml` rows 1–12 are renamed or reordered, update that mapping (the script exits with a clear error when names mismatch).
- Missing JSON for an ablation: re-run `make train_hyenadna EXPT=<n>` for that experiment index (1..12).
