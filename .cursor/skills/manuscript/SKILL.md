---
name: manuscript
description: >-
  Runs manuscript helper Python scripts from the repo root. User says /manuscript
  plus table N (filename number), figure N, optional multi-number batches, or
  all tables / all figures — numbers match table2_, table4_, figure1_, etc.
---

# Manuscript (helper script runner)

This skill only **executes** matching `helpers/table*.py` and `helpers/figure*.py` files. It does **not** describe metrics, HTML, or manuscript prose.

## Preconditions

- Working directory is the **repository root** (so `python helpers/...` and each script’s fixed `results/` / `manuscript/` paths work).

## How numbers map to files (not sorted index)

Numbers refer to the **digit immediately after `table` or `figure` in the basename**, before the next `_`.

- **Tables:** resolve *n* with glob `helpers/table{n}_*.py` (one underscore after the digits, then any suffix). Examples: `table2_tetramer.py` → *n* = 2; `table4_tetramer_uc_cap.py` → *n* = 4; `table6_embedding_uc_cap.py` → *n* = 6.
- **Figures:** resolve *n* with glob `helpers/figure{n}_*.py`. Examples: `figure1_tetramer_uc_cap.py` → *n* = 1; `figure3_embedding_uc_cap.py` → *n* = 3.

**Do not** use 1-based index into a sorted list of all `table*.py` files. Only the **filename number** *n* matters.

For each *n*:

- **Zero matches:** stop and tell the user clearly that **no** `helpers/table{n}_*.py` (or `figure{n}_*.py`) exists — do not substitute another script (e.g. do not run `table2_*.py` when they asked for `table 1`).
- **Exactly one match:** run `python helpers/<that basename>.py`.
- **More than one match:** stop; list the colliding paths and ask the user to disambiguate (should not happen with normal naming).

## Map user text to commands

After `/manuscript`, read the rest (case-insensitive keywords; flexible spacing).

### `table` + one or more numbers

Examples: `table 4`, `table 4 and 5`, `table4`, `table 2, 4, 5`.

1. Collect every positive integer the user intends as a **table** number. Prefer parsing integers from phrases that mention **table** and digits (e.g. split on `and`, commas, whitespace; accept `table4` / `table 4`).
2. For each number in **left-to-right order** as it appears in the message, resolve and run that script. If any resolution fails (zero or multiple matches), **stop** before running later numbers unless the user explicitly asked to continue on errors.

### `figure` + one or more numbers

Same rules using `helpers/figure{n}_*.py`.

### All helpers of a kind (no per-file number)

| User intent | Action |
|-------------|--------|
| `tables` or `all tables` | Run **every** `helpers/table*.py` whose basename matches `table[0-9]+_.+\.py` (i.e. has a numeric segment and underscore after `table`), sorted **lexicographically** by basename for a stable order. |
| `figures` or `all figures` | Same for `helpers/figure*.py` matching `figure[0-9]+_.+\.py`. |

If both “all tables” and “all figures” (or equivalent) appear in one request, run **all matching table scripts first**, then **all matching figure scripts**.

### No match

If the message does not fit any row above, ask for one of: `table <n>…`, `figure <n>…`, `tables`, `figures`.

## Execute

For each script path `helpers/<name>.py`:

```bash
python helpers/<name>.py
```

Run from repo root. If a run exits non-zero, report stderr/stdout as needed and **stop** (do not run remaining scripts in a batch unless the user explicitly asked to continue on errors).

## Notes for the agent

- Do not open or edit `manuscript/manuscript.md` as part of this skill unless the user separately asks.
- Do not add CLI flags; helpers are no-CLI runners.
