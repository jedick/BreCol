# BreCol: Multi-Study Benchmark for Cancer Detection from Gut Microbiomes

**BreCol** is a curated benchmark of 2,040 16S rRNA sequencing runs across 26 studies
spanning breast cancer, colorectal cancer, and healthy cohorts.
Its central question: *do microbiome-based cancer classifiers generalize to studies they have never seen?*

---

## Overview

The gut microbiome is increasingly linked to cancer risk. Machine learning models trained
on fecal microbiome profiles have shown promise for distinguishing cancer patients from
healthy controls. But a persistent problem undermines many published results: when
test samples are drawn from the same studies used for training, performance estimates are
inflated by study-level signals (e.g. differences in sequencing protocol, primer choice,
and regional microbiome variation) rather than genuine cancer biology.

BreCol addresses this by structuring evaluation around **temporal holdout**: the 26
included studies are divided chronologically into a development partition (pre-2023)
and a holdout partition (2023 and later). Models are trained and tuned on development
studies and evaluated on holdout studies they have never seen, reflecting how a
model would actually perform on future data.

We benchmark two approaches to feature representation:

- **Classical ML** — k-nearest neighbors, random forest, and SVM applied to either
  run-level tetramer frequencies or cluster abundance profiles (UC/CAP, described below).
- **Deep learning** — two pre-trained genome language models, HyenaDNA and SetBERT,
  fine-tuned for cancer classification.

Classical models reach test/holdout AUCs of **0.77/0.60** for cancer diagnosis
(cancer vs. healthy) and **1.00/0.83** for cancer type (breast vs. colorectal).
Both deep learning models underperform the best classical methods on holdout data.
Our reference-free **UC/CAP** feature method achieves the best overall holdout
performance without relying on taxonomic databases.

---

## Study Design and Methods

### Temporal holdout split

The 26 studies are split by publication year into two partitions for each cancer type:
the first seven studies (pre-2023) form the **development** partition, and the six
most recent studies (2023 onward) form the **holdout** partition. Development runs are
further divided 70/10/20 into training, validation, and test splits. Holdout studies
are never seen during training or hyperparameter selection.

This design creates a realistic challenge: predictions must transfer to datasets that
became available only after the model was trained, eliminating the shortcut of learning
study-level technical signals instead of cancer biology.

### Two classification tasks

- **Cancer diagnosis** — cancer vs. healthy, using all samples.
- **Cancer type** — breast vs. colorectal, using cancer-positive samples only.

Cancer type is the harder generalization task: because breast and colorectal samples
almost always come from different studies, a model can achieve near-perfect in-study
accuracy by learning study identity rather than disease. Holdout evaluation removes
this shortcut.

### Feature representations

**Run-level tetramer frequencies.** All 4-mer counts are summed across the sequences
in a run and converted to relative frequencies, producing a single 256-dimensional
vector per run. Tetranucleotide (4-mer) frequencies are reference-free, so they
avoid any dependence on curated taxonomic databases. The tradeoff is that averaging
across sequences discards within-run compositional structure.

**Unsupervised clustering / cluster abundance profiles (UC/CAP).** To recover
within-run structure, sequences are clustered by tetramer composition using k-means
(fit only on training-split sequences) to produce a sequence codebook. Each run is
then represented by the distribution of its sequences across clusters: a cluster
abundance profile (CAP). This is conceptually analogous to OTU-based methods but
entirely reference-free.

**HyenaDNA.** A long-range genomic sequence model pre-trained on the human reference
genome. Sequences from each run are packed into context windows and the backbone
hidden states are mean-pooled across token positions to produce a run-level embedding
for classification.

**SetBERT.** A transformer pre-trained on ~280,000 microbial 16S samples with a
relative-abundance prediction objective. Each read is encoded by a DNABERT encoder;
a stack of Set Attention Blocks (SABs) contextualizes the reads from a single run and
produces a [CLS] embedding summarizing the run. Unlike HyenaDNA, SetBERT was
pre-trained on microbial sequences, making its representations directly relevant
to the domain.

For both deep learning models, three classification head architectures were tested:
linear, MLP, and cosine similarity.

---

## Dataset Compilation

A major contribution of BreCol is the curation of a multi-study benchmark dataset.
The `data/` directory holds one CSV file per study (named by first-author initials and
year) under `data/breast/` and `data/colorectal/`. The `datasets.csv` file at the
repository root records the development/holdout partition assignment for each study.

### Study list

Sample counts reflect post-subsampling sizes. Several studies have substantially larger
sample counts than others; **stratified subsampling** (by cancer/healthy label) was
applied to improve study balance across the benchmark.

| Ref | Year | BioProject | Type | Cancer | Healthy | Partition | Country |
|---|---|---|---|---|---|---|---|
| [AAM+13](https://journals.lww.com/ajg/Fulltext/2013/10001/Fecal_Microbiota_Composition_in_Women_In_Relation.625.aspx) | 2013 | [PRJNA396901](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA396901/) | breast | 29 | 32 | development | United States |
| [GJH+15](https://doi.org/10.1093/jnci/djv147) | 2015 | [PRJNA345373](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA345373/) | breast | 47 | 47 | development | United States |
| [GHB+18](https://doi.org/10.1038/bjc.2017.435) | 2018 | [PRJNA383849](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA383849/) | breast | 48 | 48 | development | United States |
| [BVW+21](https://doi.org/10.1002/ijc.33473) | 2021 | [PRJNA658160](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA658160/) | breast | 57 | 63 | development | Ghana |
| [BSR+22](https://doi.org/10.1038/s41598-022-23793-7) | 2022 | [PRJEB54599](https://www.ncbi.nlm.nih.gov/bioproject/PRJEB54599/) | breast | 19 | 14 | development | United States |
| [WZK+22](https://doi.org/10.3389/fmicb.2022.894283) | 2022 | [PRJNA804967](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA804967/) | breast | 54 | 25 | development | China |
| [ZZZ+22](https://doi.org/10.1111/jam.15620) | 2022 | [PRJNA726050](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA726050/) | breast | 14 | 14 | development | China |
| [SKC+23](https://doi.org/10.1038/s41598-023-27436-3) | 2023 | [PRJNA872152](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA872152/) | breast | 22 | 21 | **holdout** | United States |
| [LBA+25](https://doi.org/10.3390/ijms26146801) | 2025 | [PRJNA1127492](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1127492/) | breast | 76 | 16 | **holdout** | Spain |
| [SYL+25](https://doi.org/10.1128/msystems.00879-25) | 2025 | [PRJNA1243283](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1243283/) | breast | 10 | 10 | **holdout** | China |
| [MTK+26](https://doi.org/10.1016/j.gutmic.2026.100009) | 2026 | [PRJNA914483](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA914483/) | breast | 32 | 32 | **holdout** | Malaysia |
| [SVK+26](https://doi.org/10.21203/rs.3.rs-8921895/v1) | 2026 | [PRJNA1356467](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1356467/) | breast | 22 | 30 | **holdout** | India |
| [YTK+26](https://doi.org/10.1007/s44411-026-00523-3) | 2026 | [PRJNA1190698](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1190698/) | breast | 15 | 15 | **holdout** | Turkey |
| [ZTV+14](https://doi.org/10.15252/msb.20145645) | 2014 | [PRJEB6070](https://www.ncbi.nlm.nih.gov/bioproject/PRJEB6070/) | colorectal | 41 | 75 | development | France |
| [BRRS16](https://doi.org/10.1186/s13073-016-0290-3) | 2016 | [PRJNA290926](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA290926/) | colorectal | 64 | 94 | development | United States/Canada |
| [OKN+21](https://doi.org/10.1038/s41467-021-25965-x) | 2021 | [PRJDB11246](https://www.ncbi.nlm.nih.gov/bioproject/PRJDB11246/) | colorectal | 67 | 51 | development | Japan |
| [YDS+21](https://doi.org/10.1038/s41467-021-27112-y) | 2021 | [PRJNA763023](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA763023/) | colorectal | 65 | 43 | development | China |
| [YWS+21](https://doi.org/10.1186/s13073-021-00844-8) | 2021 | [PRJEB36789](https://www.ncbi.nlm.nih.gov/bioproject/PRJEB36789/) | colorectal | 53 | 52 | development | Argentina/Chile/India/Vietnam |
| [DLT+22](https://doi.org/10.3389/fphys.2022.854545) | 2022 | [PRJNA824020](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA824020/) | colorectal | 27 | 33 | development | China |
| [PCL+22](https://doi.org/10.1038/s41598-022-14203-z) | 2022 | [PRJNA662014](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA662014/) | colorectal | 36 | 25 | development | Singapore |
| [BWY+23](https://doi.org/10.1186/s12866-023-02805-0) | 2023 | [PRJEB53415](https://www.ncbi.nlm.nih.gov/bioproject/PRJEB53415/) | colorectal | 46 | 43 | **holdout** | India |
| [BRR+24](https://doi.org/10.1186/s12864-024-10621-7) | 2024 | [PRJEB71787](https://www.ncbi.nlm.nih.gov/bioproject/PRJEB71787/) | colorectal | 51 | 51 | **holdout** | Spain |
| [CAB+24](https://doi.org/10.1002/1878-0261.13604) | 2024 | [PRJNA911189](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA911189/) | colorectal | 90 | 30 | **holdout** | Spain |
| [SGH+24](https://doi.org/10.1016/j.micpath.2024.106726) | 2024 | [PRJNA1059759](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1059759/) | colorectal | 10 | 10 | **holdout** | India |
| [ARF+25](https://doi.org/10.3389/fmicb.2025.1449642) | 2025 | [PRJEB76625](https://www.ncbi.nlm.nih.gov/bioproject/PRJEB76625/) | colorectal | 25 | 15 | **holdout** | Iran |
| [GYX+25](https://doi.org/10.1186/s12866-024-03721-7) | 2025 | [PRJNA1092526](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1092526/) | colorectal | 67 | 64 | **holdout** | China |
| | | [PRJNA1092376](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1092376/) | | | | | |

### Per-study CSV format

Each CSV file contains one row per sequencing run. Key columns:

- `Run` — SRA Run accession (SRR, ERR, or DRR prefix)
- `BioSample` — SRA BioSample accession
- `sample_label` — normalized label: `healthy`, `breast_cancer`, `colorectal_cancer`, or `benign`
- `sample_used` — Boolean; `TRUE` for runs included in the analysis

Benign samples (adenomas, benign colon polyps, DCIS) and non-fecal samples are
excluded from all modeling (`sample_used = FALSE`). Other study-specific columns
(e.g. `cohort`) vary by file.

---

## Data Analysis Pipeline

Install dependencies first, then run pipeline steps in order.

### Installation

```
pip install -r requirements.txt
```

This installs all Python dependencies, including the local `hyenadna` and `setbert`
packages in editable mode.

### Pipeline steps

```
make download_data          # Download 16S sequences from SRA (~13 GB)
make -j4 tetramer_cache     # Build hive-partitioned Parquet cache (~16 min, 1.5 GB)
make tetramer_frequencies   # Sum cached counts per run (~2 min)
make -j4 fit_tetramer EXPT=0       # Train tetramer classifiers (~1 min)
make run_uc_cap FEAT=0             # Build cluster abundance profiles (~29 min, 13 GB RAM)
make -j4 fit_uc_cap FEAT=0 EXPT=0  # Train UC/CAP classifiers (~13 min)
make hyenadna_run_tensors          # Build HyenaDNA input tensors (~12 min, 2.5 GB)
make train_hyenadna EXPT=0         # Fine-tune HyenaDNA (~6 hr)
make setbert_run_tensors           # Build SetBERT input tensors (~43 min)
make train_setbert EXPT=0          # Fine-tune SetBERT (~11.5 hr)
```

After running the above, regenerate manuscript tables and figures:

```
python helpers/table*.py
python helpers/figure*.py
```

**Notes on `EXPT` and `FEAT`:**
`EXPT=0` runs all experiments listed in `experiments.yaml` for a target;
`FEAT=0` builds all UC/CAP feature sets. Use `EXPT=N` or `FEAT=N` to run a single
entry. Non-GPU steps support `-j` for parallel execution.
`make fit_uc_cap` requires both `FEAT` and `EXPT` to be specified when sweeping
over one of them.

**Debugging Make targets:**
`make explain-<target>` prints why Make would rebuild a target and its full
prerequisite chain. For example: `make explain-run_uc_cap FEAT=0`.

**Hardware:** The full pipeline runs in approximately 20 hours on a machine with
8 CPU cores, 40 GB RAM, and a 16 GB NVIDIA GPU.

---

## Code Details

### HyenaDNA

Source files are under `hyenadna/`. The package is installed by `requirements.txt`,
or manually from the repo root:

```
pip install -e hyenadna
```

- `standalone_hyenadna.py` — downloaded from [HazyResearch/hyena-dna](https://github.com/HazyResearch/hyena-dna/tree/d553021b483b82980aa4b868b37ec2d4332e198a)
- `huggingface_wrapper.py` and `inference_example.py` — extracted from the [HyenaDNA Colab Notebook](https://colab.research.google.com/drive/1wyVEQd4R3HYLTUOXEEQmp_I8aNC_aLhL)

Local modifications are summarized in comments within each file. To verify the
installation: `cd hyenadna; python -c 'import inference_example as ex; ex.inference_single()'`

Pre-trained checkpoint: `hyenadna-small-32k-seqlen`, cloned from Hugging Face
on first use if not already present under `paths.checkpoint_dir`.

### SetBERT

Source files are under `setbert/`. The package was repackaged from three upstream
sources:

- [deepbio-toolkit](https://github.com/DLii-Research/deepbio-toolkit/tree/f46d4c10d77d090d4ccf74b4eee8872de5f7cfeb) v0.4.5
- [dbtk-dnabert](https://github.com/DLii-Research/dbtk-dnabert/tree/a4e50615d7c782cbed673115f0b89fd91754cac5) v1.2.3
- [dbtk-setbert](https://github.com/DLii-Research/setbert/tree/ecb5dc7181e0221e029fdeff694dc92c73cdac9d) v1.0.3

Local modifications:

- Use SDPA (scaled dot-product attention) for faster multi-head attention,
  including the relative position bias in `RelativeMultiHeadAttention`.
- Set `use_reentrant=False` on the activation-checkpoint call so gradients reach
  the DNABERT encoder.
- Match the destination embedding dtype to the encoder output to avoid bf16/fp32
  conversions under AMP.

Pre-trained checkpoint: `qiita-16s` (12-layer DNABERT encoder, 768-dimensional
embeddings, 6-layer SAB transformer). Note: the SetBERT paper describes a 64-dim
model (`64d-silva16s-250bp`) that is not published on Hugging Face.

---

## Repository Layout

```
data/               Per-study CSV files (breast/ and colorectal/)
datasets.csv        Study list with partition assignments (development/holdout)
defaults.yaml       Default pipeline parameters
experiments.yaml    Named experiment configurations for Make targets
scripts/            Analysis scripts called by Make targets
helpers/            Scripts for generating manuscript tables and figures
manuscript/         Manuscript source, figures, and tables
hyenadna/           Local HyenaDNA package
setbert/            Local SetBERT package
Makefile            Pipeline entry point
requirements.txt    Python dependencies
```

---

## Citation

If you use BreCol data or code, please cite the accompanying manuscript (forthcoming).
Raw sequencing data are available at the SRA accessions listed in the study table above.
