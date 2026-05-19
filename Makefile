# Run from the repository root. Paths/defaults come from defaults.yaml.

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON ?= python3
CONFIG := $(ROOT)/defaults.yaml

# Read a top-level YAML section key from defaults.yaml via awk.
define yaml_section_value
$(strip $(shell awk -F': *' 'BEGIN{inside=0} $$1=="$(1)"{inside=1; next} inside && NF && $$0 !~ /^  /{inside=0} inside && $$1=="  $(2)"{print $$2; exit}' "$(CONFIG)"))
endef

DATA_DIR := $(call yaml_section_value,paths,data_dir)
DATASETS_CSV := $(ROOT)/$(call yaml_section_value,paths,datasets_csv)
TETRAMER_FREQUENCIES_CSV := $(call yaml_section_value,paths,tetramer_frequencies_csv)
TETRAMER_CACHE_DIR := $(call yaml_section_value,paths,tetramer_cache_dir)
EMBEDDING_CACHE_DIR := $(call yaml_section_value,paths,embedding_cache_dir)
RUN_TENSORS_DIR := $(ROOT)/$(call yaml_section_value,paths,run_tensors_dir)
SEQUENCE_CACHE_N_MAX := $(call yaml_section_value,sequence_cache,n_max_per_run)
EMB ?= 0
EMB_ARG := $(if $(filter 1,$(strip $(EMB))),--emb,)
UC_CAP_EMB_SHELL_ARG := $(if $(filter 1,$(strip $(EMB))),--emb)
UC_CAP_RESULTS_DIR := $(if $(filter 1,$(strip $(EMB))),results/embedding_uc_cap,results/tetramer_uc_cap)
EXPT ?=
EXPT_ARG := $(if $(strip $(EXPT)),--expt $(EXPT),)
FEAT ?=
FEAT_ARG := $(if $(strip $(FEAT)),--feat $(FEAT),)

# Expand fit_tetramer experiment output paths from experiments.yaml template+names.
CLASSIFIER_EXPERIMENT_OUTPUTS := $(shell $(PYTHON) -c "import yaml,pathlib; r=pathlib.Path('$(ROOT)'); cfg=yaml.safe_load((r/'experiments.yaml').read_text(encoding='utf-8')); sec=cfg.get('fit_classifier',{}); tpl=sec.get('results_json_template','results/{features}/{name}.json'); exps=sec.get('experiments',[]); print(' '.join(str((r/tpl.format(name=e['name'], features='tetramer')).resolve()) for e in exps))")
# Experiment name stems (same order as --expt 1..N); used for results/<uc_cap_dir>/<feat>/<name>.json.
EXPERIMENT_NAMES := $(shell $(PYTHON) -c "import yaml,pathlib; r=pathlib.Path('$(ROOT)'); cfg=yaml.safe_load((r/'experiments.yaml').read_text(encoding='utf-8')); exps=cfg.get('fit_classifier',{}).get('experiments',[]); print(' '.join(str(e['name']) for e in exps))")
# Enumerate fit_tetramer experiments so rules can run by --expt index.
CLASSIFIER_EXPERIMENT_INDICES := $(shell seq 1 $(words $(CLASSIFIER_EXPERIMENT_OUTPUTS)))
# Select a single fit_tetramer output path by 1-based EXPT index.
CLASSIFIER_EXPERIMENT_OUTPUT := $(if $(filter-out 0,$(strip $(EXPT))),$(word $(EXPT),$(CLASSIFIER_EXPERIMENT_OUTPUTS)),)

# CAP CSV outputs for each run_uc_cap_pipeline row (EMB selects tetramer vs embedding UC/CAP root).
UC_CAP_FEATURE_OUTPUTS := $(shell $(PYTHON) $(ROOT)/helpers/list_uc_cap_feature_outputs.py "$(ROOT)" $(UC_CAP_EMB_SHELL_ARG))
UC_CAP_FEATURE_INDICES := $(shell seq 1 $(words $(UC_CAP_FEATURE_OUTPUTS)))
UC_CAP_FEATURE_OUTPUT := $(if $(filter-out 0,$(strip $(FEAT))),$(word $(FEAT),$(UC_CAP_FEATURE_OUTPUTS)),)
# Deferred: FEAT/EXPT come from the command line when these are expanded as fit_uc_cap prerequisites.
UC_CAP_SINGLE_EXPT_ALL_FEAT_OUTPUTS = $(foreach f,$(UC_CAP_FEATURE_INDICES),$(ROOT)/$(UC_CAP_RESULTS_DIR)/$(f)/$(word $(EXPT),$(EXPERIMENT_NAMES)).json)
UC_CAP_SINGLE_FEAT_ALL_EXPT_OUTPUTS = $(foreach e,$(CLASSIFIER_EXPERIMENT_INDICES),$(ROOT)/$(UC_CAP_RESULTS_DIR)/$(FEAT)/$(word $(e),$(EXPERIMENT_NAMES)).json)
UC_CAP_FULL_GRID_OUTPUTS = $(foreach f,$(UC_CAP_FEATURE_INDICES),$(foreach e,$(CLASSIFIER_EXPERIMENT_INDICES),$(ROOT)/$(UC_CAP_RESULTS_DIR)/$(f)/$(word $(e),$(EXPERIMENT_NAMES)).json))
UC_CAP_BASELINE_CAP_CSV := $(shell $(PYTHON) $(ROOT)/helpers/list_uc_cap_feature_outputs.py "$(ROOT)" --baseline $(UC_CAP_EMB_SHELL_ARG))
UC_CAP_CACHE := $(if $(filter 1,$(strip $(EMB))),$(ROOT)/$(EMBEDDING_CACHE_DIR)/n$(SEQUENCE_CACHE_N_MAX)/_complete,$(ROOT)/$(TETRAMER_CACHE_DIR)/n$(SEQUENCE_CACHE_N_MAX)/_complete)

TETRA_CSV := $(ROOT)/$(TETRAMER_FREQUENCIES_CSV)
TETRAMER_CACHE := $(ROOT)/$(TETRAMER_CACHE_DIR)/n$(SEQUENCE_CACHE_N_MAX)/_complete
EMBEDDING_CACHE := $(ROOT)/$(EMBEDDING_CACHE_DIR)/n$(SEQUENCE_CACHE_N_MAX)/_complete

# Study metadata CSVs (typically small); used to rebuild tetramer frequencies when data change.
DATA_CSVS := $(shell find $(ROOT)/$(DATA_DIR) -type f -name '*.csv' 2>/dev/null)

.DEFAULT_GOAL := help

.PHONY: help download_data tetramer_frequencies tetramer_cache embedding_cache fit_tetramer \
	fit_uc_cap run_uc_cap train_hyenadna audit_run_tensors explain explain-%

help:
	@echo "LM-cancer-detection Makefile (script defaults from defaults.yaml)"
	@echo ""
	@echo "  make download_data"
	@echo "      Run scripts/download_sra_data.py."
	@echo ""
	@echo "  make tetramer_cache"
	@echo "      FASTA -> partitioned Parquet at $(TETRAMER_CACHE) via build_tetramer_cache.py."
	@echo ""
	@echo "  make tetramer_frequencies"
	@echo "      Append missing Run rows to tetramer frequencies CSV from the tetramer cache."
	@echo "      Depends on make tetramer_cache. Delete the frequencies CSV to force a full rebuild."
	@echo ""
	@echo "  make fit_tetramer"
	@echo "      Run scripts/fit_classifier.py --tetramer with defaults.yaml; default run passes"
	@echo "      --results-json (metrics under results/scratch/). Optional: EXPT=<n> for experiments."
	@echo "      Optional: EXPT=0 builds all configured experiments incrementally."
	@echo ""
	@echo "  make embedding_cache"
	@echo "      FASTA -> partitioned HyenaDNA embedding Parquet at $(EMBEDDING_CACHE)."
	@echo ""
	@echo "  make fit_uc_cap"
	@echo "      fit_classifier.py --uc_cap; default uses baseline CAP and --results-json (scratch)."
	@echo "      EMB=0 (default) → tetramer CAP / results/tetramer_uc_cap; EMB=1 → embedding paths."
	@echo "      FEAT=<n> / EXPT=<n> mirror experiments.yaml (same indices as run_uc_cap / fit_tetramer)."
	@echo "      FEAT=0 EXPT=<n>: one experiment, all feature sets → results/<dir>/1..N/<name>.json."
	@echo "      FEAT=<n> EXPT=0: one feature set, all experiments → results/<dir>/<n>/<name>.json"
	@echo "      FEAT=0 EXPT=0: full grid. Disallowed: FEAT=0 or EXPT=0 alone without the other axis."
	@echo ""
	@echo "  make run_uc_cap"
	@echo "      Build the baseline CAP CSV (defaults.yaml) if needed; incremental vs inputs."
	@echo "      EMB=0 (default) uses tetramer cache; EMB=1 uses embedding cache."
	@echo "      Optional: FEAT=<n> builds that feature-set CAP (1-based experiments.yaml index)."
	@echo "      Optional: FEAT=0 builds all configured UC/CAP pipeline feature sets incrementally."
	@echo ""
	@echo "  make run_tensors"
	@echo "      Run scripts/build_run_tensors.py once from defaults.yaml run_tensors settings."
	@echo ""
	@echo "  make audit_run_tensors"
	@echo "      Summarize outputs/run_tensors/*.pt coverage/utilization under cache_audit/run_tensors/."
	@echo ""
	@echo "  make train_hyenadna"
	@echo "      Run scripts/train_hyenadna.py against outputs/run_tensors/*.pt."
	@echo "      Optional: EXPT=<n> runs one train_hyenadna experiment from experiments.yaml."
	@echo ""
	@echo "  make explain TARGET=<make_target>"
	@echo "      Compact dependency/mtime explanation using make --trace."
	@echo "      Examples: make explain TARGET=tetramer_cache"
	@echo "                make explain-tetramer_cache"
	@echo ""

$(TETRA_CSV): $(TETRAMER_CACHE) $(DATA_CSVS) $(ROOT)/scripts/calculate_tetramer_frequencies.py \
		$(ROOT)/scripts/cache_operations.py $(ROOT)/defaults.yaml
	@mkdir -p "$(dir $(TETRA_CSV))"
	cd "$(ROOT)" && $(PYTHON) scripts/calculate_tetramer_frequencies.py

tetramer_frequencies: $(TETRA_CSV)
	@echo "Up to date: $(TETRA_CSV)"

$(TETRAMER_CACHE): $(DATA_CSVS) $(ROOT)/scripts/build_tetramer_cache.py \
		$(ROOT)/scripts/cache_operations.py \
		$(ROOT)/defaults.yaml
	cd "$(ROOT)" && $(PYTHON) scripts/build_tetramer_cache.py

tetramer_cache: $(TETRAMER_CACHE)
	@echo "Up to date: $(TETRAMER_CACHE)"

$(EMBEDDING_CACHE): $(DATA_CSVS) $(ROOT)/scripts/build_embedding_cache.py \
		$(ROOT)/scripts/cache_operations.py \
		$(ROOT)/scripts/hyenadna_fasta_data.py \
		$(ROOT)/scripts/hyenadna_sequence_embeddings.py \
		$(ROOT)/defaults.yaml
	cd "$(ROOT)" && $(PYTHON) scripts/build_embedding_cache.py

embedding_cache: $(EMBEDDING_CACHE)
	@echo "Up to date: $(EMBEDDING_CACHE)"

$(UC_CAP_BASELINE_CAP_CSV): $(UC_CAP_CACHE) \
		$(ROOT)/scripts/run_uc_cap_pipeline.py \
		$(ROOT)/defaults.yaml \
		$(ROOT)/helpers/list_uc_cap_feature_outputs.py
	cd "$(ROOT)" && $(PYTHON) scripts/run_uc_cap_pipeline.py $(EMB_ARG)

download_data:
	cd "$(ROOT)" && $(PYTHON) scripts/download_sra_data.py

ifeq ($(strip $(EXPT)),0)
fit_tetramer: $(CLASSIFIER_EXPERIMENT_OUTPUTS)
	@echo "Up to date: all fit_tetramer experiments"
else ifneq ($(strip $(EXPT)),)
fit_tetramer: $(CLASSIFIER_EXPERIMENT_OUTPUT)
	@if test -z "$(CLASSIFIER_EXPERIMENT_OUTPUT)"; then \
		echo "Invalid EXPT=$(EXPT). Use EXPT=0 for all, or EXPT=1..N from experiments.yaml."; \
		exit 2; \
	fi
else
fit_tetramer: $(TETRA_CSV) $(ROOT)/scripts/fit_classifier.py \
		$(ROOT)/defaults.yaml $(ROOT)/experiments.yaml
	cd "$(ROOT)" && $(PYTHON) scripts/fit_classifier.py --tetramer --results-json $(EXPT_ARG)
endif

define tetramer_experiment_rule
$(word $(1),$(CLASSIFIER_EXPERIMENT_OUTPUTS)): $(TETRA_CSV) \
		$(ROOT)/scripts/fit_classifier.py \
		$(ROOT)/defaults.yaml \
		$(ROOT)/experiments.yaml
	cd "$(ROOT)" && $(PYTHON) scripts/fit_classifier.py --tetramer --expt $(1)
endef

$(foreach i,$(CLASSIFIER_EXPERIMENT_INDICES),$(eval $(call tetramer_experiment_rule,$(i))))

$(RUN_TENSORS_DIR): $(DATA_CSVS) $(DATASETS_CSV) $(ROOT)/scripts/build_run_tensors.py \
		$(ROOT)/scripts/cache_operations.py \
		$(ROOT)/scripts/hyenadna_fasta_data.py \
		$(ROOT)/scripts/shared_utilities.py \
		$(ROOT)/defaults.yaml
	cd "$(ROOT)" && $(PYTHON) scripts/build_run_tensors.py

run_tensors: $(RUN_TENSORS_DIR)
	@echo "Up to date: $(RUN_TENSORS_DIR)"

audit_run_tensors: $(ROOT)/scripts/audit_run_tensors.py \
		$(ROOT)/scripts/shared_utilities.py \
		$(ROOT)/defaults.yaml
	cd "$(ROOT)" && $(PYTHON) scripts/audit_run_tensors.py

train_hyenadna: $(DATA_CSVS) $(DATASETS_CSV) $(ROOT)/scripts/train_hyenadna.py \
		$(ROOT)/scripts/cache_operations.py \
		$(ROOT)/scripts/hyenadna_fasta_data.py \
		$(ROOT)/scripts/shared_utilities.py \
		$(ROOT)/defaults.yaml $(ROOT)/experiments.yaml
	cd "$(ROOT)" && $(PYTHON) scripts/train_hyenadna.py --results-json $(EXPT_ARG)

ifeq ($(strip $(FEAT)),0)
run_uc_cap: $(UC_CAP_FEATURE_OUTPUTS)
	@echo "Up to date: all run_uc_cap pipeline feature sets"
else ifneq ($(strip $(FEAT)),)
run_uc_cap: $(UC_CAP_FEATURE_OUTPUT)
	@if test -z "$(UC_CAP_FEATURE_OUTPUT)"; then \
		echo "Invalid FEAT=$(FEAT). Use FEAT=0 for all, or FEAT=1..N from experiments.yaml."; \
		exit 2; \
	fi
	@:
else
run_uc_cap: $(UC_CAP_BASELINE_CAP_CSV)
	@:
endif

define uc_cap_feature_rule
$(word $(1),$(UC_CAP_FEATURE_OUTPUTS)): $(UC_CAP_CACHE) \
		$(ROOT)/scripts/run_uc_cap_pipeline.py \
		$(ROOT)/helpers/list_uc_cap_feature_outputs.py \
		$(ROOT)/defaults.yaml \
		$(ROOT)/experiments.yaml
	cd "$(ROOT)" && $(PYTHON) scripts/run_uc_cap_pipeline.py --feat $(1) $(EMB_ARG)
endef

$(foreach i,$(UC_CAP_FEATURE_INDICES),$(eval $(call uc_cap_feature_rule,$(i))))

ifneq ($(filter fit_uc_cap,$(MAKECMDGOALS)),)
ifeq ($(strip $(FEAT)),0)
ifeq ($(strip $(EXPT)),)
$(error fit_uc_cap: FEAT=0 requires EXPT=0 (full grid) or EXPT=1..N (one experiment, all feature sets).)
endif
endif
ifeq ($(strip $(EXPT)),0)
ifeq ($(strip $(FEAT)),)
$(error fit_uc_cap: EXPT=0 requires FEAT=1..N, or FEAT=0 EXPT=0 for the full grid.)
endif
endif
endif

define uc_cap_subdir_json_rule
$(ROOT)/$(UC_CAP_RESULTS_DIR)/$(1)/$(word $(2),$(EXPERIMENT_NAMES)).json: $(word $(1),$(UC_CAP_FEATURE_OUTPUTS)) \
		$(ROOT)/scripts/fit_classifier.py \
		$(ROOT)/defaults.yaml \
		$(ROOT)/experiments.yaml
	cd "$(ROOT)" && $(PYTHON) scripts/fit_classifier.py --uc_cap --feat $(1) --expt $(2) $(EMB_ARG) --results-json $(ROOT)/$(UC_CAP_RESULTS_DIR)/$(1)/$(word $(2),$(EXPERIMENT_NAMES)).json
endef
$(foreach f,$(UC_CAP_FEATURE_INDICES),$(foreach e,$(CLASSIFIER_EXPERIMENT_INDICES),$(eval $(call uc_cap_subdir_json_rule,$(f),$(e)))))

ifeq ($(strip $(FEAT)),0)
ifeq ($(strip $(EXPT)),0)
fit_uc_cap: $(UC_CAP_FULL_GRID_OUTPUTS) $(ROOT)/scripts/fit_classifier.py \
		$(ROOT)/defaults.yaml $(ROOT)/experiments.yaml
	@echo "Up to date: fit_uc_cap full grid ($(UC_CAP_RESULTS_DIR)/<k>/ for each experiment JSON)."
else ifneq ($(strip $(EXPT)),)
fit_uc_cap: $(UC_CAP_SINGLE_EXPT_ALL_FEAT_OUTPUTS) $(ROOT)/scripts/fit_classifier.py \
		$(ROOT)/defaults.yaml $(ROOT)/experiments.yaml
	@echo "Up to date: fit_uc_cap EXPT=$(EXPT) (all feature sets: $(UC_CAP_RESULTS_DIR)/<k>/$(word $(EXPT),$(EXPERIMENT_NAMES)).json)"
else
# FEAT=0 with EXPT unset: invalid for fit_uc_cap (see parse-time check when fit_uc_cap is a goal).
# Dummy rule avoids expanding $(word $(EXPT),...) during parsing for other goals (e.g. run_uc_cap FEAT=0).
fit_uc_cap: $(ROOT)/scripts/fit_classifier.py
	@:
endif
else ifeq ($(strip $(EXPT)),0)
ifneq ($(strip $(FEAT)),)
fit_uc_cap: $(UC_CAP_SINGLE_FEAT_ALL_EXPT_OUTPUTS)
	@echo "Up to date: fit_uc_cap FEAT=$(FEAT) (all experiments under $(UC_CAP_RESULTS_DIR)/$(FEAT)/)"
endif
else ifneq ($(strip $(FEAT)),)
fit_uc_cap: $(UC_CAP_FEATURE_OUTPUT) $(ROOT)/scripts/fit_classifier.py \
		$(ROOT)/defaults.yaml $(ROOT)/experiments.yaml
	@if test -z "$(UC_CAP_FEATURE_OUTPUT)"; then \
		echo "Invalid FEAT=$(FEAT). Use FEAT=1..N from experiments.yaml, FEAT=0 EXPT=0 (full grid), or FEAT=0 EXPT=1..N."; \
		exit 2; \
	fi
	cd "$(ROOT)" && $(PYTHON) scripts/fit_classifier.py --uc_cap --feat $(FEAT) $(EMB_ARG) $(EXPT_ARG)
else
fit_uc_cap: $(UC_CAP_BASELINE_CAP_CSV) $(ROOT)/scripts/fit_classifier.py \
		$(ROOT)/defaults.yaml $(ROOT)/experiments.yaml
	cd "$(ROOT)" && $(PYTHON) scripts/fit_classifier.py --uc_cap $(EMB_ARG) $(EXPT_ARG) --results-json
endif

TARGET ?=

explain:
	@if test -z "$(TARGET)"; then \
		echo "Usage: make explain TARGET=<make_target>"; \
		echo "   or: make explain-<make_target>"; \
		exit 2; \
	fi
	cd "$(ROOT)" && $(PYTHON) scripts/explain_make_trace.py "$(TARGET)"

explain-%:
	cd "$(ROOT)" && $(PYTHON) scripts/explain_make_trace.py "$*"
