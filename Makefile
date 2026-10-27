# One-command entry points (brief section 12). make stops at the first failing
# command, so a failed stage never feeds a later one.
# Windows: use any GNU make (e.g. `mingw32-make`, Git-for-Windows make, or WSL).
# Scripts are run by file path because numeric prefixes are not valid module names.

PY ?= python
CONFIG ?= configs/default.yaml
QUICK = configs/quick.yaml
# Overleaf package that receives the paper figures (outside this repository).
PAPER_DIR ?= ../04_Paper_Overleaf_Package/paper
TECTONIC ?= tectonic

.PHONY: setup smoke data preprocess eda baselines activations probes analysis \
        intervention report all test lint diagrams paper guide \
        decomposition commitment-mock commitment commitment-analysis choke-paper \
        recovery-labels recovery-mock recovery-analysis \
        chokebench evaluator-robustness story

# Counterfactual commitment (stages 12-13). GPU for 12; override on the command line.
GENERATOR ?= Qwen2.5-1.5B-Instruct
MODEL ?= Qwen/$(GENERATOR)
M ?= 16
TP ?= 1
DTYPE ?= auto
ROLLOUTS ?= results/commitment/$(GENERATOR).jsonl
EXTRA ?=

setup:
	$(PY) -m pip install -r requirements.txt

smoke:
	$(PY) scripts/01_download_data.py --config $(QUICK)
	$(PY) scripts/02_preprocess.py --config $(QUICK)
	$(PY) scripts/03_run_eda.py --config $(QUICK)
	$(PY) scripts/04_run_statistics.py --config $(QUICK)
	$(PY) scripts/05_train_baselines.py --config $(QUICK)
	$(PY) scripts/06_extract_activations.py --config $(QUICK)
	$(PY) scripts/07_train_probes.py --config $(QUICK)
	$(PY) scripts/08_lead_time_analysis.py --config $(QUICK)
	$(PY) scripts/09_simulate_intervention.py --config $(QUICK)
	$(PY) scripts/10_build_report.py --config $(QUICK)

data:
	$(PY) scripts/01_download_data.py --config $(CONFIG)

preprocess:
	$(PY) scripts/02_preprocess.py --config $(CONFIG)

eda:        # E1 + E2 -> Table I, F1-F5
	$(PY) scripts/03_run_eda.py --config $(CONFIG)
	$(PY) scripts/04_run_statistics.py --config $(CONFIG)

baselines:  # E3 -> F6, F7
	$(PY) scripts/05_train_baselines.py --config $(CONFIG)

activations:  # E4 extraction (GPU; skips gracefully without CUDA)
	$(PY) scripts/06_extract_activations.py --config $(CONFIG)

probes:     # E4 -> F8, F14
	$(PY) scripts/07_train_probes.py --config $(CONFIG)

analysis:   # E5, E6, E7 -> F9, F10, F13
	$(PY) scripts/08_lead_time_analysis.py --config $(CONFIG)

intervention:  # E8 -> F11, F12
	$(PY) scripts/09_simulate_intervention.py --config $(CONFIG)

report:
	$(PY) scripts/10_build_report.py --config $(CONFIG)

decomposition:  # E9 error-vs-choke-point decomposition + difficulty controls (CPU) -> decomp_*, F16, F17
	$(PY) scripts/11_choke_point_decomposition.py --config $(CONFIG)
	$(PY) scripts/11b_decomposition_figures.py --config $(CONFIG)

commitment-mock:  # pipeline check on CPU with a fake generator; writes NO results
	$(PY) scripts/12_counterfactual_rollouts.py --generator $(GENERATOR) --backend mock --no-source-gold \
	    --m $(M) --out results/smoke/commitment_mock.jsonl
	$(PY) scripts/13_commitment_analysis.py --rollouts results/smoke/commitment_mock.jsonl \
	    --out results/smoke/commitment_mock_tables

commitment:  # E10 counterfactual rollouts (GPU, vLLM) + analysis; see docs/COMMITMENT_EXPERIMENT.md
	$(PY) scripts/12_counterfactual_rollouts.py --generator $(GENERATOR) --model $(MODEL) \
	    --m $(M) --tp $(TP) --dtype $(DTYPE) --out $(ROLLOUTS) $(EXTRA)
	$(PY) scripts/13_commitment_analysis.py --rollouts $(ROLLOUTS) \
	    --out results/commitment/tables_$(GENERATOR)

# Re-derive every counterfactual number and figure from the committed raw rollouts (CPU, no GPU).
COMMIT_GENS = Qwen2.5-1.5B-Instruct Qwen2-1.5B-Instruct Qwen2.5-Math-7B-Instruct
commitment-analysis:
	for g in $(COMMIT_GENS); do \
	  $(PY) scripts/13_commitment_analysis.py --rollouts results/commitment/$${g}_s*.jsonl \
	      --out results/commitment/tables_$${g} || exit 1; \
	done
	$(PY) scripts/13b_commitment_figures.py --tables $(addprefix results/commitment/tables_,$(COMMIT_GENS)) \
	    --labels Qwen2.5-1.5B Qwen2-1.5B Qwen2.5-Math-7B --out results/figures --delta 0.125

choke-paper:  # paper/main.pdf: regenerate every number from the tables, then compile (pdflatex)
	$(PY) paper/make_values.py
	cd paper && pdflatex -interaction=nonstopmode main.tex > /dev/null && pdflatex -interaction=nonstopmode main.tex > /dev/null

all: data preprocess eda baselines activations probes analysis intervention report

test:
	$(PY) -m pytest tests/ -q

lint:
	ruff check src scripts tests

diagrams:   # TikZ -> PDF (+ PNG preview for the README); needs Tectonic
	$(PY) docs/diagrams/render.py --tectonic $(TECTONIC)

paper:      # copy vector figures + diagrams into the Overleaf package and compile it
	$(PY) docs/diagrams/export_paper_assets.py --paper-dir $(PAPER_DIR) --tectonic $(TECTONIC)

guide:      # study and viva guide: check every number, then compile (needs Tectonic)
	$(PY) docs/diagrams/check_paper_values.py docs/study_guide/study_guide.tex --tables
	cd docs/study_guide && $(TECTONIC) study_guide.tex

# Run R: recovering from hallucinations (docs/RECOVERY_SYSTEM.md, docs/PREREGISTRATION_RECOVERY.md)
recovery-labels:  # RAGTruth human-label analysis, no model (CPU, ~10 min)
	[ -d data/external/RAGTruth ] || git clone --depth 1 https://github.com/ParticleMedia/RAGTruth data/external/RAGTruth
	$(PY) scripts/30_ragtruth_label_analysis.py

recovery-mock:  # the whole Run R sequence on CPU with fake models; writes NO results (into /tmp)
	rm -rf /tmp/recovery_mock && mkdir -p /tmp/recovery_mock && tar --exclude=.git --exclude=results/recovery -cf - . | tar -xf - -C /tmp/recovery_mock
	cd /tmp/recovery_mock && MOCK=1 ARCHIVE=/tmp/recovery_mock/out.tar.gz bash scripts/kaggle_recovery.sh
	cd /tmp/recovery_mock && $(PY) scripts/34_replay.py --items data/recovery/items.*.jsonl --out /tmp/recovery_mock/final --allow-mock --rounds 3

recovery-analysis:  # after extracting recovery_results.tar.gz here: replay, tests, mechanism
	$(PY) scripts/34_replay.py --items data/recovery/items.*.jsonl --out results/recovery/final
	$(PY) scripts/35_mechanism.py --items data/recovery/items.ragtruth.llama-2-7b-chat.jsonl --out results/recovery/final/mechanism

# CPU analyses added after Run R, from committed outputs only (no GPU, no dataset text)
chokebench:  # ChokeBench: step localisers vs human first error and counterfactual choke point
	$(PY) scripts/49_chokebench.py
	$(PY) scripts/54_chokebench_validity.py

evaluator-robustness:  # Part 3 under five evaluators; which conclusions survive (exploratory)
	$(PY) scripts/50_evaluator_robustness.py

story:  # every CPU analysis above, the figures, the paper numbers, the paper PDF and the story PDF
	$(PY) scripts/49_chokebench.py > /dev/null
	$(PY) scripts/50_evaluator_robustness.py > /dev/null
	$(PY) scripts/52_mitigation_law.py > /dev/null
	$(PY) scripts/53_law_tests.py > /dev/null
	$(PY) scripts/54_chokebench_validity.py > /dev/null
	$(PY) scripts/51_story_figure.py
	$(MAKE) choke-paper
	cd docs/story && pdflatex -interaction=nonstopmode story.tex > /dev/null && pdflatex -interaction=nonstopmode story.tex > /dev/null
	cd docs/story && pdflatex -interaction=nonstopmode story_hinglish.tex > /dev/null && pdflatex -interaction=nonstopmode story_hinglish.tex > /dev/null
