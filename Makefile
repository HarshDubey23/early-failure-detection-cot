# One-command entry points (brief section 12). Each stage prints a summary
# and stops the chain on failure (set -e semantics via `&&` chaining through
# separate targets is avoided here by explicit depends + `set -e` in sh).
# Windows note: run with `make` (conda `make`, git-bash, or WSL). Scripts are
# executed by file path (not `python -m`) because numeric prefixes are not
# valid Python module names.

PY = py
CONFIG = configs/default.yaml
QUICK = configs/quick.yaml

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

eda:
	$(PY) scripts/03_run_eda.py --config $(CONFIG)

baselines:
	$(PY) scripts/05_train_baselines.py --config $(CONFIG)

activations:
	$(PY) scripts/06_extract_activations.py --config $(CONFIG)

probes:
	$(PY) scripts/07_train_probes.py --config $(CONFIG)

analysis:
	$(PY) scripts/04_run_statistics.py --config $(CONFIG)
	$(PY) scripts/08_lead_time_analysis.py --config $(CONFIG)

intervention:
	$(PY) scripts/09_simulate_intervention.py --config $(CONFIG)

report:
	$(PY) scripts/10_build_report.py --config $(CONFIG)

all: data preprocess eda baselines activations probes analysis intervention report

test:
	$(PY) -m pytest tests/ -q

lint:
	ruff check src scripts tests
