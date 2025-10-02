.PHONY: setup demo baseline gbdt clean spark-prep spark-train

PYTHON ?= python3
VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate

setup:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && pip install --upgrade pip
	$(ACTIVATE) && pip install -r requirements.txt
	$(ACTIVATE) && python -m nltk.downloader punkt vader_lexicon

_demo_pipeline = $(ACTIVATE) && python -m src.cli

demo:
	$(ACTIVATE) && python -m src.cli prepare
	$(ACTIVATE) && python -m src.cli train-embed
	$(ACTIVATE) && python -m src.cli evaluate
	$(ACTIVATE) && python -m src.cli report

baseline:
	$(ACTIVATE) && python -m src.cli prepare
	$(ACTIVATE) && python -m src.cli train-baseline
	$(ACTIVATE) && python -m src.cli evaluate
	$(ACTIVATE) && python -m src.cli report

gbdt:
	$(ACTIVATE) && python -m src.cli prepare
	$(ACTIVATE) && python -m src.cli train-gbdt
	$(ACTIVATE) && python -m src.cli evaluate
	$(ACTIVATE) && python -m src.cli report

spark-prep:
	$(ACTIVATE) && python -m src.spark.preprocess_spark $(INPUT) $(OUTPUT)

spark-train:
	$(ACTIVATE) && python -m src.spark.train_spark $(INPUT) $(MODEL_OUT) --metrics-output $(METRICS)

clean:
	rm -rf reports/figures/* reports/html/* artifacts mlruns data/processed data/cache
