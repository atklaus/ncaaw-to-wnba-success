.PHONY: setup build-data build-model train

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -e ".[dev,viz]"

build-data:
	python scripts/build_datasets.py

build-model:
	python scripts/build_model_frame.py

train:
	python scripts/train_model.py
