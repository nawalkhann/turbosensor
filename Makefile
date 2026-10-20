.PHONY: install data train train-all test lint notebook serve

install:
	pip install -e ".[dev,serve,notebook]"

data:
	turbosensor download --data-dir data/raw

train: data
	turbosensor train --config configs/fd001.yaml

train-all: data
	for s in fd001 fd002 fd003 fd004; do turbosensor train --config configs/$$s.yaml || exit 1; done

test:
	pytest -q

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

notebook:
	python scripts/build_notebook.py
	cd notebooks && jupyter nbconvert --to notebook --execute --inplace 01_exploration_and_results.ipynb

serve:
	turbosensor serve --model artifacts/FD001/lgbm.joblib --host 0.0.0.0
