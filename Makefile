PY ?= python
export PYTHONPATH := src

.PHONY: install sample tables train serve api test lint notebooks all

install:
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

sample:            ## cut the raw Kaggle files in data/raw down to the modeling sample
	$(PY) -c "from recsys.config import load_config; from recsys.data import make_sample; c=load_config(); print(make_sample(c['paths']['raw'], c['paths']['processed'], c))"

tables:            ## candidates + features for every week
	$(PY) -m recsys.pipeline --stage tables

train:             ## tune, train, evaluate, log to MLflow, save models/
	$(PY) -m recsys.pipeline --stage train

serve:             ## batch-score customers and write API artifacts
	$(PY) -m recsys.serving

api:
	uvicorn recsys.api:app --host 0.0.0.0 --port 8000

test:
	$(PY) -m pytest -q

lint:
	ruff check src tests

notebooks:
	cd notebooks && for nb in 0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace $$nb; done

all: sample tables train serve
