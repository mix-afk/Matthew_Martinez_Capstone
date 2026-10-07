# Deployment and MLOps guide

This covers Step 8: local deployment of the best model, reproducible environments, tracking, CI, monitoring, versioning and rollback.

## 1. How the model is served

The recommender runs as a **nightly batch + online lookup** service, which is how most retail recommenders run in production:

1. A batch job (`make serve`, which runs `python -m recsys.serving`) builds candidates for every active customer, scores them with the LightGBM ranker and writes the top 12 per customer, plus product details, recent history and "bought together" neighbours, to `data/processed/serving/`.
2. A FastAPI app (`src/recsys/api.py`) loads those artifacts at start-up and answers requests in milliseconds. No model runs inside the request path, so latency stays flat.
3. Unknown or new customers get a cold-start list: last week's best sellers, or best sellers in their age band when the caller passes one.

| Endpoint | What it returns |
|---|---|
| `GET /` | Demo storefront page ("You may also like") |
| `GET /health` | Status, model name, data cut-off date, number of customers scored |
| `GET /recommendations/{customer_id}?k=12&explain=template\|llm` | Top-k items with a one-line reason each |
| `GET /similar/{article_id}` | Items often bought together with this one |
| `GET /customers/demo` | Sample customer ids for the demo |
| `GET /docs` | Interactive OpenAPI docs |

In this project the batch covers the 14,023 customers who bought in the test week, scored with data up to 15 Sep 2020. That lets the demo show which recommended items the customer really bought the following week.

## 2. Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# put articles.parquet, customers.parquet, transactions_train.parquet in data/raw/
make sample      # cut to the 13-week, ~20% customer sample
make tables      # candidates + features per week
make train       # tune, train, evaluate, log to MLflow, save to models/
make serve       # batch-score customers, write API artifacts
make api         # http://localhost:8000
```

### Docker

```bash
make serve                                  # artifacts must exist first
docker build -t hm-recommender:1.0.0 .
docker run -p 8000:8000 --env-file .env hm-recommender:1.0.0
```

The image only contains the API runtime (`requirements-api.txt`) and the small serving artifacts, not the training stack.

## 3. Reproducibility

| Practice | Where |
|---|---|
| Pinned dependencies | `requirements.txt` (full), `requirements-api.txt` (serving) |
| One config file for every run | `configs/config.yaml` (dates, sample rule, weeks, candidate sizes, tuning budget, fairness settings) |
| Fixed seeds | `seed: 42` in config, passed to sampling, ALS, PCA, LightGBM and t-SNE |
| Deterministic sample | customers kept by the last two hex characters of their hashed id, no random draw |
| Saved models and settings | `models/*.joblib`, `models/*.txt` (LightGBM boosters), `models/model_meta.json` (features, categories, tuned params, validation metrics) |
| Experiment tracking | MLflow, `mlruns/mlflow.db` (`mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`) |

## 4. Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

1. `ruff check src tests` (lint)
2. `pytest` (18 unit tests: metrics, candidate generators, fairness maths, GenAI guardrails and fallbacks, API endpoints)

The tests use small synthetic data, so CI does not need the H&M files.

## 5. Monitoring plan

| What | How | Alert when |
|---|---|---|
| Service health | `/health` probe, Docker `HEALTHCHECK` | any failed probe |
| Latency and errors | uvicorn access logs to the log stack | p95 above 200 ms or 5xx above 1% |
| Data freshness | `as_of` in `/health` | artifacts older than 8 days |
| Input drift | weekly compare of feature distributions (PSI) vs training weeks | PSI above 0.2 on a top-10 SHAP feature |
| Online quality | click-through and add-to-cart on recommended items, logged per slot | 2 weeks below the previous model |
| Offline quality | each week, score last week's lists against what customers bought (MAP@12, hit rate) | MAP@12 drops more than 15% vs model card |
| Fairness | the same week-on-week check split by age band, plus long-tail share of slots | disparate impact of hit rate below 0.8 for any band |
| Cost (GenAI) | count of LLM calls and fallbacks | fallback rate above 10% |

## 6. Versioning and rollback

- **Code**: git tags per release (`v1.0.0`), CI must pass before merge to `main`.
- **Models**: each training run is an MLflow run; the deployed version is the `models/` folder of a tagged commit plus its `model_meta.json`.
- **Artifacts**: the batch job writes to a dated folder and `manifest.json` records model name and data cut-off. The API reads `RECSYS_SERVING_DIR`, so switching versions is a config change.
- **Rollback**: point `RECSYS_SERVING_DIR` at the previous folder (or redeploy the previous image tag) and restart. No retraining needed.
- **Safe rollout**: shadow-score the new model for one week, then A/B test 10% of traffic before full rollout.

## 7. Cloud (optional)

The same image runs on any container service (AWS App Runner, GCP Cloud Run, Azure Container Apps). The training pipeline is a scheduled job that writes artifacts to object storage; the API pulls the latest folder on start. This was not deployed for the project.
