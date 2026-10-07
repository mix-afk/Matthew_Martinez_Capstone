"""Stage 2 rankers: logistic regression baseline, LightGBM classifier, LightGBM LambdaRank."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import evaluate as E
from .features import ARTICLE_CAT_FEATURES, to_model_frame


def score_map(df: pd.DataFrame, scores: np.ndarray, actual: dict, k: int, fallback: list) -> dict:
    d = df[["cid", "article_id"]].copy()
    d["score"] = scores
    pred = E.top_k_from_scores(d, "score", k)
    return E.summarize(actual, pred, k, fallback=fallback)


def make_logreg(num_cols: list[str], cat_cols: list[str], C: float, seed: int, class_weight=None) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=200), cat_cols),
        ]
    )
    model = LogisticRegression(C=C, max_iter=300, random_state=seed, class_weight=class_weight)
    return Pipeline([("pre", pre), ("model", model)])


def fit_logreg(train, valid, cols, actual_valid, fallback, cfg, seed, sample_weight=None, grid=(0.01, 0.1, 1.0)):
    cat_cols = [c for c in cols if c in ARTICLE_CAT_FEATURES]
    num_cols = [c for c in cols if c not in ARTICLE_CAT_FEATURES]
    results = []
    best = None
    for C in grid:
        t0 = time.time()
        pipe = make_logreg(num_cols, cat_cols, C, seed)
        pipe.fit(train[cols], train["label"], model__sample_weight=sample_weight)
        s = pipe.predict_proba(valid[cols])[:, 1]
        m = score_map(valid, s, actual_valid, cfg["eval"]["k"], fallback)
        m.update({"C": C, "valid_auc": E.auc(valid["label"], s), "fit_seconds": time.time() - t0})
        results.append(m)
        if best is None or m["map@12"] > best[1]["map@12"]:
            best = (pipe, m)
    return best[0], best[1], pd.DataFrame(results)


def sample_params(rng: np.random.Generator) -> dict:
    return {
        "num_leaves": int(rng.choice([15, 31, 63, 127])),
        "learning_rate": float(rng.choice([0.05, 0.1])),
        "min_child_samples": int(rng.choice([50, 100, 200, 400])),
        "colsample_bytree": float(rng.choice([0.5, 0.7, 0.9])),
        "subsample": float(rng.choice([0.6, 0.8, 1.0])),
        "reg_lambda": float(rng.choice([0.0, 1.0, 5.0])),
    }


def fit_lgbm_classifier(train, valid, cols, cats, actual_valid, fallback, cfg, seed, params, sample_weight=None):
    xt, xv = to_model_frame(train, cols, cats), to_model_frame(valid, cols, cats)
    model = lgb.LGBMClassifier(
        n_estimators=cfg["ranker"]["max_rounds"],
        metric="auc",
        subsample_freq=1,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        **params,
    )
    t0 = time.time()
    model.fit(
        xt,
        train["label"],
        sample_weight=sample_weight,
        eval_set=[(xv, valid["label"])],
        callbacks=[lgb.early_stopping(cfg["ranker"]["early_stopping_rounds"], first_metric_only=True, verbose=False)],
    )
    s = model.predict_proba(xv)[:, 1]
    m = score_map(valid, s, actual_valid, cfg["eval"]["k"], fallback)
    m.update({"valid_auc": E.auc(valid["label"], s), "best_iteration": model.best_iteration_, "fit_seconds": time.time() - t0})
    return model, m


def fit_lgbm_ranker(train, valid, cols, cats, actual_valid, fallback, cfg, seed, params):
    train = train.sort_values(["week", "cid"])
    valid = valid.sort_values(["week", "cid"])
    stop = valid[valid.groupby("cid")["label"].transform("max") > 0]
    xt, xv, xs = to_model_frame(train, cols, cats), to_model_frame(valid, cols, cats), to_model_frame(stop, cols, cats)
    gt = train.groupby(["week", "cid"], sort=False).size().to_numpy()
    gs = stop.groupby(["week", "cid"], sort=False).size().to_numpy()
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=cfg["ranker"]["max_rounds"],
        subsample_freq=1,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        **params,
    )
    t0 = time.time()
    model.fit(
        xt,
        train["label"],
        group=gt,
        eval_set=[(xs, stop["label"])],
        eval_group=[gs],
        eval_at=[cfg["eval"]["k"]],
        callbacks=[lgb.early_stopping(cfg["ranker"]["early_stopping_rounds"], verbose=False)],
    )
    s = model.predict(xv)
    m = score_map(valid, s, actual_valid, cfg["eval"]["k"], fallback)
    m.update({"valid_auc": E.auc(valid["label"], s), "best_iteration": model.best_iteration_, "fit_seconds": time.time() - t0})
    return model, m, valid


def tune(fit_fn, n_trials: int, tune_seed: int, **kwargs):
    rng = np.random.default_rng(tune_seed)
    rows, best = [], None
    for i in range(n_trials):
        params = sample_params(rng)
        out = fit_fn(params=params, **kwargs)
        model, m = out[0], out[1]
        rows.append({"trial": i, **params, **m})
        print(f"  trial {i}: MAP@12 {m['map@12']:.4f}, rounds {m.get('best_iteration')}, {m.get('fit_seconds', 0):.0f}s", flush=True)
        if best is None or m["map@12"] > best[2]["map@12"]:
            best = (model, params, m)
    return best, pd.DataFrame(rows)


def save_models(models_dir: Path, models: dict, meta: dict) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        if isinstance(model, (lgb.LGBMClassifier, lgb.LGBMRanker)):
            model.booster_.save_model(str(models_dir / f"{name}.txt"))
            joblib.dump(model, models_dir / f"{name}.joblib", compress=3)
        else:
            joblib.dump(model, models_dir / f"{name}.joblib", compress=3)
    with open(models_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


def predict(model, df: pd.DataFrame, cols: list[str], cats: dict) -> np.ndarray:
    if isinstance(model, Pipeline):
        return model.predict_proba(df[cols])[:, 1]
    x = to_model_frame(df, cols, cats)
    if isinstance(model, lgb.LGBMClassifier):
        return model.predict_proba(x)[:, 1]
    return model.predict(x)
