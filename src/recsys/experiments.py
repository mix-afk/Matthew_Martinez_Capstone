"""Feature selection experiments and fairness mitigation retrains."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import evaluate as E
from . import ranker as R
from .features import ARTICLE_CAT_FEATURES, to_model_frame


def best_params(cfg: dict, name: str = "lgbm_classifier") -> dict:
    meta = json.loads((Path(cfg["paths"]["models"]) / "model_meta.json").read_text())
    return meta["params"][name], meta["valid_metrics"][name]


def numeric_sample(train: pd.DataFrame, cols: list[str], n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    s = train.sample(min(n, len(train)), random_state=seed)
    num = [c for c in cols if c not in ARTICLE_CAT_FEATURES]
    return s[num], s["label"]


def variance_filter(x: pd.DataFrame, threshold: float = 1e-4) -> list[str]:
    vt = VarianceThreshold(threshold)
    vt.fit(SimpleImputer(strategy="median").fit_transform(x))
    return [c for c, keep in zip(x.columns, vt.get_support()) if not keep]


def mutual_information(x: pd.DataFrame, y: pd.Series, seed: int) -> pd.DataFrame:
    xi = x.fillna(-1)
    discrete = [xi[c].nunique() <= 10 for c in xi.columns]
    mi = mutual_info_classif(xi, y, discrete_features=discrete, random_state=seed, n_neighbors=3)
    return pd.DataFrame({"feature": x.columns, "mutual_info": mi}).sort_values("mutual_info", ascending=False).reset_index(drop=True)


def l1_selection(x: pd.DataFrame, y: pd.Series, C: float, seed: int) -> pd.DataFrame:
    xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(x))
    lr = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=300, random_state=seed)
    lr.fit(xs, y)
    return pd.DataFrame({"feature": x.columns, "coef": lr.coef_[0]}).assign(abs_coef=lambda d: d["coef"].abs()).sort_values("abs_coef", ascending=False).reset_index(drop=True)


def fixed_lgbm(train, cols, cats, params, n_rounds, seed, weight=None):
    model = lgb.LGBMClassifier(n_estimators=n_rounds, subsample_freq=1, random_state=seed, verbose=-1, **params)
    model.fit(to_model_frame(train, cols, cats), train["label"], sample_weight=weight)
    return model


def evaluate_sets(train, valid, vx, sets: dict, cats, params, n_rounds, cfg) -> pd.DataFrame:
    rows = []
    for name, cols in sets.items():
        m = fixed_lgbm(train, cols, cats, params, n_rounds, cfg["seed"])
        s = m.predict_proba(to_model_frame(valid, cols, cats))[:, 1]
        met = R.score_map(valid, s, vx["actual"], cfg["eval"]["k"], vx["popular"])
        rows.append({"feature_set": name, "n_features": len(cols), "valid_map@12": met["map@12"], "valid_auc": E.auc(valid["label"], s)})
    return pd.DataFrame(rows)


def scores_to_pred(df: pd.DataFrame, scores: np.ndarray, k: int) -> dict:
    d = df[["cid", "article_id"]].copy()
    d["score"] = scores
    return E.top_k_from_scores(d, "score", k)
