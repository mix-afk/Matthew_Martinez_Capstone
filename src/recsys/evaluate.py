"""Ranking metrics. MAP@12 follows the H&M Kaggle competition definition."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def average_precision_at_k(actual: set, predicted: list, k: int = 12) -> float:
    if not actual:
        return 0.0
    predicted = list(dict.fromkeys(predicted))[:k]
    hits, score = 0, 0.0
    for i, p in enumerate(predicted):
        if p in actual:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(actual), k)


def recall_at_k(actual: set, predicted: list, k: int = 12) -> float:
    if not actual:
        return 0.0
    return len(set(predicted[:k]) & actual) / len(actual)


def ndcg_at_k(actual: set, predicted: list, k: int = 12) -> float:
    if not actual:
        return 0.0
    dcg = sum(1.0 / np.log2(i + 2) for i, p in enumerate(predicted[:k]) if p in actual)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
    return dcg / idcg


def per_customer_metrics(actual: dict, predicted: dict, k: int = 12, fallback: list | None = None) -> pd.DataFrame:
    rows = []
    for cid, items in actual.items():
        pred = list(predicted.get(cid, []))
        if fallback is not None and len(pred) < k:
            pred = pred + [x for x in fallback if x not in set(pred)]
        pred = pred[:k]
        rows.append(
            {
                "cid": cid,
                "ap": average_precision_at_k(items, pred, k),
                "recall": recall_at_k(items, pred, k),
                "ndcg": ndcg_at_k(items, pred, k),
                "hit": float(len(set(pred) & items) > 0),
                "n_actual": len(items),
            }
        )
    return pd.DataFrame(rows)


def summarize(actual: dict, predicted: dict, k: int = 12, fallback: list | None = None, n_catalog: int | None = None) -> dict:
    m = per_customer_metrics(actual, predicted, k, fallback)
    out = {
        f"map@{k}": m["ap"].mean(),
        f"recall@{k}": m["recall"].mean(),
        f"ndcg@{k}": m["ndcg"].mean(),
        "hit_rate": m["hit"].mean(),
        "customers": len(m),
    }
    if n_catalog:
        recs = set()
        for cid in actual:
            pred = list(predicted.get(cid, []))
            if fallback is not None and len(pred) < k:
                pred = pred + [x for x in fallback if x not in set(pred)]
            recs.update(pred[:k])
        out["coverage"] = len(recs) / n_catalog
    return out


def candidate_recall(candidates: pd.DataFrame, actual: dict) -> float:
    cand = candidates.groupby("cid")["article_id"].agg(set).to_dict()
    found = sum(len(items & cand.get(cid, set())) for cid, items in actual.items())
    total = sum(len(items) for items in actual.values())
    return found / total if total else 0.0


def top_k_from_scores(df: pd.DataFrame, score_col: str, k: int = 12) -> dict:
    top = df.sort_values(["cid", score_col], ascending=[True, False]).groupby("cid").head(k)
    return top.groupby("cid")["article_id"].agg(list).to_dict()


def auc(y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score))
