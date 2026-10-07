"""Bias and fairness audit for the recommender: customer side (age bands) and item side (popularity)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluate as E


def top_k_flags(df: pd.DataFrame, score_col: str, k: int) -> pd.Series:
    rank = df.groupby("cid")[score_col].rank(method="first", ascending=False)
    return (rank <= k).astype("int8")


def group_quality(
    per_customer: pd.DataFrame, customers: pd.DataFrame, group_col: str = "age_band", di_threshold: float = 0.8, min_size: int = 100
) -> pd.DataFrame:
    """Quality per group. Groups smaller than min_size are reported but not flagged (too few customers to judge)."""
    d = per_customer.merge(customers[["cid", group_col]], on="cid")
    g = d.groupby(group_col, observed=True).agg(
        customers=("cid", "size"), map12=("ap", "mean"), recall12=("recall", "mean"), ndcg12=("ndcg", "mean"), hit_rate=("hit", "mean")
    )
    g["share_of_customers"] = g["customers"] / g["customers"].sum()
    g["small_group"] = g["customers"] < min_size
    g["disparate_impact"] = g["hit_rate"] / g.loc[~g["small_group"], "hit_rate"].max()
    g["di_flag"] = (g["disparate_impact"] < di_threshold) & ~g["small_group"]
    return g.reset_index()


def gap_summary(g: pd.DataFrame) -> dict:
    big = g[~g["small_group"]]
    return {
        "map@12": float((g["map12"] * g["customers"]).sum() / g["customers"].sum()),
        "min_disparate_impact": float(big["disparate_impact"].min()),
        "map_gap_best_worst": float(big["map12"].max() - big["map12"].min()),
        "lowest_band": str(big.loc[big["map12"].idxmin(), g.columns[0]]),
    }


def equalized_odds(df: pd.DataFrame, customers: pd.DataFrame, pred_col: str, group_col: str = "age_band") -> tuple[pd.DataFrame, dict]:
    """Treat 'item made the customer's top-k' as the positive prediction over candidate rows."""
    d = df[["cid", "label", pred_col]].merge(customers[["cid", group_col]], on="cid")
    rows = []
    for grp, g in d.groupby(group_col, observed=True):
        pos, neg = g[g["label"] == 1], g[g["label"] == 0]
        rows.append(
            {
                group_col: grp,
                "selection_rate": g[pred_col].mean(),
                "tpr": pos[pred_col].mean() if len(pos) else np.nan,
                "fpr": neg[pred_col].mean() if len(neg) else np.nan,
                "positives": len(pos),
            }
        )
    t = pd.DataFrame(rows)
    summary = {
        "demographic_parity_difference": float(t["selection_rate"].max() - t["selection_rate"].min()),
        "selection_rate_ratio": float(t["selection_rate"].min() / t["selection_rate"].max()),
        "tpr_gap": float(t["tpr"].max() - t["tpr"].min()),
        "fpr_gap": float(t["fpr"].max() - t["fpr"].min()),
    }
    summary["equalized_odds_difference"] = max(summary["tpr_gap"], summary["fpr_gap"])
    return t, summary


def popularity_tiers(tx_hist: pd.DataFrame, quantile: float = 0.8) -> pd.Series:
    """Head = items in the top (1 - quantile) of sales in the history window; everything else is long tail."""
    counts = tx_hist["article_id"].value_counts()
    cut = counts.quantile(quantile)
    return pd.Series(np.where(counts >= cut, "head", "long_tail"), index=counts.index)


def exposure_report(pred: dict, actual: dict, tiers: pd.Series, n_catalog: int) -> dict:
    slots = pd.Series([a for items in pred.values() for a in items])
    bought = pd.Series([a for items in actual.values() for a in items])
    tier_slots = slots.map(tiers).fillna("long_tail")
    tier_bought = bought.map(tiers).fillna("long_tail")
    counts = slots.value_counts().to_numpy()
    counts = np.sort(counts)
    n = len(counts)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(counts) / (n * counts.sum())) if n else 0.0
    return {
        "head_share_of_slots": float((tier_slots == "head").mean()),
        "long_tail_share_of_slots": float((tier_slots == "long_tail").mean()),
        "long_tail_share_of_purchases": float((tier_bought == "long_tail").mean()),
        "distinct_items_recommended": int(slots.nunique()),
        "coverage": float(slots.nunique() / n_catalog),
        "exposure_gini": gini,
    }


def inverse_frequency_weights(df: pd.DataFrame, customers: pd.DataFrame, group_col: str = "age_band") -> np.ndarray:
    grp = df[["cid"]].merge(customers[["cid", group_col]], on="cid", how="left")[group_col].astype(str)
    cust_groups = df[["week", "cid"]].drop_duplicates().merge(customers[["cid", group_col]], on="cid")[group_col].astype(str)
    freq = cust_groups.value_counts(normalize=True)
    w = 1.0 / grp.map(freq).to_numpy()
    return w / w.mean()


def rerank_long_tail(df: pd.DataFrame, score_col: str, pop_col: str, lam: float) -> pd.Series:
    """Blend the model score with a popularity penalty, both as within-customer percentiles."""
    s = df.groupby("cid")[score_col].rank(pct=True)
    p = df.groupby("cid")[pop_col].rank(pct=True)
    return s - lam * p


def tradeoff_curve(df: pd.DataFrame, score_col: str, pop_col: str, lambdas, actual, tiers, k, fallback, n_catalog) -> pd.DataFrame:
    rows = []
    for lam in lambdas:
        d = df[["cid", "article_id"]].copy()
        d["s"] = rerank_long_tail(df, score_col, pop_col, lam).to_numpy()
        pred = E.top_k_from_scores(d, "s", k)
        m = E.summarize(actual, pred, k, fallback=fallback)
        ex = exposure_report(pred, actual, tiers, n_catalog)
        rows.append({"lambda": lam, **m, **ex})
    return pd.DataFrame(rows)
