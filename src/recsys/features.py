"""Feature engineering for the stage 2 ranker. Every feature uses only data before the target week."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .candidates import build_candidates
from .data import actual_purchases, history, week_dates

ARTICLE_CAT_FEATURES = [
    "product_group_name",
    "index_group_name",
    "garment_group_name",
    "perceived_colour_master_name",
    "graphical_appearance_name",
]
SHARE_ATTRS = {
    "ci_ptype_share": "product_type_name",
    "ci_igroup_share": "index_group_name",
    "ci_garment_share": "garment_group_name",
    "ci_colour_share": "perceived_colour_master_name",
}
SENSITIVE_FEATURES = ["c_age", "c_age_missing", "ci_age_gap", "a_buyer_age_mean", "bandpop_rank"]


def article_features(hist: pd.DataFrame, customers: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    w0 = hist["week"].min()
    h = hist.merge(customers[["cid", "age_filled"]], on="cid", how="left")
    g = h.groupby("article_id")
    f = pd.DataFrame(
        {
            "a_pop_8w": g.size(),
            "a_price_mean": g["price"].mean(),
            "a_online_share": g["online"].mean(),
            "a_buyer_age_mean": g["age_filled"].mean(),
            "a_days_since_first": (ref_date - g["t_dat"].min()).dt.days,
            "a_days_since_last": (ref_date - g["t_dat"].max()).dt.days,
        }
    )
    for n in (1, 2, 4):
        f[f"a_pop_{n}w"] = h[h["week"] < w0 + n].groupby("article_id").size()
    f["a_buyers_4w"] = h[h["week"] < w0 + 4].groupby("article_id")["cid"].nunique()
    f = f.fillna({"a_pop_1w": 0, "a_pop_2w": 0, "a_pop_4w": 0, "a_buyers_4w": 0})
    f["a_trend"] = f["a_pop_1w"] / (f["a_pop_4w"] / 4 + 1)
    return f.astype("float32").reset_index()


def customer_features(hist: pd.DataFrame, customers: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    w0 = hist["week"].min()
    g = hist.groupby("cid")
    f = pd.DataFrame(
        {
            "c_n_tx": g.size(),
            "c_n_articles": g["article_id"].nunique(),
            "c_n_weeks": g["week"].nunique(),
            "c_days_since_last": (ref_date - g["t_dat"].max()).dt.days,
            "c_price_mean": g["price"].mean(),
            "c_online_share": g["online"].mean(),
        }
    )
    f["c_n_tx_2w"] = hist[hist["week"] < w0 + 2].groupby("cid").size()
    f = f.reindex(customers["cid"])
    f.index.name = "cid"
    f = f.reset_index()
    f["c_n_tx"] = f["c_n_tx"].fillna(0)
    f["c_n_tx_2w"] = f["c_n_tx_2w"].fillna(0)
    prof = customers.set_index("cid")
    f["c_age"] = prof.loc[f["cid"], "age_filled"].to_numpy()
    f["c_age_missing"] = prof.loc[f["cid"], "age_missing"].to_numpy()
    f["c_fn"] = prof.loc[f["cid"], "FN"].to_numpy()
    f["c_active"] = prof.loc[f["cid"], "Active"].to_numpy()
    f["c_club_active"] = (prof.loc[f["cid"], "club_member_status"] == "ACTIVE").to_numpy().astype("int8")
    f["c_news_regular"] = (prof.loc[f["cid"], "fashion_news_frequency"] == "Regularly").to_numpy().astype("int8")
    cols = [c for c in f.columns if c != "cid"]
    f[cols] = f[cols].astype("float32")
    return f


def interaction_features(cand: pd.DataFrame, hist: pd.DataFrame, articles: pd.DataFrame) -> pd.DataFrame:
    attrs = articles[["article_id", "product_code"] + list(SHARE_ATTRS.values())]
    h = hist[["cid", "article_id"]].merge(attrs, on="article_id")
    n_tx = h.groupby("cid").size().rename("_n")
    out = cand.merge(attrs, on="article_id", how="left")
    code_counts = h.groupby(["cid", "product_code"]).size().rename("ci_prodcode_count").reset_index()
    out = out.merge(code_counts, on=["cid", "product_code"], how="left")
    for feat, col in SHARE_ATTRS.items():
        cnt = h.groupby(["cid", col]).size().rename("_c").reset_index()
        out = out.merge(cnt, on=["cid", col], how="left")
        out = out.merge(n_tx, left_on="cid", right_index=True, how="left")
        out[feat] = (out["_c"].fillna(0) / out["_n"]).astype("float32")
        out = out.drop(columns=["_c", "_n"])
    out["ci_prodcode_count"] = out["ci_prodcode_count"].fillna(0).astype("float32")
    return out.drop(columns=["product_code"] + list(SHARE_ATTRS.values()))


def build_week_table(ds, emb: pd.DataFrame, cfg: dict, week: int, seed: int, cids: np.ndarray | None = None) -> tuple[pd.DataFrame, dict]:
    """Candidates + features (+ labels when the week has purchases) for one target week."""
    hist = history(ds.transactions, week, cfg["weeks"]["history_weeks"])
    ref = week_dates(week, ds.end_date)[0] - pd.Timedelta(days=1)
    actual = actual_purchases(ds.transactions, week) if week >= 0 else {}
    if cids is None:
        cids = np.array(sorted(actual))
    cand, extras = build_candidates(hist, cids, ds.customers, ds.articles, emb, cfg, ref, seed)
    cand = interaction_features(cand, hist, ds.articles)
    cand = cand.merge(article_features(hist, ds.customers, ref), on="article_id", how="left")
    cand = cand.merge(customer_features(hist, ds.customers, ref), on="cid", how="left")
    cats = ds.articles[["article_id"] + ARTICLE_CAT_FEATURES]
    cand = cand.merge(cats, on="article_id", how="left")
    cand["ci_price_ratio"] = (cand["a_price_mean"] / cand["c_price_mean"]).astype("float32")
    cand["ci_age_gap"] = (cand["c_age"] - cand["a_buyer_age_mean"]).abs().astype("float32")
    for col in ("rep_count", "sib_pop"):
        cand[col] = cand[col].fillna(0).astype("float32")
    cand["week"] = np.int16(week)
    if actual:
        pos = pd.DataFrame([(c, a) for c, items in actual.items() for a in items], columns=["cid", "article_id"])
        pos["label"] = np.int8(1)
        cand = cand.merge(pos, on=["cid", "article_id"], how="left")
        cand["label"] = cand["label"].fillna(0).astype("int8")
    extras["actual"] = actual
    extras["cids"] = cids
    return cand, extras


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {"cid", "article_id", "week", "label"}
    return [c for c in df.columns if c not in drop]


def to_model_frame(df: pd.DataFrame, cols: list[str], categories: dict | None = None) -> pd.DataFrame:
    x = df[cols].copy()
    for c in ARTICLE_CAT_FEATURES:
        if c in x:
            cats = categories[c] if categories else sorted(x[c].dropna().unique())
            x[c] = pd.Categorical(x[c], categories=cats)
    return x
