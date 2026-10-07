"""Analysis helpers used by the EDA, modeling and audit notebooks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluate as E
from .data import history

FEATURE_DOCS = {
    "rep_count": ("Customer x item", "Times the customer bought this exact item in the last 8 weeks"),
    "rep_days_since": ("Customer x item", "Days since the customer last bought this item"),
    "rep_rank": ("Customer x item", "Recency rank of the item in the customer's history"),
    "itemcf_score": ("Candidate score", "Item-to-item cosine score from co-purchases"),
    "itemcf_rank": ("Candidate score", "Rank inside the item-CF list"),
    "als_score": ("Candidate score", "ALS matrix factorization score"),
    "als_rank": ("Candidate score", "Rank inside the ALS list"),
    "content_score": ("Candidate score", "Cosine similarity of item to the customer's PCA taste profile"),
    "content_rank": ("Candidate score", "Rank inside the content-based list"),
    "sib_pop": ("Candidate score", "Recent sales of another colour of a product the customer bought"),
    "sib_rank": ("Candidate score", "Rank inside the sibling list"),
    "pop_rank": ("Candidate score", "Rank in last week's best sellers"),
    "bandpop_rank": ("Candidate score", "Rank in the best sellers of the customer's age band (last 2 weeks)"),
    "n_sources": ("Candidate score", "How many generators proposed the item"),
    "ci_prodcode_count": ("Customer x item", "Purchases of the same parent product (any colour)"),
    "ci_ptype_share": ("Customer x item", "Share of the customer's purchases in this product type"),
    "ci_igroup_share": ("Customer x item", "Share of the customer's purchases in this index group"),
    "ci_garment_share": ("Customer x item", "Share of the customer's purchases in this garment group"),
    "ci_colour_share": ("Customer x item", "Share of the customer's purchases in this colour family"),
    "ci_price_ratio": ("Customer x item", "Item average price divided by the customer's average price"),
    "ci_age_gap": ("Customer x item", "Gap between customer age and the item's average buyer age"),
    "a_pop_1w": ("Item", "Units sold last week"),
    "a_pop_2w": ("Item", "Units sold in the last 2 weeks"),
    "a_pop_4w": ("Item", "Units sold in the last 4 weeks"),
    "a_pop_8w": ("Item", "Units sold in the last 8 weeks"),
    "a_trend": ("Item", "Last week's sales vs the 4-week weekly average"),
    "a_buyers_4w": ("Item", "Distinct buyers in the last 4 weeks"),
    "a_price_mean": ("Item", "Average price paid"),
    "a_online_share": ("Item", "Share of units sold online"),
    "a_buyer_age_mean": ("Item", "Average age of the item's buyers"),
    "a_days_since_first": ("Item", "Days since first sale in the window (newness)"),
    "a_days_since_last": ("Item", "Days since last sale"),
    "c_n_tx": ("Customer", "Units bought in the last 8 weeks"),
    "c_n_tx_2w": ("Customer", "Units bought in the last 2 weeks"),
    "c_n_articles": ("Customer", "Distinct items bought"),
    "c_n_weeks": ("Customer", "Active weeks out of 8"),
    "c_days_since_last": ("Customer", "Days since last purchase"),
    "c_price_mean": ("Customer", "Average price paid"),
    "c_online_share": ("Customer", "Share of units bought online"),
    "c_age": ("Customer", "Age (median-imputed)"),
    "c_age_missing": ("Customer", "Age was missing"),
    "c_fn": ("Customer", "Gets fashion news"),
    "c_active": ("Customer", "Active for communication"),
    "c_club_active": ("Customer", "Active club member"),
    "c_news_regular": ("Customer", "Gets the newsletter regularly"),
    "product_group_name": ("Item (categorical)", "Product group"),
    "index_group_name": ("Item (categorical)", "Index group"),
    "garment_group_name": ("Item (categorical)", "Garment group"),
    "perceived_colour_master_name": ("Item (categorical)", "Colour family"),
    "graphical_appearance_name": ("Item (categorical)", "Print or pattern"),
}


def feature_table() -> pd.DataFrame:
    return pd.DataFrame([(k, *v) for k, v in FEATURE_DOCS.items()], columns=["feature", "group", "description"])


def predictability(tx: pd.DataFrame, articles: pd.DataFrame, weeks, hist_weeks: int, popular_n: int) -> pd.DataFrame:
    code = articles.set_index("article_id")["product_code"]
    rows = []
    for w in weeks:
        hist = history(tx, w, hist_weeks)
        now = tx[tx["week"] == w][["cid", "article_id"]].drop_duplicates()
        seen = set(zip(hist["cid"], hist["article_id"]))
        seen_code = set(zip(hist["cid"], hist["article_id"].map(code)))
        pop = set(hist[hist["week"] == w + 1]["article_id"].value_counts().head(popular_n).index)
        rows.append(
            {
                "week": w,
                "pairs": len(now),
                "bought_before": np.mean([p in seen for p in zip(now["cid"], now["article_id"])]),
                "same_product_bought_before": np.mean([p in seen_code for p in zip(now["cid"], now["article_id"].map(code))]),
                "in_last_week_top": now["article_id"].isin(pop).mean(),
                "buyers_with_history": now["cid"].drop_duplicates().isin(hist["cid"]).mean(),
            }
        )
    return pd.DataFrame(rows)


def age_mix(tx: pd.DataFrame, customers: pd.DataFrame, articles: pd.DataFrame, col: str = "index_group_name") -> pd.DataFrame:
    d = tx[["cid", "article_id"]].merge(customers[["cid", "age_band"]], on="cid").merge(articles[["article_id", col]], on="article_id")
    t = pd.crosstab(d["age_band"], d[col], normalize="index")
    return t


def table_summary(cfg: dict, load_week) -> pd.DataFrame:
    rows = []
    for w in [cfg["weeks"]["test"], cfg["weeks"]["valid"]] + list(cfg["weeks"]["train"]):
        t, x = load_week(cfg, w)
        role = "test" if w == cfg["weeks"]["test"] else "valid" if w == cfg["weeks"]["valid"] else "train"
        rows.append(
            {
                "week": w,
                "role": role,
                "rows": len(t),
                "customers": t["cid"].nunique(),
                "candidates_per_customer": len(t) / t["cid"].nunique(),
                "positives": int(t["label"].sum()),
                "positive_rate": t["label"].mean(),
                "candidate_recall": E.candidate_recall(t, x["actual"]),
                "customers_with_a_positive": t.groupby("cid")["label"].max().mean(),
            }
        )
    return pd.DataFrame(rows)


def recall_by_source(table: pd.DataFrame, actual: dict, sources: list[str]) -> pd.DataFrame:
    rows = []
    for col in sources:
        sub = table[table[col].notna()]
        rows.append({"source": col.replace("_rank", ""), "rows": len(sub), "candidate_recall": E.candidate_recall(sub, actual)})
    rows.append({"source": "union", "rows": len(table), "candidate_recall": E.candidate_recall(table, actual)})
    return pd.DataFrame(rows)


def check_leakage(ds, cfg: dict) -> pd.DataFrame:
    from .data import week_dates

    rows = []
    for w in [cfg["weeks"]["test"], cfg["weeks"]["valid"]] + list(cfg["weeks"]["train"]):
        hist = history(ds.transactions, w, cfg["weeks"]["history_weeks"])
        start, end = week_dates(w, ds.end_date)
        rows.append(
            {
                "target_week": w,
                "target_start": start.date(),
                "history_last_day": hist["t_dat"].max().date(),
                "history_first_day": hist["t_dat"].min().date(),
                "no_overlap": bool(hist["t_dat"].max() < start),
            }
        )
    return pd.DataFrame(rows)


def warm_cold(per_customer: pd.DataFrame, warm_ids: set) -> pd.DataFrame:
    d = per_customer.assign(segment=np.where(per_customer["cid"].isin(warm_ids), "has 8-week history", "no history (cold)"))
    return d.groupby("segment").agg(customers=("cid", "size"), map12=("ap", "mean"), hit_rate=("hit", "mean")).reset_index()
