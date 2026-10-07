"""Batch-score customers and write the small artifacts the API serves."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import ranker as R
from .data import history, load_dataset
from .genai import reason_facts, template_reason
from .pipeline import load_week

ARTICLE_COLS = [
    "article_id", "prod_name", "product_type_name", "product_group_name", "colour_group_name",
    "perceived_colour_master_name", "index_group_name", "garment_group_name", "detail_desc",
]


def build_serving(cfg: dict, model_name: str = "lgbm_classifier", k: int = 12) -> Path:
    out = Path(cfg["paths"]["serving"])
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((Path(cfg["paths"]["models"]) / "model_meta.json").read_text())
    model = joblib.load(Path(cfg["paths"]["models"]) / f"{model_name}.joblib")
    ds = load_dataset(cfg)
    test, tx = load_week(cfg, cfg["weeks"]["test"])
    test["score"] = R.predict(model, test, meta["features"], meta["categories"])
    top = test.sort_values(["cid", "score"], ascending=[True, False]).groupby("cid").head(k).copy()
    top["rank"] = top.groupby("cid").cumcount() + 1

    arts = ds.articles.set_index("article_id")
    hist = history(ds.transactions, cfg["weeks"]["test"], cfg["weeks"]["history_weeks"])
    last = hist.sort_values("t_dat", ascending=False).drop_duplicates(["cid", "article_id"])
    first_name = last.groupby("cid")["article_id"].first().map(arts["prod_name"])
    reasons = []
    for row in top.itertuples(index=False):
        r = pd.Series(row._asdict())
        facts = reason_facts(r, arts.loc[row.article_id], [first_name.get(row.cid)] if row.cid in first_name else None)
        reasons.append((template_reason(facts), "; ".join(facts)))
    top["reason"] = [r[0] for r in reasons]
    top["facts"] = [r[1] for r in reasons]
    actual = tx["actual"]
    top["bought_next_week"] = [a in actual.get(c, set()) for c, a in zip(top["cid"], top["article_id"])]
    ids = ds.customers.set_index("cid")["customer_id"]
    top["customer_id"] = top["cid"].map(ids)
    top[["customer_id", "cid", "rank", "article_id", "score", "reason", "facts", "bought_next_week"]].to_parquet(out / "recs.parquet", index=False)

    used = set(top["article_id"]) | set(last["article_id"]) | set(tx["popular"])
    for v in tx["band_popular"].values():
        used |= set(v)
    a = ds.articles[ds.articles["article_id"].isin(used)][ARTICLE_COLS].copy()
    a["detail_desc"] = a["detail_desc"].str.slice(0, 220)
    a.to_parquet(out / "articles.parquet", index=False)

    h = last[last["cid"].isin(top["cid"].unique())].groupby("cid").head(8)
    h = h.assign(customer_id=h["cid"].map(ids))[["customer_id", "article_id", "t_dat"]]
    h.to_parquet(out / "history.parquet", index=False)

    # Demo list: mostly typical customers (2 to 4 hits), a few with no hits, no cherry-picked extremes
    per = top.groupby("customer_id")["bought_next_week"].sum()
    hist_len = h.groupby("customer_id").size()
    typical = per[(per >= 2) & (per <= 4)].index
    typical = hist_len.reindex(typical).fillna(0).sort_values(ascending=False).head(12).index.tolist()
    demo = typical + per[per == 0].head(3).index.tolist()
    (out / "popular.json").write_text(
        json.dumps({"global": [int(x) for x in tx["popular"]], "by_age_band": {k2: [int(x) for x in v] for k2, v in tx["band_popular"].items()}})
    )
    (out / "demo_customers.json").write_text(json.dumps(demo))
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "as_of": str((pd.Timestamp(cfg["sample"]["end_date"]) - pd.Timedelta(days=7)).date()),
                "customers": int(top["cid"].nunique()),
                "k": k,
            }
        )
    )
    build_similar(ds, hist, out)
    return out


def build_similar(ds, hist: pd.DataFrame, out: Path, n_items: int = 3000, n: int = 8) -> None:
    from implicit.nearest_neighbours import CosineRecommender

    from .candidates import Interactions

    inter = Interactions(hist, len(ds.customers))
    model = CosineRecommender(K=50)
    model.fit(inter.matrix, show_progress=False)
    top_items = hist["article_id"].value_counts().head(n_items).index.to_numpy()
    idx = inter.item_index.loc[top_items].to_numpy()
    ids, scores = model.similar_items(idx, N=n + 1)
    rows = []
    for i, item in enumerate(top_items):
        for j, s in zip(ids[i], scores[i]):
            other = inter.items[j]
            if other != item and s > 0:
                rows.append((int(item), int(other), float(s)))
    pd.DataFrame(rows, columns=["article_id", "similar_id", "score"]).to_parquet(out / "similar.parquet", index=False)


if __name__ == "__main__":
    from .config import load_config

    print(build_serving(load_config()))

__all__ = ["build_serving", "np"]
