"""FastAPI app serving batch-scored recommendations. Run: uvicorn recsys.api:app --reload"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from .genai import llm_reason

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
AGE_BANDS = ["16-24", "25-34", "35-44", "45-54", "55+", "Unknown"]


def serving_dir() -> Path:
    return Path(os.environ.get("RECSYS_SERVING_DIR", ROOT / "data" / "processed" / "serving"))


@lru_cache(maxsize=1)
def store() -> dict:
    d = serving_dir()
    if not (d / "recs.parquet").exists():
        raise RuntimeError(f"No serving artifacts in {d}. Run: python -m recsys.serving")
    recs = pd.read_parquet(d / "recs.parquet")
    arts = pd.read_parquet(d / "articles.parquet").set_index("article_id")
    hist = pd.read_parquet(d / "history.parquet")
    sim = pd.read_parquet(d / "similar.parquet")
    return {
        "recs": {cid: g.sort_values("rank") for cid, g in recs.groupby("customer_id")},
        "articles": arts,
        "history": {cid: g for cid, g in hist.groupby("customer_id")},
        "similar": {a: g.sort_values("score", ascending=False) for a, g in sim.groupby("article_id")},
        "popular": json.loads((d / "popular.json").read_text()),
        "demo": json.loads((d / "demo_customers.json").read_text()),
        "manifest": json.loads((d / "manifest.json").read_text()),
    }


def article_card(article_id: int) -> dict:
    a = store()["articles"]
    if article_id not in a.index:
        return {"article_id": int(article_id)}
    r = a.loc[article_id]
    return {
        "article_id": int(article_id),
        "name": r["prod_name"],
        "type": r["product_type_name"],
        "group": r["product_group_name"],
        "colour": r["colour_group_name"],
        "colour_family": r["perceived_colour_master_name"],
        "index_group": r["index_group_name"],
        "description": r["detail_desc"],
    }


app = FastAPI(title="H&M recommender demo", version="1.0.0", description="Two-stage recommender (candidates + LightGBM ranker).")


@app.get("/health")
def health() -> dict:
    s = store()
    return {"status": "ok", **s["manifest"]}


@app.get("/customers/demo")
def demo_customers() -> list[str]:
    return store()["demo"]


@app.get("/recommendations/{customer_id}")
def recommendations(
    customer_id: str,
    k: int = Query(12, ge=1, le=12),
    explain: str = Query("template", pattern="^(template|llm)$"),
    age_band: str | None = Query(None, description="Only used for cold-start customers"),
) -> dict:
    s = store()
    recs = s["recs"].get(customer_id)
    hist = s["history"].get(customer_id)
    history = [] if hist is None else [article_card(a) for a in hist["article_id"]]
    if recs is None:
        if age_band and age_band not in AGE_BANDS:
            raise HTTPException(422, f"age_band must be one of {AGE_BANDS}")
        items = s["popular"]["by_age_band"].get(age_band or "", s["popular"]["global"])[:k]
        return {
            "customer_id": customer_id,
            "strategy": "cold_start_popular",
            "history": history,
            "items": [{**article_card(a), "rank": i + 1, "reason": "One of this week's best sellers", "reason_source": "template"} for i, a in enumerate(items)],
        }
    items = []
    for row in recs.head(k).itertuples(index=False):
        card = article_card(row.article_id)
        reason, source = row.reason, "template"
        if explain == "llm":
            reason, source = llm_reason(f"{card.get('name')} ({card.get('type')}, {card.get('colour')})", row.facts.split("; "))
        items.append({**card, "rank": int(row.rank), "score": float(row.score), "reason": reason, "reason_source": source, "bought_next_week": bool(row.bought_next_week)})
    return {"customer_id": customer_id, "strategy": "two_stage_ranker", "as_of": s["manifest"]["as_of"], "history": history, "items": items}


@app.get("/similar/{article_id}")
def similar(article_id: int, k: int = Query(8, ge=1, le=8)) -> dict:
    g = store()["similar"].get(article_id)
    if g is None:
        raise HTTPException(404, "No co-purchase neighbours for this item")
    return {"article": article_card(article_id), "similar": [{**article_card(r.similar_id), "score": float(r.score)} for r in g.head(k).itertuples()]}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
