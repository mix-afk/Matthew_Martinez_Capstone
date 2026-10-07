import json

import pandas as pd
from fastapi.testclient import TestClient


def make_artifacts(d):
    pd.DataFrame(
        {"customer_id": ["abc"] * 2, "cid": [0, 0], "rank": [1, 2], "article_id": [101, 102], "score": [0.9, 0.5],
         "reason": ["One of this week's best sellers"] * 2, "facts": ["it is one of this week's best sellers"] * 2, "bought_next_week": [True, False]}
    ).to_parquet(d / "recs.parquet")
    pd.DataFrame(
        {"article_id": [101, 102, 103], "prod_name": ["Tee", "Jeans", "Cap"], "product_type_name": ["T-shirt", "Trousers", "Cap"],
         "product_group_name": ["Upper", "Lower", "Acc"], "colour_group_name": ["Black", "Blue", "Red"],
         "perceived_colour_master_name": ["Black", "Blue", "Red"], "index_group_name": ["Ladieswear"] * 3,
         "garment_group_name": ["Jersey"] * 3, "detail_desc": ["", "", ""]}
    ).to_parquet(d / "articles.parquet")
    pd.DataFrame({"customer_id": ["abc"], "article_id": [103], "t_dat": [pd.Timestamp("2020-09-10")]}).to_parquet(d / "history.parquet")
    pd.DataFrame({"article_id": [101], "similar_id": [102], "score": [0.4]}).to_parquet(d / "similar.parquet")
    (d / "popular.json").write_text(json.dumps({"global": [103, 101], "by_age_band": {"16-24": [102]}}))
    (d / "demo_customers.json").write_text(json.dumps(["abc"]))
    (d / "manifest.json").write_text(json.dumps({"model": "test", "as_of": "2020-09-15", "customers": 1, "k": 12}))


def client(tmp_path, monkeypatch):
    make_artifacts(tmp_path)
    monkeypatch.setenv("RECSYS_SERVING_DIR", str(tmp_path))
    from recsys import api

    api.store.cache_clear()
    return TestClient(api.app)


def test_health(tmp_path, monkeypatch):
    r = client(tmp_path, monkeypatch).get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_known_customer_gets_ranked_items(tmp_path, monkeypatch):
    d = client(tmp_path, monkeypatch).get("/recommendations/abc").json()
    assert d["strategy"] == "two_stage_ranker"
    assert [i["article_id"] for i in d["items"]] == [101, 102]
    assert d["items"][0]["bought_next_week"] is True


def test_cold_start_uses_age_band_popularity(tmp_path, monkeypatch):
    d = client(tmp_path, monkeypatch).get("/recommendations/new?age_band=16-24").json()
    assert d["strategy"] == "cold_start_popular"
    assert d["items"][0]["article_id"] == 102


def test_similar_items(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    assert c.get("/similar/101").json()["similar"][0]["article_id"] == 102
    assert c.get("/similar/999").status_code == 404
