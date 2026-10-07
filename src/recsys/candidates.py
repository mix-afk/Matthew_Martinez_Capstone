"""Stage 1: candidate generators. Each one is also a standalone recommender we can evaluate."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA

from .data import ARTICLE_CATEGORICALS


class Interactions:
    """User-item matrix built from a history window."""

    def __init__(self, hist: pd.DataFrame, n_customers: int):
        self.items = np.sort(hist["article_id"].unique())
        self.item_index = pd.Series(np.arange(len(self.items)), index=self.items)
        counts = hist.groupby(["cid", "article_id"]).size().reset_index(name="n")
        rows = counts["cid"].to_numpy()
        cols = self.item_index.loc[counts["article_id"]].to_numpy()
        self.matrix = sp.csr_matrix(
            (counts["n"].to_numpy(dtype=np.float32), (rows, cols)),
            shape=(n_customers, len(self.items)),
        )


def popular(hist: pd.DataFrame, n: int, weeks: int = 1) -> list:
    recent = hist[hist["week"] <= hist["week"].min() + weeks - 1]
    return recent["article_id"].value_counts().head(n).index.tolist()


def age_band_popular(hist: pd.DataFrame, customers: pd.DataFrame, n: int, weeks: int = 2) -> dict:
    recent = hist[hist["week"] <= hist["week"].min() + weeks - 1]
    recent = recent.merge(customers[["cid", "age_band"]], on="cid")
    out = {}
    for band, g in recent.groupby("age_band", observed=True):
        out[band] = g["article_id"].value_counts().head(n).index.tolist()
    return out


def repurchase(hist: pd.DataFrame, cids: np.ndarray, max_n: int, ref_date: pd.Timestamp) -> pd.DataFrame:
    h = hist[hist["cid"].isin(cids)]
    g = h.groupby(["cid", "article_id"]).agg(rep_count=("t_dat", "size"), last=("t_dat", "max")).reset_index()
    g["rep_days_since"] = (ref_date - g["last"]).dt.days.astype("float32")
    g = g.sort_values(["cid", "last", "rep_count"], ascending=[True, False, False])
    g = g.groupby("cid").head(max_n).drop(columns="last")
    g["rep_rank"] = g.groupby("cid").cumcount().astype("float32") + 1
    return g


def product_siblings(hist: pd.DataFrame, cids: np.ndarray, articles: pd.DataFrame, n: int) -> pd.DataFrame:
    """Other colors/variants of products the customer already bought (same product_code)."""
    recent = hist[hist["week"] <= hist["week"].min() + 1]
    sold = recent["article_id"].value_counts().rename("sib_pop").reset_index()
    code = articles[["article_id", "product_code"]]
    sold = sold.merge(code, on="article_id")
    h = hist[hist["cid"].isin(cids)][["cid", "article_id"]].drop_duplicates().merge(code, on="article_id")
    s = h[["cid", "product_code"]].drop_duplicates().merge(sold, on="product_code")
    s = s[~s.set_index(["cid", "article_id"]).index.isin(h.set_index(["cid", "article_id"]).index)]
    s = s.sort_values(["cid", "sib_pop"], ascending=[True, False]).groupby("cid").head(n)
    s["sib_rank"] = s.groupby("cid").cumcount().astype("float32") + 1
    s["sib_pop"] = s["sib_pop"].astype("float32")
    return s[["cid", "article_id", "sib_pop", "sib_rank"]]


def _recommend_frame(model, inter: Interactions, cids: np.ndarray, n: int, prefix: str) -> pd.DataFrame:
    active = cids[inter.matrix[cids].getnnz(axis=1) > 0]
    if len(active) == 0:
        return pd.DataFrame(columns=["cid", "article_id", f"{prefix}_score", f"{prefix}_rank"])
    ids, scores = model.recommend(active, inter.matrix[active], N=n, filter_already_liked_items=True)
    frame = pd.DataFrame(
        {
            "cid": np.repeat(active, ids.shape[1]),
            "item": ids.ravel(),
            f"{prefix}_score": scores.ravel().astype("float32"),
            f"{prefix}_rank": np.tile(np.arange(1, ids.shape[1] + 1), len(active)).astype("float32"),
        }
    )
    frame = frame[(frame["item"] >= 0) & (frame[f"{prefix}_score"] > 0)]
    frame["article_id"] = inter.items[frame["item"].to_numpy()]
    return frame.drop(columns="item")


def item_cf(inter: Interactions, cids: np.ndarray, n: int, neighbors: int) -> pd.DataFrame:
    from implicit.nearest_neighbours import CosineRecommender

    model = CosineRecommender(K=neighbors)
    model.fit(inter.matrix, show_progress=False)
    return _recommend_frame(model, inter, cids, n, "itemcf"), model


def als(inter: Interactions, cids: np.ndarray, n: int, cfg: dict, seed: int) -> pd.DataFrame:
    from implicit.als import AlternatingLeastSquares

    c = cfg["candidates"]
    model = AlternatingLeastSquares(
        factors=c["als_factors"],
        regularization=c["als_regularization"],
        iterations=c["als_iterations"],
        alpha=c["als_alpha"],
        random_state=seed,
        num_threads=0,
    )
    model.fit(inter.matrix, show_progress=False)
    return _recommend_frame(model, inter, cids, n, "als"), model


def article_embeddings(articles: pd.DataFrame, n_components: int, seed: int) -> tuple[pd.DataFrame, PCA, np.ndarray]:
    """One-hot encode article attributes and compress them with PCA (content-based features)."""
    onehot = pd.get_dummies(articles[ARTICLE_CATEGORICALS], sparse=False, dtype=np.float32)
    x = onehot.to_numpy(dtype=np.float32)
    x = x - x.mean(axis=0)
    pca = PCA(n_components=n_components, random_state=seed, svd_solver="randomized")
    z = pca.fit_transform(x).astype(np.float32)
    emb = pd.DataFrame(z, index=articles["article_id"].to_numpy(), columns=[f"pc{i + 1}" for i in range(n_components)])
    return emb, pca, onehot.columns.to_numpy()


def content_based(hist: pd.DataFrame, cids: np.ndarray, emb: pd.DataFrame, n: int, pool_size: int) -> pd.DataFrame:
    recent = hist[hist["week"] <= hist["week"].min() + 1]
    pool = recent["article_id"].value_counts().head(pool_size).index.to_numpy()
    e_pool = emb.loc[pool].to_numpy()
    e_pool = e_pool / (np.linalg.norm(e_pool, axis=1, keepdims=True) + 1e-9)
    h = hist[hist["cid"].isin(cids)][["cid", "article_id", "week"]].copy()
    h["w"] = 1.0 / (1 + h["week"] - hist["week"].min())
    vecs = emb.loc[h["article_id"]].to_numpy() * h["w"].to_numpy()[:, None]
    prof = pd.DataFrame(vecs, index=h["cid"].to_numpy()).groupby(level=0).sum()
    users = prof.index.to_numpy()
    u = prof.to_numpy()
    u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-9)
    bought = h.groupby("cid")["article_id"].agg(set).to_dict()
    out_c, out_a, out_s = [], [], []
    for start in range(0, len(users), 4000):
        s = u[start : start + 4000] @ e_pool.T
        top = np.argpartition(-s, n + 20, axis=1)[:, : n + 20]
        for row, uid in enumerate(users[start : start + 4000]):
            idx = top[row][np.argsort(-s[row, top[row]])]
            arts = pool[idx]
            sc = s[row, idx]
            mask = ~np.isin(arts, list(bought.get(uid, ())))
            arts, sc = arts[mask][:n], sc[mask][:n]
            out_c.append(np.full(len(arts), uid))
            out_a.append(arts)
            out_s.append(sc)
    out = pd.DataFrame(
        {
            "cid": np.concatenate(out_c),
            "article_id": np.concatenate(out_a),
            "content_score": np.concatenate(out_s).astype("float32"),
        }
    )
    out["content_rank"] = out.groupby("cid").cumcount().astype("float32") + 1
    return out


SOURCE_RANKS = ["rep_rank", "itemcf_rank", "als_rank", "content_rank", "sib_rank", "pop_rank", "bandpop_rank"]


def build_candidates(
    hist: pd.DataFrame,
    cids: np.ndarray,
    customers: pd.DataFrame,
    articles: pd.DataFrame,
    emb: pd.DataFrame,
    cfg: dict,
    ref_date: pd.Timestamp,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Union of all generators for the target customers. Returns candidates and each source's top-12."""
    c = cfg["candidates"]
    k = cfg["eval"]["k"]
    inter = Interactions(hist, len(customers))
    pop = popular(hist, c["popular_n"])
    band_pop = age_band_popular(hist, customers, c["age_band_popular_n"])
    rep = repurchase(hist, cids, c["repurchase_max"], ref_date)
    icf, _ = item_cf(inter, cids, c["itemcf_n"], c["itemcf_neighbors"])
    als_df, als_model = als(inter, cids, c["als_n"], cfg, seed)
    cont = content_based(hist, cids, emb, c["content_n"], c["content_pool"])
    sib = product_siblings(hist, cids, articles, c["siblings_n"])

    pop_df = pd.DataFrame({"article_id": pop, "pop_rank": np.arange(1, len(pop) + 1, dtype="float32")})
    pop_all = pd.DataFrame({"cid": np.repeat(cids, len(pop)), "article_id": np.tile(pop, len(cids))})
    pop_all = pop_all.merge(pop_df, on="article_id")
    bands = customers.set_index("cid").loc[cids, "age_band"].astype(str)
    band_rows = []
    for band, idx in bands.groupby(bands).groups.items():
        items = band_pop.get(band, band_pop.get("25-34", pop))
        band_rows.append(
            pd.DataFrame(
                {
                    "cid": np.repeat(np.asarray(idx), len(items)),
                    "article_id": np.tile(items, len(idx)),
                    "bandpop_rank": np.tile(np.arange(1, len(items) + 1, dtype="float32"), len(idx)),
                }
            )
        )
    band_df = pd.concat(band_rows, ignore_index=True)

    cand = rep
    for part in (icf, als_df, cont, sib, pop_all, band_df):
        cand = cand.merge(part, on=["cid", "article_id"], how="outer")
    cand = cand[cand["cid"].isin(cids)].reset_index(drop=True)
    cand["n_sources"] = cand[SOURCE_RANKS].notna().sum(axis=1).astype("int8")

    def top(df, col, asc=True):
        d = df.sort_values(["cid", col], ascending=[True, asc]).groupby("cid").head(k)
        return d.groupby("cid")["article_id"].agg(list).to_dict()

    per_source = {
        "popularity": {cid: pop[:k] for cid in cids},
        "age_band_popularity": {cid: band_pop.get(b, pop)[:k] for cid, b in bands.items()},
        "repurchase": top(rep, "rep_rank"),
        "item_cf": top(icf, "itemcf_rank"),
        "als": top(als_df, "als_rank"),
        "content": top(cont, "content_rank"),
        "product_siblings": top(sib, "sib_rank"),
    }
    extras = {"popular": pop, "band_popular": band_pop, "als_model": als_model, "interactions": inter}
    return cand, {"per_source": per_source, **extras}
