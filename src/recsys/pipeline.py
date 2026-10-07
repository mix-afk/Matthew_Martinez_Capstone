"""End-to-end pipeline. Run with: python -m recsys.pipeline --stage all"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as E
from . import ranker as R
from .candidates import article_embeddings
from .config import ensure_dirs, load_config, set_seed
from .data import load_dataset
from .features import ARTICLE_CAT_FEATURES, SENSITIVE_FEATURES, build_week_table, feature_columns

MODEL_NAMES = {"logreg": "Logistic regression", "lgbm_classifier": "LightGBM classifier", "lgbm_ranker": "LightGBM LambdaRank"}


def table_dir(cfg) -> Path:
    p = Path(cfg["paths"]["processed"]) / "tables"
    p.mkdir(parents=True, exist_ok=True)
    return p


def categories(ds) -> dict:
    return {c: sorted(ds.articles[c].unique()) for c in ARTICLE_CAT_FEATURES}


def get_embeddings(ds, cfg):
    path = Path(cfg["paths"]["processed"]) / "article_embeddings.parquet"
    if path.exists():
        return pd.read_parquet(path), None
    emb, pca, _ = article_embeddings(ds.articles, cfg["candidates"]["content_pca_components"], cfg["seed"])
    emb.to_parquet(path)
    return emb, pca


def build_tables(cfg, ds=None, force: bool = False) -> dict:
    ds = ds or load_dataset(cfg)
    emb, _ = get_embeddings(ds, cfg)
    weeks = [cfg["weeks"]["test"], cfg["weeks"]["valid"]] + list(cfg["weeks"]["train"])
    out = {}
    for w in weeks:
        tp, xp = table_dir(cfg) / f"week_{w}.parquet", table_dir(cfg) / f"week_{w}_extras.pkl"
        if tp.exists() and xp.exists() and not force:
            out[w] = tp
            continue
        t0 = time.time()
        table, extras = build_week_table(ds, emb, cfg, w, cfg["seed"])
        table.to_parquet(tp, index=False)
        keep = {k: extras[k] for k in ("per_source", "popular", "band_popular", "actual", "cids")}
        with open(xp, "wb") as f:
            pickle.dump(keep, f)
        print(f"week {w}: {len(table):,} rows, {table['label'].mean():.4f} positive, {time.time() - t0:.0f}s")
        out[w] = tp
    return out


def load_week(cfg, w: int) -> tuple[pd.DataFrame, dict]:
    table = pd.read_parquet(table_dir(cfg) / f"week_{w}.parquet")
    for c in ARTICLE_CAT_FEATURES:
        table[c] = table[c].astype("category")
    with open(table_dir(cfg) / f"week_{w}_extras.pkl", "rb") as f:
        extras = pickle.load(f)
    return table, extras


def load_train(cfg) -> pd.DataFrame:
    parts = [load_week(cfg, w)[0] for w in cfg["weeks"]["train"]]
    train = pd.concat(parts, ignore_index=True)
    has_pos = train.groupby(["week", "cid"])["label"].transform("max") > 0
    return train[has_pos].reset_index(drop=True)


def mlflow_log(cfg, run_name: str, params: dict, metrics: dict, artifacts: list | None = None) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{Path(cfg['paths']['mlflow']) / 'mlflow.db'}")
        mlflow.set_experiment("hm-recommender")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: v for k, v in params.items() if v is not None})
            mlflow.log_metrics({k.replace("@", "_at_"): float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating))})
            for a in artifacts or []:
                mlflow.log_artifact(str(a))
    except Exception as exc:  # tracking must never break training
        print(f"mlflow logging skipped: {exc}")


def train(cfg, drop_features: tuple = (), tag: str = "", save: bool = True) -> dict:
    set_seed(cfg["seed"])
    ds = load_dataset(cfg)
    cats = categories(ds)
    train_df = load_train(cfg)
    valid, vx = load_week(cfg, cfg["weeks"]["valid"])
    cols = [c for c in feature_columns(train_df) if c not in drop_features]
    fallback = vx["popular"]
    seed, n_trials = cfg["seed"], cfg["ranker"]["tuning_trials"]
    results = {}

    print(f"train rows {len(train_df):,}, features {len(cols)}", flush=True)
    lr, lr_m, lr_grid = R.fit_logreg(train_df, valid, cols, vx["actual"], fallback, cfg, seed, grid=(0.1, 1.0))
    print(f"logreg done: valid MAP@12 {lr_m['map@12']:.4f}", flush=True)
    results["logreg"] = (lr, {"C": lr_m["C"]}, lr_m)
    mlflow_log(cfg, f"logreg{tag}", {"C": lr_m["C"], "n_features": len(cols)}, lr_m)

    (clf, clf_p, clf_m), clf_trials = R.tune(
        R.fit_lgbm_classifier, n_trials, seed, train=train_df, valid=valid, cols=cols, cats=cats,
        actual_valid=vx["actual"], fallback=fallback, cfg=cfg, seed=seed,
    )
    results["lgbm_classifier"] = (clf, clf_p, clf_m)
    print(f"lgbm classifier done: valid MAP@12 {clf_m['map@12']:.4f}", flush=True)
    mlflow_log(cfg, f"lgbm_classifier{tag}", {**clf_p, "n_features": len(cols)}, clf_m)

    def rank_fn(params, **kw):
        m, met, _ = R.fit_lgbm_ranker(params=params, **kw)
        return m, met

    (rk, rk_p, rk_m), rk_trials = R.tune(
        rank_fn, max(4, n_trials // 2), seed + 1, train=train_df, valid=valid, cols=cols, cats=cats,
        actual_valid=vx["actual"], fallback=fallback, cfg=cfg, seed=seed,
    )
    results["lgbm_ranker"] = (rk, rk_p, rk_m)
    print(f"lgbm ranker done: valid MAP@12 {rk_m['map@12']:.4f}", flush=True)
    mlflow_log(cfg, f"lgbm_ranker{tag}", {**rk_p, "n_features": len(cols)}, rk_m)

    if save:
        tables = Path(cfg["paths"]["tables"])
        lr_grid.to_csv(tables / f"tuning_logreg{tag}.csv", index=False)
        clf_trials.to_csv(tables / f"tuning_lgbm_classifier{tag}.csv", index=False)
        rk_trials.to_csv(tables / f"tuning_lgbm_ranker{tag}.csv", index=False)
        meta = {
            "features": cols,
            "categories": cats,
            "params": {k: v[1] for k, v in results.items()},
            "valid_metrics": {k: v[2] for k, v in results.items()},
            "train_weeks": cfg["weeks"]["train"],
            "valid_week": cfg["weeks"]["valid"],
            "seed": seed,
            "train_rows": len(train_df),
        }
        models_dir = Path(cfg["paths"]["models"]) / (tag.strip("_") or "")
        R.save_models(models_dir, {k: v[0] for k, v in results.items()}, meta)
    return {"models": {k: v[0] for k, v in results.items()}, "cols": cols, "cats": cats, "valid": {k: v[2] for k, v in results.items()}}


def evaluate_test(cfg, trained: dict) -> pd.DataFrame:
    test, tx = load_week(cfg, cfg["weeks"]["test"])
    k = cfg["eval"]["k"]
    n_catalog = int(test["article_id"].nunique())
    rows = []
    for name, pred in tx["per_source"].items():
        m = E.summarize(tx["actual"], pred, k, fallback=tx["popular"], n_catalog=n_catalog)
        rows.append({"stage": "1 candidate", "model": name, **m})
    for name, model in trained["models"].items():
        s = R.predict(model, test, trained["cols"], trained["cats"])
        d = test[["cid", "article_id"]].copy()
        d["score"] = s
        pred = E.top_k_from_scores(d, "score", k)
        m = E.summarize(tx["actual"], pred, k, fallback=tx["popular"], n_catalog=n_catalog)
        m["auc"] = E.auc(test["label"], s)
        rows.append({"stage": "2 ranker", "model": MODEL_NAMES.get(name, name), **m})
    out = pd.DataFrame(rows)
    out["candidate_recall"] = E.candidate_recall(test, tx["actual"])
    return out


def load_trained(cfg, subdir: str = "") -> dict:
    import joblib

    d = Path(cfg["paths"]["models"]) / subdir
    meta = json.loads((d / "model_meta.json").read_text())
    models = {name: joblib.load(d / f"{name}.joblib") for name in MODEL_NAMES}
    return {"models": models, "cols": meta["features"], "cats": meta["categories"], "valid": meta["valid_metrics"], "meta": meta}


def predictions(cfg, trained: dict, name: str, table: pd.DataFrame) -> dict:
    d = table[["cid", "article_id"]].copy()
    d["score"] = R.predict(trained["models"][name], table, trained["cols"], trained["cats"])
    return E.top_k_from_scores(d, "score", cfg["eval"]["k"])


def overfit_table(cfg, trained: dict, name: str) -> pd.DataFrame:
    """Same metrics on the training weeks, the validation week and the test week."""
    k = cfg["eval"]["k"]
    rows = []
    splits = [("train", w) for w in cfg["weeks"]["train"]] + [("valid", cfg["weeks"]["valid"]), ("test", cfg["weeks"]["test"])]
    scores = {}
    for split, w in splits:
        t, x = load_week(cfg, w)
        s = R.predict(trained["models"][name], t, trained["cols"], trained["cats"])
        m = R.score_map(t, s, x["actual"], k, x["popular"])
        scores.setdefault(split, []).append({**m, "auc": E.auc(t["label"], s)})
    for split, ms in scores.items():
        rows.append({"split": split, "weeks": len(ms), **{key: float(np.mean([m[key] for m in ms])) for key in ("map@12", "recall@12", "hit_rate", "auc")}})
    return pd.DataFrame(rows)


def rounds_curve(cfg, trained: dict, rounds: list[int]) -> pd.DataFrame:
    from .features import to_model_frame

    model = trained["models"]["lgbm_classifier"]
    rows = []
    for split, w in (("valid", cfg["weeks"]["valid"]), ("test", cfg["weeks"]["test"])):
        t, x = load_week(cfg, w)
        xt = to_model_frame(t, trained["cols"], trained["cats"])
        for n in rounds:
            s = model.booster_.predict(xt, num_iteration=n)
            rows.append({"rounds": n, "split": split, "map@12": R.score_map(t, s, x["actual"], cfg["eval"]["k"], x["popular"])["map@12"]})
    out = pd.DataFrame(rows).pivot(index="rounds", columns="split", values="map@12").reset_index()
    return out.rename(columns={"valid": "valid_map@12", "test": "test_map@12"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["tables", "train", "evaluate", "all"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if args.stage in ("tables", "all"):
        build_tables(cfg)
    if args.stage in ("train", "evaluate", "all"):
        trained = train(cfg)
        res = evaluate_test(cfg, trained)
        res.to_csv(Path(cfg["paths"]["tables"]) / "model_comparison_test.csv", index=False)
        print(res.round(4).to_string(index=False))
        with open(Path(cfg["paths"]["models"]) / "test_metrics.json", "w") as f:
            json.dump(res.to_dict(orient="records"), f, indent=2)


if __name__ == "__main__":
    main()

__all__ = ["SENSITIVE_FEATURES", "build_tables", "train", "evaluate_test", "load_week", "load_train"]
