"""Data sampling, loading and cleaning for the H&M dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Row counts of the official competition files, verified against the parquet copy used here
FULL_COUNTS = {"transactions": 31_788_324, "customers": 1_371_980, "articles": 105_542}

ARTICLE_CATEGORICALS = [
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "perceived_colour_value_name",
    "perceived_colour_master_name",
    "department_name",
    "index_name",
    "index_group_name",
    "section_name",
    "garment_group_name",
]


def make_sample(raw_dir: str | Path, out_dir: str | Path, cfg: dict) -> dict:
    """Cut the raw Kaggle parquet files down to the configured window and customer sample.

    Uses DuckDB so it streams the 31.8M-row transactions file without loading it into memory.
    """
    import duckdb

    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = cfg["sample"]
    keep = [format(i, "02x") for i in range(s["customer_hex_buckets"])]
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'; SET preserve_insertion_order=false")
    con.execute(f"CREATE TEMP TABLE keep AS SELECT unnest({keep}) AS h")
    tx = (raw_dir / "transactions_train.parquet").as_posix()
    cu = (raw_dir / "customers.parquet").as_posix()
    con.execute(
        f"""COPY (SELECT * FROM read_parquet('{tx}')
        WHERE t_dat >= '{s['start_date']}' AND t_dat <= '{s['end_date']}'
        AND right(customer_id, 2) IN (SELECT h FROM keep))
        TO '{(out_dir / 'transactions_sample.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)"""
    )
    con.execute(
        f"""COPY (SELECT * FROM read_parquet('{cu}') WHERE right(customer_id, 2) IN (SELECT h FROM keep))
        TO '{(out_dir / 'customers_sample.parquet').as_posix()}' (FORMAT parquet, COMPRESSION zstd)"""
    )
    con.execute(
        f"""COPY (SELECT t_dat, sales_channel_id, count(*) AS n_tx, count(DISTINCT customer_id) AS n_customers,
        count(DISTINCT article_id) AS n_articles, sum(price) AS revenue_units
        FROM read_parquet('{tx}') GROUP BY 1, 2 ORDER BY 1, 2)
        TO '{(out_dir / 'daily_full.parquet').as_posix()}' (FORMAT parquet)"""
    )
    art = pd.read_parquet(raw_dir / "articles.parquet")
    art.to_parquet(out_dir / "articles.parquet", index=False)
    n = con.execute(f"SELECT count(*) FROM read_parquet('{(out_dir / 'transactions_sample.parquet').as_posix()}')").fetchone()[0]
    return {"transactions_sample_rows": int(n)}


@dataclass
class Dataset:
    transactions: pd.DataFrame
    customers: pd.DataFrame
    articles: pd.DataFrame
    daily_full: pd.DataFrame
    end_date: pd.Timestamp


def load_raw(processed_dir: str | Path) -> dict[str, pd.DataFrame]:
    p = Path(processed_dir)
    return {
        "transactions": pd.read_parquet(p / "transactions_sample.parquet"),
        "customers": pd.read_parquet(p / "customers_sample.parquet"),
        "articles": pd.read_parquet(p / "articles.parquet"),
        "daily_full": pd.read_parquet(p / "daily_full.parquet"),
    }


def age_band(age: pd.Series, bins: list[int], labels: list[str]) -> pd.Series:
    band = pd.cut(age, bins=bins, labels=labels, right=False)
    return band.cat.add_categories(["Unknown"]).fillna("Unknown")


def clean_customers(c: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = c.copy()
    c["FN"] = c["FN"].fillna(0).astype("int8")
    c["Active"] = c["Active"].fillna(0).astype("int8")
    c["club_member_status"] = c["club_member_status"].fillna("UNKNOWN")
    c["fashion_news_frequency"] = (
        c["fashion_news_frequency"].replace({"None": "NONE"}).fillna("NONE")
    )
    c["age_missing"] = c["age"].isna().astype("int8")
    c["age_band"] = age_band(c["age"], cfg["fairness"]["age_bins"], cfg["fairness"]["age_labels"])
    c["age_filled"] = c["age"].fillna(c["age"].median())
    c = c.drop(columns=["postal_code"])
    c = c.reset_index(drop=True)
    c["cid"] = np.arange(len(c), dtype=np.int32)
    return c


def clean_articles(a: pd.DataFrame) -> pd.DataFrame:
    a = a.copy()
    a["detail_desc"] = a["detail_desc"].fillna("")
    for col in ARTICLE_CATEGORICALS:
        a[col] = a[col].fillna("Unknown")
    keep = ["article_id", "product_code", "prod_name", "detail_desc"] + ARTICLE_CATEGORICALS
    return a[keep].reset_index(drop=True)


def clean_transactions(t: pd.DataFrame, customers: pd.DataFrame, end_date: pd.Timestamp) -> pd.DataFrame:
    t = t.copy()
    t["t_dat"] = pd.to_datetime(t["t_dat"])
    t["week"] = ((end_date - t["t_dat"]).dt.days // 7).astype("int16")
    t["online"] = (t["sales_channel_id"] == 2).astype("int8")
    t = t.merge(customers[["customer_id", "cid"]], on="customer_id", how="inner")
    t = t.drop(columns=["customer_id", "sales_channel_id"])
    return t.sort_values(["t_dat", "cid"]).reset_index(drop=True)


def load_dataset(cfg: dict) -> Dataset:
    raw = load_raw(cfg["paths"]["processed"])
    end = pd.Timestamp(cfg["sample"]["end_date"])
    customers = clean_customers(raw["customers"], cfg)
    articles = clean_articles(raw["articles"])
    tx = clean_transactions(raw["transactions"], customers, end)
    daily = raw["daily_full"].copy()
    daily["t_dat"] = pd.to_datetime(daily["t_dat"])
    return Dataset(tx, customers, articles, daily, end)


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "missing": int(s.isna().sum()),
                "missing_pct": round(100 * s.isna().mean(), 2),
                "unique": int(s.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def iqr_outliers(s: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def week_dates(week: int, end_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    last = end_date - pd.Timedelta(days=7 * week)
    return last - pd.Timedelta(days=6), last


def history(tx: pd.DataFrame, target_week: int, n_weeks: int) -> pd.DataFrame:
    return tx[(tx["week"] > target_week) & (tx["week"] <= target_week + n_weeks)]


def actual_purchases(tx: pd.DataFrame, week: int) -> dict[int, set]:
    w = tx[tx["week"] == week][["cid", "article_id"]].drop_duplicates()
    return w.groupby("cid")["article_id"].agg(set).to_dict()
