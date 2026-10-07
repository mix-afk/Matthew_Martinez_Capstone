"""Data dictionary for the raw H&M tables used in this project."""
from __future__ import annotations

import pandas as pd

DICTIONARY = [
    # transactions_train
    ("transactions", "t_dat", "date (string in raw file)", "day", "2018-09-20 to 2020-09-22 (sample: 2020-06-24 to 2020-09-22)", "Purchase date. One row per unit sold"),
    ("transactions", "customer_id", "string (64-char hex hash)", "", "Matches customers.customer_id", "Anonymized customer key"),
    ("transactions", "article_id", "int64", "", "Matches articles.article_id", "Product variant key (product + colour)"),
    ("transactions", "price", "float64", "scaled currency", "0.00002 to 0.5915 (scaled by H&M, not real money)", "Unit price paid"),
    ("transactions", "sales_channel_id", "int64", "", "1, 2 (2 is widely read as online, 1 as store)", "Sales channel"),
    # customers
    ("customers", "customer_id", "string (64-char hex hash)", "", "Unique per row", "Anonymized customer key"),
    ("customers", "FN", "float64 (flag)", "", "1.0 or missing (missing = 0)", "Customer gets fashion news"),
    ("customers", "Active", "float64 (flag)", "", "1.0 or missing (missing = 0)", "Customer is active for communication"),
    ("customers", "club_member_status", "string", "", "ACTIVE, PRE-CREATE, LEFT CLUB, missing", "Loyalty club status"),
    ("customers", "fashion_news_frequency", "string", "", "NONE, None, Regularly, Monthly, missing (None and missing merged into NONE)", "Newsletter frequency"),
    ("customers", "age", "float64", "years", "16 to 99, about 1.2% missing", "Customer age. Used for the fairness audit"),
    ("customers", "postal_code", "string (hash)", "", "Hashed, 352,899 distinct values", "Postal code. Dropped (too granular, possible location proxy)"),
    # articles
    ("articles", "article_id", "int64", "", "105,542 unique", "Product variant key"),
    ("articles", "product_code", "int64", "", "47,224 unique", "Parent product (all colours of one item share it)"),
    ("articles", "prod_name", "string", "", "Free text", "Product name"),
    ("articles", "product_type_name", "string", "", "131 values, e.g. Trousers, Sweater, Dress", "Product type (paired code: product_type_no)"),
    ("articles", "product_group_name", "string", "", "19 values, e.g. Garment Upper body, Accessories", "Product group"),
    ("articles", "graphical_appearance_name", "string", "", "30 values, e.g. Solid, Stripe, All over pattern", "Print or pattern"),
    ("articles", "colour_group_name", "string", "", "50 values, e.g. Black, Light Beige", "Detailed colour (paired code: colour_group_code)"),
    ("articles", "perceived_colour_value_name", "string", "", "8 values, e.g. Dark, Light, Dusty Light", "Colour tone"),
    ("articles", "perceived_colour_master_name", "string", "", "20 values, e.g. Black, Blue, Beige", "Main colour family"),
    ("articles", "department_name", "string", "", "250 values", "Buying department"),
    ("articles", "index_name", "string", "", "10 values, e.g. Ladieswear, Divided, Menswear", "Store index"),
    ("articles", "index_group_name", "string", "", "5 values: Ladieswear, Baby/Children, Divided, Menswear, Sport", "Top-level index group"),
    ("articles", "section_name", "string", "", "56 values", "Section within index"),
    ("articles", "garment_group_name", "string", "", "21 values, e.g. Jersey Basic, Knitwear", "Garment group"),
    ("articles", "detail_desc", "string", "", "Free text, 416 missing", "Product description"),
]


def as_frame() -> pd.DataFrame:
    return pd.DataFrame(DICTIONARY, columns=["table", "variable", "type", "unit", "allowed_values_or_range", "description"])


def to_markdown(df: pd.DataFrame) -> str:
    out = ["# Data dictionary", "", "Source: H&M Personalized Fashion Recommendations (Kaggle competition data, parquet copy).", ""]
    for table, g in df.groupby("table", sort=False):
        out += [f"## {table}", "", "| Variable | Type | Unit | Allowed values / range | Description |", "|---|---|---|---|---|"]
        for _, r in g.iterrows():
            out.append(f"| {r.variable} | {r.type} | {r.unit} | {r.allowed_values_or_range} | {r.description} |")
        out.append("")
    return "\n".join(out)
