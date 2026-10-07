"""Generative AI layer: one-line 'why this item' reasons and LLM-drafted documentation.

The LLM is optional. Without ANTHROPIC_API_KEY every call falls back to a deterministic template,
so the app, tests and notebooks always run.
"""
from __future__ import annotations

import json
import os

import pandas as pd

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
BLOCKED_TERMS = ("age", "old", "young", "gender", "woman", "man ", "men ", "female", "male")

REASON_PROMPT = """You write one short line that tells an online shopper why a product was recommended.
Rules:
- Use only the facts given. Do not invent discounts, stock levels or reviews.
- Never mention or hint at the shopper's age, gender or any personal trait.
- Max 16 words, plain friendly English, no emojis, no exclamation marks.

Product: {product}
Facts: {facts}

Reply with the line only."""

DICTIONARY_PROMPT = """You are helping document a dataset for a data science report.
For each column below, write a one-sentence plain-English description, the data type, and the allowed values or range.
Return JSON: a list of objects with keys column, description, type, allowed_values.
Only describe what the profile supports; write "unknown" when the meaning cannot be inferred.

Table: {table}
Column profile:
{profile}"""


def reason_facts(row: pd.Series, article: pd.Series, history_names: list[str] | None = None) -> list[str]:
    facts = []
    if row.get("rep_count", 0) > 0:
        facts.append("the shopper bought this exact item before")
    if pd.notna(row.get("sib_rank")):
        facts.append(f"it is another color or version of a {article['product_type_name'].lower()} the shopper bought")
    if pd.notna(row.get("itemcf_rank")) and history_names:
        facts.append(f"shoppers who bought {history_names[0]} also bought this")
    if row.get("ci_ptype_share", 0) >= 0.2:
        facts.append(f"the shopper often buys {article['product_type_name'].lower()}s")
    if row.get("ci_colour_share", 0) >= 0.3:
        facts.append(f"the shopper often picks {article['perceived_colour_master_name'].lower()} pieces")
    if row.get("a_trend", 0) >= 1.5:
        facts.append("sales of this item are rising this week")
    if pd.notna(row.get("pop_rank")) and row.get("pop_rank", 99) <= 12:
        facts.append("it is one of this week's best sellers")
    if pd.notna(row.get("als_rank")) and not facts:
        facts.append("it matches the overall style of the shopper's past purchases")
    return facts or ["it is popular with shoppers right now"]


def template_reason(facts: list[str]) -> str:
    first = facts[0]
    mapping = [
        ("bought this exact item before", "Buy it again: you have ordered this before"),
        ("another color or version", "A new color of something you already love"),
        ("also bought this", "Often bought together with your past picks"),
        ("often buys", "Matches the kind of pieces you buy most"),
        ("often picks", "In the colors you usually go for"),
        ("rising this week", "Trending now: sales are climbing this week"),
        ("best sellers", "One of this week's best sellers"),
        ("overall style", "Picked to match your style"),
    ]
    for key, text in mapping:
        if key in first:
            return text
    return "Popular with shoppers right now"


def is_safe(text: str) -> bool:
    t = f" {text.lower()} "
    return not any(term in t for term in BLOCKED_TERMS) and len(text.split()) <= 20


def llm_reason(product: str, facts: list[str], client=None, model: str | None = None) -> tuple[str, str]:
    """Returns (reason, source). Falls back to the template on no key, error, or a guardrail hit."""
    fallback = template_reason(facts)
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return fallback, "template"
        try:
            import anthropic

            client = anthropic.Anthropic()
        except Exception:
            return fallback, "template"
    try:
        msg = client.messages.create(
            model=model or os.environ.get("RECSYS_LLM_MODEL", DEFAULT_MODEL),
            max_tokens=60,
            messages=[{"role": "user", "content": REASON_PROMPT.format(product=product, facts="; ".join(facts))}],
        )
        text = msg.content[0].text.strip().strip('"')
        return (text, "llm") if is_safe(text) else (fallback, "template_guardrail")
    except Exception:
        return fallback, "template"


def column_profile(df: pd.DataFrame, max_examples: int = 5) -> str:
    lines = []
    for col in df.columns:
        s = df[col]
        info = {"dtype": str(s.dtype), "missing_pct": round(100 * s.isna().mean(), 2), "unique": int(s.nunique())}
        if pd.api.types.is_numeric_dtype(s):
            info.update({"min": float(s.min()), "max": float(s.max())})
        else:
            info["examples"] = s.dropna().astype(str).value_counts().head(max_examples).index.tolist()
        lines.append(f"{col}: {json.dumps(info)}")
    return "\n".join(lines)


def draft_data_dictionary(df: pd.DataFrame, table: str, client=None, model: str | None = None) -> str:
    prompt = DICTIONARY_PROMPT.format(table=table, profile=column_profile(df))
    if client is None and not os.environ.get("ANTHROPIC_API_KEY"):
        return prompt
    import anthropic

    client = client or anthropic.Anthropic()
    msg = client.messages.create(
        model=model or os.environ.get("RECSYS_LLM_MODEL", DEFAULT_MODEL),
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text
