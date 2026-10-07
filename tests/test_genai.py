import pandas as pd

from recsys import genai as G


def test_template_reason_for_repurchase():
    facts = G.reason_facts(pd.Series({"rep_count": 2}), pd.Series({"product_type_name": "Sweater", "perceived_colour_master_name": "Black"}))
    assert "before" in G.template_reason(facts)


def test_guardrail_blocks_personal_traits():
    assert not G.is_safe("Perfect for women your age")
    assert G.is_safe("A new colour of something you already love")


def test_llm_reason_falls_back_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    text, source = G.llm_reason("Tee", ["it is one of this week's best sellers"])
    assert source == "template"
    assert text == "One of this week's best sellers"
