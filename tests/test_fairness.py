import pandas as pd
import pytest

from recsys import fairness as F


def test_disparate_impact_is_ratio_to_best_group():
    per = pd.DataFrame({"cid": [1, 2, 3, 4], "ap": [1, 0, 1, 1], "recall": [1, 0, 1, 1], "ndcg": [1, 0, 1, 1], "hit": [1, 0, 1, 1]})
    cust = pd.DataFrame({"cid": [1, 2, 3, 4], "age_band": ["a", "a", "b", "b"]})
    g = F.group_quality(per, cust, min_size=1).set_index("age_band")
    assert g.loc["a", "disparate_impact"] == pytest.approx(0.5)
    assert bool(g.loc["a", "di_flag"]) is True


def test_equalized_odds_gaps():
    df = pd.DataFrame({"cid": [1, 1, 2, 2], "label": [1, 0, 1, 0], "top": [1, 0, 0, 0]})
    cust = pd.DataFrame({"cid": [1, 2], "age_band": ["a", "b"]})
    _, s = F.equalized_odds(df, cust, "top")
    assert s["tpr_gap"] == pytest.approx(1.0)
    assert s["fpr_gap"] == pytest.approx(0.0)


def test_rerank_with_zero_lambda_keeps_order():
    df = pd.DataFrame({"cid": [1, 1, 1], "score": [0.9, 0.5, 0.1], "pop": [1, 50, 3]})
    s = F.rerank_long_tail(df, "score", "pop", 0.0)
    assert list(s.rank(ascending=False)) == [1.0, 2.0, 3.0]


def test_small_groups_are_not_flagged():
    per = pd.DataFrame({"cid": [1, 2, 3], "ap": [1, 1, 0], "recall": [1, 1, 0], "ndcg": [1, 1, 0], "hit": [1, 1, 0]})
    cust = pd.DataFrame({"cid": [1, 2, 3], "age_band": ["a", "a", "b"]})
    g = F.group_quality(per, cust, min_size=2).set_index("age_band")
    assert bool(g.loc["b", "small_group"]) is True
    assert bool(g.loc["b", "di_flag"]) is False
