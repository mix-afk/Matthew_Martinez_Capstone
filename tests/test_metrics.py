import pytest

from recsys import evaluate as E


def test_average_precision_matches_hand_calculation():
    assert E.average_precision_at_k({1, 2}, [1, 3, 2], k=12) == pytest.approx((1 + 2 / 3) / 2)


def test_average_precision_is_zero_without_hits():
    assert E.average_precision_at_k({9}, [1, 2, 3]) == 0.0


def test_average_precision_ignores_duplicate_predictions():
    assert E.average_precision_at_k({1}, [1, 1, 1]) == 1.0


def test_recall_and_ndcg_perfect_list():
    assert E.recall_at_k({1, 2}, [1, 2, 3]) == 1.0
    assert E.ndcg_at_k({1, 2}, [1, 2, 3]) == pytest.approx(1.0)


def test_summarize_uses_fallback_for_missing_customers():
    out = E.summarize({1: {5}, 2: {6}}, {1: [5]}, k=12, fallback=[6])
    assert out["map@12"] == pytest.approx(1.0)
    assert out["customers"] == 2
