import numpy as np
import pandas as pd

from recsys import candidates as C


def test_popular_returns_most_sold_first(toy_tx):
    top = C.popular(toy_tx, n=3, weeks=4)
    assert top[0] == toy_tx["article_id"].value_counts().index[0]
    assert len(top) == 3


def test_repurchase_orders_by_recency(toy_tx):
    ref = toy_tx["t_dat"].max() + pd.Timedelta(days=1)
    rep = C.repurchase(toy_tx, np.array([0, 1]), max_n=5, ref_date=ref)
    first = rep[rep["cid"] == 0].sort_values("rep_rank")
    assert (first["rep_days_since"].diff().dropna() >= 0).all()


def test_interactions_matrix_shape(toy_tx):
    inter = C.Interactions(toy_tx, n_customers=30)
    assert inter.matrix.shape == (30, toy_tx["article_id"].nunique())
    assert inter.matrix.sum() == len(toy_tx)
