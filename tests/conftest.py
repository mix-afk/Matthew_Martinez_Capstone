import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_tx():
    rng = np.random.default_rng(0)
    rows = []
    start = pd.Timestamp("2020-09-01")
    for cid in range(30):
        for _ in range(6):
            day = start + pd.Timedelta(days=int(rng.integers(0, 21)))
            rows.append((day, cid, int(rng.choice([101, 102, 103, 104, 105], p=[0.4, 0.25, 0.15, 0.1, 0.1])), 0.02, 1))
    t = pd.DataFrame(rows, columns=["t_dat", "cid", "article_id", "price", "online"])
    t["week"] = ((pd.Timestamp("2020-09-22") - t["t_dat"]).dt.days // 7).astype("int16")
    return t
