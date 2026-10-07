"""Explainability helpers: exact TreeSHAP values from LightGBM, global importance, PDP/ICE."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import to_model_frame


def shap_explanation(model, df: pd.DataFrame, cols: list[str], cats: dict, n: int = 5000, seed: int = 42):
    """LightGBM's pred_contrib returns exact TreeSHAP values (same algorithm as shap.TreeExplainer)."""
    import shap

    sample = df.sample(min(n, len(df)), random_state=seed)
    x = to_model_frame(sample, cols, cats)
    contrib = model.booster_.predict(x, pred_contrib=True)
    values, base = contrib[:, :-1], contrib[:, -1]
    data = x.copy()
    for c in data.columns:
        if isinstance(data[c].dtype, pd.CategoricalDtype):
            data[c] = data[c].cat.codes
    exp = shap.Explanation(values=values, base_values=base, data=data.to_numpy(dtype=float), feature_names=cols)
    return exp, sample


def shap_importance(exp) -> pd.DataFrame:
    imp = np.abs(exp.values).mean(axis=0)
    return pd.DataFrame({"feature": exp.feature_names, "mean_abs_shap": imp}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def gain_importance(model, cols: list[str]) -> pd.DataFrame:
    gain = model.booster_.feature_importance(importance_type="gain")
    return pd.DataFrame({"feature": cols, "gain": gain / gain.sum()}).sort_values("gain", ascending=False).reset_index(drop=True)


def local_explanation(model, row: pd.DataFrame, cols: list[str], cats: dict, top: int = 5) -> pd.DataFrame:
    x = to_model_frame(row, cols, cats)
    contrib = model.booster_.predict(x, pred_contrib=True)[0][:-1]
    out = pd.DataFrame({"feature": cols, "value": row[cols].iloc[0].to_numpy(), "shap": contrib})
    return out.reindex(out["shap"].abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)
