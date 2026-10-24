"""Shared data loading, feature building and metrics (all features use information <= t, within day)."""
from pathlib import Path
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
D, R = P / "data_derived", P / "results"

GAS = ["g_H2O", "g_CO2", "g_NH3"]
TARGETS = ["AMP", "Pz"]
# manipulated / handle variables named in SI Table S1
HANDLES = ["TI-19", "FI-19", "TI-3", "FI-11", "FI-2", "TI-12", "TI-13", "FI-38"]


def load(days=range(1, 11)):
    m = pd.read_parquet(D / "minute_aligned.parquet")
    m = m[m.day.isin(list(days))].copy()
    win = pd.read_csv(D / "step_windows.csv", parse_dates=["begin", "end"])
    m["in_step"] = 0
    for _, w in win.iterrows():
        m.loc[(m.index >= w.begin) & (m.index < w.end), "in_step"] = 1
    m["minute_of_day"] = m.index.hour * 60 + m.index.minute
    return m, win


def process_cols(m):
    drop = set(TARGETS + GAS + ["g_O2", "day", "gas_count", "in_step", "minute_of_day"])
    return [c for c in m.columns if c not in drop]


def build_features(m, cols, windows=(10, 60), with_diff=True):
    """Per-day causal rolling features. Value at t uses rows <= t only."""
    out = []
    for d, s in m.groupby("day", sort=True):
        f = {}
        x = s[cols]
        for c in cols:
            f[c + "_last"] = x[c]
        for w in windows:
            r = x.rolling(w, min_periods=max(2, w // 2)).mean()
            for c in cols:
                f[f"{c}_m{w}"] = r[c]
        if with_diff:
            dd = x - x.shift(10)
            for c in cols:
                f[c + "_d10"] = dd[c]
        out.append(pd.DataFrame(f, index=s.index))
    return pd.concat(out)


def emission_features(m, cols=TARGETS + GAS):
    out = []
    for d, s in m.groupby("day", sort=True):
        f = {}
        for c in cols:
            x = s[c]
            f[c + "_last"] = x
            f[c + "_m10"] = x.rolling(10, min_periods=5).mean()
            f[c + "_m30"] = x.rolling(30, min_periods=15).mean()
            f[c + "_m60"] = x.rolling(60, min_periods=30).mean()
            f[c + "_q90_30"] = x.rolling(30, min_periods=15).quantile(0.9)
            f[c + "_sd30"] = x.rolling(30, min_periods=15).std()
        out.append(pd.DataFrame(f, index=s.index))
    return pd.concat(out)


def future_target(m, col, h, block=10):
    """Mean of col over (t+h-block, t+h], within day; NaN if incomplete."""
    out = []
    for d, s in m.groupby("day", sort=True):
        y = s[col].rolling(block, min_periods=block).mean().shift(-h)
        out.append(y)
    return pd.concat(out)


def metrics(y, p, ref=None):
    y, p = np.asarray(y), np.asarray(p)
    e = {"MAE": np.mean(np.abs(y - p)), "RMSE": np.sqrt(np.mean((y - p) ** 2)),
         "R2": 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2),
         "MAPE": 100 * np.mean(np.abs(y - p) / np.maximum(np.abs(y), 1.0)),
         "OPE": 100 * abs(y.sum() - p.sum()) / y.sum()}
    if ref is not None:
        e["skill_MAE"] = 1 - e["MAE"] / np.mean(np.abs(y - np.asarray(ref)))
    return e
