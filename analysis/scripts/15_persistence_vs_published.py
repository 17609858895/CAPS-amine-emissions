"""Step 15: persistence forecasts under the published (Jablonka et al. 2023) preprocessing and split.

Approximate replication of their Methods/SI: days 1-8 concatenated in order (pre-shutdown data only),
linear detrending through the endpoints, z-score outlier removal (|z| > 3 -> interpolated), exponential
window smoothing (16 min), downsampling to 2 min, min-max scaling on the first 50% (train),
test = last 25%. Metrics follow their SI Note 7 (MAE in scaled units, MAPE, OPE) for horizons of
2 min, 1 h and 2 h. Persistence: y_hat(t+h) = y(t). Their reported GBDT values (SI Table S4) are
copied for comparison only. Variants show how much of the score comes from smoothing.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import load, R

OUT = R / "persistence_published"; OUT.mkdir(parents=True, exist_ok=True)
PUBLISHED = {("AMP", 2): (2.4, 0.38), ("AMP", 60): (11.0, 0.34), ("AMP", 120): (9.5, 3.8),
             ("Pz", 2): (4.3, 2.0), ("Pz", 60): (23.0, 17.0), ("Pz", 120): (21.0, 10.0)}
m, _ = load(range(1, 9))


def preprocess(x, smooth=True, detrend=True, outlier=True):
    x = x.copy().interpolate(limit_direction="both")
    if detrend:
        n = len(x); x = x - (x.iloc[0] + (x.iloc[-1] - x.iloc[0]) * np.arange(n) / (n - 1))
    if outlier:
        z = (x - x.mean()) / x.std(); x[z.abs() > 3] = np.nan; x = x.interpolate(limit_direction="both")
    if smooth:
        x = x.ewm(span=16).mean()
    x = x.iloc[::2].reset_index(drop=True)  # 2-min sampling
    n = len(x); tr = x.iloc[: n // 2]
    return (x - tr.min()) / (tr.max() - tr.min())


rows = []
for tgt in ["AMP", "Pz"]:
    raw = m[tgt].reset_index(drop=True)  # concatenated minutes of days 1-8 (calendar gaps removed)
    for name, kw in [("published_preprocessing", {}), ("no_smoothing", {"smooth": False}),
                     ("raw_2min", {"smooth": False, "detrend": False, "outlier": False})]:
        y = preprocess(raw, **kw)
        n = len(y); te0 = int(n * 0.75)
        for h_min in [2, 60, 120]:
            h = h_min // 2
            idx = np.arange(max(te0, h), n)
            yt, yp = y.values[idx], y.values[idx - h]
            ok = np.abs(yt) > 1e-6
            mape = 100 * np.mean(np.abs(yt[ok] - yp[ok]) / np.abs(yt[ok]))
            mae = np.mean(np.abs(yt - yp)); ope = 100 * abs(yt.sum() - yp.sum()) / abs(yt.sum())
            pub = PUBLISHED[(tgt, h_min)] if name == "published_preprocessing" else (np.nan, np.nan)
            rows.append({"target": tgt, "variant": name, "horizon_min": h_min, "persistence_MAPE": mape,
                         "persistence_MAE_scaled": mae, "persistence_OPE": ope,
                         "published_GBDT_MAPE": pub[0], "published_GBDT_OPE": pub[1], "n_test": len(idx)})
res = pd.DataFrame(rows)
res.to_csv(OUT / "persistence_vs_published.csv", index=False)
print(res.round(2).to_string())
