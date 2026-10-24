"""Step 18: first-order-plus-dead-time (FOPDT) fits to the CAPS dynamic effect paths of the robust core effects.

effect(t) = A * (1 - exp(-(t - theta) / tau)) for t > theta, else 0 (log scale; t in min).
For Pz the first hour contains a transient excursion that is not significant after FDR (5.7.4); fits use
t >= 60 min with theta >= 0 free, so they describe the sustained response only. Uncertainty: moving-block
bootstrap of fit residuals (block 30 min, 1000 resamples). Descriptive dynamic signatures, not structural
process models.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
sys.path.insert(0, str(Path(__file__).parent))
from common import R

OUT = R / "dynamics"; OUT.mkdir(parents=True, exist_ok=True)
DP = pd.read_csv(R / "psc" / "main" / "dynamic_paths_psc.csv")
rng = np.random.default_rng(7)


def fopdt(t, A, tau, theta):
    return np.where(t > theta, A * (1 - np.exp(-(t - theta) / tau)), 0.0)


rows, curves = [], []
for tgt, day, tmin in [("AMP", 7, 0), ("Pz", 7, 60), ("Pz", 6, 60), ("AMP", 5, 0), ("Pz", 5, 60)]:
    q = DP[(DP.target == tgt) & (DP.day == day)].sort_values("k")
    t, y = q.k.values.astype(float), q.effect.values
    sel = t >= tmin
    p0 = [y[sel][-60:].mean(), 60.0, 10.0]
    bounds = ([-3, 5, 0], [3, 600, 200])
    par, _ = curve_fit(fopdt, t[sel], y[sel], p0=p0, bounds=bounds, maxfev=20000)
    fit = fopdt(t[sel], *par); res = y[sel] - fit
    B, blk, n = [], 30, sel.sum()
    for _ in range(1000):
        idx = np.concatenate([np.arange(s, s + blk) for s in rng.integers(0, n - blk, n // blk + 1)])[:n]
        try:
            pb, _ = curve_fit(fopdt, t[sel], fit + res[idx], p0=par, bounds=bounds, maxfev=20000)
            B.append(pb)
        except RuntimeError:
            continue
    B = np.array(B)
    lo, hi = np.percentile(B, [5, 95], axis=0)
    rows.append({"target": tgt, "day": day, "fit_from_min": tmin, "A_log": par[0], "A_pct": 100 * (np.exp(par[0]) - 1),
                 "tau_min": par[1], "tau_lo": lo[1], "tau_hi": hi[1], "theta_min": par[2], "theta_lo": lo[2],
                 "theta_hi": hi[2], "t63_min": par[2] + par[1], "n_boot": len(B),
                 "r2": 1 - np.sum(res ** 2) / np.sum((y[sel] - y[sel].mean()) ** 2)})
    curves.append(pd.DataFrame({"target": tgt, "day": day, "k": t, "effect": y, "fit": fopdt(t, *par)}))
T = pd.DataFrame(rows); T.to_csv(OUT / "fopdt_fits.csv", index=False)
pd.concat(curves).to_csv(OUT / "fopdt_curves.csv", index=False)
print(T.round(2).to_string())
