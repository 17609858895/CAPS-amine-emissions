"""Step 11 (trial): do model-based what-if analyses reproduce the designed interventions?

For each step day d (2..8), models are trained WITHOUT day d. The what-if adds the observed step
magnitude of the manipulated handle(s) (in-step mean minus pre-step mean) to the handle features on
day d's 90-min pre-step rows, holding all other inputs fixed (Jablonka 2023 Note S11 assumptions).
  WI_SA     : 10-min-ahead LightGBM forecaster with emission history + SA feature set
              (TI-19, FI-19, FI-11, TI-3, TI-12, TI-13, TI-35), historical-forecast mode
  WI_SOFT   : process-only soft sensor on all process channels (no emission history)
  WI_HANDLE : handles + exogenous (CO2-3, O2-3) only ("total-effect" regression)
Reference: carryover-aware placebo-stacked counterfactual (CAPS) window-mean effect from step 9
(column kept as "PSC" for brevity).
Outputs sign agreement, Spearman correlation and magnitude ratio of what-if vs PSC.
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent))
from common import load, process_cols, build_features, emission_features, future_target, R

warnings.filterwarnings("ignore")
OUT = R / "whatif"; OUT.mkdir(parents=True, exist_ok=True)
m, win = load(range(1, 9))
pc = process_cols(m)
HANDLE = {2: ["FI-19"], 3: ["TI-3"], 4: ["FI-11"], 5: ["FI-11", "FI-2"], 6: ["TI-12", "TI-13"],
          7: ["TI-12", "TI-13", "TI-19"], 8: ["FI-38"]}
SAF = ["TI-19", "FI-19", "FI-11", "TI-3", "TI-12", "TI-13", "TI-35"]
LGB = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=80, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.6, reg_lambda=2.0, n_jobs=6, verbose=-1)
allhandles = sorted({h for v in HANDLE.values() for h in v} | {"FI-38", "FI-2"})
X_sa = pd.concat([build_features(m, SAF), emission_features(m)], axis=1)
X_soft = build_features(m, pc)
X_hd = build_features(m, allhandles + ["CO2-3", "O2-3"])
psc = pd.read_csv(R / "psc" / "main" / "step_effects_psc.csv")

rows = []
for tgt in ["AMP", "Pz"]:
    y10 = future_target(m, tgt, 10)
    ynow = np.log(m.groupby("day")[tgt].transform(lambda s: s.rolling(10, min_periods=8).mean()).clip(lower=.5))
    for d in range(2, 9):
        w = win[win.day == d].iloc[0]
        pre = (m.index >= w.begin - pd.Timedelta("90min")) & (m.index < w.begin)
        stp = (m.index >= w.begin + pd.Timedelta("60min")) & (m.index < w.end)
        delta = {h: m.loc[stp, h].mean() - m.loc[pre, h].mean() for h in HANDLE[d]}
        res = {"target": tgt, "day": d, "test": w.test, **{"delta_" + k: v for k, v in delta.items()}}
        for name, X, yy, log1p in [("WI_SA", X_sa, y10, True), ("WI_SOFT", X_soft, ynow, False), ("WI_HANDLE", X_hd, ynow, False)]:
            ok = yy.notna() & X.notna().mean(1).gt(.8)
            tr = ok & (m.day != d)
            g = lgb.LGBMRegressor(**LGB).fit(X[tr], np.log1p(yy[tr]) if log1p else yy[tr])
            base = X[pre & ok.values]
            Xw = base.copy()
            for h, dv in delta.items():
                for c in [c for c in X.columns if c.rsplit("_", 1)[0] == h]:
                    if not c.endswith("_d10"):
                        Xw[c] = Xw[c] + dv
            res[name] = float(np.mean(g.predict(Xw) - g.predict(base)))
        r = psc[(psc.target == tgt) & (psc.day == d)].iloc[0]
        res["PSC"], res["p_PSC"] = r.CAPS, r.p_CAPS  # reference = carryover-aware stacked estimate
        rows.append(res); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)

R_ = pd.DataFrame(rows); R_.to_csv(OUT / "whatif_vs_psc.csv", index=False)
summ = []
for name in ["WI_SA", "WI_SOFT", "WI_HANDLE"]:
    sig = R_.p_PSC < 0.1
    summ.append({"method": name, "sign_agree_all": np.mean(np.sign(R_[name]) == np.sign(R_.PSC)),
                 "sign_agree_sigPSC": np.mean(np.sign(R_[name][sig]) == np.sign(R_.PSC[sig])),
                 "spearman": spearmanr(R_[name], R_.PSC).correlation,
                 "median_abs_ratio": np.median(np.abs(R_[name]) / np.abs(R_.PSC))})
S = pd.DataFrame(summ); S.to_csv(OUT / "whatif_summary.csv", index=False)
print(S.round(3).to_string())
