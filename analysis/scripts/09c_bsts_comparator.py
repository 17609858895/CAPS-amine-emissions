"""Step 9c (trial): CausalImpact-style structural time-series comparator (nearest established method).

At 10-min resolution: y = log 10-min mean emission; pre-period = previous 360 min (step-free);
model = local level + regression on exogenous CO2-3, O2-3 (statsmodels UnobservedComponents, MLE);
forecast the L-minute window with observed exogenous values; effect = window-mean(y - forecast).
Model-based 90% interval for the window mean is obtained by simulating forecast paths (n=300).
Placebo windows (step-free) give (i) RMS precision and (ii) the empirical false-positive rate of the
model-based interval, i.e. whether CausalImpact-type intervals are calibrated on this plant.
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
sys.path.insert(0, str(Path(__file__).parent))
from common import load, R

warnings.filterwarnings("ignore")
OUT = R / "psc" / "bsts"; OUT.mkdir(parents=True, exist_ok=True)
PRE, STRIDE = 360, 30
EXO = ["CO2-3", "O2-3"]
m, win = load(range(1, 9))
m["seg"] = np.where(m.day <= 3, "A", "B")
blocked = pd.Series(False, index=m.index)
for _, w in win.iterrows():
    blocked |= (m.index >= w.begin) & (m.index < w.end + pd.Timedelta("60min"))
blocked |= (m.day == 1) & (m.index < win.loc[win.day == 1, "end"].iloc[0])
m["blocked"] = blocked.values
rng = np.random.default_rng(0)


def fit_effect(y, X, n_pre, n_win):
    yp, Xp, Xf = y[:n_pre], X[:n_pre], X[n_pre:n_pre + n_win]
    Xm, Xs = Xp.mean(0), Xp.std(0) + 1e-9
    mod = sm.tsa.UnobservedComponents(yp, level="local level", exog=(Xp - Xm) / Xs)
    r = mod.fit(disp=False, maxiter=200)
    fc = r.get_forecast(steps=n_win, exog=(Xf - Xm) / Xs)
    obs = y[n_pre:n_pre + n_win]
    eff = np.nanmean(obs - fc.predicted_mean)
    # exact variance of the window-mean forecast error for a local-level model (beta treated as fixed):
    # Var = P_n + s2_level * (H+1)(2H+1)/(6H) + s2_irregular / H
    H = n_win
    pn = r.filtered_state_cov[0, 0, -1]
    pn_ = dict(zip(r.model.param_names, np.asarray(r.params)))
    s2e, s2l = pn_["sigma2.irregular"], pn_["sigma2.level"]
    sd = np.sqrt(pn + s2l * (H + 1) * (2 * H + 1) / (6 * H) + s2e / H)
    lo, hi = eff - 1.645 * sd, eff + 1.645 * sd
    return eff, lo, hi


rows = []
for tgt in ["AMP", "Pz"]:
    for d in range(2, 9):
        w = win[win.day == d].iloc[0]
        L = int((w.end - w.begin).total_seconds() // 60)
        nw, npre = L // 10, PRE // 10
        for seg, s in m.groupby("seg"):
            y10 = np.log(s[tgt].rolling(10, min_periods=8).mean().clip(lower=.5))
            ex = s[EXO].rolling(10, min_periods=5).mean()
            blk = s.blocked.values
            starts = list(range(PRE, len(s) - L, STRIDE))
            if seg == ("A" if d <= 3 else "B"):
                starts.append(int(np.where(s.index == w.begin)[0][0]))
            for i in starts:
                is_step = s.index[i] == w.begin
                if not is_step and blk[i - PRE:i + L].any():
                    continue
                if is_step and blk[i - PRE:i].any():
                    pass  # pre-period may contain earlier recovery only for early steps; keep
                sl = slice(i - PRE, i + L, 10)
                yy, XX = y10.values[sl], ex.values[sl]
                if np.isnan(yy).mean() > .2 or np.isnan(XX).any():
                    continue
                yy = pd.Series(yy).interpolate(limit_direction="both").values
                try:
                    eff, lo, hi = fit_effect(yy, XX, npre, nw)
                except Exception:
                    continue
                rows.append({"target": tgt, "step_day": d, "day": s.day.iloc[i], "origin": s.index[i], "is_step": is_step,
                             "effect": eff, "lo90": lo, "hi90": hi, "flag": not (lo <= 0 <= hi)})
        print(tgt, d, sum(r["step_day"] == d and r["target"] == tgt for r in rows), flush=True)

B = pd.DataFrame(rows); B.to_csv(OUT / "bsts_windows.csv", index=False)
pl = B[~B.is_step]
S = pl.groupby("target").agg(n=("effect", "size"), RMS=("effect", lambda s: np.sqrt(np.mean(s ** 2))),
                             FPR_model_interval=("flag", "mean"))
print(S.round(3).to_string())
print(B[B.is_step][["target", "step_day", "effect", "lo90", "hi90", "flag"]].round(3).to_string())
S.to_csv(OUT / "bsts_placebo_summary.csv")
