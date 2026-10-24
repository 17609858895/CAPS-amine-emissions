"""Step 16: semi-synthetic signal-injection benchmark (known ground truth) for CAPS and its ablations.

Real step-free windows (L = 300 min; pre 90, gap 60, post 90 min) of log 10-min mean AMP / Pz receive an
injected step: effect(k) = delta * (1 - exp(-k / tau_on)) for k in [0, L) (tau_on = 20 min); with carryover,
the effect decays after the window as delta_end * exp(-(k - L) / tau_off) (tau_off = 90 min), reaching the
gap and the post window. Covariates are unchanged. True window-mean effect tau = mean effect over [0, L).
Estimators (learners refit leave-one-day-out on clean step-free data, as in step 9):
  BA_pre, LIN_BR, ML_fwd, ML_bwd, FULL (stack of all six), FWD (stack of BA_pre + ML_fwd), CAPS (carryover rule).
Ablations: FULL = CAPS without carryover test; FWD = CAPS without backward/bridge learners;
ML_fwd = no stacking. Inference uses the clean placebo null pooled over held-out days.
Outputs bias, RMSE, coverage of the 90% placebo-conformal interval and power (p <= 0.05).
"""
import sys, warnings, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).parent))
from common import load, R

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("--target", default="AMP"); ap.add_argument("--jobs", type=int, default=4)
A = ap.parse_args()
OUT = R / "semisynthetic"; OUT.mkdir(parents=True, exist_ok=True)
PRE, GAP, POST, STRIDE, L = 90, 60, 90, 10, 300
TAU_ON, TAU_OFF = 20.0, 90.0
DELTAS = [0.1, 0.2, 0.3, 0.5]
EXO = ["CO2-3", "O2-3"]
LGB = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, n_jobs=A.jobs, verbose=-1)
LEARNERS = ["BA_pre", "BA_post", "LIN_BR", "ML_fwd", "ML_bwd", "ML_BR"]
FWDL = ["BA_pre", "ML_fwd"]

m, win = load(range(1, 9))
m["seg"] = np.where(m.day <= 3, "A", "B")
tgt = A.target
m["y"] = np.log(m.groupby("seg")[tgt].transform(lambda s: s.rolling(10, min_periods=8).mean()).clip(lower=0.5))
ex30 = m.groupby("seg")[EXO].transform(lambda s: s.rolling(30, min_periods=10).mean())
blocked = pd.Series(False, index=m.index)
for _, w in win.iterrows():
    blocked |= (m.index >= w.begin) & (m.index < w.end + pd.Timedelta(f"{GAP}min"))
blocked |= (m.day == 1) & (m.index < win.loc[win.day == 1, "end"].iloc[0])
m["blocked"] = blocked.values


def get_windows(sdf):
    y = sdf.y.values; blk = sdf.blocked.values; days = sdf.day.values; ex = ex30.loc[sdf.index].values
    out = []
    for i in range(0, len(sdf), STRIDE):
        a, b = i - PRE, i + L + GAP + POST
        if a < 0 or b > len(sdf) or blk[a:b].any():
            continue
        pre, mid, post = y[a:i], y[i:i + L], y[i + L + GAP:b]
        if max(np.isnan(pre).mean(), np.isnan(post).mean(), np.isnan(mid).mean()) > .2:
            continue
        out.append({"origin": sdf.index[i], "days": set(days[a:b]), "day0": days[i], "pre": pre, "mid": mid,
                    "post": post, "ex_pre": np.nanmean(ex[a:i], 0), "ex_post": np.nanmean(ex[i + L + GAP:b], 0),
                    "ex_mid": ex[i:i + L]})
    return out


def slope(v):
    ok = ~np.isnan(v)
    return np.polyfit(np.arange(len(v))[ok], v[ok], 1)[0]


def rows(W, thin=5):
    k = np.arange(L); out = []
    for w in W:
        mp, mq = np.nanmean(w["pre"]), np.nanmean(w["post"])
        tpre, tpost = -PRE / 2, L + GAP + POST / 2
        lam = (k - tpre) / (tpost - tpre)
        df = pd.DataFrame({"k": k, "kr": L - k, "kL": k / L, "mp": mp, "mq": mq, "dq": mq - mp,
                           "sp": slope(w["pre"]), "sq": slope(w["post"]), "y": w["mid"], "bridge": mp + lam * (mq - mp)})
        for j, c in enumerate(EXO):
            exb = w["ex_pre"][j] + lam * (w["ex_post"][j] - w["ex_pre"][j])
            df["dxf_" + c] = w["ex_mid"][:, j] - w["ex_pre"][j]
            df["dxr_" + c] = w["ex_mid"][:, j] - w["ex_post"][j]
            df["dxb_" + c] = w["ex_mid"][:, j] - exb
        out.append(df.iloc[::thin])
    return pd.concat(out, ignore_index=True)


F_FWD = ["k", "mp", "sp"] + ["dxf_" + c for c in EXO]
F_BWD = ["kr", "mq", "sq"] + ["dxr_" + c for c in EXO]
F_BR = ["kL", "mp", "mq", "dq", "sp", "sq"] + ["dxb_" + c for c in EXO]


def fit(TR):
    return {"ML_fwd": lgb.LGBMRegressor(**LGB).fit(TR[F_FWD], TR.y - TR.mp),
            "ML_bwd": lgb.LGBMRegressor(**LGB).fit(TR[F_BWD], TR.y - TR.mq),
            "ML_BR": lgb.LGBMRegressor(**LGB).fit(TR[F_BR], TR.y - TR.bridge)}


def effects(w, M):
    Rw = rows([w], thin=1)
    cf = {"BA_pre": Rw.mp.values, "BA_post": Rw.mq.values, "LIN_BR": Rw.bridge.values,
          "ML_fwd": Rw.mp.values + M["ML_fwd"].predict(Rw[F_FWD]),
          "ML_bwd": Rw.mq.values + M["ML_bwd"].predict(Rw[F_BWD]),
          "ML_BR": Rw.bridge.values + M["ML_BR"].predict(Rw[F_BR])}
    return {k: float(np.nanmean(Rw.y.values - v)) for k, v in cf.items()}


def simplex(E):
    n = E.shape[1]; S = E.T @ E / len(E)
    r = minimize(lambda w: w @ S @ w, np.ones(n) / n, jac=lambda w: 2 * S @ w, method="SLSQP",
                 bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return r.x


def inject(w, delta, carry):
    k = np.arange(L)
    prof = delta * (1 - np.exp(-k / TAU_ON))
    v = dict(w); v["mid"] = w["mid"] + prof
    if carry:
        kk = L + GAP + np.arange(POST)
        v["post"] = w["post"] + prof[-1] * np.exp(-(kk - L) / TAU_OFF)
    return v, float(prof.mean())


W = get_windows(m[m.seg == "A"]) + get_windows(m[m.seg == "B"])
days = sorted({w["day0"] for w in W})
models = {j: fit(rows([w for w in W if j not in w["days"]])) for j in days}
print(tgt, "windows", len(W), "folds", len(days), flush=True)
clean = pd.DataFrame([{"day": w["day0"], "i": n, **effects(w, models[w["day0"]])} for n, w in enumerate(W)])
WF = {j: simplex(clean[clean.day != j][LEARNERS].values) for j in days}
WW = {j: simplex(clean[clean.day != j][FWDL].values) for j in days}
clean["FULL"] = [r[LEARNERS].values @ WF[r.day] for _, r in clean.iterrows()]
clean["FWD"] = [r[FWDL].values @ WW[r.day] for _, r in clean.iterrows()]
nullD = np.abs((clean.FULL - clean.FWD).values)
pc = lambda d: (1 + np.sum(nullD >= abs(d))) / (1 + len(nullD))
clean["CAPS"] = [r.FWD if pc(r.FULL - r.FWD) < 0.10 else r.FULL for _, r in clean.iterrows()]
EST = ["BA_pre", "LIN_BR", "ML_fwd", "ML_bwd", "FULL", "FWD", "CAPS"]
NULL = {e: np.abs(clean[e].values) for e in EST}
Q90 = {e: np.quantile(NULL[e], 0.90) for e in EST}

out = []
for carry in [False, True]:
    for delta in [0.0] + DELTAS:
        for n, w in enumerate(W):
            v, tau = inject(w, delta, carry) if delta > 0 else (w, 0.0)
            e = effects(v, models[w["day0"]]) if delta > 0 else clean.iloc[n][LEARNERS].to_dict()
            j = w["day0"]
            e["FULL"] = np.array([e[k] for k in LEARNERS]) @ WF[j]
            e["FWD"] = np.array([e[k] for k in FWDL]) @ WW[j]
            e["CAPS"] = e["FWD"] if pc(e["FULL"] - e["FWD"]) < 0.10 else e["FULL"]
            for est in EST:
                pval = (1 + np.sum(NULL[est] >= abs(e[est]))) / (1 + len(NULL[est]))
                out.append({"target": tgt, "carryover": carry, "delta": delta, "tau_true": tau, "day": j, "window": n,
                            "estimator": est, "est": e[est], "err": e[est] - tau,
                            "covered": abs(e[est] - tau) <= Q90[est], "reject05": pval <= 0.05})
        print(tgt, carry, delta, flush=True)

res = pd.DataFrame(out)
res.to_csv(OUT / f"injection_raw_{tgt}.csv.gz", index=False)
summ = res.groupby(["target", "carryover", "delta", "estimator"]).agg(
    bias=("err", "mean"), rmse=("err", lambda s: np.sqrt(np.mean(s ** 2))), coverage90=("covered", "mean"),
    power05=("reject05", "mean"), n=("err", "size")).reset_index()
summ.to_csv(OUT / f"injection_summary_{tgt}.csv", index=False)
print(summ.round(3).to_string())
