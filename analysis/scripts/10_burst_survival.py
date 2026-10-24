"""Step 10 (trial): aerosol-burst early warning as a recurrent time-to-event problem.

Burst minute: raw GasMET2 AMP > 1.5 x trailing-30-min median AND Pz > 1.5 x its median AND AMP excess > 15.
Onset: burst minute preceded by >= 5 burst-free minutes. Landmarks: burst-free minutes (stride 2 min).
Time-to-next-onset discretised in 5-min intervals up to 60 min; censored at segment end / missing gas.
Models (LODO over days with bursts):
  RENEWAL : discrete hazard depending only on time-since-last-onset (+ interval index)   [renewal process]
  HAZ_GBM : discrete-time hazard LightGBM with renewal terms + process/emission state    [survival ML]
  CLS_30  : direct classifier P(onset within 30 min) with the same features              [non-survival]
Metrics: AUC and Brier of P(T <= 15/30 min) on held-out day (censored-before-horizon rows dropped).
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss
sys.path.insert(0, str(Path(__file__).parent))
from common import load, process_cols, build_features, R, D

warnings.filterwarnings("ignore")
OUT = R / "survival"; OUT.mkdir(parents=True, exist_ok=True)
m, win = load(range(1, 9))
m["seg"] = np.where(m.day <= 3, "A", "B")
g = pd.read_csv(D / "gasmet2_raw.csv.gz", parse_dates=["timestamp"]).sort_values("timestamp").set_index("timestamp")
g.columns = ["day", "H2O", "CO2", "NH3", "AMP", "Pz", "O2"]
bA, bP = g.AMP.rolling("30min").median(), g.Pz.rolling("30min").median()
burst = ((g.AMP > 1.5 * bA) & (g.Pz > 1.5 * bP) & (g.AMP - bA > 15)).astype(float)
mb = burst.resample("1min", closed="right", label="right").max().reindex(m.index).fillna(0).astype(bool)
prev = mb.shift(1, fill_value=False).rolling(5, min_periods=1).max().astype(bool)
onset = mb & ~prev
m["burst"], m["onset"] = mb.values, onset.values

# renewal + state features (causal)
feat = []
for seg, s in m.groupby("seg"):
    t_last, tsl, cnt = None, [], []
    on = s.onset.values
    for i in range(len(s)):
        if on[i]:
            t_last = i
        tsl.append(np.nan if t_last is None else i - t_last)
    f = pd.DataFrame(index=s.index)
    f["tsl"] = tsl
    f["n_on_60"] = s.onset.rolling(60, min_periods=1).sum()
    f["n_on_180"] = s.onset.rolling(180, min_periods=1).sum()
    f["burst_frac_60"] = s.burst.rolling(60, min_periods=1).mean()
    feat.append(f)
F = pd.concat(feat)
pc = ["TI-19", "FI-19", "TI-3", "FI-11", "FI-2", "TI-12", "TI-13", "FI-38", "TI-35", "TI-4", "TI-34", "TI-33",
      "TI-2", "CO2-3", "O2-3", "FI-4", "TI-5"]
S = build_features(m, pc, windows=(30,), with_diff=True)
for c in ["AMP", "Pz", "g_H2O", "g_CO2"]:
    S[c + "_m30"] = m.groupby("seg")[c].transform(lambda s: s.rolling(30, min_periods=10).median())
    S[c + "_sd30"] = m.groupby("seg")[c].transform(lambda s: s.rolling(30, min_periods=10).std())
X = pd.concat([F, S], axis=1)

# time to next onset (minutes) and censoring, per segment
T, Ev = [], []
for seg, s in m.groupby("seg"):
    on = s.onset.values; valid = s.AMP.notna().values; n = len(s)
    nxt, cen = np.full(n, np.nan), np.zeros(n, bool)
    nxt_on, nxt_bad = n, n
    for i in range(n - 1, -1, -1):
        if not valid[i]:
            nxt_bad = i
        nxt[i] = min(nxt_on, nxt_bad) - i
        cen[i] = nxt_bad < nxt_on
        if on[i]:
            nxt_on = i
    T.append(pd.Series(nxt, index=s.index)); Ev.append(pd.Series(~cen, index=s.index))
m["T"], m["E"] = pd.concat(T), pd.concat(Ev)

BIN, NB = 5, 12
lm = m[(~m.burst) & m.AMP.notna() & X.tsl.notna()].iloc[::2]
lm = lm[lm.day.isin([1, 4, 5, 6, 7, 8])]


def expand(idx):
    rows = []
    for t in idx:
        T_, E_ = m.at[t, "T"], m.at[t, "E"]
        for j in range(NB):
            lo, hi = j * BIN, (j + 1) * BIN
            if T_ <= lo:
                break
            ev = int(E_ and T_ <= hi)
            if (not E_) and T_ <= hi:
                break  # censored inside interval
            rows.append((t, j, ev))
            if ev:
                break
    df = pd.DataFrame(rows, columns=["t", "j", "y"])
    return pd.concat([df.reset_index(drop=True), X.loc[df.t].reset_index(drop=True)], axis=1)


def cum_inc(model, cols, idx, horizon):
    nb = horizon // BIN
    Xi = X.loc[idx]
    surv = np.ones(len(idx))
    for j in range(nb):
        Z = Xi.assign(j=j)[cols]
        surv *= 1 - model.predict_proba(Z)[:, 1]
    return 1 - surv


LGB = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.7, n_jobs=16, verbose=-1)
REN = ["j", "tsl", "n_on_60", "n_on_180", "burst_frac_60"]
ALL = ["j"] + list(X.columns)
out = []
for d in [4, 5, 6, 7, 8]:
    tr, te = lm[lm.day != d].index, lm[lm.day == d].index
    PP = expand(tr)
    mods = {"RENEWAL": (lgb.LGBMClassifier(**LGB).fit(PP[REN], PP.y), REN),
            "HAZ_GBM": (lgb.LGBMClassifier(**LGB).fit(PP[ALL], PP.y), ALL)}
    for H in (15, 30):
        yte = (m.loc[te, "T"] <= H) & m.loc[te, "E"]
        keep = (m.loc[te, "T"] > H) | m.loc[te, "E"]       # drop censored-before-horizon
        ytr = ((m.loc[tr, "T"] <= H) & m.loc[tr, "E"]); ktr = (m.loc[tr, "T"] > H) | m.loc[tr, "E"]
        cls = lgb.LGBMClassifier(**LGB).fit(X.loc[tr][ktr.values], ytr[ktr.values])
        preds = {k: cum_inc(md, c, te, H) for k, (md, c) in mods.items()}
        preds["CLS_direct"] = cls.predict_proba(X.loc[te])[:, 1]
        for k, p in preds.items():
            yy, pp = yte[keep.values].astype(int), p[keep.values]
            out.append({"day": d, "H": H, "model": k, "auc": roc_auc_score(yy, pp), "brier": brier_score_loss(yy, pp),
                        "event_rate": yy.mean(), "n": len(yy)})
        print(d, H, {r["model"]: round(r["auc"], 3) for r in out[-3:]}, flush=True)
res = pd.DataFrame(out); res.to_csv(OUT / "burst_survival_lodo.csv", index=False)
print(res.groupby(["H", "model"])[["auc", "brier", "event_rate"]].mean().round(3).to_string())
