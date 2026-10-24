"""Step 4b (trial): lean multi-horizon forecasting benchmark + conformal calibration.

Target: 10-min mean AMP / Pz ending at t+h (h = 10, 30, 60, 120 min), within day.
Splits: LODO over days 1..8 (each held-out day = an unseen designed intervention);
        STRESS: train days 1..8, test post-shutdown days 9, 10.
Base learners:
  persistence : last 10-min mean
  GBM_level   : LightGBM on log1p(y), absolute-level features (Jablonka-2023-like direct GBDT)
  GBM_resid   : LightGBM on log(y/persistence), same features (persistence-anchored hybrid)
  LF_resid    : level-free hybrid: log(y/persistence) from features expressed as deviations from their
                own trailing 60-min mean / 10-min differences / emission log-ratios (shift-invariant)
  STACK       : NNLS weights fitted on LODO out-of-fold predictions of the OTHER days
Uncertainty (on STACK, log space): split-conformal half-width from OOF residuals of other days, and
adaptive conformal inference (ACI, Gibbs & Candes 2021) updated with an h-minute feedback delay.
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import nnls
sys.path.insert(0, str(Path(__file__).parent))
from common import load, process_cols, build_features, emission_features, future_target, metrics, R

warnings.filterwarnings("ignore")
OUT = R / "forecast"; OUT.mkdir(parents=True, exist_ok=True)
m, win = load()
pc = process_cols(m)
XL = pd.concat([build_features(m, pc), emission_features(m)], axis=1)
# level-free features
LF = {}
for c in pc:
    LF[c + "_dev60"] = XL[c + "_last"] - XL[c + "_m60"]
    LF[c + "_dev10"] = XL[c + "_m10"] - XL[c + "_m60"]
    LF[c + "_d10"] = XL[c + "_d10"]
for c in ["AMP", "Pz", "g_H2O", "g_CO2", "g_NH3"]:
    base = XL[c + "_m10"].clip(lower=0.5)
    for s in ["_last", "_m30", "_m60", "_q90_30"]:
        LF[c + s + "_lr"] = np.log(XL[c + s].clip(lower=0.5) / base)
    LF[c + "_cv30"] = XL[c + "_sd30"] / base
XF = pd.DataFrame(LF, index=XL.index)
XF["ratio_PzAMP"] = np.log(XL["Pz_m10"].clip(lower=.5) / XL["AMP_m10"].clip(lower=.5))
LGB = dict(n_estimators=300, learning_rate=0.04, num_leaves=31, min_child_samples=80, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.5, reg_lambda=2.0, n_jobs=6, verbose=-1)
BASES = ["persistence", "GBM_level", "GBM_resid", "LF_resid"]


def base_preds(tr, te, y, pers):
    lp_tr, lp_te = np.log(pers[tr].clip(lower=.5)), np.log(pers[te].clip(lower=.5))
    ly = np.log(y[tr].clip(lower=.5))
    out = {"persistence": pers[te].values}
    g = lgb.LGBMRegressor(**LGB).fit(XL[tr], np.log1p(y[tr])); out["GBM_level"] = np.expm1(g.predict(XL[te]))
    g = lgb.LGBMRegressor(**LGB).fit(XL[tr], ly - lp_tr); out["GBM_resid"] = np.exp(lp_te + g.predict(XL[te]))
    g = lgb.LGBMRegressor(**LGB).fit(XF[tr], ly - lp_tr); out["LF_resid"] = np.exp(lp_te + g.predict(XF[te]))
    return out


rows, cover = [], []
PARTS = OUT / "parts"; PARTS.mkdir(exist_ok=True)
for tgt in ["AMP", "Pz"]:
    for h in [10, 30, 60, 120]:
        pf = PARTS / f"{tgt}_{h}.pkl"
        if pf.exists():  # resumable: reuse finished (target, horizon) blocks
            z = pd.read_pickle(pf); rows += z["rows"]; cover += z["cover"]; continue
        n_r, n_c = len(rows), len(cover)
        y = future_target(m, tgt, h); pers = XL[f"{tgt}_m10"]
        ok = y.notna() & pers.notna() & XL[f"{tgt}_m60"].notna()
        idx = m.index[ok]
        P = {}  # LODO out-of-fold predictions for days 1..8
        for d in range(1, 9):
            tr, te = ok & (m.day != d) & (m.day <= 8), ok & (m.day == d)
            P[d] = pd.DataFrame(base_preds(tr, te, y, pers), index=m.index[te])
        S = {}
        for d in (9, 10):
            tr, te = ok & (m.day <= 8), ok & (m.day == d)
            S[d] = pd.DataFrame(base_preds(tr, te, y, pers), index=m.index[te])
        oof = pd.concat(P.values())
        for split, dd, src in [("LODO", range(1, 9), P), ("STRESS", (9, 10), S)]:
            for d in dd:
                other = oof[m.loc[oof.index, "day"] != d]
                w, _ = nnls(other[BASES].values, y[other.index].values)
                pr = src[d].copy(); pr["STACK"] = pr[BASES].values @ w
                yt = y[pr.index]
                st = m.loc[pr.index, "in_step"].values == 1
                for k in BASES + ["STACK"]:
                    e = metrics(yt, pr[k], ref=pr["persistence"])
                    es = metrics(yt[st], pr[k][st], ref=pr["persistence"][st]) if st.sum() > 30 else {}
                    rows.append({"target": tgt, "h": h, "split": split, "day": d, "model": k, **e,
                                 "step_skill_MAE": es.get("skill_MAE", np.nan)})
                # conformal on STACK (log space), calibrated on other days' OOF stacked residuals
                oth = other[BASES].values @ w
                res_oof = np.abs(np.log(np.clip(oth, .5, None)) - np.log(y[other.index].clip(lower=.5)))
                err = np.log(np.clip(pr["STACK"].values, .5, None)) - np.log(yt.clip(lower=.5).values)
                qst = np.quantile(res_oof, 0.9)
                a_t, qs, cov = 0.1, [], []
                for t in range(len(err)):
                    q = np.quantile(res_oof, np.clip(1 - a_t, 0, 1))
                    qs.append(q); cov.append(abs(err[t]) <= q)
                    if t - h >= 0:
                        a_t += 0.005 * (0.1 - float(abs(err[t - h]) > qs[t - h]))
                cover.append({"target": tgt, "h": h, "split": split, "day": d, "cov_static": np.mean(np.abs(err) <= qst),
                              "cov_ACI": np.mean(cov), "hw_static": qst, "hw_ACI_median": np.median(qs),
                              "weights": dict(zip(BASES, np.round(w, 3)))})
        pd.to_pickle({"rows": rows[n_r:], "cover": cover[n_c:]}, pf)
        print(tgt, h, pd.DataFrame(rows).query("target==@tgt and h==@h").groupby(["split", "model"]).skill_MAE.mean().round(3).to_dict(), flush=True)

res, cv = pd.DataFrame(rows), pd.DataFrame(cover)
res.to_csv(OUT / "forecast_lean_by_day.csv", index=False); cv.to_csv(OUT / "conformal_coverage.csv", index=False)
summ = res.groupby(["split", "target", "h", "model"])[["MAE", "R2", "MAPE", "skill_MAE", "step_skill_MAE"]].mean().round(3)
summ.to_csv(OUT / "forecast_lean_summary.csv")
print(summ.to_string())
print(cv.groupby(["split", "target", "h"])[["cov_static", "cov_ACI", "hw_static", "hw_ACI_median"]].mean().round(3).to_string())
