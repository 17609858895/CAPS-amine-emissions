"""Step 14: CAPS robustness runs (section 8 items: pathway confidence sets, Pz LOD, clock offset, transient pulse).

Same estimator as 09_placebo_stacked_cf.py (identical learners, day cross-fitting, other-day weight fitting and
carryover rule; outputs keep the 09 file layout), plus:
  * per-window summaries for every placebo window and for the real step, for PSC / FWD / CAPS:
      lv_*          level-scale window-mean error (placebo) or effect (step): mean(exp(y) - exp(cf))  [mg Nm-3]
      cfl_*         mean counterfactual level mean(exp(cf)) over the same minutes
      s0_*,s1_*,s2_* log-scale sub-window means over k in [0,60), [60,120), [120,L)
    placebo rows carry their origin, so AMP and Pz errors can be paired on the same windows;
  * --pz-lod {none,half,full}: minute Pz rebuilt from raw GasMET2 readings with readings < LOD replaced by
    LOD/2 or LOD (02_align_minutes.py aggregation otherwise unchanged; checked to reproduce it exactly);
  * --shift S: gas series moved S minutes relative to process data and step windows,
    gas_new(t) = gas_old(t + S)  (S > 0 corrects an assumed S-min analyser/transport lag).
Raw data and earlier result folders are not touched; outputs go to results/psc/<tag>/.
"""
import sys, warnings, argparse, pickle, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).parent))
from common import load, R, D

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("--gap", type=int, default=60)
ap.add_argument("--exo", default="CO2-3,O2-3"); ap.add_argument("--tag", default="main_paths")
ap.add_argument("--target", default=None); ap.add_argument("--day", type=int, default=None)
ap.add_argument("--jobs", type=int, default=2); ap.add_argument("--merge", action="store_true")
ap.add_argument("--pz-lod", dest="pz_lod", default="none", choices=["none", "half", "full"])
ap.add_argument("--lod", type=float, default=1.0); ap.add_argument("--shift", type=int, default=0)
A = ap.parse_args()
PRE, GAP, POST, STRIDE = 90, A.gap, 90, 10
EXO = A.exo.split(",") if A.exo else []
OUT = R / "psc" / A.tag; OUT.mkdir(parents=True, exist_ok=True)
LGB = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, n_jobs=A.jobs, verbose=-1)
LEARNERS = ["BA_pre", "BA_post", "LIN_BR", "ML_fwd", "ML_bwd", "ML_BR"]
FWDL = ["BA_pre", "ML_fwd"]
GASC = ["AMP", "Pz", "g_H2O", "g_CO2", "g_NH3", "g_O2"]
SUBW = [(0, 60), (60, 120), (120, None)]

m, win = load(range(1, 9))
m["seg"] = np.where(m.day <= 3, "A", "B")
info = {"tag": A.tag, "pz_lod": A.pz_lod, "lod": A.lod, "shift_min": A.shift, "gap": GAP, "exo": EXO}
pz0 = m.Pz.copy()
if A.pz_lod != "none":
    g = pd.read_csv(D / "gasmet2_raw.csv.gz", parse_dates=["timestamp"]).rename(
        columns={"2-Amino-2-methylpropanol C4H11NO": "AMP", "Piperazine C4H10N2": "Pz"})
    g = g[g.day <= 8]

    def minute_pz(gg):
        out = []
        for day, s in gg.groupby("day"):
            s = s.set_index("timestamp")[["AMP", "Pz"]]
            gm = s.resample("1min", closed="right", label="right").mean()
            gc = s.resample("1min", closed="right", label="right").count().min(axis=1)
            gm.loc[gc < 2, :] = np.nan
            out.append(gm.Pz.reindex(m.index[m.day == day]))
        return pd.concat(out)

    assert np.allclose(minute_pz(g).values, m.Pz.values, equal_nan=True), "minute rebuild != 02_align_minutes"
    below = g.Pz < A.lod
    info["raw_below_lod"] = int(below.sum())
    info["raw_below_lod_by_day"] = {int(k): int(v) for k, v in g[below].groupby("day").size().items()}
    info["raw_below_lod_span"] = [str(g[below].timestamp.min()), str(g[below].timestamp.max())]
    g.loc[below, "Pz"] = A.lod / 2 if A.pz_lod == "half" else A.lod
    m["Pz"] = minute_pz(g).values
    info["minutes_changed"] = int((~np.isclose(m.Pz.values, pz0.values, equal_nan=True)).sum())
if A.shift:
    m[GASC] = m[GASC].reindex(m.index + pd.Timedelta(minutes=A.shift)).values
for c in ["AMP", "Pz"]:
    m["y_" + c] = np.log(m.groupby("seg")[c].transform(lambda s: s.rolling(10, min_periods=8).mean()).clip(lower=0.5))
ex30 = m.groupby("seg")[EXO].transform(lambda s: s.rolling(30, min_periods=10).mean()) if EXO else None
blocked = pd.Series(False, index=m.index)
for _, w in win.iterrows():
    blocked |= (m.index >= w.begin) & (m.index < w.end + pd.Timedelta(f"{GAP}min"))
blocked |= (m.day == 1) & (m.index < win.loc[win.day == 1, "end"].iloc[0])
m["blocked"] = blocked.values
if A.pz_lod != "none":
    y0 = np.log(pd.Series(pz0.values, index=m.index).groupby(m.seg).transform(
        lambda s: s.rolling(10, min_periods=8).mean()).clip(lower=0.5))
    chg = ~np.isclose(m.y_Pz.values, y0.values, equal_nan=True)
    info["y_Pz_rows_changed"] = int(chg.sum())
    info["y_Pz_rows_changed_unblocked"] = int((chg & ~m.blocked.values).sum())


def get_windows(sdf, L, origins, tgt, require_free):
    y = sdf["y_" + tgt].values; blk = sdf.blocked.values; days = sdf.day.values
    ex = ex30.loc[sdf.index].values if EXO else np.zeros((len(sdf), 0))
    out = []
    for i in origins:
        a, b = i - PRE, i + L + GAP + POST
        if a < 0 or b > len(sdf) or (require_free and blk[a:b].any()):
            continue
        pre, mid, post = y[a:i], y[i:i + L], y[i + L + GAP:b]
        if max(np.isnan(pre).mean(), np.isnan(post).mean(), np.isnan(mid).mean()) > .2:
            continue
        out.append({"origin": sdf.index[i], "days": set(days[a:b]), "day0": days[i], "pre": pre, "mid": mid, "post": post,
                    "ex_pre": np.nanmean(ex[a:i], 0), "ex_post": np.nanmean(ex[i + L + GAP:b], 0), "ex_mid": ex[i:i + L]})
    return out


def slope(v):
    ok = ~np.isnan(v)
    return np.polyfit(np.arange(len(v))[ok], v[ok], 1)[0]


def rows(W, L, thin=5):
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
        df["oid"] = str(w["origin"]); df["day0"] = w["day0"]
        out.append(df.iloc[::thin])
    return pd.concat(out, ignore_index=True)


F_FWD = ["k", "mp", "sp"] + ["dxf_" + c for c in EXO]
F_BWD = ["kr", "mq", "sq"] + ["dxr_" + c for c in EXO]
F_BR = ["kL", "mp", "mq", "dq", "sp", "sq"] + ["dxb_" + c for c in EXO]


def fit_learners(TR):
    return {"ML_fwd": lgb.LGBMRegressor(**LGB).fit(TR[F_FWD], TR.y - TR.mp),
            "ML_bwd": lgb.LGBMRegressor(**LGB).fit(TR[F_BWD], TR.y - TR.mq),
            "ML_BR": lgb.LGBMRegressor(**LGB).fit(TR[F_BR], TR.y - TR.bridge)}


def cf_paths(Rw, M):
    return {"BA_pre": Rw.mp.values, "BA_post": Rw.mq.values, "LIN_BR": Rw.bridge.values,
            "ML_fwd": Rw.mp.values + M["ML_fwd"].predict(Rw[F_FWD]),
            "ML_bwd": Rw.mq.values + M["ML_bwd"].predict(Rw[F_BWD]),
            "ML_BR": Rw.bridge.values + M["ML_BR"].predict(Rw[F_BR])}


def simplex_weights(E):
    """E: (n_placebo, n_learners) window-mean errors. minimise mean (E w)^2, w>=0, sum w = 1."""
    n = E.shape[1]; S = E.T @ E / len(E)
    r = minimize(lambda w: w @ S @ w, np.ones(n) / n, jac=lambda w: 2 * S @ w, method="SLSQP",
                 bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return r.x


def summaries(y, cf, name):
    """Level window mean, counterfactual level and log sub-window means of y - cf (k = minute since origin)."""
    y = np.asarray(y, float); k = np.arange(len(y)); ok = ~np.isnan(y); e = y - cf
    out = {"lv_" + name: float(np.nanmean(np.exp(y) - np.exp(cf))), "cfl_" + name: float(np.mean(np.exp(cf[ok])))}
    for i, (a, b) in enumerate(SUBW):
        sel = (k >= a) & (k < (len(y) if b is None else b))
        out[f"s{i}_{name}"] = float(np.nanmean(e[sel]))
    return out


effects, placebo_all, paths, wts = [], [], [], []
TGTS = [] if A.merge else ([A.target] if A.target else ["AMP", "Pz"])
DAYS = [A.day] if A.day else list(range(2, 9))
PARTS = OUT / "parts"; PARTS.mkdir(exist_ok=True)
for tgt in TGTS:
    for d in DAYS:
        w = win[win.day == d].iloc[0]
        L = int((w.end - w.begin).total_seconds() // 60)
        segW = {seg: get_windows(sdf, L, range(0, len(sdf), STRIDE), tgt, True) for seg, sdf in m.groupby("seg")}
        allW = segW["A"] + segW["B"]
        # cross-fitted placebo errors for every held-out day j (models trained without day j)
        perr, models = [], {}
        for j in range(1, 9):
            teW = [x for x in allW if x["day0"] == j]
            if not teW and j != d:
                continue
            M = fit_learners(rows([x for x in allW if j not in x["days"]], L))
            models[j] = M
            for x in teW:
                Rw = rows([x], L, thin=1); cps = cf_paths(Rw, M)
                perr.append({"day": j, "origin": x["origin"], **{k: float(np.nanmean(Rw.y - v)) for k, v in cps.items()},
                             "_y": Rw.y.values, "_cf": cps})
        P = pd.DataFrame(perr)
        # Other-day weight fitting excludes day-j errors, but their underlying models can include day-j records.
        Js = sorted(set(P.day) | {d})
        Wfull = {j: simplex_weights(P[P.day != j][LEARNERS].values) for j in Js}
        Wfwd = {j: simplex_weights(P[P.day != j][FWDL].values) for j in Js}
        wd, wf = Wfull[d], Wfwd[d]
        wts.append({"target": tgt, "day": d, **dict(zip(LEARNERS, wd))})
        P["PSC"] = [r_[LEARNERS].values.astype(float) @ Wfull[r_.day] for _, r_ in P.iterrows()]
        P["FWD"] = [r_[FWDL].values.astype(float) @ Wfwd[r_.day] for _, r_ in P.iterrows()]
        P["DELTA"] = P.PSC - P.FWD
        nullD = np.abs(P.DELTA.values)
        P["p_carry"] = [(1 + np.sum(nullD >= abs(v))) / (1 + len(nullD)) for v in P.DELTA]
        P["CAPS"] = np.where(P.p_carry < 0.10, P.FWD, P.PSC)
        for _, r in P.iterrows():
            cf_full = np.column_stack([r["_cf"][k] for k in LEARNERS]) @ Wfull[r.day]
            cf_fwd = np.column_stack([r["_cf"][k] for k in FWDL]) @ Wfwd[r.day]
            cf_c = cf_fwd if r.p_carry < 0.10 else cf_full
            placebo_all.append({"target": tgt, "step_day": d, "L": L, "day": r.day, "origin": r.origin,
                                **{k: r[k] for k in LEARNERS + ["PSC", "FWD", "DELTA", "CAPS"]},
                                **summaries(r["_y"], cf_full, "PSC"), **summaries(r["_y"], cf_fwd, "FWD"),
                                **summaries(r["_y"], cf_c, "CAPS")})
        # real step on day d
        sdf = m[m.seg == ("A" if d <= 3 else "B")]
        i0 = int(np.where(sdf.index == w.begin)[0][0])
        Rw = rows(get_windows(sdf, L, [i0], tgt, False), L, thin=1)
        cps = cf_paths(Rw, models[d])
        e = {k: float(np.nanmean(Rw.y - v)) for k, v in cps.items()}
        cf_psc = np.column_stack([cps[k] for k in LEARNERS]) @ wd
        e["PSC"] = float(np.nanmean(Rw.y - cf_psc))
        cf_fwd = np.column_stack([cps[k] for k in FWDL]) @ wf
        e["FWD"] = float(np.nanmean(Rw.y - cf_fwd))
        e["DELTA"] = e["PSC"] - e["FWD"]
        e["p_carry"] = (1 + np.sum(nullD >= abs(e["DELTA"]))) / (1 + len(nullD))
        e["CAPS"] = e["FWD"] if e["p_carry"] < 0.10 else e["PSC"]
        cf_caps = cf_fwd if e["p_carry"] < 0.10 else cf_psc
        wsel = (FWDL, wf) if e["p_carry"] < 0.10 else (LEARNERS, wd)
        e.update({**summaries(Rw.y.values, cf_psc, "PSC"), **summaries(Rw.y.values, cf_fwd, "FWD"),
                  **summaries(Rw.y.values, cf_caps, "CAPS")})
        # dynamic path + pointwise placebo band of the stacked estimator
        Pd = [Rw_y - np.column_stack([cf[k] for k in wsel[0]]) @ wsel[1] for Rw_y, cf in zip(P["_y"], P["_cf"])]
        band = np.nanquantile(np.vstack(Pd), [0.05, 0.95], axis=0)
        paths.append(pd.DataFrame({"target": tgt, "day": d, "k": Rw.k, "y": Rw.y, "cf": cf_caps,
                                   "effect": Rw.y - cf_caps, "band_lo": band[0], "band_hi": band[1],
                                   "cf_fwd": cf_fwd, "cf_full": cf_psc}))
        effects.append({"target": tgt, "day": d, "test": w.test, "L": L, "n_placebo": len(P), **e})
        print(tgt, d, "w=", np.round(wd, 2), {k: round(v, 3) for k, v in e.items()}, flush=True)

if A.target or A.day:
    for (tgt, d) in {(e["target"], e["day"]) for e in effects}:
        pickle.dump({"effects": [e for e in effects if e["target"] == tgt and e["day"] == d],
                     "placebo": [r for r in placebo_all if r["target"] == tgt and r["step_day"] == d],
                     "paths": [q for q in paths if q.target.iloc[0] == tgt and q.day.iloc[0] == d],
                     "wts": [w_ for w_ in wts if w_["target"] == tgt and w_["day"] == d]},
                    open(PARTS / f"{tgt}_{d}.pkl", "wb"))
    sys.exit(0)
if A.merge:
    for f in sorted(PARTS.glob("*.pkl")):
        z = pickle.load(open(f, "rb"))
        effects += z["effects"]; placebo_all += z["placebo"]; paths += z["paths"]; wts += z["wts"]
    json.dump(info, open(OUT / "run_info.json", "w"), indent=1)
E, PL = pd.DataFrame(effects).sort_values(["target", "day"]).reset_index(drop=True), pd.DataFrame(placebo_all)
cols = LEARNERS + ["PSC", "FWD", "CAPS"]
rms = PL.groupby("target")[cols].agg(lambda s: np.sqrt(np.mean(s ** 2)))
print("placebo RMS of window-mean error:\n", rms.round(3).to_string())
for i, r in E.iterrows():
    Pn = PL[(PL.target == r.target) & (PL.step_day == r.day)]
    for c in cols:
        null = np.abs(Pn[c].values)
        E.loc[i, "p_" + c] = (1 + np.sum(null >= abs(r[c]))) / (1 + len(null))
        E.loc[i, "hw90_" + c] = np.quantile(null, 0.90)
# Benjamini-Hochberg across the (target x step) tests present in this run
for c in ["PSC", "CAPS"]:
    p = E["p_" + c].values; o = np.argsort(p); q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    E["q_%s_BH" % c] = np.minimum(q, 1)
    E["pct_change_" + c] = 100 * (np.exp(E[c]) - 1)
E.to_csv(OUT / "step_effects_psc.csv", index=False); PL.to_csv(OUT / "placebo_psc.csv", index=False)
pd.concat(paths).to_csv(OUT / "dynamic_paths_psc.csv", index=False); pd.DataFrame(wts).to_csv(OUT / "stack_weights.csv", index=False)
rms.to_csv(OUT / "placebo_rms.csv")
print(json.dumps(info))
print(E[["target", "day", "test", "PSC", "p_PSC", "FWD", "p_FWD", "DELTA", "p_carry", "CAPS", "pct_change_CAPS", "p_CAPS", "q_CAPS_BH", "hw90_CAPS"]].round(3).to_string())
