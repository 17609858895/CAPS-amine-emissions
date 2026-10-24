"""Step 14b: CAPS robustness analysis (section 8 items 1-4) from the 14_caps_robust_runs.py outputs.

(0) Reproduction: tag main_paths (09 rerun with per-window summaries) against results/psc/main.
(1) Pathway-bound confidence sets. Per step, AMP and Pz placebo windows are paired on the same origin
    (cross-fitted; each target keeps its own CAPS rule); e = (e_AMP, e_Pz) are level-scale window-mean errors
    [mg Nm-3]. 90% joint sets for (tau_AMP, tau_Pz), calibrated on the joint placebo distribution:
      RECT  max-|t| rectangle, T_p = max(|e_A|/s_A, |e_P|/s_P), s = placebo RMS (pre-specified; sensitivity: s = 0.9
            quantile of |e|, robust to the heavy AMP tail), c = conformal 0.9 quantile of T;
      ELL   Mahalanobis ellipse with placebo second-moment matrix S, r = conformal 0.9 quantile of sqrt(e' S^-1 e).
    Mapped through tau_V in [tau_A - tau_P/rho, tau_A] and tau_Aer in [min(0, tau_P/rho), max(0, tau_P/rho)]
    (rho = 0.484); the union over the set is the confidence set for the identified set. Sign p-values give the
    smallest level at which that union for tau_V (or tau_P, hence the sign of tau_Aer) excludes 0.
    Sensitivity 'rescaled': placebo relative errors e / mean(exp(cf)) times the step's counterfactual level.
(2) Pz LOD: raw readings < 1 mg Nm-3 set to LOD/2 or LOD (tags lod_half, lod_full) vs main; extra conservative
    stress test lod_floor3 (raw readings < 3 mg Nm-3, roughly a 3 x LOD quantification limit, set to 3).
(3) Clock: gas shifted -5 / +5 min (tags shift_m5, shift_p5) vs main.
(4) Transient pulse: placebo-conformal inference for log sub-window means (0-60, 60-120, 120 min-end) of the
    CAPS effect path and for the early-minus-late contrast; same estimator (weights and carryover rule are fitted
    on the whole window), so the placebo null calibrates exactly what is reported.
Outputs: results/psc/robust/*.csv
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import R

RHO, ALPHA = 0.484, 0.10
OUT = R / "psc" / "robust"; OUT.mkdir(parents=True, exist_ok=True)
SHORT = {2: "WW flow down", 3: "FG temp up", 4: "LS flow down", 5: "LS+FG flow down", 6: "LS temp up",
         7: "LS+WW temp up", 8: "Capture rate down"}


def rd(tag):
    return pd.read_csv(R / "psc" / tag / "step_effects_psc.csv"), pd.read_csv(R / "psc" / tag / "placebo_psc.csv")


def cq(x, a=ALPHA):
    """Conformal (1 - a) quantile with the (n + 1) finite-sample correction."""
    x = np.sort(np.asarray(x)); k = int(np.ceil((1 - a) * (len(x) + 1)))
    return np.inf if k > len(x) else x[k - 1]


def pconf(null, stat):
    return (1 + np.sum(np.asarray(null) >= stat)) / (1 + len(null))


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def sign_lab(lo, hi):
    return "up" if lo > 0 else ("down" if hi < 0 else "-")


# ---------------- (0) reproduction ----------------
E0, PL0 = rd("main"); E1, PL1 = rd("main_paths")
k = ["target", "day"]
J = E0[k + ["CAPS", "p_CAPS", "q_CAPS_BH"]].merge(E1[k + ["CAPS", "p_CAPS", "q_CAPS_BH"]], on=k, suffixes=("_09", "_14"))
rep = {"max_abs_dCAPS": float((J.CAPS_09 - J.CAPS_14).abs().max()), "max_abs_dp": float((J.p_CAPS_09 - J.p_CAPS_14).abs().max()),
       "n_placebo_09": len(PL0), "n_placebo_14": len(PL1)}
if len(PL0) == len(PL1):
    rep["max_abs_d_placebo_CAPS"] = float(np.abs(PL0.CAPS.values - PL1.CAPS.values).max())
lv_old = pd.read_csv(R / "psc" / "main" / "pathway_bounds_levels.csv").set_index("day")
for t in ["AMP", "Pz"]:
    rep[f"max_abs_dlevel_{t}"] = float(np.abs(E1[E1.target == t].set_index("day").lv_CAPS - lv_old[t]).max())
json.dump(rep, open(OUT / "reproduction_check.json", "w"), indent=1)
print("reproduction:", rep)


# ---------------- (1) pathway-bound confidence sets ----------------
def pathway_cs(E, PL, variant="raw", scale="rms"):
    out = []
    for d in range(2, 9):
        ea = E[(E.target == "AMP") & (E.day == d)].iloc[0]; ep = E[(E.target == "Pz") & (E.day == d)].iloc[0]
        tA, tP = ea.lv_CAPS, ep.lv_CAPS; dV = tA - tP / RHO
        Jw = PL[(PL.target == "AMP") & (PL.step_day == d)][["origin", "lv_CAPS", "cfl_CAPS"]].merge(
            PL[(PL.target == "Pz") & (PL.step_day == d)][["origin", "lv_CAPS", "cfl_CAPS"]], on="origin", suffixes=("_A", "_P"))
        eA, eP = Jw.lv_CAPS_A.values, Jw.lv_CAPS_P.values
        if variant == "rescaled":
            eA = eA / Jw.cfl_CAPS_A.values * ea.cfl_CAPS; eP = eP / Jw.cfl_CAPS_P.values * ep.cfl_CAPS
        n = len(Jw)
        sA, sP = (rms(eA), rms(eP)) if scale == "rms" else (np.quantile(np.abs(eA), .9), np.quantile(np.abs(eP), .9))
        r_ = {"variant": variant, "rect_scale": scale, "day": d, "test": SHORT[d], "n_pairs": n, "corr_eA_eP": float(np.corrcoef(eA, eP)[0, 1]),
              "tau_AMP": tA, "tau_Pz": tP, "rms_eA": sA, "rms_eP": sP,
              "V_point_lo": min(tA, dV), "V_point_hi": max(tA, dV), "Aer_point_lo": min(0, tP / RHO), "Aer_point_hi": max(0, tP / RHO)}
        # RECT: simultaneous max-|t| rectangle
        T = np.maximum(np.abs(eA) / sA, np.abs(eP) / sP); c = cq(T)
        aL, aU, pL, pU = tA - c * sA, tA + c * sA, tP - c * sP, tP + c * sP
        r_.update({"rect_c": c, "rect_AMP_lo": aL, "rect_AMP_hi": aU, "rect_Pz_lo": pL, "rect_Pz_hi": pU,
                   "rect_V_lo": min(aL, aL - pU / RHO), "rect_V_hi": max(aU, aU - pL / RHO),
                   "rect_Aer_lo": min(0, pL / RHO), "rect_Aer_hi": max(0, pU / RHO)})
        cV = max(min(tA / sA, dV / (sA + sP / RHO)), min(-tA / sA, -dV / (sA + sP / RHO)))
        r_["rect_p_Vsign"] = pconf(T, cV) if cV > 0 else 1.0
        r_["rect_p_Aersign"] = pconf(T, abs(tP) / sP)
        # ELL: Mahalanobis ellipse with placebo second-moment matrix
        X = np.column_stack([eA, eP]); S = X.T @ X / n; Si = np.linalg.inv(S)
        dm = np.sqrt(np.einsum("ij,jk,ik->i", X, Si, X)); rr = cq(dm)
        sd = lambda u: float(np.sqrt(u @ S @ u))
        ua, uv, up = np.array([1.0, 0]), np.array([1.0, -1 / RHO]), np.array([0, 1.0])
        r_.update({"ell_r": rr, "ell_V_lo": min(tA - rr * sd(ua), dV - rr * sd(uv)), "ell_V_hi": max(tA + rr * sd(ua), dV + rr * sd(uv)),
                   "ell_Pz_lo": tP - rr * sd(up), "ell_Pz_hi": tP + rr * sd(up),
                   "ell_Aer_lo": min(0, (tP - rr * sd(up)) / RHO), "ell_Aer_hi": max(0, (tP + rr * sd(up)) / RHO)})
        rV = max(min(tA / sd(ua), dV / sd(uv)), min(-tA / sd(ua), -dV / sd(uv)))
        r_["ell_p_Vsign"] = pconf(dm, rV) if rV > 0 else 1.0
        r_["ell_p_Aersign"] = pconf(dm, abs(tP) / sd(up))
        for m_ in ["rect", "ell"]:
            r_[f"{m_}_V_sign90"] = sign_lab(r_[f"{m_}_V_lo"], r_[f"{m_}_V_hi"])
            r_[f"{m_}_Aer_sign90"] = sign_lab(r_[f"{m_}_Pz_lo"], r_[f"{m_}_Pz_hi"]).replace("up", "up*").replace("down", "down*")
        out.append(r_)
    return pd.DataFrame(out)


CS = pd.concat([pathway_cs(E1, PL1, "raw"), pathway_cs(E1, PL1, "raw", "q90"), pathway_cs(E1, PL1, "rescaled")], ignore_index=True)
CS.to_csv(OUT / "pathway_confidence_sets.csv", index=False)
print("\n(1) pathway confidence sets (mg Nm-3; * aerosol sign set touches 0 only in the rho_a -> inf limit)")
print(CS[["variant", "rect_scale", "day", "test", "corr_eA_eP", "V_point_lo", "V_point_hi", "rect_V_lo", "rect_V_hi", "rect_p_Vsign",
          "rect_Aer_lo", "rect_Aer_hi", "rect_p_Aersign", "ell_V_lo", "ell_V_hi", "ell_p_Vsign", "ell_p_Aersign",
          "rect_V_sign90", "rect_Aer_sign90", "ell_V_sign90", "ell_Aer_sign90"]].round(3).to_string())

# ---------------- (2) Pz LOD ----------------
rows = []
for tag in ["main", "lod_half", "lod_full", "lod_floor3"]:
    Ex, PLx = (E1, PL1) if tag == "main" else rd(tag)
    Ez = pd.concat([E1[E1.target == "AMP"], Ex[Ex.target == "Pz"]]).sort_values(["target", "day"])
    Ez["q14"] = bh(Ez.p_CAPS.values)
    a = PL1[PL1.target == "Pz"][["step_day", "origin", "CAPS"]].merge(PLx[PLx.target == "Pz"][["step_day", "origin", "CAPS"]],
                                                                    on=["step_day", "origin"], how="outer", suffixes=("_m", "_x"))
    info = json.load(open(R / "psc" / tag / "run_info.json")) if tag != "main" else {}
    cs = pathway_cs(Ez, pd.concat([PL1[PL1.target == "AMP"], PLx[PLx.target == "Pz"]]), "raw").set_index("day")
    for _, r in Ez[Ez.target == "Pz"].iterrows():
        rows.append({"tag": tag, "day": r.day, "test": SHORT[r.day], "pct_CAPS": r.pct_change_CAPS, "p_CAPS": r.p_CAPS, "q14_BH": r.q14,
                     "lv_CAPS": r.lv_CAPS, "carry_rule": "FWD" if r.p_carry < 0.10 else "FULL",
                     "rect_V_lo": cs.loc[r.day, "rect_V_lo"], "rect_V_hi": cs.loc[r.day, "rect_V_hi"],
                     "rect_p_Vsign": cs.loc[r.day, "rect_p_Vsign"], "rect_p_Aersign": cs.loc[r.day, "rect_p_Aersign"],
                     "max_abs_d_placebo_CAPS_vs_main": float(np.nanmax(np.abs(a.CAPS_m - a.CAPS_x))),
                     "placebo_windows_unmatched": int(a.isna().any(axis=1).sum()),
                     "raw_below_lod": info.get("raw_below_lod"), "y_Pz_rows_changed": info.get("y_Pz_rows_changed"),
                     "y_Pz_rows_changed_unblocked": info.get("y_Pz_rows_changed_unblocked")})
LOD = pd.DataFrame(rows); LOD.to_csv(OUT / "lod_sensitivity.csv", index=False)
print("\n(2) Pz LOD sensitivity\n", LOD.round(4).to_string())

# ---------------- (3) clock offset ----------------
rows, prow = [], []
for tag, sh in [("shift_m5", -5), ("main_paths", 0), ("shift_p5", 5)]:
    Ex, PLx = (E1, PL1) if sh == 0 else rd(tag)
    cal = pd.read_csv(R / "psc" / tag / "placebo_calibration.csv") if (R / "psc" / tag / "placebo_calibration.csv").exists() else None
    for _, r in Ex.iterrows():
        rows.append({"shift_min": sh, "target": r.target, "day": r.day, "test": SHORT[r.day], "pct_CAPS": r.pct_change_CAPS,
                     "p_CAPS": r.p_CAPS, "q_BH": r.q_CAPS_BH, "hw90": r.hw90_CAPS, "carry_rule": "FWD" if r.p_carry < 0.10 else "FULL",
                     "lv_CAPS": r.lv_CAPS})
    cs = pathway_cs(Ex, PLx, "raw")
    for _, r in cs.iterrows():
        prow.append({"shift_min": sh, "day": r.day, "test": r.test, "V_point_lo": r.V_point_lo, "V_point_hi": r.V_point_hi,
                     "rect_V_lo": r.rect_V_lo, "rect_V_hi": r.rect_V_hi, "rect_p_Vsign": r.rect_p_Vsign, "rect_p_Aersign": r.rect_p_Aersign,
                     "rect_V_sign90": r.rect_V_sign90, "rect_Aer_sign90": r.rect_Aer_sign90})
    if cal is not None:
        print(f"calibration shift {sh}:\n", cal[cal.estimator == "CAPS"].to_string())
CLK = pd.DataFrame(rows); CLK.to_csv(OUT / "clock_offset_effects.csv", index=False)
CLP = pd.DataFrame(prow); CLP.to_csv(OUT / "clock_offset_pathway.csv", index=False)
wide = CLK.pivot_table(index=["target", "day"], columns="shift_min", values=["pct_CAPS", "p_CAPS", "q_BH"]).round(3)
print("\n(3) clock offset\n", wide.to_string())
print(CLK.pivot_table(index=["target", "day"], columns="shift_min", values="carry_rule", aggfunc="first").to_string())
print(CLP.round(3).to_string())

# ---------------- (4) transient pulse: sub-window inference ----------------
SW = [("0-60", "s0"), ("60-120", "s1"), ("120-end", "s2"), ("(0-60)-(120-end)", "c02")]


def subwindows(E, PL):
    rows = []
    for tgt in ["AMP", "Pz"]:
        for d in range(2, 9):
            r = E[(E.target == tgt) & (E.day == d)].iloc[0]; Pn = PL[(PL.target == tgt) & (PL.step_day == d)]
            for lab, key in SW:
                if key == "c02":
                    stat, null = r.s0_CAPS - r.s2_CAPS, (Pn.s0_CAPS - Pn.s2_CAPS).values
                else:
                    stat, null = r[f"{key}_CAPS"], Pn[f"{key}_CAPS"].values
                hw = np.quantile(np.abs(null), 0.90)
                rows.append({"target": tgt, "day": d, "test": SHORT[d], "subwindow": lab, "effect_log": stat,
                             "pct": 100 * (np.exp(stat) - 1), "pct_lo90": 100 * (np.exp(stat - hw) - 1), "pct_hi90": 100 * (np.exp(stat + hw) - 1),
                             "hw90_log": hw, "placebo_rms_log": rms(null), "p": pconf(np.abs(null), abs(stat)), "n_placebo": len(null)})
    T = pd.DataFrame(rows)
    T["q_BH14"] = T.groupby("subwindow").p.transform(bh)
    return T


SWT = subwindows(E1, PL1)
SWT.to_csv(OUT / "subwindow_inference.csv", index=False)
tv = pd.read_csv(R / "psc" / "main" / "transient_vs_sustained_pct.csv")
chk = SWT[SWT.subwindow.isin(["0-60", "60-120", "120-end"])].pivot_table(index=["target", "day"], columns="subwindow", values="pct").round(0)
print("\n(4) sub-window check vs transient_vs_sustained_pct.csv: max |diff| =",
      float(np.abs(chk[["0-60", "60-120", "120-end"]].values - tv.set_index(["target", "day"])[["0-60min", "60-120", "120-end"]].values).max()))
print(SWT.pivot_table(index=["target", "day"], columns="subwindow", values=["pct", "p", "q_BH14"]).round(3).to_string())
print(SWT.groupby(["target", "subwindow"]).placebo_rms_log.mean().round(3).to_string())

# sub-windows under the clock offsets (the 0-60 min pulse is the quantity most exposed to timing error)
SWC = pd.concat([subwindows(*((E1, PL1) if sh == 0 else rd(tag))).assign(shift_min=sh)
                 for tag, sh in [("shift_m5", -5), ("main_paths", 0), ("shift_p5", 5)]], ignore_index=True)
SWC.to_csv(OUT / "subwindow_inference_clock.csv", index=False)
print("\n(4b) sub-windows under clock offsets (Pz, then AMP)")
print(SWC[SWC.target == "Pz"].pivot_table(index=["day", "subwindow"], columns="shift_min", values=["pct", "p", "q_BH14"]).round(3).to_string())
print(SWC[SWC.target == "AMP"].pivot_table(index=["day", "subwindow"], columns="shift_min", values=["pct", "p", "q_BH14"]).round(3).to_string())
