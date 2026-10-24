"""Step 3 (exploratory): Pz-tracer apportionment of AMP into aerosol-borne and vapour-slip parts.

Transfer of the EC-tracer / minimum-R^2 (MRS) idea from atmospheric aerosol science:
  AMP_aer(t) = Pz(t) / r_a      (Pz treated as a non-volatile aerosol tracer)
  AMP_vap(t) = AMP(t) - Pz(t) / r_a
r_a (Pz/AMP mass ratio of the aerosol-borne amine) is chosen by MRS: the value that minimises
R^2(AMP_vap, Pz). Physical anchor: solvent Pz/AMP mass ratio = 0.484 (lab ATR, all days).
Also reports the Pz/AMP slope of burst excursions (robust Theil-Sen on excess over a rolling median).
Outputs: data_derived/step_windows.csv, results/tracer/*.csv, results/explore/tracer_*.png
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import theilslopes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parents[1]
D, R = P / "data_derived", P / "results"
(R / "tracer").mkdir(parents=True, exist_ok=True)
m = pd.read_parquet(D / "minute_aligned.parquet")

# --- official step-test windows from lab-sheet headers ---
L = pd.read_csv(D / "lab_sheets_rawdump.csv.gz")
win = []
for d, s in L[L.sheet == "Lean Data"].groupby("day"):
    meta = {}
    for v in s["values"].head(6):
        nn = [x for x in json.loads(v) if x is not None]
        if nn and nn[0] in ("Date:", "v", "Begin:", "End:"):
            meta[nn[0]] = nn[1]
    date = m[m.day == d].index[0].normalize()
    win.append({"day": d, "begin": date + pd.Timedelta(meta["Begin:"]), "end": date + pd.Timedelta(meta["End:"])})
win = pd.DataFrame(win)
DESC = {1: "WW temperature up (TI-19)", 2: "WW flow down (FI-19)", 3: "Flue-gas temperature up (TI-3)",
        4: "Lean-solvent flow down (FI-11)", 5: "LS flow + FG flow down (FI-11, FI-2)",
        6: "Lean-solvent temperature up (TI-12)", 7: "LS + WW temperature up (TI-12, TI-19)",
        8: "Capture rate down (steam FI-38)", 9: "post-shutdown test A", 10: "post-shutdown test B"}
win["test"] = win.day.map(DESC)
win.to_csv(D / "step_windows.csv", index=False)
print(win)

def mrs_ratio(amp, pz, grid=np.linspace(0.15, 3.0, 571)):
    ok = amp.notna() & pz.notna()
    a, z = amp[ok].values, pz[ok].values
    r2 = []
    for r in grid:
        v = a - z / r
        c = np.corrcoef(v, z)[0, 1]
        r2.append(c * c)
    r2 = np.array(r2)
    return grid[r2.argmin()], grid, r2

rows = []
fig, ax = plt.subplots(2, 5, figsize=(20, 7))
for i, (d, s) in enumerate(m.groupby("day")):
    s = s.dropna(subset=["AMP", "Pz"])
    r_mrs, grid, r2 = mrs_ratio(s.AMP, s.Pz)
    exA = s.AMP - s.AMP.rolling(31, center=True, min_periods=10).median()
    exP = s.Pz - s.Pz.rolling(31, center=True, min_periods=10).median()
    big = exA > exA.quantile(0.95)
    ts = theilslopes(exP[big], exA[big]) if big.sum() > 10 else (np.nan,)
    vap = s.AMP - s.Pz / r_mrs
    rows.append({"day": d, "r_MRS": r_mrs, "minR2": r2.min(), "burst_slope_PzAMP": ts[0],
                 "baseline_ratio_median": (s.Pz / s.AMP).median(),
                 "vap_share_MRS": vap.clip(lower=0).sum() / s.AMP.sum(),
                 "vap_share_solvent": (s.AMP - s.Pz / 0.484).clip(lower=0).sum() / s.AMP.sum(),
                 "frac_vap_negative_MRS": float((vap < 0).mean())})
    a = ax.flat[i]
    a.plot(grid, r2); a.axvline(0.484, c="k", ls="--"); a.axvline(r_mrs, c="r")
    a.set_title(f"day{d} rMRS={r_mrs:.2f}")
plt.tight_layout(); plt.savefig(R / "explore" / "tracer_MRS_curves.png", dpi=70)
res = pd.DataFrame(rows)
# pooled core days 1-8
core = m[m.day <= 8].dropna(subset=["AMP", "Pz"])
r_pool, _, r2p = mrs_ratio(core.AMP, core.Pz)
print(res.round(3)); print("pooled core-8 MRS ratio", round(r_pool, 3), "minR2", round(r2p.min(), 4))
res.to_csv(R / "tracer" / "tracer_ratio_by_day.csv", index=False)
