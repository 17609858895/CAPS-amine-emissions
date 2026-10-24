"""Step 12 (trial): out-of-day calibration of placebo-conformal inference.

Each placebo window on day j is tested against the null formed by placebo windows of the OTHER days
(same step length L / same estimator); the empirical rejection rate at alpha = 0.05 / 0.10 estimates the
false-positive rate of the full procedure.  Also reports precision (RMS) per estimator.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import R

tag = sys.argv[1] if len(sys.argv) > 1 else "main"
PL = pd.read_csv(R / "psc" / tag / "placebo_psc.csv")
EST = [c for c in ["BA_pre", "BA_post", "LIN_BR", "ML_fwd", "ML_bwd", "ML_BR", "PSC", "FWD", "CAPS"] if c in PL]
rows = []
for (tgt, sd), P in PL.groupby(["target", "step_day"]):
    for e in EST:
        pv = []
        for j, Pj in P.groupby("day"):
            null = np.abs(P[P.day != j][e].values)
            pv += [(1 + np.sum(null >= abs(v))) / (1 + len(null)) for v in Pj[e].values]
        pv = np.array(pv)
        rows.append({"target": tgt, "step_day": sd, "estimator": e, "n": len(pv), "FPR05": np.mean(pv <= .05),
                     "FPR10": np.mean(pv <= .10), "RMS": np.sqrt(np.mean(P[e] ** 2))})
C = pd.DataFrame(rows)
S = C.groupby(["target", "estimator"])[["FPR05", "FPR10", "RMS"]].mean().round(3)
S.to_csv(R / "psc" / tag / "placebo_calibration.csv")
print(S.to_string())
