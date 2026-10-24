"""Step 17: minimum detectable effect (MDE) versus step duration and number of replicate steps.

Placebo windows (step-free; pre 90 min, gap 60, post 90) give the window-mean error of the forward
pre-mean baseline (BA_pre) and the linear bridge (LIN_BR) for L = 60..420 min, on the log scale (AMP, Pz)
and on the level scale (AMP, Pz and the vapour lower bound V_lb = AMP - Pz / 0.484, mg Nm-3).
CAPS precision at other L is approximated by scaling BA_pre with the CAPS/BA_pre placebo RMS ratio
observed at the actual step lengths (240-310 min; results/psc/main). MDE (two-sided alpha = 0.05,
power 0.80, normal approximation) = 2.80 * RMS / sqrt(n_replicate_steps).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import load, R

OUT = R / "mde"; OUT.mkdir(parents=True, exist_ok=True)
PRE, GAP, POST, STRIDE, RHO = 90, 60, 90, 10, 0.484
m, win = load(range(1, 9))
m["seg"] = np.where(m.day <= 3, "A", "B")
for c in ["AMP", "Pz"]:
    lv = m.groupby("seg")[c].transform(lambda s: s.rolling(10, min_periods=8).mean())
    m["lv_" + c] = lv; m["lg_" + c] = np.log(lv.clip(lower=0.5))
m["lv_V"] = m.lv_AMP - m.lv_Pz / RHO
blocked = pd.Series(False, index=m.index)
for _, w in win.iterrows():
    blocked |= (m.index >= w.begin) & (m.index < w.end + pd.Timedelta(f"{GAP}min"))
blocked |= (m.day == 1) & (m.index < win.loc[win.day == 1, "end"].iloc[0])
m["blocked"] = blocked.values

rows = []
for L in [60, 120, 180, 240, 300, 360, 420]:
    for seg, s in m.groupby("seg"):
        blk = s.blocked.values
        for i in range(PRE, len(s) - L - GAP - POST, STRIDE):
            if blk[i - PRE:i + L + GAP + POST].any():
                continue
            r = {"L": L, "seg": seg, "day": s.day.iloc[i]}
            for col in ["lg_AMP", "lg_Pz", "lv_AMP", "lv_Pz", "lv_V"]:
                y = s[col].values
                pre, mid, post = np.nanmean(y[i - PRE:i]), np.nanmean(y[i:i + L]), np.nanmean(y[i + L + GAP:i + L + GAP + POST])
                r[col + "_BA"] = mid - pre
                r[col + "_BR"] = mid - (pre + post) / 2  # window-mean of the linear bridge is close to the midpoint
            rows.append(r)
P = pd.DataFrame(rows).dropna()
rms = P.drop(columns=["seg", "day"]).groupby("L").agg(lambda s: np.sqrt(np.mean(s ** 2)))
rms["n_windows"] = P.groupby("L").size()
cal = pd.read_csv(R / "psc" / "main" / "placebo_calibration.csv")
ratio = {t: cal[(cal.target == t) & (cal.estimator == "CAPS")].RMS.iloc[0] /
         cal[(cal.target == t) & (cal.estimator == "BA_pre")].RMS.iloc[0] for t in ["AMP", "Pz"]}
out = []
for L, r in rms.iterrows():
    for n_rep in [1, 2, 4]:
        for t in ["AMP", "Pz"]:
            caps_rms = r[f"lg_{t}_BA"] * ratio[t]
            out.append({"L": L, "n_rep": n_rep, "quantity": t, "scale": "log",
                        "rms_BA": r[f"lg_{t}_BA"], "rms_CAPS_approx": caps_rms,
                        "MDE_log": 2.80 * caps_rms / np.sqrt(n_rep),
                        "MDE_pct": 100 * (np.exp(2.80 * caps_rms / np.sqrt(n_rep)) - 1)})
        for q in ["AMP", "Pz", "V"]:
            rr = r[f"lv_{q}_BA"] * (ratio["AMP"] if q != "Pz" else ratio["Pz"])
            out.append({"L": L, "n_rep": n_rep, "quantity": q, "scale": "level", "rms_BA": r[f"lv_{q}_BA"],
                        "rms_CAPS_approx": rr, "MDE_level": 2.80 * rr / np.sqrt(n_rep)})
O = pd.DataFrame(out)
O.to_csv(OUT / "mde_by_duration.csv", index=False)
rms.to_csv(OUT / "placebo_rms_by_duration.csv")
print("CAPS/BA_pre ratio", {k: round(v, 3) for k, v in ratio.items()})
print(rms.round(3).to_string())
print(O[(O.scale == "log")].pivot_table(index="L", columns=["quantity", "n_rep"], values="MDE_pct").round(1).to_string())
print(O[(O.scale == "level")].pivot_table(index="L", columns=["quantity", "n_rep"], values="MDE_level").round(1).to_string())
