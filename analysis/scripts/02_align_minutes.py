"""Step 2: minute-level alignment (no interpolation, no cross-day stitching).

Gas: GasMET2 means over right-closed (t-1 min, t], labelled at t, require >=2 raw readings.
Process: 60-s records restricted to their own calendar day (drops the 00:00 next-day boundary row).
Excluded process channel: NH3-4 (tag switch Q1_40_M1 -> Q1_40_M2, frozen values).
Output: data_derived/minute_aligned.parquet  (all 10 days, flags only; no row removal beyond alignment).
"""
from pathlib import Path
import numpy as np
import pandas as pd

D = Path(__file__).resolve().parents[1] / "data_derived"
proc = pd.read_csv(D / "process_raw_by_tag.csv.gz", parse_dates=["timestamp"])
gas = pd.read_csv(D / "gasmet2_raw.csv.gz", parse_dates=["timestamp"])
tagmap = pd.read_csv(D / "tag_label_map.csv")

GAS_RENAME = {"H2O": "g_H2O", "Carbon dioxide CO2": "g_CO2", "Ammonia NH3": "g_NH3",
              "2-Amino-2-methylpropanol C4H11NO": "AMP", "Piperazine C4H10N2": "Pz", "Oxygen": "g_O2"}
gas = gas.rename(columns=GAS_RENAME)

drop_tags = ["Q1_40_M1", "Q1_40_M2"]
ptags = [c for c in proc.columns if c not in ["day", "timestamp"] + drop_tags]

frames = []
for day, sub in proc.groupby("day", sort=True):
    date = sub.timestamp.min().normalize()
    sub = sub[sub.timestamp.dt.normalize() == date].drop_duplicates("timestamp").set_index("timestamp")
    g = gas[gas.day == day].set_index("timestamp")[list(GAS_RENAME.values())]
    gm = g.resample("1min", closed="right", label="right").mean()
    gc = g[["AMP", "Pz"]].resample("1min", closed="right", label="right").count().min(axis=1)
    gm.loc[gc < 2, :] = np.nan
    j = sub[ptags].join(gm, how="left")
    j["day"] = day
    j["gas_count"] = gc.reindex(j.index).fillna(0).astype(int)
    frames.append(j)

m = pd.concat(frames)
m.index.name = "timestamp"
# readable labels for process tags (first label seen per tag)
lab = tagmap.drop_duplicates("tag").set_index("tag")["label"].to_dict()
# TI-35 duplicate display label: 1402TI013 is the reboiler inlet (SI calls it TI-35b)
lab["1402TI013"] = "TI-35b"
m = m.rename(columns={t: lab.get(t, t) for t in ptags})
m.to_parquet(D / "minute_aligned.parquet")
print(m.shape)
print(m.groupby("day")[["AMP", "Pz"]].agg(["count", "mean", "max"]).round(1))
print([c for c in m.columns])
