"""Step 1: read-only extraction of the 10 official Zenodo 5153417 XLSX files.

- Process Data: columns keyed by instrument tag (row 7), never by display label.
- GasMET2: all measured channels with their own timestamps (no resampling here).
- Lab sheets (Lean/Rich/WW): dumped as-is for later inspection (not used as online inputs).
Raw XLSX files are opened with openpyxl read_only/data_only and never modified.
"""
from pathlib import Path
from datetime import datetime, time
import json
import pandas as pd
import openpyxl

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "carbon_capture_zenodo_5153417" / "raw"
OUT = Path(__file__).resolve().parents[1] / "data_derived"
OUT.mkdir(exist_ok=True)

proc_frames, gas_frames, lab_frames, tagmap = [], [], [], []
for path in sorted(RAW.glob("day*_raw.xlsx"), key=lambda p: int(p.name[3:].split("_")[0])):
    day = int(path.name[3:].split("_")[0])
    print("reading", path.name, flush=True)
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True, keep_links=False)
    for sh in wb:
        rows = list(sh.values)
        if sh.title == "Process Data":
            names, units, tags = rows[4:7]
            didx = names.index("Date")
            idx = [i for i, t in enumerate(tags) if t is not None]
            recs = [r for r in rows[7:] if isinstance(r[didx], datetime)]
            df = pd.DataFrame([[r[i] for i in idx] for r in recs], columns=[str(tags[i]) for i in idx])
            df = df.apply(pd.to_numeric, errors="coerce")
            df.insert(0, "timestamp", [r[didx] for r in recs])
            df.insert(0, "day", day)
            proc_frames.append(df)
            tagmap += [{"day": day, "tag": str(tags[i]), "label": names[i], "unit": units[i]} for i in idx]
        elif sh.title == "GasMET2":
            names = [str(n).strip() if n is not None else "" for n in rows[0]]
            hasdate = "Date" in names
            tidx = names.index("Time")
            didx = names.index("Date") if hasdate else None
            meas = [(i, n) for i, n in enumerate(names)
                    if n and n not in ("Time", "Date", "Time [s]", "Time [h]", "Unit", "Compensation", "Residual")]
            out = []
            for r in rows[1:]:
                t = r[tidx]
                if hasdate and isinstance(r[didx], datetime) and isinstance(t, time):
                    ts = datetime.combine(r[didx].date(), t)
                elif isinstance(t, datetime):
                    ts = t
                else:
                    continue
                out.append([ts] + [r[i] for i, _ in meas])
            df = pd.DataFrame(out, columns=["timestamp"] + [n for _, n in meas])
            for c in df.columns[1:]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df.insert(0, "day", day)
            gas_frames.append(df)
        elif sh.title.endswith("Data") and sh.title != "Process Data":
            for k, r in enumerate(rows):
                if any(v is not None for v in r):
                    lab_frames.append({"day": day, "sheet": sh.title, "row": k + 1,
                                       "values": json.dumps([str(v) if v is not None else None for v in r])})
    wb.close()

proc = pd.concat(proc_frames, ignore_index=True)
gas = pd.concat(gas_frames, ignore_index=True)
proc.to_csv(OUT / "process_raw_by_tag.csv.gz", index=False)
gas.to_csv(OUT / "gasmet2_raw.csv.gz", index=False)
pd.DataFrame(lab_frames).to_csv(OUT / "lab_sheets_rawdump.csv.gz", index=False)
pd.DataFrame(tagmap).drop_duplicates(["tag", "label", "unit"]).to_csv(OUT / "tag_label_map.csv", index=False)
print(proc.shape, gas.shape, len(lab_frames))
print(gas.columns.tolist())
