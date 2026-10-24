"""Recompute reported placebo calibration from the retained per-window errors."""
from pathlib import Path
import shutil,subprocess,sys
import pandas as pd
ROOT=Path(__file__).resolve().parent
out=ROOT/'analysis/results/psc/verification';out.mkdir(parents=True,exist_ok=True)
shutil.copy2(ROOT/'reference/placebo_psc.csv',out/'placebo_psc.csv')
subprocess.run([sys.executable,str(ROOT/'analysis/scripts/12_placebo_calibration.py'),'verification'],check=True)
expected=pd.read_csv(ROOT/'reference/placebo_calibration.csv')
actual=pd.read_csv(out/'placebo_calibration.csv')
pd.testing.assert_frame_equal(expected,actual,check_exact=False,atol=1e-12,rtol=1e-12)
print('PASS: recomputed calibration agrees with the manuscript reference table.')
