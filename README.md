# CAPS analysis of mixed-amine emissions

Core analysis code for **Calibrated Counterfactuals Quantify Mixed-Amine Emission Trade-offs during Flexible Carbon Capture**. Contact: Chong Liu, chong.liu@auckland.ac.nz.

The study reanalyses seven designed operating schemes in the public Niederaußem CESAR1 pilot campaign. The code covers minute alignment, counterfactual learning, placebo calibration, paired pathway uncertainty, sensitivity analyses, known-effect injections, dynamic response, experimental detectability, forecasts and operating-input perturbations. This repository contains analysis code and a small verification input set; manuscript production, figure styling, development-only estimators and model caches are excluded.

## Data and attribution

Original data: [Zenodo 5153417](https://doi.org/10.5281/zenodo.5153417), licensed CC BY 4.0 by its original depositors. Cite that dataset and the original campaign study by Jablonka et al. (2023) when using these records. The source metadata, download URLs and checksums are in `source_data.json`. Source workbooks are downloaded from Zenodo rather than republished here. Reference CSVs are derived from those records and retain that attribution.

## Environment and quick verification

The retained analysis was run with Python 3.12.3 and the versions in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
python verify_calibration.py
```

The verification recomputes empirical rejection rates and RMS errors from the retained placebo-window errors and compares every reported table cell. It does not refit the learners. At nominal 5%, the duration-averaged CAPS rejection rates round to 0.053 for AMP and 0.049 for Pz.

## Refit from public workbooks

Run from this repository root. Analysis outputs are written to `analysis/results`; derived minute data go to `analysis/data_derived`. The model runs can be computationally expensive. Cached per-day parts are reused; use a fresh output tag when changing a protocol.

```bash
python download_data.py
python analysis/scripts/01_extract_raw.py
python analysis/scripts/02_align_minutes.py
python analysis/scripts/09_placebo_stacked_cf.py --tag main --jobs 2
python analysis/scripts/12_placebo_calibration.py main
python analysis/scripts/14_caps_robust_runs.py --tag main_paths --jobs 2
python analysis/scripts/14_caps_robust_runs.py --tag shift_m5 --shift -5 --jobs 2
python analysis/scripts/14_caps_robust_runs.py --tag shift_p5 --shift 5 --jobs 2
python analysis/scripts/14_caps_robust_runs.py --tag lod_half --pz-lod half --jobs 2
python analysis/scripts/14_caps_robust_runs.py --tag lod_full --pz-lod full --jobs 2
python analysis/scripts/14_caps_robust_runs.py --tag lod_floor3 --pz-lod full --lod 3 --jobs 2
python analysis/scripts/09_placebo_stacked_cf.py --tag gap120 --gap 120 --jobs 2
python analysis/scripts/09_placebo_stacked_cf.py --tag noexo --exo "" --jobs 2
```

The pathway/robustness analysis also compares its recalculations with two earlier numerical summaries. Copy `reference/pathway_bounds_levels.csv` and `reference/transient_vs_sustained_pct.csv` into `analysis/results/psc/main/`, then run:

```bash
python analysis/scripts/14b_caps_robust_analysis.py
python analysis/scripts/16_semisynthetic_benchmark.py --target AMP --jobs 2
python analysis/scripts/16_semisynthetic_benchmark.py --target Pz --jobs 2
python analysis/scripts/17_mde_design.py
python analysis/scripts/18_time_constants.py
python analysis/scripts/04b_forecast_lean.py
python analysis/scripts/11_whatif_vs_intervention.py
```

Additional scripts implement burst diagnostics, the structural time-series comparator, exploratory tracer decorrelation and a published-preprocessing sensitivity check. Their docstrings describe the inputs and outputs. `code_manifest.json` records the retained source hashes.

## Interpretation

The seven schemes are different interventions in one pilot campaign, not independent replications of one treatment. Base learners exclude the target day, whereas models generating other-day errors for weight fitting can include step-free records from that day; this is not a fully nested outer-day evaluation. Overlapping placebos define campaign-specific empirical calibration. The adaptive estimator did not consistently outperform a single forward learner in the known-effect tests.

Pathway directions assume fixed mixing ratios with aerosol transport relatively enriched in Pz. Numerical bounds additionally use negligible vapour Pz and an aerosol Pz/AMP mass ratio of at least 0.484. Forecast input substitutions and whole-step effects use different response horizons and transformations. The manuscript and Supporting Information specify these scientific comparisons in detail.

The public release was checked by parsing all scripts and executing the calibration verification. The complete model-fitting sequence was not rerun for packaging; retained model results underlie the manuscript. No claim of independent multi-plant validation is made.
