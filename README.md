#Supplementary package: data and code

Data and code for the manuscript "A whole-body electrical power model of a humanoid robot identified from onboard telemetry and tested on held-out sessions and motions".

## Contents

| Item | What it is |
|---|---|
| `Supplementary_Table_S1.csv`, `.xlsx` | One row per segment (112): session, robot, action, start and end time, rows used in the fit, subsets, measured and predicted mean power, net power of the hold, energy of the moving window |
| `data/` | Telemetry of the nine sessions, one folder per recording day and one sub-folder per segment: `*telemetry.jsonl` (10 Hz frames), `unitree_action_results.jsonl` (result of each execution), `manifest.csv` |
| `supplementary_tests/` | Telemetry of the four zero-torque tests of 23 November 2026: `Test1` computer load, `Test2` cold restarts, `Test3` battery packs, `Test4` pack voltage |
| `analysis/` | Scripts 00–27 and their result tables (nine sessions); `g1power.py` holds the shared functions |
| `analysis_supplementary/` | Scripts 28–30 and 33 (supplementary tests), 34 and 35 (analyses added in review), and their result tables |
| `g1_power_model_v2.json`, `g1_power_model.py` | Coefficients of the proposed model and a small module that evaluates it; `g1_power_model.json` is the earlier model with one coefficient per joint type |
| `manuscript/` | Figures 1–6 (`figures_v2`), and the segment tables written by the scripts |
| `figures/` | Figure 7 |

Videos of the tests and the test sheets are not part of the package.

## Requirements

Python 3 with numpy, pandas, scipy, scikit-learn, matplotlib and openpyxl. The results in the manuscript were produced with Python 3.14, numpy 2.4, pandas 3.0, scipy 1.18, scikit-learn 1.9 and matplotlib 3.11.

## How to run

All scripts take the package folder as their first argument. From the package folder:

```powershell
python analysis/01_load.py .
```

```powershell
python analysis/03_sessions_offsets.py .
```

These two read the telemetry and write a frame table (`analysis/frames.pkl`, about 30,000 frames) that the other scripts load. After that, any script can be run on its own, for example:

```powershell
python analysis_supplementary/35_review4_checks.py .
```

```powershell
python analysis_supplementary/33_offset_model_and_power_model_check.py . supplementary_tests
```

## Which script produces what

| In the manuscript | Script |
|---|---|
| Sensor response, Eq. (1), Fig. 1a | `analysis/05_bms_response.py` |
| Table 3 (coefficients and intervals) | `analysis/17_finalize_v2.py`, `analysis/24_revision_analyses.py` (part 2) |
| Table 4 (excitation) | `analysis/26_review2_checks.py` (part G) |
| Table 5 (held-out sessions) | `analysis/24_revision_analyses.py` (part 1) |
| Table 6 (energy of the moving window) | `analysis_supplementary/35_review4_checks.py` (part D) |
| Table 7 (structures and black-box regressors) | `analysis/17_finalize_v2.py`, `analysis/25_blackbox_nested_tuning.py`; sharing alone: `35_review4_checks.py` (part B) |
| Table 8 (handling of the sensor) | `analysis/23_review_checks.py` (part A), `analysis_supplementary/34_review3_analyses.py` (part C) |
| Table 9 (net power of the holds) | `analysis/14_paper_model_comparison.py`, `analysis_supplementary/35_review4_checks.py` (part C) |
| Table 10 (decomposition by gesture) | `analysis/24_revision_analyses.py` (part 4) |
| Table 11 (share of waist and legs under refits) | `analysis_supplementary/34_review3_analyses.py` (part B) |
| Tables 12 and 13, Fig. 7, Eqs. (6) and (7) | `analysis_supplementary/28`–`30`, `33_offset_model_and_power_model_check.py` |
| Section 5.5 (temperature) | `analysis/18_innovation_checks.py`, `24_revision_analyses.py` (part 5), `26_review2_checks.py` (parts B, H) |
| Section 5.6 (standing stance) | `analysis/19_stance_sequence.py`, `26_review2_checks.py` (part A) |
| Figures 1–6 | `analysis/20_paper_figures.py` |
| Supplementary Table S1 | `analysis/24_revision_analyses.py` (part 7), `26_review2_checks.py`, `analysis_supplementary/35_review4_checks.py` (part E) |

## Recomputing results from Table S1

- Table 5: `error_W` grouped by `session` (mean absolute value, mean, largest).
- Table 6: rows with `in_energy_table` = 1; `window_energy_error_J` and `window_energy_error_pct` grouped by `action`. `energy_table_exclusion` gives the reason for the three segments left out.
- Table 9, last row: `hold_net_power_predicted_W` minus `hold_net_power_measured_W` over the rows with `in_net_power_set` = 1.

## Definitions used throughout

- A session is the time between a power-on and the next power-off.
- A frame carries a fresh BMS reading when the raw BMS current differs from that of the preceding frame. Only these frames are rows of the fit.
- The regressors are filtered with the sensor response inside each segment; the first row of a segment lies 0.8 to 1.8 s after its start (`first_row_used_s`).
- Frames in the damping state are not used for fitting.
- Energy of a window: each BMS reading multiplied by the time until the next reading, cut at the edges of the window.
