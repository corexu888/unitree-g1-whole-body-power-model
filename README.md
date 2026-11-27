# Supplementary data and code

Data and code for the manuscript *"Whole-Body Electrical Power Measurement, Sensor Deconvolution, and Actuator Loss Identification for Quasi-Direct-Drive Humanoid Robots Using Onboard Telemetry"* (Boyue Xu, Xun Jiang, and Da Xu).

## Contents

| Item | What it is |
|---|---|
| `Supplementary_Table_S1.csv`, `.xlsx` | One row per segment (112): session, robot, action, start and end time, rows used in the fit, subsets, measured and predicted mean power, net power of the hold, energy of the moving window |
| `data/` | Telemetry of the nine sessions, one folder per recording day and one sub-folder per segment: `*telemetry.jsonl` (10 Hz frames), `unitree_action_results.jsonl` (result of each execution), `manifest.csv` |
| `supplementary_tests/` | Telemetry of the four zero-torque tests of 23 November 2026: `Test1` computer load, `Test2` cold restarts, `Test3` battery packs, `Test4` pack voltage |
| `analysis/` | Scripts 00–27 and their result tables for the nine sessions, plus scripts 36 (Figure 5) and 37 (alternative parameter sets). Shared model functions are in `g1power.py` |
| `analysis_supplementary/` | Scripts 28–30 and 33 (supplementary tests), 34 and 35 (additional validation analyses), and their result tables |
| `g1_power_model_v2.json`, `g1_power_model.py` | Coefficients of the proposed model and a small module that evaluates it; `g1_power_model.json` is the earlier model with one coefficient per joint type |
| `model_variants/` | Complete parameter files for both fixed N7520 ratios in Table 10 and the no-temperature fit in Table 3 |
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

These commands parse the raw telemetry and create the frame and session caches used for model fitting. The repository also includes derived tables used as inputs to individual figures. Examples of supplementary analyses:

```powershell
python analysis_supplementary/35_review4_checks.py .
```

```powershell
python analysis_supplementary/33_offset_model_and_power_model_check.py . supplementary_tests
```

## Which script produces what

| In the manuscript / Supplementary Materials | Script |
|---|---|
| Sensor response, Eq. (1), Fig. 1a | `analysis/05_bms_response.py` |
| Table 3 (coefficients and intervals) | `analysis/17_finalize_v2.py`, `analysis/24_revision_analyses.py` (part 2), `analysis/37_fit_model_variants.py` (no-temperature column) |
| Table 4 (excitation) | `analysis/26_review2_checks.py` (part G) |
| Table 5 (held-out sessions) | `analysis/24_revision_analyses.py` (part 1) |
| Table 6 (energy of the moving window) | `analysis_supplementary/35_review4_checks.py` (part D) |
| Supplementary Table S2 & Fig. 3 (structures and black-box regressors) | `analysis/17_finalize_v2.py`, `analysis/25_blackbox_nested_tuning.py`; sharing alone: `35_review4_checks.py` (part B) |
| Table 7 (handling of the BMS sensor) | `analysis/23_review_checks.py` (part A), `analysis_supplementary/34_review3_analyses.py` (part C) |
| Table 8 (net power of the holds) | `analysis/14_paper_model_comparison.py`, `analysis_supplementary/35_review4_checks.py` (part C) |
| Table 9 (decomposition by gesture) | `analysis/24_revision_analyses.py` (part 4) |
| Supplementary Table S3 (share of waist and legs under refits) | `analysis_supplementary/34_review3_analyses.py` (part B) |
| Table 10 (braking variants and fixed-ratio N7520 rows) | `analysis/37_fit_model_variants.py` |
| Supplementary Tables S4 and S5, Fig. 7, Eqs. (6) and (7) | `analysis_supplementary/28`–`30`, `33_offset_model_and_power_model_check.py` |
| Section 5.5 (temperature) | `analysis/18_innovation_checks.py`, `24_revision_analyses.py` (part 5), `26_review2_checks.py` (parts B, H) |
| Section 5.6 (standing stance) | `analysis/19_stance_sequence.py`, `26_review2_checks.py` (part A) |
| Figures 1–6 | `analysis/20_paper_figures.py` calls `analysis/36_figure5_temperature.py` for all three panels of Figure 5 |
| Supplementary Table S1 | `analysis/24_revision_analyses.py` (part 7), `26_review2_checks.py`, `analysis_supplementary/35_review4_checks.py` (part E) |

## Reproduce Figure 5

From the repository root, run:

```sh
python analysis/36_figure5_temperature.py .
```

The script writes the three-panel figure to `manuscript/figures_v2/fig5_temperature.png` and `.pdf`. It redraws the submitted figure from the underlying values. Fonts and spacing can differ from the copy embedded in the manuscript.

- **Panel a, residual drift.** The released `analysis/f4_temperature_coefficient.csv` contains the held-out drift estimates for 13 reject holds and 62 other holds. To rebuild this input after running scripts 01 and 03, run `python analysis/18_innovation_checks.py . F4`.
- **Panel b, single-motor heating.** The script reads the torque estimates and motor temperature channels in `data/` and writes `analysis/figure5_motor_heating.csv`. It contains 1,090 arm and 218 waist motor observations, with channel identity, mean temperature, heating rate and the loss used on the horizontal axis. The plotted lines are simple regressions. The Section 5.5 regressions additionally account for mean motor temperature.
- **Panel c, ankle motor differences.** `analysis/figure5_ankle_temperature.csv` contains 81 paired observations per ankle from robot B's non-locomotion segments. Motor 1 is the pitch-index channel. The empirical pitch-channel cross-term sign is positive for the waist and left ankle, negative for the right ankle. The correlations between the within-pair temperature and loss differences are 0.96835 (left) and 0.90718 (right).

**Waist heating-rate regression (Section 5.5).** The difference in heating rates is regressed on the signed pitch-channel cross term in watts. The fitted slope is 0.13547 K/J (r = 0.88840).

## Refit and load the alternative coefficient sets

After scripts 01 and 03, run `python analysis/37_fit_model_variants.py .`. Each JSON in `model_variants/` contains the fitted coefficients and the held-out errors in segment-mean power. The files support point predictions.

| Parameter file in `model_variants/` | Fit | Held-out session MAE |
|---|---|---|
| `g1_power_model_n7520_ratio_1p55.json` | Table 10, N7520-14.3 to N7520-22.5 coefficient ratio fixed at 1.55 from heating rates | 1.67 W |
| `g1_power_model_n7520_ratio_2p48.json` | Table 10, ratio fixed at the square of the gear ratio, `(22.5 / 14.3)**2`, displayed as 2.48 | 1.71 W |
| `g1_power_model_no_temperature.json` | Table 3, coefficients independently refitted with the temperature factor disabled | 1.64 W |

To evaluate a variant with the public prediction interface:

```python
from g1_power_model import G1PowerModel

model = G1PowerModel(path="model_variants/g1_power_model_n7520_ratio_2p48.json")
# tau and dq are (N, 29) arrays in SDK joint order.
# Temperature T is optional. Its reference defaults to the mean of the first 15 rows.
power_W = model.predict(tau, dq, temp=T)
```

## Recomputing results from Table S1

- Table 5: `error_W` grouped by `session` (mean absolute value, mean, largest).
- Table 6: rows with `in_energy_table` = 1; `window_energy_error_J` and `window_energy_error_pct` grouped by `action`. `energy_table_exclusion` gives the reason for the three segments left out.
- Table 8, last row: `hold_net_power_predicted_W` minus `hold_net_power_measured_W` over the rows with `in_net_power_set` = 1.

## Definitions used throughout

- A session is the time between a power-on and the next power-off.
- A frame carries a fresh BMS reading when the raw BMS current differs from that of the preceding frame. Only these frames are rows of the fit.
- The regressors are filtered with the sensor response inside each segment; the first row of a segment lies 0.8 to 1.8 s after its start (`first_row_used_s`).
- Frames in the damping state are not used for fitting.
- Energy of a window: each BMS reading multiplied by the time until the next reading, cut at the edges of the window.

