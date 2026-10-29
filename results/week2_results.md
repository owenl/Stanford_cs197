# Week 2 (Project 3) — Results (eICU demo)

Notebook `proj3/week2.ipynb` completed (all TODO cells implemented) and executed end-to-end.
Splits frozen (seed 42): train 1,101 / val 276 / test 345; mortality ≈ 3.3% in each.
X frozen at 20 features (12 numerical, 5 binary admission-dx, 3 categorical) in `x_cols.json`.

## Distribution characterization (190 numerical features)
- Most missing: several labs 100% missing in demo; median lab missingness 93%.
- Most skewed: AST 25.1, WBC 25.0, ALT 17.4, CPK 15.2, FiO2 14.8.
- Worst q99→max ratios (outliers/errors): FiO2 95× (max 9500 for a %), AST 18×, WBC 12.5×.
- Temperature mixes °C/°F (median 37.2, max 99); admissionheight max 600 cm (Week 1).
- Age: 74 unique integer values, 3.8% censored as '> 89' → treated as numerical (=90).

## The six encodings (output dim per variable)
| Encoding | heartrate | creatinine | sao2 | FiO2 | Roundtrip |
|---|---|---|---|---|---|
| 1 Standard | 1 | 1 | 1 | 1 | exact |
| 2 Quantile→Normal | 1 | 1 | 1 | 1 | exact; skew→0.00, range bounded ±5.2 |
| 3 Mode-specific | 6 | 7 | 7 | 2 | exact except FiO2 (MAE 37.4) |
| 4 Quantile binning (10) | 10 | 10 | 10 | 9 | lossy (creatinine MAE 0.263) |
| 5 PLE (8 bins) | 8 | 8 | 8 | 8 | exact on training range |
| 6 Yeo-Johnson (ours) | 1 | 1 | 1 | 1 | exact; creatinine skew 4.60→0.13 |

Mode-specific findings: unimodal heartrate still gets **5 modes** (two near-duplicates,
95.3/95.5); creatinine 6, glucose 9 (tail-slicing, not subpopulations); FiO2 gets **1 mode**
because the 9500 outliers inflate variance. Several BGM fits raise ConvergenceWarning
(max_iter=200).

## Classifier check (logistic regression, encodings fit on train only; Y = mortality_at48h, val n=276, 9 deaths)
| Encoding | Input dim | Val AUROC | Val AUPRC |
|---|---|---|---|
| Raw + StandardScaler (baseline) | 32 | 0.477 | 0.032 |
| 1 Standard | 32 | 0.477 | 0.032 |
| 2 Quantile | 32 | 0.559 | 0.039 |
| 3 Mode-specific | 89 | 0.607 | 0.045 |
| 4 Quantile binning | 140 | **0.634** | 0.045 |
| 5 PLE | 116 | 0.559 | 0.039 |
| 6 Yeo-Johnson | 32 | 0.532 | 0.036 |

⚠️ Val has only **9 deaths** — these AUROC gaps are within noise, and the wider encodings
also give the classifier more parameters (the Week-5 fairness caveat). Direction only.

## Bugs found in the provided skeleton (fixed in the working copy)
- **B1. `ModeSpecificEncoder.__init__`** never stores `max_components`/`weight_threshold`
  → `fit()` crashes with AttributeError. (Also: fit didn't drop NaNs.)
- **B2. `QuantilePLEEncoder.fit`** never sets `output_dim_` → `inverse_transform()` crashes,
  and `TabularPreprocessor` (which reads `output_dim_`) would crash too.
- Encoders were exported to `proj3/encoders.py` and into `tabular_vae/encoders.py`
  (replacing its "TODO: add the rest from Week 2!!").
