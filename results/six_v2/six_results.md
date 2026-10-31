# Corrected team six — demo re-aggregation (2026-10-31)

Six: Standardized scalar, Quantile-normalized scalar, Mode-specific (Gaussian mixtures), Quantile binning, Piecewise-linear encoding (PLE), Periodic

| Encoding | AUROC | AUPRC | Recovery R2 | Geometry Spearman | final KL |
|---|---|---|---|---|---|
| Standardized scalar | 0.661 ± 0.020 | 0.072 ± 0.015 | 0.649 ± 0.034 | 0.890 ± 0.021 | 2.691 |
| Quantile-normalized scalar | 0.686 ± 0.015 | 0.074 ± 0.014 | 0.611 ± 0.014 | 0.641 ± 0.032 | 3.236 |
| Mode-specific (Gaussian mixtures) | 0.669 ± 0.051 | 0.137 ± 0.051 | 0.229 ± 0.015 | 0.176 ± 0.012 | 0.027 |
| Quantile binning | 0.522 ± 0.050 | 0.043 ± 0.008 | 0.053 ± 0.015 | 0.056 ± 0.014 | 0.013 |
| Piecewise-linear encoding (PLE) | 0.663 ± 0.013 | 0.056 ± 0.001 | 0.345 ± 0.014 | 0.251 ± 0.025 | 0.374 |
| Periodic | 0.705 ± 0.008 | 0.081 ± 0.002 | 0.352 ± 0.006 | 0.374 ± 0.042 | 0.760 |

Panel winners: probe_test_AUROC=Periodic, probe_test_AUPRC=Mode-specific (Gaussian mixtures), mean_recovery_R2=Standardized scalar, geometry_spearman=Standardized scalar

## Paired per-seed differences vs standard (mean, 95% CI)

- Quantile-normalized scalar probe_test_AUROC: +0.025 [-0.058, +0.108] per-seed [0.029, 0.056, -0.01]
- Quantile-normalized scalar probe_test_AUPRC: +0.002 [-0.069, +0.072] per-seed [-0.002, 0.032, -0.025]
- Quantile-normalized scalar mean_recovery_R2: -0.038 [-0.089, +0.014] per-seed [-0.036, -0.059, -0.018]
- Quantile-normalized scalar geometry_spearman: -0.249 [-0.329, -0.169] per-seed [-0.274, -0.213, -0.259]
- Mode-specific (Gaussian mixtures) probe_test_AUROC: +0.008 [-0.166, +0.183] per-seed [0.063, 0.032, -0.071]
- Mode-specific (Gaussian mixtures) probe_test_AUPRC: +0.065 [-0.087, +0.216] per-seed [0.094, 0.106, -0.005]
- Mode-specific (Gaussian mixtures) mean_recovery_R2: -0.420 [-0.543, -0.296] per-seed [-0.418, -0.47, -0.371]
- Mode-specific (Gaussian mixtures) geometry_spearman: -0.714 [-0.795, -0.633] per-seed [-0.75, -0.706, -0.686]
- Quantile binning probe_test_AUROC: -0.139 [-0.313, +0.034] per-seed [-0.124, -0.078, -0.216]
- Quantile binning probe_test_AUPRC: -0.029 [-0.086, +0.029] per-seed [-0.035, -0.003, -0.048]
- Quantile binning mean_recovery_R2: -0.596 [-0.709, -0.482] per-seed [-0.585, -0.646, -0.556]
- Quantile binning geometry_spearman: -0.834 [-0.854, -0.814] per-seed [-0.84, -0.837, -0.825]
- Piecewise-linear encoding (PLE) probe_test_AUROC: +0.002 [-0.062, +0.066] per-seed [-0.002, 0.029, -0.022]
- Piecewise-linear encoding (PLE) probe_test_AUPRC: -0.016 [-0.056, +0.025] per-seed [-0.022, 0.003, -0.028]
- Piecewise-linear encoding (PLE) mean_recovery_R2: -0.303 [-0.361, -0.245] per-seed [-0.312, -0.321, -0.277]
- Piecewise-linear encoding (PLE) geometry_spearman: -0.639 [-0.718, -0.560] per-seed [-0.672, -0.609, -0.637]
- Periodic probe_test_AUROC: +0.044 [-0.005, +0.093] per-seed [0.045, 0.063, 0.024]
- Periodic probe_test_AUPRC: +0.009 [-0.025, +0.043] per-seed [0.003, 0.025, -0.001]
- Periodic mean_recovery_R2: -0.297 [-0.369, -0.225] per-seed [-0.294, -0.328, -0.27]
- Periodic geometry_spearman: -0.516 [-0.586, -0.447] per-seed [-0.514, -0.489, -0.545]

## Verdicts

```json
{
 "baseline": {
  "standard_geometry": 0.8899374576328523,
  "geometry_rule_pass": true,
  "auprc_challengers_over_0.02": {
   "mode_specific": 0.06475232217678922
  },
  "validated": false
 },
 "dissociation": {
  "auprc_gain_mean": 0.06475232217678922,
  "recovery_drop_mean": 0.4195032699996477,
  "criterion_on_means": true,
  "auprc_paired": {
   "per_seed": [
    0.09377600097713112,
    0.10585826441046742,
    -0.005377298857230858
   ],
   "mean": 0.06475232217678922,
   "ci95": [
    -0.08687614361207452,
    0.21638078796565297
   ],
   "all_positive": false,
   "all_negative": false
  },
  "recovery_paired": {
   "per_seed": [
    -0.4183539977165224,
    -0.4696043904449453,
    -0.3705514218374757
   ],
   "mean": -0.41950326999964777,
   "ci95": [
    -0.5425686504206659,
    -0.2964378895786296
   ],
   "all_positive": false,
   "all_negative": true
  }
 },
 "binning_collapse": {
  "encoding": "quantile_binning",
  "recovery_mean": 0.053027984809052485,
  "recovery_rule_pass": true,
  "kl_mean": 0.013259856381278107,
  "kl_rule_pass": true,
  "confirmed": true,
  "per_seed_recovery": [
   0.06434668735644283,
   0.036680398248880336,
   0.05805686882183428
  ],
  "per_seed_KL": [
   0.015170467234608055,
   0.011925754042855207,
   0.012683347866371058
  ]
 },
 "feature_rescue_over_0.10": {
  "age": {},
  "admissionweight": {},
  "heartrate_bin1": {},
  "respiration_bin1": {},
  "sao2_bin1": {},
  "sodium": {},
  "Hgb": {},
  "creatinine": {},
  "BUN": {},
  "glucose": {},
  "WBC x 1000": {
   "quantile": 0.39506259600188676,
   "mode_specific": 0.24787522102835052,
   "ple": 0.3512497989954069,
   "periodic": 0.12941926279420893
  },
  "platelets x 1000": {}
 },
 "panel_winners": {
  "probe_test_AUROC": "periodic",
  "probe_test_AUPRC": "mode_specific",
  "mean_recovery_R2": "standard",
  "geometry_spearman": "standard"
 }
}
```

## Per-feature recovery (seed means)

| Feature | Standardized scalar | Quantile-normalized scalar | Mode-specific (Gaussian mixtures) | Quantile binning | Piecewise-linear encoding (PLE) | Periodic |
|---|---|---|---|---|---|---|
| age | 0.767 | 0.732 | 0.287 | 0.065 | 0.448 | 0.739 |
| admissionweight | 0.842 | 0.734 | 0.244 | 0.069 | 0.392 | 0.355 |
| heartrate_bin1 | 0.740 | 0.749 | 0.274 | 0.025 | 0.381 | 0.514 |
| respiration_bin1 | 0.695 | 0.535 | 0.251 | 0.054 | 0.300 | 0.389 |
| sao2_bin1 | 0.500 | 0.492 | 0.086 | 0.020 | 0.371 | 0.186 |
| sodium | 0.621 | 0.695 | 0.063 | 0.012 | 0.183 | 0.048 |
| Hgb | 0.605 | 0.610 | 0.316 | 0.044 | 0.377 | 0.531 |
| creatinine | 0.800 | 0.484 | 0.145 | 0.044 | 0.309 | 0.282 |
| BUN | 0.775 | 0.640 | 0.363 | 0.101 | 0.568 | 0.555 |
| glucose | 0.795 | 0.648 | 0.305 | 0.144 | 0.194 | 0.323 |
| WBC x 1000 | -0.054 | 0.341 | 0.194 | -0.008 | 0.298 | 0.076 |
| platelets x 1000 | 0.695 | 0.672 | 0.221 | 0.066 | 0.324 | 0.221 |
