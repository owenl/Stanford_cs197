# Week 5 (Project 3) — Six-encoding VAE experiment: results
Grid from `week5_plan.md`, run with `train.py` on the eICU demo (frozen splits/features).
Fair runs: hidden_dim adjusted per encoding so trainable parameters match the
standard-encoding model (~14.7k). Test split: 345 patients, 11 deaths — all
probe differences are within noise; recovery/geometry are the informative metrics.

## Main table (fair-params, mean ± sd over seeds 42/1/2026)

| encoding         |   input_dim |   hidden |   n_params | val_loss       | probe_AUROC   | probe_AUPRC   | recovery_R2   | geometry_Spearman   |
|:-----------------|------------:|---------:|-----------:|:---------------|:--------------|:--------------|:--------------|:--------------------|
| standard         |          37 |       64 |      14773 | 11.682 ± 0.053 | 0.661 ± 0.020 | 0.072 ± 0.015 | 0.649 ± 0.034 | 0.890 ± 0.021       |
| log              |          37 |       64 |      14773 | 11.881 ± 0.021 | 0.659 ± 0.015 | 0.083 ± 0.014 | 0.594 ± 0.017 | 0.743 ± 0.016       |
| quantile         |          37 |       64 |      14773 | 13.259 ± 0.034 | 0.686 ± 0.015 | 0.074 ± 0.014 | 0.611 ± 0.014 | 0.641 ± 0.032       |
| minmax_binning   |         145 |       37 |      14665 | 10.625 ± 0.002 | 0.592 ± 0.017 | 0.078 ± 0.027 | 0.074 ± 0.008 | 0.126 ± 0.052       |
| ple              |         121 |       41 |      14569 | 16.791 ± 0.223 | 0.663 ± 0.013 | 0.056 ± 0.001 | 0.345 ± 0.014 | 0.251 ± 0.025       |
| mode_specific    |          94 |       47 |      14680 | 10.457 ± 0.004 | 0.669 ± 0.051 | 0.137 ± 0.051 | 0.229 ± 0.015 | 0.176 ± 0.012       |
| quantile_binning |         145 |       37 |      14665 | 14.455 ± 0.008 | 0.522 ± 0.050 | 0.043 ± 0.008 | 0.053 ± 0.015 | 0.056 ± 0.014       |
| periodic         |         121 |       41 |      14569 | 26.119 ± 0.206 | 0.705 ± 0.008 | 0.081 ± 0.002 | 0.352 ± 0.006 | 0.374 ± 0.042       |
| yeojohnson       |          37 |       64 |      14773 | 12.936 ± 0.085 | 0.666 ± 0.007 | 0.057 ± 0.005 | 0.662 ± 0.008 | 0.745 ± 0.015       |

## Fair vs fixed-hidden-64 contrast (seed 42) — the dimensionality confound

| encoding         |   fair_hidden |   fair_recovery |   fixed64_recovery |   fair_probe |   fixed64_probe |
|:-----------------|--------------:|----------------:|-------------------:|-------------:|----------------:|
| standard         |            64 |           0.649 |              0.59  |        0.651 |           0.625 |
| log              |            64 |           0.61  |              0.585 |        0.665 |           0.628 |
| quantile         |            64 |           0.613 |              0.608 |        0.68  |           0.697 |
| minmax_binning   |            37 |           0.075 |              0.092 |        0.604 |           0.634 |
| ple              |            41 |           0.338 |              0.411 |        0.65  |           0.665 |
| mode_specific    |            47 |           0.231 |              0.266 |        0.715 |           0.698 |
| quantile_binning |            37 |           0.064 |              0.051 |        0.527 |           0.461 |
| periodic         |            41 |           0.356 |              0.36  |        0.696 |           0.694 |
| yeojohnson       |            64 |           0.672 |              0.641 |        0.667 |           0.687 |

## Per-feature recovery R² (fair runs, seed-averaged) — key features

| encoding         |   WBC x 1000 |   sao2_bin1 |   creatinine |   glucose |   BUN |   sodium |   Hgb |
|:-----------------|-------------:|------------:|-------------:|----------:|------:|---------:|------:|
| standard         |       -0.054 |       0.5   |        0.8   |     0.795 | 0.775 |    0.621 | 0.605 |
| log              |        0.363 |       0.108 |        0.736 |     0.717 | 0.683 |    0.643 | 0.684 |
| quantile         |        0.341 |       0.492 |        0.484 |     0.648 | 0.64  |    0.695 | 0.61  |
| minmax_binning   |       -0.072 |       0.106 |        0.082 |     0.129 | 0.178 |    0.027 | 0.046 |
| ple              |        0.298 |       0.371 |        0.309 |     0.194 | 0.568 |    0.183 | 0.377 |
| mode_specific    |        0.194 |       0.086 |        0.145 |     0.305 | 0.363 |    0.063 | 0.316 |
| quantile_binning |       -0.008 |       0.02  |        0.044 |     0.144 | 0.101 |    0.012 | 0.044 |
| periodic         |        0.076 |       0.186 |        0.282 |     0.323 | 0.555 |    0.048 | 0.531 |
| yeojohnson       |        0.359 |       0.514 |        0.52  |     0.704 | 0.671 |    0.821 | 0.703 |

## Baselines (from Week 4, same test split)
- Logistic regression on full preprocessed X: AUROC 0.693, AUPRC 0.105
- Logistic regression on standardized numerical X only: AUROC 0.716, AUPRC 0.086
- Week 4 standardized VAE (latent 8): probe AUROC 0.625; recovery R² sodium 0.94, WBC −0.75, sao2 0.40

## Addendum (2026-10-29): the team's canonical six + hypothesis verdicts, updated

After reading the team's actual `progress_report5.tex` on Overleaf, the six
encodings named in PR4's next-week plan are: **standard scaling, log
transformation, quantile normalization, min-max binning, quantile PLE, and
mode-specific normalization**. The grid above now includes all six (log and
min-max binning were run 2026-10-29 under the identical protocol), plus three
extra conditions: quantile binning, periodic, and Yeo-Johnson.

New findings:
- **Log transformation** behaves like a strong scalar encoding: probe AUROC
  0.659, recovery 0.594, geometry 0.743. It rescues WBC (R² 0.363, vs standard
  -0.054) — like quantile and Yeo-Johnson, monotone tail-compression is what
  recovers the most-skewed feature. It hurts the bounded sao2 (0.108).
- **Min-max (equal-width) binning collapses just like quantile binning**:
  recovery 0.074, geometry 0.126, final KL ~0.0 (near-total posterior
  collapse). The binning failure is therefore about binning itself under this
  VAE, not about how bin edges are chosen (quantile vs equal-width).

Hypothesis verdicts (updated; H1-H5 in week5_plan.md):
- H1 (skew handling): mostly NOT supported as stated. Standard scaling already
  recovers creatinine (0.80), glucose (0.80) and BUN (0.78) best. Only the
  most extreme feature (WBC) needs a monotone transform (log 0.36, YJ 0.36,
  quantile 0.34 vs standard -0.05).
- H2 (mode-specific): supported with a twist — worst structure (recovery
  0.229, geometry 0.176, KL ~0.03) yet the best probe AUPRC (0.137).
- H3 (binning collapse): supported and strengthened — both binning schemes
  collapse (recovery <= 0.08, geometry <= 0.13, KL ~0).
- H4 (bounded features): NOT supported — sao2 is recovered best by scalar
  encodings (YJ 0.51, standard 0.50), worst by log (0.11) and mode-specific.
- H5 (no encoding transforms the problem): mostly supported — probe AUROC
  spans only 0.66-0.71 for all non-binning encodings; the baseline is strong.
