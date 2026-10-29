# Week 5 (Project 3) — Six-encoding VAE experiment: results
Grid from `week5_plan.md`, run with `train.py` on the eICU demo (frozen splits/features).
Fair runs: hidden_dim adjusted per encoding so trainable parameters match the
standard-encoding model (~14.7k). Test split: 345 patients, 11 deaths — all
probe differences are within noise; recovery/geometry are the informative metrics.

## Main table (fair-params, mean ± sd over seeds 42/1/2026)

| encoding         |   input_dim |   hidden |   n_params | val_loss       | probe_AUROC   | probe_AUPRC   | recovery_R2   | geometry_Spearman   |
|:-----------------|------------:|---------:|-----------:|:---------------|:--------------|:--------------|:--------------|:--------------------|
| standard         |          37 |       64 |      14773 | 11.682 ± 0.053 | 0.661 ± 0.020 | 0.072 ± 0.015 | 0.649 ± 0.034 | 0.890 ± 0.021       |
| quantile         |          37 |       64 |      14773 | 13.259 ± 0.034 | 0.686 ± 0.015 | 0.074 ± 0.014 | 0.611 ± 0.014 | 0.641 ± 0.032       |
| mode_specific    |          94 |       47 |      14680 | 10.457 ± 0.004 | 0.669 ± 0.051 | 0.137 ± 0.051 | 0.229 ± 0.015 | 0.176 ± 0.012       |
| quantile_binning |         145 |       37 |      14665 | 14.455 ± 0.008 | 0.522 ± 0.050 | 0.043 ± 0.008 | 0.053 ± 0.015 | 0.056 ± 0.014       |
| ple              |         121 |       41 |      14569 | 16.791 ± 0.223 | 0.663 ± 0.013 | 0.056 ± 0.001 | 0.345 ± 0.014 | 0.251 ± 0.025       |
| periodic         |         121 |       41 |      14569 | 26.119 ± 0.206 | 0.705 ± 0.008 | 0.081 ± 0.002 | 0.352 ± 0.006 | 0.374 ± 0.042       |
| yeojohnson       |          37 |       64 |      14773 | 12.936 ± 0.085 | 0.666 ± 0.007 | 0.057 ± 0.005 | 0.662 ± 0.008 | 0.745 ± 0.015       |

## Fair vs fixed-hidden-64 contrast (seed 42) — the dimensionality confound

| encoding         |   fair_hidden |   fair_recovery |   fixed64_recovery |   fair_probe |   fixed64_probe |
|:-----------------|--------------:|----------------:|-------------------:|-------------:|----------------:|
| standard         |            64 |           0.649 |              0.59  |        0.651 |           0.625 |
| quantile         |            64 |           0.613 |              0.608 |        0.68  |           0.697 |
| mode_specific    |            47 |           0.231 |              0.266 |        0.715 |           0.698 |
| quantile_binning |            37 |           0.064 |              0.051 |        0.527 |           0.461 |
| ple              |            41 |           0.338 |              0.411 |        0.65  |           0.665 |
| periodic         |            41 |           0.356 |              0.36  |        0.696 |           0.694 |
| yeojohnson       |            64 |           0.672 |              0.641 |        0.667 |           0.687 |

## Per-feature recovery R² (fair runs, seed-averaged) — key features

| encoding         |   WBC x 1000 |   sao2_bin1 |   creatinine |   glucose |   BUN |   sodium |   Hgb |
|:-----------------|-------------:|------------:|-------------:|----------:|------:|---------:|------:|
| standard         |       -0.054 |       0.5   |        0.8   |     0.795 | 0.775 |    0.621 | 0.605 |
| quantile         |        0.341 |       0.492 |        0.484 |     0.648 | 0.64  |    0.695 | 0.61  |
| mode_specific    |        0.194 |       0.086 |        0.145 |     0.305 | 0.363 |    0.063 | 0.316 |
| quantile_binning |       -0.008 |       0.02  |        0.044 |     0.144 | 0.101 |    0.012 | 0.044 |
| ple              |        0.298 |       0.371 |        0.309 |     0.194 | 0.568 |    0.183 | 0.377 |
| periodic         |        0.076 |       0.186 |        0.282 |     0.323 | 0.555 |    0.048 | 0.531 |
| yeojohnson       |        0.359 |       0.514 |        0.52  |     0.704 | 0.671 |    0.821 | 0.703 |

## Baselines (from Week 4, same test split)
- Logistic regression on full preprocessed X: AUROC 0.693, AUPRC 0.105
- Logistic regression on standardized numerical X only: AUROC 0.716, AUPRC 0.086
- Week 4 standardized VAE (latent 8): probe AUROC 0.625; recovery R² sodium 0.94, WBC −0.75, sao2 0.40

## Hypothesis verdicts (from week5_plan.md, scored against the runs above)

- **H1 (skewed labs favour rank-based encodings) — mostly NOT supported.** Standard
  scaling recovers creatinine (0.80), glucose (0.80) and BUN (0.78) *better* than
  quantile/PLE/binning. Rank-based encodings only win on the single most extreme
  feature, WBC (skew 25): quantile 0.34 / Yeo-Johnson 0.36 / PLE 0.30 vs standard −0.05.
  The benefit of rank transforms appears only at extreme skew, not as a general rule.
- **H2 (mode-specific will not help) — supported for representation quality, with a
  twist.** Its latent has the worst recovery (0.23) and geometry (0.18) of the scalar
  family, so its extra dimensions buy no structure. Twist: it posts the best probe
  AUPRC (0.137 ± 0.051 vs ~0.06–0.08 for the others) and the lowest encoded-space val
  loss (10.46; not comparable across encodings — target scales differ). It seems to
  keep outcome ranking while discarding feature/distance structure. Needs the
  full-data run to confirm the AUPRC edge is real (11 test deaths).
- **H3 (dimensionality confound) — supported, stronger than stated.** Quantile binning
  has no advantage to shrink: it collapses in the VAE under BOTH fairness settings
  (recovery 0.05 fair / 0.05 fixed-64; geometry ~0.05). Its Week 2 logistic-classifier
  strength does not transfer to VAE representation learning at this data scale.
- **H4 (bounded/concentrated features lose most under scalar encodings) — NOT
  supported.** sao2 recovery is best under scalar encodings (standard 0.50,
  Yeo-Johnson 0.51) and worst under binning (0.02). At this scale the wide encodings
  lose everywhere, including exactly where they were expected to win.
- **H5 (probe will not separate the encodings) — mostly supported.** Test AUROC for
  six of seven encodings sits in 0.66–0.71 (periodic highest at 0.705 ± 0.008),
  within or near seed noise; the clear exception is quantile binning (0.52).
  Encodings separate on recovery and geometry, not on downstream prediction —
  as predicted.

## Headline
On the demo data, simple scalar encodings (standard, Yeo-Johnson, quantile) produce
the most faithful latent spaces (recovery ~0.61–0.66, geometry 0.64–0.89), while
expanding encodings (binning, PLE, periodic) degrade recovery/geometry sharply in a
small-data VAE, and mode-specific normalization trades structure for outcome ranking.
Cross-encoding val-loss comparisons are not meaningful (each encoding defines its
own reconstruction target scale) — probe/recovery/geometry are the comparable metrics.
All conclusions are demo-scale (11 test deaths) and should be re-tested on full eICU.
