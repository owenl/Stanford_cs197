# Week 6 (Project 3) — Main experiments across all encodings: results
Notebook: `proj34_representation/proj3/week6.ipynb` (executed copy in `results/executed/`).
Runs: the Week 5 grid — 7 encodings (team's six + Yeo-Johnson) × 3 seeds, parameter-fair,
plus fixed-hidden-64 contrast at seed 42 = **28 VAE trainings**, Week 4 metrics throughout.
Figures in `results/week6/`: `training_curves.png`, `metric_comparison.png`,
`per_feature_recovery.png`. All numbers: eICU demo, test = 345 patients / 11 deaths.

## Training diagnostics (README: verify no training failures)
- **No failed runs.** All 28 converged (best epoch late in training, flat val-loss tails,
  no NaNs); seed SDs ≤ 0.05 on every headline metric.
- Final val-loss components (own encoding scales — health check only): categorical recon
  ≈ 3.4–3.7 for every encoding. KL tells a story: standard 2.69, quantile 3.24,
  Yeo-Johnson 3.18, but **mode-specific 0.03 and quantile-binning 0.01** — their posteriors
  sit almost on the prior mean in aggregate KL terms while still reconstructing their
  (rescaled) targets well; cross-encoding val-loss ranking would be meaningless and is
  not used anywhere.

## Main comparison (fair runs, mean ± sd over 3 seeds)
| encoding | probe AUROC | probe AUPRC | recovery R² | geometry ρ |
|---|---|---|---|---|
| standard | 0.661 ± .020 | 0.072 ± .015 | **0.649 ± .034** | **0.890 ± .021** |
| quantile | 0.686 ± .015 | 0.074 ± .014 | 0.611 ± .014 | 0.641 ± .032 |
| mode_specific | 0.669 ± .051 | **0.137 ± .051** | 0.229 ± .015 | 0.176 ± .012 |
| quantile_binning | 0.522 ± .050 | 0.043 ± .008 | 0.053 ± .015 | 0.056 ± .014 |
| ple | 0.663 ± .013 | 0.056 ± .001 | 0.345 ± .014 | 0.251 ± .025 |
| periodic | **0.705 ± .008** | 0.081 ± .002 | 0.352 ± .006 | 0.374 ± .042 |
| yeojohnson | 0.666 ± .007 | 0.057 ± .005 | **0.662 ± .008** | 0.745 ± .015 |

## Inspection findings
- **Quantile binning collapses in the VAE** (recovery ≈ 0.05, geometry ≈ 0.05) under both
  fairness settings — not a parameter-count artefact, and consistent across seeds.
- **Mode-specific collapses geometrically** (ρ ≈ 0.18, all seeds/settings) while keeping
  the best probe AUPRC — a ranking-vs-structure dissociation, flagged for Week 7.
- Everything else is a smooth gradient: scalar encodings (standard/YJ/quantile) ≫
  expanding encodings (PLE/periodic) ≫ binning, on fidelity metrics only; prediction
  metrics stay within ~0.66–0.71 AUROC for six of seven encodings.

## Patterns carried into Week 7
1. The standardized baseline is a *good* representation by fidelity metrics — the Week 4
   question now has a comparative answer.
2. Encoding choice moves recovery/geometry a lot and prediction barely — fidelity and
   utility are different axes (Week 7 tradeoff analysis).
3. Per-feature wins do not follow Week 2 distribution labels cleanly (only WBC behaves
   as the skew story predicts) — Week 7 needs a feature-level explanation, not a
   distribution-label one.

## Addendum (2026-10-29): log + min-max binning diagnostics
Run under the identical Week 5/6 protocol after the team's canonical six were
confirmed from progress_report5.tex. Log: best val loss 11.88 ± 0.02, final
val KL 2.72 (healthy posterior use, like standard's 2.69), no failed seeds.
Min-max binning: best val loss 10.63 ± 0.00 but final KL ~0.0 — the same
near-total posterior collapse as quantile binning, confirming the Week 6
diagnosis that binning's failure is a property of the representation, not of
training or of the binning scheme. Combined 9-encoding comparison figure:
results/week6/fig_week6_all_encodings.png.
