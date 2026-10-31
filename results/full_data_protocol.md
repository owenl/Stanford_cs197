# Full-data (eICU 2.0) evaluation protocol — FROZEN 2026-10-29

Locked before the full data is touched, so the demo-scale findings are tested
against criteria fixed in advance. Identical pipeline to the demo runs; only
the dataset changes. Any deviation after this freeze gets logged as a deviation.

## Baseline (finalized)

- **Primary baseline: standard scaling** in the frozen VAE (latent 8, hidden
  per parameter-fairness, 100 epochs, Adam lr 1e-3, batch 256, beta 1.0 with
  10-epoch KL warm-up). Every encoding is reported as a difference from this
  baseline, not as a bare number.
- **Reference ceiling: logistic regression on the original standardized
  features X** (the x-probe) — how much the latent compression costs.
- **Chance floor: outcome prevalence** (AUPRC of a constant predictor) and
  0.5 for AUROC, printed in every table.
- The standardized baseline is itself a hypothesis under test (H5), not an
  assumption: it "wins" only if no encoding beats it on a primary metric by
  the margins below.

## Evaluation metrics (finalized — 3 categories, primaries marked)

1. **Downstream prediction** (linear probe on Z, fit on train, scored on test)
   - PRIMARY: **AUPRC** for 48h mortality (rare outcome, prevalence ~3%;
     rankings of the positive class are what matter clinically).
   - Secondary: AUROC.
2. **Feature recovery** (linear readout Z -> each numerical feature, test)
   - PRIMARY: **mean per-feature R²** across the 12 numerical features.
   - Secondary: the per-feature table (esp. WBC, creatinine, glucose, BUN,
     sodium, Hgb, sao2 — the hypothesis-linked features).
3. **Geometry preservation** (pairwise distances, fixed 300-patient test
   subsample, seed 42)
   - PRIMARY: **Spearman correlation** between distances in Z and in X.
   - Secondary: Pearson correlation; nearest-neighbour overlap (Jaccard@10).
4. **Diagnostics (reported per encoding, never compared across encodings)**
   - Final validation KL and recon parts; seed SD of every metric above;
     failed-run / instability log (NaNs, diverged reconstructions).

Exploratory only (not part of the confirmatory readout): clustering
agreement (k-means on Z vs X, ARI) for vital-sign groupings, per the
coaching session's suggestion.

## Protocol (frozen)

- Data: full eICU 2.0, the same 12 tables and the identical Week 1–2
  procedure — same cohort filters, same frozen 20-feature set (12 numerical,
  5 binary admission-dx, 3 categorical; categoricals always one-hot),
  mortality_at48h as Y (never an input), same cleaning decisions as demo.
- Splits: stratified by Y, seed 42, same proportions as the demo pipeline.
- Encodings: the team's six are confirmatory (standardized scalar,
  quantile-normalized scalar, mode-specific normalization via Gaussian
  mixtures, quantile binning, PLE, periodic); log, min-max binning, and
  Yeo-Johnson are reported as extra conditions.
  AMENDMENT 2026-10-31 (before any full-data run): the six-method list in
  the original freeze was transcribed incorrectly from the team's Week-2
  selection; Owen confirmed the correct six above. Thresholds, metrics,
  seeds, and all other protocol terms are unchanged.
- Capacity: parameter-fair is the primary comparison (hidden width matched
  to the standard baseline, ~14,700 params at latent 8); fixed hidden-64
  runs are the secondary contrast.
- Seeds: {42, 1, 2026} for every condition; all results mean ± SD.
- Latent dimension: 8 is primary; 4 and 32 are the secondary axis (same
  loop as the demo latent study).

## Success criteria (fixed in advance)

- **Baseline validated (H5)** if standard's geometry Spearman >= 0.80 and no
  encoding beats its AUPRC by >= 0.02.
- **Dissociation confirmed (H2)** if mode-specific's AUPRC exceeds
  standard's by >= 0.02 while its mean recovery R² is >= 0.20 lower.
- **Binning collapse confirmed (H3)** if quantile binning (the binning
  method among the six) shows mean recovery R² < 0.15 and final KL < 0.5.
  (Amended 2026-10-31 with the corrected six; originally worded as "both
  binning schemes" under the incorrect six-list.)
- **Skew rule (H4/H1)** checked per-feature: a transform "rescues" a feature
  only if its R² beats standard's by >= 0.10 on that feature.
- Anything not meeting its threshold is reported as not confirmed — no
  post-hoc threshold changes.

## Status at freeze time

11 of 12 tables downloaded from Owen's Stanford Drive copy and being
verified; vitalPeriodic.csv.gz still to be obtained (PhysioNet route, run
sequentially — not in parallel with other downloads, per Owen 2026-10-29).
The run starts the moment the dataset is complete and verified.
