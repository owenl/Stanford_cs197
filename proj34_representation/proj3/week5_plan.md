# Week 5 (Project 3) — Hypotheses, frozen configuration, and experiment plan
*Week 5 (10/26–11/1) has no notebook. Its deliverables, per the README, are: hypotheses
written BEFORE running the six-encoding experiments, the finalized six encodings, a frozen
VAE configuration, agreed metrics and baselines, and one experiment script (`train.py`,
delivered alongside this plan) that runs the whole pipeline changing only the encoding.*

Data context for everything below: eICU **demo** (~2,000 patients; our pipeline keeps 1,722
stays, 57 deaths at 24–48h). All effect sizes on the demo are directional at best — the
val/test splits contain 9/11 deaths. Hypotheses are stated so they can be re-tested
unchanged on the full eICU data.

## Hypotheses (written from Week 2–4 evidence, before Week 6 runs)

**H1 — Skewed labs favour rank-based encodings.**
For strongly right-skewed labs (creatinine skew 4.6, BUN 2.5, glucose 3.1, WBC 25.0),
quantile / PLE / binning encodings will yield lower continuous reconstruction error and
higher per-feature recovery R² (Week 4 metric) than standard scaling, because under
standard scaling a handful of extreme values dominate the squared-error loss
(creatinine max z ≈ 9.7; WBC max 448 vs q99 36).

**H2 — Mode-specific normalization will NOT help on this dataset.**
In Week 2 the Bayesian GMM (a) put 5 modes on unimodal heart rate (two near-duplicates),
(b) sliced skewed tails into 6–9 pseudo-modes, and (c) found only 1 mode for FiO2 because
recording errors (max 9500%) inflate its variance. Prediction: mode-specific ≈ standard
on probe AUROC and mean recovery R², despite costing 1+K dims per variable — i.e. its
extra dimensions buy no information here. (It could still win on a truly multimodal
variable; the demo may simply not contain a clean one.)

**H3 — Dimensionality is a confound, and controlling it shrinks the wide encodings' edge.**
In Week 2's classifier check, binning (140 input dims) beat standard (32 dims) — but it
also had ~4× the features. Under the frozen fair comparison (equal trainable parameters
via `--fair-params` in train.py, or equal latent dim), the binning/PLE advantage on
recovery will shrink; if it survives parameter-matching, it is a real encoding effect.

**H4 — Bounded / concentrated variables are where scalar encodings lose most.**
sao2 (bounded, left-skew −5.2) and FiO2 (protocol spikes at 21/40/60/100) carry information
in thresholds and spikes, not in smooth magnitude. Prediction: their recovery R² gap
(binning/PLE minus standard) will be the largest of all numerical features, while
near-Gaussian features (Hgb, sodium, heart rate) will show ~no encoding effect.

**H5 — Downstream prediction will NOT separate the encodings.**
Mortality at 24–48h depends on coarse severity that every encoding preserves (Week 2:
all six encodings within AUROC 0.48–0.63 on a 9-death validation set; Week 4 probe on Z
vs X will quantify the same on test). Prediction: probe AUROC differences across encodings
< 0.05 on the demo and within seed-to-seed noise on the full data; the encodings will
separate on *recovery* and *geometry* metrics instead. If H5 fails, that is itself a
finding: encoding choice changes what the VAE keeps about patients.

## Frozen VAE configuration (all six conditions identical except the encoding)
- Splits: Week 2 frozen splits (seed 42; train 1,101 / val 276 / test 345), stratified.
- Features: the 20 X columns frozen in Week 2 (`x_cols.json`); Y = mortality_at48h.
- Imputation: numerical median (train-only), categorical → "Missing" category (Week 3 choice).
- Model: TabularVAE, latent_dim 8, hidden_dim 64, Adam lr 1e-3, batch 256,
  beta 1.0 with 10-epoch KL warm-up, 100 epochs, best-val-loss checkpoint.
- Seeds: {42, 1, 2026} per condition; report mean ± sd across seeds.
- Numerical encodings (finalized in Week 2, `encoders.py`):
  1 standard · 2 quantile→normal · 3 mode-specific (fixed) · 4 quantile binning (10) ·
  5 PLE (8 quantile bins, fixed) · 6 Yeo-Johnson + standardize (our choice).

## The fairness problem (README's "super important caveat") — and our handling
Input dims differ by encoding (standard 32 total vs binning 140), so a fixed hidden_dim
gives wide encodings more parameters. We handle it three ways, all reported:
1. **Primary:** `--fair-params` in train.py shrinks hidden_dim for wide encodings so total
   trainable parameters match the standard-encoding model (±1 layer width step).
2. **Secondary:** also run the naive fixed-hidden-64 grid; the contrast between the two
   grids *is* the dimensionality effect (tests H3).
3. Always report input_dim and n_params next to every metric — no bare scores.

## Metrics (the Week 4 suite, frozen)
- Training: val recon_cat, recon_cont, KL, total (best-checkpoint).
- PRED Y: logistic probe on Z → mortality, **test** AUROC + AUPRC.
- X RECOVERY: per-feature linear R² from Z (test), plus mean R².
- GEOMETRY: Spearman correlation between pairwise distances in Z vs standardized X (test, 300-patient sample).
- Baselines: (a) logistic regression on raw preprocessed X (Week 4), (b) PCA-8 on the
  standardized preprocessed X as a non-VAE representation, (c) Week 2's classifier numbers.

## Experiment table (to run in Week 6 — 18 runs + baselines)
| Runs | Encoding | Seeds | Config |
|---|---|---|---|
| 1–3 | standard | 42, 1, 2026 | frozen, fair-params |
| 4–6 | quantile | 42, 1, 2026 | frozen, fair-params |
| 7–9 | mode_specific | 42, 1, 2026 | frozen, fair-params |
| 10–12 | quantile_binning | 42, 1, 2026 | frozen, fair-params |
| 13–15 | ple | 42, 1, 2026 | frozen, fair-params |
| 16–18 | yeojohnson | 42, 1, 2026 | frozen, fair-params |
| 19–24 | all six, seed 42 only | 42 | fixed hidden 64 (confound contrast) |

Each run: `python train.py --encoding <name> --seed <s> --fair-params --out results_week5/<name>_seed<s>`
producing `metrics.json` (all Week 4 metrics + dims/params), `history.csv`, `model.pt`, `Z.npz`.

## Open decisions flagged for the group / course staff
See the questions section of the final report (demo vs full data for the graded runs,
whether Progress Report 5 wants the experiment table only or also pilot runs, and
whether the parameter-fairness handling above matches the staff's expectation).
