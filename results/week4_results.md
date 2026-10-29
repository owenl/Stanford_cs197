# Week 4 (Project 3) — Evaluating the latent representation: results
No helper notebook exists for Week 4; `week4.ipynb` was authored from the README spec
(same frozen splits/features and Week 3's VAE). Executed: `results/executed/week4.ipynb`.
Outputs in `results/week4/`: Z arrays (`Z_standard_latent8.npz`), model, figures,
`week4_metrics.json`. The retrain reproduces Week 3 exactly (best val loss 11.584).

## 1) PRED Y (logistic probe, TEST split — 345 patients, 11 deaths, prevalence 3.2%)
| representation | AUROC | AUPRC |
|---|---|---|
| Z — VAE latent (8 numbers) | 0.625 | 0.051 |
| X — full preprocessed matrix (37) | 0.693 | 0.105 |
| X — numerical only, standardized (12) | 0.716 | 0.086 |
Compression to 8 latent dims costs some mortality signal vs raw X, but with 11 deaths
all three are within noise of each other. (Week 3's validation probe: z 0.480 vs x 0.477.)

## 2) X RECOVERY (linear R² from Z, test) — the encoding story in one table
Best recovered: sodium 0.94, glucose 0.83, admissionweight 0.80, age 0.77, platelets 0.71.
Worst: **sao2 0.40** (bounded, skew −5.2) and **WBC −0.75** — for WBC the latent code is
*worse than predicting the mean*: the single most skewed feature (skew 25.0, max 448 vs
q99 36) is effectively erased by standard scaling + MSE. This is exactly the failure mode
Project 3's encodings are meant to fix (→ Week 5 hypotheses H1/H4).

## 3) GEOMETRY
Pairwise distances in Z vs standardized-X agree strongly on 300 test patients:
**Pearson r = 0.885, Spearman ρ = 0.886**. Nearest neighbours in Z are clinically sensible:
the test patient with creatinine 6.78 has 5 latent neighbours averaging creatinine 5.10
(renal-failure cluster); the 84-year-old who died has neighbours aged 65–88 (all survivors —
mortality itself is not what the neighbourhood structure encodes).

## 4) PCA of Z
Top-2 PCs explain 22.5% + 20.4% of latent variance (figure in `results/week4/pca_z.png`).

## 5) Latent-dim comparison (dims 2/32 retrained at 50 epochs; dim 8 = the 100-epoch model)
| latent dim | best val loss | probe AUROC (test) | mean recovery R² |
|---|---|---|---|
| 2 | 12.66 | 0.673 | 0.205 |
| 8 | 11.58 | 0.625 | 0.590 |
| 32 | 12.75 | 0.680 | 0.898 |
Capacity buys **reconstruction** (recovery R² 0.21 → 0.90) but not **prediction**
(probe AUROC flat ~0.63–0.68, i.e. noise at 11 deaths) and not lower val loss.
