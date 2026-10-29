# Week 3 (Project 3) — Tabular VAE on standardized inputs: results
Executed notebook: `results/executed/week3.ipynb` (all 36 cells, CPU torch 2.15).

## Setup (all TODOs filled)
- X = Week 2's frozen 20 columns loaded from `x_cols.json`; splits loaded frozen from
  `splits.json` (train 1,101 / val 276 / test 345).
- Imputation: categorical → "Missing" category; numerical → train median
  (e.g. age 66, creatinine 1.02, glucose 129.5). Hospital region is 10.2% "Missing" in train.
- Preprocessed input: **37 numbers/row** (25 one-hot categorical + 12 standardized continuous).
- Model: TabularVAE(input 37, hidden 64, latent 8), **14,773 parameters**; Adam lr 1e-3,
  batch 256, beta 1.0, KL warm-up 10 epochs, 100 epochs, best-val checkpoint.

## Training outcome
- Val loss 18.16 (epoch 1) → **best 11.58**; epoch 100: val total 11.87
  (recon_cat 3.46, recon_cont 5.71, KL 2.69).
- Reading: categorical reconstruction saturates early (~3.5 by epoch 30); nearly all later
  gains come from continuous reconstruction (10.5 → 5.7), while KL keeps growing
  (0.06 → 2.69) — the latent stays close to the prior (KL only 2.7 nats over 8 dims).

## Latent-dim comparison (50 epochs each)
| latent | best val loss | recon_cat | recon_cont | KL |
|---|---|---|---|---|
| 2 | 12.66 | 3.60 | 7.63 | 1.43 |
| 8 | 12.60 | 3.61 | 7.20 | 1.91 |
| 32 | 12.75 | 3.61 | 7.92 | 1.51 |
Val loss is essentially **flat in latent size** — on 1,101 training rows capacity is not
the bottleneck, and single 50-epoch runs are noisy (dim 32 is not better than dim 2).

## Linear probe (Y = mortality_at48h, validation, 9 deaths)
- z (8 numbers) → Y: **AUROC 0.480, AUPRC 0.035**
- full x (37 numbers) → Y: **AUROC 0.477, AUPRC 0.032** (prevalence baseline AUPRC 0.033)
Both are at chance on validation. With 9 positive cases this cannot rank representations;
it does say the VAE's z retains no *less* Y-signal than raw x in this (underpowered) check.
Week 4 evaluates the probe on the test split and adds recovery + geometry metrics.

## Skeleton bugs found in Week 3 (fixed in our copy only)
1. Cell 16 originally transformed the *un-dropped* val frame (crashes on unseen categories)
   and shadowed `val_df`.
2. `TabularPreprocessor.drop_unseen()` **resets the DataFrame index**, which silently
   misaligns labels (it made our probe read the first 276 rows' outcomes — 8 deaths —
   instead of the validation split's 9). Replaced with an index-preserving mask drop.
3. Cell 26 plotted `history["epoch"]` etc., but history is a list of dicts → converted
   with `pd.DataFrame(history)`.
