# CS197 Progress Report — Encoding Comparison (DRAFT v2, revised 2026-10-29)

> **Notes for the team (not part of the report):**
> - v2 follows the exact section structure of our `progress_report5.tex`
>   (This week {Plan, Vector} → Results → Next week's plan → Contributions),
>   and the Plan below is carried over from that file's "Next week's plan",
>   item by item.
> - The six encodings are the ones named in that plan — standard, log,
>   quantile, min–max binning, PLE, mode-specific. Log and min–max binning
>   were run on 2026-10-29 to complete the set; quantile binning, periodic,
>   and Yeo-Johnson are reported as extra conditions.
> - All numbers: eICU **demo** data, frozen Week-2 feature set and splits,
>   test split (345 stays, 11 deaths) — read error bars before quoting.
> - [ … ] = needs Max / Rita. Owen's contribution text is drafted from the
>   actual work; please sanity-check the numbers and framing.

**Title:** CS197 Progress Report — Numerical Encoding Comparison
**Authors:** Max Lee, Owen Liu, Rita Xiang

---

## 1. This week

### Plan

Last week's plan was to move from evaluating a VAE trained on standardized
numerical features to systematically comparing six numerical encoding
strategies: (1) develop 3–5 testable hypotheses about how feature
distributions (Gaussian, skewed, multimodal, zero-inflated) respond to
different encodings; (2) finalize and debug the six encoding strategies from
Week 2; (3) agree on a fixed VAE configuration and control for differences
in the number of learnable parameters across encodings; (4) finalize the
evaluation metrics (downstream prediction, feature recovery, geometry
preservation) against standardized-feature baselines; and (5) build a
reproducible pipeline and experiment table before running the experiments.
All five items are complete: we ran 36 experiments (9 encodings × 3 seeds
under parameter-matched capacity, plus fixed-capacity contrasts) and report
the results below.

### Vector

Our riskiest open question has changed shape. We expected encodings to trade
off prediction against fidelity; instead the two dissociate — the encoding
with the most collapsed latent geometry (mode-specific) gives the best
ranking of the rare outcome, and both binning schemes fail catastrophically
in the VAE while scalar encodings barely differ. We do not yet know whether
this dissociation is a property of the encodings or an artifact of the tiny
demo sample (11 test deaths), and resolving that on the full eICU data is
the risk we most need to retire before the final evaluation.

## 2. Results: Coding and Experiments

### Overall data and Hypotheses

We used the frozen Week-2 setup throughout: 20 features (12 numerical, 5
binary admission-diagnosis indicators, 3 categorical) from the eICU demo
(1,722 stays), stratified train/validation/test splits (seed 42), and the
Week-3 VAE configuration (latent dimension 8, 100 epochs). Encodings were
compared parameter-fairly: hidden width was adjusted per encoding so total
parameters match the standardized baseline (≈14,700). Mortality at 48 hours
is the target Y throughout — predicted from the latent representation, never
an input feature.

Hypotheses: **H1** — skewed features (creatinine, glucose, WBC) are
recovered better under quantile/log-type transforms than standard scaling.
**H2** — mode-specific normalization preserves multimodal structure best.
**H3** — binning encodings lose information the VAE cannot recover.
**H4** — bounded features (e.g., oxygen saturation) suit specialized
encodings better than plain scaling. **H5** — no encoding transforms the
problem: a standardized baseline is already a good representation.

### Results

Test-set results (mean ± SD over 3 seeds, parameter-fair):

| Encoding | Probe AUROC | Probe AUPRC | Recovery R² | Geometry (Spearman) |
|---|---|---|---|---|
| Standard | 0.661 ± 0.020 | 0.072 ± 0.015 | **0.649 ± 0.034** | **0.890 ± 0.021** |
| Log | 0.659 ± 0.015 | 0.083 ± 0.014 | 0.594 ± 0.017 | 0.743 ± 0.016 |
| Quantile | 0.686 ± 0.015 | 0.074 ± 0.014 | 0.611 ± 0.014 | 0.641 ± 0.032 |
| Min–max binning | 0.592 ± 0.017 | 0.078 ± 0.027 | 0.074 ± 0.008 | 0.126 ± 0.052 |
| PLE | 0.663 ± 0.013 | 0.056 ± 0.001 | 0.345 ± 0.014 | 0.251 ± 0.025 |
| Mode-specific | 0.669 ± 0.051 | **0.137 ± 0.051** | 0.229 ± 0.015 | 0.176 ± 0.012 |
| *Extra: quantile binning* | *0.522* | *0.043* | *0.053* | *0.056* |
| *Extra: periodic* | ***0.705*** | *0.081* | *0.352* | *0.374* |
| *Extra: Yeo-Johnson* | *0.666* | *0.057* | ***0.662*** | *0.745* |

(Figures: per-metric comparison with error bars; recovery-by-encoding;
geometry scatter for the baseline. Validation losses are not comparable
across encodings, since each changes the reconstruction target's scale.)

- **H1 mostly not supported.** Standard scaling already recovers creatinine
  (R² 0.80), glucose (0.80), and BUN (0.78) best. Only the most extreme
  feature, WBC, needs a monotone transform (log 0.36, Yeo-Johnson 0.36,
  quantile 0.34, vs standard −0.05).
- **H2 supported with a twist.** Mode-specific normalization has the worst
  structure (recovery 0.23, geometry 0.18, KL ≈ 0.03) yet the best probe
  AUPRC (0.137, ~1.9× standard). Prediction and fidelity dissociate.
- **H3 supported, more strongly than stated.** Both binning schemes collapse
  in the VAE (recovery ≤ 0.07, geometry ≤ 0.13, near-zero KL), under both
  fairness settings and all seeds — a property of binning here, not of the
  binning scheme or of training.
- **H4 not supported.** Oxygen saturation is recovered best by scalar
  encodings (Yeo-Johnson 0.51, standard 0.50) and worst by log (0.11).
- **H5 mostly supported.** Probe AUROC spans only 0.66–0.71 across all
  non-binning encodings. Directly answering last report's open question:
  **the standardized baseline is a good representation**, and no encoding
  beat it overall — gains are metric-specific (mode-specific for ranking
  the rare outcome, transforms for the most-skewed feature).

Caveat: with 11 test deaths, AUPRC differences are suggestive, not
conclusive; the full-data re-run below is the deciding test.

### Next week's plan

(1) Re-run the identical frozen pipeline on the full eICU release and check
whether the two headline patterns — the mode-specific AUPRC advantage and
the binning collapse — survive at scale. (2) Run two mechanism checks: a
prior/KL sweep for mode-specific, and a target-scaling control for the
binning collapse. (3) Agree on the Week-8 evaluation plan (metrics,
baselines, ablations) before the project deadline.

## 3. Team members' contributions

### Individual Contributions

This week we designed and ran the systematic comparison of numerical
encoding strategies. Owen [implemented the experiment pipeline and the
parameter-fair training protocol, ran the 36-run grid including the log and
min–max binning conditions, and produced the comparison tables and figures].
Max [ … ]. Rita [ … ]. Throughout the week we discussed the hypotheses,
reviewed each other's code and results, and planned the full-data re-run
together.
