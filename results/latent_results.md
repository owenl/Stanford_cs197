# Latent dimension × encoding loop (2026-10-29)

Per the coaching session: latent dimension looped over all encodings,
parameter-fair within each dimension, 3 seeds {42, 1, 2026}; dim-8 rows are
the Week 5 fair runs. 54 new runs (dims 4 and 32), all completed, no failures.
Full numbers: results/latent/latent_summary.json; figure:
results/latent/fig_latent_dims.png.

Key results (mean over seeds):
- Scalar encodings gain strongly from latent capacity on recovery:
  standard R² 0.375 (dim 4) -> 0.649 (8) -> 0.893 (32); log 0.365 -> 0.594
  -> 0.763; quantile 0.381 -> 0.611 -> 0.754; Yeo-Johnson 0.413 -> 0.662 ->
  0.787. This replicates, in our pipeline, the team's observation that
  cutting 8 -> 4 hurts standardization.
- At dim 32 the probe-AUROC ranking compresses: quantile 0.715, YJ 0.708,
  log 0.704, standard 0.703 — encoding choice matters less as capacity grows.
- Mode-specific's AUPRC signature peaks at dim 8 (0.115 @4, 0.137 @8,
  0.100 @32) and its recovery stays poor at every dim (<= 0.23).
- PLE recovery improves with dim (0.213 -> 0.345 -> 0.451) but remains well
  below scalar encodings; probe scores are flat.
- Binning never recovers at any dimension, and at dim 32 its reconstructions
  become numerically unstable in some seeds (mean recovery R² driven to
  large negative values by outlier features: min-max -141 ± 200, quantile
  -150,617 ± 197,204; probe/geometry remain in their usual poor ranges).
  Periodic shows the same instability at dim 32 (recovery -0.72 ± 1.62)
  while keeping the best probe AUROC (~0.70 at every dim).

Caveats: latent 32 against a 37-dim input is a weak bottleneck, so part of
the recovery gain is near-identity capacity rather than better structure;
all results remain demo-scale (11 test deaths).
