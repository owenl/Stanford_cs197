"""Re-aggregate the demo grid for the CORRECTED team six (Owen, 2026-10-31):
standard, quantile (quantile-normalized scalar), mode_specific (GMM),
quantile_binning, ple, periodic. Reads saved per-run metrics.json only.
Outputs results/six_v2/six_summary.json and results/six_v2/six_results.md.
"""
import json, math
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parent
W5 = ROOT / "results" / "week5"
OUT = ROOT / "results" / "six_v2"
OUT.mkdir(parents=True, exist_ok=True)

SIX = ["standard", "quantile", "mode_specific", "quantile_binning", "ple", "periodic"]
LABELS = {
    "standard": "Standardized scalar",
    "quantile": "Quantile-normalized scalar",
    "mode_specific": "Mode-specific (Gaussian mixtures)",
    "quantile_binning": "Quantile binning",
    "ple": "Piecewise-linear encoding (PLE)",
    "periodic": "Periodic",
}
SEEDS = [42, 1, 2026]
TCRIT = 4.303  # t(0.975, df=2)

def load(enc, seed, tag):
    p = W5 / f"{enc}_seed{seed}_{tag}" / "metrics.json"
    return json.loads(p.read_text())

def ms(vals):
    return (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)

def paired_ci(diffs):
    m = st.mean(diffs)
    sd = st.stdev(diffs) if len(diffs) > 1 else 0.0
    half = TCRIT * sd / math.sqrt(len(diffs))
    return m, (m - half, m + half)

runs = {}   # runs[enc][seed] = metrics (fair)
for enc in SIX:
    runs[enc] = {s: load(enc, s, "fair") for s in SEEDS}
fixed = {enc: load(enc, 42, "fixed64") for enc in SIX}

METRICS = ["probe_test_AUROC", "probe_test_AUPRC", "mean_recovery_R2", "geometry_spearman"]
table = {}
for enc in SIX:
    row = {"label": LABELS[enc], "input_dim": runs[enc][42]["input_dim"],
           "hidden_fair": runs[enc][42]["hidden_dim"], "n_params": runs[enc][42]["n_params"]}
    for mk in METRICS:
        vals = [runs[enc][s][mk] for s in SEEDS]
        m, sd = ms(vals)
        row[mk] = {"mean": m, "sd": sd, "per_seed": vals}
    kl = [runs[enc][s]["final_val_parts"]["kl"] for s in SEEDS]
    row["final_KL"] = {"mean": st.mean(kl), "per_seed": kl}
    row["fixed64_seed42"] = {mk: fixed[enc][mk] for mk in METRICS}
    table[enc] = row

# Paired per-seed differences vs standard
paired = {}
for enc in SIX[1:]:
    paired[enc] = {}
    for mk in METRICS:
        diffs = [runs[enc][s][mk] - runs["standard"][s][mk] for s in SEEDS]
        m, ci = paired_ci(diffs)
        paired[enc][mk] = {"per_seed": diffs, "mean": m, "ci95": list(ci),
                           "all_positive": all(d > 0 for d in diffs),
                           "all_negative": all(d < 0 for d in diffs)}

# Per-feature recovery, seed means
features = list(runs["standard"][42]["recovery_R2"].keys())
per_feature = {}
for enc in SIX:
    per_feature[enc] = {f: st.mean([runs[enc][s]["recovery_R2"][f] for s in SEEDS]) for f in features}

# Locked-threshold verdicts (full_data_protocol.md)
std = table["standard"]
verdicts = {}
challengers = {e: table[e]["probe_test_AUPRC"]["mean"] - std["probe_test_AUPRC"]["mean"]
               for e in SIX[1:]}
verdicts["baseline"] = {
    "standard_geometry": std["geometry_spearman"]["mean"],
    "geometry_rule_pass": std["geometry_spearman"]["mean"] >= 0.80,
    "auprc_challengers_over_0.02": {e: d for e, d in challengers.items() if d >= 0.02},
    "validated": std["geometry_spearman"]["mean"] >= 0.80 and not any(d >= 0.02 for d in challengers.values()),
}
msr = table["mode_specific"]
verdicts["dissociation"] = {
    "auprc_gain_mean": msr["probe_test_AUPRC"]["mean"] - std["probe_test_AUPRC"]["mean"],
    "recovery_drop_mean": std["mean_recovery_R2"]["mean"] - msr["mean_recovery_R2"]["mean"],
    "criterion_on_means": (msr["probe_test_AUPRC"]["mean"] - std["probe_test_AUPRC"]["mean"] >= 0.02
                           and std["mean_recovery_R2"]["mean"] - msr["mean_recovery_R2"]["mean"] >= 0.20),
    "auprc_paired": paired["mode_specific"]["probe_test_AUPRC"],
    "recovery_paired": paired["mode_specific"]["mean_recovery_R2"],
}
qb = table["quantile_binning"]
verdicts["binning_collapse"] = {
    "encoding": "quantile_binning",
    "recovery_mean": qb["mean_recovery_R2"]["mean"], "recovery_rule_pass": qb["mean_recovery_R2"]["mean"] < 0.15,
    "kl_mean": qb["final_KL"]["mean"], "kl_rule_pass": qb["final_KL"]["mean"] < 0.5,
    "confirmed": qb["mean_recovery_R2"]["mean"] < 0.15 and qb["final_KL"]["mean"] < 0.5,
    "per_seed_recovery": qb["mean_recovery_R2"]["per_seed"], "per_seed_KL": qb["final_KL"]["per_seed"],
}
# Rescue rule: R2 beats standard by >= 0.10
rescue = {}
for f in features:
    base = per_feature["standard"][f]
    rescue[f] = {e: per_feature[e][f] - base for e in SIX[1:] if per_feature[e][f] - base >= 0.10}
verdicts["feature_rescue_over_0.10"] = rescue

winners = {mk: max(SIX, key=lambda e: table[e][mk]["mean"]) for mk in METRICS}
verdicts["panel_winners"] = winners

summary = {"encodings": SIX, "labels": LABELS, "seeds": SEEDS, "table": table,
           "paired_vs_standard": paired, "per_feature_recovery_seedmean": per_feature,
           "verdicts": verdicts}
(OUT / "six_summary.json").write_text(json.dumps(summary, indent=1))

def f3(x): return f"{x:.3f}"
lines = ["# Corrected team six — demo re-aggregation (2026-10-31)", "",
         "Six: " + ", ".join(LABELS[e] for e in SIX), "",
         "| Encoding | AUROC | AUPRC | Recovery R2 | Geometry Spearman | final KL |",
         "|---|---|---|---|---|---|"]
for e in SIX:
    r = table[e]
    lines.append("| {} | {} ± {} | {} ± {} | {} ± {} | {} ± {} | {} |".format(
        LABELS[e], f3(r["probe_test_AUROC"]["mean"]), f3(r["probe_test_AUROC"]["sd"]),
        f3(r["probe_test_AUPRC"]["mean"]), f3(r["probe_test_AUPRC"]["sd"]),
        f3(r["mean_recovery_R2"]["mean"]), f3(r["mean_recovery_R2"]["sd"]),
        f3(r["geometry_spearman"]["mean"]), f3(r["geometry_spearman"]["sd"]),
        f3(r["final_KL"]["mean"])))
lines += ["", "Panel winners: " + ", ".join(f"{mk}={LABELS[winners[mk]]}" for mk in METRICS), "",
          "## Paired per-seed differences vs standard (mean, 95% CI)", ""]
for e in SIX[1:]:
    for mk in METRICS:
        p = paired[e][mk]
        lines.append(f"- {LABELS[e]} {mk}: {p['mean']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] per-seed {[round(d,3) for d in p['per_seed']]}")
lines += ["", "## Verdicts", "", "```json", json.dumps(verdicts, indent=1), "```", "",
          "## Per-feature recovery (seed means)", ""]
lines.append("| Feature | " + " | ".join(LABELS[e] for e in SIX) + " |")
lines.append("|" + "---|" * (len(SIX) + 1))
for f in features:
    lines.append("| " + f + " | " + " | ".join(f3(per_feature[e][f]) for e in SIX) + " |")
(OUT / "six_results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines[:40]))
