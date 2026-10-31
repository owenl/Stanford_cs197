"""McKinsey-style figure for the CORRECTED team six (2026-10-31).
Style locked by Owen 2026-10-29 (v6): action title inside the figure, grey
bars with each panel's winner in navy (#003A70), direct value labels,
source note at bottom, all fonts >= 13pt at print size."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
S = json.loads((ROOT / "results" / "six_v2" / "six_summary.json").read_text())
T = S["table"]
ORDER = ["standard", "quantile", "mode_specific", "quantile_binning", "ple", "periodic"]
SHORT = ["Standard", "Quantile-\nnormalized", "Mode-\nspecific", "Quantile\nbinning", "PLE", "Periodic"]
PANELS = [("probe_test_AUROC", "Probe AUROC (higher is better)"),
          ("probe_test_AUPRC", "Probe AUPRC (higher is better)"),
          ("mean_recovery_R2", "Feature recovery R\u00b2 (higher is better)"),
          ("geometry_spearman", "Geometry Spearman (higher is better)")]
GREY, NAVY = "#B4B8BB", "#003A70"

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
fig.suptitle("No encoding wins overall \u2014 the leader changes by metric",
             fontsize=19, fontweight="bold", x=0.012, ha="left")
for ax, (key, title) in zip(axes.flat, PANELS):
    vals = [T[e][key]["mean"] for e in ORDER]
    sds = [T[e][key]["sd"] for e in ORDER]
    win = max(range(len(ORDER)), key=lambda i: vals[i])
    colors = [NAVY if i == win else GREY for i in range(len(ORDER))]
    bars = ax.bar(range(len(ORDER)), vals, yerr=sds, color=colors,
                  error_kw=dict(ecolor="#444444", lw=1.4, capsize=4), zorder=3)
    for i, (v, sd) in enumerate(zip(vals, sds)):
        ax.text(i, v + sd + 0.035, f"{v:.3f}", ha="center", fontsize=13.5,
                fontweight="bold" if i == win else "normal",
                color=NAVY if i == win else "#333333")
    ax.set_title(title, fontsize=15, loc="left", fontweight="bold")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(SHORT, fontsize=12.5)
    ax.tick_params(axis="y", labelsize=12.5)
    ax.set_ylim(0, 0.27 if key == "probe_test_AUPRC" else 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E3E6E8", zorder=0)
fig.text(0.012, 0.012,
         "Source: eICU demo (1,722 stays); test 345 stays, 11 deaths. Mean \u00b1 SD, 3 seeds, parameter-fair. Panel winner in navy.",
         fontsize=12.5, color="#555555")
fig.tight_layout(rect=[0, 0.03, 1, 0.94])
out = ROOT / "results" / "week6" / "fig_week6_all_encodings.png"
fig.savefig(out, dpi=200)
print("saved", out)
