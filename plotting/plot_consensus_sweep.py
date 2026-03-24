"""
Plot: Average Cooperation Rate vs consensus_thresh and observability
Reads from outputs/consensus_sweep_results.csv
Replicates the style of sweep_GGBBBBGG_consensus_thresh_observability.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)

CSV_PATH = os.path.join(REPO_ROOT, "outputs", "consensus_sweep_results.csv")
OUT_DIR  = os.path.join(SCRIPT_DIR, "plots")
OUT_PATH = os.path.join(OUT_DIR, "sweep_consensus_thresh_observability.png")

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)


# ── load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# average_cooperation is stored on a 0-100 scale; normalise to [0, 1]
y_col    = "average_cooperation"
y_scale  = 100.0

# The reference image x-axis is 'consensus_thresh' and groups by 'q' (observability)
x_col = "consensus_thresh"
group_col = "q"          # observability / image shows "observability=X"

# ── aggregate: mean and sem per (consensus_thresh, q) ─────────────────────────
agg_map = {y_col: ["mean", "sem"]}
grouped = (
    df.groupby([x_col, group_col])
    .agg(agg_map)
)
# flatten columns and normalise
grouped.columns = ["avg_cooperation", "avg_cooperation_sem"]
grouped = (grouped / y_scale).reset_index()


obs_values = sorted(grouped[group_col].unique())

# ── colour palette matching the reference image ──────────────────────────────
# reference: blue(0), orange(0.25), green(0.5), red(0.75), purple(1)
COLORS = {0.0:  "#1f77b4",   # blue
          0.25: "#ff7f0e",   # orange
          0.5:  "#2ca02c",   # green
          0.75: "#d62728",   # red
          1.0:  "#9467bd"}   # purple

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

for obs in obs_values:
    sub = grouped[grouped[group_col] == obs].sort_values(x_col)
    color = COLORS.get(obs, None)
    
    # Line and points
    ax.plot(
        sub[x_col], sub["avg_cooperation"],
        marker="o",
        color=color,
        label=f"observability={obs}",
        linewidth=1.5,
        markersize=5,
    )
    
    # Error bars
    ax.errorbar(
        sub[x_col], sub["avg_cooperation"],
        yerr=sub["avg_cooperation_sem"],
        fmt="none",
        color=color,
        alpha=0.5,
        capsize=3
    )


# ── axes / labels ─────────────────────────────────────────────────────────────
ax.set_xlabel(x_col)
ax.set_ylabel("Average Cooperation Rate")
ax.set_title(
    "Average Cooperation Rate vs consensus_thresh and observability\n"
    "EB Social Norm: GGBBBBGG"
)

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0, 1.0)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.7)
ax.legend(loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f"\nSaved -> {OUT_PATH}")
plt.show()
