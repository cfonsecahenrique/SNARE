"""
Figure S2 (action fallback): average cooperation vs. private assessment level (1-q),
one line per consensus threshold kappa-tilde, under the action-valence fallback.

Data sources (stitched):
  - outputs/xi_robustness.csv          : q in {0.2, 0.4, 0.6, 0.8}, mu=2, gens=3000
  - outputs/consensus_sweep_results.csv: q=1 only (emotion fallback proxy — both
    fallbacks give identical results at q=1 since consensus is trivially maintained
    under full observability; see paper Methods)

Both filtered to: xi=0.01, alpha=0, gamma_center=1, Z=50

Output: plotting/plots/fig2_action.png
        ../Emotion as a Solution to Private Assessment/PRSB/figs/fig2_action.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SNARE_ROOT = os.path.dirname(_HERE)

XI = 0.01
ALPHA = 0.0
GAMMA_CENTER = 1.0

# ── Load xi_robustness.csv (mu=2, gens=3000) for q in {0.2, 0.4, 0.6, 0.8} ──
_XI_COL_NAMES = [
    "base_social_norm", "eb_social_norm", "Z", "gens", "mu", "chi", "eps", "alpha",
    "q", "consensus_thresh", "xi", "non_consensus_strategy", "b", "c", "beta",
    "convergence_period", "gamma_min", "gamma_max", "gamma_delta",
    "gamma_center", "average_cooperation", "average_consensus", "G",
    "AllD_Comp", "AllD_Coop", "Disc_Comp", "Disc_Coop",
    "pDisc_Comp", "pDisc_Coop", "AllC_Comp", "AllC_Coop",
]
df_xi = pd.read_csv(
    os.path.join(_SNARE_ROOT, "outputs", "xi_robustness.csv"),
    names=_XI_COL_NAMES, skiprows=1, on_bad_lines="skip",
)
for col in ["q", "consensus_thresh", "xi", "alpha", "average_cooperation", "gamma_center", "mu"]:
    df_xi[col] = pd.to_numeric(df_xi[col], errors="coerce")
df_xi["Z"] = pd.to_numeric(df_xi["Z"], errors="coerce")
df_xi["gens"] = pd.to_numeric(df_xi["gens"], errors="coerce")
df_xi = df_xi[
    (df_xi["non_consensus_strategy"] == "action")
    & (df_xi["xi"] == XI)
    & (df_xi["alpha"] == ALPHA)
    & (df_xi["gamma_center"] == GAMMA_CENTER)
    & (df_xi["Z"] == 50)
    & (df_xi["mu"] == 2)
    & (df_xi["gens"] == 3000)
][["q", "consensus_thresh", "average_cooperation"]]

# ── q=1 baseline from consensus_sweep_results.csv (emotion proxy, see docstring) ─
df_q1 = pd.read_csv(
    os.path.join(_SNARE_ROOT, "outputs", "consensus_sweep_results.csv"),
    on_bad_lines="skip",
)
for col in ["q", "consensus_thresh", "xi", "alpha", "average_cooperation", "gamma_center"]:
    df_q1[col] = pd.to_numeric(df_q1[col], errors="coerce")
df_q1 = df_q1[
    (df_q1["fallback"] == "emotion")
    & (df_q1["xi"] == XI)
    & (df_q1["alpha"] == ALPHA)
    & (df_q1["gamma_center"] == GAMMA_CENTER)
    & (df_q1["q"] == 1.0)
][["q", "consensus_thresh", "average_cooperation"]]

# ── Combine ───────────────────────────────────────────────────────────────────
df = pd.concat([df_xi, df_q1], ignore_index=True)
df["consensus_thresh"] = df["consensus_thresh"].round(2)
df["q"] = df["q"].round(2)
df["private_assessment"] = (1.0 - df["q"]).round(2)

K_VALS = sorted(df["consensus_thresh"].dropna().unique())
Q_VALS = sorted(df["q"].dropna().unique())

print(f"kappa-tilde values: {K_VALS}")
print(f"q values: {Q_VALS}")
print(f"n rows: {len(df)}")

agg = (
    df.groupby(["consensus_thresh", "private_assessment"])["average_cooperation"]
    .agg(["mean", "sem"])
    .reset_index()
)

# ── Style ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

K_BASELINE = 0.0
k_colored = [k for k in K_VALS if k != K_BASELINE]
palette = plt.cm.plasma(np.linspace(0.15, 0.85, len(k_colored)))
color_map = {k: c for k, c in zip(k_colored, palette)}

fig, ax = plt.subplots(figsize=(7, 5))

for k in K_VALS:
    k_data = agg[agg["consensus_thresh"] == k].sort_values("private_assessment")
    if k == K_BASELINE:
        color = "#555555"
        ls = "--"
        lw = 1.8
        zorder = 2
    else:
        color = color_map[k]
        ls = "-"
        lw = 2.0
        zorder = 3

    ax.plot(
        k_data["private_assessment"],
        k_data["mean"],
        marker="o",
        markersize=5,
        linewidth=lw,
        linestyle=ls,
        color=color,
        zorder=zorder,
    )
    ax.fill_between(
        k_data["private_assessment"],
        k_data["mean"] - k_data["sem"],
        k_data["mean"] + k_data["sem"],
        alpha=0.12,
        color=color,
        zorder=zorder - 1,
    )

# ── Legend ────────────────────────────────────────────────────────────────────
def section_header(text):
    return Patch(color="none", label=text)

legend_handles = [
    section_header(r"$\bf{Baseline}$"),
    Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.8,
           marker="o", markersize=5,
           label=r"$\tilde{\kappa} = 0.0$  (pure SJ)"),
    section_header(r"$\bf{Action\ fallback}\ (\tilde{\kappa} > 0)$"),
] + [
    Line2D([0], [0], color=color_map[k], linestyle="-", linewidth=2.0,
           marker="o", markersize=5,
           label=rf"$\tilde{{\kappa}} = {k:.1f}$")
    for k in k_colored
]

ax.legend(handles=legend_handles, frameon=False, fontsize=10,
          loc="upper right", handlelength=1.8)

ax.set_xlabel(r"Private assessment level $(1 - q)$", fontsize=13)
ax.set_ylabel("Average cooperation rate (%)", fontsize=13)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0, 100)
ax.set_xticks([round(1 - q, 2) for q in sorted(Q_VALS, reverse=True)])
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
plots_dir = os.path.join(_HERE, "plots")
os.makedirs(plots_dir, exist_ok=True)
local_path = os.path.join(plots_dir, "fig2_action.png")
plt.savefig(local_path, dpi=200, bbox_inches="tight")
print(f"Saved: {local_path}")

paper_figs = os.path.join(
    _SNARE_ROOT, "..",
    "Emotion as a Solution to Private Assessment", "PRSB", "figs",
)
if os.path.isdir(paper_figs):
    paper_path = os.path.join(paper_figs, "fig2_action.png")
    plt.savefig(paper_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {paper_path}")

plt.close()
