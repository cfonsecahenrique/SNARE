"""
Xi-robustness figures: average cooperation vs. private assessment level (1-q),
one line per consensus threshold kappa-tilde, one panel per xi value.

Data sources (stitched, same logic as plot_fig2_main.py):
  - outputs/xi_robustness.csv          : q in {0.2, 0.4, 0.6, 0.8}, mu=2, gens=3000
  - outputs/consensus_sweep_results.csv: q=1 only (non-random IM init — correct SJ baseline)

Produces four plots (xi in {0.0, 0.01, 0.05, 0.1}):
  plotting/plots/xi_robustness/sj_fallback_emotion_xi{xi_str}_alpha0.0.png
  ../Emotion as a Solution to Private Assessment/PRSB/figs/xi_robustness/{same}
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

ALPHA = 0.0
GAMMA_CENTER = 1.0
FALLBACK = "emotion"

# ── Load xi_robustness.csv (mu=2, gens=3000) for q in {0.2, 0.4, 0.6, 0.8} ──
_COL_NAMES = [
    "base_social_norm", "eb_social_norm", "Z", "gens", "mu", "chi", "eps", "alpha",
    "q", "consensus_thresh", "xi", "non_consensus_strategy", "b", "c", "beta",
    "convergence_period", "gamma_min", "gamma_max", "gamma_delta",
    "gamma_center", "average_cooperation", "average_consensus", "G",
    "AllD_Comp", "AllD_Coop", "Disc_Comp", "Disc_Coop",
    "pDisc_Comp", "pDisc_Coop", "AllC_Comp", "AllC_Coop",
]
df_xi = pd.read_csv(
    os.path.join(_SNARE_ROOT, "outputs", "xi_robustness.csv"),
    names=_COL_NAMES, skiprows=1, on_bad_lines="skip",
)
for col in ["q", "consensus_thresh", "xi", "alpha", "average_cooperation", "gamma_center", "mu"]:
    df_xi[col] = pd.to_numeric(df_xi[col], errors="coerce")
df_xi["Z"] = pd.to_numeric(df_xi["Z"], errors="coerce")
df_xi["gens"] = pd.to_numeric(df_xi["gens"], errors="coerce")
df_xi = df_xi[
    (df_xi["non_consensus_strategy"] == FALLBACK)
    & (df_xi["alpha"] == ALPHA)
    & (df_xi["gamma_center"] == GAMMA_CENTER)
    & (df_xi["Z"] == 50)
    & (df_xi["mu"] == 2)
    & (df_xi["gens"] == 3000)
][["q", "consensus_thresh", "xi", "average_cooperation"]]

# ── Load consensus_sweep_results.csv for q=1 only ────────────────────────────
df_q1_all = pd.read_csv(
    os.path.join(_SNARE_ROOT, "outputs", "consensus_sweep_results.csv"),
    on_bad_lines="skip",
)
for col in ["q", "consensus_thresh", "xi", "alpha", "average_cooperation", "gamma_center"]:
    df_q1_all[col] = pd.to_numeric(df_q1_all[col], errors="coerce")
df_q1_all = df_q1_all[
    (df_q1_all["fallback"] == FALLBACK)
    & (df_q1_all["alpha"] == ALPHA)
    & (df_q1_all["gamma_center"] == GAMMA_CENTER)
    & (df_q1_all["q"] == 1.0)
][["q", "consensus_thresh", "xi", "average_cooperation"]]

# ── Combine ───────────────────────────────────────────────────────────────────
# K_VALS from df_xi only: those are the kappa values with actual multi-point lines.
# consensus_sweep_results has denser kappa sampling; including its extras would
# produce single-point lines (at 1-q=0 only) that show in the legend but not the plot.
K_VALS = sorted(df_xi["consensus_thresh"].round(2).dropna().unique())

df = pd.concat([df_xi, df_q1_all], ignore_index=True)
df["consensus_thresh"] = df["consensus_thresh"].round(2)
df["q"] = df["q"].round(2)
df["xi"] = df["xi"].round(3)
df["private_assessment"] = (1.0 - df["q"]).round(2)
df = df[df["consensus_thresh"].isin(K_VALS)]

xi_vals = sorted(df["xi"].dropna().unique())
Q_VALS = sorted(df["q"].dropna().unique())

print(f"xi values: {xi_vals}")
print(f"kappa-tilde values: {K_VALS}")
print(f"q values: {Q_VALS}")

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

def section_header(text):
    return Patch(color="none", label=text)

legend_handles = [
    section_header(r"$\bf{Baseline}$"),
    Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.8,
           marker="o", markersize=5,
           label=r"$\tilde{\kappa} = 0.0$  (pure SJ)"),
    section_header(r"$\bf{Emotion\ fallback}\ (\tilde{\kappa} > 0)$"),
] + [
    Line2D([0], [0], color=color_map[k], linestyle="-", linewidth=2.0,
           marker="o", markersize=5,
           label=rf"$\tilde{{\kappa}} = {k:.1f}$")
    for k in k_colored
]

# ── One plot per xi ───────────────────────────────────────────────────────────
plots_dir = os.path.join(_HERE, "plots", "xi_robustness")
os.makedirs(plots_dir, exist_ok=True)

paper_figs = os.path.join(
    _SNARE_ROOT, "..",
    "Emotion as a Solution to Private Assessment", "PRSB", "figs", "xi_robustness",
)

for xi in xi_vals:
    agg = (
        df[df["xi"] == xi]
        .groupby(["consensus_thresh", "private_assessment"])["average_cooperation"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    for k in K_VALS:
        k_data = agg[agg["consensus_thresh"] == k].sort_values("private_assessment")
        if k == K_BASELINE:
            color, ls, lw, zorder = "#555555", "--", 1.8, 2
        else:
            color, ls, lw, zorder = color_map[k], "-", 2.0, 3

        ax.plot(
            k_data["private_assessment"], k_data["mean"],
            marker="o", markersize=5, linewidth=lw, linestyle=ls,
            color=color, zorder=zorder,
        )
        ax.fill_between(
            k_data["private_assessment"],
            k_data["mean"] - k_data["sem"],
            k_data["mean"] + k_data["sem"],
            alpha=0.12, color=color, zorder=zorder - 1,
        )

    ax.legend(handles=legend_handles, fontsize=10,
              loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=True, facecolor="white", edgecolor="#bbbbbb", framealpha=1.0,
              borderaxespad=0.5, handlelength=1.8)

    ax.set_xlabel(r"Private assessment level $(1 - q)$", fontsize=13)
    ax.set_ylabel(r"Average cooperation ratio $\eta$ (%)", fontsize=13)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 100)
    ax.set_xticks([round(1 - q, 2) for q in sorted(Q_VALS, reverse=True)])
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    xi_str = str(xi).replace(".", "")
    fname = f"sj_fallback_{FALLBACK}_xi{xi_str}_alpha{ALPHA}.png"

    local_path = os.path.join(plots_dir, fname)
    plt.savefig(local_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {local_path}")

    if os.path.isdir(paper_figs):
        paper_path = os.path.join(paper_figs, fname)
        plt.savefig(paper_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {paper_path}")

    plt.close()
