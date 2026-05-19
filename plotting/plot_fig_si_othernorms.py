"""
SI other-norms figures: average cooperation vs. private assessment level (1-q),
one line per consensus threshold kappa-tilde, emotion fallback.

Three plots — IS, SH, SS — each from a single self-contained sweep CSV
(mu=2, gens=3000, Z=50, emotion fallback only; includes q=0 and q=1).

Output (overwrites existing SI figures):
  plotting/plots/other norms/{norm}_consensus_thresh_observability_xi001_alpha0.0.png
  ../Emotion as a Solution to Private Assessment/PRSB/figs/other norms/{same}
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
MU = 2.0
Z = 50
GENS = 3000

NORMS = {
    "is": {"csv": "is_sweep.csv", "label": "Image Scoring (IS)"},
    "sh": {"csv": "sh_sweep.csv", "label": "Shunning (SH)"},
    "ss": {"csv": "ss_sweep.csv", "label": "Simple Standing (SS)"},
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def make_plot(norm_key: str, info: dict) -> None:
    df = pd.read_csv(
        os.path.join(_SNARE_ROOT, "outputs", info["csv"]),
        on_bad_lines="skip",
    )
    for col in ["q", "consensus_thresh", "xi", "alpha", "average_cooperation",
                "gamma_center", "mu", "Z", "gens"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[
        (df["non_consensus_strategy"] == "emotion")
        & (df["xi"] == XI)
        & (df["alpha"] == ALPHA)
        & (df["gamma_center"] == GAMMA_CENTER)
        & (df["Z"] == Z)
        & (df["mu"] == MU)
        & (df["gens"] == GENS)
    ][["q", "consensus_thresh", "average_cooperation"]]

    df["consensus_thresh"] = df["consensus_thresh"].round(2)
    df["q"] = df["q"].round(2)
    df["private_assessment"] = (1.0 - df["q"]).round(2)

    K_VALS = sorted(df["consensus_thresh"].dropna().unique())
    Q_VALS = sorted(df["q"].dropna().unique())

    print(f"[{norm_key.upper()}] kappa-tilde values: {K_VALS}")
    print(f"[{norm_key.upper()}] q values: {Q_VALS}  ({len(df)} rows)")

    agg = (
        df.groupby(["consensus_thresh", "private_assessment"])["average_cooperation"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    K_BASELINE = 0.0
    k_colored = [k for k in K_VALS if k != K_BASELINE]
    palette = plt.cm.plasma(np.linspace(0.15, 0.85, len(k_colored)))
    color_map = {k: c for k, c in zip(k_colored, palette)}

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

    def section_header(text):
        return Patch(color="none", label=text)

    legend_handles = [
        section_header(r"$\bf{Baseline}$"),
        Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.8,
               marker="o", markersize=5,
               label=r"$\tilde{\kappa} = 0.0$  (pure " + norm_key.upper() + ")"),
        section_header(r"$\bf{Emotion\ fallback}\ (\tilde{\kappa} > 0)$"),
    ] + [
        Line2D([0], [0], color=color_map[k], linestyle="-", linewidth=2.0,
               marker="o", markersize=5,
               label=rf"$\tilde{{\kappa}} = {k:.1f}$")
        for k in k_colored
    ]

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

    fname = f"{norm_key}_consensus_thresh_observability_xi001_alpha0.0.png"

    plots_dir = os.path.join(_HERE, "plots", "other norms")
    os.makedirs(plots_dir, exist_ok=True)
    local_path = os.path.join(plots_dir, fname)
    plt.savefig(local_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {local_path}")

    paper_figs = os.path.join(
        _SNARE_ROOT, "..",
        "Emotion as a Solution to Private Assessment", "PRSB", "figs", "other norms",
    )
    if os.path.isdir(paper_figs):
        paper_path = os.path.join(paper_figs, fname)
        plt.savefig(paper_path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {paper_path}")

    plt.close()


for norm_key, info in NORMS.items():
    make_plot(norm_key, info)

print("Done.")
