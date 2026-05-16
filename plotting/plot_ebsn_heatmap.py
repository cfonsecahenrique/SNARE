"""
EBSN bifurcation heatmap.

Reads the SNARE sweep CSV, filters to the public-information regime
(q=1, gamma=1, standard parameters), applies symmetry reduction to canonical
representatives, decomposes each 8-bit norm into a Cooperative half and a
Competitive half, maps each half to a named 2nd-order norm, and plots a
sorted heatmap.

Produces two figures:
  - <EBSNs>/figs/ebsn_bifurcation_top30.png  (main paper)
  - <EBSNs>/figs/ebsn_bifurcation_full.png   (supplementary)
"""

import ast
import os
from itertools import chain
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
SNARE_ROOT = HERE.parent
RESULTS_CSV = SNARE_ROOT / "outputs" / "results.csv"
METADATA_CSV = SNARE_ROOT / "data" / "all_8bit_norms_with_dnf.csv"
FIGS_DIR = SNARE_ROOT.parent / "EBSNs" / "figs"

# ----------------------------------------------------------------------------
# Filter parameters (public information regime)
# ----------------------------------------------------------------------------
FILTER = dict(
    q=1.0,
    gamma_center=1.0,
    Z_min=40,
    gens_min=400,
    chi=0.01,
    eps=0.01,
)

# ----------------------------------------------------------------------------
# Named 2nd-order norms in Ohtsuki notation (C1, C0, D1, D0)
# ----------------------------------------------------------------------------
NAMED_NORMS = {
    (1, 1, 1, 1): "AG",     # All Good
    (1, 1, 0, 1): "SS",     # Simple Standing
    (1, 1, 0, 0): "IS",     # Image Scoring
    (1, 0, 0, 1): "SJ",     # Stern Judging (self-symmetric)
    (1, 0, 0, 0): "SH",     # Shunning
    (0, 1, 1, 1): "pSH",    # paradoxical Shunning
    (0, 0, 1, 1): "pIS",    # paradoxical Image Scoring
    (0, 0, 1, 0): "pSS",    # paradoxical Simple Standing
    (0, 0, 0, 0): "AB",     # All Bad
}

NORM_COLOURS = {
    "SJ":  "#1f77b4",  # blue
    "IS":  "#d62728",  # red
    "SS":  "#2ca02c",  # green
    "SH":  "#ff7f0e",  # orange
    "AG":  "#17becf",  # teal
    "AB":  "#bcbd22",  # olive
    "pSS": "#9467bd",  # purple
    "pIS": "#e377c2",  # pink
    "pSH": "#8c564b",  # brown
}
UNNAMED_COLOUR = "#cccccc"


# ----------------------------------------------------------------------------
# EBSN bit-tuple helpers
# Bit positions (flatten order of [[(a,b),(c,d)],[(e,f),(g,h)]]):
#   0=DBM 1=DBO 2=DGM 3=DGO 4=CBM 5=CBO 6=CGM 7=CGO
# ----------------------------------------------------------------------------
def parse_ebsn(s):
    """Parse the nested-list string from results.csv into a length-8 bit tuple."""
    nested = ast.literal_eval(s)
    return tuple(int(b) for b in chain.from_iterable(chain.from_iterable(nested)))


def coop_half(bits):
    """Cooperative half in Ohtsuki order (C1, C0, D1, D0) = (CGO, CBO, DGO, DBO)."""
    return (bits[7], bits[5], bits[3], bits[1])


def comp_half(bits):
    """Competitive half in Ohtsuki order (C1, C0, D1, D0) = (CGM, CBM, DGM, DBM)."""
    return (bits[6], bits[4], bits[2], bits[0])


def half_name(half):
    if half in NAMED_NORMS:
        return NAMED_NORMS[half]
    return "U" + str(int("".join(map(str, half)), 2))


def half_colour(name):
    return NORM_COLOURS.get(name, UNNAMED_COLOUR)


# ----------------------------------------------------------------------------
# Symmetry operations -- imported from canonical_ebsns to avoid duplication.
# ----------------------------------------------------------------------------
import sys
from pathlib import Path
_SNARE_ROOT = Path(__file__).resolve().parent.parent
if str(_SNARE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SNARE_ROOT))
from canonical_ebsns import (  # noqa: E402
    paper_mirror as mirror,
    ep_swap,
    orbit,
    canonical,
)


def is_trivial(bits):
    """A norm is 'trivial' (not emotion-discriminating) when the O bits and
    M bits agree at every (action, reputation) position."""
    return (bits[0] == bits[1]
            and bits[2] == bits[3]
            and bits[4] == bits[5]
            and bits[6] == bits[7])


# ----------------------------------------------------------------------------
# Load + filter + symmetry-reduce + aggregate
# ----------------------------------------------------------------------------
def load_and_aggregate():
    df = pd.read_csv(RESULTS_CSV, on_bad_lines="skip", engine="python")

    f = df[
        (df["q"] == FILTER["q"])
        & (df["gamma_center"] == FILTER["gamma_center"])
        & (df["Z"] >= FILTER["Z_min"])
        & (df["gens"] >= FILTER["gens_min"])
        & (df["chi"] == FILTER["chi"])
        & (df["eps"] == FILTER["eps"])
    ].copy()

    f["bits"] = f["eb_social_norm"].apply(parse_ebsn)
    f = f[~f["bits"].apply(is_trivial)]
    f["canonical_bits"] = f["bits"].apply(canonical)

    agg = (
        f.groupby("canonical_bits")["average_cooperation"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"canonical_bits": "bits",
                         "mean": "ACR", "std": "ACR_sd", "count": "n_runs"})
    )
    agg["ACR_se"] = agg["ACR_sd"] / np.sqrt(agg["n_runs"])
    agg["coop_half"] = agg["bits"].apply(coop_half)
    agg["comp_half"] = agg["bits"].apply(comp_half)
    agg["coop_name"] = agg["coop_half"].apply(half_name)
    agg["comp_name"] = agg["comp_half"].apply(half_name)
    agg = agg.sort_values("ACR", ascending=False).reset_index(drop=True)
    return agg


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def plot_heatmap(agg, out_path, *, max_rows=None, top_divider_at=None,
                 row_height=0.18, title=None):
    if max_rows is not None:
        agg = agg.head(max_rows).reset_index(drop=True)
    n = len(agg)

    # Set up the colour-grade for ACR bars (normalised to global ACR range)
    cmap = cm.get_cmap("YlGnBu")
    norm = mcolors.Normalize(vmin=0, vmax=100)
    bar_colours = [cmap(norm(v)) for v in agg["ACR"]]

    fig, (ax_coop, ax_comp, ax_acr) = plt.subplots(
        1, 3,
        figsize=(8, max(4, n * row_height)),
        gridspec_kw={"width_ratios": [1.4, 1.4, 3.4], "wspace": 0.05},
        sharey=True,
    )

    y = np.arange(n)

    # --- half-chip columns -------------------------------------------------
    def draw_chips(ax, names, halves, title_text):
        for i, (name, half) in enumerate(zip(names, halves)):
            ax.add_patch(mpatches.Rectangle(
                (0, i - 0.4), 1, 0.8,
                facecolor=half_colour(name),
                edgecolor="white", linewidth=0.5,
            ))
            ax.text(0.5, i, name,
                    ha="center", va="center", fontsize=7,
                    color="white" if name in NORM_COLOURS else "black")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, n - 0.5)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_title(title_text, fontsize=9)

    draw_chips(ax_coop, agg["coop_name"], agg["coop_half"],
               "Cooperative half\n(O bits)")
    draw_chips(ax_comp, agg["comp_name"], agg["comp_half"],
               "Competitive half\n(M bits)")

    # --- ACR horizontal bars (colour-graded) with SE -----------------------
    ax_acr.barh(y, agg["ACR"], xerr=agg["ACR_se"],
                color=bar_colours, height=0.75,
                edgecolor="#222222", linewidth=0.4,
                error_kw=dict(lw=0.7, capsize=2, ecolor="#555555"))
    ax_acr.set_xlim(0, 100)
    ax_acr.set_xlabel("Average Cooperation Ratio (%, mean ± SE)")
    subtitle = f"n = {n} canonical EBSNs"
    ax_acr.set_title(subtitle, fontsize=9)
    ax_acr.grid(True, axis="x", alpha=0.3)
    for spine in ("top", "right"):
        ax_acr.spines[spine].set_visible(False)

    # --- top-N horizontal divider ------------------------------------------
    if top_divider_at is not None and top_divider_at < n:
        y_line = top_divider_at - 0.5
        for ax in (ax_coop, ax_comp, ax_acr):
            ax.axhline(y_line, color="#d62728", linewidth=1.2,
                       linestyle="--", alpha=0.85, zorder=5)
        ax_acr.text(99, y_line - 0.4, f"top {top_divider_at}",
                    ha="right", va="bottom", fontsize=7,
                    color="#d62728", style="italic")

    # --- rank labels (only every Nth on the full plot) ---------------------
    label_step = 1 if n <= 35 else 5
    yticks = [i for i in range(n) if (i + 1) % label_step == 0 or i == 0]
    ax_coop.set_yticks(yticks)
    ax_coop.set_yticklabels([f"#{i+1}" for i in yticks],
                            fontsize=6 if n > 60 else 7)

    # --- named-norm colour legend ------------------------------------------
    legend_handles = [
        mpatches.Patch(color=col, label=name)
        for name, col in NORM_COLOURS.items()
    ] + [mpatches.Patch(color=UNNAMED_COLOUR, label="U·· (unnamed)")]
    fig.legend(handles=legend_handles,
               loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02),
               fontsize=8, frameon=False)

    # --- ACR colour bar (small, embedded in the ACR panel) -----------------
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = ax_acr.inset_axes([0.55, -0.10, 0.4, 0.015])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("ACR colour scale", fontsize=7)

    fig.suptitle(
        title or "Bifurcation structure of EBSNs under public information",
        fontsize=11, y=0.995,
    )
    fig.subplots_adjust(top=0.95, bottom=0.06)

    os.makedirs(out_path.parent, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {out_path}")


# ----------------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    agg = load_and_aggregate()
    print(f"Canonical EBSNs after symmetry + triviality reduction: {len(agg)}")
    print(agg[["bits", "coop_name", "comp_name", "ACR", "ACR_se", "n_runs"]]
          .head(15).to_string(index=False))

    plot_heatmap(
        agg,
        FIGS_DIR / "ebsn_bifurcation_top30.png",
        max_rows=30,
        top_divider_at=10,
        row_height=0.30,
        title="Top 30 EBSNs by ACR, with cooperative- and competitive-half decomposition",
    )
    plot_heatmap(
        agg,
        FIGS_DIR / "ebsn_bifurcation_full.png",
        max_rows=None,
        top_divider_at=10,
        row_height=0.20,
        title=f"All {len(agg)} canonical EBSNs (symmetry-reduced)",
    )
