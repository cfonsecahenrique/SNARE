"""
Jitter plots of EBSN performance across gamma values.

Extracted from jitter_plots.ipynb (cell 44e587fd). For a chosen base
second-order norm, plots the average cooperation per EBSN as a jittered
strip plot across gamma values (0, 0.5, 1). Points are coloured either
by the cooperative half-rule or the competitive half-rule of each EBSN.

Usage:
    # all four leading-four base norms, coloured by cooperative half-rule
    python plotting/jitter_plots.py

    # a single base norm
    python plotting/jitter_plots.py --norm Shunning

    # colour by the competitive half-rule instead
    python plotting/jitter_plots.py --color competitive
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (
    LEADING_FOUR,
    NORM_COLOURS,
    PLOTS_DIR,
    canonical_filter,
    filter_standard,
    load_data,
)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_for_jitter(df: pd.DataFrame, base_norm: str) -> pd.DataFrame:
    """Mean ACR per (EBSN, gamma) for one base norm, with both half-rule labels."""
    plot_data = df[df["base_social_norm"] == base_norm].copy()
    agg = (
        plot_data.groupby([
            "eb_social_norm", "gamma_center", "DNF_literals",
            "Cooperative-Social Norm", "Competitive-Social Norm",
        ])
        .agg(average_cooperation=("average_cooperation", "mean"))
        .reset_index()
    )
    return agg


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_jitter(
    df: pd.DataFrame,
    base_norm: str,
    *,
    color_by: str = "cooperative",
    jitter_amplitude: float = 0.15,
    seed: int = 0,
    figsize: tuple[float, float] = (6.5, 5.0),
) -> plt.Figure:
    """Make the jitter strip plot for one base norm.

    Parameters
    ----------
    df : DataFrame
        Output of ``filter_standard`` (with half-rule columns labelled) further
        restricted to canonical EBSNs.
    base_norm : str
        One of ``LEADING_FOUR``.
    color_by : {'cooperative', 'competitive'}
        Which half-rule's identity to colour by.
    """
    color_column = {
        "cooperative": "Cooperative-Social Norm",
        "competitive": "Competitive-Social Norm",
    }[color_by]

    agg = aggregate_for_jitter(df, base_norm)

    if agg.empty:
        raise ValueError(f"No data for base norm {base_norm!r} after filtering.")

    # Add jitter to x except at gamma=0 (baseline)
    rng = np.random.default_rng(seed=seed)
    noise = rng.uniform(-jitter_amplitude, jitter_amplitude, size=len(agg))
    agg["gamma_jittered"] = np.where(
        agg["gamma_center"] == 0,
        agg["gamma_center"],
        agg["gamma_center"] + noise,
    )

    fig, ax = plt.subplots(figsize=figsize)
    colours = agg[color_column].map(lambda n: NORM_COLOURS.get(n, "#7f7f7f"))
    ax.scatter(
        agg["gamma_jittered"],
        agg["average_cooperation"],
        c=colours,
        s=42,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.4,
    )

    # Faint vertical guides at the three canonical gamma values
    for gx in (0, 0.5, 1):
        ax.axvline(gx, color="#888888", linewidth=0.5, alpha=0.4, zorder=0)

    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$\gamma$ (probability of using EBSN)")
    ax.set_ylabel("Average Cooperation Ratio (ACR)")
    ax.set_title(
        f"EB-extension of {base_norm}\n"
        f"(coloured by {color_column.lower()})",
        fontsize=10,
    )
    ax.grid(True, axis="y", alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Build a legend only from the categories actually present in this plot
    present = list(dict.fromkeys(agg[color_column].tolist()))
    handles = [
        mpatches.Patch(color=NORM_COLOURS.get(name, "#7f7f7f"), label=name)
        for name in present
    ]
    ax.legend(
        handles=handles,
        title=color_column.replace("-", " "),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7,
        title_fontsize=8,
        frameon=False,
    )
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--norm",
        choices=LEADING_FOUR,
        default=None,
        help="Single base norm to plot. Default: all four.",
    )
    parser.add_argument(
        "--color",
        choices=("cooperative", "competitive"),
        default="cooperative",
        help="Which half-rule's identity to colour by.",
    )
    args = parser.parse_args()

    merged = load_data()
    filtered = canonical_filter(filter_standard(merged))

    norms_to_plot = [args.norm] if args.norm else LEADING_FOUR
    for norm in norms_to_plot:
        fig = plot_jitter(filtered, norm, color_by=args.color)
        safe_name = norm.lower().replace(" ", "_")
        out_path = PLOTS_DIR / f"jitter_{safe_name}_{args.color}.png"
        save_figure(fig, out_path)


if __name__ == "__main__":
    main()
