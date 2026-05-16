"""
Slope plots: per-EBSN trajectories from gamma=0.5 to gamma=1.

For each canonical EBSN, plots two dots (its mean ACR at gamma=0.5 and at
gamma=1) connected by a line, so the reader can see which EBSNs gain when
the emotion-based norm is fully relied upon and which fall.

Usage:
    # all four leading-four base norms, coloured by cooperative half-rule
    python plotting/slope_plots.py

    # a single base norm
    python plotting/slope_plots.py --norm Shunning

    # colour by competitive half-rule instead
    python plotting/slope_plots.py --color competitive
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from _common import (
    LEADING_FOUR,
    NORM_COLOURS,
    PLOTS_DIR,
    canonical_filter,
    filter_standard,
    load_data,
)

# Endpoints of the slope. Kept as a constant so it's obvious where to change.
GAMMA_LEFT = 0.5
GAMMA_RIGHT = 1.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_for_slope(df: pd.DataFrame, base_norm: str) -> pd.DataFrame:
    """Wide table with one row per EBSN: ACR at gamma=0.5 and at gamma=1."""
    plot_data = df[df["base_social_norm"] == base_norm].copy()
    plot_data = plot_data[plot_data["gamma_center"].isin([GAMMA_LEFT, GAMMA_RIGHT])]

    agg = (
        plot_data.groupby([
            "eb_social_norm", "gamma_center",
            "Cooperative-Social Norm", "Competitive-Social Norm",
        ])
        .agg(average_cooperation=("average_cooperation", "mean"))
        .reset_index()
    )

    wide = agg.pivot_table(
        index=["eb_social_norm",
               "Cooperative-Social Norm", "Competitive-Social Norm"],
        columns="gamma_center",
        values="average_cooperation",
    ).reset_index()

    # Keep only EBSNs that have data at both gamma values
    wide = wide.dropna(subset=[GAMMA_LEFT, GAMMA_RIGHT])
    wide["delta"] = wide[GAMMA_RIGHT] - wide[GAMMA_LEFT]
    return wide


def get_baseline_acr(df: pd.DataFrame, base_norm: str) -> float | None:
    """Mean ACR for the base norm at gamma=0 (where the EBSN is never used).

    Returns None if no gamma=0 data is available for this base norm.
    """
    rows = df[(df["base_social_norm"] == base_norm) & (df["gamma_center"] == 0)]
    if rows.empty:
        return None
    return float(rows["average_cooperation"].mean())


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_slope(
    df: pd.DataFrame,
    base_norm: str,
    *,
    color_by: str = "cooperative",
    top_k: int = 10,
    figsize: tuple[float, float] = (6.5, 5.0),
) -> plt.Figure:
    """Slope plot connecting each EBSN's ACR at gamma=0.5 to its ACR at gamma=1.

    The top ``top_k`` EBSNs (by ACR at gamma=1) are drawn in their half-rule
    colours; the rest are drawn in muted grey. A horizontal dashed line marks
    the baseline ACR of the base 2nd-order norm at gamma=0.
    """
    color_column = {
        "cooperative": "Cooperative-Social Norm",
        "competitive": "Competitive-Social Norm",
    }[color_by]

    wide = aggregate_for_slope(df, base_norm)
    if wide.empty:
        raise ValueError(f"No data for base norm {base_norm!r} after filtering.")

    # Rank by ACR at gamma=1 (matches "elite EBSNs" criterion elsewhere)
    wide = wide.sort_values(GAMMA_RIGHT, ascending=False).reset_index(drop=True)
    is_top = wide.index < top_k

    baseline_acr = get_baseline_acr(df, base_norm)

    fig, ax = plt.subplots(figsize=figsize)

    BG_COLOUR = "#cccccc"

    # 1) Background trajectories (non-top-K)
    for _, row in wide.loc[~is_top].iterrows():
        ax.plot(
            [GAMMA_LEFT, GAMMA_RIGHT],
            [row[GAMMA_LEFT], row[GAMMA_RIGHT]],
            color=BG_COLOUR, alpha=0.45, linewidth=0.9, zorder=1,
        )
        ax.scatter(
            [GAMMA_LEFT, GAMMA_RIGHT],
            [row[GAMMA_LEFT], row[GAMMA_RIGHT]],
            color=BG_COLOUR, s=22, alpha=0.6,
            edgecolors="white", linewidths=0.3, zorder=2,
        )

    # 2) Top-K trajectories (foreground, coloured)
    for _, row in wide.loc[is_top].iterrows():
        colour = NORM_COLOURS.get(row[color_column], "#7f7f7f")
        ax.plot(
            [GAMMA_LEFT, GAMMA_RIGHT],
            [row[GAMMA_LEFT], row[GAMMA_RIGHT]],
            color=colour, alpha=0.9, linewidth=1.9, zorder=4,
        )
        ax.scatter(
            [GAMMA_LEFT, GAMMA_RIGHT],
            [row[GAMMA_LEFT], row[GAMMA_RIGHT]],
            color=colour, s=48, alpha=1.0,
            edgecolors="white", linewidths=0.5, zorder=5,
        )

    # 3) Baseline dashed reference line (and label on the right margin)
    if baseline_acr is not None:
        ax.axhline(
            baseline_acr,
            linestyle="--", linewidth=1.1,
            color="#444444", alpha=0.85, zorder=3,
        )
        ax.text(
            GAMMA_RIGHT + 0.015, baseline_acr,
            f"baseline {base_norm}\nACR = {baseline_acr:.2f}",
            ha="left", va="center", fontsize=7, color="#444444",
        )

    # Axis cosmetics
    margin = 0.08
    ax.set_xlim(GAMMA_LEFT - margin, GAMMA_RIGHT + margin)
    ax.set_xticks([GAMMA_LEFT, GAMMA_RIGHT])
    ax.set_xticklabels([f"$\\gamma$ = {GAMMA_LEFT}", f"$\\gamma$ = {GAMMA_RIGHT}"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Average Cooperation Ratio (ACR)")

    n_rise = int((wide["delta"] > 0).sum())
    n_fall = int((wide["delta"] < 0).sum())
    n_flat = int((wide["delta"] == 0).sum())
    ax.set_title(
        f"EB-extension of {base_norm}: trajectories from "
        f"$\\gamma$={GAMMA_LEFT} to $\\gamma$={GAMMA_RIGHT}\n"
        f"{n_rise} rise, {n_fall} fall, {n_flat} flat  "
        f"·  top {top_k} (by $\\gamma$={GAMMA_RIGHT} ACR) highlighted, "
        f"coloured by {color_column.lower()}",
        fontsize=9,
    )

    ax.grid(True, axis="y", alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Faint vertical guides at the two gamma anchors
    for gx in (GAMMA_LEFT, GAMMA_RIGHT):
        ax.axvline(gx, color="#888888", linewidth=0.5, alpha=0.4, zorder=0)

    # Legend: colours actually used by the highlighted top-K, plus a grey
    # entry for the de-emphasised remainder
    top_categories = list(dict.fromkeys(wide.loc[is_top, color_column].tolist()))
    handles = [
        mpatches.Patch(color=NORM_COLOURS.get(name, "#7f7f7f"), label=name)
        for name in top_categories
    ]
    n_others = int((~is_top).sum())
    if n_others:
        handles.append(
            mpatches.Patch(color=BG_COLOUR, label=f"other ({n_others})")
        )
    ax.legend(
        handles=handles,
        title=f"top {top_k}: {color_column.replace('-', ' ')}",
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
    # Include gamma=0 in the filter so the baseline reference line has data
    filtered = canonical_filter(
        filter_standard(merged, target_gammas=(0, GAMMA_LEFT, GAMMA_RIGHT)),
    )

    norms_to_plot = [args.norm] if args.norm else LEADING_FOUR
    for norm in norms_to_plot:
        fig = plot_slope(filtered, norm, color_by=args.color)
        safe_name = norm.lower().replace(" ", "_")
        out_path = PLOTS_DIR / f"slope_{safe_name}_{args.color}.png"
        save_figure(fig, out_path)


if __name__ == "__main__":
    main()
