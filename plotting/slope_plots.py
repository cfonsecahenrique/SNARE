"""
Slope plots: per-EBSN trajectories from gamma=0.5 to gamma=1.

Reads from the canonical-sweep CSVs (not results.csv) so the authoritative
HPC runs are used.  Left axis = gamma=0.5 (elite subsets only); right axis =
gamma=1 (all 66 EBSNs shown as background dots).  EBSNs with ACR >= THRESHOLD
at gamma=1 are highlighted; the rest are muted grey.

Usage:
    # combined 2x2 figure (default)
    python plotting/slope_plots.py

    # single base norm
    python plotting/slope_plots.py --norm "Stern Judging"

    # colour by competitive half-rule instead
    python plotting/slope_plots.py --color competitive
"""

from __future__ import annotations

import argparse
import ast
from itertools import chain
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from _common import (
    LEADING_FOUR,
    NORM_COLOURS,
    NORM_MAPPING,
    PLOTS_DIR,
    RESULTS_CSV,
    identify_base_norm,
)

SNARE_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = SNARE_ROOT / "outputs"

GAMMA_LEFT = 0.5
GAMMA_RIGHT = 1.0
THRESHOLD = 0.85

STRAT_COLS = [
    "AllD_Comp", "AllD_Coop", "Disc_Comp", "Disc_Coop",
    "pDisc_Comp", "pDisc_Coop", "AllC_Comp", "AllC_Coop",
]

EQUIL_COLOURS = {
    "Disc/Good":  "#1f77b4",   # blue
    "pDisc/Bad":  "#d62728",   # red
    "Bimodal":    "#9467bd",   # purple
}

# Canonical sweep CSVs: (gamma=1 file, gamma=0.5 file)
CANONICAL_FILES: dict[str, tuple[str, str]] = {
    "Stern Judging":   ("canonical_sweep_gamma1.csv",    "canonical_sweep_sj_gamma05.csv"),
    "Image Scoring":   ("canonical_sweep_is.csv",        "canonical_sweep_is_gamma05.csv"),
    "Shunning":        ("canonical_sweep_sh.csv",        "canonical_sweep_sh_gamma05.csv"),
    "Simple Standing": ("canonical_sweep_ss.csv",        "canonical_sweep_ss_gamma05.csv"),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _flatten_ebsn(ebsn) -> str:
    if isinstance(ebsn, str):
        ebsn = ast.literal_eval(ebsn)
    return "".join(str(int(b)) for b in chain.from_iterable(chain.from_iterable(ebsn)))


def _get_half_rules(vec8: str) -> tuple[str, str]:
    """Return (cooperative-norm-name, competitive-norm-name) from an 8-bit string."""
    comp = tuple(int(vec8[i]) for i in (0, 2, 4, 6))
    coop = tuple(int(vec8[i]) for i in (1, 3, 5, 7))
    return NORM_MAPPING.get(coop, "Other"), NORM_MAPPING.get(comp, "Other")


def _load_sweep(path: Path, gamma_value: float) -> pd.DataFrame:
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    df = df[pd.to_numeric(df["average_cooperation"], errors="coerce").notna()].copy()
    df["average_cooperation"] = pd.to_numeric(df["average_cooperation"])
    if df["average_cooperation"].max() > 1.1:
        df["average_cooperation"] /= 100.0
    df["gamma_value"] = gamma_value
    df["8bit"] = df["eb_social_norm"].apply(_flatten_ebsn)
    halves = df["8bit"].apply(_get_half_rules)
    df["Cooperative-Social Norm"] = [h[0] for h in halves]
    df["Competitive-Social Norm"] = [h[1] for h in halves]
    return df


def load_canonical_sweeps() -> dict[str, dict[float, pd.DataFrame]]:
    """Return {base_norm: {gamma: aggregated-DataFrame}} for all four norms."""
    data: dict[str, dict[float, pd.DataFrame]] = {}
    for norm, (f1, f05) in CANONICAL_FILES.items():
        d1  = _load_sweep(OUTPUTS_DIR / f1,  GAMMA_RIGHT)
        d05 = _load_sweep(OUTPUTS_DIR / f05, GAMMA_LEFT)
        data[norm] = {GAMMA_RIGHT: d1, GAMMA_LEFT: d05}
    return data


def _get_baseline_acr(base_norm: str) -> float | None:
    """Mean ACR for the base norm at gamma=0 (from results.csv)."""
    try:
        df = pd.read_csv(RESULTS_CSV, on_bad_lines="skip", engine="python")
        df["gamma_center"] = pd.to_numeric(df["gamma_center"], errors="coerce")
        df["average_cooperation"] = pd.to_numeric(df["average_cooperation"], errors="coerce")
        if df["average_cooperation"].max() > 1.1:
            df["average_cooperation"] /= 100.0
        df["base_norm_name"] = df["base_social_norm"].apply(identify_base_norm)
        rows = df[(df["base_norm_name"] == base_norm) & (df["gamma_center"] == 0)]
        if rows.empty:
            return None
        return float(rows["average_cooperation"].mean())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_for_slope(
    norm_data: dict[float, pd.DataFrame],
) -> pd.DataFrame:
    """Return a wide DataFrame with one row per EBSN.

    Columns: 8bit, Cooperative-Social Norm, Competitive-Social Norm,
             GAMMA_RIGHT (ACR at gamma=1, always present),
             GAMMA_LEFT  (ACR at gamma=0.5, NaN if not run at that gamma),
             equilibrium ("Disc/Good" or "pDisc/Bad", from mean G at gamma=1).
    """
    d1 = norm_data[GAMMA_RIGHT].copy()
    for col in STRAT_COLS + ["G"]:
        d1[col] = pd.to_numeric(d1[col], errors="coerce")

    agg_right = (
        d1.groupby(["8bit", "Cooperative-Social Norm", "Competitive-Social Norm"])
        .agg(average_cooperation=("average_cooperation", "mean"), mean_G=("G", "mean"))
        .reset_index()
        .rename(columns={"average_cooperation": GAMMA_RIGHT})
    )
    agg_right["equilibrium"] = agg_right["mean_G"].apply(
        lambda g: "Disc/Good" if g >= 0.5 else "pDisc/Bad"
    )

    agg_left = (
        norm_data[GAMMA_LEFT]
        .groupby("8bit")["average_cooperation"].mean()
        .reset_index()
        .rename(columns={"average_cooperation": GAMMA_LEFT})
    )
    wide = agg_right.merge(agg_left, on="8bit", how="left")
    wide["delta"] = wide[GAMMA_RIGHT] - wide[GAMMA_LEFT]
    return wide


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_slope(
    wide: pd.DataFrame,
    base_norm: str,
    *,
    color_by: str = "cooperative",
    threshold: float = THRESHOLD,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (5.5, 4.5),
    legend_loc: str = "upper left",
) -> plt.Figure:
    """Draw a slope plot for one base norm onto *ax* (or a new figure)."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if color_by == "equilibrium":
        color_col = "equilibrium"
        colour_map = EQUIL_COLOURS
    else:
        color_col = {
            "cooperative": "Cooperative-Social Norm",
            "competitive": "Competitive-Social Norm",
        }[color_by]
        colour_map = NORM_COLOURS

    has_both   = wide[GAMMA_LEFT].notna()
    is_elite   = wide[GAMMA_RIGHT] >= threshold
    BG = "#cccccc"

    # 1) Background dots at gamma=1 for ALL EBSNs (no line if no gamma=0.5 data)
    bg_only = wide[~has_both]
    ax.scatter(
        [GAMMA_RIGHT] * len(bg_only),
        bg_only[GAMMA_RIGHT],
        color=BG, s=18, alpha=0.5, edgecolors="white", linewidths=0.3, zorder=2,
    )

    # 2) Non-elite EBSNs that have both data points (grey lines)
    for _, row in wide[has_both & ~is_elite].iterrows():
        ax.plot([GAMMA_LEFT, GAMMA_RIGHT], [row[GAMMA_LEFT], row[GAMMA_RIGHT]],
                color=BG, alpha=0.45, linewidth=0.9, zorder=1)
        ax.scatter([GAMMA_LEFT, GAMMA_RIGHT], [row[GAMMA_LEFT], row[GAMMA_RIGHT]],
                   color=BG, s=22, alpha=0.6, edgecolors="white", linewidths=0.3, zorder=2)

    # 3) Elite EBSNs (coloured lines + dots)
    for _, row in wide[has_both & is_elite].iterrows():
        colour = colour_map.get(row[color_col], "#7f7f7f")
        ax.plot([GAMMA_LEFT, GAMMA_RIGHT], [row[GAMMA_LEFT], row[GAMMA_RIGHT]],
                color=colour, alpha=0.9, linewidth=1.9, zorder=4)
        ax.scatter([GAMMA_LEFT, GAMMA_RIGHT], [row[GAMMA_LEFT], row[GAMMA_RIGHT]],
                   color=colour, s=48, alpha=1.0, edgecolors="white", linewidths=0.5, zorder=5)

    # 4) Elite EBSNs without gamma=0.5 data: just a coloured dot on the right
    for _, row in wide[~has_both & is_elite].iterrows():
        colour = colour_map.get(row[color_col], "#7f7f7f")
        ax.scatter([GAMMA_RIGHT], [row[GAMMA_RIGHT]],
                   color=colour, s=48, alpha=1.0, edgecolors="white", linewidths=0.5, zorder=5)

    # 5) Baseline reference line
    baseline = _get_baseline_acr(base_norm)
    if baseline is not None:
        ax.axhline(baseline, linestyle="--", linewidth=1.1,
                   color="#444444", alpha=0.85, zorder=3)
        ax.text(GAMMA_RIGHT + 0.015, baseline,
                f"baseline\nACR={baseline:.2f}",
                ha="left", va="center", fontsize=6.5, color="#444444")

    # Stats
    n_elite_lines = int((has_both & is_elite).sum())
    n_non_elite   = int((~is_elite).sum())

    margin = 0.08
    ax.set_xlim(GAMMA_LEFT - margin, GAMMA_RIGHT + margin)
    ax.set_xticks([GAMMA_LEFT, GAMMA_RIGHT])
    ax.set_xticklabels([f"$\\gamma$ = {GAMMA_LEFT}", f"$\\gamma$ = {GAMMA_RIGHT}"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Average Cooperation Ratio $\\eta$")
    ax.set_title(
        f"{base_norm}\n"
        f"{n_elite_lines} elite EBSNs (≥{int(threshold*100)}%)",
        fontsize=9,
    )
    ax.grid(True, axis="y", alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for gx in (GAMMA_LEFT, GAMMA_RIGHT):
        ax.axvline(gx, color="#888888", linewidth=0.5, alpha=0.4, zorder=0)

    # Legend (elite EBSNs only)
    top_categories = list(dict.fromkeys(
        wide.loc[has_both & is_elite, color_col].tolist()
    ))
    handles = [
        mpatches.Patch(color=colour_map.get(n, "#7f7f7f"), label=n)
        for n in top_categories
    ]
    if n_non_elite:
        handles.append(mpatches.Patch(color=BG, label=f"non-elite ({n_non_elite})"))
    if handles:
        legend_title = "Equilibrium" if color_by == "equilibrium" else color_col.replace("-", " ")
        ax.legend(handles=handles, title=legend_title,
                  loc=legend_loc, fontsize=6.5, title_fontsize=7.5, frameon=False)

    if own_fig:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Combined 2x2 figure
# ---------------------------------------------------------------------------

def plot_combined(
    sweeps: dict[str, dict[float, pd.DataFrame]],
    *,
    color_by: str = "equilibrium",
    threshold: float = THRESHOLD,
    figsize: tuple[float, float] = (11, 9),
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    order = ["Stern Judging", "Simple Standing", "Shunning", "Image Scoring"]
    legend_locs = {"Stern Judging": "lower right"}
    for ax, norm in zip(axes.flat, order):
        wide = aggregate_for_slope(sweeps[norm])
        plot_slope(wide, norm, color_by=color_by, threshold=threshold, ax=ax,
                   legend_loc=legend_locs.get(norm, "upper left"))
    fig.suptitle(
        f"EB-norm extensions: ACR from $\\gamma$={GAMMA_LEFT} to $\\gamma$={GAMMA_RIGHT}"
        f"  (highlighted ≥ {int(threshold*100)}% at $\\gamma$={GAMMA_RIGHT})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        "--norm", choices=LEADING_FOUR, default=None,
        help="Single base norm (default: combined 2x2).",
    )
    parser.add_argument(
        "--color", choices=("equilibrium", "cooperative", "competitive"),
        default="equilibrium",
        help="Colour by equilibrium type (default), cooperative, or competitive half-rule.",
    )
    parser.add_argument(
        "--threshold", type=float, default=THRESHOLD,
        help=f"ACR threshold for 'elite' highlight (default {THRESHOLD}).",
    )
    args = parser.parse_args()

    sweeps = load_canonical_sweeps()

    if args.norm:
        wide = aggregate_for_slope(sweeps[args.norm])
        fig  = plot_slope(wide, args.norm, color_by=args.color, threshold=args.threshold)
        safe = args.norm.lower().replace(" ", "_")
        save_figure(fig, PLOTS_DIR / f"slope_{safe}_{args.color}.png")
    else:
        fig = plot_combined(sweeps, color_by=args.color, threshold=args.threshold)
        save_figure(fig, PLOTS_DIR / f"slope_combined_{args.color}.png")


if __name__ == "__main__":
    main()
