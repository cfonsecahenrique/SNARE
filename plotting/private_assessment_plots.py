"""
Private-assessment robustness of all 29 elite EBSNs.

Data: outputs/elite_private_assessment.csv
  29 elite EBSNs × q∈{0.4, 0.6, 0.8} × γ∈{0.5, 1} × 20 runs

Produces:
  plotting/plots/private_assessment_slopegraph.png
      Three-panel slopegraph (one column per q level).
      Within each panel: γ=0.5 (left) vs γ=1 (right), coloured by equilibrium.

  plotting/plots/private_assessment_robustness.png
      ACR vs q, one panel per equilibrium type.
      Solid = γ=1, dashed = γ=0.5.  Thin lines = individual EBSNs,
      thick line = group mean.  Horizontal dotted = q=1 reference from
      canonical sweeps (acr_1 from elite_table.csv).

Usage:
    python plotting/private_assessment_plots.py
"""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

from slope_plots import OUTPUTS_DIR, EQUIL_COLOURS, _load_sweep, GAMMA_LEFT, GAMMA_RIGHT
from _common import PLOTS_DIR

# ── Constants ────────────────────────────────────────────────────────────────

Q_LEVELS   = [0.4, 0.6, 0.8, 1.0]
GAMMAS     = [0.5, 1.0]

GAMMA05_FILES = {
    "Stern Judging":   "canonical_sweep_sj_gamma05.csv",
    "Image Scoring":   "canonical_sweep_is_gamma05.csv",
    "Shunning":        "canonical_sweep_sh_gamma05.csv",
    "Simple Standing": "canonical_sweep_ss_gamma05.csv",
}
EQUIL_ORDER = ["Disc/Good", "pDisc/Bad", "Bimodal"]

BASE_NORM_NAMES = {
    "[[1, 0], [0, 1]]": "Stern Judging",
    "[[0, 0], [0, 1]]": "Shunning",
    "[[1, 0], [1, 1]]": "Simple Standing",
    "[[0, 0], [1, 1]]": "Image Scoring",
}

# ── Parsing helpers ──────────────────────────────────────────────────────────

def _parse_base_norm(s: str) -> str:
    return BASE_NORM_NAMES.get(s.strip(), s)


def _ebsn_to_8bit(s: str) -> str:
    """Convert the CSV repr of an EBSN to an 8-bit string.

    CSV format: [[(DBm,DBn),(DGm,DGn)], [(CBm,CBn),(CGm,CGn)]]
    8-bit order: DBm DBn DGm DGn CBm CBn CGm CGn
    """
    ebsn = ast.literal_eval(s)
    bits = []
    for action_row in ebsn:         # Defect row, then Cooperate row
        for rep_pair in action_row: # Bad-rep cell, then Good-rep cell
            bits.extend([rep_pair[0], rep_pair[1]])  # (competitive, cooperative)
    return "".join(str(b) for b in bits)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    raw = pd.read_csv(OUTPUTS_DIR / "elite_private_assessment.csv")
    df  = raw[raw["q"] != "q"].copy()   # drop repeated header rows

    for col in ["q", "gamma_center", "average_cooperation", "G"]:
        df[col] = pd.to_numeric(df[col])

    df["base_norm"] = df["base_social_norm"].map(_parse_base_norm)
    df["8bit"]      = df["eb_social_norm"].map(_ebsn_to_8bit)

    et = pd.read_csv(
        Path(__file__).parent / "plots" / "elite_table.csv",
        dtype={"8bit": str},
    )
    et["8bit"] = et["8bit"].str.zfill(8)

    df = df.merge(
        et[["8bit", "base_norm", "label", "equilibrium", "acr_1"]],
        on=["8bit", "base_norm"],
        how="left",
    )
    df["acr"] = df["average_cooperation"] / 100.0
    return df


def load_q1_data(et: pd.DataFrame) -> pd.DataFrame:
    """Build q=1 rows for both gamma=0.5 and gamma=1 from canonical sweeps."""
    rows = []

    # gamma=1: acr_1 already in elite table (from canonical_sweep_gamma1 runs)
    for _, r in et.iterrows():
        rows.append(dict(
            label=r["label"], base_norm=r["base_norm"], **{"8bit": r["8bit"]},
            equilibrium=r["equilibrium"], acr_1=r["acr_1"],
            q=1.0, gamma_center=1.0, acr=r["acr_1"],
        ))

    # gamma=0.5: load each canonical gamma05 sweep and filter to elite EBSNs
    for base_norm, fname in GAMMA05_FILES.items():
        d = _load_sweep(OUTPUTS_DIR / fname, GAMMA_LEFT)
        d["average_cooperation"] = pd.to_numeric(d["average_cooperation"], errors="coerce")
        d["8bit"] = d["8bit"].astype(str).str.zfill(8)

        et_base = et[et["base_norm"] == base_norm][["8bit", "label", "equilibrium", "acr_1"]]
        merged  = d.merge(et_base, on="8bit", how="inner")
        per_ebsn = (
            merged.groupby(["8bit", "label", "equilibrium", "acr_1"])
            ["average_cooperation"].mean().reset_index()
        )
        for _, r in per_ebsn.iterrows():
            rows.append(dict(
                label=r["label"], base_norm=base_norm, **{"8bit": r["8bit"]},
                equilibrium=r["equilibrium"], acr_1=r["acr_1"],
                q=1.0, gamma_center=0.5, acr=r["average_cooperation"],
            ))

    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            ["label", "base_norm", "8bit", "equilibrium", "acr_1", "q", "gamma_center"]
        )["acr"]
        .mean()
        .reset_index()
    )


# ── Figure 1: Three-panel slopegraph ─────────────────────────────────────────

def make_slopegraph(agg: pd.DataFrame) -> plt.Figure:
    """γ=0.5 → γ=1 slopegraph at each q level (4 columns, including q=1)."""
    fig, axes = plt.subplots(1, 4, figsize=(17, 6), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    for ax, q in zip(axes, Q_LEVELS):
        sub = agg[agg["q"] == q]

        for _, row in sub[sub["gamma_center"] == 0.5].iterrows():
            match = sub[
                (sub["label"] == row["label"]) & (sub["base_norm"] == row["base_norm"])
                & (sub["gamma_center"] == 1.0)
            ]
            if match.empty:
                continue
            y_left  = row["acr"] * 100
            y_right = match["acr"].values[0] * 100
            eq      = row["equilibrium"]
            col     = EQUIL_COLOURS.get(eq, "#888888")
            ax.plot([0, 1], [y_left, y_right], color=col, alpha=0.7, lw=1.4)
            ax.scatter([0], [y_left],  color=col, s=22, zorder=3)
            ax.scatter([1], [y_right], color=col, s=22, zorder=3)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["γ = 0.5", "γ = 1"], fontsize=10)
        ax.set_xlim(-0.25, 1.25)
        ax.set_title(f"q = {q}", fontsize=11, fontweight="bold")
        ax.axhline(85, color="grey", lw=0.8, ls="--", alpha=0.5)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", lw=0.4, alpha=0.4)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.tick_params(axis="x", length=0)

    axes[0].set_ylabel("ACR (%)", fontsize=10)

    # Legend
    handles = [
        mlines.Line2D([], [], color=EQUIL_COLOURS[eq], lw=2, label=eq)
        for eq in EQUIL_ORDER if eq in EQUIL_COLOURS
    ]
    handles.append(
        mlines.Line2D([], [], color="grey", lw=0.8, ls="--", label="85% threshold")
    )
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Elite EBSNs under private assessment: γ=0.5 vs γ=1\n"
        "each line = one (EBSN, base norm) pair  |  q = 1.0 from canonical sweeps",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    return fig


# ── Figure 2: Robustness line plot ───────────────────────────────────────────

def make_robustness_plot(agg: pd.DataFrame) -> plt.Figure:
    """ACR vs q per equilibrium type; solid = γ=1, dashed = γ=0.5."""
    n_eq  = len(EQUIL_ORDER)
    fig, axes = plt.subplots(1, n_eq, figsize=(13, 5), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    ls_map = {0.5: "--", 1.0: "-"}
    lw_ind = 0.9   # individual EBSN lines
    lw_avg = 2.5   # group mean

    for ax, eq in zip(axes, EQUIL_ORDER):
        col = EQUIL_COLOURS.get(eq, "#888888")
        sub = agg[agg["equilibrium"] == eq]

        for gamma in GAMMAS:
            g_sub = sub[sub["gamma_center"] == gamma]
            # Individual EBSN lines
            for (label, base), grp in g_sub.groupby(["label", "base_norm"]):
                grp_sorted = grp.sort_values("q")
                ax.plot(
                    grp_sorted["q"],
                    grp_sorted["acr"] * 100,
                    color=col, alpha=0.25, lw=lw_ind,
                    ls=ls_map[gamma], zorder=1,
                )
            # Group mean
            mean_line = g_sub.groupby("q")["acr"].mean().reset_index().sort_values("q")
            ax.plot(
                mean_line["q"],
                mean_line["acr"] * 100,
                color=col, lw=lw_avg, ls=ls_map[gamma], zorder=3,
            )

        ax.set_title(eq, fontsize=10, fontweight="bold",
                     color=col)
        ax.set_xlabel("Observability q", fontsize=9)
        ax.set_xticks([0.4, 0.6, 0.8, 1.0])
        ax.set_xlim(0.33, 1.07)
        ax.set_ylim(0, 100)
        ax.axhline(85, color="grey", lw=0.8, ls=":", alpha=0.5)
        ax.grid(axis="y", lw=0.4, alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("ACR (%)", fontsize=10)

    # Legend (one shared)
    solid  = mlines.Line2D([], [], color="grey", lw=2,   ls="-",  label="γ = 1")
    dashed = mlines.Line2D([], [], color="grey", lw=2,   ls="--", label="γ = 0.5")
    thresh = mlines.Line2D([], [], color="grey", lw=0.8, ls=":",  label="85% threshold")
    fig.legend(
        handles=[solid, dashed, thresh],
        loc="lower center", ncol=3, fontsize=9,
        frameon=False, bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        "Robustness of elite EBSNs under private assessment\n"
        "thin = individual (EBSN, base norm) · thick = group mean",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    df  = load_data()
    agg = aggregate(df)

    # Append q=1 canonical data
    et = pd.read_csv(
        Path(__file__).parent / "plots" / "elite_table.csv",
        dtype={"8bit": str},
    )
    et["8bit"] = et["8bit"].str.zfill(8)
    q1  = load_q1_data(et)
    agg = pd.concat([agg, q1], ignore_index=True)

    print(f"Loaded {len(df)} runs, {agg['label'].nunique()} unique EBSNs across "
          f"{agg['base_norm'].nunique()} base norms")
    print("Equilibrium counts (unique EBSN×base pairs):")
    print(agg[["label","base_norm","equilibrium"]].drop_duplicates()
          ["equilibrium"].value_counts().to_string())

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig1 = make_slopegraph(agg)
    p1   = PLOTS_DIR / "private_assessment_slopegraph.png"
    fig1.savefig(p1, dpi=180, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {p1}")

    fig2 = make_robustness_plot(agg)
    p2   = PLOTS_DIR / "private_assessment_robustness.png"
    fig2.savefig(p2, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {p2}")


if __name__ == "__main__":
    main()
