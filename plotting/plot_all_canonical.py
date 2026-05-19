"""
plot_all_canonical.py — Bifurcation heatmaps and elite analysis for all base norms.

For each base norm (SJ, IS, SH, SS):
  1. Loads outputs/canonical_sweep_{norm}.csv
  2. Aggregates and symmetry-reduces to canonical EBSNs
  3. Plots bifurcation heatmaps (top-30 and full) to EBSNs/figs/
  4. Reports elites (ACR >= threshold, default 90%)

Also generates:
  - EBSNs/figs/ebsn_elite_overview.png  (4-panel cross-norm comparison)
  - inputs/canonical_sweep_{is,sh,ss}_gamma05.yaml  (ready to run)
  - Appended rows in data/new_norms.csv for is_elite, sh_elite, ss_elite
  - Console table of cross-norm overlaps (EBSNs elite under > 1 base norm)

Usage:
    cd computational-model-snare
    python plotting/plot_all_canonical.py
    python plotting/plot_all_canonical.py --threshold 90 --top-k 15
    python plotting/plot_all_canonical.py --no-yaml --no-csv
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from itertools import chain
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
SNARE_ROOT = HERE.parent
FIGS_DIR = SNARE_ROOT.parent / "EBSNs" / "figs"
INPUTS_DIR = SNARE_ROOT / "inputs"
DATA_DIR = SNARE_ROOT / "data"
OUTPUTS_DIR = SNARE_ROOT / "outputs"

if str(SNARE_ROOT) not in sys.path:
    sys.path.insert(0, str(SNARE_ROOT))

from canonical_ebsns import (   # noqa: E402
    paper_mirror as mirror,
    canonical,
    simulation_canonical,
)

# ---------------------------------------------------------------------------
# Base norm metadata (model YAML order: [DB, DG, CB, CG])
# ---------------------------------------------------------------------------
BASE_NORMS = {
    "SJ": {
        "csv":        OUTPUTS_DIR / "canonical_sweep_gamma1.csv",
        "sn_yaml":    "[1, 0, 0, 1]",
        "4bit_orig":  "1001",
        "label":      "Stern Judging",
        "elite_name": "sj_elite",
        "yaml_out":   None,   # already exists
    },
    "IS": {
        "csv":        OUTPUTS_DIR / "canonical_sweep_is.csv",
        "sn_yaml":    "[0, 0, 1, 1]",
        "4bit_orig":  "0011",
        "label":      "Image Scoring",
        "elite_name": "is_elite",
        "yaml_out":   INPUTS_DIR / "canonical_sweep_is_gamma05.yaml",
    },
    "SH": {
        "csv":        OUTPUTS_DIR / "canonical_sweep_sh.csv",
        "sn_yaml":    "[0, 0, 0, 1]",
        "4bit_orig":  "0001",
        "label":      "Shunning",
        "elite_name": "sh_elite",
        "yaml_out":   INPUTS_DIR / "canonical_sweep_sh_gamma05.yaml",
    },
    "SS": {
        "csv":        OUTPUTS_DIR / "canonical_sweep_ss.csv",
        "sn_yaml":    "[1, 0, 1, 1]",
        "4bit_orig":  "1011",
        "label":      "Simple Standing",
        "elite_name": "ss_elite",
        "yaml_out":   INPUTS_DIR / "canonical_sweep_ss_gamma05.yaml",
    },
}

# ---------------------------------------------------------------------------
# Filter parameters (public-information regime)
# ---------------------------------------------------------------------------
FILTER = dict(q=1.0, gamma_center=1.0, Z_min=40, gens_min=400, chi=0.01, eps=0.01)

# ---------------------------------------------------------------------------
# Named 2nd-order norms in Ohtsuki (C1, C0, D1, D0) order
# ---------------------------------------------------------------------------
NAMED_NORMS = {
    (1, 1, 1, 1): "AG",
    (1, 1, 0, 1): "SS",
    (1, 1, 0, 0): "IS",
    (1, 0, 0, 1): "SJ",
    (1, 0, 0, 0): "SH",
    (0, 1, 1, 1): "pSH",
    (0, 0, 1, 1): "pIS",
    (0, 0, 1, 0): "pSS",
    (0, 0, 0, 0): "AB",
}

NORM_COLOURS = {
    "SJ":  "#1f77b4",
    "IS":  "#d62728",
    "SS":  "#2ca02c",
    "SH":  "#ff7f0e",
    "AG":  "#17becf",
    "AB":  "#bcbd22",
    "pSS": "#9467bd",
    "pIS": "#e377c2",
    "pSH": "#8c564b",
}
UNNAMED_COLOUR = "#cccccc"

_STRAT_EP_COLS = [
    "AllD_Comp", "AllD_Coop", "Disc_Comp", "Disc_Coop",
    "pDisc_Comp", "pDisc_Coop", "AllC_Comp", "AllC_Coop",
]

# ---------------------------------------------------------------------------
# Bit helpers
# ---------------------------------------------------------------------------
def parse_ebsn(s: str) -> tuple:
    nested = ast.literal_eval(s)
    return tuple(int(b) for b in chain.from_iterable(chain.from_iterable(nested)))


def coop_half(bits: tuple) -> tuple:
    """Cooperative half in Ohtsuki order (C1, C0, D1, D0) = (CGO, CBO, DGO, DBO)."""
    return (bits[7], bits[5], bits[3], bits[1])


def comp_half(bits: tuple) -> tuple:
    """Competitive half in Ohtsuki order (C1, C0, D1, D0) = (CGM, CBM, DGM, DBM)."""
    return (bits[6], bits[4], bits[2], bits[0])


def half_name(half: tuple) -> str:
    return NAMED_NORMS.get(half, "U" + str(int("".join(map(str, half)), 2)))


def half_colour(name: str) -> str:
    return NORM_COLOURS.get(name, UNNAMED_COLOUR)


def is_trivial(bits: tuple) -> bool:
    return (bits[0] == bits[1] and bits[2] == bits[3]
            and bits[4] == bits[5] and bits[6] == bits[7])


def bits_to_str(bits: tuple) -> str:
    return "".join(map(str, bits)).zfill(8)


def ebsn_label(row) -> str:
    """Short human-readable label for a row from the aggregated DataFrame."""
    if row.bimodal:
        return (f"{row.coop_name_disc}|{row.comp_name_disc}"
                f" / {row.coop_name_pdisc}|{row.comp_name_pdisc}")
    return f"{row.coop_name}|{row.comp_name}"


# ---------------------------------------------------------------------------
# Aggregation (adapted from plot_ebsn_heatmap.py)
# ---------------------------------------------------------------------------
def load_and_aggregate(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, on_bad_lines="skip", engine="python")
    df = df[df["Z"] != "Z"]   # drop repeated header rows in multi-run CSVs

    numeric_cols = (["q", "gamma_center", "Z", "gens", "chi", "eps",
                     "average_cooperation", "G"] + _STRAT_EP_COLS)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    f = df[
        (df["q"]            == FILTER["q"])
        & (df["gamma_center"] == FILTER["gamma_center"])
        & (df["Z"]           >= FILTER["Z_min"])
        & (df["gens"]        >= FILTER["gens_min"])
        & (df["chi"]         == FILTER["chi"])
        & (df["eps"]         == FILTER["eps"])
    ].copy()

    f["bits"]           = f["eb_social_norm"].apply(parse_ebsn)
    f                   = f[~f["bits"].apply(is_trivial)]
    f["canonical_bits"] = f["bits"].apply(canonical)

    avail_strat = [c for c in _STRAT_EP_COLS if c in f.columns]
    if avail_strat:
        f["dom_col"]   = f[avail_strat].idxmax(axis=1)
        f["dom_strat"] = f["dom_col"].str.split("_").str[0]
        f["dom_ep"]    = f["dom_col"].str.split("_").str[1]

    def _mode(series, default):
        m = series.mode()
        return m.iloc[0] if len(m) else default

    def agg_group(grp):
        n     = len(grp)
        ACR   = grp["average_cooperation"].mean()
        sd    = grp["average_cooperation"].std()
        avg_G = grp["G"].mean() if "G" in grp.columns else 0.5

        if not avail_strat:
            return pd.Series(dict(
                ACR=ACR, ACR_sd=sd, n_runs=n, avg_G=avg_G,
                bimodal=False, maj_strat="Disc", maj_ep="Coop",
                disc_ep="Coop",  avg_G_disc=0.7,
                pdisc_ep="Comp", avg_G_pdisc=0.3,
            ))

        disc  = (grp["dom_strat"] == "Disc").sum()
        pdisc = (grp["dom_strat"] == "pDisc").sum()
        dp    = disc + pdisc
        p     = (binomtest(int(max(disc, pdisc)), int(dp), 0.5,
                           alternative="greater").pvalue
                 if dp > 0 else 1.0)

        bimodal   = p >= 0.05
        maj_strat = "Disc" if disc >= pdisc else "pDisc"
        maj_ep    = _mode(grp.loc[grp["dom_strat"] == maj_strat, "dom_ep"], "Coop")

        disc_rows  = grp[grp["dom_strat"] == "Disc"]
        pdisc_rows = grp[grp["dom_strat"] == "pDisc"]
        disc_ep    = _mode(disc_rows["dom_ep"],  "Coop")
        pdisc_ep   = _mode(pdisc_rows["dom_ep"], "Comp")
        avg_G_disc  = disc_rows["G"].mean()  if len(disc_rows)  else 0.7
        avg_G_pdisc = pdisc_rows["G"].mean() if len(pdisc_rows) else 0.3

        return pd.Series(dict(
            ACR=ACR, ACR_sd=sd, n_runs=n, avg_G=avg_G,
            bimodal=bimodal, maj_strat=maj_strat, maj_ep=maj_ep,
            disc_ep=disc_ep,   avg_G_disc=avg_G_disc,
            pdisc_ep=pdisc_ep, avg_G_pdisc=avg_G_pdisc,
        ))

    agg = (
        f.groupby("canonical_bits")
        .apply(agg_group)
        .reset_index()
        .rename(columns={"canonical_bits": "old_bits"})
    )
    agg["ACR_se"] = agg["ACR_sd"] / np.sqrt(agg["n_runs"])

    # Simulation-driven canonical labelling
    def sim_relabel(row):
        if row["bimodal"]:
            return canonical(row["old_bits"])
        target_rep = 1 if row["avg_G"] >= 0.5 else 0
        return simulation_canonical(row["old_bits"], row["maj_strat"],
                                    row["maj_ep"], target_rep)

    agg["bits"] = agg.apply(sim_relabel, axis=1)

    def _sim_bits(row, strat, ep_col, g_col, default_ep, default_g):
        ep    = row[ep_col] or default_ep
        avg_g = row[g_col]
        t_rep = 1 if (avg_g if not pd.isna(avg_g) else default_g) >= 0.5 else 0
        return simulation_canonical(row["old_bits"], strat, ep, t_rep)

    agg["bits_disc"]  = agg.apply(
        lambda r: _sim_bits(r, "Disc",  "disc_ep",  "avg_G_disc",  "Coop", 0.7), axis=1)
    agg["bits_pdisc"] = agg.apply(
        lambda r: _sim_bits(r, "pDisc", "pdisc_ep", "avg_G_pdisc", "Comp", 0.3), axis=1)

    # Force bits_pdisc = mirror(bits_disc) when both attractors map to same member
    same_mask = agg["bimodal"] & (agg["bits_disc"] == agg["bits_pdisc"])
    agg.loc[same_mask, "bits_pdisc"] = (
        agg.loc[same_mask, "bits_disc"].apply(mirror)
    )

    for col, src in [
        ("coop_name", "bits"), ("comp_name", "bits"),
        ("coop_name_disc",  "bits_disc"),  ("comp_name_disc",  "bits_disc"),
        ("coop_name_pdisc", "bits_pdisc"), ("comp_name_pdisc", "bits_pdisc"),
    ]:
        fn = coop_half if col.startswith("coop") else comp_half
        agg[col] = agg[src].apply(lambda b, f=fn: half_name(f(b)))

    return agg.sort_values("ACR", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plot: bifurcation heatmap
# ---------------------------------------------------------------------------
def plot_heatmap(agg: pd.DataFrame, out_path: Path, *,
                 max_rows: int | None = None,
                 top_divider_at: int | None = None,
                 row_height: float = 0.18,
                 title: str | None = None) -> None:
    if max_rows is not None:
        agg = agg.head(max_rows).reset_index(drop=True)
    n = len(agg)

    cmap       = cm.get_cmap("YlGnBu")
    norm       = mcolors.Normalize(vmin=0, vmax=100)
    bar_colours = [cmap(norm(v)) for v in agg["ACR"]]

    fig, (ax_coop, ax_comp, ax_acr) = plt.subplots(
        1, 3,
        figsize=(8, max(4, n * row_height)),
        gridspec_kw={"width_ratios": [1.4, 1.4, 3.4], "wspace": 0.05},
        sharey=True,
    )

    y             = np.arange(n)
    bimodal_flags = agg.get("bimodal", pd.Series([False] * n))

    def draw_chips(ax, name_col, name_disc_col, name_pdisc_col, title_text):
        for i in range(n):
            bm   = bimodal_flags.iloc[i]
            name = agg[name_col].iloc[i]
            if not bm:
                ax.add_patch(mpatches.Rectangle(
                    (0, i - 0.4), 1, 0.8,
                    facecolor=half_colour(name), edgecolor="white", linewidth=0.5))
                ax.text(0.5, i, name, ha="center", va="center", fontsize=7,
                        color="white" if name in NORM_COLOURS else "black")
            else:
                nd  = agg[name_disc_col].iloc[i]
                npd = agg[name_pdisc_col].iloc[i]
                ax.add_patch(mpatches.Rectangle(
                    (0, i - 0.4), 0.5, 0.8,
                    facecolor=half_colour(nd), edgecolor="white", linewidth=0.5))
                ax.text(0.25, i, nd, ha="center", va="center", fontsize=6,
                        color="white" if nd in NORM_COLOURS else "black")
                ax.add_patch(mpatches.Rectangle(
                    (0.5, i - 0.4), 0.5, 0.8,
                    facecolor=half_colour(npd), edgecolor="white", linewidth=0.5))
                ax.text(0.75, i, npd, ha="center", va="center", fontsize=6,
                        color="white" if npd in NORM_COLOURS else "black")
                ax.plot([0.5, 0.5], [i - 0.39, i + 0.39],
                        color="white", lw=1.2, zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, n - 0.5)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_title(title_text, fontsize=9)

    draw_chips(ax_coop, "coop_name", "coop_name_disc", "coop_name_pdisc",
               "Cooperative half\n(O bits)")
    draw_chips(ax_comp, "comp_name", "comp_name_disc", "comp_name_pdisc",
               "Competitive half\n(M bits)")

    ax_acr.barh(y, agg["ACR"], xerr=agg["ACR_se"],
                color=bar_colours, height=0.75,
                edgecolor="#222222", linewidth=0.4,
                error_kw=dict(lw=0.7, capsize=2, ecolor="#555555"))
    ax_acr.set_xlim(0, 100)
    ax_acr.set_xlabel("Average Cooperation Ratio (%, mean ± SE)")
    ax_acr.set_title(f"n = {n} canonical EBSNs", fontsize=9)
    ax_acr.grid(True, axis="x", alpha=0.3)
    for spine in ("top", "right"):
        ax_acr.spines[spine].set_visible(False)

    if top_divider_at is not None and top_divider_at < n:
        y_line = top_divider_at - 0.5
        for ax in (ax_coop, ax_comp, ax_acr):
            ax.axhline(y_line, color="#d62728", linewidth=1.2,
                       linestyle="--", alpha=0.85, zorder=5)
        ax_acr.text(99, y_line - 0.4, f"elite ({top_divider_at})",
                    ha="right", va="bottom", fontsize=7,
                    color="#d62728", style="italic")

    label_step = 1 if n <= 35 else 5
    yticks = [i for i in range(n) if (i + 1) % label_step == 0 or i == 0]
    ax_coop.set_yticks(yticks)
    ax_coop.set_yticklabels([f"#{i+1}" for i in yticks],
                             fontsize=6 if n > 60 else 7)

    legend_handles = [
        mpatches.Patch(color=col, label=name)
        for name, col in NORM_COLOURS.items()
    ] + [mpatches.Patch(color=UNNAMED_COLOUR, label="U.. (unnamed)")]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02), fontsize=8, frameon=False)

    n_bimodal = int(bimodal_flags.sum())
    if n_bimodal > 0:
        fig.text(
            0.5, -0.065,
            f"Split chips: bistable — Disc/G attractor (left) | pDisc/B attractor (right)"
            f"  [{n_bimodal} rows, binomial p >= 0.05]",
            ha="center", fontsize=7, color="#444444", style="italic",
        )

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = ax_acr.inset_axes([0.55, -0.10, 0.4, 0.015])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("ACR colour scale", fontsize=7)

    fig.suptitle(title or "Bifurcation structure of EBSNs", fontsize=11, y=0.995)
    fig.subplots_adjust(top=0.95, bottom=0.06)
    os.makedirs(out_path.parent, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved -> {out_path.name}")


# ---------------------------------------------------------------------------
# Plot: 4-panel cross-norm comparison
# ---------------------------------------------------------------------------
def plot_elite_overview(all_agg: dict[str, pd.DataFrame], out_path: Path,
                        top_k: int = 15, threshold: float = 90.0) -> None:
    """Horizontal bar charts (top-K per base norm) in a 2×2 grid.

    Bars are coloured by the cooperative half-rule. A vertical dashed line
    marks the elite threshold, and a tick annotation marks the last elite bar.
    """
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10),
        gridspec_kw={"wspace": 0.38, "hspace": 0.40},
    )
    panel_order = [("SJ", axes[0, 0]), ("IS", axes[0, 1]),
                   ("SH", axes[1, 0]), ("SS", axes[1, 1])]

    for norm_key, ax in panel_order:
        agg = all_agg[norm_key]
        top = agg.head(top_k).copy().reset_index(drop=True)

        # Build y-axis labels
        labels = []
        for r in top.itertuples():
            labels.append(ebsn_label(r))

        colours = [NORM_COLOURS.get(n, UNNAMED_COLOUR) for n in top["coop_name"]]
        y = np.arange(len(top))

        ax.barh(y, top["ACR"], xerr=top["ACR_se"],
                color=colours, height=0.75, edgecolor="#222222", linewidth=0.3,
                error_kw=dict(lw=0.6, capsize=2, ecolor="#555555"))
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("ACR (%)", fontsize=9)
        ax.axvline(threshold, color="#d62728", linewidth=1.1, linestyle="--",
                   alpha=0.85)
        ax.text(threshold + 0.5, len(top) - 0.5, f"{threshold:.0f}%",
                fontsize=7, color="#d62728", va="bottom")
        ax.grid(True, axis="x", alpha=0.25)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        # Annotate ACR values on bars
        for i, row in enumerate(top.itertuples()):
            ax.text(min(row.ACR + row.ACR_se + 0.5, 99), i,
                    f"{row.ACR:.1f}", va="center", fontsize=6.5, color="#333333")

        n_elite = int((agg["ACR"] >= threshold).sum())
        n_total = len(agg)
        ax.set_title(
            f"{norm_key} — {BASE_NORMS[norm_key]['label']}\n"
            f"top {top_k} of {n_total} canonical EBSNs  ·  {n_elite} elite "
            f"(ACR >= {threshold:.0f}%)",
            fontsize=9.5,
        )

    # Shared colour legend
    all_coop_names = {
        n
        for agg in all_agg.values()
        for n in agg["coop_name"].values
    }
    handles = [
        mpatches.Patch(color=NORM_COLOURS[name], label=name)
        for name in NORM_COLOURS
        if name in all_coop_names
    ] + [mpatches.Patch(color=UNNAMED_COLOUR, label="U.. (unnamed)")]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.01), fontsize=8.5, frameon=False,
               title="Cooperative half-rule colour")

    fig.suptitle(
        "Elite EBSNs across base norms  (γ = 1, public information)",
        fontsize=13, y=1.005,
    )
    os.makedirs(out_path.parent, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved -> {out_path.name}")


# ---------------------------------------------------------------------------
# YAML generation
# ---------------------------------------------------------------------------
_YAML_TEMPLATE = """\
# {norm_key}-elite EBSNs at gamma=0.5 ({n_elite} norms with ACR >= {threshold:.0f}% \
under {label} at gamma=1).
#
# Runs only the {n_elite} elite EBSN+{norm_key} combinations identified from {csv_name}.
# Used to build the slopegraph (gamma=1 vs gamma=0.5) for the EBSNs paper.
#
# Usage:
#   python SNARE.py inputs/canonical_sweep_{norm_lower}_gamma05.yaml

running:
  runs: 20
  cores: all
  plotting: false
  output_file: "canonical_sweep_{norm_lower}_gamma05.csv"

simulation:
  norm: {elite_name}         # loads {n_elite} EBSNs from data/new_norms.csv
  ebsn:                  # overridden by norm mode
  sn: {sn_yaml}      # {norm_key} - also set per-variant via 4bit_orig in CSV

  # Public-information regime
  z: 40
  observability: 1.0
  consensus_thresh: 0
  non_consensus_strategy: emotion

  # Errors
  chi: 0.01
  eps: 0.01
  alpha: 0
  xi: 0

  # Payoffs
  benefit: 5
  cost: 1

  # Evolutionary dynamics
  beta: 1
  mu: 1
  generations: 1600
  convergence period: 0.25

  # Gamma fixed at 0.5
  gamma_min: 0.5
  gamma_max: 0.5
  gamma_delta: 0
  gamma_gaussian_n: 0.5
"""


def make_yaml(norm_key: str, elites: pd.DataFrame, threshold: float) -> str:
    info = BASE_NORMS[norm_key]
    return _YAML_TEMPLATE.format(
        norm_key=norm_key,
        norm_lower=norm_key.lower(),
        label=info["label"],
        n_elite=len(elites),
        elite_name=info["elite_name"],
        sn_yaml=info["sn_yaml"],
        csv_name=Path(info["csv"]).name,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# new_norms.csv rows
# ---------------------------------------------------------------------------
def make_new_norms_rows(norm_key: str, elites: pd.DataFrame) -> list[dict]:
    info = BASE_NORMS[norm_key]
    rows = []
    for i, row in enumerate(elites.itertuples(), start=1):
        rows.append({
            "norm":              info["elite_name"],
            "variant_id":        f"{info['elite_name']}_v{i}",
            "4bit_orig":         info["4bit_orig"],
            "8bit_vector":       bits_to_str(row.bits),
            "DB_mean":  0, "DB_nice": 0,
            "DG_mean":  0, "DG_nice": 0,
            "CB_mean":  0, "CB_nice": 0,
            "CG_mean":  0, "CG_nice": 0,
            "Emotion_Leniency":  0,
        })
    return rows


# ---------------------------------------------------------------------------
# Cross-norm overlap analysis
# ---------------------------------------------------------------------------
def print_cross_norm_overlaps(elite_bits: dict[str, set]) -> None:
    all_bits = set()
    for s in elite_bits.values():
        all_bits |= s

    print("\n  Cross-norm overlaps (canonical bit string -> base norms where it's elite):")
    found_any = False
    for b in sorted(all_bits):
        norms_with = [k for k, s in elite_bits.items() if b in s]
        if len(norms_with) > 1:
            found_any = True
            # Compute name from first norm's agg (all should agree on canonical label)
            print(f"    {b}  ->  {', '.join(norms_with)}")
    if not found_any:
        print("    (no overlap — each elite EBSN is elite under exactly one base norm)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=90.0,
                        help="ACR threshold for elite status (default: 90)")
    parser.add_argument("--top-k", type=int, default=15,
                        help="Top-K shown in the comparison figure (default: 15)")
    parser.add_argument("--no-yaml", action="store_true",
                        help="Skip YAML generation")
    parser.add_argument("--no-csv", action="store_true",
                        help="Skip new_norms.csv update")
    args = parser.parse_args()

    all_agg:   dict[str, pd.DataFrame] = {}
    elite_bits: dict[str, set]         = {}
    new_norms_rows: list[dict]         = []

    print("=" * 70)
    print(f"Canonical EBSN analysis  |  elite threshold: ACR >= {args.threshold:.0f}%")
    print("=" * 70)

    for norm_key, info in BASE_NORMS.items():
        csv_path = Path(info["csv"])
        print(f"\n[{norm_key}]  {csv_path.name}")

        agg = load_and_aggregate(csv_path)
        all_agg[norm_key] = agg

        n_total   = len(agg)
        n_bimodal = int(agg["bimodal"].sum()) if "bimodal" in agg.columns else 0
        elites    = agg[agg["ACR"] >= args.threshold].copy()
        n_elite   = len(elites)

        print(f"  Canonical EBSNs: {n_total}   bimodal: {n_bimodal}   "
              f"elite (>= {args.threshold:.0f}%): {n_elite}")

        # Elite table
        if not elites.empty:
            print(f"  {'Rank':>4}  {'ACR%':>7}  {'SE':>5}  {'bits':>10}  label")
            for rank, r in enumerate(elites.itertuples(), start=1):
                print(f"  #{rank:3d}  {r.ACR:7.2f}  {r.ACR_se:5.2f}"
                      f"  {bits_to_str(r.bits):>10}  {ebsn_label(r)}")

        elite_bits[norm_key] = {bits_to_str(r.bits) for r in elites.itertuples()}

        # Bifurcation heatmaps
        norm_lower = norm_key.lower()
        fig_prefix = ("ebsn_bifurcation"
                      if norm_key == "SJ"
                      else f"ebsn_bifurcation_{norm_lower}")

        n_show = min(30, n_total)
        print(f"  Generating heatmaps ...")
        plot_heatmap(
            agg, FIGS_DIR / f"{fig_prefix}_top30.png",
            max_rows=n_show,
            top_divider_at=n_elite if 0 < n_elite < n_show else None,
            row_height=0.30,
            title=(f"Top {n_show} EBSNs by ACR  —  base norm: {norm_key} "
                   f"({info['label']})"),
        )
        plot_heatmap(
            agg, FIGS_DIR / f"{fig_prefix}_full.png",
            max_rows=None,
            top_divider_at=n_elite if 0 < n_elite < n_total else None,
            row_height=0.20,
            title=(f"All {n_total} canonical EBSNs  —  base norm: {norm_key} "
                   f"({info['label']})"),
        )

        # YAML
        if not args.no_yaml and info["yaml_out"] is not None and n_elite > 0:
            yaml_text = make_yaml(norm_key, elites, args.threshold)
            yaml_path = Path(info["yaml_out"])
            yaml_path.parent.mkdir(parents=True, exist_ok=True)
            yaml_path.write_text(yaml_text, encoding="utf-8")
            print(f"    Wrote YAML -> {yaml_path.name}")

        # Collect new_norms rows
        if not args.no_csv and info["yaml_out"] is not None and n_elite > 0:
            new_norms_rows.extend(make_new_norms_rows(norm_key, elites))

    # Cross-norm overlap
    print_cross_norm_overlaps(elite_bits)

    # 4-panel comparison figure
    print(f"\nGenerating overview figure (top {args.top_k} per base norm) ...")
    plot_elite_overview(
        all_agg,
        FIGS_DIR / "ebsn_elite_overview.png",
        top_k=args.top_k,
        threshold=args.threshold,
    )

    # Update new_norms.csv
    if not args.no_csv and new_norms_rows:
        norms_path = DATA_DIR / "new_norms.csv"
        existing   = pd.read_csv(norms_path)
        names_to_replace = {r["norm"] for r in new_norms_rows}
        existing = existing[~existing["norm"].isin(names_to_replace)]
        updated  = pd.concat(
            [existing, pd.DataFrame(new_norms_rows)], ignore_index=True
        )
        updated.to_csv(norms_path, index=False)
        n_added = len(new_norms_rows)
        print(f"\nUpdated {norms_path.name}: +{n_added} rows "
              f"({', '.join(sorted(names_to_replace))})")

    print("\nDone.")
    print("Next: run the gamma=0.5 YAMLs, then generate the slopegraph.")


if __name__ == "__main__":
    main()
