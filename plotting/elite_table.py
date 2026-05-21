"""
Summary table of all elite EBSNs (ACR >= 85% at gamma=1) across all four
leading base norms.

Layout: two sections separated by a banner row —
  1. Disc/Good equilibrium  (blue)
  2. pDisc/Bad equilibrium  (red)

Within each section rows are sorted by ACR descending and grouped by base norm.
The G column is colour-graded (blue=high reputation, red=low) to make the
reputation level immediately legible without reading the number.

Outputs:
  plotting/plots/elite_table.png
  plotting/plots/elite_table.csv

Usage:
    python plotting/elite_table.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from slope_plots import (
    OUTPUTS_DIR, THRESHOLD, STRAT_COLS, EQUIL_COLOURS,
    _load_sweep, GAMMA_LEFT, GAMMA_RIGHT,
)
from _common import PLOTS_DIR

CANONICAL_FILES = {
    "Stern Judging":   ("canonical_sweep_gamma1.csv",  "canonical_sweep_sj_gamma05.csv"),
    "Image Scoring":   ("canonical_sweep_is.csv",       "canonical_sweep_is_gamma05.csv"),
    "Shunning":        ("canonical_sweep_sh.csv",       "canonical_sweep_sh_gamma05.csv"),
    "Simple Standing": ("canonical_sweep_ss.csv",       "canonical_sweep_ss_gamma05.csv"),
}

SHORT = {
    "Stern Judging":               "SJ",
    "Simple Standing":             "SS",
    "Image Scoring":               "IS",
    "Shunning":                    "SH",
    "All Good":                    "AG",
    "All Bad":                     "AB",
    "paradoxical Shunning":        "pSH",
    "paradoxical Simple Standing": "pSS",
    "paradoxical Image Scoring":   "pIS",
    "U2": "U2", "U4": "U4", "U5": "U5",
    "U10": "U10", "U14": "U14",
    "Other": "?",
}

DOM_SHORT = {
    "Disc_Coop":   "Disc · Coop-EP",
    "Disc_Comp":   "Disc · Comp-EP",
    "pDisc_Coop":  "pDisc · Coop-EP",
    "pDisc_Comp":  "pDisc · Comp-EP",
    "AllD_Coop":   "AllD · Coop-EP",
    "AllD_Comp":   "AllD · Comp-EP",
    "AllC_Coop":   "AllC · Coop-EP",
    "AllC_Comp":   "AllC · Comp-EP",
}

BASE_ORDER = ["Stern Judging", "Image Scoring", "Shunning", "Simple Standing"]

# G-value colour gradient: red (G=0, all Bad) → white (G=0.5) → blue (G=1, all Good)
_G_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "g_cmap", ["#d62728", "#ffffff", "#1f77b4"]
)


def _g_colour(g: float) -> tuple:
    return _G_CMAP(float(g))


def _g_bimodal(grp: pd.DataFrame, split: float = 0.5, min_minority: int = 3) -> bool:
    """True if at least min_minority runs land in each G-attractor."""
    high = (grp["G"] >= split).sum()
    low  = (grp["G"] <  split).sum()
    return bool(high >= min_minority and low >= min_minority)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_table() -> pd.DataFrame:
    rows = []
    for norm in BASE_ORDER:
        f1, _ = CANONICAL_FILES[norm]
        d1 = _load_sweep(OUTPUTS_DIR / f1, GAMMA_RIGHT)
        for col in STRAT_COLS + ["G"]:
            d1[col] = pd.to_numeric(d1[col], errors="coerce")

        agg1 = (
            d1.groupby(["8bit", "Cooperative-Social Norm", "Competitive-Social Norm"])
            .apply(lambda g: pd.Series({
                "acr_1":    g["average_cooperation"].mean(),
                "mean_G":   g["G"].mean(),
                "n_highG":  int((g["G"] >= 0.5).sum()),
                "n_lowG":   int((g["G"] <  0.5).sum()),
                "bimodal":  _g_bimodal(g),
                "dom":      g[STRAT_COLS].mean().idxmax(),
            }))
            .reset_index()
        )
        agg1 = agg1[agg1["acr_1"] >= THRESHOLD].copy()
        # Equilibrium: bimodal if both attractors present, else majority G-state
        def _equil(row):
            if row["bimodal"]:
                return "Bimodal"
            return "Disc/Good" if row["mean_G"] >= 0.5 else "pDisc/Bad"
        agg1["equilibrium"] = agg1.apply(_equil, axis=1)
        agg1["base_norm"] = norm
        agg1["label"] = (
            agg1["Cooperative-Social Norm"].map(SHORT).fillna("?")
            + "|"
            + agg1["Competitive-Social Norm"].map(SHORT).fillna("?")
        )
        agg1["dom_short"] = agg1["dom"].map(DOM_SHORT).fillna(agg1["dom"])
        rows.append(agg1)

    df = pd.concat(rows, ignore_index=True)
    # Sort: equilibrium group (Disc/Good, pDisc/Bad, Bimodal), then ACR descending
    equil_order = {"Disc/Good": 0, "pDisc/Bad": 1, "Bimodal": 2}
    df["_esort"] = df["equilibrium"].map(equil_order)
    df = df.sort_values(["_esort", "acr_1"], ascending=[True, False])
    df = df.drop(columns=["_esort"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

DISPLAY_COLS = [
    "label", "base_norm",
    "Cooperative-Social Norm", "Competitive-Social Norm",
    "acr_1", "dom_short", "mean_G", "bimodal",
]
COL_LABELS = [
    "EBSN", "Base norm",
    "Coop. half-rule", "Comp. half-rule",
    "ACR (γ=1)", "Dominant strat.×EP", "Mean G", "Bimodal",
]

# Section banner colours
BANNER = {
    "Disc/Good": ("#1f77b4", "white", "Disc/Good"),
    "pDisc/Bad": ("#d62728", "white", "pDisc/Bad"),
    "Bimodal":   ("#9467bd", "white", "Bimodal"),
}


def render_table(df: pd.DataFrame) -> plt.Figure:
    n_data_rows = len(df)
    n_sections  = df["equilibrium"].nunique()
    n_total     = n_data_rows + n_sections   # data rows + one banner per section
    n_cols      = len(COL_LABELS)

    fig_h = max(6, 0.38 * n_total + 1.2)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    ax.axis("off")

    # Build cell text and colours, inserting banner rows
    cell_text   = []
    cell_colours = []

    prev_equil = None
    prev_base  = None

    for _, row in df.iterrows():
        eq   = row["equilibrium"]
        base = row["base_norm"]

        # Insert section banner when equilibrium changes
        if eq != prev_equil:
            bg, fg, desc = BANNER[eq]
            banner_text = [desc] + [""] * (n_cols - 1)
            cell_text.append(banner_text)
            cell_colours.append([mpl.colors.to_rgba(bg, 1.0)] * n_cols)
            prev_equil = eq
            prev_base  = None   # force bold on first row of new section too

        # Format data row
        acr_str  = f"{row['acr_1']:.1%}"
        g_val    = float(row["mean_G"])
        bim_str  = "yes" if row["bimodal"] else "no"

        cell_text.append([
            row["label"], row["base_norm"],
            row["Cooperative-Social Norm"], row["Competitive-Social Norm"],
            acr_str, row["dom_short"], f"{g_val:.2f}", bim_str,
        ])

        # Base colour: faint equilibrium tint; G cell gets its own gradient
        eq_rgba = mpl.colors.to_rgba(EQUIL_COLOURS[eq], 0.12)
        row_c   = [eq_rgba] * n_cols
        g_col_idx = DISPLAY_COLS.index("mean_G")
        row_c[g_col_idx] = _g_colour(g_val)
        cell_colours.append(row_c)

        prev_base = base

    tbl = ax.table(
        cellText=cell_text,
        colLabels=COL_LABELS,
        cellColours=cell_colours,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.auto_set_column_width(list(range(n_cols)))

    # Header row
    for j in range(n_cols):
        c = tbl[0, j]
        c.set_facecolor("#222222")
        c.set_text_props(color="white", fontweight="bold")

    # Section banner rows: bold white text spanning description
    banner_rows = [i for i, t in enumerate(cell_text) if t[1] == ""]
    for bi in banner_rows:
        tbl_row = bi + 1   # +1 for header
        for j in range(n_cols):
            c = tbl[tbl_row, j]
            c.set_text_props(color="white", fontweight="bold", fontsize=8.5)

    # Bold top border at base-norm group transitions within each section
    prev_base_render = None
    prev_equil_render = None
    data_row_idx = 0
    for _, row in df.iterrows():
        eq   = row["equilibrium"]
        base = row["base_norm"]
        # account for banner rows inserted above
        if eq != prev_equil_render:
            prev_equil_render = eq
            prev_base_render  = None
            data_row_idx      += 1   # skip the banner row
        tbl_row = data_row_idx + 1   # +1 for header
        if base != prev_base_render:
            for j in range(n_cols):
                tbl[tbl_row, j].set_text_props(fontweight="bold")
            prev_base_render = base
        data_row_idx += 1

    # G column header: annotate the gradient meaning
    g_head_idx = COL_LABELS.index("Mean G")
    tbl[0, g_head_idx].get_text().set_text("Mean G\n(blue=Good, red=Bad)")

    fig.suptitle(
        f"Elite EBSNs (ACR ≥ {int(THRESHOLD*100)}% at γ=1)  ·  "
        "grouped by cooperation mechanism",
        fontsize=12, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

DOM_LATEX = {
    "Disc · Coop-EP":  r"Disc$/n$",
    "pDisc · Coop-EP": r"pDisc$/n$",
    "pDisc · Comp-EP": r"pDisc$/m$",
    "Disc · Comp-EP":  r"Disc$/m$",
}

BASE_LATEX = {
    "Stern Judging":   "SJ",
    "Image Scoring":   "IS",
    "Shunning":        "SH",
    "Simple Standing": "SS",
}

EQUIL_LATEX = {
    "Disc/Good": r"\textit{Disc\,/\,Good}",
    "pDisc/Bad": r"\textit{pDisc\,/\,Bad}",
    "Bimodal":   r"\textit{Bimodal}",
}


def write_latex_table(df: pd.DataFrame, out_path: Path) -> None:
    """Write a booktabs LaTeX table grouped by equilibrium type."""
    lines = []
    ncols = 8
    col_spec = r"@{}llllrllc@{}"

    lines += [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{\textbf{Elite EBSNs (ACR $\geq 85\%$ at $\gamma=1$) grouped by "
        r"cooperation mechanism.} "
        r"Three equilibrium types are identified by the dominant strategy--emotional-profile "
        r"pair and mean reputation $\bar{G}$ at convergence. "
        r"\textit{Disc\,/\,Good}: Discriminators cooperate with Good-reputation agents; "
        r"the EBSN sustains high $\bar{G}$, so the condition is almost always met. "
        r"\textit{pDisc\,/\,Bad}: pDiscriminators cooperate with Bad-reputation agents; "
        r"the competitive half-rule drives $\bar{G}$ toward 0, so the condition is "
        r"almost always satisfied. "
        r"\textit{Bimodal}: runs split between the two attractors depending on initial "
        r"conditions; $\bar{G}$ is an average across attractors. "
        r"$n$ = Cooperative-EP, $m$ = Competitive-EP. "
        r"All results at $\gamma=1$, $Z=40$, $\varepsilon=\chi=0.01$, $\beta=1$.}",
        r"\label{tab:elite}",
        r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"EBSN & Base & Coop.\ half-rule & Comp.\ half-rule "
        r"& ACR\,(\%) & Dom.\ strat.$\times$EP & $\bar{G}$ & Bim. \\",
        r"\midrule",
    ]

    prev_equil = None
    prev_base  = None

    for _, row in df.iterrows():
        eq   = row["equilibrium"]
        base = BASE_LATEX.get(row["base_norm"], row["base_norm"])

        if eq != prev_equil:
            if prev_equil is not None:
                lines.append(r"\midrule")
            eq_label = EQUIL_LATEX.get(eq, eq)
            lines.append(
                rf"\multicolumn{{{ncols}}}{{l}}{{{eq_label}}} \\"
            )
            lines.append(r"\midrule")
            prev_equil = eq
            prev_base  = None

        base_cell = rf"\textbf{{{base}}}" if base != prev_base else base
        prev_base = base

        acr_str = f"{row['acr_1']*100:.1f}"
        dom_str = DOM_LATEX.get(row["dom_short"], row["dom_short"])
        g_str   = f"{row['mean_G']:.2f}"
        bim_str = r"$\bullet$" if row["bimodal"] else "---"

        coop = SHORT.get(row["Cooperative-Social Norm"], row["Cooperative-Social Norm"])
        comp = SHORT.get(row["Competitive-Social Norm"], row["Competitive-Social Norm"])

        lines.append(
            rf"{row['label']} & {base_cell} & {coop} & {comp} "
            rf"& {acr_str} & {dom_str} & {g_str} & {bim_str} \\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"LaTeX -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = build_table()

    csv_path = PLOTS_DIR / "elite_table.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"CSV  -> {csv_path}")

    pd.set_option("display.max_rows", 60)
    pd.set_option("display.width", 200)
    print(df[[
        "equilibrium", "base_norm", "label",
        "Cooperative-Social Norm", "Competitive-Social Norm",
        "acr_1", "dom_short", "mean_G", "bimodal",
    ]].to_string(index=False))

    fig = render_table(df)
    fig_path = PLOTS_DIR / "elite_table.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG  -> {fig_path}")

    tex_path = PLOTS_DIR / "elite_table.tex"
    write_latex_table(df, tex_path)


if __name__ == "__main__":
    main()
