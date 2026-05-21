"""
moral_landscape.py — Leading-Eight-style moral-landscape analysis of elite EBSNs.

For each elite EBSN, constructs the 12-cell moral table:
  - 8 cells active when the PARTNER cooperated (EBSN fires), indexed by
    (donor_action, recipient_rep, donor_EP)
  - 4 cells active when the PARTNER defected (base SN fires), indexed by
    (donor_action, recipient_rep); same outcome for both donor EPs.

Identifies which cells are universal within each equilibrium type
(Disc/Good, pDisc/Bad, Bimodal) and across all 29 elite EBSNs, then
renders a publication-quality heatmap.

8-bit layout (from make_ebsn_from_list / _common.get_emotional_norms):
  Index:  0    1    2    3    4    5    6    7
  Cell:  DBm  DBn  DGm  DGn  CBm  CBn  CGm  CGn
  where D=Defect C=Cooperate B=Bad G=Good m=Comp-EP n=Coop-EP

Base-SN 4-bit layout (model order): [DB, DG, CB, CG]

Usage:
    python plotting/moral_landscape.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PLOTTING_DIR = Path(__file__).resolve().parent
PLOTS_DIR = PLOTTING_DIR / "plots"
ELITE_CSV = PLOTS_DIR / "elite_table.csv"

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

# 4-bit norm bits in model order [DB, DG, CB, CG]
BASE_NORM_BITS: dict[str, tuple[int, int, int, int]] = {
    "Stern Judging":   (1, 0, 0, 1),
    "Simple Standing": (1, 0, 1, 1),
    "Shunning":        (0, 0, 0, 1),
    "Image Scoring":   (0, 0, 1, 1),
}

# 12-cell column definitions: (internal key, short display label, description)
# Partner cooperated (EBSN fires) — 8 cells
EBSN_COLS: list[tuple[str, str]] = [
    ("CGn", "C·G·n"),   # Cooperate, Good-rep recipient, Coop-EP donor  → bit[7]
    ("CGm", "C·G·m"),   # Cooperate, Good-rep recipient, Comp-EP donor  → bit[6]
    ("CBn", "C·B·n"),   # Cooperate, Bad-rep  recipient, Coop-EP donor  → bit[5]
    ("CBm", "C·B·m"),   # Cooperate, Bad-rep  recipient, Comp-EP donor  → bit[4]
    ("DGn", "D·G·n"),   # Defect,    Good-rep recipient, Coop-EP donor  → bit[3]
    ("DGm", "D·G·m"),   # Defect,    Good-rep recipient, Comp-EP donor  → bit[2]
    ("DBn", "D·B·n"),   # Defect,    Bad-rep  recipient, Coop-EP donor  → bit[1]
    ("DBm", "D·B·m"),   # Defect,    Bad-rep  recipient, Comp-EP donor  → bit[0]
]
# Partner defected (base SN fires) — 4 cells, EP-agnostic
BASE_COLS: list[tuple[str, str]] = [
    ("bCG", "C·G"),     # Cooperate, Good-rep → base CG
    ("bCB", "C·B"),     # Cooperate, Bad-rep  → base CB
    ("bDG", "D·G"),     # Defect,    Good-rep → base DG
    ("bDB", "D·B"),     # Defect,    Bad-rep  → base DB
]

ALL_COLS: list[tuple[str, str]] = EBSN_COLS + BASE_COLS
COL_KEYS: list[str] = [c[0] for c in ALL_COLS]
COL_LABELS: list[str] = [c[1] for c in ALL_COLS]

N_EBSN_COLS = len(EBSN_COLS)   # 8
N_BASE_COLS = len(BASE_COLS)   # 4

EQ_ORDER = ["Disc/Good", "pDisc/Bad", "Bimodal"]
EQ_COLOURS = {
    "Disc/Good": "#1565C0",
    "pDisc/Bad": "#B71C1C",
    "Bimodal":   "#6A1B9A",
}
BASE_ABBREV = {
    "Stern Judging":   "SJ",
    "Simple Standing": "SS",
    "Shunning":        "SH",
    "Image Scoring":   "IS",
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_ebsn_row(bits8: str) -> dict[str, int]:
    """Return {col_key: 0/1} for the 8 EBSN cells from the 8-char bit string.

    Layout: [DBm, DBn, DGm, DGn, CBm, CBn, CGm, CGn] (indices 0–7).
    """
    b = [int(x) for x in bits8]
    return {
        "DBm": b[0], "DBn": b[1],
        "DGm": b[2], "DGn": b[3],
        "CBm": b[4], "CBn": b[5],
        "CGm": b[6], "CGn": b[7],
    }


def _parse_base_row(base_norm_name: str) -> dict[str, int]:
    """Return {col_key: 0/1} for the 4 base-SN cells."""
    db, dg, cb, cg = BASE_NORM_BITS[base_norm_name]
    return {"bDB": db, "bDG": dg, "bCB": cb, "bCG": cg}


def build_landscape(df: pd.DataFrame) -> pd.DataFrame:
    """Append 12 moral-cell columns to the elite table DataFrame."""
    rows = []
    for _, row in df.iterrows():
        cells = {**_parse_ebsn_row(row["8bit"]), **_parse_base_row(row["base_norm"])}
        rows.append(cells)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


# ---------------------------------------------------------------------------
# Pattern analysis
# ---------------------------------------------------------------------------

def find_universals(sub: pd.DataFrame) -> dict[str, int | None]:
    """Return {col_key: value} for constant columns, None for mixed ones."""
    result: dict[str, int | None] = {}
    for key in COL_KEYS:
        vals = sub[key].unique()
        result[key] = int(vals[0]) if len(vals) == 1 else None
    return result


def print_report(df: pd.DataFrame) -> None:
    label_map = dict(ALL_COLS)

    print("\n=== MORAL LANDSCAPE — Universal Patterns ===\n")

    for eq_name in EQ_ORDER:
        sub = df[df["equilibrium"] == eq_name]
        if sub.empty:
            continue
        univs = find_universals(sub)
        print(f"--- {eq_name}  ({len(sub)} EBSNs) ---")

        print("  When partner cooperated (EBSN cells):")
        for key, label in EBSN_COLS:
            v = univs[key]
            if v is not None:
                print(f"    {label:8s}  always {'Good' if v else 'Bad':4s}  [universal]")
            else:
                g = int(sub[key].sum())
                print(f"    {label:8s}  {g}/{len(sub)} Good")

        print("  When partner defected (base-SN cells):")
        for key, label in BASE_COLS:
            v = univs[key]
            if v is not None:
                print(f"    {label:6s}  always {'Good' if v else 'Bad':4s}  [universal]")
            else:
                g = int(sub[key].sum())
                print(f"    {label:6s}  {g}/{len(sub)} Good")
        print()

    # Across all 29
    all_univs = find_universals(df)
    universal_keys = [k for k, v in all_univs.items() if v is not None]
    print(f"--- Universals across all {len(df)} elite EBSNs ---")
    if universal_keys:
        for key in COL_KEYS:
            if key in universal_keys and all_univs[key] is not None:
                val = all_univs[key]
                print(f"  {label_map[key]:8s}  always {'Good' if val else 'Bad'}")
    else:
        print("  (none)")
    print()


# ---------------------------------------------------------------------------
# Heatmap figure
# ---------------------------------------------------------------------------

def _row_label(row: pd.Series) -> str:
    base = BASE_ABBREV.get(row["base_norm"], row["base_norm"])
    return f"{row['label']} [{base}]"


def plot_heatmap(df: pd.DataFrame, output: Path) -> None:
    # Sort: equilibrium order, then descending ACR within group
    df = df.copy()
    df["_eq_ord"] = df["equilibrium"].map({e: i for i, e in enumerate(EQ_ORDER)})
    df = df.sort_values(["_eq_ord", "acr_1"], ascending=[True, False]).reset_index(drop=True)

    mat = df[COL_KEYS].values.astype(float)   # (n_rows, 12)
    n = len(df)

    # --- figure geometry ---
    fig_h = max(7, n * 0.42 + 2.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    cmap = plt.cm.RdYlGn
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto",
              interpolation="nearest")

    # --- column labels (two-level header) ---
    ax.set_xticks(range(len(COL_KEYS)))
    ax.set_xticklabels(COL_LABELS, rotation=55, ha="right", fontsize=8)

    # --- row labels, coloured by equilibrium type ---
    row_labels = [_row_label(row) for _, row in df.iterrows()]
    ax.set_yticks(range(n))
    ax.set_yticklabels(row_labels, fontsize=7.5)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.get_yticklabels()[i].set_color(EQ_COLOURS[row["equilibrium"]])

    # --- "G" / "B" text annotations in each cell ---
    for i in range(n):
        for j in range(len(COL_KEYS)):
            val = int(mat[i, j])
            text_col = "white" if val == 0 else "black"
            ax.text(j, i, "G" if val else "B",
                    ha="center", va="center", fontsize=6.5,
                    color=text_col, fontweight="bold")

    # --- group dividers ---
    prev_eq = None
    for i, (_, row) in enumerate(df.iterrows()):
        if row["equilibrium"] != prev_eq and prev_eq is not None:
            ax.axhline(i - 0.5, color="black", linewidth=2)
        prev_eq = row["equilibrium"]

    # --- column-section divider (EBSN | base SN) ---
    ax.axvline(N_EBSN_COLS - 0.5, color="navy", linewidth=2.5, linestyle="--", alpha=0.7)

    # --- highlight universal cells per equilibrium group ---
    for eq_name in EQ_ORDER:
        sub = df[df["equilibrium"] == eq_name]
        if sub.empty:
            continue
        univs = find_universals(sub)
        rows_idx = sub.index.tolist()
        y0, y1 = rows_idx[0] - 0.48, rows_idx[-1] + 0.48
        for j, key in enumerate(COL_KEYS):
            if univs[key] is not None:
                rect = mpatches.FancyBboxPatch(
                    (j - 0.48, y0), 0.96, y1 - y0,
                    boxstyle="square,pad=0",
                    linewidth=1.8, edgecolor="gold",
                    facecolor="none", zorder=5,
                )
                ax.add_patch(rect)

    # --- section header text ---
    ax.text(N_EBSN_COLS / 2 - 0.5, -1.4,
            "Partner cooperated  (EBSN fires)",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(N_EBSN_COLS + N_BASE_COLS / 2 - 0.5, -1.4,
            "Partner defected\n(base SN)",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

    # --- column sub-headers: action grouping (C / D) ---
    # Cooperate block: cols 0-3 (CGn,CGm,CBn,CBm)
    ax.annotate("", xy=(3.5, -0.5), xytext=(-0.5, -0.5),
                xycoords="data", textcoords="data",
                arrowprops=dict(arrowstyle="-", color="steelblue", lw=1.5))
    ax.text(1.5, -0.7, "Donor cooperates", ha="center", va="top",
            fontsize=7.5, color="steelblue")
    # Defect block: cols 4-7 (DGn,DGm,DBn,DBm)
    ax.annotate("", xy=(7.5, -0.5), xytext=(4.5, -0.5),
                xycoords="data", textcoords="data",
                arrowprops=dict(arrowstyle="-", color="tomato", lw=1.5))
    ax.text(5.5, -0.7, "Donor defects", ha="center", va="top",
            fontsize=7.5, color="tomato")

    # --- legend ---
    eq_patches = [mpatches.Patch(color=c, label=e) for e, c in EQ_COLOURS.items()]
    gold_patch = mpatches.Patch(facecolor="none", edgecolor="gold",
                                linewidth=2, label="Universal within group")
    ax.legend(handles=eq_patches + [gold_patch],
              loc="lower right", bbox_to_anchor=(1.0, 0.0),
              fontsize=8, framealpha=0.9, ncol=1)

    ax.set_title(
        "Moral landscape of elite EBSNs  (G = Good reputation earned, B = Bad)\n"
        "Row labels coloured by equilibrium type; gold border = universal within group",
        fontsize=10, fontweight="bold", pad=18,
    )

    plt.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    df = pd.read_csv(ELITE_CSV, dtype={"8bit": str})
    # Restore leading zeros (some 8-bit strings start with 0)
    df["8bit"] = df["8bit"].str.zfill(8)
    df = build_landscape(df)

    print_report(df)

    out = PLOTS_DIR / "moral_landscape.png"
    plot_heatmap(df, out)


if __name__ == "__main__":
    main()
