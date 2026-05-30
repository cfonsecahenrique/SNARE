"""
Generate two figures for the EBSN semantic analysis in the thesis:

  1. ebsn_semantic_heatmap.pdf  — 16 norms x 8 positions, cells G/B, sorted by eta
  2. ebsn_delta_eta.pdf         — mean eta by G/B assignment for each variable position

Output goes to ../Dissertation/Images/
"""

import csv
import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# ── paths ──────────────────────────────────────────────────────────────────
DATA_FILE = pathlib.Path(__file__).parent / "data" / "new_norms_with_results.csv"
OUT_DIR   = pathlib.Path(__file__).parent.parent / "Dissertation" / "Images"
OUT_DIR.mkdir(exist_ok=True)

# ── semantic metadata ───────────────────────────────────────────────────────
# column order in the 8-bit vector
BIT_LABELS  = ["CG$^n$", "CG$^m$", "CB$^n$", "CB$^m$",
               "DG$^n$", "DG$^m$", "DB$^n$", "DB$^m$"]
SHORT_LABEL  = ["CGⁿ", "CGᵐ", "CBⁿ", "CBᵐ",
                "DGⁿ", "DGᵐ", "DBⁿ", "DBᵐ"]
SEMANTIC     = [
    "Genuine\ncooperation\nwith Good",
    "Strategic\ncooperation\nwith Good",
    "Naive\naltruism\ntoward Bad",
    "Reluctant\ncooperation\nwith Bad",
    "Regretful\nexploitation\nof Good",
    "Joyful\nexploitation\nof Good",
    "Reluctant\npunishment\nof Bad",
    "Gleeful\npunishment\nof Bad",
]
# pos 2 (CB_nice) is always B; pos 7 (DB_mean) is always G
INVARIANT = {2: 0, 7: 1}   # position -> fixed value

# ── load data ───────────────────────────────────────────────────────────────
norms = []
with open(DATA_FILE, newline="", encoding="utf-8") as f:
    for row in csv.reader(f):
        if not row or not row[0].strip() or row[0].strip().startswith("%"):
            continue
        vec8 = row[2].strip()
        if len(vec8) != 8 or not all(c in "01" for c in vec8):
            continue
        try:
            eta1 = float(row[24]) if len(row) > 24 and row[24].strip() else None
        except (ValueError, IndexError):
            eta1 = None
        if eta1 is None:
            continue
        base = row[1].strip()
        base_name = {"1001": "SJ", "0011": "IS", "1011": "SS", "0001": "SH"}.get(base, base)
        name = row[0].strip().replace("_", " ").replace("Stern Judging", "SJ") \
                                                .replace("Simple Standing", "SS") \
                                                .replace("Image Scoring", "IS") \
                                                .replace("Shunning", "SH")
        norms.append({
            "name":  name,
            "base":  base_name,
            "bits":  [int(b) for b in vec8],
            "eta1":  eta1,
        })

norms.sort(key=lambda n: -n["eta1"])

# ── Figure 1: Heatmap ────────────────────────────────────────────────────────
n_norms = len(norms)
n_pos   = 8

matrix = np.array([n["bits"] for n in norms], dtype=float)   # (16, 8)

fig, ax = plt.subplots(figsize=(8.5, 5.5))

# Colour scheme: G=teal, B=coral; invariant columns hatched
cmap = ListedColormap(["#E07050", "#50A090"])   # 0=B coral, 1=G teal

im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

# Column labels (top)
ax.set_xticks(range(n_pos))
ax.set_xticklabels(SEMANTIC, fontsize=7.5, ha="center")
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()

# Row labels (left) = norm name + eta
ylabels = [f"{n['name']}  ({n['eta1']:.1f}%)" for n in norms]
ax.set_yticks(range(n_norms))
ax.set_yticklabels(ylabels, fontsize=8)

# Cell text: G or B
for r in range(n_norms):
    for c in range(n_pos):
        val = int(matrix[r, c])
        txt = "G" if val else "B"
        colour = "white"
        weight = "bold" if c in INVARIANT else "normal"
        ax.text(c, r, txt, ha="center", va="center",
                fontsize=8, color=colour, fontweight=weight)

# Vertical separator lines between positions
for x in np.arange(-0.5, n_pos, 1):
    ax.axvline(x, color="white", linewidth=0.8)
for y in np.arange(-0.5, n_norms, 1):
    ax.axhline(y, color="white", linewidth=0.5)

# Thick borders around the two invariant columns
for col_idx in INVARIANT:
    rect = mpatches.FancyBboxPatch(
        (col_idx - 0.5, -0.5), 1, n_norms,
        boxstyle="square,pad=0",
        linewidth=2.5, edgecolor="#222222", facecolor="none",
        zorder=5,
    )
    ax.add_patch(rect)

ax.set_xlim(-0.5, n_pos - 0.5)
ax.set_ylim(n_norms - 0.5, -0.5)

fig.suptitle(
    "Moral judgement pattern of studied EBSNs, sorted by cooperation level ($\\eta$ at $\\gamma=1$)\n"
    "$n$ = Cooperative EP (joy on CC / regret on DC); "
    "$m$ = Competitive EP (regret on CC / joy on DC)",
    fontsize=9, y=1.18,
)

plt.tight_layout(rect=[0, 0, 1, 1])
out1 = OUT_DIR / "ebsn_semantic_heatmap.pdf"
fig.savefig(out1, bbox_inches="tight", dpi=200)
print(f"Saved: {out1}")
plt.close()

# ── Figure 2: Delta-eta bar chart ────────────────────────────────────────────
# For each of the 6 variable positions, show mean eta when assigned G vs B
variable_pos = [i for i in range(8) if i not in INVARIANT]

col_g  = []   # mean eta when bit=G
col_b  = []   # mean eta when bit=B
col_ng = []   # count G
col_nb = []   # count B
for pos in variable_pos:
    g_norms = [n for n in norms if n["bits"][pos] == 1]
    b_norms = [n for n in norms if n["bits"][pos] == 0]
    col_g.append(np.mean([n["eta1"] for n in g_norms]) if g_norms else np.nan)
    col_b.append(np.mean([n["eta1"] for n in b_norms]) if b_norms else np.nan)
    col_ng.append(len(g_norms))
    col_nb.append(len(b_norms))

x = np.arange(len(variable_pos))
width = 0.35

fig2, ax2 = plt.subplots(figsize=(7, 3.8))

bars_g = ax2.bar(x - width/2, col_g, width, color="#50A090",
                 label="Assigned G (Good reputation)", zorder=3)
bars_b = ax2.bar(x + width/2, col_b, width, color="#E07050",
                 label="Assigned B (Bad reputation)", zorder=3)

# annotate n counts
for bar, ng in zip(bars_g, col_ng):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"n={ng}", ha="center", va="bottom", fontsize=7.5, color="#2a6a60")
for bar, nb in zip(bars_b, col_nb):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"n={nb}", ha="center", va="bottom", fontsize=7.5, color="#904030")

# axis labels
ax2.set_xticks(x)
ax2.set_xticklabels(
    ["CG$^n$\nGenuine coop\nw/ Good",
     "CG$^m$\nStrategic coop\nw/ Good",
     "CB$^m$\nReluctant coop\nw/ Bad",
     "DG$^n$\nRegretful\nexploitation",
     "DG$^m$\nJoyful\nexploitation",
     "DB$^n$\nReluctant\npunishment"],
    fontsize=8,
)

ax2.set_ylabel("Mean $\\eta$ at $\\gamma=1$ (%)", fontsize=9)
ax2.set_ylim(75, 96)
ax2.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax2.set_axisbelow(True)
ax2.legend(fontsize=9, loc="upper right")

# Reference line at overall mean
overall_mean = np.mean([n["eta1"] for n in norms])
ax2.axhline(overall_mean, color="grey", linewidth=1, linestyle=":",
            label=f"Overall mean ({overall_mean:.1f}%)")
ax2.text(-0.5, overall_mean + 0.3, f"overall mean\n({overall_mean:.1f}%)",
         fontsize=7.5, color="grey", va="bottom")

ax2.set_title(
    "Mean cooperation level by moral judgement (G vs B), for the six variable EBSN positions\n"
    "(CB$^n$ and DB$^m$ omitted: invariant across all norms)",
    fontsize=9,
)

plt.tight_layout()
out2 = OUT_DIR / "ebsn_delta_eta.pdf"
fig2.savefig(out2, bbox_inches="tight", dpi=200)
print(f"Saved: {out2}")
plt.close()
