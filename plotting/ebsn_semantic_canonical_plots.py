"""
Preview figures for the PRINCIPLED EBSN semantic analysis (264 canonical
configs = 66 canonical x 4 base norms, gamma=1), replacing the legacy n=16
hand-curated sample.

Produces (PNG previews):
  plots/ebsn_semantic_heatmap_canonical.png  -- 18 distinct elite vectors x 8 positions
  plots/ebsn_delta_eta_canonical.png         -- mean eta | G vs | B per position (full sample)
"""

import ast
import csv
import pathlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = pathlib.Path(__file__).parent
OUT = HERE.parent / "outputs"
PLOTS = HERE / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
IMAGES = HERE.parent.parent / "Dissertation" / "Images"   # thesis figure dir
ELITE = 85.0

FILES = {"canonical_sweep_gamma1.csv": "SJ", "canonical_sweep_is.csv": "IS",
         "canonical_sweep_sh.csv": "SH", "canonical_sweep_ss.csv": "SS"}

# thesis position order, compact labels + semantics
LAB = ["CG$^n$", "CG$^m$", "CB$^n$", "CB$^m$", "DG$^n$", "DG$^m$", "DB$^n$", "DB$^m$"]
SEM = ["sincere coop\nw/Good (joy)", "strategic coop\nw/Good (regret)",
       "naive altruism\n->Bad (joy)", "reluctant coop\nw/Bad (regret)",
       "regretful\nexploit Good", "joyful\nexploit Good",
       "reluctant\npunish Bad", "gleeful\npunish Bad"]

TEAL, CORAL = "#2a9d8f", "#e76f51"


def nested_to_thesis_bits(ebsn):
    (DB, DG), (CB, CG) = ebsn          # each cell is (competitive, cooperative)
    return [CG[1], CG[0], CB[1], CB[0], DG[1], DG[0], DB[1], DB[0]]


# load ------------------------------------------------------------------
runs = defaultdict(list)
for fname, base in FILES.items():
    with open(OUT / fname, newline="") as f:
        for row in csv.DictReader(f):
            try:
                bits = tuple(nested_to_thesis_bits(ast.literal_eval(row["eb_social_norm"])))
                eta = float(row["average_cooperation"])
            except (ValueError, SyntaxError, TypeError, KeyError):
                continue
            runs[(base, bits)].append(eta)

configs = {k: sum(v) / len(v) for k, v in runs.items()}

# distinct elite vectors with their best base + max eta + elite bases
vec_best = {}
vec_bases = defaultdict(list)
for (base, vec), eta in configs.items():
    if eta >= ELITE:
        vec_bases[vec].append((base, eta))
        if vec not in vec_best or eta > vec_best[vec]:
            vec_best[vec] = eta
elite_vecs = sorted(vec_best, key=lambda v: -vec_best[v])
print(f"{len(elite_vecs)} distinct elite vectors")

# ======================================================================
# FIG 1 -- semantic heatmap over distinct elite vectors
# ======================================================================
n = len(elite_vecs)
fig, ax = plt.subplots(figsize=(9, 0.42 * n + 2.4))
for r, vec in enumerate(elite_vecs):
    for c, b in enumerate(vec):
        ax.add_patch(Rectangle((c, n - 1 - r), 1, 1,
                     facecolor=TEAL if b else CORAL, edgecolor="white", lw=1.5))
# column %G annotation (computed over exactly the rows shown -> de-biased)
pctG = [100 * sum(v[c] for v in elite_vecs) / n for c in range(8)]
for c in range(8):
    tag = ""
    if pctG[c] == 100:
        tag = "\n(universal)"
    elif pctG[c] >= 90:
        tag = "\n(near-univ.)"
    ax.text(c + 0.5, n + 0.15, f"{LAB[c]}", ha="center", va="bottom", fontsize=11)
    ax.text(c + 0.5, -0.55, f"{pctG[c]:.0f}% G{tag}", ha="center", va="top", fontsize=7.5)
# row labels: best base + eta
for r, vec in enumerate(elite_vecs):
    bases = "/".join(sorted({b for b, _ in vec_bases[vec]}))
    ax.text(-0.15, n - 1 - r + 0.5, f"{bases}  {vec_best[vec]:.0f}%",
            ha="right", va="center", fontsize=7.5)
# intention-confirmation pair (pos1 vs pos2): same action+rep, emotion flips verdict
ax.text(1.0, n + 1.05, "intention-confirmation:\nsame action+rep, emotion flips verdict",
        ha="center", va="bottom", fontsize=8, style="italic")
ax.set_xlim(-2.6, 8.3)
ax.set_ylim(-2.4, n + 2.0)
ax.axis("off")
ax.set_title("Moral judgement pattern of the 18 distinct elite EBSN vectors "
             f"(eta>={ELITE:.0f}%, gamma=1)\nteal = Good, coral = Bad; "
             "column %G computed over these 18 vectors", fontsize=10)
fig.savefig(PLOTS / "ebsn_semantic_heatmap_canonical.png", dpi=170, bbox_inches="tight")
fig.savefig(IMAGES / "ebsn_semantic_heatmap.pdf", bbox_inches="tight")
plt.close(fig)

# ======================================================================
# FIG 2 -- Delta-eta per position, STRATIFIED by base norm
#          (within-base estimates control for base-norm confounding;
#           spread across the 4 base norms is the honest error bar)
# ======================================================================
bases = list(FILES.values())
pooled, per_base, robust = [], [], []
for p in range(8):
    g = [e for (b, v), e in configs.items() if v[p] == 1]
    bd = [e for (b, v), e in configs.items() if v[p] == 0]
    pooled.append(sum(g) / len(g) - sum(bd) / len(bd))
    vals = []
    for base in bases:
        gb = [e for (b, v), e in configs.items() if b == base and v[p] == 1]
        bb = [e for (b, v), e in configs.items() if b == base and v[p] == 0]
        vals.append(sum(gb) / len(gb) - sum(bb) / len(bb) if gb and bb else None)
    per_base.append(vals)
    present = [x for x in vals if x is not None]
    robust.append(len({x > 0 for x in present}) == 1)   # all same sign

means = [sum(v) / len(v) for v in ([x for x in vb if x is not None] for vb in per_base)]
lo = [min(x for x in vb if x is not None) for vb in per_base]
hi = [max(x for x in vb if x is not None) for vb in per_base]

fig, ax = plt.subplots(figsize=(11.5, 5.6))
x = list(range(8))
for i in x:
    m = means[i]
    if not robust[i]:
        fc, hatch = "#cccccc", "////"
    else:
        fc, hatch = (TEAL if m > 0 else CORAL), None
    ax.bar(i, m, 0.62, color=fc, hatch=hatch, edgecolor="white", zorder=2)
    ax.errorbar(i, m, yerr=[[m - lo[i]], [hi[i] - m]], fmt="none",
                ecolor="#333333", capsize=5, lw=1.4, zorder=3)
    # individual base-norm points
    for v in per_base[i]:
        if v is not None:
            ax.plot(i, v, "o", ms=4.5, color="#222222",
                    markerfacecolor="white", zorder=4)
    ax.text(i, hi[i] + 1.8 if m > 0 else lo[i] - 1.8, f"{m:+.1f}",
            ha="center", va=("bottom" if m > 0 else "top"),
            fontsize=9.5, fontweight="bold",
            color=("#1d6f64" if m > 0 else "#b5462e"))

ax.axhline(0, color="black", lw=0.9)
ax.set_xticks(x)
ax.set_xticklabels([f"{LAB[i]}\n{SEM[i]}" for i in x], fontsize=8)
ax.set_ylabel("$\\Delta\\eta$ (assign Good $-$ assign Bad), %")
ax.set_title("Effect of each moral verdict on cooperation, stratified by base norm "
             "($\\gamma=1$)\n"
             "bar = mean across the 4 base norms; whisker = range; dots = individual base norms; "
             "grey/hatched = sign flips across bases (not robust)", fontsize=9.5)
ax.margins(y=0.18)
# legend proxies
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor=TEAL, label="robustly rewarded (Good $\\uparrow$ coop)"),
    Patch(facecolor=CORAL, label="robustly penalised (Bad $\\uparrow$ coop)"),
    Patch(facecolor="#cccccc", hatch="////", label="sign flips across base norms"),
], loc="lower left", fontsize=8, framealpha=0.95)
fig.tight_layout()
fig.savefig(PLOTS / "ebsn_delta_eta_canonical.png", dpi=170, bbox_inches="tight")
fig.savefig(IMAGES / "ebsn_delta_eta.pdf", bbox_inches="tight")
plt.close(fig)

# ======================================================================
# intention-confirmation contrast, numbers
# ======================================================================
print("\nINTENTION-CONFIRMATION (cooperate with Good, action+rep identical):")
e1 = 100 * sum(v[0] for v in elite_vecs) / n
e2 = 100 * sum(v[1] for v in elite_vecs) / n
print(f"  elite-vector %Good:  CG joy (pos1) = {e1:.0f}%   CG regret (pos2) = {e2:.0f}%")
print(f"  within-base Delta-eta: CG joy {means[0]:+.1f}pp (robust={robust[0]}) | "
      f"CG regret {means[1]:+.1f}pp (robust={robust[1]})")
print("\nsaved:")
print(f"  {PLOTS / 'ebsn_semantic_heatmap_canonical.png'}")
print(f"  {PLOTS / 'ebsn_delta_eta_canonical.png'}")
