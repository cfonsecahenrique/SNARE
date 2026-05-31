"""
Principled re-derivation of the EBSN semantic / position-level analysis.

Replaces the legacy n=16 hand-curated sample (new_norms_with_results.csv,
50% SJ-derived) with the full canonical sweep at gamma=1, base-balanced over
the four leading base norms.

Two denominators, deliberately different:
  * INVARIANTS  -> over DISTINCT elite canonical vectors (eta>=85% under >=1
                   base norm), deduplicated, so the SJ-heavy pairings cannot
                   re-bias the "what do successful norms share" claim.
  * DELTA-ETA   -> over the full 66x4 (base, EBSN) configs, which retains the
                   eta spread the continuous association needs.

Thesis position order (Table tab:ebsn_positions), 1-indexed:
  1 CG_coop  2 CG_comp  3 CB_coop*  4 CB_comp
  5 DG_coop  6 DG_comp  7 DB_coop**  8 DB_comp***
   * pos3 = naive altruism toward Bad      (claimed universal Bad)
  ** pos7 = reluctant punishment of Bad    (claimed strongest +deltaeta)
 *** pos8 = gleeful punishment of Bad      (claimed universal Good)

eb_social_norm nested layout (from aux_functions.make_ebsn_from_list):
  [[(DB_comp,DB_coop),(DG_comp,DG_coop)], [(CB_comp,CB_coop),(CG_comp,CG_coop)]]
"""

import ast
import csv
import pathlib

OUT = pathlib.Path(__file__).parent / "outputs"
ELITE = 85.0

# filename -> base-norm label
FILES = {
    "canonical_sweep_gamma1.csv": "SJ",
    "canonical_sweep_is.csv": "IS",
    "canonical_sweep_sh.csv": "SH",
    "canonical_sweep_ss.csv": "SS",
}

THESIS_LABELS = ["CG_coop", "CG_comp", "CB_coop", "CB_comp",
                 "DG_coop", "DG_comp", "DB_coop", "DB_comp"]
SEMANTIC = ["genuine coop w/Good", "strategic coop w/Good",
            "naive altruism->Bad (pos3)", "reluctant coop w/Bad",
            "regretful exploit Good", "joyful exploit Good",
            "reluctant punish Bad (pos7)", "gleeful punish Bad (pos8)"]


def nested_to_thesis_bits(ebsn):
    """Return 8 bits in thesis position order from the nested eb_social_norm."""
    (DB, DG), (CB, CG) = ebsn  # each is (comp, coop)
    DB_comp, DB_coop = DB
    DG_comp, DG_coop = DG
    CB_comp, CB_coop = CB
    CG_comp, CG_coop = CG
    return [CG_coop, CG_comp, CB_coop, CB_comp,
            DG_coop, DG_comp, DB_coop, DB_comp]


# (base, vec_tuple) -> list of eta
from collections import defaultdict
runs = defaultdict(list)

skipped = defaultdict(int)
for fname, base in FILES.items():
    path = OUT / fname
    if not path.exists():
        print(f"!! missing {path}")
        continue
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ebsn = ast.literal_eval(row["eb_social_norm"])
                bits = tuple(nested_to_thesis_bits(ebsn))
                eta = float(row["average_cooperation"])
            except (ValueError, SyntaxError, TypeError, KeyError):
                skipped[base] += 1
                continue
            runs[(base, bits)].append(eta)
if skipped:
    print(f"skipped unparseable rows: {dict(skipped)}\n")

# mean eta per (base, vec)
configs = {k: sum(v) / len(v) for k, v in runs.items()}
n_runs = {k: len(v) for k, v in runs.items()}

print(f"(base, EBSN) configs: {len(configs)}")
for base in FILES.values():
    vecs = {k[1] for k in configs if k[0] == base}
    rr = [n_runs[k] for k in configs if k[0] == base]
    print(f"  {base}: {len(vecs)} distinct EBSN vectors, "
          f"runs/config min={min(rr)} max={max(rr)}")
print()

# ---------------------------------------------------------------------------
# INVARIANTS over distinct elite vectors (deduped across base norms)
# ---------------------------------------------------------------------------
elite_vecs = set()
for (base, vec), eta in configs.items():
    if eta >= ELITE:
        elite_vecs.add(vec)

print("=" * 78)
print(f"INVARIANTS over {len(elite_vecs)} DISTINCT elite vectors (eta>=85% under >=1 base)")
print("=" * 78)
print(f"{'pos':<4}{'label':<11}{'%G':>6}  {'verdict':<12} {'semantic'}")
for p in range(8):
    g = sum(v[p] for v in elite_vecs)
    pg = 100 * g / len(elite_vecs) if elite_vecs else float("nan")
    verdict = "ALWAYS G" if pg == 100 else ("ALWAYS B" if pg == 0 else "varies")
    print(f"{p+1:<4}{THESIS_LABELS[p]:<11}{pg:5.0f}%  {verdict:<12} {SEMANTIC[p]}")
print()

# ---------------------------------------------------------------------------
# DELTA-ETA over the full base-balanced sample (all configs)
# ---------------------------------------------------------------------------
print("=" * 78)
print(f"DELTA-ETA over full sample ({len(configs)} base-balanced configs)")
print("=" * 78)
print(f"{'pos':<4}{'label':<11}{'meanG':>8}{'meanB':>8}{'nG':>5}{'nB':>5}{'diff':>8}  semantic")
rows = list(configs.values())
allmean = sum(rows) / len(rows)
for p in range(8):
    g = [eta for (base, vec), eta in configs.items() if vec[p] == 1]
    b = [eta for (base, vec), eta in configs.items() if vec[p] == 0]
    mg = sum(g) / len(g) if g else float("nan")
    mb = sum(b) / len(b) if b else float("nan")
    diff = mg - mb
    flag = " ***" if abs(diff) >= 2 else ""
    print(f"{p+1:<4}{THESIS_LABELS[p]:<11}{mg:7.1f}%{mb:7.1f}%{len(g):5}{len(b):5}{diff:+7.1f}%{flag}  {SEMANTIC[p]}")
print()
print(f"overall mean eta = {allmean:.1f}%   eta range over configs = "
      f"{min(rows):.1f}%..{max(rows):.1f}%")

# Same delta-eta but restricted to elite configs only, for contrast
print()
print("-- delta-eta restricted to elite configs only (for contrast) --")
elite_cfg = {k: v for k, v in configs.items() if v >= ELITE}
print(f"   ({len(elite_cfg)} elite (base,EBSN) configs)")
for p in range(8):
    g = [eta for (base, vec), eta in elite_cfg.items() if vec[p] == 1]
    b = [eta for (base, vec), eta in elite_cfg.items() if vec[p] == 0]
    if not g or not b:
        print(f"{p+1:<4}{THESIS_LABELS[p]:<11} (one side empty: nG={len(g)} nB={len(b)})")
        continue
    mg, mb = sum(g) / len(g), sum(b) / len(b)
    print(f"{p+1:<4}{THESIS_LABELS[p]:<11}{mg:7.1f}%{mb:7.1f}%{len(g):5}{len(b):5}{mg-mb:+7.1f}%")
