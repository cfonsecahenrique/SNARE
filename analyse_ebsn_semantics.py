"""
Semantic analysis of the EBSN space.

Each 8-bit EBSN vector (bit order as stored in new_norms_with_results.csv):
  pos 0: CG_nice  = Cooperate + Good-recipient + Cooperative-EP (joy)
  pos 1: CG_mean  = Cooperate + Good-recipient + Competitive-EP (regret)
  pos 2: CB_nice  = Cooperate + Bad-recipient  + Cooperative-EP (joy)
  pos 3: CB_mean  = Cooperate + Bad-recipient  + Competitive-EP (regret)
  pos 4: DG_nice  = Defect + Good-recipient + Cooperative-EP (regret) ← "regretful exploitation"
  pos 5: DG_mean  = Defect + Good-recipient + Competitive-EP (joy)   ← "joyful exploitation"
  pos 6: DB_nice  = Defect + Bad-recipient  + Cooperative-EP (regret) ← "reluctant punishment"
  pos 7: DB_mean  = Defect + Bad-recipient  + Competitive-EP (joy)   ← "gleeful punishment"

The emotion-action alignment for each position:
  Aligned (emotion matches intent implied by action):
    CG_nice: cooperate + joy          → clearly prosocial
    DG_mean: defect against Good + joy → clearly exploitative
    DB_mean: defect against Bad  + joy → clearly punishing (willing)
    CB_mean: cooperate against Bad + regret → reluctant/coerced cooperation
  Misaligned (emotion contradicts implied intent):
    CG_mean: cooperate + regret        → strategic/unwilling cooperation
    DG_nice: defect against Good + regret → accidental / reluctant exploitation
    DB_nice: defect against Bad  + regret → reluctant punishment
    CB_nice: cooperate against Bad + joy  → naive altruism toward defectors
"""

import csv
import pathlib
import textwrap

# ---------------------------------------------------------------------------
# 1.  Load norm data from new_norms_with_results.csv
#     Columns (no header row):
#       0: norm name   1: 4-bit base   2: 8-bit vector
#       3-10: individual bit values (CG_nice .. DB_mean, matching 8-bit order)
#       11: DNF        12: leniency    13: #literals
#       14..: eta values at gamma = 0, 0.1, 0.2, ..., 1.0  (11 values, then deltas)
# ---------------------------------------------------------------------------

DATA_FILE = pathlib.Path(__file__).parent / "data" / "new_norms_with_results.csv"

BIT_LABELS = ["CG_nice", "CG_mean", "CB_nice", "CB_mean",
              "DG_nice", "DG_mean", "DB_nice", "DB_mean"]

SEMANTIC_LABELS = [
    "C + Good + Coop-EP (joy) [Genuine cooperation]",
    "C + Good + Comp-EP (regret) [Strategic cooperation]",
    "C + Bad  + Coop-EP (joy) [Naive altruism]",
    "C + Bad  + Comp-EP (regret) [Reluctant/coerced cooperation]",
    "D + Good + Coop-EP (regret) [Regretful exploitation / mistake]",
    "D + Good + Comp-EP (joy) [Joyful exploitation]",
    "D + Bad  + Coop-EP (regret) [Reluctant punishment]",
    "D + Bad  + Comp-EP (joy) [Gleeful punishment]",
]

ELITE_THRESHOLD = 85.0   # used only for the binary table; continuous analysis below

norms = []

with open(DATA_FILE, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        # skip annotation rows (col 0 or 1 empty, or starts with %)
        if not row or not row[0].strip() or row[0].strip().startswith("%"):
            continue
        name   = row[0].strip()
        base4  = row[1].strip()
        vec8   = row[2].strip()
        if len(vec8) != 8 or not all(c in "01" for c in vec8):
            continue

        bits = [int(b) for b in vec8]   # 8 bits, positions 0-7

        # eta values start at column 14; gamma=1 is the 11th value (index 14+10=24)
        # but some rows have fewer columns (partial data)
        try:
            eta_gamma1 = float(row[24]) if len(row) > 24 and row[24].strip() else None
        except (ValueError, IndexError):
            eta_gamma1 = None

        norms.append({
            "name":   name,
            "base":   base4,
            "vec":    vec8,
            "bits":   bits,
            "eta1":   eta_gamma1,
            "elite":  (eta_gamma1 is not None and eta_gamma1 >= ELITE_THRESHOLD),
        })

# ---------------------------------------------------------------------------
# 2.  Separate elite / non-elite (only rows with a valid eta)
# ---------------------------------------------------------------------------

with_eta  = [n for n in norms if n["eta1"] is not None]
elite     = [n for n in with_eta if n["elite"]]
non_elite = [n for n in with_eta if not n["elite"]]

print(f"Norms with eta@gamma=1 data: {len(with_eta)}  "
      f"(elite >={ELITE_THRESHOLD}%: {len(elite)},  non-elite: {len(non_elite)})")
print()

# ---------------------------------------------------------------------------
# 3.  Per-position frequency of G (=1) in elite vs non-elite
# ---------------------------------------------------------------------------

def pct_G(norm_list, pos):
    if not norm_list:
        return float("nan")
    return 100 * sum(n["bits"][pos] for n in norm_list) / len(norm_list)

print("=" * 90)
print(f"{'Pos':<3} {'Short label':<10} {'G% elite':>9} {'G% non-elite':>13}  Semantic meaning")
print("=" * 90)
for pos, (short, full) in enumerate(zip(BIT_LABELS, SEMANTIC_LABELS)):
    ge  = pct_G(elite, pos)
    gne = pct_G(non_elite, pos)
    flag = "  ***" if abs(ge - gne) > 30 else ""
    line = f"{pos:<3} {short:<10}  {ge:6.0f}%      {gne:6.0f}%      {full}{flag}"
    print(line.encode('ascii', errors='replace').decode('ascii'))

print()

# ---------------------------------------------------------------------------
# 4.  Full norm-by-norm table sorted by η
# ---------------------------------------------------------------------------

sorted_norms = sorted(with_eta, key=lambda n: n["eta1"], reverse=True)

header = f"{'Rank':<5} {'Name':<22} {'Base':<5} {'eta@g=1':>7}  " + "  ".join(BIT_LABELS)
print(header)
print("-" * len(header))
for rank, n in enumerate(sorted_norms, 1):
    bits_str = "  ".join(("G" if b else "B") for b in n["bits"])
    tag = " ← ELITE" if n["elite"] else ""
    print(f"{rank:<5} {n['name']:<22} {n['base']:<5} {n['eta1']:>6.1f}%  {bits_str}{tag}".encode('ascii', errors='replace').decode('ascii'))

print()

# ---------------------------------------------------------------------------
# 5.  Highlight the 4 most discriminating positions
# ---------------------------------------------------------------------------

print("=" * 60)
print("Key semantic contrasts  (elite vs non-elite)")
print("=" * 60)

discriminating = [(pos, abs(pct_G(elite, pos) - pct_G(non_elite, pos)))
                  for pos in range(8)]
discriminating.sort(key=lambda x: -x[1])

for pos, delta in discriminating:
    ge  = pct_G(elite, pos)
    gne = pct_G(non_elite, pos)
    direction = ("elite favour G" if ge > gne else "elite favour B")
    short = SEMANTIC_LABELS[pos].split("[")[1].rstrip("]")
    print(f"  d={delta:4.0f}%  {BIT_LABELS[pos]:<10}  {direction:<18}  ({short})")

print()

# ---------------------------------------------------------------------------
# 6.  Continuous analysis: mean eta by bit value, per position
#     Avoids the arbitrary elite/non-elite binarisation.
# ---------------------------------------------------------------------------

print("=" * 80)
print("Continuous analysis: mean eta(gamma=1) grouped by G/B per position")
print("(avoids the arbitrary 85% threshold)")
print("=" * 80)
print(f"{'Pos':<3} {'Label':<10}  {'mean_eta | G':>13}  {'mean_eta | B':>13}  {'n_G':>4}  {'n_B':>4}  {'diff':>6}  Semantic")
print("-" * 80)

for pos, (short, full) in enumerate(zip(BIT_LABELS, SEMANTIC_LABELS)):
    g_norms = [n for n in with_eta if n["bits"][pos] == 1]
    b_norms = [n for n in with_eta if n["bits"][pos] == 0]
    mean_g = sum(n["eta1"] for n in g_norms) / len(g_norms) if g_norms else float("nan")
    mean_b = sum(n["eta1"] for n in b_norms) / len(b_norms) if b_norms else float("nan")
    n_g, n_b = len(g_norms), len(b_norms)
    diff = mean_g - mean_b if (g_norms and b_norms) else float("nan")
    flag = "  ***" if abs(diff) > 2.0 else ""
    semantic_short = "[" + full.split("[")[1] if "[" in full else full
    line = (f"{pos:<3} {short:<10}  {mean_g:>12.2f}%  {mean_b:>12.2f}%  "
            f"{n_g:>4}  {n_b:>4}  {diff:>+6.2f}%  {semantic_short}{flag}")
    print(line.encode("ascii", errors="replace").decode("ascii"))

print()

# Per-norm scatter: show eta alongside each bit pattern so trends are visible
print("=" * 80)
print("Full norm table sorted by eta — read columns to spot per-position trends")
print("=" * 80)
header2 = (f"{'eta':>6}  " + "  ".join(f"{l:<7}" for l in BIT_LABELS) + "  Name")
print(header2)
print("-" * len(header2))
for n in sorted(with_eta, key=lambda x: -x["eta1"]):
    bits_str = "  ".join(("G      " if b else "B      ") for b in n["bits"])
    line = f"{n['eta1']:>5.1f}%  {bits_str}  {n['name']}"
    print(line.encode("ascii", errors="replace").decode("ascii"))

print()
print("Interpretation guide:")
print(textwrap.dedent("""
  G = this interaction earns a GOOD reputation under the norm
  B = this interaction earns a BAD  reputation under the norm

  'diff' = mean_eta(G norms) - mean_eta(B norms) for each position.
  A large positive diff means assigning G to this interaction is associated
  with higher cooperation across the sample (and vice versa).
  Note: sample size is only 16 norms; treat patterns as hypothesis-generating,
  not statistically conclusive.
"""))
