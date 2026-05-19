# Session handoff — 2026-05-18

Mid-flight notes for the next Claude Code instance picking up this work after
a workstation change. Read alongside the CLAUDE.md files (which carry the
durable plan); this file only covers what's *in flight*.

## Currently running

- `python SNARE.py inputs/canonical_sweep.yaml` — Phase 1 of the canonical
  EBSN sweep. 66 canonical EBSNs × 25 runs × 2000 generations at Z=50,
  q=1, γ=1, chi=eps=0.01. Writes to `outputs/canonical_sweep_gamma1.csv`.
- Several hours of wall-clock expected. On the new workstation, verify it's
  either still running or finished cleanly before kicking off Phase 2.

## Next concrete steps (Phase 2)

When `canonical_sweep_gamma1.csv` is populated:

1. Read it, take the top 10 canonical EBSNs by mean ACR (use
   `canonical_ebsns.canonical()` to dedup any orbit equivalents).
2. Generate four sibling YAMLs in `inputs/`:
   - `top10_gamma05_base_sj.yaml` (sn=[1,0,0,1])
   - `top10_gamma05_base_ss.yaml` (sn=[1,1,0,1])
   - `top10_gamma05_base_is.yaml` (sn=[1,1,0,0])
   - `top10_gamma05_base_sh.yaml` (sn=[1,0,0,0])
   Each runs the same top-10 list at γ=0.5.
3. Add a `run_ebsn_list_for_base()` helper to `SNARE.py` that accepts an
   explicit list of EBSNs (triggered by YAML `ebsn: [[bits...], ...]`).

## Recent decisions (this session)

- **Canonical reduction fixed.** `generate_unique_norms.ipynb` uses a
  buggy `invert()` (C↔D action swap instead of G↔B reputation swap). The
  authoritative reduction now lives in `canonical_ebsns.py`. Correct count:
  **66 non-trivial canonical EBSNs** (paper's "73" is arithmetically wrong).
  Old notebook and `SNARE.run_all_ebsn_variants` are deprecated with banners
  / runtime warnings. `jitter_plots.ipynb` was patched to import from
  `canonical_ebsns`.
- **Tie-break heuristic changed** to **permissive-on-cooperative**: orbit
  member with more 1-outputs on the cooperative-emotion side wins (OI
  priorities are now secondary). Grounded in the de Melo et al. definition
  of cooperative EP as the leniency-signalling profile. Heatmap and slope
  plots regenerated under the new labelling; bifurcation paragraphs in
  `../EBSNs/sn-article.tex` Results rewritten around the unified
  "permissive-coop / strict-comp" framing (SJ no longer presented as a
  privileged anchor, just the most common occupant of the strict role).
- **Paper Methods rewritten** in `../EBSNs/sn-article.tex` Social Norms
  subsection: orbit-theoretic framing, Burnside count, notational note
  about paper-vs-code flatten convention, explicit tie-break rule.

## Where to look

- `../CLAUDE.md` — top-level dissertation update plan and directory map.
- `../EBSNs/CLAUDE.md` — sub-plan to bring the EBSNs paper to thesis-ready.
- `CLAUDE.md` (this folder) — SNARE architecture, YAML conventions.
- `canonical_ebsns.py` — single source of truth for the orbit reduction.
- `plotting/_common.py` — shared data pipeline for plot scripts.
- `plotting/plot_ebsn_heatmap.py`, `plotting/slope_plots.py`,
  `plotting/jitter_plots.py` — current figure generators.

## In-flight uncommitted files (check `git status` on the new machine)

- `SNARE.py` — added `run_canonical_ebsn_sweep()`, wired `ebsn: "canonical"`,
  deprecated `run_all_ebsn_variants`.
- `canonical_ebsns.py` — new module (paper-Eq.1 reduction + permissive-on-coop).
- `plotting/_common.py` — new shared helpers.
- `plotting/plot_ebsn_heatmap.py` — refactored to import from `canonical_ebsns`.
- `plotting/jitter_plots.py`, `plotting/slope_plots.py` — new.
- `inputs/canonical_sweep.yaml` — new.
- `jitter_plots.ipynb` — patched to import `canonical_ebsns`; backup at `.ipynb.bak`.
- `generate_unique_norms.ipynb` — deprecation banner added; backup at `.ipynb.bak`.

In the EBSNs repo:
- `EBSNs/sn-article.tex` — Methods (Symmetry orbits) and Results
  (bifurcation paragraphs) rewritten. CLAUDE.md updated.
- `EBSNs/figs/ebsn_bifurcation_top30.png`, `_full.png` — regenerated under
  the new heuristic.

## Two paper loose ends not yet addressed

- Figure 1 (`Emotion-Based Social Norms.png`) is still the original PNG
  produced at q=0.75 with standard-deviation error bars. The Results prose
  was updated to say "standard error of the mean", so the figure itself
  needs regenerating to match — easiest path is re-running the relevant
  cells in `jitter_plots.ipynb` after the canonical-sweep data lands.
- The paper Methods text mentions "66 non-trivial canonical EBSNs" but
  doesn't explicitly explain why this disagrees with prior versions of
  the paper that claimed 73. If you submit the paper, decide whether to
  acknowledge the correction explicitly or silently fix it.
