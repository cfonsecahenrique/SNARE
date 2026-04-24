# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SNARE is an agent-based evolutionary simulation studying the emergence of cooperation through indirect reciprocity with **emotion-based social norms**. Agents play iterated Prisoner's Dilemma games, accumulate reputations via a shared image matrix, and evolve strategies/emotion profiles through a Moran-like imitation process with mutation.

## Running Simulations

```bash
# Local run
python SNARE.py inputs/<experiment>.yaml

# Cluster run (SLURM)
bash slurm_job_manager.sh          # throttled batch submission
bash run.sh                        # single run via micromamba on cluster
```

Experiment configuration is entirely in YAML files under `inputs/`. Parameters can be scalar (single experiment) or lists/ranges (parameter sweep). The runner auto-detects sweep vs single-value mode.

## Environment

Conda/micromamba environment defined in `environment.yml` (env name: `snare`, Python 3.11). Key deps: numpy, pandas, matplotlib, altair, tqdm, pyyaml, colorama.

## Architecture

The simulation pipeline flows: **YAML config -> SNARE.py (orchestrator) -> Model + Agent (core) -> CSV output + plots**.

- **`SNARE.py`** — Entry point and orchestrator. Reads YAML config, builds `Model` objects, runs simulations via `multiprocessing.Pool`, collects results, generates time-series plots and sweep plots. Three run modes: single-value experiment, parameter sweep (Cartesian product over list-valued params), and norm-variant sweep (reads norm definitions from `data/new_norms.csv`).

- **`model.py`** — `Model` class. Holds all simulation parameters and the **image matrix** (Z x Z numpy array of binary reputations — the distributed reputation system). Contains `prisoners_dilemma()` which is the hot-path: plays one PD round between two agents, applies reputation/execution errors, updates the image matrix for all observers. Has two code paths: optimized vectorized path for full observability (q=1) and per-observer loop for partial observability (q<1). Also implements `is_consensual()` threshold check.

- **`agent.py`** — `Agent` class. Each agent has: strategy (AllC/AllD/Disc/pDisc), emotion profile (Cooperative/Competitive), gamma (weight for emotion-based norm usage), and fitness. Handles trait mutation.

- **`constants.py`** — Enums for `Strategy` (4 strategies as 2-bit tuples) and `EmotionProfile` (binary). Constants: BAD=DEFECT=0, GOOD=COOPERATE=1.

- **`aux_functions.py`** — Helpers for frequency calculations, result export to CSV, norm formatting/display, and EBSN (emotion-based social norm) encoding. The `ebsn_to_GB()` function converts 8-bit norm vectors to human-readable "G"/"B" strings.

- **`plotting/`** — Standalone plotting scripts that read from `outputs/*.csv`. Use matplotlib and altair. `plot_consensus_lines.py` and `plot_robustness_lines.py` are the main ones. Output goes to `plotting/plots/`.

## Key Domain Concepts

- **Social Norm (SN)**: 4-bit vector → 2x2 matrix mapping (action, opponent_reputation) → new_reputation.
- **Emotion-Based Social Norm (EBSN)**: 8-bit vector → 2x2x2 matrix adding emotion profile dimension. Only applies when the opponent cooperated and the observer's gamma check passes.
- **Image Matrix**: Z x Z matrix where entry [i,j] is agent i's opinion of agent j's reputation. Replaces single-reputation model with distributed, potentially non-consensual reputations.
- **Consensus**: `|2*good - Z| / Z` for a given agent's column. When below threshold, fallback behavior applies (either action-based or emotion-based, controlled by `non_consensus_strategy`).
- **Gamma**: Per-agent probability of using the EBSN instead of the base SN. Can be fixed (delta=0) or evolve.

## YAML Config Structure

```yaml
running:
  runs: 25           # independent simulation runs
  cores: all         # "all" or integer
  plotting: true
  output_file: "results.csv"
simulation:
  ebsn: [1,1,0,0,0,0,1,1]   # 8-bit norm vector, or "all" for full sweep
  sn: [1,0,0,1]              # 4-bit base social norm
  norm:                       # meta-norm name (loads variants from CSV)
  z: 50                      # population size
  mu: 1                      # mutation rate (divided by Z internally)
  chi: 0.01                  # reputation assessment error
  eps: 0.01                  # execution error
  alpha: 0                   # judge assignment error
  xi: 0.01                   # emotion misread error
  beta: 1                    # selection strength
  benefit: 5
  cost: 1
  observability: 0.8         # fraction of population observing each game (q)
  consensus_thresh: 0.8      # threshold for consensual reputation
  non_consensus_strategy: emotion  # "emotion" or "action"
  gamma_min: 0
  gamma_max: 1
  gamma_delta: 0             # 0 = homogeneous gamma
  gamma_gaussian_n: 1        # center of gamma distribution
  generations: 2000
  convergence period: 0.2    # fraction of gens before tracking cooperation
```

Any simulation parameter can be a list (e.g., `chi: [0.01, 0.05, 0.1]`) to trigger a parameter sweep.

## Output

- `outputs/*.csv` — One row per simulation run with all parameters and final metrics (cooperation rate, strategy frequencies, consensus, etc.)
- `simulations/*.png` — Time-series and sweep plots generated during runs
- `plotting/plots/` — Publication-quality plots from standalone scripts

## Performance Notes

The simulation pre-allocates a large buffer of random values per generation to avoid per-call RNG overhead. The buffer auto-refills when nearly exhausted. The full-observability path (q=1) uses vectorized numpy operations on entire columns of the image matrix rather than per-observer loops.
