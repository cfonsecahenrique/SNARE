#!/usr/bin/env python3
"""
SNARE SLURM submission watcher.

Reads a YAML experiment file, enumerates all (combo, run) pairs, and submits
one single-core SLURM job per pair via slurm_single_run.sh, throttling at
--max-jobs concurrent jobs.

Usage:
    python submit_slurm.py inputs/sweep_z_fine_A.yaml
    python submit_slurm.py inputs/sweep_z_fine_A.yaml --max-jobs 300 --sleep 30
    python submit_slurm.py inputs/sweep_z_fine_A.yaml --dry-run
"""

import argparse
import subprocess
import sys
import time
import itertools
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np

USERNAME = "cfonsecahenrique"
JOB_SCRIPT = "slurm_single_run.sh"

# Must match the sweep-key detection logic in SNARE.py's Manual mode
SWEEP_KEYS = [
    "consensus_thresh", "observability", "alpha", "chi", "eps", "xi",
    "z", "benefit", "beta", "mu", "gamma_gaussian_n", "gamma_min", "gamma_max",
]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def expand_param(val):
    """Mirror SNARE.py's expand_parameter: list or range-string → list of values."""
    if isinstance(val, list):
        return val
    if isinstance(val, str) and val.count("-") == 2:
        start, end, step = map(float, val.split("-"))
        return list(np.arange(start, end + step, step))
    return [val]


def count_combos(sim_params: dict) -> int:
    n = 1
    for key in SWEEP_KEYS:
        val = sim_params.get(key)
        if val is None:
            continue
        vals = expand_param(val)
        if len(vals) > 1:
            n *= len(vals)
    return n


def running_job_count() -> int:
    result = subprocess.run(
        ["squeue", "-h", "-u", USERNAME, "-t", "R,PD", "-o", "%i"],
        capture_output=True, text=True,
    )
    lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    return len(lines)


def submit(yaml_path: str, combo: int, run: int, dry_run: bool) -> None:
    cmd = ["sbatch", JOB_SCRIPT, yaml_path, str(combo), str(run)]
    if dry_run:
        print("  " + " ".join(cmd))
        return
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"  ERROR c={combo:04d} r={run:04d}: {result.stderr.strip()}")
    else:
        job_id = result.stdout.strip().split()[-1]
        log(f"  Submitted c={combo:04d} r={run:04d} → job {job_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit SNARE SLURM jobs (one job per combo×run) with throttling."
    )
    parser.add_argument("yaml", help="Path to experiment YAML file")
    parser.add_argument("--max-jobs", type=int, default=400,
                        help="Max concurrent SLURM jobs (default: 400)")
    parser.add_argument("--sleep", type=int, default=60,
                        help="Seconds to wait when throttled (default: 60)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sbatch commands without submitting")
    args = parser.parse_args()

    yaml_path = args.yaml
    if not Path(yaml_path).exists():
        sys.exit(f"Error: YAML not found: {yaml_path}")

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    n_runs = int(config["running"]["runs"])
    sim_params = config["simulation"]
    n_combos = count_combos(sim_params)
    total = n_combos * n_runs

    log(f"YAML       : {yaml_path}")
    log(f"Combos     : {n_combos}  ×  Runs: {n_runs}  =  {total} total jobs")
    log(f"Throttle   : {args.max_jobs} concurrent  |  sleep: {args.sleep}s")
    if args.dry_run:
        log("DRY RUN — no jobs will be submitted.")

    submitted = 0
    start = time.time()

    for combo in range(n_combos):
        for run in range(n_runs):
            if not args.dry_run:
                while True:
                    current = running_job_count()
                    if current < args.max_jobs:
                        break
                    log(f"Throttled: {current}/{args.max_jobs} active. "
                        f"Waiting {args.sleep}s… ({submitted}/{total} submitted)")
                    time.sleep(args.sleep)

            submit(yaml_path, combo, run, args.dry_run)
            submitted += 1

    elapsed = int(time.time() - start)
    log(f"Done. {submitted}/{total} jobs submitted in {elapsed // 60}m {elapsed % 60}s.")


if __name__ == "__main__":
    main()
