#!/usr/bin/env python3
"""
Merge per-job CSV outputs from a SLURM run into a single file.

Each SLURM job writes outputs/<stem>_c<combo>_r<run>.csv.
This script globs all partial files and concatenates them into outputs/<stem>.csv.

Usage:
    python merge_slurm_outputs.py outputs/sweep_z_fine_A.csv
    python merge_slurm_outputs.py outputs/sweep_z_fine_A.csv --delete
"""

import argparse
import glob
import os
import sys

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge SNARE SLURM partial CSVs into one output file."
    )
    parser.add_argument("output", help="Target merged CSV path, e.g. outputs/sweep_z_fine_A.csv")
    parser.add_argument("--delete", action="store_true",
                        help="Delete partial files after successful merge")
    args = parser.parse_args()

    base = args.output
    stem, ext = os.path.splitext(base)
    pattern = f"{stem}_c*{ext}"

    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"No partial files matching: {pattern}")

    print(f"Found {len(files)} partial files matching {pattern}")
    dfs = [pd.read_csv(f) for f in files]
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(base, index=False)
    print(f"Wrote {len(merged)} rows to {base}")

    if args.delete:
        for f in files:
            os.remove(f)
        print(f"Deleted {len(files)} partial files.")


if __name__ == "__main__":
    main()
