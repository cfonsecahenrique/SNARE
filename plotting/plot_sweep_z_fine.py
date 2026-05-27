"""
plot_sweep_z_fine.py
====================
Aggregate the fine-grained population-size sweep (Z = 40..130, step 10,
G = 40*Z proportionally scaled generations) and produce a publication-quality
robustness figure.

Data: outputs/sweep_z_fine_c000X.csv  (one file per Z value, 25 runs each)

Output:
  plotting/plots/robustness/sweep_z_fine.png
  (also copied to both paper figs/ directories)
"""

import os
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR   = SCRIPT_DIR.parent
OUTPUTS    = ROOT_DIR / "outputs"
OUT_DIR    = SCRIPT_DIR / "plots" / "robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAPER_ROOT = ROOT_DIR.parent / "Emotion as a Solution to Private Assessment"
COPY_DIRS  = [
    PAPER_ROOT / "PRSB" / "figs",
    PAPER_ROOT / "PNAS" / "figs",
]

mpl.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          12,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})


def load_fine_sweep() -> pd.DataFrame:
    chunks = []
    for path in sorted(OUTPUTS.glob("sweep_z_fine_c[0-9]*.csv")):
        # Skip per-replica files (e.g. sweep_z_fine_c0000_r0001.csv)
        if "_r" in path.stem:
            continue
        df = pd.read_csv(path)
        chunks.append(df)
    if not chunks:
        raise FileNotFoundError("No sweep_z_fine_c*.csv files found in outputs/")
    return pd.concat(chunks, ignore_index=True)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Z"]                   = df["Z"].astype(int)
    df["average_cooperation"] = df["average_cooperation"].astype(float)
    agg = (
        df.groupby("Z")["average_cooperation"]
        .agg(mean="mean", std="std", sem="sem", count="count")
        .reset_index()
        .sort_values("Z")
    )
    return agg


def make_figure(agg: pd.DataFrame) -> plt.Figure:
    color = plt.cm.viridis(0.5)
    reg_color = "#444444"

    z    = agg["Z"].to_numpy()
    mean = agg["mean"].to_numpy()
    sem  = agg["sem"].to_numpy()

    # Linear regression on the per-Z means
    slope, intercept = np.polyfit(z, mean, 1)
    z_fit   = np.linspace(z.min(), z.max(), 200)
    y_fit   = slope * z_fit + intercept

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.fill_between(
        z,
        mean - sem,
        mean + sem,
        alpha=0.25,
        color=color,
        label=r"$\pm 1$ SEM",
    )
    ax.plot(
        z, mean,
        marker="o", markersize=7, linewidth=2.5,
        color=color,
        label="Mean cooperation",
    )
    ax.plot(
        z_fit, y_fit,
        linewidth=1.5, linestyle="--", color=reg_color,
        label=rf"Linear fit ($\beta={slope:.2f}$%/individual)",
    )

    ax.set_xlabel(r"$Z$ (Population size)", fontsize=13)
    ax.set_ylabel(r"Average cooperation $\eta$ (%)", fontsize=13)
    ax.set_ylim(0, 100)
    ax.set_xlim(z.min() - 5, z.max() + 5)
    ax.set_xticks(z.tolist())

    fixed_info = (
        r"Fixed: $q=0.8$,  $\tilde{\kappa}=0.8$,  $\xi=0.01$,  "
        r"$\alpha=0$,  $\mu=2/Z$;  $G = 40Z$"
    )
    ax.text(
        0.98, 0.98, fixed_info,
        transform=ax.transAxes,
        fontsize=7.5, color="grey",
        ha="right", va="top",
    )

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False, fontsize=10, loc="upper right")

    plt.tight_layout()
    return fig


def main() -> None:
    df  = load_fine_sweep()
    agg = aggregate(df)

    print("Fine Z sweep — per-Z summary:")
    for _, row in agg.iterrows():
        print(
            f"  Z={int(row['Z']):3d}  n={int(row['count'])}  "
            f"mean={row['mean']:5.1f}%  SD={row['std']:5.1f}%  SEM={row['sem']:4.2f}%"
        )

    fig = make_figure(agg)

    out_path = OUT_DIR / "sweep_z_fine.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    for dest_dir in COPY_DIRS:
        if dest_dir.exists():
            dest = dest_dir / "sweep_z.png"
            shutil.copy2(out_path, dest)
            print(f"Copied to {dest}")
        else:
            print(f"Skipped (dir not found): {dest_dir}")


if __name__ == "__main__":
    main()
