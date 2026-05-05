"""
plot_robustness_lines.py
========================
Generates line plots for every sweep_*_results.csv robustness check.
Style mirrors plot_consensus_lines.py (viridis palette, SEM bands, etc.)

Each sweep YAML fixes q=0.8, consensus_thresh=0.8.  We plot:
    x-axis  → swept parameter value
    y-axis  → average_cooperation (mean ± SEM across runs)
    lines   → one line per unique value of the *other* key nuisance variable
              (or a single aggregate line when there is no natural grouping variable)

Sweeps covered
--------------
  sweep_benefit  : b   ∈ {1, 3, 5, 7, 10}  (fixed: chi, eps, alpha, beta, Z, mu, xi)
  sweep_beta     : beta∈ {0.1, 0.5, 1, 5, 10}
  sweep_chi      : chi ∈ {0, 0.001, 0.01, 0.05, 0.1}
  sweep_eps      : eps ∈ {0, 0.001, 0.01, 0.05, 0.1}
  sweep_z        : Z   ∈ {50, 100, 200}          (only values present in the CSV)
  sweep_alpha    : alpha ∈ {0, 0.1, 0.25, 0.5}   (example values, dynamically plotted)
  sweep_mu       : mu  ∈ {0.001, 0.01, 0.05}     (example values, dynamically plotted)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Get absolute path to the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# ── Output directory ────────────────────────────────────────────────────────
OUT_DIR = os.path.join(SCRIPT_DIR, "plots", "robustness")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style (shared with plot_consensus_lines.py) ──────────────────────────────
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── Sweep definitions ────────────────────────────────────────────────────────
# Each entry:
#   csv        – path to results file
#   x_col      – column being swept (x-axis)
#   x_label    – LaTeX / plain-text axis label
#   title      – figure title
#   group_col  – column used to split into multiple lines (None → single line)
#   group_label– legend title for group_col (ignored when group_col is None)
#   x_dtype    – 'float' or 'int'
SWEEPS = [
    dict(
        csv="outputs/sweep_benefit_results.csv",
        x_col="b",
        x_label=r"$b$ (Benefit)",
        title=r"Average Cooperation vs. Benefit ($b$)",
        group_col=None,
        group_label=None,
        x_dtype="float",
        out_name="sweep_benefit",
    ),
    dict(
        csv="outputs/sweep_beta_results.csv",
        x_col="beta",
        x_label=r"$\beta$ (Selection Strength)",
        title=r"Average Cooperation vs. Selection Strength ($\beta$)",
        group_col=None,
        group_label=None,
        x_dtype="float",
        out_name="sweep_beta",
    ),
    dict(
        csv="outputs/sweep_chi_results.csv",
        x_col="chi",
        x_label=r"$\chi$ (Reputation Assessment Error)",
        title=r"Average Cooperation vs. Reputation Assessment Error ($\chi$)",
        group_col=None,
        group_label=None,
        x_dtype="float",
        out_name="sweep_chi",
    ),
    dict(
        csv="outputs/sweep_eps_results.csv",
        x_col="eps",
        x_label=r"$\varepsilon$ (Execution Error)",
        title=r"Average Cooperation vs. Execution Error ($\varepsilon$)",
        group_col=None,
        group_label=None,
        x_dtype="float",
        out_name="sweep_eps",
    ),
    dict(
        csv="outputs/sweep_z_results.csv",
        x_col="Z",
        x_label=r"$Z$ (Population Size)",
        title=r"Average Cooperation vs. Population Size ($Z$)",
        group_col=None,
        group_label=None,
        x_dtype="int",
        out_name="sweep_z",
    ),
    dict(
        csv="outputs/sweep_alpha_results.csv",
        x_col="alpha",
        x_label=r"$\alpha$ (Reputation Assignment Error)",
        title=r"Average Cooperation vs. Reputation Assignment Error ($\alpha$)",
        group_col=None,
        group_label=None,
        x_dtype="float",
        out_name="sweep_alpha",
    ),
    dict(
        csv="outputs/sweep_mu_results.csv",
        x_col="mu",
        x_label=r"$\mu$ (Mutation Rate)",
        title=r"Average Cooperation vs. Mutation Rate ($\mu$)",
        group_col=None,
        group_label=None,
        x_dtype="float",
        out_name="sweep_mu",
    ),
]


def plot_sweep(cfg: dict) -> None:
    try:
        df = pd.read_csv(cfg["csv"])
    except pd.errors.ParserError:
        print(f"Warning: ParserError in {cfg['csv']}, skipping bad lines.")
        df = pd.read_csv(cfg["csv"], on_bad_lines='skip')

    # Cast swept column
    if cfg["x_dtype"] == "int":
        df[cfg["x_col"]] = df[cfg["x_col"]].astype(int)
    else:
        df[cfg["x_col"]] = df[cfg["x_col"]].astype(float)

    df["average_cooperation"] = df["average_cooperation"].astype(float)

    # ── Filter for requested xi and alpha ────────────────────────────────────
    if "xi" in df.columns and cfg["x_col"] != "xi":
        df = df[df["xi"] == 0.01]
    if "alpha" in df.columns and cfg["x_col"] != "alpha":
        df = df[df["alpha"] == 0]
    
    if df.empty:
        print(f"Warning: No data found for xi=0.01 and alpha=0 in {cfg['csv']}")
        return

    # ── Aggregate ────────────────────────────────────────────────────────────
    if cfg["group_col"] is not None:
        df[cfg["group_col"]] = df[cfg["group_col"]].astype(float)
        group_vals = sorted(df[cfg["group_col"]].unique())
        palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(group_vals)))

        agg = (
            df.groupby([cfg["x_col"], cfg["group_col"]])["average_cooperation"]
            .agg(["mean", "sem"])
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(7, 5))
        for gval, color in zip(group_vals, palette):
            sub = agg[agg[cfg["group_col"]] == gval].sort_values(cfg["x_col"])
            ax.plot(
                sub[cfg["x_col"]], sub["mean"],
                marker="o", markersize=6, linewidth=2,
                color=color, label=f"{cfg['group_col']} = {gval}",
            )
            ax.fill_between(
                sub[cfg["x_col"]],
                sub["mean"] - sub["sem"],
                sub["mean"] + sub["sem"],
                alpha=0.15, color=color,
            )
        ax.legend(title=cfg["group_label"], frameon=False, fontsize=11)

    else:
        agg = (
            df.groupby(cfg["x_col"])["average_cooperation"]
            .agg(["mean", "sem"])
            .reset_index()
            .sort_values(cfg["x_col"])
        )

        fig, ax = plt.subplots(figsize=(7, 5))
        color = plt.cm.viridis(0.5)
        ax.plot(
            agg[cfg["x_col"]], agg["mean"],
            marker="o", markersize=7, linewidth=2.5, color=color,
        )
        ax.fill_between(
            agg[cfg["x_col"]],
            agg["mean"] - agg["sem"],
            agg["mean"] + agg["sem"],
            alpha=0.2, color=color,
        )

    # ── Cosmetics ────────────────────────────────────────────────────────────
    ax.set_xlabel(cfg["x_label"], fontsize=13)
    ax.set_ylabel("Average Cooperation", fontsize=13)
    ax.set_title(cfg["title"], fontsize=14)
    ax.set_ylim(0, 100)
    ax.set_xticks(sorted(df[cfg["x_col"]].unique()))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate fixed params from YAML in a small text box
    fixed_info = (
        r"Fixed: $q=0.8$, $\tilde{k}=0.8$, $\xi=0.01$, "
        r"$\alpha=0$, $\mu/Z=1$"
    )
    ax.text(
        0.02, 0.02, fixed_info,
        transform=ax.transAxes,
        fontsize=8, color="grey",
        verticalalignment="bottom",
    )

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{cfg['out_name']}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Run all sweeps ────────────────────────────────────────────────────────────
for sweep_cfg in SWEEPS:
    csv_path = os.path.join(ROOT_DIR, sweep_cfg["csv"])
    if not os.path.exists(csv_path):
        print(f"Skipping (file not found): {csv_path}")
        continue
    sweep_cfg["csv"] = csv_path
    plot_sweep(sweep_cfg)
