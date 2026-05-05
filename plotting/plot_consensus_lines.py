import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import csv
import os

# Resolve paths relative to this script so it works from any CWD
_HERE = os.path.dirname(os.path.abspath(__file__))
_SNARE_ROOT = os.path.dirname(_HERE)  # SNARE/

BASE_NORM = "sj"

if BASE_NORM != "sj":
    csv_path = os.path.join(_SNARE_ROOT, "outputs", f"{BASE_NORM}_sweep.csv")
elif BASE_NORM == "sj":
    csv_path = os.path.join(_SNARE_ROOT, "outputs", "xi_robustness.csv")

# The CSV is a mix of an old 30-column schema and the current 32-column schema
# (aux_functions now exports 8 combined strategy+EP labels instead of 4+2 separate ones).
# Provide the current column names explicitly and skip the stale header row.
col_names = [
    'base_social_norm', 'eb_social_norm', 'Z', 'gens', 'mu', 'chi', 'eps', 'alpha',
    'q', 'consensus_thresh', 'xi', 'non_consensus_strategy', 'b', 'c', 'beta',
    'convergence_period', 'gamma_min', 'gamma_max', 'gamma_delta',
    'gamma_center', 'average_cooperation', 'average_consensus', 'G',
    'AllD_Comp', 'AllD_Coop', 'Disc_Comp', 'Disc_Coop',
    'pDisc_Comp', 'pDisc_Coop', 'AllC_Comp', 'AllC_Coop',
]
df = pd.read_csv(csv_path, names=col_names, skiprows=1, on_bad_lines='skip')

for col in ['xi', 'consensus_thresh', 'q', 'alpha', 'gamma_center', 'average_cooperation']:
    df[col] = df[col].astype(float)
df['Z'] = df['Z'].astype(int)
df['consensus_thresh'] = df['consensus_thresh'].round(2)
df['q'] = df['q'].round(2)

# Base filter (common across all xi)
Q_VALS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
alpha = 0.0
df = df[
    (df['alpha'] == alpha) &
    (df['gamma_center'] == 1) &
    (df['Z'] == 50) &
    (df['q'].isin(Q_VALS)) &
    (df['mu'] == 2) &
    (df['gens'] == 3000) &
    (df['consensus_thresh'].isin(Q_VALS))
]

xi_vals = sorted(df['xi'].unique())
print(f"xi values found: {xi_vals}")

# ── Style ──────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

for xi in xi_vals:
    sub_df = df[df['xi'] == xi]

    agg = (
        sub_df.groupby(['consensus_thresh', 'q'])['average_cooperation']
        .agg(['mean', 'sem'])
        .reset_index()
    )

    q_vals = sorted(agg['q'].unique())
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(q_vals)))

    fig, ax = plt.subplots(figsize=(7, 5))

    for q, color in zip(q_vals, palette):
        q_data = agg[agg['q'] == q].sort_values('consensus_thresh')
        ax.plot(q_data['consensus_thresh'], q_data['mean'],
                marker='o', markersize=6, linewidth=2,
                color=color, label=f'q = {q}')
        ax.fill_between(q_data['consensus_thresh'],
                        q_data['mean'] - q_data['sem'],
                        q_data['mean'] + q_data['sem'],
                        alpha=0.15, color=color)

    ax.set_xlabel(r'$\tilde{k}$ (Consensus Threshold)', fontsize=13)
    ax.set_ylabel('Average Cooperation', fontsize=13)
    ax.set_title(rf'Average Cooperation vs. $\tilde{{k}}$  ($\xi = {xi}$)', fontsize=14)
    ax.set_xticks(Q_VALS)
    ax.set_ylim(0, 100)
    ax.legend(title='Observability (q)', frameon=False, fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    xi_str = str(xi).replace('.', '')
    out_path = os.path.join(_HERE, "plots", f"{BASE_NORM}_consensus_thresh_observability_xi{xi_str}_alpha{alpha}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
