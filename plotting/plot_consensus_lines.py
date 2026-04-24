import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import csv

csv_path = "outputs/consensus_sweep_results.csv"

# Load CSV (now includes 'xi' column)
df = pd.read_csv(csv_path)
for col in ['xi', 'consensus_thresh', 'q', 'alpha', 'gamma_center', 'average_cooperation']:
    df[col] = df[col].astype(float)
df['Z'] = df['Z'].astype(int)
df['consensus_thresh'] = df['consensus_thresh'].round(2)
df['q'] = df['q'].round(2)

# Base filter (common across all xi)
Q_VALS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
alpha = 0.01
df = df[
    (df['alpha'] == alpha) &
    (df['gamma_center'] == 1) &
    (df['Z'] == 50) &
    (df['q'].isin(Q_VALS)) &
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

    ax.set_xlabel(r'$\kappa_c$ (Consensus Threshold)', fontsize=13)
    ax.set_ylabel('Average Cooperation', fontsize=13)
    ax.set_title(rf'Average Cooperation vs. $\kappa_c$  ($\xi = {xi}$)', fontsize=14)
    ax.set_xticks(Q_VALS)
    ax.set_ylim(0, 100)
    ax.legend(title='Observability (q)', frameon=False, fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    xi_str = str(xi).replace('.', '')
    out_path = f"plotting/plots/consensus_thresh_observability_xi{xi_str}_alpha{alpha}.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
