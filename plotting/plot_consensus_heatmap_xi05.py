import pandas as pd
import altair as alt
import numpy as np
import csv

csv_path = "outputs/consensus_sweep_results.csv"

# Load CSV (now includes 'xi' column)
df = pd.read_csv(csv_path)
df['xi'] = df['xi'].astype(float)
df['consensus_thresh'] = round(df['consensus_thresh'].astype(float), 2)
df['Z'] = df['Z'].astype(int)
df['alpha'] = df['alpha'].astype(float)
df['eps'] = df['eps'].astype(float)
df['gamma_center'] = df['gamma_center'].astype(float)
df['q'] = round(df['q'].astype(float), 2)
df['average_cooperation'] = df['average_cooperation'].astype(float)

# Filter: Z >= 40, alpha == 0
df = df[(df["Z"] == 50) & (df["alpha"] == 0)]
# And the user wants xi == 0.05
df = df[df["xi"] == 0.05]
df = df[df["gamma_center"] == 1]
# We should filter consensus_thresh between 0 and 1
df = df[(df["consensus_thresh"] >= 0) & (df["consensus_thresh"] <= 1.0)]

# Average cooperation ratio per (consensus_thresh, q) pair
heatmap_df = (
    df.groupby(["consensus_thresh", "q"])["average_cooperation"]
    .mean()
    .reset_index()
)

heatmap_df.average_cooperation = round(heatmap_df.average_cooperation, 1)

chart = (
    alt.Chart(heatmap_df)
    .mark_rect()
    .encode(
        x=alt.X("consensus_thresh:O", title="κ_c (Consensus Threshold)"),
        y=alt.Y("q:O", title="q (Observability)", sort="descending"),
        color=alt.Color(
            "average_cooperation:Q",
            title="Avg Cooperation",
            scale=alt.Scale(scheme="redblue", domain=[0, 100]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("consensus_thresh:O", title="κ_c"),
            alt.Tooltip("q:O", title="q"),
            alt.Tooltip("average_cooperation:Q", title="Avg Cooperation", format=".3f"),
        ],
    )
    .properties(width=350, height=350, title="Average Cooperation by κ_c and q (ξ=0.05)")
)

# Add text labels inside each cell
text = chart.mark_text(baseline="middle", fontSize=13).encode(
    text=alt.Text("average_cooperation:Q", format=".1f"),
    color=alt.condition(
        alt.datum.average_cooperation > heatmap_df["average_cooperation"].median(),
        alt.value("black"),
        alt.value("white"),
    ),
)

final = (chart + text).configure_axis(
    labelFontSize=13, titleFontSize=14
).configure_title(fontSize=16)

colorbar = (
    alt.Chart(heatmap_df)
    .mark_rect()
    .encode(
        color=alt.Color(
            "average_cooperation:Q",
            title="Avg Cooperation",
            scale=alt.Scale(scheme="redblue", domain=[0, 100]),
        )
    )
    .properties(width=0, height=0)
    .configure_view(strokeWidth=0)
    .configure_axis(labels=False, ticks=False, domain=False)
)

heatmap_path = f"outputs/consensus_heatmap_xi05.png"
colorbar_path = f"outputs/consensus_colorbar_xi05.png"

final.save(heatmap_path, ppi=200)
colorbar.save(colorbar_path, ppi=200)

print(f"Saved heatmap to {heatmap_path}")
print(f"Saved colorbar to {colorbar_path}")
