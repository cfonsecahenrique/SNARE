import pandas as pd
import altair as alt
import numpy as np

csv_path = "outputs/consensus_sweep_results.csv"
df = pd.read_csv(csv_path)

# Filter: Z >= 40, alpha == 0.01
df = df[(df["Z"] >= 40) & (df["alpha"] == 0.01)]
df = df[df["eps"] == 0.01]
df = df[df["gamma_center"] == 1]

for val in np.arange(0, 1.1, 0.1):
    print(df[df.consensus_thresh == val].q.value_counts())

# Average cooperation ratio per (consensus_thresh, q) pair
heatmap_df = (
    df.groupby(["consensus_thresh", "q"])["average_cooperation"]
    .mean()
    .reset_index()
)

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
    .properties(width=350, height=350, title="Average Cooperation by κ_c and q")
)

# Add text labels inside each cell
text = chart.mark_text(baseline="middle", fontSize=13).encode(
    text=alt.Text("average_cooperation:Q", format=".2f"),
    color=alt.condition(
        alt.datum.average_cooperation > heatmap_df["average_cooperation"].median(),
        alt.value("black"),
        alt.value("white"),
    ),
)

final = (chart + text).configure_axis(
    labelFontSize=13, titleFontSize=14
).configure_title(fontSize=16)

# Generate separate colorbar
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

suffix = "action" if "action" in csv_path else "emotion"
heatmap_path = f"outputs/consensus_heatmap_{suffix}.png"
colorbar_path = f"outputs/consensus_colorbar_{suffix}.png"

final.save(heatmap_path, ppi=200)
colorbar.save(colorbar_path, ppi=200)

print(f"Saved heatmap to {heatmap_path}")
print(f"Saved colorbar to {colorbar_path}")
