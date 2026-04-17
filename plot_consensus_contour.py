import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = "outputs/consensus_sweep_results.csv"
df = pd.read_csv(csv_path)

# Filter: Z >= 40, alpha == 0
df = df[(df["Z"] >= 40) & (df["alpha"] == 0)]
df = df[df["eps"] == 0.01]
df = df[df["gamma_center"] == 1]

# Average cooperation ratio per (consensus_thresh, q) pair
heatmap_df = (
    df.groupby(["consensus_thresh", "q"])["average_cooperation"]
    .mean()
    .reset_index()
)

# Pivot to create a grid for contour plotting
grid_df = heatmap_df.pivot(index="q", columns="consensus_thresh", values="average_cooperation")

# It's possible some combinations are missing, so we'll fill or just use tricontourf if needed. 
# Assuming a complete grid based on the heatmap behavior.
X, Y = np.meshgrid(grid_df.columns, grid_df.index)
Z = grid_df.values

plt.figure(figsize=(6, 5))

# Plot filled contours with many levels for smooth blending
levels = np.linspace(0, 100, 101)
contour_filled = plt.contourf(X, Y, Z, levels=levels, cmap="RdBu", vmin=0, vmax=100)
# Optional: Add contour lines on top
contour_lines = plt.contour(X, Y, Z, levels=np.linspace(0, 100, 11), colors="black", linewidths=0.5, alpha=0.5)
plt.clabel(contour_lines, inline=True, fontsize=8, fmt="%.0f")

cbar = plt.colorbar(contour_filled, ticks=np.linspace(0, 100, 11))
cbar.set_label("Avg Cooperation")

plt.xlabel("κ_c (Consensus Threshold)")
plt.ylabel("q (Observability)")
plt.title("Average Cooperation by κ_c and q")

suffix = "action" if "action" in csv_path else "emotion"
contour_path = f"outputs/consensus_contour_{suffix}.png"

plt.savefig(contour_path, dpi=200, bbox_inches="tight")
print(f"Saved contour plot to {contour_path}")
