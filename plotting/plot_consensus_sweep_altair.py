"""
Plot: Average Cooperation Rate vs consensus_thresh and observability
Same chart as plot_consensus_sweep.py, rebuilt in Altair / Vega-Lite.
Outputs both an interactive HTML and a static PNG (via vl-convert).
"""

import os
import pandas as pd
import altair as alt

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)

CSV_PATH      = os.path.join(REPO_ROOT, "outputs", "consensus_sweep_results.csv")
OUT_DIR       = os.path.join(SCRIPT_DIR, "plots")
OUT_HTML      = os.path.join(OUT_DIR, "sweep_consensus_thresh_observability_altair.html")
OUT_PNG       = os.path.join(OUT_DIR, "sweep_consensus_thresh_observability_altair.png")

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)


# ── load & aggregate ─────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

y_scale = 100.0
# Calculate mean and standard error (sem)
grouped = (
    df.groupby(["consensus_thresh", "q"])["average_cooperation"]
    .agg(["mean", "sem"])
    .reset_index()
    .rename(columns={"q": "observability",
                     "mean": "avg_cooperation",
                     "sem": "avg_cooperation_sem"})
)

# normalise 0-100 -> 0-1
grouped["avg_cooperation"] /= y_scale
grouped["avg_cooperation_sem"] /= y_scale

# Calculated lower and upper bounds for error bars
grouped["ymin"] = grouped["avg_cooperation"] - grouped["avg_cooperation_sem"]
grouped["ymax"] = grouped["avg_cooperation"] + grouped["avg_cooperation_sem"]


# Make observability a string for a clean discrete colour legend
grouped["observability_label"] = grouped["observability"].apply(
    lambda v: f"observability={v}"
)

# ── colour scale matching the matplotlib version ──────────────────────────────
obs_order = [0.0, 0.25, 0.5, 0.75, 1.0]
label_order = [f"observability={v}" for v in obs_order]
palette     = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

color_scale = alt.Scale(domain=label_order, range=palette)

# ── chart ─────────────────────────────────────────────────────────────────────
base = alt.Chart(grouped).encode(
    x=alt.X(
        "consensus_thresh:Q",
        axis=alt.Axis(title="consensus_thresh", tickCount=5),
        scale=alt.Scale(domain=[-0.05, 1.05]),
    ),
    y=alt.Y(
        "avg_cooperation:Q",
        axis=alt.Axis(title="Average Cooperation Rate", tickCount=6),
        scale=alt.Scale(domain=[0, 1]),
    ),
    color=alt.Color(
        "observability_label:N",
        sort=label_order,
        scale=color_scale,
        legend=alt.Legend(title=None, orient="top-left"),
    ),
    order=alt.Order("observability:Q"),   # draw low-obs lines first (blue under purple)
)

lines = base.mark_line(point=True, strokeWidth=1.8)

# Add error bars
errorbars = base.mark_errorbar(extent='ci').encode(
    y=alt.Y("ymin:Q"),
    y2=alt.Y2("ymax:Q"),
)

chart = (
    (lines + errorbars)
    .encode(y=alt.Y("avg_cooperation:Q", title="Average Cooperation Rate"))
    .properties(
        width=560,
        height=340,
        title=alt.TitleParams(
            text="Average Cooperation Rate vs consensus_thresh and observability",
            subtitle="EB Social Norm: GGBBBBGG",
            fontSize=13,
            subtitleFontSize=11,
        ),
    )
    .configure_axis(grid=True, gridOpacity=0.7)
    .configure_view(strokeWidth=0)
)

# ── save HTML (interactive) ───────────────────────────────────────────────────
chart.save(OUT_HTML)
print(f"HTML saved -> {OUT_HTML}")

# ── save PNG via vl-convert ───────────────────────────────────────────────────
try:
    import vl_convert as vlc
    vega_lite_spec = chart.to_dict()
    png_bytes = vlc.vegalite_to_png(vl_spec=vega_lite_spec, scale=2)
    with open(OUT_PNG, "wb") as f:
        f.write(png_bytes)
    print(f"PNG saved  -> {OUT_PNG}")
except Exception as e:
    print(f"PNG export skipped ({e}). Open the HTML file for the interactive version.")
