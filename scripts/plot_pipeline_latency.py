import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

stages  = [
    "Video & PLY\nexport",
    "Coordinate\nrotation",
    "MDM motion\ngeneration",
    "SMPL\nextraction",
    "HUGS\nrendering",
]
values  = [10.4, 0.3, 16.3, 49.6, 399.3]
total   = 475.9
pcts    = [v / total * 100 for v in values]

HIGHLIGHT = 4   # HUGS rendering index (bottom-to-top order)

colors = ["#aec6e8"] * len(stages)
colors[HIGHLIGHT] = "#d62728"
edge_colors = ["#6a9fc0"] * len(stages)
edge_colors[HIGHLIGHT] = "#8b0000"
lws = [0.8] * len(stages)
lws[HIGHLIGHT] = 2.0

fig, ax = plt.subplots(figsize=(9, 4.4))

y = np.arange(len(stages))
bars = ax.barh(y, values, height=0.55,
               color=colors, edgecolor=edge_colors, linewidth=lws, zorder=3)

# value + percentage labels
for i, (bar, val, pct) in enumerate(zip(bars, values, pcts)):
    x_end = bar.get_width()
    label = f"{val} s  ({pct:.1f}%)"
    ax.text(x_end + 4, bar.get_y() + bar.get_height() / 2,
            label, va="center", ha="left", fontsize=9.5,
            color="#8b0000" if i == HIGHLIGHT else "#333333",
            fontweight="bold" if i == HIGHLIGHT else "normal")

# "bottleneck" annotation arrow on HUGS bar
hugs_bar = bars[HIGHLIGHT]
ax.annotate("dominant\nbottleneck (83.9%)",
            xy=(hugs_bar.get_width() / 2, hugs_bar.get_y() + hugs_bar.get_height()),
            xytext=(200, HIGHLIGHT + 0.72),
            fontsize=9, color="#8b0000", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#8b0000", lw=1.3),
            ha="center")

ax.set_yticks(y)
ax.set_yticklabels(stages, fontsize=10.5)
ax.set_xlabel("Average latency (seconds)", fontsize=11)
ax.set_xlim(0, 480)
ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# total annotation
ax.axvline(total, color="#555555", linewidth=1.0, linestyle=":", alpha=0.7)
ax.text(total + 4, len(stages) - 0.65, f"Total\n{total} s",
        fontsize=8.5, color="#555555", va="top")

caption = (
    "Fig. 4 — System parameter: pipeline stage.\n"
    "HUGS rendering dominates latency; coordinate rotation adds negligible overhead."
)
fig.text(0.5, -0.06, caption, ha="center", fontsize=9.5, style="italic",
         color="#333333")

plt.tight_layout()
plt.savefig("pipeline_latency_chart.png", dpi=150, bbox_inches="tight")
print("Saved: pipeline_latency_chart.png")
