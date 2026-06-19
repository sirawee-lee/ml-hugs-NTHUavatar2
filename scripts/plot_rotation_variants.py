import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

variants = [
    r"V1  ($r_x{=}0°,\,r_z{=}0°$)",
    r"V2  ($r_x{=}+90°,\,r_z{=}0°$)",
    r"V3  ($r_x{=}0°,\,r_z{=}+180°$)",
    r"V4  ($r_x{=}+90°,\,r_z{=}+180°$)",
]
tilt = [90, 0, 90, 0]
fy   = [ 0,-1,  0, 1]

x = np.arange(len(variants))
w = 0.3

fig, ax1 = plt.subplots(figsize=(9, 5.2))
ax2 = ax1.twinx()

CORRECT = 3   # V4 index

colors_tilt = ["#d62728" if i != CORRECT else "#e87a7a" for i in range(4)]
colors_fy   = ["#1f77b4" if i != CORRECT else "#6ab0e0" for i in range(4)]
ec_tilt     = ["#8b0000" if i != CORRECT else "#c0392b" for i in range(4)]
ec_fy       = ["#0d3f6e" if i != CORRECT else "#1a6699" for i in range(4)]

tilt_plot = [v if v != 0 else 2.5 for v in tilt]   # stub for zero so bar is visible
bars_tilt = ax1.bar(x - w/2, tilt_plot, w,
                    color=colors_tilt, edgecolor=ec_tilt,
                    linewidth=[1.0]*3 + [2.2],
                    label=r"$\theta_{\mathrm{tilt}}$ (°) — left axis",
                    zorder=3)
bars_fy   = ax2.bar(x + w/2, fy, w,
                    color=colors_fy,   edgecolor=ec_fy,
                    linewidth=[1.0]*3 + [2.2],
                    label=r"$f_y$ score — right axis",
                    zorder=3)

# value labels
for bar, val in zip(bars_tilt, tilt):
    y_off = 2 if val >= 0 else -6
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + y_off if val >= 0 else y_off,
             f"{val}°", ha="center", va="bottom", fontsize=10,
             color=bar.get_edgecolor())

for bar, val in zip(bars_fy, fy):
    y_off = 0.05
    va = "bottom" if val >= 0 else "top"
    y_pos = val + (y_off if val >= 0 else -y_off)
    ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
             f"{val:+d}", ha="center", va=va, fontsize=10,
             color=bar.get_edgecolor())

# ideal reference lines
ax1.axhline(0,  color="#d62728", linewidth=1.0, linestyle="--", alpha=0.55,
            label=r"Ideal $\theta_{\mathrm{tilt}}=0°$")
ax2.axhline(1,  color="#1f77b4", linewidth=1.0, linestyle=":",  alpha=0.55,
            label=r"Ideal $f_y=+1$")

# highlight V4 background
ax1.axvspan(CORRECT - 0.55, CORRECT + 0.55,
            color="#2ecc71", alpha=0.10, zorder=0)
ax1.text(CORRECT, 96, "correct", ha="center", va="bottom",
         fontsize=9, color="#27ae60", fontweight="bold", zorder=5)

# axes styling
ax1.set_xlabel("Rotation variant", fontsize=12)
ax1.set_ylabel(r"$\theta_{\mathrm{tilt}}$ (degrees)", fontsize=12, color="#d62728")
ax2.set_ylabel(r"$f_y$ score", fontsize=12, color="#1f77b4")
ax1.tick_params(axis="y", colors="#d62728")
ax2.tick_params(axis="y", colors="#1f77b4")
ax1.set_ylim(-10, 105)
ax2.set_ylim(-1.6, 1.6)
ax1.set_yticks([0, 30, 60, 90])
ax2.set_yticks([-1, 0, 1])
ax1.set_xticks(x)
ax1.set_xticklabels(variants, fontsize=9.5)
ax1.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
ax1.set_axisbelow(True)
ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)

# combined legend
handles = [
    mpatches.Patch(color="#d62728", label=r"$\theta_{\mathrm{tilt}}$ (°) — left axis"),
    mpatches.Patch(color="#1f77b4", label=r"$f_y$ score — right axis"),
    plt.Line2D([0],[0], color="#d62728", linewidth=1, linestyle="--",
               label=r"Ideal $\theta_{\mathrm{tilt}}=0°$"),
    plt.Line2D([0],[0], color="#1f77b4", linewidth=1, linestyle=":",
               label=r"Ideal $f_y=+1$"),
]
ax1.legend(handles=handles, fontsize=9.5, loc="upper left",
           framealpha=0.9, ncol=2)

caption = (
    r"Fig. 2 — System parameter: rotation correction variant."
    "\n"
    r"Only V4 satisfies both $\theta_{\mathrm{tilt}}=0°$ and $f_y=+1$ simultaneously."
)
fig.text(0.5, -0.04, caption, ha="center", fontsize=9.5, style="italic",
         color="#333333")

plt.tight_layout()
plt.savefig("rotation_variants_chart.png", dpi=150, bbox_inches="tight")
print("Saved: rotation_variants_chart.png")
