"""
plot_social_exposure.py
Bar chart of ΔD and ΔA under social exposure, one model per family.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

OUT_PATH = "fig4.pdf"

# One representative per family (the topmost in the original table)
original_data = [
    ("gpt-5.4",          25.06, 69.222, "GPT"),
    ("claude-opus-4.7",  28.44, 70.312, "Claude"),
    ("gemini-3.1-flash", 32.55, 70.224, "Gemini"),
    ("grok-4.3",         27.42, 67.448, "Grok"),
    ("Qwen3.5-27B",      29.82, 67.342, "Qwen"),
    ("llama-4-scout",    23.6, 68.574, "Llama"),
]

round_1_data = [
    ("gpt-5.4",          23.45, 69.94, "GPT"),
    ("claude-opus-4.7",  28.20, 70.50, "Claude"),
    ("gemini-3.1-flash", 30.09, 70.13, "Gemini"),
    ("grok-4.3",         27.27, 68.26, "Grok"),
    ("Qwen3.5-27B",      27.95, 67.68, "Qwen"),
    ("llama-4-scout",    22.28, 69.71, "Llama"),
]



labels = [d[3] for d in original_data]

delta_D = np.array([
    r[1] - o[1]
    for o, r in zip(original_data, round_1_data)
])

delta_A = np.array([
    r[2] - o[2]
    for o, r in zip(original_data, round_1_data)
])

plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "legend.fontsize": 6,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
})

fig, ax = plt.subplots(figsize=(3.3, 2.2))

x = np.arange(len(labels))
w = 0.36

# Bars
bars_D = ax.bar(x - w/2, delta_D, w, label=r"$\Delta D$ (diversity)",
                color="#D85A30", edgecolor="white", linewidth=0.4)
bars_A = ax.bar(x + w/2, delta_A, w, label=r"$\Delta A$ (alignment)",
                color="#378ADD", edgecolor="white", linewidth=0.4)

# Zero baseline
ax.axhline(0, color="black", linewidth=0.5, zorder=1)

# Annotate each bar with its value
for bars, vals in [(bars_D, delta_D), (bars_A, delta_A)]:
    for bar, v in zip(bars, vals):
        offset = 0.10 if v >= 0 else -0.10
        va = "bottom" if v >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width()/2, v + offset,
                f"{v:+.2f}", ha="center", va=va, fontsize=5,
                color="#333")

# Axis cosmetics
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=6)
ax.set_ylabel(r"$\Delta$ from static system", labelpad=2)
ax.set_ylim(-3.4, 2.0)
ax.legend(loc="upper left", frameon=False, ncol=2,
          handletextpad=0.3, columnspacing=0.8, borderpad=0.2,
          bbox_to_anchor=(0.0, 1.02))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis='both', pad=2)
ax.grid(axis="y", alpha=0.2, linewidth=0.4)
ax.set_axisbelow(True)

plt.subplots_adjust(left=0.17, right=0.97, top=0.93, bottom=0.28)
plt.savefig(OUT_PATH, bbox_inches="tight", pad_inches=0.02)
print(f"Saved {OUT_PATH}")