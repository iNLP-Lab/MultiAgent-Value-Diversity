"""
plot_per_question_AD.py  (v2, split output)
Per-question (D, A) scatter — saves two separate PDFs for PPT composition.
"""

import json, os, numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
from utils import normalize_latex_text, extract_boxed_answer

CULTURES = ["BRA", "CHN", "MEX", "NGA", "NZL"]
OUTPUT_DIR = "wvs_evaluation"
MODELS = {
    "grok-3":         {"color": "#d62728", "label": "grok-3 system (A=70.9, D=25.2)"},
    "gemini-2.5-pro": {"color": "#1f77b4", "label": "gemini-2.5-pro system (A=67.7, D=36.1)"},
}

with open("data/wvs.json") as f: references = json.load(f)
ref_dict = {r["Q_id"]: r for r in references}
with open("data/proportions_group_by_country.json") as f: culture_data = json.load(f)

def majority(d): return int(max(d.items(), key=lambda x: x[1])[0])

def load_model(model):
    out = {}
    for c in CULTURES:
        with open(f"{OUTPUT_DIR}/{model}_{c}_system.jsonl") as f:
            data = [json.loads(l) for l in f]
        data.sort(key=lambda x: x["idx"])
        per_q = {}
        for id_, item in enumerate(data):
            ref = references[id_]
            try:
                a = int(normalize_latex_text(extract_boxed_answer(item["response"]).lower())) + 1
            except Exception: continue
            per_q[ref["Q_id"]] = a
        out[c] = per_q
    return out

def load_gt():
    return {c: {q: majority(d) for q, d in culture_data[c].items() if d and q in ref_dict}
            for c in CULTURES}

def per_question_AD(answers, gt):
    common = set(answers[CULTURES[0]].keys())
    for c in CULTURES[1:]: common &= set(answers[c].keys())
    rows = []
    for q in common:
        delta = len(ref_dict[q]["option"]) - 1
        if delta == 0: continue
        vals = [answers[c][q] for c in CULTURES]
        D_q = np.mean([abs(a - b) / delta for a, b in combinations(vals, 2)])
        if any(q not in gt[c] for c in CULTURES): continue
        A_q = np.mean([1 - abs(answers[c][q] - gt[c][q]) / delta for c in CULTURES])
        rows.append((q, D_q, A_q))
    return rows

# Compute
gt = load_gt()
results = {}
for m in MODELS:
    print(f"loading {m}...")
    ans = load_model(m)
    results[m] = per_question_AD(ans, gt)
    Ds = [r[1] for r in results[m]]; As = [r[2] for r in results[m]]
    print(f"  n={len(results[m])}, mean D_q={np.mean(Ds):.3f}, mean A_q={np.mean(As):.3f}")

# Determine global axis bounds (shared by both panels so they look comparable in PPT)
all_D = np.concatenate([[r[1] for r in results[m]] for m in MODELS])
all_A = np.concatenate([[r[2] for r in results[m]] for m in MODELS])
D_max = min(1.0, np.percentile(all_D, 99) + 0.05)
A_min = max(0.0, np.percentile(all_A,  1) - 0.05)

# Plot
plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "legend.fontsize": 6,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
})
os.makedirs("Fig", exist_ok=True)

for m, info in MODELS.items():
    fig, ax = plt.subplots(figsize=(3.0, 1.6))
    rows = results[m]
    Ds = np.array([r[1] for r in rows])
    As = np.array([r[2] for r in rows])

    # === Quadrant shading: high-A/low-D region (= the "alignment-wins, no plurality" zone)
    ax.axvspan(0, np.mean([r[1] for r in results["grok-3"]]),
               ymin=(np.mean([r[2] for r in results["grok-3"]]) - A_min) / (1.0 - A_min),
               color="#ffe6e6", alpha=0.5, zorder=0, label=None)
    # high-D / low-A region (= "plurality but off-target")
    ax.axvspan(np.mean([r[1] for r in results["gemini-2.5-pro"]]), 1.0,
               ymax=(np.mean([r[2] for r in results["gemini-2.5-pro"]]) - A_min) / (1.0 - A_min),
               color="#e6efff", alpha=0.5, zorder=0, label=None)

    # === Bubble scatter: size encodes #questions at this (D, A)
    counter = Counter(zip(Ds.round(3), As.round(3)))
    xs, ys, sizes = [], [], []
    for (d, a), n in counter.items():
        xs.append(d); ys.append(a); sizes.append(6 + 4 * n)
    ax.scatter(xs, ys, s=sizes, alpha=0.55,
               color=info["color"], edgecolor="white", linewidth=0.3, zorder=3)

    # Mean cross-hair
    mD, mA = np.mean(Ds), np.mean(As)
    ax.axvline(mD, color=info["color"], linestyle=":", linewidth=0.9, alpha=0.7, zorder=2)
    ax.axhline(mA, color=info["color"], linestyle=":", linewidth=0.9, alpha=0.7, zorder=2)
    ax.plot([mD], [mA], marker="*", markersize=8, color=info["color"],
            markeredgecolor="white", markeredgewidth=0.8, zorder=4,
            label=f"mean: D={mD:.2f}, A={mA:.2f}")

    ax.set_title(info["label"], loc="left", fontweight="bold", pad=2, fontsize=7)
    ax.set_xlabel(r"Per-question diversity  $D_q$", labelpad=2)
    ax.set_ylabel(r"Per-question alignment  $A_q$", labelpad=3)
    ax.set_xlim(-0.02, D_max)
    ax.set_ylim(A_min, 1.02)
    ax.legend(frameon=False, loc="lower left", fontsize=5.5,
              handletextpad=0.3, borderpad=0.2, labelspacing=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)

    # Annotate the diagnostic quadrant for this model
    if m == "grok-3":
        ax.text(0.05, 0.95, "aligned but\nhomogeneous",
                fontsize=6, color="#a02020", va="top", ha="left", style="italic",
                transform=ax.transAxes)
    else:
        ax.text(0.98, 0.04, "diverse but\nmisaligned",
                fontsize=6, color="#1a4080", va="bottom", ha="right", style="italic",
                transform=ax.transAxes)

    plt.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.22)
    out_path = f"fig1_right_{m}.pdf"
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {out_path}")