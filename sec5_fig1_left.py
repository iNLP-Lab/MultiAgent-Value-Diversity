import json
import math
import os
from collections import defaultdict
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.sparse.csgraph import minimum_spanning_tree

from utils import (
    normalize_latex_text,
    extract_boxed_answer,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================================================
# Config
# =========================================================

OUTPUT_DIR = "wvs_evaluation"

# ---------------------------------------------------------
# Choose diversity metric here
# ---------------------------------------------------------

DIVERSITY_METRIC = "pairwise"
# DIVERSITY_METRIC = "structural"

CULTURES = ['BRA', 'CHN', 'MEX', 'NGA', 'NZL']

MODELS = [
    ("gpt-5.4", "GPT"),
    ("gpt-5-mini", "GPT"),
    ("gpt-4o-mini", "GPT"),

    ("claude-opus-4.7", "Claude"),
    ("claude-sonnet-4.5", "Claude"),
    ("claude-3.5-haiku", "Claude"),

    ("gemini-3.1-flash-lite-preview", "Gemini"),
    ("gemini-3-flash-preview", "Gemini"),
    ("gemini-2.5-pro", "Gemini"),

    ("grok-4.3", "Grok"),
    ("grok-4", "Grok"),
    ("grok-3", "Grok"),

    ("Qwen3.5-27B", "Qwen"),
    ("Qwen3-32B", "Qwen"),
    ("Qwen2.5-32B-Instruct", "Qwen"),

    ("llama-4-scout", "Llama"),
    ("llama-3.3-70b-instruct", "Llama"),
    ("llama-3.1-70b-instruct", "Llama"),
]

# =========================================================
# EXACT SAME DISPLAY NAMES AS ORIGINAL FIGURE
# =========================================================

DISPLAY_NAMES = {
    "claude-opus-4.7": "opus-4.7",
    "claude-sonnet-4.5": "sonnet-4.5",
    "claude-3.5-haiku": "haiku-3.5",

    "gemini-3.1-flash-lite-preview": "3.1-flash-lite-preview",
    "gemini-3-flash-preview": "3-flash-preview",
    "gemini-2.5-pro": "2.5-pro",
}

COLORS = {
    "GPT": "#1D9E75",
    "Claude": "#D85A30",
    "Gemini": "#378ADD",
    "Grok": "#7F77DD",
    "Qwen": "#B83A8C",
    "Llama": "#8B6F47",
}

# =========================================================
# Helpers
# =========================================================

def get_majority_answer(dist):
    return int(max(dist.items(), key=lambda x: x[1])[0])


def get_option_range(reference_item):
    num_options = len(reference_item["option"])
    return 1, num_options


# =========================================================
# Alignment
# =========================================================

def alignment(output_path, culture):

    with open("data/wvs.json", "r", encoding="utf-8") as f:
        references = json.load(f)

    with open("data/proportions_group_by_country.json", "r", encoding="utf-8") as f:
        culture_data = json.load(f)

    with open(output_path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f]

    data.sort(key=lambda x: x["idx"])

    R = {}
    q_ranges = {}

    for id_, item in enumerate(data):

        reference = references[id_]

        q_id = reference["Q_id"]

        resp = item["response"]

        try:
            ans = normalize_latex_text(
                extract_boxed_answer(resp).lower()
            )
        except:
            continue

        if not ans:
            continue

        try:
            R[q_id] = int(ans)
        except:
            continue

        q_ranges[q_id] = get_option_range(reference)

    dist_all = culture_data[culture]

    squared_sum = 0.0
    max_squared_sum = 0.0

    for q_id, r_i in R.items():

        if q_id not in dist_all:
            continue

        if not dist_all[q_id]:
            continue

        if q_id not in q_ranges:
            continue

        a_i = get_majority_answer(dist_all[q_id])

        min_opt, max_opt = q_ranges[q_id]

        squared_sum += (a_i - r_i) ** 2
        max_squared_sum += (max_opt - min_opt) ** 2

    if max_squared_sum == 0:
        return None

    score = (
        1
        - math.sqrt(squared_sum) / math.sqrt(max_squared_sum)
    ) * 100

    return score


# =========================================================
# Pairwise Diversity
# =========================================================

def compute_pairwise(all_answers, ref_dict):

    agents = sorted(all_answers.keys())

    pair_dict = {}

    pairwise_scores = []

    for a1, a2 in combinations(agents, 2):

        vec1 = all_answers[a1]
        vec2 = all_answers[a2]

        common_qs = set(vec1.keys()) & set(vec2.keys())

        squared_sum = 0.0
        max_squared_sum = 0.0

        for q in common_qs:

            if q not in ref_dict:
                continue

            num_options = len(ref_dict[q]["option"])

            delta = num_options - 1

            if delta == 0:
                continue

            squared_sum += (vec1[q] - vec2[q]) ** 2
            max_squared_sum += delta ** 2

        if max_squared_sum == 0:
            continue

        dist = (
            math.sqrt(squared_sum)
            / math.sqrt(max_squared_sum)
        )

        pairwise_scores.append(dist)

        pair_dict[(a1, a2)] = dist

    if not pairwise_scores:
        return 0.0, {}, agents

    mean_dist = sum(pairwise_scores) / len(pairwise_scores)

    return mean_dist, pair_dict, agents


# =========================================================
# Structural Diversity (MST Span)
# =========================================================

def compute_mst_span(pair_dict, agents):

    n = len(agents)

    if n < 2 or not pair_dict:
        return 0.0

    M = np.zeros((n, n))

    for i in range(n):

        for j in range(i + 1, n):

            a, b = agents[i], agents[j]

            d = pair_dict.get(
                (a, b),
                pair_dict.get((b, a), 0.0)
            )

            M[i, j] = d
            M[j, i] = d

    mst = minimum_spanning_tree(M).toarray()

    mst_span = float(mst.sum()) / (n - 1)

    return mst_span


# =========================================================
# Diversity
# =========================================================

def diversity(output_dir, culture_lst, model):

    with open("data/wvs.json", "r", encoding="utf-8") as f:
        references = json.load(f)

    all_answers = defaultdict(dict)

    ref_dict = {
        r["Q_id"]: r
        for r in references
    }

    for cul_id, culture in enumerate(culture_lst):

        output_path = (
            f"{output_dir}/{model}_{culture}_system.jsonl"
        )

        with open(output_path, 'r') as f:
            data = [json.loads(l) for l in f]

        data.sort(key=lambda x: x["idx"])

        for id_, item in enumerate(data):

            reference = references[id_]

            resp = item["response"]

            try:
                ans = normalize_latex_text(
                    extract_boxed_answer(resp).lower()
                )
            except:
                continue

            if not ans:
                continue

            q_id = reference["Q_id"]

            try:
                ans_int = int(ans)
            except:
                continue

            all_answers[
                f"{culture}_{cul_id}"
            ][q_id] = ans_int

    mean_div, pair_dict, agents = compute_pairwise(
        all_answers,
        ref_dict
    )

    mst_span = compute_mst_span(
        pair_dict,
        agents
    )

    if DIVERSITY_METRIC == "pairwise":
        return mean_div * 100

    elif DIVERSITY_METRIC == "structural":
        return mst_span * 100

    else:
        raise ValueError(DIVERSITY_METRIC)


# =========================================================
# Build Data Automatically
# =========================================================

data = []

for model, family in MODELS:

    print("\n==============================")
    print(model)
    print("==============================")

    # -----------------------------------------
    # Alignment
    # -----------------------------------------

    scores = []

    for culture in CULTURES:

        output_path = (
            f"{OUTPUT_DIR}/{model}_{culture}_system.jsonl"
        )

        score = alignment(output_path, culture)

        if score is not None:
            scores.append(score)

    avg_alignment = sum(scores) / len(scores)

    # -----------------------------------------
    # Diversity
    # -----------------------------------------

    div_score = diversity(
        OUTPUT_DIR,
        CULTURES,
        model
    )

    display_name = DISPLAY_NAMES.get(
        model,
        model
    )

    print(f"Diversity: {div_score:.2f}")
    print(f"Alignment: {avg_alignment:.3f}")

    data.append(
        (
            display_name,
            div_score,
            avg_alignment,
            family
        )
    )

# =========================================================
# EXACT SAME PLOTTING STYLE AS ORIGINAL
# =========================================================

xs = np.array([d[1] for d in data])
ys = np.array([d[2] for d in data])
fams = [d[3] for d in data]

mean_x, mean_y = xs.mean(), ys.mean()

fig, ax = plt.subplots(figsize=(3.3, 3.0))

# =========================================================
# Scatter
# =========================================================

for fam in ["GPT", "Claude", "Gemini", "Grok", "Qwen", "Llama"]:

    idx = [i for i, f in enumerate(fams) if f == fam]

    ax.scatter(
        xs[idx],
        ys[idx],
        c=COLORS[fam],
        s=24,
        label=fam,
        edgecolors="white",
        linewidths=0.6,
        zorder=3
    )

# =========================================================
# Offsets
# =========================================================
if DIVERSITY_METRIC == "pairwise":
    offsets = {
        "gpt-5.4":                  (4, 2, "left"),
        "gpt-5-mini":               (4, -4, "left"),
        "gpt-4o-mini":              (4, -8, "left"),

        "opus-4.7":                 (4, 2, "left"),
        "sonnet-4.5":               (-4, 2, "right"),
        "haiku-3.5":                (4, 2, "left"),

        "3.1-flash-lite-preview":   (8, -8, "right"),
        "3-flash-preview":          (0, -8, "left"),
        "2.5-pro":                  (16, 4, "right"),

        "grok-4.3":                 (-4, -8, "right"),
        "grok-4":                   (4, 2, "left"),
        "grok-3":                   (4, 2, "left"),

        "Qwen3.5-27B":              (4, 2, "left"),
        "Qwen3-32B":                (4, 2, "left"),
        "Qwen2.5-32B-Instruct":     (4, 2, "left"),

        "llama-4-scout":            (4, 2, "left"),
        "llama-3.3-70b-instruct":   (4, 2, "right"),
        "llama-3.1-70b-instruct":   (-32, -6, "center"),
    }
else:
    offsets = {
    "gpt-5.4":                  (4, 2, "left"),
    "gpt-5-mini":               (4, -4, "left"),
    "gpt-4o-mini":              (4, -8, "left"),

    "opus-4.7":                 (4, 2, "left"),
    "sonnet-4.5":               (-4, 2, "right"),
    "haiku-3.5":                (4, 2, "left"),

    "3.1-flash-lite-preview":   (8, -8, "right"),
    "3-flash-preview":          (0, -8, "left"),
    "2.5-pro":                  (16, 4, "right"),

    "grok-4.3":                 (-4, 0, "right"),
    "grok-4":                   (4, 2, "left"),
    "grok-3":                   (4, 2, "left"),

    "Qwen3.5-27B":              (4, -2, "left"),
    "Qwen3-32B":                (4, 2, "left"),
    "Qwen2.5-32B-Instruct":     (4, 2, "left"),

    "llama-4-scout":            (4, 2, "left"),
    "llama-3.3-70b-instruct":   (4, 2, "right"),
    "llama-3.1-70b-instruct":   (-32, -6, "center"),
    }
for name, x, y, fam in data:

    dx, dy, ha = offsets.get(
        name,
        (4, 2, "left")
    )

    ax.annotate(
        name,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=6,
        color=COLORS[fam],
        ha=ha
    )

# =========================================================
# Mean Lines
# =========================================================

ax.axvline(
    mean_x,
    color="gray",
    linestyle="--",
    linewidth=0.6,
    alpha=0.5
)

ax.axhline(
    mean_y,
    color="gray",
    linestyle="--",
    linewidth=0.6,
    alpha=0.5
)

# =========================================================
# Axis Range
# =========================================================

if DIVERSITY_METRIC == "pairwise":

    ax.set_xlim(19.5, 37.5)

else:

    ax.set_xlim(15, 31)

ax.set_ylim(66.3, 71.5)

# =========================================================
# Correlation
# =========================================================

r, p = stats.pearsonr(xs, ys)

print(f"Pearson r = {r:.4f}")
print(f"p-value   = {p:.4f}")
print(f"mean D    = {mean_x:.4f}")
print(f"mean A    = {mean_y:.4f}")

# =========================================================
# Text
# =========================================================

ax.text(
    mean_x,
    66.38,
    f"mean: D={mean_x:.2f}, A={mean_y:.2f}",
    ha="center",
    va="bottom",
    fontsize=6,
    color="gray",
    alpha=0.8
)

ax.text(
    0.02,
    0.975,
    f"Pearson r = {r:+.2f}  (n={len(xs)})",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=6.5,
    color="gray"
)

# =========================================================
# Labels
# =========================================================

if DIVERSITY_METRIC == "pairwise":
    xlabel = r"$\mathrm{Diversity}_P$ →"
else:
    xlabel = r"$\mathrm{Diversity}_S$ →"

ax.set_xlabel(
    xlabel,
    fontsize=8,
    labelpad=2
)

ax.set_ylabel(
    "Alignment →",
    fontsize=8,
    labelpad=2
)

ax.tick_params(
    axis='both',
    labelsize=7,
    pad=2
)

# =========================================================
# Style
# =========================================================

if DIVERSITY_METRIC != "structural":
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

ax.grid(
    True,
    alpha=0.12,
    linewidth=0.5
)

# =========================================================
# Legend
# =========================================================

ax.legend(
    loc="upper right",
    frameon=False,
    fontsize=6,
    handletextpad=0.2,
    borderpad=0.2,
    labelspacing=0.25,
    bbox_to_anchor=(1.0, 1.0)
)

# =========================================================
# Save
# =========================================================

plt.subplots_adjust(
    left=0.13,
    right=0.985,
    top=0.985,
    bottom=0.13
)

suffix = (
    "pairwise"
    if DIVERSITY_METRIC == "pairwise"
    else "structural"
)

plt.savefig(
    f"fig1_left_{suffix}.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.02
)

plt.savefig(
    f"fig1_left_{suffix}.pdf",
    bbox_inches="tight",
    pad_inches=0.02
)

print("\nSaved.")