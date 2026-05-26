"""
plot_multiround_trajectory.py

Run alignment & diversity evaluation across multiple rounds, then
produce two single-column PDFs:
  - Fig/fig_multiround_diversity.pdf  (main paper)
  - Fig/fig_multiround_alignment.pdf  (appendix)
"""

import os
import json
import math
from collections import defaultdict
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import minimum_spanning_tree

from utils import normalize_latex_text, extract_boxed_answer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================================
# Configuration
# ==========================================================
CULTURES = ["BRA", "CHN", "MEX", "NGA", "NZL"]

# Representative backbones (one per family) + display color
MODELS = [
    ("gpt-5.4",                       "GPT",    "#1D9E75"),
    ("claude-opus-4.7",               "Claude", "#D85A30"),
    ("gemini-3.1-flash-lite-preview", "Gemini", "#378ADD"),
    ("grok-4.3",                      "Grok",   "#7F77DD"),
    ("Qwen3.5-27B",                   "Qwen",   "#B83A8C"),
    ("llama-4-scout",                 "Llama",  "#8B6F47"),
]

MAX_ROUND = 5

OUT_DIVERSITY = "fig5_diversity.pdf"
OUT_ALIGNMENT = "fig5_alignment.pdf"

# ==========================================================
# Load shared data
# ==========================================================
with open("data/wvs.json", "r", encoding="utf-8") as f:
    REFERENCES = json.load(f)
REF_DICT = {r["Q_id"]: r for r in REFERENCES}

with open("data/proportions_group_by_country.json", "r", encoding="utf-8") as f:
    CULTURE_DATA = json.load(f)


# ==========================================================
# Helpers
# ==========================================================
def round_dir(r):
    """Directory holding round-r outputs."""
    if r == 0:
        return "wvs_evaluation"
    elif r == 1:
        return "wvs_evaluation_interaction"
    else:
        return f"wvs_evaluation_interaction_round{r}"


def get_majority_answer(dist):
    return int(max(dist.items(), key=lambda x: x[1])[0])


def get_option_range(reference_item):
    num_options = len(reference_item["option"])
    return 1, num_options


def compute_alignment(output_path, culture):
    if not os.path.exists(output_path):
        return None
    with open(output_path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f]
    data.sort(key=lambda x: x["idx"])

    R = {}
    q_ranges = {}
    if len(data) != len(REFERENCES):
        print(f"  [warn] {output_path}: {len(data)} rows vs {len(REFERENCES)} refs")
    for id_, item in enumerate(data):
        reference = REFERENCES[id_]
        q_id = reference["Q_id"]
        if q_id != item["q_id"]:
            continue
        try:
            ans = normalize_latex_text(extract_boxed_answer(item["response"]).lower())
            R[q_id] = int(ans)
            q_ranges[q_id] = get_option_range(reference)
        except Exception:
            continue

    dist_all = CULTURE_DATA[culture]
    squared_sum = 0.0
    max_squared_sum = 0.0
    for q_id, r_i in R.items():
        if q_id not in dist_all or not dist_all[q_id] or q_id not in q_ranges:
            continue
        a_i = get_majority_answer(dist_all[q_id])
        min_opt, max_opt = q_ranges[q_id]
        squared_sum += (a_i - r_i) ** 2
        max_squared_sum += (max_opt - min_opt) ** 2
    if max_squared_sum == 0:
        return None
    return (1 - math.sqrt(squared_sum) / math.sqrt(max_squared_sum)) * 100


def _compute_pairwise(all_answers):
    agents = sorted(all_answers.keys())
    pair_dict = {}
    scores = []
    for a1, a2 in combinations(agents, 2):
        vec1 = all_answers[a1]
        vec2 = all_answers[a2]
        common = set(vec1.keys()) & set(vec2.keys())
        sq, max_sq = 0.0, 0.0
        for q in common:
            if q not in REF_DICT:
                continue
            delta = len(REF_DICT[q]["option"]) - 1
            if delta == 0:
                continue
            sq     += (vec1[q] - vec2[q]) ** 2
            max_sq += delta ** 2
        if max_sq == 0:
            continue
        d = math.sqrt(sq) / math.sqrt(max_sq)
        scores.append(d)
        pair_dict[(a1, a2)] = d
    if not scores:
        return 0.0, {}, agents
    return sum(scores) / len(scores), pair_dict, agents


def _compute_mst_span(pair_dict, agents):
    n = len(agents)
    if n < 2 or not pair_dict:
        return 0.0
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = agents[i], agents[j]
            d = pair_dict.get((a, b), pair_dict.get((b, a), 0.0))
            M[i, j] = d
            M[j, i] = d
    mst = minimum_spanning_tree(M).toarray()
    return float(mst.sum()) / (n - 1)


def compute_diversity(output_dir, model, cultures, ground_truth=False):
    all_answers = defaultdict(dict)
    if not ground_truth:
        for cul_id, culture in enumerate(cultures):
            output_path = f"{output_dir}/{model}_{culture}_system.jsonl"
            if not os.path.exists(output_path):
                return None, None
            with open(output_path, "r", encoding="utf-8") as f:
                data = [json.loads(l) for l in f]
            data.sort(key=lambda x: x["idx"])
            if len(data) != len(REFERENCES):
                print(f"  [warn] {output_path}: length {len(data)} vs {len(REFERENCES)}")
            for id_, item in enumerate(data):
                reference = REFERENCES[id_]
                if reference["Q_id"] != item["q_id"]:
                    continue
                try:
                    ans = normalize_latex_text(extract_boxed_answer(item["response"]).lower())
                    ans_int = int(ans)
                except Exception:
                    continue
                all_answers[f"{culture}_{cul_id}"][reference["Q_id"]] = ans_int
    else:
        for cul_id, culture in enumerate(cultures):
            if culture not in CULTURE_DATA:
                continue
            for q_id, option_dist in CULTURE_DATA[culture].items():
                if q_id not in REF_DICT or not option_dist:
                    continue
                try:
                    all_answers[f"{culture}_{cul_id}"][q_id] = int(
                        max(option_dist.items(), key=lambda x: x[1])[0]
                    )
                except Exception:
                    continue

    mean_div, pair_dict, agents = _compute_pairwise(all_answers)
    mst_span = _compute_mst_span(pair_dict, agents)
    return mean_div * 100, mst_span * 100


def compute_system_AD(model, output_dir, cultures):
    align_scores = []
    for cul in cultures:
        path = f"{output_dir}/{model}_{cul}_system.jsonl"
        s = compute_alignment(path, cul)
        if s is not None:
            align_scores.append(s)
    if not align_scores:
        return None, None, None
    align_avg = sum(align_scores) / len(align_scores)
    div_mean, div_mst = compute_diversity(output_dir, model, cultures, ground_truth=False)
    return align_avg, div_mean, div_mst


# ==========================================================
# Evaluate all (model, round) pairs
# ==========================================================
print(f"Evaluating {len(MODELS)} models across rounds 0..{MAX_ROUND}")
print("=" * 60)

trajectories = {}
for model_name, family, color in MODELS:
    print(f"\n[{model_name}]")
    A_traj, D_traj, MST_traj = [], [], []
    for r in range(0, MAX_ROUND + 1):
        out_dir = round_dir(r)
        A, D, MST = compute_system_AD(model_name, out_dir, CULTURES)
        if A is None:
            print(f"  round {r}: SKIP (missing files in {out_dir}/)")
            A_traj.append(np.nan)
            D_traj.append(np.nan)
            MST_traj.append(np.nan)
        else:
            print(f"  round {r}: A={A:.2f}, D={D:.2f}, MST={MST:.2f}")
            A_traj.append(A)
            D_traj.append(D)
            MST_traj.append(MST)
    trajectories[model_name] = {
        "A": A_traj, "D": D_traj, "MST": MST_traj,
        "family": family, "color": color,
    }


# ==========================================================
# Shared style
# ==========================================================
os.makedirs("Fig", exist_ok=True)
plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "legend.fontsize": 5.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
})

rounds = np.arange(0, MAX_ROUND + 1)


# ==========================================================
# Figure 1: Diversity (main paper, single column)
# ==========================================================
fig, ax_D = plt.subplots(figsize=(3.3, 2.4))

for model_name, info in trajectories.items():
    ax_D.plot(rounds, info["D"],
              marker="o", markersize=3.2, linewidth=1.1,
              color=info["color"], label=model_name)

ax_D.set_xlabel("Interaction Round", labelpad=2)
ax_D.set_ylabel(r"System Diversity", labelpad=2)
ax_D.set_xticks(rounds)
ax_D.spines["top"].set_visible(False)
ax_D.spines["right"].set_visible(False)
ax_D.grid(alpha=0.2, linewidth=0.4)
ax_D.set_axisbelow(True)
ax_D.tick_params(axis='both', pad=2)

# Legend below the plot
handles, labels = ax_D.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 0.06),
           handletextpad=0.3, columnspacing=0.8)

plt.subplots_adjust(left=0.15, right=0.97, top=0.96, bottom=0.32)
plt.savefig(OUT_DIVERSITY, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"\nSaved {OUT_DIVERSITY}")


# ==========================================================
# Figure 2: Alignment (appendix, single column)
# ==========================================================
fig, ax_A = plt.subplots(figsize=(3.3, 2.4))

for model_name, info in trajectories.items():
    ax_A.plot(rounds, info["A"],
              marker="o", markersize=3.2, linewidth=1.1,
              color=info["color"], label=model_name)

ax_A.set_xlabel("Interaction round", labelpad=2)
ax_A.set_ylabel(r"System alignment", labelpad=2)
ax_A.set_xticks(rounds)
ax_A.spines["top"].set_visible(False)
ax_A.spines["right"].set_visible(False)
ax_A.grid(alpha=0.2, linewidth=0.4)
ax_A.set_axisbelow(True)
ax_A.tick_params(axis='both', pad=2)

handles, labels = ax_A.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.02),
           handletextpad=0.3, columnspacing=0.8)

plt.subplots_adjust(left=0.15, right=0.97, top=0.96, bottom=0.32)
plt.savefig(OUT_ALIGNMENT, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"Saved {OUT_ALIGNMENT}")