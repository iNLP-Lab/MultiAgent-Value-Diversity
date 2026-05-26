"""
Section 7 — WVS-grounded PB Case Study
====================================================

Visualize LOW vs HIGH value-diversity systems
under the WVS-grounded participatory budgeting task.

This version:
- automatically searches LOW/HIGH systems from WVS
- aggregates ALL runs automatically
- computes entropy + coverage
- normalizes approval frequencies
- produces publication-ready figures
"""

import json
import os
import re
import math
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

# ============================================================
# Config
# ============================================================

MODEL = "gpt-5.4"
# MODEL = "claude-opus-4.7"
# MODEL = "gemini-3.1-flash-lite-preview"
PROJECT_JSON = "data/wvs_project.json"

PB_DIR = "wvs_pb_evaluation"

WVS_DIR = "wvs_evaluation"
WVS_REF_JSON = "data/wvs.json"

ALL_CULTURES = [
    'AUS','BOL','BRA','CAN','CHN','DEU','ETH','GBR','IND',
    'KEN','MEX','NGA','NLD','NZL','RUS','THA','UKR','USA','ZWE'
]

N_AGENTS = 5

OUT_PDF = f"sec7_{MODEL}.pdf"
OUT_PNG = f"sec7_{MODEL}.png"

# ============================================================
# Global Style
# ============================================================

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

# ============================================================
# Category Colors
# ============================================================

CATEGORY_COLORS = {
    "Culture": "#C46B9F",
    "Security": "#D95F5F",
    "Transparency": "#8E6BBE",
    "Economy": "#D4A017",
    "Technology": "#4A90C4",
    "Civic Participation": "#6C9A8B",
    "Institutions": "#7B68EE",
    "Environment": "#4C9F70",
    "Community": "#7E57C2",
    "Education": "#F39C12",
    "Social Inclusion": "#E67E22",
    "Migration": "#B56576",
    "Health": "#2A9D8F",
}
# ============================================================
# Load Projects
# ============================================================

def load_projects():

    with open(PROJECT_JSON, "r", encoding="utf-8") as f:
        projects = json.load(f)

    out = {}

    for p in projects:

        out[p["id"]] = {
            "dimension": p["dimension"],
            "name": p["name"],
            "category": p["category"],
            "cost": p["cost"],
            "description": p["description"],
        }

    return out

# ============================================================
# Load PB Votes
# ============================================================

def load_llm_culture_votes(culture):

    path = f"{PB_DIR}/{MODEL}_{culture}.jsonl"

    if not os.path.exists(path):
        print(f"[missing] {path}")
        return []

    votes = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            try:
                r = json.loads(line)
            except:
                continue

            parsed = None

            if "parsed_projects" in r:
                parsed = r["parsed_projects"]

            elif "parsed_4_projects" in r:
                parsed = r["parsed_4_projects"]

            elif "parsed_5_projects" in r:
                parsed = r["parsed_5_projects"]

            if r.get("parse_ok") and parsed:
                votes.append(parsed)

    print(f"{culture}: {len(votes)} runs loaded")

    return votes

# ============================================================
# Aggregate Votes
# ============================================================

def system_vote_counts(cultures, n_projects):

    counts = np.zeros(n_projects, dtype=int)

    for c in cultures:

        votes = load_llm_culture_votes(c)

        for v in votes:

            for pid in v:

                if 1 <= pid <= n_projects:
                    counts[pid - 1] += 1

    return counts

# ============================================================
# Metrics
# ============================================================

def compute_entropy(counts):

    total = counts.sum()

    if total == 0:
        return 0.0

    p = counts / total
    p = p[p > 0]

    return float(-(p * np.log(p)).sum())

def compute_coverage(counts):

    return int((counts > 0).sum())

# ============================================================
# Load WVS Answers
# ============================================================

BOXED = re.compile(r"\\boxed\{([^}]+)\}")

def extract_boxed_answer(text):

    m = BOXED.search(text or "")

    if m:
        return m.group(1)

    return ""

def normalize_latex_text(s):

    return re.sub(r"[^\d]", "", s or "")

def load_wvs_answers():

    out = {}

    for c in ALL_CULTURES:

        path = f"{WVS_DIR}/{MODEL}_{c}_system.jsonl"

        if not os.path.exists(path):
            continue

        with open(path, "r") as f:
            data = [json.loads(l) for l in f]

        data.sort(key=lambda x: x.get("idx", 0))

        ans = {}

        for item in data:

            try:

                a = normalize_latex_text(
                    extract_boxed_answer(
                        item["response"]
                    ).lower()
                )

                ans[item["q_id"]] = int(a)

            except:
                continue

        out[c] = ans

    return out

# ============================================================
# Pairwise Diversity
# ============================================================

def pairwise_diversity(culture_lst, wvs_ans, ref_dict):

    if any(c not in wvs_ans for c in culture_lst):
        return None

    agents = {
        f"{c}_{i}": wvs_ans[c]
        for i, c in enumerate(culture_lst)
    }

    keys = list(agents.keys())

    scores = []

    for a, b in combinations(keys, 2):

        v1, v2 = agents[a], agents[b]

        common = set(v1) & set(v2)

        ss = 0.0
        ms = 0.0

        for q in common:

            if q not in ref_dict:
                continue

            delta = len(ref_dict[q]["option"]) - 1

            if delta == 0:
                continue

            ss += (v1[q] - v2[q]) ** 2
            ms += delta ** 2

        if ms > 0:

            scores.append(
                math.sqrt(ss) / math.sqrt(ms)
            )

    if not scores:
        return None

    return sum(scores) / len(scores) * 100

# ============================================================
# Search LOW/HIGH Systems
# ============================================================

def search_low_high_systems(wvs_ans, ref_dict):

    results = []

    all_cfgs = list(
        combinations(ALL_CULTURES, N_AGENTS)
    )

    print(f"\nSearching {len(all_cfgs)} systems...")

    for cfg in all_cfgs:

        d = pairwise_diversity(
            list(cfg),
            wvs_ans,
            ref_dict
        )

        if d is not None:
            results.append((cfg, d))

    results.sort(key=lambda x: x[1])

    low_cfg, low_d = results[0]
    high_cfg, high_d = results[-1]

    return {
        "LOW": {
            "cultures": list(low_cfg),
            "D": low_d
        },
        "HIGH": {
            "cultures": list(high_cfg),
            "D": high_d
        }
    }

# ============================================================
# Plot Row
# ============================================================

def draw_row(
    ax,
    counts,
    pids,
    projects,
    title,
    max_y,
):

    total_votes = counts.sum()

    if total_votes == 0:
        heights = [0.0 for _ in pids]
    else:
        heights = [
            counts[pid - 1] / total_votes
            for pid in pids
        ]

    x = np.arange(len(pids))

    colors = [
        CATEGORY_COLORS[projects[pid]["category"]]
        for pid in pids
    ]

    ax.bar(
        x,
        heights,
        color=colors,
        width=0.80,
        edgecolor="none"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [f"P{pid:02d}" for pid in pids],
        fontsize=12
    )

    ax.set_ylim(0, max_y * 1.08)

    ax.set_title(
        title,
        fontsize=18,
        loc="left",
        pad=4
    )

    ax.yaxis.set_major_locator(
        MaxNLocator(4)
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis='y',
        labelsize=16
    )

# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Projects
    # --------------------------------------------------------

    projects = load_projects()

    pids = sorted(projects.keys())

    # --------------------------------------------------------
    # Load WVS
    # --------------------------------------------------------

    print("\nLoading WVS answers...")

    wvs_ans = load_wvs_answers()

    with open(WVS_REF_JSON, "r") as f:
        references = json.load(f)

    ref_dict = {
        r["Q_id"]: r
        for r in references
    }

    # --------------------------------------------------------
    # Search LOW/HIGH systems
    # --------------------------------------------------------

    systems = search_low_high_systems(
        wvs_ans,
        ref_dict
    )

    print("\n============================")
    print("Selected Systems")
    print("============================")

    for tag in ["LOW", "HIGH"]:

        print(
            f"{tag}: "
            f"D={systems[tag]['D']:.2f} | "
            f"{systems[tag]['cultures']}"
        )

    # --------------------------------------------------------
    # Aggregate vote counts
    # --------------------------------------------------------

    low_counts = system_vote_counts(
        systems["LOW"]["cultures"],
        len(projects)
    )

    high_counts = system_vote_counts(
        systems["HIGH"]["cultures"],
        len(projects)
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    low_entropy = compute_entropy(low_counts)
    high_entropy = compute_entropy(high_counts)

    low_coverage = compute_coverage(low_counts)
    high_coverage = compute_coverage(high_counts)

    print("\n============================")
    print("System Statistics")
    print("============================")

    print(
        f"LOW  | "
        f"coverage={low_coverage} | "
        f"entropy={low_entropy:.3f} | "
        f"total_votes={low_counts.sum()}"
    )

    print(
        f"HIGH | "
        f"coverage={high_coverage} | "
        f"entropy={high_entropy:.3f} | "
        f"total_votes={high_counts.sum()}"
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    low_total = low_counts.sum()
    high_total = high_counts.sum()

    if low_total == 0:
        low_norm = np.zeros_like(low_counts)
    else:
        low_norm = low_counts / low_total

    if high_total == 0:
        high_norm = np.zeros_like(high_counts)
    else:
        high_norm = high_counts / high_total

    max_y = max(
        low_norm.max(),
        high_norm.max(),
        0.01
    )

    # ========================================================
    # Figure
    # ========================================================

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 5.2),
        sharex=True
    )

    fig.supylabel(
        "Vote Frequency",
        fontsize=20,
        x=0.005,
        y=0.58
    )

    panels = [
        (
            f"{MODEL} system @ LOW value diversity "
            f"(D={systems['LOW']['D']:.1f})  —  "
            f"{', '.join(systems['LOW']['cultures'])}",
            low_counts
        ),
        (
            f"{MODEL} system @ HIGH value diversity "
            f"(D={systems['HIGH']['D']:.1f})  —  "
            f"{', '.join(systems['HIGH']['cultures'])}",
            high_counts
        ),
    ]

    for ax, (title, counts) in zip(axes, panels):

        draw_row(
            ax=ax,
            counts=counts,
            pids=pids,
            projects=projects,
            title=title,
            max_y=max_y
        )

    # ========================================================
    # Legend
    # ========================================================

    used_categories = []

    for pid in pids:

        cat = projects[pid]["category"]

        if cat not in used_categories:
            used_categories.append(cat)

    legend_handles = [

        Patch(
            facecolor=CATEGORY_COLORS[c],
            edgecolor="none",
            label=c
        )

        for c in used_categories
    ]

    axes[-1].legend(
        handles=legend_handles,
        fontsize=13,
        frameon=False,
        ncol=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        columnspacing=1.6,
        handletextpad=0.5,
        labelspacing=0.7,
    )

    axes[-1].set_xlabel(
        "Project",
        fontsize=18,
        labelpad=2
    )

    # ========================================================
    # Layout
    # ========================================================

    plt.tight_layout()

    plt.subplots_adjust(
        hspace=0.20,
        bottom=0.25
    )

    # ========================================================
    # Save
    # ========================================================

    plt.savefig(
        OUT_PDF,
        bbox_inches="tight"
    )

    plt.savefig(
        OUT_PNG,
        dpi=300,
        bbox_inches="tight"
    )

    print("\nSaved:")
    print(f"  {OUT_PDF}")
    print(f"  {OUT_PNG}")

# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()