import os, json, math, time
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt

from utils import normalize_latex_text, extract_boxed_answer

# ============================================================
OUTPUT_DIR   = "5_10_evaluation"
ALL_CULTURES = ['AUS','BOL','BRA','CAN','CHN','DEU','ETH','GBR','IND','KEN',
                'MEX','NGA','NLD','NZL','RUS','THA','UKR','USA','ZWE']

MODELS = [
    ("gpt-5.4",               "---", "#7F77DD"),
    ("claude-opus-4.7", "---",      "#D85A30"),
    ("gemini-3.1-flash-lite-preview",        "---", "#378ADD"),
]
SAVE_PATH        = "5_14_rq4_robustness.png"
WVS_PATH         = "data/wvs.json"
PROPORTIONS_PATH = "data/proportions_group_by_country.json"


# ============================================================
def load_refs():
    with open(WVS_PATH) as f:
        refs = json.load(f)
    return refs, {r["Q_id"]: r for r in refs}


def load_answers(model, culture, refs):
    path = f"{OUTPUT_DIR}/{model}_{culture}_system.jsonl"
    with open(path) as f:
        data = sorted([json.loads(l) for l in f], key=lambda x: x["idx"])
    ans = {}
    for id_, item in enumerate(data):
        assert refs[id_]["Q_id"] == item["q_id"]
        s = normalize_latex_text(extract_boxed_answer(item["response"]).lower())
        if not s:
            continue
        try:
            ans[refs[id_]["Q_id"]] = int(s)
        except ValueError:
            continue
    return ans


def pair_distance(a1, a2, ref_dict):
    sq, m = 0.0, 0.0
    for q in a1.keys() & a2.keys():
        if q not in ref_dict:
            continue
        delta = len(ref_dict[q]["option"]) - 1
        if delta == 0:
            continue
        sq += (a1[q] - a2[q]) ** 2
        m  += delta ** 2
    return float("nan") if m == 0 else math.sqrt(sq) / math.sqrt(m)


def system_diversity(combo, pair_d):
    pairs = list(combinations(combo, 2))
    return sum(pair_d[(c1, c2)] for c1, c2 in pairs) / len(pairs) * 100


def _pair_d_from_answers(answers, ref_dict):
    pair_d = {}
    for c1, c2 in combinations(ALL_CULTURES, 2):
        d = pair_distance(answers[c1], answers[c2], ref_dict)
        pair_d[(c1, c2)] = d
        pair_d[(c2, c1)] = d
    return pair_d


def compute_pair_d(model, refs, ref_dict):
    answers = {c: load_answers(model, c, refs) for c in ALL_CULTURES}
    return _pair_d_from_answers(answers, ref_dict)


def compute_human_pair_d(refs, ref_dict):
    """Build per-culture majority-vote vectors from population proportions,
    then compute pairwise distances — same formula as the LLM side."""
    with open(PROPORTIONS_PATH) as f:
        data = json.load(f)
    q_ids = {r["Q_id"] for r in refs}

    answers = {}
    for culture in ALL_CULTURES:
        if culture not in data:
            print(f"[warn] no human data for {culture}")
            answers[culture] = {}
            continue
        vec = {}
        for q_id, option_dist in data[culture].items():
            if q_id not in q_ids or not option_dist:
                continue
            majority = max(option_dist.items(), key=lambda x: x[1])[0]
            try:
                vec[q_id] = int(majority)
            except ValueError:
                continue
        answers[culture] = vec
    return _pair_d_from_answers(answers, ref_dict)


def sweep_k(pair_d, k_min=2, k_max=19):
    rows = []
    for k in range(k_min, k_max + 1):
        combos = list(combinations(ALL_CULTURES, k))
        divs = np.array([system_diversity(c, pair_d) for c in combos])
        rows.append((k, divs.max()))
    return rows


def plot(results, human_k_rows, save_path):
    """Compact single-column (3.3in) version, stacked vertically."""
    fig, axes = plt.subplots(2, 1, figsize=(3.3, 3.7))

    short = {
        "claude-opus-4.7":               "claude-opus-4.7",
        "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite-preview",
        "gemini-3-flash-preview":        "gemini-3-flash-preview",
    }

    # ---------- (a) Effect of culture selection ----------
    ax = axes[0]
    for model, tag, color, divs, _ in results:
        sorted_d = np.sort(divs)
        x = np.arange(1, len(sorted_d) + 1)
        ax.plot(x, sorted_d, color=color, linewidth=1.2,
                label=f"{short.get(model, model)} (max={sorted_d[-1]:.1f})")
        ax.scatter([len(sorted_d)], [sorted_d[-1]], color=color, s=12, zorder=5)
    ax.set_xlabel("Culture combinations (sorted)", fontsize=7, labelpad=2)
    ax.set_ylabel("System Diversity", fontsize=7, labelpad=2)
    ax.set_title("(a) Effect of culture selection  ($N{=}5$)", fontsize=7.5, pad=1)
    ax.set_xticks([0, 4000, 8000, 12000])
    ax.set_xticklabels(["0", "4k", "8k", "12k"])
    ax.tick_params(axis='both', labelsize=6, pad=2)
    ax.legend(frameon=False, loc="upper right", fontsize=6,
              handletextpad=0.3, borderpad=0.2, labelspacing=0.25, bbox_to_anchor=(0.96, 0.35))
    ax.grid(True, alpha=0.15, linewidth=0.4)

    # ---------- (b) Gap to human ----------
    ax = axes[1]
    human_dict = dict(human_k_rows)
    for model, tag, color, _, k_rows in results:
        ks    = [r[0] for r in k_rows]
        delta = [r[1] - human_dict[r[0]] for r in k_rows]
        ax.plot(ks, delta, "o-", color=color, linewidth=1.2, markersize=2.8,
                label=short.get(model, model))
    ax.axhline(0, color="black", linestyle="--", linewidth=0.7, alpha=0.55)
    ax.text(2.2, -0.4, "Human (= 0)", fontsize=6, color="black",
            alpha=0.75, va="top")
    ax.set_xlabel("Number of agents ($k$)", fontsize=7, labelpad=2)
    ax.set_ylabel(r"Gap to Human Diversity", fontsize=7, labelpad=2)
    ax.set_title("(b) Effect of agent count", fontsize=7.5, pad=1)
    ax.set_xticks([2, 4, 6, 8, 10, 12, 14, 16, 18])
    ax.set_ylim(top=0.8)
    ax.tick_params(axis='both', labelsize=6, pad=2)
    ax.legend(frameon=False, loc="upper right", fontsize=6,
              handletextpad=0.3, borderpad=0.2, labelspacing=0.25,
              bbox_to_anchor=(0.96, 0.75))
    ax.grid(True, alpha=0.15, linewidth=0.4)

    plt.subplots_adjust(left=0.155, right=0.97, top=0.945, bottom=0.10, hspace=0.36)
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path.replace(".png", ".pdf"))
    print(f"\nSaved -> {save_path}")


# ============================================================
if __name__ == "__main__":
    t0 = time.time()
    refs, ref_dict = load_refs()

    print("--- Human (population majority votes) ---")
    human_pair_d = compute_human_pair_d(refs, ref_dict)
    human_k_rows = sweep_k(human_pair_d)
    print(f"  max Diversity at k=2: {human_k_rows[0][1]:.2f}, "
          f"k=5: {dict(human_k_rows)[5]:.2f}, "
          f"k=19: {human_k_rows[-1][1]:.2f}")

    results = []
    for model, tag, color in MODELS:
        print(f"\n--- {model} ---")
        pair_d = compute_pair_d(model, refs, ref_dict)

        divs_k5 = np.array([system_diversity(c, pair_d)
                            for c in combinations(ALL_CULTURES, 5)])
        print(f"  (a) C(19,5) Diversity range = {divs_k5.min():.2f}–{divs_k5.max():.2f}")

        k_rows = sweep_k(pair_d)
        print(f"  (b) max Diversity at k=2: {k_rows[0][1]:.2f},  "
              f"k=19: {k_rows[-1][1]:.2f}")

        results.append((model, tag, color, divs_k5, k_rows))

    plot(results, human_k_rows, SAVE_PATH)
    print(f"\nTotal: {time.time() - t0:.1f}s")