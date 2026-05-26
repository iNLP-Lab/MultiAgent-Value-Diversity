import os
import json
import math
import csv
import time
from itertools import combinations, product
from collections import defaultdict

import numpy as np

from utils import normalize_latex_text, extract_boxed_answer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================
# Config — 确认 MODELS 跟你 jsonl 文件名里的标识完全一致
# ============================================================
OUTPUT_DIR = "wvs_evaluation"
CULTURES   = ["BRA", "CHN", "MEX", "NGA", "NZL"]
MODELS = [
    "gpt-5.4",
    "gpt-5-mini",
    "gpt-4o-mini",
    "claude-opus-4.7",
    "claude-sonnet-4.5",
    "claude-3.5-haiku",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "grok-4.3",
    "grok-4",
    "grok-3",
    "Qwen3.5-27B",
    "Qwen3-32B",
    "Qwen2.5-32B-Instruct",
    "llama-4-scout",
    "llama-3.3-70b-instruct",
    "llama-3.1-70b-instruct",
]
RESULT_CSV = "backbone_combinations.csv"


# ============================================================
# Data loading
# ============================================================
def load_references():
    with open("data/wvs.json", "r", encoding="utf-8") as f:
        references = json.load(f)
    ref_dict = {r["Q_id"]: r for r in references}
    return references, ref_dict


def load_culture_data():
    with open("data/proportions_group_by_country.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_answers(model, culture, references):
    """Return {q_id: int_answer} for one (model, culture) inference file."""
    path = f"{OUTPUT_DIR}/{model}_{culture}_system.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f]
    data.sort(key=lambda x: x["idx"])
    answers = {}
    for id_, item in enumerate(data):
        reference = references[id_]
        assert reference["Q_id"] == item["q_id"]
        try:
            ans = normalize_latex_text(extract_boxed_answer(item["response"]).lower())
        except:
            continue
        if not ans:
            continue
        try:
            answers[reference["Q_id"]] = int(ans)
        except ValueError:
            continue
    return answers


# ============================================================
# Metric primitives
# ============================================================
def alignment_score(answers, culture, ref_dict, culture_data):
    """One (model, culture) vs that culture's majority human answers. [0,100] or nan."""
    dist_all = culture_data[culture]
    squared_sum = 0.0
    max_squared_sum = 0.0
    for q_id, r_i in answers.items():
        if q_id not in dist_all or not dist_all[q_id]:
            continue
        if q_id not in ref_dict:
            continue
        a_i = int(max(dist_all[q_id].items(), key=lambda x: x[1])[0])
        num_options = len(ref_dict[q_id]["option"])
        squared_sum += (a_i - r_i) ** 2
        max_squared_sum += (num_options - 1) ** 2
    if max_squared_sum == 0:
        return float("nan")
    return (1 - math.sqrt(squared_sum) / math.sqrt(max_squared_sum)) * 100


def pair_distance(ans1, ans2, ref_dict):
    """Normalized distance between two agents' answer vectors. [0,1] or nan."""
    common = ans1.keys() & ans2.keys()
    squared_sum = 0.0
    max_squared_sum = 0.0
    for q in common:
        if q not in ref_dict:
            continue
        delta = len(ref_dict[q]["option"]) - 1
        if delta == 0:
            continue
        squared_sum += (ans1[q] - ans2[q]) ** 2
        max_squared_sum += delta ** 2
    if max_squared_sum == 0:
        return float("nan")
    return math.sqrt(squared_sum) / math.sqrt(max_squared_sum)


# ============================================================
# Precompute: 75 alignment 值 + 2250 pairwise 距离
# ============================================================
def precompute(references, ref_dict, culture_data):
    n_cul, n_mod = len(CULTURES), len(MODELS)

    # answers[(culture_idx, model_idx)] = {q_id: int}
    answers = {}
    for ci, culture in enumerate(CULTURES):
        for mi, model in enumerate(MODELS):
            answers[(ci, mi)] = load_model_answers(model, culture, references)

    # align_arr[culture_idx, model_idx]
    align_arr = np.full((n_cul, n_mod), np.nan)
    for ci, culture in enumerate(CULTURES):
        for mi in range(n_mod):
            align_arr[ci, mi] = alignment_score(
                answers[(ci, mi)], culture, ref_dict, culture_data
            )

    # pair_arr[ci, cj, mi, mj] = distance  (只填 ci<cj)
    pair_arr = np.full((n_cul, n_cul, n_mod, n_mod), np.nan)
    for ci, cj in combinations(range(n_cul), 2):
        for mi in range(n_mod):
            for mj in range(n_mod):
                pair_arr[ci, cj, mi, mj] = pair_distance(
                    answers[(ci, mi)], answers[(cj, mj)], ref_dict
                )

    if np.isnan(align_arr).any():
        print("[warn] some (culture, model) alignment is nan — 检查对应 jsonl 是否完整")
    
    print("\nAnswer counts per (model, culture):")
    for ci, culture in enumerate(CULTURES):
        for mi, model in enumerate(MODELS):
            n = len(answers[(ci, mi)])
            if n < 200:  # 正常应该接近 223
                print(f"  ! {model:32s} × {culture}: {n}")
    return align_arr, pair_arr


# ============================================================
# 向量化枚举所有 15^5 组合
# ============================================================
def enumerate_combinations(align_arr, pair_arr):
    n_cul, n_mod = align_arr.shape
    combos = np.array(list(product(range(n_mod), repeat=n_cul)))  # (15^5, 5)

    # Diversity = 10 个 culture-pair 距离的平均
    div = np.zeros(len(combos))
    n_pairs = 0
    for ci, cj in combinations(range(n_cul), 2):
        div += pair_arr[ci, cj, combos[:, ci], combos[:, cj]]
        n_pairs += 1
    div = div / n_pairs * 100

    # Alignment = 5 个 (culture, 选中模型) 的 alignment 平均
    align_vals = align_arr[np.arange(n_cul)[None, :], combos]  # (N, 5)
    align = np.nanmean(align_vals, axis=1)

    return combos, div, align


# ============================================================
# 保存 + 分析
# ============================================================
def save_csv(combos, div, align):
    with open(RESULT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CULTURES + ["Diversity", "Alignment"])
        for combo, d, a in zip(combos, div, align):
            w.writerow([MODELS[i] for i in combo] + [f"{d:.4f}", f"{a:.4f}"])
    print(f"Saved {len(combos)} combinations -> {RESULT_CSV}")


def _fmt(combo):
    return ", ".join(f"{c}:{MODELS[m]}" for c, m in zip(CULTURES, combo))


def show_topn(combos, div, align, n=10):
    print(f"\n=== Top {n} by Diversity ===")
    for i in np.argsort(-div)[:n]:
        print(f"  D={div[i]:6.2f}  A={align[i]:6.2f}  | {_fmt(combos[i])}")

    print(f"\n=== Bottom {n} by Diversity ===")
    for i in np.argsort(div)[:n]:
        print(f"  D={div[i]:6.2f}  A={align[i]:6.2f}  | {_fmt(combos[i])}")

    print(f"\n=== Top {n} by Alignment ===")
    for i in np.argsort(-align)[:n]:
        print(f"  D={div[i]:6.2f}  A={align[i]:6.2f}  | {_fmt(combos[i])}")

    # Pareto 前沿
    print(f"\n=== Pareto frontier (max Diversity & max Alignment) ===")
    order = np.argsort(-div)
    best_a = -np.inf
    pareto = []
    for i in order:
        if align[i] > best_a:
            pareto.append(i)
            best_a = align[i]

    for i in sorted(pareto, key=lambda k: div[k]):
        print(f"  D={div[i]:6.2f}  A={align[i]:6.2f}  | {_fmt(combos[i])}")


def show_homogeneous(combos, div, align):
    """5 个 culture 用同一模型 —— 对照 baseline，对应你原来的单 backbone 设置。"""
    print("\n=== Homogeneous backbones (all cultures = same model) ===")
    rows = []
    for mi, model in enumerate(MODELS):
        idx = np.where(np.all(combos == mi, axis=1))[0][0]
        rows.append((model, div[idx], align[idx]))
    for name, d, a in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:32s}  D={d:6.2f}  A={a:6.2f}")


def marginal_analysis(align_arr, pair_arr, base_combo, vary_culture):
    """固定其他 4 个 culture 的 backbone，只扫第 5 个，看边际影响。"""
    n_cul, n_mod = align_arr.shape
    vci = CULTURES.index(vary_culture)
    base = list(base_combo)
    fixed = ", ".join(f"{CULTURES[c]}:{MODELS[base[c]]}"
                      for c in range(n_cul) if c != vci)
    print(f"\n=== Marginal: fix [{fixed}], sweep {vary_culture} ===")
    rows = []
    for mi in range(n_mod):
        combo = base.copy()
        combo[vci] = mi
        d, npairs = 0.0, 0
        for ci, cj in combinations(range(n_cul), 2):
            d += pair_arr[ci, cj, combo[ci], combo[cj]]
            npairs += 1
        d = d / npairs * 100
        a = np.nanmean([align_arr[c, combo[c]] for c in range(n_cul)])
        rows.append((MODELS[mi], d, a))
    for name, d, a in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:32s}  D={d:6.2f}  A={a:6.2f}")

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


import numpy as np
import matplotlib.pyplot as plt


def pareto_frontier(d, a):
    """Pareto-optimal indices (maximize both), sorted by d ascending."""
    order = np.argsort(-d)
    best_a, pareto = -np.inf, []
    for i in order:
        if a[i] > best_a:
            pareto.append(i)
            best_a = a[i]
    return sorted(pareto, key=lambda k: d[k])


def knee_index(d, a, d_ref, a_ref):
    """Index of point closest to the ideal corner, normalized by reference ranges."""
    dn = (d - d_ref.min()) / (d_ref.max() - d_ref.min())
    an = (a - a_ref.min()) / (a_ref.max() - a_ref.min())
    return int(np.argmin(np.sqrt((1 - dn) ** 2 + (1 - an) ** 2)))


"""
Single-column drop-in replacement for plot_all_combinations.
Keeps pareto_frontier and knee_index helpers unchanged.
"""
import numpy as np
import matplotlib.pyplot as plt


def pareto_frontier(d, a):
    order = np.argsort(-d)
    best_a, pareto = -np.inf, []
    for i in order:
        if a[i] > best_a:
            pareto.append(i)
            best_a = a[i]
    return sorted(pareto, key=lambda k: d[k])


def knee_index(d, a, d_ref, a_ref):
    dn = (d - d_ref.min()) / (d_ref.max() - d_ref.min())
    an = (a - a_ref.min()) / (a_ref.max() - a_ref.min())
    return int(np.argmin(np.sqrt((1 - dn) ** 2 + (1 - an) ** 2)))

"""
Single-column drop-in replacement for plot_all_combinations.
Keeps pareto_frontier and knee_index helpers unchanged.
"""
import numpy as np
import matplotlib.pyplot as plt


def pareto_frontier(d, a):
    order = np.argsort(-d)
    best_a, pareto = -np.inf, []
    for i in order:
        if a[i] > best_a:
            pareto.append(i)
            best_a = a[i]
    return sorted(pareto, key=lambda k: d[k])


def knee_index(d, a, d_ref, a_ref):
    dn = (d - d_ref.min()) / (d_ref.max() - d_ref.min())
    an = (a - a_ref.min()) / (a_ref.max() - a_ref.min())
    return int(np.argmin(np.sqrt((1 - dn) ** 2 + (1 - an) ** 2)))


def plot_all_combinations(combos, div, align, MODELS,
                          save_path="fig2.png"):
    """Single-column version: no colorbar, compact fonts, short labels."""
    CULTURES = ["BRA", "CHN", "MEX", "NGA", "NZL"]
    n_mod = combos.max() + 1

    fig, ax = plt.subplots(figsize=(3.3, 3.0))

    # 1) hexbin density (no colorbar — meaning is intuitive, save horizontal space)
    ax.hexbin(div, align, gridsize=70, bins="log",
              cmap="Blues", mincnt=1, zorder=1)

    # 2) homogeneous baselines
    homo_idx = np.array([np.where(np.all(combos == mi, axis=1))[0][0]
                         for mi in range(n_mod)])
    homo_d, homo_a = div[homo_idx], align[homo_idx]
    ax.scatter(homo_d, homo_a, c="#D85A30", s=18, marker="s",
               edgecolors="white", linewidths=0.5, zorder=4,
               label="Single-Backbone")

    # 3) homogeneous Pareto frontier
    homo_par = pareto_frontier(homo_d, homo_a)
    ax.plot(homo_d[homo_par], homo_a[homo_par], "--", color="#D85A30",
            linewidth=0.9, alpha=0.75, zorder=4,
            label="Pareto (Single-Backbone)")

    # 4) heterogeneous Pareto frontier
    par = pareto_frontier(div, align)
    ax.plot(div[par], align[par], "-",color="#1D9E75", linewidth=1.2,
            zorder=5, label="Pareto (Mixed-Backbone)")

    # 5) bests
    i_het_A = int(np.argmax(align))
    i_het_D = int(np.argmax(div))
    i_het_K = knee_index(div, align, div, align)
    i_homo_A = int(homo_idx[np.argmax(homo_a)])
    i_homo_D = int(homo_idx[np.argmax(homo_d)])
    i_homo_K = int(homo_idx[knee_index(homo_d, homo_a, div, align)])

    # het stars + short labels
    star_specs = [
        (i_het_A, "max A",    (-4,  5), "right"),
        (i_het_D, "max D",    (8,  -10), "right"),
        (i_het_K, "balanced", ( 0,  5), "left"),
    ]
    for i, lbl, off, ha in star_specs:
        ax.scatter(div[i], align[i], marker="*", s=110, c="#1D9E75",
                   edgecolors="black", linewidths=0.5, zorder=7)
        ax.annotate(lbl, (div[i], align[i]), xytext=off,
                    textcoords="offset points", fontsize=6, ha=ha,
                    fontweight="bold", color="#0E5C45", zorder=8)

    # homo reference rings
    for i in [i_homo_A, i_homo_D, i_homo_K]:
        ax.scatter(div[i], align[i], marker="o", s=70, facecolors="none",
                   edgecolors="black", linewidths=0.8, zorder=6)

    # connectors + delta labels (ha controls which side the box extends)
    def connect(i_het, i_homo, text, off, ha="left"):
        ax.annotate("", xy=(div[i_het], align[i_het]),
                    xytext=(div[i_homo], align[i_homo]),
                    arrowprops=dict(arrowstyle="-|>", color="black",
                                    lw=0.8, ls="--", mutation_scale=8),
                    zorder=6)
        mx = (div[i_het] + div[i_homo]) / 2
        my = (align[i_het] + align[i_homo]) / 2
        ax.annotate(text, (mx, my), xytext=off, textcoords="offset points",
                    fontsize=5.5, zorder=9, ha=ha,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="black", lw=0.4, alpha=0.92))

    dA   = align[i_het_A] - align[i_homo_A]
    dD   = div[i_het_D]   - div[i_homo_D]
    dD_k = div[i_het_K]   - div[i_homo_K]
    dA_k = align[i_het_K] - align[i_homo_K]

    # max A sits mid-plot; box extends right.
    connect(i_het_A, i_homo_A, f"ΔA={dA:+.2f}",                   ( 8,  0), ha="left")
    # max D and balanced sit at the right edge; box extends LEFT.
    connect(i_het_D, i_homo_D, f"ΔD={dD:+.2f}",                   (-8, -12), ha="right")
    connect(i_het_K, i_homo_K, f"ΔD={dD_k:+.2f}, ΔA={dA_k:+.2f}", (1,  6),  ha="right")

    ax.set_xlabel("Diversity →", fontsize=8, labelpad=2)
    ax.set_ylabel("Alignment →", fontsize=8, labelpad=2)
    ax.tick_params(axis='both', labelsize=7, pad=2)

    # Padded limits — labels need breathing room
    d_lo, d_hi = div.min() - 0.5, div.max() + 0.8
    a_lo, a_hi = align.min() - 0.3, align.max() + 0.5
    ax.set_xlim(d_lo, d_hi)
    ax.set_ylim(a_lo, a_hi)

    ax.legend(loc="lower left", frameon=False, fontsize=5.5,
              handletextpad=0.3, borderpad=0.2, labelspacing=0.25)
    ax.grid(True, alpha=0.12, linewidth=0.5)

    # Fixed dimensions (no bbox_inches='tight') so all saved figures
    # are exactly figsize × dpi — matches fig2 for consistent rendering.
    plt.subplots_adjust(left=0.14, right=0.985, top=0.985, bottom=0.14)
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path.replace(".png", ".pdf"))
    print(f"Saved {save_path}")

    # numeric summary
    def fmt(i):
        return ", ".join(f"{c}:{MODELS[m]}" for c, m in zip(CULTURES, combos[i]))
    print("=== Heterogeneous bests ===")
    print(f"  max A: D={div[i_het_A]:.2f} A={align[i_het_A]:.2f} | {fmt(i_het_A)}")
    print(f"  max D: D={div[i_het_D]:.2f} A={align[i_het_D]:.2f} | {fmt(i_het_D)}")
    print(f"  bal:   D={div[i_het_K]:.2f} A={align[i_het_K]:.2f} | {fmt(i_het_K)}")
    print(f"=== Gains: ΔA={dA:+.2f} | ΔD={dD:+.2f} | ΔD={dD_k:+.2f}, ΔA={dA_k:+.2f}")
    print("\n=== Homogeneous Pareto frontier (left → right) ===")

    for rank, k in enumerate(homo_par):

        i = homo_idx[k]

        model_name = MODELS[combos[i][0]]

        print(
            f"[{rank:02d}] "
            f"{model_name:32s} "
            f"D={div[i]:6.2f}  "
            f"A={align[i]:6.2f}"
        )

import json

def save_collab_dataset(combos, div, align,
                        csv_path="agent_systems.csv",
                        legend_path="agent_systems_legend.json"):
    """
    每行 = 一个 agent system（5 个 agents）
    列: agent{i}_backbone, agent{i}_identity for i=0..4, diversity, alignment
    backbone_id ∈ [0, len(MODELS)-1]; identity_id ∈ [0, len(CULTURES)-1]
    """
    n_cul = len(CULTURES)
    identity_row = list(range(n_cul))  # 所有样本都是 [0,1,2,3,4]

    header = []
    for i in range(n_cul):
        header += [f"agent{i}_backbone", f"agent{i}_identity"]
    header += ["diversity", "alignment"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for combo, d, a in zip(combos, div, align):
            row = []
            for i in range(n_cul):
                row += [int(combo[i]), identity_row[i]]
            row += [f"{d:.4f}", f"{a:.4f}"]
            w.writerow(row)

    # 附一份 id ↔ name 的对照表
    legend = {
        "backbone_id": {i: m for i, m in enumerate(MODELS)},
        "identity_id": {i: c for i, c in enumerate(CULTURES)},
        "n_systems": len(combos),
        "schema": header,
    }
    with open(legend_path, "w", encoding="utf-8") as f:
        json.dump(legend, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(combos):,} systems -> {csv_path}")
    print(f"Saved id legend           -> {legend_path}")

if __name__ == "__main__":
    t0 = time.time()
    references, ref_dict = load_references()
    culture_data = load_culture_data()

    print("Precomputing alignment + pairwise distances ...")
    align_arr, pair_arr = precompute(references, ref_dict, culture_data)
    print(f"  done in {time.time()-t0:.1f}s")

    t1 = time.time()
    print(f"Enumerating all {len(MODELS)}**{len(CULTURES)} = {len(MODELS)**len(CULTURES):,} combinations ...")
    combos, div, align = enumerate_combinations(align_arr, pair_arr)

    # plot_all_combinations(combos, div, align, align_arr, pair_arr)

    save_csv(combos, div, align)

    # ---- 分析 ----
    show_topn(combos, div, align, n=20)
    show_homogeneous(combos, div, align)

    # 示例：取 Diversity 最高的组合，扫 CHN 看边际影响
    best = combos[np.argmax(div)]
    marginal_analysis(align_arr, pair_arr, best, vary_culture="CHN")

    print(f"\nTotal: {time.time()-t0:.1f}s")

    save_csv(combos, div, align)
    save_collab_dataset(combos, div, align)

    plot_all_combinations(combos, div, align, MODELS)
    print(f"  done in {time.time()-t1:.1f}s")