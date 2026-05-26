import json
import math
import os
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from utils import normalize_latex_text, normalized_entropy, extract_boxed_answer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_majority_answer(dist):
    """dist: {"1": 0.7, "2": 0.2, ...} -> int"""
    return int(max(dist.items(), key=lambda x: x[1])[0])


def get_option_range(reference_item):
    num_options = len(reference_item["option"])
    return 1, num_options


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
    assert len(data) == len(references), print(len(data), len(references))
    for id_, item in enumerate(data):
        reference = references[id_]
        q_id = reference["Q_id"]
        assert q_id == item["q_id"]
        resp = item["response"]
        try:
            ans = normalize_latex_text(extract_boxed_answer(resp).lower())
        except:
            continue
        if not ans:
            continue
        try:
            R[q_id] = int(ans)
        except Exception:
            continue
        q_ranges[q_id] = get_option_range(reference)

    if len(q_ranges) < 200:
        print("!!!", output_path)

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
        print(None)
        return None

    score = (1 - math.sqrt(squared_sum) / math.sqrt(max_squared_sum)) * 100
    print(f"{culture} Alignment: {score:.2f}")
    return score


def compute_pairwise(all_answers, ref_dict):
    """
    Compute pairwise normalized Euclidean distances across all agent pairs.
    Returns:
      mean_dist : Diversity(S)   -- mean of pairwise distances
      pair_dict : {(agent_i, agent_j): dist}
      agents    : sorted list of agent ids
    """
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

        dist = math.sqrt(squared_sum) / math.sqrt(max_squared_sum)
        pairwise_scores.append(dist)
        pair_dict[(a1, a2)] = dist

    if not pairwise_scores:
        return 0.0, {}, agents

    mean_dist = sum(pairwise_scores) / len(pairwise_scores)
    return mean_dist, pair_dict, agents


def compute_mst_span(pair_dict, agents):
    """
    Minimum-spanning-tree total length, normalized by (N-1) so that the
    metric lies in [0, 1] and is comparable to Diversity(S).

    Returns:
      mst_span  : MST length / (N - 1)   -- normalized total spread
      mst_edges : list of (agent_i, agent_j, weight) tuples in the MST
    """
    n = len(agents)
    if n < 2 or not pair_dict:
        return 0.0, []

    # Build N x N symmetric distance matrix
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = agents[i], agents[j]
            d = pair_dict.get((a, b), pair_dict.get((b, a), 0.0))
            M[i, j] = d
            M[j, i] = d

    mst = minimum_spanning_tree(M).toarray()
    mst_span = float(mst.sum()) / (n - 1)

    mst_edges = []
    for i in range(n):
        for j in range(n):
            if mst[i, j] > 0:
                mst_edges.append((agents[i], agents[j], float(mst[i, j])))

    return mst_span, mst_edges


def diversity(output_dir, culture_lst, ground_truth=False):
    path = "data/wvs.json"
    with open(path, "r", encoding="utf-8") as f:
        references = json.load(f)

    with open("data/proportions_group_by_country.json", "r", encoding="utf-8") as f:
        culture_data = json.load(f)

    all_answers = defaultdict(dict)
    ref_dict = {r["Q_id"]: r for r in references}

    if not ground_truth:
        for cul_id, culture in enumerate(culture_lst):
            output_path = f"{output_dir}/{model}_{culture}_system.jsonl"
            print(output_path)
            with open(output_path, 'r') as f:
                data = [json.loads(l) for l in f]
            data.sort(key=lambda x: x["idx"])
            assert len(data) == len(references), f"数据长度不匹配: {len(data)} vs {len(references)}"
            for id_, item in enumerate(data):
                reference = references[id_]
                resp = item["response"]
                assert reference["Q_id"] == item["q_id"]
                try:
                    ans = normalize_latex_text(extract_boxed_answer(resp).lower())
                except:
                    continue
                if not ans:
                    continue
                q_id = reference["Q_id"]
                try:
                    ans_int = int(ans)
                except:
                    continue
                all_answers[f"{culture}_{cul_id}"][q_id] = ans_int
    else:
        for cul_id, culture in enumerate(culture_lst):
            if culture not in culture_data:
                continue
            q_dict = culture_data[culture]
            for q_id, option_dist in q_dict.items():
                if q_id not in ref_dict:
                    continue
                if not option_dist:
                    continue
                majority_option = max(option_dist.items(), key=lambda x: x[1])[0]
                try:
                    ans_int = int(majority_option)
                except:
                    continue
                all_answers[f"{culture}_{cul_id}"][q_id] = ans_int

    mean_div, pair_dict, agents = compute_pairwise(all_answers, ref_dict)
    mst_span, mst_edges = compute_mst_span(pair_dict, agents)

    print(f"Diversity (mean of pairwise):  {mean_div * 100:.2f}")
    print(f"MST Span  (MST length / N-1):  {mst_span * 100:.2f}")

    if mst_edges:
        edge_str = ", ".join(f"{a}-{b}:{w*100:.2f}" for a, b, w in mst_edges)
        print(f"  MST edges: {edge_str}")


if __name__ == "__main__":
    # model = "gpt-5.4"
    # model = "gpt-5-mini"
    # model = "gpt-4o-mini"
    # model = "claude-opus-4.7"
    # model = "claude-sonnet-4.5"
    # model = "claude-3.5-haiku"
    # model = "gemini-3.1-flash-lite-preview"
    # model = "gemini-3-flash-preview"
    # model = "gemini-2.5-pro"
    model = "grok-4.3"
    # model = "grok-4"
    # model = "grok-3"
    # model = "Qwen3.5-27B"
    # model = "Qwen3-32B"
    # model = "Qwen2.5-32B-Instruct"
    # model = "llama-4-scout"
    # model = "llama-3.3-70b-instruct"
    # model = "llama-3.1-70b-instruct"

    output_dir = "wvs_evaluation"
    culture_lst = ['BRA', 'CHN', 'MEX', 'NGA', 'NZL']

    score_lst = []
    for cul in culture_lst:
        output_path = f"{output_dir}/{model}_{cul}_system.jsonl"
        score = alignment(output_path, cul)
        score_lst.append(score)

    avg_score = sum(score_lst) / len(score_lst)
    print(f"AVG: {avg_score:.2f}")

    print("\n--- LLM system ---")
    diversity(output_dir, culture_lst, ground_truth=False)

    print("\n--- Human reference ---")
    diversity(output_dir, culture_lst, ground_truth=True)