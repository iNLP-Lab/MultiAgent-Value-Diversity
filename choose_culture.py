import json
import math
from itertools import combinations

file_name = "data/proportions_group_by_country.json"

path = "data/wvs.json"
with open(path, "r", encoding="utf-8") as f:
    wvs_data = json.load(f)
q_option_nums = {}
questions = [item["question"] for item in wvs_data]
q_ids = [item["Q_id"] for item in wvs_data]
for item in wvs_data:

    q_id = item["Q_id"]
    options = item["option"]
    q_option_nums[q_id] = len(options)

# ===== Load =====
with open(file_name, "r", encoding="utf-8") as f:
    data = json.load(f)

culture_lst = list(data.keys())

# ===== Step 1: Convert each culture to majority-vote vector =====

culture_vectors = {}

for culture, q_dict in data.items():

    vec = {}

    for q_id, option_dist in q_dict.items():
        if q_id not in q_ids:
            print(q_id)
            continue
        # majority vote option
        if not option_dist:
            continue

        majority_option = max(
            option_dist.items(),
            key=lambda x: x[1]
        )[0]

        vec[q_id] = float(majority_option)

    culture_vectors[culture] = vec
# ===== Step 2: Euclidean distance between two cultures =====

def euclidean_distance(vec1, vec2):

    common_qs = set(vec1.keys()) & set(vec2.keys())

    squared_sum = 0.0
    max_squared_sum = 0.0

    for q in common_qs:

        if q not in q_option_nums:
            exit()
            continue

        delta = q_option_nums[q] - 1

        if delta == 0:
            exit()
            continue

        # numerator
        squared_sum += (vec1[q] - vec2[q]) ** 2

        # denominator
        max_squared_sum += delta ** 2

    if max_squared_sum == 0:
        return 0.0

    return math.sqrt(squared_sum) / math.sqrt(max_squared_sum)

# ===== Step 3: Compute diversity of a culture group =====

def group_diversity(cultures):

    pair_dists = []

    for c1, c2 in combinations(cultures, 2):

        d = euclidean_distance(
            culture_vectors[c1],
            culture_vectors[c2]
        )

        pair_dists.append(d)

    return sum(pair_dists) / len(pair_dists)

# ===== Step 4: Enumerate all 5-culture combinations =====

results = []

all_cultures = list(culture_vectors.keys())

for combo in combinations(all_cultures, 5):

    div = group_diversity(combo)

    results.append((combo, div))

# ===== Step 5: Sort by diversity =====

results.sort(key=lambda x: x[1], reverse=True)

# ===== Top 20 =====

for combo, div in results[:50]:
    print(combo, round(div, 4))