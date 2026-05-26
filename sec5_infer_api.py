import json
import os
import requests
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


openrouter_key = "XXX"
# ==========================================================
# Argument Parser
# ==========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--culture",
    type=str,
    required=True,
    help="Culture code, e.g. BRA / CHN / NGA"
)

parser.add_argument(
    "--model",
    type=str,
    required=True,
)

args = parser.parse_args()

culture = args.culture
model_name = args.model
print(culture)
culture_dict = {
    "BRA": ["Brazil", "Brazilian"],
    "CHN": ["China", "Chinese"],
    "MEX": ["Mexico", "Mexican"],
    "NGA": ["Nigeria", "Nigerian"],
    "NZL": ["New Zealand", "New Zealander"]
}

culture_dict = {
    "AUS": ["Australia",       "Australian"],
    "BOL": ["Bolivia",         "Bolivian"],
    "BRA": ["Brazil",          "Brazilian"],
    "CAN": ["Canada",          "Canadian"],
    "CHN": ["China",           "Chinese"],
    "DEU": ["Germany",         "German"],
    "ETH": ["Ethiopia",        "Ethiopian"],
    "GBR": ["United Kingdom",  "British"],
    "IND": ["India",           "Indian"],
    "KEN": ["Kenya",           "Kenyan"],
    "MEX": ["Mexico",          "Mexican"],
    "NGA": ["Nigeria",         "Nigerian"],
    "NLD": ["Netherlands",     "Dutch"],
    "NZL": ["New Zealand",     "New Zealander"],
    "RUS": ["Russia",          "Russian"],
    "THA": ["Thailand",        "Thai"],
    "UKR": ["Ukraine",         "Ukrainian"],
    "USA": ["United States",   "American"],
    "ZWE": ["Zimbabwe",        "Zimbabwean"],
}


if model_name in ['gpt-5-mini', 'gemini-2.5-pro', 'grok-4']:
    aaa = {"enabled": True}
else:
    aaa = {"enabled": False}

model_key = ""

if "claude" in model_name:
    model_key = "anthropic"
elif "gpt" in model_name:
    model_key = "openai"
elif "gemini" in model_name:
    model_key = "google"
elif "grok" in model_name:
    model_key = "x-ai"
elif "llama" in model_name:
    model_key = "meta-llama"

output_dir = "wvs_evaluation"
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/{model_name}_{culture}_system.jsonl"

lock = threading.Lock()  # ⭐防止多线程写文件冲突

def call_api(idx, prompt, q_ids, questions):
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": openrouter_key,
                "Content-Type": "application/json",
            },
            json={
                "model": f"{model_key}/{model_name}",
                "messages": [
                    {"role": "system", "content": prompt["prompt"]["system"]},
                    {"role": "user", "content": prompt["prompt"]["instruction"]}
                ],
                "reasoning": aaa
            },
            timeout=20
        )

        res_json = response.json()

        if "choices" not in res_json:
            raise Exception(res_json)

        content = res_json["choices"][0]["message"]["content"]

    except Exception as e:
        content = f"ERROR: {str(e)}"

    return {
        "idx": idx,
        "q_id": q_ids[idx],
        "question": questions[idx],
        "system": prompt["prompt"]["system"],
        "instruction": prompt["prompt"]["instruction"],
        "response": content
    }


# ===== 读取数据 =====
with open("data/wvs.json", "r", encoding="utf-8") as f:
    data = json.load(f)

questions = [item["question"] for item in data]
q_ids = [item["Q_id"] for item in data]

def build_prompt(country, culture_name, question):

    system_prompt = f"You are a respondent from {country}. Answer questions based on typical cultural values in {culture_name} culture."
    
    instruction = f"""Question: {question}

Output format MUST be exactly: \\boxed{{number}}"""
    
    return {
        "system": system_prompt,
        "instruction": instruction
    }


all_prompts = []
for idx, q in enumerate(questions):
    all_prompts.append({
        "idx": idx,
        "prompt": build_prompt(culture_dict[culture][0], culture_dict[culture][1], q)
    })


# ===== 跳过已完成 =====
done_ids = set()
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                done_ids.add(json.loads(line)["idx"])
            except:
                pass

print(f"Already done: {len(done_ids)}")


# ===== 并发执行 =====
start_time = time.time()

max_workers = 5   # ⭐可以调

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []

    for idx, prompt in enumerate(all_prompts):
        if idx in done_ids:
            continue

        futures.append(
            executor.submit(call_api, idx, prompt, q_ids, questions)
        )

    with open(output_path, "a", encoding="utf-8") as fout:
        for future in as_completed(futures):
            result = future.result()

            # ⭐线程安全写入
            with lock:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

            print(f"[{result['idx']}] done")

end_time = time.time()

print(f"\nTotal time: {end_time - start_time:.2f}s")