import json
import os
import re
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

parser.add_argument(
    "--n_runs",
    type=int,
    default=20,
    help="Number of independent LLM votes for this (model, culture)"
)

args = parser.parse_args()

culture = args.culture
model_name = args.model
n_runs = args.n_runs

print(culture)

# ==========================================================
# Culture Mapping
# ==========================================================

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

# ==========================================================
# Reasoning Toggle
# ==========================================================

if model_name in ['gpt-5-mini', 'gemini-2.5-pro', 'grok-4']:
    aaa = {"enabled": True}
else:
    aaa = {"enabled": False}

# ==========================================================
# Model Provider
# ==========================================================

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

# ==========================================================
# Output
# ==========================================================

output_dir = "wvs_pb_evaluation"
os.makedirs(output_dir, exist_ok=True)

output_path = f"{output_dir}/{model_name}_{culture}.jsonl"

lock = threading.Lock()

# ==========================================================
# Load WVS-grounded PB projects
# ==========================================================

with open("data/wvs_project.json", "r", encoding="utf-8") as f:
    projects = json.load(f)

valid_pids = {p["id"] for p in projects}

# ==========================================================
# Build project text
# ==========================================================

def build_project_list_text():
    lines = []

    for p in projects:

        desc = p["description"].strip()

        lines.append(
            f"#{p['id']}: {p['name']}\n"
            f"Category: {p['category']}\n"
            f"WVS Dimension: {p['dimension']}\n"
            f"Budget: CHF {p['cost']}\n"
            f"Description: {desc}"
        )

    return "\n\n".join(lines)

PROJECT_LIST_TEXT = build_project_list_text()

# ==========================================================
# Build Prompt
# ==========================================================

def build_prompt(country, culture_name):

    system_prompt = (
        f"You are a respondent from {country}. "
        f"Answer based on typical societal priorities, cultural values, "
        f"and public policy preferences commonly associated with "
        f"{culture_name} culture."
    )

    instruction = (
        f"A national participatory budgeting program is allocating limited "
        f"public funding across societal development initiatives.\n\n"

        f"There are {len(projects)} candidate projects:\n\n"

        f"{PROJECT_LIST_TEXT}\n\n"

        f"Voting instruction:\n"
        f"- Select EXACTLY 4 projects that should receive funding.\n"
        f"- Base your choices on long-term societal priorities and values.\n"
        f"- Consider cultural preferences, governance priorities, "
        f"social development, and public well-being.\n\n"

        f"Output format MUST be exactly:\n"
        f"\\boxed{{id1, id2, id3, id4}}\n\n"

        f"For example:\n"
        f"\\boxed{{2, 5, 8, 11}}"
    )

    return {
        "system": system_prompt,
        "instruction": instruction,
    }

# ==========================================================
# Parse boxed ids
# ==========================================================

BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
INT_RE = re.compile(r"\d+")

def parse_projects(response_text, n_select=4):

    m = BOXED_RE.search(response_text)

    if m:
        candidates = [int(x) for x in INT_RE.findall(m.group(1))]
    else:
        candidates = [int(x) for x in INT_RE.findall(response_text)]

    seen = set()
    valid = []

    for pid in candidates:

        if pid in valid_pids and pid not in seen:
            seen.add(pid)
            valid.append(pid)

        if len(valid) == n_select:
            break

    if len(valid) != n_select:
        return None

    return valid

# ==========================================================
# API Call
# ==========================================================

def call_api(run_idx, prompt):

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
                    {
                        "role": "system",
                        "content": prompt["system"]
                    },
                    {
                        "role": "user",
                        "content": prompt["instruction"]
                    }
                ],
                "reasoning": aaa
            },
            timeout=60
        )

        res_json = response.json()

        if "choices" not in res_json:
            raise Exception(res_json)

        content = res_json["choices"][0]["message"]["content"]

        parsed = parse_projects(content)

    except Exception as e:

        content = f"ERROR: {str(e)}"
        parsed = None

    return {
        "run_idx": run_idx,
        "culture": culture,
        "system": prompt["system"],
        "instruction": prompt["instruction"],
        "response": content,
        "parsed_projects": parsed,
        "parse_ok": parsed is not None,
    }

# ==========================================================
# Generate Runs
# ==========================================================

prompt = build_prompt(
    culture_dict[culture][0],
    culture_dict[culture][1]
)

all_runs = [
    {
        "run_idx": i,
        "prompt": prompt
    }
    for i in range(n_runs)
]

# ==========================================================
# Resume Existing
# ==========================================================

done_ids = set()

if os.path.exists(output_path):

    with open(output_path, "r", encoding="utf-8") as f:

        for line in f:

            try:
                rec = json.loads(line)

                if rec.get("parse_ok"):
                    done_ids.add(rec["run_idx"])

            except:
                pass

print(f"Already done (parse_ok): {len(done_ids)} / {n_runs}")

# ==========================================================
# Run Concurrently
# ==========================================================

start_time = time.time()

max_workers = 5

with ThreadPoolExecutor(max_workers=max_workers) as executor:

    futures = []

    for run in all_runs:

        if run["run_idx"] in done_ids:
            continue

        futures.append(
            executor.submit(
                call_api,
                run["run_idx"],
                run["prompt"]
            )
        )

    with open(output_path, "a", encoding="utf-8") as fout:

        for future in as_completed(futures):

            result = future.result()

            with lock:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

            ok = "✓" if result["parse_ok"] else "✗"

            print(
                f"[{result['run_idx']}] "
                f"{ok} parsed={result['parsed_projects']}"
            )

end_time = time.time()

print(f"\nTotal time: {end_time - start_time:.2f}s")