import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="HF inference for cultural value questions")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--culture", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--data_name", type=str, choices=["blend", "wvs"], default="wvs")
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="Equivalent to max_new_tokens in HF generate")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Replaces vLLM's max_num_seqs")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device_map", type=str, default="auto",
                        help="'auto' shards across visible GPUs; use 'cuda:0' to pin one GPU")
    parser.add_argument("--output_dir", type=str, default="wvs_evaluation_interaction")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    culture_dict = {
        "BRA": ["Brazil", "Brazilian"],
        "CHN": ["China", "Chinese"],
        "MEX": ["Mexico", "Mexican"],
        "NGA": ["Nigeria", "Nigerian"],
        "NZL": ["New Zealand", "New Zealander"],
    }

    country, culture_name = culture_dict[args.culture]
    system_prompt = (
        f"You are a respondent from {country}. "
        f"Answer questions based on typical cultural values in {culture_name} culture."
    )

    # ---- Load data ----
    with open("data/wvs.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = [item["question"] for item in data]
    q_ids     = [item["Q_id"]    for item in data]

    input_data = {}
    
    model_tag   = args.model.rstrip("/").split("/")[-1]

    for cul in culture_dict:
        if cul == args.culture: continue
        input_path = f"wvs_evaluation/{model_tag}_{cul}_system.jsonl"
        with open(input_path, "r", encoding="utf-8") as f:
            data = [json.loads(l) for l in f]
            data.sort(key=lambda x: x["idx"])
        
        for i_idx, item in enumerate(data):
            assert item['q_id'] == q_ids[i_idx]

        input_data[cul] = data
    
    items = []
    for idx, q in enumerate(questions):
        others_solution = ""
        cot = 0
        max_cot = 4
        for cul in input_data:
            if cot == max_cot:
                break
            other_solution = input_data[cul][idx]
            assert other_solution['question'] == q
            others_solution += f"{culture_dict[cul][1]}: {other_solution['response']}\n"
            cot += 1
        user_msg = f"""Question: {q}

Here are answers from people of other cultures: {others_solution}

You may consider these answers before making your decision.

Output format MUST be exactly: \\boxed{{number}}"""
        
        items.append({"idx": idx, "user_msg": user_msg})

    print(f"Total questions: {len(items)}")

    # ---- Load model & tokenizer ----
    dtype = {"bfloat16": torch.bfloat16,
             "float16":  torch.float16,
             "float32":  torch.float32}[args.dtype]

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # causal LM batched generation needs left padding

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    # ---- Build prompts via chat template ----
    def build_prompt(user_msg):
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ]
        # enable_thinking=False is Qwen3-specific; fall back gracefully for others
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

    prompts = [build_prompt(it["user_msg"]) for it in items]

    # ---- Generate in batches ----
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    results = []
    for start in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
        batch_prompts = prompts[start: start + args.batch_size]
        batch_items   = items[start:   start + args.batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16384,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.95,
                top_k=20,
                max_new_tokens=args.max_tokens,
                num_return_sequences=args.sample_num,
                pad_token_id=tokenizer.pad_token_id,
            )

        # outputs: (batch_size * sample_num, prompt_len + gen_len), left-padded.
        # All prompts in a batch share the same padded length, so we can slice uniformly.
        input_len = inputs["input_ids"].shape[1]
        gen_only  = outputs[:, input_len:]
        decoded   = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        # When num_return_sequences > 1, samples are grouped per input:
        # [s0_in0, s1_in0, ..., s0_in1, s1_in1, ...]
        for i, it in enumerate(batch_items):
            samples = decoded[i * args.sample_num: (i + 1) * args.sample_num]
            results.append({
                "idx":      it["idx"],
                "q_id":     q_ids[it["idx"]],
                "question": questions[it["idx"]],
                "prompt":   batch_prompts[i],
                "response": samples[0] if args.sample_num == 1 else samples,
            })

    # ---- Save ----
    os.makedirs(args.output_dir, exist_ok=True)
    model_tag   = args.model.rstrip("/").split("/")[-1]
    output_path = f"{args.output_dir}/{model_tag}_{args.culture}_system.jsonl"
    with open(output_path, "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()