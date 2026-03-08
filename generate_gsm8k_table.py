import json
import os

filepaths = [
    "/data/jacky/small_project_1/results/gsm8k/eval/chat_grpo_default_eval_chat/results.json",
    "/data/jacky/small_project_1/results/gsm8k/eval/chat_grpo_default_eval_raw/results.json",
    "/data/jacky/small_project_1/results/gsm8k/eval/chat_onpolicysft_maskq_eval_chat/results.json",
    "/data/jacky/small_project_1/results/gsm8k/eval/chat_onpolicysft_maskq_eval_raw/results.json",
    "/data/jacky/small_project_1/results/gsm8k/eval/chat_sdft_eval_chat/results.json",
    "/data/jacky/small_project_1/results/gsm8k/eval/chat_sdft_eval_raw/results.json",
    "/data/jacky/small_project_1/results/gsm8k/eval/chat_sft_maskq_eval_chat/results.json",
    "/data/jacky/small_project_1/results/gsm8k/eval/chat_sft_maskq_eval_raw/results.json",
    "/data/jacky/small_project_1/results/gsm8k/eval/nochat_grpo_default_eval_chat/results.json",
    "/data/jacky/small_project_1/results/gsm8k/eval/nochat_grpo_default_eval_raw/results.json",
    "/data/jacky/small_project_1/results/gsm8k/eval/nochat_sft_maskq_eval_raw/results.json",
    "/data/jacky/small_project_1/results/gsm8k/eval/nochat_sft_maskq_eval_chat/results.json",
    "/data/jacky/small_project_1/results/gsm8k/eval/nochat_sdft_eval_raw/results.json",
    "/data/jacky/small_project_1/results/gsm8k/eval/nochat_sdft_eval_chat/results.json",
    "/data/jacky/small_project_1/results/gsm8k/eval/nochat_onpolicysft_maskq_eval_raw/results.json",
    "/data/jacky/small_project_1/results/gsm8k/eval/nochat_onpolicysft_maskq_eval_chat/results.json"
]

methods_mapping = {
    "grpo_default": "GRPO",
    "onpolicysft": "OnPolicySFT",
    "sdft": "SDFT",
    "sft": "SFT"
}

rows = []
for p in filepaths:
    if not os.path.exists(p):
        print(f"File not found: {p}")
        continue
    with open(p, 'r') as f:
        data = json.load(f)
    
    summary = data.get('summary', {})
    if not summary:
        print(f"No summary in {p}")
        continue

    acc = summary.get('accuracy', 0) * 100
    correct = summary.get('correct', 0)
    total = summary.get('total_samples', 0)
    
    # parse from path
    dirname = os.path.basename(os.path.dirname(p))
    
    # e.g. chat_grpo_default_eval_chat
    train_c = "train:chat" if dirname.startswith("chat_") else "train:raw"
    eval_c = "eval:chat" if dirname.endswith("_eval_chat") else "eval:raw"
    mask_q = "mask_q" if "_maskq" in dirname else ""
    
    method_key = dirname.split("_eval_")[0]
    if method_key.startswith("nochat_"):
        method_key = method_key[7:]
    elif method_key.startswith("chat_"):
        method_key = method_key[5:]
    if "_maskq" in method_key:
        method_key = method_key.replace("_maskq", "")
        
    method = methods_mapping.get(method_key, method_key.upper())
    
    if mask_q:
        experiment = f"{method:<3} | {train_c} | {mask_q} | {eval_c}"
    else:
        experiment = f"{method:<3} | {train_c} | {eval_c}"
    
    rows.append((method, train_c, mask_q, eval_c, experiment, acc, correct, total))

def sort_key(row):
    method, train_c, mask_q, eval_c, exp, acc, cor, tot = row
    method_order = {"GRPO": 1, "SDFT": 2, "OnPolicySFT": 3, "SFT": 4}
    return (method_order.get(method, 99), train_c, mask_q, eval_c)

rows.sort(key=sort_key)

print("="*97)
print(f"{'Experiment':<72} {'Accuracy':>8} {'Correct':>9} {'Total':>7}")
print("-" * 97)
for row in rows:
    _, _, _, _, exp, acc, cor, tot = row
    print(f"{exp:<72} {acc:>7.2f}% {cor:>9} {tot:>7}")
print("="*97)
