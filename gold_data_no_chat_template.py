#!/usr/bin/env python3
"""Gold data extraction without chat template (raw question fed to model).

Generates 8 rollouts per GSM8K training question, logs accuracy and responses
continuously to JSONL, then selects 1000 questions with the lowest accuracy
(requiring 2-6 correct rollouts out of 8).

Usage:
    python gold_data_no_chat_template.py --gpu 0
    python gold_data_no_chat_template.py --gpu 0 --batch_size 8
    python gold_data_no_chat_template.py --gpu 0,1 --batch_size 8
"""

import argparse
import json
import os
import sys


def _parse_gpu_arg():
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "0"


os.environ["CUDA_VISIBLE_DEVICES"] = _parse_gpu_arg()

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from tasks import get_task


def get_attn_implementation():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return None


def format_prompt(question):
    return question


def generate_rollouts(model, tokenizer, prompts, num_rollouts, max_new_tokens, temperature):
    """Generate num_rollouts completions for each prompt in a single batched call."""
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, add_special_tokens=False
    ).to(model.device)

    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            num_return_sequences=num_rollouts,
            pad_token_id=tokenizer.pad_token_id,
        )

    tokenizer.padding_side = original_padding_side

    all_completions = []
    for i in range(len(prompts)):
        prompt_len = prompt_lengths[i]
        pad_len = (inputs["attention_mask"][i] == 0).sum().item()
        start_idx = pad_len + prompt_len

        completions = []
        for j in range(num_rollouts):
            idx = i * num_rollouts + j
            generated_ids = output_ids[idx][start_idx:]
            completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
            completions.append(completion)
        all_completions.append(completions)

    return all_completions


def main():
    parser = argparse.ArgumentParser(
        description="Gold data extraction without chat template (raw question)"
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output_dir", type=str, default="results/gold_data/no_chat_template")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of questions per generate call")
    parser.add_argument("--num_rollouts", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--num_select", type=int, default=1000)
    parser.add_argument("--min_correct", type=int, default=2)
    parser.add_argument("--max_correct", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = get_attn_implementation()
    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.eval()

    print("Loading GSM8K training data...")
    task = get_task("gsm8k")
    train_samples = task.load_train()
    print(f"Total training examples: {len(train_samples)}")

    rollouts_path = os.path.join(args.output_dir, "rollouts.jsonl")
    processed_indices = set()
    if os.path.exists(rollouts_path):
        with open(rollouts_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    processed_indices.add(entry["index"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Resuming: {len(processed_indices)} questions already processed")

    remaining_indices = [i for i in range(len(train_samples)) if i not in processed_indices]
    if not remaining_indices:
        print("All questions already processed. Skipping to selection phase.")
    else:
        log_file = open(rollouts_path, "a")
        num_batches = (len(remaining_indices) + args.batch_size - 1) // args.batch_size
        eligible_so_far = 0

        pbar = tqdm(
            total=len(train_samples),
            initial=len(processed_indices),
            desc="Generating rollouts (no chat template)",
        )

        for batch_idx in range(num_batches):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, len(remaining_indices))
            batch_indices = remaining_indices[start:end]

            batch_samples = [train_samples[i] for i in batch_indices]
            batch_prompts = [format_prompt(s.question) for s in batch_samples]

            batch_completions = generate_rollouts(
                model, tokenizer, batch_prompts,
                args.num_rollouts, args.max_new_tokens, args.temperature,
            )

            for sample_idx, sample, completions in zip(
                batch_indices, batch_samples, batch_completions
            ):
                rollouts = []
                num_correct = 0
                for completion in completions:
                    extracted = task.extract_answer(completion)
                    gold = sample.answer_value if sample.answer_value else sample.answer
                    is_correct = task.check_answer(extracted, gold)
                    if is_correct:
                        num_correct += 1
                    rollouts.append({
                        "response": completion,
                        "extracted_answer": extracted,
                        "correct": is_correct,
                    })

                accuracy = num_correct / args.num_rollouts
                if args.min_correct <= num_correct <= args.max_correct:
                    eligible_so_far += 1

                entry = {
                    "index": sample_idx,
                    "question": sample.question,
                    "gold_answer": sample.answer,
                    "gold_value": sample.answer_value,
                    "num_correct": num_correct,
                    "num_rollouts": args.num_rollouts,
                    "accuracy": accuracy,
                    "rollouts": rollouts,
                }

                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()

                pbar.update(1)
                pbar.set_postfix({
                    "acc": f"{accuracy:.0%}",
                    "eligible": eligible_so_far,
                })

        log_file.close()
        pbar.close()

    # --- Selection phase ---
    print("\n" + "=" * 60)
    print("  Selection Phase")
    print("=" * 60)

    all_results = []
    with open(rollouts_path) as f:
        for line in f:
            try:
                all_results.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"Total questions processed: {len(all_results)}")

    eligible = [
        r for r in all_results
        if args.min_correct <= r["num_correct"] <= args.max_correct
    ]
    print(f"Questions with {args.min_correct}-{args.max_correct} correct: {len(eligible)}")

    eligible.sort(key=lambda r: r["accuracy"])
    selected = eligible[: args.num_select]
    print(f"Selected {len(selected)} questions (target: {args.num_select})")

    if len(selected) < args.num_select:
        print(
            f"WARNING: Only {len(selected)} questions met the criteria, "
            f"fewer than the target of {args.num_select}"
        )

    gold_data_path = os.path.join(args.output_dir, "gold_data.json")
    with open(gold_data_path, "w") as f:
        json.dump(
            {
                "config": {
                    "model_name": args.model_name,
                    "num_rollouts": args.num_rollouts,
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                    "min_correct": args.min_correct,
                    "max_correct": args.max_correct,
                    "num_select": args.num_select,
                    "format": "no_chat_template",
                },
                "total_questions": len(all_results),
                "eligible_questions": len(eligible),
                "selected_questions": len(selected),
                "data": selected,
            },
            f,
            indent=2,
        )
    print(f"Gold data saved to {gold_data_path}")

    from collections import Counter

    acc_dist = Counter(r["num_correct"] for r in all_results)
    print("\nAccuracy distribution (num_correct -> count):")
    for k in sorted(acc_dist.keys()):
        print(f"  {k}/{args.num_rollouts}: {acc_dist[k]} questions")


if __name__ == "__main__":
    main()
