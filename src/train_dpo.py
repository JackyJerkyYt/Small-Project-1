"""DPO training script with offline rejection sampling for pair construction."""

import argparse
import copy
import json
import os
import random
from datetime import datetime
from zoneinfo import ZoneInfo
import yaml

# =============================================================================
# EARLY CONFIG LOADING (before torch import)
# =============================================================================

def _parse_args_early():
    """Parse args early to get config path and flags before torch import."""
    import sys
    config_path = "configs/dpo.yaml"
    task = None
    use_chat_template = False
    output_dir = None

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            i += 2
        elif arg == "--task" and i + 1 < len(sys.argv):
            task = sys.argv[i + 1]
            i += 2
        elif arg == "--use_chat_template":
            use_chat_template = True
            i += 1
        elif arg == "--output_dir" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    return {
        "config_path": config_path,
        "task": task,
        "use_chat_template": use_chat_template,
        "output_dir": output_dir,
    }

def _load_config_once(config_path: str) -> dict:
    """Load config file and return a deep copy to ensure immutability."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return copy.deepcopy(cfg)

def _configure_cuda_devices(cfg: dict):
    """Set CUDA_VISIBLE_DEVICES based on config before torch is imported."""
    device_cfg = cfg.get("model", {}).get("device", "auto")

    if isinstance(device_cfg, list):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_cfg)
    elif isinstance(device_cfg, str) and device_cfg.startswith("cuda:"):
        gpu_id = device_cfg.split(":")[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

def _generate_output_dir(task: str, use_chat_template: bool) -> str:
    """Generate a descriptive output directory name."""
    training_format = "chat" if use_chat_template else "nochat"
    timestamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S")
    dir_name = f"{training_format}_dpo_{task}_{timestamp}"
    return os.path.join("results", task, "models", dir_name)

_EARLY_ARGS = _parse_args_early()
_CONFIG_PATH = _EARLY_ARGS["config_path"]
_FROZEN_CONFIG = _load_config_once(_CONFIG_PATH)
_configure_cuda_devices(_FROZEN_CONFIG)

# Now safe to import torch
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

from tasks import get_task
from src.data import format_with_chat_template, format_without_chat_template
from src.config_utils import validate_config


def get_attn_implementation() -> str | None:
    """Return 'flash_attention_2' if available, else None (use default)."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return None


def _save_config_snapshot(cfg: dict, config_path: str, output_dir: str, args_dict: dict):
    """Save a copy of the config and CLI args used for this run."""
    os.makedirs(output_dir, exist_ok=True)

    config_snapshot_path = os.path.join(output_dir, "config_snapshot.yaml")
    with open(config_snapshot_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    run_info = {
        "original_config_path": config_path,
        "cli_args": args_dict,
        "timestamp": datetime.now().isoformat(),
    }
    run_info_path = os.path.join(output_dir, "run_info.yaml")
    with open(run_info_path, "w") as f:
        yaml.dump(run_info, f, default_flow_style=False, sort_keys=False)

    print(f"Config snapshot saved to: {config_snapshot_path}")


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    gen_cfg: dict,
    num_return_sequences: int = 1,
) -> list[list[str]]:
    """Generate completions for a batch of prompts using left-padding.

    Uses num_return_sequences to generate multiple completions per prompt
    efficiently (shared KV cache for the prompt encoding).

    Returns a list of lists: completions[i] is the list of completions for prompts[i].
    """
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(model.device)

        prompt_lengths = (inputs["attention_mask"]).sum(dim=1).tolist()

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=gen_cfg.get("max_new_tokens", 512),
                temperature=gen_cfg.get("temperature", 0.7),
                do_sample=gen_cfg.get("do_sample", True),
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.pad_token_id,
            )

        # output_ids shape: [batch_size * num_return_sequences, seq_len]
        # Order: prompt0_seq0, prompt0_seq1, ..., prompt0_seqN, prompt1_seq0, ...
        # attention_mask was NOT expanded, so we index by i // num_return_sequences
        completions: list[list[str]] = [[] for _ in prompts]
        for seq_idx in range(len(output_ids)):
            prompt_idx = seq_idx // num_return_sequences
            prompt_len = prompt_lengths[prompt_idx]
            pad_len = (inputs["attention_mask"][prompt_idx] == 0).sum().item()
            start_idx = pad_len + prompt_len
            generated_ids = output_ids[seq_idx][start_idx:]
            completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
            completions[prompt_idx].append(completion)

        return completions
    finally:
        tokenizer.padding_side = original_padding_side


def generate_pairs(
    model,
    tokenizer,
    task,
    train_samples,
    use_chat_template: bool,
    extra_kwargs: dict,
    dpo_cfg: dict,
    output_dir: str,
    batch_size: int = 8,
) -> dict:
    """Generate preference pairs via batch-level rejection sampling.

    Processes examples in batches of `batch_size`. For each batch:
      1. Generate `num_generations` completions per example.
      2. Score all completions, pool correct/incorrect across the batch.
      3. If total correct < min_correct or total incorrect < min_incorrect,
         skip the entire batch.
      4. Otherwise, enumerate all same-prompt (chosen, rejected) candidate pairs
         from examples that have both correct and incorrect completions, and
         sample up to `num_pairs_per_batch` pairs (with replacement if needed).

    Returns dict with lists of prompts, chosen, rejected.
    """
    num_gen = dpo_cfg["num_generations"]
    min_correct = dpo_cfg.get("min_correct", 8)
    min_incorrect = dpo_cfg.get("min_incorrect", 8)
    num_pairs_per_batch = dpo_cfg.get("num_pairs_per_batch", 32)
    gen_cfg = {
        "max_new_tokens": dpo_cfg.get("max_new_tokens", 1024),
        "temperature": dpo_cfg.get("generation_temperature", 0.7),
        "do_sample": True,
    }

    pairs_path = os.path.join(output_dir, "pairs.json")
    if os.path.exists(pairs_path):
        print(f"Loading cached pairs from {pairs_path}")
        with open(pairs_path) as f:
            cached = json.load(f)
        print(f"  Loaded {len(cached['prompt'])} pairs")
        return cached

    all_prompts = []
    all_gold_values = []
    for s in train_samples:
        if use_chat_template:
            prompt = format_with_chat_template(tokenizer, s.question, None, extra_kwargs)
        else:
            prompt = format_without_chat_template(s.question, None)
        all_prompts.append(prompt)
        all_gold_values.append(s.answer_value if s.answer_value else s.answer)

    total_completions = len(all_prompts) * num_gen
    num_batches = (len(all_prompts) + batch_size - 1) // batch_size
    print(f"Generating {total_completions} completions "
          f"({len(all_prompts)} prompts x {num_gen} generations, "
          f"batch_size={batch_size})...")
    print(f"Batch filtering: min_correct={min_correct}, min_incorrect={min_incorrect}, "
          f"num_pairs_per_batch={num_pairs_per_batch}")

    result_prompts = []
    result_chosen = []
    result_rejected = []

    batches_kept = 0
    batches_skipped = 0
    total_correct_all = 0
    total_incorrect_all = 0

    for batch_idx in tqdm(range(num_batches), desc="Generating pairs"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(all_prompts))
        batch_prompts = all_prompts[start:end]
        batch_gold_values = all_gold_values[start:end]

        batch_completions = generate_batch(
            model, tokenizer, batch_prompts, gen_cfg,
            num_return_sequences=num_gen,
        )

        # Score completions and partition into correct/incorrect per example
        batch_total_correct = 0
        batch_total_incorrect = 0
        per_example: list[dict] = []

        for prompt, completions, gold_value in zip(
            batch_prompts, batch_completions, batch_gold_values
        ):
            correct = []
            incorrect = []
            for comp in completions:
                extracted = task.extract_answer(comp)
                if task.check_answer(extracted, gold_value):
                    correct.append(comp)
                else:
                    incorrect.append(comp)
            batch_total_correct += len(correct)
            batch_total_incorrect += len(incorrect)
            per_example.append({
                "prompt": prompt,
                "correct": correct,
                "incorrect": incorrect,
            })

        total_correct_all += batch_total_correct
        total_incorrect_all += batch_total_incorrect

        if batch_total_correct < min_correct or batch_total_incorrect < min_incorrect:
            batches_skipped += 1
            continue

        # Build candidate pairs: each pair shares the same prompt
        candidate_pairs = []
        for ex in per_example:
            if ex["correct"] and ex["incorrect"]:
                for c in ex["correct"]:
                    for i in ex["incorrect"]:
                        candidate_pairs.append((ex["prompt"], c, i))

        if not candidate_pairs:
            batches_skipped += 1
            continue

        if len(candidate_pairs) >= num_pairs_per_batch:
            sampled = random.sample(candidate_pairs, num_pairs_per_batch)
        else:
            sampled = random.choices(candidate_pairs, k=num_pairs_per_batch)

        for prompt, chosen, rejected in sampled:
            result_prompts.append(prompt)
            result_chosen.append(chosen)
            result_rejected.append(rejected)

        batches_kept += 1

    total_pairs = len(result_prompts)
    print(f"\nPair construction stats:")
    print(f"  Total batches:        {num_batches}")
    print(f"  Batches kept:         {batches_kept}")
    print(f"  Batches skipped:      {batches_skipped}")
    print(f"  Total correct comps:  {total_correct_all}")
    print(f"  Total incorrect comps:{total_incorrect_all}")
    print(f"  Total pairs:          {total_pairs}")

    pairs_data = {
        "prompt": result_prompts,
        "chosen": result_chosen,
        "rejected": result_rejected,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(pairs_path, "w") as f:
        json.dump(pairs_data, f, indent=2)
    print(f"Pairs saved to {pairs_path}")

    stats_path = os.path.join(output_dir, "pair_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "total_samples": len(all_prompts),
            "total_batches": num_batches,
            "batches_kept": batches_kept,
            "batches_skipped": batches_skipped,
            "total_correct_completions": total_correct_all,
            "total_incorrect_completions": total_incorrect_all,
            "total_pairs": total_pairs,
            "num_generations": num_gen,
            "min_correct": min_correct,
            "min_incorrect": min_incorrect,
            "num_pairs_per_batch": num_pairs_per_batch,
        }, f, indent=2)

    return pairs_data


def main():
    parser = argparse.ArgumentParser(description="DPO fine-tuning")
    parser.add_argument("--config", type=str, default="configs/dpo.yaml")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g. gsm8k)")
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Format prompts with the model's chat template")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save the fine-tuned model (auto-generated if not provided)")
    args = parser.parse_args()

    cfg = _FROZEN_CONFIG
    validate_config(cfg, "dpo")
    model_name = cfg["model"]["name"]
    extra_kwargs = cfg["model"].get("extra_chat_template_kwargs", {})
    train_cfg = cfg["training"]
    dpo_cfg = cfg["dpo"]

    if args.output_dir is None:
        args.output_dir = _generate_output_dir(args.task, args.use_chat_template)
        print(f"Auto-generated output_dir: {args.output_dir}")

    _save_config_snapshot(
        cfg=cfg,
        config_path=_CONFIG_PATH,
        output_dir=args.output_dir,
        args_dict={
            "task": args.task,
            "use_chat_template": args.use_chat_template,
            "output_dir": args.output_dir,
        }
    )

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = get_attn_implementation()

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if train_cfg.get("bf16") else torch.float32,
        "device_map": "auto",
    }
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    print(f"Loading task: {args.task}")
    task = get_task(args.task)
    train_samples = task.load_train()
    print(f"Training samples: {len(train_samples)}")

    # Phase 1: Generate preference pairs (or load cached)
    pairs = generate_pairs(
        model=model,
        tokenizer=tokenizer,
        task=task,
        train_samples=train_samples,
        use_chat_template=args.use_chat_template,
        extra_kwargs=extra_kwargs,
        dpo_cfg=dpo_cfg,
        output_dir=args.output_dir,
        batch_size=dpo_cfg.get("generation_batch_size", 8),
    )

    dataset = Dataset.from_dict(pairs)

    # Phase 2: DPO training
    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        bf16=train_cfg.get("bf16", False),
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        seed=train_cfg["seed"],
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        report_to="none",
        # DPO-specific
        beta=dpo_cfg["beta"],
        max_length=dpo_cfg["max_length"],
        max_prompt_length=dpo_cfg["max_prompt_length"],
        loss_type=dpo_cfg["loss_type"],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"Starting DPO training → {args.output_dir}")
    print(f"  chat_template={args.use_chat_template}")
    print(f"  pairs={len(dataset)}")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()
