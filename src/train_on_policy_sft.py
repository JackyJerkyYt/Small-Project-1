"""On-policy SFT: generate rollouts with the current model, train on correct ones.

At each training step:
  1. Sample a batch of questions
  2. Generate N rollouts per question using the current model
  3. Validate each rollout against the gold answer
  4. Pick the first correct rollout per question (skip if none correct)
  5. Train on the correct rollouts with SFT loss

Usage:
    python -m src.train_on_policy_sft --task gsm8k --use_chat_template
    python -m src.train_on_policy_sft --task gsm8k --use_chat_template \\
        --gold_data_path results/gold_data/chat_template/gold_data.json
"""

import argparse
import copy
import json
import math
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
    config_path = "configs/on_policy_sft.yaml"
    task = None
    use_chat_template = False
    mask_question = True
    output_dir = None
    gold_data_path = None

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
        elif arg == "--no_mask_question":
            mask_question = False
            i += 1
        elif arg == "--output_dir" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        elif arg == "--gold_data_path" and i + 1 < len(sys.argv):
            gold_data_path = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    return {
        "config_path": config_path,
        "task": task,
        "use_chat_template": use_chat_template,
        "mask_question": mask_question,
        "output_dir": output_dir,
        "gold_data_path": gold_data_path,
    }


def _load_config_once(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return copy.deepcopy(cfg)


def _configure_cuda_devices(cfg: dict):
    device_cfg = cfg.get("model", {}).get("device", "auto")
    if isinstance(device_cfg, list):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_cfg)
    elif isinstance(device_cfg, str) and device_cfg.startswith("cuda:"):
        gpu_id = device_cfg.split(":")[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


def _generate_output_dir(task: str, use_chat_template: bool, mask_question: bool) -> str:
    training_format = "chat" if use_chat_template else "nochat"
    masking = "maskq" if mask_question else "fullloss"
    timestamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S")
    dir_name = f"{training_format}_onpolicysft_{masking}_{task}_{timestamp}"
    return os.path.join("results", task, "models", dir_name)


_EARLY_ARGS = _parse_args_early()
_CONFIG_PATH = _EARLY_ARGS["config_path"]
_FROZEN_CONFIG = _load_config_once(_CONFIG_PATH)
_configure_cuda_devices(_FROZEN_CONFIG)

# Now safe to import torch
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from tqdm import tqdm

from tasks import get_task
from src.data import format_with_chat_template, format_without_chat_template, CausalLMDataCollator
from src.config_utils import validate_config


def get_attn_implementation() -> str | None:
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return None


def _save_config_snapshot(cfg: dict, config_path: str, output_dir: str, args_dict: dict):
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


def tokenize_sft_pair(tokenizer, question, answer, use_chat_template,
                      mask_question, max_seq_length, extra_kwargs):
    """Tokenize a single (question, model-generated answer) pair for SFT."""
    if use_chat_template:
        full_text = format_with_chat_template(tokenizer, question, answer, extra_kwargs)
        prompt_text = format_with_chat_template(tokenizer, question, None, extra_kwargs)
    else:
        full_text = format_without_chat_template(question, answer, eos_token=tokenizer.eos_token)
        prompt_text = format_without_chat_template(question, None)

    full_ids = tokenizer(
        full_text, truncation=True, max_length=max_seq_length, add_special_tokens=False
    )
    input_ids = full_ids["input_ids"]
    attention_mask = full_ids["attention_mask"]

    if mask_question:
        prompt_ids = tokenizer(
            prompt_text, truncation=True, max_length=max_seq_length, add_special_tokens=False
        )["input_ids"]
        prompt_len = len(prompt_ids)

        if prompt_ids != input_ids[:prompt_len]:
            for k in range(min(len(prompt_ids), len(input_ids)), 0, -1):
                if prompt_ids[:k] == input_ids[:k]:
                    prompt_len = k
                    break
            else:
                prompt_len = 0

        labels = [-100] * prompt_len + input_ids[prompt_len:]
    else:
        labels = list(input_ids)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    parser = argparse.ArgumentParser(description="On-policy SFT fine-tuning")
    parser.add_argument("--config", type=str, default="configs/on_policy_sft.yaml")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g. gsm8k)")
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Format data with the model's chat template")
    parser.add_argument("--no_mask_question", action="store_true",
                        help="Disable question masking (apply loss on both question and answer)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save the fine-tuned model (auto-generated if not provided)")
    parser.add_argument("--gold_data_path", type=str, default=None,
                        help="Path to gold_data.json; train on selected gold data instead of full set")
    args = parser.parse_args()
    args.mask_question = not args.no_mask_question

    cfg = _FROZEN_CONFIG
    validate_config(cfg, "on_policy_sft")
    model_name = cfg["model"]["name"]
    extra_kwargs = cfg["model"].get("extra_chat_template_kwargs", {})
    train_cfg = cfg["training"]
    gen_cfg = cfg["generation"]

    if args.output_dir is None:
        args.output_dir = _generate_output_dir(args.task, args.use_chat_template, args.mask_question)
        print(f"Auto-generated output_dir: {args.output_dir}")

    _save_config_snapshot(
        cfg=cfg,
        config_path=_CONFIG_PATH,
        output_dir=args.output_dir,
        args_dict={
            "task": args.task,
            "use_chat_template": args.use_chat_template,
            "mask_question": args.mask_question,
            "output_dir": args.output_dir,
            "gold_data_path": args.gold_data_path,
        },
    )

    # ---- Load model & tokenizer ----
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

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    # ---- Load data ----
    print(f"Loading task: {args.task}")
    task = get_task(args.task)

    if args.gold_data_path:
        from tasks.base import Sample
        print(f"Loading gold data from: {args.gold_data_path}")
        with open(args.gold_data_path) as f:
            gold = json.load(f)
        train_samples = [
            Sample(
                question=entry["question"],
                answer=entry["gold_answer"],
                answer_value=entry.get("gold_value"),
            )
            for entry in gold["data"]
        ]
        print(f"Gold data: {gold['selected_questions']} questions "
              f"(from {gold['total_questions']} total, {gold['eligible_questions']} eligible)")
    else:
        train_samples = task.load_train()

    print(f"Training samples: {len(train_samples)}")

    # ---- Training setup ----
    batch_size = train_cfg["per_device_train_batch_size"]
    grad_accum_steps = train_cfg["gradient_accumulation_steps"]
    num_epochs = train_cfg["num_epochs"]
    max_seq_length = train_cfg.get("max_seq_length", 1024)
    num_rollouts = gen_cfg["num_rollouts"]
    gen_max_new_tokens = gen_cfg["max_new_tokens"]
    gen_temperature = gen_cfg["temperature"]

    total_batches_per_epoch = math.ceil(len(train_samples) / batch_size)
    total_batches = total_batches_per_epoch * num_epochs
    total_optimizer_steps = math.ceil(total_batches / grad_accum_steps)

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    num_warmup_steps = int(train_cfg["warmup_ratio"] * total_optimizer_steps)
    scheduler = get_scheduler(
        train_cfg["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    collator = CausalLMDataCollator(
        tokenizer=tokenizer, padding=True, max_length=max_seq_length,
    )

    use_grad_ckpt = train_cfg.get("gradient_checkpointing", False)
    if use_grad_ckpt:
        model.gradient_checkpointing_enable()

    seed = train_cfg.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)

    print(f"Starting on-policy SFT → {args.output_dir}")
    print(f"  chat_template={args.use_chat_template}, mask_question={args.mask_question}")
    print(f"  batch_size={batch_size}, grad_accum={grad_accum_steps}, "
          f"num_rollouts={num_rollouts}")
    print(f"  total_batches={total_batches}, optimizer_steps={total_optimizer_steps}")

    # ---- Log file (continuous JSONL inside output_dir) ----
    train_log_path = os.path.join(args.output_dir, "train_log.jsonl")
    train_log_file = open(train_log_path, "a")
    print(f"  logging to: {train_log_path}")

    # ---- Training loop ----
    global_step = 0
    accum_step = 0
    log_loss = 0.0
    log_hits = 0
    log_total = 0
    log_count = 0

    for epoch in range(num_epochs):
        indices = list(range(len(train_samples)))
        random.shuffle(indices)

        pbar = tqdm(
            range(0, len(indices), batch_size),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
        )

        for batch_start in pbar:
            batch_indices = indices[batch_start: batch_start + batch_size]
            batch_samples = [train_samples[i] for i in batch_indices]

            # --- 1. Generate rollouts (eval mode, no grad) ---
            model.eval()
            if use_grad_ckpt:
                model.config.use_cache = True

            prompts = []
            for s in batch_samples:
                if args.use_chat_template:
                    p = format_with_chat_template(tokenizer, s.question, None, extra_kwargs)
                else:
                    p = format_without_chat_template(s.question, None)
                prompts.append(p)

            rollouts = generate_rollouts(
                model, tokenizer, prompts,
                num_rollouts, gen_max_new_tokens, gen_temperature,
            )
            torch.cuda.empty_cache()

            # --- 2. Find first correct rollout per question ---
            sft_pairs = []
            for sample, completions in zip(batch_samples, rollouts):
                gold = sample.answer_value if sample.answer_value else sample.answer
                for completion in completions:
                    extracted = task.extract_answer(completion)
                    if task.check_answer(extracted, gold):
                        sft_pairs.append({"question": sample.question, "answer": completion})
                        break

            log_hits += len(sft_pairs)
            log_total += len(batch_samples)

            # --- 3. SFT on correct rollouts ---
            model.train()
            if use_grad_ckpt:
                model.config.use_cache = False

            if sft_pairs:
                tokenized = [
                    tokenize_sft_pair(
                        tokenizer, p["question"], p["answer"],
                        args.use_chat_template, args.mask_question,
                        max_seq_length, extra_kwargs,
                    )
                    for p in sft_pairs
                ]
                batch_data = collator(tokenized)
                batch_data = {k: v.to(model.device) for k, v in batch_data.items()}

                outputs = model(**batch_data)
                loss = outputs.loss / grad_accum_steps
                loss.backward()

                log_loss += outputs.loss.item()
                log_count += 1

            # --- 4. Optimizer step at accumulation boundary ---
            accum_step += 1
            is_accum_boundary = (accum_step % grad_accum_steps == 0)
            is_last_batch = (batch_start + batch_size >= len(indices))

            if is_accum_boundary or is_last_batch:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                logging_steps = train_cfg.get("logging_steps", 10)
                if global_step % logging_steps == 0 or is_last_batch:
                    avg_loss = log_loss / log_count if log_count > 0 else 0.0
                    hit_rate = log_hits / log_total if log_total > 0 else 0.0
                    lr = scheduler.get_last_lr()[0]
                    msg = (
                        f"  step {global_step}/{total_optimizer_steps} | "
                        f"loss={avg_loss:.4f} | hit_rate={hit_rate:.2%} | "
                        f"lr={lr:.2e}"
                    )
                    print(msg)
                    log_entry = {
                        "step": global_step,
                        "total_steps": total_optimizer_steps,
                        "epoch": epoch + 1,
                        "loss": avg_loss,
                        "hit_rate": hit_rate,
                        "lr": lr,
                        "hits": log_hits,
                        "total_questions": log_total,
                    }
                    train_log_file.write(json.dumps(log_entry) + "\n")
                    train_log_file.flush()
                    log_loss = 0.0
                    log_hits = 0
                    log_total = 0
                    log_count = 0

            pbar.set_postfix({
                "hits": f"{len(sft_pairs)}/{len(batch_samples)}",
                "step": global_step,
            })

        # Save checkpoint at end of epoch
        if train_cfg.get("save_strategy", "epoch") == "epoch":
            ckpt_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch + 1}")
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Save final model
    train_log_file.close()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
