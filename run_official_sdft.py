#!/usr/bin/env python3
"""
Run the official SDFT (Self-Distillation Fine-Tuning) implementation on GSM8K/MATH tasks.

Uses the paper authors' DistilTrainer code from:
    official_self_distillation/Self-Distillation/

All hyperparameters match the official main.py exactly (with README learning_rate=5e-5),
except model_name (3B instead of 7B) and num_train_epochs (1 instead of 2).

Usage:
    uv run python run_official_sdft.py --task gsm8k --output_dir results/gsm8k/models/official_sdft
    uv run python run_official_sdft.py --task math  --output_dir results/math/models/official_sdft
"""

import sys
import os
import argparse
import torch
from string import Template
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the official repo to path so we can import DistilTrainer/DistilConfig
OFFICIAL_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "official_self_distillation", "Self-Distillation")
sys.path.insert(0, OFFICIAL_REPO_DIR)

from distil_trainer import DistilTrainer
from distil_config import DistilConfig

# Import task loaders from the existing project
from tasks.gsm8k import GSM8KTask
from tasks.math import MATHTask


# Matches the official repo's teacher prompt template exactly.
# See official_self_distillation/Self-Distillation/main.py lines 56-63
TEACHER_TEMPLATE = Template(
    "$question\n\n"
    "This is an example for a response to the question:\n"
    "$answer\n\n"
    "Now answer with a response of your own, including the thinking process."
)


def load_task_dataset(task_name: str, seed: int = 42, gold_data_path: str = None) -> Dataset:
    """Load a task dataset and convert to the format expected by DistilTrainer.

    DistilTrainer expects each row to have:
        - "prompt": list of messages (conversational format)
        - "teacher_prompt": list of messages with demonstration context
    """
    if task_name == "gsm8k":
        task = GSM8KTask()
    elif task_name == "math":
        task = MATHTask()
    else:
        raise ValueError(f"Unknown task: {task_name}")

    if gold_data_path:
        import json
        from tasks.base import Sample
        print(f"Loading gold data from: {gold_data_path}")
        with open(gold_data_path) as f:
            gold = json.load(f)
        samples = [
            Sample(
                question=entry["question"],
                answer=entry["gold_answer"],
                answer_value=entry.get("gold_value"),
            )
            for entry in gold["data"]
        ]
        print(f"Gold data: {gold.get('selected_questions', len(samples))} questions")
    else:
        samples = task.load_train()

    records = []
    for sample in samples:
        # Student prompt: just the question
        prompt = [{"role": "user", "content": sample.question}]

        # Teacher prompt: question + gold demonstration + instruction
        teacher_content = TEACHER_TEMPLATE.substitute(
            question=sample.question,
            answer=sample.answer,
        )
        teacher_prompt = [{"role": "user", "content": teacher_content}]

        records.append({
            "prompt": prompt,
            "teacher_prompt": teacher_prompt,
        })

    dataset = Dataset.from_list(records)
    dataset = dataset.shuffle(seed=seed)
    print(f"Loaded {len(dataset)} training examples for {task_name}")
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run official SDFT on GSM8K/MATH")
    parser.add_argument("--task", type=str, required=True, choices=["gsm8k", "math"],
                        help="Task to train on")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the trained model")
    # Model config — defaults differ from official (3B vs 7B, 1 epoch vs 2)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of training epochs")
    # Training hyperparams — all defaults match the official README examples
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate (README default: 5e-5)")
    parser.add_argument("--num_prompts_per_batch", type=int, default=32,
                        help="Effective batch size = per_device_bs(1) * this value")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01,
                        help="EMA teacher update rate (official default: 0.01)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Compatibility arguments for run_experiments.py
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Included for compatibility with run_experiments.py")
    parser.add_argument("--gold_data_path", type=str, default=None,
                        help="Included for compatibility with run_experiments.py")
                        
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = load_task_dataset(args.task, seed=args.seed, gold_data_path=args.gold_data_path)

    # DistilConfig: all values match official main.py lines 100-130 exactly,
    # except model_name and num_train_epochs which are user-configurable.
    config = DistilConfig(
        seed=args.seed,
        # --- vLLM generation settings (main.py L102-106) ---
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.3,
        vllm_enable_sleep_mode=True,
        # --- Optimizer settings (main.py L107-109) ---
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        # --- Precision and logging (main.py L110-112) ---
        logging_steps=1,
        bf16=True,
        fp16=False,
        # --- Batch size (main.py L113-114) ---
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.num_prompts_per_batch,
        # --- Sequence lengths (main.py L115-116) ---
        max_prompt_length=1024,
        max_completion_length=1024,
        # --- Training loop (main.py L117-121) ---
        num_train_epochs=args.num_train_epochs,
        num_iterations=1,
        num_generations=1,
        save_steps=100,
        max_grad_norm=1,
        # --- Logging and output (main.py L122-124) ---
        report_to="none",
        output_dir=args.output_dir,
        log_completions=False,
        # --- EMA teacher sync (main.py L125-127) ---
        sync_ref_model=True,
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        # --- SDFT-specific (main.py L128-129) ---
        vllm_importance_sampling_correction=True,
        num_loss_tokens_to_skip=3,
    )

    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save the final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")


if __name__ == "__main__":
    main()
