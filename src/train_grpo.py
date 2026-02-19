"""GRPO training script with binary reward (correct=1, wrong=0)."""

import argparse
import yaml
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from tasks import get_task
from src.data import format_with_chat_template, format_without_chat_template


def get_attn_implementation() -> str | None:
    """Return 'flash_attention_2' if available, else None (use default)."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return None


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_reward_fn(task, use_chat_template: bool, tokenizer, extra_kwargs: dict):
    """Return a reward function compatible with GRPOTrainer.
    Each completion is scored 1 if correct, 0 if wrong.
    
    Args:
        task: Task instance with extract_answer and check_answer methods
        use_chat_template: Whether prompts use chat template (for reference)
        tokenizer: Tokenizer (for potential decoding needs)
        extra_kwargs: Extra kwargs (for potential future use)
    
    Returns:
        A reward function that takes (completions, prompts, answer, ...) and returns list of floats.
    """

    def reward_fn(completions, prompts, answer, **kwargs):
        """
        Args:
            completions: List of generated completion strings from the model
            prompts: List of prompt strings
            answer: List of gold answers (from dataset column "answer")
            **kwargs: Additional columns/metadata
        
        Returns:
            List of reward floats (1.0 for correct, 0.0 for wrong)
        """
        rewards = []
        for i, completion in enumerate(completions):
            # completions are strings (model-generated text)
            text = completion
            gold = answer[i] if answer is not None and i < len(answer) else None

            extracted = task.extract_answer(text)
            if gold is not None and task.check_answer(extracted, gold):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    return reward_fn


def main():
    parser = argparse.ArgumentParser(description="GRPO fine-tuning")
    parser.add_argument("--config", type=str, default="configs/grpo.yaml")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g. gsm8k)")
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Format prompts with the model's chat template")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the fine-tuned model")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["model"]["name"]
    extra_kwargs = cfg["model"].get("extra_chat_template_kwargs", {})
    device_cfg = cfg["model"].get("device", "auto")
    train_cfg = cfg["training"]
    grpo_cfg = cfg["grpo"]

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = get_attn_implementation()
    
    # Handle device configuration
    if device_cfg == "auto" or device_cfg is None:
        device_map = "auto"
    elif isinstance(device_cfg, str) and device_cfg.startswith("cuda:"):
        device_map = {"": device_cfg}  # put whole model on specified GPU
    elif isinstance(device_cfg, list):
        device_map = "auto"  # use accelerate's auto with visible devices
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_cfg)
    else:
        device_map = device_cfg
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if train_cfg.get("bf16") else torch.float32,
        "device_map": device_map,
    }
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    print(f"Loading task: {args.task}")
    task = get_task(args.task)
    train_samples = task.load_train()
    print(f"Training samples: {len(train_samples)}")

    prompts = []
    answers = []
    for s in train_samples:
        if args.use_chat_template:
            prompt = format_with_chat_template(tokenizer, s.question, None, extra_kwargs)
        else:
            prompt = format_without_chat_template(s.question, None)
        prompts.append(prompt)
        answers.append(s.answer_value if s.answer_value else s.answer)

    dataset = Dataset.from_dict({"prompt": prompts, "answer": answers})

    reward_fn = make_reward_fn(task, args.use_chat_template, tokenizer, extra_kwargs)

    training_args = GRPOConfig(
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
        # GRPO-specific
        num_generations=grpo_cfg["num_generations"],
        max_completion_length=grpo_cfg["max_new_tokens"],
        max_prompt_length=grpo_cfg.get("max_prompt_length", 512),
        beta=grpo_cfg.get("beta", 0.1),
        temperature=grpo_cfg.get("temperature", 0.7),
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    print(f"Starting GRPO training → {args.output_dir}")
    print(f"  chat_template={args.use_chat_template}")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()
