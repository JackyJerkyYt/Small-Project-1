"""SFT training script."""

import argparse
import os
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

from tasks import get_task
from src.data import SFTDataset, CausalLMDataCollator


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


def main():
    parser = argparse.ArgumentParser(description="SFT fine-tuning")
    parser.add_argument("--config", type=str, default="configs/sft.yaml")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g. gsm8k)")
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Format data with the model's chat template")
    parser.add_argument("--mask_question", action="store_true",
                        help="Mask the question so loss only comes from the answer")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the fine-tuned model")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["model"]["name"]
    extra_kwargs = cfg["model"].get("extra_chat_template_kwargs", {})
    device_cfg = cfg["model"].get("device", "auto")
    train_cfg = cfg["training"]

    if args.mask_question and args.use_chat_template:
        print("WARNING: --mask_question with --use_chat_template is unusual. "
              "The question masking is designed for the no-chat-template setting.")

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

    dataset = SFTDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        use_chat_template=args.use_chat_template,
        mask_question=args.mask_question,
        max_seq_length=train_cfg.get("max_seq_length", 1024),
        extra_chat_template_kwargs=extra_kwargs if args.use_chat_template else None,
    )

    training_args = TrainingArguments(
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
        remove_unused_columns=False,
        report_to="none",
    )

    collator = CausalLMDataCollator(
        tokenizer=tokenizer,
        padding=True,
        max_length=train_cfg.get("max_seq_length", 1024),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print(f"Starting SFT training → {args.output_dir}")
    print(f"  chat_template={args.use_chat_template}, mask_question={args.mask_question}")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()
