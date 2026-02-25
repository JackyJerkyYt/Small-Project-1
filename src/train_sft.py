"""SFT training script."""

import argparse
import copy
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import yaml

# =============================================================================
# EARLY CONFIG LOADING (before torch import)
# =============================================================================
# We load the entire config into memory ONCE at startup to prevent race conditions.
# If the user modifies the config file after starting training, it won't affect this run.

def _parse_args_early():
    """Parse args early to get config path and flags before torch import."""
    import sys
    config_path = "configs/sft.yaml"
    task = None
    use_chat_template = False
    mask_question = False
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
        elif arg == "--mask_question":
            mask_question = True
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
        "mask_question": mask_question,
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

def _generate_output_dir(task: str, use_chat_template: bool, mask_question: bool) -> str:
    """Generate a descriptive output directory name."""
    training_format = "chat" if use_chat_template else "nochat"
    training_method = "sft"
    
    masking = "maskq" if mask_question else "fullloss"
    
    timestamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S")
    
    dir_name = f"{training_format}_{training_method}_{masking}_{task}_{timestamp}"
    return os.path.join("results", task, "models", dir_name)

# Parse args and load config ONCE at module load time
_EARLY_ARGS = _parse_args_early()
_CONFIG_PATH = _EARLY_ARGS["config_path"]
_FROZEN_CONFIG = _load_config_once(_CONFIG_PATH)
_configure_cuda_devices(_FROZEN_CONFIG)

# Now safe to import torch
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

from tasks import get_task
from src.data import SFTDataset, CausalLMDataCollator
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
    
    # Save the frozen config
    config_snapshot_path = os.path.join(output_dir, "config_snapshot.yaml")
    with open(config_snapshot_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    
    # Also save CLI args for full reproducibility
    run_info = {
        "original_config_path": config_path,
        "cli_args": args_dict,
        "timestamp": datetime.now().isoformat(),
    }
    run_info_path = os.path.join(output_dir, "run_info.yaml")
    with open(run_info_path, "w") as f:
        yaml.dump(run_info, f, default_flow_style=False, sort_keys=False)
    
    print(f"Config snapshot saved to: {config_snapshot_path}")


def main():
    parser = argparse.ArgumentParser(description="SFT fine-tuning")
    parser.add_argument("--config", type=str, default="configs/sft.yaml")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g. gsm8k)")
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Format data with the model's chat template")
    parser.add_argument("--mask_question", action="store_true",
                        help="Mask the question so loss only comes from the answer")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save the fine-tuned model (auto-generated if not provided)")
    args = parser.parse_args()

    # Use the frozen config that was loaded at module startup (prevents race conditions)
    cfg = _FROZEN_CONFIG
    validate_config(cfg, "sft")
    model_name = cfg["model"]["name"]
    extra_kwargs = cfg["model"].get("extra_chat_template_kwargs", {})
    train_cfg = cfg["training"]

    # Generate output_dir if not provided
    if args.output_dir is None:
        args.output_dir = _generate_output_dir(args.task, args.use_chat_template, args.mask_question)
        print(f"Auto-generated output_dir: {args.output_dir}")

    # Save config snapshot immediately (before any training that might take hours)
    _save_config_snapshot(
        cfg=cfg,
        config_path=_CONFIG_PATH,
        output_dir=args.output_dir,
        args_dict={
            "task": args.task,
            "use_chat_template": args.use_chat_template,
            "mask_question": args.mask_question,
            "output_dir": args.output_dir,
        }
    )

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = get_attn_implementation()
    
    # Device configuration is handled at script startup via CUDA_VISIBLE_DEVICES
    # Here we just use device_map="auto" to distribute across visible GPUs
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if train_cfg.get("bf16") else torch.float32,
        "device_map": "auto",
    }
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Pre-align model.config with the tokenizer before Trainer.train() calls
    # align_special_tokens().  Without this, transformers detects mismatches at
    # training time and emits the PAD/BOS/EOS warning about updated tokens.
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

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
