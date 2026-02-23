"""Lightweight config validation to catch typos and missing keys."""


KNOWN_KEYS = {
    "sft": {
        "model": {"name", "extra_chat_template_kwargs", "device"},
        "training": {
            "num_epochs", "per_device_train_batch_size", "gradient_accumulation_steps",
            "learning_rate", "weight_decay", "warmup_ratio", "lr_scheduler_type",
            "max_seq_length", "bf16", "logging_steps", "save_strategy", "seed",
            "gradient_checkpointing",
        },
    },
    "grpo": {
        "model": {"name", "extra_chat_template_kwargs", "device"},
        "training": {
            "num_epochs", "per_device_train_batch_size", "gradient_accumulation_steps",
            "learning_rate", "weight_decay", "warmup_ratio", "lr_scheduler_type",
            "bf16", "logging_steps", "save_strategy", "seed",
            "gradient_checkpointing",
        },
        "grpo": {
            "num_generations", "max_new_tokens", "max_prompt_length",
            "temperature", "beta",
        },
    },
    "eval": {
        "generation": {"max_new_tokens", "temperature", "do_sample"},
        "chat_template": {"enable_thinking"},
        "eval": {"per_device_batch_size", "max_samples", "bf16", "device"},
    },
}


def validate_config(cfg: dict, config_type: str):
    """Warn on unknown keys in a loaded config (likely typos).

    Only flags keys that are NOT in the known set for each section.
    Does not warn on missing keys since many are optional with defaults.

    Args:
        cfg: The parsed YAML config dict.
        config_type: One of "sft", "grpo", "eval".
    """
    known_sections = KNOWN_KEYS.get(config_type)
    if known_sections is None:
        print(f"WARNING: no schema defined for config_type='{config_type}', skipping validation")
        return

    for section, known_keys in known_sections.items():
        if section not in cfg:
            continue
        actual_keys = set(cfg[section].keys())
        unknown = actual_keys - known_keys
        if unknown:
            print(f"WARNING: unknown keys in config['{section}']: {unknown} (possible typo?)")
