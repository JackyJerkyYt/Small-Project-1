"""Self-Distillation Fine-Tuning (SDFT) training script.

Based on: "Self-Distillation Enables Continual Learning" (arXiv:2601.19897)

Algorithm:
  For each training step:
    1. Sample a batch of questions
    2. Build student prompts (question only, with/without chat template)
    3. Generate one on-policy rollout per question from the student
    4. Build teacher prompts (question + gold demonstration in-context)
    5. Forward both student and teacher on the rollout tokens
    6. Compute analytic per-token reverse KL divergence (student || teacher)
    7. Backprop through student, EMA-update teacher weights

Usage:
    python -m src.train_sdft --task gsm8k --use_chat_template
    python -m src.train_sdft --task gsm8k  # without chat template
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
# We load the entire config into memory ONCE at startup to prevent race conditions.
# If the user modifies the config file after starting training, it won't affect this run.

def _parse_args_early():
    """Parse args early to get config path and flags before torch import."""
    import sys
    config_path = "configs/sdft.yaml"
    task = None
    use_chat_template = False
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
        "output_dir": output_dir,
        "gold_data_path": gold_data_path,
    }


def _load_config_once(config_path: str) -> dict:
    """Load config file and return a deep copy to ensure immutability."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return copy.deepcopy(cfg)


def _configure_cuda_devices(cfg: dict):
    """Set CUDA_VISIBLE_DEVICES based on config before torch is imported.

    For SDFT we need both student and teacher devices visible.
    """
    student_device = cfg.get("model", {}).get("student_device", "auto")
    teacher_device = cfg.get("model", {}).get("teacher_device", "auto")

    # Also support the standard single 'device' key as fallback
    if student_device == "auto" and teacher_device == "auto":
        device_cfg = cfg.get("model", {}).get("device", "auto")
        if isinstance(device_cfg, list):
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_cfg)
        elif isinstance(device_cfg, str) and device_cfg.startswith("cuda:"):
            gpu_id = device_cfg.split(":")[1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        return

    # Collect all unique GPU IDs from student and teacher
    gpu_ids = set()
    for device_cfg in [student_device, teacher_device]:
        if isinstance(device_cfg, list):
            gpu_ids.update(device_cfg)
        elif isinstance(device_cfg, str) and device_cfg.startswith("cuda:"):
            gpu_ids.add(int(device_cfg.split(":")[1]))

    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in sorted(gpu_ids))


def _generate_output_dir(task: str, use_chat_template: bool) -> str:
    """Generate a descriptive output directory name."""
    training_format = "chat" if use_chat_template else "nochat"
    timestamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S")
    dir_name = f"{training_format}_sdft_{task}_{timestamp}"
    return os.path.join("results", task, "models", dir_name)


# Parse args and load config ONCE at module load time
_EARLY_ARGS = _parse_args_early()
_CONFIG_PATH = _EARLY_ARGS["config_path"]
_FROZEN_CONFIG = _load_config_once(_CONFIG_PATH)
_configure_cuda_devices(_FROZEN_CONFIG)

# Now safe to import torch
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from tqdm import tqdm

from tasks import get_task
from src.data import format_with_chat_template, format_without_chat_template
from src.config_utils import validate_config


# =============================================================================
# Paper's teacher prompt template (Section 3)
# =============================================================================
# This template is intentionally task-agnostic, as shown in the paper.
# It prevents verbatim copying of the demonstration while leveraging ICL.

TEACHER_TEMPLATE = (
    "{question}\n\n"
    "This is an example for a response to the question: {demonstration}\n\n"
    "Now answer with a response of your own, including the thinking process:\n"
)


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


# =============================================================================
# GPU device mapping helpers
# =============================================================================

def _resolve_device(device_cfg, visible_gpus: list[int]) -> torch.device:
    """Resolve a config device spec to a torch.device, respecting CUDA_VISIBLE_DEVICES remapping.

    When CUDA_VISIBLE_DEVICES is set, physical GPU IDs are remapped to logical indices.
    E.g., if CUDA_VISIBLE_DEVICES="4,5", physical GPU 4 becomes cuda:0, GPU 5 becomes cuda:1.
    """
    if isinstance(device_cfg, list) and len(device_cfg) == 1:
        physical_id = device_cfg[0]
        logical_id = visible_gpus.index(physical_id) if physical_id in visible_gpus else 0
        return torch.device(f"cuda:{logical_id}")
    elif isinstance(device_cfg, str) and device_cfg.startswith("cuda:"):
        physical_id = int(device_cfg.split(":")[1])
        logical_id = visible_gpus.index(physical_id) if physical_id in visible_gpus else 0
        return torch.device(f"cuda:{logical_id}")
    return torch.device("cuda:0")


def _get_visible_gpus() -> list[int]:
    """Get the list of physical GPU IDs from CUDA_VISIBLE_DEVICES."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if env:
        return [int(x) for x in env.split(",")]
    return list(range(torch.cuda.device_count()))


# =============================================================================
# EMA teacher
# =============================================================================

@torch.no_grad()
def ema_update(student_model: torch.nn.Module, teacher_model: torch.nn.Module, decay: float):
    """Update teacher parameters with EMA of student parameters.

    θ_teacher = decay * θ_teacher + (1 - decay) * θ_student
    """
    student_params = dict(student_model.named_parameters())
    for name, teacher_param in teacher_model.named_parameters():
        if name in student_params:
            student_param = student_params[name].to(teacher_param.device)
            teacher_param.mul_(decay).add_(student_param, alpha=1.0 - decay)


# =============================================================================
# Rollout generation
# =============================================================================

def generate_rollouts(model, tokenizer, prompts: list[str],
                      max_new_tokens: int, temperature: float) -> list[str]:
    """Generate one on-policy rollout per prompt from the student model.

    Returns:
        List of completion strings (one per prompt).
    """
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
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
        )

    tokenizer.padding_side = original_padding_side

    completions = []
    for i in range(len(prompts)):
        prompt_len = prompt_lengths[i]
        pad_len = (inputs["attention_mask"][i] == 0).sum().item()
        start_idx = pad_len + prompt_len
        generated_ids = output_ids[i][start_idx:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completions.append(completion)

    return completions


# =============================================================================
# Teacher prompt construction
# =============================================================================

def build_teacher_prompt(
    tokenizer,
    question: str,
    demonstration: str,
    use_chat_template: bool,
    extra_chat_template_kwargs: dict | None = None,
) -> str:
    """Build the teacher prompt: question + demonstration in-context.

    Uses the paper's template (Section 3):
        <Question> This is an example for a response to the question:
        <Demonstration> Now answer with a response of your own,
        including the thinking process:

    Args:
        tokenizer: HuggingFace tokenizer
        question: The original question
        demonstration: The gold answer to use as in-context demonstration
        use_chat_template: Whether to wrap in chat template
        extra_chat_template_kwargs: Extra kwargs for chat template
    """
    teacher_content = TEACHER_TEMPLATE.format(
        question=question,
        demonstration=demonstration,
    )

    if use_chat_template:
        # Wrap in chat template as a user message, with generation prompt
        messages = [{"role": "user", "content": teacher_content}]
        kwargs = extra_chat_template_kwargs or {}
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
    else:
        # Raw format: the template IS the prompt
        return teacher_content


# =============================================================================
# Analytic per-token KL divergence
# =============================================================================

def compute_per_token_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute analytic per-token KL divergence: KL(student || teacher).

    This is the "full analytic per-token estimator" from Appendix A.1:
        KL_t = Σ_v p_student(v) * (log p_student(v) - log p_teacher(v))

    Args:
        student_logits: [batch, seq_len, vocab_size] — student's logits
        teacher_logits: [batch, seq_len, vocab_size] — teacher's logits
        mask: [batch, seq_len] — 1 for valid positions, 0 for padding

    Returns:
        Scalar KL loss averaged over valid tokens.
    """
    # Convert to log probabilities
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # KL(student || teacher) = Σ_v exp(student_log_probs) * (student_log_probs - teacher_log_probs)
    # = Σ_v p_student * log(p_student / p_teacher)
    kl_per_token = F.kl_div(
        teacher_log_probs,  # input (target distribution in log space)
        student_log_probs,  # target (student distribution in log space)
        log_target=True,
        reduction="none",
    ).sum(dim=-1)  # sum over vocab → [batch, seq_len]

    if mask is not None:
        # Mask out padding positions
        kl_per_token = kl_per_token * mask
        total_tokens = mask.sum()
        if total_tokens > 0:
            return kl_per_token.sum() / total_tokens
        return kl_per_token.sum()  # avoid division by zero

    return kl_per_token.mean()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Self-Distillation Fine-Tuning (SDFT)")
    parser.add_argument("--config", type=str, default="configs/sdft.yaml")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g. gsm8k)")
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Format data with the model's chat template")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save the fine-tuned model (auto-generated if not provided)")
    parser.add_argument("--gold_data_path", type=str, default=None,
                        help="Path to gold_data.json; train on selected gold data instead of full set")
    args = parser.parse_args()

    cfg = _FROZEN_CONFIG
    validate_config(cfg, "sdft")
    model_name = cfg["model"]["name"]
    extra_kwargs = cfg["model"].get("extra_chat_template_kwargs", {})
    train_cfg = cfg["training"]
    sdft_cfg = cfg["sdft"]

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
            "gold_data_path": args.gold_data_path,
        },
    )

    # ---- Resolve GPU devices ----
    visible_gpus = _get_visible_gpus()
    student_device_cfg = cfg["model"].get("student_device", cfg["model"].get("device", "auto"))
    teacher_device_cfg = cfg["model"].get("teacher_device", student_device_cfg)
    student_device = _resolve_device(student_device_cfg, visible_gpus)
    teacher_device = _resolve_device(teacher_device_cfg, visible_gpus)

    same_gpu = (student_device == teacher_device)
    print(f"Student device: {student_device} (physical: {student_device_cfg})")
    print(f"Teacher device: {teacher_device} (physical: {teacher_device_cfg})")
    if same_gpu:
        print("  → Student and teacher share the same GPU")

    # ---- Load tokenizer ----
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = get_attn_implementation()
    dtype = torch.bfloat16 if train_cfg.get("bf16") else torch.float32

    # ---- Load student model ----
    student_kwargs = {"torch_dtype": dtype}
    if attn_impl:
        student_kwargs["attn_implementation"] = attn_impl
    student_model = AutoModelForCausalLM.from_pretrained(model_name, **student_kwargs)
    student_model = student_model.to(student_device)

    if not hasattr(student_model.config, "pad_token_id") or student_model.config.pad_token_id is None:
        student_model.config.pad_token_id = tokenizer.pad_token_id
    student_model.config.bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if hasattr(student_model, "generation_config") and student_model.generation_config is not None:
        student_model.generation_config.bos_token_id = getattr(tokenizer, "bos_token_id", None)
        if student_model.generation_config.pad_token_id is None:
            student_model.generation_config.pad_token_id = tokenizer.pad_token_id

    # ---- Create EMA teacher model ----
    print("Creating EMA teacher model...")
    teacher_kwargs = {"torch_dtype": dtype}
    if attn_impl:
        teacher_kwargs["attn_implementation"] = attn_impl
    teacher_model = AutoModelForCausalLM.from_pretrained(model_name, **teacher_kwargs)
    teacher_model = teacher_model.to(teacher_device)
    teacher_model.eval()
    # Freeze teacher — no gradients needed
    for param in teacher_model.parameters():
        param.requires_grad = False

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
    ema_decay = sdft_cfg["ema_decay"]
    gen_max_new_tokens = sdft_cfg["max_new_tokens"]
    gen_temperature = sdft_cfg["temperature"]

    total_batches_per_epoch = math.ceil(len(train_samples) / batch_size)
    total_batches = total_batches_per_epoch * num_epochs
    total_optimizer_steps = math.ceil(total_batches / grad_accum_steps)

    optimizer = AdamW(
        student_model.parameters(),
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

    use_grad_ckpt = train_cfg.get("gradient_checkpointing", False)
    if use_grad_ckpt:
        student_model.gradient_checkpointing_enable()

    seed = train_cfg.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)

    print(f"Starting SDFT → {args.output_dir}")
    print(f"  chat_template={args.use_chat_template}")
    print(f"  batch_size={batch_size}, grad_accum={grad_accum_steps}, ema_decay={ema_decay}")
    print(f"  total_batches={total_batches}, optimizer_steps={total_optimizer_steps}")

    # ---- Log file ----
    train_log_path = os.path.join(args.output_dir, "train_log.jsonl")
    train_log_file = open(train_log_path, "a")
    print(f"  logging to: {train_log_path}")

    # ---- Training loop ----
    global_step = 0
    accum_step = 0
    log_kl_loss = 0.0
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

            # --- 1. Build student prompts and generate on-policy rollouts ---
            student_model.eval()
            if use_grad_ckpt:
                student_model.config.use_cache = True

            student_prompts = []
            for s in batch_samples:
                if args.use_chat_template:
                    p = format_with_chat_template(tokenizer, s.question, None, extra_kwargs)
                else:
                    p = format_without_chat_template(s.question, None)
                student_prompts.append(p)

            rollouts = generate_rollouts(
                student_model, tokenizer, student_prompts,
                gen_max_new_tokens, gen_temperature,
            )
            torch.cuda.empty_cache()

            # --- 2. Build teacher prompts (question + demonstration) ---
            teacher_prompts = []
            for s in batch_samples:
                tp = build_teacher_prompt(
                    tokenizer=tokenizer,
                    question=s.question,
                    demonstration=s.answer,  # gold answer as demonstration
                    use_chat_template=args.use_chat_template,
                    extra_chat_template_kwargs=extra_kwargs,
                )
                teacher_prompts.append(tp)

            # --- 3. Tokenize: student_prompt + rollout for both models ---
            # For each sample, we need the full sequence (prompt + rollout) for
            # both student and teacher, and a mask covering only the rollout tokens.
            student_model.train()
            if use_grad_ckpt:
                student_model.config.use_cache = False

            batch_kl_loss = torch.tensor(0.0, device=student_device)
            valid_count = 0

            for i in range(len(batch_samples)):
                rollout_text = rollouts[i]
                if not rollout_text.strip():
                    continue  # skip empty rollouts

                # Student: prompt + rollout
                student_full_text = student_prompts[i] + rollout_text
                if not args.use_chat_template:
                    student_full_text += tokenizer.eos_token

                student_tokens = tokenizer(
                    student_full_text,
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(student_device)

                # Teacher: teacher_prompt + rollout
                teacher_full_text = teacher_prompts[i] + rollout_text
                if not args.use_chat_template:
                    teacher_full_text += tokenizer.eos_token

                teacher_tokens = tokenizer(
                    teacher_full_text,
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(teacher_device)

                # Tokenize prompts alone to find where rollout starts
                student_prompt_len = len(tokenizer(
                    student_prompts[i],
                    truncation=True,
                    max_length=max_seq_length,
                    add_special_tokens=False,
                )["input_ids"])
                teacher_prompt_len = len(tokenizer(
                    teacher_prompts[i],
                    truncation=True,
                    max_length=max_seq_length,
                    add_special_tokens=False,
                )["input_ids"])

                # Forward pass through both models
                student_outputs = student_model(
                    input_ids=student_tokens["input_ids"],
                    attention_mask=student_tokens["attention_mask"],
                )
                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        input_ids=teacher_tokens["input_ids"],
                        attention_mask=teacher_tokens["attention_mask"],
                    )

                # Extract logits for the rollout portion only
                # Logits at position t predict token t+1, so for rollout tokens
                # starting at position P, we want logits from P-1 to end-1
                student_seq_len = student_tokens["input_ids"].shape[1]
                teacher_seq_len = teacher_tokens["input_ids"].shape[1]

                student_rollout_start = max(student_prompt_len - 1, 0)
                teacher_rollout_start = max(teacher_prompt_len - 1, 0)

                student_rollout_logits = student_outputs.logits[:, student_rollout_start:-1, :]
                teacher_rollout_logits = teacher_outputs.logits[:, teacher_rollout_start:-1, :]

                # Align lengths: both should predict the same rollout tokens
                min_len = min(
                    student_rollout_logits.shape[1],
                    teacher_rollout_logits.shape[1],
                )
                if min_len <= 0:
                    continue

                student_rollout_logits = student_rollout_logits[:, :min_len, :]
                teacher_rollout_logits = teacher_rollout_logits[:, :min_len, :].to(student_device)

                # Compute per-token KL
                kl_loss = compute_per_token_kl(student_rollout_logits, teacher_rollout_logits)
                batch_kl_loss = batch_kl_loss + kl_loss
                valid_count += 1

            # Average over batch
            if valid_count > 0:
                batch_kl_loss = batch_kl_loss / valid_count / grad_accum_steps
                batch_kl_loss.backward()

                log_kl_loss += batch_kl_loss.item() * grad_accum_steps
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

                # EMA update teacher weights
                ema_update(student_model, teacher_model, ema_decay)

                logging_steps = train_cfg.get("logging_steps", 10)
                if global_step % logging_steps == 0 or is_last_batch:
                    avg_kl = log_kl_loss / log_count if log_count > 0 else 0.0
                    lr = scheduler.get_last_lr()[0]
                    msg = (
                        f"  step {global_step}/{total_optimizer_steps} | "
                        f"kl_loss={avg_kl:.6f} | "
                        f"lr={lr:.2e}"
                    )
                    print(msg)
                    log_entry = {
                        "step": global_step,
                        "total_steps": total_optimizer_steps,
                        "epoch": epoch + 1,
                        "kl_loss": avg_kl,
                        "lr": lr,
                    }
                    train_log_file.write(json.dumps(log_entry) + "\n")
                    train_log_file.flush()
                    log_kl_loss = 0.0
                    log_count = 0

            pbar.set_postfix({
                "valid": f"{valid_count}/{len(batch_samples)}",
                "step": global_step,
            })

        # Save checkpoint at end of epoch
        if train_cfg.get("save_strategy", "epoch") == "epoch":
            ckpt_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch + 1}")
            student_model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Save final model
    train_log_file.close()
    student_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
