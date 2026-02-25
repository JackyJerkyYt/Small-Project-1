"""Evaluation script: run a fine-tuned (or base) model on a task and produce reports + plots."""

import argparse
import json
import os
import yaml

# Must configure CUDA devices BEFORE importing torch
def _configure_cuda_devices(config_path: str):
    """Read config and set CUDA_VISIBLE_DEVICES before torch is imported."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    device_cfg = cfg.get("eval", {}).get("device", "auto")
    
    if isinstance(device_cfg, list):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_cfg)
    elif isinstance(device_cfg, str) and device_cfg.startswith("cuda:"):
        gpu_id = device_cfg.split(":")[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

# Parse args early to get config path
def _get_config_path():
    import sys
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "configs/eval.yaml"

_configure_cuda_devices(_get_config_path())

# Now safe to import torch
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

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


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    gen_cfg: dict,
) -> list[str]:
    """Generate completions for a batch of prompts using left-padding."""
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
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
            temperature=gen_cfg.get("temperature", 0.0),
            do_sample=gen_cfg.get("do_sample", False),
            pad_token_id=tokenizer.pad_token_id,
        )
    
    tokenizer.padding_side = original_padding_side
    
    completions = []
    for i, (output, prompt_len) in enumerate(zip(output_ids, prompt_lengths)):
        pad_len = (inputs["attention_mask"][i] == 0).sum().item()
        start_idx = pad_len + prompt_len
        generated_ids = output[start_idx:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completions.append(completion)
    
    return completions


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on a task")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g. gsm8k)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model (local dir or HF hub name)")
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Format prompts with the model's chat template for evaluation")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save evaluation results")
    parser.add_argument("--extra_chat_template_kwargs", type=str, default="{}",
                        help="JSON string of extra kwargs for chat template (e.g. '{\"enable_thinking\": false}')")
    parser.add_argument("--experiment_name", type=str, default="",
                        help="Name tag for this experiment (stored in summary for reporting)")
    parser.add_argument("--train_method", type=str, default="",
                        help="Training method used (sft/grpo), stored in summary")
    parser.add_argument("--train_chat_template", action="store_true",
                        help="Whether training used chat template (stored in summary)")
    parser.add_argument("--train_mask_question", action="store_true",
                        help="Whether training masked the question (stored in summary)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Override eval.max_samples from config (limit test samples)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg, "eval")
    gen_cfg = cfg.get("generation", {})
    eval_cfg = cfg.get("eval", {})
    extra_kwargs = cfg.get("chat_template", {})
    extra_kwargs.update(json.loads(args.extra_chat_template_kwargs))

    # Auto-populate training metadata from run_info.yaml if present
    run_info_path = os.path.join(args.model_path, "run_info.yaml")
    if os.path.exists(run_info_path):
        with open(run_info_path) as f:
            run_info = yaml.safe_load(f)
        cli_args = run_info.get("cli_args", {})
        if not args.train_method:
            config_path = run_info.get("original_config_path", "")
            if "grpo" in config_path:
                args.train_method = "grpo"
            elif "dpo" in config_path:
                args.train_method = "dpo"
            elif "sft" in config_path:
                args.train_method = "sft"
        if not args.train_chat_template:
            args.train_chat_template = cli_args.get("use_chat_template", False)
        if not args.train_mask_question:
            args.train_mask_question = cli_args.get("mask_question", False)
        print(f"Auto-detected training info: method={args.train_method}, "
              f"chat_template={args.train_chat_template}, mask_question={args.train_mask_question}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = get_attn_implementation()
    use_bf16 = eval_cfg.get("bf16", True)  # default to bf16 for inference
    
    # Device configuration is handled at script startup via CUDA_VISIBLE_DEVICES
    # Here we just use device_map="auto" to distribute across visible GPUs
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if use_bf16 else torch.float32,
        "device_map": "auto",
    }
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.eval()

    print(f"Loading task: {args.task}")
    task = get_task(args.task)
    test_samples = task.load_test()

    max_samples = args.max_samples if args.max_samples is not None else eval_cfg.get("max_samples")
    if max_samples is not None:
        test_samples = test_samples[:max_samples]
    batch_size = eval_cfg.get("per_device_batch_size", 1)
    print(f"Evaluating on {len(test_samples)} samples (batch_size={batch_size})")

    all_prompts = []
    for sample in test_samples:
        if args.use_chat_template:
            prompt = format_with_chat_template(tokenizer, sample.question, None, extra_kwargs)
        else:
            prompt = format_without_chat_template(sample.question, None)
        all_prompts.append(prompt)

    all_outputs = []
    num_batches = (len(all_prompts) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(all_prompts))
        batch_prompts = all_prompts[start:end]
        batch_outputs = generate_batch(model, tokenizer, batch_prompts, gen_cfg)
        all_outputs.extend(batch_outputs)

    results = []
    correct = 0
    extraction_failures = 0
    empty_outputs = 0
    for i, (sample, model_output) in enumerate(zip(test_samples, all_outputs)):
        if not model_output or not model_output.strip():
            empty_outputs += 1
        extracted = task.extract_answer(model_output)
        if extracted is None:
            extraction_failures += 1
        gold_for_check = sample.answer_value if sample.answer_value else sample.answer
        is_correct = task.check_answer(extracted, gold_for_check)
        if is_correct:
            correct += 1

        results.append({
            "index": i,
            "question": sample.question,
            "gold_answer": sample.answer,
            "gold_value": sample.answer_value,
            "model_output": model_output,
            "extracted_answer": extracted,
            "correct": is_correct,
        })

    accuracy = correct / len(test_samples) if test_samples else 0.0

    summary = {
        "experiment_name": args.experiment_name,
        "train_method": args.train_method,
        "train_chat_template": args.train_chat_template,
        "train_mask_question": args.train_mask_question,
        "eval_chat_template": args.use_chat_template,
        "model_path": args.model_path,
        "task": args.task,
        "total_samples": len(test_samples),
        "correct": correct,
        "accuracy": accuracy,
        "extraction_failures": extraction_failures,
        "empty_outputs": empty_outputs,
    }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2)
    print(f"Results saved to {results_path}")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Task:           {args.task}")
    print(f"  Model:          {args.model_path}")
    print(f"  Chat template:  {args.use_chat_template}")
    print(f"  Accuracy:       {correct}/{len(test_samples)} = {accuracy:.2%}")
    print(f"  Extract fails:  {extraction_failures}/{len(test_samples)}")
    print(f"  Empty outputs:  {empty_outputs}/{len(test_samples)}")
    print(f"{'='*50}\n")

    return summary


if __name__ == "__main__":
    main()
