#!/usr/bin/env python3
"""
Master experiment runner.

This script runs all combinations of:
  - Training format: chat_template vs no_chat_template
  - Training method: SFT vs GRPO vs DPO
  - Masking (for no_chat_template SFT only): full loss vs mask question
  - Evaluation format: chat_template vs no_chat_template

Usage:
    python run_experiments.py --task gsm8k
    python run_experiments.py --task gsm8k --only_eval   # skip training, just evaluate
    python run_experiments.py --task gsm8k --experiments sft_chat grpo_no_chat
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class Experiment:
    name: str
    method: str          # "sft", "grpo", or "dpo"
    use_chat_template: bool
    mask_question: bool  # only relevant for SFT without chat template
    description: str


ALL_EXPERIMENTS = [
    # --- With chat template ---
    Experiment(
        name="sft_chat_template",
        method="sft",
        use_chat_template=True,
        mask_question=False,
        description="SFT with chat template",
    ),
    Experiment(
        name="grpo_chat_template",
        method="grpo",
        use_chat_template=True,
        mask_question=False,
        description="GRPO with chat template",
    ),
    # --- Without chat template, full loss (question + answer) ---
    Experiment(
        name="sft_no_chat_full_loss",
        method="sft",
        use_chat_template=False,
        mask_question=False,
        description="SFT without chat template, loss on question+answer",
    ),
    Experiment(
        name="grpo_no_chat",
        method="grpo",
        use_chat_template=False,
        mask_question=False,
        description="GRPO without chat template",
    ),
    # --- Without chat template, masked question (answer-only loss) ---
    Experiment(
        name="sft_no_chat_mask_q",
        method="sft",
        use_chat_template=False,
        mask_question=True,
        description="SFT without chat template, loss on answer only (question masked)",
    ),
    Experiment(
        name="sft_chat_mask_q",
        method="sft",
        use_chat_template=True,
        mask_question=True,
        description="SFT with chat template, loss on answer only",
    ),
    # --- DPO ---
    Experiment(
        name="dpo_chat_template",
        method="dpo",
        use_chat_template=True,
        mask_question=False,
        description="DPO with chat template",
    ),
    Experiment(
        name="dpo_no_chat",
        method="dpo",
        use_chat_template=False,
        mask_question=False,
        description="DPO without chat template",
    ),
]


def run_cmd(cmd: list[str], description: str) -> int:
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
    return result.returncode


def run_training(exp: Experiment, task: str, base_dir: str):
    model_dir = os.path.join(base_dir, "models", exp.name)

    if exp.method == "sft":
        cmd = [
            sys.executable, "-m", "src.train_sft",
            "--config", "configs/sft.yaml",
            "--task", task,
            "--output_dir", model_dir,
        ]
        if exp.use_chat_template:
            cmd.append("--use_chat_template")
        if exp.mask_question:
            cmd.append("--mask_question")
    elif exp.method == "grpo":
        cmd = [
            sys.executable, "-m", "src.train_grpo",
            "--config", "configs/grpo.yaml",
            "--task", task,
            "--output_dir", model_dir,
        ]
        if exp.use_chat_template:
            cmd.append("--use_chat_template")
    elif exp.method == "dpo":
        cmd = [
            sys.executable, "-m", "src.train_dpo",
            "--config", "configs/dpo.yaml",
            "--task", task,
            "--output_dir", model_dir,
        ]
        if exp.use_chat_template:
            cmd.append("--use_chat_template")
    else:
        raise ValueError(f"Unknown method: {exp.method}")

    return run_cmd(cmd, f"TRAIN: {exp.description}")


def run_evaluation(exp: Experiment, task: str, base_dir: str, eval_with_chat: bool,
                   extra_kwargs: str, max_samples: int | None = None):
    model_dir = os.path.join(base_dir, "models", exp.name)
    eval_suffix = "eval_chat" if eval_with_chat else "eval_raw"
    eval_dir = os.path.join(base_dir, "eval", f"{exp.name}_{eval_suffix}")

    desc = f"EVAL: {exp.description} → eval {'with' if eval_with_chat else 'without'} chat template"

    cmd = [
        sys.executable, "-m", "src.evaluate",
        "--config", "configs/eval.yaml",
        "--task", task,
        "--model_path", model_dir,
        "--output_dir", eval_dir,
        "--extra_chat_template_kwargs", extra_kwargs,
        "--experiment_name", exp.name,
        "--train_method", exp.method,
    ]
    if eval_with_chat:
        cmd.append("--use_chat_template")
    if exp.use_chat_template:
        cmd.append("--train_chat_template")
    if exp.mask_question:
        cmd.append("--train_mask_question")
    if max_samples is not None:
        cmd.extend(["--max_samples", str(max_samples)])

    return run_cmd(cmd, desc)


def run_report(base_dir: str):
    eval_base = os.path.join(base_dir, "eval")
    if not os.path.isdir(eval_base):
        print("No eval results found. Skipping report.")
        return

    eval_dirs = [
        os.path.join(eval_base, d) for d in sorted(os.listdir(eval_base))
        if os.path.isfile(os.path.join(eval_base, d, "summary.json"))
    ]

    if not eval_dirs:
        print("No summary.json files found. Skipping report.")
        return

    report_dir = os.path.join(base_dir, "report")
    cmd = [
        sys.executable, "-m", "src.report",
        "--results_dirs", *eval_dirs,
        "--output_dir", report_dir,
    ]
    run_cmd(cmd, "Generating comparison report")


def main():
    parser = argparse.ArgumentParser(
        description="Run all fine-tuning + evaluation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run everything for gsm8k
  python run_experiments.py --task gsm8k

  # Only evaluate (models must already be trained)
  python run_experiments.py --task gsm8k --only_eval

  # Run only specific experiments
  python run_experiments.py --task gsm8k --experiments sft_chat_template grpo_chat_template

  # Only generate the comparison report
  python run_experiments.py --task gsm8k --only_report

Available experiments:
  sft_chat_template       - SFT with chat template
  grpo_chat_template      - GRPO with chat template
  sft_no_chat_full_loss   - SFT without chat template (full loss)
  sft_no_chat_mask_q      - SFT without chat template (answer-only loss)
  sft_chat_mask_q         - SFT with chat template (answer-only loss)
  grpo_no_chat            - GRPO without chat template
  dpo_chat_template       - DPO with chat template
  dpo_no_chat             - DPO without chat template
        """,
    )
    parser.add_argument("--task", type=str, default="gsm8k", help="Task name")
    parser.add_argument("--base_dir", type=str, default="results",
                        help="Base directory for all outputs")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Run only these experiments (by name)")
    parser.add_argument("--only_eval", action="store_true",
                        help="Skip training, only run evaluation")
    parser.add_argument("--only_report", action="store_true",
                        help="Skip training+eval, only generate the report")
    parser.add_argument("--extra_chat_template_kwargs", type=str,
                        default='{}',
                        help="JSON extra kwargs for chat template")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Limit test samples for evaluation (passed to evaluate.py --max_samples)")
    args = parser.parse_args()

    base_dir = os.path.join(args.base_dir, args.task)
    os.makedirs(base_dir, exist_ok=True)

    experiments = ALL_EXPERIMENTS
    if args.experiments:
        experiments = [e for e in ALL_EXPERIMENTS if e.name in args.experiments]
        if not experiments:
            available = [e.name for e in ALL_EXPERIMENTS]
            print(f"No matching experiments. Available: {available}")
            sys.exit(1)

    if args.only_report:
        run_report(base_dir)
        return

    for exp in experiments:
        if not args.only_eval:
            rc = run_training(exp, args.task, base_dir)
            if rc != 0:
                print(f"Training failed for {exp.name}. Skipping its evaluation.")
                continue

        for eval_with_chat in [True, False]:
            run_evaluation(exp, args.task, base_dir, eval_with_chat,
                           args.extra_chat_template_kwargs, args.max_eval_samples)

    run_report(base_dir)
    print("\nAll done! Check the results in:", base_dir)


if __name__ == "__main__":
    main()
