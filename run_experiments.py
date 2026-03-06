#!/usr/bin/env python3
"""
Master experiment runner.

Runs chat-template training + evaluation for each method (SFT, GRPO, DPO variants).

Usage:
    python run_experiments.py --task gsm8k
    python run_experiments.py --task math
    python run_experiments.py --task math --only_eval
    python run_experiments.py --task math --experiments sft_chat_template grpo_chat_template
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass


def gold_data_path(task: str) -> str:
    return f"results/gold_data/chat_template/{task}/gold_data.json"


@dataclass
class Experiment:
    name: str
    method: str          # "sft", "grpo", or "dpo"
    use_chat_template: bool
    mask_question: bool  # only relevant for SFT without chat template
    description: str
    config: str | None = None  # override default config path for this method


ALL_EXPERIMENTS = [
    Experiment(
        name="sft_chat_template",
        method="sft",
        use_chat_template=True,
        mask_question=False,
        description="SFT with chat template",
    ),
    Experiment(
        name="sft_chat_mask_q",
        method="sft",
        use_chat_template=True,
        mask_question=True,
        description="SFT with chat template, loss on answer only",
    ),
    Experiment(
        name="grpo_chat_template",
        method="grpo",
        use_chat_template=True,
        mask_question=False,
        description="GRPO with chat template",
    ),
    Experiment(
        name="dpo_chat_template",
        method="dpo",
        use_chat_template=True,
        mask_question=False,
        description="DPO with chat template",
        config="configs/dpo.yaml",
    ),
    Experiment(
        name="dpo_naive_chat_template",
        method="dpo",
        use_chat_template=True,
        mask_question=False,
        description="DPO naive with chat template",
        config="configs/dpo_naive.yaml",
    ),
    Experiment(
        name="dpo_akshat_chat_template",
        method="dpo",
        use_chat_template=True,
        mask_question=False,
        description="DPO akshat with chat template",
        config="configs/dpo_akshat.yaml",
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


def _default_config(method: str) -> str:
    return f"configs/{method}.yaml"


def run_training(exp: Experiment, task: str, base_dir: str):
    model_dir = os.path.join(base_dir, "models", exp.name)
    config_path = exp.config or _default_config(exp.method)
    gdp = gold_data_path(task)

    train_module = {
        "sft": "src.train_sft",
        "grpo": "src.train_grpo",
        "dpo": "src.train_dpo",
    }
    if exp.method not in train_module:
        raise ValueError(f"Unknown method: {exp.method}")

    cmd = [
        sys.executable, "-m", train_module[exp.method],
        "--config", config_path,
        "--task", task,
        "--output_dir", model_dir,
        "--gold_data_path", gdp,
        "--use_chat_template",
    ]
    if exp.mask_question:
        cmd.append("--mask_question")

    return run_cmd(cmd, f"TRAIN: {exp.description}")


def run_evaluation(exp: Experiment, task: str, base_dir: str,
                   extra_kwargs: str, max_samples: int | None = None):
    model_dir = os.path.join(base_dir, "models", exp.name)
    eval_dir = os.path.join(base_dir, "eval", f"{exp.name}_eval_chat")

    cmd = [
        sys.executable, "-m", "src.evaluate",
        "--config", "configs/eval.yaml",
        "--task", task,
        "--model_path", model_dir,
        "--output_dir", eval_dir,
        "--extra_chat_template_kwargs", extra_kwargs,
        "--experiment_name", exp.name,
        "--train_method", exp.method,
        "--use_chat_template",
        "--train_chat_template",
    ]
    if exp.mask_question:
        cmd.append("--train_mask_question")
    if max_samples is not None:
        cmd.extend(["--max_samples", str(max_samples)])

    return run_cmd(cmd, f"EVAL: {exp.description} → chat template")


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
        description="Run chat-template fine-tuning + evaluation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py --task gsm8k
  python run_experiments.py --task math
  python run_experiments.py --task math --only_eval
  python run_experiments.py --task math --experiments sft_chat_template grpo_chat_template
  python run_experiments.py --task math --only_report

Available experiments:
  sft_chat_template           - SFT with chat template
  sft_chat_mask_q             - SFT with chat template (answer-only loss)
  grpo_chat_template          - GRPO with chat template
  dpo_chat_template           - DPO (data-diverse) with chat template
  dpo_naive_chat_template     - DPO naive (1 pair/example) with chat template
  dpo_akshat_chat_template    - DPO akshat (64 rollouts/question) with chat template
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

        run_evaluation(exp, args.task, base_dir,
                       args.extra_chat_template_kwargs, args.max_eval_samples)

    run_report(base_dir)
    print("\nAll done! Check the results in:", base_dir)


if __name__ == "__main__":
    main()
