#!/bin/bash
set -e  # Exit on first error

echo "=== Control Eval 1/2: Qwen3-4B (no fine-tuning) → WITH chat template ==="
uv run python -m src.evaluate --task gsm8k \
    --model_path Qwen/Qwen3-4B \
    --use_chat_template \
    --output_dir results/gsm8k/eval/control_base_eval_chat

echo "=== Control Eval 2/2: Qwen3-4B (no fine-tuning) → WITHOUT chat template ==="
uv run python -m src.evaluate --task gsm8k \
    --model_path Qwen/Qwen3-4B \
    --output_dir results/gsm8k/eval/control_base_eval_raw

echo "=== Control evaluations complete ==="
