#!/bin/bash
set -e  # Exit on first error

echo "=== Eval 1/16: chat_grpo_default → WITH chat template ==="
uv run python -m src.evaluate --task gsm8k \
    --model_path results/gsm8k/models/chat_grpo_default_gsm8k_20260302_210119 \
    --use_chat_template \
    --output_dir results/gsm8k/eval/chat_grpo_default_eval_chat

echo "=== Eval 2/16: chat_grpo_default → WITHOUT chat template ==="
uv run python -m src.evaluate --task gsm8k \
    --model_path results/gsm8k/models/chat_grpo_default_gsm8k_20260302_210119 \
    --output_dir results/gsm8k/eval/chat_grpo_default_eval_raw

echo "=== Eval 3/16: chat_onpolicysft_maskq → WITH chat template ==="
uv run python -m src.evaluate --task gsm8k \
    --model_path results/gsm8k/models/chat_onpolicysft_maskq_gsm8k_20260302_201652 \
    --use_chat_template \
    --output_dir results/gsm8k/eval/chat_onpolicysft_maskq_eval_chat

echo "=== Eval 4/16: chat_onpolicysft_maskq → WITHOUT chat template ==="
uv run python -m src.evaluate --task gsm8k \
    --model_path results/gsm8k/models/chat_onpolicysft_maskq_gsm8k_20260302_201652 \
    --output_dir results/gsm8k/eval/chat_onpolicysft_maskq_eval_raw

echo "=== Eval 5/16: chat_sdft → WITH chat template ==="
uv run python -m src.evaluate --task gsm8k \
    --model_path results/gsm8k/models/chat_sdft_gsm8k_20260304_042012 \
    --use_chat_template \
    --output_dir results/gsm8k/eval/chat_sdft_eval_chat

echo "=== Eval 6/16: chat_sdft → WITHOUT chat template ==="
uv run python -m src.evaluate --task gsm8k \
    --model_path results/gsm8k/models/chat_sdft_gsm8k_20260304_042012 \
    --output_dir results/gsm8k/eval/chat_sdft_eval_raw

echo "=== Eval 7/16: chat_sft_maskq → WITH chat template ==="
uv run python -m src.evaluate --task gsm8k \
    --model_path results/gsm8k/models/chat_sft_maskq_gsm8k_20260302_145152 \
    --use_chat_template \
    --output_dir results/gsm8k/eval/chat_sft_maskq_eval_chat

echo "=== Eval 8/16: chat_sft_maskq → WITHOUT chat template ==="
uv run python -m src.evaluate --task gsm8k \
    --model_path results/gsm8k/models/chat_sft_maskq_gsm8k_20260302_145152 \
    --output_dir results/gsm8k/eval/chat_sft_maskq_eval_raw

echo "=== Eval 9/16: nochat_grpo_default → WITH chat template ==="
uv run python -m src.evaluate --task gsm8k \
    --model_path results/gsm8k/models/nochat_grpo_default_gsm8k_20260304_031138 \
    --use_chat_template \
    --output_dir results/gsm8k/eval/nochat_grpo_default_eval_chat

echo "=== Eval 10/16: nochat_grpo_default → WITHOUT chat template ==="
uv run python -m src.evaluate --task gsm8k \
    --model_path results/gsm8k/models/nochat_grpo_default_gsm8k_20260304_031138 \
    --output_dir results/gsm8k/eval/nochat_grpo_default_eval_raw

echo "=== Eval 11/16: nochat_onpolicysft_maskq → WITH chat template ==="
uv run python -m src.evaluate --task gsm8k \
    --model_path results/gsm8k/models/nochat_onpolicysft_maskq_gsm8k_20260302_201737 \
    --use_chat_template \
    --output_dir results/gsm8k/eval/nochat_onpolicysft_maskq_eval_chat

echo "=== Eval 12/16: nochat_onpolicysft_maskq → WITHOUT chat template ==="
uv run python -m src.evaluate --task gsm8k \
    --model_path results/gsm8k/models/nochat_onpolicysft_maskq_gsm8k_20260302_201737 \
    --output_dir results/gsm8k/eval/nochat_onpolicysft_maskq_eval_raw

echo "=== Eval 13/16: nochat_sdft → WITH chat template ==="
uv run python -m src.evaluate --task gsm8k \
    --model_path results/gsm8k/models/nochat_sdft_gsm8k_20260304_134107 \
    --use_chat_template \
    --output_dir results/gsm8k/eval/nochat_sdft_eval_chat

echo "=== Eval 14/16: nochat_sdft → WITHOUT chat template ==="
uv run python -m src.evaluate --task gsm8k \
    --model_path results/gsm8k/models/nochat_sdft_gsm8k_20260304_134107 \
    --output_dir results/gsm8k/eval/nochat_sdft_eval_raw

echo "=== Eval 15/16: nochat_sft_maskq → WITH chat template ==="
uv run python -m src.evaluate --task gsm8k \
    --model_path results/gsm8k/models/nochat_sft_maskq_gsm8k_20260304_154301 \
    --use_chat_template \
    --output_dir results/gsm8k/eval/nochat_sft_maskq_eval_chat

echo "=== Eval 16/16: nochat_sft_maskq → WITHOUT chat template ==="
uv run python -m src.evaluate --task gsm8k \
    --model_path results/gsm8k/models/nochat_sft_maskq_gsm8k_20260304_154301 \
    --output_dir results/gsm8k/eval/nochat_sft_maskq_eval_raw

echo "=== All evaluations complete ==="
