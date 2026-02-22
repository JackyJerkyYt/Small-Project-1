#!/bin/bash
set -e  # Exit on first error

# echo "=== Eval 1/8: nochat_sft_fullloss → WITH chat template ==="
# uv run python -m src.evaluate --task gsm8k \
#     --model_path results/gsm8k/models/nochat_sft_fullloss_gsm8k_20260221_160321 \
#     --use_chat_template \
#     --output_dir results/gsm8k/eval/nochat_sft_fullloss_eval_chat

# echo "=== Eval 2/8: nochat_sft_fullloss → WITHOUT chat template ==="
# uv run python -m src.evaluate --task gsm8k \
#     --model_path results/gsm8k/models/nochat_sft_fullloss_gsm8k_20260221_160321 \
#     --output_dir results/gsm8k/eval/nochat_sft_fullloss_eval_raw

# echo "=== Eval 3/8: chat_sft_fullloss → WITH chat template ==="
# uv run python -m src.evaluate --task gsm8k \
#     --model_path results/gsm8k/models/chat_sft_fullloss_gsm8k_20260221_160340 \
#     --use_chat_template \
#     --output_dir results/gsm8k/eval/chat_sft_fullloss_eval_chat

# echo "=== Eval 4/8: chat_sft_fullloss → WITHOUT chat template ==="
# uv run python -m src.evaluate --task gsm8k \
#     --model_path results/gsm8k/models/chat_sft_fullloss_gsm8k_20260221_160340 \
#     --output_dir results/gsm8k/eval/chat_sft_fullloss_eval_raw

# echo "=== Eval 5/8: nochat_sft_maskq → WITH chat template ==="
# uv run python -m src.evaluate --task gsm8k \
#     --model_path results/gsm8k/models/nochat_sft_maskq_gsm8k_20260221_175625 \
#     --use_chat_template \
#     --output_dir results/gsm8k/eval/nochat_sft_maskq_eval_chat

# echo "=== Eval 6/8: nochat_sft_maskq → WITHOUT chat template ==="
# uv run python -m src.evaluate --task gsm8k \
#     --model_path results/gsm8k/models/nochat_sft_maskq_gsm8k_20260221_175625 \
#     --output_dir results/gsm8k/eval/nochat_sft_maskq_eval_raw

echo "=== Eval 7/8: chat_sft_maskq → WITH chat template ==="
uv run python -m src.evaluate --task gsm8k \
    --model_path results/gsm8k/models/chat_sft_maskq_gsm8k_20260222_011944 \
    --use_chat_template \
    --output_dir results/gsm8k/eval/chat_sft_maskq_eval_chat

echo "=== Eval 8/8: chat_sft_maskq → WITHOUT chat template ==="
uv run python -m src.evaluate --task gsm8k \
    --model_path results/gsm8k/models/chat_sft_maskq_gsm8k_20260222_011944 \
    --output_dir results/gsm8k/eval/chat_sft_maskq_eval_raw

echo "=== All evaluations complete ==="
