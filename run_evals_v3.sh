#!/bin/bash
set -e  # Exit on first error


echo "=== Eval 3/16: chat_onpolicysft_maskq → WITH chat template ==="
uv run python -m src.evaluate --task gsm8k \
    --model_path results/gsm8k/models/chat_onpolicysft_maskq_gsm8k_20260306_191741 \
    --use_chat_template \
    --output_dir results/gsm8k/eval/chat_onpolicysft_maskq_eval_chat

echo "=== Eval 5/16: chat_sdft → WITH chat template ==="
uv run python -m src.evaluate --task gsm8k \
    --model_path results/gsm8k/models/chat_sdft_gsm8k_20260306_191846 \
    --use_chat_template \
    --output_dir results/gsm8k/eval/chat_sdft_eval_chat

echo "=== All evaluations complete ==="
