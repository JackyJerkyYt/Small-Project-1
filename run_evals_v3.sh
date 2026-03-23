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


uv run python -m src.evaluate --task math \
    --model_path /data/jacky/small_project_1/results/math/models/chat_sdft_math_20260309_013349 \
    --use_chat_template \
    --output_dir results/math/eval/chat_sdft_eval_chat

uv run python -m src.evaluate --task math \
    --model_path /data/jacky/small_project_1/results/math/models/chat_onpolicysft_maskq_math_20260309_062531 \
    --use_chat_template \
    --output_dir results/math/eval/chat_onpolicysft_maskq_eval_chat


uv run python -m src.evaluate --task math \
    --model_path /data/jacky/small_project_1/results/math/models/official_sdft_chat_template \
    --use_chat_template \
    --output_dir results/math/eval/official_sdft_chat_template_math