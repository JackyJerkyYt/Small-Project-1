# LLM Fine-Tune & Eval: Chat Template Experiments

Evaluate the effect of fine-tuning an instruction-tuned model (Qwen3-4B) **with or without chat template** and **evaluating with or without chat template**.

## Experiment Matrix

| # | Training Format | Training Method | Masking | Eval Format |
|---|----------------|----------------|---------|-------------|
| 1 | Chat template | SFT | — | Chat template |
| 2 | Chat template | SFT | — | No chat template |
| 3 | Chat template | GRPO (binary reward) | — | Chat template |
| 4 | Chat template | GRPO (binary reward) | — | No chat template |
| 5 | No chat template | SFT | Full (question+answer) | Chat template |
| 6 | No chat template | SFT | Full (question+answer) | No chat template |
| 7 | No chat template | SFT | Answer only (question masked) | Chat template |
| 8 | No chat template | SFT | Answer only (question masked) | No chat template |
| 9 | No chat template | GRPO (binary reward) | — | Chat template |
| 10 | No chat template | GRPO (binary reward) | — | No chat template |

## Setup

**Prerequisites:** Python 3.10+, CUDA-capable GPU, `uv` package manager.

```bash
# 1. Clone / navigate to the project
cd /data/jacky/small_project_1

# 2. Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e .

# 3. (Optional) Install flash-attention for faster training
uv pip install flash-attn --no-build-isolation
```

## Running Experiments

### Run all experiments (train + eval + report)

```bash
python run_experiments.py --task gsm8k
```

### Run specific experiments only

```bash
# Just the SFT with chat template and GRPO with chat template
python run_experiments.py --task gsm8k --experiments sft_chat_template grpo_chat_template
```

Available experiment names:
- `sft_chat_template` — SFT with chat template
- `grpo_chat_template` — GRPO with chat template
- `sft_no_chat_full_loss` — SFT without chat template, full loss
- `sft_no_chat_mask_q` — SFT without chat template, answer-only loss
- `grpo_no_chat` — GRPO without chat template

### Skip training, only evaluate (models must already be trained)

```bash
python run_experiments.py --task gsm8k --only_eval
```

### Only generate the comparison report (training + eval must be done)

```bash
python run_experiments.py --task gsm8k --only_report
```

## Running Individual Scripts

You can also run each step independently:

### SFT Training

```bash
# With chat template
python -m src.train_sft --task gsm8k --use_chat_template --output_dir results/gsm8k/models/sft_chat_template

# Without chat template, full loss
python -m src.train_sft --task gsm8k --output_dir results/gsm8k/models/sft_no_chat_full_loss

# Without chat template, masked question
python -m src.train_sft --task gsm8k --mask_question --output_dir results/gsm8k/models/sft_no_chat_mask_q
```

### GRPO Training

```bash
# With chat template
python -m src.train_grpo --task gsm8k --use_chat_template --output_dir results/gsm8k/models/grpo_chat_template

# Without chat template
python -m src.train_grpo --task gsm8k --output_dir results/gsm8k/models/grpo_no_chat
```

### Evaluation

```bash
# Evaluate with chat template
python -m src.evaluate \
    --task gsm8k \
    --model_path results/gsm8k/models/sft_chat_template \
    --use_chat_template \
    --output_dir results/gsm8k/eval/sft_chat_template_eval_chat \
    --extra_chat_template_kwargs '{"enable_thinking": false}'

# Evaluate without chat template
python -m src.evaluate \
    --task gsm8k \
    --model_path results/gsm8k/models/sft_chat_template \
    --output_dir results/gsm8k/eval/sft_chat_template_eval_raw
```

### Generate Report

```bash
python -m src.report \
    --results_dirs results/gsm8k/eval/* \
    --output_dir results/gsm8k/report
```

## Configuration

All hyperparameters are in YAML config files under `configs/`:

| File | Purpose |
|------|---------|
| `configs/sft.yaml` | SFT training hyperparameters (LR, epochs, batch size, etc.) |
| `configs/grpo.yaml` | GRPO training hyperparameters + generation settings (num_generations, beta, etc.) |
| `configs/eval.yaml` | Evaluation settings (max tokens, temperature, sample limit) |

## Output Structure

```
results/gsm8k/
├── models/
│   ├── sft_chat_template/          # Trained model checkpoints
│   ├── sft_no_chat_full_loss/
│   ├── sft_no_chat_mask_q/
│   ├── grpo_chat_template/
│   └── grpo_no_chat/
├── eval/
│   ├── sft_chat_template_eval_chat/
│   │   ├── results.json            # Full per-sample results
│   │   └── summary.json            # Accuracy summary
│   ├── sft_chat_template_eval_raw/
│   │   └── ...
│   └── ...
└── report/
    ├── comparison.csv              # All experiments in a table
    ├── accuracy_bar.png            # Bar chart of all accuracies
    ├── accuracy_grouped.png        # Grouped by training method, colored by eval format
    └── eval_template_scatter.png   # Scatter: eval with vs without chat template
```

## Adding a New Task

1. Create a new file in `tasks/` (e.g. `tasks/math500.py`)
2. Subclass `Task` and implement:
   - `name()` — short identifier
   - `load_train()` / `load_test()` — return `list[Sample]`
   - `extract_answer(model_output)` — parse model output
   - `check_answer(extracted, gold)` — compare answers
3. Register it in `tasks/registry.py`:
   ```python
   from tasks.math500 import Math500Task
   TASK_REGISTRY["math500"] = Math500Task
   ```
4. Run: `python run_experiments.py --task math500`
