"""MATH: Competition mathematics (AMC, AIME, etc.) with step-by-step solutions."""

import re
from datasets import load_dataset, concatenate_datasets
from tasks.base import Task, Sample

# EleutherAI mirror splits by subject; load all and concatenate for full MATH.
MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def _extract_boxed(text: str) -> str | None:
    """Extract content of the last \\boxed{...} in text, handling nested braces."""
    pattern = r"\\boxed\s*\{"
    start = text.rfind("\\boxed")
    if start == -1:
        return None
    match = re.match(r"\\boxed\s*\{", text[start:])
    if not match:
        return None
    brace_start = start + match.end()
    depth = 1
    i = brace_start
    while i < len(text) and depth > 0:
        if text[i] == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif text[i] == "}" and (i == 0 or text[i - 1] != "\\"):
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[brace_start : i - 1].strip()


def _normalize_answer(s: str) -> str:
    """Normalize for comparison: strip whitespace and collapse spaces."""
    return " ".join(s.strip().split())


class MATHTask(Task):
    """MATH dataset: competition math problems with \\boxed{} final answers."""

    def name(self) -> str:
        return "math"

    def _load_split(self, split: str) -> list[Sample]:
        # Use EleutherAI mirror (hendrycks/competition_math is no longer on the Hub).
        # Dataset is split by subject; load all configs and concatenate.
        parts = [
            load_dataset("EleutherAI/hendrycks_math", subj, split=split)
            for subj in MATH_SUBJECTS
        ]
        ds = concatenate_datasets(parts)
        samples = []
        for row in ds:
            question = row["problem"].strip()
            full_answer = row["solution"].strip()
            answer_value = _extract_boxed(full_answer)
            samples.append(
                Sample(
                    question=question,
                    answer=full_answer,
                    answer_value=answer_value,
                )
            )
        return samples

    def load_train(self) -> list[Sample]:
        return self._load_split("train")

    def load_test(self) -> list[Sample]:
        return self._load_split("test")

    def extract_answer(self, model_output: str) -> str | None:
        return _extract_boxed(model_output)

    def check_answer(self, extracted: str | None, gold: str) -> bool:
        gold_value = _extract_boxed(gold) if "\\boxed" in gold else gold
        if extracted is None or gold_value is None:
            return False
        return _normalize_answer(extracted) == _normalize_answer(gold_value)
