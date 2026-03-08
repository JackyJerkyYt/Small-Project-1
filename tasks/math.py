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
    """Normalize a LaTeX math answer string for robust comparison.

    Follows conventions from the MATH benchmark grading scripts:
    strip cosmetic LaTeX wrappers, unify fraction commands, remove
    spacing commands, etc.
    """
    s = s.strip()

    # Unify fraction variants
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\tfrac", "\\frac")

    # Remove \left / \right (purely cosmetic sizing)
    s = s.replace("\\left", "")
    s = s.replace("\\right", "")

    # Remove LaTeX spacing commands
    for cmd in ("\\!", "\\,", "\\;", "\\:", "\\quad", "\\qquad", "\\ "):
        s = s.replace(cmd, "")

    # Unwrap text-style commands: \text{...} → ..., etc.
    s = re.sub(r"\\(?:text|textbf|textit|textrm|mathrm|mathbf|operatorname)\s*\{([^}]*)\}", r"\1", s)

    # Strip dollar signs and stray percent signs used as literals
    s = s.replace("$", "")
    s = s.replace("\\%", "")

    # Collapse whitespace
    s = " ".join(s.split())
    return s


def _try_parse_number(s: str) -> float | None:
    """Try to parse a string as a number, return None on failure."""
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _is_equiv(a: str, b: str) -> bool:
    """Check if two normalized answer strings are equivalent."""
    if a == b:
        return True

    # Numeric fallback: "0.5" == "1/2" == ".5"
    na, nb = _try_parse_number(a), _try_parse_number(b)
    if na is not None and nb is not None:
        return abs(na - nb) < 1e-8

    # Simple fraction evaluation: \frac{p}{q} → p/q
    frac_re = re.compile(r"^\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}$")
    ma, mb = frac_re.match(a), frac_re.match(b)
    vals = []
    for s, m in [(a, ma), (b, mb)]:
        if m:
            num = _try_parse_number(m.group(1))
            den = _try_parse_number(m.group(2))
            vals.append(num / den if num is not None and den and den != 0 else None)
        else:
            vals.append(_try_parse_number(s))
    if vals[0] is not None and vals[1] is not None:
        return abs(vals[0] - vals[1]) < 1e-8

    return False


class MATHTask(Task):
    """MATH dataset: competition math problems with \\boxed{} final answers."""

    def name(self) -> str:
        return "math"

    def _load_split(self, split: str) -> list[Sample]:
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
        return _is_equiv(_normalize_answer(extracted), _normalize_answer(gold_value))
