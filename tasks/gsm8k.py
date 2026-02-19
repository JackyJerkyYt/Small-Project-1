"""GSM8K: Grade-school math word problems."""

import re
from datasets import load_dataset
from tasks.base import Task, Sample


class GSM8KTask(Task):

    def name(self) -> str:
        return "gsm8k"

    def _extract_value(self, answer_text: str) -> str:
        """Extract the number after '####' in the GSM8K answer field."""
        match = re.search(r"####\s*(.+)", answer_text)
        if match:
            return match.group(1).strip().replace(",", "")
        return answer_text.strip()

    def _load_split(self, split: str) -> list[Sample]:
        ds = load_dataset("openai/gsm8k", "main", split=split)
        samples = []
        for row in ds:
            question = row["question"].strip()
            full_answer = row["answer"].strip()
            answer_value = self._extract_value(full_answer)
            samples.append(Sample(
                question=question,
                answer=full_answer,
                answer_value=answer_value,
            ))
        return samples

    def load_train(self) -> list[Sample]:
        return self._load_split("train")

    def load_test(self) -> list[Sample]:
        return self._load_split("test")

    def extract_answer(self, model_output: str) -> str | None:
        # Try #### pattern first (model may mimic training format)
        match = re.search(r"####\s*(.+)", model_output)
        if match:
            return match.group(1).strip().replace(",", "")

        # Fall back to last number in the output
        numbers = re.findall(r"-?\d[\d,]*\.?\d*", model_output)
        if numbers:
            return numbers[-1].replace(",", "")

        return None

    def check_answer(self, extracted: str | None, gold: str) -> bool:
        """Compare extracted answer against gold.
        
        Args:
            extracted: The extracted answer from model output
            gold: Either the full answer text or just the value (we'll extract if needed)
        """
        if extracted is None:
            return False
        gold_value = self._extract_value(gold)
        try:
            return float(extracted) == float(gold_value)
        except ValueError:
            return extracted.strip() == gold_value.strip()
