"""Base class for all tasks. Extend this to add new benchmarks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Sample:
    """A single question-answer sample."""
    question: str
    answer: str  # the full answer text (for SFT training)
    answer_value: str | None = None  # the extracted value for answer checking (e.g., the number)


class Task(ABC):
    """
    To add a new task:
      1. Create a file in tasks/ (e.g. tasks/my_task.py)
      2. Subclass Task and implement the three abstract methods
      3. Register it in tasks/registry.py
    """

    @abstractmethod
    def name(self) -> str:
        """Short identifier for the task (e.g. 'gsm8k')."""
        ...

    @abstractmethod
    def load_train(self) -> list[Sample]:
        """Return training samples."""
        ...

    @abstractmethod
    def load_test(self) -> list[Sample]:
        """Return test / evaluation samples."""
        ...

    @abstractmethod
    def extract_answer(self, model_output: str) -> str | None:
        """Parse the model's free-form output and return the extracted answer string.
        Return None if no answer could be parsed."""
        ...

    @abstractmethod
    def check_answer(self, extracted: str | None, gold: str) -> bool:
        """Return True if the extracted answer matches the gold answer."""
        ...
