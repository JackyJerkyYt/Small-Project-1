"""Central registry for tasks. Add new tasks here."""

from tasks.gsm8k import GSM8KTask
from tasks.math import MATHTask

# Maps task name -> task class. To add a task, import and add it here.
TASK_REGISTRY: dict[str, type] = {
    "gsm8k": GSM8KTask,
    "math": MATHTask,
}


def get_task(name: str):
    if name not in TASK_REGISTRY:
        available = ", ".join(TASK_REGISTRY.keys())
        raise ValueError(f"Unknown task '{name}'. Available: {available}")
    return TASK_REGISTRY[name]()
