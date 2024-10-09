from scap.tasks._registry import (
    get_dataset,
    get_task_model,
    register_dataset,
    register_task,
    register_task_model,
    registered_tasks,
    task_dataset_decorator,
    task_datasets,
)

# isort: split

# Register default tasks & their datasets
import scap.tasks.functions
import scap.tasks.recall
