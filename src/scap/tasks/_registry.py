from collections import defaultdict
from typing import Callable, DefaultDict, Optional, TypeAlias, TypeVar, overload

from scap.configs import TaskDatasetConfig
from scap.model import DecoderModel

DATASET_REGISTRY: dict[str, Callable] = {}
DATASET_TASK_NAME_REGISTRY: dict[str, str] = {}
TASK_REGISTRY: set[str] = set()
MODEL_TASK_REGISTRY: dict[str, Callable[..., DecoderModel]] = {}
DATASET_TASK_INV_REGISTRY: DefaultDict[str, list[str]] = defaultdict(list)


def register_task(name: str) -> None:
    if name in TASK_REGISTRY:
        raise ValueError(f"Cannot register task {name}, already exists.")

    TASK_REGISTRY.add(name)


def register_task_model(
    *, task_name: str, model_builder: Callable[..., DecoderModel]
) -> None:
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"{task_name} task is not registered. Register it first.")

    MODEL_TASK_REGISTRY[task_name] = model_builder


def get_task_model(*, task_name: str) -> Callable[..., DecoderModel]:
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"{task_name} task is not registered. Register it first")

    if task_name not in MODEL_TASK_REGISTRY:
        raise ValueError(f"{task_name} has no model builder registered.")

    return MODEL_TASK_REGISTRY[task_name]


T = TypeVar("T", bound=TaskDatasetConfig)
ListT: TypeAlias = T | list[T]


def task_dataset_decorator(task_name: str):
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"{task_name} task is not registered. Register it first.")

    @overload
    def decorator(
        name: Optional[str],
    ) -> Callable[[Callable[[int], ListT]], Callable[[int], ListT]]: ...

    @overload
    def decorator() -> Callable[[Callable[[int], ListT]], Callable[[int], ListT]]: ...

    def decorator(
        name: Optional[str] = None,
    ) -> Callable[[Callable[[int], ListT]], Callable[[int], ListT]]:
        return register_dataset(task_name=task_name, name=name)

    return decorator


def register_dataset(
    *, task_name: str, name: Optional[str] = None
) -> Callable[[Callable[[int], ListT]], Callable[[int], ListT]]:
    # Register new task if it does not exist
    # Note: This decorator cannot be used without args, e.g. @register_dataset
    if task_name not in TASK_REGISTRY:
        register_task(task_name)

    def register_dataset_fn(
        dataset_fn: Callable[[int], ListT],
    ) -> Callable[[int], ListT]:
        dataset_name = dataset_fn.__name__ if name is None else name

        if dataset_name in DATASET_REGISTRY:
            raise ValueError(f"Cannot register dataset {dataset_name}, already exists.")

        DATASET_REGISTRY[dataset_name] = dataset_fn
        DATASET_TASK_NAME_REGISTRY[dataset_name] = task_name
        DATASET_TASK_INV_REGISTRY[task_name].append(dataset_name)
        return dataset_fn

    return register_dataset_fn


def get_dataset(
    name: str,
) -> Callable[[int], TaskDatasetConfig | list[TaskDatasetConfig]]:
    return DATASET_REGISTRY[name]


def registered_tasks() -> list[str]:
    return list(TASK_REGISTRY)[::-1]


def task_datasets(task: str) -> list[str]:
    if task not in TASK_REGISTRY:
        raise KeyError(f"No task named {task} exists.")
    return DATASET_TASK_INV_REGISTRY[task][::-1]
