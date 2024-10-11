from typing import Annotated, Optional, TypeAlias, Union

import tyro

from scap.configs import TaskConfig
from scap.tasks import registered_tasks, task_datasets


def _make_cli_task_config_type(task: str) -> TaskConfig:
    DatasetName = tyro.extras.literal_type_from_choices(task_datasets(task))

    def _build_task_configs(
        train_data: list[DatasetName],  # type: ignore
        train_size: list[int],
        test_data: Annotated[
            Optional[list[DatasetName]],  # type: ignore
            tyro.conf.arg(help_behavior_hint="(Default: set to --train_data.)"),
        ] = None,
        test_size: list[int] = [3000],
    ) -> TaskConfig:
        if len(train_data) != len(train_size):
            raise ValueError(
                f"Specified different number of task datasets and sizes: {len(train_data)} ({train_data}) != {len(train_size)} ({train_size})"
            )

        if test_data is None:
            test_data = train_data[:]

        return TaskConfig.from_benchmarks(
            task_name=task,
            train_datasets=train_data,
            train_sizes=train_size,
            test_datasets=test_data,
            test_sizes=test_size,
        )

    return Annotated[
        TaskConfig,
        tyro.conf.subcommand(
            name=task,
            constructor=_build_task_configs,
        ),  # type: ignore
    ]


TaskName: TypeAlias = str


def task_cfg_cli_constructor() -> type[TaskConfig]:
    return Union[*list(map(_make_cli_task_config_type, registered_tasks()))]  # type: ignore


def task_name_cli_constructor() -> type[str]:
    return tyro.extras.literal_type_from_choices(registered_tasks())  # type: ignore
