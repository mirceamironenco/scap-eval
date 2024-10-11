import os
from typing import Annotated

import tyro

from scap.cli._cli_configs import CLIExperimentConfig
from scap.cli._parse_tasks import task_name_cli_constructor
from scap.layers import registered_layers
from scap.logging import configure_logging, get_console
from scap.tasks import registered_tasks, task_datasets
from scap.train import train_recipe

CONSOLE = get_console()


def _cli_tasks() -> None:
    """Print available tasks."""
    CONSOLE.rule("Tasks")
    CONSOLE.print(registered_tasks())
    CONSOLE.rule("")


def run_tasks_cli() -> None:
    tyro.cli(_cli_tasks, use_underscores=True)


def _cli_benchmarks(
    task: Annotated[
        str,
        tyro.conf.arg(constructor_factory=task_name_cli_constructor),
    ],
) -> None:
    """Print datasets/benchmarks belonging to task."""
    CONSOLE.rule(f"{task.title()} task datasets")
    CONSOLE.print(task_datasets(task))
    CONSOLE.rule()


def run_benchmarks_cli() -> None:
    tyro.cli(_cli_benchmarks, use_underscores=True)


def _cli_mixers() -> None:
    """Show available layers"""
    CONSOLE.rule("Mixers")
    CONSOLE.print(registered_layers())
    CONSOLE.rule("")


def run_mixers_cli() -> None:
    tyro.cli(_cli_mixers, use_underscores=True)


@tyro.conf.configure(
    tyro.conf.ConsolidateSubcommandArgs,
    tyro.conf.SuppressFixed,
    tyro.conf.OmitArgPrefixes,
)
def _cli_train(
    config: Annotated[CLIExperimentConfig, tyro.conf.arg(name="")],
) -> None:
    """Train CLI entrypoint."""
    configure_logging()
    train_recipe(config)


def run_train_cli() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    tyro.cli(_cli_train, use_underscores=True, console_outputs=(local_rank == 0))


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    tyro.extras.subcommand_cli_from_dict(
        {
            "tasks": _cli_tasks,
            "train": _cli_train,
            "benchmarks": _cli_benchmarks,
            "mixers": _cli_mixers,
        },
        use_underscores=True,
        console_outputs=(local_rank == 0),
    )


if __name__ == "__main__":
    main()
