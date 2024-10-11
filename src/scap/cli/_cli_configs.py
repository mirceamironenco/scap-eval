from dataclasses import dataclass
from typing import Annotated

import tyro

from scap.cli._parse_layers import seqmixer_cli_constructor, statemixer_cli_constructor
from scap.cli._parse_tasks import task_cfg_cli_constructor
from scap.configs import (
    BuilderConfig,
    ExperimentConfig,
    ModelConfig,
    TaskConfig,
    TrainConfig,
)

CLITaskConfig = Annotated[
    TaskConfig,
    tyro.conf.arg(constructor_factory=task_cfg_cli_constructor),
]

CLISeqMixerConfig = Annotated[
    BuilderConfig, tyro.conf.arg(constructor_factory=seqmixer_cli_constructor)
]

CLIStateMixerConfig = Annotated[
    BuilderConfig,
    tyro.conf.arg(constructor_factory=statemixer_cli_constructor),
]


@dataclass
class CLIModelConfig(ModelConfig):
    seqmixer: CLISeqMixerConfig
    """Sequence mixer options."""

    statemixer: CLIStateMixerConfig
    """State mixer options."""


@dataclass
class CLIExperimentConfig(ExperimentConfig):
    task: CLITaskConfig
    model: CLIModelConfig
    training: TrainConfig
