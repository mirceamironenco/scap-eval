import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Self, TypeAlias

import numpy as np
import torch
import yaml

from scap.configs._base import ModelConfig, TaskDatasetConfig
from scap.configs._torch import AdamWOptimizerConfig
from scap.tasks import get_dataset

TorchBackend: TypeAlias = Literal[*torch._dynamo.list_backends()]  # type: ignore


@dataclass
class MachineConfig:
    device: str = "cuda"
    """Device type to run on (e.g., "cuda", "cpu")."""

    distributed: bool = field(init=False, default=False)
    """Whether the experiment is running in distributed mode."""

    world_size: int = field(init=False, default=1)
    """Total number of processes in distributed training."""

    rank: int = field(init=False, default=0)
    """Global rank of the current process."""

    local_rank: int = field(init=False, default=0)
    """Local rank of the current process within its node."""

    @property
    def is_primary(self) -> bool:
        """Indicates if this is the primary process (rank 0)."""

        return self.rank == 0


@dataclass
class TrainConfig:
    machine: MachineConfig
    """Config for the machine we are running on."""

    optimizer: AdamWOptimizerConfig
    """Optimizer."""

    lr_scheduler: bool = True
    """Whether to use a lr scheduler."""

    min_lr: float = 0.0
    """Minimum learning rate."""

    max_epochs: int = 50
    """Maximum number of training epochs."""

    seed: int = 123
    """Random seed for reproducibility."""

    eval_freq: int = 5
    """Evaluation frequency in epochs."""

    num_workers: int = 4
    """Number of dataloader workers."""

    torchcompile: Optional[TorchBackend] = None
    """TorchDynamo backend used for torch.compile."""

    amp: bool = True
    """Whether to use Automatic Mixed Precision."""

    amp_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    """Data type for AMP."""

    early_stop_mode: Optional[Literal["max", "min"]] = None
    """Stop early if 'max' and value >= threshold or 'min' and value <= threshold"""

    early_stop_threshold: Optional[float] = 0.995
    """Threshold for early stopping."""

    batch_size: int = 256
    """Training batch size."""

    eval_batch_size: int = 250
    """Evaluation batch size."""

    output_dir: Path = Path("outputs")
    """Directory to store data/logs for the current run."""

    project_name: str = "scap_eval"
    """Project name."""

    experiment_name: str = "{unset}"
    """Experiment name."""

    run_name: str = "{timestamp}"
    """Run name."""

    def set_run_name(self) -> None:
        if self.run_name == "{timestamp}":
            self.run_name = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    def set_experiment_name(self, name: Optional[str] = None) -> None:
        if self.experiment_name == "{unset}":
            if name is not None:
                self.experiment_name = name
            else:
                self.experiment_name = "unset_experiment"

    @property
    def data_dir(self) -> Path:
        return self.output_dir / Path("data")

    @property
    def log_dir(self) -> Path:
        self.set_experiment_name()
        return Path(f"{self.output_dir}/{self.experiment_name}/{self.run_name}")

    def seed_everything(self) -> None:
        seed = int(self.seed) + self.machine.local_rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@dataclass
class TaskConfig:
    name: str
    train_cfgs: list[TaskDatasetConfig]
    test_cfgs: list[TaskDatasetConfig]
    train_datasets: Optional[list[str]] = None
    test_datasets: Optional[list[str]] = None

    @classmethod
    def from_benchmarks(
        cls,
        task_name: str,
        train_datasets: list[str],
        train_sizes: list[int],
        test_datasets: list[str],
        test_sizes: list[int],
    ) -> Self:
        assert len(train_datasets) == len(train_sizes)
        assert len(test_datasets) == len(test_sizes)
        train_cfgs = []

        for name, size in zip(train_datasets, train_sizes):
            data_cfg = get_dataset(name)(size)

            if not isinstance(data_cfg, list):
                data_cfg = [data_cfg]

            train_cfgs += data_cfg

        test_cfgs = []

        for name, size in zip(test_datasets, test_sizes):
            data_cfg = get_dataset(name)(size)

            if not isinstance(data_cfg, list):
                data_cfg = [data_cfg]

            test_cfgs += data_cfg

        return cls(
            name=task_name,
            train_cfgs=train_cfgs,
            test_cfgs=test_cfgs,
            train_datasets=train_datasets,
            test_datasets=test_datasets,
        )


@dataclass
class ExperimentConfig:
    training: TrainConfig
    task: TaskConfig
    model: ModelConfig
    log_wandb: bool = False
    log_tensorboard: bool = False

    def save_config(self) -> None:
        run_dir = self.training.log_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        config_yaml_path = run_dir / "config.yaml"
        config_yaml_path.write_text(yaml.dump(self), "utf8")
