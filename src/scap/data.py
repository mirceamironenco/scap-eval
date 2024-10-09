from __future__ import annotations

import bisect
import itertools
from pathlib import Path
from typing import Callable, Final, Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from typing_extensions import override

from scap.configs import TaskDatasetConfig
from scap.logging import get_logger
from scap.utils import listify

_logger = get_logger()

CROSS_ENTROPY_IGNORE_INDEX = -100
FLOAT_MASK_VALUE = 65504.0  # max representable in float16


def pad_collate(
    batch: list[tuple[Tensor, Tensor]],
    *,
    input_pad_value: int | float,
    target_pad_value: int | float,
) -> tuple[Tensor, Tensor]:
    """Collator that right-pads a list of (input, targets) to have the same length."""

    inputs = [x[0] for x in batch]
    targets = [x[1] for x in batch]
    inputs = pad_sequence(inputs, batch_first=True, padding_value=input_pad_value)
    targets = pad_sequence(targets, batch_first=True, padding_value=target_pad_value)
    return inputs, targets


class TaskDataset(Dataset[tuple[Tensor, Tensor]]):
    inputs: Tensor
    targets: Tensor

    @classmethod
    def from_cfg(
        cls,
        config: TaskDatasetConfig,
        *,
        seed: int,
        force_generate: bool = False,
        cache_dir: Optional[Path | str] = None,
    ) -> TaskDataset:
        """Create or load a cached task dataset from its config.

        Args:
            config (TaskDatasetConfig): Config with `generate()` implemented.
            seed (int): Random seed used for generation.
            force_generate (bool): Whether to regenerate if cached dataset exists.
            cache_dir (Path or str, optional): Location to cache generated dataset.
        """
        cache_path = None
        if cache_dir is not None:
            Path(cache_dir).mkdir(exist_ok=True, parents=True)
            cache_path = Path(cache_dir) / config.filename(seed)

            if not force_generate and cache_path.exists():
                _logger.info(f"Loading data from on-disk cache path {cache_path}.")
                return cls(**torch.load(cache_path, weights_only=True), cfg=config)

        _logger.info("Generating dataset..")
        inputs, targets = config.generate(seed=seed)
        if cache_path is not None:
            _logger.info(f"Saving generated dataset to on-disk cache at {cache_path}.")
            torch.save(dict(inputs=inputs, targets=targets), f=cache_path)

        return cls(inputs=inputs, targets=targets, cfg=config)

    def __init__(
        self, *, inputs: Tensor, targets: Tensor, cfg: TaskDatasetConfig
    ) -> None:
        super().__init__()
        if inputs.size(0) != targets.size(0):
            raise ValueError(
                f"Inputs/targets shape mismatch: {inputs.size(0)} != {targets.size(0)}"
            )

        self.inputs, self.targets = inputs, targets
        self.cfg = cfg

    def __getitem__(self, index: int | slice) -> tuple[Tensor, Tensor]:
        inputs, targets = self.inputs[index], self.targets[index]
        inputs, targets = self.cfg.input_transform(inputs, targets)
        return inputs, targets

    def __len__(self) -> int:
        return len(self.targets)

    def __add__(self, other: TaskDataset) -> ConcatTaskDataset:
        return ConcatTaskDataset([self, other])

    def collator(self) -> Optional[Callable]:
        return self.cfg.collator()


class ConcatTaskDataset(TaskDataset):
    datasets: list[TaskDataset]
    cumulative_sizes: list[int]

    @classmethod
    def from_cfgs(
        cls,
        configs: list[TaskDatasetConfig],
        *,
        seeds: list[int],
        regenerate: bool = False,
        cache_dir: Optional[Path | str] = None,
    ) -> ConcatTaskDataset:
        assert len(configs) == len(seeds)
        datasets = [
            TaskDataset.from_cfg(
                cfg, seed=seed, force_generate=regenerate, cache_dir=cache_dir
            )
            for (cfg, seed) in zip(configs, seeds)
        ]
        return ConcatTaskDataset(datasets=datasets)

    def __init__(self, datasets: list[TaskDataset]) -> None:
        self.datasets = datasets
        self.cumulative_sizes = list(
            itertools.accumulate(map(len, self.datasets), initial=0)
        )

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        dataset_index = bisect.bisect_right(self.cumulative_sizes, index) - 1
        sample_index = index - self.cumulative_sizes[dataset_index]
        return self.datasets[dataset_index][sample_index]

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    @property
    def cfg(self) -> list[TaskDatasetConfig]:
        return [dataset.cfg for dataset in self.datasets]

    @override
    def collator(self) -> Optional[Callable]:
        # For now, this supports/assumes datasets with the same collator.
        return self.cfg[0].collator()


class TaskDatasetBuilder:
    MAX_SEED: Final[int] = 100_000
    train_cfgs: list[TaskDatasetConfig]
    eval_cfgs: list[TaskDatasetConfig]

    def __init__(
        self,
        train_cfg: TaskDatasetConfig | list[TaskDatasetConfig],
        eval_cfg: TaskDatasetConfig | list[TaskDatasetConfig],
        force_generate: bool = False,
        cache_dir: Optional[Path | str] = None,
    ) -> None:
        self.train_cfgs = listify(train_cfg)
        self.eval_cfgs = listify(eval_cfg)
        self.force_generate = force_generate
        self.cache_dir = cache_dir

    def _build_dataset(
        self, cfgs: list[TaskDatasetConfig], seeds: list[int]
    ) -> TaskDataset:
        if len(cfgs) == 1:
            return TaskDataset.from_cfg(
                config=cfgs[0],
                seed=seeds[0],
                force_generate=self.force_generate,
                cache_dir=self.cache_dir,
            )
        else:
            return ConcatTaskDataset.from_cfgs(
                cfgs,
                seeds=seeds,
                regenerate=self.force_generate,
                cache_dir=self.cache_dir,
            )

    def build_datasets(self, *, seed: int) -> tuple[TaskDataset, TaskDataset]:
        """Generate training and eval datasets.

        Args:
            seed (int): Random seed used to generate individual dataset seeds.
        """
        train_seeds, eval_seeds = self.generate_seeds(
            base_seed=seed, num_train=len(self.train_cfgs), num_eval=len(self.eval_cfgs)
        )
        train_dataset = self._build_dataset(self.train_cfgs, train_seeds)
        eval_dataset = self._build_dataset(self.eval_cfgs, eval_seeds)
        return train_dataset, eval_dataset

    @classmethod
    def generate_seeds(
        cls, *, base_seed: int, num_train: int, num_eval: int
    ) -> tuple[list[int], list[int]]:
        """Generate unique random seeds for dataset creation.

        Args:
            base_seed (int): Base random seed used to generate other seeds.
            num_train (int): Number of training seeds (1 per dataset).
            num_eval (int): Number of eval seeds (1 per dataset).
        """
        assert cls.MAX_SEED >= num_train + num_eval
        rng = np.random.default_rng(seed=base_seed)
        all_seeds = rng.choice(cls.MAX_SEED, size=num_train + num_eval, replace=False)
        train_seeds, eval_seeds = all_seeds[:num_train], all_seeds[num_train:]
        assert not (set(train_seeds) & set(eval_seeds))
        return train_seeds.tolist(), eval_seeds.tolist()
