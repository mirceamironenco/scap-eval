from dataclasses import dataclass
from typing import Any

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.optimizer import ParamsT as ParamsType

from scap.configs._base import BuilderConfig


@dataclass
class OptimizerConfig(BuilderConfig):
    _target: type[Optimizer]
    lr: float = 1e-3
    weight_decay: float = 1e-1

    def build(self, params: ParamsType) -> Any:
        """Instantiate and return optimizer using config attributes and given params."""

        fields = self._get_fields()
        return self._target(params=params, **fields)


@dataclass
class AdamOptimizerConfig(OptimizerConfig):
    _target: type[Optimizer] = torch.optim.Adam  # type: ignore
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class AdamWOptimizerConfig(OptimizerConfig):
    _target: type[Optimizer] = torch.optim.AdamW  # type: ignore
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SGDOptimizerConfig(OptimizerConfig):
    _target: type[Optimizer] = torch.optim.SGD  # type: ignore
