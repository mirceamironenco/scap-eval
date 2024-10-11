import typing
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, TypeVar

import torch
import torch.distributed as dist
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP

AutoCast: TypeAlias = AbstractContextManager[None | autocast]

T = TypeVar("T", bound=Module)


def get_rank() -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return 0
    return dist.get_rank()


def count_params(module: Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def ddp_wrap_module(*, module: T, device_ids: list[int] | torch.device) -> T:
    """DDP wrapper utility which preserves type."""
    return typing.cast(T, DDP(module, device_ids=device_ids))


def ddp_unwrapper(ddp_or_model: DDP | T) -> T:
    """If DDP, then return the .module. Otherwise, return the model."""
    if isinstance(ddp_or_model, DDP):
        return typing.cast(T, ddp_or_model.module)
    return ddp_or_model


def cuda_amp_scaler(
    amp_dtype: Literal["float16", "bfloat16"],
) -> tuple[AutoCast, GradScaler | None]:
    if amp_dtype not in ("float16", "bfloat16"):
        raise ValueError(f"Expected float16/bfloat16 amp dtype, got {amp_dtype}")

    scaler = None
    dtype = torch.bfloat16
    if amp_dtype == "float16":
        dtype = torch.float16
        scaler = GradScaler(device="cuda")
    amp_autocast = autocast(device_type="cuda", dtype=dtype)
    return amp_autocast, scaler


def listify(x: Any) -> list[Any]:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


@dataclass
class AverageMetric:
    count: int = 0
    last_value: float = 0.0
    total: float = 0.0
    average: float = 0.0

    def reset(self) -> None:
        self.count = 0
        self.last_value = 0.0
        self.total = 0.0
        self.average = 0.0

    def update(self, value: float, count: int = 1) -> None:
        self.count += count
        self.last_value = value
        self.total += value * count
        self.average = self.total / self.count
