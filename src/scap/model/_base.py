from abc import ABC, abstractmethod
from typing import Callable

import torch.nn as nn
from torch import Tensor


class DecoderModel(nn.Module, ABC):
    __call__: Callable[[Tensor], Tensor]

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def decode(self, x: Tensor) -> Tensor: ...

    @abstractmethod
    def project(self, x: Tensor) -> Tensor: ...

    @abstractmethod
    def compute_loss(self, preds: Tensor, targets: Tensor) -> Tensor: ...

    @abstractmethod
    def compute_metric(self, preds: Tensor, targets: Tensor) -> Tensor: ...

    def forward(self, x: Tensor) -> Tensor:
        x = self.decode(x)
        return self.project(x)
