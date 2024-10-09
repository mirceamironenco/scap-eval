from dataclasses import dataclass, field
from typing import Callable, Optional, Type

import torch.nn as nn
from torch import Tensor

from scap.configs import BuilderConfig
from scap.layers._registry import register_layer


@dataclass
class MlpCfg(BuilderConfig):
    _target: Type = field(default_factory=lambda: Mlp)
    dim_inner: Optional[int] = None
    drop_rate: float = 0.0
    act: Callable[[Tensor], Tensor] = nn.GELU(approximate="tanh")
    bias: bool = True


@register_layer("mlp", config=MlpCfg, as_statemixer=True)
class Mlp(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        dim_inner: Optional[int] = None,
        drop_rate: float = 0.0,
        act: Callable[[Tensor], Tensor] = nn.GELU(approximate="tanh"),
        bias: bool = True,
    ) -> None:
        super().__init__()
        dim_inner = dim * 4 if dim_inner is None else dim_inner
        self.fc = nn.Linear(dim, dim_inner, bias=bias)
        self.proj = nn.Linear(dim_inner, dim, bias=bias)
        self.act = act
        self.drop1 = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.drop2 = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

        self.init_weights()

    def init_weights(self, init_std: float = 0.02) -> None:
        for layer in (self.fc, self.proj):
            nn.init.normal_(layer.weight, mean=0.0, std=init_std)

            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.proj(x)
        x = self.drop2(x)
        return x


@dataclass
class GLUCfg(BuilderConfig):
    _target: Type = field(default_factory=lambda: GLU)
    drop_rate: float = 0.0
    act: Callable[[Tensor], Tensor] = nn.Sigmoid()
    bias: bool = False
    multiple_of: int = 16


@register_layer("glu", config=GLUCfg, as_statemixer=True)
class GLU(nn.Module):
    """Gated Linear Unit (GLU).

    Args:
        dim (int): Width of the model.
        drop_rate (float, optional): Dropout rate.
        act (Callable, optional): Activation function for the gate.
        bias (bool, optional): If True, bias is included in linear projections.
        multiple_of (int, optional): Make sure inner width is multiple of this.
    """

    def __init__(
        self,
        dim: int,
        *,
        drop_rate: float = 0.0,
        act: Callable[[Tensor], Tensor] = nn.Sigmoid(),
        bias: bool = False,
        multiple_of: int = 16,
    ) -> None:
        super().__init__()

        self.act = act
        self.multiple_of = multiple_of

        dim_inner = int(2 * dim * 4 / 3)
        dim_inner = self.multiple_of * ((dim_inner + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, dim_inner, bias=bias)
        self.w2 = nn.Linear(dim, dim_inner, bias=bias)
        self.w3 = nn.Linear(dim_inner, dim, bias=bias)

        self.drop1 = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.drop2 = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.drop3 = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

        self.init_weights()

    def init_weights(self, init_std: float = 0.02) -> None:
        for layer in (self.w1, self.w2, self.w3):
            nn.init.normal_(layer.weight, mean=0.0, std=init_std)

            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        w1_out = self.w1(x)
        w2_out = self.w2(x)
        return self.drop3(self.w3(self.drop1(self.act(w1_out)) * self.drop2(w2_out)))


@dataclass
class SwiGLUCfg(BuilderConfig):
    _target: Type = field(default_factory=lambda: SwiGLU)

    drop_rate: float = 0.0
    bias: bool = False
    multiple_of: int = 16


@register_layer("swiglu", config=SwiGLUCfg, as_statemixer=True)
class SwiGLU(GLU):
    """
    Swish-Gated Linear Unit (SwiGLU).

    Args:
        dim (int): Width of the model.
        drop_rate (float, optional): Dropout rate.
        bias (bool, optional): If True, bias is included in linear projections.
        multiple_of (int, optional): Make sure inner width is multiple of this.
    """

    def __init__(
        self,
        dim: int,
        *,
        drop_rate: float = 0.0,
        bias: bool = False,
        multiple_of: int = 16,
    ) -> None:
        super().__init__(
            dim=dim,
            drop_rate=drop_rate,
            act=nn.SiLU(),
            bias=bias,
            multiple_of=multiple_of,
        )
