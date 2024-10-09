import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import override

from scap.tasks.functions._common import InterleavedFuncConfig


@dataclass
class LinearRegressionCfg(InterleavedFuncConfig):
    input_dim: int = 6
    scale: float = 1.0

    @override
    def generate(self, seed: int) -> tuple[Tensor, Tensor]:
        return linear_regression(**self.asdict(), seed=seed)


def linear_regression(
    num_examples: int,
    input_seq_len: int,
    input_dim: int,
    seed: int,
    scale: float = 1.0,
) -> tuple[Tensor, Tensor]:
    rng = np.random.default_rng(seed=seed)
    xs = torch.tensor(
        rng.normal(size=(num_examples, input_seq_len, input_dim)), dtype=torch.float
    )
    weights = torch.tensor(
        rng.normal(size=(num_examples, input_dim, 1)), dtype=torch.float
    )
    ys = (xs @ weights).squeeze()
    ys = ys * scale
    return xs, ys


@dataclass
class QuadraticRegressionCfg(InterleavedFuncConfig):
    input_dim: int = 6
    scale: float = 1.0

    @override
    def generate(self, seed: int) -> tuple[Tensor, Tensor]:
        return quadratic_regression(**self.asdict(), seed=seed)


def quadratic_regression(
    num_examples: int,
    input_seq_len: int,
    input_dim: int,
    seed: int,
    scale: float = 1.0,
) -> tuple[Tensor, Tensor]:
    rng = np.random.default_rng(seed=seed)
    xs = torch.tensor(
        rng.normal(size=(num_examples, input_seq_len, input_dim)), dtype=torch.float
    )
    weights = torch.tensor(
        rng.normal(size=(num_examples, input_dim, 1)), dtype=torch.float
    )
    ys = ((xs**2) @ weights).squeeze()
    ys = ys / math.sqrt(3.0)
    ys = ys * scale
    return xs, ys


@dataclass
class SparseLinearRegressionCfg(InterleavedFuncConfig):
    input_dim: int = 6
    sparsity: int = 5

    @override
    def generate(self, seed: int) -> tuple[Tensor, Tensor]:
        return sparse_linear_regression(**self.asdict(), seed=seed)


def sparse_linear_regression(
    num_examples: int, input_seq_len: int, input_dim: int, sparsity: int, seed: int
) -> tuple[Tensor, Tensor]:
    assert sparsity < input_dim
    rng = np.random.default_rng(seed=seed)
    xs = rng.normal(size=(num_examples, input_seq_len, input_dim))
    weights = rng.normal(size=(num_examples, input_dim))
    mask = rng.normal(size=weights.shape)
    xs, weights, mask = map(
        lambda x: torch.tensor(x, dtype=torch.float), (xs, weights, mask)
    )

    mask = mask.argsort() < sparsity
    weights = weights.masked_fill(mask, 0.0)
    ys = (xs @ weights[..., None]).squeeze()
    return xs, ys


@dataclass
class LinearClassificationCfg(InterleavedFuncConfig):
    input_dim: int = 10

    @override
    def generate(self, seed: int) -> tuple[Tensor, Tensor]:
        return linear_classification(**self.asdict(), seed=seed)


def linear_classification(
    num_examples: int, input_seq_len: int, input_dim: int, seed: int
) -> tuple[Tensor, Tensor]:
    xs, ys = linear_regression(
        num_examples=num_examples,
        input_seq_len=input_seq_len,
        input_dim=input_dim,
        seed=seed,
    )
    ys = ys.sign()
    return xs, ys


@dataclass
class NoisyLinearRegressionCfg(InterleavedFuncConfig):
    input_dim: int = 10
    noise_std: float = 1.0
    renormalize: bool = False

    @override
    def generate(self, seed: int) -> tuple[Tensor, Tensor]:
        return noisy_linear_regression(**self.asdict(), seed=seed)


def noisy_linear_regression(
    num_examples: int,
    input_seq_len: int,
    input_dim: int,
    seed: int,
    noise_std: float = 1.0,
    renormalize: bool = False,
) -> tuple[Tensor, Tensor]:
    xs, ys = linear_regression(
        num_examples=num_examples,
        input_seq_len=input_seq_len,
        input_dim=input_dim,
        seed=seed,
    )
    rng = np.random.default_rng(seed=seed)
    noise = torch.tensor(rng.normal(size=ys.shape), dtype=torch.float)
    ys = ys + noise * noise_std

    if renormalize:
        ys = ys * math.sqrt(input_dim) / ys.std()

    return xs, ys


@dataclass
class ReluLinearRegressionCfg(InterleavedFuncConfig):
    input_dim: int = 6
    hidden_dim: int = 6
    scale: float = 1.0

    @override
    def generate(self, seed: int) -> tuple[Tensor, Tensor]:
        return relu_two_layer_nn_regression(**self.asdict(), seed=seed)


def relu_two_layer_nn_regression(
    num_examples: int,
    input_seq_len: int,
    input_dim: int,
    seed: int,
    hidden_dim: int,
    scale: float = 1.0,
) -> tuple[Tensor, Tensor]:
    rng = np.random.default_rng(seed=seed)
    xs = rng.normal(size=(num_examples, input_seq_len, input_dim))
    w1 = rng.normal(size=(num_examples, input_dim, hidden_dim))
    w2 = rng.normal(size=(num_examples, hidden_dim, 1))
    xs, w1, w2 = map(lambda x: torch.tensor(x, dtype=torch.float), (xs, w1, w2))

    ys = F.relu(xs @ w1)
    ys = (ys @ w2).squeeze()
    ys = ys * math.sqrt(2.0 / hidden_dim)
    ys = ys * scale
    return xs, ys


@dataclass
class OutlierLinearRegressionCfg(InterleavedFuncConfig):
    input_dim: int = 6
    drop_prob: float = 0.9

    @override
    def generate(self, seed: int) -> tuple[Tensor, Tensor]:
        return manyoutlier_regression(**self.asdict(), seed=seed)


def manyoutlier_regression(
    num_examples: int,
    input_seq_len: int,
    input_dim: int,
    seed: int,
    drop_prob: float = 0.9,
) -> tuple[Tensor, Tensor]:
    xs, ys = linear_regression(
        num_examples=num_examples,
        input_seq_len=input_seq_len,
        input_dim=input_dim,
        seed=seed,
    )
    rng = np.random.default_rng(seed=seed)
    indices = torch.tensor(rng.uniform(size=(num_examples, input_seq_len))) < drop_prob
    xs = xs.masked_fill(indices[..., None].expand(-1, -1, input_dim), 1.0)
    ys = ys.masked_fill(indices, 1.0)
    return xs, ys
