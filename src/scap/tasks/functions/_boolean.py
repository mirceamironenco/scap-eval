from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from typing_extensions import override

from scap.tasks.functions._common import InterleavedFuncConfig


@dataclass
class DNF3Cfg(InterleavedFuncConfig):
    input_dim: int = 6

    @override
    def generate(self, seed: int) -> tuple[Tensor, Tensor]:
        return DNF3(**self.asdict(), seed=seed)


def DNF3(
    num_examples: int, input_seq_len: int, input_dim: int, seed: int
) -> tuple[Tensor, Tensor]:
    """3-term DNF. Reference: https://github.com/satwik77/incontext-bool"""
    rng = np.random.default_rng(seed=seed)

    xs = torch.tensor(
        rng.integers(0, 2, size=(num_examples, input_seq_len, input_dim)),
        dtype=torch.float,
    )
    xs = xs * 2.0 - 1.0

    weights = [
        torch.tensor(
            rng.choice(
                [0, 1, -1], size=(num_examples, input_dim, 1), p=[0.8, 0.1, 0.1]
            ),
            dtype=torch.float,
        )
        for _ in range(3)
    ]
    kw = [torch.norm(weights[index], p=1, dim=1) - 1 for index in range(3)]

    for b in range(num_examples):
        cid = rng.choice([0, 1, 2])  # Choose a clause
        wb, k = weights[cid][b], kw[cid][b]
        pidx = [i for i in range(input_dim) if wb[i] == 1.0]
        nidx = [i for i in range(input_dim) if wb[i] == -1.0]
        for i in range(input_seq_len):
            if rng.choice([0, 1], p=[0.65, 0.35]):
                xs[b, i, pidx] = +1.0
                xs[b, i, nidx] = -1.0
                assert (xs[b, i, :] @ wb).squeeze() >= k

    ys = [(xs @ weights[index]).squeeze() - kw[index] for index in range(3)]
    ys = [ys[i].sign() for i in range(3)]

    ys = torch.stack(ys, dim=2).max(dim=2)[0]
    ys = ys.sign()
    return xs, ys


@dataclass
class CNF3Cfg(InterleavedFuncConfig):
    input_dim: int = 6

    @override
    def generate(self, seed: int) -> tuple[Tensor, Tensor]:
        return CNF3(**self.asdict(), seed=seed)


def CNF3(
    num_examples: int, input_seq_len: int, input_dim: int, seed: int
) -> tuple[Tensor, Tensor]:
    """3-term CNF. Reference: https://github.com/satwik77/incontext-bool"""

    rng = np.random.default_rng(seed=seed)
    xs = torch.tensor(
        rng.integers(0, 2, size=(num_examples, input_seq_len, input_dim)),
        dtype=torch.float,
    )
    xs = xs * 2.0 - 1.0

    weights = [
        torch.tensor(
            rng.choice(
                [0, 1, -1], size=(num_examples, input_dim, 1), p=[0.8, 0.1, 0.1]
            ),
            dtype=torch.float,
        )
        for _ in range(3)
    ]
    kw = [torch.norm(weights[index], p=1, dim=1) - 1 for index in range(3)]

    for b in range(num_examples):
        cid = rng.choice([0, 1, 2])  # Choose a clause
        wb, k = weights[cid][b], kw[cid][b]
        pidx = [i for i in range(input_dim) if wb[i] == 1.0]
        nidx = [i for i in range(input_dim) if wb[i] == -1.0]
        for i in range(input_seq_len):
            if rng.choice([0, 1], p=[0.65, 0.35]):
                xs[b, i, pidx] = -1.0
                xs[b, i, nidx] = +1.0
                assert (xs[b, i, :] @ wb).squeeze() < -k

    ys = [(xs @ weights[index]).squeeze() + kw[index] for index in range(3)]
    ys = [ys[i].sign() for i in range(3)]

    ys = torch.stack(ys, dim=2).min(dim=2)[0]
    ys = ys.sign()
    return xs, ys


@dataclass
class IntHalfspaceCfg(InterleavedFuncConfig):
    input_dim: int = 6

    @override
    def generate(self, seed: int) -> tuple[Tensor, Tensor]:
        return int_halfspace(**self.asdict(), seed=seed)


def int_halfspace(
    num_examples: int, input_seq_len: int, input_dim: int, seed: int
) -> tuple[Tensor, Tensor]:
    """Reference: https://github.com/satwik77/incontext-bool"""
    rng = np.random.default_rng(seed=seed)
    xs = torch.tensor(
        rng.integers(0, 2, size=(num_examples, input_seq_len, input_dim)),
        dtype=torch.float,
    )
    xs = xs * 2.0 - 1.0
    weights = torch.tensor(
        rng.integers(-3, 4, (num_examples, input_dim, 1)), dtype=torch.float
    )
    ys = (xs @ weights).squeeze() - 0.5
    ys = ys.sign()
    return xs, ys


@dataclass
class SparseThresholdCfg(InterleavedFuncConfig):
    input_dim: int = 6

    @override
    def generate(self, seed: int) -> tuple[Tensor, Tensor]:
        return sparse_threshold(**self.asdict(), seed=seed)


def sparse_threshold(
    num_examples: int, input_seq_len: int, input_dim: int, seed: int
) -> tuple[Tensor, Tensor]:
    """Reference: https://github.com/satwik77/incontext-bool"""
    rng = np.random.default_rng(seed=seed)

    xs = torch.tensor(
        rng.integers(0, 2, (num_examples, input_seq_len, input_dim)), dtype=torch.float
    )
    xs = xs * 2.0 - 1.0
    weights = torch.tensor(
        rng.choice([0, 1, -1], size=(num_examples, input_dim, 1), p=[0.7, 0.15, 0.15]),
        dtype=torch.float,
    )
    kw = (
        torch.tensor(rng.integers(-3, 3, size=(num_examples, 1)), dtype=torch.float)
        + 0.5
    )
    ys = (xs @ weights).squeeze() - kw
    ys = ys.sign()
    return xs, ys


@dataclass
class ParityCfg(InterleavedFuncConfig):
    input_dim: int = 6

    @override
    def generate(self, seed: int) -> tuple[Tensor, Tensor]:
        return parity(**self.asdict(), seed=seed)


def parity(
    num_examples: int, input_seq_len: int, input_dim: int, seed: int
) -> tuple[Tensor, Tensor]:
    """Reference: https://github.com/satwik77/incontext-bool"""
    rng = np.random.default_rng(seed=seed)
    xs = torch.tensor(
        rng.integers(0, 2, size=(num_examples, input_seq_len, input_dim)),
        dtype=torch.float,
    )
    # Generate for 35% of indices to be 1
    weights = torch.zeros(num_examples, input_dim)
    funcs = rng.choice(2**input_dim, size=num_examples)
    subsets = [
        [j for j in range(input_dim) if (i & 1 << j)] for i in range(2**input_dim)
    ]
    for index in range(num_examples):
        weights[index, subsets[funcs[index]]] = 1.0
    ys = (xs @ weights[..., None]).squeeze() % 2
    ys = (ys * 2.0 - 1.0).sign()
    return xs, ys


@dataclass
class SparseParityCfg(InterleavedFuncConfig):
    input_dim: int = 6
    k: int = 2

    @override
    def generate(self, seed: int) -> tuple[Tensor, Tensor]:
        return sparse_parity(**self.asdict(), seed=seed)


def sparse_parity(
    num_examples: int, input_seq_len: int, input_dim: int, k: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: https://github.com/satwik77/incontext-bool"""
    rng = np.random.default_rng(seed=seed)

    # NB: In MambaFormer, the mask is sampled only once for the entire batch and
    # reused. In the original paper (above) each example has a different mask.
    xs = torch.tensor(
        rng.integers(0, 2, size=(num_examples, input_seq_len, input_dim)),
        dtype=torch.float,
    )
    weights = torch.zeros(num_examples, input_dim)
    for index in range(num_examples):
        mask = torch.tensor(rng.choice(input_dim, k, replace=False))
        weights[index, mask] = 1.0
    ys = (xs @ weights[..., None]).squeeze() % 2
    ys = (ys * 2.0 - 1.0).sign()
    return xs, ys


@dataclass
class DisjunctionCfg(InterleavedFuncConfig):
    input_dim: int = 6

    @override
    def generate(self, seed: int) -> tuple[Tensor, Tensor]:
        return disjunction(**self.asdict(), seed=seed)


def disjunction(
    num_examples: int, input_seq_len: int, input_dim: int, seed: int
) -> tuple[Tensor, Tensor]:
    """Reference: https://github.com/satwik77/incontext-bool"""
    rng = np.random.default_rng(seed=seed)
    xs = torch.tensor(
        rng.integers(0, 2, size=(num_examples, input_seq_len, input_dim)),
        dtype=torch.float,
    )
    xs = xs * 2.0 - 1.0

    weights = torch.tensor(
        rng.choice([0, 1, -1], size=(num_examples, input_dim, 1), p=[0.7, 0.15, 0.15]),
        dtype=torch.float,
    )
    kw = torch.norm(weights, p=1, dim=1) - 1

    for b in range(num_examples):
        wb, k = weights[b], kw[b]
        pidx = [i for i in range(input_dim) if wb[i] == 1.0]

        nidx = [i for i in range(input_dim) if wb[i] == -1.0]

        for i in range(input_seq_len):
            if rng.choice([0, 1], p=[0.7, 0.3]):
                xs[b, i, pidx] = -1.0
                xs[b, i, nidx] = 1.0
                assert (xs[b, i, :] @ wb).squeeze() < -k

    ys = (xs @ weights).squeeze() + kw
    ys = ys.sign()
    return xs, ys


@dataclass
class ConjunctionCfg(InterleavedFuncConfig):
    input_dim: int = 6

    @override
    def generate(self, seed: int) -> tuple[Tensor, Tensor]:
        return conjunction(**self.asdict(), seed=seed)


def conjunction(
    num_examples: int, input_seq_len: int, input_dim: int, seed: int
) -> tuple[Tensor, Tensor]:
    """Reference: https://github.com/satwik77/incontext-bool"""
    rng = np.random.default_rng(seed=seed)
    xs = torch.tensor(
        rng.integers(0, 2, size=(num_examples, input_seq_len, input_dim)),
        dtype=torch.float,
    )
    xs = xs * 2.0 - 1.0

    weights = torch.tensor(
        rng.choice([0, 1, -1], size=(num_examples, input_dim, 1), p=[0.7, 0.15, 0.15]),
        dtype=torch.float,
    )
    kw = torch.norm(weights, p=1, dim=1) - 1

    for b in range(num_examples):
        wb, k = weights[b], kw[b]
        pidx = [i for i in range(input_dim) if wb[i] == 1.0]
        nidx = [i for i in range(input_dim) if wb[i] == -1.0]

        for i in range(input_seq_len):
            if rng.choice([0, 1], p=[0.7, 0.3]):
                xs[b, i, pidx] = 1.0
                xs[b, i, nidx] = -1.0
                assert (xs[b, i, :] @ wb).squeeze() >= k

    ys = (xs @ weights).squeeze() - kw
    ys = ys.sign()
    return xs, ys
