from typing import Final

from scap.tasks import register_task, task_dataset_decorator
from scap.tasks.functions._boolean import (
    CNF3Cfg,
    ConjunctionCfg,
    DisjunctionCfg,
    DNF3Cfg,
    IntHalfspaceCfg,
    ParityCfg,
    SparseParityCfg,
    SparseThresholdCfg,
)
from scap.tasks.functions._regression import (
    LinearClassificationCfg,
    LinearRegressionCfg,
    NoisyLinearRegressionCfg,
    OutlierLinearRegressionCfg,
    QuadraticRegressionCfg,
    ReluLinearRegressionCfg,
    SparseLinearRegressionCfg,
)

BOOLEAN_TASK: Final = "boolean"

register_task(name=BOOLEAN_TASK)
discrete_dataset = task_dataset_decorator(task_name=BOOLEAN_TASK)


@discrete_dataset()
def linear_cl_task(num_examples: int) -> LinearClassificationCfg:
    return LinearClassificationCfg(
        num_examples=num_examples, input_seq_len=10, input_dim=6
    )


@discrete_dataset()
def dnf3(num_examples: int) -> DNF3Cfg:
    return DNF3Cfg(num_examples=num_examples)


@discrete_dataset()
def cnf3(num_examples: int) -> CNF3Cfg:
    return CNF3Cfg(num_examples=num_examples)


@discrete_dataset()
def int_halfspace(num_examples: int) -> IntHalfspaceCfg:
    return IntHalfspaceCfg(num_examples=num_examples)


@discrete_dataset()
def sparse_thresh(num_examples: int) -> SparseThresholdCfg:
    return SparseThresholdCfg(num_examples=num_examples)


@discrete_dataset()
def parity(num_examples: int) -> ParityCfg:
    return ParityCfg(num_examples=num_examples)


@discrete_dataset()
def sparse_parity(num_examples: int) -> SparseParityCfg:
    return SparseParityCfg(num_examples=num_examples)


@discrete_dataset()
def conjunction(num_examples: int) -> ConjunctionCfg:
    return ConjunctionCfg(num_examples=num_examples)


@discrete_dataset()
def disjunction(num_examples: int) -> DisjunctionCfg:
    return DisjunctionCfg(num_examples=num_examples)


REGRESSION_TASK: Final = "regression"

register_task(name=REGRESSION_TASK)
regression_dataset = task_dataset_decorator(task_name=REGRESSION_TASK)


@regression_dataset()
def lr_task(num_examples: int) -> LinearRegressionCfg:
    return LinearRegressionCfg(
        num_examples=num_examples, input_dim=6, input_seq_len=30, scale=1.0
    )


@regression_dataset()
def curriculum_regression(num_examples: int) -> list[LinearRegressionCfg]:
    start_len, end_len, step = 11, 41, 2
    seq_lens = list(range(start_len, end_len, step))
    subset_size, last_size = divmod(num_examples, len(seq_lens))
    subsets = [subset_size] * len(seq_lens)

    if last_size > 0:
        subsets[-1] = last_size

    cfgs = []
    for examples, seq_len in zip(subsets, seq_lens):
        cfgs.append(
            LinearRegressionCfg(
                num_examples=examples, input_dim=6, input_seq_len=seq_len
            )
        )

    return cfgs


@regression_dataset()
def quadratic_lr_task(num_examples: int) -> QuadraticRegressionCfg:
    return QuadraticRegressionCfg(
        num_examples=num_examples, input_seq_len=20, scale=1.0
    )


@regression_dataset()
def sparse_lr_task(num_examples: int) -> SparseLinearRegressionCfg:
    return SparseLinearRegressionCfg(
        num_examples=num_examples, input_seq_len=10, sparsity=5
    )


@regression_dataset()
def noisylr_task(num_examples: int) -> NoisyLinearRegressionCfg:
    return NoisyLinearRegressionCfg(num_examples=num_examples, input_seq_len=10)


@regression_dataset()
def relu2nn_regression(num_examples: int) -> ReluLinearRegressionCfg:
    return ReluLinearRegressionCfg(
        num_examples=num_examples, input_seq_len=10, scale=1.0
    )


@regression_dataset()
def outlier_regression(num_examples: int) -> OutlierLinearRegressionCfg:
    return OutlierLinearRegressionCfg(
        num_examples=num_examples, input_seq_len=10, drop_prob=0.9
    )
