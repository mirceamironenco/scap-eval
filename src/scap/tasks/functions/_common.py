from dataclasses import dataclass

import torch
from torch import Tensor
from typing_extensions import override

from scap.configs import TaskDatasetConfig
from scap.data import FLOAT_MASK_VALUE, pad_collate


def interleave_tokens(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Interleave tokens from input features and target values.

    Interleaves each input feature vector (x) with its corresponding target value (y),
    Allows sequence models to process both the input features and target values in
    context. Doubles the sequence length & zero-pads to match (x)'s last dimension.

    Example:
        >>> x = torch.tensor([[1, 2], [3, 4], [5, 6]])  # Shape: (3, 2)
        >>> y = torch.tensor([10, 20, 30])  # Shape: (3,)
        >>> x, y = interleave_tokens(x, y)
        >>> print(x) # Shape: (6, 2)
        tensor([[ 1,  2],
                [10,  0],
                [ 3,  4],
                [20,  0],
                [ 5,  6],
                [30,  0]])
        >>> preds = model(x) # Shape: (6, 2)
        >>> loss = loss_fn(preds[..., ::2, 0], y) # Calc. loss on non-target positions.

    Args:
        x (Tensor): (*batch_dims, seq_len, dim).
        y (Tensor): (*batch_dims, seq_len).

    Returns:
        tuple[Tensor, Tensor]:
            - Interleaved tensor with shape (*batch_dims, seq_len * 2, dim).
            - Original y tensor, unchanged.
    """
    dim = x.shape[-1]
    zeros = torch.zeros((*x.shape[:-1], dim - 1), device=x.device, dtype=x.dtype)
    padded_y = torch.cat((y[..., None], zeros), dim=len(x.shape) - 1)
    xy = torch.stack((x, padded_y), dim=len(x.shape) - 1)  # (*, sqlen, 2, dim)
    x = xy.view((*xy.shape[:-3], xy.shape[-3] * xy.shape[-2], xy.shape[-1]))
    return x, y


def regression_pad_collate(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    return pad_collate(batch, input_pad_value=0.0, target_pad_value=FLOAT_MASK_VALUE)


@dataclass
class InterleavedFuncConfig(TaskDatasetConfig):
    @override
    def input_transform(self, inputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        return interleave_tokens(inputs, targets)

    @override
    def collator(self):
        return regression_pad_collate
