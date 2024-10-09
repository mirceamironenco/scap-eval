from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor
from typing_extensions import override

from scap.configs import TaskDatasetConfig
from scap.data import CROSS_ENTROPY_IGNORE_INDEX, pad_collate


def vocab_pad_collate(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    return pad_collate(
        batch, input_pad_value=0, target_pad_value=CROSS_ENTROPY_IGNORE_INDEX
    )


@dataclass
class DisjointSetCfg(TaskDatasetConfig):
    vocab_size: int = 8192
    short_length: int = 4
    long_length: int = 8
    input_seq_len: int = field(default=0, init=False)

    def __post_init__(self):
        # 3 extra tokens - prefix, sep_lists, sep_answer
        self.input_seq_len = self.short_length + self.long_length + 3

    @override
    def collator(self):
        return vocab_pad_collate

    @override
    def generate(self, seed: int) -> tuple[Tensor, Tensor]:
        return disjoint_sets_impl(
            num_examples=self.num_examples,
            vocab_size=self.vocab_size,
            short_length=self.short_length,
            long_length=self.long_length,
            seed=seed,
        )


def disjoint_sets_impl(
    num_examples: int,
    vocab_size: int,
    short_length: int,
    long_length: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Reference: https://github.com/HazyResearch/prefix-linear-attention"""
    rng = np.random.default_rng(seed=seed)

    # Special toks
    mask_tok, prefix_tok, sep_lists_tok, sep_ans_tok = range(4)
    num_special_tokens = 4

    inputs, labels = [], []
    for _ in range(num_examples):
        # get a short and long list of tokens.
        half_vocab = vocab_size // 2
        all_idx = np.arange(num_special_tokens, vocab_size)
        all_idx_shuffled = rng.permutation(all_idx)
        all_short = all_idx_shuffled[:half_vocab]
        all_long = all_idx_shuffled[half_vocab:]
        short_tokens = rng.choice(all_short, short_length, replace=False)
        long_tokens = rng.choice(all_long, long_length, replace=False)

        # make sure a token in short occurs in long
        overlap_token = short_tokens[rng.integers(short_length)]
        long_tokens[rng.integers(long_length)] = overlap_token
        answer_tok = overlap_token

        # Inputs and outputs
        input_seq = np.concatenate(
            [
                [prefix_tok],
                short_tokens,
                [sep_lists_tok],
                long_tokens,
                [sep_ans_tok],
                [answer_tok],
            ]
        )
        input = torch.tensor(input_seq)

        label = torch.full_like(input[:-1], CROSS_ENTROPY_IGNORE_INDEX)
        label[-1] = input[-1]
        input = input[:-1]

        inputs.append(input)
        labels.append(label)

    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    return inputs, labels


@dataclass
class MQARCfg(TaskDatasetConfig):
    vocab_size: int = 8192
    power_a: float = 0.01
    num_kv_pairs: int = 8
    random_non_queries: bool = True

    @override
    def collator(self):
        return vocab_pad_collate

    @override
    def generate(self, seed: int) -> tuple[Tensor, Tensor]:
        return multiquery_ar(**self.asdict(), seed=seed)


def multiquery_ar(
    num_examples: int,
    input_seq_len: int,
    vocab_size: int,
    seed: int,
    power_a: float = 0.01,
    num_kv_pairs: int = 8,
    random_non_queries: bool = True,
) -> tuple[Tensor, Tensor]:
    """Reference: https://github.com/HazyResearch/zoology"""
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert num_kv_pairs * 4 <= input_seq_len

    rng = np.random.default_rng(seed=seed)

    # two tokens for key and value
    context_size = num_kv_pairs * 2

    # create keys so that each key is present exactly once in each example
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(
        rng.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs
    )

    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(
        rng.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs
    )

    # create sequences
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # compute power law
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()

    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(
        rng.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs
    )

    # queries and answers
    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
    examples = np.concatenate([kvs, queries], axis=1)

    labels = np.full(
        (num_examples, input_seq_len + 1), CROSS_ENTROPY_IGNORE_INDEX, dtype=np.int64
    )
    np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)

    inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])

    if random_non_queries:
        inputs[inputs == 0] = torch.tensor(
            rng.integers(vocab_size, size=inputs.shape)[inputs == 0], dtype=inputs.dtype
        )

    return inputs, labels
