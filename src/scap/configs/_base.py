import hashlib
import json
import typing
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Optional, Type

from torch import Tensor
from torch.utils.data import default_collate


@dataclass
class BuilderConfig:
    """Config class which instantiates objects of class `_target`.

    Attributes:
        _target (Type): The target class to be instantiated.

    Methods:
        build(**kwargs): Instantiates the target class with the config.

    Usage:
        By default, calling `.build(**kwargs)` is equivalent to:

        ```python
        self._target(**{**asdict(self), **kwargs})
        ```

        To pass the entire config object to the target class instead of individual
        field values, one can subclass BuilderConfig and override the build method:

        ```python
        @dataclass
        class LayerConfig(BuilderConfig):
            _target: Type = field(default_factory=lambda: LayerClass)

            # Options
            ...

            def build(self, **kwargs) -> Any:
                return self._target(self, **kwargs)
        ```

        This allows the target class to receive the entire configuration object,
        which can be useful for more complex initialization scenarios.
    """

    _target: Type

    def _get_fields(self) -> dict[str, Any]:
        fields = asdict(self)
        fields.pop("_target")

        # Eliminate None-valued items to allow class constructor defaults.
        fields = {k: v for (k, v) in fields.items() if v is not None}
        return fields

    def build(self, **kwargs) -> Any:
        """Returns an instance of the target class `_target` built with the config.

        Args:
            **kwargs: Additional keyword arguments to override in the config.

        Returns:
            An instance of the target class `_target`.
        """
        fields = self._get_fields()

        # Override with options specified from kwargs (take priority over 'fields').
        fields |= kwargs

        return self._target(**fields)


def make_default_builder(target: Type) -> type[BuilderConfig]:
    """Creates a no-attribute BuilderConfig.

    Args:
        target (Type): Target class for which to create a builder.

    Returns:
        type[BuilderConfig]: Subclass of BuilderConfig that builds `target`.
    """

    @dataclass
    class DefaultBuilderConfig(BuilderConfig):
        _target: Type = target

    return DefaultBuilderConfig


@dataclass
class TaskDatasetConfig(ABC):
    num_examples: int
    """Number of elements in the dataset."""

    input_seq_len: int = 64
    """Length of the sequence of each element."""

    vocab_size: Optional[int] = field(init=False, default=None)
    """Vocabulary size. Dataset cfgs using it should make this an init=True field."""

    input_dim: Optional[int] = field(init=False, default=None)
    """Size of input samples. For non-vocab tasks, inputs are (bsz, seqlen, input_dim).
    Dataset cfgs using it should make this an init=True field."""

    @abstractmethod
    def generate(self, seed: int) -> tuple[Tensor, Tensor]:
        """Generates (inputs, targets) for map-style dataset given a random seed.

        Args:
            seed (int): Random seed used to generate the dataset.
        """

    def input_transform(self, inputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        """Input transformation to be applied to individual samples."""

        return inputs, targets

    def collator(self) -> Optional[Callable]:
        """A collator function merges a list of samples to form a mini-batch."""

        return default_collate

    def asdict(self) -> dict[str, Any]:
        """Equivalent to dataclasses.asdict(self) with None values ommitted."""

        return {k: v for (k, v) in asdict(self).items() if v is not None}

    def filename(self, seed: int) -> str:
        """Construct a unique filename from the cfg, seed and class.
        _classname is needed for uniqueness in case of identical cfgs.

        Args:
            seed (int): Same random seed used to generate the dataset.
        """
        fields = {**self.asdict(), "_seed": seed, "_classname": self.__class__.__name__}
        encoding = json.dumps(fields, sort_keys=True).encode()
        data_hash = hashlib.md5(encoding).hexdigest()
        data_file = f"data_{data_hash}.pt"
        return data_file


@dataclass
class ModelConfig:
    seqmixer: BuilderConfig
    """Config which builds a sequence mixer layer."""

    statemixer: BuilderConfig
    """Config which builds a state mixer layer."""

    model_dim: int = 128
    """Embedding dimension to use for the model."""

    n_layers: int = 2
    """Numer of layers (blocks) to use for the model."""

    tie_weights: bool = True
    """Wether to reuse embedding weights for output projection."""

    pos_enc: Optional[Literal["learnable"]] = "learnable"
    """Type of positional encoding to use."""

    embed_dropout: float = 0.0
    """Dropout rate to apply post-embrdding."""

    max_seq_len: int = field(init=False, default=-1)
    """Maximum sequence length. Can also be set from task info using 'set_input_info'"""

    vocab_size: int = field(init=False, default=-1)
    """Vocabulary size. If not applicable to current task, defaults to -1."""

    input_dim: int = field(init=False, default=-1)
    """Size of the input for non-vocabulary tasks. If not applicable defaults to -1."""

    def set_input_info(self, data_cfgs: list[TaskDatasetConfig]) -> None:
        """Set (max_seq_len, vocab_size, input_dim) based on task data configs.

        Args:
            data_cfgs (list[TaskDatasetConfig]): Configs defining train+eval datasets.

        Raises:
            ValueError: If dataset configs vary in vocab_size or input_dim.
        """
        max_seq_len = max(config.input_seq_len for config in data_cfgs)
        input_dim = vocab_size = -1

        if all(data_cfg.input_dim is not None for data_cfg in data_cfgs):
            input_dim = typing.cast(int, data_cfgs[0].input_dim)

            if not all(data_cfg.input_dim == input_dim for data_cfg in data_cfgs):
                raise ValueError("Dataset with variable input_dim not allowed.")

        if all(data_cfg.vocab_size is not None for data_cfg in data_cfgs):
            vocab_size = typing.cast(int, data_cfgs[0].vocab_size)

            if not all(data_cfg.vocab_size == vocab_size for data_cfg in data_cfgs):
                raise ValueError("Dataset with variable vocab_size not allowed.")

        # NB: x2 for tasks which interleave inputs and targets (e.g. regression)
        # For now, we do this for all non-vocab tasks.
        if vocab_size == -1:
            max_seq_len *= 2

        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.input_dim = input_dim
