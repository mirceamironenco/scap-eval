from typing import Callable, Optional, TypeVar

import torch.nn as nn

from scap.configs import BuilderConfig, make_default_builder

LAYER_REGISTRY: dict[str, type[nn.Module]] = dict()
LAYER_CLASS_NAMES: set[str] = set()
LAYER_CONFIG: dict[str, type[BuilderConfig]] = dict()
STATE_MIXER_LAYERS: set[str] = set()
SEQUENCE_MIXER_LAYERS: set[str] = set()

T = TypeVar("T", bound=nn.Module)


def register_layer(
    name: str,
    *,
    config: Optional[type[BuilderConfig]] = None,
    as_statemixer: Optional[bool] = None,
    as_seqmixer: Optional[bool] = None,
) -> Callable[[type[T]], type[T]]:
    def register_layer_cls(cls: type[T]) -> type[T]:
        if name in LAYER_REGISTRY:
            raise ValueError(f"Cannot register layer {name}, already exists.")

        if cls.__name__ in LAYER_CLASS_NAMES:
            raise ValueError(
                f"Cannot register layer with duplicate class name {cls.__name__}"
            )

        LAYER_REGISTRY[name] = cls
        LAYER_CLASS_NAMES.add(cls.__name__)

        # By default, make it available everywhere
        # This is also equivalent to as_statemixer = as_seqmixer = True
        if as_seqmixer is None and as_statemixer is None:
            STATE_MIXER_LAYERS.add(name)
            SEQUENCE_MIXER_LAYERS.add(name)

        if as_statemixer is not None and as_statemixer:
            STATE_MIXER_LAYERS.add(name)

        if as_seqmixer is not None and as_seqmixer:
            SEQUENCE_MIXER_LAYERS.add(name)

        LAYER_CONFIG[name] = config if config is not None else make_default_builder(cls)

        return cls

    return register_layer_cls


def get_layer(name: str) -> type[nn.Module]:
    return LAYER_REGISTRY[name]


def get_layer_cfg(name: str) -> type[BuilderConfig]:
    return LAYER_CONFIG[name]


def registered_layers() -> list[str]:
    return list(LAYER_REGISTRY.keys())[::-1]


def registered_state_mixers() -> list[str]:
    return list(STATE_MIXER_LAYERS)[::-1]


def registered_sequence_mixers() -> list[str]:
    return list(SEQUENCE_MIXER_LAYERS)[::-1]


# Register torch.nn.Identity as both state and sequence mixer
register_layer(name="identity")(nn.Identity)
