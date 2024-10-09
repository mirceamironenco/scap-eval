from typing import Annotated, Union

import tyro

from scap.configs import BuilderConfig
from scap.layers import (
    get_layer_cfg,
    registered_layers,
    registered_sequence_mixers,
    registered_state_mixers,
)


def layer_names_constructor() -> str:
    return tyro.extras.literal_type_from_choices(registered_layers())  # type: ignore


def seqmixer_names_constructor() -> str:
    return tyro.extras.literal_type_from_choices(registered_sequence_mixers())  # type: ignore


def statemixer_names_constructor() -> str:
    return tyro.extras.literal_type_from_choices(registered_state_mixers())  # type: ignore


def seqmixer_cli_constructor() -> type[BuilderConfig]:
    layer_cfgs = list(
        Annotated[get_layer_cfg(layer_name), tyro.conf.subcommand(name=layer_name)]
        for layer_name in registered_sequence_mixers()
    )
    return Union[*layer_cfgs]  # type: ignore


def statemixer_cli_constructor() -> type[BuilderConfig]:
    layer_cfgs = list(
        Annotated[get_layer_cfg(layer_name), tyro.conf.subcommand(name=layer_name)]
        for layer_name in registered_state_mixers()
    )
    return Union[*layer_cfgs]  # type: ignore


def layers_cli_constructor() -> type[BuilderConfig]:
    layer_cfgs = list(
        Annotated[get_layer_cfg(layer_name), tyro.conf.subcommand(name=layer_name)]
        for layer_name in registered_layers()
    )
    return Union[*layer_cfgs]  # type: ignore
