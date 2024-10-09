import logging
import os
from logging import DEBUG, INFO, Formatter, Handler, NullHandler, getLogger
from typing import Optional

import rich
from rich.console import Console
from rich.logging import RichHandler

_CONSOLE: Optional[Console] = None
_ERROR_CONSOLE: Optional[Console] = None


def get_console() -> Console:
    global _CONSOLE

    if _CONSOLE is None:
        _CONSOLE = rich.get_console()
    return _CONSOLE


def get_error_console() -> Console:
    global _ERROR_CONSOLE

    if _ERROR_CONSOLE is None:
        _ERROR_CONSOLE = Console(stderr=True, highlight=False)
    return _ERROR_CONSOLE


def configure_logging(*, debug: bool = False) -> None:
    rank = int(os.environ.get("LOCAL_RANK", 0))

    handlers: list[Handler] = []
    if rank == 0:
        console = get_error_console()

        handler = RichHandler(console=console, show_path=False, keywords=[])
        fmt = Formatter("%(name)s - %(message)s")
        handler.setFormatter(fmt)

        handlers.append(handler)

    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=DEBUG if debug else INFO, handlers=handlers, datefmt=datefmt, force=True
    )

    if rank != 0:
        getLogger().addHandler(NullHandler())


def get_logger() -> logging.Logger:
    return logging.getLogger("scap")
