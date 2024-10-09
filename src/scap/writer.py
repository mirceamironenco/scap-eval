from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from typing_extensions import override

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    has_tensorboard = False
else:
    has_tensorboard = True

try:
    import wandb
except ImportError:
    has_wandb = False
else:
    has_wandb = True


class MetricWriter(ABC):
    @abstractmethod
    def record_metrics(
        self, run: str, values: dict[str, Any], step_nr: int, *, flush: bool = True
    ) -> None: ...

    def record_config(self, run: str, config: dict[str, Any]) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class LogMetricWriter(MetricWriter):
    def __init__(self, logger: logging.Logger) -> None:
        self._log = logger

    @override
    def record_metrics(
        self, run: str, values: dict[str, Any], step_nr: int, *, flush: bool = True
    ) -> None:
        if not self._log.isEnabledFor(logging.INFO):
            return

        formatted_values = []
        for name, value in values.items():
            if name.endswith(("loss", "metric")):
                formatted_values.append(f"{name}: {value:.3f}")
            elif name == "lr":
                formatted_values.append(f"{name}: {value:.5f}")
            else:
                formatted_values.append(f"{name}: {value}")

        s = " | ".join(formatted_values)
        self._log.info(s)

    @override
    def record_config(self, run: str, config: dict[str, Any]) -> None:
        pass

    @override
    def close(self) -> None:
        pass


class TensorBoardWriter(MetricWriter):
    _log_dir: Path
    _writers: dict[str, SummaryWriter]

    def __init__(self, log_dir: Path) -> None:
        if not has_tensorboard:
            raise ImportError(
                "tensorboard not found. Install with `pip install tensorboard`"
            )

        self._log_dir = log_dir
        self._writers = {}

    @override
    def record_metrics(
        self, run: str, values: dict[str, Any], step_nr: int, *, flush: bool = True
    ) -> None:
        writer = self._get_writer(run)

        for name, value in values.items():
            writer.add_scalar(name, value, step_nr)

        if flush:
            writer.flush()

    @override
    def record_config(self, run: str, config: dict[str, Any]) -> None:
        writer = self._get_writer(run)
        writer.add_text("config", str(config))

    def _get_writer(self, run: str) -> SummaryWriter:
        try:
            writer = self._writers[run]
        except KeyError:
            writer = SummaryWriter(self._log_dir)
            self._writers[run] = writer

        return writer

    @override
    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()

        self._writers.clear()


class WandbWriter(MetricWriter):
    _log_dir: Path
    _project: str

    def __init__(self, log_dir: Path, project: str) -> None:
        if not has_wandb:
            raise ImportError("wandb not installed. Install with `pip install wandb`")

        self._log_dir = log_dir
        self._project = project

        self._run = None

    @override
    def record_metrics(
        self, run: str, values: dict[str, Any], step_nr: int, *, flush: bool = True
    ) -> None:
        self._get_run(run).log(data=values, step=step_nr, commit=flush)

    @override
    def record_config(self, run: str, config: dict[str, Any]) -> None:
        self._get_run(run).config.update(config, allow_val_change=True)

    def _get_run(self, run: str):
        if self._run is None:
            self._run = wandb.init(
                project=self._project,
                dir=self._log_dir,
                name=run,
            )

        if self._run.name != run:
            raise NotImplementedError("Multi-run setting not implemented for wandb.")

        return self._run

    @override
    def close(self) -> None:
        if self._run is not None:
            self._run.finish()
