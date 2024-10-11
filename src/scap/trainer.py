from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Optional

import torch
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from scap.configs import TrainConfig
from scap.data import TaskDataset
from scap.logging import get_logger
from scap.model import DecoderModel
from scap.utils import (
    AutoCast,
    AverageMetric,
    GradScaler,
    cuda_amp_scaler,
    ddp_unwrapper,
    ddp_wrap_module,
)
from scap.writer import MetricWriter

_logger = get_logger()


class Trainer:
    cfg: TrainConfig
    _model: DecoderModel
    train_dataset: TaskDataset
    test_dataset: TaskDataset
    train_loader: DataLoader[tuple[Tensor, Tensor]]
    test_loader: DataLoader[tuple[Tensor, Tensor]]
    optimizer: Optimizer
    lr_scheduler: Optional[LRScheduler]
    device: torch.device
    writers: Sequence[MetricWriter]

    _amp_autocast: AutoCast
    _scaler: Optional[GradScaler]

    def __init__(
        self,
        *,
        cfg: TrainConfig,
        model: DecoderModel,
        train_dataset: TaskDataset,
        test_dataset: TaskDataset,
        writers: Sequence[MetricWriter],
    ) -> None:
        self.cfg = cfg
        self._model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.writers = writers

        self.machine = self.cfg.machine
        self.device = torch.device(self.machine.device)

        self._model.to(self.device)

        if self.cfg.machine.distributed:
            self._model = ddp_wrap_module(
                module=self._model, device_ids=[self.machine.local_rank]
            )

        self._scaler = None

        self._amp_autocast = nullcontext()

    def setup(self) -> None:
        self.optimizer = self.cfg.optimizer.build(self.model.parameters())

        if self.cfg.lr_scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.max_epochs,
                eta_min=self.cfg.min_lr,  # type: ignore
            )
        else:
            self.lr_scheduler = None

        if self.cfg.torchcompile:
            _logger.info(f"Compiling model with backend: {self.cfg.torchcompile}.")
            self.model.compile(backend=self.cfg.torchcompile)

        if self.cfg.amp and self.device.type == "cuda":
            self._amp_autocast, self._scaler = cuda_amp_scaler(self.cfg.amp_dtype)

        train_sampler, test_sampler = None, None

        if self.machine.distributed:
            train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            test_sampler = DistributedSampler(self.test_dataset, shuffle=False)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collator(),
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=self.cfg.num_workers,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.cfg.eval_batch_size,
            collate_fn=self.test_dataset.collator(),
            shuffle=False,
            sampler=test_sampler,
            num_workers=self.cfg.num_workers,
        )

    def train(self) -> list[dict[str, float]]:
        _logger.info(
            f"Starting training. Logs will be saved to {self.cfg.log_dir.absolute()}"
        )

        self._model.train()

        results = []
        loss_metric = AverageMetric()
        global_step = 0
        for epoch in range(self.cfg.max_epochs):
            loss_metric.reset()

            if self.machine.distributed:
                if isinstance(self.train_loader.sampler, DistributedSampler):
                    self.train_loader.sampler.set_epoch(epoch)

            for batch in self.train_loader:
                loss = self.train_step(batch)
                global_step += 1

                if self.machine.distributed:
                    loss = self.machine.reduce_tensor(loss)

                loss_metric.update(loss.item(), batch[0].size(0))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            results.append(
                {
                    "epoch": epoch,
                    "train_loss": loss_metric.average,
                    "lr": self._average_lr(),
                }
            )
            should_stop = False
            if epoch % self.cfg.eval_freq == 0:
                eval_dict = self.evaluate()

                results[-1] |= eval_dict

                should_stop = self._should_stop(eval_dict["eval_metric"])

            for writer in self.writers:
                writer.record_metrics(
                    run=self.cfg.experiment_name,
                    values=results[-1],
                    step_nr=global_step,
                )

            if should_stop:
                _logger.info(
                    f"Stoping early, metric value reached {self.cfg.early_stop_threshold}"
                )
                break

        for writer in self.writers:
            writer.close()

        return results

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self._model.eval()

        eval_loss_metric = AverageMetric()
        eval_metric = AverageMetric()
        for batch in self.test_loader:
            loss, metric = self.eval_step(batch)

            if self.machine.distributed:
                loss, metric = map(
                    lambda x: self.machine.reduce_tensor(x), (loss, metric)
                )

            eval_loss_metric.update(loss.item(), batch[0].size(0))
            eval_metric.update(metric.item(), batch[0].size(0))

        self._model.train()
        return {
            "eval_loss": eval_loss_metric.average,
            "eval_metric": eval_metric.average,
        }

    def train_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        inputs, targets = map(lambda x: x.to(self.device), batch)
        with self._amp_autocast:
            output = self._model(inputs)
            loss = self.model.compute_loss(output, targets)

        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        return loss

    def eval_step(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        inputs, targets = map(lambda x: x.to(self.device), batch)
        with self._amp_autocast:
            output = self._model(inputs)
            loss = self.model.compute_loss(output, targets)
            metric = self.model.compute_metric(output, targets)
        return loss, metric

    @property
    def model(self) -> DecoderModel:
        return ddp_unwrapper(self._model)

    def _should_stop(self, value: float) -> bool:
        if (
            self.cfg.early_stop_mode is not None
            and self.cfg.early_stop_threshold is not None
        ):
            if self.cfg.early_stop_mode == "max":
                return value >= self.cfg.early_stop_threshold

            if self.cfg.early_stop_mode == "min":
                return value <= self.cfg.early_stop_threshold

        return False

    def _average_lr(self) -> float:
        return sum(gr["lr"] for gr in self.optimizer.param_groups) / len(
            self.optimizer.param_groups
        )
