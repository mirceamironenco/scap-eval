from dataclasses import asdict
from pathlib import Path

import torch

from scap.configs import ExperimentConfig
from scap.data import TaskDatasetBuilder
from scap.logging import get_console, get_logger
from scap.model_builder import create_decoder_model
from scap.trainer import Trainer
from scap.utils import count_params
from scap.writer import LogMetricWriter, MetricWriter, TensorBoardWriter, WandbWriter

_logger = get_logger()


def train_recipe(config: ExperimentConfig) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    config.training.set_run_name()

    config.training.seed_everything()

    # Build/load dataset
    task_builder = TaskDatasetBuilder(
        train_cfg=config.task.train_cfgs,
        eval_cfg=config.task.test_cfgs,
        force_generate=False,
        cache_dir=config.training.data_dir,
    )
    train_dataset, eval_dataset = task_builder.build_datasets(seed=config.training.seed)

    model = create_decoder_model(
        model_cfg=config.model,
        task=config.task.name,
        data_cfgs=config.task.train_cfgs + config.task.test_cfgs,
    )

    # Event writers
    writers: list[MetricWriter] = [LogMetricWriter(logger=_logger)]

    if config.log_wandb:
        if not Path.exists(config.training.log_dir):
            Path.mkdir(config.training.log_dir, exist_ok=True, parents=True)

        writers.append(
            WandbWriter(
                log_dir=config.training.log_dir, project=config.training.project_name
            )
        )

    if config.log_tensorboard:
        writers.append(TensorBoardWriter(log_dir=config.training.log_dir))

    for writer in writers:
        writer.record_config(run=config.training.experiment_name, config=asdict(config))

    # Save config
    config.save_config()

    # Log full experiment config & model
    if config.training.machine.rank == 0:
        get_console().rule("Model")
        get_console().print(model)
        get_console().print(f"Total parameters: {count_params(model) / 1e6:.2f}M")
        get_console().rule("")

        get_console().rule("Config")
        get_console().print(config)
        get_console().rule("")

    trainer = Trainer(
        cfg=config.training,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        writers=writers,
    )
    trainer.setup()
    trainer.train()
