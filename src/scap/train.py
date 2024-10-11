from dataclasses import asdict

import torch
import torch.distributed as dist

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

    config.training.setup_distributed()

    config.training.seed_everything()

    # Build/load dataset
    task_builder = TaskDatasetBuilder(
        train_cfg=config.task.train_cfgs,
        test_cfg=config.task.test_cfgs,
        force_generate=False,
        cache_dir=config.training.data_dir,
    )
    train_dataset, test_dataset = task_builder.build_datasets(seed=config.training.seed)

    model = create_decoder_model(
        model_cfg=config.model,
        task=config.task.name,
        data_cfgs=config.task.train_cfgs + config.task.test_cfgs,
    )
    # Save config
    config.save_config()

    # Event writers
    writers: list[MetricWriter] = [LogMetricWriter(logger=_logger)]

    machine = config.training.machine
    if machine.is_primary:
        if config.log_wandb:
            writers.append(
                WandbWriter(
                    log_dir=config.training.log_dir,
                    project=config.training.project_name,
                )
            )

        if config.log_tensorboard:
            writers.append(TensorBoardWriter(log_dir=config.training.log_dir))

        for writer in writers:
            writer.record_config(
                run=config.training.experiment_name, config=asdict(config)
            )

        # Log full experiment config & model
        get_console().rule("Model")
        get_console().print(model)
        get_console().print(f"Total parameters: {count_params(model) / 1e6:.2f}M")
        get_console().rule("")

        get_console().rule("Config")
        get_console().print(config)
        get_console().rule("")

    if machine.distributed:
        dist.barrier(device_ids=[machine.rank])
        get_console().log(
            "Training in distributed mode with 1 device per process."
            f"Process {machine.rank}, total {machine.world_size}, device {machine.device}."
        )
    else:
        get_console().log(
            f"Training with a single process on 1 device ({machine.device})."
        )

    trainer = Trainer(
        cfg=config.training,
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        writers=writers,
    )
    trainer.setup()
    trainer.train()
    dist.destroy_process_group()
