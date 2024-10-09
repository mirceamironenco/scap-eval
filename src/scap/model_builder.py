from typing import Optional

import torch

from scap.configs import ModelConfig, TaskDatasetConfig
from scap.model import DecoderModel
from scap.tasks import get_task_model


def create_decoder_model(
    *,
    model_cfg: ModelConfig,
    task: str,
    data_cfgs: Optional[list[TaskDatasetConfig]] = None,
    device: Optional[torch.device] = None,
) -> DecoderModel:
    if data_cfgs is not None:
        model_cfg.set_input_info(data_cfgs)

    model_builder = get_task_model(task_name=task)
    model = model_builder(model_cfg)

    if device is not None:
        model = model.to(device)

    return model
