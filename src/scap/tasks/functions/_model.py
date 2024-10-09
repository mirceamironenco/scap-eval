from typing import final

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import override

from scap.configs import ModelConfig
from scap.data import FLOAT_MASK_VALUE
from scap.model import (
    TransformerDecoderModel,
    TransformerFrontend,
)
from scap.tasks import register_task_model
from scap.tasks.functions._benchmarks import BOOLEAN_TASK, REGRESSION_TASK
from scap.tasks.recall import TransformerBuilder


@final
class RegressionDecoderModel(TransformerDecoderModel):
    @override
    def project(self, x: Tensor) -> Tensor:
        x = self.final_proj(x)
        x = x.squeeze()[:, ::2]
        return x

    @override
    def compute_loss(self, preds: Tensor, targets: Tensor) -> Tensor:
        loss_mask = (targets != FLOAT_MASK_VALUE).type_as(preds)
        loss = F.mse_loss(preds, targets, reduction="none")
        return (loss * loss_mask).sum() / loss_mask.sum()

    @override
    def compute_metric(self, preds: Tensor, targets: Tensor) -> Tensor:
        return self.compute_loss(preds, targets)


class RegressionModelBuilder(TransformerBuilder):
    def build_frontend(self) -> TransformerFrontend:
        embedding = nn.Linear(self.cfg.input_dim, self.cfg.model_dim, bias=True)

        pos_encoder = None
        if self.cfg.pos_enc is not None:
            if self.cfg.pos_enc == "learnable":
                pos_encoder = nn.Embedding(self.cfg.max_seq_len, self.cfg.model_dim)

        return TransformerFrontend(
            self.cfg.model_dim,
            embed=embedding,
            pos_encoder=pos_encoder,
            dropout_p=self.cfg.embed_dropout,
        )

    def build_model(self) -> RegressionDecoderModel:
        frontend = self.build_frontend()
        decoder = self.build_decoder()
        final_proj = nn.Linear(self.cfg.model_dim, 1)

        model = RegressionDecoderModel(
            frontend, decoder, final_proj, max_seq_len=self.cfg.max_seq_len
        )
        return model


@final
class DiscreteDecoderModel(TransformerDecoderModel):
    @override
    def project(self, x: Tensor) -> Tensor:
        x = self.final_proj(x)
        x = x.squeeze()[:, ::2]
        return x

    @override
    def compute_loss(self, preds: Tensor, targets: Tensor) -> Tensor:
        loss_mask = (targets != FLOAT_MASK_VALUE).type_as(preds)
        targets = (targets + 1.0) / 2.0
        loss = F.binary_cross_entropy_with_logits(preds, targets, reduction="none")
        return (loss * loss_mask).sum() / loss_mask.sum()

    @override
    def compute_metric(self, preds: Tensor, targets: Tensor) -> Tensor:
        preds = preds.sign()
        mask = targets != FLOAT_MASK_VALUE
        accuracy = (preds == targets)[mask].float().mean()
        return accuracy


class DiscreteModelBuilder(TransformerBuilder):
    @override
    def build_frontend(self) -> TransformerFrontend:
        embedding = nn.Linear(self.cfg.input_dim, self.cfg.model_dim, bias=True)

        pos_encoder = None
        if self.cfg.pos_enc is not None:
            if self.cfg.pos_enc == "learnable":
                pos_encoder = nn.Embedding(self.cfg.max_seq_len, self.cfg.model_dim)

        return TransformerFrontend(
            self.cfg.model_dim,
            embed=embedding,
            pos_encoder=pos_encoder,
            dropout_p=self.cfg.embed_dropout,
        )

    @override
    def build_model(self) -> DiscreteDecoderModel:
        frontend = self.build_frontend()
        decoder = self.build_decoder()
        final_proj = nn.Linear(self.cfg.model_dim, 1)

        model = DiscreteDecoderModel(
            frontend, decoder, final_proj, max_seq_len=self.cfg.max_seq_len
        )
        return model


def create_regression_model(config: ModelConfig) -> RegressionDecoderModel:
    builder = RegressionModelBuilder(config)
    return builder.build_model()


def create_discrete_model(config: ModelConfig) -> DiscreteDecoderModel:
    builder = DiscreteModelBuilder(config)
    return builder.build_model()


register_task_model(task_name=REGRESSION_TASK, model_builder=create_regression_model)
register_task_model(task_name=BOOLEAN_TASK, model_builder=create_discrete_model)
