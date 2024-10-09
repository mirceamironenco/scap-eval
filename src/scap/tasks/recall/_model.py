from typing import final

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import override

from scap.configs import ModelConfig
from scap.data import CROSS_ENTROPY_IGNORE_INDEX
from scap.model import (
    TiedProjection,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerDecoderModel,
    TransformerFrontend,
)
from scap.tasks import register_task_model
from scap.tasks.recall._benchmarks import RECALL_TASK


@final
class RecallDecoderModel(TransformerDecoderModel):
    @override
    def project(self, x: Tensor) -> Tensor:
        x = self.final_proj(x)
        return x

    @override
    def compute_loss(self, preds: Tensor, targets: Tensor) -> Tensor:
        logits, targets = preds.flatten(0, 1), targets.flatten(0, 1)
        return F.cross_entropy(logits, targets, ignore_index=CROSS_ENTROPY_IGNORE_INDEX)

    @override
    def compute_metric(self, preds: Tensor, targets: Tensor) -> Tensor:
        preds = preds.argmax(dim=-1)
        mask = targets != CROSS_ENTROPY_IGNORE_INDEX
        accuracy = (preds == targets)[mask].float().mean()
        return accuracy


class TransformerBuilder:
    cfg: ModelConfig

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg

    def build_frontend(self) -> TransformerFrontend:
        embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.model_dim)

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

    def build_decoder(self) -> TransformerDecoder:
        layers = [self.build_decoder_layer() for _ in range(self.cfg.n_layers)]
        return TransformerDecoder(layers, self.cfg.model_dim)

    def build_decoder_layer(self) -> TransformerDecoderLayer:
        sequence_mixer = self.cfg.seqmixer.build(dim=self.cfg.model_dim)
        state_mixer = self.cfg.statemixer.build(dim=self.cfg.model_dim)
        return TransformerDecoderLayer(
            sequence_mixer=sequence_mixer,
            state_mixer=state_mixer,
            model_dim=self.cfg.model_dim,
        )

    def build_model(self) -> RecallDecoderModel:
        frontend = self.build_frontend()
        decoder = self.build_decoder()

        if self.cfg.tie_weights:
            final_proj = TiedProjection(weight=frontend.embed.weight)  # type: ignore
        else:
            final_proj = nn.Linear(self.cfg.model_dim, self.cfg.vocab_size, bias=False)

        model = RecallDecoderModel(
            frontend, decoder, final_proj, max_seq_len=self.cfg.max_seq_len
        )
        return model


def create_transformer_model(config: ModelConfig) -> RecallDecoderModel:
    builder = TransformerBuilder(config)
    return builder.build_model()


register_task_model(task_name=RECALL_TASK, model_builder=create_transformer_model)
