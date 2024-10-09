from __future__ import annotations

from enum import Enum
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from typing_extensions import override

from scap.model._base import DecoderModel


class TransformerNormOrder(Enum):
    """Controls when to apply Layer Norm."""

    POST = 0
    """Apply LN after each layer's residual connection."""

    PRE = 1
    """Apply LN at the start each layer."""


class TransformerDecoderModel(DecoderModel):
    decoder_frontend: TransformerFrontend
    decoder: TransformerDecoder
    final_proj: nn.Linear | TiedProjection
    max_seq_len: int

    def __init__(
        self,
        decoder_frontend: TransformerFrontend,
        decoder: TransformerDecoder,
        final_proj: nn.Linear | TiedProjection,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.decoder_frontend = decoder_frontend
        self.decoder = decoder
        self.final_proj = final_proj
        self.max_seq_len = max_seq_len

        self.init_weights()

    def init_weights(self, init_std: float = 0.02) -> None:
        if isinstance(self.final_proj, nn.Linear):
            nn.init.normal_(self.final_proj.weight, mean=0.0, std=init_std)

            if self.final_proj.bias is not None:
                nn.init.zeros_(self.final_proj.bias)

    @override
    def decode(self, x: Tensor) -> Tensor:
        x = self.decoder_frontend(x)
        x = self.decoder(x)
        return x

    @torch.no_grad()
    def generate(
        self,
        tokens: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tensor:
        for _ in range(max_new_tokens):
            if tokens.shape[-1] > self.max_seq_len:
                tokens = tokens[:, -self.max_seq_len :]

            logits = self(tokens)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, next_tokens), dim=1)
        return tokens


class TransformerFrontend(nn.Module):
    model_dim: int
    embed: nn.Linear | nn.Embedding
    pos_encoder: Optional[nn.Embedding]

    def __init__(
        self,
        model_dim: int,
        *,
        embed: nn.Linear | nn.Embedding,
        pos_encoder: Optional[nn.Embedding] = None,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.embed = embed
        self.pos_encoder = pos_encoder

        if dropout_p > 0.0:
            self.dropout = nn.Dropout(dropout_p)
        else:
            self.register_module("dropout", None)

        self.init_weights()

    def init_weights(self, init_std: float = 0.02) -> None:
        nn.init.normal_(self.embed.weight, mean=0.0, std=init_std)

        if self.pos_encoder is not None:
            if isinstance(self.pos_encoder, nn.Embedding):
                nn.init.normal_(self.pos_encoder.weight, mean=0.0, std=init_std)

        if isinstance(self.embed, nn.Linear) and self.embed.bias is not None:
            nn.init.zeros_(self.embed.bias)

    def forward(self, x: Tensor) -> Tensor:
        seqlen = x.size(1)

        x = self.embed(x)

        if self.pos_encoder is not None:
            # Assumes learned positional encoder for now.
            positions = torch.arange(
                start=0, end=seqlen, dtype=torch.int64, device=x.device
            )
            x = x + self.pos_encoder(positions)

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class TransformerDecoderLayer(nn.Module):
    sequence_mixer: nn.Module
    state_mixer: nn.Module
    model_dim: int
    norm_order: TransformerNormOrder

    def __init__(
        self,
        *,
        sequence_mixer: nn.Module,
        state_mixer: nn.Module,
        model_dim: int,
        dropout_p: float = 0.0,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
    ) -> None:
        super().__init__()
        self.sequence_mixer = sequence_mixer
        self.state_mixer = state_mixer
        self.model_dim = model_dim
        self.dropout_p = dropout_p
        self.norm_order = norm_order

        if dropout_p > 0.0:
            self.seqmixer_dropout = nn.Dropout(dropout_p)
        else:
            self.register_module("seqmixer_dropout", None)

        if dropout_p > 0.0:
            self.statemixer_dropout = nn.Dropout(dropout_p)
        else:
            self.register_module("statemixer_dropout", None)

        self.seqmixer_norm = nn.LayerNorm(model_dim)
        self.statemix_norm = nn.LayerNorm(model_dim)

    def _forward_seqmixer(self, x: Tensor) -> Tensor:
        residual = x

        if self.norm_order == TransformerNormOrder.PRE:
            x = self.seqmixer_norm(x)

        x = self.sequence_mixer(x)

        if self.seqmixer_dropout is not None:
            x = self.seqmixer_dropout(x)

        x = x + residual

        if self.norm_order == TransformerNormOrder.POST:
            x = self.seqmixer_norm(x)

        return x

    def _forward_statemixer(self, x: Tensor) -> Tensor:
        residual = x

        if self.norm_order == TransformerNormOrder.PRE:
            x = self.statemix_norm(x)

        x = self.state_mixer(x)

        if self.statemixer_dropout is not None:
            x = self.statemixer_dropout(x)

        x = x + residual

        if self.norm_order == TransformerNormOrder.POST:
            x = self.statemix_norm(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self._forward_seqmixer(x)
        x = self._forward_statemixer(x)
        return x

    def extra_repr(self) -> str:
        return f"model_dim={self.model_dim}, norm_order={self.norm_order.name}"


class TransformerDecoder(nn.Module):
    model_dim: int
    layers: nn.ModuleList
    norm_order: TransformerNormOrder

    def __init__(
        self,
        layers: Iterable[TransformerDecoderLayer],
        model_dim: int,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.model_dim = model_dim
        self.norm_order = norm_order
        self.dropout_p = dropout_p

        if norm_order != TransformerNormOrder.POST:
            self.layer_norm = nn.LayerNorm(model_dim)
        else:
            self.register_module("layer_norm", None)

        if dropout_p > 0.0:
            self.dropout = nn.Dropout(dropout_p)
        else:
            self.register_module("dropout", None)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x

    def extra_repr(self) -> str:
        return f"model_dim={self.model_dim}, norm_order={self.norm_order.name}"


class TiedProjection(nn.Module):
    def __init__(self, weight: Parameter, bias: Optional[Parameter] = None) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)
