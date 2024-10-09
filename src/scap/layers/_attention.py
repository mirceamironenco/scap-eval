import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Type, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

with warnings.catch_warnings(action="ignore", category=ImportWarning):
    from fla.modules import FusedRMSNormSwishGate, RMSNorm
    from fla.modules.feature_map import (
        DPFPFeatureMap,
        HadamardFeatureMap,
        HedgehogFeatureMap,
        T2RFeatureMap,
    )
    from fla.ops.gla import chunk_gla
    from fla.ops.linear_attn import chunk_linear_attn
from torch import Tensor

from scap.configs import BuilderConfig
from scap.layers._registry import register_layer

FeatureMap: TypeAlias = Literal[
    "hedgehog",
    "t2r",
    "elementwise_product",
    "dpfp",
    "elu",
    "relu",
    "identity",
]


@dataclass
class LinearAttentionCfg(BuilderConfig):
    _target: Type = field(default_factory=lambda: LinearAttention)
    num_heads: int = 8
    feature_map: FeatureMap = "elementwise_product"
    expand_k: float = 1.0
    expand_v: float = 1.0
    tie_feature_map_qk: bool = False
    output_norm: str = "rmsnorm"
    norm_q: bool = False
    norm_k: bool = False
    do_feature_map_norm: bool = False
    norm_eps: float = 1e-5


@register_layer("linear_attention", config=LinearAttentionCfg, as_seqmixer=True)
class LinearAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 8,
        feature_map: FeatureMap = "elementwise_product",
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        tie_feature_map_qk: bool = False,
        output_norm: str = "rmsnorm",
        norm_q: bool = False,
        norm_k: bool = False,
        do_feature_map_norm: bool = False,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        assert int(dim * expand_k) % num_heads == 0
        assert int(dim * expand_v) % num_heads == 0
        assert output_norm in ("rmsnorm", "identity")
        assert feature_map in (
            "hedgehog",
            "t2r",
            "elementwise_product",
            "dpfp",
            "elu",
            "relu",
            "identity",
        )

        self.dim = dim
        self.num_heads = num_heads
        self.feature_map = feature_map
        self.tie_feature_map_qk = tie_feature_map_qk
        self.output_norm = output_norm
        self.norm_q = norm_q
        self.norm_k = norm_k
        self.do_feature_map_norm = do_feature_map_norm
        self.norm_eps = norm_eps

        self.key_dim = int(self.dim * expand_k)
        self.value_dim = int(self.dim * expand_v)
        self.head_qk_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads

        if feature_map == "hedgehog":
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HedgehogFeatureMap(
                    head_dim=self.head_qk_dim
                )
            else:
                self.feature_map_q = HedgehogFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == "t2r":
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = T2RFeatureMap(
                    head_dim=self.head_qk_dim
                )
            else:
                self.feature_map_q = T2RFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = T2RFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == "elementwise_product":
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HadamardFeatureMap(
                    head_dim=self.head_qk_dim
                )
            else:
                self.feature_map_q = HadamardFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = HadamardFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == "dpfp":
            self.feature_map_q = DPFPFeatureMap(head_dim=self.head_qk_dim)
            self.feature_map_k = DPFPFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == "elu":

            def elu(x: Tensor) -> Tensor:
                return F.elu(x) + 1

            self.feature_map_q = elu
            self.feature_map_k = elu

        elif feature_map == "relu":
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()

        elif feature_map == "identity":
            self.feature_map_q = nn.Identity()
            self.feature_map_k = nn.Identity()

        self.wq = nn.Linear(self.dim, self.key_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.key_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.value_dim, bias=False)
        self.wo = nn.Linear(self.value_dim, self.dim, bias=False)

        if output_norm == "rmsnorm":
            self.norm = RMSNorm(
                hidden_size=self.head_v_dim,
                eps=norm_eps,
            )
        elif output_norm == "identity":
            self.norm = nn.Identity()

        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2**-2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q, k, v = map(
            lambda x: rearrange(x, "b l (h d) -> b h l d", h=self.num_heads),
            (q, k, v),
        )
        q, k = self.feature_map_q(q), self.feature_map_k(k)

        if self.norm_q:
            q = q / (q.sum(-1, keepdim=True) + 1e-4)

        if self.norm_k:
            k = k / (k.sum(-1, keepdim=True) + 1e-4)

        output, _ = chunk_linear_attn(q, k, v, normalize=self.do_feature_map_norm)
        output = self.norm(output)
        output = rearrange(output, "b h l d -> b l (h d)")
        output = self.wo(output)
        return output


@dataclass
class AttentionCfg(BuilderConfig):
    _target: Type = field(default_factory=lambda: Attention)

    num_heads: int = 8
    """Number of attention heads."""

    attn_drop: float = 0.0
    """Attention dropout rate."""

    window_size: Optional[int] = None
    """Sliding window size."""


@register_layer("attention", config=AttentionCfg, as_seqmixer=True)
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        window_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.window_size = window_size
        self.head_dim = dim // num_heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.init_weights()

    def init_weights(self, init_std: float = 0.02) -> None:
        for layer in (self.wq, self.wk, self.wv, self.wo):
            nn.init.normal_(layer.weight, mean=0.0, std=init_std)

            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        seqlen = x.size(1)
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q, k, v = map(
            lambda x: rearrange(x, "b l (h d) -> b h l d", h=self.num_heads), (q, k, v)
        )

        is_causal, mask = True, None
        if self.window_size is not None:
            is_causal = False
            dt = torch.float32 if x.dtype == torch.bfloat16 else x.dtype
            mask = torch.ones((seqlen, seqlen), device=x.device, dtype=dt)
            mask.tril_(diagonal=0).triu_(diagonal=1 - self.window_size)
            mask.log_()

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            attn_mask=mask,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        output = rearrange(output, "b h l d -> b l (h d)")
        output = self.wo(output)
        return output


@dataclass
class GatedLinearAttentionCfg(BuilderConfig):
    _target: Type = field(default_factory=lambda: GatedLinearAttention)

    expand_v: float = 0.5
    expand_k: float = 1.0
    num_heads: int = 4
    gate_fn: str = "swish"
    norm_eps: float = 1e-5
    gate_logit_normalizer: int = 16
    gate_low_rank_dim: int = 16
    clamp_min: Optional[float] = None
    fuse_norm: bool = True


@register_layer("gla", config=GatedLinearAttentionCfg, as_seqmixer=True)
class GatedLinearAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        expand_v: float = 0.5,
        expand_k: float = 1.0,
        num_heads: int = 4,
        gate_fn: str = "swish",
        norm_eps: float = 1e-5,
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        clamp_min: Optional[float] = None,
        fuse_norm: bool = True,
    ) -> None:
        super().__init__()

        assert int(dim * expand_k) % num_heads == 0
        assert int(dim * expand_v) % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.gate_fn = F.silu
        self.norm_eps = norm_eps
        self.gate_logit_normalizer = gate_logit_normalizer
        self.gate_low_rank_dim = gate_low_rank_dim
        self.clamp_min = clamp_min

        self.key_dim = int(dim * expand_k)
        self.value_dim = int(dim * expand_v)
        self.head_qk_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads

        self.wq = nn.Linear(self.dim, self.key_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.key_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.value_dim, bias=False)
        self.wgk = nn.Sequential(
            nn.Linear(self.dim, gate_low_rank_dim, bias=False),
            nn.Linear(gate_low_rank_dim, self.key_dim, bias=True),
        )
        self.wg = nn.Linear(self.dim, self.value_dim, bias=False)
        self.wo = nn.Linear(self.value_dim, self.dim, bias=False)

        self.fuse_norm_and_gate = gate_fn == "swish" and fuse_norm
        if gate_fn == "swish" and fuse_norm:
            self.g_norm_swish_gate = FusedRMSNormSwishGate(
                self.head_v_dim, eps=norm_eps
            )
        else:
            self.g_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2**-2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v, gk = self.wq(x), self.wk(x), self.wv(x), self.wgk(x)
        q, k, v, gk = map(
            lambda x: rearrange(x, "b l (h d) -> b h l d", h=self.num_heads),
            (q, k, v, gk),
        )
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        if self.clamp_min is not None:
            gk = torch.clamp_min(gk, self.clamp_min)

        output, _ = chunk_gla(q, k, v, gk)
        output = rearrange(output, "b h l d -> b l h d")
        g = self.wg(x)

        if self.fuse_norm_and_gate:
            g = rearrange(g, "b l (h d) -> b l h d", h=self.num_heads)
            output = self.g_norm_swish_gate(output, g)
            output = rearrange(output, "b l h d -> b l (h d)")
        else:
            output = self.g_norm(output)
            output = rearrange(output, "b h l d -> b l (h d)")
            output = output * self.gate_fn(g)
        output = self.wo(output)
        return output
