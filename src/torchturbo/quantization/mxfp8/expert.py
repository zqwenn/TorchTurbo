"""MXFP8 expert conversion and grouped GEMM adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn
from torchao.core.config import AOBaseConfig
from torchao.quantization.transform_module import register_quantize_module_handler

from .recipe import make_linear_config

if TYPE_CHECKING:
    from ...engine.config import QuantizationConfig


def select_experts(
    candidates: list[tuple[str, nn.Module]], config: QuantizationConfig,
) -> set[str]:
    """Select native grouped experts with supported feature dimensions."""
    from ...models.modules.moe.experts import GroupedMLP

    if not config.moe:
        return set()
    selected = set()
    for fqn, module in candidates:
        if not isinstance(module, GroupedMLP):
            continue
        if module.use_te:
            raise ValueError("TorchAO quantization requires native GroupedMLP weights")
        if module.config.hidden_size % 64 or module.config.moe_ffn_hidden_size % 64:
            continue
        selected.add(fqn)
    return selected


@dataclass
class GroupedMLPConfig(AOBaseConfig):
    """Adapt torchao conversion to framework-owned expert execution."""

    compute_policy: QuantizationConfig


@register_quantize_module_handler(GroupedMLPConfig)
def _quantize_grouped_mlp(
    module: nn.Module, config: GroupedMLPConfig,
) -> nn.Module:
    module.quantization_config = config.compute_policy
    return module


def quantize_experts(
    model: nn.Module,
    config: QuantizationConfig,
    selected_fqns: set[str],
) -> None:
    """Configure expert compute through torchao without replacing EP or weights."""
    make_linear_config(dgrad=config.dgrad, wgrad=config.wgrad)
    from torchao.quantization import quantize_

    quantize_(
        model, GroupedMLPConfig(compute_policy=config),
        filter_fn=lambda module, fqn: fqn in selected_fqns,
    )


def grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    counts: torch.Tensor,
    *,
    trans_b: bool,
    dgrad: bool,
    wgrad: bool,
) -> torch.Tensor:
    """Delegate packed expert GEMMs while retaining routing and EP ownership."""
    recipe = make_linear_config(dgrad=dgrad, wgrad=wgrad)
    from torchao_npu.features.training.mxfp8.mx_gmm import mx_quant_group_gemm

    kernel_weight = weight.transpose(1, 2) if trans_b else weight
    groups = counts.to(device=x.device, dtype=torch.int64, non_blocking=True)
    return mx_quant_group_gemm(
        x, kernel_weight, groups, recipe,
        grad_enabled=torch.is_grad_enabled(), group_list_type=1,
    )
