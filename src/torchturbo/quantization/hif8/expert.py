"""HiFloat8 expert selection and stateless GEMM conversion targets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from ...engine.config import QuantizationConfig

if TYPE_CHECKING:
    from torchao_npu.features.training.hifloat8 import HiFloat8GMMConfig

from .recipe import make_policy


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


def make_expert_config(config: QuantizationConfig) -> HiFloat8GMMConfig:
    """Build the backend-owned grouped training configuration."""
    import torchao_npu  # noqa: F401
    from torchao_npu.features.training.hifloat8 import HiFloat8GMMConfig

    return HiFloat8GMMConfig(**make_policy(config))


def prepare_experts(
    model: nn.Module, config: QuantizationConfig, selected_experts: set[str],
) -> set[str]:
    """Attach stateless leaves without changing parameter or checkpoint names."""
    from ...models.modules.moe.grouped_gemm import GroupedGemm

    targets = set()
    for fqn in selected_experts:
        expert = model.get_submodule(fqn)
        if "grouped_gemm" not in expert._modules:
            expert.add_module("grouped_gemm", GroupedGemm())
        expert.quantization_config = config
        targets.add(f"{fqn}.grouped_gemm" if fqn else "grouped_gemm")
    return targets
