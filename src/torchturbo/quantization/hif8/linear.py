"""HiFloat8 Linear selection and backend recipe."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from ...engine.config import QuantizationConfig

if TYPE_CHECKING:
    from torchao_npu.features.training.hifloat8 import HiFloat8LinearConfig

from .recipe import make_policy


def select_linears(
    candidates: list[tuple[str, nn.Module]], config: QuantizationConfig,
) -> set[str]:
    """Select supported dense modules without touching their parameters."""
    if not config.linear:
        return set()
    from torchao_npu.features.training.hifloat8 import HiFloat8Linear

    return {
        fqn for fqn, module in candidates
        if type(module) in (nn.Linear, HiFloat8Linear)
        and module.in_features % 64 == 0
        and module.out_features % 64 == 0
    }


def make_linear_config(config: QuantizationConfig) -> HiFloat8LinearConfig:
    """Build the backend-owned dense training configuration."""
    import torchao_npu  # noqa: F401
    from torchao_npu.features.training.hifloat8 import HiFloat8LinearConfig

    return HiFloat8LinearConfig(**make_policy(config))
