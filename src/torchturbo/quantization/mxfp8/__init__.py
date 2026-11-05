"""MXFP8 module conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from ...engine.config import QuantizationConfig


def apply_quantization(
    model: nn.Module,
    config: QuantizationConfig,
    candidates: list[tuple[str, nn.Module]],
) -> tuple[int, int]:
    """Convert the selected dense and expert modules."""
    from . import expert, linear

    linears = linear.select_linears(candidates, config)
    experts = expert.select_experts(candidates, config)
    if linears:
        linear.quantize_linears(model, config, linears)
    if experts:
        expert.quantize_experts(model, config, experts)
    return len(linears), len(experts)
