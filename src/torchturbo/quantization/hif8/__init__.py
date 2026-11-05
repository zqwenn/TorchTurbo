"""One-pass HiFloat8 dense and expert conversion."""

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
    """Convert dense and expert leaves with a single composite backend recipe."""
    from . import expert, linear

    linears = linear.select_linears(candidates, config)
    experts = expert.select_experts(candidates, config)
    if not linears and not experts:
        return 0, 0

    import torchao_npu  # noqa: F401
    from torchao.quantization import quantize_
    from torchao_npu.features.training.hifloat8 import HiFloat8Config

    recipe = HiFloat8Config(
        linear_config=linear.make_linear_config(config) if linears else None,
        grouped_config=expert.make_expert_config(config) if experts else None,
    )
    targets = linears | expert.prepare_experts(model, config, experts)
    quantize_(model, recipe, filter_fn=lambda module, fqn: fqn in targets)
    return len(linears), len(experts)
