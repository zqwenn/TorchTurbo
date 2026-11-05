"""MXFP8 Linear selection and conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from ...engine.config import QuantizationConfig

from .recipe import make_linear_config


def select_linears(
    candidates: list[tuple[str, nn.Module]], config: QuantizationConfig,
) -> set[str]:
    """Select supported dense modules without touching their parameters."""
    if not config.linear:
        return set()
    return {
        fqn for fqn, module in candidates
        if type(module) is nn.Linear
        and module.in_features % 64 == 0
        and module.out_features % 64 == 0
    }


def quantize_linears(
    model: nn.Module,
    config: QuantizationConfig,
    selected_fqns: set[str],
) -> None:
    """Convert canonical-FQN-selected Linear modules through torchao."""
    recipe = make_linear_config(dgrad=config.dgrad, wgrad=config.wgrad)
    from torchao.quantization import quantize_

    quantize_(
        model, recipe,
        filter_fn=lambda module, fqn: fqn in selected_fqns,
    )
