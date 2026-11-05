"""Shared HiFloat8 training ranges and gradient precision policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...engine.config import QuantizationConfig

_HIF8_TRAINING_RANGES = dict(
    input_dst_type_max=15.0,
    weight_dst_type_max=15.0,
    grad_dst_type_max=224.0,
)


def make_policy(config: QuantizationConfig) -> dict[str, float | bool]:
    """Use the same range and gradient policy for dense and grouped GEMMs."""
    return dict(_HIF8_TRAINING_RANGES, dgrad=config.dgrad, wgrad=config.wgrad)
