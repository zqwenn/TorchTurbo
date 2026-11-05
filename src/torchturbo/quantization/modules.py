"""Shared entry point for format-specific quantization conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from ..logging import get_logger

if TYPE_CHECKING:
    from ..engine.config import QuantizationConfig
    from ..engine.model_fqn import ModelFQNMap

logger = get_logger(__name__)


def apply_quantization(
    model: nn.Module,
    config: QuantizationConfig,
    *,
    model_fqn_map: ModelFQNMap | None = None,
) -> int:
    """Route selected modules to the configured format before sharding."""
    if not config.enabled:
        return 0

    from importlib.metadata import version

    from ..engine.model_fqn import build_model_fqn_map
    from .selection import select_modules

    if version("torchao").split(".")[:2] != ["0", "18"]:
        raise ValueError("TorchTurbo quantization currently requires torchao 0.18")
    if model_fqn_map is None:
        model_fqn_map = build_model_fqn_map(model)
    candidates = select_modules(model, config, model_fqn_map)
    if config.format == "hif8":
        from .hif8 import apply_quantization as convert
    else:
        from .mxfp8 import apply_quantization as convert

    linears, experts = convert(model, config, candidates)
    logger.info_rank0(
        "torchao_npu quantization: converted %d Linear and %d expert modules (%s, dgrad=%s, wgrad=%s)",
        linears, experts, config.format, config.dgrad, config.wgrad,
    )
    return linears + experts
