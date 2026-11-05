"""Canonical FQN filtering and adapter boundary resolution."""

from __future__ import annotations

from fnmatch import fnmatchcase
from importlib.util import find_spec
from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from ..engine.config import QuantizationConfig
    from ..engine.model_fqn import ModelFQNMap


def select_modules(
    model: nn.Module, config: QuantizationConfig, model_fqn_map: ModelFQNMap,
) -> list[tuple[str, nn.Module]]:
    """Resolve selected model-authored FQNs to their active prepared modules."""
    if find_spec("peft") is not None:
        from peft.tuners.tuners_utils import BaseTunerLayer
    else:
        BaseTunerLayer = None

    selected = []
    for name, prepared_name in model_fqn_map.modules.items():
        module = model.get_submodule(prepared_name)
        if name.split(".")[-1] in {"lm_head", "router", "gate"}:
            continue
        if config.include_fqns and not any(
            fnmatchcase(name, pattern) for pattern in config.include_fqns
        ):
            continue
        if any(fnmatchcase(name, pattern) for pattern in config.exclude_fqns):
            continue
        if BaseTunerLayer is not None:
            while isinstance(module, BaseTunerLayer):
                module = module.base_layer
                prepared_name = (
                    f"{prepared_name}.base_layer" if prepared_name else "base_layer"
                )
        selected.append((prepared_name, module))
    return selected
