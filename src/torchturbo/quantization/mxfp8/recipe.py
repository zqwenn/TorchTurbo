"""MXFP8 compute recipe for TorchAO NPU."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchao_npu.features.training.mxfp8 import MXFP8LinearConfig


def make_linear_config(*, dgrad: bool, wgrad: bool) -> MXFP8LinearConfig:
    """Build an E4M3 MXFP8 compute recipe with high precision storage."""
    import torchao_npu  # noqa: F401
    from torchao_npu.features.training.mxfp8 import MXFP8LinearConfig

    return MXFP8LinearConfig(
        quant_format="E4M3",
        block_size=32,
        enable_fsdp_low_precision_all_gather=False,
        quant_mode="jit_quant",
        dgrad=dgrad,
        wgrad=wgrad,
    )
