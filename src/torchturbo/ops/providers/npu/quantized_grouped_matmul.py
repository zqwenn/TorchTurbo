"""MXFP8 expert GEMMs delegated to TorchAO NPU."""

from __future__ import annotations

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="quantized_grouped_matmul",
    impl_name="torchao_npu_quantized_grouped_matmul",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class TorchAONpuQuantizedGroupedMatmul(OpImpl):
    def available(self) -> bool:
        from importlib.util import find_spec

        return find_spec("torchao_npu") is not None and find_spec("torch_npu") is not None

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        group_list: torch.Tensor,
        *,
        format: str = "e4m3",
        trans_b: bool = False,
        dgrad: bool = True,
        wgrad: bool = True,
    ) -> torch.Tensor:
        """Multiply packed token groups by expert weights with FP8 GEMMs.

        Inputs use [sum(tokens), K] and [E, K, N] layouts (or [E, N, K]
        with trans_b). Counts contain E nonnegative token counts. Output is
        [sum(tokens), N] in the input dtype. Each gradient GEMM can retain
        the input precision independently.
        """
        if format != "e4m3":
            raise ValueError(f"Unknown quantization format: {format}")
        if x.shape[0] == 0:
            # Selecting one matrix keeps every expert's zero gradient connected.
            matrix = weight[0] if weight.shape[0] else weight.sum(dim=0)
            return x @ (matrix.t() if trans_b else matrix)
        from torchturbo.quantization.mxfp8.expert import grouped_matmul

        return grouped_matmul(
            x, weight, group_list, trans_b=trans_b, dgrad=dgrad, wgrad=wgrad,
        )
