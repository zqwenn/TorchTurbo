"""Stateless grouped GEMM leaf for backend-owned quantize_ conversion."""

from __future__ import annotations

import torch
from torch import nn

from torchturbo.ops import ops


class GroupedGemm(nn.Module):
    """Keep expert weights on their owner and pass layouts at execution time."""

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        offsets: torch.Tensor,
        trans_b: bool = False,
    ) -> torch.Tensor:
        if x.shape[0] == 0:
            matrix = weight[0] if weight.shape[0] else weight.sum(dim=0)
            return x @ (matrix.t() if trans_b else matrix)
        counts = torch.diff(offsets, prepend=offsets.new_zeros(1))
        return ops.grouped_matmul(x, weight, counts, trans_b=trans_b)
