"""GroupedMatmul operator modules and NPU kernels for Ascend950PR."""

from __future__ import annotations

from examples.ascend950pr_ops.grouped_matmul.kernels import grouped_matmul

__all__ = ["grouped_matmul"]
