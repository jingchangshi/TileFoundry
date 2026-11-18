"""Q4_0DequantMatmul operator module and Ascend 950PR dual-regime Track-B kernels."""

from examples.ascend950pr_ops.q4_0_dequant_matmul.kernels import (
    kernel,
    q4_0_dequant_matmul,
)

__all__ = ["kernel", "q4_0_dequant_matmul"]
