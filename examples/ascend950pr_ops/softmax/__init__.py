"""Softmax operator modules and Ascend 950PR SimdVF kernels."""

from examples.ascend950pr_ops.softmax.kernels import kernel, softmax

__all__ = ["kernel", "softmax"]
