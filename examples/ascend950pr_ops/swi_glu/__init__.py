"""SwiGlu operator modules and Ascend 950PR SimdVF kernels."""

from examples.ascend950pr_ops.swi_glu.kernels import kernel, swi_glu

__all__ = ["kernel", "swi_glu"]
