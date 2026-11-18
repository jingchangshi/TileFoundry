"""AddRmsNormDynamicQuant operator modules and Ascend 950PR SimdVF kernels."""

from examples.ascend950pr_ops.add_rms_norm_dynamic_quant.kernels import (
    add_rms_norm_dynamic_quant,
    kernel,
)

__all__ = ["add_rms_norm_dynamic_quant", "kernel"]
