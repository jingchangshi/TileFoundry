"""KimiDeltaAttention operator modules and Ascend 950PR Persistent-State kernels."""

from examples.ascend950pr_ops.kimi_delta_attention.kernels import (
    kernel,
    kimi_delta_attention,
)

__all__ = ["kernel", "kimi_delta_attention"]
