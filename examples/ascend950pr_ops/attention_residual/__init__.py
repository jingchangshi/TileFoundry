"""AttentionResidual operator modules and NPU kernels for Ascend950PR."""

from __future__ import annotations

from examples.ascend950pr_ops.attention_residual.kernels import attention_residual

__all__ = ["attention_residual"]
