"""ApplyRotaryPosEmb operator module and Ascend 950PR single-launch fused 56-AIV kernel."""

from examples.ascend950pr_ops.apply_rotary_pos_emb.kernels import (
    apply_rotary_pos_emb,
    kernel,
)

__all__ = ["apply_rotary_pos_emb", "kernel"]
