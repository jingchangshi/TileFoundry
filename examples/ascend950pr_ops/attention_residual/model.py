"""Authored HIR specification for Block Attention Residuals on Ascend950PR."""

from __future__ import annotations

import math

from tilefoundry.dsl import *
from tilefoundry_ascend import AscendTarget

TOPOS = (
    Topology("block", 1),
    Topology("aic", 1),
    Topology("aiv", 2),
    Topology("lane", 64),
)


def build(
    N: int = 4,
    B: int = 1,
    S: int = 4,
    D: int = 64,
    alpha: float = 0.5,
    epsilon: float = 1e-6,
    dtype: str = "bf16",
):
    """Build an authored HIR AttentionResidual module for Ascend950PR."""
    inv_sqrt_d = 1.0 / math.sqrt(D)
    one_minus_alpha = 1.0 - alpha
    eps_val = float(epsilon)
    alpha_val = float(alpha)

    @module(entry="forward", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
    class AttentionResidualModel:
        @func
        def forward(
            v_stack: Tensor[(N, B, S, D), dtype],
            x: Tensor[(B, S, D), dtype],
            w: Tensor[(D,), dtype],
            gamma: Tensor[(D,), dtype],
        ) -> Tensor[(B, S, D), dtype]:
            vf = tf.cast(v_stack, dtype="f32")
            xf = tf.cast(x, dtype="f32")
            wf = tf.cast(w, dtype="f32")
            gf = tf.cast(gamma, dtype="f32")
            k = tf.rms_norm(vf, gf, eps=eps_val)
            scores = (
                tf.reduce(
                    k * tf.reshape(wf, new_shape=(1, 1, 1, D)),
                    axes=(-1,),
                    keepdim=False,
                    kind="sum",
                )
                * inv_sqrt_d
            )
            probs = tf.softmax(scores, axis=0)
            agg = tf.reduce(
                tf.reshape(probs, new_shape=(N, B, S, 1)) * vf,
                axes=(0,),
                keepdim=False,
                kind="sum",
            )
            y = one_minus_alpha * xf + alpha_val * agg
            return tf.cast(y, dtype=dtype)

    return AttentionResidualModel


AttentionResidualModel = build()
