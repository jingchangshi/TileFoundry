"""Authored HIR specification for Multi-Head Latent Attention (MLA) on Ascend950PR."""

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
    B: int = 1,
    S: int = 16,
    S_kv: int = 16,
    N_q: int = 4,
    N_kv: int = 1,
    D_nope: int = 64,
    D_rope: int = 64,
    scale_value: float = -1.0,
    is_causal: bool = True,
    dtype: str = "f16",
):
    """Build an authored HIR MLA module (BSND layout) for Ascend950PR."""
    D_qk = D_nope + D_rope
    G = N_q // N_kv
    scale = (1.0 / math.sqrt(D_qk)) if scale_value <= 0.0 else float(scale_value)

    @module(entry="forward", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
    class MlaModel:
        if is_causal:

            @func
            def forward(
                q_nope: Tensor[(B, S, N_q, D_nope), dtype],
                q_rope: Tensor[(B, S, N_q, D_rope), dtype],
                k_nope: Tensor[(B, S_kv, N_kv, D_nope), dtype],
                k_rope: Tensor[(B, S_kv, N_kv, D_rope), dtype],
                v: Tensor[(B, S_kv, N_kv, D_nope), dtype],
            ) -> Tensor[(B, S, N_q, D_nope), dtype]:
                q = tf.concat([q_nope, q_rope], axis=-1)
                k = tf.concat([k_nope, k_rope], axis=-1)
                q_h = tf.transpose(
                    tf.reshape(q, new_shape=(B, S, N_kv, G, D_qk)),
                    perm=(0, 2, 3, 1, 4),
                )
                q_m = tf.reshape(q_h, new_shape=(B * N_kv, G * S, D_qk))
                k_h = tf.reshape(
                    tf.transpose(k, perm=(0, 2, 1, 3)),
                    new_shape=(B * N_kv, S_kv, D_qk),
                )
                v_h = tf.reshape(
                    tf.transpose(v, perm=(0, 2, 1, 3)),
                    new_shape=(B * N_kv, S_kv, D_nope),
                )
                raw_scores = tf.matmul(q_m, tf.transpose(k_h, perm=(0, 2, 1)), out_dtype="f32")
                scores = raw_scores * scale
                scores_4d = tf.reshape(scores, new_shape=(B * N_kv, G, S, S_kv))
                qi = tf.reshape(tf.arange(Tensor[(S,), "i64"], start=0), new_shape=(1, 1, S, 1))
                kj = tf.reshape(
                    tf.arange(Tensor[(S_kv,), "i64"], start=0), new_shape=(1, 1, 1, S_kv)
                )
                causal_mask = tf.cmp_gt(kj, qi + (S_kv - S))
                masked_scores = tf.where(causal_mask, tf.full_like(scores_4d, -1e9), scores_4d)
                probs_4d = tf.softmax(masked_scores, axis=-1)
                probs = tf.reshape(probs_4d, new_shape=(B * N_kv, G * S, S_kv))
                out_m = tf.matmul(tf.cast(probs, dtype=dtype), v_h, out_dtype="f32")
                out_5d = tf.reshape(out_m, new_shape=(B, N_kv, G, S, D_nope))
                out_bsnd = tf.reshape(
                    tf.transpose(out_5d, perm=(0, 3, 1, 2, 4)),
                    new_shape=(B, S, N_q, D_nope),
                )
                return tf.cast(out_bsnd, dtype=dtype)
        else:

            @func
            def forward(
                q_nope: Tensor[(B, S, N_q, D_nope), dtype],
                q_rope: Tensor[(B, S, N_q, D_rope), dtype],
                k_nope: Tensor[(B, S_kv, N_kv, D_nope), dtype],
                k_rope: Tensor[(B, S_kv, N_kv, D_rope), dtype],
                v: Tensor[(B, S_kv, N_kv, D_nope), dtype],
            ) -> Tensor[(B, S, N_q, D_nope), dtype]:
                q = tf.concat([q_nope, q_rope], axis=-1)
                k = tf.concat([k_nope, k_rope], axis=-1)
                q_h = tf.transpose(
                    tf.reshape(q, new_shape=(B, S, N_kv, G, D_qk)),
                    perm=(0, 2, 3, 1, 4),
                )
                q_m = tf.reshape(q_h, new_shape=(B * N_kv, G * S, D_qk))
                k_h = tf.reshape(
                    tf.transpose(k, perm=(0, 2, 1, 3)),
                    new_shape=(B * N_kv, S_kv, D_qk),
                )
                v_h = tf.reshape(
                    tf.transpose(v, perm=(0, 2, 1, 3)),
                    new_shape=(B * N_kv, S_kv, D_nope),
                )
                raw_scores = tf.matmul(q_m, tf.transpose(k_h, perm=(0, 2, 1)), out_dtype="f32")
                scores = raw_scores * scale
                scores_4d = tf.reshape(scores, new_shape=(B * N_kv, G, S, S_kv))
                probs_4d = tf.softmax(scores_4d, axis=-1)
                probs = tf.reshape(probs_4d, new_shape=(B * N_kv, G * S, S_kv))
                out_m = tf.matmul(tf.cast(probs, dtype=dtype), v_h, out_dtype="f32")
                out_5d = tf.reshape(out_m, new_shape=(B, N_kv, G, S, D_nope))
                out_bsnd = tf.reshape(
                    tf.transpose(out_5d, perm=(0, 3, 1, 2, 4)),
                    new_shape=(B, S, N_q, D_nope),
                )
                return tf.cast(out_bsnd, dtype=dtype)

    return MlaModel


MlaModel = build()
