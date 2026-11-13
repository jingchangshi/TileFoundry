"""Scheduled HIR AttentionResidual module with SimtVF and Ascend950PR pipelining."""

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
    """Build a scheduled HIR AttentionResidual module lowered through Ascend passes."""
    M = B * S
    inv_sqrt_d = 1.0 / math.sqrt(D)
    inv_d = 1.0 / float(D)
    one_minus_alpha = 1.0 - alpha
    eps_val = float(epsilon)
    alpha_val = float(alpha)

    @module(entry="forward", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
    class AttentionResidualScheduled:
        @func
        def forward(
            v_stack: Tensor[(N, B, S, D), dtype],
            x: Tensor[(B, S, D), dtype],
            w: Tensor[(D,), dtype],
            gamma: Tensor[(D,), dtype],
        ) -> Tensor[(B, S, D), dtype]:
            with Mesh(("aiv",), (1,), names=("v",)) as _v:
                v_flat = tf.reshape(v_stack, new_shape=(N, M, D))
                x_flat = tf.reshape(x, new_shape=(M, D))
                w_2d = tf.reshape(w, new_shape=(1, D))
                g_2d = tf.reshape(gamma, new_shape=(1, D))
                w_ub = tf.schedule(
                    (w_2d, tf.empty(Tensor[(1, D), dtype, "smem"])), op=T.ascend_gm_to_ub()
                )
                g_ub = tf.schedule(
                    (g_2d, tf.empty(Tensor[(1, D), dtype, "smem"])), op=T.ascend_gm_to_ub()
                )
                x_ub = tf.schedule(
                    (x_flat, tf.empty(Tensor[(M, D), dtype, "smem"])), op=T.ascend_gm_to_ub()
                )
                z_m1 = tf.zeros(Tensor[(M, 1), "f32", "smem"])
                inv_d_ub = tf.full_like(z_m1, inv_d)
                eps_ub = tf.full_like(z_m1, eps_val)
                scale_ub = tf.full_like(z_m1, inv_sqrt_d)
                z_md = tf.zeros(Tensor[(M, D), "f32", "smem"])
                c_res_ub = tf.full_like(z_md, one_minus_alpha)
                c_agg_ub = tf.full_like(z_md, alpha_val)
                scores_gm = tf.zeros(Tensor[(N, M), "f32", "gmem"])

                for n in tf.pipelined(N, max_stage=2, max_buffer_versions=2, enable_offset=True):
                    v_ub = tf.schedule(
                        (v_flat[n, :, :], tf.empty(Tensor[(M, D), dtype, "smem"])),
                        op=T.ascend_gm_to_ub(),
                    )
                    with T.SimtVF():
                        vr = tf.schedule(
                            (v_ub, tf.empty(Tensor[(M, D), dtype, "rmem"])),
                            op=T.ascend_ub_to_reg(),
                        )
                        wr = tf.schedule(
                            (w_ub, tf.empty(Tensor[(1, D), dtype, "rmem"])),
                            op=T.ascend_ub_to_reg(),
                        )
                        gr = tf.schedule(
                            (g_ub, tf.empty(Tensor[(1, D), dtype, "rmem"])),
                            op=T.ascend_ub_to_reg(),
                        )
                        inv_d_r = tf.schedule(
                            (inv_d_ub, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_ub_to_reg(),
                        )
                        eps_r = tf.schedule(
                            (eps_ub, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_ub_to_reg(),
                        )
                        scale_r = tf.schedule(
                            (scale_ub, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_ub_to_reg(),
                        )
                        vf = tf.schedule(
                            (vr, tf.empty(Tensor[(M, D), "f32", "rmem"])),
                            op=T.ascend_vcast(dtype="f32"),
                        )
                        wf = tf.schedule(
                            (wr, tf.empty(Tensor[(1, D), "f32", "rmem"])),
                            op=T.ascend_vcast(dtype="f32"),
                        )
                        gf = tf.schedule(
                            (gr, tf.empty(Tensor[(1, D), "f32", "rmem"])),
                            op=T.ascend_vcast(dtype="f32"),
                        )
                        v_sq = tf.schedule(
                            (vf, vf, tf.empty(Tensor[(M, D), "f32", "rmem"])),
                            op=T.ascend_vmul(),
                        )
                        sq_sum = tf.schedule(
                            (v_sq, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_vreduce_sum(axes=(-1,), keepdim=True),
                        )
                        mean_sq = tf.schedule(
                            (sq_sum, inv_d_r, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_vmul(),
                        )
                        mean_eps = tf.schedule(
                            (mean_sq, eps_r, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_vadd(),
                        )
                        inv_rms = tf.schedule(
                            (mean_eps, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_vrsqrt(),
                        )
                        v_norm = tf.schedule(
                            (vf, inv_rms, tf.empty(Tensor[(M, D), "f32", "rmem"])),
                            op=T.ascend_vmul(),
                        )
                        kw = tf.schedule(
                            (wf, gf, tf.empty(Tensor[(1, D), "f32", "rmem"])),
                            op=T.ascend_vmul(),
                        )
                        weighted = tf.schedule(
                            (
                                v_norm,
                                kw,
                                tf.empty(Tensor[(M, D), "f32", "rmem"]),
                            ),
                            op=T.ascend_vmul(),
                        )
                        dot = tf.schedule(
                            (weighted, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_vreduce_sum(axes=(-1,), keepdim=True),
                        )
                        s_r = tf.schedule(
                            (dot, scale_r, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_vmul(),
                        )
                        s_ub = tf.schedule(
                            (s_r, tf.empty(Tensor[(M, 1), "f32", "smem"])),
                            op=T.ascend_reg_to_ub(),
                        )
                    s_gm_row = tf.schedule(
                        (
                            tf.reshape(s_ub, new_shape=(M,)),
                            tf.empty(Tensor[(M,), "f32", "gmem"]),
                        ),
                        op=T.ascend_ub_to_gm(),
                    )
                    scores_gm = tf.insert_slice(
                        scores_gm, tf.reshape(s_gm_row, new_shape=(1, M)), (n, 0)
                    )

                scores_ub = tf.schedule(
                    (scores_gm, tf.empty(Tensor[(N, M), "f32", "smem"])),
                    op=T.ascend_gm_to_ub(),
                )
                probs_gm = tf.zeros(Tensor[(N, M), "f32", "gmem"])
                with T.SimtVF():
                    sr = tf.schedule(
                        (scores_ub, tf.empty(Tensor[(N, M), "f32", "rmem"])),
                        op=T.ascend_ub_to_reg(),
                    )
                    s_max = tf.schedule(
                        (sr, tf.empty(Tensor[(1, M), "f32", "rmem"])),
                        op=T.ascend_vreduce_max(axes=(0,), keepdim=True),
                    )
                    s_shift = tf.schedule(
                        (sr, s_max, tf.empty(Tensor[(N, M), "f32", "rmem"])),
                        op=T.ascend_vsub(),
                    )
                    ex = tf.schedule(
                        (s_shift, tf.empty(Tensor[(N, M), "f32", "rmem"])),
                        op=T.ascend_vexp(),
                    )
                    ex_sum = tf.schedule(
                        (ex, tf.empty(Tensor[(1, M), "f32", "rmem"])),
                        op=T.ascend_vreduce_sum(axes=(0,), keepdim=True),
                    )
                    pr = tf.schedule(
                        (ex, ex_sum, tf.empty(Tensor[(N, M), "f32", "rmem"])),
                        op=T.ascend_vdiv(),
                    )
                    p_ub = tf.schedule(
                        (pr, tf.empty(Tensor[(N, M), "f32", "smem"])),
                        op=T.ascend_reg_to_ub(),
                    )
                probs_gm = tf.schedule((p_ub, probs_gm), op=T.ascend_ub_to_gm())

                acc_ub = tf.zeros(Tensor[(M, D), "f32", "smem"])
                for n in tf.pipelined(N, max_stage=2, max_buffer_versions=2, enable_offset=True):
                    v2_ub = tf.schedule(
                        (v_flat[n, :, :], tf.empty(Tensor[(M, D), dtype, "smem"])),
                        op=T.ascend_gm_to_ub(),
                    )
                    pn_ub = tf.schedule(
                        (
                            tf.reshape(probs_gm[n, :], new_shape=(M, 1)),
                            tf.empty(Tensor[(M, 1), "f32", "smem"]),
                        ),
                        op=T.ascend_gm_to_ub(),
                    )
                    with T.SimtVF():
                        v2r = tf.schedule(
                            (v2_ub, tf.empty(Tensor[(M, D), dtype, "rmem"])),
                            op=T.ascend_ub_to_reg(),
                        )
                        v2f = tf.schedule(
                            (v2r, tf.empty(Tensor[(M, D), "f32", "rmem"])),
                            op=T.ascend_vcast(dtype="f32"),
                        )
                        pnr = tf.schedule(
                            (pn_ub, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_ub_to_reg(),
                        )
                        weighted_v = tf.schedule(
                            (v2f, pnr, tf.empty(Tensor[(M, D), "f32", "rmem"])),
                            op=T.ascend_vmul(),
                        )
                        acc_r = tf.schedule(
                            (acc_ub, tf.empty(Tensor[(M, D), "f32", "rmem"])),
                            op=T.ascend_ub_to_reg(),
                        )
                        acc_next = tf.schedule(
                            (acc_r, weighted_v, tf.empty(Tensor[(M, D), "f32", "rmem"])),
                            op=T.ascend_vadd(),
                        )
                        acc_ub = tf.schedule(
                            (acc_next, tf.empty(Tensor[(M, D), "f32", "smem"])),
                            op=T.ascend_reg_to_ub(),
                        )

                with T.SimtVF():
                    xr = tf.schedule(
                        (x_ub, tf.empty(Tensor[(M, D), dtype, "rmem"])),
                        op=T.ascend_ub_to_reg(),
                    )
                    xf = tf.schedule(
                        (xr, tf.empty(Tensor[(M, D), "f32", "rmem"])),
                        op=T.ascend_vcast(dtype="f32"),
                    )
                    acc_f = tf.schedule(
                        (acc_ub, tf.empty(Tensor[(M, D), "f32", "rmem"])),
                        op=T.ascend_ub_to_reg(),
                    )
                    c_res = tf.schedule(
                        (c_res_ub, tf.empty(Tensor[(M, D), "f32", "rmem"])),
                        op=T.ascend_ub_to_reg(),
                    )
                    c_agg = tf.schedule(
                        (c_agg_ub, tf.empty(Tensor[(M, D), "f32", "rmem"])),
                        op=T.ascend_ub_to_reg(),
                    )
                    x_scaled = tf.schedule(
                        (xf, c_res, tf.empty(Tensor[(M, D), "f32", "rmem"])),
                        op=T.ascend_vmul(),
                    )
                    agg_scaled = tf.schedule(
                        (acc_f, c_agg, tf.empty(Tensor[(M, D), "f32", "rmem"])),
                        op=T.ascend_vmul(),
                    )
                    y_f = tf.schedule(
                        (x_scaled, agg_scaled, tf.empty(Tensor[(M, D), "f32", "rmem"])),
                        op=T.ascend_vadd(),
                    )
                    y_cast = tf.schedule(
                        (y_f, tf.empty(Tensor[(M, D), dtype, "rmem"])),
                        op=T.ascend_vcast(dtype=dtype),
                    )
                    y_ub = tf.schedule(
                        (y_cast, tf.empty(Tensor[(M, D), dtype, "smem"])),
                        op=T.ascend_reg_to_ub(),
                    )
                y_gm = tf.schedule(
                    (y_ub, tf.empty(Tensor[(M, D), dtype, "gmem"])),
                    op=T.ascend_ub_to_gm(),
                )
                return tf.reshape(y_gm, new_shape=(B, S, D))

    return AttentionResidualScheduled


AttentionResidualScheduled = build()
