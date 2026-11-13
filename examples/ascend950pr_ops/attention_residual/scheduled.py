"""Scheduled HIR AttentionResidual module with SimdVF and single-pass online softmax."""

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
    """Build a single-pass online-softmax scheduled HIR AttentionResidual module."""
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
            with Mesh(("aiv",), (2,), names=("v",)) as _v:
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

                # Precompute fused RMSNorm key weight kw = w * gamma once per row.
                with T.SimdVF():
                    wr = tf.schedule(
                        (w_ub, tf.empty(Tensor[(1, D), dtype, "rmem"])),
                        op=T.ascend_ub_to_reg(),
                    )
                    gr = tf.schedule(
                        (g_ub, tf.empty(Tensor[(1, D), dtype, "rmem"])),
                        op=T.ascend_ub_to_reg(),
                    )
                    wf = tf.schedule(
                        (wr, tf.empty(Tensor[(1, D), "f32", "rmem"])),
                        op=T.ascend_vcast(dtype="f32"),
                    )
                    gf = tf.schedule(
                        (gr, tf.empty(Tensor[(1, D), "f32", "rmem"])),
                        op=T.ascend_vcast(dtype="f32"),
                    )
                    kw_r = tf.schedule(
                        (wf, gf, tf.empty(Tensor[(1, D), "f32", "rmem"])),
                        op=T.ascend_vmul(),
                    )
                    kw_ub = tf.schedule(
                        (kw_r, tf.empty(Tensor[(1, D), "f32", "smem"])),
                        op=T.ascend_reg_to_ub(),
                    )

                # Online softmax streaming state across depth N (single pass over V).
                m_ub = tf.full_like(z_m1, -1.0e30)
                l_ub = tf.zeros(Tensor[(M, 1), "f32", "smem"])
                acc_ub = tf.zeros(Tensor[(M, D), "f32", "smem"])

                for n in tf.pipelined(N, max_stage=2, max_buffer_versions=2, enable_offset=True):
                    v_ub = tf.schedule(
                        (v_flat[n, :, :], tf.empty(Tensor[(M, D), dtype, "smem"])),
                        op=T.ascend_gm_to_ub(),
                    )
                    with T.SimdVF():
                        vr = tf.schedule(
                            (v_ub, tf.empty(Tensor[(M, D), dtype, "rmem"])),
                            op=T.ascend_ub_to_reg(),
                        )
                        kw_reg = tf.schedule(
                            (kw_ub, tf.empty(Tensor[(1, D), "f32", "rmem"])),
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
                        weighted = tf.schedule(
                            (v_norm, kw_reg, tf.empty(Tensor[(M, D), "f32", "rmem"])),
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
                        m_prev = tf.schedule(
                            (m_ub, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_ub_to_reg(),
                        )
                        l_prev = tf.schedule(
                            (l_ub, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_ub_to_reg(),
                        )
                        acc_prev = tf.schedule(
                            (acc_ub, tf.empty(Tensor[(M, D), "f32", "rmem"])),
                            op=T.ascend_ub_to_reg(),
                        )
                        m_next = tf.schedule(
                            (m_prev, s_r, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_vmax(),
                        )
                        alpha_diff = tf.schedule(
                            (m_prev, m_next, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_vsub(),
                        )
                        alpha_r = tf.schedule(
                            (alpha_diff, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_vexp(),
                        )
                        p_diff = tf.schedule(
                            (s_r, m_next, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_vsub(),
                        )
                        p_r = tf.schedule(
                            (p_diff, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_vexp(),
                        )
                        l_scaled = tf.schedule(
                            (l_prev, alpha_r, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_vmul(),
                        )
                        l_next = tf.schedule(
                            (l_scaled, p_r, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_vadd(),
                        )
                        acc_scaled = tf.schedule(
                            (acc_prev, alpha_r, tf.empty(Tensor[(M, D), "f32", "rmem"])),
                            op=T.ascend_vmul(),
                        )
                        v_contrib = tf.schedule(
                            (vf, p_r, tf.empty(Tensor[(M, D), "f32", "rmem"])),
                            op=T.ascend_vmul(),
                        )
                        acc_next = tf.schedule(
                            (acc_scaled, v_contrib, tf.empty(Tensor[(M, D), "f32", "rmem"])),
                            op=T.ascend_vadd(),
                        )
                        m_ub = tf.schedule(
                            (m_next, tf.empty(Tensor[(M, 1), "f32", "smem"])),
                            op=T.ascend_reg_to_ub(),
                        )
                        l_ub = tf.schedule(
                            (l_next, tf.empty(Tensor[(M, 1), "f32", "smem"])),
                            op=T.ascend_reg_to_ub(),
                        )
                        acc_ub = tf.schedule(
                            (acc_next, tf.empty(Tensor[(M, D), "f32", "smem"])),
                            op=T.ascend_reg_to_ub(),
                        )

                with T.SimdVF():
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
                    l_f = tf.schedule(
                        (l_ub, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                        op=T.ascend_ub_to_reg(),
                    )
                    agg_f = tf.schedule(
                        (acc_f, l_f, tf.empty(Tensor[(M, D), "f32", "rmem"])),
                        op=T.ascend_vdiv(),
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
                        (agg_f, c_agg, tf.empty(Tensor[(M, D), "f32", "rmem"])),
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
