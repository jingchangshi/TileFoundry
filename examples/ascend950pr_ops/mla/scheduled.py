"""Scheduled MIX AIC+AIV HIR MLA module with SimdVF and single-loop online softmax."""

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
    dtype: str = "f16",
    kv_tile: int = 8,
):
    """Build a single-loop online FlashMLA Mix (Cube + SimdVF + dual_copy) HIR module."""
    assert B == 1 and N_kv == 1
    assert S_kv % kv_tile == 0
    M = N_q * S
    D_qk = D_nope + D_rope
    kv_tiles = S_kv // kv_tile
    scale = (1.0 / math.sqrt(D_qk)) if scale_value <= 0.0 else float(scale_value)

    @module(entry="forward", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
    class MlaScheduled:
        @func
        def forward(
            q_nope: Tensor[(B, N_q, S, D_nope), dtype],
            q_rope: Tensor[(B, N_q, S, D_rope), dtype],
            k_nope: Tensor[(B, N_kv, S_kv, D_nope), dtype],
            k_rope: Tensor[(B, N_kv, S_kv, D_rope), dtype],
            v: Tensor[(B, N_kv, S_kv, D_nope), dtype],
            causal_mask: Tensor[(M, S_kv), "f32"],
        ) -> Tensor[(B, N_q, S, D_nope), dtype]:
            with Mesh(("aiv",), (2,), names=("v",)) as _v:
                q_n_m = tf.reshape(q_nope, new_shape=(M, D_nope))
                q_r_m = tf.reshape(q_rope, new_shape=(M, D_rope))
                k_n_m = tf.reshape(k_nope, new_shape=(S_kv, D_nope))
                k_r_m = tf.reshape(k_rope, new_shape=(S_kv, D_rope))
                v_m = tf.reshape(v, new_shape=(S_kv, D_nope))
                scale_tile_ub = tf.full_like(
                    tf.zeros(Tensor[(M, kv_tile), "f32", "smem"]), scale
                )
                z_m1 = tf.zeros(Tensor[(M, 1), "f32", "smem"])
                m_ub = tf.full_like(z_m1, -1.0e30)
                l_ub = tf.zeros(Tensor[(M, 1), "f32", "smem"])
                o_acc_ub = tf.zeros(Tensor[(M, D_nope), "f32", "smem"])

                for kt in tf.pipelined(
                    kv_tiles, max_stage=2, max_buffer_versions=2, enable_offset=True
                ):
                    k_start = kt * kv_tile
                    kn_tile = k_n_m[k_start : k_start + kv_tile, :]
                    kr_tile = k_r_m[k_start : k_start + kv_tile, :]
                    v_tile = v_m[k_start : k_start + kv_tile, :]
                    mask_tile = causal_mask[:, k_start : k_start + kv_tile]
                    mask_ub = tf.schedule(
                        (mask_tile, tf.empty(Tensor[(M, kv_tile), "f32", "smem"])),
                        op=T.ascend_gm_to_ub(),
                    )
                    with Mesh(("aic",), (1,), names=("c",)) as _c:
                        qn_l1 = tf.schedule(
                            (q_n_m, tf.empty(Tensor[(M, D_nope), dtype, "tmem"])),
                            op=T.ascend_gm_to_l1(),
                        )
                        qr_l1 = tf.schedule(
                            (q_r_m, tf.empty(Tensor[(M, D_rope), dtype, "tmem"])),
                            op=T.ascend_gm_to_l1(),
                        )
                        kn_l1 = tf.schedule(
                            (kn_tile, tf.empty(Tensor[(kv_tile, D_nope), dtype, "tmem"])),
                            op=T.ascend_gm_to_l1(),
                        )
                        kr_l1 = tf.schedule(
                            (kr_tile, tf.empty(Tensor[(kv_tile, D_rope), dtype, "tmem"])),
                            op=T.ascend_gm_to_l1(),
                        )
                        qn_l0 = tf.schedule(
                            (qn_l1, tf.empty(Tensor[(M, D_nope), dtype, "tmem"])),
                            op=T.ascend_l1_to_l0a(),
                        )
                        qr_l0 = tf.schedule(
                            (qr_l1, tf.empty(Tensor[(M, D_rope), dtype, "tmem"])),
                            op=T.ascend_l1_to_l0a(),
                        )
                        kn_l0 = tf.schedule(
                            (kn_l1, tf.empty(Tensor[(kv_tile, D_nope), dtype, "tmem"])),
                            op=T.ascend_l1_to_l0b(),
                        )
                        kr_l0 = tf.schedule(
                            (kr_l1, tf.empty(Tensor[(kv_tile, D_rope), dtype, "tmem"])),
                            op=T.ascend_l1_to_l0b(),
                        )
                        s0 = tf.zeros(Tensor[(M, kv_tile), "f32", "tmem"])
                        s1 = tf.schedule(
                            (s0, qn_l0, kn_l0),
                            op=T.ascend_mmad(a_layout="MK", b_layout="NK", init_zero=True),
                        )
                        s2 = tf.schedule(
                            (s1, qr_l0, kr_l0),
                            op=T.ascend_mmad(a_layout="MK", b_layout="NK", init_zero=False),
                        )
                        s_ub = tf.schedule(
                            (s2, tf.empty(Tensor[(M, kv_tile), "f32", "smem"])),
                            op=T.ascend_dual_copy(),
                        )

                    with T.SimdVF():
                        sr = tf.schedule(
                            (s_ub, tf.empty(Tensor[(M, kv_tile), "f32", "rmem"])),
                            op=T.ascend_ub_to_reg(),
                        )
                        scale_r = tf.schedule(
                            (scale_tile_ub, tf.empty(Tensor[(M, kv_tile), "f32", "rmem"])),
                            op=T.ascend_ub_to_reg(),
                        )
                        mask_r = tf.schedule(
                            (mask_ub, tf.empty(Tensor[(M, kv_tile), "f32", "rmem"])),
                            op=T.ascend_ub_to_reg(),
                        )
                        scaled = tf.schedule(
                            (sr, scale_r, tf.empty(Tensor[(M, kv_tile), "f32", "rmem"])),
                            op=T.ascend_vmul(),
                        )
                        masked = tf.schedule(
                            (scaled, mask_r, tf.empty(Tensor[(M, kv_tile), "f32", "rmem"])),
                            op=T.ascend_vadd(),
                        )
                        m_tile = tf.schedule(
                            (masked, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_vreduce_max(axes=(-1,), keepdim=True),
                        )
                        m_prev = tf.schedule(
                            (m_ub, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_ub_to_reg(),
                        )
                        l_prev = tf.schedule(
                            (l_ub, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_ub_to_reg(),
                        )
                        m_next = tf.schedule(
                            (m_prev, m_tile, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
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
                        s_shift = tf.schedule(
                            (masked, m_next, tf.empty(Tensor[(M, kv_tile), "f32", "rmem"])),
                            op=T.ascend_vsub(),
                        )
                        p_f32 = tf.schedule(
                            (s_shift, tf.empty(Tensor[(M, kv_tile), "f32", "rmem"])),
                            op=T.ascend_vexp(),
                        )
                        row_sum = tf.schedule(
                            (p_f32, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_vreduce_sum(axes=(-1,), keepdim=True),
                        )
                        l_scaled = tf.schedule(
                            (l_prev, alpha_r, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_vmul(),
                        )
                        l_next = tf.schedule(
                            (l_scaled, row_sum, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_vadd(),
                        )
                        p_cast = tf.schedule(
                            (p_f32, tf.empty(Tensor[(M, kv_tile), dtype, "rmem"])),
                            op=T.ascend_vcast(dtype=dtype),
                        )
                        m_ub = tf.schedule(
                            (m_next, tf.empty(Tensor[(M, 1), "f32", "smem"])),
                            op=T.ascend_reg_to_ub(),
                        )
                        l_ub = tf.schedule(
                            (l_next, tf.empty(Tensor[(M, 1), "f32", "smem"])),
                            op=T.ascend_reg_to_ub(),
                        )
                        alpha_ub = tf.schedule(
                            (alpha_r, tf.empty(Tensor[(M, 1), "f32", "smem"])),
                            op=T.ascend_reg_to_ub(),
                        )
                        p_ub = tf.schedule(
                            (p_cast, tf.empty(Tensor[(M, kv_tile), dtype, "smem"])),
                            op=T.ascend_reg_to_ub(),
                        )

                    with Mesh(("aic",), (1,), names=("c2",)) as _c2:
                        p_l1 = tf.schedule(
                            (p_ub, tf.empty(Tensor[(M, kv_tile), dtype, "tmem"])),
                            op=T.ascend_dual_copy(),
                        )
                        v_l1 = tf.schedule(
                            (v_tile, tf.empty(Tensor[(kv_tile, D_nope), dtype, "tmem"])),
                            op=T.ascend_gm_to_l1(),
                        )
                        p_l0 = tf.schedule(
                            (p_l1, tf.empty(Tensor[(M, kv_tile), dtype, "tmem"])),
                            op=T.ascend_l1_to_l0a(),
                        )
                        v_l0 = tf.schedule(
                            (v_l1, tf.empty(Tensor[(kv_tile, D_nope), dtype, "tmem"])),
                            op=T.ascend_l1_to_l0b(),
                        )
                        o0 = tf.zeros(Tensor[(M, D_nope), "f32", "tmem"])
                        o1 = tf.schedule(
                            (o0, p_l0, v_l0),
                            op=T.ascend_mmad(a_layout="MK", b_layout="KN", init_zero=True),
                        )
                        o_tile_ub = tf.schedule(
                            (o1, tf.empty(Tensor[(M, D_nope), "f32", "smem"])),
                            op=T.ascend_dual_copy(),
                        )

                    with T.SimdVF():
                        o_prev = tf.schedule(
                            (o_acc_ub, tf.empty(Tensor[(M, D_nope), "f32", "rmem"])),
                            op=T.ascend_ub_to_reg(),
                        )
                        alpha_reg = tf.schedule(
                            (alpha_ub, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                            op=T.ascend_ub_to_reg(),
                        )
                        o_tile_reg = tf.schedule(
                            (o_tile_ub, tf.empty(Tensor[(M, D_nope), "f32", "rmem"])),
                            op=T.ascend_ub_to_reg(),
                        )
                        o_scaled = tf.schedule(
                            (o_prev, alpha_reg, tf.empty(Tensor[(M, D_nope), "f32", "rmem"])),
                            op=T.ascend_vmul(),
                        )
                        o_next = tf.schedule(
                            (o_scaled, o_tile_reg, tf.empty(Tensor[(M, D_nope), "f32", "rmem"])),
                            op=T.ascend_vadd(),
                        )
                        o_acc_ub = tf.schedule(
                            (o_next, tf.empty(Tensor[(M, D_nope), "f32", "smem"])),
                            op=T.ascend_reg_to_ub(),
                        )

                with T.SimdVF():
                    o_final_in = tf.schedule(
                        (o_acc_ub, tf.empty(Tensor[(M, D_nope), "f32", "rmem"])),
                        op=T.ascend_ub_to_reg(),
                    )
                    l_final = tf.schedule(
                        (l_ub, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                        op=T.ascend_ub_to_reg(),
                    )
                    o_norm = tf.schedule(
                        (o_final_in, l_final, tf.empty(Tensor[(M, D_nope), "f32", "rmem"])),
                        op=T.ascend_vdiv(),
                    )
                    o_cast = tf.schedule(
                        (o_norm, tf.empty(Tensor[(M, D_nope), dtype, "rmem"])),
                        op=T.ascend_vcast(dtype=dtype),
                    )
                    o_cast_ub = tf.schedule(
                        (o_cast, tf.empty(Tensor[(M, D_nope), dtype, "smem"])),
                        op=T.ascend_reg_to_ub(),
                    )
                out_m = tf.schedule(
                    (o_cast_ub, tf.empty(Tensor[(M, D_nope), dtype, "gmem"])),
                    op=T.ascend_ub_to_gm(),
                )
                return tf.reshape(out_m, new_shape=(B, N_q, S, D_nope))

    return MlaScheduled


MlaScheduled = build()
