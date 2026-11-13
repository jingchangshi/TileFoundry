"""Scheduled MIX AIC+AIV HIR MLA module on Ascend950PR."""

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
    """Build a scheduled MIX (Cube + SimtVF + dual_copy) BNSD MLA HIR module."""
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
            with Mesh(("aiv",), (1,), names=("v",)) as _v:
                q_n_m = tf.reshape(q_nope, new_shape=(M, D_nope))
                q_r_m = tf.reshape(q_rope, new_shape=(M, D_rope))
                k_n_m = tf.reshape(k_nope, new_shape=(S_kv, D_nope))
                k_r_m = tf.reshape(k_rope, new_shape=(S_kv, D_rope))
                v_m = tf.reshape(v, new_shape=(S_kv, D_nope))
                scale_ub = tf.full_like(tf.zeros(Tensor[(M, S_kv), "f32", "smem"]), scale)
                mask_ub = tf.schedule(
                    (causal_mask, tf.empty(Tensor[(M, S_kv), "f32", "smem"])),
                    op=T.ascend_gm_to_ub(),
                )
                scores_gm = tf.zeros(Tensor[(M, S_kv), "f32", "gmem"])

                for kt in tf.pipelined(
                    kv_tiles, max_stage=2, max_buffer_versions=2, enable_offset=True
                ):
                    k_start = kt * kv_tile
                    kn_tile = k_n_m[k_start : k_start + kv_tile, :]
                    kr_tile = k_r_m[k_start : k_start + kv_tile, :]
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
                    s_gm_tile = tf.schedule(
                        (s_ub, tf.empty(Tensor[(M, kv_tile), "f32", "gmem"])),
                        op=T.ascend_ub_to_gm(),
                    )
                    scores_gm = tf.insert_slice(scores_gm, s_gm_tile, (0, k_start))

                scores_ub = tf.schedule(
                    (scores_gm, tf.empty(Tensor[(M, S_kv), "f32", "smem"])),
                    op=T.ascend_gm_to_ub(),
                )
                with T.SimtVF():
                    sr = tf.schedule(
                        (scores_ub, tf.empty(Tensor[(M, S_kv), "f32", "rmem"])),
                        op=T.ascend_ub_to_reg(),
                    )
                    scale_r = tf.schedule(
                        (scale_ub, tf.empty(Tensor[(M, S_kv), "f32", "rmem"])),
                        op=T.ascend_ub_to_reg(),
                    )
                    mask_r = tf.schedule(
                        (mask_ub, tf.empty(Tensor[(M, S_kv), "f32", "rmem"])),
                        op=T.ascend_ub_to_reg(),
                    )
                    scaled = tf.schedule(
                        (sr, scale_r, tf.empty(Tensor[(M, S_kv), "f32", "rmem"])),
                        op=T.ascend_vmul(),
                    )
                    masked = tf.schedule(
                        (scaled, mask_r, tf.empty(Tensor[(M, S_kv), "f32", "rmem"])),
                        op=T.ascend_vadd(),
                    )
                    s_max = tf.schedule(
                        (masked, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                        op=T.ascend_vreduce_max(axes=(-1,), keepdim=True),
                    )
                    s_shift = tf.schedule(
                        (masked, s_max, tf.empty(Tensor[(M, S_kv), "f32", "rmem"])),
                        op=T.ascend_vsub(),
                    )
                    ex = tf.schedule(
                        (s_shift, tf.empty(Tensor[(M, S_kv), "f32", "rmem"])),
                        op=T.ascend_vexp(),
                    )
                    ex_sum = tf.schedule(
                        (ex, tf.empty(Tensor[(M, 1), "f32", "rmem"])),
                        op=T.ascend_vreduce_sum(axes=(-1,), keepdim=True),
                    )
                    pr = tf.schedule(
                        (ex, ex_sum, tf.empty(Tensor[(M, S_kv), "f32", "rmem"])),
                        op=T.ascend_vdiv(),
                    )
                    pr_cast = tf.schedule(
                        (pr, tf.empty(Tensor[(M, S_kv), dtype, "rmem"])),
                        op=T.ascend_vcast(dtype=dtype),
                    )
                    p_ub = tf.schedule(
                        (pr_cast, tf.empty(Tensor[(M, S_kv), dtype, "smem"])),
                        op=T.ascend_reg_to_ub(),
                    )

                with Mesh(("aic",), (1,), names=("c2",)) as _c2:
                    p_l1 = tf.schedule(
                        (p_ub, tf.empty(Tensor[(M, S_kv), dtype, "tmem"])),
                        op=T.ascend_dual_copy(),
                    )
                    v_l1 = tf.schedule(
                        (v_m, tf.empty(Tensor[(S_kv, D_nope), dtype, "tmem"])),
                        op=T.ascend_gm_to_l1(),
                    )
                    p_l0 = tf.schedule(
                        (p_l1, tf.empty(Tensor[(M, S_kv), dtype, "tmem"])),
                        op=T.ascend_l1_to_l0a(),
                    )
                    v_l0 = tf.schedule(
                        (v_l1, tf.empty(Tensor[(S_kv, D_nope), dtype, "tmem"])),
                        op=T.ascend_l1_to_l0b(),
                    )
                    o0 = tf.zeros(Tensor[(M, D_nope), "f32", "tmem"])
                    o1 = tf.schedule(
                        (o0, p_l0, v_l0),
                        op=T.ascend_mmad(a_layout="MK", b_layout="KN", init_zero=True),
                    )
                    o_ub = tf.schedule(
                        (o1, tf.empty(Tensor[(M, D_nope), "f32", "smem"])),
                        op=T.ascend_dual_copy(),
                    )
                with T.SimtVF():
                    or_reg = tf.schedule(
                        (o_ub, tf.empty(Tensor[(M, D_nope), "f32", "rmem"])),
                        op=T.ascend_ub_to_reg(),
                    )
                    or_cast = tf.schedule(
                        (or_reg, tf.empty(Tensor[(M, D_nope), dtype, "rmem"])),
                        op=T.ascend_vcast(dtype=dtype),
                    )
                    o_cast_ub = tf.schedule(
                        (or_cast, tf.empty(Tensor[(M, D_nope), dtype, "smem"])),
                        op=T.ascend_reg_to_ub(),
                    )
                out_m = tf.schedule(
                    (o_cast_ub, tf.empty(Tensor[(M, D_nope), dtype, "gmem"])),
                    op=T.ascend_ub_to_gm(),
                )
                return tf.reshape(out_m, new_shape=(B, N_q, S, D_nope))

    return MlaScheduled


MlaScheduled = build()
