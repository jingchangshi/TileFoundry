"""TileLang Ascend950PR MIX SimdVF kernel implementation for Multi-Head Latent Attention."""

import math

import tilelang
import tilelang.ascend.language as T
import torch

from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS

_KERNEL_CACHE: dict[tuple[object, ...], object] = {}


def _build_mla_prim_func(
    n_q: int,
    n_kv: int,
    d_nope: int,
    d_rope: int,
    tl_dtype: str,
    input_layout: str,
    is_causal: bool,
    bm: int,
    aligned_group_q: bool,
    schedule_mode: str,
):
    if d_nope % 64 != 0 or d_rope % 64 != 0:
        raise ValueError(f"MLA requires d_nope ({d_nope}) and d_rope ({d_rope}) divisible by 64")
    if bm not in (32, 64, 128):
        raise ValueError(f"MLA unsupported bm={bm}")
    rows_per_sid = bm // 2
    bn = 64
    d0 = min(d_nope, 256)
    d1 = d_nope - d0
    d1_alloc = max(d1, 64)
    g_ratio = n_q // n_kv
    d_nope_chunks = d_nope // 64
    d0_chunks = d0 // 64
    d1_chunks = d1 // 64
    d_rope_chunks = d_rope // 64
    num_cores = 28

    num_stages = 2 if schedule_mode == "tuned" else 1
    pipe_ann = {"enable_offset": True} if schedule_mode == "tuned" else None

    b_dim = T.dynamic("b_dim")
    q_total_rows = T.dynamic("q_total_rows")
    s_kv = T.dynamic("s_kv")

    k_nope_shape = (
        (b_dim, s_kv, n_kv, d_nope) if input_layout == "BSND" else (b_dim, n_kv, s_kv, d_nope)
    )
    k_rope_shape = (
        (b_dim, s_kv, n_kv, d_rope) if input_layout == "BSND" else (b_dim, n_kv, s_kv, d_rope)
    )

    @T.prim_func
    def main(
        Q_nope: T.Tensor((q_total_rows, d_nope), tl_dtype),
        Q_rope: T.Tensor((q_total_rows, d_rope), tl_dtype),
        K_nope: T.Tensor(k_nope_shape, tl_dtype),
        K_rope: T.Tensor(k_rope_shape, tl_dtype),
        V: T.Tensor(k_nope_shape, tl_dtype),
        Out: T.Tensor((q_total_rows, d_nope), tl_dtype),
        scale_val: T.float32,
    ):
        with T.MixedKernel(num_cores, sids=2) as (bx, sid):
            q_nope_l1 = T.alloc_l1((bm, d_nope), tl_dtype)
            q_rope_l1 = T.alloc_l1((bm, d_rope), tl_dtype)
            k_nope_l1 = T.alloc_l1((bn, d_nope), tl_dtype)
            k_rope_l1 = T.alloc_l1((bn, d_rope), tl_dtype)
            p_l1 = T.alloc_l1((bm, bn), tl_dtype)
            v0_l1 = T.alloc_l1((d0, bn), tl_dtype)
            v1_l1 = T.alloc_l1((d1_alloc, bn), tl_dtype)

            s_l0c = T.alloc_l0c((bm, bn), "float32")
            o_l0c = T.alloc_l0c((bm, d0), "float32")
            o1_l0c = o_l0c if (d1 == 0 or d1 == d0) else T.alloc_l0c((bm, d1_alloc), "float32")

            if not aligned_group_q:
                q_nope_ub = T.alloc_shared((rows_per_sid, d_nope), tl_dtype)
                q_rope_ub = T.alloc_shared((rows_per_sid, d_rope), tl_dtype)
            S_ub = T.alloc_shared((rows_per_sid, bn), "float32")
            P_ub = T.alloc_shared((rows_per_sid, bn), tl_dtype)
            O_tmp_ub = T.alloc_shared((rows_per_sid, d0), "float32")
            O_tmp1_ub = (
                O_tmp_ub
                if (d1 == 0 or d1 == d0)
                else T.alloc_shared((rows_per_sid, d1_alloc), "float32")
            )
            O_ub = T.alloc_shared((rows_per_sid, d_nope), "float32")
            out_tile_rows = 8 if (bm == 128 and rows_per_sid % 8 == 0) else rows_per_sid
            out_ub = T.alloc_shared((out_tile_rows, d_nope), tl_dtype)
            m_ub = T.alloc_shared((rows_per_sid, 8), "float32")
            l_ub = T.alloc_shared((rows_per_sid, 8), "float32")
            alpha_ub = T.alloc_shared((rows_per_sid, 8), "float32")

            pinned = {
                q_nope_l1: 1,
                q_rope_l1: 1,
                p_l1: 1,
                s_l0c: 1,
                o_l0c: 1,
                S_ub: 1,
                P_ub: 1,
                O_tmp_ub: 1,
                O_ub: 1,
                out_ub: 1,
                m_ub: 1,
                l_ub: 1,
                alpha_ub: 1,
            }
            if not aligned_group_q:
                pinned[q_nope_ub] = 1
                pinned[q_rope_ub] = 1
            if d1 > 0 and d1 != d0:
                pinned[o1_l0c] = 1
                pinned[O_tmp1_ub] = 1
            if schedule_mode != "tuned":
                pinned.update(
                    {
                        k_nope_l1: 1,
                        k_rope_l1: 1,
                        v0_l1: 1,
                        v1_l1: 1,
                    }
                )
            T.annotate_buffer_versions(pinned)
            if bm == 128:
                T.annotate_unlimit_memory("shared")

            s_dim = q_total_rows // (b_dim * n_q)
            q_rows = s_dim * g_ratio
            m_tiles_per_group = (q_rows + bm - 1) // bm
            total_groups = b_dim * n_kv
            total_m_tiles = total_groups * m_tiles_per_group
            base_tiles = total_m_tiles // num_cores
            rem_tiles = total_m_tiles % num_cores

            t_start = bx * base_tiles + T.min(bx, rem_tiles)
            t_count = base_tiles + T.if_then_else(bx < rem_tiles, 1, 0)
            for local_t in T.serial(t_count):
                tile_id = t_start + local_t
                group_id = tile_id // m_tiles_per_group
                mt = tile_id % m_tiles_per_group
                b_idx = group_id // n_kv
                kv_h = group_id % n_kv
                q_base = mt * bm

                if aligned_group_q:
                    q_gm_base = group_id * q_rows + q_base
                    T.copy(
                        Q_nope[q_gm_base : q_gm_base + bm, 0:d_nope],
                        q_nope_l1,
                    )
                    T.copy(
                        Q_rope[q_gm_base : q_gm_base + bm, 0:d_rope],
                        q_rope_l1,
                    )
                else:
                    with T.SimdVF():
                        m32 = T.simd.pset(32, "PAT_ALL")
                        m16 = T.simd.pset(16, "PAT_ALL")
                        v_zero16 = T.simd.vcvt(
                            T.simd.vdup(T.float32(0.0), "float32", mask=m32),
                            tl_dtype,
                            mask=m16,
                            part=0,
                        )
                        for r in T.serial(rows_per_sid):
                            for c in T.serial(d_nope_chunks):
                                T.simd.vsts(
                                    q_nope_ub[r, c * 64],
                                    v_zero16,
                                    mask=m32,
                                    dist="PK_B32",
                                    extent=64,
                                )
                            for c in T.serial(d_rope_chunks):
                                T.simd.vsts(
                                    q_rope_ub[r, c * 64],
                                    v_zero16,
                                    mask=m32,
                                    dist="PK_B32",
                                    extent=64,
                                )

                    for r in T.serial(rows_per_sid):
                        q_row = q_base + sid * rows_per_sid + r
                        if q_row < q_rows:
                            if input_layout == "BSND":
                                s_i = q_row // g_ratio
                                h_loc = q_row % g_ratio
                                q_h = kv_h * g_ratio + h_loc
                                gm_r = (b_idx * s_dim + s_i) * n_q + q_h
                            else:
                                gm_r = group_id * q_rows + q_row
                            T.copy(Q_nope[gm_r, 0:d_nope], q_nope_ub[r, 0:d_nope])
                            T.copy(Q_rope[gm_r, 0:d_rope], q_rope_ub[r, 0:d_rope])

                    T.dual_copy(q_nope_ub, q_nope_l1)
                    T.dual_copy(q_rope_ub, q_rope_l1)

                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    v_neg = T.simd.vdup(T.float32(-1e38), "float32", mask=m32)
                    v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                    for r in T.serial(rows_per_sid):
                        T.simd.vsts(m_ub[r, 0], v_neg, mask=m32, dist="ONEPT_B32")
                        T.simd.vsts(l_ub[r, 0], v_zero, mask=m32, dist="ONEPT_B32")
                        for c in T.serial(d_nope_chunks):
                            T.simd.vsts(
                                O_ub[r, c * 64], v_zero, mask=m32, dist="NORM_B32"
                            )

                max_kv_blocks = (s_kv + bn - 1) // bn
                if is_causal and schedule_mode == "tuned":
                    last_valid = T.min(q_rows - 1, q_base + bm - 1)
                    if input_layout == "BSND":
                        max_s = last_valid // g_ratio
                    else:
                        max_s = T.if_then_else(
                            (q_base // s_dim) == (last_valid // s_dim),
                            last_valid % s_dim,
                            s_dim - 1,
                        )
                    kv_limit = T.max(
                        1,
                        T.min(max_kv_blocks, (max_s + s_kv - s_dim) // bn + 1),
                    )
                else:
                    kv_limit = max_kv_blocks

                for kv in T.Pipelined(
                    kv_limit, num_stages=num_stages, annotations=pipe_ann
                ):
                    if input_layout == "BSND":
                        T.copy(
                            K_nope[b_idx, kv * bn : kv * bn + bn, kv_h, 0:d_nope],
                            k_nope_l1,
                        )
                        T.copy(
                            K_rope[b_idx, kv * bn : kv * bn + bn, kv_h, 0:d_rope],
                            k_rope_l1,
                        )
                        T.copy(
                            V[b_idx, kv * bn : kv * bn + bn, kv_h, 0:d0],
                            v0_l1,
                            transpose=True,
                        )
                        if d1 > 0:
                            T.copy(
                                V[b_idx, kv * bn : kv * bn + bn, kv_h, d0:d_nope],
                                v1_l1,
                                transpose=True,
                            )
                    else:
                        T.copy(
                            K_nope[b_idx, kv_h, kv * bn : kv * bn + bn, 0:d_nope],
                            k_nope_l1,
                        )
                        T.copy(
                            K_rope[b_idx, kv_h, kv * bn : kv * bn + bn, 0:d_rope],
                            k_rope_l1,
                        )
                        T.copy(
                            V[b_idx, kv_h, kv * bn : kv * bn + bn, 0:d0],
                            v0_l1,
                            transpose=True,
                        )
                        if d1 > 0:
                            T.copy(
                                V[b_idx, kv_h, kv * bn : kv * bn + bn, d0:d_nope],
                                v1_l1,
                                transpose=True,
                            )

                    T.gemm(q_nope_l1, k_nope_l1, s_l0c, transpose_B=True, clear_accum=True)
                    T.gemm(q_rope_l1, k_rope_l1, s_l0c, transpose_B=True, clear_accum=False)
                    T.dual_copy(s_l0c, S_ub)

                    with T.SimdVF():
                        m32 = T.simd.pset(32, "PAT_ALL")
                        m16 = T.simd.pset(16, "PAT_ALL")
                        v_neg_inf = T.simd.vdup(T.float32(-1e38), "float32", mask=m32)
                        v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                        col_idx = T.simd.vci(kv * bn, "int32")
                        for r in T.serial(rows_per_sid):
                            q_row = q_base + sid * rows_per_sid + r
                            if input_layout == "BSND":
                                s_idx = q_row // g_ratio
                            else:
                                s_idx = q_row % s_dim
                            raw_limit = (
                                T.min(s_kv, s_idx + s_kv - s_dim + 1)
                                if is_causal
                                else s_kv
                            )
                            limit_col = T.if_then_else(q_row < q_rows, raw_limit, 0)
                            valid_mask = T.simd.vcmps(col_idx, limit_col, mask=m32, op="lt")
                            s_vec = T.simd.vld(S_ub[r, 0], dist="NORM_B32")
                            s_scaled = T.simd.vmuls(s_vec, scale_val, mask=m32)
                            s_masked = T.simd.vsel(s_scaled, v_neg_inf, valid_mask)
                            m_tile_v = T.simd.vdupv(
                                T.simd.vcmax(s_masked, mask=m32),
                                mask=m32,
                                pos="POS_LOWEST",
                            )
                            m_old_v = T.simd.vld(m_ub[r, 0], dist="BRC_B32")
                            l_old_v = T.simd.vld(l_ub[r, 0], dist="BRC_B32")
                            m_new_v = T.simd.vmax(m_old_v, m_tile_v, mask=m32)
                            m_safe_v = T.simd.vmaxs(
                                m_new_v, T.float32(-1e30), mask=m32
                            )
                            alpha_raw = T.simd.vexp(
                                T.simd.vsub(m_old_v, m_safe_v, mask=m32), mask=m32
                            )
                            old_valid = T.simd.vcmps(
                                m_old_v, T.float32(-1e30), mask=m32, op="gt"
                            )
                            alpha_v = T.simd.vsel(alpha_raw, v_zero, old_valid)
                            p_raw = T.simd.vexp(
                                T.simd.vsub(s_masked, m_safe_v, mask=m32), mask=m32
                            )
                            new_valid = T.simd.vcmps(
                                m_new_v, T.float32(-1e30), mask=m32, op="gt"
                            )
                            p_mask = T.simd.pand(valid_mask, new_valid, m32)
                            p_f32 = T.simd.vsel(p_raw, v_zero, p_mask)
                            row_sum_v = T.simd.vdupv(
                                T.simd.vcadd(p_f32, mask=m32),
                                mask=m32,
                                pos="POS_LOWEST",
                            )
                            l_new_v = T.simd.vadd(
                                T.simd.vmul(l_old_v, alpha_v, mask=m32),
                                row_sum_v,
                                mask=m32,
                            )
                            T.simd.vsts(m_ub[r, 0], m_new_v, mask=m32, dist="ONEPT_B32")
                            T.simd.vsts(l_ub[r, 0], l_new_v, mask=m32, dist="ONEPT_B32")
                            T.simd.vsts(alpha_ub[r, 0], alpha_v, mask=m32, dist="ONEPT_B32")
                            p_16 = T.simd.vcvt(p_f32, tl_dtype, mask=m16, part=0)
                            T.simd.vsts(
                                P_ub[r, 0], p_16, mask=m32, dist="PK_B32", extent=64
                            )

                    T.dual_copy(P_ub, p_l1)
                    T.gemm(p_l1, v0_l1, o_l0c, transpose_B=True, clear_accum=True)
                    T.dual_copy(o_l0c, O_tmp_ub)
                    with T.SimdVF():
                        m32 = T.simd.pset(32, "PAT_ALL")
                        for r in T.serial(rows_per_sid):
                            alpha_v = T.simd.vld(alpha_ub[r, 0], dist="BRC_B32")
                            for c in T.serial(d0_chunks):
                                off = c * 64
                                o_old = T.simd.vld(O_ub[r, off], dist="NORM_B32")
                                o_tmp = T.simd.vld(O_tmp_ub[r, off], dist="NORM_B32")
                                o_new = T.simd.vadd(
                                    T.simd.vmul(o_old, alpha_v, mask=m32),
                                    o_tmp,
                                    mask=m32,
                                )
                                T.simd.vsts(
                                    O_ub[r, off], o_new, mask=m32, dist="NORM_B32"
                                )

                    if d1 > 0:
                        T.gemm(p_l1, v1_l1, o1_l0c, transpose_B=True, clear_accum=True)
                        T.dual_copy(o1_l0c, O_tmp1_ub)
                        with T.SimdVF():
                            m32 = T.simd.pset(32, "PAT_ALL")
                            for r in T.serial(rows_per_sid):
                                alpha_v = T.simd.vld(alpha_ub[r, 0], dist="BRC_B32")
                                for c in T.serial(d1_chunks):
                                    off = d0 + c * 64
                                    o_old = T.simd.vld(O_ub[r, off], dist="NORM_B32")
                                    o_tmp = T.simd.vld(
                                        O_tmp1_ub[r, c * 64], dist="NORM_B32"
                                    )
                                    o_new = T.simd.vadd(
                                        T.simd.vmul(o_old, alpha_v, mask=m32),
                                        o_tmp,
                                        mask=m32,
                                    )
                                    T.simd.vsts(
                                        O_ub[r, off], o_new, mask=m32, dist="NORM_B32"
                                    )

                if aligned_group_q and out_tile_rows < rows_per_sid:
                    out_row_blocks = rows_per_sid // out_tile_rows
                    for rb in T.serial(out_row_blocks):
                        with T.SimdVF():
                            m32 = T.simd.pset(32, "PAT_ALL")
                            m16 = T.simd.pset(16, "PAT_ALL")
                            v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
                            v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                            for tr in T.serial(out_tile_rows):
                                r = rb * out_tile_rows + tr
                                l_vec = T.simd.vld(l_ub[r, 0], dist="BRC_B32")
                                l_pos = T.simd.vcmps(
                                    l_vec, T.float32(0.0), mask=m32, op="gt"
                                )
                                l_safe = T.simd.vsel(l_vec, v_one, l_pos)
                                inv_l = T.simd.vsel(
                                    T.simd.vdiv(v_one, l_safe, mask=m32), v_zero, l_pos
                                )
                                for c in T.serial(d_nope_chunks):
                                    off = c * 64
                                    o_f = T.simd.vmul(
                                        T.simd.vld(O_ub[r, off], dist="NORM_B32"),
                                        inv_l,
                                        mask=m32,
                                    )
                                    o_16 = T.simd.vcvt(o_f, tl_dtype, mask=m16, part=0)
                                    T.simd.vsts(
                                        out_ub[tr, off],
                                        o_16,
                                        mask=m32,
                                        dist="PK_B32",
                                        extent=64,
                                    )
                        out_gm_base = (
                            group_id * q_rows
                            + q_base
                            + sid * rows_per_sid
                            + rb * out_tile_rows
                        )
                        T.copy(
                            out_ub,
                            Out[out_gm_base : out_gm_base + out_tile_rows, 0:d_nope],
                        )
                else:
                    with T.SimdVF():
                        m32 = T.simd.pset(32, "PAT_ALL")
                        m16 = T.simd.pset(16, "PAT_ALL")
                        v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
                        v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                        for r in T.serial(rows_per_sid):
                            l_vec = T.simd.vld(l_ub[r, 0], dist="BRC_B32")
                            l_pos = T.simd.vcmps(
                                l_vec, T.float32(0.0), mask=m32, op="gt"
                            )
                            l_safe = T.simd.vsel(l_vec, v_one, l_pos)
                            inv_l = T.simd.vsel(
                                T.simd.vdiv(v_one, l_safe, mask=m32), v_zero, l_pos
                            )
                            for c in T.serial(d_nope_chunks):
                                off = c * 64
                                o_f = T.simd.vmul(
                                    T.simd.vld(O_ub[r, off], dist="NORM_B32"),
                                    inv_l,
                                    mask=m32,
                                )
                                o_16 = T.simd.vcvt(o_f, tl_dtype, mask=m16, part=0)
                                T.simd.vsts(
                                    out_ub[r, off],
                                    o_16,
                                    mask=m32,
                                    dist="PK_B32",
                                    extent=64,
                                )

                    if aligned_group_q:
                        out_gm_base = group_id * q_rows + q_base + sid * rows_per_sid
                        T.copy(
                            out_ub,
                            Out[out_gm_base : out_gm_base + rows_per_sid, 0:d_nope],
                        )
                    else:
                        for r in T.serial(rows_per_sid):
                            q_row = q_base + sid * rows_per_sid + r
                            if q_row < q_rows:
                                if input_layout == "BSND":
                                    s_i = q_row // g_ratio
                                    h_loc = q_row % g_ratio
                                    q_h = kv_h * g_ratio + h_loc
                                    out_r = (b_idx * s_dim + s_i) * n_q + q_h
                                else:
                                    out_r = group_id * q_rows + q_row
                                T.copy(out_ub[r, 0:d_nope], Out[out_r, 0:d_nope])

    return main


def _choose_mla_bm(
    b_dim: int,
    n_kv: int,
    q_rows: int,
    d_nope: int,
    input_layout: str,
    schedule_mode: str,
) -> tuple[int, bool]:
    if schedule_mode != "tuned":
        return 32, False
    aligned = input_layout == "BNSD" or n_kv == 1
    if (
        input_layout == "BSND"
        and n_kv == 1
        and (d_nope <= 256 or d_nope == 512)
        and q_rows >= 128
        and (q_rows % 128) == 0
        and (b_dim * n_kv * (q_rows // 128)) >= 14
    ):
        return 128, True
    m_tiles_64 = (q_rows + 63) // 64
    if q_rows >= 64 and (b_dim * n_kv * m_tiles_64) >= 14:
        return 64, (aligned and (q_rows % 64) == 0)
    return 32, (aligned and (q_rows % 32) == 0)


def _get_kernel(
    n_q: int,
    n_kv: int,
    d_nope: int,
    d_rope: int,
    tl_dtype: str,
    input_layout: str,
    is_causal: bool,
    bm: int,
    aligned_group_q: bool,
    schedule_mode: str,
):
    key = (
        n_q,
        n_kv,
        d_nope,
        d_rope,
        tl_dtype,
        input_layout,
        bool(is_causal),
        int(bm),
        bool(aligned_group_q),
        schedule_mode,
    )
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        prog = _build_mla_prim_func(*key)
        flags = list(PURE_SIMDVF_UBUF_FLAGS) if schedule_mode == "tuned" else None
        compiled = tilelang.compile(prog, target="ascend", out_idx=[5], compile_flags=flags)
        _KERNEL_CACHE[key] = compiled
    return compiled


def mla(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    v: torch.Tensor,
    numKVHeads: int = 1,
    scaleValue: float = -1.0,
    inputLayout: str = "BSND",
    is_causal: bool = False,
    *,
    schedule_mode: str = "tuned",
) -> torch.Tensor:
    """Compute Multi-Head Latent Attention on Ascend950PR using a MIX SimdVF kernel."""
    if schedule_mode not in ("baseline", "tuned"):
        raise ValueError(f"unsupported schedule_mode {schedule_mode!r}")
    if inputLayout not in ("BSND", "BNSD"):
        raise ValueError(f"unsupported inputLayout {inputLayout!r}")
    if q_nope.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported dtype {q_nope.dtype}")

    if inputLayout == "BSND":
        b_dim, s_dim, n_q, d_nope = q_nope.shape
        s_kv, n_kv = k_nope.shape[1], k_nope.shape[2]
    else:
        b_dim, n_q, s_dim, d_nope = q_nope.shape
        n_kv, s_kv = k_nope.shape[1], k_nope.shape[2]
    if int(numKVHeads) != n_kv:
        raise ValueError(f"numKVHeads={numKVHeads} does not match k_nope heads={n_kv}")

    d_rope = q_rope.shape[-1]
    scale = (1.0 / math.sqrt(d_nope + d_rope)) if scaleValue <= 0 else float(scaleValue)
    tl_dtype = "bfloat16" if q_nope.dtype == torch.bfloat16 else "float16"

    g_ratio = n_q // n_kv
    q_rows = s_dim * g_ratio
    bm, aligned_group_q = _choose_mla_bm(b_dim, n_kv, q_rows, d_nope, inputLayout, schedule_mode)

    q_rows_total = b_dim * s_dim * n_q
    q_n_flat = q_nope.contiguous().view(q_rows_total, d_nope)
    q_r_flat = q_rope.contiguous().view(q_rows_total, d_rope)
    k_n_contig = k_nope.contiguous()
    k_r_contig = k_rope.contiguous()
    v_contig = v.contiguous()

    kernel = _get_kernel(
        n_q,
        n_kv,
        d_nope,
        d_rope,
        tl_dtype,
        inputLayout,
        bool(is_causal),
        bm,
        aligned_group_q,
        schedule_mode,
    )
    out_flat = kernel(q_n_flat, q_r_flat, k_n_contig, k_r_contig, v_contig, scale)
    if inputLayout == "BSND":
        return out_flat.view(b_dim, s_dim, n_q, d_nope)
    return out_flat.view(b_dim, n_q, s_dim, d_nope)


__all__ = ["mla"]
