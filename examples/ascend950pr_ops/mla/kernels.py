"""TileLang Ascend950PR MIX SimdVF kernel implementation for Multi-Head Latent Attention."""

import contextlib
import math

import tilelang
import tilelang.ascend.language as T
import torch

from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS

_KERNEL_CACHE: dict[tuple[object, ...], object] = {}
_DISPATCH_CACHE: dict[tuple[object, ...], dict[str, object]] = {}
_WORKSPACE_CACHE: dict[tuple[object, int, int], tuple[int, torch.Tensor, torch.Tensor]] = {}


def _get_split_kv_workspace(
    device: object,
    num_kv_splits: int,
    q_rows_total: int,
    d_nope: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    splits = max(1, int(num_kv_splits))
    q_rows = int(q_rows_total)
    d_val = int(d_nope)
    key = (device, q_rows, d_val)
    cached = _WORKSPACE_CACHE.get(key)
    if cached is not None:
        cap_splits, o_buf, lse_buf = cached
        if cap_splits >= splits:
            return o_buf[:splits], lse_buf[:splits]
    o_buf = torch.empty(
        (splits, q_rows, d_val),
        dtype=torch.float32,
        device=device,
    )
    lse_buf = torch.empty(
        (splits, q_rows),
        dtype=torch.float32,
        device=device,
    )
    if key not in _WORKSPACE_CACHE and len(_WORKSPACE_CACHE) >= 32:
        _WORKSPACE_CACHE.pop(next(iter(_WORKSPACE_CACHE)))
    _WORKSPACE_CACHE[key] = (splits, o_buf, lse_buf)
    return o_buf, lse_buf


def _callable_cache_key(fn: object) -> object:
    try:
        hash(fn)
        return fn
    except TypeError:
        return id(fn)


def _ascend_target_ctx():
    try:
        from tilelang import tvm
        from tilelang.backend.module import create_backend_context

        if tvm.target.Target.current(allow_none=True) is None:
            return create_backend_context("ascend").target
    except Exception:
        pass
    return contextlib.nullcontext()


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


def _build_mla_split_kv_decode_func(
    n_q: int = 128,
    n_kv: int = 1,
    d_nope: int = 512,
    d_rope: int = 64,
    tl_dtype: str = "float16",
    input_layout: str = "BSND",
    is_causal: bool = False,
    bm: int = 64,
    aligned_group_q: bool = True,
    num_kv_splits: int = 1,
    schedule_mode: str = "tuned",
    **_extra_kwargs,
):
    splits = max(1, int(num_kv_splits))
    if d_nope % 64 != 0 or d_rope % 64 != 0:
        raise ValueError(
            f"MLA requires d_nope ({d_nope}) and d_rope ({d_rope}) divisible by 64"
        )
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
        (b_dim, s_kv, n_kv, d_nope)
        if input_layout == "BSND"
        else (b_dim, n_kv, s_kv, d_nope)
    )
    k_rope_shape = (
        (b_dim, s_kv, n_kv, d_rope)
        if input_layout == "BSND"
        else (b_dim, n_kv, s_kv, d_rope)
    )

    if splits <= 1:
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
                o1_l0c = (
                    o_l0c
                    if (d1 == 0 or d1 == d0)
                    else T.alloc_l0c((bm, d1_alloc), "float32")
                )

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
                out_tile_rows = (
                    8 if (bm == 128 and rows_per_sid % 8 == 0) else rows_per_sid
                )
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

                        T.gemm(
                            q_nope_l1,
                            k_nope_l1,
                            s_l0c,
                            transpose_B=True,
                            clear_accum=True,
                        )
                        T.gemm(
                            q_rope_l1,
                            k_rope_l1,
                            s_l0c,
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.dual_copy(s_l0c, S_ub)

                        with T.SimdVF():
                            m32 = T.simd.pset(32, "PAT_ALL")
                            m16 = T.simd.pset(16, "PAT_ALL")
                            v_neg_inf = T.simd.vdup(
                                T.float32(-1e38), "float32", mask=m32
                            )
                            v_zero = T.simd.vdup(
                                T.float32(0.0), "float32", mask=m32
                            )
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
                                valid_mask = T.simd.vcmps(
                                    col_idx, limit_col, mask=m32, op="lt"
                                )
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
                                T.simd.vsts(
                                    m_ub[r, 0], m_new_v, mask=m32, dist="ONEPT_B32"
                                )
                                T.simd.vsts(
                                    l_ub[r, 0], l_new_v, mask=m32, dist="ONEPT_B32"
                                )
                                T.simd.vsts(
                                    alpha_ub[r, 0], alpha_v, mask=m32, dist="ONEPT_B32"
                                )
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
                                    o_tmp = T.simd.vld(
                                        O_tmp_ub[r, off], dist="NORM_B32"
                                    )
                                    o_new = T.simd.vadd(
                                        T.simd.vmul(o_old, alpha_v, mask=m32),
                                        o_tmp,
                                        mask=m32,
                                    )
                                    T.simd.vsts(
                                        O_ub[r, off], o_new, mask=m32, dist="NORM_B32"
                                    )

                        if d1 > 0:
                            T.gemm(
                                p_l1, v1_l1, o1_l0c, transpose_B=True, clear_accum=True
                            )
                            T.dual_copy(o1_l0c, O_tmp1_ub)
                            with T.SimdVF():
                                m32 = T.simd.pset(32, "PAT_ALL")
                                for r in T.serial(rows_per_sid):
                                    alpha_v = T.simd.vld(
                                        alpha_ub[r, 0], dist="BRC_B32"
                                    )
                                    for c in T.serial(d1_chunks):
                                        off = d0 + c * 64
                                        o_old = T.simd.vld(
                                            O_ub[r, off], dist="NORM_B32"
                                        )
                                        o_tmp = T.simd.vld(
                                            O_tmp1_ub[r, c * 64], dist="NORM_B32"
                                        )
                                        o_new = T.simd.vadd(
                                            T.simd.vmul(o_old, alpha_v, mask=m32),
                                            o_tmp,
                                            mask=m32,
                                        )
                                        T.simd.vsts(
                                            O_ub[r, off],
                                            o_new,
                                            mask=m32,
                                            dist="NORM_B32",
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
                                        T.simd.vdiv(v_one, l_safe, mask=m32),
                                        v_zero,
                                        l_pos,
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
                                    T.simd.vdiv(v_one, l_safe, mask=m32),
                                    v_zero,
                                    l_pos,
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
                            out_gm_base = (
                                group_id * q_rows + q_base + sid * rows_per_sid
                            )
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

    @T.prim_func
    def main(
        Q_nope: T.Tensor((q_total_rows, d_nope), tl_dtype),
        Q_rope: T.Tensor((q_total_rows, d_rope), tl_dtype),
        K_nope: T.Tensor(k_nope_shape, tl_dtype),
        K_rope: T.Tensor(k_rope_shape, tl_dtype),
        V: T.Tensor(k_nope_shape, tl_dtype),
        O_partial: T.Tensor((splits, q_total_rows, d_nope), "float32"),
        lse_partial: T.Tensor((splits, q_total_rows), "float32"),
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
            o1_l0c = (
                o_l0c
                if (d1 == 0 or d1 == d0)
                else T.alloc_l0c((bm, d1_alloc), "float32")
            )

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
            lse_out_ub = (
                T.alloc_shared((rows_per_sid,), "float32")
                if aligned_group_q
                else T.alloc_shared((rows_per_sid, 8), "float32")
            )
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
                lse_out_ub: 1,
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
            total_tasks = total_m_tiles * splits
            base_tasks = total_tasks // num_cores
            rem_tasks = total_tasks % num_cores

            t_start = bx * base_tasks + T.min(bx, rem_tasks)
            t_count = base_tasks + T.if_then_else(bx < rem_tasks, 1, 0)
            split_len = T.ceildiv(s_kv, splits)

            for local_t in T.serial(t_count):
                task_id = t_start + local_t
                split_idx = task_id % splits
                tile_id = task_id // splits
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

                kv_start = split_idx * split_len
                kv_end = T.min(s_kv, kv_start + split_len)
                if is_causal:
                    last_valid = T.min(q_rows - 1, q_base + bm - 1)
                    if input_layout == "BSND":
                        max_s = last_valid // g_ratio
                    else:
                        max_s = T.if_then_else(
                            (q_base // s_dim) == (last_valid // s_dim),
                            last_valid % s_dim,
                            s_dim - 1,
                        )
                    causal_max_kv = T.min(s_kv, max_s + s_kv - s_dim + 1)
                    eff_kv_end = T.min(kv_end, causal_max_kv)
                else:
                    eff_kv_end = kv_end

                if eff_kv_end > kv_start:
                    split_iters = T.ceildiv(eff_kv_end - kv_start, bn)
                    for kv_i in T.Pipelined(
                        split_iters, num_stages=num_stages, annotations=pipe_ann
                    ):
                        kv_pos = kv_start + kv_i * bn
                        if input_layout == "BSND":
                            T.copy(
                                K_nope[b_idx, kv_pos : kv_pos + bn, kv_h, 0:d_nope],
                                k_nope_l1,
                            )
                            T.copy(
                                K_rope[b_idx, kv_pos : kv_pos + bn, kv_h, 0:d_rope],
                                k_rope_l1,
                            )
                            T.copy(
                                V[b_idx, kv_pos : kv_pos + bn, kv_h, 0:d0],
                                v0_l1,
                                transpose=True,
                            )
                            if d1 > 0:
                                T.copy(
                                    V[b_idx, kv_pos : kv_pos + bn, kv_h, d0:d_nope],
                                    v1_l1,
                                    transpose=True,
                                )
                        else:
                            T.copy(
                                K_nope[b_idx, kv_h, kv_pos : kv_pos + bn, 0:d_nope],
                                k_nope_l1,
                            )
                            T.copy(
                                K_rope[b_idx, kv_h, kv_pos : kv_pos + bn, 0:d_rope],
                                k_rope_l1,
                            )
                            T.copy(
                                V[b_idx, kv_h, kv_pos : kv_pos + bn, 0:d0],
                                v0_l1,
                                transpose=True,
                            )
                            if d1 > 0:
                                T.copy(
                                    V[b_idx, kv_h, kv_pos : kv_pos + bn, d0:d_nope],
                                    v1_l1,
                                    transpose=True,
                                )

                        T.gemm(
                            q_nope_l1,
                            k_nope_l1,
                            s_l0c,
                            transpose_B=True,
                            clear_accum=True,
                        )
                        T.gemm(
                            q_rope_l1,
                            k_rope_l1,
                            s_l0c,
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.dual_copy(s_l0c, S_ub)

                        with T.SimdVF():
                            m32 = T.simd.pset(32, "PAT_ALL")
                            m16 = T.simd.pset(16, "PAT_ALL")
                            v_neg_inf = T.simd.vdup(
                                T.float32(-1e38), "float32", mask=m32
                            )
                            v_zero = T.simd.vdup(
                                T.float32(0.0), "float32", mask=m32
                            )
                            col_idx = T.simd.vci(kv_pos, "int32")
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
                                limit_col = T.if_then_else(
                                    q_row < q_rows,
                                    T.min(eff_kv_end, raw_limit),
                                    0,
                                )
                                valid_mask = T.simd.vcmps(
                                    col_idx, limit_col, mask=m32, op="lt"
                                )
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
                                T.simd.vsts(
                                    m_ub[r, 0], m_new_v, mask=m32, dist="ONEPT_B32"
                                )
                                T.simd.vsts(
                                    l_ub[r, 0], l_new_v, mask=m32, dist="ONEPT_B32"
                                )
                                T.simd.vsts(
                                    alpha_ub[r, 0], alpha_v, mask=m32, dist="ONEPT_B32"
                                )
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
                                    o_tmp = T.simd.vld(
                                        O_tmp_ub[r, off], dist="NORM_B32"
                                    )
                                    o_new = T.simd.vadd(
                                        T.simd.vmul(o_old, alpha_v, mask=m32),
                                        o_tmp,
                                        mask=m32,
                                    )
                                    T.simd.vsts(
                                        O_ub[r, off], o_new, mask=m32, dist="NORM_B32"
                                    )

                        if d1 > 0:
                            T.gemm(
                                p_l1, v1_l1, o1_l0c, transpose_B=True, clear_accum=True
                            )
                            T.dual_copy(o1_l0c, O_tmp1_ub)
                            with T.SimdVF():
                                m32 = T.simd.pset(32, "PAT_ALL")
                                for r in T.serial(rows_per_sid):
                                    alpha_v = T.simd.vld(
                                        alpha_ub[r, 0], dist="BRC_B32"
                                    )
                                    for c in T.serial(d1_chunks):
                                        off = d0 + c * 64
                                        o_old = T.simd.vld(
                                            O_ub[r, off], dist="NORM_B32"
                                        )
                                        o_tmp = T.simd.vld(
                                            O_tmp1_ub[r, c * 64], dist="NORM_B32"
                                        )
                                        o_new = T.simd.vadd(
                                            T.simd.vmul(o_old, alpha_v, mask=m32),
                                            o_tmp,
                                            mask=m32,
                                        )
                                        T.simd.vsts(
                                            O_ub[r, off],
                                            o_new,
                                            mask=m32,
                                            dist="NORM_B32",
                                        )

                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
                    v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                    v_empty_lse = T.simd.vdup(T.float32(-1e30), "float32", mask=m32)
                    for r in T.serial(rows_per_sid):
                        m_vec = T.simd.vld(m_ub[r, 0], dist="BRC_B32")
                        l_vec = T.simd.vld(l_ub[r, 0], dist="BRC_B32")
                        l_pos = T.simd.vcmps(l_vec, T.float32(0.0), mask=m32, op="gt")
                        l_safe = T.simd.vsel(l_vec, v_one, l_pos)
                        inv_l = T.simd.vsel(
                            T.simd.vdiv(v_one, l_safe, mask=m32), v_zero, l_pos
                        )
                        lse_raw = T.simd.vadd(
                            m_vec, T.simd.vln(l_safe, mask=m32), mask=m32
                        )
                        lse_val = T.simd.vsel(lse_raw, v_empty_lse, l_pos)
                        if aligned_group_q:
                            T.simd.vsts(
                                lse_out_ub[r], lse_val, mask=m32, dist="ONEPT_B32"
                            )
                        else:
                            T.simd.vsts(
                                lse_out_ub[r, 0], lse_val, mask=m32, dist="ONEPT_B32"
                            )
                        for c in T.serial(d_nope_chunks):
                            off = c * 64
                            o_f = T.simd.vmul(
                                T.simd.vld(O_ub[r, off], dist="NORM_B32"),
                                inv_l,
                                mask=m32,
                            )
                            T.simd.vsts(O_ub[r, off], o_f, mask=m32, dist="NORM_B32")

                if aligned_group_q:
                    out_gm_base = group_id * q_rows + q_base + sid * rows_per_sid
                    T.copy(
                        O_ub,
                        O_partial[
                            split_idx, out_gm_base : out_gm_base + rows_per_sid, 0:d_nope
                        ],
                    )
                    T.copy(
                        lse_out_ub,
                        lse_partial[
                            split_idx, out_gm_base : out_gm_base + rows_per_sid
                        ],
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
                            T.copy(
                                O_ub[r, 0:d_nope],
                                O_partial[split_idx, out_r, 0:d_nope],
                            )
                            T.copy(
                                lse_out_ub[r, 0:1],
                                lse_partial[split_idx, out_r : out_r + 1],
                            )

    return main


def _build_mla_split_kv_combine_func(
    d_nope: int = 512,
    tl_dtype: str = "float16",
    num_kv_splits: int = 4,
    **_extra_kwargs,
):
    splits = max(1, int(num_kv_splits))
    d_chunks = d_nope // 64
    num_cores = 56
    q_total_rows = T.dynamic("q_total_rows")

    @T.prim_func
    def main(
        O_partial: T.Tensor((splits, q_total_rows, d_nope), "float32"),
        lse_partial: T.Tensor((splits, q_total_rows), "float32"),
        Out: T.Tensor((q_total_rows, d_nope), tl_dtype),
    ):
        with T.Kernel(num_cores) as cid:
            lse_ub = T.alloc_shared((splits, 8), "float32")
            w_ub = T.alloc_shared((splits, 8), "float32")
            o_ub = T.alloc_shared((d_nope,), "float32")
            acc_ub = T.alloc_shared((d_nope,), "float32")
            out_ub = T.alloc_shared((d_nope,), tl_dtype)
            inv_sum_ub = T.alloc_shared((8,), "float32")
            T.annotate_buffer_versions(
                {
                    lse_ub: 1,
                    w_ub: 1,
                    o_ub: 1,
                    acc_ub: 1,
                    out_ub: 1,
                    inv_sum_ub: 1,
                }
            )

            base_q = q_total_rows // num_cores
            rem_q = q_total_rows % num_cores
            q_start = cid * base_q + T.min(cid, rem_q)
            q_count = base_q + T.if_then_else(cid < rem_q, 1, 0)

            for lq in T.serial(q_count):
                qr = q_start + lq
                for s in T.serial(splits):
                    T.copy(lse_partial[s, qr : qr + 1], lse_ub[s, 0:1])
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                    v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
                    m_glob = T.simd.alloc_local((1,), "float32")
                    sum_w = T.simd.alloc_local((1,), "float32")
                    m_glob[0] = T.simd.vdup(T.float32(-1e30), "float32", mask=m32)
                    sum_w[0] = v_zero
                    for s in T.serial(splits):
                        lse_s = T.simd.vld(lse_ub[s, 0], dist="BRC_B32")
                        m_glob[0] = T.simd.vmax(m_glob[0], lse_s, mask=m32)
                    for s in T.serial(splits):
                        lse_s = T.simd.vld(lse_ub[s, 0], dist="BRC_B32")
                        valid_s = T.simd.vcmps(
                            lse_s, T.float32(-1e29), mask=m32, op="gt"
                        )
                        w_raw = T.simd.vexp(
                            T.simd.vsub(lse_s, m_glob[0], mask=m32), mask=m32
                        )
                        w_s = T.simd.vsel(w_raw, v_zero, valid_s)
                        sum_w[0] = T.simd.vadd(sum_w[0], w_s, mask=m32)
                        T.simd.vsts(w_ub[s, 0], w_s, mask=m32, dist="ONEPT_B32")
                    sum_pos = T.simd.vcmps(
                        sum_w[0], T.float32(1e-30), mask=m32, op="gt"
                    )
                    sum_safe = T.simd.vsel(sum_w[0], v_one, sum_pos)
                    inv_sum = T.simd.vsel(
                        T.simd.vdiv(v_one, sum_safe, mask=m32), v_zero, sum_pos
                    )
                    T.simd.vsts(inv_sum_ub[0], inv_sum, mask=m32, dist="ONEPT_B32")
                    for c in T.serial(d_chunks):
                        T.simd.vsts(acc_ub[c * 64], v_zero, mask=m32, dist="NORM_B32")

                for s in T.serial(splits):
                    T.copy(O_partial[s, qr, 0:d_nope], o_ub)
                    with T.SimdVF():
                        m32 = T.simd.pset(32, "PAT_ALL")
                        w_s = T.simd.vld(w_ub[s, 0], dist="BRC_B32")
                        for c in T.serial(d_chunks):
                            off = c * 64
                            o_v = T.simd.vld(o_ub[off], dist="NORM_B32")
                            acc_v = T.simd.vld(acc_ub[off], dist="NORM_B32")
                            acc_new = T.simd.vadd(
                                acc_v,
                                T.simd.vmul(o_v, w_s, mask=m32),
                                mask=m32,
                            )
                            T.simd.vsts(
                                acc_ub[off], acc_new, mask=m32, dist="NORM_B32"
                            )

                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    m16 = T.simd.pset(16, "PAT_ALL")
                    inv_s = T.simd.vld(inv_sum_ub[0], dist="BRC_B32")
                    for c in T.serial(d_chunks):
                        off = c * 64
                        acc_v = T.simd.vld(acc_ub[off], dist="NORM_B32")
                        y_f = T.simd.vmul(acc_v, inv_s, mask=m32)
                        y_16 = T.simd.vcvt(y_f, tl_dtype, mask=m16, part=0)
                        T.simd.vsts(
                            out_ub[off], y_16, mask=m32, dist="PK_B32", extent=64
                        )
                T.copy(out_ub, Out[qr, 0:d_nope])

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


def _mla_default_flags(schedule_mode: str) -> list[str]:
    return list(PURE_SIMDVF_UBUF_FLAGS) if schedule_mode == "tuned" else []


def _select_mla_decode_config(
    b_dim: int = 1,
    s_dim: int = 1,
    s_kv: int = 2048,
    n_q: int = 128,
    n_kv: int = 1,
    d_nope: int = 512,
    d_rope: int = 64,
    tl_dtype: str = "float16",
    input_layout: str = "BSND",
    is_causal: bool = False,
    schedule_mode: str = "tuned",
) -> dict[str, object]:
    _ = (d_rope, tl_dtype, is_causal)
    q_rows = s_dim * (n_q // n_kv)
    bm, aligned_group_q = _choose_mla_bm(
        b_dim, n_kv, q_rows, d_nope, input_layout, schedule_mode
    )
    splits = 1
    if schedule_mode == "tuned" and s_kv >= 1024 and q_rows >= 64 and (q_rows % 64) == 0:
        m_tiles = b_dim * n_kv * ((q_rows + bm - 1) // bm)
        if m_tiles < 14:
            if q_rows >= 64:
                bm = 64
                aligned_group_q = input_layout == "BNSD" or n_kv == 1
                m_tiles = b_dim * n_kv * ((q_rows + 63) // 64)
            kv_blocks = max(1, s_kv // 64)
            target_splits = (28 + m_tiles - 1) // m_tiles
            splits = max(1, min(14, min(kv_blocks, target_splits)))
    return {
        "bm": bm,
        "aligned_group_q": aligned_group_q,
        "num_kv_splits": splits,
        "pass_configs": {},
        "compile_flags": _mla_default_flags(schedule_mode),
    }


def _select_mla_bnsd_config(
    b_dim: int = 2,
    s_dim: int = 128,
    s_kv: int = 128,
    n_q: int = 128,
    n_kv: int = 1,
    d_nope: int = 512,
    d_rope: int = 64,
    tl_dtype: str = "float16",
    input_layout: str = "BNSD",
    is_causal: bool = True,
    schedule_mode: str = "tuned",
) -> dict[str, object]:
    _ = (s_kv, d_rope, tl_dtype, is_causal)
    q_rows = s_dim * (n_q // n_kv)
    bm, aligned_group_q = _choose_mla_bm(
        b_dim, n_kv, q_rows, d_nope, input_layout, schedule_mode
    )
    return {
        "bm": bm,
        "aligned_group_q": aligned_group_q,
        "num_kv_splits": 1,
        "pass_configs": {},
        "compile_flags": _mla_default_flags(schedule_mode),
    }


def _build_mla_bnsd_gather_func(
    n_q: int = 128,
    n_kv: int = 1,
    d_nope: int = 512,
    d_rope: int = 64,
    tl_dtype: str = "float16",
    input_layout: str = "BNSD",
    is_causal: bool = True,
    bm: int = 64,
    aligned_group_q: bool = False,
    schedule_mode: str = "tuned",
    **_extra_kwargs,
):
    if input_layout != "BNSD":
        raise ValueError(
            f"_build_mla_bnsd_gather_func requires input_layout='BNSD', got {input_layout!r}"
        )
    if d_nope % 64 != 0 or d_rope % 64 != 0:
        raise ValueError(
            f"MLA requires d_nope ({d_nope}) and d_rope ({d_rope}) divisible by 64"
        )
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

    k_nope_shape = (b_dim, n_kv, s_kv, d_nope)
    k_rope_shape = (b_dim, n_kv, s_kv, d_rope)

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
            o1_l0c = (
                o_l0c
                if (d1 == 0 or d1 == d0)
                else T.alloc_l0c((bm, d1_alloc), "float32")
            )

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
            out_tile_rows = (
                8 if (bm == 128 and rows_per_sid % 8 == 0) else rows_per_sid
            )
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

                    T.gemm(
                        q_nope_l1,
                        k_nope_l1,
                        s_l0c,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.gemm(
                        q_rope_l1,
                        k_rope_l1,
                        s_l0c,
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.dual_copy(s_l0c, S_ub)

                    with T.SimdVF():
                        m32 = T.simd.pset(32, "PAT_ALL")
                        m16 = T.simd.pset(16, "PAT_ALL")
                        v_neg_inf = T.simd.vdup(T.float32(-1e38), "float32", mask=m32)
                        v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                        col_idx = T.simd.vci(kv * bn, "int32")
                        for r in T.serial(rows_per_sid):
                            q_row = q_base + sid * rows_per_sid + r
                            s_idx = q_row % s_dim
                            raw_limit = (
                                T.min(s_kv, s_idx + s_kv - s_dim + 1)
                                if is_causal
                                else s_kv
                            )
                            limit_col = T.if_then_else(q_row < q_rows, raw_limit, 0)
                            valid_mask = T.simd.vcmps(
                                col_idx, limit_col, mask=m32, op="lt"
                            )
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
                            T.simd.vsts(
                                alpha_ub[r, 0], alpha_v, mask=m32, dist="ONEPT_B32"
                            )
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
                                out_r = group_id * q_rows + q_row
                                T.copy(out_ub[r, 0:d_nope], Out[out_r, 0:d_nope])

    return main


def _select_mla_prefill_config(
    b_dim: int = 2,
    s_dim: int = 128,
    s_kv: int = 128,
    n_q: int = 128,
    n_kv: int = 1,
    d_nope: int = 512,
    d_rope: int = 64,
    tl_dtype: str = "float16",
    input_layout: str = "BSND",
    is_causal: bool = True,
    schedule_mode: str = "tuned",
) -> dict[str, object]:
    _ = (s_kv, d_rope, tl_dtype, is_causal)
    q_rows = s_dim * (n_q // n_kv)
    bm, aligned_group_q = _choose_mla_bm(
        b_dim, n_kv, q_rows, d_nope, input_layout, schedule_mode
    )
    return {
        "bm": bm,
        "aligned_group_q": aligned_group_q,
        "num_kv_splits": 1,
        "pass_configs": {},
        "compile_flags": _mla_default_flags(schedule_mode),
    }


def _build_mla_prefill_func(
    n_q: int = 128,
    n_kv: int = 1,
    d_nope: int = 512,
    d_rope: int = 64,
    tl_dtype: str = "float16",
    input_layout: str = "BSND",
    is_causal: bool = True,
    bm: int = 128,
    aligned_group_q: bool = True,
    schedule_mode: str = "tuned",
    **_extra_kwargs,
):
    if input_layout != "BSND":
        raise ValueError(
            f"_build_mla_prefill_func requires input_layout='BSND', got {input_layout!r}"
        )
    if d_nope % 64 != 0 or d_rope % 64 != 0:
        raise ValueError(
            f"MLA requires d_nope ({d_nope}) and d_rope ({d_rope}) divisible by 64"
        )
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

    k_nope_shape = (b_dim, s_kv, n_kv, d_nope)
    k_rope_shape = (b_dim, s_kv, n_kv, d_rope)

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
            o1_l0c = (
                o_l0c
                if (d1 == 0 or d1 == d0)
                else T.alloc_l0c((bm, d1_alloc), "float32")
            )

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
            out_tile_rows = (
                8 if (bm == 128 and rows_per_sid % 8 == 0) else rows_per_sid
            )
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
                            s_i = q_row // g_ratio
                            h_loc = q_row % g_ratio
                            q_h = kv_h * g_ratio + h_loc
                            gm_r = (b_idx * s_dim + s_i) * n_q + q_h
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
                    max_s = last_valid // g_ratio
                    kv_limit = T.max(
                        1,
                        T.min(max_kv_blocks, (max_s + s_kv - s_dim) // bn + 1),
                    )
                else:
                    kv_limit = max_kv_blocks

                for kv in T.Pipelined(
                    kv_limit, num_stages=num_stages, annotations=pipe_ann
                ):
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

                    T.gemm(
                        q_nope_l1,
                        k_nope_l1,
                        s_l0c,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.gemm(
                        q_rope_l1,
                        k_rope_l1,
                        s_l0c,
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.dual_copy(s_l0c, S_ub)

                    with T.SimdVF():
                        m32 = T.simd.pset(32, "PAT_ALL")
                        m16 = T.simd.pset(16, "PAT_ALL")
                        v_neg_inf = T.simd.vdup(T.float32(-1e38), "float32", mask=m32)
                        v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                        col_idx = T.simd.vci(kv * bn, "int32")
                        for r in T.serial(rows_per_sid):
                            q_row = q_base + sid * rows_per_sid + r
                            s_idx = q_row // g_ratio
                            raw_limit = (
                                T.min(s_kv, s_idx + s_kv - s_dim + 1)
                                if is_causal
                                else s_kv
                            )
                            limit_col = T.if_then_else(q_row < q_rows, raw_limit, 0)
                            valid_mask = T.simd.vcmps(
                                col_idx, limit_col, mask=m32, op="lt"
                            )
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
                            T.simd.vsts(
                                alpha_ub[r, 0], alpha_v, mask=m32, dist="ONEPT_B32"
                            )
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
                                s_i = q_row // g_ratio
                                h_loc = q_row % g_ratio
                                q_h = kv_h * g_ratio + h_loc
                                out_r = (b_idx * s_dim + s_i) * n_q + q_h
                                T.copy(out_ub[r, 0:d_nope], Out[out_r, 0:d_nope])

    return main


def _freeze_attr_value(val: object) -> object:
    if isinstance(val, dict):
        return tuple((str(k), _freeze_attr_value(v)) for k, v in sorted(val.items()))
    if isinstance(val, (list, tuple)):
        return tuple(_freeze_attr_value(x) for x in val)
    return val


def _normalize_pass_configs_key(
    pass_configs: dict[str, object] | None,
) -> tuple[tuple[str, object], ...]:
    if not pass_configs:
        return ()
    return tuple(
        (str(k), _freeze_attr_value(v))
        for k, v in sorted(pass_configs.items(), key=lambda item: str(item[0]))
    )


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
    *,
    builder_fn=None,
    num_kv_splits: int = 1,
    pass_configs: dict[str, object] | None = None,
    compile_flags: list[str] | tuple[str, ...] | None = None,
    out_idx: list[int] | None = None,
    extra_builder_kwargs: dict[str, object] | None = None,
):
    resolved_builder = builder_fn or _build_mla_prim_func
    splits = max(1, int(num_kv_splits))
    pc_dict = dict(pass_configs or {})
    extra_bk = dict(extra_builder_kwargs or {})
    if compile_flags is None:
        flags_tuple = tuple(PURE_SIMDVF_UBUF_FLAGS) if schedule_mode == "tuned" else ()
    else:
        flags_tuple = tuple(str(f) for f in compile_flags)
    resolved_out_idx = out_idx if splits <= 1 else None
    if splits <= 1 and resolved_out_idx is None:
        resolved_out_idx = [5]
    key = (
        resolved_builder.__name__,
        n_q,
        n_kv,
        d_nope,
        d_rope,
        tl_dtype,
        input_layout,
        bool(is_causal),
        int(bm),
        bool(aligned_group_q),
        splits,
        schedule_mode,
        _normalize_pass_configs_key(pc_dict),
        flags_tuple,
        _normalize_pass_configs_key(extra_bk),
    )
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        with _ascend_target_ctx():
            if resolved_builder is _build_mla_split_kv_decode_func:
                prog = resolved_builder(
                    n_q,
                    n_kv,
                    d_nope,
                    d_rope,
                    tl_dtype,
                    input_layout,
                    is_causal,
                    bm,
                    aligned_group_q,
                    splits,
                    schedule_mode,
                    **extra_bk,
                )
            else:
                prog = resolved_builder(
                    n_q,
                    n_kv,
                    d_nope,
                    d_rope,
                    tl_dtype,
                    input_layout,
                    is_causal,
                    bm,
                    aligned_group_q,
                    schedule_mode,
                    **extra_bk,
                )
        flags_arg = list(flags_tuple) if flags_tuple else None
        compiled = tilelang.compile(
            prog,
            target="ascend",
            out_idx=resolved_out_idx,
            pass_configs=pc_dict or None,
            compile_flags=flags_arg,
        )
        _KERNEL_CACHE[key] = compiled
    return compiled


def _get_combine_kernel(
    d_nope: int,
    tl_dtype: str,
    num_kv_splits: int,
    *,
    pass_configs: dict[str, object] | None = None,
    compile_flags: list[str] | tuple[str, ...] | None = None,
    extra_combine_kwargs: dict[str, object] | None = None,
):
    splits = max(1, int(num_kv_splits))
    pc_dict = dict(pass_configs or {})
    extra_ck = dict(extra_combine_kwargs or {})
    flags_tuple = tuple(str(f) for f in (compile_flags or ()))
    key = (
        "_build_mla_split_kv_combine_func",
        int(d_nope),
        tl_dtype,
        splits,
        _normalize_pass_configs_key(pc_dict),
        flags_tuple,
        _normalize_pass_configs_key(extra_ck),
    )
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        with _ascend_target_ctx():
            prog = _build_mla_split_kv_combine_func(
                d_nope=d_nope,
                tl_dtype=tl_dtype,
                num_kv_splits=splits,
                **extra_ck,
            )
        flags_arg = list(flags_tuple) if flags_tuple else None
        compiled = tilelang.compile(
            prog,
            target="ascend",
            out_idx=[2],
            pass_configs=pc_dict or None,
            compile_flags=flags_arg,
        )
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
    num_kv_splits_override: int | None = None,
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

    if (
        schedule_mode == "tuned"
        and inputLayout == "BNSD"
        and n_kv == 1
        and s_dim > 1
        and (b_dim * n_q * s_dim) <= 32768
    ):
        q_nope_bsnd = q_nope.transpose(1, 2).contiguous()
        q_rope_bsnd = q_rope.transpose(1, 2).contiguous()
        k_nope_bsnd = k_nope.contiguous().view(b_dim, s_kv, 1, d_nope)
        k_rope_bsnd = k_rope.contiguous().view(b_dim, s_kv, 1, d_rope)
        v_bsnd = v.contiguous().view(b_dim, s_kv, 1, d_nope)
        out_bsnd = mla(
            q_nope_bsnd,
            q_rope_bsnd,
            k_nope_bsnd,
            k_rope_bsnd,
            v_bsnd,
            numKVHeads=numKVHeads,
            scaleValue=scaleValue,
            inputLayout="BSND",
            is_causal=is_causal,
            schedule_mode=schedule_mode,
            num_kv_splits_override=num_kv_splits_override,
        )
        return out_bsnd.transpose(1, 2).contiguous()

    if inputLayout == "BNSD":
        selector_fn, builder_fn = _select_mla_bnsd_config, _build_mla_bnsd_gather_func
    elif s_dim <= 2:
        selector_fn, builder_fn = _select_mla_decode_config, _build_mla_split_kv_decode_func
    else:
        selector_fn, builder_fn = _select_mla_prefill_config, _build_mla_prefill_func

    dispatch_key = (
        _callable_cache_key(selector_fn),
        b_dim,
        s_dim,
        s_kv,
        n_q,
        n_kv,
        d_nope,
        d_rope,
        tl_dtype,
        inputLayout,
        bool(is_causal),
        schedule_mode,
    )
    cfg = _DISPATCH_CACHE.get(dispatch_key)
    if cfg is None:
        cfg = dict(
            selector_fn(
                b_dim=b_dim,
                s_dim=s_dim,
                s_kv=s_kv,
                n_q=n_q,
                n_kv=n_kv,
                d_nope=d_nope,
                d_rope=d_rope,
                tl_dtype=tl_dtype,
                input_layout=inputLayout,
                is_causal=bool(is_causal),
                schedule_mode=schedule_mode,
            )
        )
        _DISPATCH_CACHE[dispatch_key] = cfg
    bm = int(cfg["bm"])
    aligned_group_q = bool(cfg["aligned_group_q"])
    num_kv_splits = (
        max(1, int(num_kv_splits_override))
        if num_kv_splits_override is not None
        else max(1, int(cfg.get("num_kv_splits", 1)))
    )
    pass_configs = dict(cfg.get("pass_configs") or {})
    compile_flags = tuple(str(f) for f in (cfg.get("compile_flags") or ()))
    raw_bk = cfg.get("builder_kwargs")
    extra_builder_kwargs = dict(raw_bk) if isinstance(raw_bk, dict) else None
    raw_ck = cfg.get("combine_kwargs")
    extra_combine_kwargs = dict(raw_ck) if isinstance(raw_ck, dict) else None

    q_rows_total = b_dim * s_dim * n_q
    q_n_flat = q_nope.contiguous().view(q_rows_total, d_nope)
    q_r_flat = q_rope.contiguous().view(q_rows_total, d_rope)
    k_n_contig = k_nope.contiguous()
    k_r_contig = k_rope.contiguous()
    v_contig = v.contiguous()

    if num_kv_splits <= 1:
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
            builder_fn=builder_fn,
            num_kv_splits=1,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
            extra_builder_kwargs=extra_builder_kwargs,
        )
        out_flat = kernel(q_n_flat, q_r_flat, k_n_contig, k_r_contig, v_contig, scale)
    else:
        o_partial, lse_partial = _get_split_kv_workspace(
            q_nope.device,
            num_kv_splits,
            q_rows_total,
            d_nope,
        )
        stage1_kernel = _get_kernel(
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
            builder_fn=_build_mla_split_kv_decode_func,
            num_kv_splits=num_kv_splits,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
            extra_builder_kwargs=extra_builder_kwargs,
        )
        stage1_kernel(
            q_n_flat,
            q_r_flat,
            k_n_contig,
            k_r_contig,
            v_contig,
            o_partial,
            lse_partial,
            scale,
        )
        combine_kernel = _get_combine_kernel(
            d_nope,
            tl_dtype,
            num_kv_splits,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
            extra_combine_kwargs=extra_combine_kwargs,
        )
        out_flat = combine_kernel(o_partial, lse_partial)

    if inputLayout == "BSND":
        return out_flat.view(b_dim, s_dim, n_q, d_nope)
    return out_flat.view(b_dim, n_q, s_dim, d_nope)


__all__ = ["mla"]
