"""TileLang Ascend950PR MIX SimdVF kernel implementation for Multi-Head Latent Attention."""

import contextlib
import ctypes
import hashlib
import math
import os
from pathlib import Path
import tempfile
import uuid

import tilelang
import tilelang.ascend.language as T
from tilelang.contrib.bisheng import compile_ascend
import torch

from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS

_KERNEL_CACHE: dict[tuple[object, ...], object] = {}
_TRACKB_SO_CACHE: dict[str, object] = {}
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


def _compute_mla_l1_half_bank_plan(
    *,
    bm: int,
    bn: int,
    d_nope: int,
    d_rope: int,
    d0: int,
    d1: int,
    d1_alloc: int,
    num_stages: int,
    schedule_mode: str,
) -> tuple[bool, int, int, int, int, int, int, int, int, int, int, int]:
    """Compute Half-Bank L1 (2x256 KiB) addresses isolating Q/P from K/V."""
    if schedule_mode != "tuned":
        return (False, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    l1_half = 262144
    elem_bytes = 2
    q_nope_bytes = bm * d_nope * elem_bytes
    q_rope_bytes = bm * d_rope * elem_bytes
    p_bytes = bm * bn * elem_bytes
    bank0_end = q_nope_bytes + q_rope_bytes + p_bytes

    k_nope_slot = bn * d_nope * elem_bytes
    k_rope_slot = bn * d_rope * elem_bytes
    v0_slot = d0 * bn * elem_bytes
    v1_slot = d1_alloc * bn * elem_bytes if d1 > 0 else 0

    candidates = (
        ((2, 2, 2, 1), (2, 2, 1, 1), (1, 1, 1, 1))
        if num_stages >= 2
        else ((1, 1, 1, 1),)
    )
    for kn_v, kr_v, v0_v, v1_v in candidates:
        bank1_need = (
            kn_v * k_nope_slot
            + kr_v * k_rope_slot
            + v0_v * v0_slot
            + (v1_v * v1_slot if d1 > 0 else 0)
        )
        if bank0_end <= l1_half and bank1_need <= l1_half:
            addr_qn = 0
            addr_qr = q_nope_bytes
            addr_p = q_nope_bytes + q_rope_bytes
            addr_kn = l1_half
            addr_kr = addr_kn + kn_v * k_nope_slot
            addr_v0 = addr_kr + kr_v * k_rope_slot
            addr_v1 = addr_v0 + v0_v * v0_slot
            return (
                True,
                kn_v,
                kr_v,
                v0_v,
                v1_v,
                addr_qn,
                addr_qr,
                addr_p,
                addr_kn,
                addr_kr,
                addr_v0,
                addr_v1,
            )
    return (False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


def _resolve_mla_tile_params(
    bm: int,
    bn: int,
    num_stages: int,
    schedule_mode: str,
) -> tuple[int, int]:
    if bn != 64:
        raise ValueError(f"MLA unsupported bn={bn}")
    if int(num_stages) > 0:
        eff_stages = 2 if (int(num_stages) >= 2 and schedule_mode == "tuned") else 1
    else:
        eff_stages = 2 if schedule_mode == "tuned" else 1
    return bn, eff_stages


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
    bn: int = 64,
    num_stages: int = 0,
    **_extra_kwargs,
):
    if d_nope % 64 != 0 or d_rope % 64 != 0:
        raise ValueError(f"MLA requires d_nope ({d_nope}) and d_rope ({d_rope}) divisible by 64")
    if bm not in (32, 64, 128):
        raise ValueError(f"MLA unsupported bm={bm}")
    rows_per_sid = bm // 2
    bn, num_stages = _resolve_mla_tile_params(bm, int(bn), int(num_stages), schedule_mode)
    d0 = min(d_nope, 256)
    d1 = d_nope - d0
    d1_alloc = max(d1, 64)
    g_ratio = n_q // n_kv
    d_nope_chunks = d_nope // 64
    d0_chunks = d0 // 64
    d1_chunks = d1 // 64
    d_rope_chunks = d_rope // 64
    num_cores = 28

    pipe_ann = {"enable_offset": True} if (num_stages >= 2 and schedule_mode == "tuned") else None

    b_dim = T.dynamic("b_dim")
    q_rows_sym = T.dynamic("q_rows")
    s_dim_sym = T.dynamic("s_dim")
    s_kv = T.dynamic("s_kv")

    is_struct_4d = input_layout == "BNSD" or n_kv == 1
    eff_aligned_group_q = bool(aligned_group_q and is_struct_4d)
    q_nope_shape = (
        (b_dim, n_kv, q_rows_sym, d_nope)
        if is_struct_4d
        else (b_dim, s_dim_sym, n_kv, g_ratio, d_nope)
    )
    q_rope_shape = (
        (b_dim, n_kv, q_rows_sym, d_rope)
        if is_struct_4d
        else (b_dim, s_dim_sym, n_kv, g_ratio, d_rope)
    )
    k_nope_shape = (
        (b_dim, s_kv, n_kv, d_nope) if input_layout == "BSND" else (b_dim, n_kv, s_kv, d_nope)
    )
    k_rope_shape = (
        (b_dim, s_kv, n_kv, d_rope) if input_layout == "BSND" else (b_dim, n_kv, s_kv, d_rope)
    )

    @T.macro
    def _kv_step(
        kv, K_nope, K_rope, V,
        scale_val, q_nope_l1, q_rope_l1, k_nope_l1,
        k_rope_l1, p_l1, v0_l1, v1_l1,
        s_l0c, o_l0c, o1_l0c, S_ub,
        P_ub, O_tmp_ub, O_tmp1_ub, O_ub,
        m_ub, l_ub, alpha_ub, sid,
        b_idx, kv_h, q_base, q_rows,
        s_dim,
    ):
        """Execute one KV tile step inside the MLA loop."""
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

    (
        use_l1_iso, kn_v, kr_v, v0_v, v1_v,
        addr_qn, addr_qr, addr_p, addr_kn, addr_kr, addr_v0, addr_v1,
    ) = _compute_mla_l1_half_bank_plan(
        bm=bm, bn=bn, d_nope=d_nope, d_rope=d_rope,
        d0=d0, d1=d1, d1_alloc=d1_alloc, num_stages=num_stages,
        schedule_mode=schedule_mode,
    )
    func_attrs = {"half_l1": True, "l1_half_bank_split": True} if use_l1_iso else {}

    @T.prim_func
    def main(
        Q_nope: T.Tensor(q_nope_shape, tl_dtype),
        Q_rope: T.Tensor(q_rope_shape, tl_dtype),
        K_nope: T.Tensor(k_nope_shape, tl_dtype),
        K_rope: T.Tensor(k_rope_shape, tl_dtype),
        V: T.Tensor(k_nope_shape, tl_dtype),
        Out: T.Tensor(q_nope_shape, tl_dtype),
        scale_val: T.float32,
    ):
        T.func_attr(func_attrs)
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

            if not eff_aligned_group_q:
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
            out_tile_rows = 8 if (eff_aligned_group_q and bm == 128 and rows_per_sid % 8 == 0) else rows_per_sid
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
            if not eff_aligned_group_q:
                pinned[q_nope_ub] = 1
                pinned[q_rope_ub] = 1
            if d1 > 0 and d1 != d0:
                pinned[o1_l0c] = 1
                pinned[O_tmp1_ub] = 1
            if use_l1_iso:
                pinned.update({k_nope_l1: kn_v, k_rope_l1: kr_v, v0_l1: v0_v})
                if d1 > 0:
                    pinned[v1_l1] = v1_v
            elif schedule_mode != "tuned":
                pinned.update(
                    {
                        k_nope_l1: 1,
                        k_rope_l1: 1,
                        v0_l1: 1,
                        v1_l1: 1,
                    }
                )
            T.annotate_buffer_versions(pinned)
            if use_l1_iso:
                l1_addrs = {
                    q_nope_l1: addr_qn,
                    q_rope_l1: addr_qr,
                    p_l1: addr_p,
                    k_nope_l1: addr_kn,
                    k_rope_l1: addr_kr,
                    v0_l1: addr_v0,
                }
                if d1 > 0:
                    l1_addrs[v1_l1] = addr_v1
                T.annotate_buffer_addresses(l1_addrs)
            if bm == 128:
                T.annotate_unlimit_memory("shared")

            if is_struct_4d:
                q_rows = q_rows_sym
                s_dim = q_rows_sym // g_ratio
            else:
                s_dim = s_dim_sym
                q_rows = s_dim_sym * g_ratio
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

                if eff_aligned_group_q:
                    T.copy(
                        Q_nope[b_idx, kv_h, q_base : q_base + bm, 0:d_nope],
                        q_nope_l1,
                    )
                    T.copy(
                        Q_rope[b_idx, kv_h, q_base : q_base + bm, 0:d_rope],
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
                            if is_struct_4d:
                                T.copy(
                                    Q_nope[b_idx, kv_h, q_row, 0:d_nope],
                                    q_nope_ub[r, 0:d_nope],
                                )
                                T.copy(
                                    Q_rope[b_idx, kv_h, q_row, 0:d_rope],
                                    q_rope_ub[r, 0:d_rope],
                                )
                            else:
                                s_i = q_row // g_ratio
                                h_loc = q_row % g_ratio
                                T.copy(
                                    Q_nope[b_idx, s_i, kv_h, h_loc, 0:d_nope],
                                    q_nope_ub[r, 0:d_nope],
                                )
                                T.copy(
                                    Q_rope[b_idx, s_i, kv_h, h_loc, 0:d_rope],
                                    q_rope_ub[r, 0:d_rope],
                                )

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

                if num_stages >= 2:
                    for kv in T.Pipelined(
                        kv_limit, num_stages=num_stages, annotations=pipe_ann
                    ):
                        _kv_step(
                            kv, K_nope, K_rope, V,
                            scale_val, q_nope_l1, q_rope_l1, k_nope_l1,
                            k_rope_l1, p_l1, v0_l1, v1_l1,
                            s_l0c, o_l0c, o1_l0c, S_ub,
                            P_ub, O_tmp_ub, O_tmp1_ub, O_ub,
                            m_ub, l_ub, alpha_ub, sid,
                            b_idx, kv_h, q_base, q_rows,
                            s_dim,
                        )
                else:
                    for kv in T.serial(kv_limit):
                        _kv_step(
                            kv, K_nope, K_rope, V,
                            scale_val, q_nope_l1, q_rope_l1, k_nope_l1,
                            k_rope_l1, p_l1, v0_l1, v1_l1,
                            s_l0c, o_l0c, o1_l0c, S_ub,
                            P_ub, O_tmp_ub, O_tmp1_ub, O_ub,
                            m_ub, l_ub, alpha_ub, sid,
                            b_idx, kv_h, q_base, q_rows,
                            s_dim,
                        )

                if eff_aligned_group_q and out_tile_rows < rows_per_sid:
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
                        out_q_base = q_base + sid * rows_per_sid + rb * out_tile_rows
                        T.copy(
                            out_ub,
                            Out[
                                b_idx,
                                kv_h,
                                out_q_base : out_q_base + out_tile_rows,
                                0:d_nope,
                            ],
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

                    if eff_aligned_group_q:
                        out_q_base = q_base + sid * rows_per_sid
                        T.copy(
                            out_ub,
                            Out[
                                b_idx,
                                kv_h,
                                out_q_base : out_q_base + rows_per_sid,
                                0:d_nope,
                            ],
                        )
                    else:
                        for r in T.serial(rows_per_sid):
                            q_row = q_base + sid * rows_per_sid + r
                            if q_row < q_rows:
                                if is_struct_4d:
                                    T.copy(
                                        out_ub[r, 0:d_nope],
                                        Out[b_idx, kv_h, q_row, 0:d_nope],
                                    )
                                else:
                                    s_i = q_row // g_ratio
                                    h_loc = q_row % g_ratio
                                    T.copy(
                                        out_ub[r, 0:d_nope],
                                        Out[b_idx, s_i, kv_h, h_loc, 0:d_nope],
                                    )

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
    bn: int = 64,
    num_stages: int = 0,
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
    bn, num_stages = _resolve_mla_tile_params(bm, int(bn), int(num_stages), schedule_mode)
    d0 = min(d_nope, 256)
    d1 = d_nope - d0
    d1_alloc = max(d1, 64)
    g_ratio = n_q // n_kv
    d_nope_chunks = d_nope // 64
    d0_chunks = d0 // 64
    d1_chunks = d1 // 64
    d_rope_chunks = d_rope // 64
    num_cores = 28

    pipe_ann = {"enable_offset": True} if (num_stages >= 2 and schedule_mode == "tuned") else None
    (
        use_l1_iso, kn_v, kr_v, v0_v, v1_v,
        addr_qn, addr_qr, addr_p, addr_kn, addr_kr, addr_v0, addr_v1,
    ) = _compute_mla_l1_half_bank_plan(
        bm=bm, bn=bn, d_nope=d_nope, d_rope=d_rope,
        d0=d0, d1=d1, d1_alloc=d1_alloc, num_stages=num_stages,
        schedule_mode=schedule_mode,
    )
    func_attrs = {"half_l1": True, "l1_half_bank_split": True} if use_l1_iso else {}

    b_dim = T.dynamic("b_dim")
    q_total_rows = T.dynamic("q_total_rows")
    q_rows_sym = T.dynamic("q_rows")
    s_dim_sym = T.dynamic("s_dim")
    s_kv = T.dynamic("s_kv")

    is_struct_4d = input_layout == "BNSD" or n_kv == 1
    eff_aligned_group_q = bool(aligned_group_q and is_struct_4d)
    q_nope_shape = (
        (b_dim, n_kv, q_rows_sym, d_nope)
        if is_struct_4d
        else (b_dim, s_dim_sym, n_kv, g_ratio, d_nope)
    )
    q_rope_shape = (
        (b_dim, n_kv, q_rows_sym, d_rope)
        if is_struct_4d
        else (b_dim, s_dim_sym, n_kv, g_ratio, d_rope)
    )
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
        @T.macro
        def _kv_step(
            kv, K_nope, K_rope, V,
            scale_val, q_nope_l1, q_rope_l1, k_nope_l1,
            k_rope_l1, p_l1, v0_l1, v1_l1,
            s_l0c, o_l0c, o1_l0c, S_ub,
            P_ub, O_tmp_ub, O_tmp1_ub, O_ub,
            m_ub, l_ub, alpha_ub, sid,
            b_idx, kv_h, q_base, q_rows,
            s_dim,
        ):
            """Execute one KV tile step inside the MLA loop."""
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

        @T.prim_func
        def main(
            Q_nope: T.Tensor(q_nope_shape, tl_dtype),
            Q_rope: T.Tensor(q_rope_shape, tl_dtype),
            K_nope: T.Tensor(k_nope_shape, tl_dtype),
            K_rope: T.Tensor(k_rope_shape, tl_dtype),
            V: T.Tensor(k_nope_shape, tl_dtype),
            Out: T.Tensor(q_nope_shape, tl_dtype),
            scale_val: T.float32,
        ):
            T.func_attr(func_attrs)
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

                if not eff_aligned_group_q:
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
                    8 if (eff_aligned_group_q and bm == 128 and rows_per_sid % 8 == 0) else rows_per_sid
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
                if not eff_aligned_group_q:
                    pinned[q_nope_ub] = 1
                    pinned[q_rope_ub] = 1
                if d1 > 0 and d1 != d0:
                    pinned[o1_l0c] = 1
                    pinned[O_tmp1_ub] = 1
                if use_l1_iso:
                    pinned.update({k_nope_l1: kn_v, k_rope_l1: kr_v, v0_l1: v0_v})
                    if d1 > 0:
                        pinned[v1_l1] = v1_v
                elif schedule_mode != "tuned":
                    pinned.update(
                        {
                            k_nope_l1: 1,
                            k_rope_l1: 1,
                            v0_l1: 1,
                            v1_l1: 1,
                        }
                    )
                T.annotate_buffer_versions(pinned)
                if use_l1_iso:
                    l1_addrs = {
                        q_nope_l1: addr_qn,
                        q_rope_l1: addr_qr,
                        p_l1: addr_p,
                        k_nope_l1: addr_kn,
                        k_rope_l1: addr_kr,
                        v0_l1: addr_v0,
                    }
                    if d1 > 0:
                        l1_addrs[v1_l1] = addr_v1
                    T.annotate_buffer_addresses(l1_addrs)
                if bm == 128:
                    T.annotate_unlimit_memory("shared")

                if is_struct_4d:
                    q_rows = q_rows_sym
                    s_dim = q_rows_sym // g_ratio
                else:
                    s_dim = s_dim_sym
                    q_rows = s_dim_sym * g_ratio
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

                    if eff_aligned_group_q:
                        T.copy(
                            Q_nope[b_idx, kv_h, q_base : q_base + bm, 0:d_nope],
                            q_nope_l1,
                        )
                        T.copy(
                            Q_rope[b_idx, kv_h, q_base : q_base + bm, 0:d_rope],
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
                                if is_struct_4d:
                                    T.copy(
                                        Q_nope[b_idx, kv_h, q_row, 0:d_nope],
                                        q_nope_ub[r, 0:d_nope],
                                    )
                                    T.copy(
                                        Q_rope[b_idx, kv_h, q_row, 0:d_rope],
                                        q_rope_ub[r, 0:d_rope],
                                    )
                                else:
                                    s_i = q_row // g_ratio
                                    h_loc = q_row % g_ratio
                                    T.copy(
                                        Q_nope[b_idx, s_i, kv_h, h_loc, 0:d_nope],
                                        q_nope_ub[r, 0:d_nope],
                                    )
                                    T.copy(
                                        Q_rope[b_idx, s_i, kv_h, h_loc, 0:d_rope],
                                        q_rope_ub[r, 0:d_rope],
                                    )

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

                    if num_stages >= 2:
                        for kv in T.Pipelined(
                            kv_limit, num_stages=num_stages, annotations=pipe_ann
                        ):
                            _kv_step(
                                kv, K_nope, K_rope, V,
                                scale_val, q_nope_l1, q_rope_l1, k_nope_l1,
                                k_rope_l1, p_l1, v0_l1, v1_l1,
                                s_l0c, o_l0c, o1_l0c, S_ub,
                                P_ub, O_tmp_ub, O_tmp1_ub, O_ub,
                                m_ub, l_ub, alpha_ub, sid,
                                b_idx, kv_h, q_base, q_rows,
                                s_dim,
                            )
                    else:
                        for kv in T.serial(kv_limit):
                            _kv_step(
                                kv, K_nope, K_rope, V,
                                scale_val, q_nope_l1, q_rope_l1, k_nope_l1,
                                k_rope_l1, p_l1, v0_l1, v1_l1,
                                s_l0c, o_l0c, o1_l0c, S_ub,
                                P_ub, O_tmp_ub, O_tmp1_ub, O_ub,
                                m_ub, l_ub, alpha_ub, sid,
                                b_idx, kv_h, q_base, q_rows,
                                s_dim,
                            )

                    if eff_aligned_group_q and out_tile_rows < rows_per_sid:
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
                            out_q_base = (
                                q_base + sid * rows_per_sid + rb * out_tile_rows
                            )
                            T.copy(
                                out_ub,
                                Out[
                                    b_idx,
                                    kv_h,
                                    out_q_base : out_q_base + out_tile_rows,
                                    0:d_nope,
                                ],
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

                        if eff_aligned_group_q:
                            out_q_base = q_base + sid * rows_per_sid
                            T.copy(
                                out_ub,
                                Out[
                                    b_idx,
                                    kv_h,
                                    out_q_base : out_q_base + rows_per_sid,
                                    0:d_nope,
                                ],
                            )
                        else:
                            for r in T.serial(rows_per_sid):
                                q_row = q_base + sid * rows_per_sid + r
                                if q_row < q_rows:
                                    if is_struct_4d:
                                        T.copy(
                                            out_ub[r, 0:d_nope],
                                            Out[b_idx, kv_h, q_row, 0:d_nope],
                                        )
                                    else:
                                        s_i = q_row // g_ratio
                                        h_loc = q_row % g_ratio
                                        T.copy(
                                            out_ub[r, 0:d_nope],
                                            Out[b_idx, s_i, kv_h, h_loc, 0:d_nope],
                                        )

        return main

    @T.macro
    def _kv_split_step(
        kv_i, K_nope, K_rope, V,
        scale_val, q_nope_l1, q_rope_l1, k_nope_l1,
        k_rope_l1, p_l1, v0_l1, v1_l1,
        s_l0c, o_l0c, o1_l0c, S_ub,
        P_ub, O_tmp_ub, O_tmp1_ub, O_ub,
        m_ub, l_ub, alpha_ub, sid,
        b_idx, kv_h, q_base, q_rows,
        s_dim, kv_start, eff_kv_end,
    ):
        """Execute one KV tile step inside the MLA loop."""
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
        T.func_attr(func_attrs)
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
            if use_l1_iso:
                pinned.update({k_nope_l1: kn_v, k_rope_l1: kr_v, v0_l1: v0_v})
                if d1 > 0:
                    pinned[v1_l1] = v1_v
            elif schedule_mode != "tuned":
                pinned.update(
                    {
                        k_nope_l1: 1,
                        k_rope_l1: 1,
                        v0_l1: 1,
                        v1_l1: 1,
                    }
                )
            T.annotate_buffer_versions(pinned)
            if use_l1_iso:
                l1_addrs = {
                    q_nope_l1: addr_qn,
                    q_rope_l1: addr_qr,
                    p_l1: addr_p,
                    k_nope_l1: addr_kn,
                    k_rope_l1: addr_kr,
                    v0_l1: addr_v0,
                }
                if d1 > 0:
                    l1_addrs[v1_l1] = addr_v1
                T.annotate_buffer_addresses(l1_addrs)
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
                    if num_stages >= 2:
                        for kv_i in T.Pipelined(
                            split_iters, num_stages=num_stages, annotations=pipe_ann
                        ):
                            _kv_split_step(
                                kv_i, K_nope, K_rope, V,
                                scale_val, q_nope_l1, q_rope_l1, k_nope_l1,
                                k_rope_l1, p_l1, v0_l1, v1_l1,
                                s_l0c, o_l0c, o1_l0c, S_ub,
                                P_ub, O_tmp_ub, O_tmp1_ub, O_ub,
                                m_ub, l_ub, alpha_ub, sid,
                                b_idx, kv_h, q_base, q_rows,
                                s_dim, kv_start, eff_kv_end,
                            )
                    else:
                        for kv_i in T.serial(split_iters):
                            _kv_split_step(
                                kv_i, K_nope, K_rope, V,
                                scale_val, q_nope_l1, q_rope_l1, k_nope_l1,
                                k_rope_l1, p_l1, v0_l1, v1_l1,
                                s_l0c, o_l0c, o1_l0c, S_ub,
                                P_ub, O_tmp_ub, O_tmp1_ub, O_ub,
                                m_ub, l_ub, alpha_ub, sid,
                                b_idx, kv_h, q_base, q_rows,
                                s_dim, kv_start, eff_kv_end,
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


def _select_mla_common_overrides(
    *,
    b_dim: int,
    s_dim: int,
    n_q: int,
    n_kv: int,
    d_nope: int,
    d_rope: int,
    input_layout: str,
    schedule_mode: str,
    bm_override: int,
    bn_override: int,
    num_stages_override: int,
) -> tuple[int, int, int, bool, bool]:
    """Resolve common MLA tile knobs and Half-Bank L1 isolation flags."""
    q_rows = s_dim * (n_q // n_kv)
    bm, aligned_group_q = _choose_mla_bm(
        b_dim, n_kv, q_rows, d_nope, input_layout, schedule_mode
    )
    if int(bm_override) > 0:
        bm = int(bm_override)
        aligned_group_q = (input_layout == "BNSD" or n_kv == 1) and (q_rows % bm) == 0
    bn = 64
    if int(bn_override) > 0:
        bn = 128 if int(bn_override) >= 128 else 64
    if int(num_stages_override) > 0:
        num_stages = 2 if (int(num_stages_override) >= 2 and schedule_mode == "tuned") else 1
    else:
        num_stages = 2 if schedule_mode == "tuned" else 1
    elem_bytes = 2
    bank0_bytes = bm * (d_nope + d_rope + bn) * elem_bytes
    d0 = min(d_nope, 256)
    d1 = max(0, d_nope - d0)
    bank1_bytes = (2 * bn * (d_nope + d_rope) + bn * d0 + (bn * d1)) * elem_bytes
    l1_half_bank_split = schedule_mode == "tuned" and bank0_bytes <= 262144 and bank1_bytes <= 262144
    return bm, bn, num_stages, aligned_group_q, l1_half_bank_split


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
    bm_override: int = 0,
    bn_override: int = 0,
    num_stages_override: int = 0,
) -> dict[str, object]:
    _ = (tl_dtype, is_causal)
    q_rows = s_dim * (n_q // n_kv)
    bm, bn, num_stages, aligned_group_q, l1_half_bank_split = _select_mla_common_overrides(
        b_dim=b_dim,
        s_dim=s_dim,
        n_q=n_q,
        n_kv=n_kv,
        d_nope=d_nope,
        d_rope=d_rope,
        input_layout=input_layout,
        schedule_mode=schedule_mode,
        bm_override=bm_override,
        bn_override=bn_override,
        num_stages_override=num_stages_override,
    )
    splits = 1
    if schedule_mode == "tuned" and s_kv >= 1024 and q_rows >= 64 and (q_rows % 64) == 0:
        m_tiles = b_dim * n_kv * ((q_rows + bm - 1) // bm)
        if m_tiles < 14:
            if int(bm_override) <= 0 and q_rows >= 64:
                bm = 64
                aligned_group_q = input_layout == "BNSD" or n_kv == 1
                m_tiles = b_dim * n_kv * ((q_rows + 63) // 64)
            kv_blocks = max(1, s_kv // bn)
            target_splits = (28 + m_tiles - 1) // m_tiles
            splits = max(1, min(14, min(kv_blocks, target_splits)))
    return {
        "bm": bm,
        "bn": bn,
        "num_stages": num_stages,
        "aligned_group_q": aligned_group_q,
        "num_kv_splits": splits,
        "l1_half_bank_split": l1_half_bank_split,
        "l1_nope_bank_stride_bytes": bm * d_nope * 2,
        "l1_rope_bank_offset_bytes": 262144 + 2 * bn * d_nope * 2,
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
    bm_override: int = 0,
    bn_override: int = 0,
    num_stages_override: int = 0,
) -> dict[str, object]:
    _ = (s_kv, tl_dtype, is_causal)
    bm, bn, num_stages, aligned_group_q, l1_half_bank_split = _select_mla_common_overrides(
        b_dim=b_dim,
        s_dim=s_dim,
        n_q=n_q,
        n_kv=n_kv,
        d_nope=d_nope,
        d_rope=d_rope,
        input_layout=input_layout,
        schedule_mode=schedule_mode,
        bm_override=bm_override,
        bn_override=bn_override,
        num_stages_override=num_stages_override,
    )
    return {
        "bm": bm,
        "bn": bn,
        "num_stages": num_stages,
        "aligned_group_q": aligned_group_q,
        "num_kv_splits": 1,
        "l1_half_bank_split": l1_half_bank_split,
        "l1_nope_bank_stride_bytes": bm * d_nope * 2,
        "l1_rope_bank_offset_bytes": 262144 + 2 * bn * d_nope * 2,
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
    bn: int = 64,
    num_stages: int = 0,
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
    bn, num_stages = _resolve_mla_tile_params(bm, int(bn), int(num_stages), schedule_mode)
    d0 = min(d_nope, 256)
    d1 = d_nope - d0
    d1_alloc = max(d1, 64)
    g_ratio = n_q // n_kv
    d_nope_chunks = d_nope // 64
    d0_chunks = d0 // 64
    d1_chunks = d1 // 64
    d_rope_chunks = d_rope // 64
    num_cores = 28

    pipe_ann = {"enable_offset": True} if (num_stages >= 2 and schedule_mode == "tuned") else None
    (
        use_l1_iso, kn_v, kr_v, v0_v, v1_v,
        addr_qn, addr_qr, addr_p, addr_kn, addr_kr, addr_v0, addr_v1,
    ) = _compute_mla_l1_half_bank_plan(
        bm=bm, bn=bn, d_nope=d_nope, d_rope=d_rope,
        d0=d0, d1=d1, d1_alloc=d1_alloc, num_stages=num_stages,
        schedule_mode=schedule_mode,
    )
    func_attrs = {"half_l1": True, "l1_half_bank_split": True} if use_l1_iso else {}

    b_dim = T.dynamic("b_dim")
    q_rows_sym = T.dynamic("q_rows")
    s_kv = T.dynamic("s_kv")

    q_nope_shape = (b_dim, n_kv, q_rows_sym, d_nope)
    q_rope_shape = (b_dim, n_kv, q_rows_sym, d_rope)
    k_nope_shape = (b_dim, n_kv, s_kv, d_nope)
    k_rope_shape = (b_dim, n_kv, s_kv, d_rope)

    @T.macro
    def _kv_step(
        kv, K_nope, K_rope, V,
        scale_val, q_nope_l1, q_rope_l1, k_nope_l1,
        k_rope_l1, p_l1, v0_l1, v1_l1,
        s_l0c, o_l0c, o1_l0c, S_ub,
        P_ub, O_tmp_ub, O_tmp1_ub, O_ub,
        m_ub, l_ub, alpha_ub, sid,
        b_idx, kv_h, q_base, q_rows,
        s_dim,
    ):
        """Execute one KV tile step inside the MLA loop."""
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

    @T.prim_func
    def main(
        Q_nope: T.Tensor(q_nope_shape, tl_dtype),
        Q_rope: T.Tensor(q_rope_shape, tl_dtype),
        K_nope: T.Tensor(k_nope_shape, tl_dtype),
        K_rope: T.Tensor(k_rope_shape, tl_dtype),
        V: T.Tensor(k_nope_shape, tl_dtype),
        Out: T.Tensor(q_nope_shape, tl_dtype),
        scale_val: T.float32,
    ):
        T.func_attr(func_attrs)
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
                8 if (aligned_group_q and bm == 128 and rows_per_sid % 8 == 0) else rows_per_sid
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
            if use_l1_iso:
                pinned.update({k_nope_l1: kn_v, k_rope_l1: kr_v, v0_l1: v0_v})
                if d1 > 0:
                    pinned[v1_l1] = v1_v
            elif schedule_mode != "tuned":
                pinned.update(
                    {
                        k_nope_l1: 1,
                        k_rope_l1: 1,
                        v0_l1: 1,
                        v1_l1: 1,
                    }
                )
            T.annotate_buffer_versions(pinned)
            if use_l1_iso:
                l1_addrs = {
                    q_nope_l1: addr_qn,
                    q_rope_l1: addr_qr,
                    p_l1: addr_p,
                    k_nope_l1: addr_kn,
                    k_rope_l1: addr_kr,
                    v0_l1: addr_v0,
                }
                if d1 > 0:
                    l1_addrs[v1_l1] = addr_v1
                T.annotate_buffer_addresses(l1_addrs)
            if bm == 128:
                T.annotate_unlimit_memory("shared")

            q_rows = q_rows_sym
            s_dim = q_rows_sym // g_ratio
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
                    T.copy(
                        Q_nope[b_idx, kv_h, q_base : q_base + bm, 0:d_nope],
                        q_nope_l1,
                    )
                    T.copy(
                        Q_rope[b_idx, kv_h, q_base : q_base + bm, 0:d_rope],
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
                            T.copy(
                                Q_nope[b_idx, kv_h, q_row, 0:d_nope],
                                q_nope_ub[r, 0:d_nope],
                            )
                            T.copy(
                                Q_rope[b_idx, kv_h, q_row, 0:d_rope],
                                q_rope_ub[r, 0:d_rope],
                            )

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

                if num_stages >= 2:
                    for kv in T.Pipelined(
                        kv_limit, num_stages=num_stages, annotations=pipe_ann
                    ):
                        _kv_step(
                            kv, K_nope, K_rope, V,
                            scale_val, q_nope_l1, q_rope_l1, k_nope_l1,
                            k_rope_l1, p_l1, v0_l1, v1_l1,
                            s_l0c, o_l0c, o1_l0c, S_ub,
                            P_ub, O_tmp_ub, O_tmp1_ub, O_ub,
                            m_ub, l_ub, alpha_ub, sid,
                            b_idx, kv_h, q_base, q_rows,
                            s_dim,
                        )
                else:
                    for kv in T.serial(kv_limit):
                        _kv_step(
                            kv, K_nope, K_rope, V,
                            scale_val, q_nope_l1, q_rope_l1, k_nope_l1,
                            k_rope_l1, p_l1, v0_l1, v1_l1,
                            s_l0c, o_l0c, o1_l0c, S_ub,
                            P_ub, O_tmp_ub, O_tmp1_ub, O_ub,
                            m_ub, l_ub, alpha_ub, sid,
                            b_idx, kv_h, q_base, q_rows,
                            s_dim,
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
                        out_q_base = q_base + sid * rows_per_sid + rb * out_tile_rows
                        T.copy(
                            out_ub,
                            Out[
                                b_idx,
                                kv_h,
                                out_q_base : out_q_base + out_tile_rows,
                                0:d_nope,
                            ],
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
                        out_q_base = q_base + sid * rows_per_sid
                        T.copy(
                            out_ub,
                            Out[
                                b_idx,
                                kv_h,
                                out_q_base : out_q_base + rows_per_sid,
                                0:d_nope,
                            ],
                        )
                    else:
                        for r in T.serial(rows_per_sid):
                            q_row = q_base + sid * rows_per_sid + r
                            if q_row < q_rows:
                                T.copy(
                                    out_ub[r, 0:d_nope],
                                    Out[b_idx, kv_h, q_row, 0:d_nope],
                                )

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
    bm_override: int = 0,
    bn_override: int = 0,
    num_stages_override: int = 0,
) -> dict[str, object]:
    _ = (s_kv, tl_dtype, is_causal)
    bm, bn, num_stages, aligned_group_q, l1_half_bank_split = _select_mla_common_overrides(
        b_dim=b_dim,
        s_dim=s_dim,
        n_q=n_q,
        n_kv=n_kv,
        d_nope=d_nope,
        d_rope=d_rope,
        input_layout=input_layout,
        schedule_mode=schedule_mode,
        bm_override=bm_override,
        bn_override=bn_override,
        num_stages_override=num_stages_override,
    )
    return {
        "bm": bm,
        "bn": bn,
        "num_stages": num_stages,
        "aligned_group_q": aligned_group_q,
        "num_kv_splits": 1,
        "l1_half_bank_split": l1_half_bank_split,
        "l1_nope_bank_stride_bytes": bm * d_nope * 2,
        "l1_rope_bank_offset_bytes": 262144 + 2 * bn * d_nope * 2,
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
    bn: int = 64,
    num_stages: int = 0,
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
    bn, num_stages = _resolve_mla_tile_params(bm, int(bn), int(num_stages), schedule_mode)
    d0 = min(d_nope, 256)
    d1 = d_nope - d0
    d1_alloc = max(d1, 64)
    g_ratio = n_q // n_kv
    d_nope_chunks = d_nope // 64
    d0_chunks = d0 // 64
    d1_chunks = d1 // 64
    d_rope_chunks = d_rope // 64
    num_cores = 28

    pipe_ann = {"enable_offset": True} if (num_stages >= 2 and schedule_mode == "tuned") else None
    (
        use_l1_iso, kn_v, kr_v, v0_v, v1_v,
        addr_qn, addr_qr, addr_p, addr_kn, addr_kr, addr_v0, addr_v1,
    ) = _compute_mla_l1_half_bank_plan(
        bm=bm, bn=bn, d_nope=d_nope, d_rope=d_rope,
        d0=d0, d1=d1, d1_alloc=d1_alloc, num_stages=num_stages,
        schedule_mode=schedule_mode,
    )
    func_attrs = {"half_l1": True, "l1_half_bank_split": True} if use_l1_iso else {}

    b_dim = T.dynamic("b_dim")
    q_rows_sym = T.dynamic("q_rows")
    s_dim_sym = T.dynamic("s_dim")
    s_kv = T.dynamic("s_kv")

    is_struct_4d = n_kv == 1
    eff_aligned_group_q = bool(aligned_group_q and is_struct_4d)
    q_nope_shape = (
        (b_dim, n_kv, q_rows_sym, d_nope)
        if is_struct_4d
        else (b_dim, s_dim_sym, n_kv, g_ratio, d_nope)
    )
    q_rope_shape = (
        (b_dim, n_kv, q_rows_sym, d_rope)
        if is_struct_4d
        else (b_dim, s_dim_sym, n_kv, g_ratio, d_rope)
    )
    k_nope_shape = (b_dim, s_kv, n_kv, d_nope)
    k_rope_shape = (b_dim, s_kv, n_kv, d_rope)

    @T.macro
    def _kv_step(
        kv, K_nope, K_rope, V,
        scale_val, q_nope_l1, q_rope_l1, k_nope_l1,
        k_rope_l1, p_l1, v0_l1, v1_l1,
        s_l0c, o_l0c, o1_l0c, S_ub,
        P_ub, O_tmp_ub, O_tmp1_ub, O_ub,
        m_ub, l_ub, alpha_ub, sid,
        b_idx, kv_h, q_base, q_rows,
        s_dim,
    ):
        """Execute one KV tile step inside the MLA loop."""
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

    @T.prim_func
    def main(
        Q_nope: T.Tensor(q_nope_shape, tl_dtype),
        Q_rope: T.Tensor(q_rope_shape, tl_dtype),
        K_nope: T.Tensor(k_nope_shape, tl_dtype),
        K_rope: T.Tensor(k_rope_shape, tl_dtype),
        V: T.Tensor(k_nope_shape, tl_dtype),
        Out: T.Tensor(q_nope_shape, tl_dtype),
        scale_val: T.float32,
    ):
        T.func_attr(func_attrs)
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

            if not eff_aligned_group_q:
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
                8 if (eff_aligned_group_q and bm == 128 and rows_per_sid % 8 == 0) else rows_per_sid
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
            if not eff_aligned_group_q:
                pinned[q_nope_ub] = 1
                pinned[q_rope_ub] = 1
            if d1 > 0 and d1 != d0:
                pinned[o1_l0c] = 1
                pinned[O_tmp1_ub] = 1
            if use_l1_iso:
                pinned.update({k_nope_l1: kn_v, k_rope_l1: kr_v, v0_l1: v0_v})
                if d1 > 0:
                    pinned[v1_l1] = v1_v
            elif schedule_mode != "tuned":
                pinned.update(
                    {
                        k_nope_l1: 1,
                        k_rope_l1: 1,
                        v0_l1: 1,
                        v1_l1: 1,
                    }
                )
            T.annotate_buffer_versions(pinned)
            if use_l1_iso:
                l1_addrs = {
                    q_nope_l1: addr_qn,
                    q_rope_l1: addr_qr,
                    p_l1: addr_p,
                    k_nope_l1: addr_kn,
                    k_rope_l1: addr_kr,
                    v0_l1: addr_v0,
                }
                if d1 > 0:
                    l1_addrs[v1_l1] = addr_v1
                T.annotate_buffer_addresses(l1_addrs)
            if bm == 128:
                T.annotate_unlimit_memory("shared")

            if is_struct_4d:
                q_rows = q_rows_sym
                s_dim = q_rows_sym // g_ratio
            else:
                s_dim = s_dim_sym
                q_rows = s_dim_sym * g_ratio
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

                if eff_aligned_group_q:
                    T.copy(
                        Q_nope[b_idx, kv_h, q_base : q_base + bm, 0:d_nope],
                        q_nope_l1,
                    )
                    T.copy(
                        Q_rope[b_idx, kv_h, q_base : q_base + bm, 0:d_rope],
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
                            if is_struct_4d:
                                T.copy(
                                    Q_nope[b_idx, kv_h, q_row, 0:d_nope],
                                    q_nope_ub[r, 0:d_nope],
                                )
                                T.copy(
                                    Q_rope[b_idx, kv_h, q_row, 0:d_rope],
                                    q_rope_ub[r, 0:d_rope],
                                )
                            else:
                                s_i = q_row // g_ratio
                                h_loc = q_row % g_ratio
                                T.copy(
                                    Q_nope[b_idx, s_i, kv_h, h_loc, 0:d_nope],
                                    q_nope_ub[r, 0:d_nope],
                                )
                                T.copy(
                                    Q_rope[b_idx, s_i, kv_h, h_loc, 0:d_rope],
                                    q_rope_ub[r, 0:d_rope],
                                )

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

                if num_stages >= 2:
                    for kv in T.Pipelined(
                        kv_limit, num_stages=num_stages, annotations=pipe_ann
                    ):
                        _kv_step(
                            kv, K_nope, K_rope, V,
                            scale_val, q_nope_l1, q_rope_l1, k_nope_l1,
                            k_rope_l1, p_l1, v0_l1, v1_l1,
                            s_l0c, o_l0c, o1_l0c, S_ub,
                            P_ub, O_tmp_ub, O_tmp1_ub, O_ub,
                            m_ub, l_ub, alpha_ub, sid,
                            b_idx, kv_h, q_base, q_rows,
                            s_dim,
                        )
                else:
                    for kv in T.serial(kv_limit):
                        _kv_step(
                            kv, K_nope, K_rope, V,
                            scale_val, q_nope_l1, q_rope_l1, k_nope_l1,
                            k_rope_l1, p_l1, v0_l1, v1_l1,
                            s_l0c, o_l0c, o1_l0c, S_ub,
                            P_ub, O_tmp_ub, O_tmp1_ub, O_ub,
                            m_ub, l_ub, alpha_ub, sid,
                            b_idx, kv_h, q_base, q_rows,
                            s_dim,
                        )

                if eff_aligned_group_q and out_tile_rows < rows_per_sid:
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
                        out_q_base = q_base + sid * rows_per_sid + rb * out_tile_rows
                        T.copy(
                            out_ub,
                            Out[
                                b_idx,
                                kv_h,
                                out_q_base : out_q_base + out_tile_rows,
                                0:d_nope,
                            ],
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

                    if eff_aligned_group_q:
                        out_q_base = q_base + sid * rows_per_sid
                        T.copy(
                            out_ub,
                            Out[
                                b_idx,
                                kv_h,
                                out_q_base : out_q_base + rows_per_sid,
                                0:d_nope,
                            ],
                        )
                    else:
                        for r in T.serial(rows_per_sid):
                            q_row = q_base + sid * rows_per_sid + r
                            if q_row < q_rows:
                                if is_struct_4d:
                                    T.copy(
                                        out_ub[r, 0:d_nope],
                                        Out[b_idx, kv_h, q_row, 0:d_nope],
                                    )
                                else:
                                    s_i = q_row // g_ratio
                                    h_loc = q_row % g_ratio
                                    T.copy(
                                        out_ub[r, 0:d_nope],
                                        Out[b_idx, s_i, kv_h, h_loc, 0:d_nope],
                                    )

    return main


_MLA_TRACKB_SRC = r"""
#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/gemm.h>
#include <tl_templates/ascend/nd2nz_copy.h>
#include <tl_templates/ascend/simd_inst.h>

template <int M, int K, int N, int TILE_K_SUB, typename InT, typename AccumT>
__aicore__ inline void ascend_gemm_l1_nn(__cc__ AccumT *cc_ptr, __cbuf__ InT *cbuf_a_ptr,
                                          __cbuf__ InT *cbuf_b_ptr, int clear_accum, int buf_offset = 0) {
  constexpr int C0 = 32 / sizeof(InT);
  constexpr int FRAC_N = 16;
  constexpr int SUB_K = K / TILE_K_SUB;
  constexpr int mStepA = M / FRAC_N;
  constexpr int mStepB = N / FRAC_N;
  constexpr int kStep = TILE_K_SUB / C0;
  constexpr int L0A_STAGE = M * TILE_K_SUB;
  constexpr int L0B_STAGE = N * TILE_K_SUB;
  constexpr int srcStrideNN = K / C0;

  __ca__ InT *l0a = (__ca__ InT *)0;
  __cb__ InT *l0b = (__cb__ InT *)0;

  for (int sk = 0; sk < SUB_K; ++sk) {
    asc_lock(PIPE_MTE1, buf_offset + (sk & 1), ASC_LOCK_BLOCK);
    asc_copy_l12l0a((__ca__ InT *)(l0a + ((sk & 1) * L0A_STAGE)), cbuf_a_ptr, 0,
                    (sk * kStep), mStepA, kStep, mStepA, mStepA);
    asc_copy_l12l0b_transpose((__cb__ InT *)(l0b + ((sk & 1) * L0B_STAGE)), cbuf_b_ptr,
                              (sk * kStep), 0, kStep, mStepB, srcStrideNN, mStepB);
    asc_unlock(PIPE_MTE1, buf_offset + (sk & 1), ASC_LOCK_BLOCK);
    asc_lock(PIPE_M, buf_offset + (sk & 1), ASC_LOCK_BLOCK);
    asc_mmad(cc_ptr, (__ca__ InT *)(l0a + ((sk & 1) * L0A_STAGE)),
             (__cb__ InT *)(l0b + ((sk & 1) * L0B_STAGE)), M, TILE_K_SUB, N,
             0, true, false, clear_accum && (sk == 0));
    asc_unlock(PIPE_M, buf_offset + (sk & 1), ASC_LOCK_BLOCK);
  }
}

template <int ROWS, int D_NOPE>
__simd_vf__ inline void vf_init_state(__ubuf__ float* m_prev, __ubuf__ float* l_prev, __ubuf__ float* acc_o) {
  vector_bool m32 = asc_create_mask_b32(PAT_ALL);
  auto v_neg_inf = simd_inst::vdup<float>(-1e30f, m32, MODE_ZEROING);
  auto v_zero = simd_inst::vdup<float>(0.0f, m32, MODE_ZEROING);
  for (int r = 0; r < ROWS; ++r) {
    simd_inst::vsts_1st(v_neg_inf, m_prev + r * 8, 0, m32);
    simd_inst::vsts_1st(v_zero, l_prev + r * 8, 0, m32);
  }
  constexpr int CHUNKS = D_NOPE >> 6;
  for (int r = 0; r < ROWS; ++r) {
    for (int c = 0; c < CHUNKS; ++c) {
      simd_inst::vsts_norm(v_zero, acc_o + r * D_NOPE + (c << 6), 0, m32);
    }
  }
}

template <int ROWS, int BN, typename OutT>
__simd_vf__ inline void vf_softmax_tail(__ubuf__ float* s_ub, __ubuf__ OutT* p_ub,
                                        __ubuf__ float* m_prev, __ubuf__ float* l_prev,
                                        __ubuf__ float* alpha_ub, int cur_n, float scale) {
  vector_bool m32 = asc_create_mask_b32(PAT_ALL);
  vector_bool m16 = asc_create_mask_b16(PAT_ALL);
  auto v_zero = simd_inst::vdup<float>(0.0f, m32, MODE_ZEROING);
  auto v_neg_inf = simd_inst::vdup<float>(-1e30f, m32, MODE_ZEROING);
  auto col_base = simd_inst::vci<int32_t>(0, INC_ORDER);
  auto col_hi = simd_inst::vci<int32_t>(64, INC_ORDER);
  auto valid0 = simd_inst::vcmps_lt(col_base, cur_n, m32);
  auto valid1 = simd_inst::vcmps_lt(col_hi, cur_n, m32);

  for (int r = 0; r < ROWS; ++r) {
    auto s0 = simd_inst::vsel(simd_inst::vmuls(simd_inst::vlds_norm<float>(s_ub + r * BN, 0), scale, m32, MODE_ZEROING), v_neg_inf, valid0);
    auto s1 = simd_inst::vsel(simd_inst::vmuls(simd_inst::vlds_norm<float>(s_ub + r * BN + 64, 0), scale, m32, MODE_ZEROING), v_neg_inf, valid1);
    auto smax = simd_inst::vmax(s0, s1, m32, MODE_ZEROING);
    auto m_tile_v = simd_inst::vdupv(simd_inst::vcmax(smax, m32, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
    auto m_old_v = simd_inst::vlds_brc_elem<float>(m_prev + r * 8, 0);
    auto l_old_v = simd_inst::vlds_brc_elem<float>(l_prev + r * 8, 0);
    auto m_new_v = simd_inst::vmax(m_old_v, m_tile_v, m32, MODE_ZEROING);
    auto m_safe_v = simd_inst::vmaxs(m_new_v, -1e30f, m32, MODE_ZEROING);
    auto alpha_raw = simd_inst::vexp(simd_inst::vsub(m_old_v, m_safe_v, m32, MODE_ZEROING), m32, MODE_ZEROING);
    auto old_valid = simd_inst::vcmps_gt(m_old_v, -1e30f, m32);
    auto alpha_v = simd_inst::vsel(alpha_raw, v_zero, old_valid);
    auto new_valid = simd_inst::vcmps_gt(m_new_v, -1e30f, m32);
    auto p0_raw = simd_inst::vexp(simd_inst::vsub(s0, m_safe_v, m32, MODE_ZEROING), m32, MODE_ZEROING);
    auto p1_raw = simd_inst::vexp(simd_inst::vsub(s1, m_safe_v, m32, MODE_ZEROING), m32, MODE_ZEROING);
    auto p0_f = simd_inst::vsel(p0_raw, v_zero, simd_inst::pand(valid0, new_valid, m32));
    auto p1_f = simd_inst::vsel(p1_raw, v_zero, simd_inst::pand(valid1, new_valid, m32));
    auto row_sum_v = simd_inst::vdupv(simd_inst::vcadd(simd_inst::vadd(p0_f, p1_f, m32, MODE_ZEROING), m32, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
    auto l_new_v = simd_inst::vadd(simd_inst::vmul(l_old_v, alpha_v, m32, MODE_ZEROING), row_sum_v, m32, MODE_ZEROING);
    simd_inst::vsts_1st(m_new_v, m_prev + r * 8, 0, m32);
    simd_inst::vsts_1st(l_new_v, l_prev + r * 8, 0, m32);
    simd_inst::vsts_1st(alpha_v, alpha_ub + r * 8, 0, m32);
    auto p0_16 = simd_inst::vcvt<OutT>(p0_f, m16, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
    auto p1_16 = simd_inst::vcvt<OutT>(p1_f, m16, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
    simd_inst::vsts_pack_b32(p0_16, p_ub + r * BN, 0, m32);
    simd_inst::vsts_pack_b32(p1_16, p_ub + r * BN + 64, 0, m32);
  }
}

template <int ROWS, int BN, typename OutT>
__simd_vf__ inline void vf_softmax_full(__ubuf__ float* s_ub, __ubuf__ OutT* p_ub,
                                        __ubuf__ float* m_prev, __ubuf__ float* l_prev,
                                        __ubuf__ float* alpha_ub, float scale) {
  vector_bool m32 = asc_create_mask_b32(PAT_ALL);
  vector_bool m16 = asc_create_mask_b16(PAT_ALL);
  auto v_zero = simd_inst::vdup<float>(0.0f, m32, MODE_ZEROING);
  for (int r = 0; r < ROWS; ++r) {
    auto s0 = simd_inst::vmuls(simd_inst::vlds_norm<float>(s_ub + r * BN, 0), scale, m32, MODE_ZEROING);
    auto s1 = simd_inst::vmuls(simd_inst::vlds_norm<float>(s_ub + r * BN + 64, 0), scale, m32, MODE_ZEROING);
    auto smax = simd_inst::vmax(s0, s1, m32, MODE_ZEROING);
    auto m_tile_v = simd_inst::vdupv(simd_inst::vcmax(smax, m32, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
    auto m_old_v = simd_inst::vlds_brc_elem<float>(m_prev + r * 8, 0);
    auto l_old_v = simd_inst::vlds_brc_elem<float>(l_prev + r * 8, 0);
    auto m_new_v = simd_inst::vmax(m_old_v, m_tile_v, m32, MODE_ZEROING);
    auto alpha_raw = simd_inst::vexp(simd_inst::vsub(m_old_v, m_new_v, m32, MODE_ZEROING), m32, MODE_ZEROING);
    auto old_valid = simd_inst::vcmps_gt(m_old_v, -1e30f, m32);
    auto alpha_v = simd_inst::vsel(alpha_raw, v_zero, old_valid);
    auto p0_f = simd_inst::vexp(simd_inst::vsub(s0, m_new_v, m32, MODE_ZEROING), m32, MODE_ZEROING);
    auto p1_f = simd_inst::vexp(simd_inst::vsub(s1, m_new_v, m32, MODE_ZEROING), m32, MODE_ZEROING);
    auto row_sum_v = simd_inst::vdupv(simd_inst::vcadd(simd_inst::vadd(p0_f, p1_f, m32, MODE_ZEROING), m32, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
    auto l_new_v = simd_inst::vadd(simd_inst::vmul(l_old_v, alpha_v, m32, MODE_ZEROING), row_sum_v, m32, MODE_ZEROING);
    simd_inst::vsts_1st(m_new_v, m_prev + r * 8, 0, m32);
    simd_inst::vsts_1st(l_new_v, l_prev + r * 8, 0, m32);
    simd_inst::vsts_1st(alpha_v, alpha_ub + r * 8, 0, m32);
    auto p0_16 = simd_inst::vcvt<OutT>(p0_f, m16, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
    auto p1_16 = simd_inst::vcvt<OutT>(p1_f, m16, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
    simd_inst::vsts_pack_b32(p0_16, p_ub + r * BN, 0, m32);
    simd_inst::vsts_pack_b32(p1_16, p_ub + r * BN + 64, 0, m32);
  }
}

template <int ROWS, int N_SLICE, int D_NOPE, int COL_OFF>
__simd_vf__ inline void vf_acc_rescale(__ubuf__ float* acc_o, __ubuf__ float* o_ub, __ubuf__ float* alpha_ub) {
  vector_bool m32 = asc_create_mask_b32(PAT_ALL);
  constexpr int CHUNKS = N_SLICE >> 6;
  for (int r = 0; r < ROWS; ++r) {
    auto alpha_v = simd_inst::vlds_brc_elem<float>(alpha_ub + r * 8, 0);
    for (int c = 0; c < CHUNKS; ++c) {
      auto o_old = simd_inst::vlds_norm<float>(acc_o + r * D_NOPE + COL_OFF + (c << 6), 0);
      auto o_tmp = simd_inst::vlds_norm<float>(o_ub + r * N_SLICE + (c << 6), 0);
      auto o_new = simd_inst::vadd(simd_inst::vmul(o_old, alpha_v, m32, MODE_ZEROING), o_tmp, m32, MODE_ZEROING);
      simd_inst::vsts_norm(o_new, acc_o + r * D_NOPE + COL_OFF + (c << 6), 0, m32);
    }
  }
}

template <int ROWS, int D_NOPE, typename OutT>
__simd_vf__ inline void vf_finalize_out(__ubuf__ float* acc_o, __ubuf__ float* l_prev, __ubuf__ OutT* out_ub) {
  vector_bool m32 = asc_create_mask_b32(PAT_ALL);
  vector_bool m16 = asc_create_mask_b16(PAT_ALL);
  auto v_one = simd_inst::vdup<float>(1.0f, m32, MODE_ZEROING);
  auto v_zero = simd_inst::vdup<float>(0.0f, m32, MODE_ZEROING);
  constexpr int CHUNKS = D_NOPE >> 6;
  for (int r = 0; r < ROWS; ++r) {
    auto l_v = simd_inst::vlds_brc_elem<float>(l_prev + r * 8, 0);
    auto l_pos = simd_inst::vcmps_gt(l_v, 1e-30f, m32);
    auto l_safe = simd_inst::vsel(l_v, v_one, l_pos);
    auto inv_l = simd_inst::vsel(simd_inst::vdiv(v_one, l_safe, m32, MODE_ZEROING), v_zero, l_pos);
    for (int c = 0; c < CHUNKS; ++c) {
      auto y_f = simd_inst::vmul(simd_inst::vlds_norm<float>(acc_o + r * D_NOPE + (c << 6), 0), inv_l, m32, MODE_ZEROING);
      auto y_16 = simd_inst::vcvt<OutT>(y_f, m16, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
      simd_inst::vsts_pack_b32(y_16, out_ub + r * D_NOPE + (c << 6), 0, m32);
    }
  }
}

template <int ROWS, int D_NOPE>
__simd_vf__ inline void vf_finalize_partial(__ubuf__ float* acc_o, __ubuf__ float* m_prev, __ubuf__ float* l_prev) {
  vector_bool m32 = asc_create_mask_b32(PAT_ALL);
  auto v_one = simd_inst::vdup<float>(1.0f, m32, MODE_ZEROING);
  auto v_zero = simd_inst::vdup<float>(0.0f, m32, MODE_ZEROING);
  auto v_neg_inf = simd_inst::vdup<float>(-1e30f, m32, MODE_ZEROING);
  constexpr int CHUNKS = D_NOPE >> 6;
  for (int r = 0; r < ROWS; ++r) {
    auto m_v = simd_inst::vlds_brc_elem<float>(m_prev + r * 8, 0);
    auto l_v = simd_inst::vlds_brc_elem<float>(l_prev + r * 8, 0);
    auto l_pos = simd_inst::vcmps_gt(l_v, 1e-30f, m32);
    auto l_safe = simd_inst::vsel(l_v, v_one, l_pos);
    auto inv_l = simd_inst::vsel(simd_inst::vdiv(v_one, l_safe, m32, MODE_ZEROING), v_zero, l_pos);
    auto lse_v = simd_inst::vsel(simd_inst::vadd(m_v, simd_inst::vln(l_safe, m32, MODE_ZEROING), m32, MODE_ZEROING), v_neg_inf, l_pos);
    for (int c = 0; c < CHUNKS; ++c) {
      auto y_f = simd_inst::vmul(simd_inst::vlds_norm<float>(acc_o + r * D_NOPE + (c << 6), 0), inv_l, m32, MODE_ZEROING);
      simd_inst::vsts_norm(y_f, acc_o + r * D_NOPE + (c << 6), 0, m32);
    }
    simd_inst::vsts_1st(lse_v, l_prev + r * 8, 0, m32);
  }
}

template <int BM, int BN, int D_NOPE, int D_ROPE, typename InT>
__global__ __mix__(1, 2) void mla_kv_reuse_c1c1c2(
    __gm__ InT* Q_nope, __gm__ InT* Q_rope,
    __gm__ InT* K_nope, __gm__ InT* K_rope,
    __gm__ InT* Out, __gm__ float* O_part, __gm__ float* lse_part,
    int B, int S_q, int S_kv, int N_q, int splits, int causal, int is_bnsd, float scale) {
  asc_init();
  constexpr int ROWS = BM / 2;
  constexpr int NS = (BM <= 64) ? 256 : 128;
  constexpr int NUM_SLICES = (D_NOPE + NS - 1) / NS;
  constexpr int LAST_NS = D_NOPE - (NUM_SLICES - 1) * NS;

  constexpr int Q_NOPE_ELEMS = BM * D_NOPE;
  constexpr int Q_ROPE_ELEMS = BM * D_ROPE;
  constexpr int P_ELEMS = BM * BN;
  constexpr int K_NOPE_STAGE = BN * D_NOPE;
  constexpr int K_ROPE_STAGE = BN * D_ROPE;

  constexpr int L1_HALF_BANK_BYTES = 262144;
  constexpr int Q_P_END_BYTES = (Q_NOPE_ELEMS + Q_ROPE_ELEMS + 2 * P_ELEMS) * 2;
  constexpr int K_NOPE_STAGE_BYTES = K_NOPE_STAGE * 2;
  constexpr int l1_half_bank_split = (Q_P_END_BYTES + K_NOPE_STAGE_BYTES <= L1_HALF_BANK_BYTES) ? 1 : 0;
  constexpr int K_NOPE_BASE_BYTES = (Q_P_END_BYTES + K_NOPE_STAGE_BYTES <= L1_HALF_BANK_BYTES)
      ? (L1_HALF_BANK_BYTES - K_NOPE_STAGE_BYTES)
      : Q_P_END_BYTES;

  __cbuf__ InT* q_nope_l1 = (__cbuf__ InT*)0;
  __cbuf__ InT* q_rope_l1 = (__cbuf__ InT*)(Q_NOPE_ELEMS * 2);
  __cbuf__ InT* p_l1_base = (__cbuf__ InT*)((Q_NOPE_ELEMS + Q_ROPE_ELEMS) * 2);
  __cbuf__ InT* k_nope_l1_base = (__cbuf__ InT*)(K_NOPE_BASE_BYTES);
  __cbuf__ InT* k_rope_l1_base = (__cbuf__ InT*)(K_NOPE_BASE_BYTES + 2 * K_NOPE_STAGE_BYTES);

  __cc__ float* s_l0c = (__cc__ float*)0;
  __cc__ float* o_l0c = (__cc__ float*)(BM * BN * 4);

  constexpr int ACC_BYTES = ROWS * D_NOPE * 4;
  constexpr int O_BYTES = ROWS * NS * 4;
  constexpr int S_BYTES = ROWS * BN * 4;
  constexpr int P_BYTES = ROWS * BN * 2;
  constexpr int PNZ_BYTES = ((ROWS + 1) * BN * 2 + 255) & ~255;
  __ubuf__ float* acc_o_ub = (__ubuf__ float*)0;
  __ubuf__ float* o_ub = (__ubuf__ float*)(ACC_BYTES);
  __ubuf__ InT* out_ub = (__ubuf__ InT*)(ACC_BYTES);
  __ubuf__ float* s_ub = (__ubuf__ float*)(ACC_BYTES + O_BYTES);
  __ubuf__ InT* p_ub = (__ubuf__ InT*)(ACC_BYTES + O_BYTES + S_BYTES);
  __ubuf__ InT* p_nz_ub = (__ubuf__ InT*)(ACC_BYTES + O_BYTES + S_BYTES + P_BYTES);
  int meta_off = ACC_BYTES + O_BYTES + S_BYTES + P_BYTES + PNZ_BYTES;
  __ubuf__ float* m_prev_ub = (__ubuf__ float*)(meta_off);
  __ubuf__ float* l_prev_ub = (__ubuf__ float*)(meta_off + ROWS * 32);
  __ubuf__ float* alpha_ub0 = (__ubuf__ float*)(meta_off + ROWS * 64);
  __ubuf__ float* alpha_ub1 = (__ubuf__ float*)(meta_off + ROWS * 96);
  __ubuf__ float* lse_pack_ub = (__ubuf__ float*)(meta_off + ROWS * 128);

  int total_bs = B * S_q;
  int q_blocks = (N_q + BM - 1) / BM;
  int total_tasks = total_bs * q_blocks * splits;
  int cid = (int)block_idx;
  int num_cores = 28;
  if (cid >= total_tasks) return;

  int q_head_stride = is_bnsd ? (S_q * D_NOPE) : D_NOPE;
  int qr_head_stride = is_bnsd ? (S_q * D_ROPE) : D_ROPE;
  int out_head_stride = is_bnsd ? (S_q * D_NOPE) : D_NOPE;

  if ASCEND_IS_AIC {
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, (event_t)0);
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, (event_t)1);
    asc_sync_notify(PIPE_M, PIPE_MTE2, (event_t)2);
    asc_sync_notify(PIPE_FIX, PIPE_M, (event_t)0);
    asc_sync_notify(PIPE_FIX, PIPE_M, (event_t)1);
    bool prev_ran = false;

    for (int t = cid; t < total_tasks; t += num_cores) {
      int sp = t % splits;
      int t2 = t / splits;
      int qb = t2 % q_blocks;
      int bs = t2 / q_blocks;
      int b = bs / S_q;
      int s = bs % S_q;
      int m_off = qb * BM;
      int actual_m = N_q - m_off;
      if (actual_m > BM) actual_m = BM;

      int kv_end = causal ? (S_kv - S_q + s + 1) : S_kv;
      if (kv_end > S_kv) kv_end = S_kv;
      if (kv_end < 0) kv_end = 0;
      int total_kv_blks = (kv_end + BN - 1) / BN;
      int sp_blks = (total_kv_blks + splits - 1) / splits;
      int blk_start = sp * sp_blks;
      int blk_end = blk_start + sp_blks;
      if (blk_end > total_kv_blks) blk_end = total_kv_blks;
      if (blk_start >= blk_end) continue;

      int q_base = is_bnsd
          ? ((b * N_q + m_off) * S_q + s) * D_NOPE
          : ((b * S_q + s) * N_q + m_off) * D_NOPE;
      int qr_base = is_bnsd
          ? ((b * N_q + m_off) * S_q + s) * D_ROPE
          : ((b * S_q + s) * N_q + m_off) * D_ROPE;

      asc_sync_wait(PIPE_M, PIPE_MTE2, (event_t)2);
      asc_set_gm2l1_nz_para(1, 1, BM, 0);
      asc_copy_gm2l1_nd2nz(q_nope_l1, Q_nope + q_base, q_head_stride * 2, static_cast<asc_load_l2_cache_mode>(0), actual_m, D_NOPE, 0, 0);
      asc_copy_gm2l1_nd2nz(q_rope_l1, Q_rope + qr_base, qr_head_stride * 2, static_cast<asc_load_l2_cache_mode>(0), actual_m, D_ROPE, 0, 0);
      asc_sync_notify(PIPE_MTE2, PIPE_MTE1, (event_t)2);
      asc_sync_wait(PIPE_MTE2, PIPE_MTE1, (event_t)2);

      int k_batch_base_n = b * S_kv * D_NOPE;
      int k_batch_base_r = b * S_kv * D_ROPE;
      int num_blks = blk_end - blk_start;

      {
        int kv_idx0 = blk_start * BN;
        int cur_n0 = kv_end - kv_idx0;
        if (cur_n0 > BN) cur_n0 = BN;
        asc_sync_wait(PIPE_MTE1, PIPE_MTE2, (event_t)0);
        if (cur_n0 < BN) {
          asc_fill_value_config fc{}; fc.repeat = D_NOPE / 16; fc.blk_num = BN; fc.dst_gap = 0; asc_fill_l1((__cbuf__ uint16_t*)k_nope_l1_base, (uint32_t)0, fc);
          asc_fill_value_config fr{}; fr.repeat = D_ROPE / 16; fr.blk_num = BN; fr.dst_gap = 0; asc_fill_l1((__cbuf__ uint16_t*)k_rope_l1_base, (uint32_t)0, fr);
          pipe_barrier(PIPE_MTE2);
        }
        asc_set_gm2l1_nz_para(1, 1, BN, 0);
        asc_copy_gm2l1_nd2nz(k_nope_l1_base, K_nope + k_batch_base_n + kv_idx0 * D_NOPE, D_NOPE * 2, static_cast<asc_load_l2_cache_mode>(0), cur_n0, D_NOPE, 0, 0);
        asc_copy_gm2l1_nd2nz(k_rope_l1_base, K_rope + k_batch_base_r + kv_idx0 * D_ROPE, D_ROPE * 2, static_cast<asc_load_l2_cache_mode>(0), cur_n0, D_ROPE, 0, 0);
        asc_sync_notify(PIPE_MTE2, PIPE_MTE1, (event_t)0);

        if (num_blks > 1) {
          int kv_idx1 = (blk_start + 1) * BN;
          int cur_n1 = kv_end - kv_idx1;
          if (cur_n1 > BN) cur_n1 = BN;
          asc_sync_wait(PIPE_MTE1, PIPE_MTE2, (event_t)1);
          if (cur_n1 < BN) {
            asc_fill_value_config fc1{}; fc1.repeat = D_NOPE / 16; fc1.blk_num = BN; fc1.dst_gap = 0; asc_fill_l1((__cbuf__ uint16_t*)(k_nope_l1_base + K_NOPE_STAGE), (uint32_t)0, fc1);
            asc_fill_value_config fr1{}; fr1.repeat = D_ROPE / 16; fr1.blk_num = BN; fr1.dst_gap = 0; asc_fill_l1((__cbuf__ uint16_t*)(k_rope_l1_base + K_ROPE_STAGE), (uint32_t)0, fr1);
            pipe_barrier(PIPE_MTE2);
          }
          asc_set_gm2l1_nz_para(1, 1, BN, 0);
          asc_copy_gm2l1_nd2nz(k_nope_l1_base + K_NOPE_STAGE, K_nope + k_batch_base_n + kv_idx1 * D_NOPE, D_NOPE * 2, static_cast<asc_load_l2_cache_mode>(0), cur_n1, D_NOPE, 0, 0);
          asc_copy_gm2l1_nd2nz(k_rope_l1_base + K_ROPE_STAGE, K_rope + k_batch_base_r + kv_idx1 * D_ROPE, D_ROPE * 2, static_cast<asc_load_l2_cache_mode>(0), cur_n1, D_ROPE, 0, 0);
          asc_sync_notify(PIPE_MTE2, PIPE_MTE1, (event_t)1);
        }

        asc_sync_wait(PIPE_MTE2, PIPE_MTE1, (event_t)0);
        asc_sync_wait(PIPE_FIX, PIPE_M, (event_t)0);
        ascend_gemm_l1<BM, D_NOPE, BN, 64, true, InT>(s_l0c, q_nope_l1, k_nope_l1_base, 1, 0, 0);
        ascend_gemm_l1<BM, D_ROPE, BN, 64, true, InT>(s_l0c, q_rope_l1, k_rope_l1_base, 0, 0, 0);
        asc_sync_notify(PIPE_M, PIPE_FIX, (event_t)0);
        asc_sync_wait(PIPE_M, PIPE_FIX, (event_t)0);
        if (prev_ran) {
          asc_sync_intra_wait(PIPE_FIX, 6);
          asc_sync_intra_wait(PIPE_FIX, 22);
        }
        asc_set_l0c_copy_nz_para(1, 0, 0);
        asc_copy_l0c2ub(s_ub, s_l0c, BN, BM, BN, BM, 0, static_cast<asc_dual_dst_mode>(1), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
        asc_sync_notify(PIPE_FIX, PIPE_M, (event_t)0);
        asc_sync_intra_arrive(PIPE_FIX, 5);
        asc_sync_intra_arrive(PIPE_FIX, 21);
      }

      for (int bi = 0; bi < num_blks; ++bi) {
        int stg = bi & 1;
        __cbuf__ InT* kn_dst = k_nope_l1_base + stg * K_NOPE_STAGE;
        __cbuf__ InT* p_dst = p_l1_base + stg * P_ELEMS;

        if (bi + 1 < num_blks) {
          int nxt = (bi + 1) & 1;
          asc_sync_wait(PIPE_MTE2, PIPE_MTE1, (event_t)nxt);
          asc_sync_wait(PIPE_FIX, PIPE_M, (event_t)0);
          ascend_gemm_l1<BM, D_NOPE, BN, 64, true, InT>(s_l0c, q_nope_l1, k_nope_l1_base + nxt * K_NOPE_STAGE, 1, 0, 0);
          ascend_gemm_l1<BM, D_ROPE, BN, 64, true, InT>(s_l0c, q_rope_l1, k_rope_l1_base + nxt * K_ROPE_STAGE, 0, 0, 0);
          asc_sync_notify(PIPE_M, PIPE_FIX, (event_t)0);

          asc_sync_intra_wait(PIPE_MTE1, 4);
          asc_sync_intra_wait(PIPE_MTE1, 20);
          asc_sync_wait(PIPE_M, PIPE_FIX, (event_t)0);
          asc_set_l0c_copy_nz_para(1, 0, 0);
          asc_copy_l0c2ub(s_ub, s_l0c, BN, BM, BN, BM, 0, static_cast<asc_dual_dst_mode>(1), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
          asc_sync_notify(PIPE_FIX, PIPE_M, (event_t)0);
          asc_sync_intra_arrive(PIPE_FIX, 5);
          asc_sync_intra_arrive(PIPE_FIX, 21);
        } else {
          asc_sync_intra_wait(PIPE_MTE1, 4);
          asc_sync_intra_wait(PIPE_MTE1, 20);
        }

        for (int sl = 0; sl < NUM_SLICES; ++sl) {
          asc_sync_wait(PIPE_FIX, PIPE_M, (event_t)1);
          if (sl < NUM_SLICES - 1) {
            ascend_gemm_l1_nn<BM, BN, NS, 64, InT>(o_l0c, p_dst, kn_dst + sl * BN * NS, 1, 0);
          } else {
            ascend_gemm_l1_nn<BM, BN, LAST_NS, 64, InT>(o_l0c, p_dst, kn_dst + sl * BN * NS, 1, 0);
            asc_sync_notify(PIPE_MTE1, PIPE_MTE2, (event_t)stg);
            if (bi + 2 < num_blks) {
              int kv_idx2 = (blk_start + bi + 2) * BN;
              int cur_n2 = kv_end - kv_idx2;
              if (cur_n2 > BN) cur_n2 = BN;
              asc_sync_wait(PIPE_MTE1, PIPE_MTE2, (event_t)stg);
              if (cur_n2 < BN) {
                asc_fill_value_config fc2{}; fc2.repeat = D_NOPE / 16; fc2.blk_num = BN; fc2.dst_gap = 0; asc_fill_l1((__cbuf__ uint16_t*)(k_nope_l1_base + stg * K_NOPE_STAGE), (uint32_t)0, fc2);
                asc_fill_value_config fr2{}; fr2.repeat = D_ROPE / 16; fr2.blk_num = BN; fr2.dst_gap = 0; asc_fill_l1((__cbuf__ uint16_t*)(k_rope_l1_base + stg * K_ROPE_STAGE), (uint32_t)0, fr2);
                pipe_barrier(PIPE_MTE2);
              }
              asc_set_gm2l1_nz_para(1, 1, BN, 0);
              asc_copy_gm2l1_nd2nz(k_nope_l1_base + stg * K_NOPE_STAGE, K_nope + k_batch_base_n + kv_idx2 * D_NOPE, D_NOPE * 2, static_cast<asc_load_l2_cache_mode>(0), cur_n2, D_NOPE, 0, 0);
              asc_copy_gm2l1_nd2nz(k_rope_l1_base + stg * K_ROPE_STAGE, K_rope + k_batch_base_r + kv_idx2 * D_ROPE, D_ROPE * 2, static_cast<asc_load_l2_cache_mode>(0), cur_n2, D_ROPE, 0, 0);
              asc_sync_notify(PIPE_MTE2, PIPE_MTE1, (event_t)stg);
            }
          }
          asc_sync_notify(PIPE_M, PIPE_FIX, (event_t)1);
          asc_sync_wait(PIPE_M, PIPE_FIX, (event_t)1);
          if (!(bi == 0 && sl == 0)) {
            asc_sync_intra_wait(PIPE_FIX, 6);
            asc_sync_intra_wait(PIPE_FIX, 22);
          }
          int cur_ns = (sl < NUM_SLICES - 1) ? NS : LAST_NS;
          asc_set_l0c_copy_nz_para(1, 0, 0);
          asc_copy_l0c2ub(o_ub, o_l0c, cur_ns, BM, cur_ns, BM, 0, static_cast<asc_dual_dst_mode>(1), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
          asc_sync_notify(PIPE_FIX, PIPE_M, (event_t)1);
          asc_sync_intra_arrive(PIPE_FIX, 3);
          asc_sync_intra_arrive(PIPE_FIX, 19);
        }
      }
      asc_sync_notify(PIPE_M, PIPE_MTE2, (event_t)2);
      asc_sync_intra_wait(PIPE_FIX, 6);
      asc_sync_intra_wait(PIPE_FIX, 22);
      prev_ran = true;
    }

    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, (event_t)0);
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, (event_t)1);
    asc_sync_wait(PIPE_M, PIPE_MTE2, (event_t)2);
    asc_sync_wait(PIPE_FIX, PIPE_M, (event_t)0);
    asc_sync_wait(PIPE_FIX, PIPE_M, (event_t)1);
  }

  if ASCEND_IS_AIV {
    int sid = asc_get_sub_block_id();
    int arrive_p = (sid == 0) ? 4 : 20;
    int arrive_oub = (sid == 0) ? 6 : 22;

    for (int t = cid; t < total_tasks; t += num_cores) {
      int sp = t % splits;
      int t2 = t / splits;
      int qb = t2 % q_blocks;
      int bs = t2 / q_blocks;
      int b = bs / S_q;
      int s = bs % S_q;
      int m_off = qb * BM;
      int actual_m = N_q - m_off;
      if (actual_m > BM) actual_m = BM;
      int my_m_start = sid * ROWS;
      int my_rows = actual_m - my_m_start;
      if (my_rows > ROWS) my_rows = ROWS;
      if (my_rows < 0) my_rows = 0;

      int kv_end = causal ? (S_kv - S_q + s + 1) : S_kv;
      if (kv_end > S_kv) kv_end = S_kv;
      if (kv_end < 0) kv_end = 0;
      int total_kv_blks = (kv_end + BN - 1) / BN;
      int sp_blks = (total_kv_blks + splits - 1) / splits;
      int blk_start = sp * sp_blks;
      int blk_end = blk_start + sp_blks;
      if (blk_end > total_kv_blks) blk_end = total_kv_blks;
      if (blk_start >= blk_end) continue;

      vf_init_state<ROWS, D_NOPE>(m_prev_ub, l_prev_ub, acc_o_ub);

      int num_blks = blk_end - blk_start;
      {
        int kv_idx0 = blk_start * BN;
        int cur_n0 = kv_end - kv_idx0;
        if (cur_n0 > BN) cur_n0 = BN;
        asc_sync_intra_wait(PIPE_V, 5);
        if (cur_n0 == BN) {
          vf_softmax_full<ROWS, BN, InT>(s_ub, p_ub, m_prev_ub, l_prev_ub, alpha_ub0, scale);
        } else {
          vf_softmax_tail<ROWS, BN, InT>(s_ub, p_ub, m_prev_ub, l_prev_ub, alpha_ub0, cur_n0, scale);
        }
        ascend_nd2nz_scatter<ROWS, BN, ROWS + 1, InT, InT>(p_ub, p_nz_ub);
        asc_sync_notify(PIPE_V, PIPE_MTE3, (event_t)0);
        asc_sync_wait(PIPE_V, PIPE_MTE3, (event_t)0);
        asc_copy_ub2l1((__cbuf__ void*)(p_l1_base + sid * (ROWS * 16)), (__ubuf__ void*)p_nz_ub,
                       BN / 16, ROWS, 1, ROWS);
        asc_sync_intra_arrive(PIPE_MTE3, arrive_p);
      }

      for (int bi = 0; bi < num_blks; ++bi) {
        __ubuf__ float* cur_alpha = (bi & 1) ? alpha_ub1 : alpha_ub0;
        if (bi + 1 < num_blks) {
          int nxt = (bi + 1) & 1;
          __ubuf__ float* nxt_alpha = nxt ? alpha_ub1 : alpha_ub0;
          int kv_idx1 = (blk_start + bi + 1) * BN;
          int cur_n1 = kv_end - kv_idx1;
          if (cur_n1 > BN) cur_n1 = BN;
          asc_sync_intra_wait(PIPE_V, 5);
          if (cur_n1 == BN) {
            vf_softmax_full<ROWS, BN, InT>(s_ub, p_ub, m_prev_ub, l_prev_ub, nxt_alpha, scale);
          } else {
            vf_softmax_tail<ROWS, BN, InT>(s_ub, p_ub, m_prev_ub, l_prev_ub, nxt_alpha, cur_n1, scale);
          }
          ascend_nd2nz_scatter<ROWS, BN, ROWS + 1, InT, InT>(p_ub, p_nz_ub);
          asc_sync_notify(PIPE_V, PIPE_MTE3, (event_t)0);
          asc_sync_wait(PIPE_V, PIPE_MTE3, (event_t)0);
          asc_copy_ub2l1((__cbuf__ void*)(p_l1_base + nxt * P_ELEMS + sid * (ROWS * 16)), (__ubuf__ void*)p_nz_ub,
                         BN / 16, ROWS, 1, ROWS);
          asc_sync_intra_arrive(PIPE_MTE3, arrive_p);
        }

        for (int sl = 0; sl < NUM_SLICES; ++sl) {
          asc_sync_intra_wait(PIPE_V, 3);
          if (sl == NUM_SLICES - 1) vf_acc_rescale<ROWS, LAST_NS, D_NOPE, (NUM_SLICES - 1) * NS>(acc_o_ub, o_ub, cur_alpha);
          else if (sl == 0) vf_acc_rescale<ROWS, NS, D_NOPE, 0>(acc_o_ub, o_ub, cur_alpha);
          else if (sl == 1) vf_acc_rescale<ROWS, NS, D_NOPE, NS>(acc_o_ub, o_ub, cur_alpha);
          else if (sl == 2) vf_acc_rescale<ROWS, NS, D_NOPE, 2 * NS>(acc_o_ub, o_ub, cur_alpha);
          asc_sync_intra_arrive(PIPE_V, arrive_oub);
        }
      }

      if (splits <= 1) {
        vf_finalize_out<ROWS, D_NOPE, InT>(acc_o_ub, l_prev_ub, out_ub);
        asc_sync_notify(PIPE_V, PIPE_MTE3, (event_t)0);
        asc_sync_wait(PIPE_V, PIPE_MTE3, (event_t)0);
        if (my_rows > 0) {
          int out_base = is_bnsd
              ? ((b * N_q + m_off + my_m_start) * S_q + s) * D_NOPE
              : ((b * S_q + s) * N_q + m_off + my_m_start) * D_NOPE;
          if (!is_bnsd || S_q == 1) {
            asc_copy_ub2gm_align((__gm__ uint8_t*)(Out + out_base), (__ubuf__ uint8_t*)out_ub,
                                 1, my_rows * D_NOPE * 2, static_cast<asc_store_l2_cache_mode>(4),
                                 my_rows * D_NOPE * 2, my_rows * D_NOPE * 2);
          } else {
            asc_copy_ub2gm_align((__gm__ uint8_t*)(Out + out_base), (__ubuf__ uint8_t*)out_ub,
                                 my_rows, D_NOPE * 2, static_cast<asc_store_l2_cache_mode>(4),
                                 out_head_stride * 2, D_NOPE * 2);
          }
        }
        asc_sync_notify(PIPE_MTE3, PIPE_V, (event_t)0);
        asc_sync_wait(PIPE_MTE3, PIPE_V, (event_t)0);
        asc_sync_intra_arrive(PIPE_V, arrive_oub);
      } else {
        vf_finalize_partial<ROWS, D_NOPE>(acc_o_ub, m_prev_ub, l_prev_ub);
        asc_sync_notify(PIPE_V, PIPE_S, (event_t)0);
        asc_sync_wait(PIPE_V, PIPE_S, (event_t)0);
        for (int r = 0; r < my_rows; ++r) {
          lse_pack_ub[r] = l_prev_ub[r * 8];
        }
        asc_sync_notify(PIPE_S, PIPE_MTE3, (event_t)0);
        asc_sync_notify(PIPE_V, PIPE_MTE3, (event_t)0);
        asc_sync_wait(PIPE_S, PIPE_MTE3, (event_t)0);
        asc_sync_wait(PIPE_V, PIPE_MTE3, (event_t)0);
        if (my_rows > 0) {
          int flat_row = (bs * N_q + m_off + my_m_start);
          int q_total = total_bs * N_q;
          asc_copy_ub2gm_align((__gm__ uint8_t*)(O_part + (sp * q_total + flat_row) * D_NOPE),
                               (__ubuf__ uint8_t*)acc_o_ub, 1, my_rows * D_NOPE * 4,
                               static_cast<asc_store_l2_cache_mode>(4), my_rows * D_NOPE * 4, my_rows * D_NOPE * 4);
          asc_copy_ub2gm_align((__gm__ uint8_t*)(lse_part + sp * q_total + flat_row),
                               (__ubuf__ uint8_t*)lse_pack_ub, 1, my_rows * 4,
                               static_cast<asc_store_l2_cache_mode>(4), my_rows * 4, my_rows * 4);
        }
        asc_sync_notify(PIPE_MTE3, PIPE_V, (event_t)0);
        asc_sync_wait(PIPE_MTE3, PIPE_V, (event_t)0);
        asc_sync_intra_arrive(PIPE_V, arrive_oub);
      }
    }
  }
}

template <int D_NOPE>
__simd_vf__ inline void vf_combine_zero(__ubuf__ float* acc, int rows) {
  vector_bool m32 = asc_create_mask_b32(PAT_ALL);
  auto v_zero = simd_inst::vdup<float>(0.0f, m32, MODE_ZEROING);
  constexpr int CHUNKS = D_NOPE >> 6;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < CHUNKS; ++c) simd_inst::vsts_norm(v_zero, acc + r * D_NOPE + (c << 6), 0, m32);
  }
}

template <int D_NOPE>
__simd_vf__ inline void vf_combine_acc(__ubuf__ float* o_ub, __ubuf__ float* acc, __ubuf__ float* w_ptr, int rows) {
  vector_bool m32 = asc_create_mask_b32(PAT_ALL);
  constexpr int CHUNKS = D_NOPE >> 6;
  for (int r = 0; r < rows; ++r) {
    auto w_v = simd_inst::vlds_brc_elem<float>(w_ptr + r * 8, 0);
    for (int c = 0; c < CHUNKS; ++c) {
      auto o_v = simd_inst::vlds_norm<float>(o_ub + r * D_NOPE + (c << 6), 0);
      auto a_v = simd_inst::vlds_norm<float>(acc + r * D_NOPE + (c << 6), 0);
      simd_inst::vsts_norm(simd_inst::vadd(a_v, simd_inst::vmul(o_v, w_v, m32, MODE_ZEROING), m32, MODE_ZEROING), acc + r * D_NOPE + (c << 6), 0, m32);
    }
  }
}

template <int D_NOPE, typename OutT>
__simd_vf__ inline void vf_combine_weights(__ubuf__ float* lse_ub, __ubuf__ float* w_ub, __ubuf__ float* inv_ub, int splits, int rows) {
  vector_bool m32 = asc_create_mask_b32(PAT_ALL);
  auto v_zero = simd_inst::vdup<float>(0.0f, m32, MODE_ZEROING);
  auto v_one = simd_inst::vdup<float>(1.0f, m32, MODE_ZEROING);
  for (int r = 0; r < rows; ++r) {
    auto m_glob = simd_inst::vdup<float>(-1e30f, m32, MODE_ZEROING);
    for (int s = 0; s < splits; ++s) {
      auto lv = simd_inst::vlds_brc_elem<float>(lse_ub + (s * 4 + r) * 8, 0);
      m_glob = simd_inst::vmax(m_glob, lv, m32, MODE_ZEROING);
    }
    auto sum_w = v_zero;
    for (int s = 0; s < splits; ++s) {
      auto lv = simd_inst::vlds_brc_elem<float>(lse_ub + (s * 4 + r) * 8, 0);
      auto valid = simd_inst::vcmps_gt(lv, -1e29f, m32);
      auto ws = simd_inst::vsel(simd_inst::vexp(simd_inst::vsub(lv, m_glob, m32, MODE_ZEROING), m32, MODE_ZEROING), v_zero, valid);
      sum_w = simd_inst::vadd(sum_w, ws, m32, MODE_ZEROING);
      simd_inst::vsts_1st(ws, w_ub + (s * 4 + r) * 8, 0, m32);
    }
    auto pos = simd_inst::vcmps_gt(sum_w, 1e-30f, m32);
    auto inv = simd_inst::vsel(simd_inst::vdiv(v_one, simd_inst::vsel(sum_w, v_one, pos), m32, MODE_ZEROING), v_zero, pos);
    simd_inst::vsts_1st(inv, inv_ub + r * 8, 0, m32);
  }
}

template <int D_NOPE, typename OutT>
__simd_vf__ inline void vf_combine_fin(__ubuf__ float* acc, __ubuf__ float* inv_ub, __ubuf__ OutT* out_ub, int rows) {
  vector_bool m32 = asc_create_mask_b32(PAT_ALL);
  vector_bool m16 = asc_create_mask_b16(PAT_ALL);
  constexpr int CHUNKS = D_NOPE >> 6;
  for (int r = 0; r < rows; ++r) {
    auto inv_v = simd_inst::vlds_brc_elem<float>(inv_ub + r * 8, 0);
    for (int c = 0; c < CHUNKS; ++c) {
      auto y = simd_inst::vmul(simd_inst::vlds_norm<float>(acc + r * D_NOPE + (c << 6), 0), inv_v, m32, MODE_ZEROING);
      auto y16 = simd_inst::vcvt<OutT>(y, m16, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
      simd_inst::vsts_pack_b32(y16, out_ub + r * D_NOPE + (c << 6), 0, m32);
    }
  }
}

template <int D_NOPE, typename OutT>
__global__ __vector__ void mla_fast_combine(
    __gm__ float* O_part, __gm__ float* lse_part, __gm__ OutT* Out,
    int q_total, int splits, int B, int S_q, int N_q, int is_bnsd) {
  asc_init();
  constexpr int QT = 4;
  __ubuf__ float* o_ub0 = (__ubuf__ float*)0;
  __ubuf__ float* o_ub1 = (__ubuf__ float*)(8192);
  __ubuf__ float* acc_ub = (__ubuf__ float*)(16384);
  __ubuf__ OutT* out_ub = (__ubuf__ OutT*)(24576);
  __ubuf__ float* lse_ub = (__ubuf__ float*)(32768);
  __ubuf__ float* w_ub = (__ubuf__ float*)(40960);
  __ubuf__ float* inv_ub = (__ubuf__ float*)(49152);
  __ubuf__ float* lse_bulk = (__ubuf__ float*)(53248);

  int cid = (int)block_idx;
  int num_cores = 56;
  int base_q = q_total / num_cores;
  int rem_q = q_total % num_cores;
  int q_start = cid * base_q + (cid < rem_q ? cid : rem_q);
  int q_count = base_q + (cid < rem_q ? 1 : 0);
  if (q_count == 0) return;

  asc_sync_notify(PIPE_V, PIPE_MTE2, (event_t)0);
  asc_sync_notify(PIPE_V, PIPE_MTE2, (event_t)1);
  asc_sync_notify(PIPE_MTE3, PIPE_V, (event_t)0);
  asc_sync_notify(PIPE_V, PIPE_S, (event_t)0);

  int q_aligned = q_start & ~7;
  int q_span = ((q_start + q_count - q_aligned + 7) & ~7);
  for (int s = 0; s < splits; ++s) {
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)(lse_bulk + s * 256), (__gm__ uint8_t*)(lse_part + s * q_total + q_aligned),
                         1, q_span * 4, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), q_span * 4, q_span * 4);
  }
  asc_sync_notify(PIPE_MTE2, PIPE_S, (event_t)0);
  asc_sync_wait(PIPE_MTE2, PIPE_S, (event_t)0);

  for (int base = 0; base < q_count; base += QT) {
    int rows = q_count - base;
    if (rows > QT) rows = QT;
    int qr = q_start + base;
    int q_off = qr - q_aligned;

    asc_sync_wait(PIPE_V, PIPE_S, (event_t)0);
    for (int s = 0; s < splits; ++s) {
      for (int r = 0; r < rows; ++r) {
        lse_ub[(s * 4 + r) * 8] = lse_bulk[s * 256 + q_off + r];
      }
    }
    asc_sync_notify(PIPE_S, PIPE_V, (event_t)0);
    asc_sync_wait(PIPE_S, PIPE_V, (event_t)0);
    vf_combine_weights<D_NOPE, OutT>(lse_ub, w_ub, inv_ub, splits, rows);
    asc_sync_notify(PIPE_V, PIPE_S, (event_t)0);
    vf_combine_zero<D_NOPE>(acc_ub, rows);

    for (int s = 0; s < splits; ++s) {
      int stg = s & 1;
      __ubuf__ float* cur_o = stg ? o_ub1 : o_ub0;
      asc_sync_wait(PIPE_V, PIPE_MTE2, (event_t)stg);
      asc_copy_gm2ub_align((__ubuf__ uint8_t*)cur_o, (__gm__ uint8_t*)(O_part + (s * q_total + qr) * D_NOPE),
                           1, rows * D_NOPE * 4, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), rows * D_NOPE * 4, rows * D_NOPE * 4);
      asc_sync_notify(PIPE_MTE2, PIPE_V, (event_t)stg);
      asc_sync_wait(PIPE_MTE2, PIPE_V, (event_t)stg);
      vf_combine_acc<D_NOPE>(cur_o, acc_ub, w_ub + s * 32, rows);
      asc_sync_notify(PIPE_V, PIPE_MTE2, (event_t)stg);
    }
    asc_sync_wait(PIPE_MTE3, PIPE_V, (event_t)0);
    vf_combine_fin<D_NOPE, OutT>(acc_ub, inv_ub, out_ub, rows);
    asc_sync_notify(PIPE_V, PIPE_MTE3, (event_t)0);
    asc_sync_wait(PIPE_V, PIPE_MTE3, (event_t)0);
    if (!is_bnsd || S_q == 1) {
      asc_copy_ub2gm_align((__gm__ uint8_t*)(Out + qr * D_NOPE), (__ubuf__ uint8_t*)out_ub,
                           1, rows * D_NOPE * 2, static_cast<asc_store_l2_cache_mode>(4), rows * D_NOPE * 2, rows * D_NOPE * 2);
    } else {
      for (int r = 0; r < rows; ++r) {
        int qri = qr + r;
        int h = qri % N_q;
        int bs = qri / N_q;
        int b = bs / S_q;
        int sq = bs % S_q;
        int out_idx = (b * N_q + h) * S_q + sq;
        asc_copy_ub2gm_align((__gm__ uint8_t*)(Out + out_idx * D_NOPE), (__ubuf__ uint8_t*)(out_ub + r * D_NOPE),
                             1, D_NOPE * 2, static_cast<asc_store_l2_cache_mode>(4), D_NOPE * 2, D_NOPE * 2);
      }
    }
    asc_sync_notify(PIPE_MTE3, PIPE_V, (event_t)0);
  }
  asc_sync_wait(PIPE_V, PIPE_MTE2, (event_t)0);
  asc_sync_wait(PIPE_V, PIPE_MTE2, (event_t)1);
  asc_sync_wait(PIPE_MTE3, PIPE_V, (event_t)0);
  asc_sync_wait(PIPE_V, PIPE_S, (event_t)0);
}
"""


def _cann_include_flags() -> list[str]:
    asc_home = os.environ.get("ASCEND_HOME_PATH", "")
    if not asc_home:
        return []
    flags: list[str] = []
    for sub in ("include", "x86_64-linux/include", "aarch64-linux/include", "runtime/include"):
        asc_inc = os.path.join(asc_home, sub)
        flag = f"-I{asc_inc}"
        if os.path.isdir(asc_inc) and flag not in flags:
            flags.append(flag)
    return flags


def _merge_compile_flags(flags: list[str] | tuple[str, ...]) -> list[str] | None:
    merged = [str(f) for f in flags]
    for flag in _cann_include_flags():
        if flag not in merged:
            merged.append(flag)
    return merged or None


def _render_mla_trackb_launcher(
    trackb_src: str,
    *,
    bm: int,
    d_nope: int,
    tl_dtype: str,
) -> str:
    """Append concrete launch_mla_trackb template instantiation to *trackb_src*."""
    c_dtype = "bfloat16_t" if tl_dtype == "bfloat16" else "half"
    return (
        trackb_src
        + f"""
extern "C" int launch_mla_trackb(
    void* stream, void* q_n, void* q_r, void* k_n, void* k_r, void* out, void* o_part, void* lse_part,
    int B, int S_q, int S_kv, int N_q, int splits, int causal, int is_bnsd, float scale) {{
  mla_kv_reuse_c1c1c2<{bm}, 128, {d_nope}, 64, {c_dtype}><<<28, nullptr, stream>>>(
      (__gm__ {c_dtype}*)q_n, (__gm__ {c_dtype}*)q_r, (__gm__ {c_dtype}*)k_n, (__gm__ {c_dtype}*)k_r,
      (__gm__ {c_dtype}*)out, (__gm__ float*)o_part, (__gm__ float*)lse_part,
      B, S_q, S_kv, N_q, splits, causal, is_bnsd, scale);
  if (splits > 1) {{
    mla_fast_combine<{d_nope}, {c_dtype}><<<56, nullptr, stream>>>(
        (__gm__ float*)o_part, (__gm__ float*)lse_part, (__gm__ {c_dtype}*)out,
        B * S_q * N_q, splits, B, S_q, N_q, is_bnsd);
  }}
  return 0;
}}
"""
    )


def _resolve_trackb_bm(
    b_dim: int,
    s_dim: int,
    s_kv: int,
    n_q: int,
    trackb_bm: object = None,
) -> int:
    """Resolve the Track-B block-M tile size for MLA problem dimensions."""
    if (
        isinstance(trackb_bm, int)
        and not isinstance(trackb_bm, bool)
        and trackb_bm in (32, 64, 128)
    ):
        return int(trackb_bm)
    if n_q >= 128:
        if b_dim == 1 and s_dim == 1 and s_kv >= 1024:
            return 32
        if s_dim <= 2 and b_dim <= 32 and s_kv >= 1024:
            return 64
        return 128
    if n_q >= 64:
        return 64
    return 32


def _render_trackb_tu(**dims: object) -> str:
    """Render an instantiated C++20 Track-B translation unit for MLA."""
    raw_src = dims.get("trackb_src")
    trackb_src = str(raw_src) if raw_src is not None else _MLA_TRACKB_SRC
    raw_tb_bm = dims.get("trackb_bm")
    raw_nq = dims.get("n_q", dims.get("N_q"))
    if (
        isinstance(raw_tb_bm, int)
        and not isinstance(raw_tb_bm, bool)
        and raw_tb_bm in (32, 64, 128)
    ):
        bm = int(raw_tb_bm)
    elif raw_nq is not None:
        b_dim = int(dims.get("b_dim", dims.get("B", 1)) or 1)
        s_dim = int(dims.get("s_dim", dims.get("S_q", dims.get("S", 1))) or 1)
        s_kv = int(dims.get("s_kv", dims.get("S_kv", dims.get("K", 128))) or 128)
        bm = _resolve_trackb_bm(b_dim, s_dim, s_kv, int(raw_nq))
    else:
        bm = int(dims.get("bm", 64) or 64)
    d_nope = int(dims.get("d_nope", dims.get("D_nope", 512)) or 512)
    raw_dt = str(dims.get("tl_dtype", dims.get("dtype", "float16")) or "float16").strip()
    tl_dtype = raw_dt if raw_dt in ("float16", "bfloat16") else "float16"
    return _render_mla_trackb_launcher(
        trackb_src,
        bm=bm,
        d_nope=d_nope,
        tl_dtype=tl_dtype,
    )


def _build_mla_trackb_callable(
    n_q: int,
    d_nope: int,
    tl_dtype: str,
    input_layout: str,
    is_causal: bool,
    bm: int,
):
    launcher_src = _render_mla_trackb_launcher(
        _MLA_TRACKB_SRC,
        bm=bm,
        d_nope=d_nope,
        tl_dtype=tl_dtype,
    )
    tl_inc = str(Path(tilelang.__file__).resolve().parent.parent / "src")
    non_inc_flags = () if "__simt_vf__" in launcher_src else tuple(PURE_SIMDVF_UBUF_FLAGS)
    opts = [f"-I{tl_inc}", *_cann_include_flags(), *non_inc_flags]
    trackb_sig = hashlib.sha256(
        (launcher_src + "|" + " ".join(non_inc_flags)).encode("utf-8")
    ).hexdigest()[:16]
    cache_root = os.environ.get("TILELANG_CACHE_DIR", "").strip()
    if cache_root:
        os.makedirs(cache_root, exist_ok=True)
        so_path = os.path.join(cache_root, f"trackb_{trackb_sig}.so")
        if not (os.path.isfile(so_path) and os.path.getsize(so_path) > 0):
            tmp_so = os.path.join(
                cache_root,
                f".trackb_{trackb_sig}.{os.getpid()}.{uuid.uuid4().hex[:8]}.so",
            )
            try:
                compile_ascend(
                    launcher_src,
                    target_format="so",
                    npu_arch="dav-3510",
                    options=opts,
                    path_target=tmp_so,
                )
                os.replace(tmp_so, so_path)
            finally:
                with contextlib.suppress(OSError):
                    if os.path.exists(tmp_so):
                        os.remove(tmp_so)
    else:
        so_dir = tempfile.mkdtemp(prefix="mla_trackb_")
        so_path = os.path.join(so_dir, "mla_trackb.so")
        compile_ascend(
            launcher_src,
            target_format="so",
            npu_arch="dav-3510",
            options=opts,
            path_target=so_path,
        )
    lib = _TRACKB_SO_CACHE.get(so_path)
    if lib is None:
        lib = ctypes.CDLL(so_path)
        _TRACKB_SO_CACHE[so_path] = lib
    fn = lib.launch_mla_trackb
    fn.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 7 + [ctypes.c_float]
    is_bnsd = 1 if input_layout == "BNSD" else 0
    causal_flag = 1 if is_causal else 0

    def _run(q_n_flat, q_r_flat, k_n_contig, k_r_contig, _v_contig, scale):
        b_dim = int(k_n_contig.shape[0])
        s_kv = int(k_n_contig.shape[2] if is_bnsd else k_n_contig.shape[1])
        q_rows_total = int(q_n_flat.shape[0])
        if q_rows_total == 0 or s_kv <= 0:
            return torch.zeros_like(q_n_flat)
        s_dim = q_rows_total // (b_dim * n_q)
        out_flat = torch.empty_like(q_n_flat)
        splits = 1
        if s_dim <= 2 and s_kv >= 512:
            if b_dim == 1 and s_dim == 1 and s_kv >= 1024:
                splits = min(7, max(1, (s_kv + 127) // 128))
            elif b_dim <= 16 and s_kv >= 1024:
                splits = 4 if (s_dim == 1) else 2
            elif b_dim == 60 and s_dim == 1 and n_q == 64:
                splits = 2
            elif b_dim * s_dim <= 8 and s_kv >= 512:
                splits = min(4, max(1, (s_kv + 127) // 128))
        if splits > 1:
            o_partial, lse_partial = _get_split_kv_workspace(
                q_n_flat.device, splits, q_rows_total, d_nope
            )
            op_ptr, lp_ptr = o_partial.data_ptr(), lse_partial.data_ptr()
        else:
            op_ptr, lp_ptr = out_flat.data_ptr(), out_flat.data_ptr()
        stream_ptr = torch.npu.current_stream().npu_stream
        fn(
            stream_ptr,
            q_n_flat.data_ptr(),
            q_r_flat.data_ptr(),
            k_n_contig.data_ptr(),
            k_r_contig.data_ptr(),
            out_flat.data_ptr(),
            op_ptr,
            lp_ptr,
            b_dim,
            s_dim,
            s_kv,
            n_q,
            splits,
            causal_flag,
            is_bnsd,
            float(scale),
        )
        return out_flat

    _run._so_lib = lib
    return _run


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
        if extra_bk.get("_trackb"):
            compiled = _build_mla_trackb_callable(
                n_q=n_q,
                d_nope=d_nope,
                tl_dtype=tl_dtype,
                input_layout=input_layout,
                is_causal=bool(is_causal),
                bm=int(bm),
            )
            _KERNEL_CACHE[key] = compiled
            return compiled
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
        flags_arg = _merge_compile_flags(flags_tuple)
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
        flags_arg = _merge_compile_flags(flags_tuple)
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

    default_use_trackb = (
        schedule_mode == "tuned"
        and num_kv_splits_override is None
        and n_kv == 1
        and d_nope in (448, 512)
        and d_rope == 64
        and tl_dtype in ("float16", "bfloat16")
        and inputLayout in ("BSND", "BNSD")
        and getattr(q_nope, "is_npu", False)
    )

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
    extra_builder_kwargs = dict(raw_bk) if isinstance(raw_bk, dict) else {}
    if "bn" in cfg:
        extra_builder_kwargs.setdefault("bn", int(cfg["bn"]))
    if "num_stages" in cfg:
        extra_builder_kwargs.setdefault("num_stages", int(cfg["num_stages"]))
    raw_ck = cfg.get("combine_kwargs")
    extra_combine_kwargs = dict(raw_ck) if isinstance(raw_ck, dict) else None

    raw_use_trackb = cfg.get("use_trackb")
    if isinstance(raw_use_trackb, bool):
        use_trackb = (
            raw_use_trackb
            and num_kv_splits_override is None
            and bool(getattr(q_nope, "is_npu", False))
        )
    else:
        use_trackb = default_use_trackb

    if use_trackb:
        bm = _resolve_trackb_bm(b_dim, s_dim, s_kv, n_q, cfg.get("trackb_bm"))
        num_kv_splits = 1
        merged_bk = dict(extra_builder_kwargs or {})
        merged_bk["_trackb"] = True
        extra_builder_kwargs = merged_bk

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
            extra_builder_kwargs=extra_builder_kwargs or None,
        )
        if use_trackb:
            q_n_in, q_r_in = q_n_flat, q_r_flat
        elif inputLayout == "BNSD" or n_kv == 1:
            q_rows_group = s_dim * (n_q // n_kv)
            q_n_in = q_nope.contiguous().view(b_dim, n_kv, q_rows_group, d_nope)
            q_r_in = q_rope.contiguous().view(b_dim, n_kv, q_rows_group, d_rope)
        else:
            q_n_in = q_nope.contiguous().view(b_dim, s_dim, n_kv, n_q // n_kv, d_nope)
            q_r_in = q_rope.contiguous().view(b_dim, s_dim, n_kv, n_q // n_kv, d_rope)
        out_flat = kernel(q_n_in, q_r_in, k_n_contig, k_r_contig, v_contig, scale)
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
            extra_builder_kwargs=extra_builder_kwargs or None,
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
