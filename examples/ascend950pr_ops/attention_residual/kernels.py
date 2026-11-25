"""TileLang Ascend950PR SimdVF kernel implementation for AttentionResidual."""

import contextlib
import math
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

import tilelang
import tilelang.ascend.language as T
import torch

from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS

_KERNEL_CACHE: dict[tuple[object, ...], object] = {}
_DISPATCH_CACHE: dict[tuple[object, ...], dict[str, object]] = {}


def _callable_cache_key(fn: object) -> object:
    try:
        hash(fn)
        return fn
    except TypeError:
        return id(fn)


def _cann_include_flags() -> list[str]:
    flags: list[str] = []
    seen: set[str] = set()
    for env_key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        raw = os.environ.get(env_key, "").strip()
        if not raw:
            continue
        root = Path(raw).expanduser()
        for sub in ("include", "x86_64-linux/include", "aarch64-linux/include", "runtime/include"):
            inc = root / sub
            if inc.is_dir():
                flag = f"-I{inc}"
                if flag not in seen:
                    seen.add(flag)
                    flags.append(flag)
    return flags


def _ascend_target_ctx():
    try:
        from tilelang import tvm
        from tilelang.backend.module import create_backend_context

        if tvm.target.Target.current(allow_none=True) is None:
            return create_backend_context("ascend").target
    except Exception:
        pass
    return contextlib.nullcontext()


def _build_attention_residual_prim_func(
    n_depth: int,
    d_dim: int,
    tl_dtype: str,
    tile_m: int,
    schedule_mode: str,
):
    num_cores = 56
    full_chunks = d_dim // 64
    rem = d_dim % 64
    d_alloc = ((d_dim + 63) // 64) * 64
    num_stages = 2 if (schedule_mode == "tuned" and n_depth >= 2) else 1
    pipe_ann = {"enable_offset": True} if schedule_mode == "tuned" else None
    inv_sqrt_d = 1.0 / math.sqrt(d_dim)
    inv_d = 1.0 / float(d_dim)

    m_rows = T.dynamic("m_rows")
    v_rows = T.dynamic("v_rows")

    @T.macro
    def _simd_init_row(m_ub, l_ub, acc_ub, slot):
        m32 = T.simd.pset(32, "PAT_ALL")
        v_init_m = T.simd.vdup(T.float32(-1.0e30), "float32", mask=m32)
        v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
        T.simd.vsts(m_ub[slot, 0], v_init_m, mask=m32, dist="ONEPT_B32")
        T.simd.vsts(l_ub[slot, 0], v_zero, mask=m32, dist="ONEPT_B32")
        for c in T.serial(full_chunks):
            T.simd.vsts(acc_ub[slot, c * 64], v_zero, mask=m32, dist="NORM_B32")
        if rem > 0:
            T.simd.vsts(
                acc_ub[slot, full_chunks * 64], v_zero, mask=m32, dist="NORM_B32"
            )

    @T.macro
    def _simd_step_row(v_ub, kw_ub, m_ub, l_ub, acc_ub, eps_val, slot):
        m32 = T.simd.pset(32, "PAT_ALL")
        m16 = T.simd.pset(16, "PAT_ALL")
        v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
        v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
        sq_acc = T.simd.alloc_local((1,), "float32")
        dot_acc = T.simd.alloc_local((1,), "float32")
        sq_acc[0] = v_zero
        dot_acc[0] = v_zero
        for c in T.serial(full_chunks):
            off = c * 64
            vf = T.simd.vcvt(
                T.simd.vld(v_ub[slot, off], dist="US_B16"),
                "float32",
                mask=m16,
                part=0,
            )
            kwf = T.simd.vld(kw_ub[off], dist="NORM_B32")
            T.simd.vmula(sq_acc[0], vf, vf, mask=m32)
            T.simd.vmula(dot_acc[0], vf, kwf, mask=m32)
        if rem > 0:
            off = full_chunks * 64
            m32_tail = T.simd.update_mask(rem, width=32)
            m16_tail_x2 = T.simd.update_mask(rem * 2, width=16)
            vf_t = T.simd.vcvt(
                T.simd.vld(v_ub[slot, off], dist="US_B16"),
                "float32",
                mask=m16_tail_x2,
                part=0,
            )
            kwf_t = T.simd.vld(kw_ub[off], dist="NORM_B32")
            T.simd.vmula(
                sq_acc[0], vf_t, vf_t, mask=m32_tail, mode="MODE_MERGING"
            )
            T.simd.vmula(
                dot_acc[0], vf_t, kwf_t, mask=m32_tail, mode="MODE_MERGING"
            )

        sum_sq = T.simd.vdupv(
            T.simd.vcadd(sq_acc[0], mask=m32), mask=m32, pos="POS_LOWEST"
        )
        raw_dot = T.simd.vdupv(
            T.simd.vcadd(dot_acc[0], mask=m32), mask=m32, pos="POS_LOWEST"
        )
        mean_sq = T.simd.vmuls(sum_sq, T.float32(inv_d), mask=m32)
        mean_eps = T.simd.vadds(mean_sq, eps_val, mask=m32)
        rstd = T.simd.vdiv(v_one, T.simd.vsqrt(mean_eps, mask=m32), mask=m32)
        s_vec = T.simd.vmuls(
            T.simd.vmul(raw_dot, rstd, mask=m32),
            T.float32(inv_sqrt_d),
            mask=m32,
        )

        m_old = T.simd.vld(m_ub[slot, 0], dist="BRC_B32")
        l_old = T.simd.vld(l_ub[slot, 0], dist="BRC_B32")
        m_new = T.simd.vmax(m_old, s_vec, mask=m32)
        alpha_v = T.simd.vexp(T.simd.vsub(m_old, m_new, mask=m32), mask=m32)
        p_v = T.simd.vexp(T.simd.vsub(s_vec, m_new, mask=m32), mask=m32)
        l_new = T.simd.vadd(T.simd.vmul(l_old, alpha_v, mask=m32), p_v, mask=m32)
        T.simd.vsts(m_ub[slot, 0], m_new, mask=m32, dist="ONEPT_B32")
        T.simd.vsts(l_ub[slot, 0], l_new, mask=m32, dist="ONEPT_B32")

        for c in T.serial(full_chunks):
            off = c * 64
            vf = T.simd.vcvt(
                T.simd.vld(v_ub[slot, off], dist="US_B16"),
                "float32",
                mask=m16,
                part=0,
            )
            acc_old = T.simd.vld(acc_ub[slot, off], dist="NORM_B32")
            acc_new = T.simd.vadd(
                T.simd.vmul(acc_old, alpha_v, mask=m32),
                T.simd.vmul(vf, p_v, mask=m32),
                mask=m32,
            )
            T.simd.vsts(acc_ub[slot, off], acc_new, mask=m32, dist="NORM_B32")
        if rem > 0:
            off = full_chunks * 64
            m32_tail = T.simd.update_mask(rem, width=32)
            m16_tail_x2 = T.simd.update_mask(rem * 2, width=16)
            vf_t = T.simd.vcvt(
                T.simd.vld(v_ub[slot, off], dist="US_B16"),
                "float32",
                mask=m16_tail_x2,
                part=0,
            )
            acc_old_t = T.simd.vld(acc_ub[slot, off], dist="NORM_B32")
            acc_new_t = T.simd.vadd(
                T.simd.vmul(acc_old_t, alpha_v, mask=m32_tail),
                T.simd.vmul(vf_t, p_v, mask=m32_tail),
                mask=m32_tail,
            )
            T.simd.vsts(
                acc_ub[slot, off], acc_new_t, mask=m32_tail, dist="NORM_B32"
            )

    @T.macro
    def _simd_blend_row(x_ub, acc_ub, l_ub, alpha_val, slot):
        m32 = T.simd.pset(32, "PAT_ALL")
        m16 = T.simd.pset(16, "PAT_ALL")
        v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
        l_tot = T.simd.vld(l_ub[slot, 0], dist="BRC_B32")
        inv_l = T.simd.vdiv(v_one, l_tot, mask=m32)
        scale_agg = T.simd.vmuls(inv_l, alpha_val, mask=m32)
        one_minus_alpha = T.float32(1.0) - alpha_val
        for c in T.serial(full_chunks):
            off = c * 64
            xf = T.simd.vcvt(
                T.simd.vld(x_ub[slot, off], dist="US_B16"),
                "float32",
                mask=m16,
                part=0,
            )
            accf = T.simd.vld(acc_ub[slot, off], dist="NORM_B32")
            yf = T.simd.vadd(
                T.simd.vmuls(xf, one_minus_alpha, mask=m32),
                T.simd.vmul(accf, scale_agg, mask=m32),
                mask=m32,
            )
            y16 = T.simd.vcvt(yf, tl_dtype, mask=m16, part=0)
            T.simd.vsts(
                x_ub[slot, off], y16, mask=m32, dist="PK_B32", extent=64
            )
        if rem > 0:
            off = full_chunks * 64
            m32_tail = T.simd.update_mask(rem, width=32)
            m16_tail_x2 = T.simd.update_mask(rem * 2, width=16)
            xf_t = T.simd.vcvt(
                T.simd.vld(x_ub[slot, off], dist="US_B16"),
                "float32",
                mask=m16_tail_x2,
                part=0,
            )
            accf_t = T.simd.vld(acc_ub[slot, off], dist="NORM_B32")
            yf_t = T.simd.vadd(
                T.simd.vmuls(xf_t, one_minus_alpha, mask=m32_tail),
                T.simd.vmul(accf_t, scale_agg, mask=m32_tail),
                mask=m32_tail,
            )
            y16_t = T.simd.vcvt(yf_t, tl_dtype, mask=m16_tail_x2, part=0)
            T.simd.vsts(
                x_ub[slot, off], y16_t, mask=m32_tail, dist="PK_B32", extent=64
            )

    @T.prim_func
    def main(
        V: T.Tensor((v_rows, d_dim), tl_dtype),
        X: T.Tensor((m_rows, d_dim), tl_dtype),
        W: T.Tensor((d_dim,), tl_dtype),
        G: T.Tensor((d_dim,), tl_dtype),
        Y: T.Tensor((m_rows, d_dim), tl_dtype),
        alpha_val: T.float32,
        eps_val: T.float32,
    ):
        with T.Kernel(num_cores) as cid:
            w_ub = T.alloc_shared((d_alloc,), tl_dtype)
            g_ub = T.alloc_shared((d_alloc,), tl_dtype)
            kw_ub = T.alloc_shared((d_alloc,), "float32")
            v_ub = T.alloc_shared((tile_m, d_alloc), tl_dtype)
            x_ub = T.alloc_shared((tile_m, d_alloc), tl_dtype)
            acc_ub = T.alloc_shared((tile_m, d_alloc), "float32")
            m_ub = T.alloc_shared((tile_m, 8), "float32")
            l_ub = T.alloc_shared((tile_m, 8), "float32")
            if tile_m > 1:
                v_tail_ub = T.alloc_shared((1, d_alloc), tl_dtype)
                T.annotate_buffer_versions(
                    {
                        v_ub: num_stages,
                        v_tail_ub: num_stages,
                        acc_ub: 1,
                        m_ub: 1,
                        l_ub: 1,
                        kw_ub: 1,
                        w_ub: 1,
                        g_ub: 1,
                        x_ub: 1,
                    }
                )
                T.annotate_unlimit_memory("shared")
            else:
                T.annotate_buffer_versions(
                    {
                        v_ub: num_stages,
                        acc_ub: 1,
                        m_ub: 1,
                        l_ub: 1,
                        kw_ub: 1,
                        w_ub: 1,
                        g_ub: 1,
                        x_ub: 1,
                    }
                )

            base_rows = m_rows // num_cores
            rem_rows = m_rows % num_cores
            m_start = cid * base_rows + T.min(cid, rem_rows)
            m_count = base_rows + T.if_then_else(cid < rem_rows, 1, 0)
            if m_count > 0:
                T.copy(W[:d_dim], w_ub[:d_dim])
                T.copy(G[:d_dim], g_ub[:d_dim])
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    m16 = T.simd.pset(16, "PAT_ALL")
                    for c in T.serial(full_chunks):
                        off = c * 64
                        wf = T.simd.vcvt(
                            T.simd.vld(w_ub[off], dist="US_B16"),
                            "float32",
                            mask=m16,
                            part=0,
                        )
                        gf = T.simd.vcvt(
                            T.simd.vld(g_ub[off], dist="US_B16"),
                            "float32",
                            mask=m16,
                            part=0,
                        )
                        kwf = T.simd.vmul(wf, gf, mask=m32)
                        T.simd.vsts(kw_ub[off], kwf, mask=m32, dist="NORM_B32")
                    if rem > 0:
                        off = full_chunks * 64
                        m16_tail_x2 = T.simd.update_mask(rem * 2, width=16)
                        wf_t = T.simd.vcvt(
                            T.simd.vld(w_ub[off], dist="US_B16"),
                            "float32",
                            mask=m16_tail_x2,
                            part=0,
                        )
                        gf_t = T.simd.vcvt(
                            T.simd.vld(g_ub[off], dist="US_B16"),
                            "float32",
                            mask=m16_tail_x2,
                            part=0,
                        )
                        kwf_t = T.simd.vmul(wf_t, gf_t, mask=m32)
                        T.simd.vsts(kw_ub[off], kwf_t, mask=m32, dist="NORM_B32")

                if tile_m > 1:
                    full_blocks = m_count // tile_m
                    for mb in T.serial(full_blocks):
                        m0 = m_start + mb * tile_m
                        with T.SimdVF():
                            for tm in T.serial(tile_m):
                                _simd_init_row(m_ub, l_ub, acc_ub, tm)
                        for n in T.Pipelined(
                            n_depth, num_stages=num_stages, annotations=pipe_ann
                        ):
                            T.copy(
                                V[n * m_rows + m0 : n * m_rows + m0 + tile_m, :d_dim],
                                v_ub[:, :d_dim],
                            )
                            with T.SimdVF():
                                for tm in T.serial(tile_m):
                                    _simd_step_row(
                                        v_ub, kw_ub, m_ub, l_ub, acc_ub, eps_val, tm
                                    )
                        T.copy(X[m0 : m0 + tile_m, :d_dim], x_ub[:, :d_dim])
                        with T.SimdVF():
                            for tm in T.serial(tile_m):
                                _simd_blend_row(x_ub, acc_ub, l_ub, alpha_val, tm)
                        T.copy(x_ub[:, :d_dim], Y[m0 : m0 + tile_m, :d_dim])
                        T.ascend_pipe_barrier("PIPE_ALL")

                    tail_rows = m_count % tile_m
                    tail_start = m_start + full_blocks * tile_m
                    for rr in T.serial(tail_rows):
                        m = tail_start + rr
                        with T.SimdVF():
                            _simd_init_row(m_ub, l_ub, acc_ub, 0)
                        for n in T.Pipelined(
                            n_depth, num_stages=num_stages, annotations=pipe_ann
                        ):
                            row_off = n * m_rows + m
                            T.copy(V[row_off, :d_dim], v_tail_ub[0, :d_dim])
                            with T.SimdVF():
                                _simd_step_row(
                                    v_tail_ub, kw_ub, m_ub, l_ub, acc_ub, eps_val, 0
                                )
                        T.copy(X[m, :d_dim], x_ub[0, :d_dim])
                        with T.SimdVF():
                            _simd_blend_row(x_ub, acc_ub, l_ub, alpha_val, 0)
                        T.copy(x_ub[0, :d_dim], Y[m, :d_dim])
                        T.ascend_pipe_barrier("PIPE_ALL")
                else:
                    for local_m in T.serial(m_count):
                        m = m_start + local_m
                        with T.SimdVF():
                            _simd_init_row(m_ub, l_ub, acc_ub, 0)
                        for n in T.Pipelined(
                            n_depth, num_stages=num_stages, annotations=pipe_ann
                        ):
                            row_off = n * m_rows + m
                            T.copy(V[row_off, :d_dim], v_ub[0, :d_dim])
                            with T.SimdVF():
                                _simd_step_row(
                                    v_ub, kw_ub, m_ub, l_ub, acc_ub, eps_val, 0
                                )
                        T.copy(X[m, :d_dim], x_ub[0, :d_dim])
                        with T.SimdVF():
                            _simd_blend_row(x_ub, acc_ub, l_ub, alpha_val, 0)
                        T.copy(x_ub[0, :d_dim], Y[m, :d_dim])
                        T.ascend_pipe_barrier("PIPE_ALL")

    return main


def _choose_ar_tile_m(d_dim: int, schedule_mode: str) -> int:
    if schedule_mode != "tuned" or (d_dim % 64) != 0:
        return 1
    d_alloc = ((d_dim + 63) // 64) * 64
    ub_budget = 256 * 1024 - 1024
    max_rows = (ub_budget - 12 * d_alloc) // (10 * d_alloc + 64)
    return max(1, min(8, max_rows))


_VF_STACK_DISABLE_FLAG = "--cce-disable-vf-stack-reserved-ubuf"
_VF_STACK_RESERVE_UB_THRESHOLD = 224 * 1024


def _estimate_ar_static_ub_bytes(
    n_depth: int,
    d_dim: int,
    tile_m: int,
    schedule_mode: str,
    tl_dtype: str = "bfloat16",
    *,
    num_stages: int = 0,
    d_subtile: int = 0,
) -> int:
    elem_bytes = 4 if tl_dtype == "float32" else 2
    d_alloc = ((int(d_dim) + 63) // 64) * 64
    eff_d_sub = (
        min(d_alloc, ((int(d_subtile) + 63) // 64) * 64)
        if int(d_subtile) > 0
        else d_alloc
    )
    rows = max(1, int(tile_m))
    stages = (
        max(1, min(2, int(num_stages)))
        if int(num_stages) > 0
        else (2 if (schedule_mode == "tuned" and int(n_depth) >= 2) else 1)
    )
    fixed_bytes = (2 * elem_bytes + 4) * d_alloc
    if rows > 1 and eff_d_sub >= d_alloc:
        fixed_bytes += stages * d_alloc * elem_bytes
    per_row_bytes = (
        stages * elem_bytes * d_alloc + (elem_bytes + 4) * eff_d_sub + 64
    )
    return fixed_bytes + rows * per_row_bytes


def _choose_ar_small_decode_d_subtile(
    d_dim: int,
    m_rows: int,
    schedule_mode: str,
) -> int:
    """Choose a 256-aligned column subtile so 2D (row, d_blk) saturates 56 AIV cores."""
    if schedule_mode != "tuned" or d_dim < 1024 or m_rows >= 56:
        return int(d_dim)
    eff_m = max(1, int(m_rows))
    min_d_blks = (56 + eff_m - 1) // eff_m
    raw_subtile = int(d_dim) // min_d_blks
    d_subtile = max(256, (raw_subtile // 256) * 256)
    return min(int(d_dim), d_subtile)


def _ar_default_flags(schedule_mode: str, *, static_ub_bytes: int = 0) -> list[str]:
    if schedule_mode != "tuned":
        return []
    if static_ub_bytes > _VF_STACK_RESERVE_UB_THRESHOLD:
        return list(PURE_SIMDVF_UBUF_FLAGS)
    return [flag for flag in PURE_SIMDVF_UBUF_FLAGS if flag != _VF_STACK_DISABLE_FLAG]


@T.macro
def _ar_small_decode_init_row(m_ub, l_ub, acc_ub, slot, sub_chunks):
    m32 = T.simd.pset(32, "PAT_ALL")
    v_init_m = T.simd.vdup(T.float32(-1.0e30), "float32", mask=m32)
    v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
    T.simd.vsts(m_ub[slot, 0], v_init_m, mask=m32, dist="ONEPT_B32")
    T.simd.vsts(l_ub[slot, 0], v_zero, mask=m32, dist="ONEPT_B32")
    for c in T.serial(sub_chunks):
        T.simd.vsts(acc_ub[slot, c * 64], v_zero, mask=m32, dist="NORM_B32")


@T.macro
def _ar_small_decode_step_row(
    v_ub,
    kw_ub,
    m_ub,
    l_ub,
    acc_ub,
    eps_val,
    inv_d,
    inv_sqrt_d,
    slot,
    full_chunks,
    d_start,
    sub_chunks,
):
    m32 = T.simd.pset(32, "PAT_ALL")
    m16 = T.simd.pset(16, "PAT_ALL")
    v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
    v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
    sq_acc = T.simd.alloc_local((1,), "float32")
    dot_acc = T.simd.alloc_local((1,), "float32")
    sq_acc[0] = v_zero
    dot_acc[0] = v_zero
    for c in T.serial(full_chunks):
        off = c * 64
        vf = T.simd.vcvt(
            T.simd.vld(v_ub[slot, off], dist="US_B16"),
            "float32",
            mask=m16,
            part=0,
        )
        kwf = T.simd.vld(kw_ub[off], dist="NORM_B32")
        T.simd.vmula(sq_acc[0], vf, vf, mask=m32)
        T.simd.vmula(dot_acc[0], vf, kwf, mask=m32)

    sum_sq = T.simd.vdupv(
        T.simd.vcadd(sq_acc[0], mask=m32), mask=m32, pos="POS_LOWEST"
    )
    raw_dot = T.simd.vdupv(
        T.simd.vcadd(dot_acc[0], mask=m32), mask=m32, pos="POS_LOWEST"
    )
    mean_sq = T.simd.vmuls(sum_sq, T.float32(inv_d), mask=m32)
    mean_eps = T.simd.vadds(mean_sq, eps_val, mask=m32)
    rstd = T.simd.vdiv(v_one, T.simd.vsqrt(mean_eps, mask=m32), mask=m32)
    s_vec = T.simd.vmuls(
        T.simd.vmul(raw_dot, rstd, mask=m32),
        T.float32(inv_sqrt_d),
        mask=m32,
    )

    m_old = T.simd.vld(m_ub[slot, 0], dist="BRC_B32")
    l_old = T.simd.vld(l_ub[slot, 0], dist="BRC_B32")
    m_new = T.simd.vmax(m_old, s_vec, mask=m32)
    alpha_v = T.simd.vexp(T.simd.vsub(m_old, m_new, mask=m32), mask=m32)
    p_v = T.simd.vexp(T.simd.vsub(s_vec, m_new, mask=m32), mask=m32)
    l_new = T.simd.vadd(T.simd.vmul(l_old, alpha_v, mask=m32), p_v, mask=m32)
    T.simd.vsts(m_ub[slot, 0], m_new, mask=m32, dist="ONEPT_B32")
    T.simd.vsts(l_ub[slot, 0], l_new, mask=m32, dist="ONEPT_B32")

    for c in T.serial(sub_chunks):
        off_v = d_start + c * 64
        off_acc = c * 64
        vf = T.simd.vcvt(
            T.simd.vld(v_ub[slot, off_v], dist="US_B16"),
            "float32",
            mask=m16,
            part=0,
        )
        acc_old = T.simd.vld(acc_ub[slot, off_acc], dist="NORM_B32")
        acc_new = T.simd.vadd(
            T.simd.vmul(acc_old, alpha_v, mask=m32),
            T.simd.vmul(vf, p_v, mask=m32),
            mask=m32,
        )
        T.simd.vsts(acc_ub[slot, off_acc], acc_new, mask=m32, dist="NORM_B32")


@T.macro
def _ar_small_decode_blend_row(
    x_ub, acc_ub, l_ub, alpha_val, tl_dtype, slot, sub_chunks
):
    m32 = T.simd.pset(32, "PAT_ALL")
    m16 = T.simd.pset(16, "PAT_ALL")
    v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
    l_tot = T.simd.vld(l_ub[slot, 0], dist="BRC_B32")
    inv_l = T.simd.vdiv(v_one, l_tot, mask=m32)
    scale_agg = T.simd.vmuls(inv_l, alpha_val, mask=m32)
    one_minus_alpha = T.float32(1.0) - alpha_val
    for c in T.serial(sub_chunks):
        off = c * 64
        xf = T.simd.vcvt(
            T.simd.vld(x_ub[slot, off], dist="US_B16"),
            "float32",
            mask=m16,
            part=0,
        )
        accf = T.simd.vld(acc_ub[slot, off], dist="NORM_B32")
        yf = T.simd.vadd(
            T.simd.vmuls(xf, one_minus_alpha, mask=m32),
            T.simd.vmul(accf, scale_agg, mask=m32),
            mask=m32,
        )
        y16 = T.simd.vcvt(yf, tl_dtype, mask=m16, part=0)
        T.simd.vsts(x_ub[slot, off], y16, mask=m32, dist="PK_B32", extent=64)


def _select_ar_small_decode_config(
    n_depth: int = 2,
    d_dim: int = 4096,
    tl_dtype: str = "bfloat16",
    schedule_mode: str = "tuned",
    m_rows: int = 1,
    tile_m_override: int = 0,
    num_stages_override: int = 0,
) -> dict[str, object]:
    """Select 2D (row, d_blk) small-decode schedule saturating 56 AIV cores."""
    tile_m = max(1, int(tile_m_override)) if int(tile_m_override) > 0 else 1
    num_stages = (
        max(1, min(2, int(num_stages_override)))
        if int(num_stages_override) > 0
        else (2 if (schedule_mode == "tuned" and int(n_depth) >= 2) else 1)
    )
    d_subtile = _choose_ar_small_decode_d_subtile(int(d_dim), int(m_rows), schedule_mode)
    static_ub_bytes = _estimate_ar_static_ub_bytes(
        n_depth,
        d_dim,
        tile_m,
        schedule_mode,
        tl_dtype,
        num_stages=num_stages,
        d_subtile=d_subtile,
    )
    return {
        "tile_m": tile_m,
        "num_stages": num_stages,
        "d_subtile": d_subtile,
        "static_ub_bytes": static_ub_bytes,
        "pass_configs": {},
        "compile_flags": _ar_default_flags(schedule_mode, static_ub_bytes=static_ub_bytes),
    }


def _build_ar_small_decode_func(
    n_depth: int = 2,
    d_dim: int = 4096,
    tl_dtype: str = "bfloat16",
    tile_m: int = 1,
    schedule_mode: str = "tuned",
    num_stages: int = 0,
    d_subtile: int = 0,
    **_extra_kwargs,
):
    if (d_dim % 64) != 0:
        raise ValueError(f"small_decode regime requires d_dim % 64 == 0, got {d_dim}")
    num_cores = 56
    full_chunks = d_dim // 64
    tm_ov = int(_extra_kwargs.get("tile_m_override", 0) or 0)
    ns_ov = int(_extra_kwargs.get("num_stages_override", 0) or 0)
    rows = max(1, tm_ov if tm_ov > 0 else int(tile_m))
    raw_stages = ns_ov if ns_ov > 0 else int(num_stages)
    eff_stages = (
        max(1, min(2, raw_stages))
        if raw_stages > 0
        else (2 if (schedule_mode == "tuned" and n_depth >= 2) else 1)
    )
    if int(d_subtile) > 0:
        eff_d_subtile = min(d_dim, int(d_subtile))
    else:
        eff_d_subtile = 256 if (schedule_mode == "tuned" and d_dim >= 1024) else d_dim
    if (eff_d_subtile % 64) != 0:
        raise ValueError(
            f"small_decode d_subtile must be divisible by 64, got {eff_d_subtile}"
        )
    num_d_blks = (d_dim + eff_d_subtile - 1) // eff_d_subtile
    tail_d = d_dim - (num_d_blks - 1) * eff_d_subtile
    sub_chunks = eff_d_subtile // 64
    tail_chunks = tail_d // 64
    pipe_ann = {"enable_offset": True} if schedule_mode == "tuned" else None
    inv_sqrt_d = 1.0 / math.sqrt(d_dim)
    inv_d = 1.0 / float(d_dim)

    m_rows = T.dynamic("m_rows")
    v_rows = T.dynamic("v_rows")

    @T.macro
    def _ar_small_decode_run_task(
        V,
        X,
        Y,
        v_ub,
        kw_ub,
        x_ub,
        acc_ub,
        m_ub,
        l_ub,
        alpha_val,
        eps_val,
        row_idx,
        d_start,
        cur_d,
        cur_chunks,
    ):
        with T.SimdVF():
            _ar_small_decode_init_row(m_ub, l_ub, acc_ub, 0, cur_chunks)
        for n in T.Pipelined(n_depth, num_stages=eff_stages, annotations=pipe_ann):
            row_off = n * m_rows + row_idx
            T.copy(V[row_off, :d_dim], v_ub[0, :d_dim])
            with T.SimdVF():
                _ar_small_decode_step_row(
                    v_ub,
                    kw_ub,
                    m_ub,
                    l_ub,
                    acc_ub,
                    eps_val,
                    inv_d,
                    inv_sqrt_d,
                    0,
                    full_chunks,
                    d_start,
                    cur_chunks,
                )
        T.copy(X[row_idx, d_start : d_start + cur_d], x_ub[0, :cur_d])
        with T.SimdVF():
            _ar_small_decode_blend_row(
                x_ub,
                acc_ub,
                l_ub,
                alpha_val,
                tl_dtype,
                0,
                cur_chunks,
            )
        T.copy(x_ub[0, :cur_d], Y[row_idx, d_start : d_start + cur_d])
        T.ascend_pipe_barrier("PIPE_ALL")

    @T.prim_func
    def main(
        V: T.Tensor((v_rows, d_dim), tl_dtype),
        X: T.Tensor((m_rows, d_dim), tl_dtype),
        W: T.Tensor((d_dim,), tl_dtype),
        G: T.Tensor((d_dim,), tl_dtype),
        Y: T.Tensor((m_rows, d_dim), tl_dtype),
        alpha_val: T.float32,
        eps_val: T.float32,
    ):
        with T.Kernel(num_cores) as cid:
            w_ub = T.alloc_shared((d_dim,), tl_dtype)
            g_ub = T.alloc_shared((d_dim,), tl_dtype)
            kw_ub = T.alloc_shared((d_dim,), "float32")
            v_ub = T.alloc_shared((rows, d_dim), tl_dtype)
            x_ub = T.alloc_shared((rows, eff_d_subtile), tl_dtype)
            acc_ub = T.alloc_shared((rows, eff_d_subtile), "float32")
            m_ub = T.alloc_shared((rows, 8), "float32")
            l_ub = T.alloc_shared((rows, 8), "float32")
            if num_d_blks == 1 and rows > 1:
                v_tail_ub = T.alloc_shared((1, d_dim), tl_dtype)
                T.annotate_buffer_versions(
                    {
                        v_ub: eff_stages,
                        v_tail_ub: eff_stages,
                        acc_ub: 1,
                        m_ub: 1,
                        l_ub: 1,
                        kw_ub: 1,
                        w_ub: 1,
                        g_ub: 1,
                        x_ub: 1,
                    }
                )
                T.annotate_unlimit_memory("shared")
            else:
                T.annotate_buffer_versions(
                    {
                        v_ub: eff_stages,
                        acc_ub: 1,
                        m_ub: 1,
                        l_ub: 1,
                        kw_ub: 1,
                        w_ub: 1,
                        g_ub: 1,
                        x_ub: 1,
                    }
                )

            if num_d_blks > 1:
                total_tasks = m_rows * num_d_blks
                base_tasks = total_tasks // num_cores
                rem_tasks = total_tasks % num_cores
                t_start = cid * base_tasks + T.min(cid, rem_tasks)
                t_count = base_tasks + T.if_then_else(cid < rem_tasks, 1, 0)
                if t_count > 0:
                    T.copy(W[:d_dim], w_ub[:d_dim])
                    T.copy(G[:d_dim], g_ub[:d_dim])
                    with T.SimdVF():
                        m32 = T.simd.pset(32, "PAT_ALL")
                        m16 = T.simd.pset(16, "PAT_ALL")
                        for c in T.serial(full_chunks):
                            off = c * 64
                            wf = T.simd.vcvt(
                                T.simd.vld(w_ub[off], dist="US_B16"),
                                "float32",
                                mask=m16,
                                part=0,
                            )
                            gf = T.simd.vcvt(
                                T.simd.vld(g_ub[off], dist="US_B16"),
                                "float32",
                                mask=m16,
                                part=0,
                            )
                            kwf = T.simd.vmul(wf, gf, mask=m32)
                            T.simd.vsts(kw_ub[off], kwf, mask=m32, dist="NORM_B32")

                    for local_t in T.serial(t_count):
                        task_id = t_start + local_t
                        m = task_id // num_d_blks
                        d_blk = task_id % num_d_blks
                        d_start = d_blk * eff_d_subtile
                        cur_d = (
                            T.if_then_else(d_blk < num_d_blks - 1, eff_d_subtile, tail_d)
                            if tail_d != eff_d_subtile
                            else eff_d_subtile
                        )
                        cur_chunks = (
                            T.if_then_else(d_blk < num_d_blks - 1, sub_chunks, tail_chunks)
                            if tail_d != eff_d_subtile
                            else sub_chunks
                        )
                        _ar_small_decode_run_task(
                            V,
                            X,
                            Y,
                            v_ub,
                            kw_ub,
                            x_ub,
                            acc_ub,
                            m_ub,
                            l_ub,
                            alpha_val,
                            eps_val,
                            m,
                            d_start,
                            cur_d,
                            cur_chunks,
                        )
            else:
                base_rows = m_rows // num_cores
                rem_rows = m_rows % num_cores
                m_start = cid * base_rows + T.min(cid, rem_rows)
                m_count = base_rows + T.if_then_else(cid < rem_rows, 1, 0)
                if m_count > 0:
                    T.copy(W[:d_dim], w_ub[:d_dim])
                    T.copy(G[:d_dim], g_ub[:d_dim])
                    with T.SimdVF():
                        m32 = T.simd.pset(32, "PAT_ALL")
                        m16 = T.simd.pset(16, "PAT_ALL")
                        for c in T.serial(full_chunks):
                            off = c * 64
                            wf = T.simd.vcvt(
                                T.simd.vld(w_ub[off], dist="US_B16"),
                                "float32",
                                mask=m16,
                                part=0,
                            )
                            gf = T.simd.vcvt(
                                T.simd.vld(g_ub[off], dist="US_B16"),
                                "float32",
                                mask=m16,
                                part=0,
                            )
                            kwf = T.simd.vmul(wf, gf, mask=m32)
                            T.simd.vsts(kw_ub[off], kwf, mask=m32, dist="NORM_B32")

                    if rows > 1:
                        full_blocks = m_count // rows
                        for mb in T.serial(full_blocks):
                            m0 = m_start + mb * rows
                            with T.SimdVF():
                                for tm in T.serial(rows):
                                    _ar_small_decode_init_row(
                                        m_ub, l_ub, acc_ub, tm, full_chunks
                                    )
                            for n in T.Pipelined(
                                n_depth, num_stages=eff_stages, annotations=pipe_ann
                            ):
                                T.copy(
                                    V[n * m_rows + m0 : n * m_rows + m0 + rows, :d_dim],
                                    v_ub[:, :d_dim],
                                )
                                with T.SimdVF():
                                    for tm in T.serial(rows):
                                        _ar_small_decode_step_row(
                                            v_ub,
                                            kw_ub,
                                            m_ub,
                                            l_ub,
                                            acc_ub,
                                            eps_val,
                                            inv_d,
                                            inv_sqrt_d,
                                            tm,
                                            full_chunks,
                                            0,
                                            full_chunks,
                                        )
                            T.copy(X[m0 : m0 + rows, :d_dim], x_ub[:, :d_dim])
                            with T.SimdVF():
                                for tm in T.serial(rows):
                                    _ar_small_decode_blend_row(
                                        x_ub,
                                        acc_ub,
                                        l_ub,
                                        alpha_val,
                                        tl_dtype,
                                        tm,
                                        full_chunks,
                                    )
                            T.copy(x_ub[:, :d_dim], Y[m0 : m0 + rows, :d_dim])

                        tail_rows = m_count % rows
                        tail_start = m_start + full_blocks * rows
                        for rr in T.serial(tail_rows):
                            m = tail_start + rr
                            with T.SimdVF():
                                _ar_small_decode_init_row(
                                    m_ub, l_ub, acc_ub, 0, full_chunks
                                )
                            for n in T.Pipelined(
                                n_depth, num_stages=eff_stages, annotations=pipe_ann
                            ):
                                row_off = n * m_rows + m
                                T.copy(V[row_off, :d_dim], v_tail_ub[0, :d_dim])
                                with T.SimdVF():
                                    _ar_small_decode_step_row(
                                        v_tail_ub,
                                        kw_ub,
                                        m_ub,
                                        l_ub,
                                        acc_ub,
                                        eps_val,
                                        inv_d,
                                        inv_sqrt_d,
                                        0,
                                        full_chunks,
                                        0,
                                        full_chunks,
                                    )
                            T.copy(X[m, :d_dim], x_ub[0, :d_dim])
                            with T.SimdVF():
                                _ar_small_decode_blend_row(
                                    x_ub,
                                    acc_ub,
                                    l_ub,
                                    alpha_val,
                                    tl_dtype,
                                    0,
                                    full_chunks,
                                )
                            T.copy(x_ub[0, :d_dim], Y[m, :d_dim])
                            T.ascend_pipe_barrier("PIPE_ALL")
                    else:
                        for local_m in T.serial(m_count):
                            m = m_start + local_m
                            with T.SimdVF():
                                _ar_small_decode_init_row(
                                    m_ub, l_ub, acc_ub, 0, full_chunks
                                )
                            for n in T.Pipelined(
                                n_depth, num_stages=eff_stages, annotations=pipe_ann
                            ):
                                row_off = n * m_rows + m
                                T.copy(V[row_off, :d_dim], v_ub[0, :d_dim])
                                with T.SimdVF():
                                    _ar_small_decode_step_row(
                                        v_ub,
                                        kw_ub,
                                        m_ub,
                                        l_ub,
                                        acc_ub,
                                        eps_val,
                                        inv_d,
                                        inv_sqrt_d,
                                        0,
                                        full_chunks,
                                        0,
                                        full_chunks,
                                    )
                            T.copy(X[m, :d_dim], x_ub[0, :d_dim])
                            with T.SimdVF():
                                _ar_small_decode_blend_row(
                                    x_ub,
                                    acc_ub,
                                    l_ub,
                                    alpha_val,
                                    tl_dtype,
                                    0,
                                    full_chunks,
                                )
                            T.copy(x_ub[0, :d_dim], Y[m, :d_dim])
                            T.ascend_pipe_barrier("PIPE_ALL")

    return main


@T.macro
def _ar_unaligned_init_row(m_ub, l_ub, acc_ub, slot, full_chunks, rem):
    m32 = T.simd.pset(32, "PAT_ALL")
    v_init_m = T.simd.vdup(T.float32(-1.0e30), "float32", mask=m32)
    v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
    T.simd.vsts(m_ub[slot, 0], v_init_m, mask=m32, dist="ONEPT_B32")
    T.simd.vsts(l_ub[slot, 0], v_zero, mask=m32, dist="ONEPT_B32")
    for c in T.serial(full_chunks):
        T.simd.vsts(acc_ub[slot, c * 64], v_zero, mask=m32, dist="NORM_B32")
    if rem > 0:
        T.simd.vsts(
            acc_ub[slot, full_chunks * 64], v_zero, mask=m32, dist="NORM_B32"
        )


@T.macro
def _ar_unaligned_step_row(
    v_ub, kw_ub, m_ub, l_ub, acc_ub, eps_val, inv_d, inv_sqrt_d, slot, full_chunks, rem
):
    m32 = T.simd.pset(32, "PAT_ALL")
    m16 = T.simd.pset(16, "PAT_ALL")
    v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
    v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
    sq_acc = T.simd.alloc_local((1,), "float32")
    dot_acc = T.simd.alloc_local((1,), "float32")
    sq_acc[0] = v_zero
    dot_acc[0] = v_zero
    for c in T.serial(full_chunks):
        off = c * 64
        vf = T.simd.vcvt(
            T.simd.vld(v_ub[slot, off], dist="US_B16"),
            "float32",
            mask=m16,
            part=0,
        )
        kwf = T.simd.vld(kw_ub[off], dist="NORM_B32")
        T.simd.vmula(sq_acc[0], vf, vf, mask=m32)
        T.simd.vmula(dot_acc[0], vf, kwf, mask=m32)
    if rem > 0:
        off = full_chunks * 64
        m32_tail = T.simd.update_mask(rem, width=32)
        m16_tail_x2 = T.simd.update_mask(rem * 2, width=16)
        vf_t = T.simd.vcvt(
            T.simd.vld(v_ub[slot, off], dist="US_B16"),
            "float32",
            mask=m16_tail_x2,
            part=0,
        )
        kwf_t = T.simd.vld(kw_ub[off], dist="NORM_B32")
        T.simd.vmula(
            sq_acc[0], vf_t, vf_t, mask=m32_tail, mode="MODE_MERGING"
        )
        T.simd.vmula(
            dot_acc[0], vf_t, kwf_t, mask=m32_tail, mode="MODE_MERGING"
        )

    sum_sq = T.simd.vdupv(
        T.simd.vcadd(sq_acc[0], mask=m32), mask=m32, pos="POS_LOWEST"
    )
    raw_dot = T.simd.vdupv(
        T.simd.vcadd(dot_acc[0], mask=m32), mask=m32, pos="POS_LOWEST"
    )
    mean_sq = T.simd.vmuls(sum_sq, T.float32(inv_d), mask=m32)
    mean_eps = T.simd.vadds(mean_sq, eps_val, mask=m32)
    rstd = T.simd.vdiv(v_one, T.simd.vsqrt(mean_eps, mask=m32), mask=m32)
    s_vec = T.simd.vmuls(
        T.simd.vmul(raw_dot, rstd, mask=m32),
        T.float32(inv_sqrt_d),
        mask=m32,
    )

    m_old = T.simd.vld(m_ub[slot, 0], dist="BRC_B32")
    l_old = T.simd.vld(l_ub[slot, 0], dist="BRC_B32")
    m_new = T.simd.vmax(m_old, s_vec, mask=m32)
    alpha_v = T.simd.vexp(T.simd.vsub(m_old, m_new, mask=m32), mask=m32)
    p_v = T.simd.vexp(T.simd.vsub(s_vec, m_new, mask=m32), mask=m32)
    l_new = T.simd.vadd(T.simd.vmul(l_old, alpha_v, mask=m32), p_v, mask=m32)
    T.simd.vsts(m_ub[slot, 0], m_new, mask=m32, dist="ONEPT_B32")
    T.simd.vsts(l_ub[slot, 0], l_new, mask=m32, dist="ONEPT_B32")

    for c in T.serial(full_chunks):
        off = c * 64
        vf = T.simd.vcvt(
            T.simd.vld(v_ub[slot, off], dist="US_B16"),
            "float32",
            mask=m16,
            part=0,
        )
        acc_old = T.simd.vld(acc_ub[slot, off], dist="NORM_B32")
        acc_new = T.simd.vadd(
            T.simd.vmul(acc_old, alpha_v, mask=m32),
            T.simd.vmul(vf, p_v, mask=m32),
            mask=m32,
        )
        T.simd.vsts(acc_ub[slot, off], acc_new, mask=m32, dist="NORM_B32")
    if rem > 0:
        off = full_chunks * 64
        m32_tail = T.simd.update_mask(rem, width=32)
        m16_tail_x2 = T.simd.update_mask(rem * 2, width=16)
        vf_t = T.simd.vcvt(
            T.simd.vld(v_ub[slot, off], dist="US_B16"),
            "float32",
            mask=m16_tail_x2,
            part=0,
        )
        acc_old_t = T.simd.vld(acc_ub[slot, off], dist="NORM_B32")
        acc_new_t = T.simd.vadd(
            T.simd.vmul(acc_old_t, alpha_v, mask=m32_tail),
            T.simd.vmul(vf_t, p_v, mask=m32_tail),
            mask=m32_tail,
        )
        T.simd.vsts(
            acc_ub[slot, off], acc_new_t, mask=m32_tail, dist="NORM_B32"
        )


@T.macro
def _ar_unaligned_blend_row(
    x_ub, acc_ub, l_ub, alpha_val, tl_dtype, slot, full_chunks, rem
):
    m32 = T.simd.pset(32, "PAT_ALL")
    m16 = T.simd.pset(16, "PAT_ALL")
    v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
    l_tot = T.simd.vld(l_ub[slot, 0], dist="BRC_B32")
    inv_l = T.simd.vdiv(v_one, l_tot, mask=m32)
    scale_agg = T.simd.vmuls(inv_l, alpha_val, mask=m32)
    one_minus_alpha = T.float32(1.0) - alpha_val
    for c in T.serial(full_chunks):
        off = c * 64
        xf = T.simd.vcvt(
            T.simd.vld(x_ub[slot, off], dist="US_B16"),
            "float32",
            mask=m16,
            part=0,
        )
        accf = T.simd.vld(acc_ub[slot, off], dist="NORM_B32")
        yf = T.simd.vadd(
            T.simd.vmuls(xf, one_minus_alpha, mask=m32),
            T.simd.vmul(accf, scale_agg, mask=m32),
            mask=m32,
        )
        y16 = T.simd.vcvt(yf, tl_dtype, mask=m16, part=0)
        T.simd.vsts(
            x_ub[slot, off], y16, mask=m32, dist="PK_B32", extent=64
        )
    if rem > 0:
        off = full_chunks * 64
        m32_tail = T.simd.update_mask(rem, width=32)
        m16_tail_x2 = T.simd.update_mask(rem * 2, width=16)
        xf_t = T.simd.vcvt(
            T.simd.vld(x_ub[slot, off], dist="US_B16"),
            "float32",
            mask=m16_tail_x2,
            part=0,
        )
        accf_t = T.simd.vld(acc_ub[slot, off], dist="NORM_B32")
        yf_t = T.simd.vadd(
            T.simd.vmuls(xf_t, one_minus_alpha, mask=m32_tail),
            T.simd.vmul(accf_t, scale_agg, mask=m32_tail),
            mask=m32_tail,
        )
        y16_t = T.simd.vcvt(yf_t, tl_dtype, mask=m16_tail_x2, part=0)
        T.simd.vsts(
            x_ub[slot, off], y16_t, mask=m32_tail, dist="PK_B32", extent=64
        )


def _choose_ar_tile_m_for_rows(
    d_dim: int,
    m_rows: int,
    schedule_mode: str,
    *,
    allow_unaligned: bool = False,
    num_stages: int = 2,
) -> int:
    if schedule_mode != "tuned":
        return 1
    if not allow_unaligned and (d_dim % 64) != 0:
        return 1
    d_alloc = ((d_dim + 63) // 64) * 64
    ub_budget = 256 * 1024 - 1024
    eff_stages = max(1, min(2, int(num_stages)))
    fixed_coef = 8 + 2 * eff_stages
    row_coef = 6 + 2 * eff_stages
    max_rows = (ub_budget - fixed_coef * d_alloc) // (row_coef * d_alloc + 64)
    base_tile = max(1, min(8, max_rows))
    if m_rows < 56:
        return 1
    return max(1, min(base_tile, m_rows // 56))


def _select_ar_unaligned_config(
    n_depth: int = 8,
    d_dim: int = 1000,
    tl_dtype: str = "bfloat16",
    schedule_mode: str = "tuned",
    m_rows: int = 4,
    tile_m_override: int = 0,
    num_stages_override: int = 0,
) -> dict[str, object]:
    num_stages = (
        max(1, min(2, int(num_stages_override)))
        if int(num_stages_override) > 0
        else (2 if (schedule_mode == "tuned" and int(n_depth) >= 2) else 1)
    )
    tile_m = (
        max(1, int(tile_m_override))
        if int(tile_m_override) > 0
        else _choose_ar_tile_m_for_rows(
            d_dim,
            m_rows,
            schedule_mode,
            allow_unaligned=True,
            num_stages=num_stages,
        )
    )
    static_ub_bytes = _estimate_ar_static_ub_bytes(
        n_depth,
        d_dim,
        tile_m,
        schedule_mode,
        tl_dtype,
        num_stages=num_stages,
    )
    return {
        "tile_m": tile_m,
        "num_stages": num_stages,
        "static_ub_bytes": static_ub_bytes,
        "pass_configs": {},
        "compile_flags": _ar_default_flags(schedule_mode, static_ub_bytes=static_ub_bytes),
    }


def _build_ar_unaligned_func(
    n_depth: int = 8,
    d_dim: int = 1000,
    tl_dtype: str = "bfloat16",
    tile_m: int = 1,
    schedule_mode: str = "tuned",
    num_stages: int = 0,
    **_extra_kwargs,
):
    num_cores = 56
    full_chunks = d_dim // 64
    rem = d_dim % 64
    d_alloc = ((d_dim + 63) // 64) * 64
    tm_ov = int(_extra_kwargs.get("tile_m_override", 0) or 0)
    ns_ov = int(_extra_kwargs.get("num_stages_override", 0) or 0)
    rows = max(1, tm_ov if tm_ov > 0 else int(tile_m))
    raw_stages = ns_ov if ns_ov > 0 else int(num_stages)
    num_stages = (
        max(1, min(2, raw_stages))
        if raw_stages > 0
        else (2 if (schedule_mode == "tuned" and n_depth >= 2) else 1)
    )
    pipe_ann = {"enable_offset": True} if schedule_mode == "tuned" else None
    inv_sqrt_d = 1.0 / math.sqrt(d_dim)
    inv_d = 1.0 / float(d_dim)

    m_rows = T.dynamic("m_rows")
    v_rows = T.dynamic("v_rows")

    @T.prim_func
    def main(
        V: T.Tensor((v_rows, d_dim), tl_dtype),
        X: T.Tensor((m_rows, d_dim), tl_dtype),
        W: T.Tensor((d_dim,), tl_dtype),
        G: T.Tensor((d_dim,), tl_dtype),
        Y: T.Tensor((m_rows, d_dim), tl_dtype),
        alpha_val: T.float32,
        eps_val: T.float32,
    ):
        with T.Kernel(num_cores) as cid:
            w_ub = T.alloc_shared((d_alloc,), tl_dtype)
            g_ub = T.alloc_shared((d_alloc,), tl_dtype)
            kw_ub = T.alloc_shared((d_alloc,), "float32")
            v_ub = T.alloc_shared((rows, d_alloc), tl_dtype)
            x_ub = T.alloc_shared((rows, d_alloc), tl_dtype)
            acc_ub = T.alloc_shared((rows, d_alloc), "float32")
            m_ub = T.alloc_shared((rows, 8), "float32")
            l_ub = T.alloc_shared((rows, 8), "float32")
            if rows > 1:
                v_tail_ub = T.alloc_shared((1, d_alloc), tl_dtype)
                T.annotate_buffer_versions(
                    {
                        v_ub: num_stages,
                        v_tail_ub: num_stages,
                        acc_ub: 1,
                        m_ub: 1,
                        l_ub: 1,
                        kw_ub: 1,
                        w_ub: 1,
                        g_ub: 1,
                        x_ub: 1,
                    }
                )
                T.annotate_unlimit_memory("shared")
            else:
                T.annotate_buffer_versions(
                    {
                        v_ub: num_stages,
                        acc_ub: 1,
                        m_ub: 1,
                        l_ub: 1,
                        kw_ub: 1,
                        w_ub: 1,
                        g_ub: 1,
                        x_ub: 1,
                    }
                )

            base_rows = m_rows // num_cores
            rem_rows = m_rows % num_cores
            m_start = cid * base_rows + T.min(cid, rem_rows)
            m_count = base_rows + T.if_then_else(cid < rem_rows, 1, 0)
            if m_count > 0:
                T.copy(W[:d_dim], w_ub[:d_dim])
                T.copy(G[:d_dim], g_ub[:d_dim])
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    m16 = T.simd.pset(16, "PAT_ALL")
                    for c in T.serial(full_chunks):
                        off = c * 64
                        wf = T.simd.vcvt(
                            T.simd.vld(w_ub[off], dist="US_B16"),
                            "float32",
                            mask=m16,
                            part=0,
                        )
                        gf = T.simd.vcvt(
                            T.simd.vld(g_ub[off], dist="US_B16"),
                            "float32",
                            mask=m16,
                            part=0,
                        )
                        kwf = T.simd.vmul(wf, gf, mask=m32)
                        T.simd.vsts(kw_ub[off], kwf, mask=m32, dist="NORM_B32")
                    if rem > 0:
                        off = full_chunks * 64
                        m16_tail_x2 = T.simd.update_mask(rem * 2, width=16)
                        wf_t = T.simd.vcvt(
                            T.simd.vld(w_ub[off], dist="US_B16"),
                            "float32",
                            mask=m16_tail_x2,
                            part=0,
                        )
                        gf_t = T.simd.vcvt(
                            T.simd.vld(g_ub[off], dist="US_B16"),
                            "float32",
                            mask=m16_tail_x2,
                            part=0,
                        )
                        kwf_t = T.simd.vmul(wf_t, gf_t, mask=m32)
                        T.simd.vsts(kw_ub[off], kwf_t, mask=m32, dist="NORM_B32")

                if rows > 1:
                    full_blocks = m_count // rows
                    for mb in T.serial(full_blocks):
                        m0 = m_start + mb * rows
                        with T.SimdVF():
                            for tm in T.serial(rows):
                                _ar_unaligned_init_row(
                                    m_ub, l_ub, acc_ub, tm, full_chunks, rem
                                )
                        for n in T.Pipelined(
                            n_depth, num_stages=num_stages, annotations=pipe_ann
                        ):
                            T.copy(
                                V[n * m_rows + m0 : n * m_rows + m0 + rows, :d_dim],
                                v_ub[:, :d_dim],
                            )
                            with T.SimdVF():
                                for tm in T.serial(rows):
                                    _ar_unaligned_step_row(
                                        v_ub,
                                        kw_ub,
                                        m_ub,
                                        l_ub,
                                        acc_ub,
                                        eps_val,
                                        inv_d,
                                        inv_sqrt_d,
                                        tm,
                                        full_chunks,
                                        rem,
                                    )
                        T.copy(X[m0 : m0 + rows, :d_dim], x_ub[:, :d_dim])
                        with T.SimdVF():
                            for tm in T.serial(rows):
                                _ar_unaligned_blend_row(
                                    x_ub,
                                    acc_ub,
                                    l_ub,
                                    alpha_val,
                                    tl_dtype,
                                    tm,
                                    full_chunks,
                                    rem,
                                )
                        T.copy(x_ub[:, :d_dim], Y[m0 : m0 + rows, :d_dim])

                    tail_rows = m_count % rows
                    tail_start = m_start + full_blocks * rows
                    for rr in T.serial(tail_rows):
                        m = tail_start + rr
                        with T.SimdVF():
                            _ar_unaligned_init_row(
                                m_ub, l_ub, acc_ub, 0, full_chunks, rem
                            )
                        for n in T.Pipelined(
                            n_depth, num_stages=num_stages, annotations=pipe_ann
                        ):
                            row_off = n * m_rows + m
                            T.copy(V[row_off, :d_dim], v_tail_ub[0, :d_dim])
                            with T.SimdVF():
                                _ar_unaligned_step_row(
                                    v_tail_ub,
                                    kw_ub,
                                    m_ub,
                                    l_ub,
                                    acc_ub,
                                    eps_val,
                                    inv_d,
                                    inv_sqrt_d,
                                    0,
                                    full_chunks,
                                    rem,
                                )
                        T.copy(X[m, :d_dim], x_ub[0, :d_dim])
                        with T.SimdVF():
                            _ar_unaligned_blend_row(
                                x_ub,
                                acc_ub,
                                l_ub,
                                alpha_val,
                                tl_dtype,
                                0,
                                full_chunks,
                                rem,
                            )
                        T.copy(x_ub[0, :d_dim], Y[m, :d_dim])
                        T.ascend_pipe_barrier("PIPE_ALL")
                else:
                    for local_m in T.serial(m_count):
                        m = m_start + local_m
                        with T.SimdVF():
                            _ar_unaligned_init_row(
                                m_ub, l_ub, acc_ub, 0, full_chunks, rem
                            )
                        for n in T.Pipelined(
                            n_depth, num_stages=num_stages, annotations=pipe_ann
                        ):
                            row_off = n * m_rows + m
                            T.copy(V[row_off, :d_dim], v_ub[0, :d_dim])
                            with T.SimdVF():
                                _ar_unaligned_step_row(
                                    v_ub,
                                    kw_ub,
                                    m_ub,
                                    l_ub,
                                    acc_ub,
                                    eps_val,
                                    inv_d,
                                    inv_sqrt_d,
                                    0,
                                    full_chunks,
                                    rem,
                                )
                        T.copy(X[m, :d_dim], x_ub[0, :d_dim])
                        with T.SimdVF():
                            _ar_unaligned_blend_row(
                                x_ub,
                                acc_ub,
                                l_ub,
                                alpha_val,
                                tl_dtype,
                                0,
                                full_chunks,
                                rem,
                            )
                        T.copy(x_ub[0, :d_dim], Y[m, :d_dim])
                        T.ascend_pipe_barrier("PIPE_ALL")

    return main


@T.macro
def _ar_aligned_init_row(m_ub, l_ub, acc_ub, slot, full_chunks):
    m32 = T.simd.pset(32, "PAT_ALL")
    v_init_m = T.simd.vdup(T.float32(-1.0e30), "float32", mask=m32)
    v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
    T.simd.vsts(m_ub[slot, 0], v_init_m, mask=m32, dist="ONEPT_B32")
    T.simd.vsts(l_ub[slot, 0], v_zero, mask=m32, dist="ONEPT_B32")
    for c in T.serial(full_chunks):
        T.simd.vsts(acc_ub[slot, c * 64], v_zero, mask=m32, dist="NORM_B32")


@T.macro
def _ar_aligned_step_row(
    v_ub, kw_ub, m_ub, l_ub, acc_ub, eps_val, inv_d, inv_sqrt_d, slot, full_chunks
):
    m32 = T.simd.pset(32, "PAT_ALL")
    m16 = T.simd.pset(16, "PAT_ALL")
    v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
    v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
    sq_acc = T.simd.alloc_local((1,), "float32")
    dot_acc = T.simd.alloc_local((1,), "float32")
    sq_acc[0] = v_zero
    dot_acc[0] = v_zero
    for c in T.serial(full_chunks):
        off = c * 64
        vf = T.simd.vcvt(
            T.simd.vld(v_ub[slot, off], dist="US_B16"),
            "float32",
            mask=m16,
            part=0,
        )
        kwf = T.simd.vld(kw_ub[off], dist="NORM_B32")
        T.simd.vmula(sq_acc[0], vf, vf, mask=m32)
        T.simd.vmula(dot_acc[0], vf, kwf, mask=m32)

    sum_sq = T.simd.vdupv(
        T.simd.vcadd(sq_acc[0], mask=m32), mask=m32, pos="POS_LOWEST"
    )
    raw_dot = T.simd.vdupv(
        T.simd.vcadd(dot_acc[0], mask=m32), mask=m32, pos="POS_LOWEST"
    )
    mean_sq = T.simd.vmuls(sum_sq, T.float32(inv_d), mask=m32)
    mean_eps = T.simd.vadds(mean_sq, eps_val, mask=m32)
    rstd = T.simd.vdiv(v_one, T.simd.vsqrt(mean_eps, mask=m32), mask=m32)
    s_vec = T.simd.vmuls(
        T.simd.vmul(raw_dot, rstd, mask=m32),
        T.float32(inv_sqrt_d),
        mask=m32,
    )

    m_old = T.simd.vld(m_ub[slot, 0], dist="BRC_B32")
    l_old = T.simd.vld(l_ub[slot, 0], dist="BRC_B32")
    m_new = T.simd.vmax(m_old, s_vec, mask=m32)
    alpha_v = T.simd.vexp(T.simd.vsub(m_old, m_new, mask=m32), mask=m32)
    p_v = T.simd.vexp(T.simd.vsub(s_vec, m_new, mask=m32), mask=m32)
    l_new = T.simd.vadd(T.simd.vmul(l_old, alpha_v, mask=m32), p_v, mask=m32)
    T.simd.vsts(m_ub[slot, 0], m_new, mask=m32, dist="ONEPT_B32")
    T.simd.vsts(l_ub[slot, 0], l_new, mask=m32, dist="ONEPT_B32")

    for c in T.serial(full_chunks):
        off = c * 64
        vf = T.simd.vcvt(
            T.simd.vld(v_ub[slot, off], dist="US_B16"),
            "float32",
            mask=m16,
            part=0,
        )
        acc_old = T.simd.vld(acc_ub[slot, off], dist="NORM_B32")
        acc_new = T.simd.vadd(
            T.simd.vmul(acc_old, alpha_v, mask=m32),
            T.simd.vmul(vf, p_v, mask=m32),
            mask=m32,
        )
        T.simd.vsts(acc_ub[slot, off], acc_new, mask=m32, dist="NORM_B32")


@T.macro
def _ar_aligned_blend_row(
    x_ub, acc_ub, l_ub, alpha_val, tl_dtype, slot, full_chunks
):
    m32 = T.simd.pset(32, "PAT_ALL")
    m16 = T.simd.pset(16, "PAT_ALL")
    v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
    l_tot = T.simd.vld(l_ub[slot, 0], dist="BRC_B32")
    inv_l = T.simd.vdiv(v_one, l_tot, mask=m32)
    scale_agg = T.simd.vmuls(inv_l, alpha_val, mask=m32)
    one_minus_alpha = T.float32(1.0) - alpha_val
    for c in T.serial(full_chunks):
        off = c * 64
        xf = T.simd.vcvt(
            T.simd.vld(x_ub[slot, off], dist="US_B16"),
            "float32",
            mask=m16,
            part=0,
        )
        accf = T.simd.vld(acc_ub[slot, off], dist="NORM_B32")
        yf = T.simd.vadd(
            T.simd.vmuls(xf, one_minus_alpha, mask=m32),
            T.simd.vmul(accf, scale_agg, mask=m32),
            mask=m32,
        )
        y16 = T.simd.vcvt(yf, tl_dtype, mask=m16, part=0)
        T.simd.vsts(x_ub[slot, off], y16, mask=m32, dist="PK_B32", extent=64)


def _select_ar_aligned_config(
    n_depth: int = 8,
    d_dim: int = 4096,
    tl_dtype: str = "bfloat16",
    schedule_mode: str = "tuned",
    m_rows: int = 2048,
    tile_m_override: int = 0,
    num_stages_override: int = 0,
) -> dict[str, object]:
    num_stages = (
        max(1, min(2, int(num_stages_override)))
        if int(num_stages_override) > 0
        else (2 if (schedule_mode == "tuned" and int(n_depth) >= 2) else 1)
    )
    tile_m = (
        max(1, int(tile_m_override))
        if int(tile_m_override) > 0
        else _choose_ar_tile_m_for_rows(
            d_dim,
            m_rows,
            schedule_mode,
            allow_unaligned=False,
            num_stages=num_stages,
        )
    )
    static_ub_bytes = _estimate_ar_static_ub_bytes(
        n_depth,
        d_dim,
        tile_m,
        schedule_mode,
        tl_dtype,
        num_stages=num_stages,
    )
    return {
        "tile_m": tile_m,
        "num_stages": num_stages,
        "static_ub_bytes": static_ub_bytes,
        "pass_configs": {},
        "compile_flags": _ar_default_flags(schedule_mode, static_ub_bytes=static_ub_bytes),
    }


def _build_ar_aligned_func(
    n_depth: int = 8,
    d_dim: int = 4096,
    tl_dtype: str = "bfloat16",
    tile_m: int = 5,
    schedule_mode: str = "tuned",
    num_stages: int = 0,
    **_extra_kwargs,
):
    if (d_dim % 64) != 0:
        raise ValueError(f"aligned regime requires d_dim % 64 == 0, got {d_dim}")
    num_cores = 56
    full_chunks = d_dim // 64
    tm_ov = int(_extra_kwargs.get("tile_m_override", 0) or 0)
    ns_ov = int(_extra_kwargs.get("num_stages_override", 0) or 0)
    rows = max(1, tm_ov if tm_ov > 0 else int(tile_m))
    raw_stages = ns_ov if ns_ov > 0 else int(num_stages)
    num_stages = (
        max(1, min(2, raw_stages))
        if raw_stages > 0
        else (2 if (schedule_mode == "tuned" and n_depth >= 2) else 1)
    )
    pipe_ann = {"enable_offset": True} if schedule_mode == "tuned" else None
    inv_sqrt_d = 1.0 / math.sqrt(d_dim)
    inv_d = 1.0 / float(d_dim)

    m_rows = T.dynamic("m_rows")
    v_rows = T.dynamic("v_rows")

    @T.prim_func
    def main(
        V: T.Tensor((v_rows, d_dim), tl_dtype),
        X: T.Tensor((m_rows, d_dim), tl_dtype),
        W: T.Tensor((d_dim,), tl_dtype),
        G: T.Tensor((d_dim,), tl_dtype),
        Y: T.Tensor((m_rows, d_dim), tl_dtype),
        alpha_val: T.float32,
        eps_val: T.float32,
    ):
        with T.Kernel(num_cores) as cid:
            w_ub = T.alloc_shared((d_dim,), tl_dtype)
            g_ub = T.alloc_shared((d_dim,), tl_dtype)
            kw_ub = T.alloc_shared((d_dim,), "float32")
            v_ub = T.alloc_shared((rows, d_dim), tl_dtype)
            x_ub = T.alloc_shared((rows, d_dim), tl_dtype)
            acc_ub = T.alloc_shared((rows, d_dim), "float32")
            m_ub = T.alloc_shared((rows, 8), "float32")
            l_ub = T.alloc_shared((rows, 8), "float32")
            if rows > 1:
                v_tail_ub = T.alloc_shared((1, d_dim), tl_dtype)
                T.annotate_buffer_versions(
                    {
                        v_ub: num_stages,
                        v_tail_ub: num_stages,
                        acc_ub: 1,
                        m_ub: 1,
                        l_ub: 1,
                        kw_ub: 1,
                        w_ub: 1,
                        g_ub: 1,
                        x_ub: 1,
                    }
                )
                T.annotate_unlimit_memory("shared")
            else:
                T.annotate_buffer_versions(
                    {
                        v_ub: num_stages,
                        acc_ub: 1,
                        m_ub: 1,
                        l_ub: 1,
                        kw_ub: 1,
                        w_ub: 1,
                        g_ub: 1,
                        x_ub: 1,
                    }
                )

            base_rows = m_rows // num_cores
            rem_rows = m_rows % num_cores
            m_start = cid * base_rows + T.min(cid, rem_rows)
            m_count = base_rows + T.if_then_else(cid < rem_rows, 1, 0)
            if m_count > 0:
                T.copy(W[:d_dim], w_ub[:d_dim])
                T.copy(G[:d_dim], g_ub[:d_dim])
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    m16 = T.simd.pset(16, "PAT_ALL")
                    for c in T.serial(full_chunks):
                        off = c * 64
                        wf = T.simd.vcvt(
                            T.simd.vld(w_ub[off], dist="US_B16"),
                            "float32",
                            mask=m16,
                            part=0,
                        )
                        gf = T.simd.vcvt(
                            T.simd.vld(g_ub[off], dist="US_B16"),
                            "float32",
                            mask=m16,
                            part=0,
                        )
                        kwf = T.simd.vmul(wf, gf, mask=m32)
                        T.simd.vsts(kw_ub[off], kwf, mask=m32, dist="NORM_B32")

                if rows > 1:
                    full_blocks = m_count // rows
                    for mb in T.serial(full_blocks):
                        m0 = m_start + mb * rows
                        with T.SimdVF():
                            for tm in T.serial(rows):
                                _ar_aligned_init_row(
                                    m_ub, l_ub, acc_ub, tm, full_chunks
                                )
                        for n in T.Pipelined(
                            n_depth, num_stages=num_stages, annotations=pipe_ann
                        ):
                            T.copy(
                                V[n * m_rows + m0 : n * m_rows + m0 + rows, :d_dim],
                                v_ub[:, :d_dim],
                            )
                            with T.SimdVF():
                                for tm in T.serial(rows):
                                    _ar_aligned_step_row(
                                        v_ub,
                                        kw_ub,
                                        m_ub,
                                        l_ub,
                                        acc_ub,
                                        eps_val,
                                        inv_d,
                                        inv_sqrt_d,
                                        tm,
                                        full_chunks,
                                    )
                        T.copy(X[m0 : m0 + rows, :d_dim], x_ub[:, :d_dim])
                        with T.SimdVF():
                            for tm in T.serial(rows):
                                _ar_aligned_blend_row(
                                    x_ub,
                                    acc_ub,
                                    l_ub,
                                    alpha_val,
                                    tl_dtype,
                                    tm,
                                    full_chunks,
                                )
                        T.copy(x_ub[:, :d_dim], Y[m0 : m0 + rows, :d_dim])

                    tail_rows = m_count % rows
                    tail_start = m_start + full_blocks * rows
                    for rr in T.serial(tail_rows):
                        m = tail_start + rr
                        with T.SimdVF():
                            _ar_aligned_init_row(
                                m_ub, l_ub, acc_ub, 0, full_chunks
                            )
                        for n in T.Pipelined(
                            n_depth, num_stages=num_stages, annotations=pipe_ann
                        ):
                            row_off = n * m_rows + m
                            T.copy(V[row_off, :d_dim], v_tail_ub[0, :d_dim])
                            with T.SimdVF():
                                _ar_aligned_step_row(
                                    v_tail_ub,
                                    kw_ub,
                                    m_ub,
                                    l_ub,
                                    acc_ub,
                                    eps_val,
                                    inv_d,
                                    inv_sqrt_d,
                                    0,
                                    full_chunks,
                                )
                        T.copy(X[m, :d_dim], x_ub[0, :d_dim])
                        with T.SimdVF():
                            _ar_aligned_blend_row(
                                x_ub,
                                acc_ub,
                                l_ub,
                                alpha_val,
                                tl_dtype,
                                0,
                                full_chunks,
                            )
                        T.copy(x_ub[0, :d_dim], Y[m, :d_dim])
                        T.ascend_pipe_barrier("PIPE_ALL")
                else:
                    for local_m in T.serial(m_count):
                        m = m_start + local_m
                        with T.SimdVF():
                            _ar_aligned_init_row(
                                m_ub, l_ub, acc_ub, 0, full_chunks
                            )
                        for n in T.Pipelined(
                            n_depth, num_stages=num_stages, annotations=pipe_ann
                        ):
                            row_off = n * m_rows + m
                            T.copy(V[row_off, :d_dim], v_ub[0, :d_dim])
                            with T.SimdVF():
                                _ar_aligned_step_row(
                                    v_ub,
                                    kw_ub,
                                    m_ub,
                                    l_ub,
                                    acc_ub,
                                    eps_val,
                                    inv_d,
                                    inv_sqrt_d,
                                    0,
                                    full_chunks,
                                )
                        T.copy(X[m, :d_dim], x_ub[0, :d_dim])
                        with T.SimdVF():
                            _ar_aligned_blend_row(
                                x_ub,
                                acc_ub,
                                l_ub,
                                alpha_val,
                                tl_dtype,
                                0,
                                full_chunks,
                            )
                        T.copy(x_ub[0, :d_dim], Y[m, :d_dim])
                        T.ascend_pipe_barrier("PIPE_ALL")

    return main


def _freeze_attr_value(val: object) -> object:
    if isinstance(val, dict):
        return tuple((str(k), _freeze_attr_value(v)) for k, v in sorted(val.items()))
    if isinstance(val, (list, tuple)):
        return tuple(_freeze_attr_value(x) for x in val)
    return val


def _split_pass_configs_and_flags(
    raw_pass_configs: object,
    raw_compile_flags: object,
) -> tuple[dict[str, object], tuple[str, ...]]:
    pc_dict: dict[str, object] = {}
    flags: list[str] = []

    def _add_flag(flag_text: str) -> None:
        cleaned = flag_text.strip()
        if cleaned and cleaned not in flags:
            flags.append(cleaned)

    if isinstance(raw_pass_configs, Mapping):
        for k, v in raw_pass_configs.items():
            pc_dict[str(k)] = v
    elif isinstance(raw_pass_configs, str):
        for tok in raw_pass_configs.split():
            _add_flag(tok)
    elif isinstance(raw_pass_configs, Iterable):
        for item in raw_pass_configs:
            if (
                isinstance(item, (tuple, list))
                and len(item) == 2
                and not isinstance(item, str)
            ):
                pc_dict[str(item[0])] = item[1]
            else:
                _add_flag(str(item))

    if isinstance(raw_compile_flags, str):
        for tok in raw_compile_flags.split():
            _add_flag(tok)
    elif isinstance(raw_compile_flags, Iterable):
        for item in raw_compile_flags:
            _add_flag(str(item))
    elif raw_compile_flags is not None:
        _add_flag(str(raw_compile_flags))

    return pc_dict, tuple(flags)


def _normalize_pass_configs_key(pass_configs: object) -> tuple[tuple[str, object], ...]:
    if not pass_configs:
        return ()
    if isinstance(pass_configs, Mapping):
        return tuple(
            (str(k), _freeze_attr_value(v))
            for k, v in sorted(pass_configs.items(), key=lambda item: str(item[0]))
        )
    pc_dict, extra_flags = _split_pass_configs_and_flags(pass_configs, None)
    items: list[tuple[str, object]] = [
        (str(k), _freeze_attr_value(v))
        for k, v in sorted(pc_dict.items(), key=lambda item: str(item[0]))
    ]
    if extra_flags:
        items.append(("__compile_flags__", extra_flags))
    return tuple(items)


def _get_kernel(
    n_depth: int,
    d_dim: int,
    tl_dtype: str,
    tile_m: int,
    schedule_mode: str,
    *,
    builder_fn=None,
    pass_configs: object = None,
    compile_flags: list[str] | tuple[str, ...] | str | None = None,
    extra_builder_kwargs: dict[str, object] | None = None,
):
    resolved_builder = builder_fn or _build_attention_residual_prim_func
    pc_dict, parsed_flags = _split_pass_configs_and_flags(pass_configs, compile_flags)
    extra_bk = dict(extra_builder_kwargs or {})
    if compile_flags is None:
        static_ub_bytes = _estimate_ar_static_ub_bytes(
            n_depth,
            d_dim,
            tile_m,
            schedule_mode,
            tl_dtype,
            num_stages=int(extra_bk.get("num_stages", 0) or 0),
            d_subtile=int(extra_bk.get("d_subtile", 0) or 0),
        )
        default_flags = _ar_default_flags(schedule_mode, static_ub_bytes=static_ub_bytes)
        flags_tuple = tuple(dict.fromkeys([*default_flags, *parsed_flags]))
    else:
        flags_tuple = parsed_flags
    key = (
        resolved_builder.__name__,
        n_depth,
        d_dim,
        tl_dtype,
        int(tile_m),
        schedule_mode,
        _normalize_pass_configs_key(pc_dict),
        flags_tuple,
        _normalize_pass_configs_key(extra_bk),
    )
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        with _ascend_target_ctx():
            prog = resolved_builder(
                n_depth,
                d_dim,
                tl_dtype,
                tile_m,
                schedule_mode,
                **extra_bk,
            )
        merged_flags = list(dict.fromkeys([*_cann_include_flags(), *flags_tuple]))
        flags_arg = merged_flags or None
        compiled = tilelang.compile(
            prog,
            target="ascend",
            out_idx=[4],
            pass_configs=pc_dict or None,
            compile_flags=flags_arg,
        )
        _KERNEL_CACHE[key] = compiled
    return compiled


def attention_residual(
    v_stack: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    gamma: torch.Tensor,
    alpha: float,
    epsilon: float = 1e-6,
    *,
    schedule_mode: str = "tuned",
) -> torch.Tensor:
    """Compute AttentionResidual on Ascend950PR NPU using a compiled SimdVF kernel."""
    if schedule_mode not in ("baseline", "tuned"):
        raise ValueError(f"unsupported schedule_mode {schedule_mode!r}")
    if v_stack.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"unsupported dtype {v_stack.dtype}")
    n_depth, b_dim, t_dim, d_dim = v_stack.shape
    m_rows = b_dim * t_dim
    tl_dtype = "bfloat16" if v_stack.dtype == torch.bfloat16 else "float16"

    v_flat = v_stack.contiguous().view(n_depth * m_rows, d_dim)
    x_flat = x.contiguous().view(m_rows, d_dim)
    w_flat = w.contiguous().view(d_dim)
    g_flat = gamma.contiguous().view(d_dim)

    if (d_dim % 64) != 0:
        selector_fn, builder_fn = _select_ar_unaligned_config, _build_ar_unaligned_func
    elif m_rows < 56:
        selector_fn, builder_fn = _select_ar_small_decode_config, _build_ar_small_decode_func
    else:
        selector_fn, builder_fn = _select_ar_aligned_config, _build_ar_aligned_func

    dispatch_key = (
        _callable_cache_key(selector_fn),
        n_depth,
        d_dim,
        tl_dtype,
        schedule_mode,
        m_rows,
    )
    cfg = _DISPATCH_CACHE.get(dispatch_key)
    if cfg is None:
        cfg = dict(selector_fn(n_depth, d_dim, tl_dtype, schedule_mode, m_rows=m_rows))
        _DISPATCH_CACHE[dispatch_key] = cfg
    tile_m = int(cfg["tile_m"])
    pass_configs, compile_flags = _split_pass_configs_and_flags(
        cfg.get("pass_configs"), cfg.get("compile_flags")
    )
    raw_ub = cfg.get("static_ub_bytes")
    static_ub_bytes = (
        int(raw_ub)
        if isinstance(raw_ub, int) and not isinstance(raw_ub, bool)
        else _estimate_ar_static_ub_bytes(
            n_depth,
            d_dim,
            tile_m,
            schedule_mode,
            tl_dtype,
            num_stages=int(cfg.get("num_stages", 0) or 0),
            d_subtile=int(cfg.get("d_subtile", 0) or 0),
        )
    )
    if static_ub_bytes <= _VF_STACK_RESERVE_UB_THRESHOLD:
        compile_flags = tuple(f for f in compile_flags if f != _VF_STACK_DISABLE_FLAG)
    elif schedule_mode == "tuned" and _VF_STACK_DISABLE_FLAG not in compile_flags:
        compile_flags = (*compile_flags, _VF_STACK_DISABLE_FLAG)
    raw_bk = cfg.get("builder_kwargs")
    extra_builder_kwargs = dict(raw_bk) if isinstance(raw_bk, dict) else {}
    if "num_stages" in cfg and "num_stages" not in extra_builder_kwargs:
        extra_builder_kwargs["num_stages"] = int(cfg["num_stages"])
    if "d_subtile" in cfg and "d_subtile" not in extra_builder_kwargs:
        extra_builder_kwargs["d_subtile"] = int(cfg["d_subtile"])
    kernel_fn = _get_kernel(
        n_depth,
        d_dim,
        tl_dtype,
        tile_m,
        schedule_mode,
        builder_fn=builder_fn,
        pass_configs=pass_configs,
        compile_flags=compile_flags,
        extra_builder_kwargs=extra_builder_kwargs or None,
    )
    y_flat = kernel_fn(v_flat, x_flat, w_flat, g_flat, float(alpha), float(epsilon))
    return y_flat.view(b_dim, t_dim, d_dim)


kernel = attention_residual

__all__ = ["attention_residual", "kernel"]
