"""TileLang Ascend950PR Pure RegBase SimdVF / SimtVF kernel implementation for Softmax."""

import contextlib
import ctypes
import os
from pathlib import Path
import tempfile

import tilelang
import tilelang.ascend.language as T
from tilelang.contrib.bisheng import compile_ascend
import torch
import torch.nn.functional as F

from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS

_KERNEL_CACHE: dict[tuple[object, ...], object] = {}


def _ascend_target_ctx():
    try:
        from tilelang import tvm
        from tilelang.backend.module import create_backend_context

        if tvm.target.Target.current(allow_none=True) is None:
            return create_backend_context("ascend").target
    except Exception:
        pass
    return contextlib.nullcontext()


def _select_softmax_config(
    d_dim: int,
    tl_dtype: str = "bfloat16",
    *,
    m_rows: int = 1024,
    inner_dim: int = 1,
    schedule_mode: str = "tuned",
) -> dict[str, object]:
    """Select row packing, inner-axis tiling, and multi-buffer config for Softmax."""
    elem_bytes = 4 if tl_dtype == "float32" else 2
    ub_budget = 256 * 1024 - 4096 if schedule_mode == "tuned" else 192 * 1024

    if inner_dim > 1:
        w_tile = 64
        # Check if full column slice (d_dim, 64) fits in UB for single-load softmax
        stages_ub = 2 if schedule_mode == "tuned" else 1
        need_s2 = d_dim * w_tile * (2 * elem_bytes * stages_ub + 4)
        need_s1 = d_dim * w_tile * (2 * elem_bytes + 4)
        if need_s2 <= ub_budget:
            return {
                "regime": "inner_ub_single_load",
                "tile_m": 1,
                "d_tile": int(d_dim),
                "num_stages": stages_ub,
                "d_dim": int(d_dim),
                "inner_dim": int(inner_dim),
                "tl_dtype": tl_dtype,
                "pass_configs": {},
                "compile_flags": list(PURE_SIMDVF_UBUF_FLAGS) if schedule_mode == "tuned" else [],
            }
        if need_s1 <= ub_budget:
            return {
                "regime": "inner_ub_single_load",
                "tile_m": 1,
                "d_tile": int(d_dim),
                "num_stages": 1,
                "d_dim": int(d_dim),
                "inner_dim": int(inner_dim),
                "tl_dtype": tl_dtype,
                "pass_configs": {},
                "compile_flags": list(PURE_SIMDVF_UBUF_FLAGS) if schedule_mode == "tuned" else [],
            }
        d_blk = 256 if tl_dtype != "float32" else 128
        d_blk = min(int(d_dim), d_blk)
        return {
            "regime": "inner_online_2pass",
            "tile_m": 1,
            "d_tile": d_blk,
            "num_stages": 2 if schedule_mode == "tuned" else 1,
            "d_dim": int(d_dim),
            "inner_dim": int(inner_dim),
            "tl_dtype": tl_dtype,
            "pass_configs": {},
            "compile_flags": list(PURE_SIMDVF_UBUF_FLAGS) if schedule_mode == "tuned" else [],
        }

    if d_dim == 2:
        tile_m = 2048 if (tl_dtype != "float32" and m_rows >= 56 * 2048) else (
            1024 if m_rows >= 56 * 1024 else max(128, ((max(1, m_rows // 112) + 127) // 128) * 128)
        )
        stages = 2 if (schedule_mode == "tuned" and m_rows >= 56 * 2) else 1
        return {
            "regime": "ultra_narrow_pair_vec",
            "tile_m": tile_m,
            "d_tile": int(d_dim),
            "num_stages": stages,
            "d_dim": int(d_dim),
            "inner_dim": 1,
            "tl_dtype": tl_dtype,
            "pass_configs": {},
            "compile_flags": list(PURE_SIMDVF_UBUF_FLAGS) if schedule_mode == "tuned" else [],
        }

    if d_dim <= 16:
        tile_m = 256 if m_rows >= 56 * 256 else max(8, min(128, max(8, m_rows // 112)))
        stages = 2 if (schedule_mode == "tuned" and m_rows >= 56 * 2) else 1
        return {
            "regime": "narrow_simt_packed",
            "tile_m": tile_m,
            "d_tile": int(d_dim),
            "num_stages": stages,
            "d_dim": int(d_dim),
            "inner_dim": 1,
            "tl_dtype": tl_dtype,
            "pass_configs": {"tl.disable_data_race_check": True},
            "compile_flags": [],
        }

    d_alloc = ((max(1, int(d_dim)) + 63) // 64) * 64
    stages = 2 if (schedule_mode == "tuned" and m_rows >= 56 * 2) else 1
    row_bytes = 2 * stages * d_alloc * elem_bytes
    exp_bytes = 0 if d_dim <= 512 else (d_alloc * 4)
    if (row_bytes + exp_bytes) > ub_budget:
        stages = 1
        row_bytes = 2 * d_alloc * elem_bytes
    max_rows = max(1, min(128, (ub_budget - exp_bytes) // max(1, row_bytes)))
    if (d_dim % 64) != 0:
        max_rows = min(max_rows, 64)
    if max_rows > 1 and m_rows >= 112 and m_rows < 56 * max_rows * 2:
        max_rows = max(1, min(max_rows, max(1, m_rows // 112)))
    elif max_rows > 1 and m_rows < 56 * max_rows:
        max_rows = max(1, min(max_rows, max(1, m_rows // 56)))
    regime = (
        "reg_resident_packed"
        if d_dim <= 512
        else ("aligned_packed_regbase" if (d_dim % 64) == 0 else "unaligned_predicated")
    )
    return {
        "regime": regime,
        "tile_m": max_rows,
        "d_tile": int(d_dim),
        "num_stages": stages,
        "d_dim": int(d_dim),
        "inner_dim": 1,
        "tl_dtype": tl_dtype,
        "pass_configs": {},
        "compile_flags": list(PURE_SIMDVF_UBUF_FLAGS) if schedule_mode == "tuned" else [],
    }


def _build_softmax_prim_func(
    d_dim: int = 1024,
    tl_dtype: str = "bfloat16",
    tile_m: int = 2,
    d_tile: int | None = None,
    num_stages: int = 2,
    inner_dim: int = 1,
    regime: str | None = None,
    schedule_mode: str = "tuned",
    **_extra_kwargs,
):
    """Build Ascend 950 Pure RegBase SimdVF / SimtVF @T.prim_func for Softmax with inf/nan-safe math."""
    if d_dim <= 0:
        raise ValueError(f"d_dim must be positive, got {d_dim}")
    if tl_dtype not in ("float16", "bfloat16", "float32"):
        raise ValueError(f"unsupported dtype {tl_dtype!r}")

    num_cores = 56
    stages = max(1, min(2, int(num_stages)))
    pipe_ann = {"enable_offset": True} if schedule_mode == "tuned" else None

    if inner_dim > 1:
        outer_dyn = T.dynamic("outer_dim")
        inner_dyn = T.dynamic("inner_dim")
        w_tile = 64

        if regime == "inner_ub_single_load" or (d_tile is not None and int(d_tile) == d_dim and d_dim <= 512):
            with _ascend_target_ctx():

                @T.prim_func
                def main(
                    X: T.Tensor((outer_dyn, d_dim, inner_dyn), tl_dtype),
                    Y: T.Tensor((outer_dyn, d_dim, inner_dyn), tl_dtype),
                ):
                    with T.Kernel(num_cores) as cid:
                        x_ub = T.alloc_shared((d_dim, w_tile), tl_dtype)
                        exp_ub = T.alloc_shared((d_dim, w_tile), "float32")
                        y_ub = T.alloc_shared((d_dim, w_tile), tl_dtype)
                        T.annotate_buffer_versions({x_ub: stages, exp_ub: 1, y_ub: stages})
                        if schedule_mode == "tuned":
                            T.annotate_unlimit_memory("shared")

                        w_tiles = (inner_dyn + w_tile - 1) // w_tile
                        total_tasks = outer_dyn * w_tiles
                        base_t = total_tasks // num_cores
                        rem_t = total_tasks % num_cores
                        t_start = cid * base_t + T.min(cid, rem_t)
                        t_count = base_t + T.if_then_else(cid < rem_t, 1, 0)

                        for lt in T.Pipelined(t_count, num_stages=stages, annotations=pipe_ann):
                            tid = t_start + lt
                            o_idx = tid // w_tiles
                            wt = tid % w_tiles
                            j0 = wt * w_tile
                            w_cur = T.min(w_tile, inner_dyn - j0)
                            T.copy(X[o_idx, :d_dim, j0 : j0 + w_cur], x_ub[:, :w_cur])
                            with T.SimdVF():
                                mw = T.simd.update_mask(w_cur, width=32)
                                m16w = T.simd.update_mask(w_cur * 2, width=16)
                                v_one = T.simd.vdup(T.float32(1.0), "float32", mask=mw)
                                max_acc = T.simd.alloc_local((1,), "float32")
                                sum_acc = T.simd.alloc_local((1,), "float32")
                                max_acc[0] = T.simd.vdup(T.float32(-1.0e38), "float32", mask=mw)
                                sum_acc[0] = T.simd.vdup(T.float32(0.0), "float32", mask=mw)
                                for dd in T.serial(d_dim):
                                    if tl_dtype == "float32":
                                        xf = T.simd.vld(x_ub[dd, 0], dist="NORM_B32")
                                    else:
                                        xf = T.simd.vcvt(T.simd.vld(x_ub[dd, 0], dist="US_B16"), "float32", mask=m16w, part=0)
                                    max_acc[0] = T.simd.vmax(max_acc[0], xf, mask=mw)
                                max_v = max_acc[0]
                                for dd in T.serial(d_dim):
                                    if tl_dtype == "float32":
                                        xf = T.simd.vld(x_ub[dd, 0], dist="NORM_B32")
                                    else:
                                        xf = T.simd.vcvt(T.simd.vld(x_ub[dd, 0], dist="US_B16"), "float32", mask=m16w, part=0)
                                    ev = T.simd.vexp(T.simd.vsub(xf, max_v, mask=mw), mask=mw)
                                    T.simd.vsts(exp_ub[dd, 0], ev, mask=mw, dist="NORM_B32")
                                    sum_acc[0] = T.simd.vadd(sum_acc[0], ev, mask=mw)
                                T.simd.mem_bar("VST_VLD")
                                inv_sum = T.simd.vdiv(v_one, sum_acc[0], mask=mw, precision="ftz_true")
                                for dd in T.serial(d_dim):
                                    yf = T.simd.vmul(T.simd.vld(exp_ub[dd, 0], dist="NORM_B32"), inv_sum, mask=mw)
                                    if tl_dtype == "float32":
                                        T.simd.vsts(y_ub[dd, 0], yf, mask=mw, dist="NORM_B32")
                                    else:
                                        y16 = T.simd.vcvt(yf, tl_dtype, mask=m16w, part=0)
                                        T.simd.vsts(y_ub[dd, 0], y16, mask=mw, dist="PK_B32", extent=64)
                            T.copy(y_ub[:, :w_cur], Y[o_idx, :d_dim, j0 : j0 + w_cur])

            return main

        # Online 2-pass softmax for tall d_dim (> 512) with 2D burst DMA
        d_blk = min(d_dim, int(d_tile) if d_tile else 256)
        d_blocks = (d_dim + d_blk - 1) // d_blk

        with _ascend_target_ctx():

            @T.prim_func
            def main(
                X: T.Tensor((outer_dyn, d_dim, inner_dyn), tl_dtype),
                Y: T.Tensor((outer_dyn, d_dim, inner_dyn), tl_dtype),
            ):
                with T.Kernel(num_cores) as cid:
                    x1_ub = T.alloc_shared((d_blk, w_tile), tl_dtype)
                    x2_ub = T.alloc_shared((d_blk, w_tile), tl_dtype)
                    y_ub = T.alloc_shared((d_blk, w_tile), tl_dtype)
                    max_ub = T.alloc_shared((w_tile,), "float32")
                    sum_ub = T.alloc_shared((w_tile,), "float32")
                    T.annotate_buffer_versions(
                        {x1_ub: stages, x2_ub: stages, y_ub: stages, max_ub: 1, sum_ub: 1}
                    )
                    if schedule_mode == "tuned":
                        T.annotate_unlimit_memory("shared")

                    w_tiles = (inner_dyn + w_tile - 1) // w_tile
                    total_tasks = outer_dyn * w_tiles
                    base_t = total_tasks // num_cores
                    rem_t = total_tasks % num_cores
                    t_start = cid * base_t + T.min(cid, rem_t)
                    t_count = base_t + T.if_then_else(cid < rem_t, 1, 0)

                    for lt in T.serial(t_count):
                        tid = t_start + lt
                        o_idx = tid // w_tiles
                        wt = tid % w_tiles
                        j0 = wt * w_tile
                        w_cur = T.min(w_tile, inner_dyn - j0)

                        with T.SimdVF():
                            mw = T.simd.update_mask(w_cur, width=32)
                            T.simd.vsts(max_ub[0], T.simd.vdup(T.float32(-1.0e38), "float32", mask=mw), mask=mw, dist="NORM_B32")
                            T.simd.vsts(sum_ub[0], T.simd.vdup(T.float32(0.0), "float32", mask=mw), mask=mw, dist="NORM_B32")

                        # Pass 1: online max + rescaled exp sum in one read pass
                        for db in T.Pipelined(d_blocks, num_stages=stages, annotations=pipe_ann):
                            d0 = db * d_blk
                            d_cur = T.min(d_blk, d_dim - d0)
                            T.copy(X[o_idx, d0 : d0 + d_cur, j0 : j0 + w_cur], x1_ub[:d_cur, :w_cur])
                            with T.SimdVF():
                                mw = T.simd.update_mask(w_cur, width=32)
                                m16w = T.simd.update_mask(w_cur * 2, width=16)
                                m_old = T.simd.vld(max_ub[0], dist="NORM_B32")
                                s_old = T.simd.vld(sum_ub[0], dist="NORM_B32")
                                m_cur = T.simd.alloc_local((1,), "float32")
                                m_cur[0] = m_old
                                for dd in T.serial(d_cur):
                                    if tl_dtype == "float32":
                                        xf = T.simd.vld(x1_ub[dd, 0], dist="NORM_B32")
                                    else:
                                        xf = T.simd.vcvt(T.simd.vld(x1_ub[dd, 0], dist="US_B16"), "float32", mask=m16w, part=0)
                                    m_cur[0] = T.simd.vmax(m_cur[0], xf, mask=mw)
                                m_new = m_cur[0]
                                alpha = T.simd.vexp(T.simd.vsub(m_old, m_new, mask=mw), mask=mw)
                                s_acc = T.simd.alloc_local((1,), "float32")
                                s_acc[0] = T.simd.vmul(s_old, alpha, mask=mw)
                                for dd in T.serial(d_cur):
                                    if tl_dtype == "float32":
                                        xf = T.simd.vld(x1_ub[dd, 0], dist="NORM_B32")
                                    else:
                                        xf = T.simd.vcvt(T.simd.vld(x1_ub[dd, 0], dist="US_B16"), "float32", mask=m16w, part=0)
                                    ev = T.simd.vexp(T.simd.vsub(xf, m_new, mask=mw), mask=mw)
                                    s_acc[0] = T.simd.vadd(s_acc[0], ev, mask=mw)
                                T.simd.vsts(max_ub[0], m_new, mask=mw, dist="NORM_B32")
                                T.simd.vsts(sum_ub[0], s_acc[0], mask=mw, dist="NORM_B32")

                        # Pass 2: normalize and write Y
                        for db in T.Pipelined(d_blocks, num_stages=stages, annotations=pipe_ann):
                            d0 = db * d_blk
                            d_cur = T.min(d_blk, d_dim - d0)
                            T.copy(X[o_idx, d0 : d0 + d_cur, j0 : j0 + w_cur], x2_ub[:d_cur, :w_cur])
                            with T.SimdVF():
                                mw = T.simd.update_mask(w_cur, width=32)
                                m16w = T.simd.update_mask(w_cur * 2, width=16)
                                v_one = T.simd.vdup(T.float32(1.0), "float32", mask=mw)
                                max_v = T.simd.vld(max_ub[0], dist="NORM_B32")
                                inv_sum = T.simd.vdiv(v_one, T.simd.vld(sum_ub[0], dist="NORM_B32"), mask=mw, precision="ftz_true")
                                for dd in T.serial(d_cur):
                                    if tl_dtype == "float32":
                                        xf = T.simd.vld(x2_ub[dd, 0], dist="NORM_B32")
                                    else:
                                        xf = T.simd.vcvt(T.simd.vld(x2_ub[dd, 0], dist="US_B16"), "float32", mask=m16w, part=0)
                                    ev = T.simd.vexp(T.simd.vsub(xf, max_v, mask=mw), mask=mw)
                                    yf = T.simd.vmul(ev, inv_sum, mask=mw)
                                    if tl_dtype == "float32":
                                        T.simd.vsts(y_ub[dd, 0], yf, mask=mw, dist="NORM_B32")
                                    else:
                                        y16 = T.simd.vcvt(yf, tl_dtype, mask=m16w, part=0)
                                        T.simd.vsts(y_ub[dd, 0], y16, mask=mw, dist="PK_B32", extent=64)
                            T.copy(y_ub[:d_cur, :w_cur], Y[o_idx, d0 : d0 + d_cur, j0 : j0 + w_cur])

        return main

    rows = max(1, int(tile_m))
    m_rows = T.dynamic("m_rows")

    if d_dim == 2:
        rows_blk = max(128, ((rows + 127) // 128) * 128)
        with _ascend_target_ctx():

            @T.prim_func
            def main(
                X: T.Tensor((m_rows * 2,), tl_dtype),
                Y: T.Tensor((m_rows * 2,), tl_dtype),
            ):
                with T.Kernel(num_cores) as cid:
                    x_ub = T.alloc_shared((rows_blk * 2,), tl_dtype)
                    y_ub = T.alloc_shared((rows_blk * 2,), tl_dtype)
                    T.annotate_buffer_versions({x_ub: stages, y_ub: stages})
                    if schedule_mode == "tuned":
                        T.annotate_unlimit_memory("shared")

                    base_rows = m_rows // num_cores
                    rem_rows = m_rows % num_cores
                    m_start = cid * base_rows + T.min(cid, rem_rows)
                    m_count = base_rows + T.if_then_else(cid < rem_rows, 1, 0)
                    num_blocks = (m_count + rows_blk - 1) // rows_blk

                    for mb in T.Pipelined(num_blocks, num_stages=stages, annotations=pipe_ann):
                        r0 = m_start + mb * rows_blk
                        cur = T.min(rows_blk, m_count - mb * rows_blk)
                        T.copy(X[r0 * 2 : (r0 + cur) * 2], x_ub[: cur * 2])
                        with T.SimdVF():
                            m32 = T.simd.pset(32, "PAT_ALL")
                            m16 = T.simd.pset(16, "PAT_ALL")
                            if tl_dtype == "float32":
                                n_chunks = (cur + 63) // 64
                                for c in T.serial(n_chunks):
                                    off = c * 128
                                    x0, x1 = T.simd.vld2(x_ub[off], dist="DINTLV_B32")
                                    mx = T.simd.vmax(x0, x1, mask=m32)
                                    e0 = T.simd.vexp(T.simd.vsub(x0, mx, mask=m32), mask=m32)
                                    e1 = T.simd.vexp(T.simd.vsub(x1, mx, mask=m32), mask=m32)
                                    sm = T.simd.vadd(e0, e1, mask=m32)
                                    y0 = T.simd.vdiv(e0, sm, mask=m32, precision="ftz_true")
                                    y1 = T.simd.vdiv(e1, sm, mask=m32, precision="ftz_true")
                                    T.simd.vst2(y_ub[off], y0, y1)
                            else:
                                m16_z = T.simd.update_mask(0, width=16)
                                m_even16 = T.simd.pair_get(T.simd.pintlv(m16, m16_z, width=16), 0)
                                n_chunks = (cur + 127) // 128
                                for c in T.serial(n_chunks):
                                    off = c * 256
                                    x0_16, x1_16 = T.simd.vld2(x_ub[off], dist="DINTLV_B16")
                                    x0_e = T.simd.vcvt(x0_16, "float32", mask=m16, part=0)
                                    x0_o = T.simd.vcvt(x0_16, "float32", mask=m16, part=1)
                                    x1_e = T.simd.vcvt(x1_16, "float32", mask=m16, part=0)
                                    x1_o = T.simd.vcvt(x1_16, "float32", mask=m16, part=1)
                                    mx_e = T.simd.vmax(x0_e, x1_e, mask=m32)
                                    mx_o = T.simd.vmax(x0_o, x1_o, mask=m32)
                                    e0_e = T.simd.vexp(T.simd.vsub(x0_e, mx_e, mask=m32), mask=m32)
                                    e1_e = T.simd.vexp(T.simd.vsub(x1_e, mx_e, mask=m32), mask=m32)
                                    e0_o = T.simd.vexp(T.simd.vsub(x0_o, mx_o, mask=m32), mask=m32)
                                    e1_o = T.simd.vexp(T.simd.vsub(x1_o, mx_o, mask=m32), mask=m32)
                                    sm_e = T.simd.vadd(e0_e, e1_e, mask=m32)
                                    sm_o = T.simd.vadd(e0_o, e1_o, mask=m32)
                                    y0_e = T.simd.vcvt(T.simd.vdiv(e0_e, sm_e, mask=m32, precision="ftz_true"), tl_dtype, mask=m16, part=0)
                                    y0_o = T.simd.vcvt(T.simd.vdiv(e0_o, sm_o, mask=m32, precision="ftz_true"), tl_dtype, mask=m16, part=1)
                                    y1_e = T.simd.vcvt(T.simd.vdiv(e1_e, sm_e, mask=m32, precision="ftz_true"), tl_dtype, mask=m16, part=0)
                                    y1_o = T.simd.vcvt(T.simd.vdiv(e1_o, sm_o, mask=m32, precision="ftz_true"), tl_dtype, mask=m16, part=1)
                                    y0_16 = T.simd.vsel(y0_e, y0_o, m_even16)
                                    y1_16 = T.simd.vsel(y1_e, y1_o, m_even16)
                                    T.simd.vst2(y_ub[off], y0_16, y1_16)
                        T.copy(y_ub[: cur * 2], Y[r0 * 2 : (r0 + cur) * 2])

        return main

    if d_dim <= 16:
        rows_blk = max(8, rows)
        simt_threads = 512
        iters = (rows_blk + simt_threads - 1) // simt_threads
        with _ascend_target_ctx():

            @T.prim_func
            def main(
                X: T.Tensor((m_rows * d_dim,), tl_dtype),
                Y: T.Tensor((m_rows * d_dim,), tl_dtype),
            ):
                with T.Kernel(num_cores) as cid:
                    x_ub = T.alloc_shared((rows_blk * d_dim,), tl_dtype)
                    y_ub = T.alloc_shared((rows_blk * d_dim,), tl_dtype)
                    T.annotate_buffer_versions({x_ub: stages, y_ub: stages})

                    base_rows = m_rows // num_cores
                    rem_rows = m_rows % num_cores
                    m_start = cid * base_rows + T.min(cid, rem_rows)
                    m_count = base_rows + T.if_then_else(cid < rem_rows, 1, 0)
                    num_blocks = (m_count + rows_blk - 1) // rows_blk

                    for mb in T.Pipelined(num_blocks, num_stages=stages):
                        r0 = m_start + mb * rows_blk
                        cur = T.min(rows_blk, m_count - mb * rows_blk)
                        T.copy(X[r0 * d_dim : (r0 + cur) * d_dim], x_ub[: cur * d_dim])
                        with T.SimtVF(threads=simt_threads):
                            for tid in T.Parallel(simt_threads):
                                for idx in T.serial(iters):
                                    r = idx * simt_threads + tid
                                    if r < cur:
                                        mx_b = T.alloc_local((1,), "float32")
                                        sm_b = T.alloc_local((1,), "float32")
                                        mx_b[0] = T.cast(x_ub[r * d_dim], "float32")
                                        for dd in T.serial(1, d_dim):
                                            xv = T.cast(x_ub[r * d_dim + dd], "float32")
                                            mx_b[0] = T.max(mx_b[0], xv)
                                        mx_v = mx_b[0]
                                        sm_b[0] = T.float32(0.0)
                                        for dd in T.serial(d_dim):
                                            sm_b[0] = sm_b[0] + T.exp(T.cast(x_ub[r * d_dim + dd], "float32") - mx_v)
                                        inv_sm = T.float32(1.0) / sm_b[0]
                                        for dd in T.serial(d_dim):
                                            ev = T.exp(T.cast(x_ub[r * d_dim + dd], "float32") - mx_v)
                                            y_ub[r * d_dim + dd] = T.cast(ev * inv_sm, tl_dtype)
                        T.copy(y_ub[: cur * d_dim], Y[r0 * d_dim : (r0 + cur) * d_dim])

        return main

    full_chunks = d_dim // 64
    rem = d_dim % 64
    d_alloc = ((d_dim + 63) // 64) * 64
    use_reg_exp = d_dim <= 512

    @T.macro
    def _softmax_row_reg(x_ub, y_ub, slot):
        m32 = T.simd.pset(32, "PAT_ALL")
        m16 = T.simd.pset(16, "PAT_ALL")
        v_neg_inf = T.simd.vdup(T.float32(-1.0e38), "float32", mask=m32)
        v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
        v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
        max_acc = T.simd.alloc_local((1,), "float32")
        sum_acc = T.simd.alloc_local((1,), "float32")
        total_c = full_chunks + (1 if rem > 0 else 0)
        ev_reg = T.simd.alloc_local((total_c,), "float32")
        max_acc[0] = v_neg_inf
        sum_acc[0] = v_zero

        for c in T.unroll(full_chunks):
            off = c * 64
            if tl_dtype == "float32":
                xf = T.simd.vld(x_ub[slot, off], dist="NORM_B32")
            else:
                xf = T.simd.vcvt(T.simd.vld(x_ub[slot, off], dist="US_B16"), "float32", mask=m16, part=0)
            ev_reg[c] = xf
            max_acc[0] = T.simd.vmax(max_acc[0], xf, mask=m32)
        if rem > 0:
            off = full_chunks * 64
            m32_t = T.simd.update_mask(rem, width=32)
            if tl_dtype == "float32":
                xf_t = T.simd.vld(x_ub[slot, off], dist="NORM_B32")
            else:
                m16_t = T.simd.update_mask(rem * 2, width=16)
                xf_t = T.simd.vcvt(T.simd.vld(x_ub[slot, off], dist="US_B16"), "float32", mask=m16_t, part=0)
            ev_reg[full_chunks] = xf_t
            max_acc[0] = T.simd.vmax(max_acc[0], xf_t, mask=m32_t, mode="MODE_MERGING")

        row_max = T.simd.vdupv(T.simd.vcmax(max_acc[0], mask=m32), mask=m32, pos="POS_LOWEST")

        for c in T.unroll(full_chunks):
            ev = T.simd.vexp(T.simd.vsub(ev_reg[c], row_max, mask=m32), mask=m32)
            ev_reg[c] = ev
            sum_acc[0] = T.simd.vadd(sum_acc[0], ev, mask=m32)
        if rem > 0:
            m32_t = T.simd.update_mask(rem, width=32)
            ev_t = T.simd.vexp(T.simd.vsub(ev_reg[full_chunks], row_max, mask=m32_t), mask=m32_t)
            ev_reg[full_chunks] = ev_t
            sum_acc[0] = T.simd.vadd(sum_acc[0], ev_t, mask=m32_t, mode="MODE_MERGING")

        row_sum = T.simd.vdupv(T.simd.vcadd(sum_acc[0], mask=m32), mask=m32, pos="POS_LOWEST")
        inv_sum = T.simd.vdiv(v_one, row_sum, mask=m32, precision="ftz_true")

        for c in T.unroll(full_chunks):
            off = c * 64
            yf = T.simd.vmul(ev_reg[c], inv_sum, mask=m32)
            if tl_dtype == "float32":
                T.simd.vsts(y_ub[slot, off], yf, mask=m32, dist="NORM_B32")
            else:
                y16 = T.simd.vcvt(yf, tl_dtype, mask=m16, part=0)
                T.simd.vsts(y_ub[slot, off], y16, mask=m32, dist="PK_B32", extent=64)
        if rem > 0:
            off = full_chunks * 64
            m32_t = T.simd.update_mask(rem, width=32)
            yf_t = T.simd.vmul(ev_reg[full_chunks], inv_sum, mask=m32_t)
            if tl_dtype == "float32":
                T.simd.vsts(y_ub[slot, off], yf_t, mask=m32_t, dist="NORM_B32")
            else:
                m16_t = T.simd.update_mask(rem * 2, width=16)
                y16_t = T.simd.vcvt(yf_t, tl_dtype, mask=m16_t, part=0)
                T.simd.vsts(y_ub[slot, off], y16_t, mask=m32_t, dist="PK_B32", extent=64)

    @T.macro
    def _softmax_row(x_ub, exp_ub, y_ub, slot):
        m32 = T.simd.pset(32, "PAT_ALL")
        m16 = T.simd.pset(16, "PAT_ALL")
        v_neg_inf = T.simd.vdup(T.float32(-1.0e38), "float32", mask=m32)
        v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
        v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
        max_acc = T.simd.alloc_local((1,), "float32")
        sum_acc = T.simd.alloc_local((1,), "float32")
        max_acc[0] = v_neg_inf
        sum_acc[0] = v_zero

        for c in T.serial(full_chunks):
            off = c * 64
            if tl_dtype == "float32":
                xf = T.simd.vld(x_ub[slot, off], dist="NORM_B32")
            else:
                xf = T.simd.vcvt(T.simd.vld(x_ub[slot, off], dist="US_B16"), "float32", mask=m16, part=0)
            max_acc[0] = T.simd.vmax(max_acc[0], xf, mask=m32)
        if rem > 0:
            off = full_chunks * 64
            m32_t = T.simd.update_mask(rem, width=32)
            if tl_dtype == "float32":
                xf_t = T.simd.vld(x_ub[slot, off], dist="NORM_B32")
            else:
                m16_t = T.simd.update_mask(rem * 2, width=16)
                xf_t = T.simd.vcvt(T.simd.vld(x_ub[slot, off], dist="US_B16"), "float32", mask=m16_t, part=0)
            max_acc[0] = T.simd.vmax(max_acc[0], xf_t, mask=m32_t, mode="MODE_MERGING")

        row_max = T.simd.vdupv(T.simd.vcmax(max_acc[0], mask=m32), mask=m32, pos="POS_LOWEST")

        for c in T.serial(full_chunks):
            off = c * 64
            if tl_dtype == "float32":
                xf = T.simd.vld(x_ub[slot, off], dist="NORM_B32")
            else:
                xf = T.simd.vcvt(T.simd.vld(x_ub[slot, off], dist="US_B16"), "float32", mask=m16, part=0)
            ev = T.simd.vexp(T.simd.vsub(xf, row_max, mask=m32), mask=m32)
            T.simd.vsts(exp_ub[0, off], ev, mask=m32, dist="NORM_B32")
            sum_acc[0] = T.simd.vadd(sum_acc[0], ev, mask=m32)
        if rem > 0:
            off = full_chunks * 64
            m32_t = T.simd.update_mask(rem, width=32)
            if tl_dtype == "float32":
                xf_t = T.simd.vld(x_ub[slot, off], dist="NORM_B32")
            else:
                m16_t = T.simd.update_mask(rem * 2, width=16)
                xf_t = T.simd.vcvt(T.simd.vld(x_ub[slot, off], dist="US_B16"), "float32", mask=m16_t, part=0)
            ev_t = T.simd.vexp(T.simd.vsub(xf_t, row_max, mask=m32_t), mask=m32_t)
            T.simd.vsts(exp_ub[0, off], ev_t, mask=m32_t, dist="NORM_B32")
            sum_acc[0] = T.simd.vadd(sum_acc[0], ev_t, mask=m32_t, mode="MODE_MERGING")

        T.simd.mem_bar("VST_VLD")
        row_sum = T.simd.vdupv(T.simd.vcadd(sum_acc[0], mask=m32), mask=m32, pos="POS_LOWEST")
        inv_sum = T.simd.vdiv(v_one, row_sum, mask=m32, precision="ftz_true")

        for c in T.serial(full_chunks):
            off = c * 64
            ev = T.simd.vld(exp_ub[0, off], dist="NORM_B32")
            yf = T.simd.vmul(ev, inv_sum, mask=m32)
            if tl_dtype == "float32":
                T.simd.vsts(y_ub[slot, off], yf, mask=m32, dist="NORM_B32")
            else:
                y16 = T.simd.vcvt(yf, tl_dtype, mask=m16, part=0)
                T.simd.vsts(y_ub[slot, off], y16, mask=m32, dist="PK_B32", extent=64)
        if rem > 0:
            off = full_chunks * 64
            m32_t = T.simd.update_mask(rem, width=32)
            ev_t = T.simd.vld(exp_ub[0, off], dist="NORM_B32")
            yf_t = T.simd.vmul(ev_t, inv_sum, mask=m32_t)
            if tl_dtype == "float32":
                T.simd.vsts(y_ub[slot, off], yf_t, mask=m32_t, dist="NORM_B32")
            else:
                m16_t = T.simd.update_mask(rem * 2, width=16)
                y16_t = T.simd.vcvt(yf_t, tl_dtype, mask=m16_t, part=0)
                T.simd.vsts(y_ub[slot, off], y16_t, mask=m32_t, dist="PK_B32", extent=64)

    with _ascend_target_ctx():

        @T.prim_func
        def main(
            X: T.Tensor((m_rows, d_dim), tl_dtype),
            Y: T.Tensor((m_rows, d_dim), tl_dtype),
        ):
            with T.Kernel(num_cores) as cid:
                x_ub = T.alloc_shared((rows, d_alloc), tl_dtype)
                y_ub = T.alloc_shared((rows, d_alloc), tl_dtype)
                if use_reg_exp:
                    T.annotate_buffer_versions({x_ub: stages, y_ub: stages})
                else:
                    exp_ub = T.alloc_shared((1, d_alloc), "float32")
                    T.annotate_buffer_versions({x_ub: stages, exp_ub: 1, y_ub: stages})
                if schedule_mode == "tuned":
                    T.annotate_unlimit_memory("shared")

                base_rows = m_rows // num_cores
                rem_rows = m_rows % num_cores
                m_start = cid * base_rows + T.min(cid, rem_rows)
                m_count = base_rows + T.if_then_else(cid < rem_rows, 1, 0)
                num_blocks = (m_count + rows - 1) // rows

                for mb in T.Pipelined(num_blocks, num_stages=stages, annotations=pipe_ann):
                    m0 = m_start + mb * rows
                    m_cur = T.min(rows, m_count - mb * rows)
                    T.copy(X[m0 : m0 + m_cur, :d_dim], x_ub[:m_cur, :d_dim])
                    with T.SimdVF():
                        for rr in T.serial(m_cur):
                            if use_reg_exp:
                                _softmax_row_reg(x_ub, y_ub, rr)
                            else:
                                _softmax_row(x_ub, exp_ub, y_ub, rr)
                    T.copy(y_ub[:m_cur, :d_dim], Y[m0 : m0 + m_cur, :d_dim])

    return main


def _get_kernel(d_dim: int, tl_dtype: str, m_rows: int, inner_dim: int, schedule_mode: str):
    cfg = _select_softmax_config(
        d_dim, tl_dtype, m_rows=m_rows, inner_dim=inner_dim, schedule_mode=schedule_mode
    )
    key = (
        d_dim,
        tl_dtype,
        int(cfg["tile_m"]),
        int(cfg["d_tile"]),
        int(cfg["num_stages"]),
        1 if inner_dim == 1 else 2,
        str(cfg["regime"]),
        schedule_mode,
    )
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        with _ascend_target_ctx():
            prog = _build_softmax_prim_func(
                d_dim=d_dim,
                tl_dtype=tl_dtype,
                tile_m=int(cfg["tile_m"]),
                d_tile=int(cfg["d_tile"]),
                num_stages=int(cfg["num_stages"]),
                inner_dim=inner_dim,
                regime=str(cfg["regime"]),
                schedule_mode=schedule_mode,
            )
        compiled = tilelang.compile(
            prog,
            target="ascend",
            out_idx=[1],
            pass_configs=cfg.get("pass_configs") or None,
            compile_flags=cfg.get("compile_flags") or None,
        )
        _KERNEL_CACHE[key] = compiled
    return compiled


_SOFTMAX_TRACKB_SRC = r"""
#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/simd_inst.h>

template <typename InT>
__simd_callee__ inline vector_f32 vf_load_as_f32(__ubuf__ InT* ptr, vector_bool m16) {
  if constexpr (sizeof(InT) == 4) {
    return simd_inst::vlds_norm<float>((__ubuf__ float*)ptr, 0);
  } else {
    return simd_inst::vcvt<float>(simd_inst::vlds_upsample<InT>(ptr, 0), m16, PART_EVEN, MODE_ZEROING);
  }
}

template <typename OutT>
__simd_callee__ inline void vf_store_from_f32(__ubuf__ OutT* ptr, vector_f32 vf, vector_bool m32, vector_bool m16) {
  if constexpr (sizeof(OutT) == 4) {
    simd_inst::vsts_norm(vf, (__ubuf__ float*)ptr, 0, m32);
  } else {
    auto out16 = simd_inst::vcvt<OutT>(vf, m16, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
    simd_inst::vsts_pack_b32(out16, ptr, 0, m32);
  }
}

template <int D_DIM, int D_UB_STRIDE, int IS_REG, typename InT>
__simd_vf__ inline void vf_softmax_last_rows(__ubuf__ InT* x_ub, __ubuf__ float* exp_ub, int cur_rows) {
  constexpr int FULL_CHUNKS = D_DIM / 64;
  constexpr int REM = D_DIM % 64;
  constexpr int TOTAL_CHUNKS = FULL_CHUNKS + (REM > 0 ? 1 : 0);
  vector_bool m32 = asc_create_mask_b32(PAT_ALL);
  vector_bool m16 = asc_create_mask_b16(PAT_ALL);
  vector_bool m32_t = (REM > 0) ? simd_inst::update_mask_b32((uint32_t)REM) : m32;
  vector_bool m16_t = (REM > 0) ? simd_inst::update_mask_b16((uint32_t)(REM * 2)) : m16;
  auto v_one = simd_inst::vdup(1.0f, m32, MODE_ZEROING);
  auto v_zero = simd_inst::vdup(0.0f, m32, MODE_ZEROING);
  auto v_ninf = simd_inst::vdup(-1.0e38f, m32, MODE_ZEROING);

  int r = 0;
  if constexpr (IS_REG && TOTAL_CHUNKS <= 4) {
    for (; r + 4 <= cur_rows; r += 4) {
      __ubuf__ InT* p0 = x_ub + r * D_UB_STRIDE;
      __ubuf__ InT* p1 = x_ub + (r + 1) * D_UB_STRIDE;
      __ubuf__ InT* p2 = x_ub + (r + 2) * D_UB_STRIDE;
      __ubuf__ InT* p3 = x_ub + (r + 3) * D_UB_STRIDE;
      vector_f32 rg0[TOTAL_CHUNKS], rg1[TOTAL_CHUNKS], rg2[TOTAL_CHUNKS], rg3[TOTAL_CHUNKS];
      vector_f32 ma0 = v_ninf, ma1 = v_ninf, ma2 = v_ninf, ma3 = v_ninf;
      for (int c = 0; c < FULL_CHUNKS; ++c) {
        int off = c * 64;
        rg0[c] = vf_load_as_f32(p0 + off, m16);
        rg1[c] = vf_load_as_f32(p1 + off, m16);
        rg2[c] = vf_load_as_f32(p2 + off, m16);
        rg3[c] = vf_load_as_f32(p3 + off, m16);
        ma0 = simd_inst::vmax(ma0, rg0[c], m32, MODE_ZEROING);
        ma1 = simd_inst::vmax(ma1, rg1[c], m32, MODE_ZEROING);
        ma2 = simd_inst::vmax(ma2, rg2[c], m32, MODE_ZEROING);
        ma3 = simd_inst::vmax(ma3, rg3[c], m32, MODE_ZEROING);
      }
      if constexpr (REM > 0) {
        int off = FULL_CHUNKS * 64;
        rg0[FULL_CHUNKS] = vf_load_as_f32(p0 + off, m16_t);
        rg1[FULL_CHUNKS] = vf_load_as_f32(p1 + off, m16_t);
        rg2[FULL_CHUNKS] = vf_load_as_f32(p2 + off, m16_t);
        rg3[FULL_CHUNKS] = vf_load_as_f32(p3 + off, m16_t);
        ma0 = simd_inst::vmax(ma0, ma0, rg0[FULL_CHUNKS], m32_t, MODE_MERGING);
        ma1 = simd_inst::vmax(ma1, ma1, rg1[FULL_CHUNKS], m32_t, MODE_MERGING);
        ma2 = simd_inst::vmax(ma2, ma2, rg2[FULL_CHUNKS], m32_t, MODE_MERGING);
        ma3 = simd_inst::vmax(ma3, ma3, rg3[FULL_CHUNKS], m32_t, MODE_MERGING);
      }
      auto rm0 = simd_inst::vdupv(simd_inst::vcmax(ma0, m32, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
      auto rm1 = simd_inst::vdupv(simd_inst::vcmax(ma1, m32, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
      auto rm2 = simd_inst::vdupv(simd_inst::vcmax(ma2, m32, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
      auto rm3 = simd_inst::vdupv(simd_inst::vcmax(ma3, m32, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
      vector_f32 sa0 = v_zero, sa1 = v_zero, sa2 = v_zero, sa3 = v_zero;
      for (int c = 0; c < FULL_CHUNKS; ++c) {
        rg0[c] = simd_inst::vexp(simd_inst::vsub(rg0[c], rm0, m32, MODE_ZEROING), m32, MODE_ZEROING);
        rg1[c] = simd_inst::vexp(simd_inst::vsub(rg1[c], rm1, m32, MODE_ZEROING), m32, MODE_ZEROING);
        rg2[c] = simd_inst::vexp(simd_inst::vsub(rg2[c], rm2, m32, MODE_ZEROING), m32, MODE_ZEROING);
        rg3[c] = simd_inst::vexp(simd_inst::vsub(rg3[c], rm3, m32, MODE_ZEROING), m32, MODE_ZEROING);
        sa0 = simd_inst::vadd(sa0, rg0[c], m32, MODE_ZEROING);
        sa1 = simd_inst::vadd(sa1, rg1[c], m32, MODE_ZEROING);
        sa2 = simd_inst::vadd(sa2, rg2[c], m32, MODE_ZEROING);
        sa3 = simd_inst::vadd(sa3, rg3[c], m32, MODE_ZEROING);
      }
      if constexpr (REM > 0) {
        rg0[FULL_CHUNKS] = simd_inst::vexp(simd_inst::vsub(rg0[FULL_CHUNKS], rm0, m32_t, MODE_ZEROING), m32_t, MODE_ZEROING);
        rg1[FULL_CHUNKS] = simd_inst::vexp(simd_inst::vsub(rg1[FULL_CHUNKS], rm1, m32_t, MODE_ZEROING), m32_t, MODE_ZEROING);
        rg2[FULL_CHUNKS] = simd_inst::vexp(simd_inst::vsub(rg2[FULL_CHUNKS], rm2, m32_t, MODE_ZEROING), m32_t, MODE_ZEROING);
        rg3[FULL_CHUNKS] = simd_inst::vexp(simd_inst::vsub(rg3[FULL_CHUNKS], rm3, m32_t, MODE_ZEROING), m32_t, MODE_ZEROING);
        sa0 = simd_inst::vadd(sa0, sa0, rg0[FULL_CHUNKS], m32_t, MODE_MERGING);
        sa1 = simd_inst::vadd(sa1, sa1, rg1[FULL_CHUNKS], m32_t, MODE_MERGING);
        sa2 = simd_inst::vadd(sa2, sa2, rg2[FULL_CHUNKS], m32_t, MODE_MERGING);
        sa3 = simd_inst::vadd(sa3, sa3, rg3[FULL_CHUNKS], m32_t, MODE_MERGING);
      }
      auto rs0 = simd_inst::vdupv(simd_inst::vcadd(sa0, m32, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
      auto rs1 = simd_inst::vdupv(simd_inst::vcadd(sa1, m32, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
      auto rs2 = simd_inst::vdupv(simd_inst::vcadd(sa2, m32, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
      auto rs3 = simd_inst::vdupv(simd_inst::vcadd(sa3, m32, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
      auto is0 = simd_inst::vdiv(v_one, rs0, m32, MODE_ZEROING);
      auto is1 = simd_inst::vdiv(v_one, rs1, m32, MODE_ZEROING);
      auto is2 = simd_inst::vdiv(v_one, rs2, m32, MODE_ZEROING);
      auto is3 = simd_inst::vdiv(v_one, rs3, m32, MODE_ZEROING);
      for (int c = 0; c < FULL_CHUNKS; ++c) {
        int off = c * 64;
        vf_store_from_f32(p0 + off, simd_inst::vmul(rg0[c], is0, m32, MODE_ZEROING), m32, m16);
        vf_store_from_f32(p1 + off, simd_inst::vmul(rg1[c], is1, m32, MODE_ZEROING), m32, m16);
        vf_store_from_f32(p2 + off, simd_inst::vmul(rg2[c], is2, m32, MODE_ZEROING), m32, m16);
        vf_store_from_f32(p3 + off, simd_inst::vmul(rg3[c], is3, m32, MODE_ZEROING), m32, m16);
      }
      if constexpr (REM > 0) {
        int off = FULL_CHUNKS * 64;
        vf_store_from_f32(p0 + off, simd_inst::vmul(rg0[FULL_CHUNKS], is0, m32_t, MODE_ZEROING), m32_t, m16_t);
        vf_store_from_f32(p1 + off, simd_inst::vmul(rg1[FULL_CHUNKS], is1, m32_t, MODE_ZEROING), m32_t, m16_t);
        vf_store_from_f32(p2 + off, simd_inst::vmul(rg2[FULL_CHUNKS], is2, m32_t, MODE_ZEROING), m32_t, m16_t);
        vf_store_from_f32(p3 + off, simd_inst::vmul(rg3[FULL_CHUNKS], is3, m32_t, MODE_ZEROING), m32_t, m16_t);
      }
    }
  }
  for (; r < cur_rows; ++r) {
    __ubuf__ InT* row_ptr = x_ub + r * D_UB_STRIDE;
    __ubuf__ float* row_exp = (sizeof(InT) == 4) ? ((__ubuf__ float*)row_ptr) : exp_ub;

    if constexpr (IS_REG) {
      vector_f32 regs[TOTAL_CHUNKS];
      vector_f32 max_acc = v_ninf;
      for (int c = 0; c < FULL_CHUNKS; ++c) {
        auto xf = vf_load_as_f32(row_ptr + c * 64, m16);
        regs[c] = xf;
        max_acc = simd_inst::vmax(max_acc, xf, m32, MODE_ZEROING);
      }
      if constexpr (REM > 0) {
        auto xf_t = vf_load_as_f32(row_ptr + FULL_CHUNKS * 64, m16_t);
        regs[FULL_CHUNKS] = xf_t;
        max_acc = simd_inst::vmax(max_acc, max_acc, xf_t, m32_t, MODE_MERGING);
      }
      auto row_max = simd_inst::vdupv(simd_inst::vcmax(max_acc, m32, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);

      vector_f32 sum_acc = v_zero;
      for (int c = 0; c < FULL_CHUNKS; ++c) {
        auto ev = simd_inst::vexp(simd_inst::vsub(regs[c], row_max, m32, MODE_ZEROING), m32, MODE_ZEROING);
        regs[c] = ev;
        sum_acc = simd_inst::vadd(sum_acc, ev, m32, MODE_ZEROING);
      }
      if constexpr (REM > 0) {
        auto ev_t = simd_inst::vexp(simd_inst::vsub(regs[FULL_CHUNKS], row_max, m32_t, MODE_ZEROING), m32_t, MODE_ZEROING);
        regs[FULL_CHUNKS] = ev_t;
        sum_acc = simd_inst::vadd(sum_acc, ev_t, sum_acc, m32_t, MODE_MERGING);
      }
      auto row_sum = simd_inst::vdupv(simd_inst::vcadd(sum_acc, m32, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
      auto inv_sum = simd_inst::vdiv(v_one, row_sum, m32, MODE_ZEROING);

      for (int c = 0; c < FULL_CHUNKS; ++c) {
        vf_store_from_f32(row_ptr + c * 64, simd_inst::vmul(regs[c], inv_sum, m32, MODE_ZEROING), m32, m16);
      }
      if constexpr (REM > 0) {
        vf_store_from_f32(row_ptr + FULL_CHUNKS * 64, simd_inst::vmul(regs[FULL_CHUNKS], inv_sum, m32_t, MODE_ZEROING), m32_t, m16_t);
      }
    } else {
      vector_f32 max_acc = v_ninf;
      for (int c = 0; c < FULL_CHUNKS; ++c) {
        auto xf = vf_load_as_f32(row_ptr + c * 64, m16);
        max_acc = simd_inst::vmax(max_acc, xf, m32, MODE_ZEROING);
      }
      if constexpr (REM > 0) {
        auto xf_t = vf_load_as_f32(row_ptr + FULL_CHUNKS * 64, m16_t);
        max_acc = simd_inst::vmax(max_acc, xf_t, max_acc, m32_t, MODE_MERGING);
      }
      auto row_max = simd_inst::vdupv(simd_inst::vcmax(max_acc, m32, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);

      vector_f32 sum_acc = v_zero;
      for (int c = 0; c < FULL_CHUNKS; ++c) {
        int off = c * 64;
        auto xf = vf_load_as_f32(row_ptr + off, m16);
        auto ev = simd_inst::vexp(simd_inst::vsub(xf, row_max, m32, MODE_ZEROING), m32, MODE_ZEROING);
        simd_inst::vsts_norm(ev, row_exp + off, 0, m32);
        sum_acc = simd_inst::vadd(sum_acc, ev, m32, MODE_ZEROING);
      }
      if constexpr (REM > 0) {
        int off = FULL_CHUNKS * 64;
        auto xf_t = vf_load_as_f32(row_ptr + off, m16_t);
        auto ev_t = simd_inst::vexp(simd_inst::vsub(xf_t, row_max, m32_t, MODE_ZEROING), m32_t, MODE_ZEROING);
        simd_inst::vsts_norm(ev_t, row_exp + off, 0, m32_t);
        sum_acc = simd_inst::vadd(sum_acc, ev_t, sum_acc, m32_t, MODE_MERGING);
      }
      simd_inst::mem_bar(VST_VLD);
      auto row_sum = simd_inst::vdupv(simd_inst::vcadd(sum_acc, m32, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
      auto inv_sum = simd_inst::vdiv(v_one, row_sum, m32, MODE_ZEROING);

      for (int c = 0; c < FULL_CHUNKS; ++c) {
        int off = c * 64;
        auto ev = simd_inst::vlds_norm<float>(row_exp + off, 0);
        vf_store_from_f32(row_ptr + off, simd_inst::vmul(ev, inv_sum, m32, MODE_ZEROING), m32, m16);
      }
      if constexpr (REM > 0) {
        int off = FULL_CHUNKS * 64;
        auto ev_t = simd_inst::vlds_norm<float>(row_exp + off, 0);
        vf_store_from_f32(row_ptr + off, simd_inst::vmul(ev_t, inv_sum, m32_t, MODE_ZEROING), m32_t, m16_t);
      }
    }
  }
}

template <int D_DIM, int D_UB_STRIDE, int ROWS_BLK, int IS_REG, typename InT>
__global__ __vector__ void softmax_last_axis_kernel(
    __gm__ InT* X, __gm__ InT* Y, int m_rows) {
  asc_init();
  constexpr int STAGE_ELEMS = ROWS_BLK * D_UB_STRIDE + 64;
  constexpr int STAGE_BYTES = STAGE_ELEMS * sizeof(InT);
  __ubuf__ InT* x_ub0 = (__ubuf__ InT*)0;
  __ubuf__ InT* x_ub1 = (__ubuf__ InT*)STAGE_BYTES;
  __ubuf__ float* exp_ub = (__ubuf__ float*)(2 * STAGE_BYTES);

  int cid = (int)block_idx;
  int num_cores = 56;
  int base_rows = m_rows / num_cores;
  int rem_rows = m_rows % num_cores;
  int m_start = cid * base_rows + (cid < rem_rows ? cid : rem_rows);
  int m_count = base_rows + (cid < rem_rows ? 1 : 0);
  if (m_count <= 0) return;

  int num_blocks = (m_count + ROWS_BLK - 1) / ROWS_BLK;
  constexpr int ROW_BYTES = D_DIM * sizeof(InT);
  constexpr int UB_STRIDE_BYTES = D_UB_STRIDE * sizeof(InT);

  asc_sync_notify(PIPE_MTE3, PIPE_MTE2, (event_t)0);
  asc_sync_notify(PIPE_MTE3, PIPE_MTE2, (event_t)1);
  asc_sync_notify(PIPE_V, PIPE_MTE2, (event_t)0);
  asc_sync_notify(PIPE_V, PIPE_MTE2, (event_t)1);

  if (num_blocks > 0) {
    int cur0 = m_count < ROWS_BLK ? m_count : ROWS_BLK;
    asc_sync_wait(PIPE_MTE3, PIPE_MTE2, (event_t)0);
    asc_sync_wait(PIPE_V, PIPE_MTE2, (event_t)0);
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)x_ub0, (__gm__ uint8_t*)(X + (int64_t)m_start * D_DIM),
                         cur0, ROW_BYTES, 0, 0, false,
                         static_cast<asc_load_l2_cache_mode>(0), ROW_BYTES, UB_STRIDE_BYTES);
    asc_sync_notify(PIPE_MTE2, PIPE_V, (event_t)0);
  }

  for (int mb = 0; mb < num_blocks; ++mb) {
    int stg = mb & 1;
    if (mb + 1 < num_blocks) {
      int nstg = (mb + 1) & 1;
      int nm0 = m_start + (mb + 1) * ROWS_BLK;
      int ncur = m_count - (mb + 1) * ROWS_BLK;
      if (ncur > ROWS_BLK) ncur = ROWS_BLK;
      __ubuf__ InT* nxt_ub = nstg ? x_ub1 : x_ub0;
      asc_sync_wait(PIPE_MTE3, PIPE_MTE2, (event_t)nstg);
      asc_sync_wait(PIPE_V, PIPE_MTE2, (event_t)nstg);
      asc_copy_gm2ub_align((__ubuf__ uint8_t*)nxt_ub, (__gm__ uint8_t*)(X + (int64_t)nm0 * D_DIM),
                           ncur, ROW_BYTES, 0, 0, false,
                           static_cast<asc_load_l2_cache_mode>(0), ROW_BYTES, UB_STRIDE_BYTES);
      asc_sync_notify(PIPE_MTE2, PIPE_V, (event_t)nstg);
    }

    int m0 = m_start + mb * ROWS_BLK;
    int cur_r = m_count - mb * ROWS_BLK;
    if (cur_r > ROWS_BLK) cur_r = ROWS_BLK;
    __ubuf__ InT* cur_ub = stg ? x_ub1 : x_ub0;

    asc_sync_wait(PIPE_MTE2, PIPE_V, (event_t)stg);
    vf_softmax_last_rows<D_DIM, D_UB_STRIDE, IS_REG, InT>(cur_ub, exp_ub, cur_r);
    asc_sync_notify(PIPE_V, PIPE_MTE3, (event_t)stg);
    asc_sync_wait(PIPE_V, PIPE_MTE3, (event_t)stg);
    asc_copy_ub2gm_align((__gm__ uint8_t*)(Y + (int64_t)m0 * D_DIM), (__ubuf__ uint8_t*)cur_ub,
                         cur_r, ROW_BYTES, static_cast<asc_store_l2_cache_mode>(4),
                         ROW_BYTES, UB_STRIDE_BYTES);
    asc_sync_notify(PIPE_MTE3, PIPE_MTE2, (event_t)stg);
    asc_sync_notify(PIPE_V, PIPE_MTE2, (event_t)stg);
  }

  asc_sync_wait(PIPE_MTE3, PIPE_MTE2, (event_t)0);
  asc_sync_wait(PIPE_MTE3, PIPE_MTE2, (event_t)1);
  asc_sync_wait(PIPE_V, PIPE_MTE2, (event_t)0);
  asc_sync_wait(PIPE_V, PIPE_MTE2, (event_t)1);
}

template <int D_DIM, int W_TILE, typename InT>
__simd_vf__ inline void vf_softmax_inner_1pass(__ubuf__ InT* x_ub, __ubuf__ float* exp_ub, int w_cur) {
  vector_bool m32_full = asc_create_mask_b32(PAT_ALL);
  vector_bool m16_full = asc_create_mask_b16(PAT_ALL);
  auto v_one = simd_inst::vdup(1.0f, m32_full, MODE_ZEROING);
  auto v_zero = simd_inst::vdup(0.0f, m32_full, MODE_ZEROING);

  for (int j = 0; j < w_cur; j += 64) {
    int rem = w_cur - j;
    int cw = rem < 64 ? rem : 64;
    vector_bool mw = (cw < 64) ? simd_inst::update_mask_b32((uint32_t)cw) : m32_full;
    vector_bool m16w = (cw < 64) ? simd_inst::update_mask_b16((uint32_t)(cw * 2)) : m16_full;

    vector_f32 max_v = vf_load_as_f32(x_ub + j, m16w);
    for (int dd = 1; dd < D_DIM; ++dd) {
      auto xf = vf_load_as_f32(x_ub + dd * W_TILE + j, m16w);
      max_v = simd_inst::vmax(max_v, xf, mw, MODE_ZEROING);
    }

    vector_f32 sum_v = v_zero;
    for (int dd = 0; dd < D_DIM; ++dd) {
      int idx = dd * W_TILE + j;
      auto xf = vf_load_as_f32(x_ub + idx, m16w);
      auto ev = simd_inst::vexp(simd_inst::vsub(xf, max_v, mw, MODE_ZEROING), mw, MODE_ZEROING);
      if constexpr (sizeof(InT) == 4) {
        simd_inst::vsts_norm(ev, ((__ubuf__ float*)x_ub) + idx, 0, mw);
      } else {
        simd_inst::vsts_norm(ev, exp_ub + idx, 0, mw);
      }
      sum_v = simd_inst::vadd(sum_v, ev, mw, MODE_ZEROING);
    }
    simd_inst::mem_bar(VST_VLD);
    auto inv_sum = simd_inst::vdiv(v_one, sum_v, mw, MODE_ZEROING);

    for (int dd = 0; dd < D_DIM; ++dd) {
      int idx = dd * W_TILE + j;
      auto ev = simd_inst::vlds_norm<float>(
          ((sizeof(InT) == 4) ? ((__ubuf__ float*)x_ub) : exp_ub) + idx, 0);
      vf_store_from_f32(x_ub + idx, simd_inst::vmul(ev, inv_sum, mw, MODE_ZEROING), mw, m16w);
    }
  }
}

template <int D_DIM, int W_TILE, int STAGES, typename InT>
__global__ __vector__ void softmax_inner_1pass_kernel(
    __gm__ InT* X, __gm__ InT* Y, int outer_dim, int inner_dim) {
  asc_init();
  constexpr int STAGE_ELEMS = D_DIM * W_TILE;
  constexpr int STAGE_BYTES = STAGE_ELEMS * sizeof(InT);
  __ubuf__ InT* x_ub0 = (__ubuf__ InT*)0;
  __ubuf__ InT* x_ub1 = (__ubuf__ InT*)(STAGES == 2 ? STAGE_BYTES : 0);
  __ubuf__ float* exp_ub = (__ubuf__ float*)(STAGES * STAGE_BYTES);

  int w_tiles = (inner_dim + W_TILE - 1) / W_TILE;
  int total_tasks = outer_dim * w_tiles;
  int cid = (int)block_idx;
  int num_cores = 56;
  int base_t = total_tasks / num_cores;
  int rem_t = total_tasks % num_cores;
  int t_start = cid * base_t + (cid < rem_t ? cid : rem_t);
  int t_count = base_t + (cid < rem_t ? 1 : 0);
  if (t_count <= 0) return;

  int gm_stride_bytes = inner_dim * sizeof(InT);
  int ub_stride_bytes = W_TILE * sizeof(InT);
  asc_sync_notify(PIPE_MTE3, PIPE_MTE2, (event_t)0);
  asc_sync_notify(PIPE_MTE3, PIPE_MTE2, (event_t)1);

  if constexpr (STAGES == 2) {
    if (t_count > 0) {
      int tid0 = t_start;
      int o0 = tid0 / w_tiles;
      int wt0 = tid0 % w_tiles;
      int j0 = wt0 * W_TILE;
      int w0 = inner_dim - j0; if (w0 > W_TILE) w0 = W_TILE;
      asc_sync_wait(PIPE_MTE3, PIPE_MTE2, (event_t)0);
      asc_copy_gm2ub_align((__ubuf__ uint8_t*)x_ub0,
                           (__gm__ uint8_t*)(X + (int64_t)o0 * D_DIM * inner_dim + j0),
                           D_DIM, w0 * sizeof(InT), 0, 0, false,
                           static_cast<asc_load_l2_cache_mode>(0), gm_stride_bytes, ub_stride_bytes);
      asc_sync_notify(PIPE_MTE2, PIPE_V, (event_t)0);
    }
    for (int lt = 0; lt < t_count; ++lt) {
      int stg = lt & 1;
      if (lt + 1 < t_count) {
        int nstg = (lt + 1) & 1;
        int ntid = t_start + lt + 1;
        int no = ntid / w_tiles;
        int nwt = ntid % w_tiles;
        int nj0 = nwt * W_TILE;
        int nw = inner_dim - nj0; if (nw > W_TILE) nw = W_TILE;
        __ubuf__ InT* nxt_ub = nstg ? x_ub1 : x_ub0;
        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, (event_t)nstg);
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)nxt_ub,
                             (__gm__ uint8_t*)(X + (int64_t)no * D_DIM * inner_dim + nj0),
                             D_DIM, nw * sizeof(InT), 0, 0, false,
                             static_cast<asc_load_l2_cache_mode>(0), gm_stride_bytes, ub_stride_bytes);
        asc_sync_notify(PIPE_MTE2, PIPE_V, (event_t)nstg);
      }
      int tid = t_start + lt;
      int o_idx = tid / w_tiles;
      int wt = tid % w_tiles;
      int j0 = wt * W_TILE;
      int w_cur = inner_dim - j0; if (w_cur > W_TILE) w_cur = W_TILE;
      int w_bytes = w_cur * sizeof(InT);
      __ubuf__ InT* cur_ub = stg ? x_ub1 : x_ub0;
      int64_t gm_off = (int64_t)o_idx * D_DIM * inner_dim + j0;

      asc_sync_wait(PIPE_MTE2, PIPE_V, (event_t)stg);
      vf_softmax_inner_1pass<D_DIM, W_TILE, InT>(cur_ub, exp_ub, w_cur);
      asc_sync_notify(PIPE_V, PIPE_MTE3, (event_t)stg);
      asc_sync_wait(PIPE_V, PIPE_MTE3, (event_t)stg);
      asc_copy_ub2gm_align((__gm__ uint8_t*)(Y + gm_off), (__ubuf__ uint8_t*)cur_ub,
                           D_DIM, w_bytes, static_cast<asc_store_l2_cache_mode>(4),
                           gm_stride_bytes, ub_stride_bytes);
      asc_sync_notify(PIPE_MTE3, PIPE_MTE2, (event_t)stg);
    }
  } else {
    for (int lt = 0; lt < t_count; ++lt) {
      int tid = t_start + lt;
      int o_idx = tid / w_tiles;
      int wt = tid % w_tiles;
      int j0 = wt * W_TILE;
      int w_cur = inner_dim - j0; if (w_cur > W_TILE) w_cur = W_TILE;
      int w_bytes = w_cur * sizeof(InT);
      int64_t gm_off = (int64_t)o_idx * D_DIM * inner_dim + j0;

      asc_sync_wait(PIPE_MTE3, PIPE_MTE2, (event_t)0);
      asc_copy_gm2ub_align((__ubuf__ uint8_t*)x_ub0, (__gm__ uint8_t*)(X + gm_off),
                           D_DIM, w_bytes, 0, 0, false,
                           static_cast<asc_load_l2_cache_mode>(0), gm_stride_bytes, ub_stride_bytes);
      asc_sync_notify(PIPE_MTE2, PIPE_V, (event_t)0);
      asc_sync_wait(PIPE_MTE2, PIPE_V, (event_t)0);
      vf_softmax_inner_1pass<D_DIM, W_TILE, InT>(x_ub0, exp_ub, w_cur);
      asc_sync_notify(PIPE_V, PIPE_MTE3, (event_t)0);
      asc_sync_wait(PIPE_V, PIPE_MTE3, (event_t)0);
      asc_copy_ub2gm_align((__gm__ uint8_t*)(Y + gm_off), (__ubuf__ uint8_t*)x_ub0,
                           D_DIM, w_bytes, static_cast<asc_store_l2_cache_mode>(4),
                           gm_stride_bytes, ub_stride_bytes);
      asc_sync_notify(PIPE_MTE3, PIPE_MTE2, (event_t)0);
    }
  }
  asc_sync_wait(PIPE_MTE3, PIPE_MTE2, (event_t)0);
  asc_sync_wait(PIPE_MTE3, PIPE_MTE2, (event_t)1);
}

template <int W_TILE, typename InT>
__simd_vf__ inline void vf_inner_2pass_p1(
    __ubuf__ InT* x_ub, __ubuf__ float* m_ub, __ubuf__ float* s_ub,
    int d_cur, int w_cur, int is_first_blk) {
  vector_bool m32_full = asc_create_mask_b32(PAT_ALL);
  vector_bool m16_full = asc_create_mask_b16(PAT_ALL);
  auto v_zero = simd_inst::vdup(0.0f, m32_full, MODE_ZEROING);
  auto v_one = simd_inst::vdup(1.0f, m32_full, MODE_ZEROING);

  for (int j = 0; j < w_cur; j += 64) {
    int rem = w_cur - j;
    int cw = rem < 64 ? rem : 64;
    vector_bool mw = (cw < 64) ? simd_inst::update_mask_b32((uint32_t)cw) : m32_full;
    vector_bool m16w = (cw < 64) ? simd_inst::update_mask_b16((uint32_t)(cw * 2)) : m16_full;

    vector_f32 m_old = is_first_blk ? vf_load_as_f32(x_ub + j, m16w) : simd_inst::vlds_norm<float>(m_ub + j, 0);
    vector_f32 s_old = is_first_blk ? v_zero : simd_inst::vlds_norm<float>(s_ub + j, 0);
    vector_f32 m_new = m_old;
    for (int dd = 0; dd < d_cur; ++dd) {
      auto xf = vf_load_as_f32(x_ub + dd * W_TILE + j, m16w);
      m_new = simd_inst::vmax(m_new, xf, mw, MODE_ZEROING);
    }
    vector_f32 s_acc = v_zero;
    if (!is_first_blk) {
      vector_bool eq_mask;
      asc_eq(eq_mask, m_old, m_new, mw);
      auto raw_alpha = simd_inst::vexp(simd_inst::vsub(m_old, m_new, mw, MODE_ZEROING), mw, MODE_ZEROING);
      auto alpha = simd_inst::vsel(v_one, raw_alpha, eq_mask);
      s_acc = simd_inst::vmul(s_old, alpha, mw, MODE_ZEROING);
    }
    for (int dd = 0; dd < d_cur; ++dd) {
      auto xf = vf_load_as_f32(x_ub + dd * W_TILE + j, m16w);
      auto ev = simd_inst::vexp(simd_inst::vsub(xf, m_new, mw, MODE_ZEROING), mw, MODE_ZEROING);
      s_acc = simd_inst::vadd(s_acc, ev, mw, MODE_ZEROING);
    }
    simd_inst::vsts_norm(m_new, m_ub + j, 0, mw);
    simd_inst::vsts_norm(s_acc, s_ub + j, 0, mw);
  }
}

template <int W_TILE>
__simd_vf__ inline void vf_inner_inv_sum(__ubuf__ float* s_ub, int w_cur) {
  simd_inst::mem_bar(VST_VLD);
  vector_bool m32_full = asc_create_mask_b32(PAT_ALL);
  auto v_one = simd_inst::vdup(1.0f, m32_full, MODE_ZEROING);
  for (int j = 0; j < w_cur; j += 64) {
    int rem = w_cur - j;
    int cw = rem < 64 ? rem : 64;
    vector_bool mw = (cw < 64) ? simd_inst::update_mask_b32((uint32_t)cw) : m32_full;
    auto sv = simd_inst::vlds_norm<float>(s_ub + j, 0);
    simd_inst::vsts_norm(simd_inst::vdiv(v_one, sv, mw, MODE_ZEROING), s_ub + j, 0, mw);
  }
  simd_inst::mem_bar(VST_VLD);
}

template <int W_TILE, typename InT>
__simd_vf__ inline void vf_inner_2pass_p2(
    __ubuf__ InT* x_ub, __ubuf__ float* m_ub, __ubuf__ float* inv_s_ub,
    int d_cur, int w_cur) {
  vector_bool m32_full = asc_create_mask_b32(PAT_ALL);
  vector_bool m16_full = asc_create_mask_b16(PAT_ALL);
  for (int j = 0; j < w_cur; j += 64) {
    int rem = w_cur - j;
    int cw = rem < 64 ? rem : 64;
    vector_bool mw = (cw < 64) ? simd_inst::update_mask_b32((uint32_t)cw) : m32_full;
    vector_bool m16w = (cw < 64) ? simd_inst::update_mask_b16((uint32_t)(cw * 2)) : m16_full;
    auto mv = simd_inst::vlds_norm<float>(m_ub + j, 0);
    auto isv = simd_inst::vlds_norm<float>(inv_s_ub + j, 0);
    for (int dd = 0; dd < d_cur; ++dd) {
      int idx = dd * W_TILE + j;
      auto xf = vf_load_as_f32(x_ub + idx, m16w);
      auto yf = simd_inst::vmul(simd_inst::vexp(simd_inst::vsub(xf, mv, mw, MODE_ZEROING), mw, MODE_ZEROING), isv, mw, MODE_ZEROING);
      vf_store_from_f32(x_ub + idx, yf, mw, m16w);
    }
  }
}

template <int D_DIM, int D_BLK, int W_TILE, typename InT>
__global__ __vector__ void softmax_inner_2pass_kernel(
    __gm__ InT* X, __gm__ InT* Y, int outer_dim, int inner_dim) {
  asc_init();
  constexpr int STAGE_ELEMS = D_BLK * W_TILE;
  constexpr int STAGE_BYTES = STAGE_ELEMS * sizeof(InT);
  __ubuf__ InT* x_ub0 = (__ubuf__ InT*)0;
  __ubuf__ InT* x_ub1 = (__ubuf__ InT*)STAGE_BYTES;
  __ubuf__ float* m_ub = (__ubuf__ float*)(2 * STAGE_BYTES);
  __ubuf__ float* s_ub = (__ubuf__ float*)(2 * STAGE_BYTES + W_TILE * 4);

  int w_tiles = (inner_dim + W_TILE - 1) / W_TILE;
  int total_tasks = outer_dim * w_tiles;
  int cid = (int)block_idx;
  int num_cores = 56;
  int base_t = total_tasks / num_cores;
  int rem_t = total_tasks % num_cores;
  int t_start = cid * base_t + (cid < rem_t ? cid : rem_t);
  int t_count = base_t + (cid < rem_t ? 1 : 0);
  if (t_count <= 0) return;

  constexpr int D_BLOCKS = (D_DIM + D_BLK - 1) / D_BLK;
  int gm_stride_bytes = inner_dim * sizeof(InT);
  int ub_stride_bytes = W_TILE * sizeof(InT);
  asc_sync_notify(PIPE_MTE3, PIPE_MTE2, (event_t)0);
  asc_sync_notify(PIPE_MTE3, PIPE_MTE2, (event_t)1);
  asc_sync_notify(PIPE_V, PIPE_MTE2, (event_t)0);
  asc_sync_notify(PIPE_V, PIPE_MTE2, (event_t)1);

  for (int lt = 0; lt < t_count; ++lt) {
    int tid = t_start + lt;
    int o_idx = tid / w_tiles;
    int wt = tid % w_tiles;
    int j0 = wt * W_TILE;
    int w_cur = inner_dim - j0;
    if (w_cur > W_TILE) w_cur = W_TILE;
    int w_bytes = w_cur * sizeof(InT);

    // Pass 1: online max & sum
    if (D_BLOCKS > 0) {
      int dc0 = D_DIM < D_BLK ? D_DIM : D_BLK;
      asc_sync_wait(PIPE_V, PIPE_MTE2, (event_t)0);
      asc_copy_gm2ub_align((__ubuf__ uint8_t*)x_ub0, (__gm__ uint8_t*)(X + (int64_t)o_idx * D_DIM * inner_dim + j0),
                           dc0, w_bytes, 0, 0, false,
                           static_cast<asc_load_l2_cache_mode>(0), gm_stride_bytes, ub_stride_bytes);
      asc_sync_notify(PIPE_MTE2, PIPE_V, (event_t)0);
    }
    for (int db = 0; db < D_BLOCKS; ++db) {
      int stg = db & 1;
      if (db + 1 < D_BLOCKS) {
        int nstg = (db + 1) & 1;
        int nd0 = (db + 1) * D_BLK;
        int ndc = D_DIM - nd0; if (ndc > D_BLK) ndc = D_BLK;
        __ubuf__ InT* nxt_ub = nstg ? x_ub1 : x_ub0;
        asc_sync_wait(PIPE_V, PIPE_MTE2, (event_t)nstg);
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)nxt_ub,
                             (__gm__ uint8_t*)(X + ((int64_t)o_idx * D_DIM + nd0) * inner_dim + j0),
                             ndc, w_bytes, 0, 0, false,
                             static_cast<asc_load_l2_cache_mode>(0), gm_stride_bytes, ub_stride_bytes);
        asc_sync_notify(PIPE_MTE2, PIPE_V, (event_t)nstg);
      }
      int d0 = db * D_BLK;
      int d_cur = D_DIM - d0; if (d_cur > D_BLK) d_cur = D_BLK;
      __ubuf__ InT* cur_ub = stg ? x_ub1 : x_ub0;
      asc_sync_wait(PIPE_MTE2, PIPE_V, (event_t)stg);
      vf_inner_2pass_p1<W_TILE, InT>(cur_ub, m_ub, s_ub, d_cur, w_cur, db == 0);
      asc_sync_notify(PIPE_V, PIPE_MTE2, (event_t)stg);
    }
    vf_inner_inv_sum<W_TILE>(s_ub, w_cur);

    // Pass 2: normalize & store
    if (D_BLOCKS > 0) {
      int dc0 = D_DIM < D_BLK ? D_DIM : D_BLK;
      asc_sync_wait(PIPE_V, PIPE_MTE2, (event_t)0);
      asc_sync_wait(PIPE_MTE3, PIPE_MTE2, (event_t)0);
      asc_copy_gm2ub_align((__ubuf__ uint8_t*)x_ub0, (__gm__ uint8_t*)(X + (int64_t)o_idx * D_DIM * inner_dim + j0),
                           dc0, w_bytes, 0, 0, false,
                           static_cast<asc_load_l2_cache_mode>(0), gm_stride_bytes, ub_stride_bytes);
      asc_sync_notify(PIPE_MTE2, PIPE_V, (event_t)0);
    }
    for (int db = 0; db < D_BLOCKS; ++db) {
      int stg = db & 1;
      if (db + 1 < D_BLOCKS) {
        int nstg = (db + 1) & 1;
        int nd0 = (db + 1) * D_BLK;
        int ndc = D_DIM - nd0; if (ndc > D_BLK) ndc = D_BLK;
        __ubuf__ InT* nxt_ub = nstg ? x_ub1 : x_ub0;
        asc_sync_wait(PIPE_V, PIPE_MTE2, (event_t)nstg);
        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, (event_t)nstg);
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)nxt_ub,
                             (__gm__ uint8_t*)(X + ((int64_t)o_idx * D_DIM + nd0) * inner_dim + j0),
                             ndc, w_bytes, 0, 0, false,
                             static_cast<asc_load_l2_cache_mode>(0), gm_stride_bytes, ub_stride_bytes);
        asc_sync_notify(PIPE_MTE2, PIPE_V, (event_t)nstg);
      }
      int d0 = db * D_BLK;
      int d_cur = D_DIM - d0; if (d_cur > D_BLK) d_cur = D_BLK;
      __ubuf__ InT* cur_ub = stg ? x_ub1 : x_ub0;
      asc_sync_wait(PIPE_MTE2, PIPE_V, (event_t)stg);
      vf_inner_2pass_p2<W_TILE, InT>(cur_ub, m_ub, s_ub, d_cur, w_cur);
      asc_sync_notify(PIPE_V, PIPE_MTE3, (event_t)stg);
      asc_sync_wait(PIPE_V, PIPE_MTE3, (event_t)stg);
      asc_copy_ub2gm_align((__gm__ uint8_t*)(Y + ((int64_t)o_idx * D_DIM + d0) * inner_dim + j0),
                           (__ubuf__ uint8_t*)cur_ub, d_cur, w_bytes,
                           static_cast<asc_store_l2_cache_mode>(4), gm_stride_bytes, ub_stride_bytes);
      asc_sync_notify(PIPE_MTE3, PIPE_MTE2, (event_t)stg);
      asc_sync_notify(PIPE_V, PIPE_MTE2, (event_t)stg);
    }
  }
  asc_sync_wait(PIPE_MTE3, PIPE_MTE2, (event_t)0);
  asc_sync_wait(PIPE_MTE3, PIPE_MTE2, (event_t)1);
  asc_sync_wait(PIPE_V, PIPE_MTE2, (event_t)0);
  asc_sync_wait(PIPE_V, PIPE_MTE2, (event_t)1);
}
"""


def _select_trackb_softmax_plan(d_dim: int, tl_dtype: str, m_rows: int, inner_dim: int):
    elem_bytes = 4 if tl_dtype == "float32" else 2
    ub_budget = 248 * 1024

    if inner_dim > 1:
        # Check if 1-pass fits with generous w_tile maximizing burst length & core occupancy
        # Candidate w_tiles (multiples of 16)
        w_align = ((inner_dim + 15) // 16) * 16
        cand_ws = []
        if w_align <= 512:
            cand_ws.append(w_align)
        for wt in (256, 192, 160, 144, 128, 96, 64, 48, 32, 16):
            if wt <= max(64, w_align) and wt not in cand_ws:
                cand_ws.append(wt)

        best_1p = None
        for wt in cand_ws:
            if wt < 64 and inner_dim >= 64:
                continue
            per_stage = d_dim * wt * elem_bytes
            exp_b = 0 if elem_bytes == 4 else d_dim * wt * 4
            if per_stage + exp_b <= ub_budget:
                stages = 2 if (2 * per_stage + exp_b <= ub_budget) else 1
                w_tiles = (inner_dim + wt - 1) // wt
                tasks = m_rows * w_tiles
                waves = (tasks + 55) // 56
                score_key = (waves, -min(wt, inner_dim), -stages)
                if best_1p is None or score_key < best_1p[0]:
                    best_1p = (score_key, wt, stages)

        if best_1p is not None:
            _, wt, stages = best_1p
            return ("inner_1pass", wt, stages, d_dim)

        # 2-pass online: maximize burst width & fit in 1 wave if m_rows == 1
        if m_rows == 1:
            target_w = max(64, ((inner_dim + 55) // 56 + 15) // 16 * 16)
            wt = min(256, target_w)
        else:
            wt = min(256, w_align) if w_align <= 256 else 128
        d_blk = min(d_dim, max(32, ((ub_budget - 2 * wt * 4) // (2 * wt * elem_bytes) // 16) * 16))
        if d_blk < 32:
            d_blk = min(d_dim, max(16, (ub_budget - 2 * wt * 4) // (2 * wt * elem_bytes)))
        return ("inner_2pass", wt, 2, d_blk)

    # inner_dim == 1, d_dim > 16
    d_ub_stride = ((d_dim + 15) // 16) * 16
    is_reg = 1 if d_dim <= 512 else 0
    exp_b = 0 if (elem_bytes == 4 or is_reg) else (d_ub_stride * 4)
    # Two stages of (rows_blk * d_ub_stride + 64) * elem_bytes
    slack_b = 2 * 64 * elem_bytes
    avail = max(1, ub_budget - exp_b - slack_b)
    max_rows = max(1, min(256, avail // (2 * d_ub_stride * elem_bytes)))
    rows_per_core = (m_rows + 55) // 56
    if max_rows > 1 and rows_per_core >= 2:
        rows_blk = min(max_rows, max(1, (m_rows + 111) // 112))
    else:
        rows_blk = max_rows
    return ("last_axis", d_ub_stride, rows_blk, is_reg)


def _get_trackb_kernel(d_dim: int, tl_dtype: str, m_rows: int, inner_dim: int):
    plan = _select_trackb_softmax_plan(d_dim, tl_dtype, m_rows, inner_dim)
    key = ("softmax_trackb", d_dim, tl_dtype, plan)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    c_in = {"float16": "half", "bfloat16": "bfloat16_t", "float32": "float"}[tl_dtype]
    kind = plan[0]
    if kind == "last_axis":
        _, d_ub_stride, rows_blk, is_reg = plan
        launcher = (
            _SOFTMAX_TRACKB_SRC
            + f"""
extern "C" int launch_softmax_tb(void* stream, void* x_ptr, void* y_ptr, int a, int b) {{
  softmax_last_axis_kernel<{d_dim}, {d_ub_stride}, {rows_blk}, {is_reg}, {c_in}><<<56, nullptr, stream>>>(
      (__gm__ {c_in}*)x_ptr, (__gm__ {c_in}*)y_ptr, a);
  return 0;
}}
"""
        )
    elif kind == "inner_1pass":
        _, w_tile, stages, _ = plan
        launcher = (
            _SOFTMAX_TRACKB_SRC
            + f"""
extern "C" int launch_softmax_tb(void* stream, void* x_ptr, void* y_ptr, int a, int b) {{
  softmax_inner_1pass_kernel<{d_dim}, {w_tile}, {stages}, {c_in}><<<56, nullptr, stream>>>(
      (__gm__ {c_in}*)x_ptr, (__gm__ {c_in}*)y_ptr, a, b);
  return 0;
}}
"""
        )
    else:
        _, w_tile, _, d_blk = plan
        launcher = (
            _SOFTMAX_TRACKB_SRC
            + f"""
extern "C" int launch_softmax_tb(void* stream, void* x_ptr, void* y_ptr, int a, int b) {{
  softmax_inner_2pass_kernel<{d_dim}, {d_blk}, {w_tile}, {c_in}><<<56, nullptr, stream>>>(
      (__gm__ {c_in}*)x_ptr, (__gm__ {c_in}*)y_ptr, a, b);
  return 0;
}}
"""
        )

    tl_inc = str(Path(tilelang.__file__).resolve().parent.parent / "src")
    asc_home = os.environ.get("ASCEND_HOME_PATH", "")
    asc_inc = os.path.join(asc_home, "include") if asc_home else ""
    opts = [f"-I{tl_inc}"]
    if asc_inc and os.path.isdir(asc_inc):
        opts.append(f"-I{asc_inc}")
    opts.extend(PURE_SIMDVF_UBUF_FLAGS)
    so_dir = tempfile.mkdtemp(prefix="softmax_tb_")
    so_path = os.path.join(so_dir, "softmax_tb.so")
    compile_ascend(
        launcher,
        target_format="so",
        npu_arch="dav-3510",
        options=opts,
        path_target=so_path,
    )
    lib = ctypes.CDLL(so_path)
    fn = lib.launch_softmax_tb
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

    def _run(x_c: torch.Tensor, a: int, b: int) -> torch.Tensor:
        y_c = torch.empty_like(x_c)
        stream_ptr = torch.npu.current_stream().npu_stream
        fn(stream_ptr, x_c.data_ptr(), y_c.data_ptr(), int(a), int(b))
        return y_c

    _run._so_lib = lib
    _KERNEL_CACHE[key] = _run
    return _run


def softmax(x: torch.Tensor, dim: int = -1, *, schedule_mode: str = "tuned") -> torch.Tensor:
    """Compute Softmax matching cannbot-ops level2/softmax proto.yaml."""
    if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise ValueError(f"unsupported dtype {x.dtype}")
    ndim = x.ndim
    norm_dim = dim if dim >= 0 else ndim + dim
    if norm_dim < 0 or norm_dim >= ndim:
        raise ValueError(f"dim {dim} out of range for tensor with {ndim} dims")

    if x.device.type != "npu":
        return F.softmax(x, dim=norm_dim)

    tl_dtype = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }[x.dtype]
    x_c = x.contiguous()
    shape = tuple(x_c.shape)
    d_dim = shape[norm_dim]
    outer = 1
    for s in shape[:norm_dim]:
        outer *= s
    inner = 1
    for s in shape[norm_dim + 1 :]:
        inner *= s

    if schedule_mode == "tuned" and (inner > 1 or d_dim > 16):
        m_rows = max(1, outer)
        runner = _get_trackb_kernel(d_dim, tl_dtype, m_rows, inner)
        y_c = runner(x_c, outer if inner > 1 else m_rows, inner)
        return y_c.view(shape)

    if inner > 1:
        compiled = _get_kernel(d_dim, tl_dtype, outer, inner, schedule_mode)
        y_3d = compiled(x_c.view(outer, d_dim, inner))
        return y_3d.view(shape)

    m_rows = max(1, outer)
    compiled = _get_kernel(d_dim, tl_dtype, m_rows, 1, schedule_mode)
    if d_dim <= 16:
        y_1d = compiled(x_c.view(-1))
        return y_1d.view(shape)
    y_2d = compiled(x_c.view(m_rows, d_dim))
    return y_2d.view(shape)


kernel = softmax

__all__ = ["_build_softmax_prim_func", "_select_softmax_config", "kernel", "softmax"]
