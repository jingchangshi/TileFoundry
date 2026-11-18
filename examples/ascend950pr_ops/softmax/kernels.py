"""TileLang Ascend950PR Pure RegBase SimdVF / SimtVF kernel implementation for Softmax."""

import contextlib

import tilelang
import tilelang.ascend.language as T
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
