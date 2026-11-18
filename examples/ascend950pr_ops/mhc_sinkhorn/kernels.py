"""Ascend 950PR SimdVF & SimtVF MhcSinkhorn kernel across hc_mult in [2, 16]."""

import contextlib
import ctypes
import hashlib
from pathlib import Path
import tempfile
import tilelang
import tilelang.ascend.language as T
import torch
from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS

_KERNEL_CACHE: dict[tuple[object, ...], object] = {}
_TRACK_B_LIB = None


def _ascend_target_ctx():
    try:
        from tilelang import tvm
        from tilelang.backend.module import create_backend_context

        if tvm.target.Target.current(allow_none=True) is None:
            return create_backend_context("ascend").target
    except Exception:
        pass
    return contextlib.nullcontext()


def _select_mhc_sinkhorn_config(
    hc_mult: int = 4,
    *,
    b_dim: int = 1024,
    iter_step: int = 20,
    schedule_mode: str = "tuned",
) -> dict[str, object]:
    m_n = max(1, int(hc_mult))
    b_val = max(1, int(b_dim))
    n_cores = 1 if b_val <= 32 else 56
    mat_elems = m_n * m_n
    ub_budget = 256 * 1024 - 4096 if schedule_mode == "tuned" else 192 * 1024

    if m_n == 4 and b_val <= 56 * 32:
        return {
            "regime": "vreg_b32_4x4",
            "hc_mult": m_n,
            "iter_step": max(1, int(iter_step)),
            "batch_tile": 32,
            "num_cores": n_cores,
            "use_pipelined": False,
            "pass_configs": {},
            "compile_flags": list(PURE_SIMDVF_UBUF_FLAGS) if schedule_mode == "tuned" else [],
        }
    if m_n == 4:
        return {
            "regime": "simt_per_mat_4x4",
            "hc_mult": m_n,
            "iter_step": max(1, int(iter_step)),
            "batch_tile": 320,
            "num_cores": n_cores,
            "use_pipelined": False,
            "pass_configs": {"tl.disable_data_race_check": True, "tl.enable_fast_math": True},
            "compile_flags": [],
        }
    if m_n in (2, 3):
        return {
            "regime": f"vreg_batched_{m_n}x{m_n}",
            "hc_mult": m_n,
            "iter_step": max(1, int(iter_step)),
            "batch_tile": 64,
            "num_cores": n_cores,
            "use_pipelined": b_val > 56 * 64,
            "pass_configs": {},
            "compile_flags": list(PURE_SIMDVF_UBUF_FLAGS) if schedule_mode == "tuned" else [],
        }
    if m_n in (6, 8, 12, 16):
        return {
            "regime": f"simdvf_b32_work_{m_n}x{m_n}",
            "hc_mult": m_n,
            "iter_step": max(1, int(iter_step)),
            "batch_tile": 32,
            "num_cores": n_cores,
            "use_pipelined": False,
            "pass_configs": {},
            "compile_flags": list(PURE_SIMDVF_UBUF_FLAGS) if schedule_mode == "tuned" else [],
        }

    max_batch_tile = (
        64
        if (64 * mat_elems * 4 * 2) <= ub_budget
        else max(1, min(64, ub_budget // (2 * mat_elems * 4)))
    )
    return {
        "regime": "sfu_pipelined_tile_mat",
        "hc_mult": m_n,
        "iter_step": max(1, int(iter_step)),
        "batch_tile": max_batch_tile,
        "num_cores": n_cores,
        "use_pipelined": b_val > 56 * max_batch_tile,
        "pass_configs": {},
        "compile_flags": list(PURE_SIMDVF_UBUF_FLAGS) if schedule_mode == "tuned" else [],
    }


def _build_mhc_sinkhorn_prim_func(
    hc_mult: int = 4,
    iter_step: int = 20,
    batch_tile: int = 32,
    num_cores: int = 56,
    regime: str | None = None,
    use_pipelined: bool = False,
    schedule_mode: str = "tuned",
    **_extra_kwargs,
):
    num_cores = max(1, int(num_cores))
    m_n = int(hc_mult)
    mat_elems = m_n * m_n
    b_tile = int(batch_tile)
    n_elems = T.dynamic("n_elems")
    n_iter = max(1, int(iter_step))

    if regime is None:
        if m_n == 4:
            regime = "vreg_b32_4x4" if b_tile <= 32 else "simt_per_mat_4x4"
        elif m_n in (2, 3):
            regime = f"vreg_batched_{m_n}x{m_n}"
        elif m_n in (6, 8, 12, 16):
            regime = f"simdvf_b32_work_{m_n}x{m_n}"
        else:
            regime = "sfu_pipelined_tile_mat"

    with _ascend_target_ctx():
        # ------------------------------------------------------------------
        # 1. 4x4 2-cols-per-VREG (8 VREGs total, zero work_ub, zero mem_bar)
        # ------------------------------------------------------------------
        if m_n == 4 and regime == "vreg_b32_4x4":
            b_t = 32
            io_cap = b_t * 16 + 64

            @T.prim_func
            def main_b32_4x4(
                Comb: T.Tensor((n_elems,), "float32"),
                CombOut: T.Tensor((n_elems,), "float32"),
                eps: T.float32,
                b_dim: T.int32,
            ):
                with T.Kernel(num_cores) as cid:
                    io_ub = T.alloc_shared((io_cap,), "float32")
                    if schedule_mode == "tuned":
                        T.annotate_unlimit_memory("shared")

                    base_b = b_dim // num_cores
                    rem_b = b_dim % num_cores
                    b_start = cid * base_b + T.min(cid, rem_b)
                    b_count = base_b + T.if_then_else(cid < rem_b, 1, 0)
                    num_chunks = (b_count + b_t - 1) // b_t

                    for ch in T.serial(num_chunks):
                        b0 = b_start + ch * b_t
                        b_cur = T.min(b_t, b_count - ch * b_t)
                        elem_off = b0 * 16
                        elem_cnt = b_cur * 16
                        T.copy(Comb[elem_off : elem_off + elem_cnt], io_ub[:elem_cnt])

                        with T.SimdVF():
                            m32 = T.simd.pset(32, "PAT_ALL")
                            v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                            v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
                            lane = T.simd.vci(0, "int32")
                            sub_lane = T.simd.vand(lane, T.simd.vdup(31, "int32", mask=m32), mask=m32)
                            is_hi = T.simd.vshrs(lane, 5, mask=m32)
                            idx_swap32 = T.simd.vxor(lane, T.simd.vdup(32, "int32", mask=m32), mask=m32)
                            m_hi_active = T.simd.vcmps(sub_lane, b_cur, mask=m32, op="lt")
                            base_off = T.simd.vmuls(sub_lane, 16, mask=m32)

                            x = T.simd.alloc_local((8,), "float32")
                            for r in T.unroll(4):
                                for p in T.unroll(2):
                                    off_elem = T.simd.vadd(T.simd.vadds(base_off, r * 4 + p * 2, mask=m32), is_hi, mask=m32)
                                    idx = T.reinterpret("uint32x64", off_elem)
                                    x[r * 2 + p] = T.simd.vsel(T.simd.vgather2(io_ub[0], idx, mask=m_hi_active), v_zero, m_hi_active)

                            # Iter 1: row-softmax + eps
                            for r in T.unroll(4):
                                rm0 = T.simd.vmax(x[r * 2], x[r * 2 + 1], mask=m32)
                                rm = T.simd.vmax(rm0, T.simd.vselr(rm0, idx_swap32), mask=m32)
                                e0 = T.simd.vexpdif(x[r * 2], rm, mask=m32)
                                e1 = T.simd.vexpdif(x[r * 2 + 1], rm, mask=m32)
                                rs0 = T.simd.vadd(e0, e1, mask=m32)
                                rs = T.simd.vadd(rs0, T.simd.vselr(rs0, idx_swap32), mask=m32)
                                inv_r = T.simd.vdiv(v_one, rs, mask=m32, precision="ftz_true")
                                x[r * 2] = T.simd.vadds(T.simd.vmul(e0, inv_r, mask=m32), eps, mask=m32)
                                x[r * 2 + 1] = T.simd.vadds(T.simd.vmul(e1, inv_r, mask=m32), eps, mask=m32)

                            # Iter 1: col-norm
                            for p in T.unroll(2):
                                cs = T.simd.vadd(
                                    T.simd.vadd(x[p], x[2 + p], mask=m32),
                                    T.simd.vadd(x[4 + p], x[6 + p], mask=m32),
                                    mask=m32,
                                )
                                inv_c = T.simd.vdiv(v_one, T.simd.vadds(cs, eps, mask=m32), mask=m32, precision="ftz_true")
                                for r in T.unroll(4):
                                    x[r * 2 + p] = T.simd.vmul(x[r * 2 + p], inv_c, mask=m32)

                            for _it in T.serial(n_iter - 1):
                                for r in T.unroll(4):
                                    rs0 = T.simd.vadd(x[r * 2], x[r * 2 + 1], mask=m32)
                                    rs = T.simd.vadd(rs0, T.simd.vselr(rs0, idx_swap32), mask=m32)
                                    inv_r2 = T.simd.vdiv(v_one, T.simd.vadds(rs, eps, mask=m32), mask=m32, precision="ftz_true")
                                    x[r * 2] = T.simd.vmul(x[r * 2], inv_r2, mask=m32)
                                    x[r * 2 + 1] = T.simd.vmul(x[r * 2 + 1], inv_r2, mask=m32)
                                for p in T.unroll(2):
                                    cs = T.simd.vadd(
                                        T.simd.vadd(x[p], x[2 + p], mask=m32),
                                        T.simd.vadd(x[4 + p], x[6 + p], mask=m32),
                                        mask=m32,
                                    )
                                    inv_c2 = T.simd.vdiv(v_one, T.simd.vadds(cs, eps, mask=m32), mask=m32, precision="ftz_true")
                                    for r in T.unroll(4):
                                        x[r * 2 + p] = T.simd.vmul(x[r * 2 + p], inv_c2, mask=m32)

                            for r in T.unroll(4):
                                for p in T.unroll(2):
                                    off_elem = T.simd.vadd(T.simd.vadds(base_off, r * 4 + p * 2, mask=m32), is_hi, mask=m32)
                                    idx = T.reinterpret("uint32x64", off_elem)
                                    T.simd.vscatter(x[r * 2 + p], io_ub[0], idx, mask=m_hi_active)

                        T.copy(io_ub[:elem_cnt], CombOut[elem_off : elem_off + elem_cnt])

            return main_b32_4x4

        # ------------------------------------------------------------------
        # 2. 4x4 SIMT 1-thread-per-matrix for large B (4096, 16384)
        # ------------------------------------------------------------------
        if m_n == 4 and regime == "simt_per_mat_4x4":
            threads = 320
            b_t = max(threads, int(batch_tile))
            io_cap = b_t * 16
            iters = (b_t + threads - 1) // threads

            @T.prim_func
            def main_simt_4x4(
                Comb: T.Tensor((n_elems,), "float32"),
                CombOut: T.Tensor((n_elems,), "float32"),
                eps: T.float32,
                b_dim: T.int32,
            ):
                with T.Kernel(num_cores) as cid:
                    io_ub = T.alloc_shared((io_cap,), "float32")
                    base_b = b_dim // num_cores
                    rem_b = b_dim % num_cores
                    b_start = cid * base_b + T.min(cid, rem_b)
                    b_count = base_b + T.if_then_else(cid < rem_b, 1, 0)
                    num_chunks = (b_count + b_t - 1) // b_t

                    for ch in T.serial(num_chunks):
                        b0 = b_start + ch * b_t
                        b_cur = T.min(b_t, b_count - ch * b_t)
                        elem_off = b0 * 16
                        elem_cnt = b_cur * 16
                        T.copy(Comb[elem_off : elem_off + elem_cnt], io_ub[:elem_cnt])

                        with T.SimtVF(threads=threads):
                            for tid in T.Parallel(threads):
                                for idx in T.serial(iters):
                                    m = idx * threads + tid
                                    if m < b_cur:
                                        x = T.alloc_local((16,), "float32")
                                        base = m * 16
                                        for k in T.unroll(16):
                                            x[k] = io_ub[base + k]

                                        for r in T.unroll(4):
                                            rmax = T.max(T.max(x[r * 4 + 0], x[r * 4 + 1]), T.max(x[r * 4 + 2], x[r * 4 + 3]))
                                            e0 = T.exp(x[r * 4 + 0] - rmax)
                                            e1 = T.exp(x[r * 4 + 1] - rmax)
                                            e2 = T.exp(x[r * 4 + 2] - rmax)
                                            e3 = T.exp(x[r * 4 + 3] - rmax)
                                            inv_r = T.float32(1.0) / (e0 + e1 + e2 + e3)
                                            x[r * 4 + 0] = e0 * inv_r + eps
                                            x[r * 4 + 1] = e1 * inv_r + eps
                                            x[r * 4 + 2] = e2 * inv_r + eps
                                            x[r * 4 + 3] = e3 * inv_r + eps

                                        for c in T.unroll(4):
                                            inv_c = T.float32(1.0) / (x[0 + c] + x[4 + c] + x[8 + c] + x[12 + c] + eps)
                                            x[0 + c] *= inv_c
                                            x[4 + c] *= inv_c
                                            x[8 + c] *= inv_c
                                            x[12 + c] *= inv_c

                                        for _it in T.serial(n_iter - 1):
                                            for r in T.unroll(4):
                                                inv_r2 = T.float32(1.0) / (
                                                    x[r * 4 + 0] + x[r * 4 + 1] + x[r * 4 + 2] + x[r * 4 + 3] + eps
                                                )
                                                x[r * 4 + 0] *= inv_r2
                                                x[r * 4 + 1] *= inv_r2
                                                x[r * 4 + 2] *= inv_r2
                                                x[r * 4 + 3] *= inv_r2
                                            for c in T.unroll(4):
                                                inv_c2 = T.float32(1.0) / (x[0 + c] + x[4 + c] + x[8 + c] + x[12 + c] + eps)
                                                x[0 + c] *= inv_c2
                                                x[4 + c] *= inv_c2
                                                x[8 + c] *= inv_c2
                                                x[12 + c] *= inv_c2

                                        for k in T.unroll(16):
                                            io_ub[base + k] = x[k]

                        T.copy(io_ub[:elem_cnt], CombOut[elem_off : elem_off + elem_cnt])

            return main_simt_4x4

        # ------------------------------------------------------------------
        # 3. Batched 64-lane VREG-resident for hc_mult in (2, 3)
        # ------------------------------------------------------------------
        if m_n in (2, 3):
            io_cap = ((b_tile * mat_elems + 63) // 64) * 64 + 64
            stages = 2 if (use_pipelined and schedule_mode == "tuned") else 1

            @T.prim_func
            def main_vreg_small(
                Comb: T.Tensor((n_elems,), "float32"),
                CombOut: T.Tensor((n_elems,), "float32"),
                eps: T.float32,
                b_dim: T.int32,
            ):
                with T.Kernel(num_cores) as cid:
                    io_in_ub = T.alloc_shared((io_cap,), "float32")
                    io_out_ub = T.alloc_shared((io_cap,), "float32") if stages > 1 else io_in_ub
                    if stages > 1:
                        T.annotate_buffer_versions({io_in_ub: 2, io_out_ub: 2})
                    else:
                        T.annotate_buffer_versions({io_in_ub: 1})
                    if schedule_mode == "tuned":
                        T.annotate_unlimit_memory("shared")

                    base_b = b_dim // num_cores
                    rem_b = b_dim % num_cores
                    b_start = cid * base_b + T.min(cid, rem_b)
                    b_count = base_b + T.if_then_else(cid < rem_b, 1, 0)
                    num_chunks = (b_count + b_tile - 1) // b_tile

                    for ch in T.Pipelined(num_chunks, num_stages=stages):
                        b0 = b_start + ch * b_tile
                        b_cur = T.min(b_tile, b_count - ch * b_tile)
                        elem_off = b0 * mat_elems
                        elem_cnt = b_cur * mat_elems
                        T.copy(Comb[elem_off : elem_off + elem_cnt], io_in_ub[:elem_cnt])

                        with T.SimdVF():
                            m32 = T.simd.pset(32, "PAT_ALL")
                            v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                            v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
                            m_act = T.simd.update_mask(b_cur, width=32)
                            lane = T.simd.vci(0, "int32")
                            base_off = T.simd.vmuls(lane, mat_elems, mask=m32)

                            x = T.simd.alloc_local((mat_elems,), "float32")
                            for r in T.unroll(m_n):
                                for c in T.unroll(m_n):
                                    idx = T.reinterpret("uint32x64", T.simd.vadds(base_off, r * m_n + c, mask=m32))
                                    x[r * m_n + c] = T.simd.vsel(T.simd.vgather2(io_in_ub[0], idx, mask=m_act), v_zero, m_act)

                            rmax = T.simd.alloc_local((1,), "float32")
                            rsum = T.simd.alloc_local((1,), "float32")
                            csum = T.simd.alloc_local((1,), "float32")

                            for r in T.unroll(m_n):
                                rmax[0] = x[r * m_n + 0]
                                for c in T.unroll(1, m_n):
                                    rmax[0] = T.simd.vmax(rmax[0], x[r * m_n + c], mask=m32)
                                for c in T.unroll(m_n):
                                    x[r * m_n + c] = T.simd.vexpdif(x[r * m_n + c], rmax[0], mask=m32)
                                rsum[0] = x[r * m_n + 0]
                                for c in T.unroll(1, m_n):
                                    rsum[0] = T.simd.vadd(rsum[0], x[r * m_n + c], mask=m32)
                                inv_r = T.simd.vdiv(v_one, rsum[0], mask=m32, precision="ftz_true")
                                for c in T.unroll(m_n):
                                    x[r * m_n + c] = T.simd.vadds(T.simd.vmul(x[r * m_n + c], inv_r, mask=m32), eps, mask=m32)

                            for c in T.unroll(m_n):
                                csum[0] = x[0 * m_n + c]
                                for r in T.unroll(1, m_n):
                                    csum[0] = T.simd.vadd(csum[0], x[r * m_n + c], mask=m32)
                                inv_c = T.simd.vdiv(v_one, T.simd.vadds(csum[0], eps, mask=m32), mask=m32, precision="ftz_true")
                                for r in T.unroll(m_n):
                                    x[r * m_n + c] = T.simd.vmul(x[r * m_n + c], inv_c, mask=m32)

                            for _it in T.serial(n_iter - 1):
                                for r in T.unroll(m_n):
                                    rsum[0] = x[r * m_n + 0]
                                    for c in T.unroll(1, m_n):
                                        rsum[0] = T.simd.vadd(rsum[0], x[r * m_n + c], mask=m32)
                                    inv_r2 = T.simd.vdiv(v_one, T.simd.vadds(rsum[0], eps, mask=m32), mask=m32, precision="ftz_true")
                                    for c in T.unroll(m_n):
                                        x[r * m_n + c] = T.simd.vmul(x[r * m_n + c], inv_r2, mask=m32)

                                for c in T.unroll(m_n):
                                    csum[0] = x[0 * m_n + c]
                                    for r in T.unroll(1, m_n):
                                        csum[0] = T.simd.vadd(csum[0], x[r * m_n + c], mask=m32)
                                    inv_c2 = T.simd.vdiv(v_one, T.simd.vadds(csum[0], eps, mask=m32), mask=m32, precision="ftz_true")
                                    for r in T.unroll(m_n):
                                        x[r * m_n + c] = T.simd.vmul(x[r * m_n + c], inv_c2, mask=m32)

                            for r in T.unroll(m_n):
                                for c in T.unroll(m_n):
                                    idx = T.reinterpret("uint32x64", T.simd.vadds(base_off, r * m_n + c, mask=m32))
                                    T.simd.vscatter(x[r * m_n + c], io_out_ub[0], idx, mask=m_act)

                        T.copy(io_out_ub[:elem_cnt], CombOut[elem_off : elem_off + elem_cnt])

            return main_vreg_small

        # ------------------------------------------------------------------
        # 4. 2-cols-per-VREG UB-cached SimdVF for hc_mult in (6, 8, 12, 16)
        # ------------------------------------------------------------------
        if m_n in (6, 8, 12, 16):
            b_t = 32
            half_n = m_n // 2
            io_cap = b_t * mat_elems + 64
            cache_rows = min(m_n, 8)
            buf_cnt = max(half_n, cache_rows)

            @T.prim_func
            def main_b32_work(
                Comb: T.Tensor((n_elems,), "float32"),
                CombOut: T.Tensor((n_elems,), "float32"),
                eps: T.float32,
                b_dim: T.int32,
            ):
                with T.Kernel(num_cores) as cid:
                    io_ub = T.alloc_shared((io_cap,), "float32")
                    work_ub = T.alloc_shared((m_n, m_n, b_t), "float32")
                    if schedule_mode == "tuned":
                        T.annotate_unlimit_memory("shared")

                    base_b = b_dim // num_cores
                    rem_b = b_dim % num_cores
                    b_start = cid * base_b + T.min(cid, rem_b)
                    b_count = base_b + T.if_then_else(cid < rem_b, 1, 0)
                    num_chunks = (b_count + b_t - 1) // b_t

                    for ch in T.serial(num_chunks):
                        b0 = b_start + ch * b_t
                        b_cur = T.min(b_t, b_count - ch * b_t)
                        elem_off = b0 * mat_elems
                        elem_cnt = b_cur * mat_elems
                        T.copy(Comb[elem_off : elem_off + elem_cnt], io_ub[:elem_cnt])

                        with T.SimdVF():
                            m32 = T.simd.pset(32, "PAT_ALL")
                            v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                            lane = T.simd.vci(0, "int32")
                            sub_lane = T.simd.vand(lane, T.simd.vdup(31, "int32", mask=m32), mask=m32)
                            is_hi = T.simd.vshrs(lane, 5, mask=m32)
                            m_hi_active = T.simd.vcmps(sub_lane, b_cur, mask=m32, op="lt")
                            base_off = T.simd.vmuls(sub_lane, mat_elems, mask=m32)
                            for r in T.serial(m_n):
                                for p in T.serial(half_n):
                                    off_elem = T.simd.vadd(T.simd.vadds(base_off, r * m_n + p * 2, mask=m32), is_hi, mask=m32)
                                    idx = T.reinterpret("uint32x64", off_elem)
                                    gv = T.simd.vsel(T.simd.vgather2(io_ub[0], idx, mask=m_hi_active), v_zero, m_hi_active)
                                    T.simd.vsts(work_ub[r, p * 2, 0], gv, mask=m32, dist="NORM_B32")
                            T.simd.mem_bar("VST_VLD")

                        with T.SimdVF():
                            m32 = T.simd.pset(32, "PAT_ALL")
                            v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                            v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
                            lane = T.simd.vci(0, "int32")
                            idx_swap32 = T.simd.vxor(lane, T.simd.vdup(32, "int32", mask=m32), mask=m32)
                            buf = T.simd.alloc_local((buf_cnt,), "float32")
                            rmax = T.simd.alloc_local((1,), "float32")
                            rsum = T.simd.alloc_local((1,), "float32")
                            csum = T.simd.alloc_local((1,), "float32")

                            for r in T.serial(m_n):
                                for p in T.unroll(half_n):
                                    buf[p] = T.simd.vld(work_ub[r, p * 2, 0], dist="NORM_B32")
                                rmax[0] = buf[0]
                                for p in T.unroll(1, half_n):
                                    rmax[0] = T.simd.vmax(rmax[0], buf[p], mask=m32)
                                rm_full = T.simd.vmax(rmax[0], T.simd.vselr(rmax[0], idx_swap32), mask=m32)
                                rsum[0] = v_zero
                                for p in T.unroll(half_n):
                                    buf[p] = T.simd.vexpdif(buf[p], rm_full, mask=m32)
                                    rsum[0] = T.simd.vadd(rsum[0], buf[p], mask=m32)
                                rs_full = T.simd.vadd(rsum[0], T.simd.vselr(rsum[0], idx_swap32), mask=m32)
                                inv_r = T.simd.vdiv(v_one, rs_full, mask=m32, precision="ftz_true")
                                for p in T.unroll(half_n):
                                    T.simd.vsts(
                                        work_ub[r, p * 2, 0],
                                        T.simd.vadds(T.simd.vmul(buf[p], inv_r, mask=m32), eps, mask=m32),
                                        mask=m32,
                                        dist="NORM_B32",
                                    )
                            T.simd.mem_bar("VST_VLD")

                            for p in T.serial(half_n):
                                csum[0] = v_zero
                                for r in T.unroll(cache_rows):
                                    buf[r] = T.simd.vld(work_ub[r, p * 2, 0], dist="NORM_B32")
                                    csum[0] = T.simd.vadd(csum[0], buf[r], mask=m32)
                                for r in T.serial(cache_rows, m_n):
                                    csum[0] = T.simd.vadd(csum[0], T.simd.vld(work_ub[r, p * 2, 0], dist="NORM_B32"), mask=m32)
                                inv_c = T.simd.vdiv(v_one, T.simd.vadds(csum[0], eps, mask=m32), mask=m32, precision="ftz_true")
                                for r in T.unroll(cache_rows):
                                    T.simd.vsts(work_ub[r, p * 2, 0], T.simd.vmul(buf[r], inv_c, mask=m32), mask=m32, dist="NORM_B32")
                                for r in T.serial(cache_rows, m_n):
                                    T.simd.vsts(
                                        work_ub[r, p * 2, 0],
                                        T.simd.vmul(T.simd.vld(work_ub[r, p * 2, 0], dist="NORM_B32"), inv_c, mask=m32),
                                        mask=m32,
                                        dist="NORM_B32",
                                    )
                            T.simd.mem_bar("VST_VLD")

                            for _it in T.serial(n_iter - 1):
                                for r in T.serial(m_n):
                                    for p in T.unroll(half_n):
                                        buf[p] = T.simd.vld(work_ub[r, p * 2, 0], dist="NORM_B32")
                                    rsum[0] = buf[0]
                                    for p in T.unroll(1, half_n):
                                        rsum[0] = T.simd.vadd(rsum[0], buf[p], mask=m32)
                                    rs2 = T.simd.vadd(rsum[0], T.simd.vselr(rsum[0], idx_swap32), mask=m32)
                                    inv_r2 = T.simd.vdiv(v_one, T.simd.vadds(rs2, eps, mask=m32), mask=m32, precision="ftz_true")
                                    for p in T.unroll(half_n):
                                        T.simd.vsts(work_ub[r, p * 2, 0], T.simd.vmul(buf[p], inv_r2, mask=m32), mask=m32, dist="NORM_B32")
                                T.simd.mem_bar("VST_VLD")

                                for p in T.serial(half_n):
                                    csum[0] = v_zero
                                    for r in T.unroll(cache_rows):
                                        buf[r] = T.simd.vld(work_ub[r, p * 2, 0], dist="NORM_B32")
                                        csum[0] = T.simd.vadd(csum[0], buf[r], mask=m32)
                                    for r in T.serial(cache_rows, m_n):
                                        csum[0] = T.simd.vadd(csum[0], T.simd.vld(work_ub[r, p * 2, 0], dist="NORM_B32"), mask=m32)
                                    inv_c2 = T.simd.vdiv(v_one, T.simd.vadds(csum[0], eps, mask=m32), mask=m32, precision="ftz_true")
                                    for r in T.unroll(cache_rows):
                                        T.simd.vsts(work_ub[r, p * 2, 0], T.simd.vmul(buf[r], inv_c2, mask=m32), mask=m32, dist="NORM_B32")
                                    for r in T.serial(cache_rows, m_n):
                                        T.simd.vsts(
                                            work_ub[r, p * 2, 0],
                                            T.simd.vmul(T.simd.vld(work_ub[r, p * 2, 0], dist="NORM_B32"), inv_c2, mask=m32),
                                            mask=m32,
                                            dist="NORM_B32",
                                        )
                                T.simd.mem_bar("VST_VLD")

                        with T.SimdVF():
                            m32 = T.simd.pset(32, "PAT_ALL")
                            lane = T.simd.vci(0, "int32")
                            sub_lane = T.simd.vand(lane, T.simd.vdup(31, "int32", mask=m32), mask=m32)
                            is_hi = T.simd.vshrs(lane, 5, mask=m32)
                            m_hi_active = T.simd.vcmps(sub_lane, b_cur, mask=m32, op="lt")
                            base_off = T.simd.vmuls(sub_lane, mat_elems, mask=m32)
                            for r in T.serial(m_n):
                                for p in T.serial(half_n):
                                    off_elem = T.simd.vadd(T.simd.vadds(base_off, r * m_n + p * 2, mask=m32), is_hi, mask=m32)
                                    idx = T.reinterpret("uint32x64", off_elem)
                                    T.simd.vscatter(T.simd.vld(work_ub[r, p * 2, 0], dist="NORM_B32"), io_ub[0], idx, mask=m_hi_active)

                        T.copy(io_ub[:elem_cnt], CombOut[elem_off : elem_off + elem_cnt])

            return main_b32_work

        # ------------------------------------------------------------------
        # 5. Generic fallback for any odd/arbitrary hc_mult
        # ------------------------------------------------------------------
        io_cap = ((b_tile * mat_elems + 63) // 64) * 64 + 64
        use_double_buf = m_n <= 8
        stages = 2 if (use_double_buf and use_pipelined) else 1

        @T.prim_func
        def main_gen(
            Comb: T.Tensor((n_elems,), "float32"),
            CombOut: T.Tensor((n_elems,), "float32"),
            eps: T.float32,
            b_dim: T.int32,
        ):
            with T.Kernel(num_cores) as cid:
                io_in_ub = T.alloc_shared((io_cap,), "float32")
                io_out_ub = T.alloc_shared((io_cap,), "float32") if stages > 1 else io_in_ub
                work_ub = T.alloc_shared((m_n, m_n, b_tile), "float32")
                if stages > 1:
                    T.annotate_buffer_versions({io_in_ub: 2, io_out_ub: 2, work_ub: 1})
                else:
                    T.annotate_buffer_versions({io_in_ub: 1, work_ub: 1})
                if schedule_mode == "tuned":
                    T.annotate_unlimit_memory("shared")

                base_b = b_dim // num_cores
                rem_b = b_dim % num_cores
                b_start = cid * base_b + T.min(cid, rem_b)
                b_count = base_b + T.if_then_else(cid < rem_b, 1, 0)
                num_chunks = (b_count + b_tile - 1) // b_tile

                for ch in T.Pipelined(num_chunks, num_stages=stages):
                    b0 = b_start + ch * b_tile
                    b_cur = T.min(b_tile, b_count - ch * b_tile)
                    elem_off = b0 * mat_elems
                    elem_cnt = b_cur * mat_elems
                    T.copy(Comb[elem_off : elem_off + elem_cnt], io_in_ub[:elem_cnt])

                    with T.SimdVF():
                        m32 = T.simd.pset(32, "PAT_ALL")
                        v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                        v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
                        m_act = T.simd.update_mask(b_cur, width=32)
                        lane = T.simd.vci(0, "int32")
                        base_off = T.simd.vmuls(lane, mat_elems, mask=m32)

                        for r in T.serial(m_n):
                            for c in T.serial(m_n):
                                idx = T.reinterpret("uint32x64", T.simd.vadds(base_off, r * m_n + c, mask=m32))
                                gv = T.simd.vsel(T.simd.vgather2(io_in_ub[0], idx, mask=m_act), v_zero, m_act)
                                T.simd.vsts(work_ub[r, c, 0], gv, mask=m32, dist="NORM_B32")
                        T.simd.mem_bar("VST_VLD")

                        for r in T.serial(m_n):
                            rmax = T.simd.alloc_local((1,), "float32")
                            rmax[0] = T.simd.vld(work_ub[r, 0, 0], dist="NORM_B32")
                            for c in T.serial(1, m_n):
                                rmax[0] = T.simd.vmax(rmax[0], T.simd.vld(work_ub[r, c, 0], dist="NORM_B32"), mask=m32)
                            rsum = T.simd.alloc_local((1,), "float32")
                            rsum[0] = v_zero
                            for c in T.serial(m_n):
                                ev = T.simd.vexpdif(T.simd.vld(work_ub[r, c, 0], dist="NORM_B32"), rmax[0], mask=m32)
                                T.simd.vsts(work_ub[r, c, 0], ev, mask=m32, dist="NORM_B32")
                                rsum[0] = T.simd.vadd(rsum[0], ev, mask=m32)
                            T.simd.mem_bar("VST_VLD")
                            inv_r = T.simd.vdiv(v_one, rsum[0], mask=m32, precision="ftz_true")
                            for c in T.serial(m_n):
                                prob = T.simd.vadds(T.simd.vmul(T.simd.vld(work_ub[r, c, 0], dist="NORM_B32"), inv_r, mask=m32), eps, mask=m32)
                                T.simd.vsts(work_ub[r, c, 0], prob, mask=m32, dist="NORM_B32")
                        T.simd.mem_bar("VST_VLD")

                        for c in T.serial(m_n):
                            csum = T.simd.alloc_local((1,), "float32")
                            csum[0] = v_zero
                            for r in T.serial(m_n):
                                csum[0] = T.simd.vadd(csum[0], T.simd.vld(work_ub[r, c, 0], dist="NORM_B32"), mask=m32)
                            inv_c = T.simd.vdiv(v_one, T.simd.vadds(csum[0], eps, mask=m32), mask=m32, precision="ftz_true")
                            for r in T.serial(m_n):
                                pv = T.simd.vld(work_ub[r, c, 0], dist="NORM_B32")
                                T.simd.vsts(work_ub[r, c, 0], T.simd.vmul(pv, inv_c, mask=m32), mask=m32, dist="NORM_B32")
                        T.simd.mem_bar("VST_VLD")

                        for _it in T.serial(n_iter - 1):
                            for r in T.serial(m_n):
                                rsum2 = T.simd.alloc_local((1,), "float32")
                                rsum2[0] = v_zero
                                for c in T.serial(m_n):
                                    rsum2[0] = T.simd.vadd(rsum2[0], T.simd.vld(work_ub[r, c, 0], dist="NORM_B32"), mask=m32)
                                inv_r2 = T.simd.vdiv(v_one, T.simd.vadds(rsum2[0], eps, mask=m32), mask=m32, precision="ftz_true")
                                for c in T.serial(m_n):
                                    rv = T.simd.vld(work_ub[r, c, 0], dist="NORM_B32")
                                    T.simd.vsts(work_ub[r, c, 0], T.simd.vmul(rv, inv_r2, mask=m32), mask=m32, dist="NORM_B32")
                            T.simd.mem_bar("VST_VLD")
                            for c in T.serial(m_n):
                                csum2 = T.simd.alloc_local((1,), "float32")
                                csum2[0] = v_zero
                                for r in T.serial(m_n):
                                    csum2[0] = T.simd.vadd(csum2[0], T.simd.vld(work_ub[r, c, 0], dist="NORM_B32"), mask=m32)
                                inv_c2 = T.simd.vdiv(v_one, T.simd.vadds(csum2[0], eps, mask=m32), mask=m32, precision="ftz_true")
                                for r in T.serial(m_n):
                                    cv = T.simd.vld(work_ub[r, c, 0], dist="NORM_B32")
                                    T.simd.vsts(work_ub[r, c, 0], T.simd.vmul(cv, inv_c2, mask=m32), mask=m32, dist="NORM_B32")
                            T.simd.mem_bar("VST_VLD")

                        for r in T.serial(m_n):
                            for c in T.serial(m_n):
                                idx = T.reinterpret("uint32x64", T.simd.vadds(base_off, r * m_n + c, mask=m32))
                                val = T.simd.vld(work_ub[r, c, 0], dist="NORM_B32")
                                T.simd.vscatter(val, io_out_ub[0], idx, mask=m_act)

                    T.copy(io_out_ub[:elem_cnt], CombOut[elem_off : elem_off + elem_cnt])

        return main_gen


def _get_kernel(hc_mult: int, iter_step: int, b_dim: int, schedule_mode: str):
    cfg = _select_mhc_sinkhorn_config(
        hc_mult=hc_mult, b_dim=b_dim, iter_step=iter_step, schedule_mode=schedule_mode
    )
    key = (
        hc_mult,
        int(cfg["iter_step"]),
        int(cfg["batch_tile"]),
        int(cfg["num_cores"]),
        str(cfg["regime"]),
        bool(cfg["use_pipelined"]),
        schedule_mode,
    )
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        with _ascend_target_ctx():
            prog = _build_mhc_sinkhorn_prim_func(
                hc_mult=hc_mult,
                iter_step=int(cfg["iter_step"]),
                batch_tile=int(cfg["batch_tile"]),
                num_cores=int(cfg["num_cores"]),
                regime=str(cfg["regime"]),
                use_pipelined=bool(cfg["use_pipelined"]),
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


_TRACK_B_CODE = r"""
#include <stdint.h>
#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/simd_inst.h>

__simd_callee__ inline vector_f32 fast_rcp(vector_f32 x, vector_bool m32) {
  return asc_div(simd_inst::vdup<float>(1.0f, m32, MODE_ZEROING), x, m32);
}

__simd_vf__ inline void vf_hc4(__ubuf__ float* io, int32_t b_cur, int32_t n_iter, float eps) {
  vector_bool m32 = asc_create_mask_b32(PAT_ALL);
  vector_f32 v_zero = simd_inst::vdup<float>(0.0f, m32, MODE_ZEROING);
  vector_s32 lane = simd_inst::vci<int32_t>(0, INC_ORDER);
  vector_s32 sub_lane = simd_inst::vand(lane, simd_inst::vdup<int32_t>(31, m32, MODE_ZEROING), m32, MODE_ZEROING);
  vector_s32 is_hi = simd_inst::vshrs(lane, 5, m32, MODE_ZEROING);
  vector_s32 idx_swap32 = simd_inst::vxor(lane, simd_inst::vdup<int32_t>(32, m32, MODE_ZEROING), m32, MODE_ZEROING);
  vector_bool m_act = simd_inst::vcmps_lt(sub_lane, b_cur, m32);
  vector_s32 base_off = simd_inst::vmuls(sub_lane, 16, m32, MODE_ZEROING);

  vector_f32 x[8];
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
      vector_s32 off = simd_inst::vadd(simd_inst::vadds(base_off, r * 4 + p * 2, m32, MODE_ZEROING), is_hi, m32, MODE_ZEROING);
      x[r * 2 + p] = simd_inst::vsel(simd_inst::vgather2(io, *(vector_u32*)&off, m_act), v_zero, m_act);
    }
  }
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    vector_f32 rm0 = simd_inst::vmax(x[r * 2], x[r * 2 + 1], m32, MODE_ZEROING);
    vector_u32 rm_sw = simd_inst::vselr(*(vector_u32*)&rm0, *(vector_u32*)&idx_swap32);
    vector_f32 rm = simd_inst::vmax(rm0, *(vector_f32*)&rm_sw, m32, MODE_ZEROING);
    vector_f32 e0 = simd_inst::vexpdif(x[r * 2], rm, m32);
    vector_f32 e1 = simd_inst::vexpdif(x[r * 2 + 1], rm, m32);
    vector_f32 rs0 = simd_inst::vadd(e0, e1, m32, MODE_ZEROING);
    vector_u32 rs_sw = simd_inst::vselr(*(vector_u32*)&rs0, *(vector_u32*)&idx_swap32);
    vector_f32 rs = simd_inst::vadd(rs0, *(vector_f32*)&rs_sw, m32, MODE_ZEROING);
    vector_f32 inv_r = fast_rcp(rs, m32);
    x[r * 2] = simd_inst::vadds(simd_inst::vmul(e0, inv_r, m32, MODE_ZEROING), eps, m32, MODE_ZEROING);
    x[r * 2 + 1] = simd_inst::vadds(simd_inst::vmul(e1, inv_r, m32, MODE_ZEROING), eps, m32, MODE_ZEROING);
  }
  #pragma unroll
  for (int p = 0; p < 2; ++p) {
    vector_f32 cs = simd_inst::vadd(
        simd_inst::vadd(x[p], x[2 + p], m32, MODE_ZEROING),
        simd_inst::vadd(x[4 + p], x[6 + p], m32, MODE_ZEROING),
        m32, MODE_ZEROING);
    vector_f32 inv_c = fast_rcp(simd_inst::vadds(cs, eps, m32, MODE_ZEROING), m32);
    #pragma unroll
    for (int r = 0; r < 4; ++r) x[r * 2 + p] = simd_inst::vmul(x[r * 2 + p], inv_c, m32, MODE_ZEROING);
  }
  for (int it = 1; it < n_iter; ++it) {
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      vector_f32 rs0 = simd_inst::vadd(x[r * 2], x[r * 2 + 1], m32, MODE_ZEROING);
      vector_u32 rs_sw = simd_inst::vselr(*(vector_u32*)&rs0, *(vector_u32*)&idx_swap32);
      vector_f32 rs = simd_inst::vadd(rs0, *(vector_f32*)&rs_sw, m32, MODE_ZEROING);
      vector_f32 inv_r = fast_rcp(simd_inst::vadds(rs, eps, m32, MODE_ZEROING), m32);
      x[r * 2] = simd_inst::vmul(x[r * 2], inv_r, m32, MODE_ZEROING);
      x[r * 2 + 1] = simd_inst::vmul(x[r * 2 + 1], inv_r, m32, MODE_ZEROING);
    }
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
      vector_f32 cs = simd_inst::vadd(
          simd_inst::vadd(x[p], x[2 + p], m32, MODE_ZEROING),
          simd_inst::vadd(x[4 + p], x[6 + p], m32, MODE_ZEROING),
          m32, MODE_ZEROING);
      vector_f32 inv_c = fast_rcp(simd_inst::vadds(cs, eps, m32, MODE_ZEROING), m32);
      #pragma unroll
      for (int r = 0; r < 4; ++r) x[r * 2 + p] = simd_inst::vmul(x[r * 2 + p], inv_c, m32, MODE_ZEROING);
    }
  }
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
      vector_s32 off = simd_inst::vadd(simd_inst::vadds(base_off, r * 4 + p * 2, m32, MODE_ZEROING), is_hi, m32, MODE_ZEROING);
      simd_inst::vscatter(x[r * 2 + p], io, *(vector_u32*)&off, m_act);
    }
  }
}

__global__ __aicore__ void mhc_hc4_kernel(__gm__ float* comb, __gm__ float* out, int32_t b_dim, int32_t n_iter, float eps, int32_t n_cores) {
  __ubuf__ float* io = (__ubuf__ float*)0;
  int32_t cid = get_block_idx();
  int32_t base_b = b_dim / n_cores;
  int32_t rem_b = b_dim % n_cores;
  int32_t b_start = cid * base_b + (cid < rem_b ? cid : rem_b);
  int32_t b_count = base_b + (cid < rem_b ? 1 : 0);
  for (int32_t ch = 0; ch * 32 < b_count; ++ch) {
    int32_t b0 = b_start + ch * 32;
    int32_t b_cur = b_count - ch * 32;
    if (b_cur > 32) b_cur = 32;
    uint32_t bytes = (uint32_t)(b_cur * 16 * 4);
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)io, (__gm__ uint8_t*)(comb + (int64_t)b0 * 16), 1, bytes, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), bytes, bytes);
    asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID0);
    vf_hc4(io, b_cur, n_iter, eps);
    asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_copy_ub2gm_align((__gm__ uint8_t*)(out + (int64_t)b0 * 16), (__ubuf__ uint8_t*)io, 1, bytes, static_cast<asc_store_l2_cache_mode>(4), bytes, bytes);
    asc_sync_notify(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    asc_sync_wait(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
}

template<int HC>
__simd_vf__ inline void vf_gather_work(__ubuf__ float* io, __ubuf__ float* work, int32_t b_cur) {
  constexpr int HALF = HC / 2;
  constexpr int MAT = HC * HC;
  vector_bool m32 = asc_create_mask_b32(PAT_ALL);
  vector_f32 v_zero = simd_inst::vdup<float>(0.0f, m32, MODE_ZEROING);
  vector_s32 lane = simd_inst::vci<int32_t>(0, INC_ORDER);
  vector_s32 sub_lane = simd_inst::vand(lane, simd_inst::vdup<int32_t>(31, m32, MODE_ZEROING), m32, MODE_ZEROING);
  vector_s32 is_hi = simd_inst::vshrs(lane, 5, m32, MODE_ZEROING);
  vector_bool m_act = simd_inst::vcmps_lt(sub_lane, b_cur, m32);
  vector_s32 base_off = simd_inst::vmuls(sub_lane, MAT, m32, MODE_ZEROING);
  for (int r = 0; r < HC; ++r) {
    for (int p = 0; p < HALF; ++p) {
      vector_s32 off = simd_inst::vadd(simd_inst::vadds(base_off, r * HC + p * 2, m32, MODE_ZEROING), is_hi, m32, MODE_ZEROING);
      vector_f32 gv = simd_inst::vsel(simd_inst::vgather2(io, *(vector_u32*)&off, m_act), v_zero, m_act);
      simd_inst::vsts_norm(gv, work + (r * HALF + p) * 64, 0, m32);
    }
  }
  simd_inst::mem_bar(VST_VLD);
}

template<int HC>
__simd_vf__ inline void vf_compute_work(__ubuf__ float* work, int32_t n_iter, float eps) {
  constexpr int HALF = HC / 2;
  constexpr int CACHE_R = HC <= 8 ? HC : 8;
  vector_bool m32 = asc_create_mask_b32(PAT_ALL);
  vector_f32 v_zero = simd_inst::vdup<float>(0.0f, m32, MODE_ZEROING);
  vector_f32 v_one = simd_inst::vdup<float>(1.0f, m32, MODE_ZEROING);
  vector_s32 lane = simd_inst::vci<int32_t>(0, INC_ORDER);
  vector_s32 idx_swap32 = simd_inst::vxor(lane, simd_inst::vdup<int32_t>(32, m32, MODE_ZEROING), m32, MODE_ZEROING);

  vector_f32 buf[CACHE_R];
  for (int r = 0; r < HC; ++r) {
    #pragma unroll
    for (int p = 0; p < HALF; ++p) buf[p] = simd_inst::vlds_norm<float>(work + (r * HALF + p) * 64, 0);
    vector_f32 rmax = buf[0];
    #pragma unroll
    for (int p = 1; p < HALF; ++p) rmax = simd_inst::vmax(rmax, buf[p], m32, MODE_ZEROING);
    vector_u32 rm_sw = simd_inst::vselr(*(vector_u32*)&rmax, *(vector_u32*)&idx_swap32);
    vector_f32 rm = simd_inst::vmax(rmax, *(vector_f32*)&rm_sw, m32, MODE_ZEROING);
    vector_f32 rsum = v_zero;
    #pragma unroll
    for (int p = 0; p < HALF; ++p) {
      buf[p] = simd_inst::vexpdif(buf[p], rm, m32);
      rsum = simd_inst::vadd(rsum, buf[p], m32, MODE_ZEROING);
    }
    vector_u32 rs_sw = simd_inst::vselr(*(vector_u32*)&rsum, *(vector_u32*)&idx_swap32);
    vector_f32 rs = simd_inst::vadd(rsum, *(vector_f32*)&rs_sw, m32, MODE_ZEROING);
    vector_f32 inv_r = asc_div(v_one, rs, m32);
    #pragma unroll
    for (int p = 0; p < HALF; ++p) {
      simd_inst::vsts_norm(simd_inst::vadds(simd_inst::vmul(buf[p], inv_r, m32, MODE_ZEROING), eps, m32, MODE_ZEROING), work + (r * HALF + p) * 64, 0, m32);
    }
  }
  simd_inst::mem_bar(VST_VLD);

  for (int p = 0; p < HALF; ++p) {
    vector_f32 csum = v_zero;
    #pragma unroll
    for (int r = 0; r < CACHE_R; ++r) {
      buf[r] = simd_inst::vlds_norm<float>(work + (r * HALF + p) * 64, 0);
      csum = simd_inst::vadd(csum, buf[r], m32, MODE_ZEROING);
    }
    for (int r = CACHE_R; r < HC; ++r) {
      csum = simd_inst::vadd(csum, simd_inst::vlds_norm<float>(work + (r * HALF + p) * 64, 0), m32, MODE_ZEROING);
    }
    vector_f32 inv_c = asc_div(v_one, simd_inst::vadds(csum, eps, m32, MODE_ZEROING), m32);
    #pragma unroll
    for (int r = 0; r < CACHE_R; ++r) {
      simd_inst::vsts_norm(simd_inst::vmul(buf[r], inv_c, m32, MODE_ZEROING), work + (r * HALF + p) * 64, 0, m32);
    }
    for (int r = CACHE_R; r < HC; ++r) {
      simd_inst::vsts_norm(simd_inst::vmul(simd_inst::vlds_norm<float>(work + (r * HALF + p) * 64, 0), inv_c, m32, MODE_ZEROING), work + (r * HALF + p) * 64, 0, m32);
    }
  }
  simd_inst::mem_bar(VST_VLD);

  for (int it = 1; it < n_iter; ++it) {
    for (int r = 0; r < HC; ++r) {
      #pragma unroll
      for (int p = 0; p < HALF; ++p) buf[p] = simd_inst::vlds_norm<float>(work + (r * HALF + p) * 64, 0);
      vector_f32 rsum = buf[0];
      #pragma unroll
      for (int p = 1; p < HALF; ++p) rsum = simd_inst::vadd(rsum, buf[p], m32, MODE_ZEROING);
      vector_u32 rs_sw = simd_inst::vselr(*(vector_u32*)&rsum, *(vector_u32*)&idx_swap32);
      vector_f32 rs = simd_inst::vadd(rsum, *(vector_f32*)&rs_sw, m32, MODE_ZEROING);
      vector_f32 inv_r = asc_div(v_one, simd_inst::vadds(rs, eps, m32, MODE_ZEROING), m32);
      #pragma unroll
      for (int p = 0; p < HALF; ++p) {
        simd_inst::vsts_norm(simd_inst::vmul(buf[p], inv_r, m32, MODE_ZEROING), work + (r * HALF + p) * 64, 0, m32);
      }
    }
    simd_inst::mem_bar(VST_VLD);
    for (int p = 0; p < HALF; ++p) {
      vector_f32 csum = v_zero;
      #pragma unroll
      for (int r = 0; r < CACHE_R; ++r) {
        buf[r] = simd_inst::vlds_norm<float>(work + (r * HALF + p) * 64, 0);
        csum = simd_inst::vadd(csum, buf[r], m32, MODE_ZEROING);
      }
      for (int r = CACHE_R; r < HC; ++r) {
        csum = simd_inst::vadd(csum, simd_inst::vlds_norm<float>(work + (r * HALF + p) * 64, 0), m32, MODE_ZEROING);
      }
      vector_f32 inv_c = asc_div(v_one, simd_inst::vadds(csum, eps, m32, MODE_ZEROING), m32);
      #pragma unroll
      for (int r = 0; r < CACHE_R; ++r) {
        simd_inst::vsts_norm(simd_inst::vmul(buf[r], inv_c, m32, MODE_ZEROING), work + (r * HALF + p) * 64, 0, m32);
      }
      for (int r = CACHE_R; r < HC; ++r) {
        simd_inst::vsts_norm(simd_inst::vmul(simd_inst::vlds_norm<float>(work + (r * HALF + p) * 64, 0), inv_c, m32, MODE_ZEROING), work + (r * HALF + p) * 64, 0, m32);
      }
    }
    simd_inst::mem_bar(VST_VLD);
  }
}

template<int HC>
__simd_vf__ inline void vf_scatter_work(__ubuf__ float* io, __ubuf__ float* work, int32_t b_cur) {
  constexpr int HALF = HC / 2;
  constexpr int MAT = HC * HC;
  vector_bool m32 = asc_create_mask_b32(PAT_ALL);
  vector_s32 lane = simd_inst::vci<int32_t>(0, INC_ORDER);
  vector_s32 sub_lane = simd_inst::vand(lane, simd_inst::vdup<int32_t>(31, m32, MODE_ZEROING), m32, MODE_ZEROING);
  vector_s32 is_hi = simd_inst::vshrs(lane, 5, m32, MODE_ZEROING);
  vector_bool m_act = simd_inst::vcmps_lt(sub_lane, b_cur, m32);
  vector_s32 base_off = simd_inst::vmuls(sub_lane, MAT, m32, MODE_ZEROING);
  for (int r = 0; r < HC; ++r) {
    for (int p = 0; p < HALF; ++p) {
      vector_s32 off = simd_inst::vadd(simd_inst::vadds(base_off, r * HC + p * 2, m32, MODE_ZEROING), is_hi, m32, MODE_ZEROING);
      simd_inst::vscatter(simd_inst::vlds_norm<float>(work + (r * HALF + p) * 64, 0), io, *(vector_u32*)&off, m_act);
    }
  }
}

template<int HC>
__global__ __aicore__ void mhc_b32_work_kernel(__gm__ float* comb, __gm__ float* out, int32_t b_dim, int32_t n_iter, float eps, int32_t n_cores) {
  constexpr int MAT = HC * HC;
  __ubuf__ float* io = (__ubuf__ float*)0;
  __ubuf__ float* work = (__ubuf__ float*)((32 * MAT + 64) * 4);
  int32_t cid = get_block_idx();
  int32_t base_b = b_dim / n_cores;
  int32_t rem_b = b_dim % n_cores;
  int32_t b_start = cid * base_b + (cid < rem_b ? cid : rem_b);
  int32_t b_count = base_b + (cid < rem_b ? 1 : 0);
  for (int32_t ch = 0; ch * 32 < b_count; ++ch) {
    int32_t b0 = b_start + ch * 32;
    int32_t b_cur = b_count - ch * 32;
    if (b_cur > 32) b_cur = 32;
    uint32_t bytes = (uint32_t)(b_cur * MAT * 4);
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)io, (__gm__ uint8_t*)(comb + (int64_t)b0 * MAT), 1, bytes, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), bytes, bytes);
    asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID0);
    vf_gather_work<HC>(io, work, b_cur);
    vf_compute_work<HC>(work, n_iter, eps);
    vf_scatter_work<HC>(io, work, b_cur);
    asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_copy_ub2gm_align((__gm__ uint8_t*)(out + (int64_t)b0 * MAT), (__ubuf__ uint8_t*)io, 1, bytes, static_cast<asc_store_l2_cache_mode>(4), bytes, bytes);
    asc_sync_notify(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    asc_sync_wait(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
}

extern "C" void launch_mhc_sinkhorn(void* comb, void* out, int32_t b_dim, int32_t hc, int32_t n_iter, float eps, void* stream) {
  int32_t n_cores = b_dim <= 32 ? 1 : (b_dim < 56 ? b_dim : 56);
  if (hc == 4) mhc_hc4_kernel<<<n_cores, nullptr, (aclrtStream)stream>>>((__gm__ float*)comb, (__gm__ float*)out, b_dim, n_iter, eps, n_cores);
  else if (hc == 6) mhc_b32_work_kernel<6><<<n_cores, nullptr, (aclrtStream)stream>>>((__gm__ float*)comb, (__gm__ float*)out, b_dim, n_iter, eps, n_cores);
  else if (hc == 8) mhc_b32_work_kernel<8><<<n_cores, nullptr, (aclrtStream)stream>>>((__gm__ float*)comb, (__gm__ float*)out, b_dim, n_iter, eps, n_cores);
  else if (hc == 12) mhc_b32_work_kernel<12><<<n_cores, nullptr, (aclrtStream)stream>>>((__gm__ float*)comb, (__gm__ float*)out, b_dim, n_iter, eps, n_cores);
  else if (hc == 16) mhc_b32_work_kernel<16><<<n_cores, nullptr, (aclrtStream)stream>>>((__gm__ float*)comb, (__gm__ float*)out, b_dim, n_iter, eps, n_cores);
}
"""


def _get_track_b_lib():
    global _TRACK_B_LIB
    if _TRACK_B_LIB is None:
        from tilelang.contrib.bisheng import compile_ascend

        tl_src_dir = Path(tilelang.__file__).resolve().parents[1] / "src"
        digest = hashlib.sha1(_TRACK_B_CODE.encode("utf-8")).hexdigest()[:12]
        so_path = Path(tempfile.gettempdir()) / f"tilefoundry_mhc_sinkhorn_{digest}.so"
        if not so_path.exists():
            compile_ascend(
                _TRACK_B_CODE,
                target_format="so",
                npu_arch="dav-3510",
                options=[f"-I{tl_src_dir}", *PURE_SIMDVF_UBUF_FLAGS],
                path_target=str(so_path),
                verbose=False,
            )
        lib = ctypes.CDLL(str(so_path))
        lib.launch_mhc_sinkhorn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_void_p,
        ]
        _TRACK_B_LIB = lib
    return _TRACK_B_LIB


def mhc_sinkhorn(comb: torch.Tensor, iter_step: int = 20, eps: float = 1e-6, *, schedule_mode: str = "tuned") -> torch.Tensor:
    if comb.device.type != "npu":
        out = comb
        row_max = out.amax(dim=-1, keepdim=True)
        out = torch.exp(out - row_max)
        row_sum = out.sum(dim=-1, keepdim=True)
        out = out / row_sum + eps
        col_sum = out.sum(dim=-2, keepdim=True)
        out = out / (col_sum + eps)
        for _ in range(int(iter_step) - 1):
            row_sum = out.sum(dim=-1, keepdim=True)
            out = out / (row_sum + eps)
            col_sum = out.sum(dim=-2, keepdim=True)
            out = out / (col_sum + eps)
        return out
    b_dim, hc_mult, _ = comb.shape
    if (
        schedule_mode == "tuned"
        and comb.dtype == torch.float32
        and ((hc_mult == 4 and b_dim <= 56 * 32) or hc_mult in (6, 8, 12, 16))
    ):
        comb_c = comb.contiguous()
        out = torch.empty_like(comb_c)
        stream = torch.npu.current_stream().npu_stream
        _get_track_b_lib().launch_mhc_sinkhorn(
            comb_c.data_ptr(),
            out.data_ptr(),
            int(b_dim),
            int(hc_mult),
            int(iter_step),
            float(eps),
            stream,
        )
        return out
    compiled = _get_kernel(hc_mult, int(iter_step), b_dim, schedule_mode)
    return compiled(comb.contiguous().view(-1), float(eps), int(b_dim)).view(b_dim, hc_mult, hc_mult)


kernel = mhc_sinkhorn
