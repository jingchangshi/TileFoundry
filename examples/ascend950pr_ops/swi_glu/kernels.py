"""TileLang Ascend950PR Pure RegBase SimdVF / SimtVF kernel implementation for SwiGlu."""

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


def _select_swi_glu_config(
    half_dim: int,
    tl_dtype: str = "bfloat16",
    *,
    m_rows: int = 1024,
    inner_dim: int = 1,
    schedule_mode: str = "tuned",
) -> dict[str, object]:
    """Select tile, D-chunking, and multi-buffer configuration for SwiGlu."""
    elem_bytes = 4 if tl_dtype == "float32" else 2
    ub_budget = 256 * 1024 - 4096 if schedule_mode == "tuned" else 192 * 1024

    if inner_dim > 1:
        w_tile = 2048 if tl_dtype == "float32" else 4096
        stages = 2 if schedule_mode == "tuned" else 1
        return {
            "regime": "contiguous_split_axis",
            "tile_m": 1,
            "d_tile": w_tile,
            "num_stages": stages,
            "half_dim": int(half_dim),
            "inner_dim": int(inner_dim),
            "tl_dtype": tl_dtype,
            "pass_configs": {},
            "compile_flags": list(PURE_SIMDVF_UBUF_FLAGS) if schedule_mode == "tuned" else [],
        }

    if half_dim == 1:
        tile_m = 2048 if (tl_dtype != "float32" and m_rows >= 56 * 2048) else (
            1024 if m_rows >= 56 * 1024 else max(64, ((max(1, m_rows // 112) + 63) // 64) * 64)
        )
        stages = 2 if (schedule_mode == "tuned" and m_rows >= 56 * 2) else 1
        return {
            "regime": "ultra_narrow_packed",
            "tile_m": tile_m,
            "d_tile": 64,
            "num_stages": stages,
            "half_dim": int(half_dim),
            "inner_dim": 1,
            "tl_dtype": tl_dtype,
            "pass_configs": {},
            "compile_flags": list(PURE_SIMDVF_UBUF_FLAGS) if schedule_mode == "tuned" else [],
        }

    if half_dim <= 32 or (tl_dtype == "float32" and (half_dim % 16) != 0 and half_dim <= 64):
        rows_blk = 256 if half_dim <= 16 else (64 if half_dim <= 32 else 32)
        if m_rows < 56 * rows_blk * 2:
            rows_blk = max(8, min(rows_blk, max(8, m_rows // 112)))
        stages = 2 if (schedule_mode == "tuned" and m_rows >= 56 * 2) else 1
        return {
            "regime": "narrow_simt_packed",
            "tile_m": rows_blk,
            "d_tile": int(half_dim),
            "num_stages": stages,
            "half_dim": int(half_dim),
            "inner_dim": 1,
            "tl_dtype": tl_dtype,
            "pass_configs": {"tl.disable_data_race_check": True},
            "compile_flags": [],
        }

    half_alloc = ((max(1, int(half_dim)) + 63) // 64) * 64
    stages = 2 if (schedule_mode == "tuned" and m_rows >= 56 * 2) else 1
    max_single_d_tile = 4096 if tl_dtype == "float32" else 8192
    if half_alloc * (2 * stages + stages) * elem_bytes > ub_budget:
        d_tile = max_single_d_tile
        max_rows = 1
        regime = "d_chunked_streaming"
    else:
        d_tile = half_alloc
        row_bytes = (2 * stages + stages) * d_tile * elem_bytes
        max_rows = max(1, min(64, ub_budget // max(1, row_bytes)))
        if (half_dim % 64) != 0:
            max_rows = min(max_rows, 32)
        if max_rows > 1 and m_rows >= 112 and m_rows < 56 * max_rows * 2:
            max_rows = max(1, min(max_rows, max(1, m_rows // 112)))
        elif max_rows > 1 and m_rows < 56 * max_rows:
            max_rows = max(1, min(max_rows, max(1, m_rows // 56)))
        regime = "aligned_single_x" if (half_dim % 64) == 0 else "unaligned_predicated"

    return {
        "regime": regime,
        "tile_m": max_rows,
        "d_tile": int(d_tile),
        "num_stages": stages,
        "half_dim": int(half_dim),
        "inner_dim": 1,
        "tl_dtype": tl_dtype,
        "pass_configs": {},
        "compile_flags": list(PURE_SIMDVF_UBUF_FLAGS) if schedule_mode == "tuned" else [],
    }


def _build_swi_glu_prim_func(
    half_dim: int = 1024,
    tl_dtype: str = "bfloat16",
    tile_m: int = 4,
    d_tile: int | None = None,
    num_stages: int = 2,
    inner_dim: int = 1,
    regime: str | None = None,
    schedule_mode: str = "tuned",
    **_extra_kwargs,
):
    """Build Ascend 950 Pure RegBase SimdVF / SimtVF @T.prim_func for SwiGlu: silu(x0) * x1."""
    if half_dim <= 0:
        raise ValueError(f"half_dim must be positive, got {half_dim}")
    if tl_dtype not in ("float16", "bfloat16", "float32"):
        raise ValueError(f"unsupported dtype {tl_dtype!r}")

    num_cores = 56
    stages = max(1, min(2, int(num_stages)))
    pipe_ann = {"enable_offset": True} if schedule_mode == "tuned" else None

    @T.macro
    def _swiglu_vec_chunks(x0_ub, x1_ub, y_ub, slot, n_chunks, n_rem):
        m32 = T.simd.pset(32, "PAT_ALL")
        m16 = T.simd.pset(16, "PAT_ALL")
        v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
        for c in T.serial(n_chunks):
            off = c * 64
            if tl_dtype == "float32":
                x0 = T.simd.vld(x0_ub[slot, off], dist="NORM_B32")
                x1 = T.simd.vld(x1_ub[slot, off], dist="NORM_B32")
            else:
                x0 = T.simd.vcvt(T.simd.vld(x0_ub[slot, off], dist="US_B16"), "float32", mask=m16, part=0)
                x1 = T.simd.vcvt(T.simd.vld(x1_ub[slot, off], dist="US_B16"), "float32", mask=m16, part=0)
            neg_x0 = T.simd.vmuls(x0, T.float32(-1.0), mask=m32)
            sig = T.simd.vdiv(
                v_one,
                T.simd.vadd(v_one, T.simd.vexp(neg_x0, mask=m32), mask=m32),
                mask=m32,
                precision="ftz_true",
            )
            out_f32 = T.simd.vmul(T.simd.vmul(x0, sig, mask=m32), x1, mask=m32)
            if tl_dtype == "float32":
                T.simd.vsts(y_ub[slot, off], out_f32, mask=m32, dist="NORM_B32")
            else:
                out16 = T.simd.vcvt(out_f32, tl_dtype, mask=m16, part=0)
                T.simd.vsts(y_ub[slot, off], out16, mask=m32, dist="PK_B32", extent=64)

        if n_rem > 0:
            off = n_chunks * 64
            m32_t = T.simd.update_mask(n_rem, width=32)
            m16_t = T.simd.update_mask(n_rem * 2, width=16)
            if tl_dtype == "float32":
                x0_t = T.simd.vld(x0_ub[slot, off], dist="NORM_B32")
                x1_t = T.simd.vld(x1_ub[slot, off], dist="NORM_B32")
            else:
                x0_t = T.simd.vcvt(T.simd.vld(x0_ub[slot, off], dist="US_B16"), "float32", mask=m16_t, part=0)
                x1_t = T.simd.vcvt(T.simd.vld(x1_ub[slot, off], dist="US_B16"), "float32", mask=m16_t, part=0)
            neg_x0_t = T.simd.vmuls(x0_t, T.float32(-1.0), mask=m32_t)
            sig_t = T.simd.vdiv(
                v_one,
                T.simd.vadd(v_one, T.simd.vexp(neg_x0_t, mask=m32_t), mask=m32_t),
                mask=m32_t,
                precision="ftz_true",
            )
            out_t = T.simd.vmul(T.simd.vmul(x0_t, sig_t, mask=m32_t), x1_t, mask=m32_t)
            if tl_dtype == "float32":
                T.simd.vsts(y_ub[slot, off], out_t, mask=m32_t, dist="NORM_B32")
            else:
                out16_t = T.simd.vcvt(out_t, tl_dtype, mask=m16_t, part=0)
                T.simd.vsts(y_ub[slot, off], out16_t, mask=m32_t, dist="PK_B32", extent=64)

    if inner_dim > 1:
        w_blk = int(d_tile) if d_tile else (2048 if tl_dtype == "float32" else 4096)
        outer_dyn = T.dynamic("outer_dim")
        w_dyn = T.dynamic("w_total")

        with _ascend_target_ctx():

            @T.prim_func
            def main(
                X: T.Tensor((outer_dyn, 2, w_dyn), tl_dtype),
                Y: T.Tensor((outer_dyn, w_dyn), tl_dtype),
            ):
                with T.Kernel(num_cores) as cid:
                    x0_ub = T.alloc_shared((1, w_blk), tl_dtype)
                    x1_ub = T.alloc_shared((1, w_blk), tl_dtype)
                    y_ub = T.alloc_shared((1, w_blk), tl_dtype)
                    T.annotate_buffer_versions({x0_ub: stages, x1_ub: stages, y_ub: stages})
                    if schedule_mode == "tuned":
                        T.annotate_unlimit_memory("shared")

                    w_tiles = (w_dyn + w_blk - 1) // w_blk
                    total_tasks = outer_dyn * w_tiles
                    base_t = total_tasks // num_cores
                    rem_t = total_tasks % num_cores
                    t_start = cid * base_t + T.min(cid, rem_t)
                    t_count = base_t + T.if_then_else(cid < rem_t, 1, 0)

                    for lt in T.Pipelined(t_count, num_stages=stages, annotations=pipe_ann):
                        tid = t_start + lt
                        o_idx = tid // w_tiles
                        wt = tid % w_tiles
                        w0 = wt * w_blk
                        w_cur = T.min(w_blk, w_dyn - w0)
                        n_chunks = w_cur // 64
                        n_rem = w_cur % 64
                        T.copy(X[o_idx, 0, w0 : w0 + w_cur], x0_ub[0, :w_cur])
                        T.copy(X[o_idx, 1, w0 : w0 + w_cur], x1_ub[0, :w_cur])
                        with T.SimdVF():
                            _swiglu_vec_chunks(x0_ub, x1_ub, y_ub, 0, n_chunks, n_rem)
                        T.copy(y_ub[0, :w_cur], Y[o_idx, w0 : w0 + w_cur])

        return main

    m_rows = T.dynamic("m_rows")
    rows = max(1, int(tile_m))

    if half_dim == 1:
        rows_blk = max(64, ((rows + 63) // 64) * 64)
        with _ascend_target_ctx():

            @T.prim_func
            def main(
                X: T.Tensor((m_rows * 2,), tl_dtype),
                Y: T.Tensor((m_rows,), tl_dtype),
            ):
                with T.Kernel(num_cores) as cid:
                    x_ub = T.alloc_shared((rows_blk * 2,), tl_dtype)
                    y_ub = T.alloc_shared((rows_blk,), tl_dtype)
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
                            v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
                            n_chunks = (cur + 63) // 64
                            for c in T.serial(n_chunks):
                                off = c * 64
                                if tl_dtype == "float32":
                                    x0, x1 = T.simd.vld2(x_ub[off * 2], dist="DINTLV_B32")
                                else:
                                    v = T.simd.vld(x_ub[off * 2], dist="NORM_B16")
                                    x0 = T.simd.vcvt(v, "float32", mask=m16, part=0)
                                    x1 = T.simd.vcvt(v, "float32", mask=m16, part=1)
                                neg_x0 = T.simd.vmuls(x0, T.float32(-1.0), mask=m32)
                                sig = T.simd.vdiv(
                                    v_one,
                                    T.simd.vadd(v_one, T.simd.vexp(neg_x0, mask=m32), mask=m32),
                                    mask=m32,
                                    precision="ftz_true",
                                )
                                out_f32 = T.simd.vmul(T.simd.vmul(x0, sig, mask=m32), x1, mask=m32)
                                if tl_dtype == "float32":
                                    T.simd.vsts(y_ub[off], out_f32, mask=m32, dist="NORM_B32")
                                else:
                                    out16 = T.simd.vcvt(out_f32, tl_dtype, mask=m16, part=0)
                                    T.simd.vsts(y_ub[off], out16, mask=m32, dist="PK_B32", extent=64)
                        T.copy(y_ub[:cur], Y[r0 : r0 + cur])

        return main

    if regime == "narrow_simt_packed" or half_dim <= 32 or (
        tl_dtype == "float32" and (half_dim % 16) != 0 and half_dim <= 64
    ):
        rows_blk = max(8, rows)
        full_dim = 2 * half_dim
        simt_threads = 512
        iters = (rows_blk * half_dim + simt_threads - 1) // simt_threads
        with _ascend_target_ctx():

            @T.prim_func
            def main(
                X: T.Tensor((m_rows * full_dim,), tl_dtype),
                Y: T.Tensor((m_rows * half_dim,), tl_dtype),
            ):
                with T.Kernel(num_cores) as cid:
                    x_ub = T.alloc_shared((rows_blk * full_dim,), tl_dtype)
                    y_ub = T.alloc_shared((rows_blk * half_dim,), tl_dtype)
                    T.annotate_buffer_versions({x_ub: stages, y_ub: stages})

                    base_rows = m_rows // num_cores
                    rem_rows = m_rows % num_cores
                    m_start = cid * base_rows + T.min(cid, rem_rows)
                    m_count = base_rows + T.if_then_else(cid < rem_rows, 1, 0)
                    num_blocks = (m_count + rows_blk - 1) // rows_blk

                    for mb in T.Pipelined(num_blocks, num_stages=stages):
                        r0 = m_start + mb * rows_blk
                        cur = T.min(rows_blk, m_count - mb * rows_blk)
                        T.copy(X[r0 * full_dim : (r0 + cur) * full_dim], x_ub[: cur * full_dim])
                        with T.SimtVF(threads=simt_threads):
                            for tid in T.Parallel(simt_threads):
                                for idx in T.serial(iters):
                                    el = idx * simt_threads + tid
                                    r = el // half_dim
                                    d = el % half_dim
                                    if r < cur:
                                        x0 = T.cast(x_ub[r * full_dim + d], "float32")
                                        x1 = T.cast(x_ub[r * full_dim + half_dim + d], "float32")
                                        y_ub[el] = T.cast(
                                            (x0 / (T.float32(1.0) + T.exp(-x0))) * x1,
                                            tl_dtype,
                                        )
                        T.copy(y_ub[: cur * half_dim], Y[r0 * half_dim : (r0 + cur) * half_dim])

        return main

    half_alloc = ((half_dim + 63) // 64) * 64
    blk_d = int(d_tile) if d_tile else half_alloc
    if blk_d >= half_alloc:
        blk_d = half_alloc
        d_blocks = 1
    else:
        d_blocks = (half_dim + blk_d - 1) // blk_d
    full_dim = 2 * half_dim
    single_x_dma = (d_blocks == 1) and ((half_dim % 64) == 0)

    with _ascend_target_ctx():

        @T.prim_func
        def main(
            X: T.Tensor((m_rows, full_dim), tl_dtype),
            Y: T.Tensor((m_rows, half_dim), tl_dtype),
        ):
            with T.Kernel(num_cores) as cid:
                if single_x_dma:
                    x_ub = T.alloc_shared((rows, full_dim), tl_dtype)
                    y_ub = T.alloc_shared((rows, blk_d), tl_dtype)
                    T.annotate_buffer_versions({x_ub: stages, y_ub: stages})
                else:
                    x0_ub = T.alloc_shared((rows, blk_d), tl_dtype)
                    x1_ub = T.alloc_shared((rows, blk_d), tl_dtype)
                    y_ub = T.alloc_shared((rows, blk_d), tl_dtype)
                    T.annotate_buffer_versions({x0_ub: stages, x1_ub: stages, y_ub: stages})
                if schedule_mode == "tuned":
                    T.annotate_unlimit_memory("shared")

                base_rows = m_rows // num_cores
                rem_rows = m_rows % num_cores
                m_start = cid * base_rows + T.min(cid, rem_rows)
                m_count = base_rows + T.if_then_else(cid < rem_rows, 1, 0)

                if single_x_dma:
                    full_chunks = half_dim // 64
                    num_blocks = (m_count + rows - 1) // rows
                    for mb in T.Pipelined(num_blocks, num_stages=stages, annotations=pipe_ann):
                        m0 = m_start + mb * rows
                        m_cur = T.min(rows, m_count - mb * rows)
                        T.copy(X[m0 : m0 + m_cur, :full_dim], x_ub[:m_cur, :full_dim])
                        with T.SimdVF():
                            m32 = T.simd.pset(32, "PAT_ALL")
                            m16 = T.simd.pset(16, "PAT_ALL")
                            v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
                            for rr in T.serial(m_cur):
                                for c in T.serial(full_chunks):
                                    off = c * 64
                                    if tl_dtype == "float32":
                                        x0 = T.simd.vld(x_ub[rr, off], dist="NORM_B32")
                                        x1 = T.simd.vld(x_ub[rr, half_dim + off], dist="NORM_B32")
                                    else:
                                        x0 = T.simd.vcvt(T.simd.vld(x_ub[rr, off], dist="US_B16"), "float32", mask=m16, part=0)
                                        x1 = T.simd.vcvt(T.simd.vld(x_ub[rr, half_dim + off], dist="US_B16"), "float32", mask=m16, part=0)
                                    neg_x0 = T.simd.vmuls(x0, T.float32(-1.0), mask=m32)
                                    sig = T.simd.vdiv(
                                        v_one,
                                        T.simd.vadd(v_one, T.simd.vexp(neg_x0, mask=m32), mask=m32),
                                        mask=m32,
                                        precision="ftz_true",
                                    )
                                    out_f32 = T.simd.vmul(T.simd.vmul(x0, sig, mask=m32), x1, mask=m32)
                                    if tl_dtype == "float32":
                                        T.simd.vsts(y_ub[rr, off], out_f32, mask=m32, dist="NORM_B32")
                                    else:
                                        out16 = T.simd.vcvt(out_f32, tl_dtype, mask=m16, part=0)
                                        T.simd.vsts(y_ub[rr, off], out16, mask=m32, dist="PK_B32", extent=64)
                        T.copy(y_ub[:m_cur, :half_dim], Y[m0 : m0 + m_cur, :half_dim])
                elif d_blocks == 1:
                    full_chunks = half_dim // 64
                    rem = half_dim % 64
                    num_blocks = (m_count + rows - 1) // rows
                    for mb in T.Pipelined(num_blocks, num_stages=stages, annotations=pipe_ann):
                        m0 = m_start + mb * rows
                        m_cur = T.min(rows, m_count - mb * rows)
                        T.copy(X[m0 : m0 + m_cur, :half_dim], x0_ub[:m_cur, :half_dim])
                        T.copy(X[m0 : m0 + m_cur, half_dim:full_dim], x1_ub[:m_cur, :half_dim])
                        with T.SimdVF():
                            for rr in T.serial(m_cur):
                                _swiglu_vec_chunks(x0_ub, x1_ub, y_ub, rr, full_chunks, rem)
                        T.copy(y_ub[:m_cur, :half_dim], Y[m0 : m0 + m_cur, :half_dim])
                else:
                    total_tiles = m_count * d_blocks
                    for td in T.Pipelined(total_tiles, num_stages=stages, annotations=pipe_ann):
                        m = m_start + (td // d_blocks)
                        db = td % d_blocks
                        d_off = db * blk_d
                        d_cur = T.min(blk_d, half_dim - d_off)
                        n_chunks = d_cur // 64
                        n_rem = d_cur % 64
                        T.copy(X[m, d_off : d_off + d_cur], x0_ub[0, :d_cur])
                        T.copy(X[m, half_dim + d_off : half_dim + d_off + d_cur], x1_ub[0, :d_cur])
                        with T.SimdVF():
                            _swiglu_vec_chunks(x0_ub, x1_ub, y_ub, 0, n_chunks, n_rem)
                        T.copy(y_ub[0, :d_cur], Y[m, d_off : d_off + d_cur])

    return main


def _get_kernel(half_dim: int, tl_dtype: str, m_rows: int, inner_dim: int, schedule_mode: str):
    cfg = _select_swi_glu_config(
        half_dim, tl_dtype, m_rows=m_rows, inner_dim=inner_dim, schedule_mode=schedule_mode
    )
    key = (
        int(half_dim) if inner_dim == 1 else -1,
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
            prog = _build_swi_glu_prim_func(
                half_dim=half_dim,
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


def swi_glu(input: torch.Tensor, dim: int = -1, *, schedule_mode: str = "tuned") -> torch.Tensor:
    """Compute SwiGLU activation matching cannbot-ops level1/swi_glu proto.yaml."""
    if input.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise ValueError(f"unsupported dtype {input.dtype}")
    ndim = input.ndim
    norm_dim = dim if dim >= 0 else ndim + dim
    if norm_dim < 0 or norm_dim >= ndim:
        raise ValueError(f"dim {dim} out of range for tensor with {ndim} dims")
    if input.shape[norm_dim] % 2 != 0:
        raise ValueError(f"split dim size must be even, got {input.shape[norm_dim]}")

    if input.device.type != "npu":
        out_dtype = input.dtype
        xf = input.to(torch.float32)
        x0, x1 = xf.chunk(2, dim=norm_dim)
        return (F.silu(x0) * x1).to(out_dtype)

    tl_dtype = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }[input.dtype]

    x_c = input.contiguous()
    shape = tuple(x_c.shape)
    half_dim = shape[norm_dim] // 2
    out_shape = shape[:norm_dim] + (half_dim,) + shape[norm_dim + 1 :]

    outer = 1
    for s in shape[:norm_dim]:
        outer *= s
    inner = 1
    for s in shape[norm_dim + 1 :]:
        inner *= s

    if inner > 1:
        w_total = half_dim * inner
        x_3d = x_c.view(outer, 2, w_total)
        compiled = _get_kernel(half_dim, tl_dtype, outer, inner, schedule_mode)
        y_2d = compiled(x_3d)
        return y_2d.view(out_shape)

    m_rows = max(1, outer)
    cfg = _select_swi_glu_config(
        half_dim, tl_dtype, m_rows=m_rows, inner_dim=1, schedule_mode=schedule_mode
    )
    compiled = _get_kernel(half_dim, tl_dtype, m_rows, 1, schedule_mode)
    if cfg["regime"] in ("ultra_narrow_packed", "narrow_simt_packed"):
        y_1d = compiled(x_c.view(-1))
        return y_1d.view(out_shape)

    x_2d = x_c.view(m_rows, 2 * half_dim)
    y_2d = compiled(x_2d)
    return y_2d.view(out_shape)


kernel = swi_glu

__all__ = ["_build_swi_glu_prim_func", "_select_swi_glu_config", "kernel", "swi_glu"]
