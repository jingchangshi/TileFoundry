"""TileLang Ascend950PR Pure RegBase SimdVF / SimtVF kernel implementation for RmsNorm."""

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


def _select_rms_norm_config(
    d_dim: int,
    tl_dtype: str = "bfloat16",
    *,
    m_rows: int = 1024,
    schedule_mode: str = "tuned",
) -> dict[str, object]:
    """Select row packing and UB staging parameters for RmsNorm."""
    if d_dim == 2:
        tile_m = 2048 if (tl_dtype != "float32" and m_rows >= 56 * 2048) else (
            1024 if m_rows >= 56 * 1024 else max(128, ((max(1, m_rows // 112) + 127) // 128) * 128)
        )
        return {
            "regime": "ultra_narrow_pair_vec",
            "tile_m": tile_m,
            "num_stages": 2 if (schedule_mode == "tuned" and m_rows >= 56 * 2) else 1,
            "d_dim": int(d_dim),
            "tl_dtype": tl_dtype,
            "pass_configs": {},
            "compile_flags": list(PURE_SIMDVF_UBUF_FLAGS) if schedule_mode == "tuned" else [],
        }

    if d_dim <= 16:
        tile_m = 256 if m_rows >= 56 * 256 else max(8, min(128, max(8, m_rows // 112)))
        return {
            "regime": "narrow_simt_packed",
            "tile_m": tile_m,
            "num_stages": 2 if (schedule_mode == "tuned" and m_rows >= 56 * 2) else 1,
            "d_dim": int(d_dim),
            "tl_dtype": tl_dtype,
            "pass_configs": {"tl.disable_data_race_check": True},
            "compile_flags": [],
        }

    d_alloc = ((max(1, int(d_dim)) + 63) // 64) * 64
    elem_bytes = 4 if tl_dtype == "float32" else 2
    ub_budget = 256 * 1024 - 4096 if schedule_mode == "tuned" else 192 * 1024
    stages = 2 if (schedule_mode == "tuned" and m_rows >= 56 * 2) else 1
    gamma_bytes = d_alloc * 4 if tl_dtype == "float32" else (d_alloc * 4 + d_alloc * elem_bytes)
    row_bytes = 2 * stages * d_alloc * elem_bytes
    max_rows = max(1, min(64, (ub_budget - gamma_bytes) // max(1, row_bytes)))
    if (d_dim % 64) != 0:
        max_rows = min(max_rows, 32)
    if max_rows > 1 and m_rows >= 112 and m_rows < 56 * max_rows * 2:
        max_rows = max(1, min(max_rows, max(1, m_rows // 112)))
    elif max_rows > 1 and m_rows < 56 * max_rows:
        max_rows = max(1, min(max_rows, max(1, m_rows // 56)))
    regime = "aligned_packed" if (d_dim % 64) == 0 else "unaligned_predicated"
    return {
        "regime": regime,
        "tile_m": max_rows,
        "num_stages": stages,
        "d_dim": int(d_dim),
        "tl_dtype": tl_dtype,
        "pass_configs": {},
        "compile_flags": list(PURE_SIMDVF_UBUF_FLAGS) if schedule_mode == "tuned" else [],
    }


def _build_rms_norm_prim_func(
    d_dim: int = 1024,
    tl_dtype: str = "bfloat16",
    tile_m: int = 4,
    num_stages: int = 2,
    schedule_mode: str = "tuned",
    **_extra_kwargs,
):
    """Build Ascend 950 Pure RegBase SimdVF / SimtVF @T.prim_func for RmsNorm."""
    if d_dim <= 0:
        raise ValueError(f"d_dim must be positive, got {d_dim}")
    if tl_dtype not in ("float16", "bfloat16", "float32"):
        raise ValueError(f"unsupported dtype {tl_dtype!r}")

    num_cores = 56
    rows = max(1, int(tile_m))
    stages = max(1, min(2, int(num_stages)))
    inv_d = 1.0 / float(d_dim)
    pipe_ann = {"enable_offset": True} if schedule_mode == "tuned" else None
    m_rows = T.dynamic("m_rows")

    if d_dim == 2:
        rows_blk = max(128, ((rows + 127) // 128) * 128)
        with _ascend_target_ctx():

            @T.prim_func
            def main(
                X: T.Tensor((m_rows * 2,), tl_dtype),
                Gamma: T.Tensor((2,), tl_dtype),
                Y: T.Tensor((m_rows * 2,), tl_dtype),
                epsilon: T.float32,
            ):
                with T.Kernel(num_cores) as cid:
                    g_ub = T.alloc_shared((64,), tl_dtype)
                    g_f32_ub = T.alloc_shared((64,), "float32")
                    x_ub = T.alloc_shared((rows_blk * 2,), tl_dtype)
                    y_ub = T.alloc_shared((rows_blk * 2,), tl_dtype)
                    T.annotate_buffer_versions({g_ub: 1, g_f32_ub: 1, x_ub: stages, y_ub: stages})
                    if schedule_mode == "tuned":
                        T.annotate_unlimit_memory("shared")

                    base_rows = m_rows // num_cores
                    rem_rows = m_rows % num_cores
                    m_start = cid * base_rows + T.min(cid, rem_rows)
                    m_count = base_rows + T.if_then_else(cid < rem_rows, 1, 0)

                    T.copy(Gamma[:2], g_ub[:2])
                    with T.SimdVF():
                        m32 = T.simd.pset(32, "PAT_ALL")
                        m16 = T.simd.pset(16, "PAT_ALL")
                        if tl_dtype == "float32":
                            gf = T.simd.vld(g_ub[0], dist="NORM_B32")
                        else:
                            gf = T.simd.vcvt(T.simd.vld(g_ub[0], dist="US_B16"), "float32", mask=m16, part=0)
                        T.simd.vsts(g_f32_ub[0], gf, mask=m32, dist="NORM_B32")

                    num_blocks = (m_count + rows_blk - 1) // rows_blk
                    for mb in T.Pipelined(num_blocks, num_stages=stages, annotations=pipe_ann):
                        r0 = m_start + mb * rows_blk
                        cur = T.min(rows_blk, m_count - mb * rows_blk)
                        T.copy(X[r0 * 2 : (r0 + cur) * 2], x_ub[: cur * 2])
                        with T.SimdVF():
                            m32 = T.simd.pset(32, "PAT_ALL")
                            m16 = T.simd.pset(16, "PAT_ALL")
                            v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
                            g0 = T.simd.vld(g_f32_ub[0], dist="BRC_B32")
                            g1 = T.simd.vld(g_f32_ub[1], dist="BRC_B32")
                            if tl_dtype == "float32":
                                n_chunks = (cur + 63) // 64
                                for c in T.serial(n_chunks):
                                    off = c * 128
                                    x0, x1 = T.simd.vld2(x_ub[off], dist="DINTLV_B32")
                                    sq = T.simd.vadd(T.simd.vmul(x0, x0, mask=m32), T.simd.vmul(x1, x1, mask=m32), mask=m32)
                                    rstd = T.simd.vdiv(
                                        v_one,
                                        T.simd.vsqrt(T.simd.vadds(T.simd.vmuls(sq, T.float32(0.5), mask=m32), epsilon, mask=m32), mask=m32),
                                        mask=m32,
                                        precision="ftz_true",
                                    )
                                    y0 = T.simd.vmul(T.simd.vmul(x0, rstd, mask=m32), g0, mask=m32)
                                    y1 = T.simd.vmul(T.simd.vmul(x1, rstd, mask=m32), g1, mask=m32)
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
                                    sq_e = T.simd.vadd(T.simd.vmul(x0_e, x0_e, mask=m32), T.simd.vmul(x1_e, x1_e, mask=m32), mask=m32)
                                    sq_o = T.simd.vadd(T.simd.vmul(x0_o, x0_o, mask=m32), T.simd.vmul(x1_o, x1_o, mask=m32), mask=m32)
                                    rstd_e = T.simd.vdiv(
                                        v_one,
                                        T.simd.vsqrt(T.simd.vadds(T.simd.vmuls(sq_e, T.float32(0.5), mask=m32), epsilon, mask=m32), mask=m32),
                                        mask=m32,
                                        precision="ftz_true",
                                    )
                                    rstd_o = T.simd.vdiv(
                                        v_one,
                                        T.simd.vsqrt(T.simd.vadds(T.simd.vmuls(sq_o, T.float32(0.5), mask=m32), epsilon, mask=m32), mask=m32),
                                        mask=m32,
                                        precision="ftz_true",
                                    )
                                    y0_e = T.simd.vcvt(T.simd.vmul(T.simd.vmul(x0_e, rstd_e, mask=m32), g0, mask=m32), tl_dtype, mask=m16, part=0)
                                    y0_o = T.simd.vcvt(T.simd.vmul(T.simd.vmul(x0_o, rstd_o, mask=m32), g0, mask=m32), tl_dtype, mask=m16, part=1)
                                    y1_e = T.simd.vcvt(T.simd.vmul(T.simd.vmul(x1_e, rstd_e, mask=m32), g1, mask=m32), tl_dtype, mask=m16, part=0)
                                    y1_o = T.simd.vcvt(T.simd.vmul(T.simd.vmul(x1_o, rstd_o, mask=m32), g1, mask=m32), tl_dtype, mask=m16, part=1)
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
                Gamma: T.Tensor((d_dim,), tl_dtype),
                Y: T.Tensor((m_rows * d_dim,), tl_dtype),
                epsilon: T.float32,
            ):
                with T.Kernel(num_cores) as cid:
                    g_ub = T.alloc_shared((d_dim,), tl_dtype)
                    x_ub = T.alloc_shared((rows_blk * d_dim,), tl_dtype)
                    y_ub = T.alloc_shared((rows_blk * d_dim,), tl_dtype)
                    T.annotate_buffer_versions({g_ub: 1, x_ub: stages, y_ub: stages})

                    base_rows = m_rows // num_cores
                    rem_rows = m_rows % num_cores
                    m_start = cid * base_rows + T.min(cid, rem_rows)
                    m_count = base_rows + T.if_then_else(cid < rem_rows, 1, 0)
                    T.copy(Gamma[:d_dim], g_ub[:d_dim])
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
                                        sq_b = T.alloc_local((1,), "float32")
                                        sq_b[0] = T.float32(0.0)
                                        for dd in T.serial(d_dim):
                                            xv = T.cast(x_ub[r * d_dim + dd], "float32")
                                            sq_b[0] = sq_b[0] + xv * xv
                                        rstd = T.float32(1.0) / T.sqrt(sq_b[0] * T.float32(inv_d) + epsilon)
                                        for dd in T.serial(d_dim):
                                            xv = T.cast(x_ub[r * d_dim + dd], "float32")
                                            gv = T.cast(g_ub[dd], "float32")
                                            y_ub[r * d_dim + dd] = T.cast(xv * rstd * gv, tl_dtype)
                        T.copy(y_ub[: cur * d_dim], Y[r0 * d_dim : (r0 + cur) * d_dim])

        return main

    full_chunks = d_dim // 64
    rem = d_dim % 64
    d_alloc = ((d_dim + 63) // 64) * 64
    use_reg_cache = d_dim <= 768

    @T.macro
    def _rms_norm_row(x_ub, g_f32_ub, y_ub, eps_val, slot):
        m32 = T.simd.pset(32, "PAT_ALL")
        m16 = T.simd.pset(16, "PAT_ALL")
        v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
        v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
        sq_acc = T.simd.alloc_local((1,), "float32")
        sq_acc[0] = v_zero

        if use_reg_cache:
            total_c = full_chunks + (1 if rem > 0 else 0)
            x_reg = T.simd.alloc_local((total_c,), "float32")
            for c in T.unroll(full_chunks):
                off = c * 64
                if tl_dtype == "float32":
                    xf = T.simd.vld(x_ub[slot, off], dist="NORM_B32")
                else:
                    xf = T.simd.vcvt(T.simd.vld(x_ub[slot, off], dist="US_B16"), "float32", mask=m16, part=0)
                x_reg[c] = xf
                T.simd.vmula(sq_acc[0], xf, xf, mask=m32)
            if rem > 0:
                off = full_chunks * 64
                m32_t = T.simd.update_mask(rem, width=32)
                if tl_dtype == "float32":
                    xf_t = T.simd.vld(x_ub[slot, off], dist="NORM_B32")
                else:
                    m16_t = T.simd.update_mask(rem * 2, width=16)
                    xf_t = T.simd.vcvt(T.simd.vld(x_ub[slot, off], dist="US_B16"), "float32", mask=m16_t, part=0)
                x_reg[full_chunks] = xf_t
                T.simd.vmula(sq_acc[0], xf_t, xf_t, mask=m32_t, mode="MODE_MERGING")

            sum_sq = T.simd.vdupv(T.simd.vcadd(sq_acc[0], mask=m32), mask=m32, pos="POS_LOWEST")
            mean_eps = T.simd.vadds(T.simd.vmuls(sum_sq, T.float32(inv_d), mask=m32), eps_val, mask=m32)
            rstd = T.simd.vdiv(v_one, T.simd.vsqrt(mean_eps, mask=m32), mask=m32, precision="ftz_true")

            for c in T.unroll(full_chunks):
                off = c * 64
                gf = T.simd.vld(g_f32_ub[off], dist="NORM_B32")
                yf = T.simd.vmul(T.simd.vmul(x_reg[c], rstd, mask=m32), gf, mask=m32)
                if tl_dtype == "float32":
                    T.simd.vsts(y_ub[slot, off], yf, mask=m32, dist="NORM_B32")
                else:
                    y16 = T.simd.vcvt(yf, tl_dtype, mask=m16, part=0)
                    T.simd.vsts(y_ub[slot, off], y16, mask=m32, dist="PK_B32", extent=64)
            if rem > 0:
                off = full_chunks * 64
                m32_t = T.simd.update_mask(rem, width=32)
                gf_t = T.simd.vld(g_f32_ub[off], dist="NORM_B32")
                yf_t = T.simd.vmul(T.simd.vmul(x_reg[full_chunks], rstd, mask=m32_t), gf_t, mask=m32_t)
                if tl_dtype == "float32":
                    T.simd.vsts(y_ub[slot, off], yf_t, mask=m32_t, dist="NORM_B32")
                else:
                    m16_t = T.simd.update_mask(rem * 2, width=16)
                    y16_t = T.simd.vcvt(yf_t, tl_dtype, mask=m16_t, part=0)
                    T.simd.vsts(y_ub[slot, off], y16_t, mask=m32_t, dist="PK_B32", extent=64)
        else:
            for c in T.serial(full_chunks):
                off = c * 64
                if tl_dtype == "float32":
                    xf = T.simd.vld(x_ub[slot, off], dist="NORM_B32")
                else:
                    xf = T.simd.vcvt(T.simd.vld(x_ub[slot, off], dist="US_B16"), "float32", mask=m16, part=0)
                T.simd.vmula(sq_acc[0], xf, xf, mask=m32)
            if rem > 0:
                off = full_chunks * 64
                m32_t = T.simd.update_mask(rem, width=32)
                if tl_dtype == "float32":
                    xf_t = T.simd.vld(x_ub[slot, off], dist="NORM_B32")
                else:
                    m16_t = T.simd.update_mask(rem * 2, width=16)
                    xf_t = T.simd.vcvt(T.simd.vld(x_ub[slot, off], dist="US_B16"), "float32", mask=m16_t, part=0)
                T.simd.vmula(sq_acc[0], xf_t, xf_t, mask=m32_t, mode="MODE_MERGING")

            sum_sq = T.simd.vdupv(T.simd.vcadd(sq_acc[0], mask=m32), mask=m32, pos="POS_LOWEST")
            mean_eps = T.simd.vadds(T.simd.vmuls(sum_sq, T.float32(inv_d), mask=m32), eps_val, mask=m32)
            rstd = T.simd.vdiv(v_one, T.simd.vsqrt(mean_eps, mask=m32), mask=m32, precision="ftz_true")

            for c in T.serial(full_chunks):
                off = c * 64
                if tl_dtype == "float32":
                    xf = T.simd.vld(x_ub[slot, off], dist="NORM_B32")
                else:
                    xf = T.simd.vcvt(T.simd.vld(x_ub[slot, off], dist="US_B16"), "float32", mask=m16, part=0)
                gf = T.simd.vld(g_f32_ub[off], dist="NORM_B32")
                yf = T.simd.vmul(T.simd.vmul(xf, rstd, mask=m32), gf, mask=m32)
                if tl_dtype == "float32":
                    T.simd.vsts(y_ub[slot, off], yf, mask=m32, dist="NORM_B32")
                else:
                    y16 = T.simd.vcvt(yf, tl_dtype, mask=m16, part=0)
                    T.simd.vsts(y_ub[slot, off], y16, mask=m32, dist="PK_B32", extent=64)

            if rem > 0:
                off = full_chunks * 64
                m32_t = T.simd.update_mask(rem, width=32)
                if tl_dtype == "float32":
                    xf_t = T.simd.vld(x_ub[slot, off], dist="NORM_B32")
                else:
                    m16_t = T.simd.update_mask(rem * 2, width=16)
                    xf_t = T.simd.vcvt(T.simd.vld(x_ub[slot, off], dist="US_B16"), "float32", mask=m16_t, part=0)
                gf_t = T.simd.vld(g_f32_ub[off], dist="NORM_B32")
                yf_t = T.simd.vmul(T.simd.vmul(xf_t, rstd, mask=m32_t), gf_t, mask=m32_t)
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
            Gamma: T.Tensor((d_dim,), tl_dtype),
            Y: T.Tensor((m_rows, d_dim), tl_dtype),
            epsilon: T.float32,
        ):
            with T.Kernel(num_cores) as cid:
                g_f32_ub = T.alloc_shared((d_alloc,), "float32")
                x_ub = T.alloc_shared((rows, d_alloc), tl_dtype)
                y_ub = T.alloc_shared((rows, d_alloc), tl_dtype)
                if tl_dtype == "float32":
                    T.annotate_buffer_versions({g_f32_ub: 1, x_ub: stages, y_ub: stages})
                else:
                    g_ub = T.alloc_shared((d_alloc,), tl_dtype)
                    T.annotate_buffer_versions(
                        {g_ub: 1, g_f32_ub: 1, x_ub: stages, y_ub: stages}
                    )
                if schedule_mode == "tuned":
                    T.annotate_unlimit_memory("shared")

                base_rows = m_rows // num_cores
                rem_rows = m_rows % num_cores
                m_start = cid * base_rows + T.min(cid, rem_rows)
                m_count = base_rows + T.if_then_else(cid < rem_rows, 1, 0)

                if tl_dtype == "float32":
                    T.copy(Gamma[:d_dim], g_f32_ub[:d_dim])
                else:
                    T.copy(Gamma[:d_dim], g_ub[:d_dim])
                    with T.SimdVF():
                        m32 = T.simd.pset(32, "PAT_ALL")
                        m16 = T.simd.pset(16, "PAT_ALL")
                        for c in T.serial(full_chunks):
                            off = c * 64
                            gf = T.simd.vcvt(
                                T.simd.vld(g_ub[off], dist="US_B16"),
                                "float32",
                                mask=m16,
                                part=0,
                            )
                            T.simd.vsts(g_f32_ub[off], gf, mask=m32, dist="NORM_B32")
                        if rem > 0:
                            off = full_chunks * 64
                            m32_t = T.simd.update_mask(rem, width=32)
                            m16_t = T.simd.update_mask(rem * 2, width=16)
                            gf_t = T.simd.vcvt(
                                T.simd.vld(g_ub[off], dist="US_B16"),
                                "float32",
                                mask=m16_t,
                                part=0,
                            )
                            T.simd.vsts(g_f32_ub[off], gf_t, mask=m32_t, dist="NORM_B32")

                num_blocks = (m_count + rows - 1) // rows
                for mb in T.Pipelined(num_blocks, num_stages=stages, annotations=pipe_ann):
                    m0 = m_start + mb * rows
                    m_cur = T.min(rows, m_count - mb * rows)
                    T.copy(X[m0 : m0 + m_cur, :d_dim], x_ub[:m_cur, :d_dim])
                    with T.SimdVF():
                        for rr in T.serial(m_cur):
                            _rms_norm_row(x_ub, g_f32_ub, y_ub, epsilon, rr)
                    T.copy(y_ub[:m_cur, :d_dim], Y[m0 : m0 + m_cur, :d_dim])

    return main


def _get_kernel(d_dim: int, tl_dtype: str, m_rows: int, schedule_mode: str):
    cfg = _select_rms_norm_config(d_dim, tl_dtype, m_rows=m_rows, schedule_mode=schedule_mode)
    key = (d_dim, tl_dtype, int(cfg["tile_m"]), int(cfg["num_stages"]), str(cfg["regime"]), schedule_mode)
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        with _ascend_target_ctx():
            prog = _build_rms_norm_prim_func(
                d_dim=d_dim,
                tl_dtype=tl_dtype,
                tile_m=int(cfg["tile_m"]),
                num_stages=int(cfg["num_stages"]),
                schedule_mode=schedule_mode,
            )
        compiled = tilelang.compile(
            prog,
            target="ascend",
            out_idx=[2],
            pass_configs=cfg.get("pass_configs") or None,
            compile_flags=cfg.get("compile_flags") or None,
        )
        _KERNEL_CACHE[key] = compiled
    return compiled


def rms_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
    *,
    schedule_mode: str = "tuned",
) -> torch.Tensor:
    """Compute RMSNorm matching cannbot-ops level2/rms_norm proto.yaml."""
    if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise ValueError(f"unsupported dtype {x.dtype}")

    if x.device.type != "npu":
        return F.rms_norm(x, normalized_shape=gamma.shape, weight=gamma, eps=float(epsilon))

    tl_dtype = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }[x.dtype]
    x_contig = x.contiguous()
    d_dim = x_contig.shape[-1]
    m_rows = max(1, x_contig.numel() // d_dim)
    compiled = _get_kernel(d_dim, tl_dtype, m_rows, schedule_mode)
    if d_dim <= 16:
        y_1d = compiled(x_contig.view(-1), gamma.contiguous().view(d_dim), float(epsilon))
        return y_1d.view(x_contig.shape)
    y_2d = compiled(x_contig.view(m_rows, d_dim), gamma.contiguous().view(d_dim), float(epsilon))
    return y_2d.view(x_contig.shape)


kernel = rms_norm

__all__ = ["_build_rms_norm_prim_func", "_select_rms_norm_config", "kernel", "rms_norm"]
