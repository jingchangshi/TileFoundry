"""TileLang Ascend950PR Pure RegBase SimdVF kernel for AddRmsNormDynamicQuant."""

import contextlib
import math

import tilelang
import tilelang.ascend.language as T
import torch

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


def _select_add_rms_norm_dynamic_quant_config(
    d_dim: int,
    tl_dtype: str = "bfloat16",
    *,
    m_rows: int = 1024,
    schedule_mode: str = "tuned",
    tile_m_override: int = 0,
    num_stages_override: int = 0,
) -> dict[str, object]:
    """Select UB-resident single-pass row packing for Add+RMSNorm+DynamicQuant."""
    d_alloc = ((max(1, int(d_dim)) + 63) // 64) * 64
    raw_m = max(1, int(m_rows))
    eff_m_rows = raw_m if raw_m < 128 else (1 << (raw_m - 1).bit_length())
    ub_budget = 256 * 1024 - 4096 if schedule_mode == "tuned" else 192 * 1024
    cache_g_f32 = d_alloc <= 8192
    gamma_bytes = d_alloc * (6 if cache_g_f32 else 2)
    if int(num_stages_override) > 0:
        stages = max(1, min(2, int(num_stages_override)))
    else:
        stages = (
            2
            if (
                schedule_mode == "tuned"
                and eff_m_rows >= 56 * 2
                and d_alloc <= 8192
                and not (eff_m_rows <= 512 and d_dim <= 512)
            )
            else 1
        )
        ynorm_min = 0 if d_dim <= 128 else (4 * d_alloc)
        row_bytes_s2 = d_alloc * (7 * stages) + 64
        if stages == 2 and (gamma_bytes + ynorm_min + row_bytes_s2) > ub_budget:
            stages = 1

    if int(tile_m_override) > 0:
        max_rows = max(1, int(tile_m_override))
    else:
        ynorm_bytes_est = 4 * d_alloc
        row_bytes = d_alloc * (7 * stages) + 64
        max_rows = max(
            1,
            min(64, (ub_budget - gamma_bytes - ynorm_bytes_est) // max(1, row_bytes)),
        )
        if (d_dim % 64) != 0:
            max_rows = min(max_rows, 32)
        if eff_m_rows <= 512 and d_dim <= 512:
            max_rows = max(1, min(max_rows, (eff_m_rows + 55) // 56))
        elif max_rows > 1 and eff_m_rows >= 112 and eff_m_rows < 56 * max_rows * 2:
            max_rows = max(1, min(max_rows, max(1, eff_m_rows // 112)))
        elif max_rows > 1 and eff_m_rows < 56 * max_rows:
            max_rows = max(1, min(max_rows, max(1, eff_m_rows // 56)))
        if (d_dim % 64) == 0 and d_dim < 512:
            cacheline_rows = 512 // math.gcd(512, d_dim)
            if max_rows >= cacheline_rows and eff_m_rows >= 56 * cacheline_rows * 2:
                max_rows = max(cacheline_rows, (max_rows // cacheline_rows) * cacheline_rows)
        elif (d_dim % 64) == 0 and max_rows >= 8:
            max_rows = max(4, (max_rows // 4) * 4)

    use_reg_s = d_dim <= 128 and max_rows <= 2
    regime = (
        "aligned_reg_resident"
        if use_reg_s
        else ("aligned_single_pass_ub" if (d_dim % 64) == 0 else "unaligned_predicated")
    )
    return {
        "regime": regime,
        "tile_m": max_rows,
        "num_stages": stages,
        "cache_g_f32": cache_g_f32,
        "d_dim": int(d_dim),
        "tl_dtype": tl_dtype,
        "pass_configs": {},
        "compile_flags": list(PURE_SIMDVF_UBUF_FLAGS) if schedule_mode == "tuned" else [],
    }


def _build_add_rms_norm_dynamic_quant_prim_func(
    d_dim: int = 1024,
    tl_dtype: str = "bfloat16",
    tile_m: int = 2,
    num_stages: int = 2,
    cache_g_f32: bool = True,
    schedule_mode: str = "tuned",
    tile_m_override: int = 0,
    num_stages_override: int = 0,
    **_extra_kwargs,
):
    """Build Ascend 950 Pure RegBase SimdVF @T.prim_func for AddRmsNormDynamicQuant."""
    if d_dim <= 0:
        raise ValueError(f"d_dim must be positive, got {d_dim}")
    if tl_dtype not in ("float16", "bfloat16"):
        raise ValueError(f"unsupported dtype {tl_dtype!r}")

    num_cores = 56
    raw_rows = int(tile_m_override) if int(tile_m_override) > 0 else int(tile_m)
    rows = max(1, raw_rows)
    rows_alloc = max(8, ((rows + 7) // 8) * 8)
    raw_stages = int(num_stages_override) if int(num_stages_override) > 0 else int(num_stages)
    stages = max(1, min(2, raw_stages))
    full_chunks = d_dim // 64
    rem = d_dim % 64
    d_alloc = ((d_dim + 63) // 64) * 64
    inv_d = 1.0 / float(d_dim)
    inv_127 = 1.0 / 127.0
    use_reg_s = d_dim <= 128 and rows <= 2
    pipe_ann = {"enable_offset": True} if schedule_mode == "tuned" else None
    m_rows = T.dynamic("m_rows")

    @T.macro
    def _fused_row_reg(x1_ub, x2_ub, g_src_ub, xout_ub, y_ub, scale_1d_ub, epsilon, slot):
        m32 = T.simd.pset(32, "PAT_ALL")
        m16 = T.simd.pset(16, "PAT_ALL")
        v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
        v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
        v_127 = T.simd.vdup(T.float32(127.0), "float32", mask=m32)
        sq_acc = T.simd.alloc_local((1,), "float32")
        sq_acc[0] = v_zero
        n_reg = full_chunks + (1 if rem > 0 else 0)
        s_reg = T.simd.alloc_local((n_reg,), "float32")

        for c in T.unroll(full_chunks):
            off = c * 64
            a = T.simd.vcvt(T.simd.vld(x1_ub[slot, off], dist="US_B16"), "float32", mask=m16, part=0)
            b = T.simd.vcvt(T.simd.vld(x2_ub[slot, off], dist="US_B16"), "float32", mask=m16, part=0)
            s = T.simd.vadd(a, b, mask=m32)
            s_reg[c] = s
            s16 = T.simd.vcvt(s, tl_dtype, mask=m16, part=0)
            T.simd.vsts(xout_ub[slot, off], s16, mask=m32, dist="PK_B32", extent=64)
            T.simd.vmula(sq_acc[0], s, s, mask=m32)
        if rem > 0:
            off = full_chunks * 64
            m32_t = T.simd.update_mask(rem, width=32)
            m16_t = T.simd.update_mask(rem * 2, width=16)
            a_t = T.simd.vcvt(T.simd.vld(x1_ub[slot, off], dist="US_B16"), "float32", mask=m16_t, part=0)
            b_t = T.simd.vcvt(T.simd.vld(x2_ub[slot, off], dist="US_B16"), "float32", mask=m16_t, part=0)
            s_t = T.simd.vadd(a_t, b_t, mask=m32_t)
            s_reg[full_chunks] = s_t
            s16_t = T.simd.vcvt(s_t, tl_dtype, mask=m16_t, part=0)
            T.simd.vsts(xout_ub[slot, off], s16_t, mask=m32_t, dist="PK_B32", extent=64)
            T.simd.vmula(sq_acc[0], s_t, s_t, mask=m32_t, mode="MODE_MERGING")

        sum_sq = T.simd.vdupv(T.simd.vcadd(sq_acc[0], mask=m32), mask=m32, pos="POS_LOWEST")
        rstd = T.simd.vdiv(
            v_one,
            T.simd.vsqrt(T.simd.vadds(T.simd.vmuls(sum_sq, T.float32(inv_d), mask=m32), epsilon, mask=m32), mask=m32),
            mask=m32,
            precision="ftz_true",
        )

        amax_acc = T.simd.alloc_local((1,), "float32")
        amax_acc[0] = v_zero
        for c in T.unroll(full_chunks):
            off = c * 64
            gf = T.simd.vld(g_src_ub[off], dist="NORM_B32")
            yn = T.simd.vmul(T.simd.vmul(s_reg[c], rstd, mask=m32), gf, mask=m32)
            s_reg[c] = yn
            amax_acc[0] = T.simd.vmax(amax_acc[0], T.simd.vabs(yn, mask=m32), mask=m32)
        if rem > 0:
            off = full_chunks * 64
            m32_t = T.simd.update_mask(rem, width=32)
            gf_t = T.simd.vld(g_src_ub[off], dist="NORM_B32")
            yn_t = T.simd.vmul(T.simd.vmul(s_reg[full_chunks], rstd, mask=m32_t), gf_t, mask=m32_t)
            s_reg[full_chunks] = yn_t
            amax_acc[0] = T.simd.vmax(amax_acc[0], T.simd.vabs(yn_t, mask=m32_t), mask=m32_t, mode="MODE_MERGING")

        abs_max = T.simd.vdupv(T.simd.vcmax(amax_acc[0], mask=m32), mask=m32, pos="POS_LOWEST")
        safe_abs = T.simd.vmaxs(abs_max, T.float32(1.0e-12), mask=m32)
        scale_v = T.simd.vmuls(safe_abs, T.float32(inv_127), mask=m32)
        inv_scale = T.simd.vdiv(v_127, safe_abs, mask=m32, precision="ftz_true")
        T.simd.vsts(scale_1d_ub[slot], scale_v, mask=m32, dist="ONEPT_B32")

        for c in T.unroll(full_chunks):
            off = c * 64
            q_f = T.simd.vmaxs(T.simd.vmins(T.simd.vmul(s_reg[c], inv_scale, mask=m32), T.float32(127.0), mask=m32), T.float32(-128.0), mask=m32)
            q_f16 = T.simd.vcvt(q_f, "float16", mask=m16, round="ROUND_R", sat=True, part=0)
            q_i8 = T.simd.vcvt(q_f16, "int8", mask=m16, round="ROUND_R", sat=True, part=0)
            T.simd.vsts(y_ub[slot, off], q_i8, mask=m32, dist="PK4_B32", extent=64)
        if rem > 0:
            off = full_chunks * 64
            m32_t = T.simd.update_mask(rem, width=32)
            m16_t = T.simd.update_mask(rem * 2, width=16)
            q_f_t = T.simd.vmaxs(T.simd.vmins(T.simd.vmul(s_reg[full_chunks], inv_scale, mask=m32_t), T.float32(127.0), mask=m32_t), T.float32(-128.0), mask=m32_t)
            q_f16_t = T.simd.vcvt(q_f_t, "float16", mask=m16_t, round="ROUND_R", sat=True, part=0)
            q_i8_t = T.simd.vcvt(q_f16_t, "int8", mask=m16_t, round="ROUND_R", sat=True, part=0)
            T.simd.vsts(y_ub[slot, off], q_i8_t, mask=m32_t, dist="PK4_B32", extent=64)

    @T.macro
    def _fused_row_ub(x1_ub, x2_ub, g_src_ub, xout_ub, ynorm_ub, y_ub, scale_1d_ub, epsilon, slot):
        m32 = T.simd.pset(32, "PAT_ALL")
        m16 = T.simd.pset(16, "PAT_ALL")
        v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
        v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
        v_127 = T.simd.vdup(T.float32(127.0), "float32", mask=m32)
        sq_acc = T.simd.alloc_local((1,), "float32")
        sq_acc[0] = v_zero

        for c in T.serial(full_chunks):
            off = c * 64
            a = T.simd.vcvt(T.simd.vld(x1_ub[slot, off], dist="US_B16"), "float32", mask=m16, part=0)
            b = T.simd.vcvt(T.simd.vld(x2_ub[slot, off], dist="US_B16"), "float32", mask=m16, part=0)
            s = T.simd.vadd(a, b, mask=m32)
            T.simd.vsts(ynorm_ub[0, off], s, mask=m32, dist="NORM_B32")
            s16 = T.simd.vcvt(s, tl_dtype, mask=m16, part=0)
            T.simd.vsts(xout_ub[slot, off], s16, mask=m32, dist="PK_B32", extent=64)
            T.simd.vmula(sq_acc[0], s, s, mask=m32)
        if rem > 0:
            off = full_chunks * 64
            m32_t = T.simd.update_mask(rem, width=32)
            m16_t = T.simd.update_mask(rem * 2, width=16)
            a_t = T.simd.vcvt(T.simd.vld(x1_ub[slot, off], dist="US_B16"), "float32", mask=m16_t, part=0)
            b_t = T.simd.vcvt(T.simd.vld(x2_ub[slot, off], dist="US_B16"), "float32", mask=m16_t, part=0)
            s_t = T.simd.vadd(a_t, b_t, mask=m32_t)
            T.simd.vsts(ynorm_ub[0, off], s_t, mask=m32_t, dist="NORM_B32")
            s16_t = T.simd.vcvt(s_t, tl_dtype, mask=m16_t, part=0)
            T.simd.vsts(xout_ub[slot, off], s16_t, mask=m32_t, dist="PK_B32", extent=64)
            T.simd.vmula(sq_acc[0], s_t, s_t, mask=m32_t, mode="MODE_MERGING")

        T.simd.mem_bar("VST_VLD")
        sum_sq = T.simd.vdupv(T.simd.vcadd(sq_acc[0], mask=m32), mask=m32, pos="POS_LOWEST")
        rstd = T.simd.vdiv(
            v_one,
            T.simd.vsqrt(T.simd.vadds(T.simd.vmuls(sum_sq, T.float32(inv_d), mask=m32), epsilon, mask=m32), mask=m32),
            mask=m32,
            precision="ftz_true",
        )

        amax_acc = T.simd.alloc_local((1,), "float32")
        amax_acc[0] = v_zero
        for c in T.serial(full_chunks):
            off = c * 64
            s = T.simd.vld(ynorm_ub[0, off], dist="NORM_B32")
            if cache_g_f32:
                gf = T.simd.vld(g_src_ub[off], dist="NORM_B32")
            else:
                gf = T.simd.vcvt(T.simd.vld(g_src_ub[off], dist="US_B16"), "float32", mask=m16, part=0)
            yn = T.simd.vmul(T.simd.vmul(s, rstd, mask=m32), gf, mask=m32)
            T.simd.vsts(ynorm_ub[0, off], yn, mask=m32, dist="NORM_B32")
            amax_acc[0] = T.simd.vmax(amax_acc[0], T.simd.vabs(yn, mask=m32), mask=m32)
        if rem > 0:
            off = full_chunks * 64
            m32_t = T.simd.update_mask(rem, width=32)
            s_t = T.simd.vld(ynorm_ub[0, off], dist="NORM_B32")
            if cache_g_f32:
                gf_t = T.simd.vld(g_src_ub[off], dist="NORM_B32")
            else:
                m16_t = T.simd.update_mask(rem * 2, width=16)
                gf_t = T.simd.vcvt(T.simd.vld(g_src_ub[off], dist="US_B16"), "float32", mask=m16_t, part=0)
            yn_t = T.simd.vmul(T.simd.vmul(s_t, rstd, mask=m32_t), gf_t, mask=m32_t)
            T.simd.vsts(ynorm_ub[0, off], yn_t, mask=m32_t, dist="NORM_B32")
            amax_acc[0] = T.simd.vmax(amax_acc[0], T.simd.vabs(yn_t, mask=m32_t), mask=m32_t, mode="MODE_MERGING")

        T.simd.mem_bar("VST_VLD")
        abs_max = T.simd.vdupv(T.simd.vcmax(amax_acc[0], mask=m32), mask=m32, pos="POS_LOWEST")
        safe_abs = T.simd.vmaxs(abs_max, T.float32(1.0e-12), mask=m32)
        scale_v = T.simd.vmuls(safe_abs, T.float32(inv_127), mask=m32)
        inv_scale = T.simd.vdiv(v_127, safe_abs, mask=m32, precision="ftz_true")
        T.simd.vsts(scale_1d_ub[slot], scale_v, mask=m32, dist="ONEPT_B32")

        for c in T.serial(full_chunks):
            off = c * 64
            yn = T.simd.vld(ynorm_ub[0, off], dist="NORM_B32")
            q_f = T.simd.vmaxs(T.simd.vmins(T.simd.vmul(yn, inv_scale, mask=m32), T.float32(127.0), mask=m32), T.float32(-128.0), mask=m32)
            q_f16 = T.simd.vcvt(q_f, "float16", mask=m16, round="ROUND_R", sat=True, part=0)
            q_i8 = T.simd.vcvt(q_f16, "int8", mask=m16, round="ROUND_R", sat=True, part=0)
            T.simd.vsts(y_ub[slot, off], q_i8, mask=m32, dist="PK4_B32", extent=64)
        if rem > 0:
            off = full_chunks * 64
            m32_t = T.simd.update_mask(rem, width=32)
            m16_t = T.simd.update_mask(rem * 2, width=16)
            yn_t = T.simd.vld(ynorm_ub[0, off], dist="NORM_B32")
            q_f_t = T.simd.vmaxs(T.simd.vmins(T.simd.vmul(yn_t, inv_scale, mask=m32_t), T.float32(127.0), mask=m32_t), T.float32(-128.0), mask=m32_t)
            q_f16_t = T.simd.vcvt(q_f_t, "float16", mask=m16_t, round="ROUND_R", sat=True, part=0)
            q_i8_t = T.simd.vcvt(q_f16_t, "int8", mask=m16_t, round="ROUND_R", sat=True, part=0)
            T.simd.vsts(y_ub[slot, off], q_i8_t, mask=m32_t, dist="PK4_B32", extent=64)

    with _ascend_target_ctx():

        @T.prim_func
        def main(
            X1: T.Tensor((m_rows, d_dim), tl_dtype),
            X2: T.Tensor((m_rows, d_dim), tl_dtype),
            Gamma: T.Tensor((d_dim,), tl_dtype),
            Y: T.Tensor((m_rows, d_dim), "int8"),
            XOut: T.Tensor((m_rows, d_dim), tl_dtype),
            ScaleOut: T.Tensor((m_rows,), "float32"),
            epsilon: T.float32,
        ):
            with T.Kernel(num_cores) as cid:
                g_ub = T.alloc_shared((d_alloc,), tl_dtype)
                x1_ub = T.alloc_shared((rows, d_alloc), tl_dtype)
                x2_ub = T.alloc_shared((rows, d_alloc), tl_dtype)
                xout_ub = T.alloc_shared((rows, d_alloc), tl_dtype)
                y_ub = T.alloc_shared((rows, d_alloc), "int8")
                scale_1d_ub = T.alloc_shared((rows_alloc,), "float32")
                vers = {
                    g_ub: 1,
                    x1_ub: stages,
                    x2_ub: stages,
                    xout_ub: stages,
                    y_ub: stages,
                    scale_1d_ub: stages,
                }
                if not use_reg_s:
                    ynorm_ub = T.alloc_shared((1, d_alloc), "float32")
                    vers[ynorm_ub] = 1
                if cache_g_f32:
                    g_f32_ub = T.alloc_shared((d_alloc,), "float32")
                    vers[g_f32_ub] = 1
                T.annotate_buffer_versions(vers)
                if schedule_mode == "tuned":
                    T.annotate_unlimit_memory("shared")

                base_rows = m_rows // num_cores
                rem_rows = m_rows % num_cores
                m_start = cid * base_rows + T.min(cid, rem_rows)
                m_count = base_rows + T.if_then_else(cid < rem_rows, 1, 0)

                T.copy(Gamma[:d_dim], g_ub[:d_dim])
                if cache_g_f32:
                    with T.SimdVF():
                        m32 = T.simd.pset(32, "PAT_ALL")
                        m16 = T.simd.pset(16, "PAT_ALL")
                        for c in T.serial(full_chunks):
                            off = c * 64
                            gf = T.simd.vcvt(T.simd.vld(g_ub[off], dist="US_B16"), "float32", mask=m16, part=0)
                            T.simd.vsts(g_f32_ub[off], gf, mask=m32, dist="NORM_B32")
                        if rem > 0:
                            off = full_chunks * 64
                            m32_t = T.simd.update_mask(rem, width=32)
                            m16_t = T.simd.update_mask(rem * 2, width=16)
                            gf_t = T.simd.vcvt(T.simd.vld(g_ub[off], dist="US_B16"), "float32", mask=m16_t, part=0)
                            T.simd.vsts(g_f32_ub[off], gf_t, mask=m32_t, dist="NORM_B32")

                g_src = g_f32_ub if cache_g_f32 else g_ub
                num_blocks = (m_count + rows - 1) // rows
                for mb in T.Pipelined(num_blocks, num_stages=stages, annotations=pipe_ann):
                    m0 = m_start + mb * rows
                    m_cur = T.min(rows, m_count - mb * rows)
                    T.copy(X1[m0 : m0 + m_cur, :d_dim], x1_ub[:m_cur, :d_dim])
                    T.copy(X2[m0 : m0 + m_cur, :d_dim], x2_ub[:m_cur, :d_dim])
                    with T.SimdVF():
                        for rr in T.serial(m_cur):
                            if use_reg_s:
                                _fused_row_reg(x1_ub, x2_ub, g_src, xout_ub, y_ub, scale_1d_ub, epsilon, rr)
                            else:
                                _fused_row_ub(x1_ub, x2_ub, g_src, xout_ub, ynorm_ub, y_ub, scale_1d_ub, epsilon, rr)
                    T.copy(y_ub[:m_cur, :d_dim], Y[m0 : m0 + m_cur, :d_dim])
                    T.copy(xout_ub[:m_cur, :d_dim], XOut[m0 : m0 + m_cur, :d_dim])
                    T.copy(scale_1d_ub[:m_cur], ScaleOut[m0 : m0 + m_cur])

    return main


def _get_kernel(d_dim: int, tl_dtype: str, m_rows: int, schedule_mode: str):
    cfg = _select_add_rms_norm_dynamic_quant_config(
        d_dim, tl_dtype, m_rows=m_rows, schedule_mode=schedule_mode
    )
    key = (
        d_dim,
        tl_dtype,
        int(cfg["tile_m"]),
        int(cfg["num_stages"]),
        bool(cfg["cache_g_f32"]),
        str(cfg["regime"]),
        schedule_mode,
    )
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        with _ascend_target_ctx():
            prog = _build_add_rms_norm_dynamic_quant_prim_func(
                d_dim=d_dim,
                tl_dtype=tl_dtype,
                tile_m=int(cfg["tile_m"]),
                num_stages=int(cfg["num_stages"]),
                cache_g_f32=bool(cfg["cache_g_f32"]),
                schedule_mode=schedule_mode,
            )
        compiled = tilelang.compile(
            prog,
            target="ascend",
            out_idx=[3, 4, 5],
            pass_configs=cfg.get("pass_configs") or None,
            compile_flags=cfg.get("compile_flags") or None,
        )
        _KERNEL_CACHE[key] = compiled
    return compiled


def add_rms_norm_dynamic_quant(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
    *,
    schedule_mode: str = "tuned",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Add + RMSNorm + per-token symmetric INT8 quant matching proto.yaml."""
    if x1.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported dtype {x1.dtype}")

    if x1.device.type != "npu":
        out_dtype = x1.dtype
        x_sum = x1.to(torch.float32) + x2.to(torch.float32)
        var = x_sum.pow(2).mean(-1, keepdim=True)
        y_norm = (x_sum / torch.sqrt(var + float(epsilon))) * gamma.to(torch.float32)
        abs_max = y_norm.abs().amax(dim=-1, keepdim=True)
        scale_out = (abs_max.clamp(min=1e-12) / 127.0).to(torch.float32)
        y_int8 = torch.clamp((y_norm / scale_out).round(), -128, 127).to(torch.int8)
        return y_int8, x_sum.to(out_dtype), scale_out.squeeze(-1).to(torch.float32)

    tl_dtype = "float16" if x1.dtype == torch.float16 else "bfloat16"
    x1_c = x1.contiguous()
    x2_c = x2.contiguous()
    d_dim = x1_c.shape[-1]
    lead_shape = x1_c.shape[:-1]
    m_rows = max(1, x1_c.numel() // d_dim)
    compiled = _get_kernel(d_dim, tl_dtype, m_rows, schedule_mode)
    y_2d, xout_2d, scale_1d = compiled(
        x1_c.view(m_rows, d_dim),
        x2_c.view(m_rows, d_dim),
        gamma.contiguous().view(d_dim),
        float(epsilon),
    )
    return y_2d.view(x1_c.shape), xout_2d.view(x1_c.shape), scale_1d.view(lead_shape)


kernel = add_rms_norm_dynamic_quant

__all__ = [
    "_build_add_rms_norm_dynamic_quant_prim_func",
    "_select_add_rms_norm_dynamic_quant_config",
    "add_rms_norm_dynamic_quant",
    "kernel",
]
