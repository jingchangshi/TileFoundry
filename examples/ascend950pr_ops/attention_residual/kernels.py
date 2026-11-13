"""TileLang Ascend950PR SimdVF kernel implementation for AttentionResidual."""

import math

import tilelang
import tilelang.ascend.language as T
import torch

from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS

_KERNEL_CACHE: dict[tuple[int, int, str, int, str], object] = {}


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


def _get_kernel(
    n_depth: int,
    d_dim: int,
    tl_dtype: str,
    tile_m: int,
    schedule_mode: str,
):
    key = (n_depth, d_dim, tl_dtype, int(tile_m), schedule_mode)
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        prog = _build_attention_residual_prim_func(
            n_depth, d_dim, tl_dtype, tile_m, schedule_mode
        )
        flags = list(PURE_SIMDVF_UBUF_FLAGS) if schedule_mode == "tuned" else None
        compiled = tilelang.compile(prog, target="ascend", out_idx=[4], compile_flags=flags)
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

    tile_m = _choose_ar_tile_m(d_dim, schedule_mode)
    kernel = _get_kernel(n_depth, d_dim, tl_dtype, tile_m, schedule_mode)
    y_flat = kernel(v_flat, x_flat, w_flat, g_flat, float(alpha), float(epsilon))
    return y_flat.view(b_dim, t_dim, d_dim)


__all__ = ["attention_residual"]
