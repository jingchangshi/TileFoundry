"""TileLang Ascend950PR kernel implementation for AttentionResidual."""

from __future__ import annotations

import math

import tilelang
import tilelang.ascend.language as T
import torch

_KERNEL_CACHE: dict[tuple[int, int, int, str, str], object] = {}


def _build_attention_residual_prim_func(
    n_depth: int,
    m_rows: int,
    d_dim: int,
    tl_dtype: str,
    schedule_mode: str,
):
    num_cores = min(m_rows, 56)
    base_rows = m_rows // num_cores
    rem_rows = m_rows % num_cores
    tile_d = tilelang.next_power_of_2(d_dim)
    n_pow2 = max(64, tilelang.next_power_of_2(n_depth))
    threads = min(256 if schedule_mode == "tuned" else 128, tile_d)
    num_stages = 2 if (schedule_mode == "tuned" and n_depth >= 2) else 1
    inv_sqrt_d = 1.0 / math.sqrt(d_dim)
    inv_d = 1.0 / float(d_dim)

    @T.prim_func
    def main(
        V: T.Tensor((n_depth * m_rows, d_dim), tl_dtype),
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
            v_ub = T.alloc_shared((d_dim,), tl_dtype)
            x_ub = T.alloc_shared((d_dim,), tl_dtype)
            acc_ub = T.alloc_shared((d_dim,), "float32")
            scores_ub = T.alloc_shared((n_pow2,), "float32")
            probs_ub = T.alloc_shared((n_pow2,), "float32")
            T.annotate_buffer_versions(
                {
                    v_ub: num_stages,
                    acc_ub: 1,
                    scores_ub: 1,
                    probs_ub: 1,
                    w_ub: 1,
                    g_ub: 1,
                    x_ub: 1,
                }
            )
            m_start = cid * base_rows + T.min(cid, rem_rows)
            m_count = base_rows + T.if_then_else(cid < rem_rows, 1, 0)
            for local_m in T.serial(m_count):
                T.copy(W[:d_dim], w_ub[:d_dim])
                T.copy(G[:d_dim], g_ub[:d_dim])
                m = m_start + local_m
                for n in T.Pipelined(n_depth, num_stages=num_stages):
                    row_off = n * m_rows + m
                    T.copy(V[row_off, :d_dim], v_ub[:d_dim])
                    with T.SimtVF(threads=threads):
                        v_frag = T.alloc_fragment((tile_d,), "float32")
                        for i in T.Parallel(tile_d, coalesced_width=T.int32(1)):
                            v_frag[i] = T.if_then_else(
                                i < d_dim,
                                T.cast(v_ub[i], "float32"),
                                T.float32(0.0),
                            )
                        sum_sq = T.alloc_reducer((1,), "float32", op="sum")
                        T.reducer_init(sum_sq)
                        for i in T.Parallel(tile_d, coalesced_width=T.int32(1)):
                            if i < d_dim:
                                T.reducer_update(sum_sq[0], v_frag[i] * v_frag[i])
                        sum_sq_res = T.alloc_fragment((1,), "float32")
                        T.finalize_reducer(sum_sq, sum_sq_res)
                        dot_acc = T.alloc_reducer((1,), "float32", op="sum")
                        T.reducer_init(dot_acc)
                        rstd_val = T.rsqrt(sum_sq_res[0] * T.float32(inv_d) + eps_val)
                        for i in T.Parallel(tile_d, coalesced_width=T.int32(1)):
                            if i < d_dim:
                                prod_val = (
                                    v_frag[i]
                                    * rstd_val
                                    * T.cast(w_ub[i], "float32")
                                    * T.cast(g_ub[i], "float32")
                                )
                                T.reducer_update(dot_acc[0], prod_val)
                        dot_res = T.alloc_fragment((1,), "float32")
                        T.finalize_reducer(dot_acc, dot_res)
                        for r in T.Parallel(1):
                            scores_ub[n + r] = dot_res[r] * T.float32(inv_sqrt_d)

                with T.SimtVF(threads=64):
                    s_frag = T.alloc_fragment((n_pow2,), "float32")
                    for k in T.Parallel(n_pow2, coalesced_width=T.int32(1)):
                        s_frag[k] = T.if_then_else(
                            k < n_depth,
                            scores_ub[k],
                            T.float32(-1e38),
                        )
                    s_max_red = T.alloc_reducer((1,), "float32", op="max")
                    T.reducer_init(s_max_red)
                    for k in T.Parallel(n_pow2, coalesced_width=T.int32(1)):
                        if k < n_depth:
                            T.reducer_update(s_max_red[0], s_frag[k])
                    s_max_res = T.alloc_fragment((1,), "float32")
                    T.finalize_reducer(s_max_red, s_max_res)
                    e_frag = T.alloc_fragment((n_pow2,), "float32")
                    for k in T.Parallel(n_pow2, coalesced_width=T.int32(1)):
                        e_frag[k] = T.if_then_else(
                            k < n_depth,
                            T.exp(s_frag[k] - s_max_res[0]),
                            T.float32(0.0),
                        )
                    s_sum_red = T.alloc_reducer((1,), "float32", op="sum")
                    T.reducer_init(s_sum_red)
                    for k in T.Parallel(n_pow2, coalesced_width=T.int32(1)):
                        if k < n_depth:
                            T.reducer_update(s_sum_red[0], e_frag[k])
                    s_sum_res = T.alloc_fragment((1,), "float32")
                    T.finalize_reducer(s_sum_red, s_sum_res)
                    for k in T.Parallel(n_pow2, coalesced_width=T.int32(1)):
                        if k < n_depth:
                            probs_ub[k] = e_frag[k] / s_sum_res[0]

                with T.SimtVF(threads=threads):
                    for i in T.Parallel(tile_d, coalesced_width=T.int32(1)):
                        if i < d_dim:
                            acc_ub[i] = T.float32(0.0)

                for n in T.Pipelined(n_depth, num_stages=num_stages):
                    row_off = n * m_rows + m
                    T.copy(V[row_off, :d_dim], v_ub[:d_dim])
                    with T.SimtVF(threads=threads):
                        p_val = probs_ub[n]
                        for i in T.Parallel(tile_d, coalesced_width=T.int32(1)):
                            if i < d_dim:
                                acc_ub[i] = acc_ub[i] + T.cast(v_ub[i], "float32") * p_val

                T.copy(X[m, :d_dim], x_ub[:d_dim])
                with T.SimtVF(threads=threads):
                    for i in T.Parallel(tile_d, coalesced_width=T.int32(1)):
                        if i < d_dim:
                            blended = (
                                T.float32(1.0) - alpha_val
                            ) * T.cast(x_ub[i], "float32") + alpha_val * acc_ub[i]
                            x_ub[i] = T.cast(blended, tl_dtype)
                T.copy(x_ub[:d_dim], Y[m, :d_dim])
                T.ascend_pipe_barrier("PIPE_ALL")

    return main


def _get_kernel(
    n_depth: int,
    m_rows: int,
    d_dim: int,
    tl_dtype: str,
    schedule_mode: str,
):
    key = (n_depth, m_rows, d_dim, tl_dtype, schedule_mode)
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        prog = _build_attention_residual_prim_func(
            n_depth, m_rows, d_dim, tl_dtype, schedule_mode
        )
        compiled = tilelang.compile(prog, target="ascend", out_idx=[4])
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
    """Compute AttentionResidual on Ascend950PR NPU using a compiled TileLang kernel."""
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

    kernel = _get_kernel(n_depth, m_rows, d_dim, tl_dtype, schedule_mode)
    y_flat = kernel(v_flat, x_flat, w_flat, g_flat, float(alpha), float(epsilon))
    return y_flat.view(b_dim, t_dim, d_dim)


__all__ = ["attention_residual"]
