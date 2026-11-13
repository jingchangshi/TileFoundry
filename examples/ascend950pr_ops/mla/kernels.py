"""TileLang Ascend950PR MIX kernel implementation for Multi-Head Latent Attention."""

import math

import tilelang
import tilelang.ascend.language as T
import torch

_KERNEL_CACHE: dict[tuple[object, ...], object] = {}


def _build_mla_prim_func(
    b_dim: int,
    s_dim: int,
    s_kv: int,
    n_q: int,
    n_kv: int,
    d_nope: int,
    d_rope: int,
    tl_dtype: str,
    input_layout: str,
    is_causal: bool,
    schedule_mode: str,
):
    bm = 32
    rows_per_sid = 16
    bn = 64
    d_nope_pow2 = tilelang.next_power_of_2(d_nope)
    d0 = min(d_nope, 256)
    d1 = d_nope - d0
    d1_alloc = max(d1, 16)
    d1_pow2 = tilelang.next_power_of_2(d1_alloc)
    g_ratio = n_q // n_kv
    q_rows = s_dim * g_ratio
    m_tiles_per_group = (q_rows + bm - 1) // bm
    total_groups = b_dim * n_kv
    total_m_tiles = total_groups * m_tiles_per_group
    num_cores = min(total_m_tiles, 28)
    base_tiles = total_m_tiles // num_cores
    rem_tiles = total_m_tiles % num_cores

    max_kv_blocks = (s_kv + bn - 1) // bn
    num_stages = 2 if (schedule_mode == "tuned" and max_kv_blocks >= 2) else 1
    use_causal_bound = bool(is_causal and schedule_mode == "tuned")

    q_total_rows = b_dim * s_dim * n_q
    k_nope_shape = (
        (b_dim, s_kv, n_kv, d_nope) if input_layout == "BSND" else (b_dim, n_kv, s_kv, d_nope)
    )
    k_rope_shape = (
        (b_dim, s_kv, n_kv, d_rope) if input_layout == "BSND" else (b_dim, n_kv, s_kv, d_rope)
    )

    @T.prim_func
    def main(
        Q_nope: T.Tensor((q_total_rows, d_nope), tl_dtype),
        Q_rope: T.Tensor((q_total_rows, d_rope), tl_dtype),
        K_nope: T.Tensor(k_nope_shape, tl_dtype),
        K_rope: T.Tensor(k_rope_shape, tl_dtype),
        V: T.Tensor(k_nope_shape, tl_dtype),
        Out: T.Tensor((q_total_rows, d_nope), tl_dtype),
        scale_val: T.float32,
    ):
        with T.MixedKernel(num_cores, sids=2) as (bx, sid):
            q_nope_l1 = T.alloc_l1((bm, d_nope), tl_dtype)
            q_rope_l1 = T.alloc_l1((bm, d_rope), tl_dtype)
            k_nope_l1 = T.alloc_l1((bn, d_nope), tl_dtype)
            k_rope_l1 = T.alloc_l1((bn, d_rope), tl_dtype)
            p_l1 = T.alloc_l1((bm, bn), tl_dtype)
            v0_l1 = T.alloc_l1((d0, bn), tl_dtype)
            v1_l1 = T.alloc_l1((d1_alloc, bn), tl_dtype)

            s_l0c = T.alloc_l0c((bm, bn), "float32")
            o0_l0c = T.alloc_l0c((bm, d0), "float32")
            o1_l0c = T.alloc_l0c((bm, d1_alloc), "float32")

            q_nope_ub = T.alloc_shared((rows_per_sid, d_nope), tl_dtype)
            q_rope_ub = T.alloc_shared((rows_per_sid, d_rope), tl_dtype)
            S_ub = T.alloc_shared((rows_per_sid, bn), "float32")
            P_ub = T.alloc_shared((rows_per_sid, bn), tl_dtype)
            O_tmp0_ub = T.alloc_shared((rows_per_sid, d0), "float32")
            O_tmp1_ub = T.alloc_shared((rows_per_sid, d1_alloc), "float32")
            O_ub = T.alloc_shared((rows_per_sid, d_nope), "float32")
            m_ub = T.alloc_shared((rows_per_sid,), "float32")
            l_ub = T.alloc_shared((rows_per_sid,), "float32")
            alpha_ub = T.alloc_shared((rows_per_sid,), "float32")

            T.annotate_buffer_versions(
                {
                    q_nope_l1: 1,
                    q_rope_l1: 1,
                    k_nope_l1: num_stages,
                    k_rope_l1: num_stages,
                    v0_l1: num_stages,
                    v1_l1: num_stages,
                    p_l1: 1,
                    s_l0c: 1,
                    o0_l0c: 1,
                    o1_l0c: 1,
                    S_ub: 1,
                    P_ub: 1,
                    O_tmp0_ub: 1,
                    O_tmp1_ub: 1,
                    O_ub: 1,
                    m_ub: 1,
                    l_ub: 1,
                    alpha_ub: 1,
                }
            )

            t_start = bx * base_tiles + T.min(bx, rem_tiles)
            t_count = base_tiles + T.if_then_else(bx < rem_tiles, 1, 0)
            for local_t in T.serial(t_count):
                tile_id = t_start + local_t
                group_id = tile_id // m_tiles_per_group
                mt = tile_id % m_tiles_per_group
                b_idx = group_id // n_kv
                kv_h = group_id % n_kv
                q_base = mt * bm

                with T.SimtVF(threads=256):
                    for r, d in T.Parallel(rows_per_sid, d_nope_pow2, coalesced_width=T.int32(1)):
                        if d < d_nope:
                            q_row = q_base + sid * rows_per_sid + r
                            if q_row < q_rows:
                                if input_layout == "BSND":
                                    s_i = q_row // g_ratio
                                    h_loc = q_row % g_ratio
                                    q_h = kv_h * g_ratio + h_loc
                                    gm_r = (b_idx * s_dim + s_i) * n_q + q_h
                                else:
                                    gm_r = group_id * q_rows + q_row
                                q_nope_ub[r, d] = Q_nope[gm_r, d]
                            else:
                                q_nope_ub[r, d] = T.cast(T.float32(0.0), tl_dtype)
                    for r, d in T.Parallel(rows_per_sid, d_rope, coalesced_width=T.int32(1)):
                        q_row = q_base + sid * rows_per_sid + r
                        if q_row < q_rows:
                            if input_layout == "BSND":
                                s_i = q_row // g_ratio
                                h_loc = q_row % g_ratio
                                q_h = kv_h * g_ratio + h_loc
                                gm_r = (b_idx * s_dim + s_i) * n_q + q_h
                            else:
                                gm_r = group_id * q_rows + q_row
                            q_rope_ub[r, d] = Q_rope[gm_r, d]
                        else:
                            q_rope_ub[r, d] = T.cast(T.float32(0.0), tl_dtype)

                T.dual_copy(q_nope_ub, q_nope_l1)
                T.dual_copy(q_rope_ub, q_rope_l1)

                with T.SimtVF(threads=256):
                    for r in T.Parallel(rows_per_sid):
                        m_ub[r] = T.float32(-1e38)
                        l_ub[r] = T.float32(0.0)
                    for r, d in T.Parallel(rows_per_sid, d_nope_pow2, coalesced_width=T.int32(1)):
                        if d < d_nope:
                            O_ub[r, d] = T.float32(0.0)

                last_valid = T.min(q_rows - 1, q_base + bm - 1)
                if use_causal_bound:
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

                for kv in T.Pipelined(kv_limit, num_stages=num_stages):
                    if input_layout == "BSND":
                        T.copy(K_nope[b_idx, kv * bn : kv * bn + bn, kv_h, 0:d_nope], k_nope_l1)
                        T.copy(K_rope[b_idx, kv * bn : kv * bn + bn, kv_h, 0:d_rope], k_rope_l1)
                        T.copy(V[b_idx, kv * bn : kv * bn + bn, kv_h, 0:d0], v0_l1, transpose=True)
                        if d1 > 0:
                            T.copy(
                                V[b_idx, kv * bn : kv * bn + bn, kv_h, d0:d_nope],
                                v1_l1,
                                transpose=True,
                            )
                    else:
                        T.copy(K_nope[b_idx, kv_h, kv * bn : kv * bn + bn, 0:d_nope], k_nope_l1)
                        T.copy(K_rope[b_idx, kv_h, kv * bn : kv * bn + bn, 0:d_rope], k_rope_l1)
                        T.copy(V[b_idx, kv_h, kv * bn : kv * bn + bn, 0:d0], v0_l1, transpose=True)
                        if d1 > 0:
                            T.copy(
                                V[b_idx, kv_h, kv * bn : kv * bn + bn, d0:d_nope],
                                v1_l1,
                                transpose=True,
                            )

                    T.gemm(q_nope_l1, k_nope_l1, s_l0c, transpose_B=True, clear_accum=True)
                    T.gemm(q_rope_l1, k_rope_l1, s_l0c, transpose_B=True, clear_accum=False)
                    T.dual_copy(s_l0c, S_ub)

                    with T.SimtVF(threads=64):
                        for r in T.serial(rows_per_sid):
                            q_row = q_base + sid * rows_per_sid + r
                            if input_layout == "BSND":
                                s_idx = q_row // g_ratio
                            else:
                                s_idx = q_row % s_dim
                            s_frag = T.alloc_fragment((bn,), "float32")
                            for j in T.Parallel(bn, coalesced_width=T.int32(1)):
                                kv_col = kv * bn + j
                                valid_pos = (q_row < q_rows) and (kv_col < s_kv)
                                if is_causal:
                                    valid_mask = valid_pos and (kv_col <= s_idx + s_kv - s_dim)
                                else:
                                    valid_mask = valid_pos
                                s_frag[j] = T.if_then_else(
                                    valid_mask,
                                    S_ub[r, j] * scale_val,
                                    T.float32(-1e38),
                                )

                            m_red = T.alloc_reducer((1,), "float32", op="max")
                            T.reducer_init(m_red)
                            for j in T.Parallel(bn, coalesced_width=T.int32(1)):
                                T.reducer_update(m_red[0], s_frag[j])
                            m_tile = T.alloc_fragment((1,), "float32")
                            T.finalize_reducer(m_red, m_tile)

                            for k in T.Parallel(1):
                                m_old = m_ub[r]
                                m_new = T.max(m_old, m_tile[k])
                                alpha_ub[r] = T.if_then_else(
                                    m_old > T.float32(-1e30),
                                    T.exp(m_old - m_new),
                                    T.float32(0.0),
                                )
                                m_ub[r] = m_new

                            p_frag = T.alloc_fragment((bn,), "float32")
                            for j in T.Parallel(bn, coalesced_width=T.int32(1)):
                                kv_col = kv * bn + j
                                valid_pos = (q_row < q_rows) and (kv_col < s_kv)
                                if is_causal:
                                    valid_mask = valid_pos and (kv_col <= s_idx + s_kv - s_dim)
                                else:
                                    valid_mask = valid_pos
                                m_new_r = m_ub[r]
                                p_frag[j] = T.if_then_else(
                                    valid_mask and (m_new_r > T.float32(-1e30)),
                                    T.exp(s_frag[j] - m_new_r),
                                    T.float32(0.0),
                                )

                            row_sum_red = T.alloc_reducer((1,), "float32", op="sum")
                            T.reducer_init(row_sum_red)
                            for j in T.Parallel(bn, coalesced_width=T.int32(1)):
                                T.reducer_update(row_sum_red[0], p_frag[j])
                            row_sum_frag = T.alloc_fragment((1,), "float32")
                            T.finalize_reducer(row_sum_red, row_sum_frag)

                            for k in T.Parallel(1):
                                l_ub[r] = l_ub[r] * alpha_ub[r] + row_sum_frag[k]
                            for j in T.Parallel(bn, coalesced_width=T.int32(1)):
                                P_ub[r, j] = T.cast(p_frag[j], tl_dtype)

                    T.dual_copy(P_ub, p_l1)
                    T.gemm(p_l1, v0_l1, o0_l0c, transpose_B=True, clear_accum=True)
                    T.dual_copy(o0_l0c, O_tmp0_ub)
                    if d1 > 0:
                        T.gemm(p_l1, v1_l1, o1_l0c, transpose_B=True, clear_accum=True)
                        T.dual_copy(o1_l0c, O_tmp1_ub)

                    with T.SimtVF(threads=256):
                        for r, d in T.Parallel(rows_per_sid, d0, coalesced_width=T.int32(1)):
                            O_ub[r, d] = O_ub[r, d] * alpha_ub[r] + O_tmp0_ub[r, d]
                        if d1 > 0:
                            for r, d in T.Parallel(
                                rows_per_sid, d1_pow2, coalesced_width=T.int32(1)
                            ):
                                if d < d1:
                                    O_ub[r, d0 + d] = (
                                        O_ub[r, d0 + d] * alpha_ub[r] + O_tmp1_ub[r, d]
                                    )

                with T.SimtVF(threads=256):
                    for r, d in T.Parallel(rows_per_sid, d_nope_pow2, coalesced_width=T.int32(1)):
                        if d < d_nope:
                            q_row = q_base + sid * rows_per_sid + r
                            if q_row < q_rows:
                                l_val = l_ub[r]
                                norm_val = T.if_then_else(
                                    l_val > T.float32(0.0),
                                    O_ub[r, d] / l_val,
                                    T.float32(0.0),
                                )
                                if input_layout == "BSND":
                                    s_i = q_row // g_ratio
                                    h_loc = q_row % g_ratio
                                    q_h = kv_h * g_ratio + h_loc
                                    out_r = (b_idx * s_dim + s_i) * n_q + q_h
                                else:
                                    out_r = group_id * q_rows + q_row
                                Out[out_r, d] = T.cast(norm_val, tl_dtype)

    return main


def _get_kernel(
    b_dim: int,
    s_dim: int,
    s_kv: int,
    n_q: int,
    n_kv: int,
    d_nope: int,
    d_rope: int,
    tl_dtype: str,
    input_layout: str,
    is_causal: bool,
    schedule_mode: str,
):
    key = (
        b_dim,
        s_dim,
        s_kv,
        n_q,
        n_kv,
        d_nope,
        d_rope,
        tl_dtype,
        input_layout,
        bool(is_causal),
        schedule_mode,
    )
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        prog = _build_mla_prim_func(*key)
        compiled = tilelang.compile(prog, target="ascend", out_idx=[5])
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
) -> torch.Tensor:
    """Compute Multi-Head Latent Attention on Ascend950PR using a MIX TileLang kernel."""
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

    q_rows_total = b_dim * s_dim * n_q
    q_n_flat = q_nope.contiguous().view(q_rows_total, d_nope)
    q_r_flat = q_rope.contiguous().view(q_rows_total, d_rope)
    k_n_contig = k_nope.contiguous()
    k_r_contig = k_rope.contiguous()
    v_contig = v.contiguous()

    kernel = _get_kernel(
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
    out_flat = kernel(q_n_flat, q_r_flat, k_n_contig, k_r_contig, v_contig, scale)
    if inputLayout == "BSND":
        return out_flat.view(b_dim, s_dim, n_q, d_nope)
    return out_flat.view(b_dim, n_q, s_dim, d_nope)


__all__ = ["mla"]
