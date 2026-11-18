"""TileLang Ascend950PR Heterogeneous CV-Ring kernel implementation for GQA."""

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


def _select_gqa_config(
    s_dim: int = 128,
    n_q: int = 32,
    n_kv: int = 8,
    d_dim: int = 128,
    tl_dtype: str = "float16",
    *,
    is_causal: bool = False,
    schedule_mode: str = "tuned",
) -> dict[str, object]:
    """Select Cube-bound bm, bn=128 (tuned) / 64 (baseline), and 2-stage CV ring config for GQA."""
    g_ratio = max(1, n_q // max(1, n_kv))
    q_rows = s_dim * g_ratio
    if s_dim <= 4 and q_rows < 64:
        bm = 96 if (32 % g_ratio) != 0 and (48 % g_ratio) == 0 else 64
        regime = "decode_group_fold"
    elif schedule_mode == "tuned" and d_dim <= 128 and q_rows >= 96:
        if (64 % g_ratio) != 0 and (48 % g_ratio) == 0:
            bm = 96
            regime = "prefill_cv_ring_bm96"
        elif s_dim * n_kv >= 512:
            bm = 128
            regime = "prefill_cv_ring_bm128"
        else:
            bm = 64
            regime = "prefill_cv_ring_bm64"
    else:
        bm = 96 if (32 % g_ratio) != 0 and (48 % g_ratio) == 0 else 64
        regime = "prefill_cv_ring_bm64"
    bn = 128 if (schedule_mode == "tuned" and d_dim <= 128) else 64
    l1_stages = 2 if schedule_mode == "tuned" else 1
    return {
        "regime": regime,
        "bm": bm,
        "bn": bn,
        "l1_stages": l1_stages,
        "n_q": int(n_q),
        "n_kv": int(n_kv),
        "d_dim": int(d_dim),
        "tl_dtype": tl_dtype,
        "is_causal": bool(is_causal),
        "pass_configs": {},
        "compile_flags": list(PURE_SIMDVF_UBUF_FLAGS) if schedule_mode == "tuned" else [],
    }


def _build_gqa_prim_func(
    n_q: int = 32,
    n_kv: int = 8,
    d_dim: int = 128,
    tl_dtype: str = "float16",
    is_causal: bool = False,
    bm: int = 64,
    bn: int = 128,
    l1_stages: int = 2,
    schedule_mode: str = "tuned",
    **_extra_kwargs,
):
    """Build Ascend 950 Heterogeneous CV-Ring @T.prim_func for GQA [B, S, N_q, D]."""
    if (n_q % n_kv) != 0:
        raise ValueError(f"n_q ({n_q}) must be divisible by n_kv ({n_kv})")
    if (d_dim % 64) != 0:
        raise ValueError(f"d_dim ({d_dim}) must be divisible by 64")
    if bm not in (64, 96, 128):
        raise ValueError(f"unsupported bm={bm}")
    if bn not in (64, 128):
        raise ValueError(f"unsupported bn={bn}")
    if tl_dtype not in ("float16", "bfloat16"):
        raise ValueError(f"unsupported dtype {tl_dtype!r}")

    num_cores = 28
    g_ratio = n_q // n_kv
    rows_per_sid = bm // 2
    d_chunks = d_dim // 64
    use_group_dma = (g_ratio >= 2) and (g_ratio <= rows_per_sid) and ((rows_per_sid % g_ratio) == 0)
    stages = max(1, min(2, int(l1_stages)))
    pipe_ann = {"enable_offset": True} if schedule_mode == "tuned" else None

    b_dim = T.dynamic("b_dim")
    s_dim = T.dynamic("s_dim")
    s_kv = T.dynamic("s_kv")

    with _ascend_target_ctx():

        @T.prim_func
        def main(
            Q: T.Tensor((b_dim, s_dim, n_q, d_dim), tl_dtype),
            K: T.Tensor((b_dim, s_kv, n_kv, d_dim), tl_dtype),
            V: T.Tensor((b_dim, s_kv, n_kv, d_dim), tl_dtype),
            Out: T.Tensor((b_dim, s_dim, n_q, d_dim), tl_dtype),
            scale_val: T.float32,
        ):
            with T.MixedKernel(num_cores, sids=2) as (bx, sid):
                q_l1 = T.alloc_l1((bm, d_dim), tl_dtype)
                k_l1 = T.alloc_l1((bn, d_dim), tl_dtype)
                v_l1 = T.alloc_l1((d_dim, bn), tl_dtype)
                p_l1 = T.alloc_l1((bm, bn), tl_dtype)
                s_l0c = T.alloc_l0c((bm, bn), "float32")
                o_l0c = T.alloc_l0c((bm, d_dim), "float32")

                q_ub = T.alloc_shared((rows_per_sid, d_dim), tl_dtype)
                s_ub = T.alloc_shared((rows_per_sid, bn), "float32")
                p_ub = T.alloc_shared((rows_per_sid, bn), tl_dtype)
                o_tmp_ub = T.alloc_shared((rows_per_sid, d_dim), "float32")
                o_ub = T.alloc_shared((rows_per_sid, d_dim), "float32")
                out_ub = T.alloc_shared((rows_per_sid, d_dim), tl_dtype)
                m_ub = T.alloc_shared((rows_per_sid, 8), "float32")
                l_ub = T.alloc_shared((rows_per_sid, 8), "float32")
                alpha_ub = T.alloc_shared((rows_per_sid, 8), "float32")
                limit_ub = T.alloc_shared((rows_per_sid, 8), "int32")

                T.annotate_buffer_versions(
                    {
                        q_l1: 1,
                        k_l1: stages,
                        v_l1: stages,
                        p_l1: 1,
                        s_l0c: 1,
                        o_l0c: 1,
                        q_ub: 1,
                        s_ub: 1,
                        p_ub: 1,
                        o_tmp_ub: 1,
                        o_ub: 1,
                        out_ub: 1,
                        m_ub: 1,
                        l_ub: 1,
                        alpha_ub: 1,
                        limit_ub: 1,
                    }
                )
                if bm >= 96 or bn >= 128 or schedule_mode == "tuned":
                    T.annotate_unlimit_memory("shared")

                q_rows = s_dim * g_ratio
                m_tiles_per_group = (q_rows + bm - 1) // bm
                total_groups = b_dim * n_kv
                total_m_tiles = total_groups * m_tiles_per_group
                base_tiles = total_m_tiles // num_cores
                rem_tiles = total_m_tiles % num_cores
                t_start = bx * base_tiles + T.min(bx, rem_tiles)
                t_count = base_tiles + T.if_then_else(bx < rem_tiles, 1, 0)

                for local_t in T.serial(t_count):
                    tile_id = t_start + local_t
                    group_id = tile_id // m_tiles_per_group
                    mt = tile_id % m_tiles_per_group
                    b_idx = group_id // n_kv
                    kv_h = group_id % n_kv
                    q_base = mt * bm
                    sid_row0 = q_base + sid * rows_per_sid
                    valid_rows = T.max(0, T.min(rows_per_sid, q_rows - sid_row0))

                    if valid_rows > 0:
                        with T.SimdVF():
                            m32 = T.simd.pset(32, "PAT_ALL")
                            m16 = T.simd.pset(16, "PAT_ALL")
                            v_neg = T.simd.vdup(T.float32(-1e38), "float32", mask=m32)
                            v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                            v_zero16 = T.simd.vcvt(v_zero, tl_dtype, mask=m16, part=0)
                            for r in T.serial(valid_rows):
                                T.simd.vsts(m_ub[r, 0], v_neg, mask=m32, dist="ONEPT_B32")
                                T.simd.vsts(l_ub[r, 0], v_zero, mask=m32, dist="ONEPT_B32")
                                for c in T.serial(d_chunks):
                                    T.simd.vsts(o_ub[r, c * 64], v_zero, mask=m32, dist="NORM_B32")
                            for r_pad in T.serial(rows_per_sid - valid_rows):
                                rr = valid_rows + r_pad
                                for c in T.serial(d_chunks):
                                    T.simd.vsts(q_ub[rr, c * 64], v_zero16, mask=m32, dist="PK_B32", extent=64)

                        if is_causal:
                            for r in T.serial(valid_rows):
                                q_row = sid_row0 + r
                                s_idx = q_row // g_ratio
                                limit_ub[r, 0] = T.min(s_kv, s_idx + s_kv - s_dim + 1)
                        else:
                            for r in T.serial(valid_rows):
                                limit_ub[r, 0] = s_kv

                        if use_group_dma:
                            s_start = sid_row0 // g_ratio
                            n_toks = valid_rows // g_ratio
                            for tok in T.serial(n_toks):
                                T.copy(
                                    Q[b_idx, s_start + tok, kv_h * g_ratio : kv_h * g_ratio + g_ratio, :d_dim],
                                    q_ub[tok * g_ratio : tok * g_ratio + g_ratio, :d_dim],
                                )
                        else:
                            for r in T.serial(valid_rows):
                                q_row = sid_row0 + r
                                s_i = q_row // g_ratio
                                h_loc = q_row % g_ratio
                                q_h = kv_h * g_ratio + h_loc
                                T.copy(Q[b_idx, s_i, q_h, :d_dim], q_ub[r, :d_dim])
                    T.dual_copy(q_ub, q_l1)

                    max_kv_blocks = (s_kv + bn - 1) // bn
                    if is_causal:
                        last_valid = T.min(q_rows - 1, q_base + bm - 1)
                        max_s = last_valid // g_ratio
                        kv_limit = T.max(1, T.min(max_kv_blocks, (max_s + s_kv - s_dim) // bn + 1))
                    else:
                        kv_limit = max_kv_blocks

                    for kv in T.Pipelined(kv_limit, num_stages=stages, annotations=pipe_ann):
                        T.copy(K[b_idx, kv * bn : kv * bn + bn, kv_h, :d_dim], k_l1)
                        T.copy(V[b_idx, kv * bn : kv * bn + bn, kv_h, :d_dim], v_l1, transpose=True)
                        T.gemm(q_l1, k_l1, s_l0c, transpose_B=True, clear_accum=True)
                        T.dual_copy(s_l0c, s_ub)

                        if valid_rows > 0:
                            with T.SimdVF():
                                m32 = T.simd.pset(32, "PAT_ALL")
                                m16 = T.simd.pset(16, "PAT_ALL")
                                v_neg_inf = T.simd.vdup(T.float32(-1e38), "float32", mask=m32)
                                v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                                col_idx0 = T.simd.vci(kv * bn, "int32")
                                col_idx1 = T.simd.vci(kv * bn + 64, "int32")
                                for r in T.serial(valid_rows):
                                    limit_col = limit_ub[r, 0]
                                    valid_mask0 = T.simd.vcmps(col_idx0, limit_col, mask=m32, op="lt")
                                    s_vec0 = T.simd.vmuls(T.simd.vld(s_ub[r, 0], dist="NORM_B32"), scale_val, mask=m32)
                                    s_masked0 = T.simd.vsel(s_vec0, v_neg_inf, valid_mask0)
                                    m_tile = T.simd.vdupv(T.simd.vcmax(s_masked0, mask=m32), mask=m32, pos="POS_LOWEST")
                                    if bn == 128:
                                        valid_mask1 = T.simd.vcmps(col_idx1, limit_col, mask=m32, op="lt")
                                        s_vec1 = T.simd.vmuls(
                                            T.simd.vld(s_ub[r, 64], dist="NORM_B32"), scale_val, mask=m32
                                        )
                                        s_masked1 = T.simd.vsel(s_vec1, v_neg_inf, valid_mask1)
                                        m_tile1 = T.simd.vdupv(
                                            T.simd.vcmax(s_masked1, mask=m32), mask=m32, pos="POS_LOWEST"
                                        )
                                        m_tile = T.simd.vmax(m_tile, m_tile1, mask=m32)
                                    m_old = T.simd.vld(m_ub[r, 0], dist="BRC_B32")
                                    l_old = T.simd.vld(l_ub[r, 0], dist="BRC_B32")
                                    m_new = T.simd.vmax(m_old, m_tile, mask=m32)
                                    m_safe = T.simd.vmaxs(m_new, T.float32(-1e30), mask=m32)
                                    old_valid = T.simd.vcmps(m_old, T.float32(-1e30), mask=m32, op="gt")
                                    alpha_v = T.simd.vsel(
                                        T.simd.vexp(T.simd.vsub(m_old, m_safe, mask=m32), mask=m32),
                                        v_zero,
                                        old_valid,
                                    )
                                    new_valid = T.simd.vcmps(m_new, T.float32(-1e30), mask=m32, op="gt")
                                    p_mask0 = T.simd.pand(valid_mask0, new_valid, m32)
                                    p_f32_0 = T.simd.vsel(
                                        T.simd.vexp(T.simd.vsub(s_masked0, m_safe, mask=m32), mask=m32),
                                        v_zero,
                                        p_mask0,
                                    )
                                    row_sum = T.simd.vdupv(T.simd.vcadd(p_f32_0, mask=m32), mask=m32, pos="POS_LOWEST")
                                    p_16_0 = T.simd.vcvt(p_f32_0, tl_dtype, mask=m16, part=0)
                                    T.simd.vsts(p_ub[r, 0], p_16_0, mask=m32, dist="PK_B32", extent=64)
                                    if bn == 128:
                                        valid_mask1b = T.simd.vcmps(col_idx1, limit_col, mask=m32, op="lt")
                                        s_vec1b = T.simd.vmuls(
                                            T.simd.vld(s_ub[r, 64], dist="NORM_B32"), scale_val, mask=m32
                                        )
                                        s_masked1b = T.simd.vsel(s_vec1b, v_neg_inf, valid_mask1b)
                                        p_mask1 = T.simd.pand(valid_mask1b, new_valid, m32)
                                        p_f32_1 = T.simd.vsel(
                                            T.simd.vexp(T.simd.vsub(s_masked1b, m_safe, mask=m32), mask=m32),
                                            v_zero,
                                            p_mask1,
                                        )
                                        sum1 = T.simd.vdupv(
                                            T.simd.vcadd(p_f32_1, mask=m32), mask=m32, pos="POS_LOWEST"
                                        )
                                        row_sum = T.simd.vadd(row_sum, sum1, mask=m32)
                                        p_16_1 = T.simd.vcvt(p_f32_1, tl_dtype, mask=m16, part=0)
                                        T.simd.vsts(p_ub[r, 64], p_16_1, mask=m32, dist="PK_B32", extent=64)
                                    l_new = T.simd.vadd(T.simd.vmul(l_old, alpha_v, mask=m32), row_sum, mask=m32)
                                    T.simd.vsts(m_ub[r, 0], m_new, mask=m32, dist="ONEPT_B32")
                                    T.simd.vsts(l_ub[r, 0], l_new, mask=m32, dist="ONEPT_B32")
                                    T.simd.vsts(alpha_ub[r, 0], alpha_v, mask=m32, dist="ONEPT_B32")

                        T.dual_copy(p_ub, p_l1)
                        T.gemm(p_l1, v_l1, o_l0c, transpose_B=True, clear_accum=True)
                        T.dual_copy(o_l0c, o_tmp_ub)

                        if valid_rows > 0:
                            with T.SimdVF():
                                m32 = T.simd.pset(32, "PAT_ALL")
                                for r in T.serial(valid_rows):
                                    alpha_v = T.simd.vld(alpha_ub[r, 0], dist="BRC_B32")
                                    for c in T.serial(d_chunks):
                                        off = c * 64
                                        o_new = T.simd.vadd(
                                            T.simd.vmul(T.simd.vld(o_ub[r, off], dist="NORM_B32"), alpha_v, mask=m32),
                                            T.simd.vld(o_tmp_ub[r, off], dist="NORM_B32"),
                                            mask=m32,
                                        )
                                        T.simd.vsts(o_ub[r, off], o_new, mask=m32, dist="NORM_B32")

                    if valid_rows > 0:
                        with T.SimdVF():
                            m32 = T.simd.pset(32, "PAT_ALL")
                            m16 = T.simd.pset(16, "PAT_ALL")
                            v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
                            v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                            for r in T.serial(valid_rows):
                                l_vec = T.simd.vld(l_ub[r, 0], dist="BRC_B32")
                                l_pos = T.simd.vcmps(l_vec, T.float32(0.0), mask=m32, op="gt")
                                l_safe = T.simd.vsel(l_vec, v_one, l_pos)
                                inv_l = T.simd.vsel(T.simd.vdiv(v_one, l_safe, mask=m32), v_zero, l_pos)
                                for c in T.serial(d_chunks):
                                    off = c * 64
                                    o_f = T.simd.vmul(T.simd.vld(o_ub[r, off], dist="NORM_B32"), inv_l, mask=m32)
                                    o_16 = T.simd.vcvt(o_f, tl_dtype, mask=m16, part=0)
                                    T.simd.vsts(out_ub[r, off], o_16, mask=m32, dist="PK_B32", extent=64)

                        if use_group_dma:
                            s_start = sid_row0 // g_ratio
                            n_toks = valid_rows // g_ratio
                            for tok in T.serial(n_toks):
                                T.copy(
                                    out_ub[tok * g_ratio : tok * g_ratio + g_ratio, :d_dim],
                                    Out[b_idx, s_start + tok, kv_h * g_ratio : kv_h * g_ratio + g_ratio, :d_dim],
                                )
                        else:
                            for r in T.serial(valid_rows):
                                q_row = sid_row0 + r
                                s_i = q_row // g_ratio
                                h_loc = q_row % g_ratio
                                q_h = kv_h * g_ratio + h_loc
                                T.copy(out_ub[r, :d_dim], Out[b_idx, s_i, q_h, :d_dim])

    return main


def _get_kernel(
    s_dim: int,
    n_q: int,
    n_kv: int,
    d_dim: int,
    tl_dtype: str,
    is_causal: bool,
    schedule_mode: str,
):
    cfg = _select_gqa_config(
        s_dim=s_dim,
        n_q=n_q,
        n_kv=n_kv,
        d_dim=d_dim,
        tl_dtype=tl_dtype,
        is_causal=is_causal,
        schedule_mode=schedule_mode,
    )
    key = (n_q, n_kv, d_dim, tl_dtype, bool(is_causal), int(cfg["bm"]), int(cfg["bn"]), int(cfg["l1_stages"]), schedule_mode)
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        with _ascend_target_ctx():
            prog = _build_gqa_prim_func(
                n_q=n_q,
                n_kv=n_kv,
                d_dim=d_dim,
                tl_dtype=tl_dtype,
                is_causal=is_causal,
                bm=int(cfg["bm"]),
                bn=int(cfg["bn"]),
                l1_stages=int(cfg["l1_stages"]),
                schedule_mode=schedule_mode,
            )
        compiled = tilelang.compile(
            prog,
            target="ascend",
            out_idx=[3],
            pass_configs=cfg.get("pass_configs") or None,
            compile_flags=cfg.get("compile_flags") or None,
        )
        _KERNEL_CACHE[key] = compiled
    return compiled


def gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaleValue: float = -1.0,
    is_causal: bool = False,
    *,
    schedule_mode: str = "tuned",
) -> torch.Tensor:
    """Compute Grouped Query Attention matching cannbot-ops level4/gqa proto.yaml."""
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported dtype {query.dtype}")
    b_dim, s_dim, n_q, d_dim = query.shape
    s_kv = key.shape[1]
    n_kv = key.shape[2]
    eff_scale = (1.0 / math.sqrt(d_dim)) if scaleValue <= 0 else float(scaleValue)

    if query.device.type != "npu":
        g_ratio = n_q // n_kv
        q = (
            query.float()
            .reshape(b_dim, s_dim, n_kv, g_ratio, d_dim)
            .permute(0, 2, 3, 1, 4)
            .reshape(b_dim, n_kv, g_ratio * s_dim, d_dim)
        )
        k = key.float().permute(0, 2, 1, 3)
        v = value.float().permute(0, 2, 1, 3)
        scores = (torch.matmul(q, k.transpose(-2, -1)) * eff_scale).reshape(b_dim, n_kv, g_ratio, s_dim, s_kv)
        if is_causal:
            qi = torch.arange(s_dim, device=scores.device).unsqueeze(-1)
            kj = torch.arange(s_kv, device=scores.device).unsqueeze(0)
            scores = scores.masked_fill(kj > (qi + (s_kv - s_dim)), float("-inf"))
        scores_max = scores.max(dim=-1, keepdim=True).values
        all_masked = torch.isinf(scores_max) & (scores_max < 0)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(all_masked, torch.zeros_like(weights), weights)
        out = torch.matmul(weights.reshape(b_dim, n_kv, g_ratio * s_dim, s_kv), v)
        return (
            out.reshape(b_dim, n_kv, g_ratio, s_dim, d_dim)
            .permute(0, 3, 1, 2, 4)
            .reshape(b_dim, s_dim, n_q, d_dim)
            .to(query.dtype)
        )

    tl_dtype = "float16" if query.dtype == torch.float16 else "bfloat16"
    compiled = _get_kernel(s_dim, n_q, n_kv, d_dim, tl_dtype, bool(is_causal), schedule_mode)
    return compiled(query.contiguous(), key.contiguous(), value.contiguous(), float(eff_scale))


kernel = gqa

__all__ = ["_build_gqa_prim_func", "_select_gqa_config", "gqa", "kernel"]
