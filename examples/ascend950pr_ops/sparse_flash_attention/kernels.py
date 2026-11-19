"""Ascend 950PR MixedKernel SparseFlashAttention across BSND/BNSD, GQA/MLA, and causal regimes."""

import contextlib
import tilelang
import tilelang.ascend.language as T
import torch

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


def _select_sparse_flash_attention_config(
    g_ratio: int = 4,
    dk: int = 128,
    dv: int = 128,
    *,
    dtype: str = "float16",
    input_layout: str = "BSND",
    is_causal: bool = False,
    b_dim: int = 1,
    s1_dim: int = 1,
    top_k: int = 512,
    schedule_mode: str = "tuned",
) -> dict[str, object]:
    g_val = max(1, int(g_ratio))
    dk_val = max(16, int(dk))
    dv_val = max(16, int(dv))
    bm = 64 if (g_val >= 64 and dv_val <= 256) else 32
    bn = 128 if (schedule_mode == "tuned" and dk_val <= 256 and dv_val <= 256 and int(top_k) >= 128) else 64
    use_pipelined = False
    return {
        "g_ratio": g_val,
        "dk": dk_val,
        "dv": dv_val,
        "dtype": str(dtype),
        "input_layout": str(input_layout).upper(),
        "is_causal": bool(is_causal),
        "bm": bm,
        "bn": bn,
        "num_cores": 28,
        "use_pipelined": use_pipelined,
    }


def _build_sparse_flash_attention_prim_func(
    g_ratio: int = 4,
    dk: int = 128,
    dv: int = 128,
    dtype: str = "float16",
    input_layout: str = "BSND",
    is_causal: bool = False,
    bm: int = 32,
    bn: int = 64,
    num_cores: int = 28,
    use_pipelined: bool = False,
    schedule_mode: str = "tuned",
    **_extra_kwargs,
):
    g_val = int(g_ratio)
    dk_val = int(dk)
    dv_val = int(dv)
    bm = int(bm)
    bn = int(bn)
    half_bn = bn // 2
    rows_per_sid = bm // 2
    num_cores = max(1, int(num_cores))
    tl_dtype = str(dtype)
    layout = str(input_layout).upper()
    causal_flag = bool(is_causal)

    d0 = 256 if dv_val > 256 else dv_val
    d1 = dv_val - d0
    d0_chunks = d0 // 64
    d1_chunks = d1 // 64
    dk_chunks = dk_val // 64
    dv_chunks = dv_val // 64
    bn_chunks = bn // 64
    unroll_j = 8 if (half_bn % 8) == 0 else (4 if (half_bn % 4) == 0 else 1)
    kt = 128 if (dk_val % 128) == 0 else 64
    k_tiles = dk_val // kt
    stages = 2 if (use_pipelined and schedule_mode == "tuned") else 1

    q_total_rows = T.dynamic("q_total_rows")
    k_total_rows = T.dynamic("k_total_rows")
    si_total_groups = T.dynamic("si_total_groups")
    top_k_dyn = T.dynamic("top_k_dyn")

    with _ascend_target_ctx():

        @T.prim_func
        def main(
            Q: T.Tensor((q_total_rows, dk_val), tl_dtype),
            K: T.Tensor((k_total_rows, dk_val), tl_dtype),
            SI: T.Tensor((si_total_groups, top_k_dyn), "int32"),
            Out: T.Tensor((q_total_rows, dv_val), tl_dtype),
            scale_val: T.float32,
            b_dim: T.int32,
            s1_dim: T.int32,
            s2_dim: T.int32,
            n_kv: T.int32,
            top_k: T.int32,
            is_causal_rt: T.int32,
        ):
            with T.MixedKernel(num_cores, sids=2) as (bx, sid):
                q_l1 = T.alloc_l1((bm, dk_val), tl_dtype)
                k_l1 = T.alloc_l1((bn, dk_val), tl_dtype)
                p_l1 = T.alloc_l1((bm, bn), tl_dtype)
                qk_a_l0 = T.alloc_l0a((bm, kt), tl_dtype)
                qk_b_l0 = T.alloc_l0b((bn, kt), tl_dtype)
                pv_a_l0 = T.alloc_l0a((bm, bn), tl_dtype)
                pv_b_l0 = T.alloc_l0b((d0, bn), tl_dtype)

                s_l0c = T.alloc_l0c((bm, bn), "float32")
                o0_l0c = T.alloc_l0c((bm, d0), "float32")
                o1_l0c = T.alloc_l0c((bm, d1 if d1 > 0 else 16), "float32")

                q_ub = T.alloc_shared((rows_per_sid, dk_val), tl_dtype)
                K_ub = T.alloc_shared((half_bn, dk_val), tl_dtype)
                si_ub = T.alloc_shared((bn,), "int32")
                S_ub = T.alloc_shared((rows_per_sid, bn), "float32")
                P_ub = T.alloc_shared((rows_per_sid, bn), tl_dtype)
                O_tmp_ub = T.alloc_shared((rows_per_sid, d0), "float32")
                O_ub = T.alloc_shared((rows_per_sid, dv_val), "float32")
                out_ub = T.alloc_shared((rows_per_sid, dv_val), tl_dtype)
                m_ub = T.alloc_shared((rows_per_sid, 8), "float32")
                l_ub = T.alloc_shared((rows_per_sid, 8), "float32")
                alpha_ub = T.alloc_shared((rows_per_sid, 8), "float32")

                pinned = {
                    q_l1: 1,
                    p_l1: 1,
                    s_l0c: 1,
                    o0_l0c: 1,
                    q_ub: 1,
                    si_ub: 1,
                    S_ub: 1,
                    P_ub: 1,
                    O_tmp_ub: 1,
                    O_ub: 1,
                    out_ub: 1,
                    m_ub: 1,
                    l_ub: 1,
                    alpha_ub: 1,
                }
                if d1 > 0:
                    pinned[o1_l0c] = 1
                if stages > 1:
                    pinned[K_ub] = 2
                    pinned[k_l1] = 2
                    pinned[si_ub] = 2
                else:
                    pinned[K_ub] = 1
                    pinned[k_l1] = 1
                    pinned[si_ub] = 1
                T.annotate_buffer_versions(pinned)
                if schedule_mode == "tuned":
                    T.annotate_unlimit_memory("shared")

                m_tiles_per_group = (g_val + bm - 1) // bm
                num_groups = b_dim * s1_dim * n_kv
                total_tiles = num_groups * m_tiles_per_group
                base_tiles = total_tiles // num_cores
                rem_tiles = total_tiles % num_cores
                t_start = bx * base_tiles + T.min(bx, rem_tiles)
                t_count = base_tiles + T.if_then_else(bx < rem_tiles, 1, 0)
                num_kv_blocks = (top_k + bn - 1) // bn

                for local_t in T.serial(t_count):
                    tile_id = t_start + local_t
                    group_id = tile_id // m_tiles_per_group
                    mt = tile_id % m_tiles_per_group
                    q_base = mt * bm

                    b_idx = group_id // (s1_dim * n_kv)
                    rem_sn = group_id % (s1_dim * n_kv)
                    if layout == "BSND":
                        s1_i = rem_sn // n_kv
                        kv_h = rem_sn % n_kv
                    else:
                        kv_h = rem_sn // s1_dim
                        s1_i = rem_sn % s1_dim

                    # Initialize online-softmax state and tail q_ub/P_ub zero padding.
                    q_start_row = q_base + sid * rows_per_sid
                    q_active = T.max(0, T.min(rows_per_sid, g_val - q_start_row))
                    if layout == "BSND":
                        k_base_row = b_idx * s2_dim * n_kv + kv_h
                        k_row_stride = n_kv
                    else:
                        k_base_row = (b_idx * n_kv + kv_h) * s2_dim
                        k_row_stride = 1
                    with T.SimdVF():
                        m32 = T.simd.pset(32, "PAT_ALL")
                        m16 = T.simd.pset(16, "PAT_ALL")
                        v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                        v_neg_init = T.simd.vdup(T.float32(-1e38), "float32", mask=m32)
                        v_zero16 = T.simd.vcvt(v_zero, tl_dtype, mask=m16, part=0)
                        for r in T.serial(q_active):
                            T.simd.vsts(m_ub[r, 0], v_neg_init, mask=m32, dist="ONEPT_B32")
                            T.simd.vsts(l_ub[r, 0], v_zero, mask=m32, dist="ONEPT_B32")
                            for c in T.serial(dv_chunks):
                                T.simd.vsts(O_ub[r, c * 64], v_zero, mask=m32, dist="NORM_B32")
                        for r_pad in T.serial(rows_per_sid - q_active):
                            rr = q_active + r_pad
                            for c in T.serial(dk_chunks):
                                T.simd.vsts(q_ub[rr, c * 64], v_zero16, mask=m32, dist="PK_B32", extent=64)
                            for c_p in T.serial(bn_chunks):
                                T.simd.vsts(P_ub[rr, c_p * 64], v_zero16, mask=m32, dist="PK_B32", extent=64)

                    # Load Q rows assigned to this AIV half and send to q_l1.
                    if layout == "BSND":
                        q_gm_base = ((b_idx * s1_dim + s1_i) * n_kv + kv_h) * g_val + q_start_row
                        if q_active == rows_per_sid:
                            T.copy(Q[q_gm_base : q_gm_base + rows_per_sid, 0:dk_val], q_ub)
                        elif q_active > 0:
                            T.copy(Q[q_gm_base : q_gm_base + q_active, 0:dk_val], q_ub[:q_active, 0:dk_val])
                    else:
                        for r in T.serial(q_active):
                            q_row = q_start_row + r
                            q_gm_row = ((b_idx * n_kv + kv_h) * g_val + q_row) * s1_dim + s1_i
                            T.copy(Q[q_gm_row : q_gm_row + 1, 0:dk_val], q_ub[r : r + 1, 0:dk_val])

                    T.dual_copy(q_ub, q_l1)

                    max_kv = s1_i + (s2_dim - s1_dim)
                    causal_lim = T.if_then_else(is_causal_rt != 0, max_kv, s2_dim)

                    for jb in (T.Pipelined(num_kv_blocks, num_stages=stages) if stages > 1 else T.serial(num_kv_blocks)):
                        j_base = jb * bn
                        bn_cur = T.min(bn, top_k - j_base)
                        T.copy(SI[group_id, j_base : j_base + bn_cur], si_ub[:bn_cur])

                        if bn_cur == bn:
                            for j_out in T.serial(half_bn // unroll_j):
                                for ju in T.unroll(unroll_j):
                                    j = j_out * unroll_j + ju
                                    raw_idx = si_ub[sid * half_bn + j]
                                    k_gm_row = k_base_row + raw_idx * k_row_stride
                                    T.copy(K[k_gm_row : k_gm_row + 1, 0:dk_val], K_ub[j : j + 1, 0:dk_val])
                        else:
                            for j_out in T.serial(half_bn // unroll_j):
                                for ju in T.unroll(unroll_j):
                                    j = j_out * unroll_j + ju
                                    j_global = j_base + sid * half_bn + j
                                    raw_idx = T.if_then_else(j_global < top_k, si_ub[sid * half_bn + j], 0)
                                    safe_idx = T.min(T.max(raw_idx, 0), s2_dim - 1)
                                    k_gm_row = k_base_row + safe_idx * k_row_stride
                                    T.copy(K[k_gm_row : k_gm_row + 1, 0:dk_val], K_ub[j : j + 1, 0:dk_val])

                        T.dual_copy(K_ub, k_l1)
                        for ki in T.serial(k_tiles):
                            T.copy(q_l1[0:bm, ki * kt : (ki + 1) * kt], qk_a_l0)
                            T.copy(k_l1[0:bn, ki * kt : (ki + 1) * kt], qk_b_l0)
                            T.gemm(qk_a_l0, qk_b_l0, s_l0c, transpose_B=True, clear_accum=(ki == 0))
                        T.dual_copy(s_l0c, S_ub)

                        if q_active > 0:
                            with T.SimdVF():
                                m32 = T.simd.pset(32, "PAT_ALL")
                                m16 = T.simd.pset(16, "PAT_ALL")
                                v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                                v_neg_inf = T.simd.vdup(T.float32(-1e38), "float32", mask=m32)
                                col_base = T.simd.vci(0, "int32")
                                vm0 = T.simd.vcmps(col_base, bn_cur, mask=m32, op="lt")
                                vm0 = T.simd.pand(vm0, T.simd.vcmps(T.simd.vld(si_ub[0], dist="NORM_B32"), causal_lim, mask=m32, op="le"), m32)
                                vm1 = T.simd.vcmps(T.simd.vadds(col_base, 64, mask=m32), bn_cur, mask=m32, op="lt")
                                if bn == 128:
                                    vm1 = T.simd.pand(vm1, T.simd.vcmps(T.simd.vld(si_ub[64], dist="NORM_B32"), causal_lim, mask=m32, op="le"), m32)

                                for r in T.serial(q_active):
                                    s_vec0 = T.simd.vmuls(T.simd.vld(S_ub[r, 0], dist="NORM_B32"), scale_val, mask=m32)
                                    sm0 = T.simd.vsel(s_vec0, v_neg_inf, vm0)
                                    m_tile_v = T.simd.vdupv(T.simd.vcmax(sm0, mask=m32), mask=m32, pos="POS_LOWEST")
                                    if bn == 128:
                                        s_vec1 = T.simd.vmuls(T.simd.vld(S_ub[r, 64], dist="NORM_B32"), scale_val, mask=m32)
                                        sm1 = T.simd.vsel(s_vec1, v_neg_inf, vm1)
                                        m_tile1 = T.simd.vdupv(T.simd.vcmax(sm1, mask=m32), mask=m32, pos="POS_LOWEST")
                                        m_tile_v = T.simd.vmax(m_tile_v, m_tile1, mask=m32)

                                    m_old_v = T.simd.vld(m_ub[r, 0], dist="BRC_B32")
                                    l_old_v = T.simd.vld(l_ub[r, 0], dist="BRC_B32")
                                    m_new_v = T.simd.vmax(m_old_v, m_tile_v, mask=m32)
                                    m_safe_v = T.simd.vmaxs(m_new_v, T.float32(-1e30), mask=m32)
                                    alpha_raw = T.simd.vexp(T.simd.vsub(m_old_v, m_safe_v, mask=m32), mask=m32)
                                    old_valid = T.simd.vcmps(m_old_v, T.float32(-1e30), mask=m32, op="gt")
                                    alpha_v = T.simd.vsel(alpha_raw, v_zero, old_valid)
                                    new_valid = T.simd.vcmps(m_new_v, T.float32(-1e30), mask=m32, op="gt")
                                    p0_raw = T.simd.vexp(T.simd.vsub(sm0, m_safe_v, mask=m32), mask=m32)
                                    p0 = T.simd.vsel(p0_raw, v_zero, T.simd.pand(vm0, new_valid, m32))
                                    row_sum_v = T.simd.vdupv(T.simd.vcadd(p0, mask=m32), mask=m32, pos="POS_LOWEST")
                                    T.simd.vsts(P_ub[r, 0], T.simd.vcvt(p0, tl_dtype, mask=m16, part=0), mask=m32, dist="PK_B32", extent=64)
                                    if bn == 128:
                                        s_vec1b = T.simd.vmuls(T.simd.vld(S_ub[r, 64], dist="NORM_B32"), scale_val, mask=m32)
                                        sm1b = T.simd.vsel(s_vec1b, v_neg_inf, vm1)
                                        p1_raw = T.simd.vexp(T.simd.vsub(sm1b, m_safe_v, mask=m32), mask=m32)
                                        p1 = T.simd.vsel(p1_raw, v_zero, T.simd.pand(vm1, new_valid, m32))
                                        sum1 = T.simd.vdupv(T.simd.vcadd(p1, mask=m32), mask=m32, pos="POS_LOWEST")
                                        row_sum_v = T.simd.vadd(row_sum_v, sum1, mask=m32)
                                        T.simd.vsts(P_ub[r, 64], T.simd.vcvt(p1, tl_dtype, mask=m16, part=0), mask=m32, dist="PK_B32", extent=64)
                                    l_new_v = T.simd.vadd(T.simd.vmul(l_old_v, alpha_v, mask=m32), row_sum_v, mask=m32)
                                    T.simd.vsts(m_ub[r, 0], m_new_v, mask=m32, dist="ONEPT_B32")
                                    T.simd.vsts(l_ub[r, 0], l_new_v, mask=m32, dist="ONEPT_B32")
                                    T.simd.vsts(alpha_ub[r, 0], alpha_v, mask=m32, dist="ONEPT_B32")

                        T.dual_copy(P_ub, p_l1)
                        T.copy(p_l1, pv_a_l0)
                        T.copy(k_l1[0:bn, 0:d0], pv_b_l0, transpose=True)
                        T.gemm(pv_a_l0, pv_b_l0, o0_l0c, transpose_B=True, clear_accum=True)
                        T.dual_copy(o0_l0c, O_tmp_ub)

                        if q_active > 0:
                            with T.SimdVF():
                                m32 = T.simd.pset(32, "PAT_ALL")
                                for r in T.serial(q_active):
                                    alpha_v = T.simd.vld(alpha_ub[r, 0], dist="BRC_B32")
                                    for c in T.serial(d0_chunks):
                                        off = c * 64
                                        o_old = T.simd.vld(O_ub[r, off], dist="NORM_B32")
                                        o_tmp = T.simd.vld(O_tmp_ub[r, off], dist="NORM_B32")
                                        o_new = T.simd.vadd(T.simd.vmul(o_old, alpha_v, mask=m32), o_tmp, mask=m32)
                                        T.simd.vsts(O_ub[r, off], o_new, mask=m32, dist="NORM_B32")

                        if d1 > 0:
                            T.copy(k_l1[0:bn, d0:dv_val], pv_b_l0, transpose=True)
                            T.gemm(pv_a_l0, pv_b_l0, o1_l0c, transpose_B=True, clear_accum=True)
                            T.dual_copy(o1_l0c, O_tmp_ub)
                            if q_active > 0:
                                with T.SimdVF():
                                    m32 = T.simd.pset(32, "PAT_ALL")
                                    for r in T.serial(q_active):
                                        alpha_v = T.simd.vld(alpha_ub[r, 0], dist="BRC_B32")
                                        for c in T.serial(d1_chunks):
                                            off = d0 + c * 64
                                            o_old = T.simd.vld(O_ub[r, off], dist="NORM_B32")
                                            o_tmp = T.simd.vld(O_tmp_ub[r, c * 64], dist="NORM_B32")
                                            o_new = T.simd.vadd(T.simd.vmul(o_old, alpha_v, mask=m32), o_tmp, mask=m32)
                                            T.simd.vsts(O_ub[r, off], o_new, mask=m32, dist="NORM_B32")

                    if q_active > 0:
                        with T.SimdVF():
                            m32 = T.simd.pset(32, "PAT_ALL")
                            m16 = T.simd.pset(16, "PAT_ALL")
                            v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
                            v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                            for r in T.serial(q_active):
                                l_vec = T.simd.vld(l_ub[r, 0], dist="BRC_B32")
                                l_pos = T.simd.vcmps(l_vec, T.float32(0.0), mask=m32, op="gt")
                                l_safe = T.simd.vsel(l_vec, v_one, l_pos)
                                inv_l = T.simd.vsel(T.simd.vdiv(v_one, l_safe, mask=m32), v_zero, l_pos)
                                for c in T.serial(dv_chunks):
                                    off = c * 64
                                    o_f = T.simd.vmul(T.simd.vld(O_ub[r, off], dist="NORM_B32"), inv_l, mask=m32)
                                    o_16 = T.simd.vcvt(o_f, tl_dtype, mask=m16, part=0)
                                    T.simd.vsts(out_ub[r, off], o_16, mask=m32, dist="PK_B32", extent=64)

                    if layout == "BSND":
                        q_start_row = q_base + sid * rows_per_sid
                        q_active = T.min(rows_per_sid, g_val - q_start_row)
                        o_gm_base = ((b_idx * s1_dim + s1_i) * n_kv + kv_h) * g_val + q_start_row
                        if q_active == rows_per_sid:
                            T.copy(out_ub, Out[o_gm_base : o_gm_base + rows_per_sid, 0:dv_val])
                        elif q_active > 0:
                            T.copy(out_ub[:q_active, 0:dv_val], Out[o_gm_base : o_gm_base + q_active, 0:dv_val])
                    else:
                        for r in T.serial(rows_per_sid):
                            q_row = q_base + sid * rows_per_sid + r
                            if q_row < g_val:
                                o_gm_row = ((b_idx * n_kv + kv_h) * g_val + q_row) * s1_dim + s1_i
                                T.copy(out_ub[r : r + 1, 0:dv_val], Out[o_gm_row : o_gm_row + 1, 0:dv_val])

        return main


def _get_kernel(
    g_ratio: int,
    dk: int,
    dv: int,
    dtype: str,
    input_layout: str,
    is_causal: bool,
    b_dim: int,
    s1_dim: int,
    top_k: int,
    schedule_mode: str,
):
    cfg = _select_sparse_flash_attention_config(
        g_ratio=g_ratio,
        dk=dk,
        dv=dv,
        dtype=dtype,
        input_layout=input_layout,
        is_causal=is_causal,
        b_dim=b_dim,
        s1_dim=s1_dim,
        top_k=top_k,
        schedule_mode=schedule_mode,
    )
    key = (
        int(cfg["g_ratio"]),
        int(cfg["dk"]),
        int(cfg["dv"]),
        str(cfg["dtype"]),
        str(cfg["input_layout"]),
        int(cfg["bm"]),
        int(cfg["bn"]),
        int(cfg["num_cores"]),
        bool(cfg["use_pipelined"]),
        schedule_mode,
    )
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        with _ascend_target_ctx():
            prog = _build_sparse_flash_attention_prim_func(
                g_ratio=int(cfg["g_ratio"]),
                dk=int(cfg["dk"]),
                dv=int(cfg["dv"]),
                dtype=str(cfg["dtype"]),
                input_layout=str(cfg["input_layout"]),
                is_causal=bool(cfg["is_causal"]),
                bm=int(cfg["bm"]),
                bn=int(cfg["bn"]),
                num_cores=int(cfg["num_cores"]),
                use_pipelined=bool(cfg["use_pipelined"]),
                schedule_mode=schedule_mode,
            )
        compiled = tilelang.compile(prog, target="ascend", out_idx=[3])
        _KERNEL_CACHE[key] = compiled
    return compiled


def _sparse_flash_attention_cpu_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparseIndices: torch.Tensor,
    scaleValue: float,
    inputLayout: str = "BSND",
    is_causal: bool = False,
) -> torch.Tensor:
    if inputLayout.upper() == "BSND":
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        si = sparseIndices.permute(0, 2, 1, 3)
    else:
        q, k, v, si = query, key, value, sparseIndices

    b_dim, n1, s1, dk = q.shape
    n2 = k.shape[1]
    s2 = k.shape[2]
    g_ratio = n1 // n2
    mask = torch.zeros(b_dim, n2, s1, s2, dtype=torch.bool, device=q.device)
    mask.scatter_(-1, si.long(), True)
    if is_causal:
        s1_idx = torch.arange(s1, device=q.device).unsqueeze(-1)
        s2_idx = torch.arange(s2, device=q.device).unsqueeze(0)
        mask = mask & (s2_idx <= (s1_idx + (s2 - s1)))

    q_g = q.reshape(b_dim, n2, g_ratio, s1, dk)
    scores = torch.einsum("bngsd,bnkd->bngsk", q_g.float(), k.float()) * float(scaleValue)
    scores.masked_fill_(~mask.unsqueeze(2), float("-inf"))
    scores_max = scores.max(dim=-1, keepdim=True).values
    all_masked = torch.isinf(scores_max) & (scores_max < 0)
    scores = torch.exp(scores - torch.where(all_masked, torch.zeros_like(scores_max), scores_max))
    scores_sum = scores.sum(dim=-1, keepdim=True)
    probs = torch.where(all_masked, torch.zeros_like(scores), scores / scores_sum.clamp_min(1e-38))
    out = torch.einsum("bngsk,bnkd->bngsd", probs, v.float()).reshape(b_dim, n1, s1, v.shape[-1]).to(query.dtype)
    return out.permute(0, 2, 1, 3) if inputLayout.upper() == "BSND" else out


def sparse_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparseIndices: torch.Tensor,
    scaleValue: float,
    inputLayout: str = "BSND",
    is_causal: bool = False,
    *,
    schedule_mode: str = "tuned",
) -> torch.Tensor:
    """Sparse FlashAttention across BSND/BNSD, GQA/MLA, Dk!=Dv, and causal masking."""
    if query.device.type != "npu":
        return _sparse_flash_attention_cpu_ref(
            query, key, value, sparseIndices, scaleValue, inputLayout=inputLayout, is_causal=is_causal
        )

    layout = str(inputLayout).upper()
    if layout == "BSND":
        b_dim, s1_dim, n1_dim, dk_val = query.shape
        _, s2_dim, n2_dim, _ = key.shape
        _, _, _, dv_val = value.shape
    else:
        b_dim, n1_dim, s1_dim, dk_val = query.shape
        _, n2_dim, s2_dim, _ = key.shape
        _, _, _, dv_val = value.shape

    g_ratio = n1_dim // n2_dim
    top_k = int(sparseIndices.shape[-1])
    dtype_name = "bfloat16" if query.dtype == torch.bfloat16 else "float16"

    compiled = _get_kernel(
        g_ratio=g_ratio,
        dk=dk_val,
        dv=dv_val,
        dtype=dtype_name,
        input_layout=layout,
        is_causal=bool(is_causal),
        b_dim=b_dim,
        s1_dim=s1_dim,
        top_k=top_k,
        schedule_mode=schedule_mode,
    )
    q_flat = query.contiguous().view(-1, dk_val)
    k_flat = key.contiguous().view(-1, dk_val)
    si_flat = sparseIndices.contiguous().to(torch.int32).view(-1, top_k)
    out_flat = compiled(
        q_flat,
        k_flat,
        si_flat,
        float(scaleValue),
        int(b_dim),
        int(s1_dim),
        int(s2_dim),
        int(n2_dim),
        int(top_k),
        1 if is_causal else 0,
    )
    if layout == "BSND":
        return out_flat.view(b_dim, s1_dim, n1_dim, dv_val)
    return out_flat.view(b_dim, n1_dim, s1_dim, dv_val)


kernel = sparse_flash_attention
