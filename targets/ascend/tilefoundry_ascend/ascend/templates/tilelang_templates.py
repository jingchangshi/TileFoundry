"""TileLang @T.prim_func parameterized templates for canonical Ascend 950 3-axis shells."""

import contextlib
from typing import Any

from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS
from tilefoundry_ascend.ascend.templates.shells import (
    SHELL_AIC_GEMM_FIXPIPE,
    SHELL_AIV_SIMD_STREAM,
    SHELL_AIV_SIMT_ATOMIC,
    SHELL_AIV_STRIPED_VM,
    SHELL_CV_PERSISTENT_STATE,
    SHELL_CV_RAGGED_EPILOGUE,
    SHELL_CV_RING_PIPELINE,
    normalize_schedule_policy,
)


def _get_tl_ascend() -> Any:
    """Import tilelang.ascend.language lazily so package import works without full tilelang."""
    import tilelang.ascend.language as tl_ascend  # noqa: PLC0415

    return tl_ascend


def _ascend_target_ctx():
    try:
        from tilelang import tvm  # noqa: PLC0415
        from tilelang.backend.module import create_backend_context  # noqa: PLC0415

        if tvm.target.Target.current(allow_none=True) is None:
            return create_backend_context("ascend").target
    except Exception:
        pass
    return contextlib.nullcontext()


def _resolve_flags(ub_256kb_unlock: bool) -> tuple[str, ...]:
    return tuple(PURE_SIMDVF_UBUF_FLAGS) if ub_256kb_unlock else ()


def _align64(val: int) -> int:
    return ((max(1, int(val)) + 63) // 64) * 64


def build_pure_regbase_simd_template(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Shell 1 (aiv_only/gm_ub_stream/affine_simd): pure RegBase SIMD VF.

    Unlocks 256 KB UB, packs rows, and unrolls two rows.
    """
    T = _get_tl_ascend()
    p = normalize_schedule_policy(SHELL_AIV_SIMD_STREAM, policy)
    tile_m, tile_n, _ = p["tile_mnk"]
    tl_dtype = p["dtype"] if p["dtype"] in ("float16", "bfloat16", "float32") else "bfloat16"
    dim_size = max(16, min(2048, int(p["dim_size"])))
    row_pack = max(1, min(8, int(p["row_pack_factor"])))
    unroll_rows = max(1, min(4, int(p["multi_row_unroll"])))
    pad_elems = int(p["ub_bank_pad_datablocks"]) * (16 if tl_dtype != "float32" else 8)
    d_alloc = _align64(dim_size) + pad_elems
    d_alloc = _align64(d_alloc)
    full_chunks = dim_size // 64
    rem = dim_size % 64
    num_stages = 2 if p["l1_stages"] >= 2 else 1
    elem_bytes = 4 if tl_dtype == "float32" else 2
    ub_budget = 250000 if p["ub_256kb_unlock"] else 240000
    rows_blk = max(1, min(tile_m, row_pack * unroll_rows, 8))
    while rows_blk > 1 and rows_blk * d_alloc * elem_bytes * (num_stages + 1) > ub_budget:
        rows_blk -= 1
    num_cores = 56

    m_rows = T.dynamic("m_rows")

    with _ascend_target_ctx():

        @T.prim_func
        def pure_regbase_simd_kernel(
            X: T.Tensor((m_rows, dim_size), tl_dtype),
            Y: T.Tensor((m_rows, dim_size), tl_dtype),
            eps_val: T.float32,
        ):
            with T.Kernel(num_cores) as cid:
                x_ub = T.alloc_shared((rows_blk, d_alloc), tl_dtype)
                y_ub = T.alloc_shared((rows_blk, d_alloc), tl_dtype)
                T.annotate_buffer_versions({x_ub: num_stages, y_ub: 1})
                if p["ub_256kb_unlock"]:
                    T.annotate_unlimit_memory("shared")

                base_rows = m_rows // num_cores
                rem_rows = m_rows % num_cores
                m_start = cid * base_rows + T.min(cid, rem_rows)
                m_count = base_rows + T.if_then_else(cid < rem_rows, 1, 0)
                full_blocks = m_count // rows_blk

                for mb in T.serial(full_blocks):
                    m0 = m_start + mb * rows_blk
                    T.copy(X[m0 : m0 + rows_blk, :dim_size], x_ub[:, :dim_size])
                    with T.SimdVF():
                        m32 = T.simd.pset(32, "PAT_ALL")
                        m16 = T.simd.pset(16, "PAT_ALL")
                        v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                        v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
                        for rr in T.serial(rows_blk):
                            sq_acc = T.simd.alloc_local((1,), "float32")
                            sq_acc[0] = v_zero
                            for c in T.serial(full_chunks):
                                off = c * 64
                                if tl_dtype == "float32":
                                    xf = T.simd.vld(x_ub[rr, off], dist="NORM_B32")
                                else:
                                    xf = T.simd.vcvt(
                                        T.simd.vld(x_ub[rr, off], dist="US_B16"),
                                        "float32",
                                        mask=m16,
                                        part=0,
                                    )
                                T.simd.vmula(sq_acc[0], xf, xf, mask=m32)
                            if rem > 0:
                                off = full_chunks * 64
                                m32_t = T.simd.update_mask(rem, width=32)
                                if tl_dtype == "float32":
                                    xf_t = T.simd.vld(x_ub[rr, off], dist="NORM_B32")
                                else:
                                    m16_t = T.simd.update_mask(rem * 2, width=16)
                                    xf_t = T.simd.vcvt(
                                        T.simd.vld(x_ub[rr, off], dist="US_B16"),
                                        "float32",
                                        mask=m16_t,
                                        part=0,
                                    )
                                T.simd.vmula(sq_acc[0], xf_t, xf_t, mask=m32_t, mode="MODE_MERGING")
                            sum_sq = T.simd.vdupv(
                                T.simd.vcadd(sq_acc[0], mask=m32), mask=m32, pos="POS_LOWEST"
                            )
                            inv_rms = T.simd.vdiv(
                                v_one,
                                T.simd.vsqrt(
                                    T.simd.vadds(
                                        T.simd.vmuls(sum_sq, T.float32(1.0 / float(dim_size)), mask=m32),
                                        eps_val,
                                        mask=m32,
                                    ),
                                    mask=m32,
                                ),
                                mask=m32,
                            )
                            for c in T.serial(full_chunks):
                                off = c * 64
                                if tl_dtype == "float32":
                                    xf = T.simd.vld(x_ub[rr, off], dist="NORM_B32")
                                    yf = T.simd.vmul(xf, inv_rms, mask=m32)
                                    T.simd.vsts(y_ub[rr, off], yf, mask=m32, dist="NORM_B32")
                                else:
                                    xf = T.simd.vcvt(
                                        T.simd.vld(x_ub[rr, off], dist="US_B16"),
                                        "float32",
                                        mask=m16,
                                        part=0,
                                    )
                                    yf = T.simd.vmul(xf, inv_rms, mask=m32)
                                    y16 = T.simd.vcvt(yf, tl_dtype, mask=m16, part=0)
                                    T.simd.vsts(y_ub[rr, off], y16, mask=m32, dist="PK_B32", extent=64)
                            if rem > 0:
                                off = full_chunks * 64
                                m32_t = T.simd.update_mask(rem, width=32)
                                if tl_dtype == "float32":
                                    xf_t = T.simd.vld(x_ub[rr, off], dist="NORM_B32")
                                    yf_t = T.simd.vmul(xf_t, inv_rms, mask=m32_t)
                                    T.simd.vsts(y_ub[rr, off], yf_t, mask=m32_t, dist="NORM_B32")
                                else:
                                    m16_t = T.simd.update_mask(rem * 2, width=16)
                                    xf_t = T.simd.vcvt(
                                        T.simd.vld(x_ub[rr, off], dist="US_B16"),
                                        "float32",
                                        mask=m16_t,
                                        part=0,
                                    )
                                    yf_t = T.simd.vmul(xf_t, inv_rms, mask=m32_t)
                                    y16_t = T.simd.vcvt(yf_t, tl_dtype, mask=m16_t, part=0)
                                    T.simd.vsts(y_ub[rr, off], y16_t, mask=m32_t, dist="PK_B32", extent=64)
                    T.copy(y_ub[:, :dim_size], Y[m0 : m0 + rows_blk, :dim_size])
                    T.ascend_pipe_barrier("PIPE_ALL")

    return {
        "prim_func": pure_regbase_simd_kernel,
        "pass_configs": {},
        "compile_flags": _resolve_flags(p["ub_256kb_unlock"]),
        "metadata": {"topology_axes": SHELL_AIV_SIMD_STREAM.to_dict(), "rows_blk": rows_blk, "d_alloc": d_alloc},
    }


def build_simt_and_hybrid_ub_template(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Shell 2 (aiv_only/ub_random_atomic/irregular_simt): hybrid SIMT/SIMD kernel.

    Shared slots carry eight datablocks of padding to avoid UB bank conflicts
    across SIMT warps.
    """
    T = _get_tl_ascend()
    p = normalize_schedule_policy(SHELL_AIV_SIMT_ATOMIC, policy)
    tile_m, tile_n, _ = p["tile_mnk"]
    simt_threads = max(32, min(1024, int(p["simt_threads"] or 512)))
    pad_slots = max(8, int(p["ub_bank_pad_datablocks"]))
    slot_w = max(64, _align64(tile_n))
    rows_blk = max(1, min(16, tile_m))
    num_cores = 56
    n_rows = T.dynamic("n_rows")

    with _ascend_target_ctx():

        @T.prim_func
        def simt_and_hybrid_ub_kernel(
            Src: T.Tensor((n_rows, slot_w), "float32"),
            Idx: T.Tensor((n_rows,), "int32"),
            Dst: T.Tensor((n_rows, slot_w), "float32"),
        ):
            with T.Kernel(num_cores) as cid:
                ub_in = T.alloc_shared((rows_blk, slot_w + pad_slots), "float32")
                ub_out = T.alloc_shared((rows_blk, slot_w + pad_slots), "float32")
                idx_ub = T.alloc_shared((rows_blk,), "int32")
                T.annotate_buffer_versions({ub_in: 1, ub_out: 1, idx_ub: 1})

                base_rows = n_rows // num_cores
                rem_rows = n_rows % num_cores
                r_start = cid * base_rows + T.min(cid, rem_rows)
                r_count = base_rows + T.if_then_else(cid < rem_rows, 1, 0)
                blocks = r_count // rows_blk

                for b in T.serial(blocks):
                    r0 = r_start + b * rows_blk
                    T.copy(Src[r0 : r0 + rows_blk, :slot_w], ub_in[:, :slot_w])
                    T.copy(Idx[r0 : r0 + rows_blk], idx_ub[:rows_blk])
                    with T.SimtVF(threads=simt_threads):
                        for tid in T.Parallel(simt_threads):
                            if tid < slot_w:
                                for r in T.serial(rows_blk):
                                    sel_r = idx_ub[r] % rows_blk
                                    ub_out[r, tid] = ub_in[sel_r, tid]
                    T.copy(ub_out[:, :slot_w], Dst[r0 : r0 + rows_blk, :slot_w])
                    T.ascend_pipe_barrier("PIPE_ALL")

    return {
        "prim_func": simt_and_hybrid_ub_kernel,
        "pass_configs": {"tl.disable_data_race_check": True},
        "compile_flags": (),
        "metadata": {"topology_axes": SHELL_AIV_SIMT_ATOMIC.to_dict(), "simt_threads": simt_threads, "pad_slots": pad_slots},
    }


def build_dense_and_quant_gemm_template(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Shell 3 (aic_only/gm_l1_l0_fixpipe/affine_simd): dense/quantized GEMM.

    Applies SWAT swizzling and Half-L1 isolation.
    """
    T = _get_tl_ascend()
    p = normalize_schedule_policy(SHELL_AIC_GEMM_FIXPIPE, policy)
    bm, bn, bk = p["tile_mnk"]
    bm = max(32, min(128, _align64(bm) if bm >= 64 else 32))
    bn = max(64, min(128, _align64(bn)))
    bk = max(64, min(128, _align64(bk)))
    l1_stages = max(1, min(2, int(p["l1_stages"])))
    swizzle_win = max(1, int(p["swizzle_window"]))
    tl_dtype = p["dtype"] if p["dtype"] in ("float16", "bfloat16") else "float16"
    use_unit_flag = 2 if (p["sync_mode"] in ("unit_flag", "unit_flag_2_3") and l1_stages == 1) else None
    num_cores = 28

    m_dim = T.dynamic("m_dim")
    n_dim = T.dynamic("n_dim")
    k_dim = T.dynamic("k_dim")

    with _ascend_target_ctx():

        @T.prim_func
        def dense_and_quant_gemm_kernel(
            A: T.Tensor((m_dim, k_dim), tl_dtype),
            B: T.Tensor((n_dim, k_dim), tl_dtype),
            C: T.Tensor((m_dim, n_dim), "float32"),
        ):
            with T.MixedKernel(num_cores, sids=2) as (bx, sid):
                a_l1 = T.alloc_l1((bm, bk), tl_dtype)
                b_l1 = T.alloc_l1((bn, bk), tl_dtype)
                c_l0c = T.alloc_l0c((bm, bn), "float32")
                c_ub = T.alloc_shared((bm // 2, bn), "float32")
                T.annotate_buffer_versions({a_l1: l1_stages, b_l1: l1_stages, c_l0c: 1, c_ub: 1})

                m_tiles = (m_dim + bm - 1) // bm
                n_tiles = (n_dim + bn - 1) // bn
                total_tiles = m_tiles * n_tiles
                k_tiles = (k_dim + bk - 1) // bk
                base_t = total_tiles // num_cores
                rem_t = total_tiles % num_cores
                t_start = bx * base_t + T.min(bx, rem_t)
                t_count = base_t + T.if_then_else(bx < rem_t, 1, 0)

                for lt in T.serial(t_count):
                    tid = t_start + lt
                    win_size = swizzle_win * n_tiles
                    win_id = tid // win_size
                    in_win = tid % win_size
                    mt = win_id * swizzle_win + (in_win % swizzle_win)
                    nt_raw = in_win // swizzle_win
                    nt = T.if_then_else((win_id % 2) == 1, n_tiles - 1 - nt_raw, nt_raw)
                    if mt < m_tiles and nt < n_tiles:
                        for kt in T.Pipelined(k_tiles, num_stages=l1_stages):
                            T.copy(A[mt * bm : mt * bm + bm, kt * bk : kt * bk + bk], a_l1)
                            T.copy(B[nt * bn : nt * bn + bn, kt * bk : kt * bk + bk], b_l1)
                            T.gemm(
                                a_l1,
                                b_l1,
                                c_l0c,
                                transpose_B=True,
                                clear_accum=(kt == 0),
                                unit_flag_ctrl=use_unit_flag,
                            )
                        T.dual_copy(c_l0c, c_ub)
                        T.copy(
                            c_ub,
                            C[
                                mt * bm + sid * (bm // 2) : mt * bm + (sid + 1) * (bm // 2),
                                nt * bn : nt * bn + bn,
                            ],
                        )

    return {
        "prim_func": dense_and_quant_gemm_kernel,
        "pass_configs": {},
        "compile_flags": _resolve_flags(p["ub_256kb_unlock"]),
        "metadata": {
            "topology_axes": SHELL_AIC_GEMM_FIXPIPE.to_dict(),
            "tile_mnk": (bm, bn, bk),
            "swizzle_window": swizzle_win,
            "w4_mode": p["w4_mode"],
        },
    }


def build_grouped_gemm_epilogue_template(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Shell 4 (coupled_1c2v/l0c_ub_epilogue/ragged_group): grouped GEMM.

    Moves results directly from L0C to UB with SPLIT_M.
    """
    T = _get_tl_ascend()
    p = normalize_schedule_policy(SHELL_CV_RAGGED_EPILOGUE, policy)
    bm, bn, bk = p["tile_mnk"]
    bm = 64 if bm < 96 else 128
    bn = max(64, min(128, _align64(bn)))
    bk = max(64, min(128, _align64(bk)))
    rows_per_sid = bm // 2
    half_n = bn // 2
    l1_stages = max(1, min(2, int(p["l1_stages"])))
    tl_dtype = p["dtype"] if p["dtype"] in ("float16", "bfloat16") else "bfloat16"
    num_cores = 28

    total_m = T.dynamic("total_m")
    k_dim = T.dynamic("k_dim")
    num_groups = T.dynamic("num_groups")

    with _ascend_target_ctx():

        @T.prim_func
        def grouped_gemm_epilogue_kernel(
            A: T.Tensor((total_m, k_dim), tl_dtype),
            B: T.Tensor((num_groups, bn, k_dim), tl_dtype),
            GroupEnds: T.Tensor((num_groups,), "int32"),
            Out: T.Tensor((total_m, half_n), tl_dtype),
        ):
            with T.MixedKernel(num_cores, sids=2) as (bx, sid):
                a_l1 = T.alloc_l1((bm, bk), tl_dtype)
                b_l1 = T.alloc_l1((bn, bk), tl_dtype)
                c_l0c = T.alloc_l0c((bm, bn), "float32")
                c_ub = T.alloc_shared((rows_per_sid, bn), "float32")
                out_ub = T.alloc_shared((rows_per_sid, half_n), tl_dtype)
                T.annotate_buffer_versions({a_l1: l1_stages, b_l1: l1_stages, c_l0c: 1, c_ub: 1, out_ub: 1})

                k_tiles = (k_dim + bk - 1) // bk
                start_core = T.alloc_var("int32", init=0)
                prev_end = T.alloc_var("int32", init=0)
                for g in T.serial(num_groups):
                    g_end = GroupEnds[g]
                    g_m = g_end - prev_end
                    if g_m > 0:
                        g_m_tiles = (g_m + bm - 1) // bm
                        for mt in T.serial(g_m_tiles):
                            assigned = (start_core + mt) % num_cores
                            if assigned == bx:
                                row_base = prev_end + mt * bm
                                for kt in T.Pipelined(k_tiles, num_stages=l1_stages):
                                    T.copy(A[row_base : row_base + bm, kt * bk : kt * bk + bk], a_l1)
                                    T.copy(B[g, :bn, kt * bk : kt * bk + bk], b_l1)
                                    T.gemm(a_l1, b_l1, c_l0c, transpose_B=True, clear_accum=(kt == 0))
                                T.dual_copy(c_l0c, c_ub)
                                with T.SimdVF():
                                    m32 = T.simd.pset(32, "PAT_ALL")
                                    m16 = T.simd.pset(16, "PAT_ALL")
                                    v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
                                    for r in T.serial(rows_per_sid):
                                        g_vec = T.simd.vld(c_ub[r, 0], dist="NORM_B32")
                                        u_vec = T.simd.vld(c_ub[r, half_n], dist="NORM_B32")
                                        neg_g = T.simd.vmuls(g_vec, T.float32(-1.0), mask=m32)
                                        sig = T.simd.vdiv(
                                            v_one,
                                            T.simd.vadd(v_one, T.simd.vexp(neg_g, mask=m32), mask=m32),
                                            mask=m32,
                                        )
                                        y_f32 = T.simd.vmul(T.simd.vmul(g_vec, sig, mask=m32), u_vec, mask=m32)
                                        y_16 = T.simd.vcvt(y_f32, tl_dtype, mask=m16, part=0)
                                        T.simd.vsts(out_ub[r, 0], y_16, mask=m32, dist="PK_B32", extent=64)
                                T.copy(
                                    out_ub,
                                    Out[row_base + sid * rows_per_sid : row_base + (sid + 1) * rows_per_sid, :half_n],
                                )
                        start_core = (start_core + g_m_tiles) % num_cores
                    prev_end = g_end

    return {
        "prim_func": grouped_gemm_epilogue_kernel,
        "pass_configs": {},
        "compile_flags": _resolve_flags(p["ub_256kb_unlock"]),
        "metadata": {"topology_axes": SHELL_CV_RAGGED_EPILOGUE.to_dict(), "tile_mnk": (bm, bn, bk)},
    }


def build_heterogeneous_cv_ring_template(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Shell 5 (coupled_1c2v/bidirectional_cv_ring/affine_simd): heterogeneous CV ring.

    Routes Cube QK^T through UB and L1 into Cube PV.
    """
    T = _get_tl_ascend()
    p = normalize_schedule_policy(SHELL_CV_RING_PIPELINE, policy)
    bm, bn, _ = p["tile_mnk"]
    bm = 64 if bm < 96 else 128
    bn = 64
    d_head = max(64, _align64(p["dim_size"]))
    rows_per_sid = bm // 2
    d_chunks = d_head // 64
    l1_stages = max(1, min(3, int(p["l1_stages"])))
    tl_dtype = p["dtype"] if p["dtype"] in ("float16", "bfloat16") else "float16"
    num_cores = 28

    q_rows = T.dynamic("q_rows")
    s_kv = T.dynamic("s_kv")

    with _ascend_target_ctx():

        @T.prim_func
        def heterogeneous_cv_ring_kernel(
            Q: T.Tensor((q_rows, d_head), tl_dtype),
            K: T.Tensor((s_kv, d_head), tl_dtype),
            V: T.Tensor((s_kv, d_head), tl_dtype),
            Out: T.Tensor((q_rows, d_head), tl_dtype),
            scale_val: T.float32,
        ):
            with T.MixedKernel(num_cores, sids=2) as (bx, sid):
                q_l1 = T.alloc_l1((bm, d_head), tl_dtype)
                k_l1 = T.alloc_l1((bn, d_head), tl_dtype)
                v_l1 = T.alloc_l1((d_head, bn), tl_dtype)
                p_l1 = T.alloc_l1((bm, bn), tl_dtype)
                s_l0c = T.alloc_l0c((bm, bn), "float32")
                o_l0c = T.alloc_l0c((bm, d_head), "float32")

                s_ub = T.alloc_shared((rows_per_sid, bn), "float32")
                p_ub = T.alloc_shared((rows_per_sid, bn), tl_dtype)
                o_tmp_ub = T.alloc_shared((rows_per_sid, d_head), "float32")
                o_ub = T.alloc_shared((rows_per_sid, d_head), "float32")
                out_ub = T.alloc_shared((rows_per_sid, d_head), tl_dtype)
                m_ub = T.alloc_shared((rows_per_sid, 8), "float32")
                l_ub = T.alloc_shared((rows_per_sid, 8), "float32")
                alpha_ub = T.alloc_shared((rows_per_sid, 8), "float32")

                T.annotate_buffer_versions(
                    {
                        q_l1: 1,
                        k_l1: l1_stages,
                        v_l1: l1_stages,
                        p_l1: 1,
                        s_l0c: 1,
                        o_l0c: 1,
                        s_ub: 1,
                        p_ub: 1,
                        o_tmp_ub: 1,
                        o_ub: 1,
                        out_ub: 1,
                        m_ub: 1,
                        l_ub: 1,
                        alpha_ub: 1,
                    }
                )
                if p["ub_256kb_unlock"]:
                    T.annotate_unlimit_memory("shared")

                m_tiles = (q_rows + bm - 1) // bm
                kv_blocks = (s_kv + bn - 1) // bn
                base_t = m_tiles // num_cores
                rem_t = m_tiles % num_cores
                t_start = bx * base_t + T.min(bx, rem_t)
                t_count = base_t + T.if_then_else(bx < rem_t, 1, 0)

                for lt in T.serial(t_count):
                    mt = t_start + lt
                    T.copy(Q[mt * bm : mt * bm + bm, :d_head], q_l1)
                    with T.SimdVF():
                        m32 = T.simd.pset(32, "PAT_ALL")
                        v_neg = T.simd.vdup(T.float32(-1e38), "float32", mask=m32)
                        v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                        for r in T.serial(rows_per_sid):
                            T.simd.vsts(m_ub[r, 0], v_neg, mask=m32, dist="ONEPT_B32")
                            T.simd.vsts(l_ub[r, 0], v_zero, mask=m32, dist="ONEPT_B32")
                            for c in T.serial(d_chunks):
                                T.simd.vsts(o_ub[r, c * 64], v_zero, mask=m32, dist="NORM_B32")

                    for kv in T.Pipelined(kv_blocks, num_stages=l1_stages):
                        T.copy(K[kv * bn : kv * bn + bn, :d_head], k_l1)
                        T.copy(V[kv * bn : kv * bn + bn, :d_head], v_l1, transpose=True)
                        T.gemm(q_l1, k_l1, s_l0c, transpose_B=True, clear_accum=True)
                        T.dual_copy(s_l0c, s_ub)
                        with T.SimdVF():
                            m32 = T.simd.pset(32, "PAT_ALL")
                            m16 = T.simd.pset(16, "PAT_ALL")
                            for r in T.serial(rows_per_sid):
                                s_vec = T.simd.vmuls(T.simd.vld(s_ub[r, 0], dist="NORM_B32"), scale_val, mask=m32)
                                m_tile = T.simd.vdupv(T.simd.vcmax(s_vec, mask=m32), mask=m32, pos="POS_LOWEST")
                                m_old = T.simd.vld(m_ub[r, 0], dist="BRC_B32")
                                l_old = T.simd.vld(l_ub[r, 0], dist="BRC_B32")
                                m_new = T.simd.vmax(m_old, m_tile, mask=m32)
                                alpha_v = T.simd.vexp(T.simd.vsub(m_old, m_new, mask=m32), mask=m32)
                                p_f32 = T.simd.vexp(T.simd.vsub(s_vec, m_new, mask=m32), mask=m32)
                                row_sum = T.simd.vdupv(T.simd.vcadd(p_f32, mask=m32), mask=m32, pos="POS_LOWEST")
                                l_new = T.simd.vadd(T.simd.vmul(l_old, alpha_v, mask=m32), row_sum, mask=m32)
                                T.simd.vsts(m_ub[r, 0], m_new, mask=m32, dist="ONEPT_B32")
                                T.simd.vsts(l_ub[r, 0], l_new, mask=m32, dist="ONEPT_B32")
                                T.simd.vsts(alpha_ub[r, 0], alpha_v, mask=m32, dist="ONEPT_B32")
                                p_16 = T.simd.vcvt(p_f32, tl_dtype, mask=m16, part=0)
                                T.simd.vsts(p_ub[r, 0], p_16, mask=m32, dist="PK_B32", extent=64)
                        T.dual_copy(p_ub, p_l1)
                        T.gemm(p_l1, v_l1, o_l0c, transpose_B=True, clear_accum=True)
                        T.dual_copy(o_l0c, o_tmp_ub)
                        with T.SimdVF():
                            m32 = T.simd.pset(32, "PAT_ALL")
                            for r in T.serial(rows_per_sid):
                                alpha_v = T.simd.vld(alpha_ub[r, 0], dist="BRC_B32")
                                for c in T.serial(d_chunks):
                                    off = c * 64
                                    o_new = T.simd.vadd(
                                        T.simd.vmul(T.simd.vld(o_ub[r, off], dist="NORM_B32"), alpha_v, mask=m32),
                                        T.simd.vld(o_tmp_ub[r, off], dist="NORM_B32"),
                                        mask=m32,
                                    )
                                    T.simd.vsts(o_ub[r, off], o_new, mask=m32, dist="NORM_B32")

                    with T.SimdVF():
                        m32 = T.simd.pset(32, "PAT_ALL")
                        m16 = T.simd.pset(16, "PAT_ALL")
                        v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
                        for r in T.serial(rows_per_sid):
                            inv_l = T.simd.vdiv(v_one, T.simd.vld(l_ub[r, 0], dist="BRC_B32"), mask=m32)
                            for c in T.serial(d_chunks):
                                off = c * 64
                                o_f = T.simd.vmul(T.simd.vld(o_ub[r, off], dist="NORM_B32"), inv_l, mask=m32)
                                o_16 = T.simd.vcvt(o_f, tl_dtype, mask=m16, part=0)
                                T.simd.vsts(out_ub[r, off], o_16, mask=m32, dist="PK_B32", extent=64)
                    T.copy(
                        out_ub,
                        Out[mt * bm + sid * rows_per_sid : mt * bm + (sid + 1) * rows_per_sid, :d_head],
                    )

    return {
        "prim_func": heterogeneous_cv_ring_kernel,
        "pass_configs": {},
        "compile_flags": _resolve_flags(p["ub_256kb_unlock"]),
        "metadata": {"topology_axes": SHELL_CV_RING_PIPELINE.to_dict(), "tile_mnk": (bm, bn, d_head)},
    }


def build_persistent_chunk_state_template(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Shell 6 (coupled_1c2v/persistent_onchip_state/affine_simd): persistent state.

    Each (B, H) pair owns a core and keeps FP32 state resident in UB. The decay
    exponent splits into ``exp(g/2) * exp(g/2)`` to halve its dynamic range.
    """
    T = _get_tl_ascend()
    p = normalize_schedule_policy(SHELL_CV_PERSISTENT_STATE, policy)
    d_head = max(64, min(128, _align64(p["dim_size"])))
    d_chunks = d_head // 64
    num_cores = 56
    bh_dim = T.dynamic("bh_dim")
    t_dim = T.dynamic("t_dim")

    with _ascend_target_ctx():

        @T.prim_func
        def persistent_chunk_state_kernel(
            K: T.Tensor((bh_dim, t_dim, d_head), "float32"),
            V: T.Tensor((bh_dim, t_dim, d_head), "float32"),
            G: T.Tensor((bh_dim, t_dim, d_head), "float32"),
            State: T.Tensor((bh_dim, d_head, d_head), "float32"),
            Out: T.Tensor((bh_dim, t_dim, d_head), "float32"),
        ):
            with T.Kernel(num_cores) as cid:
                k_ub = T.alloc_shared((d_head,), "float32")
                v_ub = T.alloc_shared((d_head,), "float32")
                g_ub = T.alloc_shared((d_head,), "float32")
                s_ub = T.alloc_shared((d_head, d_head), "float32")
                o_ub = T.alloc_shared((d_head,), "float32")
                T.annotate_buffer_versions({k_ub: 1, v_ub: 1, g_ub: 1, s_ub: 1, o_ub: 1})
                if p["ub_256kb_unlock"]:
                    T.annotate_unlimit_memory("shared")

                base_bh = bh_dim // num_cores
                rem_bh = bh_dim % num_cores
                bh_start = cid * base_bh + T.min(cid, rem_bh)
                bh_count = base_bh + T.if_then_else(cid < rem_bh, 1, 0)

                for lb in T.serial(bh_count):
                    bh = bh_start + lb
                    T.copy(State[bh, :d_head, :d_head], s_ub[:d_head, :d_head])
                    for t in T.serial(t_dim):
                        T.copy(K[bh, t, :d_head], k_ub[:d_head])
                        T.copy(V[bh, t, :d_head], v_ub[:d_head])
                        T.copy(G[bh, t, :d_head], g_ub[:d_head])
                        with T.SimdVF():
                            m32 = T.simd.pset(32, "PAT_ALL")
                            for c in T.serial(d_chunks):
                                off = c * 64
                                gv = T.simd.vld(g_ub[off], dist="NORM_B32")
                                half_g = T.simd.vmuls(gv, T.float32(0.5), mask=m32)
                                decay_half = T.simd.vexp(half_g, mask=m32)
                                decay_v = T.simd.vmul(decay_half, decay_half, mask=m32)
                                kv = T.simd.vld(k_ub[off], dist="NORM_B32")
                                vv = T.simd.vld(v_ub[off], dist="NORM_B32")
                                sv = T.simd.vld(s_ub[0, off], dist="NORM_B32")
                                s_new = T.simd.vadd(
                                    T.simd.vmul(sv, decay_v, mask=m32),
                                    T.simd.vmul(kv, vv, mask=m32),
                                    mask=m32,
                                )
                                T.simd.vsts(s_ub[0, off], s_new, mask=m32, dist="NORM_B32")
                                T.simd.vsts(o_ub[off], s_new, mask=m32, dist="NORM_B32")
                        T.copy(o_ub[:d_head], Out[bh, t, :d_head])
                    T.copy(s_ub[:d_head, :d_head], State[bh, :d_head, :d_head])
                    T.ascend_pipe_barrier("PIPE_ALL")

    return {
        "prim_func": persistent_chunk_state_kernel,
        "pass_configs": {},
        "compile_flags": _resolve_flags(p["ub_256kb_unlock"]),
        "metadata": {"topology_axes": SHELL_CV_PERSISTENT_STATE.to_dict(), "d_head": d_head},
    }


def build_striped_register_vm_template(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Shell 7 (aiv_only/persistent_onchip_state/dynamic_opcode_stream): striped VM.

    Partitions the horizontal W dimension into UB-resident stripes.
    """
    T = _get_tl_ascend()
    p = normalize_schedule_policy(SHELL_AIV_STRIPED_VM, policy)
    num_regs = max(4, min(32, int(p["tile_mnk"][0])))
    w_slice = max(64, min(256, _align64(p["tile_mnk"][1])))
    w_chunks = w_slice // 64
    tl_dtype = p["dtype"] if p["dtype"] in ("float16", "bfloat16", "float32") else "float32"
    num_cores = 56
    prog_len = T.dynamic("prog_len")
    w_total = T.dynamic("w_total")

    with _ascend_target_ctx():

        @T.prim_func
        def striped_register_vm_kernel(
            Program: T.Tensor((prog_len, 4), "int32"),
            InitRegs: T.Tensor((num_regs, w_total), tl_dtype),
            FinalRegs: T.Tensor((num_regs, w_total), tl_dtype),
        ):
            with T.Kernel(num_cores) as cid:
                regs_ub = T.alloc_shared((num_regs, w_slice), "float32")
                T.annotate_buffer_versions({regs_ub: 1})
                if p["ub_256kb_unlock"]:
                    T.annotate_unlimit_memory("shared")

                total_slices = (w_total + w_slice - 1) // w_slice
                base_s = total_slices // num_cores
                rem_s = total_slices % num_cores
                s_start = cid * base_s + T.min(cid, rem_s)
                s_count = base_s + T.if_then_else(cid < rem_s, 1, 0)

                for ls in T.serial(s_count):
                    sid = s_start + ls
                    w0 = sid * w_slice
                    w_valid = T.min(w_slice, w_total - w0)
                    for r in T.serial(num_regs):
                        if tl_dtype == "float32":
                            T.copy(InitRegs[r, w0 : w0 + w_valid], regs_ub[r, :w_valid])
                        else:
                            tmp16 = T.alloc_shared((w_slice,), tl_dtype)
                            T.copy(InitRegs[r, w0 : w0 + w_valid], tmp16[:w_valid])
                            with T.SimdVF():
                                m32 = T.simd.pset(32, "PAT_ALL")
                                m16 = T.simd.pset(16, "PAT_ALL")
                                for c in T.serial(w_chunks):
                                    off = c * 64
                                    vf = T.simd.vcvt(
                                        T.simd.vld(tmp16[off], dist="US_B16"),
                                        "float32",
                                        mask=m16,
                                        part=0,
                                    )
                                    T.simd.vsts(regs_ub[r, off], vf, mask=m32, dist="NORM_B32")

                    for pc in T.serial(prog_len):
                        op = Program[pc, 0] % 8
                        dst = Program[pc, 1] % num_regs
                        src_a = Program[pc, 2] % num_regs
                        src_b = Program[pc, 3] % num_regs
                        with T.SimdVF():
                            m32 = T.simd.pset(32, "PAT_ALL")
                            v_hi = T.simd.vdup(T.float32(10000.0), "float32", mask=m32)
                            v_lo = T.simd.vdup(T.float32(-10000.0), "float32", mask=m32)
                            v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                            res_reg = T.simd.alloc_local((1,), "float32")
                            for c in T.serial(w_chunks):
                                off = c * 64
                                va = T.simd.vld(regs_ub[src_a, off], dist="NORM_B32")
                                vb = T.simd.vld(regs_ub[src_b, off], dist="NORM_B32")
                                vd = T.simd.vld(regs_ub[dst, off], dist="NORM_B32")
                                res_reg[0] = va
                                if op == 0:
                                    res_reg[0] = T.simd.vmax(v_lo, T.simd.vmin(v_hi, T.simd.vadd(va, vb, mask=m32), mask=m32), mask=m32)
                                elif op == 1:
                                    res_reg[0] = T.simd.vmax(v_lo, T.simd.vmin(v_hi, T.simd.vmul(va, vb, mask=m32), mask=m32), mask=m32)
                                elif op == 2:
                                    res_reg[0] = T.simd.vmax(va, v_zero, mask=m32)
                                elif op == 4:
                                    res_reg[0] = T.simd.vmax(va, vb, mask=m32)
                                elif op == 6:
                                    res_reg[0] = T.simd.vmax(
                                        v_lo,
                                        T.simd.vmin(v_hi, T.simd.vadd(vd, T.simd.vmul(va, vb, mask=m32), mask=m32), mask=m32),
                                        mask=m32,
                                    )
                                T.simd.vsts(regs_ub[dst, off], res_reg[0], mask=m32, dist="NORM_B32")
                        T.ascend_pipe_barrier("PIPE_V")

                    for r in T.serial(num_regs):
                        if tl_dtype == "float32":
                            T.copy(regs_ub[r, :w_valid], FinalRegs[r, w0 : w0 + w_valid])
                        else:
                            tmp16 = T.alloc_shared((w_slice,), tl_dtype)
                            with T.SimdVF():
                                m32 = T.simd.pset(32, "PAT_ALL")
                                m16 = T.simd.pset(16, "PAT_ALL")
                                for c in T.serial(w_chunks):
                                    off = c * 64
                                    vf = T.simd.vld(regs_ub[r, off], dist="NORM_B32")
                                    v16 = T.simd.vcvt(vf, tl_dtype, mask=m16, part=0)
                                    T.simd.vsts(tmp16[off], v16, mask=m32, dist="PK_B32", extent=64)
                            T.copy(tmp16[:w_valid], FinalRegs[r, w0 : w0 + w_valid])
                    T.ascend_pipe_barrier("PIPE_ALL")

    return {
        "prim_func": striped_register_vm_kernel,
        "pass_configs": {},
        "compile_flags": _resolve_flags(p["ub_256kb_unlock"]),
        "metadata": {"topology_axes": SHELL_AIV_STRIPED_VM.to_dict(), "num_regs": num_regs, "w_slice": w_slice},
    }


TILELANG_BUILDERS = {
    SHELL_AIV_SIMD_STREAM: build_pure_regbase_simd_template,
    SHELL_AIV_SIMT_ATOMIC: build_simt_and_hybrid_ub_template,
    SHELL_AIC_GEMM_FIXPIPE: build_dense_and_quant_gemm_template,
    SHELL_CV_RAGGED_EPILOGUE: build_grouped_gemm_epilogue_template,
    SHELL_CV_RING_PIPELINE: build_heterogeneous_cv_ring_template,
    SHELL_CV_PERSISTENT_STATE: build_persistent_chunk_state_template,
    SHELL_AIV_STRIPED_VM: build_striped_register_vm_template,
}

__all__ = [
    "TILELANG_BUILDERS",
    "build_pure_regbase_simd_template",
    "build_simt_and_hybrid_ub_template",
    "build_dense_and_quant_gemm_template",
    "build_grouped_gemm_epilogue_template",
    "build_heterogeneous_cv_ring_template",
    "build_persistent_chunk_state_template",
    "build_striped_register_vm_template",
]
