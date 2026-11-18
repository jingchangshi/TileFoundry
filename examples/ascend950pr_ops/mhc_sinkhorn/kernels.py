"""Round 3 MhcSinkhorn candidate: hardware SFU vdiv(precision='ftz_true') + pipelined DMA + single-load loops."""

import contextlib
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


def _select_mhc_sinkhorn_config(
    hc_mult: int = 4,
    *,
    b_dim: int = 1024,
    iter_step: int = 20,
    schedule_mode: str = "tuned",
) -> dict[str, object]:
    mat_elems = max(1, int(hc_mult)) * max(1, int(hc_mult))
    ub_budget = 256 * 1024 - 4096 if schedule_mode == "tuned" else 192 * 1024
    max_batch_tile = (
        64
        if (64 * mat_elems * 4 * 2) <= ub_budget
        else max(1, min(64, ub_budget // (2 * mat_elems * 4)))
    )
    regime = "sfu_reg_cached_small_mat" if hc_mult <= 4 else "sfu_pipelined_tile_mat"
    return {
        "regime": regime,
        "hc_mult": int(hc_mult),
        "iter_step": max(1, int(iter_step)),
        "batch_tile": max_batch_tile,
        "pass_configs": {},
        "compile_flags": list(PURE_SIMDVF_UBUF_FLAGS) if schedule_mode == "tuned" else [],
    }


def _build_mhc_sinkhorn_prim_func(
    hc_mult: int = 4,
    iter_step: int = 20,
    batch_tile: int = 64,
    schedule_mode: str = "tuned",
    **_extra_kwargs,
):
    num_cores = 56
    m_n = int(hc_mult)
    mat_elems = m_n * m_n
    b_tile = 64
    io_cap = ((b_tile * mat_elems + 63) // 64) * 64
    n_elems = T.dynamic("n_elems")

    with _ascend_target_ctx():
        if m_n == 4:

            @T.prim_func
            def main(
                Comb: T.Tensor((n_elems,), "float32"),
                CombOut: T.Tensor((n_elems,), "float32"),
                eps: T.float32,
                b_dim: T.int32,
            ):
                with T.Kernel(num_cores) as cid:
                    io_in_ub = T.alloc_shared((io_cap,), "float32")
                    io_out_ub = T.alloc_shared((io_cap,), "float32")
                    work_ub = T.alloc_shared((4, 4, b_tile), "float32")
                    T.annotate_buffer_versions({io_in_ub: 2, io_out_ub: 2, work_ub: 1})
                    if schedule_mode == "tuned":
                        T.annotate_unlimit_memory("shared")

                    base_b = b_dim // num_cores
                    rem_b = b_dim % num_cores
                    b_start = cid * base_b + T.min(cid, rem_b)
                    b_count = base_b + T.if_then_else(cid < rem_b, 1, 0)
                    num_chunks = (b_count + b_tile - 1) // b_tile

                    for ch in T.Pipelined(num_chunks, num_stages=2):
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
                            base_off = T.simd.vmuls(lane, 16, mask=m32)

                            for r in T.serial(4):
                                r0 = T.simd.vsel(T.simd.vgather2(io_in_ub[0], T.reinterpret("uint32x64", T.simd.vadds(base_off, r * 4 + 0, mask=m32)), mask=m_act), v_zero, m_act)
                                r1 = T.simd.vsel(T.simd.vgather2(io_in_ub[0], T.reinterpret("uint32x64", T.simd.vadds(base_off, r * 4 + 1, mask=m32)), mask=m_act), v_zero, m_act)
                                r2 = T.simd.vsel(T.simd.vgather2(io_in_ub[0], T.reinterpret("uint32x64", T.simd.vadds(base_off, r * 4 + 2, mask=m32)), mask=m_act), v_zero, m_act)
                                r3 = T.simd.vsel(T.simd.vgather2(io_in_ub[0], T.reinterpret("uint32x64", T.simd.vadds(base_off, r * 4 + 3, mask=m32)), mask=m_act), v_zero, m_act)
                                rmax = T.simd.vmax(T.simd.vmax(r0, r1, mask=m32), T.simd.vmax(r2, r3, mask=m32), mask=m32)
                                e0 = T.simd.vexpdif(r0, rmax, mask=m32)
                                e1 = T.simd.vexpdif(r1, rmax, mask=m32)
                                e2 = T.simd.vexpdif(r2, rmax, mask=m32)
                                e3 = T.simd.vexpdif(r3, rmax, mask=m32)
                                inv_r = T.simd.vdiv(v_one, T.simd.vadd(T.simd.vadd(e0, e1, mask=m32), T.simd.vadd(e2, e3, mask=m32), mask=m32), mask=m32, precision="ftz_true")
                                T.simd.vsts(work_ub[r, 0, 0], T.simd.vadds(T.simd.vmul(e0, inv_r, mask=m32), eps, mask=m32), mask=m32, dist="NORM_B32")
                                T.simd.vsts(work_ub[r, 1, 0], T.simd.vadds(T.simd.vmul(e1, inv_r, mask=m32), eps, mask=m32), mask=m32, dist="NORM_B32")
                                T.simd.vsts(work_ub[r, 2, 0], T.simd.vadds(T.simd.vmul(e2, inv_r, mask=m32), eps, mask=m32), mask=m32, dist="NORM_B32")
                                T.simd.vsts(work_ub[r, 3, 0], T.simd.vadds(T.simd.vmul(e3, inv_r, mask=m32), eps, mask=m32), mask=m32, dist="NORM_B32")
                            T.simd.mem_bar("VST_VLD")

                            for c in T.serial(4):
                                c0 = T.simd.vld(work_ub[0, c, 0], dist="NORM_B32")
                                c1 = T.simd.vld(work_ub[1, c, 0], dist="NORM_B32")
                                c2 = T.simd.vld(work_ub[2, c, 0], dist="NORM_B32")
                                c3 = T.simd.vld(work_ub[3, c, 0], dist="NORM_B32")
                                inv_c = T.simd.vdiv(v_one, T.simd.vadds(T.simd.vadd(T.simd.vadd(c0, c1, mask=m32), T.simd.vadd(c2, c3, mask=m32), mask=m32), eps, mask=m32), mask=m32, precision="ftz_true")
                                T.simd.vsts(work_ub[0, c, 0], T.simd.vmul(c0, inv_c, mask=m32), mask=m32, dist="NORM_B32")
                                T.simd.vsts(work_ub[1, c, 0], T.simd.vmul(c1, inv_c, mask=m32), mask=m32, dist="NORM_B32")
                                T.simd.vsts(work_ub[2, c, 0], T.simd.vmul(c2, inv_c, mask=m32), mask=m32, dist="NORM_B32")
                                T.simd.vsts(work_ub[3, c, 0], T.simd.vmul(c3, inv_c, mask=m32), mask=m32, dist="NORM_B32")
                            T.simd.mem_bar("VST_VLD")

                            for _it in T.serial(iter_step - 1):
                                for r in T.serial(4):
                                    x0 = T.simd.vld(work_ub[r, 0, 0], dist="NORM_B32")
                                    x1 = T.simd.vld(work_ub[r, 1, 0], dist="NORM_B32")
                                    x2 = T.simd.vld(work_ub[r, 2, 0], dist="NORM_B32")
                                    x3 = T.simd.vld(work_ub[r, 3, 0], dist="NORM_B32")
                                    inv_r2 = T.simd.vdiv(v_one, T.simd.vadds(T.simd.vadd(T.simd.vadd(x0, x1, mask=m32), T.simd.vadd(x2, x3, mask=m32), mask=m32), eps, mask=m32), mask=m32, precision="ftz_true")
                                    T.simd.vsts(work_ub[r, 0, 0], T.simd.vmul(x0, inv_r2, mask=m32), mask=m32, dist="NORM_B32")
                                    T.simd.vsts(work_ub[r, 1, 0], T.simd.vmul(x1, inv_r2, mask=m32), mask=m32, dist="NORM_B32")
                                    T.simd.vsts(work_ub[r, 2, 0], T.simd.vmul(x2, inv_r2, mask=m32), mask=m32, dist="NORM_B32")
                                    T.simd.vsts(work_ub[r, 3, 0], T.simd.vmul(x3, inv_r2, mask=m32), mask=m32, dist="NORM_B32")
                                T.simd.mem_bar("VST_VLD")
                                for c in T.serial(4):
                                    y0 = T.simd.vld(work_ub[0, c, 0], dist="NORM_B32")
                                    y1 = T.simd.vld(work_ub[1, c, 0], dist="NORM_B32")
                                    y2 = T.simd.vld(work_ub[2, c, 0], dist="NORM_B32")
                                    y3 = T.simd.vld(work_ub[3, c, 0], dist="NORM_B32")
                                    inv_c2 = T.simd.vdiv(v_one, T.simd.vadds(T.simd.vadd(T.simd.vadd(y0, y1, mask=m32), T.simd.vadd(y2, y3, mask=m32), mask=m32), eps, mask=m32), mask=m32, precision="ftz_true")
                                    T.simd.vsts(work_ub[0, c, 0], T.simd.vmul(y0, inv_c2, mask=m32), mask=m32, dist="NORM_B32")
                                    T.simd.vsts(work_ub[1, c, 0], T.simd.vmul(y1, inv_c2, mask=m32), mask=m32, dist="NORM_B32")
                                    T.simd.vsts(work_ub[2, c, 0], T.simd.vmul(y2, inv_c2, mask=m32), mask=m32, dist="NORM_B32")
                                    T.simd.vsts(work_ub[3, c, 0], T.simd.vmul(y3, inv_c2, mask=m32), mask=m32, dist="NORM_B32")
                                T.simd.mem_bar("VST_VLD")

                            for r in T.serial(4):
                                for c in T.serial(4):
                                    idx = T.reinterpret("uint32x64", T.simd.vadds(base_off, r * 4 + c, mask=m32))
                                    T.simd.vscatter(T.simd.vld(work_ub[r, c, 0], dist="NORM_B32"), io_out_ub[0], idx, mask=m_act)

                        T.copy(io_out_ub[:elem_cnt], CombOut[elem_off : elem_off + elem_cnt])

            return main

        use_double_buf = m_n <= 8
        stages = 2 if use_double_buf else 1

        @T.prim_func
        def main_gen(
            Comb: T.Tensor((n_elems,), "float32"),
            CombOut: T.Tensor((n_elems,), "float32"),
            eps: T.float32,
            b_dim: T.int32,
        ):
            with T.Kernel(num_cores) as cid:
                io_in_ub = T.alloc_shared((io_cap,), "float32")
                io_out_ub = T.alloc_shared((io_cap,), "float32") if use_double_buf else io_in_ub
                work_ub = T.alloc_shared((m_n, m_n, b_tile), "float32")
                if use_double_buf:
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

                        for _it in T.serial(iter_step - 1):
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
    key = (hc_mult, iter_step, int(cfg["batch_tile"]), schedule_mode)
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        with _ascend_target_ctx():
            prog = _build_mhc_sinkhorn_prim_func(
                hc_mult=hc_mult,
                iter_step=iter_step,
                batch_tile=int(cfg["batch_tile"]),
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
    compiled = _get_kernel(hc_mult, int(iter_step), b_dim, schedule_mode)
    return compiled(comb.contiguous().view(-1), float(eps), int(b_dim)).view(b_dim, hc_mult, hc_mult)

kernel = mhc_sinkhorn
