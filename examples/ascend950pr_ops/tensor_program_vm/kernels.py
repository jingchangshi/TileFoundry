"""TileLang Ascend950PR Full-Opcode Register-Resident / GM-Streaming VM kernel for TensorProgramVm."""

import contextlib

import tilelang
import tilelang.ascend.language as T
import torch

from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS

_KERNEL_CACHE: dict[tuple[object, ...], object] = {}
_WORKSPACE_CACHE: dict[tuple[int, int, int], torch.Tensor] = {}
_SYNC_CACHE: dict[int, torch.Tensor] = {}


def _ascend_target_ctx():
    try:
        from tilelang import tvm
        from tilelang.backend.module import create_backend_context

        if tvm.target.Target.current(allow_none=True) is None:
            return create_backend_context("ascend").target
    except Exception:
        pass
    return contextlib.nullcontext()


def _get_regs_workspace(num_regs: int, w_dim: int, device: torch.device) -> torch.Tensor:
    dev_idx = device.index if device.index is not None else 0
    key = (int(num_regs), int(w_dim), int(dev_idx))
    buf = _WORKSPACE_CACHE.get(key)
    if buf is None or buf.device != device:
        buf = torch.empty((int(num_regs), int(w_dim)), dtype=torch.float32, device=device)
        _WORKSPACE_CACHE[key] = buf
    return buf


def _get_sync_workspace(device: torch.device) -> torch.Tensor:
    dev_idx = device.index if device.index is not None else 0
    buf = _SYNC_CACHE.get(int(dev_idx))
    if buf is None or buf.device != device:
        buf = torch.zeros((64, 56, 8), dtype=torch.float32, device=device)
        _SYNC_CACHE[int(dev_idx)] = buf
    return buf


def _select_tensor_program_vm_config(
    num_regs: int = 16,
    w_dim: int = 1024,
    tl_dtype: str = "bfloat16",
    *,
    prog_len: int = 1024,
    schedule_mode: str = "tuned",
) -> dict[str, object]:
    """Select 56-AIV W-striped full-UB resident VM vs GM-workspace row-streaming VM."""
    ub_budget = 200 * 1024 if schedule_mode == "tuned" else 160 * 1024
    num_cores = 56
    w_per_core = (max(1, int(w_dim)) + num_cores - 1) // num_cores
    w_alloc = max(64, ((w_per_core + 63) // 64) * 64)
    use_gm = (int(num_regs) * w_alloc * 4) > ub_budget
    regime = "gm_row_streaming" if use_gm else "full_ub_resident_56aiv"
    return {
        "regime": regime,
        "num_regs": int(num_regs),
        "w_slice": w_alloc,
        "use_gm_workspace": use_gm,
        "tl_dtype": tl_dtype,
        "prog_len": int(prog_len),
        "pass_configs": {},
        "compile_flags": list(PURE_SIMDVF_UBUF_FLAGS) if schedule_mode == "tuned" else [],
    }


def _build_tensor_program_vm_prim_func(
    num_regs: int = 16,
    w_slice: int = 256,
    tl_dtype: str = "bfloat16",
    use_gm_workspace: bool = False,
    schedule_mode: str = "tuned",
    **_extra_kwargs,
):
    """Build Ascend 950 VM @T.prim_func supporting all 8 opcodes (ADD/MUL/RELU/MEANB/MAX/ROLL/MAC/COPY)."""
    if num_regs <= 0:
        raise ValueError(f"num_regs must be positive, got {num_regs}")
    if tl_dtype not in ("float16", "bfloat16", "float32"):
        raise ValueError(f"unsupported dtype {tl_dtype!r}")

    r_count = int(num_regs)
    w_blk = max(64, ((int(w_slice) + 63) // 64) * 64)
    prog_chunk = 512

    prog_len = T.dynamic("prog_len")
    w_total = T.dynamic("w_total")

    with _ascend_target_ctx():
        if not use_gm_workspace:
            num_cores = 56
            num_slots = 64

            @T.macro
            def _exec_elem(regs_ub, op, dst, src_a, src_b, w_chunks, w_rem):
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
                            res_reg[0] = T.simd.vmax(
                                v_lo, T.simd.vmin(v_hi, T.simd.vadd(va, vb, mask=m32), mask=m32), mask=m32
                            )
                        elif op == 1:
                            res_reg[0] = T.simd.vmax(
                                v_lo, T.simd.vmin(v_hi, T.simd.vmul(va, vb, mask=m32), mask=m32), mask=m32
                            )
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
                    if w_rem > 0:
                        off = w_chunks * 64
                        m32_t = T.simd.update_mask(w_rem, width=32)
                        va_t = T.simd.vld(regs_ub[src_a, off], dist="NORM_B32")
                        vb_t = T.simd.vld(regs_ub[src_b, off], dist="NORM_B32")
                        vd_t = T.simd.vld(regs_ub[dst, off], dist="NORM_B32")
                        res_reg[0] = va_t
                        if op == 0:
                            res_reg[0] = T.simd.vmax(
                                v_lo, T.simd.vmin(v_hi, T.simd.vadd(va_t, vb_t, mask=m32_t), mask=m32_t), mask=m32_t
                            )
                        elif op == 1:
                            res_reg[0] = T.simd.vmax(
                                v_lo, T.simd.vmin(v_hi, T.simd.vmul(va_t, vb_t, mask=m32_t), mask=m32_t), mask=m32_t
                            )
                        elif op == 2:
                            res_reg[0] = T.simd.vmax(va_t, v_zero, mask=m32_t)
                        elif op == 4:
                            res_reg[0] = T.simd.vmax(va_t, vb_t, mask=m32_t)
                        elif op == 6:
                            res_reg[0] = T.simd.vmax(
                                v_lo,
                                T.simd.vmin(
                                    v_hi,
                                    T.simd.vadd(vd_t, T.simd.vmul(va_t, vb_t, mask=m32_t), mask=m32_t),
                                    mask=m32_t,
                                ),
                                mask=m32_t,
                            )
                        T.simd.vsts(regs_ub[dst, off], res_reg[0], mask=m32_t, dist="NORM_B32")
                    T.simd.mem_bar("VST_VLD")

            @T.prim_func
            def main(
                Program: T.Tensor((prog_len, 4), "int32"),
                InitRegs: T.Tensor((r_count, w_total), tl_dtype),
                FinalRegs: T.Tensor((r_count, w_total), tl_dtype),
                WorkGM: T.Tensor((num_slots, 56, 8), "float32"),
            ):
                with T.Kernel(num_cores) as cid:
                    regs_ub = T.alloc_shared((r_count, w_blk), "float32")
                    io_ub = T.alloc_shared((w_blk,), tl_dtype)
                    prog_ub = T.alloc_shared((prog_chunk, 4), "int32")
                    comm_ub = T.alloc_shared((64,), "float32")
                    gather_ub = T.alloc_shared((56, 8), "float32")
                    T.annotate_buffer_versions({regs_ub: 1, io_ub: 1, prog_ub: 1, comm_ub: 1, gather_ub: 1})
                    if schedule_mode == "tuned":
                        T.annotate_unlimit_memory("shared")

                    base_w = w_total // num_cores
                    rem_w = w_total % num_cores
                    w_start = cid * base_w + T.min(cid, rem_w)
                    w_cur = base_w + T.if_then_else(cid < rem_w, 1, 0)
                    w_chunks = w_cur // 64
                    w_rem = w_cur % 64
                    inv_w = T.float32(1.0) / T.cast(w_total, "float32")
                    active_cores = T.min(num_cores, w_total)
                    prev_cid = (cid + active_cores - 1) % active_cores

                    if w_cur > 0:
                        for r in T.serial(r_count):
                            if tl_dtype == "float32":
                                T.copy(InitRegs[r, w_start : w_start + w_cur], regs_ub[r, :w_cur])
                            else:
                                T.copy(InitRegs[r, w_start : w_start + w_cur], io_ub[:w_cur])
                                with T.SimdVF():
                                    m32 = T.simd.pset(32, "PAT_ALL")
                                    m16 = T.simd.pset(16, "PAT_ALL")
                                    for c in T.serial(w_chunks):
                                        off = c * 64
                                        vf = T.simd.vcvt(
                                            T.simd.vld(io_ub[off], dist="US_B16"), "float32", mask=m16, part=0
                                        )
                                        T.simd.vsts(regs_ub[r, off], vf, mask=m32, dist="NORM_B32")
                                    if w_rem > 0:
                                        off = w_chunks * 64
                                        m32_t = T.simd.update_mask(w_rem, width=32)
                                        m16_t = T.simd.update_mask(w_rem * 2, width=16)
                                        vf_t = T.simd.vcvt(
                                            T.simd.vld(io_ub[off], dist="US_B16"), "float32", mask=m16_t, part=0
                                        )
                                        T.simd.vsts(regs_ub[r, off], vf_t, mask=m32_t, dist="NORM_B32")
                                    T.simd.mem_bar("VST_VLD")

                    bar_step = T.alloc_var("int32", init=0)
                    done = T.alloc_var("int32", init=0)
                    g_sum = T.alloc_var("float32", init=0.0)
                    p_blocks = (prog_len + prog_chunk - 1) // prog_chunk
                    for pb in T.serial(p_blocks):
                        p0 = pb * prog_chunk
                        p_cur = T.min(prog_chunk, prog_len - p0)
                        T.copy(Program[p0 : p0 + p_cur, :4], prog_ub[:p_cur, :4])
                        T.ascend_pipe_barrier("PIPE_ALL")

                        for lp in T.serial(p_cur):
                            op = prog_ub[lp, 0] % 8
                            dst = prog_ub[lp, 1] % r_count
                            src_a = prog_ub[lp, 2] % r_count
                            src_b = prog_ub[lp, 3] % r_count

                            if op == 5:
                                bar_step = bar_step + 1
                                slot = bar_step % num_slots
                                step_f = T.cast(bar_step, "float32")
                                if w_cur > 0:
                                    comm_ub[0] = regs_ub[src_a, w_cur - 1]
                                else:
                                    comm_ub[0] = T.float32(0.0)
                                comm_ub[1] = step_f
                                T.ascend_pipe_barrier("PIPE_ALL")
                                T.copy(comm_ub[:8], WorkGM[slot, cid, :8])
                                T.ascend_pipe_barrier("PIPE_ALL")
                                if cid < active_cores:
                                    done = 0
                                    while done == 0:
                                        T.copy(WorkGM[slot, prev_cid, :8], comm_ub[:8])
                                        T.ascend_pipe_barrier("PIPE_ALL")
                                        if comm_ub[1] >= step_f:
                                            done = 1
                                    for idx in T.serial(w_cur - 1):
                                        j = w_cur - 1 - idx
                                        regs_ub[dst, j] = regs_ub[src_a, j - 1]
                                    regs_ub[dst, 0] = comm_ub[0]
                            elif op == 3:
                                if w_cur > 0:
                                    with T.SimdVF():
                                        m32 = T.simd.pset(32, "PAT_ALL")
                                        v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                                        sum_v = T.simd.alloc_local((1,), "float32")
                                        sum_v[0] = v_zero
                                        for c in T.serial(w_chunks):
                                            sum_v[0] = T.simd.vadd(
                                                sum_v[0],
                                                T.simd.vld(regs_ub[src_a, c * 64], dist="NORM_B32"),
                                                mask=m32,
                                            )
                                        if w_rem > 0:
                                            m32_t = T.simd.update_mask(w_rem, width=32)
                                            tail_v = T.simd.vmuls(
                                                T.simd.vld(regs_ub[src_a, w_chunks * 64], dist="NORM_B32"),
                                                T.float32(1.0),
                                                mask=m32_t,
                                            )
                                            sum_v[0] = T.simd.vadd(sum_v[0], tail_v, mask=m32)
                                        tot_v = T.simd.vdupv(
                                            T.simd.vcadd(sum_v[0], mask=m32), mask=m32, pos="POS_LOWEST"
                                        )
                                        T.simd.vsts(comm_ub[0], tot_v, mask=m32, dist="ONEPT_B32")
                                        T.simd.mem_bar("VST_VLD")
                                else:
                                    comm_ub[0] = T.float32(0.0)
                                bar_step = bar_step + 1
                                slot = bar_step % num_slots
                                step_f = T.cast(bar_step, "float32")
                                T.ascend_pipe_barrier("PIPE_ALL")
                                comm_ub[1] = step_f
                                T.ascend_pipe_barrier("PIPE_ALL")
                                T.copy(comm_ub[:8], WorkGM[slot, cid, :8])
                                T.ascend_pipe_barrier("PIPE_ALL")
                                done = 0
                                while done == 0:
                                    T.copy(WorkGM[slot, :56, :8], gather_ub[:56, :8])
                                    T.ascend_pipe_barrier("PIPE_ALL")
                                    done = 1
                                    for k in T.serial(num_cores):
                                        if gather_ub[k, 1] < step_f:
                                            done = 0
                                if w_cur > 0:
                                    g_sum = T.float32(0.0)
                                    for k in T.serial(num_cores):
                                        g_sum = g_sum + gather_ub[k, 0]
                                    comm_ub[0] = g_sum * inv_w
                                    T.ascend_pipe_barrier("PIPE_ALL")
                                    with T.SimdVF():
                                        m32 = T.simd.pset(32, "PAT_ALL")
                                        mean_v = T.simd.vld(comm_ub[0], dist="BRC_B32")
                                        for c in T.serial(w_chunks):
                                            T.simd.vsts(
                                                regs_ub[dst, c * 64], mean_v, mask=m32, dist="NORM_B32"
                                            )
                                        if w_rem > 0:
                                            m32_t = T.simd.update_mask(w_rem, width=32)
                                            T.simd.vsts(
                                                regs_ub[dst, w_chunks * 64],
                                                mean_v,
                                                mask=m32_t,
                                                dist="NORM_B32",
                                            )
                                        T.simd.mem_bar("VST_VLD")
                            else:
                                if w_cur > 0:
                                    _exec_elem(regs_ub, op, dst, src_a, src_b, w_chunks, w_rem)
                            T.ascend_pipe_barrier("PIPE_ALL")

                    if w_cur > 0:
                        for r in T.serial(r_count):
                            if tl_dtype == "float32":
                                T.copy(regs_ub[r, :w_cur], FinalRegs[r, w_start : w_start + w_cur])
                            else:
                                with T.SimdVF():
                                    m32 = T.simd.pset(32, "PAT_ALL")
                                    m16 = T.simd.pset(16, "PAT_ALL")
                                    for c in T.serial(w_chunks):
                                        off = c * 64
                                        vf = T.simd.vld(regs_ub[r, off], dist="NORM_B32")
                                        v16 = T.simd.vcvt(vf, tl_dtype, mask=m16, part=0)
                                        T.simd.vsts(io_ub[off], v16, mask=m32, dist="PK_B32", extent=64)
                                    if w_rem > 0:
                                        off = w_chunks * 64
                                        m32_t = T.simd.update_mask(w_rem, width=32)
                                        m16_t = T.simd.update_mask(w_rem * 2, width=16)
                                        vf_t = T.simd.vld(regs_ub[r, off], dist="NORM_B32")
                                        v16_t = T.simd.vcvt(vf_t, tl_dtype, mask=m16_t, part=0)
                                        T.simd.vsts(io_ub[off], v16_t, mask=m32_t, dist="PK_B32", extent=64)
                                T.copy(io_ub[:w_cur], FinalRegs[r, w_start : w_start + w_cur])
                    T.ascend_pipe_barrier("PIPE_ALL")

            return main

        @T.prim_func
        def main_gm(
            Program: T.Tensor((prog_len, 4), "int32"),
            InitRegs: T.Tensor((r_count, w_total), tl_dtype),
            FinalRegs: T.Tensor((r_count, w_total), tl_dtype),
            RegsGM: T.Tensor((r_count, w_total), "float32"),
        ):
            with T.Kernel(1) as _cid:
                row_a_ub = T.alloc_shared((w_blk,), "float32")
                row_b_ub = T.alloc_shared((w_blk,), "float32")
                row_d_ub = T.alloc_shared((w_blk,), "float32")
                io_ub = T.alloc_shared((w_blk,), tl_dtype)
                prog_ub = T.alloc_shared((prog_chunk, 4), "int32")
                T.annotate_buffer_versions({row_a_ub: 1, row_b_ub: 1, row_d_ub: 1, io_ub: 1, prog_ub: 1})
                if schedule_mode == "tuned":
                    T.annotate_unlimit_memory("shared")

                w_chunks = w_total // 64
                w_rem = w_total % 64
                inv_w = T.float32(1.0) / T.cast(w_total, "float32")

                for r in T.serial(r_count):
                    if tl_dtype == "float32":
                        T.copy(InitRegs[r, :w_total], row_a_ub[:w_total])
                    else:
                        T.copy(InitRegs[r, :w_total], io_ub[:w_total])
                        with T.SimdVF():
                            m32 = T.simd.pset(32, "PAT_ALL")
                            m16 = T.simd.pset(16, "PAT_ALL")
                            for c in T.serial(w_chunks):
                                off = c * 64
                                vf = T.simd.vcvt(T.simd.vld(io_ub[off], dist="US_B16"), "float32", mask=m16, part=0)
                                T.simd.vsts(row_a_ub[off], vf, mask=m32, dist="NORM_B32")
                            if w_rem > 0:
                                off = w_chunks * 64
                                m32_t = T.simd.update_mask(w_rem, width=32)
                                m16_t = T.simd.update_mask(w_rem * 2, width=16)
                                vf_t = T.simd.vcvt(
                                    T.simd.vld(io_ub[off], dist="US_B16"), "float32", mask=m16_t, part=0
                                )
                                T.simd.vsts(row_a_ub[off], vf_t, mask=m32_t, dist="NORM_B32")
                    T.copy(row_a_ub[:w_total], RegsGM[r, :w_total])
                T.ascend_pipe_barrier("PIPE_ALL")

                p_blocks = (prog_len + prog_chunk - 1) // prog_chunk
                for pb in T.serial(p_blocks):
                    p0 = pb * prog_chunk
                    p_cur = T.min(prog_chunk, prog_len - p0)
                    T.copy(Program[p0 : p0 + p_cur, :4], prog_ub[:p_cur, :4])
                    T.ascend_pipe_barrier("PIPE_ALL")

                    for lp in T.serial(p_cur):
                        op = prog_ub[lp, 0] % 8
                        dst = prog_ub[lp, 1] % r_count
                        src_a = prog_ub[lp, 2] % r_count
                        src_b = prog_ub[lp, 3] % r_count

                        T.copy(RegsGM[src_a, :w_total], row_a_ub[:w_total])
                        if op == 0 or op == 1 or op == 4 or op == 6:
                            T.copy(RegsGM[src_b, :w_total], row_b_ub[:w_total])
                        if op == 6:
                            T.copy(RegsGM[dst, :w_total], row_d_ub[:w_total])

                        if op == 5:
                            T.ascend_pipe_barrier("PIPE_ALL")
                            row_d_ub[0] = row_a_ub[w_total - 1]
                            for idx in T.serial(w_total - 1):
                                row_d_ub[idx + 1] = row_a_ub[idx]
                            T.ascend_pipe_barrier("PIPE_ALL")
                        elif op == 3:
                            with T.SimdVF():
                                m32 = T.simd.pset(32, "PAT_ALL")
                                v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                                v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
                                sum_v = T.simd.alloc_local((1,), "float32")
                                sum_v[0] = v_zero
                                for c in T.serial(w_chunks):
                                    off = c * 64
                                    sum_v[0] = T.simd.vadd(
                                        sum_v[0], T.simd.vld(row_a_ub[off], dist="NORM_B32"), mask=m32
                                    )
                                if w_rem > 0:
                                    off = w_chunks * 64
                                    m32_t = T.simd.update_mask(w_rem, width=32)
                                    tail_v = T.simd.vmul(T.simd.vld(row_a_ub[off], dist="NORM_B32"), v_one, mask=m32_t)
                                    sum_v[0] = T.simd.vadd(sum_v[0], tail_v, mask=m32)
                                tot_v = T.simd.vdupv(T.simd.vcadd(sum_v[0], mask=m32), mask=m32, pos="POS_LOWEST")
                                mean_v = T.simd.vmuls(tot_v, inv_w, mask=m32)
                                for c in T.serial(w_chunks):
                                    off = c * 64
                                    T.simd.vsts(row_d_ub[off], mean_v, mask=m32, dist="NORM_B32")
                                if w_rem > 0:
                                    off = w_chunks * 64
                                    m32_t = T.simd.update_mask(w_rem, width=32)
                                    T.simd.vsts(row_d_ub[off], mean_v, mask=m32_t, dist="NORM_B32")
                        else:
                            with T.SimdVF():
                                m32 = T.simd.pset(32, "PAT_ALL")
                                v_hi = T.simd.vdup(T.float32(10000.0), "float32", mask=m32)
                                v_lo = T.simd.vdup(T.float32(-10000.0), "float32", mask=m32)
                                v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                                res_reg = T.simd.alloc_local((1,), "float32")
                                for c in T.serial(w_chunks):
                                    off = c * 64
                                    va = T.simd.vld(row_a_ub[off], dist="NORM_B32")
                                    vb = T.simd.vld(row_b_ub[off], dist="NORM_B32")
                                    vd = T.simd.vld(row_d_ub[off], dist="NORM_B32")
                                    res_reg[0] = va
                                    if op == 0:
                                        res_reg[0] = T.simd.vmax(
                                            v_lo, T.simd.vmin(v_hi, T.simd.vadd(va, vb, mask=m32), mask=m32), mask=m32
                                        )
                                    elif op == 1:
                                        res_reg[0] = T.simd.vmax(
                                            v_lo, T.simd.vmin(v_hi, T.simd.vmul(va, vb, mask=m32), mask=m32), mask=m32
                                        )
                                    elif op == 2:
                                        res_reg[0] = T.simd.vmax(va, v_zero, mask=m32)
                                    elif op == 4:
                                        res_reg[0] = T.simd.vmax(va, vb, mask=m32)
                                    elif op == 6:
                                        res_reg[0] = T.simd.vmax(
                                            v_lo,
                                            T.simd.vmin(
                                                v_hi,
                                                T.simd.vadd(vd, T.simd.vmul(va, vb, mask=m32), mask=m32),
                                                mask=m32,
                                            ),
                                            mask=m32,
                                        )
                                    T.simd.vsts(row_d_ub[off], res_reg[0], mask=m32, dist="NORM_B32")
                                if w_rem > 0:
                                    off = w_chunks * 64
                                    m32_t = T.simd.update_mask(w_rem, width=32)
                                    va_t = T.simd.vld(row_a_ub[off], dist="NORM_B32")
                                    vb_t = T.simd.vld(row_b_ub[off], dist="NORM_B32")
                                    vd_t = T.simd.vld(row_d_ub[off], dist="NORM_B32")
                                    res_reg[0] = va_t
                                    if op == 0:
                                        res_reg[0] = T.simd.vmax(
                                            v_lo,
                                            T.simd.vmin(v_hi, T.simd.vadd(va_t, vb_t, mask=m32_t), mask=m32_t),
                                            mask=m32_t,
                                        )
                                    elif op == 1:
                                        res_reg[0] = T.simd.vmax(
                                            v_lo,
                                            T.simd.vmin(v_hi, T.simd.vmul(va_t, vb_t, mask=m32_t), mask=m32_t),
                                            mask=m32_t,
                                        )
                                    elif op == 2:
                                        res_reg[0] = T.simd.vmax(va_t, v_zero, mask=m32_t)
                                    elif op == 4:
                                        res_reg[0] = T.simd.vmax(va_t, vb_t, mask=m32_t)
                                    elif op == 6:
                                        res_reg[0] = T.simd.vmax(
                                            v_lo,
                                            T.simd.vmin(
                                                v_hi,
                                                T.simd.vadd(vd_t, T.simd.vmul(va_t, vb_t, mask=m32_t), mask=m32_t),
                                                mask=m32_t,
                                            ),
                                            mask=m32_t,
                                        )
                                    T.simd.vsts(row_d_ub[off], res_reg[0], mask=m32_t, dist="NORM_B32")

                        T.copy(row_d_ub[:w_total], RegsGM[dst, :w_total])
                        T.ascend_pipe_barrier("PIPE_ALL")

                for r in T.serial(r_count):
                    T.copy(RegsGM[r, :w_total], row_a_ub[:w_total])
                    if tl_dtype == "float32":
                        T.copy(row_a_ub[:w_total], FinalRegs[r, :w_total])
                    else:
                        with T.SimdVF():
                            m32 = T.simd.pset(32, "PAT_ALL")
                            m16 = T.simd.pset(16, "PAT_ALL")
                            for c in T.serial(w_chunks):
                                off = c * 64
                                vf = T.simd.vld(row_a_ub[off], dist="NORM_B32")
                                v16 = T.simd.vcvt(vf, tl_dtype, mask=m16, part=0)
                                T.simd.vsts(io_ub[off], v16, mask=m32, dist="PK_B32", extent=64)
                            if w_rem > 0:
                                off = w_chunks * 64
                                m32_t = T.simd.update_mask(w_rem, width=32)
                                m16_t = T.simd.update_mask(w_rem * 2, width=16)
                                vf_t = T.simd.vld(row_a_ub[off], dist="NORM_B32")
                                v16_t = T.simd.vcvt(vf_t, tl_dtype, mask=m16_t, part=0)
                                T.simd.vsts(io_ub[off], v16_t, mask=m32_t, dist="PK_B32", extent=64)
                        T.copy(io_ub[:w_total], FinalRegs[r, :w_total])
                T.ascend_pipe_barrier("PIPE_ALL")

        return main_gm


def _get_kernel(num_regs: int, w_dim: int, tl_dtype: str, prog_len: int, schedule_mode: str):
    cfg = _select_tensor_program_vm_config(
        num_regs=num_regs,
        w_dim=w_dim,
        tl_dtype=tl_dtype,
        prog_len=prog_len,
        schedule_mode=schedule_mode,
    )
    use_gm = bool(cfg["use_gm_workspace"])
    key = (num_regs, int(cfg["w_slice"]), tl_dtype, use_gm, schedule_mode)
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        with _ascend_target_ctx():
            prog = _build_tensor_program_vm_prim_func(
                num_regs=num_regs,
                w_slice=int(cfg["w_slice"]),
                tl_dtype=tl_dtype,
                use_gm_workspace=use_gm,
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
    return compiled, use_gm


def tensor_program_vm(
    program: torch.Tensor,
    init_regs: torch.Tensor,
    *,
    schedule_mode: str = "tuned",
) -> torch.Tensor:
    """Execute TensorProgramVm matching cannbot-ops level5/tensor_program_vm proto.yaml."""
    if init_regs.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported dtype {init_regs.dtype}")

    if init_regs.device.type != "npu":
        prog = program.to(torch.int64)
        p_len = prog.shape[0]
        r_count, w_dim = init_regs.shape
        op_v = (prog[:, 0] % 8).tolist()
        dst_v = (prog[:, 1] % r_count).tolist()
        a_v = (prog[:, 2] % r_count).tolist()
        b_v = (prog[:, 3] % r_count).tolist()
        regs = init_regs.to(torch.float32).clone()
        lo, hi, zero = -10000.0, 10000.0, 0.0
        for t in range(p_len):
            op, dst, a, b = op_v[t], dst_v[t], a_v[t], b_v[t]
            if op == 0:
                regs[dst] = torch.clamp(regs[a] + regs[b], lo, hi)
            elif op == 1:
                regs[dst] = torch.clamp(regs[a] * regs[b], lo, hi)
            elif op == 2:
                regs[dst] = torch.clamp(regs[a], min=zero)
            elif op == 3:
                regs[dst] = regs[a].mean().expand(w_dim)
            elif op == 4:
                regs[dst] = torch.maximum(regs[a], regs[b])
            elif op == 5:
                regs[dst] = torch.roll(regs[a], shifts=1, dims=0)
            elif op == 6:
                regs[dst] = torch.clamp(regs[dst] + regs[a] * regs[b], lo, hi)
            else:
                regs[dst] = regs[a].clone()
        return regs.to(init_regs.dtype)

    tl_dtype = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }[init_regs.dtype]
    r_count, w_dim = init_regs.shape
    compiled, use_gm = _get_kernel(
        num_regs=r_count,
        w_dim=w_dim,
        tl_dtype=tl_dtype,
        prog_len=program.shape[0],
        schedule_mode=schedule_mode,
    )
    if use_gm:
        regs_gm = _get_regs_workspace(r_count, w_dim, init_regs.device)
        return compiled(program.contiguous(), init_regs.contiguous(), regs_gm)
    work_gm = _get_sync_workspace(init_regs.device)
    work_gm.zero_()
    return compiled(program.contiguous(), init_regs.contiguous(), work_gm)


kernel = tensor_program_vm

__all__ = [
    "_build_tensor_program_vm_prim_func",
    "_select_tensor_program_vm_config",
    "kernel",
    "tensor_program_vm",
]
