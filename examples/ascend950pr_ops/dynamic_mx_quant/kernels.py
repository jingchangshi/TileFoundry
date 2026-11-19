"""Ascend 950PR 56-AIV SimdVF DynamicMxQuant across FP8/FP4, arbitrary axis, and rounding modes."""

import contextlib
import math
import tilelang
import tilelang.ascend.language as T
import torch

_KERNEL_CACHE: dict[tuple[object, ...], object] = {}

_DST_TYPE_INFO = {
    35: ("float8_e5m2", 15, 5, 2, 15, 57344.0, False),
    36: ("float8_e4m3fn", 8, 4, 3, 7, 448.0, False),
    40: ("float4_e2m1", 2, 2, 1, 1, 6.0, True),
    41: ("float4_e1m2", 0, 1, 2, 0, 1.75, True),
}

_ROUND_MODES = ("rint", "floor", "round")


def _ascend_target_ctx():
    try:
        from tilelang import tvm
        from tilelang.backend.module import create_backend_context

        if tvm.target.Target.current(allow_none=True) is None:
            return create_backend_context("ascend").target
    except Exception:
        pass
    return contextlib.nullcontext()


def _select_dynamic_mx_quant_config(
    dst_type: int = 40,
    round_mode: str = "rint",
    blocksize: int = 32,
    is_tail_axis: bool = True,
    input_dtype: str = "float16",
    *,
    outer_dim: int = 1024,
    axis_dim: int = 1024,
    inner_dim: int = 1,
    schedule_mode: str = "tuned",
) -> dict[str, object]:
    bs = int(blocksize)
    dt = int(dst_type)
    tail = bool(is_tail_axis)
    od = max(1, int(outer_dim))
    axd = max(1, int(axis_dim))
    ind = max(1, int(inner_dim))
    if tail:
        if schedule_mode == "tuned" and bs == 32:
            n_blks_est = (axd + bs - 1) // bs
            total_blks = od * n_blks_est if (axd % bs == 0) else n_blks_est
            if (axd % bs == 0) and total_blks >= 56 * 256:
                blks_per_chunk = 256
            elif (axd % bs == 0 or n_blks_est >= 128) and total_blks >= 56 * 128:
                blks_per_chunk = 128
            else:
                blks_per_chunk = 64
        else:
            blks_per_chunk = 64 if bs <= 64 else 32
        col_tile = 64
    else:
        blks_per_chunk = 64 if bs <= 64 else 32
        n_blks_est = (axd + bs - 1) // bs
        col_tile = 64
        if schedule_mode == "tuned":
            for cand in (512, 256, 128, 64):
                ub_est = bs * (cand + 64) * 3
                if ub_est <= 110000 and ind >= cand and (od * n_blks_est * ((ind + cand - 1) // cand)) >= 56:
                    col_tile = cand
                    break
    use_pipelined = bool(schedule_mode == "tuned" and tail)
    return {
        "dst_type": dt,
        "round_mode": str(round_mode),
        "blocksize": bs,
        "is_tail_axis": tail,
        "input_dtype": str(input_dtype),
        "blks_per_chunk": blks_per_chunk,
        "col_tile": col_tile,
        "num_cores": 56,
        "use_pipelined": use_pipelined,
        "compile_flags": ["--cce-disable-asc-reserved-ubuf"] if schedule_mode == "tuned" else [],
    }


def _emit_quant_vec(
    v_val,
    v_inv_scale_f32,
    v_is_bad_blk,
    *,
    ele_emax: int,
    mant_bits: int,
    min_exp: int,
    fp8_bias: int,
    max_norm: float,
    is_fp4: bool,
    dst_type: int,
    round_mode: str,
    m32,
    v_zero,
    v_abs_mask,
    v_mant_mask,
    v_implicit1,
    v_i0,
    v_i1,
    v_i8,
    v_half,
):
    """Quantize and encode 64 float32 values into 64 FP8/FP4 uint8 codes (in int32x64)."""
    ret_raw = T.simd.vmul(v_val, v_inv_scale_f32, mask=m32)
    ret = T.simd.vsel(v_zero, ret_raw, v_is_bad_blk)

    ret_i = T.reinterpret("int32x64", ret)
    abs_ret_i = T.simd.vand(ret_i, v_abs_mask, mask=m32)
    priv_exp = T.simd.vmaxs(T.simd.vadds(T.simd.vshrs(abs_ret_i, 23, mask=m32), -127, mask=m32), min_exp, mask=m32)

    inv_step_i = T.simd.vshls(T.simd.vsub(T.simd.vdup(127 + mant_bits, "int32", mask=m32), priv_exp, mask=m32), 23, mask=m32)
    step_i = T.simd.vshls(T.simd.vadds(priv_exp, 127 - mant_bits, mask=m32), 23, mask=m32)

    scaled = T.simd.vmul(ret, T.reinterpret("float32x64", inv_step_i), mask=m32)
    if round_mode == "rint":
        r_i = T.simd.vcvt(scaled, "int32", mask=m32, round="ROUND_R", sat=True)
        r_f = T.simd.vcvt(r_i, "float32", mask=m32)
    elif round_mode == "floor":
        r_i = T.simd.vcvt(scaled, "int32", mask=m32, round="ROUND_F", sat=True)
        r_f = T.simd.vcvt(r_i, "float32", mask=m32)
    else:
        abs_s = T.simd.vabs(scaled, mask=m32)
        mag_i = T.simd.vcvt(T.simd.vadds(abs_s, T.float32(0.5), mask=m32), "int32", mask=m32, round="ROUND_F", sat=True)
        mag_f = T.simd.vcvt(mag_i, "float32", mask=m32)
        is_neg = T.simd.vcmps(scaled, T.float32(0.0), mask=m32, op="lt")
        r_f = T.simd.vsel(T.simd.vmuls(mag_f, T.float32(-1.0), mask=m32), mag_f, is_neg)

    q_raw = T.simd.vmul(r_f, T.reinterpret("float32x64", step_i), mask=m32)
    q_clamped = T.simd.vmaxs(T.simd.vmins(q_raw, T.float32(max_norm), mask=m32), T.float32(-max_norm), mask=m32)
    q = T.simd.vsel(v_zero, q_clamped, v_is_bad_blk)

    q_i = T.reinterpret("int32x64", q)
    abs_qi = T.simd.vand(q_i, v_abs_mask, mask=m32)

    if is_fp4:
        sign_nibble = T.simd.vand(T.simd.vshrs(ret_i, 28, mask=m32), v_i8, mask=m32)
        q_abs = T.reinterpret("float32x64", abs_qi)
        if dst_type == 41:
            mag_code = T.simd.vcvt(T.simd.vmuls(q_abs, T.float32(4.0), mask=m32), "int32", mask=m32, round="ROUND_R", sat=True)
        else:
            q_abs_i = abs_qi
            is_z = T.simd.vcmp(q_abs, v_zero, mask=m32, op="eq")
            is_h = T.simd.vcmp(q_abs, v_half, mask=m32, op="eq")
            q_exp = T.simd.vadds(T.simd.vshrs(q_abs_i, 23, mask=m32), -126, mask=m32)
            q_m1 = T.simd.vand(T.simd.vshrs(q_abs_i, 22, mask=m32), v_i1, mask=m32)
            norm_code = T.simd.vor(T.simd.vshls(q_exp, 1, mask=m32), q_m1, mask=m32)
            mag_code = T.simd.vsel(v_i0, T.simd.vsel(v_i1, norm_code, is_h), is_z)
        return T.simd.vor(mag_code, sign_nibble, mask=m32)

    sign_b = T.simd.vshls(T.simd.vand(T.simd.vshrs(ret_i, 31, mask=m32), v_i1, mask=m32), 7, mask=m32)
    mant_q = T.simd.vand(abs_qi, v_mant_mask, mask=m32)
    exp_q = T.simd.vadds(T.simd.vshrs(abs_qi, 23, mask=m32), -127, mask=m32)
    normal_b = T.simd.vor(
        T.simd.vshls(T.simd.vadds(T.simd.vmaxs(exp_q, min_exp, mask=m32), fp8_bias, mask=m32), mant_bits, mask=m32),
        T.simd.vshrs(mant_q, 23 - mant_bits, mask=m32),
        mask=m32,
    )
    is_zero_qi = T.simd.vcmps(abs_qi, 0, mask=m32, op="eq")
    sub_shift = T.simd.vmins(
        T.simd.vmaxs(T.simd.vsub(T.simd.vdup(23 + min_exp - mant_bits, "int32", mask=m32), exp_q, mask=m32), 0, mask=m32),
        31,
        mask=m32,
    )
    sub_raw = T.simd.vshr(T.simd.vor(v_implicit1, mant_q, mask=m32), sub_shift, mask=m32)
    sub_b = T.simd.vsel(v_i0, sub_raw, is_zero_qi)
    is_norm = T.simd.vcmps(exp_q, min_exp, mask=m32, op="ge")
    return T.simd.vor(sign_b, T.simd.vsel(normal_b, sub_b, is_norm), mask=m32)


def _emit_nontail_quant_64(
    x_ub,
    inv_s_ub,
    r_idx,
    col_off,
    cur_cols,
    *,
    ele_emax: int,
    mant_bits: int,
    min_exp: int,
    fp8_bias: int,
    max_norm: float,
    is_fp4: bool,
    dst_type: int,
    round_mode: str,
    m32,
    m16,
    v_zero,
    lane,
):
    m_seg = T.simd.vcmps(lane, T.min(64, cur_cols - col_off), mask=m32, op="lt")
    raw_f = T.simd.vcvt(T.simd.vld(x_ub[r_idx, col_off], dist="US_B16"), "float32", mask=m16, part=0)
    v_val = T.simd.vsel(raw_f, v_zero, m_seg)
    inv_s = T.simd.vld(inv_s_ub[col_off], dist="NORM_B32")
    is_bad = T.simd.vcmps(inv_s, T.float32(0.0), mask=m32, op="eq")
    return _emit_quant_vec(
        v_val,
        inv_s,
        is_bad,
        ele_emax=ele_emax,
        mant_bits=mant_bits,
        min_exp=min_exp,
        fp8_bias=fp8_bias,
        max_norm=max_norm,
        is_fp4=is_fp4,
        dst_type=dst_type,
        round_mode=round_mode,
        m32=m32,
        v_zero=v_zero,
        v_abs_mask=T.simd.vdup(0x7FFFFFFF, "int32", mask=m32),
        v_mant_mask=T.simd.vdup(0x7FFFFF, "int32", mask=m32),
        v_implicit1=T.simd.vdup(1 << 23, "int32", mask=m32),
        v_i0=T.simd.vdup(0, "int32", mask=m32),
        v_i1=T.simd.vdup(1, "int32", mask=m32),
        v_i8=T.simd.vdup(8, "int32", mask=m32),
        v_half=T.simd.vdup(T.float32(0.5), "float32", mask=m32),
    )


def _emit_scale_from_amax(
    v_amax,
    *,
    ele_emax: int,
    m32,
    v_zero,
    v_abs_mask,
    v_i0,
    v_i255,
):
    """Compute (inv_mxscale_val_f32, mxscale_u8_int32, is_bad_blk_mask) from block amax."""
    abs_amax_i = T.simd.vand(T.reinterpret("int32x64", v_amax), v_abs_mask, mask=m32)
    raw_exp = T.simd.vadds(T.simd.vshrs(abs_amax_i, 23, mask=m32), -(127 + ele_emax), mask=m32)
    is_zero_blk = T.simd.vcmp(v_amax, v_zero, mask=m32, op="eq")
    is_ovf = T.simd.vcmps(raw_exp, 127, mask=m32, op="gt")
    is_bad = T.simd.por(is_zero_blk, is_ovf, m32)

    is_sub = T.simd.vcmps(raw_exp, -127, mask=m32, op="le")
    clamped_e = T.simd.vmins(T.simd.vmaxs(raw_exp, -126, mask=m32), 127, mask=m32)
    biased_e = T.simd.vadds(clamped_e, 127, mask=m32)
    byte_norm = T.simd.vsel(v_i0, biased_e, is_sub)
    v_scale_u8_i = T.simd.vsel(v_i0, T.simd.vsel(v_i255, byte_norm, is_ovf), is_zero_blk)

    e_full = T.simd.vmins(T.simd.vmaxs(raw_exp, -127, mask=m32), 127, mask=m32)
    inv_biased = T.simd.vsub(T.simd.vdup(127, "int32", mask=m32), e_full, mask=m32)
    inv_norm_bits = T.simd.vshls(T.simd.vmaxs(inv_biased, 1, mask=m32), 23, mask=m32)
    inv_bits = T.simd.vsel(
        T.simd.vdup(0x00400000, "int32", mask=m32),
        inv_norm_bits,
        T.simd.vcmps(inv_biased, 0, mask=m32, op="le"),
    )
    inv_bits = T.simd.vsel(v_i0, inv_bits, is_bad)
    v_inv_scale_f32 = T.reinterpret("float32x64", inv_bits)
    return v_inv_scale_f32, v_scale_u8_i, is_bad


def _build_dynamic_mx_quant_prim_func(
    dst_type: int = 40,
    round_mode: str = "rint",
    blocksize: int = 32,
    is_tail_axis: bool = True,
    input_dtype: str = "float16",
    blks_per_chunk: int = 64,
    col_tile: int = 64,
    num_cores: int = 56,
    use_pipelined: bool = False,
    schedule_mode: str = "tuned",
    **_extra_kwargs,
):
    dt = int(dst_type)
    dtype_name, ele_emax, exp_bits, mant_bits, fp8_bias, max_norm, is_fp4 = _DST_TYPE_INFO[dt]
    min_exp = -(2 ** (exp_bits - 1)) + exp_bits if is_fp4 else -(2 ** (exp_bits - 1)) + 2
    bs = int(blocksize)
    in_tl_dtype = str(input_dtype)
    num_cores = max(1, int(num_cores))
    rm = str(round_mode)

    with _ascend_target_ctx():
        if is_tail_axis:
            c_blks = int(blks_per_chunk)
            chunk_elems = c_blks * bs
            out_chunk_cols = chunk_elems // 2 if is_fp4 else chunk_elems
            stages = 2 if (use_pipelined and schedule_mode == "tuned") else 1

            outer_dyn = T.dynamic("outer_dyn")
            axis_dyn = T.dynamic("axis_dyn")
            out_cols_expr = axis_dyn // 2 if is_fp4 else axis_dyn
            n_blks_expr = (axis_dyn + bs - 1) // bs

            @T.prim_func
            def main_tail(
                X: T.Tensor((outer_dyn, axis_dyn), in_tl_dtype),
                Y: T.Tensor((outer_dyn, out_cols_expr), "uint8"),
                Scale: T.Tensor((outer_dyn, n_blks_expr), "uint8"),
                outer_dim: T.int32,
                axis_dim: T.int32,
                n_blks: T.int32,
            ):
                with T.Kernel(num_cores) as cid:
                    x_ub = T.alloc_shared((chunk_elems + 64,), in_tl_dtype)
                    y_ub = T.alloc_shared((out_chunk_cols + 64,), "uint8")
                    s_ub = T.alloc_shared((c_blks + 64,), "uint8")
                    if stages > 1:
                        T.annotate_buffer_versions({x_ub: 2, y_ub: 2, s_ub: 2})
                    else:
                        T.annotate_buffer_versions({x_ub: 1, y_ub: 1, s_ub: 1})
                    if schedule_mode == "tuned":
                        T.annotate_unlimit_memory("shared")

                    chunks_per_row = (n_blks + c_blks - 1) // c_blks
                    total_tasks = outer_dim * chunks_per_row
                    base_t = total_tasks // num_cores
                    rem_t = total_tasks % num_cores
                    t_start = cid * base_t + T.min(cid, rem_t)
                    t_count = base_t + T.if_then_else(cid < rem_t, 1, 0)

                    for t_i in T.Pipelined(t_count, num_stages=stages):
                        tid = t_start + t_i
                        row = tid // chunks_per_row
                        ch = tid % chunks_per_row
                        blk0 = ch * c_blks
                        elem0 = blk0 * bs
                        cur_blks = T.min(c_blks, n_blks - blk0)
                        cur_elems = T.min(chunk_elems, axis_dim - elem0)
                        cur_out_cols = cur_elems // 2 if is_fp4 else cur_elems

                        T.copy(X[row, elem0 : elem0 + cur_elems], x_ub[:cur_elems])

                        with T.SimdVF():
                            m32 = T.simd.pset(32, "PAT_ALL")
                            m16 = T.simd.pset(16, "PAT_ALL")
                            m8 = T.simd.pset(8, "PAT_ALL")
                            v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                            lane = T.simd.vci(0, "int32")
                            s_vec = T.simd.alloc_local((1,), "int32")
                            pk_even_acc = T.simd.alloc_local((1,), "int32")

                            if bs == 32:
                                n_seg = (cur_blks + 63) // 64
                                for seg_b in T.serial(n_seg):
                                    s_vec[0] = T.simd.vdup(0, "int32", mask=m32)
                                    seg_blk0 = seg_b * 64
                                    seg_off = seg_blk0 * 32
                                    seg_blks = T.min(64, cur_blks - seg_blk0)
                                    pairs = (seg_blks + 1) // 2
                                    for p in T.serial(pairs):
                                        off = seg_off + p * 64
                                        valid_cnt = T.min(64, cur_elems - off)
                                        m_val = T.simd.vcmps(lane, valid_cnt, mask=m32, op="lt")
                                        raw_f = T.simd.vcvt(T.simd.vld(x_ub[off], dist="US_B16"), "float32", mask=m16, part=0)
                                        v_val = T.simd.vsel(raw_f, v_zero, m_val)
                                        a = T.simd.vabs(v_val, mask=m32)
                                        a = T.simd.vmax(a, T.simd.vselr(a, T.simd.vxor(lane, T.simd.vdup(1, "int32", mask=m32), mask=m32)), mask=m32)
                                        a = T.simd.vmax(a, T.simd.vselr(a, T.simd.vxor(lane, T.simd.vdup(2, "int32", mask=m32), mask=m32)), mask=m32)
                                        a = T.simd.vmax(a, T.simd.vselr(a, T.simd.vxor(lane, T.simd.vdup(4, "int32", mask=m32), mask=m32)), mask=m32)
                                        a = T.simd.vmax(a, T.simd.vselr(a, T.simd.vxor(lane, T.simd.vdup(8, "int32", mask=m32), mask=m32)), mask=m32)
                                        v_amax = T.simd.vmax(a, T.simd.vselr(a, T.simd.vxor(lane, T.simd.vdup(16, "int32", mask=m32), mask=m32)), mask=m32)

                                        v_i0 = T.simd.vdup(0, "int32", mask=m32)
                                        v_abs_mask = T.simd.vdup(0x7FFFFFFF, "int32", mask=m32)
                                        v_inv_scale_f32, v_scale_u8_i, is_bad = _emit_scale_from_amax(
                                            v_amax,
                                            ele_emax=ele_emax,
                                            m32=m32,
                                            v_zero=v_zero,
                                            v_abs_mask=v_abs_mask,
                                            v_i0=v_i0,
                                            v_i255=T.simd.vdup(255, "int32", mask=m32),
                                        )
                                        s_pair = T.simd.vselr(v_scale_u8_i, T.simd.vmuls(lane, 32, mask=m32))
                                        s_bcast = T.simd.vselr(s_pair, T.simd.vand(lane, T.simd.vdup(1, "int32", mask=m32), mask=m32))
                                        m_pair = T.simd.pand(
                                            T.simd.vcmps(lane, p * 2, mask=m32, op="ge"),
                                            T.simd.vcmps(lane, p * 2 + 2, mask=m32, op="lt"),
                                            m32,
                                        )
                                        s_vec[0] = T.simd.vsel(s_bcast, s_vec[0], m_pair)

                                        codes = _emit_quant_vec(
                                            v_val,
                                            v_inv_scale_f32,
                                            is_bad,
                                            ele_emax=ele_emax,
                                            mant_bits=mant_bits,
                                            min_exp=min_exp,
                                            fp8_bias=fp8_bias,
                                            max_norm=max_norm,
                                            is_fp4=is_fp4,
                                            dst_type=dt,
                                            round_mode=rm,
                                            m32=m32,
                                            v_zero=v_zero,
                                            v_abs_mask=v_abs_mask,
                                            v_mant_mask=T.simd.vdup(0x7FFFFF, "int32", mask=m32),
                                            v_implicit1=T.simd.vdup(1 << 23, "int32", mask=m32),
                                            v_i0=v_i0,
                                            v_i1=T.simd.vdup(1, "int32", mask=m32),
                                            v_i8=T.simd.vdup(8, "int32", mask=m32),
                                            v_half=T.simd.vdup(T.float32(0.5), "float32", mask=m32),
                                        )
                                        if is_fp4:
                                            m_hi0 = T.simd.vcmps(lane, 32, mask=m32, op="lt")
                                            idx_even = T.simd.vmuls(lane, 2, mask=m32)
                                            idx_odd = T.simd.vadds(idx_even, 1, mask=m32)
                                            ev = T.simd.vselr(codes, T.simd.vsel(idx_even, v_i0, m_hi0))
                                            od = T.simd.vselr(codes, T.simd.vsel(idx_odd, v_i0, m_hi0))
                                            pk = T.simd.vor(ev, T.simd.vshls(od, 4, mask=m32), mask=m32)
                                            if (p % 2) == 0:
                                                pk_even_acc[0] = pk
                                            else:
                                                lane_mod32 = T.simd.vand(lane, T.simd.vdup(31, "int32", mask=m32), mask=m32)
                                                pk_hi = T.simd.vselr(pk, lane_mod32)
                                                pk64 = T.simd.vsel(pk_even_acc[0], pk_hi, m_hi0)
                                                pk16 = T.simd.vcvt(pk64, "uint16", mask=m16, round="ROUND_R", sat=False, part=0)
                                                pk8 = T.simd.vcvt(pk16, "uint8", mask=m8, round="ROUND_R", sat=False, part=0)
                                                T.simd.vsts(y_ub[seg_off // 2 + (p // 2) * 64], pk8, mask=m32, dist="PK4_B32", extent=64)
                                        else:
                                            y16 = T.simd.vcvt(codes, "uint16", mask=m16, round="ROUND_R", sat=False, part=0)
                                            y8 = T.simd.vcvt(y16, "uint8", mask=m8, round="ROUND_R", sat=False, part=0)
                                            T.simd.vsts(y_ub[off], y8, mask=m32, dist="PK4_B32", extent=64)

                                    if is_fp4 and ((pairs % 2) == 1):
                                        pk16_tail = T.simd.vcvt(pk_even_acc[0], "uint16", mask=m16, round="ROUND_R", sat=False, part=0)
                                        pk8_tail = T.simd.vcvt(pk16_tail, "uint8", mask=m8, round="ROUND_R", sat=False, part=0)
                                        T.simd.vsts(y_ub[seg_off // 2 + (pairs // 2) * 64], pk8_tail, mask=m32, dist="PK4_B32", extent=64)

                                    s_u16 = T.simd.vcvt(s_vec[0], "uint16", mask=m16, round="ROUND_R", sat=False, part=0)
                                    s_u8 = T.simd.vcvt(s_u16, "uint8", mask=m8, round="ROUND_R", sat=False, part=0)
                                    T.simd.vsts(s_ub[seg_blk0], s_u8, mask=m32, dist="PK4_B32", extent=64)
                            else:
                                blks_64 = bs // 64
                                amax_acc = T.simd.alloc_local((1,), "float32")
                                s_vec[0] = T.simd.vdup(0, "int32", mask=m32)

                                for b_i in T.serial(cur_blks):
                                    b_off = b_i * bs
                                    amax_acc[0] = v_zero
                                    for s_i in T.serial(blks_64):
                                        off = b_off + s_i * 64
                                        valid_cnt = T.min(64, cur_elems - off)
                                        m_val = T.simd.vcmps(lane, valid_cnt, mask=m32, op="lt")
                                        raw_f = T.simd.vcvt(T.simd.vld(x_ub[off], dist="US_B16"), "float32", mask=m16, part=0)
                                        v_val = T.simd.vsel(raw_f, v_zero, m_val)
                                        amax_acc[0] = T.simd.vmax(amax_acc[0], T.simd.vabs(v_val, mask=m32), mask=m32)

                                    v_amax = T.simd.vdupv(T.simd.vcmax(amax_acc[0], mask=m32), mask=m32, pos="POS_LOWEST")
                                    v_i0 = T.simd.vdup(0, "int32", mask=m32)
                                    v_abs_mask = T.simd.vdup(0x7FFFFFFF, "int32", mask=m32)
                                    v_inv_scale_f32, v_scale_u8_i, is_bad = _emit_scale_from_amax(
                                        v_amax,
                                        ele_emax=ele_emax,
                                        m32=m32,
                                        v_zero=v_zero,
                                        v_abs_mask=v_abs_mask,
                                        v_i0=v_i0,
                                        v_i255=T.simd.vdup(255, "int32", mask=m32),
                                    )
                                    m_bi = T.simd.vcmps(lane, b_i, mask=m32, op="eq")
                                    s_vec[0] = T.simd.vsel(v_scale_u8_i, s_vec[0], m_bi)

                                    for s_i in T.serial(blks_64):
                                        off = b_off + s_i * 64
                                        valid_cnt = T.min(64, cur_elems - off)
                                        m_val = T.simd.vcmps(lane, valid_cnt, mask=m32, op="lt")
                                        raw_f = T.simd.vcvt(T.simd.vld(x_ub[off], dist="US_B16"), "float32", mask=m16, part=0)
                                        v_val = T.simd.vsel(raw_f, v_zero, m_val)
                                        codes = _emit_quant_vec(
                                            v_val,
                                            v_inv_scale_f32,
                                            is_bad,
                                            ele_emax=ele_emax,
                                            mant_bits=mant_bits,
                                            min_exp=min_exp,
                                            fp8_bias=fp8_bias,
                                            max_norm=max_norm,
                                            is_fp4=is_fp4,
                                            dst_type=dt,
                                            round_mode=rm,
                                            m32=m32,
                                            v_zero=v_zero,
                                            v_abs_mask=T.simd.vdup(0x7FFFFFFF, "int32", mask=m32),
                                            v_mant_mask=T.simd.vdup(0x7FFFFF, "int32", mask=m32),
                                            v_implicit1=T.simd.vdup(1 << 23, "int32", mask=m32),
                                            v_i0=T.simd.vdup(0, "int32", mask=m32),
                                            v_i1=T.simd.vdup(1, "int32", mask=m32),
                                            v_i8=T.simd.vdup(8, "int32", mask=m32),
                                            v_half=T.simd.vdup(T.float32(0.5), "float32", mask=m32),
                                        )
                                        if is_fp4:
                                            m_hi0 = T.simd.vcmps(lane, 32, mask=m32, op="lt")
                                            idx_even = T.simd.vmuls(lane, 2, mask=m32)
                                            idx_odd = T.simd.vadds(idx_even, 1, mask=m32)
                                            v_i0_inner = T.simd.vdup(0, "int32", mask=m32)
                                            ev = T.simd.vselr(codes, T.simd.vsel(idx_even, v_i0_inner, m_hi0))
                                            od = T.simd.vselr(codes, T.simd.vsel(idx_odd, v_i0_inner, m_hi0))
                                            pk = T.simd.vor(ev, T.simd.vshls(od, 4, mask=m32), mask=m32)
                                            k_chunk = b_i * blks_64 + s_i
                                            if (k_chunk % 2) == 0:
                                                pk_even_acc[0] = pk
                                            else:
                                                lane_mod32 = T.simd.vand(lane, T.simd.vdup(31, "int32", mask=m32), mask=m32)
                                                pk_hi = T.simd.vselr(pk, lane_mod32)
                                                pk64 = T.simd.vsel(pk_even_acc[0], pk_hi, m_hi0)
                                                pk16 = T.simd.vcvt(pk64, "uint16", mask=m16, round="ROUND_R", sat=False, part=0)
                                                pk8 = T.simd.vcvt(pk16, "uint8", mask=m8, round="ROUND_R", sat=False, part=0)
                                                T.simd.vsts(y_ub[(k_chunk // 2) * 64], pk8, mask=m32, dist="PK4_B32", extent=64)
                                        else:
                                            y16 = T.simd.vcvt(codes, "uint16", mask=m16, round="ROUND_R", sat=False, part=0)
                                            y8 = T.simd.vcvt(y16, "uint8", mask=m8, round="ROUND_R", sat=False, part=0)
                                            T.simd.vsts(y_ub[off], y8, mask=m32, dist="PK4_B32", extent=64)

                                if is_fp4 and (((cur_blks * blks_64) % 2) == 1):
                                    pk16_tail = T.simd.vcvt(pk_even_acc[0], "uint16", mask=m16, round="ROUND_R", sat=False, part=0)
                                    pk8_tail = T.simd.vcvt(pk16_tail, "uint8", mask=m8, round="ROUND_R", sat=False, part=0)
                                    T.simd.vsts(y_ub[((cur_blks * blks_64) // 2) * 64], pk8_tail, mask=m32, dist="PK4_B32", extent=64)

                                s_u16 = T.simd.vcvt(s_vec[0], "uint16", mask=m16, round="ROUND_R", sat=False, part=0)
                                s_u8 = T.simd.vcvt(s_u16, "uint8", mask=m8, round="ROUND_R", sat=False, part=0)
                                T.simd.vsts(s_ub[0], s_u8, mask=m32, dist="PK4_B32", extent=64)

                        T.copy(y_ub[:cur_out_cols], Y[row, (elem0 // 2 if is_fp4 else elem0) : (elem0 // 2 if is_fp4 else elem0) + cur_out_cols])
                        T.copy(s_ub[:cur_blks], Scale[row, blk0 : blk0 + cur_blks])

            return main_tail

        ct = int(col_tile)
        out_ct = ct // 2 if is_fp4 else ct
        y_ub_cols = ((out_ct + 63) // 64) * 64 + 64
        outer_dyn = T.dynamic("outer_dyn")
        axis_dyn = T.dynamic("axis_dyn")
        inner_dyn = T.dynamic("inner_dyn")
        out_inner_expr = inner_dyn // 2 if is_fp4 else inner_dyn
        n_blks_expr = (axis_dyn + bs - 1) // bs

        @T.prim_func
        def main_nontail(
            X: T.Tensor((outer_dyn, axis_dyn, inner_dyn), in_tl_dtype),
            Y: T.Tensor((outer_dyn, axis_dyn, out_inner_expr), "uint8"),
            Scale: T.Tensor((outer_dyn, n_blks_expr, inner_dyn), "uint8"),
            outer_dim: T.int32,
            axis_dim: T.int32,
            inner_dim: T.int32,
            n_blks: T.int32,
        ):
            with T.Kernel(num_cores) as cid:
                x_ub = T.alloc_shared((bs, ct + 64), in_tl_dtype)
                y_ub = T.alloc_shared((bs, y_ub_cols), "uint8")
                s_ub = T.alloc_shared((ct + 64,), "uint8")
                inv_s_ub = T.alloc_shared((ct + 64,), "float32")
                if schedule_mode == "tuned":
                    T.annotate_unlimit_memory("shared")

                c_chunks = (inner_dim + ct - 1) // ct
                total_tasks = outer_dim * n_blks * c_chunks
                base_t = total_tasks // num_cores
                rem_t = total_tasks % num_cores
                t_start = cid * base_t + T.min(cid, rem_t)
                t_count = base_t + T.if_then_else(cid < rem_t, 1, 0)

                for t_i in T.serial(t_count):
                    tid = t_start + t_i
                    c_ch = tid % c_chunks
                    ob = tid // c_chunks
                    o_i = ob // n_blks
                    blk = ob % n_blks
                    r0 = blk * bs
                    cur_bs = T.min(bs, axis_dim - r0)
                    c0 = c_ch * ct
                    cur_cols = T.min(ct, inner_dim - c0)
                    cur_out_cols = cur_cols // 2 if is_fp4 else cur_cols
                    out_c0 = c0 // 2 if is_fp4 else c0

                    T.copy(X[o_i, r0 : r0 + cur_bs, c0 : c0 + cur_cols], x_ub[:cur_bs, :cur_cols])

                    with T.SimdVF():
                        m32 = T.simd.pset(32, "PAT_ALL")
                        m16 = T.simd.pset(16, "PAT_ALL")
                        m8 = T.simd.pset(8, "PAT_ALL")
                        v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                        lane = T.simd.vci(0, "int32")

                        n_segs = (cur_cols + 63) // 64
                        amax_v = T.simd.alloc_local((1,), "float32")
                        for seg in T.serial(n_segs):
                            c_off = seg * 64
                            m_col = T.simd.vcmps(lane, T.min(64, cur_cols - c_off), mask=m32, op="lt")
                            amax_v[0] = v_zero
                            for r in T.serial(cur_bs):
                                raw_f = T.simd.vcvt(T.simd.vld(x_ub[r, c_off], dist="US_B16"), "float32", mask=m16, part=0)
                                v_val = T.simd.vsel(raw_f, v_zero, m_col)
                                amax_v[0] = T.simd.vmax(amax_v[0], T.simd.vabs(v_val, mask=m32), mask=m32)

                            v_inv_scale_f32, v_scale_u8_i, _ = _emit_scale_from_amax(
                                amax_v[0],
                                ele_emax=ele_emax,
                                m32=m32,
                                v_zero=v_zero,
                                v_abs_mask=T.simd.vdup(0x7FFFFFFF, "int32", mask=m32),
                                v_i0=T.simd.vdup(0, "int32", mask=m32),
                                v_i255=T.simd.vdup(255, "int32", mask=m32),
                            )
                            s_u16 = T.simd.vcvt(v_scale_u8_i, "uint16", mask=m16, round="ROUND_R", sat=False, part=0)
                            s_u8 = T.simd.vcvt(s_u16, "uint8", mask=m8, round="ROUND_R", sat=False, part=0)
                            T.simd.vsts(s_ub[c_off], s_u8, mask=m32, dist="PK4_B32", extent=64)
                            T.simd.vsts(inv_s_ub[c_off], v_inv_scale_f32, mask=m32, dist="NORM_B32")

                        for r in T.serial(cur_bs):
                            if is_fp4:
                                m_hi0 = T.simd.vcmps(lane, 32, mask=m32, op="lt")
                                n_pairs = (cur_cols + 127) // 128
                                for seg2 in T.serial(n_pairs):
                                    c0_off = seg2 * 128
                                    c_ev = _emit_nontail_quant_64(
                                        x_ub,
                                        inv_s_ub,
                                        r,
                                        c0_off,
                                        cur_cols,
                                        ele_emax=ele_emax,
                                        mant_bits=mant_bits,
                                        min_exp=min_exp,
                                        fp8_bias=fp8_bias,
                                        max_norm=max_norm,
                                        is_fp4=is_fp4,
                                        dst_type=dt,
                                        round_mode=rm,
                                        m32=m32,
                                        m16=m16,
                                        v_zero=v_zero,
                                        lane=lane,
                                    )
                                    v_i0_a = T.simd.vdup(0, "int32", mask=m32)
                                    idx_even_a = T.simd.vmuls(lane, 2, mask=m32)
                                    idx_odd_a = T.simd.vadds(idx_even_a, 1, mask=m32)
                                    pk_ev = T.simd.vor(
                                        T.simd.vselr(c_ev, T.simd.vsel(idx_even_a, v_i0_a, m_hi0)),
                                        T.simd.vshls(T.simd.vselr(c_ev, T.simd.vsel(idx_odd_a, v_i0_a, m_hi0)), 4, mask=m32),
                                        mask=m32,
                                    )
                                    if c0_off + 64 < cur_cols:
                                        c_od = _emit_nontail_quant_64(
                                            x_ub,
                                            inv_s_ub,
                                            r,
                                            c0_off + 64,
                                            cur_cols,
                                            ele_emax=ele_emax,
                                            mant_bits=mant_bits,
                                            min_exp=min_exp,
                                            fp8_bias=fp8_bias,
                                            max_norm=max_norm,
                                            is_fp4=is_fp4,
                                            dst_type=dt,
                                            round_mode=rm,
                                            m32=m32,
                                            m16=m16,
                                            v_zero=v_zero,
                                            lane=lane,
                                        )
                                        v_i0_b = T.simd.vdup(0, "int32", mask=m32)
                                        idx_even_b = T.simd.vmuls(lane, 2, mask=m32)
                                        idx_odd_b = T.simd.vadds(idx_even_b, 1, mask=m32)
                                        pk_od = T.simd.vor(
                                            T.simd.vselr(c_od, T.simd.vsel(idx_even_b, v_i0_b, m_hi0)),
                                            T.simd.vshls(T.simd.vselr(c_od, T.simd.vsel(idx_odd_b, v_i0_b, m_hi0)), 4, mask=m32),
                                            mask=m32,
                                        )
                                        lane_mod32 = T.simd.vand(lane, T.simd.vdup(31, "int32", mask=m32), mask=m32)
                                        pk64_both = T.simd.vsel(pk_ev, T.simd.vselr(pk_od, lane_mod32), m_hi0)
                                        pk16 = T.simd.vcvt(pk64_both, "uint16", mask=m16, round="ROUND_R", sat=False, part=0)
                                        pk8 = T.simd.vcvt(pk16, "uint8", mask=m8, round="ROUND_R", sat=False, part=0)
                                        T.simd.vsts(y_ub[r, seg2 * 64], pk8, mask=m32, dist="PK4_B32", extent=64)
                                    else:
                                        pk16_tail = T.simd.vcvt(pk_ev, "uint16", mask=m16, round="ROUND_R", sat=False, part=0)
                                        pk8_tail = T.simd.vcvt(pk16_tail, "uint8", mask=m8, round="ROUND_R", sat=False, part=0)
                                        T.simd.vsts(y_ub[r, seg2 * 64], pk8_tail, mask=m32, dist="PK4_B32", extent=64)
                            else:
                                for seg in T.serial(n_segs):
                                    codes = _emit_nontail_quant_64(
                                        x_ub,
                                        inv_s_ub,
                                        r,
                                        seg * 64,
                                        cur_cols,
                                        ele_emax=ele_emax,
                                        mant_bits=mant_bits,
                                        min_exp=min_exp,
                                        fp8_bias=fp8_bias,
                                        max_norm=max_norm,
                                        is_fp4=is_fp4,
                                        dst_type=dt,
                                        round_mode=rm,
                                        m32=m32,
                                        m16=m16,
                                        v_zero=v_zero,
                                        lane=lane,
                                    )
                                    y16 = T.simd.vcvt(codes, "uint16", mask=m16, round="ROUND_R", sat=False, part=0)
                                    y8 = T.simd.vcvt(y16, "uint8", mask=m8, round="ROUND_R", sat=False, part=0)
                                    T.simd.vsts(y_ub[r, seg * 64], y8, mask=m32, dist="PK4_B32", extent=64)

                    T.copy(y_ub[:cur_bs, :cur_out_cols], Y[o_i, r0 : r0 + cur_bs, out_c0 : out_c0 + cur_out_cols])
                    T.copy(s_ub[:cur_cols], Scale[o_i, blk, c0 : c0 + cur_cols])

        return main_nontail


def _get_kernel(
    dst_type: int,
    round_mode: str,
    blocksize: int,
    is_tail_axis: bool,
    input_dtype: str,
    outer_dim: int,
    axis_dim: int,
    inner_dim: int,
    schedule_mode: str,
):
    cfg = _select_dynamic_mx_quant_config(
        dst_type=dst_type,
        round_mode=round_mode,
        blocksize=blocksize,
        is_tail_axis=is_tail_axis,
        input_dtype=input_dtype,
        outer_dim=outer_dim,
        axis_dim=axis_dim,
        inner_dim=inner_dim,
        schedule_mode=schedule_mode,
    )
    key = (
        int(cfg["dst_type"]),
        str(cfg["round_mode"]),
        int(cfg["blocksize"]),
        bool(cfg["is_tail_axis"]),
        str(cfg["input_dtype"]),
        int(cfg["blks_per_chunk"]),
        int(cfg["col_tile"]),
        int(cfg["num_cores"]),
        bool(cfg["use_pipelined"]),
        schedule_mode,
    )
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        with _ascend_target_ctx():
            prog = _build_dynamic_mx_quant_prim_func(
                dst_type=int(cfg["dst_type"]),
                round_mode=str(cfg["round_mode"]),
                blocksize=int(cfg["blocksize"]),
                is_tail_axis=bool(cfg["is_tail_axis"]),
                input_dtype=str(cfg["input_dtype"]),
                blks_per_chunk=int(cfg["blks_per_chunk"]),
                col_tile=int(cfg["col_tile"]),
                num_cores=int(cfg["num_cores"]),
                use_pipelined=bool(cfg["use_pipelined"]),
                schedule_mode=schedule_mode,
            )
        compiled = tilelang.compile(
            prog,
            target="ascend",
            out_idx=[1, 2],
            compile_flags=cfg.get("compile_flags") or None,
        )
        _KERNEL_CACHE[key] = compiled
    return compiled


def _dynamic_mx_quant_cpu_ref(
    x: torch.Tensor,
    axis: int = -1,
    round_mode: str = "rint",
    dst_type: int = 40,
    blocksize: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype_name, ele_emax, exp_bits, mant_bits, _, max_norm, is_fp4 = _DST_TYPE_INFO[int(dst_type)]
    dim = x.dim()
    if axis < 0:
        axis += dim
    xf = x.float()
    tail_axis = axis == dim - 1
    xt = xf if tail_axis else xf.movedim(axis, -1)
    n = xt.shape[-1]
    n_blocks = (n + blocksize - 1) // blocksize
    pad_len = n_blocks * blocksize - n
    if pad_len > 0:
        xt = torch.nn.functional.pad(xt, (0, pad_len))
    blocks = xt.unflatten(-1, (n_blocks, blocksize))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    abs_bits = amax.contiguous().view(torch.int32) & 0x7FFFFFFF
    share_exp = ((abs_bits >> 23) - 127).to(torch.float32) - ele_emax
    share_exp = torch.where(amax == 0, torch.full_like(share_exp, -float("inf")), share_exp)
    share_exp = torch.where(share_exp > 127, torch.full_like(share_exp, float("nan")), share_exp)
    share_exp = torch.where(share_exp < -127, torch.full_like(share_exp, -127.0), share_exp)
    e = share_exp.nan_to_num(nan=0.0, posinf=127.0, neginf=-127.0).to(torch.int32)
    bits = (e + 127).clamp(1, 254) << 23
    bits = torch.where(e == -127, torch.full_like(bits, 0x00400000), bits)
    mxscale_val = bits.contiguous().view(torch.float32)
    mxscale_val = torch.where(share_exp == -float("inf"), torch.zeros_like(mxscale_val), mxscale_val)
    mxscale_val = torch.where(torch.isnan(share_exp), torch.full_like(mxscale_val, float("nan")), mxscale_val)
    ret = blocks / mxscale_val
    min_exp = -(2 ** (exp_bits - 1)) + exp_bits if is_fp4 else -(2 ** (exp_bits - 1)) + 2
    pexp = torch.floor(torch.log2(torch.abs(ret) + (ret == 0).to(torch.float32))).clamp_min(min_exp)
    step = torch.pow(2.0, pexp - mant_bits)
    scaled = ret / step
    if round_mode == "rint":
        scaled = torch.round(scaled)
    elif round_mode == "floor":
        scaled = torch.floor(scaled)
    else:
        scaled = torch.sign(scaled) * torch.floor(torch.abs(scaled) + 0.5)
    ret = torch.nan_to_num((scaled * step).clamp(-max_norm, max_norm), nan=0.0)
    y_f = ret.flatten(-2)[..., :n]
    if not tail_axis:
        y_f = y_f.movedim(-1, axis)
    scale_val = mxscale_val.squeeze(-1)
    if not tail_axis:
        scale_val = scale_val.movedim(-1, axis)
    mxscale = ((scale_val.contiguous().view(torch.int32) & 0x7FFFFFFF) >> 23).to(torch.uint8).view(torch.float8_e8m0fnu)
    if is_fp4:
        table = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0) if dst_type == 40 else (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75)
        pos = torch.tensor(table, dtype=torch.float32)
        codes = torch.argmax((y_f.abs().unsqueeze(-1) == pos).to(torch.uint8), dim=-1).to(torch.uint8)
        codes = codes + torch.where(torch.signbit(y_f), 8, 0).to(torch.uint8)
        codes = codes.reshape(y_f.shape[:-1] + (y_f.shape[-1] // 2, 2))
        y_out = codes[..., 0] | (codes[..., 1] << 4)
        return y_out, mxscale
    bias = 15 if dst_type == 35 else 7
    b = y_f.contiguous().view(torch.int32)
    ab = b & 0x7FFFFFFF
    sg = (b >> 31) & 1
    mt = ab & 0x7FFFFF
    ex = (ab >> 23) - 127
    nrm = ((ex.clamp_min(1 - bias) + bias) << mant_bits) | (mt >> (23 - mant_bits))
    sub = ((1 << 23) + mt) >> (23 - ex + (1 - bias) - mant_bits).clamp(0, 31)
    byte = torch.where(ex >= (1 - bias), nrm, sub)
    y_out = ((sg << 7) | byte).to(torch.uint8).view(getattr(torch, dtype_name))
    return y_out, mxscale


def dynamic_mx_quant(
    x: torch.Tensor,
    axis: int = -1,
    round_mode: str = "rint",
    dst_type: int = 40,
    blocksize: int = 32,
    *,
    schedule_mode: str = "tuned",
) -> tuple[torch.Tensor, torch.Tensor]:
    """OCP microscaling dynamic quantization to FP8 (E5M2/E4M3FN) or packed FP4 (E2M1/E1M2)."""
    if x.device.type != "npu":
        return _dynamic_mx_quant_cpu_ref(x, axis=axis, round_mode=round_mode, dst_type=dst_type, blocksize=blocksize)

    dt = int(dst_type)
    bs = int(blocksize)
    dim = x.dim()
    ax = int(axis)
    if ax < 0:
        ax += dim
    is_tail_axis = ax == (dim - 1)
    dtype_name, _, _, _, _, _, is_fp4 = _DST_TYPE_INFO[dt]
    in_dtype_name = "bfloat16" if x.dtype == torch.bfloat16 else "float16"

    shape = tuple(int(s) for s in x.shape)
    axis_dim = shape[ax]
    n_blks = (axis_dim + bs - 1) // bs
    outer_dim = math.prod(shape[:ax]) if ax > 0 else 1
    inner_dim = math.prod(shape[ax + 1 :]) if ax < dim - 1 else 1

    compiled = _get_kernel(
        dst_type=dt,
        round_mode=round_mode,
        blocksize=bs,
        is_tail_axis=is_tail_axis,
        input_dtype=in_dtype_name,
        outer_dim=outer_dim,
        axis_dim=axis_dim,
        inner_dim=inner_dim,
        schedule_mode=schedule_mode,
    )

    x_contig = x.contiguous()
    y_shape = shape[:-1] + (shape[-1] // 2,) if is_fp4 else shape
    scale_shape = shape[:ax] + (n_blks,) + shape[ax + 1 :]

    if is_tail_axis:
        if axis_dim % bs == 0:
            total_elems = outer_dim * axis_dim
            x_2d = x_contig.view(1, total_elems)
            y_u8, s_u8 = compiled(x_2d, 1, int(total_elems), int(total_elems // bs))
        else:
            x_2d = x_contig.view(outer_dim, axis_dim)
            y_u8, s_u8 = compiled(x_2d, int(outer_dim), int(axis_dim), int(n_blks))
    else:
        x_3d = x_contig.view(outer_dim, axis_dim, inner_dim)
        y_u8, s_u8 = compiled(x_3d, int(outer_dim), int(axis_dim), int(inner_dim), int(n_blks))

    y_out = y_u8.view(y_shape) if is_fp4 else y_u8.view(getattr(torch, dtype_name)).view(y_shape)
    mxscale = s_u8.view(torch.float8_e8m0fnu).view(scale_shape)
    return y_out, mxscale


kernel = dynamic_mx_quant
