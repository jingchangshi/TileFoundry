"""TileLang Ascend950PR Persistent-State SimdVF kernel for KimiDeltaAttention."""

import contextlib
import ctypes
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import tilelang
import tilelang.ascend.language as T
import torch

from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS

_KERNEL_CACHE: dict[tuple[object, ...], object] = {}
_TRACK_B_LIB_CACHE: dict[str, ctypes.CDLL] = {}
_TRACK_B_WS_POOL: dict[str, torch.Tensor] = {}


def _ascend_target_ctx():
    try:
        from tilelang import tvm
        from tilelang.backend.module import create_backend_context

        if tvm.target.Target.current(allow_none=True) is None:
            return create_backend_context("ascend").target
    except Exception:
        pass
    return contextlib.nullcontext()


def _select_kimi_delta_attention_config(
    dk: int = 128,
    dv: int = 128,
    tl_dtype: str = "bfloat16",
    *,
    t_dim: int = 64,
    use_qk_l2norm: bool = True,
    schedule_mode: str = "tuned",
    chunk_t_override: int = 0,
    dv_tile_override: int = 0,
) -> dict[str, object]:
    """Select persistent (B, H) UB-resident state regime and chunking for KDA."""
    dk_alloc = ((max(1, int(dk)) + 63) // 64) * 64
    dv_alloc = ((max(1, int(dv)) + 63) // 64) * 64
    state_bytes = dk_alloc * dv_alloc * 4
    if state_bytes <= 192 * 1024:
        regime = "persistent_ub_state_resident"
    else:
        regime = "chunked_dv_tile_state"
    default_chunk_t = min(max(1, int(t_dim)), 64 if schedule_mode == "tuned" else 16)
    chunk_t = int(chunk_t_override) if int(chunk_t_override) > 0 else default_chunk_t
    default_dv_tile = 64 if state_bytes > 192 * 1024 else dv_alloc
    dv_tile = int(dv_tile_override) if int(dv_tile_override) > 0 else default_dv_tile
    return {
        "regime": regime,
        "dk": int(dk),
        "dv": int(dv),
        "tl_dtype": tl_dtype,
        "use_qk_l2norm": bool(use_qk_l2norm),
        "chunk_t": chunk_t,
        "dv_tile": dv_tile,
        "pass_configs": {},
        "compile_flags": list(PURE_SIMDVF_UBUF_FLAGS) if schedule_mode == "tuned" else [],
    }


def _build_kimi_delta_attention_prim_func(
    dk: int = 128,
    dv: int = 128,
    tl_dtype: str = "bfloat16",
    use_qk_l2norm: bool = True,
    schedule_mode: str = "tuned",
    **_extra_kwargs,
):
    """Build Ascend 950 DV-sharded @T.prim_func for KimiDeltaAttention."""
    if dk <= 0 or dv <= 0:
        raise ValueError(f"dk and dv must be positive, got ({dk}, {dv})")
    if tl_dtype not in ("float16", "bfloat16", "float32"):
        raise ValueError(f"unsupported dtype {tl_dtype!r}")

    num_cores = 56
    dk_alloc = ((dk + 63) // 64) * 64
    dk_chunks = dk // 64
    dk_rem = dk % 64
    dv_tiles = (dv + 63) // 64
    dv_last = dv - (dv_tiles - 1) * 64
    t_blk = 16 if schedule_mode == "tuned" else 8

    b_dim = T.dynamic("b_dim")
    t_dim = T.dynamic("t_dim")
    h_dim = T.dynamic("h_dim")

    with _ascend_target_ctx():

        @T.prim_func
        def main(
            Q: T.Tensor((b_dim, t_dim, h_dim, dk), tl_dtype),
            K: T.Tensor((b_dim, t_dim, h_dim, dk), tl_dtype),
            V: T.Tensor((b_dim, t_dim, h_dim, dv), tl_dtype),
            G: T.Tensor((b_dim, t_dim, h_dim, dk), "float32"),
            Beta: T.Tensor((b_dim, t_dim, h_dim), "float32"),
            InitialState: T.Tensor((b_dim, h_dim, dk, dv), "float32"),
            Y: T.Tensor((b_dim, t_dim, h_dim, dv), tl_dtype),
            FinalState: T.Tensor((b_dim, h_dim, dk, dv), "float32"),
            scale_val: T.float32,
        ):
            with T.Kernel(num_cores) as cid:
                state_ub = T.alloc_shared((dk_alloc, 64), "float32")
                q_tile_ub = T.alloc_shared((t_blk, dk_alloc), tl_dtype)
                k_tile_ub = T.alloc_shared((t_blk, dk_alloc), tl_dtype)
                v_tile_ub = T.alloc_shared((t_blk, 64), tl_dtype)
                g_tile_ub = T.alloc_shared((t_blk, dk_alloc), "float32")
                beta_tile_ub = T.alloc_shared((t_blk,), "float32")
                out_tile_ub = T.alloc_shared((t_blk, 64), tl_dtype)
                qf_ub = T.alloc_shared((dk_alloc,), "float32")
                kf_ub = T.alloc_shared((dk_alloc,), "float32")
                g1_ub = T.alloc_shared((dk_alloc,), "float32")
                T.annotate_buffer_versions(
                    {
                        state_ub: 1,
                        q_tile_ub: 1,
                        k_tile_ub: 1,
                        v_tile_ub: 1,
                        g_tile_ub: 1,
                        beta_tile_ub: 1,
                        out_tile_ub: 1,
                        qf_ub: 1,
                        kf_ub: 1,
                        g1_ub: 1,
                    }
                )
                if schedule_mode == "tuned":
                    T.annotate_unlimit_memory("shared")

                total_tasks = b_dim * h_dim * dv_tiles
                base_tasks = total_tasks // num_cores
                rem_tasks = total_tasks % num_cores
                t_start = cid * base_tasks + T.min(cid, rem_tasks)
                t_count = base_tasks + T.if_then_else(cid < rem_tasks, 1, 0)
                t_blocks = (t_dim + t_blk - 1) // t_blk

                for ltask in T.serial(t_count):
                    tid = t_start + ltask
                    bh = tid // dv_tiles
                    dv_t = tid % dv_tiles
                    b_idx = bh // h_dim
                    h_idx = bh % h_dim
                    dv0 = dv_t * 64
                    dv_cur = T.min(64, dv - dv0)

                    for r in T.serial(dk):
                        T.copy(InitialState[b_idx, h_idx, r, dv0 : dv0 + dv_cur], state_ub[r, :dv_cur])

                    for tb in T.serial(t_blocks):
                        t0 = tb * t_blk
                        t_cur = T.min(t_blk, t_dim - t0)
                        for lt in T.serial(t_cur):
                            T.copy(Q[b_idx, t0 + lt, h_idx, :dk], q_tile_ub[lt, :dk])
                            T.copy(K[b_idx, t0 + lt, h_idx, :dk], k_tile_ub[lt, :dk])
                            T.copy(V[b_idx, t0 + lt, h_idx, dv0 : dv0 + dv_cur], v_tile_ub[lt, :dv_cur])
                            T.copy(G[b_idx, t0 + lt, h_idx, :dk], g_tile_ub[lt, :dk])
                            beta_tile_ub[lt] = Beta[b_idx, t0 + lt, h_idx]
                        T.ascend_pipe_barrier("PIPE_ALL")

                        with T.SimdVF():
                            m32 = T.simd.pset(32, "PAT_ALL")
                            m16 = T.simd.pset(16, "PAT_ALL")
                            mv32 = m32 if dv_last == 64 else T.simd.update_mask(dv_cur, width=32)
                            mv16 = m16 if dv_last == 64 else T.simd.update_mask(dv_cur * 2, width=16)
                            v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                            v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
                            v_eps = T.simd.vdup(T.float32(1.0e-6), "float32", mask=m32)
                            q_sq = T.simd.alloc_local((1,), "float32")
                            k_sq = T.simd.alloc_local((1,), "float32")
                            pred_reg = T.simd.alloc_local((1,), "float32")
                            y_reg = T.simd.alloc_local((1,), "float32")
                            vf_reg = T.simd.alloc_local((1,), "float32")

                            for lt in T.serial(t_cur):
                                beta_t = beta_tile_ub[lt]
                                q_sq[0] = v_zero
                                k_sq[0] = v_zero

                                for c in T.serial(dk_chunks):
                                    off = c * 64
                                    if tl_dtype == "float32":
                                        qv = T.simd.vld(q_tile_ub[lt, off], dist="NORM_B32")
                                        kv = T.simd.vld(k_tile_ub[lt, off], dist="NORM_B32")
                                    else:
                                        qv = T.simd.vcvt(T.simd.vld(q_tile_ub[lt, off], dist="US_B16"), "float32", mask=m16, part=0)
                                        kv = T.simd.vcvt(T.simd.vld(k_tile_ub[lt, off], dist="US_B16"), "float32", mask=m16, part=0)
                                    T.simd.vsts(qf_ub[off], qv, mask=m32, dist="NORM_B32")
                                    T.simd.vsts(kf_ub[off], kv, mask=m32, dist="NORM_B32")
                                    T.simd.vmula(q_sq[0], qv, qv, mask=m32)
                                    T.simd.vmula(k_sq[0], kv, kv, mask=m32)
                                if dk_rem > 0:
                                    off = dk_chunks * 64
                                    m32_t = T.simd.update_mask(dk_rem, width=32)
                                    if tl_dtype == "float32":
                                        qv_t = T.simd.vld(q_tile_ub[lt, off], dist="NORM_B32")
                                        kv_t = T.simd.vld(k_tile_ub[lt, off], dist="NORM_B32")
                                    else:
                                        m16_t = T.simd.update_mask(dk_rem * 2, width=16)
                                        qv_t = T.simd.vcvt(T.simd.vld(q_tile_ub[lt, off], dist="US_B16"), "float32", mask=m16_t, part=0)
                                        kv_t = T.simd.vcvt(T.simd.vld(k_tile_ub[lt, off], dist="US_B16"), "float32", mask=m16_t, part=0)
                                    T.simd.vsts(qf_ub[off], qv_t, mask=m32_t, dist="NORM_B32")
                                    T.simd.vsts(kf_ub[off], kv_t, mask=m32_t, dist="NORM_B32")
                                    q_sq[0] = T.simd.vadd(q_sq[0], T.simd.vmul(qv_t, qv_t, mask=m32_t), mask=m32)
                                    k_sq[0] = T.simd.vadd(k_sq[0], T.simd.vmul(kv_t, kv_t, mask=m32_t), mask=m32)

                                if use_qk_l2norm:
                                    q_inv = T.simd.vdiv(
                                        v_one,
                                        T.simd.vsqrt(
                                            T.simd.vadd(
                                                T.simd.vdupv(T.simd.vcadd(q_sq[0], mask=m32), mask=m32, pos="POS_LOWEST"),
                                                v_eps,
                                                mask=m32,
                                            ),
                                            mask=m32,
                                            precision="ftz_false",
                                        ),
                                        mask=m32,
                                        precision="exact",
                                    )
                                    k_inv = T.simd.vdiv(
                                        v_one,
                                        T.simd.vsqrt(
                                            T.simd.vadd(
                                                T.simd.vdupv(T.simd.vcadd(k_sq[0], mask=m32), mask=m32, pos="POS_LOWEST"),
                                                v_eps,
                                                mask=m32,
                                            ),
                                            mask=m32,
                                            precision="ftz_false",
                                        ),
                                        mask=m32,
                                        precision="exact",
                                    )
                                else:
                                    q_inv = v_one
                                    k_inv = v_one
                                q_scale_v = T.simd.vmuls(q_inv, scale_val, mask=m32)

                                for c in T.serial(dk_chunks):
                                    off = c * 64
                                    T.simd.vsts(
                                        qf_ub[off],
                                        T.simd.vmul(T.simd.vld(qf_ub[off], dist="NORM_B32"), q_scale_v, mask=m32),
                                        mask=m32,
                                        dist="NORM_B32",
                                    )
                                    T.simd.vsts(
                                        kf_ub[off],
                                        T.simd.vmul(T.simd.vld(kf_ub[off], dist="NORM_B32"), k_inv, mask=m32),
                                        mask=m32,
                                        dist="NORM_B32",
                                    )
                                    T.simd.vsts(
                                        g1_ub[off],
                                        T.simd.vexp(T.simd.vld(g_tile_ub[lt, off], dist="NORM_B32"), mask=m32, precision="ftz_false"),
                                        mask=m32,
                                        dist="NORM_B32",
                                    )
                                if dk_rem > 0:
                                    off = dk_chunks * 64
                                    m32_t = T.simd.update_mask(dk_rem, width=32)
                                    T.simd.vsts(
                                        qf_ub[off],
                                        T.simd.vmul(T.simd.vld(qf_ub[off], dist="NORM_B32"), q_scale_v, mask=m32_t),
                                        mask=m32_t,
                                        dist="NORM_B32",
                                    )
                                    T.simd.vsts(
                                        kf_ub[off],
                                        T.simd.vmul(T.simd.vld(kf_ub[off], dist="NORM_B32"), k_inv, mask=m32_t),
                                        mask=m32_t,
                                        dist="NORM_B32",
                                    )
                                    T.simd.vsts(
                                        g1_ub[off],
                                        T.simd.vexp(T.simd.vld(g_tile_ub[lt, off], dist="NORM_B32"), mask=m32_t, precision="ftz_false"),
                                        mask=m32_t,
                                        dist="NORM_B32",
                                    )

                                if tl_dtype == "float32":
                                    vf_reg[0] = T.simd.vld(v_tile_ub[lt, 0], dist="NORM_B32")
                                else:
                                    vf_reg[0] = T.simd.vcvt(T.simd.vld(v_tile_ub[lt, 0], dist="US_B16"), "float32", mask=mv16, part=0)
                                pred_reg[0] = v_zero
                                y_reg[0] = v_zero
                                T.simd.mem_bar("VST_VLD")

                                for r in T.serial(dk):
                                    decay_v = T.simd.vld(g1_ub[r], dist="BRC_B32")
                                    k_norm_r = T.simd.vld(kf_ub[r], dist="BRC_B32")
                                    h_prev = T.simd.vld(state_ub[r, 0], dist="NORM_B32")
                                    h_bar = T.simd.vmul(h_prev, decay_v, mask=mv32)
                                    T.simd.vsts(state_ub[r, 0], h_bar, mask=mv32, dist="NORM_B32")
                                    pred_reg[0] = T.simd.vadd(pred_reg[0], T.simd.vmul(h_bar, k_norm_r, mask=mv32), mask=mv32)

                                corr_v = T.simd.vmuls(T.simd.vsub(vf_reg[0], pred_reg[0], mask=mv32), beta_t, mask=mv32)
                                T.simd.mem_bar("VST_VLD")

                                for r in T.serial(dk):
                                    k_r = T.simd.vld(kf_ub[r], dist="BRC_B32")
                                    q_r = T.simd.vld(qf_ub[r], dist="BRC_B32")
                                    h_new = T.simd.vadd(
                                        T.simd.vld(state_ub[r, 0], dist="NORM_B32"),
                                        T.simd.vmul(k_r, corr_v, mask=mv32),
                                        mask=mv32,
                                    )
                                    T.simd.vsts(state_ub[r, 0], h_new, mask=mv32, dist="NORM_B32")
                                    y_reg[0] = T.simd.vadd(y_reg[0], T.simd.vmul(h_new, q_r, mask=mv32), mask=mv32)
                                T.simd.mem_bar("VST_VLD")

                                if tl_dtype == "float32":
                                    T.simd.vsts(out_tile_ub[lt, 0], y_reg[0], mask=mv32, dist="NORM_B32")
                                else:
                                    y16 = T.simd.vcvt(y_reg[0], tl_dtype, mask=mv16, part=0)
                                    T.simd.vsts(out_tile_ub[lt, 0], y16, mask=mv32, dist="PK_B32", extent=64)

                        T.ascend_pipe_barrier("PIPE_ALL")
                        for lt in T.serial(t_cur):
                            T.copy(out_tile_ub[lt, :dv_cur], Y[b_idx, t0 + lt, h_idx, dv0 : dv0 + dv_cur])
                        T.ascend_pipe_barrier("PIPE_ALL")

                    for r in T.serial(dk):
                        T.copy(state_ub[r, :dv_cur], FinalState[b_idx, h_idx, r, dv0 : dv0 + dv_cur])
                    T.ascend_pipe_barrier("PIPE_ALL")

    return main


def _get_kernel(dk: int, dv: int, tl_dtype: str, use_qk_l2norm: bool, t_dim: int, schedule_mode: str):
    cfg = _select_kimi_delta_attention_config(
        dk=dk,
        dv=dv,
        tl_dtype=tl_dtype,
        t_dim=t_dim,
        use_qk_l2norm=use_qk_l2norm,
        schedule_mode=schedule_mode,
    )
    key = (dk, dv, tl_dtype, bool(use_qk_l2norm), schedule_mode)
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        with _ascend_target_ctx():
            prog = _build_kimi_delta_attention_prim_func(
                dk=dk,
                dv=dv,
                tl_dtype=tl_dtype,
                use_qk_l2norm=use_qk_l2norm,
                schedule_mode=schedule_mode,
            )
        compiled = tilelang.compile(
            prog,
            target="ascend",
            out_idx=[6, 7],
            pass_configs=cfg.get("pass_configs") or None,
            compile_flags=cfg.get("compile_flags") or None,
        )
        _KERNEL_CACHE[key] = compiled
    return compiled


_TRACK_B_ASC_SRC = r'''
#include "tl_templates/ascend/common.h"
#include "tl_templates/ascend/gemm.h"
#include "tl_templates/ascend/nd2nz_copy.h"

namespace V = simd_inst;
using namespace AscendC;

// Add (float -> half) scatter specialization for COLS % 64 == 0 AND COLS % 16 == 0 (< 64)
template <int ROWS, int COLS, int DST_PITCH>
struct ascend_nd2nz_scatter_impl<ROWS, COLS, DST_PITCH, float, half,
                                 std::enable_if_t<COLS % 64 == 0>> {
  __simd_callee__ static inline void apply(__ubuf__ float *src_ub,
                                           __ubuf__ half *dst_nz_ub) {
    static_assert(ROWS % 16 == 0, "ROWS must be a multiple of 16");
    constexpr int NUM_PASSES = COLS / 64;
    constexpr uint32_t STRIDE = (uint32_t(DST_PITCH) << 16) | 1;
    auto f32Mask = asc_create_mask_b32(PAT_ALL);
    auto f16Mask = asc_create_mask_b16(PAT_VL64);
    for (int pass = 0; pass < NUM_PASSES; ++pass) {
      __ubuf__ half *dstPtr = dst_nz_ub + pass * DST_PITCH * 64;
      __ubuf__ float *srcBase = src_ub + pass * 64;
      for (int r = 0; r < ROWS; ++r) {
        auto srcReg = V::vlds_norm(srcBase + r * COLS, 0);
        auto castReg = V::vcvt<half>(srcReg, f32Mask, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
        auto packReg = V::vpack<uint16_t>((V::vec_t<uint32_t> &)castReg, LOWER);
        dstPtr = V::vsstb((V::vec_t<half> &)packReg, dstPtr, STRIDE, f16Mask, POST_UPDATE);
      }
    }
  }
};

template <int ROWS, int COLS, int DST_PITCH, typename T>
__simd_callee__ inline void scatter_nz_callee(__ubuf__ float *src_ub, __ubuf__ T *dst_nz_ub) {
  if constexpr (COLS % 64 == 0 || std::is_same_v<T, float>) {
    ascend_nd2nz_scatter_impl<ROWS, COLS, DST_PITCH, float, T>::apply(src_ub, dst_nz_ub);
  } else {
    constexpr uint32_t STRIDE = (uint32_t(DST_PITCH) << 16) | 1;
    uint32_t cnt32 = COLS;
    auto f32Mask = asc_update_mask_b32(cnt32);
    uint32_t cnt16 = COLS;
    auto b16Mask = asc_update_mask_b16(cnt16);
    __ubuf__ T *dstPtr = dst_nz_ub;
    for (int r = 0; r < ROWS; ++r) {
      auto srcReg = V::vlds_norm(src_ub + r * COLS, 0);
      if constexpr (std::is_same_v<T, half>) {
        auto castReg = V::vcvt<half>(srcReg, f32Mask, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
        auto packReg = V::vpack<uint16_t>((V::vec_t<uint32_t> &)castReg, LOWER);
        dstPtr = V::vsstb((V::vec_t<half> &)packReg, dstPtr, STRIDE, b16Mask, POST_UPDATE);
      } else {
        auto castReg = V::vcvt<bfloat16_t>(srcReg, f32Mask, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
        auto packReg = V::vpack<uint16_t>((V::vec_t<uint32_t> &)castReg, LOWER);
        dstPtr = V::vsstb((V::vec_t<bfloat16_t> &)packReg, dstPtr, STRIDE, b16Mask, POST_UPDATE);
      }
    }
  }
}

template <int ROWS, int COLS, int DST_PITCH, typename T>
__simd_vf__ inline void scatter_nz_vf(__ubuf__ float *src_ub, __ubuf__ T *dst_nz_ub) {
  scatter_nz_callee<ROWS, COLS, DST_PITCH, T>(src_ub, dst_nz_ub);
}

template <typename T>
__simd_callee__ inline vector_f32 load_f32_vec(__ubuf__ T* src, int32_t off) {
  if constexpr (std::is_same_v<T, float>) {
    return V::vlds_norm(src + off, 0);
  } else {
    auto m16 = asc_create_mask_b16(PAT_ALL);
    auto raw = V::vlds_unpack(src + off, 0);
    return V::vcvt<float>(raw, m16, PART_EVEN, MODE_ZEROING);
  }
}

template <typename T>
__simd_callee__ inline void store_f32_vec(__ubuf__ T* dst, int32_t off, vector_f32 val,
                                          vector_bool m32, vector_bool m16) {
  if constexpr (std::is_same_v<T, float>) {
    V::vsts_norm(val, dst + off, 0, m32);
  } else if constexpr (std::is_same_v<T, half>) {
    auto c = V::vcvt<half>(val, m16, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
    V::vsts_pack_b32(c, dst + off, 0, m32);
  } else {
    auto c = V::vcvt<bfloat16_t>(val, m32, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
    V::vsts_pack_b32(c, dst + off, 0, m32);
  }
}

__simd_callee__ inline vector_bool make_mask_b32(uint32_t n) {
  return asc_update_mask_b32(n);
}
__simd_callee__ inline vector_bool make_mask_b16(uint32_t n) {
  return asc_update_mask_b16(n);
}

// ============================================================================
// Kernel 1: Fast Vector Path for T == 1 or small Dk <= 32, T <= 32
// ============================================================================
template <typename T>
__simd_vf__ inline void kda_vec_fast_vf(
    __ubuf__ uint8_t* ub,
    int32_t T_len, int32_t Dk, int32_t dk_pad, int32_t dv_cur,
    int32_t use_l2norm, float scale_val) {
  __ubuf__ float* state_ub = (__ubuf__ float*)(ub + 0);
  __ubuf__ T* q_in = (__ubuf__ T*)(ub + 32768);
  __ubuf__ T* k_in = (__ubuf__ T*)(ub + 49152);
  __ubuf__ T* v_in = (__ubuf__ T*)(ub + 65536);
  __ubuf__ float* g_in = (__ubuf__ float*)(ub + 73728);
  __ubuf__ float* beta_in = (__ubuf__ float*)(ub + 90112);
  __ubuf__ float* qf_ub = (__ubuf__ float*)(ub + 92160);
  __ubuf__ float* kf_ub = (__ubuf__ float*)(ub + 108544);
  __ubuf__ float* ef_ub = (__ubuf__ float*)(ub + 124928);
  __ubuf__ T* out_ub = (__ubuf__ T*)(ub + 143360);

  auto m32 = asc_create_mask_b32(PAT_ALL);
  auto m16 = asc_create_mask_b16(PAT_ALL);
  auto mv32 = make_mask_b32((uint32_t)dv_cur);
  auto mv16 = make_mask_b16((uint32_t)(dv_cur * 2));
  auto v_zero = V::vdup<float>(0.0f, m32, MODE_ZEROING);
  auto v_one = V::vdup<float>(1.0f, m32, MODE_ZEROING);
  auto v_eps = V::vdup<float>(1.0e-6f, m32, MODE_ZEROING);

  int32_t dk_chunks = (Dk + 63) / 64;
  int32_t dk_tail = Dk - (dk_chunks - 1) * 64;

  for (int32_t t = 0; t < T_len; ++t) {
    if (dk_chunks == 1) {
      auto mc = (Dk == 64) ? m32 : make_mask_b32((uint32_t)Dk);
      auto qv = load_f32_vec(q_in, t * dk_pad);
      auto kv = load_f32_vec(k_in, t * dk_pad);
      auto gv = V::vlds_norm(g_in + t * dk_pad, 0);
      auto ev = V::vexp(gv, mc, MODE_ZEROING);
      if (use_l2norm) {
        auto q_sum = V::vdupv(V::vcadd(V::vmul(qv, qv, mc, MODE_ZEROING), mc, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
        auto k_sum = V::vdupv(V::vcadd(V::vmul(kv, kv, mc, MODE_ZEROING), mc, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
        auto q_den = V::vsqrt(V::vadd(q_sum, v_eps, m32, MODE_ZEROING), m32, MODE_ZEROING);
        auto k_den = V::vsqrt(V::vadd(k_sum, v_eps, m32, MODE_ZEROING), m32, MODE_ZEROING);
        auto q_sc = V::vmuls(V::vdiv_0ulp_ftz_true(v_one, q_den, m32, MODE_ZEROING), scale_val, m32, MODE_ZEROING);
        auto k_sc = V::vdiv_0ulp_ftz_true(v_one, k_den, m32, MODE_ZEROING);
        qv = V::vmul(qv, q_sc, mc, MODE_ZEROING);
        kv = V::vmul(kv, k_sc, mc, MODE_ZEROING);
      } else {
        qv = V::vmuls(qv, scale_val, mc, MODE_ZEROING);
      }
      V::vsts_norm(qv, qf_ub + t * dk_pad, 0, mc);
      V::vsts_norm(kv, kf_ub + t * dk_pad, 0, mc);
      V::vsts_norm(ev, ef_ub + t * dk_pad, 0, mc);
    } else {
      auto q0 = load_f32_vec(q_in, t * dk_pad);
      auto q1 = load_f32_vec(q_in, t * dk_pad + 64);
      auto k0 = load_f32_vec(k_in, t * dk_pad);
      auto k1 = load_f32_vec(k_in, t * dk_pad + 64);
      auto mc1 = (dk_tail == 64) ? m32 : make_mask_b32((uint32_t)dk_tail);
      auto e0 = V::vexp(V::vlds_norm(g_in + t * dk_pad, 0), m32, MODE_ZEROING);
      auto e1 = V::vexp(V::vlds_norm(g_in + t * dk_pad + 64, 0), mc1, MODE_ZEROING);
      if (use_l2norm) {
        auto q_sq = V::vadd(V::vmul(q0, q0, m32, MODE_ZEROING), V::vmul(q1, q1, mc1, MODE_ZEROING), m32, MODE_ZEROING);
        auto k_sq = V::vadd(V::vmul(k0, k0, m32, MODE_ZEROING), V::vmul(k1, k1, mc1, MODE_ZEROING), m32, MODE_ZEROING);
        auto qs = V::vdupv(V::vcadd(q_sq, m32, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
        auto ks = V::vdupv(V::vcadd(k_sq, m32, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
        auto q_sc = V::vmuls(V::vdiv_0ulp_ftz_true(v_one, V::vsqrt(V::vadd(qs, v_eps, m32, MODE_ZEROING), m32, MODE_ZEROING), m32, MODE_ZEROING), scale_val, m32, MODE_ZEROING);
        auto k_sc = V::vdiv_0ulp_ftz_true(v_one, V::vsqrt(V::vadd(ks, v_eps, m32, MODE_ZEROING), m32, MODE_ZEROING), m32, MODE_ZEROING);
        q0 = V::vmul(q0, q_sc, m32, MODE_ZEROING); q1 = V::vmul(q1, q_sc, mc1, MODE_ZEROING);
        k0 = V::vmul(k0, k_sc, m32, MODE_ZEROING); k1 = V::vmul(k1, k_sc, mc1, MODE_ZEROING);
      } else {
        q0 = V::vmuls(q0, scale_val, m32, MODE_ZEROING); q1 = V::vmuls(q1, scale_val, mc1, MODE_ZEROING);
      }
      V::vsts_norm(q0, qf_ub + t * dk_pad, 0, m32);
      V::vsts_norm(q1, qf_ub + t * dk_pad + 64, 0, mc1);
      V::vsts_norm(k0, kf_ub + t * dk_pad, 0, m32);
      V::vsts_norm(k1, kf_ub + t * dk_pad + 64, 0, mc1);
      V::vsts_norm(e0, ef_ub + t * dk_pad, 0, m32);
      V::vsts_norm(e1, ef_ub + t * dk_pad + 64, 0, mc1);
    }
  }
  asc_mem_bar(VST_VLD);

  for (int32_t t = 0; t < T_len; ++t) {
    auto k_dot_h = v_zero;
    for (int32_t d = 0; d < Dk; ++d) {
      auto e_s = V::vlds_brc_elem(ef_ub + t * dk_pad + d, 0);
      auto k_s = V::vlds_brc_elem(kf_ub + t * dk_pad + d, 0);
      auto h_row = V::vlds_norm(state_ub + d * 64, 0);
      auto h_dec = V::vmul(h_row, e_s, mv32, MODE_ZEROING);
      V::vsts_norm(h_dec, state_ub + d * 64, 0, mv32);
      k_dot_h = V::vadd(k_dot_h, V::vmul(h_dec, k_s, mv32, MODE_ZEROING), mv32, MODE_ZEROING);
    }
    asc_mem_bar(VST_VLD);
    auto v_t = load_f32_vec(v_in, t * 64);
    auto b_t = V::vlds_brc_elem(beta_in + t, 0);
    auto delta = V::vmul(V::vsub(v_t, k_dot_h, mv32, MODE_ZEROING), b_t, mv32, MODE_ZEROING);
    auto out_acc = v_zero;
    for (int32_t d = 0; d < Dk; ++d) {
      auto k_s = V::vlds_brc_elem(kf_ub + t * dk_pad + d, 0);
      auto q_s = V::vlds_brc_elem(qf_ub + t * dk_pad + d, 0);
      auto h_dec = V::vlds_norm(state_ub + d * 64, 0);
      auto h_new = V::vadd(h_dec, V::vmul(delta, k_s, mv32, MODE_ZEROING), mv32, MODE_ZEROING);
      V::vsts_norm(h_new, state_ub + d * 64, 0, mv32);
      out_acc = V::vadd(out_acc, V::vmul(h_new, q_s, mv32, MODE_ZEROING), mv32, MODE_ZEROING);
    }
    if (t + 1 < T_len) asc_mem_bar(VST_VLD);
    store_f32_vec(out_ub, t * 64, out_acc, mv32, mv16);
  }
  asc_mem_bar(VST_VLD);
}

template <typename T>
__aicore__ inline void kda_vec_fast_impl(
    __gm__ T* q, __gm__ T* k, __gm__ T* v, __gm__ float* g, __gm__ float* beta, __gm__ float* init_state,
    __gm__ T* y, __gm__ float* final_state,
    int32_t B, int32_t T_len, int32_t H, int32_t Dk, int32_t Dv,
    int32_t use_l2norm, float scale_val) {
  asc_init();
  __ubuf__ uint8_t* ub = (__ubuf__ uint8_t*)0;
  __ubuf__ float* state_ub = (__ubuf__ float*)(ub + 0);
  __ubuf__ T* q_in = (__ubuf__ T*)(ub + 32768);
  __ubuf__ T* k_in = (__ubuf__ T*)(ub + 49152);
  __ubuf__ T* v_in = (__ubuf__ T*)(ub + 65536);
  __ubuf__ float* g_in = (__ubuf__ float*)(ub + 73728);
  __ubuf__ float* beta_in = (__ubuf__ float*)(ub + 90112);
  __ubuf__ T* out_ub = (__ubuf__ T*)(ub + 143360);

  int32_t dk_pad = ((Dk + 63) / 64) * 64;
  int32_t dv_tiles = (Dv + 63) / 64;
  int32_t total_tasks = B * H * dv_tiles;
  int32_t ncores = (int32_t)block_num;
  int32_t cid = (int32_t)block_idx;
  int32_t base = total_tasks / ncores;
  int32_t rem = total_tasks % ncores;
  int32_t start = cid * base + (cid < rem ? cid : rem);
  int32_t count = base + (cid < rem ? 1 : 0);

  for (int32_t lt = 0; lt < count; ++lt) {
    int32_t tid = start + lt;
    int32_t bh = tid / dv_tiles;
    int32_t dv_t = tid % dv_tiles;
    int32_t b = bh / H;
    int32_t h = bh % H;
    int32_t dv0 = dv_t * 64;
    int32_t dv_cur = (Dv - dv0 < 64) ? (Dv - dv0) : 64;

    int64_t state_off = ((int64_t)b * H + h) * Dk * Dv + dv0;
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)state_ub, (__gm__ uint8_t*)(init_state + state_off),
                         (uint32_t)Dk, (uint32_t)(dv_cur * 4), 0, 0, false,
                         asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                         (uint64_t)(Dv * 4), (uint32_t)(64 * 4));
    int64_t qk_off = ((int64_t)b * T_len * H + h) * Dk;
    int64_t v_off = ((int64_t)b * T_len * H + h) * Dv + dv0;
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)q_in, (__gm__ uint8_t*)(q + qk_off),
                         (uint32_t)T_len, (uint32_t)(Dk * sizeof(T)), 0, 0, false,
                         asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                         (uint64_t)(H * Dk * sizeof(T)), (uint32_t)(dk_pad * sizeof(T)));
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)k_in, (__gm__ uint8_t*)(k + qk_off),
                         (uint32_t)T_len, (uint32_t)(Dk * sizeof(T)), 0, 0, false,
                         asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                         (uint64_t)(H * Dk * sizeof(T)), (uint32_t)(dk_pad * sizeof(T)));
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)v_in, (__gm__ uint8_t*)(v + v_off),
                         (uint32_t)T_len, (uint32_t)(dv_cur * sizeof(T)), 0, 0, false,
                         asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                         (uint64_t)(H * Dv * sizeof(T)), (uint32_t)(64 * sizeof(T)));
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)g_in, (__gm__ uint8_t*)(g + qk_off),
                         (uint32_t)T_len, (uint32_t)(Dk * 4), 0, 0, false,
                         asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                         (uint64_t)(H * Dk * 4), (uint32_t)(dk_pad * 4));
    for (int32_t t = 0; t < T_len; ++t) {
      beta_in[t] = beta[((int64_t)b * T_len + t) * H + h];
    }

    asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID0);
    asc_sync_notify(PIPE_S, PIPE_V, EVENT_ID0);
    asc_sync_wait(PIPE_S, PIPE_V, EVENT_ID0);

    kda_vec_fast_vf<T>(ub, T_len, Dk, dk_pad, dv_cur, use_l2norm, scale_val);

    asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_copy_ub2gm_align((__gm__ uint8_t*)(y + v_off), (__ubuf__ uint8_t*)out_ub,
                         (uint32_t)T_len, (uint32_t)(dv_cur * sizeof(T)),
                         asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                         (uint64_t)(H * Dv * sizeof(T)), (uint32_t)(64 * sizeof(T)));
    asc_copy_ub2gm_align((__gm__ uint8_t*)(final_state + state_off), (__ubuf__ uint8_t*)state_ub,
                         (uint32_t)Dk, (uint32_t)(dv_cur * 4),
                         asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                         (uint64_t)(Dv * 4), (uint32_t)(64 * 4));
    asc_sync_notify(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    asc_sync_wait(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID0);
    asc_sync_notify(PIPE_MTE3, PIPE_V, EVENT_ID0);
    asc_sync_wait(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  asc_sync();
}

extern "C" __global__ __aicore__ __vector__ void kda_vec_fast_f16(
    __gm__ half* q, __gm__ half* k, __gm__ half* v, __gm__ float* g, __gm__ float* beta, __gm__ float* init_state,
    __gm__ half* y, __gm__ float* final_state,
    int32_t B, int32_t T_len, int32_t H, int32_t Dk, int32_t Dv, int32_t use_l2norm, float scale_val) {
  kda_vec_fast_impl<half>(q, k, v, g, beta, init_state, y, final_state, B, T_len, H, Dk, Dv, use_l2norm, scale_val);
}
extern "C" __global__ __aicore__ __vector__ void kda_vec_fast_bf16(
    __gm__ bfloat16_t* q, __gm__ bfloat16_t* k, __gm__ bfloat16_t* v, __gm__ float* g, __gm__ float* beta, __gm__ float* init_state,
    __gm__ bfloat16_t* y, __gm__ float* final_state,
    int32_t B, int32_t T_len, int32_t H, int32_t Dk, int32_t Dv, int32_t use_l2norm, float scale_val) {
  kda_vec_fast_impl<bfloat16_t>(q, k, v, g, beta, init_state, y, final_state, B, T_len, H, Dk, Dv, use_l2norm, scale_val);
}
extern "C" __global__ __aicore__ __vector__ void kda_vec_fast_f32(
    __gm__ float* q, __gm__ float* k, __gm__ float* v, __gm__ float* g, __gm__ float* beta, __gm__ float* init_state,
    __gm__ float* y, __gm__ float* final_state,
    int32_t B, int32_t T_len, int32_t H, int32_t Dk, int32_t Dv, int32_t use_l2norm, float scale_val) {
  kda_vec_fast_impl<float>(q, k, v, g, beta, init_state, y, final_state, B, T_len, H, Dk, Dv, use_l2norm, scale_val);
}

// ============================================================================
// Kernel 2: Cube-Accelerated Chunked WY Stage 1 (__mix__(1, 2), WY_C = 32)
// ============================================================================
constexpr int WY_C = 32;
constexpr int WY_H = 16;

// Transpose [16, N] float -> [N, 16] float using hardware asc_transto5hd_b32
__aicore__ inline void transpose_16xN_to_Nx16(__ubuf__ float* src, __ubuf__ float* dst, int32_t N) {
  __ubuf__ float* sp[16];
  __ubuf__ float* dp[16];
  for (int32_t i = 0; i < 16; ++i) sp[i] = src + i * N;
  for (int32_t r = 0; r < 8; ++r) {
    dp[2 * r + 0] = dst + r * 16 + 0;
    dp[2 * r + 1] = dst + r * 16 + 8;
  }
  asc_set_va_reg(VA0, sp);
  asc_set_va_reg(VA1, sp + 8);
  asc_set_va_reg(VA2, dp);
  asc_set_va_reg(VA3, dp + 8);
  asc_sync_notify(PIPE_S, PIPE_V, EVENT_ID3);
  asc_sync_wait(PIPE_S, PIPE_V, EVENT_ID3);
  asc_transto5hd_b32(VA2, VA0, (uint8_t)(N / 8), 16, 1);
  asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID3);
  asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID3);
}

template <typename T>
__simd_vf__ inline void stage1_prep_vf(
    __ubuf__ T* q_in, __ubuf__ T* k_in, __ubuf__ T* v_in, __ubuf__ float* g_in,
    __ubuf__ float* q_f, __ubuf__ float* k_f, __ubuf__ float* v_f, __ubuf__ float* g_cum, __ubuf__ float* g_rev,
    __ubuf__ float* q_tilde, __ubuf__ float* kv_tilde_v, __ubuf__ float* khat_out, __ubuf__ float* gamma_end,
    int32_t sid, int32_t c_len, int32_t Dk, int32_t Dv, int32_t dk_pad, int32_t dv_pad,
    int32_t use_l2norm, float scale_val) {
  auto m32 = asc_create_mask_b32(PAT_ALL);
  auto v_zero = V::vdup<float>(0.0f, m32, MODE_ZEROING);
  auto v_one = V::vdup<float>(1.0f, m32, MODE_ZEROING);
  auto v_eps = V::vdup<float>(1.0e-6f, m32, MODE_ZEROING);
  int32_t dk_chunks = (Dk + 63) / 64;
  int32_t dk_tail = Dk - (dk_chunks - 1) * 64;
  int32_t dv_chunks = (Dv + 63) / 64;
  int32_t dv_tail = Dv - (dv_chunks - 1) * 64;
  int32_t n_total = dk_pad + dv_pad;

  // 1. Compute forward cumsum(g_in) AND backward reverse-cumsum(g_in) in registers (zero subtraction cancellation!)
  for (int32_t c = 0; c < dk_pad / 64; ++c) {
    if (c < dk_chunks) {
      int32_t w = (c == dk_chunks - 1) ? dk_tail : 64;
      auto mc = (w == 64) ? m32 : make_mask_b32((uint32_t)w);
      auto acc_f = v_zero;
      for (int32_t t = 0; t < c_len; ++t) {
        auto gv = V::vlds_norm(g_in + t * dk_pad + c * 64, 0);
        acc_f = V::vadd(acc_f, gv, mc, MODE_ZEROING);
        V::vsts_norm(acc_f, g_cum + t * dk_pad + c * 64, 0, m32);
      }
      V::vsts_norm(V::vexp(acc_f, mc, MODE_ZEROING), gamma_end + c * 64, 0, m32);
      auto acc_r = v_zero;
      V::vsts_norm(v_zero, g_rev + (c_len - 1) * dk_pad + c * 64, 0, m32);
      for (int32_t t = c_len - 2; t >= 0; --t) {
        auto gv1 = V::vlds_norm(g_in + (t + 1) * dk_pad + c * 64, 0);
        acc_r = V::vadd(acc_r, gv1, mc, MODE_ZEROING);
        V::vsts_norm(acc_r, g_rev + t * dk_pad + c * 64, 0, m32);
      }
    } else {
      V::vsts_norm(v_zero, gamma_end + c * 64, 0, m32);
    }
  }
  asc_mem_bar(VST_VLD);

  // 2. Each AIV processes its 16 tokens in VREGs (zero barriers inside token loop!)
  int32_t t_base = sid * WY_H;
  int32_t my_len = c_len - t_base;
  if (my_len < 0) my_len = 0;
  if (my_len > WY_H) my_len = WY_H;

  for (int32_t tl = 0; tl < WY_H; ++tl) {
    if (tl < my_len) {
      int32_t tg = t_base + tl;
      if (dk_chunks == 1) {
        auto mc = (Dk == 64) ? m32 : make_mask_b32((uint32_t)Dk);
        auto qv = load_f32_vec(q_in, tl * dk_pad);
        auto kv = load_f32_vec(k_in, tl * dk_pad);
        if (use_l2norm) {
          auto qs = V::vdupv(V::vcadd(V::vmul(qv, qv, mc, MODE_ZEROING), mc, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
          auto ks = V::vdupv(V::vcadd(V::vmul(kv, kv, mc, MODE_ZEROING), mc, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
          auto qsc = V::vmuls(V::vdiv_0ulp_ftz_true(v_one, V::vsqrt(V::vadd(qs, v_eps, m32, MODE_ZEROING), m32, MODE_ZEROING), m32, MODE_ZEROING), scale_val, m32, MODE_ZEROING);
          auto ksc = V::vdiv_0ulp_ftz_true(v_one, V::vsqrt(V::vadd(ks, v_eps, m32, MODE_ZEROING), m32, MODE_ZEROING), m32, MODE_ZEROING);
          qv = V::vmul(qv, qsc, mc, MODE_ZEROING);
          kv = V::vmul(kv, ksc, mc, MODE_ZEROING);
        } else {
          qv = V::vmuls(qv, scale_val, mc, MODE_ZEROING);
          kv = V::vmul(kv, v_one, mc, MODE_ZEROING);
        }
        V::vsts_norm(qv, q_f + tl * dk_pad, 0, m32);
        V::vsts_norm(kv, k_f + tl * dk_pad, 0, m32);
        auto gt = V::vlds_norm(g_cum + tg * dk_pad, 0);
        auto gam_t = V::vexp(gt, mc, MODE_ZEROING);
        auto gam_e = (tg == c_len - 1) ? v_one : V::vexp(V::vlds_norm(g_rev + tg * dk_pad, 0), mc, MODE_ZEROING);
        V::vsts_norm(V::vmul(qv, gam_t, mc, MODE_ZEROING), q_tilde + tl * dk_pad, 0, m32);
        V::vsts_norm(V::vmul(kv, gam_t, mc, MODE_ZEROING), kv_tilde_v + tl * n_total, 0, m32);
        V::vsts_norm(V::vmul(kv, gam_e, mc, MODE_ZEROING), khat_out + tl * dk_pad, 0, m32);
        for (int32_t c = 1; c < dk_pad / 64; ++c) {
          V::vsts_norm(v_zero, q_tilde + tl * dk_pad + c * 64, 0, m32);
          V::vsts_norm(v_zero, kv_tilde_v + tl * n_total + c * 64, 0, m32);
          V::vsts_norm(v_zero, khat_out + tl * dk_pad + c * 64, 0, m32);
        }
      } else {
        auto mc1 = (dk_tail == 64) ? m32 : make_mask_b32((uint32_t)dk_tail);
        auto q0 = load_f32_vec(q_in, tl * dk_pad);
        auto q1 = load_f32_vec(q_in, tl * dk_pad + 64);
        auto k0 = load_f32_vec(k_in, tl * dk_pad);
        auto k1 = load_f32_vec(k_in, tl * dk_pad + 64);
        if (use_l2norm) {
          auto q_sq = V::vadd(V::vmul(q0, q0, m32, MODE_ZEROING), V::vmul(q1, q1, mc1, MODE_ZEROING), m32, MODE_ZEROING);
          auto k_sq = V::vadd(V::vmul(k0, k0, m32, MODE_ZEROING), V::vmul(k1, k1, mc1, MODE_ZEROING), m32, MODE_ZEROING);
          auto qs = V::vdupv(V::vcadd(q_sq, m32, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
          auto ks = V::vdupv(V::vcadd(k_sq, m32, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
          auto qsc = V::vmuls(V::vdiv_0ulp_ftz_true(v_one, V::vsqrt(V::vadd(qs, v_eps, m32, MODE_ZEROING), m32, MODE_ZEROING), m32, MODE_ZEROING), scale_val, m32, MODE_ZEROING);
          auto ksc = V::vdiv_0ulp_ftz_true(v_one, V::vsqrt(V::vadd(ks, v_eps, m32, MODE_ZEROING), m32, MODE_ZEROING), m32, MODE_ZEROING);
          q0 = V::vmul(q0, qsc, m32, MODE_ZEROING); q1 = V::vmul(q1, qsc, mc1, MODE_ZEROING);
          k0 = V::vmul(k0, ksc, m32, MODE_ZEROING); k1 = V::vmul(k1, ksc, mc1, MODE_ZEROING);
        } else {
          q0 = V::vmuls(q0, scale_val, m32, MODE_ZEROING); q1 = V::vmuls(q1, scale_val, mc1, MODE_ZEROING);
          k1 = V::vmul(k1, v_one, mc1, MODE_ZEROING);
        }
        V::vsts_norm(q0, q_f + tl * dk_pad, 0, m32); V::vsts_norm(q1, q_f + tl * dk_pad + 64, 0, m32);
        V::vsts_norm(k0, k_f + tl * dk_pad, 0, m32); V::vsts_norm(k1, k_f + tl * dk_pad + 64, 0, m32);
        auto gt0 = V::vlds_norm(g_cum + tg * dk_pad, 0);
        auto gt1 = V::vlds_norm(g_cum + tg * dk_pad + 64, 0);
        auto gam_t0 = V::vexp(gt0, m32, MODE_ZEROING);
        auto gam_t1 = V::vexp(gt1, mc1, MODE_ZEROING);
        auto gam_e0 = (tg == c_len - 1) ? v_one : V::vexp(V::vlds_norm(g_rev + tg * dk_pad, 0), m32, MODE_ZEROING);
        auto gam_e1 = (tg == c_len - 1) ? v_one : V::vexp(V::vlds_norm(g_rev + tg * dk_pad + 64, 0), mc1, MODE_ZEROING);
        V::vsts_norm(V::vmul(q0, gam_t0, m32, MODE_ZEROING), q_tilde + tl * dk_pad, 0, m32);
        V::vsts_norm(V::vmul(q1, gam_t1, mc1, MODE_ZEROING), q_tilde + tl * dk_pad + 64, 0, m32);
        V::vsts_norm(V::vmul(k0, gam_t0, m32, MODE_ZEROING), kv_tilde_v + tl * n_total, 0, m32);
        V::vsts_norm(V::vmul(k1, gam_t1, mc1, MODE_ZEROING), kv_tilde_v + tl * n_total + 64, 0, m32);
        V::vsts_norm(V::vmul(k0, gam_e0, m32, MODE_ZEROING), khat_out + tl * dk_pad, 0, m32);
        V::vsts_norm(V::vmul(k1, gam_e1, mc1, MODE_ZEROING), khat_out + tl * dk_pad + 64, 0, m32);
      }
      for (int32_t c = 0; c < dv_pad / 64; ++c) {
        if (c < dv_chunks) {
          int32_t w = (c == dv_chunks - 1) ? dv_tail : 64;
          auto mc = (w == 64) ? m32 : make_mask_b32((uint32_t)w);
          auto vv = load_f32_vec(v_in, tl * dv_pad + c * 64);
          V::vsts_norm(V::vmul(vv, v_one, mc, MODE_ZEROING), kv_tilde_v + tl * n_total + dk_pad + c * 64, 0, m32);
        } else {
          V::vsts_norm(v_zero, kv_tilde_v + tl * n_total + dk_pad + c * 64, 0, m32);
        }
      }
    } else {
      for (int32_t c = 0; c < dk_pad / 64; ++c) {
        V::vsts_norm(v_zero, q_f + tl * dk_pad + c * 64, 0, m32);
        V::vsts_norm(v_zero, k_f + tl * dk_pad + c * 64, 0, m32);
        V::vsts_norm(v_zero, q_tilde + tl * dk_pad + c * 64, 0, m32);
        V::vsts_norm(v_zero, khat_out + tl * dk_pad + c * 64, 0, m32);
        V::vsts_norm(v_zero, kv_tilde_v + tl * n_total + c * 64, 0, m32);
      }
      for (int32_t c = 0; c < dv_pad / 64; ++c) {
        V::vsts_norm(v_zero, kv_tilde_v + tl * n_total + dk_pad + c * 64, 0, m32);
      }
    }
  }
  asc_mem_bar(VST_VLD);
}

template <int DK_PAD, bool BUILD_B, typename T>
__simd_vf__ inline void stage1_build_kp_kn_vf(
    __ubuf__ float* q_f, __ubuf__ float* k_f, __ubuf__ float* g_cum,
    __ubuf__ float* a_f, __ubuf__ float* b_f,
    __ubuf__ T* a_nz, __ubuf__ T* b_nz,
    int32_t t_base, int32_t my_len, int32_t anchor_g_idx, int32_t Dk) {
  auto m32 = asc_create_mask_b32(PAT_ALL);
  auto v_zero = V::vdup<float>(0.0f, m32, MODE_ZEROING);
  int32_t dk_chunks = (Dk + 63) / 64;
  int32_t dk_tail = Dk - (dk_chunks - 1) * 64;
  for (int32_t tl = 0; tl < WY_H; ++tl) {
    for (int32_t c = 0; c < DK_PAD / 64; ++c) {
      if (tl < my_len && c < dk_chunks) {
        int32_t w = (c == dk_chunks - 1) ? dk_tail : 64;
        auto mc = (w == 64) ? m32 : make_mask_b32((uint32_t)w);
        auto gt = V::vlds_norm(g_cum + (t_base + tl) * DK_PAD + c * 64, 0);
        auto ga = V::vlds_norm(g_cum + anchor_g_idx * DK_PAD + c * 64, 0);
        auto ep = V::vexp(V::vsub(gt, ga, mc, MODE_ZEROING), mc, MODE_ZEROING);
        auto kt = V::vlds_norm(k_f + tl * DK_PAD + c * 64, 0);
        auto qt = V::vlds_norm(q_f + tl * DK_PAD + c * 64, 0);
        V::vsts_norm(V::vmul(kt, ep, mc, MODE_ZEROING), a_f + tl * DK_PAD + c * 64, 0, m32);
        V::vsts_norm(V::vmul(qt, ep, mc, MODE_ZEROING), a_f + (WY_H + tl) * DK_PAD + c * 64, 0, m32);
        if constexpr (BUILD_B) {
          auto en = V::vexp(V::vsub(ga, gt, mc, MODE_ZEROING), mc, MODE_ZEROING);
          V::vsts_norm(V::vmul(kt, en, mc, MODE_ZEROING), b_f + tl * DK_PAD + c * 64, 0, m32);
        }
      } else {
        V::vsts_norm(v_zero, a_f + tl * DK_PAD + c * 64, 0, m32);
        V::vsts_norm(v_zero, a_f + (WY_H + tl) * DK_PAD + c * 64, 0, m32);
        if constexpr (BUILD_B) V::vsts_norm(v_zero, b_f + tl * DK_PAD + c * 64, 0, m32);
      }
    }
  }
  asc_mem_bar(VST_VLD);
  scatter_nz_callee<32, DK_PAD, 32, T>(a_f, a_nz);
  if constexpr (BUILD_B) {
    scatter_nz_callee<16, DK_PAD, 16, T>(b_f, b_nz);
  }
}

template <int DK_PAD, typename T>
__simd_vf__ inline void stage1_build_kn_only_vf(
    __ubuf__ float* k_f, __ubuf__ float* g_cum, __ubuf__ float* b_f, __ubuf__ T* b_nz,
    int32_t my_len, int32_t anchor_g_idx, int32_t Dk) {
  auto m32 = asc_create_mask_b32(PAT_ALL);
  auto v_zero = V::vdup<float>(0.0f, m32, MODE_ZEROING);
  int32_t dk_chunks = (Dk + 63) / 64;
  int32_t dk_tail = Dk - (dk_chunks - 1) * 64;
  for (int32_t tl = 0; tl < WY_H; ++tl) {
    for (int32_t c = 0; c < DK_PAD / 64; ++c) {
      if (tl < my_len && c < dk_chunks) {
        int32_t w = (c == dk_chunks - 1) ? dk_tail : 64;
        auto mc = (w == 64) ? m32 : make_mask_b32((uint32_t)w);
        auto gs = V::vlds_norm(g_cum + tl * DK_PAD + c * 64, 0);
        auto ga = V::vlds_norm(g_cum + anchor_g_idx * DK_PAD + c * 64, 0);
        auto en = V::vexp(V::vsub(ga, gs, mc, MODE_ZEROING), mc, MODE_ZEROING);
        auto ks = V::vlds_norm(k_f + tl * DK_PAD + c * 64, 0);
        V::vsts_norm(V::vmul(ks, en, mc, MODE_ZEROING), b_f + tl * DK_PAD + c * 64, 0, m32);
      } else {
        V::vsts_norm(v_zero, b_f + tl * DK_PAD + c * 64, 0, m32);
      }
    }
  }
  asc_mem_bar(VST_VLD);
  scatter_nz_callee<16, DK_PAD, 16, T>(b_f, b_nz);
}

template <typename T>
__simd_vf__ inline void stage1_solve_B_vf(
    __ubuf__ float* g_00, __ubuf__ float* g_11, __ubuf__ float* g_10,
    __ubuf__ float* beta_in, __ubuf__ float* A_mat, __ubuf__ float* B_mat,
    __ubuf__ T* B_nz, int32_t c_len) {
  auto m32 = asc_create_mask_b32(PAT_ALL);
  auto m16_l = make_mask_b32(16);
  auto m32_w = make_mask_b32(32);
  auto v_zero = V::vdup<float>(0.0f, m32, MODE_ZEROING);
  for (int32_t i = 0; i < 16; ++i) {
    V::vsts_norm(v_zero, A_mat + i * 64, 0, m32);
    V::vsts_norm(v_zero, B_mat + i * 64, 0, m32);
  }
  asc_mem_bar(VST_VLD);
  for (int32_t t = 0; t < c_len; ++t) {
    auto bt = V::vlds_brc_elem(beta_in + t, 0);
    if (t < 16) {
      if (t > 0) {
        auto mt = make_mask_b32((uint32_t)t);
        auto row = V::vmul(V::vlds_norm(g_00 + t * 16, 0), bt, mt, MODE_ZEROING);
        V::vsts_norm(row, A_mat + t * 32, 0, mt);
      }
    } else {
      int32_t tl = t - 16;
      auto r_lo = V::vmul(V::vlds_norm(g_10 + tl * 16, 0), bt, m16_l, MODE_ZEROING);
      V::vsts_norm(r_lo, A_mat + t * 32, 0, m16_l);
      if (tl > 0) {
        auto mt = make_mask_b32((uint32_t)tl);
        auto r_hi = V::vmul(V::vlds_norm(g_11 + tl * 16, 0), bt, mt, MODE_ZEROING);
        V::vsts_norm(r_hi, A_mat + t * 32 + 16, 0, mt);
      }
    }
  }
  asc_mem_bar(VST_VLD);
  for (int32_t t = 0; t < c_len; ++t) {
    auto acc = v_zero;
    for (int32_t s = 0; s < t; ++s) {
      auto ats = V::vlds_brc_elem(A_mat + t * 32 + s, 0);
      auto bs = V::vlds_norm(B_mat + s * 32, 0);
      acc = V::vsub(acc, V::vmul(bs, ats, m32_w, MODE_ZEROING), m32_w, MODE_ZEROING);
    }
    V::vsts_norm(acc, B_mat + t * 32, 0, m32_w);
    asc_mem_bar(VST_VLD);
    V::vsts_1st(V::vlds_brc_elem(beta_in + t, 0), B_mat + t * 32 + t, 0, m32);
    asc_mem_bar(VST_VLD);
  }
  scatter_nz_callee<32, 32, 32, T>(B_mat, B_nz);
}

template <typename T>
__simd_vf__ inline void stage1_mask_S_vf(
    __ubuf__ float* g_00, __ubuf__ float* g_11, __ubuf__ float* g_10,
    __ubuf__ float* S_mat, __ubuf__ T* S_nz, int32_t c_len) {
  auto m32 = asc_create_mask_b32(PAT_ALL);
  auto m16_l = make_mask_b32(16);
  auto v_zero = V::vdup<float>(0.0f, m32, MODE_ZEROING);
  for (int32_t i = 0; i < 16; ++i) {
    V::vsts_norm(v_zero, S_mat + i * 64, 0, m32);
  }
  asc_mem_bar(VST_VLD);
  for (int32_t t = 0; t < c_len; ++t) {
    if (t < 16) {
      auto mt = make_mask_b32((uint32_t)(t + 1));
      V::vsts_norm(V::vlds_norm(g_00 + t * 16, 0), S_mat + t * 32, 0, mt);
    } else {
      int32_t tl = t - 16;
      V::vsts_norm(V::vlds_norm(g_10 + tl * 16, 0), S_mat + t * 32, 0, m16_l);
      auto mt = make_mask_b32((uint32_t)(tl + 1));
      V::vsts_norm(V::vlds_norm(g_11 + tl * 16, 0), S_mat + t * 32 + 16, 0, mt);
    }
  }
  asc_mem_bar(VST_VLD);
  scatter_nz_callee<32, 32, 32, T>(S_mat, S_nz);
}

template <typename T>
__simd_vf__ inline void cast_and_split_wu_vf(
    __ubuf__ float* wu_ub, __ubuf__ T* w_out, __ubuf__ float* u_out,
    int32_t dk_pad, int32_t dv_pad) {
  auto m32 = asc_create_mask_b32(PAT_ALL);
  auto m16 = asc_create_mask_b16(PAT_ALL);
  int32_t n_total = dk_pad + dv_pad;
  for (int32_t t = 0; t < WY_H; ++t) {
    for (int32_t c = 0; c < dk_pad / 64; ++c) {
      store_f32_vec(w_out, t * dk_pad + c * 64, V::vlds_norm(wu_ub + t * n_total + c * 64, 0), m32, m16);
    }
    for (int32_t c = 0; c < dv_pad / 64; ++c) {
      V::vsts_norm(V::vlds_norm(wu_ub + t * n_total + dk_pad + c * 64, 0), u_out + t * dv_pad + c * 64, 0, m32);
    }
  }
  asc_mem_bar(VST_VLD);
}

template <typename T>
__simd_vf__ inline void compute_qeff_yloc_vf(
    __ubuf__ float* q_tilde, __ubuf__ float* sy_ub,
    __ubuf__ T* qeff_out, __ubuf__ float* yloc_out,
    int32_t dk_pad, int32_t dv_pad) {
  auto m32 = asc_create_mask_b32(PAT_ALL);
  auto m16 = asc_create_mask_b16(PAT_ALL);
  int32_t n_total = dk_pad + dv_pad;
  for (int32_t t = 0; t < WY_H; ++t) {
    for (int32_t c = 0; c < dk_pad / 64; ++c) {
      auto qt = V::vlds_norm(q_tilde + t * dk_pad + c * 64, 0);
      auto sw = V::vlds_norm(sy_ub + t * n_total + c * 64, 0);
      store_f32_vec(qeff_out, t * dk_pad + c * 64, V::vsub(qt, sw, m32, MODE_ZEROING), m32, m16);
    }
    for (int32_t c = 0; c < dv_pad / 64; ++c) {
      V::vsts_norm(V::vlds_norm(sy_ub + t * n_total + dk_pad + c * 64, 0), yloc_out + t * dv_pad + c * 64, 0, m32);
    }
  }
  asc_mem_bar(VST_VLD);
}

template <typename T>
__simd_vf__ inline void cast_khat_vf(
    __ubuf__ float* khat_f, __ubuf__ T* khat_out, int32_t dk_pad) {
  auto m32 = asc_create_mask_b32(PAT_ALL);
  auto m16 = asc_create_mask_b16(PAT_ALL);
  for (int32_t t = 0; t < WY_H; ++t) {
    for (int32_t c = 0; c < dk_pad / 64; ++c) {
      store_f32_vec(khat_out, t * dk_pad + c * 64, V::vlds_norm(khat_f + t * dk_pad + c * 64, 0), m32, m16);
    }
  }
  asc_mem_bar(VST_VLD);
}

template <int DK_PAD, int DV_PAD, typename T_io>
__aicore__ inline void kda_wy_stage1_mix_impl(
    __gm__ T_io* q, __gm__ T_io* k, __gm__ T_io* v, __gm__ float* g, __gm__ float* beta,
    __gm__ float* ws_wq, __gm__ float* ws_khat, __gm__ float* ws_uloc, __gm__ float* ws_yloc, __gm__ float* ws_gamma,
    int32_t B, int32_t T_len, int32_t H, int32_t Dk, int32_t Dv,
    int32_t use_l2norm, float scale_val) {
  asc_init();
  using T = float;
  constexpr int32_t N_TOT = DK_PAD + DV_PAD;
  __cbuf__ uint8_t* l1 = (__cbuf__ uint8_t*)0;
  __cc__ float* l0c = (__cc__ float*)0;
  __ubuf__ uint8_t* ub = (__ubuf__ uint8_t*)0;

  // L1 layout (max sizes for float & N_TOT=256):
  // l1_A: [32, DK_PAD] T (max 16KB) at 0
  // l1_B: [16, DK_PAD] T (max 8KB) at 16KB
  // l1_Bmat: [32, 32] T (max 4KB) at 24KB
  // l1_Smat: [32, 32] T (max 4KB) at 28KB
  // l1_kv: [N_TOT, 32] T (max 32KB) at 32KB
  __cbuf__ T* l1_A = (__cbuf__ T*)(l1 + 0);
  __cbuf__ T* l1_B = (__cbuf__ T*)(l1 + 16384);
  __cbuf__ T* l1_Bmat = (__cbuf__ T*)(l1 + 24576);
  __cbuf__ T* l1_Smat = (__cbuf__ T*)(l1 + 28672);
  __cbuf__ T* l1_kv = (__cbuf__ T*)(l1 + 32768);

  int32_t Nc = (T_len + WY_C - 1) / WY_C;
  int32_t total_tasks = B * H * Nc;
  int32_t ncores = (int32_t)block_num;
  int32_t cid = (int32_t)block_idx;
  int32_t base = total_tasks / ncores;
  int32_t rem = total_tasks % ncores;
  int32_t start = cid * base + (cid < rem ? cid : rem);
  int32_t count = base + (cid < rem ? 1 : 0);

  constexpr int32_t C0 = 32 / sizeof(T);
  constexpr int32_t TILE_K_DK = (std::is_same_v<T, float> && DK_PAD > 64) ? 64 : DK_PAD;

  if ASC_IS_AIC {
    for (int32_t lt = 0; lt < count; ++lt) {
      int32_t tid = start + lt;
      int32_t bh = tid / Nc;
      int32_t ch = tid % Nc;
      int32_t t0 = ch * WY_C;
      int32_t c_len = (T_len - t0 < WY_C) ? (T_len - t0) : WY_C;

      // Wait AIV0 & AIV1 UB ready for GEMM 1A output
      asc_sync_intra_wait(PIPE_FIX, 10);
      asc_sync_intra_wait(PIPE_FIX, 26);

      // GEMM 1A: [K_p0; Q_p0] @ K_n0^T -> [A00; S00]
      asc_sync_intra_wait(PIPE_MTE1, 0);
      ascend_gemm_l1<32, DK_PAD, 16, TILE_K_DK, true, T>(l0c, l1_A, l1_B, 1, 0, 0);
      asc_sync_notify(PIPE_M, PIPE_FIX, EVENT_ID0);
      asc_sync_wait(PIPE_M, PIPE_FIX, EVENT_ID0);
      asc_set_l0c_copy_nz_para(1, 0, 0);
      asc_copy_l0c2ub((__ubuf__ float*)(ub + 135168), l0c, 16, 32, 16, 32, 0,
                      asc_dual_dst_mode::DUAL_DST_SPLIT_M, asc_unit_flag_mode::DISABLE,
                      static_cast<asc_quant_mode>(0), asc_relu_pre_mode::NONE, 0, 1, 0, 0);
      asc_sync_intra_arrive(PIPE_FIX, 1);
      asc_sync_intra_arrive(PIPE_FIX, 17);

      if (c_len > 16) {
        // GEMM 1B: AIV1 provides l1_A, l1_B
        asc_sync_intra_wait(PIPE_MTE1, 18);
        ascend_gemm_l1<32, DK_PAD, 16, TILE_K_DK, true, T>(l0c, l1_A, l1_B, 1, 0, 0);
        asc_sync_notify(PIPE_M, PIPE_FIX, EVENT_ID0);
        asc_sync_wait(PIPE_M, PIPE_FIX, EVENT_ID0);
        asc_set_l0c_copy_nz_para(1, 0, 0);
        asc_copy_l0c2ub((__ubuf__ float*)(ub + 137216), l0c, 16, 32, 16, 32, 0,
                        asc_dual_dst_mode::DUAL_DST_SPLIT_M, asc_unit_flag_mode::DISABLE,
                        static_cast<asc_quant_mode>(0), asc_relu_pre_mode::NONE, 0, 1, 0, 0);
        asc_sync_intra_arrive(PIPE_FIX, 3);
        asc_sync_intra_arrive(PIPE_FIX, 19);

        // GEMM 1C: AIV1 provides l1_A, AIV0 provides l1_B
        asc_sync_intra_wait(PIPE_MTE1, 4);
        asc_sync_intra_wait(PIPE_MTE1, 20);
        ascend_gemm_l1<32, DK_PAD, 16, TILE_K_DK, true, T>(l0c, l1_A, l1_B, 1, 0, 0);
        asc_sync_notify(PIPE_M, PIPE_FIX, EVENT_ID0);
        asc_sync_wait(PIPE_M, PIPE_FIX, EVENT_ID0);
        asc_set_l0c_copy_nz_para(1, 0, 0);
        asc_copy_l0c2ub((__ubuf__ float*)(ub + 139264), l0c, 16, 32, 16, 32, 0,
                        asc_dual_dst_mode::DUAL_DST_SPLIT_M, asc_unit_flag_mode::DISABLE,
                        static_cast<asc_quant_mode>(0), asc_relu_pre_mode::NONE, 0, 1, 0, 0);
        asc_sync_intra_arrive(PIPE_FIX, 5);
        asc_sync_intra_arrive(PIPE_FIX, 21);
      }

      // GEMM 2: B_mat[32, 32] @ l1_kv[N_TOT, 32]^T -> [W; U_loc] ([32, N_TOT])
      asc_sync_intra_wait(PIPE_MTE1, 6);
      asc_sync_intra_wait(PIPE_MTE1, 22);
      ascend_gemm_l1<32, 32, N_TOT, 32, true, T>(l0c, l1_Bmat, l1_kv, 1, 0, 0);
      asc_sync_notify(PIPE_M, PIPE_FIX, EVENT_ID0);
      asc_sync_wait(PIPE_M, PIPE_FIX, EVENT_ID0);
      asc_set_l0c_copy_nz_para(1, 0, 0);
      asc_copy_l0c2ub((__ubuf__ float*)(ub + 147456), l0c, (uint16_t)N_TOT, 32, (uint32_t)N_TOT, 32, 0,
                      asc_dual_dst_mode::DUAL_DST_SPLIT_M, asc_unit_flag_mode::DISABLE,
                      static_cast<asc_quant_mode>(0), asc_relu_pre_mode::NONE, 0, 1, 0, 0);
      asc_sync_intra_arrive(PIPE_FIX, 7);
      asc_sync_intra_arrive(PIPE_FIX, 23);

      // GEMM 3: S_mat[32, 32] @ l1_kv(W,U)^T -> [SW; Y_loc]
      asc_sync_intra_wait(PIPE_MTE1, 8);
      asc_sync_intra_wait(PIPE_MTE1, 24);
      ascend_gemm_l1<32, 32, N_TOT, 32, true, T>(l0c, l1_Smat, l1_kv, 1, 0, 0);
      asc_sync_notify(PIPE_M, PIPE_FIX, EVENT_ID0);
      asc_sync_wait(PIPE_M, PIPE_FIX, EVENT_ID0);
      asc_set_l0c_copy_nz_para(1, 0, 0);
      asc_copy_l0c2ub((__ubuf__ float*)(ub + 147456), l0c, (uint16_t)N_TOT, 32, (uint32_t)N_TOT, 32, 0,
                      asc_dual_dst_mode::DUAL_DST_SPLIT_M, asc_unit_flag_mode::DISABLE,
                      static_cast<asc_quant_mode>(0), asc_relu_pre_mode::NONE, 0, 1, 0, 0);
      asc_sync_intra_arrive(PIPE_FIX, 9);
      asc_sync_intra_arrive(PIPE_FIX, 25);
    }
  }

  if ASC_IS_AIV {
    int32_t sid = (int32_t)asc_get_sub_block_id();
    // UB layout (per AIV, < 180KB):
    // q_in[16, 128] T (8KB) at 0
    // k_in[16, 128] T (8KB) at 8KB
    // v_in[16, 128] T (8KB) at 16KB
    // g_in[32, 128] float (16KB) at 24KB
    // beta_in[32] float at 40KB
    // q_f[16, 128] float (8KB) at 42KB
    // k_f[16, 128] float (8KB) at 50KB
    // v_f[16, 128] float (8KB) at 58KB
    // g_cum[32, 128] float (16KB) at 66KB
    // q_tilde[16, 128] float (8KB) at 82KB
    // kv_tilde_v[16, 256] float (16KB) at 90KB (overlays a_f[32,128]=16KB? Wait: keep separate!)
    // a_f[32, 128] float (16KB) at 106KB
    // b_f[16, 128] float (8KB) at 122KB
    // g_00: 128KB (131072), g_11: 130KB (133120), g_10: 132KB (135168)
    // mat_ub[32, 32] float (4KB) at 136KB (139264)
    // wu_ub[16, 256] float (16KB) at 140KB (143360) -- shared with sy_ub (GEMM 2 & 3!)
    // khat_f[16, 128] float (8KB) at 156KB (159744)
    // nz_tmp[256, 16] T (16KB) at 164KB (167936)
    // tr_tmp[256, 16] float (16KB) at 180KB (184320)
    // out_wq[16, 128] T (8KB) at 196KB (200704), out_uloc[16, 128] float (8KB) at 204KB (208896), gamma_end[128] float (0.5KB) at 212KB (217088)
    __ubuf__ T_io* q_in = (__ubuf__ T_io*)(ub + 0);
    __ubuf__ T_io* k_in = (__ubuf__ T_io*)(ub + 8192);
    __ubuf__ T_io* v_in = (__ubuf__ T_io*)(ub + 16384);
    __ubuf__ float* g_in = (__ubuf__ float*)(ub + 24576);
    __ubuf__ float* beta_in = (__ubuf__ float*)(ub + 40960);
    __ubuf__ float* q_f = (__ubuf__ float*)(ub + 43008);
    __ubuf__ float* k_f = (__ubuf__ float*)(ub + 51200);
    __ubuf__ float* v_f = (__ubuf__ float*)(ub + 59392);
    __ubuf__ float* g_cum = (__ubuf__ float*)(ub + 67584);
    __ubuf__ float* q_tilde = (__ubuf__ float*)(ub + 83968);
    __ubuf__ float* kv_tilde_v = (__ubuf__ float*)(ub + 92160);
    __ubuf__ float* a_f = (__ubuf__ float*)(ub + 108544);
    __ubuf__ float* b_f = (__ubuf__ float*)(ub + 124928);
    __ubuf__ float* g_00 = (__ubuf__ float*)(ub + 135168);
    __ubuf__ float* g_11 = (__ubuf__ float*)(ub + 137216);
    __ubuf__ float* g_10 = (__ubuf__ float*)(ub + 139264);
    __ubuf__ float* mat_ub = (__ubuf__ float*)(ub + 143360);
    __ubuf__ float* wu_ub = (__ubuf__ float*)(ub + 147456);
    __ubuf__ float* khat_f = (__ubuf__ float*)(ub + 163840);
    __ubuf__ T* nz_tmp = (__ubuf__ T*)(ub + 172032);
    __ubuf__ float* tr_tmp = (__ubuf__ float*)(ub + 196608);
    __ubuf__ T* out_wq = (__ubuf__ T*)(ub + 212992);
    __ubuf__ float* out_loc = (__ubuf__ float*)(ub + 221184);
    __ubuf__ float* gamma_end = (__ubuf__ float*)(ub + 229376);

    for (int32_t lt = 0; lt < count; ++lt) {
      int32_t tid = start + lt;
      int32_t bh = tid / Nc;
      int32_t ch = tid % Nc;
      int32_t b = bh / H;
      int32_t h = bh % H;
      int32_t t0 = ch * WY_C;
      int32_t c_len = (T_len - t0 < WY_C) ? (T_len - t0) : WY_C;
      int32_t t_base = sid * WY_H;
      int32_t my_len = c_len - t_base;
      if (my_len < 0) my_len = 0;
      if (my_len > WY_H) my_len = WY_H;

      // Signal AIC UB free for GEMM 1A
      asc_sync_intra_arrive(PIPE_V, 10);

      // Load g[0..c_len-1], and sid's slice of q, k, v, beta
      int64_t g_off = (((int64_t)b * T_len + t0) * H + h) * Dk;
      asc_copy_gm2ub_align((__ubuf__ uint8_t*)g_in, (__gm__ uint8_t*)(g + g_off),
                           (uint32_t)c_len, (uint32_t)(Dk * 4), 0, 0, false,
                           asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                           (uint64_t)(H * Dk * 4), (uint32_t)(DK_PAD * 4));
      if (my_len > 0) {
        int64_t qk_off = (((int64_t)b * T_len + t0 + t_base) * H + h) * Dk;
        int64_t v_off = (((int64_t)b * T_len + t0 + t_base) * H + h) * Dv;
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)q_in, (__gm__ uint8_t*)(q + qk_off),
                             (uint32_t)my_len, (uint32_t)(Dk * sizeof(T_io)), 0, 0, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)(H * Dk * sizeof(T_io)), (uint32_t)(DK_PAD * sizeof(T_io)));
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)k_in, (__gm__ uint8_t*)(k + qk_off),
                             (uint32_t)my_len, (uint32_t)(Dk * sizeof(T_io)), 0, 0, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)(H * Dk * sizeof(T_io)), (uint32_t)(DK_PAD * sizeof(T_io)));
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)v_in, (__gm__ uint8_t*)(v + v_off),
                             (uint32_t)my_len, (uint32_t)(Dv * sizeof(T_io)), 0, 0, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)(H * Dv * sizeof(T_io)), (uint32_t)(DV_PAD * sizeof(T_io)));
      }
      if (sid == 0) {
        for (int32_t t = 0; t < c_len; ++t) {
          beta_in[t] = beta[((int64_t)b * T_len + (t0 + t)) * H + h];
        }
      }
      asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);
      asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID0);
      asc_sync_notify(PIPE_S, PIPE_V, EVENT_ID0);
      asc_sync_wait(PIPE_S, PIPE_V, EVENT_ID0);

      stage1_prep_vf<T_io>(q_in, k_in, v_in, g_in, q_f, k_f, v_f, g_cum, a_f, q_tilde, kv_tilde_v, khat_f, gamma_end,
                        sid, c_len, Dk, Dv, DK_PAD, DV_PAD, use_l2norm, scale_val);

      // Build and push GEMM 1A from AIV0
      if (sid == 0) {
        int32_t m0 = (my_len > 8) ? 8 : (my_len - 1);
        stage1_build_kp_kn_vf<DK_PAD, true, T>(q_f, k_f, g_cum, a_f, b_f, nz_tmp, (__ubuf__ T*)(nz_tmp + 32 * DK_PAD),
                                               0, my_len, m0, Dk);
        asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
        asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
        asc_copy_ub2l1((__cbuf__ void*)l1_A, (__ubuf__ void*)nz_tmp, (uint16_t)(DK_PAD / C0), 32 * C0 * sizeof(T) / 32, 0, 0);
        asc_copy_ub2l1((__cbuf__ void*)l1_B, (__ubuf__ void*)(nz_tmp + 32 * DK_PAD), (uint16_t)(DK_PAD / C0), 16 * C0 * sizeof(T) / 32, 0, 0);
        asc_sync_intra_arrive(PIPE_MTE3, 0);
      }

      asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID1);
      asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID1);
      transpose_16xN_to_Nx16(kv_tilde_v, tr_tmp, N_TOT);

      // Wait GEMM 1A result
      asc_sync_intra_wait(PIPE_V, 1);

      if (c_len > 16) {
        if (sid == 1) {
          int32_t m1 = 16 + ((my_len > 8) ? 8 : (my_len - 1));
          stage1_build_kp_kn_vf<DK_PAD, true, T>(q_f, k_f, g_cum, a_f, b_f, nz_tmp, (__ubuf__ T*)(nz_tmp + 32 * DK_PAD),
                                                 16, my_len, m1, Dk);
          asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
          asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
          asc_copy_ub2l1((__cbuf__ void*)l1_A, (__ubuf__ void*)nz_tmp, (uint16_t)(DK_PAD / C0), 32 * C0 * sizeof(T) / 32, 0, 0);
          asc_copy_ub2l1((__cbuf__ void*)l1_B, (__ubuf__ void*)(nz_tmp + 32 * DK_PAD), (uint16_t)(DK_PAD / C0), 16 * C0 * sizeof(T) / 32, 0, 0);
          asc_sync_intra_arrive(PIPE_MTE3, 2);
        }
        asc_sync_intra_wait(PIPE_V, 3);

        // GEMM 1C: AIV1 pushes l1_A, AIV0 pushes l1_B
        if (sid == 1) {
          stage1_build_kp_kn_vf<DK_PAD, false, T>(q_f, k_f, g_cum, a_f, b_f, nz_tmp, (__ubuf__ T*)nz_tmp,
                                                  16, my_len, 16, Dk);
          asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
          asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
          asc_copy_ub2l1((__cbuf__ void*)l1_A, (__ubuf__ void*)nz_tmp, (uint16_t)(DK_PAD / C0), 32 * C0 * sizeof(T) / 32, 0, 0);
          asc_sync_intra_arrive(PIPE_MTE3, 4);
        } else {
          stage1_build_kn_only_vf<DK_PAD, T>(k_f, g_cum, b_f, nz_tmp, my_len, 16, Dk);
          asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
          asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
          asc_copy_ub2l1((__cbuf__ void*)l1_B, (__ubuf__ void*)nz_tmp, (uint16_t)(DK_PAD / C0), 16 * C0 * sizeof(T) / 32, 0, 0);
          asc_sync_intra_arrive(PIPE_MTE3, 4);
        }
        asc_sync_intra_wait(PIPE_V, 5);
      }

      // Solve B (AIV0) / mask S (AIV1), and push l1_kv[:, sid*16 : sid*16+16]
      asc_sync_notify(PIPE_MTE3, PIPE_V, EVENT_ID0);
      asc_sync_wait(PIPE_MTE3, PIPE_V, EVENT_ID0);
      scatter_nz_vf<N_TOT, 16, N_TOT, T>(tr_tmp, nz_tmp);
      asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID1);
      asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID1);
      asc_copy_ub2l1((__cbuf__ void*)(l1_kv + sid * N_TOT * 16), (__ubuf__ void*)nz_tmp,
                     (uint16_t)(16 / C0), N_TOT * C0 * sizeof(T) / 32, 0, 0);
      asc_sync_notify(PIPE_MTE3, PIPE_V, EVENT_ID1);
      asc_sync_wait(PIPE_MTE3, PIPE_V, EVENT_ID1);

      if (sid == 0) {
        stage1_solve_B_vf<T>(g_00, g_11, g_10, beta_in, a_f, mat_ub, nz_tmp, c_len);
        asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID1);
        asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID1);
        asc_copy_ub2l1((__cbuf__ void*)l1_Bmat, (__ubuf__ void*)nz_tmp, (uint16_t)(32 / C0), 32 * C0 * sizeof(T) / 32, 0, 0);
        asc_sync_intra_arrive(PIPE_MTE3, 6);
      } else {
        stage1_mask_S_vf<T>(g_00, g_11, g_10, mat_ub, nz_tmp, c_len);
        asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID1);
        asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID1);
        asc_copy_ub2l1((__cbuf__ void*)l1_Smat, (__ubuf__ void*)nz_tmp, (uint16_t)(32 / C0), 32 * C0 * sizeof(T) / 32, 0, 0);
        asc_sync_intra_arrive(PIPE_MTE3, 6);
      }

      // Wait GEMM 2 result in wu_ub[16, N_TOT]
      asc_sync_intra_wait(PIPE_V, 7);

      // Cast W -> out_wq, split U_loc -> out_loc, write to ws_wq[W], ws_uloc, AND transpose wu_ub -> l1_kv for GEMM 3!
      cast_and_split_wu_vf<T>(wu_ub, out_wq, out_loc, DK_PAD, DV_PAD);
      asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID2);
      asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID2);
      int64_t ws_idx = (int64_t)bh * Nc + ch;
      asc_copy_ub2gm_align((__gm__ uint8_t*)(ws_wq + ws_idx * (2 * WY_C * DK_PAD) + sid * WY_H * DK_PAD),
                           (__ubuf__ uint8_t*)out_wq, 1, (uint32_t)(WY_H * DK_PAD * sizeof(T)),
                           asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                           (uint32_t)(WY_H * DK_PAD * sizeof(T)), (uint32_t)(WY_H * DK_PAD * sizeof(T)));
      asc_copy_ub2gm_align((__gm__ uint8_t*)(ws_uloc + ws_idx * (WY_C * DV_PAD) + sid * WY_H * DV_PAD),
                           (__ubuf__ uint8_t*)out_loc, 1, (uint32_t)(WY_H * DV_PAD * 4),
                           asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                           (uint32_t)(WY_H * DV_PAD * 4), (uint32_t)(WY_H * DV_PAD * 4));

      asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID2);
      asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID2);
      transpose_16xN_to_Nx16(wu_ub, tr_tmp, N_TOT);
      asc_sync_notify(PIPE_MTE3, PIPE_V, EVENT_ID2);
      asc_sync_wait(PIPE_MTE3, PIPE_V, EVENT_ID2);
      scatter_nz_vf<N_TOT, 16, N_TOT, T>(tr_tmp, nz_tmp);
      asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID2);
      asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID2);
      asc_copy_ub2l1((__cbuf__ void*)(l1_kv + sid * N_TOT * 16), (__ubuf__ void*)nz_tmp,
                     (uint16_t)(16 / C0), N_TOT * C0 * sizeof(T) / 32, 0, 0);
      asc_sync_intra_arrive(PIPE_MTE3, 8);

      // Wait GEMM 3 result in wu_ub (now sy_ub)
      asc_sync_intra_wait(PIPE_V, 9);
      compute_qeff_yloc_vf<T>(q_tilde, wu_ub, out_wq, out_loc, DK_PAD, DV_PAD);
      cast_khat_vf<T>(khat_f, (__ubuf__ T*)wu_ub, DK_PAD);
      asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID3);
      asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID3);
      asc_copy_ub2gm_align((__gm__ uint8_t*)(ws_wq + ws_idx * (2 * WY_C * DK_PAD) + (WY_C + sid * WY_H) * DK_PAD),
                           (__ubuf__ uint8_t*)out_wq, 1, (uint32_t)(WY_H * DK_PAD * sizeof(T)),
                           asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                           (uint32_t)(WY_H * DK_PAD * sizeof(T)), (uint32_t)(WY_H * DK_PAD * sizeof(T)));
      asc_copy_ub2gm_align((__gm__ uint8_t*)(ws_yloc + ws_idx * (WY_C * DV_PAD) + sid * WY_H * DV_PAD),
                           (__ubuf__ uint8_t*)out_loc, 1, (uint32_t)(WY_H * DV_PAD * 4),
                           asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                           (uint32_t)(WY_H * DV_PAD * 4), (uint32_t)(WY_H * DV_PAD * 4));
      asc_copy_ub2gm_align((__gm__ uint8_t*)(ws_khat + ws_idx * (WY_C * DK_PAD) + sid * WY_H * DK_PAD),
                           (__ubuf__ uint8_t*)wu_ub, 1, (uint32_t)(WY_H * DK_PAD * sizeof(T)),
                           asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                           (uint32_t)(WY_H * DK_PAD * sizeof(T)), (uint32_t)(WY_H * DK_PAD * sizeof(T)));
      if (sid == 0) {
        asc_copy_ub2gm_align((__gm__ uint8_t*)(ws_gamma + ws_idx * DK_PAD),
                             (__ubuf__ uint8_t*)gamma_end, 1, (uint32_t)(DK_PAD * 4),
                             asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint32_t)(DK_PAD * 4), (uint32_t)(DK_PAD * 4));
      }

      asc_sync_notify(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      asc_sync_wait(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID0);
      asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID0);
      asc_sync_notify(PIPE_MTE3, PIPE_V, EVENT_ID3);
      asc_sync_wait(PIPE_MTE3, PIPE_V, EVENT_ID3);
    }
  }
  asc_sync();
}

#define DEF_STAGE1_MIX(SUFFIX, DK_PAD, DV_PAD, T) \
extern "C" __global__ __aicore__ __mix__(1, 2) void kda_wy_stage1_mix_##SUFFIX( \
    __gm__ T* q, __gm__ T* k, __gm__ T* v, __gm__ float* g, __gm__ float* beta, \
    __gm__ float* ws_wq, __gm__ float* ws_khat, __gm__ float* ws_uloc, __gm__ float* ws_yloc, __gm__ float* ws_gamma, \
    int32_t B, int32_t T_len, int32_t H, int32_t Dk, int32_t Dv, int32_t use_l2norm, float scale_val) { \
  kda_wy_stage1_mix_impl<DK_PAD, DV_PAD, T>(q, k, v, g, beta, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, B, T_len, H, Dk, Dv, use_l2norm, scale_val); \
}

DEF_STAGE1_MIX(f16_64_64, 64, 64, half)
DEF_STAGE1_MIX(f16_64_128, 64, 128, half)
DEF_STAGE1_MIX(f16_128_128, 128, 128, half)
DEF_STAGE1_MIX(bf16_64_64, 64, 64, bfloat16_t)
DEF_STAGE1_MIX(bf16_64_128, 64, 128, bfloat16_t)
DEF_STAGE1_MIX(bf16_128_128, 128, 128, bfloat16_t)
DEF_STAGE1_MIX(f32_64_64, 64, 64, float)
DEF_STAGE1_MIX(f32_64_128, 64, 128, float)
DEF_STAGE1_MIX(f32_128_128, 128, 128, float)

// ============================================================================
// Kernel 3: Chunked WY Stage 2 (__mix__(1, 2), WY_C = 32, DV_TILE = 32/64/128)
// ============================================================================
template <int DV_HALF, int DK_PAD, typename T>
__simd_vf__ inline void scatter_ht_vf(__ubuf__ float* src_f32, __ubuf__ T* dst_nz) {
  ascend_nd2nz_scatter_callee<DV_HALF, DK_PAD, DV_HALF + 1, float, T>(src_f32, dst_nz);
}

template <int DV_HALF, typename T>
__simd_vf__ inline void compute_u_and_cast_vf(
    __ubuf__ float* u_loc, __ubuf__ float* p_wy, __ubuf__ T* u_out,
    int32_t dv_pad_off, int32_t dv_pad) {
  auto m32 = asc_create_mask_b32(PAT_ALL);
  auto m16 = asc_create_mask_b16(PAT_ALL);
  constexpr int32_t DV_TILE = DV_HALF * 2;
  constexpr int32_t CHUNKS = DV_TILE / 64;
  if constexpr (DV_TILE == 32) {
    auto mv32 = make_mask_b32(32);
    auto mv16 = make_mask_b16(64);
    for (int32_t t = 0; t < WY_C; ++t) {
      auto ul = V::vlds_norm(u_loc + t * dv_pad + dv_pad_off, 0);
      auto pw = V::vlds_norm(p_wy + t * 32, 0);
      store_f32_vec(u_out, t * 32, V::vsub(ul, pw, mv32, MODE_ZEROING), mv32, mv16);
    }
  } else {
    for (int32_t t = 0; t < WY_C; ++t) {
      for (int32_t c = 0; c < CHUNKS; ++c) {
        auto ul = V::vlds_norm(u_loc + t * dv_pad + dv_pad_off + c * 64, 0);
        auto pw = V::vlds_norm(p_wy + t * DV_TILE + c * 64, 0);
        store_f32_vec(u_out, t * DV_TILE + c * 64, V::vsub(ul, pw, m32, MODE_ZEROING), m32, m16);
      }
    }
  }
}

template <int DV_HALF, typename T>
__simd_vf__ inline void compute_y_and_cast_vf(
    __ubuf__ float* y_loc, __ubuf__ float* q_h, __ubuf__ T* y_out,
    int32_t c_len, int32_t dv_cur, int32_t dv_pad_off, int32_t dv_pad) {
  auto m32 = asc_create_mask_b32(PAT_ALL);
  auto m16 = asc_create_mask_b16(PAT_ALL);
  constexpr int32_t DV_TILE = DV_HALF * 2;
  constexpr int32_t CHUNKS = (DV_TILE + 63) / 64;
  for (int32_t t = 0; t < c_len; ++t) {
    for (int32_t c = 0; c < CHUNKS; ++c) {
      int32_t rem = dv_cur - c * 64;
      if (rem > 0) {
        int32_t w = (rem < 64) ? rem : 64;
        auto mv32 = (w == 64) ? m32 : make_mask_b32((uint32_t)w);
        auto mv16 = (w == 64) ? m16 : make_mask_b16((uint32_t)(w * 2));
        auto yl = V::vlds_norm(y_loc + t * dv_pad + dv_pad_off + c * 64, 0);
        auto qh = V::vlds_norm(q_h + t * DV_TILE + c * 64, 0);
        store_f32_vec(y_out, t * DV_TILE + c * 64, V::vadd(yl, qh, mv32, MODE_ZEROING), mv32, mv16);
      }
    }
  }
}

template <int DV_HALF>
__simd_vf__ inline void update_ht_vf(
    __ubuf__ float* ht, __ubuf__ float* dht, __ubuf__ float* gamma_end, int32_t dk_pad) {
  auto m32 = asc_create_mask_b32(PAT_ALL);
  int32_t chunks = dk_pad / 64;
  for (int32_t r = 0; r < DV_HALF; ++r) {
    for (int32_t c = 0; c < chunks; ++c) {
      int32_t off = r * dk_pad + c * 64;
      auto ge = V::vlds_norm(gamma_end + c * 64, 0);
      auto h_old = V::vlds_norm(ht + off, 0);
      auto dh = V::vlds_norm(dht + off, 0);
      auto h_new = V::vadd(V::vmul(h_old, ge, m32, MODE_ZEROING), dh, m32, MODE_ZEROING);
      V::vsts_norm(h_new, ht + off, 0, m32);
    }
  }
  asc_mem_bar(VST_VLD);
}

__simd_vf__ inline void zero_ub_vf(__ubuf__ float* ptr, int32_t floats) {
  auto m32 = asc_create_mask_b32(PAT_ALL);
  auto z = V::vdup<float>(0.0f, m32, MODE_ZEROING);
  for (int32_t i = 0; i < floats; i += 64) {
    V::vsts_norm(z, ptr + i, 0, m32);
  }
  asc_mem_bar(VST_VLD);
}

template <int DV_TILE, int DK_PAD, typename T_io>
__aicore__ inline void kda_wy_stage2_impl(
    __gm__ float* init_state, __gm__ T_io* y, __gm__ float* final_state,
    __gm__ float* ws_wq, __gm__ float* ws_khat, __gm__ float* ws_uloc, __gm__ float* ws_yloc,
    __gm__ float* ws_gamma, __gm__ float* ws_u_scratch,
    int32_t B, int32_t T_len, int32_t H, int32_t Dk, int32_t Dv) {
  asc_init();
  using T = float;
  constexpr int32_t DV_HALF = DV_TILE / 2;
  __cbuf__ uint8_t* l1_base = (__cbuf__ uint8_t*)0;
  __cc__ float* l0c_0 = (__cc__ float*)0;
  __cc__ float* l0c_1 = (__cc__ float*)(2 * WY_C * DV_TILE); // after [64, DV_TILE] floats
  __ubuf__ uint8_t* ub = (__ubuf__ uint8_t*)0;

  // L1 layout (max for DK_PAD=128, DV_TILE=128, WY_C=32, bf16/fp16/fp32):
  // l1_wq: [64, DK_PAD] T (max 64*128*4 = 32KB) at 0
  // l1_ht: [DV_TILE, DK_PAD] T (max 128*128*2 = 32KB for b16, 64*128*4 = 32KB for f32) at 32KB
  // l1_ut: [DV_TILE, 32] T (max 128*32*4 = 16KB) at 96KB
  // l1_khatT: [DK_PAD, 32] T (max 128*32*4 = 16KB) at 112KB
  __cbuf__ T* l1_wq = (__cbuf__ T*)(l1_base + 0);
  __cbuf__ T* l1_ht = (__cbuf__ T*)(l1_base + 32768);
  __cbuf__ T* l1_ut = (__cbuf__ T*)(l1_base + 98304);
  __cbuf__ T* l1_khatT = (__cbuf__ T*)(l1_base + 114688);

  // UB layout (per AIV):
  // ht_ub: [DV_HALF, DK_PAD] float (max 64*128*4 = 32KB) at 0
  // ht_nz: [DV_HALF+1, DK_PAD] T (max 65*128*4 = 33.28KB) at 32KB
  // dht_ub: [DV_HALF, DK_PAD] float (max 32KB) at 68KB
  // gemm1_ub: [32, DV_TILE] float (max 32*128*4 = 16KB) at 100KB
  // vec_ws_ub: [32, DV_PAD_MAX=128] float (max 16KB) at 116KB
  // gamma_ub: [128] float (0.5KB) at 132KB
  // io_ub: [32, DV_TILE] T (max 16KB) at 134KB
  // state_io: [DK_PAD, DV_HALF] float (max 128*64*4 = 32KB) at 150KB
  __ubuf__ float* ht_ub = (__ubuf__ float*)(ub + 0);
  __ubuf__ T* ht_nz = (__ubuf__ T*)(ub + 32768);
  __ubuf__ float* dht_ub = (__ubuf__ float*)(ub + 69632);
  __ubuf__ float* gemm1_ub = (__ubuf__ float*)(ub + 102400);
  __ubuf__ float* vec_ws_ub = (__ubuf__ float*)(ub + 118784);
  __ubuf__ float* gamma_ub = (__ubuf__ float*)(ub + 135168);
  __ubuf__ T* io_ub = (__ubuf__ T*)(ub + 137216);
  __ubuf__ float* state_io = (__ubuf__ float*)(ub + 153600);

  int32_t Nc = (T_len + WY_C - 1) / WY_C;
  int32_t dv_tiles = (Dv + DV_TILE - 1) / DV_TILE;
  int32_t dv_pad = ((Dv + 63) / 64) * 64; // workspace dv_pad is multiple of 64!
  int32_t total_tasks = B * H * dv_tiles;
  int32_t ncores = (int32_t)block_num;
  int32_t cid = (int32_t)block_idx;
  int32_t base = total_tasks / ncores;
  int32_t rem = total_tasks % ncores;
  int32_t start = cid * base + (cid < rem ? cid : rem);
  int32_t count = base + (cid < rem ? 1 : 0);

  constexpr int32_t C0 = 32 / sizeof(T);
  constexpr int32_t BURSTS_HT = DK_PAD / C0;
  constexpr int32_t LEN_HT = (DV_HALF * C0 * sizeof(T)) / 32; // = DV_HALF
  constexpr int32_t TILE_K_HT = (std::is_same_v<T, float> && DK_PAD > 64) ? 64 : DK_PAD;
  constexpr int32_t TILE_K_U = (std::is_same_v<T, float> && WY_C > 16 && DV_TILE == 128) ? 16 : WY_C;

  if ASC_IS_AIC {
    for (int32_t lt = 0; lt < count; ++lt) {
      int32_t tid = start + lt;
      int32_t bh = tid / dv_tiles;
      for (int32_t ch = 0; ch < Nc; ++ch) {
        int64_t ws_idx = (int64_t)bh * Nc + ch;
        asc_set_gm2l1_nz_para(1, 1, (uint16_t)(2 * WY_C), 0);
        asc_copy_gm2l1_nd2nz(l1_wq, ws_wq + ws_idx * (2 * WY_C * DK_PAD),
                             (uint64_t)(DK_PAD * sizeof(T)),
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             2 * WY_C, (uint32_t)DK_PAD, 0, 0);
        asc_set_gm2l1_nz_para(1, 1, (uint16_t)DK_PAD, 0);
        asc_copy_gm2l1_dn2nz(l1_khatT, ws_khat + ws_idx * (WY_C * DK_PAD),
                             (uint64_t)(DK_PAD * sizeof(T)),
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint16_t)DK_PAD, (uint32_t)WY_C, 0, 0);
        asc_sync_notify(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        asc_sync_wait(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        asc_sync_intra_wait(PIPE_MTE1, 1);
        asc_sync_intra_wait(PIPE_MTE1, 17);
        asc_sync_intra_wait(PIPE_FIX, 0);
        asc_sync_intra_wait(PIPE_FIX, 16);

        ascend_gemm_l1<2 * WY_C, DK_PAD, DV_TILE, TILE_K_HT, true, T>(l0c_0, l1_wq, l1_ht, 1, 0, 0);
        asc_sync_notify(PIPE_M, PIPE_FIX, EVENT_ID0);
        asc_sync_wait(PIPE_M, PIPE_FIX, EVENT_ID0);
        asc_set_l0c_copy_nz_para(1, 0, 0);
        asc_copy_l0c2ub(gemm1_ub, l0c_0, (uint16_t)DV_TILE, 2 * WY_C, (uint32_t)DV_TILE, 2 * WY_C, 0,
                        asc_dual_dst_mode::DUAL_DST_SPLIT_M, asc_unit_flag_mode::DISABLE,
                        static_cast<asc_quant_mode>(0), asc_relu_pre_mode::NONE, 0, 1, 0, 0);
        asc_sync_intra_arrive(PIPE_FIX, 2);
        asc_sync_intra_arrive(PIPE_FIX, 18);

        asc_sync_intra_wait(PIPE_MTE2, 3);
        asc_set_gm2l1_nz_para(1, 1, (uint16_t)DV_TILE, 0);
        asc_copy_gm2l1_dn2nz(l1_ut, ws_u_scratch + (int64_t)cid * (WY_C * DV_TILE),
                             (uint64_t)(DV_TILE * sizeof(T)),
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint16_t)DV_TILE, (uint32_t)WY_C, 0, 0);
        asc_sync_notify(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        asc_sync_wait(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);

        asc_sync_intra_wait(PIPE_FIX, 4);
        asc_sync_intra_wait(PIPE_FIX, 20);

        ascend_gemm_l1<DV_TILE, WY_C, DK_PAD, TILE_K_U, true, T>(l0c_1, l1_ut, l1_khatT, 1, 0, 0);
        asc_sync_notify(PIPE_M, PIPE_FIX, EVENT_ID1);
        asc_sync_wait(PIPE_M, PIPE_FIX, EVENT_ID1);
        asc_set_l0c_copy_nz_para(1, 0, 0);
        asc_copy_l0c2ub(dht_ub, l0c_1, (uint16_t)DK_PAD, (uint16_t)DV_TILE, (uint32_t)DK_PAD, (uint32_t)DV_TILE, 0,
                        asc_dual_dst_mode::DUAL_DST_SPLIT_M, asc_unit_flag_mode::DISABLE,
                        static_cast<asc_quant_mode>(0), asc_relu_pre_mode::NONE, 0, 1, 0, 0);
        asc_sync_intra_arrive(PIPE_FIX, 5);
        asc_sync_intra_arrive(PIPE_FIX, 21);
        asc_sync_notify(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        asc_sync_wait(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      }
    }
  }

  if ASC_IS_AIV {
    int32_t sid = (int32_t)asc_get_sub_block_id();
    for (int32_t lt = 0; lt < count; ++lt) {
      int32_t tid = start + lt;
      int32_t bh = tid / dv_tiles;
      int32_t dv_t = tid % dv_tiles;
      int32_t b = bh / H;
      int32_t h = bh % H;
      int32_t dv0 = dv_t * DV_TILE;
      int32_t dv_cur = (Dv - dv0 < DV_TILE) ? (Dv - dv0) : DV_TILE;
      int32_t dv_sid_off = sid * DV_HALF;
      int32_t dv_sid_cur = (dv_cur - dv_sid_off > 0) ? ((dv_cur - dv_sid_off < DV_HALF) ? (dv_cur - dv_sid_off) : DV_HALF) : 0;

      zero_ub_vf(state_io, DK_PAD * DV_HALF);
      asc_sync_notify(PIPE_V, PIPE_MTE2, EVENT_ID0);
      asc_sync_wait(PIPE_V, PIPE_MTE2, EVENT_ID0);

      int64_t state_off = ((int64_t)b * H + h) * Dk * Dv + dv0 + dv_sid_off;
      if (dv_sid_cur > 0) {
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)state_io, (__gm__ uint8_t*)(init_state + state_off),
                             (uint32_t)Dk, (uint32_t)(dv_sid_cur * 4), 0, 0, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)(Dv * 4), (uint32_t)(DV_HALF * 4));
      }
      asc_sync_notify(PIPE_MTE2, PIPE_S, EVENT_ID0);
      asc_sync_wait(PIPE_MTE2, PIPE_S, EVENT_ID0);

      constexpr int32_t COL_BLKS_INIT = DV_HALF / 8;
      constexpr int32_t SRC_STRIDE_INIT = 16 * DV_HALF / 8;
      for (int32_t col_blk = 0; col_blk < COL_BLKS_INIT; ++col_blk) {
        __ubuf__ float* sp[16];
        __ubuf__ float* dp[16];
        for (int32_t i = 0; i < 16; ++i) sp[i] = state_io + i * DV_HALF + col_blk * 8;
        for (int32_t r = 0; r < 8; ++r) {
          dp[2 * r + 0] = ht_ub + (col_blk * 8 + r) * DK_PAD + 0;
          dp[2 * r + 1] = ht_ub + (col_blk * 8 + r) * DK_PAD + 8;
        }
        asc_set_va_reg(VA0, sp);
        asc_set_va_reg(VA1, sp + 8);
        asc_set_va_reg(VA2, dp);
        asc_set_va_reg(VA3, dp + 8);
        asc_sync_notify(PIPE_S, PIPE_V, EVENT_ID0);
        asc_sync_wait(PIPE_S, PIPE_V, EVENT_ID0);
        asc_transto5hd_b32(VA2, VA0, (uint8_t)(DK_PAD / 16), 2, (uint16_t)SRC_STRIDE_INIT);
        asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID0);
        asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID0);
      }

      for (int32_t ch = 0; ch < Nc; ++ch) {
        int32_t t0 = ch * WY_C;
        int32_t c_len = (T_len - t0 < WY_C) ? (T_len - t0) : WY_C;
        int64_t ws_idx = (int64_t)bh * Nc + ch;

        scatter_ht_vf<DV_HALF, DK_PAD, T>(ht_ub, ht_nz);
        asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
        asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
        asc_copy_ub2l1((__cbuf__ void*)((__cbuf__ uint8_t*)l1_ht + sid * DV_HALF * 32),
                       (__ubuf__ void*)ht_nz, (uint16_t)BURSTS_HT, (uint16_t)LEN_HT, 1, (uint16_t)LEN_HT);
        asc_sync_intra_arrive(PIPE_MTE3, 1);

        if (sid == 0) {
          asc_copy_gm2ub_align((__ubuf__ uint8_t*)vec_ws_ub,
                               (__gm__ uint8_t*)(ws_uloc + ws_idx * (WY_C * dv_pad)),
                               1, (uint32_t)(WY_C * dv_pad * 4), 0, 0, false,
                               asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                               (uint32_t)(WY_C * dv_pad * 4), (uint32_t)(WY_C * dv_pad * 4));
        } else {
          asc_copy_gm2ub_align((__ubuf__ uint8_t*)vec_ws_ub,
                               (__gm__ uint8_t*)(ws_yloc + ws_idx * (WY_C * dv_pad)),
                               1, (uint32_t)(WY_C * dv_pad * 4), 0, 0, false,
                               asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                               (uint32_t)(WY_C * dv_pad * 4), (uint32_t)(WY_C * dv_pad * 4));
        }
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)gamma_ub,
                             (__gm__ uint8_t*)(ws_gamma + ws_idx * DK_PAD),
                             1, (uint32_t)(DK_PAD * 4), 0, 0, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint32_t)(DK_PAD * 4), (uint32_t)(DK_PAD * 4));
        asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID1);
        asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID1);

        asc_sync_intra_arrive(PIPE_V, 0);
        asc_sync_intra_wait(PIPE_V, 2);

        if (sid == 0) {
          compute_u_and_cast_vf<DV_HALF, T>(vec_ws_ub, gemm1_ub, io_ub, dv0, dv_pad);
          asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID1);
          asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID1);
          asc_copy_ub2gm_align((__gm__ uint8_t*)(ws_u_scratch + (int64_t)cid * (WY_C * DV_TILE)),
                               (__ubuf__ uint8_t*)io_ub, 1, (uint32_t)(WY_C * DV_TILE * sizeof(T)),
                               asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                               (uint32_t)(WY_C * DV_TILE * sizeof(T)), (uint32_t)(WY_C * DV_TILE * sizeof(T)));
          asc_sync_intra_arrive(PIPE_MTE3, 3);
        } else {
          compute_y_and_cast_vf<DV_HALF, T_io>(vec_ws_ub, gemm1_ub, (__ubuf__ T_io*)io_ub, c_len, dv_cur, dv0, dv_pad);
          asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID1);
          asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID1);
          int64_t y_off = (((int64_t)b * T_len + t0) * H + h) * Dv + dv0;
          asc_copy_ub2gm_align((__gm__ uint8_t*)(y + y_off), (__ubuf__ uint8_t*)io_ub,
                               (uint32_t)c_len, (uint32_t)(dv_cur * sizeof(T_io)),
                               asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                               (uint64_t)(H * Dv * sizeof(T_io)), (uint32_t)(DV_TILE * sizeof(T_io)));
        }

        asc_sync_intra_arrive(PIPE_V, 4);
        asc_sync_intra_wait(PIPE_V, 5);

        update_ht_vf<DV_HALF>(ht_ub, dht_ub, gamma_ub, DK_PAD);
        asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID2);
        asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID2);
        asc_sync_notify(PIPE_V, PIPE_MTE2, EVENT_ID2);
        asc_sync_wait(PIPE_V, PIPE_MTE2, EVENT_ID2);
        asc_sync_notify(PIPE_MTE3, PIPE_V, EVENT_ID2);
        asc_sync_wait(PIPE_MTE3, PIPE_V, EVENT_ID2);
      }

      asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID2);
      asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID2);
      constexpr int32_t COL_BLKS_FIN = DV_HALF / 16;
      constexpr int32_t DST_STRIDE_FIN = 8 * DV_HALF / 8;
      for (int32_t col_blk = 0; col_blk < COL_BLKS_FIN; ++col_blk) {
        __ubuf__ float* sp[16];
        __ubuf__ float* dp[16];
        for (int32_t i = 0; i < 16; ++i) sp[i] = ht_ub + (col_blk * 16 + i) * DK_PAD;
        for (int32_t r = 0; r < 8; ++r) {
          dp[2 * r + 0] = state_io + r * DV_HALF + col_blk * 16 + 0;
          dp[2 * r + 1] = state_io + r * DV_HALF + col_blk * 16 + 8;
        }
        asc_set_va_reg(VA0, sp);
        asc_set_va_reg(VA1, sp + 8);
        asc_set_va_reg(VA2, dp);
        asc_set_va_reg(VA3, dp + 8);
        asc_sync_notify(PIPE_S, PIPE_V, EVENT_ID2);
        asc_sync_wait(PIPE_S, PIPE_V, EVENT_ID2);
        asc_transto5hd_b32(VA2, VA0, (uint8_t)(DK_PAD / 8), (uint16_t)DST_STRIDE_FIN, 1);
        asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID2);
        asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID2);
      }
      asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID2);
      asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID2);
      if (dv_sid_cur > 0) {
        asc_copy_ub2gm_align((__gm__ uint8_t*)(final_state + state_off), (__ubuf__ uint8_t*)state_io,
                             (uint32_t)Dk, (uint32_t)(dv_sid_cur * 4),
                             asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)(Dv * 4), (uint32_t)(DV_HALF * 4));
        asc_sync_notify(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      }
      asc_sync_notify(PIPE_MTE3, PIPE_V, EVENT_ID3);
      asc_sync_wait(PIPE_MTE3, PIPE_V, EVENT_ID3);
    }
  }
  asc_sync();
}

#define DEF_STAGE2(SUFFIX, DV_TILE, DK_PAD, T) \
extern "C" __global__ __aicore__ __mix__(1, 2) void kda_wy_stage2_##SUFFIX( \
    __gm__ float* init_state, __gm__ T* y, __gm__ float* final_state, \
    __gm__ float* ws_wq, __gm__ float* ws_khat, __gm__ float* ws_uloc, __gm__ float* ws_yloc, \
    __gm__ float* ws_gamma, __gm__ float* ws_u_scratch, \
    int32_t B, int32_t T_len, int32_t H, int32_t Dk, int32_t Dv) { \
  kda_wy_stage2_impl<DV_TILE, DK_PAD, T>(init_state, y, final_state, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, ws_u_scratch, B, T_len, H, Dk, Dv); \
}

DEF_STAGE2(f16_32_64, 32, 64, half)
DEF_STAGE2(f16_64_64, 64, 64, half)
DEF_STAGE2(f16_64_128, 64, 128, half)
DEF_STAGE2(f16_128_64, 128, 64, half)
DEF_STAGE2(f16_128_128, 128, 128, half)
DEF_STAGE2(bf16_32_64, 32, 64, bfloat16_t)
DEF_STAGE2(bf16_64_64, 64, 64, bfloat16_t)
DEF_STAGE2(bf16_64_128, 64, 128, bfloat16_t)
DEF_STAGE2(bf16_128_64, 128, 64, bfloat16_t)
DEF_STAGE2(bf16_128_128, 128, 128, bfloat16_t)
DEF_STAGE2(f32_32_64, 32, 64, float)
DEF_STAGE2(f32_64_64, 64, 64, float)
DEF_STAGE2(f32_64_128, 64, 128, float)
DEF_STAGE2(f32_128_64, 128, 64, float)
DEF_STAGE2(f32_128_128, 128, 128, float)

template <typename T>
void dispatch_stage1_stage2(
    void* stream,
    __gm__ T* q, __gm__ T* k, __gm__ T* v, __gm__ float* g, __gm__ float* beta, __gm__ float* init_state,
    __gm__ T* y, __gm__ float* final_state,
    __gm__ float* ws_wq, __gm__ float* ws_khat, __gm__ float* ws_uloc, __gm__ float* ws_yloc, __gm__ float* ws_gamma, __gm__ float* ws_u_scratch,
    int32_t B, int32_t T_len, int32_t H, int32_t Dk, int32_t Dv, int32_t use_l2norm, float scale_val) {
  int32_t Nc = (T_len + WY_C - 1) / WY_C;
  int32_t s1_tasks = B * H * Nc;
  uint32_t g1 = (uint32_t)(s1_tasks < 28 ? s1_tasks : 28);
  int32_t dk_pad = ((Dk + 63) / 64) * 64;
  int32_t dv_pad = ((Dv + 63) / 64) * 64;

  if constexpr (std::is_same_v<T, half>) {
    if (dk_pad == 64 && dv_pad == 64)
      kda_wy_stage1_mix_f16_64_64<<<g1, nullptr, stream>>>(q, k, v, g, beta, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, B, T_len, H, Dk, Dv, use_l2norm, scale_val);
    else if (dk_pad == 64)
      kda_wy_stage1_mix_f16_64_128<<<g1, nullptr, stream>>>(q, k, v, g, beta, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, B, T_len, H, Dk, Dv, use_l2norm, scale_val);
    else
      kda_wy_stage1_mix_f16_128_128<<<g1, nullptr, stream>>>(q, k, v, g, beta, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, B, T_len, H, Dk, Dv, use_l2norm, scale_val);
  } else if constexpr (std::is_same_v<T, bfloat16_t>) {
    if (dk_pad == 64 && dv_pad == 64)
      kda_wy_stage1_mix_bf16_64_64<<<g1, nullptr, stream>>>(q, k, v, g, beta, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, B, T_len, H, Dk, Dv, use_l2norm, scale_val);
    else if (dk_pad == 64)
      kda_wy_stage1_mix_bf16_64_128<<<g1, nullptr, stream>>>(q, k, v, g, beta, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, B, T_len, H, Dk, Dv, use_l2norm, scale_val);
    else
      kda_wy_stage1_mix_bf16_128_128<<<g1, nullptr, stream>>>(q, k, v, g, beta, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, B, T_len, H, Dk, Dv, use_l2norm, scale_val);
  } else {
    if (dk_pad == 64 && dv_pad == 64)
      kda_wy_stage1_mix_f32_64_64<<<g1, nullptr, stream>>>(q, k, v, g, beta, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, B, T_len, H, Dk, Dv, use_l2norm, scale_val);
    else if (dk_pad == 64)
      kda_wy_stage1_mix_f32_64_128<<<g1, nullptr, stream>>>(q, k, v, g, beta, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, B, T_len, H, Dk, Dv, use_l2norm, scale_val);
    else
      kda_wy_stage1_mix_f32_128_128<<<g1, nullptr, stream>>>(q, k, v, g, beta, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, B, T_len, H, Dk, Dv, use_l2norm, scale_val);
  }

  // Choose Stage 2 dv_tile for optimal 28-core utilization
  int32_t dv_tile = 64;
  if (Dv <= 64 && B * H * 2 <= 28) dv_tile = 32;
  else if (Dv > 64 && Dv <= 128 && B * H <= 28 && B * H * 2 > 28) dv_tile = 128;
  // Wait: also check if dv_tile=128 is faster for B*H=16 vs 64 (16 tasks vs 32 -> 1 wave vs 2 waves!)
  int32_t s2_tasks = B * H * ((Dv + dv_tile - 1) / dv_tile);
  uint32_t g2 = (uint32_t)(s2_tasks < 28 ? s2_tasks : 28);

  if constexpr (std::is_same_v<T, half>) {
    if (dv_tile == 32)
      kda_wy_stage2_f16_32_64<<<g2, nullptr, stream>>>(init_state, y, final_state, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, ws_u_scratch, B, T_len, H, Dk, Dv);
    else if (dv_tile == 128 && dk_pad == 64)
      kda_wy_stage2_f16_128_64<<<g2, nullptr, stream>>>(init_state, y, final_state, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, ws_u_scratch, B, T_len, H, Dk, Dv);
    else if (dv_tile == 128)
      kda_wy_stage2_f16_128_128<<<g2, nullptr, stream>>>(init_state, y, final_state, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, ws_u_scratch, B, T_len, H, Dk, Dv);
    else if (dk_pad == 64)
      kda_wy_stage2_f16_64_64<<<g2, nullptr, stream>>>(init_state, y, final_state, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, ws_u_scratch, B, T_len, H, Dk, Dv);
    else
      kda_wy_stage2_f16_64_128<<<g2, nullptr, stream>>>(init_state, y, final_state, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, ws_u_scratch, B, T_len, H, Dk, Dv);
  } else if constexpr (std::is_same_v<T, bfloat16_t>) {
    if (dv_tile == 32)
      kda_wy_stage2_bf16_32_64<<<g2, nullptr, stream>>>(init_state, y, final_state, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, ws_u_scratch, B, T_len, H, Dk, Dv);
    else if (dv_tile == 128 && dk_pad == 64)
      kda_wy_stage2_bf16_128_64<<<g2, nullptr, stream>>>(init_state, y, final_state, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, ws_u_scratch, B, T_len, H, Dk, Dv);
    else if (dv_tile == 128)
      kda_wy_stage2_bf16_128_128<<<g2, nullptr, stream>>>(init_state, y, final_state, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, ws_u_scratch, B, T_len, H, Dk, Dv);
    else if (dk_pad == 64)
      kda_wy_stage2_bf16_64_64<<<g2, nullptr, stream>>>(init_state, y, final_state, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, ws_u_scratch, B, T_len, H, Dk, Dv);
    else
      kda_wy_stage2_bf16_64_128<<<g2, nullptr, stream>>>(init_state, y, final_state, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, ws_u_scratch, B, T_len, H, Dk, Dv);
  } else {
    if (dv_tile == 32)
      kda_wy_stage2_f32_32_64<<<g2, nullptr, stream>>>(init_state, y, final_state, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, ws_u_scratch, B, T_len, H, Dk, Dv);
    else if (dv_tile == 128 && dk_pad == 64)
      kda_wy_stage2_f32_128_64<<<g2, nullptr, stream>>>(init_state, y, final_state, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, ws_u_scratch, B, T_len, H, Dk, Dv);
    else if (dv_tile == 128)
      kda_wy_stage2_f32_128_128<<<g2, nullptr, stream>>>(init_state, y, final_state, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, ws_u_scratch, B, T_len, H, Dk, Dv);
    else if (dk_pad == 64)
      kda_wy_stage2_f32_64_64<<<g2, nullptr, stream>>>(init_state, y, final_state, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, ws_u_scratch, B, T_len, H, Dk, Dv);
    else
      kda_wy_stage2_f32_64_128<<<g2, nullptr, stream>>>(init_state, y, final_state, ws_wq, ws_khat, ws_uloc, ws_yloc, ws_gamma, ws_u_scratch, B, T_len, H, Dk, Dv);
  }
}

// ============================================================================
// Kernel 4: DV_TILE=64 UB-Resident Column-Tiled Kernel (Dk<=256, BT=16 DBuf)
// ============================================================================
template <typename T>
__simd_vf__ inline void kda_dv64_block_vf(
    __ubuf__ float* state_ub,
    __ubuf__ T* q_in,
    __ubuf__ T* k_in,
    __ubuf__ T* v_in,
    __ubuf__ float* g_in,
    __ubuf__ float* beta_in,
    __ubuf__ float* qf_ub,
    __ubuf__ float* kf_ub,
    __ubuf__ T* out_ub,
    int32_t t_cur, int32_t Dk, int32_t dk_pad, int32_t dv_cur,
    int32_t use_l2norm, float scale_val) {
  auto m32 = asc_create_mask_b32(PAT_ALL);
  auto mv32 = make_mask_b32((uint32_t)dv_cur);
  auto mv16 = make_mask_b16((uint32_t)(dv_cur * 2));
  auto v_zero = V::vdup<float>(0.0f, m32, MODE_ZEROING);
  auto v_one = V::vdup<float>(1.0f, m32, MODE_ZEROING);
  auto v_eps = V::vdup<float>(1.0e-6f, m32, MODE_ZEROING);

  int32_t dk_chunks = (Dk + 63) / 64;
  int32_t dk_tail = Dk - (dk_chunks - 1) * 64;

  for (int32_t t = 0; t < t_cur; ++t) {
    if (dk_chunks == 1) {
      auto mc = (Dk == 64) ? m32 : make_mask_b32((uint32_t)Dk);
      auto qv = load_f32_vec(q_in, t * dk_pad);
      auto kv = load_f32_vec(k_in, t * dk_pad);
      auto gv = V::vlds_norm(g_in + t * dk_pad, 0);
      auto ev = V::vexp(gv, mc, MODE_ZEROING);
      if (use_l2norm) {
        auto q_sum = V::vdupv(V::vcadd(V::vmul(qv, qv, mc, MODE_ZEROING), mc, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
        auto k_sum = V::vdupv(V::vcadd(V::vmul(kv, kv, mc, MODE_ZEROING), mc, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
        auto q_den = V::vsqrt(V::vadd(q_sum, v_eps, m32, MODE_ZEROING), m32, MODE_ZEROING);
        auto k_den = V::vsqrt(V::vadd(k_sum, v_eps, m32, MODE_ZEROING), m32, MODE_ZEROING);
        auto q_sc = V::vmuls(V::vdiv_0ulp_ftz_true(v_one, q_den, m32, MODE_ZEROING), scale_val, m32, MODE_ZEROING);
        auto k_sc = V::vdiv_0ulp_ftz_true(v_one, k_den, m32, MODE_ZEROING);
        qv = V::vmul(qv, q_sc, mc, MODE_ZEROING);
        kv = V::vmul(kv, k_sc, mc, MODE_ZEROING);
      } else {
        qv = V::vmuls(qv, scale_val, mc, MODE_ZEROING);
      }
      V::vsts_norm(qv, qf_ub + t * dk_pad, 0, mc);
      V::vsts_norm(kv, kf_ub + t * dk_pad, 0, mc);
      V::vsts_norm(ev, g_in + t * dk_pad, 0, mc);
    } else {
      auto q_sq = v_zero;
      auto k_sq = v_zero;
      for (int32_t c = 0; c < dk_chunks; ++c) {
        int32_t w = (c == dk_chunks - 1) ? dk_tail : 64;
        auto mc = (w == 64) ? m32 : make_mask_b32((uint32_t)w);
        int32_t off = t * dk_pad + c * 64;
        auto qv = load_f32_vec(q_in, off);
        auto kv = load_f32_vec(k_in, off);
        auto ev = V::vexp(V::vlds_norm(g_in + off, 0), mc, MODE_ZEROING);
        if (use_l2norm) {
          q_sq = V::vadd(q_sq, V::vmul(qv, qv, mc, MODE_ZEROING), m32, MODE_ZEROING);
          k_sq = V::vadd(k_sq, V::vmul(kv, kv, mc, MODE_ZEROING), m32, MODE_ZEROING);
        } else {
          qv = V::vmuls(qv, scale_val, mc, MODE_ZEROING);
        }
        V::vsts_norm(qv, qf_ub + off, 0, mc);
        V::vsts_norm(kv, kf_ub + off, 0, mc);
        V::vsts_norm(ev, g_in + off, 0, mc);
      }
      if (use_l2norm) {
        auto q_sum = V::vdupv(V::vcadd(q_sq, m32, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
        auto k_sum = V::vdupv(V::vcadd(k_sq, m32, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
        auto q_sc = V::vmuls(V::vdiv_0ulp_ftz_true(v_one, V::vsqrt(V::vadd(q_sum, v_eps, m32, MODE_ZEROING), m32, MODE_ZEROING), m32, MODE_ZEROING), scale_val, m32, MODE_ZEROING);
        auto k_sc = V::vdiv_0ulp_ftz_true(v_one, V::vsqrt(V::vadd(k_sum, v_eps, m32, MODE_ZEROING), m32, MODE_ZEROING), m32, MODE_ZEROING);
        V::vsts_1st(q_sc, beta_in + t * 8 + 1, 0, m32);
        V::vsts_1st(k_sc, beta_in + t * 8 + 2, 0, m32);
      }
    }
  }
  asc_mem_bar(VST_VLD);

  if (dk_chunks > 1 && use_l2norm) {
    for (int32_t t = 0; t < t_cur; ++t) {
      auto q_sc = V::vlds_brc_elem(beta_in + t * 8 + 1, 0);
      auto k_sc = V::vlds_brc_elem(beta_in + t * 8 + 2, 0);
      for (int32_t c = 0; c < dk_chunks; ++c) {
        int32_t w = (c == dk_chunks - 1) ? dk_tail : 64;
        auto mc = (w == 64) ? m32 : make_mask_b32((uint32_t)w);
        int32_t off = t * dk_pad + c * 64;
        V::vsts_norm(V::vmul(V::vlds_norm(qf_ub + off, 0), q_sc, mc, MODE_ZEROING), qf_ub + off, 0, mc);
        V::vsts_norm(V::vmul(V::vlds_norm(kf_ub + off, 0), k_sc, mc, MODE_ZEROING), kf_ub + off, 0, mc);
      }
    }
    asc_mem_bar(VST_VLD);
  }

  for (int32_t t = 0; t < t_cur; ++t) {
    auto k_dot_h = v_zero;
    for (int32_t d = 0; d < Dk; ++d) {
      auto e_s = V::vlds_brc_elem(g_in + t * dk_pad + d, 0);
      auto k_s = V::vlds_brc_elem(kf_ub + t * dk_pad + d, 0);
      auto h_row = V::vlds_norm(state_ub + d * 64, 0);
      auto h_dec = V::vmul(h_row, e_s, mv32, MODE_ZEROING);
      V::vsts_norm(h_dec, state_ub + d * 64, 0, mv32);
      k_dot_h = V::vadd(k_dot_h, V::vmul(h_dec, k_s, mv32, MODE_ZEROING), mv32, MODE_ZEROING);
    }
    asc_mem_bar(VST_VLD);
    auto v_t = load_f32_vec(v_in, t * 64);
    auto b_t = V::vlds_brc_elem(beta_in + t * 8, 0);
    auto delta = V::vmul(V::vsub(v_t, k_dot_h, mv32, MODE_ZEROING), b_t, mv32, MODE_ZEROING);
    auto out_acc = v_zero;
    for (int32_t d = 0; d < Dk; ++d) {
      auto k_s = V::vlds_brc_elem(kf_ub + t * dk_pad + d, 0);
      auto q_s = V::vlds_brc_elem(qf_ub + t * dk_pad + d, 0);
      auto h_dec = V::vlds_norm(state_ub + d * 64, 0);
      auto h_new = V::vadd(h_dec, V::vmul(delta, k_s, mv32, MODE_ZEROING), mv32, MODE_ZEROING);
      V::vsts_norm(h_new, state_ub + d * 64, 0, mv32);
      out_acc = V::vadd(out_acc, V::vmul(h_new, q_s, mv32, MODE_ZEROING), mv32, MODE_ZEROING);
    }
    if (t + 1 < t_cur) asc_mem_bar(VST_VLD);
    store_f32_vec(out_ub, t * 64, out_acc, mv32, mv16);
  }
  asc_mem_bar(VST_VLD);
}

template <typename T>
__aicore__ inline void kda_dv64_issue_tb(
    __gm__ T* q, __gm__ T* k, __gm__ T* v, __gm__ float* g, __gm__ float* beta,
    __ubuf__ T* q_dst, __ubuf__ T* k_dst, __ubuf__ T* v_dst, __ubuf__ float* g_dst, __ubuf__ float* beta_dst,
    int32_t b, int32_t h, int32_t T_len, int32_t H, int32_t Dk, int32_t Dv, int32_t dk_pad,
    int32_t dv0, int32_t dv_cur, int32_t t0, int32_t t_cur) {
  int64_t qk_off = (((int64_t)b * T_len + t0) * H + h) * Dk;
  int64_t v_off = (((int64_t)b * T_len + t0) * H + h) * Dv + dv0;
  int64_t b_off = ((int64_t)b * T_len + t0) * H + h;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)q_dst, (__gm__ uint8_t*)(q + qk_off),
                       (uint32_t)t_cur, (uint32_t)(Dk * sizeof(T)), 0, 0, false,
                       asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                       (uint64_t)(H * Dk * sizeof(T)), (uint32_t)(dk_pad * sizeof(T)));
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)k_dst, (__gm__ uint8_t*)(k + qk_off),
                       (uint32_t)t_cur, (uint32_t)(Dk * sizeof(T)), 0, 0, false,
                       asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                       (uint64_t)(H * Dk * sizeof(T)), (uint32_t)(dk_pad * sizeof(T)));
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)v_dst, (__gm__ uint8_t*)(v + v_off),
                       (uint32_t)t_cur, (uint32_t)(dv_cur * sizeof(T)), 0, 0, false,
                       asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                       (uint64_t)(H * Dv * sizeof(T)), (uint32_t)(64 * sizeof(T)));
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)g_dst, (__gm__ uint8_t*)(g + qk_off),
                       (uint32_t)t_cur, (uint32_t)(Dk * 4), 0, 0, false,
                       asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                       (uint64_t)(H * Dk * 4), (uint32_t)(dk_pad * 4));
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)beta_dst, (__gm__ uint8_t*)(beta + b_off),
                       (uint32_t)t_cur, (uint32_t)4, 0, 0, false,
                       asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                       (uint64_t)(H * 4), (uint32_t)32);
}

template <typename T>
__aicore__ inline void kda_dv64_ub_impl(
    __gm__ T* q, __gm__ T* k, __gm__ T* v, __gm__ float* g, __gm__ float* beta, __gm__ float* init_state,
    __gm__ T* y, __gm__ float* final_state,
    int32_t B, int32_t T_len, int32_t H, int32_t Dk, int32_t Dv,
    int32_t use_l2norm, float scale_val) {
  asc_init();
  __ubuf__ uint8_t* ub = (__ubuf__ uint8_t*)0;
  __ubuf__ float* state_ub = (__ubuf__ float*)(ub + 0);

  __ubuf__ T* q_in0 = (__ubuf__ T*)(ub + 65536);
  __ubuf__ T* k_in0 = (__ubuf__ T*)(ub + 81920);
  __ubuf__ T* v_in0 = (__ubuf__ T*)(ub + 98304);
  __ubuf__ float* g_in0 = (__ubuf__ float*)(ub + 102400);
  __ubuf__ float* beta_in0 = (__ubuf__ float*)(ub + 118784);

  __ubuf__ T* q_in1 = (__ubuf__ T*)(ub + 119296);
  __ubuf__ T* k_in1 = (__ubuf__ T*)(ub + 135680);
  __ubuf__ T* v_in1 = (__ubuf__ T*)(ub + 152064);
  __ubuf__ float* g_in1 = (__ubuf__ float*)(ub + 156160);
  __ubuf__ float* beta_in1 = (__ubuf__ float*)(ub + 172544);

  __ubuf__ T* out_ub0 = (__ubuf__ T*)(ub + 173056);
  __ubuf__ T* out_ub1 = (__ubuf__ T*)(ub + 177152);
  __ubuf__ float* qf_ub = (__ubuf__ float*)(ub + 181248);
  __ubuf__ float* kf_ub = (__ubuf__ float*)(ub + 197632);

  constexpr int32_t BT = 16;
  int32_t dk_pad = ((Dk + 63) / 64) * 64;
  int32_t dv_tiles = (Dv + 63) / 64;
  int32_t total_tasks = B * H * dv_tiles;
  int32_t ncores = (int32_t)block_num;
  int32_t cid = (int32_t)block_idx;
  int32_t base = total_tasks / ncores;
  int32_t rem = total_tasks % ncores;
  int32_t start = cid * base + (cid < rem ? cid : rem);
  int32_t count = base + (cid < rem ? 1 : 0);
  int32_t t_blocks = (T_len + BT - 1) / BT;

  for (int32_t lt = 0; lt < count; ++lt) {
    int32_t tid = start + lt;
    int32_t bh = tid / dv_tiles;
    int32_t dv_t = tid % dv_tiles;
    int32_t b = bh / H;
    int32_t h = bh % H;
    int32_t dv0 = dv_t * 64;
    int32_t dv_cur = (Dv - dv0 < 64) ? (Dv - dv0) : 64;
    int64_t state_off = ((int64_t)b * H + h) * Dk * Dv + dv0;

    asc_copy_gm2ub_align((__ubuf__ uint8_t*)state_ub, (__gm__ uint8_t*)(init_state + state_off),
                         (uint32_t)Dk, (uint32_t)(dv_cur * 4), 0, 0, false,
                         asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                         (uint64_t)(Dv * 4), (uint32_t)(64 * 4));
    int32_t t_cur0 = (T_len < BT) ? T_len : BT;
    kda_dv64_issue_tb<T>(
        q, k, v, g, beta, q_in0, k_in0, v_in0, g_in0, beta_in0,
        b, h, T_len, H, Dk, Dv, dk_pad, dv0, dv_cur, 0, t_cur0);
    asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);
    asc_sync_notify(PIPE_V, PIPE_MTE2, EVENT_ID1);
    asc_sync_notify(PIPE_MTE3, PIPE_V, EVENT_ID0);
    asc_sync_notify(PIPE_MTE3, PIPE_V, EVENT_ID1);

    for (int32_t tb = 0; tb < t_blocks; ++tb) {
      int32_t buf = tb & 1;
      event_t ev = buf ? EVENT_ID1 : EVENT_ID0;
      event_t nxt_ev = buf ? EVENT_ID0 : EVENT_ID1;
      if (tb + 1 < t_blocks) {
        int32_t nxt_t0 = (tb + 1) * BT;
        int32_t nxt_t_cur = (T_len - nxt_t0 < BT) ? (T_len - nxt_t0) : BT;
        __ubuf__ T* nxt_q = buf ? q_in0 : q_in1;
        __ubuf__ T* nxt_k = buf ? k_in0 : k_in1;
        __ubuf__ T* nxt_v = buf ? v_in0 : v_in1;
        __ubuf__ float* nxt_g = buf ? g_in0 : g_in1;
        __ubuf__ float* nxt_beta = buf ? beta_in0 : beta_in1;
        asc_sync_wait(PIPE_V, PIPE_MTE2, nxt_ev);
        kda_dv64_issue_tb<T>(
            q, k, v, g, beta, nxt_q, nxt_k, nxt_v, nxt_g, nxt_beta,
            b, h, T_len, H, Dk, Dv, dk_pad, dv0, dv_cur, nxt_t0, nxt_t_cur);
        asc_sync_notify(PIPE_MTE2, PIPE_V, nxt_ev);
      }

      int32_t t0 = tb * BT;
      int32_t t_cur = (T_len - t0 < BT) ? (T_len - t0) : BT;
      __ubuf__ T* cur_q = buf ? q_in1 : q_in0;
      __ubuf__ T* cur_k = buf ? k_in1 : k_in0;
      __ubuf__ T* cur_v = buf ? v_in1 : v_in0;
      __ubuf__ float* cur_g = buf ? g_in1 : g_in0;
      __ubuf__ float* cur_beta = buf ? beta_in1 : beta_in0;
      __ubuf__ T* cur_out = buf ? out_ub1 : out_ub0;

      asc_sync_wait(PIPE_MTE2, PIPE_V, ev);
      asc_sync_wait(PIPE_MTE3, PIPE_V, ev);
      kda_dv64_block_vf<T>(state_ub, cur_q, cur_k, cur_v, cur_g, cur_beta, qf_ub, kf_ub, cur_out,
                           t_cur, Dk, dk_pad, dv_cur, use_l2norm, scale_val);
      asc_sync_notify(PIPE_V, PIPE_MTE2, ev);
      asc_sync_notify(PIPE_V, PIPE_MTE3, ev);
      asc_sync_wait(PIPE_V, PIPE_MTE3, ev);
      int64_t y_off = (((int64_t)b * T_len + t0) * H + h) * Dv + dv0;
      asc_copy_ub2gm_align((__gm__ uint8_t*)(y + y_off), (__ubuf__ uint8_t*)cur_out,
                           (uint32_t)t_cur, (uint32_t)(dv_cur * sizeof(T)),
                           asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                           (uint64_t)(H * Dv * sizeof(T)), (uint32_t)(64 * sizeof(T)));
      asc_sync_notify(PIPE_MTE3, PIPE_V, ev);
    }

    asc_sync_wait(PIPE_V, PIPE_MTE2, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_MTE2, EVENT_ID1);
    asc_sync_wait(PIPE_MTE3, PIPE_V, EVENT_ID0);
    asc_sync_wait(PIPE_MTE3, PIPE_V, EVENT_ID1);

    asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_copy_ub2gm_align((__gm__ uint8_t*)(final_state + state_off), (__ubuf__ uint8_t*)state_ub,
                         (uint32_t)Dk, (uint32_t)(dv_cur * 4),
                         asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                         (uint64_t)(Dv * 4), (uint32_t)(64 * 4));
    asc_sync_notify(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    asc_sync_wait(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    asc_sync_notify(PIPE_MTE3, PIPE_V, EVENT_ID0);
    asc_sync_wait(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  asc_sync();
}

extern "C" __global__ __aicore__ __vector__ void kda_dv64_ub_f16(
    __gm__ half* q, __gm__ half* k, __gm__ half* v, __gm__ float* g, __gm__ float* beta, __gm__ float* init_state,
    __gm__ half* y, __gm__ float* final_state,
    int32_t B, int32_t T_len, int32_t H, int32_t Dk, int32_t Dv, int32_t use_l2norm, float scale_val) {
  kda_dv64_ub_impl<half>(q, k, v, g, beta, init_state, y, final_state, B, T_len, H, Dk, Dv, use_l2norm, scale_val);
}
extern "C" __global__ __aicore__ __vector__ void kda_dv64_ub_bf16(
    __gm__ bfloat16_t* q, __gm__ bfloat16_t* k, __gm__ bfloat16_t* v, __gm__ float* g, __gm__ float* beta, __gm__ float* init_state,
    __gm__ bfloat16_t* y, __gm__ float* final_state,
    int32_t B, int32_t T_len, int32_t H, int32_t Dk, int32_t Dv, int32_t use_l2norm, float scale_val) {
  kda_dv64_ub_impl<bfloat16_t>(q, k, v, g, beta, init_state, y, final_state, B, T_len, H, Dk, Dv, use_l2norm, scale_val);
}
extern "C" __global__ __aicore__ __vector__ void kda_dv64_ub_f32(
    __gm__ float* q, __gm__ float* k, __gm__ float* v, __gm__ float* g, __gm__ float* beta, __gm__ float* init_state,
    __gm__ float* y, __gm__ float* final_state,
    int32_t B, int32_t T_len, int32_t H, int32_t Dk, int32_t Dv, int32_t use_l2norm, float scale_val) {
  kda_dv64_ub_impl<float>(q, k, v, g, beta, init_state, y, final_state, B, T_len, H, Dk, Dv, use_l2norm, scale_val);
}

extern "C" void launch_kda(
    void* stream,
    void* q, void* k, void* v, void* g, void* beta, void* init_state,
    void* y, void* final_state,
    void* ws_wq, void* ws_khat, void* ws_uloc, void* ws_yloc, void* ws_gamma, void* ws_u_scratch,
    int32_t B, int32_t T_len, int32_t H, int32_t Dk, int32_t Dv,
    int32_t use_l2norm, float scale_val, int32_t dtype_code) {
  int32_t dv_tiles = (Dv + 63) / 64;
  if (Dk > 128 || Dv > 128 || (int64_t)Dk * Dv * 4 > 192 * 1024 || (Dk <= 16 && T_len > 64)) {
    int32_t tasks = B * H * dv_tiles;
    uint32_t grid = (uint32_t)(tasks < 56 ? tasks : 56);
    if (dtype_code == 0) {
      kda_dv64_ub_f16<<<grid, nullptr, stream>>>(
          (__gm__ half*)q, (__gm__ half*)k, (__gm__ half*)v, (__gm__ float*)g, (__gm__ float*)beta,
          (__gm__ float*)init_state, (__gm__ half*)y, (__gm__ float*)final_state,
          B, T_len, H, Dk, Dv, use_l2norm, scale_val);
    } else if (dtype_code == 1) {
      kda_dv64_ub_bf16<<<grid, nullptr, stream>>>(
          (__gm__ bfloat16_t*)q, (__gm__ bfloat16_t*)k, (__gm__ bfloat16_t*)v, (__gm__ float*)g, (__gm__ float*)beta,
          (__gm__ float*)init_state, (__gm__ bfloat16_t*)y, (__gm__ float*)final_state,
          B, T_len, H, Dk, Dv, use_l2norm, scale_val);
    } else {
      kda_dv64_ub_f32<<<grid, nullptr, stream>>>(
          (__gm__ float*)q, (__gm__ float*)k, (__gm__ float*)v, (__gm__ float*)g, (__gm__ float*)beta,
          (__gm__ float*)init_state, (__gm__ float*)y, (__gm__ float*)final_state,
          B, T_len, H, Dk, Dv, use_l2norm, scale_val);
    }
    return;
  }

  if (T_len <= 4 || (Dk <= 16) || (Dk <= 32 && T_len <= 32)) {
    int32_t tasks = B * H * dv_tiles;
    uint32_t grid = (uint32_t)(tasks < 56 ? tasks : 56);
    if (dtype_code == 0) {
      kda_vec_fast_f16<<<grid, nullptr, stream>>>(
          (__gm__ half*)q, (__gm__ half*)k, (__gm__ half*)v, (__gm__ float*)g, (__gm__ float*)beta,
          (__gm__ float*)init_state, (__gm__ half*)y, (__gm__ float*)final_state,
          B, T_len, H, Dk, Dv, use_l2norm, scale_val);
    } else if (dtype_code == 1) {
      kda_vec_fast_bf16<<<grid, nullptr, stream>>>(
          (__gm__ bfloat16_t*)q, (__gm__ bfloat16_t*)k, (__gm__ bfloat16_t*)v, (__gm__ float*)g, (__gm__ float*)beta,
          (__gm__ float*)init_state, (__gm__ bfloat16_t*)y, (__gm__ float*)final_state,
          B, T_len, H, Dk, Dv, use_l2norm, scale_val);
    } else {
      kda_vec_fast_f32<<<grid, nullptr, stream>>>(
          (__gm__ float*)q, (__gm__ float*)k, (__gm__ float*)v, (__gm__ float*)g, (__gm__ float*)beta,
          (__gm__ float*)init_state, (__gm__ float*)y, (__gm__ float*)final_state,
          B, T_len, H, Dk, Dv, use_l2norm, scale_val);
    }
    return;
  }

  if (dtype_code == 0) {
    dispatch_stage1_stage2<half>(
        stream, (__gm__ half*)q, (__gm__ half*)k, (__gm__ half*)v, (__gm__ float*)g, (__gm__ float*)beta, (__gm__ float*)init_state,
        (__gm__ half*)y, (__gm__ float*)final_state,
        (__gm__ float*)ws_wq, (__gm__ float*)ws_khat, (__gm__ float*)ws_uloc, (__gm__ float*)ws_yloc, (__gm__ float*)ws_gamma, (__gm__ float*)ws_u_scratch,
        B, T_len, H, Dk, Dv, use_l2norm, scale_val);
  } else if (dtype_code == 1) {
    dispatch_stage1_stage2<bfloat16_t>(
        stream, (__gm__ bfloat16_t*)q, (__gm__ bfloat16_t*)k, (__gm__ bfloat16_t*)v, (__gm__ float*)g, (__gm__ float*)beta, (__gm__ float*)init_state,
        (__gm__ bfloat16_t*)y, (__gm__ float*)final_state,
        (__gm__ float*)ws_wq, (__gm__ float*)ws_khat, (__gm__ float*)ws_uloc, (__gm__ float*)ws_yloc, (__gm__ float*)ws_gamma, (__gm__ float*)ws_u_scratch,
        B, T_len, H, Dk, Dv, use_l2norm, scale_val);
  } else {
    dispatch_stage1_stage2<float>(
        stream, (__gm__ float*)q, (__gm__ float*)k, (__gm__ float*)v, (__gm__ float*)g, (__gm__ float*)beta, (__gm__ float*)init_state,
        (__gm__ float*)y, (__gm__ float*)final_state,
        (__gm__ float*)ws_wq, (__gm__ float*)ws_khat, (__gm__ float*)ws_uloc, (__gm__ float*)ws_yloc, (__gm__ float*)ws_gamma, (__gm__ float*)ws_u_scratch,
        B, T_len, H, Dk, Dv, use_l2norm, scale_val);
  }
}
'''


def _render_trackb_tu(**dims: object) -> str:
    """Render an instantiated C++20 Track-B translation unit for kimi_delta_attention."""
    raw_src = dims.get("trackb_src")
    return str(raw_src) if raw_src is not None else _TRACK_B_ASC_SRC


def _find_bisheng() -> str | None:
    found = shutil.which("bisheng")
    if found:
        return found
    for env_key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        base = os.environ.get(env_key, "")
        if base:
            for rel in ("bin/bisheng", "compiler/ccec_compiler/bin/bisheng"):
                cand = Path(base) / rel
                if cand.is_file():
                    return str(cand)
    return None


def _find_tl_include() -> str:
    pkg_dir = Path(tilelang.__file__).resolve().parent
    for cand in (pkg_dir.parent / "src", pkg_dir.parent, pkg_dir):
        if (cand / "tl_templates" / "ascend" / "common.h").is_file():
            return str(cand)
    return str(pkg_dir.parent / "src")


def _get_track_b_lib() -> ctypes.CDLL | None:
    bisheng = _find_bisheng()
    if not bisheng:
        return None
    src_hash = hashlib.sha256(_TRACK_B_ASC_SRC.encode("utf-8")).hexdigest()[:16]
    if src_hash in _TRACK_B_LIB_CACHE:
        return _TRACK_B_LIB_CACHE[src_hash]
    build_dir = Path(tempfile.gettempdir()) / f"tl_npu_kda_tb_{src_hash}"
    build_dir.mkdir(parents=True, exist_ok=True)
    so_path = build_dir / "kda_trackb.so"
    if not so_path.is_file():
        src_path = build_dir / "kda_trackb.asc"
        tmp_so = build_dir / f"kda_trackb.{os.getpid()}.tmp.so"
        src_path.write_text(_TRACK_B_ASC_SRC, encoding="utf-8")
        tl_inc = _find_tl_include()
        proc = subprocess.run(
            [
                bisheng,
                "-O2",
                "-fPIC",
                "-std=c++20",
                "-Wno-deprecated-declarations",
                "-Wno-ignored-attributes",
                "--npu-arch=dav-3510",
                f"-I{tl_inc}",
                "-shared",
                str(src_path),
                "-o",
                str(tmp_so),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"bisheng Track B compile failed: {proc.stderr[-2000:]}")
        os.replace(tmp_so, so_path)
    lib = ctypes.CDLL(str(so_path))
    lib.launch_kda.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_int32,
    ]
    _TRACK_B_LIB_CACHE[src_hash] = lib
    return lib


def _get_ws_pool(device: torch.device, needed_bytes: int) -> torch.Tensor:
    dev_key = str(device)
    buf = _TRACK_B_WS_POOL.get(dev_key)
    if buf is None or buf.numel() < needed_bytes:
        buf = torch.empty(max(needed_bytes, 4 * 1024 * 1024), dtype=torch.uint8, device=device)
        _TRACK_B_WS_POOL[dev_key] = buf
    return buf


def _run_track_b(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    use_qk_l2norm: bool,
    eff_scale: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    lib = _get_track_b_lib()
    if lib is None:
        return None
    import torch_npu

    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    gc = g.to(torch.float32).contiguous()
    bc = beta.to(torch.float32).contiguous()
    sc = initial_state.to(torch.float32).contiguous()
    y = torch.empty_like(vc)
    fs = torch.empty_like(sc)

    B, T_len, H, Dk = qc.shape
    Dv = vc.shape[-1]
    elem_sz = 2 if qc.dtype in (torch.float16, torch.bfloat16) else 4
    dcode = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}[qc.dtype]

    if (
        T_len <= 4
        or Dk <= 16
        or (Dk <= 32 and T_len <= 32)
        or Dk > 128
        or Dv > 128
        or Dk * Dv * 4 > 192 * 1024
    ):
        ws_ptrs = (0, 0, 0, 0, 0, 0)
    else:
        dk_pad = ((Dk + 63) // 64) * 64
        dv_pad = ((Dv + 63) // 64) * 64
        nc = (T_len + 31) // 32
        bh_nc = B * H * nc

        def align512(n: int) -> int:
            return (n + 511) & ~511

        s0 = align512(bh_nc * 64 * dk_pad * 4)
        s1 = align512(bh_nc * 32 * dk_pad * 4)
        s2 = align512(bh_nc * 32 * dv_pad * 4)
        s3 = align512(bh_nc * 32 * dv_pad * 4)
        s4 = align512(bh_nc * dk_pad * 4)
        s5 = align512(28 * 32 * 128 * 4)
        total = s0 + s1 + s2 + s3 + s4 + s5
        base = _get_ws_pool(qc.device, total).data_ptr()
        ws_ptrs = (
            base,
            base + s0,
            base + s0 + s1,
            base + s0 + s1 + s2,
            base + s0 + s1 + s2 + s3,
            base + s0 + s1 + s2 + s3 + s4,
        )

    stream = torch_npu.npu.current_stream().npu_stream
    lib.launch_kda(
        ctypes.c_void_p(stream),
        ctypes.c_void_p(qc.data_ptr()),
        ctypes.c_void_p(kc.data_ptr()),
        ctypes.c_void_p(vc.data_ptr()),
        ctypes.c_void_p(gc.data_ptr()),
        ctypes.c_void_p(bc.data_ptr()),
        ctypes.c_void_p(sc.data_ptr()),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_void_p(fs.data_ptr()),
        ctypes.c_void_p(ws_ptrs[0]),
        ctypes.c_void_p(ws_ptrs[1]),
        ctypes.c_void_p(ws_ptrs[2]),
        ctypes.c_void_p(ws_ptrs[3]),
        ctypes.c_void_p(ws_ptrs[4]),
        ctypes.c_void_p(ws_ptrs[5]),
        B,
        T_len,
        H,
        Dk,
        Dv,
        int(bool(use_qk_l2norm)),
        float(eff_scale),
        dcode,
    )
    return y, fs


def kimi_delta_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    use_qk_l2norm: bool = True,
    scaleValue: float = -1.0,
    *,
    schedule_mode: str = "tuned",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute KimiDeltaAttention matching cannbot-ops level5/kimi_delta_attention proto.yaml."""
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported dtype {q.dtype}")
    dk = q.shape[-1]
    eff_scale = (dk ** -0.5) if scaleValue <= 0 else float(scaleValue)

    if q.device.type != "npu":
        qf = q.to(torch.float32)
        kf = k.to(torch.float32)
        vf = v.to(torch.float32)
        decay = g.to(torch.float32).exp()
        beta_f = beta.to(torch.float32)
        if use_qk_l2norm:
            qf = qf * torch.rsqrt(qf.square().sum(dim=-1, keepdim=True) + 1e-6)
            kf = kf * torch.rsqrt(kf.square().sum(dim=-1, keepdim=True) + 1e-6)
        qf = qf * eff_scale
        state = initial_state.to(torch.float32).clone()
        out = torch.empty(vf.shape, dtype=torch.float32, device=q.device)
        for tok in range(q.shape[1]):
            hbar = state * decay[:, tok].unsqueeze(-1)
            kt = kf[:, tok]
            pred = torch.sum(hbar * kt.unsqueeze(-1), dim=-2)
            corr = (vf[:, tok] - pred) * beta_f[:, tok].unsqueeze(-1)
            state = hbar + kt.unsqueeze(-1) * corr.unsqueeze(-2)
            out[:, tok] = torch.sum(state * qf[:, tok].unsqueeze(-1), dim=-2)
        return out.to(q.dtype), state

    if schedule_mode == "tuned" and dk <= 256 and v.shape[-1] <= 512:
        res = _run_track_b(q, k, v, g, beta, initial_state, use_qk_l2norm, eff_scale)
        if res is not None:
            return res

    tl_dtype = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }[q.dtype]
    compiled = _get_kernel(
        dk=dk,
        dv=v.shape[-1],
        tl_dtype=tl_dtype,
        use_qk_l2norm=bool(use_qk_l2norm),
        t_dim=q.shape[1],
        schedule_mode=schedule_mode,
    )
    return compiled(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        initial_state.contiguous(),
        float(eff_scale),
    )


kernel = kimi_delta_attention

__all__ = [
    "_build_kimi_delta_attention_prim_func",
    "_select_kimi_delta_attention_config",
    "kernel",
    "kimi_delta_attention",
]
