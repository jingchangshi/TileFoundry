"""GroupedMatmul (level3/grouped_matmul) operator modules and NPU kernels for Ascend950PR."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import tempfile
from typing import Optional, Sequence

import tilelang
import torch
from tilelang.contrib.bisheng import compile_ascend

from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS

_KERNEL_CACHE: dict[tuple[object, ...], object] = {}


def _select_grouped_matmul_config(
    *,
    group_sizes: tuple[int, ...] = (128,),
    k_dim: int = 256,
    n_dim: int = 512,
    tl_dtype: str = "float16",
    transpose_weight: bool = False,
    has_bias: bool = False,
    schedule_mode: str = "tuned",
) -> dict[str, object]:
    """Select tile/schedule parameters across ragged group regimes on Ascend950PR."""
    non_empty = [m for m in group_sizes if m > 0]
    max_m = max(non_empty) if non_empty else 0
    total_m = sum(non_empty)
    num_groups = len(group_sizes)

    if schedule_mode != "tuned":
        return {
            "bm": 64,
            "bn": 64,
            "bk": 64,
            "k_dim": int(k_dim),
            "n_dim": int(n_dim),
            "num_stages": 2,
            "num_cores": 28,
            "has_bias": bool(has_bias),
            "transpose_weight": bool(transpose_weight),
            "tl_dtype": tl_dtype,
            "schedule_mode": schedule_mode,
            "compile_flags": list(PURE_SIMDVF_UBUF_FLAGS),
        }

    if tl_dtype == "float32":
        bm, bn = 64, 64
    elif max_m <= 64:
        bm = 32
        bn = 128 if (n_dim >= 512 and total_m >= 96) else 64
    elif max_m <= 256 and n_dim <= 256:
        bm = 128 if total_m >= 512 else 64
        bn = 128 if (n_dim > 128 and total_m >= 512) else 64
    elif max_m <= 256 and n_dim <= 512:
        bm, bn = 64, 64
    elif tl_dtype != "float32" and total_m >= 512 and n_dim >= 2047:
        bm, bn = 128, 256
    elif tl_dtype != "float32" and total_m >= 512 and n_dim >= 1023:
        bm, bn = 128, 128
    elif max_m <= 256 and n_dim <= 1024:
        bm, bn = 64, 128
    else:
        bm, bn = 128, 128

    bk = 128 if (k_dim >= 256 and tl_dtype != "float32") else (64 if k_dim >= 64 else 32)

    return {
        "bm": bm,
        "bn": bn,
        "bk": bk,
        "k_dim": int(k_dim),
        "n_dim": int(n_dim),
        "num_stages": 2,
        "num_cores": 28,
        "has_bias": bool(has_bias),
        "transpose_weight": bool(transpose_weight),
        "tl_dtype": tl_dtype,
        "schedule_mode": "tuned",
        "compile_flags": list(PURE_SIMDVF_UBUF_FLAGS),
    }


_GMM_TRACKB_SRC = r"""
#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/gemm.h>
#include <tl_templates/ascend/simd_inst.h>

struct GmmGroupMeta {
  int starts[32];
  int ends[32];
};

template <int M, int K, int N, int TILE_K_SUB, typename InT, typename AccumT>
__aicore__ inline void ascend_gemm_l1_nn(__cc__ AccumT *cc_ptr, __cbuf__ InT *cbuf_a_ptr,
                                          __cbuf__ InT *cbuf_b_ptr, int clear_accum, int buf_offset = 0) {
  constexpr int C0 = 32 / sizeof(InT);
  constexpr int FRAC_N = 16;
  constexpr int SUB_K = K / TILE_K_SUB;
  constexpr int mStepA = M / FRAC_N;
  constexpr int mStepB = N / FRAC_N;
  constexpr int kStep = TILE_K_SUB / C0;
  constexpr int L0A_STAGE = M * TILE_K_SUB;
  constexpr int L0B_STAGE = N * TILE_K_SUB;
  constexpr int srcStrideNN = K / C0;

  __ca__ InT *l0a = (__ca__ InT *)0;
  __cb__ InT *l0b = (__cb__ InT *)0;

  for (int sk = 0; sk < SUB_K; ++sk) {
    asc_lock(PIPE_MTE1, buf_offset + (sk & 1), ASC_LOCK_BLOCK);
    asc_copy_l12l0a((__ca__ InT *)(l0a + ((sk & 1) * L0A_STAGE)), cbuf_a_ptr, 0,
                    (sk * kStep), mStepA, kStep, mStepA, mStepA);
    asc_copy_l12l0b_transpose((__cb__ InT *)(l0b + ((sk & 1) * L0B_STAGE)), cbuf_b_ptr,
                              (sk * kStep), 0, kStep, mStepB, srcStrideNN, mStepB);
    asc_unlock(PIPE_MTE1, buf_offset + (sk & 1), ASC_LOCK_BLOCK);
    asc_lock(PIPE_M, buf_offset + (sk & 1), ASC_LOCK_BLOCK);
    asc_mmad(cc_ptr, (__ca__ InT *)(l0a + ((sk & 1) * L0A_STAGE)),
             (__cb__ InT *)(l0b + ((sk & 1) * L0B_STAGE)), M, TILE_K_SUB, N,
             0, true, false, clear_accum && (sk == 0));
    asc_unlock(PIPE_M, buf_offset + (sk & 1), ASC_LOCK_BLOCK);
  }
}

template <int BN, typename BiasT>
__simd_vf__ inline void vf_bias_to_f32(__ubuf__ BiasT* bias_in, __ubuf__ float* bias_f32) {
  vector_bool m32 = asc_create_mask_b32(PAT_ALL);
  vector_bool m16 = asc_create_mask_b16(PAT_ALL);
  constexpr int CHUNKS = BN >> 6;
  for (int c = 0; c < CHUNKS; ++c) {
    auto bv = simd_inst::vcvt<float>(simd_inst::vlds_upsample<BiasT>(bias_in + (c << 6), 0), m16, PART_EVEN, MODE_ZEROING);
    simd_inst::vsts_norm(bv, bias_f32 + (c << 6), 0, m32);
  }
}

template <int ROWS, int BN, int HAS_BIAS, typename OutT>
__simd_vf__ inline void vf_gmm_epilogue_16(__ubuf__ float* c_ub, __ubuf__ float* bias_f32, __ubuf__ OutT* out_ub, int my_rows) {
  vector_bool m32 = asc_create_mask_b32(PAT_ALL);
  vector_bool m16 = asc_create_mask_b16(PAT_ALL);
  constexpr int CHUNKS = BN >> 6;
  for (int r = 0; r < my_rows; ++r) {
    for (int c = 0; c < CHUNKS; ++c) {
      int off = c << 6;
      auto cv = simd_inst::vlds_norm<float>(c_ub + r * BN + off, 0);
      if constexpr (HAS_BIAS) {
        auto bv = simd_inst::vlds_norm<float>(bias_f32 + off, 0);
        cv = simd_inst::vadd(cv, bv, m32, MODE_ZEROING);
      }
      auto ov = simd_inst::vcvt<OutT>(cv, m16, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
      simd_inst::vsts_pack_b32(ov, out_ub + r * BN + off, 0, m32);
    }
  }
}

template <int ROWS, int BN, int HAS_BIAS>
__simd_vf__ inline void vf_gmm_epilogue_f32(__ubuf__ float* c_ub, __ubuf__ float* bias_f32, int my_rows) {
  vector_bool m32 = asc_create_mask_b32(PAT_ALL);
  constexpr int CHUNKS = BN >> 6;
  for (int r = 0; r < my_rows; ++r) {
    for (int c = 0; c < CHUNKS; ++c) {
      int off = c << 6;
      auto cv = simd_inst::vlds_norm<float>(c_ub + r * BN + off, 0);
      if constexpr (HAS_BIAS) {
        auto bv = simd_inst::vlds_norm<float>(bias_f32 + off, 0);
        cv = simd_inst::vadd(cv, bv, m32, MODE_ZEROING);
      }
      simd_inst::vsts_norm(cv, c_ub + r * BN + off, 0, m32);
    }
  }
}

__simd_vf__ inline void vf_split_f32_hilo(__ubuf__ float* x_ub, __ubuf__ half* hi_ub, __ubuf__ half* lo_ub, int chunks) {
  vector_bool m32 = asc_create_mask_b32(PAT_ALL);
  vector_bool m16 = asc_create_mask_b16(PAT_ALL);
  for (int c = 0; c < chunks; ++c) {
    int off = c << 6;
    auto xf = simd_inst::vlds_norm<float>(x_ub + off, 0);
    auto h16 = simd_inst::vcvt<half>(xf, m16, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
    simd_inst::vsts_pack_b32(h16, hi_ub + off, 0, m32);
    auto hf = simd_inst::vcvt<float>(h16, m16, PART_EVEN, MODE_ZEROING);
    auto rem = simd_inst::vsub(xf, hf, m32, MODE_ZEROING);
    auto l16 = simd_inst::vcvt<half>(rem, m16, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
    simd_inst::vsts_pack_b32(l16, lo_ub + off, 0, m32);
  }
}

__global__ __vector__ void split_f32_to_hilo_pair_kernel(
    __gm__ float* src_x, __gm__ half* dst_x_hi, __gm__ half* dst_x_lo, int nx,
    __gm__ float* src_w, __gm__ half* dst_w_hi, __gm__ half* dst_w_lo, int nw) {
  asc_init();
  constexpr int BLK = 8192;
  __ubuf__ float* x0 = (__ubuf__ float*)0;
  __ubuf__ float* x1 = (__ubuf__ float*)(BLK * 4);
  __ubuf__ half* h0 = (__ubuf__ half*)(BLK * 8);
  __ubuf__ half* h1 = (__ubuf__ half*)(BLK * 8 + BLK * 2);
  __ubuf__ half* l0 = (__ubuf__ half*)(BLK * 8 + BLK * 4);
  __ubuf__ half* l1 = (__ubuf__ half*)(BLK * 8 + BLK * 6);
  int cid = (int)block_idx;
  int num_cores = 56;
  int blks_x = (nx + BLK - 1) / BLK;
  int blks_w = (nw + BLK - 1) / BLK;
  int total_blks = blks_x + blks_w;
  if (cid >= total_blks) return;
  asc_sync_notify(PIPE_V, PIPE_MTE2, (event_t)0);
  asc_sync_notify(PIPE_V, PIPE_MTE2, (event_t)1);
  asc_sync_notify(PIPE_MTE3, PIPE_V, (event_t)0);
  asc_sync_notify(PIPE_MTE3, PIPE_V, (event_t)1);
  int seq = 0;
  for (int bi = cid; bi < total_blks; bi += num_cores) {
    int stg = seq & 1;
    bool is_w = (bi >= blks_x);
    int local_b = is_w ? (bi - blks_x) : bi;
    int off = local_b * BLK;
    int lim = is_w ? nw : nx;
    int cur = lim - off;
    if (cur > BLK) cur = BLK;
    int chunks = (cur + 63) >> 6;
    __gm__ float* s_ptr = (is_w ? src_w : src_x) + off;
    __gm__ half* dh_ptr = (is_w ? dst_w_hi : dst_x_hi) + off;
    __gm__ half* dl_ptr = (is_w ? dst_w_lo : dst_x_lo) + off;
    __ubuf__ float* cur_x = stg ? x1 : x0;
    __ubuf__ half* cur_h = stg ? h1 : h0;
    __ubuf__ half* cur_l = stg ? l1 : l0;
    asc_sync_wait(PIPE_V, PIPE_MTE2, (event_t)stg);
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)cur_x, (__gm__ uint8_t*)s_ptr,
                         1, cur * 4, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), cur * 4, cur * 4);
    asc_sync_notify(PIPE_MTE2, PIPE_V, (event_t)stg);
    asc_sync_wait(PIPE_MTE2, PIPE_V, (event_t)stg);
    asc_sync_wait(PIPE_MTE3, PIPE_V, (event_t)stg);
    vf_split_f32_hilo(cur_x, cur_h, cur_l, chunks);
    asc_sync_notify(PIPE_V, PIPE_MTE2, (event_t)stg);
    asc_sync_notify(PIPE_V, PIPE_MTE3, (event_t)stg);
    asc_sync_wait(PIPE_V, PIPE_MTE3, (event_t)stg);
    asc_copy_ub2gm_align((__gm__ uint8_t*)dh_ptr, (__ubuf__ uint8_t*)cur_h,
                         1, cur * 2, static_cast<asc_store_l2_cache_mode>(4), cur * 2, cur * 2);
    asc_copy_ub2gm_align((__gm__ uint8_t*)dl_ptr, (__ubuf__ uint8_t*)cur_l,
                         1, cur * 2, static_cast<asc_store_l2_cache_mode>(4), cur * 2, cur * 2);
    asc_sync_notify(PIPE_MTE3, PIPE_V, (event_t)stg);
    ++seq;
  }
  asc_sync_wait(PIPE_V, PIPE_MTE2, (event_t)0);
  asc_sync_wait(PIPE_V, PIPE_MTE2, (event_t)1);
  asc_sync_wait(PIPE_MTE3, PIPE_V, (event_t)0);
  asc_sync_wait(PIPE_MTE3, PIPE_V, (event_t)1);
}

template <int BM, int BN, int BK, int TRANS_W, int A_STAGE, int B_STAGE, typename InT>
__aicore__ inline void gmm_issue_k_dma(
    __cbuf__ InT* a_l1_base, __cbuf__ InT* b_l1_base,
    __gm__ InT* X, __gm__ InT* W,
    int g, int m0, int m_cur, int n0, int n_cur, int K, int N,
    int k_idx, int stg_id) {
  int k0 = k_idx * BK;
  int k_cur = K - k0;
  if (k_cur > BK) k_cur = BK;
  __cbuf__ InT* a_dst = a_l1_base + stg_id * A_STAGE;
  __cbuf__ InT* b_dst = b_l1_base + stg_id * B_STAGE;
  asc_sync_wait(PIPE_MTE1, PIPE_MTE2, (event_t)stg_id);
  if (k_cur <= BK - 16) {
    asc_fill_value_config fa{}; fa.repeat = BK / 16; fa.blk_num = BM; fa.dst_gap = 0;
    asc_fill_l1((__cbuf__ uint16_t*)a_dst, (uint32_t)0, fa);
    if constexpr (TRANS_W) {
      asc_fill_value_config fb{}; fb.repeat = BK / 16; fb.blk_num = BN; fb.dst_gap = 0;
      asc_fill_l1((__cbuf__ uint16_t*)b_dst, (uint32_t)0, fb);
    } else {
      asc_fill_value_config fb{}; fb.repeat = BN / 16; fb.blk_num = BK; fb.dst_gap = 0;
      asc_fill_l1((__cbuf__ uint16_t*)b_dst, (uint32_t)0, fb);
    }
    pipe_barrier(PIPE_MTE2);
  }
  asc_set_gm2l1_nz_para(1, 1, BM, 0);
  asc_copy_gm2l1_nd2nz(a_dst, X + (int64_t)m0 * K + k0, K * 2,
                       static_cast<asc_load_l2_cache_mode>(0), m_cur, k_cur, 0, 0);

  if constexpr (TRANS_W) {
    asc_set_gm2l1_nz_para(1, 1, BN, 0);
    asc_copy_gm2l1_nd2nz(b_dst, W + ((int64_t)g * N + n0) * K + k0, K * 2,
                         static_cast<asc_load_l2_cache_mode>(0), n_cur, k_cur, 0, 0);
  } else {
    asc_set_gm2l1_nz_para(1, 1, BK, 0);
    asc_copy_gm2l1_nd2nz(b_dst, W + ((int64_t)g * K + k0) * N + n0, N * 2,
                         static_cast<asc_load_l2_cache_mode>(0), k_cur, n_cur, 0, 0);
  }
  asc_sync_notify(PIPE_MTE2, PIPE_MTE1, (event_t)stg_id);
}

template <int BM, int BN, int BK, int TRANS_W, int A_STAGE, int B_STAGE, typename InT>
__aicore__ inline void gmm_consume_k_stage(
    __cc__ float* cur_l0c, __cbuf__ InT* a_l1_base, __cbuf__ InT* b_l1_base,
    int stg, int clear_accum) {
  __cbuf__ InT* a_dst = a_l1_base + stg * A_STAGE;
  __cbuf__ InT* b_dst = b_l1_base + stg * B_STAGE;
  asc_sync_wait(PIPE_MTE2, PIPE_MTE1, (event_t)stg);

  constexpr int SUB_K_STEP = (BK < 64) ? BK : 64;
  if constexpr (TRANS_W) {
    ascend_gemm_l1<BM, BK, BN, SUB_K_STEP, true, InT>(cur_l0c, a_dst, b_dst, clear_accum, 0, 0);
  } else {
    ascend_gemm_l1_nn<BM, BK, BN, SUB_K_STEP, InT, float>(cur_l0c, a_dst, b_dst, clear_accum, 0);
  }
  asc_sync_notify(PIPE_MTE1, PIPE_MTE2, (event_t)stg);
}

template <int BM, int BN, int BK, int TRANS_W, int HAS_BIAS, typename InT, typename BiasT>
__global__ __mix__(1, 2) void gmm_persistent_16(
    __gm__ InT* X, __gm__ InT* W, __gm__ BiasT* Bias, __gm__ InT* Y,
    GmmGroupMeta meta, int E, int K, int N) {
  asc_init();
  constexpr int ROWS = BM / 2;
  constexpr int A_STAGE = BM * BK;
  constexpr int B_STAGE = TRANS_W ? (BN * BK) : (BK * BN);

  __cbuf__ InT* a_l1_base = (__cbuf__ InT*)0;
  __cbuf__ InT* b_l1_base = (__cbuf__ InT*)(2 * A_STAGE * 2);

  __cc__ float* c_l0c0 = (__cc__ float*)0;
  __cc__ float* c_l0c1 = (__cc__ float*)(BM * BN * 4);

  constexpr int C_UB_BYTES = ROWS * BN * 4;
  constexpr int OUT_UB_BYTES = ROWS * BN * 2;
  __ubuf__ float* c_ub0 = (__ubuf__ float*)0;
  __ubuf__ float* c_ub1 = (__ubuf__ float*)(C_UB_BYTES);
  __ubuf__ InT* out_ub0 = (__ubuf__ InT*)(2 * C_UB_BYTES);
  __ubuf__ InT* out_ub1 = (__ubuf__ InT*)(2 * C_UB_BYTES + OUT_UB_BYTES);
  __ubuf__ BiasT* bias_raw_ub0 = (__ubuf__ BiasT*)(2 * C_UB_BYTES + 2 * OUT_UB_BYTES);
  __ubuf__ BiasT* bias_raw_ub1 = (__ubuf__ BiasT*)(2 * C_UB_BYTES + 2 * OUT_UB_BYTES + 1024);
  __ubuf__ float* bias_f32_ub0 = (__ubuf__ float*)(2 * C_UB_BYTES + 2 * OUT_UB_BYTES + 2048);
  __ubuf__ float* bias_f32_ub1 = (__ubuf__ float*)(2 * C_UB_BYTES + 2 * OUT_UB_BYTES + 4096);

  int n_tiles = (N + BN - 1) / BN;
  int total_tiles = 0;
  for (int g = 0; g < E; ++g) {
    int mg = meta.ends[g] - meta.starts[g];
    if (mg > 0) total_tiles += ((mg + BM - 1) / BM) * n_tiles;
  }
  int cid = (int)block_idx;
  int num_cores = 28;
  if (cid >= total_tiles) return;

  int k_iters = (K + BK - 1) / BK;

  if ASCEND_IS_AIC {
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, (event_t)0);
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, (event_t)1);
    asc_sync_notify(PIPE_FIX, PIPE_M, (event_t)0);
    asc_sync_notify(PIPE_FIX, PIPE_M, (event_t)1);
    uint32_t inflight = 0;
    int tile_seq = 0;

    for (int tid = cid; tid < total_tiles; tid += num_cores) {
      int rem = tid;
      int g = 0;
      int mt = 0;
      int nt = 0;
      for (; g < E; ++g) {
        int mg = meta.ends[g] - meta.starts[g];
        if (mg <= 0) continue;
        int m_blks = (mg + BM - 1) / BM;
        int g_tiles = m_blks * n_tiles;
        if (rem < g_tiles) {
          mt = rem / n_tiles;
          nt = rem % n_tiles;
          break;
        }
        rem -= g_tiles;
      }
      int m0 = meta.starts[g] + mt * BM;
      int m_cur = meta.ends[g] - m0;
      if (m_cur > BM) m_cur = BM;
      int n0 = nt * BN;
      int n_cur = N - n0;
      if (n_cur > BN) n_cur = BN;

      int buf = tile_seq & 1;
      __cc__ float* cur_l0c = buf ? c_l0c1 : c_l0c0;
      __ubuf__ float* cur_cub = buf ? c_ub1 : c_ub0;
      asc_sync_wait(PIPE_FIX, PIPE_M, (event_t)buf);

      if (k_iters > 0) {
        gmm_issue_k_dma<BM, BN, BK, TRANS_W, A_STAGE, B_STAGE, InT>(
            a_l1_base, b_l1_base, X, W, g, m0, m_cur, n0, n_cur, K, N, 0, 0);
      }
      for (int ki = 0; ki < k_iters; ++ki) {
        int stg = ki & 1;
        if (ki + 1 < k_iters) {
          gmm_issue_k_dma<BM, BN, BK, TRANS_W, A_STAGE, B_STAGE, InT>(
              a_l1_base, b_l1_base, X, W, g, m0, m_cur, n0, n_cur, K, N, ki + 1, (ki + 1) & 1);
        }
        gmm_consume_k_stage<BM, BN, BK, TRANS_W, A_STAGE, B_STAGE, InT>(
            cur_l0c, a_l1_base, b_l1_base, stg, (ki == 0) ? 1 : 0);
      }

      asc_sync_notify(PIPE_M, PIPE_FIX, (event_t)buf);
      asc_sync_wait(PIPE_M, PIPE_FIX, (event_t)buf);
      if (inflight & (1u << buf)) {
        if (buf == 0) { asc_sync_intra_wait(PIPE_FIX, 6); asc_sync_intra_wait(PIPE_FIX, 22); }
        else { asc_sync_intra_wait(PIPE_FIX, 8); asc_sync_intra_wait(PIPE_FIX, 24); }
      }
      asc_set_l0c_copy_nz_para(1, 0, 0);
      asc_copy_l0c2ub(cur_cub, cur_l0c, BN, BM, BN, BM, 0,
                      static_cast<asc_dual_dst_mode>(1), static_cast<asc_unit_flag_mode>(0),
                      static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
      asc_sync_notify(PIPE_FIX, PIPE_M, (event_t)buf);
      if (buf == 0) { asc_sync_intra_arrive(PIPE_FIX, 3); asc_sync_intra_arrive(PIPE_FIX, 19); }
      else { asc_sync_intra_arrive(PIPE_FIX, 7); asc_sync_intra_arrive(PIPE_FIX, 23); }
      inflight |= (1u << buf);
      ++tile_seq;
    }

    if (inflight & 1u) { asc_sync_intra_wait(PIPE_FIX, 6); asc_sync_intra_wait(PIPE_FIX, 22); }
    if (inflight & 2u) { asc_sync_intra_wait(PIPE_FIX, 8); asc_sync_intra_wait(PIPE_FIX, 24); }
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, (event_t)0);
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, (event_t)1);
    asc_sync_wait(PIPE_FIX, PIPE_M, (event_t)0);
    asc_sync_wait(PIPE_FIX, PIPE_M, (event_t)1);
  }

  if ASCEND_IS_AIV {
    int sid = asc_get_sub_block_id();
    int done0 = (sid == 0) ? 6 : 22;
    int done1 = (sid == 0) ? 8 : 24;
    asc_sync_notify(PIPE_MTE3, PIPE_V, (event_t)0);
    asc_sync_notify(PIPE_MTE3, PIPE_V, (event_t)1);
    asc_sync_notify(PIPE_V, PIPE_MTE2, (event_t)0);
    asc_sync_notify(PIPE_V, PIPE_MTE2, (event_t)1);
    int tile_seq = 0;

    for (int tid = cid; tid < total_tiles; tid += num_cores) {
      int rem = tid;
      int g = 0;
      int mt = 0;
      int nt = 0;
      for (; g < E; ++g) {
        int mg = meta.ends[g] - meta.starts[g];
        if (mg <= 0) continue;
        int m_blks = (mg + BM - 1) / BM;
        int g_tiles = m_blks * n_tiles;
        if (rem < g_tiles) {
          mt = rem / n_tiles;
          nt = rem % n_tiles;
          break;
        }
        rem -= g_tiles;
      }
      int m0 = meta.starts[g] + mt * BM;
      int m_cur = meta.ends[g] - m0;
      if (m_cur > BM) m_cur = BM;
      int n0 = nt * BN;
      int n_cur = N - n0;
      if (n_cur > BN) n_cur = BN;

      int my_m_start = sid * ROWS;
      int my_rows = m_cur - my_m_start;
      if (my_rows > ROWS) my_rows = ROWS;
      if (my_rows < 0) my_rows = 0;

      int buf = tile_seq & 1;
      __ubuf__ float* cur_cub = buf ? c_ub1 : c_ub0;
      __ubuf__ InT* cur_oub = buf ? out_ub1 : out_ub0;
      __ubuf__ BiasT* cur_braw = buf ? bias_raw_ub1 : bias_raw_ub0;
      __ubuf__ float* cur_bf32 = buf ? bias_f32_ub1 : bias_f32_ub0;

      if constexpr (HAS_BIAS) {
        asc_sync_wait(PIPE_V, PIPE_MTE2, (event_t)buf);
        if constexpr (sizeof(BiasT) == 4) {
          asc_copy_gm2ub_align((__ubuf__ uint8_t*)cur_bf32, (__gm__ uint8_t*)(Bias + (int64_t)g * N + n0),
                               1, n_cur * 4, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), n_cur * 4, n_cur * 4);
          asc_sync_notify(PIPE_MTE2, PIPE_V, (event_t)buf);
          asc_sync_wait(PIPE_MTE2, PIPE_V, (event_t)buf);
        } else {
          asc_copy_gm2ub_align((__ubuf__ uint8_t*)cur_braw, (__gm__ uint8_t*)(Bias + (int64_t)g * N + n0),
                               1, n_cur * 2, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), n_cur * 2, n_cur * 2);
          asc_sync_notify(PIPE_MTE2, PIPE_V, (event_t)buf);
          asc_sync_wait(PIPE_MTE2, PIPE_V, (event_t)buf);
          vf_bias_to_f32<BN, BiasT>(cur_braw, cur_bf32);
        }
      }

      asc_sync_intra_wait(PIPE_V, buf ? 7 : 3);
      asc_sync_wait(PIPE_MTE3, PIPE_V, (event_t)buf);
      if (my_rows > 0) {
        vf_gmm_epilogue_16<ROWS, BN, HAS_BIAS, InT>(cur_cub, cur_bf32, cur_oub, my_rows);
      }
      if constexpr (HAS_BIAS) {
        asc_sync_notify(PIPE_V, PIPE_MTE2, (event_t)buf);
      }
      asc_sync_notify(PIPE_V, PIPE_MTE3, (event_t)buf);
      asc_sync_wait(PIPE_V, PIPE_MTE3, (event_t)buf);
      if (my_rows > 0) {
        asc_copy_ub2gm_align((__gm__ uint8_t*)(Y + ((int64_t)m0 + my_m_start) * N + n0), (__ubuf__ uint8_t*)cur_oub,
                             my_rows, n_cur * 2, static_cast<asc_store_l2_cache_mode>(4),
                             N * 2, BN * 2);
      }
      asc_sync_notify(PIPE_MTE3, PIPE_V, (event_t)buf);
      asc_sync_intra_arrive(PIPE_V, buf ? done1 : done0);
      ++tile_seq;
    }

    asc_sync_wait(PIPE_MTE3, PIPE_V, (event_t)0);
    asc_sync_wait(PIPE_MTE3, PIPE_V, (event_t)1);
    asc_sync_wait(PIPE_V, PIPE_MTE2, (event_t)0);
    asc_sync_wait(PIPE_V, PIPE_MTE2, (event_t)1);
  }
}

template <int BM, int BN, int BK, int TRANS_W, int A_STAGE, int B_STAGE>
__aicore__ inline void gmm_issue_f32_k_dma(
    __cbuf__ half* a_hi_base, __cbuf__ half* a_lo_base,
    __cbuf__ half* b_hi_base, __cbuf__ half* b_lo_base,
    __gm__ half* X_hi, __gm__ half* X_lo,
    __gm__ half* W_hi, __gm__ half* W_lo,
    int g, int m0, int m_cur, int n0, int n_cur, int K, int N,
    int k_idx, int stg_id) {
  int k0 = k_idx * BK;
  int k_cur = K - k0;
  if (k_cur > BK) k_cur = BK;
  __cbuf__ half* ah = a_hi_base + stg_id * A_STAGE;
  __cbuf__ half* al = a_lo_base + stg_id * A_STAGE;
  __cbuf__ half* bh = b_hi_base + stg_id * B_STAGE;
  __cbuf__ half* bl = b_lo_base + stg_id * B_STAGE;
  asc_sync_wait(PIPE_MTE1, PIPE_MTE2, (event_t)stg_id);
  if (k_cur <= BK - 16) {
    asc_fill_value_config fa{}; fa.repeat = BK / 16; fa.blk_num = BM; fa.dst_gap = 0;
    asc_fill_l1((__cbuf__ uint16_t*)ah, (uint32_t)0, fa);
    asc_fill_l1((__cbuf__ uint16_t*)al, (uint32_t)0, fa);
    if constexpr (TRANS_W) {
      asc_fill_value_config fb{}; fb.repeat = BK / 16; fb.blk_num = BN; fb.dst_gap = 0;
      asc_fill_l1((__cbuf__ uint16_t*)bh, (uint32_t)0, fb);
      asc_fill_l1((__cbuf__ uint16_t*)bl, (uint32_t)0, fb);
    } else {
      asc_fill_value_config fb{}; fb.repeat = BN / 16; fb.blk_num = BK; fb.dst_gap = 0;
      asc_fill_l1((__cbuf__ uint16_t*)bh, (uint32_t)0, fb);
      asc_fill_l1((__cbuf__ uint16_t*)bl, (uint32_t)0, fb);
    }
    pipe_barrier(PIPE_MTE2);
  }
  asc_set_gm2l1_nz_para(1, 1, BM, 0);
  asc_copy_gm2l1_nd2nz(ah, X_hi + (int64_t)m0 * K + k0, K * 2, static_cast<asc_load_l2_cache_mode>(0), m_cur, k_cur, 0, 0);
  asc_copy_gm2l1_nd2nz(al, X_lo + (int64_t)m0 * K + k0, K * 2, static_cast<asc_load_l2_cache_mode>(0), m_cur, k_cur, 0, 0);
  if constexpr (TRANS_W) {
    asc_set_gm2l1_nz_para(1, 1, BN, 0);
    asc_copy_gm2l1_nd2nz(bh, W_hi + ((int64_t)g * N + n0) * K + k0, K * 2, static_cast<asc_load_l2_cache_mode>(0), n_cur, k_cur, 0, 0);
    asc_copy_gm2l1_nd2nz(bl, W_lo + ((int64_t)g * N + n0) * K + k0, K * 2, static_cast<asc_load_l2_cache_mode>(0), n_cur, k_cur, 0, 0);
  } else {
    asc_set_gm2l1_nz_para(1, 1, BK, 0);
    asc_copy_gm2l1_nd2nz(bh, W_hi + ((int64_t)g * K + k0) * N + n0, N * 2, static_cast<asc_load_l2_cache_mode>(0), k_cur, n_cur, 0, 0);
    asc_copy_gm2l1_nd2nz(bl, W_lo + ((int64_t)g * K + k0) * N + n0, N * 2, static_cast<asc_load_l2_cache_mode>(0), k_cur, n_cur, 0, 0);
  }
  asc_sync_notify(PIPE_MTE2, PIPE_MTE1, (event_t)stg_id);
}

template <int BM, int BN, int BK, int TRANS_W, int HAS_BIAS>
__global__ __mix__(1, 2) void gmm_persistent_f32_split(
    __gm__ half* X_hi, __gm__ half* X_lo,
    __gm__ half* W_hi, __gm__ half* W_lo,
    __gm__ float* Bias, __gm__ float* Y,
    GmmGroupMeta meta, int E, int K, int N) {
  asc_init();
  constexpr int ROWS = BM / 2;
  constexpr int A_STAGE = BM * BK;
  constexpr int B_STAGE = TRANS_W ? (BN * BK) : (BK * BN);

  __cbuf__ half* a_hi_base = (__cbuf__ half*)0;
  __cbuf__ half* a_lo_base = (__cbuf__ half*)(2 * A_STAGE * 2);
  __cbuf__ half* b_hi_base = (__cbuf__ half*)(4 * A_STAGE * 2);
  __cbuf__ half* b_lo_base = (__cbuf__ half*)((4 * A_STAGE + 2 * B_STAGE) * 2);

  __cc__ float* c_l0c0 = (__cc__ float*)0;
  __cc__ float* c_l0c1 = (__cc__ float*)(BM * BN * 4);

  constexpr int C_UB_BYTES = ROWS * BN * 4;
  __ubuf__ float* c_ub0 = (__ubuf__ float*)0;
  __ubuf__ float* c_ub1 = (__ubuf__ float*)(C_UB_BYTES);
  __ubuf__ float* bias_f32_ub = (__ubuf__ float*)(2 * C_UB_BYTES);

  int n_tiles = (N + BN - 1) / BN;
  int total_tiles = 0;
  for (int g = 0; g < E; ++g) {
    int mg = meta.ends[g] - meta.starts[g];
    if (mg > 0) total_tiles += ((mg + BM - 1) / BM) * n_tiles;
  }
  int cid = (int)block_idx;
  int num_cores = 28;
  if (cid >= total_tiles) return;

  int k_iters = (K + BK - 1) / BK;

  if ASCEND_IS_AIC {
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, (event_t)0);
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, (event_t)1);
    asc_sync_notify(PIPE_FIX, PIPE_M, (event_t)0);
    asc_sync_notify(PIPE_FIX, PIPE_M, (event_t)1);
    uint32_t inflight = 0;
    int tile_seq = 0;

    for (int tid = cid; tid < total_tiles; tid += num_cores) {
      int rem = tid;
      int g = 0, mt = 0, nt = 0;
      for (; g < E; ++g) {
        int mg = meta.ends[g] - meta.starts[g];
        if (mg <= 0) continue;
        int m_blks = (mg + BM - 1) / BM;
        int g_tiles = m_blks * n_tiles;
        if (rem < g_tiles) { mt = rem / n_tiles; nt = rem % n_tiles; break; }
        rem -= g_tiles;
      }
      int m0 = meta.starts[g] + mt * BM;
      int m_cur = meta.ends[g] - m0; if (m_cur > BM) m_cur = BM;
      int n0 = nt * BN;
      int n_cur = N - n0; if (n_cur > BN) n_cur = BN;

      int buf = tile_seq & 1;
      __cc__ float* cur_l0c = buf ? c_l0c1 : c_l0c0;
      __ubuf__ float* cur_cub = buf ? c_ub1 : c_ub0;
      asc_sync_wait(PIPE_FIX, PIPE_M, (event_t)buf);

      if (k_iters > 0) {
        gmm_issue_f32_k_dma<BM, BN, BK, TRANS_W, A_STAGE, B_STAGE>(
            a_hi_base, a_lo_base, b_hi_base, b_lo_base,
            X_hi, X_lo, W_hi, W_lo, g, m0, m_cur, n0, n_cur, K, N, 0, 0);
      }
      for (int ki = 0; ki < k_iters; ++ki) {
        int stg = ki & 1;
        if (ki + 1 < k_iters) {
          gmm_issue_f32_k_dma<BM, BN, BK, TRANS_W, A_STAGE, B_STAGE>(
              a_hi_base, a_lo_base, b_hi_base, b_lo_base,
              X_hi, X_lo, W_hi, W_lo, g, m0, m_cur, n0, n_cur, K, N, ki + 1, (ki + 1) & 1);
        }
        __cbuf__ half* ah = a_hi_base + stg * A_STAGE;
        __cbuf__ half* al = a_lo_base + stg * A_STAGE;
        __cbuf__ half* bh = b_hi_base + stg * B_STAGE;
        __cbuf__ half* bl = b_lo_base + stg * B_STAGE;
        asc_sync_wait(PIPE_MTE2, PIPE_MTE1, (event_t)stg);

        if constexpr (TRANS_W) {
          ascend_gemm_l1<BM, BK, BN, 64, true, half>(cur_l0c, ah, bh, (ki == 0) ? 1 : 0, 0, 0);
          ascend_gemm_l1<BM, BK, BN, 64, true, half>(cur_l0c, ah, bl, 0, 0, 0);
          ascend_gemm_l1<BM, BK, BN, 64, true, half>(cur_l0c, al, bh, 0, 0, 0);
        } else {
          ascend_gemm_l1_nn<BM, BK, BN, 64, half, float>(cur_l0c, ah, bh, (ki == 0) ? 1 : 0, 0);
          ascend_gemm_l1_nn<BM, BK, BN, 64, half, float>(cur_l0c, ah, bl, 0, 0);
          ascend_gemm_l1_nn<BM, BK, BN, 64, half, float>(cur_l0c, al, bh, 0, 0);
        }
        asc_sync_notify(PIPE_MTE1, PIPE_MTE2, (event_t)stg);
      }

      asc_sync_notify(PIPE_M, PIPE_FIX, (event_t)buf);
      asc_sync_wait(PIPE_M, PIPE_FIX, (event_t)buf);
      if (inflight & (1u << buf)) {
        if (buf == 0) { asc_sync_intra_wait(PIPE_FIX, 6); asc_sync_intra_wait(PIPE_FIX, 22); }
        else { asc_sync_intra_wait(PIPE_FIX, 8); asc_sync_intra_wait(PIPE_FIX, 24); }
      }
      asc_set_l0c_copy_nz_para(1, 0, 0);
      asc_copy_l0c2ub(cur_cub, cur_l0c, BN, BM, BN, BM, 0,
                      static_cast<asc_dual_dst_mode>(1), static_cast<asc_unit_flag_mode>(0),
                      static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
      asc_sync_notify(PIPE_FIX, PIPE_M, (event_t)buf);
      if (buf == 0) { asc_sync_intra_arrive(PIPE_FIX, 3); asc_sync_intra_arrive(PIPE_FIX, 19); }
      else { asc_sync_intra_arrive(PIPE_FIX, 7); asc_sync_intra_arrive(PIPE_FIX, 23); }
      inflight |= (1u << buf);
      ++tile_seq;
    }

    if (inflight & 1u) { asc_sync_intra_wait(PIPE_FIX, 6); asc_sync_intra_wait(PIPE_FIX, 22); }
    if (inflight & 2u) { asc_sync_intra_wait(PIPE_FIX, 8); asc_sync_intra_wait(PIPE_FIX, 24); }
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, (event_t)0);
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, (event_t)1);
    asc_sync_wait(PIPE_FIX, PIPE_M, (event_t)0);
    asc_sync_wait(PIPE_FIX, PIPE_M, (event_t)1);
  }

  if ASCEND_IS_AIV {
    int sid = asc_get_sub_block_id();
    int done0 = (sid == 0) ? 6 : 22;
    int done1 = (sid == 0) ? 8 : 24;
    asc_sync_notify(PIPE_MTE3, PIPE_V, (event_t)0);
    asc_sync_notify(PIPE_MTE3, PIPE_V, (event_t)1);
    asc_sync_notify(PIPE_V, PIPE_MTE2, (event_t)0);
    asc_sync_notify(PIPE_V, PIPE_MTE2, (event_t)1);
    int tile_seq = 0;

    for (int tid = cid; tid < total_tiles; tid += num_cores) {
      int rem = tid;
      int g = 0, mt = 0, nt = 0;
      for (; g < E; ++g) {
        int mg = meta.ends[g] - meta.starts[g];
        if (mg <= 0) continue;
        int m_blks = (mg + BM - 1) / BM;
        int g_tiles = m_blks * n_tiles;
        if (rem < g_tiles) { mt = rem / n_tiles; nt = rem % n_tiles; break; }
        rem -= g_tiles;
      }
      int m0 = meta.starts[g] + mt * BM;
      int m_cur = meta.ends[g] - m0; if (m_cur > BM) m_cur = BM;
      int n0 = nt * BN;
      int n_cur = N - n0; if (n_cur > BN) n_cur = BN;

      int my_m_start = sid * ROWS;
      int my_rows = m_cur - my_m_start;
      if (my_rows > ROWS) my_rows = ROWS;
      if (my_rows < 0) my_rows = 0;

      int buf = tile_seq & 1;
      __ubuf__ float* cur_cub = buf ? c_ub1 : c_ub0;
      __ubuf__ float* cur_bf32 = buf ? (bias_f32_ub + 1024) : bias_f32_ub;

      if constexpr (HAS_BIAS) {
        asc_sync_wait(PIPE_V, PIPE_MTE2, (event_t)buf);
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)cur_bf32, (__gm__ uint8_t*)(Bias + (int64_t)g * N + n0),
                             1, n_cur * 4, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), n_cur * 4, n_cur * 4);
        asc_sync_notify(PIPE_MTE2, PIPE_V, (event_t)buf);
        asc_sync_wait(PIPE_MTE2, PIPE_V, (event_t)buf);
      }

      asc_sync_intra_wait(PIPE_V, buf ? 7 : 3);
      asc_sync_wait(PIPE_MTE3, PIPE_V, (event_t)buf);
      if (my_rows > 0) {
        vf_gmm_epilogue_f32<ROWS, BN, HAS_BIAS>(cur_cub, cur_bf32, my_rows);
      }
      if constexpr (HAS_BIAS) {
        asc_sync_notify(PIPE_V, PIPE_MTE2, (event_t)buf);
      }
      asc_sync_notify(PIPE_V, PIPE_MTE3, (event_t)buf);
      asc_sync_wait(PIPE_V, PIPE_MTE3, (event_t)buf);
      if (my_rows > 0) {
        asc_copy_ub2gm_align((__gm__ uint8_t*)(Y + ((int64_t)m0 + my_m_start) * N + n0), (__ubuf__ uint8_t*)cur_cub,
                             my_rows, n_cur * 4, static_cast<asc_store_l2_cache_mode>(4),
                             N * 4, BN * 4);
      }
      asc_sync_notify(PIPE_MTE3, PIPE_V, (event_t)buf);
      asc_sync_intra_arrive(PIPE_V, buf ? done1 : done0);
      ++tile_seq;
    }

    asc_sync_wait(PIPE_MTE3, PIPE_V, (event_t)0);
    asc_sync_wait(PIPE_MTE3, PIPE_V, (event_t)1);
    asc_sync_wait(PIPE_V, PIPE_MTE2, (event_t)0);
    asc_sync_wait(PIPE_V, PIPE_MTE2, (event_t)1);
  }
}

template <int BM, int BN>
__aicore__ inline void gmm_decode_tile(
    const GmmGroupMeta& meta, int E, int N, int n_tiles, int tid,
    int& g, int& m0, int& m_cur, int& n0, int& n_cur) {
  int rem = tid;
  g = 0;
  int mt = 0;
  int nt = 0;
  for (; g < E; ++g) {
    int mg = meta.ends[g] - meta.starts[g];
    if (mg <= 0) continue;
    int m_blks = (mg + BM - 1) / BM;
    int g_tiles = m_blks * n_tiles;
    if (rem < g_tiles) {
      mt = rem / n_tiles;
      nt = rem % n_tiles;
      break;
    }
    rem -= g_tiles;
  }
  m0 = meta.starts[g] + mt * BM;
  m_cur = meta.ends[g] - m0;
  if (m_cur > BM) m_cur = BM;
  n0 = nt * BN;
  n_cur = N - n0;
  if (n_cur > BN) n_cur = BN;
}

template <int BM, int BN, int BK, int TRANS_W, int QUANT_MODE, typename InT>
__global__ __cube__ void gmm_persistent_16_cube(
    __gm__ InT* X, __gm__ InT* W, __gm__ InT* Y,
    GmmGroupMeta meta, int E, int K, int N) {
  asc_init();
  constexpr int A_STAGE = BM * BK;
  constexpr int B_STAGE = TRANS_W ? (BN * BK) : (BK * BN);
  constexpr int SUB_K_STEP = (BK < 64) ? BK : 64;

  __cbuf__ InT* a_l1_base = (__cbuf__ InT*)0;
  __cbuf__ InT* b_l1_base = (__cbuf__ InT*)(2 * A_STAGE * 2);
  __cc__ float* c_l0c0 = (__cc__ float*)0;
  __cc__ float* c_l0c1 = (__cc__ float*)(BM * BN * 4);

  int n_tiles = (N + BN - 1) / BN;
  int total_tiles = 0;
  for (int g = 0; g < E; ++g) {
    int mg = meta.ends[g] - meta.starts[g];
    if (mg > 0) total_tiles += ((mg + BM - 1) / BM) * n_tiles;
  }
  int cid = (int)block_idx;
  int num_cores = 28;
  if (cid >= total_tiles) return;

  int k_iters = (K + BK - 1) / BK;
  asc_sync_notify(PIPE_MTE1, PIPE_MTE2, (event_t)0);
  asc_sync_notify(PIPE_MTE1, PIPE_MTE2, (event_t)1);
  asc_sync_notify(PIPE_FIX, PIPE_M, (event_t)0);
  asc_sync_notify(PIPE_FIX, PIPE_M, (event_t)1);
  int tile_seq = 0;

  for (int tid = cid; tid < total_tiles; tid += num_cores) {
    int g, m0, m_cur, n0, n_cur;
    gmm_decode_tile<BM, BN>(meta, E, N, n_tiles, tid, g, m0, m_cur, n0, n_cur);

    int buf = tile_seq & 1;
    __cc__ float* cur_l0c = buf ? c_l0c1 : c_l0c0;
    asc_sync_wait(PIPE_FIX, PIPE_M, (event_t)buf);

    if (k_iters > 0) {
      gmm_issue_k_dma<BM, BN, BK, TRANS_W, A_STAGE, B_STAGE, InT>(
          a_l1_base, b_l1_base, X, W, g, m0, m_cur, n0, n_cur, K, N, 0, 0);
    }
    for (int ki = 0; ki < k_iters; ++ki) {
      int stg = ki & 1;
      if (ki + 1 < k_iters) {
        gmm_issue_k_dma<BM, BN, BK, TRANS_W, A_STAGE, B_STAGE, InT>(
            a_l1_base, b_l1_base, X, W, g, m0, m_cur, n0, n_cur, K, N, ki + 1, (ki + 1) & 1);
      }
      __cbuf__ InT* a_dst = a_l1_base + stg * A_STAGE;
      __cbuf__ InT* b_dst = b_l1_base + stg * B_STAGE;
      asc_sync_wait(PIPE_MTE2, PIPE_MTE1, (event_t)stg);
      if constexpr (TRANS_W) {
        ascend_gemm_l1<BM, BK, BN, SUB_K_STEP, true, InT>(cur_l0c, a_dst, b_dst, (ki == 0) ? 1 : 0, 0, 0);
      } else {
        ascend_gemm_l1_nn<BM, BK, BN, SUB_K_STEP, InT, float>(cur_l0c, a_dst, b_dst, (ki == 0) ? 1 : 0, 0);
      }
      asc_sync_notify(PIPE_MTE1, PIPE_MTE2, (event_t)stg);
    }

    asc_sync_notify(PIPE_M, PIPE_FIX, (event_t)buf);
    asc_sync_wait(PIPE_M, PIPE_FIX, (event_t)buf);
    asc_set_l0c_copy_nz_para(1, 0, 0);
    asc_copy_l0c2gm(
        Y + (int64_t)m0 * N + n0, cur_l0c,
        (uint16_t)n_cur, (uint16_t)m_cur, (uint32_t)N, (uint16_t)BM,
        static_cast<asc_store_l2_cache_mode>(0), static_cast<asc_unit_flag_mode>(0),
        static_cast<asc_quant_mode>(QUANT_MODE), static_cast<asc_relu_pre_mode>(0),
        false, true, false, false);
    asc_sync_notify(PIPE_FIX, PIPE_M, (event_t)buf);
    ++tile_seq;
  }

  asc_sync_wait(PIPE_MTE1, PIPE_MTE2, (event_t)0);
  asc_sync_wait(PIPE_MTE1, PIPE_MTE2, (event_t)1);
  asc_sync_wait(PIPE_FIX, PIPE_M, (event_t)0);
  asc_sync_wait(PIPE_FIX, PIPE_M, (event_t)1);
}

template <int BM, int BN, int BK, int TRANS_W>
__global__ __cube__ void gmm_persistent_f32_cube(
    __gm__ half* X_hi, __gm__ half* X_lo, __gm__ half* W_hi, __gm__ half* W_lo,
    __gm__ float* Y, GmmGroupMeta meta, int E, int K, int N) {
  asc_init();
  constexpr int A_STAGE = BM * BK;
  constexpr int B_STAGE = TRANS_W ? (BN * BK) : (BK * BN);

  __cbuf__ half* a_hi_base = (__cbuf__ half*)0;
  __cbuf__ half* a_lo_base = (__cbuf__ half*)(2 * A_STAGE * 2);
  __cbuf__ half* b_hi_base = (__cbuf__ half*)(4 * A_STAGE * 2);
  __cbuf__ half* b_lo_base = (__cbuf__ half*)(4 * A_STAGE * 2 + 2 * B_STAGE * 2);
  __cc__ float* c_l0c0 = (__cc__ float*)0;
  __cc__ float* c_l0c1 = (__cc__ float*)(BM * BN * 4);

  int n_tiles = (N + BN - 1) / BN;
  int total_tiles = 0;
  for (int g = 0; g < E; ++g) {
    int mg = meta.ends[g] - meta.starts[g];
    if (mg > 0) total_tiles += ((mg + BM - 1) / BM) * n_tiles;
  }
  int cid = (int)block_idx;
  int num_cores = 28;
  if (cid >= total_tiles) return;

  int k_iters = (K + BK - 1) / BK;
  asc_sync_notify(PIPE_MTE1, PIPE_MTE2, (event_t)0);
  asc_sync_notify(PIPE_MTE1, PIPE_MTE2, (event_t)1);
  asc_sync_notify(PIPE_FIX, PIPE_M, (event_t)0);
  asc_sync_notify(PIPE_FIX, PIPE_M, (event_t)1);
  int tile_seq = 0;

  for (int tid = cid; tid < total_tiles; tid += num_cores) {
    int g, m0, m_cur, n0, n_cur;
    gmm_decode_tile<BM, BN>(meta, E, N, n_tiles, tid, g, m0, m_cur, n0, n_cur);

    int buf = tile_seq & 1;
    __cc__ float* cur_l0c = buf ? c_l0c1 : c_l0c0;
    asc_sync_wait(PIPE_FIX, PIPE_M, (event_t)buf);

    if (k_iters > 0) {
      gmm_issue_f32_k_dma<BM, BN, BK, TRANS_W, A_STAGE, B_STAGE>(
          a_hi_base, a_lo_base, b_hi_base, b_lo_base,
          X_hi, X_lo, W_hi, W_lo, g, m0, m_cur, n0, n_cur, K, N, 0, 0);
    }
    for (int ki = 0; ki < k_iters; ++ki) {
      int stg = ki & 1;
      if (ki + 1 < k_iters) {
        gmm_issue_f32_k_dma<BM, BN, BK, TRANS_W, A_STAGE, B_STAGE>(
            a_hi_base, a_lo_base, b_hi_base, b_lo_base,
            X_hi, X_lo, W_hi, W_lo, g, m0, m_cur, n0, n_cur, K, N, ki + 1, (ki + 1) & 1);
      }
      __cbuf__ half* ah = a_hi_base + stg * A_STAGE;
      __cbuf__ half* al = a_lo_base + stg * A_STAGE;
      __cbuf__ half* bh = b_hi_base + stg * B_STAGE;
      __cbuf__ half* bl = b_lo_base + stg * B_STAGE;
      asc_sync_wait(PIPE_MTE2, PIPE_MTE1, (event_t)stg);
      if constexpr (TRANS_W) {
        ascend_gemm_l1<BM, BK, BN, 64, true, half>(cur_l0c, ah, bh, (ki == 0) ? 1 : 0, 0, 0);
        ascend_gemm_l1<BM, BK, BN, 64, true, half>(cur_l0c, ah, bl, 0, 0, 0);
        ascend_gemm_l1<BM, BK, BN, 64, true, half>(cur_l0c, al, bh, 0, 0, 0);
      } else {
        ascend_gemm_l1_nn<BM, BK, BN, 64, half, float>(cur_l0c, ah, bh, (ki == 0) ? 1 : 0, 0);
        ascend_gemm_l1_nn<BM, BK, BN, 64, half, float>(cur_l0c, ah, bl, 0, 0);
        ascend_gemm_l1_nn<BM, BK, BN, 64, half, float>(cur_l0c, al, bh, 0, 0);
      }
      asc_sync_notify(PIPE_MTE1, PIPE_MTE2, (event_t)stg);
    }

    asc_sync_notify(PIPE_M, PIPE_FIX, (event_t)buf);
    asc_sync_wait(PIPE_M, PIPE_FIX, (event_t)buf);
    asc_set_l0c_copy_nz_para(1, 0, 0);
    asc_copy_l0c2gm(
        Y + (int64_t)m0 * N + n0, cur_l0c,
        (uint16_t)n_cur, (uint16_t)m_cur, (uint32_t)N, (uint16_t)BM,
        static_cast<asc_store_l2_cache_mode>(0), static_cast<asc_unit_flag_mode>(0),
        static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0),
        false, true, false, false);
    asc_sync_notify(PIPE_FIX, PIPE_M, (event_t)buf);
    ++tile_seq;
  }

  asc_sync_wait(PIPE_MTE1, PIPE_MTE2, (event_t)0);
  asc_sync_wait(PIPE_MTE1, PIPE_MTE2, (event_t)1);
  asc_sync_wait(PIPE_FIX, PIPE_M, (event_t)0);
  asc_sync_wait(PIPE_FIX, PIPE_M, (event_t)1);
}
"""


class _GmmGroupMeta(ctypes.Structure):
    _fields_ = [("starts", ctypes.c_int * 32), ("ends", ctypes.c_int * 32)]


def _render_gmm_trackb_launcher(
    trackb_src: str,
    *,
    bm: int,
    bn: int,
    bk: int,
    tl_dtype: str,
    bias_dtype: str,
    transpose_weight: bool,
    has_bias: bool,
) -> str:
    """Append concrete launch_gmm_* template instantiation to *trackb_src*."""
    trans_flag = 1 if transpose_weight else 0
    bias_flag = 1 if has_bias else 0

    if tl_dtype == "float32":
        kernel_call = (
            f"gmm_persistent_f32_split<{bm}, {bn}, {bk}, {trans_flag}, {bias_flag}><<<28, nullptr, stream>>>(\n"
            f"      (__gm__ half*)x_hi, (__gm__ half*)x_lo, (__gm__ half*)w_hi, (__gm__ half*)w_lo,\n"
            f"      (__gm__ float*)bias_ptr, (__gm__ float*)y_ptr, meta, E, K, N);"
            if has_bias
            else f"gmm_persistent_f32_cube<{bm}, {bn}, {bk}, {trans_flag}><<<28, nullptr, stream>>>(\n"
            f"      (__gm__ half*)x_hi, (__gm__ half*)x_lo, (__gm__ half*)w_hi, (__gm__ half*)w_lo,\n"
            f"      (__gm__ float*)y_ptr, meta, E, K, N);"
        )
        return (
            trackb_src
            + f"""
extern "C" int launch_gmm_f32(
    void* stream, void* x_ptr, void* w_ptr, void* bias_ptr, void* y_ptr,
    void* x_hi, void* x_lo, void* w_hi, void* w_lo,
    GmmGroupMeta meta, int E, int M, int K, int N) {{
  int x_elems = M * K;
  int w_elems = E * K * N;
  split_f32_to_hilo_pair_kernel<<<56, nullptr, stream>>>(
      (__gm__ float*)x_ptr, (__gm__ half*)x_hi, (__gm__ half*)x_lo, x_elems,
      (__gm__ float*)w_ptr, (__gm__ half*)w_hi, (__gm__ half*)w_lo, w_elems);
  {kernel_call}
  return 0;
}}
"""
        )
    c_in = "bfloat16_t" if tl_dtype == "bfloat16" else "half"
    quant_mode = 16 if tl_dtype == "bfloat16" else 1
    if bias_dtype == "float32":
        c_bias = "float"
    elif bias_dtype == "bfloat16":
        c_bias = "bfloat16_t"
    else:
        c_bias = "half"
    kernel_call = (
        f"gmm_persistent_16<{bm}, {bn}, {bk}, {trans_flag}, {bias_flag}, {c_in}, {c_bias}><<<28, nullptr, stream>>>(\n"
        f"      (__gm__ {c_in}*)x_ptr, (__gm__ {c_in}*)w_ptr, (__gm__ {c_bias}*)bias_ptr,\n"
        f"      (__gm__ {c_in}*)y_ptr, meta, E, K, N);"
        if has_bias
        else f"gmm_persistent_16_cube<{bm}, {bn}, {bk}, {trans_flag}, {quant_mode}, {c_in}><<<28, nullptr, stream>>>(\n"
        f"      (__gm__ {c_in}*)x_ptr, (__gm__ {c_in}*)w_ptr, (__gm__ {c_in}*)y_ptr, meta, E, K, N);"
    )
    return (
        trackb_src
        + f"""
extern "C" int launch_gmm_16(
    void* stream, void* x_ptr, void* w_ptr, void* bias_ptr, void* y_ptr,
    GmmGroupMeta meta, int E, int K, int N) {{
  {kernel_call}
  return 0;
}}
"""
    )


def _render_trackb_tu(**dims: object) -> str:
    """Render an instantiated C++20 Track-B translation unit for grouped_matmul."""
    raw_src = dims.get("trackb_src")
    trackb_src = str(raw_src) if raw_src is not None else _GMM_TRACKB_SRC
    raw_dt = str(dims.get("tl_dtype", dims.get("dtype", "float16")) or "float16").strip()
    tl_dtype = raw_dt if raw_dt in ("float16", "bfloat16", "float32") else "float16"
    raw_bdt = str(dims.get("bias_dtype", tl_dtype) or tl_dtype).strip()
    bias_dtype = raw_bdt if raw_bdt in ("float16", "bfloat16", "float32") else tl_dtype
    k_dim = int(dims.get("k_dim", dims.get("K", 256)) or 256)
    n_dim = int(dims.get("n_dim", dims.get("N", 512)) or 512)
    m_dim = int(dims.get("m_dim", dims.get("M", 128)) or 128)
    transpose_weight = bool(dims.get("transpose_weight", False))
    has_bias = bool(dims.get("has_bias", False))
    cfg = _select_grouped_matmul_config(
        group_sizes=(m_dim,),
        k_dim=k_dim,
        n_dim=n_dim,
        tl_dtype=tl_dtype,
        transpose_weight=transpose_weight,
        has_bias=has_bias,
    )
    bm = int(dims.get("bm", cfg["bm"]) or cfg["bm"])
    bn = int(dims.get("bn", cfg["bn"]) or cfg["bn"])
    bk = int(dims.get("bk", cfg["bk"]) or cfg["bk"])
    return _render_gmm_trackb_launcher(
        trackb_src,
        bm=bm,
        bn=bn,
        bk=bk,
        tl_dtype=tl_dtype,
        bias_dtype=bias_dtype,
        transpose_weight=transpose_weight,
        has_bias=has_bias,
    )


def _build_gmm_trackb_callable(
    *,
    bm: int,
    bn: int,
    bk: int,
    tl_dtype: str,
    bias_dtype: str,
    transpose_weight: bool,
    has_bias: bool,
):
    key = ("gmm_trackb", bm, bn, bk, tl_dtype, bias_dtype, bool(transpose_weight), bool(has_bias))
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    launcher = _render_gmm_trackb_launcher(
        _GMM_TRACKB_SRC,
        bm=bm,
        bn=bn,
        bk=bk,
        tl_dtype=tl_dtype,
        bias_dtype=bias_dtype,
        transpose_weight=transpose_weight,
        has_bias=has_bias,
    )

    tl_inc = str(Path(tilelang.__file__).resolve().parent.parent / "src")
    asc_home = os.environ.get("ASCEND_HOME_PATH", "")
    opts = [f"-I{tl_inc}"]
    if asc_home:
        for sub in ("include", "x86_64-linux/include", "aarch64-linux/include", "runtime/include"):
            asc_inc = os.path.join(asc_home, sub)
            if os.path.isdir(asc_inc) and f"-I{asc_inc}" not in opts:
                opts.append(f"-I{asc_inc}")
    opts.extend(PURE_SIMDVF_UBUF_FLAGS)
    so_dir = tempfile.mkdtemp(prefix="gmm_trackb_")
    so_path = os.path.join(so_dir, "gmm_trackb.so")
    compile_ascend(
        launcher,
        target_format="so",
        npu_arch="dav-3510",
        options=opts,
        path_target=so_path,
    )
    lib = ctypes.CDLL(so_path)
    if tl_dtype == "float32":
        fn = lib.launch_gmm_f32
        fn.argtypes = (
            [ctypes.c_void_p] * 9
            + [_GmmGroupMeta, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        )

        def _run(x_c, w_c, bias_c, y_c, starts, ends):
            e_dim = len(starts)
            m_dim, k_dim = int(x_c.shape[0]), int(x_c.shape[1])
            n_dim = int(y_c.shape[1])
            x_hi = torch.empty_like(x_c, dtype=torch.float16)
            x_lo = torch.empty_like(x_c, dtype=torch.float16)
            w_hi = torch.empty_like(w_c, dtype=torch.float16)
            w_lo = torch.empty_like(w_c, dtype=torch.float16)
            stream_ptr = torch.npu.current_stream().npu_stream
            for off in range(0, e_dim, 32):
                sub_e = min(32, e_dim - off)
                meta = _GmmGroupMeta()
                for idx in range(sub_e):
                    meta.starts[idx] = int(starts[off + idx])
                    meta.ends[idx] = int(ends[off + idx])
                b_ptr = (
                    bias_c[off:].data_ptr()
                    if (has_bias and bias_c is not None)
                    else y_c.data_ptr()
                )
                fn(
                    stream_ptr,
                    x_c.data_ptr(),
                    w_c[off:].data_ptr(),
                    b_ptr,
                    y_c.data_ptr(),
                    x_hi.data_ptr(),
                    x_lo.data_ptr(),
                    w_hi[off:].data_ptr(),
                    w_lo[off:].data_ptr(),
                    meta,
                    sub_e,
                    m_dim,
                    k_dim,
                    n_dim,
                )

    else:
        fn = lib.launch_gmm_16
        fn.argtypes = [ctypes.c_void_p] * 5 + [
            _GmmGroupMeta,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]

        def _run(x_c, w_c, bias_c, y_c, starts, ends):
            e_dim = len(starts)
            k_dim = int(x_c.shape[1])
            n_dim = int(y_c.shape[1])
            stream_ptr = torch.npu.current_stream().npu_stream
            for off in range(0, e_dim, 32):
                sub_e = min(32, e_dim - off)
                meta = _GmmGroupMeta()
                for idx in range(sub_e):
                    meta.starts[idx] = int(starts[off + idx])
                    meta.ends[idx] = int(ends[off + idx])
                b_ptr = (
                    bias_c[off:].data_ptr()
                    if (has_bias and bias_c is not None)
                    else y_c.data_ptr()
                )
                fn(
                    stream_ptr,
                    x_c.data_ptr(),
                    w_c[off:].data_ptr(),
                    b_ptr,
                    y_c.data_ptr(),
                    meta,
                    sub_e,
                    k_dim,
                    n_dim,
                )

    _run._so_lib = lib
    _KERNEL_CACHE[key] = _run
    return _run


def _dtype_name(dt: torch.dtype) -> str:
    if dt == torch.bfloat16:
        return "bfloat16"
    if dt == torch.float32:
        return "float32"
    return "float16"


def grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    group_list: Optional[Sequence[int] | torch.Tensor] = None,
    split_item: int = 0,
    transpose_weight: bool = False,
    *,
    schedule_mode: str = "tuned",
) -> list[torch.Tensor]:
    """Grouped matrix multiplication (cumsum group_list) on Ascend950PR."""
    if group_list is None:
        raise ValueError("group_list is required for grouped_matmul")
    if isinstance(group_list, torch.Tensor):
        ends = [int(v) for v in group_list.tolist()]
    else:
        ends = [int(v) for v in group_list]

    e_dim = int(weight.shape[0])
    if len(ends) != e_dim:
        raise ValueError(f"group_list length {len(ends)} != weight.shape[0] {e_dim}")

    starts: list[int] = []
    group_sizes: list[int] = []
    prev = 0
    for end_val in ends:
        starts.append(prev)
        group_sizes.append(max(0, end_val - prev))
        prev = end_val

    m_dim, k_dim = int(x.shape[0]), int(x.shape[1])
    n_dim = int(weight.shape[1] if transpose_weight else weight.shape[2])

    x_c = x.contiguous()
    w_c = weight.contiguous()
    bias_c = bias.contiguous() if bias is not None else None
    y_full = torch.empty((m_dim, n_dim), dtype=x.dtype, device=x.device)

    if m_dim > 0 and n_dim > 0 and k_dim > 0 and any(sz > 0 for sz in group_sizes):
        tl_dtype = _dtype_name(x.dtype)
        bias_dtype = _dtype_name(bias.dtype) if bias is not None else tl_dtype
        cfg = _select_grouped_matmul_config(
            group_sizes=tuple(group_sizes),
            k_dim=k_dim,
            n_dim=n_dim,
            tl_dtype=tl_dtype,
            transpose_weight=bool(transpose_weight),
            has_bias=bias is not None,
            schedule_mode=schedule_mode,
        )
        runner = _build_gmm_trackb_callable(
            bm=int(cfg["bm"]),
            bn=int(cfg["bn"]),
            bk=int(cfg["bk"]),
            tl_dtype=tl_dtype,
            bias_dtype=bias_dtype,
            transpose_weight=bool(transpose_weight),
            has_bias=bias is not None,
        )
        runner(x_c, w_c, bias_c, y_full, starts, ends)

    if int(split_item) in (2, 3):
        return [y_full]
    return list(y_full.split(group_sizes, dim=0))
