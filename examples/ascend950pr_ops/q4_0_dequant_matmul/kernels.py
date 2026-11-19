"""Ascend 950PR high-performance Q4_0 dequant-matmul kernel package.

Dual-regime Track-B AscendC architecture compiled dynamically via ``bisheng``:
1. GEMV / Tiny-M regime (M <= 8, 56 AIVs): ``vlds_unpack4`` int4 extraction + SIMD dot.
2. MixCore GEMM regime (M > 8, 28 MixCores): 2 AIV sub-blocks dequantize into NZ L1 tiles
   while AIC executes ``ascend_gemm_l1`` FP32 L0C accumulation and fixpipe epilogue.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import tilelang
import torch

_TRACK_B_LIB_CACHE: dict[str, ctypes.CDLL] = {}

_Q4_0_ASC_SRC = r'''
#include "tl_templates/ascend/common.h"
#include "tl_templates/ascend/gemm.h"
#include "tl_templates/ascend/nd2nz_copy.h"

namespace V = simd_inst;
using namespace AscendC;

__simd_callee__ inline vector_bool make_mask_b32(uint32_t n) { return asc_update_mask_b32(n); }
__simd_callee__ inline vector_bool make_mask_b16(uint32_t n) { return asc_update_mask_b16(n); }

template <typename T>
__simd_callee__ inline vector_f32 load_f32(__ubuf__ T* src, int32_t off) {
  if constexpr (std::is_same_v<T, float>) {
    return V::vlds_norm(src + off, 0);
  } else {
    auto m16 = asc_create_mask_b16(PAT_ALL);
    auto raw = V::vlds_unpack(src + off, 0);
    return V::vcvt<float>(raw, m16, PART_EVEN, MODE_ZEROING);
  }
}

template <typename T>
__simd_callee__ inline void store_f32(__ubuf__ T* dst, int32_t off, vector_f32 val,
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

__simd_callee__ inline vector_f32 int32_to_f32(vector_int32_t src, vector_bool mask) {
  vector_float dst;
  asc_cast(dst, src, mask, std::integral_constant<asc_cast_round_mode, asc_cast_round_mode::RN>{});
  return dst;
}

template <typename T>
__simd_vf__ inline void q4_gemv_vf(
    __ubuf__ T* x_in, __ubuf__ uint8_t* qw_in, __ubuf__ half* sc_in,
    __ubuf__ float* x_ev, __ubuf__ float* x_od, __ubuf__ float* sc_f,
    __ubuf__ float* sc_exp, __ubuf__ float* out_f, __ubuf__ float* tmp_sum,
    int32_t M, int32_t K, int32_t n_cur, int32_t row_stride_bytes, int32_t sc_stride) {
  auto m32 = asc_create_mask_b32(PAT_ALL);
  auto m16 = asc_create_mask_b16(PAT_ALL);
  auto m_b16 = make_mask_b32(16);
  auto v_zero = V::vdup<float>(0.0f, m32, MODE_ZEROING);
  auto v_eight = V::vdup<float>(8.0f, m32, MODE_ZEROING);
  auto v_mask0f = V::vdup<uint32_t>(0x0Fu, m32, MODE_ZEROING);

  int32_t k_blks = K / 32;
  int32_t k_half = K / 2;
  int32_t k_half_pad = ((k_half + 63) / 64) * 64;
  int32_t k_chunks = (k_half + 63) / 64;
  int32_t k_tail = k_half - (k_chunks - 1) * 64;
  int32_t k_pad = ((K + 63) / 64) * 64;

  for (int32_t m = 0; m < M; ++m) {
    for (int32_t c = 0; c < k_chunks; ++c) {
      V::vsts_norm(v_zero, x_ev + m * k_half_pad + c * 64, 0, m32);
      V::vsts_norm(v_zero, x_od + m * k_half_pad + c * 64, 0, m32);
    }
  }
  asc_mem_bar(VST_VLD);

  for (int32_t m = 0; m < M; ++m) {
    for (int32_t c = 0; c < k_pad / 64; ++c) {
      V::vsts_norm(load_f32(x_in, m * k_pad + c * 64), out_f + m * k_pad + c * 64, 0, m32);
    }
  }
  asc_mem_bar(VST_VLD);
  for (int32_t m = 0; m < M; ++m) {
    for (int32_t b = 0; b < k_blks; ++b) {
      auto ev = V::vlds_norm(out_f + m * k_pad + b * 32, 0);
      auto od = V::vlds_norm(out_f + m * k_pad + b * 32 + 16, 0);
      V::vsts_norm(ev, x_ev + m * k_half_pad + b * 16, 0, m_b16);
      V::vsts_norm(od, x_od + m * k_half_pad + b * 16, 0, m_b16);
    }
  }
  asc_mem_bar(VST_VLD);

  for (int32_t n = 0; n < n_cur; ++n) {
    auto sv = V::vcvt<float>(V::vlds_unpack(sc_in + n * sc_stride, 0), m16, PART_EVEN, MODE_ZEROING);
    V::vsts_norm(sv, sc_f + n * 64, 0, m32);
  }
  asc_mem_bar(VST_VLD);

  for (int32_t n = 0; n < n_cur; ++n) {
    for (int32_t b = 0; b < k_blks; ++b) {
      V::vsts_norm(V::vlds_brc_elem(sc_f + n * 64 + b, 0), sc_exp + b * 16, 0, m_b16);
    }
    asc_mem_bar(VST_VLD);

    for (int32_t m = 0; m < M; ++m) {
      auto acc = v_zero;
      for (int32_t c = 0; c < k_chunks; ++c) {
        int32_t w = (c == k_chunks - 1) ? k_tail : 64;
        auto mc = (w == 64) ? m32 : make_mask_b32((uint32_t)w);
        auto raw_u8 = V::vlds_unpack4(qw_in + n * row_stride_bytes + c * 64, 0);
        auto raw_u32 = reinterpret_cast<vector_u32&>(raw_u8);
        auto lo_u = V::vand(raw_u32, v_mask0f, mc, MODE_ZEROING);
        auto hi_u = V::vshrs(raw_u32, (uint32_t)4, mc, MODE_ZEROING);
        auto lo_f = V::vsub(int32_to_f32(reinterpret_cast<vector_int32_t&>(lo_u), mc), v_eight, mc, MODE_ZEROING);
        auto hi_f = V::vsub(int32_to_f32(reinterpret_cast<vector_int32_t&>(hi_u), mc), v_eight, mc, MODE_ZEROING);
        auto sc_v = V::vlds_norm(sc_exp + c * 64, 0);
        auto xe_v = V::vlds_norm(x_ev + m * k_half_pad + c * 64, 0);
        auto xo_v = V::vlds_norm(x_od + m * k_half_pad + c * 64, 0);
        auto dot_v = V::vadd(V::vmul(lo_f, xe_v, mc, MODE_ZEROING), V::vmul(hi_f, xo_v, mc, MODE_ZEROING), mc, MODE_ZEROING);
        acc = V::vadd(acc, V::vmul(dot_v, sc_v, mc, MODE_ZEROING), m32, MODE_ZEROING);
      }
      auto sum_v = V::vdupv(V::vcadd(acc, m32, MODE_ZEROING), m32, POS_LOWEST, MODE_ZEROING);
      V::vsts_norm(sum_v, tmp_sum + (m * 32 + n) * 64, 0, m32);
    }
  }
  asc_mem_bar(VST_VLD);
}

template <typename T>
__simd_vf__ inline void q4_gemv_cast_vf(__ubuf__ float* out_f, __ubuf__ T* out_ub, int32_t M, int32_t n_cur) {
  auto mn = make_mask_b32((uint32_t)n_cur);
  auto mn16 = make_mask_b16((uint32_t)(n_cur * 2));
  for (int32_t m = 0; m < M; ++m) {
    store_f32(out_ub, m * 64, V::vlds_norm(out_f + m * 64, 0), mn, mn16);
  }
  asc_mem_bar(VST_VLD);
}

template <typename T>
__aicore__ inline void q4_gemv_impl(
    __gm__ T* x, __gm__ uint8_t* qw, __gm__ half* sc, __gm__ T* y,
    int32_t M, int32_t N, int32_t K) {
  asc_init();
  __ubuf__ uint8_t* ub = (__ubuf__ uint8_t*)0;
  __ubuf__ T* x_in = (__ubuf__ T*)(ub + 0);
  __ubuf__ uint8_t* qw_in = (__ubuf__ uint8_t*)(ub + 16384);
  __ubuf__ half* sc_in = (__ubuf__ half*)(ub + 32768);
  __ubuf__ float* x_ev = (__ubuf__ float*)(ub + 40960);
  __ubuf__ float* x_od = (__ubuf__ float*)(ub + 49152);
  __ubuf__ float* sc_f = (__ubuf__ float*)(ub + 57344);
  __ubuf__ float* sc_exp = (__ubuf__ float*)(ub + 65536);
  __ubuf__ float* out_f = (__ubuf__ float*)(ub + 69632);
  __ubuf__ T* out_ub = (__ubuf__ T*)(ub + 90112);
  __ubuf__ float* tmp_sum = (__ubuf__ float*)(ub + 98304);

  int32_t k_pad = ((K + 63) / 64) * 64;
  int32_t k_blks = K / 32;
  int32_t row_bytes = K / 2;
  int32_t row_stride_bytes = ((row_bytes + 63) / 64) * 64;
  int32_t sc_stride = 64;

  asc_copy_gm2ub_align((__ubuf__ uint8_t*)x_in, (__gm__ uint8_t*)x,
                       (uint32_t)M, (uint32_t)(K * sizeof(T)), 0, 0, false,
                       asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                       (uint64_t)(K * sizeof(T)), (uint32_t)(k_pad * sizeof(T)));

  int32_t n_tile = (N <= 56) ? 1 : ((N + 55) / 56);
  if (n_tile > 32) n_tile = 32;
  int32_t total_tasks = (N + n_tile - 1) / n_tile;
  int32_t ncores = (int32_t)block_num;
  int32_t cid = (int32_t)block_idx;
  int32_t base = total_tasks / ncores;
  int32_t rem = total_tasks % ncores;
  int32_t start = cid * base + (cid < rem ? cid : rem);
  int32_t count = base + (cid < rem ? 1 : 0);

  for (int32_t lt = 0; lt < count; ++lt) {
    int32_t tid = start + lt;
    int32_t n0 = tid * n_tile;
    int32_t n_cur = (N - n0 < n_tile) ? (N - n0) : n_tile;
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)qw_in, (__gm__ uint8_t*)(qw + (int64_t)n0 * row_bytes),
                         (uint32_t)n_cur, (uint32_t)row_bytes, 0, 0, false,
                         asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                         (uint64_t)row_bytes, (uint32_t)row_stride_bytes);
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)sc_in, (__gm__ uint8_t*)(sc + (int64_t)n0 * k_blks),
                         (uint32_t)n_cur, (uint32_t)(k_blks * 2), 0, 0, false,
                         asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                         (uint64_t)(k_blks * 2), (uint32_t)(sc_stride * 2));
    asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID0);

    q4_gemv_vf<T>(x_in, qw_in, sc_in, x_ev, x_od, sc_f, sc_exp, out_f, tmp_sum,
                  M, K, n_cur, row_stride_bytes, sc_stride);
    asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID1);
    asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID1);
    for (int32_t m = 0; m < M; ++m) {
      for (int32_t n = 0; n < n_cur; ++n) {
        out_f[m * 64 + n] = tmp_sum[(m * 32 + n) * 64];
      }
    }
    asc_sync_notify(PIPE_S, PIPE_V, EVENT_ID1);
    asc_sync_wait(PIPE_S, PIPE_V, EVENT_ID1);
    q4_gemv_cast_vf<T>(out_f, out_ub, M, n_cur);

    asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_copy_ub2gm_align((__gm__ uint8_t*)(y + n0), (__ubuf__ uint8_t*)out_ub,
                         (uint32_t)M, (uint32_t)(n_cur * sizeof(T)),
                         asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                         (uint64_t)(N * sizeof(T)), (uint32_t)(64 * sizeof(T)));
    if (lt + 1 < count) {
      asc_sync_notify(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      asc_sync_wait(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
  }
  asc_sync();
}

template <int MT, int NT, int KT_PAD>
__simd_vf__ inline void q4_dequant_tile_vf(
    __ubuf__ uint8_t* qw_in, __ubuf__ half* sc_in, __ubuf__ float* w_f32, __ubuf__ float* w_nz,
    int32_t n_cur, int32_t k_cur, int32_t row_stride_bytes, int32_t sc_stride) {
  auto m32 = asc_create_mask_b32(PAT_ALL);
  auto m16 = asc_create_mask_b16(PAT_ALL);
  auto m_b16 = make_mask_b32(16);
  auto v_zero = V::vdup<float>(0.0f, m32, MODE_ZEROING);
  auto v_eight = V::vdup<float>(8.0f, m32, MODE_ZEROING);
  auto v_mask0f = V::vdup<uint32_t>(0x0Fu, m32, MODE_ZEROING);
  constexpr int32_t N_HALF = NT / 2;
  int32_t k_blks = k_cur / 32;
  int32_t qw_chunks = (k_cur / 2 + 63) / 64;
  int32_t qw_tail = (k_cur / 2) - (qw_chunks - 1) * 64;

  for (int32_t n = 0; n < N_HALF; ++n) {
    for (int32_t c = 0; c < (KT_PAD + 63) / 64; ++c) {
      V::vsts_norm(v_zero, w_f32 + n * KT_PAD + c * 64, 0, m32);
    }
  }
  __ubuf__ float* sc_f = w_f32 + N_HALF * KT_PAD + 64;
  __ubuf__ float* tmp_lo = sc_f + N_HALF * 64;
  __ubuf__ float* tmp_hi = tmp_lo + 64;
  for (int32_t n = 0; n < n_cur; ++n) {
    auto sv = V::vcvt<float>(V::vlds_unpack(sc_in + n * sc_stride, 0), m16, PART_EVEN, MODE_ZEROING);
    V::vsts_norm(sv, sc_f + n * 64, 0, m32);
  }
  asc_mem_bar(VST_VLD);

  for (int32_t n = 0; n < n_cur; ++n) {
    for (int32_t c = 0; c < qw_chunks; ++c) {
      int32_t w = (c == qw_chunks - 1) ? qw_tail : 64;
      auto mc = (w == 64) ? m32 : make_mask_b32((uint32_t)w);
      auto raw_u8 = V::vlds_unpack4(qw_in + n * row_stride_bytes + c * 64, 0);
      auto raw_u32 = reinterpret_cast<vector_u32&>(raw_u8);
      auto lo_u = V::vand(raw_u32, v_mask0f, mc, MODE_ZEROING);
      auto hi_u = V::vshrs(raw_u32, (uint32_t)4, mc, MODE_ZEROING);
      auto lo_f = V::vsub(int32_to_f32(reinterpret_cast<vector_int32_t&>(lo_u), mc), v_eight, mc, MODE_ZEROING);
      auto hi_f = V::vsub(int32_to_f32(reinterpret_cast<vector_int32_t&>(hi_u), mc), v_eight, mc, MODE_ZEROING);
      V::vsts_norm(lo_f, tmp_lo, 0, m32);
      V::vsts_norm(hi_f, tmp_hi, 0, m32);
      asc_mem_bar(VST_VLD);
      int32_t b_base = c * 4;
      int32_t b_cnt = (k_blks - b_base < 4) ? (k_blks - b_base) : 4;
      for (int32_t jb = 0; jb < b_cnt; ++jb) {
        int32_t b = b_base + jb;
        auto sb = V::vlds_brc_elem(sc_f + n * 64 + b, 0);
        auto lv = V::vmul(V::vlds_norm(tmp_lo + jb * 16, 0), sb, m_b16, MODE_ZEROING);
        auto hv = V::vmul(V::vlds_norm(tmp_hi + jb * 16, 0), sb, m_b16, MODE_ZEROING);
        V::vsts_norm(lv, w_f32 + n * KT_PAD + b * 32, 0, m_b16);
        V::vsts_norm(hv, w_f32 + n * KT_PAD + b * 32 + 16, 0, m_b16);
      }
      asc_mem_bar(VST_VLD);
    }
  }
  ascend_nd2nz_scatter_impl<N_HALF, KT_PAD, N_HALF, float, float>::apply(w_f32, w_nz);
}

template <int M_HALF, int KT_PAD, typename T_io>
__simd_vf__ inline void q4_x_cast_scatter_vf(
    __ubuf__ T_io* x_in, __ubuf__ float* x_f32, __ubuf__ float* x_nz,
    int32_t m_cur, int32_t k_cur, int32_t x_stride) {
  auto m32 = asc_create_mask_b32(PAT_ALL);
  auto v_zero = V::vdup<float>(0.0f, m32, MODE_ZEROING);
  for (int32_t m = 0; m < M_HALF; ++m) {
    for (int32_t c = 0; c < (KT_PAD + 63) / 64; ++c) {
      V::vsts_norm(v_zero, x_f32 + m * KT_PAD + c * 64, 0, m32);
    }
  }
  asc_mem_bar(VST_VLD);
  int32_t chunks = (k_cur + 63) / 64;
  int32_t tail = k_cur - (chunks - 1) * 64;
  for (int32_t m = 0; m < m_cur; ++m) {
    for (int32_t c = 0; c < chunks; ++c) {
      int32_t w = (c == chunks - 1) ? tail : 64;
      auto mc = (w == 64) ? m32 : make_mask_b32((uint32_t)w);
      V::vsts_norm(load_f32(x_in, m * x_stride + c * 64), x_f32 + m * KT_PAD + c * 64, 0, mc);
    }
  }
  asc_mem_bar(VST_VLD);
  ascend_nd2nz_scatter_impl<M_HALF, KT_PAD, M_HALF, float, float>::apply(x_f32, x_nz);
}

template <int MT, typename T_io>
__simd_vf__ inline void q4_cast_y_vf(__ubuf__ float* y_f, __ubuf__ T_io* y_out, int32_t m_cur, int32_t n_cur, int32_t n_tile) {
  auto m32 = asc_create_mask_b32(PAT_ALL);
  auto m16 = asc_create_mask_b16(PAT_ALL);
  auto mn = (n_cur == 64) ? m32 : make_mask_b32((uint32_t)n_cur);
  auto mn16 = (n_cur == 64) ? m16 : make_mask_b16((uint32_t)(n_cur * 2));
  for (int32_t m = 0; m < m_cur; ++m) {
    store_f32(y_out, m * n_tile, V::vlds_norm(y_f + m * n_tile, 0), mn, mn16);
  }
  asc_mem_bar(VST_VLD);
}

template <int MT, int NT, int KT_PAD, typename T_io>
__aicore__ inline void q4_mix_gemm_impl(
    __gm__ T_io* x, __gm__ uint8_t* qw, __gm__ half* sc, __gm__ T_io* y,
    int32_t M, int32_t N, int32_t K) {
  asc_init();
  constexpr int32_t M_HALF = MT / 2;
  constexpr int32_t N_HALF = NT / 2;
  constexpr int32_t C0 = 8; // float C0 = 32 / 4 = 8
  constexpr int32_t TILE_K_SUB = (KT_PAD > 64) ? 64 : KT_PAD;
  __cbuf__ float* l1_X = (__cbuf__ float*)0;
  __cbuf__ float* l1_W = (__cbuf__ float*)65536;
  __cc__ float* l0c = (__cc__ float*)0;
  __ubuf__ uint8_t* ub = (__ubuf__ uint8_t*)0;
  __ubuf__ uint8_t* qw_in = (__ubuf__ uint8_t*)(ub + 0);
  __ubuf__ half* sc_in = (__ubuf__ half*)(ub + 8192);
  __ubuf__ float* w_f32 = (__ubuf__ float*)(ub + 16384);
  __ubuf__ float* w_nz = (__ubuf__ float*)(ub + 49152);
  __ubuf__ T_io* x_in = (__ubuf__ T_io*)(ub + 65536);
  __ubuf__ float* x_f32 = (__ubuf__ float*)(ub + 73728);
  __ubuf__ float* x_nz = (__ubuf__ float*)(ub + 81920);
  __ubuf__ float* y_f = (__ubuf__ float*)(ub + 106496);
  __ubuf__ T_io* y_out = (__ubuf__ T_io*)(ub + 122880);

  int32_t m_tiles = (M + MT - 1) / MT;
  int32_t n_tiles = (N + NT - 1) / NT;
  int32_t k_iters = (K + KT_PAD - 1) / KT_PAD;
  int32_t total_tasks = m_tiles * n_tiles;
  int32_t ncores = (int32_t)block_num;
  int32_t cid = (int32_t)block_idx;
  int32_t base = total_tasks / ncores;
  int32_t rem = total_tasks % ncores;
  int32_t start = cid * base + (cid < rem ? cid : rem);
  int32_t count = base + (cid < rem ? 1 : 0);

  if ASC_IS_AIC {
    for (int32_t lt = 0; lt < count; ++lt) {
      for (int32_t ki = 0; ki < k_iters; ++ki) {
        asc_sync_intra_wait(PIPE_MTE1, 0);
        asc_sync_intra_wait(PIPE_MTE1, 16);
        ascend_gemm_l1<MT, KT_PAD, NT, TILE_K_SUB, true, float>(l0c, l1_X, l1_W, (ki == 0) ? 1 : 0, 0, 0);
        asc_sync_intra_arrive(PIPE_MTE1, 3);
        asc_sync_intra_arrive(PIPE_MTE1, 19);
      }
      asc_sync_intra_wait(PIPE_FIX, 2);
      asc_sync_intra_wait(PIPE_FIX, 18);
      asc_sync_notify(PIPE_M, PIPE_FIX, EVENT_ID0);
      asc_sync_wait(PIPE_M, PIPE_FIX, EVENT_ID0);
      asc_set_l0c_copy_nz_para(1, 0, 0);
      asc_copy_l0c2ub(y_f, l0c, (uint16_t)NT, (uint16_t)MT, (uint32_t)NT, (uint16_t)MT, 0,
                      asc_dual_dst_mode::DUAL_DST_SPLIT_M, asc_unit_flag_mode::DISABLE,
                      static_cast<asc_quant_mode>(0), asc_relu_pre_mode::NONE, 0, 1, 0, 0);
      asc_sync_intra_arrive(PIPE_FIX, 1);
      asc_sync_intra_arrive(PIPE_FIX, 17);
    }
  }

  if ASC_IS_AIV {
    int32_t sid = (int32_t)asc_get_sub_block_id();
    int32_t row_bytes = K / 2;
    int32_t k_blks = K / 32;
    for (int32_t lt = 0; lt < count; ++lt) {
      int32_t tid = start + lt;
      int32_t mt_idx = tid / n_tiles;
      int32_t nt_idx = tid % n_tiles;
      int32_t m0 = mt_idx * MT;
      int32_t m_cur = (M - m0 < MT) ? (M - m0) : MT;
      int32_t m_sid_off = sid * M_HALF;
      int32_t m_sid_cur = (m_cur - m_sid_off > 0) ? ((m_cur - m_sid_off < M_HALF) ? (m_cur - m_sid_off) : M_HALF) : 0;
      int32_t n0 = nt_idx * NT;
      int32_t n_cur = (N - n0 < NT) ? (N - n0) : NT;
      int32_t n_sid_off = sid * N_HALF;
      int32_t n_sid_cur = (n_cur - n_sid_off > 0) ? ((n_cur - n_sid_off < N_HALF) ? (n_cur - n_sid_off) : N_HALF) : 0;

      asc_sync_intra_arrive(PIPE_V, 2);

      for (int32_t ki = 0; ki < k_iters; ++ki) {
        int32_t k0 = ki * KT_PAD;
        int32_t k_cur = (K - k0 < KT_PAD) ? (K - k0) : KT_PAD;
        int32_t qw_stride = ((k_cur / 2 + 63) / 64) * 64;
        int32_t sc_stride = 64;
        int32_t x_stride = ((k_cur + 63) / 64) * 64;
        if (n_sid_cur > 0) {
          asc_copy_gm2ub_align((__ubuf__ uint8_t*)qw_in, (__gm__ uint8_t*)(qw + (int64_t)(n0 + n_sid_off) * row_bytes + k0 / 2),
                               (uint32_t)n_sid_cur, (uint32_t)(k_cur / 2), 0, 0, false,
                               asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                               (uint64_t)row_bytes, (uint32_t)qw_stride);
          asc_copy_gm2ub_align((__ubuf__ uint8_t*)sc_in, (__gm__ uint8_t*)(sc + (int64_t)(n0 + n_sid_off) * k_blks + k0 / 32),
                               (uint32_t)n_sid_cur, (uint32_t)((k_cur / 32) * 2), 0, 0, false,
                               asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                               (uint64_t)(k_blks * 2), (uint32_t)(sc_stride * 2));
        }
        if (m_sid_cur > 0) {
          asc_copy_gm2ub_align((__ubuf__ uint8_t*)x_in, (__gm__ uint8_t*)(x + (int64_t)(m0 + m_sid_off) * K + k0),
                               (uint32_t)m_sid_cur, (uint32_t)(k_cur * sizeof(T_io)), 0, 0, false,
                               asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                               (uint64_t)(K * sizeof(T_io)), (uint32_t)(x_stride * sizeof(T_io)));
        }
        asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);
        asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID0);
        q4_dequant_tile_vf<MT, NT, KT_PAD>(qw_in, sc_in, w_f32, w_nz, n_sid_cur, k_cur, qw_stride, sc_stride);
        q4_x_cast_scatter_vf<M_HALF, KT_PAD, T_io>(x_in, x_f32, x_nz, m_sid_cur, k_cur, x_stride);
        asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
        asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
        asc_copy_ub2l1((__cbuf__ void*)((__cbuf__ uint8_t*)l1_X + sid * M_HALF * C0 * sizeof(float)),
                       (__ubuf__ void*)x_nz, (uint16_t)(KT_PAD / C0), (uint16_t)(M_HALF * C0 * sizeof(float) / 32),
                       0, (uint16_t)(M_HALF * C0 * sizeof(float) / 32));
        asc_copy_ub2l1((__cbuf__ void*)((__cbuf__ uint8_t*)l1_W + sid * N_HALF * C0 * sizeof(float)),
                       (__ubuf__ void*)w_nz, (uint16_t)(KT_PAD / C0), (uint16_t)(N_HALF * C0 * sizeof(float) / 32),
                       0, (uint16_t)(N_HALF * C0 * sizeof(float) / 32));
        asc_sync_intra_arrive(PIPE_MTE3, 0);
        asc_sync_intra_wait(PIPE_MTE3, 3);
        asc_sync_notify(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        asc_sync_notify(PIPE_MTE3, PIPE_V, EVENT_ID0);
        asc_sync_wait(PIPE_MTE3, PIPE_V, EVENT_ID0);
      }

      asc_sync_intra_wait(PIPE_V, 1);
      if (m_sid_cur > 0) {
        q4_cast_y_vf<MT, T_io>(y_f, y_out, m_sid_cur, n_cur, NT);
        asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID1);
        asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID1);
        asc_copy_ub2gm_align((__gm__ uint8_t*)(y + (int64_t)(m0 + m_sid_off) * N + n0), (__ubuf__ uint8_t*)y_out,
                             (uint32_t)m_sid_cur, (uint32_t)(n_cur * sizeof(T_io)),
                             asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)(N * sizeof(T_io)), (uint32_t)(NT * sizeof(T_io)));
        asc_sync_notify(PIPE_MTE3, PIPE_V, EVENT_ID1);
        asc_sync_wait(PIPE_MTE3, PIPE_V, EVENT_ID1);
      }
    }
  }
  asc_sync();
}

#define DEF_Q4_GEMV(SUFFIX, T) \
extern "C" __global__ __aicore__ __vector__ void q4_gemv_##SUFFIX( \
    __gm__ T* x, __gm__ uint8_t* qw, __gm__ half* sc, __gm__ T* y, int32_t M, int32_t N, int32_t K) { \
  q4_gemv_impl<T>(x, qw, sc, y, M, N, K); \
}

DEF_Q4_GEMV(f16, half)
DEF_Q4_GEMV(bf16, bfloat16_t)
DEF_Q4_GEMV(f32, float)

#define DEF_Q4_MIX(SUFFIX, MT, NT, KT, T_IO) \
extern "C" __global__ __aicore__ __mix__(1, 2) void q4_mix_##SUFFIX( \
    __gm__ T_IO* x, __gm__ uint8_t* qw, __gm__ half* sc, __gm__ T_IO* y, int32_t M, int32_t N, int32_t K) { \
  q4_mix_gemm_impl<MT, NT, KT, T_IO>(x, qw, sc, y, M, N, K); \
}

DEF_Q4_MIX(f16_32_32_128, 32, 32, 128, half)
DEF_Q4_MIX(f16_32_64_32, 32, 64, 32, half)
DEF_Q4_MIX(f16_32_64_64, 32, 64, 64, half)
DEF_Q4_MIX(f16_32_64_128, 32, 64, 128, half)
DEF_Q4_MIX(bf16_32_32_128, 32, 32, 128, bfloat16_t)
DEF_Q4_MIX(bf16_32_64_32, 32, 64, 32, bfloat16_t)
DEF_Q4_MIX(bf16_32_64_64, 32, 64, 64, bfloat16_t)
DEF_Q4_MIX(bf16_32_64_128, 32, 64, 128, bfloat16_t)
DEF_Q4_MIX(f32_32_32_128, 32, 32, 128, float)
DEF_Q4_MIX(f32_32_64_32, 32, 64, 32, float)
DEF_Q4_MIX(f32_32_64_64, 32, 64, 64, float)
DEF_Q4_MIX(f32_32_64_128, 32, 64, 128, float)

extern "C" void launch_q4_0(
    void* stream, void* x, void* qw, void* sc, void* y,
    int32_t M, int32_t N, int32_t K, int32_t dtype_code) {
  if (M <= 8) {
    int32_t n_tile = (N <= 56) ? 1 : ((N + 55) / 56);
    if (n_tile > 32) n_tile = 32;
    int32_t tasks = (N + n_tile - 1) / n_tile;
    uint32_t grid = (uint32_t)(tasks < 56 ? tasks : 56);
    if (dtype_code == 0) q4_gemv_f16<<<grid, nullptr, stream>>>((__gm__ half*)x, (__gm__ uint8_t*)qw, (__gm__ half*)sc, (__gm__ half*)y, M, N, K);
    else if (dtype_code == 1) q4_gemv_bf16<<<grid, nullptr, stream>>>((__gm__ bfloat16_t*)x, (__gm__ uint8_t*)qw, (__gm__ half*)sc, (__gm__ bfloat16_t*)y, M, N, K);
    else q4_gemv_f32<<<grid, nullptr, stream>>>((__gm__ float*)x, (__gm__ uint8_t*)qw, (__gm__ half*)sc, (__gm__ float*)y, M, N, K);
    return;
  }
  int32_t kt = (K % 128 == 0) ? 128 : ((K % 64 == 0) ? 64 : 32);
  int32_t nt = (N <= 32 && kt == 128) ? 32 : 64;
  int32_t tasks = ((M + 31) / 32) * ((N + nt - 1) / nt);
  uint32_t grid = (uint32_t)(tasks < 28 ? tasks : 28);
  if (dtype_code == 0) {
    if (nt == 32) q4_mix_f16_32_32_128<<<grid, nullptr, stream>>>((__gm__ half*)x, (__gm__ uint8_t*)qw, (__gm__ half*)sc, (__gm__ half*)y, M, N, K);
    else if (kt == 32) q4_mix_f16_32_64_32<<<grid, nullptr, stream>>>((__gm__ half*)x, (__gm__ uint8_t*)qw, (__gm__ half*)sc, (__gm__ half*)y, M, N, K);
    else if (kt == 64) q4_mix_f16_32_64_64<<<grid, nullptr, stream>>>((__gm__ half*)x, (__gm__ uint8_t*)qw, (__gm__ half*)sc, (__gm__ half*)y, M, N, K);
    else q4_mix_f16_32_64_128<<<grid, nullptr, stream>>>((__gm__ half*)x, (__gm__ uint8_t*)qw, (__gm__ half*)sc, (__gm__ half*)y, M, N, K);
  } else if (dtype_code == 1) {
    if (nt == 32) q4_mix_bf16_32_32_128<<<grid, nullptr, stream>>>((__gm__ bfloat16_t*)x, (__gm__ uint8_t*)qw, (__gm__ half*)sc, (__gm__ bfloat16_t*)y, M, N, K);
    else if (kt == 32) q4_mix_bf16_32_64_32<<<grid, nullptr, stream>>>((__gm__ bfloat16_t*)x, (__gm__ uint8_t*)qw, (__gm__ half*)sc, (__gm__ bfloat16_t*)y, M, N, K);
    else if (kt == 64) q4_mix_bf16_32_64_64<<<grid, nullptr, stream>>>((__gm__ bfloat16_t*)x, (__gm__ uint8_t*)qw, (__gm__ half*)sc, (__gm__ bfloat16_t*)y, M, N, K);
    else q4_mix_bf16_32_64_128<<<grid, nullptr, stream>>>((__gm__ bfloat16_t*)x, (__gm__ uint8_t*)qw, (__gm__ half*)sc, (__gm__ bfloat16_t*)y, M, N, K);
  } else {
    if (nt == 32) q4_mix_f32_32_32_128<<<grid, nullptr, stream>>>((__gm__ float*)x, (__gm__ uint8_t*)qw, (__gm__ half*)sc, (__gm__ float*)y, M, N, K);
    else if (kt == 32) q4_mix_f32_32_64_32<<<grid, nullptr, stream>>>((__gm__ float*)x, (__gm__ uint8_t*)qw, (__gm__ half*)sc, (__gm__ float*)y, M, N, K);
    else if (kt == 64) q4_mix_f32_32_64_64<<<grid, nullptr, stream>>>((__gm__ float*)x, (__gm__ uint8_t*)qw, (__gm__ half*)sc, (__gm__ float*)y, M, N, K);
    else q4_mix_f32_32_64_128<<<grid, nullptr, stream>>>((__gm__ float*)x, (__gm__ uint8_t*)qw, (__gm__ half*)sc, (__gm__ float*)y, M, N, K);
  }
}
'''


def _render_trackb_tu(**dims: object) -> str:
    """Render an instantiated C++20 Track-B translation unit for q4_0_dequant_matmul."""
    raw_src = dims.get("trackb_src")
    return str(raw_src) if raw_src is not None else _Q4_0_ASC_SRC


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
    src_hash = hashlib.sha256(_Q4_0_ASC_SRC.encode("utf-8")).hexdigest()[:16]
    cached = _TRACK_B_LIB_CACHE.get(src_hash)
    if cached is not None:
        return cached
    build_dir = Path(tempfile.gettempdir()) / f"tl_npu_q4_tb_{src_hash}"
    build_dir.mkdir(parents=True, exist_ok=True)
    so_path = build_dir / "q4_0_trackb.so"
    lock_path = build_dir / "build.lock"
    if not so_path.is_file():
        with lock_path.open("w", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            if not so_path.is_file():
                src_path = build_dir / "q4_0_trackb.asc"
                tmp_so = build_dir / f"q4_0_trackb.{os.getpid()}.tmp.so"
                src_path.write_text(_Q4_0_ASC_SRC, encoding="utf-8")
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
                    raise RuntimeError(f"bisheng q4_0 Track B compile failed: {proc.stderr[-2000:]}")
                os.replace(tmp_so, so_path)
    lib = ctypes.CDLL(str(so_path))
    lib.launch_q4_0.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    _TRACK_B_LIB_CACHE[src_hash] = lib
    return lib


def _select_q4_0_dequant_matmul_config(**kwargs) -> dict[str, object]:
    """Select TileLang / Track-B config for q4_0_dequant_matmul."""
    raw_dt = str(kwargs.get("tl_dtype", "bfloat16")).strip()
    tl_dtype = raw_dt if raw_dt in ("float16", "bfloat16", "float32") else "bfloat16"
    m_dim = int(kwargs.get("m_dim", 64))
    n_dim = int(kwargs.get("n_dim", 64))
    k_dim = int(kwargs.get("k_dim", 128))
    if m_dim <= 8:
        tile_m = max(1, m_dim)
        tile_n = min(32, max(1, (n_dim + 55) // 56))
    else:
        tile_m = 32 if (m_dim >= 64 and n_dim >= 256) or (m_dim >= 32 and n_dim >= 128) else 16
        tile_n = 32 if n_dim <= 32 else 64
    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": 32 if k_dim % 64 != 0 else (64 if k_dim % 128 != 0 else 128),
        "num_stages": 2,
        "tl_dtype": tl_dtype,
        "pass_configs": {},
        "compile_flags": [],
    }


_ACL_LIB = None


def _get_acl_lib() -> ctypes.CDLL:
    global _ACL_LIB
    if _ACL_LIB is None:
        lib = ctypes.CDLL("libascendcl.so")
        lib.aclrtMemsetAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        lib.aclrtMemsetAsync.restype = ctypes.c_int
        lib.aclrtMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.aclrtMemcpyAsync.restype = ctypes.c_int
        _ACL_LIB = lib
    return _ACL_LIB


def q4_0_dequant_matmul(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    out_features: int,
    in_features: int,
    block_size: int = 32,
    quant_type: str = "q4_0",
) -> torch.Tensor:
    """Compute Q4_0 dequantized matrix multiplication on Ascend 950PR NPU."""
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported dtype {x.dtype}")
    if quant_type != "q4_0" or block_size != 32:
        raise ValueError(f"unsupported quant_type={quant_type!r} block_size={block_size}")
    if x.device.type != "npu":
        raise RuntimeError("q4_0_dequant_matmul requires NPU device")

    n_feat = int(out_features)
    k_feat = int(in_features)
    if k_feat % 32 != 0:
        raise ValueError(f"in_features must be divisible by 32, got {k_feat}")
    m_rows = x.numel() // k_feat
    out_shape = (*x.shape[:-1], n_feat) if x.ndim > 1 else (n_feat,)
    if m_rows == 0 or n_feat == 0 or k_feat == 0:
        return torch.empty(out_shape, dtype=x.dtype, device=x.device)

    lib = _get_track_b_lib()
    if lib is None:
        raise RuntimeError("bisheng compiler unavailable for q4_0_dequant_matmul Track-B")
    import torch_npu

    xc = x.contiguous()
    qwc = qweight.contiguous()
    scc = scales.to(torch.float16).contiguous()
    dcode = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}[xc.dtype]
    stream = torch_npu.npu.current_stream().npu_stream

    if m_rows <= 8 and k_feat > 512:
        x_2d = xc.view(m_rows, k_feat)
        x_pad = torch.empty((16, k_feat), dtype=xc.dtype, device=xc.device)
        libacl = _get_acl_lib()
        pad_bytes = x_pad.numel() * x_pad.element_size()
        rc_memset = libacl.aclrtMemsetAsync(
            ctypes.c_void_p(x_pad.data_ptr()),
            pad_bytes,
            0,
            pad_bytes,
            ctypes.c_void_p(stream),
        )
        if rc_memset != 0:
            raise RuntimeError(f"aclrtMemsetAsync failed with acl error code {rc_memset}")
        src_bytes = x_2d.numel() * x_2d.element_size()
        rc_cp_in = libacl.aclrtMemcpyAsync(
            ctypes.c_void_p(x_pad.data_ptr()),
            pad_bytes,
            ctypes.c_void_p(x_2d.data_ptr()),
            src_bytes,
            3,
            ctypes.c_void_p(stream),
        )
        if rc_cp_in != 0:
            raise RuntimeError(f"aclrtMemcpyAsync input failed with acl error code {rc_cp_in}")
        y_pad = torch.empty((16, n_feat), dtype=xc.dtype, device=xc.device)
        lib.launch_q4_0(
            ctypes.c_void_p(stream),
            ctypes.c_void_p(x_pad.data_ptr()),
            ctypes.c_void_p(qwc.data_ptr()),
            ctypes.c_void_p(scc.data_ptr()),
            ctypes.c_void_p(y_pad.data_ptr()),
            16,
            n_feat,
            k_feat,
            dcode,
        )
        y_out = torch.empty(out_shape, dtype=xc.dtype, device=xc.device)
        out_bytes = y_out.numel() * y_out.element_size()
        rc_cp_out = libacl.aclrtMemcpyAsync(
            ctypes.c_void_p(y_out.data_ptr()),
            out_bytes,
            ctypes.c_void_p(y_pad.data_ptr()),
            out_bytes,
            3,
            ctypes.c_void_p(stream),
        )
        if rc_cp_out != 0:
            raise RuntimeError(f"aclrtMemcpyAsync output failed with acl error code {rc_cp_out}")
        return y_out

    y = torch.empty(out_shape, dtype=xc.dtype, device=xc.device)
    lib.launch_q4_0(
        ctypes.c_void_p(stream),
        ctypes.c_void_p(xc.data_ptr()),
        ctypes.c_void_p(qwc.data_ptr()),
        ctypes.c_void_p(scc.data_ptr()),
        ctypes.c_void_p(y.data_ptr()),
        m_rows,
        n_feat,
        k_feat,
        dcode,
    )
    return y


kernel = q4_0_dequant_matmul

__all__ = [
    "_select_q4_0_dequant_matmul_config",
    "kernel",
    "q4_0_dequant_matmul",
]
