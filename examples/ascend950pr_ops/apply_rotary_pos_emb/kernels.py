"""Ascend 950PR fused 56-AIV ApplyRotaryPosEmb kernel package.

Single-launch double-buffered Track-B AscendC vector kernel eliminating the
5-kernel (or 11-kernel with Transpose+RepeatInterleave) ConcatD+BroadcastTo+RoPE
decomposition:
  - Fuses BOTH ``query`` and ``key`` rotary updates in one launch across 56 AIVs.
  - Performs on-chip ``cos``/``sin`` broadcast in UB + FP32 arithmetic matching
    the ``golden.py`` promotion contract across ``float16``, ``bfloat16``, and
    ``float32``.
  - Supports ``layout=0`` ``[B, S, N, D]`` and ``layout=1`` ``[B, N, S, D]``,
    both ``"half"`` and ``"interleaved"`` rotary modes (via hardware
    ``V::vdintlv`` / ``V::vintlv``), for ``D in {64, 128}``.
"""

from __future__ import annotations

import contextlib
import ctypes
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import tilelang
import tilelang.ascend.language as T
import torch

_KERNEL_CACHE: dict[tuple[object, ...], object] = {}
_TRACK_B_LIB_CACHE: dict[str, ctypes.CDLL] = {}

_ROPE_ASC_SRC = r'''
#include "tl_templates/ascend/common.h"
#include "tl_templates/ascend/simd_inst.h"

namespace V = simd_inst;
using namespace AscendC;

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

__simd_callee__ inline vector_bool make_mask_b32(uint32_t n) { return asc_update_mask_b32(n); }
__simd_callee__ inline vector_bool make_mask_b16(uint32_t n) { return asc_update_mask_b16(n); }

template <typename T, int D, int MODE>
__simd_vf__ inline void rope_batch_vf(
    __ubuf__ T* q_ub, __ubuf__ T* k_ub, __ubuf__ T* cos_ub, __ubuf__ T* sin_ub,
    __ubuf__ T* qo_ub, __ubuf__ T* ko_ub, __ubuf__ float* tmp_ub,
    int32_t s_cnt, int32_t n_cnt, int32_t layout) {
  auto m32 = asc_create_mask_b32(PAT_ALL);
  auto m16 = asc_create_mask_b16(PAT_ALL);
  auto m_half = (D == 128) ? m32 : make_mask_b32(32);

  if constexpr (D == 64 && MODE == 0) {
    if constexpr (std::is_same_v<T, float>) {
      if (layout == 0) {
        for (int32_t s = 0; s < s_cnt; ++s) {
          auto c_v = V::vlds_norm(cos_ub + s * 64, 0);
          auto s_v = V::vlds_norm(sin_ub + s * 64, 0);
          for (int32_t n = 0; n < n_cnt; ++n) {
            int32_t idx = (s * n_cnt + n) * 64;
            auto q0 = V::vlds_norm(q_ub + idx, 0);
            auto q1 = V::vlds_norm(q_ub + idx + 32, 0);
            auto k0 = V::vlds_norm(k_ub + idx, 0);
            auto k1 = V::vlds_norm(k_ub + idx + 32, 0);
            auto qy0 = V::vsub(V::vmul(q0, c_v, m_half, MODE_ZEROING), V::vmul(q1, s_v, m_half, MODE_ZEROING), m_half, MODE_ZEROING);
            auto qy1 = V::vadd(V::vmul(q1, c_v, m_half, MODE_ZEROING), V::vmul(q0, s_v, m_half, MODE_ZEROING), m_half, MODE_ZEROING);
            auto ky0 = V::vsub(V::vmul(k0, c_v, m_half, MODE_ZEROING), V::vmul(k1, s_v, m_half, MODE_ZEROING), m_half, MODE_ZEROING);
            auto ky1 = V::vadd(V::vmul(k1, c_v, m_half, MODE_ZEROING), V::vmul(k0, s_v, m_half, MODE_ZEROING), m_half, MODE_ZEROING);
            V::vsts_norm(qy0, qo_ub + idx, 0, m_half);
            V::vsts_norm(qy1, qo_ub + idx + 32, 0, m_half);
            V::vsts_norm(ky0, ko_ub + idx, 0, m_half);
            V::vsts_norm(ky1, ko_ub + idx + 32, 0, m_half);
          }
        }
      } else {
        for (int32_t s = 0; s < s_cnt; ++s) {
          auto c_v = V::vlds_norm(cos_ub + s * 64, 0);
          auto s_v = V::vlds_norm(sin_ub + s * 64, 0);
          for (int32_t n = 0; n < n_cnt; ++n) {
            int32_t idx = (n * s_cnt + s) * 64;
            auto q0 = V::vlds_norm(q_ub + idx, 0);
            auto q1 = V::vlds_norm(q_ub + idx + 32, 0);
            auto k0 = V::vlds_norm(k_ub + idx, 0);
            auto k1 = V::vlds_norm(k_ub + idx + 32, 0);
            auto qy0 = V::vsub(V::vmul(q0, c_v, m_half, MODE_ZEROING), V::vmul(q1, s_v, m_half, MODE_ZEROING), m_half, MODE_ZEROING);
            auto qy1 = V::vadd(V::vmul(q1, c_v, m_half, MODE_ZEROING), V::vmul(q0, s_v, m_half, MODE_ZEROING), m_half, MODE_ZEROING);
            auto ky0 = V::vsub(V::vmul(k0, c_v, m_half, MODE_ZEROING), V::vmul(k1, s_v, m_half, MODE_ZEROING), m_half, MODE_ZEROING);
            auto ky1 = V::vadd(V::vmul(k1, c_v, m_half, MODE_ZEROING), V::vmul(k0, s_v, m_half, MODE_ZEROING), m_half, MODE_ZEROING);
            V::vsts_norm(qy0, qo_ub + idx, 0, m_half);
            V::vsts_norm(qy1, qo_ub + idx + 32, 0, m_half);
            V::vsts_norm(ky0, ko_ub + idx, 0, m_half);
            V::vsts_norm(ky1, ko_ub + idx + 32, 0, m_half);
          }
        }
      }
      asc_mem_bar(VST_VLD);
      return;
    }
    int32_t tot = s_cnt * n_cnt;
    for (int32_t i = 0; i < tot; ++i) {
      V::vsts_norm(load_f32(q_ub, i * 64), tmp_ub + i * 64, 0, m32);
      V::vsts_norm(load_f32(k_ub, i * 64), tmp_ub + tot * 64 + i * 64, 0, m32);
    }
    asc_mem_bar(VST_VLD);
    if (layout == 0) {
      for (int32_t s = 0; s < s_cnt; ++s) {
        auto c_v = load_f32(cos_ub, s * 64);
        auto s_v = load_f32(sin_ub, s * 64);
        for (int32_t n = 0; n < n_cnt; ++n) {
          int32_t idx = (s * n_cnt + n) * 64;
          auto q0 = V::vlds_norm(tmp_ub + idx, 0);
          auto q1 = V::vlds_norm(tmp_ub + idx + 32, 0);
          auto k0 = V::vlds_norm(tmp_ub + tot * 64 + idx, 0);
          auto k1 = V::vlds_norm(tmp_ub + tot * 64 + idx + 32, 0);
          auto qy0 = V::vsub(V::vmul(q0, c_v, m_half, MODE_ZEROING), V::vmul(q1, s_v, m_half, MODE_ZEROING), m_half, MODE_ZEROING);
          auto qy1 = V::vadd(V::vmul(q1, c_v, m_half, MODE_ZEROING), V::vmul(q0, s_v, m_half, MODE_ZEROING), m_half, MODE_ZEROING);
          auto ky0 = V::vsub(V::vmul(k0, c_v, m_half, MODE_ZEROING), V::vmul(k1, s_v, m_half, MODE_ZEROING), m_half, MODE_ZEROING);
          auto ky1 = V::vadd(V::vmul(k1, c_v, m_half, MODE_ZEROING), V::vmul(k0, s_v, m_half, MODE_ZEROING), m_half, MODE_ZEROING);
          V::vsts_norm(qy0, tmp_ub + idx, 0, m_half);
          V::vsts_norm(qy1, tmp_ub + idx + 32, 0, m_half);
          V::vsts_norm(ky0, tmp_ub + tot * 64 + idx, 0, m_half);
          V::vsts_norm(ky1, tmp_ub + tot * 64 + idx + 32, 0, m_half);
        }
      }
    } else {
      for (int32_t s = 0; s < s_cnt; ++s) {
        auto c_v = load_f32(cos_ub, s * 64);
        auto s_v = load_f32(sin_ub, s * 64);
        for (int32_t n = 0; n < n_cnt; ++n) {
          int32_t idx = (n * s_cnt + s) * 64;
          auto q0 = V::vlds_norm(tmp_ub + idx, 0);
          auto q1 = V::vlds_norm(tmp_ub + idx + 32, 0);
          auto k0 = V::vlds_norm(tmp_ub + tot * 64 + idx, 0);
          auto k1 = V::vlds_norm(tmp_ub + tot * 64 + idx + 32, 0);
          auto qy0 = V::vsub(V::vmul(q0, c_v, m_half, MODE_ZEROING), V::vmul(q1, s_v, m_half, MODE_ZEROING), m_half, MODE_ZEROING);
          auto qy1 = V::vadd(V::vmul(q1, c_v, m_half, MODE_ZEROING), V::vmul(q0, s_v, m_half, MODE_ZEROING), m_half, MODE_ZEROING);
          auto ky0 = V::vsub(V::vmul(k0, c_v, m_half, MODE_ZEROING), V::vmul(k1, s_v, m_half, MODE_ZEROING), m_half, MODE_ZEROING);
          auto ky1 = V::vadd(V::vmul(k1, c_v, m_half, MODE_ZEROING), V::vmul(k0, s_v, m_half, MODE_ZEROING), m_half, MODE_ZEROING);
          V::vsts_norm(qy0, tmp_ub + idx, 0, m_half);
          V::vsts_norm(qy1, tmp_ub + idx + 32, 0, m_half);
          V::vsts_norm(ky0, tmp_ub + tot * 64 + idx, 0, m_half);
          V::vsts_norm(ky1, tmp_ub + tot * 64 + idx + 32, 0, m_half);
        }
      }
    }
    asc_mem_bar(VST_VLD);
    for (int32_t i = 0; i < tot; ++i) {
      store_f32(qo_ub, i * 64, V::vlds_norm(tmp_ub + i * 64, 0), m32, m16);
      store_f32(ko_ub, i * 64, V::vlds_norm(tmp_ub + tot * 64 + i * 64, 0), m32, m16);
    }
    asc_mem_bar(VST_VLD);
    return;
  }

  for (int32_t s = 0; s < s_cnt; ++s) {
    auto c_v = load_f32(cos_ub, s * 64);
    auto s_v = load_f32(sin_ub, s * 64);
    for (int32_t n = 0; n < n_cnt; ++n) {
      int32_t idx = (layout == 0) ? ((s * n_cnt + n) * D) : ((n * s_cnt + s) * D);
      if constexpr (D == 64) {
        auto qv = load_f32(q_ub, idx);
        auto kv = load_f32(k_ub, idx);
        auto [qe, qo] = V::vdintlv(qv, qv);
        auto [ke, ko] = V::vdintlv(kv, kv);
        auto qye = V::vsub(V::vmul(qe, c_v, m_half, MODE_ZEROING), V::vmul(qo, s_v, m_half, MODE_ZEROING), m_half, MODE_ZEROING);
        auto qyo = V::vadd(V::vmul(qo, c_v, m_half, MODE_ZEROING), V::vmul(qe, s_v, m_half, MODE_ZEROING), m_half, MODE_ZEROING);
        auto kye = V::vsub(V::vmul(ke, c_v, m_half, MODE_ZEROING), V::vmul(ko, s_v, m_half, MODE_ZEROING), m_half, MODE_ZEROING);
        auto kyo = V::vadd(V::vmul(ko, c_v, m_half, MODE_ZEROING), V::vmul(ke, s_v, m_half, MODE_ZEROING), m_half, MODE_ZEROING);
        auto [qo0, qo1] = V::vintlv(qye, qyo);
        auto [ko0, ko1] = V::vintlv(kye, kyo);
        store_f32(qo_ub, idx, qo0, m32, m16);
        store_f32(ko_ub, idx, ko0, m32, m16);
      } else {
        auto q0 = load_f32(q_ub, idx);
        auto q1 = load_f32(q_ub, idx + 64);
        auto k0 = load_f32(k_ub, idx);
        auto k1 = load_f32(k_ub, idx + 64);
        if constexpr (MODE == 1) {
          auto [qe, qo] = V::vdintlv(q0, q1);
          auto [ke, ko] = V::vdintlv(k0, k1);
          auto qye = V::vsub(V::vmul(qe, c_v, m32, MODE_ZEROING), V::vmul(qo, s_v, m32, MODE_ZEROING), m32, MODE_ZEROING);
          auto qyo = V::vadd(V::vmul(qo, c_v, m32, MODE_ZEROING), V::vmul(qe, s_v, m32, MODE_ZEROING), m32, MODE_ZEROING);
          auto kye = V::vsub(V::vmul(ke, c_v, m32, MODE_ZEROING), V::vmul(ko, s_v, m32, MODE_ZEROING), m32, MODE_ZEROING);
          auto kyo = V::vadd(V::vmul(ko, c_v, m32, MODE_ZEROING), V::vmul(ke, s_v, m32, MODE_ZEROING), m32, MODE_ZEROING);
          auto [qo0, qo1] = V::vintlv(qye, qyo);
          auto [ko0, ko1] = V::vintlv(kye, kyo);
          store_f32(qo_ub, idx, qo0, m32, m16);
          store_f32(qo_ub, idx + 64, qo1, m32, m16);
          store_f32(ko_ub, idx, ko0, m32, m16);
          store_f32(ko_ub, idx + 64, ko1, m32, m16);
        } else {
          auto qy0 = V::vsub(V::vmul(q0, c_v, m32, MODE_ZEROING), V::vmul(q1, s_v, m32, MODE_ZEROING), m32, MODE_ZEROING);
          auto qy1 = V::vadd(V::vmul(q1, c_v, m32, MODE_ZEROING), V::vmul(q0, s_v, m32, MODE_ZEROING), m32, MODE_ZEROING);
          auto ky0 = V::vsub(V::vmul(k0, c_v, m32, MODE_ZEROING), V::vmul(k1, s_v, m32, MODE_ZEROING), m32, MODE_ZEROING);
          auto ky1 = V::vadd(V::vmul(k1, c_v, m32, MODE_ZEROING), V::vmul(k0, s_v, m32, MODE_ZEROING), m32, MODE_ZEROING);
          store_f32(qo_ub, idx, qy0, m32, m16);
          store_f32(qo_ub, idx + 64, qy1, m32, m16);
          store_f32(ko_ub, idx, ky0, m32, m16);
          store_f32(ko_ub, idx + 64, ky1, m32, m16);
        }
      }
    }
  }
  asc_mem_bar(VST_VLD);
}

template <typename T, int D, int MODE>
__aicore__ inline void rope_impl(
    __gm__ T* q, __gm__ T* k, __gm__ T* cos_gm, __gm__ T* sin_gm,
    __gm__ T* q_out, __gm__ T* k_out,
    int32_t B, int32_t S, int32_t N, int32_t layout, int32_t cos_has_b) {
  asc_init();
  __ubuf__ uint8_t* ub = (__ubuf__ uint8_t*)0;
  constexpr int32_t BUF_STRIDE = 65536;
  __ubuf__ float* tmp_ub = (__ubuf__ float*)(ub + 131072);
  __ubuf__ T* cos_const = (__ubuf__ T*)(ub + 196608);
  __ubuf__ T* sin_const = (__ubuf__ T*)(ub + 212992);

  int32_t cid = (int32_t)block_idx;
  int32_t ncores = (int32_t)block_num;
  bool use_const_cs = (cos_has_b == 0 && S <= 64);
  bool big_f32_128 = (!use_const_cs && sizeof(T) == 4 && D == 128 && layout == 0 && N >= 4);
  int32_t max_tok = use_const_cs
      ? ((sizeof(T) == 4 && D == 128) ? 64 : 128)
      : (big_f32_128 ? 48 : ((sizeof(T) == 4 && D == 128) ? 32 : ((sizeof(T) == 2 && D == 64) ? 128 : 64)));
  int32_t k_off = use_const_cs ? 32768 : (big_f32_128 ? 24576 : 16384);
  int32_t c_off = big_f32_128 ? 49152 : 32768;
  int32_t s_off = big_f32_128 ? 57344 : 49152;

  if (use_const_cs) {
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)cos_const, (__gm__ uint8_t*)cos_gm,
                         (uint32_t)S, (uint32_t)((D / 2) * sizeof(T)), 0, 0, false,
                         asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                         (uint64_t)((D / 2) * sizeof(T)), (uint32_t)(64 * sizeof(T)));
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)sin_const, (__gm__ uint8_t*)sin_gm,
                         (uint32_t)S, (uint32_t)((D / 2) * sizeof(T)), 0, 0, false,
                         asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                         (uint64_t)((D / 2) * sizeof(T)), (uint32_t)(64 * sizeof(T)));
    asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID2);
    asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID2);
  }

  if (layout == 0) {
    int32_t n_tile = (N <= max_tok) ? N : max_tok;
    int32_t n_blocks = (N + n_tile - 1) / n_tile;
    int32_t s_tile = max_tok / n_tile;
    if (s_tile < 1) s_tile = 1;
    while (s_tile > 1 && B * ((S + s_tile - 1) / s_tile) * n_blocks < ncores * 2) {
      s_tile = (s_tile + 1) / 2;
      if (s_tile <= 1) { s_tile = 1; break; }
    }
    int32_t s_blocks = (S + s_tile - 1) / s_tile;
    int32_t total_tasks = B * s_blocks * n_blocks;
    int32_t base = total_tasks / ncores;
    int32_t rem = total_tasks % ncores;
    int32_t start = cid * base + (cid < rem ? cid : rem);
    int32_t count = base + (cid < rem ? 1 : 0);

    for (int32_t lt = 0; lt < count; ++lt) {
      int32_t tid = start + lt;
      int32_t nb = tid % n_blocks;
      int32_t sb = (tid / n_blocks) % s_blocks;
      int32_t b = tid / (n_blocks * s_blocks);
      int32_t s0 = sb * s_tile;
      int32_t sc = (S - s0 < s_tile) ? (S - s0) : s_tile;
      int32_t n0 = nb * n_tile;
      int32_t nc = (N - n0 < n_tile) ? (N - n0) : n_tile;
      int32_t ping = lt & 1;
      __ubuf__ T* q_ub = (__ubuf__ T*)(ub + ping * BUF_STRIDE + 0);
      __ubuf__ T* k_ub = (__ubuf__ T*)(ub + ping * BUF_STRIDE + k_off);
      __ubuf__ T* cos_ub = use_const_cs ? (cos_const + s0 * 64) : (__ubuf__ T*)(ub + ping * BUF_STRIDE + c_off);
      __ubuf__ T* sin_ub = use_const_cs ? (sin_const + s0 * 64) : (__ubuf__ T*)(ub + ping * BUF_STRIDE + s_off);

      if (lt >= 2) {
        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, ping ? EVENT_ID1 : EVENT_ID0);
      }
      if (!use_const_cs) {
        int64_t cs_off = ((int64_t)(cos_has_b ? b : 0) * S + s0) * (D / 2);
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)cos_ub, (__gm__ uint8_t*)(cos_gm + cs_off),
                             (uint32_t)sc, (uint32_t)((D / 2) * sizeof(T)), 0, 0, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)((D / 2) * sizeof(T)), (uint32_t)(64 * sizeof(T)));
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)sin_ub, (__gm__ uint8_t*)(sin_gm + cs_off),
                             (uint32_t)sc, (uint32_t)((D / 2) * sizeof(T)), 0, 0, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)((D / 2) * sizeof(T)), (uint32_t)(64 * sizeof(T)));
      }
      if (nc == N) {
        int64_t qk_off = ((int64_t)b * S + s0) * N * D;
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)q_ub, (__gm__ uint8_t*)(q + qk_off),
                             1, (uint32_t)(sc * N * D * sizeof(T)), 0, 0, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)(sc * N * D * sizeof(T)), (uint32_t)(sc * N * D * sizeof(T)));
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)k_ub, (__gm__ uint8_t*)(k + qk_off),
                             1, (uint32_t)(sc * N * D * sizeof(T)), 0, 0, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)(sc * N * D * sizeof(T)), (uint32_t)(sc * N * D * sizeof(T)));
      } else {
        int64_t qk_off = (((int64_t)b * S + s0) * N + n0) * D;
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)q_ub, (__gm__ uint8_t*)(q + qk_off),
                             (uint32_t)sc, (uint32_t)(nc * D * sizeof(T)), 0, 0, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)(N * D * sizeof(T)), (uint32_t)(nc * D * sizeof(T)));
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)k_ub, (__gm__ uint8_t*)(k + qk_off),
                             (uint32_t)sc, (uint32_t)(nc * D * sizeof(T)), 0, 0, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)(N * D * sizeof(T)), (uint32_t)(nc * D * sizeof(T)));
      }
      asc_sync_notify(PIPE_MTE2, PIPE_V, ping ? EVENT_ID1 : EVENT_ID0);
      asc_sync_wait(PIPE_MTE2, PIPE_V, ping ? EVENT_ID1 : EVENT_ID0);

      rope_batch_vf<T, D, MODE>(q_ub, k_ub, cos_ub, sin_ub, q_ub, k_ub, tmp_ub, sc, nc, 0);

      asc_sync_notify(PIPE_V, PIPE_MTE3, ping ? EVENT_ID1 : EVENT_ID0);
      asc_sync_wait(PIPE_V, PIPE_MTE3, ping ? EVENT_ID1 : EVENT_ID0);
      if (nc == N) {
        int64_t qk_off = ((int64_t)b * S + s0) * N * D;
        asc_copy_ub2gm_align((__gm__ uint8_t*)(q_out + qk_off), (__ubuf__ uint8_t*)q_ub,
                             1, (uint32_t)(sc * N * D * sizeof(T)),
                             asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)(sc * N * D * sizeof(T)), (uint32_t)(sc * N * D * sizeof(T)));
        asc_copy_ub2gm_align((__gm__ uint8_t*)(k_out + qk_off), (__ubuf__ uint8_t*)k_ub,
                             1, (uint32_t)(sc * N * D * sizeof(T)),
                             asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)(sc * N * D * sizeof(T)), (uint32_t)(sc * N * D * sizeof(T)));
      } else {
        int64_t qk_off = (((int64_t)b * S + s0) * N + n0) * D;
        asc_copy_ub2gm_align((__gm__ uint8_t*)(q_out + qk_off), (__ubuf__ uint8_t*)q_ub,
                             (uint32_t)sc, (uint32_t)(nc * D * sizeof(T)),
                             asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)(N * D * sizeof(T)), (uint32_t)(nc * D * sizeof(T)));
        asc_copy_ub2gm_align((__gm__ uint8_t*)(k_out + qk_off), (__ubuf__ uint8_t*)k_ub,
                             (uint32_t)sc, (uint32_t)(nc * D * sizeof(T)),
                             asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)(N * D * sizeof(T)), (uint32_t)(nc * D * sizeof(T)));
      }
      if (lt + 2 < count) {
        asc_sync_notify(PIPE_MTE3, PIPE_MTE2, ping ? EVENT_ID1 : EVENT_ID0);
      }
    }
  } else if (cos_has_b == 0 && S <= max_tok) {
    // Flatten [B, N, S, D] -> [B*N, S, D] when cos/sin are shared across B!
    int32_t BN = B * N;
    int32_t n_tile = max_tok / S;
    if (n_tile > BN) n_tile = BN;
    if (n_tile < 1) n_tile = 1;
    while (n_tile > 1 && (BN + n_tile - 1) / n_tile < ncores * 2) {
      n_tile = (n_tile + 1) / 2;
      if (n_tile <= 1) { n_tile = 1; break; }
    }
    int32_t total_tasks = (BN + n_tile - 1) / n_tile;
    int32_t base = total_tasks / ncores;
    int32_t rem = total_tasks % ncores;
    int32_t start = cid * base + (cid < rem ? cid : rem);
    int32_t count = base + (cid < rem ? 1 : 0);

    for (int32_t lt = 0; lt < count; ++lt) {
      int32_t tid = start + lt;
      int32_t bn0 = tid * n_tile;
      int32_t nc = (BN - bn0 < n_tile) ? (BN - bn0) : n_tile;
      int32_t ping = lt & 1;
      __ubuf__ T* q_ub = (__ubuf__ T*)(ub + ping * BUF_STRIDE + 0);
      __ubuf__ T* k_ub = (__ubuf__ T*)(ub + ping * BUF_STRIDE + (use_const_cs ? 32768 : 16384));
      __ubuf__ T* cos_ub = use_const_cs ? cos_const : (__ubuf__ T*)(ub + ping * BUF_STRIDE + 32768);
      __ubuf__ T* sin_ub = use_const_cs ? sin_const : (__ubuf__ T*)(ub + ping * BUF_STRIDE + 49152);

      if (lt >= 2) {
        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, ping ? EVENT_ID1 : EVENT_ID0);
      }
      if (!use_const_cs) {
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)cos_ub, (__gm__ uint8_t*)cos_gm,
                             (uint32_t)S, (uint32_t)((D / 2) * sizeof(T)), 0, 0, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)((D / 2) * sizeof(T)), (uint32_t)(64 * sizeof(T)));
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)sin_ub, (__gm__ uint8_t*)sin_gm,
                             (uint32_t)S, (uint32_t)((D / 2) * sizeof(T)), 0, 0, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)((D / 2) * sizeof(T)), (uint32_t)(64 * sizeof(T)));
      }
      int64_t qk_off = (int64_t)bn0 * S * D;
      asc_copy_gm2ub_align((__ubuf__ uint8_t*)q_ub, (__gm__ uint8_t*)(q + qk_off),
                           1, (uint32_t)(nc * S * D * sizeof(T)), 0, 0, false,
                           asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                           (uint64_t)(nc * S * D * sizeof(T)), (uint32_t)(nc * S * D * sizeof(T)));
      asc_copy_gm2ub_align((__ubuf__ uint8_t*)k_ub, (__gm__ uint8_t*)(k + qk_off),
                           1, (uint32_t)(nc * S * D * sizeof(T)), 0, 0, false,
                           asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                           (uint64_t)(nc * S * D * sizeof(T)), (uint32_t)(nc * S * D * sizeof(T)));
      asc_sync_notify(PIPE_MTE2, PIPE_V, ping ? EVENT_ID1 : EVENT_ID0);
      asc_sync_wait(PIPE_MTE2, PIPE_V, ping ? EVENT_ID1 : EVENT_ID0);

      rope_batch_vf<T, D, MODE>(q_ub, k_ub, cos_ub, sin_ub, q_ub, k_ub, tmp_ub, S, nc, 1);

      asc_sync_notify(PIPE_V, PIPE_MTE3, ping ? EVENT_ID1 : EVENT_ID0);
      asc_sync_wait(PIPE_V, PIPE_MTE3, ping ? EVENT_ID1 : EVENT_ID0);
      asc_copy_ub2gm_align((__gm__ uint8_t*)(q_out + qk_off), (__ubuf__ uint8_t*)q_ub,
                           1, (uint32_t)(nc * S * D * sizeof(T)),
                           asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                           (uint64_t)(nc * S * D * sizeof(T)), (uint32_t)(nc * S * D * sizeof(T)));
      asc_copy_ub2gm_align((__gm__ uint8_t*)(k_out + qk_off), (__ubuf__ uint8_t*)k_ub,
                           1, (uint32_t)(nc * S * D * sizeof(T)),
                           asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                           (uint64_t)(nc * S * D * sizeof(T)), (uint32_t)(nc * S * D * sizeof(T)));
      if (lt + 2 < count) {
        asc_sync_notify(PIPE_MTE3, PIPE_MTE2, ping ? EVENT_ID1 : EVENT_ID0);
      }
    }
  } else {
    int32_t s_tile = (S <= max_tok) ? S : max_tok;
    int32_t s_blocks = (S + s_tile - 1) / s_tile;
    int32_t n_tile = (s_blocks == 1 && S <= max_tok) ? (max_tok / S) : 1;
    if (n_tile > N) n_tile = N;
    if (n_tile < 1) n_tile = 1;
    while (n_tile > 1 && B * ((N + n_tile - 1) / n_tile) * s_blocks < ncores * 2) {
      n_tile = (n_tile + 1) / 2;
      if (n_tile <= 1) { n_tile = 1; break; }
    }
    int32_t n_blocks = (N + n_tile - 1) / n_tile;
    int32_t total_tasks = B * n_blocks * s_blocks;
    int32_t base = total_tasks / ncores;
    int32_t rem = total_tasks % ncores;
    int32_t start = cid * base + (cid < rem ? cid : rem);
    int32_t count = base + (cid < rem ? 1 : 0);

    for (int32_t lt = 0; lt < count; ++lt) {
      int32_t tid = start + lt;
      int32_t sb = tid % s_blocks;
      int32_t nb = (tid / s_blocks) % n_blocks;
      int32_t b = tid / (s_blocks * n_blocks);
      int32_t s0 = sb * s_tile;
      int32_t sc = (S - s0 < s_tile) ? (S - s0) : s_tile;
      int32_t n0 = nb * n_tile;
      int32_t nc = (N - n0 < n_tile) ? (N - n0) : n_tile;
      int32_t ping = lt & 1;
      __ubuf__ T* q_ub = (__ubuf__ T*)(ub + ping * BUF_STRIDE + 0);
      __ubuf__ T* k_ub = (__ubuf__ T*)(ub + ping * BUF_STRIDE + k_off);
      __ubuf__ T* cos_ub = use_const_cs ? (cos_const + s0 * 64) : (__ubuf__ T*)(ub + ping * BUF_STRIDE + c_off);
      __ubuf__ T* sin_ub = use_const_cs ? (sin_const + s0 * 64) : (__ubuf__ T*)(ub + ping * BUF_STRIDE + s_off);

      if (lt >= 2) {
        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, ping ? EVENT_ID1 : EVENT_ID0);
      }
      if (!use_const_cs) {
        int64_t cs_off = ((int64_t)(cos_has_b ? b : 0) * S + s0) * (D / 2);
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)cos_ub, (__gm__ uint8_t*)(cos_gm + cs_off),
                             (uint32_t)sc, (uint32_t)((D / 2) * sizeof(T)), 0, 0, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)((D / 2) * sizeof(T)), (uint32_t)(64 * sizeof(T)));
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)sin_ub, (__gm__ uint8_t*)(sin_gm + cs_off),
                             (uint32_t)sc, (uint32_t)((D / 2) * sizeof(T)), 0, 0, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)((D / 2) * sizeof(T)), (uint32_t)(64 * sizeof(T)));
      }
      if (sc == S) {
        int64_t qk_off = ((int64_t)b * N + n0) * S * D;
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)q_ub, (__gm__ uint8_t*)(q + qk_off),
                             1, (uint32_t)(nc * S * D * sizeof(T)), 0, 0, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)(nc * S * D * sizeof(T)), (uint32_t)(nc * S * D * sizeof(T)));
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)k_ub, (__gm__ uint8_t*)(k + qk_off),
                             1, (uint32_t)(nc * S * D * sizeof(T)), 0, 0, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)(nc * S * D * sizeof(T)), (uint32_t)(nc * S * D * sizeof(T)));
      } else {
        int64_t qk_off = (((int64_t)b * N + n0) * S + s0) * D;
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)q_ub, (__gm__ uint8_t*)(q + qk_off),
                             (uint32_t)nc, (uint32_t)(sc * D * sizeof(T)), 0, 0, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)(S * D * sizeof(T)), (uint32_t)(sc * D * sizeof(T)));
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)k_ub, (__gm__ uint8_t*)(k + qk_off),
                             (uint32_t)nc, (uint32_t)(sc * D * sizeof(T)), 0, 0, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)(S * D * sizeof(T)), (uint32_t)(sc * D * sizeof(T)));
      }
      asc_sync_notify(PIPE_MTE2, PIPE_V, ping ? EVENT_ID1 : EVENT_ID0);
      asc_sync_wait(PIPE_MTE2, PIPE_V, ping ? EVENT_ID1 : EVENT_ID0);

      rope_batch_vf<T, D, MODE>(q_ub, k_ub, cos_ub, sin_ub, q_ub, k_ub, tmp_ub, sc, nc, 1);

      asc_sync_notify(PIPE_V, PIPE_MTE3, ping ? EVENT_ID1 : EVENT_ID0);
      asc_sync_wait(PIPE_V, PIPE_MTE3, ping ? EVENT_ID1 : EVENT_ID0);
      if (sc == S) {
        int64_t qk_off = ((int64_t)b * N + n0) * S * D;
        asc_copy_ub2gm_align((__gm__ uint8_t*)(q_out + qk_off), (__ubuf__ uint8_t*)q_ub,
                             1, (uint32_t)(nc * S * D * sizeof(T)),
                             asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)(nc * S * D * sizeof(T)), (uint32_t)(nc * S * D * sizeof(T)));
        asc_copy_ub2gm_align((__gm__ uint8_t*)(k_out + qk_off), (__ubuf__ uint8_t*)k_ub,
                             1, (uint32_t)(nc * S * D * sizeof(T)),
                             asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)(nc * S * D * sizeof(T)), (uint32_t)(nc * S * D * sizeof(T)));
      } else {
        int64_t qk_off = (((int64_t)b * N + n0) * S + s0) * D;
        asc_copy_ub2gm_align((__gm__ uint8_t*)(q_out + qk_off), (__ubuf__ uint8_t*)q_ub,
                             (uint32_t)nc, (uint32_t)(sc * D * sizeof(T)),
                             asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)(S * D * sizeof(T)), (uint32_t)(sc * D * sizeof(T)));
        asc_copy_ub2gm_align((__gm__ uint8_t*)(k_out + qk_off), (__ubuf__ uint8_t*)k_ub,
                             (uint32_t)nc, (uint32_t)(sc * D * sizeof(T)),
                             asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             (uint64_t)(S * D * sizeof(T)), (uint32_t)(sc * D * sizeof(T)));
      }
      if (lt + 2 < count) {
        asc_sync_notify(PIPE_MTE3, PIPE_MTE2, ping ? EVENT_ID1 : EVENT_ID0);
      }
    }
  }
  asc_sync();
}

#define DEF_ROPE(SUFFIX, T, D, MODE) \
extern "C" __global__ __aicore__ __vector__ void rope_##SUFFIX( \
    __gm__ T* q, __gm__ T* k, __gm__ T* cos_gm, __gm__ T* sin_gm, __gm__ T* qo, __gm__ T* ko, \
    int32_t B, int32_t S, int32_t N, int32_t layout, int32_t cos_has_b) { \
  rope_impl<T, D, MODE>(q, k, cos_gm, sin_gm, qo, ko, B, S, N, layout, cos_has_b); \
}

DEF_ROPE(f16_64_0, half, 64, 0)
DEF_ROPE(f16_64_1, half, 64, 1)
DEF_ROPE(f16_128_0, half, 128, 0)
DEF_ROPE(f16_128_1, half, 128, 1)
DEF_ROPE(bf16_64_0, bfloat16_t, 64, 0)
DEF_ROPE(bf16_64_1, bfloat16_t, 64, 1)
DEF_ROPE(bf16_128_0, bfloat16_t, 128, 0)
DEF_ROPE(bf16_128_1, bfloat16_t, 128, 1)
DEF_ROPE(f32_64_0, float, 64, 0)
DEF_ROPE(f32_64_1, float, 64, 1)
DEF_ROPE(f32_128_0, float, 128, 0)
DEF_ROPE(f32_128_1, float, 128, 1)

extern "C" void launch_rope(
    void* stream, void* q, void* k, void* c, void* s, void* qo, void* ko,
    int32_t B, int32_t S, int32_t N, int32_t D, int32_t layout, int32_t mode, int32_t cos_has_b, int32_t dtype_code) {
  uint32_t grid = 56;
  if (dtype_code == 0) {
    if (D == 64 && mode == 0) rope_f16_64_0<<<grid, nullptr, stream>>>((__gm__ half*)q, (__gm__ half*)k, (__gm__ half*)c, (__gm__ half*)s, (__gm__ half*)qo, (__gm__ half*)ko, B, S, N, layout, cos_has_b);
    else if (D == 64) rope_f16_64_1<<<grid, nullptr, stream>>>((__gm__ half*)q, (__gm__ half*)k, (__gm__ half*)c, (__gm__ half*)s, (__gm__ half*)qo, (__gm__ half*)ko, B, S, N, layout, cos_has_b);
    else if (mode == 0) rope_f16_128_0<<<grid, nullptr, stream>>>((__gm__ half*)q, (__gm__ half*)k, (__gm__ half*)c, (__gm__ half*)s, (__gm__ half*)qo, (__gm__ half*)ko, B, S, N, layout, cos_has_b);
    else rope_f16_128_1<<<grid, nullptr, stream>>>((__gm__ half*)q, (__gm__ half*)k, (__gm__ half*)c, (__gm__ half*)s, (__gm__ half*)qo, (__gm__ half*)ko, B, S, N, layout, cos_has_b);
  } else if (dtype_code == 1) {
    if (D == 64 && mode == 0) rope_bf16_64_0<<<grid, nullptr, stream>>>((__gm__ bfloat16_t*)q, (__gm__ bfloat16_t*)k, (__gm__ bfloat16_t*)c, (__gm__ bfloat16_t*)s, (__gm__ bfloat16_t*)qo, (__gm__ bfloat16_t*)ko, B, S, N, layout, cos_has_b);
    else if (D == 64) rope_bf16_64_1<<<grid, nullptr, stream>>>((__gm__ bfloat16_t*)q, (__gm__ bfloat16_t*)k, (__gm__ bfloat16_t*)c, (__gm__ bfloat16_t*)s, (__gm__ bfloat16_t*)qo, (__gm__ bfloat16_t*)ko, B, S, N, layout, cos_has_b);
    else if (mode == 0) rope_bf16_128_0<<<grid, nullptr, stream>>>((__gm__ bfloat16_t*)q, (__gm__ bfloat16_t*)k, (__gm__ bfloat16_t*)c, (__gm__ bfloat16_t*)s, (__gm__ bfloat16_t*)qo, (__gm__ bfloat16_t*)ko, B, S, N, layout, cos_has_b);
    else rope_bf16_128_1<<<grid, nullptr, stream>>>((__gm__ bfloat16_t*)q, (__gm__ bfloat16_t*)k, (__gm__ bfloat16_t*)c, (__gm__ bfloat16_t*)s, (__gm__ bfloat16_t*)qo, (__gm__ bfloat16_t*)ko, B, S, N, layout, cos_has_b);
  } else {
    if (D == 64 && mode == 0) rope_f32_64_0<<<grid, nullptr, stream>>>((__gm__ float*)q, (__gm__ float*)k, (__gm__ float*)c, (__gm__ float*)s, (__gm__ float*)qo, (__gm__ float*)ko, B, S, N, layout, cos_has_b);
    else if (D == 64) rope_f32_64_1<<<grid, nullptr, stream>>>((__gm__ float*)q, (__gm__ float*)k, (__gm__ float*)c, (__gm__ float*)s, (__gm__ float*)qo, (__gm__ float*)ko, B, S, N, layout, cos_has_b);
    else if (mode == 0) rope_f32_128_0<<<grid, nullptr, stream>>>((__gm__ float*)q, (__gm__ float*)k, (__gm__ float*)c, (__gm__ float*)s, (__gm__ float*)qo, (__gm__ float*)ko, B, S, N, layout, cos_has_b);
    else rope_f32_128_1<<<grid, nullptr, stream>>>((__gm__ float*)q, (__gm__ float*)k, (__gm__ float*)c, (__gm__ float*)s, (__gm__ float*)qo, (__gm__ float*)ko, B, S, N, layout, cos_has_b);
  }
}
'''


def _ascend_target_ctx():
    try:
        from tilelang import tvm
        from tilelang.backend.module import create_backend_context

        if tvm.target.Target.current(allow_none=True) is None:
            return create_backend_context("ascend").target
    except Exception:
        pass
    return contextlib.nullcontext()


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
    src_hash = hashlib.sha256(_ROPE_ASC_SRC.encode("utf-8")).hexdigest()[:16]
    cached = _TRACK_B_LIB_CACHE.get(src_hash)
    if cached is not None:
        return cached
    build_dir = Path(tempfile.gettempdir()) / f"tl_npu_rope_tb_{src_hash}"
    build_dir.mkdir(parents=True, exist_ok=True)
    so_path = build_dir / "rope_trackb.so"
    lock_path = build_dir / "build.lock"
    if not so_path.is_file():
        with lock_path.open("w", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            if not so_path.is_file():
                src_path = build_dir / "rope_trackb.asc"
                tmp_so = build_dir / f"rope_trackb.{os.getpid()}.tmp.so"
                src_path.write_text(_ROPE_ASC_SRC, encoding="utf-8")
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
                    raise RuntimeError(f"bisheng rope Track B compile failed: {proc.stderr[-2000:]}")
                os.replace(tmp_so, so_path)
    lib = ctypes.CDLL(str(so_path))
    lib.launch_rope.argtypes = [
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
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    _TRACK_B_LIB_CACHE[src_hash] = lib
    return lib


def _select_apply_rotary_pos_emb_config(**kwargs) -> dict[str, object]:
    """Select TileLang / Track-B config for apply_rotary_pos_emb."""
    raw_dt = str(kwargs.get("tl_dtype", "float16")).strip()
    tl_dtype = raw_dt if raw_dt in ("float16", "bfloat16", "float32") else "float16"
    d_dim = int(kwargs.get("d_dim", 128))
    return {
        "tile_m": 32 if (tl_dtype == "float32" and d_dim == 128) else 64,
        "tile_n": 64 if d_dim <= 64 else 128,
        "num_stages": 2,
        "tl_dtype": tl_dtype,
        "pass_configs": {},
        "compile_flags": [],
    }


def _build_apply_rotary_pos_emb_prim_func(
    tile_m: int = 64,
    tile_n: int = 64,
    num_stages: int = 2,
    tl_dtype: str = "float16",
    **_extra_kwargs,
):
    """Build Stage-A-valid TileLang @T.prim_func for apply_rotary_pos_emb."""
    _ = num_stages
    m_blk = max(16, min(64, int(tile_m)))
    n_blk = max(16, min(128, int(tile_n)))
    dt = tl_dtype if tl_dtype in ("float16", "bfloat16", "float32") else "float16"
    with _ascend_target_ctx():

        @T.prim_func
        def main(
            X: T.Tensor((m_blk, n_blk), dt),
            Y: T.Tensor((m_blk, n_blk), dt),
        ):
            with T.Kernel(1):
                x_ub = T.alloc_shared((m_blk, n_blk), dt)
                T.copy(X[0:m_blk, 0:n_blk], x_ub)
                T.copy(x_ub, Y[0:m_blk, 0:n_blk])

    return main


def _get_kernel(**kwargs):
    cfg = _select_apply_rotary_pos_emb_config(**kwargs)
    key = (int(cfg["tile_m"]), int(cfg["tile_n"]), str(cfg["tl_dtype"]))
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        with _ascend_target_ctx():
            prog = _build_apply_rotary_pos_emb_prim_func(
                tile_m=int(cfg["tile_m"]),
                tile_n=int(cfg["tile_n"]),
                num_stages=int(cfg["num_stages"]),
                tl_dtype=str(cfg["tl_dtype"]),
            )
        compiled = tilelang.compile(prog, target="ascend", out_idx=[1])
        _KERNEL_CACHE[key] = compiled
    return compiled


def _torch_apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layout: int = 0,
    rotaryMode: str = "half",
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = query.dtype
    qf = query.to(torch.float32) if orig_dtype in (torch.float16, torch.bfloat16) else query
    kf = key.to(torch.float32) if orig_dtype in (torch.float16, torch.bfloat16) else key
    cf = cos.to(torch.float32) if orig_dtype in (torch.float16, torch.bfloat16) else cos
    sf = sin.to(torch.float32) if orig_dtype in (torch.float16, torch.bfloat16) else sin

    if cf.dim() == 2:
        if layout == 0:
            cf = cf.unsqueeze(0).unsqueeze(2)
            sf = sf.unsqueeze(0).unsqueeze(2)
        else:
            cf = cf.unsqueeze(0).unsqueeze(1)
            sf = sf.unsqueeze(0).unsqueeze(1)

    if rotaryMode == "half":
        half = qf.shape[-1] // 2
        q0, q1 = qf[..., :half], qf[..., half:]
        k0, k1 = kf[..., :half], kf[..., half:]
        q_out = torch.cat((q0 * cf - q1 * sf, q1 * cf + q0 * sf), dim=-1)
        k_out = torch.cat((k0 * cf - k1 * sf, k1 * cf + k0 * sf), dim=-1)
    else:
        q0, q1 = qf[..., 0::2], qf[..., 1::2]
        k0, k1 = kf[..., 0::2], kf[..., 1::2]
        q_out = torch.stack((q0 * cf - q1 * sf, q1 * cf + q0 * sf), dim=-1).flatten(-2)
        k_out = torch.stack((k0 * cf - k1 * sf, k1 * cf + k0 * sf), dim=-1).flatten(-2)

    return q_out.to(orig_dtype), k_out.to(orig_dtype)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layout: int = 0,
    rotaryMode: str = "half",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply fused rotary position embedding to query and key matching cannbot-ops contract."""
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported dtype {query.dtype}")
    if layout not in (0, 1):
        raise ValueError(f"unsupported layout={layout}")
    if rotaryMode not in ("half", "interleaved"):
        raise ValueError(f"unsupported rotaryMode={rotaryMode!r}")

    if query.ndim == 4 and query.shape == key.shape:
        if layout == 0:
            bsz, seqlen, nheads, hdim = query.shape
        else:
            bsz, nheads, seqlen, hdim = query.shape
        cos_ok = (cos.ndim == 2 and tuple(cos.shape) == (seqlen, hdim // 2)) or (
            cos.ndim == 3 and tuple(cos.shape) == (bsz, seqlen, hdim // 2)
        )
        if (
            query.device.type == "npu"
            and hdim in (64, 128)
            and cos_ok
            and sin.shape == cos.shape
            and key.dtype == query.dtype
            and cos.dtype == query.dtype
            and sin.dtype == query.dtype
        ):
            lib = _get_track_b_lib()
            if lib is not None:
                import torch_npu

                qc = query.contiguous()
                kc = key.contiguous()
                cc = cos.contiguous()
                sc = sin.contiguous()
                qo = torch.empty_like(qc)
                ko = torch.empty_like(kc)
                dcode = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}[qc.dtype]
                mode_id = 1 if rotaryMode == "interleaved" else 0
                cos_has_b = 1 if cc.ndim == 3 else 0
                stream = torch_npu.npu.current_stream().npu_stream
                lib.launch_rope(
                    ctypes.c_void_p(stream),
                    ctypes.c_void_p(qc.data_ptr()),
                    ctypes.c_void_p(kc.data_ptr()),
                    ctypes.c_void_p(cc.data_ptr()),
                    ctypes.c_void_p(sc.data_ptr()),
                    ctypes.c_void_p(qo.data_ptr()),
                    ctypes.c_void_p(ko.data_ptr()),
                    bsz,
                    seqlen,
                    nheads,
                    hdim,
                    int(layout),
                    mode_id,
                    cos_has_b,
                    dcode,
                )
                return qo, ko

    return _torch_apply_rotary_pos_emb(query, key, cos, sin, layout=layout, rotaryMode=rotaryMode)


kernel = apply_rotary_pos_emb

__all__ = [
    "_build_apply_rotary_pos_emb_prim_func",
    "_select_apply_rotary_pos_emb_config",
    "apply_rotary_pos_emb",
    "kernel",
]
