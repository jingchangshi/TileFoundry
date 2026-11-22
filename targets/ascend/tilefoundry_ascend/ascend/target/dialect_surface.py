"""Single source of truth for Ascend TileLang DSL and Track-B CCE C++20 surfaces."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

DialectId = Literal["tilelang_dsl", "inline_cce_vf", "cce_cpp20"]

_IDENTIFIER_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def _forbidden_token_pattern(token: str) -> str:
    prefix = r"(?<![A-Za-z0-9_:.>])"
    suffix = r"(?![A-Za-z0-9_])" if token and token[-1] in _IDENTIFIER_CHARS else ""
    return f"{prefix}{re.escape(token)}{suffix}"


@dataclass(frozen=True)
class PrimitiveEntry:
    """Metadata describing one verified callable primitive on an Ascend dialect surface."""

    symbol: str
    pipe: str
    facet: str
    signature: str


@dataclass(frozen=True)
class DialectSurfaceSpec:
    """Verified dialect surface contract used by prompt generation, repair, and verifiers."""

    dialect_id: DialectId
    display_label: str
    required_headers_or_imports: tuple[str, ...]
    primitives: tuple[PrimitiveEntry, ...]
    forbidden_tokens: tuple[str, ...]
    prompt_notes: tuple[str, ...]

    def render_prompt_reference(self) -> str:
        """Render markdown bullet list of callable APIs, headers, and forbidden tokens."""
        lines = [
            f"Verified `{self.display_label}` (`{self.dialect_id}`) callable surface:",
            f"- Imports / Headers: {', '.join(self.required_headers_or_imports)}",
        ]
        for entry in self.primitives:
            lines.append(f"- `{entry.signature}` [{entry.pipe} | facet={entry.facet}]")
        if self.forbidden_tokens:
            joined = ", ".join(f"`{tok}`" for tok in self.forbidden_tokens)
            lines.append(f"- Strictly Forbidden Symbols: {joined}")
        for note in self.prompt_notes:
            lines.append(f"- {note}")
        return "\n".join(lines)

    def count_forbidden_tokens(self, text: str) -> dict[str, int]:
        """Count boundary-delimited occurrences of each forbidden token in *text*."""
        counts: dict[str, int] = {}
        for token in self.forbidden_tokens:
            counts[token] = len(re.findall(_forbidden_token_pattern(token), text))
        return counts

    def introduced_forbidden_tokens(self, before: str, after: str) -> tuple[str, ...]:
        """Return forbidden tokens whose boundary-delimited count grows from *before* to *after*."""
        before_counts = self.count_forbidden_tokens(before)
        after_counts = self.count_forbidden_tokens(after)
        return tuple(
            token
            for token in self.forbidden_tokens
            if after_counts[token] > before_counts[token]
        )


TILELANG_DSL_SURFACE = DialectSurfaceSpec(
    dialect_id="tilelang_dsl",
    display_label="TileLang DSL",
    required_headers_or_imports=(
        "import tilelang",
        "import tilelang.ascend.language as T",
    ),
    primitives=(
        PrimitiveEntry(
            "T.Kernel",
            "SCALAR",
            "grid_and_workspace_topology",
            "T.Kernel(blocks) as cid",
        ),
        PrimitiveEntry(
            "T.MixedKernel",
            "SCALAR",
            "grid_and_workspace_topology",
            "T.MixedKernel(blocks, sids=2) as (bx, sid)",
        ),
        PrimitiveEntry(
            "T.Persistent",
            "SCALAR",
            "grid_and_workspace_topology",
            "T.Persistent(...)",
        ),
        PrimitiveEntry(
            "T.Pipelined",
            "SCALAR",
            "pipeline_and_sync_graph",
            "T.Pipelined(extent, num_stages>=2, annotations={'enable_offset': True})",
        ),
        PrimitiveEntry(
            "T.serial",
            "SCALAR",
            "tile_and_stage_knobs",
            "T.serial(extent)",
        ),
        PrimitiveEntry(
            "T.dynamic",
            "SCALAR",
            "tile_and_stage_knobs",
            "T.dynamic(name, dtype=\"int32\")",
        ),
        PrimitiveEntry(
            "T.SimdVF",
            "PIPE_V",
            "exec_mode_and_spr",
            "T.SimdVF()",
        ),
        PrimitiveEntry(
            "T.SimtVF",
            "PIPE_V",
            "exec_mode_and_spr",
            "T.SimtVF(threads=128)",
        ),
        PrimitiveEntry(
            "T.alloc_shared",
            "UB",
            "sram_layout_and_aliasing",
            "T.alloc_shared(shape, dtype)",
        ),
        PrimitiveEntry(
            "T.alloc_l1",
            "L1",
            "sram_layout_and_aliasing",
            "T.alloc_l1(shape, dtype)",
        ),
        PrimitiveEntry(
            "T.alloc_l0a",
            "L0A",
            "sram_layout_and_aliasing",
            "T.alloc_l0a(shape, dtype)",
        ),
        PrimitiveEntry(
            "T.alloc_l0b",
            "L0B",
            "sram_layout_and_aliasing",
            "T.alloc_l0b(shape, dtype)",
        ),
        PrimitiveEntry(
            "T.alloc_l0c",
            "L0C",
            "sram_layout_and_aliasing",
            "T.alloc_l0c(shape, dtype)",
        ),
        PrimitiveEntry(
            "T.alloc_ss",
            "SSBUF",
            "sram_layout_and_aliasing",
            "T.alloc_ss(shape, dtype)",
        ),
        PrimitiveEntry(
            "T.copy",
            "PIPE_MTE2",
            "dma_and_cache_hierarchy",
            "T.copy(src, dst)",
        ),
        PrimitiveEntry(
            "T.dual_copy",
            "PIPE_FIX",
            "dma_and_cache_hierarchy",
            "T.dual_copy(src, dst, ...)",
        ),
        PrimitiveEntry(
            "T.gemm",
            "PIPE_M",
            "alg_compute_and_monoid",
            "T.gemm(a_l1, b_l1, c_l0c, transpose_A=..., transpose_B=..., clear_accum=...)",
        ),
        PrimitiveEntry(
            "T.ascend_pipe_barrier",
            "SCALAR",
            "pipeline_and_sync_graph",
            "T.ascend_pipe_barrier(pipe)",
        ),
        PrimitiveEntry(
            "T.fill",
            "PIPE_V",
            "alg_compute_and_monoid",
            "T.fill(buf, val)",
        ),
        PrimitiveEntry(
            "T.clear",
            "PIPE_V",
            "alg_compute_and_monoid",
            "T.clear(buf)",
        ),
        PrimitiveEntry(
            "T.reduce_sum",
            "PIPE_V",
            "alg_compute_and_monoid",
            "T.reduce_sum(src, dst, dim=...)",
        ),
        PrimitiveEntry(
            "T.reduce_max",
            "PIPE_V",
            "alg_compute_and_monoid",
            "T.reduce_max(src, dst, dim=...)",
        ),
        PrimitiveEntry(
            "T.reduce_absmax",
            "PIPE_V",
            "alg_compute_and_monoid",
            "T.reduce_absmax(src, dst, dim=...)",
        ),
        PrimitiveEntry(
            "T.simd.pset",
            "PIPE_V",
            "exec_mode_and_spr",
            "T.simd.pset(width, \"PAT_ALL\")",
        ),
        PrimitiveEntry(
            "T.simd.update_mask",
            "PIPE_V",
            "exec_mode_and_spr",
            "T.simd.update_mask(count, width=32)",
        ),
        PrimitiveEntry(
            "T.simd.vdup",
            "PIPE_V",
            "exec_mode_and_spr",
            "T.simd.vdup(value, dtype, mask=mask)",
        ),
        PrimitiveEntry(
            "T.simd.vld",
            "PIPE_V",
            "exec_mode_and_spr",
            "T.simd.vld(buf[offset], dist=\"NORM_B32\")",
        ),
        PrimitiveEntry(
            "T.simd.vcvt",
            "PIPE_V",
            "exec_mode_and_spr",
            "T.simd.vcvt(vec, dtype, mask=mask, part=0)",
        ),
        PrimitiveEntry(
            "T.simd.vexp",
            "PIPE_V",
            "exec_mode_and_spr",
            "T.simd.vexp(vec, mask=mask)",
        ),
        PrimitiveEntry(
            "T.simd.vln",
            "PIPE_V",
            "exec_mode_and_spr",
            "T.simd.vln(vec, mask=mask)",
        ),
        PrimitiveEntry(
            "T.simd.vsqrt",
            "PIPE_V",
            "exec_mode_and_spr",
            "T.simd.vsqrt(vec, mask=mask)",
        ),
        PrimitiveEntry(
            "T.simd.vabs",
            "PIPE_V",
            "alg_compute_and_monoid",
            "T.simd.vabs(vec, mask=mask)",
        ),
        PrimitiveEntry(
            "T.simd.vadd",
            "PIPE_V",
            "alg_compute_and_monoid",
            "T.simd.vadd(lhs, rhs, mask=mask)",
        ),
        PrimitiveEntry(
            "T.simd.vadds",
            "PIPE_V",
            "alg_compute_and_monoid",
            "T.simd.vadds(vec, scalar, mask=mask)",
        ),
        PrimitiveEntry(
            "T.simd.vsub",
            "PIPE_V",
            "alg_compute_and_monoid",
            "T.simd.vsub(lhs, rhs, mask=mask)",
        ),
        PrimitiveEntry(
            "T.simd.vmul",
            "PIPE_V",
            "alg_compute_and_monoid",
            "T.simd.vmul(lhs, rhs, mask=mask)",
        ),
        PrimitiveEntry(
            "T.simd.vmuls",
            "PIPE_V",
            "alg_compute_and_monoid",
            "T.simd.vmuls(vec, scalar, mask=mask)",
        ),
        PrimitiveEntry(
            "T.simd.vdiv",
            "PIPE_V",
            "alg_compute_and_monoid",
            "T.simd.vdiv(lhs, rhs, mask=mask)",
        ),
        PrimitiveEntry(
            "T.simd.vmax",
            "PIPE_V",
            "alg_compute_and_monoid",
            "T.simd.vmax(lhs, rhs, mask=mask)",
        ),
        PrimitiveEntry(
            "T.simd.vmin",
            "PIPE_V",
            "alg_compute_and_monoid",
            "T.simd.vmin(lhs, rhs, mask=mask)",
        ),
        PrimitiveEntry(
            "T.simd.vmaxs",
            "PIPE_V",
            "alg_compute_and_monoid",
            "T.simd.vmaxs(vec, scalar, mask=mask)",
        ),
        PrimitiveEntry(
            "T.simd.vcadd",
            "PIPE_V",
            "alg_compute_and_monoid",
            "T.simd.vcadd(vec, mask=mask)",
        ),
        PrimitiveEntry(
            "T.simd.vcmax",
            "PIPE_V",
            "alg_compute_and_monoid",
            "T.simd.vcmax(vec, mask=mask)",
        ),
        PrimitiveEntry(
            "T.simd.vsts",
            "PIPE_V",
            "exec_mode_and_spr",
            "T.simd.vsts(buf[offset], vec, mask=mask, dist=\"PK_B32\", extent=64)",
        ),
        PrimitiveEntry(
            "T.annotate_buffer_versions",
            "SCALAR",
            "pipeline_and_sync_graph",
            "T.annotate_buffer_versions({buf: stages})",
        ),
        PrimitiveEntry(
            "T.annotate_unlimit_memory",
            "SCALAR",
            "sram_layout_and_aliasing",
            "T.annotate_unlimit_memory(\"shared\")",
        ),
    ),
    forbidden_tokens=(
        "tilelang.language",
        "is_npu=",
        "T.Scope",
        "T.alloc_ub",
        "T.alloc_L1",
        "T.gemm_v0",
    ),
    prompt_notes=(
        (
            "Use T.serial(extent) for single-stage unpipelined loops; "
            "reserve T.Pipelined for num_stages >= 2."
        ),
        "Unlock 256 KiB UB via PURE_SIMDVF_UBUF_FLAGS in selector pass_configs.",
        (
            "Inside T.Parallel never call T.Cast, T.exp, T.log, T.tanh, T.sigmoid, or T.rsqrt "
            "unless the loop is nested under `with T.SimtVF(threads=128):`; the SIMD-VF "
            "parallel lowering rejects scalar cast and transcendental expressions."
        ),
        (
            "Run elementwise transcendental or mixed-precision math under `with T.SimdVF():` "
            "with `T.simd.*` (f32 via NORM_B32; f16/bf16 via US_B16 -> vcvt(\"float32\") -> "
            "compute -> vcvt(tl_dtype) -> PK_B32), or under `with T.SimtVF(threads=128):` "
            "with lowercase T.cast."
        ),
        (
            "T.simd.vrsqrt does not exist; implement rsqrt(x) inside `with T.SimdVF():` "
            "via `T.simd.vsqrt(x, mask=mask)` and `T.simd.vdiv(T.simd.vdup(1.0, \"float32\", "
            "mask=mask), sqrt_x, mask=mask)`."
        ),
        (
            "Declare runtime-varying batch or outer extents with T.dynamic(\"m_rows\") "
            "/ T.dynamic(\"b_dim\") / T.dynamic(\"s_dim\") / T.dynamic(\"s_kv\"); keep only "
            "tl_dtype and static block tiles (bm, bn, tile_m, num_stages) as compile-time constants."
        ),
        (
            "For matmul or coupled Cube+Vector workloads, use `with T.MixedKernel(28, sids=2) "
            "as (bx, sid):` with T.alloc_l1, T.alloc_l0c, T.gemm, T.alloc_shared, and T.SimdVF."
        ),
        (
            "For non-last-axis or high-rank reductions, reshape/view as 3D "
            "(outer, reduce_extent, inner) on NPU, pad UB tiles to 64-element multiples with "
            "T.simd.update_mask(rem, width=32/16), and guard all -inf rows so (-inf)-(-inf) "
            "never yields spurious NaN."
        ),
    ),
)

INLINE_CCE_VF_SURFACE = DialectSurfaceSpec(
    dialect_id="inline_cce_vf",
    display_label="Inline CCE VF (tilelang.ascend.language)",
    required_headers_or_imports=(
        "import tilelang",
        "import tilelang.ascend.language as T",
    ),
    primitives=(
        PrimitiveEntry(
            "T.Kernel",
            "SCALAR",
            "grid_and_workspace_topology",
            "T.Kernel(blocks) as cid",
        ),
        PrimitiveEntry(
            "T.Pipelined",
            "SCALAR",
            "pipeline_and_sync_graph",
            "T.Pipelined(extent, num_stages>=2)",
        ),
        PrimitiveEntry(
            "T.SimdVF",
            "PIPE_V",
            "exec_mode_and_spr",
            "with T.SimdVF(): ... / T.simd.* register-resident intrinsics",
        ),
        PrimitiveEntry(
            "T.copy",
            "PIPE_MTE2",
            "dma_and_cache_hierarchy",
            "T.copy(gm_src, ub_dst)",
        ),
        PrimitiveEntry(
            "__no_simd_vf_fusion__",
            "PIPE_V",
            "exec_mode_and_spr",
            "__no_simd_vf_fusion__ __simd_vf__ (avoid LocalMemBar inside VF)",
        ),
    ),
    forbidden_tokens=(
        "tilelang.language",
        "is_npu=",
        "T.Scope",
        "LocalMemBar",
    ),
    prompt_notes=(
        "Keep outer T.Kernel, T.Pipelined, and GM<->UB T.copy in TileLang.",
        "Keep inner vector chains register-resident without intermediate UB roundtrips.",
    ),
)

CCE_CPP20_SURFACE = DialectSurfaceSpec(
    dialect_id="cce_cpp20",
    display_label="CCE C++20 (tl_templates/ascend)",
    required_headers_or_imports=(
        "#include <tl_templates/ascend/common.h>",
        "#include <tl_templates/ascend/gemm.h>",
        "#include <tl_templates/ascend/nd2nz_copy.h>",
        "#include <tl_templates/ascend/dcache_bypass.h>",
        "#include <tl_templates/ascend/reduce.h>",
        "#include <tl_templates/ascend/simd_inst.h>",
    ),
    primitives=(
        PrimitiveEntry(
            "asc_copy_gm2ub_align",
            "PIPE_MTE2",
            "dma_and_cache_hierarchy",
            (
                "asc_copy_gm2ub_align(dst_ub, src_gm, size_bytes) | "
                "(dst_ub, src_gm, burst_count, burst_len, left_pad, right_pad, src_gap, dst_gap) "
                "[3/8/10 args only; non-const __gm__ T*]"
            ),
        ),
        PrimitiveEntry(
            "asc_copy_ub2gm_align",
            "PIPE_MTE3",
            "dma_and_cache_hierarchy",
            (
                "asc_copy_ub2gm_align(dst_gm, src_ub, size_bytes) | "
                "(dst_gm, src_ub, burst_count, burst_len, left_pad, right_pad, src_gap, dst_gap) "
                "[3/7/8 args only; non-const __gm__ T*]"
            ),
        ),
        PrimitiveEntry(
            "asc_copy_gm2ub",
            "PIPE_MTE2",
            "dma_and_cache_hierarchy",
            "asc_copy_gm2ub(dst_ub, src_gm, burst_count, burst_len, src_stride, dst_stride)",
        ),
        PrimitiveEntry(
            "asc_copy_ub2gm",
            "PIPE_MTE3",
            "dma_and_cache_hierarchy",
            "asc_copy_ub2gm(dst_gm, src_ub, burst_count, burst_len, dst_stride, src_stride)",
        ),
        PrimitiveEntry(
            "get_block_idx",
            "SCALAR",
            "grid_and_workspace_topology",
            "get_block_idx()",
        ),
        PrimitiveEntry(
            "get_block_num",
            "SCALAR",
            "grid_and_workspace_topology",
            "get_block_num()",
        ),
        PrimitiveEntry(
            "get_subblockid",
            "SCALAR",
            "grid_and_workspace_topology",
            "get_subblockid()",
        ),
        PrimitiveEntry(
            "get_subblockdim",
            "SCALAR",
            "grid_and_workspace_topology",
            "get_subblockdim()",
        ),
        PrimitiveEntry(
            "asc_copy_gm2l1_nd2nz",
            "PIPE_MTE2",
            "dma_and_cache_hierarchy",
            "asc_copy_gm2l1_nd2nz(dst_l1, src_gm, ...)",
        ),
        PrimitiveEntry(
            "asc_copy_l12l0a",
            "PIPE_MTE1",
            "dma_and_cache_hierarchy",
            "asc_copy_l12l0a(dst_l0a, src_l1, ...)",
        ),
        PrimitiveEntry(
            "asc_copy_l12l0b_transpose",
            "PIPE_MTE1",
            "dma_and_cache_hierarchy",
            "asc_copy_l12l0b_transpose(dst_l0b, src_l1, ...)",
        ),
        PrimitiveEntry(
            "asc_copy_l0c2ub",
            "PIPE_FIX",
            "dma_and_cache_hierarchy",
            "asc_copy_l0c2ub(dst_ub, src_l0c, ...)",
        ),
        PrimitiveEntry(
            "asc_mmad",
            "PIPE_M",
            "alg_compute_and_monoid",
            "asc_mmad(c_l0c, a_l0a, b_l0b, m, k, n, 0, true, false, init_c)",
        ),
        PrimitiveEntry(
            "asc_sync_notify",
            "SCALAR",
            "pipeline_and_sync_graph",
            "asc_sync_notify(src_pipe, dst_pipe, event_id)",
        ),
        PrimitiveEntry(
            "asc_sync_wait",
            "SCALAR",
            "pipeline_and_sync_graph",
            "asc_sync_wait(src_pipe, dst_pipe, event_id)",
        ),
        PrimitiveEntry(
            "asc_lock",
            "SCALAR",
            "pipeline_and_sync_graph",
            "asc_lock(pipe, slot_id, ASC_LOCK_BLOCK)",
        ),
        PrimitiveEntry(
            "asc_unlock",
            "SCALAR",
            "pipeline_and_sync_graph",
            "asc_unlock(pipe, slot_id, ASC_LOCK_BLOCK)",
        ),
        PrimitiveEntry(
            "simd_inst::vlds_norm",
            "PIPE_V",
            "exec_mode_and_spr",
            "simd_inst::vlds_norm<T>(ub_ptr, offset)",
        ),
        PrimitiveEntry(
            "simd_inst::vsts_norm",
            "PIPE_V",
            "exec_mode_and_spr",
            "simd_inst::vsts_norm(vec, ub_ptr, offset, mask)",
        ),
        PrimitiveEntry(
            "simd_inst::vsts_pack_b32",
            "PIPE_V",
            "exec_mode_and_spr",
            "simd_inst::vsts_pack_b32(vec, ub_ptr, offset, mask)",
        ),
        PrimitiveEntry(
            "__simd_vf__",
            "PIPE_V",
            "exec_mode_and_spr",
            "__simd_vf__ inline void vf_fn(...)",
        ),
        PrimitiveEntry(
            "__simt_vf__",
            "PIPE_V",
            "exec_mode_and_spr",
            "__simt_vf__ __launch_bounds__(N) void simt_fn(...)",
        ),
    ),
    forbidden_tokens=(
        "kernel_operator.h",
        "using namespace AscendC;",
        "AscendC::",
        "TPipe",
        "TQue",
        "DataCopyPad",
        "FixpipeParamsArch3510",
        "copy_gm_to_ubuf",
        "copy_ubuf_to_gm",
        "set_flag(",
        "wait_flag(",
        "__shfl_down_sync",
        "__shfl_sync",
        "__shfl_xor_sync",
        "__ballot_sync",
        "__shared__",
        "__syncthreads",
        "threadIdx",
        "blockIdx",
        "blockDim",
        "gridDim",
        "get_subblocknum",
    ),
    prompt_notes=(
        "Compile target is Track-B CCE C++20 (-std=c++20 --npu-arch=dav-3510) via tl_templates.",
        "Do NOT emit high-level AscendC classes or legacy 910B raw CCE intrinsics.",
        "Every asc_sync_notify and asc_lock MUST pair symmetrically across all CFG paths.",
        (
            "The __simd_vf__ stack frame is exactly 0 bytes: never declare local C arrays "
            "(for example float tmp[64];) or runtime-indexed local structs inside a "
            "__simd_vf__ body; stage scratch in __ubuf__ pointer ranges or vector registers."
        ),
        (
            "Every __global__ GM pointer parameter must carry literal __gm__ T* directly in "
            "the signature (never via a using/typedef alias); in host extern \"C\" launchers "
            "cast void* arguments with C-style (__gm__ T*)ptr (never static_cast<__gm__ T*>)."
        ),
        (
            "Never call __builtin_expf, std::exp, or std::sqrt inside device kernels "
            "(ld.lld fails with undefined symbol). Inside __simt_vf__, call expf(x), sqrtf(x), "
            "rsqrtf(x), or logf(x) from <tl_templates/ascend/common.h> (simt_api/math_functions.h); "
            "inside __simd_vf__, call simd_inst::vexp(v, mask, MODE_ZEROING) or "
            "simd_inst::vsqrt(v, mask, MODE_ZEROING)."
        ),
        (
            "Helpers called from __simd_vf__ must be annotated __simd_callee__; helpers called "
            "from __simt_vf__ must be annotated __simt_callee__ (not bare __aicore__)."
        ),
        (
            "In Python entrypoints, guard any non-NPU CPU fallback inside "
            "if x.device.type != \"npu\":."
        ),
        (
            "PURE_SIMDVF_UBUF_FLAGS (--cce-disable-vf-stack-reserved-ubuf, "
            "--cce-disable-asc-reserved-ubuf) and raw integer UB pointer casts "
            "((__ubuf__ T*)0 or (__ubuf__ T*)offset) are ONLY valid in pure __simd_vf__ "
            "translation units with zero __simt_vf__ functions. In any TU defining "
            "__simt_vf__, omit PURE_SIMDVF_UBUF_FLAGS and never pass raw integer "
            "(__ubuf__ T*) offsets into __simt_vf__ (doing so overwrites the SIMT "
            "thread stack/data cache in UB and triggers vector core exception 507035); "
            "operate directly on __gm__ pointers or pass an explicit __gm__ scratch "
            "tensor allocated from Python instead."
        ),
        (
            "CUDA warp intrinsics (__shfl_down_sync, __shfl_sync, __shfl_xor_sync, "
            "__ballot_sync, warpSize) do not exist on Ascend dav-3510; use explicit "
            "shared scratch or __simd_vf__ vector reductions instead."
        ),
        (
            "Cast away const on __gm__ T* and __ubuf__ T* arguments passed to "
            "asc_copy_gm2ub* / asc_copy_ub2gm* (passing const __gm__ T* fails bisheng "
            "overload resolution), and never form negative base offsets (src_gm - pad) "
            "before a tensor allocation."
        ),
        (
            "Never mix PIPE_S scalar stores (dst_gm[i] = v) with PIPE_MTE3 asc_copy_ub2gm* "
            "to the same output __gm__ buffer without explicit cache flushing: dirty 64B "
            "Scalar DCache lines overwrite DMA stores with stale NaN bits. Route unaligned "
            "tail stores through byte-granular asc_copy_ub2gm_align."
        ),
        (
            "Invoke __simt_vf__ / __simd_vf__ functions inside __aicore__ kernels via "
            "asc_vf_call<fn>(...), never <<<...>>> (chevron syntax is valid only in Host "
            "extern \"C\" launchers calling __global__ kernels); query sub-cores with "
            "get_subblockdim() and get_subblockid(), not get_subblocknum()."
        ),
    ),
)

TL_TEMPLATES_CPP20_SURFACE = CCE_CPP20_SURFACE
INLINE_ASCENDC_VF_SURFACE = INLINE_CCE_VF_SURFACE

_TRACKB_CCE_ALIASES = (
    "cce_cpp20",
    "cce_trackb",
    "hybrid_trackb",
    "tl_templates_cpp20",
    "ascendc_trackb",
    "ascendc_template",
)


def resolve_trackb_cce_surface(backend_or_tier: str) -> DialectSurfaceSpec:
    """Resolve the verified CCE C++20 surface while keeping legacy aliases compatible."""
    norm = str(backend_or_tier or "cce_cpp20").strip().lower()
    if norm in _TRACKB_CCE_ALIASES:
        return CCE_CPP20_SURFACE
    raise ValueError(f"Unsupported Track-B CCE dialect identifier: {backend_or_tier!r}")


def get_dialect_surface(backend_or_tier: str) -> DialectSurfaceSpec:
    """Resolve the verified DialectSurfaceSpec for a backend or escalation tier."""
    norm = str(backend_or_tier or "tilelang_dsl").strip().lower()
    if norm in _TRACKB_CCE_ALIASES or norm == "ascendc":
        return CCE_CPP20_SURFACE
    if norm in ("inline_cce_vf", "inline_ascendc_vf"):
        return INLINE_CCE_VF_SURFACE
    return TILELANG_DSL_SURFACE


__all__ = [
    "CCE_CPP20_SURFACE",
    "DialectId",
    "DialectSurfaceSpec",
    "INLINE_ASCENDC_VF_SURFACE",
    "INLINE_CCE_VF_SURFACE",
    "PrimitiveEntry",
    "TILELANG_DSL_SURFACE",
    "TL_TEMPLATES_CPP20_SURFACE",
    "get_dialect_surface",
    "resolve_trackb_cce_surface",
]
