"""Single source of truth for Ascend TileLang DSL and tl_templates C++20 surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DialectId = Literal["tilelang_dsl", "inline_ascendc_vf", "tl_templates_cpp20"]


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
    required_headers_or_imports: tuple[str, ...]
    primitives: tuple[PrimitiveEntry, ...]
    forbidden_tokens: tuple[str, ...]
    prompt_notes: tuple[str, ...]

    def render_prompt_reference(self) -> str:
        """Render markdown bullet list of callable APIs, headers, and forbidden tokens."""
        lines = [
            f"Verified `{self.dialect_id}` callable surface:",
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


TILELANG_DSL_SURFACE = DialectSurfaceSpec(
    dialect_id="tilelang_dsl",
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
    ),
    forbidden_tokens=(
        "tilelang.language",
        "is_npu=",
        "threads=",
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
    ),
)

INLINE_ASCENDC_VF_SURFACE = DialectSurfaceSpec(
    dialect_id="inline_ascendc_vf",
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

TL_TEMPLATES_CPP20_SURFACE = DialectSurfaceSpec(
    dialect_id="tl_templates_cpp20",
    required_headers_or_imports=(
        '#include "tl_templates/ascend/common.h"',
        '#include "tl_templates/ascend/gemm.h"',
        '#include "tl_templates/ascend/simd_inst.h"',
    ),
    primitives=(
        PrimitiveEntry(
            "asc_copy_gm2ub_align",
            "PIPE_MTE2",
            "dma_and_cache_hierarchy",
            "asc_copy_gm2ub_align(dst_ub, src_gm, rows, row_bytes, ...)",
        ),
        PrimitiveEntry(
            "asc_copy_ub2gm_align",
            "PIPE_MTE3",
            "dma_and_cache_hierarchy",
            "asc_copy_ub2gm_align(dst_gm, src_ub, rows, row_bytes, ...)",
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
        "AscendC::DataCopyPad",
        "FixpipeParamsArch3510",
    ),
    prompt_notes=(
        "Compile target is C++20 (-std=c++20 --npu-arch=dav-3510) against tl_templates/ascend.",
        "Every asc_sync_notify and asc_lock MUST pair symmetrically across all CFG paths.",
    ),
)


def get_dialect_surface(backend_or_tier: str) -> DialectSurfaceSpec:
    """Resolve the verified DialectSurfaceSpec for a backend or escalation tier."""
    norm = str(backend_or_tier or "tilelang_dsl").strip().lower()
    if norm in ("ascendc_template", "ascendc_trackb", "hybrid_trackb", "tl_templates_cpp20"):
        return TL_TEMPLATES_CPP20_SURFACE
    if norm == "inline_ascendc_vf":
        return INLINE_ASCENDC_VF_SURFACE
    return TILELANG_DSL_SURFACE


__all__ = [
    "DialectId",
    "DialectSurfaceSpec",
    "INLINE_ASCENDC_VF_SURFACE",
    "PrimitiveEntry",
    "TILELANG_DSL_SURFACE",
    "TL_TEMPLATES_CPP20_SURFACE",
    "get_dialect_surface",
]
