"""Ascend TileLang DSL and Track-B CCE C++20 surface definitions.

Active SDK target is cann_9_3 (dav-3510) exclusively; cann_9_2 is an inert placeholder.
"""

from __future__ import annotations

import functools
import os
import re
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DialectId = Literal["tilelang_dsl", "inline_cce_vf", "cce_cpp20"]
VALID_CANN_SERIES: tuple[str, ...] = ("cann_9_2", "cann_9_3")
DEFAULT_CANN_SERIES: str = "cann_9_3"
_CANN_SURFACE_DIR = Path(__file__).resolve().parent / "cann_surface"

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
    cann_series: tuple[str, ...] = VALID_CANN_SERIES
    pipes: tuple[str, ...] = ()
    facets: tuple[str, ...] = ()
    required_headers: tuple[str, ...] = ()
    pitfall: str = ""
    symptoms: tuple[str, ...] = ()


LOWERING_GUARDRAILS: tuple[str, ...] = (
    (
        "G-L1 (TIR Boolean): Ban Python `and`/`or`/`not` on TIR `PrimExpr`; "
        "require `tirx.all(...)` / `tirx.any(...)`."
    ),
    (
        "G-L2 (SimtVF Sub-Word Casts): Inside `T.SimtVF`, ban sub-word vector casts "
        "(`int8x2`/`int8x4`); widen to `float32` scalar or aligned register first before "
        "scalar or 32B-aligned writeback."
    ),
    (
        "G-L3 (Track-B BF16 Math): In Track-B CCE C++20, ban direct `+ - * /` or `sqrtf` "
        "on `__bf16`/`bfloat16_t`; cast explicitly to `float` first then convert back."
    ),
    (
        "G-L4 (Wide Vector Store): Ban wide vector `BufferStore` (e.g., `float32x64`) "
        "on scalar indices; use `T.copy` for tile moves and keep `T.SimtVF` lane width <= 8."
    ),
)

_SYMPTOM_FACET_MAP: dict[str, tuple[str, ...]] = {
    "DMA_REDUNDANT_GM_ROUNDTRIP": (
        "dma_and_cache_hierarchy",
        "sram_layout_and_aliasing",
        "pipeline_and_sync_graph",
    ),
    "DMA_UNALIGNED_SHORT_BURST": (
        "tile_and_stage_knobs",
        "dma_and_cache_hierarchy",
    ),
    "DMA_L2_THRASH_REUSE_MISS": (
        "grid_and_workspace_topology",
        "dma_and_cache_hierarchy",
    ),
    "COMPUTE_VECTOR_LOOP_DEGRADED": (
        "exec_mode_and_spr",
        "alg_compute_and_monoid",
    ),
    "COMPUTE_INEFFECTIVE_THROUGHPUT": (
        "exec_mode_and_spr",
        "tile_and_stage_knobs",
    ),
    "COMPUTE_FIXPIPE_ACCUM_IMBALANCE": (
        "tile_and_stage_knobs",
        "sram_layout_and_aliasing",
    ),
    "COMPUTE_1C2V_SUBCORE_IMBALANCE": (
        "grid_and_workspace_topology",
        "pipeline_and_sync_graph",
    ),
    "SCALAR_LOOP_TRAP": (
        "exec_mode_and_spr",
        "dma_and_cache_hierarchy",
    ),
    "SRAM_REGISTER_SPILL_STACK": (
        "tile_and_stage_knobs",
        "exec_mode_and_spr",
    ),
    "SRAM_UB_L1_BANK_CONFLICT": ("sram_layout_and_aliasing",),
    "SYNC_PSEUDO_DOUBLE_BUFFER": (
        "pipeline_and_sync_graph",
        "sram_layout_and_aliasing",
    ),
    "SYNC_BARRIER_SERIALIZATION": (
        "pipeline_and_sync_graph",
        "tile_and_stage_knobs",
    ),
    "GRID_TAIL_WAVE_UNDERUTIL": (
        "grid_and_workspace_topology",
        "tile_and_stage_knobs",
    ),
}

_CRITICAL_SIGNATURE_SYMBOLS: tuple[str, ...] = (
    "asc_copy_l0c2ub",
    "ascend_gemm_l1",
    "asc_copy_l0c2gm",
    "asc_mmad",
    "asc_copy_gm2l1_nd2nz",
    "asc_copy_gm2ub_align",
    "asc_copy_ub2gm_align",
    "asc_vf_call",
    "__simt_vf__",
    "T.MixedKernel",
    "T.Kernel",
    "T.gemm",
    "T.dual_copy",
    "T.SimdVF",
    "T.SimtVF",
    "T.copy",
)


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
            if entry.pitfall:
                lines.append(
                    f"- `{entry.signature}` [{entry.pipe} | facet={entry.facet}] -- {entry.pitfall}"
                )
            else:
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
            "T.alloc_var",
            "SCALAR",
            "exec_mode_and_spr",
            "T.alloc_var(\"float32\" | \"int32\")",
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
            "T.copy(src, dst) | T.copy(src, dst, transpose=True)",
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
            (
                "T.gemm(a_l1, b_l1, c_l0c, transpose_B=True, clear_accum=True) "
                "[NT only: trans_A=False, trans_B=True]"
            ),
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
            "T.simd.alloc_var",
            "PIPE_V",
            "exec_mode_and_spr",
            "T.simd.alloc_var(\"float32\")",
        ),
        PrimitiveEntry(
            "T.simd.alloc_local",
            "PIPE_V",
            "exec_mode_and_spr",
            "T.simd.alloc_local(shape, \"float32\")",
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
            "T.simd.vdupv",
            "PIPE_V",
            "exec_mode_and_spr",
            "T.simd.vdupv(vec, mask=mask, pos=\"POS_LOWEST\")",
        ),
        PrimitiveEntry(
            "T.simd.vld",
            "PIPE_V",
            "exec_mode_and_spr",
            "T.simd.vld(buf[offset], dist=\"NORM_B32\" | \"US_B16\" | \"BRC_B32\")",
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
            "T.simd.vmula",
            "PIPE_V",
            "alg_compute_and_monoid",
            "T.simd.vmula(acc, lhs, rhs, mask=mask, mode=\"MODE_MERGING\")",
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
            "T.simd.vmins",
            "PIPE_V",
            "alg_compute_and_monoid",
            "T.simd.vmins(vec, scalar, mask=mask)",
        ),
        PrimitiveEntry(
            "T.simd.vneg",
            "PIPE_V",
            "alg_compute_and_monoid",
            "T.simd.vneg(vec, mask=mask)",
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
            "T.simd.vcmin",
            "PIPE_V",
            "alg_compute_and_monoid",
            "T.simd.vcmin(vec, mask=mask)",
        ),
        PrimitiveEntry(
            "T.simd.vsel",
            "PIPE_V",
            "exec_mode_and_spr",
            "T.simd.vsel(src0, src1, mask)",
        ),
        PrimitiveEntry(
            "T.simd.vcmp",
            "PIPE_V",
            "exec_mode_and_spr",
            "T.simd.vcmp(lhs, rhs, mask=mask, op=\"lt\"|\"le\"|\"gt\"|\"ge\"|\"eq\"|\"ne\")",
        ),
        PrimitiveEntry(
            "T.simd.vcmps",
            "PIPE_V",
            "exec_mode_and_spr",
            "T.simd.vcmps(vec, scalar, mask=mask, op=\"lt\"|\"le\"|\"gt\"|\"ge\"|\"eq\"|\"ne\")",
        ),
        PrimitiveEntry(
            "T.simd.vci",
            "PIPE_V",
            "exec_mode_and_spr",
            "T.simd.vci(base_idx, \"int32\")",
        ),
        PrimitiveEntry(
            "T.simd.vsts",
            "PIPE_V",
            "exec_mode_and_spr",
            (
                "T.simd.vsts(buf[offset], vec, mask=mask, "
                "dist=\"NORM_B32\" | \"ONEPT_B32\" | \"PK_B32\", extent=64)"
            ),
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
        (
            "Inside `with T.SimdVF():`, never apply Python +, -, *, / operators to SIMD vector "
            "registers; always use T.simd.vadd/vsub/vmul/vdiv/vadds/vmuls/vmula."
        ),
        (
            "T.simd.vcadd/vcmax/vcmin return a 64-lane vector (`float32x64`) whose lane 0 holds "
            "the reduced value: never store them into a scalar buffer element (`buf[i] = ...`) "
            "or pass them to scalar T.simd.vdup(...); broadcast lane 0 via "
            "T.simd.vdupv(reduced, mask=m32, pos=\"POS_LOWEST\"), or store lane 0 via "
            "T.simd.vsts(buf[off], reduced, mask=m32, dist=\"ONEPT_B32\") and reload via "
            "T.simd.vld(buf[off], dist=\"BRC_B32\")."
        ),
        (
            "Never reassign plain Python variables across `if`/`for` branches inside `@T.prim_func`; "
            "declare mutable registers with T.simd.alloc_var(\"float32\") or "
            "T.simd.alloc_local(N, \"float32\") (and mutable scalars with T.alloc_var(...)) "
            "before the branch/loop, using mode=\"MODE_MERGING\" on masked tail updates."
        ),
        (
            "For `rem < 64` fp16/bf16 tails loaded with US_B16 and converted via "
            "T.simd.vcvt(..., mask=m16_t, part=0) or stored with PK_B32, use "
            "`m16_t = T.simd.update_mask(rem * 2, width=16)` alongside "
            "`m32_t = T.simd.update_mask(rem, width=32)`."
        ),
        (
            "Ascend L1 T.gemm only supports transpose_A=False, transpose_B=True (NT); when operand "
            "B is row-major (K, N) in GM, copy it transposed into (N, K) L1 with "
            "T.copy(B_gm[...], b_l1, transpose=True) before T.gemm(a_l1, b_l1, c_l0c, "
            "transpose_B=True, clear_accum=...)."
        ),
        (
            "Inside `with T.SimdVF():`, store vector register results to UB exclusively via "
            "`T.simd.vsts(...)` (never assign a vector expression directly like `ub_buf[off] = vec_expr`)."
        ),
        (
            "Keep UB float32 scalar-slot offsets 32-byte aligned (`off % 8 == 0`, e.g., `(row, 8)` "
            "stride rows) whenever storing with `dist=\"ONEPT_B32\"` or loading with `dist=\"BRC_B32\"`."
        ),
        (
            "Inside `with T.MixedKernel(blocks, sids=2) as (bx, sid):`, every `UB -> GM` `T.copy` "
            "to a shared GM/workspace buffer must write disjoint `sid`-partitioned slices "
            "(e.g., `sid * half_m : (sid + 1) * half_m`, paired with `T.assume(...)` when needed) "
            "so AIV0 and AIV1 never collide on `PIPE_MTE3`."
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

_BASE_CCE_CPP20_SURFACE = DialectSurfaceSpec(
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
            "ascend_gemm_l1",
            "PIPE_M",
            "alg_compute_and_monoid",
            (
                "ascend_gemm_l1<M, K, N, TILE_K_SUB, TRANS_B>(cc_ptr, cbuf_a_ptr, "
                "cbuf_b_ptr, clear_accum) [gemm.h; pre-unlock PIPE_MTE1 0/1, drain PIPE_M 0/1]"
            ),
        ),
        PrimitiveEntry(
            "asc_mmad",
            "PIPE_M",
            "alg_compute_and_monoid",
            "asc_mmad(c_l0c, a_l0a, b_l0b, m, k, n, asc_unit_flag_mode::DISABLE, true, false, init_c)",
        ),
        PrimitiveEntry(
            "asc_sync_notify",
            "SCALAR",
            "pipeline_and_sync_graph",
            (
                "asc_sync_notify(src_pipe, dst_pipe, EVENT_ID0) "
                "[3rd arg is event_t EVENT_ID0..EVENT_ID7, never raw int]"
            ),
        ),
        PrimitiveEntry(
            "asc_sync_wait",
            "SCALAR",
            "pipeline_and_sync_graph",
            (
                "asc_sync_wait(src_pipe, dst_pipe, EVENT_ID0) "
                "[3rd arg is event_t EVENT_ID0..EVENT_ID7, never raw int]"
            ),
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
            "simd_inst::vdupv",
            "PIPE_V",
            "exec_mode_and_spr",
            "simd_inst::vdupv(vec, mask, POS_LOWEST, MODE_ZEROING)",
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
        PrimitiveEntry(
            "asc_create_mask_b32",
            "PIPE_V",
            "exec_mode_and_spr",
            "vector_bool m32 = asc_create_mask_b32(PAT_ALL)",
        ),
        PrimitiveEntry(
            "asc_create_mask_b16",
            "PIPE_V",
            "exec_mode_and_spr",
            "vector_bool m16 = asc_create_mask_b16(PAT_ALL)",
        ),
        PrimitiveEntry(
            "simd_inst::update_mask_b32",
            "PIPE_V",
            "exec_mode_and_spr",
            "vector_bool m32_tail = simd_inst::update_mask_b32(rem)",
        ),
        PrimitiveEntry(
            "simd_inst::update_mask_b16",
            "PIPE_V",
            "exec_mode_and_spr",
            "vector_bool m16_tail = simd_inst::update_mask_b16(rem * 2)",
        ),
        PrimitiveEntry(
            "simd_inst::vdup",
            "PIPE_V",
            "exec_mode_and_spr",
            "simd_inst::vdup<T>(scalar, mask, MODE_ZEROING)",
        ),
        PrimitiveEntry(
            "simd_inst::vcvt",
            "PIPE_V",
            "exec_mode_and_spr",
            "simd_inst::vcvt<DstT>(src_vec, mask, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING)",
        ),
        PrimitiveEntry(
            "simd_inst::vlds_brc_elem",
            "PIPE_V",
            "exec_mode_and_spr",
            "simd_inst::vlds_brc_elem<T>(ub_ptr, offset)",
        ),
        PrimitiveEntry(
            "simd_inst::vsts_1st",
            "PIPE_V",
            "exec_mode_and_spr",
            "simd_inst::vsts_1st(vec, ub_ptr, offset, mask)",
        ),
        PrimitiveEntry(
            "simd_inst::vadd",
            "PIPE_V",
            "alg_compute_and_monoid",
            "simd_inst::vadd(lhs, rhs, mask, MODE_ZEROING)",
        ),
        PrimitiveEntry(
            "asc_shfl",
            "PIPE_V",
            "grid_and_workspace_topology",
            "asc_shfl(var, src_lane, warpSize) [inside __simt_vf__ only]",
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
        "Do NOT emit high-level AscendC classes or legacy non-dav-3510 CCE intrinsics.",
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
        (
            "Pass event_t enum constants (EVENT_ID0..EVENT_ID7, never integer literals) to "
            "asc_sync_notify/asc_sync_wait; prefer ascend_gemm_l1 from "
            "<tl_templates/ascend/gemm.h> or asc_mmad(..., asc_unit_flag_mode::DISABLE, ...) "
            "for Cube GEMM."
        ),
        (
            "Inside __simd_vf__ loops on dav-3510, keep all floating-point arithmetic in "
            "simd_inst::* intrinsics and avoid scalar float math, casts, or scalar UB "
            "element accesses inside vector loops (`bisheng` rejects unsupported scalar "
            "instructions in AIV loops)."
        ),
        (
            "Never use pge_b32 as a true mask or invent nonexistent simd_inst::* or asc_* "
            "names outside the verified surface above; use asc_create_mask_b32(PAT_ALL) / "
            "asc_create_mask_b16(PAT_ALL) for full masks and simd_inst::update_mask_b32(rem) / "
            "simd_inst::update_mask_b16(rem * 2) for tail masks."
        ),
        (
            "Every __simd_vf__ function must return void (for example "
            "`__simd_vf__ inline void vf_fn(...)`), never returning a vector or scalar type."
        ),
        (
            "Parallelize __simt_vf__ across >=32 threads (`__launch_bounds__(128)` and "
            "`dim3(128, 1, 1)`) using `Simt::GetThreadIdx<0>()` (never `threadIdx` and "
            "never single-lane `__launch_bounds__(1)`)."
        ),
    ),
)


def _normalize_cann_series_from_path(raw: str) -> str | None:
    text = raw.strip()
    if not text:
        return None
    if "9.2" in text or "cann_9_2" in text:
        return "cann_9_2"
    if "9.3" in text or "9.4" in text or "cann_9_3" in text:
        return "cann_9_3"
    cand = Path(text).expanduser()
    for ver_file in (cand / "version.info", cand / "ascend_toolkit_install.info"):
        if ver_file.is_file():
            content = ver_file.read_text(encoding="utf-8", errors="ignore")
            if "9.2." in content:
                return "cann_9_2"
            if "9.3." in content or "9.4." in content:
                return "cann_9_3"
    return None


def _clamp_implicit_cann_series(series: str) -> str:
    return DEFAULT_CANN_SERIES if series == "cann_9_2" else series


def resolve_cann_series(
    explicit: str | None = None,
    *,
    cann_home: str | Path | None = None,
) -> str:
    """Resolve CANN snapshot series, clamping implicit/env/path cann_9_2 to cann_9_3."""
    if explicit is not None and str(explicit).strip():
        norm = str(explicit).strip().lower()
        if norm not in VALID_CANN_SERIES:
            raise ValueError(
                f"Unsupported CANN series {explicit!r}; expected one of {VALID_CANN_SERIES}"
            )
        return norm
    for env_key in ("TILEFOUNDRY_CANN_SERIES", "TILEFOUNDRY_NPU_CANN_SERIES"):
        raw = os.environ.get(env_key, "").strip().lower()
        if raw:
            if raw not in VALID_CANN_SERIES:
                raise ValueError(
                    f"Unsupported CANN series {raw!r} from {env_key}; "
                    f"expected one of {VALID_CANN_SERIES}"
                )
            return _clamp_implicit_cann_series(raw)
    if cann_home is not None:
        inferred = _normalize_cann_series_from_path(str(cann_home))
        if inferred is not None:
            return _clamp_implicit_cann_series(inferred)
    for env_key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        inferred = _normalize_cann_series_from_path(os.environ.get(env_key, ""))
        if inferred is not None:
            return _clamp_implicit_cann_series(inferred)
    return DEFAULT_CANN_SERIES


@functools.lru_cache(maxsize=4)
def _load_cached_cann_surface(series: str) -> DialectSurfaceSpec:
    snap_path = _CANN_SURFACE_DIR / f"{series}.toml"
    data = tomllib.loads(snap_path.read_text(encoding="utf-8"))
    meta = data.get("surface", {})
    raw_prims = data.get("primitives", [])
    primitives = tuple(
        PrimitiveEntry(
            symbol=str(item["symbol"]),
            pipe=str(item["pipe"]),
            facet=str(item["facet"]),
            signature=str(item["signature"]),
            cann_series=(series,),
            pipes=tuple(str(p) for p in item.get("pipes", ())),
            facets=tuple(str(f) for f in item.get("facets", ())),
            required_headers=tuple(str(h) for h in item.get("required_headers", ())),
            pitfall=str(item.get("pitfall", "")),
            symptoms=tuple(str(s) for s in item.get("symptoms", ())),
        )
        for item in raw_prims
    )
    headers = tuple(str(h) for h in meta.get("umbrella_headers", ()))
    toml_forbidden = tuple(str(t) for t in meta.get("forbidden_tokens", ()))
    merged_forbidden = tuple(
        dict.fromkeys([*toml_forbidden, *_BASE_CCE_CPP20_SURFACE.forbidden_tokens])
    )
    return DialectSurfaceSpec(
        dialect_id="cce_cpp20",
        display_label=str(meta.get("display_label", f"CCE C++20 ({series})")),
        required_headers_or_imports=headers or _BASE_CCE_CPP20_SURFACE.required_headers_or_imports,
        primitives=primitives,
        forbidden_tokens=merged_forbidden,
        prompt_notes=_BASE_CCE_CPP20_SURFACE.prompt_notes,
    )


def load_cann_surface(cann_series: str | None = None) -> DialectSurfaceSpec:
    """Load a self-contained dav-3510 C++20 DialectSurfaceSpec from TOML snapshot resources.

    CCE_CPP20_SURFACE binds to DEFAULT_CANN_SERIES ("cann_9_3"); cann_9_2 is inert.
    """
    return _load_cached_cann_surface(resolve_cann_series(cann_series))


CCE_CPP20_SURFACE = load_cann_surface(DEFAULT_CANN_SERIES)
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


def resolve_trackb_cce_surface(
    backend_or_tier: str,
    *,
    cann_series: str | None = None,
) -> DialectSurfaceSpec:
    """Resolve the verified CCE C++20 surface for the selected CANN series."""
    norm = str(backend_or_tier or "cce_cpp20").strip().lower()
    if norm in _TRACKB_CCE_ALIASES:
        return _load_cached_cann_surface(resolve_cann_series(cann_series))
    raise ValueError(f"Unsupported Track-B CCE dialect identifier: {backend_or_tier!r}")


def get_dialect_surface(
    backend_or_tier: str,
    *,
    cann_series: str | None = None,
) -> DialectSurfaceSpec:
    """Resolve the verified DialectSurfaceSpec for a backend or escalation tier."""
    norm = str(backend_or_tier or "tilelang_dsl").strip().lower()
    if norm in _TRACKB_CCE_ALIASES or norm == "ascendc":
        return _load_cached_cann_surface(resolve_cann_series(cann_series))
    if norm in ("inline_cce_vf", "inline_ascendc_vf"):
        return INLINE_CCE_VF_SURFACE
    return TILELANG_DSL_SURFACE


def _is_cube_or_coupled_entry(entry: PrimitiveEntry) -> bool:
    if entry.pipe in ("PIPE_M", "PIPE_MTE1", "PIPE_FIX", "L1", "L0A", "L0B", "L0C"):
        return True
    if any(
        p in ("pipe_m_cube", "pipe_mte1_l1_to_l0", "pipe_fix_fixpipe") for p in entry.pipes
    ):
        return True
    sym_l = entry.symbol.lower()
    return any(
        tok in sym_l
        for tok in (
            "gemm",
            "mmad",
            "_l1",
            "2l1",
            "l12l0",
            "l0a",
            "l0b",
            "l0c",
            "dual_copy",
            "mixedkernel",
            "sync_block_",
        )
    )


def _is_coupled_only_entry(entry: PrimitiveEntry) -> bool:
    sym_l = entry.symbol.lower()
    return any(
        tok in sym_l
        for tok in ("mixedkernel", "dual_copy", "sync_block_", "l0c2ub", "ub2l1")
    )


def _is_simt_entry(entry: PrimitiveEntry) -> bool:
    sym_l = entry.symbol.lower()
    if "simt" in sym_l or entry.symbol in ("asc_vf_call", "asc_shfl", "asc_reduce_add"):
        return True
    return "__simt_vf__" in entry.pitfall.lower() or "__simt_vf__" in entry.signature.lower()


def _is_aiv_vector_entry(entry: PrimitiveEntry) -> bool:
    if _is_simt_entry(entry):
        return True
    if entry.pipe in ("PIPE_V", "UB") or "pipe_v_simd_simt" in entry.pipes:
        return True
    sym_l = entry.symbol.lower()
    return any(tok in sym_l for tok in ("gm2ub", "ub2gm", "ub2ub", "simdvf", "alloc_shared"))


def _core_symbols_for(
    dialect_id: str,
    topology_axes: "TopologyAxes | None",
    simt_allowed: bool,
    is_subatom: bool,
) -> set[str]:
    engine = topology_axes.compute_engine if topology_axes is not None else "aiv_only"
    ring = topology_axes.memory_ring if topology_axes is not None else ""
    reg = topology_axes.iteration_regularity if topology_axes is not None else ""
    want_simt = simt_allowed and (
        topology_axes is None or reg == "irregular_simt" or is_subatom
    )
    if dialect_id in ("tilelang_dsl", "inline_cce_vf"):
        core = {"T.Kernel", "T.Pipelined", "T.serial", "T.copy"}
        if engine == "aic_only":
            core.update({"T.alloc_l1", "T.alloc_l0a", "T.alloc_l0b", "T.alloc_l0c", "T.gemm"})
        elif engine == "coupled_1c2v":
            core.update(
                {
                    "T.MixedKernel",
                    "T.alloc_l1",
                    "T.alloc_l0c",
                    "T.alloc_shared",
                    "T.gemm",
                    "T.dual_copy",
                    "T.SimdVF",
                }
            )
            if ring in ("persistent_onchip_state", "bidirectional_cv_ring"):
                core.add("T.alloc_ss")
        else:
            core.update({"T.alloc_shared", "T.SimdVF", "T.simd.vld", "T.simd.vsts", "T.fill"})
        if want_simt and engine != "aic_only":
            core.add("T.SimtVF")
        return core
    if engine == "aic_only":
        return {
            "asc_copy_gm2l1_nd2nz",
            "asc_copy_l12l0a",
            "asc_copy_l12l0b",
            "ascend_gemm_l1",
            "asc_mmad",
            "asc_copy_l0c2gm",
            "asc_sync_notify",
            "asc_sync_wait",
            "asc_lock",
            "asc_unlock",
        }
    if engine == "coupled_1c2v":
        core = {
            "asc_copy_gm2l1_nd2nz",
            "ascend_gemm_l1",
            "asc_mmad",
            "asc_copy_l0c2ub",
            "asc_copy_gm2ub_align",
            "asc_copy_ub2gm_align",
            "asc_sync_notify",
            "asc_sync_wait",
            "asc_sync_block_arrive",
            "asc_sync_block_wait",
            "asc_lock",
            "asc_unlock",
        }
        if want_simt:
            core.update({"asc_vf_call", "asc_shfl", "asc_reduce_add", "__simt_vf__"})
        return core
    core = {
        "asc_copy_gm2ub",
        "asc_copy_gm2ub_align",
        "asc_copy_ub2gm",
        "asc_copy_ub2gm_align",
        "asc_sync_notify",
        "asc_sync_wait",
        "asc_create_mask_b32",
        "simd_inst::vlds_norm",
        "simd_inst::vsts_1st",
        "simd_inst::vadd",
    }
    if want_simt:
        core.update({"asc_vf_call", "asc_shfl", "asc_reduce_add", "__simt_vf__"})
    return core


def filter_primitives_on_demand(
    surface: DialectSurfaceSpec,
    *,
    track: str = "both",
    topology_axes: "TopologyAxes | None" = None,
    freedom_mask: "ConstraintRelaxationMask | None" = None,
    active_symptom: str | None = None,
    case_ratios: "CasePhysicsRatios | None" = None,
) -> tuple[PrimitiveEntry, ...]:
    """Run B2-fixed two-stage on-demand primitive pruning over a DialectSurfaceSpec."""
    unlocked = (
        None
        if (freedom_mask is None or freedom_mask.is_unrestricted_top)
        else set(freedom_mask.unlocked_facet_names())
    )
    is_subatom = case_ratios is not None and case_ratios.contig_atom_ratio < 1.0
    track_norm = str(track or "both").strip().lower()
    simt_allowed = (
        topology_axes is None
        or topology_axes.iteration_regularity == "irregular_simt"
        or is_subatom
        or unlocked is None
        or "exec_mode_and_spr" in unlocked
        or track_norm in ("simt_vf", "simtvf")
    )
    engine = topology_axes.compute_engine if topology_axes is not None else ""
    if engine == "aic_only":
        simt_allowed = False
    core_symbols = _core_symbols_for(
        surface.dialect_id,
        topology_axes,
        simt_allowed,
        is_subatom,
    )

    stage1: list[PrimitiveEntry] = []
    for entry in surface.primitives:
        if engine == "aiv_only" and _is_cube_or_coupled_entry(entry):
            continue
        if engine == "aic_only" and (_is_aiv_vector_entry(entry) or _is_coupled_only_entry(entry)):
            continue
        if not simt_allowed and _is_simt_entry(entry):
            continue
        if unlocked is not None:
            entry_facets = {entry.facet, *entry.facets}
            if not (entry_facets & unlocked) and entry.symbol not in core_symbols:
                continue
        stage1.append(entry)

    if not stage1 and surface.primitives:
        stage1 = list(surface.primitives)

    if active_symptom is None or not str(active_symptom).strip():
        return tuple(stage1)

    sym_key = str(active_symptom).strip().upper()
    direct = [
        entry
        for entry in stage1
        if any(s.strip().upper() == sym_key for s in entry.symptoms)
    ]
    if direct:
        return tuple(direct)
    target_facets = set(_SYMPTOM_FACET_MAP.get(sym_key, ()))
    if target_facets:
        facet_matched = [
            entry
            for entry in stage1
            if ({entry.facet, *entry.facets} & target_facets)
        ]
        if facet_matched:
            return tuple(facet_matched)
    return tuple(stage1)


def _priority_sort_key(entry: PrimitiveEntry) -> tuple[int, str]:
    try:
        return (_CRITICAL_SIGNATURE_SYMBOLS.index(entry.symbol), entry.symbol)
    except ValueError:
        return (len(_CRITICAL_SIGNATURE_SYMBOLS), entry.symbol)


def _note_relevant(
    note: str,
    entries: Sequence[PrimitiveEntry],
    has_simt: bool,
    has_cube: bool,
) -> bool:
    low = note.lower()
    if "do not emit" in low:
        return True
    if ("__simt_vf__" in low or "t.simtvf" in low) and not has_simt:
        return False
    if any(tok in low for tok in ("t.gemm", "ascend_gemm_l1", "asc_mmad", "t.mixedkernel")):
        if not has_cube:
            return False
    return True


def render_layer2_surface(
    surface: DialectSurfaceSpec,
    *,
    primitives: Sequence[PrimitiveEntry] | None = None,
    include_guardrails: bool = True,
    max_chars: int = 10000,
) -> str:
    """Render compact Layer-2 dialect markdown guaranteed <= max_chars."""
    entries = tuple(primitives) if primitives is not None else surface.primitives
    header_lines = [
        f"Verified `{surface.display_label}` (`{surface.dialect_id}`) callable surface:",
        f"- Imports / Headers: {', '.join(surface.required_headers_or_imports)}",
    ]
    tail_lines: list[str] = []
    if surface.forbidden_tokens:
        joined = ", ".join(f"`{tok}`" for tok in surface.forbidden_tokens)
        tail_lines.append(f"- Strictly Forbidden Symbols: {joined}")
    if include_guardrails:
        tail_lines.append("- Lowering Guardrails:")
        for rule in LOWERING_GUARDRAILS:
            tail_lines.append(f"  * {rule}")

    has_simt = any(_is_simt_entry(e) for e in entries)
    has_cube = any(_is_cube_or_coupled_entry(e) for e in entries)
    relevant_notes = [
        note
        for note in surface.prompt_notes
        if _note_relevant(note, entries, has_simt, has_cube)
    ]
    prioritized_notes = sorted(
        relevant_notes,
        key=lambda n: (
            0
            if "Do NOT emit" in n
            else (
                1
                if ("__simt_vf__" in n or "T.SimtVF" in n)
                else (2 if len(n) <= 180 else 3)
            )
        ),
    )
    must_keep_notes = [n for n in prioritized_notes if "Do NOT emit" in n]
    optional_notes = [n for n in prioritized_notes if "Do NOT emit" not in n]
    must_keep_lines = [f"- {n}" for n in must_keep_notes]

    fixed_block = "\n".join([*header_lines, *tail_lines, *must_keep_lines])
    if len(fixed_block) >= max_chars:
        return fixed_block[:max_chars]

    remaining = max_chars - len(fixed_block)
    ordered_entries = sorted(entries, key=_priority_sort_key)
    selected_prim_lines: list[str] = []
    for entry in ordered_entries:
        line = f"- `{entry.signature}` [{entry.pipe}]"
        cost = len(line) + 1
        if cost <= remaining:
            selected_prim_lines.append(line)
            remaining -= cost

    entry_order = {entry.signature: idx for idx, entry in enumerate(entries)}
    selected_prim_lines.sort(
        key=lambda line: entry_order.get(line.split("`", 2)[1], len(entries))
    )

    notes_budget = min(remaining, 700)
    selected_note_lines = list(must_keep_lines)
    for note in optional_notes:
        line = f"- {note}"
        cost = len(line) + 1
        if cost <= notes_budget:
            selected_note_lines.append(line)
            notes_budget -= cost

    rendered = "\n".join(
        [*header_lines, *selected_prim_lines, *tail_lines, *selected_note_lines]
    )
    return rendered[:max_chars] if len(rendered) > max_chars else rendered


__all__ = [
    "CCE_CPP20_SURFACE",
    "DEFAULT_CANN_SERIES",
    "LOWERING_GUARDRAILS",
    "VALID_CANN_SERIES",
    "DialectId",
    "DialectSurfaceSpec",
    "INLINE_ASCENDC_VF_SURFACE",
    "INLINE_CCE_VF_SURFACE",
    "PrimitiveEntry",
    "TILELANG_DSL_SURFACE",
    "TL_TEMPLATES_CPP20_SURFACE",
    "filter_primitives_on_demand",
    "get_dialect_surface",
    "load_cann_surface",
    "render_layer2_surface",
    "resolve_cann_series",
    "resolve_trackb_cce_surface",
]
