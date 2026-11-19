"""Automated ISA-gap scanner comparing asc-devkit hardware ISA against TileLang lowering."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import re
from typing import Any

VALID_HARDWARE_PIPES: tuple[str, ...] = (
    "pipe_s_scalar_spr",
    "pipe_mte2_mte3_dma",
    "pipe_mte1_l1_to_l0",
    "pipe_m_cube",
    "pipe_fix_fixpipe",
    "pipe_v_simd_simt",
    "pipe_sync_barrier_mutex",
)


@dataclass(frozen=True)
class IsaPrimitiveSpec:
    """One asc-devkit hardware capability and its expected TileLang lowering symbol(s)."""

    name: str
    hardware_pipe: str
    ascend_devkit_intrinsic: str
    tilelang_symbols: tuple[str, ...]
    workaround_track: str = "ascendc_template"
    description: str = ""

    def __post_init__(self) -> None:
        if self.hardware_pipe not in VALID_HARDWARE_PIPES:
            raise ValueError(
                f"Unknown hardware pipe {self.hardware_pipe!r}; expected one of {VALID_HARDWARE_PIPES}"
            )


@dataclass(frozen=True)
class PipeGapBreakdown:
    """Coverage and missing-primitive gap breakdown for one physical hardware execution pipe."""

    pipe: str
    total_capabilities: int
    supported_primitives: tuple[str, ...]
    missing_primitives: tuple[str, ...]
    coverage_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PipeGapReport:
    """Diff between asc-devkit hardware ISA surface and TileLang lowering surface by pipe."""

    surface_source: str
    pipes: dict[str, PipeGapBreakdown]
    supported_primitives: tuple[str, ...]
    missing_primitives: tuple[str, ...]
    coverage_ratio: float

    def has_gap(self, primitive_or_pipe: str) -> bool:
        """Return True if *primitive_or_pipe* has any missing TileLang lowering primitive."""
        if primitive_or_pipe in self.pipes:
            return len(self.pipes[primitive_or_pipe].missing_primitives) > 0
        return primitive_or_pipe in self.missing_primitives

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_source": self.surface_source,
            "pipes": {k: v.to_dict() for k, v in self.pipes.items()},
            "supported_primitives": list(self.supported_primitives),
            "missing_primitives": list(self.missing_primitives),
            "coverage_ratio": self.coverage_ratio,
        }


ASCEND_DEVKIT_ISA_CATALOG: tuple[IsaPrimitiveSpec, ...] = (
    IsaPrimitiveSpec(
        name="fp4_e2m1_dtype",
        hardware_pipe="pipe_m_cube",
        ascend_devkit_intrinsic="fp4_e2m1fnx2",
        tilelang_symbols=("float4_e2m1fn", "float4_e2m1fnx2"),
        workaround_track="ascendc_template",
        description="Packed 4-bit FP4 E2M1 tensor representation",
    ),
    IsaPrimitiveSpec(
        name="mx_e8m0_blockscale_gemm",
        hardware_pipe="pipe_m_cube",
        ascend_devkit_intrinsic="MmadBlockScaled<e8m0>",
        tilelang_symbols=("gemm_blockscaled", "float8_e8m0fnu"),
        workaround_track="ascendc_template",
        description="Block-scaled MX GEMM with E8M0 microscaling factors",
    ),
    IsaPrimitiveSpec(
        name="mx_spr_scale_config",
        hardware_pipe="pipe_s_scalar_spr",
        ascend_devkit_intrinsic="SetMxScaleSPR / QuantMx",
        tilelang_symbols=("set_mx_scale_spr", "ascend_mx_quant"),
        workaround_track="ascendc_template",
        description="Hardware SPR microscaling configuration and online MX quantization",
    ),
    IsaPrimitiveSpec(
        name="subbyte_q4_bitshift_unpack",
        hardware_pipe="pipe_v_simd_simt",
        ascend_devkit_intrinsic="BitUnpackQ4_0 / FixPipeDeqScale",
        tilelang_symbols=("bit_unpack_q4", "fixpipe_deq_scale"),
        workaround_track="ascendc_template",
        description="Sub-byte INT4/Q4_0 register bitshift unpack and FixPipe dequant scale",
    ),
    IsaPrimitiveSpec(
        name="cube_mmad_sparse",
        hardware_pipe="pipe_m_cube",
        ascend_devkit_intrinsic="MmadSparse",
        tilelang_symbols=("gemm_sparse", "mmad_sparse", "ascend_mmad_sparse"),
        workaround_track="ascendc_template",
        description="2:4 structured sparse Cube MMAD execution primitive",
    ),
    IsaPrimitiveSpec(
        name="fixpipe_deq_scale",
        hardware_pipe="pipe_fix_fixpipe",
        ascend_devkit_intrinsic="FixPipeDeqScale",
        tilelang_symbols=("fixpipe_deq_scale", "ascend_fixpipe_deq_scale"),
        workaround_track="ascendc_template",
        description="FixPipe hardware per-channel/block dequantization scaling",
    ),
    IsaPrimitiveSpec(
        name="scalar_set_ctrl_spr",
        hardware_pipe="pipe_s_scalar_spr",
        ascend_devkit_intrinsic="SetCtrlSpr",
        tilelang_symbols=("set_ctrl_spr", "ascend_set_ctrl_spr"),
        workaround_track="ascendc_template",
        description="Scalar SPR control register configuration for quantized/MX modes",
    ),
    IsaPrimitiveSpec(
        name="simd_vld_vst",
        hardware_pipe="pipe_v_simd_simt",
        ascend_devkit_intrinsic="Reg::Load / Reg::Store",
        tilelang_symbols=("simd.vld", "simd.vst2", "simd.vsts"),
        workaround_track="inline_ascendc_vf",
        description="Register-base UB load/store intrinsics",
    ),
    IsaPrimitiveSpec(
        name="simd_vgather_vscatter",
        hardware_pipe="pipe_v_simd_simt",
        ascend_devkit_intrinsic="Reg::Gather / Reg::Scatter",
        tilelang_symbols=("simd.vgather2", "simd.vgatherb", "simd.vscatter"),
        workaround_track="inline_ascendc_vf",
        description="Register-base indexed gather/scatter",
    ),
    IsaPrimitiveSpec(
        name="regbase_dist_unpack_pack",
        hardware_pipe="pipe_v_simd_simt",
        ascend_devkit_intrinsic="LoadDist::DIST_UNPACK_B16 / StoreDist::DIST_PACK_B32",
        tilelang_symbols=("simd.dist_unpack_b16", "simd.dist_pack_b32"),
        workaround_track="inline_ascendc_vf",
        description="Explicit RegBase DIST_UNPACK_B16 / DIST_PACK_B32 lane distribution",
    ),
    IsaPrimitiveSpec(
        name="regbase_reduce_data_block",
        hardware_pipe="pipe_v_simd_simt",
        ascend_devkit_intrinsic="Reg::ReduceDataBlock",
        tilelang_symbols=("simd.reduce_data_block", "simd.reduce_datablock"),
        workaround_track="inline_ascendc_vf",
        description="Hardware 32B DataBlock intra-vector reduction intrinsic",
    ),
    IsaPrimitiveSpec(
        name="crosscore_flag_set_wait",
        hardware_pipe="pipe_sync_barrier_mutex",
        ascend_devkit_intrinsic="CrossCoreSetFlag / CrossCoreWaitFlag",
        tilelang_symbols=("ascend_cross_core_set_flag", "ascend_cross_core_wait_flag"),
        workaround_track="ascendc_template",
        description="Cross-core AIC/AIV synchronization flags",
    ),
    IsaPrimitiveSpec(
        name="crosscore_mode4_dual_dst",
        hardware_pipe="pipe_sync_barrier_mutex",
        ascend_devkit_intrinsic="CrossCoreSetFlag<mode=4, dualDstCtl>",
        tilelang_symbols=("ascend_cross_core_mode4", "cross_core_dual_dst"),
        workaround_track="ascendc_template",
        description="Mode-4 symmetric 1C2V dual-destination cross-core signaling",
    ),
    IsaPrimitiveSpec(
        name="hardware_mutex_28",
        hardware_pipe="pipe_sync_barrier_mutex",
        ascend_devkit_intrinsic="HwMutexLock / HwMutexUnlock",
        tilelang_symbols=("ascend_hw_mutex_lock", "hw_mutex_lock"),
        workaround_track="ascendc_template",
        description="28 hardware mutex locks for inter-core atomic serialization",
    ),
    IsaPrimitiveSpec(
        name="simt_warp_shuffle_ballot",
        hardware_pipe="pipe_v_simd_simt",
        ascend_devkit_intrinsic="__shfl / __ballot",
        tilelang_symbols=("shfl_sync", "ballot_sync", "activemask"),
        workaround_track="ascendc_template",
        description="SIMT warp shuffle and ballot collectives",
    ),
    IsaPrimitiveSpec(
        name="simt_atomic_add_minmax",
        hardware_pipe="pipe_v_simd_simt",
        ascend_devkit_intrinsic="atomicAdd / atomicMax / atomicMin",
        tilelang_symbols=("atomic_add", "atomic_max", "atomic_min"),
        workaround_track="ascendc_template",
        description="Standard warp/block atomic arithmetic",
    ),
    IsaPrimitiveSpec(
        name="simt_gm_atomic_cas",
        hardware_pipe="pipe_v_simd_simt",
        ascend_devkit_intrinsic="asc_atomic_cas",
        tilelang_symbols=("atomic_cas", "asc_atomic_cas"),
        workaround_track="ascendc_template",
        description="In-place global memory compare-and-swap atomic",
    ),
    IsaPrimitiveSpec(
        name="simt_uint_div_fast",
        hardware_pipe="pipe_v_simd_simt",
        ascend_devkit_intrinsic="Simt::UintDiv",
        tilelang_symbols=("simt.uint_div", "uint_div"),
        workaround_track="ascendc_template",
        description="Strength-reduced integer divider for dynamic SIMT stride indexing",
    ),
    IsaPrimitiveSpec(
        name="dma_dual_copy_and_nd2nz",
        hardware_pipe="pipe_mte2_mte3_dma",
        ascend_devkit_intrinsic="DataCopy / Nd2NzParams",
        tilelang_symbols=("dual_copy", "ascend_nd2nz_post_copy", "copy"),
        workaround_track="ascendc_template",
        description="Dual-core copy and UB->L1 ND2NZ fractal layout conversion",
    ),
    IsaPrimitiveSpec(
        name="dma_l2_cache_mode_disable",
        hardware_pipe="pipe_mte2_mte3_dma",
        ascend_devkit_intrinsic="CACHE_MODE_DISABLE / L2BypassHint",
        tilelang_symbols=("dma_cache_mode_disable", "l2_cache_bypass_hint"),
        workaround_track="ascendc_template",
        description="Per-operand GM->L1/UB L2 cache bypass hint",
    ),
    IsaPrimitiveSpec(
        name="dma_datacopy_pad_unaligned",
        hardware_pipe="pipe_mte2_mte3_dma",
        ascend_devkit_intrinsic="DataCopyPad",
        tilelang_symbols=("datacopy_pad", "ascend_datacopy_pad"),
        workaround_track="ascendc_template",
        description="Hardware 32B-unaligned DataCopyPad tail DMA",
    ),
    IsaPrimitiveSpec(
        name="mte1_load_2d_3d",
        hardware_pipe="pipe_mte1_l1_to_l0",
        ascend_devkit_intrinsic="Load2D / Load3D",
        tilelang_symbols=("load_2d", "load_3d", "ascend_load2d", "ascend_load3d"),
        workaround_track="ascendc_template",
        description="MTE1 L1->L0A/L0B 2D fractal and 3D im2col tile loading",
    ),
    IsaPrimitiveSpec(
        name="fixpipe_l0c_to_ub_split_m",
        hardware_pipe="pipe_fix_fixpipe",
        ascend_devkit_intrinsic="L0CToUbSplitM",
        tilelang_symbols=("l0c_to_ub_split_m", "ascend_l0c_to_ub_split_m", "fixpipe_l0c_to_ub"),
        workaround_track="ascendc_template",
        description="FixPipe direct L0C->UB SPLIT_M routing across coupled AIV sub-blocks",
    ),
    IsaPrimitiveSpec(
        name="scalar_dcci_cache_invalidate",
        hardware_pipe="pipe_s_scalar_spr",
        ascend_devkit_intrinsic="dcci",
        tilelang_symbols=("dcci", "ascend_dcci"),
        workaround_track="ascendc_template",
        description="Scalar data-cache clean/invalidate for SIMT-to-SIMD/GM coherence",
    ),
    IsaPrimitiveSpec(
        name="simd_vci_vcmax",
        hardware_pipe="pipe_v_simd_simt",
        ascend_devkit_intrinsic="Reg::Vci / Reg::Vcmax",
        tilelang_symbols=("simd.vci", "simd.vcmax", "simd.vcmin"),
        workaround_track="inline_ascendc_vf",
        description="Index generation and vector max/min reduction",
    ),
    IsaPrimitiveSpec(
        name="hardware_bitonic_sort32",
        hardware_pipe="pipe_v_simd_simt",
        ascend_devkit_intrinsic="BitonicSort32 / Sort32",
        tilelang_symbols=("bitonic_sort32", "simd.sort32"),
        workaround_track="ascendc_template",
        description="Hardware 32-element bitonic sort unit",
    ),
    IsaPrimitiveSpec(
        name="hardware_mrgsort4_topk",
        hardware_pipe="pipe_v_simd_simt",
        ascend_devkit_intrinsic="MrgSort4 / TopK",
        tilelang_symbols=("mrgsort4", "hardware_topk", "simd.mrgsort4"),
        workaround_track="ascendc_template",
        description="Hardware 4-way merge sort and TopK selector",
    ),
)


def _default_ascend_sdk_header_roots() -> tuple[str, ...]:
    roots: list[str] = []
    env_devkit = os.environ.get("TILEFOUNDRY_NPU_ASC_DEVKIT_REPO", "").strip()
    if env_devkit:
        roots.append(str(Path(env_devkit).expanduser() / "include"))
    sibling_devkit = Path(__file__).resolve().parents[5].parent / "asc-devkit" / "include"
    roots.append(str(sibling_devkit))
    for env_key in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH"):
        cann = os.environ.get(env_key, "").strip()
        if cann:
            roots.append(str(Path(cann).expanduser() / "x86_64-linux" / "include" / "ascendc"))
            roots.append(str(Path(cann).expanduser() / "include" / "ascendc"))
    return tuple(dict.fromkeys(roots))


def _default_tilelang_codegen_path() -> str:
    env_tl = (
        os.environ.get("TILEFOUNDRY_NPU_TILELANG_REPO", "").strip()
        or os.environ.get("TILEFOUNDRY_NPU_TILELANG_SOURCE", "").strip()
    )
    if env_tl:
        return str(Path(env_tl).expanduser() / "tilelang" / "ascend" / "codegen.py")
    return str(
        Path(__file__).resolve().parents[5].parent / "tilelang" / "tilelang" / "ascend" / "codegen.py"
    )


DEFAULT_ASCEND_SDK_HEADER_ROOTS: tuple[str, ...] = _default_ascend_sdk_header_roots()
DEFAULT_TILELANG_CODEGEN_PATH = _default_tilelang_codegen_path()

BASELINE_TILELANG_ASCEND_SURFACE: frozenset[str] = frozenset(
    {
        "copy",
        "dual_copy",
        "gemm",
        "simd.vld",
        "simd.vst2",
        "simd.vsts",
        "simd.vgather2",
        "simd.vgatherb",
        "simd.vscatter",
        "simd.vci",
        "simd.vcmax",
        "simd.vcmin",
        "shfl_sync",
        "ballot_sync",
        "activemask",
        "atomic_add",
        "atomic_max",
        "atomic_min",
        "ascend_cross_core_set_flag",
        "ascend_cross_core_wait_flag",
        "dcci",
    }
)


@dataclass(frozen=True)
class SdkHeaderScanResult:
    """Result of mechanically scanning Ascend C++ SDK headers on disk (with catalog fallback)."""

    source: str
    fallback_used: bool
    scanned_roots: tuple[str, ...]
    header_files_scanned: int
    symbols: frozenset[str]
    by_category: dict[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "fallback_used": self.fallback_used,
            "scanned_roots": list(self.scanned_roots),
            "header_files_scanned": self.header_files_scanned,
            "symbols": sorted(self.symbols),
            "by_category": {k: list(v) for k, v in self.by_category.items()},
        }


_QUALIFIED_SDK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Reg", re.compile(r"\bReg::([A-Za-z_][A-Za-z0-9_]*)")),
    ("Simt", re.compile(r"\bSimt::([A-Za-z_][A-Za-z0-9_]*)")),
    ("LoadDist", re.compile(r"\bLoadDist::([A-Za-z_][A-Za-z0-9_]*)")),
    ("StoreDist", re.compile(r"\bStoreDist::([A-Za-z_][A-Za-z0-9_]*)")),
)

_NS_DECL_RE = re.compile(
    r"(?:__aicore__|__simd_callee__|__simt_callee__|inline|static|constexpr|template\s*<[^>]*>)*"
    r"\s*(?:void|bool|int\d*_t|uint\d*_t|float|double|auto|T|U|struct|class)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(|<|\{)"
)
_ENUM_BLOCK_RE = re.compile(
    r"enum\s+(?:class\s+)?(LoadDist|StoreDist)\b[^{]*\{([^}]*)\}",
    re.DOTALL,
)
_ENUM_MEMBER_RE = re.compile(r"\b([A-Z][A-Z0-9_]+)\b")
_NAMED_ISA_INTRINSICS_RE = re.compile(
    r"\b("
    r"CrossCoreSetFlag|CrossCoreWaitFlag|SetFlag|WaitFlag|"
    r"HwMutexLock|HwMutexUnlock|PipeBarrier|LocalMemBar|"
    r"Mmad[A-Za-z0-9_]*|Load2D|Load3D|LoadData2D[A-Za-z0-9_]*|LoadData3D[A-Za-z0-9_]*|"
    r"DataCopyPad[A-Za-z0-9_]*|DataCopy|Nd2NzParams|"
    r"dcci|SetCtrlSpr|SetMxScaleSPR|CACHE_MODE_DISABLE|"
    r"Fixpipe[A-Za-z0-9_]*|BitonicSort32|MrgSort4|TopK|"
    r"asc_atomic_[A-Za-z0-9_]+|__shfl[A-Za-z0-9_]*|__ballot[A-Za-z0-9_]*"
    r")\b"
)


def _extract_namespace_symbols(text: str, ns_name: str, out: set[str]) -> None:
    ns_pat = re.compile(rf"\bnamespace\s+{ns_name}\s*\{{")
    for m in ns_pat.finditer(text):
        start = m.end()
        depth = 1
        pos = start
        n = len(text)
        while pos < n and depth > 0:
            ch = text[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            pos += 1
        block = text[start : pos - 1]
        for dm in _NS_DECL_RE.finditer(block):
            ident = dm.group(1)
            if ident not in ("if", "for", "while", "switch", "return", "inline", "void"):
                out.add(f"{ns_name}::{ident}")


def scan_ascend_sdk_headers(
    header_roots: Sequence[str | Path] | str | Path | None = None,
) -> SdkHeaderScanResult:
    """Mechanically scan installed Ascend C++ SDK headers on disk, falling back to catalog."""
    if header_roots is None:
        roots = [Path(p) for p in DEFAULT_ASCEND_SDK_HEADER_ROOTS]
    elif isinstance(header_roots, (str, Path)):
        roots = [Path(header_roots)]
    else:
        roots = [Path(p) for p in header_roots]

    existing_roots = [r for r in roots if r.is_dir()]
    symbols: set[str] = set()
    reg_syms: set[str] = set()
    simt_syms: set[str] = set()
    load_dist_syms: set[str] = set()
    store_dist_syms: set[str] = set()
    other_syms: set[str] = set()

    seen_real_files: set[Path] = set()
    scanned_count = 0

    for root in existing_roots:
        for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
            for fname in filenames:
                if not fname.endswith((".h", ".hpp")):
                    continue
                fpath = Path(dirpath) / fname
                try:
                    resolved = fpath.resolve()
                except OSError:
                    continue
                if resolved in seen_real_files or not resolved.is_file():
                    continue
                seen_real_files.add(resolved)
                scanned_count += 1
                try:
                    content = resolved.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue

                for prefix, pat in _QUALIFIED_SDK_PATTERNS:
                    for m in pat.finditer(content):
                        sym = f"{prefix}::{m.group(1)}"
                        symbols.add(sym)
                        if prefix == "Reg":
                            reg_syms.add(sym)
                        elif prefix == "Simt":
                            simt_syms.add(sym)
                        elif prefix == "LoadDist":
                            load_dist_syms.add(sym)
                        elif prefix == "StoreDist":
                            store_dist_syms.add(sym)

                if "namespace Reg" in content:
                    before = len(reg_syms)
                    _extract_namespace_symbols(content, "Reg", reg_syms)
                    if len(reg_syms) > before:
                        symbols.update(reg_syms)
                if "namespace Simt" in content:
                    before = len(simt_syms)
                    _extract_namespace_symbols(content, "Simt", simt_syms)
                    if len(simt_syms) > before:
                        symbols.update(simt_syms)

                if "LoadDist" in content or "StoreDist" in content:
                    for em in _ENUM_BLOCK_RE.finditer(content):
                        enum_name, body = em.group(1), em.group(2)
                        for mem in _ENUM_MEMBER_RE.finditer(body):
                            sym = f"{enum_name}::{mem.group(1)}"
                            symbols.add(sym)
                            if enum_name == "LoadDist":
                                load_dist_syms.add(sym)
                            else:
                                store_dist_syms.add(sym)

                for im in _NAMED_ISA_INTRINSICS_RE.finditer(content):
                    tok = im.group(1)
                    symbols.add(tok)
                    other_syms.add(tok)

    if scanned_count == 0 or not symbols:
        for spec in ASCEND_DEVKIT_ISA_CATALOG:
            for part in spec.ascend_devkit_intrinsic.split("/"):
                tok = part.strip()
                if not tok:
                    continue
                symbols.add(tok)
                if tok.startswith("Reg::"):
                    reg_syms.add(tok)
                elif tok.startswith("Simt::"):
                    simt_syms.add(tok)
                elif tok.startswith("LoadDist::"):
                    load_dist_syms.add(tok)
                elif tok.startswith("StoreDist::"):
                    store_dist_syms.add(tok)
                else:
                    other_syms.add(tok)
        return SdkHeaderScanResult(
            source="catalog_fallback",
            fallback_used=True,
            scanned_roots=tuple(str(r) for r in existing_roots),
            header_files_scanned=0,
            symbols=frozenset(symbols),
            by_category={
                "Reg": tuple(sorted(reg_syms)),
                "Simt": tuple(sorted(simt_syms)),
                "LoadDist": tuple(sorted(load_dist_syms)),
                "StoreDist": tuple(sorted(store_dist_syms)),
                "sync_dma_compute": tuple(sorted(other_syms)),
            },
        )

    return SdkHeaderScanResult(
        source="sdk_headers",
        fallback_used=False,
        scanned_roots=tuple(str(r) for r in existing_roots),
        header_files_scanned=scanned_count,
        symbols=frozenset(symbols),
        by_category={
            "Reg": tuple(sorted(reg_syms)),
            "Simt": tuple(sorted(simt_syms)),
            "LoadDist": tuple(sorted(load_dist_syms)),
            "StoreDist": tuple(sorted(store_dist_syms)),
            "sync_dma_compute": tuple(sorted(other_syms)),
        },
    )


def _scan_tilelang_codegen_file(codegen_path: Path) -> set[str]:
    """Scan tilelang/ascend/codegen.py (and sibling C++ codegen tables) for lowering symbols."""
    found: set[str] = set()
    if codegen_path.is_file():
        text = codegen_path.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r'["\'](target\.build\.[A-Za-z0-9_]+)["\']', text):
            found.add(m.group(1))
        for m in re.finditer(r'\b([A-Z][A-Z0-9_]*_CODEGEN)\b', text):
            found.add(m.group(1))
        for m in re.finditer(r'DeviceCodegen\(\s*["\']([A-Za-z0-9_]+)["\']', text):
            found.add(f"codegen:{m.group(1)}")

        repo_root = codegen_path.resolve().parents[2]
        cpp_candidates = (
            repo_root / "src" / "ascend" / "codegen" / "codegen_ascend.cc",
            repo_root / "src" / "ascend" / "codegen" / "intrin_rule_ascend.cc",
        )
        for cpp_file in cpp_candidates:
            if not cpp_file.is_file():
                continue
            cpp_text = cpp_file.read_text(encoding="utf-8", errors="ignore")
            for m in re.finditer(r"\btl::(ascend_[A-Za-z0-9_]+)\s*\(", cpp_text):
                found.add(m.group(1))
            for m in re.finditer(r'["\']tl\.(ascend_[A-Za-z0-9_]+)["\']', cpp_text):
                found.add(m.group(1))
            for m in re.finditer(r'["\']tl\.(simd\.[A-Za-z0-9_]+)["\']', cpp_text):
                found.add(m.group(1))
    return found


def discover_tilelang_lowering_surface(
    tilelang_module: object | None = None,
    *,
    codegen_path: str | Path | None = None,
) -> tuple[str, frozenset[str]]:
    """Discover registered TileLang Ascend lowering symbols from language + codegen.py."""
    mod = tilelang_module
    source = "explicit_module"
    if mod is None:
        try:
            import tilelang.ascend.language as tl_ascend  # noqa: PLC0415

            mod = tl_ascend
            source = "tilelang.ascend.language"
        except Exception:
            mod = None

    symbols: set[str] = set()
    if mod is not None:
        for name in dir(mod):
            if not name.startswith("_"):
                symbols.add(name)
        for sub_attr in ("simd", "simt"):
            sub_obj = getattr(mod, sub_attr, None)
            if sub_obj is not None:
                for name in dir(sub_obj):
                    if not name.startswith("_"):
                        symbols.add(f"{sub_attr}.{name}")

    cg_candidates: list[Path] = []
    if codegen_path is not None:
        cg_candidates.append(Path(codegen_path))
    else:
        if mod is not None and getattr(mod, "__file__", None):
            cg_candidates.append(Path(str(mod.__file__)).resolve().parents[1] / "codegen.py")
        cg_candidates.append(Path(DEFAULT_TILELANG_CODEGEN_PATH))

    codegen_scanned = False
    for cg_path in cg_candidates:
        if cg_path.is_file():
            symbols.update(_scan_tilelang_codegen_file(cg_path))
            codegen_scanned = True
            break

    if mod is None and not codegen_scanned:
        return "baseline_fallback", BASELINE_TILELANG_ASCEND_SURFACE
    if mod is not None and codegen_scanned and source == "tilelang.ascend.language":
        source = "tilelang.ascend.language+codegen"
    elif mod is None and codegen_scanned:
        source = "tilelang.ascend.codegen"
    return source, frozenset(symbols)


def _resolve_surface(
    tilelang_surface: Iterable[str] | object | None,
) -> tuple[str, frozenset[str]]:
    if tilelang_surface is None:
        return discover_tilelang_lowering_surface()
    if isinstance(tilelang_surface, (set, frozenset, list, tuple)):
        return "explicit_surface", frozenset(str(s) for s in tilelang_surface)
    return discover_tilelang_lowering_surface(tilelang_surface)


def scan_tilelang_pipe_gaps(
    *,
    tilelang_surface: Iterable[str] | object | None = None,
    pipes: Iterable[str] | None = None,
    devkit_catalog: Iterable[IsaPrimitiveSpec] | None = None,
) -> PipeGapReport:
    """Diff asc-devkit hardware capabilities against TileLang across the 7 hardware pipes."""
    surface_source, surface = _resolve_surface(tilelang_surface)

    selected_pipes = tuple(pipes) if pipes is not None else VALID_HARDWARE_PIPES
    for pipe in selected_pipes:
        if pipe not in VALID_HARDWARE_PIPES:
            raise ValueError(
                f"Unknown hardware pipe {pipe!r}; expected one of {VALID_HARDWARE_PIPES}"
            )

    catalog = tuple(devkit_catalog) if devkit_catalog is not None else ASCEND_DEVKIT_ISA_CATALOG

    pipe_reports: dict[str, PipeGapBreakdown] = {}
    all_supported: list[str] = []
    all_missing: list[str] = []

    for pipe in selected_pipes:
        pipe_specs = [spec for spec in catalog if spec.hardware_pipe == pipe]
        supported: list[str] = []
        missing: list[str] = []
        for spec in pipe_specs:
            if any(sym in surface for sym in spec.tilelang_symbols):
                supported.append(spec.name)
            else:
                missing.append(spec.name)
        total = len(pipe_specs)
        ratio = (len(supported) / float(total)) if total > 0 else 1.0
        pipe_reports[pipe] = PipeGapBreakdown(
            pipe=pipe,
            total_capabilities=total,
            supported_primitives=tuple(supported),
            missing_primitives=tuple(missing),
            coverage_ratio=round(ratio, 4),
        )
        all_supported.extend(supported)
        all_missing.extend(missing)

    total_all = len(all_supported) + len(all_missing)
    overall_ratio = (len(all_supported) / float(total_all)) if total_all > 0 else 1.0
    return PipeGapReport(
        surface_source=surface_source,
        pipes=pipe_reports,
        supported_primitives=tuple(all_supported),
        missing_primitives=tuple(all_missing),
        coverage_ratio=round(overall_ratio, 4),
    )


__all__ = [
    "ASCEND_DEVKIT_ISA_CATALOG",
    "DEFAULT_ASCEND_SDK_HEADER_ROOTS",
    "DEFAULT_TILELANG_CODEGEN_PATH",
    "IsaPrimitiveSpec",
    "PipeGapBreakdown",
    "PipeGapReport",
    "SdkHeaderScanResult",
    "VALID_HARDWARE_PIPES",
    "discover_tilelang_lowering_surface",
    "scan_ascend_sdk_headers",
    "scan_tilelang_pipe_gaps",
]
