"""Automated ISA-gap scanner comparing CANN dav-3510 hardware ISA against TileLang lowering."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
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
    """One CANN dav-3510 hardware capability and its expected TileLang lowering symbol(s)."""

    name: str
    hardware_pipe: str
    cann_intrinsic: str
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
    """Diff between CANN dav-3510 hardware ISA surface and TileLang lowering surface by pipe."""

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


_CATALOG_ROWS: tuple[tuple[str, str, str, tuple[str, ...], str, str], ...] = (
    ("fp4_e2m1_dtype", "pipe_m_cube", "fp4_e2m1fnx2", ("float4_e2m1fn", "float4_e2m1fnx2"), "ascendc_template", "Packed 4-bit FP4 E2M1 tensor representation"),
    ("mx_e8m0_blockscale_gemm", "pipe_m_cube", "MmadBlockScaled<e8m0>", ("gemm_blockscaled", "float8_e8m0fnu"), "ascendc_template", "Block-scaled MX GEMM with E8M0 microscaling factors"),
    ("mx_spr_scale_config", "pipe_s_scalar_spr", "SetMxScaleSPR / QuantMx", ("set_mx_scale_spr", "ascend_mx_quant"), "ascendc_template", "Hardware SPR microscaling configuration and online MX quantization"),
    ("subbyte_q4_bitshift_unpack", "pipe_v_simd_simt", "BitUnpackQ4_0 / FixPipeDeqScale", ("bit_unpack_q4", "fixpipe_deq_scale"), "ascendc_template", "Sub-byte INT4/Q4_0 register bitshift unpack and FixPipe dequant scale"),
    ("cube_mmad_sparse", "pipe_m_cube", "MmadSparse", ("gemm_sparse", "mmad_sparse", "ascend_mmad_sparse"), "ascendc_template", "2:4 structured sparse Cube MMAD execution primitive"),
    ("fixpipe_deq_scale", "pipe_fix_fixpipe", "FixPipeDeqScale", ("fixpipe_deq_scale", "ascend_fixpipe_deq_scale"), "ascendc_template", "FixPipe hardware per-channel/block dequantization scaling"),
    ("scalar_set_ctrl_spr", "pipe_s_scalar_spr", "SetCtrlSpr", ("set_ctrl_spr", "ascend_set_ctrl_spr"), "ascendc_template", "Scalar SPR control register configuration for quantized/MX modes"),
    ("simd_vld_vst", "pipe_v_simd_simt", "Reg::Load / Reg::Store", ("simd.vld", "simd.vst2", "simd.vsts"), "inline_ascendc_vf", "Register-base UB load/store intrinsics"),
    ("simd_vgather_vscatter", "pipe_v_simd_simt", "Reg::Gather / Reg::Scatter", ("simd.vgather2", "simd.vgatherb", "simd.vscatter"), "inline_ascendc_vf", "Register-base indexed gather/scatter"),
    ("regbase_dist_unpack_pack", "pipe_v_simd_simt", "LoadDist::DIST_UNPACK_B16 / StoreDist::DIST_PACK_B32", ("simd.dist_unpack_b16", "simd.dist_pack_b32"), "inline_ascendc_vf", "Explicit RegBase DIST_UNPACK_B16 / DIST_PACK_B32 lane distribution"),
    ("regbase_reduce_data_block", "pipe_v_simd_simt", "Reg::ReduceDataBlock", ("simd.reduce_data_block", "simd.reduce_datablock"), "inline_ascendc_vf", "Hardware 32B DataBlock intra-vector reduction intrinsic"),
    ("crosscore_flag_set_wait", "pipe_sync_barrier_mutex", "CrossCoreSetFlag / CrossCoreWaitFlag", ("ascend_cross_core_set_flag", "ascend_cross_core_wait_flag"), "ascendc_template", "Cross-core AIC/AIV synchronization flags"),
    ("crosscore_mode4_dual_dst", "pipe_sync_barrier_mutex", "CrossCoreSetFlag<mode=4, dualDstCtl>", ("ascend_cross_core_mode4", "cross_core_dual_dst"), "ascendc_template", "Mode-4 symmetric 1C2V dual-destination cross-core signaling"),
    ("hardware_mutex_28", "pipe_sync_barrier_mutex", "HwMutexLock / HwMutexUnlock", ("ascend_hw_mutex_lock", "hw_mutex_lock"), "ascendc_template", "28 hardware mutex locks for inter-core atomic serialization"),
    ("simt_warp_shuffle_ballot", "pipe_v_simd_simt", "__shfl / __ballot", ("shfl_sync", "ballot_sync", "activemask"), "ascendc_template", "SIMT warp shuffle and ballot collectives"),
    ("simt_atomic_add_minmax", "pipe_v_simd_simt", "atomicAdd / atomicMax / atomicMin", ("atomic_add", "atomic_max", "atomic_min"), "ascendc_template", "Standard warp/block atomic arithmetic"),
    ("simt_gm_atomic_cas", "pipe_v_simd_simt", "asc_atomic_cas", ("atomic_cas", "asc_atomic_cas"), "ascendc_template", "In-place global memory compare-and-swap atomic"),
    ("simt_uint_div_fast", "pipe_v_simd_simt", "Simt::UintDiv", ("simt.uint_div", "uint_div"), "ascendc_template", "Strength-reduced integer divider for dynamic SIMT stride indexing"),
    ("dma_dual_copy_and_nd2nz", "pipe_mte2_mte3_dma", "DataCopy / Nd2NzParams", ("dual_copy", "ascend_nd2nz_post_copy", "copy"), "ascendc_template", "Dual-core copy and UB->L1 ND2NZ fractal layout conversion"),
    ("dma_l2_cache_mode_disable", "pipe_mte2_mte3_dma", "CACHE_MODE_DISABLE / L2BypassHint", ("dma_cache_mode_disable", "l2_cache_bypass_hint"), "ascendc_template", "Per-operand GM->L1/UB L2 cache bypass hint"),
    ("dma_datacopy_pad_unaligned", "pipe_mte2_mte3_dma", "DataCopyPad", ("datacopy_pad", "ascend_datacopy_pad"), "ascendc_template", "Hardware 32B-unaligned DataCopyPad tail DMA"),
    ("mte1_load_2d_3d", "pipe_mte1_l1_to_l0", "Load2D / Load3D", ("load_2d", "load_3d", "ascend_load2d", "ascend_load3d"), "ascendc_template", "MTE1 L1->L0A/L0B 2D fractal and 3D im2col tile loading"),
    ("fixpipe_l0c_to_ub_split_m", "pipe_fix_fixpipe", "L0CToUbSplitM", ("l0c_to_ub_split_m", "ascend_l0c_to_ub_split_m", "fixpipe_l0c_to_ub"), "ascendc_template", "FixPipe direct L0C->UB SPLIT_M routing across coupled AIV sub-blocks"),
    ("scalar_dcci_cache_invalidate", "pipe_s_scalar_spr", "dcci", ("dcci", "ascend_dcci"), "ascendc_template", "Scalar data-cache clean/invalidate for SIMT-to-SIMD/GM coherence"),
    ("simd_vci_vcmax", "pipe_v_simd_simt", "Reg::Vci / Reg::Vcmax", ("simd.vci", "simd.vcmax", "simd.vcmin"), "inline_ascendc_vf", "Index generation and vector max/min reduction"),
    ("hardware_bitonic_sort32", "pipe_v_simd_simt", "BitonicSort32 / Sort32", ("bitonic_sort32", "simd.sort32"), "ascendc_template", "Hardware 32-element bitonic sort unit"),
    ("hardware_mrgsort4_topk", "pipe_v_simd_simt", "MrgSort4 / TopK", ("mrgsort4", "hardware_topk", "simd.mrgsort4"), "ascendc_template", "Hardware 4-way merge sort and TopK selector"),
)

DAV3510_CANN_ISA_CATALOG: tuple[IsaPrimitiveSpec, ...] = tuple(
    IsaPrimitiveSpec(*row) for row in _CATALOG_ROWS
)


def _default_ascend_sdk_header_roots() -> tuple[str, ...]:
    roots: list[str] = []
    for env_key in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH"):
        if cann := os.environ.get(env_key, "").strip():
            base = Path(cann).expanduser()
            roots.extend(
                str(base / sub)
                for sub in (
                    "x86_64-linux/asc/include",
                    "aarch64-linux/asc/include",
                    "asc/include",
                    "x86_64-linux/include",
                    "include",
                )
            )
    sibling_tl = Path(__file__).resolve().parents[5].parent / "tilelang" / "src" / "tl_templates" / "ascend"
    if sibling_tl.is_dir():
        roots.append(str(sibling_tl))
    return tuple(dict.fromkeys(roots))


def _default_tilelang_codegen_path() -> str:
    env_tl = (
        os.environ.get("TILEFOUNDRY_NPU_TILELANG_REPO", "").strip()
        or os.environ.get("TILEFOUNDRY_NPU_TILELANG_SOURCE", "").strip()
    )
    root = Path(env_tl).expanduser() if env_tl else Path(__file__).resolve().parents[5].parent / "tilelang"
    return str(root / "tilelang" / "ascend" / "codegen.py")


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


_QUALIFIED_SDK_RE = re.compile(r"\b(Reg|Simt|LoadDist|StoreDist)::([A-Za-z_][A-Za-z0-9_]*)")
_NS_DECL_RE = re.compile(
    r"(?:__aicore__|__simd_callee__|__simt_callee__|inline|static|constexpr|template\s*<[^>]*>)*"
    r"\s*(?:void|bool|int\d*_t|uint\d*_t|float|double|auto|T|U|struct|class)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(|<|\{)"
)
_ENUM_BLOCK_RE = re.compile(r"enum\s+(?:class\s+)?(LoadDist|StoreDist)\b[^{]*\{([^}]*)\}", re.DOTALL)
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
_SDK_SKIP_IDENTS = frozenset({"if", "for", "while", "switch", "return", "inline", "void"})


def _extract_namespace_symbols(text: str, ns_name: str, out: set[str]) -> None:
    ns_pat = re.compile(rf"\bnamespace\s+{ns_name}\s*\{{")
    n = len(text)
    for m in ns_pat.finditer(text):
        start, pos, depth = m.end(), m.end(), 1
        while pos < n and depth > 0:
            depth += (text[pos] == "{") - (text[pos] == "}")
            pos += 1
        for dm in _NS_DECL_RE.finditer(text[start : pos - 1]):
            if (ident := dm.group(1)) not in _SDK_SKIP_IDENTS:
                out.add(f"{ns_name}::{ident}")


def scan_ascend_sdk_headers(
    header_roots: Sequence[str | Path] | str | Path | None = None,
    *,
    allow_offline_catalog: bool = False,
) -> SdkHeaderScanResult:
    """Mechanically scan installed Ascend C++ SDK headers on disk."""
    if header_roots is None:
        roots = [Path(p) for p in DEFAULT_ASCEND_SDK_HEADER_ROOTS]
    elif isinstance(header_roots, (str, Path)):
        roots = [Path(header_roots)]
    else:
        roots = [Path(p) for p in header_roots]

    existing_roots = [r for r in roots if r.is_dir()]
    symbols: set[str] = set()
    cats: dict[str, set[str]] = {
        "Reg": set(),
        "Simt": set(),
        "LoadDist": set(),
        "StoreDist": set(),
        "sync_dma_compute": set(),
    }

    def _add_sym(sym: str, cat: str | None = None) -> None:
        symbols.add(sym)
        if cat is not None:
            cats[cat].add(sym)
            return
        prefix = sym.split("::", 1)[0] if "::" in sym else ""
        cats[prefix if prefix in cats else "sync_dma_compute"].add(sym)

    seen_real: set[Path] = set()
    scanned_count = 0
    for root in existing_roots:
        for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
            for fname in filenames:
                if not fname.endswith((".h", ".hpp")):
                    continue
                try:
                    resolved = (Path(dirpath) / fname).resolve()
                except OSError:
                    continue
                if resolved in seen_real or not resolved.is_file():
                    continue
                seen_real.add(resolved)
                scanned_count += 1
                try:
                    content = resolved.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for m in _QUALIFIED_SDK_RE.finditer(content):
                    _add_sym(f"{m.group(1)}::{m.group(2)}", m.group(1))
                for ns in ("Reg", "Simt"):
                    if f"namespace {ns}" in content:
                        before = len(cats[ns])
                        _extract_namespace_symbols(content, ns, cats[ns])
                        if len(cats[ns]) > before:
                            symbols.update(cats[ns])
                if "LoadDist" in content or "StoreDist" in content:
                    for em in _ENUM_BLOCK_RE.finditer(content):
                        enum_name, body = em.group(1), em.group(2)
                        for mem in _ENUM_MEMBER_RE.finditer(body):
                            _add_sym(f"{enum_name}::{mem.group(1)}", enum_name)
                for im in _NAMED_ISA_INTRINSICS_RE.finditer(content):
                    _add_sym(im.group(1), "sync_dma_compute")

    fallback = scanned_count == 0 or not symbols
    if fallback:
        if not allow_offline_catalog:
            raise FileNotFoundError(
                f"No Ascend SDK or tl_templates headers found under {tuple(str(r) for r in existing_roots)}"
            )
        for spec in DAV3510_CANN_ISA_CATALOG:
            for part in spec.cann_intrinsic.split("/"):
                if tok := part.strip():
                    _add_sym(tok)

    return SdkHeaderScanResult(
        source="catalog_fallback" if fallback else "sdk_headers",
        fallback_used=fallback,
        scanned_roots=tuple(str(r) for r in existing_roots),
        header_files_scanned=0 if fallback else scanned_count,
        symbols=frozenset(symbols),
        by_category={k: tuple(sorted(v)) for k, v in cats.items()},
    )


def _scan_tilelang_codegen_file(codegen_path: Path) -> set[str]:
    found: set[str] = set()
    if not codegen_path.is_file():
        return found
    text = codegen_path.read_text(encoding="utf-8", errors="ignore")
    for pat, fmt in (
        (r'["\'](target\.build\.[A-Za-z0-9_]+)["\']', "{}"),
        (r'\b([A-Z][A-Z0-9_]*_CODEGEN)\b', "{}"),
        (r'DeviceCodegen\(\s*["\']([A-Za-z0-9_]+)["\']', "codegen:{}"),
    ):
        found.update(fmt.format(m.group(1)) for m in re.finditer(pat, text))

    repo_root = codegen_path.resolve().parents[2]
    for cpp_name in ("codegen_ascend.cc", "intrin_rule_ascend.cc"):
        cpp_file = repo_root / "src" / "ascend" / "codegen" / cpp_name
        if not cpp_file.is_file():
            continue
        cpp_text = cpp_file.read_text(encoding="utf-8", errors="ignore")
        for pat in (
            r"\btl::(ascend_[A-Za-z0-9_]+)\s*\(",
            r'["\']tl\.(ascend_[A-Za-z0-9_]+)["\']',
            r'["\']tl\.(simd\.[A-Za-z0-9_]+)["\']',
        ):
            found.update(m.group(1) for m in re.finditer(pat, cpp_text))
    return found


def discover_tilelang_lowering_surface(
    tilelang_module: object | None = None,
    *,
    codegen_path: str | Path | None = None,
    allow_offline_catalog: bool = False,
) -> tuple[str, frozenset[str]]:
    """Discover registered TileLang Ascend lowering symbols from language + codegen.py."""
    mod, source = tilelang_module, "explicit_module"
    if mod is None:
        try:
            import tilelang.ascend.language as tl_ascend  # noqa: PLC0415

            mod, source = tl_ascend, "tilelang.ascend.language"
        except ImportError:
            mod = None

    symbols: set[str] = set()
    if mod is not None:
        symbols.update(n for n in dir(mod) if not n.startswith("_"))
        for sub_attr in ("simd", "simt"):
            if (sub_obj := getattr(mod, sub_attr, None)) is not None:
                symbols.update(f"{sub_attr}.{n}" for n in dir(sub_obj) if not n.startswith("_"))

    cg_candidates: list[Path] = [Path(codegen_path)] if codegen_path is not None else []
    if codegen_path is None:
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
        if not allow_offline_catalog:
            raise ImportError("tilelang.ascend.language and codegen.py are both unavailable")
        return "baseline_fallback", BASELINE_TILELANG_ASCEND_SURFACE
    if codegen_scanned and source == "tilelang.ascend.language":
        source = "tilelang.ascend.language+codegen"
    elif mod is None and codegen_scanned:
        source = "tilelang.ascend.codegen"
    return source, frozenset(symbols)


def _resolve_surface(
    tilelang_surface: Iterable[str] | object | None,
    *,
    allow_offline_catalog: bool = False,
) -> tuple[str, frozenset[str]]:
    if tilelang_surface is None:
        return discover_tilelang_lowering_surface(allow_offline_catalog=allow_offline_catalog)
    if isinstance(tilelang_surface, (set, frozenset, list, tuple)):
        return "explicit_surface", frozenset(str(s) for s in tilelang_surface)
    return discover_tilelang_lowering_surface(tilelang_surface, allow_offline_catalog=allow_offline_catalog)


def scan_tilelang_pipe_gaps(
    *,
    tilelang_surface: Iterable[str] | object | None = None,
    pipes: Iterable[str] | None = None,
    cann_catalog: Iterable[IsaPrimitiveSpec] | None = None,
    allow_offline_catalog: bool = False,
) -> PipeGapReport:
    """Diff CANN dav-3510 hardware capabilities against TileLang across the 7 hardware pipes."""
    surface_source, surface = _resolve_surface(
        tilelang_surface, allow_offline_catalog=allow_offline_catalog
    )
    selected_pipes = tuple(pipes) if pipes is not None else VALID_HARDWARE_PIPES
    for pipe in selected_pipes:
        if pipe not in VALID_HARDWARE_PIPES:
            raise ValueError(f"Unknown hardware pipe {pipe!r}; expected one of {VALID_HARDWARE_PIPES}")
    catalog = tuple(cann_catalog) if cann_catalog is not None else DAV3510_CANN_ISA_CATALOG

    pipe_reports: dict[str, PipeGapBreakdown] = {}
    all_supported: list[str] = []
    all_missing: list[str] = []
    for pipe in selected_pipes:
        pipe_specs = [s for s in catalog if s.hardware_pipe == pipe]
        supported = [s.name for s in pipe_specs if any(sym in surface for sym in s.tilelang_symbols)]
        missing = [s.name for s in pipe_specs if not any(sym in surface for sym in s.tilelang_symbols)]
        total = len(pipe_specs)
        pipe_reports[pipe] = PipeGapBreakdown(
            pipe=pipe,
            total_capabilities=total,
            supported_primitives=tuple(supported),
            missing_primitives=tuple(missing),
            coverage_ratio=round((len(supported) / float(total)) if total > 0 else 1.0, 4),
        )
        all_supported.extend(supported)
        all_missing.extend(missing)

    total_all = len(all_supported) + len(all_missing)
    return PipeGapReport(
        surface_source=surface_source,
        pipes=pipe_reports,
        supported_primitives=tuple(all_supported),
        missing_primitives=tuple(all_missing),
        coverage_ratio=round((len(all_supported) / float(total_all)) if total_all > 0 else 1.0, 4),
    )


__all__ = [
    "DAV3510_CANN_ISA_CATALOG",
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
