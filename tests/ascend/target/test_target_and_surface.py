"""Consolidated L0 tests for Ascend target facts, resource masks, dialect surface, and AST gate."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest

from tests.core.target._hardware import (
    bare_projection_cases,
    check_bare_projection,
    check_document_evidence,
    check_hardware_defaults,
)
from tests.core.target._instructions import check_architecture_capabilities
from tilefoundry.analysis.facts import (
    MemoryHierarchyFacts,
    MemoryRelationKind,
    ThroughputFacts,
)
from tilefoundry.ir.types import BoolDType, DType, FloatDType, IntegerDType, Topology
from tilefoundry.target import (
    TopologyFacts,
    TopologyLevelFacts,
    UnsupportedCapabilityError,
    registered_targets,
)
from tilefoundry_ascend import PURE_SIMDVF_UBUF_FLAGS, AscendTarget, NodeResourceMask
from tilefoundry_ascend.agent.kb.schema import (
    AscendCausalRule,
    default_seed_rules_path,
    load_seed_rules,
    normalize_runtime_backend,
)
from tilefoundry_ascend.agent.orchestrator.cpp_slice_protocol import apply_scoped_trackb_patch
from tilefoundry_ascend.agent.orchestrator.gate_funnel import _check_custom_kernel_redlines
from tilefoundry_ascend.agent.orchestrator.llm_constraints import format_target_dialect_layer
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import build_syntax_repair_request
from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask, TopologyAxes
from tilefoundry_ascend.ascend.target import (
    ASCEND_950DT_ID,
    ASCEND_950PR_ID,
    DEFAULT_CANN_SERIES,
    TILELANG_DSL_SURFACE,
    VALID_CANN_SERIES,
    AscendSpaceFacts,
    cube_macs_per_cycle,
    detect_runtime_core_counts,
    get_dialect_surface,
    load_cann_surface,
    resolve_cann_series,
)
from tilefoundry_ascend.ascend.target import spec as ascend_spec
from tilefoundry_ascend.ascend.target.dialect_surface import (
    CCE_CPP20_SURFACE,
    INLINE_ASCENDC_VF_SURFACE,
    INLINE_CCE_VF_SURFACE,
    TL_TEMPLATES_CPP20_SURFACE,
    DialectSurfaceSpec,
    PrimitiveEntry,
    filter_primitives_on_demand,
    render_layer2_surface,
    resolve_trackb_cce_surface,
)
from tilefoundry_ascend.ascend.target.facts import ascend_space_facts
from tilefoundry_ascend.ascend.target.hardware import BUILTIN_HARDWARE_DOCUMENTS
from tilefoundry_ascend.ascend.target.ast_semantic_gate import (
    SemanticViolation,
    TileLangSemanticGate,
    check_tilelang_semantic_contracts,
)
from tilefoundry_ascend.ascend.templates.ascendc_templates import ASCENDC_RENDERERS

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CANN_SURFACE_DIR = (
    _REPO_ROOT / "targets" / "ascend" / "tilefoundry_ascend" / "ascend" / "target" / "cann_surface"
)
TRACKB_ALIASES = (
    "cce_cpp20",
    "cce_trackb",
    "hybrid_trackb",
    "tl_templates_cpp20",
    "ascendc_trackb",
    "ascendc_template",
)
FORBIDDEN_CCE_TOKENS = (
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
)
CLEAN_TRACKB_SOURCE = (
    "#include <tl_templates/ascend/common.h>\n"
    'extern "C" __global__ void kernel_active() {\n'
    "    asc_copy_gm2ub_align(ub, gm, 128);\n"
    "}\n"
)


def _positive_ascendc_lines(text: str) -> list[str]:
    return [
        line
        for line in text.splitlines()
        if "AscendC" in line and "Strictly Forbidden" not in line and "Do NOT emit" not in line
    ]


def _patch(search: str, replace: str) -> str:
    return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


def _resolve_tilelang_root() -> Path | None:
    env_root = os.environ.get("TILELANG_ROOT", "").strip()
    if env_root and Path(env_root).is_dir():
        return Path(env_root)
    for cand in (_REPO_ROOT.parent / "tilelang", _REPO_ROOT.parents[1] / "tilelang"):
        if cand.is_dir():
            return cand
    return None


@pytest.mark.parametrize(
    ("construct", "message"),
    bare_projection_cases(
        lambda architecture: AscendTarget(architecture=architecture),
        AscendTarget,
        "ascend",
        (
            "AscendTarget.architecture must be an installed ID string, a document path, or "
            "a DaVinci950; got _BareArchitecture",
            "AscendTarget.device must be an installed ID string, a document path, or "
            "an Ascend950PR; got _BareDevice",
        ),
    ),
)
def test_hardware_documents_and_bare_projections(construct, message: str) -> None:
    """Enforce hardware document ownership, defaults, capabilities, and registration."""
    check_bare_projection(construct, message)
    assert {"ascend"} <= registered_targets().keys()
    check_hardware_defaults(
        AscendTarget.hardware,
        (ascend_spec.DAVINCI_950_ID, ascend_spec.ASCEND_950PR_ID, ascend_spec.ASCEND_950DT_ID),
    )
    check_document_evidence(
        AscendTarget.hardware,
        {
            "ascend.davinci_950": {
                "memory.tmem.owner": "aic",
                "memory.smem.owner": "aiv",
                "memory.rmem.owner": "lane",
                "memory.ssbuf.owner": "block",
            },
            "ascend.950pr": {"memory.hbm.owner": "target"},
            "ascend.950dt": {"memory.hbm.owner": "target"},
        },
    )
    check_architecture_capabilities(AscendTarget, "tilefoundry_ascend.")


def test_ascend_target_topology_memory_hierarchy_and_burst_curve() -> None:
    """Verify 950PR topology, memory hierarchy relations, UB capacities, and HBM burst curve."""
    target = AscendTarget()
    assert AscendTarget("ascend.950pr") == target
    assert target.to_python().text == 'AscendTarget("ascend.950pr")'

    levels = target.get_facts(TopologyFacts)
    assert tuple(lvl.name for lvl in levels.topologies) == ("block", "aic", "aiv", "lane")
    assert levels.level("block") == TopologyLevelFacts("block", 28, 28)
    assert levels.level("aic") == TopologyLevelFacts("aic", 1, 28)
    assert levels.level("aiv") == TopologyLevelFacts("aiv", 2, 56)
    assert levels.level("lane") == TopologyLevelFacts("lane", 64, 56 * 64)
    assert levels.parallel() == levels.level("block")

    with pytest.raises(ValueError, match="unsupported topology level 'cta'"):
        target.validate_program_topology(Topology("cta", 4))
    with pytest.raises(UnsupportedCapabilityError, match="no Facts projection"):
        target.get_facts(TopologyLevelFacts, "cta")
    target.validate_program_topology(Topology("block", 28))
    with pytest.raises(ValueError, match="must satisfy 1 <= extent <= 28"):
        target.validate_program_topology(Topology("block", 29))

    assert target.get_facts(ThroughputFacts).bandwidth_level == "gmem"
    hierarchy = target.get_facts(MemoryHierarchyFacts)
    assert {lvl.name: lvl.owner for lvl in hierarchy.explicit_levels} == {
        "host": "target",
        "gmem": "target",
        "tmem": "aic",
        "smem": "aiv",
        "rmem": "lane",
    }
    by_name = {lvl.name: lvl for lvl in hierarchy.explicit_levels}
    assert by_name["smem"].capacity_bytes == 248 * 1024
    simtvf_lvls = {
        lvl.name: lvl for lvl in target.get_facts(MemoryHierarchyFacts, "simtvf").explicit_levels
    }
    assert simtvf_lvls["smem"].capacity_bytes == 216 * 1024
    unres_lvls = {
        lvl.name: lvl
        for lvl in target.get_facts(MemoryHierarchyFacts, "simdvf_unreserved").explicit_levels
    }
    assert unres_lvls["smem"].capacity_bytes == 256 * 1024
    assert tuple(lvl.name for lvl in hierarchy.implicit_levels) == ("simt_dcache",)
    sharing = [r for r in hierarchy.relations if r.kind is MemoryRelationKind.SHARES_CAPACITY_WITH]
    assert len(sharing) == 1
    assert (sharing[0].near, sharing[0].far, sharing[0].shared_capacity_bytes) == (
        "simt_dcache",
        "smem",
        256 * 1024,
    )

    spaces = ascend_space_facts(target)
    assert (spaces.ub_nominal_bytes, spaces.ub_usable_bytes) == (256 * 1024, 216 * 1024)
    assert (spaces.ub_simtvf_usable_bytes, spaces.ub_simdvf_usable_bytes) == (216 * 1024, 248 * 1024)
    assert spaces.ub_limit_bytes("simtvf") == 216 * 1024
    assert spaces.ub_limit_bytes("simdvf") == 248 * 1024
    assert spaces.ub_limit_bytes("simdvf", unreserved_ubuf=True) == 256 * 1024
    assert target.get_facts(AscendSpaceFacts, "simdvf").ub_usable_bytes == 248 * 1024
    assert target.get_facts(AscendSpaceFacts, "simdvf_unreserved").ub_usable_bytes == 256 * 1024
    assert target.get_facts(AscendSpaceFacts, "simtvf").ub_usable_bytes == 216 * 1024
    assert PURE_SIMDVF_UBUF_FLAGS == (
        "--cce-disable-vf-stack-reserved-ubuf",
        "--cce-disable-asc-reserved-ubuf",
    )
    assert (spaces.l1_bytes, spaces.l0a_bytes, spaces.l0b_bytes, spaces.l0c_bytes) == (
        512 * 1024,
        64 * 1024,
        64 * 1024,
        256 * 1024,
    )
    assert spaces.hbm_burst_bandwidth_table[0][0] == 4096
    assert spaces.hbm_burst_bandwidth_table[-1][0] == 65536
    assert spaces.effective_hbm_bandwidth_bps(4096) < spaces.effective_hbm_bandwidth_bps(32768)
    assert spaces.dma_burst_efficiency(4096) < 0.70
    assert spaces.dma_burst_efficiency(32768) == pytest.approx(1.0)


def test_dtypes_950dt_dav3510_facts_and_runtime_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Ascend 950 DTypes, 950DT registration, dav-3510 facts, and offline core probing."""
    for name in ("f32", "f16", "bf16", "fp8e4m3", "f8e8m0", "f4e2m1", "i32", "i64", "bool"):
        assert DType.from_name(name) is getattr(DType, name)
    for name, bits, signed in (
        ("i8", 8, True),
        ("u8", 8, False),
        ("i16", 16, True),
        ("u16", 16, False),
        ("u32", 32, False),
        ("u64", 64, False),
        ("i4", 4, True),
        ("u4", 4, False),
    ):
        assert DType.from_name(name) == IntegerDType(name=name, bit_width=bits, signed=signed)
    for name, bits, exp, man in (("fp8e5m2", 8, 5, 2), ("hif8", 8, 4, 3), ("f4e1m2", 4, 1, 2)):
        assert DType.from_name(name) == FloatDType(
            name=name, bit_width=bits, exponent_bits=exp, mantissa_bits=man
        )
    assert isinstance(DType.bool, BoolDType)

    assert set(BUILTIN_HARDWARE_DOCUMENTS) >= {
        "davinci_950.toml",
        "ascend_950pr.toml",
        "ascend_950dt.toml",
    }
    assert {ASCEND_950PR_ID, ASCEND_950DT_ID} <= {t.device_id for t in AscendTarget.available()}

    dt = AscendTarget("ascend.950dt")
    assert (dt.device.aic_count, dt.device.aiv_count, dt.device.sm_count) == (36, 72, 36)
    assert dt.device.hbm_capacity_bytes == 144 * 1024 * 1024 * 1024
    assert dt.device.hbm_bandwidth_bytes_per_second == 4_000_000_000_000
    assert dt.device.clock_ghz == pytest.approx(1.65)
    assert dt.to_python().text == 'AscendTarget("ascend.950dt")'
    assert dt.get_facts(TopologyFacts).level("aiv") == TopologyLevelFacts("aiv", 2, 72)

    expected_macs = {
        "f4e2m1": 16384,
        "f4e1m2": 16384,
        "fp8e4m3": 8192,
        "fp8e5m2": 8192,
        "hif8": 8192,
        "i8": 8192,
        "u8": 8192,
        "f16": 4096,
        "bf16": 4096,
        "hf32": 2048,
        "f32": 256,
    }
    for dtype_name, macs in expected_macs.items():
        assert cube_macs_per_cycle(dtype_name) == macs
    assert cube_macs_per_cycle("f32", hf32=True) == 2048
    with pytest.raises(ValueError, match="unsupported Ascend 950 Cube dtype"):
        cube_macs_per_cycle("i4")

    for target_id in (ASCEND_950PR_ID, ASCEND_950DT_ID):
        target = AscendTarget(target_id)
        spaces = target.get_facts(AscendSpaceFacts)
        assert target.architecture.cube_macs_per_cycle_map == expected_macs
        assert spaces.cube_macs_per_cycle("fp8e4m3") == 8192
        assert (spaces.l1_banks, spaces.l1_bank_bytes) == (2, 262144)
        assert (spaces.ub_bank_groups, spaces.ub_banks_per_group, spaces.ub_bank_bytes) == (
            8,
            2,
            16384,
        )
        assert (spaces.ub_datablock_bytes, spaces.bias_table_bytes, spaces.fixbuf_bytes) == (
            32,
            4096,
            16384,
        )
        assert (spaces.l2_capacity_bytes, spaces.l2_bytes_per_cycle_per_aic) == (134217728, 101)
        assert (spaces.l0a_layout, spaces.l0b_layout, spaces.l0c_layout) == ("NZ", "ZN", "NZ")
        assert (spaces.max_crosscore_flag_depth, spaces.hardware_mutex_count) == (15, 28)
        assert (spaces.max_mmad_mnk, spaces.forbid_cube_int4) == (4095, True)

    aic, aiv = detect_runtime_core_counts()
    assert aic > 0 and aiv == 2 * aic
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    assert detect_runtime_core_counts(default_aic=36) == (36, 72)
    monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/local/sbin/npu-smi")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="\tAicore Freq(MHZ) : 1650\n\tAicore Count : 36\n",
        ),
    )
    assert detect_runtime_core_counts(default_aic=28) == (36, 72)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("npu-smi failed")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert detect_runtime_core_counts(default_aic=28) == (28, 56)


def test_coupled_core_resource_mask_ssbuf_and_dma_paths() -> None:
    """Verify 24-bit NodeResourceMask, SSBUF parameters, measured HBM, and DMA path rules."""
    m_aiv0 = NodeResourceMask.for_aiv0("MTE2", "VECTOR")
    assert (m_aiv0.aiv0_bits, m_aiv0.aic_bits, m_aiv0.aiv1_bits) == ((1 << 1) | (1 << 4), 0, 0)
    assert int(m_aiv0) == (1 << 1) | (1 << 4)
    m_aic = NodeResourceMask.for_aic("MTE1", "CUBE", "FIXPIPE")
    assert not m_aiv0.conflicts_with(m_aic)
    m_aiv1 = NodeResourceMask.for_aiv1("MTE3", "SCALAR")
    assert not m_aiv0.conflicts_with(m_aiv1)
    m_dual = NodeResourceMask.for_dual_aiv("MTE2", "VECTOR")
    assert m_dual.conflicts_with(m_aiv0)
    assert not m_dual.conflicts_with(NodeResourceMask.for_aic("MTE2"))

    for fn, arg in (
        (NodeResourceMask.for_aiv0, "CUBE"),
        (NodeResourceMask.for_aiv1, "FIXPIPE"),
        (NodeResourceMask.for_aic, "VECTOR"),
    ):
        with pytest.raises(ValueError, match="does not support pipe"):
            fn(arg)
    with pytest.raises(ValueError, match="out of range"):
        NodeResourceMask(1 << 24)
    with pytest.raises(ValueError, match="invalid pipe bits"):
        NodeResourceMask(1 << 3)

    combined = NodeResourceMask.for_dual_aiv("MTE2") | NodeResourceMask.for_aic("PIPE_FIX")
    assert combined.to_solver_resources(duration=2, offset=1) == (
        ("aiv", "aiv0", "MTE2", 2, 1),
        ("aic", "aic", "FIXPIPE", 2, 1),
        ("aiv", "aiv1", "MTE2", 2, 1),
    )

    target = AscendTarget()
    spaces = target.get_facts(AscendSpaceFacts)
    assert (spaces.ssbuf_bytes, spaces.limit_bytes("SSBUF")) == (3072, 3072)
    assert (spaces.ssbuf_align_bytes, spaces.ssbuf_latency_cycles, spaces.ssbuf_owner) == (
        8,
        8,
        "block",
    )
    assert (spaces.max_mte2_inflight_per_aiv, spaces.fixpipe_bytes_per_cycle) == (16, 128)
    assert target.device.hbm_bandwidth_bytes_per_second == 1_633_108_573_296
    assert AscendTarget.hardware.documents()["ascend.950pr"].fact("memory.hbm.bandwidth").origin == (
        "measured"
    )
    assert spaces.is_dma_path_legal("L0C", "UB") is True
    assert spaces.is_dma_path_legal("UB", "L1") is True
    assert spaces.is_dma_path_legal("L1", "GM") is False
    assert spaces.is_dma_path_legal("GM", "L0A") is False
    assert spaces.is_dma_path_legal("GM", "L0B") is False


def test_cann_surface_loader_and_dav3510_tilelang_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify CANN 9.3 snapshot loading, series resolution, and absorbed dav-3510 surface."""
    target = AscendTarget(ASCEND_950PR_ID)
    arch = target.architecture
    assert (arch.name, arch.series, arch.cce_arch) == ("dav_3510", "ascend_950", "dav-3510")
    assert arch.supported_skus == ("ascend_950pr", "ascend_950dt")
    assert (arch.default_cann_series, arch.supported_cann_series) == ("cann_9_3", ("cann_9_3",))

    for var in (
        "TILEFOUNDRY_CANN_SERIES",
        "TILEFOUNDRY_NPU_CANN_SERIES",
        "ASCEND_HOME_PATH",
        "ASCEND_TOOLKIT_HOME",
    ):
        monkeypatch.delenv(var, raising=False)
    assert resolve_cann_series() == DEFAULT_CANN_SERIES == "cann_9_3"
    monkeypatch.setenv("ASCEND_HOME_PATH", "/opt/Ascend/cann-9.3.0")
    assert resolve_cann_series() == "cann_9_3"
    monkeypatch.setenv("TILEFOUNDRY_NPU_CANN_SERIES", "cann_9_3")
    assert resolve_cann_series() == "cann_9_3"
    monkeypatch.setenv("TILEFOUNDRY_CANN_SERIES", "cann_9_3")
    assert resolve_cann_series("cann_9_3") == "cann_9_3"
    with pytest.raises(ValueError, match="Unsupported CANN series"):
        resolve_cann_series("cann_8_0")

    assert VALID_CANN_SERIES == ("cann_9_3",)
    raw = tomllib.loads((_CANN_SURFACE_DIR / "cann_9_3.toml").read_text(encoding="utf-8"))
    assert raw["surface"]["cann_series"] == "cann_9_3"
    spec = load_cann_surface("cann_9_3")
    assert spec.dialect_id == "cce_cpp20"
    by_sym = {entry.symbol: entry for entry in spec.primitives}
    assert {"ascend_gemm_l1", "asc_copy_l0c2ub", "asc_copy_l0c2gm", "asc_mmad_mx"} <= set(by_sym)
    for entry in spec.primitives:
        assert "..." not in entry.signature
        assert entry.pipes and entry.facets and entry.required_headers
    assert "#include <c_api/asc_cube.h>" in spec.required_headers_or_imports

    surf_93 = get_dialect_surface("cce_trackb")
    assert resolve_trackb_cce_surface("cce_trackb") is surf_93
    sig_93 = {e.symbol: e.signature for e in surf_93.primitives}["asc_copy_l0c2gm"]
    assert "asc_store_l2_cache_mode" in sig_93 and sig_93.count(",") == 13

    tl_by_sym = {entry.symbol: entry for entry in TILELANG_DSL_SURFACE.primitives}
    assert len(tl_by_sym) == len(TILELANG_DSL_SURFACE.primitives)
    expected_tl = {
        "T.Task": "latency=",
        "T.PerCoreTask": "T.PerCoreTask",
        "T.copy": "l2_cache_ctrl=",
        "T.annotate_buffer_versions": "T.annotate_buffer_versions({buf: stages})",
        "T.assume_no_conflict": "group=",
        "T.assume_conflict": "cross=",
        "T.SimdVF": "latency=",
        "T.SimtVF": "threads=128, latency=",
        "T.Stage": "with T.Stage(stage_id):",
        "T.simd.vld": 'dist="NORM_B32" | "US_B16" | "BRC_B32"',
        "T.simd.vsts": 'dist="NORM_B32" | "ONEPT_B32" | "PK_B32", extent=64',
        "T.simd.vcvt": "T.simd.vcvt(vec, dtype, mask=mask, part=0)",
        "T.simd.vexp": "T.simd.vexp(vec, mask=mask)",
    }
    for sym, needle in expected_tl.items():
        assert needle in tl_by_sym[sym].signature
    assert "latency >= ii > 0" in tl_by_sym["T.Task"].pitfall
    assert "{buf: 1 | count | (count, mode) | mode}" in tl_by_sym["T.annotate_buffer_versions"].pitfall
    assert "threads=" not in TILELANG_DSL_SURFACE.forbidden_tokens

    rendered_full = render_layer2_surface(
        TILELANG_DSL_SURFACE,
        primitives=filter_primitives_on_demand(
            TILELANG_DSL_SURFACE,
            topology_axes=TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd"),
        ),
        max_chars=10000,
    )
    assert len(rendered_full) <= 10000
    notes_len = sum(
        len(line) + 1
        for line in rendered_full.splitlines()
        if line.startswith("- ") and "[" not in line
    )
    assert notes_len > 700
    assert "Prefer compiler-managed multibuffering" in rendered_full
    assert "T.simd.vrsqrt does not exist" in rendered_full

    synthetic = DialectSurfaceSpec(
        dialect_id="tilelang_dsl",
        display_label="Synthetic",
        required_headers_or_imports=("import tilelang",),
        primitives=(PrimitiveEntry("T.Kernel", "SCALAR", "grid_and_workspace_topology", "T.Kernel(n)"),),
        forbidden_tokens=(),
        prompt_notes=(
            "Optional tuning suggestion one with plenty of explanatory detail.",
            "You must keep DMA buffers aligned to 32 bytes.",
            "never allocate dynamic local arrays inside SIMD bodies.",
            "Forbid cross-lane scalar store aliasing.",
            "Do not use threads= on T.Kernel.",
        ),
    )
    tight = render_layer2_surface(synthetic, include_guardrails=False, max_chars=315)
    assert "must keep DMA buffers" in tight and "Optional tuning suggestion" not in tight

    unlocked = {
        e.symbol
        for e in filter_primitives_on_demand(
            TILELANG_DSL_SURFACE,
            topology_axes=TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd"),
            freedom_mask=ConstraintRelaxationMask(pipeline_and_sync_graph=True),
        )
    }
    assert {"T.Stage", "T.Task", "T.annotate_buffer_versions"} <= unlocked


def test_cce_cpp20_surface_hygiene_redlines_and_scoped_patches() -> None:
    """Enforce Track-B CCE C++20 surface isolation, redlines, KB aliases, and patch checks."""
    assert CCE_CPP20_SURFACE.dialect_id == "cce_cpp20"
    assert CCE_CPP20_SURFACE.display_label == "CCE C++20 (dav-3510 / CANN 9.3)"
    assert TL_TEMPLATES_CPP20_SURFACE is CCE_CPP20_SURFACE
    assert set(FORBIDDEN_CCE_TOKENS) <= set(CCE_CPP20_SURFACE.forbidden_tokens)
    rendered = CCE_CPP20_SURFACE.render_prompt_reference()
    assert "The __simd_vf__ stack frame is exactly 0 bytes" in rendered
    assert "Do NOT emit high-level AscendC classes" in rendered
    for token in FORBIDDEN_CCE_TOKENS:
        assert f"`{token}`" in rendered

    for alias in TRACKB_ALIASES:
        assert resolve_trackb_cce_surface(alias) is CCE_CPP20_SURFACE
        assert get_dialect_surface(alias) is CCE_CPP20_SURFACE
        prompt = format_target_dialect_layer(escalation_tier=alias)
        assert "asc_copy_gm2ub_align" in prompt and "T.Kernel" not in prompt
        assert _positive_ascendc_lines(prompt) == []

    assert resolve_trackb_cce_surface("") is CCE_CPP20_SURFACE
    with pytest.raises(ValueError, match="Unsupported Track-B CCE dialect identifier"):
        resolve_trackb_cce_surface("tilelang_dsl")
    assert INLINE_ASCENDC_VF_SURFACE is INLINE_CCE_VF_SURFACE
    assert get_dialect_surface("inline_cce_vf") is INLINE_CCE_VF_SURFACE
    assert get_dialect_surface("inline_ascendc_vf") is INLINE_CCE_VF_SURFACE
    assert get_dialect_surface("tilelang_dsl") is TILELANG_DSL_SURFACE
    assert get_dialect_surface("python_kernel") is TILELANG_DSL_SURFACE

    tb_req = build_syntax_repair_request(
        "__global__ void kernel_active() {}", "err", backend_kind="ascendc_trackb"
    )
    assert "```cpp" in tb_req.user_prompt
    assert _positive_ascendc_lines(f"{tb_req.system_prompt}\n{tb_req.user_prompt}") == []
    tl_req = build_syntax_repair_request("def f():\n    pass\n", "err")
    assert "```python" in tl_req.user_prompt and "T.MixedKernel" in tl_req.user_prompt

    flagged = (
        "__simd_vf__ inline void vf(__ubuf__ float* x) {\n  float tmp[64];\n}\n",
        "__simd_vf__ inline void vf(__ubuf__ float* x) {\n  uint16_t idx[8];\n}\n",
    )
    for src in flagged:
        assert len(_check_custom_kernel_redlines(src)) == 1
    for src in (
        '// float tmp[64];\n__simd_vf__ inline void vf(__ubuf__ float* x) {\n  x[0] = 1.0f;\n}\n',
        '__simd_vf__ inline void vf(__ubuf__ float* x) {\n  vector_f32 rg0[4];\n}\n',
        'float tmp[64];\nextern "C" __global__ void k() {}\n',
    ):
        assert _check_custom_kernel_redlines(src) == []

    assert normalize_runtime_backend("cce_cpp20") == "ascendc_trackb"
    r02 = {r.rule_id: r for r in load_seed_rules()}["R02_ONCHIP_L0C_TO_UB_DIRECT"]
    assert "asc_copy_l0c2ub" in r02.render_for_backend("cce_trackb").transform_template
    assert "T.dual_copy" in r02.render_for_backend("hybrid_trackb").transform_template
    roundtrip = AscendCausalRule.from_dict(
        {
            "rule_id": "R_ALIAS",
            "regime_guard": "is_legal",
            "precondition": "PIPE_V > 0",
            "transform_template": "Opt.",
            "expected_postcondition": "PIPE_V <= prev.PIPE_V",
            "source_provenance": "flywheel:test",
            "compatible_backends": ["cce_cpp20", "hybrid_trackb"],
        }
    )
    assert roundtrip.supports_backend("cce_trackb")
    seed_text = default_seed_rules_path().read_text(encoding="utf-8")
    assert "AscendC" not in seed_text and all(tok not in seed_text for tok in FORBIDDEN_CCE_TOKENS)

    legal_cpp = (
        "simd_inst::set_flag(PIPE_V, PIPE_MTE3, 0);\nasc_set_flag(PIPE_V, PIPE_MTE3, 0);\n"
        "myTPipe\nDataCopyPadV2\nmy_copy_gm_to_ubuf\n"
    )
    assert all(v == 0 for v in CCE_CPP20_SURFACE.count_forbidden_tokens(legal_cpp).values())
    assert CCE_CPP20_SURFACE.introduced_forbidden_tokens(
        "set_flag(PIPE_V, PIPE_MTE3, 0);\n",
        "set_flag(PIPE_V, PIPE_MTE3, 0);\nset_flag(PIPE_MTE3, PIPE_V, 0);\n",
    ) == ("set_flag(",)

    with pytest.raises(ValueError, match="introduces forbidden symbols: AscendC::"):
        apply_scoped_trackb_patch(
            CLEAN_TRACKB_SOURCE,
            _patch("    asc_copy_gm2ub_align(ub, gm, 128);", "    AscendC::DataCopyPad(ub, gm, 128);"),
            active_kernel="kernel_active",
        )
    updated, applied = apply_scoped_trackb_patch(
        CLEAN_TRACKB_SOURCE,
        _patch("    asc_copy_gm2ub_align(ub, gm, 128);", "    asc_copy_gm2ub_align(ub, gm, 256);"),
        active_kernel="kernel_active",
    )
    assert applied == 1 and "asc_copy_gm2ub_align(ub, gm, 256);" in updated

    assert len(ASCENDC_RENDERERS) == 7
    for axes, renderer in ASCENDC_RENDERERS.items():
        cpp = renderer()
        assert all(cnt == 0 for cnt in CCE_CPP20_SURFACE.count_forbidden_tokens(cpp).values()), (
            axes.key()
        )
        for legacy in ("PipeBarrier", "LocalTensor", "Mmad", "DataCopy", "CrossCoreSetFlag"):
            assert re.search(rf"\b{re.escape(legacy)}\b", cpp) is None


def test_ast_semantic_gate_and_surface_sync_auditor(tmp_path: Path) -> None:
    """Verify S-A1..S-A4 AST gate rules, example kernels, and surface sync auditor."""
    assert TileLangSemanticGate is not None
    assert SemanticViolation("X", 1, "msg").code == "X"

    bad_nested = dedent(
        """
        def build():
            @T.prim_func
            def kern(A: T.Tensor((16,), "float32")):
                def inner_helper(x):
                    return x + 1
                with T.Kernel(1) as bx:
                    pass
            return kern
        """
    )
    assert [v.code for v in check_tilelang_semantic_contracts(bad_nested)] == [
        "NESTED_DEF_IN_PRIM_FUNC"
    ]

    bad_rebind = dedent(
        """
        kernel = lambda x: x
        def ref_kernel(x):
            return x
        """
    )
    assert [
        v.code
        for v in check_tilelang_semantic_contracts(
            bad_rebind, protected_exports=frozenset({"kernel", "ref_kernel"})
        )
    ] == ["PROTECTED_SYMBOL_REBIND", "PROTECTED_SYMBOL_REBIND"]

    bad_nesting = dedent(
        """
        @T.prim_func
        def kern(A: T.Tensor((16, 16), "float32")):
            with T.Kernel(1) as bx:
                for i in T.Parallel(16):
                    T.copy(A[i, :], A[i, :])
                    for k in T.Pipelined(4, num_stages=2):
                        pass
                with T.SimdVF():
                    with T.Stage(0):
                        pass
                with T.SimtVF(threads=128), T.Stage(0):
                    pass
                for k in T.Pipelined(4, num_stages=2):
                    with T.Stage(1):
                        pass
        """
    )
    codes = [v.code for v in check_tilelang_semantic_contracts(bad_nesting)]
    assert "TILE_OP_INSIDE_PARALLEL" in codes
    assert "PIPELINED_INSIDE_PARALLEL" in codes
    assert codes.count("STAGE_INSIDE_LEAF_SCOPE") == 2
    assert "NONZERO_STAGE_WITHOUT_ENABLE_OFFSET" in codes

    bad_simd_off = dedent(
        """
        @T.prim_func
        def kern(buf: T.Tensor((64,), "float32")):
            with T.Kernel(1) as bx:
                with T.SimdVF():
                    r = T.simd.vld(buf[0], off=0)
                    T.simd.vsts(buf[0], r, off=0)
                    p = T.simd.pld(buf[0], off=0)
                    T.simd.pst(buf[0], p, off=0)
        """
    )
    v_sa4 = check_tilelang_semantic_contracts(bad_simd_off)
    assert len(v_sa4) == 4 and all(v.code == "SIMD_FORBIDDEN_OFF_KWARG" for v in v_sa4)

    kernel_files = sorted((_REPO_ROOT / "examples" / "ascend950pr_ops").glob("*/kernels.py"))
    assert len(kernel_files) >= 15
    for path in kernel_files:
        assert check_tilelang_semantic_contracts(path.read_text(encoding="utf-8")) == (), path

    scripts_dir = _REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import audit_tilelang_surface_sync as auditor  # noqa: PLC0415

    upstream = _resolve_tilelang_root()
    if upstream is not None:
        assert auditor.audit_surface_sync(upstream) == []
        assert auditor.main(["--tilelang-root", str(upstream)]) == 0

    fake_lang = tmp_path / "tilelang" / "ascend" / "language"
    fake_common = tmp_path / "tilelang" / "language"
    fake_lang.mkdir(parents=True)
    fake_common.mkdir(parents=True)
    (fake_lang / "__init__.py").write_text("_ASCEND_API_ALL = ('Kernel',)\n", encoding="utf-8")
    (fake_common / "common.py").write_text("_LOCAL_EXPORTS = ()\n", encoding="utf-8")
    (fake_lang / "simd.py").write_text("def vld(addr, dist='NORM'):\n    pass\n", encoding="utf-8")
    assert len(auditor.audit_surface_sync(tmp_path)) > 0
    assert auditor.main(["--tilelang-root", str(tmp_path)]) == 1
