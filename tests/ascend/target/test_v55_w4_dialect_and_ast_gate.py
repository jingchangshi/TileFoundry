"""Tests for TileLang Ascend dialect surface updates, AST gate, and sync auditor."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask, TopologyAxes
from tilefoundry_ascend.ascend.target.ast_semantic_gate import (
    SemanticViolation,
    TileLangSemanticGate,
    check_tilelang_semantic_contracts,
)
from tilefoundry_ascend.ascend.target.dialect_surface import (
    TILELANG_DSL_SURFACE,
    DialectSurfaceSpec,
    PrimitiveEntry,
    filter_primitives_on_demand,
    render_layer2_surface,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_tilelang_root() -> Path | None:
    env_root = os.environ.get("TILELANG_ROOT", "").strip()
    if env_root and Path(env_root).is_dir():
        return Path(env_root)
    sibling = _REPO_ROOT.parent / "tilelang"
    if sibling.is_dir():
        return sibling
    workspace_sibling = _REPO_ROOT.parents[1] / "tilelang"
    if workspace_sibling.is_dir():
        return workspace_sibling
    return None


def test_dialect_surface_absorbs_dav3510_primitives_and_notes() -> None:
    """Verify absorbed dav-3510 primitives, 4-mode version semantics, and prompt notes."""
    by_sym = {entry.symbol: entry for entry in TILELANG_DSL_SURFACE.primitives}
    expected = (
        "T.Task",
        "T.PerCoreTask",
        "T.copy",
        "T.annotate_buffer_versions",
        "T.assume_no_conflict",
        "T.assume_conflict",
        "T.alloc_reducer",
        "T.reducer_init",
        "T.reducer_update",
        "T.finalize_reducer",
        "T.SimdVF",
        "T.SimtVF",
        "T.Stage",
    )
    for sym in expected:
        assert sym in by_sym, sym

    assert "latency=" in by_sym["T.Task"].signature
    assert "ii=" in by_sym["T.Task"].signature
    assert "latency >= ii > 0" in by_sym["T.Task"].pitfall
    assert "T.PerCoreTask" in by_sym["T.PerCoreTask"].signature
    for kw in ("transpose=", "pad_value=", "data_select=", "l2_cache_ctrl=", "sub_blockid="):
        assert kw in by_sym["T.copy"].signature
    assert "{buf: 1 | count | (count, mode) | mode}" in by_sym["T.annotate_buffer_versions"].pitfall
    for kw in ("cross=", "level=", "group="):
        assert kw in by_sym["T.assume_no_conflict"].signature
        assert kw in by_sym["T.assume_conflict"].signature
    assert "latency=" in by_sym["T.SimdVF"].signature
    assert "threads=128, latency=" in by_sym["T.SimtVF"].signature
    assert "with T.Stage(stage_id):" == by_sym["T.Stage"].signature

    rendered = TILELANG_DSL_SURFACE.render_prompt_reference()
    assert "Prefer compiler-managed multibuffering" in rendered
    assert "never hand-roll manual multi-buffers" in rendered
    assert "Never nest `def` inside `@T.prim_func`" in rendered
    assert "do not take `off=`" in rendered


def test_render_layer2_surface_removes_700_cap_and_prioritizes_safety_notes() -> None:
    """Allocate full remaining budget and prioritize case-insensitive safety notes first."""
    prims = filter_primitives_on_demand(
        TILELANG_DSL_SURFACE,
        topology_axes=TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd"),
    )
    rendered = render_layer2_surface(
        TILELANG_DSL_SURFACE,
        primitives=prims,
        max_chars=10000,
    )
    assert len(rendered) <= 10000
    notes_section = "\n".join(
        line for line in rendered.splitlines() if line.startswith("- ") and "[" not in line
    )
    assert len(notes_section) > 700
    assert "Prefer compiler-managed multibuffering" in rendered
    assert "Never nest `def` inside `@T.prim_func`" in rendered
    assert "do not take `off=`" in rendered

    synthetic = DialectSurfaceSpec(
        dialect_id="tilelang_dsl",
        display_label="Synthetic",
        required_headers_or_imports=("import tilelang",),
        primitives=(PrimitiveEntry("T.Kernel", "SCALAR", "grid_and_workspace_topology", "T.Kernel(n)"),),
        forbidden_tokens=(),
        prompt_notes=(
            "Optional tuning suggestion one with plenty of explanatory detail.",
            "You must keep DMA buffers aligned to 32 bytes.",
            "Another optional hint about tile sizes.",
            "never allocate dynamic local arrays inside SIMD bodies.",
            "Forbid cross-lane scalar store aliasing.",
            "Do not use threads= on T.Kernel.",
        ),
    )
    tight = render_layer2_surface(synthetic, include_guardrails=False, max_chars=315)
    assert "must keep DMA buffers" in tight
    assert "never allocate dynamic" in tight
    assert "Forbid cross-lane" in tight
    assert "Do not use threads=" in tight
    assert "Optional tuning suggestion" not in tight


def test_core_symbols_retain_pipeline_primitives_when_unlocked() -> None:
    """Keep T.Stage, T.Task, and T.annotate_buffer_versions when pipeline_and_sync_graph unlocks."""
    mask = ConstraintRelaxationMask(pipeline_and_sync_graph=True)
    filtered = filter_primitives_on_demand(
        TILELANG_DSL_SURFACE,
        topology_axes=TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd"),
        freedom_mask=mask,
    )
    syms = {entry.symbol for entry in filtered}
    assert {"T.Stage", "T.Task", "T.annotate_buffer_versions"} <= syms


def test_ast_semantic_gate_rules_sa1_through_sa4() -> None:
    """Verify deterministic S-A1..S-A4 violations and near-neighbor valid patterns."""
    assert TileLangSemanticGate is not None
    assert SemanticViolation("X", 1, "msg").code == "X"

    bad_nested_def = dedent(
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
    v_sa1 = check_tilelang_semantic_contracts(bad_nested_def)
    assert [v.code for v in v_sa1] == ["NESTED_DEF_IN_PRIM_FUNC"]

    good_outer_helper = dedent(
        """
        def build():
            def outer_helper(x):
                return x + 1
            @T.prim_func
            def kern(A: T.Tensor((16,), "float32")):
                with T.Kernel(1) as bx:
                    pass
            return kern
        """
    )
    assert check_tilelang_semantic_contracts(good_outer_helper) == ()

    bad_rebind = dedent(
        """
        def _build():
            kernel = 1
            return kernel

        kernel = lambda x: x
        def ref_kernel(x):
            return x
        """
    )
    v_sa2 = check_tilelang_semantic_contracts(
        bad_rebind,
        protected_exports=frozenset({"kernel", "ref_kernel", "build_kernel"}),
    )
    assert [v.code for v in v_sa2] == ["PROTECTED_SYMBOL_REBIND", "PROTECTED_SYMBOL_REBIND"]
    assert check_tilelang_semantic_contracts(bad_rebind) == ()

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
    v_sa3 = check_tilelang_semantic_contracts(bad_nesting)
    codes = [v.code for v in v_sa3]
    assert "TILE_OP_INSIDE_PARALLEL" in codes
    assert "PIPELINED_INSIDE_PARALLEL" in codes
    assert codes.count("STAGE_INSIDE_LEAF_SCOPE") == 2
    assert "NONZERO_STAGE_WITHOUT_ENABLE_OFFSET" in codes

    good_nesting = dedent(
        """
        @T.prim_func
        def kern(A: T.Tensor((16, 16), "float32")):
            with T.Kernel(1) as bx:
                for k in T.Pipelined(4, num_stages=2, annotations={"enable_offset": True}):
                    with T.Stage(0):
                        T.copy(A[k, :], A[k, :])
                    with T.Stage(1), T.SimtVF(threads=128):
                        for i in T.Parallel(16):
                            T.copy(A[k, i], A[k, i])
        """
    )
    assert check_tilelang_semantic_contracts(good_nesting) == ()

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
                    r2 = T.simd.vld2(buf[0], off=0)
        """
    )
    v_sa4 = check_tilelang_semantic_contracts(bad_simd_off)
    assert len(v_sa4) == 4
    assert all(v.code == "SIMD_FORBIDDEN_OFF_KWARG" for v in v_sa4)


def test_ast_semantic_gate_zero_violations_on_all_example_kernels() -> None:
    """Verify zero AST gate violations across all ascend950pr example kernels."""
    kernel_files = sorted((_REPO_ROOT / "examples" / "ascend950pr_ops").glob("*/kernels.py"))
    assert len(kernel_files) >= 15
    for path in kernel_files:
        violations = check_tilelang_semantic_contracts(path.read_text(encoding="utf-8"))
        assert violations == (), f"{path}: {violations}"


def test_audit_tilelang_surface_sync_script_and_drift_detection(tmp_path: Path) -> None:
    """Verify offline surface sync auditor passes on upstream checkout and flags stale items."""
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
    errors = auditor.audit_surface_sync(tmp_path)
    assert len(errors) > 0
    assert auditor.main(["--tilelang-root", str(tmp_path)]) == 1
