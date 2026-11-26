"""Unit tests for regime slice patching, Track-B compaction, host tail split, and round context."""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.campaign.anchor_specs import (
    AnchorCaseSpec,
    PerfTargetSpec,
    RegimeTargetSpec,
)
from tilefoundry_ascend.agent.campaign.regime_slice_patcher import (
    DEFAULT_READONLY_HOST_EXPORTS,
    _compact_trackb_snippet_source,
    extract_regime_snippet,
    split_readonly_host_tail,
)
from tilefoundry_ascend.agent.campaign.round_context_builder import (
    ChampionState,
    RoundContextBuilder,
    RoundSynthesisBundle,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    aggregate_surrogate_result,
)
from tilefoundry_ascend.agent.cost_advisor import create_cost_advisor
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.kb.schema import (
    AscendCausalRule,
    SymptomFamily,
)
from tilefoundry_ascend.agent.orchestrator import llm_constraints
from tilefoundry_ascend.agent.orchestrator.gate_funnel import StageAResult
from tilefoundry_ascend.agent.router.isa_gap_scanner import scan_tilelang_pipe_gaps
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import (
    HirBlueprintPacket,
    HirRooflineSummary,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger
from tilefoundry_ascend.ascend.schedule.regime_solver import RegimePiecewiseSolver
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes
from tilefoundry_ascend.ascend.target import AscendTarget

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class _DummyConfig:
    operator_name: str = "softmax"
    pure_golden: bool = False
    stagnation_escalate_rounds: int = 2
    k_rule_workers: int = 1
    enable_tilesize_prescreen: bool = False
    exploit_effort: str = "medium"
    allow_unreviewed_distilled_rules: bool = False


def test_trackb_extract_regime_snippet_strips_dead_tracka_builders() -> None:
    """Strip dead Track-A _build_*_func symbols from Track-B slices."""
    module_source = (
        "import tilelang.ascend.language as T\n\n"
        '_SOFTMAX_TRACKB_SRC = r"""\n'
        "#include <tl_templates/ascend/common.h>\n"
        'extern "C" __global__ __aicore__ void softmax_tb(__gm__ float* x) {}\n'
        '"""\n\n'
        "def _select_softmax_config(d_dim: int = 256) -> dict[str, object]:\n"
        '    return {"d_dim": d_dim}\n\n'
        "def _select_other_regime_config(d_dim: int = 16) -> dict[str, object]:\n"
        '    return {"d_dim": d_dim}\n\n'
        "def _render_trackb_tu(**dims: object) -> str:\n"
        '    return str(dims.get("trackb_src", ""))\n\n'
        "def _build_softmax_func(d_dim: int = 256) -> object:\n"
        "    return d_dim\n\n"
        "def _build_softmax_prim_func(d_dim: int = 256) -> object:\n"
        "    return d_dim\n"
    )
    for backend in ("ascendc_trackb", "hybrid_trackb"):
        spec = RegimeTargetSpec(
            regime_tag="large_row",
            builder_candidates=("_build_softmax_func", "_build_softmax_prim_func"),
            sample_builder_kwargs={"d_dim": 256},
            config_selector_fn="_select_softmax_config",
            editable_symbols=(
                "_select_softmax_config",
                "_build_softmax_func",
                "_build_softmax_prim_func",
                "_render_trackb_tu",
                "_SOFTMAX_TRACKB_SRC",
            ),
            runtime_backend=backend,
            trackb_symbols=("_SOFTMAX_TRACKB_SRC",),
        )
        primary, snippet = extract_regime_snippet(module_source, spec)
        assert primary == "_select_softmax_config"
        assert "_SOFTMAX_TRACKB_SRC" in snippet
        assert "def _select_softmax_config(" in snippet
        assert "def _render_trackb_tu(" in snippet
        assert "_build_softmax_func" not in snippet
        assert "_build_softmax_prim_func" not in snippet
        assert "_select_other_regime_config" not in snippet


def test_compact_trackb_snippet_has_no_byte_caps_and_zero_truncation() -> None:
    """Render >64KB active Track-B closure verbatim with single-line sibling decls."""
    sig = inspect.signature(_compact_trackb_snippet_source)
    assert sig.parameters["max_bytes"].default is None
    assert sig.parameters["cpp_max_bytes"].default is None

    active_body = "".join(
        f"    acc += x[{idx}] * helper_fn((float){idx});\n" for idx in range(1300)
    )
    assert len(active_body.encode("utf-8")) > 55000

    cpp_tu = (
        "#include <tl_templates/ascend/common.h>\n"
        "__aicore__ inline float helper_fn(float v) {\n"
        "    return v + 1.0f;\n"
        "}\n"
        "__aicore__ inline float sibling_only_fn(float v) {\n"
        "    return v * 2.0f;\n"
        "}\n"
        '__global__ __aicore__ void active_large_kernel(__gm__ float* x, __gm__ float* y) {\n'
        "    float acc = 0.0f;\n"
        f"{active_body}"
        "    y[0] = acc;\n"
        "}\n"
        '__global__ __aicore__ void sibling_other_kernel(__gm__ float* a) {\n'
        "    a[0] = sibling_only_fn(a[0]);\n"
        "}\n"
    )
    snippet = f'_LARGE_TRACKB_SRC = r"""\n{cpp_tu}\n"""\n'
    spec = RegimeTargetSpec(
        regime_tag="main",
        builder_candidates=("_build_main",),
        sample_builder_kwargs={},
        runtime_backend="ascendc_trackb",
        trackb_symbols=("_LARGE_TRACKB_SRC",),
    )
    compacted = _compact_trackb_snippet_source(
        snippet,
        spec,
        incumbent_source=snippet,
        active_kernel="active_large_kernel",
    )
    assert len(compacted.encode("utf-8")) > 60000
    assert "MIDDLE BODY OMITTED" not in compacted
    assert "LINES OMITTED" not in compacted
    assert "acc += x[1299] * helper_fn((float)1299);" in compacted
    assert "helper_fn" in compacted
    assert "// [READONLY_SIBLING_DECLARATION]" in compacted
    assert "sibling_other_kernel" in compacted
    assert "sibling_only_fn(a[0])" not in compacted


def test_split_readonly_host_tail_and_tracka_prompt_rendering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Separate protected tail bindings into <read_only_host_tail> outside the code fence."""
    monkeypatch.setattr(
        llm_constraints,
        "scan_tilelang_pipe_gaps",
        lambda: scan_tilelang_pipe_gaps(allow_offline_catalog=True),
    )
    full_slice = (
        "def _select_op_config(d_dim: int = 128) -> dict[str, object]:\n"
        '    return {"d_dim": d_dim}\n\n'
        "def _build_op_func(d_dim: int = 128) -> object:\n"
        "    return d_dim\n\n"
        "_KERNEL_CACHE: dict[object, object] = {}\n\n"
        "_ENTRY_CACHE: dict[object, object] = {}\n\n"
        "def _get_kernel(d_dim: int) -> object:\n"
        "    return _build_op_func(d_dim)\n\n"
        "def build_kernel(d_dim: int) -> object:\n"
        "    return _get_kernel(d_dim)\n\n"
        "def ref_kernel(x: object) -> object:\n"
        "    return x\n\n"
        "def kernel(x: object) -> object:\n"
        "    return _get_kernel(128)\n"
    )
    editable, tail = split_readonly_host_tail(full_slice, DEFAULT_READONLY_HOST_EXPORTS)
    assert "def _select_op_config(" in editable
    assert "def _build_op_func(" in editable
    for sym in DEFAULT_READONLY_HOST_EXPORTS:
        assert sym not in editable
        assert sym in tail

    builder = _make_builder()
    spec = RegimeTargetSpec(
        regime_tag="main",
        builder_candidates=("_build_op_func",),
        sample_builder_kwargs={"d_dim": 128},
        config_selector_fn="_select_op_config",
        editable_symbols=("_select_op_config", "_build_op_func"),
        runtime_backend="tilelang_dsl",
    )
    anchor = AnchorEvalMetrics(
        case_num=0,
        regime_tag="main",
        accuracy_passed=True,
        cand_us=100.0,
        ref_perf_us=150.0,
        t_hw_us=25.0,
        speedup=1.5,
        roofline_eta=0.25,
        eq3_score=0.5,
    )
    best_res = aggregate_surrogate_result("inc", "softmax", [anchor])
    rule = _dummy_rule()
    exploit_req = builder._build_exploit_regime_request(
        snippet_source=full_slice,
        func_name="_build_op_func",
        regime_spec=spec,
        rule=rule,
        best_result=best_res,
        incumbent_source=full_slice,
    )
    creative_req = builder._build_creative_regime_request(
        snippet_source=full_slice,
        func_name="_build_op_func",
        regime_spec=spec,
        best_result=best_res,
        effort="high",
        incumbent_source=full_slice,
    )
    for req in (exploit_req, creative_req):
        assert req.json_schema is None
        assert "<read_only_host_tail>" in req.user_prompt
        assert "</read_only_host_tail>" in req.user_prompt
        assert "Never define a nested `def` inside `@T.prim_func`" in req.user_prompt
        fence_start = req.user_prompt.index("### TARGET FUNCTION SNIPPET\n```python\n")
        fence_end = req.user_prompt.index("\n```", fence_start + 30)
        fenced_code = req.user_prompt[fence_start:fence_end]
        tail_start = req.user_prompt.index("<read_only_host_tail>\n")
        assert tail_start > fence_end
        assert "</read_only_host_tail>" not in fenced_code
        assert "_KERNEL_CACHE" not in fenced_code
        assert "def _get_kernel(" not in fenced_code
        assert "def kernel(" not in fenced_code
        assert "def ref_kernel(" not in fenced_code
        assert "def build_kernel(" not in fenced_code
        assert "# target_bottleneck:" in req.system_prompt
        assert "# structural_hypothesis:" in req.system_prompt
        assert "raw JSON" not in req.system_prompt
        assert '"physical_hypothesis"' not in req.system_prompt
        assert '"dsl_source"' not in req.system_prompt


def test_same_case_eta_and_build_round_bundle_uses_msprof_by_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compute eta_case from msprof_by_case[(parent_sha, bn.case_num)] without static roofline."""
    monkeypatch.setattr(
        llm_constraints,
        "scan_tilelang_pipe_gaps",
        lambda: scan_tilelang_pipe_gaps(allow_offline_catalog=True),
    )
    module_src = (
        "def _select_softmax_config(d_dim: int = 256) -> dict[str, object]:\n"
        '    return {"d_dim": d_dim}\n\n'
        "def _build_softmax_func(d_dim: int = 256) -> object:\n"
        "    return d_dim\n\n"
        "def _get_kernel(d_dim: int) -> object:\n"
        "    return _build_softmax_func(d_dim)\n\n"
        "def softmax(x: object) -> object:\n"
        "    return _get_kernel(256)\n"
    )
    spec = RegimeTargetSpec(
        regime_tag="main",
        builder_candidates=("_build_softmax_func",),
        sample_builder_kwargs={"d_dim": 256},
        config_selector_fn="_select_softmax_config",
        editable_symbols=("_select_softmax_config", "_build_softmax_func"),
        representative_anchor_dims={"d_dim": 256},
        runtime_backend="tilelang_dsl",
    )
    parent_sha = hashlib.sha256(module_src.encode("utf-8")).hexdigest()
    bn_anchor = AnchorEvalMetrics(
        case_num=7,
        regime_tag="main",
        accuracy_passed=True,
        cand_us=250.0,
        ref_perf_us=300.0,
        t_hw_us=50.0,
        speedup=1.2,
        roofline_eta=0.20,
        eq3_score=0.4,
    )
    other_anchor = AnchorEvalMetrics(
        case_num=8,
        regime_tag="main",
        accuracy_passed=True,
        cand_us=10.0,
        ref_perf_us=20.0,
        t_hw_us=9.5,
        speedup=2.0,
        roofline_eta=0.95,
        eq3_score=0.9,
    )
    best_res = aggregate_surrogate_result("inc", "softmax", [bn_anchor, other_anchor])
    static_blueprint = HirBlueprintPacket(
        entry_name="softmax",
        roofline=HirRooflineSummary(
            anchor_dims={"d_dim": 256},
            compute_flops=1_000_000,
            min_gm_bytes=9_999_999,
            arithmetic_intensity=1.0,
            compute_floor_us=1000.0,
            memory_floor_us=500.0,
            ideal_roofline_us=1000.0,
            dominant_bound="compute_bound",
        ),
    )
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=1,
        pipe_cycles={"AIV0.PIPE_V": 1000, "AIV0.PIPE_MTE2": 800},
        waterlines={},
        gm_roundtrip_bytes=4096,
    )
    builder = _make_builder(
        regime_specs=(spec,),
        regime_blueprints={"main": static_blueprint},
        run_l0_fn=lambda _src, _reg, **_kw: StageAResult(
            passed=True,
            prim_func=None,
            ledger=ledger,
        ),
    )
    state = ChampionState(
        incumbent_source=module_src,
        best_result=best_res,
        active_anchors=(
            AnchorCaseSpec("op/softmax", 7, "main", 300.0, 50.0),
            AnchorCaseSpec("op/softmax", 8, "main", 20.0, 9.5),
        ),
        regime_specs=(spec,),
        msprof_by_case={
            (parent_sha, 7): {"dur_us": 200.0, "aiv_vec_ratio": 0.45},
        },
        regime_blueprints={"main": static_blueprint},
    )
    bundle = builder.build_round_bundle(state, round_idx=1)
    assert isinstance(bundle, RoundSynthesisBundle)
    assert bundle.parent_sha == parent_sha
    assert bundle.parent_residual.measured_latency_us == 200.0
    assert bundle.parent_residual.observation is not None
    assert abs(bundle.parent_residual.observation.compute_issue.l2_roofline_eta - 0.25) < 1e-6
    assert abs(bundle.parent_residual.effective_compute_eta - 0.25) < 1e-6
    assert bundle.parent_residual.transfer_inflation_ratio == 0.0
    assert len(bundle.worker_specs) >= 2
    for _wid, _wkind, _rule, req in bundle.worker_specs:
        assert req.json_schema is None
        assert "# target_bottleneck:" in req.system_prompt
        assert "# structural_hypothesis:" in req.system_prompt


def _dummy_rule() -> AscendCausalRule:
    return AscendCausalRule(
        rule_id="R_TEST_VEC",
        regime_guard="True",
        precondition="True",
        transform_template="Vectorize loop tail.",
        expected_postcondition="True",
        source_provenance="unit_test",
        symptom_family=SymptomFamily.COMPUTE_VECTOR_LOOP_DEGRADED.value,
    )


def test_eta_case_clamps_to_one_and_trackb_unified_header_protocol() -> None:
    """Clamp eta_case at 1.0 when t_hw_us > dur_us and enforce header+fence on Track-B."""
    module_src = (
        '_TB_SRC = r"""\n'
        "#include <tl_templates/ascend/common.h>\n"
        "__global__ __aicore__ void tb_main(__gm__ float* x) {}\n"
        '"""\n\n'
        "def _select_tb_config(d_dim: int = 128) -> dict[str, object]:\n"
        '    return {"d_dim": d_dim}\n\n'
        "def _build_tb_func(d_dim: int = 128) -> object:\n"
        "    return d_dim\n"
    )
    spec = RegimeTargetSpec(
        regime_tag="main",
        builder_candidates=("_build_tb_func",),
        sample_builder_kwargs={"d_dim": 128},
        config_selector_fn="_select_tb_config",
        editable_symbols=("_select_tb_config", "_build_tb_func", "_TB_SRC"),
        runtime_backend="ascendc_trackb",
        trackb_symbols=("_TB_SRC",),
    )
    parent_sha = hashlib.sha256(module_src.encode("utf-8")).hexdigest()
    bn_anchor = AnchorEvalMetrics(
        case_num=3,
        regime_tag="main",
        accuracy_passed=True,
        cand_us=12.0,
        ref_perf_us=30.0,
        t_hw_us=15.0,
        speedup=2.5,
        roofline_eta=1.0,
        eq3_score=0.9,
    )
    best_res = aggregate_surrogate_result("inc", "softmax", [bn_anchor])
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=1,
        pipe_cycles={"AIV0.PIPE_V": 200},
        waterlines={},
        gm_roundtrip_bytes=1024,
    )
    builder = _make_builder(
        regime_specs=(spec,),
        run_l0_fn=lambda _src, _reg, **_kw: StageAResult(
            passed=True,
            prim_func=None,
            ledger=ledger,
        ),
    )
    state = ChampionState(
        incumbent_source=module_src,
        best_result=best_res,
        active_anchors=(AnchorCaseSpec("op/softmax", 3, "main", 30.0, 15.0),),
        regime_specs=(spec,),
        msprof_by_case={(parent_sha, 3): {"dur_us": 10.0, "aiv_vec_ratio": 0.9}},
    )
    bundle = builder.build_round_bundle(state, round_idx=2)
    assert bundle.parent_residual.observation is not None
    assert bundle.parent_residual.observation.compute_issue.l2_roofline_eta == 1.0
    assert bundle.parent_residual.effective_compute_eta == 1.0
    assert "_build_tb_func" not in bundle.parent_snippet
    for _wid, _wkind, _rule, req in bundle.worker_specs:
        assert req.json_schema is None
        assert "# target_bottleneck:" in req.system_prompt
        assert "# structural_hypothesis:" in req.system_prompt
        assert "raw JSON" not in req.system_prompt
        assert "<read_only_host_tail>" not in req.user_prompt


def _make_builder(
    *,
    regime_specs: tuple[RegimeTargetSpec, ...] = (),
    regime_blueprints: dict[str, HirBlueprintPacket] | None = None,
    run_l0_fn: object | None = None,
) -> RoundContextBuilder:
    target = AscendTarget("ascend.950pr")
    specs = regime_specs or (
        RegimeTargetSpec(
            regime_tag="main",
            builder_candidates=("_build_op_func",),
            sample_builder_kwargs={"d_dim": 128},
            config_selector_fn="_select_op_config",
            editable_symbols=("_select_op_config", "_build_op_func"),
            runtime_backend="tilelang_dsl",
        ),
    )
    tracker = BayesianRuleTracker.from_rules([_dummy_rule()])
    cost_advisor = create_cost_advisor("off", target=target, operator_name="softmax")
    solver = RegimePiecewiseSolver(target)
    return RoundContextBuilder(
        config=_DummyConfig(),
        regime_specs=specs,
        target=target,
        tracker=tracker,
        cost_advisor=cost_advisor,
        regime_solver=solver,
        source_root=REPO_ROOT,
        perf_target=PerfTargetSpec("softmax", 1.5, 1.1, 0.85),
        regime_blueprints=regime_blueprints,
        run_l0_fn=run_l0_fn if callable(run_l0_fn) else None,
        topology_axes=TopologyAxes(
            compute_engine="aiv_only",
            memory_ring="gm_ub_stream",
            iteration_regularity="affine_simd",
        ),
    )
