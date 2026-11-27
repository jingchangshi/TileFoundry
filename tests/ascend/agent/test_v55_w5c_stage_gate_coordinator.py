"""Unit tests for StageGateCoordinator, speculative creative lane, and typed L2 faults."""

from __future__ import annotations

import ast
import asyncio
import importlib
import json
import subprocess
import sys
import time
import types
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest


def _bootstrap_isolated_worktree() -> None:
    """Resolve worktree modules directly even if another checkout is installed."""
    root = Path(__file__).resolve().parents[3]
    ascend_root = str(root / "targets" / "ascend")
    if ascend_root not in sys.path:
        sys.path.insert(0, ascend_root)
    pkg_dir = root / "targets" / "ascend" / "tilefoundry_ascend"
    try:
        mod = importlib.import_module("tilefoundry_ascend.agent.campaign.stage_gate_coordinator")
        mod_file = getattr(mod, "__file__", "") or ""
        if Path(mod_file).resolve().is_relative_to(root):
            return
    except ImportError:
        pass
    stale = [
        k
        for k in sys.modules
        if k == "tilefoundry_ascend" or k.startswith("tilefoundry_ascend.")
    ]
    for key in stale:
        del sys.modules[key]
    for mod_name, sub_path in (
        ("tilefoundry_ascend", pkg_dir),
        ("tilefoundry_ascend.agent", pkg_dir / "agent"),
        ("tilefoundry_ascend.agent.campaign", pkg_dir / "agent" / "campaign"),
    ):
        mod = types.ModuleType(mod_name)
        mod.__path__ = [str(sub_path)]
        sys.modules[mod_name] = mod


_bootstrap_isolated_worktree()

from tilefoundry_ascend.agent.campaign import (  # noqa: E402
    stage_gate_coordinator,
    surrogate_cannbench,
)
from tilefoundry_ascend.agent.specs.anchor_specs import (  # noqa: E402
    AnchorCaseSpec,
    GuardrailCaseSpec,
    RegimeTargetSpec,
)
from tilefoundry_ascend.agent.specs.fault_types import FaultPlane  # noqa: E402
from tilefoundry_ascend.agent.campaign.stage_gate_coordinator import (  # noqa: E402
    CandidateBatch,
    OccupancyExhaustedError,
    SpeculativeCreativeResult,
    StageGateCoordinator,
    StageGateOutcome,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (  # noqa: E402
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    L0L1CandidateCheck,
    L2BatchResults,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (  # noqa: E402
    StageAResult,
    StageBResult,
    _normalize_trackb_unit_source,
    resolve_trackb_compile_flags,
)
from tilefoundry_ascend.agent.orchestrator.stage_b_gate import (  # noqa: E402
    CompileTaskSpec,
    has_unreserved_vf_stack_overflow,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import GenerationRequest  # noqa: E402
from tilefoundry_ascend.ascend.compile_cache.artifact_store import (  # noqa: E402
    ArtifactStore,
    compute_compile_task_key,
)
from tilefoundry_ascend.ascend.cost_model import BishengResourceSummary  # noqa: E402
from tilefoundry_ascend.ascend.passes import (  # noqa: E402
    build_objective_physical_ledger,
    build_scope_waterline,
)

_PARENT_MODULE_SRC = (
    "def _build_main(tile_m=16):\n"
    "    return int(tile_m)\n\n"
    "def kernel(x):\n"
    "    return _build_main(), x\n"
)

_PARENT_SNIPPET = "def _build_main(tile_m=16):\n    return int(tile_m)\n"
_EXPLOIT_SNIPPET = "def _build_main(tile_m=32):\n    return int(tile_m)\n"
_CREATIVE_SNIPPET = "def _build_main(tile_m=64):\n    return int(tile_m)\n"


def _make_regime_spec() -> RegimeTargetSpec:
    return RegimeTargetSpec(
        regime_tag="decode",
        builder_candidates=("_build_main",),
        editable_symbols=("_build_main",),
        sample_builder_kwargs={"tile_m": 16},
        representative_anchor_dims={"D": 128},
    )


def _make_anchors() -> tuple[AnchorCaseSpec, ...]:
    return (AnchorCaseSpec("level2/rms_norm", 0, "decode", 20.0, 2.0),)


def _make_stage_a() -> StageAResult:
    waterlines = {
        "UB": build_scope_waterline(
            scope="UB",
            capacity_bytes=262144,
            explicit_buffers={"a": 32768},
            implicit_buffers={},
        )
    }
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=8,
        pipe_cycles={"AIV0.PIPE_V": 100, "AIV1.PIPE_V": 100},
        waterlines=waterlines,
        gm_roundtrip_bytes=16384,
        grid_blocks=16,
    )
    return StageAResult(passed=True, prim_func=None, ledger=ledger)


def _make_surrogate_result(cand_id: str, cand_us: float = 10.0) -> CandidateSurrogateResult:
    anc = AnchorEvalMetrics(
        case_num=0,
        regime_tag="decode",
        accuracy_passed=True,
        cand_us=cand_us,
        ref_perf_us=20.0,
        t_hw_us=2.0,
        speedup=20.0 / cand_us,
        roofline_eta=2.0 / cand_us,
        eq3_score=20.0 / cand_us,
        paired_rel_mad=0.01,
    )
    return CandidateSurrogateResult(
        candidate_id=cand_id,
        operator_name="rms_norm",
        all_passed=True,
        avg_speedup=anc.speedup,
        min_case_speedup=anc.speedup,
        avg_roofline_eta=anc.roofline_eta,
        surrogate_score=anc.speedup,
        anchors=(anc,),
    )


def _make_req(role: str) -> GenerationRequest:
    return GenerationRequest(
        system_prompt="sys",
        user_prompt="usr",
        role=role,
        effort="medium",
    )


def _make_coordinator(
    tmp_path: Path,
    *,
    backend: object | None = None,
    l2_batch_evaluator: object | None = None,
    l1_compiler: object | None = None,
    static_width_dims_fn: object | None = None,
    regime_specs: Sequence[RegimeTargetSpec] | None = None,
    guardrail_cases: Sequence[GuardrailCaseSpec] = (),
) -> StageGateCoordinator:
    target_path = tmp_path / "kernel.py"
    target_path.write_text(_PARENT_MODULE_SRC, encoding="utf-8")
    specs = tuple(regime_specs) if regime_specs is not None else (_make_regime_spec(),)
    return StageGateCoordinator(
        operator_name="rms_norm",
        target_module_path=target_path,
        regime_specs=specs,
        initial_anchors=_make_anchors(),
        scoreboard_anchors=_make_anchors(),
        guardrail_cases=guardrail_cases,
        backend=backend,
        l0_checker=lambda _src, _reg: _make_stage_a(),
        l1_compiler=l1_compiler or (lambda _src, _reg: StageBResult(passed=True)),
        l2_batch_evaluator=l2_batch_evaluator
        or (
            lambda cands, _ancs: {
                cid: _make_surrogate_result(cid) for cid, _path in cands
            }
        ),
        static_width_dims_fn=static_width_dims_fn,
        max_syntax_repairs=0,
    )


@pytest.mark.anyio
async def test_slow_creative_nonblocking_and_harvested_next_round(tmp_path: Path) -> None:
    """Slow creative worker must not block round 0 and harvests cleanly into round 1."""
    creative_gate = asyncio.Event()

    class _Backend:
        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            _ = effort_override
            if req.role == "explore_creative":
                await creative_gate.wait()
                return (
                    "# target_bottleneck: aiv_vec\n"
                    "# structural_hypothesis: Widen vector tile to 64\n"
                    f"```python\n{_CREATIVE_SNIPPET}\n```"
                )
            return _EXPLOIT_SNIPPET

    coord = _make_coordinator(tmp_path, backend=_Backend())
    spec = _make_regime_spec()
    anchors = _make_anchors()

    batch_r0 = CandidateBatch(
        round_idx=0,
        regime_spec=spec,
        worker_specs=[
            ("r0_rule_0", "rule_exploit", None, _make_req("exploit_rule")),
            ("r0_creative", "creative_explore", None, _make_req("explore_creative")),
        ],
        parent_module_source=_PARENT_MODULE_SRC,
        parent_snippet=_PARENT_SNIPPET,
        func_name="_build_main",
    )
    out_r0 = await coord.execute_round_gates(batch_r0, anchors)
    assert isinstance(out_r0, StageGateOutcome)
    assert [cid for cid, _ in out_r0.raw_candidates] == ["r0_rule_0"]
    assert "r0_rule_0" in out_r0.l2_results
    assert len(coord._pending_speculative) == 1
    assert coord._pending_speculative[0][0] == 0
    assert coord._pending_speculative[0][1] == "r0_creative"

    creative_gate.set()
    await asyncio.sleep(0)

    batch_r1 = CandidateBatch(
        round_idx=1,
        regime_spec=spec,
        worker_specs=[
            ("r1_rule_0", "rule_exploit", None, _make_req("exploit_rule")),
        ],
        parent_module_source=_PARENT_MODULE_SRC,
        parent_snippet=_PARENT_SNIPPET,
        func_name="_build_main",
    )
    out_r1 = await coord.execute_round_gates(batch_r1, anchors)
    assert coord._pending_speculative == []
    r1_cids = [cid for cid, _ in out_r1.raw_candidates]
    assert "r0_creative" in r1_cids
    assert "r1_rule_0" in r1_cids
    assert out_r1.meta_by_id["r0_creative"][0] == "creative_explore"
    assert "Widen vector tile to 64" in out_r1.meta_by_id["r0_creative"][2]
    assert "r0_creative" in out_r1.l2_results


@pytest.mark.anyio
async def test_fast_creative_finishes_in_same_round(tmp_path: Path) -> None:
    """Fast creative worker finishing before exploit workers is harvested in same round."""

    class _FastCreativeBackend:
        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            _ = effort_override
            if req.role == "explore_creative":
                await asyncio.sleep(0.005)
                return (
                    "# target_bottleneck: ub_bytes\n"
                    "# structural_hypothesis: Fast creative tile\n"
                    f"```python\n{_CREATIVE_SNIPPET}\n```"
                )
            await asyncio.sleep(0.05)
            return _EXPLOIT_SNIPPET

    coord = _make_coordinator(tmp_path, backend=_FastCreativeBackend())
    batch = CandidateBatch(
        round_idx=0,
        regime_spec=_make_regime_spec(),
        worker_specs=[
            ("r0_rule_0", "rule_exploit", None, _make_req("exploit_rule")),
            ("r0_creative", "creative_explore", None, _make_req("explore_creative")),
        ],
        parent_module_source=_PARENT_MODULE_SRC,
        parent_snippet=_PARENT_SNIPPET,
        func_name="_build_main",
    )
    out = await coord.execute_round_gates(batch, _make_anchors())
    assert coord._pending_speculative == []
    cids = {cid for cid, _ in out.raw_candidates}
    assert cids == {"r0_rule_0", "r0_creative"}
    assert out.meta_by_id["r0_creative"][0] == "creative_explore"
    assert "r0_creative" in out.l2_results


@pytest.mark.anyio
async def test_creative_only_round_awaits_synchronously(tmp_path: Path) -> None:
    """When zero exploit workers exist, creative worker is awaited in the same round."""

    class _CreativeOnlyBackend:
        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            _ = (req, effort_override)
            await asyncio.sleep(0.01)
            return (
                "# target_bottleneck: aiv_vec\n"
                "# structural_hypothesis: Creative only round\n"
                f"```python\n{_CREATIVE_SNIPPET}\n```"
            )

    coord = _make_coordinator(tmp_path, backend=_CreativeOnlyBackend())
    batch = CandidateBatch(
        round_idx=0,
        regime_spec=_make_regime_spec(),
        worker_specs=[
            ("r0_creative", "creative_explore", None, _make_req("explore_creative")),
        ],
        parent_module_source=_PARENT_MODULE_SRC,
        parent_snippet=_PARENT_SNIPPET,
        func_name="_build_main",
    )
    out = await coord.execute_round_gates(batch, _make_anchors())
    assert coord._pending_speculative == []
    assert [cid for cid, _ in out.raw_candidates] == ["r0_creative"]


@pytest.mark.anyio
async def test_occupancy_exceptions_produce_typed_infra_occupancy_fault(
    tmp_path: Path,
) -> None:
    """OccupancyExhaustedError and fault_plane=INFRA_OCCUPANCY yield TypedFault."""

    class _SingleBackend:
        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            _ = (req, effort_override)
            return _EXPLOIT_SNIPPET

    class _CustomOccupancyError(Exception):
        fault_plane = "INFRA_OCCUPANCY"

    for exc_to_raise in (
        OccupancyExhaustedError(3, ("all NPU cards busy",)),
        _CustomOccupancyError("card pool saturated"),
    ):

        def _raise_occ(
            _cands: Sequence[tuple[str, Path]],
            _ancs: Sequence[AnchorCaseSpec],
            *,
            _exc: BaseException = exc_to_raise,
        ) -> Mapping[str, CandidateSurrogateResult]:
            raise _exc

        coord = _make_coordinator(
            tmp_path,
            backend=_SingleBackend(),
            l2_batch_evaluator=_raise_occ,
        )
        batch = CandidateBatch(
            round_idx=0,
            regime_spec=_make_regime_spec(),
            worker_specs=[
                ("r0_rule_0", "rule_exploit", None, _make_req("exploit_rule")),
            ],
            parent_module_source=_PARENT_MODULE_SRC,
            parent_snippet=_PARENT_SNIPPET,
            func_name="_build_main",
        )
        out = await coord.execute_round_gates(batch, _make_anchors())
        assert out.l2_infra_fault is not None
        assert out.l2_infra_fault.plane == FaultPlane.INFRA_OCCUPANCY
        assert out.l2_infra_fault.code == "L2_INFRA_OCCUPANCY"
        assert out.l2_infra_fault.failed_tier == "l2_infra_occupancy"
        assert out.l2_infra_fault.retriable is True
        assert out.l2_results == {}


@pytest.mark.anyio
async def test_cancel_pending_speculative_cancels_cleanly(tmp_path: Path) -> None:
    """cancel_pending_speculative cancels running speculative tasks without leaking."""
    started = asyncio.Event()
    cancelled_flag = False

    class _HangCreativeBackend:
        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            nonlocal cancelled_flag
            _ = effort_override
            if req.role == "explore_creative":
                started.set()
                try:
                    await asyncio.sleep(60.0)
                except asyncio.CancelledError:
                    cancelled_flag = True
                    raise
            return _EXPLOIT_SNIPPET

    coord = _make_coordinator(tmp_path, backend=_HangCreativeBackend())
    batch = CandidateBatch(
        round_idx=0,
        regime_spec=_make_regime_spec(),
        worker_specs=[
            ("r0_rule_0", "rule_exploit", None, _make_req("exploit_rule")),
            ("r0_creative", "creative_explore", None, _make_req("explore_creative")),
        ],
        parent_module_source=_PARENT_MODULE_SRC,
        parent_snippet=_PARENT_SNIPPET,
        func_name="_build_main",
    )
    await coord.execute_round_gates(batch, _make_anchors())
    await started.wait()
    assert len(coord._pending_speculative) == 1
    task = coord._pending_speculative[0][3]

    await coord.cancel_pending_speculative()
    assert coord._pending_speculative == []
    assert task.cancelled()
    assert cancelled_flag is True


def test_run_l1_static_width_dims_and_vf_stack_overflow(tmp_path: Path) -> None:
    """_run_l1 checks distinct static_width_dims_fn dims and flags VF_STACK_OVERFLOW."""
    seen_widths: list[int] = []

    def _l1_check(_src: str, spec: RegimeTargetSpec) -> StageBResult:
        d_val = int(spec.representative_anchor_dims.get("D", 0) or 0)
        seen_widths.append(d_val)
        if d_val >= 512:
            return StageBResult(
                passed=False,
                error_message="exceeded vf stack size (128 > 0)",
                compiler_summary=BishengResourceSummary(stack_bytes=128),
                elapsed_sec=0.1,
            )
        return StageBResult(passed=True, elapsed_sec=0.1)

    coord = _make_coordinator(
        tmp_path,
        l1_compiler=_l1_check,
        static_width_dims_fn=lambda _spec: ({"D": 128}, {"D": 256}, {"D": 512}),
    )
    res = coord._run_l1(_PARENT_MODULE_SRC, _make_regime_spec())
    assert seen_widths == [128, 256, 512]
    assert res.passed is False
    assert res.fault_code == "VF_STACK_OVERFLOW"
    assert res.fault_plane == "STAGE_B_COMPILE"
    unreserved_spec = CompileTaskSpec("k1", "_build_main", (), (), ())
    disabled_spec = CompileTaskSpec(
        "k2",
        "_build_main",
        (),
        (),
        ("--cce-disable-vf-stack-reserved-ubuf",),
    )
    assert has_unreserved_vf_stack_overflow(unreserved_spec, 0, "exceeded vf stack size")
    assert not has_unreserved_vf_stack_overflow(unreserved_spec, 0, "")
    assert not has_unreserved_vf_stack_overflow(unreserved_spec, 128, "")
    assert has_unreserved_vf_stack_overflow(disabled_spec, 128, "")


def test_paired_l2_normalization_and_unchanged_siblings(tmp_path: Path) -> None:
    """Paired L2 re-anchoring and unchanged sibling AST hashing work cleanly."""
    small_spec = RegimeTargetSpec(
        regime_tag="small",
        builder_candidates=("_build_small",),
        sample_builder_kwargs={},
        editable_symbols=("_build_small",),
    )
    large_spec = RegimeTargetSpec(
        regime_tag="large",
        builder_candidates=("_build_large",),
        sample_builder_kwargs={},
        editable_symbols=("_build_large",),
    )
    inc_src = (
        "def _build_small(tile=16):\n"
        "    return tile\n\n"
        "def _build_large(tile=64):\n"
        "    return tile\n"
    )
    cand_src = (
        "def _build_small(tile=32):\n"
        "    return tile\n\n"
        "def _build_large(tile=64):\n"
        "    return tile\n"
    )
    coord = _make_coordinator(
        tmp_path,
        regime_specs=(small_spec, large_spec),
    )
    coord._incumbent_anchor_us_by_case = {0: 20.0}
    unchanged = coord._unchanged_sibling_regimes(inc_src, cand_src, small_spec)
    assert unchanged == {"large"}

    paired_anc = AnchorEvalMetrics(
        case_num=0,
        regime_tag="small",
        accuracy_passed=True,
        cand_us=19.0,
        ref_perf_us=40.0,
        t_hw_us=4.0,
        speedup=40.0 / 19.0,
        roofline_eta=4.0 / 19.0,
        eq3_score=40.0 / 19.0,
        paired_speedup_vs_incumbent=2.0,
    )
    raw = {
        "c0": CandidateSurrogateResult(
            candidate_id="c0",
            operator_name="rms_norm",
            all_passed=True,
            avg_speedup=paired_anc.speedup,
            min_case_speedup=paired_anc.speedup,
            avg_roofline_eta=paired_anc.roofline_eta,
            surrogate_score=paired_anc.speedup,
            anchors=(paired_anc,),
        )
    }
    norm = coord._normalize_paired_l2_results(raw, None)
    assert norm["c0"].anchors[0].cand_us == pytest.approx(10.0)
    assert norm["c0"].anchors[0].speedup == pytest.approx(4.0)
    assert SpeculativeCreativeResult(0, "w", "src").worker_id == "w"


def test_tail_guard_recheck_and_combine_gate_helpers(tmp_path: Path) -> None:
    """Tail guard delegation, first-round L1 summary cache, and combine gate resolve cleanly."""
    l1_calls = 0

    def _counting_l1(_src: str, _reg: RegimeTargetSpec) -> StageBResult:
        nonlocal l1_calls
        l1_calls += 1
        return StageBResult(
            passed=True,
            compiler_summary=BishengResourceSummary(stack_bytes=0),
        )

    coord = _make_coordinator(tmp_path, l1_compiler=_counting_l1)
    spec = _make_regime_spec()
    summary1 = coord._l1_summary_for_first_round(_PARENT_MODULE_SRC, spec)
    summary2 = coord._l1_summary_for_first_round(_PARENT_MODULE_SRC, spec)
    assert summary1 is not None and summary1 is summary2
    assert l1_calls == 1

    best_res = _make_surrogate_result("inc", cand_us=20.0)
    better_res = _make_surrogate_result("cand", cand_us=10.0)
    ok, updated = coord._evaluate_dual_layer_tail_guard(
        candidate_id="cand",
        candidate_source=_PARENT_MODULE_SRC.replace("16", "32"),
        top_res=better_res,
        best_result=best_res,
        regime_spec=spec,
        active_anchors=_make_anchors(),
        incumbent_source=_PARENT_MODULE_SRC,
    )
    assert ok is True
    assert updated.all_passed is True
    assert len(coord._collect_regime_cases_by_id(spec)) >= 1


@pytest.mark.anyio
async def test_occupancy_error_clears_stale_l2_timing_meta(tmp_path: Path) -> None:
    """Failed L2 batches must not inherit timing or prewarm stats from prior rounds."""

    class _SingleBackend:
        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            _ = (req, effort_override)
            return _EXPLOIT_SNIPPET

    round_state = {"fail": False}

    def _l2_eval(
        cands: Sequence[tuple[str, Path]],
        _ancs: Sequence[AnchorCaseSpec],
    ) -> Mapping[str, CandidateSurrogateResult]:
        if round_state["fail"]:
            raise OccupancyExhaustedError(3, ("all NPU cards busy",))
        raw = {cid: _make_surrogate_result(cid) for cid, _path in cands}
        return L2BatchResults(
            raw,
            {cid: "run-0" for cid, _path in cands},
            prewarm_stats={
                "total_tasks": 3,
                "cache_hits": 1,
                "compiled_tasks": 2,
                "failed_candidates": 0,
            },
            l2_prewarm_sec=4.5,
            l2_npu_sec=2.5,
        )

    coord = _make_coordinator(
        tmp_path,
        backend=_SingleBackend(),
        l2_batch_evaluator=_l2_eval,
    )
    spec = _make_regime_spec()
    anchors = _make_anchors()
    batch_r0 = CandidateBatch(
        round_idx=0,
        regime_spec=spec,
        worker_specs=[("r0_rule_0", "rule_exploit", None, _make_req("exploit_rule"))],
        parent_module_source=_PARENT_MODULE_SRC,
        parent_snippet=_PARENT_SNIPPET,
        func_name="_build_main",
    )
    out_r0 = await coord.execute_round_gates(batch_r0, anchors)
    assert out_r0.l2_prewarm_sec == pytest.approx(4.5)
    assert out_r0.l2_npu_sec == pytest.approx(2.5)
    assert out_r0.round_prewarm_stats["compiled_tasks"] == 2
    assert coord._last_l2_timing_meta

    round_state["fail"] = True
    batch_r1 = CandidateBatch(
        round_idx=1,
        regime_spec=spec,
        worker_specs=[("r1_rule_0", "rule_exploit", None, _make_req("exploit_rule"))],
        parent_module_source=_PARENT_MODULE_SRC,
        parent_snippet=_PARENT_SNIPPET,
        func_name="_build_main",
    )
    out_r1 = await coord.execute_round_gates(batch_r1, anchors)
    assert coord._last_l2_timing_meta == {}
    assert out_r1.l2_infra_fault is not None
    assert out_r1.l2_infra_fault.plane == FaultPlane.INFRA_OCCUPANCY
    assert out_r1.l2_prewarm_sec == 0.0
    assert out_r1.l2_npu_sec == 0.0
    assert out_r1.round_prewarm_stats["compiled_tasks"] == 0


def test_compile_trackb_stage_b_forwards_unified_artifact_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_compile_trackb_stage_b forwards an ArtifactStore rooted at unified_tl_cache."""
    unified_cache = tmp_path / "unified_tl_cache"
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(unified_cache))
    captured: dict[str, object] = {}

    def _fake_compile(
        cpp_source: str,
        *,
        compile_flags: Sequence[str] | str | None = None,
        bisheng_bin: str | Path | None = None,
        cann_root: str | Path | None = None,
        artifact_store: ArtifactStore | None = None,
        regime_tag: str = "",
    ) -> StageBResult:
        captured["cpp_source"] = cpp_source
        captured["compile_flags"] = compile_flags
        captured["bisheng_bin"] = bisheng_bin
        captured["cann_root"] = cann_root
        captured["artifact_store"] = artifact_store
        captured["regime_tag"] = regime_tag
        return StageBResult(passed=True)

    monkeypatch.setattr(stage_gate_coordinator, "run_stage_b_ascendc_compile", _fake_compile)
    coord = _make_coordinator(tmp_path)
    res = coord._compile_trackb_stage_b(
        'extern "C" __global__ void k() {}',
        regime_tag="decode",
    )
    assert res.passed is True
    assert isinstance(captured["artifact_store"], ArtifactStore)
    assert captured["artifact_store"].root == unified_cache.resolve()
    assert captured["regime_tag"] == "decode"


@pytest.mark.anyio
async def test_speculative_creative_does_not_pile_up_when_pending(tmp_path: Path) -> None:
    """Do not launch an overlapping explore_creative task when one is still in flight."""
    creative_gate = asyncio.Event()
    creative_calls = 0

    class _Backend:
        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            nonlocal creative_calls
            _ = effort_override
            if req.role == "explore_creative":
                creative_calls += 1
                if creative_calls == 1:
                    await creative_gate.wait()
                return (
                    "# target_bottleneck: aiv_vec\n"
                    "# structural_hypothesis: Widen tile\n"
                    f"```python\n{_CREATIVE_SNIPPET}\n```"
                )
            return _EXPLOIT_SNIPPET

    coord = _make_coordinator(tmp_path, backend=_Backend())
    spec = _make_regime_spec()
    anchors = _make_anchors()

    batch_r0 = CandidateBatch(
        round_idx=0,
        regime_spec=spec,
        worker_specs=[
            ("r0_rule_0", "rule_exploit", None, _make_req("exploit_rule")),
            ("r0_creative", "creative_explore", None, _make_req("explore_creative")),
        ],
        parent_module_source=_PARENT_MODULE_SRC,
        parent_snippet=_PARENT_SNIPPET,
        func_name="_build_main",
    )
    await coord.execute_round_gates(batch_r0, anchors)
    assert creative_calls == 1
    assert len(coord._pending_speculative) == 1
    assert coord._pending_speculative[0][1] == "r0_creative"

    batch_r1 = CandidateBatch(
        round_idx=1,
        regime_spec=spec,
        worker_specs=[
            ("r1_rule_0", "rule_exploit", None, _make_req("exploit_rule")),
            ("r1_creative", "creative_explore", None, _make_req("explore_creative")),
        ],
        parent_module_source=_PARENT_MODULE_SRC,
        parent_snippet=_PARENT_SNIPPET,
        func_name="_build_main",
    )
    await coord.execute_round_gates(batch_r1, anchors)
    assert creative_calls == 1
    assert len(coord._pending_speculative) == 1
    assert coord._pending_speculative[0][1] == "r0_creative"

    creative_gate.set()
    await asyncio.sleep(0)

    batch_r2 = CandidateBatch(
        round_idx=2,
        regime_spec=spec,
        worker_specs=[
            ("r2_rule_0", "rule_exploit", None, _make_req("exploit_rule")),
            ("r2_creative", "creative_explore", None, _make_req("explore_creative")),
        ],
        parent_module_source=_PARENT_MODULE_SRC,
        parent_snippet=_PARENT_SNIPPET,
        func_name="_build_main",
    )
    out_r2 = await coord.execute_round_gates(batch_r2, anchors)
    assert creative_calls == 2
    r2_cids = [cid for cid, _ in out_r2.raw_candidates]
    assert "r0_creative" in r2_cids
    assert "r2_creative" in r2_cids


@pytest.mark.anyio
async def test_skip_repair_when_survivors_exist_forwarded_when_supported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward skip_repair_when_survivors_exist=True when supported and skip otherwise."""

    class _SingleBackend:
        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            _ = (req, effort_override)
            return _EXPLOIT_SNIPPET

    captured_flags: list[bool] = []

    async def _accepting_filter(
        candidates: Sequence[tuple[str, str]],
        *,
        patch_fn: object,
        l0_checker: object,
        l1_compiler: object,
        skip_repair_when_survivors_exist: bool = False,
        **_extra: object,
    ) -> list[L0L1CandidateCheck]:
        _ = (patch_fn, l0_checker, l1_compiler, _extra)
        captured_flags.append(skip_repair_when_survivors_exist)
        return [
            L0L1CandidateCheck(
                candidate_id=cid,
                passed=True,
                failed_tier=None,
                patched_module_source=_PARENT_MODULE_SRC,
                snippet_source=snip,
                stage_a=_make_stage_a(),
                stage_b=StageBResult(passed=True),
            )
            for cid, snip in candidates
        ]

    monkeypatch.setattr(stage_gate_coordinator, "run_l0_l1_round_filter", _accepting_filter)
    coord = _make_coordinator(tmp_path, backend=_SingleBackend())
    batch = CandidateBatch(
        round_idx=0,
        regime_spec=_make_regime_spec(),
        worker_specs=[("r0_rule_0", "rule_exploit", None, _make_req("exploit_rule"))],
        parent_module_source=_PARENT_MODULE_SRC,
        parent_snippet=_PARENT_SNIPPET,
        func_name="_build_main",
    )
    out = await coord.execute_round_gates(batch, _make_anchors())
    assert captured_flags == [True]
    assert "r0_rule_0" in out.l2_results

    legacy_called: list[str] = []

    async def _legacy_filter(
        candidates: Sequence[tuple[str, str]],
        *,
        patch_fn: object,
        l0_checker: object,
        l1_compiler: object,
    ) -> list[L0L1CandidateCheck]:
        _ = (patch_fn, l0_checker, l1_compiler)
        legacy_called.extend(cid for cid, _ in candidates)
        return []

    monkeypatch.setattr(stage_gate_coordinator, "run_l0_l1_round_filter", _legacy_filter)
    out_legacy = await coord.execute_round_gates(batch, _make_anchors())
    assert legacy_called == ["r0_rule_0"]
    assert out_legacy.passed_checks == []


def test_run_l1_tracka_forwards_anchor_specs_and_parallelizes_extra_widths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Track-A _run_l1 forwards anchor_case_specs and runs extra widths cleanly."""
    cases_dir = tmp_path / "tasks" / "level2" / "rms_norm"
    cases_dir.mkdir(parents=True)
    cases_file = cases_dir / "cases.yaml"
    cases_file.write_text("cases:\n  - case_id: 0\n  - case_id: 1\n", encoding="utf-8")

    monkeypatch.setattr(
        stage_gate_coordinator,
        "resolve_cannbench_repo",
        lambda **_kw: tmp_path,
    )
    monkeypatch.setattr(
        stage_gate_coordinator,
        "resolve_cannbench_tasks_root",
        lambda _repo: tmp_path / "tasks",
    )

    spec_by_case = {
        0: {"D": 128, "tl_dtype": "bfloat16"},
        1: {"D": 256, "tl_dtype": "bfloat16"},
    }

    def _fake_lookup_static(
        _op: str,
        case_num: int,
        **_kw: object,
    ) -> dict[str, object]:
        if int(case_num) not in spec_by_case:
            raise FileNotFoundError("missing case")
        return dict(spec_by_case[int(case_num)])

    monkeypatch.setattr(stage_gate_coordinator, "lookup_case_static_spec", _fake_lookup_static)

    captured_calls: list[dict[str, object]] = []

    def _fake_l1_check(
        _src: str,
        *,
        dims: Mapping[str, object] | None = None,
        static_spec_dims_list: Sequence[Mapping[str, object]] | None = None,
        anchor_case_specs: Sequence[tuple[int, str, Mapping[str, object]]] | None = None,
        operator_name: str = "",
        cases_yaml_sha256: str = "",
        **_kw: object,
    ) -> StageBResult:
        captured_calls.append(
            {
                "dims": dict(dims) if dims else {},
                "static_spec_dims_list": list(static_spec_dims_list or ()),
                "anchor_case_specs": list(anchor_case_specs or ()),
                "operator_name": operator_name,
                "cases_yaml_sha256": cases_yaml_sha256,
            }
        )
        return StageBResult(passed=True, elapsed_sec=0.25)

    monkeypatch.setattr(stage_gate_coordinator, "run_l1_compile_check", _fake_l1_check)

    mod_src = (
        "def _select_main_config(D=128, **_kw):\n"
        "    return {'tile_m': int(D)}\n"
        + _PARENT_MODULE_SRC
    )
    spec = RegimeTargetSpec(
        regime_tag="decode",
        builder_candidates=("_build_main",),
        sample_builder_kwargs={"tile_m": 16},
        representative_anchor_dims={"D": 128},
        config_selector_fn="_select_main_config",
        editable_symbols=("_select_main_config", "_build_main"),
    )

    coord = StageGateCoordinator(
        operator_name="rms_norm",
        source_root=tmp_path,
        target_module_path=tmp_path / "kernel.py",
        regime_specs=(spec,),
        scoreboard_anchors=(
            AnchorCaseSpec("level2/rms_norm", 0, "decode", 20.0, 2.0),
        ),
        guardrail_cases=(
            GuardrailCaseSpec("level2/rms_norm", 0, "decode", 20.0, 2.0),
            GuardrailCaseSpec("level2/rms_norm", 1, "decode", 25.0, 2.5),
            GuardrailCaseSpec("level2/rms_norm", 99, "decode", 30.0, 3.0),
        ),
        static_width_dims_fn=lambda _spec: (
            {"D": 128, "tl_dtype": "bfloat16"},
            {"D": 512, "tl_dtype": "bfloat16"},
            {"D": 1024, "tl_dtype": "bfloat16"},
        ),
    )
    resolved_specs = coord._resolve_stage_b_anchor_case_specs()
    assert [c for c, _r, _s in resolved_specs] == [0, 1]

    res = coord._run_l1(mod_src, spec)
    assert res.passed is True
    assert len(captured_calls) == 4


def test_run_l1_trackb_dedups_and_records_anchor_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Track-B _run_l1 deduplicates identical TUs and records anchor plan entries."""
    bin_file = tmp_path / "kernel.o"
    bin_file.write_bytes(b"ELF_TEST")
    tb_module = (
        '_TRACKB_CPP = """\n'
        'extern "C" __global__ void tb_kernel() {}\n'
        '"""\n'
        "def _render_trackb_tu(D=128, trackb_src='', **_kw):\n"
        "    return f'// D={D}\\n{trackb_src}'\n"
    )
    tb_spec = RegimeTargetSpec(
        regime_tag="decode",
        builder_candidates=(),
        sample_builder_kwargs={},
        editable_symbols=("_TRACKB_CPP", "_render_trackb_tu"),
        runtime_backend="ascendc_trackb",
        trackb_symbols=("_TRACKB_CPP",),
        representative_anchor_dims={"D": 128},
    )
    compile_tus: list[str] = []

    def _fake_ascendc_compile(
        cpp_source: str,
        **_kw: object,
    ) -> StageBResult:
        compile_tus.append(cpp_source)
        return StageBResult(
            passed=True,
            elapsed_sec=0.5,
            binary_paths=(str(bin_file),),
        )

    monkeypatch.setattr(
        stage_gate_coordinator,
        "run_stage_b_ascendc_compile",
        _fake_ascendc_compile,
    )
    monkeypatch.setattr(
        stage_gate_coordinator,
        "resolve_cannbench_repo",
        lambda **_kw: (_ for _ in ()).throw(FileNotFoundError("isolated")),
    )
    monkeypatch.setattr(
        stage_gate_coordinator,
        "lookup_case_static_spec",
        lambda _op, cnum, **_kw: {"D": 128 if int(cnum) in (0, 1) else 256},
    )
    coord = StageGateCoordinator(
        operator_name="rms_norm",
        source_root=tmp_path,
        target_module_path=tmp_path / "kernel.py",
        regime_specs=(tb_spec,),
        scoreboard_anchors=(
            AnchorCaseSpec("level2/rms_norm", 0, "decode", 20.0, 2.0),
            AnchorCaseSpec("level2/rms_norm", 1, "decode", 21.0, 2.1),
            AnchorCaseSpec("level2/rms_norm", 2, "decode", 22.0, 2.2),
        ),
        static_width_dims_fn=lambda _s: (),
    )
    res = coord._run_l1(tb_module, tb_spec)
    assert res.passed is True
    assert len(compile_tus) == 2
    assert res.elapsed_sec == pytest.approx(1.0)


def test_run_l1_hybrid_trackb_skips_compile_when_verified_in_eff_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip hybrid_trackb bisheng compile on verified eff_store hit unless missing/overridden."""
    unified_cache = tmp_path / "unified_tl_cache"
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(unified_cache))
    monkeypatch.delenv("TILEFOUNDRY_NPU_ENV_FILE", raising=False)

    cpp_body = 'extern "C" __global__ void hyb_tb_kernel() {}'
    hyb_src = (
        f'_HYB_CPP = """\n{cpp_body}\n"""\n'
        "def _build_hyb_prim(tile_m=16):\n"
        "    return int(tile_m)\n"
    )
    hyb_spec = RegimeTargetSpec(
        regime_tag="decode",
        builder_candidates=("_build_hyb_prim",),
        sample_builder_kwargs={"tile_m": 16},
        editable_symbols=("_build_hyb_prim", "_HYB_CPP"),
        runtime_backend="hybrid_trackb",
        trackb_symbols=("_HYB_CPP",),
        representative_anchor_dims={"D": 128},
    )

    bin_file = tmp_path / "prebuilt.o"
    bin_file.write_bytes(b"ELF_HYB_TEST")
    coord = StageGateCoordinator(
        operator_name="mla",
        source_root=tmp_path,
        target_module_path=tmp_path / "kernel.py",
        regime_specs=(hyb_spec,),
        scoreboard_anchors=_make_anchors(),
        static_width_dims_fn=lambda _s: (),
    )
    instantiated = coord._instantiate_trackb_tu(hyb_src, cpp_body, hyb_spec)
    norm_tu = _normalize_trackb_unit_source(instantiated, regime_tag="decode")
    eff_flags = list(resolve_trackb_compile_flags(instantiated, ()))
    t_key = compute_compile_task_key(
        normalized_kernel_unit_src=norm_tu,
        cann_target="dav-3510",
        pass_configs=None,
        compile_flags=eff_flags,
        static_spec_signature="",
    )
    store = ArtifactStore(unified_cache)
    store.record_verified_artifact(
        t_key,
        binary_paths=(bin_file,),
        backend_kind="ascendc",
        flags=" ".join(eff_flags),
        closure_src=norm_tu,
    )

    monkeypatch.setattr(
        stage_gate_coordinator,
        "run_l1_compile_check",
        lambda *_a, **_kw: StageBResult(passed=True, elapsed_sec=0.25),
    )
    monkeypatch.setattr(
        stage_gate_coordinator,
        "resolve_regime_config",
        lambda *_a, **_kw: types.SimpleNamespace(
            builder_name="_build_hyb_prim",
            builder_kwargs={"tile_m": 16},
            pass_configs={},
            compile_flags=(),
            selector_name="",
            combine_kwargs={},
        ),
    )
    monkeypatch.setattr(
        stage_gate_coordinator,
        "resolve_cannbench_repo",
        lambda **_kw: (_ for _ in ()).throw(FileNotFoundError("isolated")),
    )

    tb_compile_calls: list[str] = []

    def _fake_ascendc_compile(cpp_source: str, **_kw: object) -> StageBResult:
        tb_compile_calls.append(cpp_source)
        return StageBResult(passed=True, elapsed_sec=0.5, binary_paths=(str(bin_file),))

    monkeypatch.setattr(
        stage_gate_coordinator,
        "run_stage_b_ascendc_compile",
        _fake_ascendc_compile,
    )

    cached_res = coord._run_l1(hyb_src, hyb_spec)
    assert cached_res.passed is True
    assert tb_compile_calls == []
    assert cached_res.artifact_id == t_key
    assert cached_res.elapsed_sec == pytest.approx(0.25)

    edited_src = hyb_src.replace("hyb_tb_kernel()", "hyb_tb_kernel_v2()")
    miss_res = coord._run_l1(edited_src, hyb_spec)
    assert miss_res.passed is True
    assert len(tb_compile_calls) == 1
    assert miss_res.elapsed_sec == pytest.approx(0.75)

    owner_calls: list[str] = []
    owner_obj = types.SimpleNamespace(
        _compile_trackb_stage_b=lambda cpp: (
            owner_calls.append(cpp),
            StageBResult(passed=True, elapsed_sec=0.1),
        )[1]
    )
    coord_with_owner = StageGateCoordinator(
        operator_name="mla",
        source_root=tmp_path,
        target_module_path=tmp_path / "kernel.py",
        regime_specs=(hyb_spec,),
        scoreboard_anchors=_make_anchors(),
        static_width_dims_fn=lambda _s: (),
        owner=owner_obj,
    )
    overridden_res = coord_with_owner._run_l1(hyb_src, hyb_spec)
    assert overridden_res.passed is True
    assert len(owner_calls) == 1
    assert overridden_res.elapsed_sec == pytest.approx(0.35)


@pytest.mark.anyio
async def test_streamed_l0_l1_overlaps_fast_exploit_and_drained_creative_with_slow_llm(
    tmp_path: Path,
) -> None:
    """Streamed L0/L1 overlaps fast exploit and drained creative with slow LLM."""
    loop = asyncio.get_running_loop()
    overlap_seen = asyncio.Event()
    creative_gate = asyncio.Event()
    started_l1_tiles: set[str] = set()

    class _OverlapBackend:
        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            _ = effort_override
            if req.role == "explore_creative":
                await creative_gate.wait()
                return (
                    "# target_bottleneck: aiv_vec\n"
                    "# structural_hypothesis: Drained creative\n"
                    f"```python\n{_CREATIVE_SNIPPET}\n```"
                )
            if req.user_prompt == "slow":
                await asyncio.wait_for(overlap_seen.wait(), timeout=2.0)
                return _EXPLOIT_SNIPPET.replace("32", "48")
            return _EXPLOIT_SNIPPET

    def _l1_overlap(src: str, _reg: RegimeTargetSpec) -> StageBResult:
        if "tile_m=64" in src:
            started_l1_tiles.add("creative")
            time.sleep(0.05)
        elif "tile_m=32" in src:
            started_l1_tiles.add("fast")
            time.sleep(0.05)
        else:
            time.sleep(0.01)
        if {"creative", "fast"}.issubset(started_l1_tiles):
            loop.call_soon_threadsafe(overlap_seen.set)
        return StageBResult(passed=True)

    coord = _make_coordinator(tmp_path, backend=_OverlapBackend(), l1_compiler=_l1_overlap)
    spec = _make_regime_spec()
    anchors = _make_anchors()

    batch_r0 = CandidateBatch(
        round_idx=0,
        regime_spec=spec,
        worker_specs=[
            ("r0_rule_0", "rule_exploit", None, _make_req("exploit_rule")),
            ("r0_creative", "creative_explore", None, _make_req("explore_creative")),
        ],
        parent_module_source=_PARENT_MODULE_SRC,
        parent_snippet=_PARENT_SNIPPET,
        func_name="_build_main",
    )
    await coord.execute_round_gates(batch_r0, anchors)
    creative_gate.set()
    await asyncio.sleep(0.02)

    started_l1_tiles.clear()
    overlap_seen.clear()
    slow_req = GenerationRequest(
        system_prompt="sys",
        user_prompt="slow",
        role="exploit_rule",
        effort="medium",
    )
    fast_req = GenerationRequest(
        system_prompt="sys",
        user_prompt="fast",
        role="exploit_rule",
        effort="medium",
    )
    batch_r1 = CandidateBatch(
        round_idx=1,
        regime_spec=spec,
        worker_specs=[
            ("r1_slow", "rule_exploit", None, slow_req),
            ("r1_fast", "rule_exploit", None, fast_req),
        ],
        parent_module_source=_PARENT_MODULE_SRC,
        parent_snippet=_PARENT_SNIPPET,
        func_name="_build_main",
    )
    out_r1 = await coord.execute_round_gates(batch_r1, anchors)
    assert [cid for cid, _ in out_r1.raw_candidates] == [
        "r0_creative",
        "r1_slow",
        "r1_fast",
    ]
    assert [chk.candidate_id for chk in out_r1.l0_l1_checks] == [
        "r0_creative",
        "r1_slow",
        "r1_fast",
    ]
    assert out_r1.l0_l1_sec < 0.05


@pytest.mark.anyio
async def test_streamed_l0_l1_triggers_repair_only_when_zero_survive(
    tmp_path: Path,
) -> None:
    """Repair runs when 0 streamed candidates survive and skips when >=1 survives."""
    repair_calls = 0

    class _RepairBackend:
        def __init__(self, *, first_valid: bool) -> None:
            self._first_valid = first_valid

        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            nonlocal repair_calls
            _ = effort_override
            if "repair" in req.role.lower() or "repair" in req.system_prompt.lower():
                repair_calls += 1
                await asyncio.sleep(0.01)
                return _EXPLOIT_SNIPPET
            if self._first_valid and req.user_prompt == "c0":
                return _EXPLOIT_SNIPPET
            return "def _build_main(tile_m=999):\n    return int(tile_m)\n"

    def _l0_check(src: str, _reg: RegimeTargetSpec) -> StageAResult:
        if "tile_m=999" in src:
            return StageAResult(
                passed=False,
                prim_func=None,
                ledger=None,
                error_message="bad tile_m=999",
            )
        return _make_stage_a()

    target_path = tmp_path / "kernel.py"
    target_path.write_text(_PARENT_MODULE_SRC, encoding="utf-8")
    spec = _make_regime_spec()
    anchors = _make_anchors()

    coord_all_fail = StageGateCoordinator(
        operator_name="rms_norm",
        target_module_path=target_path,
        regime_specs=(spec,),
        initial_anchors=anchors,
        scoreboard_anchors=anchors,
        backend=_RepairBackend(first_valid=False),
        l0_checker=_l0_check,
        l1_compiler=lambda _s, _r: StageBResult(passed=True),
        l2_batch_evaluator=lambda cands, _a: {
            cid: _make_surrogate_result(cid) for cid, _p in cands
        },
        max_syntax_repairs=1,
    )
    req0 = GenerationRequest(
        role="exploit_rule",
        system_prompt="sys",
        user_prompt="c0",
        effort="medium",
    )
    req1 = GenerationRequest(
        role="exploit_rule",
        system_prompt="sys",
        user_prompt="c1",
        effort="medium",
    )
    batch = CandidateBatch(
        round_idx=0,
        regime_spec=spec,
        worker_specs=[
            ("w0", "rule_exploit", None, req0),
            ("w1", "rule_exploit", None, req1),
        ],
        parent_module_source=_PARENT_MODULE_SRC,
        parent_snippet=_PARENT_SNIPPET,
        func_name="_build_main",
    )
    out_repaired = await coord_all_fail.execute_round_gates(batch, anchors)
    assert repair_calls == 2
    assert out_repaired.syntax_repair_sec > 0.0
    assert [c.candidate_id for c in out_repaired.passed_checks] == ["w0", "w1"]
    assert all(c.repaired for c in out_repaired.passed_checks)

    repair_calls = 0
    coord_one_survives = StageGateCoordinator(
        operator_name="rms_norm",
        target_module_path=target_path,
        regime_specs=(spec,),
        initial_anchors=anchors,
        scoreboard_anchors=anchors,
        backend=_RepairBackend(first_valid=True),
        l0_checker=_l0_check,
        l1_compiler=lambda _s, _r: StageBResult(passed=True),
        l2_batch_evaluator=lambda cands, _a: {
            cid: _make_surrogate_result(cid) for cid, _p in cands
        },
        max_syntax_repairs=1,
    )
    out_skip = await coord_one_survives.execute_round_gates(batch, anchors)
    assert repair_calls == 0
    assert out_skip.syntax_repair_sec == 0.0
    assert [c.candidate_id for c in out_skip.passed_checks] == ["w0"]
    assert [c.candidate_id for c in out_skip.l0_l1_checks] == ["w0", "w1"]


@pytest.mark.anyio
async def test_monkeypatched_filter_stays_batched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Monkeypatched run_l0_l1_round_filter must run the single batched call."""

    class _StaggeredBackend:
        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            _ = effort_override
            if req.user_prompt == "slow":
                await asyncio.sleep(0.02)
            return _EXPLOIT_SNIPPET

    batched_calls: list[list[tuple[str, str]]] = []

    async def _spy_filter(
        candidates: Sequence[tuple[str, str]],
        **_kw: object,
    ) -> list[L0L1CandidateCheck]:
        batched_calls.append(list(candidates))
        return [
            L0L1CandidateCheck(
                candidate_id=cid,
                passed=True,
                failed_tier=None,
                patched_module_source=_PARENT_MODULE_SRC,
                snippet_source=snip,
                stage_a=_make_stage_a(),
                stage_b=StageBResult(passed=True),
            )
            for cid, snip in candidates
        ]

    monkeypatch.setattr(stage_gate_coordinator, "run_l0_l1_round_filter", _spy_filter)
    coord = _make_coordinator(tmp_path, backend=_StaggeredBackend())
    batch = CandidateBatch(
        round_idx=0,
        regime_spec=_make_regime_spec(),
        worker_specs=[
            (
                "w0",
                "rule_exploit",
                None,
                GenerationRequest(
                    role="exploit",
                    system_prompt="sys",
                    user_prompt="fast",
                    effort="medium",
                ),
            ),
            (
                "w1",
                "rule_exploit",
                None,
                GenerationRequest(
                    role="exploit",
                    system_prompt="sys",
                    user_prompt="slow",
                    effort="medium",
                ),
            ),
        ],
        parent_module_source=_PARENT_MODULE_SRC,
        parent_snippet=_PARENT_SNIPPET,
        func_name="_build_main",
    )
    out = await coord.execute_round_gates(batch, _make_anchors())
    assert len(batched_calls) == 1
    assert [cid for cid, _ in batched_calls[0]] == ["w0", "w1"]
    assert [chk.candidate_id for chk in out.passed_checks] == ["w0", "w1"]


def test_run_l1_hybrid_trackb_runs_trackb_fail_fast_before_tilelang(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid Track-B _run_l1 validates C++ and prebuilds anchor .so before TileLang."""
    unified_cache = tmp_path / "unified_tl_cache"
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(unified_cache))
    monkeypatch.delenv("TILEFOUNDRY_NPU_ENV_FILE", raising=False)

    hyb_src = (
        '_HYB_CPP = """\n'
        '__global__ void hyb_tb_kernel() {}\n'
        'extern "C" void launch_hyb() { hyb_tb_kernel<<<1, 0, 0>>>(); }\n'
        '"""\n'
        "def _render_trackb_tu(D=128, trackb_src='', **_kw):\n"
        "    return f'// D={D}\\n{trackb_src}'\n"
        "def _build_hyb_prim(tile_m=16):\n"
        "    return int(tile_m)\n"
    )
    hyb_spec = RegimeTargetSpec(
        regime_tag="decode",
        builder_candidates=("_build_hyb_prim",),
        sample_builder_kwargs={"tile_m": 16},
        editable_symbols=("_build_hyb_prim", "_HYB_CPP", "_render_trackb_tu"),
        runtime_backend="hybrid_trackb",
        trackb_symbols=("_HYB_CPP",),
        representative_anchor_dims={"D": 128},
    )

    monkeypatch.setattr(
        stage_gate_coordinator,
        "resolve_regime_config",
        lambda *_a, **_kw: types.SimpleNamespace(
            builder_name="_build_hyb_prim",
            builder_kwargs={"tile_m": 16},
            pass_configs={},
            compile_flags=(),
            selector_name="",
            combine_kwargs={},
        ),
    )
    monkeypatch.setattr(
        stage_gate_coordinator,
        "resolve_cannbench_repo",
        lambda **_kw: (_ for _ in ()).throw(FileNotFoundError("isolated")),
    )
    monkeypatch.setattr(
        stage_gate_coordinator,
        "lookup_case_static_spec",
        lambda _op, cnum, **_kw: {"D": 128 if int(cnum) == 0 else 256},
    )

    l1_calls: list[str] = []

    def _fake_l1_check(src: str, **_kw: object) -> StageBResult:
        l1_calls.append(src)
        return StageBResult(passed=True, elapsed_sec=0.25)

    monkeypatch.setattr(stage_gate_coordinator, "run_l1_compile_check", _fake_l1_check)

    bin_file = tmp_path / "prebuilt.o"
    bin_file.write_bytes(b"ELF_HYB_FF")
    tb_calls: list[dict[str, object]] = []
    sec_calls: list[str] = []
    tb_should_fail = True

    def _fake_ascendc_compile(
        cpp_source: str,
        *,
        verify_shared_link: bool = False,
        emit_shared_so_cache_dir: Path | None = None,
        **_kw: object,
    ) -> StageBResult:
        tb_calls.append(
            {
                "cpp_source": cpp_source,
                "verify_shared_link": verify_shared_link,
                "emit_shared_so_cache_dir": emit_shared_so_cache_dir,
            }
        )
        if tb_should_fail:
            return StageBResult(
                passed=False,
                error_message="syntax error in hyb_tb_kernel",
                elapsed_sec=0.2,
                fault_plane="STAGE_B_COMPILE",
                fault_code="STAGE_B_ASCENDC_COMPILE_FAILED",
                failed_tier="stage_b",
            )
        return StageBResult(
            passed=True,
            elapsed_sec=0.3,
            binary_paths=(str(bin_file),),
        )

    def _fake_compile_shared_so(
        cpp_source: str,
        *,
        cache_dir: Path,
        compile_flags: Sequence[str] | str | None = None,
        **_kw: object,
    ) -> tuple[bool, Path | None, str]:
        sec_calls.append(cpp_source)
        so_p = Path(cache_dir) / stage_gate_coordinator.compute_trackb_shared_so_filename(
            cpp_source, compile_flags
        )
        so_p.parent.mkdir(parents=True, exist_ok=True)
        so_p.write_bytes(b"ELF_SO_SEC")
        return True, so_p, ""

    monkeypatch.setattr(
        stage_gate_coordinator,
        "run_stage_b_ascendc_compile",
        _fake_ascendc_compile,
    )
    monkeypatch.setattr(
        stage_gate_coordinator,
        "compile_trackb_shared_so",
        _fake_compile_shared_so,
    )

    coord = StageGateCoordinator(
        operator_name="mla",
        source_root=tmp_path,
        target_module_path=tmp_path / "kernel.py",
        regime_specs=(hyb_spec,),
        scoreboard_anchors=(
            AnchorCaseSpec("level4/mla", 0, "decode", 20.0, 2.0),
            AnchorCaseSpec("level4/mla", 1, "prefill", 20.0, 2.0),
        ),
        static_width_dims_fn=lambda _s: (),
    )

    fail_res = coord._run_l1(hyb_src, hyb_spec)
    assert fail_res.passed is False
    assert "trackb_ascendc(_HYB_CPP):" in fail_res.error_message
    assert len(l1_calls) == 0
    assert len(tb_calls) == 1
    assert tb_calls[0]["verify_shared_link"] is True
    assert tb_calls[0]["emit_shared_so_cache_dir"] == unified_cache
    assert len(sec_calls) == 1
    assert "// D=256" in sec_calls[0]

    tb_calls.clear()
    sec_calls.clear()
    l1_calls.clear()
    tb_should_fail = False

    pass_res = coord._run_l1(hyb_src, hyb_spec)
    assert pass_res.passed is True
    assert len(tb_calls) == 1
    assert len(l1_calls) == 1
    assert len(sec_calls) == 0
    assert pass_res.elapsed_sec == pytest.approx(0.55)








