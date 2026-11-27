"""Consolidated tests for CampaignStateSnapshot, CampaignRunner loop, and scoreboard."""

from __future__ import annotations

import ast
import asyncio
import dataclasses
from collections.abc import Sequence
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.campaign import campaign_runner, stage_gate_coordinator
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CampaignConfig,
    CampaignReport,
    ExtremeCampaignRunner,
    PerfTargetCampaignRunner,
    build_resilient_l2_evaluator,
)
from tilefoundry_ascend.agent.contracts import (
    CampaignRoundRecord,
    CampaignStateSnapshot,
    RecoveryReceipt,
    RoundContextReceipt,
    StageGateRoundReceipt,
)
from tilefoundry_ascend.agent.gates.gate_funnel import StageAResult, StageBResult
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.kb.schema import AscendCausalRule, SymptomFamily
from tilefoundry_ascend.agent.npu import (
    AgentGatesDeviceFaultError,
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    L2BatchResults,
    L3AcceptanceResult,
    L4CannbenchResult,
    aggregate_surrogate_result,
    build_anchor_metrics,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import GenerationRequest, MockLLMBackend
from tilefoundry_ascend.agent.specs.anchor_specs import (
    AnchorCaseSpec,
    PerfTargetSpec,
    RegimeTargetSpec,
)
from tilefoundry_ascend.agent.specs.fault_types import L3Resolution
from tilefoundry_ascend.ascend.passes import build_objective_physical_ledger, build_scope_waterline
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import HirBlueprintPacket, HirRooflineSummary

_SLICE_SRC = (
    "def _select_cfg(tile_m=16):\n"
    "    return {'tile_m': int(tile_m)}\n\n"
    "def _build_main(tile_m=16):\n"
    "    return int(tile_m)\n"
)

_BASE_SRC = (
    f"{_SLICE_SRC}\n"
    "def kernel(x):\n"
    "    return _build_main(), x\n"
)

_MULTI_REGIME_SRC = (
    "def _build_small(tile_m=1):\n"
    "    return tile_m\n\n"
    "def _build_unaligned(tile_m=1):\n"
    "    return tile_m\n\n"
    "def _build_prefill(tile_m=4):\n"
    "    return tile_m\n"
)


def _make_ledger(ii: int = 16, blocks: int = 16) -> object:
    waterlines = {
        "UB": build_scope_waterline(
            scope="UB",
            capacity_bytes=262144,
            explicit_buffers={"a": 32768},
            implicit_buffers={},
        )
    }
    return build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=ii,
        pipe_cycles={"AIV0.PIPE_V": 120, "AIV1.PIPE_V": 120},
        waterlines=waterlines,
        gm_roundtrip_bytes=16384,
        grid_blocks=blocks,
        wave_occupancy=0.5,
    )


def _blueprint(entry_name: str = "kernel") -> HirBlueprintPacket:
    return HirBlueprintPacket(
        entry_name=entry_name,
        roofline=HirRooflineSummary(
            anchor_dims={"D": 128},
            compute_flops=1024,
            min_gm_bytes=8192,
            arithmetic_intensity=1.0,
            compute_floor_us=2.0,
            memory_floor_us=2.0,
            ideal_roofline_us=2.0,
            dominant_bound="memory_bound",
        ),
    )


def _make_metric(
    case_num: int,
    regime_tag: str,
    cand_us: float,
    *,
    ref_us: float = 100.0,
    t_hw_us: float = 10.0,
    accuracy_passed: bool = True,
) -> AnchorEvalMetrics:
    return AnchorEvalMetrics(
        case_num=case_num,
        regime_tag=regime_tag,
        accuracy_passed=accuracy_passed,
        cand_us=cand_us,
        ref_perf_us=ref_us,
        t_hw_us=t_hw_us,
        speedup=(ref_us / cand_us) if (accuracy_passed and cand_us > 0.0) else 0.0,
        roofline_eta=(t_hw_us / cand_us) if (accuracy_passed and cand_us > 0.0) else 0.0,
        eq3_score=(ref_us / cand_us) if (accuracy_passed and cand_us > 0.0) else 0.0,
        ring_burst_k=4,
        burst_batch_cv=0.02,
        paired_rel_mad=0.01,
    )


def _make_surrogate(
    cid: str,
    metrics: Sequence[AnchorEvalMetrics],
) -> CandidateSurrogateResult:
    return aggregate_surrogate_result(cid, "synthetic_op", tuple(metrics))


def _single_regime() -> RegimeTargetSpec:
    return RegimeTargetSpec(
        regime_tag="main",
        builder_candidates=("_build_main",),
        sample_builder_kwargs={"tile_m": 16},
        editable_symbols=("_build_main", "_select_cfg"),
        config_selector_fn="_select_cfg",
        representative_anchor_dims={"D": 128},
        representative_hir_dims={"D": 128},
    )


def _three_regimes() -> tuple[RegimeTargetSpec, ...]:
    return (
        RegimeTargetSpec(
            regime_tag="small_decode",
            builder_candidates=("_build_small",),
            sample_builder_kwargs={"tile_m": 1},
            editable_symbols=("_build_small",),
            representative_anchor_dims={"B": 1, "S": 1, "D": 4096},
            representative_hir_dims={"B": 1, "S": 1, "D": 64},
        ),
        RegimeTargetSpec(
            regime_tag="unaligned_d",
            builder_candidates=("_build_unaligned",),
            sample_builder_kwargs={"tile_m": 1},
            editable_symbols=("_build_unaligned",),
            representative_anchor_dims={"B": 1, "S": 16, "D": 1000},
            representative_hir_dims={"B": 1, "S": 4, "D": 1000},
        ),
        RegimeTargetSpec(
            regime_tag="aligned_prefill",
            builder_candidates=("_build_prefill",),
            sample_builder_kwargs={"tile_m": 4},
            editable_symbols=("_build_prefill",),
            representative_anchor_dims={"B": 1, "S": 256, "D": 4096},
            representative_hir_dims={"B": 1, "S": 16, "D": 512},
        ),
    )


def test_campaign_state_snapshot_immutability_and_zero_reflection_glue() -> None:
    """Verify CampaignStateSnapshot immutability and absence of reflection glue in L5."""
    snap = CampaignStateSnapshot(
        run_id="run_01",
        round_index=1,
        incumbent_source=_BASE_SRC,
        incumbent_sha256="abc123",
        dispatch_floor_us=1.5,
        active_regime_tag="main",
        guardrail_case_ids=("14", "15"),
        stagnation_by_regime={"main": 0},
    )
    assert dataclasses.is_dataclass(snap)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(snap, "round_index", 2)
    assert snap.last_round_promoted is True
    assert snap.frozen_regimes == ()

    for mod in (campaign_runner, stage_gate_coordinator):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "_sync_coordinators",
            "_PROPAGATED_HOOKS",
            "_owner_override",
            "_invoke_with_optional_kwargs",
            "self._owner",
        ):
            assert forbidden not in src
        tree = ast.parse(src)
        defined = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert "_sync_coordinators" not in defined
        assert "_invoke_with_optional_kwargs" not in defined


def test_campaign_runner_multi_round_convergence_and_timings(tmp_path: Path) -> None:
    """Verify multi-round convergence, rule admission, goal_met stop, and timing telemetry."""
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_BASE_SRC, encoding="utf-8")
    regime = _single_regime()
    anchors = (
        AnchorCaseSpec("op/synthetic", 1, "main", 100.0, 10.0),
        AnchorCaseSpec("op/synthetic", 2, "main", 100.0, 10.0),
    )
    rule = AscendCausalRule(
        rule_id="R_TILE_32",
        regime_guard="True",
        precondition="True",
        transform_template="Widen tile_m to 32 then 64.",
        expected_postcondition="measured_latency_us < prev.measured_latency_us",
        source_provenance="unit_test",
        symptom_family=SymptomFamily.COMPUTE_VECTOR_LOOP_DEGRADED.value,
    )
    tracker = BayesianRuleTracker.from_rules([rule])

    r1_src = _SLICE_SRC.replace("tile_m=16", "tile_m=32")
    r2_src = _SLICE_SRC.replace("tile_m=16", "tile_m=64")
    backend = MockLLMBackend(
        responses={
            "exploit_rule": [r1_src, r2_src],
            "explore_creative": [
                "# target_bottleneck: aiv\n# structural_hypothesis: Tile 32\n"
                f"```python\n{r1_src}```",
                "# target_bottleneck: aiv\n# structural_hypothesis: Tile 64\n"
                f"```python\n{r2_src}```",
            ],
        }
    )

    def _l2_eval(
        cands: Sequence[tuple[str, Path]],
        eval_anchors: Sequence[AnchorCaseSpec],
    ) -> L2BatchResults:
        out: dict[str, CandidateSurrogateResult] = {}
        rids: dict[str, str] = {}
        for cid, mpath in cands:
            src = mpath.read_text(encoding="utf-8")
            lat = 10.2 if "tile_m=64" in src else (25.0 if "tile_m=32" in src else 50.0)
            out[cid] = _make_surrogate(
                cid,
                tuple(_make_metric(a.case_num, a.regime_tag, lat) for a in eval_anchors),
            )
            rids[cid] = f"rid_{cid}"
        return L2BatchResults(
            out,
            rids,
            prewarm_stats={"total_tasks": 2, "cache_hits": 1, "compiled_tasks": 1, "failed_candidates": 0},
            l2_prewarm_sec=0.05,
            l2_npu_sec=0.25,
            null_launch_us=1.25,
        )

    l3_calls: list[str] = []

    def _l3_run(mod_path: Path) -> L3AcceptanceResult:
        text = mod_path.read_text(encoding="utf-8")
        l3_calls.append("64" if "tile_m=64" in text else "32")
        return L3AcceptanceResult(passed=True, execution_id="ex1", run_id="r1")

    l4_calls = 0

    def _l4_run(_mod_path: Path) -> L4CannbenchResult:
        nonlocal l4_calls
        l4_calls += 1
        return L4CannbenchResult(passed=True, case_count=2, passed_cases=2, per_case_elapsed_us={1: 10.2, 2: 10.2})

    ticks = iter([float(i) for i in range(1, 50)])
    ledger = _make_ledger()
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="synthetic_op",
            max_rounds=5,
            k_rule_workers=1,
            max_syntax_repairs=0,
            run_final_l4=True,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec(
            "synthetic_op",
            9.0,
            9.0,
            0.95,
            target_surrogate_score=5.0,
            min_optimization_rounds=2,
        ),
        anchors=anchors,
        regime_specs=(regime,),
        regime_blueprints={"main": _blueprint()},
        llm_backend=backend,
        tracker=tracker,
        l0_checker=lambda _s, _r: StageAResult(passed=True, prim_func=None, ledger=ledger),
        l1_compiler=lambda _s, _r: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=_l3_run,
        l4_cannbench_runner=_l4_run,
        msprof_collector=lambda *_a: {},
        p2_deep_dive_collector=lambda *_a: {},
        checkpoint_path=tmp_path / "ckpt.json",
        clock=lambda: next(ticks, 100.0),
    )

    report = asyncio.run(runner.run())
    assert isinstance(report, CampaignReport)
    assert report.stop_reason == "goal_met"
    assert report.total_rounds == 2
    assert report.promoted_rounds == (1, 2)
    assert report.admitted_rule_ids == ("R_TILE_32",)
    assert l3_calls == ["32", "64"]
    assert l4_calls == 1
    assert report.l4_status == "passed"
    assert report.calibrated_dispatch_floor_us == pytest.approx(1.25)
    assert report.geometric_speedup > 1.0
    for rec in report.rounds:
        assert "l0_l1_sec" in rec.stage_timings_sec
        assert "l2_npu_sec" in rec.stage_timings_sec
        assert rec.prewarm_stats["cache_hits"] == 1
        assert rec.cost_model_mode == "off"
    assert "tile_m=64" in target_path.read_text(encoding="utf-8")


def test_async_speculative_creative_merges_into_next_round(tmp_path: Path) -> None:
    """Verify slow explore_creative merges into subsequent round without blocking round 1."""
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_BASE_SRC, encoding="utf-8")
    regime = _single_regime()
    anchors = (AnchorCaseSpec("op/synthetic", 1, "main", 100.0, 10.0),)
    rule = AscendCausalRule(
        rule_id="R_SPEC",
        regime_guard="True",
        precondition="True",
        transform_template="Adjust tile_m.",
        expected_postcondition="True",
        source_provenance="unit_test",
    )
    r1_exploit = _SLICE_SRC.replace("tile_m=16", "tile_m=24")
    r1_creative = _SLICE_SRC.replace("tile_m=16", "tile_m=64")
    r2_exploit = _SLICE_SRC.replace("tile_m=16", "tile_m=32")
    creative_gate = asyncio.Event()

    class _StaggeredBackend:
        def __init__(self) -> None:
            self.exploit_idx = 0

        async def generate(self, req: GenerationRequest, *, effort_override: object = None) -> str:
            _ = effort_override
            if req.role == "explore_creative":
                await creative_gate.wait()
                return (
                    "# target_bottleneck: aiv\n# structural_hypothesis: Speculative 64\n"
                    f"```python\n{r1_creative}```"
                )
            self.exploit_idx += 1
            if self.exploit_idx == 2:
                creative_gate.set()
                await asyncio.sleep(0)
                return r2_exploit
            return r1_exploit

    def _l2_eval(
        cands: Sequence[tuple[str, Path]],
        eval_anchors: Sequence[AnchorCaseSpec],
    ) -> dict[str, CandidateSurrogateResult]:
        out: dict[str, CandidateSurrogateResult] = {}
        for cid, mpath in cands:
            src = mpath.read_text(encoding="utf-8")
            lat = 12.0 if "tile_m=64" in src else (30.0 if "tile_m=32" in src else (40.0 if "tile_m=24" in src else 50.0))
            out[cid] = _make_surrogate(cid, (_make_metric(eval_anchors[0].case_num, "main", lat),))
        return out

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="synthetic_op",
            max_rounds=3,
            k_rule_workers=1,
            max_syntax_repairs=0,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("synthetic_op", 10.0, 10.0, 0.99, min_optimization_rounds=5),
        anchors=anchors,
        regime_specs=(regime,),
        regime_blueprints={"main": _blueprint()},
        llm_backend=_StaggeredBackend(),
        tracker=BayesianRuleTracker.from_rules([rule]),
        l0_checker=lambda _s, _r: StageAResult(passed=True, prim_func=None, ledger=_make_ledger()),
        l1_compiler=lambda _s, _r: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        msprof_collector=lambda *_a: {},
        p2_deep_dive_collector=lambda *_a: {},
        checkpoint_path=tmp_path / "ckpt.json",
    )

    report = asyncio.run(runner.run())
    assert report.total_rounds == 3
    assert "r1_creative" not in report.rounds[0].l0_l1_passed_ids
    assert "r1_creative" in report.rounds[2].l0_l1_passed_ids
    assert "tile_m=64" in target_path.read_text(encoding="utf-8")


def test_scoreboard_invariance_budget_stop_and_l2_occupancy_deferral(tmp_path: Path) -> None:
    """Verify scoreboard immutability on L3 failure, budget stop, and occupancy deferral."""
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_MULTI_REGIME_SRC, encoding="utf-8")
    anchors = (
        AnchorCaseSpec("op/synthetic", 2, "small_decode", 15.0, 1.5),
        AnchorCaseSpec("op/synthetic", 4, "unaligned_d", 30.0, 5.0),
        AnchorCaseSpec("op/synthetic", 14, "unaligned_d", 60.0, 20.0),
        AnchorCaseSpec("op/synthetic", 16, "aligned_prefill", 500.0, 200.0),
        AnchorCaseSpec("op/synthetic", 17, "aligned_prefill", 500.0, 200.0),
    )
    regimes = _three_regimes()
    blueprints = {r.regime_tag: _blueprint() for r in regimes}

    class _MultiLLM:
        async def generate(self, req: GenerationRequest, *, effort_override: object = None) -> str:
            _ = effort_override
            if "unaligned_d" in req.user_prompt:
                code = "def _build_unaligned(tile_m=2):\n    return tile_m\n"
            elif "aligned_prefill" in req.user_prompt:
                code = "def _build_prefill(tile_m=2):\n    return tile_m\n"
            else:
                code = "def _build_small(tile_m=2):\n    return tile_m\n"
            if req.role == "explore_creative":
                return f"# target_bottleneck: aiv\n# structural_hypothesis: opt\n```python\n{code}```\n"
            return code

    eval_calls = {"n": 0}

    def _l2_eval(
        cands: Sequence[tuple[str, Path]],
        eval_anchors: Sequence[AnchorCaseSpec],
    ) -> dict[str, CandidateSurrogateResult]:
        eval_calls["n"] += 1
        if eval_calls["n"] == 2:
            raise AgentGatesDeviceFaultError(
                "BLOCKED_ENV: device occupied",
                fault_plane="INFRA_OCCUPANCY",
                fault_code="NPU_OCCUPANCY_BUSY",
                failed_tier="l2_npu",
                device_id="0",
            )
        out: dict[str, CandidateSurrogateResult] = {}
        for cid, mpath in cands:
            src = mpath.read_text(encoding="utf-8")
            is_fast = "tile_m=2" in src
            metrics = []
            for anc in eval_anchors:
                lat = 10.0 if (anc.regime_tag == "small_decode" and is_fast) else (
                    15.0 if anc.regime_tag == "small_decode" else (30.0 if anc.regime_tag == "unaligned_d" else 250.0)
                )
                metrics.append(build_anchor_metrics(anc, accuracy_passed=True, cand_us=lat))
            out[cid] = _make_surrogate(cid, metrics)
        return out

    ticks = iter([0.0, 0.1, 0.2, 100.0])
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="synthetic_op",
            max_rounds=3,
            max_wall_clock_min=1.0,
            k_rule_workers=0,
            max_syntax_repairs=0,
            run_final_l4=True,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("synthetic_op", 5.0, 2.0, 0.99),
        anchors=anchors,
        regime_specs=regimes,
        regime_blueprints=blueprints,
        llm_backend=_MultiLLM(),
        tracker=BayesianRuleTracker(rules={}),
        l0_checker=lambda _s, _r: StageAResult(passed=True, prim_func=None, ledger=_make_ledger()),
        l1_compiler=lambda _s, _r: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p: L3AcceptanceResult(
            passed=False,
            failure_kind="candidate",
            failure_stage="accuracy",
            failed_case_nums=(99,),
            error_message="failed case=99",
        ),
        l4_cannbench_runner=lambda _p: L4CannbenchResult(passed=True, case_count=5, passed_cases=5),
        msprof_collector=lambda *_a: {},
        p2_deep_dive_collector=lambda *_a: {},
        checkpoint_path=tmp_path / "ckpt.json",
        clock=lambda: next(ticks, 200.0),
    )
    report = asyncio.run(runner.run())
    assert report.stop_reason == "budget_exhausted"
    assert report.total_rounds == 2
    assert report.rounds[0].l3_resolution == L3Resolution.INFRA_DEFERRED.value
    assert report.rounds[1].l3_resolution == L3Resolution.CANDIDATE_REJECTED.value
    assert report.rounds[1].l3_failed_cases == (99,)
    assert report.scoreboard_anchor_count == 5
    assert [int(a.case_num) for a in report.scoreboard_anchors] == [2, 4, 14, 16, 17]
    assert report.best_result.avg_speedup == report.baseline_result.avg_speedup


def test_extreme_runner_rotation_and_resilient_l2(tmp_path: Path) -> None:
    """Verify ExtremeCampaignRunner stagnation-discounted rotation and resilient L2 batch."""
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_MULTI_REGIME_SRC, encoding="utf-8")
    r_a = RegimeTargetSpec("reg_a", ("_build_small",), {}, editable_symbols=("_build_small",))
    r_b = RegimeTargetSpec("reg_b", ("_build_unaligned",), {}, editable_symbols=("_build_unaligned",))
    anchors = (
        AnchorCaseSpec("op/synthetic", 1, "reg_a", 10.0, 2.0),
        AnchorCaseSpec("op/synthetic", 2, "reg_b", 10.0, 4.0),
    )

    seen_regimes: list[str] = []

    class _RegimeLLM:
        async def generate(self, req: GenerationRequest, *, effort_override: object = None) -> str:
            _ = effort_override
            tag = "reg_b" if "reg_b" in req.user_prompt else "reg_a"
            seen_regimes.append(tag)
            fn = "_build_unaligned" if tag == "reg_b" else "_build_small"
            code = f"def {fn}(tile_m=1):\n    return tile_m\n"
            return f"# target_bottleneck: aiv\n# structural_hypothesis: noop\n```python\n{code}```"

    base_res = _make_surrogate(
        "base",
        (_make_metric(1, "reg_a", 4.0, ref_us=10.0, t_hw_us=2.0), _make_metric(2, "reg_b", 6.0, ref_us=10.0, t_hw_us=4.0)),
    )
    runner = ExtremeCampaignRunner(
        config=CampaignConfig(operator_name="synthetic_op", max_rounds=2, k_rule_workers=0, max_syntax_repairs=0),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("synthetic_op", 5.0, 5.0, 0.99),
        anchors=anchors,
        regime_specs=(r_a, r_b),
        regime_blueprints={"reg_a": _blueprint(), "reg_b": _blueprint()},
        llm_backend=_RegimeLLM(),
        tracker=BayesianRuleTracker(rules={}),
        l0_checker=lambda _s, _r: StageAResult(passed=True, prim_func=None, ledger=_make_ledger()),
        l1_compiler=lambda _s, _r: StageBResult(passed=True),
        l2_batch_evaluator=lambda cands, _a: {cid: base_res for cid, _ in cands},
        msprof_collector=lambda *_a: {},
        p2_deep_dive_collector=lambda *_a: {},
        stagnation_discount=0.45,
    )
    rep = asyncio.run(runner.run())
    assert rep.total_rounds == 2
    assert set(seen_regimes) == {"reg_a", "reg_b"}

    eval_fn = build_resilient_l2_evaluator(
        operator_name="synthetic_op",
        worktree=tmp_path,
        env_file=tmp_path / "env.sh",
        device_id="1",
        batch_fn=lambda _op, cands, _ancs, **_kw: ({cid: base_res for cid, _ in cands}, "batch-rid", "rcp"),
    )
    shared = eval_fn([("c1", target_path), ("c2", target_path)], anchors)
    assert set(shared) == {"c1", "c2"}
    assert shared.run_ids == {"c1": "batch-rid", "c2": "batch-rid"}


@pytest.mark.parametrize(
    ("frozen_obj", "field_name", "bad_value"),
    [
        (
            CampaignStateSnapshot(
                run_id="r0",
                round_index=1,
                incumbent_source=_BASE_SRC,
                incumbent_sha256="sha0",
                dispatch_floor_us=1.0,
                active_regime_tag="main",
                guardrail_case_ids=("14",),
                stagnation_by_regime={"main": 0},
            ),
            "active_regime_tag",
            "other",
        ),
        (
            RoundContextReceipt(1, "main", "_build_main", "exploit", "sha0", "tilelang_dsl"),
            "stagnation_rounds",
            9,
        ),
        (
            StageGateRoundReceipt(1, "main", 2, ("c1",), (), ("c1",)),
            "infeasible_boundaries_added",
            5,
        ),
        (
            RecoveryReceipt("l4_cannbench", True, True),
            "passed",
            False,
        ),
    ],
)
def test_reject_snapshot_mutation_and_cross_object_private_writes(
    frozen_obj: object, field_name: str, bad_value: object
) -> None:
    """Reject mutations on frozen snapshots/receipts and cross-object private writes in L4/L5."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(frozen_obj, field_name, bad_value)
    agent_root = Path(campaign_runner.__file__).resolve().parents[1]
    violations: list[str] = []
    for subdir in ("npu", "probes", "campaign"):
        for py_path in sorted((agent_root / subdir).rglob("*.py")):
            tree = ast.parse(py_path.read_text(encoding="utf-8"), str(py_path))
            for node in ast.walk(tree):
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else ([node.target] if isinstance(node, ast.AugAssign) else ())
                )
                for target in targets:
                    if isinstance(target, ast.Attribute) and target.attr.startswith("_"):
                        if isinstance(target.value, ast.Name) and target.value.id in ("self", "cls"):
                            continue
                        violations.append(f"{py_path.name}:{node.lineno}")
    assert violations == []


@pytest.mark.parametrize("l3_passes", [False, True])
def test_accept_single_point_receipt_state_advancement(tmp_path: Path, l3_passes: bool) -> None:
    """Accept single-point receipt advancement and conserve guardrails/incumbent."""
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_BASE_SRC, encoding="utf-8")
    cand_snip = _SLICE_SRC.replace("tile_m=16", "tile_m=32")
    expected_promoted_src = _BASE_SRC.replace("tile_m=16", "tile_m=32")
    regime = _single_regime()
    anchors = (AnchorCaseSpec("op/synthetic", 1, "main", 100.0, 10.0),)

    class _LLM:
        async def generate(self, req: GenerationRequest, *, effort_override: object = None) -> str:
            _ = (req, effort_override)
            return f"# target_bottleneck: aiv\n# structural_hypothesis: widen\n```python\n{cand_snip}```"

    base_res = _make_surrogate("base", (_make_metric(1, "main", 50.0, ref_us=100.0, t_hw_us=10.0),))
    cand_res = _make_surrogate("r1_creative", (_make_metric(1, "main", 25.0, ref_us=100.0, t_hw_us=10.0),))

    def _l2_eval(cands: Sequence[tuple[str, Path]], eval_ancs: Sequence[AnchorCaseSpec]) -> L2BatchResults:
        _ = eval_ancs
        out: dict[str, CandidateSurrogateResult] = {}
        for cid, _ in cands:
            if cid in ("incumbent_baseline", "incumbent_slot"):
                out[cid] = base_res
            else:
                out[cid] = cand_res
        return L2BatchResults(out, {cid: "rid_1" for cid in out})

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="synthetic_op", max_rounds=1, k_rule_workers=0,
            max_syntax_repairs=0, run_final_l4=True, disable_early_goal_stop=True,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("synthetic_op", 10.0, 10.0, 0.99),
        anchors=anchors,
        regime_specs=(regime,),
        regime_blueprints={"main": _blueprint()},
        llm_backend=_LLM(),
        tracker=BayesianRuleTracker(rules={}),
        l0_checker=lambda _s, _r: StageAResult(passed=True, prim_func=None, ledger=_make_ledger()),
        l1_compiler=lambda _s, _r: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p, **_kw: L3AcceptanceResult(
            passed=l3_passes,
            failure_kind="" if l3_passes else "candidate",
            failed_case_nums=() if l3_passes else (77,),
            error_message="" if l3_passes else "case=77 fail",
        ),
        l4_cannbench_runner=lambda _p, **_kw: L4CannbenchResult(passed=True, execution_id="l4_ex"),
        msprof_collector=lambda *_a: {},
        p2_deep_dive_collector=lambda *_a: {},
    )
    report = asyncio.run(runner.run())
    assert isinstance(runner.last_round_context_receipt, RoundContextReceipt)
    assert runner.last_round_context_receipt.round_idx == 1
    assert isinstance(runner.last_stage_gate_receipt, StageGateRoundReceipt)
    assert runner.last_stage_gate_receipt.round_idx == 1
    assert isinstance(runner.last_recovery_receipt, RecoveryReceipt)
    assert runner.last_recovery_receipt.stage == "l4_cannbench"
    assert runner.last_recovery_receipt.passed is True
    if l3_passes:
        assert report.promoted_rounds == (1,)
        assert runner.guardrail_cases == ()
        assert target_path.read_text(encoding="utf-8") == expected_promoted_src
    else:
        assert report.rolled_back_rounds == (1,)
        assert [int(g.case_num) for g in runner.guardrail_cases] == [77]
        assert target_path.read_text(encoding="utf-8") == _BASE_SRC
