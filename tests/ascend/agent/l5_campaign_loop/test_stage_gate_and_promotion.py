"""Consolidated tests for StageGateCoordinator, L0/L1 filter, and Top-2 promotion."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.campaign.l3_l4_recovery import L3L4RecoveryCoordinator
from tilefoundry_ascend.agent.campaign.promotion_coordinator import (
    PendingPromotion,
    PromotionCoordinator,
    RankedL2Candidate,
    evaluate_promotion_gate,
    select_top_two_l2_survivors,
    to_conservation_point,
)
from tilefoundry_ascend.agent.campaign.stage_gate_coordinator import (
    CandidateBatch,
    OccupancyExhaustedError,
    StageGateCoordinator,
    StageGateOutcome,
    run_l0_l1_round_filter,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    FAILURE_KIND_CANDIDATE,
    FAILURE_KIND_INFRASTRUCTURE,
    AgentGatesDeviceFaultError,
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    L2BatchResults,
    L3AcceptanceResult,
    L4CannbenchResult,
    aggregate_surrogate_result,
)
from tilefoundry_ascend.agent.contracts import CampaignStateSnapshot, NpuExecutionPort
from tilefoundry_ascend.agent.orchestrator.gate_funnel import StageAResult, StageBResult
from tilefoundry_ascend.agent.orchestrator.llm_client import GenerationRequest
from tilefoundry_ascend.agent.specs.anchor_specs import (
    AnchorCaseSpec,
    GuardrailCaseSpec,
    RegimeTargetSpec,
)
from tilefoundry_ascend.agent.specs.fault_types import FaultPlane, L3Resolution
from tilefoundry_ascend.ascend.passes import build_objective_physical_ledger, build_scope_waterline

_PARENT_MODULE = (
    "def _build_main(tile_m=16):\n"
    "    return int(tile_m)\n\n"
    "def kernel(x):\n"
    "    return _build_main(), x\n"
)
_PARENT_SNIP = "def _build_main(tile_m=16):\n    return int(tile_m)\n"
_EXPLOIT_SNIP = "def _build_main(tile_m=32):\n    return int(tile_m)\n"
_CREATIVE_SNIP = "def _build_main(tile_m=64):\n    return int(tile_m)\n"


def _regime() -> RegimeTargetSpec:
    return RegimeTargetSpec(
        regime_tag="decode",
        builder_candidates=("_build_main",),
        editable_symbols=("_build_main",),
        sample_builder_kwargs={"tile_m": 16},
        representative_anchor_dims={"D": 128},
    )


def _anchors() -> tuple[AnchorCaseSpec, ...]:
    return (AnchorCaseSpec("op/synthetic", 0, "decode", 20.0, 2.0),)


def _stage_a() -> StageAResult:
    wl = {
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
        waterlines=wl,
        gm_roundtrip_bytes=16384,
        grid_blocks=16,
    )
    return StageAResult(passed=True, prim_func=None, ledger=ledger)


def _metric(
    case_num: int,
    cand_us: float,
    *,
    regime_tag: str = "decode",
    ref_us: float = 100.0,
    t_hw_us: float = 10.0,
    accuracy_passed: bool = True,
    paired_speedup: float = 0.0,
    burst_batch_cv: float = 0.01,
    ring_burst_k: int = 8,
    paired_rel_mad: float = 0.01,
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
        ring_burst_k=ring_burst_k,
        burst_batch_cv=burst_batch_cv,
        paired_speedup_vs_incumbent=paired_speedup,
        paired_rel_mad=paired_rel_mad,
    )


def _surrogate(cid: str, cand_us: float = 10.0, *, all_passed: bool = True) -> CandidateSurrogateResult:
    metrics = (_metric(1, cand_us, accuracy_passed=all_passed), _metric(2, cand_us * 1.1, accuracy_passed=all_passed))
    return aggregate_surrogate_result(cid, "synthetic_op", metrics)


class _FakeNpuPort:
    def __init__(self, latency_by_cid: Mapping[str, float] | None = None) -> None:
        self.calls: list[list[str]] = []
        self._lat = dict(latency_by_cid or {})

    def run_l2_batch(
        self,
        candidates: Sequence[tuple[str, Path]],
        anchors: Sequence[AnchorCaseSpec],
    ) -> L2BatchResults:
        self.calls.append([cid for cid, _ in candidates])
        res: dict[str, CandidateSurrogateResult] = {}
        rids: dict[str, str] = {}
        for cid, _ in candidates:
            lat = self._lat.get(cid, 10.0)
            res[cid] = aggregate_surrogate_result(
                cid,
                "synthetic_op",
                tuple(_metric(a.case_num, lat, regime_tag=a.regime_tag, ref_us=a.ref_perf_us, t_hw_us=a.t_hw_us) for a in anchors),
            )
            rids[cid] = f"npu_rid_{cid}"
        return L2BatchResults(res, rids, prewarm_stats={"total_tasks": len(candidates), "cache_hits": 1})

    def run_l3_contract(self, module_path: Path, **_kw: object) -> L3AcceptanceResult:
        _ = module_path
        return L3AcceptanceResult(passed=True)

    def run_l4_cannbench(self, module_path: Path, **_kw: object) -> L4CannbenchResult:
        _ = module_path
        return L4CannbenchResult(passed=True)


def test_stage_gate_coordinator_port_injection_and_speculative_creative(tmp_path: Path) -> None:
    """Verify screen_and_score_round via NpuExecutionPort and non-blocking creative merge."""
    target_path = tmp_path / "kernel.py"
    target_path.write_text(_PARENT_MODULE, encoding="utf-8")
    npu_port: NpuExecutionPort = _FakeNpuPort({"r0_rule_0": 8.0, "r0_creative": 5.0})

    persisted: list[tuple[str, int]] = []
    kb_events: list[tuple[str, bool]] = []
    gate = asyncio.Event()

    class _Backend:
        async def generate(self, req: GenerationRequest, *, effort_override: object = None) -> str:
            _ = effort_override
            if req.role == "explore_creative":
                await gate.wait()
                return (
                    "# target_bottleneck: aiv_vec\n"
                    "# structural_hypothesis: Widen tile to 64\n"
                    f"```python\n{_CREATIVE_SNIP}\n```"
                )
            return _EXPLOIT_SNIP

    coord = StageGateCoordinator(
        operator_name="synthetic_op",
        target_module_path=target_path,
        regime_specs=(_regime(),),
        initial_anchors=_anchors(),
        scoreboard_anchors=_anchors(),
        backend=_Backend(),
        npu_port=npu_port,
        l0_checker=lambda _s, _r: _stage_a(),
        l1_compiler=lambda _s, _r: StageBResult(passed=True),
        persist_artifact_fn=lambda chk, ridx: persisted.append((chk.candidate_id, ridx)),
        record_kb_event_fn=lambda **kw: kb_events.append((str(kw["rule_id"]), bool(kw["postcondition_realized"]))),
        max_syntax_repairs=0,
    )
    snap = CampaignStateSnapshot(
        run_id="run_port",
        round_index=0,
        incumbent_source=_PARENT_MODULE,
        incumbent_sha256="sha0",
        dispatch_floor_us=1.0,
        active_regime_tag="decode",
        guardrail_case_ids=(),
        stagnation_by_regime={"decode": 0},
    )
    req_ex = GenerationRequest("exploit_rule", "sys", "usr")
    req_cr = GenerationRequest("explore_creative", "sys", "usr")

    async def _run_two_rounds() -> tuple[StageGateOutcome, StageGateOutcome]:
        b0 = CandidateBatch(
            round_idx=0,
            regime_spec=_regime(),
            worker_specs=[("r0_rule_0", "rule_exploit", None, req_ex), ("r0_creative", "creative_explore", None, req_cr)],
            parent_module_source=_PARENT_MODULE,
            parent_snippet=_PARENT_SNIP,
            func_name="_build_main",
        )
        o0 = await coord.screen_and_score_round(snap, b0)
        gate.set()
        await asyncio.sleep(0)
        b1 = CandidateBatch(
            round_idx=1,
            regime_spec=_regime(),
            worker_specs=[("r1_rule_0", "rule_exploit", None, req_ex)],
            parent_module_source=_PARENT_MODULE,
            parent_snippet=_PARENT_SNIP,
            func_name="_build_main",
        )
        o1 = await coord.execute_round_gates(b1, _anchors(), snapshot=dataclasses.replace(snap, round_index=1))
        await coord.cancel_pending_speculative()
        return o0, o1

    out0, out1 = asyncio.run(_run_two_rounds())
    assert [cid for cid, _ in out0.raw_candidates] == ["r0_rule_0"]
    assert "r0_rule_0" in out0.l2_results
    assert out0.round_prewarm_stats["cache_hits"] == 1
    assert {"r0_creative", "r1_rule_0"} == {cid for cid, _ in out1.raw_candidates}
    assert "Widen tile to 64" in out1.meta_by_id["r0_creative"][2]
    assert ("r0_rule_0", 0) in persisted
    assert ("r0_creative", 1) in persisted


def test_stage_gate_typed_faults_and_run_l0_l1_filter(tmp_path: Path) -> None:
    """Verify OccupancyExhaustedError/DeviceFault mapping and run_l0_l1_round_filter repairs."""
    target_path = tmp_path / "kernel.py"
    target_path.write_text(_PARENT_MODULE, encoding="utf-8")

    class _Backend:
        async def generate(self, req: GenerationRequest, *, effort_override: object = None) -> str:
            _ = (req, effort_override)
            return _EXPLOIT_SNIP

    for exc_obj, expected_plane in (
        (OccupancyExhaustedError(2, ("busy",)), FaultPlane.INFRA_OCCUPANCY),
        (
            AgentGatesDeviceFaultError("fault", fault_plane="INFRA_DEVICE_FAULT", fault_code="DEVICE_ERR"),
            FaultPlane.INFRA_DEVICE_FAULT,
        ),
    ):
        def _raise_l2(_c: object, _a: object, *, _e: BaseException = exc_obj) -> dict[str, CandidateSurrogateResult]:
            raise _e

        coord = StageGateCoordinator(
            operator_name="synthetic_op",
            target_module_path=target_path,
            regime_specs=(_regime(),),
            initial_anchors=_anchors(),
            backend=_Backend(),
            l0_checker=lambda _s, _r: _stage_a(),
            l1_compiler=lambda _s, _r: StageBResult(passed=True),
            l2_batch_evaluator=_raise_l2,
            max_syntax_repairs=0,
        )
        batch = CandidateBatch(
            round_idx=0,
            regime_spec=_regime(),
            worker_specs=[("r0_rule_0", "rule_exploit", None, GenerationRequest("exploit_rule", "s", "u"))],
            parent_module_source=_PARENT_MODULE,
            parent_snippet=_PARENT_SNIP,
            func_name="_build_main",
        )
        out = asyncio.run(coord.execute_round_gates(batch, _anchors()))
        assert out.l2_infra_fault is not None
        assert out.l2_infra_fault.plane == expected_plane

    class _RepairBackend:
        async def generate(self, req: GenerationRequest, *, effort_override: object = None) -> str:
            _ = (req, effort_override)
            return f"```python\n{_EXPLOIT_SNIP}```"

    def _l0_fn(src: str) -> StageAResult:
        if "bad" in src:
            return StageAResult(passed=False, prim_func=None, ledger=None, error_message="bad syntax", fault_code="STAGE_A_SYNTAX")
        return _stage_a()

    res = asyncio.run(
        run_l0_l1_round_filter(
            [("c_noop", _PARENT_SNIP), ("c_rep", "def _build_main(bad=1):\n    return bad\n")],
            patch_fn=lambda snip: snip,
            l0_checker=_l0_fn,
            l1_compiler=lambda _s: StageBResult(passed=True),
            repair_backend=_RepairBackend(),
            max_syntax_repairs=1,
            parent_source=_PARENT_SNIP,
        )
    )
    by_id = {c.candidate_id: c for c in res}
    assert by_id["c_noop"].failed_tier == "noop_identical"
    assert by_id["c_rep"].passed is True
    assert by_id["c_rep"].repaired is True


def test_top_two_l2_survivors_and_l3_intra_round_fallback(tmp_path: Path) -> None:
    """Verify Top-2 ranking, rank-1 promotion, rank-2 fallback, dual rejection, and deferral."""
    target_path = tmp_path / "kernels.py"
    incumbent_src = "INCUMBENT = 1\n"
    target_path.write_text(incumbent_src, encoding="utf-8")

    def _write_and_verify(path: Path, text: str) -> str:
        path.write_text(text, encoding="utf-8")
        return ""

    coordinator = PromotionCoordinator(target_module_path=target_path, write_and_verify_fn=_write_and_verify)
    regime = RegimeTargetSpec("decode", ("_build",), {})
    anchors = (AnchorCaseSpec("op/synthetic", 1, "decode", 100.0, 10.0), AnchorCaseSpec("op/synthetic", 2, "decode", 100.0, 10.0))
    best_res = _surrogate("inc", 60.0)

    cands = (
        RankedL2Candidate("c_fail", "FAIL = 1\n", _surrogate("c_fail", 20.0, all_passed=False)),
        RankedL2Candidate("c_slow", "SLOW = 1\n", _surrogate("c_slow", 59.5)),
        RankedL2Candidate("c_fast", "FAST = 1\n", _surrogate("c_fast", 30.0)),
        RankedL2Candidate("c_good", "GOOD = 1\n", _surrogate("c_good", 40.0)),
    )
    ranked = select_top_two_l2_survivors(cands, best_result=best_res, regime_spec=regime, active_anchors=anchors)
    assert ranked.rank_1 is not None and ranked.rank_1.candidate_id == "c_fast"
    assert ranked.rank_2 is not None and ranked.rank_2.candidate_id == "c_good"
    assert dict(ranked.rejected) == {"c_fail": "l2_not_passed", "c_slow": "insufficient_gain"}

    empty = coordinator.coordinate_top_two_l3_promotion(
        select_top_two_l2_survivors((), best_result=best_res, regime_spec=regime),
        incumbent_source=incumbent_src,
        round_idx=1,
        regime_tag="decode",
        run_l3_fn=lambda _c: L3AcceptanceResult(passed=True),
    )
    assert empty.resolution is L3Resolution.NOT_RUN

    win_fb = coordinator.coordinate_top_two_l3_promotion(
        ranked,
        incumbent_source=incumbent_src,
        round_idx=2,
        regime_tag="decode",
        run_l3_fn=lambda c: (
            L3AcceptanceResult(passed=False, failure_kind=FAILURE_KIND_CANDIDATE, failed_case_nums=(10, 15))
            if c.candidate_id == "c_fast"
            else L3AcceptanceResult(passed=True, run_id="r2")
        ),
    )
    assert win_fb.resolution is L3Resolution.PROMOTED
    assert win_fb.promoted_candidate is not None and win_fb.promoted_candidate.candidate_id == "c_good"
    assert win_fb.fallback_used is True
    assert win_fb.new_guardrail_cases == (10, 15)
    assert target_path.read_text(encoding="utf-8") == "GOOD = 1\n"

    both_rej = coordinator.coordinate_top_two_l3_promotion(
        ranked,
        incumbent_source=incumbent_src,
        round_idx=3,
        regime_tag="decode",
        run_l3_fn=lambda c: L3AcceptanceResult(
            passed=False,
            failure_kind=FAILURE_KIND_CANDIDATE,
            failed_case_nums=(10,) if c.candidate_id == "c_fast" else (10, 22),
        ),
    )
    assert both_rej.resolution is L3Resolution.CANDIDATE_REJECTED
    assert both_rej.new_guardrail_cases == (10, 22)
    assert target_path.read_text(encoding="utf-8") == incumbent_src

    deferred = coordinator.coordinate_top_two_l3_promotion(
        ranked,
        incumbent_source=incumbent_src,
        round_idx=4,
        regime_tag="decode",
        run_l3_fn=lambda _c: L3AcceptanceResult(passed=False, failure_kind=FAILURE_KIND_INFRASTRUCTURE, execution_terminal=False),
    )
    assert deferred.resolution is L3Resolution.INFRA_DEFERRED
    assert isinstance(deferred.pending_promotion, PendingPromotion)
    assert target_path.read_text(encoding="utf-8") == incumbent_src


def test_promotion_gate_guardrails_sibling_recheck_and_monotonic_guardrails(tmp_path: Path) -> None:
    """Verify paired guardrails, sibling recheck, and monotonic L3 guardrail case accumulation."""
    regime = RegimeTargetSpec("decode", ("_build",), {})
    anchors = (
        AnchorCaseSpec("op/synthetic", 1, "decode", 40.0, 10.0),
        AnchorCaseSpec("op/synthetic", 2, "prefill", 50.0, 12.0),
    )
    best_res = aggregate_surrogate_result(
        "inc",
        "synthetic_op",
        (_metric(1, 50.0, regime_tag="decode", ref_us=40.0), _metric(2, 60.0, regime_tag="prefill", ref_us=50.0)),
    )
    cand_res = aggregate_surrogate_result(
        "cand",
        "synthetic_op",
        (
            _metric(1, 40.0, regime_tag="decode", ref_us=40.0, paired_speedup=1.25),
            _metric(2, 60.0, regime_tag="prefill", ref_us=50.0, paired_speedup=1.0),
        ),
    )
    guardrail = GuardrailCaseSpec("op/synthetic", 3, "decode", incumbent_ref_us=42.0, t_hw_us=10.0)

    ok_fast, _, r_fast = evaluate_promotion_gate(
        candidate_id="cand",
        cur_res=cand_res,
        best_result=best_res,
        regime_spec=regime,
        guardrail_cases=(guardrail,),
        cand_guardrail_metrics={3: _metric(3, 52.0, paired_speedup=1.05)},
        unchanged_siblings=set(),
        active_anchors=anchors,
    )
    assert ok_fast is True and r_fast == ""

    ok_reg, _, r_reg = evaluate_promotion_gate(
        candidate_id="cand",
        cur_res=cand_res,
        best_result=best_res,
        regime_spec=regime,
        guardrail_cases=(guardrail,),
        cand_guardrail_metrics={3: _metric(3, 65.0, paired_speedup=0.75)},
        unchanged_siblings=set(),
        active_anchors=anchors,
    )
    assert ok_reg is False and r_reg == "guardrail_latency_regression"

    ok_acc, _, r_acc = evaluate_promotion_gate(
        candidate_id="cand",
        cur_res=cand_res,
        best_result=best_res,
        regime_spec=regime,
        guardrail_cases=(guardrail,),
        cand_guardrail_metrics={3: _metric(3, 35.0, accuracy_passed=False)},
        unchanged_siblings=set(),
        active_anchors=anchors,
    )
    assert ok_acc is False and r_acc == "guardrail_accuracy_failed"

    dipped_cand = aggregate_surrogate_result(
        "cand",
        "synthetic_op",
        (_metric(1, 35.0, regime_tag="decode", ref_us=40.0), _metric(2, 95.0, regime_tag="prefill", ref_us=50.0)),
    )
    ok_recheck, updated, r_recheck = evaluate_promotion_gate(
        candidate_id="cand",
        cur_res=dipped_cand,
        best_result=best_res,
        regime_spec=regime,
        guardrail_cases=(),
        cand_guardrail_metrics={},
        unchanged_siblings=set(),
        active_anchors=anchors,
        recheck_single_sibling_fn=lambda sp: aggregate_surrogate_result(
            "cand", "synthetic_op", (_metric(sp.case_num, 59.0, regime_tag="prefill", ref_us=50.0),)
        ),
    )
    assert ok_recheck is True and r_recheck == ""
    assert any(a.case_num == 2 and a.cand_us == pytest.approx(59.0) for a in updated.anchors)

    pt = to_conservation_point(_metric(1, 20.0), _metric(1, 30.0))
    assert pt.inc_us == pytest.approx(30.0)

    target_path = tmp_path / "k.py"
    target_path.write_text(_PARENT_MODULE, encoding="utf-8")
    recovery = L3L4RecoveryCoordinator(
        operator_name="synthetic_op",
        target_module_path=target_path,
        source_root=tmp_path,
        scoreboard_anchors=anchors,
        regime_specs=(regime,),
        lookup_case_dims_fn=lambda c, _t: {"M": 64, "K": c},
        l2_batch_runner=lambda cands, specs: (
            {cands[0][0]: aggregate_surrogate_result("inc", "synthetic_op", tuple(_metric(s.case_num, 15.0 + s.case_num) for s in specs))},
            {},
        ),
    )
    added1 = recovery.record_l3_guardrail_cases(
        L3AcceptanceResult(passed=False, failure_kind=FAILURE_KIND_CANDIDATE, failed_case_nums=(1, 14, 15)),
        _PARENT_MODULE,
        regime,
    )
    added2 = recovery.record_l3_guardrail_cases(
        L3AcceptanceResult(passed=False, failure_kind=FAILURE_KIND_CANDIDATE, failed_case_nums=(14, 16)),
        _PARENT_MODULE,
        regime,
    )
    assert [g.case_num for g in added1] == [14, 15]
    assert [g.case_num for g in added2] == [16]
    assert [g.case_num for g in recovery.guardrail_cases] == [14, 15, 16]
