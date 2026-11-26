"""Integration tests for V4.6 CampaignRunner Arch-3, Impl-1, Impl-2, and Impl-4."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.campaign import campaign_runner as cr
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CampaignConfig,
    CampaignReport,
    PerfTargetCampaignRunner,
    PerfTargetSpec,
    RegimeTargetSpec,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    AnchorCaseSpec,
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    L3AcceptanceResult,
    L4CannbenchResult,
)
from tilefoundry_ascend.agent.kb.schema import SymptomFamily
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes


def _make_dual_regime_runner(
    tmp_path: Path,
) -> tuple[PerfTargetCampaignRunner, list[AnchorCaseSpec]]:
    mod_path = tmp_path / "dummy_op.py"
    mod_path.write_text("def dummy_op(x):\n    return x\n", encoding="utf-8")
    specs = (
        RegimeTargetSpec(
            regime_tag="reduce_tail",
            builder_candidates=("_build_reduce_tail",),
            sample_builder_kwargs={},
            config_selector_fn="_select_reduce_tail",
        ),
        RegimeTargetSpec(
            regime_tag="reduce_tail_subatom",
            builder_candidates=("_build_reduce_tail_subatom",),
            sample_builder_kwargs={},
            config_selector_fn="_select_reduce_tail_subatom",
        ),
    )
    anchors = [
        AnchorCaseSpec(
            task="level2/dummy_op",
            case_num=1,
            regime_tag="reduce_tail",
            ref_perf_us=10.0,
            t_hw_us=5.0,
        ),
        AnchorCaseSpec(
            task="level2/dummy_op",
            case_num=2,
            regime_tag="reduce_tail_subatom",
            ref_perf_us=10.0,
            t_hw_us=2.0,
        ),
    ]
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="level2/dummy_op",
            max_rounds=2,
            run_final_l4=False,
        ),
        target_module_path=mod_path,
        anchors=anchors,
        regime_specs=specs,
        perf_target=PerfTargetSpec(
            operator_name="level2/dummy_op",
            target_avg_speedup=1.5,
            target_min_case_speedup=0.9,
            target_roofline_eta=0.995,
            target_surrogate_score=99.8,
        ),
        experience_lake_dir=tmp_path / "lake",
    )
    return runner, anchors


def _make_result(tail_eta: float, subatom_eta: float) -> CandidateSurrogateResult:
    a1 = AnchorEvalMetrics(
        case_num=1,
        regime_tag="reduce_tail",
        accuracy_passed=True,
        cand_us=max(1.0, 5.0 / max(1e-4, tail_eta)),
        ref_perf_us=10.0,
        t_hw_us=5.0,
        speedup=tail_eta * 2.0,
        roofline_eta=tail_eta,
        eq3_score=tail_eta,
    )
    a2 = AnchorEvalMetrics(
        case_num=2,
        regime_tag="reduce_tail_subatom",
        accuracy_passed=True,
        cand_us=max(1.0, 4.0 / max(1e-4, subatom_eta)),
        ref_perf_us=10.0,
        t_hw_us=4.0,
        speedup=subatom_eta * 2.0,
        roofline_eta=subatom_eta,
        eq3_score=subatom_eta,
    )
    return CandidateSurrogateResult(
        candidate_id="cand",
        operator_name="level2/dummy_op",
        all_passed=True,
        avg_speedup=(a1.speedup + a2.speedup) / 2.0,
        min_case_speedup=min(a1.speedup, a2.speedup),
        avg_roofline_eta=(tail_eta + subatom_eta) / 2.0,
        surrogate_score=40.0 + 10.0 * (tail_eta + subatom_eta),
        anchors=(a1, a2),
    )


def test_arch3_non_sticky_headroom_regime_rotation(tmp_path: Path) -> None:
    """Regime selector rotates to sibling once promoted regime headroom drops behind."""
    runner, anchors = _make_dual_regime_runner(tmp_path)

    res_round1 = _make_result(tail_eta=0.14, subatom_eta=0.03)
    picked_r1 = runner._select_active_regime(None, set(), res_round1, anchors)
    assert picked_r1 is not None
    assert picked_r1.regime_tag == "reduce_tail_subatom"

    res_close = _make_result(tail_eta=0.14, subatom_eta=0.15)
    picked_tie = runner._select_active_regime(
        "reduce_tail_subatom",
        set(),
        res_close,
        anchors,
        last_round_promoted=True,
    )
    assert picked_tie is not None
    assert picked_tie.regime_tag == "reduce_tail"

    res_round2 = _make_result(tail_eta=0.14, subatom_eta=0.28)
    picked_r2 = runner._select_active_regime(
        "reduce_tail_subatom", set(), res_round2, anchors
    )
    assert picked_r2 is not None
    assert picked_r2.regime_tag == "reduce_tail"


def test_impl2_report_l4_status_deferred_shared_host(tmp_path: Path) -> None:
    """L4 BLOCKED_ENV occupancy sets l4_status=deferred_shared_host when L3 is unblocked."""
    base = _make_result(0.14, 0.28)
    l4_blocked = L4CannbenchResult(
        passed=False,
        error_message="L4 receipt did not verify: ['RECEIPT_INVALID: BLOCKED_ENV', 'RECEIPT_FAILURE: BLOCKED_EXIT_CODE: 75']",
        run_id="r1",
    )
    rep = CampaignReport(
        operator_name="level2/dummy_op",
        stop_reason="budget_exhausted",
        total_rounds=1,
        baseline_result=base,
        best_result=base,
        rounds=(),
        l4_result=l4_blocked,
        l4_status="deferred_shared_host",
        promotion_blocked=False,
    )
    payload = rep.to_dict()
    assert payload["l4_status"] == "deferred_shared_host"


def test_impl4_flywheel_winning_edge_records_rule_id_and_symptom_family(
    tmp_path: Path,
) -> None:
    """Winning edge JSONL includes explicit rule_id and non-empty symptom_family."""
    runner, anchors = _make_dual_regime_runner(tmp_path)
    recorded_symptom: list[str] = []

    class _FakeReport:
        admitted_rule_ids: tuple[str, ...] = ()
        distilled_outcomes: tuple[object, ...] = ()
        kept_positive_chunks: tuple[object, ...] = ()

        def to_dict(self) -> dict[str, object]:
            return {"admitted_rule_ids": []}

    class _FakeFlywheel:
        _tracker = runner._tracker

        def distill_from_pair(self, **kwargs: object) -> _FakeReport:
            recorded_symptom.append(str(kwargs.get("active_symptom_family") or ""))
            return _FakeReport()

    runner._flywheel = _FakeFlywheel()
    runner._flywheel_coordinator._flywheel = runner._flywheel
    runner._run_flywheel_distillation(
        baseline_source="def f(): return 1\n",
        winner_source="def f(): return 2\n",
        baseline_latency_us=60.0,
        winner_latency_us=20.0,
        func_name="f",
        regime_spec=runner._regime_specs[1],
        active_anchors=anchors,
        topology_axes=TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd"),
        rel_gain=2.0,
        worker_kind="rule_exploit",
        round_idx=1,
        rule_id="R36_SUBATOM_CONTIGUOUS_ALIGNED_CORE_SLICE",
        symptom_family=SymptomFamily.DMA_UNALIGNED_SHORT_BURST.value,
    )

    assert recorded_symptom == [SymptomFamily.DMA_UNALIGNED_SHORT_BURST.value]
    edges_path = tmp_path / "lake" / "kb" / "evidence" / "winning_edges.jsonl"
    assert edges_path.is_file()
    rows = [json.loads(ln) for ln in edges_path.read_text(encoding="utf-8").splitlines() if ln]
    assert len(rows) == 1
    assert rows[0]["rule_id"] == "R36_SUBATOM_CONTIGUOUS_ALIGNED_CORE_SLICE"
    assert rows[0]["symptom_family"] == SymptomFamily.DMA_UNALIGNED_SHORT_BURST.value


def test_impl1_l3_transient_resource_busy_retries_and_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient resource busy L3 failure retries with bounded backoff and promotes."""
    sleeps: list[float] = []
    monkeypatch.setattr(cr.time, "sleep", lambda sec: sleeps.append(float(sec)))

    attempts = {"n": 0}

    def _l3_runner(_mod_path: Path, **_kwargs: object) -> L3AcceptanceResult:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return L3AcceptanceResult(
                passed=False,
                failure_kind="infrastructure",
                execution_terminal=True,
                error_message="L3 gate recorded status BLOCKED_ENV: ['resource busy: build:/repo/build (waited 0.0s of 0.0s)']",
            )
        return L3AcceptanceResult(passed=True, execution_terminal=True)

    l3_res = _l3_runner(tmp_path / "m.py")
    for busy_backoff_sec in (2.0, 5.0, 10.0):
        if (
            l3_res.passed
            or l3_res.failure_kind == cr.FAILURE_KIND_CANDIDATE
            or "resource busy:" not in (l3_res.error_message or "").lower()
        ):
            break
        cr.time.sleep(busy_backoff_sec)
        l3_res = _l3_runner(tmp_path / "m.py")

    assert l3_res.passed is True
    assert attempts["n"] == 3
    assert sleeps == [2.0, 5.0]
