"""Unit tests for L3/L4 recovery, card-pool rotation, and monotonic guardrails."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.campaign import l3_l4_recovery
from tilefoundry_ascend.agent.specs.anchor_specs import (
    AnchorCaseSpec,
    GuardrailCaseSpec,
    RegimeTargetSpec,
)
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CampaignConfig,
    CampaignRoundRecord,
    ResolvedRegimeConfig,
)
from tilefoundry_ascend.agent.specs.fault_types import L3Resolution
from tilefoundry_ascend.agent.campaign.l3_l4_recovery import (
    L3L4RecoveryCoordinator,
    PendingL4,
    canonical_device_id,
    case_dim_alignment_flags,
    extract_knob_signature,
    extract_l3_failed_case_nums,
    is_prime_gt2,
    parse_device_ids,
    resolve_campaign_device_plan,
    resolve_l4_device_pool,
    source_sha256,
)
from tilefoundry_ascend.agent.campaign.promotion_coordinator import PendingPromotion
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    FAILURE_KIND_CANDIDATE,
    FAILURE_KIND_INFRASTRUCTURE,
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    L3AcceptanceResult,
    L4CannbenchResult,
    aggregate_surrogate_result,
)
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes


def _make_anchor_metric(
    case_num: int,
    regime_tag: str = "r0",
    *,
    cand_us: float = 5.0,
    ref_perf_us: float = 10.0,
    t_hw_us: float = 2.0,
    accuracy_passed: bool = True,
) -> AnchorEvalMetrics:
    speedup = ref_perf_us / cand_us if cand_us > 0.0 else 0.0
    eta = t_hw_us / cand_us if cand_us > 0.0 else 0.0
    return AnchorEvalMetrics(
        case_num=case_num,
        regime_tag=regime_tag,
        ref_perf_us=ref_perf_us,
        t_hw_us=t_hw_us,
        cand_us=cand_us,
        speedup=speedup,
        roofline_eta=eta,
        accuracy_passed=accuracy_passed,
        eq3_score=0.5,
    )


def _make_surrogate_result(
    candidate_id: str,
    case_nums: tuple[int, ...] = (1,),
    *,
    cand_us: float = 5.0,
) -> CandidateSurrogateResult:
    metrics = tuple(_make_anchor_metric(cnum, cand_us=cand_us) for cnum in case_nums)
    return aggregate_surrogate_result(candidate_id, "attention_residual", metrics)


def _make_round_record(round_idx: int = 1, regime_tag: str = "r0") -> CampaignRoundRecord:
    return CampaignRoundRecord(
        round_idx=round_idx,
        regime_tag=regime_tag,
        builder_name="_build",
        parent_lane="exploit",
        creative_effort="medium",
        l0_l1_passed_ids=("c1",),
        l0_l1_failed=(),
        best_round_speedup=2.0,
        best_so_far_speedup=2.0,
        promoted=False,
        l3_resolution=L3Resolution.INFRA_DEFERRED.value,
        stagnation_rounds_after=0,
    )


def test_zero_l3_cx_repair_symbols_in_l3_l4_recovery() -> None:
    """Verify inline L3 counterexample repair symbols are absent from l3_l4_recovery."""
    module_path = Path(l3_l4_recovery.__file__).resolve()
    source = module_path.read_text(encoding="utf-8")
    assert "l3_cx_repair" not in source
    assert "counterexample_repair" not in source
    tree = ast.parse(source)
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert all("cx_repair" not in name for name in names)
    assert all("cx_repair" not in attr for attr in dir(l3_l4_recovery))


def test_device_plan_and_l4_pool_resolution() -> None:
    """Verify device ID parsing, canonicalization, plan narrowing, and pool ordering."""
    assert parse_device_ids("") == ()
    assert parse_device_ids("0, 01, 2 ") == ("0", "01", "2")
    with pytest.raises(ValueError, match="empty entry"):
        parse_device_ids("0,,2")

    assert canonical_device_id(" 007 ", field="device_id") == "7"
    with pytest.raises(ValueError, match="nonnegative decimal device ID"):
        canonical_device_id("npu0", field="device_id")

    assert resolve_campaign_device_plan(
        device_id="",
        device_ids=(),
        env_device_ids=("02", "0"),
        visible_device_ids=("0",),
    ) == ("0", ("0",))
    assert resolve_campaign_device_plan(
        device_id="1",
        device_ids=(),
        env_device_ids=(),
        visible_device_ids=("1",),
    ) == ("1", ("1",))
    with pytest.raises(ValueError, match="not visible"):
        resolve_campaign_device_plan(
            device_id="0",
            device_ids=(),
            env_device_ids=("0", "2"),
            visible_device_ids=("2",),
        )
    with pytest.raises(ValueError, match="no card visible"):
        resolve_campaign_device_plan(
            device_id="",
            device_ids=(),
            env_device_ids=("0", "2"),
            visible_device_ids=("3",),
        )

    cfg = CampaignConfig(
        operator_name="attention_residual",
        device_id="07",
        device_ids=("2", "07", "10"),
        retry_device_id="2",
    )
    assert resolve_l4_device_pool(cfg) == ("7", "2", "10")


def test_pending_l4_roundtrip_and_legacy_keys() -> None:
    """Verify PendingL4 serialization roundtrip and legacy retry key compatibility."""
    legacy_payload = {
        "target_path": "kernels.py",
        "candidate_relative": "kernels.py",
        "candidate_sha256": "abc",
        "device_id": "3",
        "same_card_retries_used": 1,
        "same_card_retry_limit": 2,
    }
    restored = PendingL4.from_dict(legacy_payload)
    assert restored.blocked_env_rounds_used == 1
    assert restored.blocked_env_round_limit == 2
    serialized = restored.to_dict()
    assert serialized["blocked_env_rounds_used"] == 1
    assert serialized["blocked_env_round_limit"] == 2
    assert PendingL4.from_dict(serialized) == restored


def test_resolve_pending_promotion_promoted(tmp_path: Path) -> None:
    """Verify PROMOTED clears pending state, refreshes guardrails, and zeroes stagnation."""
    target_path = tmp_path / "kernels.py"
    ckpt_path = tmp_path / "ckpt.json"
    inc_src = "def _build():\n    return 1\n"
    cand_src = "def _build():\n    return 2\n"
    target_path.write_text(inc_src, encoding="utf-8")

    guardrails = [
        GuardrailCaseSpec(
            task="operator/attention_residual",
            case_num=14,
            regime_tag="r0",
            incumbent_ref_us=12.0,
            t_hw_us=2.0,
        )
    ]
    metrics_by_cid = {
        "c1": {14: _make_anchor_metric(14, "r0", cand_us=7.5)},
    }

    def _l3_resume(
        module_path: Path,
        *,
        resume_execution_id: str = "",
        resume_run_id: str = "",
        device_id: str = "",
        on_execution_started=None,
    ) -> L3AcceptanceResult:
        _ = on_execution_started
        assert module_path == target_path.resolve()
        assert resume_execution_id == "exec-1"
        assert resume_run_id == "run-1"
        assert device_id == "3"
        return L3AcceptanceResult(
            passed=True,
            execution_id="exec-1",
            run_id="run-1",
            device_id="3",
            execution_terminal=True,
        )

    coord = L3L4RecoveryCoordinator(
        operator_name="attention_residual",
        target_module_path=target_path,
        source_root=tmp_path,
        device_id="3",
        checkpoint_path=ckpt_path,
        l3_contract_runner=_l3_resume,
        guardrail_cases=guardrails,
        last_guardrail_metrics_by_cid=metrics_by_cid,
    )
    cand_res = _make_surrogate_result("c1", cand_us=4.0)
    base_res = _make_surrogate_result("base", cand_us=9.0)
    pending = PendingPromotion(
        candidate_source=cand_src,
        candidate_sha256=source_sha256(cand_src),
        target_path=str(target_path),
        round_idx=1,
        regime_tag="r0",
        candidate_result=cand_res,
        candidate_id="c1",
        execution_id="exec-1",
        run_id="run-1",
        device_id="3",
        execution_terminal=False,
    )
    rounds = [_make_round_record(round_idx=1, regime_tag="r0")]
    promoted_rounds: list[int] = []
    rolled_back_rounds: list[int] = [1]
    stagnation = {"r0": 2}

    new_inc, new_best, new_stag, next_pending, blocked = coord.resolve_pending_promotion(
        pending,
        incumbent_source=inc_src,
        best_result=base_res,
        stagnation_by_regime=stagnation,
        rounds=rounds,
        promoted_rounds=promoted_rounds,
        rolled_back_rounds=rolled_back_rounds,
        active_anchors=(),
    )
    assert blocked is False
    assert next_pending is None
    assert new_inc == cand_src
    assert new_best == cand_res
    assert new_stag["r0"] == 0
    assert promoted_rounds == [1]
    assert rolled_back_rounds == []
    assert target_path.read_text(encoding="utf-8") == cand_src
    assert guardrails[0].incumbent_ref_us == 7.5
    assert rounds[0].promoted is True
    assert rounds[0].l3_resolution == L3Resolution.PROMOTED.value
    assert rounds[0].stagnation_rounds_after == 0
    assert coord.pending_promotion is None
    assert coord.active_promotion is None
    saved = json.loads(ckpt_path.read_text(encoding="utf-8"))
    assert saved["pending_promotion"] is None
    assert saved["active_promotion"] is None


def test_resolve_pending_promotion_candidate_rejected_rolls_back_and_records_guardrail(
    tmp_path: Path,
) -> None:
    """Verify CANDIDATE_REJECTED restores incumbent bytes and ingests guardrails."""
    target_path = tmp_path / "kernels.py"
    ckpt_path = tmp_path / "ckpt.json"
    inc_src = "def _build():\n    return 1\n"
    cand_src = "def _build():\n    return 2\n"
    target_path.write_text(cand_src, encoding="utf-8")

    spec = RegimeTargetSpec(
        regime_tag="r0",
        builder_candidates=("_build",),
        sample_builder_kwargs={},
        representative_anchor_dims={"M": 64, "K": 127},
    )
    scoreboard = (
        AnchorCaseSpec("operator/attention_residual", 2, "r0", 10.0, 2.0),
    )

    def _l3_reject(
        module_path: Path,
        *,
        resume_execution_id: str = "",
        resume_run_id: str = "",
        device_id: str = "",
        on_execution_started=None,
    ) -> L3AcceptanceResult:
        _ = (module_path, resume_execution_id, resume_run_id, on_execution_started)
        return L3AcceptanceResult(
            passed=False,
            execution_id="exec-rej",
            run_id="run-rej",
            device_id=device_id,
            failure_kind=FAILURE_KIND_CANDIDATE,
            failure_category="numeric",
            error_message="numeric error case=2 case=14 case=15",
            failed_case_nums=(2, 14, 15),
            execution_terminal=True,
        )

    def _l2_probe(
        cands: tuple[tuple[str, Path], ...],
        specs: list[AnchorCaseSpec],
    ) -> tuple[dict[str, CandidateSurrogateResult], dict[str, str]]:
        metrics = tuple(
            _make_anchor_metric(s.case_num, s.regime_tag, cand_us=11.0 + s.case_num)
            for s in specs
        )
        return {cands[0][0]: aggregate_surrogate_result("inc", "attention_residual", metrics)}, {}

    coord = L3L4RecoveryCoordinator(
        operator_name="attention_residual",
        target_module_path=target_path,
        source_root=tmp_path,
        device_id="1",
        checkpoint_path=ckpt_path,
        l3_contract_runner=_l3_reject,
        l2_batch_runner=_l2_probe,
        lookup_case_dims_fn=lambda cnum, _tag: {"M": 64, "K": cnum},
        scoreboard_anchors=scoreboard,
        regime_specs=(spec,),
    )
    base_res = _make_surrogate_result("base", cand_us=9.0)
    cand_res = _make_surrogate_result("c1", cand_us=4.0)
    pending = PendingPromotion(
        candidate_source=cand_src,
        candidate_sha256=source_sha256(cand_src),
        target_path=str(target_path),
        round_idx=1,
        regime_tag="r0",
        candidate_result=cand_res,
        candidate_id="c1",
        execution_id="exec-rej",
        run_id="run-rej",
        device_id="1",
        execution_terminal=False,
    )
    rounds = [_make_round_record(round_idx=1, regime_tag="r0")]
    promoted_rounds: list[int] = []
    rolled_back_rounds: list[int] = []
    stagnation = {"r0": 1}

    new_inc, new_best, new_stag, next_pending, blocked = coord.resolve_pending_promotion(
        pending,
        incumbent_source=inc_src,
        best_result=base_res,
        stagnation_by_regime=stagnation,
        rounds=rounds,
        promoted_rounds=promoted_rounds,
        rolled_back_rounds=rolled_back_rounds,
        active_anchors=scoreboard,
    )
    assert blocked is False
    assert next_pending is None
    assert new_inc == inc_src
    assert new_best == base_res
    assert target_path.read_text(encoding="utf-8") == inc_src
    assert new_stag["r0"] == 2
    assert rolled_back_rounds == [1]
    assert rounds[0].promoted is False
    assert rounds[0].l3_resolution == L3Resolution.CANDIDATE_REJECTED.value
    assert rounds[0].stagnation_rounds_after == 2
    assert [g.case_num for g in coord.guardrail_cases] == [14, 15]
    assert coord.guardrail_cases[0].incumbent_ref_us == 25.0
    assert coord.guardrail_cases[1].incumbent_ref_us == 26.0
    assert coord.pending_promotion is None
    assert coord.active_promotion is None


def test_resolve_pending_promotion_infra_deferred_preserves_state_without_stagnation(
    tmp_path: Path,
) -> None:
    """Verify INFRA_DEFERRED preserves pending promotion with zero stagnation penalty."""
    target_path = tmp_path / "kernels.py"
    ckpt_path = tmp_path / "ckpt.json"
    inc_src = "def _build():\n    return 1\n"
    cand_src = "def _build():\n    return 2\n"
    target_path.write_text(cand_src, encoding="utf-8")

    def _l3_timeout(
        module_path: Path,
        *,
        resume_execution_id: str = "",
        resume_run_id: str = "",
        device_id: str = "",
        on_execution_started=None,
    ) -> L3AcceptanceResult:
        _ = (module_path, resume_run_id, on_execution_started)
        return L3AcceptanceResult(
            passed=False,
            execution_id=resume_execution_id,
            device_id=device_id,
            state="RUNNING",
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
            failure_category="timeout",
            error_message="wait timeout",
            execution_terminal=False,
        )

    coord = L3L4RecoveryCoordinator(
        operator_name="attention_residual",
        target_module_path=target_path,
        source_root=tmp_path,
        device_id="4",
        checkpoint_path=ckpt_path,
        l3_contract_runner=_l3_timeout,
    )
    base_res = _make_surrogate_result("base", cand_us=9.0)
    cand_res = _make_surrogate_result("c1", cand_us=4.0)
    pending = PendingPromotion(
        candidate_source=cand_src,
        candidate_sha256=source_sha256(cand_src),
        target_path=str(target_path),
        round_idx=1,
        regime_tag="r0",
        candidate_result=cand_res,
        candidate_id="c1",
        execution_id="exec-wait",
        device_id="4",
        execution_terminal=False,
    )
    rounds = [_make_round_record(round_idx=1, regime_tag="r0")]
    stagnation = {"r0": 1}

    new_inc, new_best, new_stag, next_pending, blocked = coord.resolve_pending_promotion(
        pending,
        incumbent_source=inc_src,
        best_result=base_res,
        stagnation_by_regime=stagnation,
        rounds=rounds,
        promoted_rounds=[],
        rolled_back_rounds=[],
        active_anchors=(),
    )
    assert blocked is True
    assert next_pending is not None
    assert next_pending.execution_id == "exec-wait"
    assert next_pending.execution_terminal is False
    assert new_inc == inc_src
    assert new_best == base_res
    assert new_stag["r0"] == 1
    assert rounds[0].l3_resolution == L3Resolution.INFRA_DEFERRED.value
    assert rounds[0].stagnation_rounds_after == 1
    assert target_path.read_text(encoding="utf-8") == cand_src
    assert coord.pending_promotion == next_pending
    assert coord.active_promotion == next_pending
    saved = json.loads(ckpt_path.read_text(encoding="utf-8"))
    assert saved["pending_promotion"]["execution_id"] == "exec-wait"
    assert saved["active_promotion"]["execution_id"] == "exec-wait"


def test_resolve_pending_promotion_rotates_l3_pool_on_terminal_infra(
    tmp_path: Path,
) -> None:
    """Verify terminal L3 infra failure rotates through untried pool cards."""
    target_path = tmp_path / "kernels.py"
    inc_src = "def _build():\n    return 1\n"
    cand_src = "def _build():\n    return 2\n"
    target_path.write_text(cand_src, encoding="utf-8")
    seen_devices: list[str] = []

    def _l3_runner(
        module_path: Path,
        *,
        resume_execution_id: str = "",
        resume_run_id: str = "",
        device_id: str = "",
        on_execution_started=None,
    ) -> L3AcceptanceResult:
        _ = (module_path, resume_execution_id, resume_run_id, on_execution_started)
        seen_devices.append(device_id)
        if device_id == "4":
            return L3AcceptanceResult(
                passed=True,
                execution_id="exec-4",
                run_id="run-4",
                device_id="4",
                execution_terminal=True,
            )
        return L3AcceptanceResult(
            passed=False,
            execution_id=f"exec-{device_id}",
            device_id=device_id,
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
            error_message="device unavailable",
            execution_terminal=True,
        )

    coord = L3L4RecoveryCoordinator(
        operator_name="attention_residual",
        target_module_path=target_path,
        source_root=tmp_path,
        device_id="3",
        device_ids=("4", "5"),
        l3_contract_runner=_l3_runner,
    )
    pending = PendingPromotion(
        candidate_source=cand_src,
        candidate_sha256=source_sha256(cand_src),
        target_path=str(target_path),
        round_idx=1,
        regime_tag="r0",
        candidate_result=_make_surrogate_result("c1"),
        attempts=({"device_id": "3", "failure_kind": "infrastructure"},),
        device_id="3",
        execution_terminal=True,
    )
    assert coord.can_retry_on_alt_card(pending, "3") is True
    assert coord.next_l3_alt_card(pending, "3") == "4"
    _inc, _best, _stag, next_pending, blocked = coord.resolve_pending_promotion(
        pending,
        incumbent_source=inc_src,
        best_result=_make_surrogate_result("base"),
        stagnation_by_regime={"r0": 0},
        rounds=[],
        promoted_rounds=[],
        rolled_back_rounds=[],
        active_anchors=(),
    )
    assert seen_devices == ["4"]
    assert next_pending is None
    assert blocked is False


def test_advance_l4_card_failure_occupancy_sweeps_pool_with_bounded_backoff(
    tmp_path: Path,
) -> None:
    """Verify occupancy rotates all pool cards first, then bounds backoff rounds."""
    target_path = tmp_path / "kernels.py"
    target_path.write_text("candidate", encoding="utf-8")
    calls: list[str] = []
    sleeps: list[float] = []

    def _l4_runner(
        module_path: Path,
        *,
        resume_execution_id: str = "",
        resume_run_id: str = "",
        device_id: str = "",
        on_execution_started=None,
    ) -> L4CannbenchResult:
        _ = (module_path, resume_execution_id, resume_run_id)
        calls.append(device_id)
        if on_execution_started is not None:
            on_execution_started(f"exec-{device_id}", device_id)
        if len(calls) == 4:
            return L4CannbenchResult(
                passed=True,
                device_id=device_id,
                execution_id="exec-win",
                execution_terminal=True,
            )
        return L4CannbenchResult(
            passed=False,
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
            blocked_env_category="occupancy",
            error_message="blocked_env occupancy",
            device_id=device_id,
            state="BLOCKED",
            execution_id=f"exec-blk-{device_id}",
            execution_terminal=True,
        )

    coord = L3L4RecoveryCoordinator(
        operator_name="attention_residual",
        target_module_path=target_path,
        source_root=tmp_path,
        device_id="0",
        device_ids=("1", "2"),
        l4_blocked_env_max_retries=2,
        l4_retry_backoff_sec=10.0,
        checkpoint_path=tmp_path / "ckpt.json",
        l4_cannbench_runner=_l4_runner,
        sleep_fn=sleeps.append,
    )
    res = coord.run_final_l4_tracked(
        incumbent_source="candidate",
        prior_stop_reason="goal_met",
    )
    assert res.passed is True
    assert res.device_id == "0"
    assert calls == ["0", "1", "2", "0"]
    assert sleeps == [10.0]
    assert coord.pending_l4 is None


def test_resolve_pending_l4_device_rejected_skips_and_excludes_refused_cards(
    tmp_path: Path,
) -> None:
    """Verify device_rejected advances past refused cards and halts if all refused."""
    target_path = tmp_path / "kernels.py"
    target_path.write_text("candidate", encoding="utf-8")
    digest = hashlib.sha256(target_path.read_bytes()).hexdigest()
    calls: list[str] = []
    sleeps: list[float] = []

    def _l4_runner(
        module_path: Path,
        *,
        resume_execution_id: str = "",
        resume_run_id: str = "",
        device_id: str = "",
        on_execution_started=None,
    ) -> L4CannbenchResult:
        _ = (module_path, resume_execution_id, resume_run_id, on_execution_started)
        calls.append(device_id)
        return L4CannbenchResult(
            passed=False,
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
            device_rejected=True,
            error_message=f"device {device_id} unauthorized",
            device_id=device_id,
            state="REJECTED",
            execution_terminal=True,
        )

    coord = L3L4RecoveryCoordinator(
        operator_name="attention_residual",
        target_module_path=target_path,
        source_root=tmp_path,
        device_id="0",
        device_ids=("1", "2"),
        l4_blocked_env_max_retries=3,
        l4_retry_backoff_sec=10.0,
        checkpoint_path=tmp_path / "ckpt.json",
        l4_cannbench_runner=_l4_runner,
        sleep_fn=sleeps.append,
    )
    pending = PendingL4(
        target_path=str(target_path),
        candidate_relative="kernels.py",
        candidate_sha256=digest,
        device_id="0",
        state="REJECTED",
        failure_kind=FAILURE_KIND_INFRASTRUCTURE,
        execution_terminal=True,
        device_rejected=True,
        rejected_device_ids=("0",),
        device_pool=("0", "1", "2"),
        blocked_env_round_attempted_device_ids=("0",),
        blocked_env_round_limit=3,
    )
    res, updated = coord.resolve_pending_l4(pending, incumbent_source="candidate")
    assert res.passed is False
    assert res.device_rejected is True
    assert calls == ["1", "2"]
    assert sleeps == []
    assert updated is not None
    assert updated.rejected_device_ids == ("0", "1", "2")
    assert updated.blocked_env_rounds_used == 0

    calls.clear()
    res_again, updated_again = coord.resolve_pending_l4(
        updated,
        incumbent_source="candidate",
    )
    assert res_again.passed is False
    assert calls == []
    assert updated_again == updated


def test_resolve_pending_l4_occupancy_interrupted_and_digest_guard(
    tmp_path: Path,
) -> None:
    """Verify pending L4 occupancy recovery handles resume and digest mismatch."""
    target_path = tmp_path / "kernels.py"
    target_path.write_text("candidate", encoding="utf-8")
    digest = hashlib.sha256(target_path.read_bytes()).hexdigest()
    calls: list[tuple[str, str, str]] = []

    def _l4_runner(
        module_path: Path,
        *,
        resume_execution_id: str = "",
        resume_run_id: str = "",
        device_id: str = "",
        on_execution_started=None,
    ) -> L4CannbenchResult:
        _ = module_path
        calls.append((resume_execution_id, resume_run_id, device_id))
        if on_execution_started is not None:
            on_execution_started(resume_execution_id or f"exec-{device_id}", device_id)
        if device_id == "1":
            return L4CannbenchResult(
                passed=True,
                device_id="1",
                execution_id="exec-1",
                execution_terminal=True,
            )
        return L4CannbenchResult(
            passed=False,
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
            blocked_env_category="occupancy",
            device_id=device_id,
            state="BLOCKED",
            execution_id=resume_execution_id or f"exec-{device_id}",
            execution_terminal=True,
        )

    coord = L3L4RecoveryCoordinator(
        operator_name="attention_residual",
        target_module_path=target_path,
        source_root=tmp_path,
        device_id="0",
        device_ids=("1",),
        l4_blocked_env_max_retries=1,
        l4_retry_backoff_sec=0.0,
        l4_cannbench_runner=_l4_runner,
    )
    pending = PendingL4(
        target_path=str(target_path),
        candidate_relative="kernels.py",
        candidate_sha256=digest,
        execution_id="exec-0",
        device_id="0",
        state="RUNNING",
        execution_terminal=False,
        device_pool=("0", "1"),
        blocked_env_round_attempted_device_ids=("0",),
        blocked_env_round_limit=1,
    )
    res, updated = coord.resolve_pending_l4(pending, incumbent_source="candidate")
    assert res.passed is True
    assert updated is None
    assert calls == [("exec-0", "", "0"), ("", "", "1")]

    bad_pending = PendingL4(
        target_path=str(target_path),
        candidate_relative="kernels.py",
        candidate_sha256="0" * 64,
        execution_id="exec-bad",
        device_id="0",
    )
    bad_res, bad_updated = coord.resolve_pending_l4(
        bad_pending,
        incumbent_source="candidate",
    )
    assert bad_res.passed is False
    assert "does not match" in bad_res.error_message
    assert bad_updated is not None


def test_record_l3_guardrail_cases_monotonic_and_deduplicated(tmp_path: Path) -> None:
    """Verify failed L3 cases ingest monotonically without duplicates or scoreboard."""
    lake_dir = tmp_path / "lake"
    scoreboard = (
        AnchorCaseSpec("operator/attention_residual", 2, "r0", 10.0, 2.0),
    )
    spec = RegimeTargetSpec(
        regime_tag="r0",
        builder_candidates=("_build",),
        sample_builder_kwargs={},
    )
    probe_calls: list[list[int]] = []

    def _l2_probe(
        cands: tuple[tuple[str, Path], ...],
        specs: list[AnchorCaseSpec],
    ) -> tuple[dict[str, CandidateSurrogateResult], dict[str, str]]:
        probe_calls.append([s.case_num for s in specs])
        metrics = tuple(
            _make_anchor_metric(s.case_num, s.regime_tag, cand_us=8.0 + s.case_num)
            for s in specs
        )
        return {cands[0][0]: aggregate_surrogate_result("inc", "attention_residual", metrics)}, {}

    coord = L3L4RecoveryCoordinator(
        operator_name="attention_residual",
        target_module_path=tmp_path / "kernels.py",
        source_root=tmp_path,
        experience_lake_dir=lake_dir,
        run_id="run-guardrail",
        l2_batch_runner=_l2_probe,
        lookup_case_dims_fn=lambda cnum, _tag: {"M": 64, "K": 127 if cnum == 14 else 256},
        scoreboard_anchors=scoreboard,
        regime_specs=(spec,),
        topology_axes=TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd"),
    )

    first_res = L3AcceptanceResult(
        passed=False,
        failure_kind=FAILURE_KIND_CANDIDATE,
        failed_case_nums=(2, 14, 14, 15),
        failure_stage="numeric",
    )
    added_1 = coord.record_l3_guardrail_cases(first_res, "def _build():\n    return 1\n", spec)
    assert [g.case_num for g in added_1] == [14, 15]
    assert [g.case_num for g in coord.guardrail_cases] == [14, 15]

    second_res = L3AcceptanceResult(
        passed=False,
        failure_kind=FAILURE_KIND_CANDIDATE,
        error_message="mismatch on case=14 case=15 case=16",
    )
    added_2 = coord.record_l3_guardrail_cases(second_res, "def _build():\n    return 1\n", spec)
    assert [g.case_num for g in added_2] == [16]
    assert [g.case_num for g in coord.guardrail_cases] == [14, 15, 16]
    assert probe_calls == [[14, 15], [16]]

    added_3 = coord.record_l3_guardrail_cases(second_res, "def _build():\n    return 1\n", spec)
    assert added_3 == []
    assert [g.case_num for g in coord.guardrail_cases] == [14, 15, 16]


def test_guardrail_helpers_knob_signature_and_alignment_flags() -> None:
    """Verify failed-case extraction, prime/alignment flags, and knob signatures."""
    assert extract_l3_failed_case_nums(
        L3AcceptanceResult(passed=False, error_message="err case=9 case=3 case=9")
    ) == [9, 3]
    assert is_prime_gt2(2) is False
    assert is_prime_gt2(3) is True
    assert is_prime_gt2(9) is False
    assert is_prime_gt2(127) is True

    flags = case_dim_alignment_flags({"M": 64, "K": 127, "dtype": "float16", "batch": 3})
    assert flags["has_odd_dim"] is True
    assert flags["has_prime_dim"] is True
    assert flags["non_32b_aligned"] is True
    assert flags["is_unaligned"] is True

    spec = RegimeTargetSpec(
        regime_tag="r0",
        builder_candidates=("_build",),
        sample_builder_kwargs={},
        representative_anchor_dims={"M": 64, "K": 128},
    )

    def _fake_resolve(
        _src: str,
        _spec: RegimeTargetSpec,
        _dims: object,
    ) -> ResolvedRegimeConfig:
        return ResolvedRegimeConfig(
            builder_name="_build",
            builder_kwargs={"M": 64, "block_m": 32, "threads": 128},
            pass_configs={"vec_pipe": True},
            compile_flags=("-O2", "-DNDEBUG"),
            selector_name="select_cfg",
        )

    sig = extract_knob_signature("def _build():\n    pass\n", spec, resolve_regime_config_fn=_fake_resolve)
    assert sig == "block_m=32,threads=128,pc.vec_pipe=True,flags=-DNDEBUG -O2"
