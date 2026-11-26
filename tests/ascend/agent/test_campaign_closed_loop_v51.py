"""Closed-loop integration and regression tests for campaign runner resilience and feedback."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.campaign import campaign_runner, surrogate_cannbench
from tilefoundry_ascend.agent.campaign.anchor_specs import (
    AnchorCaseSpec,
    PerfTargetSpec,
    RegimeTargetSpec,
)
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CampaignConfig,
    CampaignRoundRecord,
    PerfTargetCampaignRunner,
)
from tilefoundry_ascend.agent.campaign.funnel_telemetry import (
    DeterministicSearchBoundaryLedger,
)
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.kb.schema import AscendCausalRule, SymptomFamily
from tilefoundry_ascend.agent.orchestrator.gate_funnel import StageAResult, StageBResult
from tilefoundry_ascend.agent.orchestrator.llm_client import GenerationRequest
from tilefoundry_ascend.agent.router.capability_gate import route_execution_track
from tilefoundry_ascend.ascend.cost_model.experience_lake import (
    ExperienceLake,
    append_counterexample_boundary,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes


def _make_anchor_metric(
    *,
    case_num: int,
    regime_tag: str,
    cand_us: float,
    ref_perf_us: float = 10.0,
    t_hw_us: float = 2.0,
    accuracy_passed: bool = True,
    eq3_score: float = 0.8,
    ring_burst_k: int = 1,
    burst_batch_cv: float = 0.0,
    spike_retries: int = 0,
    card_ext_hbm_mb: int = 0,
) -> surrogate_cannbench.AnchorEvalMetrics:
    speedup = (ref_perf_us / cand_us) if (accuracy_passed and cand_us > 0.0) else 0.0
    eta = (t_hw_us / cand_us) if (accuracy_passed and cand_us > 0.0) else 0.0
    return surrogate_cannbench.AnchorEvalMetrics(
        case_num=case_num,
        regime_tag=regime_tag,
        accuracy_passed=accuracy_passed,
        cand_us=cand_us,
        ref_perf_us=ref_perf_us,
        t_hw_us=t_hw_us,
        speedup=speedup,
        roofline_eta=eta,
        eq3_score=eq3_score if accuracy_passed else 0.0,
        ring_burst_k=ring_burst_k,
        burst_batch_cv=burst_batch_cv,
        spike_retries=spike_retries,
        card_ext_hbm_mb=card_ext_hbm_mb,
    )


def _make_surrogate_result(
    candidate_id: str,
    operator_name: str,
    metrics: Sequence[surrogate_cannbench.AnchorEvalMetrics],
) -> surrogate_cannbench.CandidateSurrogateResult:
    return surrogate_cannbench.aggregate_surrogate_result(
        candidate_id,
        operator_name,
        tuple(metrics),
    )


def _make_ledger(
    *,
    steady_state_ii: int = 20,
    gm_roundtrip_bytes: int = 4096,
    grid_blocks: int = 16,
) -> object:
    return build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=steady_state_ii,
        pipe_cycles={"AIV0.PIPE_V": steady_state_ii},
        waterlines={},
        gm_roundtrip_bytes=gm_roundtrip_bytes,
        grid_blocks=grid_blocks,
        wave_occupancy=0.5,
    )


def _patch_prompt_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(campaign_runner, "scan_tilelang_pipe_gaps", lambda: None)
    from tilefoundry_ascend.agent.orchestrator import llm_constraints

    monkeypatch.setattr(llm_constraints, "format_target_dialect_layer", lambda *_a, **_kw: "")


def _basic_spec(
    tag: str = "small_decode",
    *,
    dims: dict[str, int] | None = None,
) -> RegimeTargetSpec:
    rep_dims = dict(dims) if dims is not None else {"N": 2, "B": 1, "S": 1, "D": 128}
    return RegimeTargetSpec(
        regime_tag=tag,
        builder_candidates=("_build_op",),
        sample_builder_kwargs={"tile_m": 4},
        editable_symbols=("_select_op_config", "_build_op"),
        config_selector_fn="_select_op_config",
        representative_anchor_dims=rep_dims,
        representative_hir_dims=dict(rep_dims),
    )


_BASE_MODULE_SRC = '''
def _select_op_config(**_kw):
    return {"tile_m": 4}

def _build_op(tile_m=4):
    return tile_m

def kernel(x):
    return x
'''


def test_round_record_closed_loop_fields_roundtrip() -> None:
    """Verify new CampaignRoundRecord fields serialize and deserialize cleanly."""
    rec = CampaignRoundRecord(
        round_idx=2,
        regime_tag="small_decode",
        builder_name="_build_op",
        parent_lane="exploit",
        creative_effort="high",
        l0_l1_passed_ids=("c1",),
        l0_l1_failed=(),
        best_round_speedup=1.5,
        best_so_far_speedup=1.5,
        promoted=True,
        l3_resolution="PROMOTED",
        stagnation_rounds_after=0,
        round_spearman_rho=0.8125,
        prescreen_pruned_ids=("c2",),
        l3_failed_cases=(14,),
        l3_failure_stage="accuracy",
        l3_cx_repair_passed=True,
        infeasible_boundaries_added=2,
    )
    encoded = rec.to_dict()
    assert encoded["round_spearman_rho"] == pytest.approx(0.8125)
    assert encoded["prescreen_pruned_ids"] == ["c2"]
    assert encoded["l3_failed_cases"] == [14]
    assert encoded["l3_failure_stage"] == "accuracy"
    assert encoded["l3_cx_repair_passed"] is True
    assert encoded["infeasible_boundaries_added"] == 2

    decoded = CampaignRoundRecord.from_dict(encoded)
    assert decoded.round_spearman_rho == pytest.approx(0.8125)
    assert decoded.prescreen_pruned_ids == ("c2",)
    assert decoded.l3_failed_cases == (14,)
    assert decoded.l3_failure_stage == "accuracy"
    assert decoded.l3_cx_repair_passed is True
    assert decoded.infeasible_boundaries_added == 2


@pytest.mark.anyio
async def test_p0_1a_l3_counterexample_reaggregates_best_result_and_avoids_false_tail_reject(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify slow L3 counterexample re-aggregates best_result before repair tail guard."""
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_BASE_MODULE_SRC, encoding="utf-8")
    spec = _basic_spec("ragged_gmm", dims={"M": 64, "K": 256, "N": 128})
    anchors = [AnchorCaseSpec("level4/grouped_matmul", 2, "ragged_gmm", 10.0, 2.0)]

    fast_cand = '''
def _select_op_config(**_kw):
    return {"tile_m": 8}

def _build_op(tile_m=8):
    return tile_m + 1
'''
    repaired_cand = '''
def _select_op_config(**_kw):
    return {"tile_m": 8}

def _build_op(tile_m=8):
    # tail_fixed
    return tile_m + 2
'''

    class _RepairLLM:
        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            _ = effort_override
            if req.role_hint == "l3_counterexample_repair":
                return repaired_cand
            if req.role == "explore_creative":
                return json.dumps(
                    {
                        "physical_hypothesis": "Increase tile_m",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": fast_cand,
                    }
                )
            return fast_cand

    def _l2_eval(
        cands: Sequence[tuple[str, Path]],
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> dict[str, surrogate_cannbench.CandidateSurrogateResult]:
        out: dict[str, surrogate_cannbench.CandidateSurrogateResult] = {}
        for cid, mod_path in cands:
            src = mod_path.read_text(encoding="utf-8")
            is_fixed = "tail_fixed" in src
            is_fast = "tile_m=8" in src
            metrics: list[surrogate_cannbench.AnchorEvalMetrics] = []
            for anc in active_anchors:
                if int(anc.case_num) == 14:
                    cand_us = 18.0 if is_fixed else 20.0
                else:
                    cand_us = 3.0 if is_fast else 4.0
                metrics.append(
                    _make_anchor_metric(
                        case_num=anc.case_num,
                        regime_tag=anc.regime_tag,
                        cand_us=cand_us,
                        ref_perf_us=anc.ref_perf_us,
                        t_hw_us=anc.t_hw_us,
                        eq3_score=0.9 if (is_fast or is_fixed) else 0.5,
                    )
                )
            out[cid] = _make_surrogate_result(cid, "grouped_matmul", metrics)
        return out

    def _l3_runner(mod_path: Path, **_kw: object) -> surrogate_cannbench.L3AcceptanceResult:
        src = mod_path.read_text(encoding="utf-8")
        if "tail_fixed" in src:
            return surrogate_cannbench.L3AcceptanceResult(
                passed=True,
                run_id="l3-fixed",
                execution_id="exec-fixed",
                device_id="1",
            )
        return surrogate_cannbench.L3AcceptanceResult(
            passed=False,
            error_message="NUMERIC_MISMATCH case=14",
            failure_kind=surrogate_cannbench.FAILURE_KIND_CANDIDATE,
            failure_category="numeric",
            failure_stage="accuracy",
            failed_case_nums=(14,),
            execution_id="exec-fail",
            run_id="l3-fail",
            device_id="1",
        )

    ledger = _make_ledger()
    route = route_execution_track(None, ledger=ledger)
    lake_dir = tmp_path / "lake"
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="grouped_matmul",
            max_rounds=1,
            k_rule_workers=0,
            enable_tilesize_prescreen=False,
            min_surrogate_gain=0.01,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("grouped_matmul", 3.0, 2.0, 0.90),
        anchors=anchors,
        regime_specs=(spec,),
        llm_backend=_RepairLLM(),
        tracker=BayesianRuleTracker(rules={}),
        l0_checker=lambda _s, _r: StageAResult(
            passed=True,
            prim_func=None,
            ledger=ledger,
            features=route.features,
            route=route,
        ),
        l1_compiler=lambda _s, _r: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=_l3_runner,
        checkpoint_path=tmp_path / "ckpt.json",
        experience_lake_dir=lake_dir,
    )
    _patch_prompt_env(monkeypatch)
    monkeypatch.setattr(runner, "_collect_msprof_for_regimes", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        runner,
        "_lookup_case_dims_safe",
        lambda cnum, _tag="": {"B": 1, "S": 1, "D": 127} if int(cnum) == 14 else {"B": 1, "S": 1, "D": 128},
    )
    monkeypatch.setattr(
        campaign_runner,
        "lookup_case_metadata",
        lambda *_a, **_kw: (20.0, 2.0),
    )

    best_result = _make_surrogate_result(
        "base",
        "grouped_matmul",
        [_make_anchor_metric(case_num=2, regime_tag="ragged_gmm", cand_us=4.0, ref_perf_us=10.0)],
    )
    rec, new_inc, new_best, _stag, promoted, rolled_back, _esc, blocked = (
        await runner._execute_single_round(
            round_idx=1,
            regime_spec=spec,
            incumbent_source=_BASE_MODULE_SRC,
            best_result=best_result,
            active_anchors=anchors,
            stagnation_rounds=0,
            admitted_rule_ids=[],
        )
    )

    assert [a.case_num for a in anchors] == [2]
    assert [g.case_num for g in runner._guardrail_cases] == [14]
    assert promoted is True
    assert rolled_back is False
    assert blocked is False
    assert "tail_fixed" in new_inc
    assert {int(a.case_num) for a in new_best.anchors} == {2}
    assert rec.l3_failed_cases == (14,)
    assert rec.l3_failure_stage == "accuracy"
    assert rec.l3_cx_repair_passed is True
    assert rec.tail_guard_reject_reason == ""

    cx_path = lake_dir / "counterexample_anchors.jsonl"
    assert cx_path.is_file()
    cx_records = [json.loads(line) for line in cx_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(cx_records) == 1
    assert cx_records[0]["case_num"] == 14
    assert cx_records[0]["failure_stage"] == "accuracy"
    assert cx_records[0]["has_odd_dim"] is True
    assert cx_records[0]["has_prime_dim"] is True
    assert cx_records[0]["non_32b_aligned"] is True


@pytest.mark.anyio
async def test_p0_1b_non_terminal_repaired_l3_preserves_target_and_saves_active_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify non-terminal repaired_l3 preserves target file and persists active_promotion."""
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_BASE_MODULE_SRC, encoding="utf-8")
    ckpt_path = tmp_path / "ckpt.json"
    spec = _basic_spec("ragged_gmm", dims={"M": 64, "K": 256, "N": 128})
    anchors = (AnchorCaseSpec("level4/grouped_matmul", 2, "ragged_gmm", 10.0, 2.0),)

    fast_cand = '''
def _select_op_config(**_kw):
    return {"tile_m": 8}

def _build_op(tile_m=8):
    return tile_m + 1
'''
    repaired_cand = '''
def _select_op_config(**_kw):
    return {"tile_m": 8}

def _build_op(tile_m=8):
    # repaired_pending_tag
    return tile_m + 2
'''

    class _RepairLLM:
        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            _ = effort_override
            if req.role_hint == "l3_counterexample_repair":
                return repaired_cand
            if req.role == "explore_creative":
                return json.dumps(
                    {
                        "physical_hypothesis": "Increase tile_m",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": fast_cand,
                    }
                )
            return fast_cand

    def _l2_eval(
        cands: Sequence[tuple[str, Path]],
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> dict[str, surrogate_cannbench.CandidateSurrogateResult]:
        out: dict[str, surrogate_cannbench.CandidateSurrogateResult] = {}
        for cid, mod_path in cands:
            src = mod_path.read_text(encoding="utf-8")
            cand_us = 3.0 if "tile_m=8" in src else 6.0
            metrics = [
                _make_anchor_metric(
                    case_num=anc.case_num,
                    regime_tag=anc.regime_tag,
                    cand_us=cand_us,
                    ref_perf_us=anc.ref_perf_us,
                    t_hw_us=anc.t_hw_us,
                    eq3_score=0.85 if "tile_m=8" in src else 0.4,
                )
                for anc in active_anchors
            ]
            out[cid] = _make_surrogate_result(cid, "grouped_matmul", metrics)
        return out

    def _l3_runner(
        mod_path: Path,
        *,
        on_execution_started: object = None,
        **_kw: object,
    ) -> surrogate_cannbench.L3AcceptanceResult:
        src = mod_path.read_text(encoding="utf-8")
        if callable(on_execution_started):
            on_execution_started("exec-live-99", "1")
        if "repaired_pending_tag" in src:
            return surrogate_cannbench.L3AcceptanceResult(
                passed=False,
                failure_kind=surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE,
                execution_terminal=False,
                pending=True,
                execution_id="exec-live-99",
                run_id="run-live-99",
                device_id="1",
                error_message="wait_timeout_still_running",
            )
        return surrogate_cannbench.L3AcceptanceResult(
            passed=False,
            failure_kind=surrogate_cannbench.FAILURE_KIND_CANDIDATE,
            failed_case_nums=(14,),
            failure_stage="accuracy",
            error_message="numeric mismatch case=14",
            execution_id="exec-cx-14",
            run_id="run-cx-14",
            device_id="1",
        )

    ledger = _make_ledger()
    route = route_execution_track(None, ledger=ledger)
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="grouped_matmul",
            max_rounds=1,
            k_rule_workers=0,
            enable_tilesize_prescreen=False,
            min_surrogate_gain=0.01,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("grouped_matmul", 3.0, 2.0, 0.90),
        anchors=anchors,
        regime_specs=(spec,),
        llm_backend=_RepairLLM(),
        tracker=BayesianRuleTracker(rules={}),
        l0_checker=lambda _s, _r: StageAResult(
            passed=True,
            prim_func=None,
            ledger=ledger,
            features=route.features,
            route=route,
        ),
        l1_compiler=lambda _s, _r: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=_l3_runner,
        checkpoint_path=ckpt_path,
    )
    _patch_prompt_env(monkeypatch)
    monkeypatch.setattr(runner, "_collect_msprof_for_regimes", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        campaign_runner,
        "lookup_case_metadata",
        lambda *_a, **_kw: (12.0, 2.0),
    )

    report = await runner.run(resume=False)

    assert report.stop_reason == "promotion_blocked"
    assert report.promotion_blocked is True
    assert report.rolled_back_rounds == ()
    disk_source = target_path.read_text(encoding="utf-8")
    assert "repaired_pending_tag" in disk_source

    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
    assert ckpt["active_promotion"] is not None
    assert ckpt["pending_promotion"] is not None
    assert ckpt["active_promotion"]["execution_terminal"] is False
    assert ckpt["active_promotion"]["execution_id"] == "exec-live-99"
    assert "repaired_pending_tag" in ckpt["active_promotion"]["candidate_source"]


def test_p1_1_regime_filtered_max_row_width_and_strict_vf_stack_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify small regime ignores sibling large dims and non-stack errors are not vf_stack."""
    target_path = tmp_path / "kernels.py"
    trackb_src = '''
_SMALL_CPP = """
__global__ void small_k() {}
"""

def _select_small_config(**_kw):
    return {"compile_flags": ["--cce-disable-vf-stack-reserved-ubuf"]}

def _build_small():
    return 1

def _select_large_config(**_kw):
    return {}

def _build_large():
    return 2
'''
    target_path.write_text(trackb_src, encoding="utf-8")
    small_spec = RegimeTargetSpec(
        regime_tag="small_decode",
        builder_candidates=("_build_small",),
        sample_builder_kwargs={},
        representative_anchor_dims={"N": 2, "B": 1, "S": 1, "D": 128},
        representative_hir_dims={"N": 2, "B": 1, "S": 1, "D": 128},
        config_selector_fn="_select_small_config",
        editable_symbols=("_select_small_config", "_build_small", "_SMALL_CPP"),
        runtime_backend="ascendc_trackb",
        trackb_symbols=("_SMALL_CPP",),
    )
    large_spec = RegimeTargetSpec(
        regime_tag="aligned_prefill",
        builder_candidates=("_build_large",),
        sample_builder_kwargs={},
        representative_anchor_dims={"N": 8, "B": 1, "S": 4096, "D": 6144},
        representative_hir_dims={"N": 8, "B": 1, "S": 4096, "D": 6144},
        config_selector_fn="_select_large_config",
        editable_symbols=("_select_large_config", "_build_large"),
    )
    anchors = (
        AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 5.0, 1.0),
        AnchorCaseSpec("level4/attention_residual", 16, "aligned_prefill", 500.0, 100.0),
    )
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="attention_residual", max_rounds=1),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 2.0, 1.5, 0.85),
        anchors=anchors,
        regime_specs=(small_spec, large_spec),
        checkpoint_path=tmp_path / "ckpt.json",
    )
    monkeypatch.setattr(runner, "_resolve_cannbench_tasks_root", lambda: None)

    def _lookup_dims(case_num: int, _tag: str = "") -> dict[str, object]:
        if int(case_num) == 16:
            return {"N": 8, "B": 1, "S": 4096, "D": 6144}
        return {"N": 2, "B": 1, "S": 1, "D": 128}

    monkeypatch.setattr(runner, "_lookup_case_dims_safe", _lookup_dims)
    monkeypatch.setattr(
        campaign_runner,
        "resolve_regime_for_case",
        lambda _op, dims, _specs: (
            large_spec if int(dims.get("D", 0) or 0) >= 4096 else small_spec
        ),
    )

    resolved_small = runner._resolve_max_row_width_dims(small_spec)
    assert resolved_small == {"N": 2, "B": 1, "S": 1, "D": 128}

    non_stack_fail = StageBResult(
        passed=False,
        error_message="error: undeclared identifier in call_stack_helper",
        full_diagnostic="fatal error: invalid argument in stack_alloc_stub",
    )
    assert runner._has_vf_stack_overflow(non_stack_fail) is False

    stack_fail = StageBResult(
        passed=False,
        error_message="bisheng error: exceeded vf stack size",
        full_diagnostic="",
    )
    assert runner._has_vf_stack_overflow(stack_fail) is True

    compile_calls: list[str] = []

    def _fake_compile_trackb(tu_src: str, **_kw: object) -> StageBResult:
        compile_calls.append(tu_src)
        if len(compile_calls) == 1:
            return StageBResult(passed=True)
        return non_stack_fail

    monkeypatch.setattr(runner, "_compile_trackb_stage_b", _fake_compile_trackb)
    monkeypatch.setattr(
        runner,
        "_resolve_max_row_width_dims",
        lambda _reg: {"N": 2, "B": 1, "S": 2, "D": 256},
    )
    monkeypatch.setattr(
        runner,
        "_instantiate_trackb_tu",
        lambda _src, cpp, spec: f"// D={spec.representative_anchor_dims.get('D')}\n{cpp}",
    )

    stage_b = runner._run_l1(trackb_src, small_spec)
    assert stage_b.passed is False
    assert stage_b.error_message.startswith("stage_b_max_row_width_compile_failed:")
    assert "vf_stack_overflow" not in stage_b.error_message


@pytest.mark.anyio
async def test_p1_2_non_associative_recurrent_aiv_does_not_trigger_paradigm_shift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify has_recurrent_loop alone does not trigger paradigm_shift_active."""
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_BASE_MODULE_SRC, encoding="utf-8")
    spec = _basic_spec("small_decode")
    anchors = (AnchorCaseSpec("level2/rms_norm", 0, "small_decode", 10.0, 1.0),)

    class _SpyLLM:
        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            _ = effort_override
            if req.role == "explore_creative":
                return json.dumps(
                    {
                        "physical_hypothesis": "Tune vector tile",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": _BASE_MODULE_SRC,
                    }
                )
            return _BASE_MODULE_SRC

    ledger = _make_ledger()
    route = route_execution_track(None, ledger=ledger)
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="rms_norm",
            max_rounds=2,
            k_rule_workers=0,
            enable_tilesize_prescreen=False,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("rms_norm", 2.0, 1.5, 0.85),
        anchors=anchors,
        regime_specs=(spec,),
        llm_backend=_SpyLLM(),
        tracker=BayesianRuleTracker(rules={}),
        l0_checker=lambda _s, _r: StageAResult(
            passed=True,
            prim_func=None,
            ledger=ledger,
            features=route.features,
            route=route,
        ),
        l1_compiler=lambda _s, _r: StageBResult(passed=True),
        l2_batch_evaluator=lambda *_a: {},
        checkpoint_path=tmp_path / "ckpt.json",
    )
    _patch_prompt_env(monkeypatch)
    monkeypatch.setattr(runner, "_collect_msprof_for_regimes", lambda *_a, **_kw: None)

    async def _empty_filter(*_a: object, **_kw: object) -> list[surrogate_cannbench.L0L1CandidateCheck]:
        return []

    monkeypatch.setattr(campaign_runner, "run_l0_l1_round_filter", _empty_filter)

    class _NonAssociativeRecurrent:
        has_recurrent_loop = True
        has_associative_outer_recurrence = False

    monkeypatch.setattr(
        runner,
        "_resolve_operator_semantics_for_gates",
        lambda: _NonAssociativeRecurrent(),
    )

    low_eta_best = _make_surrogate_result(
        "base",
        "rms_norm",
        [_make_anchor_metric(case_num=0, regime_tag="small_decode", cand_us=10.0, ref_perf_us=10.0, t_hw_us=0.5)],
    )
    rec, *_rest = await runner._execute_single_round(
        round_idx=3,
        regime_spec=spec,
        incumbent_source=_BASE_MODULE_SRC,
        best_result=low_eta_best,
        active_anchors=anchors,
        stagnation_rounds=2,
        admitted_rule_ids=[],
    )
    assert rec.paradigm_shift_active is False

    class _AssociativeRecurrent:
        has_recurrent_loop = True
        has_associative_outer_recurrence = True

    monkeypatch.setattr(
        runner,
        "_resolve_operator_semantics_for_gates",
        lambda: _AssociativeRecurrent(),
    )
    rec_assoc, *_rest2 = await runner._execute_single_round(
        round_idx=4,
        regime_spec=spec,
        incumbent_source=_BASE_MODULE_SRC,
        best_result=low_eta_best,
        active_anchors=anchors,
        stagnation_rounds=2,
        admitted_rule_ids=[],
    )
    assert rec_assoc.paradigm_shift_active is True


@pytest.mark.anyio
async def test_arc1_vf_stack_sha_short_circuits_stage_a_without_l1_and_persists_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify known VF_STACK_OVERFLOW SHA is blocked at Stage-A in 0ms without invoking L1."""
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_BASE_MODULE_SRC, encoding="utf-8")
    ckpt_path = tmp_path / "ckpt.json"
    spec = _basic_spec("small_decode")
    anchors = (AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 10.0, 2.0),)

    bad_cand = '''
def _select_op_config(**_kw):
    return {"tile_m": 256, "compile_flags": ["--cce-disable-vf-stack-reserved-ubuf"]}

def _build_op(tile_m=256):
    return tile_m
'''
    captured_requests: list[GenerationRequest] = []

    class _RepeatedBadLLM:
        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            _ = effort_override
            captured_requests.append(req)
            if req.role == "explore_creative":
                return json.dumps(
                    {
                        "physical_hypothesis": "Try large tile_m",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": bad_cand,
                    }
                )
            return bad_cand

    l1_invocations: list[str] = []
    ledger = _make_ledger()
    route = route_execution_track(None, ledger=ledger)

    def _l1_fail_vf_stack(src: str, _reg: RegimeTargetSpec) -> StageBResult:
        l1_invocations.append(src)
        return StageBResult(
            passed=False,
            error_message="stage_b_max_row_width_vf_stack_overflow: stack_bytes=4096 (exceeded vf stack size)",
            full_diagnostic="exceeded vf stack size",
            fault_plane="STAGE_B_COMPILE",
            fault_code="VF_STACK_OVERFLOW",
            failed_tier="stage_b",
        )

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            max_rounds=2,
            max_syntax_repairs=0,
            k_rule_workers=0,
            enable_tilesize_prescreen=False,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 3.0, 2.0, 0.90),
        anchors=anchors,
        regime_specs=(spec,),
        llm_backend=_RepeatedBadLLM(),
        tracker=BayesianRuleTracker(rules={}),
        l0_checker=lambda _s, _r: StageAResult(
            passed=True,
            prim_func=None,
            ledger=ledger,
            features=route.features,
            route=route,
        ),
        l1_compiler=_l1_fail_vf_stack,
        l2_batch_evaluator=lambda cands, anc: {
            cid: _make_surrogate_result(
                cid,
                "attention_residual",
                [_make_anchor_metric(case_num=a.case_num, regime_tag=a.regime_tag, cand_us=8.0) for a in anc],
            )
            for cid, _p in cands
        },
        checkpoint_path=ckpt_path,
    )
    _patch_prompt_env(monkeypatch)
    monkeypatch.setattr(runner, "_collect_msprof_for_regimes", lambda *_a, **_kw: None)

    report = await runner.run(resume=False)

    assert len(report.rounds) == 2
    cand_l1_calls = [src for src in l1_invocations if "tile_m=256" in src]
    assert len(cand_l1_calls) == 1
    assert report.rounds[0].infeasible_boundaries_added >= 1
    assert report.rounds[1].infeasible_boundaries_added == 0
    assert any(
        "deterministic_boundary_" in err
        for _cid, _tier, err in report.rounds[1].l0_l1_failed
    )
    creative_r2 = [req for req in captured_requests if req.role == "explore_creative"][1]
    assert "VERIFIED PHYSICAL SEARCH BOUNDARIES" in creative_r2.user_prompt

    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
    restored_ledger = DeterministicSearchBoundaryLedger.from_dict(ckpt["search_boundary_ledger"])
    assert len(restored_ledger.rejected_cand_shas) >= 1


def test_arc2_and_arc3_clean_gating_anomaly_log_and_lake_seed_injection(
    tmp_path: Path,
) -> None:
    """Verify burst CV gates CostModel updates, logs anomalies, and seeds lake anchors."""
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_BASE_MODULE_SRC, encoding="utf-8")
    ckpt_path = tmp_path / "ckpt.json"
    lake_dir = tmp_path / "lake"
    spec = _basic_spec("small_decode")
    anchors = (AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 10.0, 2.0),)

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            max_rounds=1,
            cost_model_mode="advisory",
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 3.0, 2.0, 0.90),
        anchors=anchors,
        regime_specs=(spec,),
        checkpoint_path=ckpt_path,
        experience_lake_dir=lake_dir,
    )
    ledger = _make_ledger()

    noisy_res = _make_surrogate_result(
        "noisy",
        "attention_residual",
        [
            _make_anchor_metric(
                case_num=2,
                regime_tag="small_decode",
                cand_us=9.0,
                ring_burst_k=5,
                burst_batch_cv=0.22,
                spike_retries=1,
                card_ext_hbm_mb=2048,
            )
        ],
    )
    runner._record_surrogate_observations(
        source=_BASE_MODULE_SRC,
        result=noisy_res,
        active_anchors=anchors,
        ledger=ledger,
        regime_tag="small_decode",
    )
    cost_model = runner._cost_advisor.underlying_cost_model
    assert cost_model is not None
    assert len(cost_model._samples) == 0

    lake = ExperienceLake(lake_dir)
    _cands, observations = lake.load_records()
    assert len(observations) == 1
    assert observations[0].burst_batch_cv == pytest.approx(0.22)
    assert observations[0].ring_burst_k == 5
    assert observations[0].spike_retries == 1
    assert observations[0].card_ext_hbm_mb == 2048
    assert observations[0].analytical_pred_us > 0.0

    clean_res = _make_surrogate_result(
        "clean_outlier",
        "attention_residual",
        [
            _make_anchor_metric(
                case_num=2,
                regime_tag="small_decode",
                cand_us=50000.0,
                ring_burst_k=5,
                burst_batch_cv=0.02,
                spike_retries=0,
                card_ext_hbm_mb=512,
            )
        ],
    )
    runner._record_surrogate_observations(
        source="def _build_op(tile_m=4):\n    return tile_m + 10\n",
        result=clean_res,
        active_anchors=anchors,
        ledger=ledger,
        regime_tag="small_decode",
    )
    anomaly_file = ckpt_path.parent / "unmodeled_anomalies.jsonl"
    assert anomaly_file.is_file()
    anomaly_rows = [json.loads(line) for line in anomaly_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(anomaly_rows) == 1
    assert anomaly_rows[0]["case_num"] == 2
    assert anomaly_rows[0]["log_error"] > 1.2

    clean_normal = _make_surrogate_result(
        "clean_normal",
        "attention_residual",
        [
            _make_anchor_metric(
                case_num=2,
                regime_tag="small_decode",
                cand_us=max(0.5, observations[0].analytical_pred_us * 1.1),
                burst_batch_cv=0.03,
            )
        ],
    )
    runner._record_surrogate_observations(
        source="def _build_op(tile_m=4):\n    return tile_m + 20\n",
        result=clean_normal,
        active_anchors=anchors,
        ledger=ledger,
        regime_tag="small_decode",
    )
    assert len(cost_model._samples) == 1

    append_counterexample_boundary(
        lake_dir,
        {
            "operator": "attention_residual",
            "regime_tag": "unaligned_d",
            "case_num": 14,
            "case_dims": {"N": 8, "B": 1, "S": 16, "D": 1000},
            "failure_stage": "accuracy",
            "ref_perf_us": 67.0,
            "t_hw_us": 22.0,
            "has_unaligned_dim": True,
        },
    )
    seeded_runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="attention_residual", max_rounds=1),
        target_module_path=target_path,
        anchors=anchors,
        regime_specs=(spec,),
        checkpoint_path=tmp_path / "ckpt_seeded.json",
        experience_lake_dir=lake_dir,
    )
    assert [int(a.case_num) for a in seeded_runner._scoreboard_anchors] == [2]


@pytest.mark.anyio
async def test_arc3_spearman_quarantine_and_round_funnel_events_emitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify 2 consecutive rounds with Spearman rho < 0.2 quarantine CostGate pruning."""
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_BASE_MODULE_SRC, encoding="utf-8")
    ckpt_path = tmp_path / "ckpt.json"
    spec = _basic_spec("small_decode")
    anchors = (AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 10.0, 2.0),)

    rule0 = AscendCausalRule(
        rule_id="R_TEST_0",
        regime_guard="True",
        precondition="True",
        transform_template="T0",
        expected_postcondition="True",
        source_provenance="test",
    )
    rule1 = AscendCausalRule(
        rule_id="R_TEST_1",
        regime_guard="True",
        precondition="True",
        transform_template="T1",
        expected_postcondition="True",
        source_provenance="test",
    )

    class _MultiCandLLM:
        def __init__(self) -> None:
            self.call_idx = 0

        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            _ = effort_override
            self.call_idx += 1
            if req.role == "explore_creative":
                return json.dumps(
                    {
                        "physical_hypothesis": "Variant C",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": (
                            f"def _build_op(tile_m=4):\n    # marker_c\n    return tile_m + {10 + self.call_idx}\n"
                        ),
                    }
                )
            if "R_TEST_0" in req.user_prompt:
                return f"def _build_op(tile_m=4):\n    # marker_a\n    return tile_m + {100 + self.call_idx}\n"
            return f"def _build_op(tile_m=4):\n    # marker_b\n    return tile_m + {200 + self.call_idx}\n"

    def _l0_checker(src: str, _reg: RegimeTargetSpec) -> StageAResult:
        if "marker_a" in src:
            ii = 100
        elif "marker_b" in src:
            ii = 500
        elif "marker_c" in src:
            ii = 2000
        else:
            ii = 50
        ledger = _make_ledger(steady_state_ii=ii, gm_roundtrip_bytes=ii * 16)
        route = route_execution_track(None, ledger=ledger)
        return StageAResult(
            passed=True,
            prim_func=None,
            ledger=ledger,
            features=route.features,
            route=route,
        )

    def _l2_inverted(
        cands: Sequence[tuple[str, Path]],
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> dict[str, surrogate_cannbench.CandidateSurrogateResult]:
        out: dict[str, surrogate_cannbench.CandidateSurrogateResult] = {}
        for cid, mod_path in cands:
            src = mod_path.read_text(encoding="utf-8")
            if "marker_a" in src:
                lat = 20.0
            elif "marker_b" in src:
                lat = 10.0
            elif "marker_c" in src:
                lat = 5.0
            else:
                lat = 12.0
            out[cid] = _make_surrogate_result(
                cid,
                "attention_residual",
                [
                    _make_anchor_metric(
                        case_num=a.case_num,
                        regime_tag=a.regime_tag,
                        cand_us=lat,
                        ref_perf_us=a.ref_perf_us,
                        t_hw_us=a.t_hw_us,
                        burst_batch_cv=0.02,
                    )
                    for a in active_anchors
                ],
            )
        return out

    tracker = BayesianRuleTracker(rules={})
    tracker.admit_rule(rule0)
    tracker.admit_rule(rule1)
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            max_rounds=3,
            k_rule_workers=2,
            enable_tilesize_prescreen=True,
            cost_prescreen_top_k=None,
            min_surrogate_gain=999.0,
            cost_model_mode="advisory",
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 50.0, 50.0, 0.99),
        anchors=anchors,
        regime_specs=(spec,),
        llm_backend=_MultiCandLLM(),
        tracker=tracker,
        l0_checker=_l0_checker,
        l1_compiler=lambda _s, _r: StageBResult(passed=True),
        l2_batch_evaluator=_l2_inverted,
        l3_contract_runner=lambda _p: surrogate_cannbench.L3AcceptanceResult(passed=True),
        checkpoint_path=ckpt_path,
    )
    _patch_prompt_env(monkeypatch)
    monkeypatch.setattr(runner, "_collect_msprof_for_regimes", lambda *_a, **_kw: None)
    monkeypatch.setattr(runner._tracker, "rank_applicable_rules", lambda *_a, **_kw: [rule0, rule1])
    monkeypatch.setattr(
        campaign_runner,
        "rank_symptom_families",
        lambda _ctx: [(next(iter(SymptomFamily)), 1.0)],
    )

    seen_prescreen_top_k: list[int | None] = []
    orig_filter = campaign_runner.run_l0_l1_round_filter

    async def _spy_filter(candidates: Sequence[tuple[str, str]], **kwargs: object) -> list[surrogate_cannbench.L0L1CandidateCheck]:
        seen_prescreen_top_k.append(kwargs.get("cost_prescreen_top_k"))
        return await orig_filter(candidates, **kwargs)

    monkeypatch.setattr(campaign_runner, "run_l0_l1_round_filter", _spy_filter)

    report = await runner.run(resume=False)

    assert len(report.rounds) == 3
    assert report.rounds[0].round_spearman_rho is not None
    assert report.rounds[0].round_spearman_rho < 0.2
    assert report.rounds[1].round_spearman_rho is not None
    assert report.rounds[1].round_spearman_rho < 0.2
    assert report.rounds[2].cost_model_reason == "spearman_rank_inverted"
    assert seen_prescreen_top_k[0] is None
    assert seen_prescreen_top_k[2] is None

    funnel_path = ckpt_path.parent / "telemetry" / "round_funnel_events.jsonl"
    assert funnel_path.is_file()
    funnel_events = [json.loads(line) for line in funnel_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(funnel_events) >= 9
    assert all("candidate_id" in ev and "analytical_pred_us" in ev for ev in funnel_events)


def test_arc3_noisy_tail_regression_triggers_single_retest(tmp_path: Path) -> None:
    """Verify noisy active_tail_regression triggers 1 retest batch and admits clean result."""
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_BASE_MODULE_SRC, encoding="utf-8")
    spec = _basic_spec("small_decode")
    anchors = (AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 10.0, 2.0),)

    retest_calls: list[list[int]] = []

    def _retest_l2(
        cands: Sequence[tuple[str, Path]],
        req_anchors: Sequence[AnchorCaseSpec],
    ) -> dict[str, surrogate_cannbench.CandidateSurrogateResult]:
        retest_calls.append([int(a.case_num) for a in req_anchors])
        out: dict[str, surrogate_cannbench.CandidateSurrogateResult] = {}
        for cid, _p in cands:
            out[cid] = _make_surrogate_result(
                cid,
                "attention_residual",
                [
                    _make_anchor_metric(
                        case_num=a.case_num,
                        regime_tag=a.regime_tag,
                        cand_us=3.0,
                        ref_perf_us=a.ref_perf_us,
                        t_hw_us=a.t_hw_us,
                        eq3_score=0.95,
                        burst_batch_cv=0.02,
                    )
                    for a in req_anchors
                ],
            )
        return out

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            max_rounds=1,
            min_surrogate_gain=0.05,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 3.0, 2.0, 0.90),
        anchors=anchors,
        regime_specs=(spec,),
        l2_batch_evaluator=_retest_l2,
        checkpoint_path=tmp_path / "ckpt.json",
    )

    best_res = _make_surrogate_result(
        "base",
        "attention_residual",
        [_make_anchor_metric(case_num=2, regime_tag="small_decode", cand_us=5.0, ref_perf_us=10.0, eq3_score=0.5)],
    )
    noisy_top_res = _make_surrogate_result(
        "cand",
        "attention_residual",
        [
            _make_anchor_metric(
                case_num=2,
                regime_tag="small_decode",
                cand_us=8.0,
                ref_perf_us=10.0,
                eq3_score=0.2,
                burst_batch_cv=0.18,
            )
        ],
    )
    ok, updated_res = runner._evaluate_dual_layer_tail_guard(
        candidate_id="cand",
        candidate_source="def _build_op(tile_m=8):\n    return tile_m\n",
        top_res=noisy_top_res,
        best_result=best_res,
        regime_spec=spec,
        active_anchors=anchors,
        incumbent_source=_BASE_MODULE_SRC,
    )
    assert retest_calls == [[2]]
    assert ok is True
    assert updated_res.anchors[0].cand_us == pytest.approx(3.0)
    assert updated_res.anchors[0].burst_batch_cv == pytest.approx(0.02)
