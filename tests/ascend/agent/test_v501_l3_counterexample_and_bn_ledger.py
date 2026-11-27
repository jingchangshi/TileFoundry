"""Tests for v5.0.1 bottleneck anchor ledger, L3 counterexample repair, and device pool."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from tilefoundry_ascend.agent.campaign import (
    campaign_runner,
    l3_l4_recovery,
    round_context_builder,
    stage_gate_coordinator,
    surrogate_cannbench,
)
from tilefoundry_ascend.agent.specs.anchor_specs import (
    AnchorCaseSpec,
    PerfTargetSpec,
    RegimeTargetSpec,
)
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CampaignConfig,
    PerfTargetCampaignRunner,
)
from tilefoundry_ascend.agent.flywheel.ablation_distiller import FlywheelDistiller
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.kb.schema import AscendCausalRule, SymptomFamily
from tilefoundry_ascend.agent.orchestrator.gate_funnel import GateFunnel, StageAResult, StageBResult
from tilefoundry_ascend.agent.orchestrator.llm_client import GenerationRequest, MockLLMBackend
from tilefoundry_ascend.agent.router.capability_gate import route_execution_track
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger
from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask


def _make_anchor_metric(
    *,
    case_num: int,
    regime_tag: str,
    cand_us: float,
    ref_perf_us: float = 10.0,
    t_hw_us: float = 2.0,
    eq3_score: float = 0.5,
) -> surrogate_cannbench.AnchorEvalMetrics:
    return surrogate_cannbench.AnchorEvalMetrics(
        case_num=case_num,
        regime_tag=regime_tag,
        accuracy_passed=True,
        cand_us=cand_us,
        ref_perf_us=ref_perf_us,
        t_hw_us=t_hw_us,
        speedup=ref_perf_us / cand_us if cand_us > 0.0 else 0.0,
        roofline_eta=t_hw_us / cand_us if cand_us > 0.0 else 0.0,
        eq3_score=eq3_score,
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


def _make_ledger(*, in_loop_pipe_all_barriers: int = 0) -> object:
    return build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=10,
        pipe_cycles={"AIV0.PIPE_V": 10},
        waterlines={},
        grid_blocks=16,
        wave_occupancy=0.5,
        in_loop_pipe_all_barriers=in_loop_pipe_all_barriers,
    )


@pytest.mark.anyio
async def test_stage_a_uses_bottleneck_anchor_dims_and_fresh_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _execute_single_round uses bottleneck case 16 dims and fresh ledger."""
    mod_src = '''
def _select_small_decode_config(D=1024, **_kw):
    return {"tile_m": 1 if D >= 6144 else 4}

def _build_small_decode(tile_m=4):
    return tile_m

def kernel(x):
    return x
'''
    target_path = tmp_path / "kernels.py"
    target_path.write_text(mod_src, encoding="utf-8")

    regime_spec = RegimeTargetSpec(
        regime_tag="small_decode",
        builder_candidates=("_build_small_decode",),
        sample_builder_kwargs={"tile_m": 4},
        config_selector_fn="_select_small_decode_config",
        editable_symbols=("_select_small_decode_config", "_build_small_decode"),
        representative_anchor_dims={"B": 1, "S": 1, "D": 1024},
        representative_hir_dims={"B": 1, "S": 1, "D": 1024},
    )
    anchors = (
        AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 10.0, 2.0),
        AnchorCaseSpec("level4/attention_residual", 16, "small_decode", 20.0, 2.0),
    )
    seen_l0_overrides: list[dict[str, object] | None] = []
    seen_prompt_barriers: list[str] = []

    def _l0_checker(
        src: str,
        reg: RegimeTargetSpec,
        *,
        anchor_dims_override: Mapping[str, object] | None = None,
    ) -> StageAResult:
        _ = (src, reg)
        seen_l0_overrides.append(
            dict(anchor_dims_override) if anchor_dims_override is not None else None
        )
        d_val = int((anchor_dims_override or {}).get("D", 1024))
        barriers = 3 if d_val == 6144 else 0
        ledger = _make_ledger(in_loop_pipe_all_barriers=barriers)
        route = route_execution_track(None, ledger=ledger)
        return StageAResult(
            passed=True,
            prim_func=None,
            ledger=ledger,
            features=route.features,
            route=route,
        )

    class _SpyLLM:
        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            _ = effort_override
            seen_prompt_barriers.append(req.user_prompt)
            if req.role == "explore_creative":
                return (
                    "# target_bottleneck: sync\n"
                    "# structural_hypothesis: Eliminate barrier\n"
                    "```python\ndef _build_small_decode(tile_m=4):\n    return tile_m\n```"
                )
            return "def _build_small_decode(tile_m=4):\n    return tile_m\n"

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            max_rounds=1,
            k_rule_workers=1,
            enable_tilesize_prescreen=False,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 3.0, 2.0, 0.90),
        anchors=anchors,
        regime_specs=(regime_spec,),
        llm_backend=_SpyLLM(),
        tracker=BayesianRuleTracker.from_seed_yaml(),
        l0_checker=_l0_checker,
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=lambda cands, anc: {},
        checkpoint_path=tmp_path / "ckpt.json",
        campaign_kb_path=tmp_path / "kb.json",
    )

    def _lookup_dims(case_num: int, regime_tag: str = "") -> dict[str, object]:
        _ = regime_tag
        if int(case_num) == 16:
            return {"B": 1, "S": 1, "D": 6144}
        return {"B": 1, "S": 1, "D": 1024}

    monkeypatch.setattr(runner, "_lookup_case_dims_safe", _lookup_dims)
    monkeypatch.setattr(runner._round_context_builder, "_lookup_case_dims_safe", _lookup_dims)
    monkeypatch.setattr(runner._stage_gate_coordinator, "_lookup_case_dims_safe", _lookup_dims)
    monkeypatch.setattr(runner, "_collect_msprof_for_regimes", lambda *_a, **_kw: None)
    monkeypatch.setattr(campaign_runner, "scan_tilelang_pipe_gaps", lambda: None)
    monkeypatch.setattr(round_context_builder, "scan_tilelang_pipe_gaps", lambda: None)
    from tilefoundry_ascend.agent.orchestrator import llm_constraints

    monkeypatch.setattr(llm_constraints, "format_target_dialect_layer", lambda *_a, **_kw: "")

    stale_stone_ledger = _make_ledger(in_loop_pipe_all_barriers=0)
    stale_route = route_execution_track(None, ledger=stale_stone_ledger)
    from tilefoundry_ascend.agent.orchestrator.residual_beam import (
        BeamCandidate,
        compute_static_dynamic_residual,
    )

    runner.beam_for_regime("small_decode").stepping_stone_beam.append(
        BeamCandidate(
            candidate_id="stone_r1",
            round_idx=1,
            worker_kind="creative_explore",
            rule_id=None,
            dsl_source="def _build_small_decode(tile_m=4):\n    return tile_m\n",
            track="unconstrained",
            ledger=stale_stone_ledger,
            residual=compute_static_dynamic_residual(stale_stone_ledger, {"dur_us": 20.0}),
            latency_us=20.0,
            accuracy_passed=True,
            structural_signature="sig_stone",
            features=stale_route.features,
        )
    )

    captured_ctx_barriers: list[int] = []
    orig_rank = campaign_runner.rank_symptom_families

    def _spy_rank(ctx: Mapping[str, object]) -> list[tuple[SymptomFamily, float]]:
        captured_ctx_barriers.append(int(ctx.get("in_loop_pipe_all_barriers", 0) or 0))
        return list(orig_rank(ctx))

    monkeypatch.setattr(campaign_runner, "rank_symptom_families", _spy_rank)
    monkeypatch.setattr(round_context_builder, "rank_symptom_families", _spy_rank)

    async def _empty_filter(*_a: object, **_kw: object) -> list[surrogate_cannbench.L0L1CandidateCheck]:
        return []

    monkeypatch.setattr(campaign_runner, "run_l0_l1_round_filter", _empty_filter)
    monkeypatch.setattr(stage_gate_coordinator, "run_l0_l1_round_filter", _empty_filter)

    best_result = _make_surrogate_result(
        "base",
        "attention_residual",
        [
            _make_anchor_metric(case_num=2, regime_tag="small_decode", cand_us=4.0, ref_perf_us=10.0, eq3_score=0.9),
            _make_anchor_metric(case_num=16, regime_tag="small_decode", cand_us=18.0, ref_perf_us=20.0, eq3_score=0.1),
        ],
    )
    await runner._execute_single_round(
        round_idx=2,
        regime_spec=regime_spec,
        incumbent_source=mod_src,
        best_result=best_result,
        active_anchors=list(anchors),
        stagnation_rounds=1,
        admitted_rule_ids=[],
    )

    assert seen_l0_overrides
    assert seen_l0_overrides[0] == {"B": 1, "S": 1, "D": 6144}
    assert captured_ctx_barriers == [3]


def test_default_l0_checker_trackb_uses_anchor_dims_override(tmp_path: Path) -> None:
    """Verify _default_l0_checker renders Track-B TU using anchor_dims_override."""
    mod_src = '''
def _render_trackb_tu(**dims):
    d_val = int(dims.get("D", 1024))
    flag = "BARRIER_BRANCH" if d_val == 6144 else "FAST_BRANCH"
    return f"// {flag}\\n" + str(dims["trackb_src"])

_TRACKB_CPP = """
__global__ void my_kernel() {}
"""

def _build_small_decode():
    return 1
'''
    target_path = tmp_path / "kernels.py"
    target_path.write_text(mod_src, encoding="utf-8")
    spec = RegimeTargetSpec(
        regime_tag="small_decode",
        builder_candidates=("_build_small_decode",),
        sample_builder_kwargs={},
        runtime_backend="ascendc_trackb",
        trackb_symbols=("_TRACKB_CPP",),
        representative_anchor_dims={"B": 1, "S": 1, "D": 1024},
        representative_hir_dims={"B": 1, "S": 1, "D": 1024},
    )
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="attention_residual", max_rounds=1),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 3.0, 2.0, 0.90),
        anchors=(),
        regime_specs=(spec,),
        llm_backend=MockLLMBackend(),
        checkpoint_path=tmp_path / "ckpt.json",
    )
    rendered = runner._instantiate_trackb_tu(
        mod_src,
        "__global__ void my_kernel() {}",
        spec,
        anchor_dims_override={"D": 6144},
    )
    assert "BARRIER_BRANCH" in rendered
    assert "FAST_BRANCH" not in rendered


@pytest.mark.anyio
async def test_l3_counterexample_promotes_anchor_and_one_shot_repair_promotes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Top-2 L3 fallback promotes rank-2 when rank-1 fails L3 with zero cx_repair calls."""
    incumbent_src = '''
_TRACKB_CPP = """
__aicore__ inline void tail_helper(int k_cur, int bk) {
    if (k_cur <= bk - 16) {
        return;
    }
}

__global__ void gmm_kernel(int k_cur, int bk) {
    tail_helper(k_cur, bk);
}
"""

def _build_gmm():
    return 1

def kernel(x):
    return x
'''
    broken_cand_snippet = '''
_TRACKB_CPP = """
__aicore__ inline void tail_helper(int k_cur, int bk) {
    if (k_cur <= bk - 16) {
        return;
    }
}

__global__ void gmm_kernel(int k_cur, int bk) {
    tail_helper(k_cur, bk);
    int fast_tag = 1;
}
"""
'''
    fallback_cand_snippet = '''
_TRACKB_CPP = """
__aicore__ inline void tail_helper(int k_cur, int bk) {
    if (k_cur < bk) {
        return;
    }
}

__global__ void gmm_kernel(int k_cur, int bk) {
    tail_helper(k_cur, bk);
    int fast_tag = 1;
}
"""
'''
    target_path = tmp_path / "kernels.py"
    target_path.write_text(incumbent_src, encoding="utf-8")

    spec = RegimeTargetSpec(
        regime_tag="ragged_gmm",
        builder_candidates=("_build_gmm",),
        sample_builder_kwargs={},
        runtime_backend="ascendc_trackb",
        trackb_symbols=("_TRACKB_CPP",),
        editable_symbols=("_TRACKB_CPP", "_build_gmm"),
        representative_anchor_dims={"M": 64, "K": 256, "N": 128},
        representative_hir_dims={"M": 64, "K": 256, "N": 128},
    )
    anchors = [
        AnchorCaseSpec("level4/grouped_matmul", 2, "ragged_gmm", 10.0, 2.0),
    ]

    repair_prompts: list[GenerationRequest] = []

    class _TopTwoLLM:
        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            _ = effort_override
            hint = str(getattr(req, "role_hint", "") or "")
            if hint == "l3_counterexample_repair" or "l3_cx_repair" in hint:
                repair_prompts.append(req)
                return f"```python\n{fallback_cand_snippet}\n```"
            if req.role == "explore_creative":
                return (
                    "# target_bottleneck: aic\n"
                    "# structural_hypothesis: Fix tail loop\n"
                    f"```python\n{fallback_cand_snippet}\n```"
                )
            await asyncio.sleep(0.02)
            return broken_cand_snippet

    def _l2_eval(
        cands: Sequence[tuple[str, Path]],
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> dict[str, surrogate_cannbench.CandidateSurrogateResult]:
        out: dict[str, surrogate_cannbench.CandidateSurrogateResult] = {}
        for cid, mod_path in cands:
            src = mod_path.read_text(encoding="utf-8")
            has_fast = "fast_tag = 1" in src
            has_fixed_tail = "k_cur < bk" in src
            metrics: list[surrogate_cannbench.AnchorEvalMetrics] = []
            for anc in active_anchors:
                if anc.case_num == 14 and has_fast and not has_fixed_tail:
                    metrics.append(
                        surrogate_cannbench.AnchorEvalMetrics(
                            case_num=14,
                            regime_tag=anc.regime_tag,
                            accuracy_passed=False,
                            cand_us=99.0,
                            ref_perf_us=anc.ref_perf_us,
                            t_hw_us=anc.t_hw_us,
                            speedup=0.0,
                            roofline_eta=0.0,
                            eq3_score=0.0,
                        )
                    )
                else:
                    if has_fast and not has_fixed_tail:
                        cand_us = 2.5
                    elif has_fast:
                        cand_us = 3.0
                    else:
                        cand_us = 9.0
                    metrics.append(
                        _make_anchor_metric(
                            case_num=anc.case_num,
                            regime_tag=anc.regime_tag,
                            cand_us=cand_us,
                            ref_perf_us=anc.ref_perf_us,
                            t_hw_us=anc.t_hw_us,
                            eq3_score=0.9 if has_fast else 0.4,
                        )
                    )
            out[cid] = _make_surrogate_result(cid, "grouped_matmul", metrics)
        return out

    l3_seen_sources: list[str] = []

    def _l3_runner(mod_path: Path, **_kw: object) -> surrogate_cannbench.L3AcceptanceResult:
        src = mod_path.read_text(encoding="utf-8")
        l3_seen_sources.append(src)
        if "k_cur < bk" in src:
            return surrogate_cannbench.L3AcceptanceResult(
                passed=True,
                run_id="run-l3-fixed",
                execution_id="exec-l3-fixed",
                device_id="1",
            )
        return surrogate_cannbench.L3AcceptanceResult(
            passed=False,
            error_message="NUMERIC_MISMATCH at case=14 (K=127 NaN)",
            failure_kind=surrogate_cannbench.FAILURE_KIND_CANDIDATE,
            failure_category="numeric",
            execution_id="exec-l3-fail",
            run_id="run-l3-fail",
            device_id="1",
        )

    rule = AscendCausalRule(
        rule_id="R_TEST_TOP2_FALLBACK",
        regime_guard="True",
        precondition="True",
        transform_template="Optimize main loop",
        expected_postcondition="True",
        source_provenance="test",
        compatible_backends=("ascendc_trackb",),
    )
    tracker = BayesianRuleTracker(rules={})
    tracker.admit_rule(rule)
    ledger = _make_ledger()
    route = route_execution_track(None, ledger=ledger)
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="grouped_matmul",
            max_rounds=1,
            k_rule_workers=1,
            enable_tilesize_prescreen=False,
            min_surrogate_gain=0.05,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("grouped_matmul", 3.0, 2.0, 0.90),
        anchors=anchors,
        regime_specs=(spec,),
        llm_backend=_TopTwoLLM(),
        tracker=tracker,
        l0_checker=lambda _s, _r: StageAResult(
            passed=True,
            prim_func=None,
            ledger=ledger,
            features=route.features,
            route=route,
            backend_kind="ascendc",
        ),
        l1_compiler=lambda _s, _r: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=_l3_runner,
        checkpoint_path=tmp_path / "ckpt.json",
    )

    def _lookup_dims(cnum: int, _tag: str = "") -> dict[str, object]:
        return {"K": 127} if int(cnum) == 14 else {"K": 256}

    monkeypatch.setattr(runner, "_lookup_case_dims_safe", _lookup_dims)
    monkeypatch.setattr(runner._stage_gate_coordinator, "_lookup_case_dims_safe", _lookup_dims)
    monkeypatch.setattr(runner._round_context_builder, "_lookup_case_dims_safe", _lookup_dims)
    monkeypatch.setattr(runner._l3_l4_recovery, "_lookup_case_dims_fn", _lookup_dims)
    monkeypatch.setattr(runner, "_collect_msprof_for_regimes", lambda *_a, **_kw: None)
    monkeypatch.setattr(campaign_runner, "scan_tilelang_pipe_gaps", lambda: None)
    monkeypatch.setattr(round_context_builder, "scan_tilelang_pipe_gaps", lambda: None)
    monkeypatch.setattr(campaign_runner, "lookup_case_metadata", lambda *_a, **_kw: (12.0, 2.0))
    monkeypatch.setattr(l3_l4_recovery, "lookup_case_metadata", lambda *_a, **_kw: (12.0, 2.0))
    monkeypatch.setattr(runner._tracker, "rank_applicable_rules", lambda *_a, **_kw: [rule])

    best_result = _make_surrogate_result(
        "base",
        "grouped_matmul",
        [_make_anchor_metric(case_num=2, regime_tag="ragged_gmm", cand_us=9.0, ref_perf_us=10.0, eq3_score=0.4)],
    )
    rec, new_inc, _new_best, stag_after, promoted, rolled_back, _esc, blocked = (
        await runner._execute_single_round(
            round_idx=1,
            regime_spec=spec,
            incumbent_source=incumbent_src,
            best_result=best_result,
            active_anchors=anchors,
            stagnation_rounds=0,
            admitted_rule_ids=[],
        )
    )

    assert repair_prompts == []
    assert [a.case_num for a in anchors] == [2]
    assert len(l3_seen_sources) == 2
    assert "k_cur <= bk - 16" in l3_seen_sources[0]
    assert "k_cur < bk" in l3_seen_sources[1]
    assert promoted is True
    assert rolled_back is False
    assert blocked is False
    assert stag_after == 0
    assert rec.promoted is True
    assert "k_cur < bk" in new_inc
    assert "k_cur < bk" in target_path.read_text(encoding="utf-8")


@pytest.mark.anyio
async def test_l3_counterexample_repair_failure_keeps_anchor_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify L3 rejection records monotonic guardrails, rolls back, and skips cx_repair."""
    incumbent_src = '''
def _build_gmm():
    return 1

def kernel(x):
    return x
'''
    cand_snippet = "def _build_gmm():\n    return 2\n"
    target_path = tmp_path / "kernels.py"
    target_path.write_text(incumbent_src, encoding="utf-8")

    spec = RegimeTargetSpec(
        regime_tag="ragged_gmm",
        builder_candidates=("_build_gmm",),
        sample_builder_kwargs={},
        editable_symbols=("_build_gmm",),
        representative_anchor_dims={"M": 64, "K": 256, "N": 128},
        representative_hir_dims={"M": 64, "K": 256, "N": 128},
    )
    anchors = [AnchorCaseSpec("level4/grouped_matmul", 2, "ragged_gmm", 10.0, 2.0)]
    repair_prompts: list[GenerationRequest] = []

    class _RejectLLM:
        async def generate(
            self,
            req: GenerationRequest,
            *,
            effort_override: object = None,
        ) -> str:
            _ = effort_override
            hint = str(getattr(req, "role_hint", "") or "")
            if hint == "l3_counterexample_repair" or "l3_cx_repair" in hint:
                repair_prompts.append(req)
                return cand_snippet
            if req.role == "explore_creative":
                return (
                    "# target_bottleneck: aiv\n"
                    "# structural_hypothesis: Try return 2\n"
                    f"```python\n{cand_snippet}\n```"
                )
            return cand_snippet

    def _l2_eval(
        cands: Sequence[tuple[str, Path]],
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> dict[str, surrogate_cannbench.CandidateSurrogateResult]:
        out: dict[str, surrogate_cannbench.CandidateSurrogateResult] = {}
        for cid, _p in cands:
            metrics = [
                _make_anchor_metric(
                    case_num=anc.case_num,
                    regime_tag=anc.regime_tag,
                    cand_us=3.0,
                    ref_perf_us=anc.ref_perf_us,
                    t_hw_us=anc.t_hw_us,
                    eq3_score=0.9,
                )
                for anc in active_anchors
            ]
            out[cid] = _make_surrogate_result(cid, "grouped_matmul", metrics)
        return out

    ledger = _make_ledger()
    route = route_execution_track(None, ledger=ledger)
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="grouped_matmul",
            max_rounds=1,
            k_rule_workers=0,
            enable_tilesize_prescreen=False,
            min_surrogate_gain=0.05,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("grouped_matmul", 3.0, 2.0, 0.90),
        anchors=anchors,
        regime_specs=(spec,),
        llm_backend=_RejectLLM(),
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
        l3_contract_runner=lambda _p, **_kw: surrogate_cannbench.L3AcceptanceResult(
            passed=False,
            error_message="numeric error case=14 case=15 case=16",
            failure_kind=surrogate_cannbench.FAILURE_KIND_CANDIDATE,
        ),
        checkpoint_path=tmp_path / "ckpt.json",
    )
    monkeypatch.setattr(runner, "_lookup_case_dims_safe", lambda cnum, _tag="": {"K": int(cnum)})
    monkeypatch.setattr(
        runner._stage_gate_coordinator,
        "_lookup_case_dims_safe",
        lambda cnum, _tag="": {"K": int(cnum)},
    )
    monkeypatch.setattr(
        runner._l3_l4_recovery,
        "_lookup_case_dims_fn",
        lambda cnum, _tag="": {"K": int(cnum)},
    )
    monkeypatch.setattr(
        l3_l4_recovery,
        "lookup_case_dims",
        lambda _op, cnum, **_kw: {"K": int(cnum)},
    )
    monkeypatch.setattr(runner, "_collect_msprof_for_regimes", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        campaign_runner,
        "lookup_case_metadata",
        lambda *_a, **_kw: (_ for _ in ()).throw(FileNotFoundError("no metadata")),
    )
    monkeypatch.setattr(
        l3_l4_recovery,
        "lookup_case_metadata",
        lambda *_a, **_kw: (_ for _ in ()).throw(FileNotFoundError("no metadata")),
    )

    best_result = _make_surrogate_result(
        "base",
        "grouped_matmul",
        [_make_anchor_metric(case_num=2, regime_tag="ragged_gmm", cand_us=9.0, ref_perf_us=10.0, eq3_score=0.4)],
    )
    rec, new_inc, _new_best, stag_after, promoted, rolled_back, _esc, _blocked = (
        await runner._execute_single_round(
            round_idx=1,
            regime_spec=spec,
            incumbent_source=incumbent_src,
            best_result=best_result,
            active_anchors=anchors,
            stagnation_rounds=1,
            admitted_rule_ids=[],
        )
    )

    assert repair_prompts == []
    assert [a.case_num for a in anchors] == [2]
    assert [g.case_num for g in runner._guardrail_cases] == [14, 15, 16]
    assert promoted is False
    assert rolled_back is True
    assert stag_after == 2
    assert rec.promoted is False
    assert rec.l3_resolution == "CANDIDATE_REJECTED"
    assert new_inc == incumbent_src
    assert target_path.read_text(encoding="utf-8") == incumbent_src


def test_device_pool_forwarding_l3_pool_rotation_and_main_wiring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify device_pool propagates across gates, L3 rotates pool cards, and main binds backend."""
    target_path = tmp_path / "kernels.py"
    target_path.write_text("def _build():\n    return 1\n", encoding="utf-8")
    env_path = tmp_path / "env.sh"
    env_path.write_text(
        "TILEFOUNDRY_NPU_CAMPAIGN_DEVICE_IDS=3,4,5\n"
        "TILEFOUNDRY_NPU_LLM_BACKEND=mock\n",
        encoding="utf-8",
    )

    seen_pools: dict[str, object] = {}

    def _fake_l2(*_a: object, device_pool: object = None, **_kw: object) -> tuple[dict[str, surrogate_cannbench.CandidateSurrogateResult], str, str]:
        seen_pools["l2"] = device_pool
        return {}, "run-l2", ""

    def _fake_msprof(*_a: object, device_pool: object = None, **_kw: object) -> dict[str, dict[str, float]]:
        seen_pools["msprof"] = device_pool
        return {}

    def _fake_p2(*_a: object, device_pool: object = None, **_kw: object) -> dict[str, object]:
        seen_pools["p2"] = device_pool
        return {}

    def _fake_probe(*_a: object, device_pool: object = None, **_kw: object) -> dict[str, float]:
        seen_pools["probe"] = device_pool
        return {"speedup": 1.1}

    l3_devices: list[str] = []

    def _fake_l3(
        *,
        candidate_module_path: Path | None = None,
        device_id: str = "",
        device_pool: object = None,
        on_execution_started: object = None,
    ) -> surrogate_cannbench.L3AcceptanceResult:
        _ = (candidate_module_path, on_execution_started)
        seen_pools["l3"] = device_pool
        l3_devices.append(device_id)
        if device_id in ("3", "4"):
            return surrogate_cannbench.L3AcceptanceResult(
                passed=False,
                failure_kind=surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE,
                failure_category="occupancy",
                error_message="blocked occupancy",
                device_id=device_id,
                execution_terminal=True,
            )
        return surrogate_cannbench.L3AcceptanceResult(
            passed=True,
            device_id=device_id,
            execution_id="exec-5",
            run_id="run-5",
        )

    monkeypatch.setattr(campaign_runner, "run_surrogate_batch_via_agent_gates", _fake_l2)
    monkeypatch.setattr(stage_gate_coordinator, "run_surrogate_batch_via_agent_gates", _fake_l2)
    monkeypatch.setattr(campaign_runner, "collect_bottleneck_msprof_via_agent_gates", _fake_msprof)
    monkeypatch.setattr(campaign_runner, "collect_p2_deep_dive_via_agent_gates", _fake_p2)
    monkeypatch.setattr(campaign_runner, "run_micro_probe_via_agent_gates", _fake_probe)
    monkeypatch.setattr(campaign_runner, "run_l3_full_contract_via_agent_gates", _fake_l3)
    monkeypatch.setattr(l3_l4_recovery, "run_l3_full_contract_via_agent_gates", _fake_l3)

    flywheel = FlywheelDistiller(
        funnel=GateFunnel(skip_stage_b=True),
        tracker=BayesianRuleTracker(rules={}),
    )
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            device_id="3",
            device_ids=("4", "5"),
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 3.0, 2.0, 0.90),
        anchors=(),
        regime_specs=(RegimeTargetSpec("r0", ("_build",), {}, representative_anchor_dims={"B": 1, "S": 1, "D": 1024}, representative_hir_dims={"B": 1, "S": 1, "D": 1024}),),
        llm_backend=MockLLMBackend(),
        flywheel_distiller=flywheel,
        env_file=env_path,
        checkpoint_path=tmp_path / "ckpt.json",
    )
    runner._run_l2_batch([], [])
    assert runner._msprof_collector is not None
    runner._msprof_collector("attention_residual", str(target_path), [])
    assert runner._p2_deep_dive_collector is not None
    runner._p2_deep_dive_collector("attention_residual", str(target_path), [])
    assert flywheel._probe_runner is not None
    flywheel._probe_runner({"baseline": "a", "optimized": "b"})
    runner._run_l3_contract(target_path)

    assert seen_pools == {
        "l2": ("3", "4", "5"),
        "msprof": ("3", "4", "5"),
        "p2": ("3", "4", "5"),
        "probe": ("3", "4", "5"),
        "l3": ("3", "4", "5"),
    }

    pending = campaign_runner.PendingPromotion(
        candidate_source="def _build():\n    return 2\n",
        candidate_sha256=runner._source_sha256("def _build():\n    return 2\n"),
        target_path=str(target_path),
        round_idx=1,
        regime_tag="r0",
        candidate_result=_make_surrogate_result("c1", "attention_residual", ()),
        attempts=({"device_id": "3", "failure_kind": "infrastructure"},),
        device_id="3",
        execution_terminal=True,
    )
    l3_devices.clear()
    _inc, _best, _stag, next_pending, blocked = runner._resolve_pending_promotion(
        pending,
        incumbent_source="def _build():\n    return 1\n",
        best_result=_make_surrogate_result("base", "attention_residual", ()),
        stagnation_by_regime={"r0": 0},
        rounds=[],
        promoted_rounds=[],
        rolled_back_rounds=[],
        active_anchors=(),
    )
    assert l3_devices == ["4", "5"]
    assert next_pending is None
    assert blocked is False

    seen_main_runners: list[PerfTargetCampaignRunner] = []

    class _FakeRunner(PerfTargetCampaignRunner):
        async def run(self, *, resume: bool = False) -> campaign_runner.CampaignReport:
            _ = resume
            seen_main_runners.append(self)
            empty_res = _make_surrogate_result("base", "attention_residual", ())
            return campaign_runner.CampaignReport(
                operator_name="attention_residual",
                stop_reason="budget_exhausted",
                total_rounds=0,
                baseline_result=empty_res,
                best_result=empty_res,
                rounds=(),
            )

    monkeypatch.setattr(campaign_runner, "PerfTargetCampaignRunner", _FakeRunner)
    rc = campaign_runner.main(
        [
            "--operator",
            "attention_residual",
            "--env-file",
            str(env_path),
            "--target-module",
            str(target_path),
            "--checkpoint",
            str(tmp_path / "main_ckpt.json"),
        ]
    )
    assert rc == 0
    assert len(seen_main_runners) == 1
    assert seen_main_runners[0]._config.device_id == "3"
    assert seen_main_runners[0]._l4_device_pool == ("3", "4", "5")
    assert isinstance(seen_main_runners[0]._backend, MockLLMBackend)
