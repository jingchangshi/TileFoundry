"""Unit tests for v5.3 immutable 5-anchor scoreboard, flywheel, bandit, and timings."""

from __future__ import annotations

import importlib
import json
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


def _ensure_cost_advisor_stub() -> None:
    """Provide an in-memory cost_advisor stub when running before WP-1 merges."""
    mod_name = "tilefoundry_ascend.agent.cost_advisor"
    try:
        importlib.import_module(mod_name)
        return
    except ModuleNotFoundError:
        pass

    from tilefoundry_ascend.ascend.cost_model.cost_gate_policy import (
        CostGateDecision,
        CostModelMode,
        evaluate_cost_gate_policy,
    )
    from tilefoundry_ascend.ascend.target import AscendTarget

    stub = types.ModuleType(mod_name)

    class CostModelAdvisor:
        """Minimal stub matching the frozen CostModelAdvisor interface."""

        def __init__(
            self,
            mode: CostModelMode | str = CostModelMode.OFF,
            *,
            target: AscendTarget | None = None,
            operator_name: str = "",
            calibration_path: Path | None = None,
            experience_lake: object | None = None,
        ) -> None:
            self.mode = CostModelMode(str(mode).strip().lower())
            self.target = target or AscendTarget("ascend.950pr")
            self.operator_name = operator_name
            self.calibration_path = calibration_path
            self.experience_lake = experience_lake
            self.predictions_invoked = 0
            self.underlying_cost_model: object | None = None

        def sync_experience_lake(
            self,
            lake: object | None,
            *,
            persist_calibration: bool = False,
            exclude_run_id: str = "",
        ) -> None:
            _ = (lake, persist_calibration, exclude_run_id)

        def evaluate_round_policy(
            self,
            *,
            enable_prescreen_flag: bool = True,
            bucket_clean_samples: int = 0,
            out_of_sample_spearman_rho: float | None = None,
        ) -> CostGateDecision:
            return evaluate_cost_gate_policy(
                mode=self.mode,
                enable_prescreen_flag=enable_prescreen_flag,
                bucket_clean_samples=bucket_clean_samples,
                out_of_sample_spearman_rho=out_of_sample_spearman_rho,
            )

        def predict_candidate_us(
            self,
            workload: object,
            *,
            bisheng_res: object | None = None,
        ) -> object | None:
            _ = (workload, bisheng_res)
            if self.mode == CostModelMode.OFF:
                return None
            self.predictions_invoked += 1
            return SimpleNamespace(
                pred_us=12.0,
                tier0_physics_us=10.0,
                tier1_penalty_us=2.0,
                dominant_bound="vec",
            )

        def predict_anchor_mean_us(
            self,
            ledger: object,
            *,
            anchors: Sequence[object] = (),
            live_plan: object | None = None,
            axes: object | None = None,
            blueprint: object | None = None,
            sil_advisory: object | None = None,
            bisheng_res: object | None = None,
        ) -> float | None:
            _ = (ledger, anchors, live_plan, axes, blueprint, sil_advisory, bisheng_res)
            if self.mode == CostModelMode.OFF:
                return None
            self.predictions_invoked += 1
            return 12.0

        def build_prompt_guidance(
            self,
            *,
            decision: CostGateDecision,
            base_cfg: Mapping[str, object],
            dims: Mapping[str, object] | None = None,
            regime_tag: str = "",
            top_k: int = 4,
        ) -> str:
            _ = (decision, base_cfg, dims, regime_tag, top_k)
            return ""

        def filter_stage_a_passers(
            self,
            passers: Sequence[tuple[str, str, object]],
            **_kwargs: object,
        ) -> list[tuple[str, float]]:
            return [(cid, 10.0 + idx) for idx, (cid, _src, _sa) in enumerate(passers)]

        def observe_surrogate_batch(
            self,
            *,
            observations: Sequence[Mapping[str, object]] = (),
            msprof_records: Sequence[Mapping[str, object]] = (),
            pair_deltas: Sequence[Mapping[str, object]] = (),
            refit: bool = True,
        ) -> None:
            _ = (observations, msprof_records, pair_deltas, refit)

        def resolve_dominant_bound(
            self,
            ledger: object | None,
            residual: object | None,
            *,
            fallback_cost_pred: object | None = None,
        ) -> str:
            _ = (ledger, residual)
            if fallback_cost_pred is not None and hasattr(fallback_cost_pred, "dominant_bound"):
                return str(getattr(fallback_cost_pred, "dominant_bound"))
            return "memory"

    def create_cost_advisor(
        mode: CostModelMode | str = CostModelMode.OFF,
        *,
        target: AscendTarget | None = None,
        operator_name: str = "",
        calibration_path: Path | None = None,
        experience_lake: object | None = None,
    ) -> CostModelAdvisor:
        return CostModelAdvisor(
            mode=mode,
            target=target,
            operator_name=operator_name,
            calibration_path=calibration_path,
            experience_lake=experience_lake,
        )

    setattr(stub, "CostModelAdvisor", CostModelAdvisor)
    setattr(stub, "create_cost_advisor", create_cost_advisor)
    sys.modules[mod_name] = stub


_ensure_cost_advisor_stub()

from tilefoundry_ascend.agent.campaign import campaign_runner  # noqa: E402
from tilefoundry_ascend.agent.campaign.anchor_specs import (  # noqa: E402
    AnchorCaseSpec,
    GuardrailCaseSpec,
    PerfTargetSpec,
    RegimeTargetSpec,
)
from tilefoundry_ascend.agent.campaign.campaign_runner import (  # noqa: E402
    CampaignConfig,
    PerfTargetCampaignRunner,
    _compact_trackb_snippet_source,
)
from tilefoundry_ascend.agent.campaign.funnel_telemetry import RoundFunnelEvent  # noqa: E402
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (  # noqa: E402
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    L2BatchResults,
    L3AcceptanceResult,
    L4CannbenchResult,
    aggregate_surrogate_result,
    build_anchor_metrics,
)
from tilefoundry_ascend.agent.flywheel.ablation_distiller import FlywheelDistiller  # noqa: E402
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker  # noqa: E402
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (  # noqa: E402
    GateFunnel,
    StageAResult,
    StageBResult,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import GenerationRequest  # noqa: E402
from tilefoundry_ascend.ascend.passes.physical_ledger import (  # noqa: E402
    build_objective_physical_ledger,
)
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes  # noqa: E402


def _make_ledger(ii: int = 20, blocks: int = 8) -> object:
    return build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=ii,
        pipe_cycles={"AIV0.PIPE_V": ii},
        waterlines={},
        gm_roundtrip_bytes=4096,
        grid_blocks=blocks,
        wave_occupancy=0.5,
    )


def _five_anchors() -> tuple[AnchorCaseSpec, ...]:
    return (
        AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 15.0, 1.5),
        AnchorCaseSpec("level4/attention_residual", 4, "unaligned_d", 30.0, 5.0),
        AnchorCaseSpec("level4/attention_residual", 14, "unaligned_d", 60.0, 20.0),
        AnchorCaseSpec("level4/attention_residual", 16, "aligned_prefill", 500.0, 200.0),
        AnchorCaseSpec("level4/attention_residual", 17, "aligned_prefill", 500.0, 200.0),
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


def _base_module_src() -> str:
    return (
        "def _build_small(tile_m=1):\n    return tile_m\n\n"
        "def _build_unaligned(tile_m=1):\n    return tile_m\n\n"
        "def _build_prefill(tile_m=4):\n    return tile_m\n"
    )


def _patch_env(monkeypatch: pytest.MonkeyPatch, runner: PerfTargetCampaignRunner) -> None:
    monkeypatch.setattr(runner, "_collect_msprof_for_regimes", lambda *_a, **_kw: None)
    monkeypatch.setattr(campaign_runner, "scan_tilelang_pipe_gaps", lambda: None)
    from tilefoundry_ascend.agent.orchestrator import llm_constraints

    monkeypatch.setattr(llm_constraints, "format_target_dialect_layer", lambda *_a, **_kw: "")


@pytest.mark.anyio
async def test_scoreboard_invariance_on_l3_failure_and_no_post_budget_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L3 failure adds true-regime guardrails without mutating 5-anchor scoreboard or rounds."""
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_base_module_src(), encoding="utf-8")
    anchors = _five_anchors()
    regimes = _three_regimes()
    seen_l2_anchor_cases: list[list[int]] = []
    l4_calls = 0

    class _LLM:
        async def generate(self, req: GenerationRequest, *, effort_override: object = None) -> str:
            _ = effort_override
            if req.role == "syntax_repair":
                return ""
            if "unaligned_d" in req.user_prompt:
                code = "def _build_unaligned(tile_m=2):\n    return tile_m\n"
            elif "aligned_prefill" in req.user_prompt:
                code = "def _build_prefill(tile_m=2):\n    return tile_m\n"
            else:
                code = "def _build_small(tile_m=2):\n    return tile_m\n"
            if req.role == "explore_creative":
                return (
                    "# target_bottleneck: aiv\n"
                    "# structural_hypothesis: test\n"
                    f"```python\n{code}```\n"
                )
            return code

    def _l2_eval(
        cands: Sequence[tuple[str, Path]],
        eval_anchors: Sequence[AnchorCaseSpec],
    ) -> dict[str, CandidateSurrogateResult]:
        seen_l2_anchor_cases.append([int(a.case_num) for a in eval_anchors])
        out: dict[str, CandidateSurrogateResult] = {}
        for cid, mpath in cands:
            src = mpath.read_text(encoding="utf-8")
            is_fast = "tile_m=2" in src
            metrics = []
            for anc in eval_anchors:
                if anc.case_num in (99, 100):
                    cand_us = 25.0
                elif anc.regime_tag == "small_decode":
                    cand_us = 10.0 if is_fast else 15.0
                elif anc.regime_tag == "unaligned_d":
                    cand_us = 20.0 if is_fast else 30.0
                else:
                    cand_us = 250.0
                metrics.append(build_anchor_metrics(anc, accuracy_passed=True, cand_us=cand_us))
            out[cid] = aggregate_surrogate_result(cid, "attention_residual", metrics)
        return out

    def _l3_fail(_p: Path) -> L3AcceptanceResult:
        return L3AcceptanceResult(
            passed=False,
            failure_kind="candidate",
            failure_stage="accuracy",
            failed_case_nums=(99, 100),
            error_message="failed case=99 case=100",
        )

    def _l4_pass(_p: Path) -> L4CannbenchResult:
        nonlocal l4_calls
        l4_calls += 1
        return L4CannbenchResult(
            passed=True,
            case_count=10,
            passed_cases=10,
            per_case_speedups={200: 0.25},
        )

    def _lookup_dims(
        _op: str,
        case_num: int,
        **_kw: object,
    ) -> dict[str, object]:
        if int(case_num) == 99:
            return {"B": 1, "S": 128, "D": 4096}
        if int(case_num) == 100:
            return {"B": 1, "S": 8, "D": 1000}
        return {"B": 1, "S": 1, "D": 4096}

    monkeypatch.setattr(campaign_runner, "lookup_case_dims", _lookup_dims)
    monkeypatch.setattr(
        campaign_runner,
        "lookup_case_metadata",
        lambda _op, cnum, **_kw: (25.0, 5.0) if int(cnum) in (99, 100) else (10.0, 2.0),
    )

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            max_rounds=1,
            k_rule_workers=0,
            max_syntax_repairs=0,
            run_final_l4=True,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 5.0, 2.0, 0.99),
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=_LLM(),
        tracker=BayesianRuleTracker(rules={}),
        l0_checker=lambda _s, _r: StageAResult(passed=True, prim_func=None, ledger=_make_ledger()),
        l1_compiler=lambda _s, _r: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=_l3_fail,
        l4_cannbench_runner=_l4_pass,
        checkpoint_path=tmp_path / "ckpt.json",
    )
    _patch_env(monkeypatch, runner)

    report = await runner.run()
    assert len(report.rounds) == 1
    assert l4_calls == 1
    assert report.rounds[0].l3_resolution == "CANDIDATE_REJECTED"
    assert report.scoreboard_anchor_count == 5
    assert [int(a.case_num) for a in report.active_anchors] == [2, 4, 14, 16, 17]
    assert [int(a.case_num) for a in report.scoreboard_anchors] == [2, 4, 14, 16, 17]
    assert report.best_result.avg_speedup == report.baseline_result.avg_speedup
    assert report.best_result.surrogate_score == report.baseline_result.surrogate_score
    assert len(runner._guardrail_cases) == 2
    by_case = {g.case_num: g for g in runner._guardrail_cases}
    assert by_case[99].regime_tag == "aligned_prefill"
    assert by_case[100].regime_tag == "unaligned_d"
    assert by_case[99].incumbent_ref_us == pytest.approx(25.0)
    assert by_case[100].incumbent_ref_us == pytest.approx(25.0)

    c_file = tmp_path / "cand.py"
    c_file.write_text(_base_module_src(), encoding="utf-8")
    stripped_res, _ = runner._run_l2_batch([("probe", c_file)], anchors, include_guardrails=True)
    assert seen_l2_anchor_cases[-1] == [2, 4, 14, 16, 17, 99, 100]
    assert [int(a.case_num) for a in stripped_res["probe"].anchors] == [2, 4, 14, 16, 17]


@pytest.mark.anyio
async def test_guardrail_105x_rejection_and_winner_refresh_keeps_scoreboard_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guardrail blocks >1.05x latency or accuracy failure and refreshes on winner promotion."""
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_base_module_src(), encoding="utf-8")
    anchors = _five_anchors()
    regimes = _three_regimes()
    mode_state = {"guardrail_us": 21.5, "guardrail_acc": True}

    class _LLM:
        async def generate(self, req: GenerationRequest, *, effort_override: object = None) -> str:
            _ = effort_override
            if req.role == "explore_creative":
                return (
                    "# target_bottleneck: aiv\n"
                    "# structural_hypothesis: opt\n"
                    "```python\n"
                    "def _build_small(tile_m=2):\n"
                    "    return tile_m\n"
                    "```\n"
                )
            return "def _build_small(tile_m=2):\n    return tile_m\n"

    def _l2_eval(
        cands: Sequence[tuple[str, Path]],
        eval_anchors: Sequence[AnchorCaseSpec],
    ) -> dict[str, CandidateSurrogateResult]:
        out: dict[str, CandidateSurrogateResult] = {}
        for cid, mpath in cands:
            src = mpath.read_text(encoding="utf-8")
            is_fast = "tile_m=2" in src
            metrics = []
            for anc in eval_anchors:
                if int(anc.case_num) == 99:
                    if cid in ("incumbent_baseline", "incumbent_slot"):
                        metrics.append(build_anchor_metrics(anc, accuracy_passed=True, cand_us=20.0))
                    else:
                        metrics.append(
                            build_anchor_metrics(
                                anc,
                                accuracy_passed=bool(mode_state["guardrail_acc"]),
                                cand_us=float(mode_state["guardrail_us"]),
                            )
                        )
                    continue
                if anc.regime_tag == "small_decode":
                    cand_us = 10.0 if is_fast else 20.0
                elif anc.regime_tag == "unaligned_d":
                    cand_us = 25.0
                else:
                    cand_us = 250.0
                metrics.append(build_anchor_metrics(anc, accuracy_passed=True, cand_us=cand_us))
            out[cid] = aggregate_surrogate_result(cid, "attention_residual", metrics)
        return out

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            max_rounds=1,
            k_rule_workers=0,
            max_syntax_repairs=0,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 5.0, 2.0, 0.99),
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=_LLM(),
        tracker=BayesianRuleTracker(rules={}),
        l0_checker=lambda _s, _r: StageAResult(passed=True, prim_func=None, ledger=_make_ledger()),
        l1_compiler=lambda _s, _r: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        checkpoint_path=tmp_path / "ckpt.json",
    )
    _patch_env(monkeypatch, runner)
    runner._guardrail_cases = [
        GuardrailCaseSpec(
            task="level4/attention_residual",
            case_num=99,
            regime_tag="aligned_prefill",
            incumbent_ref_us=20.0,
            t_hw_us=5.0,
        )
    ]
    base_res = runner._evaluate_single_baseline(_base_module_src(), anchors)

    rec_reg, inc_reg, best_reg, _stag, promoted_reg, *_ = await runner._execute_single_round(
        round_idx=1,
        regime_spec=regimes[0],
        incumbent_source=_base_module_src(),
        best_result=base_res,
        active_anchors=anchors,
        stagnation_rounds=0,
        admitted_rule_ids=[],
    )
    assert promoted_reg is False
    assert rec_reg.tail_guard_reject_reason == "guardrail_latency_regression"
    assert best_reg.avg_speedup == base_res.avg_speedup
    assert inc_reg == _base_module_src()

    mode_state["guardrail_us"] = 18.0
    mode_state["guardrail_acc"] = False
    rec_acc, _, best_acc, _, promoted_acc, *_ = await runner._execute_single_round(
        round_idx=2,
        regime_spec=regimes[0],
        incumbent_source=_base_module_src(),
        best_result=base_res,
        active_anchors=anchors,
        stagnation_rounds=1,
        admitted_rule_ids=[],
    )
    assert promoted_acc is False
    assert rec_acc.tail_guard_reject_reason == "guardrail_accuracy_failed"
    assert best_acc.avg_speedup == base_res.avg_speedup

    mode_state["guardrail_us"] = 17.5
    mode_state["guardrail_acc"] = True
    rec_win, _, best_win, _, promoted_win, *_ = await runner._execute_single_round(
        round_idx=3,
        regime_spec=regimes[0],
        incumbent_source=_base_module_src(),
        best_result=base_res,
        active_anchors=anchors,
        stagnation_rounds=2,
        admitted_rule_ids=[],
    )
    assert promoted_win is True
    assert rec_win.tail_guard_reject_reason == ""
    assert [int(a.case_num) for a in best_win.anchors] == [2, 4, 14, 16, 17]
    assert runner._guardrail_cases[0].incumbent_ref_us == pytest.approx(17.5)


def test_flywheel_persistence_unlock_real_npu_and_cumulative_8pct_trigger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """is_real_npu=True and >=8% cumulative drop unlock kb/distilled_rules persistence."""
    target_path = tmp_path / "kernels.py"
    base_dsl = (
        "import tilelang.ascend.language as T\n\n"
        "def make_kernel():\n"
        "    @T.prim_func\n"
        "    def main(A: T.Tensor((128,), 'float32'), B: T.Tensor((128,), 'float32')):\n"
        "        with T.Kernel(1) as bx:\n"
        "            a_ub = T.alloc_shared((128,), 'float32')\n"
        "            T.copy(A[0:128], a_ub)\n"
        "            T.copy(a_ub, B[0:128])\n"
        "    return main\n"
    )
    win_dsl = (
        "import tilelang.ascend.language as T\n\n"
        "def make_kernel():\n"
        "    @T.prim_func\n"
        "    def main(A: T.Tensor((128,), 'float32'), B: T.Tensor((128,), 'float32')):\n"
        "        with T.Kernel(8) as bx:\n"
        "            a_ub = T.alloc_shared((128,), 'float32')\n"
        "            for k in T.Pipelined(2, num_stages=2):\n"
        "                T.copy(A[0:128], a_ub)\n"
        "                T.copy(a_ub, B[0:128])\n"
        "    return main\n"
    )
    target_path.write_text(base_dsl, encoding="utf-8")
    lake_dir = tmp_path / "lake"
    spec = RegimeTargetSpec(
        regime_tag="reg_a",
        builder_candidates=("make_kernel",),
        sample_builder_kwargs={},
        editable_symbols=("make_kernel",),
        representative_anchor_dims={"M": 128},
        representative_hir_dims={"M": 128},
    )
    anchors = (AnchorCaseSpec("level2/demo_op", 1, "reg_a", 100.0, 5.0),)
    tracker = BayesianRuleTracker(rules={})
    flywheel = FlywheelDistiller(
        funnel=GateFunnel(skip_stage_b=True),
        tracker=tracker,
    )

    class _RealNPUL2Evaluator:
        is_real_npu = True

        def __call__(
            self,
            cands: Sequence[tuple[str, Path]],
            eval_anchors: Sequence[AnchorCaseSpec],
        ) -> dict[str, CandidateSurrogateResult]:
            out: dict[str, CandidateSurrogateResult] = {}
            for cid, mpath in cands:
                src = mpath.read_text(encoding="utf-8")
                lat = 100.0 if ("T.Kernel(1)" in src or "num_stages=2" not in src) else 91.5
                out[cid] = aggregate_surrogate_result(
                    cid,
                    "demo_op",
                    [build_anchor_metrics(a, accuracy_passed=True, cand_us=lat) for a in eval_anchors],
                )
            return out

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="demo_op",
            max_rounds=1,
            allow_unreviewed_distilled_rules=False,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("demo_op", 2.0, 1.5, 0.9),
        anchors=anchors,
        regime_specs=(spec,),
        tracker=tracker,
        flywheel_distiller=flywheel,
        l2_batch_evaluator=_RealNPUL2Evaluator(),
        experience_lake_dir=lake_dir,
    )
    ledger = _make_ledger(ii=20, blocks=8)
    monkeypatch.setattr(
        "tilefoundry_ascend.agent.flywheel.ablation_distiller.run_stage_a_static",
        lambda *_a, **_kw: StageAResult(passed=True, prim_func=None, ledger=ledger),
    )
    monkeypatch.setattr(
        flywheel._funnel,
        "evaluate",
        lambda *_a, **_kw: SimpleNamespace(
            passed=True,
            stage_a=StageAResult(passed=True, prim_func=None, ledger=ledger),
            stage_c=None,
            failed_stage="",
            error_message="",
        ),
    )

    runner._flywheel_coordinator.seed_baseline("reg_a", base_dsl, 100.0, 0.5)
    admitted = runner._run_flywheel_distillation(
        baseline_source=base_dsl,
        winner_source=win_dsl,
        baseline_latency_us=94.0,
        winner_latency_us=91.5,
        func_name="make_kernel",
        regime_spec=spec,
        active_anchors=anchors,
        topology_axes=TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd"),
        rel_gain=0.0266,
        local_rel_gain=0.0266,
        worker_kind="creative_explore",
        round_idx=3,
        rule_id=None,
        symptom_family="GRID_TAIL_WAVE_UNDERUTIL",
        sigma_paired_us=0.5,
    )
    assert admitted
    distilled_files = sorted((lake_dir / "kb" / "distilled_rules").glob("*.json"))
    assert len(distilled_files) == len(admitted)
    assert (
        runner._flywheel_coordinator.last_distilled_ckpt_by_regime["reg_a"].latency_us
        == pytest.approx(91.5)
    )


def test_bandit_launch_floor_detrapping_and_sticky_gate(tmp_path: Path) -> None:
    """Launch-floor saturated small_decode is deprioritized below regimes with excess > 5us."""
    target_path = tmp_path / "kernels.py"
    target_path.write_text(_base_module_src(), encoding="utf-8")
    anchors = (
        AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 15.0, 1.5),
        AnchorCaseSpec("level4/attention_residual", 16, "aligned_prefill", 80.0, 20.0),
    )
    regimes = (
        RegimeTargetSpec(
            regime_tag="small_decode",
            builder_candidates=("_build_small",),
            sample_builder_kwargs={},
            representative_anchor_dims={"B": 1, "S": 1, "D": 4096},
        ),
        RegimeTargetSpec(
            regime_tag="aligned_prefill",
            builder_candidates=("_build_prefill",),
            sample_builder_kwargs={},
            representative_anchor_dims={"B": 1, "S": 128, "D": 4096},
        ),
    )
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="attention_residual", max_rounds=2),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 5.0, 2.0, 0.90),
        anchors=anchors,
        regime_specs=regimes,
    )

    best_res = aggregate_surrogate_result(
        "base",
        "attention_residual",
        (
            build_anchor_metrics(anchors[0], accuracy_passed=True, cand_us=12.0),
            build_anchor_metrics(anchors[1], accuracy_passed=True, cand_us=40.0),
        ),
    )
    picked = runner._select_active_regime(None, set(), best_res, anchors)
    assert picked is not None
    assert picked.regime_tag == "aligned_prefill"

    runner._last_round_promoted_by_regime["small_decode"] = False
    picked_no_sticky = runner._select_active_regime("small_decode", set(), best_res, anchors)
    assert picked_no_sticky is not None
    assert picked_no_sticky.regime_tag == "aligned_prefill"


@pytest.mark.anyio
async def test_substage_timings_prewarm_stats_and_48kb_trackb_passthrough(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timings, prewarm_stats, advisor predictions, and 48KB Track-B budget wire cleanly."""
    cfg_default = CampaignConfig(operator_name="attention_residual")
    assert cfg_default.cost_model_mode == "off"
    assert cfg_default.max_syntax_repairs == 1
    assert cfg_default.enable_phase0_lattice_sweep is False

    big_cpp_body = "  int x = 1;\n" * 2400
    assert 20000 < len(big_cpp_body.encode("utf-8")) < 49152
    trackb_src = (
        '_MLA_TRACKB_SRC = r"""\n'
        "#include <tl_templates/ascend/common.h>\n"
        "__global__ __vector__ void tb_kernel(__gm__ float* p) {\n"
        f"{big_cpp_body}"
        "}\n"
        '"""\n'
    )
    tb_spec = RegimeTargetSpec(
        regime_tag="split_kv_decode",
        builder_candidates=("_build_mla",),
        sample_builder_kwargs={},
        runtime_backend="hybrid_trackb",
        trackb_symbols=("_MLA_TRACKB_SRC",),
    )
    compacted = _compact_trackb_snippet_source(
        trackb_src,
        tb_spec,
        incumbent_source=trackb_src,
        max_bytes=14800,
    )
    assert "MIDDLE BODY OMITTED" not in compacted
    assert "LINES OMITTED" not in compacted
    assert "/* TRACK-B SCOPED SLICE: symbol=_MLA_TRACKB_SRC active=tb_kernel */" in compacted
    assert "[L0002] #include <tl_templates/ascend/common.h>" in compacted
    assert "[L2404] }" in compacted

    target_path = tmp_path / "kernels.py"
    target_path.write_text(_base_module_src(), encoding="utf-8")
    anchors = _five_anchors()
    regimes = _three_regimes()
    seen_l1_tail_dims: list[object] = []

    def _l1_spy(
        _src: str,
        _reg: RegimeTargetSpec,
        *,
        tail_anchor_dims: object = None,
    ) -> StageBResult:
        seen_l1_tail_dims.append(tail_anchor_dims)
        return StageBResult(passed=True)

    @dataclass
    class _TimedL2Eval:
        def __call__(
            self,
            cands: Sequence[tuple[str, Path]],
            eval_anchors: Sequence[AnchorCaseSpec],
        ) -> L2BatchResults:
            res_map = {
                cid: aggregate_surrogate_result(
                    cid,
                    "attention_residual",
                    [build_anchor_metrics(a, accuracy_passed=True, cand_us=10.0) for a in eval_anchors],
                )
                for cid, _p in cands
            }
            out = L2BatchResults(res_map, {cid: "run-1" for cid in res_map})
            setattr(
                out,
                "prewarm_stats",
                {
                    "total_tasks": 5,
                    "cache_hits": 3,
                    "compiled_tasks": 2,
                    "failed_candidates": 0,
                },
            )
            setattr(out, "l2_prewarm_sec", 1.25)
            setattr(out, "l2_npu_sec", 2.50)
            return out

    class _RepairLLM:
        async def generate(self, req: GenerationRequest, *, effort_override: object = None) -> str:
            _ = effort_override
            if req.role == "syntax_repair":
                return "def _build_small(tile_m=2):\n    return tile_m\n"
            if req.role == "explore_creative":
                return (
                    "# target_bottleneck: aiv\n"
                    "# structural_hypothesis: opt\n"
                    "```python\n"
                    "def _build_small(tile_m=bad_syntax(\n"
                    "```\n"
                )
            return "def _build_small(tile_m=2):\n    return tile_m\n"

    ckpt_path = tmp_path / "ckpt.json"
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            max_rounds=1,
            k_rule_workers=0,
            max_syntax_repairs=1,
            cost_model_mode="off",
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 5.0, 2.0, 0.99),
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=_RepairLLM(),
        tracker=BayesianRuleTracker(rules={}),
        l0_checker=lambda _s, _r: StageAResult(passed=True, prim_func=None, ledger=_make_ledger()),
        l1_compiler=_l1_spy,
        l2_batch_evaluator=_TimedL2Eval(),
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        checkpoint_path=ckpt_path,
    )
    _patch_env(monkeypatch, runner)
    base_res = runner._evaluate_single_baseline(_base_module_src(), anchors)

    rec, *_ = await runner._execute_single_round(
        round_idx=1,
        regime_spec=regimes[0],
        incumbent_source=_base_module_src(),
        best_result=base_res,
        active_anchors=anchors,
        stagnation_rounds=0,
        admitted_rule_ids=[],
    )
    assert seen_l1_tail_dims and seen_l1_tail_dims[0] == {"B": 1, "S": 1, "D": 64}
    assert set(rec.stage_timings_sec.keys()) == {
        "llm_gen_sec",
        "l0_l1_sec",
        "syntax_repair_sec",
        "l2_prewarm_sec",
        "l2_npu_sec",
        "l3_sec",
        "post_l3_sec",
    }
    assert rec.stage_timings_sec["l2_prewarm_sec"] == pytest.approx(1.25)
    assert rec.stage_timings_sec["l2_npu_sec"] == pytest.approx(2.50)
    assert rec.prewarm_stats == {
        "total_tasks": 5,
        "cache_hits": 3,
        "compiled_tasks": 2,
        "failed_candidates": 0,
    }
    assert rec.cost_advisor_predictions == 0

    funnel_path = tmp_path / "telemetry" / "round_funnel_events.jsonl"
    assert funnel_path.is_file()
    lines = [json.loads(ln) for ln in funnel_path.read_text(encoding="utf-8").splitlines() if ln]
    assert lines
    assert lines[0]["prewarm_stats"] == rec.prewarm_stats
    assert lines[0]["stage_timings_sec"]["l2_prewarm_sec"] == pytest.approx(1.25)
    assert lines[0]["cost_advisor_predictions"] == 0

    round_trip = campaign_runner.CampaignRoundRecord.from_dict(rec.to_dict())
    assert round_trip.prewarm_stats == rec.prewarm_stats
    assert round_trip.stage_timings_sec == rec.stage_timings_sec
    assert round_trip.cost_advisor_predictions == 0

    auto_runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="attention_residual",
            max_rounds=1,
            k_rule_workers=0,
            max_syntax_repairs=0,
            cost_model_mode="advisory",
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("attention_residual", 5.0, 2.0, 0.99),
        anchors=anchors,
        regime_specs=regimes,
        llm_backend=_RepairLLM(),
        tracker=BayesianRuleTracker(rules={}),
        l0_checker=lambda _s, _r: StageAResult(passed=True, prim_func=None, ledger=_make_ledger()),
        l1_compiler=lambda _s, _r: StageBResult(passed=True),
        l2_batch_evaluator=_TimedL2Eval(),
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        checkpoint_path=tmp_path / "ckpt_auto.json",
    )
    _patch_env(monkeypatch, auto_runner)
    auto_rec, *_ = await auto_runner._execute_single_round(
        round_idx=1,
        regime_spec=regimes[0],
        incumbent_source=_base_module_src(),
        best_result=base_res,
        active_anchors=anchors,
        stagnation_rounds=0,
        admitted_rule_ids=[],
    )
    assert auto_rec.cost_advisor_predictions > 0

    evt = RoundFunnelEvent(
        run_id="r1",
        operator="attention_residual",
        round_idx=1,
        regime_tag="small_decode",
        topology_axes={},
        candidate_id="c1",
        cand_sha="abc",
        worker_kind="creative_explore",
        stage_timings_sec=rec.stage_timings_sec,
        prewarm_stats=rec.prewarm_stats,
        cost_advisor_predictions=auto_rec.cost_advisor_predictions,
    )
    assert evt.to_dict()["cost_advisor_predictions"] == auto_rec.cost_advisor_predictions
