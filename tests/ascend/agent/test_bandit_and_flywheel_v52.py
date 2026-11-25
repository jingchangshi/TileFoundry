"""Unit tests for v5.2 anchor selector fallback, Stage-A calibration, Bandit, and Flywheel."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from tilefoundry_ascend.agent.campaign import campaign_runner, surrogate_cannbench
from tilefoundry_ascend.agent.campaign.anchor_specs import (
    AnchorCaseSpec,
    CannbenchOpsRegistry,
    PerfTargetSpec,
    RegimeTargetSpec,
    build_semantic_regime_specs,
)
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CampaignConfig,
    PerfTargetCampaignRunner,
    patch_regime_into_module,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    L3AcceptanceResult,
    aggregate_surrogate_result,
    build_anchor_metrics,
)
from tilefoundry_ascend.agent.campaign.transfer_audit import summarize_funnel_and_transfer
from tilefoundry_ascend.agent.flywheel.ablation_distiller import (
    FlywheelDistiller,
    _validate_distilled_rule,
    decompose_leave_one_out_ablations,
    distill_causal_rule,
)
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.kb.schema import AscendCausalRule, SymptomFamily
from tilefoundry_ascend.agent.orchestrator import (
    cpp_slice_protocol,
    tilesize_prescreen,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    GateFunnel,
    StageAResult,
    StageBResult,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import GenerationRequest
from tilefoundry_ascend.ascend.cost_model.experience_lake import ExperienceLake
from tilefoundry_ascend.ascend.cost_model.hybrid_cost_model import CostPrediction
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger
from tilefoundry_ascend.ascend.sil.pytorch_frontend import PyTorchSemanticAnalyzer
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes


def _make_anchor_metric(
    *,
    case_num: int,
    regime_tag: str,
    cand_us: float,
    ref_perf_us: float = 10.0,
    t_hw_us: float = 2.0,
    accuracy_passed: bool = True,
    eq3_score: float = 0.7,
) -> AnchorEvalMetrics:
    speedup = (ref_perf_us / cand_us) if (accuracy_passed and cand_us > 0.0) else 0.0
    eta = (t_hw_us / cand_us) if (accuracy_passed and cand_us > 0.0) else 0.0
    return AnchorEvalMetrics(
        case_num=case_num,
        regime_tag=regime_tag,
        accuracy_passed=accuracy_passed,
        cand_us=cand_us,
        ref_perf_us=ref_perf_us,
        t_hw_us=t_hw_us,
        speedup=speedup,
        roofline_eta=eta,
        eq3_score=eq3_score if accuracy_passed else 0.0,
    )


def _make_ledger(*, ii: int = 20, blocks: int = 16) -> object:
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


def _make_cost_pred(pred_us: float, tier0_us: float, tier1_us: float = 0.0) -> CostPrediction:
    return CostPrediction(
        pred_us=pred_us,
        tier0_physics_us=tier0_us,
        tier1_penalty_us=tier1_us,
        tier2_residual_factor=1.0,
        cube_floor_us=0.0,
        vec_floor_us=tier0_us,
        hbm_floor_us=0.0,
        l2_hit_rate=0.9,
        wave_quantization_ratio=1.0,
        bank_conflict_us=0.0,
        effective_hbm_bw_bps=1.0e12,
        dominant_bound="vec",
    )


def _patch_prompt_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(campaign_runner, "scan_tilelang_pipe_gaps", lambda: None)
    from tilefoundry_ascend.agent.orchestrator import llm_constraints

    monkeypatch.setattr(llm_constraints, "format_target_dialect_layer", lambda *_a, **_kw: "")


def test_a_anchor_specs_shared_selector_fallback_and_hybrid_regime_backend() -> None:
    """Multi-regime modules fall back to shared selector and classify guarded hybrid regimes."""
    reg = CannbenchOpsRegistry.default()
    gmm_specs = reg.get_regime_specs("grouped_matmul")
    assert len(gmm_specs) >= 2
    for spec in gmm_specs:
        assert spec.config_selector_fn == "_select_grouped_matmul_config"
        assert "_select_grouped_matmul_config" in spec.editable_symbols
        assert "_render_trackb_tu" in spec.editable_symbols
        assert "_render_gmm_trackb_launcher" in spec.editable_symbols
        assert f"_select_grouped_matmul_{spec.regime_tag}_config" not in spec.editable_symbols

    golden = """
import torch

def demo_kda(q, k, v):
    return q + k + v
"""
    mod_text = '''
_TRACK_B_SRC = r"""
#include <tl_templates/ascend/common.h>
__global__ __vector__ void kda_kernel() {}
"""

def _select_demo_kda_config(dk: int = 128, dv: int = 128):
    return {"chunk_t": 32}

def _build_demo_kda_prim_func(dk: int = 128, dv: int = 128):
    return None

def _render_trackb_tu(**dims):
    return _TRACK_B_SRC

def _render_demo_kda_trackb_launcher(**dims):
    return _TRACK_B_SRC

def _get_track_b_lib():
    return None

def _run_track_b(q, k, v):
    return _get_track_b_lib()

def demo_kda(q, k, v):
    dk = q.shape[-1]
    if dk <= 128 and v.shape[-1] <= 128:
        return _run_track_b(q, k, v)
    return tilelang.compile(None)
'''
    cases = [
        {"case_id": 1, "input_shape": [[1, 64, 16, 128], [1, 64, 16, 128], [1, 64, 16, 128]], "dtype": ["float16", "float16", "float16"]},
        {"case_id": 2, "input_shape": [[1, 64, 16, 256], [1, 64, 16, 256], [1, 64, 16, 256]], "dtype": ["float16", "float16", "float16"]},
    ]
    proto = {"inputs": [{"name": "q"}, {"name": "k"}, {"name": "v"}], "outputs": [{"name": "y"}]}
    analyzer = PyTorchSemanticAnalyzer()
    sem = analyzer.analyze_source(golden, entry_name="demo_kda", proto=proto)
    ratios = tuple(analyzer.compute_case_ratios(sem, c) for c in cases)
    specs = build_semantic_regime_specs(
        operator_name="demo_kda",
        entrypoint="demo_kda",
        cases=cases,
        proto=proto,
        base_axes=sem.base_axes,
        has_reduction=sem.has_reduction,
        case_ratios=ratios,
        runtime_backend="hybrid_trackb",
        trackb_symbols=("_TRACK_B_SRC",),
        module_text=mod_text,
    )
    assert len(specs) == 2
    by_dk = {int(s.representative_anchor_dims["q_dim_3"]): s for s in specs}
    assert by_dk[128].runtime_backend == "hybrid_trackb"
    assert "_select_demo_kda_config" in by_dk[128].editable_symbols
    assert "_render_trackb_tu" in by_dk[128].editable_symbols
    assert "_render_demo_kda_trackb_launcher" in by_dk[128].editable_symbols
    assert by_dk[256].runtime_backend == "tilelang_dsl"
    assert by_dk[256].trackb_symbols == ()


@pytest.mark.anyio
async def test_b_stage_a_uses_calibrated_pred_us(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage-A coarse prescreen and telemetry use calibrated pred_us over raw tier0+tier1."""
    target_path = tmp_path / "kernels.py"
    base_src = (
        "def _select_op_config(**_kw):\n"
        "    return {'tile_m': 4}\n\n"
        "def _build_op(tile_m=4):\n"
        "    return tile_m\n"
    )
    target_path.write_text(base_src, encoding="utf-8")
    spec = RegimeTargetSpec(
        regime_tag="reg_a",
        builder_candidates=("_build_op",),
        sample_builder_kwargs={"tile_m": 4},
        editable_symbols=("_select_op_config", "_build_op"),
        config_selector_fn="_select_op_config",
        representative_anchor_dims={"M": 64, "D": 64},
        representative_hir_dims={"M": 64, "D": 64},
    )
    anchors = (AnchorCaseSpec("level2/demo_op", 1, "reg_a", 20.0, 4.0),)
    lake_dir = tmp_path / "lake"

    class _FakeLLM:
        async def generate(self, req: GenerationRequest, *, effort_override: object = None) -> str:
            _ = effort_override
            if req.role == "explore_creative":
                return json.dumps(
                    {
                        "physical_hypothesis": "test tile",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": "def _build_op(tile_m=32):\n    return tile_m + 3\n",
                    }
                )
            if "R_SLOW" in req.user_prompt:
                return "def _build_op(tile_m=8):\n    return tile_m + 1\n"
            return "def _build_op(tile_m=16):\n    return tile_m + 2\n"

    ledger_parent = _make_ledger(ii=100)
    ledger_cand_slow = _make_ledger(ii=110)
    ledger_cand_fast = _make_ledger(ii=120)

    def _l0_check(src: str, _r: RegimeTargetSpec) -> StageAResult:
        if "tile_m=8" in src:
            return StageAResult(passed=True, prim_func=None, ledger=ledger_cand_slow)
        if "tile_m=16" in src or "tile_m=32" in src:
            return StageAResult(passed=True, prim_func=None, ledger=ledger_cand_fast)
        return StageAResult(passed=True, prim_func=None, ledger=ledger_parent)

    def _l2_eval(
        cands: Sequence[tuple[str, Path]],
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> dict[str, CandidateSurrogateResult]:
        return {
            cid: aggregate_surrogate_result(
                cid,
                "demo_op",
                [
                    _make_anchor_metric(
                        case_num=a.case_num,
                        regime_tag=a.regime_tag,
                        cand_us=15.0,
                        ref_perf_us=a.ref_perf_us,
                        t_hw_us=a.t_hw_us,
                    )
                    for a in active_anchors
                ],
            )
            for cid, _p in cands
        }

    r_slow = AscendCausalRule(
        rule_id="R_SLOW",
        regime_guard="is_legal",
        precondition="PIPE_V > 0",
        transform_template="Try tile_m=8",
        expected_postcondition="measured_latency_us < prev.measured_latency_us",
        source_provenance="test",
        symptom_family=SymptomFamily.COMPUTE_INEFFECTIVE_THROUGHPUT.value,
    )
    r_fast = AscendCausalRule(
        rule_id="R_FAST",
        regime_guard="is_legal",
        precondition="PIPE_V > 0",
        transform_template="Try tile_m=16",
        expected_postcondition="measured_latency_us < prev.measured_latency_us",
        source_provenance="test",
        symptom_family=SymptomFamily.COMPUTE_INEFFECTIVE_THROUGHPUT.value,
    )

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="demo_op",
            max_rounds=1,
            k_rule_workers=2,
            enable_tilesize_prescreen=True,
            cost_prescreen_top_k=1,
            cost_model_mode="auto",
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("demo_op", 2.0, 1.5, 0.90),
        anchors=anchors,
        regime_specs=(spec,),
        llm_backend=_FakeLLM(),
        tracker=BayesianRuleTracker.from_rules([r_slow, r_fast]),
        l0_checker=_l0_check,
        l1_compiler=lambda _s, _r: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=False),
        checkpoint_path=tmp_path / "ckpt.json",
        experience_lake_dir=lake_dir,
    )
    _patch_prompt_env(monkeypatch)
    monkeypatch.setattr(runner, "_collect_msprof_for_regimes", lambda *_a, **_kw: None)

    def _calibrated_predict(features: object, bisheng_res: object = None) -> CostPrediction:
        _ = bisheng_res
        ii = float(
            features.get("steady_state_ii", 100.0)
            if isinstance(features, Mapping)
            else getattr(features, "steady_state_ii", 100.0)
        )
        if ii >= 115.0:
            return _make_cost_pred(5.0, 500.0, 500.0)
        if ii >= 105.0:
            return _make_cost_pred(500.0, 10.0, 10.0)
        return _make_cost_pred(50.0, 50.0, 0.0)

    monkeypatch.setattr(runner._cost_model, "predict", _calibrated_predict)
    monkeypatch.setattr(
        campaign_runner,
        "rank_stage_a_candidates_with_cost_model",
        lambda passers, **_kw: sorted(
            [
                SimpleNamespace(
                    candidate_id=cid,
                    pred_us=(5.0 if "tile_m=16" in src else 500.0),
                )
                for cid, src, _sa in passers
            ],
            key=lambda item: item.pred_us,
        ),
    )

    best_result = aggregate_surrogate_result(
        "base",
        "demo_op",
        [_make_anchor_metric(case_num=1, regime_tag="reg_a", cand_us=15.0, ref_perf_us=20.0, t_hw_us=4.0)],
    )
    rec, *_rest = await runner._execute_single_round(
        round_idx=1,
        regime_spec=spec,
        incumbent_source=base_src,
        best_result=best_result,
        active_anchors=anchors,
        stagnation_rounds=0,
        admitted_rule_ids=[],
    )
    assert len(rec.prescreen_pruned_ids) == 1
    assert len(rec.l0_l1_passed_ids) == 2

    obs_records = [
        r for r in ExperienceLake(lake_dir)._load_observations() if r.operator == "demo_op"
    ]
    assert obs_records
    assert all(r.analytical_pred_us in (5.0, 50.0, 500.0) for r in obs_records)


@pytest.mark.anyio
async def test_c_bandit_scheduler_and_compile_wall_regime_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bandit ranks by Amdahl headroom and switches away after 2 compile-wall rounds."""
    target_path = tmp_path / "kernels.py"
    base_src = (
        "def _build_reg_a():\n    return 1\n\n"
        "def _build_reg_b():\n    return 2\n"
    )
    target_path.write_text(base_src, encoding="utf-8")
    spec_a = RegimeTargetSpec(
        regime_tag="reg_a",
        builder_candidates=("_build_reg_a",),
        sample_builder_kwargs={},
        editable_symbols=("_build_reg_a",),
    )
    spec_b = RegimeTargetSpec(
        regime_tag="reg_b",
        builder_candidates=("_build_reg_b",),
        sample_builder_kwargs={},
        editable_symbols=("_build_reg_b",),
    )
    anchors = (
        AnchorCaseSpec("level2/demo_op", 1, "reg_a", 100.0, 5.0),
        AnchorCaseSpec("level2/demo_op", 2, "reg_a", 100.0, 5.0),
        AnchorCaseSpec("level2/demo_op", 3, "reg_b", 50.0, 5.0),
    )

    class _WallAndPromoteLLM:
        async def generate(self, req: GenerationRequest, *, effort_override: object = None) -> str:
            _ = effort_override
            if "reg_b" in req.user_prompt:
                return "def _build_reg_b():\n    return 20\n"
            return "def _build_reg_a():\n    return invalid_syntax_marker(\n"

    def _l2_eval(
        cands: Sequence[tuple[str, Path]],
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> dict[str, CandidateSurrogateResult]:
        out: dict[str, CandidateSurrogateResult] = {}
        for cid, mpath in cands:
            src = mpath.read_text(encoding="utf-8")
            b_fast = "return 20" in src
            out[cid] = aggregate_surrogate_result(
                cid,
                "demo_op",
                [
                    build_anchor_metrics(
                        a,
                        accuracy_passed=True,
                        cand_us=(
                            50.0
                            if a.regime_tag == "reg_a"
                            else (10.0 if b_fast else 15.0)
                        ),
                    )
                    for a in active_anchors
                ],
            )
        return out

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="demo_op",
            max_rounds=4,
            k_rule_workers=0,
            max_syntax_repairs=0,
            stagnation_freeze_rounds=6,
            enable_tilesize_prescreen=False,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("demo_op", 10.0, 5.0, 0.99),
        anchors=anchors,
        regime_specs=(spec_a, spec_b),
        llm_backend=_WallAndPromoteLLM(),
        tracker=BayesianRuleTracker(rules={}),
        l0_checker=lambda _s, _r: StageAResult(passed=True, prim_func=None, ledger=_make_ledger()),
        l1_compiler=lambda _s, _r: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        checkpoint_path=tmp_path / "ckpt.json",
        experience_lake_dir=tmp_path / "lake",
    )
    _patch_prompt_env(monkeypatch)
    monkeypatch.setattr(runner, "_collect_msprof_for_regimes", lambda *_a, **_kw: None)

    best_for_bandit = _l2_eval([("base", target_path)], anchors)["base"]
    picked = runner._select_active_regime(None, set(), best_for_bandit, anchors)
    assert picked is not None and picked.regime_tag == "reg_a"

    runner._compile_wall_streak_by_regime = {"reg_a": 2, "reg_b": 0}
    switched = runner._select_active_regime(None, set(), best_for_bandit, anchors)
    assert switched is not None and switched.regime_tag == "reg_b"

    runner._compile_wall_streak_by_regime = {"reg_a": 3, "reg_b": 0}
    frozen_pick = runner._select_active_regime(
        None, set(), best_for_bandit, anchors, exclude_regime="reg_b"
    )
    assert frozen_pick is not None and frozen_pick.regime_tag == "reg_b"

    runner._compile_wall_streak_by_regime = {}
    stag_pick = runner._select_active_regime(
        None,
        set(),
        best_for_bandit,
        anchors,
        stagnation_by_regime={"reg_a": 16, "reg_b": 0},
    )
    assert stag_pick is not None and stag_pick.regime_tag == "reg_b"

    report = await runner.run()
    assert [r.regime_tag for r in report.rounds] == ["reg_a", "reg_b", "reg_a", "reg_b"]
    assert runner._compile_wall_streak_by_regime["reg_a"] == 2


def test_d_real_npu_flywheel_auto_persists_distilled_rule_with_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real NPU path (_l2_batch_evaluator=None, _env_file!=None) auto-saves kb/distilled_rules."""
    target_path = tmp_path / "kernels.py"
    base_dsl = """
import tilelang.ascend.language as T

def make_kernel():
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(1) as bx:
            a_ub = T.alloc_shared((128,), "float32")
            T.copy(A[0:128], a_ub)
            T.copy(a_ub, B[0:128])
    return main
"""
    win_dsl = """
import tilelang.ascend.language as T

def make_kernel():
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(8) as bx:
            a_ub = T.alloc_shared((128,), "float32")
            for k in T.Pipelined(2, num_stages=2):
                T.copy(A[0:128], a_ub)
                T.copy(a_ub, B[0:128])
    return main
"""
    target_path.write_text(base_dsl, encoding="utf-8")
    env_file = tmp_path / "npu.env"
    env_file.write_text("TILEFOUNDRY_NPU_CANNBENCH_REPO=/tmp/cb\n", encoding="utf-8")
    lake_dir = tmp_path / "lake"
    spec = RegimeTargetSpec(
        regime_tag="reg_a",
        builder_candidates=("make_kernel",),
        sample_builder_kwargs={},
        editable_symbols=("make_kernel",),
        representative_anchor_dims={"M": 128},
        representative_hir_dims={"M": 128},
    )
    anchors = (AnchorCaseSpec("level2/demo_op", 1, "reg_a", 40.0, 5.0),)
    tracker = BayesianRuleTracker(rules={})
    flywheel = FlywheelDistiller(
        funnel=GateFunnel(skip_stage_b=True),
        tracker=tracker,
        min_edit_gain=0.03,
    )
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="demo_op",
            max_rounds=1,
            allow_unreviewed_distilled_rules=False,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("demo_op", 2.0, 1.5, 0.9),
        env_file=env_file,
        anchors=anchors,
        regime_specs=(spec,),
        tracker=tracker,
        flywheel_distiller=flywheel,
        l2_batch_evaluator=None,
        experience_lake_dir=lake_dir,
    )
    assert runner._l2_batch_evaluator is None
    assert runner._env_file == env_file.resolve()

    def _fake_npu_batch(
        _op_name: str,
        cands: Sequence[tuple[str, Path]],
        active_anchors: Sequence[AnchorCaseSpec],
        **_kw: object,
    ) -> tuple[dict[str, CandidateSurrogateResult], str, str]:
        out: dict[str, CandidateSurrogateResult] = {}
        for cid, mpath in cands:
            src = mpath.read_text(encoding="utf-8")
            lat = 40.0 if ("T.Kernel(1)" in src or "num_stages=2" not in src) else 10.0
            out[cid] = aggregate_surrogate_result(
                cid,
                "demo_op",
                [
                    build_anchor_metrics(
                        a,
                        accuracy_passed=True,
                        cand_us=lat,
                    )
                    for a in active_anchors
                ],
            )
        return out, "run-npu-1", "receipt.json"

    monkeypatch.setattr(campaign_runner, "run_surrogate_batch_via_agent_gates", _fake_npu_batch)
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

    admitted = runner._run_flywheel_distillation(
        baseline_source=base_dsl,
        winner_source=win_dsl,
        baseline_latency_us=40.0,
        winner_latency_us=10.0,
        func_name="make_kernel",
        regime_spec=spec,
        active_anchors=anchors,
        topology_axes=TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd"),
        rel_gain=0.75,
        worker_kind="creative_explore",
        round_idx=1,
        rule_id=None,
        symptom_family="GRID_TAIL_WAVE_UNDERUTIL",
    )
    assert admitted
    distilled_files = sorted((lake_dir / "kb" / "distilled_rules").glob("*.json"))
    assert len(distilled_files) == len(admitted)
    saved_payload = json.loads(distilled_files[0].read_text(encoding="utf-8"))
    assert saved_payload["origin_operator"] == "demo_op"
    assert float(saved_payload["origin_timestamp"]) > 0.0

    edge_lines = (lake_dir / "kb" / "evidence" / "winning_edges.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(edge_lines) == 1
    edge = json.loads(edge_lines[0])
    assert edge["origin_operator"] == "demo_op"
    assert float(edge["origin_timestamp"]) > 0.0
    assert edge["distilled_rule_ids"] == admitted


def test_e_transfer_audit_excludes_seed_rules_and_pre_origin_events(tmp_path: Path) -> None:
    """Transfer matrix excludes seed rules and causal-filters same-op events <= origin_ts."""
    lake_dir = tmp_path / "lake"
    rules_dir = lake_dir / "kb" / "distilled_rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    (rules_dir / "R_FLYWHEEL_01.json").write_text(
        json.dumps(
            {
                "rule_id": "R_FLYWHEEL_01",
                "origin_operator": "op_src",
                "origin_timestamp": 1000.0,
                "topology_axes": {
                    "compute_engine": "aiv_only",
                    "memory_ring": "gm_ub_stream",
                    "iteration_regularity": "dense_static",
                },
            }
        ),
        encoding="utf-8",
    )
    ev_dir = lake_dir / "kb" / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)
    (ev_dir / "winning_edges.jsonl").write_text(
        json.dumps(
            {
                "operator": "op_src",
                "rule_id": "R08_SEED_RULE",
                "staged_rule_ids": ["R_FLYWHEEL_01"],
                "timestamp": 1000.0,
                "topology_axes": {
                    "compute_engine": "aiv_only",
                    "memory_ring": "gm_ub_stream",
                    "iteration_regularity": "dense_static",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    axes = {
        "compute_engine": "aiv_only",
        "memory_ring": "gm_ub_stream",
        "iteration_regularity": "dense_static",
    }
    events = [
        {
            "operator": "op_other",
            "rule_id": "R08_SEED_RULE",
            "topology_axes": axes,
            "promoted_l3": True,
            "delta_speedup": 0.40,
            "timestamp": 2000.0,
        },
        {
            "operator": "op_src",
            "rule_id": "R_FLYWHEEL_01",
            "topology_axes": axes,
            "promoted_l3": True,
            "delta_speedup": 0.50,
            "timestamp": 900.0,
        },
        {
            "operator": "op_src",
            "rule_id": "R_FLYWHEEL_01",
            "topology_axes": axes,
            "promoted_l3": True,
            "delta_speedup": 0.20,
            "timestamp": 1100.0,
        },
        {
            "operator": "op_target",
            "rule_id": "R_FLYWHEEL_01",
            "topology_axes": axes,
            "promoted_l3": True,
            "delta_speedup": 0.30,
            "timestamp": 950.0,
        },
    ]
    report = summarize_funnel_and_transfer(events, experience_lake_dir=lake_dir)
    matrix = report["distilled_rule_transfer_matrix"]
    assert isinstance(matrix, list)
    rids = {row["rule_id"] for row in matrix}
    assert "R08_SEED_RULE" not in rids
    assert rids == {"R_FLYWHEEL_01"}
    by_target = {row["target_operator"]: row for row in matrix}
    assert by_target["op_src"]["dispatches"] == 1
    assert by_target["op_src"]["first_hit_delta_speedup"] == 0.2
    assert by_target["op_target"]["dispatches"] == 1


def test_f_trackb_malformed_search_replace_raises_patch_anchor_miss() -> None:
    """Malformed SEARCH/REPLACE markers emit objective PATCH_ANCHOR_MISS with unit anchors."""
    mod_src = '''
_DEMO_TRACKB_SRC = r"""
#include <tl_templates/ascend/common.h>
__global__ __vector__ void demo_tb(__gm__ half* x) {
  int first_anchor = 1;
  int last_anchor = 2;
}
"""

def _select_demo_config():
    return {"bm": 64}
'''
    spec = RegimeTargetSpec(
        regime_tag="reg_a",
        builder_candidates=("_build_demo",),
        sample_builder_kwargs={},
        editable_symbols=("_select_demo_config", "_DEMO_TRACKB_SRC"),
        config_selector_fn="_select_demo_config",
        runtime_backend="ascendc_trackb",
        trackb_symbols=("_DEMO_TRACKB_SRC",),
    )
    malformed = "<<<<<<< SEARCH\nint missing_end = 1;\n"
    expected_exc = getattr(cpp_slice_protocol, "PatchAnchorMissError", ValueError)
    with pytest.raises(expected_exc) as exc_info:
        patch_regime_into_module(mod_src, spec, malformed)
    msg = str(exc_info.value)
    assert "PATCH_ANCHOR_MISS:" in msg
    assert "demo_tb" in msg
    assert "first_anchor" in msg


def test_g_phase0_lattice_warmstart_and_paired_l2_incumbent_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase-0 sweeps lattice without LLM and _run_l2_batch strips __incumbent__ slot."""
    target_path = tmp_path / "kernels.py"
    base_src = (
        "def _select_op_config(**_kw):\n"
        "    return {'tile_m': 4}\n\n"
        "def _build_op(tile_m=4):\n"
        "    return tile_m\n"
    )
    fast_src = (
        "def _select_op_config(**_kw):\n"
        "    return {'tile_m': 16}\n\n"
        "def _build_op(tile_m=16):\n"
        "    return tile_m\n"
    )
    target_path.write_text(base_src, encoding="utf-8")
    spec = RegimeTargetSpec(
        regime_tag="reg_a",
        builder_candidates=("_build_op",),
        sample_builder_kwargs={"tile_m": 4},
        editable_symbols=("_select_op_config", "_build_op"),
        config_selector_fn="_select_op_config",
        representative_anchor_dims={"M": 64},
        representative_hir_dims={"M": 64},
    )
    anchors = (AnchorCaseSpec("level2/demo_op", 1, "reg_a", 20.0, 2.0),)
    seen_batch_cids: list[list[str]] = []

    def _l2_eval(
        cands: Sequence[tuple[str, Path]],
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> dict[str, CandidateSurrogateResult]:
        seen_batch_cids.append([cid for cid, _p in cands])
        out: dict[str, CandidateSurrogateResult] = {}
        for cid, mpath in cands:
            src = mpath.read_text(encoding="utf-8")
            lat = 5.0 if "'tile_m': 16" in src else 10.0
            out[cid] = aggregate_surrogate_result(
                cid,
                "demo_op",
                [
                    build_anchor_metrics(
                        a,
                        accuracy_passed=True,
                        cand_us=lat,
                    )
                    for a in active_anchors
                ],
            )
        return out

    @dataclass(frozen=True)
    class _FakeLatticeCand:
        knob_overrides: dict[str, int]
        patched_source: str
        pred_us: float
        dominant_bound: str
        ub_bytes: int
        l1_bytes: int

    monkeypatch.setattr(
        tilesize_prescreen,
        "sweep_executable_lattice_zero_llm",
        lambda *_a, **_kw: [
            _FakeLatticeCand({"tile_m": 16}, fast_src, 5.0, "PIPE_V", 4096, 0)
        ],
        raising=False,
    )

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="demo_op",
            max_rounds=0,
            enable_tilesize_prescreen=True,
        ),
        target_module_path=target_path,
        perf_target=PerfTargetSpec("demo_op", 2.0, 1.5, 0.9),
        anchors=anchors,
        regime_specs=(spec,),
        l0_checker=lambda _s, _r: StageAResult(passed=True, prim_func=None, ledger=_make_ledger()),
        l1_compiler=lambda _s, _r: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        checkpoint_path=tmp_path / "ckpt.json",
    )
    monkeypatch.setattr(runner, "_collect_msprof_for_regimes", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        runner._cost_model,
        "predict",
        lambda *_a, **_kw: _make_cost_pred(10.0, 10.0, 0.0),
    )

    report = asyncio.run(runner.run())
    assert "'tile_m': 16" in target_path.read_text(encoding="utf-8")
    assert report.best_result.avg_speedup == pytest.approx(4.0)
    assert seen_batch_cids[0] == ["incumbent_baseline"]
    assert seen_batch_cids[1][0] == "incumbent_slot"
    assert "incumbent_slot" not in report.best_result.candidate_id
    assert "__incumbent__" not in report.best_result.candidate_id

    @dataclass(frozen=True)
    class _PairedAnchorMetrics(AnchorEvalMetrics):
        paired_speedup_vs_incumbent: float = 0.0

    runner._incumbent_anchor_us_by_case = {1: 10.0}
    raw_anc = build_anchor_metrics(anchors[0], accuracy_passed=True, cand_us=12.0)
    paired_anc = _PairedAnchorMetrics(
        case_num=raw_anc.case_num,
        regime_tag=raw_anc.regime_tag,
        accuracy_passed=raw_anc.accuracy_passed,
        cand_us=raw_anc.cand_us,
        ref_perf_us=raw_anc.ref_perf_us,
        t_hw_us=raw_anc.t_hw_us,
        speedup=raw_anc.speedup,
        roofline_eta=raw_anc.roofline_eta,
        eq3_score=raw_anc.eq3_score,
        paired_speedup_vs_incumbent=2.0,
    )
    paired_res = aggregate_surrogate_result("cand_p", "demo_op", [paired_anc])
    norm_map = runner._normalize_paired_l2_results({"cand_p": paired_res}, None)
    assert norm_map["cand_p"].anchors[0].cand_us == pytest.approx(5.0)
    assert norm_map["cand_p"].avg_speedup == pytest.approx(4.0)


def test_h_ablation_distiller_extracts_selector_and_trackb_template_chunks() -> None:
    """Distiller decomposes _select_*_config dicts and _render_trackb_tu template blocks."""
    base_mod = '''
_DEMO_TB_SRC = r"""
#include <tl_templates/ascend/common.h>
__global__ __vector__ void tb_k() {}
"""

def _select_demo_config():
    return {"bm": 64, "num_stages": 1}

def _render_trackb_tu(**dims):
    bm = 64
    return _DEMO_TB_SRC + f"""
extern "C" int launch_k() {{
  tb_k<{bm}><<<14, nullptr, stream>>>();
  return 0;
}}
"""
'''
    win_mod = '''
_DEMO_TB_SRC = r"""
#include <tl_templates/ascend/common.h>
__global__ __vector__ void tb_k() {}
"""

def _select_demo_config():
    return {"bm": 128, "num_stages": 2}

def _render_trackb_tu(**dims):
    bm = 128
    return _DEMO_TB_SRC + f"""
extern "C" int launch_k() {{
  tb_k<{bm}><<<56, nullptr, stream>>>();
  return 0;
}}
"""
'''
    chunks = decompose_leave_one_out_ablations(base_mod, win_mod)
    paths = [c.target_path for c in chunks]
    assert any("dict:bm" in p for p in paths)
    assert any("dict:num_stages" in p for p in paths)
    assert any("_render_trackb_tu" in p for p in paths)
    tu_chunks = [c for c in chunks if "_render_trackb_tu" in c.target_path]
    for ch in tu_chunks:
        rule = distill_causal_rule(ch, rule_id=f"R_{ch.chunk_id}")
        assert rule.compatible_backends == ("ascendc_trackb", "hybrid_trackb")
        ok, err = _validate_distilled_rule(rule)
        assert ok is True, err
