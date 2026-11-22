"""Unified Wave-5 E2E tests: SIL, Solver, Router, Templates, CostModel, and Marathon."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest
import torch

from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.hir.pipeline import PipelineSpec
from tilefoundry.ir.tir.memory import AllocTensor
from tilefoundry.ir.tir.stmts import Evaluate, For, LetStmt, Sequential
from tilefoundry.ir.types import DType, StorageKind, TensorType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry_ascend.agent.campaign.anchor_specs import (
    OPERATOR_METADATA,
    AnchorCaseSpec,
    CannbenchOpsRegistry,
    PerfTargetSpec,
    RegimeTargetSpec,
    default_source_root,
    get_default_anchors,
    get_default_regime_specs,
    get_operator_entrypoint,
    get_operator_module_relpath,
    get_operator_task,
)
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CampaignConfig,
    extract_regime_snippet,
    resolve_regime_config,
)
from tilefoundry_ascend.agent.campaign.marathon import (
    ExtremeCampaignRunner,
    FastCreativeAgyBackend,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    CandidateSurrogateResult,
    L3AcceptanceResult,
    aggregate_surrogate_result,
    build_anchor_metrics,
)
from tilefoundry_ascend.agent.flywheel.ablation_distiller import FlywheelDistiller
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.kb.schema import load_seed_rules
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    GateFunnel,
    StageAResult,
    StageBResult,
    StageCResult,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    GenerationRequest,
    MockLLMBackend,
)
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import PortfolioRunner
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    BeamCandidate,
    TwoLaneBeamSearch,
    compute_static_dynamic_residual,
    extract_structural_signature,
)
from tilefoundry_ascend.agent.orchestrator.twin_bridge import CannbenchContractOracle
from tilefoundry_ascend.agent.router.capability_gate import (
    route_from_op_spec,
)
from tilefoundry_ascend.ascend.cost_model import (
    HybridAscendCostModel,
)
from tilefoundry_ascend.ascend.ir.tir.instructions import AscendGmToUb, AscendUbToGm
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger
from tilefoundry_ascend.ascend.schedule.pipeline_solver import (
    extract_tir_loop_problem,
    solve_ascend_pipeline,
)
from tilefoundry_ascend.ascend.schedule.regime_solver import RegimePiecewiseSolver
from tilefoundry_ascend.ascend.sil import (
    AdvisoryEnvelope,
    OpAdvisoryExtractor,
    PolicyEnvelope,
    RegimeSlice,
    SchedulePolicy,
    SchedulingEnvelope,
    SILVerifier,
    verify_policy_envelope,
)
from tilefoundry_ascend.ascend.sil.schema import (
    VALID_COMPUTE_ENGINES,
    VALID_ITERATION_REGULARITIES,
    VALID_MEMORY_RINGS,
    TopologyAxes,
)
from tilefoundry_ascend.ascend.target import AscendTarget
from tilefoundry_ascend.ascend.templates import (
    get_template_registry,
    render_template,
)

IMPLEMENTED_OPS: tuple[str, ...] = (
    "attention_residual",
    "mla",
    "swi_glu",
    "rms_norm",
    "softmax",
    "add_rms_norm_dynamic_quant",
    "mhc_sinkhorn",
    "gqa",
    "kimi_delta_attention",
    "tensor_program_vm",
)


def _cannbench_tasks_dir() -> Path:
    return default_source_root().parent / "cann-bench" / "tasks"


def _make_static_ub_loop(*, max_stage: int = 1, max_versions: int = 2, with_pipeline: bool = True) -> For:
    d_dyn = DimVar("D", 32, 2048)
    gm_ty = TensorType((4, d_dyn), DType.f32, None, StorageKind.GMEM)
    ub_ty = TensorType((d_dyn,), DType.f32, None, StorageKind.SMEM)
    sc = TensorType.umat_scalar()
    k = Var("k", type=sc)
    gm_in = Var("gm_in", type=gm_ty)
    gm_out = Var("gm_out", type=gm_ty)
    ub_var = Var("ub_buf", type=ub_ty)
    return For(
        k,
        Constant(0, type=sc),
        Constant(4, type=sc),
        Constant(1, type=sc),
        Sequential(
            (
                LetStmt(ub_var, Call(AllocTensor(tensor_type=ub_ty), (), type=ub_ty)),
                Evaluate(AscendGmToUb(), (gm_in, ub_var)),
                Evaluate(AscendUbToGm(), (ub_var, gm_out)),
            )
        ),
        pipeline=(
            PipelineSpec(max_stage=max_stage, max_buffer_versions=max_versions, enable_offset=True)
            if with_pipeline
            else None
        ),
    )


def test_all_10_implemented_ops_resolve_metadata_and_regime_selectors() -> None:
    """Verify all 10 implemented ops resolve through anchor_specs and campaign_runner."""
    root = default_source_root()
    registry = CannbenchOpsRegistry(tasks_root=_cannbench_tasks_dir(), source_root=root)

    for op_name in IMPLEMENTED_OPS:
        assert op_name in OPERATOR_METADATA
        expected_rel = f"examples/ascend950pr_ops/{op_name}/kernels.py"
        assert (root / expected_rel).is_file()
        assert get_operator_module_relpath(op_name) == expected_rel
        assert registry.resolve_module_path(op_name) == expected_rel
        task = get_operator_task(op_name)
        assert task.endswith(f"/{op_name}")
        assert get_operator_module_relpath(task) == expected_rel
        assert get_operator_entrypoint(op_name) == op_name

        anchors = get_default_anchors(op_name)
        regime_specs = get_default_regime_specs(op_name)
        assert regime_specs
        assert 4 <= len(anchors) <= 6
        assert {anchor.regime_tag for anchor in anchors} >= {
            regime.regime_tag for regime in regime_specs
        }

        if op_name not in ("attention_residual", "mla"):
            resolved = resolve_regime_config(
                root / expected_rel,
                regime_specs[0],
                dims=regime_specs[0].representative_anchor_dims,
            )
            assert resolved.selector_name == f"_select_{op_name}_config"
            assert resolved.builder_name == f"_build_{op_name}_prim_func"
            primary_fn, snippet = extract_regime_snippet(
                (root / expected_rel).read_text(encoding="utf-8"),
                regime_specs[0],
            )
            assert primary_fn == f"_build_{op_name}_prim_func"
            assert f"def _select_{op_name}_config(" in snippet


def test_all_60_cannbench_ops_unified_sil_router_solver_template_cost_pipeline(tmp_path: Path) -> None:
    """Test (a): All 60 ops flow through SIL, Router, Solver, Templates, and CostModel."""
    ops_root = _cannbench_tasks_dir()
    extractor = OpAdvisoryExtractor(ops_root)
    advisories = extractor.extract_all()
    assert len(advisories) == 60

    solver = RegimePiecewiseSolver(AscendTarget("ascend.950pr"))
    verifier = SILVerifier()
    tmpl_reg = get_template_registry()
    cost_model = HybridAscendCostModel(AscendTarget("ascend.950pr"))

    observed_engines: set[str] = set()
    observed_rings: set[str] = set()
    observed_regs: set[str] = set()
    for op_name, advisory in sorted(advisories.items()):
        route = route_from_op_spec(advisory.to_dict())
        assert route.topology_axes.compute_engine in VALID_COMPUTE_ENGINES
        assert route.topology_axes.memory_ring in VALID_MEMORY_RINGS
        assert route.topology_axes.iteration_regularity in VALID_ITERATION_REGULARITIES
        assert route.recommended_backend_tier in ("tilelang_dsl", "ascendc_template")
        observed_engines.add(route.topology_axes.compute_engine)
        observed_rings.add(route.topology_axes.memory_ring)
        observed_regs.add(route.topology_axes.iteration_regularity)

        plan = solver.solve_all_regimes(
            advisory.regimes,
            op_name=op_name,
            topology_axes=route.topology_axes,
            roofline=advisory.roofline,
            input_dtypes=advisory.input_dtypes,
            has_nan_inf_cases=advisory.has_nan_inf_cases,
            has_indirect_gather=bool(advisory.structural_features.get("uses_simt", False)),
            has_recurrence=bool(advisory.structural_features.get("is_recurrent", False)),
        )
        policy_envelope = PolicyEnvelope.from_dict(plan.to_dict())
        report = verify_policy_envelope(policy_envelope, advisory)
        assert report.passed is True, f"{op_name} failed SILVerifier: {report.errors}"

        sched_env = SchedulingEnvelope(
            advisory=advisory,
            policy=policy_envelope,
            verification=report,
        )
        assert SchedulingEnvelope.from_dict(sched_env.to_dict()) == sched_env

        for regime in advisory.regimes:
            pol = policy_envelope.policies[regime.regime_tag]
            assert verifier.verify(pol, advisory).passed is True

            rendered_via_reg = tmpl_reg.render_template(route.topology_axes, pol, track="both")
            rendered_via_dict = render_template(route.topology_axes, pol.to_dict(), track="both")
            assert rendered_via_reg.prim_func is not None
            assert rendered_via_dict.prim_func is not None
            assert "Kernel" in rendered_via_reg.ascendc_cpp or "AscendC" in rendered_via_reg.ascendc_cpp

            gm_bytes = float(advisory.roofline.get("median_gm_bytes") or 65536)
            uses_cube = bool(advisory.structural_features.get("uses_cube", False))
            workload = {
                **pol.to_dict(),
                "cube_flops": gm_bytes * 64.0 if uses_cube else 0.0,
                "vec_flops": gm_bytes * 4.0,
                "gm_bytes": gm_bytes,
                "total_tiles": 56,
            }
            pred_from_policy = cost_model.predict(pol)
            assert pred_from_policy.pred_us >= 0.0
            pred = cost_model.predict(workload)
            assert pred.pred_us > 0.0

            t_hw = regime.t_hw_us if regime.t_hw_us > 0.0 else 5.0
            ref_perf = regime.ref_perf_us if regime.ref_perf_us > 0.0 else t_hw * 3.0
            score_pred = cost_model.predict_cannbench_eq3_score(pred.pred_us, ref_perf, t_hw)
            assert score_pred.eq3_score >= 0.0
            assert 0.0 < score_pred.hap_reward <= 3.0

    assert observed_engines == set(VALID_COMPUTE_ENGINES)
    assert observed_rings == set(VALID_MEMORY_RINGS)
    assert observed_regs == set(VALID_ITERATION_REGULARITIES)
    sample_adv = advisories["rms_norm"]
    sample_plan = solver.solve_all_regimes(sample_adv.regimes, op_name="rms_norm")
    saved = SchedulingEnvelope(
        advisory=sample_adv,
        policy=PolicyEnvelope.from_dict(sample_plan.to_dict()),
        verification=verify_policy_envelope(PolicyEnvelope.from_dict(sample_plan.to_dict()), sample_adv),
    ).save_json(tmp_path / "rms_norm_sil.json")
    assert saved.is_file()


def test_sil_verifier_catches_invalid_policies_and_solvers_fallback_cleanly() -> None:
    """Test (b): Invalid policies are caught by SILVerifier and trigger clean solver fallback."""
    verifier = SILVerifier()
    ax_gemm = TopologyAxes("aic_only", "gm_l1_l0_fixpipe", "affine_simd")
    ax_cv_ring = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd")
    ax_simd = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    ax_simt = TopologyAxes("aiv_only", "ub_random_atomic", "irregular_simt")
    adv = AdvisoryEnvelope(
        op_name="invalid_probe_op",
        cannbot_task="level4/invalid_probe_op",
        level=4,
        topology_axes=ax_gemm,
        input_dtypes=("float16",),
        output_dtypes=("float16",),
        roofline={"dominant_bound": "cube_compute"},
        structural_features={"uses_cube": True},
        regimes=(
            RegimeSlice(
                regime_tag="main",
                case_nums=(1,),
                representative_dims={"M": 256, "N": 256, "K": 128},
                has_unaligned_tail=False,
                ref_perf_us=30.0,
                t_hw_us=10.0,
            ),
        ),
        has_nan_inf_cases=True,
    )

    invalid_cases = [
        (
            "L1->GM",
            SchedulePolicy(
                regime_tag="main",
                topology_axes=ax_gemm,
                dma_paths_used=("GM->L1", "L1->GM"),
            ),
            "Forbidden dav-3510 DMA path",
        ),
        (
            "Cube int4 without unpack",
            SchedulePolicy(
                regime_tag="main",
                topology_axes=ax_gemm,
                cube_operand_dtype="int4",
                w4_mode="none",
            ),
            "Direct Cube INT4",
        ),
        (
            "tile > 4095",
            SchedulePolicy(
                regime_tag="main",
                topology_axes=ax_gemm,
                tile_mnk=(4096, 128, 64),
            ),
            "max_mmad_mnk",
        ),
        (
            "CrossCore > 15",
            SchedulePolicy(
                regime_tag="main",
                topology_axes=ax_cv_ring,
                sync_mode="mode4_flag",
                reverse_ack_period=16,
            ),
            "max_crosscore_depth",
        ),
        (
            "fast_unshifted_exp on inf/nan",
            SchedulePolicy(
                regime_tag="main",
                topology_axes=ax_simd,
                fast_unshifted_exp=True,
            ),
            "has_nan_inf_cases=True",
        ),
        (
            "ub_256kb_unlock with simtvf",
            SchedulePolicy(
                regime_tag="main",
                topology_axes=ax_simt,
                vector_mode="simtvf",
                ub_256kb_unlock=True,
            ),
            "ub_256kb_unlock=True is forbidden",
        ),
    ]

    for label, bad_policy, expected_substr in invalid_cases:
        rep = verifier.verify_policy(bad_policy, adv)
        assert rep.passed is False, f"{label} unexpectedly passed"
        assert any(expected_substr in err for err in rep.errors), (label, rep.errors)

    loop = _make_static_ub_loop(with_pipeline=True)
    problem, _ = extract_tir_loop_problem(
        loop,
        dim_hints={"D": 1024},
        allow_dynamic_fallback=True,
    )
    fallback_pipe = solve_ascend_pipeline(
        problem,
        sil_policy={"l1_stages": 0, "ub_256kb_unlock": False},
    )
    assert fallback_pipe.fallback_used is True
    assert "invalid l1_stages" in fallback_pipe.fallback_reason
    assert fallback_pipe.assignment.ii >= 1

    from tilefoundry.ir.hir.pipeline import PipelineError  # noqa: PLC0415

    unpipelined_loop = _make_static_ub_loop(with_pipeline=False)
    regime_solver = RegimePiecewiseSolver()
    with pytest.raises(PipelineError, match="PIPELINE_MISSING"):
        regime_solver.solve_regime(
            {
                "regime_tag": "main",
                "representative_dims": {"M": 16, "N": 1024, "D": 1024},
                "has_unaligned_tail": False,
            },
            op_name="rms_norm",
            topology_axes=ax_simd,
            loop=unpipelined_loop,
        )
    analytical_reg = regime_solver.solve_regime(
        {
            "regime_tag": "main",
            "representative_dims": {"M": 16, "N": 1024, "D": 1024},
            "has_unaligned_tail": False,
        },
        op_name="rms_norm",
        topology_axes=ax_simd,
        loop=None,
    )
    assert analytical_reg.solver_mode == "algebraic_piecewise"
    assert verifier.verify_policy(
        SchedulePolicy.from_dict(analytical_reg.to_dict()),
        AdvisoryEnvelope(
            op_name="rms_norm",
            cannbot_task="level2/rms_norm",
            level=2,
            topology_axes=ax_simd,
            input_dtypes=("float16",),
            output_dtypes=("float16",),
            roofline={},
            structural_features={},
            regimes=(
                RegimeSlice(
                    regime_tag="main",
                    case_nums=(1,),
                    representative_dims={"M": 16, "N": 1024, "D": 1024},
                    has_unaligned_tail=False,
                ),
            ),
        ),
    ).passed is True


class _EffortTrackingMockBackend(MockLLMBackend):
    """MockLLMBackend recording every GenerationRequest and effort argument."""

    def __init__(self, responses: dict[str, list[str]], default_response: str) -> None:
        super().__init__(responses=responses, default_response=default_response)
        self.efforts: list[str] = []

    async def generate(
        self,
        req: GenerationRequest,
        *,
        effort_override: str | None = None,
    ) -> str:
        eff = effort_override or req.effort
        self.efforts.append(eff)
        return await super().generate(req, effort_override=effort_override)


@pytest.mark.parametrize(
    "op_name,oracle_verify",
    [
        ("rms_norm", True),
        ("swi_glu", True),
        ("mhc_sinkhorn", True),
        ("kimi_delta_attention", False),
        ("tensor_program_vm", False),
    ],
)
def test_multi_round_marathon_portfolio_kb_cost_oracle_non_ar_mla(
    tmp_path: Path,
    op_name: str,
    oracle_verify: bool,
) -> None:
    """Test (c): Multi-round ExtremeCampaignRunner + PortfolioRunner + KB + Oracle."""
    root = default_source_root()
    src_kernel = root / get_operator_module_relpath(op_name)
    work_kernel = tmp_path / f"{op_name}_kernels.py"
    shutil.copy2(src_kernel, work_kernel)

    rules = load_seed_rules()
    rule_ids = {r.rule_id for r in rules}
    assert len(rules) >= 33
    for idx in range(1, 34):
        assert any(rid.startswith(f"R{idx:02d}_") for rid in rule_ids)

    fast_backend = FastCreativeAgyBackend()
    assert (
        fast_backend.select_effort(
            GenerationRequest(role="explore_creative", system_prompt="s", user_prompt="u"),
            effort_override="xhigh",
        )
        == "high"
    )

    advisory = OpAdvisoryExtractor(_cannbench_tasks_dir()).extract(op_name)
    route = route_from_op_spec(advisory.to_dict())
    anchors = get_default_anchors(op_name)
    regime_specs = get_default_regime_specs(op_name)
    single_regime = (regime_specs[0],)

    orig_text = work_kernel.read_text(encoding="utf-8")
    _, base_slice = extract_regime_snippet(orig_text, single_regime[0])
    round1_slice = f"# marathon_opt_tier=1\n{base_slice}"
    round2_slice = f"# marathon_opt_tier=2\n{base_slice}"

    backend = _EffortTrackingMockBackend(
        responses={
            "exploit_rule": [round1_slice, round2_slice, round2_slice],
            "explore_creative": [
                json.dumps(
                    {
                        "physical_hypothesis": f"Unlock 256KB UB + burst packing for {op_name}",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": round1_slice,
                    }
                ),
                json.dumps(
                    {
                        "physical_hypothesis": f"Static ping-pong + VLOOPv2 for {op_name}",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": round2_slice,
                    }
                ),
            ],
        },
        default_response=round1_slice,
    )

    cost_model = HybridAscendCostModel(AscendTarget("ascend.950pr"))
    oracle = CannbenchContractOracle(get_operator_task(op_name), tasks_root=_cannbench_tasks_dir())

    def _stage_a_checker(src: str, _reg: RegimeTargetSpec) -> StageAResult:
        tier = 2 if "marathon_opt_tier=2" in src else (1 if "marathon_opt_tier=1" in src else 0)
        ii = max(4, 30 - tier * 8)
        ledger = build_objective_physical_ledger(
            analysis_mode="readonly_envelope",
            is_legal=True,
            steady_state_ii=ii,
            pipe_cycles={"AIV0.PIPE_V": ii * 2, "AIV1.PIPE_V": ii * 2 if tier >= 1 else 0},
            waterlines={},
        )
        return StageAResult(
            passed=True,
            prim_func=None,
            ledger=ledger,
            features=route.features,
            route=route,
        )

    def _l2_eval(
        cands: list[tuple[str, Path]],
        active_anchors: list[AnchorCaseSpec],
    ) -> dict[str, CandidateSurrogateResult]:
        out: dict[str, CandidateSurrogateResult] = {}
        for cid, mod_path in cands:
            src = mod_path.read_text(encoding="utf-8")
            tier = 2 if "marathon_opt_tier=2" in src else (1 if "marathon_opt_tier=1" in src else 0)
            metrics = []
            for anc in active_anchors:
                t_hw = anc.t_hw_us if anc.t_hw_us > 0.0 else 5.0
                ref_us = anc.ref_perf_us if anc.ref_perf_us > 0.0 else t_hw * 3.0
                pred = cost_model.predict(
                    {
                        "topology_axes": route.topology_axes,
                        "vec_flops": 2e10 if tier == 0 else (1.2e10 if tier == 1 else 8e9),
                        "gm_bytes": 1e6,
                        "contig_burst_bytes": 4096 if tier == 0 else 32768,
                        "l1_stages": 1 if tier == 0 else 2,
                        "l1_half_bank_split": True,
                        "ub_bank_pad_datablocks": 0 if tier == 0 else 1,
                        "spill_bytes": 128 if tier == 0 else 0,
                        "software_loops": 1 if tier == 0 else 0,
                        "vloopv2_loops": 0 if tier == 0 else 2,
                    }
                )
                cand_us = max(t_hw * 1.05, min(ref_us * 1.5, pred.pred_us * (1.0 - 0.12 * tier)))
                if tier == 0:
                    cand_us = max(cand_us, ref_us * 0.95)
                elif tier == 1:
                    cand_us = max(t_hw * 1.25, ref_us * 0.58)
                else:
                    cand_us = max(t_hw * 1.05, ref_us * 0.40)
                score_pred = cost_model.predict_cannbench_eq3_score(cand_us, ref_us, t_hw)
                assert score_pred.hap_reward > 0.0
                metrics.append(
                    build_anchor_metrics(
                        anc,
                        accuracy_passed=True,
                        cand_us=cand_us,
                    )
                )
            out[cid] = aggregate_surrogate_result(cid, op_name, metrics)
        return out

    if oracle_verify:
        small_case = min(
            oracle.cases,
            key=lambda c: sum(
                int(torch.tensor(s).prod().item())
                for s in (c.get("input_shape") or [[1]])
                if isinstance(s, list) and all(isinstance(d, int) for d in s)
            ),
        )
        cid_small = int(small_case["case_id"])
        ok, rep = oracle.verify_candidate(oracle.reference_fn, case_nums=[cid_small])
        assert ok is True, rep

    tracker = BayesianRuleTracker.from_seed_yaml()
    flywheel = FlywheelDistiller(
        funnel=GateFunnel(skip_stage_b=True),
        tracker=tracker,
        min_edit_gain=0.0,
        min_probe_speedup=1.0,
    )

    runner = ExtremeCampaignRunner(
        config=CampaignConfig(
            operator_name=op_name,
            max_rounds=2,
            k_rule_workers=2,
            stagnation_escalate_rounds=1,
            disable_early_goal_stop=True,
        ),
        target_module_path=work_kernel,
        perf_target=PerfTargetSpec(
            operator_name=op_name,
            target_avg_speedup=5.0,
            target_min_case_speedup=4.0,
            target_roofline_eta=0.99,
            target_surrogate_score=99.9,
            min_optimization_rounds=2,
        ),
        anchors=anchors,
        regime_specs=single_regime,
        llm_backend=backend,
        tracker=tracker,
        flywheel_distiller=flywheel,
        l0_checker=_stage_a_checker,
        l1_compiler=lambda _src, _reg: StageBResult(passed=True),
        l2_batch_evaluator=_l2_eval,
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        msprof_collector=lambda _op, _p, anc_seq: {
            f"case_{a.case_num}": {"aiv_vec": 0.72, "mte2": 0.35} for a in anc_seq
        },
        checkpoint_path=tmp_path / f"{op_name}_ckpt.json",
        campaign_kb_path=tmp_path / f"{op_name}_kb.yaml",
        source_root=root,
        cannbench_repo=_cannbench_tasks_dir().parent,
    )

    report = asyncio.run(runner.run())
    assert report.total_rounds == 2
    assert len(report.rounds) == 2
    assert "xhigh" not in [(e or "").lower() for e in backend.efforts]
    assert all(r.creative_effort != "xhigh" for r in report.rounds)

    scores = [r.best_so_far_surrogate_score for r in report.rounds]
    speedups = [r.best_so_far_speedup for r in report.rounds]
    assert scores[0] <= scores[1]
    assert speedups[0] <= speedups[1]
    assert report.best_result.avg_speedup >= report.baseline_result.avg_speedup

    portfolio_tracker = BayesianRuleTracker.from_seed_yaml()
    portfolio_funnel = GateFunnel(
        compile_runner=lambda _dsl: StageBResult(passed=True),
        device_runner=lambda _dsl, st_a: StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=10.0,
            residual=compute_static_dynamic_residual(st_a.ledger, {"dur_us": 10.0}),
            telemetry={"dur_us": 10.0, "aiv_vec": 0.8},
        ),
    )
    portfolio = PortfolioRunner(
        backend=backend,
        funnel=portfolio_funnel,
        tracker=portfolio_tracker,
        beam=TwoLaneBeamSearch(),
        k_rule_workers=1,
    )
    portfolio_seed = (
        "import tilelang.ascend.language as T\n"
        "from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS\n\n"
        "def _build_portfolio_probe():\n"
        "    @T.prim_func\n"
        "    def main(X: T.Tensor((64, 128), 'bfloat16'), Y: T.Tensor((64, 128), 'bfloat16')):\n"
        "        with T.Kernel(1):\n"
        "            x_ub = T.alloc_shared((64, 128), 'bfloat16')\n"
        "            T.copy(X[0:64, 0:128], x_ub)\n"
        "            T.copy(x_ub, Y[0:64, 0:128])\n"
        "    return main\n"
    )
    portfolio_backend = _EffortTrackingMockBackend(
        responses={
            "exploit_rule": [portfolio_seed],
            "explore_creative": [
                json.dumps(
                    {
                        "physical_hypothesis": f"Optimize {op_name} UB staging",
                        "target_local_metric": "aiv_vec",
                        "dsl_source": portfolio_seed,
                    }
                )
            ],
        },
        default_response=portfolio_seed,
    )
    portfolio = PortfolioRunner(
        backend=portfolio_backend,
        funnel=portfolio_funnel,
        tracker=portfolio_tracker,
        beam=TwoLaneBeamSearch(),
        k_rule_workers=1,
    )
    parent_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=24,
        pipe_cycles={"AIV0.PIPE_V": 48},
        waterlines={},
        grid_blocks=4,
        wave_occupancy=0.25,
    )
    parent = BeamCandidate(
        candidate_id=f"{op_name}_seed",
        round_idx=0,
        worker_kind="seed",
        rule_id=None,
        dsl_source=portfolio_seed,
        track="unconstrained",
        ledger=parent_ledger,
        residual=compute_static_dynamic_residual(parent_ledger, {"dur_us": 24.0}),
        latency_us=24.0,
        accuracy_passed=True,
        structural_signature=extract_structural_signature(portfolio_seed),
    )
    p_round = asyncio.run(
        portfolio.run_round(
            parent,
            round_idx=1,
            factory_name="_build_portfolio_probe",
            sil_advisory=advisory,
        )
    )
    assert p_round.evaluated_candidates
    assert f"mr:{advisory.topology_axes.memory_ring}" in portfolio_tracker.axis_posteriors
    assert "xhigh" not in [(e or "").lower() for e in backend.efforts]
    assert "xhigh" not in [(e or "").lower() for e in portfolio_backend.efforts]
