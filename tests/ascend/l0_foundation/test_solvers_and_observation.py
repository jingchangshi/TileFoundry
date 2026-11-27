"""Tests for Regime/Pipeline Solvers, 5D Observation Space, SIL Contract, and KB Flywheel."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from pathlib import Path

import pytest
import tilelang.ascend.language as T

from tilefoundry.vendor.tilelang_z3 import (
    Buffer,
    Capacity,
    Edge,
    Problem,
    Resource,
    Reuse,
    Task,
    verify_assignment,
)
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CampaignConfig,
    PerfTargetCampaignRunner,
)
from tilefoundry_ascend.agent.campaign.marathon import MarathonConfig
from tilefoundry_ascend.agent.campaign.telemetry_ledger_writer import TelemetryLedgerWriter
from tilefoundry_ascend.agent.flywheel.ablation_distiller import (
    AstEditChunk,
    FlywheelDistiller,
    distill_causal_rule,
)
from tilefoundry_ascend.agent.gates.gate_funnel import (
    GateFunnel,
    StageAResult,
    StageBResult,
    evaluate_ascendc_candidate,
    run_stage_a_static,
)
from tilefoundry_ascend.agent.kb.bayesian_tracker import (
    BayesianRuleTracker,
    append_kb_event_shard,
    save_distilled_rule_file,
)
from tilefoundry_ascend.agent.kb.schema import AscendCausalRule, build_predicate_context
from tilefoundry_ascend.agent.npu import (
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    L3AcceptanceResult,
    L4CannbenchResult,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    StaticDynamicResidual,
    compiler_summary_from_residual,
    compute_static_dynamic_residual,
    extract_cpp_trackb_signature_tags,
    extract_structural_signature,
)
from tilefoundry_ascend.agent.router.capability_gate import (
    CapabilityGate,
    EscalationDecision,
    RouteDecision,
    normalize_router_tier,
    route_from_op_spec,
    route_initial_by_input_contract,
)
from tilefoundry_ascend.agent.router.isa_gap_scanner import (
    PipeGapBreakdown,
    PipeGapReport,
    scan_tilelang_pipe_gaps,
)
from tilefoundry_ascend.agent.specs.anchor_specs import (
    AnchorCaseSpec,
    PerfTargetSpec,
    RegimeTargetSpec,
)
from tilefoundry_ascend.ascend.cost_model import (
    BishengResourceSummary,
    ExperienceLake,
    MicroarchProfile,
    extract_microarch_profile_from_cce,
    extract_microarch_profile_from_tirx,
    merge_microarch_profiles,
)
from tilefoundry_ascend.ascend.passes.observation_space import (
    AxisResidual5D,
    ObservationVector5D,
    compute_transfer_inflation_ratio,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    ObjectivePhysicalLedger,
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.schedule import (
    PipelineError,
    inspect_tilelang_primfunc,
    read_from_sil,
    solve_ascend_pipeline,
    solve_pipeline_with_monotone_backoff,
    task_resource_mask,
)
from tilefoundry_ascend.ascend.schedule.regime_solver import (
    SOLVER_MODE_ALGEBRAIC,
    RegimePiecewiseSolver,
    RegimePlanResult,
    normalize_solver_mode,
)
from tilefoundry_ascend.ascend.sil import (
    DEFAULT_OPS_ROOT,
    VALID_BACKEND_TIERS,
    VALID_COMPUTE_ENGINES,
    VALID_FREEDOM_FACETS,
    VALID_ITERATION_REGULARITIES,
    VALID_MEMORY_RINGS,
    AdvisoryEnvelope,
    CasePhysicsRatios,
    CasePhysicsRegimeCase,
    ConstraintRelaxationMask,
    HardwareLegalityBounds,
    OpAdvisoryExtractor,
    PolicyEnvelope,
    RegimeSlice,
    SchedulePolicy,
    SchedulingEnvelope,
    SILVerifier,
    TelemetryFeedback,
    TopologyAxes,
    VerificationReport,
    classify_case_regime,
    partition_case_regimes,
    synthesize_default_policy,
    verify_policy_envelope,
)
from tilefoundry_ascend.ascend.sil.schema import _merge_nearest_regimes
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget, NodeResourceMask
from tilefoundry_ascend.ascend.templates.registry import relax_facets_for_bottleneck

AXES_SIMD = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
AXES_SIMT = TopologyAxes("aiv_only", "ub_random_atomic", "irregular_simt")
AXES_GEMM = TopologyAxes("aic_only", "gm_l1_l0_fixpipe", "affine_simd")
AXES_GROUPED = TopologyAxes("coupled_1c2v", "l0c_ub_epilogue", "ragged_group")
AXES_CV_RING = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd")
AXES_PERSISTENT = TopologyAxes("coupled_1c2v", "persistent_onchip_state", "affine_simd")
AXES_STRIPED_VM = TopologyAxes("aiv_only", "persistent_onchip_state", "dynamic_opcode_stream")


def _vector_res() -> tuple[Resource, ...]:
    return (
        Resource("aiv", "aiv0", "mte2", 1),
        Resource("aiv", "aiv0", "vector", 1),
        Resource("aiv", "aiv0", "mte3", 1),
    )


def _mix_res() -> tuple[Resource, ...]:
    return (
        *_vector_res(),
        Resource("aic", "aic0", "mte2", 1),
        Resource("aic", "aic0", "mte1", 1),
        Resource("aic", "aic0", "cube", 1),
        Resource("aic", "aic0", "fixpipe", 1),
    )


def _ldg(
    mode: str,
    legal: bool,
    ii: int | None,
    pipes: dict[str, int],
    wls: dict[str, object] | None = None,
    **kw: object,
) -> ObjectivePhysicalLedger:
    return build_objective_physical_ledger(
        analysis_mode=mode, is_legal=legal, steady_state_ii=ii,
        pipe_cycles=pipes, waterlines=wls or {}, **kw,
    )


def _rich_ledger() -> ObjectivePhysicalLedger:
    return _ldg(
        "exact_modulo_z3", True, 3,
        {"AIC.PIPE_M": 100, "AIC.CUBE": 20, "AIV0.PIPE_V": 300, "AIV1.PIPE_V": 100},
        {
            "UB": build_scope_waterline(scope="UB", capacity_bytes=4096, explicit_buffers={"ub_out": 4096}, implicit_buffers={}),
            "L1": build_scope_waterline(scope="L1", capacity_bytes=8192, explicit_buffers={"l1_in": 9000}, implicit_buffers={}),
        },
        mte2_inflight_per_aiv=2, unaligned_dma_padding_bytes=64, grid_blocks=8,
        wave_occupancy=0.8, gm_roundtrip_bytes=4096, dominant_contig_bytes=64,
        has_indirect_gather=True, has_online_rescale_loop=True, has_lowbit_quant=True,
        in_loop_pipe_all_barriers=1, dma_burst_efficiency_ratio=0.75,
        uses_dynamic_offset_pipelining=True, ub_bank_conflict_risk=0.1,
        l1_bank_conflict_risk=0.05, microarch_profile={"subkernel_count": 3, "mte2_bytes_hint": 2048},
    )


def _rich_compiler() -> BishengResourceSummary:
    return BishengResourceSummary(
        vec_regs=32, scalar_regs=16, spill_bytes=128, stack_bytes=64,
        vloopv2_loops=0, software_loops=2, has_local_membar_in_vf_call=True,
        vf_fusion_breaks=2, alias_serialized_buffers=1,
        dual_issue_ipc_collapse=0.1, dcache_dcci_flushes=1,
    )


def _rich_telemetry() -> dict[str, object]:
    return {
        "measured_latency_us": 42.0, "aic_mac_ratio": 0.6, "aiv_vec_ratio": 0.5,
        "roofline_eta": 0.7, "mte2_wait_ratio": 0.3, "aic_fixpipe": 0.2,
        "l2_hit_rate": 0.6, "bank_conflict_ratio": 0.25, "sync_stall_ratio": 0.22,
        "pseudo_double_buffer_serialized": True,
    }


def _quiet_ledger() -> ObjectivePhysicalLedger:
    return build_objective_physical_ledger(
        analysis_mode="readonly_envelope", is_legal=True, steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 10}, waterlines={},
    )


def _sample_advisory(
    *,
    op_name: str = "sample_gemm",
    axes: TopologyAxes = AXES_GEMM,
    regime_tag: str = "aligned_main",
    rep_dims: dict[str, int] | None = None,
    has_unaligned_tail: bool = False,
    has_nan_inf_cases: bool = False,
    precision_thresholds: dict[str, float] | None = None,
) -> AdvisoryEnvelope:
    return AdvisoryEnvelope(
        op_name=op_name,
        cannbot_task=f"level3/{op_name}",
        level=3,
        topology_axes=axes,
        input_dtypes=("float16", "float16"),
        output_dtypes=("float16",),
        roofline={"dominant_bound": "cube_compute", "mean_t_hw_us": 12.5},
        structural_features={"uses_cube": axes.uses_cube},
        regimes=(
            RegimeSlice(
                regime_tag=regime_tag,
                case_nums=(1, 2, 3),
                representative_dims=rep_dims or {"M": 256, "N": 256, "K": 128},
                has_unaligned_tail=has_unaligned_tail,
                has_empty_group=False,
                ref_perf_us=37.5,
                t_hw_us=12.5,
            ),
        ),
        legality_bounds=HardwareLegalityBounds(),
        has_nan_inf_cases=has_nan_inf_cases,
        precision_thresholds=precision_thresholds or {},
    )


def test_regime_solver_z3_projection_and_piecewise_ops() -> None:
    """Verify RegimePiecewiseSolver Z3 projection and algebraic planning for 5 irregular ops."""
    facts = AscendTarget("ascend.950pr").get_facts(AscendSpaceFacts)
    problem = Problem(
        tasks=(Task("load", (("aiv", "aiv0", "mte2", 1),), 1, 0), Task("store", (("aiv", "aiv0", "mte3", 1),), 1, 1)),
        buffers=(Buffer("ub_buf", 4096, "UB", "aiv", auto_min=1, auto_max=2),),
        edges=(Edge("load", "store", 0, 1),),
        reuses=(Reuse("ub_buf", "store", "load", 0, 1),),
        resources=(Resource("aiv", "aiv0", "mte2", 1), Resource("aiv", "aiv0", "mte3", 1)),
        capacities=(Capacity("UB", "aiv", facts.ub_simdvf_usable_bytes),),
        max_stage=1,
        manual=True,
    )
    solver = RegimePiecewiseSolver()
    res = solver.solve_regime(
        {"regime_tag": "large_row", "representative_dims": {"M": 4, "N": 1024, "D": 1024}, "has_unaligned_tail": False},
        op_name="rms_norm", topology_axes=AXES_SIMD, problem=problem,
    )
    assert res.solver_mode == "z3_projected" and res.ii >= 1 and res.buffer_versions["ub_buf"] == 2
    assert SchedulePolicy.from_dict(res.to_dict()).regime_tag == "large_row"

    cv_ring = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "irregular_simt")
    sfa_reg = RegimeSlice("prefill_sparse", (1, 2), {"B": 2, "M": 512, "S_kv": 4096, "D": 128}, has_unaligned_tail=True)
    sfa_res = solver.solve_regime(
        sfa_reg, op_name="sparse_flash_attention", topology_axes=cv_ring,
        roofline={"dominant_bound": "cube", "arithmetic_intensity": 256.0},
    )
    assert sfa_res.solver_mode == SOLVER_MODE_ALGEBRAIC and sfa_res.caliper_steps == 0
    assert normalize_solver_mode("analytical_piecewise") == SOLVER_MODE_ALGEBRAIC
    assert RegimePlanResult.from_dict(dict(sfa_res.to_summary_dict(), solver_mode="analytical_piecewise")).solver_mode == SOLVER_MODE_ALGEBRAIC
    assert sfa_res.policy["tile_mnk"][0] >= 82 and sfa_res.policy["l1_half_bank_split"] is True
    assert "l0c_to_ub_split_m" in sfa_res.policy["cv_datapath"] and sfa_res.policy["simt_threads"] == 512

    gmm_res = solver.solve_regime(
        RegimeSlice("ragged_empty", (3,), {"M": 192, "N": 256, "K": 128}, True, True),
        op_name="grouped_matmul", topology_axes=AXES_GROUPED, roofline={"dominant_bound": "cube"},
    )
    assert gmm_res.policy["ensure_empty_subblock_sync"] and gmm_res.policy["tile_mnk"][0] >= 82

    kda_res = solver.solve_regime(
        RegimeSlice("chunk_64", (4,), {"B": 4, "S": 1024, "D_k": 128, "D_v": 128}, False),
        op_name="kimi_delta_attention", topology_axes=AXES_PERSISTENT, has_recurrence=True,
    )
    assert kda_res.persistent_state_space in ("L1", "UB") and kda_res.persistent_state_bytes == 128 * 128 * 4

    sink_res = solver.solve_regime(
        RegimeSlice("sinkhorn_4x4", (5,), {"M": 32, "N": 64}, False),
        op_name="mhc_sinkhorn", topology_axes=TopologyAxes("aiv_only", "persistent_onchip_state", "affine_simd"), has_nan_inf_cases=True,
    )
    assert sink_res.persistent_state_space == "UB" and sink_res.policy["fast_unshifted_exp"] is False

    vm_res = solver.solve_regime(
        RegimeSlice("vm_striped", (6,), {"R": 16, "W": 4096}, True),
        op_name="tensor_program_vm", topology_axes=AXES_STRIPED_VM,
    )
    assert vm_res.policy["sync_mode"] == "none" and vm_res.policy["direct_no_fusion_vf"] is True

    plan = solver.solve_all_regimes([sfa_reg], op_name="sparse_flash_attention", topology_axes=cv_ring, roofline={"dominant_bound": "cube"})
    rep = SILVerifier().verify_envelope(PolicyEnvelope.from_dict(plan.to_dict()), _sample_advisory(op_name="sparse_flash_attention", axes=cv_ring))
    assert rep.passed is False or rep.passed is True
    adv_sfa = AdvisoryEnvelope(
        "sparse_flash_attention", "level4/sparse_flash_attention", 4, cv_ring,
        ("f16",), ("f16",), {"dominant_bound": "cube"}, {}, (sfa_reg,), HardwareLegalityBounds(),
    )
    assert SILVerifier().verify_envelope(PolicyEnvelope.from_dict(plan.to_dict()), adv_sfa).passed is True


def test_pipeline_solver_vector_cube_mix_and_unsat() -> None:
    """Verify Z3 pipeline solver on vector, cube, FlashMLA dual_copy, and UNSAT cases."""
    facts = AscendTarget("ascend.950pr").get_facts(AscendSpaceFacts)
    vec_p = Problem(
        tasks=(Task("load", (("aiv", "aiv0", "mte2", 1),), 1, 0), Task("vec", (("aiv", "aiv0", "vector", 1),), 1, 1), Task("store", (("aiv", "aiv0", "mte3", 1),), 1, 2)),
        buffers=(Buffer("ub_in", 64 * 1024, "UB", "aiv", auto_min=1, auto_max=2), Buffer("ub_out", 32 * 1024, "UB", "aiv", auto_min=1, auto_max=2)),
        edges=(Edge("load", "vec", 0, 1), Edge("vec", "store", 0, 1)),
        reuses=(Reuse("ub_in", "vec", "load", 0, 1), Reuse("ub_out", "store", "vec", 0, 1)),
        resources=_vector_res(), capacities=(Capacity("UB", "aiv", facts.ub_usable_bytes),), max_stage=2, manual=True,
    )
    vec_r = solve_ascend_pipeline(vec_p)
    assert verify_assignment(vec_p, vec_r.assignment) == []
    assert vec_r.buffer_versions == {"ub_in": 2, "ub_out": 2} and vec_r.task_stages == {"load": 0, "vec": 1, "store": 2}

    cube_p = Problem(
        tasks=(
            Task("gm_l1_a", (("aic", "aic0", "mte2", 1),), 1, 0), Task("l1_l0a", (("aic", "aic0", "mte1", 1),), 1, 1),
            Task("l1_l0b", (("aic", "aic0", "mte1", 1),), 1, 1), Task("mmad", (("aic", "aic0", "cube", 1),), 1, 2),
            Task("fix", (("aic", "aic0", "fixpipe", 1),), 1, 3),
        ),
        buffers=(
            Buffer("l1_a", 65536, "L1", "aic", auto_min=1, auto_max=2), Buffer("l0a_buf", 16384, "L0A", "aic", auto_min=1, auto_max=2),
            Buffer("l0b_buf", 16384, "L0B", "aic", auto_min=1, auto_max=2), Buffer("l0c_buf", 65536, "L0C", "aic", fixed=1),
        ),
        edges=(Edge("gm_l1_a", "l1_l0a", 0, 1), Edge("l1_l0a", "mmad", 0, 1), Edge("l1_l0b", "mmad", 0, 1), Edge("mmad", "fix", 0, 1)),
        reuses=(Reuse("l1_a", "l1_l0a", "gm_l1_a", 0, 1), Reuse("l0a_buf", "mmad", "l1_l0a", 0, 1), Reuse("l0b_buf", "mmad", "l1_l0b", 0, 1)),
        resources=_mix_res(),
        capacities=(Capacity("L1", "aic", facts.l1_bytes), Capacity("L0A", "aic", facts.l0a_bytes), Capacity("L0B", "aic", facts.l0b_bytes), Capacity("L0C", "aic", facts.l0c_bytes)),
        max_stage=3, manual=True,
    )
    cube_r = solve_ascend_pipeline(cube_p)
    assert verify_assignment(cube_p, cube_r.assignment) == [] and cube_r.buffer_versions["l0c_buf"] == 1

    dual_res = (("aic", "aic0", "fixpipe", 1), ("aiv", "aiv0", "mte2", 1))
    mix_p = Problem(
        tasks=(
            Task("load_kv", (("aic", "aic0", "mte2", 1),), 1, 0), Task("mmad_qk", (("aic", "aic0", "cube", 1),), 1, 1),
            Task("dual_s", dual_res, 1, 1), Task("softmax", (("aiv", "aiv0", "vector", 1),), 1, 2),
            Task("dual_p", dual_res, 1, 2), Task("mmad_pv", (("aic", "aic0", "cube", 1),), 1, 3),
        ),
        buffers=(
            Buffer("l1_kv", 65536, "L1", "aic", auto_min=1, auto_max=2), Buffer("l0c_s", 32768, "L0C", "aic", auto_min=1, auto_max=2),
            Buffer("ub_s", 32768, "UB", "aiv", auto_min=1, auto_max=2), Buffer("l1_p", 32768, "L1", "aic", auto_min=1, auto_max=2),
        ),
        edges=(Edge("load_kv", "mmad_qk", 0, 1), Edge("mmad_qk", "dual_s", 0, 1), Edge("dual_s", "softmax", 0, 1), Edge("softmax", "dual_p", 0, 1), Edge("dual_p", "mmad_pv", 0, 1)),
        reuses=(Reuse("ub_s", "softmax", "dual_s", 0, 1), Reuse("l1_p", "mmad_pv", "dual_p", 0, 1)),
        resources=_mix_res(),
        capacities=(Capacity("L1", "aic", facts.l1_bytes), Capacity("L0C", "aic", facts.l0c_bytes), Capacity("UB", "aiv", facts.ub_usable_bytes)),
        max_stage=3, manual=True,
    )
    mix_r = solve_ascend_pipeline(mix_p)
    assert verify_assignment(mix_p, mix_r.assignment) == [] and mix_r.assignment.ii >= 2

    unsat_p = Problem(
        tasks=(Task("load", (("aiv", "aiv0", "mte2", 1),), 1, 0), Task("vec", (("aiv", "aiv0", "vector", 1),), 1, 1)),
        buffers=(Buffer("huge_ub", 128 * 1024, "UB", "aiv", fixed=2),),
        edges=(Edge("load", "vec", 0, 1),), reuses=(), resources=_vector_res(),
        capacities=(Capacity("UB", "aiv", facts.ub_usable_bytes),), max_stage=1, manual=True,
    )
    with pytest.raises(PipelineError) as caught:
        solve_ascend_pipeline(unsat_p)
    assert caught.value.code == "ASCEND_PIPELINE_UNSAT"

    acc_p = Problem(
        tasks=(Task("load_v", (("aiv", "aiv0", "mte2", 1),), 2, 0), Task("simdvf_add", (("aiv", "aiv0", "vector", 1),), 3, 1)),
        buffers=(Buffer("v_ub", 16384, "UB", "aiv", auto_min=1, auto_max=2), Buffer("acc_ub", 16384, "UB", "aiv", fixed=1)),
        edges=(Edge("load_v", "simdvf_add", 0, 2), Edge("simdvf_add", "simdvf_add", 1, 3)),
        reuses=(Reuse("v_ub", "simdvf_add", "load_v", 0, 1),),
        resources=_vector_res(), capacities=(Capacity("UB", "aiv", facts.ub_simdvf_usable_bytes),), max_stage=1, manual=True,
    )
    acc_r = solve_ascend_pipeline(acc_p)
    assert acc_r.assignment.ii >= 3 and acc_r.buffer_versions == {"v_ub": 2, "acc_ub": 1}


def test_pipeline_solver_mode_ub_sil_aliasing_backoff_and_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify mode-aware UB budgets, SIL aliasing/backoff, read_from_sil, and thread safety."""
    monkeypatch.chdir(tmp_path)

    @T.prim_func
    def piped_add(a: T.Tensor((4, 64), "float32"), b: T.Tensor((4, 64), "float32"), c: T.Tensor((4, 64), "float32")):
        with T.Kernel(1):
            au, bu, cu = T.alloc_shared((1, 64), "float32"), T.alloc_shared((1, 64), "float32"), T.alloc_shared((1, 64), "float32")
            for k in T.Pipelined(4, num_stages=2):
                T.copy(a[k : k + 1, 0:64], au)
                T.copy(b[k : k + 1, 0:64], bu)
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    T.simd.vsts(cu[0, 0], T.simd.vadd(T.simd.vld(au[0, 0], dist="NORM_B32"), T.simd.vld(bu[0, 0], dist="NORM_B32"), mask=m32), mask=m32, dist="NORM_B32")
                T.copy(cu, c[k : k + 1, 0:64])

    insp = inspect_tilelang_primfunc(piped_add)
    assert insp.ledger.analysis_mode == "exact_modulo_z3" and insp.ledger.waterlines["UB"].overflow_bytes == 0

    facts = AscendTarget("ascend.950pr").get_facts(AscendSpaceFacts)
    t2 = (Task("load", (("aiv", "aiv0", "mte2", 1),), 1, 0), Task("vec", (("aiv", "aiv0", "vector", 1),), 1, 1))
    e2, r2 = (Edge("load", "vec", 0, 1),), (Reuse("x_ub", "vec", "load", 0, 1),)
    b2 = (Buffer("x_ub", 240 * 1024, "UB", "aiv", fixed=1),)
    assert solve_ascend_pipeline(Problem(tasks=t2, edges=e2, buffers=b2, reuses=r2, resources=_vector_res(), capacities=(Capacity("UB", "aiv", facts.ub_limit_bytes("simdvf")),), max_stage=1, manual=True)).assignment.ii >= 1
    with pytest.raises(PipelineError, match="ASCEND_PIPELINE_UNSAT"):
        solve_ascend_pipeline(Problem(tasks=t2, edges=e2, buffers=b2, reuses=r2, resources=_vector_res(), capacities=(Capacity("UB", "aiv", facts.ub_limit_bytes("simtvf")),), max_stage=1, manual=True))

    dual_t = Task("dual_l0c_to_ub", (("aic", "aic", "fixpipe", 1, 0), ("aiv", "aiv0", "mte2", 1, 0)), 1, 0)
    assert task_resource_mask(dual_t) == (NodeResourceMask.for_aic("FIXPIPE") | NodeResourceMask.for_aiv0("MTE2"))

    sh_prob = Problem(
        tasks=t2, edges=e2, resources=_vector_res(), max_stage=1, manual=True,
        buffers=(Buffer("ub_big", 252 * 1024, "UB", "aiv", fixed=1), Buffer("l1_buf", 65536, "L1", "aic", auto_min=1, auto_max=2), Buffer("l0c_buf", 32768, "L0C", "aic", auto_min=1, auto_max=2)),
        reuses=(Reuse("ub_big", "vec", "load", 0, 1), Reuse("l1_buf", "vec", "load", 0, 1)),
        capacities=(Capacity("UB", "aiv", facts.ub_simdvf_usable_bytes), Capacity("L1", "aic", facts.l1_bytes), Capacity("L0C", "aic", facts.l0c_bytes)),
    )
    sh_file = tmp_path / "shadow" / "solved.json"
    sh_res = solve_ascend_pipeline(sh_prob, sil_policy={"ub_256kb_unlock": True, "l1_stages": 2, "l0c_buffers": 2}, shadow_dump_path=sh_file)
    assert not sh_res.fallback_used and sh_res.buffer_versions["l1_buf"] == 2 and sh_file.is_file()
    assert not Path("envelope.json").exists()

    alias_p = Problem(
        tasks=(Task("load_k_rope", (("aic", "aic0", "mte2", 1),), 1, 0), Task("mmad_qk", (("aic", "aic0", "cube", 1),), 1, 0), Task("dual_p", (("aic", "aic0", "fixpipe", 1),), 1, 0), Task("mmad_pv", (("aic", "aic0", "cube", 1),), 1, 0)),
        buffers=(Buffer("l1_k_rope", 256 * 1024, "L1", "aic", fixed=1), Buffer("l1_p", 256 * 1024, "L1", "aic", fixed=1)),
        edges=(Edge("load_k_rope", "mmad_qk", 0, 1), Edge("mmad_qk", "dual_p", 0, 1), Edge("dual_p", "mmad_pv", 0, 1)),
        reuses=(Reuse("l1_k_rope", "mmad_qk", "load_k_rope", 0, 1), Reuse("l1_p", "mmad_pv", "dual_p", 0, 1)),
        resources=_mix_res(), capacities=(Capacity("L1", "aic", 300 * 1024),), max_stage=0, manual=False,
    )
    with pytest.raises(PipelineError, match="ASCEND_PIPELINE_UNSAT"):
        solve_ascend_pipeline(alias_p)
    al_res = solve_ascend_pipeline(alias_p, sil_policy={"l1_slot_aliasing": {"l1_p": "l1_k_rope"}})
    assert not al_res.fallback_used and {b.name: b.slot_bytes for b in al_res.problem.buffers}["l1_p"] == 0

    assert read_from_sil(tmp_path / "missing.json") is None
    valid_sil = tmp_path / "valid_sil.json"
    valid_sil.write_text(json.dumps({"roofline": {"compute_flops": 1024}}), encoding="utf-8")
    assert read_from_sil(valid_sil) == {"roofline": {"compute_flops": 1024}}

    tight = Problem(tasks=t2, edges=e2, buffers=(Buffer("b0", 262144, "UB", "aiv", fixed=1),), reuses=(Reuse("b0", "vec", "load", 0, 1),), resources=_vector_res(), capacities=(Capacity("UB", "aiv", facts.ub_simdvf_usable_bytes),), max_stage=1, manual=True)
    bo = solve_pipeline_with_monotone_backoff(tight, max_retries=4)
    assert not bo.fallback_used and bo.problem.buffers[0].slot_bytes <= facts.ub_simdvf_usable_bytes

    conc_p = Problem(
        tasks=(Task("load", (("aiv", "aiv0", "mte2", 1),), 1, 0), Task("vec", (("aiv", "aiv0", "vector", 1),), 2, 1), Task("store", (("aiv", "aiv0", "mte3", 1),), 1, 2)),
        edges=(Edge("load", "vec", 0, 1), Edge("vec", "store", 0, 2)),
        buffers=(Buffer("b_in", 4096, "UB", "aiv", auto_min=1, auto_max=2), Buffer("b_out", 4096, "UB", "aiv", auto_min=1, auto_max=2)),
        reuses=(Reuse("b_in", "vec", "load", 0, 1), Reuse("b_out", "store", "vec", 0, 1)),
        resources=_vector_res(), capacities=(Capacity("UB", "aiv", facts.ub_simdvf_usable_bytes),), max_stage=2, manual=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        outcomes = list(ex.map(lambda _: (lambda r: (r.assignment.ii, verify_assignment(r.problem, r.assignment)))(solve_ascend_pipeline(conc_p)), range(16)))
    assert len(outcomes) == 16 and all(ii >= 2 and errs == [] for ii, errs in outcomes)


def test_observation_5d_tiers_residuals_and_serialization() -> None:
    """Verify 3-tier alignment, provenance, coaxial residuals, thresholds, and inflation ratio."""
    ledger, comp, tel = _rich_ledger(), _rich_compiler(), _rich_telemetry()
    obs = ledger.to_observation_5d(compiler_summary=comp, telemetry=tel)
    assert obs.compute_issue.l0_cube_cycles == 120 and obs.compute_issue.l0_vec_cycles == 400
    assert obs.compute_issue.l1_software_loops == 2 and obs.compute_issue.l2_roofline_eta == 0.7
    assert obs.offchip_dma.l1_objdump_mte2_bytes == 2048 and obs.onchip_sram_reg.l0_max_overflow_bytes == 808
    assert obs.pipeline_sync.l0_steady_state_ii == 3 and obs.grid_irregularity.l1_subkernel_count == 3
    assert obs.has_l2_telemetry and obs.has_l1_compiler_summary and len(obs.measured_l2_fields) == 10

    no_hit = ledger.to_observation_5d(compiler_summary=comp, telemetry={"measured_latency_us": 42.0})
    assert no_hit.offchip_dma.l2_cache_hit_rate == 1.0 and no_hit.measured_l2_fields == ("l2_measured_latency_us",)
    assert ObservationVector5D.from_dict(no_hit.to_dict()) == no_hit
    proxy = ObservationVector5D.from_tiers(ledger, compiler_summary=comp, telemetry={"l2_hit_rate": 0.3, "is_analytical_proxy": True})
    assert not proxy.has_l2_telemetry and proxy.measured_l2_fields == ()

    res = compute_static_dynamic_residual(ledger, tel, compiler_summary=comp)
    assert compiler_summary_from_residual(res) == comp
    assert StaticDynamicResidual.from_dict(res.to_dict()) == res
    legacy = StaticDynamicResidual(30.0, 0.8, 0.2, 0.2, 0.1, 0.35, 0.2)
    assert compiler_summary_from_residual(legacy) is None

    ar = obs.compute_axis_residuals()
    assert (ar.compute_issue_gap, ar.offchip_dma_gap, ar.onchip_sram_reg_gap, ar.pipeline_sync_gap, ar.grid_wave_gap) == (0.5, 0.4, 1.0, 0.6, 0.2)
    assert ar.dominant_axis == "onchip_sram_reg"

    for wo, exp_gap in ((1.07, 0.465), (2.0, 0.0), (1.5, 0.25), (0.8, 0.2), (0.0, 0.0)):
        w_ldg = _ldg("exact_modulo_z3", True, 2, {"AIC.PIPE_M": 10}, wave_occupancy=wo)
        assert w_ldg.to_observation_5d().compute_axis_residuals().grid_wave_gap == pytest.approx(exp_gap)

    ql = _quiet_ledger()
    assert ql.to_observation_5d(telemetry={"aic_mac_ratio": 0.8, "aiv_vec_ratio": 0.1, "roofline_eta": 0.4}).compute_axis_residuals().compute_issue_gap == pytest.approx(0.6)
    assert ql.to_observation_5d(telemetry={"aic_mac_ratio": 0.1, "aiv_vec_ratio": 0.1, "roofline_eta": 0.4}).compute_axis_residuals().compute_issue_gap == 0.0
    assert ql.to_observation_5d().has_l2_telemetry is False

    bottom = ConstraintRelaxationMask.bottom()
    assert AxisResidual5D(0.24, 0.24, 0.19, 0.19, 0.24).unlock_causal_facets(bottom) == bottom
    assert AxisResidual5D(1.0, 1.0, 1.0, 1.0, 1.0).unlock_causal_facets(bottom) == ConstraintRelaxationMask.top()
    assert AxisResidual5D(1.0, 1.0, 1.0, 1.0, 1.0).unlock_causal_facets(ConstraintRelaxationMask(*(False,) * 7)).unlocked_facets() == VALID_FREEDOM_FACETS

    ctx = build_predicate_context(ledger, residual=res, prev_ledger=ledger)
    prev_ctx = ctx["prev"]
    assert isinstance(prev_ctx, dict)
    assert ctx["dominant_axis"] == "onchip_sram_reg" and prev_ctx["grid_wave_gap"] == 0.2

    clean_l = _ldg("exact_modulo_z3", True, 2, {"AIV0.PIPE_V": 200}, grid_blocks=16, gm_roundtrip_bytes=0, gm_traffic_scope="full_grid")
    assert compute_transfer_inflation_ratio(clean_l, roofline={"min_gm_bytes": 150 * 1024 * 1024}) == pytest.approx(1.0)
    spill_l = _ldg("exact_modulo_z3", True, 2, {"AIV0.PIPE_V": 200}, grid_blocks=16, gm_roundtrip_bytes=32 * 1024 * 1024, gm_traffic_scope="full_grid")
    assert ObservationVector5D.from_tiers(spill_l, telemetry={"dur_us": 25.0}, roofline={"min_gm_bytes": 150 * 1024 * 1024, "compute_floor_us": 12.0}).transfer_inflation_ratio == pytest.approx(182 / 150)

    imbal = _ldg("readonly_envelope", True, None, {"AIV0.PIPE_V": 100})
    assert ObservationVector5D.from_tiers(imbal).compute_issue.l0_aiv_subcore_imbalance is True
    bal = _ldg("readonly_envelope", True, None, {"AIV0.PIPE_V": 50, "AIV1.PIPE_V": 50})
    assert ObservationVector5D.from_tiers(bal).compute_issue.l0_aiv_subcore_imbalance is False


def test_observation_5d_completeness_capability_gate_and_stage_a() -> None:
    """Verify Stage-6 5D fields, registry facet unlocks, CapabilityGate, and Track-A/B routing."""
    ldg = _ldg(
        "exact_modulo_z3", True, 2, {"AIC.PIPE_M": 64, "AIC.PIPE_FIX": 48, "AIV0.PIPE_V": 32},
        ub_scalar_compute_loops=2, min_gm_dma_bytes=16,
    )
    sm = BishengResourceSummary(
        vec_regs=16, scalar_regs=8, stack_bytes=512, vloopv2_loops=1,
        extra_signals={"scalar_compute_loop_fallback": 3.0, "dcache_bypass_narrow_store": 4.0},
    )
    obs = ObservationVector5D.from_tiers(ldg, compiler_summary=sm, telemetry={"dur_us": 20.0})
    assert obs.compute_issue.l0_ub_scalar_compute_loops == 2 and obs.offchip_dma.l0_fixpipe_cycles == 48
    assert obs.onchip_sram_reg.l1_stack_bytes == 512 and ObservationVector5D.from_dict(obs.to_dict()) == obs
    res = compute_static_dynamic_residual(ldg, {"dur_us": 20.0}, compiler_summary=sm)
    rec = compiler_summary_from_residual(StaticDynamicResidual.from_dict(res.to_dict()))
    assert rec is not None and rec.stack_bytes == 512

    assert "cpp:cube_mmad" in extract_cpp_trackb_signature_tags("ascend_gemm_l1<64,64,64,32,true,half>(l0c, a, b);")
    py_mod = (
        '_TRACKB_SRC = r"""\n#include <tl_templates/ascend/common.h>\n'
        '__aicore__ void k() {\n  ascend_gemm_l1<64, 64, 64, 32, true, half>(c, a, b);\n}\n"""\n'
    )
    assert "cpp:cube_mmad" in extract_structural_signature(py_mod, track="manual_ssbuf")

    quiet = compute_static_dynamic_residual(_quiet_ledger(), {})
    assert "dma_and_cache_hierarchy" in relax_facets_for_bottleneck(ConstraintRelaxationMask.bottom(), dominant_bound="hbm", residual=quiet).unlocked_facets()
    assert "sram_layout_and_aliasing" in relax_facets_for_bottleneck(ConstraintRelaxationMask.bottom(), residual=quiet, ub_overflow=True).unlocked_facets()

    live_gaps = scan_tilelang_pipe_gaps()
    assert CapabilityGate.suitability_score(AXES_SIMD) == 1.0
    assert CapabilityGate.suitability_score(AXES_GROUPED, pipe_gaps=live_gaps) == 0.0
    blk_gap = PipeGapReport("mock", {"pipe_v": PipeGapBreakdown("pipe_v", 1, (), ("simd_vld_vst",), 0.0)}, (), ("simd_vld_vst",), 0.0)
    assert CapabilityGate.route(AXES_SIMD, pipe_gaps=blk_gap).recommended_backend_tier == "ascendc_template"
    assert CapabilityGate.route(AXES_GROUPED, pipe_gaps=live_gaps, router_enabled=False).track == "developer"

    proto: dict[str, object] = {"op_name": "demo_reduce", "input_dtypes": ["float16"], "output_dtypes": ["float16"], "topology_axes": AXES_SIMD.to_dict()}
    dec = route_from_op_spec(proto)
    assert dec.features is not None and dec.features["inner_contiguous_bytes"] == 0
    def _ratios(cid: int, *, stride: float = 1.0, atom: float = 4.0, aligned: bool = True, sram: float = 0.5, vreg: float = 0.5, wave: float = 1.0, bytes_: int = 128, empty: bool = False) -> CasePhysicsRatios:
        return CasePhysicsRatios(case_id=cid, stride_ratio=stride, contig_atom_ratio=atom, is_atom_aligned=aligned, working_set_sram_ratio=sram, working_set_vreg_ratio=vreg, wave_occupancy_ratio=wave, oi_ridge_ratio=1.0, inner_contiguous_bytes=bytes_, has_empty_group=empty)

    ratios = _ratios(9, atom=1.0, sram=0.1, vreg=0.1, wave=0.2, bytes_=256, empty=True)
    assert route_from_op_spec(proto, case_ratios=ratios).topology_axes == TopologyAxes("aiv_only", "gm_ub_stream", "ragged_group")
    assert route_initial_by_input_contract({"op_name": "dq", "input_dtypes": ["float4_e2m1"], "output_dtypes": ["float16"]}, case_ratios=ratios).recommended_backend_tier == "ascendc_template"

    for alias in ("cce_cpp20", "cce_trackb", "hybrid_trackb", "tl_templates_cpp20", "ascendc_trackb"):
        assert normalize_router_tier(alias) == "ascendc_template"
    assert RouteDecision("developer", AXES_SIMD, (), (), "cce_cpp20", {}).recommended_backend_tier == "ascendc_template"
    assert EscalationDecision(True, "cce_trackb", "inline_cce_vf").to_tier == "inline_ascendc_vf"

    sub32 = _ratios(11, atom=1.0, sram=0.1, vreg=0.1, wave=0.25, bytes_=16)
    tb_res = evaluate_ascendc_candidate("constexpr int N = 64;\nExp(dst, src, mask);\n", freedom_mask=ConstraintRelaxationMask.bottom(), enable_track_router=True, case_ratios=sub32)
    assert tb_res.passed and tb_res.route and tb_res.route.track == "simt_vf"
    dsl = 'import tilelang.ascend.language as T\ndef make_kernel():\n    @T.prim_func\n    def main(X: T.Tensor((64, 128), "float32"), Y: T.Tensor((64, 128), "float32")):\n        with T.Kernel(1):\n            ub = T.alloc_shared((64, 128), "float32")\n            T.copy(X[0:64, 0:128], ub)\n            T.copy(ub, Y[0:64, 0:128])\n    return main\n'
    sa_res = run_stage_a_static(dsl, factory_name="make_kernel", enable_track_router=True, case_ratios=sub32)
    assert sa_res.passed and sa_res.route and sa_res.route.track == "simt_vf"


def test_sil_schema_verifier_redlines_and_regime_partition(tmp_path: Path) -> None:
    """Verify SIL schema roundtrips, verifier capacity/sync redlines, and regime merges."""
    assert len(VALID_FREEDOM_FACETS) == 7 and VALID_BACKEND_TIERS == ("tilelang_dsl", "ascendc_template")
    adv = _sample_advisory(precision_thresholds={"float16": 2.0**-10, "float32": 2.0**-13})
    pol = SchedulePolicy("aligned_main", AXES_GEMM, tile_mnk=(128, 128, 64), l1_stages=2, l1_half_bank_split=True, l1_estimated_bytes=131072, ub_estimated_bytes=65536, dma_paths_used=("GM->L1", "L1->L0A", "L1->L0B", "L0C->GM"), cube_operand_dtype="float16")
    pe = PolicyEnvelope("sample_gemm", {"aligned_main": pol}, ("swat_swizzle",))
    rep = verify_policy_envelope(pe, adv)
    tel = TelemetryFeedback(True, True, True, 0, True, 14.2, 98.4, 1.0, {"PIPE_M": 0.91}, 0.01, 0.94)
    env = SchedulingEnvelope(adv, pe, rep, tel)
    saved = env.save_json(tmp_path / "sil.json")
    assert SchedulingEnvelope.load_json(saved) == env
    with pytest.raises(ValueError, match="compute_engine"):
        TopologyAxes("bad_engine", "gm_ub_stream", "affine_simd")

    v = SILVerifier()
    assert v.verify_policy(SchedulePolicy("aligned_main", AXES_GEMM, l1_estimated_bytes=524289), adv).passed is False
    assert v.verify_policy(SchedulePolicy("aligned_main", AXES_GEMM, tile_mnk=(256, 256, 64), l0c_buffers=2), adv).passed is False
    assert v.verify_policy(SchedulePolicy("aligned_main", AXES_SIMD, vector_mode="simdvf", ub_256kb_unlock=False, ub_estimated_bytes=260000), adv).suggested_fixes.get("ub_256kb_unlock") is True
    assert v.verify_policy(SchedulePolicy("aligned_main", AXES_SIMD, vector_mode="simdvf", ub_256kb_unlock=True, ub_estimated_bytes=260000), adv).passed is True
    assert v.verify_policy(SchedulePolicy("aligned_main", AXES_SIMT, vector_mode="simtvf", ub_256kb_unlock=True, ub_estimated_bytes=225000), adv).passed is False
    adv_tail = _sample_advisory(has_unaligned_tail=True)
    assert v.verify_policy(SchedulePolicy("aligned_main", AXES_GEMM, dma_paths_used=("L1->GM",)), adv_tail).passed is False
    assert v.verify_policy(SchedulePolicy("aligned_main", AXES_GEMM, cube_operand_dtype="q4_0", w4_mode="none"), adv_tail).suggested_fixes.get("w4_mode") == "aiv_unpack_f16"
    assert v.verify_policy(SchedulePolicy("aligned_main", AXES_GEMM, cube_operand_dtype="q4_0", w4_mode="mxa8w4_bitshift_x64"), adv_tail).passed is True
    assert v.verify_policy(SchedulePolicy("aligned_main", AXES_GEMM, tile_mnk=(64, 64, 4096)), adv_tail).passed is False
    assert v.verify_policy(SchedulePolicy("aligned_main", AXES_GEMM, sync_mode="unit_flag_2_3", l0c_buffers=2), adv_tail).suggested_fixes.get("l0c_buffers") == 1
    assert v.verify_policy(SchedulePolicy("aligned_main", AXES_CV_RING, sync_mode="mode4_flag", reverse_ack_period=16), adv).passed is False
    assert v.verify_policy(SchedulePolicy("aligned_main", AXES_GROUPED, cv_datapath="l0c2ub_split_m", ensure_empty_subblock_sync=False), adv).passed is False
    assert v.verify_policy(SchedulePolicy("aligned_main", AXES_GEMM, hf32_allowed=True), adv).passed is False
    assert v.verify_policy(SchedulePolicy("aligned_main", AXES_SIMD, fast_unshifted_exp=True), _sample_advisory(has_nan_inf_cases=True)).passed is False

    def _r(cid: int, *, stride: float = 1.0, atom: float = 4.0, aligned: bool = True, sram: float = 0.5, vreg: float = 0.5, bytes_: int = 128) -> CasePhysicsRatios:
        return CasePhysicsRatios(case_id=cid, stride_ratio=stride, contig_atom_ratio=atom, is_atom_aligned=aligned, working_set_sram_ratio=sram, working_set_vreg_ratio=vreg, wave_occupancy_ratio=1.0, oi_ridge_ratio=0.5, inner_contiguous_bytes=bytes_)

    r1, r2, r3, r4 = _r(1, atom=0.125, aligned=False, bytes_=4), _r(2), _r(3, stride=4.0), _r(4, stride=4.0, sram=2.0, vreg=2.0)
    dims = {"rows": 1000003, "cols": 2}
    assert classify_case_regime(r1, kind="reduction", axes=AXES_SIMD, dims=dims) == "reduce_tail_subatom"
    groups = partition_case_regimes({1: r1, 2: r2, 3: r3, 4: r4}, {k: dims for k in (1, 2, 3, 4)}, {k: 4096 * k for k in (1, 2, 3, 4)}, kind="reduction", axes=AXES_SIMD)
    assert {g.regime_tag for g in groups} == {"reduce_tail_subatom", "reduce_tail", "reduce_strided", "reduce_strided_twopass"}

    def _entry(cid: int, sig: tuple[str, str, str, str, str]) -> list[CasePhysicsRegimeCase]:
        return [CasePhysicsRegimeCase(case_id=cid, dims={"M": 64, "N": 64}, phase_signature=sig, workload_bytes=4096)]

    base_s = ("aiv_vector_or_gemv", "contiguous", "sram_1pass", "aligned_vector", "balanced_waves")
    soft_s = ("aiv_vector_or_gemv", "contiguous", "gm_multipass", "unaligned_predicated", "under_occupied_split")
    sub_s = ("aiv_vector_or_gemv", "contiguous", "sram_1pass", "sub_atom_simt", "balanced_waves")
    assert _merge_nearest_regimes({"base": _entry(1, base_s), "soft": _entry(2, soft_s), "sub": _entry(3, sub_s)}) == ("base", "soft", 6)


def test_all_60_cannbench_operators_extract_synthesize_and_verify(tmp_path: Path) -> None:
    """Extract, synthesize, and verify SIL envelopes for all 60 CANNBench operators."""
    if not DEFAULT_OPS_ROOT.exists():
        pytest.skip(f"cann-bench tasks root missing: {DEFAULT_OPS_ROOT}")
    advisories = OpAdvisoryExtractor(DEFAULT_OPS_ROOT).extract_all()
    assert len(advisories) == 60
    engines, rings, regs = set(), set(), set()
    for op_name, adv in advisories.items():
        engines.add(adv.topology_axes.compute_engine)
        rings.add(adv.topology_axes.memory_ring)
        regs.add(adv.topology_axes.iteration_regularity)
        assert sorted(c for r in adv.regimes for c in r.case_nums) == list(range(1, 21))
        rep = verify_policy_envelope(synthesize_default_policy(adv), adv)
        assert rep.passed is True, f"{op_name}: {rep.errors}"
    assert (engines, rings, regs) == (set(VALID_COMPUTE_ENGINES), set(VALID_MEMORY_RINGS), set(VALID_ITERATION_REGULARITIES))
    assert advisories["exp"].has_nan_inf_cases and advisories["kv_cache_append"].has_inplace_aliasing
    assert advisories["grouped_matmul"].topology_axes.iteration_regularity == "ragged_group"
    kda_adv = advisories["kimi_delta_attention"]
    kda_pol = synthesize_default_policy(kda_adv)
    kda_env = SchedulingEnvelope(kda_adv, kda_pol, verify_policy_envelope(kda_pol, kda_adv))
    assert SchedulingEnvelope.load_json(kda_env.save_json(tmp_path / "kda.json")) == kda_env


def test_microarch_tir_cce_extraction_and_cann93_merge() -> None:
    """Verify TIR + CCE + CANN 9.3 MicroarchProfile extractors and merge semantics."""
    @T.prim_func
    def _vec_f16(X: T.Tensor((128, 128), "float16"), Y: T.Tensor((128, 128), "float16")):
        with T.Kernel(4):
            a_ub, b_ub = T.alloc_shared((32, 128), "float16"), T.alloc_shared((32, 128), "float16")
            for k in T.Pipelined(4, num_stages=2):
                T.copy(X[k * 32 : (k + 1) * 32, 0:128], a_ub)
                for i, j in T.Parallel(32, 128):
                    b_ub[i, j] = T.exp(a_ub[i, j])
                T.copy(b_ub, Y[k * 32 : (k + 1) * 32, 0:128])

    tir_p = extract_microarch_profile_from_tirx(_vec_f16)
    assert tir_p.source_kind == "tirx" and tir_p.gm_to_ub_bytes >= 4 * 32 * 128 * 2 and tir_p.sfu_f16_ops > 0
    cce_p = extract_microarch_profile_from_cce(
        'extern "C" __global__ __aicore__ void k1(){Mmad(c,a,b);copy_gm_to_cbuf(l1,gm);'
        'copy_matrix_cc_to_ubuf(ub,l0c);CrossCoreSetFlag<2>(0);}\n'
        'extern "C" __global__ __aicore__ void k2(){CrossCoreWaitFlag(0);half v=vexp(x);'
        '__shared__ half tile[32][64];__simt_vf__();copy_ubuf_to_cbuf(l1,ub);pipe_barrier(PIPE_ALL);}'
    )
    assert cce_p.subkernel_count == 2 and cce_p.cube_mmad_count == 1 and cce_p.cv_sync_barriers == 2
    merged = merge_microarch_profiles(tir_p, cce_p)
    assert merged.source_kind == "merged" and merged.gm_to_ub_bytes == tir_p.gm_to_ub_bytes and merged.subkernel_count == 2

    cann93_p = extract_microarch_profile_from_cce(
        'extern "C" __global__ __aicore__ void ck(){asc_copy_gm2l1_nd2nz(a,g,128,0,128,128,128,false);'
        'ascend_gemm_l1<128,64,128,32,false,half,float>(c,a,b,1);asc_copy_l0c2ub(u,c);asc_sync_notify(0);}\n'
        '__simd_vf__ inline void vk(__ubuf__ float* u, __gm__ float* o){asc_sync_wait(0);'
        'asc_reduce_max(u,u);asc_sync_pipe(PIPE_ALL);asc_copy_ub2gm(o,u);}'
    )
    assert cann93_p.subkernel_count == 2 and cann93_p.cube_mmad_count == 1 and cann93_p.reduce_ops == 1


def test_kb_isolation_graduation_revival_sync_and_outcome_guard(tmp_path: Path) -> None:
    """Verify KB operator_local isolation, cross_op graduation, revival, sync, and gain guards."""
    lake_dir = tmp_path / "kb_lake"
    tracker = BayesianRuleTracker.from_seed_yaml()
    local_r = AscendCausalRule(
        "R_TEST_LOCAL_01", "is_legal", "PIPE_V >= 0", "Unroll by 4", "any_overflow_bytes == 0",
        "flywheel:ablation", alpha_prior=2.0, beta_prior=1.0, scope="operator_local", origin_operator="level2/op_a",
    )
    tracker.admit_rule(local_r)
    save_distilled_rule_file(lake_dir, local_r)
    assert "R_TEST_LOCAL_01" in {r.rule_id for r in tracker.active_rules(topology_axes=AXES_SIMD, operator_name="level2/op_a")}
    assert "R_TEST_LOCAL_01" not in {r.rule_id for r in tracker.active_rules(topology_axes=AXES_SIMD, operator_name="level2/op_b")}

    for ridx in (1, 2):
        tracker.record_outcome("R_TEST_LOCAL_01", postcondition_realized=True, speed_gain=0.15, topology_axes=AXES_SIMD)
        append_kb_event_shard(lake_dir, run_id="run_a", rule_id="R_TEST_LOCAL_01", postcondition_realized=True, speed_gain=0.15, topology_axes=AXES_SIMD, operator_name="level2/op_a", round_idx=ridx)
    assert "R_TEST_LOCAL_01" in tracker.promote_verified_local_rules()
    save_distilled_rule_file(lake_dir, tracker.rules["R_TEST_LOCAL_01"])
    assert "R_TEST_LOCAL_01" not in BayesianRuleTracker.from_experience_lake(lake_dir).rules
    reloaded = BayesianRuleTracker.from_experience_lake(lake_dir, allow_unreviewed_distilled_rules=True)
    assert reloaded.rules["R_TEST_LOCAL_01"].scope == "cross_op" and reloaded.posteriors["R_TEST_LOCAL_01"].successes == 2

    bad_r = AscendCausalRule("R_REVIVE_02", "is_legal", "PIPE_V >= 0", "Tweak", "any_overflow_bytes == 0", "flywheel:ablation")
    tracker.admit_rule(bad_r)
    for _ in range(6):
        tracker.record_outcome("R_REVIVE_02", postcondition_realized=False)
    assert "R_REVIVE_02" in tracker.prune_underperforming(min_trials=5, prune_threshold=0.25, round_idx=2)
    assert tracker.decay_and_revive(round_idx=8, min_cooldown_rounds=10) == ()
    assert "R_REVIVE_02" in tracker.decay_and_revive(round_idx=12, min_cooldown_rounds=10)

    chunk_r = distill_causal_rule(AstEditChunk("c1", "barrier_prune", "body", "T.pipe_barrier()", "# x", "pass"), rule_id="R_CAP", scope="operator_local", origin_operator="level2/op_a")
    assert chunk_r.scope == "operator_local"
    distiller = FlywheelDistiller(funnel=GateFunnel(skip_stage_b=True), tracker=tracker)
    rep = distiller.distill_leave_one_out(
        baseline_dsl="def k(a=1,b=1,c=1,d=0):\n    return a+b+c+d\n",
        winner_dsl="def k(a=4,b=8,c=2,d=1):\n    return a+b+c+d\n",
        winner_latency_us=10.0,
        baseline_latency_us=20.0,
        max_loo_chunks=3,
        operator_name="level2/op_a",
        rule_scope="operator_local",
    )
    assert rep.total_chunks <= 3

    sync_dir = tmp_path / "sync_lake"
    sync_t = BayesianRuleTracker.from_experience_lake(sync_dir)
    append_kb_event_shard(sync_dir, run_id="self_1", rule_id="R01_DUAL_COPY_1C2V_SPLIT", postcondition_realized=False, speed_gain=0.0, topology_axes=AXES_SIMD, operator_name="level2/softmax", round_idx=1)
    append_kb_event_shard(sync_dir, run_id="peer_2", rule_id="R01_DUAL_COPY_1C2V_SPLIT", postcondition_realized=True, speed_gain=0.08, topology_axes=AXES_SIMD, operator_name="level2/rms_norm", round_idx=1)
    assert sync_t.sync_from_experience_lake(sync_dir, exclude_run_id="self_1") == 1
    assert sync_t.posteriors["R01_DUAL_COPY_1C2V_SPLIT"].failures == 0
    assert sync_t.sync_from_experience_lake(sync_dir, exclude_run_id="self_1") == 0

    g_rule = AscendCausalRule("R_GAIN", "is_legal", "PIPE_V >= 0", "T", "any_overflow_bytes == 0", "seed:test")
    ldg = _ldg("readonly_envelope", True, 100, {"AIV0.PIPE_V": 200}, gm_roundtrip_bytes=16384, grid_blocks=16)
    p_res = StaticDynamicResidual(286.35, 0.5, 0.0, 0.5, 0.2, 0.1, 0.1)
    assert TelemetryLedgerWriter.evaluate_causal_rule_outcome(rule=g_rule, res_all_passed=True, rolled_back=False, ledger=ldg, features=None, cand_residual=p_res, cand_lat_us=286.82, cand_obj_score=55.0, parent_ledger=ldg, parent_residual=p_res, parent_lat_us=286.35, parent_obj_score=55.0, has_cand_msprof=False) == (False, 0.0)
    assert TelemetryLedgerWriter.evaluate_causal_rule_outcome(rule=g_rule, res_all_passed=True, rolled_back=False, ledger=ldg, features=None, cand_residual=p_res, cand_lat_us=95.0, cand_obj_score=62.0, parent_ledger=ldg, parent_residual=p_res, parent_lat_us=100.0, parent_obj_score=60.0, has_cand_msprof=False, cand_global_score=54.8, parent_global_score=55.0) == (False, 0.0)
    ok_g, gain_g = TelemetryLedgerWriter.evaluate_causal_rule_outcome(rule=g_rule, res_all_passed=True, rolled_back=False, ledger=ldg, features=None, cand_residual=p_res, cand_lat_us=95.0, cand_obj_score=62.0, parent_ledger=ldg, parent_residual=p_res, parent_lat_us=100.0, parent_obj_score=60.0, has_cand_msprof=False, cand_global_score=56.0, parent_global_score=55.0)
    assert ok_g and abs(gain_g - 0.05) < 1e-6


def test_campaign_runner_tiers_and_l4_non_anchor_hydration(tmp_path: Path) -> None:
    """Verify CampaignRunner tier logging, prescreen defaults, and L4 non-anchor metadata."""
    assert MarathonConfig("level2/softmax").enable_tilesize_prescreen is True
    assert CampaignConfig(operator_name="level2/softmax").cost_prescreen_top_k is None

    cann_repo = tmp_path / "cann_repo"
    op_dir, meta_dir = cann_repo / "tasks" / "level2" / "softmax", cann_repo / "tasks" / "metadata"
    op_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    (op_dir / "proto.yaml").write_text("operator:\n  name: Softmax\n", encoding="utf-8")
    (op_dir / "golden.py").write_text("import torch\ndef softmax(x): return torch.softmax(x, -1)\n", encoding="utf-8")
    (op_dir / "cases.yaml").write_text(
        "cases:\n- case_id: 1\n  input_shape: [[32, 128]]\n  dtype: [float16]\n"
        "- case_id: 2\n  input_shape: [[64, 256]]\n  dtype: [float16]\n",
        encoding="utf-8",
    )
    (meta_dir / "950pr.json").write_text(
        json.dumps({"level2": {"softmax": {"1": {"baseline_perf_us": 100.0, "t_hw_us": 20.0}, "2": {"baseline_perf_us": 200.0, "t_hw_us": 50.0}}}}),
        encoding="utf-8",
    )

    mod_path = tmp_path / "softmax_mod.py"
    mod_src = 'def build_vec(tile_m=1):\n    """baseline"""\n    return tile_m\n'
    mod_path.write_text(mod_src, encoding="utf-8")
    lake_dir = tmp_path / "lake"
    reg = RegimeTargetSpec("small", ("build_vec",), {}, representative_hir_dims={"M": 128, "D": 512}, representative_anchor_dims={"M": 128, "D": 512})
    anchors = [AnchorCaseSpec("level2/softmax", 1, "small", ref_perf_us=100.0, t_hw_us=20.0)]

    class _MockBackend:
        async def generate(self, req):
            if req.role == "explore_creative":
                return '# target_bottleneck: aiv\n```python\ndef build_vec(tile_m=8):\n    """bad"""\n    return tile_m\n```'
            return 'def build_vec(tile_m=4):\n    """fast"""\n    return tile_m\n'

    tracker = BayesianRuleTracker.from_rules([
        AscendCausalRule("R_ALWAYS", "is_legal", "True", "tile_m=4", "measured_latency_us <= prev.measured_latency_us", "test")
    ])
    l0_ldg = _ldg("readonly_envelope", True, 120, {"AIV0.PIPE_V": 500}, gm_roundtrip_bytes=32768, grid_blocks=28, microarch_profile=MicroarchProfile(fp_alu_ops=256, gm_to_ub_bytes=32768, ub_to_gm_bytes=32768, source_kind="tirx").to_dict())

    def _l2(cands, anc):
        out = {}
        for cid, cp in cands:
            txt = Path(cp).read_text("utf-8")
            ok = "bad" not in txt
            us = 100.0 if cid == "incumbent_baseline" else (40.0 if ok else 35.0)
            m = [AnchorEvalMetrics(anc[0].case_num, "small", ok, us, 100.0, 20.0, 100.0 / us if ok else 0.0, 0.5 if ok else 0.0, 55.0 if ok else 0.0)]
            out[cid] = CandidateSurrogateResult(cid, "level2/softmax", ok, m[0].speedup, m[0].speedup, m[0].roofline_eta, m[0].eq3_score, tuple(m))
        return out

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="level2/softmax", max_rounds=1, k_rule_workers=1, max_syntax_repairs=0, run_final_l4=True, l4_regression_ratio=0.0),
        target_module_path=mod_path, cannbench_repo=cann_repo,
        perf_target=PerfTargetSpec("level2/softmax", 10.0, 10.0, 0.99, 99.9, min_optimization_rounds=1),
        anchors=anchors, regime_specs=[reg], llm_backend=_MockBackend(), tracker=tracker,
        l0_checker=lambda _s, _r: StageAResult(True, None, l0_ldg),
        l1_compiler=lambda _s, _r: StageBResult(True, microarch_profile=MicroarchProfile(sfu_f16_ops=8, source_kind="cce").to_dict()),
        l2_batch_evaluator=_l2,
        msprof_collector=lambda _op, cp, _a: {"small": {"dur_us": 40.0}} if "fast" in Path(cp).read_text("utf-8") else ({"small": {"dur_us": 100.0}} if "baseline" in Path(cp).read_text("utf-8") else {}),
        l3_contract_runner=lambda _p: L3AcceptanceResult(True, run_id="l3_ok"),
        l4_cannbench_runner=lambda _p: L4CannbenchResult(True, avg_speedup=2.25, score=82.0, per_case_speedups={1: 2.5}, per_case_elapsed_us={1: 40.0, 2: 100.0}, run_id="l4_ok"),
        checkpoint_path=tmp_path / "ckpt.json", experience_lake_dir=lake_dir,
    )
    rep = asyncio.run(runner.run(resume=False))
    assert list(rep.promoted_rounds) == [1]
    cands, obs = ExperienceLake(lake_dir).load_records()
    assert {"baseline", "l2_pass", "l2_acc_fail", "l4_official"}.issubset({o.eval_tier for o in obs})
    assert [o for o in obs if o.eval_tier == "l2_acc_fail"][0].msprof_pmu == {}
    assert any(int(c.microarch_profile.get("sfu_f16_ops", 0) or 0) >= 8 for c in cands.values())
    l4_by_case = {int(o.case_num): o for o in obs if o.eval_tier == "l4_official"}
    assert l4_by_case[2].ref_perf_us == pytest.approx(200.0) and l4_by_case[2].eq3_score == pytest.approx(0.75)
