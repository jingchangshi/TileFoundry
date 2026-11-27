"""Unattended performance-driven campaign runner delegating to focused coordinators."""

from __future__ import annotations

import argparse, ast, asyncio, hashlib, json, logging, math, os, shutil, sys, tempfile, time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from tilefoundry_ascend.agent.campaign.calibration_service import CalibrationService, CalibrationSnapshot
from tilefoundry_ascend.agent.campaign.flywheel_coordinator import FlywheelCoordinator
from tilefoundry_ascend.agent.campaign.funnel_telemetry import DeterministicSearchBoundaryLedger, compute_spearman_rho
from tilefoundry_ascend.agent.campaign.helpers import (
    ensure_candidate_bytes, extract_valid_non_negative_float, is_regime_satisfied,
    lookup_case_dims_safe, persist_runtime_state, regime_mean_recoverable_excess_us,
    resolve_cann_root_from_env_file, resolve_operator_entrypoint_safe, resolve_operator_semantics,
    resolve_trackb_freedom_mask, run_l0_gate, select_active_regime, slice_contract_instruction,
    source_sha256, surrogate_rank_key, to_conservation_point, write_source_and_verify,
)
from tilefoundry_ascend.agent.campaign.l3_l4_recovery import (
    CAMPAIGN_DEVICE_IDS_ENV_KEY, L3L4RecoveryCoordinator, PendingL4, canonical_device_id,
    parse_device_ids, resolve_campaign_device_plan, resolve_l4_device_pool,
)
from tilefoundry_ascend.agent.campaign.promotion_coordinator import (
    PendingPromotion, PromotionCoordinator, RankedL2Candidate, RankedL2Survivors, select_top_two_l2_survivors,
)
from tilefoundry_ascend.agent.campaign.regime_slice_patcher import (
    ResolvedRegimeConfig, _collect_top_level_symbols, _has_prim_func_builder, _looks_like_bare_trackb_cpp,
    _parses_as_python, _render_trackb_tu_from_module, _resolve_selector_compile_flags,
    _trackb_python_residual, apply_ast_patch, extract_regime_snippet, patch_regime_into_module,
    resolve_regime_config, split_readonly_host_tail,
)
from tilefoundry_ascend.agent.campaign.round_context_builder import ChampionState, RoundContextBuilder
from tilefoundry_ascend.agent.campaign.seed_synthesizer import (
    GoldenEntrypointContract, OracleAccuracyResult, SeedSynthesisOutcome,
    extract_golden_entrypoint_contract, prepare_operator_seed_or_takeover,
    validate_entrypoint_signature_compatibility,
)
from tilefoundry_ascend.agent.campaign.stage_gate_coordinator import (
    CandidateBatch, StageGateCoordinator, run_l0_l1_round_filter,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    FAILURE_KIND_CANDIDATE, _LLM_API_KEY_ENV, _NON_AGENT_GATES_ENV_KEYS, AnchorEvalMetrics,
    CandidateSurrogateResult, L0L1CandidateCheck, L2BatchResults, L3AcceptanceResult,
    L4CannbenchResult, _format_env_line, _validate_llm_api_key_env, bottleneck_anchor_for_regime,
    collect_bottleneck_msprof_via_agent_gates, collect_p2_deep_dive_via_agent_gates,
    extract_ordered_tensor_shapes, run_l3_full_contract_via_agent_gates,
    run_micro_probe_via_agent_gates, run_surrogate_batch_via_agent_gates,
)
from tilefoundry_ascend.agent.campaign.telemetry_ledger_writer import TelemetryLedgerWriter
from tilefoundry_ascend.agent.contracts import CampaignRoundRecord, CampaignStateSnapshot
from tilefoundry_ascend.agent.cost_advisor import CostModelAdvisor, create_cost_advisor
from tilefoundry_ascend.agent.flywheel.ablation_distiller import FlywheelDistiller
from tilefoundry_ascend.agent.kb.bayesian_tracker import (
    BayesianRuleTracker, append_rule_misapplication, append_winning_edge,
)
from tilefoundry_ascend.agent.kb.schema import AscendCausalRule, build_predicate_context
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    StageAResult, check_npu_entrypoint_purity, evaluate_ascendc_candidate,
    resolve_trackb_compile_flags, run_stage_a_static, run_stage_b_ascendc_compile,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import KernelLLMBackend
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    BeamCandidate, TwoLaneBeamSearch, compiler_summary_from_residual,
    compute_static_dynamic_residual, extract_structural_signature,
)
from tilefoundry_ascend.agent.orchestrator.stage_b_gate import run_l1_compile_check
from tilefoundry_ascend.agent.orchestrator.tilesize_prescreen import rank_stage_a_candidates_with_cost_model
from tilefoundry_ascend.agent.specs.anchor_specs import (
    AnchorCaseSpec, CannbenchOpsRegistry, GuardrailCaseSpec, PerfTargetSpec, RegimeTargetSpec,
    default_source_root, discover_module_trackb_symbols, extract_trackb_cpp_source,
    get_default_anchors, get_default_perf_target, get_default_regime_specs,
    get_operator_module_relpath, get_operator_task, parse_simple_env_file, resolve_regime_for_case,
)
from tilefoundry_ascend.agent.specs.fault_types import FaultPlane, L3Resolution, coerce_fault_plane
from tilefoundry_ascend.agent.specs.latency_algebra import calibrate_dispatch_floor_us, geometric_speedup
from tilefoundry_ascend.ascend.cost_model import BishengResourceSummary
from tilefoundry_ascend.ascend.cost_model.experience_lake import (
    ExperienceLake, resolve_default_experience_lake_dir,
)
from tilefoundry_ascend.ascend.json_record import atomic_write_text
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import HirBlueprintPacket
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger
from tilefoundry_ascend.ascend.schedule.regime_solver import RegimePiecewiseSolver, RegimePlanResult
from tilefoundry_ascend.ascend.sil.pytorch_frontend import PyTorchOpSemantics
from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask, TopologyAxes
from tilefoundry_ascend.ascend.target import AscendTarget
from tilefoundry_ascend.ascend.target.dialect_surface import CCE_CPP20_SURFACE

logger = logging.getLogger(__name__)
_LLM_ENV_PREFIX = "TILEFOUNDRY_NPU_LLM_"
_LLM_CLI_OPTIONS = {
    "llm_backend": ("BACKEND", "Generation backend."),
    "exploit_model": ("EXPLOIT_MODEL", "Model for rule-guided exploitation."),
    "explore_model": ("EXPLORE_MODEL", "Model for creative exploration."),
    "escalate_model": ("ESCALATE_MODEL", "Model for escalated exploration."),
    "llm_api_base": ("API_BASE", "Base URL for the HTTP generation backend."),
}
_LEGACY_AGY_ENV_KEYS = frozenset({
    "TILEFOUNDRY_NPU_AGY_BIN", "TILEFOUNDRY_NPU_AGY_EXPLOIT_MODEL",
    "TILEFOUNDRY_NPU_AGY_EXPLORE_MODEL", "TILEFOUNDRY_NPU_AGY_ESCALATE_MODEL",
    "TILEFOUNDRY_NPU_AGY_COMMON_ARGS", "TILEFOUNDRY_NPU_AGY_MAX_CONCURRENCY",
})
_AGENT_GATES_ALLOWED_ENV_KEYS = frozenset({
    "TILEFOUNDRY_NPU_CAMPAIGN_DEVICE_IDS", "TILEFOUNDRY_NPU_CANN_SETUP", "TILEFOUNDRY_NPU_CONDA_SH",
    "TILEFOUNDRY_NPU_CONDA_ENV", "TILEFOUNDRY_NPU_CONDA_ENVS_PATH", "TILEFOUNDRY_NPU_PYTHON",
    "TILEFOUNDRY_NPU_TILELANG_SOURCE", "TILEFOUNDRY_NPU_TILELANG_BUILD", "TILEFOUNDRY_NPU_CPU_TESTS",
    "TILEFOUNDRY_NPU_DEVICE_TESTS", "TILEFOUNDRY_NPU_BUILD_SUBDIR", "TILEFOUNDRY_NPU_CANNBENCH_REPO",
    "TILEFOUNDRY_NPU_WORK_ROOT", "TILEFOUNDRY_NPU_CANNBENCH_ARCH", "TILEFOUNDRY_NPU_CANNBENCH_TARGETS_FILE",
    "TILEFOUNDRY_NPU_CANNBENCH_TARGETS", "TILEFOUNDRY_NPU_CANNBENCH_MCP_URL",
    "TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN", "TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN_FILE",
    "TILEFOUNDRY_NPU_CANNBENCH_BENCHMARK_VERSION", "TILEFOUNDRY_NPU_CANNBENCH_POOL_ID",
    "TILEFOUNDRY_NPU_CANNBENCH_AGGREGATION_TOKEN", "TILEFOUNDRY_NPU_CANNBENCH_INDEPENDENT_ENV_CONSENT",
    "TILEFOUNDRY_NPU_TILELANG_REPO", "TILEFOUNDRY_NPU_CANNBOT_DSL_REPO",
    "TILEFOUNDRY_NPU_OPS_TRANSFORMER_REPO", "TILEFOUNDRY_NPU_CANN_SERIES", "TILEFOUNDRY_CANN_SERIES",
    "TILEFOUNDRY_NPU_OCCUPANCY_POLICY", "TILEFOUNDRY_NPU_SHARED_MAX_EXT_PROCS",
    "TILEFOUNDRY_NPU_SHARED_MAX_HBM_MB", "TILEFOUNDRY_NPU_L4_EXTERNAL_PROBE_LIMIT",
})
_ADAPTER_TOOLCHAIN_ENV_KEYS = (
    "TILEFOUNDRY_NPU_CANN_SETUP", "TILEFOUNDRY_NPU_CONDA_SH", "TILEFOUNDRY_NPU_CONDA_ENV",
    "TILEFOUNDRY_NPU_CONDA_ENVS_PATH", "TILEFOUNDRY_NPU_CANNBENCH_ARCH",
)
_ADAPTER_PATH_ENV_KEYS = frozenset({"TILEFOUNDRY_NPU_CANN_SETUP", "TILEFOUNDRY_NPU_CONDA_SH"})


@dataclass(frozen=True)
class CampaignConfig:
    """Configuration controlling stopping policy, escalation, and gate thresholds."""

    operator_name: str
    max_rounds: int = 12
    max_wall_clock_min: float = 180.0
    k_rule_workers: int = 2
    max_syntax_repairs: int = 1
    min_surrogate_gain: float = 0.25
    stagnation_escalate_rounds: int = 2
    stagnation_freeze_rounds: int = 4
    disable_early_goal_stop: bool = False
    run_final_l4: bool = False
    l4_regression_ratio: float = 0.95
    device_id: str = "1"
    retry_device_id: str = ""
    device_ids: tuple[str, ...] = ()
    l4_blocked_env_max_retries: int = 2
    l4_retry_backoff_sec: float = 45.0
    warmup: int = 3
    repeat: int = 5
    enable_tilesize_prescreen: bool = True
    cost_prescreen_top_k: int | None = None
    exploit_effort: str = "medium"
    round_regime_tags: tuple[str, ...] = ()
    allow_unreviewed_distilled_rules: bool = False
    pure_golden: bool = False
    cost_model_mode: str = "off"
    enable_phase0_lattice_sweep: bool = False


@dataclass
class CampaignReport:
    """Final convergence summary emitted by PerfTargetCampaignRunner."""

    operator_name: str
    stop_reason: str
    total_rounds: int
    baseline_result: CandidateSurrogateResult
    best_result: CandidateSurrogateResult
    rounds: tuple[CampaignRoundRecord, ...]
    promoted_rounds: tuple[int, ...] = ()
    rolled_back_rounds: tuple[int, ...] = ()
    escalated_rounds: tuple[int, ...] = ()
    frozen_regimes: tuple[str, ...] = ()
    active_anchors: tuple[AnchorCaseSpec, ...] = ()
    admitted_rule_ids: tuple[str, ...] = ()
    l4_result: L4CannbenchResult | None = None
    l4_status: str = "not_run"
    pending_promotion: PendingPromotion | None = None
    pending_l4: PendingL4 | None = None
    promotion_blocked: bool = False
    pure_golden_mode: bool = False
    blueprint_provenance: dict[str, object] = field(default_factory=dict)
    stagnation_diagnosis: dict[str, object] = field(default_factory=dict)
    scoreboard_anchor_count: int = 0
    scoreboard_anchors: tuple[AnchorCaseSpec, ...] = ()
    real_npu_path: bool = False
    calibrated_dispatch_floor_us: float = 0.0
    geometric_speedup: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Serialize CampaignReport into a JSON-compatible dictionary."""
        anchors = self.scoreboard_anchors or self.active_anchors
        return {
            "operator_name": self.operator_name, "stop_reason": self.stop_reason,
            "total_rounds": self.total_rounds, "baseline_result": self.baseline_result.to_dict(),
            "best_result": self.best_result.to_dict(), "rounds": [r.to_dict() for r in self.rounds],
            "promoted_rounds": list(self.promoted_rounds), "rolled_back_rounds": list(self.rolled_back_rounds),
            "escalated_rounds": list(self.escalated_rounds), "frozen_regimes": list(self.frozen_regimes),
            "active_anchors": [a.to_dict() for a in self.active_anchors],
            "scoreboard_anchor_count": int(self.scoreboard_anchor_count) if self.scoreboard_anchor_count > 0 else len(anchors),
            "scoreboard_anchors": [a.to_dict() for a in anchors], "admitted_rule_ids": list(self.admitted_rule_ids),
            "l4_result": self.l4_result.to_dict() if self.l4_result else None, "l4_status": self.l4_status,
            "pending_promotion": self.pending_promotion.to_dict() if self.pending_promotion else None,
            "pending_l4": self.pending_l4.to_dict() if self.pending_l4 else None,
            "promotion_blocked": self.promotion_blocked, "pure_golden_mode": self.pure_golden_mode,
            "blueprint_provenance": dict(self.blueprint_provenance),
            "stagnation_diagnosis": dict(self.stagnation_diagnosis), "real_npu_path": bool(self.real_npu_path),
            "calibrated_dispatch_floor_us": round(float(self.calibrated_dispatch_floor_us), 6),
            "geometric_speedup": round(float(self.geometric_speedup), 6),
        }


def _derive_candidate_fault_info(chk: L0L1CandidateCheck) -> tuple[str, str]:
    raw_fp = getattr(chk, "fault_plane", "") or (getattr(chk.stage_b, "fault_plane", "") if chk.stage_b else "") or (getattr(chk.stage_a, "fault_plane", "") if chk.stage_a else "")
    raw_fc = getattr(chk, "fault_code", "") or (getattr(chk.stage_b, "fault_code", "") if chk.stage_b else "") or (getattr(chk.stage_a, "fault_code", "") if chk.stage_a else "")
    tl = str(chk.failed_tier or "").lower()
    if raw_fp:
        plane = coerce_fault_plane(raw_fp).value
    elif "noop" in tl:
        plane = FaultPlane.NOOP_IDENTICAL.value
    elif "stage_b" in tl or "l1" in tl:
        plane = FaultPlane.COMPILER_LOWERING.value
    elif "stage_a" in tl or "l0" in tl or "ast" in tl:
        plane = FaultPlane.SYNTAX_AST.value
    else:
        plane = FaultPlane.COMPILER_LOWERING.value
    return plane, str(raw_fc or (chk.failed_tier or "L0_L1_FAILED").upper())


class PerfTargetCampaignRunner:
    """Closed-loop multi-round NPU optimization campaign runner with Protocol coordinators."""

    def __init__(
        self, *, config: CampaignConfig, target_module_path: str | Path | None = None,
        perf_target: PerfTargetSpec | None = None, anchors: Sequence[AnchorCaseSpec] | None = None,
        regime_specs: Sequence[RegimeTargetSpec] | None = None, llm_backend: KernelLLMBackend | None = None,
        tracker: BayesianRuleTracker | None = None, beam: TwoLaneBeamSearch | None = None,
        flywheel_distiller: FlywheelDistiller | None = None,
        l0_checker: Callable[..., StageAResult] | None = None,
        l1_compiler: Callable[[str, RegimeTargetSpec], StageBResult] | None = None,
        l2_batch_evaluator: Callable[..., Mapping[str, CandidateSurrogateResult]] | None = None,
        l3_contract_runner: Callable[..., L3AcceptanceResult] | None = None,
        l4_cannbench_runner: Callable[..., L4CannbenchResult] | None = None,
        msprof_collector: Callable[[str, str, Sequence[AnchorCaseSpec]], Mapping[str, Mapping[str, float]]] | None = None,
        p2_deep_dive_collector: Callable[[str, str, Sequence[AnchorCaseSpec]], Mapping[str, object]] | None = None,
        checkpoint_path: str | Path | None = None, campaign_kb_path: str | Path | None = None,
        source_root: str | Path | None = None, env_file: str | Path | None = None,
        cannbench_repo: str | Path | None = None, enable_track_router: bool = False,
        target: AscendTarget | None = None, clock: Callable[[], float] = time.monotonic,
        sil_advisory: object | None = None, topology_axes: TopologyAxes | Mapping[str, object] | None = None,
        regime_blueprints: Mapping[str, HirBlueprintPacket] | None = None,
        cost_calibration_path: str | Path | None = None, experience_lake_dir: str | Path | None = None,
    ) -> None:
        self._config = config
        self._l4_device_pool = resolve_l4_device_pool(config)
        self._retry_device_id = canonical_device_id(config.retry_device_id, field="retry_device_id") if config.retry_device_id.strip() else ""
        self._source_root = Path(source_root).resolve() if source_root is not None else default_source_root()
        self._target_module_path = Path(target_module_path).resolve() if target_module_path is not None else (self._source_root / get_operator_module_relpath(config.operator_name)).resolve()
        self._env_file = Path(env_file).resolve() if env_file is not None else None
        self._cannbench_repo = Path(cannbench_repo).resolve() if cannbench_repo is not None else None
        self._perf_target = perf_target or get_default_perf_target(config.operator_name)
        self._initial_anchors = tuple(anchors if anchors is not None else get_default_anchors(config.operator_name, cannbench_repo=self._cannbench_repo, env_file=self._env_file))
        self._regime_specs = tuple(regime_specs if regime_specs is not None else get_default_regime_specs(config.operator_name))
        known = {s.regime_tag for s in self._regime_specs}
        for tag in config.round_regime_tags:
            if tag not in known:
                raise ValueError(f"round regime tag {tag!r} is not configured; available: {sorted(known)}")
        self._backend, self._tracker = llm_backend, (tracker or BayesianRuleTracker.from_seed_yaml())
        bew, bsw = (beam.exploit_width if beam else 2), (beam.stepping_stone_width if beam else 2)
        self._beams: dict[str, TwoLaneBeamSearch] = {
            r.regime_tag: (beam if idx == 0 and beam else TwoLaneBeamSearch(exploit_width=bew, stepping_stone_width=bsw))
            for idx, r in enumerate(self._regime_specs)
        }
        self._active_regime_tag: str | None = self._regime_specs[0].regime_tag if self._regime_specs else None
        self._fallback_beam = beam if (beam and not self._regime_specs) else TwoLaneBeamSearch(exploit_width=bew, stepping_stone_width=bsw)
        self._flywheel, self._l0_checker, self._l1_compiler = flywheel_distiller, l0_checker, l1_compiler
        self._l2_batch_evaluator, self._l3_contract_runner, self._l4_cannbench_runner = l2_batch_evaluator, l3_contract_runner, l4_cannbench_runner
        self._msprof_collector = msprof_collector or ((lambda op, cp, anc: collect_bottleneck_msprof_via_agent_gates(op, cp, anc, source_root=self._source_root, env_file=self._env_file, device_id=self._config.device_id, device_pool=self._l4_device_pool)) if l2_batch_evaluator is None else None)
        self._p2_deep_dive_collector = p2_deep_dive_collector or ((lambda op, cp, anc: collect_p2_deep_dive_via_agent_gates(op, cp, anc, source_root=self._source_root, env_file=self._env_file, device_id=self._config.device_id, device_pool=self._l4_device_pool)) if (l2_batch_evaluator is None and self._env_file is not None) else None)
        self._msprof_cache: dict[tuple[str, str], dict[str, float]] = {}
        self._msprof_by_case: dict[tuple[str, int], dict[str, float]] = {}
        self._p2_cache: dict[tuple[str, str], dict[str, object]] = {}
        self._checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path else None
        self._campaign_kb_path = Path(campaign_kb_path).resolve() if campaign_kb_path else None
        self._enable_track_router, self._target = bool(enable_track_router), (target or AscendTarget("ascend.950pr"))
        self._regime_solver = RegimePiecewiseSolver(self._target)
        self._cost_calibration_path = Path(cost_calibration_path).resolve() if cost_calibration_path else ((self._checkpoint_path or self._campaign_kb_path).parent / "cost_model_calibration.jsonl" if (self._checkpoint_path or self._campaign_kb_path) else None)
        self._experience_lake_dir = resolve_default_experience_lake_dir(worktree=None, checkpoint_path=self._checkpoint_path or self._campaign_kb_path, explicit_dir=experience_lake_dir)
        self._experience_lake = ExperienceLake(self._experience_lake_dir) if self._experience_lake_dir else None
        self._cost_advisor: CostModelAdvisor = create_cost_advisor(self._config.cost_model_mode, target=self._target, operator_name=self._config.operator_name, calibration_path=self._cost_calibration_path, experience_lake=self._experience_lake)
        self._run_id = f"{self._config.operator_name.replace('/', '_')}_{int(time.time())}_{os.getpid()}"
        self._case_dims_cache: dict[int, dict[str, object]] = {}
        if self._experience_lake_dir and self._experience_lake_dir.exists():
            try:
                self._cost_advisor.sync_experience_lake(self._experience_lake_dir, persist_calibration=False, exclude_run_id=self._run_id)
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug("ExperienceLake startup replay skipped: %s", exc)
        self._freedom_masks: dict[str, ConstraintRelaxationMask] = {}
        self._live_regime_plans: dict[str, RegimePlanResult] = {}
        self._regime_diag_stats: dict[str, dict[str, object]] = {}
        self._clock, self._sil_advisory = clock, sil_advisory
        self._semantic_topology_axes: TopologyAxes | None = None
        self._topology_axes: TopologyAxes | None = topology_axes if isinstance(topology_axes, TopologyAxes) else (TopologyAxes.from_dict(topology_axes) if isinstance(topology_axes, Mapping) else None)
        self._boundary_ledger = DeterministicSearchBoundaryLedger()
        self._scoreboard_anchors: tuple[AnchorCaseSpec, ...] = tuple(self._initial_anchors)
        self._active_anchors: list[AnchorCaseSpec] = list(self._scoreboard_anchors)
        self._guardrail_cases: list[GuardrailCaseSpec] = []
        self._last_guardrail_metrics_by_cid: dict[str, dict[int, AnchorEvalMetrics]] = {}
        self._calibrated_dispatch_floor_us: float = 0.0
        self._null_launch_us: float | None = None
        self._flywheel_coordinator = FlywheelCoordinator(flywheel=self._flywheel)
        self._last_round_promoted: bool = True
        self._last_round_promoted_by_regime: dict[str, bool] = {}
        self._last_l2_timing_meta: dict[str, object] = {}
        self._active_incumbent_source: str = ""
        self._pending_promotion: PendingPromotion | None = None
        self._active_promotion: PendingPromotion | None = None
        self._pending_l4: PendingL4 | None = None
        self._last_local_rel_gain_by_regime: dict[str, float] = {}
        self._incumbent_anchor_us_by_case: dict[int, float] = {}
        self._stagnation_by_regime: dict[str, int] = {}
        self._compile_wall_streak_by_regime: dict[str, int] = {}
        self._l1_summary_cache: dict[tuple[str, str], BishengResourceSummary | None] = {}
        self._active_kernel_cache: dict[tuple[str, str], str | None] = {}
        self._operator_proto_cache: Mapping[str, object] | None = None
        self._live_stag_by_regime: dict[str, int] = {}
        self._stagnation_discount: float = 0.45
        if self._flywheel and self._flywheel._probe_runner is None and l2_batch_evaluator is None and (self._env_file or os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()):
            self._flywheel._probe_runner = lambda sp: run_micro_probe_via_agent_gates(sp, source_root=self._source_root, env_file=self._env_file, device_id=self._config.device_id, device_pool=self._l4_device_pool)

        self._stage_gate_coordinator = StageGateCoordinator(
            operator_name=config.operator_name, target_module_path=self._target_module_path,
            regime_specs=self._regime_specs, initial_anchors=self._initial_anchors,
            scoreboard_anchors=self._scoreboard_anchors, guardrail_cases=self._guardrail_cases,
            backend=self._backend, tracker=self._tracker, cost_advisor=self._cost_advisor,
            boundary_ledger=self._boundary_ledger, target=self._target, source_root=self._source_root,
            env_file=self._env_file, cannbench_repo=self._cannbench_repo, checkpoint_path=self._checkpoint_path,
            experience_lake_dir=self._experience_lake_dir, run_id=self._run_id, device_id=self._config.device_id,
            l4_device_pool=self._l4_device_pool, warmup=self._config.warmup, repeat=self._config.repeat,
            max_syntax_repairs=self._config.max_syntax_repairs, cost_prescreen_top_k=self._config.cost_prescreen_top_k,
            pure_golden=self._config.pure_golden, enable_track_router=self._enable_track_router,
            sil_advisory=self._sil_advisory, topology_axes=self._topology_axes, semantic_topology_axes=self._semantic_topology_axes,
            l0_checker=self._l0_checker, l1_compiler=self._l1_compiler, l2_batch_evaluator=self._l2_batch_evaluator,
            config=self._config, active_incumbent_source=self._active_incumbent_source, freedom_masks=self._freedom_masks,
        )
        self._round_context_builder = RoundContextBuilder(
            config=self._config, regime_specs=self._regime_specs, target=self._target, tracker=self._tracker,
            cost_advisor=self._cost_advisor, regime_solver=self._regime_solver, source_root=self._source_root,
            perf_target=self._perf_target, env_file=self._env_file, cannbench_repo=self._cannbench_repo,
            enable_track_router=self._enable_track_router, run_id=self._run_id, experience_lake=self._experience_lake,
            experience_lake_dir=self._experience_lake_dir, sil_advisory=self._sil_advisory, topology_axes=self._topology_axes,
            regime_blueprints=regime_blueprints, freedom_masks=self._freedom_masks, boundary_ledger=self._boundary_ledger,
            run_l0_fn=self._stage_gate_coordinator._run_l0, l1_summary_fn=self._stage_gate_coordinator._l1_summary_for_first_round,
            collect_msprof_fn=self._collect_msprof_for_regimes, collect_p2_fn=self._collect_p2_deep_dive_for_regime,
            resolve_op_semantics_fn=self._stage_gate_coordinator._resolve_operator_semantics_for_gates,
        )
        self._round_context_builder._regime_blueprints = self._round_context_builder._validated_regime_blueprints(regime_blueprints)
        self._regime_blueprints = self._round_context_builder._regime_blueprints
        self._blueprint_provenance = self._round_context_builder._blueprint_provenance
        self._semantic_topology_axes = self._round_context_builder._semantic_topology_axes
        self._stage_gate_coordinator._semantic_topology_axes = self._semantic_topology_axes

        self._telemetry_writer = TelemetryLedgerWriter(
            operator_name=config.operator_name, run_id=self._run_id, device_id=self._config.device_id,
            allow_unreviewed_distilled_rules=self._config.allow_unreviewed_distilled_rules,
            target_module_path=self._target_module_path, checkpoint_path=self._checkpoint_path,
            campaign_kb_path=self._campaign_kb_path, experience_lake_dir=self._experience_lake_dir,
            experience_lake=self._experience_lake, cost_advisor=self._cost_advisor, target=self._target,
            perf_target=self._perf_target, regime_specs=self._regime_specs, regime_blueprints=self._regime_blueprints,
            tracker=self._tracker, flywheel_coordinator=self._flywheel_coordinator, flywheel=self._flywheel,
            boundary_ledger=self._boundary_ledger, beams=self._beams, msprof_cache=self._msprof_cache,
            case_dims_cache=self._case_dims_cache, regime_diag_stats=self._regime_diag_stats,
            scoreboard_anchors=self._scoreboard_anchors, guardrail_cases=self._guardrail_cases,
            calibrated_dispatch_floor_us=self._calibrated_dispatch_floor_us, cannbench_repo=self._cannbench_repo,
            env_file=self._env_file, enable_track_router=self._enable_track_router, topology_axes=self._topology_axes,
            sil_advisory=self._sil_advisory, real_npu_path_fn=self._is_real_npu_path,
            l0_runner=self._stage_gate_coordinator._run_l0, l2_batch_runner=self._stage_gate_coordinator._run_l2_batch,
            derive_topology_axes_fn=self._stage_gate_coordinator._derive_topology_axes,
            has_l2_executor_fn=lambda: bool(self._l2_batch_evaluator is not None or self._is_real_npu_path()),
            operator_proto=self._operator_proto_cache,
        )
        self._l3_l4_recovery = L3L4RecoveryCoordinator(
            operator_name=config.operator_name, target_module_path=self._target_module_path,
            source_root=self._source_root, device_id=self._config.device_id, retry_device_id=self._retry_device_id,
            device_ids=self._config.device_ids, l4_device_pool=self._l4_device_pool,
            l4_blocked_env_max_retries=self._config.l4_blocked_env_max_retries,
            l4_retry_backoff_sec=self._config.l4_retry_backoff_sec, env_file=self._env_file,
            cannbench_repo=self._cannbench_repo, checkpoint_path=self._checkpoint_path,
            experience_lake_dir=self._experience_lake_dir, run_id=self._run_id,
            l3_contract_runner=self._l3_contract_runner, l4_cannbench_runner=self._l4_cannbench_runner,
            l2_batch_runner=lambda cands, ancs: self._stage_gate_coordinator._run_l2_batch(cands, ancs),
            resolve_regime_config_fn=lambda mod_src, spec, dims=None: resolve_regime_config(mod_src, spec, dims=dims),
            scoreboard_anchors=self._scoreboard_anchors, regime_specs=self._regime_specs,
            regime_blueprints=self._regime_blueprints, topology_axes=self._topology_axes,
            guardrail_cases=self._guardrail_cases,
            last_guardrail_metrics_by_cid=self._stage_gate_coordinator._last_guardrail_metrics_by_cid,
            case_dims_cache=self._case_dims_cache,
        )
        self._promotion_coordinator = PromotionCoordinator(
            target_module_path=self._target_module_path,
            write_and_verify_fn=lambda p, s: write_source_and_verify(p, s),
        )
        self._calibration_service = CalibrationService(
            operator_name=config.operator_name, target_module_path=self._target_module_path,
            run_l2_batch_fn=lambda cands, ancs: self._stage_gate_coordinator._run_l2_batch(cands, ancs),
            msprof_collector=self._msprof_collector, p2_deep_dive_collector=self._p2_deep_dive_collector,
            run_l0_fn=self._stage_gate_coordinator._run_l0,
            derive_topology_axes_fn=self._stage_gate_coordinator._derive_topology_axes,
            predict_candidate_us_fn=lambda feat: self._cost_advisor.predict_candidate_us(feat),
            underlying_cost_model=self._cost_advisor.underlying_cost_model, cost_advisor_target=self._target,
            select_active_regime_fn=self._select_active_regime,
            evaluate_tail_guard_fn=self._stage_gate_coordinator._evaluate_dual_layer_tail_guard,
            run_l3_contract_fn=self._l3_l4_recovery.run_l3_contract,
            write_and_verify_fn=lambda p, s: write_source_and_verify(p, s),
            refresh_guardrail_refs_fn=self._l3_l4_recovery.refresh_guardrail_incumbent_refs,
            on_warmstart_promoted_fn=self._on_warmstart_promoted,
            regime_blueprints=self._regime_blueprints, live_regime_plans=self._live_regime_plans,
            sil_advisory=self._sil_advisory, enable_phase0_lattice_sweep=self._config.enable_phase0_lattice_sweep,
            regime_specs=self._regime_specs, msprof_by_case=self._msprof_by_case, p2_cache=self._p2_cache,
        )

    def _build_state_snapshot(
        self, *, round_idx: int = 0, incumbent_source: str = "", active_regime_tag: str | None = None,
        frozen_regimes: set[str] = frozenset(), last_round_promoted: bool = True,
    ) -> CampaignStateSnapshot:
        return CampaignStateSnapshot(
            run_id=self._run_id, round_index=int(round_idx), incumbent_source=incumbent_source,
            incumbent_sha256=source_sha256(incumbent_source) if incumbent_source else "",
            dispatch_floor_us=float(self._calibrated_dispatch_floor_us),
            active_regime_tag=active_regime_tag,
            guardrail_case_ids=tuple(int(g.case_num) for g in self._l3_l4_recovery.guardrail_cases),
            stagnation_by_regime=tuple(sorted(self._stagnation_by_regime.items())),
            compile_wall_streak_by_regime=tuple(sorted(self._compile_wall_streak_by_regime.items())),
            last_round_promoted=bool(last_round_promoted),
            last_round_promoted_by_regime=tuple(sorted(self._last_round_promoted_by_regime.items())),
            last_local_rel_gain_by_regime=tuple(sorted(self._last_local_rel_gain_by_regime.items())),
            incumbent_anchor_us_by_case=tuple(sorted(self._incumbent_anchor_us_by_case.items())),
            frozen_regimes=tuple(sorted(frozen_regimes)),
        )

    def beam_for_regime(self, regime_tag: str | None = None) -> TwoLaneBeamSearch:
        """Return the active TwoLaneBeamSearch instance for *regime_tag*."""
        tag = regime_tag or self._active_regime_tag
        if tag and tag in self._beams:
            return self._beams[tag]
        return next(iter(self._beams.values())) if self._beams else self._fallback_beam

    def _is_real_npu_path(self) -> bool:
        return bool(
            getattr(self._l2_batch_evaluator, "is_real_npu", False)
            or (self._l2_batch_evaluator is None and (self._env_file is not None or os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()))
        )

    def _on_warmstart_promoted(self, src: str, res: CandidateSurrogateResult) -> None:
        self._active_incumbent_source = src
        self._incumbent_anchor_us_by_case = {int(a.case_num): float(a.cand_us) for a in res.anchors if a.cand_us > 0.0}
        self._stage_gate_coordinator._active_incumbent_source = src
        self._stage_gate_coordinator._incumbent_anchor_us_by_case = dict(self._incumbent_anchor_us_by_case)

    def _collect_msprof_for_regimes(
        self, source: str, result: CandidateSurrogateResult, active_anchors: Sequence[AnchorCaseSpec], target_regimes: Sequence[RegimeTargetSpec],
    ) -> dict[tuple[str, int], dict[str, float]]:
        collected = self._calibration_service.collect_msprof_by_case(source, result, active_anchors, target_regimes)
        if collected:
            sha = source_sha256(source)
            for (_csha, cnum), metrics in collected.items():
                self._msprof_by_case[(sha, int(cnum))] = dict(metrics)
                for anc in active_anchors:
                    if int(anc.case_num) == int(cnum):
                        self._msprof_cache[(sha, anc.regime_tag)] = dict(metrics)
        return collected

    def _collect_p2_deep_dive_for_regime(
        self, source: str, result: CandidateSurrogateResult, active_anchors: Sequence[AnchorCaseSpec], regime_spec: RegimeTargetSpec,
    ) -> dict[str, object] | None:
        res = self._calibration_service.collect_p2_deep_dive_for_regime(source, result, active_anchors, regime_spec)
        if res is not None:
            self._p2_cache[(source_sha256(source), regime_spec.regime_tag)] = dict(res)
        return res

    def _select_active_regime(
        self, current_regime: str | None, frozen_regimes: set[str], best_result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec], *, allow_satisfied_fallback: bool = False,
        exclude_regime: str | None = None, stagnation_by_regime: Mapping[str, int] | None = None,
        last_round_promoted: bool | None = None,
    ) -> RegimeTargetSpec | None:
        return select_active_regime(
            self._regime_specs, self._perf_target, current_regime, frozen_regimes, best_result, active_anchors,
            allow_satisfied_fallback=allow_satisfied_fallback, exclude_regime=exclude_regime,
            stagnation_by_regime=stagnation_by_regime if stagnation_by_regime is not None else self._stagnation_by_regime,
            compile_wall_streak_by_regime=self._compile_wall_streak_by_regime,
            last_round_promoted_by_regime=self._last_round_promoted_by_regime,
            last_round_promoted=self._last_round_promoted if last_round_promoted is None else last_round_promoted,
            dispatch_floor_us=self._calibrated_dispatch_floor_us,
        )

    def _allow_early_stop(self, rounds_completed: int) -> bool:
        return (not self._config.disable_early_goal_stop) and rounds_completed > 0 and rounds_completed >= self._perf_target.min_optimization_rounds

    def _round_regime_override(self, round_idx: int) -> RegimeTargetSpec | None:
        tags = self._config.round_regime_tags
        if not tags or round_idx < 1 or round_idx > len(tags):
            return None
        target_tag = tags[round_idx - 1]
        for spec in self._regime_specs:
            if spec.regime_tag == target_tag:
                return spec
        raise ValueError(f"Unknown forced round regime tag {target_tag!r} for round {round_idx}")

    @staticmethod
    def _touched_symbols_for_regime_snippet(regime_spec: RegimeTargetSpec, dsl_text: str) -> set[str]:
        if regime_spec.runtime_backend in ("ascendc_trackb", "hybrid_trackb"):
            if "<<<<<<< SEARCH" in dsl_text or ("=======" in dsl_text and ">>>>>>> REPLACE" in dsl_text):
                touched = set(regime_spec.trackb_symbols)
                if res := _trackb_python_residual(dsl_text):
                    touched.update(s.name for s in _collect_top_level_symbols(ast.parse(res)))
                return touched
            if not _parses_as_python(dsl_text) and _looks_like_bare_trackb_cpp(dsl_text):
                return set(regime_spec.trackb_symbols)
        return {s.name for s in _collect_top_level_symbols(ast.parse(dsl_text))}

    def _prune_stale_sibling_stepping_stones(self, promoted_regime: RegimeTargetSpec, promoted_snippet: str) -> None:
        own = set(promoted_regime.editable_symbols)
        touched_own = (self._touched_symbols_for_regime_snippet(promoted_regime, promoted_snippet) & own) if own else set()
        if not touched_own:
            return
        for other in self._regime_specs:
            if other.regime_tag == promoted_regime.regime_tag or not (shared := touched_own & set(other.editable_symbols)):
                continue
            beam = self._beams.get(other.regime_tag)
            if beam and beam.stepping_stone_beam:
                beam.stepping_stone_beam = [s for s in beam.stepping_stone_beam if not (self._touched_symbols_for_regime_snippet(other, s.dsl_source) & shared)]

    def _run_l3_with_card_retry(
        self, *, cand: RankedL2Candidate, round_idx: int, regime_tag: str, attempts: list[dict[str, str]],
    ) -> L3AcceptanceResult:
        tried: set[str] = set()
        stub = PendingPromotion(
            candidate_source=cand.candidate_source, candidate_sha256=cand.candidate_sha256,
            target_path=str(self._target_module_path), round_idx=round_idx, regime_tag=regime_tag,
            candidate_result=cand.candidate_result, candidate_id=cand.candidate_id,
            builder_name=cand.builder_name, parent_lane=cand.parent_lane, creative_effort=cand.creative_effort,
        )

        def _on_started(*args: str) -> None:
            eid = args[0] if args else ""
            rid = args[1] if len(args) == 4 else ""
            rcp = args[2] if len(args) == 4 else ""
            dev = args[3] if len(args) == 4 else (args[1] if len(args) == 2 else "")
            active = replace(stub, execution_id=eid, run_id=rid, receipt_path=rcp, device_id=dev, execution_terminal=False)
            self._active_promotion = self._pending_promotion = active
            self._telemetry_writer.active_promotion = self._telemetry_writer.pending_promotion = active
            persist_runtime_state(self._checkpoint_path, "active_promotion", active.to_dict())

        dev_id = self._l4_device_pool[0] if self._l4_device_pool else self._config.device_id
        pool = (*self._l4_device_pool, self._retry_device_id) if self._retry_device_id else tuple(self._l4_device_pool)
        while True:
            tried.add(dev_id)
            l3_res = self._l3_l4_recovery.run_l3_contract(self._target_module_path, device_id=dev_id, on_execution_started=_on_started)
            for bsec in (2.0, 5.0, 10.0):
                if l3_res.passed or l3_res.failure_kind == FAILURE_KIND_CANDIDATE or "resource busy:" not in (l3_res.error_message or "").lower():
                    break
                logger.warning("Round %d L3 hit transient resource busy; retrying after %.1fs", round_idx, bsec)
                time.sleep(bsec)
                l3_res = self._l3_l4_recovery.run_l3_contract(self._target_module_path, device_id=dev_id, on_execution_started=_on_started)
            if l3_res.execution_terminal:
                self._active_promotion = self._telemetry_writer.active_promotion = None
            attempts.append(L3L4RecoveryCoordinator.promotion_attempt(l3_res, l3_res.device_id or dev_id, candidate_sha256=cand.candidate_sha256))
            if l3_res.passed or l3_res.failure_kind == FAILURE_KIND_CANDIDATE or not l3_res.execution_terminal:
                return l3_res
            alt = next((d for d in pool if d and d not in tried), "")
            if not alt:
                return l3_res
            dev_id = alt

    def _save_checkpoint(self, **kwargs: object) -> None:
        self._telemetry_writer.guardrail_cases = list(self._l3_l4_recovery.guardrail_cases)
        self._telemetry_writer.scoreboard_anchors = self._l3_l4_recovery.scoreboard_anchors or self._scoreboard_anchors
        self._telemetry_writer.calibrated_dispatch_floor_us = float(self._calibrated_dispatch_floor_us)
        self._telemetry_writer.active_regime_tag = self._active_regime_tag
        self._telemetry_writer.last_round_promoted_by_regime = dict(self._last_round_promoted_by_regime)
        self._telemetry_writer.compile_wall_streak_by_regime = dict(self._compile_wall_streak_by_regime)
        self._telemetry_writer.stagnation_by_regime = dict(self._stagnation_by_regime)
        self._telemetry_writer.save_checkpoint(
            pending_promotion=self._pending_promotion, pending_l4=self._pending_l4,
            active_promotion=self._active_promotion, **kwargs,  # type: ignore[arg-type]
        )

    def _restore_from_checkpoint(self):
        restored = self._telemetry_writer.restore_from_checkpoint(pending_l4_from_dict=PendingL4.from_dict)
        self._scoreboard_anchors = self._telemetry_writer.scoreboard_anchors
        self._l3_l4_recovery.scoreboard_anchors = self._scoreboard_anchors
        self._l3_l4_recovery.guardrail_cases = list(self._telemetry_writer.guardrail_cases)
        self._stage_gate_coordinator._scoreboard_anchors = self._scoreboard_anchors
        self._stage_gate_coordinator._guardrail_cases = self._l3_l4_recovery.guardrail_cases
        self._calibrated_dispatch_floor_us = self._telemetry_writer.calibrated_dispatch_floor_us
        self._last_round_promoted_by_regime = dict(self._telemetry_writer.last_round_promoted_by_regime)
        self._last_local_rel_gain_by_regime = dict(self._telemetry_writer.last_local_rel_gain_by_regime)
        self._compile_wall_streak_by_regime = dict(self._telemetry_writer.compile_wall_streak_by_regime)
        self._active_promotion = self._telemetry_writer.active_promotion
        self._pending_promotion = self._telemetry_writer.pending_promotion
        self._pending_l4 = self._telemetry_writer.pending_l4  # type: ignore[assignment]
        self._live_stag_by_regime = dict(restored[12])
        return restored

    async def _execute_single_round(
        self, *, round_idx: int, regime_spec: RegimeTargetSpec, incumbent_source: str,
        best_result: CandidateSurrogateResult, active_anchors: Sequence[AnchorCaseSpec],
        stagnation_rounds: int, admitted_rule_ids: list[str], frozen_regimes: set[str] = frozenset(),
    ) -> tuple[CampaignRoundRecord, str, CandidateSurrogateResult, int, bool, bool, bool, bool]:
        self._active_incumbent_source = incumbent_source
        self._active_regime_tag = regime_spec.regime_tag
        self._incumbent_anchor_us_by_case = {int(a.case_num): float(a.cand_us) for a in best_result.anchors if a.cand_us > 0.0}
        self._stage_gate_coordinator._active_incumbent_source = incumbent_source
        self._stage_gate_coordinator._incumbent_anchor_us_by_case = dict(self._incumbent_anchor_us_by_case)
        self._stage_gate_coordinator._guardrail_cases = self._l3_l4_recovery.guardrail_cases
        stag_map = {**self._stagnation_by_regime, regime_spec.regime_tag: stagnation_rounds}
        merged_msprof = TelemetryLedgerWriter.fold_regime_msprof_cache(
            self._msprof_by_case, self._msprof_cache, active_anchors,
            active_regime_tag=regime_spec.regime_tag, single_regime=len(self._regime_specs) <= 1,
        )
        state = ChampionState(
            incumbent_source=incumbent_source, best_result=best_result, active_anchors=tuple(active_anchors),
            regime_specs=self._regime_specs or (regime_spec,), stagnation_rounds=stagnation_rounds,
            stagnation_by_regime=stag_map, compile_wall_streak_by_regime=self._compile_wall_streak_by_regime,
            last_round_promoted_by_regime=self._last_round_promoted_by_regime, last_round_promoted=self._last_round_promoted,
            last_local_rel_gain_by_regime=self._last_local_rel_gain_by_regime, freedom_masks=self._freedom_masks,
            msprof_by_case=merged_msprof, p2_cache=self._p2_cache,
            calibrated_dispatch_floor_us=self._calibrated_dispatch_floor_us, boundary_ledger=self._boundary_ledger,
            beams=self._beams, current_regime=regime_spec.regime_tag, active_kernel_cache=self._active_kernel_cache,
            live_regime_plans=self._live_regime_plans, operator_proto_cache=self._operator_proto_cache,
            sil_advisory=self._sil_advisory, topology_axes=self._topology_axes,
            semantic_topology_axes=self._semantic_topology_axes, regime_blueprints=self._regime_blueprints,
        )
        bundle = self._round_context_builder.build_round_bundle(state, round_idx=round_idx, regime_spec=regime_spec, working_anchors=active_anchors, creative_effort="high")
        self._freedom_masks.update(self._round_context_builder._freedom_masks)
        self._blueprint_provenance = dict(self._round_context_builder._blueprint_provenance)
        bn_anchor = bottleneck_anchor_for_regime(best_result, regime_spec.regime_tag)
        bn_dims = None
        if bn_anchor is not None:
            rbn = lookup_case_dims_safe(self._config.operator_name, bn_anchor.case_num, regime_tag=regime_spec.regime_tag, regime_specs=self._regime_specs, cache=self._case_dims_cache, cannbench_repo=self._cannbench_repo, env_file=self._env_file)
            if rbn:
                bn_dims = {str(k): int(v) for k, v in rbn.items() if isinstance(v, int) and not isinstance(v, bool)}
        snapshot = self._build_state_snapshot(round_idx=round_idx, incumbent_source=incumbent_source, active_regime_tag=regime_spec.regime_tag, frozen_regimes=frozen_regimes, last_round_promoted=self._last_round_promoted)
        batch = CandidateBatch(
            round_idx=round_idx, regime_spec=regime_spec, worker_specs=bundle.worker_specs,
            parent_module_source=bundle.parent_module_source, parent_snippet=bundle.parent_snippet,
            parent_ledger=bundle.parent_ledger, parent_axes=bundle.parent_axes, active_mask=bundle.active_mask,
            bn_dims=bn_dims, live_plan=bundle.live_plan, blueprint=bundle.blueprint,
            gate_decision=bundle.gate_decision, parent_p2_diag=bundle.parent_p2_diag, func_name=bundle.func_name,
        )
        outcome = await self._stage_gate_coordinator.execute_round_gates(
            batch, active_anchors, backend=self._backend, parent_sha=bundle.parent_sha, parent_workload=bundle.parent_workload, snapshot=snapshot,
        )
        self._last_l2_timing_meta = dict(self._stage_gate_coordinator._last_l2_timing_meta)
        op_entry = resolve_operator_entrypoint_safe(self._config.operator_name)
        regime_beam, checks_by_cid = self.beam_for_regime(regime_spec.regime_tag), {c.candidate_id: c for c in outcome.passed_checks}
        local_best = best_result.for_regime(regime_spec.regime_tag)
        best_sp, best_sc = best_result.avg_speedup, best_result.surrogate_score
        all_pass_l2 = [c for c in outcome.passed_checks if c.candidate_id in outcome.l2_results and outcome.l2_results[c.candidate_id].all_passed]
        if all_pass_l2:
            top_l2 = outcome.l2_results[max(all_pass_l2, key=lambda c: surrogate_rank_key(outcome.l2_results[c.candidate_id])).candidate_id]
            best_sp, best_sc = top_l2.avg_speedup, top_l2.surrogate_score
        promoted, promo_blocked, l3_res_str, l3_fcases, l3_fstage = False, False, L3Resolution.NOT_RUN.value, (), ""
        new_inc, new_best, stag_after, rel_gain, lrel_gain, tg_reason, promo_atts = incumbent_source, best_result, stagnation_rounds, 0.0, 0.0, "", ()
        top_chk, top_res, init_top_cid, l3_sec = None, None, None, 0.0
        l2_results, meta_by_id = dict(outcome.l2_results), dict(outcome.meta_by_id)
        if outcome.l2_infra_fault is not None and outcome.l2_infra_fault.plane in (FaultPlane.INFRA_OCCUPANCY, FaultPlane.INFRA_DEVICE_FAULT):
            l3_res_str = L3Resolution.INFRA_DEFERRED.value
            promo_blocked = outcome.l2_infra_fault.plane == FaultPlane.INFRA_DEVICE_FAULT
        else:
            rcands, uchg = [], {}
            gr_by_cid = dict(self._stage_gate_coordinator._last_guardrail_metrics_by_cid)
            for chk in all_pass_l2:
                cres, (_, _, ceff) = l2_results[chk.candidate_id], meta_by_id.get(chk.candidate_id, ("exploit_rule", None, ""))
                uchg[chk.candidate_id] = self._stage_gate_coordinator._unchanged_sibling_regimes(incumbent_source, chk.patched_module_source, regime_spec)
                rc = RankedL2Candidate(
                    candidate_id=chk.candidate_id, candidate_source=chk.patched_module_source, candidate_result=cres,
                    candidate_sha256=source_sha256(chk.patched_module_source), builder_name=bundle.func_name,
                    parent_lane=bundle.parent_lane, creative_effort=ceff or ("high" if meta_by_id.get(chk.candidate_id, ("",))[0] == "creative_explore" else ""),
                )
                seval = select_top_two_l2_survivors(
                    [rc], best_result=best_result, regime_spec=regime_spec, guardrail_cases=self._l3_l4_recovery.guardrail_cases,
                    guardrail_metrics_by_cid=gr_by_cid, unchanged_siblings_by_cid=uchg, active_anchors=active_anchors,
                    retest_noisy_fn=lambda d, cr, c=chk.candidate_id, s=chk.patched_module_source: self._stage_gate_coordinator._retest_noisy_tail_anchors(candidate_id=c, candidate_source=s, cur_res=cr, regressed_anchors=d, active_anchors=active_anchors),
                    recheck_single_sibling_fn=lambda a, c=chk.candidate_id, s=chk.patched_module_source: self._stage_gate_coordinator._retest_single_or_tail(c, s, [a]),
                )
                if seval.survivors:
                    rcands.append(replace(rc, candidate_result=seval.survivors[0].candidate_result, speedup=seval.survivors[0].speedup))
                elif seval.rejected and not tg_reason:
                    tg_reason = seval.rejected[0][1]
            survivors = RankedL2Survivors(survivors=tuple(replace(c, rank=i) for i, c in enumerate(sorted(rcands, key=lambda it: (-float(it.speedup), -float(it.candidate_result.surrogate_score), it.candidate_id))[:2], start=1)))
            if not survivors.survivors:
                stag_after = stagnation_rounds + 1
            else:
                init_top_cid, tg_reason, lt0, iatts = survivors.survivors[0].candidate_id, "", time.monotonic(), []
                dec = self._promotion_coordinator.coordinate_top_two_l3_promotion(
                    survivors, incumbent_source=incumbent_source, round_idx=round_idx, regime_tag=regime_spec.regime_tag,
                    run_l3_fn=lambda cand: self._run_l3_with_card_retry(cand=cand, round_idx=round_idx, regime_tag=regime_spec.regime_tag, attempts=iatts),
                    target_path=str(self._target_module_path),
                )
                promo_atts = tuple(iatts)
                if dec.resolution == L3Resolution.PROMOTED and dec.promoted_candidate is not None:
                    wc, wres = dec.promoted_candidate, dec.promoted_candidate.candidate_result
                    top_chk, top_res, promoted, l3_res_str, new_inc, new_best, stag_after = checks_by_cid.get(wc.candidate_id), wres, True, L3Resolution.PROMOTED.value, wc.candidate_source, wres, 0
                    best_sp, best_sc = wres.avg_speedup, wres.surrogate_score
                    rel_gain = (wres.avg_speedup - best_result.avg_speedup) / max(best_result.avg_speedup, 1e-6)
                    tl = wres.for_regime(regime_spec.regime_tag)
                    if tl.anchors and local_best.anchors:
                        tlat, blat = tl.mean_latency_us(), local_best.mean_latency_us()
                        lrel_gain = max((tl.avg_speedup - local_best.avg_speedup) / max(local_best.avg_speedup, 1e-6), ((blat - tlat) / max(blat, 1e-6)) if (tlat > 0.0 and blat > 0.0) else 0.0)
                    self._last_local_rel_gain_by_regime[regime_spec.regime_tag] = lrel_gain
                    self._l3_l4_recovery.refresh_guardrail_incumbent_refs(wc.candidate_id)
                    self._pending_promotion = self._active_promotion = self._telemetry_writer.pending_promotion = self._telemetry_writer.active_promotion = None
                    persist_runtime_state(self._checkpoint_path, "active_promotion", None)
                    persist_runtime_state(self._checkpoint_path, "pending_promotion", None)
                    self._collect_msprof_for_regimes(wc.candidate_source, wres, active_anchors, [regime_spec])
                elif dec.resolution == L3Resolution.INFRA_DEFERRED:
                    l3_res_str, promo_blocked, stag_after = L3Resolution.INFRA_DEFERRED.value, True, stagnation_rounds
                    dcid = dec.pending_promotion.candidate_id if dec.pending_promotion else init_top_cid
                    top_chk = checks_by_cid.get(dcid) if dcid else None
                    if dec.l3_result is not None:
                        l3_fcases = tuple(L3L4RecoveryCoordinator.extract_l3_failed_case_nums(dec.l3_result))
                        l3_fstage = str(getattr(dec.l3_result, "failure_stage", "") or "")
                    if dec.pending_promotion is not None:
                        pobj = replace(dec.pending_promotion, attempts=tuple(iatts) or dec.pending_promotion.attempts, retry_used=len(iatts) > 1 or dec.pending_promotion.retry_used)
                        self._pending_promotion = self._telemetry_writer.pending_promotion = pobj
                        promo_atts = pobj.attempts
                        persist_runtime_state(self._checkpoint_path, "pending_promotion", pobj.to_dict())
                        if not pobj.execution_terminal:
                            write_source_and_verify(self._target_module_path, pobj.candidate_source)
                            self._active_promotion = self._telemetry_writer.active_promotion = pobj
                            persist_runtime_state(self._checkpoint_path, "active_promotion", pobj.to_dict())
                        else:
                            self._active_promotion = self._telemetry_writer.active_promotion = None
                            persist_runtime_state(self._checkpoint_path, "active_promotion", None)
                else:
                    l3_res_str, stag_after = L3Resolution.CANDIDATE_REJECTED.value, stagnation_rounds + 1
                    rcid = dec.attempts[-1][0] if dec.attempts else init_top_cid
                    top_chk = checks_by_cid.get(rcid) if rcid else None
                    self._pending_promotion = self._active_promotion = self._telemetry_writer.pending_promotion = self._telemetry_writer.active_promotion = None
                    persist_runtime_state(self._checkpoint_path, "active_promotion", None)
                    persist_runtime_state(self._checkpoint_path, "pending_promotion", None)
                    if dec.l3_result is not None:
                        l3_fstage = str(getattr(dec.l3_result, "failure_stage", "") or "")
                        self._l3_l4_recovery.record_l3_guardrail_cases(dec.l3_result, incumbent_source, regime_spec, topology_axes=bundle.parent_axes)
                        l3_fcases = tuple(dec.new_guardrail_cases) or tuple(L3L4RecoveryCoordinator.extract_l3_failed_case_nums(dec.l3_result))
                l3_sec = max(0.0, time.monotonic() - lt0)

        pt0 = time.monotonic()
        beam_upd, r_adm, meas_cid = [], [], {}
        wcid, wrid, wwk, waxes = (top_chk.candidate_id if top_chk else None), None, "", bundle.parent_axes
        pbn_lat = float(bundle.parent_residual.measured_latency_us) if bundle.parent_residual.measured_latency_us > 0.0 else bundle.parent_lat_us
        top_fam = bundle.ranked_families[0][0].name if bundle.ranked_families else ""
        bp_rf = bundle.blueprint.roofline if bundle.blueprint else None
        sel_stone = next((i for i in (*getattr(regime_beam, "stepping_stone_beam", ()), *getattr(regime_beam, "exploit_beam", ())) if i.candidate_id == bundle.selected_stone_id), None) if bundle.selected_stone_id else None
        for chk in outcome.passed_checks:
            if (res := l2_results.get(chk.candidate_id)) is None:
                continue
            wkind, rule_obj, cand_hyp = meta_by_id.get(chk.candidate_id, ("exploit_rule", None, ""))
            caxes = outcome.cand_axes_by_cid.get(chk.candidate_id, bundle.parent_axes)
            cloc = res.for_regime(regime_spec.regime_tag)
            lres = cloc if cloc.anchors else res
            clat = lres.mean_latency_us()
            if clat > 0.0:
                meas_cid[chk.candidate_id] = float(clat)
            csha = source_sha256(chk.patched_module_source)
            cmp_data = self._msprof_cache.get((csha, regime_spec.regime_tag))
            cobj = float(cloc.surrogate_score if cloc.anchors else res.surrogate_score)
            ledger = chk.stage_a.ledger if (chk.stage_a and chk.stage_a.ledger) else bundle.parent_ledger
            ccs = chk.stage_b.compiler_summary if chk.stage_b else None
            if chk.stage_a and chk.stage_a.ledger:
                ctel = {"dur_us": clat, "measured_latency_us": clat, "roofline_eta": lres.avg_roofline_eta, **{k: v for k, v in (cmp_data or {}).items() if not (k == "dur_us" and float(v) <= 0.0)}}
                cres_obj = compute_static_dynamic_residual(ledger, ctel, compiler_summary=ccs, roofline=bp_rf, topology_axes=caxes)
            else:
                cres_obj = bundle.parent_residual
            ctrack = chk.stage_a.route.track if (self._enable_track_router and chk.stage_a and chk.stage_a.route) else bundle.track
            try:
                _, cslice = extract_regime_snippet(chk.patched_module_source, regime_spec, operator_entrypoint=op_entry, sibling_specs=self._regime_specs)
            except (SyntaxError, ValueError):
                cslice = chk.snippet_source
            beam_upd.append(BeamCandidate(
                candidate_id=chk.candidate_id, round_idx=round_idx, worker_kind=wkind,
                rule_id=rule_obj.rule_id if rule_obj else None, dsl_source=cslice, track=ctrack, ledger=ledger,
                residual=cres_obj, latency_us=clat, accuracy_passed=lres.all_passed,
                structural_signature=extract_structural_signature(cslice, track=ctrack, residual=cres_obj, stage_b=chk.stage_b),
                hypothesis=cand_hyp, parent_id=bundle.active_parent_id,
                features=(chk.stage_a.features if (self._enable_track_router and chk.stage_a) else None),
                objective_score=cobj, case_latencies_us={int(a.case_num): float(a.cand_us) for a in (res.anchors or lres.anchors) if a.cand_us > 0.0},
                policy=dict(getattr(bundle.live_plan, "policy", {}) or {}),
            ))
            spr = TelemetryLedgerWriter.build_stone_parent_result(operator_name=self._config.operator_name, parent_source=bundle.parent_module_source, incumbent_source=incumbent_source, best_result=best_result, cand_result=res, selected_stone=sel_stone)
            self._telemetry_writer.record_surrogate_observations(
                source=chk.patched_module_source, result=res, active_anchors=active_anchors,
                parent_source=bundle.parent_module_source, parent_result=spr, parent_ledger=bundle.parent_ledger,
                parent_bisheng_res=bundle.parent_l1_summary, parent_axes=bundle.parent_axes,
                ledger=chk.stage_a.ledger if chk.stage_a else None, bisheng_summary=ccs,
                stage_b_profile=(chk.stage_b.microarch_profile if chk.stage_b else None), axes=caxes,
                policy=getattr(bundle.live_plan, "policy", None),
                parent_policy=(sel_stone.policy or None if sel_stone and bundle.parent_module_source != incumbent_source else None),
                worker_kind=wkind, rule_id=rule_obj.rule_id if rule_obj else None, backend_track=ctrack, regime_tag=regime_spec.regime_tag,
            )
            if rule_obj is not None and l3_res_str != L3Resolution.INFRA_DEFERRED.value:
                cl3 = l3_res_str if chk.candidate_id == wcid else L3Resolution.NOT_RUN.value
                crb = bool(cl3 == L3Resolution.CANDIDATE_REJECTED.value)
                cf = chk.stage_a.features if chk.stage_a else None
                cok, sgain = TelemetryLedgerWriter.evaluate_causal_rule_outcome(
                    rule=rule_obj, res_all_passed=res.all_passed, rolled_back=crb, ledger=ledger, features=cf,
                    cand_residual=cres_obj, cand_lat_us=clat, cand_obj_score=cobj, parent_ledger=bundle.parent_ledger,
                    parent_residual=bundle.parent_residual, parent_lat_us=pbn_lat, parent_obj_score=bundle.parent_obj_score,
                    has_cand_msprof=cmp_data is not None,
                    cand_global_score=(res.surrogate_score if bundle.selected_stone_id is None else None),
                    parent_global_score=(best_result.surrogate_score if bundle.selected_stone_id is None else None),
                    cand_compiler_summary=ccs, roofline=bp_rf,
                )
                self._tracker.record_outcome(rule_obj.rule_id, postcondition_realized=cok and res.all_passed, speed_gain=sgain, topology_axes=caxes)
                self._telemetry_writer.record_kb_event(rule_id=rule_obj.rule_id, postcondition_realized=cok and res.all_passed, speed_gain=sgain, topology_axes=caxes, round_idx=round_idx)
                if cok and res.all_passed:
                    self._tracker.reset_scoped_family_streak(regime_tag=regime_spec.regime_tag, rule_id=rule_obj.rule_id, parent_sha=bundle.parent_sha)
                    r_adm.append(rule_obj.rule_id)
                    if rule_obj.rule_id not in admitted_rule_ids:
                        admitted_rule_ids.append(rule_obj.rule_id)
                else:
                    freason = "l3_rolled_back" if crb else ("l2_accuracy_failed" if not res.all_passed else "l2_postcondition_unrealized_no_speedup")
                    if (not res.all_passed) or crb or pbn_lat <= 0.0 or clat >= pbn_lat * 0.99:
                        self._tracker.record_scoped_failure(regime_tag=regime_spec.regime_tag, rule_id=rule_obj.rule_id, parent_sha=bundle.parent_sha, current_round=round_idx, reason=freason)
                    r3 = bool(pbn_lat > 0.0 and clat > pbn_lat * 1.03)
                    if self._experience_lake_dir is not None and (not (cok and res.all_passed) or r3):
                        pctx = build_predicate_context(ledger, features=cf, residual=cres_obj, compiler_summary=ccs, roofline=bp_rf)
                        try:
                            append_rule_misapplication(self._experience_lake_dir, {
                                "run_id": self._run_id, "operator": self._config.operator_name, "regime_tag": regime_spec.regime_tag,
                                "round_idx": int(round_idx), "rule_id": rule_obj.rule_id, "symptom_family": rule_obj.symptom_family,
                                "parent_sha": bundle.parent_sha, "cand_sha": csha, "parent_latency_us": float(pbn_lat),
                                "cand_latency_us": float(clat), "latency_ratio": round(float(clat) / float(pbn_lat), 6) if (pbn_lat > 0.0 and clat > 0.0) else 0.0,
                                "postcondition_realized": bool(cok and res.all_passed), "l2_all_passed": bool(res.all_passed), "l3_resolution": cl3,
                                "reason": "latency_regression_gt_3pct" if r3 else freason,
                                "transfer_inflation_ratio": float(pctx.get("transfer_inflation_ratio", 1.0)),
                                "effective_compute_eta": float(pctx.get("effective_compute_eta", 0.0)),
                                "ub_occupancy_ratio": float(pctx.get("ub_occupancy_ratio", 0.0)),
                                "parent_observation": bundle.parent_residual.observation.to_dict() if bundle.parent_residual.observation else None,
                                "cand_observation": cres_obj.observation.to_dict() if cres_obj.observation else None,
                            })
                        except (OSError, ValueError) as exc:
                            logger.debug("Appending rule misapplication skipped: %s", exc)
            if chk.candidate_id == wcid:
                wrid, wwk, waxes = (rule_obj.rule_id if rule_obj else None), wkind, caxes
        if beam_upd:
            regime_beam.update(beam_upd)
        if promoted and top_chk is not None:
            self._prune_stale_sibling_stepping_stones(regime_spec, top_chk.snippet_source)
            wl, bl = new_best.for_regime(regime_spec.regime_tag), best_result.for_regime(regime_spec.regime_tag)
            fbl, wlat = (bl.mean_latency_us() if bl.anchors else best_result.mean_latency_us()), (wl.mean_latency_us() if wl.anchors else new_best.mean_latency_us())
            fpts = [to_conservation_point(a, a) for a in (wl.anchors if wl.anchors else new_best.anchors)]
            fsig = (math.sqrt(sum(p.sigma_paired_us() ** 2 for p in fpts)) / max(1, len(fpts))) if fpts else 0.0
            egain = max(rel_gain, lrel_gain)
            if wwk in ("rule_exploit", "exploit_rule") and wrid:
                self._flywheel_coordinator.seed_baseline(regime_spec.regime_tag, new_inc, wlat, fsig)
            elif self._flywheel_coordinator.should_trigger(regime_spec.regime_tag, wlat, fsig):
                for dr in self._telemetry_writer.run_flywheel_distillation(
                    baseline_source=incumbent_source, winner_source=new_inc, baseline_latency_us=fbl, winner_latency_us=wlat,
                    func_name=bundle.func_name, regime_spec=regime_spec, active_anchors=active_anchors,
                    topology_axes=waxes, rel_gain=egain, local_rel_gain=lrel_gain, worker_kind=wwk,
                    round_idx=round_idx, rule_id=wrid, symptom_family=top_fam, sigma_paired_us=fsig,
                ):
                    if dr not in r_adm:
                        r_adm.append(dr)
                    if dr not in admitted_rule_ids:
                        admitted_rule_ids.append(dr)
            elif wwk == "creative_explore" and egain >= 0.05 and self._experience_lake_dir is not None:
                try:
                    append_winning_edge(self._experience_lake_dir, {
                        "run_id": self._run_id, "operator": self._config.operator_name, "regime_tag": regime_spec.regime_tag,
                        "round_idx": int(round_idx), "worker_kind": wwk, "rule_id": wrid, "symptom_family": top_fam,
                        "parent_sha": source_sha256(incumbent_source), "winner_sha": source_sha256(new_inc),
                        "baseline_latency_us": float(fbl), "winner_latency_us": float(wlat), "relative_gain": float(egain),
                        "loo_chunks": [], "staged_rule_ids": [], "topology_axes": waxes.to_dict(),
                    })
                except (OSError, ValueError) as exc:
                    logger.debug("Appending explore winning edge skipped: %s", exc)
        tb_ev = self._telemetry_writer.collect_trackb_round_evidence(
            round_idx=round_idx, regime_spec=regime_spec, blueprint=bundle.blueprint, live_plan=bundle.live_plan,
            active_kernel=outcome.round_active_kernel, active_mask=bundle.active_mask or ConstraintRelaxationMask.bottom(),
            checks=outcome.passed_checks, l2_results=l2_results, l2_run_ids=outcome.l2_run_ids,
            parent_sha256=bundle.parent_sha, parent_l1_summary=bundle.parent_l1_summary, parent_l1_summary_source=bundle.parent_l1_summary_source,
        )
        rrho = compute_spearman_rho(outcome.pred_by_cid, meas_cid)
        self._boundary_ledger.record_spearman_rho(f"{regime_spec.regime_tag}:{bundle.parent_axes.key()}", rrho)
        de = self._telemetry_writer.ensure_regime_diag_entry(regime_spec.regime_tag)
        if not bundle.ranked_rules:
            de["rule_starvation_rounds"] = int(de.get("rule_starvation_rounds", 0)) + 1
        de["cost_prescreen_filtered_count"] = int(de.get("cost_prescreen_filtered_count", 0)) + len(outcome.prescreen_pruned_list)
        if l3_res_str == L3Resolution.CANDIDATE_REJECTED.value:
            de["l3_rollback_count"] = int(de.get("l3_rollback_count", 0)) + 1
        elif not promoted and l2_results:
            de["l2_no_gain_rounds"] = int(de.get("l2_no_gain_rounds", 0)) + 1
        l0r = de.setdefault("l0_reject_reasons", {})
        for chk in outcome.l0_l1_checks:
            if not chk.passed:
                tl = str(chk.failed_tier or "").lower()
                if "stage_b" in tl or "l1" in tl:
                    de["l1_compiler_penalty_rejects"] = int(de.get("l1_compiler_penalty_rejects", 0)) + 1
                elif isinstance(l0r, dict):
                    _, fc = _derive_candidate_fault_info(chk)
                    l0r[fc] = int(l0r.get(fc, 0)) + 1
        if bundle.ranked_families:
            de["top_symptom_families"] = [f.name for f, _ in bundle.ranked_families[:3]]
        pruned_rules = tuple(self._tracker.prune_underperforming(topology_axes=bundle.parent_axes, round_idx=round_idx))
        self._tracker.decay_and_revive(round_idx=round_idx, min_cooldown_rounds=10, decay_factor=0.99)
        timings = {
            "llm_gen_sec": round(max(0.0, float(outcome.llm_gen_sec)), 6), "l0_l1_sec": round(max(0.0, float(outcome.l0_l1_sec)), 6),
            "syntax_repair_sec": round(max(0.0, float(outcome.syntax_repair_sec)), 6), "l2_prewarm_sec": round(max(0.0, float(outcome.l2_prewarm_sec)), 6),
            "l2_npu_sec": round(max(0.0, float(outcome.l2_npu_sec)), 6), "l3_sec": round(max(0.0, float(l3_sec)), 6),
            "post_l3_sec": round(max(0.0, time.monotonic() - pt0), 6),
        }
        ca_preds = (1 if bundle.parent_cost_pred is not None else 0) + len(outcome.pred_by_cid)
        self._telemetry_writer.emit_round_funnel_events(
            round_idx=round_idx, regime_spec=regime_spec, l0_l1_checks=outcome.l0_l1_checks, failed_entries=outcome.failed_entries,
            meta_by_id=meta_by_id, l2_results=l2_results, l0_l1_faults_by_cid=outcome.l0_l1_faults_by_cid,
            cand_axes_by_cid=outcome.cand_axes_by_cid, parent_axes=bundle.parent_axes, top_chk=top_chk,
            l2_infra_fault=outcome.l2_infra_fault, initial_top_cid=init_top_cid, promoted=promoted, l3_resolution=l3_res_str,
            l3_failure_stage=l3_fstage, stagnation_after=stag_after, admitted_rule_ids=r_adm, local_best=local_best,
            parent_snippet=bundle.parent_snippet, ub_bytes_by_cid=outcome.ub_bytes_by_cid, knob_sig_by_cid=outcome.knob_sig_by_cid,
            pred_by_cid=outcome.pred_by_cid, measured_by_cid=meas_cid, stage_timings_sec=timings,
            prewarm_stats=outcome.round_prewarm_stats, cost_advisor_predictions=ca_preds,
        )
        if self._experience_lake_dir is not None and (outcome.l0_l1_checks or outcome.failed_entries):
            self._telemetry_writer.record_kb_event(rule_id=wrid or "__round__", postcondition_realized=promoted, speed_gain=lrel_gain if promoted else 0.0, topology_axes=bundle.parent_axes, round_idx=round_idx)
        rec = CampaignRoundRecord(
            round_idx=round_idx, regime_tag=regime_spec.regime_tag, builder_name=bundle.func_name, parent_lane=bundle.parent_lane,
            creative_effort="high", l0_l1_passed_ids=tuple(c.candidate_id for c in outcome.passed_checks),
            l0_l1_failed=tuple(outcome.failed_entries), best_round_speedup=best_sp, best_so_far_speedup=new_best.avg_speedup,
            promoted=promoted, l3_resolution=l3_res_str, stagnation_rounds_after=stag_after, admitted_rule_ids=tuple(r_adm),
            best_round_surrogate_score=best_sc, best_so_far_surrogate_score=new_best.surrogate_score, pruned_rule_ids=pruned_rules,
            promotion_attempts=promo_atts, trackb_evidence=tb_ev, cost_model_mode=str(bundle.gate_decision.mode),
            cost_model_engaged=bool(bundle.gate_decision.engaged), cost_model_reason=str(bundle.gate_decision.reason),
            local_rel_gain=float(lrel_gain), tail_guard_reject_reason=tg_reason, paradigm_shift_active=bool(bundle.paradigm_shift_active),
            paradigm_shift_rule_ids=tuple(str(r.rule_id) for r in bundle.ranked_rules if getattr(r, "paradigm_shift", False)),
            round_spearman_rho=rrho, prescreen_pruned_ids=tuple(outcome.prescreen_pruned_list), l3_failed_cases=l3_fcases,
            l3_failure_stage=l3_fstage, infeasible_boundaries_added=outcome.infeasible_boundaries_added, stage_timings_sec=timings,
            prewarm_stats=dict(outcome.round_prewarm_stats), cost_advisor_predictions=ca_preds,
        )
        return rec, new_inc, new_best, stag_after, promoted, l3_res_str == L3Resolution.CANDIDATE_REJECTED.value, bundle.escalated, promo_blocked

    async def run(self, *, resume: bool = False) -> CampaignReport:
        """Execute bounded multi-round convergence campaign with checkpointing and final L4."""
        cstart, budget = self._clock(), self._config.max_wall_clock_min * 60.0
        frozen: set[str] = set()
        creg: str | None = None
        last_unprom: str | None = None
        rounds, prom_r, rb_r, esc_r, adm_ids = [], [], [], [], []
        stag_by_reg: dict[str, int] = {}
        stag_rounds, pending_p, pending_l4, l4_res = 0, None, None, None
        if resume and self._checkpoint_path is not None and self._checkpoint_path.exists():
            (inc_src, base_res, best_res, act_ancs, frozen, creg, stag_rounds, rounds, prom_r, rb_r, esc_r, adm_ids, stag_by_reg, last_unprom, pending_p, pending_l4) = self._restore_from_checkpoint()
        else:
            if not self._target_module_path.exists():
                raise FileNotFoundError(f"Target module does not exist: {self._target_module_path}")
            inc_src = self._target_module_path.read_text(encoding="utf-8")
            act_ancs = list(self._scoreboard_anchors)
            csnap = self._calibration_service.build_snapshot(inc_src, act_ancs, self._regime_specs)
            base_res, self._null_launch_us, self._calibrated_dispatch_floor_us = csnap.baseline_result, csnap.null_launch_us, csnap.calibrated_dispatch_floor_us
            self._msprof_by_case.update(csnap.msprof_by_case)
            sbase = source_sha256(inc_src)
            for (_, cnum), m in csnap.msprof_by_case.items():
                for a in act_ancs:
                    if int(a.case_num) == int(cnum):
                        self._msprof_cache[(sbase, a.regime_tag)] = dict(m)
            self._on_warmstart_promoted(inc_src, base_res)
            if not base_res.all_passed:
                raise RuntimeError(f"Baseline evaluation failed L2 surrogate gates: {base_res.error_message}")
            if self._regime_specs:
                matched_any = any(base_res.for_regime(r.regime_tag).anchors for r in self._regime_specs)
                for idx, rs in enumerate(self._regime_specs):
                    try:
                        bsa = self._stage_gate_coordinator._run_l0(inc_src, rs)
                    except (SyntaxError, RuntimeError, ValueError, TypeError, OSError):
                        bsa = None
                    try:
                        bl1 = self._stage_gate_coordinator._l1_summary_for_first_round(inc_src, rs)
                    except (SyntaxError, RuntimeError, ValueError, TypeError, OSError):
                        bl1 = None
                    bl = bsa.ledger if bsa else None
                    ba = self._stage_gate_coordinator._derive_topology_axes(bl, features=bsa.features if bsa else None, route=bsa.route if bsa else None)
                    rr = base_res.for_regime(rs.regime_tag)
                    if not rr.anchors and (len(self._regime_specs) == 1 or (not matched_any and idx == 0)):
                        rr = base_res
                    self._telemetry_writer.record_surrogate_observations(
                        source=inc_src, result=rr, active_anchors=act_ancs, eval_tier_override="baseline",
                        ledger=bl, bisheng_summary=bl1, axes=ba, worker_kind="baseline",
                        backend_track=(bsa.route.track if self._enable_track_router and bsa and bsa.route else "tilelang_dsl"), regime_tag=rs.regime_tag,
                    )
            else:
                self._telemetry_writer.record_surrogate_observations(source=inc_src, result=base_res, active_anchors=act_ancs, eval_tier_override="baseline", ledger=None, axes=None, worker_kind="baseline", backend_track="tilelang_dsl")
            best_res = base_res
            if not self._calibrated_dispatch_floor_us:
                self._calibrated_dispatch_floor_us = calibrate_dispatch_floor_us([to_conservation_point(a, a) for a in base_res.anchors])
            for rs in self._regime_specs:
                rb = base_res.for_regime(rs.regime_tag)
                rpts = [to_conservation_point(a, a) for a in (rb.anchors if rb.anchors else base_res.anchors)]
                rlat = rb.mean_latency_us() if rb.anchors else base_res.mean_latency_us()
                rsig = (math.sqrt(sum(p.sigma_paired_us() ** 2 for p in rpts)) / max(1, len(rpts))) if rpts else 0.0
                self._flywheel_coordinator.seed_baseline(rs.regime_tag, inc_src, rlat, rsig)
            inc_src, best_res = self._calibration_service.run_phase0_lattice_warmstart(inc_src, best_res, act_ancs)
            self._compile_wall_streak_by_regime = {}
            self._save_checkpoint(
                incumbent_source=inc_src, baseline_result=base_res, best_result=best_res, active_anchors=act_ancs,
                frozen_regimes=frozen, current_regime=creg, stagnation_rounds=stag_rounds, rounds=rounds,
                promoted_rounds=prom_r, rolled_back_rounds=rb_r, escalated_rounds=esc_r, admitted_rule_ids=adm_ids,
                stop_reason="in_progress", l4_result=None, stagnation_by_regime=stag_by_reg, last_unpromoted_regime=last_unprom,
            )
        self._active_anchors = list(act_ancs)
        self._on_warmstart_promoted(inc_src, best_res)
        self._pending_promotion, self._pending_l4 = pending_p, pending_l4
        stop_reason, promo_blocked, l4_resolved = "", False, False
        if pending_p is not None:
            retried = pending_p
            inc_src, best_res, stag_by_reg, pending_p, promo_blocked = self._l3_l4_recovery.resolve_pending_promotion(
                retried, incumbent_source=inc_src, best_result=best_res, stagnation_by_regime=stag_by_reg,
                rounds=rounds, promoted_rounds=prom_r, rolled_back_rounds=rb_r, active_anchors=act_ancs,
            )
            self._pending_promotion = pending_p
            stag_rounds = stag_by_reg.get(retried.regime_tag, 0)
            if promo_blocked:
                stop_reason = "promotion_blocked"
            elif pending_p is None and retried.round_idx in prom_r:
                creg, last_unprom = retried.regime_tag, None
        if pending_l4 is not None and not promo_blocked:
            prior_sr = pending_l4.prior_stop_reason
            l4_res, pending_l4 = self._l3_l4_recovery.resolve_pending_l4(pending_l4, incumbent_source=inc_src)
            self._pending_l4, l4_resolved = pending_l4, True
            if not stop_reason:
                stop_reason = prior_sr or "l4_pending"
            self._telemetry_writer.record_l4_observations(source=inc_src, l4_result=l4_res, active_anchors=act_ancs)
        elif pending_l4 is not None:
            l4_resolved = True
        if not stop_reason and self._allow_early_stop(len(rounds)):
            stop_reason = "goal_met" if best_res.meets_perf_target(self._perf_target) else ("roofline_saturated" if best_res.is_roofline_saturated(self._perf_target) else "")
        self._on_warmstart_promoted(inc_src, best_res)
        self._stagnation_by_regime = stag_by_reg
        cwall = self._compile_wall_streak_by_regime
        start_r = (rounds[-1].round_idx + 1) if rounds else 1
        try:
            for ridx in range(start_r, max(0, self._config.max_rounds) + 1):
                if stop_reason:
                    break
                if (self._clock() - cstart) >= budget:
                    stop_reason = "budget_exhausted"
                    break
                act_unf = [s.regime_tag for s in self._regime_specs if s.regime_tag not in frozen]
                for s in self._regime_specs:
                    if stag_by_reg.get(s.regime_tag, 0) >= self._config.stagnation_freeze_rounds or (len(act_unf) > 1 and cwall.get(s.regime_tag, 0) >= 3):
                        frozen.add(s.regime_tag)
                if creg and creg in frozen:
                    creg = None
                forced = self._round_regime_override(ridx)
                act_reg = forced
                if act_reg is None:
                    wy = next((s.regime_tag for s in self._regime_specs if cwall.get(s.regime_tag, 0) >= 2 and s.regime_tag not in frozen), None)
                    act_reg = self._select_active_regime(creg, frozen, best_res, act_ancs, allow_satisfied_fallback=not self._allow_early_stop(len(rounds)), exclude_regime=last_unprom or wy, stagnation_by_regime=stag_by_reg)
                if act_reg is None and forced is None and last_unprom is not None:
                    act_reg = self._select_active_regime(creg, frozen, best_res, act_ancs, allow_satisfied_fallback=not self._allow_early_stop(len(rounds)), stagnation_by_regime=stag_by_reg)
                if act_reg is None:
                    stop_reason = "all_regimes_converged"
                    break
                atag = act_reg.regime_tag
                creg = self._active_regime_tag = atag
                rec, inc_src, best_res, stag_rounds, prom, rb, esc, promo_blocked = await self._execute_single_round(
                    round_idx=ridx, regime_spec=act_reg, incumbent_source=inc_src, best_result=best_res,
                    active_anchors=act_ancs, stagnation_rounds=stag_by_reg.get(atag, 0), admitted_rule_ids=adm_ids, frozen_regimes=frozen,
                )
                rounds.append(rec)
                pending_p = self._pending_promotion
                self._last_round_promoted, self._last_round_promoted_by_regime[atag] = bool(prom), bool(prom)
                all_noop = bool(rec.l0_l1_failed) and all(st == "noop_identical" or "noop_identical" in er for _, st, er in rec.l0_l1_failed)
                infra_def = rec.l3_resolution == L3Resolution.INFRA_DEFERRED.value or (bool(rec.l0_l1_failed) and all(st in ("l2_infra_occupancy", "l2_infra_device_fault") for _, st, _ in rec.l0_l1_failed))
                if prom:
                    stag_rounds, stag_by_reg[atag], cwall[atag], creg, last_unprom = 0, 0, 0, atag, None
                    prom_r.append(ridx)
                elif infra_def:
                    stag_by_reg[atag], creg, last_unprom = stag_by_reg.get(atag, 0), None, atag
                else:
                    stag_by_reg[atag] = stag_rounds
                    cwall[atag] = (cwall.get(atag, 0) + 1) if (len(rec.l0_l1_passed_ids) == 0 or all_noop) else 0
                    creg, last_unprom = None, atag
                other_unf = any(s.regime_tag != atag and s.regime_tag not in frozen for s in self._regime_specs)
                if other_unf and cwall.get(atag, 0) >= 3:
                    frozen.add(atag)
                if rb:
                    rb_r.append(ridx)
                if esc:
                    esc_r.append(ridx)
                if promo_blocked:
                    stop_reason = "promotion_blocked"
                es_ok = self._allow_early_stop(len(rounds))
                if not promo_blocked:
                    if es_ok and best_res.meets_perf_target(self._perf_target):
                        stop_reason = "goal_met"
                    elif es_ok and best_res.is_roofline_saturated(self._perf_target):
                        stop_reason = "roofline_saturated"
                    elif stag_by_reg.get(atag, 0) >= self._config.stagnation_freeze_rounds or (other_unf and cwall.get(atag, 0) >= 3):
                        frozen.add(atag)
                        if self._select_active_regime(None, frozen, best_res, act_ancs, allow_satisfied_fallback=not es_ok, exclude_regime=last_unprom, stagnation_by_regime=stag_by_reg) is None:
                            stop_reason = "all_regimes_converged"
                self._stagnation_by_regime, self._compile_wall_streak_by_regime = stag_by_reg, cwall
                self._save_checkpoint(
                    incumbent_source=inc_src, baseline_result=base_res, best_result=best_res, active_anchors=act_ancs,
                    frozen_regimes=frozen, current_regime=creg, stagnation_rounds=stag_rounds, rounds=rounds,
                    promoted_rounds=prom_r, rolled_back_rounds=rb_r, escalated_rounds=esc_r, admitted_rule_ids=adm_ids,
                    stop_reason=stop_reason or "in_progress", l4_result=None, stagnation_by_regime=stag_by_reg, last_unpromoted_regime=last_unprom,
                )
                if stop_reason:
                    break
        finally:
            await self._stage_gate_coordinator.cancel_pending_speculative()
        stop_reason = stop_reason or "budget_exhausted"
        if self._config.run_final_l4 and not promo_blocked and not l4_resolved:
            l4_res = self._l3_l4_recovery.run_final_l4_tracked(incumbent_source=inc_src, prior_stop_reason=stop_reason)
            pending_l4 = self._pending_l4 = self._l3_l4_recovery.pending_l4
            if pending_l4 is not None and not stop_reason:
                stop_reason = "l4_pending"
            self._telemetry_writer.record_l4_observations(source=inc_src, l4_result=l4_res, active_anchors=act_ancs)
        self._save_checkpoint(
            incumbent_source=inc_src, baseline_result=base_res, best_result=best_res, active_anchors=act_ancs,
            frozen_regimes=frozen, current_regime=creg, stagnation_rounds=stag_rounds, rounds=rounds,
            promoted_rounds=prom_r, rolled_back_rounds=rb_r, escalated_rounds=esc_r, admitted_rule_ids=adm_ids,
            stop_reason=stop_reason, l4_result=l4_res, stagnation_by_regime=stag_by_reg, last_unpromoted_regime=last_unprom,
        )
        stag_diag = self._telemetry_writer.build_stagnation_diagnosis(best_result=best_res, stop_reason=stop_reason)
        l4_status = "not_run"
        if l4_res is not None:
            l4_err = l4_res.error_message or ""
            l4_status = "passed" if l4_res.passed else ("pending_infrastructure" if pending_l4 is not None else ("deferred_shared_host" if not promo_blocked and ("BLOCKED_ENV" in l4_err or "BLOCKED_EXIT_CODE: 75" in l4_err or "occupancy" in l4_err.lower()) else "failed"))
        return CampaignReport(
            operator_name=self._config.operator_name, stop_reason=stop_reason, total_rounds=len(rounds),
            baseline_result=base_res, best_result=best_res, rounds=tuple(rounds), promoted_rounds=tuple(prom_r),
            rolled_back_rounds=tuple(rb_r), escalated_rounds=tuple(esc_r), frozen_regimes=tuple(sorted(frozen)),
            active_anchors=tuple(self._scoreboard_anchors), scoreboard_anchor_count=len(self._scoreboard_anchors),
            scoreboard_anchors=tuple(self._scoreboard_anchors), admitted_rule_ids=tuple(adm_ids),
            l4_result=l4_res, l4_status=l4_status, pending_promotion=pending_p, pending_l4=pending_l4,
            promotion_blocked=promo_blocked or pending_p is not None, pure_golden_mode=bool(self._config.pure_golden),
            blueprint_provenance=dict(self._blueprint_provenance), stagnation_diagnosis=stag_diag,
            real_npu_path=self._is_real_npu_path(), calibrated_dispatch_floor_us=self._calibrated_dispatch_floor_us,
            geometric_speedup=geometric_speedup([to_conservation_point(a, a) for a in best_res.anchors]),
        )


def short_operator_tag(operator_name: str) -> str:
    """Return short filesystem tag for an operator name or task id."""
    return (operator_name.rsplit("/", 1)[-1] if "/" in operator_name else operator_name).replace("_", "-")


def _add_llm_arguments(parser: argparse.ArgumentParser) -> None:
    for name, (_, help_text) in _LLM_CLI_OPTIONS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", help=help_text)


def _llm_backend_overrides(env_values: Mapping[str, str], options: object) -> dict[str, str]:
    is_llm = lambda k: k.startswith(_LLM_ENV_PREFIX) or k in _LEGACY_AGY_ENV_KEYS
    out = {k: v for k, v in env_values.items() if is_llm(k) and k not in _NON_AGENT_GATES_ENV_KEYS}
    out.update({f"{_LLM_ENV_PREFIX}{k}": val for name, (k, _) in _LLM_CLI_OPTIONS.items() if (val := getattr(options, name, None)) is not None})
    return out


def _add_common_campaign_arguments(parser: argparse.ArgumentParser, *, required_op: bool = True, default_rounds: int | None = 12, default_wall: float | None = 180.0) -> None:
    parser.add_argument("--operator", required=required_op, default=None if not required_op else argparse.SUPPRESS, help="Target operator name or task id.")
    parser.add_argument("--device-id", default="", help="Authorized NPU device ID.")
    parser.add_argument("--device-ids", type=parse_device_ids, default=(), help="Comma-separated pool of NPU device IDs.")
    parser.add_argument("--retry-device-id", default="", help="Fallback NPU device ID for L3/L4 retries.")
    parser.add_argument("--source-root", "--source", dest="source_root", default=None, help="TileFoundry workspace/worktree root.")
    parser.add_argument("--env-file", default=None, help="Adapter env.sh file path.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint JSON path.")
    parser.add_argument("--campaign-kb", default=None, help="Campaign KB YAML path.")
    parser.add_argument("--max-rounds", type=int, default=default_rounds, help="Maximum optimization rounds.")
    parser.add_argument("--max-wall-clock-min", type=float, default=default_wall, help="Maximum wall-clock budget in minutes.")
    parser.add_argument("--k-rule-workers", type=int, default=2, help="Parallel rule-exploit worker count.")
    parser.add_argument("--disable-early-goal-stop", action="store_true", help="Disable early stopping when goal is reached.")
    parser.add_argument("--enable-tilesize-prescreen", dest="enable_tilesize_prescreen", action="store_true", default=True)
    parser.add_argument("--no-enable-tilesize-prescreen", "--disable-tilesize-prescreen", dest="enable_tilesize_prescreen", action="store_false")
    parser.add_argument("--cost-prescreen-top-k", type=int, default=None)
    parser.add_argument("--cost-model-mode", choices=("off", "shadow", "advisory"), default="off")
    parser.add_argument("--allow-unreviewed-distilled-rules", action="store_true")
    parser.add_argument("--round-regime", action="append", default=None, metavar="TAG")
    _add_llm_arguments(parser)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct standalone CLI argument parser for PerfTargetCampaignRunner."""
    p = argparse.ArgumentParser(description="Run unattended performance optimization campaign.")
    _add_common_campaign_arguments(p)
    p.add_argument("--target-file", "--target-module", dest="target_file", default=None)
    p.add_argument("--run-final-l4", action="store_true")
    p.add_argument("--pure-golden", action="store_true")
    return p


def _find_cann_setup_script(seeded: Mapping[str, str]) -> str:
    if (direct := seeded.get("TILEFOUNDRY_NPU_CANN_SETUP", "").strip()) and Path(direct).is_file():
        return str(Path(direct).resolve())
    cands: list[Path] = []
    seen: set[Path] = set()
    add_cand = lambda path: cands.append(path.resolve()) if path.is_file() and path.resolve() not in seen and not seen.add(path.resolve()) else None
    roots = [Path(os.environ[k]).expanduser() for k in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH") if os.environ.get(k, "").strip()]
    for b in roots:
        add_cand(b / "set_env.sh"); add_cand(b.parent / "set_env.sh")
    vdirs = [r for b in roots for r in (b.parent, b.parent.parent) if r.is_dir()]
    if (bs := shutil.which("bisheng")):
        for par in Path(bs).resolve().parents[:5]:
            add_cand(par / "set_env.sh")
            vdirs.extend(r for r in (par.parent, par.parent.parent) if r.is_dir())
    for vr in vdirs:
        add_cand(vr / "Ascend" / "9.3.0-weekly" / "cann-9.3.0" / "set_env.sh")
        for pat in ("Ascend/*/cann-*/set_env.sh", "*/cann-*/set_env.sh", "cann-*/set_env.sh"):
            for m in sorted(vr.glob(pat)):
                add_cand(m)
    opp_ok = lambda p: (p.parent / "opp" / "built-in" / "op_impl" / "ai_core" / "tbe" / "kernel" / "config").is_dir()
    return next((str(c) for c in cands if opp_ok(c)), str(cands[0]) if cands else "")


def discover_marathon_environment(
    *, operator_name: str, worktree: Path, state_dir: Path, base_env_file: str | Path | None = None, source_root: Path | None = None,
) -> dict[str, str]:
    """Discover runtime toolchain and workspace paths with no hardcoded machine paths."""
    ehint = base_env_file or os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()
    epath = Path(ehint).expanduser() if ehint else None
    epath = epath if (epath and epath.is_file()) else None
    seeded = parse_simple_env_file(epath) if epath else {}
    is_llm = lambda k: k.startswith(_LLM_ENV_PREFIX) or k in _LEGACY_AGY_ENV_KEYS
    for k, v in os.environ.items():
        if not (epath and is_llm(k)) and k.startswith("TILEFOUNDRY_NPU_") and v.strip() and k not in seeded:
            seeded[k] = v.strip()
    wsp = list(dict.fromkeys([worktree.resolve().parent, *([source_root.resolve().parent] if source_root else [])]))
    if epath is None:
        for par in wsp:
            if (aenv := par / "AgentGates" / "agent_gates" / "adapters" / "tilefoundry-npu" / "env.sh").is_file():
                av = parse_simple_env_file(aenv)
                for k in _ADAPTER_TOOLCHAIN_ENV_KEYS:
                    if k not in seeded and (val := av.get(k, "").strip()) and (k not in _ADAPTER_PATH_ENV_KEYS or Path(val).expanduser().is_file()):
                        seeded[k] = val
                break
    py = seeded.get("TILEFOUNDRY_NPU_PYTHON") or os.environ.get("TILEFOUNDRY_NPU_PYTHON", "").strip() or sys.executable
    vals = dict(seeded)
    vals["TILEFOUNDRY_NPU_PYTHON"] = py
    parts = (Path(py).expanduser().resolve() if Path(py).expanduser().exists() else Path(py).expanduser()).parents
    if len(parts) >= 4 and parts[2].name == "envs":
        vals.setdefault("TILEFOUNDRY_NPU_CONDA_ENV", parts[1].name)
        vals.setdefault("TILEFOUNDRY_NPU_CONDA_ENVS_PATH", str(parts[2]))
        if (csh := parts[3] / "etc" / "profile.d" / "conda.sh").is_file():
            vals.setdefault("TILEFOUNDRY_NPU_CONDA_SH", str(csh))
    if (cset := _find_cann_setup_script(seeded)):
        vals["TILEFOUNDRY_NPU_CANN_SETUP"] = cset
    tl = next((p / "tilelang" for p in wsp if (p / "tilelang").is_dir()), wsp[0] / "tilelang")
    if tl.is_dir():
        vals.setdefault("TILEFOUNDRY_NPU_TILELANG_SOURCE", str(tl)); vals.setdefault("TILEFOUNDRY_NPU_TILELANG_REPO", str(tl))
    tl_src = Path(vals.get("TILEFOUNDRY_NPU_TILELANG_SOURCE") or tl)
    if (tl_src / "build").is_dir():
        vals.setdefault("TILEFOUNDRY_NPU_TILELANG_BUILD", str(tl_src / "build"))
    if not vals.get("TILEFOUNDRY_NPU_CANNBENCH_REPO"):
        for p in wsp:
            if (cb := p / "cann-bench").is_dir() and (cb / "tasks").is_dir():
                vals["TILEFOUNDRY_NPU_CANNBENCH_REPO"] = str(cb); break
    for k, fn in (("TILEFOUNDRY_NPU_CANNBOT_DSL_REPO", "cannbot-dsl"), ("TILEFOUNDRY_NPU_OPS_TRANSFORMER_REPO", "ops-transformer")):
        if not vals.get(k):
            for p in wsp:
                if (c := p / fn).is_dir():
                    vals[k] = str(c); break
    if not vals.get("TILEFOUNDRY_NPU_AGENT_GATES_BIN"):
        ag = next((str(p / "AgentGates" / ".venv" / "bin" / "agent-gates") for p in wsp if (p / "AgentGates" / ".venv" / "bin" / "agent-gates").is_file()), shutil.which("agent-gates") or "")
        if ag:
            vals["TILEFOUNDRY_NPU_AGENT_GATES_BIN"] = ag
    vals.setdefault("TILEFOUNDRY_NPU_CPU_TESTS", ""); vals.setdefault("TILEFOUNDRY_NPU_DEVICE_TESTS", ""); vals.setdefault("TILEFOUNDRY_NPU_BUILD_SUBDIR", "build")
    vals["TILEFOUNDRY_NPU_WORK_ROOT"] = str((state_dir / "cannbench-work").resolve())
    vals.setdefault("TILEFOUNDRY_NPU_CANNBENCH_ARCH", "ascend950")
    vals["TILEFOUNDRY_NPU_CANNBENCH_TARGETS"] = f"{get_operator_task(operator_name)}:{get_operator_module_relpath(operator_name)}"
    return vals


def ensure_marathon_env_file(
    *, operator_name: str, worktree: Path, state_dir: Path, env_file: str | Path | None = None, source_root: Path | None = None,
) -> Path:
    """Reuse an existing env.sh or materialize a discovered zero-config env.sh."""
    target = (Path(env_file).expanduser() if env_file is not None else (state_dir / "env.sh")).resolve()
    allowed = lambda k: k not in _NON_AGENT_GATES_ENV_KEYS and (k.startswith(_LLM_ENV_PREFIX) or k in _LEGACY_AGY_ENV_KEYS or k in _AGENT_GATES_ALLOWED_ENV_KEYS or k.startswith("TILEFOUNDRY_NPU_CANNBENCH_"))
    disc = {k: v for k, v in discover_marathon_environment(operator_name=operator_name, worktree=worktree, state_dir=state_dir, base_env_file=target if target.is_file() else None, source_root=source_root).items() if allowed(k)}
    _validate_llm_api_key_env(disc.get(_LLM_API_KEY_ENV, "").strip())
    if target.is_file():
        ex = parse_simple_env_file(target)
        if not any(not allowed(k) for k in ex) and not any(v and not ex.get(k) for k, v in disc.items()):
            return target
        merged = {**disc, **{k: v for k, v in ex.items() if v and allowed(k)}}
    else:
        merged = disc
    lines = [f"# Isolated marathon environment for {operator_name}", "# Plain KEY=VALUE assignments.", "", *(_format_env_line(k, merged[k]) for k in sorted(merged)), ""]
    atomic_write_text(target, "\n".join(lines))
    return target


def configure_marathon_process_env(*, operator_name: str, env_file: Path, worktree: Path | None = None, source_root: Path | None = None) -> dict[str, str]:
    """Configure coordinator process env vars cleanly for isolated worker subprocesses."""
    _ = (worktree, source_root)
    tdir = Path(tempfile.gettempdir()) / f"tf_marathon_{short_operator_tag(operator_name)}"
    tdir.mkdir(parents=True, exist_ok=True)
    os.environ.update({"TMPDIR": str(tdir), "TEMP": str(tdir), "TMP": str(tdir), "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "USE_CUDA": "0", "TILEFOUNDRY_NPU_ENV_FILE": str(env_file)})
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    parsed = parse_simple_env_file(env_file)
    for k in ("TILEFOUNDRY_NPU_AGENT_GATES_BIN", "TILEFOUNDRY_NPU_CANNBENCH_REPO", "TILEFOUNDRY_NPU_TILELANG_SOURCE", "TILEFOUNDRY_NPU_TILELANG_BUILD"):
        if (val := parsed.get(k, "").strip()) and not os.environ.get(k):
            os.environ[k] = val
    if not os.environ.get("TILEFOUNDRY_NPU_AGENT_GATES_BIN", "").strip() and (ag := shutil.which("agent-gates")):
        os.environ["TILEFOUNDRY_NPU_AGENT_GATES_BIN"] = ag
    return parsed


def _stage0_contract_err(mod_src: str, entrypoint: str, track: str, *, contract: GoldenEntrypointContract | None = None) -> str:
    if track not in ("cce_trackb", "hybrid_trackb", "tilelang_dsl"):
        return f"unknown Stage-0 track {track!r}"
    try:
        tree = ast.parse(mod_src)
    except SyntaxError as exc:
        return f"seed module syntax error: {exc}"
    if not any(isinstance(n, ast.FunctionDef) and n.name == entrypoint for n in tree.body):
        return f"seed module does not define the public entrypoint {entrypoint}"
    if not any((isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "kernel" for t in n.targets) or isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "kernel") and isinstance(n.value, ast.Name) and n.value.id == entrypoint for n in tree.body):
        return f"seed module does not bind `kernel = {entrypoint}`"
    if contract is not None and (serr := validate_entrypoint_signature_compatibility(mod_src, contract)):
        return serr
    if (perr := check_npu_entrypoint_purity(mod_src, entrypoint, track=track, require_kernel_cache=True)):
        return "entrypoint purity rejection: " + "; ".join(perr)
    if track in ("cce_trackb", "hybrid_trackb") and (off := {t: c for t, c in CCE_CPP20_SURFACE.count_forbidden_tokens(mod_src).items() if c}):
        return "CCE C++20 seed references forbidden tokens: " + ", ".join(f"{t}={c}" for t, c in off.items())
    return ""


def _stage0_agg_engine_err(ledgers: Sequence[ObjectivePhysicalLedger | None], semantics: PyTorchOpSemantics | None) -> str:
    active = [lg for lg in ledgers if lg is not None]
    if semantics and semantics.has_gemm_contraction and not any(any(k in ("AIC.PIPE_M", "AIC.CUBE", "CUBE_L0C") and int(v) > 0 for k, v in lg.pipe_cycles.items()) for lg in active):
        return "aggregate L0 hardware-engine rejection: missing_cube_engine_primitives (AIC.PIPE_M == 0)"
    if semantics and not semantics.has_indirect_index and not any(lg.has_indirect_gather or any(k.endswith("PIPE_MTE2") and int(v) > 0 for k, v in lg.pipe_cycles.items()) for lg in active):
        return "aggregate L0 hardware-engine rejection: missing_hardware_dma_staging (PIPE_MTE2 == 0)"
    return ""


def build_stage0_static_validator(
    regime_specs: Sequence[RegimeTargetSpec], *, entrypoint: str, worktree: Path, env_file: Path | None,
    entrypoint_contract: GoldenEntrypointContract | None = None, semantics: PyTorchOpSemantics | None = None,
    op_proto: Mapping[str, object] | None = None,
) -> Callable[[str, str], tuple[bool, str]]:
    """Build the L0/L1 gate pair the bounded Stage-0 seed loop invokes per attempt."""
    assert regime_specs, "Stage-0 seed synthesis requires at least one regime specification"

    def _run_ta(ms: str, rs: RegimeTargetSpec, *, enforce: bool) -> tuple[bool, str, ObjectivePhysicalLedger | None]:
        if (ce := _stage0_contract_err(ms, entrypoint, "tilelang_dsl", contract=entrypoint_contract)):
            return False, ce, None
        dims = dict(rs.representative_anchor_dims or rs.representative_hir_dims)
        try:
            res = resolve_regime_config(ms, rs, dims=dims)
        except (SyntaxError, ImportError, KeyError, NameError, RuntimeError, TypeError, ValueError) as exc:
            return False, f"seed config resolution failed: {type(exc).__name__}: {exc}", None
        sa = run_stage_a_static(
            ms, factory_name=res.builder_name, factory_kwargs=res.builder_kwargs, pass_configs=res.pass_configs or None,
            compile_flags=res.compile_flags or None, dims=dims or None, case_ratios=rs.representative_case_ratios,
            require_hardware_dma=bool(enforce and semantics and not semantics.has_indirect_index),
            require_cube_engine=bool(enforce and semantics and semantics.has_gemm_contraction),
            ordered_tensor_shapes=extract_ordered_tensor_shapes(case_entry=rs.representative_case_entry, dims=dims, proto=op_proto),
        )
        if not sa.passed:
            return False, f"L0 static/waterline rejection: {sa.error_message}", sa.ledger
        sb = run_l1_compile_check(ms, builder_name=res.builder_name, builder_kwargs=res.builder_kwargs, config_selector_name=res.selector_name or None, pass_configs=res.pass_configs or None, compile_flags=res.compile_flags or None, source_root=worktree, env_file=env_file)
        return (True, "", sa.ledger) if sb.passed else (False, f"L1 device compilation rejection: {sb.error_message}", sa.ledger)

    def _run_tb(ms: str, rs: RegimeTargetSpec, *, track: str, enforce: bool) -> tuple[bool, str, ObjectivePhysicalLedger | None]:
        if (ce := _stage0_contract_err(ms, entrypoint, track, contract=entrypoint_contract)):
            return False, ce, None
        tb_syms = rs.trackb_symbols or discover_module_trackb_symbols(ms)
        if (ext := extract_trackb_cpp_source(ms, tb_syms)) is None:
            return False, "seed exposes no Track-B C++20 translation-unit string constant", None
        try:
            tu = _render_trackb_tu_from_module(ms, ext[1], rs, require_renderer=True)
        except FileNotFoundError:
            raise
        except (SyntaxError, AttributeError, ImportError, KeyError, NameError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return False, f"Track-B _render_trackb_tu instantiation failed ({ext[0]}): {exc}", None
        ver = evaluate_ascendc_candidate(tu, freedom_mask=ConstraintRelaxationMask.top(), case_ratios=rs.representative_case_ratios, require_hardware_dma=bool(enforce and semantics and not semantics.has_indirect_index), require_cube_engine=bool(enforce and semantics and semantics.has_gemm_contraction))
        if not ver.passed:
            return False, f"Track-B L0 token purity ({ext[0]}): {ver.error_message}", ver.ledger
        croot = resolve_cann_root_from_env_file(env_file)
        flags = list(resolve_trackb_compile_flags(tu, _resolve_selector_compile_flags(ms, rs)))
        bs = str(croot / "bin" / "bisheng") if (croot and (croot / "bin" / "bisheng").is_file()) else None
        if croot and (croot / "include").is_dir():
            flags.append(f"-I{croot / 'include'}")
        comp = run_stage_b_ascendc_compile(tu, compile_flags=flags, bisheng_bin=bs, cann_root=croot, verify_shared_link=True)
        return (True, "", ver.ledger) if comp.passed else (False, f"Track-B L1 bisheng compilation ({ext[0]}): {comp.error_message}", ver.ledger)

    def _validate(ms: str, track: str) -> tuple[bool, str]:
        if track not in ("cce_trackb", "hybrid_trackb", "tilelang_dsl"):
            return False, f"[{regime_specs[0].regime_tag}] {_stage0_contract_err(ms, entrypoint, track, contract=entrypoint_contract)}"
        fails = []
        for rs in regime_specs:
            if track == "cce_trackb":
                ok, err, _ = _run_tb(ms, rs, track=track, enforce=True)
            elif track == "hybrid_trackb":
                hp = _has_prim_func_builder(ms, rs.builder_candidates)
                ok, err, ltb = _run_tb(ms, rs, track=track, enforce=not hp)
                if ok and hp:
                    ok, err, lta = _run_ta(ms, rs, enforce=False)
                    if ok and (ae := _stage0_agg_engine_err((ltb, lta), semantics)):
                        ok, err = False, ae
            else:
                ok, err, _ = _run_ta(ms, rs, enforce=True)
            if not ok:
                fails.append(f"[{rs.regime_tag}] {err}")
        return (False, "; ".join(fails)) if fails else (True, "")

    return _validate


def build_stage0_oracle_validator(
    *, operator_name: str, staging_dir: Path, worktree: Path, device_id: str, op_logger: logging.Logger,
    blocked_env_max_retries: int = 12, blocked_env_backoff_sec: tuple[float, ...] = (10.0, 15.0, 20.0, 25.0, 30.0),
) -> Callable[[str], OracleAccuracyResult]:
    """Build the NPU accuracy validator that stages seed bytes by digest before L3 verification."""
    def _validate(module_source: str) -> OracleAccuracyResult:
        digest = hashlib.sha256(module_source.encode("utf-8")).hexdigest()
        staged = staging_dir / f"{digest}.py"
        atomic_write_text(staged, module_source)
        res, transient = None, False
        for ridx in range(max(0, int(blocked_env_max_retries)) + 1):
            res = run_l3_full_contract_via_agent_gates(operator_name=operator_name, candidate_module_path=staged, source_root=worktree, device_id=str(device_id))
            if res.passed:
                transient = False; break
            el = (res.error_message or "").lower()
            if res.failure_kind != "candidate" and ("cannot locate 'agent-gates'" in el or "env.sh file is required" in el or "filenotfounderror" in el):
                raise FileNotFoundError(res.error_message)
            transient = res.failure_kind != "candidate" and ("blocked_env" in el or "resource busy" in el)
            if not transient or ridx >= max(0, int(blocked_env_max_retries)):
                break
            wsec = float(blocked_env_backoff_sec[min(ridx, len(blocked_env_backoff_sec) - 1)]) if blocked_env_backoff_sec else 10.0
            op_logger.info("STAGE0_ORACLE_RETRY_BLOCKED_ENV op=%s retry=%d wait=%.1fs", operator_name, ridx + 1, wsec)
            time.sleep(wsec)
        assert res is not None
        fkind = "environment" if (not res.passed and transient) else (res.failure_kind or "candidate")
        return OracleAccuracyResult(passed=res.passed, error_message=res.error_message, candidate_sha256=res.candidate_sha256, l3_run_id=res.run_id, l3_receipt_path=res.receipt_path, l3_state=res.state, elapsed_sec=res.elapsed_sec, failure_kind=fkind)

    return _validate


def prepare_marathon_seed(
    *, operator_name: str, target_module: Path, staging_dir: Path, worktree: Path, env_file: Path | None,
    cannbench_repo: Path | None, llm_backend: KernelLLMBackend, cold_start: bool, device_id: str,
    op_logger: logging.Logger, pure_golden: bool = False,
    static_validator_factory: Callable[..., Callable[[str, str], tuple[bool, str]]] = build_stage0_static_validator,
    oracle_validator_factory: Callable[..., Callable[[str], OracleAccuracyResult]] = build_stage0_oracle_validator,
) -> SeedSynthesisOutcome:
    """Run bounded Stage-0 seed synthesis or explicit incumbent takeover before the campaign."""
    eff_cold = bool(cold_start or pure_golden)
    reg = CannbenchOpsRegistry(tasks_root=cannbench_repo, source_root=worktree, pure_golden=pure_golden)
    spec = reg.resolve(operator_name)
    if not spec.golden_path.is_file():
        raise FileNotFoundError(f"operator {operator_name} has no PyTorch golden reference at {spec.golden_path}")
    gsrc = spec.golden_path.read_text(encoding="utf-8")
    dtxt = spec.desc_path.read_text(encoding="utf-8") if spec.desc_path.is_file() else ""
    gcon = extract_golden_entrypoint_contract(gsrc, spec.entrypoint, proto=spec.proto, desc_text=dtxt)
    takeover = not eff_cold and target_module.is_file()
    init_path = target_module.parent / "__init__.py"
    created_init = not takeover and not init_path.exists()
    if created_init:
        target_module.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(init_path, f'"""Cold-start Ascend 950PR operator package for {spec.op_id}."""\n\nfrom .kernels import {spec.entrypoint}, kernel\n\n__all__ = [{spec.entrypoint!r}, "kernel"]\n')
    try:
        outcome = asyncio.run(prepare_operator_seed_or_takeover(
            operator_name=operator_name, entrypoint=spec.entrypoint, pytorch_golden_source=gsrc, proto=spec.proto,
            cases=spec.cases, legacy_regime_specs=(reg.get_regime_specs(operator_name) if takeover else None),
            llm_backend=llm_backend,
            l0_l1_validator_factory=lambda rs, *, semantics=None: static_validator_factory(rs, entrypoint=spec.entrypoint, worktree=worktree, env_file=env_file, entrypoint_contract=gcon, semantics=semantics, op_proto=spec.proto),
            oracle_accuracy_validator=oracle_validator_factory(operator_name=operator_name, staging_dir=staging_dir, worktree=worktree, device_id=device_id, op_logger=op_logger),
            existing_module_path=None if eff_cold else target_module, force_cold_start=eff_cold,
            max_seed_attempts=8, desc_text=dtxt, golden_contract=gcon,
        ))
    except (OSError, RuntimeError, ValueError):
        if created_init:
            init_path.unlink(missing_ok=True)
        raise
    if outcome.mode == "cold_start":
        digest = hashlib.sha256(outcome.module_source.encode("utf-8")).hexdigest()
        assert outcome.verification_state == "l3_verified" and outcome.candidate_sha256 == digest
        atomic_write_text(target_module, outcome.module_source)
    return outcome


class ExtremeCampaignRunner(PerfTargetCampaignRunner):
    """Runner with stagnation-discounted regime rotation and per-round telemetry."""

    def __init__(self, *args: object, stagnation_discount: float = 0.45, op_logger: logging.Logger | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._stagnation_discount = max(0.0, float(stagnation_discount))
        self._op_logger = op_logger or logger

    def _select_active_regime(
        self, current_regime: str | None, frozen_regimes: set[str], best_result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec], *, allow_satisfied_fallback: bool = False,
        exclude_regime: str | None = None, stagnation_by_regime: Mapping[str, int] | None = None,
        last_round_promoted: bool | None = None,
    ) -> RegimeTargetSpec | None:
        _ = last_round_promoted
        by_tag = {r.regime_tag: r for r in self._regime_specs}
        sm = {str(k): int(v) for k, v in stagnation_by_regime.items()} if stagnation_by_regime is not None else self._live_stag_by_regime
        if current_regime and current_regime != exclude_regime and current_regime not in frozen_regimes and current_regime in by_tag:
            if not is_regime_satisfied(self._perf_target, by_tag[current_regime], best_result, active_anchors):
                return by_tag[current_regime]
        cands = [s for s in self._regime_specs if s.regime_tag not in frozen_regimes and not is_regime_satisfied(self._perf_target, s, best_result, active_anchors)]
        if not cands and allow_satisfied_fallback:
            if current_regime and current_regime != exclude_regime and current_regime not in frozen_regimes and current_regime in by_tag:
                return by_tag[current_regime]
            cands = [s for s in self._regime_specs if s.regime_tag not in frozen_regimes]
        if not cands:
            return None
        if exclude_regime and any(s.regime_tag != exclude_regime for s in cands):
            cands = [s for s in cands if s.regime_tag != exclude_regime]
        cands.sort(key=lambda s: best_result.regime_headroom(s.regime_tag) / (1.0 + self._stagnation_discount * sm.get(s.regime_tag, 0)), reverse=True)
        return cands[0]

    def _save_checkpoint(self, **kwargs: object) -> None:
        raw_stag = kwargs.get("stagnation_by_regime")
        if isinstance(raw_stag, Mapping):
            self._live_stag_by_regime = {str(k): int(v) for k, v in raw_stag.items()}
        super()._save_checkpoint(**kwargs)
        rounds, best_res = kwargs.get("rounds") or (), kwargs.get("best_result")
        if rounds:
            last = rounds[-1]  # type: ignore[index]
            self._op_logger.info("ROUND_TELEMETRY op=%s round=%d regime=%s promoted=%s best_score=%.6f best_speedup=%.6f", self._config.operator_name, last.round_idx, last.regime_tag, last.promoted, best_res.surrogate_score if best_res else 0.0, best_res.avg_speedup if best_res else 0.0)


def build_resilient_l2_evaluator(
    *, operator_name: str, worktree: Path, env_file: Path, device_id: str,
    device_pool: Sequence[str] = (), warmup: int = 3, repeat: int = 5,
    batch_fn: Callable[..., tuple[dict[str, CandidateSurrogateResult], str, str]] | None = None,
    op_logger: logging.Logger | None = None,
) -> Callable[[Sequence[tuple[str, Path]], Sequence[AnchorCaseSpec]], L2BatchResults]:
    """Build batched L2 evaluator propagating AgentGates errors directly."""
    _ = op_logger
    eval_fn = batch_fn or run_surrogate_batch_via_agent_gates

    def _eval(cands: Sequence[tuple[str, Path]], anc_seq: Sequence[AnchorCaseSpec]) -> L2BatchResults:
        pk = {"device_pool": device_pool} if (batch_fn is None and device_pool) else {}
        res, rid, _ = eval_fn(operator_name, list(cands), anc_seq, source_root=worktree, env_file=env_file, device_id=device_id, warmup=warmup, repeat=repeat, require_all_passed=False, timeout_sec=660, **pk)
        pstats = dict(res.prewarm_stats) if isinstance(res, L2BatchResults) else {"total_tasks": 0, "cache_hits": 0, "compiled_tasks": 0, "failed_candidates": 0}
        return L2BatchResults(res, {cid: rid for cid in res}, prewarm_stats=pstats, l2_prewarm_sec=float(res.l2_prewarm_sec) if isinstance(res, L2BatchResults) else 0.0, l2_npu_sec=float(res.l2_npu_sec) if isinstance(res, L2BatchResults) else 0.0)

    setattr(_eval, "is_real_npu", True)
    return _eval


__all__ = [
    "CAMPAIGN_DEVICE_IDS_ENV_KEY", "CalibrationService", "CalibrationSnapshot", "CampaignConfig",
    "CampaignReport", "CampaignRoundRecord", "ExtremeCampaignRunner", "L3L4RecoveryCoordinator",
    "PendingL4", "PendingPromotion", "PerfTargetCampaignRunner", "ResolvedRegimeConfig",
    "RoundContextBuilder", "StageGateCoordinator", "TelemetryLedgerWriter", "apply_ast_patch",
    "build_arg_parser", "build_resilient_l2_evaluator", "build_stage0_oracle_validator",
    "build_stage0_static_validator", "configure_marathon_process_env", "discover_marathon_environment",
    "ensure_marathon_env_file", "extract_regime_snippet", "parse_device_ids", "patch_regime_into_module",
    "prepare_marathon_seed", "rank_stage_a_candidates_with_cost_model", "resolve_campaign_device_plan",
    "resolve_regime_config", "resolve_regime_for_case", "run_l1_compile_check", "short_operator_tag",
    "split_readonly_host_tail",
]
