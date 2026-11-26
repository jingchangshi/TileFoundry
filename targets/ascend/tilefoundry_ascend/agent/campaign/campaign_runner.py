"""Unattended performance-driven campaign runner delegating to focused coordinators."""

from __future__ import annotations

import argparse
import asyncio
import ast
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import TypeVar

from tilefoundry_ascend.agent.campaign import (
    l3_l4_recovery as _l3_l4_recovery_mod,
    stage_gate_coordinator as _stage_gate_coordinator_mod,
    telemetry_ledger_writer as _telemetry_ledger_writer_mod,
)
from tilefoundry_ascend.agent.campaign.anchor_specs import (
    AnchorCaseSpec, CasePhysicsRatios, GuardrailCaseSpec, PerfTargetSpec, RegimeTargetSpec,
    compute_case_eq3_score, default_source_root, extract_trackb_cpp_source, get_default_anchors,
    get_default_perf_target, get_default_regime_specs, get_operator_module_relpath,
    lookup_case_dims, lookup_case_metadata, parse_simple_env_file, resolve_cannbench_repo,
    resolve_cannbench_tasks_root, resolve_regime_for_case,
)
from tilefoundry_ascend.agent.campaign.calibration_service import (
    CalibrationService, CalibrationSnapshot,
    _invoke_with_optional_kwargs as _cal_invoke_with_optional_kwargs,
)
from tilefoundry_ascend.agent.campaign.fault_types import (
    FaultPlane, L3Resolution, TypedFault, coerce_fault_plane,
)
from tilefoundry_ascend.agent.campaign.flywheel_coordinator import FlywheelCoordinator
from tilefoundry_ascend.agent.campaign.funnel_telemetry import (
    DeterministicSearchBoundaryLedger, RoundFunnelEvent, append_round_funnel_event,
    compute_spearman_rho,
)
from tilefoundry_ascend.agent.campaign.l3_l4_recovery import (
    CAMPAIGN_DEVICE_IDS_ENV_KEY, L3L4RecoveryCoordinator, PendingL4,
    atomic_write_text as _atomic_write_text, canonical_device_id as _canonical_device_id,
    is_occupancy_blocked_env as _is_occupancy_blocked_env, parse_device_ids,
    resolve_campaign_device_plan, resolve_l4_device_pool as _resolve_l4_device_pool,
    write_source_and_verify,
)
from tilefoundry_ascend.agent.campaign.latency_algebra import (
    calibrate_dispatch_floor_us, compute_paired_sigma_us, geometric_speedup,
)
from tilefoundry_ascend.agent.campaign.promotion_coordinator import (
    PendingPromotion, PromotionCoordinator, RankedL2Candidate, RankedL2Survivors,
    select_top_two_l2_survivors, to_conservation_point,
)
from tilefoundry_ascend.agent.campaign.regime_slice_patcher import (
    _HIR_DIM_ALIASES, ResolvedRegimeConfig, _active_trackb_kernel,
    _apply_scoped_trackb_patch_to_module, _apply_spliced_trackb_units_to_module,
    _case_ratio_predicate_entries, _collect_top_level_symbols, _compact_trackb_snippet_source,
    _derive_dynamic_protected_symbols, _detect_trackb_dead_code_edit,
    _extract_p2_focus_lines_and_diag, _extract_regime_abi_context, _extract_trackb_cpp_source,
    _filter_callable_kwargs, _has_prim_func_builder, _is_real_tir_primfunc,
    _looks_like_bare_trackb_cpp, _ordered_top_level_trackb_kernels, _parses_as_python,
    _reject_trackb_context_markers, _render_trackb_tu_from_module,
    _resolve_cann_root_from_env_file, _resolve_operator_entrypoint_safe,
    _resolve_selector_compile_flags, _trackb_kernel_names, _trackb_python_residual,
    apply_ast_patch, extract_regime_snippet, patch_regime_into_module,
    resolve_regime_config, split_readonly_host_tail,
)
from tilefoundry_ascend.agent.campaign.round_context_builder import (
    ChampionState, RoundContextBuilder, RoundSynthesisBundle, _format_anchor_telemetry,
)
from tilefoundry_ascend.agent.campaign.stage_gate_coordinator import (
    CandidateBatch, StageGateCoordinator, StageGateOutcome,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    FAILURE_KIND_CANDIDATE, FAILURE_KIND_INFRASTRUCTURE, AnchorEvalMetrics,
    CandidateSurrogateResult, L0L1CandidateCheck, L2BatchResults, L3AcceptanceResult,
    L4CannbenchResult, OccupancyExhaustedError, aggregate_surrogate_result,
    bottleneck_anchor_for_regime, collect_bottleneck_msprof_via_agent_gates,
    collect_p2_deep_dive_via_agent_gates, run_l0_l1_round_filter, run_l1_compile_check,
    run_l3_full_contract_via_agent_gates, run_l4_official_cannbench_via_agent_gates,
    run_micro_probe_via_agent_gates, run_surrogate_batch_via_agent_gates,
)
from tilefoundry_ascend.agent.campaign.telemetry_ledger_writer import (
    TelemetryLedgerWriter, source_sha256,
)
from tilefoundry_ascend.agent.cost_advisor import CostModelAdvisor, create_cost_advisor
from tilefoundry_ascend.agent.flywheel.ablation_distiller import FlywheelDistiller
from tilefoundry_ascend.agent.kb.bayesian_tracker import (
    BayesianRuleTracker, append_rule_misapplication, append_winning_edge,
    save_distilled_rule_file,
)
from tilefoundry_ascend.agent.kb.schema import (
    AscendCausalRule, build_predicate_context, rank_symptom_families,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    GateFunnel, StageAResult, StageBResult, evaluate_ascendc_candidate,
    resolve_trackb_compile_flags, run_stage_a_static, run_stage_b_ascendc_compile,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    GenerationRequest, KernelLLMBackend, create_llm_backend_from_env,
)
from tilefoundry_ascend.agent.orchestrator.llm_constraints import scan_tilelang_pipe_gaps
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import (
    build_prescreen_summary_from_kwargs,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    BeamCandidate, StaticDynamicResidual, TwoLaneBeamSearch, compiler_summary_from_residual,
    compute_static_dynamic_residual, extract_structural_signature,
)
from tilefoundry_ascend.agent.orchestrator.tilesize_prescreen import (
    rank_stage_a_candidates_with_cost_model,
)
from tilefoundry_ascend.ascend.cost_model import BishengResourceSummary
from tilefoundry_ascend.ascend.cost_model.experience_lake import (
    ExperienceLake, resolve_default_experience_lake_dir,
)
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import (
    HirBlueprintPacket, HirRooflineSummary,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger
from tilefoundry_ascend.ascend.schedule.regime_solver import (
    RegimePiecewiseSolver, RegimePlanResult,
)
from tilefoundry_ascend.ascend.sil.pytorch_frontend import PyTorchOpSemantics
from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask, TopologyAxes
from tilefoundry_ascend.ascend.target import AscendTarget

logger = logging.getLogger(__name__)
_T = TypeVar("_T")
_ORIG_HOOKS: dict[str, object] = {
    "resolve_cannbench_repo": resolve_cannbench_repo,
    "scan_tilelang_pipe_gaps": scan_tilelang_pipe_gaps,
    "lookup_case_dims": lookup_case_dims,
    "lookup_case_metadata": lookup_case_metadata,
    "evaluate_ascendc_candidate": evaluate_ascendc_candidate,
    "run_stage_a_static": run_stage_a_static,
    "run_stage_b_ascendc_compile": run_stage_b_ascendc_compile,
    "resolve_regime_config": resolve_regime_config,
    "run_l0_l1_round_filter": run_l0_l1_round_filter,
    "run_surrogate_batch_via_agent_gates": run_surrogate_batch_via_agent_gates,
    "collect_bottleneck_msprof_via_agent_gates": collect_bottleneck_msprof_via_agent_gates,
    "collect_p2_deep_dive_via_agent_gates": collect_p2_deep_dive_via_agent_gates,
    "run_micro_probe_via_agent_gates": run_micro_probe_via_agent_gates,
    "run_l3_full_contract_via_agent_gates": run_l3_full_contract_via_agent_gates,
    "run_l4_official_cannbench_via_agent_gates": run_l4_official_cannbench_via_agent_gates,
    "build_prescreen_summary_from_kwargs": build_prescreen_summary_from_kwargs,
    "rank_symptom_families": rank_symptom_families,
    "rank_stage_a_candidates_with_cost_model": rank_stage_a_candidates_with_cost_model,
    "resolve_regime_for_case": resolve_regime_for_case,
    "run_l1_compile_check": run_l1_compile_check,
}
_PROPAGATED_HOOKS: set[tuple[object, str]] = set()


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


@dataclass(frozen=True)
class CampaignRoundRecord:
    """Telemetry snapshot recorded at the completion of one campaign round."""

    round_idx: int
    regime_tag: str
    builder_name: str
    parent_lane: str
    creative_effort: str
    l0_l1_passed_ids: tuple[str, ...]
    l0_l1_failed: tuple[tuple[str, str, str], ...]
    best_round_speedup: float
    best_so_far_speedup: float
    promoted: bool
    l3_resolution: str = L3Resolution.NOT_RUN.value
    stagnation_rounds_after: int = 0
    admitted_rule_ids: tuple[str, ...] = ()
    best_round_surrogate_score: float = 0.0
    best_so_far_surrogate_score: float = 0.0
    pruned_rule_ids: tuple[str, ...] = ()
    promotion_attempts: tuple[dict[str, str], ...] = ()
    trackb_evidence: tuple[dict[str, object], ...] = ()
    cost_model_mode: str = "off"
    cost_model_engaged: bool = False
    cost_model_reason: str = ""
    local_rel_gain: float = 0.0
    tail_guard_reject_reason: str = ""
    paradigm_shift_active: bool = False
    paradigm_shift_rule_ids: tuple[str, ...] = ()
    round_spearman_rho: float | None = None
    prescreen_pruned_ids: tuple[str, ...] = ()
    l3_failed_cases: tuple[int, ...] = ()
    l3_failure_stage: str = ""
    infeasible_boundaries_added: int = 0
    stage_timings_sec: dict[str, float] = field(default_factory=dict)
    prewarm_stats: dict[str, int] = field(default_factory=dict)
    cost_advisor_predictions: int = 0

    def to_dict(self) -> dict[str, object]:
        """Serialize CampaignRoundRecord into a JSON-compatible dictionary."""
        return {
            "round_idx": self.round_idx, "regime_tag": self.regime_tag,
            "builder_name": self.builder_name, "parent_lane": self.parent_lane,
            "creative_effort": self.creative_effort,
            "l0_l1_passed_ids": list(self.l0_l1_passed_ids),
            "l0_l1_failed": [list(item) for item in self.l0_l1_failed],
            "best_round_speedup": round(self.best_round_speedup, 6),
            "best_so_far_speedup": round(self.best_so_far_speedup, 6),
            "best_round_surrogate_score": round(self.best_round_surrogate_score, 6),
            "best_so_far_surrogate_score": round(self.best_so_far_surrogate_score, 6),
            "promoted": self.promoted, "l3_resolution": self.l3_resolution,
            "stagnation_rounds_after": self.stagnation_rounds_after,
            "admitted_rule_ids": list(self.admitted_rule_ids),
            "pruned_rule_ids": list(self.pruned_rule_ids),
            "promotion_attempts": [dict(item) for item in self.promotion_attempts],
            "trackb_evidence": [dict(item) for item in self.trackb_evidence],
            "cost_model_mode": self.cost_model_mode,
            "cost_model_engaged": self.cost_model_engaged,
            "cost_model_reason": self.cost_model_reason,
            "local_rel_gain": round(self.local_rel_gain, 6),
            "tail_guard_reject_reason": self.tail_guard_reject_reason,
            "paradigm_shift_active": bool(self.paradigm_shift_active),
            "paradigm_shift_rule_ids": list(self.paradigm_shift_rule_ids),
            "round_spearman_rho": (
                round(float(self.round_spearman_rho), 6)
                if self.round_spearman_rho is not None else None
            ),
            "prescreen_pruned_ids": list(self.prescreen_pruned_ids),
            "l3_failed_cases": list(self.l3_failed_cases),
            "l3_failure_stage": self.l3_failure_stage,
            "infeasible_boundaries_added": int(self.infeasible_boundaries_added),
            "stage_timings_sec": {
                str(k): round(float(v), 6) for k, v in self.stage_timings_sec.items()
            },
            "prewarm_stats": {str(k): int(v) for k, v in self.prewarm_stats.items()},
            "cost_advisor_predictions": int(self.cost_advisor_predictions),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CampaignRoundRecord:
        """Reconstruct CampaignRoundRecord from a JSON mapping."""
        failed = tuple(
            (str(it[0]), str(it[1]), str(it[2]))
            for it in payload.get("l0_l1_failed", ())
            if isinstance(it, Sequence) and len(it) >= 3
        )
        raw_rho = payload.get("round_spearman_rho")
        rho = float(raw_rho) if raw_rho is not None and not isinstance(raw_rho, bool) else None
        raw_t = payload.get("stage_timings_sec")
        timings = (
            {str(k): float(v) for k, v in raw_t.items()
             if isinstance(v, (int, float)) and not isinstance(v, bool)}
            if isinstance(raw_t, Mapping) else {}
        )
        raw_pw = payload.get("prewarm_stats")
        prewarm = (
            {str(k): int(v) for k, v in raw_pw.items()
             if isinstance(v, (int, float)) and not isinstance(v, bool)}
            if isinstance(raw_pw, Mapping) else {}
        )
        return cls(
            round_idx=int(payload["round_idx"]), regime_tag=str(payload["regime_tag"]),
            builder_name=str(payload["builder_name"]), parent_lane=str(payload["parent_lane"]),
            creative_effort=str(payload["creative_effort"]),
            l0_l1_passed_ids=tuple(str(x) for x in payload.get("l0_l1_passed_ids", ())),
            l0_l1_failed=failed, best_round_speedup=float(payload["best_round_speedup"]),
            best_so_far_speedup=float(payload["best_so_far_speedup"]),
            promoted=bool(payload["promoted"]),
            l3_resolution=str(payload.get("l3_resolution", L3Resolution.NOT_RUN.value)),
            stagnation_rounds_after=int(payload["stagnation_rounds_after"]),
            admitted_rule_ids=tuple(str(x) for x in payload.get("admitted_rule_ids", ())),
            best_round_surrogate_score=float(payload.get("best_round_surrogate_score", 0.0)),
            best_so_far_surrogate_score=float(payload.get("best_so_far_surrogate_score", 0.0)),
            pruned_rule_ids=tuple(str(x) for x in payload.get("pruned_rule_ids", ())),
            promotion_attempts=tuple(
                {str(k): str(v) for k, v in item.items()}
                for item in payload.get("promotion_attempts", ()) if isinstance(item, Mapping)
            ),
            trackb_evidence=tuple(
                dict(it) for it in payload.get("trackb_evidence", ()) if isinstance(it, Mapping)
            ),
            cost_model_mode=str(payload.get("cost_model_mode", "off") or "off"),
            cost_model_engaged=bool(payload.get("cost_model_engaged", False)),
            cost_model_reason=str(payload.get("cost_model_reason", "") or ""),
            local_rel_gain=float(payload.get("local_rel_gain", 0.0) or 0.0),
            tail_guard_reject_reason=str(payload.get("tail_guard_reject_reason", "") or ""),
            paradigm_shift_active=bool(payload.get("paradigm_shift_active", False)),
            paradigm_shift_rule_ids=tuple(
                str(x) for x in payload.get("paradigm_shift_rule_ids", ())
            ),
            round_spearman_rho=rho,
            prescreen_pruned_ids=tuple(str(x) for x in (payload.get("prescreen_pruned_ids") or ())),
            l3_failed_cases=tuple(int(x) for x in (payload.get("l3_failed_cases") or ())),
            l3_failure_stage=str(payload.get("l3_failure_stage", "") or ""),
            infeasible_boundaries_added=int(payload.get("infeasible_boundaries_added", 0) or 0),
            stage_timings_sec=timings, prewarm_stats=prewarm,
            cost_advisor_predictions=int(payload.get("cost_advisor_predictions", 0) or 0),
        )


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
            "total_rounds": self.total_rounds,
            "baseline_result": self.baseline_result.to_dict(),
            "best_result": self.best_result.to_dict(),
            "rounds": [r.to_dict() for r in self.rounds],
            "promoted_rounds": list(self.promoted_rounds),
            "rolled_back_rounds": list(self.rolled_back_rounds),
            "escalated_rounds": list(self.escalated_rounds),
            "frozen_regimes": list(self.frozen_regimes),
            "active_anchors": [a.to_dict() for a in self.active_anchors],
            "scoreboard_anchor_count": (
                int(self.scoreboard_anchor_count)
                if self.scoreboard_anchor_count > 0 else len(anchors)
            ),
            "scoreboard_anchors": [a.to_dict() for a in anchors],
            "admitted_rule_ids": list(self.admitted_rule_ids),
            "l4_result": self.l4_result.to_dict() if self.l4_result else None,
            "l4_status": self.l4_status,
            "pending_promotion": (
                self.pending_promotion.to_dict() if self.pending_promotion else None
            ),
            "pending_l4": self.pending_l4.to_dict() if self.pending_l4 else None,
            "promotion_blocked": self.promotion_blocked, "pure_golden_mode": self.pure_golden_mode,
            "blueprint_provenance": dict(self.blueprint_provenance),
            "stagnation_diagnosis": dict(self.stagnation_diagnosis),
            "real_npu_path": bool(self.real_npu_path),
            "calibrated_dispatch_floor_us": round(float(self.calibrated_dispatch_floor_us), 6),
            "geometric_speedup": round(float(self.geometric_speedup), 6),
        }


def _surrogate_rank_key(res: CandidateSurrogateResult) -> tuple[float, float, float]:
    return (res.surrogate_score, res.min_case_speedup, res.avg_speedup)


def _derive_candidate_fault_info(chk: L0L1CandidateCheck) -> tuple[str, str]:
    raw_fp = (
        getattr(chk, "fault_plane", "")
        or (getattr(chk.stage_b, "fault_plane", "") if chk.stage_b else "")
        or (getattr(chk.stage_a, "fault_plane", "") if chk.stage_a else "")
    )
    raw_fc = (
        getattr(chk, "fault_code", "")
        or (getattr(chk.stage_b, "fault_code", "") if chk.stage_b else "")
        or (getattr(chk.stage_a, "fault_code", "") if chk.stage_a else "")
    )
    tier_low = str(chk.failed_tier or "").lower()
    if raw_fp:
        plane = coerce_fault_plane(raw_fp).value
    elif "noop" in tier_low:
        plane = FaultPlane.NOOP_IDENTICAL.value
    elif "stage_b" in tier_low or "l1" in tier_low:
        plane = FaultPlane.COMPILER_LOWERING.value
    elif "stage_a" in tier_low or "l0" in tier_low or "ast" in tier_low:
        plane = FaultPlane.SYNTAX_AST.value
    else:
        plane = FaultPlane.COMPILER_LOWERING.value
    code = str(raw_fc or (chk.failed_tier or "L0_L1_FAILED").upper())
    return plane, code


class PerfTargetCampaignRunner:
    """Host-CPU multi-round optimization controller delegating to focused coordinators."""

    def __init__(
        self,
        *,
        config: CampaignConfig,
        target_module_path: str | Path | None = None,
        perf_target: PerfTargetSpec | None = None,
        anchors: Sequence[AnchorCaseSpec] | None = None,
        regime_specs: Sequence[RegimeTargetSpec] | None = None,
        llm_backend: KernelLLMBackend | None = None,
        tracker: BayesianRuleTracker | None = None,
        beam: TwoLaneBeamSearch | None = None,
        flywheel_distiller: FlywheelDistiller | None = None,
        l0_checker: Callable[[str, RegimeTargetSpec], StageAResult] | None = None,
        l1_compiler: Callable[[str, RegimeTargetSpec], StageBResult] | None = None,
        l2_batch_evaluator: Callable[
            [Sequence[tuple[str, Path]], Sequence[AnchorCaseSpec]],
            Mapping[str, CandidateSurrogateResult],
        ]
        | None = None,
        l3_contract_runner: Callable[[Path], L3AcceptanceResult] | None = None,
        l4_cannbench_runner: Callable[[Path], L4CannbenchResult] | None = None,
        msprof_collector: Callable[
            [str, str, Sequence[AnchorCaseSpec]],
            Mapping[str, Mapping[str, float]],
        ]
        | None = None,
        p2_deep_dive_collector: Callable[
            [str, str, Sequence[AnchorCaseSpec]],
            Mapping[str, object],
        ]
        | None = None,
        checkpoint_path: str | Path | None = None,
        campaign_kb_path: str | Path | None = None,
        source_root: str | Path | None = None,
        env_file: str | Path | None = None,
        cannbench_repo: str | Path | None = None,
        enable_track_router: bool = False,
        target: AscendTarget | None = None,
        clock: Callable[[], float] = time.monotonic,
        sil_advisory: object | None = None,
        topology_axes: TopologyAxes | Mapping[str, object] | None = None,
        regime_blueprints: Mapping[str, HirBlueprintPacket] | None = None,
        cost_calibration_path: str | Path | None = None,
        experience_lake_dir: str | Path | None = None,
    ) -> None:
        self._config = config
        self._l4_device_pool = _resolve_l4_device_pool(config)
        retry_id = config.retry_device_id.strip()
        self._retry_device_id = (
            _canonical_device_id(retry_id, field="retry_device_id") if retry_id else ""
        )
        self._source_root = (
            Path(source_root).resolve() if source_root is not None else default_source_root()
        )
        self._target_module_path = (
            Path(target_module_path).resolve()
            if target_module_path is not None
            else (self._source_root / get_operator_module_relpath(config.operator_name)).resolve()
        )
        self._env_file = Path(env_file).resolve() if env_file is not None else None
        self._cannbench_repo = (
            Path(cannbench_repo).resolve() if cannbench_repo is not None else None
        )
        self._perf_target = perf_target or get_default_perf_target(config.operator_name)
        self._initial_anchors = tuple(
            anchors
            if anchors is not None
            else get_default_anchors(
                config.operator_name,
                cannbench_repo=self._cannbench_repo,
                env_file=self._env_file,
            )
        )
        self._regime_specs = tuple(
            regime_specs
            if regime_specs is not None
            else get_default_regime_specs(config.operator_name)
        )
        known_tags = {spec.regime_tag for spec in self._regime_specs}
        for forced_tag in config.round_regime_tags:
            if forced_tag not in known_tags:
                raise ValueError(
                    f"round regime tag {forced_tag!r} is not configured; "
                    f"available regimes: {sorted(known_tags)}"
                )
        self._backend = llm_backend
        self._tracker = tracker or BayesianRuleTracker.from_seed_yaml()
        base_ew = beam.exploit_width if beam is not None else 2
        base_sw = beam.stepping_stone_width if beam is not None else 2
        self._beam_widths = (base_ew, base_sw)
        self._beams: dict[str, TwoLaneBeamSearch] = {
            rspec.regime_tag: (
                beam
                if idx == 0 and beam is not None
                else TwoLaneBeamSearch(exploit_width=base_ew, stepping_stone_width=base_sw)
            )
            for idx, rspec in enumerate(self._regime_specs)
        }
        self._active_regime_tag: str | None = (
            self._regime_specs[0].regime_tag if self._regime_specs else None
        )
        self._fallback_beam = (
            beam
            if (beam is not None and not self._regime_specs)
            else TwoLaneBeamSearch(exploit_width=base_ew, stepping_stone_width=base_sw)
        )
        self._flywheel = flywheel_distiller
        self._l0_checker = l0_checker
        self._l1_compiler = l1_compiler
        self._l2_batch_evaluator = l2_batch_evaluator
        self._l3_contract_runner = l3_contract_runner
        self._l4_cannbench_runner = l4_cannbench_runner
        if msprof_collector is not None:
            self._msprof_collector: (
                Callable[[str, str, Sequence[AnchorCaseSpec]], Mapping[str, Mapping[str, float]]]
                | None
            ) = msprof_collector
        elif l2_batch_evaluator is None:
            self._msprof_collector = (
                lambda op_name, cand_path, anc_seq: L3L4RecoveryCoordinator._invoke(
                    collect_bottleneck_msprof_via_agent_gates,
                    op_name,
                    cand_path,
                    anc_seq,
                    source_root=self._source_root,
                    env_file=self._env_file,
                    device_id=self._config.device_id,
                    device_pool=self._l4_device_pool,
                )
                if hasattr(L3L4RecoveryCoordinator, "_invoke")
                else collect_bottleneck_msprof_via_agent_gates(
                    op_name,
                    cand_path,
                    anc_seq,
                    source_root=self._source_root,
                    env_file=self._env_file,
                    device_id=self._config.device_id,
                    device_pool=self._l4_device_pool,
                )
            )
        else:
            self._msprof_collector = None
        if p2_deep_dive_collector is not None:
            self._p2_deep_dive_collector: (
                Callable[[str, str, Sequence[AnchorCaseSpec]], Mapping[str, object]] | None
            ) = p2_deep_dive_collector
        elif l2_batch_evaluator is None and self._env_file is not None:
            self._p2_deep_dive_collector = (
                lambda op_name, cand_path, anc_seq: collect_p2_deep_dive_via_agent_gates(
                    op_name,
                    cand_path,
                    anc_seq,
                    source_root=self._source_root,
                    env_file=self._env_file,
                    device_id=self._config.device_id,
                    device_pool=self._l4_device_pool,
                )
            )
        else:
            self._p2_deep_dive_collector = None
        self._msprof_cache: dict[tuple[str, str], dict[str, float]] = {}
        self._msprof_by_case: dict[tuple[str, int], dict[str, float]] = {}
        self._p2_cache: dict[tuple[str, str], dict[str, object]] = {}
        self._checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path else None
        self._campaign_kb_path = Path(campaign_kb_path).resolve() if campaign_kb_path else None
        self._enable_track_router = bool(enable_track_router)
        self._target = target or AscendTarget("ascend.950pr")
        self._regime_solver = RegimePiecewiseSolver(self._target)
        if cost_calibration_path is not None:
            self._cost_calibration_path: Path | None = Path(cost_calibration_path).resolve()
        elif self._checkpoint_path is not None:
            self._cost_calibration_path = (
                self._checkpoint_path.parent / "cost_model_calibration.jsonl"
            )
        elif self._campaign_kb_path is not None:
            self._cost_calibration_path = (
                self._campaign_kb_path.parent / "cost_model_calibration.jsonl"
            )
        else:
            self._cost_calibration_path = None
        self._experience_lake_dir = resolve_default_experience_lake_dir(
            worktree=None,
            checkpoint_path=self._checkpoint_path or self._campaign_kb_path,
            explicit_dir=experience_lake_dir,
        )
        self._experience_lake = (
            ExperienceLake(self._experience_lake_dir)
            if self._experience_lake_dir is not None
            else None
        )
        self._cost_advisor: CostModelAdvisor = create_cost_advisor(
            self._config.cost_model_mode,
            target=self._target,
            operator_name=self._config.operator_name,
            calibration_path=self._cost_calibration_path,
            experience_lake=self._experience_lake,
        )
        op_slug = self._config.operator_name.replace("/", "_")
        self._run_id = f"{op_slug}_{int(time.time())}_{os.getpid()}"
        self._case_dims_cache: dict[int, dict[str, object]] = {}
        if self._experience_lake_dir is not None and self._experience_lake_dir.exists():
            try:
                self._invoke_with_optional_kwargs(
                    self._cost_advisor.sync_experience_lake,
                    self._experience_lake_dir,
                    persist_calibration=False,
                    exclude_run_id=self._run_id,
                )
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug("ExperienceLake startup replay skipped: %s", exc)
        self._freedom_masks: dict[str, ConstraintRelaxationMask] = {}
        self._live_regime_plans: dict[str, RegimePlanResult] = {}
        self._regime_diag_stats: dict[str, dict[str, object]] = {}
        self._clock = clock
        self._sil_advisory = sil_advisory
        self._semantic_topology_axes: TopologyAxes | None = None
        if isinstance(topology_axes, TopologyAxes):
            self._topology_axes: TopologyAxes | None = topology_axes
        elif isinstance(topology_axes, Mapping):
            self._topology_axes = TopologyAxes.from_dict(topology_axes)
        else:
            self._topology_axes = None
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
        self._last_tail_guard_reject_reason: str = ""
        self._incumbent_anchor_us_by_case: dict[int, float] = {}
        self._stagnation_by_regime: dict[str, int] = {}
        self._compile_wall_streak_by_regime: dict[str, int] = {}
        self._l1_summary_cache: dict[tuple[str, str], BishengResourceSummary | None] = {}
        self._active_kernel_cache: dict[tuple[str, str], str | None] = {}
        self._operator_semantics_cache: PyTorchOpSemantics | None = None
        self._operator_proto_cache: Mapping[str, object] | None = None
        self._operator_semantics_loaded: bool = False
        self._live_stag_by_regime: dict[str, int] = {}
        self._stagnation_discount: float = 0.45

        if (
            self._flywheel is not None
            and self._flywheel._probe_runner is None
            and l2_batch_evaluator is None
            and (
                self._env_file is not None
                or bool(os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip())
            )
        ):
            self._flywheel._probe_runner = lambda spec: self._invoke_with_optional_kwargs(
                run_micro_probe_via_agent_gates,
                spec,
                source_root=self._source_root,
                env_file=self._env_file,
                device_id=self._config.device_id,
                device_pool=self._l4_device_pool,
            )

        self._stage_gate_coordinator = StageGateCoordinator(
            operator_name=config.operator_name,
            target_module_path=self._target_module_path,
            regime_specs=self._regime_specs,
            initial_anchors=self._initial_anchors,
            scoreboard_anchors=self._scoreboard_anchors,
            guardrail_cases=self._guardrail_cases,
            backend=self._backend,
            tracker=self._tracker,
            cost_advisor=self._cost_advisor,
            boundary_ledger=self._boundary_ledger,
            target=self._target,
            source_root=self._source_root,
            env_file=self._env_file,
            cannbench_repo=self._cannbench_repo,
            checkpoint_path=self._checkpoint_path,
            experience_lake_dir=self._experience_lake_dir,
            run_id=self._run_id,
            device_id=self._config.device_id,
            l4_device_pool=self._l4_device_pool,
            warmup=self._config.warmup,
            repeat=self._config.repeat,
            max_syntax_repairs=self._config.max_syntax_repairs,
            cost_prescreen_top_k=self._config.cost_prescreen_top_k,
            pure_golden=self._config.pure_golden,
            enable_track_router=self._enable_track_router,
            sil_advisory=self._sil_advisory,
            topology_axes=self._topology_axes,
            semantic_topology_axes=self._semantic_topology_axes,
            l0_checker=self._l0_checker,
            l1_compiler=self._l1_compiler,
            l2_batch_evaluator=self._l2_batch_evaluator,
            config=self._config,
            active_incumbent_source=self._active_incumbent_source,
            owner=self,
        )
        self._stage_gate_coordinator._guardrail_cases = self._guardrail_cases
        self._stage_gate_coordinator._freedom_masks = self._freedom_masks
        self._stage_gate_coordinator._incumbent_anchor_us_by_case = (
            self._incumbent_anchor_us_by_case
        )
        self._stage_gate_coordinator._last_guardrail_metrics_by_cid = (
            self._last_guardrail_metrics_by_cid
        )
        self._stage_gate_coordinator._active_kernel_cache = self._active_kernel_cache
        self._stage_gate_coordinator._l1_summary_cache = self._l1_summary_cache
        self._stage_gate_coordinator._case_dims_cache = self._case_dims_cache

        self._regime_blueprints: dict[str, HirBlueprintPacket] = {}
        self._round_context_builder = RoundContextBuilder(
            config=self._config,
            regime_specs=self._regime_specs,
            target=self._target,
            tracker=self._tracker,
            cost_advisor=self._cost_advisor,
            regime_solver=self._regime_solver,
            source_root=self._source_root,
            perf_target=self._perf_target,
            env_file=self._env_file,
            cannbench_repo=self._cannbench_repo,
            enable_track_router=self._enable_track_router,
            run_id=self._run_id,
            experience_lake=self._experience_lake,
            experience_lake_dir=self._experience_lake_dir,
            sil_advisory=self._sil_advisory,
            topology_axes=self._topology_axes,
            regime_blueprints=regime_blueprints,
            freedom_masks=self._freedom_masks,
            boundary_ledger=self._boundary_ledger,
            run_l0_fn=self._run_l0,
            l1_summary_fn=self._l1_summary_for_first_round,
            collect_msprof_fn=self._collect_msprof_for_regimes,
            collect_p2_fn=self._collect_p2_deep_dive_for_regime,
            resolve_op_semantics_fn=self._resolve_operator_semantics_for_gates,
            resolve_active_trackb_kernel_fn=self._resolve_active_trackb_kernel,
            lookup_case_dims_fn=self._lookup_case_dims_safe,
            resolve_trackb_freedom_mask_fn=self._resolve_trackb_freedom_mask,
        )
        cur_repo_fn = globals()["resolve_cannbench_repo"]
        if cur_repo_fn is not _ORIG_HOOKS["resolve_cannbench_repo"]:
            self._round_context_builder._hook_resolve_cannbench_repo = cur_repo_fn
        if regime_blueprints is None:
            self._round_context_builder._regime_blueprints = (
                self._round_context_builder._build_semantic_first_regime_blueprints()
            )
        else:
            self._round_context_builder._regime_blueprints = (
                self._round_context_builder._validated_regime_blueprints(regime_blueprints)
            )
        self._regime_blueprints = self._round_context_builder._regime_blueprints
        self._blueprint_provenance = self._round_context_builder._blueprint_provenance
        self._golden_abi_block_cache = self._round_context_builder._golden_abi_block_cache
        self._round_context_builder._case_dims_cache = self._case_dims_cache
        self._round_context_builder._active_kernel_cache = self._active_kernel_cache

        self._telemetry_writer = TelemetryLedgerWriter(
            operator_name=config.operator_name, run_id=self._run_id,
            device_id=self._config.device_id,
            allow_unreviewed_distilled_rules=self._config.allow_unreviewed_distilled_rules,
            target_module_path=self._target_module_path, checkpoint_path=self._checkpoint_path,
            campaign_kb_path=self._campaign_kb_path, experience_lake_dir=self._experience_lake_dir,
            experience_lake=self._experience_lake, cost_advisor=self._cost_advisor,
            target=self._target, perf_target=self._perf_target, regime_specs=self._regime_specs,
            regime_blueprints=self._regime_blueprints, tracker=self._tracker,
            flywheel_coordinator=self._flywheel_coordinator, flywheel=self._flywheel,
            boundary_ledger=self._boundary_ledger, beams=self._beams,
            msprof_cache=self._msprof_cache, case_dims_cache=self._case_dims_cache,
            regime_diag_stats=self._regime_diag_stats, scoreboard_anchors=self._scoreboard_anchors,
            guardrail_cases=self._guardrail_cases,
            calibrated_dispatch_floor_us=self._calibrated_dispatch_floor_us,
            cannbench_repo=self._cannbench_repo, env_file=self._env_file,
            enable_track_router=self._enable_track_router, topology_axes=self._topology_axes,
            sil_advisory=self._sil_advisory, real_npu_path_fn=self._is_real_npu_path,
            l0_runner=self._run_l0, l2_batch_runner=self._run_l2_batch,
            derive_topology_axes_fn=self._stage_gate_coordinator._derive_topology_axes,
            has_l2_executor_fn=lambda: (
                self._l2_batch_evaluator is not None or self._is_real_npu_path()
            ),
        )
        self._telemetry_writer.guardrail_cases = self._guardrail_cases

        self._l3_l4_recovery = L3L4RecoveryCoordinator(
            operator_name=config.operator_name, target_module_path=self._target_module_path,
            source_root=self._source_root, device_id=self._config.device_id,
            retry_device_id=self._retry_device_id, device_ids=self._config.device_ids,
            l4_device_pool=self._l4_device_pool,
            l4_blocked_env_max_retries=self._config.l4_blocked_env_max_retries,
            l4_retry_backoff_sec=self._config.l4_retry_backoff_sec,
            env_file=self._env_file, cannbench_repo=self._cannbench_repo,
            checkpoint_path=self._checkpoint_path, experience_lake_dir=self._experience_lake_dir,
            run_id=self._run_id, l3_contract_runner=self._l3_contract_runner,
            l4_cannbench_runner=self._l4_cannbench_runner, l2_batch_runner=self._run_l2_batch,
            lookup_case_dims_fn=self._lookup_case_dims_safe,
            resolve_regime_config_fn=lambda mod_src, spec, dims=None: resolve_regime_config(
                mod_src, spec, dims=dims
            ),
            persist_state_fn=self._persist_runtime_state,
            write_and_verify_fn=self._write_source_and_verify,
            scoreboard_anchors=self._scoreboard_anchors, regime_specs=self._regime_specs,
            regime_blueprints=self._regime_blueprints, topology_axes=self._topology_axes,
            guardrail_cases=self._guardrail_cases,
            last_guardrail_metrics_by_cid=self._last_guardrail_metrics_by_cid,
            case_dims_cache=self._case_dims_cache,
        )

        self._promotion_coordinator = PromotionCoordinator(
            target_module_path=self._target_module_path,
            write_and_verify_fn=self._write_source_and_verify,
        )

        self._calibration_service = CalibrationService(
            operator_name=config.operator_name, target_module_path=self._target_module_path,
            run_l2_batch_fn=self._run_l2_batch, msprof_collector=self._msprof_collector,
            p2_deep_dive_collector=self._p2_deep_dive_collector, run_l0_fn=self._run_l0,
            derive_topology_axes_fn=self._stage_gate_coordinator._derive_topology_axes,
            predict_candidate_us_fn=lambda feat: self._cost_advisor.predict_candidate_us(feat),
            underlying_cost_model=self._cost_advisor.underlying_cost_model,
            cost_advisor_target=self._target, select_active_regime_fn=self._select_active_regime,
            evaluate_tail_guard_fn=self._evaluate_dual_layer_tail_guard,
            run_l3_contract_fn=self._run_l3_contract,
            write_and_verify_fn=self._write_source_and_verify,
            refresh_guardrail_refs_fn=self._refresh_guardrail_incumbent_refs,
            on_warmstart_promoted_fn=self._on_warmstart_promoted,
            regime_blueprints=self._regime_blueprints, live_regime_plans=self._live_regime_plans,
            sil_advisory=self._sil_advisory,
            enable_phase0_lattice_sweep=self._config.enable_phase0_lattice_sweep,
            regime_specs=self._regime_specs, msprof_by_case=self._msprof_by_case,
            p2_cache=self._p2_cache,
        )

    def _sync_coordinators(self) -> None:
        g = globals()
        sgc = self._stage_gate_coordinator
        sgc._backend = self._backend
        sgc._l0_checker = self._l0_checker
        sgc._l1_compiler = self._l1_compiler
        sgc._l2_batch_evaluator = self._l2_batch_evaluator
        sgc._regime_specs = tuple(self._regime_specs)
        sgc._scoreboard_anchors = tuple(self._scoreboard_anchors)
        sgc._guardrail_cases = self._guardrail_cases
        sgc._incumbent_anchor_us_by_case = self._incumbent_anchor_us_by_case
        sgc._active_incumbent_source = self._active_incumbent_source
        sgc._topology_axes = self._topology_axes
        sgc._semantic_topology_axes = self._semantic_topology_axes
        sgc._sil_advisory = self._sil_advisory
        sgc._run_id = self._run_id

        rcb = self._round_context_builder
        rcb._run_id = self._run_id
        rcb._regime_specs = tuple(self._regime_specs)
        rcb._perf_target = self._perf_target
        rcb._topology_axes = self._topology_axes
        rcb._semantic_topology_axes = self._semantic_topology_axes
        rcb._regime_blueprints = self._regime_blueprints
        rcb._sil_advisory = self._sil_advisory
        rcb._calibrated_dispatch_floor_us = self._calibrated_dispatch_floor_us
        rcb._stagnation_by_regime = self._stagnation_by_regime
        rcb._compile_wall_streak_by_regime = self._compile_wall_streak_by_regime
        rcb._last_round_promoted = self._last_round_promoted
        rcb._last_round_promoted_by_regime = self._last_round_promoted_by_regime
        rcb._run_l0_fn = self._run_l0
        rcb._l1_summary_fn = self._l1_summary_for_first_round
        rcb._collect_msprof_fn = self._collect_msprof_for_regimes
        rcb._collect_p2_fn = self._collect_p2_deep_dive_for_regime
        rcb._resolve_op_semantics_fn = self._resolve_operator_semantics_for_gates
        rcb._resolve_active_trackb_kernel_fn = self._resolve_active_trackb_kernel
        rcb._lookup_case_dims_fn = self._lookup_case_dims_safe
        rcb._resolve_cannbench_tasks_root_fn = self.__dict__.get("_resolve_cannbench_tasks_root")
        rcb._resolve_trackb_freedom_mask_fn = self._resolve_trackb_freedom_mask

        for hook_name, rcb_attr in (
            ("resolve_cannbench_repo", "_hook_resolve_cannbench_repo"),
            ("lookup_case_dims", "_hook_lookup_case_dims"),
            ("resolve_regime_config", "_hook_resolve_regime_config"),
            ("rank_symptom_families", "_hook_rank_symptom_families"),
            ("evaluate_ascendc_candidate", "_hook_evaluate_ascendc_candidate"),
            ("run_stage_a_static", "_hook_run_stage_a_static"),
            ("build_prescreen_summary_from_kwargs", "_hook_build_prescreen_summary_from_kwargs"),
            ("scan_tilelang_pipe_gaps", "_hook_scan_tilelang_pipe_gaps"),
        ):
            cur_fn = g[hook_name]
            setattr(rcb, rcb_attr, cur_fn if cur_fn is not _ORIG_HOOKS[hook_name] else None)

        for mod, names in (
            (
                _stage_gate_coordinator_mod,
                (
                    "resolve_cannbench_repo",
                    "lookup_case_dims",
                    "resolve_regime_for_case",
                    "resolve_regime_config",
                    "run_l0_l1_round_filter",
                    "run_surrogate_batch_via_agent_gates",
                    "evaluate_ascendc_candidate",
                    "run_stage_a_static",
                    "run_stage_b_ascendc_compile",
                    "rank_stage_a_candidates_with_cost_model",
                    "run_l1_compile_check",
                ),
            ),
            (
                _l3_l4_recovery_mod,
                (
                    "lookup_case_dims",
                    "lookup_case_metadata",
                    "resolve_regime_for_case",
                    "run_l3_full_contract_via_agent_gates",
                    "run_l4_official_cannbench_via_agent_gates",
                ),
            ),
            (
                _telemetry_ledger_writer_mod,
                (
                    "lookup_case_dims",
                    "lookup_case_metadata",
                ),
            ),
        ):
            for name in names:
                cur_fn = g[name]
                orig_fn = _ORIG_HOOKS[name]
                if cur_fn is not orig_fn:
                    setattr(mod, name, cur_fn)
                    _PROPAGATED_HOOKS.add((mod, name))
                elif (mod, name) in _PROPAGATED_HOOKS:
                    setattr(mod, name, orig_fn)
                    _PROPAGATED_HOOKS.discard((mod, name))

        tlw = self._telemetry_writer
        tlw.run_id = self._run_id
        tlw._regime_specs = tuple(self._regime_specs)
        tlw._perf_target = self._perf_target
        tlw.scoreboard_anchors = tuple(self._scoreboard_anchors)
        tlw.guardrail_cases = self._guardrail_cases
        tlw.calibrated_dispatch_floor_us = float(self._calibrated_dispatch_floor_us)
        tlw.active_regime_tag = self._active_regime_tag
        tlw._topology_axes = self._topology_axes
        tlw.active_promotion = self._active_promotion
        tlw.pending_promotion = self._pending_promotion
        tlw.pending_l4 = self._pending_l4
        tlw._regime_blueprints = self._regime_blueprints
        tlw._flywheel = self._flywheel
        tlw._flywheel_coordinator = self._flywheel_coordinator
        tlw.stagnation_by_regime = self._stagnation_by_regime
        tlw.compile_wall_streak_by_regime = self._compile_wall_streak_by_regime
        tlw.last_round_promoted_by_regime = self._last_round_promoted_by_regime
        tlw.last_local_rel_gain_by_regime = self._last_local_rel_gain_by_regime

        rec = self._l3_l4_recovery
        rec._run_id = self._run_id
        rec._l3_contract_runner = self._l3_contract_runner
        rec._l4_cannbench_runner = self._l4_cannbench_runner
        rec.scoreboard_anchors = tuple(self._scoreboard_anchors)
        rec.regime_specs = tuple(self._regime_specs)
        rec.regime_blueprints = self._regime_blueprints
        rec.topology_axes = self._topology_axes
        rec.guardrail_cases = self._guardrail_cases
        rec.pending_l4 = self._pending_l4
        rec._lookup_case_dims_fn = self._lookup_case_dims_safe
        rec.last_guardrail_metrics_by_cid = sgc._last_guardrail_metrics_by_cid
        rec._sleep_fn = time.sleep

        cal = self._calibration_service
        cal._msprof_collector = self._msprof_collector
        cal._p2_deep_dive_collector = self._p2_deep_dive_collector
        cal._regime_specs = tuple(self._regime_specs)
        cal._regime_blueprints = self._regime_blueprints
        cal._enable_phase0_lattice_sweep = bool(self._config.enable_phase0_lattice_sweep)
        cal._underlying_cost_model = self._cost_advisor.underlying_cost_model
        cal._live_regime_plans = self._live_regime_plans
        cal._sil_advisory = self._sil_advisory

    @property
    def _beam(self) -> TwoLaneBeamSearch:
        if self._active_regime_tag and self._active_regime_tag in self._beams:
            return self._beams[self._active_regime_tag]
        if self._beams:
            return next(iter(self._beams.values()))
        return self._fallback_beam

    def beam_for_regime(self, regime_tag: str) -> TwoLaneBeamSearch:
        """Return the dedicated TwoLaneBeamSearch owned by ``regime_tag``."""
        if regime_tag not in self._beams:
            self._beams[regime_tag] = TwoLaneBeamSearch(
                exploit_width=self._beam_widths[0],
                stepping_stone_width=self._beam_widths[1],
            )
        return self._beams[regime_tag]

    @staticmethod
    def _source_sha256(source: str) -> str:
        return source_sha256(source)

    @staticmethod
    def _invoke_with_optional_kwargs(
        runner: Callable[..., _T],
        *args: object,
        **kwargs: object,
    ) -> _T:
        return _cal_invoke_with_optional_kwargs(runner, *args, **kwargs)

    @staticmethod
    def _write_source_and_verify(path: Path, source: str) -> str:
        return write_source_and_verify(path, source)

    def _persist_runtime_state(self, key: str, value: object) -> None:
        self._telemetry_writer.persist_runtime_state(key, value)

    def _is_real_npu_path(self) -> bool:
        if self._l2_batch_evaluator is not None:
            return bool(getattr(self._l2_batch_evaluator, "is_real_npu", False))
        return bool(
            self._env_file is not None
            or os.environ.get("TILEFOUNDRY_NPU_AGENT_GATES_BIN", "").strip()
            or os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()
        )

    def _resolve_cannbench_tasks_root(self) -> Path | None:
        rcb = self._round_context_builder
        cur_repo_fn = globals()["resolve_cannbench_repo"]
        rcb._hook_resolve_cannbench_repo = (
            cur_repo_fn if cur_repo_fn is not _ORIG_HOOKS["resolve_cannbench_repo"] else None
        )
        rcb._resolve_cannbench_tasks_root_fn = self.__dict__.get("_resolve_cannbench_tasks_root")
        return rcb._resolve_cannbench_tasks_root()

    def _resolve_operator_semantics_for_gates(self) -> PyTorchOpSemantics | None:
        self._sync_coordinators()
        return self._stage_gate_coordinator._resolve_operator_semantics_for_gates()

    def _lookup_case_dims_safe(self, case_num: int, regime_tag: str = "") -> dict[str, object]:
        self._sync_coordinators()
        return self._stage_gate_coordinator._lookup_case_dims_safe(case_num, regime_tag)

    def _has_vf_stack_overflow(self, stage_b: StageBResult) -> bool:
        return StageGateCoordinator._has_vf_stack_overflow(stage_b)

    def _resolve_max_row_width_dims(self, regime_spec: RegimeTargetSpec) -> dict[str, object]:
        self._sync_coordinators()
        return self._stage_gate_coordinator._resolve_max_row_width_dims(regime_spec)

    def _resolve_combine_gate_spec(
        self,
        module_source: str,
        regime_spec: RegimeTargetSpec,
        resolved: ResolvedRegimeConfig,
    ) -> tuple[str, dict[str, object]] | None:
        self._sync_coordinators()
        return self._stage_gate_coordinator._resolve_combine_gate_spec(
            module_source,
            regime_spec,
            resolved,
        )

    def _compile_trackb_stage_b(
        self,
        cpp_source: str,
        selector_flags: Sequence[str] | str | None = None,
    ) -> StageBResult:
        self._sync_coordinators()
        return self._stage_gate_coordinator._compile_trackb_stage_b(
            cpp_source,
            selector_flags,
        )

    def _instantiate_trackb_tu(
        self,
        module_source: str,
        cpp_source: str,
        regime_spec: RegimeTargetSpec,
        *,
        anchor_dims_override: Mapping[str, object] | None = None,
    ) -> str:
        return self._stage_gate_coordinator._instantiate_trackb_tu(
            module_source,
            cpp_source,
            regime_spec,
            anchor_dims_override=anchor_dims_override,
        )

    def _resolve_active_trackb_kernel(
        self,
        module_source: str,
        regime_spec: RegimeTargetSpec,
    ) -> str | None:
        self._sync_coordinators()
        return self._stage_gate_coordinator._resolve_active_trackb_kernel(
            module_source,
            regime_spec,
        )

    def _resolve_active_trackb_kernels(
        self,
        module_source: str,
        regime_spec: RegimeTargetSpec,
    ) -> tuple[str, ...]:
        self._sync_coordinators()
        return self._stage_gate_coordinator._resolve_active_trackb_kernels(
            module_source,
            regime_spec,
        )

    def _resolve_trackb_freedom_mask(
        self,
        regime_spec: RegimeTargetSpec,
    ) -> ConstraintRelaxationMask:
        self._sync_coordinators()
        return self._stage_gate_coordinator._resolve_trackb_freedom_mask(regime_spec)

    def _default_l0_checker(
        self,
        module_source: str,
        regime_spec: RegimeTargetSpec,
        *,
        anchor_dims_override: Mapping[str, object] | None = None,
    ) -> StageAResult:
        self._sync_coordinators()
        return self._stage_gate_coordinator._default_l0_checker(
            module_source,
            regime_spec,
            anchor_dims_override=anchor_dims_override,
        )

    def _run_l0(
        self,
        module_source: str,
        regime_spec: RegimeTargetSpec,
        *,
        anchor_dims_override: Mapping[str, object] | None = None,
    ) -> StageAResult:
        self._sync_coordinators()
        return self._stage_gate_coordinator._run_l0(
            module_source,
            regime_spec,
            anchor_dims_override=anchor_dims_override,
        )

    def _run_l1(self, module_source: str, regime_spec: RegimeTargetSpec) -> StageBResult:
        self._sync_coordinators()
        return self._stage_gate_coordinator._run_l1(module_source, regime_spec)

    def _l1_summary_for_first_round(
        self,
        module_source: str,
        regime_spec: RegimeTargetSpec,
    ) -> BishengResourceSummary | None:
        self._sync_coordinators()
        return self._stage_gate_coordinator._l1_summary_for_first_round(
            module_source,
            regime_spec,
        )

    def _normalize_paired_l2_results(
        self,
        raw_results: Mapping[str, CandidateSurrogateResult],
        incumbent_slot_result: CandidateSurrogateResult | None,
    ) -> dict[str, CandidateSurrogateResult]:
        self._sync_coordinators()
        return self._stage_gate_coordinator._normalize_paired_l2_results(
            raw_results,
            incumbent_slot_result,
        )

    def _run_l2_batch(
        self,
        candidates: Sequence[tuple[str, Path]],
        active_anchors: Sequence[AnchorCaseSpec],
        *,
        include_guardrails: bool = True,
    ) -> tuple[dict[str, CandidateSurrogateResult], dict[str, str]]:
        self._sync_coordinators()
        res = self._stage_gate_coordinator._run_l2_batch(
            candidates,
            active_anchors,
            include_guardrails=include_guardrails,
        )
        self._last_l2_timing_meta = dict(self._stage_gate_coordinator._last_l2_timing_meta)
        return res

    def _unchanged_sibling_regimes(
        self,
        incumbent_src: str,
        candidate_src: str,
        active_regime_spec: RegimeTargetSpec,
    ) -> set[str]:
        self._sync_coordinators()
        return self._stage_gate_coordinator._unchanged_sibling_regimes(
            incumbent_src,
            candidate_src,
            active_regime_spec,
        )

    @staticmethod
    def _touched_symbols_for_regime_snippet(
        regime_spec: RegimeTargetSpec,
        dsl_text: str,
    ) -> set[str]:
        if regime_spec.runtime_backend in ("ascendc_trackb", "hybrid_trackb"):
            if "<<<<<<< SEARCH" in dsl_text or (
                "=======" in dsl_text and ">>>>>>> REPLACE" in dsl_text
            ):
                touched = set(regime_spec.trackb_symbols)
                residual = _trackb_python_residual(dsl_text)
                if residual:
                    touched.update(
                        sym.name for sym in _collect_top_level_symbols(ast.parse(residual))
                    )
                return touched
            if not _parses_as_python(dsl_text) and _looks_like_bare_trackb_cpp(dsl_text):
                return set(regime_spec.trackb_symbols)
        return {sym.name for sym in _collect_top_level_symbols(ast.parse(dsl_text))}

    def _prune_stale_sibling_stepping_stones(
        self,
        promoted_regime: RegimeTargetSpec,
        promoted_snippet: str,
    ) -> None:
        own_editable = set(promoted_regime.editable_symbols)
        if not own_editable:
            return
        promoted_syms = self._touched_symbols_for_regime_snippet(
            promoted_regime, promoted_snippet
        )
        touched_own = promoted_syms & own_editable
        if not touched_own:
            return
        for other_spec in self._regime_specs:
            if other_spec.regime_tag == promoted_regime.regime_tag:
                continue
            shared_syms = touched_own & set(other_spec.editable_symbols)
            if not shared_syms:
                continue
            other_beam = self._beams.get(other_spec.regime_tag)
            if other_beam is None or not other_beam.stepping_stone_beam:
                continue
            kept_stones: list[BeamCandidate] = []
            for stone in other_beam.stepping_stone_beam:
                stone_syms = self._touched_symbols_for_regime_snippet(
                    other_spec, stone.dsl_source
                )
                if not (stone_syms & shared_syms):
                    kept_stones.append(stone)
            other_beam.stepping_stone_beam = kept_stones

    def _evaluate_dual_layer_tail_guard(
        self,
        *,
        candidate_id: str,
        candidate_source: str,
        top_res: CandidateSurrogateResult,
        best_result: CandidateSurrogateResult,
        regime_spec: RegimeTargetSpec,
        active_anchors: Sequence[AnchorCaseSpec],
        incumbent_source: str = "",
    ) -> tuple[bool, CandidateSurrogateResult]:
        self._sync_coordinators()
        ok, updated = self._stage_gate_coordinator._evaluate_dual_layer_tail_guard(
            candidate_id=candidate_id,
            candidate_source=candidate_source,
            top_res=top_res,
            best_result=best_result,
            regime_spec=regime_spec,
            active_anchors=active_anchors,
            incumbent_source=incumbent_source,
        )
        self._last_tail_guard_reject_reason = (
            self._stage_gate_coordinator._last_tail_guard_reject_reason
        )
        return ok, updated

    def _refresh_guardrail_incumbent_refs(self, winner_candidate_id: str) -> None:
        self._sync_coordinators()
        self._l3_l4_recovery.refresh_guardrail_incumbent_refs(winner_candidate_id)

    def _record_l3_guardrail_cases(
        self,
        l3_res: L3AcceptanceResult,
        incumbent_source: str,
        regime_spec: RegimeTargetSpec,
        *,
        topology_axes: TopologyAxes | None = None,
    ) -> list[GuardrailCaseSpec]:
        self._sync_coordinators()
        return self._l3_l4_recovery.record_l3_guardrail_cases(
            l3_res,
            incumbent_source,
            regime_spec,
            topology_axes=topology_axes if topology_axes is not None else self._topology_axes,
        )

    def _is_regime_satisfied(
        self,
        regime_tag: str | RegimeTargetSpec,
        best_result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> bool:
        tag = regime_tag if isinstance(regime_tag, str) else regime_tag.regime_tag
        rcb = getattr(self, "_round_context_builder", None)
        if rcb is not None:
            self._sync_coordinators()
            return rcb._is_regime_satisfied(tag, best_result, active_anchors)
        regime_res = best_result.for_regime(tag)
        if not regime_res.anchors:
            return False
        return (
            regime_res.all_passed
            and regime_res.min_case_speedup >= self._perf_target.target_min_case_speedup
            and regime_res.avg_roofline_eta >= self._perf_target.target_roofline_eta
        )

    def _regime_mean_recoverable_excess_us(
        self,
        best_result: CandidateSurrogateResult,
        regime_spec: RegimeTargetSpec | str,
    ) -> float | None:
        rcb = getattr(self, "_round_context_builder", None)
        if rcb is not None:
            self._sync_coordinators()
            return rcb._regime_mean_recoverable_excess_us(best_result, regime_spec)
        return None

    def _select_active_regime(
        self,
        current_regime: str | None,
        frozen_regimes: set[str],
        best_result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec],
        *,
        allow_satisfied_fallback: bool = False,
        exclude_regime: str | None = None,
        stagnation_by_regime: Mapping[str, int] | None = None,
        last_round_promoted: bool | None = None,
        **_extra: object,
    ) -> RegimeTargetSpec | None:
        rcb = getattr(self, "_round_context_builder", None)
        eff_stag = (
            stagnation_by_regime
            if stagnation_by_regime is not None
            else getattr(self, "_live_stag_by_regime", {})
        )
        if rcb is not None:
            self._sync_coordinators()
            return rcb._select_active_regime(
                current_regime,
                frozen_regimes,
                best_result,
                active_anchors,
                allow_satisfied_fallback=allow_satisfied_fallback,
                exclude_regime=exclude_regime,
                stagnation_by_regime=eff_stag,
                last_round_promoted=last_round_promoted,
                regime_specs=self._regime_specs,
            )
        candidates = [
            s
            for s in self._regime_specs
            if s.regime_tag not in frozen_regimes and s.regime_tag != exclude_regime
        ]
        if not candidates:
            return None
        unsatisfied = [
            s
            for s in candidates
            if not self._is_regime_satisfied(s.regime_tag, best_result, active_anchors)
        ]
        pool = unsatisfied if unsatisfied else (candidates if allow_satisfied_fallback else [])
        if not pool:
            return None
        discount = float(getattr(self, "_stagnation_discount", 0.45))

        def _score(spec: RegimeTargetSpec) -> tuple[float, int, str]:
            r_res = best_result.for_regime(spec.regime_tag)
            gap = max(
                0.0,
                float(self._perf_target.target_min_case_speedup) - float(r_res.min_case_speedup),
            )
            stag = int(eff_stag.get(spec.regime_tag, 0))
            return (-(gap * (discount**stag)), stag, spec.regime_tag)

        return min(pool, key=_score)

    def _allow_early_stop(self, rounds_completed: int) -> bool:
        if self._config.disable_early_goal_stop:
            return False
        return (
            rounds_completed > 0
            and rounds_completed >= self._perf_target.min_optimization_rounds
        )

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
    def _slice_contract_instruction(
        func_name: str,
        regime_spec: RegimeTargetSpec,
        *,
        operator_entrypoint: str = "",
    ) -> str:
        return RoundContextBuilder._slice_contract_instruction(
            func_name,
            regime_spec,
            operator_entrypoint=operator_entrypoint,
        )

    @staticmethod
    def _format_msprof_section(
        residual: object,
        *,
        p2_diag: Mapping[str, object] | None = None,
    ) -> str:
        return RoundContextBuilder._format_msprof_section(residual, p2_diag=p2_diag)

    def _build_exploit_regime_request(
        self,
        *,
        snippet_source: str,
        func_name: str,
        regime_spec: RegimeTargetSpec,
        rule: AscendCausalRule,
        best_result: CandidateSurrogateResult,
        blueprint: HirBlueprintPacket | None = None,
        parent_ledger: object | None = None,
        parent_residual: object | None = None,
        track: str = "unconstrained",
        incumbent_source: str = "",
        sil_policy_section: str = "",
        effort: str | None = None,
        p2_diag: Mapping[str, object] | None = None,
        **_extra: object,
    ) -> GenerationRequest:
        self._sync_coordinators()
        return self._round_context_builder._build_exploit_regime_request(
            snippet_source=snippet_source,
            func_name=func_name,
            regime_spec=regime_spec,
            rule=rule,
            best_result=best_result,
            blueprint=blueprint,
            parent_ledger=parent_ledger,
            parent_residual=parent_residual,
            track=track,
            incumbent_source=incumbent_source,
            sil_policy_section=sil_policy_section,
            effort=effort,
            p2_diag=p2_diag,
        )

    def _build_creative_regime_request(
        self,
        *,
        snippet_source: str,
        func_name: str,
        regime_spec: RegimeTargetSpec,
        best_result: CandidateSurrogateResult,
        effort: str,
        excluded_mechanisms: Sequence[str] = (),
        focus_hint: str = "",
        prescreen_summary: str = "",
        blueprint: HirBlueprintPacket | None = None,
        parent_ledger: object | None = None,
        parent_residual: object | None = None,
        track: str = "unconstrained",
        incumbent_source: str = "",
        parent_hypothesis: str = "",
        sil_policy_section: str = "",
        p2_diag: Mapping[str, object] | None = None,
        paradigm_shift: bool = False,
        **_extra: object,
    ) -> GenerationRequest:
        self._sync_coordinators()
        return self._round_context_builder._build_creative_regime_request(
            snippet_source=snippet_source,
            func_name=func_name,
            regime_spec=regime_spec,
            best_result=best_result,
            effort=effort,
            excluded_mechanisms=excluded_mechanisms,
            focus_hint=focus_hint,
            prescreen_summary=prescreen_summary,
            blueprint=blueprint,
            parent_ledger=parent_ledger,
            parent_residual=parent_residual,
            track=track,
            incumbent_source=incumbent_source,
            parent_hypothesis=parent_hypothesis,
            sil_policy_section=sil_policy_section,
            p2_diag=p2_diag,
            paradigm_shift=paradigm_shift,
        )

    def _collect_msprof_for_regimes(
        self,
        source: str,
        result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec],
        target_regimes: Sequence[RegimeTargetSpec],
    ) -> dict[tuple[str, int], dict[str, float]]:
        self._sync_coordinators()
        collected = self._calibration_service.collect_msprof_by_case(
            source,
            result,
            active_anchors,
            target_regimes,
        )
        sha = self._source_sha256(source)
        for (_csha, cnum), metrics in collected.items():
            for anc in active_anchors:
                if int(anc.case_num) == int(cnum):
                    self._msprof_cache[(sha, anc.regime_tag)] = dict(metrics)
            for spec in target_regimes:
                if (sha, spec.regime_tag) not in self._msprof_cache:
                    self._msprof_cache[(sha, spec.regime_tag)] = dict(metrics)
        return collected

    def _collect_p2_deep_dive_for_regime(
        self,
        source: str,
        result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec],
        regime_spec: RegimeTargetSpec,
    ) -> dict[str, object] | None:
        self._sync_coordinators()
        return self._calibration_service.collect_p2_deep_dive_for_regime(
            source,
            result,
            active_anchors,
            regime_spec,
        )

    def _evaluate_single_baseline(
        self,
        incumbent_source: str,
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> CandidateSurrogateResult:
        self._sync_coordinators()
        base_res, null_launch_us = self._calibration_service.evaluate_baseline(
            incumbent_source,
            active_anchors,
        )
        self._null_launch_us = null_launch_us
        return base_res

    def _on_warmstart_promoted(self, src: str, res: CandidateSurrogateResult) -> None:
        self._active_incumbent_source = src
        self._incumbent_anchor_us_by_case = {
            int(a.case_num): float(a.cand_us) for a in res.anchors if a.cand_us > 0.0
        }
        self._stage_gate_coordinator._incumbent_anchor_us_by_case = (
            self._incumbent_anchor_us_by_case
        )
        self._stage_gate_coordinator._active_incumbent_source = src

    def _run_phase0_lattice_warmstart(
        self,
        incumbent_source: str,
        best_result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> tuple[str, CandidateSurrogateResult]:
        self._sync_coordinators()
        return self._calibration_service.run_phase0_lattice_warmstart(
            incumbent_source,
            best_result,
            active_anchors,
        )

    @staticmethod
    def _evaluate_causal_rule_outcome(
        *,
        rule: AscendCausalRule | None = None,
        res_all_passed: bool = True,
        rolled_back: bool = False,
        ledger: ObjectivePhysicalLedger | None = None,
        features: object = None,
        cand_residual: StaticDynamicResidual | None = None,
        cand_lat_us: float = 0.0,
        cand_obj_score: float = 0.0,
        parent_ledger: ObjectivePhysicalLedger | None = None,
        parent_residual: StaticDynamicResidual | None = None,
        parent_lat_us: float = 0.0,
        parent_obj_score: float = 0.0,
        has_cand_msprof: bool = False,
        cand_global_score: float | None = None,
        parent_global_score: float | None = None,
        min_speedup_gain: float = 0.01,
        min_score_gain: float = 0.05,
        cand_compiler_summary: BishengResourceSummary | None = None,
        roofline: HirRooflineSummary | None = None,
        **extra: object,
    ) -> tuple[bool, float]:
        if (
            rule is not None
            and ledger is not None
            and cand_residual is not None
            and parent_ledger is not None
            and parent_residual is not None
        ):
            return TelemetryLedgerWriter.evaluate_causal_rule_outcome(
                rule=rule,
                res_all_passed=res_all_passed,
                rolled_back=rolled_back,
                ledger=ledger,
                features=features,
                cand_residual=cand_residual,
                cand_lat_us=cand_lat_us,
                cand_obj_score=cand_obj_score,
                parent_ledger=parent_ledger,
                parent_residual=parent_residual,
                parent_lat_us=parent_lat_us,
                parent_obj_score=parent_obj_score,
                has_cand_msprof=has_cand_msprof,
                cand_global_score=cand_global_score,
                parent_global_score=parent_global_score,
                min_speedup_gain=min_speedup_gain,
                min_score_gain=min_score_gain,
                cand_compiler_summary=cand_compiler_summary,
                roofline=roofline,
            )
        _ = extra
        if not res_all_passed or rolled_back:
            return False, 0.0
        if parent_lat_us > 0.0 and cand_lat_us > 0.0 and cand_lat_us >= parent_lat_us:
            return False, 0.0
        if cand_obj_score < parent_obj_score - 1e-6:
            return False, 0.0
        rel_gain = (
            (parent_lat_us - cand_lat_us) / max(parent_lat_us, 1e-6)
            if (parent_lat_us > 0.0 and cand_lat_us > 0.0)
            else 0.0
        )
        ok = rel_gain >= float(min_speedup_gain) or (
            cand_obj_score - parent_obj_score >= float(min_score_gain)
            and (parent_lat_us <= 0.0 or cand_lat_us <= 0.0 or cand_lat_us < parent_lat_us)
        )
        return bool(ok), (rel_gain if ok else 0.0)

    def _record_surrogate_observations(
        self,
        *,
        source: str,
        result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec],
        eval_tier_override: str | None = None,
        parent_source: str = "",
        parent_result: CandidateSurrogateResult | None = None,
        parent_ledger: object | None = None,
        parent_bisheng_res: BishengResourceSummary | Mapping[str, object] | None = None,
        parent_stage_b_profile: Mapping[str, object] | None = None,
        parent_axes: TopologyAxes | None = None,
        ledger: object | None = None,
        bisheng_summary: BishengResourceSummary | Mapping[str, object] | None = None,
        stage_b_profile: Mapping[str, object] | None = None,
        axes: TopologyAxes | None = None,
        policy: Mapping[str, object] | None = None,
        parent_policy: Mapping[str, object] | None = None,
        worker_kind: str = "",
        rule_id: str | None = None,
        backend_track: str = "tilelang_dsl",
        regime_tag: str = "",
    ) -> None:
        self._sync_coordinators()
        self._telemetry_writer.record_surrogate_observations(
            source=source,
            result=result,
            active_anchors=active_anchors,
            eval_tier_override=eval_tier_override,
            parent_source=parent_source,
            parent_result=parent_result,
            parent_ledger=parent_ledger,
            parent_bisheng_res=parent_bisheng_res,
            parent_stage_b_profile=parent_stage_b_profile,
            parent_axes=parent_axes,
            ledger=ledger,
            bisheng_summary=bisheng_summary,
            stage_b_profile=stage_b_profile,
            axes=axes,
            policy=policy,
            parent_policy=parent_policy,
            worker_kind=worker_kind,
            rule_id=rule_id,
            backend_track=backend_track,
            regime_tag=regime_tag,
        )

    def _record_l4_observations(
        self,
        *,
        source: str,
        l4_result: L4CannbenchResult,
        best_result: CandidateSurrogateResult | None = None,
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> None:
        _ = best_result
        self._sync_coordinators()
        self._telemetry_writer.record_l4_observations(
            source=source,
            l4_result=l4_result,
            active_anchors=active_anchors,
        )

    def _collect_trackb_round_evidence(
        self,
        *,
        round_idx: int,
        regime_spec: RegimeTargetSpec,
        blueprint: HirBlueprintPacket | None,
        live_plan: RegimePlanResult | None,
        active_mask: ConstraintRelaxationMask,
        checks: Sequence[L0L1CandidateCheck],
        l2_results: Mapping[str, CandidateSurrogateResult],
        active_kernel: str,
        l2_run_ids: Mapping[str, str],
        parent_sha256: str,
        parent_l1_summary: BishengResourceSummary | None,
        parent_l1_summary_source: str,
    ) -> tuple[dict[str, object], ...]:
        self._sync_coordinators()
        return self._telemetry_writer.collect_trackb_round_evidence(
            round_idx=round_idx,
            regime_spec=regime_spec,
            blueprint=blueprint,
            live_plan=live_plan,
            active_mask=active_mask,
            checks=checks,
            l2_results=l2_results,
            active_kernel=active_kernel,
            l2_run_ids=l2_run_ids,
            parent_sha256=parent_sha256,
            parent_l1_summary=parent_l1_summary,
            parent_l1_summary_source=parent_l1_summary_source,
        )

    def _run_flywheel_distillation(
        self,
        *,
        baseline_source: str,
        winner_source: str,
        baseline_latency_us: float,
        winner_latency_us: float,
        func_name: str,
        regime_spec: RegimeTargetSpec,
        active_anchors: Sequence[AnchorCaseSpec],
        topology_axes: TopologyAxes,
        rel_gain: float = 0.0,
        local_rel_gain: float = 0.0,
        worker_kind: str = "",
        round_idx: int = 0,
        rule_id: str | None = None,
        symptom_family: str = "",
        sigma_paired_us: float = 0.0,
    ) -> list[str]:
        self._sync_coordinators()
        return self._telemetry_writer.run_flywheel_distillation(
            baseline_source=baseline_source,
            winner_source=winner_source,
            baseline_latency_us=baseline_latency_us,
            winner_latency_us=winner_latency_us,
            func_name=func_name,
            regime_spec=regime_spec,
            active_anchors=active_anchors,
            topology_axes=topology_axes,
            rel_gain=rel_gain,
            local_rel_gain=local_rel_gain,
            worker_kind=worker_kind,
            round_idx=round_idx,
            rule_id=rule_id,
            symptom_family=symptom_family,
            sigma_paired_us=sigma_paired_us,
            tracker=self._tracker,
        )

    def _build_stagnation_diagnosis(
        self,
        *,
        best_result: CandidateSurrogateResult,
        stop_reason: str,
    ) -> dict[str, object]:
        self._sync_coordinators()
        return self._telemetry_writer.build_stagnation_diagnosis(
            best_result=best_result,
            stop_reason=stop_reason,
        )

    def _save_checkpoint(
        self,
        *,
        incumbent_source: str,
        baseline_result: CandidateSurrogateResult,
        best_result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec],
        frozen_regimes: set[str],
        current_regime: str | None,
        stagnation_rounds: int,
        rounds: Sequence[CampaignRoundRecord],
        promoted_rounds: Sequence[int],
        rolled_back_rounds: Sequence[int],
        escalated_rounds: Sequence[int],
        admitted_rule_ids: Sequence[str],
        stop_reason: str,
        l4_result: L4CannbenchResult | None,
        stagnation_by_regime: Mapping[str, int] | None = None,
        last_unpromoted_regime: str | None = None,
    ) -> None:
        self._sync_coordinators()
        self._telemetry_writer.save_checkpoint(
            incumbent_source=incumbent_source,
            baseline_result=baseline_result,
            best_result=best_result,
            active_anchors=active_anchors,
            frozen_regimes=frozen_regimes,
            current_regime=current_regime,
            stagnation_rounds=stagnation_rounds,
            rounds=rounds,
            promoted_rounds=promoted_rounds,
            rolled_back_rounds=rolled_back_rounds,
            escalated_rounds=escalated_rounds,
            admitted_rule_ids=admitted_rule_ids,
            stop_reason=stop_reason,
            l4_result=l4_result,
            stagnation_by_regime=stagnation_by_regime,
            last_unpromoted_regime=last_unpromoted_regime,
            pending_promotion=self._pending_promotion,
            pending_l4=self._pending_l4,
            active_promotion=self._active_promotion,
            scoreboard_anchors=self._scoreboard_anchors,
            guardrail_cases=self._guardrail_cases,
            calibrated_dispatch_floor_us=self._calibrated_dispatch_floor_us,
            last_round_promoted_by_regime=self._last_round_promoted_by_regime,
            compile_wall_streak_by_regime=self._compile_wall_streak_by_regime,
        )

    def _restore_from_checkpoint(self):
        self._sync_coordinators()
        restored = self._telemetry_writer.restore_from_checkpoint(
            round_record_from_dict=CampaignRoundRecord.from_dict,
            pending_l4_from_dict=PendingL4.from_dict,
        )
        tlw = self._telemetry_writer
        self._run_id = tlw.run_id
        self._active_regime_tag = tlw.active_regime_tag
        self._scoreboard_anchors = tlw.scoreboard_anchors
        self._guardrail_cases = tlw.guardrail_cases
        self._calibrated_dispatch_floor_us = tlw.calibrated_dispatch_floor_us
        self._stagnation_by_regime = tlw.stagnation_by_regime
        self._compile_wall_streak_by_regime = tlw.compile_wall_streak_by_regime
        self._last_round_promoted_by_regime = tlw.last_round_promoted_by_regime
        self._last_local_rel_gain_by_regime = tlw.last_local_rel_gain_by_regime
        self._active_promotion = tlw.active_promotion
        self._pending_promotion = tlw.pending_promotion
        self._pending_l4 = tlw.pending_l4
        return restored

    def _run_l3_contract(
        self,
        module_path: Path,
        *,
        resume_execution_id: str | None = None,
        resume_run_id: str | None = None,
        device_id: str | None = None,
        on_execution_started: Callable[..., None] | None = None,
    ) -> L3AcceptanceResult:
        self._sync_coordinators()
        if resume_execution_id and self._l3_contract_runner is not None:
            return self._invoke_with_optional_kwargs(
                self._l3_contract_runner,
                module_path,
                resume_execution_id=resume_execution_id,
                resume_run_id=resume_run_id,
                device_id=device_id,
                on_execution_started=on_execution_started,
            )
        return self._l3_l4_recovery.run_l3_contract(
            module_path,
            device_id=device_id,
            on_execution_started=on_execution_started,
        )

    def _resume_l3_contract(self, pending: PendingPromotion) -> L3AcceptanceResult:
        self._sync_coordinators()
        return self._l3_l4_recovery.resume_l3_contract(
            self._target_module_path,
            pending,
        )

    def _resolve_pending_promotion(
        self,
        pending: PendingPromotion,
        *,
        incumbent_source: str,
        best_result: CandidateSurrogateResult,
        stagnation_by_regime: dict[str, int],
        rounds: list[CampaignRoundRecord],
        promoted_rounds: list[int],
        rolled_back_rounds: list[int],
        active_anchors: Sequence[AnchorCaseSpec],
    ):
        self._sync_coordinators()
        out = self._l3_l4_recovery.resolve_pending_promotion(
            pending,
            incumbent_source=incumbent_source,
            best_result=best_result,
            stagnation_by_regime=stagnation_by_regime,
            rounds=rounds,
            promoted_rounds=promoted_rounds,
            rolled_back_rounds=rolled_back_rounds,
            active_anchors=active_anchors,
        )
        self._active_promotion = out[3]
        self._pending_promotion = out[3]
        return out

    def _run_l4_cannbench(
        self,
        module_path: Path,
        *,
        resume_execution_id: str | None = None,
        resume_run_id: str | None = None,
        device_id: str | None = None,
        on_execution_started: Callable[..., None] | None = None,
    ) -> L4CannbenchResult:
        self._sync_coordinators()
        return self._l3_l4_recovery.run_l4_cannbench(
            module_path,
            resume_execution_id=resume_execution_id or "",
            resume_run_id=resume_run_id or "",
            device_id=device_id,
            on_execution_started=on_execution_started,
        )

    def _resolve_pending_l4(
        self,
        pending: PendingL4,
        *,
        incumbent_source: str = "",
    ) -> tuple[L4CannbenchResult, PendingL4 | None]:
        self._sync_coordinators()
        eff_inc = incumbent_source or self._active_incumbent_source
        res, new_pending = self._l3_l4_recovery.resolve_pending_l4(
            pending,
            incumbent_source=eff_inc,
        )
        self._pending_l4 = new_pending
        return res, new_pending

    def _run_final_l4_tracked(
        self,
        *,
        incumbent_source: str,
        prior_stop_reason: str,
    ) -> L4CannbenchResult:
        self._sync_coordinators()
        res = self._l3_l4_recovery.run_final_l4_tracked(
            incumbent_source=incumbent_source,
            prior_stop_reason=prior_stop_reason,
        )
        self._pending_l4 = self._l3_l4_recovery.pending_l4
        return res

    def _run_l3_with_card_retry(
        self,
        *,
        cand: RankedL2Candidate,
        round_idx: int,
        regime_tag: str,
        attempts: list[dict[str, str]],
    ) -> L3AcceptanceResult:
        self._sync_coordinators()
        attempted_devices: set[str] = set()
        pending_stub = PendingPromotion(
            candidate_source=cand.candidate_source,
            candidate_sha256=cand.candidate_sha256,
            target_path=str(self._target_module_path),
            round_idx=round_idx,
            regime_tag=regime_tag,
            candidate_result=cand.candidate_result,
            candidate_id=cand.candidate_id,
            builder_name=cand.builder_name,
            parent_lane=cand.parent_lane,
            creative_effort=cand.creative_effort,
        )

        def _on_started(*args: str) -> None:
            exec_id = args[0] if len(args) > 0 else ""
            run_id = args[2] if len(args) > 3 else ""
            receipt_path = args[3] if len(args) > 3 else ""
            dev_id = (args[1] if len(args) == 2 else (args[3] if len(args) > 3 else ""))
            if len(args) == 4:
                run_id, receipt_path, dev_id = args[1], args[2], args[3]
            active = replace(
                pending_stub,
                execution_id=exec_id,
                run_id=run_id,
                receipt_path=receipt_path,
                device_id=dev_id,
                execution_terminal=False,
            )
            self._active_promotion = active
            self._pending_promotion = active
            self._persist_runtime_state("active_promotion", active.to_dict())

        device_id = self._l4_device_pool[0] if self._l4_device_pool else self._config.device_id
        extra_pool = (
            (*self._l4_device_pool, self._retry_device_id)
            if self._retry_device_id
            else tuple(self._l4_device_pool)
        )
        while True:
            attempted_devices.add(device_id)
            l3_res = self._run_l3_contract(
                self._target_module_path,
                device_id=device_id,
                on_execution_started=_on_started,
            )
            for busy_backoff_sec in (2.0, 5.0, 10.0):
                if (
                    l3_res.passed
                    or l3_res.failure_kind == FAILURE_KIND_CANDIDATE
                    or "resource busy:" not in (l3_res.error_message or "").lower()
                ):
                    break
                logger.warning(
                    "Round %d L3 hit transient resource busy (%s); retrying after %.1fs",
                    round_idx,
                    (l3_res.error_message or "")[:160],
                    busy_backoff_sec,
                )
                time.sleep(busy_backoff_sec)
                l3_res = self._run_l3_contract(
                    self._target_module_path,
                    device_id=device_id,
                    on_execution_started=_on_started,
                )
            if l3_res.execution_terminal:
                self._active_promotion = None
            attempts.append(
                self._l3_l4_recovery.promotion_attempt(
                    l3_res,
                    l3_res.device_id or device_id,
                    candidate_sha256=cand.candidate_sha256,
                )
            )
            if l3_res.passed or l3_res.failure_kind == FAILURE_KIND_CANDIDATE:
                return l3_res
            if not l3_res.execution_terminal:
                return l3_res
            alt = next(
                (d for d in extra_pool if d and d not in attempted_devices),
                "",
            )
            if not alt:
                return l3_res
            device_id = alt

    async def _execute_single_round(
        self,
        *,
        round_idx: int,
        regime_spec: RegimeTargetSpec,
        incumbent_source: str,
        best_result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec],
        stagnation_rounds: int,
        admitted_rule_ids: list[str],
    ) -> tuple[
        CampaignRoundRecord,
        str,
        CandidateSurrogateResult,
        int,
        bool,
        bool,
        bool,
        bool,
    ]:
        """Execute one optimization round across context synthesis, gates, L3, and telemetry."""
        self._sync_coordinators()
        self._active_incumbent_source = incumbent_source
        self._active_regime_tag = regime_spec.regime_tag
        self._stage_gate_coordinator._active_incumbent_source = incumbent_source
        self._incumbent_anchor_us_by_case = {
            int(a.case_num): float(a.cand_us)
            for a in best_result.anchors
            if a.cand_us > 0.0
        }
        self._stage_gate_coordinator._incumbent_anchor_us_by_case = (
            self._incumbent_anchor_us_by_case
        )

        stagnation_map = dict(self._stagnation_by_regime)
        stagnation_map[regime_spec.regime_tag] = stagnation_rounds
        creative_effort = "high"
        merged_msprof_by_case = TelemetryLedgerWriter.fold_regime_msprof_cache(
            self._msprof_by_case, self._msprof_cache, active_anchors,
            active_regime_tag=regime_spec.regime_tag,
            single_regime=len(self._regime_specs) <= 1,
        )
        state = ChampionState(
            incumbent_source=incumbent_source, best_result=best_result,
            active_anchors=tuple(active_anchors), regime_specs=self._regime_specs or (regime_spec,),
            stagnation_rounds=stagnation_rounds, stagnation_by_regime=stagnation_map,
            compile_wall_streak_by_regime=self._compile_wall_streak_by_regime,
            last_round_promoted_by_regime=self._last_round_promoted_by_regime,
            last_round_promoted=self._last_round_promoted,
            last_local_rel_gain_by_regime=self._last_local_rel_gain_by_regime,
            freedom_masks=self._freedom_masks, msprof_by_case=merged_msprof_by_case,
            p2_cache=self._p2_cache,
            calibrated_dispatch_floor_us=self._calibrated_dispatch_floor_us,
            boundary_ledger=self._boundary_ledger, beams=self._beams,
            current_regime=regime_spec.regime_tag, active_kernel_cache=self._active_kernel_cache,
            live_regime_plans=self._live_regime_plans,
            operator_proto_cache=self._operator_proto_cache,
            sil_advisory=self._sil_advisory, topology_axes=self._topology_axes,
            semantic_topology_axes=self._semantic_topology_axes,
            regime_blueprints=self._regime_blueprints,
        )
        bundle = self._round_context_builder.build_round_bundle(
            state, round_idx=round_idx, regime_spec=regime_spec,
            working_anchors=active_anchors, creative_effort=creative_effort,
        )
        self._freedom_masks.update(self._round_context_builder._freedom_masks)
        self._blueprint_provenance = dict(self._round_context_builder._blueprint_provenance)
        self._msprof_by_case.update(self._calibration_service.msprof_by_case)

        bn_anchor = bottleneck_anchor_for_regime(best_result, regime_spec.regime_tag)
        bn_dims: dict[str, int] | None = None
        if bn_anchor is not None:
            resolved_bn = self._lookup_case_dims_safe(bn_anchor.case_num, regime_spec.regime_tag)
            if resolved_bn:
                bn_dims = {
                    str(k): int(v) for k, v in resolved_bn.items()
                    if isinstance(v, int) and not isinstance(v, bool)
                }

        batch = CandidateBatch(
            round_idx=round_idx, regime_spec=regime_spec, worker_specs=bundle.worker_specs,
            parent_module_source=bundle.parent_module_source, parent_snippet=bundle.parent_snippet,
            parent_ledger=bundle.parent_ledger, parent_axes=bundle.parent_axes,
            active_mask=bundle.active_mask, bn_dims=bn_dims, live_plan=bundle.live_plan,
            blueprint=bundle.blueprint, gate_decision=bundle.gate_decision,
            parent_p2_diag=bundle.parent_p2_diag, func_name=bundle.func_name,
        )
        outcome = await self._stage_gate_coordinator.execute_round_gates(
            batch, active_anchors, backend=self._backend,
            parent_sha=bundle.parent_sha, parent_workload=bundle.parent_workload,
        )

        self._last_l2_timing_meta = dict(self._stage_gate_coordinator._last_l2_timing_meta)
        op_entry = _resolve_operator_entrypoint_safe(self._config.operator_name)
        regime_beam = self.beam_for_regime(regime_spec.regime_tag)
        checks_by_cid = {c.candidate_id: c for c in outcome.passed_checks}

        local_best = best_result.for_regime(regime_spec.regime_tag)
        best_round_speedup = best_result.avg_speedup
        best_round_surrogate_score = best_result.surrogate_score
        all_passing_l2 = [
            chk for chk in outcome.passed_checks
            if chk.candidate_id in outcome.l2_results
            and outcome.l2_results[chk.candidate_id].all_passed
        ]
        if all_passing_l2:
            top_l2_only = max(
                all_passing_l2,
                key=lambda c: _surrogate_rank_key(outcome.l2_results[c.candidate_id]),
            )
            top_l2_res = outcome.l2_results[top_l2_only.candidate_id]
            best_round_speedup = top_l2_res.avg_speedup
            best_round_surrogate_score = top_l2_res.surrogate_score

        promoted = False
        promotion_blocked = False
        l3_resolution = L3Resolution.NOT_RUN.value
        l3_failed_cases: tuple[int, ...] = ()
        l3_failure_stage = ""
        new_incumbent = incumbent_source
        new_best = best_result
        stagnation_after = stagnation_rounds
        rel_gain = 0.0
        local_rel_gain = 0.0
        tail_guard_reject_reason = ""
        promotion_attempts: tuple[dict[str, str], ...] = ()
        top_chk: L0L1CandidateCheck | None = None
        top_res: CandidateSurrogateResult | None = None
        initial_top_cid: str | None = None
        l3_sec = 0.0
        passed_checks = list(outcome.passed_checks)
        l2_results = dict(outcome.l2_results)
        meta_by_id = dict(outcome.meta_by_id)

        is_occupancy_fault = (
            outcome.l2_infra_fault is not None
            and outcome.l2_infra_fault.plane
            in (FaultPlane.INFRA_OCCUPANCY, FaultPlane.INFRA_DEVICE_FAULT)
        )
        if is_occupancy_fault:
            l3_resolution = L3Resolution.INFRA_DEFERRED.value
            promotion_blocked = True
            stagnation_after = stagnation_rounds
        else:
            ranked_candidates: list[RankedL2Candidate] = []
            unchanged_by_cid: dict[str, set[str]] = {}
            guardrail_by_cid = dict(self._stage_gate_coordinator._last_guardrail_metrics_by_cid)
            for chk in all_passing_l2:
                cand_res = l2_results[chk.candidate_id]
                wkind, _rule, ceff = meta_by_id.get(
                    chk.candidate_id, ("exploit_rule", None, "")
                )
                unchanged = self._unchanged_sibling_regimes(
                    incumbent_source,
                    chk.patched_module_source,
                    regime_spec,
                )
                unchanged_by_cid[chk.candidate_id] = unchanged
                rc = RankedL2Candidate(
                    candidate_id=chk.candidate_id,
                    candidate_source=chk.patched_module_source,
                    candidate_result=cand_res,
                    candidate_sha256=self._source_sha256(chk.patched_module_source),
                    builder_name=bundle.func_name,
                    parent_lane=bundle.parent_lane,
                    creative_effort=(
                        ceff or (creative_effort if wkind == "creative_explore" else "")
                    ),
                )

                def _make_retest(cid: str, src: str):
                    return lambda dipped, cur_res: (
                        self._stage_gate_coordinator._retest_noisy_tail_anchors(
                            candidate_id=cid,
                            candidate_source=src,
                            cur_res=cur_res,
                            regressed_anchors=dipped,
                            active_anchors=active_anchors,
                        )
                    )

                def _make_recheck(cid: str, src: str):
                    return lambda anc: (
                        self._stage_gate_coordinator._recheck_single_sibling_anchor(
                            cid, src, anc
                        )
                    )

                single_eval = select_top_two_l2_survivors(
                    [rc],
                    best_result=best_result,
                    regime_spec=regime_spec,
                    guardrail_cases=self._guardrail_cases,
                    guardrail_metrics_by_cid=guardrail_by_cid,
                    unchanged_siblings_by_cid=unchanged_by_cid,
                    active_anchors=active_anchors,
                    retest_noisy_fn=_make_retest(chk.candidate_id, chk.patched_module_source),
                    recheck_single_sibling_fn=_make_recheck(
                        chk.candidate_id, chk.patched_module_source
                    ),
                )
                if single_eval.survivors:
                    ranked_candidates.append(
                        replace(
                            rc,
                            candidate_result=single_eval.survivors[0].candidate_result,
                            speedup=single_eval.survivors[0].speedup,
                        )
                    )
                elif single_eval.rejected and not tail_guard_reject_reason:
                    tail_guard_reject_reason = single_eval.rejected[0][1]

            survivors = RankedL2Survivors(
                survivors=tuple(
                    replace(c, rank=i)
                    for i, c in enumerate(
                        sorted(
                            ranked_candidates,
                            key=lambda item: (
                                -float(item.speedup),
                                -float(item.candidate_result.surrogate_score),
                                item.candidate_id,
                            ),
                        )[:2],
                        start=1,
                    )
                )
            )
            if not survivors.survivors:
                stagnation_after = stagnation_rounds + 1
            else:
                initial_top_cid = survivors.survivors[0].candidate_id
                tail_guard_reject_reason = ""
                l3_t0 = time.monotonic()
                inline_attempts: list[dict[str, str]] = []
                decision = self._promotion_coordinator.coordinate_top_two_l3_promotion(
                    survivors,
                    incumbent_source=incumbent_source,
                    round_idx=round_idx,
                    regime_tag=regime_spec.regime_tag,
                    run_l3_fn=lambda cand: self._run_l3_with_card_retry(
                        cand=cand,
                        round_idx=round_idx,
                        regime_tag=regime_spec.regime_tag,
                        attempts=inline_attempts,
                    ),
                    target_path=str(self._target_module_path),
                )
                promotion_attempts = tuple(inline_attempts)
                if (
                    decision.resolution == L3Resolution.PROMOTED
                    and decision.promoted_candidate is not None
                ):
                    win_cand = decision.promoted_candidate
                    win_res = win_cand.candidate_result
                    top_chk = checks_by_cid.get(win_cand.candidate_id)
                    top_res = win_res
                    promoted = True
                    l3_resolution = L3Resolution.PROMOTED.value
                    new_incumbent = win_cand.candidate_source
                    new_best = win_res
                    stagnation_after = 0
                    best_round_speedup = win_res.avg_speedup
                    best_round_surrogate_score = win_res.surrogate_score
                    rel_gain = (win_res.avg_speedup - best_result.avg_speedup) / max(
                        best_result.avg_speedup, 1e-6
                    )
                    top_local = win_res.for_regime(regime_spec.regime_tag)
                    if top_local.anchors and local_best.anchors:
                        t_lat = top_local.mean_latency_us()
                        b_lat = local_best.mean_latency_us()
                        local_rel_gain = max(
                            (top_local.avg_speedup - local_best.avg_speedup)
                            / max(local_best.avg_speedup, 1e-6),
                            ((b_lat - t_lat) / max(b_lat, 1e-6))
                            if (t_lat > 0.0 and b_lat > 0.0)
                            else 0.0,
                        )
                    self._last_local_rel_gain_by_regime[regime_spec.regime_tag] = local_rel_gain
                    self._refresh_guardrail_incumbent_refs(win_cand.candidate_id)
                    self._pending_promotion = None
                    self._active_promotion = None
                    self._persist_runtime_state("active_promotion", None)
                    self._persist_runtime_state("pending_promotion", None)
                    self._collect_msprof_for_regimes(
                        win_cand.candidate_source,
                        win_res,
                        active_anchors,
                        [regime_spec],
                    )
                elif decision.resolution == L3Resolution.INFRA_DEFERRED:
                    l3_resolution = L3Resolution.INFRA_DEFERRED.value
                    promotion_blocked = True
                    stagnation_after = stagnation_rounds
                    def_cid = (
                        decision.pending_promotion.candidate_id
                        if decision.pending_promotion is not None
                        else initial_top_cid
                    )
                    top_chk = checks_by_cid.get(def_cid) if def_cid else None
                    if decision.l3_result is not None:
                        l3_failed_cases = tuple(
                            self._l3_l4_recovery.extract_l3_failed_case_nums(decision.l3_result)
                        )
                        l3_failure_stage = str(
                            getattr(decision.l3_result, "failure_stage", "") or ""
                        )
                    if decision.pending_promotion is not None:
                        pending_obj = replace(
                            decision.pending_promotion,
                            attempts=tuple(inline_attempts)
                            or decision.pending_promotion.attempts,
                            retry_used=len(inline_attempts) > 1
                            or decision.pending_promotion.retry_used,
                        )
                        self._pending_promotion = pending_obj
                        promotion_attempts = pending_obj.attempts
                        self._persist_runtime_state("pending_promotion", pending_obj.to_dict())
                        if not pending_obj.execution_terminal:
                            self._write_source_and_verify(
                                self._target_module_path, pending_obj.candidate_source
                            )
                            self._active_promotion = pending_obj
                            self._persist_runtime_state(
                                "active_promotion", pending_obj.to_dict()
                            )
                        else:
                            self._active_promotion = None
                            self._persist_runtime_state("active_promotion", None)
                else:
                    l3_resolution = L3Resolution.CANDIDATE_REJECTED.value
                    stagnation_after = stagnation_rounds + 1
                    rej_cid = decision.attempts[-1][0] if decision.attempts else initial_top_cid
                    top_chk = checks_by_cid.get(rej_cid) if rej_cid else None
                    self._pending_promotion = None
                    self._active_promotion = None
                    self._persist_runtime_state("active_promotion", None)
                    self._persist_runtime_state("pending_promotion", None)
                    if decision.l3_result is not None:
                        l3_failure_stage = str(
                            getattr(decision.l3_result, "failure_stage", "") or ""
                        )
                        self._record_l3_guardrail_cases(
                            decision.l3_result,
                            incumbent_source,
                            regime_spec,
                            topology_axes=bundle.parent_axes,
                        )
                        l3_failed_cases = tuple(decision.new_guardrail_cases) or tuple(
                            self._l3_l4_recovery.extract_l3_failed_case_nums(decision.l3_result)
                        )
                l3_sec = max(0.0, time.monotonic() - l3_t0)

        post_t0 = time.monotonic()
        beam_updates: list[BeamCandidate] = []
        round_admitted: list[str] = []
        measured_by_cid: dict[str, float] = {}
        winner_cid = top_chk.candidate_id if top_chk is not None else None
        winner_rule_id: str | None = None
        winner_wkind = ""
        winner_axes = bundle.parent_axes
        parent_bn_lat_us = (
            float(bundle.parent_residual.measured_latency_us)
            if bundle.parent_residual.measured_latency_us > 0.0
            else bundle.parent_lat_us
        )
        top_family = (
            bundle.ranked_families[0][0].name if bundle.ranked_families else ""
        )
        bp_roofline = bundle.blueprint.roofline if bundle.blueprint is not None else None
        selected_stone: BeamCandidate | None = None
        if bundle.selected_stone_id:
            selected_stone = next(
                (
                    item for item in (
                        *getattr(regime_beam, "stepping_stone_beam", ()),
                        *getattr(regime_beam, "exploit_beam", ()),
                    )
                    if item.candidate_id == bundle.selected_stone_id
                ),
                None,
            )

        for chk in passed_checks:
            res = l2_results.get(chk.candidate_id)
            if res is None:
                continue
            wkind, rule_obj, cand_hyp = meta_by_id.get(
                chk.candidate_id, ("exploit_rule", None, "")
            )
            cand_axes = outcome.cand_axes_by_cid.get(chk.candidate_id, bundle.parent_axes)
            cand_local = res.for_regime(regime_spec.regime_tag)
            local_res = cand_local if cand_local.anchors else res
            cand_lat = local_res.mean_latency_us()
            if cand_lat > 0.0:
                measured_by_cid[chk.candidate_id] = float(cand_lat)
            cand_sha = self._source_sha256(chk.patched_module_source)
            cand_msprof = self._msprof_cache.get((cand_sha, regime_spec.regime_tag))
            cand_obj = float(
                cand_local.surrogate_score if cand_local.anchors else res.surrogate_score
            )
            ledger = (
                chk.stage_a.ledger
                if (chk.stage_a is not None and chk.stage_a.ledger is not None)
                else bundle.parent_ledger
            )
            cand_comp_sum = chk.stage_b.compiler_summary if chk.stage_b is not None else None
            if chk.stage_a is not None and chk.stage_a.ledger is not None:
                cand_tel: dict[str, object] = {
                    "dur_us": cand_lat,
                    "measured_latency_us": cand_lat,
                    "roofline_eta": local_res.avg_roofline_eta,
                }
                if cand_msprof:
                    for mk, mv in cand_msprof.items():
                        if mk == "dur_us" and float(mv) <= 0.0:
                            continue
                        cand_tel[mk] = mv
                cand_res_obj = compute_static_dynamic_residual(
                    ledger,
                    cand_tel,
                    compiler_summary=cand_comp_sum,
                    roofline=bp_roofline,
                    topology_axes=cand_axes,
                )
            else:
                cand_res_obj = bundle.parent_residual
            cand_track = (
                chk.stage_a.route.track
                if self._enable_track_router
                and chk.stage_a is not None
                and chk.stage_a.route is not None
                else bundle.track
            )
            try:
                _, canonical_slice = extract_regime_snippet(
                    chk.patched_module_source,
                    regime_spec,
                    operator_entrypoint=op_entry,
                    sibling_specs=self._regime_specs,
                )
            except (SyntaxError, ValueError):
                canonical_slice = chk.snippet_source
            beam_updates.append(
                BeamCandidate(
                    candidate_id=chk.candidate_id, round_idx=round_idx, worker_kind=wkind,
                    rule_id=rule_obj.rule_id if rule_obj is not None else None,
                    dsl_source=canonical_slice, track=cand_track, ledger=ledger,
                    residual=cand_res_obj, latency_us=cand_lat,
                    accuracy_passed=local_res.all_passed,
                    structural_signature=extract_structural_signature(
                        canonical_slice, track=cand_track,
                        residual=cand_res_obj, stage_b=chk.stage_b,
                    ),
                    hypothesis=cand_hyp, parent_id=bundle.active_parent_id,
                    features=(
                        chk.stage_a.features
                        if (self._enable_track_router and chk.stage_a is not None) else None
                    ),
                    objective_score=cand_obj,
                    case_latencies_us={
                        int(a.case_num): float(a.cand_us)
                        for a in (res.anchors or local_res.anchors) if a.cand_us > 0.0
                    },
                    policy=dict(getattr(bundle.live_plan, "policy", {}) or {}),
                )
            )
            stone_parent_result = TelemetryLedgerWriter.build_stone_parent_result(
                operator_name=self._config.operator_name,
                parent_source=bundle.parent_module_source, incumbent_source=incumbent_source,
                best_result=best_result, cand_result=res, selected_stone=selected_stone,
            )
            self._record_surrogate_observations(
                source=chk.patched_module_source, result=res, active_anchors=active_anchors,
                parent_source=bundle.parent_module_source, parent_result=stone_parent_result,
                parent_ledger=bundle.parent_ledger, parent_bisheng_res=bundle.parent_l1_summary,
                parent_axes=bundle.parent_axes,
                ledger=chk.stage_a.ledger if chk.stage_a is not None else None,
                bisheng_summary=cand_comp_sum,
                stage_b_profile=(chk.stage_b.microarch_profile if chk.stage_b else None),
                axes=cand_axes, policy=getattr(bundle.live_plan, "policy", None),
                parent_policy=(
                    selected_stone.policy or None
                    if selected_stone is not None and bundle.parent_module_source != incumbent_source
                    else None
                ),
                worker_kind=wkind, rule_id=rule_obj.rule_id if rule_obj is not None else None,
                backend_track=cand_track, regime_tag=regime_spec.regime_tag,
            )
            if rule_obj is not None and l3_resolution != L3Resolution.INFRA_DEFERRED.value:
                cand_l3_res = (
                    l3_resolution if chk.candidate_id == winner_cid else L3Resolution.NOT_RUN.value
                )
                cand_rolled_back = bool(cand_l3_res == L3Resolution.CANDIDATE_REJECTED.value)
                cand_feats = chk.stage_a.features if chk.stage_a is not None else None
                causal_ok, speed_gain = self._evaluate_causal_rule_outcome(
                    rule=rule_obj, res_all_passed=res.all_passed, rolled_back=cand_rolled_back,
                    ledger=ledger, features=cand_feats, cand_residual=cand_res_obj,
                    cand_lat_us=cand_lat, cand_obj_score=cand_obj,
                    parent_ledger=bundle.parent_ledger, parent_residual=bundle.parent_residual,
                    parent_lat_us=parent_bn_lat_us, parent_obj_score=bundle.parent_obj_score,
                    has_cand_msprof=cand_msprof is not None,
                    cand_global_score=(
                        res.surrogate_score if bundle.selected_stone_id is None else None
                    ),
                    parent_global_score=(
                        best_result.surrogate_score if bundle.selected_stone_id is None else None
                    ),
                    cand_compiler_summary=cand_comp_sum, roofline=bp_roofline,
                )
                self._tracker.record_outcome(
                    rule_obj.rule_id, postcondition_realized=causal_ok and res.all_passed,
                    speed_gain=speed_gain, topology_axes=cand_axes,
                )
                self._telemetry_writer.record_kb_event(
                    rule_id=rule_obj.rule_id, postcondition_realized=causal_ok and res.all_passed,
                    speed_gain=speed_gain, topology_axes=cand_axes, round_idx=round_idx,
                )
                if causal_ok and res.all_passed:
                    self._tracker.reset_scoped_family_streak(
                        regime_tag=regime_spec.regime_tag, rule_id=rule_obj.rule_id,
                        parent_sha=bundle.parent_sha,
                    )
                    round_admitted.append(rule_obj.rule_id)
                    if rule_obj.rule_id not in admitted_rule_ids:
                        admitted_rule_ids.append(rule_obj.rule_id)
                else:
                    no_net_speedup = (
                        (not res.all_passed) or cand_rolled_back or parent_bn_lat_us <= 0.0
                        or cand_lat >= parent_bn_lat_us * 0.99
                    )
                    fail_reason = (
                        "l3_rolled_back" if cand_rolled_back else (
                            "l2_accuracy_failed" if not res.all_passed
                            else "l2_postcondition_unrealized_no_speedup"
                        )
                    )
                    if no_net_speedup:
                        self._tracker.record_scoped_failure(
                            regime_tag=regime_spec.regime_tag, rule_id=rule_obj.rule_id,
                            parent_sha=bundle.parent_sha, current_round=round_idx,
                            reason=fail_reason,
                        )
                    regressed_gt_3pct = bool(
                        parent_bn_lat_us > 0.0 and cand_lat > parent_bn_lat_us * 1.03
                    )
                    if self._experience_lake_dir is not None and (
                        not (causal_ok and res.all_passed) or regressed_gt_3pct
                    ):
                        pred_ctx = build_predicate_context(
                            ledger, features=cand_feats, residual=cand_res_obj,
                            compiler_summary=cand_comp_sum, roofline=bp_roofline,
                        )
                        lat_ratio = (
                            round(float(cand_lat) / float(parent_bn_lat_us), 6)
                            if (parent_bn_lat_us > 0.0 and cand_lat > 0.0) else 0.0
                        )
                        try:
                            append_rule_misapplication(
                                self._experience_lake_dir,
                                {
                                    "run_id": self._run_id, "operator": self._config.operator_name,
                                    "regime_tag": regime_spec.regime_tag,
                                    "round_idx": int(round_idx), "rule_id": rule_obj.rule_id,
                                    "symptom_family": rule_obj.symptom_family,
                                    "parent_sha": bundle.parent_sha, "cand_sha": cand_sha,
                                    "parent_latency_us": float(parent_bn_lat_us),
                                    "cand_latency_us": float(cand_lat), "latency_ratio": lat_ratio,
                                    "postcondition_realized": bool(causal_ok and res.all_passed),
                                    "l2_all_passed": bool(res.all_passed),
                                    "l3_resolution": cand_l3_res,
                                    "reason": (
                                        "latency_regression_gt_3pct"
                                        if regressed_gt_3pct else fail_reason
                                    ),
                                    "transfer_inflation_ratio": float(
                                        pred_ctx.get("transfer_inflation_ratio", 1.0)
                                    ),
                                    "effective_compute_eta": float(
                                        pred_ctx.get("effective_compute_eta", 0.0)
                                    ),
                                    "ub_occupancy_ratio": float(
                                        pred_ctx.get("ub_occupancy_ratio", 0.0)
                                    ),
                                    "parent_observation": (
                                        bundle.parent_residual.observation.to_dict()
                                        if bundle.parent_residual.observation is not None else None
                                    ),
                                    "cand_observation": (
                                        cand_res_obj.observation.to_dict()
                                        if cand_res_obj.observation is not None else None
                                    ),
                                },
                            )
                        except (OSError, ValueError) as exc:
                            logger.debug("Appending rule misapplication skipped: %s", exc)
            if chk.candidate_id == winner_cid:
                winner_rule_id = rule_obj.rule_id if rule_obj is not None else None
                winner_wkind = wkind
                winner_axes = cand_axes

        if beam_updates:
            regime_beam.update(beam_updates)
        if promoted and top_chk is not None:
            self._prune_stale_sibling_stepping_stones(regime_spec, top_chk.snippet_source)

        flywheel_sec = 0.0
        if promoted and top_chk is not None and top_res is not None:
            t_fw0 = time.monotonic()
            win_local = new_best.for_regime(regime_spec.regime_tag)
            best_local = best_result.for_regime(regime_spec.regime_tag)
            fly_base_lat = (
                best_local.mean_latency_us()
                if best_local.anchors
                else best_result.mean_latency_us()
            )
            win_lat = (
                win_local.mean_latency_us() if win_local.anchors else new_best.mean_latency_us()
            )
            fly_pts = [
                to_conservation_point(a, a)
                for a in (win_local.anchors if win_local.anchors else new_best.anchors)
            ]
            fly_sigma = (
                math.sqrt(sum(p.sigma_paired_us() ** 2 for p in fly_pts)) / max(1, len(fly_pts))
                if fly_pts else 0.0
            )
            eff_gain = max(rel_gain, local_rel_gain)
            if self._flywheel_coordinator.should_trigger(
                regime_spec.regime_tag, win_lat, fly_sigma
            ):
                distilled = self._run_flywheel_distillation(
                    baseline_source=incumbent_source, winner_source=new_incumbent,
                    baseline_latency_us=fly_base_lat, winner_latency_us=win_lat,
                    func_name=bundle.func_name, regime_spec=regime_spec,
                    active_anchors=active_anchors, topology_axes=winner_axes,
                    rel_gain=eff_gain, local_rel_gain=local_rel_gain,
                    worker_kind=winner_wkind, round_idx=round_idx, rule_id=winner_rule_id,
                    symptom_family=top_family, sigma_paired_us=fly_sigma,
                )
                for dr_id in distilled:
                    if dr_id not in round_admitted:
                        round_admitted.append(dr_id)
                    if dr_id not in admitted_rule_ids:
                        admitted_rule_ids.append(dr_id)
            elif (
                winner_wkind == "creative_explore"
                and eff_gain >= 0.05
                and self._experience_lake_dir is not None
            ):
                try:
                    append_winning_edge(
                        self._experience_lake_dir,
                        {
                            "run_id": self._run_id, "operator": self._config.operator_name,
                            "regime_tag": regime_spec.regime_tag, "round_idx": int(round_idx),
                            "worker_kind": winner_wkind, "rule_id": winner_rule_id,
                            "symptom_family": top_family,
                            "parent_sha": self._source_sha256(incumbent_source),
                            "winner_sha": self._source_sha256(new_incumbent),
                            "baseline_latency_us": float(fly_base_lat),
                            "winner_latency_us": float(win_lat), "relative_gain": float(eff_gain),
                            "loo_chunks": [], "staged_rule_ids": [],
                            "topology_axes": winner_axes.to_dict(),
                        },
                    )
                except (OSError, ValueError) as exc:
                    logger.debug("Appending explore winning edge skipped: %s", exc)
            flywheel_sec = max(0.0, time.monotonic() - t_fw0)

        trackb_evidence = self._collect_trackb_round_evidence(
            round_idx=round_idx, regime_spec=regime_spec, blueprint=bundle.blueprint,
            live_plan=bundle.live_plan, active_kernel=outcome.round_active_kernel,
            active_mask=bundle.active_mask or ConstraintRelaxationMask.bottom(),
            checks=passed_checks, l2_results=l2_results, l2_run_ids=outcome.l2_run_ids,
            parent_sha256=bundle.parent_sha, parent_l1_summary=bundle.parent_l1_summary,
            parent_l1_summary_source=bundle.parent_l1_summary_source,
        )
        round_spearman_rho = compute_spearman_rho(outcome.pred_by_cid, measured_by_cid)
        bucket_key = f"{regime_spec.regime_tag}:{bundle.parent_axes.key()}"
        self._boundary_ledger.record_spearman_rho(bucket_key, round_spearman_rho)
        diag_entry = self._telemetry_writer.ensure_regime_diag_entry(regime_spec.regime_tag)
        if not bundle.ranked_rules:
            diag_entry["rule_starvation_rounds"] = (
                int(diag_entry.get("rule_starvation_rounds", 0)) + 1
            )
        diag_entry["cost_prescreen_filtered_count"] = (
            int(diag_entry.get("cost_prescreen_filtered_count", 0))
            + len(outcome.prescreen_pruned_list)
        )
        if l3_resolution == L3Resolution.CANDIDATE_REJECTED.value:
            diag_entry["l3_rollback_count"] = int(diag_entry.get("l3_rollback_count", 0)) + 1
        elif not promoted and l2_results:
            diag_entry["l2_no_gain_rounds"] = int(diag_entry.get("l2_no_gain_rounds", 0)) + 1
        l0_reasons = diag_entry.setdefault("l0_reject_reasons", {})
        for chk in outcome.l0_l1_checks:
            if not chk.passed:
                tier_low = str(chk.failed_tier or "").lower()
                if "stage_b" in tier_low or "l1" in tier_low:
                    diag_entry["l1_compiler_penalty_rejects"] = (
                        int(diag_entry.get("l1_compiler_penalty_rejects", 0)) + 1
                    )
                elif isinstance(l0_reasons, dict):
                    _, fcode = _derive_candidate_fault_info(chk)
                    l0_reasons[fcode] = int(l0_reasons.get(fcode, 0)) + 1
        if bundle.ranked_families:
            diag_entry["top_symptom_families"] = [f.name for f, _ in bundle.ranked_families[:3]]
        pruned_rules = tuple(
            self._tracker.prune_underperforming(
                topology_axes=bundle.parent_axes, round_idx=round_idx
            )
        )
        self._tracker.decay_and_revive(
            round_idx=round_idx, min_cooldown_rounds=10, decay_factor=0.99
        )
        post_l3_sec = max(0.0, time.monotonic() - post_t0)
        stage_timings_sec = {
            "llm_gen_sec": round(max(0.0, float(outcome.llm_gen_sec)), 6),
            "l0_l1_sec": round(max(0.0, float(outcome.l0_l1_sec)), 6),
            "syntax_repair_sec": round(max(0.0, float(outcome.syntax_repair_sec)), 6),
            "l2_prewarm_sec": round(max(0.0, float(outcome.l2_prewarm_sec)), 6),
            "l2_npu_sec": round(max(0.0, float(outcome.l2_npu_sec)), 6),
            "l3_sec": round(max(0.0, float(l3_sec)), 6),
            "post_l3_sec": round(max(0.0, float(post_l3_sec)), 6),
        }
        cost_advisor_predictions = (
            (1 if bundle.parent_cost_pred is not None else 0) + len(outcome.pred_by_cid)
        )
        self._telemetry_writer.emit_round_funnel_events(
            round_idx=round_idx, regime_spec=regime_spec, l0_l1_checks=outcome.l0_l1_checks,
            failed_entries=outcome.failed_entries, meta_by_id=meta_by_id, l2_results=l2_results,
            l0_l1_faults_by_cid=outcome.l0_l1_faults_by_cid,
            cand_axes_by_cid=outcome.cand_axes_by_cid,
            parent_axes=bundle.parent_axes, top_chk=top_chk, l2_infra_fault=outcome.l2_infra_fault,
            initial_top_cid=initial_top_cid, promoted=promoted, l3_resolution=l3_resolution,
            l3_failure_stage=l3_failure_stage, stagnation_after=stagnation_after,
            admitted_rule_ids=round_admitted, local_best=local_best,
            parent_snippet=bundle.parent_snippet, ub_bytes_by_cid=outcome.ub_bytes_by_cid,
            knob_sig_by_cid=outcome.knob_sig_by_cid, pred_by_cid=outcome.pred_by_cid,
            measured_by_cid=measured_by_cid, stage_timings_sec=stage_timings_sec,
            prewarm_stats=outcome.round_prewarm_stats,
            cost_advisor_predictions=cost_advisor_predictions,
        )
        if self._experience_lake_dir is not None and (
            outcome.l0_l1_checks or outcome.failed_entries
        ):
            self._telemetry_writer.record_kb_event(
                rule_id=winner_rule_id or "__round__", postcondition_realized=promoted,
                speed_gain=local_rel_gain if promoted else 0.0,
                topology_axes=bundle.parent_axes, round_idx=round_idx,
            )
        paradigm_shift_rule_ids = tuple(
            str(r.rule_id) for r in bundle.ranked_rules if getattr(r, "paradigm_shift", False)
        )
        record = CampaignRoundRecord(
            round_idx=round_idx, regime_tag=regime_spec.regime_tag, builder_name=bundle.func_name,
            parent_lane=bundle.parent_lane, creative_effort=creative_effort,
            l0_l1_passed_ids=tuple(c.candidate_id for c in passed_checks),
            l0_l1_failed=tuple(outcome.failed_entries), best_round_speedup=best_round_speedup,
            best_so_far_speedup=new_best.avg_speedup, promoted=promoted,
            l3_resolution=l3_resolution, stagnation_rounds_after=stagnation_after,
            admitted_rule_ids=tuple(round_admitted),
            best_round_surrogate_score=best_round_surrogate_score,
            best_so_far_surrogate_score=new_best.surrogate_score, pruned_rule_ids=pruned_rules,
            promotion_attempts=promotion_attempts, trackb_evidence=trackb_evidence,
            cost_model_mode=str(bundle.gate_decision.mode),
            cost_model_engaged=bool(bundle.gate_decision.engaged),
            cost_model_reason=str(bundle.gate_decision.reason),
            local_rel_gain=float(local_rel_gain), tail_guard_reject_reason=tail_guard_reject_reason,
            paradigm_shift_active=bool(bundle.paradigm_shift_active),
            paradigm_shift_rule_ids=paradigm_shift_rule_ids,
            round_spearman_rho=round_spearman_rho,
            prescreen_pruned_ids=tuple(outcome.prescreen_pruned_list),
            l3_failed_cases=l3_failed_cases, l3_failure_stage=l3_failure_stage,
            infeasible_boundaries_added=outcome.infeasible_boundaries_added,
            stage_timings_sec=stage_timings_sec, prewarm_stats=dict(outcome.round_prewarm_stats),
            cost_advisor_predictions=cost_advisor_predictions,
        )
        return (
            record, new_incumbent, new_best, stagnation_after, promoted,
            l3_resolution == L3Resolution.CANDIDATE_REJECTED.value,
            bundle.escalated, promotion_blocked,
        )

    async def run(self, *, resume: bool = False) -> CampaignReport:
        """Execute bounded multi-round convergence campaign with checkpointing and final L4."""
        self._sync_coordinators()
        campaign_start = self._clock()
        budget_sec = self._config.max_wall_clock_min * 60.0

        frozen_regimes: set[str] = set()
        current_regime: str | None = None
        last_unpromoted_regime: str | None = None
        rounds: list[CampaignRoundRecord] = []
        promoted_rounds: list[int] = []
        rolled_back_rounds: list[int] = []
        escalated_rounds: list[int] = []
        admitted_rule_ids: list[str] = []
        stagnation_by_regime: dict[str, int] = {}
        stagnation_rounds = 0
        pending_promotion: PendingPromotion | None = None
        pending_l4: PendingL4 | None = None
        l4_result: L4CannbenchResult | None = None

        if resume and self._checkpoint_path is not None and self._checkpoint_path.exists():
            (
                incumbent_source,
                baseline_result,
                best_result,
                active_anchors,
                frozen_regimes,
                current_regime,
                stagnation_rounds,
                rounds,
                promoted_rounds,
                rolled_back_rounds,
                escalated_rounds,
                admitted_rule_ids,
                stagnation_by_regime,
                last_unpromoted_regime,
                pending_promotion,
                pending_l4,
            ) = self._restore_from_checkpoint()
        else:
            if not self._target_module_path.exists():
                raise FileNotFoundError(
                    f"Target module does not exist: {self._target_module_path}"
                )
            incumbent_source = self._target_module_path.read_text(encoding="utf-8")
            active_anchors = list(self._scoreboard_anchors)
            snapshot = self._calibration_service.build_snapshot(
                incumbent_source,
                active_anchors,
                self._regime_specs,
            )
            baseline_result = snapshot.baseline_result
            self._null_launch_us = snapshot.null_launch_us
            self._calibrated_dispatch_floor_us = snapshot.calibrated_dispatch_floor_us
            self._msprof_by_case.update(snapshot.msprof_by_case)
            sha_base = self._source_sha256(incumbent_source)
            for (_csha, cnum), metrics in snapshot.msprof_by_case.items():
                for anc in active_anchors:
                    if int(anc.case_num) == int(cnum):
                        self._msprof_cache[(sha_base, anc.regime_tag)] = dict(metrics)
            self._active_incumbent_source = incumbent_source
            self._incumbent_anchor_us_by_case = {
                int(a.case_num): float(a.cand_us)
                for a in baseline_result.anchors
                if a.cand_us > 0.0
            }
            if not baseline_result.all_passed:
                msg = baseline_result.error_message
                raise RuntimeError(f"Baseline evaluation failed L2 surrogate gates: {msg}")
            if self._regime_specs:
                matched_any_regime = any(
                    baseline_result.for_regime(r.regime_tag).anchors for r in self._regime_specs
                )
                for idx, rspec in enumerate(self._regime_specs):
                    base_sa = None
                    try:
                        base_sa = self._run_l0(incumbent_source, rspec)
                    except (SyntaxError, RuntimeError, ValueError, TypeError, OSError):
                        base_sa = None
                    base_l1 = None
                    try:
                        base_l1 = self._l1_summary_for_first_round(incumbent_source, rspec)
                    except (SyntaxError, RuntimeError, ValueError, TypeError, OSError):
                        base_l1 = None
                    base_ledger = base_sa.ledger if base_sa is not None else None
                    base_axes = self._stage_gate_coordinator._derive_topology_axes(
                        base_ledger, features=base_sa.features if base_sa is not None else None,
                        route=base_sa.route if base_sa is not None else None,
                    )
                    regime_res = baseline_result.for_regime(rspec.regime_tag)
                    if not regime_res.anchors and (
                        len(self._regime_specs) == 1 or (not matched_any_regime and idx == 0)
                    ):
                        regime_res = baseline_result
                    self._record_surrogate_observations(
                        source=incumbent_source, result=regime_res, active_anchors=active_anchors,
                        eval_tier_override="baseline", ledger=base_ledger, bisheng_summary=base_l1,
                        axes=base_axes, worker_kind="baseline",
                        backend_track=(
                            base_sa.route.track
                            if self._enable_track_router
                            and base_sa is not None and base_sa.route is not None
                            else "tilelang_dsl"
                        ),
                        regime_tag=rspec.regime_tag,
                    )
            else:
                self._record_surrogate_observations(
                    source=incumbent_source, result=baseline_result, active_anchors=active_anchors,
                    eval_tier_override="baseline", ledger=None, axes=None,
                    worker_kind="baseline", backend_track="tilelang_dsl",
                )
            best_result = baseline_result
            if not self._calibrated_dispatch_floor_us:
                self._calibrated_dispatch_floor_us = calibrate_dispatch_floor_us(
                    [to_conservation_point(a, a) for a in baseline_result.anchors]
                )
            for rspec in self._regime_specs:
                reg_base = baseline_result.for_regime(rspec.regime_tag)
                reg_pts = [
                    to_conservation_point(a, a)
                    for a in (reg_base.anchors if reg_base.anchors else baseline_result.anchors)
                ]
                reg_lat = (
                    reg_base.mean_latency_us()
                    if reg_base.anchors else baseline_result.mean_latency_us()
                )
                reg_sigma = (
                    math.sqrt(sum(p.sigma_paired_us() ** 2 for p in reg_pts)) / max(1, len(reg_pts))
                    if reg_pts else 0.0
                )
                self._flywheel_coordinator.seed_baseline(
                    rspec.regime_tag, incumbent_source, reg_lat, reg_sigma
                )
            incumbent_source, best_result = self._run_phase0_lattice_warmstart(
                incumbent_source, best_result, active_anchors
            )
            self._compile_wall_streak_by_regime = {}
            self._save_checkpoint(
                incumbent_source=incumbent_source, baseline_result=baseline_result,
                best_result=best_result, active_anchors=active_anchors,
                frozen_regimes=frozen_regimes, current_regime=current_regime,
                stagnation_rounds=stagnation_rounds, rounds=rounds,
                promoted_rounds=promoted_rounds, rolled_back_rounds=rolled_back_rounds,
                escalated_rounds=escalated_rounds, admitted_rule_ids=admitted_rule_ids,
                stop_reason="in_progress", l4_result=None,
                stagnation_by_regime=stagnation_by_regime,
                last_unpromoted_regime=last_unpromoted_regime,
            )

        self._active_anchors = list(active_anchors)
        self._active_incumbent_source = incumbent_source
        self._incumbent_anchor_us_by_case = {
            int(a.case_num): float(a.cand_us)
            for a in best_result.anchors
            if a.cand_us > 0.0
        }
        self._stage_gate_coordinator._incumbent_anchor_us_by_case = (
            self._incumbent_anchor_us_by_case
        )
        self._pending_promotion = pending_promotion
        self._pending_l4 = pending_l4
        stop_reason = ""
        promotion_blocked = False
        l4_was_resolved = False

        if pending_promotion is not None:
            retried_pending = pending_promotion
            self._pending_promotion = retried_pending
            (
                incumbent_source,
                best_result,
                stagnation_by_regime,
                pending_promotion,
                promotion_blocked,
            ) = self._resolve_pending_promotion(
                retried_pending,
                incumbent_source=incumbent_source,
                best_result=best_result,
                stagnation_by_regime=stagnation_by_regime,
                rounds=rounds,
                promoted_rounds=promoted_rounds,
                rolled_back_rounds=rolled_back_rounds,
                active_anchors=active_anchors,
            )
            stagnation_rounds = stagnation_by_regime.get(retried_pending.regime_tag, 0)
            if promotion_blocked:
                stop_reason = "promotion_blocked"
            elif pending_promotion is None and retried_pending.round_idx in promoted_rounds:
                current_regime = retried_pending.regime_tag
                last_unpromoted_regime = None

        if pending_l4 is not None and not promotion_blocked:
            prior_stop_reason = pending_l4.prior_stop_reason
            l4_result, pending_l4 = self._resolve_pending_l4(
                pending_l4,
                incumbent_source=incumbent_source,
            )
            self._pending_l4 = pending_l4
            l4_was_resolved = True
            if not stop_reason:
                stop_reason = prior_stop_reason or "l4_pending"
            self._record_l4_observations(
                source=incumbent_source,
                l4_result=l4_result,
                best_result=best_result,
                active_anchors=active_anchors,
            )
        elif pending_l4 is not None:
            l4_was_resolved = True

        if not stop_reason and self._allow_early_stop(len(rounds)):
            if best_result.meets_perf_target(self._perf_target):
                stop_reason = "goal_met"
            elif best_result.is_roofline_saturated(self._perf_target):
                stop_reason = "roofline_saturated"

        self._active_incumbent_source = incumbent_source
        self._incumbent_anchor_us_by_case = {
            int(a.case_num): float(a.cand_us)
            for a in best_result.anchors
            if a.cand_us > 0.0
        }
        self._stagnation_by_regime = stagnation_by_regime
        compile_wall_streak_by_regime = self._compile_wall_streak_by_regime
        start_round = (rounds[-1].round_idx + 1) if rounds else 1
        try:
            for round_idx in range(start_round, max(0, self._config.max_rounds) + 1):
                if stop_reason:
                    break
                if (self._clock() - campaign_start) >= budget_sec:
                    stop_reason = "budget_exhausted"
                    break
                active_unfrozen = [
                    spec.regime_tag
                    for spec in self._regime_specs
                    if spec.regime_tag not in frozen_regimes
                ]
                for spec in self._regime_specs:
                    if (
                        stagnation_by_regime.get(spec.regime_tag, 0)
                        >= self._config.stagnation_freeze_rounds
                    ):
                        frozen_regimes.add(spec.regime_tag)
                    elif (
                        len(active_unfrozen) > 1
                        and compile_wall_streak_by_regime.get(spec.regime_tag, 0) >= 3
                    ):
                        frozen_regimes.add(spec.regime_tag)
                if current_regime and current_regime in frozen_regimes:
                    current_regime = None

                forced_regime = self._round_regime_override(round_idx)
                active_regime = forced_regime
                if active_regime is None:
                    wall_yield_regime = next(
                        (
                            spec.regime_tag
                            for spec in self._regime_specs
                            if compile_wall_streak_by_regime.get(spec.regime_tag, 0) >= 2
                            and spec.regime_tag not in frozen_regimes
                        ),
                        None,
                    )
                    active_regime = self._select_active_regime(
                        current_regime,
                        frozen_regimes,
                        best_result,
                        active_anchors,
                        allow_satisfied_fallback=not self._allow_early_stop(len(rounds)),
                        exclude_regime=last_unpromoted_regime or wall_yield_regime,
                        stagnation_by_regime=stagnation_by_regime,
                    )
                if (
                    active_regime is None
                    and forced_regime is None
                    and last_unpromoted_regime is not None
                ):
                    active_regime = self._select_active_regime(
                        current_regime,
                        frozen_regimes,
                        best_result,
                        active_anchors,
                        allow_satisfied_fallback=not self._allow_early_stop(len(rounds)),
                        stagnation_by_regime=stagnation_by_regime,
                    )
                if active_regime is None:
                    stop_reason = "all_regimes_converged"
                    break
                active_tag = active_regime.regime_tag
                current_regime = active_tag
                self._active_regime_tag = active_tag
                (
                    rec,
                    incumbent_source,
                    best_result,
                    stagnation_rounds,
                    promoted,
                    rolled_back,
                    escalated,
                    promotion_blocked,
                ) = await self._execute_single_round(
                    round_idx=round_idx,
                    regime_spec=active_regime,
                    incumbent_source=incumbent_source,
                    best_result=best_result,
                    active_anchors=active_anchors,
                    stagnation_rounds=stagnation_by_regime.get(active_tag, 0),
                    admitted_rule_ids=admitted_rule_ids,
                )
                rounds.append(rec)
                pending_promotion = self._pending_promotion
                self._last_round_promoted = bool(promoted)
                self._last_round_promoted_by_regime[active_tag] = bool(promoted)
                all_noop = bool(rec.l0_l1_failed) and all(
                    stage == "noop_identical" or "noop_identical" in err
                    for _cid, stage, err in rec.l0_l1_failed
                )
                is_infra_deferred = rec.l3_resolution == L3Resolution.INFRA_DEFERRED.value or (
                    bool(rec.l0_l1_failed)
                    and all(
                        stage in ("l2_infra_occupancy", "l2_infra_device_fault")
                        for _cid, stage, _err in rec.l0_l1_failed
                    )
                )
                if promoted:
                    stagnation_rounds = 0
                    stagnation_by_regime[active_tag] = 0
                    compile_wall_streak_by_regime[active_tag] = 0
                    current_regime = active_tag
                    last_unpromoted_regime = None
                    promoted_rounds.append(round_idx)
                elif is_infra_deferred:
                    stagnation_by_regime[active_tag] = stagnation_by_regime.get(active_tag, 0)
                    current_regime = None
                    last_unpromoted_regime = active_tag
                else:
                    stagnation_by_regime[active_tag] = stagnation_rounds
                    if len(rec.l0_l1_passed_ids) == 0 or all_noop:
                        compile_wall_streak_by_regime[active_tag] = (
                            compile_wall_streak_by_regime.get(active_tag, 0) + 1
                        )
                    else:
                        compile_wall_streak_by_regime[active_tag] = 0
                    current_regime = None
                    last_unpromoted_regime = active_tag
                other_unfrozen = any(
                    s.regime_tag != active_tag and s.regime_tag not in frozen_regimes
                    for s in self._regime_specs
                )
                if other_unfrozen and compile_wall_streak_by_regime.get(active_tag, 0) >= 3:
                    frozen_regimes.add(active_tag)
                if rolled_back:
                    rolled_back_rounds.append(round_idx)
                if escalated:
                    escalated_rounds.append(round_idx)
                if promotion_blocked:
                    stop_reason = "promotion_blocked"
                early_stop_allowed = self._allow_early_stop(len(rounds))
                if not promotion_blocked:
                    if early_stop_allowed and best_result.meets_perf_target(self._perf_target):
                        stop_reason = "goal_met"
                    elif (
                        early_stop_allowed
                        and best_result.is_roofline_saturated(self._perf_target)
                    ):
                        stop_reason = "roofline_saturated"
                    elif (
                        stagnation_by_regime.get(active_tag, 0)
                        >= self._config.stagnation_freeze_rounds
                        or (
                            other_unfrozen
                            and compile_wall_streak_by_regime.get(active_tag, 0) >= 3
                        )
                    ):
                        frozen_regimes.add(active_tag)
                        next_reg = self._select_active_regime(
                            None,
                            frozen_regimes,
                            best_result,
                            active_anchors,
                            allow_satisfied_fallback=not early_stop_allowed,
                            exclude_regime=last_unpromoted_regime,
                            stagnation_by_regime=stagnation_by_regime,
                        )
                        if next_reg is None:
                            stop_reason = "all_regimes_converged"
                self._stagnation_by_regime = stagnation_by_regime
                self._compile_wall_streak_by_regime = compile_wall_streak_by_regime
                self._save_checkpoint(
                    incumbent_source=incumbent_source, baseline_result=baseline_result,
                    best_result=best_result, active_anchors=active_anchors,
                    frozen_regimes=frozen_regimes, current_regime=current_regime,
                    stagnation_rounds=stagnation_rounds, rounds=rounds,
                    promoted_rounds=promoted_rounds, rolled_back_rounds=rolled_back_rounds,
                    escalated_rounds=escalated_rounds, admitted_rule_ids=admitted_rule_ids,
                    stop_reason=stop_reason or "in_progress", l4_result=None,
                    stagnation_by_regime=stagnation_by_regime,
                    last_unpromoted_regime=last_unpromoted_regime,
                )
                if stop_reason:
                    break
        finally:
            await self._stage_gate_coordinator.cancel_pending_speculative()

        if not stop_reason:
            stop_reason = "budget_exhausted"

        if self._config.run_final_l4 and not promotion_blocked and not l4_was_resolved:
            l4_result = self._run_final_l4_tracked(
                incumbent_source=incumbent_source, prior_stop_reason=stop_reason
            )
            pending_l4 = self._pending_l4
            if pending_l4 is not None and not stop_reason:
                stop_reason = "l4_pending"
            self._record_l4_observations(
                source=incumbent_source, l4_result=l4_result,
                best_result=best_result, active_anchors=active_anchors,
            )

        self._save_checkpoint(
            incumbent_source=incumbent_source, baseline_result=baseline_result,
            best_result=best_result, active_anchors=active_anchors,
            frozen_regimes=frozen_regimes, current_regime=current_regime,
            stagnation_rounds=stagnation_rounds, rounds=rounds,
            promoted_rounds=promoted_rounds, rolled_back_rounds=rolled_back_rounds,
            escalated_rounds=escalated_rounds, admitted_rule_ids=admitted_rule_ids,
            stop_reason=stop_reason, l4_result=l4_result,
            stagnation_by_regime=stagnation_by_regime,
            last_unpromoted_regime=last_unpromoted_regime,
        )

        stagnation_diagnosis = self._build_stagnation_diagnosis(
            best_result=best_result,
            stop_reason=stop_reason,
        )
        l4_status = "not_run"
        if l4_result is not None:
            l4_err = l4_result.error_message or ""
            if l4_result.passed:
                l4_status = "passed"
            elif pending_l4 is not None:
                l4_status = "pending_infrastructure"
            elif not promotion_blocked and (
                "BLOCKED_ENV" in l4_err
                or "BLOCKED_EXIT_CODE: 75" in l4_err
                or "occupancy" in l4_err.lower()
            ):
                l4_status = "deferred_shared_host"
            else:
                l4_status = "failed"
        return CampaignReport(
            operator_name=self._config.operator_name, stop_reason=stop_reason,
            total_rounds=len(rounds), baseline_result=baseline_result, best_result=best_result,
            rounds=tuple(rounds), promoted_rounds=tuple(promoted_rounds),
            rolled_back_rounds=tuple(rolled_back_rounds),
            escalated_rounds=tuple(escalated_rounds),
            frozen_regimes=tuple(sorted(frozen_regimes)),
            active_anchors=tuple(self._scoreboard_anchors),
            scoreboard_anchor_count=len(self._scoreboard_anchors),
            scoreboard_anchors=tuple(self._scoreboard_anchors),
            admitted_rule_ids=tuple(admitted_rule_ids), l4_result=l4_result, l4_status=l4_status,
            pending_promotion=pending_promotion, pending_l4=pending_l4,
            promotion_blocked=promotion_blocked or pending_promotion is not None,
            pure_golden_mode=bool(self._config.pure_golden),
            blueprint_provenance=dict(self._blueprint_provenance),
            stagnation_diagnosis=stagnation_diagnosis, real_npu_path=self._is_real_npu_path(),
            calibrated_dispatch_floor_us=self._calibrated_dispatch_floor_us,
            geometric_speedup=geometric_speedup(
                [to_conservation_point(a, a) for a in best_result.anchors]
            ),
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI parser for PerfTargetCampaignRunner."""
    parser = argparse.ArgumentParser(
        prog="tilefoundry-ascend-campaign-runner",
        description="Run unattended performance optimization campaigns.",
    )
    parser.add_argument(
        "--operator",
        required=True,
        help="Target operator (short name e.g. rms_norm or task id e.g. level2/rms_norm).",
    )
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--max-wall-clock-min", type=float, default=180.0)
    parser.add_argument("--device-id", default="", help="Authorized primary NPU device ID.")
    parser.add_argument(
        "--device-ids",
        type=parse_device_ids,
        default=(),
        help="Extra authorized NPU cards for whole-L4 attempts, comma-separated.",
    )
    parser.add_argument(
        "--retry-device-id",
        default="",
        help="Alternate NPU card for single fresh retry after infrastructure failure.",
    )
    parser.add_argument("--disable-early-goal-stop", action="store_true")
    parser.add_argument("--min-surrogate-gain", type=float, default=0.25)
    parser.add_argument("--checkpoint", default=None, help="Atomic checkpoint JSON path.")
    parser.add_argument("--report-json", default=None, help="Optional final report JSON path.")
    parser.add_argument("--run-final-l4", action="store_true", help="Run final official L4 gate.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint.")
    parser.add_argument("--k-rule-workers", type=int, default=2)
    parser.add_argument("--target-module", default=None, help="Override path to kernels module.")
    parser.add_argument("--source-root", default=None, help="TileFoundry workspace root.")
    parser.add_argument("--env-file", default=None, help="Path to tilefoundry-npu env.sh.")
    parser.add_argument(
        "--enable-tilesize-prescreen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable tile-size prescreen guidance table in creative prompts.",
    )
    parser.add_argument("--cost-prescreen-top-k", type=int, default=None)
    parser.add_argument(
        "--cost-model-mode",
        choices=("advisory", "shadow", "off"),
        default="off",
        help="Cost model advisory gate policy mode.",
    )
    parser.add_argument("--enable-phase0-lattice-sweep", action="store_true")
    parser.add_argument("--pure-golden", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute PerfTargetCampaignRunner from CLI arguments and emit JSON summary."""
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    env_hint = args.env_file or os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()
    env_values = parse_simple_env_file(Path(env_hint).expanduser()) if env_hint else {}
    primary_device_id, device_pool = resolve_campaign_device_plan(
        device_id=str(args.device_id),
        device_ids=tuple(args.device_ids),
        env_device_ids=parse_device_ids(env_values.get(CAMPAIGN_DEVICE_IDS_ENV_KEY, "")),
        visible_device_ids=parse_device_ids(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")),
    )
    llm_backend = create_llm_backend_from_env(overrides=env_values or None)
    cfg = CampaignConfig(
        operator_name=args.operator,
        max_rounds=args.max_rounds,
        max_wall_clock_min=args.max_wall_clock_min,
        k_rule_workers=args.k_rule_workers,
        min_surrogate_gain=args.min_surrogate_gain,
        disable_early_goal_stop=bool(args.disable_early_goal_stop),
        run_final_l4=bool(args.run_final_l4),
        device_id=primary_device_id,
        retry_device_id=str(args.retry_device_id),
        device_ids=tuple(d for d in device_pool if d != primary_device_id),
        enable_tilesize_prescreen=bool(args.enable_tilesize_prescreen),
        cost_prescreen_top_k=(
            int(args.cost_prescreen_top_k) if args.cost_prescreen_top_k is not None else None
        ),
        cost_model_mode=str(args.cost_model_mode),
        enable_phase0_lattice_sweep=bool(args.enable_phase0_lattice_sweep),
        pure_golden=bool(args.pure_golden),
    )
    tracker = BayesianRuleTracker.from_seed_yaml()
    flywheel = FlywheelDistiller(
        funnel=GateFunnel(skip_stage_b=True),
        tracker=tracker,
    )
    runner = PerfTargetCampaignRunner(
        config=cfg,
        target_module_path=args.target_module,
        llm_backend=llm_backend,
        tracker=tracker,
        flywheel_distiller=flywheel,
        checkpoint_path=args.checkpoint,
        source_root=args.source_root,
        env_file=args.env_file,
    )
    report = asyncio.run(runner.run(resume=bool(args.resume)))
    rendered = json.dumps(report.to_dict(), indent=2)
    if args.report_json:
        _atomic_write_text(Path(args.report_json), rendered + "\n")
    print(rendered)
    return 0


__all__ = [
    "CAMPAIGN_DEVICE_IDS_ENV_KEY",
    "CalibrationService",
    "CalibrationSnapshot",
    "CampaignConfig",
    "CampaignReport",
    "CampaignRoundRecord",
    "L3L4RecoveryCoordinator",
    "PendingL4",
    "PendingPromotion",
    "PerfTargetCampaignRunner",
    "ResolvedRegimeConfig",
    "RoundContextBuilder",
    "StageGateCoordinator",
    "TelemetryLedgerWriter",
    "apply_ast_patch",
    "build_arg_parser",
    "extract_regime_snippet",
    "main",
    "parse_device_ids",
    "patch_regime_into_module",
    "rank_stage_a_candidates_with_cost_model",
    "resolve_campaign_device_plan",
    "resolve_regime_config",
    "resolve_regime_for_case",
    "run_l1_compile_check",
    "split_readonly_host_tail",
]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
