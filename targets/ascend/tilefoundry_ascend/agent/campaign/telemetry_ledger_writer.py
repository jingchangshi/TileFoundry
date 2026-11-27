"""Telemetry, ExperienceLake, checkpoint, and diagnosis persistence for campaigns."""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from tilefoundry_ascend.agent.campaign.flywheel_coordinator import FlywheelCoordinator
from tilefoundry_ascend.agent.campaign.funnel_telemetry import (
    ANOMALY_LOG_ERROR_THRESHOLD, DeterministicSearchBoundaryLedger, RoundFunnelEvent,
    append_round_funnel_event, is_clean_calibration_sample,
)
from tilefoundry_ascend.agent.campaign.helpers import (
    derive_topology_axes, ensure_candidate_bytes as _helpers_ensure_candidate_bytes,
    lookup_case_dims_safe, persist_round_candidate_artifact as _helpers_persist_candidate_artifact,
    persist_runtime_state as _helpers_persist_runtime_state,
    source_sha256, write_source_and_verify as _helpers_write_source_and_verify,
)
from tilefoundry_ascend.agent.campaign.l3_l4_recovery import PendingL4
from tilefoundry_ascend.agent.campaign.promotion_coordinator import PendingPromotion
from tilefoundry_ascend.agent.contracts import CampaignRoundRecord, CampaignStateSnapshot
from tilefoundry_ascend.agent.cost_advisor import CostModelAdvisor
from tilefoundry_ascend.agent.flywheel.ablation_distiller import FlywheelDistiller
from tilefoundry_ascend.agent.gates.gate_funnel import StageAResult, extract_ordered_tensor_shapes
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker, append_kb_event_shard
from tilefoundry_ascend.agent.kb.schema import AscendCausalRule, build_predicate_context, load_seed_rules
from tilefoundry_ascend.agent.npu import (
    AnchorEvalMetrics, CandidateSurrogateResult, L0L1CandidateCheck, L4CannbenchResult,
    aggregate_surrogate_result, bottleneck_anchor_for_regime,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    StaticDynamicResidual, TwoLaneBeamSearch, compiler_summary_from_residual,
)
from tilefoundry_ascend.agent.specs.anchor_specs import (
    AnchorCaseSpec, GuardrailCaseSpec, PerfTargetSpec, RegimeTargetSpec,
    compute_case_eq3_score, lookup_case_metadata,
)
from tilefoundry_ascend.agent.specs.fault_types import FaultPlane, L3Resolution, TypedFault, coerce_fault_plane
from tilefoundry_ascend.ascend.cost_model import BishengResourceSummary
from tilefoundry_ascend.ascend.cost_model.experience_lake import (
    CandidateArtifactRecord, CaseObservationRecord, ExperienceLake, compute_source_sha256, project_case_workload,
)
from tilefoundry_ascend.ascend.cost_model.microarch_profile import merge_microarch_profiles
from tilefoundry_ascend.ascend.json_record import JsonRecordMixin, atomic_write_text
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import HirBlueprintPacket, HirRooflineSummary
from tilefoundry_ascend.ascend.passes.observation_space import compute_transfer_inflation_ratio
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger
from tilefoundry_ascend.ascend.schedule.regime_solver import RegimePlanResult
from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask, TopologyAxes
from tilefoundry_ascend.ascend.target import AscendTarget

logger = logging.getLogger(__name__)

_STAGE_TO_PLANE: dict[str, FaultPlane] = {
    "l2_infra_occupancy": FaultPlane.INFRA_OCCUPANCY, "infra_occupancy": FaultPlane.INFRA_OCCUPANCY,
    "l2_infra_device_fault": FaultPlane.INFRA_DEVICE_FAULT, "infra_device_fault": FaultPlane.INFRA_DEVICE_FAULT,
    "infra_toolchain": FaultPlane.INFRA_DEVICE_FAULT, "llm_transport": FaultPlane.LLM_TRANSPORT,
    "noop_identical": FaultPlane.NOOP_IDENTICAL, "syntax": FaultPlane.SYNTAX_AST,
    "syntax_ast": FaultPlane.SYNTAX_AST, "syntax_repair": FaultPlane.SYNTAX_AST, "ast": FaultPlane.SYNTAX_AST,
    "stage_b": FaultPlane.COMPILER_LOWERING, "stage_b_compile": FaultPlane.COMPILER_LOWERING,
    "l1": FaultPlane.COMPILER_LOWERING, "l1_compile": FaultPlane.COMPILER_LOWERING,
    "compiler_lowering": FaultPlane.COMPILER_LOWERING, "stage_a": FaultPlane.STATIC_CONTRACT,
    "stage_a_static": FaultPlane.STATIC_CONTRACT, "l0": FaultPlane.STATIC_CONTRACT,
    "l0_l1": FaultPlane.STATIC_CONTRACT, "l0_static": FaultPlane.STATIC_CONTRACT,
    "cost_prescreen": FaultPlane.STATIC_CONTRACT, "static_contract": FaultPlane.STATIC_CONTRACT,
    "l2_runtime": FaultPlane.CANDIDATE_RUNTIME, "candidate_runtime": FaultPlane.CANDIDATE_RUNTIME,
    "l2_numeric": FaultPlane.CANDIDATE_NUMERIC, "l2_accuracy": FaultPlane.CANDIDATE_NUMERIC,
    "l2_error": FaultPlane.CANDIDATE_NUMERIC, "candidate_numeric": FaultPlane.CANDIDATE_NUMERIC,
}


def _resolve_fault_plane(raw_plane: object, *, failed_tier: str = "", detail: str = "") -> FaultPlane:
    if isinstance(raw_plane, FaultPlane):
        return raw_plane
    raw_key, tier_key, low = str(raw_plane or "").strip().lower(), failed_tier.strip().lower(), detail.lower()
    if raw_key in _STAGE_TO_PLANE:
        return _STAGE_TO_PLANE[raw_key]
    coerced = coerce_fault_plane(raw_plane)
    if coerced not in (FaultPlane.NONE, FaultPlane.UNMODELED_FAULT):
        return coerced
    if tier_key in _STAGE_TO_PLANE:
        return _STAGE_TO_PLANE[tier_key]
    if "occupancy" in tier_key or "occupancy" in raw_key or "occupancy" in low:
        return FaultPlane.INFRA_OCCUPANCY
    if "device_fault" in tier_key or "tsd" in low:
        return FaultPlane.INFRA_DEVICE_FAULT
    if coerced == FaultPlane.NONE and (raw_key or tier_key):
        return FaultPlane.UNMODELED_FAULT
    return coerced


@dataclass(frozen=True)
class RoundReceipt(JsonRecordMixin):
    """Immutable telemetry and outcome receipt emitted at the end of a round."""

    round_idx: int
    regime_tag: str
    promoted: bool
    l3_resolution: str
    stagnation_after: int
    admitted_rule_ids: tuple[str, ...] = ()
    stage_timings_sec: dict[str, float] = field(default_factory=dict)
    prewarm_stats: dict[str, int] = field(default_factory=dict)
    cost_advisor_predictions: int = 0
    emitted_event_count: int = 0

    def to_dict(self) -> dict[str, object]:
        """Serialize receipt into a JSON-compatible dictionary."""
        out = super().to_dict()
        out["stage_timings_sec"] = {str(k): round(float(v), 6) for k, v in self.stage_timings_sec.items()}
        out["prewarm_stats"] = {str(k): int(v) for k, v in self.prewarm_stats.items()}
        return out


class TelemetryLedgerWriter:
    """Coordinate ExperienceLake writes, funnel telemetry, flywheel, and checkpoints."""

    def __init__(
        self, *, operator_name: str = "", run_id: str = "", device_id: str = "1",
        allow_unreviewed_distilled_rules: bool = False, target_module_path: Path | None = None,
        checkpoint_path: Path | None = None, campaign_kb_path: Path | None = None,
        experience_lake_dir: Path | None = None, experience_lake: ExperienceLake | None = None,
        cost_advisor: CostModelAdvisor | None = None, target: AscendTarget | None = None,
        perf_target: PerfTargetSpec | None = None, regime_specs: Sequence[RegimeTargetSpec] = (),
        regime_blueprints: Mapping[str, HirBlueprintPacket] | None = None,
        tracker: BayesianRuleTracker | None = None,
        flywheel_coordinator: FlywheelCoordinator | None = None,
        flywheel: FlywheelDistiller | None = None,
        boundary_ledger: DeterministicSearchBoundaryLedger | None = None,
        beams: dict[str, TwoLaneBeamSearch] | None = None,
        msprof_cache: dict[tuple[str, str], dict[str, float]] | None = None,
        case_dims_cache: dict[int, dict[str, object]] | None = None,
        regime_diag_stats: dict[str, dict[str, object]] | None = None,
        scoreboard_anchors: Sequence[AnchorCaseSpec] = (),
        guardrail_cases: Sequence[GuardrailCaseSpec] = (),
        calibrated_dispatch_floor_us: float = 0.0,
        cannbench_repo: Path | None = None, env_file: Path | None = None,
        agent_gates_bin: Path | None = None, enable_track_router: bool = False,
        topology_axes: TopologyAxes | None = None, sil_advisory: object | None = None,
        real_npu_path_fn: Callable[[], bool] | None = None, is_real_npu_path: bool = False,
        l0_runner: Callable[[str, RegimeTargetSpec], StageAResult] | None = None,
        l2_batch_runner: Callable[..., tuple[Mapping[str, CandidateSurrogateResult], object]] | None = None,
        derive_topology_axes_fn: Callable[..., TopologyAxes] | None = None,
        has_l2_executor_fn: Callable[[], bool] | None = None,
        operator_proto: Mapping[str, object] | None = None,
    ) -> None:
        op_slug = operator_name.replace("/", "_") or "campaign"
        self._operator_name, self._device_id = operator_name, str(device_id)
        self.run_id = run_id or f"{op_slug}_{int(time.time())}_{os.getpid()}"
        self._allow_unreviewed_distilled_rules = bool(allow_unreviewed_distilled_rules)
        self._target_module_path = target_module_path
        self._checkpoint_path, self._campaign_kb_path = checkpoint_path, campaign_kb_path
        self._experience_lake_dir = experience_lake_dir
        self._experience_lake = experience_lake if experience_lake is not None else (ExperienceLake(experience_lake_dir) if experience_lake_dir is not None else None)
        self._cost_advisor = cost_advisor
        self._target = target or AscendTarget("ascend.950pr")
        self._perf_target = perf_target
        self._regime_specs = tuple(regime_specs)
        self._regime_blueprints = dict(regime_blueprints or {})
        self._tracker, self._flywheel = tracker, flywheel
        self._flywheel_coordinator = flywheel_coordinator or FlywheelCoordinator(flywheel=flywheel)
        self.boundary_ledger = boundary_ledger or DeterministicSearchBoundaryLedger()
        self.beams: dict[str, TwoLaneBeamSearch] = beams if beams is not None else {}
        self.msprof_cache: dict[tuple[str, str], dict[str, float]] = msprof_cache if msprof_cache is not None else {}
        self._case_dims_cache: dict[int, dict[str, object]] = case_dims_cache if case_dims_cache is not None else {}
        self.regime_diag_stats: dict[str, dict[str, object]] = regime_diag_stats if regime_diag_stats is not None else {}
        self.scoreboard_anchors: tuple[AnchorCaseSpec, ...] = tuple(scoreboard_anchors)
        self.guardrail_cases: list[GuardrailCaseSpec] = list(guardrail_cases)
        self.calibrated_dispatch_floor_us: float = float(calibrated_dispatch_floor_us)
        self._cannbench_repo, self._env_file, self._agent_gates_bin = cannbench_repo, env_file, agent_gates_bin
        self._enable_track_router, self._topology_axes, self._sil_advisory = bool(enable_track_router), topology_axes, sil_advisory
        self._real_npu_path_fn, self._is_real_npu_path_flag = real_npu_path_fn, bool(is_real_npu_path)
        self._l0_runner, self._l2_batch_runner = l0_runner, l2_batch_runner
        self._derive_topology_axes_fn, self._has_l2_executor_fn = derive_topology_axes_fn, has_l2_executor_fn
        self._operator_proto = operator_proto
        self.active_regime_tag: str | None = self._regime_specs[0].regime_tag if self._regime_specs else None
        self.last_round_promoted_by_regime: dict[str, bool] = {}
        self.last_local_rel_gain_by_regime: dict[str, float] = {}
        self.stagnation_by_regime: dict[str, int] = {}
        self.compile_wall_streak_by_regime: dict[str, int] = {}
        self.active_promotion: PendingPromotion | None = None
        self.pending_promotion: PendingPromotion | None = None
        self.pending_l4: PendingL4 | None = None

    def _is_real_npu_path(self) -> bool:
        return bool(self._real_npu_path_fn()) if self._real_npu_path_fn is not None else self._is_real_npu_path_flag

    def _resolve_axes(
        self, ledger: ObjectivePhysicalLedger | None, *, features: object | None = None, route: object | None = None
    ) -> TopologyAxes:
        if self._derive_topology_axes_fn is not None:
            return self._derive_topology_axes_fn(ledger, features=features, route=route)
        return derive_topology_axes(
            self._operator_name, ledger, topology_axes=self._topology_axes,
            sil_advisory=self._sil_advisory, features=features, route=route,
        )

    def _dims(self, case_num: int, regime_tag: str = "") -> dict[str, object]:
        return lookup_case_dims_safe(
            self._operator_name, case_num, regime_tag=regime_tag, regime_specs=self._regime_specs,
            cache=self._case_dims_cache, cannbench_repo=self._cannbench_repo, env_file=self._env_file,
        )

    def resolve_case_roofline_metrics(
        self, regime_tag: str, ledger: ObjectivePhysicalLedger | None, telemetry: Mapping[str, object]
    ) -> tuple[float, float, float]:
        """Compute min GM bytes, compute floor, and transfer inflation for a regime."""
        bp = self._regime_blueprints.get(regime_tag)
        if bp is None:
            return 0.0, 0.0, 0.0
        rf = bp.roofline
        min_gm, comp_floor = float(rf.min_gm_bytes), float(rf.compute_floor_us)
        inf = compute_transfer_inflation_ratio(ledger, telemetry=telemetry, roofline=rf) if (ledger is not None and min_gm > 0.0) else 0.0
        return min_gm, comp_floor, inf

    def record_kb_event(
        self, *, rule_id: str, postcondition_realized: bool, speed_gain: float = 0.0,
        topology_axes: object | None = None, round_idx: int = 0,
        operator_name: str | None = None, run_id: str | None = None,
    ) -> Path | None:
        """Append one causal rule application telemetry shard to the KB event lake."""
        if self._experience_lake_dir is None:
            return None
        try:
            return append_kb_event_shard(
                self._experience_lake_dir,
                run_id=run_id or self.run_id,
                rule_id=rule_id,
                postcondition_realized=postcondition_realized,
                speed_gain=speed_gain,
                topology_axes=topology_axes,
                operator_name=operator_name or self._operator_name,
                round_idx=round_idx,
            )
        except OSError as exc:
            logger.debug("Appending KB event shard skipped: %s", exc)
            return None

    def record_surrogate_observations(
        self, *, source: str, result: CandidateSurrogateResult, active_anchors: Sequence[AnchorCaseSpec],
        eval_tier_override: str | None = None, parent_source: str = "",
        parent_result: CandidateSurrogateResult | None = None, parent_ledger: ObjectivePhysicalLedger | None = None,
        parent_bisheng_res: BishengResourceSummary | Mapping[str, object] | None = None,
        parent_stage_b_profile: Mapping[str, object] | None = None, parent_axes: TopologyAxes | None = None,
        ledger: ObjectivePhysicalLedger | None = None,
        bisheng_summary: BishengResourceSummary | Mapping[str, object] | None = None,
        stage_b_profile: Mapping[str, object] | None = None, axes: TopologyAxes | None = None,
        policy: Mapping[str, object] | None = None, parent_policy: Mapping[str, object] | None = None,
        worker_kind: str = "", rule_id: str | None = None, backend_track: str = "tilelang_dsl", regime_tag: str = "",
    ) -> None:
        """Persist candidate artifact, L2 observations, anomaly logs, and advisor updates."""
        _ = active_anchors
        cand_sha = compute_source_sha256(source)
        parent_sha = compute_source_sha256(parent_source) if parent_source else ""
        merged_prof = merge_microarch_profiles(getattr(ledger, "microarch_profile", None) if ledger else None, stage_b_profile)
        prof_dict = merged_prof.to_dict() if not merged_prof.is_empty else {}
        ledger_dict = ledger.to_dict() if ledger is not None else {}
        bsum_dict = dict(bisheng_summary.to_dict()) if hasattr(bisheng_summary, "to_dict") else (dict(bisheng_summary) if isinstance(bisheng_summary, Mapping) else {})
        axes_dict = dict(axes.to_dict()) if (axes is not None and hasattr(axes, "to_dict")) else ({"compute_engine": axes.compute_engine, "memory_ring": axes.memory_ring, "iteration_regularity": axes.iteration_regularity} if axes else {})
        pol_dict = dict(policy or {})
        eff_regime = regime_tag or (result.anchors[0].regime_tag if result.anchors else "")
        if self._experience_lake is not None:
            try:
                self._experience_lake.register_candidate(
                    CandidateArtifactRecord(
                        cand_sha=cand_sha, operator=self._operator_name, regime_tag=eff_regime,
                        backend_track=backend_track, parent_sha=parent_sha, worker_kind=worker_kind,
                        rule_id=rule_id, config_knobs=pol_dict, topology_axes=axes_dict,
                        base_ledger=ledger_dict, microarch_profile=prof_dict, bisheng_summary=bsum_dict,
                        dsl_line_count=len(source.splitlines()), subkernel_count=max(1, merged_prof.subkernel_count),
                    )
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                logger.debug("ExperienceLake register_candidate skipped: %s", exc)
        parent_by_case = {int(a.case_num): a for a in parent_result.anchors} if parent_result else {}
        parent_prof = merge_microarch_profiles(getattr(parent_ledger, "microarch_profile", None) if parent_ledger else None, parent_stage_b_profile) if (parent_ledger is not None or parent_stage_b_profile is not None) else None
        bn_by_regime: dict[str, int] = {}
        for anc in result.anchors:
            if anc.regime_tag not in bn_by_regime:
                bn = bottleneck_anchor_for_regime(result, anc.regime_tag)
                bn_by_regime[anc.regime_tag] = int(bn.case_num) if bn is not None else -1
        obs_batch, msprof_batch, pair_batch = [], [], []
        for anc in result.anchors:
            if anc.cand_us <= 0.0:
                continue
            eval_tier = eval_tier_override if eval_tier_override is not None else ("l2_pass" if anc.accuracy_passed else "l2_acc_fail")
            dims = self._dims(anc.case_num, anc.regime_tag)
            proj_wl = project_case_workload(ledger_dict, microarch_profile=prof_dict, case_dims=dims, policy=pol_dict, t_hw_us=anc.t_hw_us, axes=axes, target=self._target)
            cand_msprof = self.msprof_cache.get((cand_sha, anc.regime_tag))
            bn_case = bn_by_regime.get(anc.regime_tag, -1)
            attach_msprof = dict(cand_msprof) if (cand_msprof and (bn_case < 0 or int(anc.case_num) == bn_case)) else {}
            raw_ana, ana_pred = 0.0, 0.0
            if self._cost_advisor is not None:
                try:
                    pr = self._cost_advisor.predict_candidate_us(proj_wl, bisheng_res=bisheng_summary)
                    if isinstance(pr, (int, float)) and not isinstance(pr, bool):
                        raw_ana = ana_pred = float(pr)
                    elif pr is not None:
                        raw_ana = float(getattr(pr, "tier0_physics_us", 0.0) or 0.0) + float(getattr(pr, "tier1_penalty_us", 0.0) or 0.0)
                        cal = float(getattr(pr, "pred_us", 0.0) or 0.0)
                        ana_pred = cal if cal > 0.0 else raw_ana
                except (TypeError, ValueError):
                    ana_pred = 0.0
            ring_k, bcv = max(1, int(getattr(anc, "ring_burst_k", 1) or 1)), max(0.0, float(getattr(anc, "burst_batch_cv", 0.0) or 0.0))
            if self._experience_lake is not None:
                bp_tag = anc.regime_tag if anc.regime_tag in self._regime_blueprints else eff_regime
                mgm, cf, tr = self.resolve_case_roofline_metrics(bp_tag, ledger, {"dur_us": float(anc.cand_us), "measured_latency_us": float(anc.cand_us), "tier01_pred_us": float(raw_ana), **attach_msprof})
                try:
                    self._experience_lake.append_observation(
                        CaseObservationRecord(
                            run_id=self.run_id, operator=self._operator_name, regime_tag=anc.regime_tag,
                            cand_sha=cand_sha, eval_tier=eval_tier, case_num=int(anc.case_num),
                            measured_us=float(anc.cand_us), accuracy_passed=bool(anc.accuracy_passed),
                            parent_sha=parent_sha, device_id=self._device_id, case_dims=dims,
                            t_hw_us=float(anc.t_hw_us), ref_perf_us=float(anc.ref_perf_us),
                            speedup=float(anc.speedup), eq3_score=float(anc.eq3_score), msprof_pmu=attach_msprof,
                            projected_tiles=int(proj_wl.get("total_tiles", 0) or 0),
                            projected_gm_bytes=float(proj_wl.get("gm_bytes", 0.0) or 0.0),
                            projected_contig_bytes=int(proj_wl.get("contig_burst_bytes", 32768) or 32768),
                            min_gm_bytes=mgm, compute_floor_us=cf, transfer_inflation_ratio=tr,
                            analytical_pred_us=ana_pred, ring_burst_k=ring_k, burst_batch_cv=bcv,
                            spike_retries=max(0, int(getattr(anc, "spike_retries", 0) or 0)),
                            card_ext_hbm_mb=max(0, int(getattr(anc, "card_ext_hbm_mb", 0) or 0)),
                            is_real_npu=self._is_real_npu_path(),
                        )
                    )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    logger.debug("ExperienceLake append_observation skipped: %s", exc)
            if not is_clean_calibration_sample(bcv):
                continue
            if self._checkpoint_path is not None and ana_pred > 0.0 and anc.cand_us > 0.0 and abs(math.log(float(anc.cand_us) / float(ana_pred))) > ANOMALY_LOG_ERROR_THRESHOLD:
                apath = self._checkpoint_path.parent / "unmodeled_anomalies.jsonl"
                arec = {"run_id": self.run_id, "operator": self._operator_name, "regime_tag": anc.regime_tag, "cand_sha": cand_sha, "case_num": int(anc.case_num), "measured_us": float(anc.cand_us), "analytical_pred_us": float(ana_pred), "log_error": abs(math.log(float(anc.cand_us) / float(ana_pred))), "burst_batch_cv": bcv}
                try:
                    apath.parent.mkdir(parents=True, exist_ok=True)
                    with apath.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(arec, sort_keys=True) + "\n")
                except OSError as exc:
                    logger.debug("Appending unmodeled anomaly skipped: %s", exc)
            if attach_msprof:
                msprof_batch.append({"workload": proj_wl, "msprof_pmu": attach_msprof, "measured_us": float(anc.cand_us), "bisheng_res": bisheng_summary})
            obs_batch.append({"workload": proj_wl, "measured_us": float(anc.cand_us), "bisheng_res": bisheng_summary, "cand_sha": cand_sha, "operator": self._operator_name, "regime_tag": anc.regime_tag, "case_num": int(anc.case_num), "eval_tier": eval_tier})
            p_anc = parent_by_case.get(int(anc.case_num))
            if p_anc is not None and p_anc.cand_us > 0.0 and parent_sha and parent_sha != cand_sha:
                pproj = project_case_workload(
                    parent_ledger.to_dict() if parent_ledger is not None else ledger_dict,
                    microarch_profile=(parent_prof.to_dict() if parent_prof is not None and not parent_prof.is_empty else prof_dict),
                    case_dims=dims, policy=(dict(parent_policy) if parent_policy is not None else pol_dict),
                    t_hw_us=p_anc.t_hw_us, axes=parent_axes or axes, target=self._target,
                )
                pair_batch.append({"cand_workload": proj_wl, "cand_us": float(anc.cand_us), "parent_workload": pproj, "parent_us": float(p_anc.cand_us), "cand_bisheng_res": bisheng_summary, "parent_bisheng_res": parent_bisheng_res, "cand_sha": cand_sha, "parent_sha": parent_sha, "operator": self._operator_name, "regime_tag": anc.regime_tag, "case_num": int(anc.case_num)})
        if self._cost_advisor is not None and (obs_batch or msprof_batch or pair_batch):
            try:
                self._cost_advisor.observe_surrogate_batch(observations=obs_batch, msprof_records=msprof_batch, pair_deltas=pair_batch, refit=True)
            except (OSError, ValueError, TypeError) as exc:
                logger.debug("Cost advisor online observation skipped: %s", exc)

    def record_l4_observations(self, *, source: str, l4_result: L4CannbenchResult, active_anchors: Sequence[AnchorCaseSpec]) -> None:
        """Persist official L4 execution observations into ExperienceLake and CostAdvisor."""
        if not l4_result.per_case_elapsed_us:
            return
        cand_sha = compute_source_sha256(source)
        stage_a: StageAResult | None = None
        if self._regime_specs and self._l0_runner is not None:
            try:
                stage_a = self._l0_runner(source, self._regime_specs[0])
            except (SyntaxError, RuntimeError, ValueError, TypeError, OSError):
                stage_a = None
        ledger = stage_a.ledger if stage_a is not None else None
        axes = self._resolve_axes(ledger, features=stage_a.features if stage_a else None, route=stage_a.route if stage_a else None)
        prof_dict = dict(getattr(ledger, "microarch_profile", None) or {})
        ledger_dict = ledger.to_dict() if ledger is not None else {}
        axes_dict = axes.to_dict() if hasattr(axes, "to_dict") else {}
        first_reg = self._regime_specs[0].regime_tag if self._regime_specs else ""
        if self._experience_lake is not None:
            try:
                self._experience_lake.register_candidate(
                    CandidateArtifactRecord(
                        cand_sha=cand_sha, operator=self._operator_name, regime_tag=first_reg,
                        backend_track=(stage_a.route.track if self._enable_track_router and stage_a and stage_a.route else "tilelang_dsl"),
                        topology_axes=axes_dict, base_ledger=ledger_dict, microarch_profile=prof_dict, dsl_line_count=len(source.splitlines()),
                    )
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                logger.debug("ExperienceLake register L4 candidate skipped: %s", exc)
        by_case = {int(a.case_num): a for a in active_anchors}
        l4_obs: list[dict[str, object]] = []
        for cnum_raw, elapsed_raw in sorted(l4_result.per_case_elapsed_us.items()):
            cnum, elapsed_us = int(cnum_raw), float(elapsed_raw or 0.0)
            if elapsed_us <= 0.0:
                continue
            anc = by_case.get(cnum)
            rtag, dims = (anc.regime_tag if anc else first_reg), self._dims(cnum, anc.regime_tag if anc else first_reg)
            if anc is not None:
                t_hw, ref_perf = anc.t_hw_us, anc.ref_perf_us
            else:
                try:
                    ref_perf, t_hw = lookup_case_metadata(self._operator_name, cnum, cannbench_repo=self._cannbench_repo, env_file=self._env_file)
                except (FileNotFoundError, OSError, KeyError, ValueError):
                    ref_perf, t_hw = 0.0, 0.0
            sp = float(l4_result.per_case_speedups.get(cnum, 0.0) or 0.0) or (ref_perf / elapsed_us if ref_perf > 0.0 else 0.0)
            eq3 = compute_case_eq3_score(ref_perf, elapsed_us, t_hw) if (ref_perf > 0.0 and t_hw > 0.0) else 0.0
            proj_wl = project_case_workload(ledger_dict, microarch_profile=prof_dict, case_dims=dims, t_hw_us=t_hw, axes=axes, target=self._target)
            if self._experience_lake is not None:
                mgm, cf, tr = self.resolve_case_roofline_metrics(rtag, ledger, {"dur_us": elapsed_us, "measured_latency_us": elapsed_us})
                try:
                    self._experience_lake.append_observation(
                        CaseObservationRecord(
                            run_id=self.run_id, operator=self._operator_name, regime_tag=rtag, cand_sha=cand_sha,
                            eval_tier="l4_official", case_num=cnum, measured_us=elapsed_us,
                            accuracy_passed=bool(l4_result.passed), device_id=self._device_id, case_dims=dims,
                            t_hw_us=t_hw, ref_perf_us=ref_perf, speedup=sp, eq3_score=eq3,
                            projected_tiles=int(proj_wl.get("total_tiles", 0) or 0),
                            projected_gm_bytes=float(proj_wl.get("gm_bytes", 0.0) or 0.0),
                            projected_contig_bytes=int(proj_wl.get("contig_burst_bytes", 32768) or 32768),
                            min_gm_bytes=mgm, compute_floor_us=cf, transfer_inflation_ratio=tr,
                        )
                    )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    logger.debug("ExperienceLake L4 append skipped: %s", exc)
            l4_obs.append({"workload": proj_wl, "measured_us": elapsed_us, "cand_sha": cand_sha, "operator": self._operator_name, "regime_tag": rtag, "case_num": cnum, "eval_tier": "l4_official"})
        if self._cost_advisor is not None and l4_obs:
            try:
                self._cost_advisor.observe_surrogate_batch(observations=l4_obs, msprof_records=[], pair_deltas=[], refit=True)
            except (OSError, ValueError, TypeError) as exc:
                logger.debug("Cost advisor L4 observe skipped: %s", exc)

    @staticmethod
    def evaluate_causal_rule_outcome(
        *, rule: AscendCausalRule, res_all_passed: bool, rolled_back: bool,
        ledger: ObjectivePhysicalLedger, features: object, cand_residual: StaticDynamicResidual,
        cand_lat_us: float, cand_obj_score: float, parent_ledger: ObjectivePhysicalLedger,
        parent_residual: StaticDynamicResidual, parent_lat_us: float, parent_obj_score: float,
        has_cand_msprof: bool, cand_global_score: float | None = None,
        parent_global_score: float | None = None, min_speedup_gain: float = 0.01,
        min_score_gain: float = 0.05, cand_compiler_summary: BishengResourceSummary | None = None,
        roofline: HirRooflineSummary | None = None,
    ) -> tuple[bool, float]:
        """Evaluate whether a causal postcondition materialized with non-regressive gain."""
        if not res_all_passed or not ledger.is_legal or rolled_back:
            return False, 0.0
        if cand_global_score is not None and parent_global_score is not None and cand_global_score < parent_global_score - 1e-6:
            return False, 0.0
        if parent_lat_us > 0.0 and cand_lat_us > 0.0 and cand_lat_us >= parent_lat_us:
            return False, 0.0
        if cand_obj_score < parent_obj_score - 1e-6:
            return False, 0.0
        rel_gain = (parent_lat_us - cand_lat_us) / max(parent_lat_us, 1e-6) if (parent_lat_us > 0.0 and cand_lat_us > 0.0) else 0.0
        local_gain = cand_obj_score - parent_obj_score
        if not (rel_gain >= float(min_speedup_gain) or (local_gain >= float(min_score_gain) and (parent_lat_us <= 0.0 or cand_lat_us <= 0.0 or cand_lat_us < parent_lat_us))):
            return False, 0.0
        rcomp = cand_compiler_summary if cand_compiler_summary is not None else compiler_summary_from_residual(cand_residual)
        ctx = build_predicate_context(
            ledger, features=features, residual=cand_residual, compiler_summary=rcomp,
            prev_ledger=parent_ledger, prev_residual=parent_residual,
            prev_compiler_summary=compiler_summary_from_residual(parent_residual), roofline=roofline,
        )
        prev_ns = ctx.setdefault("prev", {})
        if isinstance(prev_ns, dict):
            prev_ns.update(parent_residual.to_dict())
            prev_ns["measured_latency_us"] = float(parent_lat_us)
        pmu = ("measured_aic_mac_ratio", "measured_aiv_vec_ratio", "measured_mte2_wait_ratio", "measured_sync_stall_ratio", "unexplained_gap_ratio")
        if not has_cand_msprof:
            for rk in pmu:
                ctx[rk] = getattr(parent_residual, rk)
        ok = rule.postcondition_realized(ctx)
        if not ok and not has_cand_msprof and any(k in rule.expected_postcondition for k in pmu):
            ok = bool(parent_lat_us > 0.0 and cand_lat_us < 0.98 * parent_lat_us and cand_obj_score > parent_obj_score + 0.02)
        return bool(ok), (rel_gain if ok else 0.0)

    def run_flywheel_distillation(
        self, *, baseline_source: str, winner_source: str, baseline_latency_us: float,
        winner_latency_us: float, func_name: str, regime_spec: RegimeTargetSpec,
        active_anchors: Sequence[AnchorCaseSpec], topology_axes: TopologyAxes,
        rel_gain: float = 0.0, local_rel_gain: float = 0.0, worker_kind: str = "",
        round_idx: int = 0, rule_id: str | None = None, symptom_family: str = "",
        sigma_paired_us: float = 0.0, tracker: BayesianRuleTracker | None = None,
    ) -> list[str]:
        """Distill counterfactual causal rules via FlywheelCoordinator."""
        _ = local_rel_gain
        eff_tracker = tracker if tracker is not None else self._tracker
        if self._flywheel is None or eff_tracker is None:
            return []
        reg_anchors = [a for a in active_anchors if a.regime_tag == regime_spec.regime_tag] or list(active_anchors)
        adims = regime_spec.representative_anchor_dims or regime_spec.representative_hir_dims

        def _batch_chunk_latencies(chunk_pairs: Sequence[tuple[str, str]]) -> dict[str, float]:
            if not chunk_pairs or not reg_anchors or self._l2_batch_runner is None:
                return {}
            with tempfile.TemporaryDirectory(prefix="tf_fly_batch_") as tmp_dir:
                files = []
                for cid, src in chunk_pairs:
                    cp = Path(tmp_dir) / f"{cid}.py"
                    cp.write_text(src, encoding="utf-8")
                    files.append((cid, cp))
                try:
                    bres, _ = self._l2_batch_runner(files, reg_anchors, include_guardrails=False)
                except (OSError, subprocess.SubprocessError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
                    logger.debug("Flywheel batched ablation timing skipped: %s", exc)
                    return {}
            return {cid: r.mean_latency_us() for cid, r in bres.items() if r.all_passed and r.mean_latency_us() > 0.0}

        has_l2 = bool(self._has_l2_executor_fn()) if self._has_l2_executor_fn else bool(
            self._l2_batch_runner is not None or self._env_file is not None or self._agent_gates_bin is not None
            or os.environ.get("TILEFOUNDRY_NPU_AGENT_GATES_BIN", "").strip()
            or os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()
        )
        shapes = extract_ordered_tensor_shapes(case_entry=regime_spec.representative_case_entry, dims=adims or None, proto=self._operator_proto)
        return self._flywheel_coordinator.run_distillation(
            operator_name=self._operator_name, run_id=self.run_id, tracker=eff_tracker,
            baseline_source=baseline_source, winner_source=winner_source,
            baseline_latency_us=baseline_latency_us, winner_latency_us=winner_latency_us,
            sigma_paired_us=sigma_paired_us, func_name=func_name, regime_spec=regime_spec,
            topology_axes=topology_axes, batch_latency_evaluator=_batch_chunk_latencies,
            campaign_kb_path=self._campaign_kb_path, experience_lake_dir=self._experience_lake_dir,
            allow_unreviewed=self._allow_unreviewed_distilled_rules, has_l2_executor=has_l2,
            real_npu_path=self._is_real_npu_path(), source_sha256_fn=source_sha256,
            write_text_fn=atomic_write_text, distill_shapes=shapes, rel_gain=rel_gain,
            worker_kind=worker_kind, round_idx=round_idx, rule_id=rule_id,
            symptom_family=symptom_family, proto_dims=adims or None,
        )

    def collect_trackb_round_evidence(
        self, *, round_idx: int, regime_spec: RegimeTargetSpec, blueprint: HirBlueprintPacket | None,
        live_plan: RegimePlanResult | None, active_mask: ConstraintRelaxationMask,
        checks: Sequence[L0L1CandidateCheck], l2_results: Mapping[str, CandidateSurrogateResult],
        active_kernel: str, l2_run_ids: Mapping[str, str], parent_sha256: str,
        parent_l1_summary: BishengResourceSummary | None, parent_l1_summary_source: str,
    ) -> tuple[dict[str, object], ...]:
        """Record per-candidate Track-B blueprint, 5D observation, residual, and provenance."""
        if regime_spec.runtime_backend not in ("ascendc_trackb", "hybrid_trackb") or blueprint is None:
            return ()
        policy = live_plan.policy if live_plan is not None else {}
        ce = regime_spec.representative_case_entry
        raw_cid, raw_dt = ce.get("case_id", 0), ce.get("dtype", "")
        rep_cid = raw_cid if isinstance(raw_cid, int) and not isinstance(raw_cid, bool) else 0
        rep_dt: str | list[str] = raw_dt if isinstance(raw_dt, str) else ([str(i) for i in raw_dt] if isinstance(raw_dt, (list, tuple)) else "")
        ps_payload = parent_l1_summary.to_dict() if parent_l1_summary is not None else None
        rf = blueprint.roofline
        bp_sum: dict[str, object] = {
            "entry_name": blueprint.entry_name, "ideal_roofline_us": float(rf.ideal_roofline_us),
            "compute_floor_us": float(rf.compute_floor_us), "memory_floor_us": float(rf.memory_floor_us),
            "dominant_bound": rf.dominant_bound,
            "anchor_dims": {str(k): int(v) for k, v in rf.anchor_dims.items()},
            "requested_anchor_dims": {str(k): int(v) for k, v in rf.requested_anchor_dims.items()},
            "resolved_hir_dims": {str(k): int(v) for k, v in rf.resolved_hir_dims.items()},
            "shape_exact_match": bool(rf.shape_exact_match), "fallback_reason": rf.fallback_reason,
        }
        active_unlocked = active_mask.unlocked_facets()
        ev: list[dict[str, object]] = []
        for chk in checks:
            sa = chk.stage_a
            ledger = sa.ledger if sa is not None else None
            if ledger is None:
                continue
            l2r = l2_results.get(chk.candidate_id)
            local = l2r.for_regime(regime_spec.regime_tag) if l2r is not None else None
            has_lat = local is not None and bool(local.anchors)
            lat_us = local.mean_latency_us() if has_lat else 0.0
            csha = source_sha256(chk.patched_module_source)
            tel: dict[str, object] = {"dur_us": lat_us, "roofline_eta": local.avg_roofline_eta if has_lat else 0.0}
            msprof = self.msprof_cache.get((csha, regime_spec.regime_tag))
            if msprof:
                tel.update(msprof)
            else:
                tel["measured_latency_us"], tel["is_analytical_proxy"] = lat_us, True
            csum = chk.stage_b.compiler_summary if chk.stage_b is not None else None
            obs = ledger.to_observation_5d(compiler_summary=csum, telemetry=tel, roofline=rf)
            res = obs.compute_axis_residuals()
            unlocked = tuple(f for f in res.unlock_causal_facets(active_mask).unlocked_facets() if f not in active_unlocked)
            route = sa.route if sa is not None else None
            ev.append({
                "round_idx": round_idx, "regime_tag": regime_spec.regime_tag, "candidate_id": chk.candidate_id,
                "candidate_sha256": csha, "active_kernel": active_kernel,
                "l2_run_id": l2_run_ids.get(chk.candidate_id, "") if l2r is not None else "",
                "parent_candidate_sha256": parent_sha256,
                "parent_compiler_summary": dict(ps_payload) if ps_payload is not None else None,
                "parent_compiler_summary_source": parent_l1_summary_source,
                "representative_case_id": rep_cid, "representative_input_dtype": rep_dt,
                "runtime_backend": regime_spec.runtime_backend, "backend_tier": "ascendc_template",
                "solver_backend_tier": str(policy.get("backend_tier", "")),
                "track": route.track if self._enable_track_router and route is not None else "unconstrained",
                "stage_a_backend_kind": sa.backend_kind, "l0_l1_passed": chk.passed,
                "l1_compiler_passed": bool(chk.stage_b.passed) if chk.stage_b else False,
                "l2_evaluated": l2r is not None, "l2_all_passed": bool(l2r.all_passed) if l2r is not None else False,
                "latency_us": float(lat_us), "blueprint": dict(bp_sum), "observation": obs.to_dict(),
                "axis_residual": {"compute_issue_gap": res.compute_issue_gap, "offchip_dma_gap": res.offchip_dma_gap, "onchip_sram_reg_gap": res.onchip_sram_reg_gap, "pipeline_sync_gap": res.pipeline_sync_gap, "grid_wave_gap": res.grid_wave_gap},
                "dominant_axis": res.dominant_axis, "unlocked_facets": list(unlocked), "active_unlocked_facets": list(active_unlocked),
                "field_provenance": {
                    "has_l1_compiler_summary": obs.has_l1_compiler_summary, "has_l2_telemetry": obs.has_l2_telemetry,
                    "measured_l2_fields": list(obs.measured_l2_fields),
                    "latency_source": "l2_batch_mean" if has_lat else "none",
                    "observation_source": "msprof_l2" if msprof else ("l2_batch" if has_lat else "roofline_static"),
                    "compiler_summary_source": "bisheng_l1" if csum is not None else "absent",
                },
            })
        return tuple(ev)

    def emit_round_funnel_events(
        self, *, round_idx: int, regime_spec: RegimeTargetSpec, l0_l1_checks: Sequence[L0L1CandidateCheck],
        failed_entries: Sequence[tuple[str, str, str]] = (),
        meta_by_id: Mapping[str, tuple[str, AscendCausalRule | None, str]] | None = None,
        l2_results: Mapping[str, CandidateSurrogateResult] | None = None,
        l0_l1_faults_by_cid: Mapping[str, TypedFault] | None = None,
        cand_axes_by_cid: Mapping[str, TopologyAxes] | None = None,
        parent_axes: TopologyAxes | None = None, top_chk: L0L1CandidateCheck | None = None,
        l2_infra_fault: TypedFault | None = None, initial_top_cid: str | None = None,
        promoted: bool = False, l3_resolution: str = L3Resolution.NOT_RUN.value,
        l3_failure_stage: str = "", stagnation_after: int = 0, admitted_rule_ids: Sequence[str] = (),
        local_best: CandidateSurrogateResult | None = None, parent_snippet: str = "",
        ub_bytes_by_cid: Mapping[str, int] | None = None, knob_sig_by_cid: Mapping[str, str] | None = None,
        pred_by_cid: Mapping[str, float] | None = None, measured_by_cid: Mapping[str, float] | None = None,
        stage_timings_sec: Mapping[str, float] | None = None, prewarm_stats: Mapping[str, int] | None = None,
        cost_advisor_predictions: int = 0, telemetry_dir: Path | None = None,
    ) -> RoundReceipt:
        """Emit typed RoundFunnelEvent rows to JSONL and return a RoundReceipt."""
        eff_meta, eff_l2, eff_faults, eff_axes = meta_by_id or {}, l2_results or {}, l0_l1_faults_by_cid or {}, cand_axes_by_cid or {}
        eff_ub, eff_knobs, eff_preds, eff_meas = ub_bytes_by_cid or {}, knob_sig_by_cid or {}, pred_by_cid or {}, dict(measured_by_cid or {})
        eff_timings = {str(k): float(v) for k, v in (stage_timings_sec or {}).items()}
        eff_prewarm = {str(k): int(v) for k, v in (prewarm_stats or {}).items()}
        eff_paxes = parent_axes if parent_axes is not None else self._resolve_axes(None)
        eff_tdir = telemetry_dir if telemetry_dir is not None else (self._checkpoint_path.parent / "telemetry" if self._checkpoint_path is not None else None)
        top_cid, emitted = (top_chk.candidate_id if top_chk is not None else None), 0
        if eff_tdir is not None:
            emitted_ids: set[str] = set()
            all_chks = list(l0_l1_checks) + ([top_chk] if top_chk is not None and all(c.candidate_id != top_chk.candidate_id for c in l0_l1_checks) else [])
            for chk in all_chks:
                cid = chk.candidate_id
                emitted_ids.add(cid)
                wkind, rule_obj, _ = eff_meta.get(cid, ("unknown", None, ""))
                l2r = eff_l2.get(cid)
                ev_res = l2r.for_regime(regime_spec.regime_tag) if l2r is not None else None
                if ev_res is not None and not ev_res.anchors:
                    ev_res = l2r
                if cid not in eff_meas and ev_res is not None and ev_res.anchors and ev_res.mean_latency_us() > 0.0:
                    eff_meas[cid] = float(ev_res.mean_latency_us())
                cfault = eff_faults.get(cid)
                if chk.passed:
                    if l2r is not None and not l2r.all_passed:
                        fp, ec = FaultPlane.CANDIDATE_NUMERIC.value, "L2_ACCURACY_OR_RUNTIME_FAILED"
                    elif l2_infra_fault is not None:
                        fp, ec = l2_infra_fault.plane.value, l2_infra_fault.code
                    else:
                        fp, ec = FaultPlane.NONE.value, "OK"
                elif cfault is not None:
                    fp, ec = cfault.plane.value, cfault.code or "L0_L1_FAILED"
                else:
                    raw_fp = getattr(chk, "fault_plane", "") or (getattr(chk.stage_b, "fault_plane", "") if chk.stage_b else "") or (getattr(chk.stage_a, "fault_plane", "") if chk.stage_a else "")
                    fp = _resolve_fault_plane(raw_fp, failed_tier=str(chk.failed_tier or "l0_l1"), detail=str(chk.error_message or "")).value
                    ec = str(getattr(chk, "fault_code", "") or (getattr(chk.stage_b, "fault_code", "") if chk.stage_b else "") or (getattr(chk.stage_a, "fault_code", "") if chk.stage_a else "") or f"{(chk.failed_tier or 'L0_L1').upper()}_FAILED")
                ev = RoundFunnelEvent(
                    run_id=self.run_id, operator=self._operator_name, round_idx=int(round_idx),
                    regime_tag=regime_spec.regime_tag, topology_axes=eff_axes.get(cid, eff_paxes).to_dict(),
                    candidate_id=cid, cand_sha=compute_source_sha256(chk.patched_module_source) if chk.patched_module_source else "",
                    worker_kind=wkind, rule_id=rule_obj.rule_id if rule_obj is not None else None,
                    passed_l0_l1=bool(chk.passed), failed_tier=str(chk.failed_tier or ""), error_code_norm=ec,
                    repaired=bool(chk.repaired), repair_delta_lines=abs(len((chk.snippet_source or "").splitlines()) - len((parent_snippet or "").splitlines())) if chk.repaired else 0,
                    spill_bytes=int(chk.stage_b.spill_bytes) if chk.stage_b is not None else 0,
                    stack_bytes=int(chk.stage_b.compiler_summary.stack_bytes) if (chk.stage_b is not None and chk.stage_b.compiler_summary is not None) else 0,
                    ub_bytes=eff_ub.get(cid, 0), knob_signature=eff_knobs.get(cid, ""),
                    analytical_pred_us=float(eff_preds.get(cid, 0.0)), measured_l2_us=float(eff_meas.get(cid, 0.0)),
                    burst_batch_cv=max((float(a.burst_batch_cv) for a in l2r.anchors), default=0.0) if (l2r and l2r.anchors) else 0.0,
                    card_ext_hbm_mb=max((int(a.card_ext_hbm_mb) for a in l2r.anchors), default=0) if (l2r and l2r.anchors) else 0,
                    promoted_l3=bool(promoted and top_chk is not None and cid == top_chk.candidate_id),
                    l3_failure_stage=l3_failure_stage if (cid == initial_top_cid and l3_failure_stage) or (top_chk is not None and cid == top_chk.candidate_id and not promoted and l3_failure_stage) else "",
                    delta_speedup=float(ev_res.avg_speedup - local_best.avg_speedup) if (local_best is not None and ev_res is not None and ev_res.anchors) else 0.0,
                    noop_identical=bool(chk.failed_tier == "noop_identical" or "noop_identical" in (chk.error_message or "")),
                    wall_sec=float(chk.stage_b.elapsed_sec) if chk.stage_b is not None else 0.0,
                    stage_timings_sec=dict(eff_timings), prewarm_stats=dict(eff_prewarm),
                    cost_advisor_predictions=int(cost_advisor_predictions), fault_plane=fp,
                    l3_resolution=l3_resolution if cid == top_cid else L3Resolution.NOT_RUN.value,
                    is_real_npu=self._is_real_npu_path(),
                )
                try:
                    append_round_funnel_event(eff_tdir, ev)
                    emitted += 1
                except OSError as exc:
                    logger.debug("Appending round funnel event skipped: %s", exc)
            for fail_cid, fail_stage, fail_err in failed_entries:
                if fail_cid in emitted_ids:
                    continue
                emitted_ids.add(fail_cid)
                wkind, rule_obj, _ = eff_meta.get(fail_cid, ("unknown", None, ""))
                ff = eff_faults.get(fail_cid)
                if ff is None:
                    if fail_stage == "l2_infra_device_fault":
                        ff = TypedFault(plane=FaultPlane.INFRA_DEVICE_FAULT, code="L2_INFRA_DEVICE_FAULT", failed_tier=fail_stage, detail=fail_err[:500], retriable=True)
                    elif fail_stage == "l2_infra_occupancy" or (l2_infra_fault is not None and l2_infra_fault.plane == FaultPlane.INFRA_OCCUPANCY and fail_stage == l2_infra_fault.failed_tier):
                        ff = TypedFault(plane=FaultPlane.INFRA_OCCUPANCY, code=l2_infra_fault.code if l2_infra_fault else "L2_INFRA_OCCUPANCY", failed_tier=fail_stage, detail=fail_err[:500], retriable=True)
                    else:
                        rp = _resolve_fault_plane(fail_stage, failed_tier=fail_stage, detail=fail_err)
                        ff = TypedFault(plane=rp, code=f"{(fail_stage or 'UNMODELED').upper()}_FAILED", failed_tier=fail_stage, detail=fail_err[:500], retriable=rp in (FaultPlane.INFRA_OCCUPANCY, FaultPlane.INFRA_DEVICE_FAULT, FaultPlane.LLM_TRANSPORT))
                fev = replace(
                    RoundFunnelEvent.from_typed_fault(
                        run_id=self.run_id, operator=self._operator_name, round_idx=int(round_idx),
                        regime_tag=regime_spec.regime_tag, topology_axes=eff_paxes.to_dict(), candidate_id=fail_cid,
                        cand_sha="", worker_kind=wkind, fault=ff, rule_id=rule_obj.rule_id if rule_obj else None,
                        l3_resolution=L3Resolution.NOT_RUN.value, is_real_npu=self._is_real_npu_path(),
                    ),
                    stage_timings_sec=dict(eff_timings), prewarm_stats=dict(eff_prewarm),
                    cost_advisor_predictions=int(cost_advisor_predictions),
                )
                try:
                    append_round_funnel_event(eff_tdir, fev)
                    emitted += 1
                except OSError as exc:
                    logger.debug("Appending failed round funnel event skipped: %s", exc)
        return RoundReceipt(
            round_idx=int(round_idx), regime_tag=regime_spec.regime_tag, promoted=bool(promoted),
            l3_resolution=str(l3_resolution), stagnation_after=int(stagnation_after),
            admitted_rule_ids=tuple(admitted_rule_ids), stage_timings_sec=eff_timings,
            prewarm_stats=eff_prewarm, cost_advisor_predictions=int(cost_advisor_predictions), emitted_event_count=emitted,
        )

    def write_source_and_verify(self, path: Path, source: str, *, expected_sha256: str | None = None) -> str:
        """Atomically write *source* to *path* and verify SHA-256 matches."""
        return _helpers_write_source_and_verify(path, source, expected_sha256=expected_sha256)

    def ensure_candidate_bytes(self, *, path: Path, source: str, expected_sha256: str) -> str:
        """Restore *source* to *path* if missing or if its digest drifted."""
        return _helpers_ensure_candidate_bytes(path=path, source=source, expected_sha256=expected_sha256)

    def persist_runtime_state(self, key: str, value: Mapping[str, object] | None) -> None:
        """Upsert a recoverable execution record into the campaign checkpoint JSON."""
        _helpers_persist_runtime_state(self._checkpoint_path, key, value)

    def round_candidate_artifact_dir(self) -> Path | None:
        """Return the checkpoint-adjacent candidates directory for audit artifacts."""
        return None if self._checkpoint_path is None else self._checkpoint_path.parent / "candidates"

    def persist_round_candidate_artifact(self, check: L0L1CandidateCheck, *, round_idx: int) -> None:
        """Write generated source and untruncated L0/L1 diagnostics beside the checkpoint."""
        _helpers_persist_candidate_artifact(self._checkpoint_path, check, round_idx=round_idx)

    def save_checkpoint(
        self, *, incumbent_source: str, baseline_result: CandidateSurrogateResult,
        best_result: CandidateSurrogateResult, active_anchors: Sequence[AnchorCaseSpec],
        frozen_regimes: set[str], current_regime: str | None, stagnation_rounds: int,
        rounds: Sequence[object], promoted_rounds: Sequence[int], rolled_back_rounds: Sequence[int],
        escalated_rounds: Sequence[int], admitted_rule_ids: Sequence[str], stop_reason: str,
        l4_result: L4CannbenchResult | None, stagnation_by_regime: Mapping[str, int] | None = None,
        last_unpromoted_regime: str | None = None, pending_promotion: PendingPromotion | None = None,
        pending_l4: PendingL4 | None = None, active_promotion: PendingPromotion | None = None,
        scoreboard_anchors: Sequence[AnchorCaseSpec] | None = None,
        guardrail_cases: Sequence[GuardrailCaseSpec] | None = None,
        calibrated_dispatch_floor_us: float | None = None,
        last_round_promoted_by_regime: Mapping[str, bool] | None = None,
        compile_wall_streak_by_regime: Mapping[str, int] | None = None,
        snapshot: CampaignStateSnapshot | None = None,
    ) -> None:
        """Atomically persist full campaign checkpoint state to disk."""
        if self._checkpoint_path is None:
            return
        eff_active = active_promotion if active_promotion is not None else self.active_promotion
        promo = eff_active or pending_promotion or self.pending_promotion
        pl4 = pending_l4 if pending_l4 is not None else self.pending_l4
        eff_sb = tuple(scoreboard_anchors) if scoreboard_anchors is not None else self.scoreboard_anchors
        eff_gr = list(guardrail_cases) if guardrail_cases is not None else self.guardrail_cases
        eff_floor = float(calibrated_dispatch_floor_us) if calibrated_dispatch_floor_us is not None else (snapshot.dispatch_floor_us if snapshot else float(self.calibrated_dispatch_floor_us))
        eff_pmap = last_round_promoted_by_regime if last_round_promoted_by_regime is not None else (dict(snapshot.last_round_promoted_by_regime) if snapshot else self.last_round_promoted_by_regime)
        eff_wmap = compile_wall_streak_by_regime if compile_wall_streak_by_regime is not None else (dict(snapshot.compile_wall_streak_by_regime) if snapshot else self.compile_wall_streak_by_regime)
        eff_stag_map = stagnation_by_regime if stagnation_by_regime is not None else (dict(snapshot.stagnation_by_regime) if snapshot else {})
        beam_key = (snapshot.active_regime_tag if snapshot else None) or self.active_regime_tag
        primary_beam = self.beams[beam_key] if (beam_key and beam_key in self.beams) else (next(iter(self.beams.values())) if self.beams else TwoLaneBeamSearch())
        payload = {
            "version": 1, "run_id": (snapshot.run_id if snapshot and snapshot.run_id else self.run_id),
            "operator_name": self._operator_name, "target_module_path": str(self._target_module_path) if self._target_module_path else "",
            "incumbent_source": incumbent_source, "baseline_result": baseline_result.to_dict(),
            "best_result": best_result.to_dict(), "active_anchors": [a.to_dict() for a in active_anchors],
            "scoreboard_anchors": [a.to_dict() for a in eff_sb], "guardrail_cases": [g.to_dict() for g in eff_gr],
            "calibrated_dispatch_floor_us": eff_floor,
            "last_distilled_lat_by_regime": self._flywheel_coordinator.serialize_latencies(),
            "last_distilled_ckpt_by_regime": self._flywheel_coordinator.serialize_checkpoints(),
            "last_round_promoted_by_regime": {str(k): bool(v) for k, v in sorted(eff_pmap.items())},
            "frozen_regimes": sorted(frozen_regimes), "current_regime": current_regime,
            "stagnation_rounds": stagnation_rounds,
            "stagnation_by_regime": {str(k): int(v) for k, v in sorted(eff_stag_map.items())},
            "compile_wall_streak_by_regime": {str(k): int(v) for k, v in sorted(eff_wmap.items())},
            "last_unpromoted_regime": last_unpromoted_regime, "rounds": [r.to_dict() for r in rounds],
            "promoted_rounds": list(promoted_rounds), "rolled_back_rounds": list(rolled_back_rounds),
            "escalated_rounds": list(escalated_rounds), "admitted_rule_ids": list(admitted_rule_ids),
            "tracker_state": self._tracker.to_dict() if self._tracker is not None else {},
            "beam_state": primary_beam.to_dict(), "beam_states": {tag: b.to_dict() for tag, b in sorted(self.beams.items())},
            "msprof_cache": {f"{sha}:{tag}": dict(m) for (sha, tag), m in sorted(self.msprof_cache.items())},
            "stagnation_diagnosis_stats": self.serialize_diag_stats(),
            "search_boundary_ledger": self.boundary_ledger.to_dict(), "stop_reason": stop_reason,
            "l4_result": l4_result.to_dict() if l4_result else None,
            "pending_promotion": promo.to_dict() if promo is not None else None,
            "active_promotion": eff_active.to_dict() if eff_active is not None else None,
            "pending_l4": pl4.to_dict() if pl4 is not None else None,
        }
        atomic_write_text(self._checkpoint_path, json.dumps(payload, indent=2) + "\n")

    def restore_from_checkpoint(
        self, *, round_record_from_dict: Callable[[Mapping[str, object]], object] | None = None,
        pending_l4_from_dict: Callable[[Mapping[str, object]], PendingL4] | None = None,
    ) -> tuple[
        str, CandidateSurrogateResult, CandidateSurrogateResult, list[AnchorCaseSpec], set[str],
        str | None, int, list[object], list[int], list[int], list[int], list[str],
        dict[str, int], str | None, PendingPromotion | None, PendingL4 | None,
    ]:
        """Rehydrate persisted campaign state and verify candidate bytes on disk."""
        assert self._checkpoint_path is not None
        eff_l4, eff_round = pending_l4_from_dict or PendingL4.from_dict, round_record_from_dict or CampaignRoundRecord.from_dict
        payload = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
        if isinstance(payload.get("run_id"), str) and payload["run_id"].strip():
            self.run_id = payload["run_id"].strip()
        inc_src = str(payload["incumbent_source"])
        if self._tracker is not None:
            if self._allow_unreviewed_distilled_rules and self._campaign_kb_path and self._campaign_kb_path.is_file():
                for rule in load_seed_rules(self._campaign_kb_path):
                    self._tracker.admit_rule(rule)
            if isinstance(payload.get("tracker_state"), Mapping):
                self._tracker.load_posteriors_dict(payload["tracker_state"])
        creg = payload.get("current_regime")
        if creg and str(creg) in self.beams:
            self.active_regime_tag = str(creg)
        if isinstance(bs := payload.get("beam_states"), Mapping):
            for tag, rb in bs.items():
                if isinstance(rb, Mapping):
                    self.beams[str(tag)] = TwoLaneBeamSearch.from_dict(rb)
        elif isinstance(payload.get("beam_state"), Mapping):
            self.beams[self.active_regime_tag or (self._regime_specs[0].regime_tag if self._regime_specs else "default")] = TwoLaneBeamSearch.from_dict(payload["beam_state"])
        self.msprof_cache.clear()
        if isinstance(rmc := payload.get("msprof_cache"), Mapping):
            for rk, rm in rmc.items():
                if ":" in str(rk) and isinstance(rm, Mapping):
                    sha, tag = str(rk).split(":", 1)
                    self.msprof_cache[(sha, tag)] = {str(k): float(v) for k, v in rm.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        self.restore_diag_stats(payload.get("stagnation_diagnosis_stats"))
        rl = DeterministicSearchBoundaryLedger.from_dict(payload.get("search_boundary_ledger"))
        self.boundary_ledger.rejected_cand_shas, self.boundary_ledger.infeasible_knobs_by_regime = rl.rejected_cand_shas, rl.infeasible_knobs_by_regime
        self.boundary_ledger.min_overflow_ub_bytes, self.boundary_ledger.min_overflow_stack_knobs, self.boundary_ledger.spearman_history = rl.min_overflow_ub_bytes, rl.min_overflow_stack_knobs, rl.spearman_history
        raw_sb = payload.get("scoreboard_anchors")
        if isinstance(raw_sb, Sequence) and not isinstance(raw_sb, (str, bytes)) and raw_sb:
            self.scoreboard_anchors = tuple(AnchorCaseSpec.from_dict(i) for i in raw_sb if isinstance(i, Mapping))
        elif not self.scoreboard_anchors and isinstance(payload.get("active_anchors"), Sequence) and not isinstance(payload.get("active_anchors"), (str, bytes)):
            self.scoreboard_anchors = tuple(AnchorCaseSpec.from_dict(i) for i in payload["active_anchors"] if isinstance(i, Mapping))
        raw_gr = payload.get("guardrail_cases")
        self.guardrail_cases = [GuardrailCaseSpec.from_dict(i) for i in raw_gr if isinstance(i, Mapping)] if isinstance(raw_gr, Sequence) and not isinstance(raw_gr, (str, bytes)) else []
        self.calibrated_dispatch_floor_us = float(payload.get("calibrated_dispatch_floor_us", 0.0) or 0.0)
        self._flywheel_coordinator.restore_checkpoints(payload.get("last_distilled_ckpt_by_regime"), raw_latencies=payload.get("last_distilled_lat_by_regime"), fallback_source=inc_src)
        self.last_round_promoted_by_regime = {str(k): bool(v) for k, v in rpm.items()} if isinstance(rpm := payload.get("last_round_promoted_by_regime"), Mapping) else {}
        rounds = [eff_round(r) for r in payload.get("rounds", ())]
        self.last_local_rel_gain_by_regime = {str(getattr(r, "regime_tag", "")): float(getattr(r, "local_rel_gain", 0.0)) for r in rounds}
        if isinstance(rsm := payload.get("stagnation_by_regime"), Mapping):
            stag_map = {str(k): int(v) for k, v in rsm.items()}
        else:
            stag_map, lst = {}, int(payload.get("stagnation_rounds", 0))
            if lst > 0 and (ft := str(creg or (getattr(rounds[-1], "regime_tag", "") if rounds else ""))):
                stag_map[ft] = lst
        self.stagnation_by_regime = dict(stag_map)
        self.compile_wall_streak_by_regime = {str(k): int(v) for k, v in rwm.items()} if isinstance(rwm := payload.get("compile_wall_streak_by_regime"), Mapping) else {}
        last_unprom = str(payload["last_unpromoted_regime"]) if payload.get("last_unpromoted_regime") else None
        pending_p = PendingPromotion.from_dict(payload["pending_promotion"]) if isinstance(payload.get("pending_promotion"), Mapping) else None
        active_p = PendingPromotion.from_dict(payload["active_promotion"]) if isinstance(payload.get("active_promotion"), Mapping) else None
        pending_p = active_p if active_p is not None else pending_p
        self.active_promotion, self.pending_promotion = active_p, pending_p
        self.pending_l4 = pending_l4 = eff_l4(payload["pending_l4"]) if isinstance(payload.get("pending_l4"), Mapping) else None
        if self._target_module_path is not None:
            if pending_p is not None and not pending_p.execution_terminal:
                err = self.ensure_candidate_bytes(path=self._target_module_path, source=pending_p.candidate_source, expected_sha256=pending_p.candidate_sha256)
            elif pending_l4 is not None and not bool(getattr(pending_l4, "execution_terminal", True)):
                err = self.ensure_candidate_bytes(path=self._target_module_path, source=inc_src, expected_sha256=str(getattr(pending_l4, "candidate_sha256", "")))
            else:
                err = self.write_source_and_verify(self._target_module_path, inc_src)
            if err:
                raise RuntimeError(err)
        return (
            inc_src, CandidateSurrogateResult.from_dict(payload["baseline_result"]), CandidateSurrogateResult.from_dict(payload["best_result"]),
            list(self.scoreboard_anchors), {str(x) for x in payload.get("frozen_regimes", ())}, creg, int(payload.get("stagnation_rounds", 0)),
            rounds, [int(x) for x in payload.get("promoted_rounds", ())], [int(x) for x in payload.get("rolled_back_rounds", ())],
            [int(x) for x in payload.get("escalated_rounds", ())], [str(x) for x in payload.get("admitted_rule_ids", ())],
            stag_map, last_unprom, pending_p, pending_l4,
        )

    def ensure_regime_diag_entry(self, regime_tag: str) -> dict[str, object]:
        """Return or initialize the per-regime stagnation diagnostic counters."""
        entry = self.regime_diag_stats.get(regime_tag)
        if entry is None:
            entry = {"rule_starvation_rounds": 0, "cost_prescreen_filtered_count": 0, "l0_reject_reasons": {}, "l1_compiler_penalty_rejects": 0, "l2_no_gain_rounds": 0, "l3_rollback_count": 0, "top_symptom_families": []}
            self.regime_diag_stats[regime_tag] = entry
        return entry

    def serialize_diag_stats(self) -> dict[str, dict[str, object]]:
        """Serialize per-regime stagnation diagnostics to a JSON-safe mapping."""
        out: dict[str, dict[str, object]] = {}
        for rtag, raw in sorted(self.regime_diag_stats.items()):
            rr = raw.get("l0_reject_reasons")
            tf = raw.get("top_symptom_families")
            out[str(rtag)] = {
                "rule_starvation_rounds": int(raw.get("rule_starvation_rounds", 0) or 0),
                "cost_prescreen_filtered_count": int(raw.get("cost_prescreen_filtered_count", 0) or 0),
                "l0_reject_reasons": {str(k): int(v) for k, v in sorted(rr.items())} if isinstance(rr, Mapping) else {},
                "l1_compiler_penalty_rejects": int(raw.get("l1_compiler_penalty_rejects", 0) or 0),
                "l2_no_gain_rounds": int(raw.get("l2_no_gain_rounds", 0) or 0),
                "l3_rollback_count": int(raw.get("l3_rollback_count", 0) or 0),
                "top_symptom_families": list(tf) if isinstance(tf, list) else [],
            }
        return out

    def restore_diag_stats(self, raw_stats: object) -> None:
        """Restore per-regime stagnation diagnostics from checkpoint payload."""
        self.regime_diag_stats.clear()
        if not isinstance(raw_stats, Mapping):
            return
        for rtag, raw in raw_stats.items():
            if not isinstance(raw, Mapping):
                continue
            rr = raw.get("l0_reject_reasons")
            tf = raw.get("top_symptom_families")
            self.regime_diag_stats[str(rtag)] = {
                "rule_starvation_rounds": int(raw.get("rule_starvation_rounds", 0) or 0),
                "cost_prescreen_filtered_count": int(raw.get("cost_prescreen_filtered_count", 0) or 0),
                "l0_reject_reasons": {str(k): int(v) for k, v in rr.items()} if isinstance(rr, Mapping) else {},
                "l1_compiler_penalty_rejects": int(raw.get("l1_compiler_penalty_rejects", 0) or 0),
                "l2_no_gain_rounds": int(raw.get("l2_no_gain_rounds", 0) or 0),
                "l3_rollback_count": int(raw.get("l3_rollback_count", 0) or 0),
                "top_symptom_families": [str(x) for x in tf] if isinstance(tf, Sequence) and not isinstance(tf, (str, bytes)) else [],
            }

    @staticmethod
    def classify_primary_stagnation_cause(
        *, rule_starvation_rounds: int, cost_prescreen_filtered_count: int, l0_rejects: int,
        l1_rejects: int, l2_no_gain_rounds: int, l3_rollback_count: int,
        final_roofline_eta: float, target_roofline_eta: float, stop_reason: str = "",
    ) -> str:
        """Classify the dominant bottleneck behind campaign termination or stall."""
        crej = l0_rejects + l1_rejects
        if stop_reason == "roofline_saturated" or (target_roofline_eta > 0.0 and final_roofline_eta >= target_roofline_eta):
            return "roofline_saturated"
        if l3_rollback_count > 0 and l3_rollback_count >= max(1, l2_no_gain_rounds // 2):
            return "l3_accuracy_instability"
        if cost_prescreen_filtered_count > 0 and cost_prescreen_filtered_count >= crej and (l2_no_gain_rounds > 0 or cost_prescreen_filtered_count >= 2):
            return "cost_prescreen_mismatch"
        if crej > 0 and crej >= max(rule_starvation_rounds, l2_no_gain_rounds):
            return "l0_l1_compile_wall"
        if rule_starvation_rounds > 0 and rule_starvation_rounds >= l2_no_gain_rounds:
            return "rule_starvation"
        if l2_no_gain_rounds > 0:
            return "hardware_unmodeled_stall"
        return "cost_prescreen_mismatch" if cost_prescreen_filtered_count > 0 else "hardware_unmodeled_stall"

    def build_stagnation_diagnosis(
        self, *, best_result: CandidateSurrogateResult, stop_reason: str,
        target_roofline_eta: float | None = None, regime_specs: Sequence[RegimeTargetSpec] | None = None,
    ) -> dict[str, object]:
        """Aggregate per-regime and overall stagnation diagnosis summary."""
        target_eta = float(target_roofline_eta) if target_roofline_eta is not None else (float(self._perf_target.target_roofline_eta) if self._perf_target else 0.0)
        eff_specs = regime_specs if regime_specs is not None else self._regime_specs
        tags = sorted(set(self.regime_diag_stats.keys()) | {s.regime_tag for s in eff_specs})
        reg_pay: dict[str, dict[str, object]] = {}
        t_st = t_pr = t_l0 = t_l1 = t_l2 = t_l3 = 0
        for rtag in tags:
            raw = self.regime_diag_stats.get(rtag) or {}
            rr = raw.get("l0_reject_reasons")
            reasons = {str(k): int(v) for k, v in sorted(rr.items())} if isinstance(rr, Mapping) else {}
            st, pr, l0c, l1c, l2n, l3r = (
                int(raw.get("rule_starvation_rounds", 0) or 0), int(raw.get("cost_prescreen_filtered_count", 0) or 0),
                sum(reasons.values()), int(raw.get("l1_compiler_penalty_rejects", 0) or 0),
                int(raw.get("l2_no_gain_rounds", 0) or 0), int(raw.get("l3_rollback_count", 0) or 0),
            )
            tf = raw.get("top_symptom_families")
            rres = best_result.for_regime(rtag)
            reta = float(rres.avg_roofline_eta if rres.anchors else best_result.avg_roofline_eta)
            cause = self.classify_primary_stagnation_cause(
                rule_starvation_rounds=st, cost_prescreen_filtered_count=pr, l0_rejects=l0c, l1_rejects=l1c,
                l2_no_gain_rounds=l2n, l3_rollback_count=l3r, final_roofline_eta=reta, target_roofline_eta=target_eta,
                stop_reason=(stop_reason if len(tags) <= 1 else ""),
            )
            reg_pay[rtag] = {
                "rule_starvation_rounds": st, "cost_prescreen_filtered_count": pr, "l0_reject_reasons": reasons,
                "l1_compiler_penalty_rejects": l1c, "l2_no_gain_rounds": l2n, "l3_rollback_count": l3r,
                "top_symptom_families": list(tf) if isinstance(tf, list) else [], "final_roofline_eta": round(reta, 6), "primary_stagnation_cause": cause,
            }
            t_st, t_pr, t_l0, t_l1, t_l2, t_l3 = t_st + st, t_pr + pr, t_l0 + l0c, t_l1 + l1c, t_l2 + l2n, t_l3 + l3r
        overall_eta = float(best_result.avg_roofline_eta)
        overall_cause = self.classify_primary_stagnation_cause(
            rule_starvation_rounds=t_st, cost_prescreen_filtered_count=t_pr, l0_rejects=t_l0, l1_rejects=t_l1,
            l2_no_gain_rounds=t_l2, l3_rollback_count=t_l3, final_roofline_eta=overall_eta, target_roofline_eta=target_eta, stop_reason=stop_reason,
        )
        return {"primary_stagnation_cause": overall_cause, "final_roofline_eta": round(overall_eta, 6), "regimes": reg_pay}

    @staticmethod
    def fold_regime_msprof_cache(
        msprof_by_case: Mapping[tuple[str, int], Mapping[str, float]],
        regime_msprof_cache: Mapping[tuple[str, str], Mapping[str, float]],
        active_anchors: Sequence[AnchorCaseSpec], *, active_regime_tag: str, single_regime: bool,
    ) -> dict[tuple[str, int], dict[str, float]]:
        """Fold legacy (sha, regime_tag) msprof entries into case-keyed telemetry."""
        merged = {k: dict(v) for k, v in msprof_by_case.items()}
        for (csha, rtag), metrics in regime_msprof_cache.items():
            matched = False
            for anc in active_anchors:
                if anc.regime_tag == rtag:
                    merged.setdefault((str(csha), int(anc.case_num)), dict(metrics))
                    matched = True
            if not matched and single_regime and rtag == active_regime_tag:
                for anc in active_anchors:
                    merged.setdefault((str(csha), int(anc.case_num)), dict(metrics))
        return merged

    @staticmethod
    def build_stone_parent_result(
        *, operator_name: str, parent_source: str, incumbent_source: str,
        best_result: CandidateSurrogateResult, cand_result: CandidateSurrogateResult, selected_stone: object | None,
    ) -> CandidateSurrogateResult | None:
        """Build paired parent result from incumbent best or selected stepping stone."""
        if parent_source == incumbent_source:
            return best_result
        clats = getattr(selected_stone, "case_latencies_us", None) if selected_stone else None
        if selected_stone is None or not isinstance(clats, Mapping) or not clats:
            return None
        ancs = tuple(
            AnchorEvalMetrics(
                case_num=int(a.case_num), regime_tag=a.regime_tag, accuracy_passed=True,
                cand_us=float(clats[int(a.case_num)]), ref_perf_us=float(a.ref_perf_us), t_hw_us=float(a.t_hw_us),
                speedup=(float(a.ref_perf_us) / float(clats[int(a.case_num)]) if a.ref_perf_us > 0.0 else 0.0),
                roofline_eta=(float(a.t_hw_us) / float(clats[int(a.case_num)]) if a.t_hw_us > 0.0 else 0.0),
                eq3_score=0.0,
            )
            for a in cand_result.anchors if int(a.case_num) in clats and float(clats[int(a.case_num)]) > 0.0
        )
        return aggregate_surrogate_result(str(getattr(selected_stone, "candidate_id", "")), operator_name, ancs) if ancs else None


__all__ = ["RoundReceipt", "TelemetryLedgerWriter", "atomic_write_text", "source_sha256"]
