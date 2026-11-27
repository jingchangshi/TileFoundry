"""Canonical L1 pass-boundary contracts and telemetry records for TileFoundry Ascend."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class CompileTaskSpec:
    """Canonical post-selector compilation unit specialization."""

    task_key: str
    builder_name: str
    static_kwargs: tuple[tuple[str, Any], ...]
    pass_configs: tuple[tuple[str, Any], ...]
    compile_flags: tuple[str, ...]

    @property
    def disables_vf_stack_ubuf(self) -> bool:
        """Whether this specialization disables the reserved VF stack UB buffer."""
        return any(
            "--cce-disable-vf-stack-reserved-ubuf" in flag
            for flag in self.compile_flags
        )


@dataclass(frozen=True)
class RegimeCompilePlan:
    """Post-selector 1-to-N compile plan across all regime cases."""

    regime_tag: str
    tasks_by_case_id: dict[str, tuple[CompileTaskSpec, ...]]
    unique_specs_by_key: dict[str, CompileTaskSpec]


@dataclass(frozen=True)
class StageBCompileReceipt:
    """Immutable receipt produced by Stage-B host-CPU cross-compilation."""

    passed: bool
    manifest_path: str
    max_stack_bytes: int = 0
    fault_code: str = ""
    diagnostic: str = ""
    elapsed_sec: float = 0.0


class CasePlanExtractor(Protocol):
    """Extract ordered CompileTaskSpecs emitted by candidate selector on one case."""

    def extract_case_tasks(
        self,
        module_source: str,
        case_id: str,
        case_spec: dict[str, Any],
    ) -> tuple[CompileTaskSpec, ...]: ...


@dataclass(frozen=True)
class CampaignStateSnapshot:
    """Immutable state passed into a single optimization round."""

    run_id: str
    round_index: int
    incumbent_source: str
    incumbent_sha256: str
    dispatch_floor_us: float
    active_regime_tag: str
    guardrail_case_ids: tuple[str, ...]
    stagnation_by_regime: dict[str, int]
    compile_wall_streak_by_regime: tuple[tuple[str, int], ...] = ()
    last_round_promoted: bool = True
    last_round_promoted_by_regime: tuple[tuple[str, bool], ...] = ()
    last_local_rel_gain_by_regime: tuple[tuple[str, float], ...] = ()
    incumbent_anchor_us_by_case: tuple[tuple[int, float], ...] = ()
    frozen_regimes: tuple[str, ...] = ()


class StageGatePort(Protocol):
    """Typed port replacing monkeypatched globals and owner overrides."""

    def screen_and_score_round(
        self,
        snapshot: CampaignStateSnapshot,
        candidate_sources: tuple[tuple[str, str], ...],
    ) -> tuple[tuple[str, float], ...]: ...


class RoundGatePort(Protocol):
    """Typed port for round LLM generation, L0/L1 filtering, and L2 evaluation."""

    async def execute_round_gates(
        self,
        batch: Any,
        working_anchors: Sequence[Any] = (),
        *,
        snapshot: CampaignStateSnapshot | None = None,
        parent_sha: str = "",
        parent_workload: Any = None,
    ) -> Any: ...

    async def cancel_pending_speculative(self) -> None: ...


class RoundContextPort(Protocol):
    """Typed port for building immutable per-round synthesis bundles."""

    def build_round_bundle(
        self,
        state: Any,
        *,
        round_idx: int,
        regime_spec: Any = None,
        working_anchors: Sequence[Any] | None = None,
        exclude_regime: str | None = None,
        creative_effort: str = "high",
    ) -> Any: ...


class CalibrationPort(Protocol):
    """Typed port for baseline evaluation, dispatch floor calibration, and profiling."""

    def build_snapshot(
        self,
        incumbent_source: str,
        active_anchors: Sequence[Any],
        regime_specs: Sequence[Any],
    ) -> Any: ...

    def collect_msprof_by_case(
        self,
        source: str,
        result: Any,
        active_anchors: Sequence[Any],
        target_regimes: Sequence[Any],
    ) -> dict[tuple[str, int], dict[str, float]]: ...


class RecoveryPort(Protocol):
    """Typed port for L3/L4 execution, card rotation, and pending recovery."""

    def resolve_pending_promotion(self, pending: Any, **kwargs: Any) -> Any: ...

    def resolve_pending_l4(
        self, pending: Any, *, incumbent_source: str = ""
    ) -> Any: ...

    def run_final_l4_tracked(
        self, *, incumbent_source: str, prior_stop_reason: str
    ) -> Any: ...


class TelemetryLedgerPort(Protocol):
    """Typed port for ExperienceLake, KB, funnel JSONL, and checkpoint persistence."""

    def record_surrogate_observations(self, **kwargs: Any) -> None: ...

    def emit_round_funnel_events(self, **kwargs: Any) -> Any: ...

    def save_checkpoint(self, **kwargs: Any) -> None: ...


class NpuExecutionPort(Protocol):
    """Typed port decoupling L4 NPU execution from L5 campaign orchestration."""

    def run_l2_batch(
        self,
        candidates: Sequence[tuple[str, Any]],
        anchors: Sequence[Any],
    ) -> Any: ...

    def run_l3_contract(
        self,
        module_path: Any,
        *,
        device_id: str = "",
        resume_execution_id: str = "",
        resume_run_id: str = "",
        on_execution_started: Callable[..., None] | None = None,
    ) -> Any: ...

    def run_l4_cannbench(
        self,
        module_path: Any,
        *,
        device_id: str = "",
        resume_execution_id: str = "",
        resume_run_id: str = "",
        on_execution_started: Callable[..., None] | None = None,
    ) -> Any: ...


class HardwareProbePort(Protocol):
    """Typed port for CANN environment setup and msprof micro-probe profiling."""

    def ensure_cann_include_cpath(
        self,
        env: dict[str, str] | None = None,
    ) -> dict[str, str] | list[str]: ...

    def run_msprof_detail(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class WorkerStepRecord:
    """Single worker attempt and funnel telemetry inside one evolution round."""

    worker_id: str
    worker_kind: str
    rule_id: str | None
    repaired: bool
    passed: bool
    failed_stage: str | None
    latency_us: float
    estimated_cycles: float
    dominant_pipe: str
    unexplained_gap_ratio: float
    hypothesis: str
    dsl_sha256: str

    def to_dict(self) -> dict[str, object]:
        """Serialize worker step telemetry as a JSON-compatible dictionary."""
        return {
            "worker_id": self.worker_id,
            "worker_kind": self.worker_kind,
            "rule_id": self.rule_id,
            "repaired": self.repaired,
            "passed": self.passed,
            "failed_stage": self.failed_stage,
            "latency_us": round(self.latency_us, 6),
            "estimated_cycles": round(self.estimated_cycles, 3),
            "dominant_pipe": self.dominant_pipe,
            "unexplained_gap_ratio": round(self.unexplained_gap_ratio, 6),
            "hypothesis": self.hypothesis,
            "dsl_sha256": self.dsl_sha256,
        }


@dataclass(frozen=True)
class EvolutionRoundRecord:
    """Per-round beam, pruning, and latency convergence snapshot."""

    round_idx: int
    parent_id: str
    parent_latency_us: float
    parent_dominant_pipe: str
    workers: tuple[WorkerStepRecord, ...]
    exploit_beam_ids: tuple[str, ...]
    stepping_stone_beam_ids: tuple[str, ...]
    pruned_rule_ids: tuple[str, ...]
    best_round_latency_us: float
    best_so_far_latency_us: float

    def to_dict(self) -> dict[str, object]:
        """Serialize evolution round snapshot to a JSON-compatible dictionary."""
        return {
            "round_idx": self.round_idx,
            "parent_id": self.parent_id,
            "parent_latency_us": round(self.parent_latency_us, 6),
            "parent_dominant_pipe": self.parent_dominant_pipe,
            "workers": [w.to_dict() for w in self.workers],
            "exploit_beam_ids": list(self.exploit_beam_ids),
            "stepping_stone_beam_ids": list(self.stepping_stone_beam_ids),
            "pruned_rule_ids": list(self.pruned_rule_ids),
            "best_round_latency_us": round(self.best_round_latency_us, 6),
            "best_so_far_latency_us": round(self.best_so_far_latency_us, 6),
        }


@dataclass
class EvolutionTrajectory:
    """Complete multi-round convergence record from unscheduled HIR to evolved DSL."""

    problem_name: str
    track: str
    total_rounds: int
    baseline_candidate_id: str
    baseline_latency_us: float
    baseline_estimated_cycles: float
    best_candidate_id: str
    best_latency_us: float
    best_estimated_cycles: float
    speedup_vs_baseline: float
    best_dsl_source: str
    rounds: tuple[EvolutionRoundRecord, ...] = field(default_factory=tuple)
    active_rules_after: tuple[str, ...] = field(default_factory=tuple)
    feature_vector: dict[str, object] = field(default_factory=dict)
    hir_blueprint_summary: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize complete evolution trajectory to a JSON-compatible dictionary."""
        return {
            "problem_name": self.problem_name,
            "track": self.track,
            "feature_vector": dict(self.feature_vector),
            "total_rounds": self.total_rounds,
            "baseline_candidate_id": self.baseline_candidate_id,
            "baseline_latency_us": round(self.baseline_latency_us, 6),
            "baseline_estimated_cycles": round(self.baseline_estimated_cycles, 3),
            "best_candidate_id": self.best_candidate_id,
            "best_latency_us": round(self.best_latency_us, 6),
            "best_estimated_cycles": round(self.best_estimated_cycles, 3),
            "speedup_vs_baseline": round(self.speedup_vs_baseline, 6),
            "best_dsl_source": self.best_dsl_source,
            "rounds": [r.to_dict() for r in self.rounds],
            "active_rules_after": list(self.active_rules_after),
            "hir_blueprint_summary": dict(self.hir_blueprint_summary),
        }

    def to_json(self) -> str:
        """Render formatted JSON representation of the evolution trajectory."""
        return json.dumps(self.to_dict(), indent=2)


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
    l3_resolution: str = "NOT_RUN"
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
            "round_idx": self.round_idx,
            "regime_tag": self.regime_tag,
            "builder_name": self.builder_name,
            "parent_lane": self.parent_lane,
            "creative_effort": self.creative_effort,
            "l0_l1_passed_ids": list(self.l0_l1_passed_ids),
            "l0_l1_failed": [list(item) for item in self.l0_l1_failed],
            "best_round_speedup": round(self.best_round_speedup, 6),
            "best_so_far_speedup": round(self.best_so_far_speedup, 6),
            "best_round_surrogate_score": round(self.best_round_surrogate_score, 6),
            "best_so_far_surrogate_score": round(self.best_so_far_surrogate_score, 6),
            "promoted": self.promoted,
            "l3_resolution": self.l3_resolution,
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
                if self.round_spearman_rho is not None
                else None
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
            {
                str(k): float(v)
                for k, v in raw_t.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if isinstance(raw_t, Mapping)
            else {}
        )
        raw_pw = payload.get("prewarm_stats")
        prewarm = (
            {
                str(k): int(v)
                for k, v in raw_pw.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if isinstance(raw_pw, Mapping)
            else {}
        )
        return cls(
            round_idx=int(payload["round_idx"]),
            regime_tag=str(payload["regime_tag"]),
            builder_name=str(payload["builder_name"]),
            parent_lane=str(payload["parent_lane"]),
            creative_effort=str(payload["creative_effort"]),
            l0_l1_passed_ids=tuple(str(x) for x in payload.get("l0_l1_passed_ids", ())),
            l0_l1_failed=failed,
            best_round_speedup=float(payload["best_round_speedup"]),
            best_so_far_speedup=float(payload["best_so_far_speedup"]),
            promoted=bool(payload["promoted"]),
            l3_resolution=str(payload.get("l3_resolution", "NOT_RUN")),
            stagnation_rounds_after=int(payload["stagnation_rounds_after"]),
            admitted_rule_ids=tuple(str(x) for x in payload.get("admitted_rule_ids", ())),
            best_round_surrogate_score=float(payload.get("best_round_surrogate_score", 0.0)),
            best_so_far_surrogate_score=float(payload.get("best_so_far_surrogate_score", 0.0)),
            pruned_rule_ids=tuple(str(x) for x in payload.get("pruned_rule_ids", ())),
            promotion_attempts=tuple(
                {str(k): str(v) for k, v in item.items()}
                for item in payload.get("promotion_attempts", ())
                if isinstance(item, Mapping)
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
            stage_timings_sec=timings,
            prewarm_stats=prewarm,
            cost_advisor_predictions=int(payload.get("cost_advisor_predictions", 0) or 0),
        )


__all__ = [
    "CalibrationPort",
    "CampaignRoundRecord",
    "CampaignStateSnapshot",
    "CasePlanExtractor",
    "CompileTaskSpec",
    "EvolutionRoundRecord",
    "EvolutionTrajectory",
    "HardwareProbePort",
    "NpuExecutionPort",
    "RecoveryPort",
    "RegimeCompilePlan",
    "RoundContextPort",
    "RoundGatePort",
    "StageBCompileReceipt",
    "StageGatePort",
    "TelemetryLedgerPort",
    "WorkerStepRecord",
]
