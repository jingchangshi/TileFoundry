"""Unattended performance-driven campaign runner with checkpointing and L3 rollback guard."""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import importlib.util
import inspect
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from tilefoundry_ascend.agent.campaign.anchor_specs import (
    AnchorCaseSpec,
    CannbenchOpSpec,
    CannbenchOpsRegistry,
    CasePhysicsRatios,
    PerfTargetSpec,
    RegimeTargetSpec,
    _has_flat_operator_layout,
    append_dynamic_anchor,
    build_anchor_var_bindings,
    compute_case_eq3_score,
    default_source_root,
    extract_trackb_cpp_source,
    get_default_anchors,
    get_default_perf_target,
    get_default_regime_specs,
    get_input_domain_envelope,
    get_operator_entrypoint,
    get_operator_module_relpath,
    lookup_case_dims,
    lookup_case_metadata,
    parse_simple_env_file,
    resolve_cannbench_repo,
    resolve_cannbench_tasks_root,
    resolve_regime_for_case,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    FAILURE_KIND_CANDIDATE,
    FAILURE_KIND_INFRASTRUCTURE,
    CandidateSurrogateResult,
    L0L1CandidateCheck,
    L2BatchResults,
    L3AcceptanceResult,
    L4CannbenchResult,
    bottleneck_anchor_for_regime,
    collect_bottleneck_msprof_via_agent_gates,
    run_l0_l1_round_filter,
    run_l1_compile_check,
    run_l3_full_contract_via_agent_gates,
    run_l4_official_cannbench_via_agent_gates,
    run_micro_probe_via_agent_gates,
    run_surrogate_batch_via_agent_gates,
)
from tilefoundry_ascend.agent.flywheel.ablation_distiller import FlywheelDistiller
from tilefoundry_ascend.agent.kb.bayesian_tracker import (
    BayesianRuleTracker,
    append_kb_event_shard,
    save_distilled_rule_file,
)
from tilefoundry_ascend.agent.kb.schema import (
    AscendCausalRule,
    build_predicate_context,
    load_seed_rules,
    normalize_runtime_backend,
)
from tilefoundry_ascend.agent.orchestrator.cpp_slice_protocol import (
    _LAUNCH_CONFIG_RE,
    _SEARCH_REPLACE_PATTERNS,
    _scoped_editable_units,
    _split_cpp_units,
    apply_scoped_trackb_patch,
    build_trackb_context_view,
    escalate_primfunc_to_trackb_tu,
    parse_scoped_search_replace_blocks,
    resolve_instantiated_trackb_kernel,
    select_active_trackb_kernel,
    splice_cpp_units_into_tu,
    strip_cpp_comments_and_strings,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    GateFunnel,
    StageAResult,
    StageBResult,
    evaluate_ascendc_candidate,
    run_stage_a_static,
    run_stage_b_ascendc_compile,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    GenerationRequest,
    KernelLLMBackend,
    create_llm_backend_from_env,
    extract_creative_payload,
    extract_python_dsl,
    sanitize_effort,
)
from tilefoundry_ascend.agent.orchestrator.llm_constraints import build_progressive_llm_constraints
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import (
    CREATIVE_FOCUS_HINTS,
    build_prescreen_summary_from_kwargs,
    summarize_excluded_mechanisms,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    BeamCandidate,
    StaticDynamicResidual,
    TwoLaneBeamSearch,
    compiler_summary_from_residual,
    compute_static_dynamic_residual,
    extract_structural_signature,
)
from tilefoundry_ascend.agent.orchestrator.tilesize_prescreen import (
    rank_stage_a_candidates_with_cost_model,
)
from tilefoundry_ascend.agent.router.capability_gate import evaluate_track_escalation
from tilefoundry_ascend.agent.router.isa_gap_scanner import scan_tilelang_pipe_gaps
from tilefoundry_ascend.ascend.cost_model import (
    BishengResourceSummary,
    CostFeatures,
    HybridAscendCostModel,
    build_cost_features_from_ledger,
)
from tilefoundry_ascend.ascend.cost_model.experience_lake import (
    CandidateArtifactRecord,
    CaseObservationRecord,
    ExperienceLake,
    compute_source_sha256,
    project_case_workload,
    resolve_default_experience_lake_dir,
)
from tilefoundry_ascend.ascend.cost_model.microarch_profile import merge_microarch_profiles
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import (
    HirBlueprintPacket,
    compile_hir_blueprint,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    ObjectivePhysicalLedger,
)
from tilefoundry_ascend.ascend.schedule.regime_solver import (
    RegimePiecewiseSolver,
    RegimePlanResult,
    recompute_regime_for_live_buffers,
)
from tilefoundry_ascend.ascend.sil.op_analyzer import (
    OpAdvisoryExtractor,
    resolve_canonical_topology_axes,
)
from tilefoundry_ascend.ascend.sil.pytorch_frontend import (
    PyTorchSemanticAnalyzer,
    RepresentativeCase,
)
from tilefoundry_ascend.ascend.sil.schema import (
    ConstraintRelaxationMask,
    RegimeSlice,
    TopologyAxes,
)
from tilefoundry_ascend.ascend.sil.verifier import verify_source_against_facets
from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS, AscendTarget
from tilefoundry_ascend.ascend.templates.registry import relax_facets_for_bottleneck

logger = logging.getLogger(__name__)


def _is_occupancy_blocked_env(failure_kind: str, blocked_env_category: str) -> bool:
    """Return True only for a signed occupancy BLOCKED_ENV preemption.

    A shared build lock or a machine-wide environment fault also ends as
    BLOCKED_ENV; rotating every card on those would only repeat the same
    failure. The receipt's ``errorCategory`` must say ``occupancy``.
    """
    return (
        failure_kind == FAILURE_KIND_INFRASTRUCTURE
        and blocked_env_category == "occupancy"
    )


@dataclass(frozen=True)
class CampaignConfig:
    """Configuration controlling stopping policy, escalation, and gate thresholds.

    ``device_id`` is the campaign's primary card and the first whole-L4
    candidate. ``device_ids`` names additional explicitly authorized physical
    cards for whole-L4 attempts: a terminal BLOCKED_ENV (exit 75) moves the
    complete official run to the next untried pool card before any backoff,
    while only a fully blocked round waits the bounded linear retry. The legacy
    ``retry_device_id`` keeps its single whole-run retry after a proven-terminal
    infrastructure failure and also joins the blocked-env pool. An empty pool
    therefore keeps the single-card bounded backoff. The pool may also be
    declared once in the managed env file under
    ``TILEFOUNDRY_NPU_CAMPAIGN_DEVICE_IDS``; entry points resolve that file into
    ``device_ids`` before constructing this config. ``round_regime_tags``
    forces one regime per 1-based round position and falls back to adaptive
    rotation for later rounds.
    """

    operator_name: str
    max_rounds: int = 12
    max_wall_clock_min: float = 180.0
    k_rule_workers: int = 2
    max_syntax_repairs: int = 1
    min_l2_gain_for_l3: float = 0.015
    min_surrogate_gain: float = 0.25
    min_flywheel_gain: float = 0.03
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
    enable_tilesize_prescreen: bool = False
    cost_prescreen_top_k: int | None = None
    exploit_effort: str = "medium"
    round_regime_tags: tuple[str, ...] = ()


_DEVICE_ID_PATTERN = re.compile(r"[0-9]+")
CAMPAIGN_DEVICE_IDS_ENV_KEY = "TILEFOUNDRY_NPU_CAMPAIGN_DEVICE_IDS"


def parse_device_ids(raw: str) -> tuple[str, ...]:
    """Parse one comma-separated ``--device-ids`` value into physical card IDs."""
    if not raw.strip():
        return ()
    parts = [part.strip() for part in raw.split(",")]
    if any(not part for part in parts):
        raise ValueError(f"device list contains an empty entry: {raw!r}")
    return tuple(parts)


def _canonical_device_id(value: str, *, field: str) -> str:
    """Canonicalize one physical NPU ID exactly as AgentGates records it."""
    text = value.strip()
    if not _DEVICE_ID_PATTERN.fullmatch(text):
        raise ValueError(f"{field} must be a nonnegative decimal device ID, got {value!r}")
    return str(int(text))


def resolve_campaign_device_plan(
    *,
    device_id: str,
    device_ids: tuple[str, ...],
    env_device_ids: tuple[str, ...],
    visible_device_ids: tuple[str, ...] = (),
) -> tuple[str, tuple[str, ...]]:
    """Resolve the primary card and ordered pool from CLI and managed env declarations.

    Explicit ``--device-ids`` cards override the env-file pool; otherwise the
    env-file pool is the declaration source. The primary is an explicitly
    supplied ``--device-id``, otherwise the first configured pool card, and only
    then the legacy ``"1"``. A CLI default therefore never injects a card that
    the managed pool did not declare.

    ``visible_device_ids`` is the parent process's physical visibility set. It
    only narrows the resolved plan: a card missing from it is never used, and a
    plan with no visible card is a configuration error rather than an implicit
    expansion to the full pool.
    """
    configured = device_ids or env_device_ids
    canonical_configured = tuple(
        _canonical_device_id(value, field=f"configured_device_ids[{index}]")
        for index, value in enumerate(configured)
    )
    explicit_primary = bool(device_id.strip())
    primary = _canonical_device_id(device_id, field="device_id") if explicit_primary else ""
    if not primary:
        primary = canonical_configured[0] if canonical_configured else "1"
    pool = [primary]
    for canonical in canonical_configured:
        if canonical not in pool:
            pool.append(canonical)

    if visible_device_ids:
        visible = {
            _canonical_device_id(value, field=f"visible_device_ids[{index}]")
            for index, value in enumerate(visible_device_ids)
        }
        if explicit_primary and primary not in visible:
            raise ValueError(
                f"requested device {primary!r} is not visible to this process "
                f"(visible: {', '.join(sorted(visible))})"
            )
        narrowed = [device for device in pool if device in visible]
        if not narrowed:
            raise ValueError(
                "configured device pool has no card visible to this process "
                f"(pool: {', '.join(pool)}; visible: {', '.join(sorted(visible))})"
            )
        primary = primary if primary in visible else narrowed[0]
        pool = [primary, *(device for device in narrowed if device != primary)]
    return primary, tuple(pool)


def _resolve_l4_device_pool(config: CampaignConfig) -> tuple[str, ...]:
    """Return the ordered physical-card pool for whole-L4 attempts.

    The campaign's primary card is always first; configured extra cards follow
    in declaration order and the legacy single-retry card is appended. The pool
    only narrows what AgentGates authorizes: every attempt is still a V2
    submission on that card, so the adapter keeps ownership of the lease and
    the idle preflight.
    """
    pool = [_canonical_device_id(config.device_id, field="device_id")]
    for index, value in enumerate(config.device_ids):
        device_id = _canonical_device_id(value, field=f"device_ids[{index}]")
        if device_id not in pool:
            pool.append(device_id)
    retry_device_id = config.retry_device_id.strip()
    if retry_device_id:
        retry_device = _canonical_device_id(retry_device_id, field="retry_device_id")
        if retry_device not in pool:
            pool.append(retry_device)
    return tuple(pool)


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
    rolled_back_l3: bool
    stagnation_rounds_after: int
    admitted_rule_ids: tuple[str, ...] = ()
    best_round_surrogate_score: float = 0.0
    best_so_far_surrogate_score: float = 0.0
    pruned_rule_ids: tuple[str, ...] = ()
    promotion_attempts: tuple[dict[str, str], ...] = ()
    trackb_evidence: tuple[dict[str, object], ...] = ()

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
            "rolled_back_l3": self.rolled_back_l3,
            "stagnation_rounds_after": self.stagnation_rounds_after,
            "admitted_rule_ids": list(self.admitted_rule_ids),
            "pruned_rule_ids": list(self.pruned_rule_ids),
            "promotion_attempts": [dict(item) for item in self.promotion_attempts],
            "trackb_evidence": [dict(item) for item in self.trackb_evidence],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CampaignRoundRecord:
        """Reconstruct CampaignRoundRecord from a JSON mapping."""
        failed_raw = payload.get("l0_l1_failed", ())
        failed = tuple(
            (str(item[0]), str(item[1]), str(item[2]))
            for item in failed_raw
            if isinstance(item, Sequence) and len(item) >= 3
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
            rolled_back_l3=bool(payload["rolled_back_l3"]),
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
                dict(item)
                for item in payload.get("trackb_evidence", ())
                if isinstance(item, Mapping)
            ),
        )


@dataclass(frozen=True)
class PendingPromotion:
    """Candidate held for promotion after an infrastructure L3 failure.

    The candidate source and its evaluation survive the failure so a resumed campaign
    can retry verification without regenerating it. Execution/run IDs and receipts are
    kept for diagnosis and never reused for a different candidate.
    """

    candidate_source: str
    candidate_sha256: str
    target_path: str
    round_idx: int
    regime_tag: str
    candidate_result: CandidateSurrogateResult
    candidate_id: str = ""
    builder_name: str = ""
    parent_lane: str = ""
    creative_effort: str = ""
    execution_id: str = ""
    run_id: str = ""
    receipt_path: str = ""
    error_message: str = ""
    attempts: tuple[dict[str, str], ...] = ()
    device_id: str = ""
    execution_terminal: bool = True
    retry_used: bool = False

    def to_dict(self) -> dict[str, object]:
        """Serialize the pending promotion record for the checkpoint and report."""
        return {
            "candidate_source": self.candidate_source,
            "candidate_sha256": self.candidate_sha256,
            "target_path": self.target_path,
            "round_idx": self.round_idx,
            "regime_tag": self.regime_tag,
            "candidate_result": self.candidate_result.to_dict(),
            "candidate_id": self.candidate_id,
            "builder_name": self.builder_name,
            "parent_lane": self.parent_lane,
            "creative_effort": self.creative_effort,
            "execution_id": self.execution_id,
            "run_id": self.run_id,
            "receipt_path": self.receipt_path,
            "error_message": self.error_message,
            "attempts": [dict(item) for item in self.attempts],
            "device_id": self.device_id,
            "execution_terminal": self.execution_terminal,
            "retry_used": self.retry_used,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PendingPromotion:
        """Reconstruct a pending promotion record from a checkpoint mapping."""
        raw_result = payload.get("candidate_result")
        if not isinstance(raw_result, Mapping):
            raise ValueError("pending_promotion.candidate_result must be a mapping")
        raw_attempts = payload.get("attempts", ())
        attempts = tuple(
            {str(k): str(v) for k, v in item.items()}
            for item in raw_attempts
            if isinstance(item, Mapping)
        )
        return cls(
            candidate_source=str(payload["candidate_source"]),
            candidate_sha256=str(payload.get("candidate_sha256", "")),
            target_path=str(payload.get("target_path", "")),
            round_idx=int(payload.get("round_idx", 0)),
            regime_tag=str(payload.get("regime_tag", "")),
            candidate_result=CandidateSurrogateResult.from_dict(raw_result),
            candidate_id=str(payload.get("candidate_id", "")),
            builder_name=str(payload.get("builder_name", "")),
            parent_lane=str(payload.get("parent_lane", "")),
            creative_effort=str(payload.get("creative_effort", "")),
            execution_id=str(payload.get("execution_id", "")),
            run_id=str(payload.get("run_id", "")),
            receipt_path=str(payload.get("receipt_path", "")),
            error_message=str(payload.get("error_message", "")),
            attempts=attempts,
            device_id=str(payload.get("device_id", "")),
            execution_terminal=bool(payload.get("execution_terminal", True)),
            retry_used=bool(payload.get("retry_used", False)),
        )


@dataclass(frozen=True)
class PendingL4:
    """An official L4 execution preserved across wait timeouts and process restarts.

    L4 is never split across cards: a retry on another card is a fresh, complete
    run. The original execution/run IDs, receipt, card and candidate digest stay
    attached so an operator can prove what happened. ``device_pool`` snapshots
    the cards this run may use, ``blocked_env_round_attempted_device_ids``
    records which cards the current blocked-env round already tried, and
    ``blocked_env_rounds_used``/``blocked_env_round_limit`` bound the number of
    complete blocked rounds, so a restart continues at the next untried card
    instead of spending a recorded round or retry twice. A ``None`` limit marks
    a legacy record without a stored quota; an explicit ``0`` forbids every
    additional round. ``blocked_env_category`` keeps the signed occupancy
    classification so a build-lock BLOCKED_ENV never rotates cards.
    ``device_rejected`` records that V2 refused the last submitted card as
    unauthorized; ``rejected_device_ids`` keeps every such card out of later
    attempts so an exhausted authorization terminates after a bounded sweep.
    """

    target_path: str
    candidate_relative: str
    candidate_sha256: str
    execution_id: str = ""
    run_id: str = ""
    receipt_path: str = ""
    device_id: str = ""
    state: str = ""
    error_message: str = ""
    failure_kind: str = FAILURE_KIND_INFRASTRUCTURE
    execution_terminal: bool = True
    retry_used: bool = False
    device_pool: tuple[str, ...] = ()
    blocked_env_round_attempted_device_ids: tuple[str, ...] = ()
    blocked_env_rounds_used: int = 0
    blocked_env_round_limit: int | None = None
    blocked_env_category: str = ""
    device_rejected: bool = False
    rejected_device_ids: tuple[str, ...] = ()
    attempts: tuple[dict[str, str], ...] = ()
    prior_stop_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialize the pending L4 record for the checkpoint and report."""
        return {
            "target_path": self.target_path,
            "candidate_relative": self.candidate_relative,
            "candidate_sha256": self.candidate_sha256,
            "execution_id": self.execution_id,
            "run_id": self.run_id,
            "receipt_path": self.receipt_path,
            "device_id": self.device_id,
            "state": self.state,
            "error_message": self.error_message,
            "failure_kind": self.failure_kind,
            "execution_terminal": self.execution_terminal,
            "retry_used": self.retry_used,
            "device_pool": list(self.device_pool),
            "blocked_env_round_attempted_device_ids": list(
                self.blocked_env_round_attempted_device_ids
            ),
            "blocked_env_rounds_used": self.blocked_env_rounds_used,
            "blocked_env_round_limit": self.blocked_env_round_limit,
            "blocked_env_category": self.blocked_env_category,
            "device_rejected": self.device_rejected,
            "rejected_device_ids": list(self.rejected_device_ids),
            "attempts": [dict(item) for item in self.attempts],
            "prior_stop_reason": self.prior_stop_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PendingL4:
        """Reconstruct a pending L4 record from a checkpoint mapping."""
        attempts = tuple(
            {str(k): str(v) for k, v in item.items()}
            for item in payload.get("attempts", ())
            if isinstance(item, Mapping)
        )
        return cls(
            target_path=str(payload.get("target_path", "")),
            candidate_relative=str(payload.get("candidate_relative", "")),
            candidate_sha256=str(payload.get("candidate_sha256", "")),
            execution_id=str(payload.get("execution_id", "")),
            run_id=str(payload.get("run_id", "")),
            receipt_path=str(payload.get("receipt_path", "")),
            device_id=str(payload.get("device_id", "")),
            state=str(payload.get("state", "")),
            error_message=str(payload.get("error_message", "")),
            failure_kind=str(payload.get("failure_kind", FAILURE_KIND_INFRASTRUCTURE)),
            execution_terminal=bool(payload.get("execution_terminal", True)),
            retry_used=bool(payload.get("retry_used", False)),
            device_pool=tuple(str(item) for item in payload.get("device_pool", ())),
            blocked_env_round_attempted_device_ids=tuple(
                str(item)
                for item in payload.get(
                    "blocked_env_round_attempted_device_ids",
                    (),
                )
            ),
            # Checkpoints written before the pool existed counted same-card
            # retries, which are exactly single-card blocked-env rounds.
            blocked_env_rounds_used=max(
                0,
                int(
                    payload.get(
                        "blocked_env_rounds_used",
                        payload.get("same_card_retries_used", 0),
                    )
                    or 0
                ),
            ),
            blocked_env_round_limit=(
                max(0, int(payload["blocked_env_round_limit"]))
                if payload.get("blocked_env_round_limit") is not None
                else (
                    max(0, int(payload["same_card_retry_limit"]))
                    if payload.get("same_card_retry_limit") is not None
                    else None
                )
            ),
            blocked_env_category=str(payload.get("blocked_env_category", "")),
            device_rejected=bool(payload.get("device_rejected", False)),
            rejected_device_ids=tuple(
                str(item) for item in payload.get("rejected_device_ids", ())
            ),
            attempts=attempts,
            prior_stop_reason=str(payload.get("prior_stop_reason", "")),
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
    pending_promotion: PendingPromotion | None = None
    pending_l4: PendingL4 | None = None
    promotion_blocked: bool = False

    def to_dict(self) -> dict[str, object]:
        """Serialize CampaignReport into a JSON-compatible dictionary."""
        return {
            "operator_name": self.operator_name,
            "stop_reason": self.stop_reason,
            "total_rounds": self.total_rounds,
            "baseline_result": self.baseline_result.to_dict(),
            "best_result": self.best_result.to_dict(),
            "rounds": [r.to_dict() for r in self.rounds],
            "promoted_rounds": list(self.promoted_rounds),
            "rolled_back_rounds": list(self.rolled_back_rounds),
            "escalated_rounds": list(self.escalated_rounds),
            "frozen_regimes": list(self.frozen_regimes),
            "active_anchors": [a.to_dict() for a in self.active_anchors],
            "admitted_rule_ids": list(self.admitted_rule_ids),
            "l4_result": self.l4_result.to_dict() if self.l4_result else None,
            "pending_promotion": (
                self.pending_promotion.to_dict() if self.pending_promotion else None
            ),
            "pending_l4": self.pending_l4.to_dict() if self.pending_l4 else None,
            "promotion_blocked": self.promotion_blocked,
        }


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


_FORBIDDEN_SLICE_SYMBOLS: frozenset[str] = frozenset(
    {
        "attention_residual",
        "mla",
        "_get_kernel",
        "_get_combine_kernel",
        "_ascend_target_ctx",
        "_KERNEL_CACHE",
        "_DISPATCH_CACHE",
        "_WORKSPACE_CACHE",
    }
)

_HIR_DIM_ALIASES: dict[str, str] = {
    "N": "n_depth",
    "B": "b_dim",
    "S": "s_dim",
    "S_kv": "s_kv",
    "D": "d_dim",
    "N_q": "n_q",
    "N_kv": "n_kv",
    "D_nope": "d_nope",
    "D_rope": "d_rope",
    "inner": "inner_dim",
}


@dataclass(frozen=True)
class ResolvedRegimeConfig:
    """Resolved builder, tile hyperparameters, and pass/compile flags for a regime."""

    builder_name: str
    builder_kwargs: dict[str, object]
    pass_configs: dict[str, object]
    compile_flags: tuple[str, ...]
    selector_name: str = ""
    combine_kwargs: dict[str, object] = field(default_factory=dict)


def _filter_callable_kwargs(fn: object, kwargs: Mapping[str, object]) -> dict[str, object]:
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _norm_flag_seq(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return text.split() if text else []
    if isinstance(raw, (tuple, list)) or (
        hasattr(raw, "__iter__") and not isinstance(raw, (bytes, bytearray, Mapping))
    ):
        out: list[str] = []
        for item in raw:
            val = str(getattr(item, "value", item)).strip()
            if val and val not in out:
                out.append(val)
        return out
    text = str(raw).strip()
    return [text] if text else []


def resolve_regime_config(
    module_source_or_path: str | Path,
    regime_spec: RegimeTargetSpec,
    dims: Mapping[str, object] | None = None,
) -> ResolvedRegimeConfig:
    """Invoke regime config_selector_fn dynamically to resolve builder kwargs and flags."""
    if isinstance(module_source_or_path, Path):
        mod_path = module_source_or_path.resolve()
        src_text = mod_path.read_text(encoding="utf-8")
        is_existing_file = True
    else:
        raw_str = str(module_source_or_path)
        is_existing_file = "\n" not in raw_str and Path(raw_str).is_file()
        if is_existing_file:
            mod_path = Path(raw_str).resolve()
            src_text = mod_path.read_text(encoding="utf-8")
        else:
            mod_path = None
            src_text = raw_str

    def _load_and_resolve(file_path: Path) -> ResolvedRegimeConfig:
        spec = importlib.util.spec_from_file_location("_tf_resolve_cfg", str(file_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load module spec from {file_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        context: dict[str, object] = dict(regime_spec.sample_builder_kwargs)
        for h_key, h_val in regime_spec.representative_hir_dims.items():
            mapped = _HIR_DIM_ALIASES.get(str(h_key), str(h_key))
            if mapped not in context:
                context[mapped] = h_val
        if dims:
            for d_key, d_val in dims.items():
                mapped = _HIR_DIM_ALIASES.get(str(d_key), str(d_key))
                context[mapped] = d_val
        if "half_dim" not in context and isinstance(context.get("d_dim"), int):
            context["half_dim"] = max(1, int(context["d_dim"]) // 2)

        pass_configs: dict[str, object] = {}
        compile_flags: list[str] = []
        combine_kwargs: dict[str, object] = {}
        selector_name = regime_spec.config_selector_fn
        if not (selector_name and callable(getattr(mod, selector_name, None))):
            for cand_sym in regime_spec.editable_symbols:
                if cand_sym.startswith("_select_") and callable(getattr(mod, cand_sym, None)):
                    selector_name = cand_sym
                    break
        if selector_name and callable(getattr(mod, selector_name, None)):
            selector_fn = getattr(mod, selector_name)
            sel_args = _filter_callable_kwargs(selector_fn, context)
            sel_out = selector_fn(**sel_args)
            if isinstance(sel_out, Mapping):
                raw_pc = sel_out.get("pass_configs")
                if isinstance(raw_pc, Mapping):
                    for k, v in raw_pc.items():
                        if str(k) == "tl.device_compile_flags":
                            for fl in _norm_flag_seq(v):
                                if fl not in compile_flags:
                                    compile_flags.append(fl)
                        else:
                            pass_configs[str(k)] = v
                elif raw_pc is not None:
                    for fl in _norm_flag_seq(raw_pc):
                        if fl not in compile_flags:
                            compile_flags.append(fl)
                for fl in _norm_flag_seq(sel_out.get("compile_flags")):
                    if fl not in compile_flags:
                        compile_flags.append(fl)
                for k, v in sel_out.items():
                    if k not in (
                        "pass_configs",
                        "compile_flags",
                        "builder_kwargs",
                        "combine_kwargs",
                    ):
                        context[str(k)] = v
                raw_bk = sel_out.get("builder_kwargs")
                if isinstance(raw_bk, Mapping):
                    for k, v in raw_bk.items():
                        context[str(k)] = v
                raw_ck = sel_out.get("combine_kwargs")
                if isinstance(raw_ck, Mapping):
                    combine_kwargs = {str(k): v for k, v in raw_ck.items()}

        builder_name = ""
        for cand in regime_spec.builder_candidates:
            if callable(getattr(mod, cand, None)):
                builder_name = cand
                break
        if not builder_name:
            for cand in regime_spec.editable_symbols:
                if cand != selector_name and callable(getattr(mod, cand, None)):
                    builder_name = cand
                    break
        if not builder_name:
            raise ValueError(
                f"No builder callable matching {regime_spec.builder_candidates} in module."
            )
        builder_fn = getattr(mod, builder_name)
        builder_kwargs = _filter_callable_kwargs(builder_fn, context)
        return ResolvedRegimeConfig(
            builder_name=builder_name,
            builder_kwargs=builder_kwargs,
            pass_configs=pass_configs,
            compile_flags=tuple(compile_flags),
            selector_name=selector_name,
            combine_kwargs=combine_kwargs,
        )

    if mod_path is not None and is_existing_file:
        return _load_and_resolve(mod_path)
    with tempfile.TemporaryDirectory(prefix="tf_resolve_cfg_") as tmp_dir:
        tmp_file = Path(tmp_dir) / "module_src.py"
        tmp_file.write_text(src_text, encoding="utf-8")
        return _load_and_resolve(tmp_file)


@dataclass(frozen=True)
class _TopLevelSymbol:
    name: str
    kind: str
    start: int
    end: int


def _stmt_line_span(node: ast.stmt) -> tuple[int, int]:
    decs = getattr(node, "decorator_list", ())
    start = min((dec.lineno for dec in decs), default=node.lineno)
    end = node.end_lineno or node.lineno
    return start, end


def _collect_top_level_symbols(tree: ast.Module) -> list[_TopLevelSymbol]:
    items: list[_TopLevelSymbol] = []
    for node in tree.body:
        start, end = _stmt_line_span(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            items.append(_TopLevelSymbol(node.name, "func", start, end))
        elif isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                items.append(_TopLevelSymbol(node.targets[0].id, "assign", start, end))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                items.append(_TopLevelSymbol(node.target.id, "assign", start, end))
    return items


def _extract_trackb_cpp_source(
    module_source: str,
    regime_spec: RegimeTargetSpec,
) -> tuple[str, str] | None:
    """Return `(symbol_name, cpp_source)` for the first Track-B C++20 string constant."""
    return extract_trackb_cpp_source(module_source, regime_spec.trackb_symbols)


def _top_level_symbol_sources(module_source: str) -> dict[str, str]:
    tree = ast.parse(module_source)
    lines = module_source.splitlines(keepends=True)
    return {
        sym.name: "".join(lines[sym.start - 1 : sym.end]).strip()
        for sym in _collect_top_level_symbols(tree)
    }


_TRACKB_FOLDING_MARKERS: tuple[str, ...] = (
    "READ-ONLY SIBLING KERNEL",
    "READ-ONLY HELPER",
    "MIDDLE BODY OMITTED",
)


def _reject_trackb_folding_markers(candidate_text: str, regime_tag: str) -> None:
    """Reject a Track-B reply that echoes truncated-context folding markers."""
    found = [marker for marker in _TRACKB_FOLDING_MARKERS if marker in candidate_text]
    if found:
        raise ValueError(
            f"Track-B candidate for regime {regime_tag!r} echoes truncated context markers "
            f"{found}; reply with SEARCH/REPLACE hunks or the complete TU instead"
        )


def _strip_trackb_patch_hunks(patch_text: str) -> str:
    """Return *patch_text* with its SEARCH/REPLACE blocks removed."""
    residual = patch_text
    for pattern in _SEARCH_REPLACE_PATTERNS:
        residual = pattern.sub("", residual)
    return residual.strip()


def _looks_like_bare_trackb_cpp(candidate_text: str) -> bool:
    """Return True when a Track-B reply carries bare C++ units rather than Python."""
    if re.search(r"(?m)^\s*(?:async\s+)?def\s+", candidate_text):
        return False
    _, units = _split_cpp_units(candidate_text)
    return any(unit.kind in ("kernel", "helper") and unit.name for unit in units)


def _parses_as_python(candidate_text: str) -> bool:
    """Return True when *candidate_text* is a syntactically valid Python snippet."""
    try:
        ast.parse(candidate_text)
    except SyntaxError:
        return False
    return True


def _render_trackb_tu_from_module(
    module_source: str,
    cpp_source: str,
    regime_spec: RegimeTargetSpec,
    *,
    require_renderer: bool,
) -> str:
    """Render a module's Track-B launcher with the shared dimension merge.

    The same sample, HIR, anchor, selector, and alias dimensions feed the
    runner and the Stage-0 gate, so both compile the bytes a regime really
    launches. A module without a renderer keeps its raw translation unit
    unless *require_renderer* marks the cold-start ABI as mandatory.
    """
    tree = ast.parse(module_source)
    if _find_top_level_func(tree, ("_render_trackb_tu",)) is None:
        if require_renderer:
            raise ValueError(
                "Track-B seed must declare def _render_trackb_tu(**dims: object) -> str"
            )
        return cpp_source
    dims: dict[str, object] = {
        **regime_spec.sample_builder_kwargs,
        **regime_spec.representative_hir_dims,
        **regime_spec.representative_anchor_dims,
    }
    try:
        resolved = resolve_regime_config(module_source, regime_spec, dims=dims)
        alias_sources = {
            target: source for source, target in _HIR_DIM_ALIASES.items()
        }
        for key, value in resolved.builder_kwargs.items():
            source = alias_sources.get(key)
            if source is not None and source in dims and dims[source] != value:
                continue
            dims.setdefault(key, value)
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    dims["trackb_src"] = cpp_source
    with tempfile.TemporaryDirectory(prefix="tf_trackb_inst_") as tmp_dir:
        module_file = Path(tmp_dir) / "candidate_mod.py"
        module_file.write_text(module_source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("_tf_trackb_render", str(module_file))
        if spec is None or spec.loader is None:
            if require_renderer:
                raise RuntimeError("cannot import the seed module to render its Track-B TU")
            return cpp_source
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        render = getattr(module, "_render_trackb_tu", None)
        if not callable(render):
            if require_renderer:
                raise ValueError(
                    "Track-B seed must declare def _render_trackb_tu(**dims: object) -> str"
                )
            return cpp_source
        rendered = render(**dims)
    if not isinstance(rendered, str) or not rendered.strip():
        if require_renderer:
            raise ValueError("_render_trackb_tu returned an empty C++ translation unit")
        return cpp_source
    return rendered


def _active_trackb_kernel(
    instantiated_tu: str,
    raw_cpp_source: str,
    *,
    regime_tag: str,
) -> str:
    """Resolve the launched kernel from instantiation truth, never from name heuristics."""
    resolved = resolve_instantiated_trackb_kernel(instantiated_tu, raw_cpp_source)
    if resolved is not None:
        return resolved
    kernels = sorted(_trackb_kernel_names(raw_cpp_source))
    if len(kernels) == 1:
        return kernels[0]
    if not kernels:
        raise ValueError(
            f"Track-B instantiation for regime {regime_tag!r} contains no __global__ kernel"
        )
    launched, primary = _launched_trackb_kernels(
        instantiated_tu,
        raw_cpp_source,
        set(kernels),
    )
    if len(primary) == 1:
        return next(iter(primary))
    if not launched:
        raise ValueError(
            f"Track-B instantiation for regime {regime_tag!r} contains multiple kernels "
            f"{kernels} and no launcher stub; the instantiated translation unit must launch "
            "exactly one primary kernel"
        )
    raise ValueError(
        f"Track-B instantiation for regime {regime_tag!r} launches kernels {sorted(launched)} "
        f"with no unique top-level launch target among {sorted(primary)}; the launcher stub must "
        "launch exactly one primary kernel outside conditional branches"
    )


def _launched_trackb_kernels(
    instantiated_tu: str,
    raw_cpp_source: str,
    kernels: set[str],
) -> tuple[set[str], set[str]]:
    """Return all launched kernels and those launched at the launcher's top level."""
    masked = strip_cpp_comments_and_strings(instantiated_tu)
    launch_region = (
        masked[len(raw_cpp_source) :]
        if instantiated_tu.startswith(raw_cpp_source)
        else masked
    )
    launches: list[tuple[int, str]] = []
    depth = 0
    scanned = 0
    for match in _LAUNCH_CONFIG_RE.finditer(launch_region):
        depth += launch_region.count("{", scanned, match.start()) - launch_region.count(
            "}", scanned, match.start()
        )
        scanned = match.start()
        name = match.group(1)
        if name in kernels:
            launches.append((depth, name))
    if not launches:
        return set(), set()
    shallowest = min(depth for depth, _name in launches)
    return (
        {name for _depth, name in launches},
        {name for depth, name in launches if depth == shallowest},
    )


def _trackb_kernel_names(cpp_source: str) -> set[str]:
    _, units = _split_cpp_units(cpp_source)
    return {unit.name for unit in units if unit.kind == "kernel" and unit.name}


def _trackb_units_by_name(cpp_source: str) -> dict[str, str]:
    """Return named top-level Track-B unit text keyed by symbol name."""
    _, units = _split_cpp_units(cpp_source)
    return {unit.name: unit.text for unit in units if unit.name}


def _normalized_trackb_unit(text: str) -> str:
    """Collapse whitespace so pure reformatting is not treated as a unit edit."""
    return " ".join(text.split())


def _assert_trackb_replacement_scoped(
    orig_cpp: str,
    new_cpp: str,
    regime_spec: RegimeTargetSpec,
    *,
    module_source: str,
    active_kernel: str | None,
) -> None:
    """Reject a full-TU replacement that edits frozen siblings or shared helpers."""
    orig_units = _trackb_units_by_name(orig_cpp)
    new_units = _trackb_units_by_name(new_cpp)
    if active_kernel is None:
        active_kernel = select_active_trackb_kernel(
            orig_cpp,
            regime_tag=regime_spec.regime_tag,
            module_source=module_source,
        )
    if active_kernel is None:
        raise ValueError(
            f"Track-B full replacement for regime {regime_spec.regime_tag!r} has no resolvable "
            "active kernel; reply with SEARCH/REPLACE hunks or one bare C++ unit"
        )
    editable = {
        unit.name
        for unit in _scoped_editable_units(
            orig_cpp,
            active_kernel,
            allow_exclusive_helpers=True,
        )
    }
    dropped = sorted(set(orig_units) - set(new_units))
    if dropped:
        raise ValueError(
            f"Track-B replacement for regime {regime_spec.regime_tag!r} drops units {dropped}; "
            "reply with SEARCH/REPLACE hunks or one bare C++ unit per changed function"
        )
    added = sorted(set(new_units) - set(orig_units))
    if added:
        raise ValueError(
            f"Track-B replacement for regime {regime_spec.regime_tag!r} adds units {added}; "
            "reply with SEARCH/REPLACE hunks or one bare C++ unit per changed function"
        )
    changed = sorted(
        name
        for name in orig_units
        if _normalized_trackb_unit(orig_units[name])
        != _normalized_trackb_unit(new_units[name])
    )
    frozen = sorted(name for name in changed if name not in editable)
    if frozen:
        raise ValueError(
            f"Track-B replacement for regime {regime_spec.regime_tag!r} edits frozen sibling "
            f"kernels or shared helpers {frozen}; only active kernel {active_kernel!r} and its "
            "exclusive helpers may change"
        )


def _case_ratio_predicate_entries(ratios: CasePhysicsRatios | None) -> dict[str, object]:
    """Return collision-free dimensionless predicate keys for the representative case."""
    if ratios is None:
        return {}
    return {
        f"case_{key.removeprefix('case_')}": value for key, value in ratios.to_dict().items()
    }


def _detect_trackb_dead_code_edit(
    parent_source: str,
    candidate_source: str,
    regime_spec: RegimeTargetSpec,
) -> str | None:
    """Reject candidates that edit only inactive TileLang fallbacks in a Track-B regime."""
    if (
        regime_spec.runtime_backend not in ("ascendc_trackb", "hybrid_trackb")
        or not regime_spec.trackb_symbols
    ):
        return None
    parent_syms = _top_level_symbol_sources(parent_source)
    cand_syms = _top_level_symbol_sources(candidate_source)
    if not parent_syms or not cand_syms:
        return None
    changed = {
        name
        for name in (set(parent_syms) | set(cand_syms))
        if parent_syms.get(name, "") != cand_syms.get(name, "")
    }
    if not changed:
        return None
    active_syms = set(regime_spec.trackb_symbols)
    if regime_spec.config_selector_fn:
        active_syms.add(regime_spec.config_selector_fn)
    if not (changed & active_syms):
        return (
            f"DEAD_CODE_TRACKB_EDIT: regime {regime_spec.regime_tag!r} executes "
            f"runtime_backend={regime_spec.runtime_backend} via {list(regime_spec.trackb_symbols)}; "
            f"candidate modified only inactive fallback symbols {sorted(changed)}."
        )
    return None


def _resolve_cann_root_from_env_file(env_file: str | Path | None) -> Path | None:
    if env_file is None:
        return None
    p = Path(env_file)
    if not p.is_file():
        return None
    vals = parse_simple_env_file(p)
    setup_raw = vals.get("TILEFOUNDRY_NPU_CANN_SETUP", "").strip()
    if not setup_raw:
        return None
    parent = Path(setup_raw).expanduser().parent
    return parent if parent.is_dir() else None


def _find_top_level_func(tree: ast.Module, names: Sequence[str]) -> ast.FunctionDef | None:
    by_name = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    for name in names:
        if name in by_name:
            return by_name[name]
    return None


def _func_line_span(node: ast.FunctionDef) -> tuple[int, int]:
    start = min((dec.lineno for dec in node.decorator_list), default=node.lineno)
    end = node.end_lineno or node.lineno
    return start, end


def _resolve_operator_entrypoint_safe(operator_name: str) -> str:
    try:
        return get_operator_entrypoint(operator_name)
    except (KeyError, FileNotFoundError):
        return str(operator_name).rsplit("/", 1)[-1].strip()


def _other_regime_forbidden_symbols(
    regime_spec: RegimeTargetSpec,
    *,
    operator_entrypoint: str = "",
    sibling_specs: Sequence[RegimeTargetSpec] = (),
) -> set[str]:
    forbidden: set[str] = set(_FORBIDDEN_SLICE_SYMBOLS)
    if operator_entrypoint:
        forbidden.add(operator_entrypoint)
    forbidden.add("kernel")
    own_editable = set(regime_spec.editable_symbols)
    for spec in sibling_specs:
        if spec.regime_tag == regime_spec.regime_tag:
            continue
        for sym in spec.editable_symbols:
            if sym not in own_editable:
                forbidden.add(sym)
        if spec.config_selector_fn and spec.config_selector_fn not in own_editable:
            forbidden.add(spec.config_selector_fn)
    forbidden.difference_update(own_editable)
    return forbidden


def extract_regime_snippet(
    module_source: str,
    regime_spec: RegimeTargetSpec,
) -> tuple[str, str]:
    """Extract target regime multi-symbol slice (selector, builders, helpers) cleanly."""
    tree = ast.parse(module_source)
    lines = module_source.splitlines(keepends=True)

    if regime_spec.editable_symbols:
        symbols = _collect_top_level_symbols(tree)
        editable_set = set(regime_spec.editable_symbols)
        prefix = regime_spec.allowed_helper_prefix
        matched = [
            sym
            for sym in symbols
            if sym.name in editable_set or (prefix and sym.name.startswith(prefix))
        ]
        if matched:
            func_names = {sym.name for sym in matched if sym.kind == "func"}
            primary = ""
            for cand in regime_spec.builder_candidates:
                if cand in func_names:
                    primary = cand
                    break
            if not primary:
                primary = next(
                    (sym.name for sym in matched if sym.kind == "func"),
                    matched[0].name,
                )
            blocks = [
                "".join(lines[sym.start - 1 : sym.end]).rstrip()
                for sym in matched
            ]
            return primary, "\n\n".join(blocks).rstrip() + "\n"

    node = _find_top_level_func(tree, regime_spec.builder_candidates)
    if node is None:
        raise ValueError(
            f"No builder function matching {regime_spec.builder_candidates} found in module."
        )
    start, end = _func_line_span(node)
    return node.name, "".join(lines[start - 1 : end]).rstrip() + "\n"


def _is_prim_func_decorator(dec: ast.expr) -> bool:
    if isinstance(dec, ast.Name):
        return dec.id == "prim_func"
    if isinstance(dec, ast.Attribute):
        return dec.attr == "prim_func"
    if isinstance(dec, ast.Call):
        return _is_prim_func_decorator(dec.func)
    return False


def _extract_func_signature_header(lines: Sequence[str], node: ast.FunctionDef) -> str:
    if node.body and node.body[0].lineno > node.lineno:
        end_idx = node.body[0].lineno - 1
        raw = "".join(lines[node.lineno - 1 : end_idx]).rstrip()
    else:
        raw = "".join(lines[node.lineno - 1 : node.lineno]).split(":", 1)[0].rstrip() + ":"
    return textwrap.dedent(raw).strip()


def _extract_regime_abi_context(module_source: str, regime_spec: RegimeTargetSpec) -> str:
    """Extract slice, helper builder, and @T.prim_func ABI signatures for the target regime."""
    tree = ast.parse(module_source)
    lines = module_source.splitlines(keepends=True)
    top_funcs = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    editable_set = set(regime_spec.editable_symbols)
    prefix = regime_spec.allowed_helper_prefix
    seed_names = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and (
            node.name in editable_set
            or (prefix and node.name.startswith(prefix))
            or node.name in regime_spec.builder_candidates
        )
    ]
    if not seed_names:
        return ""

    visited: set[str] = set()
    ordered_slice_funcs: list[ast.FunctionDef] = []
    ordered_helper_funcs: list[ast.FunctionDef] = []
    prim_func_sigs: list[str] = []
    seen_prim_sigs: set[str] = set()
    queue = list(seed_names)
    while queue:
        fname = queue.pop(0)
        if fname in visited:
            continue
        visited.add(fname)
        fnode = top_funcs.get(fname)
        if fnode is None:
            continue
        is_prefix_helper = bool(prefix and fname.startswith(prefix))
        if fname in editable_set or fname in regime_spec.builder_candidates:
            ordered_slice_funcs.append(fnode)
        elif not is_prefix_helper:
            ordered_helper_funcs.append(fnode)
        for sub in ast.walk(fnode):
            if isinstance(sub, ast.FunctionDef) and sub is not fnode:
                if any(_is_prim_func_decorator(dec) for dec in sub.decorator_list):
                    sig = "@T.prim_func\n" + _extract_func_signature_header(lines, sub)
                    if sig not in seen_prim_sigs:
                        seen_prim_sigs.add(sig)
                        prim_func_sigs.append(sig)
            elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                callee = sub.func.id
                if callee in top_funcs and callee not in visited:
                    queue.append(callee)

    blocks: list[str] = []
    for hnode in ordered_helper_funcs:
        blocks.append(_extract_func_signature_header(lines, hnode))
    for snode in ordered_slice_funcs:
        blocks.append(_extract_func_signature_header(lines, snode))
    blocks.extend(prim_func_sigs)
    if not blocks:
        return ""
    sigs_text = "\n\n".join(blocks)
    return (
        "### IMMUTABLE HOST / PRIMFUNC ABI CONTRACT\n"
        f"```python\n{sigs_text}\n```\n"
        "Any custom @T.prim_func returned for this slice MUST keep the exact tensor/scalar "
        "parameter order, ranks, and dtypes above expected by _get_kernel (do NOT substitute "
        "the HIR forward(...) parameter list, and do NOT import tilelang.language as T or use "
        "T.Scope). _select_*_config() may return optional 'builder_kwargs': {...} forwarded to "
        "custom builders for compile-time constexpr specialization.\n\n"
    )


def apply_ast_patch(
    module_source: str,
    target_func: str,
    candidate_dsl: str,
) -> str:
    """Robust SEARCH/REPLACE patcher using Python AST anchors."""
    orig_tree = ast.parse(module_source)
    cand_tree = ast.parse(candidate_dsl)

    cand_node = _find_top_level_func(cand_tree, (target_func,))
    if cand_node is None:
        raise ValueError(f"Candidate source must define function {target_func!r}.")

    cand_lines = candidate_dsl.splitlines(keepends=True)
    c_start, c_end = _func_line_span(cand_node)
    replacement = "".join(cand_lines[c_start - 1 : c_end])
    if not replacement.endswith("\n"):
        replacement += "\n"

    orig_node = _find_top_level_func(orig_tree, (target_func,))
    if orig_node is None:
        raise ValueError(f"Original module missing target function {target_func!r}.")

    o_start, o_end = _func_line_span(orig_node)
    orig_lines = module_source.splitlines(keepends=True)

    patched = "".join([*orig_lines[: o_start - 1], replacement, *orig_lines[o_end:]])
    ast.parse(patched)
    return patched


def _collect_top_level_ast_nodes(tree: ast.Module) -> dict[str, ast.stmt]:
    nodes: dict[str, ast.stmt] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes[node.name] = node
        elif isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                nodes[node.targets[0].id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            nodes[node.target.id] = node
    return nodes


def _collect_referenced_helper_names(
    merged_ast_by_name: Mapping[str, ast.stmt],
    *,
    helper_names: set[str],
    seed_roots: Sequence[str],
) -> set[str]:
    live: set[str] = set()
    queue = list(seed_roots)
    visited: set[str] = set()
    while queue:
        sym = queue.pop(0)
        if sym in visited:
            continue
        visited.add(sym)
        node = merged_ast_by_name.get(sym)
        if node is None:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                ref = sub.id
                if ref in helper_names and ref not in live:
                    live.add(ref)
                    if ref not in visited:
                        queue.append(ref)
    return live


def _ast_char_offset(lines: Sequence[str], lineno: int, col_offset: int) -> int:
    prefix = sum(len(lines[idx]) for idx in range(lineno - 1))
    line_bytes = lines[lineno - 1].encode("utf-8")
    return prefix + len(line_bytes[:col_offset].decode("utf-8"))


def _format_raw_cpp_literal(cpp_text: str) -> str:
    if '"""' not in cpp_text and not cpp_text.endswith('"'):
        return f'r"""{cpp_text}"""'
    return repr(cpp_text)


def _find_top_level_str_assign_span(
    tree: ast.Module,
    lines: Sequence[str],
    target_names: set[str],
) -> list[tuple[str, str, int, int]]:
    hits: list[tuple[str, str, int, int]] = []
    for node in tree.body:
        name: str | None = None
        val: ast.AST | None = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name, val = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, val = node.target.id, node.value
        if (
            name in target_names
            and isinstance(val, ast.Constant)
            and isinstance(val.value, str)
            and val.end_lineno is not None
            and val.end_col_offset is not None
        ):
            start_pos = _ast_char_offset(lines, val.lineno, val.col_offset)
            end_pos = _ast_char_offset(lines, val.end_lineno, val.end_col_offset)
            hits.append((name, val.value, start_pos, end_pos))
    return hits


def _replace_trackb_constant(module_source: str, sym_name: str, cpp_source: str) -> str:
    """Rewrite one top-level Track-B C++ string constant in *module_source*."""
    orig_tree = ast.parse(module_source)
    lines = module_source.splitlines(keepends=True)
    hits = _find_top_level_str_assign_span(orig_tree, lines, {sym_name})
    if not hits:
        raise ValueError(f"Track-B symbol {sym_name!r} assignment not found in module_source.")
    _, _, start_pos, end_pos = hits[0]
    patched = (
        module_source[:start_pos]
        + _format_raw_cpp_literal(cpp_source)
        + module_source[end_pos:]
    )
    ast.parse(patched)
    return patched


def _apply_scoped_trackb_patch_to_module(
    module_source: str,
    regime_spec: RegimeTargetSpec,
    patch_text: str,
    *,
    active_kernel: str | None = None,
) -> str:
    extracted = _extract_trackb_cpp_source(module_source, regime_spec)
    if extracted is None:
        raise ValueError(
            f"Original module missing Track-B C++ constant in {regime_spec.trackb_symbols}."
        )
    sym_name, orig_cpp = extracted
    resolved_kernel = active_kernel or select_active_trackb_kernel(
        orig_cpp,
        regime_tag=regime_spec.regime_tag,
        module_source=module_source,
    )
    updated_cpp, _applied = apply_scoped_trackb_patch(
        orig_cpp,
        patch_text,
        active_kernel=resolved_kernel,
    )
    return _replace_trackb_constant(module_source, sym_name, updated_cpp)


def _apply_spliced_trackb_units_to_module(
    module_source: str,
    regime_spec: RegimeTargetSpec,
    candidate_cpp: str,
    *,
    active_kernel: str | None = None,
) -> str:
    """Replace same-name bare C++ units of the active kernel without touching siblings."""
    extracted = _extract_trackb_cpp_source(module_source, regime_spec)
    if extracted is None:
        raise ValueError(
            f"Original module missing Track-B C++ constant in {regime_spec.trackb_symbols}."
        )
    sym_name, orig_cpp = extracted
    resolved_kernel = active_kernel or select_active_trackb_kernel(
        orig_cpp,
        regime_tag=regime_spec.regime_tag,
        module_source=module_source,
    )
    updated_cpp, _applied = splice_cpp_units_into_tu(
        orig_cpp,
        candidate_cpp,
        active_kernel=resolved_kernel,
    )
    return _replace_trackb_constant(module_source, sym_name, updated_cpp)


def _trackb_python_residual(patch_text: str) -> str:
    """Return the valid Python snippet outside SEARCH/REPLACE blocks, else empty string."""
    residual = _strip_trackb_patch_hunks(patch_text)
    if not residual:
        return ""
    try:
        ast.parse(residual)
    except SyntaxError:
        return ""
    return residual


def _compact_trackb_snippet_source(
    snippet_source: str,
    regime_spec: RegimeTargetSpec,
    *,
    incumbent_source: str = "",
    max_bytes: int = 14800,
    active_kernel: str | None = None,
) -> str:
    if regime_spec.runtime_backend not in ("ascendc_trackb", "hybrid_trackb"):
        return snippet_source
    if not regime_spec.trackb_symbols:
        return snippet_source
    try:
        tree = ast.parse(snippet_source)
    except SyntaxError:
        return snippet_source
    lines = snippet_source.splitlines(keepends=True)
    hits = _find_top_level_str_assign_span(tree, lines, set(regime_spec.trackb_symbols))
    replacements: list[tuple[int, int, str]] = []
    for sym_name, cpp_text, start_pos, end_pos in hits:
        if len(cpp_text.encode("utf-8")) <= max_bytes:
            continue
        view = build_trackb_context_view(
            cpp_text,
            regime_tag=regime_spec.regime_tag,
            module_source=incumbent_source or snippet_source,
            trackb_symbol=sym_name,
            max_bytes=max_bytes,
            active_kernel=active_kernel,
        )
        wrapped = view if view.startswith("\n") else f"\n{view}\n"
        replacements.append((start_pos, end_pos, _format_raw_cpp_literal(wrapped)))
    if not replacements:
        return snippet_source
    updated = snippet_source
    for start_pos, end_pos, literal in reversed(replacements):
        updated = updated[:start_pos] + literal + updated[end_pos:]
    return updated


def _is_real_tir_primfunc(obj: object) -> bool:
    if obj is None or isinstance(obj, (tuple, list, dict, str, bytes, int, float, bool)):
        return False
    try:
        import tilelang  # noqa: F401, PLC0415
        import tvm  # noqa: PLC0415

        tirx_mod = getattr(tvm, "tirx", None)
        tir_mod = getattr(tvm, "tir", None)
        prim_cls = getattr(tirx_mod, "PrimFunc", None) or getattr(tir_mod, "PrimFunc", None)
        return isinstance(prim_cls, type) and isinstance(obj, prim_cls)
    except ImportError:
        return False


def patch_regime_into_module(
    module_source: str,
    regime_or_func: RegimeTargetSpec | str,
    candidate_raw_source: str,
    *,
    operator_entrypoint: str = "",
    sibling_specs: Sequence[RegimeTargetSpec] = (),
    active_kernel: str | None = None,
) -> str:
    """Merge a scoped or Python regime slice into module_source without losing siblings.

    Track-B replies route by shape: SEARCH/REPLACE hunks patch the launched C++ kernel
    and only the helpers it reaches, bare C++ units replace same-name units in place,
    and a full tuple is only accepted from a parseable Python assignment that keeps
    every sibling kernel. Any parseable Python beside patch hunks is merged by symbol.
    """
    candidate_dsl = extract_python_dsl(candidate_raw_source)
    if isinstance(regime_or_func, str):
        return apply_ast_patch(module_source, regime_or_func, candidate_dsl)

    regime_spec = regime_or_func
    python_snippet = candidate_dsl
    if (
        regime_spec.runtime_backend in ("ascendc_trackb", "hybrid_trackb")
        and regime_spec.trackb_symbols
    ):
        _reject_trackb_folding_markers(candidate_dsl, regime_spec.regime_tag)
        if parse_scoped_search_replace_blocks(candidate_dsl):
            module_source = _apply_scoped_trackb_patch_to_module(
                module_source,
                regime_spec,
                candidate_dsl,
                active_kernel=active_kernel,
            )
            python_snippet = _trackb_python_residual(candidate_dsl)
            if not python_snippet:
                return module_source
        elif not _parses_as_python(candidate_dsl) and _looks_like_bare_trackb_cpp(
            candidate_dsl
        ):
            return _apply_spliced_trackb_units_to_module(
                module_source,
                regime_spec,
                candidate_dsl,
                active_kernel=active_kernel,
            )
    orig_tree = ast.parse(module_source)
    if not regime_spec.editable_symbols:
        orig_node = _find_top_level_func(orig_tree, regime_spec.builder_candidates)
        if orig_node is None:
            raise ValueError(
                f"Original module missing builder in {regime_spec.builder_candidates}."
            )
        return apply_ast_patch(module_source, orig_node.name, python_snippet)

    orig_symbols = _collect_top_level_symbols(orig_tree)
    orig_by_name = {sym.name: sym for sym in orig_symbols}
    editable_set = set(regime_spec.editable_symbols)
    prefix = regime_spec.allowed_helper_prefix

    required_symbols = [sym for sym in regime_spec.editable_symbols if sym in orig_by_name]
    if not required_symbols:
        orig_node = _find_top_level_func(orig_tree, regime_spec.builder_candidates)
        if orig_node is None:
            raise ValueError(
                f"Original module missing regime symbols {regime_spec.editable_symbols}."
            )
        return apply_ast_patch(module_source, orig_node.name, python_snippet)

    cand_tree = ast.parse(python_snippet)
    cand_symbols: list[_TopLevelSymbol] = []
    cand_ast_by_name: dict[str, ast.stmt] = {}
    seen_cand: set[str] = set()
    forbidden = _other_regime_forbidden_symbols(
        regime_spec,
        operator_entrypoint=operator_entrypoint,
        sibling_specs=sibling_specs,
    )
    for sym_name in orig_by_name:
        if sym_name not in editable_set and not (prefix and sym_name.startswith(prefix)):
            forbidden.add(sym_name)

    for node in cand_tree.body:
        start, end = _stmt_line_span(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name, kind = node.name, "func"
        elif isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise ValueError("Candidate assignment must target a single variable name.")
            name, kind = node.targets[0].id, "assign"
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, kind = node.target.id, "assign"
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        else:
            raise ValueError(
                f"Candidate snippet contains unsupported top-level {type(node).__name__}."
            )

        if name in seen_cand:
            raise ValueError(f"Candidate snippet defines duplicate symbol {name!r}.")
        seen_cand.add(name)

        if name in forbidden:
            raise ValueError(f"Candidate cannot define or modify protected symbol {name!r}.")
        if name not in editable_set:
            if not prefix or not name.startswith(prefix):
                raise ValueError(
                    f"Unauthorized top-level symbol {name!r}; expected {editable_set} "
                    f"or prefix {prefix!r}."
                )
        cand_symbols.append(_TopLevelSymbol(name, kind, start, end))
        cand_ast_by_name[name] = node

    if not cand_symbols:
        raise ValueError(
            "Candidate snippet must define at least one authorized regime symbol."
        )

    cand_lines = python_snippet.splitlines(keepends=True)
    cand_slices = {
        sym.name: "".join(cand_lines[sym.start - 1 : sym.end]).rstrip() + "\n"
        for sym in cand_symbols
    }

    orig_regime_items = [
        sym
        for sym in orig_symbols
        if sym.name in editable_set or (prefix and sym.name.startswith(prefix))
    ]
    merged_ast_by_name = _collect_top_level_ast_nodes(orig_tree)
    merged_ast_by_name.update(cand_ast_by_name)
    parent_helper_names = {
        sym.name
        for sym in orig_regime_items
        if sym.name not in editable_set and prefix and sym.name.startswith(prefix)
    }
    all_helper_names = set(parent_helper_names) | {
        sym.name for sym in cand_symbols if sym.name not in editable_set
    }
    seed_roots = [
        name
        for name in merged_ast_by_name
        if name not in parent_helper_names or name in cand_slices
    ]
    live_helpers = _collect_referenced_helper_names(
        merged_ast_by_name,
        helper_names=all_helper_names,
        seed_roots=seed_roots,
    )

    new_helpers = [
        sym
        for sym in cand_symbols
        if sym.name not in editable_set and sym.name not in orig_by_name
    ]
    insert_line_idx = orig_regime_items[0].start - 1

    orig_lines = module_source.splitlines(keepends=True)
    for item in reversed(orig_regime_items):
        if item.name in cand_slices:
            orig_lines[item.start - 1 : item.end] = [cand_slices[item.name]]
        elif item.name in editable_set or item.name in live_helpers:
            continue
        else:
            del_end = item.end
            while del_end < len(orig_lines) and orig_lines[del_end].strip() == "":
                del_end += 1
            del orig_lines[item.start - 1 : del_end]

    if new_helpers:
        helper_block = "".join(cand_slices[h.name] + "\n" for h in new_helpers)
        orig_lines[insert_line_idx:insert_line_idx] = [helper_block]

    patched = "".join(orig_lines)
    ast.parse(patched)
    if set(cand_slices) & set(regime_spec.trackb_symbols):
        orig_trackb = _extract_trackb_cpp_source(module_source, regime_spec)
        new_trackb = _extract_trackb_cpp_source(patched, regime_spec)
        if orig_trackb is None or new_trackb is None:
            raise ValueError(
                f"Track-B symbols {sorted(set(cand_slices) & set(regime_spec.trackb_symbols))} "
                "must be assigned a raw C++ string literal."
            )
        _assert_trackb_replacement_scoped(
            orig_trackb[1],
            new_trackb[1],
            regime_spec,
            module_source=module_source,
            active_kernel=active_kernel,
        )
    return patched


def _surrogate_rank_key(res: CandidateSurrogateResult) -> tuple[float, float, float]:
    return (res.surrogate_score, res.min_case_speedup, res.avg_speedup)


def _format_anchor_telemetry(
    result: CandidateSurrogateResult,
    regime_tag: str,
) -> str:
    rows = []
    for anchor in result.anchors:
        rows.append(
            {
                "case_num": anchor.case_num,
                "regime_tag": anchor.regime_tag,
                "target_regime": anchor.regime_tag == regime_tag,
                "accuracy_passed": anchor.accuracy_passed,
                "cand_us": round(anchor.cand_us, 4),
                "ref_perf_us": round(anchor.ref_perf_us, 4),
                "t_hw_us": round(anchor.t_hw_us, 4),
                "speedup": round(anchor.speedup, 4),
                "roofline_eta": round(anchor.roofline_eta, 4),
                "headroom": round(max(0.0, 1.0 - anchor.roofline_eta), 4),
            }
        )
    return json.dumps(
        {
            "operator": result.operator_name,
            "avg_speedup": round(result.avg_speedup, 4),
            "min_case_speedup": round(result.min_case_speedup, 4),
            "avg_roofline_eta": round(result.avg_roofline_eta, 4),
            "surrogate_score": round(result.surrogate_score, 4),
            "anchors": rows,
        },
        indent=2,
    )


class PerfTargetCampaignRunner:
    """Host-CPU multi-round optimization controller driving L0->L1->L2->L3->L4 pyramid."""

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
        checkpoint_path: str | Path | None = None,
        campaign_kb_path: str | Path | None = None,
        source_root: str | Path | None = None,
        env_file: str | Path | None = None,
        cannbench_repo: str | Path | None = None,
        authored_hir_path: str | Path | None = None,
        scheduled_hir_path: str | Path | None = None,
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
        retry_device_id = config.retry_device_id.strip()
        self._retry_device_id = (
            _canonical_device_id(retry_device_id, field="retry_device_id")
            if retry_device_id
            else ""
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
        self._cannbench_repo = Path(cannbench_repo).resolve() if cannbench_repo is not None else None
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
        known_regime_tags = {spec.regime_tag for spec in self._regime_specs}
        for forced_tag in config.round_regime_tags:
            if forced_tag not in known_regime_tags:
                raise ValueError(
                    f"round regime tag {forced_tag!r} is not configured; "
                    f"available regimes: {sorted(known_regime_tags)}"
                )
        self._backend = llm_backend
        self._tracker = tracker or BayesianRuleTracker.from_seed_yaml()
        base_ew = beam.exploit_width if beam is not None else 2
        base_sw = beam.stepping_stone_width if beam is not None else 2
        self._beam_widths = (base_ew, base_sw)
        self._beams: dict[str, TwoLaneBeamSearch] = {}
        for idx, rspec in enumerate(self._regime_specs):
            if idx == 0 and beam is not None:
                self._beams[rspec.regime_tag] = beam
            else:
                self._beams[rspec.regime_tag] = TwoLaneBeamSearch(
                    exploit_width=base_ew,
                    stepping_stone_width=base_sw,
                )
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
                lambda op_name, cand_path, anc_seq: collect_bottleneck_msprof_via_agent_gates(
                    op_name,
                    cand_path,
                    anc_seq,
                    source_root=self._source_root,
                    env_file=self._env_file,
                    device_id=self._config.device_id,
                )
            )
        else:
            self._msprof_collector = None
        self._msprof_cache: dict[tuple[str, str], dict[str, float]] = {}
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
        self._cost_model = HybridAscendCostModel(
            self._target,
            calibration_path=self._cost_calibration_path,
        )
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
        op_slug = self._config.operator_name.replace("/", "_")
        self._run_id = f"{op_slug}_{int(time.time())}_{os.getpid()}"
        self._case_dims_cache: dict[int, dict[str, object]] = {}
        if self._experience_lake_dir is not None and self._experience_lake_dir.exists():
            try:
                self._cost_model.replay_experience_lake(self._experience_lake_dir)
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug("ExperienceLake startup replay skipped: %s", exc)
        self._freedom_masks: dict[str, ConstraintRelaxationMask] = {}
        self._live_regime_plans: dict[str, RegimePlanResult] = {}
        self._clock = clock
        if sil_advisory is not None:
            self._sil_advisory = sil_advisory
        else:
            self._sil_advisory = None
            try:
                ops_root = self._resolve_cannbench_tasks_root()
                extractor = (
                    OpAdvisoryExtractor(ops_root)
                    if ops_root is not None
                    else OpAdvisoryExtractor()
                )
                short_op = self._config.operator_name.rsplit("/", 1)[-1]
                self._sil_advisory = extractor.extract(short_op)
            except (FileNotFoundError, OSError, KeyError):
                self._sil_advisory = None
        if isinstance(topology_axes, TopologyAxes):
            self._topology_axes: TopologyAxes | None = topology_axes
        elif isinstance(topology_axes, Mapping):
            self._topology_axes = TopologyAxes.from_dict(topology_axes)
        else:
            self._topology_axes = None

        if authored_hir_path is not None:
            self._authored_hir_path: Path | None = Path(authored_hir_path).resolve()
        else:
            sibling_model = self._target_module_path.parent / "model.py"
            self._authored_hir_path = sibling_model.resolve() if sibling_model.is_file() else None

        if scheduled_hir_path is not None:
            self._scheduled_hir_path: Path | None = Path(scheduled_hir_path).resolve()
        else:
            sibling_sched = self._target_module_path.parent / "scheduled.py"
            self._scheduled_hir_path = sibling_sched.resolve() if sibling_sched.is_file() else None

        self._regime_blueprints: dict[str, HirBlueprintPacket] = (
            self._validated_regime_blueprints(regime_blueprints)
        )
        self._l1_summary_cache: dict[tuple[str, str], BishengResourceSummary | None] = {}
        self._active_kernel_cache: dict[tuple[str, str], str | None] = {}
        self._active_incumbent_source: str = ""
        self._pending_promotion: PendingPromotion | None = None
        self._active_promotion: PendingPromotion | None = None
        self._pending_l4: PendingL4 | None = None
        if (
            self._flywheel is not None
            and self._flywheel._probe_runner is None
            and l2_batch_evaluator is None
            and (
                self._env_file is not None
                or bool(os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip())
            )
        ):
            self._flywheel._probe_runner = lambda spec: run_micro_probe_via_agent_gates(
                spec,
                source_root=self._source_root,
                env_file=self._env_file,
                device_id=self._config.device_id,
                min_speedup=self._flywheel._min_probe_speedup,
            )

    @property
    def _beam(self) -> TwoLaneBeamSearch:
        if self._active_regime_tag and self._active_regime_tag in self._beams:
            return self._beams[self._active_regime_tag]
        if self._beams:
            return next(iter(self._beams.values()))
        return self._fallback_beam

    def beam_for_regime(self, regime_tag: str) -> TwoLaneBeamSearch:
        """Return the isolated TwoLaneBeamSearch instance for regime_tag."""
        if regime_tag not in self._beams:
            ew, sw = self._beam_widths
            self._beams[regime_tag] = TwoLaneBeamSearch(
                exploit_width=ew,
                stepping_stone_width=sw,
            )
        return self._beams[regime_tag]

    @staticmethod
    def _source_sha256(source: str) -> str:
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    def _write_source_and_verify(
        self,
        path: Path,
        source: str,
        *,
        expected_sha256: str | None = None,
    ) -> str:
        """Atomically write source and prove the restored bytes have the expected identity."""
        expected = expected_sha256 or self._source_sha256(source)
        _atomic_write_text(path, source)
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            return f"restored module {path} cannot be read back: {exc}"
        if actual != expected:
            return (
                f"restored module {path} digest {actual} does not match expected {expected}"
            )
        return ""

    def _ensure_candidate_bytes(
        self,
        *,
        path: Path,
        source: str,
        expected_sha256: str,
    ) -> str:
        """Leave the exact submitted candidate at its path, even after a crash.

        An active worker may still be testing this path; restoring the incumbent
        here would swap the code under it. If the bytes drifted, put the original
        candidate back and prove the digest before any resume.
        """
        expected = expected_sha256 or self._source_sha256(source)
        try:
            current = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            current = ""
        if current == expected:
            return ""
        error = self._write_source_and_verify(path, source, expected_sha256=expected)
        if error:
            return error
        return ""

    def _persist_runtime_state(
        self,
        key: str,
        value: Mapping[str, object] | None,
    ) -> None:
        """Upsert a recoverable execution record before a long V2 wait starts.

        The campaign checkpoint also carries round state; this read-modify-write
        adds only ``key`` so a process killed during submit/wait still leaves the
        exact execution and candidate identity on disk.
        """
        if self._checkpoint_path is None:
            return
        payload: dict[str, object] = {}
        if self._checkpoint_path.is_file():
            try:
                loaded = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
                if isinstance(loaded, Mapping):
                    payload = dict(loaded)
            except (OSError, ValueError):
                payload = {}
        payload[key] = dict(value) if value is not None else None
        _atomic_write_text(
            self._checkpoint_path,
            json.dumps(payload, indent=2) + "\n",
        )

    def _round_candidate_artifact_dir(self) -> Path | None:
        """Return the checkpoint-adjacent candidates directory for audit artifacts."""
        if self._checkpoint_path is None:
            return None
        return self._checkpoint_path.parent / "candidates"

    def _persist_round_candidate_artifact(
        self,
        check: L0L1CandidateCheck,
        *,
        round_idx: int,
    ) -> None:
        """Write generated source and untruncated L0/L1 diagnostics beside the checkpoint."""
        artifact_dir = self._round_candidate_artifact_dir()
        if artifact_dir is None:
            return
        stem = f"r{round_idx}_{check.candidate_id.removeprefix(f'r{round_idx}_')}"
        _atomic_write_text(
            artifact_dir / f"{stem}.py",
            check.patched_module_source or check.snippet_source,
        )
        diag_lines = [
            f"candidate_id: {check.candidate_id}",
            f"round: {round_idx}",
            f"passed: {check.passed}",
            f"failed_tier: {check.failed_tier or ''}",
            f"repaired: {check.repaired}",
            f"error_message: {check.error_message}",
        ]
        if check.stage_a is not None:
            diag_lines.extend(
                [
                    f"stage_a_passed: {check.stage_a.passed}",
                    f"stage_a_backend_kind: {check.stage_a.backend_kind}",
                    f"stage_a_error: {check.stage_a.error_message}",
                ]
            )
        if check.stage_b is not None:
            diag_lines.extend(
                [
                    f"stage_b_passed: {check.stage_b.passed}",
                    f"stage_b_error: {check.stage_b.error_message}",
                    f"stage_b_spill_bytes: {check.stage_b.spill_bytes}",
                    f"stage_b_vloopv2_hit: {check.stage_b.vloopv2_hit}",
                ]
            )
            if check.stage_b.full_diagnostic:
                diag_lines.extend(
                    [
                        "stage_b_full_diagnostic:",
                        check.stage_b.full_diagnostic,
                    ]
                )
        _atomic_write_text(
            artifact_dir / f"{stem}.diag.txt",
            "\n".join(diag_lines) + "\n",
        )

    @staticmethod
    def _invoke_with_optional_kwargs(
        runner: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> object:
        """Pass only the optional keywords an injected runner actually accepts."""
        try:
            signature = inspect.signature(runner)
        except (TypeError, ValueError):
            return runner(*args)
        accepts_any = any(
            param.kind is inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
        accepted = {
            name: value
            for name, value in kwargs.items()
            if accepts_any or name in signature.parameters
        }
        return runner(*args, **accepted)

    def _derive_topology_axes(
        self,
        ledger: ObjectivePhysicalLedger | None,
        *,
        features: object | None = None,
        route: object | None = None,
    ) -> TopologyAxes:
        if self._topology_axes is not None:
            return self._topology_axes
        if self._sil_advisory is not None:
            return resolve_canonical_topology_axes(self._sil_advisory, features=features)
        if route is not None:
            r_axes = (
                route.get("topology_axes")
                if isinstance(route, Mapping)
                else getattr(route, "topology_axes", None)
            )
            if isinstance(r_axes, TopologyAxes):
                return r_axes
            if isinstance(r_axes, Mapping) and all(
                k in r_axes for k in ("compute_engine", "memory_ring", "iteration_regularity")
            ):
                return TopologyAxes.from_dict(r_axes)
        uses_cube = bool(
            ledger is not None
            and (
                ledger.pipe_cycles.get("AIC.PIPE_M", 0) > 0
                or ledger.pipe_cycles.get("AIC.CUBE", 0) > 0
            )
        )
        return resolve_canonical_topology_axes(
            {"op_name": self._config.operator_name, "uses_cube": uses_cube},
            features=features,
        )

    def _collect_msprof_for_regimes(
        self,
        source: str,
        result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec],
        target_regimes: Sequence[RegimeTargetSpec],
    ) -> None:
        if self._msprof_collector is None:
            return
        sha = self._source_sha256(source)
        by_case_and_tag = {(int(a.case_num), a.regime_tag): a for a in active_anchors}
        by_case = {int(a.case_num): a for a in active_anchors}
        anchors_to_collect: list[AnchorCaseSpec] = []
        for rspec in target_regimes:
            if (sha, rspec.regime_tag) in self._msprof_cache:
                continue
            bn = bottleneck_anchor_for_regime(result, rspec.regime_tag)
            if bn is None:
                continue
            spec = by_case_and_tag.get((int(bn.case_num), rspec.regime_tag)) or by_case.get(
                int(bn.case_num)
            )
            if spec is None:
                continue
            if spec.regime_tag != rspec.regime_tag:
                spec = AnchorCaseSpec(
                    spec.task,
                    spec.case_num,
                    rspec.regime_tag,
                    spec.ref_perf_us,
                    spec.t_hw_us,
                )
            anchors_to_collect.append(spec)
        if not anchors_to_collect:
            return

        try:
            with tempfile.TemporaryDirectory(prefix="tf_msprof_cand_") as tmp_dir:
                cand_path = Path(tmp_dir) / "candidate.py"
                cand_path.write_text(source, encoding="utf-8")
                collected = self._msprof_collector(
                    self._config.operator_name,
                    str(cand_path),
                    anchors_to_collect,
                )
        except (OSError, subprocess.SubprocessError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            logger.warning("On-demand msprof collection failed: %s", exc)
            return

        if isinstance(collected, Mapping):
            for tag, raw_metrics in collected.items():
                if not isinstance(raw_metrics, Mapping):
                    continue
                clean = {
                    str(k): float(v)
                    for k, v in raw_metrics.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
                if clean:
                    self._msprof_cache[(sha, str(tag))] = clean

    def _resolve_cannbench_tasks_root(self) -> Path | None:
        """Resolve the cann-bench tasks root, falling back to the in-repo mirror."""
        try:
            repo = resolve_cannbench_repo(
                cannbench_repo=self._cannbench_repo,
                env_file=self._env_file,
            )
        except FileNotFoundError:
            repo = None
        if repo is not None:
            return resolve_cannbench_tasks_root(repo)
        in_repo = self._source_root / "tests" / "ascend" / "benchmarks"
        if _has_flat_operator_layout(in_repo):
            return in_repo
        return None

    def _validated_regime_blueprints(
        self,
        provided: Mapping[str, HirBlueprintPacket] | None,
    ) -> dict[str, HirBlueprintPacket]:
        """Validate caller-supplied per-regime blueprints or compile them in-process."""
        if provided is None:
            return self._compile_regime_blueprints()
        missing = sorted(
            spec.regime_tag for spec in self._regime_specs if spec.regime_tag not in provided
        )
        if missing:
            raise ValueError(
                f"regime_blueprints missing regime tags {missing}; "
                f"got {sorted(provided)}"
            )
        return dict(provided)

    def _resolve_operator_spec(self) -> CannbenchOpSpec | None:
        """Resolve the official cann-bench task spec for the campaign operator."""
        tasks_root = self._resolve_cannbench_tasks_root()
        if tasks_root is None:
            return None
        registry = CannbenchOpsRegistry(tasks_root=tasks_root, source_root=self._source_root)
        if not registry.has_operator(self._config.operator_name):
            return None
        return registry.resolve(self._config.operator_name)

    def _synthesize_universal_regime_blueprints(self) -> dict[str, HirBlueprintPacket]:
        """Synthesize per-regime blueprints from the PyTorch golden when model.py is absent."""
        spec = self._resolve_operator_spec()
        if spec is None:
            return {}
        if not spec.golden_path.is_file():
            raise FileNotFoundError(
                f"official PyTorch golden reference missing for {spec.op_id} at {spec.golden_path}"
            )
        try:
            golden_source = spec.golden_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(
                f"cannot read PyTorch golden reference for {spec.op_id} at {spec.golden_path}: {exc}"
            ) from exc
        analyzer = PyTorchSemanticAnalyzer(self._target)
        try:
            semantics = analyzer.analyze_source(
                golden_source,
                entry_name=spec.entrypoint,
                proto=spec.proto,
            )
        except (SyntaxError, ValueError) as exc:
            raise RuntimeError(
                f"cannot analyze PyTorch golden semantics for {spec.op_id}: {exc}"
            ) from exc
        packets: dict[str, HirBlueprintPacket] = {}
        for regime_spec in self._regime_specs:
            try:
                packets[regime_spec.regime_tag] = analyzer.build_universal_blueprint(
                    semantics,
                    regime_spec.representative_hir_dims,
                    representative_case=(
                        RepresentativeCase.from_case_entry(
                            regime_spec.representative_case_entry
                        )
                        if regime_spec.representative_case_entry
                        else None
                    ),
                )
            except (FileNotFoundError, OSError, SyntaxError, ValueError, KeyError) as exc:
                raise RuntimeError(
                    f"failed to synthesize universal HIR blueprint for {spec.op_id} regime "
                    f"{regime_spec.regime_tag}: {type(exc).__name__}: {exc}"
                ) from exc
        assert set(packets) == {regime.regime_tag for regime in self._regime_specs}
        return packets

    def _compile_regime_blueprints(self) -> dict[str, HirBlueprintPacket]:
        """Compile non-empty per-regime Layer-3 blueprints, preferring authored model.py."""
        authored_hir_source: str | None = None
        if self._authored_hir_path is not None and self._authored_hir_path.is_file():
            try:
                authored_hir_source = self._authored_hir_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise RuntimeError(
                    f"cannot read authored HIR source at {self._authored_hir_path}: {exc}"
                ) from exc
        scheduled_hir_source: str | None = None
        if self._scheduled_hir_path is not None and self._scheduled_hir_path.is_file():
            try:
                scheduled_hir_source = self._scheduled_hir_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise RuntimeError(
                    f"cannot read scheduled HIR source at {self._scheduled_hir_path}: {exc}"
                ) from exc

        if authored_hir_source is None or not authored_hir_source.strip():
            return self._synthesize_universal_regime_blueprints()

        packets: dict[str, HirBlueprintPacket] = {}
        for regime_spec in self._regime_specs:
            try:
                packets[regime_spec.regime_tag] = compile_hir_blueprint(
                    authored_hir_source,
                    scheduled_hir_source=scheduled_hir_source,
                    anchor_dims=regime_spec.representative_hir_dims,
                    target=self._target,
                )
            except (FileNotFoundError, OSError, SyntaxError, ValueError, KeyError) as exc:
                raise RuntimeError(
                    f"failed to compile authored HIR blueprint for regime "
                    f"{regime_spec.regime_tag}: {type(exc).__name__}: {exc}"
                ) from exc
        assert set(packets) == {regime.regime_tag for regime in self._regime_specs}
        return packets

    def _get_backend(self) -> KernelLLMBackend:
        if self._backend is None:
            self._backend = create_llm_backend_from_env()
        return self._backend

    @staticmethod
    def _function_ast_dump(module_source: str, func_name: str) -> str:
        tree = ast.parse(module_source)
        node = _find_top_level_func(tree, (func_name,))
        return ast.dump(node) if node is not None else ""

    def _resolve_combine_gate_spec(
        self,
        module_source: str,
        regime_spec: RegimeTargetSpec,
        resolved: ResolvedRegimeConfig,
    ) -> tuple[str, dict[str, object]] | None:
        combine_fn = next(
            (
                sym
                for sym in regime_spec.editable_symbols
                if sym.startswith("_build_") and "combine" in sym
            ),
            "",
        )
        if not combine_fn:
            return None
        cand_dump = self._function_ast_dump(module_source, combine_fn)
        if not cand_dump:
            return None
        splits = int(resolved.builder_kwargs.get("num_kv_splits", 1) or 1)
        modified = False
        baseline_src = self._active_incumbent_source
        if not baseline_src and self._target_module_path.is_file():
            try:
                baseline_src = self._target_module_path.read_text(encoding="utf-8")
            except OSError:
                baseline_src = ""
        if not baseline_src:
            modified = True
        else:
            base_dump = self._function_ast_dump(baseline_src, combine_fn)
            modified = cand_dump != base_dump
        if splits <= 1 and not modified:
            return None
        combine_splits = splits if splits > 1 else 4
        merged_kwargs: dict[str, object] = {
            **regime_spec.sample_builder_kwargs,
            **resolved.builder_kwargs,
            "num_kv_splits": combine_splits,
            **resolved.combine_kwargs,
        }
        mod_tree = ast.parse(module_source)
        fn_node = _find_top_level_func(mod_tree, (combine_fn,))
        if fn_node is not None and fn_node.args.kwarg is None:
            accepted = {
                *(a.arg for a in fn_node.args.posonlyargs),
                *(a.arg for a in fn_node.args.args),
                *(a.arg for a in fn_node.args.kwonlyargs),
            }
            combine_kwargs = {k: v for k, v in merged_kwargs.items() if k in accepted}
        else:
            combine_kwargs = merged_kwargs
        return combine_fn, combine_kwargs

    def _instantiate_trackb_tu(
        self,
        module_source: str,
        cpp_source: str,
        regime_spec: RegimeTargetSpec,
    ) -> str:
        """Instantiate Track-B C++20 TU via module _render_trackb_tu if present."""
        return _render_trackb_tu_from_module(
            module_source,
            cpp_source,
            regime_spec,
            require_renderer=False,
        )

    def _resolve_active_trackb_kernel(
        self,
        module_source: str,
        regime_spec: RegimeTargetSpec,
    ) -> str | None:
        """Resolve the launched Track-B kernel from instantiation truth, else the raw TU.

        A pass-through renderer leaves the launcher inside Python, so the raw
        translation unit is scanned for the kernel the module starts instead of
        demanding a C++ launcher stub it never had.
        """
        if regime_spec.runtime_backend not in ("ascendc_trackb", "hybrid_trackb"):
            return None
        extracted = _extract_trackb_cpp_source(module_source, regime_spec)
        if extracted is None:
            return None
        cache_key = (self._source_sha256(module_source), regime_spec.regime_tag)
        if cache_key not in self._active_kernel_cache:
            rendered = self._instantiate_trackb_tu(module_source, extracted[1], regime_spec)
            if rendered == extracted[1]:
                fallback = select_active_trackb_kernel(
                    extracted[1],
                    regime_tag=regime_spec.regime_tag,
                    module_source=module_source,
                )
                if fallback is None:
                    raise ValueError(
                        f"Track-B source for regime {regime_spec.regime_tag!r} "
                        "contains no __global__ kernel"
                    )
                self._active_kernel_cache[cache_key] = fallback
            else:
                self._active_kernel_cache[cache_key] = _active_trackb_kernel(
                    rendered,
                    extracted[1],
                    regime_tag=regime_spec.regime_tag,
                )
        return self._active_kernel_cache[cache_key]

    def _resolve_trackb_freedom_mask(
        self,
        regime_spec: RegimeTargetSpec,
    ) -> ConstraintRelaxationMask:
        """Return active 7-facet mask for *regime_spec*, seeding Track-B baseline facets once."""
        existing = self._freedom_masks.get(regime_spec.regime_tag)
        if existing is not None:
            return existing
        baseline_src = self._active_incumbent_source
        if not baseline_src and self._target_module_path.is_file():
            baseline_src = self._target_module_path.read_text(encoding="utf-8")
        if baseline_src and regime_spec.runtime_backend in ("ascendc_trackb", "hybrid_trackb"):
            base_tb = _extract_trackb_cpp_source(baseline_src, regime_spec)
            if base_tb is not None:
                base_tu = self._instantiate_trackb_tu(baseline_src, base_tb[1], regime_spec)
                rep = verify_source_against_facets(base_tu, ConstraintRelaxationMask.bottom())
                fixes = rep.suggested_fixes
                if not rep.passed and fixes:
                    return ConstraintRelaxationMask(
                        alg_compute_and_monoid=not bool(fixes.get("unlock_alg_compute_and_monoid")),
                        tile_and_stage_knobs=True,
                        exec_mode_and_spr=bool(fixes.get("unlock_exec_mode_and_spr", False)),
                        sram_layout_and_aliasing=bool(
                            fixes.get("unlock_sram_layout_and_aliasing", False)
                        ),
                        dma_and_cache_hierarchy=bool(
                            fixes.get("unlock_dma_and_cache_hierarchy", False)
                        ),
                        pipeline_and_sync_graph=bool(
                            fixes.get("unlock_pipeline_and_sync_graph", False)
                        ),
                        grid_and_workspace_topology=bool(
                            fixes.get("unlock_grid_and_workspace_topology", False)
                        ),
                    )
        return ConstraintRelaxationMask.bottom()

    def _compile_trackb_stage_b(self, cpp_source: str) -> StageBResult:
        cann_root = _resolve_cann_root_from_env_file(self._env_file)
        bisheng_bin: str | None = None
        flags = list(PURE_SIMDVF_UBUF_FLAGS)
        if cann_root is not None:
            cand_bin = cann_root / "bin" / "bisheng"
            if cand_bin.is_file():
                bisheng_bin = str(cand_bin)
            if (cann_root / "include").is_dir():
                flags.append(f"-I{cann_root / 'include'}")
        return run_stage_b_ascendc_compile(
            cpp_source,
            compile_flags=flags,
            bisheng_bin=bisheng_bin,
            cann_root=cann_root,
        )

    def _default_l0_checker(
        self,
        module_source: str,
        regime_spec: RegimeTargetSpec,
    ) -> StageAResult:
        tb_extracted = (
            _extract_trackb_cpp_source(module_source, regime_spec)
            if regime_spec.runtime_backend in ("ascendc_trackb", "hybrid_trackb")
            else None
        )
        if regime_spec.runtime_backend == "ascendc_trackb" and tb_extracted is not None:
            active_mask = self._resolve_trackb_freedom_mask(regime_spec)
            instantiated_tu = self._instantiate_trackb_tu(
                module_source, tb_extracted[1], regime_spec
            )
            return evaluate_ascendc_candidate(
                instantiated_tu,
                freedom_mask=active_mask,
                target=self._target,
                enable_track_router=self._enable_track_router,
                case_ratios=regime_spec.representative_case_ratios,
            )

        anchor_dims = (
            regime_spec.representative_anchor_dims or regime_spec.representative_hir_dims
        )
        try:
            resolved = resolve_regime_config(module_source, regime_spec, dims=anchor_dims)
        except (
            SyntaxError,
            AttributeError,
            ImportError,
            KeyError,
            NameError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            return StageAResult(
                passed=False,
                prim_func=None,
                ledger=None,
                error_message=f"stage_a_config_resolve_error: {type(exc).__name__}: {exc}",
            )
        var_bindings = build_anchor_var_bindings(
            anchor_dims,
            kwargs=resolved.builder_kwargs,
        )
        primary_res = run_stage_a_static(
            module_source,
            factory_name=resolved.builder_name,
            factory_kwargs=resolved.builder_kwargs,
            pass_configs=resolved.pass_configs or None,
            compile_flags=resolved.compile_flags or None,
            target=self._target,
            var_bindings=var_bindings or None,
            enable_track_router=self._enable_track_router,
            case_ratios=regime_spec.representative_case_ratios,
        )
        if not primary_res.passed:
            return primary_res
        combine_spec = self._resolve_combine_gate_spec(module_source, regime_spec, resolved)
        if combine_spec is not None:
            combine_fn, combine_kwargs = combine_spec
            combine_res = run_stage_a_static(
                module_source,
                factory_name=combine_fn,
                factory_kwargs=combine_kwargs,
                pass_configs=resolved.pass_configs or None,
                compile_flags=resolved.compile_flags or None,
                target=self._target,
                var_bindings=var_bindings or None,
                enable_track_router=self._enable_track_router,
                case_ratios=regime_spec.representative_case_ratios,
            )
            if not combine_res.passed:
                return StageAResult(
                    passed=False,
                    prim_func=primary_res.prim_func,
                    ledger=primary_res.ledger,
                    features=primary_res.features,
                    route=primary_res.route,
                    error_message=f"combine_kernel({combine_fn}): {combine_res.error_message}",
                    elapsed_ms=primary_res.elapsed_ms + combine_res.elapsed_ms,
                )
        if regime_spec.runtime_backend == "hybrid_trackb" and tb_extracted is not None:
            tb_sym, tb_cpp = tb_extracted
            active_mask = self._resolve_trackb_freedom_mask(regime_spec)
            instantiated_tu = self._instantiate_trackb_tu(module_source, tb_cpp, regime_spec)
            tb_res = evaluate_ascendc_candidate(
                instantiated_tu,
                freedom_mask=active_mask,
                target=self._target,
                enable_track_router=self._enable_track_router,
                case_ratios=regime_spec.representative_case_ratios,
            )
            if not tb_res.passed:
                return replace(
                    tb_res,
                    error_message=f"trackb_ascendc({tb_sym}): {tb_res.error_message}",
                    elapsed_ms=primary_res.elapsed_ms + tb_res.elapsed_ms,
                )
            return replace(
                tb_res,
                elapsed_ms=primary_res.elapsed_ms + tb_res.elapsed_ms,
            )
        return primary_res

    def _run_l0(self, module_source: str, regime_spec: RegimeTargetSpec) -> StageAResult:
        if self._l0_checker is not None:
            return self._l0_checker(module_source, regime_spec)
        return self._default_l0_checker(module_source, regime_spec)

    def _run_l1(self, module_source: str, regime_spec: RegimeTargetSpec) -> StageBResult:
        if self._l1_compiler is not None:
            return self._l1_compiler(module_source, regime_spec)
        tb_extracted = (
            _extract_trackb_cpp_source(module_source, regime_spec)
            if regime_spec.runtime_backend in ("ascendc_trackb", "hybrid_trackb")
            else None
        )
        if regime_spec.runtime_backend == "ascendc_trackb" and tb_extracted is not None:
            instantiated_tu = self._instantiate_trackb_tu(
                module_source, tb_extracted[1], regime_spec
            )
            return self._compile_trackb_stage_b(instantiated_tu)

        anchor_dims = (
            regime_spec.representative_anchor_dims or regime_spec.representative_hir_dims
        )
        try:
            resolved = resolve_regime_config(module_source, regime_spec, dims=anchor_dims)
        except (
            SyntaxError,
            AttributeError,
            ImportError,
            KeyError,
            NameError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            return StageBResult(
                passed=False,
                error_message=f"l1_config_resolve_error: {type(exc).__name__}: {exc}",
            )
        primary_b = run_l1_compile_check(
            module_source,
            builder_name=resolved.builder_name,
            builder_kwargs=resolved.builder_kwargs,
            config_selector_name=regime_spec.config_selector_fn or None,
            pass_configs=resolved.pass_configs or None,
            compile_flags=resolved.compile_flags or None,
            source_root=self._source_root,
            env_file=self._env_file,
        )
        if not primary_b.passed:
            return primary_b
        final_b = primary_b
        combine_elapsed_sec = 0.0
        combine_spec = self._resolve_combine_gate_spec(module_source, regime_spec, resolved)
        if combine_spec is not None:
            combine_fn, combine_kwargs = combine_spec
            combine_b = run_l1_compile_check(
                module_source,
                builder_name=combine_fn,
                builder_kwargs=combine_kwargs,
                config_selector_name=None,
                pass_configs=resolved.pass_configs or None,
                compile_flags=resolved.compile_flags or None,
                source_root=self._source_root,
                env_file=self._env_file,
            )
            if not combine_b.passed:
                return StageBResult(
                    passed=False,
                    error_message=f"combine_kernel({combine_fn}): {combine_b.error_message}",
                    full_diagnostic=combine_b.full_diagnostic,
                    elapsed_sec=primary_b.elapsed_sec + combine_b.elapsed_sec,
                )
            combine_elapsed_sec = combine_b.elapsed_sec
        if regime_spec.runtime_backend == "hybrid_trackb" and tb_extracted is not None:
            tb_sym, tb_cpp = tb_extracted
            instantiated_tu = self._instantiate_trackb_tu(module_source, tb_cpp, regime_spec)
            tb_b = self._compile_trackb_stage_b(instantiated_tu)
            total_elapsed_sec = final_b.elapsed_sec + combine_elapsed_sec + tb_b.elapsed_sec
            if not tb_b.passed:
                return replace(
                    tb_b,
                    error_message=f"trackb_ascendc({tb_sym}): {tb_b.error_message}",
                    elapsed_sec=total_elapsed_sec,
                )
            return replace(tb_b, elapsed_sec=total_elapsed_sec)
        return final_b

    def _l1_summary_for_first_round(
        self,
        module_source: str,
        regime_spec: RegimeTargetSpec,
    ) -> BishengResourceSummary | None:
        """Compile the parent once per source when no promoted stone carries L1 data."""
        cache_key = (self._source_sha256(module_source), regime_spec.regime_tag)
        if cache_key not in self._l1_summary_cache:
            result = self._run_l1(module_source, regime_spec)
            if not result.passed:
                logger.warning(
                    "First-round parent L1 for regime %s failed: %s",
                    regime_spec.regime_tag,
                    result.error_message,
                )
            self._l1_summary_cache[cache_key] = (
                result.compiler_summary if result.passed else None
            )
        return self._l1_summary_cache[cache_key]

    def _run_l2_batch(
        self,
        candidates: Sequence[tuple[str, Path]],
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> tuple[dict[str, CandidateSurrogateResult], dict[str, str]]:
        """Evaluate one batch and return results plus each measured candidate's V2 run id.

        Legacy injected evaluators return a plain mapping; their results keep working and
        carry no run ids rather than borrowing an id from a different execution.
        """
        if self._l2_batch_evaluator is not None:
            evaluated = self._l2_batch_evaluator(candidates, active_anchors)
            if isinstance(evaluated, L2BatchResults):
                return dict(evaluated), dict(evaluated.run_ids)
            return dict(evaluated), {}
        results, run_id, _receipt = run_surrogate_batch_via_agent_gates(
            self._config.operator_name,
            candidates,
            active_anchors,
            source_root=self._source_root,
            env_file=self._env_file,
            device_id=self._config.device_id,
            warmup=self._config.warmup,
            repeat=self._config.repeat,
            require_all_passed=False,
        )
        return results, {candidate_id: run_id for candidate_id in results}

    def _run_l3_contract(
        self,
        module_path: Path,
        *,
        device_id: str | None = None,
        on_execution_started: Callable[[str, str], None] | None = None,
    ) -> L3AcceptanceResult:
        requested_device = str(device_id or self._config.device_id)
        if self._l3_contract_runner is not None:
            return self._invoke_with_optional_kwargs(
                self._l3_contract_runner,
                module_path,
                device_id=requested_device,
                on_execution_started=on_execution_started,
            )  # type: ignore[return-value]
        return self._invoke_with_optional_kwargs(
            run_l3_full_contract_via_agent_gates,
            operator_name=self._config.operator_name,
            candidate_module_path=module_path,
            source_root=self._source_root,
            env_file=self._env_file,
            device_id=requested_device,
            on_execution_started=on_execution_started,
        )  # type: ignore[return-value]

    def _resume_l3_contract(
        self,
        module_path: Path,
        pending: PendingPromotion,
        *,
        on_execution_started: Callable[[str, str], None] | None = None,
    ) -> L3AcceptanceResult:
        """Continue the saved L3 execution instead of submitting a duplicate."""
        if self._l3_contract_runner is not None:
            return self._invoke_with_optional_kwargs(
                self._l3_contract_runner,
                module_path,
                resume_execution_id=pending.execution_id,
                resume_run_id=pending.run_id,
                device_id=pending.device_id or self._config.device_id,
                on_execution_started=on_execution_started,
            )  # type: ignore[return-value]
        return self._invoke_with_optional_kwargs(
            run_l3_full_contract_via_agent_gates,
            operator_name=self._config.operator_name,
            candidate_module_path=module_path,
            source_root=self._source_root,
            env_file=self._env_file,
            device_id=pending.device_id or self._config.device_id,
            resume_execution_id=pending.execution_id,
            resume_run_id=pending.run_id,
            on_execution_started=on_execution_started,
        )  # type: ignore[return-value]

    def _run_l4_cannbench(
        self,
        module_path: Path,
        *,
        resume_execution_id: str = "",
        resume_run_id: str = "",
        device_id: str | None = None,
        on_execution_started: Callable[[str, str], None] | None = None,
    ) -> L4CannbenchResult:
        requested_device = str(device_id or self._config.device_id)
        if self._l4_cannbench_runner is not None:
            return self._invoke_with_optional_kwargs(
                self._l4_cannbench_runner,
                module_path,
                resume_execution_id=resume_execution_id,
                resume_run_id=resume_run_id,
                device_id=requested_device,
                on_execution_started=on_execution_started,
            )  # type: ignore[return-value]
        return self._invoke_with_optional_kwargs(
            run_l4_official_cannbench_via_agent_gates,
            operator_name=self._config.operator_name,
            candidate_module_path=module_path,
            source_root=self._source_root,
            env_file=self._env_file,
            device_id=requested_device,
            resume_execution_id=resume_execution_id,
            resume_run_id=resume_run_id,
            on_execution_started=on_execution_started,
        )  # type: ignore[return-value]

    def _is_regime_satisfied(
        self,
        regime_spec: RegimeTargetSpec,
        best_result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> bool:
        anchor_nums = {
            a.case_num for a in active_anchors if a.regime_tag == regime_spec.regime_tag
        }
        matched = [a for a in best_result.anchors if a.case_num in anchor_nums]
        if not matched:
            return False
        for metric in matched:
            if not metric.accuracy_passed:
                return False
            if metric.roofline_eta < self._perf_target.target_roofline_eta:
                return False
            if metric.speedup < self._perf_target.target_min_case_speedup:
                return False
            max_us = self._perf_target.per_anchor_max_us.get(metric.case_num)
            if max_us is not None and metric.cand_us > max_us:
                return False
        return True

    def _allow_early_stop(self, rounds_completed: int) -> bool:
        if self._config.disable_early_goal_stop:
            return False
        return rounds_completed > 0 and rounds_completed >= self._perf_target.min_optimization_rounds

    def _round_regime_override(self, round_idx: int) -> RegimeTargetSpec | None:
        """Return the configured regime for *round_idx*, or None for adaptive rotation."""
        if round_idx > len(self._config.round_regime_tags):
            return None
        tag = self._config.round_regime_tags[round_idx - 1]
        if not tag:
            return None
        return next(spec for spec in self._regime_specs if spec.regime_tag == tag)

    def _select_active_regime(
        self,
        current_regime: str | None,
        frozen_regimes: set[str],
        best_result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec],
        *,
        allow_satisfied_fallback: bool = False,
        exclude_regime: str | None = None,
    ) -> RegimeTargetSpec | None:
        by_tag = {r.regime_tag: r for r in self._regime_specs}
        if current_regime and current_regime not in frozen_regimes and current_regime in by_tag:
            spec = by_tag[current_regime]
            if not self._is_regime_satisfied(spec, best_result, active_anchors):
                return spec

        candidates = [
            spec
            for spec in self._regime_specs
            if spec.regime_tag not in frozen_regimes
            and not self._is_regime_satisfied(spec, best_result, active_anchors)
        ]
        if not candidates and allow_satisfied_fallback:
            if current_regime and current_regime not in frozen_regimes and current_regime in by_tag:
                return by_tag[current_regime]
            candidates = [
                spec for spec in self._regime_specs if spec.regime_tag not in frozen_regimes
            ]
        if not candidates:
            return None
        if exclude_regime and any(spec.regime_tag != exclude_regime for spec in candidates):
            candidates = [spec for spec in candidates if spec.regime_tag != exclude_regime]
        candidates.sort(
            key=lambda s: best_result.regime_headroom(s.regime_tag),
            reverse=True,
        )
        return candidates[0]

    @staticmethod
    def _slice_contract_instruction(func_name: str, regime_spec: RegimeTargetSpec) -> str:
        if not regime_spec.editable_symbols:
            return f"function '{func_name}'"
        syms = ", ".join(regime_spec.editable_symbols)
        helper_note = (
            f" or '{regime_spec.allowed_helper_prefix}*' helpers"
            if regime_spec.allowed_helper_prefix
            else ""
        )
        return (
            f"authorized slice symbols ({syms}){helper_note}. You may output ONLY the "
            "subset of functions or prefixed macros you modified (e.g., just _select_*_config, "
            "a single helper macro, or one builder); omitted sibling symbols in this regime "
            "automatically inherit their parent implementation. Do NOT define or modify public "
            "entrypoints or symbols from other regimes"
        )

    @staticmethod
    def _format_msprof_section(residual: StaticDynamicResidual | None) -> str:
        if residual is None:
            return ""
        has_pmu = (
            residual.measured_aic_mac_ratio > 0.0
            or residual.measured_aiv_vec_ratio > 0.0
            or residual.measured_mte2_wait_ratio > 0.0
            or residual.measured_sync_stall_ratio > 0.0
        )
        has_compiler_hazard = (
            residual.spill_bytes > 0
            or (not residual.vloopv2_hit)
            or residual.vf_fusion_breaks > 0
            or residual.alias_serialized_buffers > 0
            or residual.dual_issue_ipc_collapse > 0.0
            or residual.dcache_dcci_flushes > 0
            or bool(residual.has_local_membar_in_vf_call)
        )
        if not has_pmu and not has_compiler_hazard:
            return ""
        lines = ["### HARDWARE MSPROF TELEMETRY"]
        if has_pmu:
            lines.extend(
                [
                    f"aic_mac_ratio: {residual.measured_aic_mac_ratio:.4f}",
                    f"aiv_vec_ratio: {residual.measured_aiv_vec_ratio:.4f}",
                    f"mte2_wait_ratio: {residual.measured_mte2_wait_ratio:.4f}",
                    f"sync_stall_ratio: {residual.measured_sync_stall_ratio:.4f}",
                    f"unexplained_gap_ratio: {residual.unexplained_gap_ratio:.4f}",
                ]
            )
        if has_compiler_hazard:
            lines.extend(
                [
                    f"spill_bytes: {residual.spill_bytes}",
                    f"vloopv2_hit: {residual.vloopv2_hit}",
                    f"vf_fusion_breaks: {residual.vf_fusion_breaks}",
                    f"alias_serialized_buffers: {residual.alias_serialized_buffers}",
                    f"dual_issue_ipc_collapse: {residual.dual_issue_ipc_collapse:.4f}",
                    f"dcache_dcci_flushes: {residual.dcache_dcci_flushes}",
                    f"has_local_membar_in_vf_call: {residual.has_local_membar_in_vf_call}",
                ]
            )
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def _format_sil_live_policy_section(
        *,
        live_plan: RegimePlanResult | None,
        active_mask: ConstraintRelaxationMask | None,
        escalation: object | None = None,
        pipe_gaps: object | None = None,
        cost_pred: object | None = None,
        escalated_trackb_tu: str = "",
    ) -> str:
        if live_plan is None and active_mask is None:
            return ""
        lines = ["### SIL LIVE POLICY & 7-FACET CONSTRAINT ENVELOPE"]
        if live_plan is not None:
            pol = live_plan.policy
            lines.append(
                f"- topology_axes: {live_plan.topology_axes.key()} "
                f"(solver_mode={live_plan.solver_mode}, backend_tier={pol.get('backend_tier', 'tilelang_dsl')})"
            )
            lines.append(
                f"- live_solved_knobs: tile_mnk={pol.get('tile_mnk')}, "
                f"row_pack_factor={pol.get('row_pack_factor', 1)}, "
                f"l1_stages={pol.get('l1_stages', 1)}, "
                f"ub_bytes={pol.get('ub_estimated_bytes', 0)}, "
                f"l1_bytes={pol.get('l1_estimated_bytes', 0)}"
            )
            if live_plan.notes:
                lines.append(f"- solver_note: {live_plan.notes[-1]}")
        if active_mask is not None:
            lines.append(f"- unlocked_facets: {', '.join(active_mask.unlocked_facets())}")
            locked = active_mask.locked_facets()
            lines.append(f"- locked_facets: {', '.join(locked) if locked else 'none (top)'}")
        if cost_pred is not None:
            lines.append(
                f"- cost_model_pred_us: {float(getattr(cost_pred, 'pred_us', 0.0)):.2f} "
                f"(dominant_bound={getattr(cost_pred, 'dominant_bound', '')}, "
                f"tier1_penalty_us={float(getattr(cost_pred, 'tier1_penalty_us', 0.0)):.2f})"
            )
        if escalation is not None:
            should_esc = bool(getattr(escalation, "should_escalate", False))
            to_tier = getattr(escalation, "to_tier", "")
            reasons = tuple(getattr(escalation, "trigger_reasons", ()) or ())
            lines.append(
                f"- track_escalation: should_escalate={should_esc}, to_tier={to_tier}, "
                f"reasons={list(reasons)}"
            )
        if escalated_trackb_tu.strip():
            lines.append("- escalated_trackb_seed_cpp:")
            lines.append("```cpp")
            lines.append(escalated_trackb_tu.strip())
            lines.append("```")
        if pipe_gaps is not None:
            cov = float(getattr(pipe_gaps, "coverage_ratio", 1.0))
            missing = tuple(getattr(pipe_gaps, "missing_primitives", ()) or ())
            lines.append(
                f"- tilelang_7pipe_isa_coverage: {cov:.2%}, missing_count={len(missing)}, "
                f"sample_gaps={list(missing[:6])}"
            )
        return "\n".join(lines) + "\n\n"

    def _build_cost_workload(
        self,
        ledger: ObjectivePhysicalLedger,
        *,
        live_plan: RegimePlanResult | None = None,
        axes: TopologyAxes | None = None,
        blueprint: HirBlueprintPacket | None = None,
        stage_b_profile: Mapping[str, object] | None = None,
    ) -> CostFeatures:
        return build_cost_features_from_ledger(
            ledger,
            live_plan=live_plan,
            axes=axes,
            blueprint=blueprint,
            sil_advisory=self._sil_advisory,
            target=self._target,
            stage_b_profile=stage_b_profile,
        )

    def _lookup_case_dims_safe(
        self,
        case_num: int,
        regime_tag: str = "",
    ) -> dict[str, object]:
        cnum = int(case_num)
        if cnum in self._case_dims_cache:
            return dict(self._case_dims_cache[cnum])
        dims: dict[str, object] = {}
        try:
            dims = dict(
                lookup_case_dims(
                    self._config.operator_name,
                    cnum,
                    cannbench_repo=self._cannbench_repo,
                    env_file=self._env_file,
                )
            )
        except (FileNotFoundError, OSError, KeyError, ValueError):
            for spec in self._regime_specs:
                if regime_tag and spec.regime_tag == regime_tag:
                    dims = dict(spec.representative_anchor_dims or spec.representative_hir_dims)
                    break
            if not dims and self._regime_specs:
                first = self._regime_specs[0]
                dims = dict(first.representative_anchor_dims or first.representative_hir_dims)
        self._case_dims_cache[cnum] = dict(dims)
        return dims

    def _record_kb_event(
        self,
        *,
        rule_id: str,
        postcondition_realized: bool,
        speed_gain: float = 0.0,
        topology_axes: object | None = None,
        round_idx: int = 0,
    ) -> None:
        if self._experience_lake_dir is None:
            return
        try:
            append_kb_event_shard(
                self._experience_lake_dir,
                run_id=self._run_id,
                rule_id=rule_id,
                postcondition_realized=postcondition_realized,
                speed_gain=speed_gain,
                topology_axes=topology_axes,
                operator_name=self._config.operator_name,
                round_idx=round_idx,
            )
        except (OSError, ValueError) as exc:
            logger.debug("KB event shard append skipped: %s", exc)

    def _record_surrogate_observations(
        self,
        *,
        source: str,
        result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec],
        eval_tier_override: str | None = None,
        parent_source: str = "",
        parent_result: CandidateSurrogateResult | None = None,
        parent_ledger: ObjectivePhysicalLedger | None = None,
        parent_bisheng_res: BishengResourceSummary | Mapping[str, object] | None = None,
        parent_stage_b_profile: Mapping[str, object] | None = None,
        parent_axes: TopologyAxes | None = None,
        ledger: ObjectivePhysicalLedger | None = None,
        bisheng_summary: BishengResourceSummary | Mapping[str, object] | None = None,
        stage_b_profile: Mapping[str, object] | None = None,
        axes: TopologyAxes | None = None,
        policy: Mapping[str, object] | None = None,
        worker_kind: str = "",
        rule_id: str | None = None,
        backend_track: str = "tilelang_dsl",
        regime_tag: str = "",
    ) -> None:
        cand_sha = compute_source_sha256(source)
        parent_sha = compute_source_sha256(parent_source) if parent_source else ""
        stage_a_prof = getattr(ledger, "microarch_profile", None) if ledger is not None else None
        merged_prof = merge_microarch_profiles(stage_a_prof, stage_b_profile)
        prof_dict = merged_prof.to_dict() if not merged_prof.is_empty else {}
        ledger_dict = ledger.to_dict() if ledger is not None else {}
        if hasattr(bisheng_summary, "to_dict"):
            bsum_dict = dict(bisheng_summary.to_dict())
        elif isinstance(bisheng_summary, Mapping):
            bsum_dict = dict(bisheng_summary)
        else:
            bsum_dict = {}
        if axes is not None and hasattr(axes, "to_dict"):
            axes_dict = dict(axes.to_dict())
        elif axes is not None:
            axes_dict = {
                "compute_engine": axes.compute_engine,
                "memory_ring": axes.memory_ring,
                "iteration_regularity": axes.iteration_regularity,
            }
        else:
            axes_dict = {}
        pol_dict = dict(policy or {})
        eff_regime = regime_tag or (result.anchors[0].regime_tag if result.anchors else "")

        if self._experience_lake is not None:
            try:
                self._experience_lake.register_candidate(
                    CandidateArtifactRecord(
                        cand_sha=cand_sha,
                        operator=self._config.operator_name,
                        regime_tag=eff_regime,
                        backend_track=backend_track,
                        parent_sha=parent_sha,
                        worker_kind=worker_kind,
                        rule_id=rule_id,
                        config_knobs=pol_dict,
                        topology_axes=axes_dict,
                        base_ledger=ledger_dict,
                        microarch_profile=prof_dict,
                        bisheng_summary=bsum_dict,
                        dsl_line_count=len(source.splitlines()),
                        subkernel_count=max(1, merged_prof.subkernel_count),
                    )
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                logger.debug("ExperienceLake register_candidate skipped: %s", exc)

        parent_by_case = (
            {int(a.case_num): a for a in parent_result.anchors}
            if parent_result is not None
            else {}
        )
        parent_prof = (
            merge_microarch_profiles(
                getattr(parent_ledger, "microarch_profile", None) if parent_ledger is not None else None,
                parent_stage_b_profile,
            )
            if (parent_ledger is not None or parent_stage_b_profile is not None)
            else None
        )
        bn_by_regime: dict[str, int] = {}
        for anc in result.anchors:
            if anc.regime_tag not in bn_by_regime:
                bn = bottleneck_anchor_for_regime(result, anc.regime_tag)
                bn_by_regime[anc.regime_tag] = int(bn.case_num) if bn is not None else -1

        observed_count = 0
        delta_count = 0
        msprof_count = 0
        for anc in result.anchors:
            if anc.cand_us <= 0.0:
                continue
            eval_tier = (
                eval_tier_override
                if eval_tier_override is not None
                else ("l2_pass" if anc.accuracy_passed else "l2_acc_fail")
            )
            dims = self._lookup_case_dims_safe(anc.case_num, anc.regime_tag)
            proj_wl = project_case_workload(
                ledger_dict,
                microarch_profile=prof_dict,
                case_dims=dims,
                policy=pol_dict,
                t_hw_us=anc.t_hw_us,
                axes=axes,
                target=self._target,
            )
            cand_msprof = self._msprof_cache.get((cand_sha, anc.regime_tag))
            bn_case = bn_by_regime.get(anc.regime_tag, -1)
            attach_msprof = (
                dict(cand_msprof)
                if (cand_msprof and (bn_case < 0 or int(anc.case_num) == bn_case))
                else {}
            )
            if self._experience_lake is not None:
                try:
                    self._experience_lake.append_observation(
                        CaseObservationRecord(
                            run_id=self._run_id,
                            operator=self._config.operator_name,
                            regime_tag=anc.regime_tag,
                            cand_sha=cand_sha,
                            eval_tier=eval_tier,
                            case_num=int(anc.case_num),
                            measured_us=float(anc.cand_us),
                            accuracy_passed=bool(anc.accuracy_passed),
                            parent_sha=parent_sha,
                            device_id=str(self._config.device_id),
                            case_dims=dims,
                            t_hw_us=float(anc.t_hw_us),
                            ref_perf_us=float(anc.ref_perf_us),
                            speedup=float(anc.speedup),
                            eq3_score=float(anc.eq3_score),
                            msprof_pmu=attach_msprof,
                            projected_tiles=int(proj_wl.get("total_tiles", 0) or 0),
                            projected_gm_bytes=float(proj_wl.get("gm_bytes", 0.0) or 0.0),
                            projected_contig_bytes=int(proj_wl.get("contig_burst_bytes", 32768) or 32768),
                        )
                    )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    logger.debug("ExperienceLake append_observation skipped: %s", exc)

            try:
                if attach_msprof:
                    self._cost_model.observe_msprof(
                        proj_wl,
                        attach_msprof,
                        measured_us=float(anc.cand_us),
                        bisheng_res=bisheng_summary,
                        persist=False,
                    )
                    msprof_count += 1
                self._cost_model.observe(
                    proj_wl,
                    float(anc.cand_us),
                    bisheng_res=bisheng_summary,
                    persist=False,
                    refit=False,
                )
                observed_count += 1
                p_anc = parent_by_case.get(int(anc.case_num))
                if (
                    p_anc is not None
                    and p_anc.cand_us > 0.0
                    and parent_sha
                    and parent_sha != cand_sha
                ):
                    parent_proj = project_case_workload(
                        parent_ledger.to_dict() if parent_ledger is not None else ledger_dict,
                        microarch_profile=(parent_prof.to_dict() if parent_prof is not None and not parent_prof.is_empty else prof_dict),
                        case_dims=dims,
                        policy=pol_dict,
                        t_hw_us=p_anc.t_hw_us,
                        axes=parent_axes or axes,
                        target=self._target,
                    )
                    self._cost_model.observe_pair_delta(
                        proj_wl,
                        float(anc.cand_us),
                        parent_proj,
                        float(p_anc.cand_us),
                        cand_bisheng_res=bisheng_summary,
                        parent_bisheng_res=parent_bisheng_res,
                        refit=False,
                    )
                    delta_count += 1
            except (OSError, ValueError) as exc:
                logger.debug("Cost model online observation skipped: %s", exc)

        if delta_count > 0:
            self._cost_model._refit_delta_ridge()
        if observed_count > 0 or msprof_count > 0:
            if observed_count > 0:
                self._cost_model._refit_ridge()
            if self._cost_calibration_path is not None:
                self._cost_model.save_calibration(self._cost_calibration_path)

    def _record_l4_observations(
        self,
        *,
        source: str,
        l4_result: L4CannbenchResult,
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> None:
        if not l4_result.per_case_elapsed_us:
            return
        cand_sha = compute_source_sha256(source)
        stage_a = None
        if self._regime_specs:
            try:
                stage_a = self._run_l0(source, self._regime_specs[0])
            except (SyntaxError, RuntimeError, ValueError, TypeError, OSError):
                stage_a = None
        ledger = stage_a.ledger if stage_a is not None else None
        axes = self._derive_topology_axes(
            ledger,
            features=stage_a.features if stage_a is not None else None,
            route=stage_a.route if stage_a is not None else None,
        )
        prof_dict = dict(getattr(ledger, "microarch_profile", None) or {})
        ledger_dict = ledger.to_dict() if ledger is not None else {}
        axes_dict = axes.to_dict() if hasattr(axes, "to_dict") else {}
        first_reg = self._regime_specs[0].regime_tag if self._regime_specs else ""
        if self._experience_lake is not None:
            try:
                self._experience_lake.register_candidate(
                    CandidateArtifactRecord(
                        cand_sha=cand_sha,
                        operator=self._config.operator_name,
                        regime_tag=first_reg,
                        backend_track=(
                            stage_a.route.track
                            if self._enable_track_router and stage_a is not None and stage_a.route is not None
                            else "tilelang_dsl"
                        ),
                        topology_axes=axes_dict,
                        base_ledger=ledger_dict,
                        microarch_profile=prof_dict,
                        dsl_line_count=len(source.splitlines()),
                    )
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                logger.debug("ExperienceLake register L4 candidate skipped: %s", exc)

        by_case_anchor = {int(a.case_num): a for a in active_anchors}
        observed = 0
        for cnum_raw, elapsed_raw in sorted(l4_result.per_case_elapsed_us.items()):
            cnum = int(cnum_raw)
            elapsed_us = float(elapsed_raw or 0.0)
            if elapsed_us <= 0.0:
                continue
            anc = by_case_anchor.get(cnum)
            rtag = anc.regime_tag if anc is not None else first_reg
            dims = self._lookup_case_dims_safe(cnum, rtag)
            if anc is not None:
                t_hw = anc.t_hw_us
                ref_perf = anc.ref_perf_us
            else:
                try:
                    ref_perf, t_hw = lookup_case_metadata(
                        self._config.operator_name,
                        cnum,
                        cannbench_repo=self._cannbench_repo,
                        env_file=self._env_file,
                    )
                except (FileNotFoundError, OSError, KeyError, ValueError):
                    ref_perf, t_hw = 0.0, 0.0
            sp = float(l4_result.per_case_speedups.get(cnum, 0.0) or 0.0)
            if sp <= 0.0 and ref_perf > 0.0 and elapsed_us > 0.0:
                sp = ref_perf / elapsed_us
            eq3 = (
                compute_case_eq3_score(ref_perf, elapsed_us, t_hw)
                if (ref_perf > 0.0 and t_hw > 0.0)
                else 0.0
            )
            proj_wl = project_case_workload(
                ledger_dict,
                microarch_profile=prof_dict,
                case_dims=dims,
                t_hw_us=t_hw,
                axes=axes,
                target=self._target,
            )
            if self._experience_lake is not None:
                try:
                    self._experience_lake.append_observation(
                        CaseObservationRecord(
                            run_id=self._run_id,
                            operator=self._config.operator_name,
                            regime_tag=rtag,
                            cand_sha=cand_sha,
                            eval_tier="l4_official",
                            case_num=cnum,
                            measured_us=elapsed_us,
                            accuracy_passed=bool(l4_result.passed),
                            device_id=str(self._config.device_id),
                            case_dims=dims,
                            t_hw_us=t_hw,
                            ref_perf_us=ref_perf,
                            speedup=sp,
                            eq3_score=eq3,
                            projected_tiles=int(proj_wl.get("total_tiles", 0) or 0),
                            projected_gm_bytes=float(proj_wl.get("gm_bytes", 0.0) or 0.0),
                            projected_contig_bytes=int(proj_wl.get("contig_burst_bytes", 32768) or 32768),
                        )
                    )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    logger.debug("ExperienceLake L4 append skipped: %s", exc)
            try:
                self._cost_model.observe(
                    proj_wl,
                    elapsed_us,
                    persist=False,
                    refit=False,
                )
                observed += 1
            except (OSError, ValueError) as exc:
                logger.debug("Cost model L4 observe skipped: %s", exc)
        if observed > 0:
            self._cost_model._refit_ridge()
            if self._cost_calibration_path is not None:
                self._cost_model.save_calibration(self._cost_calibration_path)

    def _prune_stale_sibling_stepping_stones(
        self,
        promoted_regime: RegimeTargetSpec,
        promoted_snippet: str,
    ) -> None:
        own_editable = set(promoted_regime.editable_symbols)
        if not own_editable:
            return
        promoted_dsl = extract_python_dsl(promoted_snippet)
        if (
            promoted_regime.runtime_backend in ("ascendc_trackb", "hybrid_trackb")
            and promoted_regime.trackb_symbols
            and parse_scoped_search_replace_blocks(promoted_dsl)
        ):
            promoted_syms = set(promoted_regime.trackb_symbols)
        else:
            promoted_syms = {
                sym.name for sym in _collect_top_level_symbols(ast.parse(promoted_dsl))
            }
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
                stone_dsl = extract_python_dsl(stone.dsl_source)
                if (
                    other_spec.runtime_backend in ("ascendc_trackb", "hybrid_trackb")
                    and other_spec.trackb_symbols
                    and parse_scoped_search_replace_blocks(stone_dsl)
                ):
                    stone_syms = set(other_spec.trackb_symbols)
                else:
                    stone_syms = {
                        sym.name for sym in _collect_top_level_symbols(ast.parse(stone_dsl))
                    }
                if not (stone_syms & shared_syms):
                    kept_stones.append(stone)
            other_beam.stepping_stone_beam = kept_stones

    @staticmethod
    @staticmethod
    def _format_trackb_hotpath_section(
        regime_spec: RegimeTargetSpec,
        active_kernel: str | None = None,
    ) -> tuple[str, str]:
        """Return `(escalation_tier, hotpath_prompt_section)` for *regime_spec*."""
        backend = normalize_runtime_backend(regime_spec.runtime_backend)
        esc_tier = (
            "ascendc_template"
            if backend in ("ascendc_trackb", "hybrid_trackb")
            else "tilelang_dsl"
        )
        if not regime_spec.trackb_symbols:
            return esc_tier, ""
        active_line = f"- active_kernel: {active_kernel}\n" if active_kernel else ""
        if "_render_trackb_tu" in regime_spec.editable_symbols:
            signature_rule = (
                "- _render_trackb_tu is editable: keep its callable ABI ``_render_trackb_tu(**dims)`` "
                "and the launcher dims it receives consistent with the launched kernel.\n"
            )
        else:
            signature_rule = (
                "- _render_trackb_tu is immutable: keep the launched kernel's ``template <...>`` "
                "and parameter signature unchanged so instantiation still binds.\n"
            )
        section = (
            "### ACTIVE RUNTIME HOT PATH\n"
            f"- runtime_backend: {regime_spec.runtime_backend}\n"
            f"- active_trackb_symbols: {list(regime_spec.trackb_symbols)}\n"
            f"{active_line}"
            "- CRITICAL: On NPU this regime executes the Track-B CCE (C++20) kernel string "
            "and config selector shown in TARGET FUNCTION SNIPPET, NOT dead TileLang fallback "
            "builders. Edit the active Track-B CCE (C++20) kernel string / selector directly.\n\n"
            "### TRACK-B REPLY CONTRACT\n"
            "- Reply with exactly one fenced block in one of these forms:\n"
            "  1. SEARCH/REPLACE hunks against the launched kernel and the helpers it calls "
            "(preferred); hunks outside that closed set are rejected.\n"
            "  2. One complete bare C++ unit per changed function: either the ``__global__`` "
            "kernel or one ``__simd_vf__`` helper, keeping its exact name and signature.\n"
            "  3. A Python slice that reassigns one authorized Track-B C++ constant in full.\n"
            "- Never echo context folding markers ``READ-ONLY SIBLING KERNEL``, "
            "``READ-ONLY HELPER``, or ``MIDDLE BODY OMITTED``.\n"
            "- A full C++ constant must still contain every sibling ``__global__`` kernel "
            "present in the original string; a replacement that drops one is rejected.\n"
            f"{signature_rule}"
            "- Put any selector change outside the hunks as a Python ``_select_*_config`` "
            "definition in the same fenced block.\n\n"
        )
        return esc_tier, section

    def _build_exploit_regime_request(
        self,
        *,
        snippet_source: str,
        func_name: str,
        regime_spec: RegimeTargetSpec,
        rule: AscendCausalRule,
        best_result: CandidateSurrogateResult,
        blueprint: HirBlueprintPacket | None = None,
        parent_ledger: ObjectivePhysicalLedger | None = None,
        parent_residual: StaticDynamicResidual | None = None,
        track: str = "unconstrained",
        incumbent_source: str = "",
        sil_policy_section: str = "",
        effort: str | None = None,
    ) -> GenerationRequest:
        module_source = incumbent_source or snippet_source
        active_kernel = self._resolve_active_trackb_kernel(module_source, regime_spec)
        snippet_source = _compact_trackb_snippet_source(
            snippet_source,
            regime_spec,
            incumbent_source=incumbent_source,
            max_bytes=14800,
            active_kernel=active_kernel,
        )
        slice_desc = self._slice_contract_instruction(func_name, regime_spec)
        esc_tier, hotpath_section = self._format_trackb_hotpath_section(
            regime_spec,
            active_kernel=active_kernel,
        )
        if regime_spec.trackb_symbols:
            sys_prompt = (
                "You are an Ascend 950PR kernel performance engineer. Apply the selected causal "
                "optimization rule cleanly while preserving exact numerical semantics, dynamic shape "
                f"compatibility, and {slice_desc}. Output ONLY the reply form required by the "
                "TRACK-B REPLY CONTRACT."
            )
        else:
            sys_prompt = (
                "You are an Ascend 950PR kernel performance engineer. Apply the selected causal "
                f"optimization rule cleanly while preserving exact numerical semantics, dynamic shape "
                f"compatibility, and {slice_desc}. Output ONLY Python code for this regime slice."
            )
        constraint_text = build_progressive_llm_constraints(
            escalation_tier=esc_tier,
            track=track,
            enable_track_router=bool(self._enable_track_router),
            sil_advisory=getattr(self, "_sil_advisory", None),
            freedom_mask=self._freedom_masks.get(regime_spec.regime_tag),
            ub_256kb_unlock=bool(parent_ledger.uses_unreserved_ubuf) if parent_ledger else False,
            blueprint=blueprint,
            current_ledger=parent_ledger,
        )
        track_section = f"{constraint_text}\n\n"
        nav_section = ""
        msprof_section = self._format_msprof_section(parent_residual)
        cfg_obj = getattr(self, "_config", None)
        op_name = str(
            getattr(cfg_obj, "operator_name", "") or getattr(cfg_obj, "operator", "") or ""
        )
        domain_env = get_input_domain_envelope(op_name)
        domain_section = (
            f"{domain_env.format_prompt_block()}\n\n" if domain_env is not None else ""
        )
        abi_section = (
            _extract_regime_abi_context(incumbent_source, regime_spec)
            if incumbent_source
            else ""
        )
        eff_rule = (
            rule.render_for_backend(regime_spec.runtime_backend)
            if hasattr(rule, "render_for_backend")
            else rule
        )
        user_prompt = (
            f"### TARGET REGIME\n{regime_spec.regime_tag}: {regime_spec.description}\n\n"
            f"{hotpath_section}"
            f"{track_section}"
            f"{sil_policy_section}"
            f"{nav_section}"
            f"### SELECTED CAUSAL RULE\n"
            f"Rule ID: {eff_rule.rule_id}\n"
            f"Precondition: {eff_rule.precondition}\n"
            f"Transform Template:\n{eff_rule.transform_template.strip()}\n"
            f"Expected Postcondition: {eff_rule.expected_postcondition}\n\n"
            f"{msprof_section}"
            f"{domain_section}"
            f"### SURROGATE ANCHOR TELEMETRY\n"
            f"{_format_anchor_telemetry(best_result, regime_spec.regime_tag)}\n\n"
            f"{abi_section}"
            f"### TARGET FUNCTION SNIPPET\n```python\n{snippet_source}\n```\n"
        )
        cfg_effort = getattr(getattr(self, "_config", None), "exploit_effort", "medium")
        eff = sanitize_effort(effort if effort is not None else cfg_effort, default="medium")
        return GenerationRequest(
            role="exploit_rule",
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            json_schema=None,
            effort=eff,
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
        parent_ledger: ObjectivePhysicalLedger | None = None,
        parent_residual: StaticDynamicResidual | None = None,
        track: str = "unconstrained",
        incumbent_source: str = "",
        parent_hypothesis: str = "",
        sil_policy_section: str = "",
    ) -> GenerationRequest:
        module_source = incumbent_source or snippet_source
        active_kernel = self._resolve_active_trackb_kernel(module_source, regime_spec)
        snippet_source = _compact_trackb_snippet_source(
            snippet_source,
            regime_spec,
            incumbent_source=incumbent_source,
            max_bytes=14800,
            active_kernel=active_kernel,
        )
        slice_desc = self._slice_contract_instruction(func_name, regime_spec)
        esc_tier, hotpath_section = self._format_trackb_hotpath_section(
            regime_spec,
            active_kernel=active_kernel,
        )
        if regime_spec.trackb_symbols:
            reply_desc = (
                'the TRACK-B REPLY CONTRACT form (SEARCH/REPLACE hunks, one bare C++ unit per '
                "changed function, or a Python slice reassigning one authorized constant)"
            )
        else:
            reply_desc = "Python code for this regime slice only"
        sys_prompt = (
            "You are an Ascend 950PR (dav-3510) systems architect. Propose an architectural "
            f"optimization grounded in Roofline limits (56 AIV / 28 MixCore blocks, 3 KiB SSBUF, "
            "L0C->UB direct path, UB->L1 direct path, 64-elem UB alignment, Split-KV parallelism). "
            f"Preserve exact numerical semantics, signatures, and {slice_desc}. Return ONLY a "
            'raw JSON object with exact keys "physical_hypothesis" (string), '
            '"target_local_metric" (string), and "dsl_source" (string containing '
            f"{reply_desc}; no prose outside JSON)."
        )
        if excluded_mechanisms:
            constraint_lines = "\n".join(f"- {item}" for item in excluded_mechanisms)
        else:
            constraint_lines = "- None recorded yet; explore any high-impact structural axis."
        neg_section = (
            "### NEGATIVE CONSTRAINTS (ALREADY EXPLORED BY PARALLEL WORKERS)\n"
            f"{constraint_lines}\n"
            "Propose a structurally orthogonal transformation that does NOT duplicate the "
            "mechanisms listed above.\n\n"
        )
        constraint_text = build_progressive_llm_constraints(
            escalation_tier=esc_tier,
            track=track,
            enable_track_router=bool(self._enable_track_router),
            sil_advisory=getattr(self, "_sil_advisory", None),
            freedom_mask=self._freedom_masks.get(regime_spec.regime_tag),
            ub_256kb_unlock=bool(parent_ledger.uses_unreserved_ubuf) if parent_ledger else False,
            blueprint=blueprint,
            current_ledger=parent_ledger,
            excluded_mechanisms=excluded_mechanisms,
        )
        track_section = f"{constraint_text}\n\n"
        nav_section = ""
        focus_section = f"### ARCHITECTURAL FOCUS HINT\n{focus_hint}\n\n" if focus_hint else ""
        parent_hyp_section = (
            f"### STEPPING-STONE PARENT HYPOTHESIS\n{parent_hypothesis}\n\n"
            if parent_hypothesis
            else ""
        )
        prescreen_section = (
            f"### TILE PRESCREEN GUIDANCE\n{prescreen_summary}\n\n"
            if prescreen_summary
            else ""
        )
        msprof_section = self._format_msprof_section(parent_residual)
        cfg_obj = getattr(self, "_config", None)
        op_name = str(
            getattr(cfg_obj, "operator_name", "") or getattr(cfg_obj, "operator", "") or ""
        )
        domain_env = get_input_domain_envelope(op_name)
        domain_section = (
            f"{domain_env.format_prompt_block()}\n\n" if domain_env is not None else ""
        )
        abi_section = (
            _extract_regime_abi_context(incumbent_source, regime_spec)
            if incumbent_source
            else ""
        )
        user_prompt = (
            f"### TARGET REGIME\n{regime_spec.regime_tag}: {regime_spec.description}\n\n"
            f"{hotpath_section}"
            f"{track_section}"
            f"{sil_policy_section}"
            f"{nav_section}"
            f"{focus_section}"
            f"{parent_hyp_section}"
            f"{neg_section}"
            f"{prescreen_section}"
            f"{msprof_section}"
            f"{domain_section}"
            f"### SURROGATE ANCHOR TELEMETRY\n"
            f"{_format_anchor_telemetry(best_result, regime_spec.regime_tag)}\n\n"
            f"{abi_section}"
            f"### TARGET FUNCTION SNIPPET\n```python\n{snippet_source}\n```\n"
        )
        return GenerationRequest(
            role="explore_creative",
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            json_schema=None,
            effort=effort,
        )

    def _evaluate_single_baseline(
        self,
        incumbent_source: str,
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> CandidateSurrogateResult:
        with tempfile.TemporaryDirectory(prefix="tf_camp_base_") as tmp_dir:
            mod_path = Path(tmp_dir) / "incumbent.py"
            mod_path.write_text(incumbent_source, encoding="utf-8")
            results, _run_ids = self._run_l2_batch(
                [("incumbent_baseline", mod_path)], active_anchors
            )
        if "incumbent_baseline" not in results:
            raise RuntimeError("Baseline evaluation missing incumbent_baseline result.")
        return results["incumbent_baseline"]

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
        pending_promotion: PendingPromotion | None = None,
        pending_l4: PendingL4 | None = None,
    ) -> None:
        if self._checkpoint_path is None:
            return
        active_promotion = self._active_promotion
        promotion = active_promotion or pending_promotion or self._pending_promotion
        pending_l4_record = pending_l4 or self._pending_l4
        payload = {
            "version": 1,
            "run_id": self._run_id,
            "operator_name": self._config.operator_name,
            "target_module_path": str(self._target_module_path),
            "incumbent_source": incumbent_source,
            "baseline_result": baseline_result.to_dict(),
            "best_result": best_result.to_dict(),
            "active_anchors": [a.to_dict() for a in active_anchors],
            "frozen_regimes": sorted(frozen_regimes),
            "current_regime": current_regime,
            "stagnation_rounds": stagnation_rounds,
            "stagnation_by_regime": {
                str(k): int(v) for k, v in sorted((stagnation_by_regime or {}).items())
            },
            "last_unpromoted_regime": last_unpromoted_regime,
            "rounds": [r.to_dict() for r in rounds],
            "promoted_rounds": list(promoted_rounds),
            "rolled_back_rounds": list(rolled_back_rounds),
            "escalated_rounds": list(escalated_rounds),
            "admitted_rule_ids": list(admitted_rule_ids),
            "tracker_state": self._tracker.to_dict(),
            "beam_state": self._beam.to_dict(),
            "beam_states": {tag: b.to_dict() for tag, b in sorted(self._beams.items())},
            "msprof_cache": {
                f"{sha}:{tag}": dict(metrics)
                for (sha, tag), metrics in sorted(self._msprof_cache.items())
            },
            "stop_reason": stop_reason,
            "l4_result": l4_result.to_dict() if l4_result else None,
            "pending_promotion": promotion.to_dict() if promotion is not None else None,
            "active_promotion": (
                active_promotion.to_dict() if active_promotion is not None else None
            ),
            "pending_l4": (
                pending_l4_record.to_dict() if pending_l4_record is not None else None
            ),
        }
        _atomic_write_text(
            self._checkpoint_path,
            json.dumps(payload, indent=2) + "\n",
        )

    def _restore_from_checkpoint(
        self,
    ) -> tuple[
        str,
        CandidateSurrogateResult,
        CandidateSurrogateResult,
        list[AnchorCaseSpec],
        set[str],
        str | None,
        int,
        list[CampaignRoundRecord],
        list[int],
        list[int],
        list[int],
        list[str],
        dict[str, int],
        str | None,
        PendingPromotion | None,
        PendingL4 | None,
    ]:
        """Rehydrate campaign state and re-prove the bytes a live execution owns.

        An ``active_promotion`` record was persisted by the submission callback
        and owns the live execution identity even when the round-level pending
        record still describes an earlier attempt. A non-terminal execution
        keeps exactly the source it was submitted against; replacing it with
        the incumbent would silently change the code under a queued or running
        worker.
        """
        assert self._checkpoint_path is not None
        payload = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
        saved_run_id = payload.get("run_id")
        if isinstance(saved_run_id, str) and saved_run_id.strip():
            self._run_id = saved_run_id.strip()
        incumbent_source = str(payload["incumbent_source"])
        if self._campaign_kb_path is not None and self._campaign_kb_path.is_file():
            for rule in load_seed_rules(self._campaign_kb_path):
                self._tracker.admit_rule(rule)
        if isinstance(payload.get("tracker_state"), Mapping):
            self._tracker.load_posteriors_dict(payload["tracker_state"])
        current_regime = payload.get("current_regime")
        if current_regime and str(current_regime) in self._beams:
            self._active_regime_tag = str(current_regime)
        if isinstance(payload.get("beam_states"), Mapping):
            for tag, raw_beam in payload["beam_states"].items():
                if isinstance(raw_beam, Mapping):
                    self._beams[str(tag)] = TwoLaneBeamSearch.from_dict(raw_beam)
        elif isinstance(payload.get("beam_state"), Mapping):
            target_tag = self._active_regime_tag or (
                self._regime_specs[0].regime_tag if self._regime_specs else "default"
            )
            self._beams[target_tag] = TwoLaneBeamSearch.from_dict(payload["beam_state"])
        self._msprof_cache = {}
        raw_msprof_cache = payload.get("msprof_cache")
        if isinstance(raw_msprof_cache, Mapping):
            for raw_key, raw_metrics in raw_msprof_cache.items():
                key_str = str(raw_key)
                if ":" not in key_str or not isinstance(raw_metrics, Mapping):
                    continue
                sha, tag = key_str.split(":", 1)
                self._msprof_cache[(sha, tag)] = {
                    str(k): float(v)
                    for k, v in raw_metrics.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
        rounds = [CampaignRoundRecord.from_dict(r) for r in payload.get("rounds", ())]
        raw_stag_map = payload.get("stagnation_by_regime")
        if isinstance(raw_stag_map, Mapping):
            stagnation_by_regime = {str(k): int(v) for k, v in raw_stag_map.items()}
        else:
            stagnation_by_regime = {}
            legacy_stag = int(payload.get("stagnation_rounds", 0))
            if legacy_stag > 0:
                fallback_tag = str(current_regime or (rounds[-1].regime_tag if rounds else ""))
                if fallback_tag:
                    stagnation_by_regime[fallback_tag] = legacy_stag
        raw_last_unpromoted = payload.get("last_unpromoted_regime")
        last_unpromoted_regime = str(raw_last_unpromoted) if raw_last_unpromoted else None
        raw_pending = payload.get("pending_promotion")
        pending_promotion = (
            PendingPromotion.from_dict(raw_pending)
            if isinstance(raw_pending, Mapping)
            else None
        )
        raw_active = payload.get("active_promotion")
        active_promotion = (
            PendingPromotion.from_dict(raw_active)
            if isinstance(raw_active, Mapping)
            else None
        )
        if active_promotion is not None:
            pending_promotion = active_promotion
        raw_pending_l4 = payload.get("pending_l4")
        pending_l4 = (
            PendingL4.from_dict(raw_pending_l4)
            if isinstance(raw_pending_l4, Mapping)
            else None
        )
        if pending_promotion is not None and not pending_promotion.execution_terminal:
            restore_error = self._ensure_candidate_bytes(
                path=self._target_module_path,
                source=pending_promotion.candidate_source,
                expected_sha256=pending_promotion.candidate_sha256,
            )
        elif pending_l4 is not None and not pending_l4.execution_terminal:
            restore_error = self._ensure_candidate_bytes(
                path=self._target_module_path,
                source=incumbent_source,
                expected_sha256=pending_l4.candidate_sha256,
            )
        else:
            restore_error = self._write_source_and_verify(
                self._target_module_path, incumbent_source
            )
        if restore_error:
            raise RuntimeError(restore_error)
        return (
            incumbent_source,
            CandidateSurrogateResult.from_dict(payload["baseline_result"]),
            CandidateSurrogateResult.from_dict(payload["best_result"]),
            [AnchorCaseSpec.from_dict(item) for item in payload.get("active_anchors", ())],
            set(str(x) for x in payload.get("frozen_regimes", ())),
            current_regime,
            int(payload.get("stagnation_rounds", 0)),
            rounds,
            [int(x) for x in payload.get("promoted_rounds", ())],
            [int(x) for x in payload.get("rolled_back_rounds", ())],
            [int(x) for x in payload.get("escalated_rounds", ())],
            [str(x) for x in payload.get("admitted_rule_ids", ())],
            stagnation_by_regime,
            last_unpromoted_regime,
            pending_promotion,
            pending_l4,
        )

    def _resolve_round_parent(
        self,
        *,
        incumbent_source: str,
        regime_spec: RegimeTargetSpec,
        regime_beam: TwoLaneBeamSearch,
        escalated: bool,
    ) -> tuple[str, str, str, BeamCandidate | None, str]:
        """Select parent module, full slice snippet, lane, and stepping stone cleanly."""
        _, default_snippet = extract_regime_snippet(incumbent_source, regime_spec)
        if not escalated:
            return incumbent_source, default_snippet, "exploit", None, ""
        for stone in regime_beam.stepping_stone_beam:
            try:
                stone_patched = patch_regime_into_module(
                    incumbent_source,
                    regime_spec,
                    stone.dsl_source,
                    operator_entrypoint=_resolve_operator_entrypoint_safe(
                        self._config.operator_name
                    ),
                    sibling_specs=self._regime_specs,
                    active_kernel=self._resolve_active_trackb_kernel(
                        incumbent_source,
                        regime_spec,
                    ),
                )
                l0_probe = self._run_l0(stone_patched, regime_spec)
                if not l0_probe.passed:
                    logger.warning(
                        "Skipping invalid stepping-stone %s for regime %s: %s",
                        stone.candidate_id,
                        regime_spec.regime_tag,
                        l0_probe.error_message[:160],
                    )
                    continue
                _, hydrated_snippet = extract_regime_snippet(stone_patched, regime_spec)
                regime_beam.mark_parent_selected(stone)
                hyp = stone.hypothesis if stone.worker_kind == "creative_explore" else ""
                return stone_patched, hydrated_snippet, "stepping_stone", stone, hyp
            except (SyntaxError, ValueError, KeyError) as exc:
                logger.warning(
                    "Failed to patch stepping-stone slice %s for regime %s: %s",
                    stone.candidate_id,
                    regime_spec.regime_tag,
                    exc,
                )
        return incumbent_source, default_snippet, "stepping_stone", None, ""

    @staticmethod
    def _evaluate_causal_rule_outcome(
        *,
        rule: AscendCausalRule,
        res_all_passed: bool,
        rolled_back: bool,
        ledger: ObjectivePhysicalLedger,
        features: object,
        cand_residual: StaticDynamicResidual,
        cand_lat_us: float,
        cand_obj_score: float,
        parent_ledger: ObjectivePhysicalLedger,
        parent_residual: StaticDynamicResidual,
        parent_lat_us: float,
        parent_obj_score: float,
        has_cand_msprof: bool,
        cand_global_score: float | None = None,
        parent_global_score: float | None = None,
        min_speedup_gain: float = 0.01,
        min_score_gain: float = 0.05,
    ) -> tuple[bool, float]:
        """Evaluate whether a causal postcondition materialized with non-regressive gain."""
        if not res_all_passed or not ledger.is_legal or rolled_back:
            return False, 0.0
        if (
            cand_global_score is not None
            and parent_global_score is not None
            and cand_global_score < parent_global_score - 1e-6
        ):
            return False, 0.0
        if parent_lat_us > 0.0 and cand_lat_us > 0.0 and cand_lat_us >= parent_lat_us:
            return False, 0.0
        if cand_obj_score < parent_obj_score - 1e-6:
            return False, 0.0

        rel_lat_gain = (
            (parent_lat_us - cand_lat_us) / max(parent_lat_us, 1e-6)
            if (parent_lat_us > 0.0 and cand_lat_us > 0.0)
            else 0.0
        )
        local_score_gain = cand_obj_score - parent_obj_score
        has_real_gain = (
            rel_lat_gain >= float(min_speedup_gain)
            or (
                local_score_gain >= float(min_score_gain)
                and (parent_lat_us <= 0.0 or cand_lat_us <= 0.0 or cand_lat_us < parent_lat_us)
            )
        )
        if not has_real_gain:
            return False, 0.0

        post_ctx = build_predicate_context(
            ledger,
            features=features,
            residual=cand_residual,
            prev_ledger=parent_ledger,
        )
        prev_ns = post_ctx.setdefault("prev", {})
        if isinstance(prev_ns, dict):
            prev_ns.update(parent_residual.to_dict())
            prev_ns["measured_latency_us"] = float(parent_lat_us)
        if not has_cand_msprof:
            for ratio_key in (
                "measured_aic_mac_ratio",
                "measured_aiv_vec_ratio",
                "measured_mte2_wait_ratio",
                "measured_sync_stall_ratio",
                "unexplained_gap_ratio",
            ):
                post_ctx[ratio_key] = getattr(parent_residual, ratio_key)

        causal_ok = rule.postcondition_realized(post_ctx)
        if not causal_ok and not has_cand_msprof:
            pmu_fields = (
                "measured_mte2_wait_ratio",
                "measured_sync_stall_ratio",
                "unexplained_gap_ratio",
                "measured_aiv_vec_ratio",
                "measured_aic_mac_ratio",
            )
            if any(k in rule.expected_postcondition for k in pmu_fields):
                causal_ok = bool(
                    parent_lat_us > 0.0
                    and cand_lat_us < 0.98 * parent_lat_us
                    and cand_obj_score > parent_obj_score + 0.02
                )
        return bool(causal_ok), (rel_lat_gain if causal_ok else 0.0)

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
    ) -> list[str]:
        """Distill and verify causal rules on hardware through AgentGates."""
        if self._flywheel is None:
            return []
        regime_anchors = [
            a for a in active_anchors if a.regime_tag == regime_spec.regime_tag
        ] or list(active_anchors)
        anchor_dims = (
            regime_spec.representative_anchor_dims or regime_spec.representative_hir_dims
        )

        def _batch_chunk_latencies(chunk_pairs: Sequence[tuple[str, str]]) -> dict[str, float]:
            if not chunk_pairs or not regime_anchors:
                return {}
            with tempfile.TemporaryDirectory(prefix="tf_fly_batch_") as tmp_dir:
                cand_files: list[tuple[str, Path]] = []
                for chunk_id, ablated_src in chunk_pairs:
                    cpath = Path(tmp_dir) / f"{chunk_id}.py"
                    cpath.write_text(ablated_src, encoding="utf-8")
                    cand_files.append((chunk_id, cpath))
                try:
                    batch_res, _run_ids = self._run_l2_batch(cand_files, regime_anchors)
                except (OSError, subprocess.SubprocessError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
                    logger.debug("Flywheel batched ablation timing skipped: %s", exc)
                    return {}
            out: dict[str, float] = {}
            for cid, res in batch_res.items():
                if res.all_passed and res.mean_latency_us() > 0.0:
                    out[cid] = res.mean_latency_us()
            return out

        distill_rep = self._flywheel.distill_from_pair(
            baseline_dsl=baseline_source,
            winner_dsl=winner_source,
            baseline_latency_us=baseline_latency_us,
            winner_latency_us=winner_latency_us,
            factory_name=func_name,
            factory_kwargs=regime_spec.sample_builder_kwargs,
            config_selector_name=regime_spec.config_selector_fn or None,
            dims=anchor_dims or None,
            batch_latency_evaluator=_batch_chunk_latencies,
            topology_axes=topology_axes,
            kb_output_path=self._campaign_kb_path,
            rule_id_prefix="R_FLYWHEEL",
            verify_on_candidate=(rel_gain >= 0.05 and self._l2_batch_evaluator is not None),
            max_loo_chunks=3,
            operator_name=self._config.operator_name,
            rule_scope="operator_local",
        )
        if self._experience_lake_dir is not None:
            for rid in distill_rep.admitted_rule_ids:
                rule_obj = self._tracker.rules.get(rid)
                if rule_obj is not None:
                    try:
                        save_distilled_rule_file(self._experience_lake_dir, rule_obj)
                    except (OSError, ValueError) as exc:
                        logger.debug("Persisting distilled rule %s skipped: %s", rid, exc)
        return list(distill_rep.admitted_rule_ids)

    def _can_retry_on_alt_card(self, pending: PendingPromotion, attempted_device: str) -> bool:
        retry_device = self._retry_device_id
        return bool(
            retry_device
            and not pending.retry_used
            and retry_device != (attempted_device or pending.device_id)
        )

    @staticmethod
    def _promotion_attempt(result: L3AcceptanceResult, device_id: str) -> dict[str, str]:
        """One auditable L3 attempt, including the card and candidate identity."""
        return {
            "execution_id": result.execution_id,
            "run_id": result.run_id,
            "receipt_path": result.receipt_path,
            "error_message": result.error_message,
            "failure_kind": result.failure_kind,
            "failure_category": result.failure_category,
            "pending": str(
                result.pending
                or result.failure_kind != FAILURE_KIND_CANDIDATE
                or not result.execution_terminal
            ).lower(),
            "device_id": result.device_id or device_id,
            "state": result.state,
            "candidate_sha256": result.candidate_sha256,
            "execution_terminal": str(result.execution_terminal).lower(),
        }

    def _retry_l3_on_alt_card(
        self,
        pending: PendingPromotion,
        attempts: tuple[dict[str, str], ...],
    ) -> L3AcceptanceResult:
        """Start one fresh L3 execution on the explicitly authorized alternate card."""
        retry_device = self._retry_device_id
        assert retry_device
        retry_pending = replace(
            pending,
            execution_id="",
            run_id="",
            receipt_path="",
            error_message="",
            attempts=attempts,
            device_id=retry_device,
            execution_terminal=False,
            retry_used=True,
        )

        def _record_started(execution_id: str, device: str) -> None:
            started = replace(
                retry_pending,
                execution_id=execution_id,
                device_id=device,
                execution_terminal=False,
            )
            self._active_promotion = started
            self._pending_promotion = started
            self._persist_runtime_state("active_promotion", started.to_dict())

        result = self._run_l3_contract(
            self._target_module_path,
            device_id=retry_device,
            on_execution_started=_record_started,
        )
        self._active_promotion = None
        return result

    @staticmethod
    def _record_promotion_attempts(
        rounds: list[CampaignRoundRecord],
        pending: PendingPromotion,
        attempts: tuple[dict[str, str], ...],
        *,
        promoted: bool,
        rolled_back: bool,
        stagnation_after: int,
    ) -> None:
        for idx, record in enumerate(rounds):
            if record.round_idx == pending.round_idx:
                rounds[idx] = replace(
                    record,
                    promoted=promoted,
                    rolled_back_l3=rolled_back,
                    stagnation_rounds_after=stagnation_after,
                    promotion_attempts=attempts,
                )
                break

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
    ) -> tuple[
        str,
        CandidateSurrogateResult,
        dict[str, int],
        PendingPromotion | None,
        bool,
    ]:
        """Resume, retry once on another card, or return a recoverable pending L3.

        Candidate bytes are re-proved first. A live execution has no run id and a
        terminal one with a run id can have its receipt re-read; neither path
        submits again. A wait timeout keeps the candidate and continues the same
        execution. Only a proven-terminal infrastructure failure restores the
        incumbent, and only then may one fresh retry start on an explicitly
        authorized other card. Numeric/assertion candidate failures never retry.
        """
        _ = active_anchors
        ensure_error = self._ensure_candidate_bytes(
            path=self._target_module_path,
            source=pending.candidate_source,
            expected_sha256=pending.candidate_sha256,
        )
        if ensure_error:
            updated = replace(pending, error_message=ensure_error)
            self._pending_promotion = updated
            self._active_promotion = updated if not updated.execution_terminal else None
            return incumbent_source, best_result, stagnation_by_regime, updated, True

        attempts = list(pending.attempts)
        current = pending

        def _keep_pending(result: L3AcceptanceResult, *, terminal: bool) -> tuple[
            str,
            CandidateSurrogateResult,
            dict[str, int],
            PendingPromotion | None,
            bool,
        ]:
            nonlocal current
            attempt = self._promotion_attempt(result, current.device_id)
            attempts.append(attempt)
            current = replace(
                current,
                execution_id=result.execution_id or current.execution_id,
                run_id=result.run_id or current.run_id,
                receipt_path=result.receipt_path or current.receipt_path,
                error_message=result.error_message or current.error_message,
                attempts=tuple(attempts),
                device_id=result.device_id or current.device_id,
                execution_terminal=terminal,
            )
            if terminal:
                restore_error = self._write_source_and_verify(
                    self._target_module_path, incumbent_source
                )
                if restore_error:
                    raise RuntimeError(restore_error)
                self._active_promotion = None
                self._persist_runtime_state("active_promotion", None)
            else:
                self._active_promotion = current
                self._persist_runtime_state("active_promotion", current.to_dict())
            self._pending_promotion = current
            self._persist_runtime_state("pending_promotion", current.to_dict())
            self._record_promotion_attempts(
                rounds,
                current,
                tuple(attempts),
                promoted=False,
                rolled_back=terminal,
                stagnation_after=stagnation_by_regime.get(current.regime_tag, 0),
            )
            return incumbent_source, best_result, stagnation_by_regime, current, True

        def _promote() -> tuple[
            str,
            CandidateSurrogateResult,
            dict[str, int],
            PendingPromotion | None,
            bool,
        ]:
            nonlocal current
            self._record_promotion_attempts(
                rounds,
                current,
                tuple(attempts),
                promoted=True,
                rolled_back=False,
                stagnation_after=0,
            )
            if current.round_idx in rolled_back_rounds:
                rolled_back_rounds.remove(current.round_idx)
            if current.round_idx not in promoted_rounds:
                promoted_rounds.append(current.round_idx)
            stagnation_by_regime[current.regime_tag] = 0
            self._pending_promotion = None
            self._active_promotion = None
            self._persist_runtime_state("active_promotion", None)
            self._persist_runtime_state("pending_promotion", None)
            return (
                current.candidate_source,
                current.candidate_result,
                stagnation_by_regime,
                None,
                False,
            )

        def _candidate_failure(result: L3AcceptanceResult) -> tuple[
            str,
            CandidateSurrogateResult,
            dict[str, int],
            PendingPromotion | None,
            bool,
        ]:
            nonlocal current
            attempts.append(self._promotion_attempt(result, current.device_id))
            restore_error = self._write_source_and_verify(
                self._target_module_path, incumbent_source
            )
            if restore_error:
                raise RuntimeError(restore_error)
            stagnation_by_regime[current.regime_tag] = (
                stagnation_by_regime.get(current.regime_tag, 0) + 1
            )
            self._record_promotion_attempts(
                rounds,
                current,
                tuple(attempts),
                promoted=False,
                rolled_back=True,
                stagnation_after=stagnation_by_regime[current.regime_tag],
            )
            self._pending_promotion = None
            self._active_promotion = None
            self._persist_runtime_state("active_promotion", None)
            self._persist_runtime_state("pending_promotion", None)
            return incumbent_source, best_result, stagnation_by_regime, None, False

        result: L3AcceptanceResult | None = None
        if pending.execution_id and (not pending.execution_terminal or pending.run_id):
            result = self._resume_l3_contract(self._target_module_path, pending)
            attempts.append(self._promotion_attempt(result, pending.device_id))
            current = replace(
                pending,
                execution_id=result.execution_id or pending.execution_id,
                run_id=result.run_id or pending.run_id,
                receipt_path=result.receipt_path or pending.receipt_path,
                error_message=result.error_message or pending.error_message,
                attempts=tuple(attempts),
                device_id=result.device_id or pending.device_id,
                execution_terminal=result.execution_terminal,
            )
            if result.passed:
                return _promote()
            if not result.execution_terminal:
                return _keep_pending(result, terminal=False)
            if result.failure_kind == FAILURE_KIND_CANDIDATE:
                return _candidate_failure(result)
            if self._can_retry_on_alt_card(current, result.device_id):
                retry_result = self._retry_l3_on_alt_card(current, tuple(attempts))
                attempts.append(
                    self._promotion_attempt(retry_result, current.device_id)
                )
                current = replace(
                    current,
                    execution_id=retry_result.execution_id or current.execution_id,
                    run_id=retry_result.run_id or current.run_id,
                    receipt_path=retry_result.receipt_path or current.receipt_path,
                    error_message=retry_result.error_message or current.error_message,
                    attempts=tuple(attempts),
                    device_id=retry_result.device_id or current.device_id,
                    execution_terminal=retry_result.execution_terminal,
                    retry_used=True,
                )
                if retry_result.passed:
                    return _promote()
                if not retry_result.execution_terminal:
                    return _keep_pending(retry_result, terminal=False)
                if retry_result.failure_kind == FAILURE_KIND_CANDIDATE:
                    return _candidate_failure(retry_result)
                return _keep_pending(retry_result, terminal=retry_result.execution_terminal)
            return _keep_pending(result, terminal=True)

        if self._can_retry_on_alt_card(pending, pending.device_id):
            retry_result = self._retry_l3_on_alt_card(pending, tuple(attempts))
            attempts.append(self._promotion_attempt(retry_result, pending.device_id))
            current = replace(
                pending,
                execution_id=retry_result.execution_id or pending.execution_id,
                run_id=retry_result.run_id or pending.run_id,
                receipt_path=retry_result.receipt_path or pending.receipt_path,
                error_message=retry_result.error_message or pending.error_message,
                attempts=tuple(attempts),
                device_id=retry_result.device_id or pending.device_id,
                execution_terminal=retry_result.execution_terminal,
                retry_used=True,
            )
            if retry_result.passed:
                return _promote()
            if not retry_result.execution_terminal:
                return _keep_pending(retry_result, terminal=False)
            if retry_result.failure_kind == FAILURE_KIND_CANDIDATE:
                return _candidate_failure(retry_result)
            return _keep_pending(retry_result, terminal=retry_result.execution_terminal)
        self._pending_promotion = pending
        self._active_promotion = pending if not pending.execution_terminal else None
        return incumbent_source, best_result, stagnation_by_regime, pending, True

    @staticmethod
    def _l4_attempt(result: L4CannbenchResult) -> dict[str, str]:
        """One auditable L4 attempt, including the card and official coverage."""
        return {
            "execution_id": result.execution_id,
            "run_id": result.run_id,
            "receipt_path": result.receipt_path,
            "error_message": result.error_message,
            "failure_kind": result.failure_kind,
            "device_id": result.device_id,
            "state": result.state,
            "execution_terminal": str(result.execution_terminal).lower(),
            "blocked_env_category": result.blocked_env_category,
            "device_rejected": str(result.device_rejected).lower(),
            "official_report_valid": str(result.official_report_valid).lower(),
            "all_cases_passed": str(result.all_cases_passed).lower(),
            "all_timings_valid": str(result.all_timings_valid).lower(),
            "case_count": str(result.case_count),
            "passed_cases": str(result.passed_cases),
        }

    def _can_retry_l4_on_alt_card(
        self,
        pending: PendingL4,
        attempted_device: str,
    ) -> bool:
        retry_device = self._retry_device_id
        attempted = attempted_device or pending.device_id
        return bool(
            retry_device
            and not pending.retry_used
            and retry_device != attempted
            and retry_device not in self._l4_round_attempted_devices(pending)
        )

    @staticmethod
    def _l4_round_attempted_devices(pending: PendingL4) -> tuple[str, ...]:
        """Cards already tried in the current blocked-env round.

        Checkpoints written before the pool existed have no explicit round
        list; their current card is the same-card attempt that record describes.
        """
        if pending.blocked_env_round_attempted_device_ids:
            return pending.blocked_env_round_attempted_device_ids
        return (pending.device_id,) if pending.device_id else ()

    def _effective_l4_round_limit(self, pending: PendingL4) -> int:
        """Return the stored quota, or the current config for legacy records."""
        if pending.blocked_env_round_limit is not None:
            return max(0, pending.blocked_env_round_limit)
        return max(0, self._config.l4_blocked_env_max_retries)

    def _effective_l4_pool(self, pending: PendingL4) -> tuple[str, ...]:
        """Intersect a saved pool with the pool the current config authorizes."""
        if not pending.device_pool:
            return self._l4_device_pool
        allowed = set(self._l4_device_pool)
        narrowed = tuple(device for device in pending.device_pool if device in allowed)
        return narrowed or self._l4_device_pool

    def _advance_l4_card_failure(
        self,
        pending: PendingL4,
        result: L4CannbenchResult,
        attempts: list[dict[str, str]],
    ) -> tuple[L4CannbenchResult, PendingL4]:
        """Move a terminal L4 run across the configured card pool.

        An occupancy-blocked card or a submit refusal for an unauthorized card
        advances immediately to the next untried card: the complete official L4
        restarts there for the same candidate SHA, and no timings are stitched.
        A rejected card stays excluded on every later round. Only after every
        remaining card was occupancy-blocked does one bounded linear backoff run
        before the next round, while a rejected-only sweep terminates without
        spending a round. A candidate verdict, a live execution or exhausted
        rounds leave the loop. Each submission intent is persisted before its
        long wait, so a restart continues on the next untried card instead of
        spending a recorded round twice.
        """
        pool = self._effective_l4_pool(pending)
        limit = self._effective_l4_round_limit(pending)
        current = replace(
            self._pending_l4_after(
                pending,
                result,
                tuple(attempts),
                retry_used=pending.retry_used,
            ),
            device_pool=pool,
            blocked_env_round_limit=limit,
        )

        def _record_started(execution_id: str, device: str) -> None:
            nonlocal current
            current = replace(
                current,
                execution_id=execution_id or current.execution_id,
                device_id=device or current.device_id,
                state="SUBMITTED",
                execution_terminal=False,
            )
            self._pending_l4 = current
            self._persist_runtime_state("pending_l4", current.to_dict())

        while result.execution_terminal and (
            _is_occupancy_blocked_env(
                result.failure_kind,
                result.blocked_env_category,
            )
            or self._is_rejected_card_l4(result)
        ):
            rejected = self._is_rejected_card_l4(result)
            attempted = self._l4_round_attempted_devices(current)
            eligible = tuple(
                device for device in pool if device not in current.rejected_device_ids
            )
            untried = tuple(device for device in eligible if device not in attempted)
            if untried:
                next_device = untried[0]
                next_rounds = current.blocked_env_rounds_used
                next_attempted = (*attempted, next_device)
                delay = 0.0
            elif (
                rejected
                or not eligible
                or current.blocked_env_rounds_used >= limit
            ):
                break
            else:
                next_rounds = current.blocked_env_rounds_used + 1
                next_device = eligible[0]
                next_attempted = (next_device,)
                delay = max(0.0, self._config.l4_retry_backoff_sec) * next_rounds
            current = replace(
                current,
                execution_id="",
                run_id="",
                receipt_path="",
                device_id=next_device,
                state="SUBMITTED",
                error_message=result.error_message,
                failure_kind=FAILURE_KIND_INFRASTRUCTURE,
                execution_terminal=False,
                blocked_env_rounds_used=next_rounds,
                blocked_env_round_attempted_device_ids=next_attempted,
                blocked_env_category="",
                device_rejected=False,
            )
            self._pending_l4 = current
            self._persist_runtime_state("pending_l4", current.to_dict())
            if delay > 0.0:
                time.sleep(delay)
            result = self._run_l4_cannbench(
                self._target_module_path,
                device_id=next_device,
                on_execution_started=_record_started,
            )
            attempts.append(self._l4_attempt(result))
            current = self._pending_l4_after(
                current,
                result,
                tuple(attempts),
                retry_used=current.retry_used,
            )
            self._pending_l4 = current
            self._persist_runtime_state("pending_l4", current.to_dict())
        return result, current

    @staticmethod
    def _merge_l4_identity(
        result: L4CannbenchResult,
        pending: PendingL4,
    ) -> L4CannbenchResult:
        return replace(
            result,
            target_path=result.target_path or pending.target_path,
            candidate_sha256=result.candidate_sha256 or pending.candidate_sha256,
            device_id=result.device_id or pending.device_id,
            blocked_env_category=result.blocked_env_category or pending.blocked_env_category,
            attempts=pending.attempts or result.attempts,
        )

    @staticmethod
    def _pending_l4_after(
        pending: PendingL4,
        result: L4CannbenchResult,
        attempts: tuple[dict[str, str], ...],
        *,
        retry_used: bool,
    ) -> PendingL4:
        rejected_device_ids = pending.rejected_device_ids
        if (
            result.device_rejected
            and result.device_id
            and result.device_id not in rejected_device_ids
        ):
            rejected_device_ids = (*rejected_device_ids, result.device_id)
        attempted_device_ids = pending.blocked_env_round_attempted_device_ids
        if (
            result.device_id
            and (
                result.device_rejected
                or _is_occupancy_blocked_env(
                    result.failure_kind,
                    result.blocked_env_category,
                )
            )
            and result.device_id not in attempted_device_ids
        ):
            attempted_device_ids = (*attempted_device_ids, result.device_id)
        return replace(
            pending,
            target_path=result.target_path or pending.target_path,
            candidate_sha256=result.candidate_sha256 or pending.candidate_sha256,
            execution_id=result.execution_id or pending.execution_id,
            run_id=result.run_id or pending.run_id,
            receipt_path=result.receipt_path or pending.receipt_path,
            device_id=result.device_id or pending.device_id,
            state=result.state or pending.state,
            error_message=result.error_message or pending.error_message,
            failure_kind=result.failure_kind or pending.failure_kind,
            execution_terminal=result.execution_terminal,
            retry_used=retry_used,
            blocked_env_category=result.blocked_env_category,
            device_rejected=result.device_rejected,
            rejected_device_ids=rejected_device_ids,
            blocked_env_round_attempted_device_ids=attempted_device_ids,
            attempts=attempts,
        )

    def _resolve_pending_l4(
        self,
        pending: PendingL4,
        *,
        incumbent_source: str,
    ) -> tuple[L4CannbenchResult, PendingL4 | None]:
        """Continue the saved L4 execution, or move the complete run to another card.

        A live execution has no run id and a terminal one with a run id can have
        its signed receipt re-read; neither path submits again. A bounded wait
        that expired is not a failure: the same execution ID is polled again on
        the original card. A terminal BLOCKED_ENV preemption continues on the
        next untried card of the saved pool before any bounded round backoff.
        Any other proven-terminal infrastructure failure may start one complete
        L4 run on the explicitly authorized alternate card, and its identity is
        persisted before the wait. Numeric/official-validation failures are
        verdicts and never retry; timings are never stitched.
        """
        expected = pending.candidate_sha256 or self._source_sha256(incumbent_source)
        digest_error = ""
        try:
            current_digest = hashlib.sha256(
                self._target_module_path.read_bytes()
            ).hexdigest()
        except OSError as exc:
            digest_error = (
                f"pending L4 candidate {self._target_module_path} cannot be read: {exc}"
            )
        else:
            if current_digest != expected:
                digest_error = (
                    f"pending L4 candidate {self._target_module_path} digest "
                    f"{current_digest} does not match submitted {expected}; "
                    "refusing to resume or replace it"
                )
        if digest_error:
            failed = L4CannbenchResult(
                passed=False,
                error_message=digest_error,
                failure_kind=FAILURE_KIND_INFRASTRUCTURE,
                execution_id=pending.execution_id,
                run_id=pending.run_id,
                receipt_path=pending.receipt_path,
                device_id=pending.device_id,
                state=pending.state,
                execution_terminal=pending.execution_terminal,
                target_path=pending.target_path,
                candidate_sha256=expected,
                attempts=pending.attempts,
            )
            updated = replace(
                pending,
                candidate_sha256=expected,
                error_message=digest_error,
            )
            self._pending_l4 = updated
            return failed, updated

        attempts = list(pending.attempts)
        effective_pool = self._effective_l4_pool(pending)
        effective_limit = self._effective_l4_round_limit(pending)
        current = replace(
            pending,
            device_pool=effective_pool,
            blocked_env_round_limit=effective_limit,
        )

        def _record_started(execution_id: str, device: str) -> None:
            nonlocal current
            current = replace(
                current,
                execution_id=execution_id or current.execution_id,
                device_id=device or current.device_id,
                state="SUBMITTED",
                execution_terminal=False,
            )
            self._pending_l4 = current
            self._persist_runtime_state("pending_l4", current.to_dict())

        def _keep(
            result: L4CannbenchResult,
            *,
            retry_used: bool,
        ) -> tuple[L4CannbenchResult, PendingL4 | None]:
            nonlocal current
            updated = self._pending_l4_after(
                current,
                result,
                tuple(attempts),
                retry_used=retry_used,
            )
            self._pending_l4 = updated
            self._persist_runtime_state("pending_l4", updated.to_dict())
            return self._merge_l4_identity(result, updated), updated

        def _finish(
            result: L4CannbenchResult,
        ) -> tuple[L4CannbenchResult, PendingL4 | None]:
            self._pending_l4 = None
            self._persist_runtime_state("pending_l4", None)
            return self._merge_l4_identity(result, current), None

        def _retry_whole_l4() -> tuple[L4CannbenchResult, PendingL4 | None]:
            nonlocal current
            retry_device = self._retry_device_id
            assert retry_device
            current = replace(
                current,
                execution_id="",
                run_id="",
                receipt_path="",
                device_id=retry_device,
                state="SUBMITTED",
                error_message="",
                failure_kind=FAILURE_KIND_INFRASTRUCTURE,
                execution_terminal=False,
                retry_used=True,
                blocked_env_category="",
                device_rejected=False,
            )
            retry_result = self._run_l4_cannbench(
                self._target_module_path,
                device_id=retry_device,
                on_execution_started=_record_started,
            )
            attempts.append(self._l4_attempt(retry_result))
            current = replace(
                current,
                execution_id=retry_result.execution_id or current.execution_id,
                run_id=retry_result.run_id or current.run_id,
                receipt_path=retry_result.receipt_path or current.receipt_path,
                device_id=retry_result.device_id or current.device_id,
                state=retry_result.state or current.state,
                error_message=retry_result.error_message or current.error_message,
                failure_kind=retry_result.failure_kind or current.failure_kind,
                execution_terminal=retry_result.execution_terminal,
                attempts=tuple(attempts),
            )
            if retry_result.passed:
                return _finish(retry_result)
            if retry_result.failure_kind == FAILURE_KIND_CANDIDATE:
                return _finish(retry_result)
            return _keep(retry_result, retry_used=True)

        def _after_infrastructure(
            result: L4CannbenchResult,
        ) -> tuple[L4CannbenchResult, PendingL4 | None]:
            nonlocal current
            if (
                _is_occupancy_blocked_env(
                    result.failure_kind,
                    result.blocked_env_category,
                )
                or self._is_rejected_card_l4(result)
            ):
                result, current = self._advance_l4_card_failure(
                    current,
                    result,
                    attempts,
                )
            if result.passed or result.failure_kind == FAILURE_KIND_CANDIDATE:
                return _finish(result)
            if not result.execution_terminal:
                return _keep(result, retry_used=current.retry_used)
            if self._can_retry_l4_on_alt_card(current, result.device_id):
                return _retry_whole_l4()
            return _keep(result, retry_used=current.retry_used)

        if pending.execution_id and (not pending.execution_terminal or pending.run_id):
            result = self._run_l4_cannbench(
                self._target_module_path,
                resume_execution_id=pending.execution_id,
                resume_run_id=pending.run_id,
                device_id=pending.device_id or self._config.device_id,
                on_execution_started=_record_started,
            )
            attempts.append(self._l4_attempt(result))
            current = self._pending_l4_after(
                current,
                result,
                tuple(attempts),
                retry_used=current.retry_used,
            )
            self._pending_l4 = current
            self._persist_runtime_state("pending_l4", current.to_dict())
            return _after_infrastructure(result)

        blocked_env_pending = _is_occupancy_blocked_env(
            pending.failure_kind,
            pending.blocked_env_category,
        )
        rejected_pending = (
            pending.device_rejected
            and pending.failure_kind == FAILURE_KIND_INFRASTRUCTURE
        )
        interrupted_submission = (
            not pending.execution_terminal
            and not pending.execution_id
            and not pending.run_id
            and pending.state == "SUBMITTED"
        )
        if (
            (blocked_env_pending and pending.execution_terminal)
            or rejected_pending
            or (
                interrupted_submission
                and current.blocked_env_rounds_used <= effective_limit
            )
        ):
            if interrupted_submission:
                interrupted_device = (
                    current.device_id
                    if current.device_id in effective_pool
                    else effective_pool[0]
                )
                current = replace(current, device_id=interrupted_device)
                result = self._run_l4_cannbench(
                    self._target_module_path,
                    device_id=interrupted_device,
                    on_execution_started=_record_started,
                )
                attempts.append(self._l4_attempt(result))
                current = self._pending_l4_after(
                    current,
                    result,
                    tuple(attempts),
                    retry_used=current.retry_used,
                )
                self._pending_l4 = current
                self._persist_runtime_state("pending_l4", current.to_dict())
            else:
                result = L4CannbenchResult(
                    passed=False,
                    error_message=pending.error_message,
                    failure_kind=pending.failure_kind,
                    execution_id=pending.execution_id,
                    run_id=pending.run_id,
                    receipt_path=pending.receipt_path,
                    device_id=pending.device_id,
                    state=pending.state,
                    execution_terminal=pending.execution_terminal,
                    blocked_env_category=pending.blocked_env_category,
                    device_rejected=pending.device_rejected,
                    target_path=pending.target_path,
                    candidate_sha256=expected,
                    attempts=pending.attempts,
                )
            return _after_infrastructure(result)

        if self._can_retry_l4_on_alt_card(pending, pending.device_id):
            return _retry_whole_l4()
        self._pending_l4 = pending
        failed = L4CannbenchResult(
            passed=False,
            error_message=(
                pending.error_message
                or "L4 infrastructure failure has no resumable execution and no "
                "other authorized card is configured"
            ),
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
            execution_id=pending.execution_id,
            run_id=pending.run_id,
            receipt_path=pending.receipt_path,
            device_id=pending.device_id,
            state=pending.state,
            execution_terminal=pending.execution_terminal,
            blocked_env_category=pending.blocked_env_category,
            device_rejected=pending.device_rejected,
            target_path=pending.target_path,
            candidate_sha256=expected,
            attempts=pending.attempts,
        )
        return failed, pending

    def _l4_candidate_relative(self) -> str:
        try:
            return self._target_module_path.resolve().relative_to(
                Path(self._source_root).resolve()
            ).as_posix()
        except (OSError, ValueError):
            return self._target_module_path.name

    @staticmethod
    def _is_blocked_env_l4(result: L4CannbenchResult) -> bool:
        """Return True only for an occupancy-classified BLOCKED_ENV receipt."""
        return _is_occupancy_blocked_env(
            result.failure_kind,
            result.blocked_env_category,
        )

    @staticmethod
    def _is_rejected_card_l4(result: L4CannbenchResult) -> bool:
        """Return True when V2 refused the submitted card as unauthorized.

        Submit is the sole authorization authority. A refusal proves only that
        this card is outside the current allowlist, so the pool keeps moving;
        it never widens authorization and never hides a usable card.
        """
        return (
            result.device_rejected
            and result.failure_kind == FAILURE_KIND_INFRASTRUCTURE
        )

    def _run_final_l4_tracked(
        self,
        *,
        incumbent_source: str,
        prior_stop_reason: str,
    ) -> L4CannbenchResult:
        """Submit the official L4 run and persist its identity before waiting.

        The first submission uses the first configured pool card. A terminal
        BLOCKED_ENV preemption advances through the remaining pool cards with a
        complete run on each; a candidate verdict ends the gate, and any other
        infrastructure failure stays pending for the next invocation. No other
        terminal failure is retried here, so a fresh failure is never re-spent
        before its saved identity has been examined.
        """
        candidate_sha = self._source_sha256(incumbent_source)
        candidate_relative = self._l4_candidate_relative()
        pool = self._l4_device_pool
        attempts: list[dict[str, str]] = []
        current = PendingL4(
            target_path=str(self._target_module_path),
            candidate_relative=candidate_relative,
            candidate_sha256=candidate_sha,
            device_id=pool[0],
            state="SUBMITTED",
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
            execution_terminal=False,
            device_pool=pool,
            blocked_env_round_attempted_device_ids=(pool[0],),
            blocked_env_round_limit=max(0, self._config.l4_blocked_env_max_retries),
            prior_stop_reason=prior_stop_reason,
        )

        def _record_started(execution_id: str, device: str) -> None:
            nonlocal current
            current = replace(
                current,
                execution_id=execution_id or current.execution_id,
                device_id=device or current.device_id,
                state="SUBMITTED",
                execution_terminal=False,
            )
            self._pending_l4 = current
            self._persist_runtime_state("pending_l4", current.to_dict())

        result = self._run_l4_cannbench(
            self._target_module_path,
            device_id=pool[0],
            on_execution_started=_record_started,
        )
        attempts.append(self._l4_attempt(result))
        current = self._pending_l4_after(
            current,
            result,
            tuple(attempts),
            retry_used=False,
        )
        if self._is_blocked_env_l4(result) or self._is_rejected_card_l4(result):
            result, current = self._advance_l4_card_failure(current, result, attempts)
        if result.passed or result.failure_kind == FAILURE_KIND_CANDIDATE:
            self._pending_l4 = None
            self._persist_runtime_state("pending_l4", None)
            return self._merge_l4_identity(result, current)
        self._pending_l4 = current
        self._persist_runtime_state("pending_l4", current.to_dict())
        return self._merge_l4_identity(result, current)

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
        """Record per-candidate Track-B blueprint, 5D observation, residual, and provenance."""
        if regime_spec.runtime_backend not in ("ascendc_trackb", "hybrid_trackb"):
            return ()
        if blueprint is None:
            return ()
        policy = live_plan.policy if live_plan is not None else {}
        solver_backend_tier = str(policy.get("backend_tier", ""))
        case_entry = regime_spec.representative_case_entry
        raw_case_id = case_entry.get("case_id", 0)
        representative_case_id = (
            raw_case_id
            if isinstance(raw_case_id, int) and not isinstance(raw_case_id, bool)
            else 0
        )
        raw_dtype = case_entry.get("dtype", "")
        representative_input_dtype: str | list[str]
        if isinstance(raw_dtype, str):
            representative_input_dtype = raw_dtype
        elif isinstance(raw_dtype, (list, tuple)):
            representative_input_dtype = [str(item) for item in raw_dtype]
        else:
            representative_input_dtype = ""
        parent_summary_payload = (
            parent_l1_summary.to_dict() if parent_l1_summary is not None else None
        )
        roofline = blueprint.roofline
        blueprint_summary: dict[str, object] = {
            "entry_name": blueprint.entry_name,
            "ideal_roofline_us": float(roofline.ideal_roofline_us),
            "compute_floor_us": float(roofline.compute_floor_us),
            "memory_floor_us": float(roofline.memory_floor_us),
            "dominant_bound": roofline.dominant_bound,
            "anchor_dims": {str(k): int(v) for k, v in roofline.anchor_dims.items()},
            "requested_anchor_dims": {
                str(k): int(v) for k, v in roofline.requested_anchor_dims.items()
            },
            "resolved_hir_dims": {
                str(k): int(v) for k, v in roofline.resolved_hir_dims.items()
            },
            "shape_exact_match": bool(roofline.shape_exact_match),
            "fallback_reason": roofline.fallback_reason,
        }
        active_unlocked = active_mask.unlocked_facets()
        evidence: list[dict[str, object]] = []
        for chk in checks:
            stage_a = chk.stage_a
            ledger = stage_a.ledger if stage_a is not None else None
            if ledger is None:
                continue
            l2_result = l2_results.get(chk.candidate_id)
            local = (
                l2_result.for_regime(regime_spec.regime_tag)
                if l2_result is not None
                else None
            )
            has_latency = local is not None and bool(local.anchors)
            latency_us = local.mean_latency_us() if has_latency else 0.0
            cand_sha = self._source_sha256(chk.patched_module_source)
            telemetry: dict[str, object] = {
                "dur_us": latency_us,
                "roofline_eta": local.avg_roofline_eta if has_latency else 0.0,
            }
            msprof = self._msprof_cache.get((cand_sha, regime_spec.regime_tag))
            if msprof:
                telemetry.update(msprof)
            else:
                telemetry["measured_latency_us"] = latency_us
                telemetry["is_analytical_proxy"] = True
            compiler_summary = (
                chk.stage_b.compiler_summary if chk.stage_b is not None else None
            )
            observation = ledger.to_observation_5d(
                compiler_summary=compiler_summary,
                telemetry=telemetry,
            )
            residual = observation.compute_axis_residuals()
            unlocked = tuple(
                facet
                for facet in residual.unlock_causal_facets(active_mask).unlocked_facets()
                if facet not in active_unlocked
            )
            route = stage_a.route if stage_a is not None else None
            evidence.append(
                {
                    "round_idx": round_idx,
                    "regime_tag": regime_spec.regime_tag,
                    "candidate_id": chk.candidate_id,
                    "candidate_sha256": cand_sha,
                    "active_kernel": active_kernel,
                    "l2_run_id": (
                        l2_run_ids.get(chk.candidate_id, "")
                        if l2_result is not None
                        else ""
                    ),
                    "parent_candidate_sha256": parent_sha256,
                    "parent_compiler_summary": (
                        dict(parent_summary_payload)
                        if parent_summary_payload is not None
                        else None
                    ),
                    "parent_compiler_summary_source": parent_l1_summary_source,
                    "representative_case_id": representative_case_id,
                    "representative_input_dtype": representative_input_dtype,
                    "runtime_backend": regime_spec.runtime_backend,
                    "backend_tier": "ascendc_template",
                    "solver_backend_tier": solver_backend_tier,
                    "track": (
                        route.track
                        if self._enable_track_router and route is not None
                        else "unconstrained"
                    ),
                    "stage_a_backend_kind": stage_a.backend_kind,
                    "l0_l1_passed": chk.passed,
                    "l1_compiler_passed": bool(chk.stage_b.passed) if chk.stage_b else False,
                    "l2_evaluated": l2_result is not None,
                    "l2_all_passed": bool(l2_result.all_passed) if l2_result is not None else False,
                    "latency_us": float(latency_us),
                    "blueprint": dict(blueprint_summary),
                    "observation": observation.to_dict(),
                    "axis_residual": {
                        "compute_issue_gap": residual.compute_issue_gap,
                        "offchip_dma_gap": residual.offchip_dma_gap,
                        "onchip_sram_reg_gap": residual.onchip_sram_reg_gap,
                        "pipeline_sync_gap": residual.pipeline_sync_gap,
                        "grid_wave_gap": residual.grid_wave_gap,
                    },
                    "dominant_axis": residual.dominant_axis,
                    "unlocked_facets": list(unlocked),
                    "active_unlocked_facets": list(active_unlocked),
                    "field_provenance": {
                        "has_l1_compiler_summary": observation.has_l1_compiler_summary,
                        "has_l2_telemetry": observation.has_l2_telemetry,
                        "measured_l2_fields": list(observation.measured_l2_fields),
                        "latency_source": "l2_batch_mean" if has_latency else "none",
                        "observation_source": (
                            "msprof_l2"
                            if msprof
                            else ("l2_batch" if has_latency else "roofline_static")
                        ),
                        "compiler_summary_source": (
                            "bisheng_l1" if compiler_summary is not None else "absent"
                        ),
                    },
                }
            )
        return tuple(evidence)

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
        """Run one optimisation round and promote the best candidate through L3.

        The execution identity is persisted at submission, before the potentially
        long wait, so a killed process can recover the same run. A terminal
        infrastructure fault restores the incumbent and blocks promotion without
        counting stagnation; a live or queued execution keeps the exact candidate
        bytes because replacing them would silently change the code under test.
        """
        backend = self._get_backend()
        self._active_incumbent_source = incumbent_source
        self._active_regime_tag = regime_spec.regime_tag
        regime_beam = self.beam_for_regime(regime_spec.regime_tag)
        func_name, _default_snippet = extract_regime_snippet(incumbent_source, regime_spec)

        escalated = stagnation_rounds >= self._config.stagnation_escalate_rounds
        creative_effort = "high"
        (
            parent_module_source,
            parent_snippet,
            parent_lane,
            selected_stone,
            selected_stone_hypothesis,
        ) = self._resolve_round_parent(
            incumbent_source=incumbent_source,
            regime_spec=regime_spec,
            regime_beam=regime_beam,
            escalated=escalated,
        )
        stone_ledger = selected_stone.ledger if selected_stone is not None else None
        stone_track = selected_stone.track if selected_stone is not None else None
        selected_stone_id = selected_stone.candidate_id if selected_stone is not None else None

        active_parent_id = (
            selected_stone_id
            if selected_stone_id
            else (
                regime_beam.exploit_beam[0].candidate_id
                if regime_beam.exploit_beam
                else f"incumbent_r{max(0, round_idx - 1)}"
            )
        )

        base_stage_a = self._run_l0(incumbent_source, regime_spec)
        parent_ledger = stone_ledger if stone_ledger is not None else base_stage_a.ledger
        if parent_ledger is None:
            raise RuntimeError(
                f"Stage-A L0 produced no ledger for regime {regime_spec.regime_tag!r}: "
                f"{base_stage_a.error_message}"
            )
        if self._enable_track_router:
            parent_track = (
                stone_track
                if stone_track is not None
                else (
                    base_stage_a.route.track
                    if base_stage_a.route is not None
                    else "unconstrained"
                )
            )
            parent_features = base_stage_a.features
        else:
            parent_track = "unconstrained"
            parent_features = None

        regime_best = best_result.for_regime(regime_spec.regime_tag)
        local_best = regime_best if regime_best.anchors else best_result
        parent_axes = self._derive_topology_axes(
            parent_ledger,
            features=parent_features,
            route=base_stage_a.route,
        )
        rep_dims = {
            str(k): int(v)
            for k, v in (
                regime_spec.representative_anchor_dims or regime_spec.representative_hir_dims
            ).items()
            if isinstance(v, int) and not isinstance(v, bool)
        }
        sil_regime = None
        if self._sil_advisory is not None:
            for reg in getattr(self._sil_advisory, "regimes", ()) or ():
                if getattr(reg, "regime_tag", None) == regime_spec.regime_tag:
                    sil_regime = reg
                    break
        if sil_regime is None:
            sil_regime = RegimeSlice(
                regime_tag=regime_spec.regime_tag,
                case_nums=(),
                representative_dims=rep_dims,
                has_unaligned_tail=(
                    "unaligned" in regime_spec.regime_tag or "tail" in regime_spec.regime_tag
                ),
            )
        adv_dtypes = tuple(getattr(self._sil_advisory, "input_dtypes", ()) or ())
        adv_nan_inf = bool(getattr(self._sil_advisory, "has_nan_inf_cases", False))
        live_plan = recompute_regime_for_live_buffers(
            self._regime_solver,
            sil_regime,
            parent_ledger,
            op_name=self._config.operator_name,
            topology_axes=parent_axes,
            input_dtypes=adv_dtypes or ("float16",),
            has_nan_inf_cases=adv_nan_inf,
            has_indirect_gather=(parent_axes.iteration_regularity == "irregular_simt"),
            has_recurrence=(
                parent_axes.memory_ring in ("persistent_onchip_state", "bidirectional_cv_ring")
            ),
            uses_cube=parent_axes.uses_cube,
        )
        self._live_regime_plans[regime_spec.regime_tag] = live_plan
        if stagnation_rounds >= 1:
            self._collect_msprof_for_regimes(
                parent_module_source,
                best_result,
                active_anchors,
                [regime_spec],
            )
        parent_sha = self._source_sha256(parent_module_source)
        cached_parent_msprof = self._msprof_cache.get((parent_sha, regime_spec.regime_tag))
        parent_lat_us = (
            selected_stone.latency_us
            if (selected_stone is not None and selected_stone.latency_us > 0.0)
            else local_best.mean_latency_us()
        )
        parent_obj_score = (
            selected_stone.objective_score
            if (selected_stone is not None and selected_stone.objective_score is not None)
            else local_best.surrogate_score
        )
        parent_l1_summary = (
            compiler_summary_from_residual(selected_stone.residual)
            if selected_stone is not None
            else self._l1_summary_for_first_round(parent_module_source, regime_spec)
        )
        parent_l1_summary_source = (
            "stepping_stone_residual"
            if selected_stone is not None and parent_l1_summary is not None
            else ("bisheng_l1" if parent_l1_summary is not None else "absent")
        )
        if cached_parent_msprof:
            telemetry_payload: dict[str, object] = {
                "dur_us": parent_lat_us,
                "roofline_eta": local_best.avg_roofline_eta,
            }
            for k, v in cached_parent_msprof.items():
                if k == "dur_us" and float(v) <= 0.0:
                    continue
                telemetry_payload[k] = v
            parent_residual = compute_static_dynamic_residual(
                parent_ledger,
                telemetry_payload,
                compiler_summary=parent_l1_summary,
            )
        elif selected_stone is not None and (
            selected_stone.residual.measured_aiv_vec_ratio > 0.0
            or selected_stone.residual.measured_aic_mac_ratio > 0.0
            or selected_stone.residual.measured_mte2_wait_ratio > 0.0
            or selected_stone.residual.measured_sync_stall_ratio > 0.0
            or selected_stone.residual.spill_bytes > 0
            or (not selected_stone.residual.vloopv2_hit)
            or selected_stone.residual.vf_fusion_breaks > 0
            or selected_stone.residual.alias_serialized_buffers > 0
            or selected_stone.residual.dual_issue_ipc_collapse > 0.0
            or selected_stone.residual.dcache_dcci_flushes > 0
            or bool(selected_stone.residual.has_local_membar_in_vf_call)
        ):
            parent_residual = selected_stone.residual
        else:
            telemetry_payload = {
                "dur_us": parent_lat_us,
                "roofline_eta": local_best.avg_roofline_eta,
            }
            parent_residual = compute_static_dynamic_residual(
                parent_ledger,
                telemetry_payload,
                compiler_summary=parent_l1_summary,
            )
        ctx = build_predicate_context(
            parent_ledger,
            features=parent_features,
            residual=parent_residual,
            extra={
                "topology_axes": parent_axes,
                **_case_ratio_predicate_entries(regime_spec.representative_case_ratios),
            },
        )
        parent_comp_summary = compiler_summary_from_residual(parent_residual)
        parent_waterlines = getattr(parent_ledger, "waterlines", {}) or {}
        l1_wl = parent_waterlines.get("L1")
        ub_wl = parent_waterlines.get("UB")
        l1_overflow = bool(l1_wl is not None and int(getattr(l1_wl, "overflow_bytes", 0) or 0) > 0)
        ub_overflow = bool(ub_wl is not None and int(getattr(ub_wl, "overflow_bytes", 0) or 0) > 0)
        blueprint = self._regime_blueprints.get(regime_spec.regime_tag)
        parent_workload = self._build_cost_workload(
            parent_ledger,
            live_plan=live_plan,
            axes=parent_axes,
            blueprint=blueprint,
        )
        if self._experience_lake_dir is not None:
            try:
                self._tracker.sync_from_experience_lake(
                    self._experience_lake_dir,
                    exclude_run_id=self._run_id,
                )
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug("ExperienceLake per-round KB sync skipped: %s", exc)
            try:
                self._cost_model.replay_experience_lake(self._experience_lake_dir)
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug("ExperienceLake per-round cost replay skipped: %s", exc)
        parent_cost_pred = self._cost_model.predict(
            parent_workload,
            bisheng_res=parent_comp_summary,
        )
        active_mask = self._resolve_trackb_freedom_mask(regime_spec)
        escalation_decision = None
        pipe_gap_report = None
        escalated_trackb_tu = ""
        if escalated:
            active_mask = relax_facets_for_bottleneck(
                active_mask,
                dominant_bound=parent_cost_pred.dominant_bound,
                residual=parent_residual,
                compiler_summary=parent_comp_summary,
                l1_overflow=l1_overflow,
                ub_overflow=ub_overflow,
            )
            self._freedom_masks[regime_spec.regime_tag] = active_mask
            if normalize_runtime_backend(regime_spec.runtime_backend) in (
                "ascendc_trackb",
                "hybrid_trackb",
            ):
                current_tier = "ascendc_template"
            else:
                current_tier = normalize_runtime_backend(
                    str(live_plan.policy.get("backend_tier", "tilelang_dsl"))
                )
                if current_tier not in ("tilelang_dsl", "ascendc_trackb", "hybrid_trackb"):
                    current_tier = "tilelang_dsl"
                elif current_tier in ("ascendc_trackb", "hybrid_trackb"):
                    current_tier = "ascendc_template"
            sf_ctx = (
                parent_features.to_dict()
                if hasattr(parent_features, "to_dict")
                else (dict(parent_features) if isinstance(parent_features, Mapping) else {})
            )
            sf_ctx["l1_overflow"] = l1_overflow
            sf_ctx["ub_overflow"] = ub_overflow
            sf_ctx["freedom_mask"] = active_mask
            sf_ctx.setdefault("uses_cube", parent_axes.uses_cube)
            case_ratios = regime_spec.representative_case_ratios
            if case_ratios is not None:
                sf_ctx["representative_case_id"] = case_ratios.case_id
                sf_ctx["phi_7"] = case_ratios.inner_contiguous_bytes
                sf_ctx["inner_contiguous_bytes"] = case_ratios.inner_contiguous_bytes
                sf_ctx["phi_8"] = case_ratios.wave_occupancy_ratio
                sf_ctx["persistent_wave_occupancy"] = case_ratios.wave_occupancy_ratio
            escalation_decision = evaluate_track_escalation(
                current_tier=current_tier,
                roofline_eta=float(local_best.avg_roofline_eta),
                target_eta=float(self._perf_target.target_roofline_eta),
                stagnation_rounds=int(stagnation_rounds),
                dominant_bound=parent_cost_pred.dominant_bound,
                compiler_summary=parent_comp_summary,
                structural_features=sf_ctx,
            )
            pipe_gap_report = scan_tilelang_pipe_gaps()
            if (
                escalation_decision is not None
                and bool(getattr(escalation_decision, "should_escalate", False))
                and str(getattr(escalation_decision, "to_tier", "")) == "ascendc_template"
                and _is_real_tir_primfunc(base_stage_a.prim_func)
            ):
                try:
                    raw_tu = escalate_primfunc_to_trackb_tu(base_stage_a.prim_func)
                    if len(raw_tu.encode("utf-8")) > 8192:
                        escalated_trackb_tu = build_trackb_context_view(
                            raw_tu,
                            regime_tag=regime_spec.regime_tag,
                            module_source=parent_module_source,
                            max_bytes=8192,
                            active_kernel=_active_trackb_kernel(
                                raw_tu,
                                raw_tu,
                                regime_tag=regime_spec.regime_tag,
                            ),
                        )
                    else:
                        escalated_trackb_tu = raw_tu
                except (ImportError, OSError, RuntimeError, ValueError) as exc:
                    logger.warning("Track-B PrimFunc escalation lowering skipped: %s", exc)
        else:
            self._freedom_masks.setdefault(regime_spec.regime_tag, active_mask)

        sil_policy_section = self._format_sil_live_policy_section(
            live_plan=live_plan,
            active_mask=active_mask,
            escalation=escalation_decision,
            pipe_gaps=pipe_gap_report,
            cost_pred=parent_cost_pred,
            escalated_trackb_tu=escalated_trackb_tu,
        )
        ranked_rules = list(
            self._tracker.rank_applicable_rules(
                ctx,
                top_k=self._config.k_rule_workers,
                topology_axes=parent_axes,
                operator_name=self._config.operator_name,
                runtime_backend=regime_spec.runtime_backend,
            )
        )

        worker_specs: list[tuple[str, str, AscendCausalRule | None, GenerationRequest]] = []
        for idx, rule in enumerate(ranked_rules[: self._config.k_rule_workers]):
            worker_specs.append(
                (
                    f"r{round_idx}_rule_{idx}",
                    "rule_exploit",
                    rule,
                    self._build_exploit_regime_request(
                        snippet_source=parent_snippet,
                        func_name=func_name,
                        regime_spec=regime_spec,
                        rule=rule,
                        best_result=local_best,
                        blueprint=blueprint,
                        parent_ledger=parent_ledger,
                        parent_residual=parent_residual,
                        track=parent_track,
                        incumbent_source=parent_module_source,
                        sil_policy_section=sil_policy_section,
                        effort=self._config.exploit_effort,
                    ),
                )
            )
        excluded = summarize_excluded_mechanisms(
            ranked_rules,
            regime_beam.exploited_signatures,
        )
        if self._config.enable_tilesize_prescreen:
            anchor_dims = (
                regime_spec.representative_anchor_dims or regime_spec.representative_hir_dims
            )
            prescreen_kwargs: dict[str, object] = dict(regime_spec.sample_builder_kwargs)
            prescreen_factory = func_name
            prescreen_pc: dict[str, object] = {}
            prescreen_cf: tuple[str, ...] = ()
            try:
                resolved_cfg = resolve_regime_config(
                    parent_module_source,
                    regime_spec,
                    dims=anchor_dims,
                )
                prescreen_kwargs.update(resolved_cfg.builder_kwargs)
                prescreen_factory = resolved_cfg.builder_name
                prescreen_pc = dict(resolved_cfg.pass_configs)
                prescreen_cf = resolved_cfg.compile_flags
            except (
                SyntaxError,
                AttributeError,
                ImportError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                logger.warning(
                    "Failed to resolve prescreen config for regime %s: %s",
                    regime_spec.regime_tag,
                    exc,
                )
            var_bindings = build_anchor_var_bindings(anchor_dims, kwargs=prescreen_kwargs)
            prescreen_summary = build_prescreen_summary_from_kwargs(
                prescreen_kwargs,
                dsl_source=parent_module_source,
                factory_name=prescreen_factory,
                var_bindings=var_bindings or None,
                config_selector_name=regime_spec.config_selector_fn or None,
                dims=anchor_dims,
                pass_configs=prescreen_pc or None,
                compile_flags=prescreen_cf or None,
                cost_model=self._cost_model,
                axes=parent_axes,
                live_plan=live_plan,
                blueprint=blueprint,
                sil_advisory=self._sil_advisory,
            )
        else:
            prescreen_summary = ""
        deficit = max(0, self._config.k_rule_workers - len(ranked_rules))
        total_creative = 1 + deficit
        for c_idx in range(total_creative):
            wid = f"r{round_idx}_creative" if c_idx == 0 else f"r{round_idx}_creative_{c_idx}"
            hint = (
                CREATIVE_FOCUS_HINTS[c_idx % len(CREATIVE_FOCUS_HINTS)]
                if total_creative > 1
                else ""
            )
            worker_specs.append(
                (
                    wid,
                    "creative_explore",
                    None,
                    self._build_creative_regime_request(
                        snippet_source=parent_snippet,
                        func_name=func_name,
                        regime_spec=regime_spec,
                        best_result=local_best,
                        effort=creative_effort,
                        excluded_mechanisms=excluded,
                        focus_hint=hint,
                        prescreen_summary=prescreen_summary,
                        blueprint=blueprint,
                        parent_ledger=parent_ledger,
                        parent_residual=parent_residual,
                        track=parent_track,
                        incumbent_source=parent_module_source,
                        parent_hypothesis=selected_stone_hypothesis,
                        sil_policy_section=sil_policy_section,
                    ),
                )
            )

        async def _call_worker(
            wid: str,
            wkind: str,
            rule: AscendCausalRule | None,
            req: GenerationRequest,
        ) -> tuple[str, str, AscendCausalRule | None, str | None, str, str, bool]:
            try:
                raw = await backend.generate(req)
            except (OSError, subprocess.SubprocessError, TimeoutError, RuntimeError, ConnectionError) as exc:
                return wid, wkind, rule, None, "", f"llm_transport:{type(exc).__name__}:{exc}", True
            try:
                if wkind == "creative_explore":
                    creative = extract_creative_payload(raw)
                    snippet = creative["dsl_source"]
                    raw_hyp = str(creative.get("physical_hypothesis", "")).strip()
                    raw_metric = str(creative.get("target_local_metric", "")).strip()
                    if raw_hyp and raw_metric:
                        hypothesis_text = f"{raw_hyp} [target_metric={raw_metric}]"
                    else:
                        hypothesis_text = raw_hyp or raw_metric
                else:
                    snippet = extract_python_dsl(raw)
                    hypothesis_text = ""
                return wid, wkind, rule, snippet, hypothesis_text, "", False
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                return wid, wkind, rule, None, "", f"llm_parse:{type(exc).__name__}:{exc}", False

        worker_replies = await asyncio.gather(
            *(_call_worker(wid, wkind, rule, req) for wid, wkind, rule, req in worker_specs)
        )

        raw_candidates: list[tuple[str, str]] = []
        failed_entries: list[tuple[str, str, str]] = []
        meta_by_id: dict[str, tuple[str, AscendCausalRule | None, str]] = {}
        for wid, wkind, rule, snippet, cand_hypothesis, err, is_transport_error in worker_replies:
            meta_by_id[wid] = (wkind, rule, cand_hypothesis)
            if snippet is None:
                stage_tag = "llm_transport" if is_transport_error else "llm_generate"
                failed_entries.append((wid, stage_tag, err[:1500]))
                if rule is not None and not is_transport_error:
                    self._tracker.record_outcome(
                        rule.rule_id,
                        postcondition_realized=False,
                        topology_axes=parent_axes,
                    )
                    self._record_kb_event(
                        rule_id=rule.rule_id,
                        postcondition_realized=False,
                        topology_axes=parent_axes,
                        round_idx=round_idx,
                    )
            else:
                raw_candidates.append((wid, snippet))

        l0_l1_checks: list[L0L1CandidateCheck] = []
        round_active_kernel = ""
        if raw_candidates:
            prescreen_top_k = (
                int(self._config.cost_prescreen_top_k)
                if (
                    self._config.enable_tilesize_prescreen
                    and self._config.cost_prescreen_top_k is not None
                    and int(self._config.cost_prescreen_top_k) > 0
                )
                else None
            )
            stage_a_ranker = (
                (
                    lambda passers: [
                        (item.candidate_id, item.pred_us)
                        for item in rank_stage_a_candidates_with_cost_model(
                            passers,
                            cost_model=self._cost_model,
                            live_plan=live_plan,
                            axes=parent_axes,
                            blueprint=blueprint,
                            sil_advisory=self._sil_advisory,
                            parent_ledger=parent_ledger,
                        )
                    ]
                )
                if prescreen_top_k is not None
                else None
            )
            def _candidate_l0_check(mod_src: str) -> StageAResult:
                dead_err = _detect_trackb_dead_code_edit(
                    parent_module_source, mod_src, regime_spec
                )
                if dead_err is not None:
                    return StageAResult(
                        passed=False,
                        prim_func=None,
                        ledger=None,
                        error_message=dead_err,
                        backend_kind="ascendc",
                    )
                return self._run_l0(mod_src, regime_spec)

            round_active_kernel = (
                self._resolve_active_trackb_kernel(
                    parent_module_source,
                    regime_spec,
                )
                or ""
            )
            l0_l1_checks = await run_l0_l1_round_filter(
                raw_candidates,
                patch_fn=lambda snip: patch_regime_into_module(
                    parent_module_source,
                    regime_spec,
                    snip,
                    operator_entrypoint=_resolve_operator_entrypoint_safe(
                        self._config.operator_name
                    ),
                    sibling_specs=self._regime_specs,
                    active_kernel=round_active_kernel,
                ),
                l0_checker=_candidate_l0_check,
                l1_compiler=lambda mod_src: self._run_l1(mod_src, regime_spec),
                repair_backend=backend if self._config.max_syntax_repairs > 0 else None,
                max_syntax_repairs=self._config.max_syntax_repairs,
                slice_hint=self._slice_contract_instruction(func_name, regime_spec),
                backend_kind=regime_spec.runtime_backend,
                stage_a_cost_ranker=stage_a_ranker,
                cost_prescreen_top_k=prescreen_top_k,
            )

        passed_checks = [chk for chk in l0_l1_checks if chk.passed]
        for chk in l0_l1_checks:
            self._persist_round_candidate_artifact(chk, round_idx=round_idx)
        for chk in l0_l1_checks:
            if not chk.passed:
                failed_entries.append(
                    (chk.candidate_id, chk.failed_tier or "l0_l1", chk.error_message[:1500])
                )
                _wkind, rule, _hyp = meta_by_id[chk.candidate_id]
                if (
                    rule is not None
                    and chk.failed_tier not in ("llm_transport", "cost_prescreen")
                    and "[repair_timeout]" not in chk.error_message
                ):
                    self._tracker.record_outcome(
                        rule.rule_id,
                        postcondition_realized=False,
                        topology_axes=parent_axes,
                    )
                    self._record_kb_event(
                        rule_id=rule.rule_id,
                        postcondition_realized=False,
                        topology_axes=parent_axes,
                        round_idx=round_idx,
                    )

        l2_results: dict[str, CandidateSurrogateResult] = {}
        l2_run_ids: dict[str, str] = {}
        if passed_checks:
            with tempfile.TemporaryDirectory(prefix=f"tf_camp_r{round_idx}_") as tmp_dir:
                cand_paths: list[tuple[str, Path]] = []
                for chk in passed_checks:
                    cand_file = Path(tmp_dir) / f"{chk.candidate_id}.py"
                    cand_file.write_text(chk.patched_module_source, encoding="utf-8")
                    cand_paths.append((chk.candidate_id, cand_file))
                try:
                    l2_results, l2_run_ids = self._run_l2_batch(cand_paths, active_anchors)
                except (OSError, subprocess.SubprocessError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
                    logger.warning(
                        "Round %d L2 batch failed after retries: %s",
                        round_idx,
                        exc,
                    )
                    l2_results = {}
                    l2_run_ids = {}
                    for chk in passed_checks:
                        failed_entries.append((chk.candidate_id, "l2_error", str(exc)[:1500]))

        passing_candidates: list[tuple[L0L1CandidateCheck, CandidateSurrogateResult]] = [
            (chk, l2_results[chk.candidate_id])
            for chk in passed_checks
            if chk.candidate_id in l2_results and l2_results[chk.candidate_id].all_passed
        ]

        best_round_speedup = best_result.avg_speedup
        best_round_surrogate_score = best_result.surrogate_score
        promoted = False
        rolled_back_l3 = False
        promotion_blocked = False
        new_incumbent = incumbent_source
        new_best = best_result
        round_admitted: list[str] = []
        top_chk: L0L1CandidateCheck | None = None
        top_res: CandidateSurrogateResult | None = None
        rel_gain = 0.0
        local_rel_gain = 0.0
        stagnation_after = stagnation_rounds + 1
        promotion_attempts: tuple[dict[str, str], ...] = ()

        if passing_candidates:
            top_chk, top_res = max(
                passing_candidates,
                key=lambda item: _surrogate_rank_key(item[1]),
            )
            best_round_speedup = top_res.avg_speedup
            best_round_surrogate_score = top_res.surrogate_score
            surrogate_gain = top_res.surrogate_score - best_result.surrogate_score
            rel_gain = (
                (top_res.avg_speedup - best_result.avg_speedup)
                / max(best_result.avg_speedup, 1e-6)
            )
            tail_guard_ok = (
                top_res.min_case_speedup >= 0.97 * best_result.min_case_speedup
                and top_res.surrogate_score >= best_result.surrogate_score - 1e-6
            )
            global_gain_ok = (
                surrogate_gain >= self._config.min_surrogate_gain
                or (
                    top_res.min_case_speedup >= 1.03 * best_result.min_case_speedup
                    and top_res.surrogate_score >= best_result.surrogate_score - 1e-6
                )
            )
            top_local = top_res.for_regime(regime_spec.regime_tag)
            best_local = best_result.for_regime(regime_spec.regime_tag)
            local_gain_ok = bool(
                top_local.anchors
                and best_local.anchors
                and (
                    (top_local.surrogate_score - best_local.surrogate_score)
                    >= self._config.min_surrogate_gain
                    or (
                        top_local.avg_speedup >= 1.02 * best_local.avg_speedup
                        and top_res.surrogate_score > best_result.surrogate_score + 1e-6
                    )
                )
            )
            if top_local.anchors and best_local.anchors:
                top_local_lat = top_local.mean_latency_us()
                best_local_lat = best_local.mean_latency_us()
                local_rel_gain = max(
                    (top_local.avg_speedup - best_local.avg_speedup)
                    / max(best_local.avg_speedup, 1e-6),
                    (
                        (best_local_lat - top_local_lat) / max(best_local_lat, 1e-6)
                        if (top_local_lat > 0.0 and best_local_lat > 0.0)
                        else 0.0
                    ),
                )
            else:
                top_local_lat = 0.0
            dual_criterion_ok = global_gain_ok or local_gain_ok
            if (tail_guard_ok and dual_criterion_ok) or (
                top_local_lat > 0.0 and top_local_lat < 0.98 * local_best.mean_latency_us()
            ):
                self._collect_msprof_for_regimes(
                    top_chk.patched_module_source,
                    top_res,
                    active_anchors,
                    [regime_spec],
                )
            if tail_guard_ok and dual_criterion_ok:
                _atomic_write_text(self._target_module_path, top_chk.patched_module_source)
                candidate_sha = self._source_sha256(top_chk.patched_module_source)

                def _record_l3_started(execution_id: str, device: str) -> None:
                    provisional = PendingPromotion(
                        candidate_source=top_chk.patched_module_source,
                        candidate_sha256=candidate_sha,
                        target_path=str(self._target_module_path),
                        round_idx=round_idx,
                        regime_tag=regime_spec.regime_tag,
                        candidate_result=top_res,
                        candidate_id=top_chk.candidate_id,
                        builder_name=func_name,
                        parent_lane=parent_lane,
                        creative_effort=creative_effort,
                        execution_id=execution_id,
                        device_id=device,
                        execution_terminal=False,
                    )
                    self._active_promotion = provisional
                    self._pending_promotion = provisional
                    self._persist_runtime_state(
                        "active_promotion", provisional.to_dict()
                    )

                l3_res = self._run_l3_contract(
                    self._target_module_path,
                    on_execution_started=_record_l3_started,
                )
                self._active_promotion = None
                if l3_res.passed:
                    promoted = True
                    new_incumbent = top_chk.patched_module_source
                    new_best = top_res
                    stagnation_after = 0
                    self._pending_promotion = None
                    self._persist_runtime_state("active_promotion", None)
                    self._persist_runtime_state("pending_promotion", None)
                elif (
                    l3_res.failure_kind != FAILURE_KIND_CANDIDATE
                    or not l3_res.execution_terminal
                ):
                    if l3_res.execution_terminal:
                        restore_error = self._write_source_and_verify(
                            self._target_module_path, incumbent_source
                        )
                        if restore_error:
                            raise RuntimeError(restore_error)
                        rolled_back_l3 = True
                    else:
                        rolled_back_l3 = False
                    promotion_blocked = True
                    stagnation_after = stagnation_rounds
                    attempt = {
                        "execution_id": l3_res.execution_id,
                        "run_id": l3_res.run_id,
                        "receipt_path": l3_res.receipt_path,
                        "error_message": l3_res.error_message,
                        "failure_kind": l3_res.failure_kind,
                        "failure_category": l3_res.failure_category,
                        "pending": str(
                            l3_res.pending
                            or l3_res.failure_kind != FAILURE_KIND_CANDIDATE
                            or not l3_res.execution_terminal
                        ).lower(),
                        "device_id": l3_res.device_id or self._config.device_id,
                        "state": l3_res.state,
                        "candidate_sha256": l3_res.candidate_sha256 or candidate_sha,
                        "execution_terminal": str(l3_res.execution_terminal).lower(),
                    }
                    self._pending_promotion = PendingPromotion(
                        candidate_source=top_chk.patched_module_source,
                        candidate_sha256=l3_res.candidate_sha256 or candidate_sha,
                        target_path=l3_res.target_path or str(self._target_module_path),
                        round_idx=round_idx,
                        regime_tag=regime_spec.regime_tag,
                        candidate_result=top_res,
                        candidate_id=top_chk.candidate_id,
                        builder_name=func_name,
                        parent_lane=parent_lane,
                        creative_effort=creative_effort,
                        execution_id=l3_res.execution_id,
                        run_id=l3_res.run_id,
                        receipt_path=l3_res.receipt_path,
                        error_message=l3_res.error_message,
                        attempts=(attempt,),
                        device_id=l3_res.device_id or self._config.device_id,
                        execution_terminal=l3_res.execution_terminal,
                    )
                    promotion_attempts = self._pending_promotion.attempts
                    self._persist_runtime_state(
                        "pending_promotion", self._pending_promotion.to_dict()
                    )
                    if l3_res.execution_terminal:
                        self._persist_runtime_state("active_promotion", None)
                    else:
                        self._active_promotion = self._pending_promotion
                        self._persist_runtime_state(
                            "active_promotion", self._pending_promotion.to_dict()
                        )
                else:
                    promotion_attempts = (
                        self._promotion_attempt(l3_res, self._config.device_id),
                    )
                    restore_error = self._write_source_and_verify(
                        self._target_module_path, incumbent_source
                    )
                    if restore_error:
                        raise RuntimeError(restore_error)
                    rolled_back_l3 = True
                    stagnation_after = stagnation_rounds + 1
                    self._pending_promotion = None
                    self._persist_runtime_state("active_promotion", None)
                    self._persist_runtime_state("pending_promotion", None)

        beam_updates: list[BeamCandidate] = []
        top_chk_id = top_chk.candidate_id if top_chk is not None else None
        for chk in passed_checks:
            res = l2_results.get(chk.candidate_id)
            if res is None:
                continue
            wkind, rule, cand_hypothesis = meta_by_id[chk.candidate_id]
            ledger = chk.stage_a.ledger if chk.stage_a else None
            if ledger is None:
                continue
            track = (
                chk.stage_a.route.track
                if self._enable_track_router and chk.stage_a and chk.stage_a.route is not None
                else "unconstrained"
            )
            cand_local_res = res.for_regime(regime_spec.regime_tag)
            local_res = cand_local_res if cand_local_res.anchors else res
            lat_us = local_res.mean_latency_us()
            obj_score = (
                cand_local_res.surrogate_score
                if cand_local_res.anchors
                else res.surrogate_score
            )
            cand_sha = self._source_sha256(chk.patched_module_source)
            cached_cand_msprof = self._msprof_cache.get((cand_sha, regime_spec.regime_tag))
            cand_tel: dict[str, object] = {
                "dur_us": lat_us,
                "measured_latency_us": lat_us,
                "roofline_eta": local_res.avg_roofline_eta,
            }
            if cached_cand_msprof:
                for k, v in cached_cand_msprof.items():
                    if k == "dur_us" and float(v) <= 0.0:
                        continue
                    cand_tel[k] = v
            cand_comp_summary = (
                chk.stage_b.compiler_summary if getattr(chk, "stage_b", None) is not None else None
            )
            cand_stage_b_prof = (
                getattr(chk.stage_b, "microarch_profile", None)
                if getattr(chk, "stage_b", None) is not None
                else None
            )
            cand_residual = compute_static_dynamic_residual(
                ledger, cand_tel, compiler_summary=cand_comp_summary
            )
            cand_axes = self._derive_topology_axes(
                ledger,
                features=chk.stage_a.features if chk.stage_a else None,
                route=chk.stage_a.route if chk.stage_a else None,
            )
            self._record_surrogate_observations(
                source=chk.patched_module_source,
                result=res,
                active_anchors=active_anchors,
                parent_source=parent_module_source,
                parent_result=(best_result if parent_module_source == incumbent_source else None),
                parent_ledger=parent_ledger,
                parent_bisheng_res=parent_comp_summary,
                parent_axes=parent_axes,
                ledger=ledger,
                bisheng_summary=cand_comp_summary,
                stage_b_profile=cand_stage_b_prof,
                axes=cand_axes,
                policy=getattr(live_plan, "policy", None),
                worker_kind=wkind,
                rule_id=rule.rule_id if rule else None,
                backend_track=track,
                regime_tag=regime_spec.regime_tag,
            )
            try:
                _, canonical_slice = extract_regime_snippet(chk.patched_module_source, regime_spec)
            except (SyntaxError, ValueError):
                canonical_slice = chk.snippet_source

            beam_updates.append(
                BeamCandidate(
                    candidate_id=chk.candidate_id,
                    round_idx=round_idx,
                    worker_kind=wkind,
                    rule_id=rule.rule_id if rule else None,
                    dsl_source=canonical_slice,
                    track=track,
                    ledger=ledger,
                    residual=cand_residual,
                    latency_us=lat_us,
                    accuracy_passed=local_res.all_passed,
                    structural_signature=extract_structural_signature(
                        canonical_slice,
                        track=track,
                        residual=cand_residual,
                        stage_b=getattr(chk, "stage_b", None),
                    ),
                    hypothesis=cand_hypothesis,
                    parent_id=active_parent_id,
                    features=(
                        chk.stage_a.features
                        if (self._enable_track_router and chk.stage_a)
                        else None
                    ),
                    objective_score=obj_score,
                )
            )
            if rule is not None:
                cand_rolled_back = bool(
                    rolled_back_l3 and not promotion_blocked and chk.candidate_id == top_chk_id
                )
                realized, causal_gain = self._evaluate_causal_rule_outcome(
                    rule=rule,
                    res_all_passed=res.all_passed,
                    rolled_back=cand_rolled_back,
                    ledger=ledger,
                    features=chk.stage_a.features if chk.stage_a else None,
                    cand_residual=cand_residual,
                    cand_lat_us=lat_us,
                    cand_obj_score=obj_score,
                    parent_ledger=parent_ledger,
                    parent_residual=parent_residual,
                    parent_lat_us=parent_lat_us,
                    parent_obj_score=parent_obj_score,
                    has_cand_msprof=bool(cached_cand_msprof),
                    cand_global_score=(res.surrogate_score if selected_stone is None else None),
                    parent_global_score=(best_result.surrogate_score if selected_stone is None else None),
                )
                self._tracker.record_outcome(
                    rule.rule_id,
                    postcondition_realized=realized,
                    speed_gain=causal_gain,
                    topology_axes=cand_axes,
                )
                self._record_kb_event(
                    rule_id=rule.rule_id,
                    postcondition_realized=realized,
                    speed_gain=causal_gain,
                    topology_axes=cand_axes,
                    round_idx=round_idx,
                )

        regime_beam.update(beam_updates)
        if promoted and top_chk is not None:
            self._prune_stale_sibling_stepping_stones(regime_spec, top_chk.snippet_source)

        if (
            promoted
            and top_chk is not None
            and top_res is not None
            and max(rel_gain, local_rel_gain) >= self._config.min_flywheel_gain
            and self._flywheel is not None
        ):
            top_local_eval = top_res.for_regime(regime_spec.regime_tag)
            best_local_eval = best_result.for_regime(regime_spec.regime_tag)
            fly_base_lat = (
                best_local_eval.mean_latency_us()
                if best_local_eval.anchors
                else best_result.mean_latency_us()
            )
            fly_win_lat = (
                top_local_eval.mean_latency_us()
                if top_local_eval.anchors
                else top_res.mean_latency_us()
            )
            try:
                round_admitted.extend(
                    self._run_flywheel_distillation(
                        baseline_source=incumbent_source,
                        winner_source=new_incumbent,
                        baseline_latency_us=fly_base_lat,
                        winner_latency_us=fly_win_lat,
                        func_name=func_name,
                        regime_spec=regime_spec,
                        active_anchors=active_anchors,
                        topology_axes=parent_axes,
                        rel_gain=max(rel_gain, local_rel_gain),
                    )
                )
                admitted_rule_ids.extend(round_admitted)
            except (OSError, subprocess.SubprocessError, TimeoutError, RuntimeError, ValueError) as exc:
                logger.warning(
                    "Flywheel distillation failed for regime %s: %s",
                    regime_spec.regime_tag,
                    exc,
                )

        pruned_rules = self._tracker.prune_underperforming(
            topology_axes=parent_axes,
            round_idx=round_idx,
        )
        if pruned_rules:
            logger.debug("Pruned rules for topology_axes %s: %s", parent_axes.key(), pruned_rules)
        self._tracker.decay_and_revive(
            round_idx=round_idx,
            min_cooldown_rounds=10,
            decay_factor=0.99,
        )
        promoted_local = self._tracker.promote_verified_local_rules()
        if promoted_local and self._experience_lake_dir is not None:
            for rid in promoted_local:
                rule_obj = self._tracker.rules.get(rid)
                if rule_obj is not None:
                    try:
                        save_distilled_rule_file(self._experience_lake_dir, rule_obj)
                    except (OSError, ValueError) as exc:
                        logger.debug("Persisting promoted rule %s skipped: %s", rid, exc)

        trackb_evidence = self._collect_trackb_round_evidence(
            round_idx=round_idx,
            regime_spec=regime_spec,
            blueprint=blueprint,
            live_plan=live_plan,
            active_mask=active_mask,
            checks=l0_l1_checks,
            l2_results=l2_results,
            active_kernel=round_active_kernel,
            l2_run_ids=l2_run_ids,
            parent_sha256=parent_sha,
            parent_l1_summary=parent_l1_summary,
            parent_l1_summary_source=parent_l1_summary_source,
        )
        record = CampaignRoundRecord(
            round_idx=round_idx,
            regime_tag=regime_spec.regime_tag,
            builder_name=func_name,
            parent_lane=parent_lane,
            creative_effort=creative_effort,
            l0_l1_passed_ids=tuple(c.candidate_id for c in passed_checks),
            l0_l1_failed=tuple(failed_entries),
            best_round_speedup=best_round_speedup,
            best_so_far_speedup=new_best.avg_speedup,
            promoted=promoted,
            rolled_back_l3=rolled_back_l3,
            stagnation_rounds_after=stagnation_after,
            admitted_rule_ids=tuple(round_admitted),
            best_round_surrogate_score=best_round_surrogate_score,
            best_so_far_surrogate_score=new_best.surrogate_score,
            pruned_rule_ids=pruned_rules,
            promotion_attempts=promotion_attempts,
            trackb_evidence=trackb_evidence,
        )
        return (
            record,
            new_incumbent,
            new_best,
            stagnation_after,
            promoted,
            rolled_back_l3,
            escalated,
            promotion_blocked,
        )

    async def run(self, *, resume: bool = False) -> CampaignReport:
        """Execute the unattended optimisation campaign until a stopping criterion.

        A resumed campaign first continues any saved L3 execution, then any saved
        L4 execution, without regenerating or resubmitting the candidate. L4 is
        never submitted while L3 promotion evidence is unresolved, and a
        successful same-candidate retry resumes from the promoted regime.
        """
        campaign_start = self._clock()
        if resume and self._checkpoint_path is not None and self._checkpoint_path.is_file():
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
            incumbent_source = self._target_module_path.read_text(encoding="utf-8")
            active_anchors = list(self._initial_anchors)
            baseline_result = self._evaluate_single_baseline(incumbent_source, active_anchors)
            self._collect_msprof_for_regimes(
                incumbent_source,
                baseline_result,
                active_anchors,
                self._regime_specs,
            )
            base_sa = None
            if self._regime_specs:
                try:
                    base_sa = self._run_l0(incumbent_source, self._regime_specs[0])
                except (SyntaxError, RuntimeError, ValueError, TypeError, OSError):
                    base_sa = None
            base_ledger = base_sa.ledger if base_sa is not None else None
            base_axes = self._derive_topology_axes(
                base_ledger,
                features=base_sa.features if base_sa is not None else None,
                route=base_sa.route if base_sa is not None else None,
            )
            self._record_surrogate_observations(
                source=incumbent_source,
                result=baseline_result,
                active_anchors=active_anchors,
                eval_tier_override="baseline",
                ledger=base_ledger,
                axes=base_axes,
                worker_kind="baseline",
                backend_track=(
                    base_sa.route.track
                    if self._enable_track_router and base_sa is not None and base_sa.route is not None
                    else "tilelang_dsl"
                ),
            )
            best_result = baseline_result
            frozen_regimes = set()
            current_regime = None
            stagnation_rounds = 0
            stagnation_by_regime = {}
            last_unpromoted_regime = None
            pending_promotion = None
            pending_l4 = None
            rounds = []
            promoted_rounds = []
            rolled_back_rounds = []
            escalated_rounds = []
            admitted_rule_ids = []
            self._save_checkpoint(
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
                stop_reason="in_progress",
                l4_result=None,
                stagnation_by_regime=stagnation_by_regime,
                last_unpromoted_regime=last_unpromoted_regime,
                pending_promotion=pending_promotion,
                pending_l4=pending_l4,
            )

        self._pending_promotion = pending_promotion
        self._pending_l4 = pending_l4
        stop_reason = ""
        promotion_blocked = False
        l4_result: L4CannbenchResult | None = None
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
            elif (
                pending_promotion is None
                and retried_pending.round_idx in promoted_rounds
            ):
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
        start_round = (rounds[-1].round_idx + 1) if rounds else 1
        budget_sec = self._config.max_wall_clock_min * 60.0

        for round_idx in range(start_round, max(0, self._config.max_rounds) + 1):
            if stop_reason:
                break
            if (self._clock() - campaign_start) >= budget_sec:
                stop_reason = "budget_exhausted"
                break

            for spec in self._regime_specs:
                if stagnation_by_regime.get(spec.regime_tag, 0) >= self._config.stagnation_freeze_rounds:
                    frozen_regimes.add(spec.regime_tag)
            if current_regime and current_regime in frozen_regimes:
                current_regime = None

            active_regime = self._round_regime_override(round_idx)
            if active_regime is None:
                active_regime = self._select_active_regime(
                    current_regime,
                    frozen_regimes,
                    best_result,
                    active_anchors,
                    allow_satisfied_fallback=not self._allow_early_stop(len(rounds)),
                    exclude_regime=last_unpromoted_regime,
                )
            if active_regime is None:
                stop_reason = "all_regimes_converged"
                break
            active_tag = active_regime.regime_tag

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
            if promoted:
                stagnation_rounds = 0
                stagnation_by_regime[active_tag] = 0
                current_regime = active_tag
                last_unpromoted_regime = None
                promoted_rounds.append(round_idx)
            else:
                stagnation_by_regime[active_tag] = stagnation_rounds
                current_regime = None
                last_unpromoted_regime = active_tag
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
                elif early_stop_allowed and best_result.is_roofline_saturated(self._perf_target):
                    stop_reason = "roofline_saturated"
                elif stagnation_by_regime.get(active_tag, 0) >= self._config.stagnation_freeze_rounds:
                    frozen_regimes.add(active_tag)
                    next_reg = self._select_active_regime(
                        None,
                        frozen_regimes,
                        best_result,
                        active_anchors,
                        allow_satisfied_fallback=not early_stop_allowed,
                        exclude_regime=last_unpromoted_regime,
                    )
                    if next_reg is None:
                        stop_reason = "all_regimes_converged"

            self._save_checkpoint(
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
                stop_reason=stop_reason or "in_progress",
                l4_result=None,
                stagnation_by_regime=stagnation_by_regime,
                last_unpromoted_regime=last_unpromoted_regime,
                pending_promotion=pending_promotion,
                pending_l4=self._pending_l4,
            )
            if stop_reason:
                break

        if not stop_reason:
            stop_reason = "budget_exhausted"

        if (
            self._config.run_final_l4
            and not promotion_blocked
            and not l4_was_resolved
        ):
            l4_result = self._run_final_l4_tracked(
                incumbent_source=incumbent_source,
                prior_stop_reason=stop_reason,
            )
            self._record_l4_observations(
                source=incumbent_source,
                l4_result=l4_result,
                active_anchors=active_anchors,
            )
            anchor_nums = {a.case_num for a in active_anchors}
            regressed_nums = sorted(
                cnum
                for cnum, sp in l4_result.per_case_speedups.items()
                if cnum not in anchor_nums and sp < self._config.l4_regression_ratio
            )
            if regressed_nums and l4_result.passed:
                resolved_by_cnum: dict[int, RegimeTargetSpec] = {}
                for cnum in regressed_nums:
                    resolved = self._regime_specs[0]
                    try:
                        dims = lookup_case_dims(
                            self._config.operator_name,
                            cnum,
                            cannbench_repo=self._cannbench_repo,
                            env_file=self._env_file,
                        )
                        resolved = resolve_regime_for_case(
                            self._config.operator_name,
                            dims,
                            self._regime_specs,
                        )
                    except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
                        logger.warning(
                            "Failed to resolve regime for regressed case %s: %s",
                            cnum,
                            exc,
                        )
                    resolved_by_cnum[cnum] = resolved
                    active_anchors = list(
                        append_dynamic_anchor(
                            active_anchors,
                            operator_name=self._config.operator_name,
                            case_num=cnum,
                            regime_tag=resolved.regime_tag,
                            cannbench_repo=self._cannbench_repo,
                            env_file=self._env_file,
                        )
                    )
                worst_cnum = min(
                    regressed_nums,
                    key=lambda c: (l4_result.per_case_speedups.get(c, float("inf")), c),
                )
                catch_regime = resolved_by_cnum[worst_cnum]
                catch_tag = catch_regime.regime_tag
                catch_idx = (rounds[-1].round_idx + 1) if rounds else 1
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
                    round_idx=catch_idx,
                    regime_spec=catch_regime,
                    incumbent_source=incumbent_source,
                    best_result=best_result,
                    active_anchors=active_anchors,
                    stagnation_rounds=stagnation_by_regime.get(catch_tag, 0),
                    admitted_rule_ids=admitted_rule_ids,
                )
                rounds.append(rec)
                pending_promotion = self._pending_promotion
                if promoted:
                    stagnation_by_regime[catch_tag] = 0
                    promoted_rounds.append(catch_idx)
                else:
                    stagnation_by_regime[catch_tag] = stagnation_rounds
                if rolled_back:
                    rolled_back_rounds.append(catch_idx)
                if escalated:
                    escalated_rounds.append(catch_idx)
                if promotion_blocked:
                    stop_reason = "promotion_blocked"
                if l4_result.passed and not promotion_blocked:
                    l4_result = self._run_final_l4_tracked(
                        incumbent_source=incumbent_source,
                        prior_stop_reason=stop_reason,
                    )
                    self._record_l4_observations(
                        source=incumbent_source,
                        l4_result=l4_result,
                        active_anchors=active_anchors,
                    )

        self._save_checkpoint(
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
            pending_promotion=pending_promotion,
            pending_l4=self._pending_l4,
        )

        return CampaignReport(
            operator_name=self._config.operator_name,
            stop_reason=stop_reason,
            total_rounds=len(rounds),
            baseline_result=baseline_result,
            best_result=best_result,
            rounds=tuple(rounds),
            promoted_rounds=tuple(promoted_rounds),
            rolled_back_rounds=tuple(rolled_back_rounds),
            escalated_rounds=tuple(escalated_rounds),
            frozen_regimes=tuple(sorted(frozen_regimes)),
            active_anchors=tuple(active_anchors),
            admitted_rule_ids=tuple(admitted_rule_ids),
            l4_result=l4_result,
            pending_promotion=pending_promotion,
            pending_l4=self._pending_l4,
            promotion_blocked=promotion_blocked,
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
        help="Target operator to optimize (short name e.g. rms_norm or task id e.g. level2/rms_norm).",
    )
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--max-wall-clock-min", type=float, default=180.0)
    parser.add_argument("--device-id", default="1", help="Authorized NPU device ID.")
    parser.add_argument(
        "--device-ids",
        type=parse_device_ids,
        default=(),
        help=(
            "Extra explicitly authorized NPU cards for whole-L4 attempts, "
            "comma-separated; a blocked-env preemption moves to the next "
            "untried pool card."
        ),
    )
    parser.add_argument(
        "--retry-device-id",
        default="",
        help=(
            "Explicitly authorized alternate NPU card for at most one fresh "
            "retry after a proven-terminal infrastructure failure."
        ),
    )
    parser.add_argument(
        "--disable-early-goal-stop",
        action="store_true",
        help="Continue optimizing through max-rounds even after target thresholds pass.",
    )
    parser.add_argument(
        "--min-surrogate-gain",
        type=float,
        default=0.25,
        help="Minimum surrogate_score gain on [0, 100] scale required for L3 promotion.",
    )
    parser.add_argument("--checkpoint", default=None, help="Atomic checkpoint JSON path.")
    parser.add_argument("--report-json", default=None, help="Optional final report JSON path.")
    parser.add_argument("--run-final-l4", action="store_true", help="Run final official L4 gate.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint.")
    parser.add_argument("--k-rule-workers", type=int, default=2)
    parser.add_argument(
        "--target-module",
        default=None,
        help="Optional override path to target operator kernels module.",
    )
    parser.add_argument("--source-root", default=None, help="TileFoundry workspace root.")
    parser.add_argument("--env-file", default=None, help="Path to tilefoundry-npu env.sh.")
    parser.add_argument(
        "--enable-tilesize-prescreen",
        action="store_true",
        help="Enable tile-size prescreen guidance table in creative prompts.",
    )
    parser.add_argument(
        "--cost-prescreen-top-k",
        type=int,
        default=None,
        help="Keep only the top-K cost-model candidates during creative prescreen.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute PerfTargetCampaignRunner from CLI arguments and emit JSON summary."""
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    cfg = CampaignConfig(
        operator_name=args.operator,
        max_rounds=args.max_rounds,
        max_wall_clock_min=args.max_wall_clock_min,
        k_rule_workers=args.k_rule_workers,
        min_surrogate_gain=args.min_surrogate_gain,
        disable_early_goal_stop=bool(args.disable_early_goal_stop),
        run_final_l4=bool(args.run_final_l4),
        device_id=str(args.device_id),
        retry_device_id=str(args.retry_device_id),
        device_ids=tuple(args.device_ids),
        enable_tilesize_prescreen=bool(args.enable_tilesize_prescreen),
        cost_prescreen_top_k=(
            int(args.cost_prescreen_top_k) if args.cost_prescreen_top_k is not None else None
        ),
    )
    tracker = BayesianRuleTracker.from_seed_yaml()
    flywheel = FlywheelDistiller(
        funnel=GateFunnel(skip_stage_b=True),
        tracker=tracker,
    )
    runner = PerfTargetCampaignRunner(
        config=cfg,
        target_module_path=args.target_module,
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
    "CampaignConfig",
    "CampaignReport",
    "CampaignRoundRecord",
    "PerfTargetCampaignRunner",
    "ResolvedRegimeConfig",
    "build_arg_parser",
    "extract_regime_snippet",
    "main",
    "parse_device_ids",
    "patch_regime_into_module",
    "resolve_regime_config",
]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
