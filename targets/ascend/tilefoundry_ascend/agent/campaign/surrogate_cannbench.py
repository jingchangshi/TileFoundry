"""Four-tier host-outside-gate evaluation pyramid (L0->L1->L2->L3->L4) for campaigns."""

from __future__ import annotations

import ast
import asyncio
import fcntl
import hashlib
import importlib.util
import inspect
import json
import logging
import math
import os
import re
import select
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import uuid
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

import yaml

from tilefoundry_ascend.agent.campaign.anchor_specs import (
    AnchorCaseSpec,
    PerfTargetSpec,
    compute_case_eq3_score,
    default_source_root,
    get_default_anchors,
    get_operator_entrypoint,
    get_operator_module_relpath,
    get_operator_task,
    lookup_case_dims,
    parse_simple_env_file,
    resolve_cannbench_repo,
    resolve_cannbench_tasks_root,
)
from tilefoundry_ascend.agent.campaign.latency_algebra import (
    compute_adaptive_poison_pill_timeout_sec,
)
from tilefoundry_ascend.agent.orchestrator.cpp_slice_protocol import (
    classify_stage_failure_for_repair,
    parse_scoped_search_replace_blocks,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    StageAResult,
    StageBResult,
    _bounded_compile_diagnostic,
    _classify_stage_b_overflow_fault_code,
    _has_vf_stack_overflow_signal,
    _merge_dims_into_kwargs,
    _merge_pass_configs_and_flags,
    _normalize_trackb_unit_source,
    _parse_cce_res_usage_with_summary,
    build_stage_b_driver_source,
    check_repair_monotonicity,
    extract_ordered_tensor_shapes,
    run_stage_a_static,
    run_stage_b_compile,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    KernelLLMBackend,
    extract_python_dsl,
)
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import (
    build_syntax_repair_request,
    extract_lowered_compiler_excerpt as _extract_lowered_compiler_excerpt,
    extract_lowered_compiler_excerpt_with_lang,
)
from tilefoundry_ascend.ascend.compile_cache.artifact_store import (
    ArtifactStore,
    collect_new_tl_cache_binaries,
    compute_compile_task_key,
    extract_reachable_closure_source,
    is_verified_binary_file,
    merge_tl_cache_tree,
    resolve_unified_tl_cache_dir,
    seed_tl_cache_tree,
    snapshot_tl_cache_binaries,
)
from tilefoundry_ascend.ascend.cost_model.bisheng_res_parser import (
    BishengResourceSummary,
)
from tilefoundry_ascend.ascend.cost_model.cost_gate_policy import (
    is_paradigm_shift_exempt,
)
from tilefoundry_ascend.ascend.cost_model.microarch_profile import (
    extract_microarch_profile_from_cce,
)
from tilefoundry_ascend.ascend.sil.schema import (
    ConstraintRelaxationMask,
    TopologyAxes,
)

logger = logging.getLogger(__name__)


def wait_for_npu_quiescent(max_retries: int = 10):
    """Wait for NPU devices (davinci) to be closed in /proc/self/fd/."""
    for _ in range(max_retries):
        found = False
        try:
            fd_path = Path("/proc/self/fd")
            for fd in fd_path.iterdir():
                try:
                    if "davinci" in str(fd.readlink()):
                        found = True
                        break
                except OSError:
                    continue
        except OSError:
            pass
        if not found:
            return
        time.sleep(1.0)
    logger.warning("NPU still busy after waiting for quiescence.")

DEFAULT_AGENT_GATES_MAX_ATTEMPTS = 5
DEFAULT_AGENT_GATES_BACKOFF_SEC = 3.0

AGENT_GATES_ADAPTER = "tilefoundry-npu"
AGENT_GATES_DEVICE_FAST_GATE = "tilefoundry-npu-device-fast"
AGENT_GATES_L3_PROMOTION_GATE = "tilefoundry-npu-l3-promotion"
AGENT_GATES_L4_CANNBENCH_GATE = "tilefoundry-npu-cannbench"
AGENT_GATES_L4_CANNBENCH_PREP_GATE = "tilefoundry-npu-cannbench-prep"
FAST_REQUEST_SCHEMA = "tilefoundry-fast-request/v1"
FAST_RESPONSE_SCHEMA = "tilefoundry-fast-response/v1"
FAST_REQUEST_ENV = "TF_FAST_REQUEST"
FAST_SOURCE_ROOT_ENV = "TF_FAST_SOURCE_ROOT"
FAST_STATUS_OK = "ok"
FAST_STATUS_CANDIDATE_FAILURE = "candidate_failure"
FAST_STATUS_REQUEST_ERROR = "request_error"
FAST_STATUS_INFRASTRUCTURE_ERROR = "infrastructure_error"
DEFAULT_L3_PROMOTION_WAIT_SEC = 1920
DEFAULT_L4_CANNBENCH_WAIT_SEC = 5400
_V2_WAIT_SLICE_SEC = 60.0
_V2_SUBMIT_TIMEOUT_SEC = 60.0
_V2_POST_WAIT_TIMEOUT_SEC = 60.0
_V2_STRIPPED_PYTHON_ENV_KEYS = frozenset(
    {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE"}
)

FAILURE_KIND_CANDIDATE = "candidate"
FAILURE_KIND_INFRASTRUCTURE = "infrastructure"


@dataclass(frozen=True)
class AnchorEvalMetrics:
    """Single CANNBench anchor accuracy, latency, speedup, and Roofline eta record."""

    case_num: int
    regime_tag: str
    accuracy_passed: bool
    cand_us: float
    ref_perf_us: float
    t_hw_us: float
    speedup: float
    roofline_eta: float
    eq3_score: float
    error_message: str = ""
    ring_burst_k: int = 1
    burst_batch_cv: float = 0.0
    spike_retries: int = 0
    card_ext_hbm_mb: int = 0
    paired_speedup_vs_incumbent: float = 0.0
    paired_rel_mad: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Serialize anchor evaluation metrics to a JSON-compatible dictionary."""
        return {
            "case_num": self.case_num,
            "regime_tag": self.regime_tag,
            "accuracy_passed": self.accuracy_passed,
            "cand_us": round(self.cand_us, 6),
            "ref_perf_us": round(self.ref_perf_us, 6),
            "t_hw_us": round(self.t_hw_us, 6),
            "speedup": round(self.speedup, 6),
            "roofline_eta": round(self.roofline_eta, 6),
            "eq3_score": round(self.eq3_score, 6),
            "error_message": self.error_message,
            "ring_burst_k": int(self.ring_burst_k),
            "burst_batch_cv": round(self.burst_batch_cv, 6),
            "spike_retries": int(self.spike_retries),
            "card_ext_hbm_mb": int(self.card_ext_hbm_mb),
            "paired_speedup_vs_incumbent": round(self.paired_speedup_vs_incumbent, 6),
            "paired_rel_mad": round(self.paired_rel_mad, 6),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AnchorEvalMetrics:
        """Reconstruct AnchorEvalMetrics from a JSON dictionary."""
        return cls(
            case_num=int(payload["case_num"]),
            regime_tag=str(payload["regime_tag"]),
            accuracy_passed=bool(payload["accuracy_passed"]),
            cand_us=float(payload["cand_us"]),
            ref_perf_us=float(payload["ref_perf_us"]),
            t_hw_us=float(payload["t_hw_us"]),
            speedup=float(payload["speedup"]),
            roofline_eta=float(payload["roofline_eta"]),
            eq3_score=float(payload["eq3_score"]),
            error_message=str(payload.get("error_message", "")),
            ring_burst_k=max(1, int(payload.get("ring_burst_k", 1) or 1)),
            burst_batch_cv=float(payload.get("burst_batch_cv", 0.0) or 0.0),
            spike_retries=max(0, int(payload.get("spike_retries", 0) or 0)),
            card_ext_hbm_mb=max(0, int(payload.get("card_ext_hbm_mb", 0) or 0)),
            paired_speedup_vs_incumbent=float(
                payload.get("paired_speedup_vs_incumbent", 0.0) or 0.0
            ),
            paired_rel_mad=float(payload.get("paired_rel_mad", 0.0) or 0.0),
        )


@dataclass(frozen=True)
class CandidateSurrogateResult:
    """Aggregated L2 surrogate CANNBench evaluation summary across active anchors."""

    candidate_id: str
    operator_name: str
    all_passed: bool
    avg_speedup: float
    min_case_speedup: float
    avg_roofline_eta: float
    surrogate_score: float
    anchors: tuple[AnchorEvalMetrics, ...] = ()
    error_message: str = ""

    def regime_headroom(self, regime_tag: str) -> float:
        """Return mean Roofline headroom (1 - eta) across anchors matching regime_tag."""
        matched = [a for a in self.anchors if a.regime_tag == regime_tag]
        if not matched:
            return max(0.0, 1.0 - self.avg_roofline_eta)
        return sum(max(0.0, 1.0 - a.roofline_eta) for a in matched) / len(matched)

    def for_regime(self, regime_tag: str) -> CandidateSurrogateResult:
        """Return CandidateSurrogateResult filtered strictly to anchors of regime_tag."""
        matched = tuple(a for a in self.anchors if a.regime_tag == regime_tag)
        if not matched:
            return CandidateSurrogateResult(
                candidate_id=self.candidate_id,
                operator_name=self.operator_name,
                all_passed=False,
                avg_speedup=0.0,
                min_case_speedup=0.0,
                avg_roofline_eta=0.0,
                surrogate_score=0.0,
                anchors=(),
                error_message=self.error_message,
            )
        return aggregate_surrogate_result(
            self.candidate_id,
            self.operator_name,
            matched,
            error_message=self.error_message,
        )

    def meets_perf_target(self, spec: PerfTargetSpec) -> bool:
        """Return True iff accuracy, score, speedup, eta, and latency limits pass."""
        if not self.all_passed or not self.anchors:
            return False
        if self.surrogate_score < spec.target_surrogate_score:
            return False
        if self.avg_speedup < spec.target_avg_speedup:
            return False
        if self.min_case_speedup < spec.target_min_case_speedup:
            return False
        if self.avg_roofline_eta < spec.target_roofline_eta:
            return False
        by_case = {a.case_num: a for a in self.anchors}
        for case_num, max_us in spec.per_anchor_max_us.items():
            anchor = by_case.get(int(case_num))
            if anchor is None or anchor.cand_us <= 0.0 or anchor.cand_us > float(max_us):
                return False
        return True

    def is_roofline_saturated(self, spec: PerfTargetSpec) -> bool:
        """Return True iff every active anchor has reached target_roofline_eta."""
        if not self.all_passed or not self.anchors:
            return False
        return all(a.roofline_eta >= spec.target_roofline_eta for a in self.anchors)

    def mean_latency_us(self) -> float:
        """Return arithmetic mean candidate latency in microseconds across anchors."""
        valid = [a.cand_us for a in self.anchors if a.cand_us > 0.0]
        if valid:
            return sum(valid) / len(valid)
        if self.avg_speedup > 0.0:
            return 1000.0 / self.avg_speedup
        return 1e9

    def to_dict(self) -> dict[str, object]:
        """Serialize CandidateSurrogateResult to a JSON-compatible dictionary."""
        return {
            "candidate_id": self.candidate_id,
            "operator_name": self.operator_name,
            "all_passed": self.all_passed,
            "avg_speedup": round(self.avg_speedup, 6),
            "min_case_speedup": round(self.min_case_speedup, 6),
            "avg_roofline_eta": round(self.avg_roofline_eta, 6),
            "surrogate_score": round(self.surrogate_score, 6),
            "anchors": [a.to_dict() for a in self.anchors],
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CandidateSurrogateResult:
        """Reconstruct CandidateSurrogateResult from a JSON dictionary."""
        anchors = tuple(
            AnchorEvalMetrics.from_dict(item)
            for item in payload.get("anchors", ())
            if isinstance(item, Mapping)
        )
        return cls(
            candidate_id=str(payload["candidate_id"]),
            operator_name=str(payload["operator_name"]),
            all_passed=bool(payload["all_passed"]),
            avg_speedup=float(payload["avg_speedup"]),
            min_case_speedup=float(payload["min_case_speedup"]),
            avg_roofline_eta=float(payload["avg_roofline_eta"]),
            surrogate_score=float(payload["surrogate_score"]),
            anchors=anchors,
            error_message=str(payload.get("error_message", "")),
        )


class L2BatchResults(dict[str, CandidateSurrogateResult]):
    """L2 candidate results carrying AgentGates run ids and prewarm/NPU telemetry.

    The mapping protocol is unchanged so legacy evaluators can keep returning plain
    dictionaries; ``run_ids`` maps each measured candidate to its AgentGates run id,
    and ``prewarm_stats``, ``l2_prewarm_sec``, ``l2_npu_sec`` capture L2 timing.
    """

    def __init__(
        self,
        results: Mapping[str, CandidateSurrogateResult],
        run_ids: Mapping[str, str],
        *,
        prewarm_stats: Mapping[str, int] | None = None,
        l2_prewarm_sec: float = 0.0,
        l2_npu_sec: float = 0.0,
        null_launch_us: float | None = None,
    ) -> None:
        super().__init__(results)
        self.run_ids: dict[str, str] = dict(run_ids)
        stats: dict[str, int] = {
            "total_tasks": 0,
            "cache_hits": 0,
            "compiled_tasks": 0,
            "failed_candidates": 0,
        }
        if prewarm_stats is not None:
            for key in stats:
                if key in prewarm_stats:
                    stats[key] = int(prewarm_stats[key])
        self.prewarm_stats: dict[str, int] = stats
        self.l2_prewarm_sec: float = float(l2_prewarm_sec)
        self.l2_npu_sec: float = float(l2_npu_sec)
        self.null_launch_us: float | None = (
            float(null_launch_us)
            if null_launch_us is not None and not isinstance(null_launch_us, bool)
            else None
        )


@dataclass(frozen=True)
class L0L1CandidateCheck:
    """Host-CPU Stage L0 static + Stage L1 bisheng pre-filter result for one candidate."""

    candidate_id: str
    passed: bool
    failed_tier: str | None
    patched_module_source: str
    snippet_source: str
    repaired: bool = False
    stage_a: StageAResult | None = None
    stage_b: StageBResult | None = None
    error_message: str = ""
    fault_plane: str = ""
    fault_code: str = ""
    wall_repair_sec: float = 0.0
    cum_repair_worker_sec: float = 0.0


class L0L1RoundResults(list[L0L1CandidateCheck]):
    """Round filter result list carrying concurrent and cumulative repair timing."""

    def __init__(
        self,
        items: Sequence[L0L1CandidateCheck],
        *,
        wall_repair_sec: float = 0.0,
        cum_repair_worker_sec: float = 0.0,
    ) -> None:
        super().__init__(items)
        self.wall_repair_sec = float(wall_repair_sec)
        self.cum_repair_worker_sec = float(cum_repair_worker_sec)
        self.round_metadata: dict[str, float] = {
            "wall_repair_sec": self.wall_repair_sec,
            "cum_repair_worker_sec": self.cum_repair_worker_sec,
        }


class AgentGatesDeviceFaultError(RuntimeError):
    """Structured error raised when an NPU device reports device_fault or rejection."""

    def __init__(
        self,
        message: str,
        *,
        fault_plane: str = "INFRA_DEVICE_FAULT",
        fault_code: str = "NPU_DEVICE_TSD_FAULT",
        failed_tier: str = "l2_npu",
        device_id: str = "",
        rejected_cards: Collection[str] = (),
    ) -> None:
        super().__init__(message)
        self.fault_plane = fault_plane
        self.fault_code = fault_code
        self.failed_tier = failed_tier
        self.device_id = device_id
        self.rejected_cards = tuple(sorted(str(c) for c in rejected_cards))


class OccupancyExhaustedError(RuntimeError):
    """Raised when NPU host-gate contention or BLOCKED_ENV (rc=75) exhausts retries."""

    def __init__(self, attempts: int, receipt_reasons: tuple[str, ...]) -> None:
        self.fault_plane = "INFRA_OCCUPANCY"
        self.exit_code = 75
        self.attempts = attempts
        self.receipt_reasons = receipt_reasons
        joined = ", ".join(receipt_reasons)
        super().__init__(
            f"NPU occupancy/env blocked after {attempts} attempts (rc=75): [{joined}]"
        )


@dataclass(frozen=True)
class L3AcceptanceResult:
    """Outcome of AgentGates L3 full-contract acceptance verification."""

    passed: bool
    run_id: str = ""
    receipt_path: str = ""
    elapsed_sec: float = 0.0
    error_message: str = ""
    failure_kind: str = ""
    failure_category: str = ""
    pending: bool = False
    execution_id: str = ""
    candidate_sha256: str = ""
    target_path: str = ""
    device_id: str = ""
    state: str = ""
    execution_terminal: bool = True
    blocked_env_category: str = ""
    failed_case_nums: tuple[int, ...] = ()
    failure_stage: str = ""


@dataclass(frozen=True)
class L4CannbenchResult:
    """Outcome and per-case telemetry of official L4 tilefoundry-npu-cannbench gate."""

    passed: bool
    avg_speedup: float = 0.0
    score: float = 0.0
    per_case_speedups: dict[int, float] = field(default_factory=dict)
    per_case_elapsed_us: dict[int, float] = field(default_factory=dict)
    run_id: str = ""
    receipt_path: str = ""
    report_dir: str = ""
    error_message: str = ""
    failure_kind: str = ""
    execution_id: str = ""
    official_report_valid: bool = False
    all_cases_passed: bool = False
    all_timings_valid: bool = False
    case_count: int = 0
    passed_cases: int = 0
    device_id: str = ""
    state: str = ""
    execution_terminal: bool = True
    blocked_env_category: str = ""
    device_rejected: bool = False
    target_path: str = ""
    candidate_sha256: str = ""
    attempts: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize L4 official CANNBench result to a JSON-compatible dictionary."""
        return {
            "passed": self.passed,
            "avg_speedup": round(self.avg_speedup, 6),
            "score": round(self.score, 6),
            "per_case_speedups": {str(k): v for k, v in sorted(self.per_case_speedups.items())},
            "per_case_elapsed_us": {str(k): v for k, v in sorted(self.per_case_elapsed_us.items())},
            "run_id": self.run_id,
            "receipt_path": self.receipt_path,
            "report_dir": self.report_dir,
            "error_message": self.error_message,
            "failure_kind": self.failure_kind,
            "execution_id": self.execution_id,
            "official_report_valid": self.official_report_valid,
            "all_cases_passed": self.all_cases_passed,
            "all_timings_valid": self.all_timings_valid,
            "case_count": self.case_count,
            "passed_cases": self.passed_cases,
            "device_id": self.device_id,
            "state": self.state,
            "execution_terminal": self.execution_terminal,
            "blocked_env_category": self.blocked_env_category,
            "device_rejected": self.device_rejected,
            "target_path": self.target_path,
            "candidate_sha256": self.candidate_sha256,
            "attempts": [dict(item) for item in self.attempts],
        }


def build_anchor_metrics(
    spec: AnchorCaseSpec,
    *,
    accuracy_passed: bool,
    cand_us: float,
    error_message: str = "",
    ring_burst_k: int = 1,
    burst_batch_cv: float = 0.0,
    spike_retries: int = 0,
    card_ext_hbm_mb: int = 0,
    paired_speedup_vs_incumbent: float = 0.0,
    paired_rel_mad: float = 0.0,
) -> AnchorEvalMetrics:
    """Compute speedup, Roofline eta, and Eq.3 score from measured latency."""
    valid_lat = accuracy_passed and math.isfinite(cand_us) and cand_us > 0.0
    safe_lat = float(cand_us) if (math.isfinite(cand_us) and cand_us > 0.0) else 0.0
    speedup = (spec.ref_perf_us / safe_lat) if (valid_lat and spec.ref_perf_us > 0.0) else 0.0
    eta = (spec.t_hw_us / safe_lat) if (valid_lat and spec.t_hw_us > 0.0) else 0.0
    eq3 = compute_case_eq3_score(spec.ref_perf_us, safe_lat, spec.t_hw_us) if valid_lat else 0.0
    safe_cv = (
        float(burst_batch_cv)
        if math.isfinite(float(burst_batch_cv)) and float(burst_batch_cv) >= 0.0
        else 0.0
    )
    safe_paired_speedup = (
        float(paired_speedup_vs_incumbent)
        if valid_lat
        and math.isfinite(float(paired_speedup_vs_incumbent))
        and float(paired_speedup_vs_incumbent) >= 0.0
        else 0.0
    )
    safe_paired_rel_mad = (
        float(paired_rel_mad)
        if math.isfinite(float(paired_rel_mad)) and float(paired_rel_mad) >= 0.0
        else 0.0
    )
    return AnchorEvalMetrics(
        case_num=spec.case_num,
        regime_tag=spec.regime_tag,
        accuracy_passed=bool(accuracy_passed and safe_lat > 0.0),
        cand_us=safe_lat,
        ref_perf_us=spec.ref_perf_us,
        t_hw_us=spec.t_hw_us,
        speedup=speedup,
        roofline_eta=eta,
        eq3_score=eq3,
        error_message=error_message,
        ring_burst_k=max(1, int(ring_burst_k)),
        burst_batch_cv=safe_cv,
        spike_retries=max(0, int(spike_retries)),
        card_ext_hbm_mb=max(0, int(card_ext_hbm_mb)),
        paired_speedup_vs_incumbent=safe_paired_speedup,
        paired_rel_mad=safe_paired_rel_mad,
    )


COLD_L2_HW_THRESHOLD_US = 5.0
COLD_L2_PROBE_THRESHOLD_US = 8.0
RING_BURST_MIN_K = 24
RING_BURST_MAX_K = 32
RING_BURST_TARGET_BYTES = 64 * 1024 * 1024
RING_BURST_MAX_VRAM_BYTES = 1024 * 1024 * 1024


def compute_ring_burst_k(
    per_call_bytes: int,
    *,
    t_hw_us: float | None = None,
    first_probe_us: float | None = None,
    min_k: int = RING_BURST_MIN_K,
    max_k: int = RING_BURST_MAX_K,
    target_bytes: int = RING_BURST_TARGET_BYTES,
    max_vram_bytes: int = RING_BURST_MAX_VRAM_BYTES,
) -> int:
    """Return rotating ring-buffer burst factor K for cold-L2 micro-anchor timing."""
    needs_burst = (
        (t_hw_us is not None and math.isfinite(float(t_hw_us)) and 0.0 < float(t_hw_us) < COLD_L2_HW_THRESHOLD_US)
        or (
            first_probe_us is not None
            and math.isfinite(float(first_probe_us))
            and 0.0 < float(first_probe_us) < COLD_L2_PROBE_THRESHOLD_US
        )
    )
    if not needs_burst:
        return 1
    safe_min = max(1, int(min_k))
    safe_max = max(safe_min, int(max_k))
    call_bytes = max(0, int(per_call_bytes))
    if call_bytes <= 0:
        return safe_min
    vram_cap = max(1, int(max_vram_bytes) // call_bytes)
    if vram_cap < safe_min:
        return max(1, vram_cap)
    if call_bytes < target_bytes:
        scaled = math.ceil(float(target_bytes) / float(call_bytes))
        desired = min(safe_max, max(safe_min, scaled))
    else:
        desired = safe_min
    return max(1, min(desired, vram_cap))


def _tensor_nbytes(obj: object) -> int:
    if hasattr(obj, "numel") and hasattr(obj, "element_size"):
        return max(0, int(obj.numel()) * int(obj.element_size()))
    if hasattr(obj, "nbytes"):
        return max(0, int(obj.nbytes))
    return 0


def estimate_params_io_bytes(params: object) -> int:
    """Estimate total input/output tensor bytes referenced by a call parameter container."""
    total = 0
    seen: set[int] = set()

    def _walk(item: object) -> None:
        nonlocal total
        if isinstance(item, Mapping):
            for val in item.values():
                _walk(val)
            return
        if isinstance(item, (list, tuple)):
            for val in item:
                _walk(val)
            return
        nbytes = _tensor_nbytes(item)
        if nbytes > 0:
            obj_id = id(item)
            if obj_id not in seen:
                seen.add(obj_id)
                total += nbytes

    _walk(params)
    return total


def _clone_param_tree(item: object, clone_fn: Callable[[object], object] | None) -> object:
    if isinstance(item, Mapping):
        return {k: _clone_param_tree(v, clone_fn) for k, v in item.items()}
    if isinstance(item, tuple):
        return tuple(_clone_param_tree(v, clone_fn) for v in item)
    if isinstance(item, list):
        return [_clone_param_tree(v, clone_fn) for v in item]
    if _tensor_nbytes(item) > 0:
        if clone_fn is not None:
            return clone_fn(item)
        if hasattr(item, "clone") and callable(item.clone):
            return item.clone()
    return item


def allocate_rotating_ring_params(
    params: object,
    k: int,
    *,
    clone_fn: Callable[[object], object] | None = None,
) -> list[object]:
    """Pre-allocate *k* independent input/output parameter buffer copies for cold-L2 bursts."""
    count = max(1, int(k))
    return [_clone_param_tree(params, clone_fn) for _ in range(count)]


def _median_and_cv(samples: Sequence[float]) -> tuple[float, float]:
    med = float(statistics.median(samples))
    cv = float(statistics.pstdev(samples)) / max(med, 1e-9) if len(samples) >= 2 else 0.0
    return med, cv


def _invoke_callable(func: Callable[..., object], params: object) -> None:
    if isinstance(params, Mapping):
        func(**params)
    elif isinstance(params, (list, tuple)):
        func(*params)
    else:
        func(params)


def _resolve_npu_timing_fns(
    event_factory: Callable[[], object] | None,
    synchronize_fn: Callable[[], None] | None,
) -> tuple[Callable[[], object], Callable[[], None]]:
    if event_factory is not None and synchronize_fn is not None:
        return event_factory, synchronize_fn
    import torch  # noqa: PLC0415

    eff_event = (
        event_factory
        if event_factory is not None
        else (lambda: torch.npu.Event(enable_timing=True))
    )
    eff_sync = synchronize_fn if synchronize_fn is not None else torch.npu.synchronize
    return eff_event, eff_sync


def measure_npu_ring_burst_stats(
    func: Callable[..., object],
    params: object,
    *,
    warmup: int = 3,
    repeat: int = 5,
    t_hw_us: float | None = None,
    first_probe_us: float | None = None,
    event_factory: Callable[[], object] | None = None,
    synchronize_fn: Callable[[], None] | None = None,
    clone_fn: Callable[[object], object] | None = None,
    max_vram_bytes: int = RING_BURST_MAX_VRAM_BYTES,
) -> tuple[float, int, float, int]:
    """Measure micro-anchor latency returning (median_us, ring_burst_k, cv, spike_retries)."""
    is_real_npu = event_factory is None and synchronize_fn is None
    eff_event, eff_sync = _resolve_npu_timing_fns(event_factory, synchronize_fn)

    def _time_single_samples(p: object, reps: int) -> list[float]:
        samples: list[float] = []
        for _ in range(max(1, int(reps))):
            start_ev = eff_event()
            end_ev = eff_event()
            start_ev.record()
            _invoke_callable(func, p)
            end_ev.record()
            eff_sync()
            samples.append(float(start_ev.elapsed_time(end_ev)) * 1000.0)
        return samples

    for _ in range(max(1, int(warmup))):
        _invoke_callable(func, params)
    eff_sync()

    eff_probe_us = first_probe_us
    if eff_probe_us is None and (t_hw_us is None or float(t_hw_us) >= COLD_L2_HW_THRESHOLD_US):
        eff_probe_us = float(statistics.median(_time_single_samples(params, 1)))

    per_call_bytes = estimate_params_io_bytes(params)
    k = compute_ring_burst_k(
        per_call_bytes,
        t_hw_us=t_hw_us,
        first_probe_us=eff_probe_us,
        max_vram_bytes=max_vram_bytes,
    )
    if k <= 1:
        med, cv = _median_and_cv(_time_single_samples(params, repeat))
        return med, 1, cv, 0

    try:
        ring_inputs = allocate_rotating_ring_params(params, k, clone_fn=clone_fn)
    except RuntimeError:
        med, cv = _median_and_cv(_time_single_samples(params, repeat))
        return med, 1, cv, 0

    for step in range(min(k, max(1, int(warmup)))):
        _invoke_callable(func, ring_inputs[step % k])
    eff_sync()

    def _collect_burst_samples() -> list[float]:
        collected: list[float] = []
        for _ in range(max(1, int(repeat))):
            start_ev = eff_event()
            end_ev = eff_event()
            start_ev.record()
            for step in range(k):
                _invoke_callable(func, ring_inputs[step % k])
            end_ev.record()
            eff_sync()
            total_us = float(start_ev.elapsed_time(end_ev)) * 1000.0
            collected.append(total_us / float(k))
        return collected

    samples_us = _collect_burst_samples()
    if len(samples_us) < 3:
        med, cv = _median_and_cv(samples_us)
        return med, k, cv, 0

    med, cv = _median_and_cv(samples_us)
    if cv <= 0.12:
        return med, k, cv, 0

    batches: list[tuple[list[float], float, float]] = [(samples_us, med, cv)]
    spike_retries = 0
    for _ in range(2):
        spike_retries += 1
        if is_real_npu:
            time.sleep(0.02)
        retry_samples = _collect_burst_samples()
        retry_med, retry_cv = _median_and_cv(retry_samples)
        batches.append((retry_samples, retry_med, retry_cv))
        if retry_cv <= 0.12:
            break
    clean_batches = [b for b in batches if b[2] <= 0.12]
    if clean_batches:
        best = min(clean_batches, key=lambda b: (b[1], b[2]))
        return float(best[1]), k, float(best[2]), spike_retries
    pooled = sorted(val for batch, _, _ in batches for val in batch)
    keep = max(3, (len(pooled) + 1) // 2)
    lower_med, lower_cv = _median_and_cv(pooled[:keep])
    return lower_med, k, lower_cv, spike_retries


def measure_npu_ring_burst_us(
    func: Callable[..., object],
    params: object,
    *,
    warmup: int = 3,
    repeat: int = 5,
    t_hw_us: float | None = None,
    first_probe_us: float | None = None,
    event_factory: Callable[[], object] | None = None,
    synchronize_fn: Callable[[], None] | None = None,
    clone_fn: Callable[[object], object] | None = None,
    max_vram_bytes: int = RING_BURST_MAX_VRAM_BYTES,
) -> tuple[float, int]:
    """Measure micro-anchor latency with cold-L2 rotating ring-buffer burst enqueueing."""
    med, k, _cv, _retries = measure_npu_ring_burst_stats(
        func,
        params,
        warmup=warmup,
        repeat=repeat,
        t_hw_us=t_hw_us,
        first_probe_us=first_probe_us,
        event_factory=event_factory,
        synchronize_fn=synchronize_fn,
        clone_fn=clone_fn,
        max_vram_bytes=max_vram_bytes,
    )
    return med, k


def measure_paired_interleaved_micro_burst(
    candidates: Sequence[tuple[str, Callable[..., object]]],
    params: object,
    *,
    warmup: int = 3,
    repeat: int = 7,
    t_hw_us: float | None = None,
    first_probe_us: float | None = None,
    event_factory: Callable[[], object] | None = None,
    synchronize_fn: Callable[[], None] | None = None,
    clone_fn: Callable[[object], object] | None = None,
    max_vram_bytes: int = RING_BURST_MAX_VRAM_BYTES,
) -> dict[str, tuple[float, int, float, float, float]]:
    """Run B-round interleaved micro-burst timing across candidates on shared buffers.

    Returns ``{cid: (median_us, ring_burst_k, raw_cv, med_ratio_vs_0, paired_rel_mad)}``.
    """
    if not candidates:
        return {}
    eff_event, eff_sync = _resolve_npu_timing_fns(event_factory, synchronize_fn)

    def _time_once(
        fn: Callable[..., object],
        ring_inputs: Sequence[object],
        round_idx: int,
        k: int,
    ) -> float:
        start_ev = eff_event()
        end_ev = eff_event()
        start_ev.record()
        for step in range(k):
            _invoke_callable(fn, ring_inputs[(round_idx + step) % len(ring_inputs)])
        end_ev.record()
        eff_sync()
        return (float(start_ev.elapsed_time(end_ev)) * 1000.0) / float(k)

    warm_iters = max(1, int(warmup))
    for _cid, fn in candidates:
        for _ in range(warm_iters):
            _invoke_callable(fn, params)
    eff_sync()

    eff_probe_us = first_probe_us
    if eff_probe_us is None and (
        t_hw_us is None
        or not math.isfinite(float(t_hw_us))
        or float(t_hw_us) >= COLD_L2_HW_THRESHOLD_US
    ):
        eff_probe_us = _time_once(candidates[0][1], (params,), 0, 1)

    per_call_bytes = estimate_params_io_bytes(params)
    k = compute_ring_burst_k(
        per_call_bytes,
        t_hw_us=t_hw_us,
        first_probe_us=eff_probe_us,
        max_vram_bytes=max_vram_bytes,
    )
    ring_inputs: list[object] = [params]
    if k > 1:
        try:
            ring_inputs = allocate_rotating_ring_params(params, k, clone_fn=clone_fn)
            for step in range(min(k, warm_iters)):
                _invoke_callable(candidates[0][1], ring_inputs[step % k])
            eff_sync()
        except RuntimeError:
            k = 1
            ring_inputs = [params]

    num_rounds = max(7, int(repeat))
    num_cands = len(candidates)
    samples_by_idx: list[list[float]] = [[] for _ in range(num_cands)]
    for round_idx in range(num_rounds):
        for offset in range(num_cands):
            idx = (offset + round_idx) % num_cands
            _cid, fn = candidates[idx]
            samples_by_idx[idx].append(_time_once(fn, ring_inputs, round_idx, k))

    ref_samples = samples_by_idx[0]
    out: dict[str, tuple[float, int, float, float, float]] = {}
    for idx, (cid, _fn) in enumerate(candidates):
        cand_samples = samples_by_idx[idx]
        cand_us, raw_cv = _median_and_cv(cand_samples)
        if idx == 0:
            med_r = 1.0
            rel_mad = 0.0
        else:
            ratios = [
                ref_samples[b] / max(cand_samples[b], 1e-9)
                for b in range(num_rounds)
            ]
            med_r = float(statistics.median(ratios))
            mad_r = float(statistics.median([abs(x - med_r) for x in ratios]))
            rel_mad = mad_r / max(med_r, 1e-9)
        out[cid] = (cand_us, k, raw_cv, med_r, rel_mad)
    return out


def aggregate_surrogate_result(
    candidate_id: str,
    operator_name: str,
    anchor_metrics: Sequence[AnchorEvalMetrics],
    *,
    error_message: str = "",
) -> CandidateSurrogateResult:
    """Aggregate per-anchor metrics into speedup, eta, and surrogate CANNBench score."""
    metrics = tuple(anchor_metrics)
    if not metrics:
        return CandidateSurrogateResult(
            candidate_id=candidate_id,
            operator_name=operator_name,
            all_passed=False,
            avg_speedup=0.0,
            min_case_speedup=0.0,
            avg_roofline_eta=0.0,
            surrogate_score=0.0,
            anchors=(),
            error_message=error_message or "no anchor metrics evaluated",
        )
    all_ok = all(m.accuracy_passed and m.cand_us > 0.0 for m in metrics)
    avg_sp = sum(m.speedup for m in metrics) / len(metrics)
    min_sp = min(m.speedup for m in metrics)
    avg_eta = sum(m.roofline_eta for m in metrics) / len(metrics)
    case_terms = [(0.20 + 0.70 * m.eq3_score) if m.accuracy_passed else 0.0 for m in metrics]
    comp_score = 0.05 if any(m.cand_us > 0.0 for m in metrics) else 0.0
    bonus = 0.05 if all_ok else 0.0
    surrogate_score = (comp_score + bonus + (sum(case_terms) / len(metrics))) * 100.0
    return CandidateSurrogateResult(
        candidate_id=candidate_id,
        operator_name=operator_name,
        all_passed=all_ok,
        avg_speedup=avg_sp,
        min_case_speedup=min_sp,
        avg_roofline_eta=avg_eta,
        surrogate_score=surrogate_score,
        anchors=metrics,
        error_message=error_message,
    )


def _format_worst_element_diagnostic(actual: object, golden: object) -> str:
    """Summarize the single finite element with largest absolute difference."""
    import torch  # noqa: PLC0415

    if not isinstance(actual, torch.Tensor) or not isinstance(golden, torch.Tensor):
        return ""
    if actual.shape != golden.shape or actual.numel() == 0:
        return ""
    if actual.is_complex() or golden.is_complex():
        return ""
    with torch.no_grad():
        act = actual.detach().cpu().to(dtype=torch.float64)
        gold = golden.detach().cpu().to(dtype=torch.float64)
        diff = (act - gold).abs()
        finite_mask = torch.isfinite(diff) & torch.isfinite(act) & torch.isfinite(gold)
        if not bool(finite_mask.any().item()):
            return ""
        masked_diff = torch.where(finite_mask, diff, torch.full_like(diff, -1.0))
        flat_idx = int(torch.argmax(masked_diff.reshape(-1)).item())
        rem = flat_idx
        coord_rev: list[int] = []
        for dim in reversed(tuple(int(d) for d in act.shape)):
            coord_rev.append(rem % dim)
            rem //= dim
        coord = list(reversed(coord_rev))
        ref_val = float(gold.reshape(-1)[flat_idx].item())
        got_val = float(act.reshape(-1)[flat_idx].item())
        abs_diff = float(diff.reshape(-1)[flat_idx].item())
        rel_err = abs_diff / max(abs(ref_val), 1e-12)
        return (
            f"worst_element=(idx={coord}, ref={ref_val:.6g}, got={got_val:.6g}, "
            f"abs_diff={abs_diff:.6e}, rel_err={rel_err:.6e})"
        )


_NPU_COMPARE_SHIM_SOURCE = '''"""Runtime bridge adapting kernel_eval.utils.compare to npu_compare API."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


def _load_compare_tensors():
    cann_repo = Path({cann_repo!r})
    bench_src = str(cann_repo / "src")
    if bench_src not in sys.path:
        sys.path.insert(0, bench_src)
    from kernel_eval.utils.compare import compare_tensors

    return compare_tensors


_COMPARE_TENSORS = _load_compare_tensors()


def _format_nonfinite_diagnostic(actual: object, golden: object) -> str:
    """Summarize non-finite NaN or Inf mismatches when numerical diff metrics are zero."""
    import torch

    if not isinstance(actual, torch.Tensor) or not isinstance(golden, torch.Tensor):
        return ""
    if actual.shape != golden.shape or actual.numel() == 0:
        return ""
    if not actual.is_floating_point():
        return ""
    with torch.no_grad():
        act = actual.detach().cpu()
        gold = golden.detach().cpu()
        nan_act = torch.isnan(act)
        nan_gold = torch.isnan(gold)
        nan_diff = nan_act != nan_gold
        if bool(nan_diff.any().item()):
            first_coord = torch.nonzero(nan_diff, as_tuple=False)[0].tolist()
            act_nan_cnt = int(nan_act.sum().item())
            gold_nan_cnt = int(nan_gold.sum().item())
            return (
                f"NaN position mismatch (actual_nan={{act_nan_cnt}}/{{act.numel()}}, "
                f"golden_nan={{gold_nan_cnt}}/{{gold.numel()}}, first_coord={{first_coord}})"
            )
        inf_act = torch.isinf(act)
        inf_gold = torch.isinf(gold)
        inf_pos_diff = inf_act != inf_gold
        both_inf = inf_act & inf_gold
        if bool(both_inf.any().item()):
            inf_sign_diff = both_inf.clone()
            inf_sign_diff[both_inf] = torch.sign(act[both_inf].float()) != torch.sign(
                gold[both_inf].float()
            )
            inf_diff = inf_pos_diff | inf_sign_diff
        else:
            inf_diff = inf_pos_diff
        if bool(inf_diff.any().item()):
            first_coord = torch.nonzero(inf_diff, as_tuple=False)[0].tolist()
            act_inf_cnt = int(inf_act.sum().item())
            gold_inf_cnt = int(inf_gold.sum().item())
            return (
                f"Inf mismatch (actual_inf={{act_inf_cnt}}/{{act.numel()}}, "
                f"golden_inf={{gold_inf_cnt}}/{{gold.numel()}}, first_coord={{first_coord}})"
            )
    return ""


def _format_worst_element_diagnostic(actual: object, golden: object) -> str:
    """Summarize the single finite element with largest absolute difference."""
    import torch  # noqa: PLC0415

    if not isinstance(actual, torch.Tensor) or not isinstance(golden, torch.Tensor):
        return ""
    if actual.shape != golden.shape or actual.numel() == 0:
        return ""
    if actual.is_complex() or golden.is_complex():
        return ""
    with torch.no_grad():
        act = actual.detach().cpu().to(dtype=torch.float64)
        gold = golden.detach().cpu().to(dtype=torch.float64)
        diff = (act - gold).abs()
        finite_mask = torch.isfinite(diff) & torch.isfinite(act) & torch.isfinite(gold)
        if not bool(finite_mask.any().item()):
            return ""
        masked_diff = torch.where(finite_mask, diff, torch.full_like(diff, -1.0))
        flat_idx = int(torch.argmax(masked_diff.reshape(-1)).item())
        rem = flat_idx
        coord_rev: list[int] = []
        for dim in reversed(tuple(int(d) for d in act.shape)):
            coord_rev.append(rem % dim)
            rem //= dim
        coord = list(reversed(coord_rev))
        ref_val = float(gold.reshape(-1)[flat_idx].item())
        got_val = float(act.reshape(-1)[flat_idx].item())
        abs_diff = float(diff.reshape(-1)[flat_idx].item())
        rel_err = abs_diff / max(abs(ref_val), 1e-12)
        return (
            f"worst_element=(idx={{coord}}, ref={{ref_val:.6g}}, got={{got_val:.6g}}, "
            f"abs_diff={{abs_diff:.6e}}, rel_err={{rel_err:.6e}})"
        )


def compare_case_result(
    *,
    case_id: str,
    actual: object,
    golden: object,
    inputs: object = (),
    compute: bool = True,
    accuracy_mode: str | None = None,
    native_ref_operator=None,
    extra: dict[str, object] | None = None,
):
    """Compare candidate tensor against golden oracle using CANNBench thresholds."""
    _ = (case_id, inputs, compute, accuracy_mode)
    custom_thresholds = (extra or {{}}).get("precision_thresholds") or {{}}
    dtype_str = str(getattr(actual, "dtype", "float32")).removeprefix("torch.")
    res = _COMPARE_TENSORS(
        actual,
        golden,
        dtype=dtype_str,
        native_output=None,
        custom_thresholds=custom_thresholds,
    )
    if not res.passed and native_ref_operator is not None:
        native_val = native_ref_operator(None)
        res = _COMPARE_TENSORS(
            actual,
            golden,
            dtype=dtype_str,
            native_output=native_val,
            custom_thresholds=custom_thresholds,
        )
    msg = res.error_msg or (
        f"MERE={{res.mere:.6e}}, MARE={{res.mare:.6e}}, max_diff={{res.max_diff:.6e}}"
    )
    if not bool(res.passed):
        diag_parts: list[str] = []
        nonfinite_diag = _format_nonfinite_diagnostic(actual, golden)
        if nonfinite_diag and nonfinite_diag not in msg:
            diag_parts.append(nonfinite_diag)
        worst_diag = _format_worst_element_diagnostic(actual, golden)
        if worst_diag and worst_diag not in msg:
            diag_parts.append(worst_diag)
        if diag_parts:
            diag = "; ".join(diag_parts)
            msg = f"{{diag}}; raw={{msg}}" if msg else diag
    return SimpleNamespace(passed=bool(res.passed), message=msg)
'''


def ensure_npu_compare_shim_dir(
    target_dir: str | Path,
    cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None,
) -> Path:
    """Create a runtime npu_compare.py shim delegating to CANNBench compare_tensors."""
    repo = resolve_cannbench_repo(cannbench_repo=cannbench_repo, env_file=env_file)
    out_dir = Path(target_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    shim_path = out_dir / "npu_compare.py"
    shim_path.write_text(
        _NPU_COMPARE_SHIM_SOURCE.format(cann_repo=str(repo)),
        encoding="utf-8",
    )
    return out_dir


def run_l0_static_precheck(
    module_source: str,
    *,
    builder_name: str,
    builder_kwargs: Mapping[str, object],
    config_selector_name: str | None = None,
    pass_configs: dict[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None,
    dims: Mapping[str, object] | None = None,
    var_bindings: Mapping[str, int] | None = None,
    case_entry: Mapping[str, object] | None = None,
    proto: Mapping[str, object] | None = None,
    ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
) -> StageAResult:
    """Run Tier L0 static AST, ledger, and router inspection on Host CPU."""
    resolved_shapes = extract_ordered_tensor_shapes(
        case_entry=case_entry,
        dims=dims,
        proto=proto,
        ordered_tensor_shapes=ordered_tensor_shapes,
    )
    return run_stage_a_static(
        module_source,
        factory_name=builder_name,
        factory_kwargs=builder_kwargs,
        pass_configs=pass_configs,
        compile_flags=compile_flags,
        config_selector_name=config_selector_name,
        dims=dims,
        var_bindings=var_bindings,
        ordered_tensor_shapes=resolved_shapes,
    )


def _resolve_env_values(env_file: str | Path | None = None) -> dict[str, str]:
    resolved = (
        str(env_file)
        if env_file is not None
        else os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()
    )
    if resolved:
        return parse_simple_env_file(resolved)
    return {}


def _has_distinct_tail_dims(
    dims: Mapping[str, object] | None,
    tail_anchor_dims: Mapping[str, object] | None,
) -> bool:
    if not tail_anchor_dims:
        return False
    return dict(tail_anchor_dims) != dict(dims or {})


_EXCLUDED_TRACKA_SIG_KEYS: frozenset[str] = frozenset(
    {
        "pass_configs",
        "compile_flags",
        "builder_fn",
        "extra_builder_kwargs",
        "extra_combine_kwargs",
    }
)
_RUNTIME_ONLY_TRACKB_DIM_KEYS: frozenset[str] = frozenset(
    {
        "B",
        "S",
        "S_kv",
        "S1",
        "S2",
        "b_dim",
        "s_dim",
        "s_kv",
        "s1_dim",
        "s2_dim",
        "m_rows",
        "input_shape",
        "output_shape",
        "input_shapes",
        "output_shapes",
        "ordered_tensor_shapes",
        "attrs",
    }
)


def _json_safe_value(val: object) -> object:
    if isinstance(val, Mapping):
        return {str(k): _json_safe_value(v) for k, v in sorted(val.items(), key=lambda kv: str(kv[0]))}
    if isinstance(val, (list, tuple)):
        return [_json_safe_value(item) for item in val]
    if isinstance(val, set):
        return sorted(
            (_json_safe_value(item) for item in val),
            key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val
    return str(val)


def _filter_eff_kwargs_for_builder_source(
    module_source: str,
    builder_name: str,
    eff_kwargs: Mapping[str, object],
) -> dict[str, object]:
    try:
        tree = ast.parse(module_source)
    except SyntaxError:
        return dict(eff_kwargs)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == builder_name:
            if node.args.kwarg is not None:
                return dict(eff_kwargs)
            params = [
                arg.arg
                for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
            ]
            filtered = {k: v for k, v in eff_kwargs.items() if k in params}
            lowered = {str(k).lower(): v for k, v in eff_kwargs.items()}
            for pname in params:
                if pname not in filtered and pname.lower() in lowered:
                    filtered[pname] = lowered[pname.lower()]
            return filtered
    return dict(eff_kwargs)


def _filter_eff_kwargs_for_entry_source(
    module_source: str,
    builder_name: str,
    eff_kwargs: Mapping[str, object],
) -> dict[str, object]:
    try:
        tree = ast.parse(module_source)
    except SyntaxError:
        return dict(eff_kwargs)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_get_kernel":
            pos_args = [*node.args.posonlyargs, *node.args.args]
            n_req_pos = len(pos_args) - len(node.args.defaults)
            req_names = [arg.arg for arg in pos_args[:n_req_pos]]
            for kw_arg, kw_def in zip(node.args.kwonlyargs, node.args.kw_defaults):
                if kw_def is None:
                    req_names.append(kw_arg.arg)
            lowered = {str(k).lower(): v for k, v in eff_kwargs.items()}
            req_check = [r for r in req_names if r not in _EXCLUDED_TRACKA_SIG_KEYS]
            if all(r in eff_kwargs or r.lower() in lowered for r in req_check):
                if node.args.kwarg is not None:
                    return dict(eff_kwargs)
                params = [
                    arg.arg
                    for arg in (*pos_args, *node.args.kwonlyargs)
                    if arg.arg not in _EXCLUDED_TRACKA_SIG_KEYS
                ]
                filtered = {k: v for k, v in eff_kwargs.items() if k in params}
                for pname in params:
                    if pname not in filtered and pname.lower() in lowered:
                        filtered[pname] = lowered[pname.lower()]
                return filtered
            break
    return _filter_eff_kwargs_for_builder_source(module_source, builder_name, eff_kwargs)


def _canonical_tracka_spec_signature(
    builder_name: str,
    kwargs: Mapping[str, object] | None,
    *,
    spec_key: str = "",
) -> str:
    if spec_key:
        return str(spec_key)
    filtered = {
        str(k): _json_safe_value(v)
        for k, v in sorted((kwargs or {}).items(), key=lambda kv: str(kv[0]))
        if str(k) not in _EXCLUDED_TRACKA_SIG_KEYS
    }
    return json.dumps(
        {"builder_name": str(builder_name), "kwargs": filtered},
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_trackb_spec_signature(
    dims: Mapping[str, object] | None,
    *,
    regime_tag: str = "",
    spec_key: str = "",
) -> str:
    if spec_key:
        return str(spec_key)
    filtered = {
        str(k): _json_safe_value(v)
        for k, v in sorted((dims or {}).items(), key=lambda kv: str(kv[0]))
        if str(k) not in _RUNTIME_ONLY_TRACKB_DIM_KEYS
    }
    return json.dumps(
        {"regime_tag": str(regime_tag), "spec": filtered},
        sort_keys=True,
        separators=(",", ":"),
    )


def _resolve_l1_cache_store(
    env_vals: Mapping[str, str],
    tl_cache_dir: str | Path | None,
    artifact_store: ArtifactStore | None,
    source_root: Path | None,
) -> tuple[Path, ArtifactStore]:
    if artifact_store is not None:
        return artifact_store.root, artifact_store
    if tl_cache_dir is not None and str(tl_cache_dir).strip():
        resolved = Path(tl_cache_dir).expanduser().resolve()
        return resolved, ArtifactStore(resolved)
    has_env_dir = any(
        str(env_vals.get(k, "") or os.environ.get(k, "")).strip()
        for k in (
            "TILELANG_CACHE_DIR",
            "TF_SURROGATE_TL_CACHE_DIR",
            "TILEFOUNDRY_NPU_ARTIFACT_STORE",
            "TILEFOUNDRY_NPU_WORK_ROOT",
        )
    )
    if has_env_dir:
        resolved = resolve_unified_tl_cache_dir(
            None,
            env_vals=env_vals,
            source_root=source_root,
        )
        return resolved, ArtifactStore(resolved)
    resolved = _default_surrogate_tl_cache_dir().resolve()
    return resolved, ArtifactStore(resolved)


_STAGE_B_DIAG_CACHE_DIR = "stage_b_diag_v1"


def _load_stage_b_diag_cache(
    resolved_tl_cache: Path,
    primary_t_key: str,
) -> dict[str, object] | None:
    diag_path = resolved_tl_cache / _STAGE_B_DIAG_CACHE_DIR / f"{primary_t_key}.json"
    if not diag_path.is_file():
        return None
    try:
        payload = json.loads(diag_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, Mapping) or payload.get("version") != 1:
        return None
    raw_summary = payload.get("summary")
    summary: BishengResourceSummary | None = None
    if isinstance(raw_summary, Mapping):
        try:
            summary = BishengResourceSummary.from_dict(raw_summary)
        except (KeyError, TypeError, ValueError):
            return None
    raw_profile = payload.get("ma_profile")
    ma_profile = dict(raw_profile) if isinstance(raw_profile, Mapping) else None
    return {
        "ok": True,
        "timeout": False,
        "rc": 0,
        "stdout": "",
        "stderr": "",
        "combined_out": "",
        "lowered_dump": "",
        "spill_bytes": int(payload.get("spill_bytes") or 0),
        "vloopv2_hit": bool(payload.get("vloopv2_hit")),
        "summary": summary,
        "ma_profile": ma_profile,
    }


def _save_stage_b_diag_cache(
    resolved_tl_cache: Path,
    primary_t_key: str,
    *,
    spill_bytes: int,
    vloopv2_hit: bool,
    summary: BishengResourceSummary | None,
    ma_profile: Mapping[str, object] | None,
) -> None:
    cache_dir = resolved_tl_cache / _STAGE_B_DIAG_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    diag_path = cache_dir / f"{primary_t_key}.json"
    payload = {
        "version": 1,
        "spill_bytes": int(spill_bytes),
        "vloopv2_hit": bool(vloopv2_hit),
        "summary": summary.to_dict() if summary is not None else None,
        "ma_profile": dict(ma_profile) if isinstance(ma_profile, Mapping) else None,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(dir=str(cache_dir), prefix=f".{primary_t_key}.")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(encoded)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, diag_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _record_l1_artifacts_and_manifest(
    *,
    resolved_tl_cache: Path,
    resolved_store: ArtifactStore,
    task_entries: Sequence[tuple[str, str, Mapping[str, object]]],
    binary_paths: Sequence[Path],
    builder_name: str,
    closure_src: str,
    cand_sha256: str,
    tl_lib_stamp: str,
    norm_cf: Sequence[str],
    plan_key: str,
    per_task_records: Sequence[
        tuple[str, str, str, str, str, Sequence[str], Sequence[Path]]
    ] = (),
    plan_index_entries: Mapping[str, Sequence[str]] | None = None,
    cached_manifest_tasks: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[str, tuple[str, ...]]:
    recorded_aid = ""
    recorded_bins: tuple[str, ...] = ()
    task_keys: list[str] = []
    manifest_path = resolved_tl_cache / _PREWARM_MANIFEST_FILENAME
    lock_path = resolved_tl_cache / ".prewarm_manifest_v1.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            latest = _load_prewarm_manifest(manifest_path)
            merged_tasks = dict(latest["tasks"])
            merged_plan = dict(latest["plan_index"])
            if per_task_records:
                for (
                    t_key,
                    t_sig,
                    t_builder,
                    t_csha,
                    t_csrc,
                    t_cf,
                    t_bins,
                ) in per_task_records:
                    valid_bins = [Path(p) for p in t_bins if is_verified_binary_file(Path(p))]
                    if valid_bins:
                        rec = resolved_store.record_verified_artifact(
                            t_key,
                            binary_paths=valid_bins,
                            backend_kind="tilelang",
                            flags=" ".join(t_cf),
                            spec_key=t_sig,
                            compiler_stamp=tl_lib_stamp,
                            closure_src=t_csrc,
                            copy_binaries=True,
                        )
                        if not recorded_aid:
                            recorded_aid = rec.artifact_id
                            recorded_bins = rec.binary_paths
                        merged_tasks[t_key] = {
                            "builder_name": str(t_builder),
                            "cand_sha256": t_csha,
                            "tilelang_lib_stamp": tl_lib_stamp,
                        }
                    else:
                        existing = resolved_store.get_verified_artifact(t_key)
                        if existing is not None:
                            if not recorded_aid:
                                recorded_aid = existing.artifact_id
                                recorded_bins = existing.binary_paths
                            merged_tasks[t_key] = {
                                "builder_name": str(t_builder),
                                "cand_sha256": t_csha,
                                "tilelang_lib_stamp": tl_lib_stamp,
                            }
            else:
                for t_key, v_spec_sig, _v_dims in task_entries:
                    task_keys.append(t_key)
                    if binary_paths:
                        rec = resolved_store.record_verified_artifact(
                            t_key,
                            binary_paths=binary_paths,
                            backend_kind="tilelang",
                            flags=" ".join(norm_cf),
                            spec_key=v_spec_sig,
                            compiler_stamp=tl_lib_stamp,
                            closure_src=closure_src,
                            copy_binaries=True,
                        )
                        if not recorded_aid:
                            recorded_aid = rec.artifact_id
                            recorded_bins = rec.binary_paths
                        merged_tasks[t_key] = {
                            "builder_name": str(builder_name),
                            "cand_sha256": cand_sha256,
                            "tilelang_lib_stamp": tl_lib_stamp,
                        }
                    else:
                        existing = resolved_store.get_verified_artifact(t_key)
                        if existing is not None:
                            if not recorded_aid:
                                recorded_aid = existing.artifact_id
                                recorded_bins = existing.binary_paths
                            merged_tasks[t_key] = {
                                "builder_name": str(builder_name),
                                "cand_sha256": cand_sha256,
                                "tilelang_lib_stamp": tl_lib_stamp,
                            }
                if task_keys and all(tk in merged_tasks for tk in task_keys):
                    merged_plan[plan_key] = list(task_keys)
            if cached_manifest_tasks:
                for c_tkey, c_meta in cached_manifest_tasks.items():
                    c_key_str = str(c_tkey)
                    merged_tasks[c_key_str] = dict(c_meta)
                    if not recorded_aid:
                        existing = resolved_store.get_verified_artifact(c_key_str)
                        if existing is not None:
                            recorded_aid = existing.artifact_id
                            recorded_bins = existing.binary_paths
                if task_entries:
                    var_keys = [str(tk) for tk, _sig, _vd in task_entries]
                    if var_keys and all(tk in merged_tasks for tk in var_keys):
                        merged_plan[plan_key] = var_keys
            if plan_index_entries:
                for p_key, t_keys in plan_index_entries.items():
                    key_list = [str(tk) for tk in t_keys]
                    if key_list and all(tk in merged_tasks for tk in key_list):
                        merged_plan[str(p_key)] = key_list
            _save_prewarm_manifest(manifest_path, merged_tasks, merged_plan)
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
    return recorded_aid, recorded_bins


def record_trackb_stage_b_artifacts_and_plan(
    *,
    module_source: str,
    operator_name: str,
    anchor_entries: Sequence[tuple[int, str, str, Sequence[str], Sequence[Path]]],
    tl_cache_dir: str | Path,
    cases_yaml_sha256: str = "",
    env_vals: Mapping[str, str] | None = None,
) -> int:
    """Record verified Track-B Stage-B artifacts and prewarm plan index entries."""
    resolved_tl_cache = Path(tl_cache_dir).expanduser().resolve()
    resolved_tl_cache.mkdir(parents=True, exist_ok=True)
    artifact_store = ArtifactStore(resolved_tl_cache)
    tl_lib_stamp = _compute_tilelang_lib_stamp(dict(env_vals) if env_vals is not None else {})
    manifest_path = resolved_tl_cache / _PREWARM_MANIFEST_FILENAME
    lock_path = resolved_tl_cache / ".prewarm_manifest_v1.lock"
    recorded_plans: set[str] = set()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            latest = _load_prewarm_manifest(manifest_path)
            merged_tasks = dict(latest["tasks"])
            merged_plan = dict(latest["plan_index"])
            for case_num, regime_tag, norm_cpp_unit, compile_flags, binary_paths in anchor_entries:
                norm_unit = str(norm_cpp_unit or "").strip()
                cf_tuple = tuple(str(f) for f in compile_flags if str(f).strip())
                valid_bins = [Path(p) for p in binary_paths if is_verified_binary_file(Path(p))]
                if not norm_unit or not valid_bins:
                    continue
                rtag = str(regime_tag or "")
                c_src = extract_reachable_closure_source(module_source, regime_tag=rtag)
                c_id = compute_compile_task_key(
                    normalized_kernel_unit_src=c_src,
                    cann_target="dav-3510",
                    pass_configs=None,
                    compile_flags=(),
                    static_spec_signature=f"{operator_name}:{rtag}:{cases_yaml_sha256}",
                    compiler_stamp=tl_lib_stamp,
                )
                p_key = _build_prewarm_plan_key(
                    operator_name=operator_name,
                    cand_sha256=c_id,
                    case_num=int(case_num),
                    cases_yaml_sha256=cases_yaml_sha256,
                    tilelang_lib_stamp=tl_lib_stamp,
                )
                t_key = compute_compile_task_key(
                    normalized_kernel_unit_src=norm_unit,
                    cann_target="dav-3510",
                    pass_configs=None,
                    compile_flags=cf_tuple,
                    static_spec_signature="",
                    compiler_stamp="",
                )
                artifact_store.record_verified_artifact(
                    t_key,
                    binary_paths=valid_bins,
                    backend_kind="ascendc_trackb",
                    flags=" ".join(cf_tuple),
                    spec_key="",
                    compiler_stamp="",
                    closure_src=norm_unit,
                    copy_binaries=True,
                )
                merged_tasks[t_key] = {
                    "builder_name": "_trackb_tu",
                    "cand_sha256": c_id,
                    "tilelang_lib_stamp": tl_lib_stamp,
                }
                merged_plan[p_key] = [t_key]
                recorded_plans.add(p_key)
            _save_prewarm_manifest(manifest_path, merged_tasks, merged_plan)
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
    return len(recorded_plans)


_L1_ANCHOR_PLAN_TEMPLATE = """import importlib.util
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, {ascend_target_dir!r})
from tilefoundry_ascend.agent.campaign.anchor_specs import (
    discover_module_trackb_symbols,
    get_default_regime_specs,
    get_operator_entrypoint,
    infer_module_runtime_backend,
    resolve_regime_for_case,
)
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    _HIR_DIM_ALIASES,
    _filter_callable_kwargs,
    resolve_regime_config,
)

def _json_safe(val):
    if isinstance(val, dict):
        return {{str(k): _json_safe(v) for k, v in sorted(val.items())}}
    if isinstance(val, (list, tuple)):
        return [_json_safe(x) for x in val]
    if isinstance(val, set):
        return sorted((_json_safe(x) for x in val), key=lambda x: json.dumps(x, sort_keys=True))
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val
    return str(val)

def _run_sel(mod, sel_name, pool):
    raw_bk, raw_ck, pc_safe, cf_list = {{}}, {{}}, {{}}, []
    if sel_name and callable(getattr(mod, sel_name, None)):
        sel_fn = getattr(mod, sel_name)
        sel_out = sel_fn(**_filter_callable_kwargs(sel_fn, pool))
        if isinstance(sel_out, dict):
            raw_pc = sel_out.get("pass_configs")
            if isinstance(raw_pc, dict):
                for pk, pv in raw_pc.items():
                    if pk == "tl.device_compile_flags":
                        items = pv if isinstance(pv, (list, tuple)) else [pv]
                        cf_list.extend(str(f) for f in items if str(f).strip())
                    else:
                        pc_safe[str(pk)] = pv
            raw_cf = sel_out.get("compile_flags") or ()
            if isinstance(raw_cf, str):
                cf_list.extend(raw_cf.split())
            else:
                cf_list.extend(str(f) for f in raw_cf if str(f).strip())
            if isinstance(sel_out.get("builder_kwargs"), dict):
                raw_bk = dict(sel_out.get("builder_kwargs"))
            if isinstance(sel_out.get("combine_kwargs"), dict):
                raw_ck = dict(sel_out.get("combine_kwargs"))
            for sk, sv in sel_out.items():
                if sk not in ("pass_configs", "compile_flags", "builder_kwargs", "combine_kwargs"):
                    pool[str(sk)] = sv
    return raw_bk, raw_ck, pc_safe, cf_list

op_name = {op_name!r}
mod_path = {mod_path!r}
anchor_specs = json.loads({anchor_specs_json!r})
def_builder = {def_builder!r}
def_bk = json.loads({def_bk_json!r})
def_sel = {def_sel!r}
def_pc = json.loads({def_pc_json!r})
def_cf = json.loads({def_cf_json!r})

spec = importlib.util.spec_from_file_location("_tf_l1_plan", mod_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load module spec from {{mod_path}}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

regime_specs = []
if op_name:
    try:
        regime_specs = list(get_default_regime_specs(op_name))
    except Exception:
        regime_specs = []

get_kernel_fn = getattr(mod, "_get_kernel", None)
combine_fn = getattr(mod, "_get_combine_kernel", None)
render_tb_fn = getattr(mod, "_render_trackb_tu", None)
mod_src = Path(mod_path).read_text(encoding="utf-8")
entrypoint = ""
if op_name:
    try:
        entrypoint = get_operator_entrypoint(op_name)
    except Exception:
        entrypoint = ""
mod_backend, mod_tb_syms = infer_module_runtime_backend(mod_src, entrypoint)
tasks = []

for case_num, rtag, case_dims in anchor_specs:
    regime = None
    if regime_specs:
        try:
            regime = resolve_regime_for_case(op_name, case_dims, regime_specs)
        except Exception:
            for cand_r in regime_specs:
                if getattr(cand_r, "regime_tag", "") == rtag:
                    regime = cand_r
                    break
    rt_backend = str(getattr(regime, "runtime_backend", "") or "") if regime is not None else ""
    tb_syms = list(
        (getattr(regime, "trackb_symbols", ()) if regime is not None else ())
        or mod_tb_syms
        or ()
    )
    is_pure_tb = rt_backend in ("ascendc_trackb", "ascendc") or mod_backend == "ascendc_trackb"
    has_tb_cpp = bool(callable(render_tb_fn) or tb_syms or discover_module_trackb_symbols(mod_src))
    is_hybrid_tb = (rt_backend == "hybrid_trackb" or mod_backend == "hybrid_trackb") and has_tb_cpp
    resolved = None
    if regime is not None:
        pool = dict(getattr(regime, "sample_builder_kwargs", {{}}))
        for hk, hv in getattr(regime, "representative_hir_dims", {{}}).items():
            pool.setdefault(_HIR_DIM_ALIASES.get(str(hk), str(hk)), hv)
        for dk, dv in case_dims.items():
            pool[_HIR_DIM_ALIASES.get(str(dk), str(dk))] = dv
        sel_name = getattr(regime, "config_selector_fn", None)
        raw_bk, raw_ck, pc_safe, cf_list = _run_sel(mod, sel_name, pool)
        b_name = def_builder
        b_fn = getattr(mod, b_name, None)
        b_kwargs_base = dict(def_bk)
        try:
            resolved = resolve_regime_config(mod_path, regime, dims=case_dims)
            b_name = resolved.builder_name
            b_fn = getattr(mod, b_name, None)
            pool.update(resolved.builder_kwargs)
            if isinstance(resolved.pass_configs, dict):
                pc_safe = dict(resolved.pass_configs)
            if resolved.compile_flags:
                cf_list = [str(f) for f in resolved.compile_flags]
            b_kwargs_base = dict(resolved.builder_kwargs)
        except Exception:
            if not (is_pure_tb or is_hybrid_tb):
                raise
    else:
        pool = dict(def_bk)
        for dk, dv in case_dims.items():
            pool[_HIR_DIM_ALIASES.get(str(dk), str(dk))] = dv
        sel_name = def_sel
        raw_bk, raw_ck, pc_safe, cf_list = _run_sel(mod, sel_name, pool)
        for pk, pv in def_pc.items():
            pc_safe[str(pk)] = pv
        for fl in def_cf:
            if fl and fl not in cf_list:
                cf_list.append(str(fl))
        b_name = def_builder
        b_fn = getattr(mod, b_name, None)
        b_kwargs_base = dict(def_bk)

    tb_only_builder = bool(
        resolved is not None and resolved.builder_name == sel_name and has_tb_cpp
    )
    if is_pure_tb or (is_hybrid_tb and (b_fn is None or tb_only_builder)):
        continue

    if callable(b_fn):
        pool["builder_fn"] = b_fn
    pool["pass_configs"] = pc_safe
    pool["compile_flags"] = tuple(cf_list)
    pool["extra_builder_kwargs"] = dict(raw_bk) if raw_bk else None
    pool["extra_combine_kwargs"] = dict(raw_ck) if raw_ck else None

    used_gk = False
    if callable(get_kernel_fn) and callable(b_fn):
        gk_sig = inspect.signature(get_kernel_fn)
        var_kinds = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        eff_sel = sel_name or (getattr(resolved, "selector_name", "") if resolved is not None else "")
        for src_fn in (
            getattr(mod, entrypoint, None) if entrypoint else None,
            getattr(mod, "kernel", None),
            getattr(mod, eff_sel, None) if eff_sel else None,
            b_fn,
        ):
            if callable(src_fn):
                try:
                    for p in inspect.signature(src_fn).parameters.values():
                        if (
                            p.default is not inspect.Parameter.empty
                            and p.kind not in var_kinds
                            and p.name not in pool
                        ):
                            pool[p.name] = p.default
                except (TypeError, ValueError):
                    pass
        has_req = all(
            p.name in pool
            for p in gk_sig.parameters.values()
            if p.default is inspect.Parameter.empty and p.kind not in var_kinds
        )
        if has_req:
            gk_kw = _filter_callable_kwargs(get_kernel_fn, pool)
            serial_gk = {{
                str(k): _json_safe(v)
                for k, v in sorted(gk_kw.items())
                if k != "builder_fn"
            }}
            fkw_out = dict(b_kwargs_base)
            for k, v in serial_gk.items():
                if k not in (
                    "pass_configs",
                    "compile_flags",
                    "extra_builder_kwargs",
                    "extra_combine_kwargs",
                ) and k not in fkw_out:
                    fkw_out[k] = v
            tasks.append({{
                "case_num": int(case_num),
                "regime_tag": str(rtag),
                "mode": "_get_kernel",
                "builder_name": str(b_name),
                "selector_name": sel_name or "",
                "factory_kwargs": _json_safe(fkw_out),
                "dims": _json_safe(case_dims),
                "kwargs": serial_gk,
                "pass_configs": _json_safe(pc_safe),
                "compile_flags": list(cf_list),
            }})
            splits = int(pool.get("num_kv_splits", 1) or 1)
            if splits > 1 and callable(combine_fn):
                comb_kw = _filter_callable_kwargs(combine_fn, pool)
                serial_comb = {{
                    str(k): _json_safe(v) for k, v in sorted(comb_kw.items())
                }}
                tasks.append({{
                    "case_num": int(case_num),
                    "regime_tag": str(rtag),
                    "mode": "_get_combine_kernel",
                    "builder_name": "_get_combine_kernel",
                    "selector_name": "",
                    "factory_kwargs": serial_comb,
                    "dims": {{}},
                    "kwargs": serial_comb,
                    "pass_configs": _json_safe(pc_safe),
                    "compile_flags": list(cf_list),
                }})
            used_gk = True
    if not used_gk and callable(b_fn):
        gen_pool = dict(b_kwargs_base)
        gen_pool.update(raw_bk)
        b_kw = _filter_callable_kwargs(b_fn, gen_pool)
        serial_bk = {{str(k): _json_safe(v) for k, v in sorted(b_kw.items())}}
        tasks.append({{
            "case_num": int(case_num),
            "regime_tag": str(rtag),
            "mode": "generic",
            "builder_name": str(b_name),
            "selector_name": sel_name or "",
            "factory_kwargs": _json_safe(b_kwargs_base),
            "dims": _json_safe(case_dims),
            "kwargs": serial_bk,
            "pass_configs": _json_safe(pc_safe),
            "compile_flags": list(cf_list),
        }})

print(json.dumps({{"tasks": tasks}}), flush=True)
"""


def _resolve_worker_cache_bins(
    emit_path: Path | None,
    worker_cache: Path,
    resolved_tl_cache: Path,
    bin_rels: Sequence[Path],
) -> list[Path]:
    resolved_bins: list[Path] = []
    if emit_path is not None and emit_path.is_file():
        try:
            raw_list = json.loads(emit_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raw_list = []
        if isinstance(raw_list, Sequence) and not isinstance(raw_list, (str, bytes)):
            for raw_p in raw_list:
                p_obj = Path(str(raw_p))
                try:
                    rel = p_obj.relative_to(worker_cache)
                    cand = resolved_tl_cache / rel
                except ValueError:
                    cand = p_obj
                if is_verified_binary_file(cand):
                    resolved_bins.append(cand)
    for rel in bin_rels:
        cand = resolved_tl_cache / rel
        if is_verified_binary_file(cand):
            resolved_bins.append(cand)
    return list(dict.fromkeys(resolved_bins))


def run_l1_compile_check(
    module_source: str,
    *,
    builder_name: str,
    builder_kwargs: Mapping[str, object],
    config_selector_name: str | None = None,
    pass_configs: Mapping[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None,
    dims: Mapping[str, object] | None = None,
    tail_anchor_dims: Mapping[str, object] | None = None,
    static_spec_dims_list: Sequence[Mapping[str, object]] | None = None,
    source_root: str | Path | None = None,
    env_file: str | Path | None = None,
    timeout_sec: int = 120,
    artifact_store: ArtifactStore | None = None,
    artifact_id: str = "",
    regime_tag: str = "",
    spec_key: str = "",
    tl_cache_dir: str | Path | None = None,
    anchor_case_specs: Sequence[tuple[int, str, Mapping[str, object]]] | None = None,
    operator_name: str = "",
    cases_yaml_sha256: str = "",
) -> StageBResult:
    """Run Tier L1 TileLang + bisheng device compilation on Host CPU without NPU locks."""
    env_vals = _resolve_env_values(env_file)
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    resolved_tl_cache, resolved_store = _resolve_l1_cache_store(
        env_vals, tl_cache_dir, artifact_store, root
    )
    resolved_tl_cache.mkdir(parents=True, exist_ok=True)
    tl_lib_stamp = _compute_tilelang_lib_stamp(env_vals)
    norm_pc, norm_cf = _merge_pass_configs_and_flags(None, None, pass_configs, compile_flags)
    if regime_tag:
        closure_src = extract_reachable_closure_source(module_source, regime_tag=regime_tag)
    else:
        syms = tuple(s for s in (config_selector_name, builder_name, "_get_kernel") if s)
        closure_src = (
            extract_reachable_closure_source(module_source, active_symbols=syms)
            if syms
            else module_source
        )
    cand_sha = hashlib.sha256(closure_src.encode("utf-8")).hexdigest()

    primary_dims: dict[str, object] = dict(dims) if dims else {}
    all_dims_variants: list[dict[str, object]] = [primary_dims]
    seen_dims_json: set[str] = {json.dumps(primary_dims, sort_keys=True, default=str)}
    has_tail = _has_distinct_tail_dims(dims, tail_anchor_dims)
    if has_tail and tail_anchor_dims is not None:
        tail_dict = dict(tail_anchor_dims)
        tail_json = json.dumps(tail_dict, sort_keys=True, default=str)
        seen_dims_json.add(tail_json)
        all_dims_variants.append(tail_dict)
    if static_spec_dims_list:
        for raw_spec_dims in static_spec_dims_list:
            spec_dict = dict(raw_spec_dims)
            spec_json = json.dumps(spec_dict, sort_keys=True, default=str)
            if spec_json not in seen_dims_json:
                seen_dims_json.add(spec_json)
                all_dims_variants.append(spec_dict)

    task_entries: list[tuple[str, str, dict[str, object]]] = []
    for idx, v_dims in enumerate(all_dims_variants):
        eff_kw = _merge_dims_into_kwargs(builder_kwargs, v_dims)
        filtered_kw = _filter_eff_kwargs_for_entry_source(module_source, builder_name, eff_kw)
        v_spec_sig = _canonical_tracka_spec_signature(
            builder_name,
            filtered_kw,
            spec_key=spec_key if (idx == 0 and len(all_dims_variants) == 1) else "",
        )
        t_key = (
            artifact_id
            if (idx == 0 and artifact_id)
            else compute_compile_task_key(
                normalized_kernel_unit_src=closure_src,
                cann_target="dav-3510",
                pass_configs=norm_pc,
                compile_flags=norm_cf,
                static_spec_signature=v_spec_sig,
                compiler_stamp=tl_lib_stamp,
            )
        )
        task_entries.append((t_key, v_spec_sig, v_dims))

    primary_t_key = task_entries[0][0]
    plan_key = f"l1:{builder_name}:{primary_t_key}"

    python_bin = (
        env_vals.get("TILEFOUNDRY_NPU_PYTHON")
        or os.environ.get("TILEFOUNDRY_NPU_PYTHON")
        or sys.executable
    )
    cann_setup = (
        env_vals.get("TILEFOUNDRY_NPU_CANN_SETUP")
        or os.environ.get("TILEFOUNDRY_NPU_CANN_SETUP")
        or ""
    ).strip()
    tl_source = (
        env_vals.get("TILEFOUNDRY_NPU_TILELANG_SOURCE")
        or os.environ.get("TILEFOUNDRY_NPU_TILELANG_SOURCE")
        or ""
    ).strip()

    if not cann_setup and not tl_source:
        primary_t_key, primary_spec_sig, primary_v_dims = task_entries[0]
        primary_res = run_stage_b_compile(
            module_source,
            factory_name=builder_name,
            factory_kwargs=builder_kwargs,
            config_selector_name=config_selector_name,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
            dims=primary_v_dims or None,
            python_bin=python_bin,
            timeout_sec=timeout_sec,
            artifact_store=resolved_store,
            artifact_id=primary_t_key,
            regime_tag=regime_tag,
            spec_key=primary_spec_sig,
        )
        if not primary_res.passed:
            return primary_res
        total_elapsed = primary_res.elapsed_sec
        all_bins: list[Path] = [Path(p) for p in primary_res.binary_paths if Path(p).is_file()]
        for idx, (t_key, v_spec_sig, v_dims) in enumerate(task_entries[1:], start=1):
            extra_res = run_stage_b_compile(
                module_source,
                factory_name=builder_name,
                factory_kwargs=builder_kwargs,
                config_selector_name=config_selector_name,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
                dims=v_dims,
                python_bin=python_bin,
                timeout_sec=timeout_sec,
                artifact_store=resolved_store,
                artifact_id=t_key,
                regime_tag=regime_tag,
                spec_key=v_spec_sig,
            )
            total_elapsed += extra_res.elapsed_sec
            if not extra_res.passed:
                is_tail = has_tail and idx == 1 and dict(v_dims) == dict(tail_anchor_dims or {})
                default_code = (
                    "STAGE_B_TAIL_ANCHOR_FAILED"
                    if is_tail
                    else "STAGE_B_STATIC_SPEC_FAILED"
                )
                err_prefix = (
                    "stage_b_tail_anchor_failed"
                    if is_tail
                    else "stage_b_static_spec_failed"
                )
                eff_code = (
                    extra_res.fault_code
                    if extra_res.fault_code
                    in {
                        "VF_STACK_OVERFLOW",
                        "UB_SRAM_OVERFLOW",
                        "L1_SRAM_OVERFLOW",
                        "L0C_SRAM_OVERFLOW",
                    }
                    else default_code
                )
                return StageBResult(
                    passed=False,
                    error_message=f"{err_prefix}: {extra_res.error_message}",
                    elapsed_sec=total_elapsed,
                    spill_bytes=max(primary_res.spill_bytes, extra_res.spill_bytes),
                    vloopv2_hit=primary_res.vloopv2_hit and extra_res.vloopv2_hit,
                    compiler_summary=extra_res.compiler_summary or primary_res.compiler_summary,
                    microarch_profile=primary_res.microarch_profile,
                    full_diagnostic=extra_res.full_diagnostic,
                    fault_plane="STAGE_B_COMPILE",
                    fault_code=eff_code,
                    failed_tier="stage_b",
                )
            for p_str in extra_res.binary_paths:
                p_obj = Path(p_str)
                if p_obj.is_file() and p_obj not in all_bins:
                    all_bins.append(p_obj)
        _record_l1_artifacts_and_manifest(
            resolved_tl_cache=resolved_tl_cache,
            resolved_store=resolved_store,
            task_entries=task_entries,
            binary_paths=all_bins,
            builder_name=builder_name,
            closure_src=closure_src,
            cand_sha256=cand_sha,
            tl_lib_stamp=tl_lib_stamp,
            norm_cf=norm_cf,
            plan_key=plan_key,
        )
        return StageBResult(
            passed=True,
            elapsed_sec=total_elapsed,
            spill_bytes=primary_res.spill_bytes,
            vloopv2_hit=primary_res.vloopv2_hit,
            compiler_summary=primary_res.compiler_summary,
            microarch_profile=primary_res.microarch_profile,
            artifact_id=primary_res.artifact_id or primary_t_key,
            binary_paths=primary_res.binary_paths,
        )

    t0 = time.monotonic()
    pp_parts = [str(root), str(root / "src"), str(root / "targets" / "ascend")]
    if tl_source:
        pp_parts.insert(0, tl_source)
    if os.environ.get("PYTHONPATH"):
        pp_parts.append(os.environ["PYTHONPATH"])

    run_env = dict(os.environ)
    run_env["PYTHONPATH"] = os.pathsep.join(pp_parts)
    run_env["TILELANG_DISABLE_CACHE"] = "0"
    run_env["TILELANG_KERNEL_CACHE_USE_LIB_STAMP"] = "1"
    run_env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    run_env.setdefault("USE_CUDA", "0")

    need_cann_source = bool(
        cann_setup and Path(cann_setup).is_file() and shutil.which("bisheng") is None
    )

    def _build_l1_argv(script_path: Path) -> list[str]:
        if need_cann_source:
            bash_cmd = (
                f"set +u && source {shlex.quote(cann_setup)} >/dev/null 2>&1 "
                f"&& exec {shlex.quote(python_bin)} {shlex.quote(str(script_path))}"
            )
            return ["/bin/bash", "-c", bash_cmd]
        return [python_bin, str(script_path)]

    use_parallel_cache = bool(anchor_case_specs) or bool(static_spec_dims_list)

    with tempfile.TemporaryDirectory(prefix="tf_l1_compile_") as tmp_dir:
        tmp = Path(tmp_dir)
        cand_path = tmp / "candidate_dsl.py"
        cand_path.write_text(module_source, encoding="utf-8")

        if not use_parallel_cache:
            worker_cache = tmp / "tl_cache"
            seed_tl_cache_tree(resolved_tl_cache, worker_cache)
            pre_bins = snapshot_tl_cache_binaries(worker_cache)
            run_env["TILELANG_CACHE_DIR"] = str(worker_cache)
            kernel_dump_path = tmp / "lowered_kernel.cce"
            driver_path = tmp / "compile_driver.py"
            driver_src = build_stage_b_driver_source(
                cand_path,
                factory_name=builder_name,
                factory_kwargs=builder_kwargs,
                config_selector_name=config_selector_name,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
                dims=dims,
                enable_device_compile=True,
                kernel_dump_path=kernel_dump_path,
            )
            for extra_idx, extra_dims in enumerate(all_dims_variants[1:], start=1):
                extra_driver_src = build_stage_b_driver_source(
                    cand_path,
                    factory_name=builder_name,
                    factory_kwargs=builder_kwargs,
                    config_selector_name=config_selector_name,
                    pass_configs=pass_configs,
                    compile_flags=compile_flags,
                    dims=extra_dims,
                    enable_device_compile=True,
                    kernel_dump_path=None,
                )
                extra_dims_json = json.dumps(extra_dims, sort_keys=True, default=str)
                label = (
                    "stage_b_tail_anchor_failed"
                    if (has_tail and extra_idx == 1 and extra_dims == dict(tail_anchor_dims or {}))
                    else "stage_b_static_spec_failed"
                )
                driver_src = (
                    driver_src
                    + "\n"
                    + f"_tf_extra_driver_src_{extra_idx} = {extra_driver_src!r}\n"
                    + f"_tf_extra_dims_json_{extra_idx} = {extra_dims_json!r}\n"
                    + "try:\n"
                    + f"    exec(compile(_tf_extra_driver_src_{extra_idx}, "
                    + f"'<stage_b_spec_{extra_idx}>', 'exec'), {{}})\n"
                    + f"except Exception as _tf_extra_exc_{extra_idx}:\n"
                    + "    import sys as _tf_sys\n"
                    + "    _tf_sys.stderr.write(\n"
                    + f"        f'{label} dims={{_tf_extra_dims_json_{extra_idx}}}: '\n"
                    + f"        f'{{type(_tf_extra_exc_{extra_idx}).__name__}}: "
                    + f"{{_tf_extra_exc_{extra_idx}}}\\n'\n"
                    + "    )\n"
                    + "    raise\n"
                )
            driver_path.write_text(driver_src, encoding="utf-8")
            try:
                proc = subprocess.run(
                    _build_l1_argv(driver_path),
                    text=True,
                    capture_output=True,
                    cwd=tmp_dir,
                    env=run_env,
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired as exc:
                captured = "".join(
                    part for part in (exc.stdout, exc.stderr) if isinstance(part, str)
                )
                return StageBResult(
                    passed=False,
                    error_message=f"l1_compile_timeout:{timeout_sec}s",
                    elapsed_sec=time.monotonic() - t0,
                    full_diagnostic=captured,
                    fault_plane="STAGE_B_COMPILE",
                    fault_code="L1_COMPILE_TIMEOUT",
                    failed_tier="stage_b",
                )
            lowered_dump = (
                kernel_dump_path.read_text(encoding="utf-8", errors="ignore")
                if kernel_dump_path.is_file()
                else ""
            )
            combined_out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            spill_bytes, vloopv2_hit, summary = _parse_cce_res_usage_with_summary(
                combined_out,
                asm_or_dump_text=lowered_dump,
            )
            ma_profile = (
                extract_microarch_profile_from_cce(lowered_dump).to_dict()
                if lowered_dump.strip()
                else None
            )
            if proc.returncode != 0:
                raw_diagnostic = proc.stderr or proc.stdout or ""
                detail = _bounded_compile_diagnostic(raw_diagnostic)
                fault_code = _classify_stage_b_overflow_fault_code(
                    f"{raw_diagnostic}\n{lowered_dump}",
                    summary,
                    "L1_COMPILE_FAILED",
                )
                return StageBResult(
                    passed=False,
                    error_message=f"l1_compile_failed (rc={proc.returncode}): {detail}",
                    elapsed_sec=time.monotonic() - t0,
                    spill_bytes=spill_bytes,
                    vloopv2_hit=vloopv2_hit,
                    compiler_summary=summary,
                    microarch_profile=ma_profile,
                    full_diagnostic=raw_diagnostic,
                    fault_plane="STAGE_B_COMPILE",
                    fault_code=fault_code,
                    failed_tier="stage_b",
                )
            if _has_vf_stack_overflow_signal(combined_out, summary, norm_cf, norm_pc):
                return StageBResult(
                    passed=False,
                    error_message=f"l1_vf_stack_overflow: stack_bytes={summary.stack_bytes}",
                    elapsed_sec=time.monotonic() - t0,
                    spill_bytes=spill_bytes,
                    vloopv2_hit=vloopv2_hit,
                    compiler_summary=summary,
                    microarch_profile=ma_profile,
                    full_diagnostic=combined_out,
                    fault_plane="STAGE_B_COMPILE",
                    fault_code="VF_STACK_OVERFLOW",
                    failed_tier="stage_b",
                )

            lock_path = resolved_tl_cache / ".prewarm_manifest_v1.lock"
            with lock_path.open("a+", encoding="utf-8") as lock_fh:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
                try:
                    merge_tl_cache_tree(worker_cache, resolved_tl_cache)
                finally:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            post_new_rels = collect_new_tl_cache_binaries(worker_cache, pre_bins)
            cache_bins = [
                resolved_tl_cache / rel
                for rel in post_new_rels
                if is_verified_binary_file(resolved_tl_cache / rel)
            ]
            tmp_bins = [
                p
                for p in sorted(tmp.rglob("*"))
                if not p.is_relative_to(worker_cache) and is_verified_binary_file(p)
            ]
            built_bins = [*cache_bins, *tmp_bins]
            recorded_aid, recorded_bins = _record_l1_artifacts_and_manifest(
                resolved_tl_cache=resolved_tl_cache,
                resolved_store=resolved_store,
                task_entries=task_entries,
                binary_paths=built_bins,
                builder_name=builder_name,
                closure_src=closure_src,
                cand_sha256=cand_sha,
                tl_lib_stamp=tl_lib_stamp,
                norm_cf=norm_cf,
                plan_key=plan_key,
            )
            return StageBResult(
                passed=True,
                elapsed_sec=time.monotonic() - t0,
                spill_bytes=spill_bytes,
                vloopv2_hit=vloopv2_hit,
                compiler_summary=summary,
                microarch_profile=ma_profile,
                artifact_id=recorded_aid or primary_t_key,
                binary_paths=recorded_bins,
            )

        cached_diag = (
            _load_stage_b_diag_cache(resolved_tl_cache, primary_t_key)
            if resolved_store.has_verified_artifact(primary_t_key)
            else None
        )

        def _run_diag_worker() -> dict[str, object]:
            d_dir = tmp / "diag_worker"
            d_cache = d_dir / "tl_cache"
            d_dir.mkdir(parents=True, exist_ok=True)
            seed_tl_cache_tree(resolved_tl_cache, d_cache)
            kdump_path = d_dir / "lowered_kernel.cce"
            drv_path = d_dir / "diag_driver.py"
            drv_src = build_stage_b_driver_source(
                cand_path,
                factory_name=builder_name,
                factory_kwargs=builder_kwargs,
                config_selector_name=config_selector_name,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
                dims=dims,
                enable_device_compile=True,
                kernel_dump_path=kdump_path,
                compile_for_cache=False,
            )
            drv_path.write_text(drv_src, encoding="utf-8")
            d_env = dict(run_env)
            d_env["TILELANG_CACHE_DIR"] = str(d_cache)
            try:
                d_proc = subprocess.run(
                    _build_l1_argv(drv_path),
                    text=True,
                    capture_output=True,
                    cwd=str(d_dir),
                    env=d_env,
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired as exc:
                captured = "".join(
                    part for part in (exc.stdout, exc.stderr) if isinstance(part, str)
                )
                return {"ok": False, "timeout": True, "diag": captured}
            lowered_dump = (
                kdump_path.read_text(encoding="utf-8", errors="ignore")
                if kdump_path.is_file()
                else ""
            )
            combined_out = (d_proc.stdout or "") + "\n" + (d_proc.stderr or "")
            spill_bytes, vloopv2_hit, summary = _parse_cce_res_usage_with_summary(
                combined_out,
                asm_or_dump_text=lowered_dump,
            )
            ma_profile = (
                extract_microarch_profile_from_cce(lowered_dump).to_dict()
                if lowered_dump.strip()
                else None
            )
            return {
                "ok": d_proc.returncode == 0,
                "timeout": False,
                "rc": d_proc.returncode,
                "stdout": d_proc.stdout or "",
                "stderr": d_proc.stderr or "",
                "combined_out": combined_out,
                "lowered_dump": lowered_dump,
                "spill_bytes": spill_bytes,
                "vloopv2_hit": vloopv2_hit,
                "summary": summary,
                "ma_profile": ma_profile,
            }

        planned_anchor_tasks: list[dict[str, object]] = []
        pre_diag_pool: ThreadPoolExecutor | None = None
        diag_fut: Future[dict[str, object]] | None = None
        diag_res: dict[str, object] = {}
        compile_results: dict[int, dict[str, object]] = {}
        try:
            if anchor_case_specs:
                if cached_diag is None:
                    pre_diag_pool = ThreadPoolExecutor(max_workers=1)
                    diag_fut = pre_diag_pool.submit(_run_diag_worker)
                anchor_list = [
                    [int(cnum), str(rtag), _json_safe_value(dict(cdims))]
                    for cnum, rtag, cdims in anchor_case_specs
                ]
                plan_script = tmp / "anchor_plan.py"
                plan_script.write_text(
                    _L1_ANCHOR_PLAN_TEMPLATE.format(
                        ascend_target_dir=str(root / "targets" / "ascend"),
                        op_name=str(operator_name or ""),
                        mod_path=str(cand_path),
                        anchor_specs_json=json.dumps(anchor_list),
                        def_builder=str(builder_name),
                        def_bk_json=json.dumps(_json_safe_value(dict(builder_kwargs))),
                        def_sel=str(config_selector_name or ""),
                        def_pc_json=json.dumps(_json_safe_value(dict(norm_pc))),
                        def_cf_json=json.dumps(list(norm_cf)),
                    ),
                    encoding="utf-8",
                )
                try:
                    plan_proc = subprocess.run(
                        _build_l1_argv(plan_script),
                        text=True,
                        capture_output=True,
                        cwd=tmp_dir,
                        env=run_env,
                        timeout=timeout_sec,
                    )
                except subprocess.TimeoutExpired as exc:
                    captured = "".join(
                        part for part in (exc.stdout, exc.stderr) if isinstance(part, str)
                    )
                    return StageBResult(
                        passed=False,
                        error_message=f"l1_compile_timeout:{timeout_sec}s",
                        elapsed_sec=time.monotonic() - t0,
                        full_diagnostic=captured,
                        fault_plane="STAGE_B_COMPILE",
                        fault_code="L1_COMPILE_TIMEOUT",
                        failed_tier="stage_b",
                    )
                if plan_proc.returncode != 0:
                    raw_diag = plan_proc.stderr or plan_proc.stdout or ""
                    detail = _bounded_compile_diagnostic(raw_diag)
                    fault_code = _classify_stage_b_overflow_fault_code(
                        raw_diag, None, "L1_COMPILE_FAILED"
                    )
                    return StageBResult(
                        passed=False,
                        error_message=f"l1_compile_failed (rc={plan_proc.returncode}): {detail}",
                        elapsed_sec=time.monotonic() - t0,
                        full_diagnostic=raw_diag,
                        fault_plane="STAGE_B_COMPILE",
                        fault_code=fault_code,
                        failed_tier="stage_b",
                    )
                last_line = (plan_proc.stdout or "").strip().splitlines()
                if last_line:
                    try:
                        parsed_plan = json.loads(last_line[-1])
                        if isinstance(parsed_plan, Mapping):
                            planned_anchor_tasks = [
                                dict(t)
                                for t in parsed_plan.get("tasks", ())
                                if isinstance(t, Mapping)
                            ]
                    except json.JSONDecodeError:
                        planned_anchor_tasks = []

            unique_compile_specs: list[dict[str, object]] = []
            dedup_idx_by_key: dict[str, int] = {}
            task_record_bindings: list[
                tuple[int | None, str, str, str, str, str, tuple[str, ...], tuple[Path, ...]]
            ] = []
            anchor_plan_index: dict[str, list[str]] = {}
            anchor_closure_cache: dict[str, tuple[str, str]] = {}
            cached_manifest_tasks: dict[str, dict[str, object]] = {}
            cached_bins_for_return: list[Path] = []
            cached_aid_for_return = ""
            covered_anchor_t_keys: set[str] = set()
            cached_bins_by_dedup_key: dict[str, tuple[Path, ...]] = {}
            covered_anchor_variants: dict[
                tuple[str, str, str, tuple[str, ...]],
                tuple[int | None, tuple[Path, ...]],
            ] = {}

            def _get_anchor_closure(rtag: str) -> tuple[str, str]:
                if rtag not in anchor_closure_cache:
                    c_src = extract_reachable_closure_source(module_source, regime_tag=rtag)
                    c_id = compute_compile_task_key(
                        normalized_kernel_unit_src=c_src,
                        cann_target="dav-3510",
                        pass_configs=None,
                        compile_flags=(),
                        static_spec_signature=f"{operator_name}:{rtag}:{cases_yaml_sha256}",
                        compiler_stamp=tl_lib_stamp,
                    )
                    anchor_closure_cache[rtag] = (c_src, c_id)
                return anchor_closure_cache[rtag]

            for ptask in planned_anchor_tasks:
                cnum = int(ptask.get("case_num", -1))
                rtag = str(ptask.get("regime_tag", ""))
                t_mode = str(ptask.get("mode", ""))
                t_bname = str(ptask.get("builder_name", builder_name))
                t_sel = str(ptask.get("selector_name", ""))
                t_fkw = dict(ptask.get("factory_kwargs") or {})
                t_dims = dict(ptask.get("dims") or {})
                t_kwargs = dict(ptask.get("kwargs") or {})
                t_pc = dict(ptask.get("pass_configs") or {})
                t_cf = tuple(str(f) for f in (ptask.get("compile_flags") or ()))
                c_src, c_id = _get_anchor_closure(rtag)
                ta_sig = _canonical_tracka_spec_signature(t_bname, t_kwargs)
                t_key = compute_compile_task_key(
                    normalized_kernel_unit_src=c_src,
                    cann_target="dav-3510",
                    pass_configs=t_pc,
                    compile_flags=t_cf,
                    static_spec_signature=ta_sig,
                    compiler_stamp=tl_lib_stamp,
                )
                p_key = _build_prewarm_plan_key(
                    operator_name=operator_name,
                    cand_sha256=c_id,
                    case_num=cnum,
                    cases_yaml_sha256=cases_yaml_sha256,
                    tilelang_lib_stamp=tl_lib_stamp,
                )
                anchor_plan_index.setdefault(p_key, []).append(t_key)
                dedup_kwargs = {
                    str(k): _json_safe_value(v)
                    for k, v in t_kwargs.items()
                    if str(k) not in _EXCLUDED_TRACKA_SIG_KEYS
                }
                dedup_key = json.dumps(
                    {
                        "mode": t_mode,
                        "builder_name": t_bname,
                        "kwargs": dedup_kwargs,
                        "pass_configs": _json_safe_value(t_pc),
                        "compile_flags": list(t_cf),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
                t_dims_sig = (
                    json.dumps(
                        _json_safe_value(t_dims),
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    )
                    if t_dims
                    else ""
                )
                t_pc_sig = json.dumps(
                    _json_safe_value(t_pc),
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
                covered_anchor_t_keys.add(t_key)
                if resolved_store.has_verified_artifact(t_key):
                    cached_manifest_tasks[t_key] = {
                        "builder_name": str(t_bname),
                        "cand_sha256": c_id,
                        "tilelang_lib_stamp": tl_lib_stamp,
                    }
                    existing_art = resolved_store.get_verified_artifact(t_key)
                    art_bins: list[Path] = []
                    if existing_art is not None:
                        if not cached_aid_for_return:
                            cached_aid_for_return = existing_art.artifact_id
                        for bp in existing_art.binary_paths:
                            bp_obj = Path(bp)
                            if is_verified_binary_file(bp_obj):
                                art_bins.append(bp_obj)
                                if bp_obj not in cached_bins_for_return:
                                    cached_bins_for_return.append(bp_obj)
                    art_bins_tuple = tuple(art_bins)
                    if art_bins_tuple:
                        cached_bins_by_dedup_key.setdefault(dedup_key, art_bins_tuple)
                    if t_dims_sig:
                        covered_anchor_variants.setdefault(
                            (t_bname, t_dims_sig, t_pc_sig, t_cf),
                            (None, art_bins_tuple),
                        )
                    continue
                if dedup_key not in dedup_idx_by_key:
                    dedup_idx_by_key[dedup_key] = len(unique_compile_specs)
                    unique_compile_specs.append(
                        {
                            "factory_name": t_bname,
                            "factory_kwargs": t_fkw,
                            "config_selector_name": t_sel or None,
                            "pass_configs": t_pc,
                            "compile_flags": t_cf,
                            "dims": t_dims or None,
                            "is_tail": False,
                        }
                    )
                w_idx = dedup_idx_by_key[dedup_key]
                if t_dims_sig:
                    covered_anchor_variants.setdefault(
                        (t_bname, t_dims_sig, t_pc_sig, t_cf),
                        (w_idx, ()),
                    )
                spec_label = f"{operator_name}:{cnum}" if operator_name else ta_sig
                task_record_bindings.append(
                    (w_idx, t_key, spec_label, t_bname, c_id, c_src, t_cf, ())
                )

            norm_pc_sig = json.dumps(
                _json_safe_value(norm_pc),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            norm_cf_tuple = tuple(norm_cf)
            for idx, (t_key, v_spec_sig, v_dims) in enumerate(task_entries):
                if resolved_store.has_verified_artifact(t_key):
                    cached_manifest_tasks[t_key] = {
                        "builder_name": str(builder_name),
                        "cand_sha256": cand_sha,
                        "tilelang_lib_stamp": tl_lib_stamp,
                    }
                    existing_art = resolved_store.get_verified_artifact(t_key)
                    if existing_art is not None:
                        if not cached_aid_for_return:
                            cached_aid_for_return = existing_art.artifact_id
                        for bp in existing_art.binary_paths:
                            bp_obj = Path(bp)
                            if (
                                is_verified_binary_file(bp_obj)
                                and bp_obj not in cached_bins_for_return
                            ):
                                cached_bins_for_return.append(bp_obj)
                    continue
                if t_key in covered_anchor_t_keys:
                    continue
                eff_kw = _merge_dims_into_kwargs(builder_kwargs, v_dims)
                filtered_kw = _filter_eff_kwargs_for_entry_source(
                    module_source, builder_name, eff_kw
                )
                t_mode = "_get_kernel" if "_get_kernel" in module_source else "generic"
                dedup_payload: dict[str, object] = {
                    "mode": t_mode,
                    "builder_name": builder_name,
                    "kwargs": _json_safe_value(filtered_kw),
                    "pass_configs": _json_safe_value(norm_pc),
                    "compile_flags": list(norm_cf),
                }
                dedup_key = json.dumps(
                    dedup_payload, sort_keys=True, separators=(",", ":"), default=str
                )
                v_dims_sig = (
                    json.dumps(
                        _json_safe_value(v_dims),
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    )
                    if v_dims
                    else ""
                )
                variant_key = (builder_name, v_dims_sig, norm_pc_sig, norm_cf_tuple)
                if dedup_key in dedup_idx_by_key:
                    w_idx = dedup_idx_by_key[dedup_key]
                    task_record_bindings.append(
                        (
                            w_idx,
                            t_key,
                            v_spec_sig,
                            builder_name,
                            cand_sha,
                            closure_src,
                            norm_cf_tuple,
                            (),
                        )
                    )
                    continue
                if dedup_key in cached_bins_by_dedup_key:
                    task_record_bindings.append(
                        (
                            None,
                            t_key,
                            v_spec_sig,
                            builder_name,
                            cand_sha,
                            closure_src,
                            norm_cf_tuple,
                            cached_bins_by_dedup_key[dedup_key],
                        )
                    )
                    continue
                if v_dims_sig and variant_key in covered_anchor_variants:
                    cov_w_idx, cov_bins = covered_anchor_variants[variant_key]
                    task_record_bindings.append(
                        (
                            cov_w_idx,
                            t_key,
                            v_spec_sig,
                            builder_name,
                            cand_sha,
                            closure_src,
                            norm_cf_tuple,
                            cov_bins,
                        )
                    )
                    continue
                is_tail_var = (
                    has_tail and idx == 1 and dict(v_dims) == dict(tail_anchor_dims or {})
                )
                dedup_idx_by_key[dedup_key] = len(unique_compile_specs)
                unique_compile_specs.append(
                    {
                        "factory_name": builder_name,
                        "factory_kwargs": dict(builder_kwargs),
                        "config_selector_name": config_selector_name,
                        "pass_configs": pass_configs,
                        "compile_flags": compile_flags,
                        "dims": v_dims or None,
                        "is_tail": is_tail_var,
                    }
                )
                w_idx = dedup_idx_by_key[dedup_key]
                task_record_bindings.append(
                    (
                        w_idx,
                        t_key,
                        v_spec_sig,
                        builder_name,
                        cand_sha,
                        closure_src,
                        norm_cf_tuple,
                        (),
                    )
                )

            def _run_compile_worker(w_idx: int, spec: Mapping[str, object]) -> dict[str, object]:
                w_dir = tmp / f"worker_{w_idx}"
                w_cache = w_dir / "tl_cache"
                w_dir.mkdir(parents=True, exist_ok=True)
                seed_tl_cache_tree(resolved_tl_cache, w_cache)
                pre_bins = snapshot_tl_cache_binaries(w_cache)
                emit_path = w_dir / "emit_cache_bins.json"
                drv_path = w_dir / "compile_driver.py"
                drv_src = build_stage_b_driver_source(
                    cand_path,
                    factory_name=str(spec["factory_name"]),
                    factory_kwargs=spec["factory_kwargs"],
                    config_selector_name=spec["config_selector_name"],
                    pass_configs=spec["pass_configs"],
                    compile_flags=spec["compile_flags"],
                    dims=spec["dims"],
                    enable_device_compile=True,
                    kernel_dump_path=None,
                    compile_for_cache=True,
                    emit_cache_bins_path=emit_path,
                )
                drv_path.write_text(drv_src, encoding="utf-8")
                w_env = dict(run_env)
                w_env["TILELANG_CACHE_DIR"] = str(w_cache)
                try:
                    w_proc = subprocess.run(
                        _build_l1_argv(drv_path),
                        text=True,
                        capture_output=True,
                        cwd=str(w_dir),
                        env=w_env,
                        timeout=timeout_sec,
                    )
                except subprocess.TimeoutExpired as exc:
                    captured = "".join(
                        part for part in (exc.stdout, exc.stderr) if isinstance(part, str)
                    )
                    return {
                        "ok": False,
                        "timeout": True,
                        "is_tail": bool(spec.get("is_tail")),
                        "diag": captured,
                    }
                if w_proc.returncode != 0:
                    return {
                        "ok": False,
                        "timeout": False,
                        "rc": w_proc.returncode,
                        "is_tail": bool(spec.get("is_tail")),
                        "stdout": w_proc.stdout or "",
                        "stderr": w_proc.stderr or "",
                    }
                new_rels = collect_new_tl_cache_binaries(w_cache, pre_bins)
                ext_bins = [
                    p
                    for p in sorted(w_dir.rglob("*"))
                    if not p.is_relative_to(w_cache) and is_verified_binary_file(p)
                ]
                return {
                    "ok": True,
                    "worker_cache": w_cache,
                    "emit_path": emit_path,
                    "new_rels": new_rels,
                    "ext_bins": ext_bins,
                }

            if cached_diag is not None and not unique_compile_specs:
                diag_res = cached_diag
            elif diag_fut is not None and not unique_compile_specs:
                diag_res = diag_fut.result()
            else:
                diag_jobs = 0 if (cached_diag is not None or diag_fut is not None) else 1
                total_jobs = len(unique_compile_specs) + diag_jobs
                max_workers = min(12, os.cpu_count() or 4, max(1, total_jobs))
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    if cached_diag is None and diag_fut is None:
                        diag_fut = pool.submit(_run_diag_worker)
                    compile_futs = {
                        w_idx: pool.submit(_run_compile_worker, w_idx, spec)
                        for w_idx, spec in enumerate(unique_compile_specs)
                    }
                    diag_res = cached_diag if diag_fut is None else diag_fut.result()
                    for w_idx, fut in compile_futs.items():
                        compile_results[w_idx] = fut.result()
        finally:
            if pre_diag_pool is not None:
                pre_diag_pool.shutdown(wait=True)

        if diag_res.get("timeout"):
            return StageBResult(
                passed=False,
                error_message=f"l1_compile_timeout:{timeout_sec}s",
                elapsed_sec=time.monotonic() - t0,
                full_diagnostic=str(diag_res.get("diag") or ""),
                fault_plane="STAGE_B_COMPILE",
                fault_code="L1_COMPILE_TIMEOUT",
                failed_tier="stage_b",
            )
        spill_bytes = int(diag_res.get("spill_bytes") or 0)
        vloopv2_hit = bool(diag_res.get("vloopv2_hit"))
        summary = diag_res.get("summary")
        ma_profile = diag_res.get("ma_profile")
        lowered_dump = str(diag_res.get("lowered_dump") or "")
        combined_out = str(diag_res.get("combined_out") or "")
        if not diag_res.get("ok"):
            raw_diagnostic = str(diag_res.get("stderr") or diag_res.get("stdout") or "")
            detail = _bounded_compile_diagnostic(raw_diagnostic)
            fault_code = _classify_stage_b_overflow_fault_code(
                f"{raw_diagnostic}\n{lowered_dump}",
                summary,
                "L1_COMPILE_FAILED",
            )
            return StageBResult(
                passed=False,
                error_message=f"l1_compile_failed (rc={diag_res.get('rc', 1)}): {detail}",
                elapsed_sec=time.monotonic() - t0,
                spill_bytes=spill_bytes,
                vloopv2_hit=vloopv2_hit,
                compiler_summary=summary,
                microarch_profile=ma_profile,
                full_diagnostic=raw_diagnostic,
                fault_plane="STAGE_B_COMPILE",
                fault_code=fault_code,
                failed_tier="stage_b",
            )
        if _has_vf_stack_overflow_signal(combined_out, summary, norm_cf, norm_pc):
            return StageBResult(
                passed=False,
                error_message=f"l1_vf_stack_overflow: stack_bytes={summary.stack_bytes}",
                elapsed_sec=time.monotonic() - t0,
                spill_bytes=spill_bytes,
                vloopv2_hit=vloopv2_hit,
                compiler_summary=summary,
                microarch_profile=ma_profile,
                full_diagnostic=combined_out,
                fault_plane="STAGE_B_COMPILE",
                fault_code="VF_STACK_OVERFLOW",
                failed_tier="stage_b",
            )

        for w_idx in sorted(compile_results):
            cres = compile_results[w_idx]
            if cres.get("timeout"):
                return StageBResult(
                    passed=False,
                    error_message=f"l1_compile_timeout:{timeout_sec}s",
                    elapsed_sec=time.monotonic() - t0,
                    spill_bytes=spill_bytes,
                    vloopv2_hit=vloopv2_hit,
                    compiler_summary=summary,
                    microarch_profile=ma_profile,
                    full_diagnostic=str(cres.get("diag") or ""),
                    fault_plane="STAGE_B_COMPILE",
                    fault_code="L1_COMPILE_TIMEOUT",
                    failed_tier="stage_b",
                )
            if not cres.get("ok"):
                raw_diagnostic = str(cres.get("stderr") or cres.get("stdout") or "")
                detail = _bounded_compile_diagnostic(raw_diagnostic)
                default_code = (
                    "STAGE_B_TAIL_ANCHOR_FAILED"
                    if cres.get("is_tail")
                    else "L1_COMPILE_FAILED"
                )
                fault_code = _classify_stage_b_overflow_fault_code(
                    raw_diagnostic,
                    None,
                    default_code,
                )
                return StageBResult(
                    passed=False,
                    error_message=f"l1_compile_failed (rc={cres.get('rc', 1)}): {detail}",
                    elapsed_sec=time.monotonic() - t0,
                    spill_bytes=spill_bytes,
                    vloopv2_hit=vloopv2_hit,
                    compiler_summary=summary,
                    microarch_profile=ma_profile,
                    full_diagnostic=raw_diagnostic,
                    fault_plane="STAGE_B_COMPILE",
                    fault_code=fault_code,
                    failed_tier="stage_b",
                )

        if compile_results:
            lock_path = resolved_tl_cache / ".prewarm_manifest_v1.lock"
            with lock_path.open("a+", encoding="utf-8") as lock_fh:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
                try:
                    for w_idx in sorted(compile_results):
                        w_cache = compile_results[w_idx]["worker_cache"]
                        merge_tl_cache_tree(w_cache, resolved_tl_cache)
                finally:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

        worker_bins_map: dict[int, list[Path]] = {}
        for w_idx, cres in compile_results.items():
            w_cache = cres["worker_cache"]
            w_bins = _resolve_worker_cache_bins(
                cres["emit_path"],
                w_cache,
                resolved_tl_cache,
                cres["new_rels"],
            )
            w_bins.extend(cres["ext_bins"])
            worker_bins_map[w_idx] = list(dict.fromkeys(w_bins))

        per_task_records = [
            (
                t_key,
                t_sig,
                t_builder,
                t_csha,
                t_csrc,
                t_cf,
                list(preset_bins) if w_idx is None else worker_bins_map.get(w_idx, []),
            )
            for (
                w_idx,
                t_key,
                t_sig,
                t_builder,
                t_csha,
                t_csrc,
                t_cf,
                preset_bins,
            ) in task_record_bindings
        ]
        all_plan_entries = dict(anchor_plan_index)
        var_task_keys = [tk for tk, _sig, _vd in task_entries]
        if var_task_keys:
            all_plan_entries[plan_key] = var_task_keys

        recorded_aid, recorded_bins = _record_l1_artifacts_and_manifest(
            resolved_tl_cache=resolved_tl_cache,
            resolved_store=resolved_store,
            task_entries=task_entries,
            binary_paths=(),
            builder_name=builder_name,
            closure_src=closure_src,
            cand_sha256=cand_sha,
            tl_lib_stamp=tl_lib_stamp,
            norm_cf=norm_cf,
            plan_key=plan_key,
            per_task_records=per_task_records,
            plan_index_entries=all_plan_entries,
            cached_manifest_tasks=cached_manifest_tasks,
        )
        if cached_diag is None:
            _save_stage_b_diag_cache(
                resolved_tl_cache,
                primary_t_key,
                spill_bytes=spill_bytes,
                vloopv2_hit=vloopv2_hit,
                summary=summary,
                ma_profile=ma_profile,
            )
    return StageBResult(
        passed=True,
        elapsed_sec=time.monotonic() - t0,
        spill_bytes=spill_bytes,
        vloopv2_hit=vloopv2_hit,
        compiler_summary=summary,
        microarch_profile=ma_profile,
        artifact_id=recorded_aid or cached_aid_for_return or primary_t_key,
        binary_paths=recorded_bins or tuple(str(p) for p in cached_bins_for_return),
    )


def _repair_timeout_message(error_message: str, is_transport: bool) -> str:
    """Append the repair timeout marker while preserving the original failure tier."""
    return f"{error_message} [repair_timeout]" if is_transport else error_message


def _strip_cpp_comments_keep_strings(text: str) -> str:
    """Strip C/C++ line and block comments while keeping string literals intact."""
    out: list[str] = []
    i = 0
    n = len(text)
    in_str = ""
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(nxt)
                i += 2
                continue
            if ch == in_str:
                in_str = ""
            i += 1
            continue
        if ch in ('"', "'"):
            in_str = ch
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            i += 2
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _normalize_trackb_cpp_literal(text: str) -> str:
    cleaned = _strip_cpp_comments_keep_strings(text)
    collapsed = " ".join(cleaned.split())
    return hashlib.sha256(collapsed.encode("utf-8")).hexdigest()


_CPP_LITERAL_MARKERS = ("__global__", "__aicore__", "__ubuf__", "__simd_vf__", "#include", 'extern "C"')


class _NormalizedAstStrings(ast.NodeTransformer):
    """Canonicalize Track-B C++ string literals inside Python AST before dumping."""

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str) and any(tok in node.value for tok in _CPP_LITERAL_MARKERS):
            return ast.copy_location(
                ast.Constant(value=f"__cpp_tu__:{_normalize_trackb_cpp_literal(node.value)}"),
                node,
            )
        return node


def _normalized_module_ast_signature(source: str) -> str | None:
    """Return canonical AST+TU sha256 signature for Python module source, or None on SyntaxError."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    normalized = _NormalizedAstStrings().visit(tree)
    ast.fix_missing_locations(normalized)
    dumped = ast.dump(normalized, include_attributes=False)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


_GLOBAL_KERNEL_DEF_RE = re.compile(
    r"__global__\s+(?:__aicore__|__vector__|__cube__|__mix__\([^)]*\)|\s)*void\s+([A-Za-z_][A-Za-z0-9_]*)"
)


def _count_trackb_workspace_stages(source: str) -> int:
    """Count distinct __global__ kernels and _plan_workspace_bytes stage hooks in *source*."""
    kernels = set(_GLOBAL_KERNEL_DEF_RE.findall(source))
    has_ws = bool(re.search(r"(?m)^\s*def\s+(?:_[a-z0-9_]+_)?plan_workspace_bytes\b", source))
    return len(kernels) + (1 if has_ws else 0)


def _extract_candidate_topology_axes(
    stage_a: StageAResult,
) -> TopologyAxes | None:
    route = getattr(stage_a, "route", None)
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
    return None


def _should_skip_syntax_repair(fault_code: str) -> bool:
    return not classify_stage_failure_for_repair(fault_code)


def _is_noop_identical_failure(
    *,
    failed_tier: str = "",
    error_message: str = "",
    result: object | None = None,
) -> bool:
    fp = str(getattr(result, "fault_plane", "") or "").upper()
    fc = str(getattr(result, "fault_code", "") or "").upper()
    if fp == "NOOP_IDENTICAL" or fc == "NOOP_IDENTICAL":
        return True
    if str(failed_tier or "").lower() == "noop_identical":
        return True
    return "noop_identical" in str(error_message or "").lower()


def _merge_interval_wall_sec(intervals: Sequence[tuple[float, float]]) -> float:
    ordered = sorted((s, e) for s, e in intervals if e > s)
    if not ordered:
        return 0.0
    total = 0.0
    cur_s, cur_e = ordered[0]
    for s, e in ordered[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total


async def run_l0_l1_round_filter(
    candidates: Sequence[tuple[str, str]],
    *,
    patch_fn: Callable[[str], str],
    l0_checker: Callable[[str], StageAResult],
    l1_compiler: Callable[[str], StageBResult],
    repatch_from_current_fn: Callable[[str, str], str] | None = None,
    repair_backend: KernelLLMBackend | None = None,
    max_syntax_repairs: int = 1,
    slice_hint: str = "",
    backend_kind: str = "tilelang_dsl",
    parent_source: str = "",
    parent_module_source: str = "",
    trackb_prologue_line_offset: int = 0,
    stage_a_cost_ranker: Callable[
        [Sequence[tuple[str, str, StageAResult]]], Sequence[tuple[str, float]]
    ]
    | None = None,
    cost_prescreen_top_k: int | None = None,
    sil_advisory: object | None = None,
    sil_verification: object | None = None,
    freedom_mask: ConstraintRelaxationMask | None = None,
    vector_mode: str = "simdvf",
    ub_256kb_unlock: bool = False,
    creative_candidate_ids: Collection[str] = (),
    parent_analytical_us: float | None = None,
    stage_a_analytical_scorer: Callable[[str, str, StageAResult], float] | None = None,
    parent_topology_axes: TopologyAxes | None = None,
    candidate_topology_axes_fn: Callable[[str, StageAResult], TopologyAxes | None] | None = None,
    parent_workspace_stages: int | None = None,
    candidate_workspace_stages_fn: Callable[[str, StageAResult], int] | None = None,
    skip_repair_when_survivors_exist: bool = False,
) -> list[L0L1CandidateCheck]:
    """Filter candidates concurrently via noop guard, L0 static, prescreen, and L1 compile."""
    _ = trackb_prologue_line_offset
    max_attempts = max(0, max_syntax_repairs)
    bisheng_sem = asyncio.Semaphore(min(8, os.cpu_count() or 4))
    repair_intervals: list[tuple[float, float]] = []
    cum_repair_worker_sec = 0.0
    baseline_compare_source = (
        parent_module_source if parent_module_source.strip() else parent_source
    )
    baseline_ast_sig = (
        _normalized_module_ast_signature(baseline_compare_source)
        if baseline_compare_source.strip()
        else None
    )
    eff_parent_ws_stages = (
        int(parent_workspace_stages)
        if parent_workspace_stages is not None
        else (
            _count_trackb_workspace_stages(baseline_compare_source)
            if baseline_compare_source.strip()
            else 0
        )
    )

    async def _try_repair(
        cur_src: str,
        diag: str,
        *,
        lowered_excerpt: str = "",
        lowered_excerpt_lang: str | None = None,
    ) -> tuple[str | None, str, bool]:
        nonlocal cum_repair_worker_sec
        assert repair_backend is not None
        req = build_syntax_repair_request(
            cur_src,
            diag,
            slice_hint=slice_hint,
            sil_advisory=sil_advisory,
            sil_verification=sil_verification,
            backend_kind=backend_kind,
            parent_source=parent_source,
            freedom_mask=freedom_mask,
            vector_mode=vector_mode,
            ub_256kb_unlock=ub_256kb_unlock,
            lowered_excerpt=lowered_excerpt,
            lowered_excerpt_lang=lowered_excerpt_lang,
        )
        t_rep0 = time.monotonic()
        try:
            raw_reply = await repair_backend.generate(req)
        except Exception as exc:
            t_rep1 = time.monotonic()
            repair_intervals.append((t_rep0, t_rep1))
            cum_repair_worker_sec += max(0.0, t_rep1 - t_rep0)
            return None, f"{diag} | repair_transport: {type(exc).__name__}: {exc}", True
        t_rep1 = time.monotonic()
        repair_intervals.append((t_rep0, t_rep1))
        cum_repair_worker_sec += max(0.0, t_rep1 - t_rep0)
        try:
            repaired_src = extract_python_dsl(raw_reply)
        except Exception as exc:
            return None, f"{diag} | repair_parse: {type(exc).__name__}: {exc}", False
        if parent_source.strip():
            violations = check_repair_monotonicity(
                parent_source,
                repaired_src,
                backend_kind=backend_kind,
            )
            if violations:
                return (
                    None,
                    f"{diag} | repair_monotonicity_violation: {'; '.join(violations)}",
                    False,
                )
        return repaired_src, "", False

    async def _run_stage_a_phase(
        cand_id: str,
        cur_snippet: str,
        *,
        start_attempt: int = 0,
        repaired: bool = False,
        initial_patch_fn: Callable[[str], str] | None = None,
        attempt_limit: int | None = None,
    ) -> tuple[L0L1CandidateCheck | None, tuple[str, str, StageAResult, bool, int] | None]:
        snippet = cur_snippet
        is_repaired = repaired
        active_patch_fn = initial_patch_fn or patch_fn
        eff_max_attempts = max_attempts if attempt_limit is None else max(0, attempt_limit)
        for attempt in range(start_attempt, eff_max_attempts + 1):
            try:
                patched = active_patch_fn(snippet)
            except Exception as exc:
                err = f"ast_patch_error: {type(exc).__name__}: {exc}"
                if _is_noop_identical_failure(error_message=err):
                    return (
                        L0L1CandidateCheck(
                            candidate_id=cand_id,
                            passed=False,
                            failed_tier="noop_identical",
                            patched_module_source="",
                            snippet_source=snippet,
                            repaired=is_repaired,
                            error_message=err,
                            fault_plane="NOOP_IDENTICAL",
                            fault_code="NOOP_IDENTICAL",
                        ),
                        None,
                    )
                if repair_backend is None or attempt >= eff_max_attempts:
                    return (
                        L0L1CandidateCheck(
                            candidate_id=cand_id,
                            passed=False,
                            failed_tier="stage_a",
                            patched_module_source="",
                            snippet_source=snippet,
                            repaired=is_repaired,
                            error_message=err,
                            fault_plane="STAGE_A_STATIC",
                            fault_code="AST_PATCH_ERROR",
                        ),
                        None,
                    )
                next_snippet, rep_err, is_transport = await _try_repair(snippet, err)
                if next_snippet is None:
                    return (
                        L0L1CandidateCheck(
                            candidate_id=cand_id,
                            passed=False,
                            failed_tier="stage_a",
                            patched_module_source="",
                            snippet_source=snippet,
                            repaired=is_repaired,
                            error_message=_repair_timeout_message(rep_err, is_transport),
                            fault_plane="STAGE_A_STATIC",
                            fault_code="SYNTAX_REPAIR_FAILED",
                        ),
                        None,
                    )
                snippet = next_snippet
                is_repaired = True
                active_patch_fn = patch_fn
                continue

            active_patch_fn = patch_fn
            if baseline_ast_sig is not None:
                patched_sig = _normalized_module_ast_signature(patched)
                if patched_sig is not None and patched_sig == baseline_ast_sig:
                    return (
                        L0L1CandidateCheck(
                            candidate_id=cand_id,
                            passed=False,
                            failed_tier="noop_identical",
                            patched_module_source=patched,
                            snippet_source=snippet,
                            repaired=is_repaired,
                            error_message="noop_identical_to_parent",
                            fault_plane="NOOP_IDENTICAL",
                            fault_code="NOOP_IDENTICAL",
                        ),
                        None,
                    )
            stage_a = await asyncio.to_thread(l0_checker, patched)
            if not stage_a.passed:
                err = stage_a.error_message or "l0_static_failed"
                if _is_noop_identical_failure(error_message=err, result=stage_a):
                    return (
                        L0L1CandidateCheck(
                            candidate_id=cand_id,
                            passed=False,
                            failed_tier="noop_identical",
                            patched_module_source=patched,
                            snippet_source=snippet,
                            repaired=is_repaired,
                            stage_a=stage_a,
                            error_message=err,
                            fault_plane="NOOP_IDENTICAL",
                            fault_code="NOOP_IDENTICAL",
                        ),
                        None,
                    )
                fp = str(getattr(stage_a, "fault_plane", "") or "STAGE_A_STATIC")
                fc = str(getattr(stage_a, "fault_code", "") or "STAGE_A_FAILED")
                if (
                    repair_backend is None
                    or attempt >= eff_max_attempts
                    or _should_skip_syntax_repair(fc)
                ):
                    return (
                        L0L1CandidateCheck(
                            candidate_id=cand_id,
                            passed=False,
                            failed_tier="stage_a",
                            patched_module_source=patched,
                            snippet_source=snippet,
                            repaired=is_repaired,
                            stage_a=stage_a,
                            error_message=err,
                            fault_plane=fp,
                            fault_code=fc,
                        ),
                        None,
                    )
                next_snippet, rep_err, is_transport = await _try_repair(snippet, err)
                if next_snippet is None:
                    return (
                        L0L1CandidateCheck(
                            candidate_id=cand_id,
                            passed=False,
                            failed_tier="stage_a",
                            patched_module_source=patched,
                            snippet_source=snippet,
                            repaired=is_repaired,
                            stage_a=stage_a,
                            error_message=_repair_timeout_message(rep_err, is_transport),
                            fault_plane=fp,
                            fault_code=fc,
                        ),
                        None,
                    )
                snippet = next_snippet
                is_repaired = True
                continue
            return None, (snippet, patched, stage_a, is_repaired, attempt)
        raise RuntimeError("unreachable repair termination")

    def _make_stage_b_repair_patch_fn(base_patched: str) -> Callable[[str], str]:
        if repatch_from_current_fn is None or not base_patched.strip():
            return patch_fn

        def _patch_repaired(snip: str) -> str:
            try:
                return repatch_from_current_fn(base_patched, snip)
            except Exception:
                if not parse_scoped_search_replace_blocks(snip):
                    return patch_fn(snip)
                raise

        return _patch_repaired

    async def _run_stage_b_and_repairs(
        cand_id: str,
        snippet: str,
        patched: str,
        stage_a: StageAResult,
        repaired: bool,
        attempt: int,
        *,
        attempt_limit: int | None = None,
    ) -> L0L1CandidateCheck:
        cur_snippet = snippet
        cur_patched = patched
        cur_stage_a = stage_a
        cur_repaired = repaired
        cur_attempt = attempt
        eff_max_attempts = max_attempts if attempt_limit is None else max(0, attempt_limit)
        while True:
            async with bisheng_sem:
                stage_b = await asyncio.to_thread(l1_compiler, cur_patched)
            if stage_b.passed:
                return L0L1CandidateCheck(
                    candidate_id=cand_id,
                    passed=True,
                    failed_tier=None,
                    patched_module_source=cur_patched,
                    snippet_source=cur_snippet,
                    repaired=cur_repaired,
                    stage_a=cur_stage_a,
                    stage_b=stage_b,
                )
            err = stage_b.error_message or "l1_compile_failed"
            if _is_noop_identical_failure(error_message=err, result=stage_b):
                return L0L1CandidateCheck(
                    candidate_id=cand_id,
                    passed=False,
                    failed_tier="noop_identical",
                    patched_module_source=cur_patched,
                    snippet_source=cur_snippet,
                    repaired=cur_repaired,
                    stage_a=cur_stage_a,
                    stage_b=stage_b,
                    error_message=err,
                    fault_plane="NOOP_IDENTICAL",
                    fault_code="NOOP_IDENTICAL",
                )
            fp = str(getattr(stage_b, "fault_plane", "") or "STAGE_B_COMPILE")
            fc = str(getattr(stage_b, "fault_code", "") or "STAGE_B_FAILED")
            if (
                repair_backend is None
                or cur_attempt >= eff_max_attempts
                or _should_skip_syntax_repair(fc)
            ):
                return L0L1CandidateCheck(
                    candidate_id=cand_id,
                    passed=False,
                    failed_tier="stage_b",
                    patched_module_source=cur_patched,
                    snippet_source=cur_snippet,
                    repaired=cur_repaired,
                    stage_a=cur_stage_a,
                    stage_b=stage_b,
                    error_message=err,
                    fault_plane=fp,
                    fault_code=fc,
                )
            lowered_excerpt, excerpt_lang = extract_lowered_compiler_excerpt_with_lang(stage_b)
            next_snippet, rep_err, is_transport = await _try_repair(
                cur_snippet,
                err,
                lowered_excerpt=lowered_excerpt,
                lowered_excerpt_lang=excerpt_lang,
            )
            if next_snippet is None:
                return L0L1CandidateCheck(
                    candidate_id=cand_id,
                    passed=False,
                    failed_tier="stage_b",
                    patched_module_source=cur_patched,
                    snippet_source=cur_snippet,
                    repaired=cur_repaired,
                    stage_a=cur_stage_a,
                    stage_b=stage_b,
                    error_message=_repair_timeout_message(rep_err, is_transport),
                    fault_plane=fp,
                    fault_code=fc,
                )
            failed_chk, passed_state = await _run_stage_a_phase(
                cand_id,
                next_snippet,
                start_attempt=cur_attempt + 1,
                repaired=True,
                initial_patch_fn=_make_stage_b_repair_patch_fn(cur_patched),
            )
            if failed_chk is not None:
                return failed_chk
            assert passed_state is not None
            cur_snippet, cur_patched, cur_stage_a, cur_repaired, cur_attempt = passed_state

    async def _check_one(cand_id: str, snippet: str) -> L0L1CandidateCheck:
        failed_chk, passed_state = await _run_stage_a_phase(cand_id, snippet)
        if failed_chk is not None:
            return failed_chk
        assert passed_state is not None
        cur_snippet, patched, stage_a, repaired, attempt = passed_state
        return await _run_stage_b_and_repairs(
            cand_id, cur_snippet, patched, stage_a, repaired, attempt
        )

    async def _phase1_no_repair(cand_id: str, snippet: str) -> L0L1CandidateCheck:
        failed_chk, passed_state = await _run_stage_a_phase(
            cand_id, snippet, attempt_limit=0
        )
        if failed_chk is not None:
            return failed_chk
        assert passed_state is not None
        cur_snippet, patched, stage_a, repaired, attempt = passed_state
        return await _run_stage_b_and_repairs(
            cand_id, cur_snippet, patched, stage_a, repaired, attempt, attempt_limit=0
        )

    async def _phase2_repair_failed(chk: L0L1CandidateCheck) -> L0L1CandidateCheck:
        if chk.passed or chk.failed_tier == "noop_identical":
            return chk
        fc = str(chk.fault_code or "")
        fp = str(chk.fault_plane or "")
        if _should_skip_syntax_repair(fc):
            return chk
        err = chk.error_message or "l1_compile_failed"
        if chk.failed_tier == "stage_b" and chk.stage_b is not None:
            lowered_excerpt, excerpt_lang = extract_lowered_compiler_excerpt_with_lang(
                chk.stage_b
            )
            next_snippet, rep_err, is_transport = await _try_repair(
                chk.snippet_source,
                err,
                lowered_excerpt=lowered_excerpt,
                lowered_excerpt_lang=excerpt_lang,
            )
            if next_snippet is None:
                return replace(
                    chk, error_message=_repair_timeout_message(rep_err, is_transport)
                )
            failed_chk, passed_state = await _run_stage_a_phase(
                chk.candidate_id,
                next_snippet,
                start_attempt=1,
                repaired=True,
                initial_patch_fn=_make_stage_b_repair_patch_fn(chk.patched_module_source),
            )
            if failed_chk is not None:
                return failed_chk
            assert passed_state is not None
            cur_snip, patched, stage_a, repaired, attempt = passed_state
            return await _run_stage_b_and_repairs(
                chk.candidate_id, cur_snip, patched, stage_a, repaired, attempt
            )
        next_snippet, rep_err, is_transport = await _try_repair(chk.snippet_source, err)
        if next_snippet is None:
            out_fc = "SYNTAX_REPAIR_FAILED" if fc == "AST_PATCH_ERROR" else fc
            return L0L1CandidateCheck(
                candidate_id=chk.candidate_id,
                passed=False,
                failed_tier="stage_a",
                patched_module_source=chk.patched_module_source,
                snippet_source=chk.snippet_source,
                repaired=chk.repaired,
                stage_a=chk.stage_a,
                error_message=_repair_timeout_message(rep_err, is_transport),
                fault_plane=fp or "STAGE_A_STATIC",
                fault_code=out_fc or "STAGE_A_FAILED",
            )
        failed_chk, passed_state = await _run_stage_a_phase(
            chk.candidate_id, next_snippet, start_attempt=1, repaired=True
        )
        if failed_chk is not None:
            return failed_chk
        assert passed_state is not None
        cur_snip, patched, stage_a, repaired, attempt = passed_state
        return await _run_stage_b_and_repairs(
            chk.candidate_id, cur_snip, patched, stage_a, repaired, attempt
        )

    def _finalize(items: Sequence[L0L1CandidateCheck]) -> L0L1RoundResults:
        wall_sec = _merge_interval_wall_sec(repair_intervals)
        cum_sec = float(cum_repair_worker_sec)
        tagged = [
            replace(chk, wall_repair_sec=wall_sec, cum_repair_worker_sec=cum_sec)
            for chk in items
        ]
        return L0L1RoundResults(
            tagged,
            wall_repair_sec=wall_sec,
            cum_repair_worker_sec=cum_sec,
        )

    use_prescreen = (
        stage_a_cost_ranker is not None
        and cost_prescreen_top_k is not None
        and int(cost_prescreen_top_k) > 0
    )
    if not use_prescreen:
        if skip_repair_when_survivors_exist and repair_backend is not None and max_attempts > 0:
            phase1 = list(
                await asyncio.gather(*(_phase1_no_repair(cid, src) for cid, src in candidates))
            )
            if any(chk.passed for chk in phase1):
                return _finalize(phase1)
            phase2 = list(
                await asyncio.gather(*(_phase2_repair_failed(chk) for chk in phase1))
            )
            return _finalize(phase2)
        raw_res = list(await asyncio.gather(*(_check_one(cid, src) for cid, src in candidates)))
        return _finalize(raw_res)

    assert stage_a_cost_ranker is not None
    top_k = int(cost_prescreen_top_k or 0)
    stage_a_outcomes = list(
        await asyncio.gather(*(_run_stage_a_phase(cid, src) for cid, src in candidates))
    )
    results_by_idx: dict[int, L0L1CandidateCheck] = {}
    passers: list[tuple[int, str, str, str, StageAResult, bool, int]] = []
    for idx, ((cid, _src), (failed_chk, passed_state)) in enumerate(
        zip(candidates, stage_a_outcomes)
    ):
        if failed_chk is not None:
            results_by_idx[idx] = failed_chk
        else:
            assert passed_state is not None
            cur_snippet, patched, stage_a, repaired, attempt = passed_state
            passers.append((idx, cid, cur_snippet, patched, stage_a, repaired, attempt))

    if not passers:
        return _finalize([results_by_idx[i] for i in range(len(candidates))])

    creative_id_set = {str(cid) for cid in creative_candidate_ids}
    exempt_passers: list[tuple[int, str, str, str, StageAResult, bool, int]] = []
    same_topo_passers: list[tuple[int, str, str, str, StageAResult, bool, int]] = []
    for item in passers:
        _idx, cid, _snip, patched, stage_a, _rep, _att = item
        cand_axes = (
            candidate_topology_axes_fn(patched, stage_a)
            if candidate_topology_axes_fn is not None
            else _extract_candidate_topology_axes(stage_a)
        )
        cand_ws_stages = (
            int(candidate_workspace_stages_fn(patched, stage_a))
            if candidate_workspace_stages_fn is not None
            else _count_trackb_workspace_stages(patched)
        )
        if is_paradigm_shift_exempt(
            cand_axes,
            parent_topology_axes,
            cand_workspace_stages=cand_ws_stages,
            parent_workspace_stages=eff_parent_ws_stages,
            is_creative=(cid in creative_id_set),
        ):
            exempt_passers.append(item)
        else:
            same_topo_passers.append(item)

    if len(same_topo_passers) > top_k:
        rank_input = [
            (cid, patched, stage_a)
            for _idx, cid, _snip, patched, stage_a, _rep, _att in same_topo_passers
        ]
        ranked_pairs = list(stage_a_cost_ranker(rank_input))
        pred_by_id: dict[str, float] = {}
        order_by_id: dict[str, int] = {}
        for rank_idx, (rcid, rscore) in enumerate(ranked_pairs):
            sid = str(rcid)
            if sid not in pred_by_id:
                pred_by_id[sid] = float(rscore)
                order_by_id[sid] = rank_idx
        sorted_same = sorted(
            same_topo_passers,
            key=lambda item: (
                pred_by_id.get(item[1], float("inf")),
                order_by_id.get(item[1], len(same_topo_passers) + item[0]),
                item[0],
            ),
        )
        kept_same = sorted_same[:top_k]
        reserve_same = sorted_same[top_k:]
    else:
        pred_by_id = {}
        kept_same = same_topo_passers
        reserve_same = []

    kept_passers = [*exempt_passers, *kept_same]
    kept_same_ids = {item[1] for item in kept_same}
    passed_l1 = 0
    if kept_passers:
        stage_b_checks = await asyncio.gather(
            *(
                _run_stage_b_and_repairs(cid, cur_snippet, patched, stage_a, repaired, attempt)
                for _idx, cid, cur_snippet, patched, stage_a, repaired, attempt in kept_passers
            )
        )
        for (idx, cid, *_rest), chk in zip(kept_passers, stage_b_checks):
            results_by_idx[idx] = chk
            if chk.passed and cid in kept_same_ids:
                passed_l1 += 1

    if reserve_same:
        _ = (parent_analytical_us, stage_a_analytical_scorer)
        if passed_l1 < top_k:
            for idx, cid, cur_snippet, patched, stage_a, repaired, attempt in reserve_same:
                if passed_l1 >= top_k:
                    pred_us = pred_by_id.get(cid, float("inf"))
                    results_by_idx[idx] = L0L1CandidateCheck(
                        candidate_id=cid,
                        passed=False,
                        failed_tier="cost_prescreen",
                        patched_module_source=patched,
                        snippet_source=cur_snippet,
                        repaired=repaired,
                        stage_a=stage_a,
                        stage_b=None,
                        error_message=f"cost_prescreen_pruned:pred_us={pred_us:.2f}",
                        fault_plane="COST_PRESCREEN",
                        fault_code="COST_PRESCREEN_PRUNED",
                    )
                    continue
                chk = await _run_stage_b_and_repairs(
                    cid, cur_snippet, patched, stage_a, repaired, attempt
                )
                results_by_idx[idx] = chk
                if chk.passed:
                    passed_l1 += 1
        else:
            for idx, cid, cur_snippet, patched, stage_a, _repaired, _attempt in reserve_same:
                pred_us = pred_by_id.get(cid, float("inf"))
                results_by_idx[idx] = L0L1CandidateCheck(
                    candidate_id=cid,
                    passed=False,
                    failed_tier="cost_prescreen",
                    patched_module_source=patched,
                    snippet_source=cur_snippet,
                    repaired=_repaired,
                    stage_a=stage_a,
                    stage_b=None,
                    error_message=f"cost_prescreen_pruned:pred_us={pred_us:.2f}",
                    fault_plane="COST_PRESCREEN",
                    fault_code="COST_PRESCREEN_PRUNED",
                )

    return _finalize([results_by_idx[i] for i in range(len(candidates))])


def _resolve_agent_gates_bin(source_root: Path | None = None) -> str:
    """Resolve the AgentGates CLI from explicit configuration, PATH, or relative sibling venv."""
    env_bin = os.environ.get("TILEFOUNDRY_NPU_AGENT_GATES_BIN", "").strip()
    if env_bin and Path(env_bin).is_file():
        return env_bin
    on_path = shutil.which("agent-gates")
    if on_path:
        return on_path
    candidates: list[Path] = []
    env_file_raw = os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()
    if env_file_raw:
        env_path = Path(env_file_raw).expanduser().resolve()
        ag_root = env_path.parent
        candidates.append(ag_root / ".venv" / "bin" / "agent-gates")
        if env_path.is_file():
            cfg_bin = (
                parse_simple_env_file(env_path)
                .get("TILEFOUNDRY_NPU_AGENT_GATES_BIN", "")
                .strip()
            )
            if cfg_bin:
                candidates.insert(0, Path(cfg_bin).expanduser())
    if source_root is not None:
        candidates.append(
            Path(source_root).expanduser().resolve().parent
            / "AgentGates"
            / ".venv"
            / "bin"
            / "agent-gates"
        )
    repo_root = Path(__file__).resolve().parents[5]
    candidates.append(repo_root.parent / "AgentGates" / ".venv" / "bin" / "agent-gates")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "Cannot locate 'agent-gates' binary. Set TILEFOUNDRY_NPU_AGENT_GATES_BIN or PATH."
    )


def _resolve_env_file_path(env_file: str | Path | None = None) -> Path:
    raw = (
        str(env_file)
        if env_file is not None
        else os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()
    )
    if raw and Path(raw).is_file():
        return Path(raw).resolve()
    raise FileNotFoundError("Valid tilefoundry-npu env.sh file is required for AgentGates runs.")


def _default_surrogate_cache_dir() -> Path:
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return Path(tempfile.gettempdir()) / f"tf_surrogate_ref_cache_{uid}"


def _default_surrogate_tl_cache_dir() -> Path:
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return Path(tempfile.gettempdir()) / f"tf_surrogate_tl_cache_{uid}"


def _load_ref_prewarm_marker(marker_path: Path) -> dict[str, object] | None:
    if not marker_path.is_file():
        return None
    try:
        raw = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        return None
    return dict(raw) if isinstance(raw, Mapping) else None


def _ref_prewarm_marker_matches(
    marker: Mapping[str, object] | None,
    *,
    op_dir: Path,
    cases_yaml_sha256: str,
    sorted_case_ids: Sequence[int],
) -> bool:
    if marker is None or not (op_dir / "identity.json").is_file():
        return False
    if marker.get("cases_yaml_sha256") != cases_yaml_sha256:
        return False
    raw_case_ids = marker.get("case_ids")
    raw_files = marker.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        return False
    try:
        cached_ids = {int(x) for x in (raw_case_ids or ())}
        if not set(sorted_case_ids).issubset(cached_ids):
            return False
        for fname in raw_files:
            if not isinstance(fname, str) or not fname:
                return False
            fpath = op_dir / fname
            if not fpath.is_file() or fpath.stat().st_size <= 0:
                return False
    except (OSError, ValueError, TypeError):
        return False
    return True


def _write_ref_prewarm_marker(
    marker_path: Path,
    *,
    task: str,
    cases_yaml_sha256: str,
    case_ids: Sequence[int],
    files: Sequence[str],
) -> None:
    payload = {
        "version": 1,
        "task": task,
        "cases_yaml_sha256": cases_yaml_sha256,
        "case_ids": list(case_ids),
        "files": list(files),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(dir=str(marker_path.parent), prefix=".ref_prewarm_v1.")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(encoded)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, marker_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


_REF_CACHE_THREAD_ENV_KEYS: frozenset[str] = frozenset(
    {
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "ATEN_CPU_CAPABILITY",
        "TORCH_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    }
)


def prewarm_anchor_reference_cache(
    operator_name: str,
    anchors: Sequence[AnchorCaseSpec],
    *,
    cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None,
    cache_dir: str | Path | None = None,
    source_root: str | Path | None = None,
) -> Path:
    """Populate CANNBench oracle and native reference cache on Host CPU without NPU locks."""
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    repo = resolve_cannbench_repo(cannbench_repo=cannbench_repo, env_file=env_file)
    resolved_cache = Path(cache_dir) if cache_dir is not None else _default_surrogate_cache_dir()
    resolved_cache.mkdir(parents=True, exist_ok=True)

    try:
        task = get_operator_task(operator_name)
    except (KeyError, ValueError):
        task = ""
    sorted_case_ids = sorted({int(a.case_num) for a in anchors})
    op_dir = resolved_cache / "operators" / task.replace("/", "__") if task else None
    marker_path = op_dir / ".ref_prewarm_v1.json" if op_dir is not None else None
    cases_yaml_sha256 = ""
    if task:
        try:
            cases_path = resolve_cannbench_tasks_root(repo) / task / "cases.yaml"
            if cases_path.is_file():
                cases_yaml_sha256 = _sha256_file(cases_path)
        except (OSError, KeyError, ValueError, RuntimeError):
            cases_yaml_sha256 = ""

    if (
        task
        and sorted_case_ids
        and op_dir is not None
        and marker_path is not None
        and _ref_prewarm_marker_matches(
            _load_ref_prewarm_marker(marker_path),
            op_dir=op_dir,
            cases_yaml_sha256=cases_yaml_sha256,
            sorted_case_ids=sorted_case_ids,
        )
    ):
        return resolved_cache

    env_vals = parse_simple_env_file(env_file) if env_file is not None else {}
    python_bin = (
        env_vals.get("TILEFOUNDRY_NPU_PYTHON")
        or os.environ.get("TILEFOUNDRY_NPU_PYTHON")
        or sys.executable
    )

    run_env = dict(os.environ)
    for _k in _REF_CACHE_THREAD_ENV_KEYS:
        run_env.pop(_k, None)
    run_env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    run_env["CANNBENCH_CACHE_DIR"] = str(resolved_cache)
    run_env["PYTHONDONTWRITEBYTECODE"] = "1"
    pp_parts = [str(root), str(root / "src"), str(root / "targets" / "ascend")]
    if run_env.get("PYTHONPATH"):
        pp_parts.append(run_env["PYTHONPATH"])
    run_env["PYTHONPATH"] = os.pathsep.join(pp_parts)

    case_ids = [int(a.case_num) for a in anchors]
    script = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(root / 'targets' / 'ascend')!r})\n"
        "from tilefoundry_ascend.agent.campaign.anchor_specs import (\n"
        "    AnchorCaseSpec, load_anchor_harness_and_cases, load_cannbench_contract_module\n"
        ")\n"
        f"op_name = {operator_name!r}\n"
        f"case_ids = {json.dumps(case_ids)}\n"
        f"repo = {str(repo)!r}\n"
        f"root = {str(root)!r}\n"
        "contract = load_cannbench_contract_module(root)\n"
        "specs = [AnchorCaseSpec('', cid, '', 1.0, 1.0) for cid in case_ids]\n"
        "harness, cases = load_anchor_harness_and_cases(op_name, specs, cannbench_repo=repo, source_root=root)\n"
        "with contract.ReferenceCache(harness) as cache:\n"
        "    for case in cases:\n"
        "        def _prep(c=case, h=harness):\n"
        "            ts = h.prepare(c)\n"
        "            return {'inputs': ts, 'oracle': h.reference_output(c, ts)}\n"
        "        bundle = cache.get(case, 'oracle', _prep)\n"
        "        cache.get(case, 'native', lambda c=case, b=bundle, h=harness: h.reference_output(c, b['inputs'], native=True))\n"
    )
    proc = subprocess.run(
        [python_bin, "-c", script],
        text=True,
        capture_output=True,
        cwd=str(root),
        env=run_env,
        timeout=180,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-1000:]
        raise RuntimeError(f"prewarm_anchor_reference_cache failed: {detail}")

    if (
        task
        and sorted_case_ids
        and op_dir is not None
        and marker_path is not None
        and (op_dir / "identity.json").is_file()
    ):
        pt_files = sorted(
            {
                p.name
                for pattern in ("*.oracle.pt", "*.native.pt")
                for p in op_dir.glob(pattern)
                if p.is_file() and p.stat().st_size > 0
            }
        )
        if pt_files:
            merged_ids = set(sorted_case_ids)
            existing = _load_ref_prewarm_marker(marker_path)
            if existing is not None and existing.get("cases_yaml_sha256") == cases_yaml_sha256:
                try:
                    merged_ids.update(int(x) for x in (existing.get("case_ids") or ()))
                except (ValueError, TypeError):
                    pass
            _write_ref_prewarm_marker(
                marker_path,
                task=task,
                cases_yaml_sha256=cases_yaml_sha256,
                case_ids=sorted(merged_ids),
                files=pt_files,
            )
    return resolved_cache


_PREWARM_TL_PLAN_TEMPLATE = """import importlib.util
import inspect
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, {ascend_target_dir!r})
from tilefoundry_ascend.agent.campaign.anchor_specs import (
    discover_module_trackb_symbols,
    extract_trackb_cpp_source,
    get_default_regime_specs,
    get_operator_entrypoint,
    infer_module_runtime_backend,
    resolve_regime_for_case,
)
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    _HIR_DIM_ALIASES,
    _filter_callable_kwargs,
    resolve_regime_config,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    _RUNTIME_ONLY_TRACKB_DIM_KEYS,
    lookup_case_static_spec,
)
from tilefoundry_ascend.agent.orchestrator.cpp_slice_protocol import (
    extract_reachable_trackb_source,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    _normalize_trackb_unit_source,
    resolve_trackb_compile_flags,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s:%(name)s:%(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("tilefoundry_ascend.prewarm")

def _json_safe(val):
    if isinstance(val, dict):
        return {{str(k): _json_safe(v) for k, v in sorted(val.items())}}
    if isinstance(val, (list, tuple)):
        return [_json_safe(x) for x in val]
    if isinstance(val, set):
        return sorted((_json_safe(x) for x in val), key=lambda x: json.dumps(x, sort_keys=True))
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val
    return str(val)

op_name = {op_name!r}
case_ids = {case_ids_json}
cand_paths = {cand_paths_json}
repo = {repo!r}
anchor_regimes = {anchor_regimes_json}

regime_specs = get_default_regime_specs(op_name)
case_specs = []
for cid_num in case_ids:
    try:
        case_dims = lookup_case_static_spec(op_name, cid_num, cannbench_repo=repo)
        regime = resolve_regime_for_case(op_name, case_dims, regime_specs)
        case_specs.append((cid_num, case_dims, regime))
    except Exception as exc:
        logger.warning(
            "Prewarm static spec skipped for op=%s case=%s: %s: %s",
            op_name,
            cid_num,
            type(exc).__name__,
            exc,
        )

tasks = []
errors = []
cand_errors = []
entrypoint = get_operator_entrypoint(op_name)

for idx, (cid, mod_path) in enumerate(cand_paths):
    try:
        mod_src = Path(mod_path).read_text(encoding="utf-8")
        mod_name = f"_tf_warm_plan_{{idx}}"
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load module spec from {{mod_path}}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod_backend, mod_tb_syms = infer_module_runtime_backend(mod_src, entrypoint)
    except Exception as exc:
        msg = f"candidate={{cid}} module_load_failed: {{type(exc).__name__}}: {{exc}}"
        errors.append(msg)
        cand_errors.append({{"candidate_id": cid, "case_num": -1, "error": msg}})
        continue

    get_kernel_fn = getattr(mod, "_get_kernel", None)
    combine_fn = getattr(mod, "_get_combine_kernel", None)
    render_tb_fn = getattr(mod, "_render_trackb_tu", None)

    for cid_num, case_dims, regime in case_specs:
        rt_backend = str(getattr(regime, "runtime_backend", "") or "")
        r_tag = str(anchor_regimes.get(str(cid_num)) or getattr(regime, "regime_tag", "") or "")
        tb_syms = list(getattr(regime, "trackb_symbols", ()) or mod_tb_syms or ())
        try:
            pool = dict(getattr(regime, "sample_builder_kwargs", {{}}))
            for h_key, h_val in getattr(regime, "representative_hir_dims", {{}}).items():
                mapped = _HIR_DIM_ALIASES.get(str(h_key), str(h_key))
                pool.setdefault(mapped, h_val)
            for d_key, d_val in case_dims.items():
                mapped = _HIR_DIM_ALIASES.get(str(d_key), str(d_key))
                pool[mapped] = d_val
            raw_bk = {{}}
            raw_ck = {{}}
            pc_safe = {{}}
            cf_list = []
            sel_name = getattr(regime, "config_selector_fn", None)
            if sel_name and callable(getattr(mod, sel_name, None)):
                sel_fn = getattr(mod, sel_name)
                sel_out = sel_fn(**_filter_callable_kwargs(sel_fn, pool))
                if isinstance(sel_out, dict):
                    raw_pc = sel_out.get("pass_configs")
                    if isinstance(raw_pc, dict):
                        for pk, pv in raw_pc.items():
                            if pk == "tl.device_compile_flags":
                                cf_list.extend(str(f) for f in (pv if isinstance(pv, (list, tuple)) else [pv]))
                            else:
                                pc_safe[str(pk)] = pv
                    raw_cf = sel_out.get("compile_flags") or ()
                    if isinstance(raw_cf, str):
                        cf_list.extend(raw_cf.split())
                    else:
                        cf_list.extend(str(f) for f in raw_cf if str(f).strip())
                    maybe_bk = sel_out.get("builder_kwargs")
                    if isinstance(maybe_bk, dict):
                        raw_bk = dict(maybe_bk)
                    maybe_ck = sel_out.get("combine_kwargs")
                    if isinstance(maybe_ck, dict):
                        raw_ck = dict(maybe_ck)
                    for sk, sv in sel_out.items():
                        if sk not in (
                            "pass_configs",
                            "compile_flags",
                            "builder_kwargs",
                            "combine_kwargs",
                        ):
                            pool[str(sk)] = sv

            is_pure_tb = rt_backend in ("ascendc_trackb", "ascendc") or mod_backend == "ascendc_trackb"
            has_tb_cpp = bool(callable(render_tb_fn) or tb_syms or discover_module_trackb_symbols(mod_src))
            is_hybrid_tb = (
                rt_backend == "hybrid_trackb" or mod_backend == "hybrid_trackb"
            ) and has_tb_cpp

            resolved = None
            builder_fn = None
            try:
                resolved = resolve_regime_config(mod_path, regime, dims=case_dims)
                builder_fn = getattr(mod, resolved.builder_name, None)
                pool.update(resolved.builder_kwargs)
                if isinstance(resolved.pass_configs, dict):
                    pc_safe = dict(resolved.pass_configs)
                if resolved.compile_flags:
                    cf_list = [str(f) for f in resolved.compile_flags]
            except Exception:
                if not (is_pure_tb or is_hybrid_tb):
                    raise

            tb_only_builder = bool(
                resolved is not None
                and resolved.builder_name == sel_name
                and has_tb_cpp
            )
            if is_pure_tb or (is_hybrid_tb and (builder_fn is None or tb_only_builder)):
                tb_builder_name = "_render_trackb_tu" if callable(render_tb_fn) else (
                    resolved.builder_name if (resolved is not None and callable(builder_fn)) else "_trackb_tu"
                )
                cpp_src = ""
                if callable(render_tb_fn):
                    cpp_src = str(render_tb_fn(**_filter_callable_kwargs(render_tb_fn, pool)))
                if not cpp_src.strip() and callable(builder_fn):
                    maybe = builder_fn(**_filter_callable_kwargs(builder_fn, pool))
                    if isinstance(maybe, str) and ("__global__" in maybe or "#include" in maybe):
                        cpp_src = maybe
                if not cpp_src.strip():
                    syms = tb_syms or list(discover_module_trackb_symbols(mod_src))
                    extracted = extract_trackb_cpp_source(mod_src, syms)
                    if extracted is not None:
                        _kname, cpp_src = extract_reachable_trackb_source(
                            extracted[1],
                            module_source=mod_src,
                            regime_tag=r_tag,
                        )
                eff_flags = (
                    list(resolve_trackb_compile_flags(cpp_src, cf_list))
                    if cpp_src.strip()
                    else list(cf_list)
                )
                norm_unit = (
                    _normalize_trackb_unit_source(cpp_src, regime_tag=r_tag)
                    if cpp_src.strip()
                    else ""
                )
                static_dims = {{
                    str(dk): dv
                    for dk, dv in pool.items()
                    if str(dk) not in _RUNTIME_ONLY_TRACKB_DIM_KEYS
                }}
                tasks.append({{
                    "candidate_id": cid,
                    "case_num": int(cid_num),
                    "mod_path": mod_path,
                    "mode": "trackb",
                    "builder_name": tb_builder_name,
                    "kwargs": {{
                        "dims": _json_safe(static_dims),
                        "regime_tag": r_tag,
                        "trackb_symbols": tb_syms,
                    }},
                    "cpp_src": cpp_src,
                    "normalized_cpp_unit": norm_unit,
                    "pass_configs": _json_safe(pc_safe),
                    "compile_flags": eff_flags,
                }})
                continue

            if not callable(builder_fn):
                raise ValueError(f"Builder {{resolved.builder_name if resolved else 'missing'!r}} not callable")

            pool["builder_fn"] = builder_fn
            pool["pass_configs"] = pc_safe
            pool["compile_flags"] = tuple(cf_list)
            pool["extra_builder_kwargs"] = dict(raw_bk) if raw_bk else None
            pool["extra_combine_kwargs"] = dict(raw_ck) if raw_ck else None

            used_get_kernel = False
            if callable(get_kernel_fn):
                gk_sig = inspect.signature(get_kernel_fn)
                var_kinds = (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
                eff_sel = sel_name or (
                    getattr(resolved, "selector_name", "") if resolved is not None else ""
                )
                for src_fn in (
                    getattr(mod, entrypoint, None) if entrypoint else None,
                    getattr(mod, "kernel", None),
                    getattr(mod, eff_sel, None) if eff_sel else None,
                    builder_fn,
                ):
                    if callable(src_fn):
                        try:
                            for p in inspect.signature(src_fn).parameters.values():
                                if (
                                    p.default is not inspect.Parameter.empty
                                    and p.kind not in var_kinds
                                    and p.name not in pool
                                ):
                                    pool[p.name] = p.default
                        except (TypeError, ValueError):
                            pass
                has_req = all(
                    p.name in pool
                    for p in gk_sig.parameters.values()
                    if p.default is inspect.Parameter.empty
                    and p.kind not in var_kinds
                )
                if has_req:
                    gk_kwargs = _filter_callable_kwargs(get_kernel_fn, pool)
                    serial_gk = {{
                        str(k): _json_safe(v)
                        for k, v in sorted(gk_kwargs.items())
                        if k != "builder_fn"
                    }}
                    tasks.append({{
                        "candidate_id": cid,
                        "case_num": int(cid_num),
                        "mod_path": mod_path,
                        "mode": "_get_kernel",
                        "builder_name": resolved.builder_name,
                        "kwargs": serial_gk,
                        "pass_configs": _json_safe(pc_safe),
                        "compile_flags": cf_list,
                    }})
                    splits = int(pool.get("num_kv_splits", 1) or 1)
                    if splits > 1 and callable(combine_fn):
                        comb_kwargs = _filter_callable_kwargs(combine_fn, pool)
                        serial_comb = {{
                            str(k): _json_safe(v)
                            for k, v in sorted(comb_kwargs.items())
                        }}
                        tasks.append({{
                            "candidate_id": cid,
                            "case_num": int(cid_num),
                            "mod_path": mod_path,
                            "mode": "_get_combine_kernel",
                            "builder_name": "_get_combine_kernel",
                            "kwargs": serial_comb,
                            "pass_configs": _json_safe(pc_safe),
                            "compile_flags": cf_list,
                        }})
                    used_get_kernel = True

            if not used_get_kernel:
                generic_pool = dict(resolved.builder_kwargs)
                generic_pool.update(raw_bk)
                b_kwargs = _filter_callable_kwargs(builder_fn, generic_pool)
                serial_bk = {{
                    str(k): _json_safe(v)
                    for k, v in sorted(b_kwargs.items())
                }}
                tasks.append({{
                    "candidate_id": cid,
                    "case_num": int(cid_num),
                    "mod_path": mod_path,
                    "mode": "generic",
                    "builder_name": resolved.builder_name,
                    "kwargs": serial_bk,
                    "pass_configs": _json_safe(pc_safe),
                    "compile_flags": cf_list,
                }})
        except Exception as exc:
            msg = f"candidate={{cid}} case={{cid_num}} plan_failed: {{type(exc).__name__}}: {{exc}}"
            errors.append(msg)
            cand_errors.append({{"candidate_id": cid, "case_num": int(cid_num), "error": msg}})

print(json.dumps({{"tasks": tasks, "errors": errors, "cand_errors": cand_errors}}), flush=True)
"""


_PREWARM_TL_WORKER_TEMPLATE = """import hashlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
import tilelang

task = json.loads({task_json!r})
spec = importlib.util.spec_from_file_location("_tf_warm_exec", task["mod_path"])
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load module spec from {{task['mod_path']}}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mode = task["mode"]
kwargs = dict(task["kwargs"])
compiled_obj = None
if mode == "_get_kernel":
    fn = getattr(mod, "_get_kernel")
    if "compile_flags" in kwargs and isinstance(kwargs["compile_flags"], list):
        kwargs["compile_flags"] = tuple(kwargs["compile_flags"])
    fn_params = inspect.signature(fn).parameters
    has_varkw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in fn_params.values()
    )
    if "builder_fn" in fn_params or has_varkw:
        kwargs["builder_fn"] = getattr(mod, task["builder_name"])
    else:
        kwargs.pop("builder_fn", None)
    compiled_obj = fn(**kwargs)
elif mode == "_get_combine_kernel":
    fn = getattr(mod, "_get_combine_kernel")
    if "compile_flags" in kwargs and isinstance(kwargs["compile_flags"], list):
        kwargs["compile_flags"] = tuple(kwargs["compile_flags"])
    compiled_obj = fn(**kwargs)
elif mode == "trackb":
    from tilefoundry_ascend.agent.campaign.anchor_specs import (
        discover_module_trackb_symbols,
        extract_trackb_cpp_source,
    )
    from tilefoundry_ascend.agent.campaign.campaign_runner import _filter_callable_kwargs
    from tilefoundry_ascend.agent.orchestrator.cpp_slice_protocol import (
        extract_reachable_trackb_source,
    )
    from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
        resolve_trackb_compile_flags,
        run_stage_b_ascendc_compile,
    )
    from tilefoundry_ascend.ascend.compile_cache.artifact_store import ArtifactStore

    cache_dir = Path(os.environ["TILELANG_CACHE_DIR"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    dims_pool = dict(kwargs.get("dims") or {{}})
    regime_tag = str(kwargs.get("regime_tag") or "")
    tb_syms = list(kwargs.get("trackb_symbols") or [])
    mod_src = Path(task["mod_path"]).read_text(encoding="utf-8")
    cpp_src = str(task.get("cpp_src") or "")
    if not cpp_src.strip():
        render_fn = getattr(mod, "_render_trackb_tu", None)
        if callable(render_fn):
            cpp_src = str(render_fn(**_filter_callable_kwargs(render_fn, dims_pool)))
    if not cpp_src.strip():
        builder_fn = getattr(mod, task.get("builder_name", ""), None)
        if callable(builder_fn):
            maybe = builder_fn(**_filter_callable_kwargs(builder_fn, dims_pool))
            if isinstance(maybe, str) and ("__global__" in maybe or "#include" in maybe):
                cpp_src = maybe
    if not cpp_src.strip():
        syms = tb_syms or list(discover_module_trackb_symbols(mod_src))
        extracted = extract_trackb_cpp_source(mod_src, syms)
        if extracted is not None:
            _kname, cpp_src = extract_reachable_trackb_source(
                extracted[1],
                module_source=mod_src,
                regime_tag=regime_tag,
            )
    if not cpp_src.strip():
        raise RuntimeError("trackb prewarm could not extract C++ translation unit")
    cf_list = list(resolve_trackb_compile_flags(cpp_src, task.get("compile_flags") or []))
    task_sig = hashlib.sha256(
        (cpp_src + "|" + " ".join(cf_list)).encode("utf-8")
    ).hexdigest()[:16]
    compile_ascend_fn = getattr(tilelang, "compile_ascend", None)
    if not callable(compile_ascend_fn):
        try:
            from tilelang.contrib.bisheng import compile_ascend as compile_ascend_fn
        except Exception:
            compile_ascend_fn = None
    if callable(compile_ascend_fn):
        so_path = cache_dir / f"trackb_{{task_sig}}.so"
        compile_ascend_fn(
            cpp_src,
            target_format="so",
            npu_arch="dav-3510",
            options=cf_list or None,
            path_target=str(so_path),
        )
        if not so_path.is_file() or so_path.stat().st_size <= 0:
            raise RuntimeError(f"compile_ascend produced no valid binary at {{so_path}}")
        compiled_obj = str(so_path)
    else:
        sb_res = run_stage_b_ascendc_compile(
            cpp_src,
            compile_flags=cf_list,
            artifact_store=ArtifactStore(cache_dir),
            artifact_id=f"trackb_{{task_sig}}",
            regime_tag=regime_tag,
        )
        if not sb_res.passed:
            raise RuntimeError(sb_res.error_message or "trackb stage_b compile failed")
        compiled_obj = list(sb_res.binary_paths)
else:
    builder_fn = getattr(mod, task["builder_name"])
    prog = builder_fn(**kwargs)
    pc_dict = task.get("pass_configs") or {{}}
    cf_list = task.get("compile_flags") or []
    compiled_obj = tilelang.compile(
        prog,
        target="ascend",
        pass_configs=pc_dict or None,
        compile_flags=list(cf_list) if cf_list else None,
    )

emit_bins_raw = task.get("emit_cache_bins_path")
if emit_bins_raw:
    found_bins = []
    seen_bins = set()
    def _add_bin(raw_p):
        if not raw_p:
            return
        try:
            p_obj = Path(str(raw_p))
            if not p_obj.is_file():
                return
            if p_obj.suffix not in (".so", ".o", ".bin") and ".so" not in p_obj.suffixes:
                return
            if p_obj.stat().st_size <= 0:
                return
            rp = str(p_obj.resolve())
        except OSError:
            return
        if rp not in seen_bins:
            seen_bins.add(rp)
            found_bins.append(rp)
    objs = list(compiled_obj) if isinstance(compiled_obj, (tuple, list)) else [compiled_obj]
    for cur in list(objs):
        if cur is None:
            continue
        for sa in ("adapter", "lib", "kernel_lib", "artifact"):
            sub = getattr(cur, sa, None)
            if sub is not None:
                objs.append(sub)
    tl_cache_env = os.environ.get("TILELANG_CACHE_DIR", "")
    tl_cache_res = str(Path(tl_cache_env).resolve()) if tl_cache_env else ""
    for cur in objs:
        if cur is None:
            continue
        if isinstance(cur, (str, Path)):
            _add_bin(cur)
            continue
        if isinstance(cur, dict):
            for v in cur.values():
                if isinstance(v, (str, Path)):
                    _add_bin(v)
            continue
        for attr in (
            "kernel_path", "libpath", "lib_path", "so_path",
            "binary_path", "artifact_path", "path", "_name",
            "_tilelang_cache_path",
        ):
            attr_val = getattr(cur, attr, None)
            _add_bin(attr_val)
            if attr == "_tilelang_cache_path" and attr_val:
                try:
                    tdir = Path(str(attr_val))
                    if tdir.is_dir():
                        for cand in sorted(tdir.rglob("*")):
                            _add_bin(cand)
                except OSError:
                    pass
        ckey = getattr(cur, "cache_key", None) or getattr(
            cur, "_tilelang_cache_key", None
        )
        if ckey and tl_cache_res:
            cdir = Path(tl_cache_res)
            kdir = cdir / str(ckey)
            if kdir.is_dir():
                for cand in sorted(kdir.rglob("*")):
                    _add_bin(cand)
            for cand in sorted(cdir.glob(f"*{{ckey}}*")):
                _add_bin(cand)
        try:
            obj_vars = vars(cur)
        except TypeError:
            obj_vars = None
        if isinstance(obj_vars, dict) and tl_cache_res:
            for v in obj_vars.values():
                if isinstance(v, (str, Path)):
                    try:
                        vp = str(Path(str(v)).resolve())
                        if vp.startswith(tl_cache_res):
                            _add_bin(vp)
                    except OSError:
                        pass
    emit_p = Path(str(emit_bins_raw))
    emit_p.parent.mkdir(parents=True, exist_ok=True)
    emit_p.write_text(json.dumps(found_bins), encoding="utf-8")
"""


_PREWARM_MANIFEST_FILENAME = "prewarm_manifest_v1.json"
_INCUMBENT_CANDIDATE_IDS = frozenset({"incumbent_slot", "__incumbent__", "incumbent_baseline"})


@dataclass(frozen=True)
class PrewarmTileLangResult:
    """Result and cache telemetry from content-addressed TileLang prewarming."""

    tl_cache_dir: Path
    prewarm_stats: dict[str, int]
    failed_by_cid: dict[str, str]
    elapsed_sec: float


def _compute_tilelang_lib_stamp(env_vals: Mapping[str, str]) -> str:
    """Derive a fast stamp from loaded or configured TileLang libraries."""
    parts: list[str] = []
    tl_build = (
        env_vals.get("TILEFOUNDRY_NPU_TILELANG_BUILD")
        or os.environ.get("TILEFOUNDRY_NPU_TILELANG_BUILD")
        or ""
    ).strip()
    tl_source = (
        env_vals.get("TILEFOUNDRY_NPU_TILELANG_SOURCE")
        or os.environ.get("TILEFOUNDRY_NPU_TILELANG_SOURCE")
        or ""
    ).strip()

    lib_dirs: list[Path] = []
    if tl_build:
        b_path = Path(tl_build)
        lib_dirs.extend([b_path / "lib", b_path])
    if tl_source:
        s_path = Path(tl_source)
        lib_dirs.extend([s_path / "build" / "lib", s_path / "tilelang" / "lib"])
        for marker in (s_path / "tilelang.py", s_path / "tilelang" / "__init__.py"):
            if marker.is_file():
                try:
                    st = marker.stat()
                    parts.append(f"src:{marker.resolve()}:{st.st_mtime_ns}:{st.st_size}")
                except OSError:
                    pass

    seen_so: set[Path] = set()
    for lib_dir in lib_dirs:
        if not lib_dir.is_dir():
            continue
        so_files = sorted(lib_dir.glob("*.so*"))
        for so_path in so_files:
            resolved_so = so_path.resolve()
            if resolved_so in seen_so or not resolved_so.is_file():
                continue
            seen_so.add(resolved_so)
            try:
                st = resolved_so.stat()
                parts.append(f"so:{resolved_so.name}:{st.st_mtime_ns}:{st.st_size}")
            except OSError:
                continue
        if seen_so:
            break

    if not parts:
        mod = sys.modules.get("tilelang")
        origin = getattr(mod, "__file__", None) if mod is not None else None
        if origin is None:
            try:
                spec = importlib.util.find_spec("tilelang")
                origin = spec.origin if spec is not None else None
            except (ImportError, ValueError, AttributeError):
                origin = None
        if mod is not None:
            ver = getattr(mod, "__version__", "")
            if ver:
                parts.append(f"ver:{ver}")
        if origin:
            origin_path = Path(origin).resolve()
            if origin_path.is_file():
                try:
                    st = origin_path.stat()
                    parts.append(f"mod:{origin_path}:{st.st_mtime_ns}:{st.st_size}")
                except OSError:
                    pass
                pkg_dir = origin_path.parent
                for sub_dir in (pkg_dir / "lib", pkg_dir.parent / "build" / "lib"):
                    if sub_dir.is_dir():
                        for so_path in sorted(sub_dir.glob("*.so*")):
                            if so_path.is_file():
                                try:
                                    st = so_path.stat()
                                    parts.append(f"so:{so_path.name}:{st.st_mtime_ns}:{st.st_size}")
                                except OSError:
                                    continue
                        break

    if not parts:
        return "none"
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


_ARTIFACT_STORE_INTERNAL_FILES = frozenset(
    {
        "artifact_manifest_v1.json",
        "artifact_index_v1.jsonl",
        ".artifact_store_v1.lock",
    }
)


def _load_prewarm_manifest(manifest_path: Path) -> dict[str, dict[str, object]]:
    if not manifest_path.is_file():
        return {"tasks": {}, "plan_index": {}}
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {"tasks": {}, "plan_index": {}}
    if not isinstance(raw, Mapping):
        return {"tasks": {}, "plan_index": {}}
    raw_tasks = raw.get("tasks")
    raw_plan = raw.get("plan_index")
    return {
        "tasks": dict(raw_tasks) if isinstance(raw_tasks, Mapping) else {},
        "plan_index": dict(raw_plan) if isinstance(raw_plan, Mapping) else {},
    }


def _save_prewarm_manifest(
    manifest_path: Path,
    tasks_map: Mapping[str, object],
    plan_index_map: Mapping[str, object],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "tasks": dict(tasks_map),
        "plan_index": dict(plan_index_map),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(dir=str(manifest_path.parent), prefix=".prewarm_manifest_v1.")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(encoded)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, manifest_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def lookup_case_static_spec(
    operator_name: str,
    case_num: int,
    *,
    cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None,
) -> dict[str, object]:
    """Read static dims, tl_dtype, and attributes directly from cases.yaml."""
    repo = resolve_cannbench_repo(cannbench_repo=cannbench_repo, env_file=env_file)
    task = get_operator_task(operator_name)
    cases_path = resolve_cannbench_tasks_root(repo) / task / "cases.yaml"
    if not cases_path.is_file():
        raise FileNotFoundError(f"cases.yaml not found at {cases_path}")
    parsed = yaml.safe_load(cases_path.read_text(encoding="utf-8")) or {}
    cases_list = parsed.get("cases") or []
    target_id = int(case_num)
    entry: Mapping[str, object] | None = None
    for row in cases_list:
        if isinstance(row, Mapping) and int(row.get("case_id", -1)) == target_id:
            entry = row
            break
    if entry is None:
        raise KeyError(f"case_num={target_id} missing in {cases_path}")

    spec = lookup_case_dims(
        operator_name,
        target_id,
        cannbench_repo=repo,
        env_file=env_file,
    )
    raw_dtype = entry.get("dtype", "bfloat16")
    if isinstance(raw_dtype, (list, tuple)):
        tl_dtype = str(raw_dtype[0]) if raw_dtype else "bfloat16"
    else:
        tl_dtype = str(raw_dtype)
    tl_dtype = tl_dtype.strip() or "bfloat16"
    if tl_dtype.startswith("torch."):
        tl_dtype = tl_dtype[len("torch.") :]
    spec["tl_dtype"] = tl_dtype
    if "inputLayout" in spec and "input_layout" not in spec:
        spec["input_layout"] = str(spec["inputLayout"]).upper()
    if "is_causal" in spec:
        spec["is_causal"] = bool(spec["is_causal"])
    if (
        "B" in spec
        and "S" in spec
        and "m_rows" not in spec
        and isinstance(spec["B"], int)
        and isinstance(spec["S"], int)
    ):
        spec["m_rows"] = int(spec["B"]) * int(spec["S"])
    return spec


def _build_prewarm_plan_key(
    *,
    operator_name: str,
    cand_sha256: str,
    case_num: int,
    cases_yaml_sha256: str,
    tilelang_lib_stamp: str,
) -> str:
    raw = f"{operator_name}|{cand_sha256}|{case_num}|{cases_yaml_sha256}|{tilelang_lib_stamp}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def prewarm_surrogate_tilelang_cache_with_stats(
    operator_name: str,
    candidates: Sequence[tuple[str, str | Path]],
    anchors: Sequence[AnchorCaseSpec],
    *,
    cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None,
    cache_dir: str | Path | None = None,
    tl_cache_dir: str | Path | None = None,
    source_root: str | Path | None = None,
    timeout_sec: int = 300,
    isolate_non_incumbent: bool = True,
) -> PrewarmTileLangResult:
    """Closure-addressed TileLang and Track-B prewarm with binary ArtifactStore checks."""
    t0 = time.monotonic()
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    repo = resolve_cannbench_repo(cannbench_repo=cannbench_repo, env_file=env_file)
    resolved_ref_cache = Path(cache_dir) if cache_dir is not None else _default_surrogate_cache_dir()
    resolved_ref_cache.mkdir(parents=True, exist_ok=True)
    env_vals = _resolve_env_values(env_file)
    resolved_tl_cache, artifact_store = _resolve_l1_cache_store(
        env_vals, tl_cache_dir, None, root
    )
    resolved_tl_cache.mkdir(parents=True, exist_ok=True)
    if not candidates or not anchors:
        return PrewarmTileLangResult(
            tl_cache_dir=resolved_tl_cache,
            prewarm_stats={
                "total_tasks": 0,
                "cache_hits": 0,
                "compiled_tasks": 0,
                "failed_candidates": 0,
            },
            failed_by_cid={},
            elapsed_sec=time.monotonic() - t0,
        )

    tl_lib_stamp = _compute_tilelang_lib_stamp(env_vals)
    cases_yaml_sha256 = ""
    try:
        cases_path = resolve_cannbench_tasks_root(repo) / get_operator_task(operator_name) / "cases.yaml"
        if cases_path.is_file():
            cases_yaml_sha256 = _sha256_file(cases_path)
    except (OSError, ValueError, KeyError, RuntimeError):
        cases_yaml_sha256 = ""

    manifest_path = resolved_tl_cache / _PREWARM_MANIFEST_FILENAME
    manifest = _load_prewarm_manifest(manifest_path)
    manifest_tasks: dict[str, object] = dict(manifest["tasks"])
    manifest_plan: dict[str, object] = dict(manifest["plan_index"])

    failed_by_cid: dict[str, str] = {}
    cand_sources: dict[str, str] = {}
    cand_resolved_paths: dict[str, str] = {}
    valid_candidates: list[tuple[str, str]] = []
    for raw_cid, raw_path in candidates:
        cid = str(raw_cid)
        try:
            resolved_p = Path(raw_path).resolve()
            cand_sources[cid] = resolved_p.read_text(encoding="utf-8")
            cand_resolved_paths[cid] = str(resolved_p)
            valid_candidates.append((cid, str(resolved_p)))
        except OSError as exc:
            err = f"candidate={cid} module_read_failed: {type(exc).__name__}: {exc}"
            if cid == "incumbent_slot" or not isolate_non_incumbent:
                raise RuntimeError(f"prewarm_surrogate_tilelang_cache failed: {err}") from exc
            failed_by_cid[cid] = err

    case_ids = [int(a.case_num) for a in anchors]
    anchor_regimes = {str(int(a.case_num)): str(a.regime_tag or "") for a in anchors}
    case_closure_srcs: dict[tuple[str, int], str] = {}
    case_closure_shas: dict[tuple[str, int], str] = {}
    for cid, _mod_path_str in valid_candidates:
        mod_src = cand_sources[cid]
        for anchor in anchors:
            cnum = int(anchor.case_num)
            rtag = str(anchor.regime_tag or "")
            c_src = extract_reachable_closure_source(mod_src, regime_tag=rtag)
            c_id = compute_compile_task_key(
                normalized_kernel_unit_src=c_src,
                cann_target="dav-3510",
                pass_configs=None,
                compile_flags=(),
                static_spec_signature=f"{operator_name}:{rtag}:{cases_yaml_sha256}",
                compiler_stamp=tl_lib_stamp,
            )
            case_closure_srcs[(cid, cnum)] = c_src
            case_closure_shas[(cid, cnum)] = c_id

    cache_hits = 0
    compiled_tasks = 0
    failed_task_count = 0
    missed_cands: dict[str, str] = {}
    missed_cases_by_cid: dict[str, set[int]] = {}
    hit_cases_by_cid: dict[str, set[int]] = {}

    for cid, mod_path_str in valid_candidates:
        for cnum in case_ids:
            c_sha = case_closure_shas[(cid, cnum)]
            p_key = _build_prewarm_plan_key(
                operator_name=operator_name,
                cand_sha256=c_sha,
                case_num=cnum,
                cases_yaml_sha256=cases_yaml_sha256,
                tilelang_lib_stamp=tl_lib_stamp,
            )
            entry_keys = manifest_plan.get(p_key)
            if (
                isinstance(entry_keys, list)
                and len(entry_keys) > 0
                and all(
                    isinstance(tk, str)
                    and tk in manifest_tasks
                    and artifact_store.has_verified_artifact(tk)
                    for tk in entry_keys
                )
            ):
                cache_hits += len(entry_keys)
                hit_cases_by_cid.setdefault(cid, set()).add(cnum)
            else:
                missed_cands[cid] = mod_path_str
                missed_cases_by_cid.setdefault(cid, set()).add(cnum)

    if not missed_cands:
        stats = {
            "total_tasks": cache_hits,
            "cache_hits": cache_hits,
            "compiled_tasks": 0,
            "failed_candidates": len(failed_by_cid),
        }
        return PrewarmTileLangResult(
            tl_cache_dir=resolved_tl_cache,
            prewarm_stats=stats,
            failed_by_cid=failed_by_cid,
            elapsed_sec=time.monotonic() - t0,
        )

    python_bin = (
        env_vals.get("TILEFOUNDRY_NPU_PYTHON")
        or os.environ.get("TILEFOUNDRY_NPU_PYTHON")
        or sys.executable
    )
    cann_setup = (
        env_vals.get("TILEFOUNDRY_NPU_CANN_SETUP")
        or os.environ.get("TILEFOUNDRY_NPU_CANN_SETUP")
        or ""
    ).strip()
    tl_source = (
        env_vals.get("TILEFOUNDRY_NPU_TILELANG_SOURCE")
        or os.environ.get("TILEFOUNDRY_NPU_TILELANG_SOURCE")
        or ""
    ).strip()

    run_env = dict(os.environ)
    run_env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    run_env["ASCEND_RT_VISIBLE_DEVICES"] = ""
    run_env["USE_CUDA"] = "0"
    run_env["CANNBENCH_CACHE_DIR"] = str(resolved_ref_cache)
    run_env["TILELANG_CACHE_DIR"] = str(resolved_tl_cache)
    run_env["TILELANG_KERNEL_CACHE_USE_LIB_STAMP"] = "1"
    run_env["TILELANG_DISABLE_CACHE"] = "0"
    run_env["PYTHONDONTWRITEBYTECODE"] = "1"
    pp_parts = [str(root), str(root / "src"), str(root / "targets" / "ascend")]
    if tl_source:
        pp_parts.insert(0, tl_source)
    if run_env.get("PYTHONPATH"):
        pp_parts.append(run_env["PYTHONPATH"])
    run_env["PYTHONPATH"] = os.pathsep.join(pp_parts)

    need_cann_source = bool(
        cann_setup and Path(cann_setup).is_file() and shutil.which("bisheng") is None
    )

    def _build_argv(script_path: Path) -> list[str]:
        if need_cann_source:
            bash_cmd = (
                f"set +u && source {shlex.quote(cann_setup)} >/dev/null 2>&1 "
                f"&& exec {shlex.quote(python_bin)} {shlex.quote(str(script_path))}"
            )
            return ["/bin/bash", "-c", bash_cmd]
        return [python_bin, str(script_path)]

    missed_cand_paths = [
        (cid, missed_cands[cid])
        for cid, _p in valid_candidates
        if cid in missed_cands
    ]
    missed_case_ids = [
        cnum
        for cnum in case_ids
        if any(cnum in missed_cases_by_cid.get(cid, set()) for cid in missed_cands)
    ]

    with tempfile.TemporaryDirectory(prefix="tf_tl_prewarm_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        plan_path = tmp_root / "prewarm_tl_plan.py"
        plan_path.write_text(
            _PREWARM_TL_PLAN_TEMPLATE.format(
                ascend_target_dir=str(root / "targets" / "ascend"),
                op_name=operator_name,
                case_ids_json=json.dumps(missed_case_ids),
                cand_paths_json=json.dumps(missed_cand_paths),
                repo=str(repo),
                anchor_regimes_json=json.dumps(anchor_regimes),
            ),
            encoding="utf-8",
        )
        plan_proc = subprocess.run(
            _build_argv(plan_path),
            text=True,
            capture_output=True,
            cwd=str(root),
            env=run_env,
            timeout=timeout_sec,
        )
        if plan_proc.returncode != 0:
            detail = (plan_proc.stderr or plan_proc.stdout or "").strip()[-1000:]
            raise RuntimeError(f"prewarm_surrogate_tilelang_cache failed: {detail}")
        if plan_proc.stderr and plan_proc.stderr.strip():
            logging.getLogger(__name__).warning(
                "prewarm_surrogate_tilelang_cache warnings: %s",
                plan_proc.stderr.strip()[-1000:],
            )

        plan_payload = json.loads((plan_proc.stdout or "{}").strip().splitlines()[-1])
        cand_errors_raw = plan_payload.get("cand_errors") or ()
        plan_errors = [str(e) for e in plan_payload.get("errors", ())]
        if cand_errors_raw:
            for item in cand_errors_raw:
                if not isinstance(item, Mapping):
                    continue
                err_cid = str(item.get("candidate_id", ""))
                err_msg = str(item.get("error", ""))
                if err_cid == "incumbent_slot" or not isolate_non_incumbent:
                    raise RuntimeError(
                        f"prewarm_surrogate_tilelang_cache failed: {'; '.join(plan_errors)[:1000]}"
                    )
                if err_cid and err_cid not in failed_by_cid:
                    failed_by_cid[err_cid] = err_msg
        elif plan_errors:
            raise RuntimeError(
                f"prewarm_surrogate_tilelang_cache failed: {'; '.join(plan_errors)[:1000]}"
            )

        unique_miss_tasks: list[dict[str, object]] = []
        unique_miss_keys: list[str] = []
        seen_miss_idx: dict[str, int] = {}
        task_cids_by_idx: dict[int, set[str]] = {}
        planned_plan_keys: dict[tuple[str, int], list[str]] = {}
        seen_hit_task_keys: set[str] = set()
        backfill_tasks: dict[str, dict[str, object]] = {}

        for task in plan_payload.get("tasks", ()):
            if not isinstance(task, Mapping):
                continue
            raw_cid = str(task.get("candidate_id", ""))
            cid = raw_cid or (valid_candidates[0][0] if len(valid_candidates) == 1 else "")
            if cid in failed_by_cid:
                continue
            case_num = int(task.get("case_num", -1))
            if case_num in hit_cases_by_cid.get(cid, set()):
                continue
            mode = str(task.get("mode", ""))
            builder_name = str(task.get("builder_name", ""))
            compile_flags = tuple(str(f) for f in (task.get("compile_flags") or ()))
            raw_pc = task.get("pass_configs")
            pass_configs = dict(raw_pc) if isinstance(raw_pc, Mapping) else {}
            c_sha = case_closure_shas.get((cid, case_num))
            if c_sha is None:
                if raw_cid:
                    continue
                c_sha = "unknown"
            task_kwargs = task.get("kwargs")
            kwargs_map = dict(task_kwargs) if isinstance(task_kwargs, Mapping) else {}
            if mode == "trackb":
                norm_unit = str(task.get("normalized_cpp_unit") or "").strip()
                if norm_unit:
                    t_key = compute_compile_task_key(
                        normalized_kernel_unit_src=norm_unit,
                        cann_target="dav-3510",
                        pass_configs=None,
                        compile_flags=compile_flags,
                        static_spec_signature="",
                        compiler_stamp="",
                    )
                else:
                    tb_dims = kwargs_map.get("dims")
                    tb_sig = _canonical_trackb_spec_signature(
                        tb_dims if isinstance(tb_dims, Mapping) else {},
                        regime_tag=str(kwargs_map.get("regime_tag", "")),
                    )
                    t_key = compute_compile_task_key(
                        normalized_kernel_unit_src=case_closure_srcs.get((cid, case_num), ""),
                        cann_target="dav-3510",
                        pass_configs=pass_configs,
                        compile_flags=compile_flags,
                        static_spec_signature=tb_sig,
                        compiler_stamp="",
                    )
            else:
                ta_sig = _canonical_tracka_spec_signature(builder_name, kwargs_map)
                t_key = compute_compile_task_key(
                    normalized_kernel_unit_src=case_closure_srcs.get((cid, case_num), ""),
                    cann_target="dav-3510",
                    pass_configs=pass_configs,
                    compile_flags=compile_flags,
                    static_spec_signature=ta_sig,
                    compiler_stamp=tl_lib_stamp,
                )
            if (cid, case_num) in case_closure_shas and case_num >= 0:
                planned_plan_keys.setdefault((cid, case_num), []).append(t_key)
            if artifact_store.has_verified_artifact(t_key):
                if t_key not in seen_hit_task_keys:
                    seen_hit_task_keys.add(t_key)
                    cache_hits += 1
                backfill_tasks[t_key] = {
                    "builder_name": builder_name,
                    "cand_sha256": c_sha,
                    "tilelang_lib_stamp": tl_lib_stamp,
                }
                continue
            if t_key in seen_miss_idx:
                if cid:
                    task_cids_by_idx[seen_miss_idx[t_key]].add(cid)
                continue
            idx = len(unique_miss_tasks)
            seen_miss_idx[t_key] = idx
            task_copy = dict(task)
            task_copy["artifact_id"] = t_key
            unique_miss_tasks.append(task_copy)
            unique_miss_keys.append(t_key)
            task_cids_by_idx[idx] = {cid} if cid else set()

        if unique_miss_tasks:
            def _compile_one(
                idx: int, task: dict[str, object]
            ) -> tuple[int, Path, Path, tuple[Path, ...]]:
                worker_dir = tmp_root / f"worker_{idx}"
                worker_cache = worker_dir / "tl_cache"
                worker_cache.mkdir(parents=True, exist_ok=True)
                seed_tl_cache_tree(resolved_tl_cache, worker_cache)
                pre_bins = snapshot_tl_cache_binaries(worker_cache)
                emit_path = worker_dir / "emit_cache_bins.json"
                task_payload = dict(task)
                task_payload["emit_cache_bins_path"] = str(emit_path)
                worker_script = worker_dir / "compile_task.py"
                worker_script.write_text(
                    _PREWARM_TL_WORKER_TEMPLATE.format(
                        task_json=json.dumps(task_payload, sort_keys=True)
                    ),
                    encoding="utf-8",
                )
                worker_env = dict(run_env)
                worker_env["TILELANG_CACHE_DIR"] = str(worker_cache)
                proc = subprocess.run(
                    _build_argv(worker_script),
                    text=True,
                    capture_output=True,
                    cwd=str(root),
                    env=worker_env,
                    timeout=timeout_sec,
                )
                if proc.returncode != 0:
                    detail = (proc.stderr or proc.stdout or "").strip()[-800:]
                    raise RuntimeError(
                        f"candidate={task.get('candidate_id')} case={task.get('case_num')} "
                        f"builder={task.get('builder_name')}: {detail}"
                    )
                return (
                    idx,
                    worker_cache,
                    emit_path,
                    collect_new_tl_cache_binaries(worker_cache, pre_bins),
                )

            max_workers = min(4, len(unique_miss_tasks))
            incumbent_or_fatal_errors: list[str] = []
            worker_caches: dict[int, tuple[Path, Path, tuple[Path, ...]]] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    (idx, pool.submit(_compile_one, idx, task))
                    for idx, task in enumerate(unique_miss_tasks)
                ]
                for idx, fut in futures:
                    try:
                        res_idx, worker_cache, emit_path, bin_rels = fut.result()
                        worker_caches[res_idx] = (worker_cache, emit_path, bin_rels)
                    except Exception as exc:
                        err_text = str(exc)
                        assoc_cids = task_cids_by_idx.get(idx, set())
                        failed_task_count += 1
                        if "incumbent_slot" in assoc_cids or not isolate_non_incumbent:
                            incumbent_or_fatal_errors.append(err_text)
                        for err_cid in assoc_cids:
                            if err_cid not in failed_by_cid:
                                failed_by_cid[err_cid] = err_text

            if incumbent_or_fatal_errors:
                raise RuntimeError(
                    f"prewarm_surrogate_tilelang_cache failed: "
                    f"{'; '.join(incumbent_or_fatal_errors)[:1000]}"
                )

            lock_path = resolved_tl_cache / ".prewarm_manifest_v1.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+", encoding="utf-8") as lock_fh:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
                try:
                    latest = _load_prewarm_manifest(manifest_path)
                    merged_tasks = dict(latest["tasks"])
                    merged_tasks.update(backfill_tasks)
                    merged_plan = dict(latest["plan_index"])
                    for idx in sorted(worker_caches):
                        worker_cache, emit_path, bin_rels = worker_caches[idx]
                        merge_tl_cache_tree(worker_cache, resolved_tl_cache)
                        compiled_tasks += 1
                        task = unique_miss_tasks[idx]
                        t_key = unique_miss_keys[idx]
                        cid = str(task.get("candidate_id", ""))
                        cnum = int(task.get("case_num", -1))
                        c_sha = case_closure_shas.get((cid, cnum), "")
                        is_trackb = task.get("mode") == "trackb"
                        merged_bins = _resolve_worker_cache_bins(
                            emit_path,
                            worker_cache,
                            resolved_tl_cache,
                            bin_rels,
                        )
                        if merged_bins:
                            unit_src = (
                                str(task.get("normalized_cpp_unit") or "")
                                if is_trackb
                                else ""
                            ) or case_closure_srcs.get((cid, cnum), "")
                            artifact_store.record_verified_artifact(
                                t_key,
                                binary_paths=merged_bins,
                                backend_kind="ascendc_trackb" if is_trackb else "tilelang",
                                flags=" ".join(str(f) for f in (task.get("compile_flags") or ())),
                                spec_key="" if is_trackb else f"{operator_name}:{cnum}",
                                compiler_stamp="" if is_trackb else tl_lib_stamp,
                                closure_src=unit_src,
                            )
                        merged_tasks[t_key] = {
                            "builder_name": str(task.get("builder_name", "")),
                            "cand_sha256": c_sha,
                            "tilelang_lib_stamp": tl_lib_stamp,
                        }
                    for (cid, case_num), t_keys in planned_plan_keys.items():
                        if cid in failed_by_cid:
                            continue
                        if all(tk in merged_tasks for tk in t_keys):
                            p_key = _build_prewarm_plan_key(
                                operator_name=operator_name,
                                cand_sha256=case_closure_shas[(cid, case_num)],
                                case_num=case_num,
                                cases_yaml_sha256=cases_yaml_sha256,
                                tilelang_lib_stamp=tl_lib_stamp,
                            )
                            merged_plan[p_key] = list(t_keys)
                    for cid in list(missed_cases_by_cid):
                        if cid in failed_by_cid:
                            continue
                        for cnum in missed_cases_by_cid[cid]:
                            if (cid, cnum) not in planned_plan_keys:
                                p_key = _build_prewarm_plan_key(
                                    operator_name=operator_name,
                                    cand_sha256=case_closure_shas[(cid, cnum)],
                                    case_num=cnum,
                                    cases_yaml_sha256=cases_yaml_sha256,
                                    tilelang_lib_stamp=tl_lib_stamp,
                                )
                                merged_plan[p_key] = []
                    _save_prewarm_manifest(manifest_path, merged_tasks, merged_plan)
                finally:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        else:
            lock_path = resolved_tl_cache / ".prewarm_manifest_v1.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+", encoding="utf-8") as lock_fh:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
                try:
                    latest = _load_prewarm_manifest(manifest_path)
                    merged_tasks = dict(latest["tasks"])
                    merged_tasks.update(backfill_tasks)
                    merged_plan = dict(latest["plan_index"])
                    for (cid, case_num), t_keys in planned_plan_keys.items():
                        if cid in failed_by_cid:
                            continue
                        if all(tk in merged_tasks for tk in t_keys):
                            p_key = _build_prewarm_plan_key(
                                operator_name=operator_name,
                                cand_sha256=case_closure_shas[(cid, case_num)],
                                case_num=case_num,
                                cases_yaml_sha256=cases_yaml_sha256,
                                tilelang_lib_stamp=tl_lib_stamp,
                            )
                            merged_plan[p_key] = list(t_keys)
                    _save_prewarm_manifest(manifest_path, merged_tasks, merged_plan)
                finally:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

    stats = {
        "total_tasks": cache_hits + compiled_tasks + failed_task_count,
        "cache_hits": cache_hits,
        "compiled_tasks": compiled_tasks,
        "failed_candidates": len(failed_by_cid),
    }
    return PrewarmTileLangResult(
        tl_cache_dir=resolved_tl_cache,
        prewarm_stats=stats,
        failed_by_cid=failed_by_cid,
        elapsed_sec=time.monotonic() - t0,
    )


def prewarm_surrogate_tilelang_cache(
    operator_name: str,
    candidates: Sequence[tuple[str, str | Path]],
    anchors: Sequence[AnchorCaseSpec],
    *,
    cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None,
    cache_dir: str | Path | None = None,
    tl_cache_dir: str | Path | None = None,
    source_root: str | Path | None = None,
    timeout_sec: int = 300,
) -> Path:
    """Pre-compile deduplicated candidate anchor kernels into TileLang disk cache on Host CPU."""
    result = prewarm_surrogate_tilelang_cache_with_stats(
        operator_name,
        candidates,
        anchors,
        cannbench_repo=cannbench_repo,
        env_file=env_file,
        cache_dir=cache_dir,
        tl_cache_dir=tl_cache_dir,
        source_root=source_root,
        timeout_sec=timeout_sec,
        isolate_non_incumbent=False,
    )
    return result.tl_cache_dir


_ORIG_PREWARM_SURROGATE_TILELANG_CACHE = prewarm_surrogate_tilelang_cache


_LLM_API_KEY_ENV = "TILEFOUNDRY_NPU_LLM_API_KEY_ENV"
_NON_AGENT_GATES_ENV_KEYS = frozenset(
    {"TILEFOUNDRY_NPU_AGENT_GATES_BIN", "TILEFOUNDRY_NPU_LLM_API_KEY"}
)


def _validate_llm_api_key_env(name: str) -> None:
    if name and (
        not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
        or name.startswith("TILEFOUNDRY_NPU_")
    ):
        raise ValueError(f"{_LLM_API_KEY_ENV} must name a separate environment variable")


def _format_env_line(key: str, value: str) -> str:
    if any(char in value for char in "\n\r\x00"):
        raise ValueError(f"{key} must be single-line")
    return f"{key}={shlex.quote(value)}"


def _write_temp_env_file(
    base_env: Path,
    dest_path: Path,
    overrides: Mapping[str, str],
) -> Path:
    values = parse_simple_env_file(base_env)
    values.update(overrides)
    _validate_llm_api_key_env(values.get(_LLM_API_KEY_ENV, "").strip())
    stripped_keys = set(overrides) | _NON_AGENT_GATES_ENV_KEYS
    lines = [
        line
        for line in base_env.read_text(encoding="utf-8").splitlines()
        if line.partition("=")[0].strip().removeprefix("export ").strip() not in stripped_keys
    ]
    for key, value in overrides.items():
        if key in _NON_AGENT_GATES_ENV_KEYS:
            continue
        lines.append(_format_env_line(key, value))
    dest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest_path


_STRIPPED_HOST_CPU_ENV_KEYS = frozenset(
    {
        "ASCEND_RT_VISIBLE_DEVICES",
        "TILELANG_DISABLE_CACHE",
        "TORCH_DEVICE_BACKEND_AUTOLOAD",
        "USE_CUDA",
    }
)


def _clean_agent_gates_subprocess_env(extra_env: Mapping[str, str]) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("TILEFOUNDRY_NPU_")
        and k not in _STRIPPED_HOST_CPU_ENV_KEYS
        and k not in _REF_CACHE_THREAD_ENV_KEYS
    }
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.update(extra_env)
    return env


def _parse_agent_gates_json(stdout: str) -> dict[str, object]:
    text = (stdout or "").strip()
    for line in reversed(text.splitlines()):
        cand = line.strip()
        if cand.startswith("{") and cand.endswith("}"):
            try:
                parsed = json.loads(cand)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    idx = text.rfind("\n{")
    start = idx + 1 if idx >= 0 else text.find("{")
    if start < 0:
        return {}
    try:
        parsed = json.loads(text[start:])
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _inspect_v2_device_fault(
    report: Mapping[str, object] | None,
    verification: Mapping[str, object] | None = None,
) -> tuple[bool, bool, str]:
    """Inspect AgentGates managed receipts for device_fault or deviceRejected flags."""
    is_device_fault = False
    is_device_rejected = False
    category = ""

    def _scan_mapping(obj: Mapping[str, object] | None) -> None:
        nonlocal is_device_fault, is_device_rejected, category
        if not isinstance(obj, Mapping):
            return
        if obj.get("deviceRejected") is True or obj.get("device_rejected") is True:
            is_device_rejected = True
        for key in ("errorCategory", "failureCategory"):
            raw_cat = str(obj.get(key) or "").strip().lower()
            if raw_cat:
                if not category:
                    category = raw_cat
                if raw_cat == "device_fault":
                    is_device_fault = True

    _scan_mapping(report)
    _scan_mapping(verification)
    if isinstance(report, Mapping):
        gates = report.get("gates")
        if isinstance(gates, Sequence) and not isinstance(gates, (str, bytes)):
            for row in gates:
                if not isinstance(row, Mapping):
                    continue
                _scan_mapping(row)
                obs = row.get("observations")
                if isinstance(obs, Mapping):
                    _scan_mapping(obs)
    return is_device_fault, is_device_rejected, category


def compute_adaptive_poison_pill_timeout(inc_latencies_us: Sequence[float]) -> float:
    """Compute bounded poison-pill kernel timeout in [10.0, 60.0] seconds."""
    return compute_adaptive_poison_pill_timeout_sec(inc_latencies_us)


def run_poison_pill_subprocess(
    cmd: Sequence[str],
    *,
    cwd: str | Path,
    env: Mapping[str, str] | None = None,
    inc_latencies_us: Sequence[float] = (),
    exec_timeout_sec: float | None = None,
    boot_timeout_sec: float = 30.0,
) -> tuple[bool, str, int, str, str]:
    """Run poison-pill worker with boot-ready pipe handshake before exec timeout starts."""
    eff_exec_timeout = (
        max(0.05, float(exec_timeout_sec))
        if exec_timeout_sec is not None
        else compute_adaptive_poison_pill_timeout(inc_latencies_us)
    )
    eff_boot_timeout = max(1.0, float(boot_timeout_sec))
    read_fd, write_fd = os.pipe()
    child_env = dict(os.environ if env is None else env)
    child_env["TF_POISON_PILL_READY_FD"] = str(write_fd)

    def _kill_proc(proc: subprocess.Popen[str]) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()

    proc = subprocess.Popen(
        [str(part) for part in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd),
        env=child_env,
        start_new_session=True,
        pass_fds=(write_fd,),
    )
    os.close(write_fd)

    boot_ready = False
    boot_deadline = time.monotonic() + eff_boot_timeout
    buf = b""
    try:
        while True:
            rem = boot_deadline - time.monotonic()
            if rem <= 0.0:
                _kill_proc(proc)
                stdout, stderr = proc.communicate()
                return (
                    False,
                    f"poison_pill_timeout_or_crash: boot timeout after {eff_boot_timeout:g}s",
                    proc.returncode if proc.returncode is not None else -9,
                    stdout or "",
                    stderr or "",
                )
            ready, _, _ = select.select([read_fd], [], [], rem)
            if not ready:
                continue
            chunk = os.read(read_fd, 256)
            if not chunk:
                break
            buf += chunk
            if b"READY" in buf:
                boot_ready = True
                break
    finally:
        os.close(read_fd)

    if not boot_ready and proc.poll() is not None:
        stdout, stderr = proc.communicate()
        detail = (stderr or stdout or "").strip()[-300:]
        msg = f"poison_pill_timeout_or_crash: exit_code={proc.returncode}"
        if detail:
            msg = f"{msg} ({detail})"
        return False, msg[:300], int(proc.returncode or 1), stdout or "", stderr or ""

    try:
        stdout, stderr = proc.communicate(timeout=eff_exec_timeout)
    except subprocess.TimeoutExpired:
        _kill_proc(proc)
        stdout, stderr = proc.communicate()
        return (
            False,
            f"poison_pill_timeout_or_crash: timeout after {eff_exec_timeout:g}s",
            -9,
            stdout or "",
            stderr or "",
        )

    if proc.returncode != 0:
        detail = (stderr or stdout or "").strip()[-300:]
        msg = f"poison_pill_timeout_or_crash: exit_code={proc.returncode}"
        if detail:
            msg = f"{msg} ({detail})"
        return False, msg[:300], int(proc.returncode), stdout or "", stderr or ""

    return True, "", 0, stdout or "", stderr or ""


def _is_transient_blocked_env(
    proc: subprocess.CompletedProcess[str],
    parsed: Mapping[str, object],
) -> bool:
    if proc.returncode == 0:
        return False
    is_dev_fault, is_dev_rejected, _cat = _inspect_v2_device_fault(parsed)
    if is_dev_fault or is_dev_rejected:
        return False
    if parsed.get("status") == "BLOCKED_ENV":
        return True
    if parsed.get("status") == "FAIL":
        gates = parsed.get("gates")
        if isinstance(gates, Sequence) and not isinstance(gates, (str, bytes)):
            for gate in gates:
                if not isinstance(gate, Mapping):
                    continue
                failures = gate.get("failures")
                if (
                    gate.get("gateId") == "tilefoundry-npu-device-fast"
                    and gate.get("status") == "NOT_RUN"
                    and isinstance(failures, Sequence)
                    and not isinstance(failures, (str, bytes))
                    and "DEPENDENCY_NOT_PASSED" in failures
                ):
                    return True
    return False


def _run_agent_gates_cmd(
    cmd: Sequence[str],
    *,
    cwd: str,
    env: Mapping[str, str],
    timeout_sec: int,
    max_attempts: int = DEFAULT_AGENT_GATES_MAX_ATTEMPTS,
    backoff_sec: float = DEFAULT_AGENT_GATES_BACKOFF_SEC,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    wait_for_npu_quiescent()
    attempts = max(1, int(max_attempts))
    cmd_list = [str(part) for part in cmd]
    env_dict = dict(env)
    last_proc: subprocess.CompletedProcess[str] | None = None
    last_parsed: dict[str, object] = {}
    for attempt in range(1, attempts + 1):
        last_proc = subprocess.run(
            cmd_list,
            text=True,
            capture_output=True,
            cwd=cwd,
            env=env_dict,
            timeout=timeout_sec,
        )
        last_parsed = _parse_agent_gates_json(last_proc.stdout)
        if not (_is_transient_blocked_env(last_proc, last_parsed) and attempt < attempts):
            return last_proc, last_parsed
        delay = min(20.0, max(0.0, float(backoff_sec)) * (2 ** (attempt - 1)))
        logger.warning(
            "AgentGates transient BLOCKED_ENV (attempt %d/%d); retrying in %.1fs",
            attempt,
            attempts,
            delay,
        )
        if delay > 0.0:
            sleep_fn(delay)
    assert last_proc is not None
    return last_proc, last_parsed


def _normalize_device_pool(
    device_id: str,
    device_pool: Sequence[str] = (),
    excluded: Collection[str] = (),
) -> tuple[str, ...]:
    """Return canonical deduplicated decimal card IDs excluding rejected cards."""
    ordered: list[str] = []
    for raw in (device_id, *device_pool):
        txt = str(raw or "").strip()
        if txt.isdigit():
            canon = str(int(txt))
            if canon not in ordered and canon not in excluded:
                ordered.append(canon)
    return tuple(ordered)


_STDLIB_SUBPROCESS_RUN = subprocess.run


def _run_npu_smi_query(smi: str, args: Sequence[str]) -> str | None:
    """Run read-only npu-smi query with short backoff retries on transient exit 215."""
    retry_backoffs = (0.15, 0.35)
    for attempt in range(len(retry_backoffs) + 1):
        try:
            if subprocess.run is not _STDLIB_SUBPROCESS_RUN:
                proc = subprocess.run(
                    [smi, *args],
                    capture_output=True,
                    text=True,
                    timeout=10.0,
                    check=False,
                )
                returncode = proc.returncode
                stdout = proc.stdout or ""
            else:
                child = subprocess.Popen(
                    [smi, *args],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                timed_out = False

                def _kill_child() -> None:
                    nonlocal timed_out
                    timed_out = True
                    child.kill()

                timer = threading.Timer(10.0, _kill_child)
                timer.start()
                try:
                    stdout_raw, _stderr_raw = child.communicate()
                finally:
                    timer.cancel()
                if timed_out:
                    return None
                returncode = child.returncode
                stdout = stdout_raw or ""
        except (OSError, subprocess.SubprocessError, TypeError):
            return None
        if returncode == 0:
            return stdout
        if returncode == 215 and attempt < len(retry_backoffs):
            time.sleep(retry_backoffs[attempt])
            continue
        return None
    return None


def _probe_host_card_load(
    device_id: str,
    *,
    max_quiet_hbm_mb: int | None = None,
    env_vals: Mapping[str, str] | None = None,
) -> tuple[int, int, int] | None:
    """Probe card lock/quarantine/process load as (tier, total_mem_mb, proc_count).

    Tiers: 0=free, 1=quiet shared (<=max_quiet_hbm_mb), 2=heavy shared, 3=locked/quarantined.
    """
    txt = str(device_id or "").strip()
    if not txt.isdigit():
        return None
    dev_idx = int(txt)
    if max_quiet_hbm_mb is not None:
        resolved_max_hbm = max(256, int(max_quiet_hbm_mb))
    else:
        raw_hbm = (
            (env_vals.get("TILEFOUNDRY_NPU_SHARED_MAX_HBM_MB", "") if env_vals is not None else "")
            or os.environ.get("TILEFOUNDRY_NPU_SHARED_MAX_HBM_MB", "")
        ).strip()
        try:
            resolved_max_hbm = max(256, int(raw_hbm)) if raw_hbm else 16384
        except ValueError:
            resolved_max_hbm = 16384
    lock_dir_raw = (
        (env_vals.get("AGENT_GATES_DEVICE_LOCK_ROOT", "") if env_vals is not None else "")
        or os.environ.get("AGENT_GATES_DEVICE_LOCK_ROOT", "")
    ).strip()
    lock_root = (
        Path(lock_dir_raw).expanduser()
        if lock_dir_raw
        else Path.home() / ".local/state/agent-gates/locks"
    )
    lease_hash = hashlib.sha256(f"npu:{dev_idx}".encode()).hexdigest()
    lease_file = lock_root / f"{lease_hash}.lock"
    quarantine_file = Path(f"{lease_file}.quarantine.json")
    if quarantine_file.is_file():
        return (3, 999999, 999999)
    if lease_file.is_file():
        try:
            fd = os.open(lease_file, os.O_RDONLY)
        except OSError:
            return (3, 999999, 999999)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            return (3, 999999, 999999)
        finally:
            os.close(fd)
    smi_cmd = (
        (env_vals.get("AGENT_GATES_SMI_COMMAND", "") if env_vals is not None else "").strip()
        or os.environ.get("AGENT_GATES_SMI_COMMAND")
        or "npu-smi"
    )
    smi = shutil.which(smi_cmd)
    if smi is None:
        return None
    map_out = _run_npu_smi_query(smi, ("info", "-m"))
    if map_out is None:
        return None
    npu_ids: list[int] = []
    for line in map_out.splitlines():
        fields = line.split()
        if len(fields) >= 4 and all(field.isdigit() for field in fields[:4]):
            if int(fields[3]) == dev_idx:
                npu_ids.append(int(fields[0]))
        elif len(fields) >= 3 and all(field.isdigit() for field in fields[:3]):
            if int(fields[2]) == dev_idx:
                npu_ids.append(int(fields[0]))
    unique_ids = sorted(set(npu_ids))
    if len(unique_ids) != 1:
        return None
    npu_id = unique_ids[0]
    proc_out = _run_npu_smi_query(smi, ("info", "-t", "proc-mem", "-i", str(npu_id)))
    if proc_out is None:
        return None
    if not re.search(rf"^\s*NPU ID\s*:\s*{npu_id}\s*$", proc_out, re.MULTILINE):
        return None
    proc_ids = re.findall(r"Process id\s*:\s*\d+", proc_out)
    if proc_ids:
        mem_matches = re.findall(r"Process memory\(MB\)\s*:\s*(\d+)", proc_out)
        total_mem_mb = sum(int(val) for val in mem_matches)
        tier = 1 if total_mem_mb <= resolved_max_hbm else 2
        return (tier, total_mem_mb, len(proc_ids))
    if re.search(r"^\s*No process in device\.?\s*$", proc_out, re.MULTILINE):
        return (0, 0, 0)
    return None


def _probe_host_card_idle(device_id: str) -> bool | None:
    """Return True if card has 0 compute processes, False if busy/locked, None if unavailable."""
    load = _probe_host_card_load(device_id)
    if load is None:
        return None
    return load[0] == 0


def order_candidates_by_idle_preflight(
    device_id: str,
    device_pool: Sequence[str] = (),
    excluded: Collection[str] = (),
    env_vals: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Order candidate cards: free, unknown, quiet shared (lightest first), heavy, locked."""
    candidates = _normalize_device_pool(device_id, device_pool, excluded=excluded)
    if len(candidates) <= 1:
        return candidates
    tier0: list[str] = []
    unknown_cards: list[str] = []
    tier1: list[tuple[int, int, int, str]] = []
    tier2: list[tuple[int, int, int, str]] = []
    tier3: list[str] = []
    for pool_idx, card in enumerate(candidates):
        load = _probe_host_card_load(card, env_vals=env_vals)
        if load is None:
            unknown_cards.append(card)
            continue
        tier, total_mem_mb, proc_count = load
        if tier == 0:
            tier0.append(card)
        elif tier == 1:
            tier1.append((total_mem_mb, proc_count, pool_idx, card))
        elif tier == 2:
            tier2.append((total_mem_mb, proc_count, pool_idx, card))
        else:
            tier3.append(card)
    tier1.sort(key=lambda row: (row[0], row[1], row[2]))
    tier2.sort(key=lambda row: (row[0], row[1], row[2]))
    return (
        *tier0,
        *unknown_cards,
        *[row[3] for row in tier1],
        *[row[3] for row in tier2],
        *tier3,
    )


def _call_order_candidates_by_idle_preflight(
    device_id: str,
    device_pool: Sequence[str],
    *,
    excluded: Collection[str],
    env_vals: Mapping[str, str] | None,
) -> tuple[str, ...]:
    """Forward env_vals to order_candidates_by_idle_preflight only when supported."""
    if env_vals is not None:
        try:
            sig = inspect.signature(order_candidates_by_idle_preflight)
            accepts_env = "env_vals" in sig.parameters or any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
        except (TypeError, ValueError):
            accepts_env = False
        if accepts_env:
            return order_candidates_by_idle_preflight(
                device_id,
                device_pool,
                excluded=excluded,
                env_vals=env_vals,
            )
    return order_candidates_by_idle_preflight(
        device_id,
        device_pool,
        excluded=excluded,
    )


def select_idle_campaign_device(
    device_id: str,
    device_pool: Sequence[str] = (),
    excluded: Collection[str] = (),
    env_vals: Mapping[str, str] | None = None,
) -> str:
    """Select the highest-priority idle or fallback device from the candidate pool."""
    ordered = _call_order_candidates_by_idle_preflight(
        device_id,
        device_pool,
        excluded=excluded,
        env_vals=env_vals,
    )
    return ordered[0] if ordered else str(device_id or "").strip()


def _failed_prewarm_candidate_result(
    candidate_id: str,
    operator_name: str,
    anchors: Sequence[AnchorCaseSpec],
    error_detail: str,
) -> CandidateSurrogateResult:
    msg = f"prewarm_compile_failed: {error_detail}"
    anchor_metrics = tuple(
        build_anchor_metrics(
            anchor,
            accuracy_passed=False,
            cand_us=0.0,
            error_message=msg[:300],
        )
        for anchor in anchors
    )
    return CandidateSurrogateResult(
        candidate_id=candidate_id,
        operator_name=operator_name,
        all_passed=False,
        avg_speedup=0.0,
        min_case_speedup=0.0,
        avg_roofline_eta=0.0,
        surrogate_score=0.0,
        anchors=anchor_metrics,
        error_message=msg[:500],
    )


def run_multi_operator_surrogate_batch_via_agent_gates(
    batches: Sequence[tuple[str, Sequence[tuple[str, str | Path]], Sequence[AnchorCaseSpec]]],
    *,
    source_root: str | Path | None = None,
    env_file: str | Path | None = None,
    device_id: str = "1",
    device_pool: Sequence[str] = (),
    warmup: int = 3,
    repeat: int = 5,
    require_all_passed: bool = False,
    timeout_sec: int = 300,
    prewarm_cache: bool = True,
) -> tuple[dict[str, dict[str, CandidateSurrogateResult]], str, str, float]:
    """Evaluate candidate batches across operators inside one AgentGates device lease."""
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    resolved_env = _resolve_env_file_path(env_file)
    ag_bin = _resolve_agent_gates_bin(root)
    env_vals = parse_simple_env_file(resolved_env)
    cann_repo = resolve_cannbench_repo(
        cannbench_repo=env_vals.get("TILEFOUNDRY_NPU_CANNBENCH_REPO"),
        env_file=resolved_env,
    )
    cache_dir, tl_cache_dir = _fast_cache_roots(root, env_vals)

    prewarm_stats: dict[str, int] = {
        "total_tasks": 0,
        "cache_hits": 0,
        "compiled_tasks": 0,
        "failed_candidates": 0,
    }
    failed_by_cid_by_op: dict[str, dict[str, str]] = {}
    t_prewarm0 = time.monotonic()
    if prewarm_cache:
        use_stats_prewarm = (
            prewarm_surrogate_tilelang_cache is _ORIG_PREWARM_SURROGATE_TILELANG_CACHE
        )
        for op_name, cands, op_anchors in batches:
            prewarm_anchor_reference_cache(
                op_name,
                op_anchors,
                cannbench_repo=cann_repo,
                env_file=resolved_env,
                cache_dir=cache_dir,
                source_root=root,
            )
            if use_stats_prewarm:
                pw_res = prewarm_surrogate_tilelang_cache_with_stats(
                    op_name,
                    cands,
                    op_anchors,
                    cannbench_repo=cann_repo,
                    env_file=resolved_env,
                    cache_dir=cache_dir,
                    tl_cache_dir=tl_cache_dir,
                    source_root=root,
                    isolate_non_incumbent=True,
                )
                for key in prewarm_stats:
                    prewarm_stats[key] += int(pw_res.prewarm_stats.get(key, 0))
                if pw_res.failed_by_cid:
                    failed_by_cid_by_op[op_name] = dict(pw_res.failed_by_cid)
            else:
                prewarm_surrogate_tilelang_cache(
                    op_name,
                    cands,
                    op_anchors,
                    cannbench_repo=cann_repo,
                    env_file=resolved_env,
                    cache_dir=cache_dir,
                    tl_cache_dir=tl_cache_dir,
                    source_root=root,
                )
    l2_prewarm_sec = time.monotonic() - t_prewarm0

    request_root = _fast_request_root(root, env_vals)
    request_id = uuid.uuid4().hex
    request_dir = request_root / request_id
    request_path = request_root / f"{request_id}.request.json"
    response_path = request_root / f"{request_id}.response.json"

    staged: dict[str, dict[str, dict[str, str]]] = {}
    wire_ops: dict[str, str] = {}
    request_batches: list[dict[str, object]] = []
    seen_operators: set[str] = set()
    seen_wire_ops: set[str] = set()
    failed_results_by_op: dict[str, dict[str, CandidateSurrogateResult]] = {}
    valid_cands_by_op: dict[str, list[tuple[str, str | Path]]] = {}
    failures: list[str] = []

    for op_name, cands, op_anchors in batches:
        if op_name in seen_operators:
            raise ValueError(f"duplicate operator_name {op_name!r} in one fast batch")
        seen_operators.add(op_name)
        wire_op = _fast_wire_operator_name(op_name)
        if wire_op in seen_wire_ops:
            raise ValueError(
                f"duplicate wire operator_name {wire_op!r} (from {op_name!r}) in one fast batch"
            )
        seen_wire_ops.add(wire_op)
        wire_ops[op_name] = wire_op

        op_failed_map = failed_by_cid_by_op.get(op_name, {})
        if "incumbent_slot" in op_failed_map:
            raise RuntimeError(
                f"prewarm_surrogate_tilelang_cache failed for incumbent_slot: "
                f"{op_failed_map['incumbent_slot']}"
            )

        op_failed_res: dict[str, CandidateSurrogateResult] = {}
        valid_cands: list[tuple[str, str | Path]] = []
        seen_cids: set[str] = set()
        for candidate_id, module_path in cands:
            if candidate_id in seen_cids:
                raise ValueError(
                    f"duplicate candidate_id {candidate_id!r} for operator {op_name!r}"
                )
            seen_cids.add(candidate_id)
            if candidate_id in op_failed_map:
                op_failed_res[candidate_id] = _failed_prewarm_candidate_result(
                    candidate_id,
                    op_name,
                    op_anchors,
                    op_failed_map[candidate_id],
                )
                failures.append(f"{op_name}/{candidate_id}")
            else:
                valid_cands.append((candidate_id, module_path))

        failed_results_by_op[op_name] = op_failed_res
        had_non_inc = any(cid not in _INCUMBENT_CANDIDATE_IDS for cid, _p in cands)
        has_valid_non_inc = any(cid not in _INCUMBENT_CANDIDATE_IDS for cid, _p in valid_cands)
        if had_non_inc and not has_valid_non_inc:
            valid_cands_by_op[op_name] = []
            continue

        valid_cands_by_op[op_name] = valid_cands
        if not valid_cands:
            continue

        bindings: dict[str, dict[str, str]] = {}
        entries: list[dict[str, str]] = []
        for index, (candidate_id, module_path) in enumerate(valid_cands):
            relative, digest = _stage_fast_candidate(
                request_dir,
                root,
                wire_op,
                index,
                candidate_id,
                Path(module_path),
            )
            bindings[candidate_id] = {"module_path": relative, "sha256": digest}
            entries.append(
                {"candidate_id": candidate_id, "module_path": relative, "sha256": digest}
            )
        staged[wire_op] = bindings
        adaptive_pill_timeout = compute_adaptive_poison_pill_timeout(
            [float(a.ref_perf_us) for a in op_anchors]
        )
        request_batches.append(
            {
                "operator_name": wire_op,
                "entrypoint": get_operator_entrypoint(op_name),
                "candidates": entries,
                "anchors": [a.to_dict() for a in op_anchors],
                "poison_pill_timeout_sec": adaptive_pill_timeout,
            }
        )

    if not request_batches:
        out_empty: dict[str, dict[str, CandidateSurrogateResult]] = {}
        for op_name, cands, _anchors in batches:
            op_failed_res = failed_results_by_op.get(op_name, {})
            ordered_failed = {
                cid: op_failed_res[cid]
                for cid, _p in cands
                if cid in op_failed_res
            }
            out_empty[op_name] = L2BatchResults(
                ordered_failed,
                {},
                prewarm_stats=prewarm_stats,
                l2_prewarm_sec=l2_prewarm_sec,
                l2_npu_sec=0.0,
            )
        if require_all_passed and failures:
            raise RuntimeError(
                "AgentGates L2 batch rejected candidate(s): " + ", ".join(failures)
            )
        return out_empty, "", "", 0.0

    payload: dict[str, object] = {
        "schema": FAST_REQUEST_SCHEMA,
        "request_id": request_id,
        "mode": "batch",
        "warmup": int(warmup),
        "repeat": int(repeat),
        "require_all_passed": bool(require_all_passed),
        "response_path": response_path.relative_to(root).as_posix(),
        "batches": request_batches,
        "msprof_targets": [],
    }
    request_sha256 = _write_fast_request(request_path, payload)
    workspace = _agent_gates_workspace_name(root)
    t_npu0 = time.monotonic()
    raw_resp, execution = _submit_fast_gate(
        agent_gates_bin=ag_bin,
        root=root,
        workspace=workspace,
        request_id=request_id,
        request_relative_path=request_path.relative_to(root).as_posix(),
        request_path=request_path,
        response_path=response_path,
        request_sha256=request_sha256,
        mode="batch",
        timeout_sec=float(timeout_sec),
        device_id=str(device_id),
        device_pool=device_pool,
        env_vals=env_vals,
    )
    l2_npu_sec = time.monotonic() - t_npu0
    is_dev_fault, is_dev_rejected, _cat = _inspect_v2_device_fault(
        execution.report, execution.verification
    )
    if is_dev_fault or is_dev_rejected or execution.device_rejected:
        bad_card = execution.device_id or str(device_id)
        raise AgentGatesDeviceFaultError(
            f"AgentGates device-fast receipt reported device_fault on card {bad_card}",
            fault_plane="INFRA_DEVICE_FAULT",
            fault_code="NPU_DEVICE_TSD_FAULT",
            failed_tier="l2_npu",
            device_id=bad_card,
            rejected_cards=[bad_card] if bad_card else (),
        )
    _verify_fast_bindings(raw_resp, staged)
    _verify_fast_coverage(
        raw_resp,
        expected_candidates=sum(len(valid_cands_by_op[op_name]) for op_name, _c, _a in batches),
        expected_msprof_targets=0,
    )
    raw_ext_hbm = raw_resp.get("_selected_card_ext_hbm_mb", 0)
    selected_card_ext_hbm_mb = (
        max(0, int(raw_ext_hbm))
        if isinstance(raw_ext_hbm, (int, float)) and not isinstance(raw_ext_hbm, bool)
        else 0
    )
    raw_null_launch = raw_resp.get("null_launch_us")
    null_launch_us: float | None = (
        float(raw_null_launch)
        if isinstance(raw_null_launch, (int, float)) and not isinstance(raw_null_launch, bool)
        else None
    )
    operators_raw = _fast_response_mapping(raw_resp, "operators")
    out_by_op: dict[str, dict[str, CandidateSurrogateResult]] = {}
    for op_name, cands, _anchors in batches:
        op_failed_res = failed_results_by_op.get(op_name, {})
        valid_cands = valid_cands_by_op.get(op_name, [])
        parsed_candidates: dict[str, CandidateSurrogateResult] = {}
        run_ids_for_op: dict[str, str] = {}
        if valid_cands:
            wire_op = wire_ops[op_name]
            cands_raw = operators_raw.get(wire_op)
            if cands_raw is None and wire_op != op_name:
                cands_raw = operators_raw.get(op_name)
            if not isinstance(cands_raw, Mapping):
                raise RuntimeError(f"fast response has no results for operator {op_name!r}")
            for candidate_id, _module_path in valid_cands:
                item = cands_raw.get(candidate_id)
                if not isinstance(item, Mapping):
                    raise RuntimeError(
                        f"fast response has no result for {op_name}/{candidate_id}"
                    )
                result = CandidateSurrogateResult.from_dict(item)
                if selected_card_ext_hbm_mb > 0 and any(
                    anc.card_ext_hbm_mb == 0 for anc in result.anchors
                ):
                    updated_anchors = tuple(
                        replace(anc, card_ext_hbm_mb=selected_card_ext_hbm_mb)
                        if anc.card_ext_hbm_mb == 0
                        else anc
                        for anc in result.anchors
                    )
                    result = replace(result, anchors=updated_anchors)
                parsed_candidates[candidate_id] = result
                run_ids_for_op[candidate_id] = execution.run_id
                if not result.all_passed:
                    failures.append(f"{op_name}/{candidate_id}")
        merged_ordered: dict[str, CandidateSurrogateResult] = {}
        for candidate_id, _module_path in cands:
            if candidate_id in parsed_candidates:
                merged_ordered[candidate_id] = parsed_candidates[candidate_id]
            elif candidate_id in op_failed_res:
                merged_ordered[candidate_id] = op_failed_res[candidate_id]
        out_by_op[op_name] = L2BatchResults(
            merged_ordered,
            run_ids_for_op,
            prewarm_stats=prewarm_stats,
            l2_prewarm_sec=l2_prewarm_sec,
            l2_npu_sec=l2_npu_sec,
            null_launch_us=null_launch_us,
        )
    if require_all_passed and failures:
        raise RuntimeError(
            "AgentGates L2 batch rejected candidate(s): " + ", ".join(failures)
        )
    return (
        out_by_op,
        execution.run_id,
        execution.receipt_path,
        _fast_held_seconds(execution.report),
    )


def run_surrogate_batch_via_agent_gates(
    operator_name: str,
    candidates: Sequence[tuple[str, str | Path]],
    anchors: Sequence[AnchorCaseSpec] | None = None,
    *,
    source_root: str | Path | None = None,
    env_file: str | Path | None = None,
    device_id: str = "1",
    device_pool: Sequence[str] = (),
    warmup: int = 3,
    repeat: int = 5,
    require_all_passed: bool = False,
    timeout_sec: int = 300,
) -> tuple[dict[str, CandidateSurrogateResult], str, str]:
    """Evaluate all L1-passing candidates of one round inside a single AgentGates window."""
    active_anchors = tuple(anchors) if anchors is not None else get_default_anchors(operator_name)
    by_op, run_id, receipt_path, _held = run_multi_operator_surrogate_batch_via_agent_gates(
        [(operator_name, candidates, active_anchors)],
        source_root=source_root,
        env_file=env_file,
        device_id=device_id,
        device_pool=device_pool,
        warmup=warmup,
        repeat=repeat,
        require_all_passed=require_all_passed,
        timeout_sec=timeout_sec,
    )
    op_res = by_op.get(operator_name)
    if op_res is None:
        return L2BatchResults({}, {}), run_id, receipt_path
    return op_res, run_id, receipt_path


def bottleneck_anchor_for_regime(
    result: CandidateSurrogateResult,
    regime_tag: str,
) -> AnchorEvalMetrics | None:
    """Return the lowest-Eq.3-score evaluated anchor matching regime_tag."""
    matched = [a for a in result.anchors if a.regime_tag == regime_tag]
    if not matched:
        return None
    return min(matched, key=lambda a: (a.eq3_score, -a.cand_us, a.case_num))


def _collect_profile_via_agent_gates(
    operator_name: str,
    candidate_path: str | Path,
    anchors_by_regime: Sequence[AnchorCaseSpec] | Mapping[str, AnchorCaseSpec],
    *,
    candidate_id: str,
    source_root: str | Path | None = None,
    env_file: str | Path | None = None,
    device_id: str = "1",
    device_pool: Sequence[str] = (),
    warmup: int = 1,
    repeat: int = 3,
    timeout_sec: int = 300,
) -> dict[str, dict[str, object]]:
    """Collect mode='profile' msprof telemetry inside an AgentGates lease."""
    if isinstance(anchors_by_regime, Mapping):
        regime_items = [(str(tag), spec) for tag, spec in anchors_by_regime.items()]
    else:
        regime_items = [(spec.regime_tag, spec) for spec in anchors_by_regime]
    if not regime_items:
        return {}

    unique_case_nums: list[int] = []
    seen_cases: set[int] = set()
    for _tag, spec in regime_items:
        cnum = int(spec.case_num)
        if cnum not in seen_cases:
            seen_cases.add(cnum)
            unique_case_nums.append(cnum)

    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    resolved_env = _resolve_env_file_path(env_file)
    ag_bin = _resolve_agent_gates_bin(root)
    env_vals = parse_simple_env_file(resolved_env)
    cann_repo = resolve_cannbench_repo(
        cannbench_repo=env_vals.get("TILEFOUNDRY_NPU_CANNBENCH_REPO"),
        env_file=resolved_env,
    )
    cache_dir, tl_cache_dir = _fast_cache_roots(root, env_vals)
    cand_resolved = Path(candidate_path).resolve()

    unique_specs = [
        next(spec for _t, spec in regime_items if int(spec.case_num) == cnum)
        for cnum in unique_case_nums
    ]
    prewarm_anchor_reference_cache(
        operator_name,
        unique_specs,
        cannbench_repo=cann_repo,
        env_file=resolved_env,
        cache_dir=cache_dir,
        source_root=root,
    )
    prewarm_surrogate_tilelang_cache(
        operator_name,
        [(candidate_id, cand_resolved)],
        unique_specs,
        cannbench_repo=cann_repo,
        env_file=resolved_env,
        cache_dir=cache_dir,
        tl_cache_dir=tl_cache_dir,
        source_root=root,
    )

    request_root = _fast_request_root(root, env_vals)
    request_id = uuid.uuid4().hex
    request_dir = request_root / request_id
    request_path = request_root / f"{request_id}.request.json"
    response_path = request_root / f"{request_id}.response.json"
    wire_op = _fast_wire_operator_name(operator_name)
    relative, digest = _stage_fast_candidate(
        request_dir,
        root,
        wire_op,
        0,
        candidate_id,
        cand_resolved,
    )
    payload: dict[str, object] = {
        "schema": FAST_REQUEST_SCHEMA,
        "request_id": request_id,
        "mode": "profile",
        "warmup": int(warmup),
        "repeat": int(repeat),
        "require_all_passed": False,
        "response_path": response_path.relative_to(root).as_posix(),
        "batches": [],
        "msprof_targets": [
            {
                "operator_name": wire_op,
                "candidate_id": candidate_id,
                "module_path": relative,
                "sha256": digest,
                "case_num": cnum,
                "warmup": int(warmup),
                "repeat": int(repeat),
            }
            for cnum in unique_case_nums
        ],
    }
    request_sha256 = _write_fast_request(request_path, payload)
    workspace = _agent_gates_workspace_name(root)
    raw_resp, _execution = _submit_fast_gate(
        agent_gates_bin=ag_bin,
        root=root,
        workspace=workspace,
        request_id=request_id,
        request_relative_path=request_path.relative_to(root).as_posix(),
        request_path=request_path,
        response_path=response_path,
        request_sha256=request_sha256,
        mode="profile",
        timeout_sec=float(timeout_sec),
        device_id=str(device_id),
        device_pool=device_pool,
        env_vals=env_vals,
    )
    _verify_fast_bindings(
        raw_resp,
        {wire_op: {candidate_id: {"module_path": relative, "sha256": digest}}},
    )
    _verify_fast_coverage(
        raw_resp,
        expected_candidates=1,
        expected_msprof_targets=len(unique_case_nums),
    )
    telemetry_all = _fast_response_mapping(raw_resp, "msprof_telemetry")
    telemetry_raw = (
        telemetry_all.get(wire_op) if wire_op in telemetry_all else telemetry_all.get(operator_name)
    )
    by_case = (
        telemetry_raw.get(candidate_id) if isinstance(telemetry_raw, Mapping) else None
    )
    if not isinstance(by_case, Mapping):
        raise RuntimeError(
            f"fast profile response has no msprof telemetry for {operator_name!r}"
        )
    out: dict[str, dict[str, object]] = {}
    for tag, spec in regime_items:
        raw_metrics = by_case.get(str(int(spec.case_num)))
        if isinstance(raw_metrics, Mapping) and "error" not in raw_metrics:
            out[tag] = {str(k): v for k, v in raw_metrics.items()}
    return out


def collect_bottleneck_msprof_via_agent_gates(
    operator_name: str,
    candidate_path: str | Path,
    anchors_by_regime: Sequence[AnchorCaseSpec] | Mapping[str, AnchorCaseSpec],
    *,
    source_root: str | Path | None = None,
    env_file: str | Path | None = None,
    device_id: str = "1",
    device_pool: Sequence[str] = (),
    warmup: int = 1,
    repeat: int = 3,
    timeout_sec: int = 300,
) -> dict[str, dict[str, float]]:
    """Collect numeric msprof telemetry for bottleneck anchors inside an AgentGates lease."""
    raw_by_regime = _collect_profile_via_agent_gates(
        operator_name,
        candidate_path,
        anchors_by_regime,
        candidate_id="bottleneck_probe",
        source_root=source_root,
        env_file=env_file,
        device_id=device_id,
        device_pool=device_pool,
        warmup=warmup,
        repeat=repeat,
        timeout_sec=timeout_sec,
    )
    out: dict[str, dict[str, float]] = {}
    for tag, raw_metrics in raw_by_regime.items():
        out[tag] = {
            str(k): float(v)
            for k, v in raw_metrics.items()
            if isinstance(v, (int, float))
            and not isinstance(v, bool)
            and math.isfinite(float(v))
        }
    return out


def collect_p2_deep_dive_via_agent_gates(
    operator_name: str,
    candidate_path: str | Path,
    anchors_by_regime: Sequence[AnchorCaseSpec] | Mapping[str, AnchorCaseSpec],
    *,
    source_root: str | Path | None = None,
    env_file: str | Path | None = None,
    device_id: str = "1",
    device_pool: Sequence[str] = (),
    warmup: int = 1,
    repeat: int = 3,
    timeout_sec: int = 300,
) -> dict[str, object]:
    """Collect instruction-level deep-dive telemetry and DWARF hotspots via AgentGates."""
    raw_by_regime = _collect_profile_via_agent_gates(
        operator_name,
        candidate_path,
        anchors_by_regime,
        candidate_id="bottleneck_p2_deep_dive",
        source_root=source_root,
        env_file=env_file,
        device_id=device_id,
        device_pool=device_pool,
        warmup=warmup,
        repeat=repeat,
        timeout_sec=timeout_sec,
    )
    return dict(raw_by_regime)


_AGENT_GATES_CONFIG_ENV = "AGENT_GATES_CONFIG"


def _agent_gates_config_path() -> Path:
    configured = os.environ.get(_AGENT_GATES_CONFIG_ENV, "").strip()
    return (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".config/agent-gates/config.toml"
    )


def _agent_gates_config() -> Mapping[str, object]:
    return tomllib.loads(_agent_gates_config_path().read_text(encoding="utf-8"))


def _agent_gates_workspace_name(source_root: Path) -> str:
    """Resolve the configured AgentGates workspace whose source root is ``source_root``."""
    raw = _agent_gates_config()
    workspaces = raw.get("workspaces")
    if not isinstance(workspaces, Mapping):
        raise ValueError("AgentGates config must define [workspaces]")
    wanted = source_root.resolve()
    for name, entry in workspaces.items():
        if not isinstance(entry, Mapping):
            continue
        configured = entry.get("source_root")
        if isinstance(configured, str) and Path(configured).expanduser().resolve() == wanted:
            return str(name)
    raise ValueError(f"No AgentGates workspace is configured for source root {wanted}")


def _workspace_relative_candidate(source_root: Path, candidate_path: Path) -> str | None:
    try:
        relative = candidate_path.resolve().relative_to(source_root.resolve())
    except ValueError:
        return None
    return relative.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class _V2GateExecution:
    """Outcome of one AgentGates submit/wait/report/verify sequence."""

    execution_id: str = ""
    run_id: str = ""
    receipt_path: str = ""
    report: Mapping[str, object] = field(default_factory=dict)
    verification: Mapping[str, object] = field(default_factory=dict)
    transport_error: str = ""
    elapsed_sec: float = 0.0
    device_id: str = ""
    state: str = ""
    execution_terminal: bool = True
    device_rejected: bool = False

    @property
    def execution_active(self) -> bool:
        """Return whether an execution may still be running and must be resumed."""
        return bool(self.execution_id) and not self.execution_terminal

    @property
    def verified(self) -> bool:
        """Return whether verify-requirements proved this run against current sources."""
        return (
            not self.transport_error
            and self.execution_id != ""
            and self.verification.get("satisfied") is True
        )


def _agent_gates_cli_env() -> dict[str, str]:
    """Build the CLI environment without project Python startup injection."""
    env = _clean_agent_gates_subprocess_env({})
    for key in _V2_STRIPPED_PYTHON_ENV_KEYS:
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _gate_row(report: Mapping[str, object], gate_id: str) -> Mapping[str, object] | None:
    for row in report.get("gates", ()):
        if isinstance(row, Mapping) and row.get("gateId") == gate_id:
            return row
    return None


def _gate_observation(
    report: Mapping[str, object], gate_id: str
) -> tuple[Mapping[str, object] | None, Mapping[str, object]]:
    row = _gate_row(report, gate_id)
    if row is None:
        return None, {}
    obs = row.get("observations")
    return row, dict(obs) if isinstance(obs, Mapping) else {}


def _gate_recorded_status(row: Mapping[str, object] | None) -> str:
    if row is None:
        return ""
    return str(row.get("recordedStatus") or row.get("status") or "")


def _gate_failure_text(row: Mapping[str, object]) -> str:
    raw_failures = row.get("failures")
    if isinstance(raw_failures, Sequence) and not isinstance(raw_failures, (str, bytes)):
        return " ".join(str(item) for item in raw_failures)
    return str(raw_failures or "")


def _blocked_env_category(
    report: Mapping[str, object],
    gate_ids: Sequence[str] = (
        AGENT_GATES_DEVICE_FAST_GATE,
        AGENT_GATES_L3_PROMOTION_GATE,
        AGENT_GATES_L4_CANNBENCH_GATE,
        AGENT_GATES_L4_CANNBENCH_PREP_GATE,
    ),
) -> str:
    """Return the signed BLOCKED_ENV category across device-fast, L3, and L4 gate rows.

    Missing or unparsable categories stay unclassified and never trigger card rotation.
    """
    for gate_id in gate_ids:
        row, observation = _gate_observation(report, gate_id)
        if row is None:
            continue
        blocked = _gate_recorded_status(row).upper() == "BLOCKED_ENV"
        if not blocked and "BLOCKED_EXIT_CODE: 75" not in _gate_failure_text(row):
            continue
        category = str(
            observation.get("errorCategory") or observation.get("failureCategory") or ""
        ).strip().lower()
        if category:
            return category
    return ""


def _submit_device_rejected(
    proc: subprocess.CompletedProcess[str],
    parsed: Mapping[str, object],
) -> bool:
    """Return True when AgentGates refused the named card as unauthorized.

    Submit is the sole authorization authority; doctor deliberately does not
    check the allowlist. The CLI reports the refusal as a structured
    ``DEVICE_NOT_AUTHORIZED`` payload with exit code 2. Anything else is an
    ordinary submission failure and must not mark the card itself unusable.
    """
    if str(parsed.get("code", "")) == "DEVICE_NOT_AUTHORIZED":
        return True
    return "DEVICE_NOT_AUTHORIZED" in f"{proc.stdout}\n{proc.stderr}"


def _as_positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def _run_v2_gate(
    *,
    agent_gates_bin: str,
    root: Path,
    workspace: str,
    gate_id: str,
    target: str,
    candidate_path: str | None = None,
    timeout_sec: float,
    device_id: str = "",
    resume_execution_id: str = "",
    resume_run_id: str = "",
    on_execution_started: Callable[[str, str], None] | None = None,
) -> _V2GateExecution:
    """Run one managed gate with a single monotonic deadline and one execution identity.

    A new submission first diagnoses the same route with the same target and
    candidate, so a blocked machine or an unusable pair is refused before an
    execution exists; a resumed execution goes straight to its wait and recovers
    the submitted run by its IDs. A bounded local wait slice continues the same
    execution; a terminal state wins even at the ``timed_out`` boundary.

    Note:
        An unproven terminal state stays recoverable. The submission callback
        persists the identity immediately so a killed campaign recovers it, and
        verification binds the receipt to the requested target and candidate;
        ``device_id`` is part of the execution identity and a resume keeps its
        card.
    """
    started = time.monotonic()
    deadline = started + max(1.0, float(timeout_sec))
    requested_device = str(device_id or "").strip()
    submit_cmd = [
        agent_gates_bin,
        "submit",
        "--workspace",
        workspace,
        "--adapter",
        AGENT_GATES_ADAPTER,
        "--gate",
        gate_id,
        "--target",
        target,
        "--json",
    ]
    if candidate_path is not None:
        submit_cmd.extend(["--candidate", candidate_path])
    if requested_device:
        submit_cmd.extend(["--device-id", requested_device])

    def _failed(
        execution_id: str,
        message: str,
        *,
        run_id: str = "",
        receipt_path: str = "",
        resolved_device_id: str = "",
        state: str = "",
        execution_terminal: bool = True,
        device_rejected: bool = False,
    ) -> _V2GateExecution:
        return _V2GateExecution(
            execution_id=execution_id,
            run_id=run_id,
            receipt_path=receipt_path,
            transport_error=message[-1000:],
            elapsed_sec=time.monotonic() - started,
            device_id=resolved_device_id or requested_device,
            state=state,
            execution_terminal=execution_terminal,
            device_rejected=device_rejected,
        )

    if resume_execution_id:
        execution_id = resume_execution_id
        resolved_device = requested_device
    else:
        doctor_cmd = [
            agent_gates_bin,
            "doctor",
            "--adapter",
            AGENT_GATES_ADAPTER,
            "--source",
            str(root),
            "--gate",
            gate_id,
            "--target",
            target,
            "--json",
        ]
        if candidate_path is not None:
            doctor_cmd.extend(["--candidate", candidate_path])
        if requested_device:
            doctor_cmd.extend(["--device-id", requested_device])
        try:
            doctor_proc, doctor = _run_agent_gates_v2_cmd(
                doctor_cmd,
                root=root,
                timeout_sec=min(_V2_SUBMIT_TIMEOUT_SEC, max(1.0, deadline - started)),
            )
        except FileNotFoundError:
            raise
        except (OSError, subprocess.TimeoutExpired) as exc:
            return _failed("", f"{gate_id} doctor failed: {exc}")
        if doctor_proc.returncode != 0 or doctor.get("status") != "PASS":
            raw_failures = doctor.get("failures")
            failures = (
                "; ".join(str(item) for item in raw_failures)
                if isinstance(raw_failures, list)
                else ""
            )
            return _failed(
                "",
                failures
                or doctor_proc.stderr
                or doctor_proc.stdout
                or f"{gate_id} doctor did not pass for target {target}",
            )
        try:
            proc, parsed = _run_agent_gates_v2_cmd(
                submit_cmd,
                root=root,
                timeout_sec=min(_V2_SUBMIT_TIMEOUT_SEC, max(1.0, deadline - started)),
            )
        except FileNotFoundError:
            raise
        except (OSError, subprocess.TimeoutExpired) as exc:
            return _failed("", f"{gate_id} submit failed: {exc}")
        if proc.returncode != 0:
            return _failed(
                "",
                proc.stderr or proc.stdout or f"{gate_id} submit failed",
                device_rejected=_submit_device_rejected(proc, parsed),
            )
        execution_id = str(parsed.get("execution_id", ""))
        if not execution_id:
            return _failed("", f"{gate_id} submit returned no execution_id")
        resolved_device = str(
            parsed.get("device_id") or parsed.get("deviceId") or requested_device or ""
        )
        if on_execution_started is not None:
            on_execution_started(execution_id, resolved_device)

    run_id = resume_run_id
    last_state = ""
    if not run_id:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                terminal = last_state in {"FINISHED", "CANCELLED", "INTERRUPTED"}
                return _failed(
                    execution_id,
                    f"{gate_id} execution {execution_id} wait timed out",
                    resolved_device_id=resolved_device,
                    state=last_state or "RUNNING",
                    execution_terminal=terminal,
                )
            wait_seconds = min(_V2_WAIT_SLICE_SEC, remaining)
            wait_cmd = [
                agent_gates_bin,
                "wait",
                "--execution-id",
                execution_id,
                "--timeout",
                f"{wait_seconds:g}",
                "--json",
            ]
            try:
                proc, parsed = _run_agent_gates_v2_cmd(
                    wait_cmd,
                    root=root,
                    timeout_sec=wait_seconds + 5.0,
                )
            except subprocess.TimeoutExpired:
                continue
            except OSError as exc:
                return _failed(
                    execution_id,
                    f"{gate_id} execution {execution_id} wait failed: {exc}",
                    resolved_device_id=resolved_device,
                    state=last_state,
                    execution_terminal=False,
                )
            if proc.returncode != 0:
                return _failed(
                    execution_id,
                    proc.stderr or proc.stdout or f"{gate_id} execution {execution_id} wait failed",
                    resolved_device_id=resolved_device,
                    state=last_state,
                    execution_terminal=False,
                )
            status = parsed.get("status")
            state = str(status.get("state") or "") if isinstance(status, Mapping) else ""
            if state:
                last_state = state
            if state == "FINISHED":
                run_id = str(status.get("run_id") or "")
                break
            if state in {"CANCELLED", "INTERRUPTED"}:
                return _failed(
                    execution_id,
                    f"{gate_id} execution {execution_id} ended in state {state!r}",
                    resolved_device_id=resolved_device,
                    state=state,
                    execution_terminal=True,
                )
            if parsed.get("timed_out") is True:
                continue
            if state is None or not state:
                return _failed(
                    execution_id,
                    f"{gate_id} execution {execution_id} wait returned no status",
                    resolved_device_id=resolved_device,
                    state=last_state,
                    execution_terminal=False,
                )
            if state != "FINISHED":
                return _failed(
                    execution_id,
                    f"{gate_id} execution {execution_id} ended in state {state!r}",
                    resolved_device_id=resolved_device,
                    state=state,
                    execution_terminal=True,
                )
            run_id = str(status.get("run_id") or "")
            break
    if not run_id:
        return _failed(
            execution_id,
            f"{gate_id} execution {execution_id} finished without run_id",
            resolved_device_id=resolved_device,
            state=last_state or "FINISHED",
            execution_terminal=True,
        )

    try:
        evidence_root = _agent_gates_evidence_root()
    except (OSError, ValueError) as exc:
        return _failed(
            execution_id,
            f"{gate_id} evidence root unavailable: {exc}",
            run_id=run_id,
            resolved_device_id=resolved_device,
            state=last_state or "FINISHED",
        )
    report_cmd = [
        agent_gates_bin,
        "report",
        "--run-id",
        run_id,
        "--evidence-root",
        str(evidence_root),
        "--json",
    ]
    post_wait = min(_V2_POST_WAIT_TIMEOUT_SEC, max(1.0, deadline - time.monotonic()))
    try:
        proc, report = _run_agent_gates_v2_cmd(report_cmd, root=root, timeout_sec=post_wait)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _failed(
            execution_id,
            f"{gate_id} report failed: {exc}",
            run_id=run_id,
            resolved_device_id=resolved_device,
            state=last_state or "FINISHED",
        )
    if not report or "receiptPath" not in report:
        return _failed(
            execution_id,
            proc.stderr or proc.stdout or f"{gate_id} report returned no receipt",
            run_id=run_id,
            resolved_device_id=resolved_device,
            state=last_state or "FINISHED",
        )
    receipt_path = str(report.get("receiptPath", ""))

    verify_cmd = [
        agent_gates_bin,
        "verify-requirements",
        "--workspace",
        workspace,
        "--adapter",
        AGENT_GATES_ADAPTER,
        "--gate",
        gate_id,
        "--run-id",
        run_id,
        "--json",
    ]
    if candidate_path is not None:
        verify_cmd.extend(["--target", target, "--candidate", candidate_path])
    post_wait = min(_V2_POST_WAIT_TIMEOUT_SEC, max(1.0, deadline - time.monotonic()))
    try:
        proc, verification = _run_agent_gates_v2_cmd(verify_cmd, root=root, timeout_sec=post_wait)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _failed(
            execution_id,
            f"{gate_id} requirement verification failed: {exc}",
            run_id=run_id,
            receipt_path=receipt_path,
            resolved_device_id=resolved_device,
            state=last_state or "FINISHED",
        )
    if not verification or "satisfied" not in verification:
        return _failed(
            execution_id,
            proc.stderr or proc.stdout or f"{gate_id} requirement verification returned no verdict",
            run_id=run_id,
            receipt_path=receipt_path,
            resolved_device_id=resolved_device,
            state=last_state or "FINISHED",
        )
    is_dev_fault, is_dev_rejected, _cat = _inspect_v2_device_fault(report, verification)
    return _V2GateExecution(
        execution_id=execution_id,
        run_id=run_id,
        receipt_path=receipt_path,
        report=report,
        verification=verification,
        elapsed_sec=time.monotonic() - started,
        device_id=resolved_device,
        state=last_state or "FINISHED",
        execution_terminal=True,
        device_rejected=bool(is_dev_rejected or is_dev_fault),
    )


_FAST_ID_SANITIZER = re.compile(r"[^A-Za-z0-9_.-]+")


def _fast_slug(value: str, *, fallback: str) -> str:
    slug = _FAST_ID_SANITIZER.sub("_", value).strip("._")
    return slug or fallback


def _fast_wire_operator_name(operator_name: str) -> str:
    """Return leaf operator name matching AgentGates FAST_NAME_RE without slashes."""
    leaf = str(operator_name or "").strip().rsplit("/", 1)[-1]
    return _fast_slug(leaf, fallback="operator")


def _fast_request_root(root: Path, env_vals: Mapping[str, str]) -> Path:
    """Resolve the request staging root under the AgentGates lease-managed build directory."""
    build_subdir = env_vals.get("TILEFOUNDRY_NPU_BUILD_SUBDIR", "").strip() or "build"
    if build_subdir in {".", ".."} or Path(build_subdir).name != build_subdir:
        raise ValueError(
            "TILEFOUNDRY_NPU_BUILD_SUBDIR must be one relative directory name, "
            f"got {build_subdir!r}"
        )
    return root / build_subdir / "agent_gates" / "fast"


def _fast_cache_roots(root: Path, env_vals: Mapping[str, str]) -> tuple[Path, Path]:
    """Resolve the reference and TileLang caches the device-fast child also uses."""
    raw = env_vals.get("TILEFOUNDRY_NPU_WORK_ROOT", "").strip()
    if not raw:
        raise ValueError(
            "AgentGates machine env has no TILEFOUNDRY_NPU_WORK_ROOT for device-fast"
        )
    work_root = Path(raw).expanduser()
    if not work_root.is_absolute():
        raise ValueError(
            f"TILEFOUNDRY_NPU_WORK_ROOT must be an absolute path, got {raw!r}"
        )
    resolved = work_root.resolve()
    if resolved == root or resolved.is_relative_to(root):
        raise ValueError(
            f"TILEFOUNDRY_NPU_WORK_ROOT must be outside the source root, got {resolved}"
        )
    base = resolved / "surrogate"
    return base / "reference", base / "tilelang"


def _stage_fast_candidate(
    request_dir: Path,
    root: Path,
    operator_name: str,
    index: int,
    candidate_id: str,
    module_path: Path,
) -> tuple[str, str]:
    """Copy a candidate into the request staging tree, returning (relative path, sha256)."""
    source = module_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"candidate module {source} does not exist")
    if source.suffix != ".py":
        raise ValueError(f"candidate module {source} must be a Python file")
    operator_dir = request_dir / "candidates" / _fast_slug(operator_name, fallback="operator")
    operator_dir.mkdir(parents=True, exist_ok=True)
    staged = operator_dir / f"{index:03d}_{_fast_slug(candidate_id, fallback='candidate')}.py"
    if staged.exists():
        raise FileExistsError(f"staged candidate {staged} already exists")
    shutil.copyfile(source, staged)
    staged.chmod(0o444)
    return staged.relative_to(root).as_posix(), _sha256_file(staged)


def _write_fast_request(path: Path, payload: Mapping[str, object]) -> str:
    """Write the request once as an immutable descriptor and return its sha256."""
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fchmod(stream.fileno(), 0o444)
    return hashlib.sha256(data).hexdigest()


def _fast_response_mapping(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"fast response field {key!r} must be a mapping")
    return value


def _require_fast_envelope(
    raw: Mapping[str, object],
    *,
    request_id: str,
    request_sha256: str,
    mode: str,
) -> str:
    """Validate the response envelope against the submitted request and return its status."""
    if raw.get("schema") != FAST_RESPONSE_SCHEMA:
        raise RuntimeError(
            f"fast response schema {raw.get('schema')!r} != {FAST_RESPONSE_SCHEMA!r}"
        )
    if raw.get("request_id") != request_id:
        raise RuntimeError(
            f"fast response request_id {raw.get('request_id')!r} != {request_id!r}"
        )
    if raw.get("request_sha256") != request_sha256:
        raise RuntimeError("fast response request_sha256 does not match the submitted request")
    if raw.get("mode") != mode:
        raise RuntimeError(f"fast response mode {raw.get('mode')!r} != {mode!r}")
    status = raw.get("status")
    if status not in (
        FAST_STATUS_OK,
        FAST_STATUS_CANDIDATE_FAILURE,
        FAST_STATUS_REQUEST_ERROR,
        FAST_STATUS_INFRASTRUCTURE_ERROR,
    ):
        raise RuntimeError(f"fast response status {status!r} is not recognized")
    return str(status)


def _raise_for_fast_status(status: str, raw: Mapping[str, object]) -> None:
    if status in (FAST_STATUS_REQUEST_ERROR, FAST_STATUS_INFRASTRUCTURE_ERROR):
        detail = str(raw.get("error", "")).strip() or "no detail"
        raise RuntimeError(f"fast gate {status}: {detail}")


def _verify_fast_bindings(
    raw: Mapping[str, object],
    expected: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> None:
    """Prove every evaluated candidate matches its staged path and content digest."""
    bindings = _fast_response_mapping(raw, "candidates")
    for operator_name, candidates in expected.items():
        operator_bindings = bindings.get(operator_name)
        if not isinstance(operator_bindings, Mapping):
            raise RuntimeError(
                f"fast response has no candidate bindings for operator {operator_name!r}"
            )
        if set(operator_bindings) != set(candidates):
            raise RuntimeError(
                f"fast response candidate set for operator {operator_name!r} "
                "does not match the request"
            )
        for candidate_id, binding in candidates.items():
            actual = operator_bindings.get(candidate_id)
            if not isinstance(actual, Mapping):
                raise RuntimeError(
                    f"fast response binding for {operator_name}/{candidate_id} is malformed"
                )
            if actual.get("module_path") != binding["module_path"]:
                raise RuntimeError(
                    f"fast response module_path for {operator_name}/{candidate_id} "
                    "does not match the staged candidate"
                )
            if actual.get("sha256") != binding["sha256"]:
                raise RuntimeError(
                    f"fast response sha256 for {operator_name}/{candidate_id} "
                    "does not match the staged candidate"
                )
            if actual.get("post_sha256") != binding["sha256"]:
                raise RuntimeError(
                    f"fast gate {FAST_STATUS_INFRASTRUCTURE_ERROR}: candidate "
                    f"{operator_name}/{candidate_id} changed during evaluation"
                )


def _verify_fast_coverage(
    raw: Mapping[str, object],
    *,
    expected_candidates: int,
    expected_msprof_targets: int,
) -> None:
    coverage = _fast_response_mapping(raw, "coverage")
    expected_seen = coverage.get("expected_candidates")
    evaluated = coverage.get("evaluated_candidates")
    targets_expected = coverage.get("expected_msprof_targets")
    targets_observed = coverage.get("observed_msprof_targets")
    if expected_seen != expected_candidates or evaluated != expected_candidates:
        raise RuntimeError(
            "fast response candidate coverage "
            f"expected={expected_seen!r} evaluated={evaluated!r} "
            f"request={expected_candidates}"
        )
    if targets_expected != expected_msprof_targets or targets_observed != expected_msprof_targets:
        raise RuntimeError(
            "fast response msprof coverage "
            f"expected={targets_expected!r} observed={targets_observed!r} "
            f"request={expected_msprof_targets}"
        )


def _is_fast_transient_build_lock(execution: _V2GateExecution) -> bool:
    if execution.transport_error:
        low = execution.transport_error.lower()
        return "resource busy" in low and "build:" in low
    if execution.verified:
        return False
    row, _ = _gate_observation(execution.report, AGENT_GATES_DEVICE_FAST_GATE)
    status = str(
        (row or {}).get("recordedStatus")
        or (row or {}).get("status")
        or execution.report.get("status")
        or ""
    ).upper()
    if status != "BLOCKED_ENV":
        return False
    failures_txt = (
        (_gate_failure_text(row) if row is not None else "")
        + " "
        + str(execution.verification.get("failures") or "")
    ).lower()
    return "resource busy" in failures_txt and "build:" in failures_txt


def _is_fast_occupancy_blocked(
    execution: _V2GateExecution,
    blocked_category: str,
    receipt_cat: str,
    eff_dev_fault: bool,
) -> tuple[bool, str]:
    if eff_dev_fault:
        return False, ""
    row, _ = _gate_observation(execution.report, AGENT_GATES_DEVICE_FAST_GATE)
    row_status = _gate_recorded_status(row).upper()
    report_status = str(execution.report.get("status") or "").upper()
    fail_txt = _gate_failure_text(row) if row is not None else ""
    ver_fail = str(execution.verification.get("failures") or "")
    transport = str(execution.transport_error or "")
    is_occ = (
        blocked_category == "occupancy"
        or receipt_cat == "occupancy"
        or row_status == "BLOCKED_ENV"
        or report_status == "BLOCKED_ENV"
        or "BLOCKED_EXIT_CODE: 75" in fail_txt
        or "BLOCKED_EXIT_CODE: 75" in ver_fail
        or "BLOCKED_EXIT_CODE: 75" in transport
        or "BLOCKED_ENV" in transport
    )
    if not is_occ:
        return False, ""
    reason = (
        blocked_category
        or receipt_cat
        or fail_txt.strip()
        or ver_fail.strip()
        or transport.strip()
        or report_status
        or "BLOCKED_ENV(75)"
    )
    return True, reason[:200]


def _submit_fast_gate(
    *,
    agent_gates_bin: str,
    root: Path,
    workspace: str,
    request_id: str,
    request_relative_path: str,
    request_path: Path,
    response_path: Path,
    request_sha256: str,
    mode: str,
    timeout_sec: float,
    device_id: str,
    device_pool: Sequence[str] = (),
    env_vals: Mapping[str, str] | None = None,
) -> tuple[Mapping[str, object], _V2GateExecution]:
    """Submit the device-fast target with idle card selection and contamination retest."""
    if _sha256_file(request_path) != request_sha256:
        raise RuntimeError("fast request file changed after it was written")
    backoff = (8.0, 12.0, 16.0, 20.0, 25.0, 30.0)
    max_retries = 14
    occupancy_backoff = (4.0, 6.0, 8.0, 10.0, 15.0)
    max_pool_passes = 16
    occupancy_budget = min(max(float(timeout_sec), 10.0), 240.0)
    total_occupancy_sleep = 0.0
    rejected_cards: set[str] = set()
    selected_card = ""
    saw_device_fault = False
    occupancy_attempts = 0
    occupancy_reasons: list[str] = []
    execution: _V2GateExecution | None = None
    for pass_idx in range(max_pool_passes):
        candidates = _call_order_candidates_by_idle_preflight(
            device_id,
            device_pool,
            excluded=rejected_cards,
            env_vals=env_vals,
        )
        if not candidates:
            if not rejected_cards and str(device_id or "").strip():
                candidates = (str(device_id).strip(),)
            else:
                break
        contaminated_in_pass = False
        for idx, card in enumerate(candidates):
            attempt_exec: _V2GateExecution | None = None
            for retry_idx in range(max_retries + 1):
                attempt_exec = _run_v2_gate(
                    agent_gates_bin=agent_gates_bin,
                    root=root,
                    workspace=workspace,
                    gate_id=AGENT_GATES_DEVICE_FAST_GATE,
                    target=f"fast/{request_id}",
                    candidate_path=request_relative_path,
                    timeout_sec=timeout_sec,
                    device_id=card,
                )
                if not _is_fast_transient_build_lock(attempt_exec) or retry_idx >= max_retries:
                    break
                wait_sec = backoff[min(retry_idx, len(backoff) - 1)]
                logger.info(
                    "FAST_GATE_RETRY_BUILD_LOCK request=%s retry=%d/%d wait=%.1fs",
                    request_id,
                    retry_idx + 1,
                    max_retries,
                    wait_sec,
                )
                time.sleep(wait_sec)
            assert attempt_exec is not None
            execution = attempt_exec
            blocked_category = _blocked_env_category(
                attempt_exec.report,
                (AGENT_GATES_DEVICE_FAST_GATE,),
            )
            is_dev_fault, is_dev_rejected, receipt_cat = _inspect_v2_device_fault(
                attempt_exec.report,
                attempt_exec.verification,
            )
            eff_dev_fault = bool(
                is_dev_fault or blocked_category == "device_fault" or receipt_cat == "device_fault"
            )
            eff_rejected = bool(attempt_exec.device_rejected or is_dev_rejected or eff_dev_fault)
            is_occ_blocked, occ_reason = _is_fast_occupancy_blocked(
                attempt_exec,
                blocked_category,
                receipt_cat,
                eff_dev_fault,
            )
            if eff_rejected or is_occ_blocked:
                response_path.unlink(missing_ok=True)
                if eff_rejected:
                    rejected_cards.add(card)
                if eff_dev_fault:
                    saw_device_fault = True
                if is_occ_blocked and not eff_rejected:
                    occupancy_attempts += 1
                    if occ_reason and occ_reason not in occupancy_reasons:
                        occupancy_reasons.append(occ_reason)
                reason = (
                    "device_fault"
                    if eff_dev_fault
                    else ("device_rejected" if eff_rejected else "occupancy")
                )
                next_card = candidates[idx + 1] if idx + 1 < len(candidates) else "backoff"
                logger.warning(
                    "FAST_GATE_CONTAMINATION_RETEST request=%s tainted_device=%s "
                    "reason=%s next_card=%s",
                    request_id,
                    card,
                    reason,
                    next_card,
                )
                contaminated_in_pass = True
                continue
            selected_card = card
            break
        else:
            if (
                contaminated_in_pass
                and pass_idx + 1 < max_pool_passes
                and total_occupancy_sleep < occupancy_budget
            ):
                remaining = _normalize_device_pool(
                    device_id,
                    device_pool,
                    excluded=rejected_cards,
                )
                if remaining:
                    wait_sec = occupancy_backoff[min(pass_idx, len(occupancy_backoff) - 1)]
                    time.sleep(wait_sec)
                    total_occupancy_sleep += wait_sec
                    continue
        break
    assert execution is not None
    if not selected_card and occupancy_attempts > 0 and not saw_device_fault:
        raise OccupancyExhaustedError(occupancy_attempts, tuple(occupancy_reasons))
    if not selected_card and (saw_device_fault or rejected_cards):
        raise AgentGatesDeviceFaultError(
            "AgentGates device-fast rejected all candidate NPU cards "
            f"(rejected={sorted(rejected_cards)}, detail={execution.transport_error or 'device_fault'})",
            fault_plane="INFRA_DEVICE_FAULT",
            fault_code="NPU_DEVICE_TSD_FAULT",
            failed_tier="l2_npu",
            device_id=execution.device_id or str(device_id or "").strip(),
            rejected_cards=rejected_cards,
        )
    if execution.transport_error:
        raise RuntimeError(f"AgentGates device-fast batch failed: {execution.transport_error}")
    if not execution.verified:
        failures = execution.verification.get("failures")
        response_detail = ""
        if response_path.is_file():
            try:
                failed_raw = json.loads(response_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                failed_raw = None
            if isinstance(failed_raw, Mapping):
                status = str(failed_raw.get("status", "")).strip()
                error = str(failed_raw.get("error", "")).strip()
                if status:
                    response_detail = f"; response status {status}: {error or 'no detail'}"
        raise RuntimeError(
            "AgentGates device-fast receipt did not verify: "
            f"{failures or 'unsatisfied'}{response_detail}"
        )
    if not _source_identity_matches(execution.verification, root):
        raise RuntimeError(
            "AgentGates device-fast verified source identity does not match requested source"
        )
    if not response_path.is_file():
        row, _observation = _gate_observation(execution.report, AGENT_GATES_DEVICE_FAST_GATE)
        detail = (
            (row.get("failures") if row is not None else None)
            or execution.report.get("status")
            or "missing response"
        )
        raise RuntimeError(f"AgentGates device-fast batch produced no response: {detail}")
    raw = json.loads(response_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("AgentGates device-fast response must be a JSON object")
    status = _require_fast_envelope(
        raw,
        request_id=request_id,
        request_sha256=request_sha256,
        mode=mode,
    )
    _raise_for_fast_status(status, raw)
    probe_card = selected_card or execution.device_id or str(device_id or "").strip()
    load = _probe_host_card_load(probe_card, env_vals=env_vals) if probe_card else None
    selected_card_ext_hbm_mb = int(load[1]) if (load is not None and load[0] < 3) else 0
    if selected_card_ext_hbm_mb > 0:
        raw.setdefault("_selected_card_ext_hbm_mb", selected_card_ext_hbm_mb)
    return raw, execution


def _fast_held_seconds(report: Mapping[str, object]) -> float:
    _row, observation = _gate_observation(report, AGENT_GATES_DEVICE_FAST_GATE)
    window = observation.get("deviceWindow")
    if not isinstance(window, Mapping):
        return 0.0
    held = window.get("heldSeconds")
    if isinstance(held, bool) or not isinstance(held, (int, float)):
        return 0.0
    return float(held)


def _source_identity_matches(verification: Mapping[str, object], root: Path) -> bool:
    identity = verification.get("current_source_identity")
    identity_root = identity.get("root") if isinstance(identity, Mapping) else None
    return bool(identity_root) and Path(str(identity_root)).resolve() == root


def _l3_evidence_payload(observation: Mapping[str, object]) -> Mapping[str, object]:
    for key in ("l3Evidence", "evidence"):
        candidate = observation.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    return observation


def _l3_signed_failure_category(
    observation: Mapping[str, object],
    evidence: Mapping[str, object],
) -> str:
    """Return the adapter-signed category, falling back to the structured kind."""
    for container in (observation, evidence):
        raw = container.get("failureCategory")
        if raw is not None and str(raw).strip():
            return str(raw).strip().lower()
    structured = evidence.get("failure")
    if isinstance(structured, Mapping):
        return str(structured.get("kind") or "").strip().lower()
    return FAILURE_KIND_INFRASTRUCTURE


def _l3_structured_failure(
    observation: Mapping[str, object],
    evidence: Mapping[str, object],
) -> Mapping[str, object] | None:
    """Return the producer's highest-precedence structured failure, if any."""
    for container in (evidence, observation):
        structured = container.get("failure")
        if isinstance(structured, Mapping) and structured:
            return structured
    return None


def _candidate_failed_node_names(rows: Sequence[Mapping[str, object]]) -> set[str]:
    """Return node names that carry an explicit structured candidate failure row."""
    return {
        str(row.get("node") or "").strip()
        for row in rows
        if str(row.get("kind") or "").strip().lower() == "candidate"
        and str(row.get("node") or "").strip()
    }


def _is_collateral_missing_case_row(
    row: Mapping[str, object],
    candidate_failed_nodes: set[str],
) -> bool:
    """Return True only for collateral missing_case rows on a candidate-failed node."""
    if str(row.get("kind") or "").strip().lower() != "coverage":
        return False
    if str(row.get("category") or "").strip().lower() != "missing_case":
        return False
    node_name = str(row.get("node") or "").strip()
    return bool(node_name and node_name in candidate_failed_nodes)


def _l3_candidate_numeric_failure(
    marker: Mapping[str, object],
    evidence: Mapping[str, object],
    raw_category: str,
) -> bool:
    """Whether a producer marker binds a numeric regression to this candidate.

    A bare failed node/case is never enough. The marker must name a candidate
    comparison failure, and the signed summary must show that same node and
    case in a failed state.
    """
    if raw_category != "numeric":
        return False
    expected = {
        "kind": "candidate",
        "category": "numeric",
        "role": "candidate",
        "stage": "comparison",
    }
    for key, value in expected.items():
        if str(marker.get(key) or "").strip().lower() != value:
            return False
    if not str(marker.get("comparator") or "").strip():
        return False
    node_name = marker.get("node")
    case_id = marker.get("case")
    if not isinstance(node_name, str) or not node_name or case_id is None:
        return False
    comparisons = marker.get("comparisons")
    if not isinstance(comparisons, Sequence) or isinstance(comparisons, (str, bytes)):
        return False
    if not any(
        isinstance(row, Mapping) and row.get("passed") is False
        for row in comparisons
    ):
        return False
    failures = evidence.get("failures")
    if isinstance(failures, Sequence) and not isinstance(failures, (str, bytes)):
        failure_rows = [row for row in failures if isinstance(row, Mapping)]
        candidate_nodes = _candidate_failed_node_names([marker, *failure_rows])
        blocked = {"infrastructure", "coverage"}
        if any(
            str(row.get("kind") or "").strip().lower() in blocked
            and not _is_collateral_missing_case_row(row, candidate_nodes)
            for row in failure_rows
        ):
            return False
    nodes = evidence.get("nodes")
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
        return False
    for node in nodes:
        if not isinstance(node, Mapping) or node.get("name") != node_name:
            continue
        if str(node.get("status") or "").strip().lower() not in {"failed", "error"}:
            return False
        cases = node.get("cases")
        if not isinstance(cases, Mapping):
            return False
        if str(cases.get(str(case_id)) or "").strip().lower() not in {"failed", "error"}:
            return False
        return True
    return False


_CANDIDATE_KERNEL_STAGES = frozenset({"kernel", "compile", "cache_idempotence"})
_CANDIDATE_KERNEL_DETAIL_SIGNALS = (
    "exceeded vf stack size",
    "ascend compilation failed",
    "bisheng",
    "ld.lld",
    "undefined symbol",
    "tvmerror",
    "l3 cache idempotence violation",
    "syntaxerror",
    "nameerror",
    "typeerror",
    "attributeerror",
    "keyerror",
    "valueerror",
    "importerror",
    "ub overflow",
    "out of ubuf",
    "kernel launch",
)
_INFRA_KERNEL_DETAIL_EXCLUDES = (
    "device offline",
    "device unavailable",
    "npu-smi",
    "drverror",
    "e39999",
    "ez9999",
    "aclrtgetdevice",
    "timeout",
    "reference cache",
    "no space left",
    "permission denied",
    "segmentation fault",
    "bus error",
)


def _is_candidate_kernel_marker(
    row: Mapping[str, object],
    *,
    node_records: Mapping[str, Mapping[str, object]],
) -> bool:
    role = str(row.get("role") or "").strip().lower()
    if role == "baseline":
        return False
    stage = str(row.get("stage") or "").strip().lower()
    if stage not in _CANDIDATE_KERNEL_STAGES:
        return False
    node_name = row.get("node")
    if not isinstance(node_name, str) or not node_name:
        return False
    node = node_records.get(node_name)
    if not isinstance(node, Mapping):
        return False
    if str(node.get("status") or "").strip().lower() not in {"failed", "error"}:
        return False
    case_id = row.get("case")
    if case_id is not None:
        cases = node.get("cases")
        if (
            not isinstance(cases, Mapping)
            or str(cases.get(str(case_id)) or "").strip().lower() not in {"failed", "error"}
        ):
            return False
    detail_text = f"{row.get('exceptionType') or ''} {row.get('detail') or ''}".lower()
    if any(excl in detail_text for excl in _INFRA_KERNEL_DETAIL_EXCLUDES):
        return False
    kind = str(row.get("kind") or "").strip().lower()
    if kind == "candidate" and (role == "candidate" or stage in {"kernel", "compile"}):
        return True
    return any(sig in detail_text for sig in _CANDIDATE_KERNEL_DETAIL_SIGNALS)


def _l3_candidate_kernel_failure(
    structured: Mapping[str, object],
    evidence: Mapping[str, object],
    raw_category: str,
) -> bool:
    """Return True when L3 evidence unambiguously binds a candidate compile/kernel failure."""
    if raw_category not in {"candidate", "infrastructure", "environment"}:
        return False
    rows: list[Mapping[str, object]] = []
    raw_failures = evidence.get("failures")
    if isinstance(raw_failures, Sequence) and not isinstance(raw_failures, (str, bytes)):
        for item in raw_failures:
            if isinstance(item, Mapping):
                rows.append(item)
    if not rows:
        rows.append(structured)
    nodes = evidence.get("nodes")
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
        return False
    node_records = {
        str(node.get("name")): node
        for node in nodes
        if isinstance(node, Mapping) and node.get("name")
    }
    candidate_nodes = {
        str(row.get("node") or "").strip()
        for row in rows
        if str(row.get("node") or "").strip()
        and (
            str(row.get("kind") or "").strip().lower() == "candidate"
            or _is_candidate_kernel_marker(row, node_records=node_records)
        )
    }
    matched = False
    for row in rows:
        if _is_candidate_kernel_marker(row, node_records=node_records):
            matched = True
            continue
        if _is_collateral_missing_case_row(row, candidate_nodes):
            continue
        kind = str(row.get("kind") or "").strip().lower()
        if kind in {"infrastructure", "coverage"}:
            return False
    return matched


def _l3_failure_kind(
    observation: Mapping[str, object],
    evidence: Mapping[str, object],
    coverage: Mapping[str, object] | None,
) -> str:
    """Classify a candidate defect only from explicit signed producer evidence.

    A failed JUnit/node/case row, a bare failure category, or a marker whose
    role/node/case/comparison does not line up is infrastructure. Missing or
    skipped coverage, interruptions, write errors and unreadable snapshots
    preserve the pending candidate and never consume stagnation.
    """
    _ = coverage
    interruption = evidence.get("interruption")
    if isinstance(interruption, Mapping):
        error_category = str(interruption.get("errorCategory") or "").strip()
        if error_category or interruption.get("signal") or interruption.get("interrupted"):
            return FAILURE_KIND_INFRASTRUCTURE
    write_errors = evidence.get("writeErrors")
    if (
        isinstance(write_errors, Sequence)
        and not isinstance(write_errors, (str, bytes))
        and len(write_errors) > 0
    ):
        return FAILURE_KIND_INFRASTRUCTURE
    phase = str(evidence.get("phase") or "").strip().lower()
    if phase != "finished":
        return FAILURE_KIND_INFRASTRUCTURE
    timing = evidence.get("timing")
    if isinstance(timing, Mapping) and not timing.get("finishedAt"):
        return FAILURE_KIND_INFRASTRUCTURE
    interpreter = evidence.get("interpreter")
    if isinstance(interpreter, Mapping) and interpreter.get("match") is False:
        return FAILURE_KIND_INFRASTRUCTURE
    ref_cache = evidence.get("referenceCache")
    if isinstance(ref_cache, Mapping):
        if ref_cache.get("errors"):
            return FAILURE_KIND_INFRASTRUCTURE
        if ref_cache.get("required") is True and ref_cache.get("usable") is False:
            return FAILURE_KIND_INFRASTRUCTURE
    if isinstance(coverage, Mapping) and coverage.get("referenceCacheProblem"):
        return FAILURE_KIND_INFRASTRUCTURE
    raw = _l3_signed_failure_category(observation, evidence)
    structured = _l3_structured_failure(observation, evidence)
    if structured is None:
        return FAILURE_KIND_INFRASTRUCTURE
    if _l3_candidate_numeric_failure(structured, evidence, raw):
        return FAILURE_KIND_CANDIDATE
    if _l3_candidate_kernel_failure(structured, evidence, raw):
        return FAILURE_KIND_CANDIDATE
    return FAILURE_KIND_INFRASTRUCTURE


def classify_l3_observation_failure(
    observation: Mapping[str, object],
) -> tuple[str, bool]:
    """Return ``(failure_kind, pending)`` for a signed L3 gate observation."""
    evidence = _l3_evidence_payload(observation)
    coverage_raw = evidence.get("coverage")
    kind = _l3_failure_kind(
        observation,
        evidence,
        coverage_raw if isinstance(coverage_raw, Mapping) else None,
    )
    return kind, kind != FAILURE_KIND_CANDIDATE


def _parse_l3_case_num(raw: object) -> int | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw >= 0 else None
    if isinstance(raw, str):
        txt = raw.strip()
        if txt.isdigit():
            return int(txt)
    return None


def _extract_l3_failed_case_nums(evidence: Mapping[str, object]) -> tuple[int, ...]:
    """Extract sorted deduplicated failed or errored case indices from L3 evidence."""
    found: set[int] = set()

    def _add_case(raw: object) -> None:
        parsed = _parse_l3_case_num(raw)
        if parsed is not None:
            found.add(parsed)

    ev_failure = evidence.get("failure")
    if isinstance(ev_failure, Mapping):
        _add_case(ev_failure.get("case"))

    failures_seq = evidence.get("failures")
    if isinstance(failures_seq, Sequence) and not isinstance(failures_seq, (str, bytes)):
        for item in failures_seq:
            if isinstance(item, Mapping):
                _add_case(item.get("case"))

    nodes = evidence.get("nodes")
    if isinstance(nodes, Sequence) and not isinstance(nodes, (str, bytes)):
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            node_fail = node.get("failure")
            if isinstance(node_fail, Mapping):
                _add_case(node_fail.get("case"))
            cases = node.get("cases")
            if isinstance(cases, Mapping):
                for case_key, case_status in cases.items():
                    if str(case_status or "").strip().lower() in {"failed", "error"}:
                        _add_case(case_key)

    return tuple(sorted(found))


def _l3_failure_stage(evidence: Mapping[str, object]) -> str:
    primary = _select_primary_l3_failure(evidence)
    if not isinstance(primary, Mapping):
        return ""
    return str(primary.get("stage") or "").strip().lower()


def l3_failure_result_from_observation(
    observation: Mapping[str, object],
    *,
    run_id: str = "",
    receipt_path: str = "",
    execution_id: str = "",
    candidate_sha256: str = "",
    target_path: str = "",
    device_id: str = "",
    state: str = "",
    execution_terminal: bool = True,
    blocked_env_category: str = "",
) -> L3AcceptanceResult:
    """Map an adapter L3 observation to the campaign failure result."""
    evidence = _l3_evidence_payload(observation)
    kind, pending = classify_l3_observation_failure(observation)
    return L3AcceptanceResult(
        passed=False,
        run_id=run_id,
        receipt_path=receipt_path,
        failure_kind=kind,
        failure_category=_l3_signed_failure_category(observation, evidence),
        pending=pending,
        execution_id=execution_id,
        candidate_sha256=candidate_sha256,
        target_path=target_path,
        device_id=device_id,
        state=state,
        execution_terminal=execution_terminal,
        blocked_env_category=blocked_env_category,
        failed_case_nums=_extract_l3_failed_case_nums(evidence),
        failure_stage=_l3_failure_stage(evidence),
    )


def _is_coverage_only_failure(row: object) -> bool:
    if not isinstance(row, Mapping) or not row:
        return True
    category = str(row.get("category") or "").strip().lower()
    stage = str(row.get("stage") or "").strip().lower()
    return category == "missing_case" or stage == "coverage"


def _select_primary_l3_failure(evidence: Mapping[str, object]) -> Mapping[str, object] | None:
    nodes = evidence.get("nodes")
    if isinstance(nodes, Sequence) and not isinstance(nodes, (str, bytes)):
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            node_fail = node.get("failure")
            if isinstance(node_fail, Mapping) and not _is_coverage_only_failure(node_fail):
                return node_fail
    failures_seq = evidence.get("failures")
    if isinstance(failures_seq, Sequence) and not isinstance(failures_seq, (str, bytes)):
        for item in failures_seq:
            if isinstance(item, Mapping) and not _is_coverage_only_failure(item):
                return item
    ev_failure = evidence.get("failure")
    if isinstance(ev_failure, Mapping) and ev_failure and not _is_coverage_only_failure(ev_failure):
        return ev_failure
    if isinstance(ev_failure, Mapping) and ev_failure:
        return ev_failure
    if isinstance(failures_seq, Sequence) and not isinstance(failures_seq, (str, bytes)):
        for item in failures_seq:
            if isinstance(item, Mapping) and item:
                return item
    return None


def _format_l3_failure_detail(
    row: Mapping[str, object],
    observation: Mapping[str, object],
    evidence: Mapping[str, object],
) -> str:
    """Extract concise compiler/runtime/comparison diagnostics from a failed L3 gate."""
    base_raw = row.get("failures") or observation.get("error") or "no detail"
    base_text = str(base_raw)
    failure = _select_primary_l3_failure(evidence)
    if not isinstance(failure, Mapping):
        return base_text

    parts: list[str] = []
    case_id = failure.get("case")
    stage = failure.get("stage")
    exc_type = failure.get("exceptionType")
    header_bits = [
        f"case={case_id}" if case_id is not None else "",
        f"stage={stage}" if stage else "",
        f"type={exc_type}" if exc_type else "",
    ]
    header = " ".join(bit for bit in header_bits if bit)
    if header:
        parts.append(header)

    detail_raw = str(failure.get("detail") or "").strip()
    if detail_raw:
        marker = "Compiler output:"
        idx = detail_raw.find(marker)
        if idx >= 0:
            parts.append(detail_raw[idx : idx + 900])
        elif len(detail_raw) <= 900:
            parts.append(detail_raw)
        else:
            parts.append(f"{detail_raw[:420]} ... {detail_raw[-450:]}")

    comparisons = failure.get("comparisons")
    if isinstance(comparisons, Sequence) and not isinstance(comparisons, (str, bytes)):
        failing_cmp = next(
            (c for c in comparisons if isinstance(c, Mapping) and c.get("passed") is False),
            None,
        )
        if failing_cmp is None:
            failing_cmp = next(
                (
                    c
                    for c in comparisons
                    if isinstance(c, Mapping)
                    and (str(c.get("message") or "").strip() or str(c.get("detail") or "").strip())
                ),
                None,
            )
        if isinstance(failing_cmp, Mapping):
            cmp_bits: list[str] = []
            for key in ("name", "max_abs_diff", "max_rel_diff", "atol", "rtol", "detail", "message"):
                val = failing_cmp.get(key)
                if val is not None and str(val).strip():
                    val_str = str(val).strip()
                    if key in ("message", "detail") and len(val_str) > 600:
                        val_str = val_str[:600]
                    cmp_bits.append(f"{key}={val_str}")
            if cmp_bits:
                parts.append("comparison(" + ", ".join(cmp_bits) + ")")

    extracted = " | ".join(part for part in parts if part)
    if not extracted:
        return base_text
    return f"{base_text} | {extracted}"


def run_l3_full_contract_via_agent_gates(
    *,
    operator_name: str | None = None,
    candidate_module_path: str | Path | None = None,
    source_root: str | Path | None = None,
    env_file: str | Path | None = None,
    device_id: str = "1",
    device_pool: Sequence[str] = (),
    timeout_sec: int = DEFAULT_L3_PROMOTION_WAIT_SEC,
    resume_execution_id: str = "",
    resume_run_id: str = "",
    on_execution_started: Callable[[str, str], None] | None = None,
) -> L3AcceptanceResult:
    """Run the L3 promotion gate through AgentGates and validate its signed promotion evidence.

    The adapter owns the interpreter, CANN/TileLang setup, cache policy and device.
    TileFoundry supplies only the canonical target and the workspace-relative candidate;
    the signed receipt must bind the actual imported candidate digest and full coverage.
    """
    env_vals: dict[str, str] | None = None
    try:
        resolved_env = _resolve_env_file_path(env_file)
    except FileNotFoundError:
        resolved_env = None
    if resolved_env is not None:
        env_vals = parse_simple_env_file(resolved_env)
    result_device = str(device_id or "")
    started = time.monotonic()

    def _fail(
        message: str,
        *,
        execution_id: str = "",
        run_id: str = "",
        receipt: str = "",
        result_device_id: str = "",
        state: str = "",
        execution_terminal: bool = True,
        blocked_env_category: str = "",
    ) -> L3AcceptanceResult:
        return L3AcceptanceResult(
            passed=False,
            run_id=run_id,
            receipt_path=receipt,
            elapsed_sec=time.monotonic() - started,
            error_message=message[-1000:],
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
            execution_id=execution_id,
            candidate_sha256="",
            target_path="",
            device_id=result_device_id or result_device,
            state=state,
            execution_terminal=execution_terminal,
            blocked_env_category=blocked_env_category,
        )

    if operator_name is None or candidate_module_path is None:
        return _fail("L3 promotion requires an operator name and candidate module path")
    try:
        target_task = get_operator_task(operator_name)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return _fail(f"L3 promotion target resolution failed: {exc}")
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    candidate = Path(candidate_module_path).resolve()
    relative = _workspace_relative_candidate(root, candidate)
    if relative is None:
        return _fail(f"L3 candidate {candidate} is outside workspace source root {root}")
    if not candidate.is_file():
        return _fail(f"L3 candidate module {candidate} does not exist")
    candidate_digest = _sha256_file(candidate)
    try:
        agent_gates_bin = _resolve_agent_gates_bin(root)
        workspace = _agent_gates_workspace_name(root)
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as exc:
        return _fail(str(exc))
    target_path = str(candidate)

    blocked_env_category = ""
    if resume_execution_id or resume_run_id:
        execution = _run_v2_gate(
            agent_gates_bin=agent_gates_bin,
            root=root,
            workspace=workspace,
            gate_id=AGENT_GATES_L3_PROMOTION_GATE,
            target=target_task,
            candidate_path=relative,
            timeout_sec=timeout_sec,
            device_id=result_device,
            resume_execution_id=resume_execution_id,
            resume_run_id=resume_run_id,
            on_execution_started=on_execution_started,
        )
        blocked_env_category = _blocked_env_category(
            execution.report,
            (AGENT_GATES_L3_PROMOTION_GATE,),
        )
    else:
        rejected_cards: set[str] = set()
        initial_device_id = result_device
        occupancy_backoff = (15.0, 30.0, 45.0)
        max_pool_passes = 4
        occupancy_budget = 120.0
        total_occupancy_sleep = 0.0
        execution_opt: _V2GateExecution | None = None
        for pass_idx in range(max_pool_passes):
            candidates = _call_order_candidates_by_idle_preflight(
                initial_device_id,
                device_pool,
                excluded=rejected_cards,
                env_vals=env_vals,
            )
            if not candidates:
                if not rejected_cards and initial_device_id:
                    candidates = (initial_device_id,)
                else:
                    break
            saw_occupancy_in_pass = False
            for idx, card in enumerate(candidates):
                result_device = card
                attempt_exec = _run_v2_gate(
                    agent_gates_bin=agent_gates_bin,
                    root=root,
                    workspace=workspace,
                    gate_id=AGENT_GATES_L3_PROMOTION_GATE,
                    target=target_task,
                    candidate_path=relative,
                    timeout_sec=timeout_sec,
                    device_id=card,
                    resume_execution_id="",
                    resume_run_id="",
                    on_execution_started=on_execution_started,
                )
                execution_opt = attempt_exec
                blocked_env_category = _blocked_env_category(
                    attempt_exec.report,
                    (AGENT_GATES_L3_PROMOTION_GATE,),
                )
                if attempt_exec.device_rejected or blocked_env_category == "occupancy":
                    if attempt_exec.device_rejected and card:
                        rejected_cards.add(card)
                    else:
                        saw_occupancy_in_pass = True
                    if idx + 1 < len(candidates):
                        logger.warning(
                            "L3_CONTAMINATION_RETEST task=%s tainted_device=%s "
                            "reason=%s next_card=%s",
                            target_task,
                            card,
                            "device_rejected" if attempt_exec.device_rejected else "occupancy",
                            candidates[idx + 1],
                        )
                        continue
                else:
                    break
            else:
                if (
                    saw_occupancy_in_pass
                    and pass_idx + 1 < max_pool_passes
                    and total_occupancy_sleep < occupancy_budget
                ):
                    remaining = _call_order_candidates_by_idle_preflight(
                        initial_device_id,
                        device_pool,
                        excluded=rejected_cards,
                        env_vals=env_vals,
                    )
                    if not remaining and not rejected_cards and initial_device_id:
                        remaining = (initial_device_id,)
                    if remaining:
                        wait_sec = min(
                            occupancy_backoff[min(pass_idx, len(occupancy_backoff) - 1)],
                            occupancy_budget - total_occupancy_sleep,
                        )
                        logger.warning(
                            "L3_CONTAMINATION_RETEST task=%s tainted_device=%s "
                            "reason=occupancy next_card=backoff",
                            target_task,
                            result_device,
                        )
                        time.sleep(wait_sec)
                        total_occupancy_sleep += wait_sec
                        continue
            break
        assert execution_opt is not None
        execution = execution_opt

    if execution.transport_error:
        return _fail(
            execution.transport_error,
            execution_id=execution.execution_id,
            run_id=execution.run_id,
            receipt=execution.receipt_path,
            result_device_id=execution.device_id,
            state=execution.state,
            execution_terminal=execution.execution_terminal,
            blocked_env_category=blocked_env_category,
        )
    row, observation = _gate_observation(execution.report, AGENT_GATES_L3_PROMOTION_GATE)
    if row is None:
        return _fail(
            "L3 signed receipt has no promotion gate observation",
            execution_id=execution.execution_id,
            run_id=execution.run_id,
            receipt=execution.receipt_path,
            result_device_id=execution.device_id,
            state=execution.state,
            execution_terminal=execution.execution_terminal,
            blocked_env_category=blocked_env_category,
        )
    evidence = _l3_evidence_payload(observation)
    coverage_raw = evidence.get("coverage")
    coverage = coverage_raw if isinstance(coverage_raw, Mapping) else None
    recorded_status = _gate_recorded_status(row)
    failed_case_nums = _extract_l3_failed_case_nums(evidence)
    failure_stage = _l3_failure_stage(evidence)

    def _reject(message: str, *, failure_kind: str) -> L3AcceptanceResult:
        return L3AcceptanceResult(
            passed=False,
            run_id=execution.run_id,
            receipt_path=execution.receipt_path,
            elapsed_sec=time.monotonic() - started,
            error_message=message[:1800],
            failure_kind=failure_kind,
            failure_category=_l3_signed_failure_category(observation, evidence),
            pending=failure_kind != FAILURE_KIND_CANDIDATE,
            execution_id=execution.execution_id,
            candidate_sha256=candidate_digest,
            target_path=target_path,
            device_id=execution.device_id or result_device,
            state=execution.state,
            execution_terminal=execution.execution_terminal,
            blocked_env_category=blocked_env_category,
            failed_case_nums=failed_case_nums,
            failure_stage=failure_stage,
        )

    if recorded_status != "PASS":
        return _reject(
            f"L3 gate recorded status {recorded_status or 'missing'}: "
            f"{_format_l3_failure_detail(row, observation, evidence)}",
            failure_kind=_l3_failure_kind(observation, evidence, coverage),
        )
    if not execution.verified:
        return _reject(
            f"L3 receipt did not verify: {execution.verification.get('failures') or 'unsatisfied'}",
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
        )
    if not _source_identity_matches(execution.verification, root):
        return _reject(
            "L3 verified source identity does not match requested source",
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
        )
    schema = str(evidence.get("schema", ""))
    profile = str(evidence.get("profile", ""))
    if schema != "tilefoundry-l3-evidence/v1" or profile != "l3-promotion":
        return _reject(
            f"L3 evidence schema/profile mismatch: schema={schema or 'missing'} "
            f"profile={profile or 'missing'}",
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
        )
    target = evidence.get("target")
    declared_task = target.get("task") if isinstance(target, Mapping) else None
    if not declared_task or str(declared_task) != target_task:
        return _reject(
            f"L3 evidence target task {declared_task or 'missing'} does not match {target_task}",
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
        )
    target_module = target.get("modulePath") if isinstance(target, Mapping) else None
    if not target_module or Path(str(target_module)).resolve() != Path(target_path):
        return _reject(
            "L3 evidence target does not match requested target module",
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
        )
    declared = evidence.get("candidate")
    declared_path = declared.get("modulePath") if isinstance(declared, Mapping) else None
    declared_digest = declared.get("sha256") if isinstance(declared, Mapping) else None
    if not declared_path or Path(str(declared_path)).resolve() != candidate:
        return _reject(
            "L3 evidence candidate path does not match the promoted candidate",
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
        )
    if str(declared_digest or "") != candidate_digest:
        return _reject(
            "L3 evidence candidate digest does not match the promoted candidate",
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
        )
    interpreter = evidence.get("interpreter")
    if not isinstance(interpreter, Mapping) or interpreter.get("match") is not True:
        return _reject(
            "L3 evidence interpreter did not match the adapter-resolved interpreter",
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
        )
    if coverage is None or coverage.get("qualified") is not True:
        return _reject(
            "L3 evidence did not qualify full contract coverage",
            failure_kind=_l3_failure_kind(observation, evidence, coverage),
        )
    nodes = coverage.get("nodes") if isinstance(coverage.get("nodes"), Mapping) else {}
    cases = coverage.get("cases") if isinstance(coverage.get("cases"), Mapping) else {}
    assert isinstance(nodes, Mapping) and isinstance(cases, Mapping)
    if (
        nodes.get("missing")
        or nodes.get("failed")
        or nodes.get("skipped")
        or nodes.get("blocked")
        or cases.get("missing")
        or cases.get("unexpected")
        or coverage.get("referenceCacheProblem")
    ):
        return _reject(
            "L3 evidence coverage is incomplete: "
            f"{dict(nodes)} / {dict(cases)} / "
            f"referenceCacheProblem={coverage.get('referenceCacheProblem')}",
            failure_kind=_l3_failure_kind(observation, evidence, coverage),
        )
    expected_cases = _as_positive_int(cases.get("expected"))
    completed_cases = _as_positive_int(cases.get("completed"))
    if expected_cases <= 0 or completed_cases != expected_cases:
        return _reject(
            f"L3 evidence case coverage incomplete: {completed_cases}/{expected_cases}",
            failure_kind=_l3_failure_kind(observation, evidence, coverage),
        )
    return L3AcceptanceResult(
        passed=True,
        run_id=execution.run_id,
        receipt_path=execution.receipt_path,
        elapsed_sec=time.monotonic() - started,
        execution_id=execution.execution_id,
        candidate_sha256=candidate_digest,
        target_path=target_path,
        device_id=execution.device_id or result_device,
        state=execution.state,
        execution_terminal=execution.execution_terminal,
    )


class MicroProbeSources(Protocol):
    """Metadata and the two probe source texts a hardware request carries."""

    probe_id: str
    probe_kind: str
    factory_name: str
    baseline_probe_source: str
    optimized_probe_source: str
    line_count_baseline: int
    line_count_optimized: int
    tensor_shape: tuple[int, ...]
    tensor_dtype: str


def _prewarm_micro_probe_cache(
    spec: MicroProbeSources,
    *,
    source_root: Path,
    env_file: Path,
    tl_cache_dir: Path,
) -> None:
    env_vals = parse_simple_env_file(env_file)
    python_bin = (
        env_vals.get("TILEFOUNDRY_NPU_PYTHON")
        or os.environ.get("TILEFOUNDRY_NPU_PYTHON")
        or sys.executable
    )
    cann_setup = (env_vals.get("TILEFOUNDRY_NPU_CANN_SETUP") or "").strip()
    tl_source = (env_vals.get("TILEFOUNDRY_NPU_TILELANG_SOURCE") or "").strip()

    tl_cache_dir.mkdir(parents=True, exist_ok=True)
    run_env = dict(os.environ)
    run_env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    run_env["ASCEND_RT_VISIBLE_DEVICES"] = ""
    run_env["USE_CUDA"] = "0"
    run_env["TILELANG_CACHE_DIR"] = str(tl_cache_dir)
    run_env["TILELANG_KERNEL_CACHE_USE_LIB_STAMP"] = "1"
    run_env["TILELANG_DISABLE_CACHE"] = "0"
    run_env["PYTHONDONTWRITEBYTECODE"] = "1"
    pp_parts = [str(source_root), str(source_root / "src"), str(source_root / "targets" / "ascend")]
    if tl_source:
        pp_parts.insert(0, tl_source)
    run_env["PYTHONPATH"] = os.pathsep.join(pp_parts)

    factory_name = spec.factory_name
    sources = (
        ("base", spec.baseline_probe_source),
        ("opt", spec.optimized_probe_source),
    )
    script = (
        "import importlib, importlib.util, os, sys, tempfile\n"
        "from pathlib import Path\n"
        f"os.environ['TILELANG_CACHE_DIR'] = {str(tl_cache_dir)!r}\n"
        "os.environ['TILELANG_KERNEL_CACHE_USE_LIB_STAMP'] = '1'\n"
        "os.environ['TILELANG_DISABLE_CACHE'] = '0'\n"
        "import tilelang\n"
        "tl_env = importlib.import_module('tilelang.env')\n"
        "tl_kc = importlib.import_module('tilelang.cache.kernel_cache')\n"
        "tl_env.env.enable_cache()\n"
        "tl_kc.KernelCache._get_cache_namespace.cache_clear()\n"
        "tl_kc.KernelCache._get_tilelang_lib_stamp.cache_clear()\n"
        "tl_kc.KernelCache._create_dirs()\n"
        f"sources = {json.dumps(sources)}\n"
        f"factory = {factory_name!r}\n"
        "with tempfile.TemporaryDirectory(prefix='tf_probe_pre_') as td:\n"
        "    for tag, src in sources:\n"
        "        p = Path(td) / f'{tag}.py'\n"
        "        p.write_text(src, encoding='utf-8')\n"
        "        sp = importlib.util.spec_from_file_location(f'_tf_p_{tag}', str(p))\n"
        "        mod = importlib.util.module_from_spec(sp)\n"
        "        sp.loader.exec_module(mod)\n"
        "        fn = getattr(mod, factory)()\n"
        "        tilelang.compile(fn, target='ascend', out_idx=-1)\n"
    )
    with tempfile.TemporaryDirectory(prefix="tf_probe_cpu_") as tmp_dir:
        drv = Path(tmp_dir) / "prewarm_probe.py"
        drv.write_text(script, encoding="utf-8")
        if cann_setup and Path(cann_setup).is_file() and shutil.which("bisheng") is None:
            bash_cmd = (
                f"set +u && source {shlex.quote(cann_setup)} >/dev/null 2>&1 "
                f"&& exec {shlex.quote(python_bin)} {shlex.quote(str(drv))}"
            )
            argv = ["/bin/bash", "-c", bash_cmd]
        else:
            argv = [python_bin, str(drv)]
        try:
            subprocess.run(
                argv,
                text=True,
                capture_output=True,
                cwd=str(source_root),
                env=run_env,
                timeout=180,
            )
        except Exception as exc:
            logger.debug("Micro-probe CPU prewarm skipped: %s", exc)


def run_micro_probe_via_agent_gates(
    spec: MicroProbeSources,
    *,
    source_root: str | Path | None = None,
    env_file: str | Path | None = None,
    device_id: str = "1",
    device_pool: Sequence[str] = (),
    min_speedup: float = 1.08,
    timeout_sec: int = 360,
    prewarm_cache: bool = True,
) -> dict[str, float]:
    """Run hardware micro-probe verification through a typed AgentGates fast request."""
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    resolved_env = _resolve_env_file_path(env_file)
    ag_bin = _resolve_agent_gates_bin(root)
    env_vals = parse_simple_env_file(resolved_env)
    _reference_cache_dir, tl_cache_dir = _fast_cache_roots(root, env_vals)

    if prewarm_cache:
        _prewarm_micro_probe_cache(
            spec,
            source_root=root,
            env_file=resolved_env,
            tl_cache_dir=tl_cache_dir,
        )

    request_root = _fast_request_root(root, env_vals)
    request_id = uuid.uuid4().hex
    request_dir = request_root / request_id
    request_path = request_root / f"{request_id}.request.json"
    response_path = request_root / f"{request_id}.response.json"

    staged: dict[str, dict[str, dict[str, str]]] = {"micro_probe": {}}
    entries: list[dict[str, str]] = []
    for index, (candidate_id, source_text) in enumerate(
        (
            ("baseline", spec.baseline_probe_source),
            ("optimized", spec.optimized_probe_source),
        )
    ):
        staged_path = (
            request_dir / "candidates" / "micro_probe" / f"{index:03d}_{candidate_id}.py"
        )
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        staged_path.write_text(source_text, encoding="utf-8")
        staged_path.chmod(0o444)
        relative = staged_path.relative_to(root).as_posix()
        digest = _sha256_file(staged_path)
        staged["micro_probe"][candidate_id] = {
            "module_path": relative,
            "sha256": digest,
        }
        entries.append(
            {"candidate_id": candidate_id, "module_path": relative, "sha256": digest}
        )

    payload: dict[str, object] = {
        "schema": FAST_REQUEST_SCHEMA,
        "request_id": request_id,
        "mode": "micro_probe",
        "warmup": 1,
        "repeat": 1,
        "require_all_passed": False,
        "response_path": response_path.relative_to(root).as_posix(),
        "probe": {
            "probe_id": spec.probe_id,
            "probe_kind": spec.probe_kind,
            "factory_name": spec.factory_name,
            "line_count_baseline": int(spec.line_count_baseline),
            "line_count_optimized": int(spec.line_count_optimized),
            "tensor_shape": [int(dim) for dim in spec.tensor_shape],
            "tensor_dtype": spec.tensor_dtype,
            "min_speedup": float(min_speedup),
        },
        "candidates": entries,
    }
    request_sha256 = _write_fast_request(request_path, payload)
    raw_resp, _execution = _submit_fast_gate(
        agent_gates_bin=ag_bin,
        root=root,
        workspace=_agent_gates_workspace_name(root),
        request_id=request_id,
        request_relative_path=request_path.relative_to(root).as_posix(),
        request_path=request_path,
        response_path=response_path,
        request_sha256=request_sha256,
        mode="micro_probe",
        timeout_sec=float(timeout_sec),
        device_id=str(device_id),
        device_pool=device_pool,
        env_vals=env_vals,
    )
    _verify_fast_bindings(raw_resp, staged)
    _verify_fast_coverage(raw_resp, expected_candidates=2, expected_msprof_targets=0)
    probe_row = _fast_response_mapping(raw_resp, "micro_probe")
    if probe_row.get("probe_id") != spec.probe_id:
        raise RuntimeError(
            "fast micro-probe response probe_id "
            f"{probe_row.get('probe_id')!r} != {spec.probe_id!r}"
        )
    metrics_raw = probe_row.get("metrics")
    if not isinstance(metrics_raw, Mapping):
        raise RuntimeError("fast micro-probe response metrics must be a mapping")
    metrics = {
        str(key): float(value)
        for key, value in metrics_raw.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if not metrics:
        raise RuntimeError("fast micro-probe response carried no numeric metrics")
    return metrics


def _agent_gates_evidence_root() -> Path:
    """Read the configured AgentGates evidence root used by the managed worker."""
    raw = _agent_gates_config()
    evidence_root = raw.get("evidence_root")
    if not isinstance(evidence_root, str) or not evidence_root.strip():
        raise ValueError("AgentGates evidence_root must be configured")
    return Path(evidence_root).expanduser().resolve()


def _extract_l4_metrics(
    parsed: Mapping[str, object],
    target_task: str,
) -> tuple[float, float, str, dict[int, float], dict[int, float]]:
    """Extract L4 aggregate and per-case metrics from signed gate observations."""
    avg_speedup = 0.0
    score = 0.0
    report_dir = ""
    per_case_speedups: dict[int, float] = {}
    per_case_elapsed: dict[int, float] = {}
    for gate_row in parsed.get("gates", ()):
        if not isinstance(gate_row, Mapping):
            continue
        if gate_row.get("gateId") != AGENT_GATES_L4_CANNBENCH_GATE:
            continue
        obs = gate_row.get("observations") or {}
        if not isinstance(obs, Mapping):
            continue
        avg_speedup = float(obs.get("avgSpeedup", 0.0))
        score = float(obs.get("score", 0.0))
        for res_row in obs.get("results", ()):
            if not isinstance(res_row, Mapping):
                continue
            if target_task and res_row.get("task") != target_task:
                continue
            report_dir = str(res_row.get("reportDir", ""))
            for case_row in res_row.get("cases", ()):
                if not isinstance(case_row, Mapping):
                    continue
                cid_str = str(case_row.get("case_id", ""))
                if "_" not in cid_str or not cid_str.rsplit("_", 1)[-1].isdigit():
                    continue
                cnum = int(cid_str.rsplit("_", 1)[-1])
                per_case_speedups[cnum] = float(case_row.get("speedup") or 0.0)
                per_case_elapsed[cnum] = float(case_row.get("elapsed_us") or 0.0)
    return avg_speedup, score, report_dir, per_case_speedups, per_case_elapsed


def _run_agent_gates_v2_cmd(
    cmd: Sequence[str],
    *,
    root: Path,
    timeout_sec: float,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    """Run one managed AgentGates command against the configured worker evidence."""
    proc = subprocess.run(
        [str(part) for part in cmd],
        text=True,
        capture_output=True,
        cwd=str(root),
        env=_agent_gates_cli_env(),
        timeout=timeout_sec,
    )
    return proc, _parse_agent_gates_json(proc.stdout)


def _l4_failure(
    error_message: str,
    *,
    failure_kind: str = FAILURE_KIND_INFRASTRUCTURE,
    execution_id: str = "",
    run_id: str = "",
    receipt_path: str = "",
    observation: Mapping[str, object] | None = None,
    device_id: str = "",
    state: str = "",
    execution_terminal: bool = True,
    blocked_env_category: str = "",
    device_rejected: bool = False,
    target_path: str = "",
    candidate_sha256: str = "",
    attempts: tuple[dict[str, str], ...] = (),
) -> L4CannbenchResult:
    """Return a failed L4 result without metrics or promotion evidence."""
    obs = observation or {}
    return L4CannbenchResult(
        passed=False,
        run_id=run_id,
        receipt_path=receipt_path,
        error_message=error_message[-1000:],
        failure_kind=failure_kind,
        execution_id=execution_id,
        official_report_valid=obs.get("officialReportValid") is True,
        all_cases_passed=obs.get("allCasesPassed") is True,
        all_timings_valid=obs.get("allTimingsValid") is True,
        case_count=_as_positive_int(obs.get("caseCount")),
        passed_cases=_as_positive_int(obs.get("passedCases")),
        device_id=device_id,
        state=state,
        execution_terminal=execution_terminal,
        blocked_env_category=blocked_env_category,
        device_rejected=device_rejected,
        target_path=target_path,
        candidate_sha256=candidate_sha256,
        attempts=attempts,
    )


def _l4_candidate_matches(raw: object, expected: str, root: Path) -> bool:
    text = str(raw or "").strip()
    if not text:
        return False
    if text.endswith(expected) and (
        text == expected or text.endswith("/" + expected)
    ):
        return True
    path = Path(text)
    return path.is_absolute() and path.resolve() == (root / expected).resolve()


def _validate_l4_report(
    execution: _V2GateExecution,
    *,
    root: Path,
    target_task: str,
    expected_candidate: str,
) -> tuple[str, str]:
    """Validate signed L4 evidence; return ``(error_message, failure_kind)``.

    Only a complete official case set where every case passed is a candidate
    verdict. Missing cases, a missing or non-numeric passed count, and missing
    timings are unknown coverage: they preserve the pending candidate instead of
    blaming it. Report validation flags and a short passed-case count are
    candidate failures; identity, directory and signature problems are not.
    """
    if not execution.verified:
        return (
            f"L4 receipt did not verify: {execution.verification.get('failures') or 'unsatisfied'}",
            FAILURE_KIND_INFRASTRUCTURE,
        )
    if not _source_identity_matches(execution.verification, root):
        return (
            "L4 verified source identity does not match requested source",
            FAILURE_KIND_INFRASTRUCTURE,
        )
    row, obs = _gate_observation(execution.report, AGENT_GATES_L4_CANNBENCH_GATE)
    if row is None:
        return ("L4 signed receipt has no cannbench gate observation", FAILURE_KIND_INFRASTRUCTURE)
    status = _gate_recorded_status(row)
    if status != "PASS":
        return (
            f"L4 gate recorded status {status or 'missing'}: "
            f"{row.get('failures') or obs.get('error') or 'no detail'}",
            FAILURE_KIND_CANDIDATE if status == "FAIL" else FAILURE_KIND_INFRASTRUCTURE,
        )
    flags = {
        "officialReportValid": obs.get("officialReportValid"),
        "allCasesPassed": obs.get("allCasesPassed"),
        "allTimingsValid": obs.get("allTimingsValid"),
    }
    bad_flags = [name for name, value in flags.items() if value is not True]
    if bad_flags:
        return (
            f"L4 official report validation failed: {', '.join(bad_flags)}",
            FAILURE_KIND_CANDIDATE,
        )
    case_count = _as_positive_int(obs.get("caseCount"))
    passed_raw = obs.get("passedCases")
    passed_cases = _as_positive_int(passed_raw)
    timed_cases = _as_positive_int(obs.get("timedCases"))
    if case_count <= 0:
        return ("L4 signed observation reports no official cases", FAILURE_KIND_INFRASTRUCTURE)
    if isinstance(passed_raw, bool) or not isinstance(passed_raw, (int, float)):
        return (
            "L4 signed observation has no valid passed-case count",
            FAILURE_KIND_INFRASTRUCTURE,
        )
    if passed_cases != case_count:
        return (
            f"L4 passed {passed_cases}/{case_count} official cases",
            FAILURE_KIND_CANDIDATE,
        )
    if timed_cases != case_count:
        return (
            f"L4 timed {timed_cases}/{case_count} official cases",
            FAILURE_KIND_INFRASTRUCTURE,
        )
    results = obs.get("results")
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        return ("L4 signed observation has no per-target results", FAILURE_KIND_INFRASTRUCTURE)
    matched = 0
    for res_row in results:
        if not isinstance(res_row, Mapping):
            continue
        if target_task and res_row.get("task") != target_task:
            continue
        matched += 1
        if res_row.get("reportValid") is not True:
            return (
                f"L4 official report is not valid for task {target_task or res_row.get('task')}",
                FAILURE_KIND_CANDIDATE,
            )
        if not str(res_row.get("reportDir", "")).strip():
            return ("L4 official report directory is missing", FAILURE_KIND_INFRASTRUCTURE)
        if not _l4_candidate_matches(res_row.get("candidate"), expected_candidate, root):
            return (
                f"L4 receipt candidate does not match {expected_candidate}",
                FAILURE_KIND_INFRASTRUCTURE,
            )
    if matched == 0:
        return (
            f"L4 signed observation has no result for task {target_task or '<any>'}",
            FAILURE_KIND_INFRASTRUCTURE,
        )
    return ("", "")


def run_l4_official_cannbench_via_agent_gates(
    *,
    operator_name: str | None = None,
    candidate_module_path: str | Path | None = None,
    workspace: str | None = None,
    source_root: str | Path | None = None,
    env_file: str | Path | None = None,
    device_id: str = "1",
    timeout_sec: int = DEFAULT_L4_CANNBENCH_WAIT_SEC,
    resume_execution_id: str = "",
    resume_run_id: str = "",
    on_execution_started: Callable[[str, str], None] | None = None,
) -> L4CannbenchResult:
    """Run official tilefoundry-npu-cannbench through AgentGates and verify its signed receipt.

    The campaign's actual entry always uses AgentGates: the adapter owns environment, device
    lease and cache policy, while the candidate module is bound by path and signed
    observation. The default wait budgets queueing, prep and the 3600 s official bound.
    Metrics are returned only after the official-final validation flags, case counts,
    source identity and candidate binding check out. A resume keeps its original card,
    and a retry on another card is a fresh call so no timings are stitched across cards.
    """
    _ = env_file
    result_device = str(device_id or "")
    if operator_name is None:
        return _l4_failure("L4 cannbench requires a canonical operator name", device_id=result_device)
    try:
        target_task = get_operator_task(operator_name)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return _l4_failure(
            f"L4 cannbench target resolution failed: {exc}", device_id=result_device
        )
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    if candidate_module_path is not None:
        expected_path = Path(candidate_module_path).resolve()
    else:
        expected_path = (root / get_operator_module_relpath(operator_name)).resolve()
    relative = _workspace_relative_candidate(root, expected_path)
    if relative is None:
        return _l4_failure(
            f"L4 candidate {expected_path} is outside workspace source root {root}",
            device_id=result_device,
        )
    try:
        candidate_digest = _sha256_file(expected_path)
    except OSError as exc:
        return _l4_failure(
            f"L4 candidate module {expected_path} cannot be read: {exc}",
            device_id=result_device,
            target_path=str(expected_path),
        )
    try:
        agent_gates_bin = _resolve_agent_gates_bin(root)
        resolved_workspace = workspace or _agent_gates_workspace_name(root)
    except (OSError, ValueError) as exc:
        return _l4_failure(str(exc), device_id=result_device)

    if resume_execution_id or resume_run_id:
        execution = _run_v2_gate(
            agent_gates_bin=agent_gates_bin,
            root=root,
            workspace=resolved_workspace,
            gate_id=AGENT_GATES_L4_CANNBENCH_GATE,
            target=target_task,
            candidate_path=relative,
            timeout_sec=timeout_sec,
            device_id=result_device,
            resume_execution_id=resume_execution_id,
            resume_run_id=resume_run_id,
            on_execution_started=on_execution_started,
        )
    else:
        backoff = (10.0, 15.0, 20.0, 25.0, 30.0)
        max_retries = 12
        execution_opt: _V2GateExecution | None = None
        for retry_idx in range(max_retries + 1):
            execution_opt = _run_v2_gate(
                agent_gates_bin=agent_gates_bin,
                root=root,
                workspace=resolved_workspace,
                gate_id=AGENT_GATES_L4_CANNBENCH_GATE,
                target=target_task,
                candidate_path=relative,
                timeout_sec=timeout_sec,
                device_id=result_device,
                resume_execution_id=resume_execution_id,
                resume_run_id=resume_run_id,
                on_execution_started=on_execution_started,
            )
            if not _is_fast_transient_build_lock(execution_opt) or retry_idx >= max_retries:
                break
            wait_sec = backoff[min(retry_idx, len(backoff) - 1)]
            logger.info(
                "L4_GATE_RETRY_BUILD_LOCK op=%s retry=%d/%d wait=%.1fs",
                operator_name,
                retry_idx + 1,
                max_retries,
                wait_sec,
            )
            time.sleep(wait_sec)
        assert execution_opt is not None
        execution = execution_opt
    _row, observation = _gate_observation(execution.report, AGENT_GATES_L4_CANNBENCH_GATE)
    blocked_env_category = _blocked_env_category(execution.report)
    if execution.transport_error:
        return _l4_failure(
            execution.transport_error,
            execution_id=execution.execution_id,
            run_id=execution.run_id,
            receipt_path=execution.receipt_path,
            device_id=execution.device_id or result_device,
            state=execution.state,
            execution_terminal=execution.execution_terminal,
            blocked_env_category=blocked_env_category,
            device_rejected=execution.device_rejected,
            target_path=str(expected_path),
            candidate_sha256=candidate_digest,
        )
    error, failure_kind = _validate_l4_report(
        execution,
        root=root,
        target_task=target_task,
        expected_candidate=relative,
    )
    if error:
        return _l4_failure(
            error,
            failure_kind=failure_kind,
            execution_id=execution.execution_id,
            run_id=execution.run_id,
            receipt_path=execution.receipt_path,
            observation=observation,
            device_id=execution.device_id or result_device,
            state=execution.state,
            execution_terminal=execution.execution_terminal,
            blocked_env_category=blocked_env_category,
            target_path=str(expected_path),
            candidate_sha256=candidate_digest,
        )
    avg_speedup, score, report_dir, speedups, elapsed = _extract_l4_metrics(
        execution.report, target_task
    )
    return L4CannbenchResult(
        passed=True,
        avg_speedup=avg_speedup,
        score=score,
        per_case_speedups=speedups,
        per_case_elapsed_us=elapsed,
        run_id=execution.run_id,
        receipt_path=execution.receipt_path,
        report_dir=report_dir,
        execution_id=execution.execution_id,
        official_report_valid=True,
        all_cases_passed=True,
        all_timings_valid=True,
        case_count=_as_positive_int(observation.get("caseCount")),
        passed_cases=_as_positive_int(observation.get("passedCases")),
        device_id=execution.device_id or result_device,
        state=execution.state,
        execution_terminal=execution.execution_terminal,
        target_path=str(expected_path),
        candidate_sha256=candidate_digest,
    )


__all__ = [
    "AGENT_GATES_ADAPTER",
    "AGENT_GATES_DEVICE_FAST_GATE",
    "AGENT_GATES_L3_PROMOTION_GATE",
    "AGENT_GATES_L4_CANNBENCH_GATE",
    "AgentGatesDeviceFaultError",
    "AnchorEvalMetrics",
    "CandidateSurrogateResult",
    "DEFAULT_L3_PROMOTION_WAIT_SEC",
    "DEFAULT_L4_CANNBENCH_WAIT_SEC",
    "FAST_REQUEST_ENV",
    "FAST_REQUEST_SCHEMA",
    "FAST_RESPONSE_SCHEMA",
    "FAST_SOURCE_ROOT_ENV",
    "FAST_STATUS_CANDIDATE_FAILURE",
    "FAST_STATUS_INFRASTRUCTURE_ERROR",
    "FAST_STATUS_OK",
    "FAST_STATUS_REQUEST_ERROR",
    "FAILURE_KIND_CANDIDATE",
    "FAILURE_KIND_INFRASTRUCTURE",
    "L0L1CandidateCheck",
    "L0L1RoundResults",
    "L2BatchResults",
    "L3AcceptanceResult",
    "L4CannbenchResult",
    "OccupancyExhaustedError",
    "PrewarmTileLangResult",
    "aggregate_surrogate_result",
    "bottleneck_anchor_for_regime",
    "build_anchor_metrics",
    "collect_bottleneck_msprof_via_agent_gates",
    "collect_p2_deep_dive_via_agent_gates",
    "compute_adaptive_poison_pill_timeout",
    "ensure_npu_compare_shim_dir",
    "extract_ordered_tensor_shapes",
    "lookup_case_static_spec",
    "measure_npu_ring_burst_stats",
    "measure_paired_interleaved_micro_burst",
    "order_candidates_by_idle_preflight",
    "prewarm_anchor_reference_cache",
    "prewarm_surrogate_tilelang_cache",
    "prewarm_surrogate_tilelang_cache_with_stats",
    "record_trackb_stage_b_artifacts_and_plan",
    "run_l0_l1_round_filter",
    "run_l0_static_precheck",
    "run_l1_compile_check",
    "run_l3_full_contract_via_agent_gates",
    "run_l4_official_cannbench_via_agent_gates",
    "run_micro_probe_via_agent_gates",
    "run_multi_operator_surrogate_batch_via_agent_gates",
    "run_poison_pill_subprocess",
    "run_surrogate_batch_via_agent_gates",
    "select_idle_campaign_device",
]
