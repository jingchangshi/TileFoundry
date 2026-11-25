"""Four-tier host-outside-gate evaluation pyramid (L0->L1->L2->L3->L4) for campaigns."""

from __future__ import annotations

import ast
import asyncio
import fcntl
import hashlib
import inspect
import json
import logging
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import uuid
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
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
from tilefoundry_ascend.agent.orchestrator.cpp_slice_protocol import (
    parse_scoped_search_replace_blocks,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    StageAResult,
    StageBResult,
    _bounded_compile_diagnostic,
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
    _ANY_FOLDING_MARKER_RE,
    _compact_trackb_repair_source,
    _detect_trackb_prologue_line_offset,
    _resplice_trackb_omitted_spans,
    build_syntax_repair_request,
    extract_lowered_compiler_excerpt as _extract_lowered_compiler_excerpt,
    extract_lowered_compiler_excerpt_with_lang,
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
    """L2 candidate results that also carry the AgentGates run id of their measuring execution.

    The mapping protocol is unchanged so legacy evaluators can keep returning plain
    dictionaries; ``run_ids`` maps each candidate that was actually measured to the
    AgentGates run that produced its L2 result.
    """

    def __init__(
        self,
        results: Mapping[str, CandidateSurrogateResult],
        run_ids: Mapping[str, str],
    ) -> None:
        super().__init__(results)
        self.run_ids: dict[str, str] = dict(run_ids)


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


def run_l1_compile_check(
    module_source: str,
    *,
    builder_name: str,
    builder_kwargs: Mapping[str, object],
    config_selector_name: str | None = None,
    pass_configs: Mapping[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None,
    dims: Mapping[str, object] | None = None,
    source_root: str | Path | None = None,
    env_file: str | Path | None = None,
    timeout_sec: int = 120,
) -> StageBResult:
    """Run Tier L1 TileLang + bisheng device compilation on Host CPU without NPU locks."""
    env_vals = _resolve_env_values(env_file)
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
        return run_stage_b_compile(
            module_source,
            factory_name=builder_name,
            factory_kwargs=builder_kwargs,
            config_selector_name=config_selector_name,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
            dims=dims,
            python_bin=python_bin,
            timeout_sec=timeout_sec,
        )

    t0 = time.monotonic()
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    pp_parts = [str(root / "src"), str(root / "targets" / "ascend")]
    if tl_source:
        pp_parts.insert(0, tl_source)
    if os.environ.get("PYTHONPATH"):
        pp_parts.append(os.environ["PYTHONPATH"])

    run_env = dict(os.environ)
    run_env["PYTHONPATH"] = os.pathsep.join(pp_parts)
    run_env["TILELANG_DISABLE_CACHE"] = "1"
    run_env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    run_env.setdefault("USE_CUDA", "0")

    with tempfile.TemporaryDirectory(prefix="tf_l1_compile_") as tmp_dir:
        tmp = Path(tmp_dir)
        cand_path = tmp / "candidate_dsl.py"
        cand_path.write_text(module_source, encoding="utf-8")
        kernel_dump_path = tmp / "lowered_kernel.cce"
        driver_path = tmp / "compile_driver.py"
        driver_path.write_text(
            build_stage_b_driver_source(
                cand_path,
                factory_name=builder_name,
                factory_kwargs=builder_kwargs,
                config_selector_name=config_selector_name,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
                dims=dims,
                enable_device_compile=True,
                kernel_dump_path=kernel_dump_path,
            ),
            encoding="utf-8",
        )
        if cann_setup and Path(cann_setup).is_file() and shutil.which("bisheng") is None:
            bash_cmd = (
                f"set +u && source {shlex.quote(cann_setup)} >/dev/null 2>&1 "
                f"&& exec {shlex.quote(python_bin)} {shlex.quote(str(driver_path))}"
            )
            argv = ["/bin/bash", "-c", bash_cmd]
        else:
            argv = [python_bin, str(driver_path)]
        try:
            proc = subprocess.run(
                argv,
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
        return StageBResult(
            passed=False,
            error_message=f"l1_compile_failed (rc={proc.returncode}): {detail}",
            elapsed_sec=time.monotonic() - t0,
            spill_bytes=spill_bytes,
            vloopv2_hit=vloopv2_hit,
            compiler_summary=summary,
            microarch_profile=ma_profile,
            full_diagnostic=raw_diagnostic,
        )
    return StageBResult(
        passed=True,
        elapsed_sec=time.monotonic() - t0,
        spill_bytes=spill_bytes,
        vloopv2_hit=vloopv2_hit,
        compiler_summary=summary,
        microarch_profile=ma_profile,
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
    cost_coarse_reject_ratio: float = 3.0,
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
) -> list[L0L1CandidateCheck]:
    """Filter candidates via noop guard, L0 static, optional cost prescreen, and L1 compile."""
    max_attempts = max(0, max_syntax_repairs)
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
    trackb_backends = (
        "ascendc",
        "ascendc_template",
        "ascendc_trackb",
        "cce_trackb",
        "hybrid_trackb",
    )

    async def _try_repair(
        cur_src: str,
        diag: str,
        *,
        lowered_excerpt: str = "",
        lowered_excerpt_lang: str | None = None,
        extra_pre_compaction_sources: Sequence[str] = (),
    ) -> tuple[str | None, str, bool]:
        assert repair_backend is not None
        is_trackb = backend_kind in trackb_backends
        repair_src = (
            _compact_trackb_repair_source(
                cur_src,
                diag,
                parent_source=parent_source,
                prologue_line_offset=trackb_prologue_line_offset,
            )
            if is_trackb
            else cur_src
        )
        req = build_syntax_repair_request(
            repair_src,
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
        try:
            raw_reply = await repair_backend.generate(req)
        except Exception as exc:
            return None, f"{diag} | repair_transport: {type(exc).__name__}: {exc}", True
        try:
            repaired_src = extract_python_dsl(raw_reply)
        except Exception as exc:
            return None, f"{diag} | repair_parse: {type(exc).__name__}: {exc}", False
        if is_trackb:
            repaired_src = _resplice_trackb_omitted_spans(
                repaired_src,
                pre_compaction_source=cur_src,
                parent_source=parent_source,
                extra_sources=extra_pre_compaction_sources,
            )
            if _ANY_FOLDING_MARKER_RE.search(repaired_src):
                return (
                    None,
                    f"{diag} | trackb_folding_marker_rejected: unresolvable folding marker",
                    False,
                )
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
    ) -> tuple[L0L1CandidateCheck | None, tuple[str, str, StageAResult, bool, int] | None]:
        snippet = cur_snippet
        is_repaired = repaired
        active_patch_fn = initial_patch_fn or patch_fn
        for attempt in range(start_attempt, max_attempts + 1):
            try:
                patched = active_patch_fn(snippet)
            except Exception as exc:
                err = f"ast_patch_error: {type(exc).__name__}: {exc}"
                if repair_backend is None or attempt >= max_attempts:
                    return (
                        L0L1CandidateCheck(
                            candidate_id=cand_id,
                            passed=False,
                            failed_tier="stage_a",
                            patched_module_source="",
                            snippet_source=snippet,
                            repaired=is_repaired,
                            error_message=err,
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
                        ),
                        None,
                    )
            stage_a = await asyncio.to_thread(l0_checker, patched)
            if not stage_a.passed:
                err = stage_a.error_message or "l0_static_failed"
                if repair_backend is None or attempt >= max_attempts:
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
                        ),
                        None,
                    )
                next_snippet, rep_err, is_transport = await _try_repair(
                    snippet,
                    err,
                    extra_pre_compaction_sources=(patched,),
                )
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
    ) -> L0L1CandidateCheck:
        cur_snippet = snippet
        cur_patched = patched
        cur_stage_a = stage_a
        cur_repaired = repaired
        cur_attempt = attempt
        while True:
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
            if repair_backend is None or cur_attempt >= max_attempts:
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
                )
            lowered_excerpt, excerpt_lang = extract_lowered_compiler_excerpt_with_lang(stage_b)
            next_snippet, rep_err, is_transport = await _try_repair(
                cur_snippet,
                err,
                lowered_excerpt=lowered_excerpt,
                lowered_excerpt_lang=excerpt_lang,
                extra_pre_compaction_sources=(cur_patched,),
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

    use_prescreen = (
        stage_a_cost_ranker is not None
        and cost_prescreen_top_k is not None
        and int(cost_prescreen_top_k) > 0
    )
    if not use_prescreen:
        return list(await asyncio.gather(*(_check_one(cid, src) for cid, src in candidates)))

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
        return [results_by_idx[i] for i in range(len(candidates))]

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
        has_parent_ana = (
            parent_analytical_us is not None
            and math.isfinite(float(parent_analytical_us))
            and float(parent_analytical_us) > 0.0
        )
        eff_reject_ratio = max(1.0, float(cost_coarse_reject_ratio))
        for idx, cid, cur_snippet, patched, stage_a, repaired, attempt in reserve_same:
            pred_us = pred_by_id.get(cid, float("inf"))
            raw_ana_ratio = float("nan")
            if has_parent_ana:
                analytical_us = (
                    float(stage_a_analytical_scorer(cid, patched, stage_a))
                    if stage_a_analytical_scorer is not None
                    else pred_us
                )
                if math.isfinite(analytical_us):
                    raw_ana_ratio = analytical_us / float(parent_analytical_us)
                else:
                    raw_ana_ratio = float("inf")
            should_prune = passed_l1 >= top_k and (
                (not has_parent_ana) or raw_ana_ratio > eff_reject_ratio
            )
            if should_prune:
                err_msg = (
                    f"cost_prescreen_pruned:pred_us={pred_us:.2f},"
                    f"analytical_ratio={raw_ana_ratio:.3f}"
                    if math.isfinite(raw_ana_ratio)
                    else f"cost_prescreen_pruned:pred_us={pred_us:.2f}"
                )
                results_by_idx[idx] = L0L1CandidateCheck(
                    candidate_id=cid,
                    passed=False,
                    failed_tier="cost_prescreen",
                    patched_module_source=patched,
                    snippet_source=cur_snippet,
                    repaired=repaired,
                    stage_a=stage_a,
                    stage_b=None,
                    error_message=err_msg,
                )
                continue
            chk = await _run_stage_b_and_repairs(
                cid, cur_snippet, patched, stage_a, repaired, attempt
            )
            results_by_idx[idx] = chk
            if chk.passed:
                passed_l1 += 1

    return [results_by_idx[i] for i in range(len(candidates))]


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
    env_vals = parse_simple_env_file(env_file) if env_file is not None else {}
    python_bin = (
        env_vals.get("TILEFOUNDRY_NPU_PYTHON")
        or os.environ.get("TILEFOUNDRY_NPU_PYTHON")
        or sys.executable
    )

    run_env = dict(os.environ)
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
    return resolved_cache


_PREWARM_TL_PLAN_TEMPLATE = """import importlib.util
import inspect
import json
import logging
import sys

sys.path.insert(0, {ascend_target_dir!r})
from tilefoundry_ascend.agent.campaign.anchor_specs import (
    get_default_regime_specs,
    resolve_regime_for_case,
)
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    _HIR_DIM_ALIASES,
    _filter_callable_kwargs,
    resolve_regime_config,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    lookup_case_static_spec,
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

for idx, (cid, mod_path) in enumerate(cand_paths):
    try:
        mod_name = f"_tf_warm_plan_{{idx}}"
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load module spec from {{mod_path}}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as exc:
        errors.append(f"candidate={{cid}} module_load_failed: {{type(exc).__name__}}: {{exc}}")
        continue

    get_kernel_fn = getattr(mod, "_get_kernel", None)
    combine_fn = getattr(mod, "_get_combine_kernel", None)

    for cid_num, case_dims, regime in case_specs:
        if getattr(regime, "runtime_backend", "") == "ascendc_trackb":
            continue
        try:
            resolved = resolve_regime_config(mod_path, regime, dims=case_dims)
            builder_fn = getattr(mod, resolved.builder_name, None)
            if not callable(builder_fn):
                raise ValueError(f"Builder {{resolved.builder_name!r}} not callable")

            pool = dict(regime.sample_builder_kwargs)
            for h_key, h_val in regime.representative_hir_dims.items():
                mapped = _HIR_DIM_ALIASES.get(str(h_key), str(h_key))
                pool.setdefault(mapped, h_val)
            for d_key, d_val in case_dims.items():
                mapped = _HIR_DIM_ALIASES.get(str(d_key), str(d_key))
                pool[mapped] = d_val
            raw_bk = {{}}
            raw_ck = {{}}
            sel_name = regime.config_selector_fn
            if sel_name and callable(getattr(mod, sel_name, None)):
                sel_fn = getattr(mod, sel_name)
                sel_out = sel_fn(**_filter_callable_kwargs(sel_fn, pool))
                if isinstance(sel_out, dict):
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
            pool.update(resolved.builder_kwargs)
            pc_safe = dict(resolved.pass_configs) if isinstance(resolved.pass_configs, dict) else {{}}
            cf_list = [str(f) for f in resolved.compile_flags]
            pool["builder_fn"] = builder_fn
            pool["pass_configs"] = pc_safe
            pool["compile_flags"] = tuple(cf_list)
            pool["extra_builder_kwargs"] = dict(raw_bk) if raw_bk else None
            pool["extra_combine_kwargs"] = dict(raw_ck) if raw_ck else None

            used_get_kernel = False
            if callable(get_kernel_fn):
                gk_sig = inspect.signature(get_kernel_fn)
                has_req = all(
                    p.name in pool
                    for p in gk_sig.parameters.values()
                    if p.default is inspect.Parameter.empty
                    and p.kind not in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    )
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
            errors.append(
                f"candidate={{cid}} case={{cid_num}} plan_failed: {{type(exc).__name__}}: {{exc}}"
            )

print(json.dumps({{"tasks": tasks, "errors": errors}}), flush=True)
"""


_PREWARM_TL_WORKER_TEMPLATE = """import importlib.util
import json
import sys
import tilelang

task = json.loads({task_json!r})
spec = importlib.util.spec_from_file_location("_tf_warm_exec", task["mod_path"])
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load module spec from {{task['mod_path']}}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mode = task["mode"]
kwargs = dict(task["kwargs"])
if mode == "_get_kernel":
    fn = getattr(mod, "_get_kernel")
    kwargs["builder_fn"] = getattr(mod, task["builder_name"])
    if "compile_flags" in kwargs and isinstance(kwargs["compile_flags"], list):
        kwargs["compile_flags"] = tuple(kwargs["compile_flags"])
    fn(**kwargs)
elif mode == "_get_combine_kernel":
    fn = getattr(mod, "_get_combine_kernel")
    if "compile_flags" in kwargs and isinstance(kwargs["compile_flags"], list):
        kwargs["compile_flags"] = tuple(kwargs["compile_flags"])
    fn(**kwargs)
else:
    builder_fn = getattr(mod, task["builder_name"])
    prog = builder_fn(**kwargs)
    pc_dict = task.get("pass_configs") or {{}}
    cf_list = task.get("compile_flags") or []
    tilelang.compile(
        prog,
        target="ascend",
        pass_configs=pc_dict or None,
        compile_flags=list(cf_list) if cf_list else None,
    )
"""


def _merge_prewarm_cache_tree(src_dir: Path, dst_dir: Path) -> None:
    """Merge an isolated worker's TileLang cache outputs into the shared session cache."""
    if not src_dir.is_dir():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(src_dir.rglob("*")):
        rel = item.relative_to(src_dir)
        target = dst_dir / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if item.name == "compiled_entries.jsonl" and target.is_file():
            existing = target.read_text(encoding="utf-8")
            addition = item.read_text(encoding="utf-8")
            if existing and not existing.endswith("\n"):
                existing += "\n"
            target.write_text(existing + addition, encoding="utf-8")
        elif not target.exists():
            shutil.copy2(item, target)


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
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    repo = resolve_cannbench_repo(cannbench_repo=cannbench_repo, env_file=env_file)
    resolved_ref_cache = Path(cache_dir) if cache_dir is not None else _default_surrogate_cache_dir()
    resolved_ref_cache.mkdir(parents=True, exist_ok=True)
    resolved_tl_cache = (
        Path(tl_cache_dir) if tl_cache_dir is not None else _default_surrogate_tl_cache_dir()
    )
    resolved_tl_cache.mkdir(parents=True, exist_ok=True)
    if not candidates or not anchors:
        return resolved_tl_cache

    env_vals = _resolve_env_values(env_file)
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

    case_ids = [int(a.case_num) for a in anchors]
    cand_paths = [(str(cid), str(Path(p).resolve())) for cid, p in candidates]

    with tempfile.TemporaryDirectory(prefix="tf_tl_prewarm_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        plan_path = tmp_root / "prewarm_tl_plan.py"
        plan_path.write_text(
            _PREWARM_TL_PLAN_TEMPLATE.format(
                ascend_target_dir=str(root / "targets" / "ascend"),
                op_name=operator_name,
                case_ids_json=json.dumps(case_ids),
                cand_paths_json=json.dumps(cand_paths),
                repo=str(repo),
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
        plan_errors = [str(e) for e in plan_payload.get("errors", ())]
        if plan_errors:
            raise RuntimeError(
                f"prewarm_surrogate_tilelang_cache failed: {'; '.join(plan_errors)[:1000]}"
            )

        deduped_tasks: list[dict[str, object]] = []
        seen_keys: set[tuple[str, str, tuple[str, ...]]] = set()
        for task in plan_payload.get("tasks", ()):
            if not isinstance(task, Mapping):
                continue
            builder_name = str(task.get("builder_name", ""))
            canonical_kwargs_json = json.dumps(
                {
                    "mode": task.get("mode", ""),
                    "kwargs": task.get("kwargs", {}),
                    "pass_configs": task.get("pass_configs", {}),
                },
                sort_keys=True,
            )
            compile_flags = tuple(str(f) for f in (task.get("compile_flags") or ()))
            dedup_key = (builder_name, canonical_kwargs_json, compile_flags)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            deduped_tasks.append(dict(task))

        if not deduped_tasks:
            return resolved_tl_cache

        def _compile_one(idx: int, task: dict[str, object]) -> tuple[int, Path]:
            worker_dir = tmp_root / f"worker_{idx}"
            worker_cache = worker_dir / "tl_cache"
            worker_cache.mkdir(parents=True, exist_ok=True)
            worker_script = worker_dir / "compile_task.py"
            worker_script.write_text(
                _PREWARM_TL_WORKER_TEMPLATE.format(task_json=json.dumps(task, sort_keys=True)),
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
            return idx, worker_cache

        max_workers = min(4, len(deduped_tasks))
        compile_errors: list[str] = []
        worker_caches: dict[int, Path] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_compile_one, idx, task)
                for idx, task in enumerate(deduped_tasks)
            ]
            for fut in futures:
                try:
                    idx, worker_cache = fut.result()
                    worker_caches[idx] = worker_cache
                except Exception as exc:
                    compile_errors.append(str(exc))

        if compile_errors:
            raise RuntimeError(
                f"prewarm_surrogate_tilelang_cache failed: {'; '.join(compile_errors)[:1000]}"
            )
        for idx in range(len(deduped_tasks)):
            _merge_prewarm_cache_tree(worker_caches[idx], resolved_tl_cache)

    return resolved_tl_cache


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
        if not k.startswith("TILEFOUNDRY_NPU_") and k not in _STRIPPED_HOST_CPU_ENV_KEYS
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


def _is_transient_blocked_env(
    proc: subprocess.CompletedProcess[str],
    parsed: Mapping[str, object],
) -> bool:
    if proc.returncode == 0:
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

    if prewarm_cache:
        for op_name, cands, op_anchors in batches:
            prewarm_anchor_reference_cache(
                op_name,
                op_anchors,
                cannbench_repo=cann_repo,
                env_file=resolved_env,
                cache_dir=cache_dir,
                source_root=root,
            )
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
        bindings: dict[str, dict[str, str]] = {}
        entries: list[dict[str, str]] = []
        for index, (candidate_id, module_path) in enumerate(cands):
            if candidate_id in bindings:
                raise ValueError(
                    f"duplicate candidate_id {candidate_id!r} for operator {op_name!r}"
                )
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
        request_batches.append(
            {
                "operator_name": wire_op,
                "entrypoint": get_operator_entrypoint(op_name),
                "candidates": entries,
                "anchors": [a.to_dict() for a in op_anchors],
            }
        )

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
    _verify_fast_bindings(raw_resp, staged)
    _verify_fast_coverage(
        raw_resp,
        expected_candidates=sum(len(cands) for _op, cands, _anchors in batches),
        expected_msprof_targets=0,
    )
    raw_ext_hbm = raw_resp.get("_selected_card_ext_hbm_mb", 0)
    selected_card_ext_hbm_mb = (
        max(0, int(raw_ext_hbm))
        if isinstance(raw_ext_hbm, (int, float)) and not isinstance(raw_ext_hbm, bool)
        else 0
    )
    operators_raw = _fast_response_mapping(raw_resp, "operators")
    out_by_op: dict[str, dict[str, CandidateSurrogateResult]] = {}
    failures: list[str] = []
    for op_name, cands, _anchors in batches:
        wire_op = wire_ops[op_name]
        cands_raw = operators_raw.get(wire_op)
        if cands_raw is None and wire_op != op_name:
            cands_raw = operators_raw.get(op_name)
        if not isinstance(cands_raw, Mapping):
            raise RuntimeError(f"fast response has no results for operator {op_name!r}")
        parsed_candidates: dict[str, CandidateSurrogateResult] = {}
        for candidate_id, _module_path in cands:
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
            if not result.all_passed:
                failures.append(f"{op_name}/{candidate_id}")
        out_by_op[op_name] = parsed_candidates
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
    return by_op.get(operator_name, {}), run_id, receipt_path


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
            if attempt_exec.device_rejected or blocked_category == "occupancy":
                response_path.unlink(missing_ok=True)
                if attempt_exec.device_rejected:
                    rejected_cards.add(card)
                reason = "device_rejected" if attempt_exec.device_rejected else "occupancy"
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
        candidates = _call_order_candidates_by_idle_preflight(
            result_device,
            device_pool,
            excluded=rejected_cards,
            env_vals=env_vals,
        )
        if not candidates:
            candidates = (result_device,)
        execution_opt: _V2GateExecution | None = None
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
    "L3AcceptanceResult",
    "L4CannbenchResult",
    "aggregate_surrogate_result",
    "bottleneck_anchor_for_regime",
    "build_anchor_metrics",
    "collect_bottleneck_msprof_via_agent_gates",
    "collect_p2_deep_dive_via_agent_gates",
    "ensure_npu_compare_shim_dir",
    "extract_ordered_tensor_shapes",
    "lookup_case_static_spec",
    "measure_npu_ring_burst_stats",
    "measure_paired_interleaved_micro_burst",
    "order_candidates_by_idle_preflight",
    "prewarm_anchor_reference_cache",
    "prewarm_surrogate_tilelang_cache",
    "run_l0_l1_round_filter",
    "run_l0_static_precheck",
    "run_l1_compile_check",
    "run_l3_full_contract_via_agent_gates",
    "run_l4_official_cannbench_via_agent_gates",
    "run_micro_probe_via_agent_gates",
    "run_multi_operator_surrogate_batch_via_agent_gates",
    "run_surrogate_batch_via_agent_gates",
    "select_idle_campaign_device",
]
