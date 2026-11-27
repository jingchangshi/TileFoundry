"""L2 paired evaluation, 30s poison-pill batching, and prewarm manifest validation."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import logging
import math
import os
import re
import select
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from tilefoundry_ascend.agent.npu.ag_transport import (
    _NON_AGENT_GATES_ENV_KEYS, _REF_CACHE_THREAD_ENV_KEYS, FAST_REQUEST_SCHEMA,
    AgentGatesDeviceFaultError, _agent_gates_workspace_name,
    _fast_cache_roots, _fast_held_seconds, _fast_request_root, _fast_response_mapping,
    _fast_wire_operator_name, _hook, _inspect_v2_device_fault, _resolve_agent_gates_bin,
    _resolve_env_file_path, _resolve_env_values, _sha256_file, _stage_fast_candidate,
    _submit_fast_gate, _verify_fast_bindings, _verify_fast_coverage, _write_fast_request,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    StageAResult, StageBResult, extract_ordered_tensor_shapes, run_stage_a_static,
)
from tilefoundry_ascend.agent.orchestrator.stage_b_gate import (
    ModuleSelectorPlanExtractor, build_post_selector_regime_plan, run_l1_compile_check,
    validate_prewarm_manifest_v1,
)
from tilefoundry_ascend.agent.specs.anchor_specs import (
    AnchorCaseSpec, PerfTargetSpec, compute_case_eq3_score,
    default_source_root, get_default_anchors, get_default_regime_specs, get_operator_entrypoint,
    get_operator_task, lookup_case_dims, parse_simple_env_file, resolve_cannbench_repo,
    resolve_cannbench_tasks_root, resolve_regime_for_case,
)
from tilefoundry_ascend.agent.specs.latency_algebra import compute_adaptive_poison_pill_timeout_sec
from tilefoundry_ascend.ascend.compile_cache.artifact_store import (
    ArtifactStore, resolve_unified_tl_cache_dir,
)
from tilefoundry_ascend.ascend.json_record import JsonRecordMixin, atomic_write_text

logger = logging.getLogger(__name__)

COLD_L2_HW_THRESHOLD_US = 5.0
COLD_L2_PROBE_THRESHOLD_US = 8.0
RING_BURST_MIN_K = 24
RING_BURST_MAX_K = 32
RING_BURST_TARGET_BYTES = 64 * 1024 * 1024
RING_BURST_MAX_VRAM_BYTES = 1024 * 1024 * 1024
_PREWARM_MANIFEST_FILENAME = "prewarm_manifest_v1.json"
_INCUMBENT_CANDIDATE_IDS = frozenset({"incumbent_slot", "__incumbent__", "incumbent_baseline"})
_CPP_LITERAL_MARKERS = ("__global__", "__aicore__", "__ubuf__", "__simd_vf__", "#include", 'extern "C"')


_R6_FIELDS = frozenset({"cand_us", "ref_perf_us", "t_hw_us", "speedup", "roofline_eta", "eq3_score", "burst_batch_cv", "paired_speedup_vs_incumbent", "paired_rel_mad"})


@dataclass(frozen=True)
class AnchorEvalMetrics(JsonRecordMixin):
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
        """Serialize anchor evaluation metrics with 6-decimal float rounding."""
        return {k: (round(v, 6) if k in _R6_FIELDS else v) for k, v in super().to_dict().items()}


@dataclass(frozen=True)
class CandidateSurrogateResult(JsonRecordMixin):
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
        m = [a for a in self.anchors if a.regime_tag == regime_tag]
        return (sum(max(0.0, 1.0 - a.roofline_eta) for a in m) / len(m)) if m else max(0.0, 1.0 - self.avg_roofline_eta)

    def for_regime(self, regime_tag: str) -> CandidateSurrogateResult:
        """Return CandidateSurrogateResult filtered strictly to anchors of regime_tag."""
        m = tuple(a for a in self.anchors if a.regime_tag == regime_tag)
        return aggregate_surrogate_result(self.candidate_id, self.operator_name, m, error_message=self.error_message) if m else CandidateSurrogateResult(self.candidate_id, self.operator_name, False, 0.0, 0.0, 0.0, 0.0, (), self.error_message)

    def meets_perf_target(self, spec: PerfTargetSpec) -> bool:
        """Return True iff accuracy, score, speedup, eta, and latency limits pass."""
        if not self.all_passed or not self.anchors or self.surrogate_score < spec.target_surrogate_score or self.avg_speedup < spec.target_avg_speedup or self.min_case_speedup < spec.target_min_case_speedup or self.avg_roofline_eta < spec.target_roofline_eta:
            return False
        by_case = {a.case_num: a for a in self.anchors}
        return all((anc := by_case.get(int(cn))) is not None and 0.0 < anc.cand_us <= float(mx) for cn, mx in spec.per_anchor_max_us.items())

    def is_roofline_saturated(self, spec: PerfTargetSpec) -> bool:
        """Return True iff every active anchor has reached target_roofline_eta."""
        return bool(self.all_passed and self.anchors and all(a.roofline_eta >= spec.target_roofline_eta for a in self.anchors))

    def mean_latency_us(self) -> float:
        """Return arithmetic mean candidate latency in microseconds across anchors."""
        valid = [a.cand_us for a in self.anchors if a.cand_us > 0.0]
        return sum(valid) / len(valid) if valid else (1000.0 / self.avg_speedup if self.avg_speedup > 0.0 else 1e9)


class L2BatchResults(dict[str, CandidateSurrogateResult]):
    """L2 candidate results carrying AgentGates run ids and prewarm/NPU telemetry."""

    def __init__(
        self, results: Mapping[str, CandidateSurrogateResult], run_ids: Mapping[str, str], *,
        prewarm_stats: Mapping[str, int] | None = None, l2_prewarm_sec: float = 0.0,
        l2_npu_sec: float = 0.0, null_launch_us: float | None = None,
    ) -> None:
        super().__init__(results)
        self.run_ids = dict(run_ids)
        stats = {"total_tasks": 0, "cache_hits": 0, "compiled_tasks": 0, "failed_candidates": 0}
        if prewarm_stats is not None:
            for k in stats:
                if k in prewarm_stats:
                    stats[k] = int(prewarm_stats[k])
        self.prewarm_stats, self.l2_prewarm_sec, self.l2_npu_sec = stats, float(l2_prewarm_sec), float(l2_npu_sec)
        self.null_launch_us = float(null_launch_us) if null_launch_us is not None and not isinstance(null_launch_us, bool) else None


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

    def __init__(self, items: Sequence[L0L1CandidateCheck], *, wall_repair_sec: float = 0.0, cum_repair_worker_sec: float = 0.0) -> None:
        super().__init__(items)
        self.wall_repair_sec, self.cum_repair_worker_sec = float(wall_repair_sec), float(cum_repair_worker_sec)
        self.round_metadata = {"wall_repair_sec": self.wall_repair_sec, "cum_repair_worker_sec": self.cum_repair_worker_sec}


@dataclass(frozen=True)
class PrewarmTileLangResult:
    """Result and cache telemetry from content-addressed TileLang prewarming."""

    tl_cache_dir: Path
    prewarm_stats: dict[str, int]
    failed_by_cid: dict[str, str]
    elapsed_sec: float


def build_anchor_metrics(
    spec: AnchorCaseSpec, *, accuracy_passed: bool, cand_us: float, error_message: str = "",
    ring_burst_k: int = 1, burst_batch_cv: float = 0.0, spike_retries: int = 0,
    card_ext_hbm_mb: int = 0, paired_speedup_vs_incumbent: float = 0.0, paired_rel_mad: float = 0.0,
) -> AnchorEvalMetrics:
    """Compute speedup, Roofline eta, and Eq.3 score from measured latency."""
    valid = accuracy_passed and math.isfinite(cand_us) and cand_us > 0.0
    lat = float(cand_us) if (math.isfinite(cand_us) and cand_us > 0.0) else 0.0
    speedup = (spec.ref_perf_us / lat) if (valid and spec.ref_perf_us > 0.0) else 0.0
    eta = (spec.t_hw_us / lat) if (valid and spec.t_hw_us > 0.0) else 0.0
    eq3 = compute_case_eq3_score(spec.ref_perf_us, lat, spec.t_hw_us) if valid else 0.0
    cv = float(burst_batch_cv) if math.isfinite(float(burst_batch_cv)) and float(burst_batch_cv) >= 0.0 else 0.0
    ps = float(paired_speedup_vs_incumbent) if valid and math.isfinite(float(paired_speedup_vs_incumbent)) and float(paired_speedup_vs_incumbent) >= 0.0 else 0.0
    pm = float(paired_rel_mad) if math.isfinite(float(paired_rel_mad)) and float(paired_rel_mad) >= 0.0 else 0.0
    return AnchorEvalMetrics(
        case_num=spec.case_num, regime_tag=spec.regime_tag, accuracy_passed=bool(accuracy_passed and lat > 0.0),
        cand_us=lat, ref_perf_us=spec.ref_perf_us, t_hw_us=spec.t_hw_us, speedup=speedup, roofline_eta=eta,
        eq3_score=eq3, error_message=error_message, ring_burst_k=max(1, int(ring_burst_k)), burst_batch_cv=cv,
        spike_retries=max(0, int(spike_retries)), card_ext_hbm_mb=max(0, int(card_ext_hbm_mb)),
        paired_speedup_vs_incumbent=ps, paired_rel_mad=pm,
    )


def compute_ring_burst_k(
    per_call_bytes: int, *, t_hw_us: float | None = None, first_probe_us: float | None = None,
    min_k: int = RING_BURST_MIN_K, max_k: int = RING_BURST_MAX_K,
    target_bytes: int = RING_BURST_TARGET_BYTES, max_vram_bytes: int = RING_BURST_MAX_VRAM_BYTES,
) -> int:
    """Return rotating ring-buffer burst factor K for cold-L2 micro-anchor timing."""
    needs_burst = (
        (t_hw_us is not None and math.isfinite(float(t_hw_us)) and 0.0 < float(t_hw_us) < COLD_L2_HW_THRESHOLD_US)
        or (first_probe_us is not None and math.isfinite(float(first_probe_us)) and 0.0 < float(first_probe_us) < COLD_L2_PROBE_THRESHOLD_US)
    )
    if not needs_burst:
        return 1
    smin, smax, cbytes = max(1, int(min_k)), max(max(1, int(min_k)), int(max_k)), max(0, int(per_call_bytes))
    if cbytes <= 0:
        return smin
    vcap = max(1, int(max_vram_bytes) // cbytes)
    if vcap < smin:
        return max(1, vcap)
    desired = min(smax, max(smin, math.ceil(float(target_bytes) / float(cbytes)))) if cbytes < target_bytes else smin
    return max(1, min(desired, vcap))


def _tensor_nbytes(obj: object) -> int:
    if hasattr(obj, "numel") and hasattr(obj, "element_size"):
        return max(0, int(obj.numel()) * int(obj.element_size()))
    return max(0, int(obj.nbytes)) if hasattr(obj, "nbytes") else 0


def estimate_params_io_bytes(params: object) -> int:
    """Estimate total input/output tensor bytes referenced by a call parameter container."""
    total, seen = 0, set()

    def _walk(item: object) -> None:
        nonlocal total
        if isinstance(item, Mapping):
            for v in item.values():
                _walk(v)
        elif isinstance(item, (list, tuple)):
            for v in item:
                _walk(v)
        elif (nb := _tensor_nbytes(item)) > 0 and id(item) not in seen:
            seen.add(id(item))
            total += nb

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


def allocate_rotating_ring_params(params: object, k: int, *, clone_fn: Callable[[object], object] | None = None) -> list[object]:
    """Pre-allocate *k* independent input/output parameter buffer copies for cold-L2 bursts."""
    return [_clone_param_tree(params, clone_fn) for _ in range(max(1, int(k)))]


def _median_and_cv(samples: Sequence[float]) -> tuple[float, float]:
    med = float(statistics.median(samples))
    return med, (float(statistics.pstdev(samples)) / max(med, 1e-9) if len(samples) >= 2 else 0.0)


def _invoke_callable(func: Callable[..., object], params: object) -> None:
    if isinstance(params, Mapping):
        func(**params)
    elif isinstance(params, (list, tuple)):
        func(*params)
    else:
        func(params)


def _resolve_npu_timing_fns(event_factory: Callable[[], object] | None, synchronize_fn: Callable[[], None] | None) -> tuple[Callable[[], object], Callable[[], None]]:
    if event_factory is not None and synchronize_fn is not None:
        return event_factory, synchronize_fn
    import torch  # noqa: PLC0415
    return event_factory or (lambda: torch.npu.Event(enable_timing=True)), synchronize_fn or torch.npu.synchronize


def _time_burst_once(
    func: Callable[..., object], ring_inputs: Sequence[object], round_idx: int, k: int,
    eff_event: Callable[[], object], eff_sync: Callable[[], None],
) -> float:
    s, e = eff_event(), eff_event()
    s.record()
    for step in range(k):
        _invoke_callable(func, ring_inputs[(round_idx + step) % len(ring_inputs)])
    e.record()
    eff_sync()
    return (float(s.elapsed_time(e)) * 1000.0) / float(k)


def measure_npu_ring_burst_stats(
    func: Callable[..., object], params: object, *, warmup: int = 3, repeat: int = 5,
    t_hw_us: float | None = None, first_probe_us: float | None = None,
    event_factory: Callable[[], object] | None = None, synchronize_fn: Callable[[], None] | None = None,
    clone_fn: Callable[[object], object] | None = None, max_vram_bytes: int = RING_BURST_MAX_VRAM_BYTES,
) -> tuple[float, int, float, int]:
    """Measure micro-anchor latency returning (median_us, ring_burst_k, cv, spike_retries)."""
    is_real_npu = event_factory is None and synchronize_fn is None
    eff_event, eff_sync = _resolve_npu_timing_fns(event_factory, synchronize_fn)
    for _ in range(max(1, int(warmup))):
        _invoke_callable(func, params)
    eff_sync()
    eff_probe_us = first_probe_us
    if eff_probe_us is None and (t_hw_us is None or float(t_hw_us) >= COLD_L2_HW_THRESHOLD_US):
        eff_probe_us = _time_burst_once(func, (params,), 0, 1, eff_event, eff_sync)
    k = compute_ring_burst_k(estimate_params_io_bytes(params), t_hw_us=t_hw_us, first_probe_us=eff_probe_us, max_vram_bytes=max_vram_bytes)
    if k <= 1:
        med, cv = _median_and_cv([_time_burst_once(func, (params,), 0, 1, eff_event, eff_sync) for _ in range(max(1, int(repeat)))])
        return med, 1, cv, 0
    try:
        ring_inputs = allocate_rotating_ring_params(params, k, clone_fn=clone_fn)
    except RuntimeError:
        med, cv = _median_and_cv([_time_burst_once(func, (params,), 0, 1, eff_event, eff_sync) for _ in range(max(1, int(repeat)))])
        return med, 1, cv, 0
    for step in range(min(k, max(1, int(warmup)))):
        _invoke_callable(func, ring_inputs[step % k])
    eff_sync()
    samples = [_time_burst_once(func, ring_inputs, 0, k, eff_event, eff_sync) for _ in range(max(1, int(repeat)))]
    med, cv = _median_and_cv(samples)
    if len(samples) < 3 or cv <= 0.12:
        return med, k, cv, 0
    batches, spike_retries = [(samples, med, cv)], 0
    for _ in range(2):
        spike_retries += 1
        if is_real_npu:
            time.sleep(0.02)
        rs = [_time_burst_once(func, ring_inputs, 0, k, eff_event, eff_sync) for _ in range(max(1, int(repeat)))]
        rmed, rcv = _median_and_cv(rs)
        batches.append((rs, rmed, rcv))
        if rcv <= 0.12:
            break
    if clean := [b for b in batches if b[2] <= 0.12]:
        best = min(clean, key=lambda b: (b[1], b[2]))
        return float(best[1]), k, float(best[2]), spike_retries
    pooled = sorted(v for batch, _, _ in batches for v in batch)
    lmed, lcv = _median_and_cv(pooled[: max(3, (len(pooled) + 1) // 2)])
    return lmed, k, lcv, spike_retries


def measure_npu_ring_burst_us(
    func: Callable[..., object], params: object, *, warmup: int = 3, repeat: int = 5,
    t_hw_us: float | None = None, first_probe_us: float | None = None,
    event_factory: Callable[[], object] | None = None, synchronize_fn: Callable[[], None] | None = None,
    clone_fn: Callable[[object], object] | None = None, max_vram_bytes: int = RING_BURST_MAX_VRAM_BYTES,
) -> tuple[float, int]:
    """Measure micro-anchor latency with cold-L2 rotating ring-buffer burst enqueueing."""
    med, k, _, _ = measure_npu_ring_burst_stats(
        func, params, warmup=warmup, repeat=repeat, t_hw_us=t_hw_us, first_probe_us=first_probe_us,
        event_factory=event_factory, synchronize_fn=synchronize_fn, clone_fn=clone_fn, max_vram_bytes=max_vram_bytes,
    )
    return med, k


def measure_paired_interleaved_micro_burst(
    candidates: Sequence[tuple[str, Callable[..., object]]], params: object, *, warmup: int = 3, repeat: int = 7,
    t_hw_us: float | None = None, first_probe_us: float | None = None,
    event_factory: Callable[[], object] | None = None, synchronize_fn: Callable[[], None] | None = None,
    clone_fn: Callable[[object], object] | None = None, max_vram_bytes: int = RING_BURST_MAX_VRAM_BYTES,
) -> dict[str, tuple[float, int, float, float, float]]:
    """Run B-round interleaved micro-burst timing across candidates on shared buffers."""
    if not candidates:
        return {}
    eff_event, eff_sync = _resolve_npu_timing_fns(event_factory, synchronize_fn)
    is_real = event_factory is None and synchronize_fn is None
    warm_iters = max(6 if is_real else 1, int(warmup))
    for _cid, fn in candidates:
        for _ in range(warm_iters):
            _invoke_callable(fn, params)
    eff_sync()
    eff_probe_us = first_probe_us
    if eff_probe_us is None and (t_hw_us is None or not math.isfinite(float(t_hw_us)) or float(t_hw_us) >= COLD_L2_HW_THRESHOLD_US):
        eff_probe_us = _time_burst_once(candidates[0][1], (params,), 0, 1, eff_event, eff_sync)
    k = compute_ring_burst_k(estimate_params_io_bytes(params), t_hw_us=t_hw_us, first_probe_us=eff_probe_us, max_vram_bytes=max_vram_bytes)
    ring_inputs: list[object] = [params]
    if k > 1:
        try:
            ring_inputs = allocate_rotating_ring_params(params, k, clone_fn=clone_fn)
            for step in range(min(k, warm_iters)):
                _invoke_callable(candidates[0][1], ring_inputs[step % k])
            eff_sync()
        except RuntimeError:
            k, ring_inputs = 1, [params]
    timing_reps = max(1, min(8, int(200.0 // eff_probe_us))) if (is_real and k == 1 and eff_probe_us and math.isfinite(eff_probe_us) and 0.0 < eff_probe_us < 200.0) else k
    num_rounds, num_cands = max(7, int(repeat)), len(candidates)
    samples_by_idx: list[list[float]] = [[] for _ in range(num_cands)]
    for r_idx in range(num_rounds):
        for off in range(num_cands):
            idx = (off + r_idx) % num_cands
            samples_by_idx[idx].append(_time_burst_once(candidates[idx][1], ring_inputs, r_idx, timing_reps, eff_event, eff_sync))
    ref_samples = samples_by_idx[0]
    out: dict[str, tuple[float, int, float, float, float]] = {}
    for idx, (cid, _fn) in enumerate(candidates):
        cs = samples_by_idx[idx]
        cand_us, raw_cv = _median_and_cv(cs)
        if idx == 0:
            med_r, rel_mad = 1.0, 0.0
        else:
            ratios = [ref_samples[b] / max(cs[b], 1e-9) for b in range(num_rounds)]
            med_r = float(statistics.median(ratios))
            rel_mad = float(statistics.median([abs(x - med_r) for x in ratios])) / max(med_r, 1e-9)
        out[cid] = (cand_us, k, raw_cv, med_r, rel_mad)
    return out


def aggregate_surrogate_result(
    candidate_id: str, operator_name: str, anchor_metrics: Sequence[AnchorEvalMetrics], *, error_message: str = ""
) -> CandidateSurrogateResult:
    """Aggregate per-anchor metrics into speedup, eta, and surrogate CANNBench score."""
    metrics = tuple(anchor_metrics)
    if not metrics:
        return CandidateSurrogateResult(candidate_id, operator_name, False, 0.0, 0.0, 0.0, 0.0, (), error_message or "no anchor metrics evaluated")
    all_ok = all(m.accuracy_passed and m.cand_us > 0.0 for m in metrics)
    avg_sp, min_sp = sum(m.speedup for m in metrics) / len(metrics), min(m.speedup for m in metrics)
    avg_eta = sum(m.roofline_eta for m in metrics) / len(metrics)
    case_terms = [(0.20 + 0.70 * m.eq3_score) if m.accuracy_passed else 0.0 for m in metrics]
    comp = 0.05 if any(m.cand_us > 0.0 for m in metrics) else 0.0
    bonus = 0.05 if all_ok else 0.0
    return CandidateSurrogateResult(
        candidate_id, operator_name, all_ok, avg_sp, min_sp, avg_eta,
        (comp + bonus + (sum(case_terms) / len(metrics))) * 100.0, metrics, error_message,
    )


_NPU_COMPARE_SHIM_SOURCE = '''"""Runtime bridge adapting kernel_eval.utils.compare to npu_compare API."""

from __future__ import annotations

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
    import torch
    if not isinstance(actual, torch.Tensor) or not isinstance(golden, torch.Tensor) or actual.shape != golden.shape or actual.numel() == 0 or not actual.is_floating_point():
        return ""
    with torch.no_grad():
        act, gold = actual.detach().cpu(), golden.detach().cpu()
        nan_a, nan_g = torch.isnan(act), torch.isnan(gold)
        if bool((nan_diff := (nan_a != nan_g)).any().item()):
            return f"NaN position mismatch (actual_nan={{int(nan_a.sum().item())}}/{{act.numel()}}, golden_nan={{int(nan_g.sum().item())}}/{{gold.numel()}}, first_coord={{torch.nonzero(nan_diff, as_tuple=False)[0].tolist()}})"
        inf_a, inf_g = torch.isinf(act), torch.isinf(gold)
        both_inf, inf_diff = inf_a & inf_g, inf_a != inf_g
        if bool(both_inf.any().item()):
            sd = both_inf.clone()
            sd[both_inf] = torch.sign(act[both_inf].float()) != torch.sign(gold[both_inf].float())
            inf_diff = inf_diff | sd
        if bool(inf_diff.any().item()):
            return f"Inf mismatch (actual_inf={{int(inf_a.sum().item())}}/{{act.numel()}}, golden_inf={{int(inf_g.sum().item())}}/{{gold.numel()}}, first_coord={{torch.nonzero(inf_diff, as_tuple=False)[0].tolist()}})"
    return ""


def _format_worst_element_diagnostic(actual: object, golden: object) -> str:
    import torch
    if not isinstance(actual, torch.Tensor) or not isinstance(golden, torch.Tensor) or actual.shape != golden.shape or actual.numel() == 0 or actual.is_complex() or golden.is_complex():
        return ""
    with torch.no_grad():
        act, gold = actual.detach().cpu().to(dtype=torch.float64), golden.detach().cpu().to(dtype=torch.float64)
        diff = (act - gold).abs()
        fm = torch.isfinite(diff) & torch.isfinite(act) & torch.isfinite(gold)
        if not bool(fm.any().item()):
            return ""
        fidx = int(torch.argmax(torch.where(fm, diff, torch.full_like(diff, -1.0)).reshape(-1)).item())
        rem, crev = fidx, []
        for dim in reversed(tuple(int(d) for d in act.shape)):
            crev.append(rem % dim); rem //= dim
        rv, gv, ad = float(gold.reshape(-1)[fidx].item()), float(act.reshape(-1)[fidx].item()), float(diff.reshape(-1)[fidx].item())
        return f"worst_element=(idx={{list(reversed(crev))}}, ref={{rv:.6g}}, got={{gv:.6g}}, abs_diff={{ad:.6e}}, rel_err={{ad / max(abs(rv), 1e-12):.6e}})"


def compare_case_result(*, case_id: str, actual: object, golden: object, inputs: object = (), compute: bool = True, accuracy_mode: str | None = None, native_ref_operator=None, extra: dict[str, object] | None = None):
    _ = (case_id, inputs, compute, accuracy_mode)
    ct, dt = (extra or {{}}).get("precision_thresholds") or {{}}, str(getattr(actual, "dtype", "float32")).removeprefix("torch.")
    res = _COMPARE_TENSORS(actual, golden, dtype=dt, native_output=None, custom_thresholds=ct)
    if not res.passed and native_ref_operator is not None:
        res = _COMPARE_TENSORS(actual, golden, dtype=dt, native_output=native_ref_operator(None), custom_thresholds=ct)
    msg = res.error_msg or f"MERE={{res.mere:.6e}}, MARE={{res.mare:.6e}}, max_diff={{res.max_diff:.6e}}"
    if not bool(res.passed):
        if parts := [p for p in (_format_nonfinite_diagnostic(actual, golden), _format_worst_element_diagnostic(actual, golden)) if p and p not in msg]:
            diag = "; ".join(parts)
            msg = f"{{diag}}; raw={{msg}}" if msg else diag
    return SimpleNamespace(passed=bool(res.passed), message=msg)
'''


def ensure_npu_compare_shim_dir(target_dir: str | Path, cannbench_repo: str | Path | None = None, env_file: str | Path | None = None) -> Path:
    """Create a runtime npu_compare.py shim delegating to CANNBench compare_tensors."""
    repo = _hook("resolve_cannbench_repo", resolve_cannbench_repo)(cannbench_repo=cannbench_repo, env_file=env_file)
    out_dir = Path(target_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "npu_compare.py").write_text(_NPU_COMPARE_SHIM_SOURCE.format(cann_repo=str(repo)), encoding="utf-8")
    return out_dir


def _default_surrogate_cache_dir() -> Path:
    return Path(tempfile.gettempdir()) / f"tf_surrogate_ref_cache_{os.getuid() if hasattr(os, 'getuid') else 0}"


def _default_surrogate_tl_cache_dir() -> Path:
    return Path(tempfile.gettempdir()) / f"tf_surrogate_tl_cache_{os.getuid() if hasattr(os, 'getuid') else 0}"


def _load_json_dict(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return dict(raw) if isinstance(raw, Mapping) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        return None


def _load_json_list_or_empty(path: Path | None) -> list[object]:
    if path is None or not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return list(raw) if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else []
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []


def _ref_prewarm_marker_matches(
    marker: Mapping[str, object] | None, *, op_dir: Path, cases_yaml_sha256: str, sorted_case_ids: Sequence[int]
) -> bool:
    if marker is None or not (op_dir / "identity.json").is_file() or marker.get("cases_yaml_sha256") != cases_yaml_sha256:
        return False
    if not isinstance(raw_files := marker.get("files"), list) or not raw_files:
        return False
    try:
        if not set(sorted_case_ids).issubset({int(x) for x in (marker.get("case_ids") or ())}):
            return False
        return all(isinstance(f, str) and f and (op_dir / f).is_file() and (op_dir / f).stat().st_size > 0 for f in raw_files)
    except (OSError, ValueError, TypeError):
        return False


def prewarm_anchor_reference_cache(
    operator_name: str, anchors: Sequence[AnchorCaseSpec], *, cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None, cache_dir: str | Path | None = None, source_root: str | Path | None = None,
) -> Path:
    """Populate CANNBench oracle and native reference cache on Host CPU without NPU locks."""
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    repo = _hook("resolve_cannbench_repo", resolve_cannbench_repo)(cannbench_repo=cannbench_repo, env_file=env_file)
    resolved_cache = Path(cache_dir) if cache_dir is not None else _hook("_default_surrogate_cache_dir", _default_surrogate_cache_dir)()
    resolved_cache.mkdir(parents=True, exist_ok=True)
    try:
        task = _hook("get_operator_task", get_operator_task)(operator_name)
    except (KeyError, ValueError):
        task = ""
    sorted_ids = sorted({int(a.case_num) for a in anchors})
    op_dir = resolved_cache / "operators" / task.replace("/", "__") if task else None
    marker_p = op_dir / ".ref_prewarm_v1.json" if op_dir is not None else None
    cy_sha = _resolve_cases_yaml_sha256(repo, operator_name) if task else ""
    if task and sorted_ids and op_dir and marker_p and _ref_prewarm_marker_matches(_load_json_dict(marker_p), op_dir=op_dir, cases_yaml_sha256=cy_sha, sorted_case_ids=sorted_ids):
        return resolved_cache
    env_vals = parse_simple_env_file(env_file) if env_file is not None else {}
    py_bin = env_vals.get("TILEFOUNDRY_NPU_PYTHON") or os.environ.get("TILEFOUNDRY_NPU_PYTHON") or sys.executable
    run_env = {k: v for k, v in os.environ.items() if k not in _REF_CACHE_THREAD_ENV_KEYS}
    pp = [str(root), str(root / "src"), str(root / "targets" / "ascend"), *([run_env["PYTHONPATH"]] if run_env.get("PYTHONPATH") else [])]
    run_env.update({"TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "CANNBENCH_CACHE_DIR": str(resolved_cache), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": os.pathsep.join(pp)})
    script = (
        f"import json, sys\nsys.path.insert(0, {str(root / 'targets' / 'ascend')!r})\n"
        "from tilefoundry_ascend.agent.specs.anchor_specs import AnchorCaseSpec, load_anchor_harness_and_cases, load_cannbench_contract_module\n"
        f"op_name, case_ids, repo, root = {operator_name!r}, {json.dumps([int(a.case_num) for a in anchors])}, {str(repo)!r}, {str(root)!r}\n"
        "contract = load_cannbench_contract_module(root)\n"
        "harness, cases = load_anchor_harness_and_cases(op_name, [AnchorCaseSpec('', c, '', 1.0, 1.0) for c in case_ids], cannbench_repo=repo, source_root=root)\n"
        "with contract.ReferenceCache(harness) as cache:\n"
        "    for case in cases:\n"
        "        def _prep(c=case, h=harness):\n            ts = h.prepare(c); return {'inputs': ts, 'oracle': h.reference_output(c, ts)}\n"
        "        b = cache.get(case, 'oracle', _prep)\n"
        "        cache.get(case, 'native', lambda c=case, b=b, h=harness: h.reference_output(c, b['inputs'], native=True))\n"
    )
    proc = subprocess.run([py_bin, "-c", script], text=True, capture_output=True, cwd=str(root), env=run_env, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(f"prewarm_anchor_reference_cache failed: {(proc.stderr or proc.stdout or '').strip()[-1000:]}")
    if task and sorted_ids and op_dir and marker_p and (op_dir / "identity.json").is_file():
        if pt_files := sorted({p.name for pat in ("*.oracle.pt", "*.native.pt") for p in op_dir.glob(pat) if p.is_file() and p.stat().st_size > 0}):
            mids = set(sorted_ids)
            if (ex := _load_json_dict(marker_p)) and ex.get("cases_yaml_sha256") == cy_sha:
                try:
                    mids.update(int(x) for x in (ex.get("case_ids") or ()))
                except (ValueError, TypeError):
                    pass
            atomic_write_text(marker_p, json.dumps({"version": 1, "task": task, "cases_yaml_sha256": cy_sha, "case_ids": sorted(mids), "files": pt_files}, indent=2, sort_keys=True) + "\n")
    return resolved_cache


def _resolve_cases_yaml_sha256(repo: Path, operator_name: str) -> str:
    try:
        cases_path = resolve_cannbench_tasks_root(repo) / _hook("get_operator_task", get_operator_task)(operator_name) / "cases.yaml"
        return _sha256_file(cases_path) if cases_path.is_file() else ""
    except (OSError, ValueError, KeyError, RuntimeError):
        return ""


def _resolve_l1_cache_store(
    env_vals: Mapping[str, str], tl_cache_dir: str | Path | None, artifact_store: ArtifactStore | None, source_root: Path | None
) -> tuple[Path, ArtifactStore]:
    if artifact_store is not None:
        return artifact_store.root, artifact_store
    if tl_cache_dir is not None and str(tl_cache_dir).strip():
        resolved = Path(tl_cache_dir).expanduser().resolve()
        return resolved, ArtifactStore(resolved)
    has_env = any(str(env_vals.get(k, "") or os.environ.get(k, "")).strip() for k in ("TILELANG_CACHE_DIR", "TF_SURROGATE_TL_CACHE_DIR", "TILEFOUNDRY_NPU_ARTIFACT_STORE", "TILEFOUNDRY_NPU_WORK_ROOT"))
    resolved = resolve_unified_tl_cache_dir(None, env_vals=env_vals, source_root=source_root) if has_env else _hook("_default_surrogate_tl_cache_dir", _default_surrogate_tl_cache_dir)().resolve()
    return resolved, ArtifactStore(resolved)


def lookup_case_static_spec(
    operator_name: str, case_num: int, *, cannbench_repo: str | Path | None = None, env_file: str | Path | None = None
) -> dict[str, object]:
    """Read static dims, tl_dtype, and attributes directly from cases.yaml."""
    repo = _hook("resolve_cannbench_repo", resolve_cannbench_repo)(cannbench_repo=cannbench_repo, env_file=env_file)
    task = _hook("get_operator_task", get_operator_task)(operator_name)
    cases_path = resolve_cannbench_tasks_root(repo) / task / "cases.yaml"
    if not cases_path.is_file():
        raise FileNotFoundError(f"cases.yaml not found at {cases_path}")
    parsed = yaml.safe_load(cases_path.read_text(encoding="utf-8")) or {}
    target_id = int(case_num)
    entry = next((r for r in (parsed.get("cases") or ()) if isinstance(r, Mapping) and int(r.get("case_id", -1)) == target_id), None)
    if entry is None:
        raise KeyError(f"case_num={target_id} missing in {cases_path}")
    spec = lookup_case_dims(operator_name, target_id, cannbench_repo=repo, env_file=env_file)
    raw_dt = entry.get("dtype", "bfloat16")
    dt = str(raw_dt[0] if isinstance(raw_dt, (list, tuple)) and raw_dt else raw_dt).strip() or "bfloat16"
    spec["tl_dtype"] = dt.removeprefix("torch.")
    if "inputLayout" in spec and "input_layout" not in spec:
        spec["input_layout"] = str(spec["inputLayout"]).upper()
    if "is_causal" in spec:
        spec["is_causal"] = bool(spec["is_causal"])
    if "B" in spec and "S" in spec and "m_rows" not in spec and isinstance(spec["B"], int) and isinstance(spec["S"], int):
        spec["m_rows"] = int(spec["B"]) * int(spec["S"])
    return spec


def prewarm_surrogate_tilelang_cache_with_stats(
    operator_name: str, candidates: Sequence[tuple[str, str | Path]], anchors: Sequence[AnchorCaseSpec], *,
    cannbench_repo: str | Path | None = None, env_file: str | Path | None = None,
    cache_dir: str | Path | None = None, tl_cache_dir: str | Path | None = None,
    source_root: str | Path | None = None, timeout_sec: int = 300, isolate_non_incumbent: bool = True,
) -> PrewarmTileLangResult:
    """Validate prewarm_manifest_v1.json + .so artifacts without secondary compilation."""
    _ = (cache_dir, timeout_sec)
    t0 = time.monotonic()
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    repo = _hook("resolve_cannbench_repo", resolve_cannbench_repo)(cannbench_repo=cannbench_repo, env_file=env_file)
    env_vals = _resolve_env_values(env_file)
    tl_c, _ = _resolve_l1_cache_store(env_vals, tl_cache_dir, None, root)
    tl_c.mkdir(parents=True, exist_ok=True)
    if not candidates or not anchors:
        return PrewarmTileLangResult(tl_c, {"total_tasks": 0, "cache_hits": 0, "compiled_tasks": 0, "failed_candidates": 0}, {}, time.monotonic() - t0)
    manifest_path = tl_c / _PREWARM_MANIFEST_FILENAME
    try:
        regimes = get_default_regime_specs(operator_name)
    except (KeyError, FileNotFoundError, ValueError):
        regimes = ()
    cases_by_id: dict[str, dict[str, object]] = {}
    for a in anchors:
        cn = int(a.case_num)
        try:
            dims = dict(lookup_case_static_spec(operator_name, cn, cannbench_repo=repo, env_file=env_file))
        except (FileNotFoundError, KeyError, OSError, ValueError):
            dims = {}
        reg = next((r for r in regimes if r.regime_tag == a.regime_tag), None)
        if reg is None and regimes and dims:
            try:
                reg = resolve_regime_for_case(operator_name, dims, regimes)
            except (KeyError, TypeError, ValueError):
                reg = None
        if reg is not None:
            for k, v in reg.sample_builder_kwargs.items():
                dims.setdefault(str(k), v)
            if reg.config_selector_fn:
                dims.setdefault("config_selector_name", reg.config_selector_fn)
            if reg.builder_candidates:
                dims.setdefault("builder_name", reg.builder_candidates[0])
        cases_by_id[str(cn)] = dims
    extractor = ModuleSelectorPlanExtractor(target="dav-3510")
    total_tasks, cache_hits, failed_by_cid = 0, 0, {}
    for rcid, rp in candidates:
        cid = str(rcid)
        try:
            mod_src = Path(rp).resolve().read_text(encoding="utf-8")
        except OSError as exc:
            err = f"candidate={cid} module_read_failed: {type(exc).__name__}: {exc}"
            if cid == "incumbent_slot" or not isolate_non_incumbent:
                raise RuntimeError(f"prewarm_surrogate_tilelang_cache failed: {err}") from exc
            failed_by_cid[cid] = err
            continue
        plan = build_post_selector_regime_plan(mod_src, str(anchors[0].regime_tag or operator_name or "default"), cases_by_id, extractor)
        ok, details = validate_prewarm_manifest_v1(manifest_path, plan)
        total_tasks += int(details.get("total_tasks", 0))
        cache_hits += int(details.get("cache_hits", 0))
        if not ok:
            err = f"prewarm_manifest_v1_miss: missing={len(details.get('missing_tasks', ()))} failed={len(details.get('failed_tasks', ()))}"
            if cid == "incumbent_slot" or not isolate_non_incumbent:
                raise RuntimeError(f"prewarm_surrogate_tilelang_cache failed: {err}")
            failed_by_cid[cid] = err
    stats = {"total_tasks": total_tasks, "cache_hits": cache_hits, "compiled_tasks": 0, "failed_candidates": len(failed_by_cid)}
    return PrewarmTileLangResult(tl_c, stats, failed_by_cid, time.monotonic() - t0)


def prewarm_surrogate_tilelang_cache(
    operator_name: str, candidates: Sequence[tuple[str, str | Path]], anchors: Sequence[AnchorCaseSpec], *,
    cannbench_repo: str | Path | None = None, env_file: str | Path | None = None,
    cache_dir: str | Path | None = None, tl_cache_dir: str | Path | None = None,
    source_root: str | Path | None = None, timeout_sec: int = 300,
) -> Path:
    """Pre-compile deduplicated candidate anchor kernels into TileLang disk cache on Host CPU."""
    return _hook("prewarm_surrogate_tilelang_cache_with_stats", prewarm_surrogate_tilelang_cache_with_stats)(
        operator_name, candidates, anchors, cannbench_repo=cannbench_repo, env_file=env_file,
        cache_dir=cache_dir, tl_cache_dir=tl_cache_dir, source_root=source_root, timeout_sec=timeout_sec, isolate_non_incumbent=False,
    ).tl_cache_dir


_ORIG_PREWARM_SURROGATE_TILELANG_CACHE = prewarm_surrogate_tilelang_cache


def compute_adaptive_poison_pill_timeout(inc_latencies_us: Sequence[float]) -> float:
    """Compute bounded poison-pill kernel timeout in [10.0, 60.0] seconds."""
    return compute_adaptive_poison_pill_timeout_sec(inc_latencies_us)


def run_poison_pill_subprocess(
    cmd: Sequence[str], *, cwd: str | Path, env: Mapping[str, str] | None = None,
    inc_latencies_us: Sequence[float] = (), exec_timeout_sec: float | None = None, boot_timeout_sec: float = 30.0,
) -> tuple[bool, str, int, str, str]:
    """Run poison-pill worker with boot-ready pipe handshake before exec timeout starts."""
    eff_exec = max(0.05, float(exec_timeout_sec)) if exec_timeout_sec is not None else compute_adaptive_poison_pill_timeout(inc_latencies_us)
    eff_boot = max(1.0, float(boot_timeout_sec))
    rfd, wfd = os.pipe()
    child_env = {**dict(os.environ if env is None else env), "TF_POISON_PILL_READY_FD": str(wfd)}

    def _kill(p: subprocess.Popen[str]) -> None:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            p.kill()

    def _fail_msg(rc: int | None, so: str, se: str, prefix: str = "") -> tuple[bool, str, int, str, str]:
        det = (se or so or "").strip()[-300:]
        msg = prefix or (f"poison_pill_timeout_or_crash: exit_code={rc}" + (f" ({det})" if det else ""))
        return False, msg[:300], int(rc if rc is not None else 1), so or "", se or ""

    proc = subprocess.Popen([str(p) for p in cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(cwd), env=child_env, start_new_session=True, pass_fds=(wfd,))
    os.close(wfd)
    boot_ready, deadline, buf = False, time.monotonic() + eff_boot, b""
    try:
        while (rem := deadline - time.monotonic()) > 0.0:
            if select.select([rfd], [], [], rem)[0]:
                if not (chunk := os.read(rfd, 256)):
                    break
                buf += chunk
                if b"READY" in buf:
                    boot_ready = True; break
        else:
            _kill(proc); so, se = proc.communicate()
            return _fail_msg(proc.returncode if proc.returncode is not None else -9, so, se, f"poison_pill_timeout_or_crash: boot timeout after {eff_boot:g}s")
    finally:
        os.close(rfd)
    if not boot_ready and proc.poll() is not None:
        so, se = proc.communicate()
        return _fail_msg(proc.returncode or 1, so, se)
    try:
        so, se = proc.communicate(timeout=eff_exec)
    except subprocess.TimeoutExpired:
        _kill(proc); so, se = proc.communicate()
        return _fail_msg(-9, so, se, f"poison_pill_timeout_or_crash: timeout after {eff_exec:g}s")
    return (True, "", 0, so or "", se or "") if proc.returncode == 0 else _fail_msg(proc.returncode, so, se)


def _failed_prewarm_candidate_result(candidate_id: str, operator_name: str, anchors: Sequence[AnchorCaseSpec], error_detail: str) -> CandidateSurrogateResult:
    msg = f"prewarm_compile_failed: {error_detail}"
    return CandidateSurrogateResult(
        candidate_id, operator_name, False, 0.0, 0.0, 0.0, 0.0,
        tuple(build_anchor_metrics(a, accuracy_passed=False, cand_us=0.0, error_message=msg[:300]) for a in anchors), msg[:500],
    )


def run_multi_operator_surrogate_batch_via_agent_gates(
    batches: Sequence[tuple[str, Sequence[tuple[str, str | Path]], Sequence[AnchorCaseSpec]]], *,
    source_root: str | Path | None = None, env_file: str | Path | None = None,
    device_id: str = "1", device_pool: Sequence[str] = (), warmup: int = 3, repeat: int = 5,
    require_all_passed: bool = False, timeout_sec: int = 300, prewarm_cache: bool = True,
) -> tuple[dict[str, dict[str, CandidateSurrogateResult]], str, str, float]:
    """Evaluate candidate batches across operators inside one AgentGates device lease."""
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    resolved_env = _hook("_resolve_env_file_path", _resolve_env_file_path)(env_file)
    ag_bin = _hook("_resolve_agent_gates_bin", _resolve_agent_gates_bin)(root)
    env_vals = parse_simple_env_file(resolved_env)
    cann_repo = _hook("resolve_cannbench_repo", resolve_cannbench_repo)(cannbench_repo=env_vals.get("TILEFOUNDRY_NPU_CANNBENCH_REPO"), env_file=resolved_env)
    cache_dir, tl_cache_dir = _fast_cache_roots(root, env_vals)
    pw_stats = {"total_tasks": 0, "cache_hits": 0, "compiled_tasks": 0, "failed_candidates": 0}
    failed_by_op: dict[str, dict[str, str]] = {}
    t_pw0 = time.monotonic()
    if prewarm_cache:
        pw_tl, pw_s, pw_ref = _hook("prewarm_surrogate_tilelang_cache", prewarm_surrogate_tilelang_cache), _hook("prewarm_surrogate_tilelang_cache_with_stats", prewarm_surrogate_tilelang_cache_with_stats), _hook("prewarm_anchor_reference_cache", prewarm_anchor_reference_cache)
        use_stats = pw_tl is _ORIG_PREWARM_SURROGATE_TILELANG_CACHE
        for op_name, cands, op_anchors in batches:
            pw_ref(op_name, op_anchors, cannbench_repo=cann_repo, env_file=resolved_env, cache_dir=cache_dir, source_root=root)
            if use_stats:
                res = pw_s(op_name, cands, op_anchors, cannbench_repo=cann_repo, env_file=resolved_env, cache_dir=cache_dir, tl_cache_dir=tl_cache_dir, source_root=root, isolate_non_incumbent=True)
                for k in pw_stats:
                    pw_stats[k] += int(res.prewarm_stats.get(k, 0))
                if res.failed_by_cid:
                    failed_by_op[op_name] = dict(res.failed_by_cid)
            else:
                pw_tl(op_name, cands, op_anchors, cannbench_repo=cann_repo, env_file=resolved_env, cache_dir=cache_dir, tl_cache_dir=tl_cache_dir, source_root=root)
    l2_pw_sec = time.monotonic() - t_pw0

    req_root, req_id = _fast_request_root(root, env_vals), uuid.uuid4().hex
    req_dir, req_path, resp_path = req_root / req_id, req_root / f"{req_id}.request.json", req_root / f"{req_id}.response.json"
    staged, wire_ops, req_batches, seen_ops, seen_wire, failed_res_by_op, valid_by_op, failures = {}, {}, [], set(), set(), {}, {}, []
    for op_name, cands, op_anchors in batches:
        if op_name in seen_ops:
            raise ValueError(f"duplicate operator_name {op_name!r} in one fast batch")
        seen_ops.add(op_name)
        wire_op = _fast_wire_operator_name(op_name)
        if wire_op in seen_wire:
            raise ValueError(f"duplicate wire operator_name {wire_op!r} (from {op_name!r}) in one fast batch")
        seen_wire.add(wire_op); wire_ops[op_name] = wire_op
        op_fmap = failed_by_op.get(op_name, {})
        if "incumbent_slot" in op_fmap:
            raise RuntimeError(f"prewarm_surrogate_tilelang_cache failed for incumbent_slot: {op_fmap['incumbent_slot']}")
        op_fres, valid_c, seen_cids = {}, [], set()
        for cid, mpath in cands:
            if cid in seen_cids:
                raise ValueError(f"duplicate candidate_id {cid!r} for operator {op_name!r}")
            seen_cids.add(cid)
            if cid in op_fmap:
                op_fres[cid] = _failed_prewarm_candidate_result(cid, op_name, op_anchors, op_fmap[cid]); failures.append(f"{op_name}/{cid}")
            else:
                valid_c.append((cid, mpath))
        failed_res_by_op[op_name] = op_fres
        if (any(c not in _INCUMBENT_CANDIDATE_IDS for c, _ in cands) and not any(c not in _INCUMBENT_CANDIDATE_IDS for c, _ in valid_c)) or not valid_c:
            valid_by_op[op_name] = []; continue
        valid_by_op[op_name] = valid_c
        bindings, entries = {}, []
        for idx, (cid, mpath) in enumerate(valid_c):
            rel, dig = _stage_fast_candidate(req_dir, root, wire_op, idx, cid, Path(mpath))
            bindings[cid] = {"module_path": rel, "sha256": dig}
            entries.append({"candidate_id": cid, "module_path": rel, "sha256": dig})
        staged[wire_op] = bindings
        req_batches.append({"operator_name": wire_op, "entrypoint": get_operator_entrypoint(op_name), "candidates": entries, "anchors": [a.to_dict() for a in op_anchors], "poison_pill_timeout_sec": compute_adaptive_poison_pill_timeout([float(a.ref_perf_us) for a in op_anchors])})

    if not req_batches:
        out_empty = {op: L2BatchResults({c: failed_res_by_op[op][c] for c, _ in cs if c in failed_res_by_op[op]}, {}, prewarm_stats=pw_stats, l2_prewarm_sec=l2_pw_sec, l2_npu_sec=0.0) for op, cs, _ in batches}
        if require_all_passed and failures:
            raise RuntimeError("AgentGates L2 batch rejected candidate(s): " + ", ".join(failures))
        return out_empty, "", "", 0.0

    payload: dict[str, object] = {"schema": FAST_REQUEST_SCHEMA, "request_id": req_id, "mode": "batch", "warmup": int(warmup), "repeat": int(repeat), "require_all_passed": bool(require_all_passed), "response_path": resp_path.relative_to(root).as_posix(), "batches": req_batches, "msprof_targets": []}
    t_npu0 = time.monotonic()
    raw_resp, execution = _hook("_submit_fast_gate", _submit_fast_gate)(
        agent_gates_bin=ag_bin, root=root, workspace=_hook("_agent_gates_workspace_name", _agent_gates_workspace_name)(root),
        request_id=req_id, request_relative_path=req_path.relative_to(root).as_posix(), request_path=req_path, response_path=resp_path,
        request_sha256=_write_fast_request(req_path, payload), mode="batch", timeout_sec=float(timeout_sec), device_id=str(device_id), device_pool=device_pool, env_vals=env_vals,
    )
    l2_npu_sec = time.monotonic() - t_npu0
    is_df, is_dr, _ = _inspect_v2_device_fault(execution.report, execution.verification)
    if is_df or is_dr or execution.device_rejected:
        bad = execution.device_id or str(device_id)
        raise AgentGatesDeviceFaultError(f"AgentGates device-fast receipt reported device_fault on card {bad}", fault_plane="INFRA_DEVICE_FAULT", fault_code="NPU_DEVICE_TSD_FAULT", failed_tier="l2_npu", device_id=bad, rejected_cards=[bad] if bad else ())
    _verify_fast_bindings(raw_resp, staged)
    _verify_fast_coverage(raw_resp, expected_candidates=sum(len(valid_by_op[op]) for op, _, _ in batches), expected_msprof_targets=0)
    raw_hbm, raw_nl = raw_resp.get("_selected_card_ext_hbm_mb", 0), raw_resp.get("null_launch_us")
    card_hbm = max(0, int(raw_hbm)) if isinstance(raw_hbm, (int, float)) and not isinstance(raw_hbm, bool) else 0
    null_us = float(raw_nl) if isinstance(raw_nl, (int, float)) and not isinstance(raw_nl, bool) else None
    ops_raw = _fast_response_mapping(raw_resp, "operators")
    out_by_op: dict[str, dict[str, CandidateSurrogateResult]] = {}
    for op_name, cands, _ in batches:
        op_fres, valid_c, parsed, run_ids = failed_res_by_op.get(op_name, {}), valid_by_op.get(op_name, []), {}, {}
        if valid_c:
            wop = wire_ops[op_name]
            craw = ops_raw.get(wop) if wop in ops_raw or wop == op_name else ops_raw.get(op_name)
            if not isinstance(craw, Mapping):
                raise RuntimeError(f"fast response has no results for operator {op_name!r}")
            for cid, _ in valid_c:
                if not isinstance(item := craw.get(cid), Mapping):
                    raise RuntimeError(f"fast response has no result for {op_name}/{cid}")
                res = CandidateSurrogateResult.from_dict(item)
                if card_hbm > 0 and any(a.card_ext_hbm_mb == 0 for a in res.anchors):
                    res = replace(res, anchors=tuple(replace(a, card_ext_hbm_mb=card_hbm) if a.card_ext_hbm_mb == 0 else a for a in res.anchors))
                parsed[cid], run_ids[cid] = res, execution.run_id
                if not res.all_passed:
                    failures.append(f"{op_name}/{cid}")
        out_by_op[op_name] = L2BatchResults({c: (parsed[c] if c in parsed else op_fres[c]) for c, _ in cands if c in parsed or c in op_fres}, run_ids, prewarm_stats=pw_stats, l2_prewarm_sec=l2_pw_sec, l2_npu_sec=l2_npu_sec, null_launch_us=null_us)
    if require_all_passed and failures:
        raise RuntimeError("AgentGates L2 batch rejected candidate(s): " + ", ".join(failures))
    return out_by_op, execution.run_id, execution.receipt_path, _fast_held_seconds(execution.report)


def run_surrogate_batch_via_agent_gates(
    operator_name: str, candidates: Sequence[tuple[str, str | Path]], anchors: Sequence[AnchorCaseSpec] | None = None, *,
    source_root: str | Path | None = None, env_file: str | Path | None = None,
    device_id: str = "1", device_pool: Sequence[str] = (), warmup: int = 3, repeat: int = 5,
    require_all_passed: bool = False, timeout_sec: int = 300,
) -> tuple[dict[str, CandidateSurrogateResult], str, str]:
    """Evaluate all L1-passing candidates of one round inside a single AgentGates window."""
    active = tuple(anchors) if anchors is not None else get_default_anchors(operator_name)
    by_op, run_id, receipt_path, _ = run_multi_operator_surrogate_batch_via_agent_gates(
        [(operator_name, candidates, active)], source_root=source_root, env_file=env_file,
        device_id=device_id, device_pool=device_pool, warmup=warmup, repeat=repeat,
        require_all_passed=require_all_passed, timeout_sec=timeout_sec,
    )
    return (L2BatchResults({}, {}) if (op_res := by_op.get(operator_name)) is None else op_res), run_id, receipt_path


def _strip_cpp_comments_keep_strings(text: str) -> str:
    out, i, n, in_str = [], 0, len(text), ""
    while i < n:
        ch, nxt = text[i], text[i + 1] if i + 1 < n else ""
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(nxt); i += 2; continue
            if ch == in_str:
                in_str = ""
            i += 1
        elif ch in ('"', "'"):
            in_str = ch; out.append(ch); i += 1
        elif ch == "/" and nxt == "/":
            i += 2
            while i < n and text[i] != "\n":
                i += 1
        elif ch == "/" and nxt == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
        else:
            out.append(ch); i += 1
    return "".join(out)


class _NormalizedAstStrings(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str) and any(tok in node.value for tok in _CPP_LITERAL_MARKERS):
            norm = hashlib.sha256(" ".join(_strip_cpp_comments_keep_strings(node.value).split()).encode("utf-8")).hexdigest()
            return ast.copy_location(ast.Constant(value=f"__cpp_tu__:{norm}"), node)
        return node


def _normalized_module_ast_signature(source: str) -> str | None:
    """Return canonical AST+TU sha256 signature for Python module source, or None on SyntaxError."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    norm = _NormalizedAstStrings().visit(tree)
    ast.fix_missing_locations(norm)
    return hashlib.sha256(ast.dump(norm, include_attributes=False).encode("utf-8")).hexdigest()


def run_l0_static_precheck(
    module_source: str, *, builder_name: str, builder_kwargs: Mapping[str, object],
    config_selector_name: str | None = None, pass_configs: dict[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None, dims: Mapping[str, object] | None = None,
    var_bindings: Mapping[str, int] | None = None, case_entry: Mapping[str, object] | None = None,
    proto: Mapping[str, object] | None = None, ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
) -> StageAResult:
    """Run Tier L0 static AST, ledger, and router inspection on Host CPU."""
    shapes = extract_ordered_tensor_shapes(case_entry=case_entry, dims=dims, proto=proto, ordered_tensor_shapes=ordered_tensor_shapes)
    return run_stage_a_static(
        module_source, factory_name=builder_name, factory_kwargs=builder_kwargs, pass_configs=pass_configs,
        compile_flags=compile_flags, config_selector_name=config_selector_name, dims=dims, var_bindings=var_bindings, ordered_tensor_shapes=shapes,
    )
