"""Structured round-funnel telemetry and deterministic search-boundary precipitation."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from tilefoundry_ascend.agent.specs.fault_types import FaultPlane, L3Resolution, TypedFault
from tilefoundry_ascend.ascend.json_record import JsonRecordMixin

MAX_CLEAN_CALIBRATION_BURST_CV: float = 0.08
SPEARMAN_QUARANTINE_FLOOR: float = 0.20
ANOMALY_LOG_ERROR_THRESHOLD: float = 1.20
_SHA_CODES = frozenset(
    {
        "NOOP_IDENTICAL", "PATCH_ANCHOR_MISS", "TRACKB_SLICE_SPLICE_ERR",
        "SIGNATURE_CONTRACT_MISMATCH", "VF_STACK_OVERFLOW", "UB_SRAM_OVERFLOW", "L1_SRAM_OVERFLOW",
    }
)
_KNOB_CODES = frozenset({"VF_STACK_OVERFLOW", "UB_SRAM_OVERFLOW", "L1_SRAM_OVERFLOW"})


def _num(raw: object, cast: type[int] | type[float], default: int | float) -> int | float:
    if isinstance(raw, bool):
        return default
    if isinstance(raw, (int, float)):
        return cast(raw)
    if isinstance(raw, str):
        try:
            return cast(raw.strip())
        except ValueError:
            return default
    return default


def is_clean_calibration_sample(
    burst_batch_cv: float, *, max_cv: float = MAX_CLEAN_CALIBRATION_BURST_CV
) -> bool:
    """Return True when ring-burst jitter is quiet enough for Tier-2 weight updates."""
    return math.isfinite(burst_batch_cv) and float(burst_batch_cv) <= float(max_cv)


def compute_spearman_rho(
    predicted_us: Mapping[str, float], measured_us: Mapping[str, float]
) -> float | None:
    """Compute Spearman rank correlation across shared candidate IDs (requires n >= 3)."""
    shared = [
        cid for cid in predicted_us
        if cid in measured_us and predicted_us[cid] > 0.0 and measured_us[cid] > 0.0
    ]
    n = len(shared)
    if n < 3:
        return None

    def _ranks(values: Sequence[float]) -> list[float]:
        indexed = sorted(enumerate(values), key=lambda item: item[1])
        out = [0.0] * len(values)
        idx = 0
        while idx < len(indexed):
            end = idx
            while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[idx][1]:
                end += 1
            avg = 0.5 * (idx + end) + 1.0
            for pos in range(idx, end + 1):
                out[indexed[pos][0]] = avg
            idx = end + 1
        return out

    rx = _ranks([float(predicted_us[cid]) for cid in shared])
    ry = _ranks([float(measured_us[cid]) for cid in shared])
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    denom = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return 0.0 if denom <= 1e-12 else max(-1.0, min(1.0, cov / denom))


@dataclass(frozen=True)
class RoundFunnelEvent(JsonRecordMixin):
    """Single candidate lifecycle record persisted to round_funnel_events.jsonl."""

    run_id: str
    operator: str
    round_idx: int
    regime_tag: str
    topology_axes: dict[str, str]
    candidate_id: str
    cand_sha: str
    worker_kind: str
    rule_id: str | None = None
    passed_l0_l1: bool = False
    failed_tier: str = ""
    error_code_norm: str = "OK"
    repaired: bool = False
    repair_delta_lines: int = 0
    spill_bytes: int = 0
    stack_bytes: int = 0
    ub_bytes: int = 0
    knob_signature: str = ""
    analytical_pred_us: float = 0.0
    measured_l2_us: float = 0.0
    burst_batch_cv: float = 0.0
    card_ext_hbm_mb: int = 0
    promoted_l3: bool = False
    l3_failure_stage: str = ""
    delta_speedup: float = 0.0
    noop_identical: bool = False
    wall_sec: float = 0.0
    stage_timings_sec: dict[str, float] = field(default_factory=dict)
    prewarm_stats: dict[str, int] = field(default_factory=dict)
    cost_advisor_predictions: int = 0
    fault_plane: str = FaultPlane.NONE.value
    l3_resolution: str = L3Resolution.NOT_RUN.value
    is_real_npu: bool = False
    timestamp: float = field(default_factory=time.time)

    def apply_typed_fault(self, fault: TypedFault | None) -> RoundFunnelEvent:
        """Return an updated event populated from a structured TypedFault."""
        if fault is None:
            return replace(
                self, fault_plane=FaultPlane.NONE.value, error_code_norm="OK",
                failed_tier="", noop_identical=False,
            )
        return replace(
            self, fault_plane=fault.plane.value, error_code_norm=fault.code or self.error_code_norm,
            failed_tier=fault.failed_tier or self.failed_tier,
            noop_identical=bool(self.noop_identical or fault.plane == FaultPlane.NOOP_IDENTICAL),
        )

    @classmethod
    def from_typed_fault(
        cls, *, run_id: str, operator: str, round_idx: int, regime_tag: str,
        topology_axes: dict[str, str], candidate_id: str, cand_sha: str, worker_kind: str,
        fault: TypedFault | None, rule_id: str | None = None,
        l3_resolution: str = L3Resolution.NOT_RUN.value, is_real_npu: bool = False,
        wall_sec: float = 0.0,
    ) -> RoundFunnelEvent:
        """Construct a RoundFunnelEvent directly from a TypedFault payload."""
        passed = fault is None or fault.plane == FaultPlane.NONE
        return cls(
            run_id=run_id, operator=operator, round_idx=round_idx, regime_tag=regime_tag,
            topology_axes=topology_axes, candidate_id=candidate_id, cand_sha=cand_sha,
            worker_kind=worker_kind, rule_id=rule_id, passed_l0_l1=passed,
            failed_tier="" if fault is None else fault.failed_tier,
            error_code_norm="OK" if fault is None else (fault.code or ("OK" if passed else fault.plane.value)),
            noop_identical=bool(fault is not None and fault.plane == FaultPlane.NOOP_IDENTICAL),
            fault_plane=FaultPlane.NONE.value if fault is None else fault.plane.value,
            l3_resolution=l3_resolution, is_real_npu=is_real_npu, wall_sec=wall_sec,
        )


def assert_unmodeled_fault_share_bounded(
    events: Sequence[RoundFunnelEvent], *, max_share: float = 0.10
) -> float:
    """Assert unmodeled faults stay strictly below *max_share* of faulted events."""
    faults = [e for e in events if str(e.fault_plane) != FaultPlane.NONE.value]
    if not faults:
        return 0.0
    share = sum(1 for e in faults if str(e.fault_plane) == FaultPlane.UNMODELED_FAULT.value) / len(faults)
    if share >= float(max_share):
        raise AssertionError(f"unmodeled fault share {share:.3f} >= {float(max_share):.3f}")
    return share


def append_round_funnel_event(telemetry_dir: Path, event: RoundFunnelEvent) -> Path:
    """Append one candidate funnel record to telemetry/round_funnel_events.jsonl."""
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    out_path = telemetry_dir / "round_funnel_events.jsonl"
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
    return out_path


@dataclass
class DeterministicSearchBoundaryLedger(JsonRecordMixin):
    """Persist verified infeasible envelopes, noop digests, and CostModel quarantines."""

    rejected_cand_shas: dict[str, str] = field(default_factory=dict)
    infeasible_knobs_by_regime: dict[str, dict[str, str]] = field(default_factory=dict)
    min_overflow_ub_bytes: dict[str, int] = field(default_factory=dict)
    min_overflow_stack_knobs: dict[str, list[str]] = field(default_factory=dict)
    spearman_history: dict[str, list[float]] = field(default_factory=dict)

    def record_compile_wall(
        self,
        *,
        regime_tag: str,
        cand_sha: str,
        knob_signature: str,
        error_code_norm: str,
        ub_bytes: int = 0,
        stack_bytes: int = 0,
    ) -> bool:
        """Record a deterministic physical or noop compile boundary; return True if new."""
        added = False
        if cand_sha and error_code_norm in _SHA_CODES and cand_sha not in self.rejected_cand_shas:
            self.rejected_cand_shas[cand_sha] = error_code_norm
            added = True
        if knob_signature and error_code_norm in _KNOB_CODES:
            bucket = self.infeasible_knobs_by_regime.setdefault(regime_tag, {})
            if knob_signature not in bucket:
                bucket[knob_signature] = error_code_norm
                added = True
        if error_code_norm == "UB_SRAM_OVERFLOW" and ub_bytes > 0:
            prev = self.min_overflow_ub_bytes.get(regime_tag)
            if prev is None or ub_bytes < prev:
                self.min_overflow_ub_bytes[regime_tag] = int(ub_bytes)
                added = True
        if error_code_norm == "VF_STACK_OVERFLOW" and stack_bytes > 0 and knob_signature:
            items = self.min_overflow_stack_knobs.setdefault(regime_tag, [])
            entry = f"{knob_signature} (stack_bytes={stack_bytes})"
            if entry not in items:
                items.append(entry)
                del items[:-6]
                added = True
        return added

    def precheck_known_infeasible(
        self, *, regime_tag: str, cand_sha: str, knob_signature: str = ""
    ) -> str | None:
        """Return deterministic rejection reason if candidate hits a known wall."""
        if cand_sha and cand_sha in self.rejected_cand_shas:
            return f"deterministic_boundary_sha:{self.rejected_cand_shas[cand_sha]}"
        bucket = self.infeasible_knobs_by_regime.get(regime_tag) or {}
        if knob_signature and knob_signature in bucket:
            return f"deterministic_boundary_knobs:{bucket[knob_signature]}"
        return None

    def format_regime_boundary_prompt(self, regime_tag: str) -> str:
        """Render purely objective physical search boundaries for Round t+1 prompts."""
        lines: list[str] = []
        ub_ceil = self.min_overflow_ub_bytes.get(regime_tag)
        if ub_ceil is not None:
            lines.append(f"- Verified UB SRAM overflow ceiling: ub_bytes >= {ub_ceil} failed Stage-B.")
        for item in self.min_overflow_stack_knobs.get(regime_tag) or ():
            lines.append(f"- Verified VF stack overflow point: {item}.")
        knob_map = self.infeasible_knobs_by_regime.get(regime_tag) or {}
        if knob_map:
            formatted = ", ".join(f"{k}->{v}" for k, v in list(knob_map.items())[-5:])
            lines.append(f"- Infeasible knob tuples (do not repeat): {formatted}")
        if not lines:
            return ""
        return "### VERIFIED PHYSICAL SEARCH BOUNDARIES (ROUND HISTORY)\n" + "\n".join(lines)

    def record_spearman_rho(self, bucket_key: str, rho: float | None) -> None:
        """Append one finite Spearman rank correlation sample to the rolling window."""
        if rho is None or not math.isfinite(rho):
            return
        hist = self.spearman_history.setdefault(bucket_key, [])
        hist.append(round(float(rho), 6))
        del hist[:-4]

    def is_cost_prescreen_quarantined(self, bucket_key: str) -> bool:
        """Quarantine CostGate pruning if 2 consecutive rounds invert ranking (rho < 0.2)."""
        hist = self.spearman_history.get(bucket_key) or []
        return len(hist) >= 2 and all(val < SPEARMAN_QUARANTINE_FLOOR for val in hist[-2:])

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> DeterministicSearchBoundaryLedger:
        """Reconstruct ledger state from a serialized dictionary."""
        if not isinstance(payload, Mapping):
            return cls()
        shas, knobs = payload.get("rejected_cand_shas"), payload.get("infeasible_knobs_by_regime")
        ub, stk, sp = (
            payload.get("min_overflow_ub_bytes"),
            payload.get("min_overflow_stack_knobs"),
            payload.get("spearman_history"),
        )
        return cls(
            rejected_cand_shas={str(k): str(v) for k, v in shas.items()} if isinstance(shas, Mapping) else {},
            infeasible_knobs_by_regime=(
                {str(r): {str(k): str(v) for k, v in m.items()} for r, m in knobs.items() if isinstance(m, Mapping)}
                if isinstance(knobs, Mapping)
                else {}
            ),
            min_overflow_ub_bytes={str(k): int(_num(v, int, 0)) for k, v in ub.items()} if isinstance(ub, Mapping) else {},
            min_overflow_stack_knobs=(
                {str(k): [str(x) for x in v] for k, v in stk.items() if isinstance(v, Sequence) and not isinstance(v, (str, bytes))}
                if isinstance(stk, Mapping)
                else {}
            ),
            spearman_history=(
                {str(k): [float(_num(x, float, 0.0)) for x in v] for k, v in sp.items() if isinstance(v, Sequence) and not isinstance(v, (str, bytes))}
                if isinstance(sp, Mapping)
                else {}
            ),
        )


__all__ = [
    "ANOMALY_LOG_ERROR_THRESHOLD", "DeterministicSearchBoundaryLedger",
    "MAX_CLEAN_CALIBRATION_BURST_CV", "RoundFunnelEvent", "SPEARMAN_QUARANTINE_FLOOR",
    "append_round_funnel_event", "assert_unmodeled_fault_share_bounded",
    "compute_spearman_rho", "is_clean_calibration_sample",
]
