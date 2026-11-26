"""Structured round-funnel telemetry and deterministic search-boundary precipitation."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

MAX_CLEAN_CALIBRATION_BURST_CV: float = 0.08
SPEARMAN_QUARANTINE_FLOOR: float = 0.20
ANOMALY_LOG_ERROR_THRESHOLD: float = 1.20

_ERROR_CODE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("NOOP_IDENTICAL", re.compile(r"noop_identical|identical_to_parent", re.IGNORECASE)),
    (
        "PATCH_ANCHOR_MISS",
        re.compile(
            r"patch_anchor_miss|patchanchormiss|search block did not match|"
            r"identical to parent track-b|zero effective change|"
            r"valid search/replace block|empty search block",
            re.IGNORECASE,
        ),
    ),
    (
        "VF_STACK_OVERFLOW",
        re.compile(
            r"exceeded vf stack size|vf_stack|stack frame size|stack overflow",
            re.IGNORECASE,
        ),
    ),
    (
        "UB_SRAM_OVERFLOW",
        re.compile(r"ub[_\s-]*overflow|ubuf.*exceed|ub_bytes", re.IGNORECASE),
    ),
    ("L1_SRAM_OVERFLOW", re.compile(r"l1[_\s-]*overflow|cbuf.*exceed", re.IGNORECASE)),
    ("T_STAGE_NESTING_ILLEGAL", re.compile(r"t\.stage|stage.*nest", re.IGNORECASE)),
    ("G5_MONOTONICITY_LOCK", re.compile(r"g5|monotonicity|forbidden_regression", re.IGNORECASE)),
    ("FREEDOM_MASK_VIOLATION", re.compile(r"freedom_mask|locked_facet", re.IGNORECASE)),
    (
        "TRACKB_SLICE_SPLICE_ERR",
        re.compile(
            r"omitted_span|dead_code.*trackb|trackb.*dead_code|splice",
            re.IGNORECASE,
        ),
    ),
    (
        "SIGNATURE_CONTRACT_MISMATCH",
        re.compile(
            r"signature.*mismatch|missing regime symbols|entrypoint.*not define|"
            r"arity|unexpected keyword",
            re.IGNORECASE,
        ),
    ),
    (
        "DYNAMIC_STRIDE_CONTRACT_ERR",
        re.compile(
            r"stride.*contract|non[-_\s]?contiguous|buffer_map.*shape",
            re.IGNORECASE,
        ),
    ),
    (
        "CCE_UNDECLARED_INTRINSIC",
        re.compile(
            r"undeclared identifier|no matching function|implicit declaration|unknown type name",
            re.IGNORECASE,
        ),
    ),
    ("COST_COARSE_REJECT", re.compile(r"cost_prescreen|coarse_reject", re.IGNORECASE)),
)


def _coerce_int(raw: object, default: int = 0) -> int:
    if isinstance(raw, bool):
        return default
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(raw.strip())
        except ValueError:
            return default
    return default


def _coerce_float(raw: object, default: float = 0.0) -> float:
    if isinstance(raw, bool):
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw.strip())
        except ValueError:
            return default
    return default


def normalize_compile_error_code(
    failed_tier: str | None,
    error_message: str,
    full_diagnostic: str = "",
) -> str:
    """Classify Stage-A/Stage-B/prescreen rejections into canonical telemetry codes."""
    joined = f"{failed_tier or ''} {error_message or ''} {full_diagnostic or ''}".strip()
    if not joined:
        return "OK"
    for code, pattern in _ERROR_CODE_RULES:
        if pattern.search(joined):
            return code
    tier_slug = re.sub(r"[^A-Za-z0-9_]+", "_", (failed_tier or "UNKNOWN").upper()).strip("_")
    return f"{tier_slug}_OTHER"


def is_clean_calibration_sample(
    burst_batch_cv: float,
    *,
    max_cv: float = MAX_CLEAN_CALIBRATION_BURST_CV,
) -> bool:
    """Return True when ring-burst jitter is quiet enough for Tier-2 weight updates."""
    return math.isfinite(burst_batch_cv) and float(burst_batch_cv) <= float(max_cv)


def compute_spearman_rho(
    predicted_us: Mapping[str, float],
    measured_us: Mapping[str, float],
) -> float | None:
    """Compute Spearman rank correlation across shared candidate IDs (requires n >= 3)."""
    shared = [
        cid
        for cid in predicted_us
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
            avg_rank = 0.5 * (idx + end) + 1.0
            for pos in range(idx, end + 1):
                out[indexed[pos][0]] = avg_rank
            idx = end + 1
        return out

    rx = _ranks([float(predicted_us[cid]) for cid in shared])
    ry = _ranks([float(measured_us[cid]) for cid in shared])
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    denom = math.sqrt(vx * vy)
    if denom <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, cov / denom))


@dataclass(frozen=True)
class RoundFunnelEvent:
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
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, object]:
        """Serialize funnel event to a JSON-compatible dictionary."""
        return asdict(self)


def append_round_funnel_event(telemetry_dir: Path, event: RoundFunnelEvent) -> Path:
    """Append one candidate funnel record to telemetry/round_funnel_events.jsonl."""
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    out_path = telemetry_dir / "round_funnel_events.jsonl"
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
    return out_path


@dataclass
class DeterministicSearchBoundaryLedger:
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
        if cand_sha and error_code_norm in {
            "NOOP_IDENTICAL",
            "PATCH_ANCHOR_MISS",
            "TRACKB_SLICE_SPLICE_ERR",
            "SIGNATURE_CONTRACT_MISMATCH",
            "VF_STACK_OVERFLOW",
            "UB_SRAM_OVERFLOW",
            "L1_SRAM_OVERFLOW",
        }:
            if cand_sha not in self.rejected_cand_shas:
                self.rejected_cand_shas[cand_sha] = error_code_norm
                added = True
        if knob_signature and error_code_norm in {
            "VF_STACK_OVERFLOW",
            "UB_SRAM_OVERFLOW",
            "L1_SRAM_OVERFLOW",
        }:
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
        self,
        *,
        regime_tag: str,
        cand_sha: str,
        knob_signature: str = "",
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
            lines.append(
                f"- Verified UB SRAM overflow ceiling: ub_bytes >= {ub_ceil} failed Stage-B."
            )
        for item in self.min_overflow_stack_knobs.get(regime_tag) or ():
            lines.append(f"- Verified VF stack overflow point: {item}.")
        knob_map = self.infeasible_knobs_by_regime.get(regime_tag) or {}
        if knob_map:
            recent = list(knob_map.items())[-5:]
            formatted = ", ".join(f"{k}->{v}" for k, v in recent)
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

    def to_dict(self) -> dict[str, object]:
        """Serialize ledger state to a JSON-compatible dictionary."""
        return {
            "rejected_cand_shas": dict(self.rejected_cand_shas),
            "infeasible_knobs_by_regime": {
                k: dict(v) for k, v in self.infeasible_knobs_by_regime.items()
            },
            "min_overflow_ub_bytes": dict(self.min_overflow_ub_bytes),
            "min_overflow_stack_knobs": {
                k: list(v) for k, v in self.min_overflow_stack_knobs.items()
            },
            "spearman_history": {k: list(v) for k, v in self.spearman_history.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> DeterministicSearchBoundaryLedger:
        """Reconstruct ledger state from a serialized dictionary."""
        if not isinstance(payload, Mapping):
            return cls()
        raw_shas = payload.get("rejected_cand_shas")
        raw_knobs = payload.get("infeasible_knobs_by_regime")
        raw_ub = payload.get("min_overflow_ub_bytes")
        raw_stack = payload.get("min_overflow_stack_knobs")
        raw_spearman = payload.get("spearman_history")
        return cls(
            rejected_cand_shas=(
                {str(k): str(v) for k, v in raw_shas.items()}
                if isinstance(raw_shas, Mapping)
                else {}
            ),
            infeasible_knobs_by_regime=(
                {
                    str(r): {str(k): str(v) for k, v in m.items()}
                    for r, m in raw_knobs.items()
                    if isinstance(m, Mapping)
                }
                if isinstance(raw_knobs, Mapping)
                else {}
            ),
            min_overflow_ub_bytes=(
                {str(k): _coerce_int(v) for k, v in raw_ub.items()}
                if isinstance(raw_ub, Mapping)
                else {}
            ),
            min_overflow_stack_knobs=(
                {
                    str(k): [str(x) for x in v]
                    for k, v in raw_stack.items()
                    if isinstance(v, Sequence) and not isinstance(v, (str, bytes))
                }
                if isinstance(raw_stack, Mapping)
                else {}
            ),
            spearman_history=(
                {
                    str(k): [_coerce_float(x) for x in v]
                    for k, v in raw_spearman.items()
                    if isinstance(v, Sequence) and not isinstance(v, (str, bytes))
                }
                if isinstance(raw_spearman, Mapping)
                else {}
            ),
        )


__all__ = [
    "ANOMALY_LOG_ERROR_THRESHOLD",
    "DeterministicSearchBoundaryLedger",
    "MAX_CLEAN_CALIBRATION_BURST_CV",
    "RoundFunnelEvent",
    "SPEARMAN_QUARANTINE_FLOOR",
    "append_round_funnel_event",
    "compute_spearman_rho",
    "is_clean_calibration_sample",
    "normalize_compile_error_code",
]
