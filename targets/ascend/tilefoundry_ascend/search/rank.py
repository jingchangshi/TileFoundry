"""Device-only ranking boundary for native search results.

The ranking layer accepts only candidates measured by a real device timer with a
nonempty synchronization note. Simulated-clock samples may be recorded for
diagnostics, but they are never rankable and never satisfy a device gate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

OK = "OK"
COMPILE_FAILED = "COMPILE_FAILED"
COMPILE_TIMEOUT = "COMPILE_TIMEOUT"
NUMERIC_FAILED = "NUMERIC_FAILED"
LAUNCH_CONTRACT_VIOLATED = "LAUNCH_CONTRACT_VIOLATED"
RUN_TIMEOUT = "RUN_TIMEOUT"
RUN_ERROR = "RUN_ERROR"
PRUNED_STATIC = "PRUNED_STATIC"
STATUSES = (
    OK,
    COMPILE_FAILED,
    COMPILE_TIMEOUT,
    NUMERIC_FAILED,
    LAUNCH_CONTRACT_VIOLATED,
    RUN_TIMEOUT,
    RUN_ERROR,
    PRUNED_STATIC,
)

MEASUREMENT_SOURCE_DEVICE = "DEVICE_TIMER"
MEASUREMENT_SOURCE_SIMULATED = "SIMULATED_CLOCK"
MEASUREMENT_SOURCES = (MEASUREMENT_SOURCE_DEVICE, MEASUREMENT_SOURCE_SIMULATED)


class RankError(RuntimeError):
    """A ranking contract violation with a stable rejection code."""

    def __init__(self, code: str, message: str, subject: str = "", expected=None, actual=None):
        self.code = code
        self.message = message
        self.subject = subject
        self.expected = expected
        self.actual = actual
        super().__init__("%s: %s" % (code, message))

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "subject": self.subject,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass(frozen=True)
class CandidateMeasurement:
    """One candidate's recorded outcome and, only when device-timed, samples."""

    key: str
    operator: str
    status: str = OK
    measurement_source: str = MEASUREMENT_SOURCE_SIMULATED
    samples_us: tuple[float, ...] = ()
    footprint_bytes: int = 0
    run_id: str = ""
    identity_key: str = ""
    sync_note: str = ""
    plan_digest: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise RankError("RANK_KEY_INVALID", "candidate key must be a nonempty string", "key", "nonempty string", self.key)
        if self.status not in STATUSES:
            raise RankError("RANK_UNKNOWN_STATUS", "unknown candidate status", "status", STATUSES, self.status)
        if self.measurement_source not in MEASUREMENT_SOURCES:
            raise RankError(
                "RANK_UNKNOWN_MEASUREMENT_SOURCE",
                "measurement source must be DEVICE_TIMER or SIMULATED_CLOCK",
                "measurement_source",
                MEASUREMENT_SOURCES,
                self.measurement_source,
            )
        if not isinstance(self.footprint_bytes, int) or isinstance(self.footprint_bytes, bool) or self.footprint_bytes < 0:
            raise RankError("RANK_BAD_FOOTPRINT", "footprint_bytes must be a nonnegative int", "footprint_bytes", ">= 0", self.footprint_bytes)
        samples = tuple(self.samples_us)
        for value in samples:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
                raise RankError("RANK_BAD_SAMPLE", "every sample must be a finite positive number of microseconds", "samples_us", "finite > 0", value)
        object.__setattr__(self, "samples_us", tuple(float(value) for value in samples))
        if self.status != OK and samples:
            raise RankError(
                "RANK_REJECTED_HAS_SAMPLES",
                "a non-OK candidate must not carry latency samples",
                self.key,
                "no samples",
                self.status,
            )
        if self.status == OK and not samples:
            raise RankError("RANK_OK_WITHOUT_SAMPLES", "an OK candidate must carry raw samples", self.key, "nonempty samples", ())
        if self.status == OK and self.measurement_source == MEASUREMENT_SOURCE_DEVICE and not self.sync_note:
            raise RankError(
                "RANK_DEVICE_SYNC_NOTE_MISSING",
                "a device-timed candidate must state the synchronization note",
                self.key,
                "nonempty sync_note",
                self.sync_note,
            )
        if self.status == OK and self.measurement_source == MEASUREMENT_SOURCE_DEVICE and not self.identity_key:
            raise RankError(
                "RANK_DEVICE_IDENTITY_MISSING",
                "a device-timed candidate must name the measured binary identity",
                self.key,
                "nonempty identity_key",
                self.identity_key,
            )

    @property
    def rankable(self) -> bool:
        return (
            self.status == OK
            and self.measurement_source == MEASUREMENT_SOURCE_DEVICE
            and bool(self.samples_us)
            and bool(self.sync_note)
            and bool(self.identity_key)
        )

    @property
    def median_us(self) -> float:
        ordered = sorted(self.samples_us)
        if not ordered:
            raise RankError("RANK_NO_SAMPLES", "candidate has no samples", self.key, "samples", ())
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    @property
    def relative_spread(self) -> float:
        if not self.samples_us:
            return math.inf
        mean = sum(self.samples_us) / len(self.samples_us)
        if mean <= 0:
            return math.inf
        variance = sum((value - mean) ** 2 for value in self.samples_us) / len(self.samples_us)
        return math.sqrt(variance) / mean

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "operator": self.operator,
            "status": self.status,
            "measurement_source": self.measurement_source,
            "samples_us": list(self.samples_us),
            "footprint_bytes": self.footprint_bytes,
            "run_id": self.run_id,
            "identity_key": self.identity_key,
            "sync_note": self.sync_note,
            "plan_digest": self.plan_digest,
            "reason": self.reason,
            "rankable": self.rankable,
        }


@dataclass(frozen=True)
class Ranking:
    winner: CandidateMeasurement | None
    ordered: tuple[CandidateMeasurement, ...]
    rejected: tuple[CandidateMeasurement, ...]
    rule: str
    comparable: bool

    def as_dict(self) -> dict:
        return {
            "winner": self.winner.key if self.winner is not None else None,
            "ordered": [candidate.key for candidate in self.ordered],
            "rejected": [candidate.key for candidate in self.rejected],
            "rule": self.rule,
            "comparable": self.comparable,
        }


def rankable(candidate: CandidateMeasurement) -> bool:
    """True only for a device-timed, synchronized, successful candidate."""
    return isinstance(candidate, CandidateMeasurement) and candidate.rankable


def require_device_source(candidate: CandidateMeasurement) -> CandidateMeasurement:
    """Refuse a non-device measurement instead of silently ranking it."""
    if candidate.measurement_source != MEASUREMENT_SOURCE_DEVICE:
        raise RankError(
            "RANK_SIMULATED_NOT_COMPARABLE",
            "a simulated measurement cannot be ranked or satisfy a device gate",
            candidate.key,
            MEASUREMENT_SOURCE_DEVICE,
            candidate.measurement_source,
        )
    if not candidate.rankable:
        raise RankError("RANK_NOT_DEVICE_READY", "candidate is not device-rankable", candidate.key, "device-timed OK candidate", candidate.status)
    return candidate


def rank(candidates: Iterable[CandidateMeasurement]) -> Ranking:
    """Rank only device-timed candidates; keep every rejection visible."""
    rows = tuple(candidates)
    for candidate in rows:
        if not isinstance(candidate, CandidateMeasurement):
            raise RankError("RANK_BAD_CANDIDATE", "rank input must be CandidateMeasurement", "candidate", CandidateMeasurement.__name__, type(candidate).__name__)
    timed = tuple(candidate for candidate in rows if candidate.rankable)
    rejected = tuple(candidate for candidate in rows if not candidate.rankable)
    if not timed:
        return Ranking(
            winner=None,
            ordered=(),
            rejected=rejected,
            rule="no device-timed candidate passed; simulated or failed candidates are never rankable",
            comparable=False,
        )
    ordered = tuple(sorted(timed, key=lambda candidate: (candidate.median_us, candidate.footprint_bytes, candidate.relative_spread, candidate.key)))
    return Ranking(
        winner=ordered[0],
        ordered=ordered,
        rejected=rejected,
        rule="device timer median over raw samples; ties broken by footprint, spread, then key",
        comparable=True,
    )


def top_k(candidates: Iterable[CandidateMeasurement], k: int) -> tuple[CandidateMeasurement, ...]:
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise RankError("RANK_BAD_K", "k must be a nonnegative int", "k", ">= 0", k)
    return rank(candidates).ordered[:k]


def confirmation_required(left: CandidateMeasurement, right: CandidateMeasurement, *, noise_fraction: float = 0.02) -> bool:
    """Return whether two device-timed candidates are inside their combined noise."""
    left = require_device_source(left)
    right = require_device_source(right)
    if not isinstance(noise_fraction, (int, float)) or isinstance(noise_fraction, bool) or not math.isfinite(float(noise_fraction)) or noise_fraction < 0:
        raise RankError("RANK_BAD_NOISE", "noise_fraction must be finite and nonnegative", "noise_fraction", ">= 0", noise_fraction)
    gap = abs(left.median_us - right.median_us)
    noise = max(
        left.median_us * left.relative_spread,
        right.median_us * right.relative_spread,
        left.median_us * float(noise_fraction),
        right.median_us * float(noise_fraction),
    )
    return gap <= noise


__all__ = [
    "OK",
    "COMPILE_FAILED",
    "COMPILE_TIMEOUT",
    "NUMERIC_FAILED",
    "LAUNCH_CONTRACT_VIOLATED",
    "RUN_TIMEOUT",
    "RUN_ERROR",
    "PRUNED_STATIC",
    "STATUSES",
    "MEASUREMENT_SOURCE_DEVICE",
    "MEASUREMENT_SOURCE_SIMULATED",
    "MEASUREMENT_SOURCES",
    "RankError",
    "CandidateMeasurement",
    "Ranking",
    "rankable",
    "require_device_source",
    "rank",
    "top_k",
    "confirmation_required",
]
