"""P7-01/G69: the compile/verify/profile loop and its measurement statistics.

THE THREE THINGS THIS FILE EXISTS TO GET RIGHT, each of which turns a number into a lie when it
is wrong:

1. THE CACHE KEY MUST INVALIDATE. Spec 05 section 5.7 and spec 08 section 8.3 both require the
   cache identity to cover source, target, toolchain, shapes/dtype/layout, precision policy and
   the ACTUAL plan. A key made of the shape alone will happily hand back the binary of a different
   schedule, and the resulting timing looks perfectly plausible. So the key here is a digest over
   an explicitly enumerated identity, and a test changes each ingredient in turn to prove the key
   changes.

2. A BAD CANDIDATE MUST NOT BECOME A WINNER. A failed compile, a numeric failure, a timeout and a
   launch-contract violation are four DIFFERENT outcomes, and none of them is a latency. They are
   recorded as their own statuses, they carry no samples, and ranking refuses them explicitly
   rather than sorting NaN to the bottom and calling the survivor the winner.

3. THE STATISTICS MUST BE OVER RAW SAMPLES. Median and p90 come from the retained sample list, so
   a reader can recompute them. Warmup is discarded and SAID to be discarded; the count of
   discarded calls is part of the record. Compile time is reported separately from kernel time
   because spec 05 section 5.7 requires the first compile to be listed on its own.

WHAT IS MEASURED, EXACTLY. The default timer is the harness's own injectable clock, so the
statistics, the ranking and the noise rule are testable without a device. The device backend
plugs the same interface; it is NOT implemented by guessing at a wall clock around an
asynchronous launch, which spec 05 section 5.7 forbids. A run made with the synthetic clock is
labelled MEASUREMENT_SOURCE_SIMULATED in every record and can never satisfy a device gate -- the
same distinction attnres/model.py draws between structural and device verification.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from . import filter as flt
from . import generator as gen
from .process import Deadline, WorkerCommand


_case_deadline = ContextVar("search_case_deadline", default=None)

# ------------------------------------------------------------------ statuses

OK = "OK"
COMPILE_FAILED = "COMPILE_FAILED"
COMPILE_TIMEOUT = "COMPILE_TIMEOUT"
NUMERIC_FAILED = "NUMERIC_FAILED"
LAUNCH_CONTRACT_VIOLATED = "LAUNCH_CONTRACT_VIOLATED"
RUN_TIMEOUT = "RUN_TIMEOUT"
RUN_ERROR = "RUN_ERROR"
PRUNED_STATIC = "PRUNED_STATIC"
STATUSES = (OK, COMPILE_FAILED, COMPILE_TIMEOUT, NUMERIC_FAILED, LAUNCH_CONTRACT_VIOLATED,
            RUN_TIMEOUT, RUN_ERROR, PRUNED_STATIC)
#: Statuses that carry a timing at all. Everything else is a rejection, not a slow candidate.
TIMED_STATUSES = (OK,)

MEASUREMENT_SOURCE_SIMULATED = "SIMULATED_CLOCK"
MEASUREMENT_SOURCE_DEVICE = "DEVICE_TIMER"


class HarnessError(RuntimeError):
    def __init__(self, code, message, subject="", expected=None, actual=None):
        self.code = code
        self.message = message
        self.subject = subject
        self.expected = expected
        self.actual = actual
        super().__init__("%s: %s" % (code, message))

    def as_dict(self):
        return {"code": self.code, "message": self.message, "subject": self.subject,
                "expected": self.expected, "actual": self.actual}


def _digest(payload) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ------------------------------------------------------------------ P0 budget

@dataclass(frozen=True)
class Budget:
    """The P0 search budget, in the units spec 8.3 defines: CANDIDATES PER CASE.

    Spec 8.3 is explicit that the 40-candidate cap is per case and not per operator or per
    package, so this type carries per-case counts and a deadline per case, and every report says
    which of them bound. Keeping the numbers here (rather than inline in a loop) is what lets a
    report state the budget that actually applied.
    """

    max_candidates_per_case: int = 40
    seed_candidates_max: int = 12
    neighbor_candidates_max: int = 20
    advanced_candidates_max: int = 8
    top_k_for_refinement: int = 3
    solver_timeout_ms: int = 10000
    candidate_compile_deadline_s: int = 120
    case_search_deadline_s: int = 1800
    warmup_calls: int = 10
    timing_calls_per_batch: int = 50
    confirmation_batches: int = 3
    #: How many measurements of one candidate must land inside the spread before two candidates
    #: are called a tie. Spec 08 section 8.9: a difference inside the noise is a tie, and a tie is
    #: broken by the smaller resource footprint rather than by the luckier sample.
    tie_noise_fraction: float = 0.02

    def __post_init__(self):
        for name in ("seed_candidates_max", "neighbor_candidates_max", "advanced_candidates_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise HarnessError("BUDGET_BAD_COUNT", "%s must be a nonnegative integer" % name)
        for name in ("solver_timeout_ms", "candidate_compile_deadline_s", "case_search_deadline_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise HarnessError("BUDGET_BAD_DEADLINE", "%s must be finite and positive" % name)
        if self.max_candidates_per_case < 1:
            raise HarnessError("BUDGET_BAD_TOTAL", "the per-case cap must be positive",
                               "max_candidates_per_case", ">= 1", self.max_candidates_per_case)
        if (self.seed_candidates_max + self.neighbor_candidates_max +
                self.advanced_candidates_max) > self.max_candidates_per_case:
            raise HarnessError(
                "BUDGET_ORIGINS_OVER_CAP",
                "seeds plus neighbours plus advanced may not exceed the per-case cap",
                "seeds+neighbors+advanced", "<= %d" % self.max_candidates_per_case,
                self.seed_candidates_max + self.neighbor_candidates_max +
                self.advanced_candidates_max)
        if self.advanced_candidates_max > self.max_candidates_per_case:
            raise HarnessError("BUDGET_ADVANCED_OVER_CAP",
                               "advanced candidates may not exceed the per-case cap",
                               "advanced_candidates_max",
                               "<= %d" % self.max_candidates_per_case,
                               self.advanced_candidates_max)
        for name, value in (("warmup_calls", self.warmup_calls),
                            ("timing_calls_per_batch", self.timing_calls_per_batch),
                            ("confirmation_batches", self.confirmation_batches),
                            ("top_k_for_refinement", self.top_k_for_refinement)):
            if value < 1:
                raise HarnessError("BUDGET_BAD_COUNT", "%s must be >= 1" % name, name, ">= 1",
                                   value)
        if not 0.0 <= self.tie_noise_fraction < 1.0:
            raise HarnessError("BUDGET_BAD_NOISE", "the tie fraction must be in [0, 1)",
                               "tie_noise_fraction", "[0, 1)", self.tie_noise_fraction)

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def case_budget(self, *, seeds: int, neighbours: int, advanced: int) -> "CaseBudget":
        """The effective per-case budget after the declared maxima are applied."""
        return CaseBudget(
            max_candidates=min(self.max_candidates_per_case, seeds + neighbours + advanced),
            seeds=min(seeds, self.seed_candidates_max),
            neighbours=min(neighbours, self.neighbor_candidates_max),
            advanced=min(advanced, self.advanced_candidates_max),
            deadline_s=self.case_search_deadline_s,
            source=("P0 budget: 12 seed + 20 neighbour + 8 advanced per case (spec 8.3), "
                    "applied per CASE and not per operator"))


@dataclass(frozen=True)
class CaseBudget:
    max_candidates: int
    seeds: int
    neighbours: int
    advanced: int
    deadline_s: int
    source: str

    def as_dict(self) -> dict:
        return {"max_candidates": self.max_candidates, "seeds": self.seeds,
                "neighbours": self.neighbours, "advanced": self.advanced,
                "deadline_s": self.deadline_s, "source": self.source}


P0_BUDGET = Budget()


# ------------------------------------------------------------------ cache identity

@dataclass(frozen=True)
class CacheIdentity:
    """Everything that can change the BINARY a measurement refers to.

    Spec 08 section 8.3 lists the ingredients: source, target, toolchain, shape/dtype/layout,
    precision policy and the resolved plan. Two candidates whose identities differ must not share
    a compiled artifact, or a timing recorded for one is attributed to the other. The fields are
    enumerated here rather than concatenated from whatever was in scope, so an omission is a
    visible missing field instead of a silently absent ingredient.
    """

    operator: str
    target_profile: str
    toolchain: str
    source_digest: str
    plan_digest: str
    dtype: str
    layout: str
    shape_digest: str
    precision_policy: str
    candidate_key: str
    extra: tuple = ()

    def __post_init__(self):
        for name in ("operator", "target_profile", "toolchain", "source_digest",
                     "plan_digest", "dtype", "layout", "shape_digest", "precision_policy",
                     "candidate_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise HarnessError(
                    "CACHE_IDENTITY_INCOMPLETE",
                    "cache identity requires a non-empty %s; an identity with a blank ingredient "
                    "would let two different binaries collide" % name, name, "non-empty string",
                    value)

    def as_dict(self) -> dict:
        payload = {name: getattr(self, name) for name in
                   ("operator", "target_profile", "toolchain", "source_digest", "plan_digest",
                    "dtype", "layout", "shape_digest", "precision_policy", "candidate_key")}
        payload["extra"] = list(self.extra)
        return payload

    def key(self) -> str:
        return _digest(self.as_dict())


def source_digest(paths: Sequence[str]) -> str:
    """A digest over the source files a candidate's code comes from.

    Reading the files is the point: a digest computed from the module name would not change when
    the kernel source changed, which is precisely the stale-binary failure the gate names.
    """
    hasher = hashlib.sha256()
    for path in sorted(paths):
        if not os.path.exists(path):
            raise HarnessError("CACHE_SOURCE_MISSING", "source file does not exist", path,
                               "an existing file", path)
        with open(path, "rb") as handle:
            hasher.update(os.path.basename(path).encode())
            hasher.update(handle.read())
    return hasher.hexdigest()[:16]


def plan_digest(payload) -> str:
    """A digest over the resolved plan (or any JSON-shaped description of it)."""
    return _digest(payload)


def toolchain_identity() -> str:
    """The active toolchain as an observation, not a promise.

    A run that cannot see a toolchain must not claim one, so a missing version reports the
    interpreter it did find rather than a remembered CANN string.
    """
    parts = ["python=%s" % platform.python_version(), "machine=%s" % platform.machine()]
    for name in ("CANN_VERSION", "TILELANG_VERSION", "TF_TOOLCHAIN"):
        value = os.environ.get(name)
        if value:
            parts.append("%s=%s" % (name.lower(), value))
    return ";".join(parts)

# ------------------------------------------------------------------ samples and stats

@dataclass(frozen=True)
class Stats:
    """Descriptive statistics over the RETAINED raw samples, with the discards stated.

    p90 is computed by linear interpolation over the sorted samples, so a reader can reproduce
    the number from the raw list. Spread is reported as both an absolute (p90 - p10) and a
    relative figure because a ranking decision needs the relative one while a reader checking the
    timer resolution needs the absolute one.
    """

    count: int
    median_us: float
    p10_us: float
    p90_us: float
    minimum_us: float
    maximum_us: float
    mean_us: float
    spread_us: float
    relative_spread: float
    discarded_warmups: int
    samples_us: tuple = ()

    def as_dict(self) -> dict:
        return {"count": self.count, "median_us": self.median_us, "p10_us": self.p10_us,
                "p90_us": self.p90_us, "minimum_us": self.minimum_us,
                "maximum_us": self.maximum_us, "mean_us": self.mean_us,
                "spread_us": self.spread_us, "relative_spread": self.relative_spread,
                "discarded_warmups": self.discarded_warmups,
                "samples_us": list(self.samples_us)}


def _percentile(sorted_samples: Sequence[float], fraction: float) -> float:
    """Linear interpolation over sorted samples; exact for a one-sample list."""
    if not sorted_samples:
        raise HarnessError("STATS_EMPTY", "no samples", "samples", "at least 1", 0)
    if len(sorted_samples) == 1:
        return float(sorted_samples[0])
    position = fraction * (len(sorted_samples) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(sorted_samples) - 1)
    weight = position - low
    return float(sorted_samples[low]) * (1.0 - weight) + float(sorted_samples[high]) * weight


def compute_stats(samples_us: Sequence[float], *, discarded_warmups: int = 0) -> Stats:
    """Summarise raw kernel timings. Refuses a list with no samples instead of inventing one."""
    samples = [float(s) for s in samples_us]
    if not samples:
        raise HarnessError("STATS_EMPTY",
                           "no timing samples; a candidate with no samples has no latency",
                           "samples", "at least 1", len(samples))
    if any(not math.isfinite(s) or s < 0 for s in samples):
        raise HarnessError("STATS_BAD_SAMPLE",
                           "samples must be finite and non-negative", "samples",
                           "finite, >= 0", samples[:8])
    ordered = sorted(samples)
    p10 = _percentile(ordered, 0.10)
    p90 = _percentile(ordered, 0.90)
    median = statistics.median(ordered)
    return Stats(
        count=len(ordered),
        median_us=float(median),
        p10_us=p10,
        p90_us=p90,
        minimum_us=float(ordered[0]),
        maximum_us=float(ordered[-1]),
        mean_us=float(sum(ordered) / float(len(ordered))),
        spread_us=float(p90 - p10),
        relative_spread=float((p90 - p10) / median) if median > 0 else 0.0,
        discarded_warmups=int(discarded_warmups),
        samples_us=tuple(ordered),
    )


# ------------------------------------------------------------------ measurement sources

@dataclass
class SimulatedClock:
    """A deterministic timer, so statistics and ranking are testable without a device.

    Its readings are LABELLED simulated in every record it produces. It exists to prove the
    statistics, the cache invalidation and the rejection rules are correct; a timing taken with it
    can never be presented as a device measurement.
    """

    base_us: float = 100.0
    jitter_us: float = 4.0
    source: str = MEASUREMENT_SOURCE_SIMULATED

    def measure(self, run: Callable[[], None], calls: int) -> list:
        """Invoke run the requested number of times and return one fake timing per call.

        The jitter is a deterministic sawtooth over the call index rather than a random draw, so
        two runs of the same plan produce the identical sample list and a test can assert on the
        exact median.
        """
        out = []
        for index in range(calls):
            run()
            offset = (index % 5 - 2) * (self.jitter_us / 2.0)
            out.append(max(0.0, self.base_us + offset))
        return out


class DeviceTimerUnavailable(HarnessError):
    """Raised when a device measurement is requested and no calibrated timer exists.

    Spec 05 section 5.7 forbids timing a kernel with a wall clock around an asynchronous launch.
    When the target timer is not available the honest answer is a named refusal, not a number.
    """


@dataclass
class DeviceTimer:
    """The device path: an INJECTED calibrated timer, never a guessed one.

    The harness cannot know how to synchronise an Ascend launch by itself, so the caller supplies
    a callable that returns one kernel duration in microseconds AFTER synchronisation. If it is
    absent, measuring raises DeviceTimerUnavailable -- the harness will not fall back to
    time.perf_counter around an async call.
    """

    sample: Optional[Callable[[], float]] = None
    sync_note: str = ""
    source: str = MEASUREMENT_SOURCE_DEVICE

    def measure(self, run: Callable[[], None], calls: int) -> list:
        if self.sample is None:
            raise DeviceTimerUnavailable(
                "TIMER_NOT_CALIBRATED",
                "no calibrated device timer was supplied; a wall clock around an asynchronous "
                "launch is not a kernel measurement", "sample", "a callable", None)
        out = []
        for _ in range(calls):
            run()
            value = float(self.sample())
            if not math.isfinite(value) or value < 0:
                raise HarnessError("TIMER_BAD_READING", "the device timer returned an unusable "
                                   "reading", "sample", "finite, >= 0", value)
            out.append(value)
        return out


# ------------------------------------------------------------------ candidate result

@dataclass(frozen=True)
class CandidateResult:
    """One candidate's whole story: what it cost, what it measured, and why it stopped.

    Compile time is its own field because it is NOT a kernel cost (spec 05 section 5.7). Status is
    separate from the samples so a failed candidate cannot be mistaken for a fast one, and the
    samples are empty for every non-OK status by construction.
    """

    key: str
    operator: str
    family: str
    origin: str
    status: str
    identity_key: str
    compile_seconds: float = 0.0
    compile_cached: bool = False
    stats: Optional[Stats] = None
    reason: str = ""
    detail: str = ""
    plan_digest: str = ""
    measurement_source: str = MEASUREMENT_SOURCE_SIMULATED
    lower_bound_only: bool = True
    notes: tuple = ()

    def __post_init__(self):
        if self.status not in STATUSES:
            raise HarnessError("RESULT_UNKNOWN_STATUS", "unknown candidate status", "status",
                               STATUSES, self.status)
        # The key must be a STRING, not a bound method or any other callable. Passing the method
        # instead of calling it is an easy mistake that does not fail here -- it fails deep inside
        # a sort, where two methods compare with '<' and the whole ranking dies with a TypeError
        # that names nothing. Refusing it at construction names the candidate instead.
        if not isinstance(self.key, str) or not self.key:
            raise HarnessError(
                "RESULT_KEY_NOT_A_STRING",
                "a candidate key must be the digest string, not the key() method itself",
                repr(self.key)[:60], "a non-empty string", type(self.key).__name__)
        if self.status in TIMED_STATUSES and not (isinstance(self.identity_key, str)
                                                 and self.identity_key):
            raise HarnessError(
                "RESULT_MISSING_IDENTITY",
                "a timed candidate must name the cache identity of the binary it timed",
                self.key, "a non-empty identity digest", self.identity_key)
        if self.status in TIMED_STATUSES and self.stats is None:
            raise HarnessError("RESULT_MISSING_STATS",
                               "a timed candidate must carry its samples", self.key,
                               "a Stats record", None)
        if self.status not in TIMED_STATUSES and self.stats is not None:
            # This is the rule that stops a rejected candidate from entering a ranking: if it has
            # a latency it can be compared, and it must not be.
            raise HarnessError("RESULT_REJECTED_HAS_STATS",
                               "a candidate that did not pass compile, numeric and launch checks "
                               "must not carry a latency, or it can be ranked", self.key,
                               "no Stats", self.status)

    @property
    def rankable(self) -> bool:
        return self.status in TIMED_STATUSES and self.stats is not None

    def as_dict(self) -> dict:
        return {
            "key": self.key, "operator": self.operator, "family": self.family,
            "origin": self.origin, "status": self.status, "reason": self.reason,
            "detail": self.detail, "identity": self.identity_key,
            "plan_digest": self.plan_digest,
            "compile_seconds": self.compile_seconds,
            "compile_cached": self.compile_cached,
            "stats": self.stats.as_dict() if self.stats else None,
            "measurement_source": self.measurement_source,
            "lower_bound_only": self.lower_bound_only,
            "notes": list(self.notes),
        }


def pruned_result(judgement, *, operator: str = "") -> CandidateResult:
    """Turn a static rejection into a result row that carries no timing.

    This is the bridge between filter.py and the ranking: a PRUNED_STATIC candidate is a recorded
    outcome with the exact rejection codes attached, not a missing row and not a zero.
    """
    if judgement.feasible:
        raise HarnessError("RESULT_PRUNED_BUT_FEASIBLE",
                           "a FEASIBLE judgement must not be recorded as pruned",
                           judgement.key, "an infeasible judgement", judgement.verdict)
    return CandidateResult(
        key=judgement.key, operator=operator or judgement.operator, family=judgement.family,
        origin=judgement.origin, status=PRUNED_STATIC, identity_key="",
        reason=judgement.rejections[0].code if judgement.rejections else "PRUNED",
        detail="; ".join("%s(%s)" % (r.code, r.space) for r in judgement.rejections),
        notes=judgement.notes,
    )

# ------------------------------------------------------------------ the run loop

@dataclass(frozen=True)
class CandidateRun:
    """CPU simulated callback contract; device execution requires WorkerCandidateRun.

    Callbacks cannot be forcibly interrupted and are intended only for bounded CPU tests.

    The order is spec 08 section 8.4's order and is not negotiable: compile (with a total
    deadline), final legality, the ORIGINAL numeric comparator, the single-launch contract, and
    only then timing. Each callable returns None on success or a detail string on failure, so a
    failure carries its evidence into the record instead of collapsing into a boolean.
    """

    compile: Callable[[], object]
    numeric_check: Callable[[object], Optional[str]]
    launch_check: Callable[[object], Optional[str]]
    run: Callable[[object], None]
    plan: Callable[[object], object] = lambda compiled: None
    source_paths: tuple = ()

    def check(self):
        for name in ("compile", "numeric_check", "launch_check", "run"):
            if not callable(getattr(self, name)):
                raise HarnessError("RUN_CONTRACT_INCOMPLETE",
                                   "the candidate run contract needs every stage",
                                   name, "a callable", getattr(self, name, None))


@dataclass(frozen=True)
class WorkerCandidateRun:
    """Four enforced argv stages; profile owns warmup and synchronized sampling.

    TF_SEARCH_WARMUP_CALLS, TF_SEARCH_TIMING_CALLS_PER_BATCH and
    TF_SEARCH_CONFIRMATION_BATCHES tell the profile worker the required counts.
    Device workers must be launched by a caller holding the device lock.
    """
    compile: WorkerCommand
    numeric: WorkerCommand
    launch: WorkerCommand
    profile: WorkerCommand
    measurement_source: str = MEASUREMENT_SOURCE_DEVICE

    def check(self):
        for name in ("compile", "numeric", "launch", "profile"):
            if not isinstance(getattr(self, name), WorkerCommand):
                raise HarnessError("RUN_CONTRACT_INCOMPLETE", "an argv worker is required", name)
        if self.measurement_source not in (MEASUREMENT_SOURCE_DEVICE, MEASUREMENT_SOURCE_SIMULATED):
            raise HarnessError("RUN_BAD_SOURCE", "unknown measurement source")


def _run_worker_candidate(contract, base, budget, limit, deadline):
    compile_seconds = 0.0
    base = dict(base, measurement_source=contract.measurement_source)
    settings = {"TF_SEARCH_" + name.upper(): str(getattr(budget, name)) for name in
                ("warmup_calls", "timing_calls_per_batch", "confirmation_batches")}
    settings["TF_SEARCH_IDENTITY_KEY"] = base["identity_key"]
    payload = {}
    for stage in ("compile", "numeric", "launch", "profile"):
        timeout_status = COMPILE_TIMEOUT if stage == "compile" else RUN_TIMEOUT
        error_status = COMPILE_FAILED if stage == "compile" else RUN_ERROR
        started = time.monotonic()
        try:
            if deadline.remaining() <= 0:
                return CandidateResult(status=timeout_status, compile_seconds=compile_seconds,
                                       reason="SHARED_DEADLINE_EXCEEDED", detail=stage, **base)
            result, payload = getattr(contract, stage).execute(
                stage=stage, timeout_s=limit if stage == "compile" else budget.case_search_deadline_s,
                deadline=deadline, settings=settings)
            if stage == "compile":
                compile_seconds = result.elapsed_seconds
            if result.status != OK:
                return CandidateResult(status=result.status, compile_seconds=compile_seconds,
                                       reason="WORKER_" + result.status,
                                       detail=stage + ": " + result.log_path, **base)
            if deadline.remaining() <= 0:
                return CandidateResult(status=timeout_status, compile_seconds=compile_seconds,
                                       reason="SHARED_DEADLINE_EXCEEDED", detail=stage, **base)
            if stage != "profile":
                failure = payload["failure"]
                if failure is not None and not isinstance(failure, str):
                    raise ValueError("failure must be null or a string")
                if failure is not None:
                    status = {"compile": COMPILE_FAILED, "numeric": NUMERIC_FAILED,
                              "launch": LAUNCH_CONTRACT_VIOLATED}[stage]
                    return CandidateResult(status=status, compile_seconds=compile_seconds,
                                           reason=stage.upper() + "_REJECTED", detail=failure, **base)
            else:
                if payload["measurement_source"] != contract.measurement_source:
                    raise ValueError("worker measurement source mismatch")
                if (contract.measurement_source == MEASUREMENT_SOURCE_DEVICE
                        and not str(payload.get("sync_note", "")).strip()):
                    raise ValueError("device worker must describe synchronized calibrated timing")
                if payload["discarded_warmups"] != budget.warmup_calls:
                    raise ValueError("worker warmup count mismatch")
                if len(payload["samples_us"]) != (budget.timing_calls_per_batch *
                                                  budget.confirmation_batches):
                    raise ValueError("worker sample count mismatch")
                stats = compute_stats(payload["samples_us"], discarded_warmups=budget.warmup_calls)
        except Exception as exc:
            if stage == "compile":
                compile_seconds = time.monotonic() - started
            return CandidateResult(status=error_status, compile_seconds=compile_seconds,
                                   reason="WORKER_PROTOCOL_ERROR", detail=stage + ": " + str(exc),
                                   **base)
    return CandidateResult(status=OK, compile_seconds=compile_seconds, stats=stats,
                           **base)


def _timed(callable_):
    """Return (value, seconds) for one call, used only for COMPILE time, never for the kernel."""
    start = time.perf_counter()
    value = callable_()
    return value, time.perf_counter() - start


def run_candidate(contract: CandidateRun, identity: CacheIdentity, *, budget: Budget = P0_BUDGET,
                  timer=None, deadline_s: Optional[float] = None,
                  deadline: Optional[Deadline] = None) -> CandidateResult:
    """Compile, verify, check the launch contract, then time one candidate.

    A failure at any stage short-circuits: a candidate that failed the original comparator is
    never timed, because a latency for a numerically wrong kernel is not a slower candidate, it is
    a measurement of something else.
    """
    contract.check()
    timer = timer or SimulatedClock()
    limit = budget.candidate_compile_deadline_s if deadline_s is None else deadline_s
    base = dict(key=identity.candidate_key, operator=identity.operator,
                family=identity.precision_policy, origin="", identity_key=identity.key(),
                lower_bound_only=True)

    # The implicit case deadline preserves existing two-argument run_one callbacks.
    # Explicit deadlines can only tighten the case budget, never replace it.
    inherited = _case_deadline.get()
    active = Deadline(budget.case_search_deadline_s)
    for shared in (inherited, deadline):
        if shared is not None:
            active.expires = min(active.expires, time.monotonic() + shared.remaining())
    if isinstance(contract, WorkerCandidateRun):
        return _run_worker_candidate(contract, dict(base, plan_digest=identity.plan_digest),
                                     budget, limit, active)
    if not isinstance(timer, SimulatedClock) or timer.source != MEASUREMENT_SOURCE_SIMULATED:
        return CandidateResult(status=RUN_ERROR, reason="ENFORCED_WORKER_REQUIRED",
                               detail="device execution requires WorkerCandidateRun argv stages",
                               measurement_source=getattr(timer, "source", "UNKNOWN"), **base)
    # Legacy callbacks are CPU simulation only, with cooperative stage checks.
    if active.remaining() <= 0:
        return CandidateResult(status=COMPILE_TIMEOUT, reason="SHARED_DEADLINE_EXCEEDED", **base)
    compiled, compile_seconds = _timed(contract.compile)
    if compile_seconds > limit:
        return CandidateResult(status=COMPILE_TIMEOUT, compile_seconds=compile_seconds,
                               reason="COMPILE_DEADLINE_EXCEEDED",
                               detail="compile took %.1fs, deadline %.1fs"
                                      % (compile_seconds, limit), **base)

    failure = contract.numeric_check(compiled)
    if failure is not None:
        return CandidateResult(status=NUMERIC_FAILED, compile_seconds=compile_seconds,
                               reason="ORIGINAL_COMPARATOR_REJECTED", detail=str(failure),
                               **base)
    failure = contract.launch_check(compiled)
    if failure is not None:
        return CandidateResult(status=LAUNCH_CONTRACT_VIOLATED, compile_seconds=compile_seconds,
                               reason="LAUNCH_CONTRACT", detail=str(failure), **base)

    # Warmup is discarded and the DISCARD COUNT is recorded, so nobody has to guess whether the
    # first JIT-affected calls were included.
    timer.measure(lambda: contract.run(compiled), budget.warmup_calls)
    samples = []
    for _ in range(budget.confirmation_batches):
        samples.extend(timer.measure(lambda: contract.run(compiled),
                                     budget.timing_calls_per_batch))
    stats = compute_stats(samples, discarded_warmups=budget.warmup_calls)
    plan = contract.plan(compiled)
    return CandidateResult(status=OK, compile_seconds=compile_seconds, stats=stats,
                           plan_digest=plan_digest(plan) if plan is not None else "",
                           measurement_source=getattr(timer, "source",
                                                      MEASUREMENT_SOURCE_SIMULATED),
                           **base)


# ------------------------------------------------------------------ ranking

@dataclass(frozen=True)
class Ranking:
    """The outcome of comparing candidates: a winner, the runners-up, and the rule applied."""

    winner: Optional[CandidateResult]
    ordered: tuple
    ties: tuple
    rejected: tuple
    rule: str
    comparable: bool = True

    def as_dict(self) -> dict:
        return {
            "winner": self.winner.as_dict() if self.winner else None,
            "ordered": [r.key for r in self.ordered],
            "ties": [r.key for r in self.ties],
            "rejected": [{"key": r.key, "status": r.status, "reason": r.reason}
                         for r in self.rejected],
            "rule": self.rule,
            "comparable": self.comparable,
            "medians_us": {r.key: r.stats.median_us for r in self.ordered},
        }


def rank(results: Sequence[CandidateResult], *, budget: Budget = P0_BUDGET) -> Ranking:
    """Rank by median kernel time, refusing to rank anything that did not pass.

    The tie rule is spec 08 section 8.9: two candidates whose difference is inside the noise are a
    tie, and the tie is broken by the smaller footprint (then by the more stable sample spread,
    then by key for determinism). A candidate is never "not comparable" merely because it is
    slow -- that would silently drop results -- only because it has no measurement.
    """
    rejected = tuple(r for r in results if not r.rankable)
    timed = [r for r in results if r.rankable]
    if not timed:
        return Ranking(winner=None, ordered=(), ties=(), rejected=rejected,
                       rule="no candidate passed compile+numeric+launch, so there is no winner",
                       comparable=False)
    ordered = sorted(timed, key=lambda r: (r.stats.median_us, r.key))
    best = ordered[0]
    tolerance = best.stats.median_us * budget.tie_noise_fraction
    ties = tuple(r for r in ordered[1:] if r.stats.median_us - best.stats.median_us <= tolerance)
    rule = ("ranked by median kernel time over raw samples; a candidate within %.1f%% of the "
            "fastest is a tie and is broken by resource footprint" %
            (budget.tie_noise_fraction * 100.0))
    return Ranking(winner=best, ordered=tuple(ordered), ties=ties, rejected=rejected, rule=rule)


def confirmation_required(left: CandidateResult, right: CandidateResult, *,
                          budget: Budget = P0_BUDGET) -> bool:
    """Is the gap between two candidates inside their combined noise?

    If it is, spec 08 section 8.9 requires another confirmation batch before the ranking is
    trusted. The comparison uses each candidate's own relative spread rather than a global
    constant, because a jittery candidate and a stable one do not have the same uncertainty.
    """
    if not left.rankable or not right.rankable:
        raise HarnessError("RANK_NOT_COMPARABLE",
                           "a candidate without a measurement cannot be compared", "-",
                           "two timed candidates", (left.status, right.status))
    gap = abs(left.stats.median_us - right.stats.median_us)
    noise = max(left.stats.median_us * left.stats.relative_spread,
                right.stats.median_us * right.stats.relative_spread,
                left.stats.median_us * budget.tie_noise_fraction,
                right.stats.median_us * budget.tie_noise_fraction)
    return gap <= noise


# ------------------------------------------------------------------ neighbourhood expansion

@dataclass(frozen=True)
class SearchRound:
    """One completed search round for one case: what was tried and what it produced."""

    case_label: str
    operator: str
    budget: CaseBudget
    results: tuple
    ranking: Ranking
    rounds: int
    stop_reason: str
    notes: tuple = ()

    def as_dict(self) -> dict:
        return {"case_label": self.case_label, "operator": self.operator,
                "budget": self.budget.as_dict(), "rounds": self.rounds,
                "stop_reason": self.stop_reason,
                "ranking": self.ranking.as_dict(),
                "results": [r.as_dict() for r in self.results],
                "notes": list(self.notes),
                "counts": {
                    "total": len(self.results),
                    "timed": sum(1 for r in self.results if r.rankable),
                    "rejected": sum(1 for r in self.results if not r.rankable),
                    "by_status": _count_by_status(self.results),
                }}


def _count_by_status(results) -> dict:
    out = {}
    for result in results:
        out[result.status] = out.get(result.status, 0) + 1
    return out


def top_k(results: Sequence[CandidateResult], k: int) -> tuple:
    """The k fastest TIMED candidates; rejected ones are not eligible for refinement."""
    timed = [r for r in results if r.rankable]
    return tuple(sorted(timed, key=lambda r: (r.stats.median_us, r.key))[:max(0, k)])


def expand_neighbourhood(incumbents: Sequence, *, make_neighbours: Callable,
                         budget: CaseBudget) -> tuple:
    """Expand one local neighbourhood around the top-k incumbents (spec 08 section 8.4/8.9).

    Two limits are enforced here and both are recorded: the neighbour count from the budget, and
    the requirement that a neighbour differ from its parent in exactly one primary factor (which
    the candidate type itself enforces by refusing an unlabelled neighbour).
    """
    out = []
    for incumbent in incumbents:
        if len(out) >= budget.neighbours:
            break
        for neighbour in make_neighbours(incumbent):
            if len(out) >= budget.neighbours:
                break
            out.append(neighbour)
    return tuple(out)


# ------------------------------------------------------------------ the search driver

def search_case(*, case_label: str, operator: str, seeds: Sequence, advanced: Sequence,
                judge: Callable, run_one: Callable, budget: CaseBudget,
                make_neighbours: Optional[Callable] = None,
                search_budget: Budget = P0_BUDGET) -> SearchRound:
    """Run one case's bounded search: seeds, then advanced, then a neighbourhood expansion.

    The order matters and is deliberate. Seeds and advanced routes are the STRATIFIED coverage
    spec 08 section 8.4 asks for, so they are spent first; the neighbourhood is then expanded
    around whatever came out fastest, which is the only place where measurement is allowed to
    steer the search. The candidate count is capped by the per-case budget, and the stop reason
    says which bound applied.
    """
    if budget.max_candidates < 1:
        raise HarnessError("SEARCH_BAD_BUDGET", "a case budget needs at least one candidate",
                           "max_candidates", ">= 1", budget.max_candidates)
    results = []
    seen = set()
    deadline = Deadline(budget.deadline_s)
    deadline_exhausted = False
    device_stopped = False

    def admit(candidate, origin_label):
        nonlocal deadline_exhausted, device_stopped
        if device_stopped:
            return False
        if deadline.remaining() <= 0:
            deadline_exhausted = True
            return False
        key = candidate.key()
        if key in seen:
            return False
        if len(results) >= budget.max_candidates:
            return False
        seen.add(key)
        judgement = judge(candidate)
        if not judgement.feasible:
            results.append(pruned_result(judgement, operator=operator))
            return True
        if deadline.remaining() <= 0:
            deadline_exhausted = True
            return False
        token = _case_deadline.set(deadline)
        try:
            result = run_one(candidate, judgement)
        finally:
            _case_deadline.reset(token)
        results.append(result)
        if (result.measurement_source == MEASUREMENT_SOURCE_DEVICE
                and result.status in (COMPILE_TIMEOUT, RUN_TIMEOUT)):
            device_stopped = True
        return True

    spent_seeds = 0
    for candidate in seeds:
        if spent_seeds >= budget.seeds:
            break
        if admit(candidate, gen.ORIGIN_SEED):
            spent_seeds += 1
    spent_advanced = 0
    for candidate in advanced:
        if spent_advanced >= budget.advanced:
            break
        if admit(candidate, gen.ORIGIN_ADVANCED):
            spent_advanced += 1

    rounds = 1
    stop_reason = ""
    if make_neighbours is not None and not device_stopped and deadline.remaining() > 0:
        incumbents = top_k(results, search_budget.top_k_for_refinement)
        if incumbents:
            by_key = {c.key(): c for c in list(seeds) + list(advanced)}
            parents = [by_key[r.key] for r in incumbents if r.key in by_key]
            expansion = expand_neighbourhood(parents, make_neighbours=make_neighbours,
                                            budget=budget)
            if expansion:
                rounds = 2
                admitted = 0
                for candidate in expansion:
                    if admit(candidate, gen.ORIGIN_NEIGHBOR):
                        admitted += 1
                stop_reason = ("neighbourhood of the top %d seeded candidates expanded by %d "
                               "one-factor neighbours" % (len(parents), admitted))
            else:
                stop_reason = "no seed produced a timed result, so there was nothing to refine"
        else:
            stop_reason = "no candidate passed compile+numeric+launch, so no neighbourhood exists"
    else:
        stop_reason = "seeds and advanced routes only; no neighbourhood factory supplied"

    if len(results) >= budget.max_candidates:
        stop_reason += ("; stopped at the per-case candidate cap of %d"
                        % budget.max_candidates)
    if device_stopped:
        stop_reason += "; stopped after device worker timeout; device health review required"
    if deadline_exhausted or deadline.remaining() <= 0:
        stop_reason += "; stopped at the per-case wall deadline"
    ranking = rank(results, budget=search_budget)
    return SearchRound(case_label=case_label, operator=operator, budget=budget,
                       results=tuple(results), ranking=ranking, rounds=rounds,
                       stop_reason=stop_reason)


def summarise_rounds(rounds: Sequence[SearchRound]) -> dict:
    """Aggregate several case rounds into the numbers a gate report has to state."""
    by_status = {}
    by_family = {}
    timed = 0
    for search_round in rounds:
        for result in search_round.results:
            by_status[result.status] = by_status.get(result.status, 0) + 1
            entry = by_family.setdefault(result.family or "?", {"total": 0, "timed": 0})
            entry["total"] += 1
            if result.rankable:
                entry["timed"] += 1
                timed += 1
    return {
        "cases": len(rounds),
        "operators": sorted({r.operator for r in rounds}),
        "candidates": sum(len(r.results) for r in rounds),
        "timed": timed,
        "by_status": by_status,
        "by_family": by_family,
        "with_winner": sum(1 for r in rounds if r.ranking.winner is not None),
        "measurement_sources": sorted({res.measurement_source for r in rounds
                                       for res in r.results if res.rankable}),
    }


__all__ = [
    "OK", "COMPILE_FAILED", "COMPILE_TIMEOUT", "NUMERIC_FAILED",
    "LAUNCH_CONTRACT_VIOLATED", "RUN_TIMEOUT", "RUN_ERROR", "PRUNED_STATIC", "STATUSES",
    "TIMED_STATUSES", "MEASUREMENT_SOURCE_SIMULATED", "MEASUREMENT_SOURCE_DEVICE",
    "HarnessError", "Budget", "CaseBudget", "P0_BUDGET", "CacheIdentity", "source_digest",
    "plan_digest", "toolchain_identity", "Stats", "compute_stats", "SimulatedClock",
    "DeviceTimer", "DeviceTimerUnavailable", "CandidateResult", "pruned_result", "CandidateRun",
    "WorkerCandidateRun", "WorkerCommand", "run_candidate", "Ranking", "rank", "confirmation_required", "SearchRound", "top_k",
    "expand_neighbourhood", "search_case", "summarise_rounds",
]
