"""Target event facts and the completion record vocabulary.

NOTHING HERE READS A PROGRAM. This module answers three questions the rest of the package
must not answer twice:

  1. which CORE DOMAINS exist, so a cross-core handoff is not mistaken for an intra-core one;
  2. which EVENT RESOURCE NAMESPACES the target exposes, and how many finite ids each holds;
  3. which named refusal a violation gets.

An "event namespace" is a HARDWARE resource identity, not a logical channel name. Two
channels that resolve to the same namespace compete for the same finite ids; two channels in
different namespaces may legitimately hold the same number. The namespace is therefore
(core_domain, kind, pool pipe pair, bank) -- resolved through the target's POOLS, so a
wildcard pool that covers several pipe pairs correctly collapses them into one shared
namespace instead of pretending they are independent.

Capacity facts are DECLARED and BOUNDED, not measured. The 950PR record below is the local
3510 target record (28 AIC / 56 AIV, GM/UB/L1/L0 spaces) plus a bounded event budget per
namespace; every fact carries its source and is overridable, because an event budget that
nobody declared would make "the pool is exhausted" untraceable. A profile with no declared
budget is a refusal, not a default.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Optional

# ---------------------------------------------------------------- core domains
#: The Vector engine domain. Its UB/L1 shadow and its pipes are per-AIV.
AIV = "aiv"
#: The Cube engine domain.
AIC = "aic"
#: Device-global mechanisms (a cross-core handoff is NOT owned by either side alone).
DEVICE = "device"

#: The completion obligation kinds this package derives. These are LOGICAL kinds; the
#: hardware mechanism is chosen later from the target's pools and pipe pairs.
KIND_READY = "ready"
KIND_FREE = "free"
KIND_STATE = "state"
KIND_EXIT = "exit"
KIND_CROSS = "cross"
OBLIGATION_KINDS = (KIND_READY, KIND_FREE, KIND_STATE, KIND_EXIT, KIND_CROSS)


class CompletionError(RuntimeError):
    """Raised for every refusal in this package.

    'code' is the stable identity a test asserts on; the message is for a human and is
    allowed to be longer. 'needs_replan' distinguishes "this completion program is unusable"
    from "the schedule must be asked for again".
    """

    def __init__(self, code, message, source_id="", expected=None, actual=None,
                 needs_replan=False):
        self.code = code
        self.message = message
        self.source_id = source_id
        self.expected = expected
        self.actual = actual
        self.needs_replan = bool(needs_replan)
        super().__init__("%s: %s (source=%r expected=%r actual=%r)"
                         % (code, message, source_id, expected, actual))

    def as_dict(self):
        return {"code": self.code, "message": self.message, "source": self.source_id,
                "expected": self.expected, "actual": self.actual}


def digest(payload) -> str:
    """A stable short digest over a JSON-shaped payload (dict order independent)."""
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ------------------------------------------------------------------ use intervals
@dataclass(frozen=True)
class UseInterval:
    """One event id's ACTIVE LIFETIME inside its namespace.

    Half-open [start, stop) in the completion trace's instance order, which is the order in
    which the emitted statements execute. 'generation' is the ring generation of the buffer
    this use belongs to (None for a non-ring one-shot use). Reuse is legal only when two
    intervals do not overlap; the generation is then the evidence that the later use is a
    NEW data generation rather than a replay of the old one.
    """

    start: int
    stop: int
    generation: Optional[int] = None

    def overlaps(self, other: "UseInterval") -> bool:
        return self.start < other.stop and other.start < self.stop

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Use:
    """One logical channel's use of a physical event id, at one ring generation."""

    channel: str
    obligation_id: str
    role: str                     # set | wait
    generation: int
    slot: int
    interval: UseInterval

    def as_dict(self):
        return asdict(self)


# ------------------------------------------------------------------- obligation
@dataclass(frozen=True)
class Obligation:
    """One completion obligation, bound to the REAL accesses that create it.

    'producer_access' / 'consumer_access' are plan access ids. An obligation with no
    supporting access cannot be constructed by this package, which is what makes "the set is
    derived, not templated" checkable rather than a claim.

    'consumers' is the full consumer set for a FREE edge: the obligation is discharged only
    when the LAST of them has completed its read, matching the ring protocol's
    done == set(consumers) rule. A FREE that waits on one consumer is early FREE.
    """

    obligation_id: str
    kind: str
    buffer_root: str
    producer_task: str
    consumer_task: str
    producer_pipe: str
    consumer_pipe: str
    producer_core: str
    consumer_core: str
    versions: int
    distance: int
    generation_domain: tuple
    consumers: tuple = ()
    producer_access: str = ""
    consumer_access: str = ""
    guard: object = None
    ring: bool = True
    evidence: str = ""

    @property
    def cross_core(self) -> bool:
        return self.producer_core != self.consumer_core

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


# ----------------------------------------------------------- pipe completion facts
#: What this target DECLARES about each pipe's retirement, and which core domain it lives in.
#:
#: 'visible_at_issue' is the only fact that can justify dropping a synchronisation: it says
#: the pipe's memory effect is already visible when the instruction is issued, so program
#: order in the same core domain is sufficient. The vector pipes declare "issue" as their
#: completion in the ISA contract; the DMA pipes declare "issue+visible", which means the
#: transfer is NOT complete at issue and an event is required. A pipe with no entry here
#: cannot be claimed to be ordered at all -- the caller must use an event.
PIPE_FACTS = {
    "vector": {"core": AIV, "completion": "issue", "visible_at_issue": True,
               "async": False, "source": "ISA contract: vector ops complete at issue"},
    "scalar": {"core": AIV, "completion": "issue", "visible_at_issue": True,
               "async": False, "source": "ISA contract: scalar ops complete at issue"},
    "mte2": {"core": AIV, "completion": "issue+visible", "visible_at_issue": False,
             "async": True, "source": "ISA contract: GM->UB DMA is not complete at issue"},
    "mte3": {"core": AIV, "completion": "issue+visible", "visible_at_issue": False,
             "async": True, "source": "ISA contract: UB->GM DMA is not complete at issue"},
    "fix": {"core": AIC, "completion": "issue+write_visible", "visible_at_issue": False,
            "async": True, "source": "ISA contract: FIXPIPE write visibility follows issue"},
    "cube": {"core": AIC, "completion": "issue+write_visible", "visible_at_issue": False,
             "async": True, "source": "ISA contract: MMAD result is not visible at issue"},
    # The DMA pipes as they are spelled in the ledger/projection layer.
    "DMA": {"core": AIV, "completion": "issue+visible", "visible_at_issue": False,
            "async": True, "source": "projection DMA pipe maps to the asynchronous MTE path"},
}


def is_async_pipe(pipe) -> bool:
    """Does this pipe complete asynchronously? An unknown pipe is treated as async."""
    fact = PIPE_FACTS.get(str(pipe))
    return True if fact is None else bool(fact["async"])


def core_for_pipe(pipe, facts=None) -> str:
    """The CORE DOMAIN a pipe belongs to. An unknown pipe is refused, never defaulted.

    Defaulting to AIV would silently make an unknown Cube pipe look like a Vector pipe and
    turn a real cross-core handoff into a same-core edge -- the exact mistake the cross-core
    obligations exist to prevent.
    """
    fact = PIPE_FACTS.get(str(pipe))
    if fact is None:
        raise CompletionError("COMPLETION_PIPE_UNKNOWN",
                              "pipe has no declared core domain, so an edge on it cannot be"
                              " classified as same-core or cross-core",
                              str(pipe), sorted(PIPE_FACTS), pipe)
    return fact["core"]


# --------------------------------------------------------------- target namespaces
@dataclass(frozen=True)
class EventPool:
    """A finite event-id budget the target exposes.

    A 'pipe_pair' of None means the pool is shared by every pipe pair in that core domain,
    so the resource namespace must NOT include the pipe pair: those channels compete for the
    same ids. That is the difference between "independent namespace" and "shared namespace",
    and it is a target fact rather than an allocator convention.
    """

    core_domain: str
    kind: str
    capacity: int
    pipe_pair: Optional[tuple] = None
    banks: int = 1
    source: str = "declared bounded event budget"

    def covers(self, core_domain, kind, pipe_pair) -> bool:
        if self.core_domain != core_domain or self.kind != kind:
            return False
        return self.pipe_pair is None or tuple(self.pipe_pair) == tuple(pipe_pair)

    def as_dict(self):
        return {"core_domain": self.core_domain, "kind": self.kind,
                "capacity": self.capacity,
                "pipe_pair": list(self.pipe_pair) if self.pipe_pair else None,
                "banks": self.banks, "source": self.source}


@dataclass(frozen=True)
class EventNamespace:
    """A resolved HARDWARE resource identity. Ids are independent across namespaces."""

    core_domain: str
    kind: str
    pipe_pair: tuple = ()
    bank: int = 0

    def key(self) -> str:
        pipes = "-".join(self.pipe_pair) if self.pipe_pair else "*"
        return "%s/%s/%s/bank%d" % (self.core_domain, self.kind, pipes, self.bank)

    def as_dict(self):
        return {"core_domain": self.core_domain, "kind": self.kind,
                "pipe_pair": list(self.pipe_pair), "bank": self.bank}


@dataclass(frozen=True)
class TargetEventFacts:
    """The event surface of one target profile, with its declared capacities."""

    profile_id: str
    core_domains: tuple
    pools: tuple
    cross_core_mechanism: str = "cross_core_event"
    source: str = ""

    def pool_for(self, core_domain, kind, pipe_pair):
        """The pool that OWNS this channel, or None when the target declares none.

        A pool that names the pipe pair wins over a wildcard pool for the same
        (core_domain, kind): the more specific resource is the one the hardware has.
        """
        specific = None
        wildcard = None
        for pool in self.pools:
            if not pool.covers(core_domain, kind, pipe_pair):
                continue
            if pool.pipe_pair is None:
                wildcard = wildcard or pool
            else:
                specific = specific or pool
        return specific or wildcard

    def namespace_for(self, core_domain, kind, pipe_pair):
        """The resource namespace + its pool, or (None, None) when undeclared.

        The pool's OWN pipe pair (not the channel's) goes into the key: a wildcard pool
        therefore yields ONE namespace for every pipe pair. This is why two channels on
        different pipe pairs are not automatically independent -- they are independent only
        when the target gives each pair its own pool.
        """
        pool = self.pool_for(core_domain, kind, pipe_pair)
        if pool is None:
            return None, None
        pipes = tuple(pool.pipe_pair) if pool.pipe_pair is not None else ()
        return EventNamespace(core_domain, kind, pipes, 0), pool

    def as_dict(self):
        return {"profile_id": self.profile_id,
                "core_domains": list(self.core_domains),
                "cross_core_mechanism": self.cross_core_mechanism,
                "source": self.source,
                "pools": [pool.as_dict() for pool in self.pools]}


#: The declared 950PR event surface.
#:
#: The capacities are a DECLARED BOUND from this round's target record, not a measurement of
#: silicon: this package needs a finite number so "exhausted" is a real state, and a test can
#: pass a smaller pool to observe exhaustion. Every pool is per core domain, so the AIV and
#: AIC namespaces are independent by construction. The AIV pools are keyed by core domain and
#: kind but NOT by pipe pair, so all AIV pipe pairs in one kind share ids: a program with
#: more concurrent channels than the budget is refused rather than silently over-numbered.
ASCEND_EVENT_FACTS = TargetEventFacts(
    profile_id="ascend950pr_9579",
    core_domains=(AIV, AIC, DEVICE),
    pools=(
        EventPool(AIV, KIND_READY, 16, pipe_pair=None, banks=2,
                  source="declared AIV ready-event budget (2 banks x 16), local 3510 record"),
        EventPool(AIV, KIND_FREE, 16, pipe_pair=None, banks=2,
                  source="declared AIV free-event budget (2 banks x 16), local 3510 record"),
        EventPool(AIV, KIND_STATE, 8, pipe_pair=None, banks=1,
                  source="declared AIV state-ordering budget"),
        EventPool(AIV, KIND_EXIT, 4, pipe_pair=None, banks=1,
                  source="declared AIV region-exit budget"),
        EventPool(AIC, KIND_READY, 16, pipe_pair=None, banks=2,
                  source="declared AIC ready-event budget (2 banks x 16)"),
        EventPool(AIC, KIND_FREE, 16, pipe_pair=None, banks=2,
                  source="declared AIC free-event budget (2 banks x 16)"),
        EventPool(AIC, KIND_STATE, 8, pipe_pair=None, banks=1,
                  source="declared AIC state-ordering budget"),
        EventPool(AIC, KIND_EXIT, 4, pipe_pair=None, banks=1,
                  source="declared AIC region-exit budget"),
        # A cross-core handoff is a DEVICE-level resource: AIC->AIV and AIV->AIC do not share
        # ids with each other or with the per-domain events above.
        EventPool(DEVICE, KIND_CROSS, 16, pipe_pair=None, banks=1,
                  source="declared cross-core handoff event budget"),
    ),
    source="local Ascend950PR_9579 target record; capacities are declared bounds, not measured",
)


def event_facts_for(profile_id: Optional[str] = None, *, facts=None) -> TargetEventFacts:
    """The event facts for a profile. An unknown profile is a refusal, never a default."""
    if facts is not None:
        return facts
    if profile_id is None or profile_id == ASCEND_EVENT_FACTS.profile_id:
        return ASCEND_EVENT_FACTS
    raise CompletionError("COMPLETION_TARGET_UNKNOWN",
                          "no declared event surface for this target profile",
                          str(profile_id), ASCEND_EVENT_FACTS.profile_id, profile_id)


__all__ = [
    "AIV", "AIC", "DEVICE", "KIND_READY", "KIND_FREE", "KIND_STATE", "KIND_EXIT",
    "KIND_CROSS", "OBLIGATION_KINDS", "CompletionError", "digest", "UseInterval", "Use",
    "Obligation", "EventPool", "EventNamespace", "TargetEventFacts",
    "PIPE_FACTS", "is_async_pipe", "core_for_pipe",
    "ASCEND_EVENT_FACTS", "event_facts_for",
]
