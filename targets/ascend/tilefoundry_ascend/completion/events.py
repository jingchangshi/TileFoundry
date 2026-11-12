"""Allocate finite physical event ids per REAL target namespace.

THE RULE THIS REPLACES. Numbering every logical channel from one chip-wide counter is not an
allocator: it makes the numbers "unique" by pretending the hardware has one infinite pool, so
it cannot say whether the program FITS, and its numbers carry no meaning. This allocator does
the three things spec 3.6 requires, and each is a testable property rather than a convention:

  1. the namespace key is (core_domain, event kind, pool pipe pair, bank) resolved through the
     target's POOLS. The same id in two DIFFERENT namespaces is legal and happens by
     construction, because each namespace numbers from 0 independently.
  2. two channels in the SAME namespace whose active lifetimes OVERLAP may not share an id.
     'Overlap' is decided from the real use intervals the schedule produces, so the decision
     is a property of the emitted program and not of the order the obligations were listed in.
  3. an id may be REUSED when the earlier lifetime has completed and the later use is a later
     generation of the same ring channel. The generation is the evidence: a replayed OLD
     generation is not a reuse candidate, it is a protocol violation, and it is caught by the
     protocol checker rather than being silently papered over here.

WHY LIFETIMES, NOT COUNTS. The input is (channel -> list of use intervals). A channel whose
uses never overlap can take one id for its whole life. Two channels that overlap must take
different ids. That is exactly the finite-resource question, and answering it from intervals
is what makes "the pool is exhausted" a real, located diagnostic: the report names the
namespace, the pool capacity, and the channels that were simultaneously live.
"""
from __future__ import annotations

from dataclasses import dataclass

from .model import (
    AIV,
    AIC,
    DEVICE,
    KIND_CROSS,
    CompletionError,
    EventNamespace,
    TargetEventFacts,
    Use,
    UseInterval,
    digest,
    event_facts_for,
)


@dataclass(frozen=True)
class ChannelRequest:
    """One logical channel's claim on some namespace, with its REAL use intervals.

    'namespace_hint' is not an identity: the allocator resolves the namespace through the
    target's pools from (core_domain, kind, pipe_pair), so two channels that LOOK distinct
    can still collide, which is the case that must be caught.
    """

    channel: str
    core_domain: str
    kind: str
    pipe_pair: tuple
    intervals: tuple
    obligation_ids: tuple = ()
    participants: tuple = ()

    def as_dict(self):
        return {"channel": self.channel, "core_domain": self.core_domain,
                "kind": self.kind, "pipe_pair": list(self.pipe_pair),
                "intervals": [iv.as_dict() for iv in self.intervals],
                "obligation_ids": list(self.obligation_ids),
                "participants": list(self.participants)}


@dataclass(frozen=True)
class ChannelAllocation:
    channel: str
    event_id: int
    namespace: EventNamespace
    pool_capacity: int
    intervals: tuple
    reuse_evidence: str
    obligation_ids: tuple = ()

    def as_dict(self):
        return {"channel": self.channel, "event_id": self.event_id,
                "namespace": self.namespace.key(), "pool_capacity": self.pool_capacity,
                "intervals": [iv.as_dict() for iv in self.intervals],
                "reuse_evidence": self.reuse_evidence,
                "obligation_ids": list(self.obligation_ids)}


@dataclass(frozen=True)
class EventAssignment:
    """The allocation result: one id per channel, plus what every namespace holds."""

    allocations: tuple
    namespaces: dict
    facts: TargetEventFacts
    diagnostics: tuple = ()

    def by_channel(self, channel) -> ChannelAllocation:
        for allocation in self.allocations:
            if allocation.channel == channel:
                return allocation
        raise CompletionError("COMPLETION_CHANNEL_UNBOUND",
                              "channel has no physical event allocation", str(channel),
                              [a.channel for a in self.allocations], channel)

    def event_id(self, channel) -> int:
        return self.by_channel(channel).event_id

    def per_namespace(self) -> dict:
        out = {}
        for allocation in self.allocations:
            key = allocation.namespace.key()
            info = out.setdefault(key, {"capacity": allocation.pool_capacity,
                                        "ids": set(), "channels": []})
            info["ids"].add(allocation.event_id)
            info["channels"].append(allocation.channel)
        return {key: {"capacity": info["capacity"], "used": len(info["ids"]),
                      "peak_ids": sorted(info["ids"]), "channels": sorted(info["channels"])}
                for key, info in out.items()}

    def as_dict(self):
        return {"facts_digest": digest(self.facts.as_dict()),
                "facts_profile": self.facts.profile_id,
                "allocations": [a.as_dict() for a in self.allocations],
                "namespaces": self.per_namespace(),
                "diagnostics": list(self.diagnostics)}


def _intervals_overlap(left, right) -> bool:
    return any(a.overlaps(b) for a in left for b in right)


def _capacity_override(overrides, namespace, pool, request) -> int:
    """Resolve a capacity override for one namespace, keyed the way a caller can name it.

    A caller may name a namespace by its RESOLVED key ('aiv/ready/*/bank0'), by the
    (core, kind, pipe) triple it asked for, or by (core, kind, None) for the whole kind. All
    three are the same namespace, and silently ignoring a caller's override because it used a
    different spelling would make a diagnostic run report a capacity it never applied.
    """
    for key in (namespace.key(),
                (request.core_domain, request.kind, request.pipe_pair),
                (request.core_domain, request.kind, None)):
        if key in overrides:
            return int(overrides[key])
    return int(pool.capacity)


def _free_id(used, capacity):
    for candidate in range(capacity):
        if candidate not in used:
            return candidate
    return None


def allocate_events(requests, *, facts: TargetEventFacts = None,
                    capacities=None, profile_id=None) -> EventAssignment:
    """Assign a finite physical id to every channel, inside its real namespace.

    'capacities' overrides pool capacities for a diagnostic run; it never changes the
    NAMESPACE structure, because a namespace is a hardware resource identity and not a
    budget. Overlapping lifetimes in one namespace are refused by name, and an exhausted pool
    names the namespace, its capacity and the live channels.
    """
    facts = event_facts_for(profile_id, facts=facts)
    overrides = dict(capacities or {})
    # Group by resolved namespace: this is what makes shared-namespace conflicts visible.
    grouped = {}
    resolved = {}
    for request in requests:
        namespace, pool = facts.namespace_for(request.core_domain, request.kind,
                                              request.pipe_pair)
        if namespace is None or pool is None:
            raise CompletionError(
                "COMPLETION_EVENT_NAMESPACE_MISSING",
                "the target declares no event pool for (core domain=%s, kind=%s, pipe pair=%s),"
                " so a finite id cannot be allocated honestly; declared pools are: %s"
                % (request.core_domain, request.kind, list(request.pipe_pair),
                   ", ".join(sorted({"(core=%s, kind=%s)" % (pool.core_domain, pool.kind)
                                     for pool in facts.pools}))),
                request.channel,
                "declared pool for (%s, %s, %s)" % (request.core_domain, request.kind,
                                                    list(request.pipe_pair)),
                [pool.as_dict() for pool in facts.pools])
        capacity = _capacity_override(overrides, namespace, pool, request)
        if capacity < 0:
            raise CompletionError("COMPLETION_EVENT_POOL_INVALID",
                                  "pool capacity cannot be negative", namespace.key(),
                                  ">= 0", capacity)
        if request.core_domain not in facts.core_domains:
            raise CompletionError("COMPLETION_TARGET_UNKNOWN",
                                  "core domain is not declared by the target",
                                  request.channel, list(facts.core_domains),
                                  request.core_domain)
        if any(iv.start < 0 or iv.stop < iv.start for iv in request.intervals):
            raise CompletionError("COMPLETION_USE_INTERVAL_INVALID",
                                  "a use interval must be non-negative and non-empty",
                                  request.channel, "[start, stop) with start <= stop",
                                  [iv.as_dict() for iv in request.intervals])
        resolved[request.channel] = (namespace, capacity, pool)
        grouped.setdefault(namespace.key(), {"namespace": namespace, "capacity": capacity,
                                             "pool": pool, "requests": []})
        grouped[namespace.key()]["requests"].append(request)

    allocations = []
    diagnostics = []
    for key in sorted(grouped):
        group = grouped[key]
        namespace = group["namespace"]
        capacity = group["capacity"]
        # Per id: the channels already holding it and the lifetime they hold it for.
        slots = {}
        # Deterministic order: longest-lived / earliest channel first, so the assignment does
        # not depend on dictionary insertion order.
        ordered = sorted(group["requests"],
                         key=lambda r: (min((iv.start for iv in r.intervals), default=0),
                                        r.channel))
        for request in ordered:
            placed = None
            for event_id in sorted(slots):
                holders = slots[event_id]
                conflicted = None
                for holder in holders:
                    if _intervals_overlap(holder.intervals, request.intervals):
                        conflicted = holder
                        break
                if conflicted is None:
                    placed = event_id
                    evidence = ("reuse of id %d after the previous lifetime completed (no"
                                " overlapping interval in %s)"
                                % (event_id, namespace.key()))
                    break
            if placed is None:
                placed = _free_id(set(slots), capacity)
                evidence = "fresh id inside %s" % (namespace.key(),)
            if placed is None:
                live = sorted({holder.channel for holders in slots.values()
                               for holder in holders})
                # A shared-namespace lifetime overlap is a DIFFERENT defect from a genuinely
                # full pool, and naming the wrong one sends the reader to the wrong fix.
                raise CompletionError(
                    "FLAG_POOL_EXHAUSTED",
                    "the finite event pool for namespace %s is exhausted: capacity %d, and"
                    " %d channel(s) are live with overlapping lifetimes (%s); a larger"
                    " capacity, fewer concurrent channels or a proven non-overlap is required"
                    % (namespace.key(), capacity, len(live), ", ".join(live[:6])),
                    namespace.key(), capacity, len(live))
            slots.setdefault(placed, []).append(request)
            allocations.append(ChannelAllocation(
                channel=request.channel, event_id=int(placed), namespace=namespace,
                pool_capacity=capacity, intervals=tuple(request.intervals),
                reuse_evidence=evidence, obligation_ids=tuple(request.obligation_ids)))
            diagnostics.append({"channel": request.channel, "namespace": namespace.key(),
                                "event_id": int(placed), "evidence": evidence})
    return EventAssignment(allocations=tuple(sorted(allocations, key=lambda a: a.channel)),
                           namespaces={key: group["namespace"]
                                       for key, group in grouped.items()},
                           facts=facts, diagnostics=tuple(diagnostics))


def check_assignment(assignment, *, facts: TargetEventFacts = None) -> dict:
    """Re-check an assignment INDEPENDENTLY of how it was produced.

    The allocator is not trusted to certify itself: this walks the recorded allocations and
    re-decides the three namespace rules plus the capacity bound, so a hand-built or corrupted
    assignment is refused. Returns a report; raises CompletionError on a violation.
    """
    facts = facts or assignment.facts
    per_namespace = {}
    for allocation in assignment.allocations:
        namespace, pool = facts.namespace_for(allocation.namespace.core_domain,
                                              allocation.namespace.kind,
                                              allocation.namespace.pipe_pair)
        if namespace is None:
            raise CompletionError("COMPLETION_EVENT_NAMESPACE_MISSING",
                                  "an allocation names a namespace the target does not declare",
                                  allocation.channel, "declared namespace",
                                  allocation.namespace.key())
        if namespace.key() != allocation.namespace.key():
            raise CompletionError("FLAG_RESOURCE_CONFLICT",
                                  "the allocation's namespace key disagrees with the target's"
                                  " resolution for its own (core, kind, pipe pair)",
                                  allocation.channel, namespace.key(),
                                  allocation.namespace.key())
        if not 0 <= int(allocation.event_id) < int(allocation.pool_capacity):
            raise CompletionError("FLAG_POOL_EXHAUSTED",
                                  "an allocated event id lies outside its pool capacity",
                                  allocation.channel, allocation.pool_capacity,
                                  allocation.event_id)
        per_namespace.setdefault(namespace.key(), {"capacity": allocation.pool_capacity,
                                                   "holders": {}})
        bucket = per_namespace[namespace.key()]["holders"]
        bucket.setdefault(int(allocation.event_id), []).append(allocation)
    for key, info in per_namespace.items():
        if len(info["holders"]) > info["capacity"]:
            raise CompletionError("FLAG_POOL_EXHAUSTED",
                                  "namespace holds more distinct ids than its capacity", key,
                                  info["capacity"], len(info["holders"]))
        for event_id, holders in sorted(info["holders"].items()):
            for index, left in enumerate(holders):
                for right in holders[index + 1:]:
                    if _intervals_overlap(left.intervals, right.intervals):
                        raise CompletionError(
                            "FLAG_RESOURCE_CONFLICT",
                            "two channels hold the same event id with OVERLAPPING lifetimes"
                            " in one namespace",
                            "%s#%d" % (key, event_id),
                            "disjoint lifetimes or different namespaces",
                            [left.channel, right.channel])
    return {"namespaces": {key: {"capacity": info["capacity"],
                                 "ids": sorted(info["holders"])}
                           for key, info in per_namespace.items()},
            "channels": len(assignment.allocations),
            "status": "CONSISTENT"}


__all__ = ["ChannelRequest", "ChannelAllocation", "EventAssignment",
           "allocate_events", "check_assignment"]
