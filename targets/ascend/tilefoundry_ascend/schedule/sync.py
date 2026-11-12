"""Completion obligations -> logical channels -> physical Ascend flags.

ready/free obligations are derived from the Problem dependencies and the ring
buffer versions. A logical channel is NOT a physical flag id: physical flags are
allocated per channel from a bounded pool, and slot generation is the *logical
iteration ordinal* t (not the recompute tile index).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from ..completion.model import CompletionError


@dataclass(frozen=True)
class Obligation:
    channel: str
    producer: str          # task id
    producer_pipe: str
    consumer: str
    consumer_pipe: str
    event_domain: str      # "x_slot" | "y_slot" | "once"
    versions: int
    distance: int
    meaning: str

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SyncAction:
    channel: str
    logical_channel: str
    physical_flag: int
    producer_pipe: str
    consumer_pipe: str
    direction: str         # "set_wait"

    def as_dict(self):
        return asdict(self)


def build_obligations(problem) -> list:
    """Derive completion obligations from the Problem's REAL accesses and dependencies.

    WHAT THIS REPLACED AND WHY. This function used to return a fixed list of six channels --
    x_ready, x_free, y_ready, y_free, w_ready, w_conv -- with their pipes, versions and
    distances written into the source. That set is RMSNorm's, not the program's: a different
    kernel silently got RMSNorm's channels, and a changed consumer set (the case the ring
    protocol cares about most) was invisible because the FREE channel named no consumers at
    all. The obligations are now a FUNCTION of the problem:

      * a READY edge for every (writer, reader) pair sharing a root, taken from the real
        accesses and typed by the real RAW dependency;
      * a FREE edge per (root, producer) naming ALL readers of that root, so the "wait for the
        last consumer" rule has something to be checked against;
      * a WAW edge where two tasks write the same root (the reuse ordering).

    The versions come from the buffer declaration and the distance from the dependency's
    iteration distance, so a V-byte ring is not assumed anywhere.
    """
    from ..completion.obligations import obligations_from_edges

    relations = obligations_from_edges(_edges_of(problem), _plan_of(problem))
    return [
        Obligation(
            channel=_channel_of(relation),
            producer=relation.producer_task,
            producer_pipe=relation.producer_pipe,
            consumer=relation.consumer_task,
            consumer_pipe=relation.consumer_pipe,
            event_domain=_domain_of(relation),
            versions=int(relation.versions),
            distance=int(relation.distance),
            meaning=relation.evidence)
        for relation in relations
    ]


def _channel_of(relation) -> str:
    """A stable logical channel name for one derived relation.

    Derived from the relation's own content rather than from a positional index, so the name of
    a channel does not change when an unrelated edge is added.
    """
    return "%s_%s_%s_%s" % (
        relation.kind,
        _root_short(relation.buffer_root),
        relation.producer_task.replace("/", "_"),
        relation.consumer_task.replace("/", "_"))


def _domain_of(relation) -> str:
    """The event domain a channel lives in, derived from what it actually orders."""
    if relation.kind in ("ready", "free"):
        return "%s_slot" % _root_short(relation.buffer_root)
    return "once"


def _root_short(root) -> str:
    text = str(root)
    return text.split("/")[-1].replace("_ub", "").replace("_", "") or "buf"


def _edges_of(problem) -> tuple:
    """The Problem's typed dependencies, as edges the obligation derivation can read.

    The schedule-level Dependency records (producer, consumer, distance, kind, evidence) and
    NOT the shared root -- the root is what the ledger layer resolves when it builds the window
    intersection. It is recovered here from the REAL accesses of the two endpoints, and an edge
    whose endpoints share NO root is refused rather than assigned one: guessing a root would
    put the obligation on the wrong buffer, which is worse than reporting the edge.
    """
    accesses = {}
    for access in getattr(problem, "accesses", ()) or ():
        accesses.setdefault(str(getattr(access, "task", "")), []).append(
            str(getattr(access, "root", "")))
    edges = []
    for edge in getattr(problem, "dependencies", ()) or ():
        producer = str(getattr(edge, "producer", ""))
        consumer = str(getattr(edge, "consumer", ""))
        shared = [root for root in accesses.get(producer, ())
                  if root and root in accesses.get(consumer, ())]
        if not shared:
            raise CompletionError(
                "COMPLETION_SOURCE_COVERAGE_MISSING",
                "a dependency edge's endpoints share no buffer, so the completion obligation it"
                " implies cannot be placed on a root",
                "%s->%s" % (producer, consumer),
                "a shared root among the endpoints' accesses",
                [accesses.get(producer, []), accesses.get(consumer, [])])
        edges.append(_Edge(producer=producer, consumer=consumer,
                           kind=str(getattr(edge, "kind", "RAW")),
                           distance=int(getattr(edge, "distance", 0) or 0),
                           shared_root=shared[0]))
    return tuple(edges)


@dataclass(frozen=True)
class _Edge:
    """One dependency edge with the root it is placed on."""

    producer: str
    consumer: str
    kind: str
    distance: int
    shared_root: str;


def _plan_of(problem):
    """A minimal plan VIEW over the Problem so obligations can be derived from IT.

    The schedule-level Problem carries the real accesses, buffers and loop domain but not the
    resolved plan's task objects. Rather than fabricating a ResolvedPlan, this adapter exposes
    exactly the three things the derivation reads -- accesses, per-task pipes and versions --
    so the SAME derivation code serves both layers and the two cannot drift.
    """
    from ..completion.sync_adapter import problem_plan_view

    return problem_plan_view(problem)


def allocate_flags(obligations) -> dict:
    """Assign a physical flag id per logical channel, inside the real event namespaces.

    WHAT THIS REPLACED AND WHY. This used to hand out 0, 1, 2, ... across a single chip-wide
    counter. That is not an allocation: on this target the id spaces are per (core domain,
    event kind, pipe pair), so a chip-wide counter (a) treats two genuinely independent
    namespaces as if they competed, and (b) cannot express that two channels in the SAME
    namespace may legally share an id once the earlier lifetime completed. The real allocator
    does both; this function is now a thin adapter that keeps the old return shape.

    Returns {channel: event_id}. A capacity refusal raises with the namespace named, which is
    the located diagnostic the spec requires.
    """
    from ..completion.events import ChannelRequest, allocate_events
    from ..completion.model import event_facts_for

    requests = []
    for obligation in obligations:
        requests.append(ChannelRequest(
            channel=obligation.channel,
            core_domain=_core_of(obligation.producer_pipe, event_facts_for(None)),
            kind=("free" if "free" in obligation.channel or obligation.distance else "ready"),
            pipe_pair=(obligation.producer_pipe, obligation.consumer_pipe),
            intervals=(_interval_of(obligation),),
            obligation_ids=(obligation.channel,)))
    assignment = allocate_events(requests, facts=event_facts_for(None))
    return {allocation.channel: allocation.event_id
            for allocation in assignment.allocations}


def _core_of(pipe, facts):
    from ..completion.model import core_for_pipe

    try:
        return core_for_pipe(pipe)
    except Exception:  # noqa: BLE001 - an unknown pipe is charged to the device namespace
        return facts.core_domains[0]


def _interval_of(obligation):
    """The lifetime one logical channel occupies its id for.

    The distance is the ring depth: a channel ordering generation t against t+d is live across
    those generations, so its interval is [0, d + 1). Two channels whose intervals do not
    overlap may legally share an id, and the allocator will do so.
    """
    from ..completion.model import UseInterval

    return UseInterval(start=0, stop=max(1, int(obligation.distance) + 1), generation=0)


def build_plan(problem, solve_result) -> dict:
    obls = build_obligations(problem)
    flags = allocate_flags(obls)
    actions = [SyncAction(ob.channel, ob.channel, flags[ob.channel], ob.producer_pipe,
                          ob.consumer_pipe, "set_wait") for ob in obls]
    Vx = problem.policy["x_versions"]; Vy = problem.policy["y_versions"]
    buffers = {}
    for b in problem.buffers:
        if b.root in ("x_ub", "y_ub"):
            V = b.versions
            buffers[b.root] = {"root": b.root, "space": b.space, "versions": V,
                               "owner_loop": b.owner_loop, "slot_rule": f"t % {V}",
                               "generation_rule": "t (logical iteration ordinal)",
                               "bytes": b.bytes_}
        else:
            buffers[b.root] = {"root": b.root, "space": b.space, "versions": 1,
                               "owner_loop": b.owner_loop, "slot_rule": "static",
                               "generation_rule": "-", "bytes": b.bytes_}
    return {
        "status": solve_result.status,
        "ii": solve_result.ii,
        "stages": dict(solve_result.stages),
        "model_version": solve_result.model_version,
        "solver_version": solve_result.solver_version,
        "seed": solve_result.seed,
        "buffers": buffers,
        "obligations": [o.as_dict() for o in obls],
        "sync_actions": [a.as_dict() for a in actions],
        "flag_pool": flags,
        "diagnostics": list(solve_result.diagnostics),
    }


__all__ = ["Obligation", "SyncAction", "build_obligations", "allocate_flags", "build_plan"]
