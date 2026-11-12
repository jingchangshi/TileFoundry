"""Derive completion obligations from the REAL resolved plan.

WHAT "DERIVED" MEANS HERE. An obligation exists because the plan contains a real write
access and a real read access to the same buffer root, at real tasks with real pipes and
real version counts. Nothing is named after a program family and nothing is counted:

  * the obligation SET changes when a buffer is added, removed or gains a consumer;
  * it changes when the version count changes, because a WAR edge's distance is the version
    count (the ring's wrap distance);
  * it changes when a task moves to another pipe or core, because that decides whether the
    edge can be discharged by issue order or needs a real event;
  * a plan with no accesses produces no obligations at all, and trip=0 produces none either.

The four derivable obligation kinds, each with the rule that creates it:

  ready   a writer task W and a reader task R on one root. R may not read before W's write
          completed.
  free    a writer W and EVERY reader of the root. W may not overwrite generation k until
          generation k-versions has been read by ALL of its consumers. The consumer set is
          carried on the obligation, so a FREE that discharges after only the first consumer
          is a different (wrong) object, not a formatting difference.
  state   two writers on one root at the same generation (WAW). The later write must not
          land before the earlier one completed, and the next setter must not overtake the
          reader that observes it.
  exit    an asynchronously written root whose last generation is still live when the
          region exits (no in-region reader, or the ring is not drained by the trip count),
          plus the region's final drain when ANY asynchronous access exists.

MECHANISM CHOICE IS PER EDGE, not per program. A same-core edge whose producer retires with
its memory effect visible can be discharged by ISSUE order; every other edge needs an EVENT.
A single "same_pipe" switch that made the whole compiler skip synchronisation is exactly the
defect this replaces, so the decision is recorded per obligation with the fact it used.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .model import (
    AIV,
    AIC,
    DEVICE,
    KIND_EXIT,
    KIND_FREE,
    KIND_READY,
    KIND_STATE,
    CompletionError,
    Obligation,
    PIPE_FACTS,
    core_for_pipe,
    is_async_pipe,
    digest,
)

MECH_EVENT = "EVENT"
MECH_ISSUE_ORDER = "ISSUE_ORDER"
MECH_DRAIN = "DRAIN"


@dataclass(frozen=True)
class Relation:
    """One derived producer->consumer completion relation on one buffer root."""

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
    producer_access: str
    consumer_access: str
    consumers: tuple = ()
    guard: object = None
    evidence: str = ""

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class ObligationSet:
    """What the plan's own accesses require, with the mechanism chosen per obligation."""

    obligations: tuple
    relations: tuple
    discharges: tuple
    cross_handoffs: tuple
    exit_roots: tuple
    trip: int
    versions: dict
    cores: dict
    facts_digest: str = ""
    diagnostics: tuple = ()

    @property
    def by_kind(self) -> dict:
        counts = {}
        for obligation in self.obligations:
            counts[obligation.kind] = counts.get(obligation.kind, 0) + 1
        return counts

    def as_dict(self):
        return {"trip": self.trip,
                "obligations": [o.as_dict() for o in self.obligations],
                "relations": [r.as_dict() for r in self.relations],
                "discharges": list(self.discharges),
                "cross_handoffs": list(self.cross_handoffs),
                "exit_roots": list(self.exit_roots),
                "by_kind": self.by_kind,
                "versions": dict(self.versions),
                "cores": dict(self.cores),
                "diagnostics": list(self.diagnostics)}


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _trip(plan) -> int:
    domain = _get(plan.loop, "domain", {}) or {}
    start = int(domain["start"])
    stop = int(domain["stop"])
    step = int(domain["step"])
    if step <= 0:
        raise CompletionError("COMPLETION_LOOP_DOMAIN_INVALID", "step must be positive",
                              str(_get(plan.loop, "source_loop_id", "loop")), ">0", step)
    return max(0, (stop - start + step - 1) // step)


class _AccessIndex:
    """Root -> writers/readers, task -> accesses. Built once from the plan's own list."""

    def __init__(self, plan):
        self.root_writers = {}
        self.root_readers = {}
        self.task_order = {}
        self.task_pipe = {}
        self.task_guard = {}
        self.versions = {}
        self.root_async_write = {}
        self.root_async_read = {}
        self.root_space = {}
        self.writer_access = {}
        self.reader_accesses = {}
        for index, task in enumerate(_get(plan, "tasks", ()) or ()):
            task_id = str(_get(task, "task_id"))
            self.task_order[task_id] = int(_get(task, "order", index) or index)
            self.task_pipe[task_id] = str(_get(task, "pipe", "") or "")
            guard = _get(task, "guard", None)
            self.task_guard[task_id] = guard
        for buffer in _get(plan, "buffers", ()) or ():
            root = str(_get(buffer, "root"))
            self.versions[root] = int(_get(buffer, "slot_count", 1) or 1)
            self.root_space[root] = str(_get(buffer, "space", ""))
            self.root_writers.setdefault(root, [])
            self.root_readers.setdefault(root, [])
            self.writer_access.setdefault(root, {})
            self.reader_accesses.setdefault(root, {})
            self.root_async_write.setdefault(root, False)
            self.root_async_read.setdefault(root, False)
        for access in _get(plan, "accesses", ()) or ():
            root = str(_get(access, "buffer_root"))
            task = str(_get(access, "task_id"))
            if root not in self.versions:
                raise CompletionError(
                    "COMPLETION_SOURCE_COVERAGE_MISSING",
                    "access names a root the plan has no buffer for",
                    str(_get(access, "access_id")), "root in plan.buffers", root)
            mode = str(_get(access, "mode", "read")).lower()
            pipe = self.task_pipe.get(task, "")
            access_id = str(_get(access, "access_id"))
            if mode == "write":
                self.root_writers[root].append(task)
                self.writer_access[root].setdefault(task, []).append(access_id)
                if is_async_pipe(pipe):
                    self.root_async_write[root] = True
            elif mode == "read":
                self.root_readers[root].append(task)
                self.reader_accesses[root].setdefault(task, []).append(access_id)
                if is_async_pipe(pipe):
                    self.root_async_read[root] = True
            else:
                raise CompletionError("COMPLETION_SOURCE_COVERAGE_MISSING",
                                      "access mode is neither read nor write",
                                      access_id, ("read", "write"), mode)
        for root in list(self.root_writers):
            self.root_writers[root] = _ordered_unique(self.root_writers[root], self.task_order)
            self.root_readers[root] = _ordered_unique(self.root_readers[root], self.task_order)


def _ordered_unique(tasks, order):
    seen = []
    for task in tasks:
        if task not in seen:
            seen.append(task)
    return sorted(seen, key=lambda task: (order.get(task, 0), task))


def _mechanism(relation: Relation) -> tuple:
    """(mechanism, reason) for ONE edge. Never a program-wide switch.

    ISSUE order is used only when BOTH endpoints live in the same core domain and the
    producer retires with its memory effect visible. Everything else needs an event, because
    this target declares the pipe asynchronous: an issued DMA has not completed.
    """
    if relation.producer_core != relation.consumer_core:
        return MECH_EVENT, ("cross-core %s->%s: visibility needs a device-level event"
                            % (relation.producer_core, relation.consumer_core))
    fact = PIPE_FACTS.get(relation.producer_pipe)
    if fact is None:
        return MECH_EVENT, ("pipe %r has no declared completion fact, so issue order cannot"
                            " be claimed" % (relation.producer_pipe,))
    if not fact["visible_at_issue"]:
        return MECH_EVENT, ("pipe %s is asynchronous (declared %r): the issued operation has"
                            " not completed its memory effect"
                            % (relation.producer_pipe, fact["completion"]))
    return MECH_ISSUE_ORDER, ("pipe %s retires with its memory effect visible at issue and"
                              " both endpoints are in %s, so program order is the proof"
                              % (relation.producer_pipe, relation.producer_core))


def _obligation_id(kind, relation, ordinal) -> str:
    return "ob-%s-%s-%s-%s" % (
        kind, relation.buffer_root.split("/")[-1] or relation.buffer_root,
        digest(relation.as_dict()), ordinal)


def derive_relations(plan, *, facts=None) -> tuple:
    """Every completion relation the plan's own accesses require."""
    index = _AccessIndex(plan)
    relations = []
    for root in sorted(index.versions):
        versions = index.versions[root]
        writers = index.root_writers.get(root, [])
        readers = index.root_readers.get(root, [])
        writer_pipes = {task: index.task_pipe.get(task, "") for task in writers}
        reader_pipes = {task: index.task_pipe.get(task, "") for task in readers}
        reader_set = tuple(readers)
        # READY: every writer x every reader. distance 0 because the reader of generation k
        # consumes the write of generation k.
        for writer in writers:
            for reader in readers:
                relations.append(Relation(
                    kind=KIND_READY, buffer_root=root,
                    producer_task=writer, consumer_task=reader,
                    producer_pipe=writer_pipes[writer], consumer_pipe=reader_pipes[reader],
                    producer_core=core_for_pipe(writer_pipes[writer], facts),
                    consumer_core=core_for_pipe(reader_pipes[reader], facts),
                    versions=versions, distance=0,
                    producer_access=(index.writer_access[root].get(writer) or [""])[0],
                    consumer_access=(index.reader_accesses[root].get(reader) or [""])[0],
                    guard=index.task_guard.get(reader),
                    evidence="RAW on one root: write of generation k precedes read of k"))
        # FREE: the writer vs EVERY consumer. The distance is the ring wrap distance, so a
        # plan with more versions frees later (and needs fewer waits).
        for writer in writers:
            if not readers:
                continue
            relations.append(Relation(
                kind=KIND_FREE, buffer_root=root,
                producer_task=writer, consumer_task=reader_set[-1],
                producer_pipe=writer_pipes[writer],
                consumer_pipe=reader_pipes.get(reader_set[-1], ""),
                producer_core=core_for_pipe(writer_pipes[writer], facts),
                consumer_core=core_for_pipe(reader_pipes.get(reader_set[-1], ""), facts),
                versions=versions, distance=versions,
                producer_access=(index.writer_access[root].get(writer) or [""])[0],
                consumer_access=(index.reader_accesses[root].get(reader_set[-1]) or [""])[0],
                consumers=reader_set,
                guard=index.task_guard.get(writer),
                evidence=("WAR on one root: generation k is freed only after ALL %d consumer(s)"
                          " completed their read of k, and the writer overwrites k+%d"
                          % (len(reader_set), versions))))
        # STATE: two writers on one root at one generation; the later one must not land
        # before the earlier completed, and the reader must observe the earlier one.
        for first in writers:
            for second in writers:
                if first == second:
                    continue
                if index.task_order.get(first, 0) >= index.task_order.get(second, 0):
                    continue
                relations.append(Relation(
                    kind=KIND_STATE, buffer_root=root,
                    producer_task=first, consumer_task=second,
                    producer_pipe=writer_pipes[first], consumer_pipe=writer_pipes[second],
                    producer_core=core_for_pipe(writer_pipes[first], facts),
                    consumer_core=core_for_pipe(writer_pipes[second], facts),
                    versions=versions, distance=0,
                    producer_access=(index.writer_access[root].get(first) or [""])[0],
                    consumer_access=(index.writer_access[root].get(second) or [""])[0],
                    guard=index.task_guard.get(second),
                    evidence="WAW on one root: setter order must be preserved"))
    return tuple(relations)


def _exit_roots(plan, index, trip) -> tuple:
    """Roots whose asynchronously written last generation is live when the region exits.

    Derived, not listed: a write-only root written by an asynchronous pipe has no in-region
    consumer, so its completion is only observable at the region exit. A ring root whose
    trip count is not a multiple of its version count also leaves its final generations
    unconsumed by a following producer, so the readers of those generations are still
    outstanding at exit.
    """
    live = []
    for root in sorted(index.versions):
        if not index.root_async_write.get(root):
            continue
        readers = index.root_readers.get(root, [])
        versions = index.versions[root]
        if not readers:
            live.append(root)
            continue
        if trip == 0:
            continue
        # The last generations have no successor writer that would wait for them, so the
        # exit drain is what makes their async writes/reads complete.
        last = trip - 1
        if versions <= 0 or (last + 1) % versions != 0 or trip < versions:
            live.append(root)
    return tuple(live)


def derive_obligations(plan, *, facts=None) -> ObligationSet:
    """The full obligation set for one resolved plan."""
    trip = _trip(plan)
    index = _AccessIndex(plan)
    relations = derive_relations(plan, facts=facts)
    obligations = []
    discharges = []
    cross_handoffs = []
    exit_roots = _exit_roots(plan, index, trip)
    any_async = any(index.root_async_write.values()) or any(index.root_async_read.values())
    generations = tuple(range(trip))
    for ordinal, relation in enumerate(relations):
        mechanism, reason = _mechanism(relation)
        obligation = Obligation(
            obligation_id=_obligation_id(relation.kind, relation, ordinal),
            kind=relation.kind, buffer_root=relation.buffer_root,
            producer_task=relation.producer_task, consumer_task=relation.consumer_task,
            producer_pipe=relation.producer_pipe, consumer_pipe=relation.consumer_pipe,
            producer_core=relation.producer_core, consumer_core=relation.consumer_core,
            versions=relation.versions, distance=relation.distance,
            generation_domain=generations, consumers=relation.consumers,
            producer_access=relation.producer_access,
            consumer_access=relation.consumer_access,
            guard=relation.guard, ring=relation.versions > 1,
            evidence=relation.evidence)
        obligations.append(obligation)
        if mechanism == MECH_EVENT:
            pass
        else:
            discharges.append({"obligation_id": obligation.obligation_id,
                               "mechanism": mechanism, "reason": reason})
        if relation.producer_core != relation.consumer_core:
            cross_handoffs.append({
                "obligation_id": obligation.obligation_id,
                "role": relation.kind,
                "from_domain": relation.producer_core,
                "to_domain": relation.consumer_core,
                "buffer_root": relation.buffer_root,
                "producer_task": relation.producer_task,
                "consumer_task": relation.consumer_task,
                "participants": [relation.producer_core, relation.consumer_core],
                "visibility": "device-visible before the consumer's read",
                "guard": _guard_key(relation.guard),
            })
    # EXIT: one obligation per live root, plus the region's final drain whenever any
    # asynchronous access exists. Both depend on the program, never on a constant.
    for root in exit_roots:
        obligations.append(Obligation(
            obligation_id="ob-exit-%s-%s" % (root.split("/")[-1] or root,
                                             digest({"root": root, "trip": trip})),
            kind=KIND_EXIT, buffer_root=root,
            producer_task=(index.root_writers.get(root) or [""])[0],
            consumer_task="", producer_pipe=index.task_pipe.get(
                (index.root_writers.get(root) or [""])[0], ""),
            consumer_pipe="", producer_core=core_for_pipe(
                index.task_pipe.get((index.root_writers.get(root) or [""])[0], ""), facts),
            consumer_core=DEVICE, versions=index.versions[root], distance=0,
            generation_domain=generations, consumers=(),
            producer_access=(index.writer_access.get(root, {}).get(
                (index.root_writers.get(root) or [""])[0]) or [""])[0],
            consumer_access="", ring=index.versions[root] > 1,
            evidence=("region exit: this root is written asynchronously and its last"
                      " generation is unconsumed in-region, so its completion is only"
                      " observable at the exit barrier")))
    obligations.append(Obligation(
        obligation_id="ob-exit-final-drain-%s" % (digest({"trip": trip,
                                                          "async": any_async}),),
        kind=KIND_EXIT, buffer_root="", producer_task="", consumer_task="",
        producer_pipe="", consumer_pipe="", producer_core=DEVICE, consumer_core=DEVICE,
        versions=1, distance=0, generation_domain=(), consumers=(),
        producer_access="", consumer_access="", ring=False,
        evidence=("region exit drain: %s" % ("asynchronous work exists" if any_async
                                             else "no asynchronous work"))))
    if not any_async and trip == 0:
        # Nothing executes and nothing is asynchronous: the drain obligation is vacuous, but
        # keeping it makes the region-exit check uniform and it costs no event.
        pass
    return ObligationSet(
        obligations=tuple(obligations), relations=relations,
        discharges=tuple(discharges), cross_handoffs=tuple(cross_handoffs),
        exit_roots=exit_roots, trip=trip, versions=dict(index.versions),
        cores={task: core_for_pipe(pipe, facts) for task, pipe in index.task_pipe.items()},
        facts_digest=digest({"profile": getattr(facts, "profile_id", "ascend950pr_9579")}),
        diagnostics=tuple())


def _guard_key(guard):
    """A comparable identity for a guard expression (None means unconditional)."""
    if guard is None:
        return None
    if isinstance(guard, dict):
        return digest(guard)
    if hasattr(guard, "to_dict"):
        return digest(guard.to_dict())
    return digest(repr(guard))


def obligations_from_edges(edges, plan, *, facts=None) -> tuple:
    """Turn an explicit dependency-edge list into relations (the L5/source-driven route).

    The plan-driven route above infers edges from real accesses. This route is used when a
    dependency source already exists (the ledger's typed RAW/WAR/WAW/REUSE edges) and keeps
    the same shape, so both routes produce obligations the same verifier can judge. An edge
    whose endpoints are not in the plan is refused rather than skipped.
    """
    index = _AccessIndex(plan)
    relations = []
    for ordinal, edge in enumerate(edges):
        kind_in = str(_get(edge, "kind", "RAW")).upper()
        producer = str(_get(edge, "producer"))
        consumer = str(_get(edge, "consumer"))
        root = str(_get(edge, "shared_root", _get(edge, "root")) or "")
        if root not in index.versions:
            raise CompletionError("COMPLETION_SOURCE_COVERAGE_MISSING",
                                  "edge root is not a plan buffer", root, "plan root", root)
        if producer not in index.task_pipe or consumer not in index.task_pipe:
            raise CompletionError("COMPLETION_SOURCE_COVERAGE_MISSING",
                                  "edge endpoint is not a plan task",
                                  "%s->%s" % (producer, consumer),
                                  sorted(index.task_pipe), [producer, consumer])
        mapped = {"RAW": KIND_READY, "WAR": KIND_FREE, "WAW": KIND_STATE,
                  "REUSE": KIND_STATE}.get(kind_in)
        if mapped is None:
            raise CompletionError("COMPLETION_SOURCE_COVERAGE_MISSING",
                                  "edge kind has no completion mapping", kind_in,
                                  sorted(["RAW", "WAR", "WAW", "REUSE"]), kind_in)
        versions = index.versions[root]
        readers = index.root_readers.get(root, [])
        relations.append(Relation(
            kind=mapped, buffer_root=root, producer_task=producer, consumer_task=consumer,
            producer_pipe=index.task_pipe.get(producer, ""),
            consumer_pipe=index.task_pipe.get(consumer, ""),
            producer_core=core_for_pipe(index.task_pipe.get(producer, ""), facts),
            consumer_core=core_for_pipe(index.task_pipe.get(consumer, ""), facts),
            versions=versions,
            distance=int(_get(edge, "distance", 0) or 0),
            producer_access=(index.writer_access.get(root, {}).get(producer)
                             or index.reader_accesses.get(root, {}).get(producer)
                             or [""])[0],
            consumer_access=(index.reader_accesses.get(root, {}).get(consumer)
                             or index.writer_access.get(root, {}).get(consumer)
                             or [""])[0],
            consumers=(tuple(readers) if mapped == KIND_FREE else ()),
            guard=_get(edge, "guard")))
    return tuple(relations)


__all__ = [
    "MECH_EVENT", "MECH_ISSUE_ORDER", "MECH_DRAIN", "Relation", "ObligationSet",
    "derive_relations", "derive_obligations", "obligations_from_edges",
]
