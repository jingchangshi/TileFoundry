"""Build the instance-ordered completion program for a resolved plan.

WHAT AN "INSTANCE" IS. The plan's loop has a static trip count N and every task has a
physical issue rank and a stage shift. The completion program therefore enumerates
(sweep, task) instances and places completion events INSIDE them:

    for sweep p in [0, N + depth):
      for task u in physical issue order:
        k = p - shift[u]
        if 0 <= k < N:
          <u's completion events for generation k, in the order below>

That is the same instance domain spec 3.2/3.9 requires of the materialized TIR, so a Set and
its Wait land on REAL statements of the REAL program rather than on a separate trace nobody
emits.

THE EVENT ORDER INSIDE ONE INSTANCE, and the reason for each position:

  free_wait    BEFORE the write: the slot may not be overwritten until the previous
               generation's consumers finished reading it.
  state_wait   BEFORE the write: the earlier writer's generation must be visible first.
  write_begin  the real write.
  ready_set    AFTER the write: the write's completion is what the consumer may rely on.
  ready_wait   BEFORE the read.
  read_begin / read_end
  free_set     only on the LAST consumer of the generation, so the FREE names the last read
               completion rather than an arbitrary one.

INITIAL EMPTY SLOT. Generation g can be overwritten only if a generation g-versions exists,
so the FREE wait is emitted for g >= versions and NEVER for the first 'versions' writes. There
is no wait for a consumer that never ran, which is the protocol the spec demands; the
alternative (a hardware init token) is a different, also-legal mechanism and is not invented
here.

SHORT LOOPS. trip=0 produces no instances at all (only the exit drain); trip=1 produces the
first generation only, so no FREE wait exists; trip < versions produces no FREE wait at all;
trip exactly == versions produces exactly one FREE wait. All four are enumerated as tests.

GUARDS. A task's authored guard is carried onto every completion record it owns, so a Set and
its Wait can be compared for implication (guards.py) instead of being counted. A guard is
never dropped to make the counts line up.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .model import (
    DEVICE,
    KIND_CROSS,
    KIND_EXIT,
    KIND_FREE,
    KIND_READY,
    KIND_STATE,
    CompletionError,
    digest,
)
from .guards import GUARD_ALWAYS, normalize_guard

#: Roles a trace record can have. The access roles are RETAINED because a completion checker
#: that only sees flags cannot tell an early FREE from a correct one: it needs the real
#: read/write points the flags are supposed to order.
ROLE_WRITE_BEGIN = "write_begin"
ROLE_READ_BEGIN = "read_begin"
ROLE_READ_END = "read_end"
ROLE_SET = "set"
ROLE_WAIT = "wait"
ROLE_DRAIN = "drain"
ACCESS_ROLES = (ROLE_WRITE_BEGIN, ROLE_READ_BEGIN, ROLE_READ_END)

#: Refuse to statically enumerate more instances than this. The materialized TIR has the same
#: kind of bound (plan.materialize.MAX_STATIC_TRIP); a symbolic emission is a later increment
#: and is named rather than faked.
MAX_COMPLETION_INSTANCES = 4096


@dataclass(frozen=True)
class TraceRecord:
    """One statement of the completion program, in execution order."""

    index: int
    role: str
    channel: str
    obligation_id: str
    kind: str
    generation: int
    slot: int
    versions: int
    task: str
    core: str
    pipe: str
    producer_pipe: str
    consumer_pipe: str
    guard: object = None
    event_id: int = -1
    cross_core: bool = False
    sweep: int = 0
    issue_order: int = 0

    def with_event(self, event_id) -> "TraceRecord":
        return TraceRecord(**{**self.__dict__, "event_id": int(event_id)})

    def as_dict(self):
        return {"index": self.index, "role": self.role, "channel": self.channel,
                "obligation_id": self.obligation_id, "kind": self.kind,
                "generation": self.generation, "slot": self.slot,
                "versions": self.versions, "task": self.task, "core": self.core,
                "pipe": self.pipe, "producer_pipe": self.producer_pipe,
                "consumer_pipe": self.consumer_pipe,
                "guard": normalize_guard(self.guard).as_dict(),
                "event_id": self.event_id, "cross_core": self.cross_core,
                "sweep": self.sweep, "issue_order": self.issue_order,
                "is_access": self.role in ACCESS_ROLES}


@dataclass(frozen=True)
class CompletionTrace:
    """The instance-ordered completion program, plus the facts a checker needs."""

    records: tuple
    obligations: tuple
    trip: int
    depth: int
    channels: tuple
    cross_pairs: tuple
    exit_records: tuple
    versions: dict
    cores: dict
    diagnostics: tuple = ()

    def by_channel(self) -> dict:
        out = {}
        for record in self.records:
            out.setdefault(record.channel, []).append(record)
        return out

    def sets(self):
        return tuple(r for r in self.records if r.role == ROLE_SET)

    def waits(self):
        return tuple(r for r in self.records if r.role == ROLE_WAIT)

    def drains(self):
        return tuple(r for r in self.records if r.role == ROLE_DRAIN)

    def channel_requests(self):
        """One ChannelRequest per (channel, slot): the ring's OWN reuse granularity.

        A slot is revisited every 'versions' generations, and the ring protocol's free/ready
        sequence makes those visits non-overlapping. Making that explicit is what lets the
        allocator reuse a finite id LEGALLY, and it is also what makes an overlapping lifetime
        a detectable defect instead of an invisible one.
        """
        from .events import ChannelRequest

        _obligation_by_id = {obligation.obligation_id: obligation
                            for obligation in self.obligations}
        grouped = {}
        for record in self.records:
            if record.role not in (ROLE_SET, ROLE_WAIT):
                continue
            key = (record.channel, record.slot)
            grouped.setdefault(key, {"records": [], "obligations": set(),
                                     "obligations_resolved": [], "core": record.core,
                                     "kind": record.kind, "pipes": set(),
                                     "cross": record.cross_core})
            grouped[key]["records"].append(record)
            if record.obligation_id not in grouped[key]["obligations"]:
                grouped[key]["obligations"].add(record.obligation_id)
                obligation = _obligation_by_id.get(record.obligation_id)
                if obligation is not None:
                    grouped[key]["obligations_resolved"].append(obligation)
            grouped[key]["pipes"].add((record.producer_pipe, record.consumer_pipe))
        requests = []
        for (channel, slot) in sorted(grouped):
            info = grouped[(channel, slot)]
            records = sorted(info["records"], key=lambda r: r.index)
            pipes = sorted(info["pipes"])
            pipe_pair = pipes[0] if len(pipes) == 1 else tuple(sorted(
                {pipe for pair in pipes for pipe in pair}))
            intervals = _slot_intervals(records, channel, slot)
            # A cross-core handoff is a DEVICE-level resource whose participants are the two
            # domains: it must NOT be charged to either side's local pool, or one side could
            # exhaust a pool it never uses.
            if info["cross"]:
                core_domain, kind = DEVICE, KIND_CROSS
                # Pparticipants are the two CORE DOMAINS of the handoff, taken from the
                # obligations that produced the channel. They are what a device-level event
                # unblocks, and they are recorded so a checker can require both sides.
                participants = tuple(sorted(
                    {domain for obligation in info["obligations_resolved"]
                     for domain in (obligation.producer_core, obligation.consumer_core)
                     if domain and domain != DEVICE})) or (DEVICE,)
            else:
                core_domain, kind = info["core"], info["kind"]
                participants = (info["core"],)
            requests.append(ChannelRequest(
                channel="%s#s%d" % (channel, slot),
                core_domain=core_domain, kind=kind,
                pipe_pair=tuple(pipe_pair) if isinstance(pipe_pair, tuple) else pipe_pair,
                intervals=intervals,
                obligation_ids=tuple(sorted(info["obligations"])),
                participants=participants))
        return tuple(requests)

    def as_dict(self):
        return {"trip": self.trip, "depth": self.depth,
                "records": [r.as_dict() for r in self.records],
                "channels": list(self.channels),
                "cross_pairs": [dict(pair) for pair in self.cross_pairs],
                "exit_records": [r.as_dict() for r in self.exit_records],
                "diagnostics": list(self.diagnostics)}


def _slot_intervals(records, channel, slot):
    """Non-overlapping per-generation intervals for one (channel, slot).

    Each generation contributes [first set, last event+1); generations of the same slot are
    separated by 'versions' rings, so they must not overlap. An overlap is a REAL schedule
    defect (the ring slot is reused before its previous generation completed) and is refused
    here rather than being averaged away.
    """
    from .model import UseInterval

    per_generation = {}
    for record in records:
        entry = per_generation.setdefault(record.generation, [record.index, record.index])
        entry[0] = min(entry[0], record.index)
        entry[1] = max(entry[1], record.index)
    intervals = []
    for generation in sorted(per_generation):
        start, last = per_generation[generation]
        intervals.append(UseInterval(start=start, stop=last + 1, generation=generation))
    for index, left in enumerate(intervals):
        for right in intervals[index + 1:]:
            if left.overlaps(right):
                raise CompletionError(
                    "FLAG_RESOURCE_CONFLICT",
                    "the same ring slot is used by two overlapping generations, so one finite"
                    " event id cannot order both",
                    "%s#s%d" % (channel, slot),
                    "disjoint generation lifetimes",
                    [left.as_dict(), right.as_dict()])
    return tuple(intervals)


def _task_index(plan):
    index = {}
    for ordinal, task in enumerate(plan.tasks):
        task_id = str(task.task_id)
        index[task_id] = {
            "order": int(getattr(task, "order", ordinal) or 0),
            "shift": int(getattr(task, "iteration_shift", 0) or 0),
            "pipe": str(getattr(task, "pipe", "") or ""),
            "guard": getattr(task, "guard", None),
        }
    return index


def _channel_name(obligation) -> str:
    """A deterministic channel identity: kind + root + the real producer/consumer pair.

    The name is derived, so adding a consumer renames (and re-derives) the FREE channel
    instead of silently reusing a stale one.
    """
    root = obligation.buffer_root.split("/")[-1] or obligation.buffer_root or "region"
    producer = obligation.producer_task.split("/")[-1] or obligation.producer_task or "none"
    consumer = obligation.consumer_task.split("/")[-1] or obligation.consumer_task or "none"
    return "%s.%s.%s.%s" % (obligation.kind, root, producer, consumer)


def build_completion_trace(plan, obligation_set, *, facts=None) -> CompletionTrace:
    """Enumerate the completion program in real execution order."""
    trip = int(obligation_set.trip)
    tasks = _task_index(plan)
    depth = max([info["shift"] for info in tasks.values()] or [0]) + 1
    instances = trip * max(1, len(tasks))
    if instances > MAX_COMPLETION_INSTANCES:
        raise CompletionError(
            "COMPLETION_STATIC_ENUM_LIMIT",
            "the completion program is enumerated statically and this plan exceeds the"
            " declared instance budget; a symbolic emission is a later increment",
            str(trip), "<= %d instances" % MAX_COMPLETION_INSTANCES, instances)
    obligations = tuple(obligation_set.obligations)
    # Index the obligations by the task instance each one acts on.
    ready_by_consumer = {}
    ready_by_producer = {}
    free_writers = {}
    state_by_producer = {}
    state_by_consumer = {}
    last_consumer = {}
    for obligation in obligations:
        channel = _channel_name(obligation)
        if obligation.kind == KIND_READY:
            ready_by_consumer.setdefault(obligation.consumer_task, []).append((channel, obligation))
            ready_by_producer.setdefault(obligation.producer_task, []).append((channel, obligation))
        elif obligation.kind == KIND_FREE:
            free_writers.setdefault(obligation.producer_task, []).append((channel, obligation))
            for ordinal, consumer in enumerate(obligation.consumers):
                last_consumer.setdefault((obligation.buffer_root,
                                          obligation.producer_task), {})[consumer] =                     (ordinal == len(obligation.consumers) - 1, channel, obligation)
        elif obligation.kind == KIND_STATE:
            state_by_producer.setdefault(obligation.producer_task, []).append((channel, obligation))
            state_by_consumer.setdefault(obligation.consumer_task, []).append((channel, obligation))

    writers_by_root = {}
    readers_by_root = {}
    for access in plan.accesses:
        root = str(access.buffer_root)
        task = str(access.task_id)
        mode = str(access.mode).lower()
        bucket = writers_by_root if mode == "write" else readers_by_root
        bucket.setdefault(root, {})
        bucket[root].setdefault(task, []).append(str(access.access_id))

    records = []
    counter = {"n": 0}

    def emit(role, channel, obligation, generation, slot, task, **extra):
        info = tasks.get(task, {})
        # The record's CORE is the domain the statement executes in: a Set runs where the
        # producer runs, a Wait where the consumer runs. A cross-core handoff belongs to
        # NEITHER side's local namespace, so it is attributed to the device-level resource.
        if obligation.cross_core:
            core = DEVICE
        elif role == ROLE_SET:
            core = obligation.producer_core
        else:
            core = obligation.consumer_core
        records.append(TraceRecord(
            index=counter["n"], role=role, channel=channel,
            obligation_id=obligation.obligation_id, kind=obligation.kind,
            generation=int(generation), slot=int(slot),
            versions=int(obligation.versions), task=task,
            core=str(extra.pop("core", "") or core), pipe=str(info.get("pipe", "")),
            producer_pipe=obligation.producer_pipe,
            consumer_pipe=obligation.consumer_pipe,
            guard=info.get("guard", None),
            cross_core=bool(obligation.cross_core),
            sweep=int(extra.pop("sweep", 0)),
            issue_order=int(info.get("order", 0))))
        counter["n"] += 1
        return records[-1]

    # ---- the instance domain: (sweep, issue order) over real generations.
    slots = []
    for task_id, info in tasks.items():
        for generation in range(trip):
            sweep = generation + info["shift"]
            slots.append((sweep, info["order"], task_id, generation))
    slots.sort()
    for sweep, order, task_id, generation in slots:
        info = tasks[task_id]
        # 1. FREE wait before this instance's write, when a previous generation exists.
        for channel, obligation in free_writers.get(task_id, ()):
            if generation >= int(obligation.versions):
                emit(ROLE_WAIT, channel, obligation, generation,
                     generation % int(obligation.versions), task_id, sweep=sweep)
        # 2. STATE wait before the later writer's write.
        for channel, obligation in state_by_consumer.get(task_id, ()):
            emit(ROLE_WAIT, channel, obligation, generation,
                 generation % max(1, int(obligation.versions)), task_id, sweep=sweep)
        # 3. the real write, then READY/STATE sets.
        wrote = False
        for root, per_task in writers_by_root.items():
            if task_id in per_task:
                records.append(TraceRecord(
                    index=counter["n"], role=ROLE_WRITE_BEGIN,
                    channel="access:%s" % root, obligation_id="", kind="",
                    generation=int(generation),
                    slot=generation % max(1, int(obligation_set.versions.get(root, 1))),
                    versions=int(obligation_set.versions.get(root, 1)), task=task_id,
                    core="", pipe=str(info.get("pipe", "")),
                    producer_pipe=str(info.get("pipe", "")),
                    consumer_pipe=str(info.get("pipe", "")),
                    guard=info.get("guard", None), sweep=int(sweep),
                    issue_order=int(info.get("order", 0))))
                counter["n"] += 1
                wrote = True
        for channel, obligation in ready_by_producer.get(task_id, ()):
            emit(ROLE_SET, channel, obligation, generation,
                 generation % max(1, int(obligation.versions)), task_id, sweep=sweep)
        for channel, obligation in state_by_producer.get(task_id, ()):
            emit(ROLE_SET, channel, obligation, generation,
                 generation % max(1, int(obligation.versions)), task_id, sweep=sweep)
        # 4. READY waits, the real read, and the FREE set on the LAST consumer.
        for channel, obligation in ready_by_consumer.get(task_id, ()):
            emit(ROLE_WAIT, channel, obligation, generation,
                 generation % max(1, int(obligation.versions)), task_id, sweep=sweep)
        for root, per_task in readers_by_root.items():
            if task_id not in per_task:
                continue
            records.append(TraceRecord(
                index=counter["n"], role=ROLE_READ_BEGIN,
                channel="access:%s" % root, obligation_id="", kind="",
                generation=int(generation),
                slot=generation % max(1, int(obligation_set.versions.get(root, 1))),
                versions=int(obligation_set.versions.get(root, 1)), task=task_id,
                core="", pipe=str(info.get("pipe", "")),
                producer_pipe=str(info.get("pipe", "")),
                consumer_pipe=str(info.get("pipe", "")),
                guard=info.get("guard", None), sweep=int(sweep),
                issue_order=int(info.get("order", 0))))
            counter["n"] += 1
            records.append(TraceRecord(
                index=counter["n"], role=ROLE_READ_END,
                channel="access:%s" % root, obligation_id="", kind="",
                generation=int(generation),
                slot=generation % max(1, int(obligation_set.versions.get(root, 1))),
                versions=int(obligation_set.versions.get(root, 1)), task=task_id,
                core="", pipe=str(info.get("pipe", "")),
                producer_pipe=str(info.get("pipe", "")),
                consumer_pipe=str(info.get("pipe", "")),
                guard=info.get("guard", None), sweep=int(sweep),
                issue_order=int(info.get("order", 0))))
            counter["n"] += 1
            for (last_root, _producer), consumers in last_consumer.items():
                if last_root != root:
                    continue
                entry = consumers.get(task_id)
                if entry is None or not entry[0]:
                    continue
                _is_last, channel, obligation = entry
                emit(ROLE_SET, channel, obligation, generation,
                     generation % max(1, int(obligation.versions)), task_id, sweep=sweep)

    # 5. The exit drain. Its presence is derived: any asynchronous access at all, or a live
    #    exit root, requires it; a program with nothing asynchronous and no instances still
    #    emits it so the region-exit check is uniform.
    exit_records = []
    exit_obligations = [o for o in obligations if o.kind == KIND_EXIT]
    for obligation in exit_obligations:
        record = TraceRecord(
            index=counter["n"], role=ROLE_DRAIN, channel="drain.%s" % (
                obligation.buffer_root.split("/")[-1] or "region"),
            obligation_id=obligation.obligation_id, kind=KIND_EXIT,
            generation=(trip - 1 if trip else 0), slot=0, versions=1,
            task="", core=DEVICE, pipe="", producer_pipe="", consumer_pipe="",
            guard=GUARD_ALWAYS, sweep=(trip + depth), issue_order=10 ** 6)
        records.append(record)
        exit_records.append(record)
        counter["n"] += 1

    channels = tuple(sorted({record.channel for record in records
                             if record.role in (ROLE_SET, ROLE_WAIT)}))
    cross_pairs = tuple(obligation_set.cross_handoffs)
    return CompletionTrace(
        records=tuple(records), obligations=obligations, trip=trip, depth=depth,
        channels=channels, cross_pairs=cross_pairs, exit_records=tuple(exit_records),
        versions=dict(obligation_set.versions), cores=dict(obligation_set.cores),
        diagnostics=tuple(obligation_set.diagnostics))


__all__ = ["ROLE_WRITE_BEGIN", "ROLE_READ_BEGIN", "ROLE_READ_END", "ROLE_SET",
           "ROLE_WAIT", "ROLE_DRAIN", "ACCESS_ROLES", "MAX_COMPLETION_INSTANCES",
           "TraceRecord", "CompletionTrace", "build_completion_trace"]
