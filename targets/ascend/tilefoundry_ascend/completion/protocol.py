"""Ring-protocol simulation and completion analysis of a completion program.

WHY A SIMULATION AND NOT A COUNT. Comparing the number of Sets with the number of Waits proves
nothing: a program can have one Set and one Wait per generation and still hang, because the
Wait may execute BEFORE its Set, or under a guard the Set does not share, or wait for a
generation no Set ever reaches. Spec 3.5 states this explicitly ("Set/Wait 数量相等不能证明不存
在死锁"), so this module replays the trace through a real per-(buffer, slot) state machine,
exactly like reference_core/completion_reference.py, and additionally decides GUARD
compatibility and CROSS-CORE participant agreement.

THE STATE MACHINE, per (buffer root, slot):

    empty --write_begin(g)--> written(g) --ready_set(g)--> ready(g)
    ready(g) --read_begin(g,c)--> ready(g) with c active
    ready(g) --read_end(g,c)--> ready(g) with c done
    ready(g) --free_wait(g')--> ... only when g' = g + versions and every consumer of g is done
    free_set(g) when done(g) == consumers(g) -> empty(retired g)

The transitions encode the five spec-3.5 guarantees and the four refusals the gate names:

  * a read before READY                -> COMPLETION_READ_WITHOUT_READY
  * a FREE before the last consumer    -> COMPLETION_EARLY_FREE
  * a write onto a live slot           -> COMPLETION_OVERWRITE_BEFORE_FREE
  * a wait for a generation nobody set -> COMPLETION_WAIT_WITHOUT_SET (dropped Set / bad guard)
  * a slot still live at region exit   -> COMPLETION_EXIT_LIVE_STORAGE
  * a replayed old generation          -> COMPLETION_STALE_GENERATION

GENERATION, SLOT AND VERSIONS are re-derived here from the record's own numbers; a slot that
is not generation % versions is COMPLETION_SLOT_GENERATION_MISMATCH. The checker never reads
a 'verified' flag as its input.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .guards import ALWAYS, COND, NEVER, guard_implies, normalize_guard, guards_agree
from .model import (
    KIND_CROSS,
    KIND_EXIT,
    KIND_FREE,
    KIND_READY,
    KIND_STATE,
    CompletionError,
    digest,
)

# Access roles are imported lazily inside functions to keep this module import-light.


@dataclass
class SlotState:
    """One ring slot's state.

    Readiness is a SET OF EDGES, not a boolean. A buffer written by one producer and read by
    three consumers has three ready edges, and each consumer waits on its own; a single flag
    would make the second producer's Set look like a duplicate and would let one consumer's
    wait satisfy another's. The generation and the consumer sets stay slot-level because they
    are genuinely properties of the slot.
    """

    generation: object = None
    ready_edges: set = field(default_factory=set)
    active: set = field(default_factory=set)
    done: set = field(default_factory=set)
    retired: int = -1
    readers: tuple = ()


@dataclass(frozen=True)
class ProtocolVerdict:
    status: str                          # VERIFIED | FAILED
    code: str = ""
    detail: str = ""
    live_slots: tuple = ()
    waiting: tuple = ()
    checked_generations: int = 0
    cross_checks: tuple = ()
    deadlock_free: bool = False
    counts: dict = field(default_factory=dict)

    def as_dict(self):
        return {"status": self.status, "code": self.code, "detail": self.detail,
                "live_slots": [list(slot) for slot in self.live_slots],
                "waiting": [list(item) for item in self.waiting],
                "checked_generations": self.checked_generations,
                "cross_checks": [dict(item) for item in self.cross_checks],
                "deadlock_free": self.deadlock_free, "counts": dict(self.counts)}


def _fail(code, detail, live=(), waiting=(), counts=None, cross=()):
    return ProtocolVerdict("FAILED", code=code, detail=detail,
                           live_slots=tuple(live), waiting=tuple(waiting),
                           counts=dict(counts or {}), cross_checks=tuple(cross))


def _access_role(role):
    return role in ("write_begin", "read_begin", "read_end")


def simulate_trace(trace, *, obligations=None, consumers=None) -> ProtocolVerdict:
    """Replay the completion trace through the ring protocol.

    'consumers' maps (root, generation) -> the consumer set that must all finish before FREE.
    It is taken from the obligations when not supplied, so the FREE rule is the plan's own
    consumer set and not a guess from the trace.
    """
    obligations = tuple(obligations if obligations is not None else trace.obligations)
    # Which consumers must ALL have finished before a FREE may fire. Keyed by the FREE's own
    # obligation, and taken from the obligation set (the plan's REAL consumer set) rather than
    # guessed from the trace, so a program that silently drops a consumer from its FREE -- the
    # exact RingProtocol defect -- is caught instead of being confirmed.
    expected_consumers = {}
    # WHICH READY EDGES A GIVEN CONSUMER MUST OBSERVE, per (root, consumer task). With ONE
    # producer this is a single edge; with several producers writing disjoint regions of the
    # same root it is more than one, and a consumer that observes only some of them has not
    # been ordered after the others. Tracking readiness per SLOT instead of per EDGE would
    # conflate the two and both accept a partial wait and reject a legitimate second edge.
    required_ready = {}
    ready_channels = {}
    for obligation in obligations:
        if obligation.kind == KIND_FREE:
            expected_consumers[obligation.obligation_id] = tuple(obligation.consumers)
        elif obligation.kind == KIND_READY:
            ready_channels[obligation.obligation_id] = None
            required_ready.setdefault(
                (obligation.buffer_root, obligation.consumer_task), set()).add(
                    obligation.obligation_id)
    if consumers:
        expected_consumers.update(consumers)

    # WHICH ROOTS ARE *OBSERVED*. A root that some consumer reads has a FREE obligation, and
    # its slot may not be overwritten until that FREE fired. A root with NO reader at all
    # cannot have a stale read: overwriting it is not a hazard, because nothing can observe
    # the previous generation. Such a root is still covered at region exit by the drain (its
    # last asynchronous write may be in flight), which is exactly what the exit obligation is
    # for -- but it is NOT a FREE candidate, and demanding a FREE for it would invent a
    # consumer that does not exist.
    observed_roots = {obligation.buffer_root for obligation in obligations
                      if obligation.kind == KIND_FREE and obligation.consumers}
    unobserved_overwrites = 0

    slots = {}
    waiting = []
    # (consumer task, root, generation, kind) triples this program actually OBSERVED by
    # executing a satisfied wait. Used to refuse a read that relies on program order alone.
    served = set()
    # (consumer task, root, generation) triples whose FREE wait this program executed. A write
    # that reuses the slot must appear here, for the same reason a read must observe READY.
    freed = set()
    cross_checks = []
    counts = {"write_begin": 0, "read_begin": 0, "read_end": 0, "set": 0, "wait": 0,
              "drain": 0, "unobserved_overwrites": 0}
    live = {}
    pending_sets = {}
    cross_guard = {}
    generations_checked = set()

    for record in trace.records:
        guard = normalize_guard(record.guard)
        if guard.kind == NEVER:
            # A statically dead statement cannot order anything, and it cannot be coverage.
            continue
        counts[record.role] = counts.get(record.role, 0) + 1
        if _access_role(record.role):
            root = record.channel.split("access:", 1)[-1]
            slot = record.slot
            key = (root, slot)
            state = slots.setdefault(key, SlotState())
            generations_checked.add((root, record.generation))
            if record.role == "write_begin":
                if record.slot != record.generation % max(1, record.versions):
                    return _fail("COMPLETION_SLOT_GENERATION_MISMATCH",
                                 "write slot %d != generation %d %% versions %d"
                                 % (record.slot, record.generation, record.versions),
                                 live=sorted(live), waiting=waiting, counts=counts)
                if state.generation is not None:
                    if root in observed_roots:
                        return _fail(
                            "COMPLETION_OVERWRITE_BEFORE_FREE",
                            "generation %d writes slot %d of %s while generation %s still"
                            " occupies it and its consumers may not have read it"
                            % (record.generation, slot, root, state.generation),
                            live=sorted(live), waiting=waiting, counts=counts)
                    # No reader exists for this root: nothing can observe the previous
                    # generation, so the overwrite is not a hazard. Counted, not ignored.
                    unobserved_overwrites += 1
                if record.generation <= state.retired:
                    return _fail(
                        "COMPLETION_STALE_GENERATION",
                        "generation %d was written after generation %d was retired: an old"
                        " generation was replayed" % (record.generation, state.retired),
                        live=sorted(live), waiting=waiting, counts=counts)
                # A write that REUSES a slot must OBSERVE that the previous generation was
                # released. Program order is not that observation: the FREE set and the next
                # write are issued on different pipes (and possibly different cores), so
                # seeing the Set earlier in the statement list does not order them. Only a
                # satisfied FREE wait does. Without this check, deleting every FREE wait from
                # a correct program still "verifies", because the replay sees the FREE set
                # retire the slot before the next write reads it.
                # Only an OBSERVED root needs this: with no readers there is no consumer whose
                # read must complete, so the ring has no FREE edge to observe and demanding one
                # would invent a consumer (see observed_roots above).
                if (root in observed_roots
                        and int(record.generation) >= int(record.versions)
                        and not _waited(freed, record, KIND_FREE)):
                    return _fail(
                        "COMPLETION_OVERWRITE_BEFORE_FREE",
                        "generation %d writes slot %d of %s without executing a satisfied FREE"
                        " wait for the previous generation %d, so nothing orders the overwrite"
                        " after its consumers' reads"
                        % (record.generation, slot, root,
                           int(record.generation) - int(record.versions)),
                        live=sorted(live), waiting=waiting, counts=counts)
                state.generation = record.generation
                state.ready_edges = set()
                state.active = set()
                state.done = set()
                live[key] = record.generation
            else:
                if state.generation is None or state.generation != record.generation:
                    return _fail(
                        "COMPLETION_STALE_GENERATION",
                        "%s of generation %d on slot %d, but the slot holds generation %s"
                        % (record.role, record.generation, slot, state.generation),
                        live=sorted(live), waiting=waiting, counts=counts)
                if record.role == "read_begin":
                    if not state.ready_edges:
                        return _fail(
                            "COMPLETION_READ_WITHOUT_READY",
                            "read of %s generation %d begins before any READY set"
                            % (root, record.generation),
                            live=sorted(live), waiting=waiting, counts=counts)
                    # HAVING a READY set in the trace is not the same as the consumer
                    # OBSERVING it. A trace is one core's statement order; the producer may be
                    # on another core or an asynchronous pipe, so program order alone does not
                    # order a read after its producer. Only a satisfied ready WAIT does. A
                    # consumer that reads without waiting is therefore refused even though the
                    # REPLAY happens to see the set first -- which is precisely the defect a
                    # pure in-order simulation cannot see, and the reason this check exists.
                    if not _all_ready_waited(served, record, required_ready):
                        return _fail(
                            "COMPLETION_READ_WITHOUT_READY",
                            "consumer %r reads %s generation %d without executing a satisfied"
                            " READY wait for that generation, so nothing orders the read after"
                            " the producer" % (record.task, root, record.generation),
                            live=sorted(live), waiting=waiting, counts=counts)
                    consumer = record.task
                    if consumer in state.active or consumer in state.done:
                        return _fail(
                            "COMPLETION_DUPLICATE_CONSUMER",
                            "consumer %r reads generation %d twice" % (consumer,
                                                                      record.generation),
                            live=sorted(live), waiting=waiting, counts=counts)
                    state.active.add(consumer)
                else:  # read_end
                    consumer = record.task
                    if consumer not in state.active:
                        return _fail(
                            "COMPLETION_READ_COMPLETION_WITHOUT_READ",
                            "read completion for %r has no matching read begin" % (consumer,),
                            live=sorted(live), waiting=waiting, counts=counts)
                    state.active.discard(consumer)
                    state.done.add(consumer)
            continue

        if record.role == "set":
            key = (record.channel, record.generation, record.kind)
            root, generation = _root_of(record, obligations), record.generation
            slot_key = (root, record.slot)
            state = slots.setdefault(slot_key, SlotState())
            if state.generation is None or state.generation != record.generation:
                return _fail(
                    "COMPLETION_STALE_GENERATION",
                    "SET of %s generation %d does not match the slot's generation %s"
                    % (record.channel, record.generation, state.generation),
                    live=sorted(live), waiting=waiting, counts=counts)
            if record.kind == KIND_READY:
                # Idempotence is per EDGE, not per slot: two producers writing disjoint regions
                # of one buffer each have their own ready edge, and the second is not a
                # duplicate of the first.
                edge = record.obligation_id or record.channel
                if edge in state.ready_edges:
                    return _fail("COMPLETION_DUPLICATE_READY",
                                 "READY set twice for %s generation %d"
                                 % (record.channel, record.generation),
                                 live=sorted(live), waiting=waiting, counts=counts)
                state.ready_edges.add(edge)
                pending_sets[(record.channel, record.generation)] = record.index
            elif record.kind == KIND_FREE:
                required = expected_consumers.get(record.obligation_id, ()) or ()
                _note_set(pending_sets, record)
                if not state.ready_edges:
                    return _fail("COMPLETION_EARLY_FREE",
                                 "FREE for %s generation %d while the slot was never READY"
                                 % (record.channel, record.generation),
                                 live=sorted(live), waiting=waiting, counts=counts)
                if state.active:
                    return _fail(
                        "COMPLETION_EARLY_FREE",
                        "FREE for %s generation %d while consumers %r are still reading"
                        % (record.channel, record.generation, sorted(state.active)),
                        live=sorted(live), waiting=waiting, counts=counts)
                if required and set(state.done) != set(required):
                    missing = sorted(set(required) - set(state.done))
                    return _fail(
                        "COMPLETION_EARLY_FREE",
                        "FREE for %s generation %d before every consumer completed: missing %r"
                        % (record.channel, record.generation, missing),
                        live=sorted(live), waiting=waiting, counts=counts)
                state.retired = state.generation
                state.generation = None
                state.ready_edges = set()
                state.active = set()
                state.done = set()
                live.pop(slot_key, None)
            else:
                _note_set(pending_sets, record)
            continue

        if record.role == "wait":
            root = _root_of(record, obligations)
            slot_key = (root, record.slot)
            state = slots.setdefault(slot_key, SlotState())
            key = (record.channel, record.generation, record.kind)
            waiting.append((record.channel, record.generation, record.kind))
            if record.kind == KIND_FREE:
                # A FREE wait WAITS FOR THE SLOT TO BE FREE, and "free" is a statement about a
                # SPECIFIC predecessor: the generation 'versions' rings earlier. Two failures
                # follow from that and both are checked, because an empty slot is not on its
                # own evidence that the right generation was released:
                #   * the slot is still occupied  -> the wait would block forever;
                #   * the slot is empty but the generation released was NOT g - versions
                #     -> the wait is keyed to a generation this ring never produced (a stale
                #        descriptor), which in-order replay would otherwise accept silently
                #        whenever the slot happens to be empty at that moment.
                previous = int(record.generation) - int(record.versions)
                if state.generation is not None:
                    return _fail(
                        "COMPLETION_FREE_WAIT_UNSATISFIED",
                        "FREE wait for generation %d of %s runs while generation %s still"
                        " occupies the slot, so the wait would block forever"
                        % (record.generation, record.channel, state.generation),
                        live=sorted(live), waiting=waiting, counts=counts)
                if previous < 0:
                    return _fail(
                        "COMPLETION_FREE_WAIT_UNSATISFIED",
                        "FREE wait for generation %d of %s has no predecessor: generation %d"
                        " precedes the first write of this ring"
                        % (record.generation, record.channel, previous),
                        live=sorted(live), waiting=waiting, counts=counts)
                # Record the observation against the root and the generation being ACQUIRED,
                # not against the task that executed the wait: a ring's free edge and its next
                # write belong to the same producer task, but the statement that waits may be
                # emitted under either the producer's or the consumer's instance depending on
                # the schedule. Keying on the ROOT + generation is what makes the check
                # "this slot's reuse was observed", which is the actual hazard.
                freed.add((root, int(record.generation)))
                if state.retired != previous:
                    return _fail(
                        "COMPLETION_STALE_GENERATION",
                        "FREE wait for generation %d of %s expects generation %d to have been"
                        " released, but the slot last released generation %d: the wait is"
                        " keyed to a generation this ring did not produce"
                        % (record.generation, record.channel, previous, state.retired),
                        live=sorted(live), waiting=waiting, counts=counts)
                waiting.pop()
                continue
            if record.kind == KIND_STATE:
                waiting.pop()
                continue
            if state.generation is None or state.generation != record.generation:
                return _fail(
                    "COMPLETION_WAIT_WITHOUT_SET",
                    "WAIT for %s generation %d, but the slot holds generation %s: no Set of"
                    " that generation executes before this wait"
                    % (record.channel, record.generation, state.generation),
                    live=sorted(live), waiting=waiting, counts=counts)
            if not state.ready_edges:
                return _fail(
                    "COMPLETION_WAIT_BEFORE_SET",
                    "WAIT for %s generation %d executes before the READY set of that generation"
                    % (record.channel, record.generation),
                    live=sorted(live), waiting=waiting, counts=counts)
            # Record that this consumer OBSERVED this EDGE's generation. The read check
            # requires EVERY ready edge into the consumer, so an observation of one producer is
            # not mistaken for ordering against another.
            served.add((root, record.obligation_id, record.generation, record.kind))
            set_index = pending_sets.get((record.channel, record.generation))
            if set_index is not None and set_index > record.index:
                return _fail(
                    "COMPLETION_WAIT_BEFORE_SET",
                    "WAIT@%d for %s generation %d precedes its SET@%d"
                    % (record.index, record.channel, record.generation, set_index),
                    live=sorted(live), waiting=waiting, counts=counts)
            waiting.pop()
            continue

        if record.role == "drain":
            continue

        return _fail("COMPLETION_UNKNOWN_ROLE", "unknown trace role %r" % (record.role,),
                     live=sorted(live), waiting=waiting, counts=counts)

    # ---- region exit. A slot may legitimately still be live at exit WHEN A DRAIN COVERS IT:
    # that is precisely what the drain is for (a final asynchronous write whose completion
    # nobody reads). Without a reachable drain, a live asynchronous slot at region exit is the
    # defect the spec names, because nothing waits for the hardware to finish.
    if live and not _reachable_drains(trace):
        return _fail(
            "COMPLETION_EXIT_LIVE_STORAGE",
            "the region exits with %d live asynchronous slot(s) %r and no reachable drain"
            % (len(live), sorted(live)),
            live=sorted(live), waiting=waiting, counts=counts, cross=cross_checks)
    counts["drained_live_slots"] = len(live) if _reachable_drains(trace) else 0
    if waiting:
        return _fail(
            "COMPLETION_EXIT_UNSATISFIED_WAIT",
            "the region exits with %d unsatisfied wait(s): %r"
            % (len(waiting), waiting[:6]),
            live=(), waiting=waiting, counts=counts, cross=cross_checks)
    if not _reachable_drains(trace):
        return _fail("FINAL_DRAIN_MISSING", "no drain statement in the completion program",
                     counts=counts, cross=cross_checks)

    # ---- guard compatibility for every edge whose two sides both exist.
    guard_problems = _guard_checks(trace)
    if guard_problems:
        first = guard_problems[0]
        return _fail("COMPLETION_GUARD_INCOMPATIBLE", first, counts=counts,
                     cross=cross_checks)

    # ---- cross-core participant/visibility agreement.
    cross_problems, cross_checks = _cross_core_checks(trace)
    if cross_problems:
        return _fail("COMPLETION_CROSS_CORE_MISMATCH", cross_problems[0], counts=counts,
                     cross=cross_checks)

    # ---- deadlock analysis: every wait that survives the sweep is satisfied by a set that
    #      provably executes on an earlier or equal sweep for the same generation.
    deadlock = _deadlock_analysis(trace)
    if deadlock is not None:
        return _fail(deadlock[0], deadlock[1], counts=counts, cross=cross_checks)

    return ProtocolVerdict("VERIFIED", counts=counts,
                           checked_generations=len(generations_checked),
                           cross_checks=tuple(cross_checks), deadlock_free=True)


def _waited(served, record, kind):
    """Did this root's FREE edge get observed for this generation?"""
    root = record.channel.split("access:", 1)[-1]
    if kind == KIND_FREE:
        return (root, int(record.generation)) in served
    return (root, record.generation, kind) in served


def _all_ready_waited(served, record, required_ready):
    """Did this consumer observe EVERY ready edge into it, for this generation?

    An edge counts as observed when a satisfied wait named the obligation that carries it. A
    consumer fed by two producers must wait on both; observing one is not ordering against the
    other, and a slot-level "was ready" flag cannot tell the difference.
    """
    root = record.channel.split("access:", 1)[-1]
    edges = required_ready.get((root, record.task))
    if edges is None:
        return (root, record.generation, KIND_READY) in served
    return all((root, obligation_id, record.generation, KIND_READY) in served
               for obligation_id in edges)


def _reachable_drains(trace):
    """The drains that can actually execute. A drain inside a statically dead arm is not a
    drain: the hardware would leave the region with the final write still in flight."""
    out = []
    for record in trace.records:
        if record.role != "drain":
            continue
        if not getattr(record, "reachable", True):
            continue
        if normalize_guard(record.guard).kind == NEVER:
            continue
        out.append(record)
    return out


def _note_set(pending_sets, record):
    pending_sets[(record.channel, record.generation)] = record.index


def _root_of(record, obligations):
    """The buffer root a completion record belongs to (from its obligation)."""
    for obligation in obligations:
        if obligation.obligation_id == record.obligation_id:
            return obligation.buffer_root
    # A synthesized record (a test mutation) may carry no matching obligation: derive the
    # root from the channel name so the state machine still keys on the real slot.
    return record.channel.split(".")[1] if record.channel.count(".") >= 1 else record.channel


def _guard_checks(trace):
    problems = []
    sets = {}
    waits = {}
    for record in trace.records:
        if record.role == "set":
            sets.setdefault((record.channel, record.generation, record.kind), []).append(record)
        elif record.role == "wait":
            waits.setdefault((record.channel, record.generation, record.kind), []).append(record)
    for key, waiters in sorted(waits.items()):
        setters = sets.get(key, [])
        if not setters:
            problems.append("WAIT for %r generation %d has NO Set at all (dropped Set or a"
                            " guard the setter never shares)" % (key[0], key[1]))
            continue
        for waiter in waiters:
            waiter_guard = normalize_guard(waiter.guard)
            ok = any(guard_implies(waiter_guard, normalize_guard(setter.guard))[0]
                     for setter in setters)
            if not ok:
                problems.append(
                    "WAIT for %r generation %d can execute while its Set is skipped: waiter"
                    " guard %r vs setter guard %r"
                    % (key[0], key[1], normalize_guard(waiter.guard).as_dict(),
                       normalize_guard(setters[0].guard).as_dict()))
    return problems


def _cross_core_checks(trace):
    """Both sides of a cross-core handoff must agree on participants and the group predicate."""
    problems = []
    checks = []
    pairs = {pair["obligation_id"]: pair for pair in trace.cross_pairs}
    sets = {}
    waits = {}
    for record in trace.records:
        if record.obligation_id not in pairs:
            continue
        if record.role == "set":
            sets.setdefault(record.obligation_id, []).append(record)
        elif record.role == "wait":
            waits.setdefault(record.obligation_id, []).append(record)
    for obligation_id, pair in sorted(pairs.items()):
        handoff_sets = sets.get(obligation_id, [])
        handoff_waits = waits.get(obligation_id, [])
        if not handoff_sets or not handoff_waits:
            problems.append("cross-core handoff %s has sets=%d waits=%d; a handoff needs both"
                            " sides" % (obligation_id, len(handoff_sets), len(handoff_waits)))
            continue
        if pair["from_domain"] == pair["to_domain"]:
            problems.append("cross-core handoff %s names one domain on both sides (%s)"
                            % (obligation_id, pair["from_domain"]))
            continue
        setter_guard = normalize_guard(handoff_sets[0].guard)
        waiter_guard = normalize_guard(handoff_waits[0].guard)
        ok, reason = guards_agree(setter_guard, waiter_guard)
        if not ok:
            problems.append("cross-core handoff %s: %s" % (obligation_id, reason))
            continue
        checks.append({"obligation_id": obligation_id,
                       "from": pair["from_domain"], "to": pair["to_domain"],
                       "participants": list(pair["participants"]),
                       "visibility": pair["visibility"],
                       "predicate": setter_guard.as_dict()})
    return problems, checks


def _deadlock_analysis(trace):
    """Every WAIT must be satisfied by a Set that PROVABLY executes for the same generation.

    A wait is satisfied when, for its (channel, generation), some set with a compatible guard
    exists at an EARLIER or EQUAL index in the same sweep domain, or an earlier sweep for the
    same channel (the ring's cross-iteration case). This is a real dependency argument: it uses
    the generation, the guard and the instance order, not a count.
    """
    sets = {}
    for record in trace.records:
        if record.role != "set":
            continue
        guard = normalize_guard(record.guard)
        if guard.kind == NEVER:
            continue
        sets.setdefault((record.channel, record.kind), []).append(record)
    for record in trace.records:
        if record.role != "wait":
            continue
        waiter_guard = normalize_guard(record.guard)
        if waiter_guard.kind == NEVER:
            continue
        candidates = sets.get((record.channel, record.kind), [])
        cover = [setter for setter in candidates
                 if setter.generation == record.generation
                 and guard_implies(waiter_guard, normalize_guard(setter.guard))[0]
                 and (setter.index < record.index or setter.sweep < record.sweep
                      or setter.generation <= record.generation)]
        if not cover:
            return ("COMPLETION_STARTUP_UNSATISFIED",
                    "WAIT@%d for %r generation %d has no Set that provably executes for that"
                    " generation (a dropped wait/Set pairing or a mismatched generation)"
                    % (record.index, record.channel, record.generation))
    return None


__all__ = ["SlotState", "ProtocolVerdict", "simulate_trace"]
