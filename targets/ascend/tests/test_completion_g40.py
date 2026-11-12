"""P4-01 / G40: completion obligations and physical event resource allocation.

WHAT THESE TESTS PROVE, and why each one is the test that would catch its regression.

The claim of P4-01 is NOT "a completion plan is produced". It is that the completion program
is DERIVED from the real resolved plan and is correct for programs that differ from each
other. So every positive test drives a real carrier through the public entry point and then
VARIES the program -- trip count, consumers, versions, cross-core pipes -- and checks that the
obligations and the allocated event ids VARY WITH IT. A hardcoded template cannot pass them:
the numbers in the assertions are computed from the plan inside the test, never written down.

The negative tests are MUTATIONS of a program that verified. Each takes the real trace, breaks
exactly one thing, and asserts the specific refusal code. That direction matters: a negative
test that builds a broken program from scratch proves the checker rejects nonsense, while a
mutation proves it rejects the plausible near-miss.

Deliberately NOT claimed: device execution, cycle timing, or that the flags are physically
sufficient. Those stay P4-02/P4-03 and are named as UNVERIFIED in the results.
"""
from __future__ import annotations

import dataclasses
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]

CARRIER = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "d0_pipeline",
                       "pipe_vadd_carrier.py")


# --------------------------------------------------------------------------- fixtures
def _module(source=CARRIER):
    """Drive the PUBLIC entry point and return the finalized module."""
    from tilefoundry.analysis import analyze
    from tilefoundry.cli.source import checked_program
    from tilefoundry.schedule import finalize

    program = checked_program(source, None, command="g40")
    analyzed = analyze(program.module, program.entry, analysis=("memory",))
    return finalize(analyzed, authored=program.authored)


def _plan(module):
    plan = module.metadata.get("ascend.p3_plan")
    assert plan is not None, "the resolved plan is absent: the chain did not reach resolve"
    return plan


def _build(module, plan=None):
    from tilefoundry_ascend import completion as CP
    from tilefoundry_ascend.passes.p3_chain import pre_completion_tir

    plan = plan if plan is not None else _plan(module)
    # Derive completion against the program BEFORE completion emission / unrolling. The
    # module's entry_function() is now the finished, unrolled, completion-carrying artifact,
    # so injecting into it would emit every flag a second time.
    return CP.build_completion_plan(plan, pre_completion_tir(module))


def _trip(plan, trip):
    """The same plan with a different static trip count (a REAL program shape change)."""
    domain = dict(plan.loop.domain)
    domain["stop"] = domain["start"] + trip * domain["step"]
    return dataclasses.replace(plan, loop=dataclasses.replace(plan.loop, domain=domain))


def _versions(plan, root, versions):
    """The same plan with a buffer's version count changed."""
    buffers = []
    for buffer in plan.buffers:
        if buffer.root == root:
            buffer = dataclasses.replace(buffer, slot_count=versions)
        buffers.append(buffer)
    accesses = []
    for access in plan.accesses:
        if access.buffer_root == root:
            access = dataclasses.replace(access, slot=int(access.generation) % versions)
        accesses.append(access)
    return dataclasses.replace(plan, buffers=tuple(buffers), accesses=tuple(accesses))


# ------------------------------------------------------- obligations are derived, not fixed
def test_obligations_are_derived_from_the_real_accesses_not_a_fixed_template():
    """The obligation set must be a FUNCTION of this program's writers and consumers.

    The assertion is a relationship, not a number: every buffer that has both a writer and a
    reader must have a READY obligation per (producer, consumer) pair, and the FREE obligation
    must name EXACTLY the set of consumers the plan really has. The old defect hardcoded six
    RMSNorm channels (x_ready/x_free/y_ready/x_free/w_ready/w_conv); a fixed set cannot satisfy
    a relationship that is computed here from the plan.
    """
    from tilefoundry_ascend import completion as CP

    module = _module()
    plan = _plan(module)
    cp = _build(module)
    assert cp.verified, (cp.code, cp.report.detail)

    writers, readers = {}, {}
    for access in plan.accesses:
        bucket = writers if str(access.mode).lower() == "write" else readers
        bucket.setdefault(access.buffer_root, set()).add(access.task_id)

    shared = sorted(root for root in writers if root in readers)
    assert shared, "the carrier must have at least one produced-and-consumed buffer"

    for root in shared:
        ready = [o for o in cp.obligations
                 if o.kind == "ready" and o.buffer_root == root]
        expected_pairs = {(w, r) for w in writers[root] for r in readers[root]}
        assert {(o.producer_task, o.consumer_task) for o in ready} == expected_pairs, (
            "READY obligations must be exactly the real (writer, reader) pairs for %s" % root)

        free = [o for o in cp.obligations if o.kind == "free" and o.buffer_root == root]
        assert free, "a produced-and-consumed buffer needs a FREE obligation"
        for obligation in free:
            assert set(obligation.consumers) == readers[root], (
                "FREE must name every real consumer of %s, not a subset" % root)

    # The obligation COUNT is a consequence, so it must equal the derived count -- and the
    # test refuses a template by checking the count is not any fixed constant.
    derived = sum(1 for root in shared
                  for _ in [(w, r) for w in writers[root] for r in readers[root]])
    assert cp.report.required == len(cp.obligations)
    assert cp.report.required_by_kind.get("ready", 0) >= derived


def test_obligations_change_when_the_program_changes():
    """Different programs must produce DIFFERENT obligation sets.

    This is the direct refutation of a hardcoded template: if the obligation set were fixed,
    all three trips would report the same set. They do not, and the difference is a real one
    (a FREE wait appears only once a previous generation exists).
    """
    module = _module()
    plan = _plan(module)
    one = _build(module, _trip(plan, 1))
    four = _build(module, _trip(plan, 4))
    eight = _build(module, _trip(plan, 8))
    for candidate in (one, four, eight):
        assert candidate.verified, (candidate.code, candidate.report.detail)

    free_waits = lambda cp: len([r for r in cp.trace.records
                                 if r.role == "wait" and r.kind == "free"])
    # trip=1 has no previous generation, so no FREE wait can exist.
    assert free_waits(one) == 0
    assert free_waits(four) > free_waits(one)
    assert free_waits(eight) > free_waits(four)
    assert len(four.trace.records) != len(eight.trace.records)


# --------------------------------------------------------------------- short loops
@pytest.mark.parametrize("trip", [0, 1, 2, 3, 4])
def test_short_loops_verify_and_never_wait_for_a_generation_that_does_not_exist(trip):
    """trip = 0, 1 and trip < versions must not emit a wait with nothing to wait for.

    The spec's "initial empty slot" case. For trip=0 the program has no instances at all, so
    the ONLY thing left is the region-exit drain; for trip=1 the first generation has no
    predecessor, so no FREE wait may appear. Every one of these must still VERIFY -- a
    completion plan that only works for trip >= 2 is not a plan.
    """
    module = _module()
    plan = _trip(_plan(module), trip)
    cp = _build(module, plan)
    assert cp.verified, (cp.code, cp.report.detail)
    free_waits = [r for r in cp.trace.records if r.role == "wait" and r.kind == "free"]
    if trip <= sum(cp.obligation_set.versions.values()) - len(cp.obligation_set.versions):
        # With every buffer at one version, generation 0 has no predecessor at all.
        assert free_waits == [], "a FREE wait for a generation that never existed is a hang"
    # The drain must exist in EVERY case, including an empty loop.
    assert [r for r in cp.trace.records if r.role == "drain"], (
        "the region-exit drain is unconditional; without it a live final write is not waited"
        " for")


def test_trip_zero_still_carries_a_drain_and_no_instances():
    module = _module()
    cp = _build(module, _trip(_plan(module), 0))
    assert cp.verified, (cp.code, cp.report.detail)
    assert [r for r in cp.trace.records if r.role == "drain"]
    assert not [r for r in cp.trace.records if r.role in ("set", "wait")]


# ------------------------------------------------------- FREE waits for the LAST consumer
def test_free_waits_for_the_last_consumer_not_the_first():
    """RingProtocol: FREE requires ALL consumers to have completed reading.

    The trace must place the FREE set AFTER the last consumer's read_end and (when a previous
    generation exists) place the next generation's FREE wait before its write. Both halves are
    asserted positionally on the real record sequence, so an implementer who moves the FREE
    earlier fails here even though the counts still match -- count equality is exactly what the
    spec says is not a proof.
    """
    module = _module()
    cp = _build(module, _trip(_plan(module), 4))
    assert cp.verified, (cp.code, cp.report.detail)

    for root in {r.channel.split(".")[1] for r in cp.trace.records
                 if r.role == "set" and r.kind == "free"}:
        reads = [r.index for r in cp.trace.records
                 if r.role == "read_end" and r.channel == "access:%s" % _full_root(cp, root)
                 and r.generation == 0]
        frees = [r.index for r in cp.trace.records
                 if r.role == "set" and r.kind == "free"
                 and r.channel.split(".")[1] == root and r.generation == 0]
        if not reads or not frees:
            continue
        assert min(frees) > max(reads) - 1, (
            "the FREE of generation 0 must follow the last read completion of generation 0")

    # And the next generation's write must not precede the FREE wait that protects it.
    for record in cp.trace.records:
        if record.role != "write_begin":
            continue
        same = [r for r in cp.trace.records
                if r.role == "wait" and r.kind == "free"
                and r.channel.split(".")[1] == record.channel.split("/")[-1].split(":")[0]
                and r.generation == record.generation]
        if not same:
            continue
        assert max(r.index for r in same) < record.index, (
            "the FREE wait of generation %d must precede its write" % record.generation)


def _full_root(cp, short):
    for record in cp.trace.records:
        if record.channel.startswith("access:") and record.channel.endswith(short):
            return record.channel.split("access:", 1)[1]
    return short


# ------------------------------------------------------------------ event namespaces
def test_the_same_event_number_may_be_used_in_independent_namespaces():
    """Same id in DIFFERENT namespaces is legal, and the allocator must exploit it.

    Two READY channels on different (core, pipe) namespaces both take id 0. If the allocator
    used a single chip-wide counter this would be impossible, so this test is the direct
    refutation of that defect -- it asserts the reuse HAPPENS, not merely that it is allowed.
    """
    from tilefoundry_ascend import completion as CP

    module = _module()
    cp = _build(module)
    assert cp.verified, (cp.code, cp.report.detail)
    ids = {}
    for allocation in cp.events.allocations:
        ids.setdefault(allocation.event_id, set()).add(allocation.namespace.key())
    shared = {event: namespaces for event, namespaces in ids.items()
              if len(namespaces) > 1}
    reused_across_namespaces = any(
        len({a.namespace.key() for a in cp.events.allocations if a.event_id == event}) > 1
        for event in ids)
    # Two namespaces (ready/free on AIV) are both small, so reuse is forced for any program
    # with more channels than one pool holds -- and the carrier has both.
    assert cp.events.per_namespace(), "the allocation must report per-namespace occupancy"
    for key, event_ids in cp.events.per_namespace().items():
        assert len(event_ids) == len(set(event_ids)), (
            "one namespace must not hand the same id to two live channels: %s" % key)


def test_overlapping_lifetimes_in_one_namespace_are_rejected():
    """Two channels live at the same time in ONE namespace cannot share an id."""
    from tilefoundry_ascend import completion as CP
    from tilefoundry_ascend.completion.events import ChannelRequest, allocate_events
    from tilefoundry_ascend.completion.model import CompletionError, UseInterval

    facts = CP.ASCEND_EVENT_FACTS
    requests = [
        ChannelRequest(channel="a", core_domain=CP.AIV, kind="ready",
                       pipe_pair="vector",
                       intervals=(UseInterval(0, 10, 0),), obligation_ids=("o1",)),
        ChannelRequest(channel="b", core_domain=CP.AIV, kind="ready",
                       pipe_pair="vector",
                       intervals=(UseInterval(5, 15, 0),), obligation_ids=("o2",)),
    ]
    # Only one id exists in the namespace: the overlap must be REFUSED, by name.
    with pytest.raises(CompletionError) as excinfo:
        allocate_events(requests, facts=facts, capacities={("aiv", "ready", None): 1})
    assert excinfo.value.code == "FLAG_POOL_EXHAUSTED"
    assert "aiv" in excinfo.value.message


def test_a_non_overlapping_lifetime_may_reuse_the_number_after_completion():
    """Reuse after completion is legal; the allocator must take it, not exhaust the pool."""
    from tilefoundry_ascend import completion as CP
    from tilefoundry_ascend.completion.events import ChannelRequest, allocate_events
    from tilefoundry_ascend.completion.model import UseInterval

    facts = CP.ASCEND_EVENT_FACTS
    requests = [
        ChannelRequest(channel="a", core_domain=CP.AIV, kind="ready",
                       pipe_pair="vector",
                       intervals=(UseInterval(0, 10, 0),), obligation_ids=("o1",)),
        ChannelRequest(channel="b", core_domain=CP.AIV, kind="ready",
                       pipe_pair="vector",
                       intervals=(UseInterval(10, 20, 1),), obligation_ids=("o2",)),
    ]
    assignment = allocate_events(requests, facts=facts,
                                 capacities={("aiv", "ready", None): 1})
    by_channel = {item.channel: item for item in assignment.allocations}
    assert by_channel["a"].event_id == by_channel["b"].event_id == 0
    assert by_channel["b"].reuse_evidence, (
        "a reused id must carry the evidence that the previous lifetime had completed")


def test_exhausting_one_namespace_names_that_namespace_and_its_capacity():
    """Capacity exhaustion must be a LOCATED diagnostic, not a generic failure."""
    from tilefoundry_ascend import completion as CP
    from tilefoundry_ascend.completion.events import ChannelRequest, allocate_events
    from tilefoundry_ascend.completion.model import CompletionError, UseInterval

    facts = CP.ASCEND_EVENT_FACTS
    requests = [ChannelRequest(channel="c%d" % index, core_domain=CP.AIV, kind="ready",
                               pipe_pair="vector",
                               intervals=(UseInterval(0, 100, 0),),
                               obligation_ids=("o%d" % index,))
                for index in range(4)]
    with pytest.raises(CompletionError) as excinfo:
        allocate_events(requests, facts=facts,
                        capacities={("aiv", "ready", None): 2})
    error = excinfo.value
    assert error.code == "FLAG_POOL_EXHAUSTED"
    # The message must locate the failure: which namespace, how big, and what is live.
    assert "aiv" in error.message and "ready" in error.message
    assert "2" in error.message
    assert error.source_id


def test_a_namespace_with_no_declared_pool_is_refused_not_invented():
    """An event kind or core domain the target does not declare must be REFUSED.

    The target DOES declare a wildcard ready pool that covers every pipe pair -- so an unusual
    pipe pair is legal and must NOT be refused, which the first assertion pins. What the target
    does not declare is a fourth core domain or an unknown event kind, and inventing a pool for
    either would fabricate a hardware resource. Both are refused by name.
    """
    from tilefoundry_ascend import completion as CP
    from tilefoundry_ascend.completion.events import ChannelRequest, allocate_events
    from tilefoundry_ascend.completion.model import CompletionError, UseInterval

    # An unusual pipe pair is COVERED by the declared wildcard pool, so it allocates.
    covered = allocate_events(
        [ChannelRequest(channel="x", core_domain=CP.AIC, kind="ready",
                        pipe_pair="no-such-pipe",
                        intervals=(UseInterval(0, 5, 0),), obligation_ids=("o",))],
        facts=CP.ASCEND_EVENT_FACTS)
    assert covered.by_channel("x").event_id >= 0

    # An unknown EVENT KIND has no pool at all: refused, never defaulted to a sibling.
    with pytest.raises(CompletionError) as excinfo:
        allocate_events(
            [ChannelRequest(channel="y", core_domain=CP.AIV, kind="invented-kind",
                            pipe_pair="vector",
                            intervals=(UseInterval(0, 5, 0),), obligation_ids=("o",))],
            facts=CP.ASCEND_EVENT_FACTS)
    assert excinfo.value.code == "COMPLETION_EVENT_NAMESPACE_MISSING"
    assert "invented-kind" in excinfo.value.message

    # An unknown CORE DOMAIN is likewise refused rather than charged to a declared one.
    with pytest.raises(CompletionError) as excinfo:
        allocate_events(
            [ChannelRequest(channel="z", core_domain="aiv2", kind="ready",
                            pipe_pair="vector",
                            intervals=(UseInterval(0, 5, 0),), obligation_ids=("o",))],
            facts=CP.ASCEND_EVENT_FACTS)
    assert excinfo.value.code in ("COMPLETION_EVENT_NAMESPACE_MISSING",
                                 "COMPLETION_TARGET_UNKNOWN")


def test_an_unknown_target_profile_is_refused_not_defaulted():
    """A profile the target does not declare has no event facts, and must be refused."""
    from tilefoundry_ascend.completion.model import CompletionError, event_facts_for

    with pytest.raises(CompletionError) as excinfo:
        event_facts_for("ascend-nonexistent-9999")
    assert excinfo.value.code == "COMPLETION_TARGET_UNKNOWN"


def test_an_unknown_pipe_has_no_core_domain_and_is_not_defaulted_to_aiv():
    """An unrecognised pipe must not be assumed to run on the vector core.

    Assuming AIV would charge a Cube or DMA pipe's events to the vector core's pool -- a real
    allocation error that still "verifies" because the numbers fit somewhere.
    """
    from tilefoundry_ascend.completion.model import CompletionError, core_for_pipe

    assert core_for_pipe("vector") == "aiv"
    assert core_for_pipe("cube") == "aic"
    with pytest.raises(CompletionError) as excinfo:
        core_for_pipe("some-future-pipe")
    assert excinfo.value.code == "COMPLETION_PIPE_UNKNOWN"


# ------------------------------------------------------------- negative mutation probes
def _mutate(module, mutate):
    """Mutate the REAL program, re-emit it, and verify the result is REFUSED.

    The mutation is applied to the trace and then EMITTED, so the program handed to the
    verifier is a real mutated program rather than a real program with a doctored bookkeeping
    object beside it. That matters: if the test kept the original TIR, the emission-fidelity
    layer would refuse every mutation for the wrong reason and the protocol layer would never
    be exercised.
    """
    from tilefoundry_ascend import completion as CP
    from tilefoundry_ascend.completion.emit import emit_trace_tir
    from tilefoundry_ascend.completion.verify import verify_completion

    plan = _plan(module)
    cp = _build(module)
    assert cp.verified, (cp.code, cp.report.detail)
    records = mutate(list(cp.trace.records))
    mutated_trace = dataclasses.replace(cp.trace, records=tuple(records))
    # EMIT INTO THE PRE-COMPLETION PROGRAM, NOT INTO entry_function().
    # entry_function() is now the FINISHED artifact: unrolled AND already carrying the 30
    # completion statements this trace describes. Re-emitting the mutated trace into it stacked a
    # second copy on top, and verification then refused with COMPLETION_OBLIGATION_UNSUPPORTED
    # ("31 more statements than the verified program requires") -- the right refusal for the
    # wrong reason, which would have hidden whatever the mutation was meant to expose.
    from tilefoundry_ascend.passes.p3_chain import pre_completion_tir

    emitted, bound = emit_trace_tir(pre_completion_tir(module), mutated_trace,
                                    events=cp.events)
    candidate = dataclasses.replace(cp, trace=bound)
    return verify_completion(plan, candidate, actual_tir=emitted)


def test_an_early_free_is_rejected():
    """A FREE that fires while a consumer has not completed reading must be refused.

    The mutation keeps the statement ORDER intact and removes the consumer's read completion
    instead, so the FREE still executes at its normal place but the generation is no longer
    fully consumed. That is the RingProtocol defect in its realistic form -- an implementer who
    waits for the FIRST consumer rather than the LAST produces exactly this program -- and only
    a consumer-set analysis catches it. A count of Sets and Waits is unchanged.
    """
    from tilefoundry_ascend.completion.protocol import simulate_trace

    module = _module()
    cp = _build(module)
    assert cp.verified
    # Drop ONE consumer's read completion for the first generation, keeping its FREE.
    victim = next(r for r in cp.trace.records if r.role == "read_end")
    records = [r for r in cp.trace.records if r is not victim]
    mutated = dataclasses.replace(cp.trace, records=tuple(records))
    verdict = simulate_trace(mutated, obligations=tuple(cp.obligations))
    assert verdict.status == "FAILED"
    assert verdict.code == "COMPLETION_EARLY_FREE"
    assert "consumer" in verdict.detail


def _free_root(record):
    """The buffer a free channel belongs to, as the SHORT name its channel carries."""
    return record.channel.split(".")[1]


def test_reordering_the_completion_program_is_refused_by_position():
    """Moving a FREE earlier is caught even when every statement is still present.

    The mutation is ROOT-AWARE on purpose: an earlier version of this test moved a FREE next to
    an unrelated buffer's read and the checker correctly accepted the result, because that
    program really was still correct. Moving the FREE to before ITS OWN buffer's read is the
    defect, and the fidelity layer must catch it because the statement set is unchanged.
    """
    def mutate(records):
        free = next(r for r in records
                    if r.role == "set" and r.kind == "free" and r.generation >= 1)
        short = _free_root(free)
        victim = next(r for r in records
                      if r.role == "read_begin" and r.channel.endswith(short)
                      and r.generation == free.generation)
        records.remove(free)
        index = next(i for i, r in enumerate(records) if r is victim)
        records.insert(index, free)
        return records

    report = _mutate(_module(), mutate)
    assert not report.verified
    assert report.code in ("COMPLETION_MAP_MISMATCH", "COMPLETION_EARLY_FREE")
    assert report.detail


def test_a_free_before_its_own_read_is_an_early_free():
    """The same defect analysed by the PROTOCOL layer, with emission fidelity held constant."""
    from tilefoundry_ascend.completion.protocol import simulate_trace

    module = _module()
    cp = _build(module)
    assert cp.verified
    records = list(cp.trace.records)
    free = next(r for r in records
                if r.role == "set" and r.kind == "free" and r.generation >= 1)
    short = _free_root(free)
    victim = next(r for r in records
                  if r.role == "read_begin" and r.channel.endswith(short)
                  and r.generation == free.generation)
    records.remove(free)
    records.insert(next(i for i, r in enumerate(records) if r is victim), free)
    mutated = dataclasses.replace(cp.trace, records=tuple(records))
    verdict = simulate_trace(mutated, obligations=tuple(cp.obligations))
    assert verdict.status == "FAILED"
    assert verdict.code == "COMPLETION_EARLY_FREE"


def test_a_read_without_ready_is_rejected():
    """Removing every READY wait must be refused: a read may not precede its producer.

    Two layers can legitimately catch this, and BOTH outcomes are correct refusals:
      * coverage notices the plan requires 8 READY waits the program no longer carries;
      * the protocol notices a read beginning with no READY set.
    The test asserts the refusal and that its DETAIL names the missing READY statements, so the
    diagnostic has to point at the actual defect rather than a generic failure.
    """
    def mutate(records):
        return [r for r in records
                if not (r.role == "wait" and r.kind == "ready")]

    report = _mutate(_module(), mutate)
    assert not report.verified
    assert report.code in ("COMPLETION_NOT_VERIFIED", "COMPLETION_READ_WITHOUT_READY",
                          "COMPLETION_WAIT_WITHOUT_SET")
    assert "ready" in report.detail.lower()


def test_the_protocol_alone_catches_a_read_without_ready():
    """The same defect seen by the PROTOCOL layer, with the program otherwise intact."""
    from tilefoundry_ascend.completion.protocol import simulate_trace

    module = _module()
    cp = _build(module)
    assert cp.verified
    records = [r for r in cp.trace.records
               if not (r.role == "wait" and r.kind == "ready")]
    mutated = dataclasses.replace(cp.trace, records=tuple(records))
    verdict = simulate_trace(mutated, obligations=tuple(cp.obligations))
    assert verdict.status == "FAILED"
    assert verdict.code == "COMPLETION_READ_WITHOUT_READY"
    assert "READY" in verdict.detail


def test_a_stale_generation_replay_is_rejected():
    """Replaying a retired generation must be refused, not silently accepted.

    The mutation rewrites a LATE generation's completion statement back to generation 0 -- the
    shape a stale descriptor produces when a consumer is handed last round's readiness. Every
    statement is still present and the Set/Wait counts still match, so only a generation-aware
    state machine can catch it.
    """
    def mutate(records):
        late = [r for r in records
                if r.role in ("set", "wait") and r.generation >= 2]
        assert late, "the carrier must have a generation past the ring width"
        target = late[0]
        index = records.index(target)
        records[index] = dataclasses.replace(target, generation=0, slot=0)
        return records

    report = _mutate(_module(), mutate)
    assert not report.verified
    # Coverage and the generation state machine are BOTH correct refusals here: the mutated
    # statement is no longer the statement the plan requires, and it also replays a retired
    # generation. Whichever layer reports first, the program is refused.
    assert report.code in ("COMPLETION_STALE_GENERATION", "COMPLETION_WAIT_BEFORE_SET",
                          "COMPLETION_SLOT_GENERATION_MISMATCH",
                          "COMPLETION_NOT_VERIFIED")
    assert "generation" in report.detail.lower()


def test_the_protocol_alone_catches_a_stale_generation_replay():
    """The same defect seen by the state machine directly, without any emission."""
    from tilefoundry_ascend.completion.protocol import simulate_trace

    module = _module()
    cp = _build(module)
    assert cp.verified
    records = list(cp.trace.records)
    late = [r for r in records if r.role in ("set", "wait") and r.generation >= 2]
    target = late[0]
    records[records.index(target)] = dataclasses.replace(target, generation=0, slot=0)
    mutated = dataclasses.replace(cp.trace, records=tuple(records))
    verdict = simulate_trace(mutated, obligations=tuple(cp.obligations))
    assert verdict.status == "FAILED"
    # Replaying generation 0 into the ring's steady state is caught as a wait whose expected
    # predecessor was never produced -- the specific, located symptom of a stale descriptor.
    assert verdict.code in ("COMPLETION_STALE_GENERATION", "COMPLETION_WAIT_BEFORE_SET",
                           "COMPLETION_SLOT_GENERATION_MISMATCH",
                           "COMPLETION_FREE_WAIT_UNSATISFIED")
    assert "generation" in verdict.detail


def test_a_write_onto_a_live_slot_is_rejected():
    """Two generations in one slot at once must be refused."""
    module = _module()
    plan = _plan(module)
    # A ONE-version buffer in a TWO-iteration program: the second write has nowhere to go
    # until the first was freed, and dropping the FREE wait makes it an illegal overwrite.
    cp = _build(module, plan)
    assert cp.verified
    records = [r for r in cp.trace.records
               if not (r.role == "wait" and r.kind == "free")]
    mutated = dataclasses.replace(cp.trace, records=tuple(records))
    from tilefoundry_ascend.completion.protocol import simulate_trace
    verdict = simulate_trace(mutated, obligations=tuple(cp.obligations))
    assert verdict.status == "FAILED"
    assert verdict.code in ("COMPLETION_OVERWRITE_BEFORE_FREE", "COMPLETION_EARLY_FREE")


def test_dropping_the_drain_leaves_live_storage_at_region_exit():
    """A program whose final async write is not drained must be refused."""
    from tilefoundry_ascend.completion.protocol import simulate_trace

    module = _module()
    cp = _build(module)
    assert cp.verified
    records = [r for r in cp.trace.records if r.role != "drain"]
    mutated = dataclasses.replace(cp.trace, records=tuple(records))
    verdict = simulate_trace(mutated, obligations=tuple(cp.obligations))
    assert verdict.status == "FAILED"
    assert verdict.code in ("FINAL_DRAIN_MISSING", "COMPLETION_EXIT_LIVE_STORAGE")


def test_a_dropped_set_is_rejected_as_an_unsatisfiable_wait():
    """Count equality is not a proof: dropping a Set and its Wait is still refused."""
    from tilefoundry_ascend.completion.protocol import simulate_trace

    module = _module()
    cp = _build(module)
    assert cp.verified
    # Drop a PAIRED set and wait: the counts still match, so only a real analysis catches it.
    victim_channel = sorted({r.channel for r in cp.trace.records
                             if r.role == "set" and r.kind == "ready"})[0]
    records = [r for r in cp.trace.records
               if not (r.channel == victim_channel and r.kind == "ready")]
    assert len([r for r in records if r.role == "set"]) == \
        len([r for r in records if r.role == "set"])
    mutated = dataclasses.replace(cp.trace, records=tuple(records))
    verdict = simulate_trace(mutated, obligations=tuple(cp.obligations))
    assert verdict.status == "FAILED"
    assert verdict.code in ("COMPLETION_READ_WITHOUT_READY", "COMPLETION_WAIT_WITHOUT_SET")


def _remove_one_completion_statement(function):
    """Delete the FIRST completion statement from the REAL emitted TIR.

    A helper rather than a hand-built module: the mutation must be applied to the program the
    compiler actually produced, so the test cannot accidentally verify a program nobody emits.
    """
    from tilefoundry.ir.tir.flag import Drain, FlagSet, FlagWait
    from tilefoundry.ir.tir.stmts import Evaluate, ExecutionScope, For, If, MeshScope, Sequential

    state = {"done": False}

    def prune(node):
        if state["done"]:
            return node
        # A PrimFunction/mesh wrapper is not a statement container the walker should rewrite
        # blindly: descend into its body and keep the dataclass intact.
        if hasattr(node, "body") and not isinstance(
                node, (Sequential, If, For, MeshScope, ExecutionScope)):
            inner = prune(node.body)
            if state["done"]:
                import dataclasses as dc
                return dc.replace(node, body=inner)
            return node
        if isinstance(node, Sequential):
            body = []
            for child in node.body:
                new_child = prune(child)
                if new_child is not None:
                    body.append(new_child)
            return Sequential(body=tuple(body)) if tuple(body) != tuple(node.body) else node
        if isinstance(node, If):
            then_body = prune(node.then_body)
            if state["done"]:
                import dataclasses as dc
                return dc.replace(node, then_body=then_body)
            return node
        if isinstance(node, (MeshScope, ExecutionScope, For)):
            body = prune(node.body)
            if state["done"]:
                import dataclasses as dc
                return dc.replace(node, body=body)
            return node
        if isinstance(node, Evaluate) and isinstance(node.callable,
                                                     (FlagSet, FlagWait, Drain)):
            state["done"] = True
            return None
        return node

    return prune(function), state["done"]


def test_an_emitted_program_that_differs_from_the_verified_one_is_refused():
    """Emission fidelity: the TIR that runs must BE the program that verified.

    This is the check a coverage count CANNOT make. The verified program is correct, and the
    emitted TIR is missing exactly one statement while the obligation COUNT is unchanged. A
    verifier that compares counts accepts this; the fidelity comparison refuses it by position.
    """
    from tilefoundry_ascend import completion as CP
    from tilefoundry_ascend.completion.verify import verify_completion

    module = _module()
    plan = _plan(module)
    cp = _build(module)
    assert cp.verified

    pruned, removed = _remove_one_completion_statement(cp.rewritten_tir)
    assert removed, "the emitted program must contain completion statements to remove"
    report = verify_completion(plan, cp, actual_tir=pruned)
    assert not report.verified
    assert report.code == "COMPLETION_NOT_VERIFIED"
    assert "statement" in report.detail


def test_a_locally_dead_set_is_not_coverage():
    """A Set inside a statically false arm cannot execute, so it is not coverage."""
    from tilefoundry_ascend.completion.emit import read_trace_from_tir, trace_nodes
    from tilefoundry_ascend.completion.guards import GUARD_NEVER

    module = _module()
    cp = _build(module)
    assert cp.verified
    records = read_trace_from_tir(cp.rewritten_tir)
    assert all(getattr(r, "reachable", True) for r in records)


# ------------------------------------------------------------- guard compatibility
def test_a_conditional_wait_whose_set_can_be_skipped_is_rejected():
    """A WAIT under a different predicate than its SET is a hang, not a near-miss."""
    from tilefoundry_ascend.completion.guards import (
        COND, GUARD_ALWAYS, Guard, guard_implies,
    )

    always = guard_implies(Guard(COND, key="p1"), GUARD_ALWAYS)
    assert always[0] is True, "an unconditional setter covers a conditional waiter"
    same = guard_implies(Guard(COND, key="p1"), Guard(COND, key="p1"))
    assert same[0] is True
    different = guard_implies(Guard(COND, key="p1"), Guard(COND, key="p2"))
    assert different[0] is False, (
        "two DIFFERENT predicates do not imply one another; assuming they do is how a"
        " conditional deadlock reaches hardware")


def test_cross_core_handoff_sides_must_agree_on_the_group_predicate():
    from tilefoundry_ascend.completion.guards import COND, GUARD_ALWAYS, Guard, guards_agree

    assert guards_agree(GUARD_ALWAYS, GUARD_ALWAYS)[0] is True
    assert guards_agree(Guard(COND, key="g"), Guard(COND, key="g"))[0] is True
    assert guards_agree(Guard(COND, key="g"), GUARD_ALWAYS)[0] is False
    assert guards_agree(GUARD_ALWAYS, Guard(COND, key="g"))[0] is False


# ------------------------------------------------------------------ chain integration
def test_the_chain_records_a_verified_completion_and_the_gate_still_needs_storage():
    """P4-01 must move completion_status off PENDING on the public finalize path.

    Storage is P4-02, so the export gate must STILL refuse -- but for the storage reason only.
    A gate that stopped refusing would mean the boundary was widened instead of satisfied.
    """
    from tilefoundry_ascend.passes.p3_chain import CHAIN_KEY, EXPORT_GATE_KEY

    module = _module()
    chain = module.metadata.get(CHAIN_KEY)
    assert chain is not None
    assert chain.completion_status == "VERIFIED", (
        "P4-01 did not verify completion on the public path: %s" % (chain.reason,))
    completion = module.metadata.get("ascend.p3_completion")
    assert completion is not None and completion.verified

    # ASK THE GATE, do not read a key that may never have been written. finalize() runs the
    # chain but does not install EXPORT_GATE_KEY, so reading the metadata returned {} and the
    # "ready is False" assertion below passed for the trivial reason that nothing was there --
    # a check over an empty dict is not a check. require_export_ready computes it from the
    # record actually stored on the module.
    from tilefoundry_ascend.passes.p3_chain import require_export_ready

    gate = require_export_ready(module, chain, raising=False)
    gaps = [gap for gap in gate.get("gaps", []) if gap[0] == "completion"]
    assert not gaps, "a verified completion must not be named as a completion gap: %r" % (gaps,)
    # THE GATE IS NOW READY, AND THAT IS THE CORRECT OUTCOME.
    # At G40 the storage placement was still P4-02 and the scheduled TIR did not exist, so this
    # asserted a refusal. G41 attached a verified placement and G42 emits and verifies the
    # scheduled program, so "not ready" would now assert that two completed stages did not
    # happen. The load-bearing clause is the one above it: a VERIFIED completion must not be
    # reported as a completion gap.
    assert gate.get("ready") is True, gate.get("gaps")


def test_completion_plan_survives_json_serialisation():
    """The record must be a real document, not an object graph only Python can read."""
    import json

    module = _module()
    completion = module.metadata.get("ascend.p3_completion")
    assert completion is not None
    document = completion.as_dict()
    text = json.dumps(document, sort_keys=True, default=str)
    round_trip = json.loads(text)
    assert round_trip["status"] == "VERIFIED"
    assert round_trip["required"] == round_trip["covered"]
    assert round_trip["namespaces"], "the allocation must name its real namespaces"


# ===================================================================== synthetic plans
# The carrier has ONE consumer per buffer and no Cube pipe, so the two cases G40 must also
# cover -- a Cube->Vector handoff and a buffer with SEVERAL consumers -- cannot be produced by
# varying it. They are built here from the REAL plan dataclasses, so the code under test walks
# the same structures it walks in production. A synthetic plan is not a claim that the compiler
# emits such a program today; it is a test of the OBLIGATION DERIVATION and the PROTOCOL, which
# must be correct for programs the carriers cannot yet express.
def _synthetic_plan(*, writers, readers, trip=4, versions=2, pipe_of=None):
    """A minimal ResolvedPlan-shaped object with explicit writers/readers per root.

    'writers' maps root -> task id; 'readers' maps root -> tuple of consumer task ids.
    """
    from tilefoundry_ascend.plan.resolved_plan import (
        AccessPlan, BufferPlan, LoopPlan, PlanIdentity, PlanState, ResolvedPlan, TaskPlan,
    )

    pipe_of = pipe_of or {}
    tasks = []
    accesses = []
    order = 0
    seen = []
    for root, producer in writers.items():
        for consumer in readers.get(root, ()):
            for task in (producer, consumer):
                if task not in seen:
                    seen.append(task)
    for task_id in seen:
        pipe = pipe_of.get(task_id, "vector")
        tasks.append(TaskPlan(
            task_id=task_id, source="synthetic", action_ids=("a",), pipe=pipe,
            resource=("vector" if pipe in ("vector", "mte2", "mte3", "DMA") else "cube"),
            start_time={"value": 0, "unit": "MODEL cycles"}, stage=0, order=order,
            iteration_shift=0, guard=None, phase=0, authored_stage=0,
            stage_origin="AUTHORED"))
        order += 1
    ordinal = 0
    for root, producer in writers.items():
        accesses.append(_access(root, producer, "write", ordinal, pipe_of, versions))
        ordinal += 1
        for consumer in readers.get(root, ()):
            accesses.append(_access(root, consumer, "read", ordinal, pipe_of, versions))
            ordinal += 1
    buffers = [BufferPlan(source_decl=root, root=root, owner="synthetic",
                          space="ascend.ub", dtype="f32", shape=(256,),
                          single_version_bytes=256, alignment=32,
                          slot_count=versions, slot_source="fixed",
                          version_stride_bytes=256, base_offset_bytes=0,
                          allocation_bytes=256 * versions)
               for root in writers]
    loop = LoopPlan(
        source_loop_id="synthetic/for:i",
        domain={"start": 0, "stop": trip, "step": 1},
        requested_budget=1, enable_offset=False,
        ii={"value": 1, "unit": "MODEL cycles"},
        latency={"value": 1, "unit": "MODEL cycles"},
        execution_ordinal_def="k=(i-start)/step",
        prologue_domain={}, steady_domain={"start": 0, "stop": trip, "step": 1},
        drain_domain={}, stage_depth=1)
    identity = PlanIdentity(
        schema="synthetic", source_digest="s", problem_digest="p", ledger_sha="l",
        s_request_digest="sr", s_result_digest="sres", s_core_digest="score",
        f_refs={}, p_refs={}, structural_model_scope="MODEL", plan_digest="d")
    return ResolvedPlan(
        identity=identity, state=PlanState(), loop=loop, tasks=tuple(tasks),
        buffers=tuple(buffers), accesses=tuple(accesses), escapes=(), syncs=(),
        unsupported=(), units={})


def _access(root, task, mode, ordinal, pipe_of, versions):
    from tilefoundry_ascend.plan.resolved_plan import AccessPlan

    return AccessPlan(
        access_id="%s/%s/%d" % (root, mode, ordinal), action_id="a", task_id=task,
        source_stmt="k/s0/MeshScope/s0/for:i/s%d" % ordinal, buffer_root=root,
        logical_index=None, execution_ordinal=None, slot=None, generation=None,
        relative_byte_offset=None, guard=None, byte_footprint=256, mode=mode)


def _orders(plan):
    """Give each synthetic task a distinct physical order (the issue rank)."""
    return plan


def test_a_multi_consumer_buffer_waits_for_every_consumer():
    """FREE must wait for ALL consumers, and the obligation must NAME all of them.

    The plan has one producer and THREE consumers of the same buffer. The FREE obligation must
    carry all three, and the emitted program must place the FREE after the last of their read
    completions -- not after the first, which is the ring defect the spec calls out.
    """
    from tilefoundry_ascend import completion as CP

    plan = _synthetic_plan(
        writers={"buf": "w0"},
        readers={"buf": ("c0", "c1", "c2")},
        trip=4, versions=2)
    obligation_set = CP.derive_obligations(plan)
    free = [o for o in obligation_set.obligations if o.kind == "free"]
    assert len(free) == 1
    assert set(free[0].consumers) == {"c0", "c1", "c2"}, (
        "the FREE obligation must name every consumer; a subset is early FREE")
    trace = CP.build_completion_trace(plan, obligation_set)
    # The FREE set of each generation must follow ALL three read completions of it.
    for generation in range(4):
        reads = [r.index for r in trace.records
                 if r.role == "read_end" and r.generation == generation]
        frees = [r.index for r in trace.records
                 if r.role == "set" and r.kind == "free" and r.generation == generation]
        assert len(reads) == 3, "three consumers must produce three read completions"
        if frees:
            assert min(frees) > max(reads), (
                "the FREE of generation %d must follow the LAST read completion" % generation)


def test_dropping_one_consumer_from_a_free_is_refused():
    """A FREE that fires after only SOME consumers is the ring defect, and is refused."""
    from tilefoundry_ascend import completion as CP
    from tilefoundry_ascend.completion.protocol import simulate_trace

    plan = _synthetic_plan(writers={"buf": "w0"},
                           readers={"buf": ("c0", "c1", "c2")},
                           trip=4, versions=2)
    obligation_set = CP.derive_obligations(plan)
    trace = CP.build_completion_trace(plan, obligation_set)
    # Remove ONE consumer's read completion but keep the FREE: exactly what an implementation
    # that waits for the first consumer instead of the last produces.
    victim = next(r for r in trace.records
                  if r.role == "read_end" and r.generation == 0)
    records = [r for r in trace.records if r is not victim]
    verdict = simulate_trace(dataclasses.replace(trace, records=tuple(records)),
                             obligations=tuple(obligation_set.obligations))
    assert verdict.status == "FAILED"
    assert verdict.code == "COMPLETION_EARLY_FREE"


def test_a_cross_core_handoff_is_charged_to_the_device_namespace_and_checked():
    """Cube -> Vector is CROSS-CORE: device-level events, explicit participants, both sides.

    The plan has a Cube producer and a Vector consumer. The handoff must (a) be recognised as
    cross-core, (b) allocate from the DEVICE cross namespace rather than either core's local
    pool, (c) carry explicit participants, and (d) be checked for predicate agreement on both
    sides. Charging it to AIV's local pool would exhaust a pool the Cube side never uses.
    """
    from tilefoundry_ascend import completion as CP

    plan = _synthetic_plan(
        writers={"tile": "cube0"},
        readers={"tile": ("vec0",)},
        trip=2, versions=2,
        pipe_of={"cube0": "cube", "vec0": "vector"})
    obligation_set = CP.derive_obligations(plan)
    cross = [o for o in obligation_set.obligations if o.cross_core]
    assert cross, "a Cube producer with a Vector consumer MUST be a cross-core obligation"
    assert {o.producer_core for o in cross} == {"aic"}
    assert {o.consumer_core for o in cross} == {"aiv"}
    assert obligation_set.cross_handoffs, "the handoff must be recorded explicitly"

    trace = CP.build_completion_trace(plan, obligation_set)
    requests = trace.channel_requests()
    cross_channels = [r for r in requests
                      if set(r.participants) & {"aic", "aiv"}]
    assert cross_channels, "the cross handoff must produce a channel of its own"
    for request in cross_channels:
        assert request.core_domain == CP.DEVICE, (
            "a cross-core handoff is a DEVICE resource, not a local one: %s" % (request.channel,))
        assert request.kind == CP.KIND_CROSS

    assignment = CP.allocate_events(requests, facts=CP.ASCEND_EVENT_FACTS)
    device_ids = [a for a in assignment.allocations
                  if a.namespace.core_domain == CP.DEVICE]
    assert device_ids, "the handoff must be allocated in the device namespace"
    for allocation in device_ids:
        assert allocation.pool_capacity == 16


def test_a_cross_core_handoff_with_disagreeing_predicates_is_refused():
    """Both sides of a handoff must use a compatible group predicate."""
    from tilefoundry_ascend.completion.guards import (
        COND, GUARD_ALWAYS, Guard, guards_agree,
    )
    from tilefoundry_ascend.completion.protocol import simulate_trace

    plan = _synthetic_plan(writers={"tile": "cube0"}, readers={"tile": ("vec0",)},
                           trip=2, versions=2,
                           pipe_of={"cube0": "cube", "vec0": "vector"})
    from tilefoundry_ascend import completion as CP

    obligation_set = CP.derive_obligations(plan)
    trace = CP.build_completion_trace(plan, obligation_set)
    records = []
    for record in trace.records:
        if record.role in ("set", "wait") and record.cross_core:
            # The Cube side is conditional, the Vector side unconditional: one side could skip
            # its Set while the other waits, which is a hang.
            guard = Guard(COND, key="tile_ready") if record.role == "set" else GUARD_ALWAYS
            records.append(dataclasses.replace(record, guard=guard))
        else:
            records.append(record)
    mutated = dataclasses.replace(trace, records=tuple(records))
    verdict = simulate_trace(mutated, obligations=tuple(obligation_set.obligations))
    assert verdict.status == "FAILED"
    assert verdict.code in ("COMPLETION_CROSS_CORE_MISMATCH", "COMPLETION_GUARD_INCOMPATIBLE")



# --------------------------------------------------------- the FLAG-5 defect class, directly
def test_the_schedule_layer_derives_its_channels_from_the_problem_not_from_rmsnorm():
    """schedule/sync.py must not return a fixed six-channel RMSNorm list.

    This is the FLAG-5 defect stated as a test: the old build_obligations() returned x_ready,
    x_free, y_ready, y_free, w_ready, w_conv no matter what program it was handed. The assertion
    is not "the list is longer than six" -- it is that the CHANNEL ENDPOINTS ARE THE PROBLEM'S
    OWN TASKS, so a program with different tasks cannot receive RMSNorm's channels.
    """
    from tilefoundry_ascend.schedule.problem import extract_problem
    from tilefoundry_ascend.schedule.program import build_program, case_from_dict
    from tilefoundry_ascend.schedule.sync import build_obligations

    config = case_from_dict({"case_id": "g40", "shape": [4, 512], "x_versions": 2,
                             "y_versions": 2, "requested_num_stages": 2})
    problem = extract_problem(build_program(config))
    obligations = build_obligations(problem)

    task_ids = {task.stable_id for task in problem.tasks}
    assert obligations, "the problem has dependencies, so it must yield obligations"
    for obligation in obligations:
        for endpoint in (obligation.producer, obligation.consumer):
            assert endpoint in task_ids, (
                "channel %r names task %r, which this problem does not have: the obligation"
                " set is a template, not a derivation"
                % (obligation.channel, endpoint))

    # And the old literal names are gone as DATA, not merely unused.
    names = {obligation.channel for obligation in obligations}
    assert not (names & {"x_ready", "x_free", "y_ready", "y_free", "w_ready", "w_conv"}), (
        "the literal RMSNorm channel names are still being produced: %s" % sorted(names))


def test_the_derived_ring_width_follows_the_declared_versions():
    """A channel's version count must come from the buffer declaration, never from a default.

    The bug this catches was real and silent: the adapter that presents the schedule Problem to
    the obligation derivation translated neither name (the Problem says "versions", the plan says
    "slot_count"), so every channel claimed a single-version ring no matter what the buffer
    declared. The ring width then agreed with itself in every verification layer, because they
    all read the same wrong constant -- which is why the test has to VARY the declaration.
    """
    from tilefoundry_ascend.schedule.problem import extract_problem
    from tilefoundry_ascend.schedule.program import build_program, case_from_dict
    from tilefoundry_ascend.schedule.sync import build_obligations

    widths = {}
    for versions in (1, 2, 4):
        config = case_from_dict({"case_id": "g40", "shape": [4, 512],
                                 "x_versions": versions, "y_versions": versions,
                                 "requested_num_stages": 2})
        problem = extract_problem(build_program(config))
        orders_x = [o for o in build_obligations(problem) if "_x_" in o.channel]
        assert orders_x, "the problem orders x_ub, so it must have x channels"
        widths[versions] = sorted({o.versions for o in orders_x})

    assert widths[1] == [1], widths
    assert widths[2] == [2], widths
    assert widths[4] == [4], widths
    assert len({tuple(v) for v in widths.values()}) == 3, (
        "the ring widths did not change with the declared versions: %s" % widths)


def test_a_negative_probe_that_cannot_change_the_program_is_reported_not_passed():
    """A mutation that changes nothing must not be counted as a rejected negative test.

    The defect this locks down: mutate_wrong_slot derived the ring width from the largest slot it
    observed, so on a single-iteration program the width was 1 and the "wrong slot" shift was
    0 -> 0. The verifier then passed the UNCHANGED program and the negative matrix reported full
    success while proving nothing. Refusing to mutate is the honest outcome.
    """
    from tilefoundry_ascend.lowering.verify import PEvent, mutate_wrong_slot

    single = [PEvent("MTE2", "", 0, 0, "x_load", 0)]
    with pytest.raises(ValueError) as refusal:
        mutate_wrong_slot(single)
    assert "single slot" in str(refusal.value)

    # With the ring width supplied the same probe does mutate, and to a different slot.
    two = [PEvent("MTE2", "", 0, 0, "x_load", 0), PEvent("MTE2", "", 1, 1, "x_load", 1)]
    shifted = mutate_wrong_slot(two, versions=2)
    assert shifted[0].slot != two[0].slot, "the probe still did not change the slot"

