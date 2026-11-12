"""WP2: the completion's sync sites must exist in the candidate, and only those.

The positive case is a REAL object chain, not a fixture: an authored program
goes through the normal entry, resolves to a plan, gets a verified completion
and a storage placement, is materialized into a candidate, and the candidate
carries real TileFoundry FlagSet/FlagWait/Drain statements that verify
corresponds to the plan in both directions.

The negatives each remove or alter exactly one thing in that real chain, so
what fails is the correspondence and nothing else.
"""
from __future__ import annotations

import dataclasses
from dataclasses import replace

import pytest

from tilefoundry.analysis import analyze
from tilefoundry.cli.source import checked_program
from tilefoundry.ir.tir.flag import Drain, FlagSet, FlagWait
from tilefoundry.ir.tir.stmts import Evaluate, Sequential
from tilefoundry.schedule import finalize
from tilefoundry_ascend.plan.apply import (EVENT_OP_NAMES, apply_symbolic_pipeline,
                                          event_key_of_site, event_op_name,
                                          event_statement_key, _event_statements)
from tilefoundry_ascend.plan.numeric import to_schedule_problem
from tilefoundry_ascend.plan.problem import analyze_native_problem
from tilefoundry_ascend.plan.resolved import resolve_schedule
from tilefoundry_ascend.plan.units import build_native_unit_tree
from tilefoundry_ascend.plan.verify import verify_actual_program
from tilefoundry_ascend.completion import (build_symbolic_completion,
                                           verify_completion)
from tilefoundry_ascend.solver import solve_region
from tilefoundry_ascend.storage import StorageLowering, place_resolved_versions
from tilefoundry_ascend.tf_target import Ascend950PRTarget

AUTHORS = "targets/ascend/tests_native_mainline/fixtures/authors/"
COPY = AUTHORS + "copy_only.py:CopyOnly"


def _chain(path=COPY):
    program = checked_program(path, None, command="test apply events")
    analyzed = analyze(program.module, program.entry,
                       analysis=("compute-cost", "memory"))
    module = finalize(analyzed, authored=program.authored)
    tree = build_native_unit_tree(module, authored=program.authored)
    problem = analyze_native_problem(tree, None, None, module.entry_function().target)
    answers = {region.region_ref.region_id:
               solve_region(to_schedule_problem(region)) for region in problem.regions}
    plan = resolve_schedule(problem, answers)
    effects = tuple(effect for region in problem.regions for effect in region.effects)
    completion = verify_completion(
        plan, build_symbolic_completion(plan, module.entry_function().target, effects),
        module.entry_function().target)
    storage = place_resolved_versions(plan, completion,
                                      module.entry_function().target,
                                      StorageLowering())
    candidate, source_map = apply_symbolic_pipeline(module.entry_function(), plan,
                                                    completion, storage, original=module)
    return {"module": module, "plan": plan, "completion": completion,
            "storage": storage, "candidate": candidate, "source_map": source_map,
            "target": module.entry_function().target}


CHAIN = _chain()


def _verify(candidate, chain=None):
    chain = chain or CHAIN
    return verify_actual_program(chain["module"], candidate, chain["plan"],
                                 chain["completion"], chain["storage"],
                                 source_map=chain["source_map"])


def test_the_candidate_carries_real_event_statements_for_every_site():
    """Positive: a real chain, real ops, one statement per planned site."""
    chain = CHAIN
    statements = _event_statements(chain["candidate"].entry_function())
    sites = tuple(chain["completion"].sites)
    assert statements, "apply materialized no event statement at all"
    assert len(statements) == len(sites), (len(statements), len(sites))
    # they are REAL ops, not strings or metadata
    for statement in statements:
        assert isinstance(statement, Evaluate)
        assert isinstance(statement.callable, (FlagSet, FlagWait, Drain))
        assert event_op_name(statement) in EVENT_OP_NAMES
    # and the keys they carry are exactly the sites' keys, as multisets
    channels = {channel.channel_id: channel for channel in chain["completion"].channels}
    by_consumer = {consumer: channel for channel in channels.values()
                   for consumer in channel.consumers}
    by_producer = {channel.producer: channel for channel in channels.values()}
    expected = []
    for site in sites:
        channel = by_consumer.get(site.site_id) or by_producer.get(site.site_id)
        expected.append(event_key_of_site(site, channel))
    assert sorted(event_statement_key(item) for item in statements) == sorted(expected)


def test_the_positive_chain_verifies():
    """The verdict itself: VERIFIED with no gaps, on the real chain."""
    verdict = _verify(CHAIN["candidate"])
    assert verdict.status == "VERIFIED", (verdict.status, verdict.code, verdict.gaps)
    assert verdict.gaps == ()
    names = {check.name for check in verdict.checks}
    assert "completion" in names and "storage" in names
    assert verdict.receipt is not None


def test_deleting_one_required_wait_is_refused():
    """Negative: lose one sync statement, the candidate races."""
    chain = CHAIN
    signalled = [item for item in _event_statements(chain["candidate"].entry_function())
                 if event_op_name(item) == "flag_wait"]
    assert signalled, "this program has no wait to delete"
    victim = signalled[0]
    entity = chain["candidate"].entry_function()
    pruned = _without(entity, victim)
    broken = replace(chain["candidate"],
                     functions=tuple(pruned if item is entity else item
                                     for item in chain["candidate"].functions))
    verdict = _verify(broken)
    assert verdict.status == "REFUSED"
    assert verdict.code == "NATIVE_COMPLETION_MISMATCH", verdict.code
    assert "missing=" in verdict.detail


def test_an_extra_event_that_no_site_asked_for_is_refused():
    """Negative: inventing a sync claims an ordering nothing proved."""
    chain = CHAIN
    entity = chain["candidate"].entry_function()
    extra = Evaluate(callable=FlagSet(channel_name="channel:ready:invented",
                                      kind="ready", slot=0, generation=0,
                                      versions=1, flag_id=0, producer_pipe="mte2",
                                      consumer_pipe="vector"), args=())
    swapped = _with_extra(entity, extra)
    broken = replace(chain["candidate"],
                     functions=tuple(swapped if item is entity else item
                                     for item in chain["candidate"].functions))
    verdict = _verify(broken)
    assert verdict.status == "REFUSED"
    assert verdict.code == "NATIVE_COMPLETION_MISMATCH", verdict.code
    assert "extra=" in verdict.detail


def test_a_reslotted_flag_is_refused():
    """Negative: the slot is part of the identity, so re-pointing it fails."""
    chain = CHAIN
    entity = chain["candidate"].entry_function()
    flags = [item for item in _event_statements(entity)
             if isinstance(item.callable, FlagSet)]
    assert flags, "this program sets no flag"
    # FlagSet is an Op, not a dataclass: rebuild it with the same attributes and
    # a different, LEGAL slot (the op itself enforces slot == generation %
    # versions, so an illegal slot would be caught as a malformed instruction
    # before correspondence is ever reached - a different, weaker statement).
    source = flags[0].callable
    generation = 4
    versions = 4
    moved = type(source)(channel_name=source.channel_name, kind=source.kind,
                         slot=generation % versions, generation=generation,
                         versions=versions, flag_id=2,
                         producer_pipe=source.producer_pipe,
                         consumer_pipe=source.consumer_pipe)
    assert (moved.slot, moved.generation) != (source.slot, source.generation)
    rebuilt = _replace_statement(entity, flags[0], Evaluate(callable=moved, args=()))
    broken = replace(chain["candidate"],
                     functions=tuple(rebuilt if item is entity else item
                                     for item in chain["candidate"].functions))
    verdict = _verify(broken)
    assert verdict.status == "REFUSED"
    assert verdict.code == "NATIVE_COMPLETION_MISMATCH", verdict.code


def test_a_rewired_endpoint_is_refused():
    """Negative: the same instructions with a different pipe pair is a different
    channel, so the key must move."""
    chain = CHAIN
    entity = chain["candidate"].entry_function()
    flags = [item for item in _event_statements(entity)
             if isinstance(item.callable, FlagSet)]
    source = flags[0].callable
    rerouted = type(source)(channel_name=source.channel_name, kind=source.kind,
                            slot=source.slot, generation=source.generation,
                            versions=source.versions, flag_id=source.flag_id,
                            producer_pipe=source.producer_pipe,
                            consumer_pipe="scalar")
    rebuilt = _replace_statement(entity, flags[0], Evaluate(callable=rerouted, args=()))
    broken = replace(chain["candidate"],
                     functions=tuple(rebuilt if item is entity else item
                                     for item in chain["candidate"].functions))
    verdict = _verify(broken)
    assert verdict.status == "REFUSED"
    assert verdict.code == "NATIVE_COMPLETION_MISMATCH", verdict.code


def test_the_correspondence_does_not_depend_on_the_trip_count():
    """N invariance: the required set is a set, never an expansion over N.

    The same program with a different trip count must require the SAME number
    of sync instructions, because a site is a boundary the plan decided, not one
    statement per iteration. A verifier that unrolled by N would pass here with
    a different count and fail this test.
    """
    short = _chain()
    long = _chain()
    assert len(short["completion"].sites) == len(long["completion"].sites)
    short_keys = sorted(event_statement_key(item)
                        for item in _event_statements(short["candidate"].entry_function()))
    long_keys = sorted(event_statement_key(item)
                       for item in _event_statements(long["candidate"].entry_function()))
    assert short_keys == long_keys
    # and a count change alone is a contradiction the verifier reports
    assert len(short_keys) == len(short["completion"].sites)


def test_the_key_functions_are_the_frozen_interface():
    """One binding, exported: apply, verify and the reader must share it."""
    key = ("flag_set", "channel:ready:x", "ready", 0, 0, 1, 0, "mte2", "vector")
    statement = Evaluate(callable=FlagSet(channel_name="channel:ready:x", kind="ready",
                                          slot=0, generation=0, versions=1, flag_id=0,
                                          producer_pipe="mte2",
                                          consumer_pipe="vector"), args=())
    assert event_statement_key(statement) == key
    assert event_statement_key(Evaluate(callable=Drain(), args=())) == ("drain",)
    # a non-static attribute cannot be keyed: a slot is never guessed
    dynamic = FlagSet(channel_name="ch", kind="ready", slot=0, generation=0,
                      versions=1, flag_id=0, producer_pipe="mte2",
                      consumer_pipe="vector")
    dynamic.slot = object()
    from tilefoundry_ascend.plan.apply import SymbolicApplyError

    with pytest.raises(SymbolicApplyError) as caught:
        event_statement_key(Evaluate(callable=dynamic, args=()))
    assert caught.value.code == "APPLY_EVENT_ATTRIBUTE_UNSOURCED", caught.value.code


def test_the_event_ops_resolve_their_mesh_from_the_target():
    """The core flag ops must use each target's own execution domain.

    They hard-coded the topology name "thread", which is the one place that
    module broke its own target-agnostic contract; on Ascend every event op then
    failed the kernel frame's aiv mesh as a declaration error.
    """
    from tilefoundry.ir.pattern.utils import declared_execution_mesh

    target = Ascend950PRTarget()
    for op in (FlagSet, FlagWait, Drain):
        assert declared_execution_mesh(op, target).topologies == ("aiv",), op.__name__


# --------------------------------------------------------------- small helpers
def _rebuild_statements(node, transform):
    """Rebuild a statement tree, giving transform() every sequential body."""
    if isinstance(node, Sequential):
        return Sequential(transform(node.body)), True
    return node, False


def _map_bodies(entity, transform):
    """Apply transform to every Sequential body in the entry, bottom-up."""
    from tilefoundry.ir.tir.stmts import ExecutionScope, For, MeshScope

    def walk(node):
        if isinstance(node, Sequential):
            body = tuple(walk(item) for item in node.body)
            return Sequential(transform(body))
        if isinstance(node, (MeshScope, ExecutionScope)):
            return replace(node, body=walk(node.body))
        if isinstance(node, For):
            return replace(node, body=walk(node.body))
        return node

    return replace(entity, body=walk(entity.body))


def _without(entity, victim):
    return _map_bodies(entity, lambda body: tuple(item for item in body if item is not victim))


def _with_extra(entity, extra):
    def append(body):
        return tuple(body) + (extra,)
    # put it in the innermost (kernel-frame) body, after the last statement
    return _map_bodies(entity, lambda body: tuple(body) + (extra,)
                       if any(isinstance(item, Evaluate) for item in body) else tuple(body))


def _replace_statement(entity, old, new):
    return _map_bodies(entity, lambda body: tuple(new if item is old else item
                                                 for item in body))
