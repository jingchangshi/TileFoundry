"""Symbolic completion over the real resolved plan, and the ring protocol.

The first two tests drive the ONE normal entry (checked_program -> analyze ->
finalize -> build_native_unit_tree -> analyze_native_problem), project the real
region problem with the shared numeric projection, solve it with the shared
solver, and resolve the one plan. Nothing here hand-builds TIR, hand-writes a
schedule number, or skips a stage.

The V=2 case is a SYNTHETIC problem (a three-task ring: one DMA writer and two
readers on different pipes) built from the real typed records, because no
authored fixture in this slice declares a Fixed(2) buffer yet. It states the
protocol the plan cannot show by itself: writer k waits FREE(k-V), every maximal
reader publishes its own FREE channel, and the tail is one symbolic window.

No test here enumerates generations: a template that grew with N would fail the
count and identity assertions.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tilefoundry.analysis import analyze
from tilefoundry.cli.source import checked_program
from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.types import DType, Layout, StorageKind, TensorType
from tilefoundry.schedule import finalize

from tilefoundry_ascend.completion import (
    AFTER_TASK,
    BEFORE_TASK,
    EXIT,
    FREE,
    READY,
    REGION_EXIT,
    CompletionError,
    CompletionPlan,
    CompletionVerification,
    build_symbolic_completion,
    completion_identity,
    free_wait_generation,
    free_wait_guard,
    region_ordinal,
    region_tail_guard,
    region_trip_count,
    require_completion_verified,
    tail_window,
    verify_completion,
)
from tilefoundry_ascend.completion.plan import _static_int, expr_equal
from tilefoundry_ascend.plan.numeric import to_schedule_problem
from tilefoundry_ascend.plan.problem import (
    Capacity,
    Dependence,
    LogicalBuffer,
    RegionProblem,
    NormalizedProblem,
    ResourceFact,
    Reuse,
    Task,
    analyze_native_problem,
)
from tilefoundry_ascend.plan.resolved import (
    BufferPlan,
    RegionPlan,
    ResolvedPlan,
    TaskPlan,
    resolve_schedule,
)
from tilefoundry_ascend.plan.units import (
    AccessFact,
    Auto,
    Effect,
    Fixed,
    OwnerKey,
    Region,
    ResourceKey,
    ResourceUse,
    StorageKey,
    TRUE_GUARD,
    build_native_unit_tree,
)
from tilefoundry_ascend.solver import solve_region
from tilefoundry_ascend.tf_target import Ascend950PRTarget

AUTHORS = Path(__file__).resolve().parent / "fixtures" / "authors"
FIXTURE = AUTHORS / "minimal_aiv.py"
INDEPENDENT_UB = "%s:IndependentUB" % (AUTHORS / "independent_ub.py",)
SOURCE = "%s:MinimalAivPipeline" % (FIXTURE,)
I64 = TensorType.umat_scalar()


# --------------------------------------------------------------- real chain
def _real_problem(source=SOURCE):
    program = checked_program(source, None, command="test completion")
    analyzed = analyze(program.module, program.entry, analysis=("compute-cost", "memory"))
    module = finalize(analyzed, authored=program.authored)
    tree = build_native_unit_tree(module, authored=program.authored)
    target = module.entry_function().target
    return analyze_native_problem(tree, None, None, target), target


def _loop_region(problem):
    for region in problem.regions:
        if region.region_ref.induction_var_ref is not None:
            return region
    raise AssertionError("the author fixture declares no loop region")


def _real_plan(source=SOURCE):
    problem, target = _real_problem(source)
    region = _loop_region(problem)
    assignment = solve_region(to_schedule_problem(region))
    assert assignment.status == "SAT", assignment.detail
    plan = resolve_schedule(replace(problem, regions=(region,)), assignment)
    return plan, region, target


def _real_plan_bound(source=INDEPENDENT_UB):
    """The real chain to one ResolvedPlan bound to its own problem.

    checked_program -> analyze -> finalize -> build_native_unit_tree ->
    analyze_native_problem -> solve every region -> resolve_schedule. Nothing is
    hand-assembled: the plan the completion is built from is the plan the solver
    produced, and its problem_ref is that problem object.
    """
    program = checked_program(source, None, command="test completion")
    analyzed = analyze(program.module, program.entry, analysis=("compute-cost", "memory"))
    module = finalize(analyzed, authored=program.authored)
    tree = build_native_unit_tree(module, authored=program.authored)
    target = module.entry_function().target
    problem = analyze_native_problem(tree, None, None, target)
    assignments = {region.region_ref.region_id: solve_region(to_schedule_problem(region))
                   for region in problem.regions}
    return resolve_schedule(problem, assignments), target


def _variant(tmp_path, trips, name="variant.py"):
    text = FIXTURE.read_text(encoding="utf-8")
    assert "tf.pipelined(4," in text
    path = tmp_path / name
    path.write_text(text.replace("tf.pipelined(4,", "tf.pipelined(%d," % trips),
                     encoding="utf-8")
    return "%s:MinimalAivPipeline" % (path,)


def _task_ids(plan):
    mapping = {}
    for region_plan in plan.regions:
        for task in region_plan.tasks:
            mapping[id(task.task_ref)] = task.task_id
    return mapping


def _channels(completion, reason):
    return [channel for channel in completion.channels if channel.reason == reason]


def _site(completion, site_id):
    for site in completion.sites:
        if site.site_id == site_id:
            return site
    raise AssertionError("no site %r" % (site_id,))


def _window(completion, channel):
    tails = [_site(completion, consumer) for consumer in channel.consumers
             if _site(completion, consumer).anchor == REGION_EXIT]
    assert len(tails) == 1
    return tail_window(tails[0])


# ------------------------------------------------------------- synthetic V=2
def _synthetic_region(trips, versions):
    owner = OwnerKey(program_group=None, core_kind="AIV")
    root = Var(name="tile",
               type=TensorType(shape=(64,), dtype=DType.f32,
                               layout=Layout((64,), (1,)), storage=StorageKind.SMEM))
    region = Region(
        region_id="region:synthetic",
        node_ref=object(),
        parent_id=None,
        direct_children=(),
        induction_var_ref=object(),
        lower_expr=Constant(type=I64, value=0),
        stop_expr=Constant(type=I64, value=trips),
        step_expr=Constant(type=I64, value=1),
        binding_summary=None,
        activation_guard=TRUE_GUARD,
    )
    plan_tasks = (
        TaskPlan(task_ref=object(), task_id="unit:0", authored_stage=None,
                 resolved_stage=0, phase=0, start=0, issue_rank=0),
        TaskPlan(task_ref=object(), task_id="unit:1", authored_stage=None,
                 resolved_stage=0, phase=1, start=1, issue_rank=1),
        TaskPlan(task_ref=object(), task_id="unit:2", authored_stage=None,
                 resolved_stage=0, phase=2, start=2, issue_rank=2),
    )
    tasks = tuple(
        Task(task_id=task.task_id, unit_ref=task.task_ref,
             uses=(ResourceUse(ResourceKey(owner, pipe, 0), 1, 0),),
             latency=1, source_order=index, authored_stage=None)
        for index, (task, pipe) in enumerate(zip(plan_tasks, ("mte2", "vector", "scalar")))
    )
    accesses = (
        AccessFact(access_id="access:write", task_id="unit:0", source_ref=object(),
                   operand_index=1, root_ref=root, view_ref=root, relation=None,
                   mode="WRITE", owner=owner, guard=TRUE_GUARD, generation=Constant(type=I64, value=0)),
        AccessFact(access_id="access:read-vector", task_id="unit:1", source_ref=object(),
                   operand_index=0, root_ref=root, view_ref=root, relation=None,
                   mode="READ", owner=owner, guard=TRUE_GUARD, generation=Constant(type=I64, value=0)),
        AccessFact(access_id="access:read-scalar", task_id="unit:2", source_ref=object(),
                   operand_index=0, root_ref=root, view_ref=root, relation=None,
                   mode="READ", owner=owner, guard=TRUE_GUARD, generation=Constant(type=I64, value=0)),
    )
    effects = (
        Effect(effect_id="effect:write", task_id="unit:0", source_ref=object(),
               kind="MEMORY", access_ids=("access:write",), state_ref=None,
               generation=Constant(type=I64, value=0), guard=TRUE_GUARD,
               completion="WRITE_VISIBLE", pipe="mte2"),
        Effect(effect_id="effect:vector", task_id="unit:1", source_ref=object(),
               kind="MEMORY", access_ids=("access:read-vector",), state_ref=None,
               generation=Constant(type=I64, value=0), guard=TRUE_GUARD,
               completion="READ_COMPLETE", pipe="vector"),
        Effect(effect_id="effect:scalar", task_id="unit:2", source_ref=object(),
               kind="MEMORY", access_ids=("access:read-scalar",), state_ref=None,
               generation=Constant(type=I64, value=0), guard=TRUE_GUARD,
               completion="READ_COMPLETE", pipe="scalar"),
    )
    buffer = LogicalBuffer(
        buffer_id="buffer:tile", root_ref=root, allocation_ref=root, owner=owner,
        space="smem", dtype=DType.f32, shape=(64,), layout=None,
        domain=Fixed(versions), generation_delta=0, slot_bytes=256,
        alignment_bytes=4, capacity_key=StorageKey("smem", owner), initial_values=(),
        terminal_readers=("access:read-vector", "access:read-scalar"),
    )
    reuse = Reuse(last_reader="access:read-vector", next_writer="access:write",
                  buffer="buffer:tile", delay=0, distance_offset=0)
    problem_region = RegionProblem(
        region_ref=region, tasks=tasks, buffers=(buffer,), accesses=accesses,
        effects=effects,
        resources=tuple(ResourceFact(key=ResourceKey(owner, pipe, 0), capacity=1,
                                     source="synthetic test fixture",
                                     status="REFERENCE_IMPLEMENTATION")
                        for pipe in ("mte2", "vector", "scalar")),
        owners=(owner,), pipes=("mte2", "vector", "scalar"),
        edges=(Dependence(producer="unit:0", consumer="unit:1", distance=0, delay=0,
                          kind="RAW", effect_ids=("effect:write", "effect:vector")),
               Dependence(producer="unit:0", consumer="unit:2", distance=0, delay=0,
                          kind="RAW", effect_ids=("effect:write", "effect:scalar"))),
        reuse=(reuse,),
        capacities=(Capacity(storage_key=StorageKey("smem", owner),
                             buffers=("buffer:tile",), usable_bytes=256 * 1024,
                             reserved_bytes=0, source="synthetic test fixture",
                             status="REFERENCE_IMPLEMENTATION"),),
        issue_after=(), manual_pipe_order=(), enable_offset=False, max_stage=1,
        fixed_ii=None, ii_min=1, ii_max=8,
    )
    region_plan = RegionPlan(
        region_ref=region, lower_expr=region.lower_expr, stop_expr=region.stop_expr,
        step_expr=region.step_expr, ii=4, origin=0, max_shift=0,
        tasks=plan_tasks,
        buffers=(BufferPlan(buffer_ref=root, buffer_id="buffer:tile",
                            domain=Fixed(versions), resolved_versions=versions),),
        identity="sha256:synthetic-region",
    )
    problem = NormalizedProblem(
        # completion reads the region's accesses/effects/edges/reuse and nothing
        # else from this record, so the fields it does not read are left unset
        # rather than fabricated: a fixture may be incomplete, it may not lie.
        source_snapshot=None,
        target_identity="sha256:synthetic-target",
        model_version="sha256:synthetic-model",
        unit_tree=None,
        regions=(problem_region,),
        coverage=None,
        problem_identity="sha256:synthetic-problem",
    )
    plan = ResolvedPlan(
        source_identity="sha256:synthetic-source",
        target_identity="sha256:synthetic-target",
        model_version="sha256:synthetic-model",
        regions=(region_plan,),
        task_to_native_source={task.task_id: task.unit_ref for task in tasks},
        feasibility="SAT",
        optimality=None,
        plan_identity="sha256:synthetic-plan",
        problem_ref=problem,
        problem_identity="sha256:synthetic-problem",
        target_ref=Ascend950PRTarget(),
    )
    return plan, problem_region, Ascend950PRTarget()


# ------------------------------------------------------------------- positive
def test_real_program_completion_verifies_and_authorizes():
    """independent_ub through the ONE entry: build, verify, require, no skips."""
    plan, target = _real_plan_bound()
    effects = tuple(effect for region in plan.problem_ref.regions
                    for effect in region.effects)
    completion = build_symbolic_completion(plan, target, effects)
    assert completion.verification is None, "construction is not certification"

    verified = verify_completion(plan, completion, target)
    assert verified.verification.status == "VERIFIED", verified.refusals
    assert len(verified.sites) == 9 and len(verified.channels) == 5
    consumed = require_completion_verified(verified)
    assert consumed.identity == verified.identity


def test_channels_are_two_way_bound_to_declared_dependences():
    """Every READY channel answers a RAW edge, and every edge has a channel.

    The obligation set is re-enumerated from plan.problem_ref, so this pins the
    correspondence in both directions: none of these channels is decoration, and
    no declared dependence is left unprotected.
    """
    plan, target = _real_plan_bound()
    effects = tuple(effect for region in plan.problem_ref.regions
                    for effect in region.effects)
    verified = verify_completion(plan, build_symbolic_completion(plan, target, effects),
                                 target)
    edges = {(edge.producer, edge.consumer) for region in plan.problem_ref.regions
             for edge in region.edges}
    task_of = {id(task.task_ref): task.task_id for region in plan.regions
               for task in region.tasks}
    ready = set()
    for channel in verified.channels:
        if channel.reason != READY:
            continue
        producer = _site(verified, channel.producer)
        for consumer in channel.consumers:
            ready.add((task_of[id(producer.task_ref)], task_of[id(_site(
                verified, consumer).task_ref)]))
    assert ready == edges, (sorted(ready), sorted(edges))


def test_forgeable_receipt_field_grants_nothing():
    """A hand-built receipt whose every identity field matches still authorizes nothing.

    Three separate things must fail for consumption, and this pins that none of
    them can be satisfied by writing fields:
      1. content never verified - absent from the registry;
      2. content verified, then changed - the changed content is a different
         identity and is absent even with a receipt pasted onto it;
      3. the tampered content verified afresh - refused, so no entry is created.
    The tamper rewrites the FREE writer's wait generation from
    ordinal-(versions+offset) to the plain ordinal: a writer would then overwrite
    a slot whose read has not landed. Structural rules alone would still pass the
    surrounding guards, which is why (3) matters.
    """
    plan, region, target = _synthetic_region(trips=4, versions=2)
    effects = tuple(effect for entry in plan.problem_ref.regions
                    for effect in entry.effects)
    completion = build_symbolic_completion(plan, target, effects)
    free = _channels(completion, FREE)[0]
    wait_id = next(consumer for consumer in free.consumers
                   if _site(completion, consumer).anchor == BEFORE_TASK)
    tampered = replace(
        completion,
        sites=tuple(replace(site, generation=site.task_ordinal)
                    if site.site_id == wait_id else site for site in completion.sites))
    tampered = replace(tampered, identity=completion_identity(tampered))
    # The tamper MUST have changed content identity: otherwise require() would be
    # looking up the original, honest entry and this test would prove nothing.
    assert tampered.identity != completion_identity(completion), (
        "a no-op tamper: the generation rewrite did not change content identity")

    with pytest.raises(CompletionError) as first:
        require_completion_verified(tampered)
    assert first.value.code == "COMPLETION_NOT_VERIFIED"

    # Authority now exists - for the UNtampered content's identity only.
    honest = verify_completion(plan, completion, target)
    forged = replace(tampered, verification=CompletionVerification(
        status="VERIFIED", completion_identity=tampered.identity,
        plan_identity=plan.plan_identity, target_identity=tampered.target_identity))
    assert forged.identity != honest.identity
    assert forged.verification.plan_identity == honest.plan_identity
    with pytest.raises(CompletionError) as second:
        require_completion_verified(forged)
    assert second.value.code == "COMPLETION_NOT_VERIFIED"

    # And the tampered content cannot earn authority by being verified either.
    with pytest.raises(CompletionError) as third:
        verify_completion(plan, tampered, target)
    assert third.value.code == "COMPLETION_GENERATION_MISMATCH"
    with pytest.raises(CompletionError):
        require_completion_verified(tampered)


# ------------------------------------------------------------------ finite in N
def test_ring_template_does_not_grow_with_trip_count():
    """V=2 ring: sites/channels are constant for every N, tail stays symbolic.

    This is the property the whole slice exists for. A template that enumerated
    generations would fail here by count, not by prose.
    """
    counts = {}
    for trips in (1, 2, 3, 4, 8, 64):
        plan, region, target = _synthetic_region(trips=trips, versions=2)
        effects = tuple(effect for entry in plan.problem_ref.regions
                        for effect in entry.effects)
        completion = build_symbolic_completion(plan, target, effects)
        verified = verify_completion(plan, completion, target)
        counts[trips] = (len(verified.sites), len(verified.channels))
        free = _channels(verified, FREE)
        assert free, "a looped ring must publish at least one FREE obligation"
        windows = [_window(verified, channel) for channel in free]
        assert all(window is not None for window in windows), \
            "the tail must stay one symbolic window, never an expanded list"
    assert len(set(counts.values())) == 1, counts
    # N=0 states no obligation at all: an empty domain executes nothing.
    plan, region, target = _synthetic_region(trips=0, versions=2)
    empty = build_symbolic_completion(plan, target, tuple(
        effect for entry in plan.problem_ref.regions for effect in entry.effects))
    assert empty.sites == () and empty.channels == ()


def test_deleted_free_channel_is_refused_by_the_reenumerated_obligation():
    """Removing a whole channel must fail even though the rest stays balanced.

    The channel, its publish site, the writer wait and the tail site are dropped
    together, so no structural rule is broken: only re-enumerating the problem's
    obligations shows that a maximal terminal reader no longer publishes FREE.
    """
    plan, region, target = _synthetic_region(trips=4, versions=2)
    effects = tuple(effect for entry in plan.problem_ref.regions
                    for effect in entry.effects)
    completion = build_symbolic_completion(plan, target, effects)
    victim = _channels(completion, FREE)[-1]
    dropped = set(victim.consumers) | {victim.producer}
    trimmed = replace(
        completion,
        channels=tuple(ch for ch in completion.channels if ch != victim),
        sites=tuple(site for site in completion.sites if site.site_id not in dropped),
        exit_site_ids=tuple(site_id for site_id in completion.exit_site_ids
                            if site_id not in dropped))
    # Recomputed identity: the digest only describes the content it is given, so a
    # consistent recomputation must NOT clear verification. Identity is never a
    # completeness proof.
    trimmed = replace(trimmed, identity=completion_identity(trimmed))
    assert trimmed.identity != completion_identity(completion), \
        "a no-op tamper: dropping the channel did not change content identity"
    with pytest.raises(CompletionError) as caught:
        verify_completion(plan, trimmed, target)
    assert caught.value.code == "COMPLETION_OBLIGATION_MISSING"
    assert victim.channel_id not in str(caught.value.expected)


def test_reuse_offset_is_checked_against_the_problem_not_itself():
    """An edited distance_offset with consistent guards must still be refused.

    The channel's offset becomes 1 AND the writer wait's guard/generation are
    rebuilt to the offset-1 forms, so every internal-consistency rule passes.
    Only the problem's own reuse record says the offset is 0.
    """
    plan, region, target = _synthetic_region(trips=4, versions=2)
    effects = tuple(effect for entry in plan.problem_ref.regions
                    for effect in entry.effects)
    completion = build_symbolic_completion(plan, target, effects)
    free = _channels(completion, FREE)[0]
    region_plan = plan.regions[0]
    versions, offset = free.versions, 1
    rebuilt_guard = free_wait_guard(region_plan.region_ref.region_id, versions, offset,
                                   region_trip_count(region_plan))
    rebuilt_generation = free_wait_generation(region_plan.region_ref.region_id,
                                             versions, offset)
    wait_id = next(consumer for consumer in free.consumers
                   if _site(completion, consumer).anchor == BEFORE_TASK)
    tail_id = next(consumer for consumer in free.consumers
                   if _site(completion, consumer).anchor == REGION_EXIT)
    offset_one = replace(free, distance_offset=offset)
    sites = []
    for site in completion.sites:
        if site.site_id == wait_id:
            site = replace(site, guard=rebuilt_guard, generation=rebuilt_generation)
        sites.append(site)
    # The tail window is rebuilt for the forged offset as well, so every
    # internal-consistency rule in _check_channel passes. What still refuses is
    # the problem's own reuse record: it says distance_offset == 0.
    tail_guard = region_tail_guard(region_plan.region_ref.region_id,
                                   region_trip_count(region_plan), versions, offset)
    for index, site in enumerate(sites):
        if site.site_id == tail_id:
            sites[index] = replace(site, guard=tail_guard)
    edited = replace(completion, sites=tuple(sites),
                     channels=tuple(offset_one if ch is free else ch
                                    for ch in completion.channels))
    edited = replace(edited, identity=completion_identity(edited))
    assert edited.identity != completion_identity(completion), \
        "a no-op tamper: the offset rewrite did not change content identity"
    with pytest.raises(CompletionError) as caught:
        verify_completion(plan, edited, target)
    assert caught.value.code == "COMPLETION_REUSE_OFFSET_MISMATCH"
    assert caught.value.expected == 0 and caught.value.actual == 1


# ------------------------------------------------------------------- refusals
def test_effects_must_be_the_plan_problems_own_records():
    """No second input shape: a coordinated copy is refused by identity.

    Same ids, same content, different objects - a record the plan was not
    resolved against cannot decide which obligations exist. A list, and a wrong
    length, are refused too.
    """
    plan, target = _real_plan_bound()
    effects = tuple(effect for region in plan.problem_ref.regions
                    for effect in region.effects)
    # The tamper here is deliberately VALUE-EQUAL: the obligation content is
    # unchanged and only the OBJECT differs. That is the point - a coordinated
    # copy is a different record from the one the plan was resolved against, so
    # the refusal must key on identity, not content. Asserted explicitly so no
    # reader mistakes this for a content comparison, and so the test cannot
    # quietly degrade into a no-op value check.
    copied = tuple(replace(effect, instruction=effect.instruction) for effect in effects)
    assert all(new is not old for new, old in zip(copied, effects))
    assert all(new == old for new, old in zip(copied, effects)), (
        "the forged tuple must be value-equal and object-distinct, or this test "
        "is refusing for the wrong reason")
    with pytest.raises(CompletionError) as caught:
        build_symbolic_completion(plan, target, copied)
    assert caught.value.code == "COMPLETION_EFFECT_NOT_FROM_PROBLEM"

    with pytest.raises(CompletionError) as listing:
        build_symbolic_completion(plan, target, list(effects))
    assert listing.value.code == "COMPLETION_EFFECTS_UNBOUND"

    with pytest.raises(CompletionError) as short:
        build_symbolic_completion(plan, target, effects[:-1])
    assert short.value.code == "COMPLETION_EFFECTS_UNBOUND"


def test_unbound_plan_names_the_missing_field_and_regions():
    """A refusal must say WHAT it looked for, not just that something was absent."""
    plan, region, target = _synthetic_region(trips=4, versions=2)
    unbound = replace(plan, problem_ref=None)
    effects = tuple(effect for effect in region.effects)
    with pytest.raises(CompletionError) as caught:
        build_symbolic_completion(unbound, target, effects)
    message = str(caught.value)
    assert caught.value.code == "COMPLETION_PROBLEM_BINDING_UNAVAILABLE"
    assert "problem_ref" in message, message
    assert region.region_ref.region_id in message, message
    assert caught.value.expected == "NormalizedProblem"


def test_multiple_reuse_edges_on_one_buffer_are_refused_not_merged():
    """Two reuse edges for one buffer id must not be collapsed by a dict.

    The earlier builder stored reuse in a dict keyed by buffer id, so a second
    edge silently OVERWROTE the first: the surviving distance_offset and delay
    looked authoritative while the dropped obligation simply vanished. A refusal
    is required here, and it must name the buffer.
    """
    plan, region, target = _synthetic_region(trips=4, versions=2)
    first = region.reuse[0]
    second = replace(first, last_reader="access:read-scalar", distance_offset=1)
    doubled = replace(region, reuse=region.reuse + (second,))
    bound = replace(plan.problem_ref, regions=(doubled,))
    plan = replace(plan, problem_ref=bound)
    effects = tuple(effect for entry in bound.regions for effect in entry.effects)
    with pytest.raises(CompletionError) as caught:
        build_symbolic_completion(plan, target, effects)
    assert caught.value.code == "COMPLETION_MULTIPLE_REUSE_UNSUPPORTED"
    assert first.buffer in str(caught.value), str(caught.value)


def test_rewording_target_prose_cannot_drift_completion_identity():
    """Documentation must not move identity bytes, in either direction.

    A pool's source sentence is prose. Rewriting it may not change the target
    identity, the completion identity, or anything downstream that hashes them.
    Changing a CAPACITY must. Without this, every copy edit silently invalidates
    the whole chain.
    """
    plan, region, target = _synthetic_region(trips=4, versions=2)
    effects = tuple(effect for entry in plan.problem_ref.regions
                    for effect in entry.effects)
    before = build_symbolic_completion(plan, target, effects)

    from tilefoundry_ascend.completion import model as facts_module

    original = facts_module.ASCEND_EVENT_FACTS
    reworded = replace(original, pools=tuple(
        replace(pool, source="rewritten documentation sentence %d" % index)
        for index, pool in enumerate(original.pools)))
    facts_module.ASCEND_EVENT_FACTS = reworded
    try:
        after = build_symbolic_completion(plan, target, effects)
    finally:
        facts_module.ASCEND_EVENT_FACTS = original
    assert after.identity == before.identity, \
        "rewording prose drifted the completion identity"

    shrunk = replace(original, pools=tuple(
        replace(pool, capacity=max(1, pool.capacity - 1)) if index == 0 else pool
        for index, pool in enumerate(original.pools)))
    facts_module.ASCEND_EVENT_FACTS = shrunk
    try:
        moved = build_symbolic_completion(plan, target, effects)
    finally:
        facts_module.ASCEND_EVENT_FACTS = original
    assert moved.target_identity != before.target_identity, \
        "a capacity change failed to move the target identity"


def test_equivalent_content_is_consumable_and_changed_content_is_not():
    """The registry's authority rule, in both directions, as behaviour.

    Content equality (not object identity) is what grants consumption, and that
    is deliberate - see require_completion_verified's docstring for why an
    is-binding would be weaker rather than stronger. What must hold is:

      (a) an equal-content twin of the verified completion is consumable, because
          every check the gate performs is a check on content plus a re-verification
          against the SAVED real plan/target;
      (b) content that differs by one token is refused - even with a hand-written
          receipt whose every field matches its own content, and even when the
          digest is recomputed correctly;
      (c) the differing content cannot earn authority by being verified again.
    """
    plan, target = _real_plan_bound()
    effects = tuple(effect for region in plan.problem_ref.regions
                    for effect in region.effects)
    completion = build_symbolic_completion(plan, target, effects)
    verified = verify_completion(plan, completion, target)

    # (a) a different OBJECT, identical content
    twin = replace(verified)
    assert twin is not verified and twin == verified
    assert require_completion_verified(twin).identity == verified.identity

    # (b) one channel's version count changed, identity recomputed honestly
    channels = list(verified.channels)
    first = channels[0]
    channels[0] = replace(first, versions=int(first.versions) + 1)
    changed = replace(verified, channels=tuple(channels))
    changed = replace(changed, identity=completion_identity(changed))
    assert changed.identity != verified.identity, (
        "the tamper must change content identity, or this test proves nothing")
    with pytest.raises(CompletionError) as caught:
        require_completion_verified(changed)
    assert caught.value.code == "COMPLETION_NOT_VERIFIED"

    # (c) re-verifying the changed content must refuse, not register it
    with pytest.raises(CompletionError) as refused:
        verify_completion(plan, changed, target)
    assert refused.value.code in (
        "COMPLETION_VERSIONS_OUT_OF_DOMAIN", "COMPLETION_CHANNEL_VERSIONS_MISMATCH")
    with pytest.raises(CompletionError):
        require_completion_verified(changed)


def test_deleted_exit_channel_is_caught_by_site_balance_with_real_codes():
    """Pin the MECHANISM that catches a dropped EXIT channel, by measured code.

    The lead's correction: EXIT is NOT uncovered - deleting the channel alone is
    caught, and the catching mechanism is SITE BALANCE (verify.py:383-384
    COMPLETION_SITE_UNBALANCED) plus the exit_site_ids set check
    (COMPLETION_EXIT_SITES_MISMATCH), not obligation-level two-way matching. This
    test records exactly which of the four ablations fires which code, measured on
    the real independent_ub plan whose baseline is 9 sites / 5 channels (3 READY +
    2 EXIT).

    Four ways to remove one EXIT channel, and what actually happens (all asserted
    here, so a future change in any of them is a visible regression, not a silent
    one):
      (a) drop only the channel, leave its two sites -> COMPLETION_SITE_UNBALANCED
          (the producer publish site and the drain-exit site now belong to no
          channel, so the balance rule fires);
      (c) drop the channel and its sites, leave exit_site_ids stale ->
          COMPLETION_EXIT_SITES_MISMATCH;
      (d) point the channel's producer at a site id that is not present ->
          COMPLETION_CHANNEL_ENDPOINT_UNKNOWN, raised by the STRUCTURE pass, which
          runs before _correspondence. This is why the continue at the
          correspondence site is unreachable for a dangling producer - the lead
          retracted that claim and it stays retracted.

    (b) is reported honestly rather than asserted as a pass: dropping the channel
    AND both its sites AND their exit_site_ids entries, then recomputing the
    identity, is CURRENTLY ACCEPTED by verify_completion. That is the one shape of
    EXIT deletion site balance cannot see, because nothing is left unbalanced. It
    is the case an EXIT obligation-level check would close; the lead narrowed the
    request to a test, so the gap is pinned here (assert accepted) instead of being
    quietly closed or quietly ignored - a change to either outcome fails this test.
    """
    from dataclasses import replace as _replace

    plan, target = _real_plan_bound()
    effects = tuple(effect for region in plan.problem_ref.regions
                    for effect in region.effects)
    base = build_symbolic_completion(plan, target, effects)
    exit_channels = [c for c in base.channels if c.reason == EXIT]
    assert len(base.sites) == 9 and len(exit_channels) == 2, (
        "the independent_ub baseline changed; re-measure which mechanism catches "
        "which deletion before trusting the codes below")
    victim = exit_channels[0]
    ids = {victim.producer} | set(victim.consumers)

    def recompute(**changes):
        return _replace(_replace(base, **changes),
                        identity=completion_identity(_replace(base, **changes)))

    def code_of(obj):
        try:
            verify_completion(plan, obj, target)
        except CompletionError as error:
            return error.code
        return "ACCEPTED"

    # (a) channel only: the two orphan sites are unbalanced.
    assert code_of(recompute(channels=tuple(
        c for c in base.channels if c != victim))) == "COMPLETION_SITE_UNBALANCED"

    # (c) channel + sites, stale exit_site_ids.
    assert code_of(recompute(
        channels=tuple(c for c in base.channels if c != victim),
        sites=tuple(s for s in base.sites if s.site_id not in ids),
    )) == "COMPLETION_EXIT_SITES_MISMATCH"

    # (d) a producer id that names no site: the structure pass refuses it first.
    assert code_of(recompute(channels=tuple(
        _replace(c, producer=c.producer + "-gone") if c is victim else c
        for c in base.channels))) == "COMPLETION_CHANNEL_ENDPOINT_UNKNOWN"

    # (b) the coherent deletion. This is the case site balance CANNOT see, and it
    # is now refused by the obligation-level check: bind.exit_obligations() derives
    # ("EXIT", region, task) from the problem's own records, independently of the
    # builder's _drains(), and _correspondence requires a channel for every one.
    #
    # No new code was invented. COMPLETION_OBLIGATION_MISSING already means "the
    # problem states this obligation and no channel protects it", which is exactly
    # what happened; a COMPLETION_EXIT_OBLIGATION_MISSING would be a synonym of it,
    # and the lead's ruling forbids synonyms. The obligation's own locator is in
    # the payload, so the refusal still says WHICH one.
    coherent = recompute(
        channels=tuple(c for c in base.channels if c != victim),
        sites=tuple(s for s in base.sites if s.site_id not in ids),
        exit_site_ids=tuple(x for x in base.exit_site_ids if x not in ids))
    code = code_of(coherent)
    assert code == "COMPLETION_OBLIGATION_MISSING", code
    try:
        verify_completion(plan, coherent, target)
    except CompletionError as error:
        # The message must name the deleted obligation, not just count it.
        assert "EXIT" in str(error), str(error)[:300]

def test_removed_exit_channel_names_the_deleted_obligation():
    """The refusal must locate the dropped drain, not merely report a count.

    Case (b) of the deletion matrix checked that a coherent EXIT removal is now
    refused; this checks the QUALITY of that refusal, because a code with no
    locator is what made the earlier 'verification inputs unavailable' message
    useless. The payload must carry the obligation tuple - kind, region, task -
    so a reader can see which drain vanished without re-deriving it.
    """
    from dataclasses import replace as _replace

    plan, target = _real_plan_bound()
    effects = tuple(effect for region in plan.problem_ref.regions
                    for effect in region.effects)
    base = build_symbolic_completion(plan, target, effects)
    victim = [c for c in base.channels if c.reason == EXIT][0]
    ids = {victim.producer} | set(victim.consumers)
    coherent = _replace(
        base,
        channels=tuple(c for c in base.channels if c != victim),
        sites=tuple(s for s in base.sites if s.site_id not in ids),
        exit_site_ids=tuple(x for x in base.exit_site_ids if x not in ids))
    coherent = _replace(coherent, identity=completion_identity(coherent))

    producer_task = _task_ids(plan)[id(_site(base, victim.producer).task_ref)]
    region_id = _site(base, victim.producer).region_id
    with pytest.raises(CompletionError) as caught:
        verify_completion(plan, coherent, target)
    assert caught.value.code == "COMPLETION_OBLIGATION_MISSING"
    # expected is the required-but-unprotected side, actual is what the
    # template does protect; both are locator strings, so the reader can see
    # the hole without re-deriving it.
    listed = list(caught.value.expected or [])
    assert any("EXIT" in item and region_id in item and producer_task in item
               for item in listed), (listed, producer_task, region_id)
    # The other EXIT drain must still appear in actual: the refusal names
    # exactly the obligation that lost its channel, not every EXIT obligation
    # in the program. An earlier version of this assertion required NO EXIT in
    # actual at all, which is wrong - the surviving drain is legitimately
    # protected and legitimately shows up in the protected set.
    survivor = [c for c in base.channels if c.reason == EXIT][1]
    survivor_task = _task_ids(plan)[id(_site(base, survivor.producer).task_ref)]
    assert survivor_task != producer_task
    protected = list(caught.value.actual or [])
    assert any("EXIT" in item and survivor_task in item for item in protected), (
        protected, survivor_task)
    assert not any("EXIT" in item and producer_task in item for item in protected)


def test_extra_exit_channel_is_refused_not_tolerated():
    """The other direction: a drain for a task that owes none is unjustified.

    A completion that ADDS an EXIT channel (copying an existing one onto a task
    with no boundary write) stays internally consistent - the new sites are
    balanced, exit_site_ids is recomputed - so only the re-derived obligation set
    can call it out. Without this half, 'every obligation has a channel' would be
    satisfiable by inventing channels, which is how a template would drift from
    the schedule it claims to protect.
    """
    from dataclasses import replace as _replace

    plan, target = _real_plan_bound()
    effects = tuple(effect for region in plan.problem_ref.regions
                    for effect in region.effects)
    base = build_symbolic_completion(plan, target, effects)
    existing = [c for c in base.channels if c.reason == EXIT][0]
    producer = _site(base, existing.producer)
    # Re-point a copy at a task that declares no boundary write: the consumer is
    # the same region-exit envelope, so every SHAPE rule still passes.
    other = [site for site in base.sites
             if site.anchor == AFTER_TASK and site.task_ref is not producer.task_ref][0]
    extra_producer = _replace(producer, site_id=producer.site_id + ":extra",
                              task_ref=other.task_ref)
    extra_exit = _replace(_site(base, existing.consumers[0]),
                          site_id="site:extra-exit")
    duplicated = _replace(existing, channel_id=existing.channel_id + ":extra",
                          producer=extra_producer.site_id,
                          consumers=(extra_exit.site_id,))
    padded = _replace(
        base,
        sites=base.sites + (extra_producer, extra_exit),
        channels=base.channels + (duplicated,),
        exit_site_ids=base.exit_site_ids + (extra_exit.site_id,))
    padded = _replace(padded, identity=completion_identity(padded))

    with pytest.raises(CompletionError) as caught:
        verify_completion(plan, padded, target)
    assert caught.value.code == "COMPLETION_CHANNEL_UNJUSTIFIED", caught.value.code



def test_a_fabricated_read_cannot_delete_a_synchronisation_without_adding_one():
    """The challenger's attack, pinned at the derivation where it was real.

    Measured before the fix: independent_ub's unit:2 writes 'out' with
    WRITE_VISIBLE and nothing reads it, so it owes a region-exit drain. Adding
    ONE fabricated READ access on that root - with an effect but no dependence -
    removed ("EXIT", region:0, unit:2) from the required obligations and added
    no READY obligation either. The writer then owed neither a drain nor a flag,
    i.e. a bare access object had deleted a synchronisation.

    Root cause was an asymmetry in the derivation, not in the builder: the drain
    was dropped on the presence of an ACCESS while READY obligations are keyed on
    a declared EDGE, so the two halves consulted different objects. Both now read
    the same relation - a declared RAW edge from the writer to a task that really
    reads that root - so forging an access changes nothing, and discharging a
    drain by declaring an edge ADDS an obligation to the required set instead of
    silently removing one.

    Asserted on bind.exit_obligations() directly rather than through
    resolve_schedule, because the plan gate refuses these mutants upstream
    (measured: SOURCE_EFFECT_COVERAGE for the effect-only mutant, PLAN_NOT_SAT
    when an edge is fabricated too). That is good defence in depth, but a
    derivation only ever exercised behind another layer's refusal is not proven:
    this is the layer the obligation set is actually read from.
    """
    from tilefoundry_ascend.completion.bind import exit_obligations

    plan, target = _real_plan_bound()
    region = plan.problem_ref.regions[0]
    region_plan = plan.regions[0]

    reads = [access for access in region.accesses
             if access.mode in ("READ", "READ_WRITE")]
    writes = [access for access in region.accesses
              if access.mode in ("WRITE", "READ_WRITE")]
    read_roots = set(access.root_ref for access in reads)
    boundary = [access for access in writes if access.root_ref not in read_roots]
    assert boundary, "independent_ub is expected to have an unread GMEM boundary write"
    victim = boundary[0]
    owed_before = exit_obligations(region, region_plan)
    assert ("EXIT", region.region_ref.region_id, victim.task_id) in owed_before, owed_before

    template = reads[0]
    forged_read = replace(template, access_id="access:FORGED",
                          root_ref=victim.root_ref, view_ref=victim.root_ref,
                          mode="READ")
    forged_effect = replace(next(e for e in region.effects if e.kind == "MEMORY"),
                            effect_id="effect:FORGED",
                            access_ids=("access:FORGED",),
                            completion="READ_COMPLETE")
    mutated = replace(region, accesses=tuple(region.accesses) + (forged_read,),
                      effects=tuple(region.effects) + (forged_effect,))

    owed_after = exit_obligations(mutated, region_plan)
    # The drain is still owed: a bare access cannot discharge it.
    assert owed_after == owed_before, (owed_before, owed_after)

    # And the symmetric statement on the required set the verifier compares:
    # the READY obligations do not change either, so nothing silently moved.
    from tilefoundry_ascend.completion.bind import obligations_of

    before = obligations_of(region, region_plan)
    after = obligations_of(mutated, region_plan)
    assert after[0] == before[0], "READY obligations moved without a declared edge"
    assert after[3] == before[3], "EXIT obligations moved without a declared edge"

    # The legitimate way to discharge a drain is to DECLARE the edge, and that
    # must ADD an obligation rather than remove coverage. Prove the mechanism is
    # live (otherwise the test above would pass simply because nothing can ever
    # discharge a drain): with a real RAW edge from the writer to a task that
    # reads the root, the drain is dropped AND a READY obligation appears.
    covering_edge = Dependence(producer=victim.task_id,
                               consumer=template.task_id, distance=0, delay=0,
                               kind="RAW", effect_ids=())
    with_edge = replace(region, accesses=tuple(region.accesses) + (forged_read,),
                        effects=tuple(region.effects) + (forged_effect,),
                        edges=tuple(region.edges) + (covering_edge,))
    discharged = exit_obligations(with_edge, region_plan)
    assert ("EXIT", region.region_ref.region_id, victim.task_id) not in discharged, (
        "a declared RAW edge to a real reader of the root should discharge the drain; "
        "if this never fires, the edge rule above is dead code")
    ready_with_edge = obligations_of(with_edge, region_plan)[0]
    assert ("READY", region.region_ref.region_id, victim.task_id,
              template.task_id) in ready_with_edge, ready_with_edge


def test_a_lying_plan_digest_is_refused_though_its_bytes_are_honest():
    """The stored plan_identity is raw material for an attack, not evidence.

    Measured before the fix (R21): replace(plan, plan_identity=<fabricated>) with
    plan_canonical_bytes left honest was ACCEPTED by build_symbolic_completion,
    which copied the fabricated string into completion.plan_identity, and then
    verify_completion returned VERIFIED with 9 sites. Not an authorization hole on
    its own - require_completion_verified goes through the content registry and
    never reads that string - but a stored field read and never recomputed is
    precisely the R21 shape, one field over from the bounds tamper.

    Both stored fields must agree with the single recipe: bytes with
    plan_canonical_bytes(plan), digest with plan_identity_of(plan)[1].
    """
    from dataclasses import replace as _replace

    from tilefoundry_ascend.completion import CompletionError
    from tilefoundry_ascend.plan.resolved import plan_identity_of

    plan, target = _real_plan_bound()
    honest_bytes = plan.plan_canonical_bytes
    effects = tuple(effect for region in plan.problem_ref.regions
                    for effect in region.effects)

    # Control first: the untampered plan is self-consistent AND accepted, so a
    # later refusal cannot be an artefact of the fixture.
    _canonical, honest_digest = plan_identity_of(plan)
    assert plan.plan_identity == honest_digest
    assert build_symbolic_completion(plan, target, effects).sites

    # (1) the digest lies while the bytes stay honest.
    lie = _replace(plan, plan_identity="sha256:" + "a" * 64)
    assert lie.plan_canonical_bytes == honest_bytes
    with pytest.raises(CompletionError) as caught:
        build_symbolic_completion(lie, target, effects)
    assert caught.value.code == "COMPLETION_PLAN_STALE", caught.value.code
    assert "a" * 64 in str(caught.value.expected)
    assert honest_digest in str(caught.value.actual)

    # (2) the bytes lie while the digest stays honest - the bytes half catches it.
    bytes_lie = _replace(plan, plan_canonical_bytes=b"{}")
    with pytest.raises(CompletionError) as caught:
        build_symbolic_completion(bytes_lie, target, effects)
    assert caught.value.code == "COMPLETION_PLAN_STALE"

    # (3) BOTH stored fields replaced by a pair that is self-consistent. This is
    # the case a forger who recomputes both from tampered content actually
    # produces, and the pair agrees with itself - so only recomputation from the
    # LIVE objects can see it. Asserted explicitly rather than assumed.
    pair_lie = _replace(plan, plan_identity="sha256:" + "b" * 64,
                        plan_canonical_bytes=b'{"forged": true}')
    with pytest.raises(CompletionError) as caught:
        build_symbolic_completion(pair_lie, target, effects)
    assert caught.value.code == "COMPLETION_PLAN_STALE"


def test_a_zero_trip_plan_cannot_be_reached_by_editing_the_bounds():
    """R20: a bound edit must be refused, not answered with 0 sites / 0 channels.

    Measured: replace(plan, regions=(replace(rp, lower=0, stop=0, step=1),)) leaves
    plan_identity byte-identical (replace carries stored fields across untouched), so
    a consumer reading only the recorded digest saw a valid plan while the object it
    held said the region never executes. build then returned sites=0 channels=0 -
    every obligation silently gone - and verify raised a bare ValueError because the
    zero-trip early return answered 3 values where its caller unpacks 4.

    Three assertions, one per defect: the input is re-checked; the early return has
    the same arity as every other return so a refusal is never a crash; and the
    legitimate static-zero region still answers 0/0 because it is not an edit.
    """
    from dataclasses import replace as _replace

    from tilefoundry.ir.core import Constant

    from tilefoundry_ascend.completion import CompletionError
    from tilefoundry_ascend.completion.bind import obligations_of

    plan, target = _real_plan_bound()
    region_plan = plan.regions[0]
    zero = Constant(type=region_plan.lower_expr.type, value=0)
    edited_region = _replace(region_plan, lower_expr=zero, stop_expr=zero,
                             step_expr=_replace(region_plan.step_expr, value=1))
    edited = _replace(plan, regions=(edited_region,))
    assert edited.plan_identity == plan.plan_identity, (
        "replace() is expected to carry the stored digest across; if this fails the "
        "attack shape changed and this test needs re-deriving, not deleting")

    effects = tuple(effect for region in plan.problem_ref.regions
                    for effect in region.effects)
    with pytest.raises(CompletionError) as caught:
        build_symbolic_completion(edited, target, effects)
    assert caught.value.code == "COMPLETION_PLAN_STALE", caught.value.code

    # Arity: even a genuinely static-zero region answers 4 values, never 3.
    ring, ring_problem_region, _ring_target = _synthetic_region(trips=0, versions=2)
    assert len(obligations_of(ring_problem_region, ring.regions[0])) == 4
    assert obligations_of(ring_problem_region, ring.regions[0]) == ((), (), {}, ())

    # The same zero reached through EDITED bounds is refused instead: provenance is
    # checked before the zero is believed. Here the edit is on the RegionPlan copy,
    # which is where region_bounds() looks.
    ring_plan_region = ring.regions[0]
    edited_ring = _replace(ring_plan_region,
                           stop_expr=Constant(type=ring_plan_region.stop_expr.type,
                                              value=9))
    with pytest.raises(CompletionError) as caught:
        obligations_of(ring_problem_region, edited_ring)
    assert caught.value.code == "COMPLETION_BOUNDS_MISMATCH", caught.value.code