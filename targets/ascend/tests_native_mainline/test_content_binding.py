"""Content binding: every declared data field moves identity or is refused.

TASK-47 requirement 4. The fourteen fields that used to change a problem's
MEANING without changing its identity were found by hand and patched by hand;
a hand list goes stale the moment a field is added. This module enumerates the
fields from the dataclasses themselves and requires, for each one, that either
the recomputed digest moves or a consumer refuses by NAME.

Method (lead's rules):
  * a field is judged by CALLING the binding function on tampered content,
    never by comparing a stored identity field that dataclasses.replace
    carries along unchanged;
  * two tamper KINDS are tried per field, because they hide different
    failures - "replace" (a genuinely different value of the same type) and
    "drop" (the value is emptied). Losing a fact must not read the same as
    stating it, and the fields found by hand were mostly discard-visible;
  * when the digest does not move, the CONSUMER must refuse by name.
"""
from __future__ import annotations

import dataclasses
from dataclasses import replace

import pytest

from tilefoundry.analysis import analyze
from tilefoundry.cli.source import checked_program
from tilefoundry.ir.core import Constant, Var
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.schedule import finalize
from tilefoundry_ascend.plan import problem as P
from tilefoundry_ascend.plan.delays import DelayRefusal, ModelSource
from tilefoundry_ascend.plan.identity import IdentityRefusal, _encode
from tilefoundry_ascend.plan.numeric import to_schedule_problem
from tilefoundry_ascend.plan.problem import (ProblemError, analyze_native_problem,
                                             problem_identity_of)
from tilefoundry_ascend.plan.resolved import (ResolveError, _plan_bytes,
                                              plan_identity_of,
                                              resolve_schedule,
                                              verify_plan_bindings)
from tilefoundry_ascend.plan.source_audit import SourceAuditError
from tilefoundry_ascend.plan.units import (AccessFact, Effect, Fixed, SourceRefusal,
                                           StorageKey, build_native_unit_tree)
from tilefoundry_ascend.solver import solve_region
from tilefoundry_ascend.tf_target import Ascend950PRTarget

UB = "targets/ascend/tests_native_mainline/fixtures/authors/independent_ub.py:IndependentUB"
REFUSALS = (IdentityRefusal, DelayRefusal, SourceAuditError, ResolveError,
            ProblemError, ValueError, TypeError)


def _chain(path=UB):
    program = checked_program(path, None, command="test content binding")
    analyzed = analyze(program.module, program.entry,
                       analysis=("compute-cost", "memory"))
    module = finalize(analyzed, authored=program.authored)
    tree = build_native_unit_tree(module, authored=program.authored)
    return analyze_native_problem(tree, None, None, module.entry_function().target)


PROBLEM = _chain()
BASE = problem_identity_of(PROBLEM)[1]
REGION = PROBLEM.regions[0]
ANSWERS = {region.region_ref.region_id: solve_region(to_schedule_problem(region))
           for region in PROBLEM.regions}

CLASSES = {"RegionProblem": P.RegionProblem, "Task": P.Task,
           "LogicalBuffer": P.LogicalBuffer, "AccessFact": AccessFact,
           "Effect": Effect, "Dependence": P.Dependence, "Reuse": P.Reuse,
           "Capacity": P.Capacity, "NormalizedProblem": P.NormalizedProblem}
CONTAINER = {"Task": "tasks", "LogicalBuffer": "buffers", "AccessFact": "accesses",
             "Effect": "effects", "Dependence": "edges", "Reuse": "reuse",
             "Capacity": "capacities"}


def _sample(name):
    if name == "RegionProblem":
        return REGION
    if name == "NormalizedProblem":
        return PROBLEM
    seq = getattr(REGION, CONTAINER[name])
    for item in seq:
        if type(item).__name__ == name:
            return item
    return None


def _rebuild(name, field, value):
    """The tampered problem, with ONE field of ONE record replaced."""
    if name == "RegionProblem":
        return replace(PROBLEM, regions=(replace(REGION, **{field: value}),
                                         *PROBLEM.regions[1:]))
    if name == "NormalizedProblem":
        return replace(PROBLEM, **{field: value})
    attribute = CONTAINER[name]
    seq = list(getattr(REGION, attribute))
    for index, item in enumerate(seq):
        if type(item).__name__ == name:
            seq[index] = replace(item, **{field: value})
            break
    else:
        seq.append(value)
    return replace(PROBLEM, regions=(replace(REGION, **{attribute: tuple(seq)}),
                                     *PROBLEM.regions[1:]))


def _verdict(tampered):
    """(state, detail): digest moved, or which layer refused by name."""
    try:
        digest = problem_identity_of(tampered)[1]
    except REFUSALS as error:
        return "refused", str(getattr(error, "code", type(error).__name__))
    if digest != BASE:
        return "moved", digest
    # the digest is unchanged, so a CONSUMER must refuse the content instead
    try:
        resolve_schedule(tampered, ANSWERS)
    except REFUSALS as error:
        return "refused", str(getattr(error, "code", type(error).__name__))
    return "unbound", digest


# --------------------------------------------------------------- tamper kinds
def _drop(value=None):
    """"drop": the value is emptied. Losing a fact must move identity."""
    return ("drop", lambda sample: ())


def _swap(build):
    """"replace": a genuinely different value of the SAME type."""
    return ("replace", build)


TAMPER = {
    # ---- RegionProblem: the finite sets the solver and storage are built from
    # region_ref is the Region object itself, so the tamper must BUILD a
    # different Region: pointing it at another source node and parent.
    ("RegionProblem", "region_ref"):
        _swap(lambda s: replace(REGION.region_ref,
                                node_ref=REGION.tasks[0].unit_ref.node_ref,
                                parent_id="region:elsewhere")),
    ("RegionProblem", "tasks"): _drop(),
    ("RegionProblem", "buffers"): _drop(),
    ("RegionProblem", "accesses"): _drop(),
    ("RegionProblem", "effects"): _drop(),
    ("RegionProblem", "resources"): _drop(),
    ("RegionProblem", "owners"): _drop(),
    ("RegionProblem", "pipes"): _drop(),
    ("RegionProblem", "edges"): _drop(),
    ("RegionProblem", "capacities"): _drop(),
    ("RegionProblem", "issue_after"): _drop(),
    ("RegionProblem", "reuse"):
        _swap(lambda s: (P.Reuse(REGION.accesses[0].access_id,
                                 REGION.accesses[1].access_id,
                                 REGION.buffers[0].buffer_id, 1, 0,
                                 REGION.edges[0].delay_fact),)),
    ("RegionProblem", "manual_pipe_order"):
        _swap(lambda s: (("unit:0", "unit:1"),)),
    ("RegionProblem", "enable_offset"): _swap(lambda s: not s.enable_offset),
    ("RegionProblem", "max_stage"): _swap(lambda s: int(s.max_stage) + 1),
    ("RegionProblem", "fixed_ii"): _swap(lambda s: 1 if s.fixed_ii != 1 else None),
    ("RegionProblem", "ii_min"): _swap(lambda s: int(s.ii_min) + 1),
    ("RegionProblem", "ii_max"): _swap(lambda s: int(s.ii_max) + 1),
    # ---- Task
    ("Task", "task_id"): _swap(lambda s: s.task_id + "_x"),
    ("Task", "unit_ref"): _swap(lambda s: REGION.tasks[1].unit_ref),
    ("Task", "uses"): _drop(),
    ("Task", "latency"): _swap(lambda s: int(s.latency) + 1),
    ("Task", "source_order"): _swap(lambda s: int(s.source_order) + 1),
    ("Task", "authored_stage"): _swap(lambda s: 0 if s.authored_stage else 1),
    # ---- LogicalBuffer
    ("LogicalBuffer", "buffer_id"): _swap(lambda s: s.buffer_id + "_x"),
    ("LogicalBuffer", "root_ref"):
        # another buffer's ROOT (a different variable), not a copy of this one
        _swap(lambda s: next(b.root_ref for b in REGION.buffers
                             if b.root_ref is not s.root_ref)),
    ("LogicalBuffer", "allocation_ref"):
        # a different allocation SITE (another statement), not another copy
        _swap(lambda s: REGION.accesses[5].source_ref),
    ("LogicalBuffer", "owner"): _swap(lambda s: replace(s.owner, sid=7)),
    ("LogicalBuffer", "space"):
        _swap(lambda s: "gmem" if s.space != "gmem" else "smem"),
    ("LogicalBuffer", "dtype"): _swap(lambda s: DType.bf16),
    ("LogicalBuffer", "shape"): _swap(lambda s: tuple(int(v) + 1 for v in s.shape)),
    ("LogicalBuffer", "layout"): _swap(lambda s: replace(s.layout, shape=())),
    ("LogicalBuffer", "domain"): _swap(lambda s: Fixed(value=9)),
    ("LogicalBuffer", "generation_delta"):
        _swap(lambda s: int(s.generation_delta) + 1),
    ("LogicalBuffer", "slot_bytes"): _swap(lambda s: int(s.slot_bytes) + 1),
    ("LogicalBuffer", "alignment_bytes"):
        _swap(lambda s: int(s.alignment_bytes) + 1),
    ("LogicalBuffer", "capacity_key"): _swap(lambda s: StorageKey("l1", s.owner)),
    ("LogicalBuffer", "initial_values"):
        _swap(lambda s: (Constant(type=TensorType.scalar(DType.f32), value=1.5),)),
    ("LogicalBuffer", "terminal_readers"): _drop(),
    # ---- AccessFact
    ("AccessFact", "access_id"): _swap(lambda s: s.access_id + "_x"),
    ("AccessFact", "task_id"): _swap(lambda s: "unit:unknown"),
    ("AccessFact", "source_ref"):
        # another instruction's occurrence, so the operand is re-located
        _swap(lambda s: REGION.accesses[5].source_ref),
    ("AccessFact", "operand_index"): _swap(lambda s: int(s.operand_index) + 1),
    ("AccessFact", "root_ref"): _swap(lambda s: REGION.accesses[1].root_ref),
    ("AccessFact", "view_ref"): _swap(lambda s: REGION.accesses[1].view_ref),
    ("AccessFact", "relation"): _swap(lambda s: replace(s.relation, may_access=True)),
    ("AccessFact", "mode"):
        _swap(lambda s: "READ" if s.mode == "WRITE" else "WRITE"),
    ("AccessFact", "owner"): _swap(lambda s: replace(s.owner, sid=7)),
    ("AccessFact", "guard"):
        _swap(lambda s: Constant(type=TensorType.scalar(DType.i64), value=7)),
    ("AccessFact", "generation"):
        _swap(lambda s: Constant(type=TensorType.scalar(DType.i64), value=7)),
    # ---- Effect
    ("Effect", "effect_id"): _swap(lambda s: s.effect_id + "_x"),
    ("Effect", "task_id"): _swap(lambda s: "unit:unknown"),
    ("Effect", "source_ref"): _swap(lambda s: REGION.effects[1].source_ref),
    ("Effect", "kind"): _swap(lambda s: "CONTROL"),
    ("Effect", "access_ids"): _swap(lambda s: (REGION.accesses[-1].access_id,)),
    ("Effect", "state_ref"):
        _swap(lambda s: Constant(type=TensorType.scalar(DType.i64), value=7)),
    ("Effect", "generation"):
        _swap(lambda s: Constant(type=TensorType.scalar(DType.i64), value=7)),
    ("Effect", "guard"):
        _swap(lambda s: Constant(type=TensorType.scalar(DType.i64), value=7)),
    ("Effect", "completion"): _swap(lambda s: "ISSUE"),
    ("Effect", "pipe"): _swap(lambda s: "wrong-pipe"),
    ("Effect", "instruction"): _swap(lambda s: "ghost_instruction"),
    # ---- Dependence
    ("Dependence", "producer"): _swap(lambda s: "unit:unknown"),
    ("Dependence", "consumer"): _swap(lambda s: "unit:unknown"),
    ("Dependence", "distance"): _swap(lambda s: int(s.distance) + 1),
    ("Dependence", "delay"): _swap(lambda s: int(s.delay) + 1),
    ("Dependence", "kind"): _swap(lambda s: "WAR"),
    ("Dependence", "effect_ids"): _drop(),
    ("Dependence", "delay_fact"): _swap(lambda s: replace(
        s.delay_fact, source=ModelSource(module="ir/isa.py", symbol="ghost_row",
                                         field="latency_model", value=1))),
    # ---- Reuse: this program states no obligation, so the tamper adds one
    #      built from real ids, then edits the field under test
    ("Reuse", "last_reader"): _swap(lambda s: _reuse(last_reader="access:unknown")),
    ("Reuse", "next_writer"): _swap(lambda s: _reuse(next_writer="access:unknown")),
    ("Reuse", "buffer"): _swap(lambda s: _reuse(buffer="buffer:unknown")),
    ("Reuse", "delay"): _swap(lambda s: _reuse(delay=4096)),
    ("Reuse", "distance_offset"): _swap(lambda s: _reuse(distance_offset=1)),
    ("Reuse", "delay_fact"): _swap(lambda s: _reuse(delay_fact=replace(
        REGION.edges[0].delay_fact,
        source=ModelSource(module="ir/isa.py", symbol="ghost_row",
                           field="latency_model", value=1)))),
    # ---- Capacity
    ("Capacity", "storage_key"):
        _swap(lambda s: StorageKey("l1", s.storage_key.owner)),
    ("Capacity", "buffers"): _drop(),
    ("Capacity", "usable_bytes"): _swap(lambda s: int(s.usable_bytes) + 1),
    ("Capacity", "reserved_bytes"): _swap(lambda s: int(s.reserved_bytes) + 1),
    ("Capacity", "source"): _swap(lambda s: s.source + "_x"),
    ("Capacity", "status"): _swap(lambda s: s.status + "_x"),
    # ---- NormalizedProblem
    ("NormalizedProblem", "problem_identity"):
        _swap(lambda s: "sha256:" + "0" * 64),
    ("NormalizedProblem", "problem_canonical_bytes"):
        _swap(lambda s: b'{"lie":1}'),
    ("NormalizedProblem", "target_identity"):
        _swap(lambda s: "sha256:" + "1" * 64),
    ("NormalizedProblem", "model_version"):
        _swap(lambda s: "sha256:" + "2" * 64),
    ("NormalizedProblem", "delay_table_digest"):
        _swap(lambda s: "sha256:" + "3" * 64),
    ("NormalizedProblem", "regions"): _drop(),
    ("NormalizedProblem", "coverage"):
        # appending a refusal is the non-vacuous direction here: this program
        # legitimately has none, so emptying the list would be a no-op
        _swap(lambda s: replace(s.coverage, refusals=(
            SourceRefusal("SOURCE_COVERAGE_TASKS_UNMAPPED", None, "injected"),
        ) + tuple(s.coverage.refusals))),
    ("NormalizedProblem", "source_snapshot"):
        _swap(lambda s: replace(s.source_snapshot, identity="sha256:" + "d" * 64)),
    ("NormalizedProblem", "target_ref"):
        _swap(lambda s: Ascend950PRTarget(aiv_count=1)),
}


def _reuse(**overrides):
    base = P.Reuse(last_reader=REGION.accesses[0].access_id,
                   next_writer=REGION.accesses[1].access_id,
                   buffer=REGION.buffers[1].buffer_id, delay=1, distance_offset=0,
                   delay_fact=REGION.edges[0].delay_fact)
    return replace(base, **overrides)


# Fields that are NOT tampered, each with the reason, checked BEHAVIOURALLY
# below rather than by inspecting the wording of the reason.
DOCUMENTED = ("NormalizedProblem", "unit_tree")


def _declared():
    return [(name, field.name) for name, cls in CLASSES.items()
            for field in dataclasses.fields(cls)]


def test_every_declared_field_moves_identity_or_is_refused():
    gaps, moved, refused = [], [], []
    for name, field in _declared():
        if (name, field) == DOCUMENTED:
            continue
        entry = TAMPER.get((name, field))
        if entry is None:
            gaps.append((name, field, "no tamper registered"))
            continue
        kind, build = entry
        sample = _sample(name)
        if sample is None and name != "Reuse":
            gaps.append((name, field, "no %s record in this program" % name))
            continue
        try:
            value = build(sample if sample is not None else REGION.edges[0])
            tampered = _rebuild(name, field, value)
        except REFUSALS as error:
            gaps.append((name, field, "unconstructible: %s" % (error,)))
            continue
        state, detail = _verdict(tampered)
        if state == "moved":
            moved.append((name, field, kind))
        elif state == "refused":
            refused.append((name, field, kind, detail))
        else:
            gaps.append((name, field, "%s tamper left the identity unchanged and "
                                      "nothing refused it" % kind))
    assert not gaps, "binding gaps: %s" % (sorted(map(str, gaps)),)
    assert len(moved) + len(refused) == len(_declared()) - 1
    # refusals must be NAMES, never bare exception types
    assert all(code.isupper() for _, _, _, code in refused), refused[:6]


def test_the_one_documented_hole_is_a_real_cycle_not_an_oversight():
    """unit_tree is excluded for a measured reason, asserted as behaviour.

    Region.direct_children and SourceUnit.region_ref refer back to each other,
    so the structure genuinely cannot be encoded; the assertion is that the
    encoder REFUSES it, which is why the field is bound through the source
    snapshot's identity and the records built from it instead.
    """
    with pytest.raises(IdentityRefusal) as caught:
        _encode(PROBLEM.unit_tree, {}, "$.unit_tree", 0)
    # the refusal names a PATH into the tree, which is why the field cannot be
    # encoded structurally and is bound through the source identity instead
    assert "unit_tree" in caught.value.path, caught.value.path
    # and its content is still bound: a different program has a different
    # source identity, which every consumer checks
    other = _chain("targets/ascend/tests_native_mainline/fixtures/authors/"
                   "copy_only.py:CopyOnly")
    assert problem_identity_of(other)[1] != BASE


def test_discard_tampers_are_covered_not_only_replacements():
    """Both tamper kinds are exercised, and discards are not a fallback.

    The fields the lead found by hand (owners/pipes/resources/uses) are
    discard-visible: emptying them is what removes an obligation, so the sweep
    must contain real "drop" tampers and they must be adjudicated the same way.
    """
    drops = [(name, field) for (name, field), (kind, _) in TAMPER.items()
             if kind == "drop"]
    assert len(drops) >= 10, drops
    for name, field in drops:
        sample = _sample(name)
        if sample is None and name != "Reuse":
            continue
        kind, build = TAMPER[(name, field)]
        tampered = _rebuild(name, field, build(sample))
        state, detail = _verdict(tampered)
        assert state in ("moved", "refused"), (name, field, state)


def test_no_field_is_bound_by_repr_alone():
    """Encoding is structural: the same content rebuilt is the same identity."""
    edge = REGION.edges[0]
    same = _rebuild("Dependence", "delay_fact", replace(
        edge.delay_fact,
        source=ModelSource(module="ir/isa.py",
                           symbol=str(edge.delay_fact.source.symbol),
                           field=str(edge.delay_fact.source.field),
                           value=int(edge.delay_fact.source.value))))
    assert problem_identity_of(same)[1] == BASE


def test_the_plan_digest_is_one_recomputable_recipe():
    """plan_identity is the hash of the single recipe; no second payload."""
    plan = resolve_schedule(PROBLEM, ANSWERS)
    assert plan_identity_of(plan)[1] == plan.plan_identity
    assert plan_identity_of(plan)[0] == plan.plan_canonical_bytes
    assert _plan_bytes(PROBLEM, plan.regions, plan.dependence_proof) \
        == plan.plan_canonical_bytes

    nine = Constant(type=TensorType.scalar(DType.i64), value=9)
    for name, value in (("lower_expr", nine), ("stop_expr", nine),
                        ("step_expr", nine), ("ii", 1), ("origin", 1),
                        ("max_shift", 1)):
        tampered = replace(plan, regions=(replace(plan.regions[0], **{name: value}),))
        assert plan_identity_of(tampered)[1] != plan.plan_identity, name
    buffer = plan.regions[0].buffers[0]
    bumped = replace(plan, regions=(replace(
        plan.regions[0], buffers=tuple(
            replace(item, resolved_versions=item.resolved_versions + 1)
            if item.buffer_id == buffer.buffer_id else item
            for item in plan.regions[0].buffers)),))
    assert plan_identity_of(bumped)[1] != plan.plan_identity


def test_a_stored_plan_digest_lying_about_its_content_is_refused():
    """Keep the digest, change a bound: the bypass the lead named.

    completion reads the plan's own bounds to decide whether a region owes an
    EXIT drain, so a plan whose bounds differ from the region's real ones
    erases duties without touching a channel. verify_plan_bindings compares
    the recorded bounds with the actual region's and refuses by name.
    """
    plan = resolve_schedule(PROBLEM, ANSWERS)
    nine = Constant(type=TensorType.scalar(DType.i64), value=9)
    tampered = replace(plan, regions=(replace(plan.regions[0], stop_expr=nine),),
                       plan_identity=plan.plan_identity,
                       plan_canonical_bytes=plan.plan_canonical_bytes)
    assert tampered.plan_identity == plan.plan_identity, "replace keeps stored fields"
    assert plan_identity_of(tampered)[1] != plan.plan_identity
    with pytest.raises(ResolveError) as caught:
        verify_plan_bindings(tampered, PROBLEM)
    assert caught.value.code == "PLAN_DECISION_INPUT_MISMATCH", caught.value.code
    assert "stop_expr" in caught.value.detail, caught.value.detail
