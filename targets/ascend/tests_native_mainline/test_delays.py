"""Sourced delays: three states, one explicit predicate, no silent zero.

The tests here are the negative controls for the lead's ruling D1-D5: a delay
that is not stated may never become 0 or 1 on its way into a constraint, and
editing the registered table must move the problem identity.
"""
from dataclasses import replace

import pytest

from tilefoundry.analysis import analyze
from tilefoundry.cli.source import checked_program
from tilefoundry.schedule import finalize
from tilefoundry_ascend.plan.delays import (
    DECLARED,
    DelayFact,
    DelayRefusal,
    MODEL_COST,
    ModelConflict,
    ModelSource,
    OCCUPANCY_FALLBACK,
    PROVEN_ZERO,
    UNSOURCED,
    declared_occupancy,
    delay_table_bytes,
    edge_delay_fact,
    proven_zero_predicate,
    reuse_delay_fact,
)
from tilefoundry_ascend.plan.numeric import to_core, to_schedule_problem
from tilefoundry_ascend.plan.problem import Dependence, analyze_native_problem
from tilefoundry_ascend.plan.units import build_native_unit_tree
from tilefoundry_ascend.solver import solve_region

AUTHOR = "targets/ascend/tests_native_mainline/fixtures/authors/minimal_aiv.py:MinimalAivPipeline"


def _problem():
    program = checked_program(AUTHOR, None, command="test delays")
    analyzed = analyze(program.module, program.entry,
                       analysis=("compute-cost", "memory"))
    module = finalize(analyzed, authored=program.authored)
    tree = build_native_unit_tree(module, authored=program.authored)
    return analyze_native_problem(tree, None, None, module.entry_function().target)


def _loop_region(problem):
    return next(r for r in problem.regions if r.region_ref.induction_var_ref is not None)


# ------------------------------------------------------------------- states
def _same_owner():
    from tilefoundry_ascend.plan.units import OwnerKey

    return OwnerKey(None, "AIV", None)


def test_proven_zero_needs_every_clause_of_the_declared_family():
    owner = _same_owner()
    call = dict(producer_pipe="vector", producer_completion="issue",
                consumer_pipe="vector", same_region=True, straight_line=True,
                producer_owner=owner, consumer_owner=owner, grid_instances=1)
    proof = proven_zero_predicate(**call)
    assert proof, "one AIV instance, same pipe, issue-complete order is provable"
    # The proof is (field, value) data, never a sentence: an identity-bound
    # proof must not be movable by wording.
    clauses = dict(proof)
    assert clauses["grid_instances"] == 1
    assert clauses["same_region"] is True and clauses["straight_line"] is True
    assert clauses["distance"] == 0
    assert clauses["pipe"] == "vector" and clauses["producer_completion"] == "issue"
    assert clauses["visible_at_issue"] is True and clauses["same_pipe_order"] is True
    # the instance is named by core_kind AND sid, not by a formatted string
    assert clauses["core_kind"] == "AIV" and "sid" in clauses
    assert all(isinstance(field, str) for field, _ in proof)
    for kwargs in (
        {"producer_completion": "write_visible"},
        {"producer_pipe": "mte2", "consumer_pipe": "mte2"},
        {"same_region": False},
        {"straight_line": False},
        {"distance": 1},
        {"consumer_pipe": "scalar"},
        {"grid_instances": 2},
        {"grid_instances": 0},
    ):
        bad = dict(call)
        bad.update(kwargs)
        assert not proven_zero_predicate(**bad), kwargs


def test_proven_zero_requires_the_same_core_instance_not_just_the_kind():
    from tilefoundry_ascend.plan.units import OwnerKey

    owner = _same_owner()
    base = dict(producer_pipe="vector", producer_completion="issue",
                consumer_pipe="vector", same_region=True, straight_line=True,
                grid_instances=1)
    # Same core KIND, different sid: two AIV instances share a pipe name and
    # share no ordering guarantee, so this must NOT be proven zero.
    other = OwnerKey(None, "AIV", 1)
    assert not proven_zero_predicate(producer_owner=owner, consumer_owner=other, **base)
    edge = edge_delay_fact(producer_instruction="vadd", producer_pipe="vector",
                           producer_completion="issue", consumer_pipe="vector",
                           producer_occupancy=declared_occupancy("vadd"),
                           same_region=True, straight_line=True,
                           producer_owner=owner, consumer_owner=other,
                           grid_instances=1)
    assert edge.basis == OCCUPANCY_FALLBACK and edge.ticks == 3
    # Same instance but a different pipe is still refused.
    assert not proven_zero_predicate(producer_owner=owner, consumer_owner=owner,
                                     **dict(base, consumer_pipe="scalar"))
    # An owner that states nothing is a named refusal, never an assumption.
    with pytest.raises(DelayRefusal) as caught:
        proven_zero_predicate(producer_owner=None, consumer_owner=owner, **base)
    assert caught.value.code == "DELAY_OWNER_MISSING"
    # And a grid whose instance count is not explicit refuses too.
    with pytest.raises(DelayRefusal) as caught:
        proven_zero_predicate(producer_owner=owner, consumer_owner=owner,
                             **dict(base, grid_instances=None))
    assert caught.value.code == "DELAY_GRID_UNSOURCED"


def test_reuse_never_proves_zero_from_the_grid_instance_count():
    owner = _same_owner()
    base = dict(same_region=True, straight_line=True, grid_instances=1,
                producer_owner=owner, consumer_owner=owner)
    # one instance is necessary but not sufficient: reuse crosses generations
    assert not proven_zero_predicate(producer_pipe="vector",
                                     producer_completion="issue",
                                     consumer_pipe="vector", distance=1, **base)


def test_reuse_is_never_in_the_proven_zero_family():
    fact = reuse_delay_fact(producer_instruction="vsts", producer_pipe="vector",
                            producer_completion="issue",
                            producer_occupancy=declared_occupancy("vsts"),
                            distance_offset=0)
    assert fact.state != PROVEN_ZERO
    assert fact.resolve() == declared_occupancy("vsts").resolve()


def test_unsourced_delay_refuses_and_never_projects_a_number():
    fact = DelayFact(state=UNSOURCED, ticks=None, basis=UNSOURCED, unit="",
                     source=ModelSource(module="plan/delays.py", symbol="row",
                                        field="occupancy_row", value=None),
                     status=UNSOURCED)
    with pytest.raises(DelayRefusal) as caught:
        fact.resolve()
    assert caught.value.code == "DELAY_UNSOURCED"
    # a zero that is not PROVEN_ZERO is the fabrication this module forbids
    forged = DelayFact(state=DECLARED, ticks=0, basis=DECLARED, unit="MODEL_TICK",
                       source=ModelSource(module="m", symbol="s", field="f", value=0),
                       status="x")
    with pytest.raises(DelayRefusal) as caught:
        forged.resolve()
    assert caught.value.code == "DELAY_ZERO_UNJUSTIFIED"
    # so is a number without a unit
    unitted = DelayFact(state=DECLARED, ticks=4, basis=DECLARED, unit="",
                        source=ModelSource(module="m", symbol="s", field="f", value=4),
                        status="x")
    with pytest.raises(DelayRefusal) as caught:
        unitted.resolve()
    assert caught.value.code == "DELAY_UNIT_UNKNOWN"


def test_a_locator_must_be_a_record_and_never_prose():
    """Wording cannot be allowed back into identity bytes through a string.

    The lead's ruling is bidirectional: a reworded explanation must not move
    an identity, and a locator must still name where the number came from.
    Both hold only if the field is a record, so a str is refused at
    construction instead of being hashed as text.
    """
    with pytest.raises(TypeError):
        DelayFact(state=DECLARED, ticks=4, basis=DECLARED, unit="MODEL_TICK",
                  source="ISA says the latency is four", status="x")
    with pytest.raises(TypeError):
        DelayFact(state=DECLARED, ticks=4, basis=DECLARED, unit="MODEL_TICK",
                  source=ModelSource("m", "s", "f", 4), status="x",
                  uncertainty="the pipe disagrees")


# ---------------------------------------------------------------- occupancy
def test_pset_occupancy_is_an_explicit_model_row_not_a_fabricated_tick():
    fact = declared_occupancy("pset")
    assert fact.state == MODEL_COST and fact.basis == MODEL_COST
    # the locator names the profile it came from; nothing here is a sentence
    assert fact.source.module == "plan/delays.py" and fact.source.symbol == "pset"
    assert fact.source.field == "model_profile_window"
    assert fact.source.value == 1
    # and the ISA really declares none, which is why this row exists at all
    from tilefoundry_ascend.ir.isa import ISA
    assert ISA["pset"].latency_model is None
    assert ("isa_latency_model", None) in fact.proof
    assert fact.resolve() == fact.ticks
    missing = declared_occupancy("no_such_instruction")
    assert missing.state == UNSOURCED
    assert missing.source.field == "occupancy_row" and missing.source.value is None
    # the VF window is the sum of its registered members, from the table
    problem = _problem()
    vf = next(task for task in _loop_region(problem).tasks
              if len(task.uses) == 1 and str(task.uses[0].resource.pipe) == "vector")
    members = sum(declared_occupancy(name).resolve()
                  for name in ("vld", "vld", "pset", "vadd", "vsts"))
    assert vf.uses[0].duration == members


# -------------------------------------------------------- problem projection
def test_real_edges_carry_a_sourced_fact_and_no_bare_zero():
    problem = _problem()
    for region in problem.regions:
        for edge in region.edges:
            assert isinstance(edge.delay_fact, DelayFact), edge
            assert edge.delay_fact.unit == "MODEL_TICK"
            assert edge.delay_fact.basis in (PROVEN_ZERO, DECLARED,
                                             OCCUPANCY_FALLBACK, MODEL_COST)
            assert edge.delay == edge.delay_fact.resolve()


def test_an_unstated_edge_is_refused_at_the_projection():
    region = _loop_region(_problem())
    silent = replace(region.edges[0], delay=None, delay_fact=None)
    with pytest.raises(Exception) as caught:
        to_core(replace(region, edges=(silent, *region.edges[1:])))
    assert caught.value.code == "DELAY_UNSOURCED"
    # the solver view keeps working when the fact is present
    assert to_core(region).edges


def test_the_edge_delay_reaches_the_solver_constraint():
    region = _loop_region(_problem())
    natural = solve_region(to_schedule_problem(region))
    assert natural.status == "SAT", natural.detail
    tick = region.edges[0].delay
    tightened = replace(region, fixed_ii=tick, ii_min=tick, ii_max=tick)
    assert solve_region(to_schedule_problem(tightened)).status == "UNSAT"
    assert solve_region(to_schedule_problem(region)).ii == tick + 1


def test_editing_the_delay_table_changes_the_problem_identity():
    from tilefoundry_ascend.plan import delays

    before = _problem().problem_identity
    row = delays._OCCUPANCY["gm_to_ub"]
    edited = replace(row, ticks=row.ticks + 1)
    delays._OCCUPANCY["gm_to_ub"] = edited
    try:
        after = _problem().problem_identity
    finally:
        delays._OCCUPANCY["gm_to_ub"] = row
    assert before != after, "a tick value must be identity-bearing"
    assert before == _problem().problem_identity, "and stable once restored"

    status_row = replace(row, status="DEVICE_VALIDATED")
    delays._OCCUPANCY["gm_to_ub"] = status_row
    try:
        assert _problem().problem_identity != before
    finally:
        delays._OCCUPANCY["gm_to_ub"] = row
    assert delay_table_bytes()


# ------------------------------------------------------------------- vld D5
def test_vld_completion_conflict_is_recorded_as_fields_not_words():
    fact = declared_occupancy("vld")
    conflict = fact.uncertainty
    assert isinstance(conflict, ModelConflict), conflict
    # the field that disagrees, what each side states, and which one applies
    assert conflict.conflict_field == "completion"
    assert conflict.declared == "issue+data_ready"
    assert conflict.other_field == "PIPE_FACTS[vector].visible_at_issue"
    assert conflict.other_claim == "True"
    assert conflict.resolution == "DECLARATION_WINS"
    # the disagreement is recorded, and it never relaxes a delay
    assert fact.uncertainty is not None and fact.ticks == 4
    # an edge produced by vld is still NOT proven zero: it takes the model
    # bound, and the recorded conflict travels with it as data
    owner = _same_owner()
    edge = edge_delay_fact(producer_instruction="vld", producer_pipe="vector",
                           producer_completion="READ_COMPLETE", consumer_pipe="vector",
                           producer_occupancy=fact, same_region=True, straight_line=True,
                           producer_owner=owner, consumer_owner=owner,
                           grid_instances=1)
    assert edge.basis == OCCUPANCY_FALLBACK
    assert edge.resolve() == fact.resolve()
    assert edge.uncertainty is conflict
    # and the same shape with the pipe's own issue completion IS proven zero
    zero = edge_delay_fact(producer_instruction="pset", producer_pipe="vector",
                           producer_completion="issue", consumer_pipe="vector",
                           producer_occupancy=declared_occupancy("pset"),
                           same_region=True, straight_line=True,
                           producer_owner=owner, consumer_owner=owner,
                           grid_instances=1)
    assert zero.state == PROVEN_ZERO and zero.resolve() == 0


def test_wording_never_moves_the_delay_table_but_any_struct_change_does():
    """The bidirectional property: rephrasing cannot invalidate stored plans.

    Renaming a module, symbol or field, or changing a value, is a real change
    and must move the bytes. Sentence builders (report()) live outside
    canonical(), so no wording is hashed at all.
    """
    from tilefoundry_ascend.plan import delays

    baseline = delays.delay_table_bytes()
    assert b"UNRESOLVED" not in baseline and b"never projected" not in baseline

    original = delays._OCCUPANCY["vld"]
    # same values, different object identity: the bytes are the bytes
    delays._OCCUPANCY["vld"] = replace(original, ticks=original.ticks)
    try:
        assert delays.delay_table_bytes() == baseline
    finally:
        delays._OCCUPANCY["vld"] = original

    row = original
    mutations = (
        ("locator value", replace(row, source=replace(row.source, value=5))),
        ("locator field", replace(row, source=replace(row.source, field="ii_model"))),
        ("locator symbol", replace(row, source=replace(row.source, symbol="vldx"))),
        ("locator module", replace(row, source=replace(row.source,
                                                       module="ir/isa.py.bak"))),
        ("registered window", replace(row, ticks=row.ticks + 1)),
        ("conflict resolution", replace(row, uncertainty=replace(
            row.uncertainty, resolution="PIPE_FACT_WINS"))),
        ("conflict claim", replace(row, uncertainty=replace(row.uncertainty,
                                                            other_claim="False"))),
        ("proof clause", replace(row, proof=tuple(
            ("latency_model", 999) if field == "latency_model" else (field, value)
            for field, value in row.proof))),
    )
    for label, changed in mutations:
        delays._OCCUPANCY["vld"] = changed
        try:
            assert delays.delay_table_bytes() != baseline, label
        finally:
            delays._OCCUPANCY["vld"] = row
    assert delays.delay_table_bytes() == baseline


def test_report_wording_is_available_without_entering_the_bytes():
    """Refusals and reports still read well; only data is hashed."""
    from tilefoundry_ascend.plan import delays

    row = delays._OCCUPANCY["vld"]
    sentence = row.uncertainty.report()
    assert "UNRESOLVED" in sentence and "completion" in sentence
    assert sentence.encode() not in delays.delay_table_bytes()
    assert row.source.report().encode() not in delays.delay_table_bytes()
