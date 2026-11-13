"""WP3 register-budget gate: facts on one side, a lowering report on the other.

WHY THIS EXISTS. storage used to answer UNKNOWN_CAPACITY for rmem/carrier forever,
because the target states no BYTE capacity for them. That refusal is honest but it
was also permanent: register spaces are COUNTED resources, and no amount of byte
capacity will ever describe them. A permanent refusal is not fail-closed - it is
indistinguishable from every other permanent refusal and it hides which side is
missing.

THE TWO SIDES, and why both are required:
  fact side    - memory.REGISTER_BUDGETS: how many registers one execution domain
                 has, with value/source/status. A 256 B register WIDTH is not a
                 budget, and this file asserts nothing ever divides one by the other.
  report side  - storage.RegisterOccupancy: what the lowering measured for the
                 candidate it produced (count, dtype, lanes, cross-domain residency,
                 spill path).

  either side missing        -> STORAGE_REGISTER_UNSOURCED
  both stated, demand > budget -> STORAGE_REGISTER_OVER_BUDGET
  both stated, demand <= budget -> placed and VERIFIED, and never via byte capacity

WHAT IS TRUE TODAY, stated so this file cannot be read as more than it is:
every entry in REGISTER_BUDGETS is UNSOURCED, because this build genuinely does not
state a per-domain register COUNT (only a per-register WIDTH). So the positive test
below SUBSTITUTES a budget through the documented facts table - it replaces the
facts entry, exercises the real gate functions, and restores it. That is not a
monkeypatch of the logic under test: place_resolved_versions, _register_gaps and
verify_storage_current are the real ones, and the substituted entry is a real
RegisterBudgetFacts object flowing through the same code path. The day a sourced
number lands, the first test below starts failing and says to switch to it.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tilefoundry.analysis import analyze
from tilefoundry.cli.source import checked_program
from tilefoundry.schedule import finalize

import tilefoundry_ascend.memory as memory
from tilefoundry_ascend.storage import (
    RegisterOccupancy,
    StorageError,
    StorageLowering,
    place_resolved_versions,
    require_storage_verified,
    verify_storage_current,
)

AUTHORS = Path(__file__).resolve().parent / "fixtures" / "authors"
INDEPENDENT_UB = "%s:IndependentUB" % (AUTHORS / "independent_ub.py")
COPY_ONLY = "%s:CopyOnly" % (AUTHORS / "copy_only.py")


def _chain(path):
    """A real plan and a verified completion for one authored fixture."""
    from tilefoundry_ascend.completion import (build_symbolic_completion,
                                              verify_completion)
    from tilefoundry_ascend.plan.numeric import to_schedule_problem
    from tilefoundry_ascend.plan.problem import analyze_native_problem
    from tilefoundry_ascend.plan.resolved import resolve_schedule
    from tilefoundry_ascend.plan.units import build_native_unit_tree
    from tilefoundry_ascend.solver import solve_region

    program = checked_program(path, None, command="test register budget")
    module = finalize(analyze(program.module, program.entry,
                              analysis=("compute-cost", "memory")),
                      authored=program.authored)
    entry = module.entry_function()
    problem = analyze_native_problem(build_native_unit_tree(module), None, None,
                                     entry.target)
    plan = resolve_schedule(problem, {
        region.region_ref.region_id: solve_region(to_schedule_problem(region))
        for region in problem.regions})
    effects = tuple(effect for region in problem.regions for effect in region.effects)
    completion = verify_completion(
        plan, build_symbolic_completion(plan, entry.target, effects), entry.target)
    return plan, completion, entry.target


@pytest.fixture()
def sourced_budget():
    """Install a sourced aiv/vreg budget, then restore the table exactly."""
    original = dict(memory.REGISTER_BUDGETS)
    memory.REGISTER_BUDGETS[("aiv", "vreg")] = replace(
        original[("aiv", "vreg")],
        value=16, source="test fixture: a stated per-domain count",
        status=memory.STATUS_REFERENCE_IMPLEMENTATION)
    try:
        yield 16
    finally:
        memory.REGISTER_BUDGETS.clear()
        memory.REGISTER_BUDGETS.update(original)

def test_no_register_budget_is_declared_today_and_a_width_is_not_a_count():
    """The fact side is honestly empty, and it cannot be derived from a width.

    This is the measured state, not a placeholder: it must be UPDATED, not deleted,
    when a real per-domain register count arrives. The second half is the important
    one - it pins that the 256 B vreg WIDTH is a different fact from a register
    COUNT, so no arithmetic can convert one into the other and let the gate pass on
    a number nobody stated.
    """
    for (domain, register_class), facts in memory.REGISTER_BUDGETS.items():
        assert facts.value is None, (
            "a register budget is now DECLARED for %s/%s (value=%r, source=%r). If "
            "that is a real sourced fact, switch the positive test below to use it "
            "instead of the substituted one." % (domain, register_class, facts.value,
                                                facts.source))
        assert facts.status == memory.STATUS_REGISTER_UNSOURCED, (
            domain, register_class, facts.status)
        assert facts.reason, "an unsourced budget must say WHY it is unsourced"
        assert not facts.is_sourced()

    width = memory.SPACE_FACTS["ascend.vreg"].nominal_bytes
    assert width == 256, width
    budget = memory.register_budget("aiv", "vreg")
    assert budget.value is None
    assert budget.register_class == "vreg"

    # With no fact and no report, the gate refuses by name at placement.
    plan, completion, target = _chain(INDEPENDENT_UB)
    with pytest.raises(StorageError) as refused:
        place_resolved_versions(plan, completion, target, StorageLowering())
    assert refused.value.code == "STORAGE_REGISTER_UNSOURCED", refused.value.code


def test_each_side_alone_is_unsourced_and_both_are_needed():
    """The gate is a CONJUNCTION: facts alone and report alone both refuse.

    A gate that passed on one side would be the "requirement backfilled as a
    capacity" failure in a new place: the target would claim registers exist
    without the candidate demand being measured, or the demand would be compared
    against nothing.
    """
    plan, completion, target = _chain(INDEPENDENT_UB)
    report_only = StorageLowering(register_occupancy=(
        RegisterOccupancy("aiv", "vreg", count=4, source="lowering measured 4"),))

    # (a) report present, fact missing.
    with pytest.raises(StorageError) as no_fact:
        place_resolved_versions(plan, completion, target, report_only)
    assert no_fact.value.code == "STORAGE_REGISTER_UNSOURCED"
    assert "register budget" in str(no_fact.value), str(no_fact.value)[:250]

    original = dict(memory.REGISTER_BUDGETS)
    try:
        memory.REGISTER_BUDGETS[("aiv", "vreg")] = replace(
            original[("aiv", "vreg")], value=16,
            source="test fixture: a stated per-domain count",
            status=memory.STATUS_REFERENCE_IMPLEMENTATION)

        # (b) fact present, report missing.
        with pytest.raises(StorageError) as no_report:
            place_resolved_versions(plan, completion, target, StorageLowering())
        assert no_report.value.code == "STORAGE_REGISTER_UNSOURCED"
        assert "occupancy report" in str(no_report.value), str(no_report.value)[:250]

        # (c) both present -> the POSITIVE path, all the way to verified.
        placed = place_resolved_versions(plan, completion, target, report_only)
        assert placed.entries, "a satisfied register gate must still place the plan"
        verified = verify_storage_current(plan, placed, target,
                                          completion=completion, lowering=report_only)
        again = require_storage_verified(verified, plan=plan, completion=completion,
                                         target=target, lowering=report_only)
        assert again.identity == verified.identity
        # A counted resource must not acquire a byte capacity through the record.
        register_rows = [row for row in verified.capacities
                         if "register-budget" in str(row.source)]
        assert register_rows, [row.source for row in verified.capacities]
        assert all(row.usable_bytes == 0 for row in register_rows), (
            [(row.source, row.usable_bytes) for row in register_rows])
    finally:
        memory.REGISTER_BUDGETS.clear()
        memory.REGISTER_BUDGETS.update(original)


def test_demand_above_the_declared_budget_is_refused_not_clipped():
    """Over budget names both numbers; nothing is silently reduced."""
    plan, completion, target = _chain(INDEPENDENT_UB)
    original = dict(memory.REGISTER_BUDGETS)
    try:
        memory.REGISTER_BUDGETS[("aiv", "vreg")] = replace(
            original[("aiv", "vreg")], value=8,
            source="test fixture: a deliberately small stated budget",
            status=memory.STATUS_REFERENCE_IMPLEMENTATION)
        demand = StorageLowering(register_occupancy=(
            RegisterOccupancy("aiv", "vreg", count=9, source="lowering"),))
        with pytest.raises(StorageError) as refused:
            place_resolved_versions(plan, completion, target, demand)
        assert refused.value.code == "STORAGE_REGISTER_OVER_BUDGET"
        text = str(refused.value)
        assert "demand=9" in text and "budget=8" in text, text[:250]
    finally:
        memory.REGISTER_BUDGETS.clear()
        memory.REGISTER_BUDGETS.update(original)


def test_cross_domain_residency_without_a_spill_path_stays_unsourced():
    """A 16-bit cross-domain shape cannot be guessed into a budget.

    Carrier registers transport bf16 windows ACROSS execution domains, so one
    residency consumes budget on both sides; with no spill path the demand cannot be
    relieved. This build states no carrier count, so the honest answer is UNSOURCED
    rather than a computed number.

    Tested at the GATE, not through placement, and the difference matters: no
    fixture in this tree places a carrier space at all (measured - independent_ub
    places gmem/rmem/smem), so a placement-level test could never reach this branch.
    My first version did drive placement and "passed" because the refusal came from
    vreg, not from carrier - a false positive. Calling _register_gaps directly is
    the honest scope for a space no current program uses.
    """
    from tilefoundry_ascend.storage import _register_gaps

    _plan, _completion, target = _chain(INDEPENDENT_UB)
    carrier = StorageLowering(register_occupancy=(
        RegisterOccupancy("aiv", "carrier", count=2, dtype="bf16", lane_count=128,
                          cross_domain_residency=True, spill_path=None,
                          source="lowering: bf16 carrier window"),))
    gaps = _register_gaps(target, "carrier", carrier)
    assert gaps, "a carrier residency with no declared budget must not pass"
    codes = [code for code, _message, _where in gaps]
    assert codes == ["STORAGE_REGISTER_UNSOURCED"], codes
    assert "carrier" in gaps[0][1], gaps[0]
    assert "carrier" in gaps[0][2], gaps[0]

    # And the report is not silently treated as zero demand: a carrier space with NO
    # report at all is also unsourced, not "nothing to check".
    silent = StorageLowering()
    assert [code for code, _m, _w in _register_gaps(target, "carrier", silent)] == [
        "STORAGE_REGISTER_UNSOURCED"]

def test_a_register_free_program_still_places_normally():
    """The gate must not poison programs that need no registers."""
    plan, completion, target = _chain(COPY_ONLY)
    placed = place_resolved_versions(plan, completion, target, StorageLowering())
    assert placed.entries, "a register-free program must place without the gate"
    spaces = {requirement.space for requirement in placed.requirements}
    assert "rmem" not in spaces, sorted(spaces)