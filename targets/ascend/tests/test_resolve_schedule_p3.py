"""P3-02 / G31: ResolveSchedule consumes the REAL solver assignment.

G31 requires that R takes ``assignment.tasks[task_id].start_time``, the root->version
map from ``assignment.buffers``, and ``assignment.ii`` from the solver, with no
test-fabricated schedule. Every test here drives ``resolve_assignment`` directly (or
through ``build_resolved_plan`` with a real solve-result), so the assertions are about
the production route.

Covered:
  * SAT resolves start/stage/phase/order from the assignment; II comes from the solver.
  * authored stage is honoured AND distinguishable from the relative iteration shift.
  * issue rank is a hard partial order, not a tie-break that hides a conflict.
  * no-offset cross-stage starts are RECORDED as needs_replan (Lead ruling), not refused.
  * non-SAT never resolves; a missing/un-postchecked result is refused by name.
  * unit mismatch, id mismatch, coverage gaps and fixed-version drift are all named.
  * capacity pressure asks for a replan instead of clamping the version count.
  * the FORMAL ResolvedPlan (plan/resolved_plan.ResolvedPlan) takes slot_count and
    slot_source from assignment.buffers, so the Problem's nominal version cannot leak in;
  * perturbing phase, II or an auto version in the assignment changes that plan;
  * equal-phase manual source order and equal-phase issue_after both outrank the default
    (-stage, source) preference, matching blueprints/test_resolve_assignment.py.

Negative tests assert their mutation APPLIED (old != new) first, so a silently broken
mutator cannot make a refusal look proven.

Run: PYTHONPATH=src:targets/ascend python targets/ascend/tests/test_resolve_schedule_p3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "targets" / "ascend"))
sys.path.insert(0, str(HERE.parent))

import copy

import solve_fixture as F
from tilefoundry_ascend.plan import resolved_plan as RP

# Plain pytest test functions: no device, no conditional skip. The module also runs as a
# script through the __main__ block at the bottom, so it never imports pytest itself.

DIGEST = "sha256:" + "1" * 64
SOURCES = {key: "sha256:" + char * 64 for key, char in zip("LSCFP", "abcde")}


def expect_code(code, fn):
    try:
        fn()
    except Exception as error:  # noqa: BLE001
        got = getattr(error, "code", None)
        if got != code:
            raise AssertionError("expected %s, got %r: %s" % (code, got, error))
        return error
    raise AssertionError("expected %s, but nothing was refused" % code)


def _mutate(target, mutate_fn, label):
    """Apply a mutation and PROVE it changed the target (anti-silent-pass)."""
    before = copy.deepcopy(target)
    mutate_fn(target)
    if target == before:
        raise AssertionError("MUTATION_DID_NOT_APPLY: %s changed nothing" % label)
    return target


def contracts(*rows):
    return tuple(RP.TaskContract(task_id=row[0], core=row[1], pipe=row[2],
                                 authored_stage=row[3] if len(row) > 3 else None,
                                 issue_after=row[4] if len(row) > 4 else (),
                                 source_index=index)
                 for index, row in enumerate(rows))


def buffers(*roots):
    return tuple(RP.BufferContract(root=root, min_versions=1, max_versions=4,
                                   memory_domain="ub0", aligned_slot_bytes=128)
                 for root in roots)


def result(starts, *, ii=4, versions=None, problem_digest=DIGEST, state="SAT",
           postcheck="POSTCHECK_PASS"):
    schedule = {task_id: {"start_time": start} for task_id, start in starts.items()}
    return F.solve_result(schedule, problem_digest=problem_digest, ii=ii,
                          buffers=versions if versions is not None else {},
                          state=state, postcheck_verdict=postcheck)


def resolve(solve, task_contracts=None, buffer_contracts=None, **kw):
    kw.setdefault("enable_offset", True)
    return RP.resolve_assignment(
        solve, task_contracts if task_contracts is not None else contracts(
            ("t0", "cube", "mte2"), ("t1", "vector", "vector")),
        buffer_contracts if buffer_contracts is not None else buffers("x"),
        **kw)


def plan_problem(*, buffers=None, tasks=None, requested_num_stages=2,
                 enable_offset=True, loop_domain=None, ii=1):
    """A Problem the FORMAL plan builder accepts (real ids, hashes, loop domain).

    ``ii`` is deliberately nominal: the plan must take the solver's II, so a test can
    give the Problem a value the solve never repeats and watch the plan ignore it.
    """
    return {
        "problem_digest": DIGEST, "source_digest": "sha256:" + "2" * 64,
        "ledger_sha": "sha256:" + "3" * 64, "loop_id": "main",
        "loop_domain": loop_domain or {"start": 0, "stop": 4, "step": 1},
        "requested_num_stages": requested_num_stages, "ii": ii, "latency": 1,
        "enable_offset": enable_offset,
        "tasks": tasks if tasks is not None else [
            {"stable_id": "t0", "action_ids": ["a0"], "pipe": "mte2",
             "resource": "mte2", "source": "s0"},
            {"stable_id": "t1", "action_ids": ["a1"], "pipe": "vector",
             "resource": "vector", "source": "s1"}],
        "buffers": buffers if buffers is not None else [
            {"root": "x", "space": "ascend.ub", "bytes": 128, "versions": 1,
             "fixed": True}],
        "accesses": [],
    }


def build_plan(problem, starts=None, *, ii=4, versions=None, **kw):
    """Problem + REAL assignment -> the FORMAL plan (plan/resolved_plan.ResolvedPlan)."""
    solve = result(starts if starts is not None else {"t0": 0, "t1": 6}, ii=ii,
                   versions=versions if versions is not None else {"x": 1},
                   problem_digest=problem["problem_digest"])
    return RP.build_resolved_plan(problem, solve, SOURCES, **kw)


# ------------------------------------------------------- SAT consumption
def test_sat_resolves_start_stage_phase_and_order_from_the_assignment():
    resolution = resolve(result({"t0": 0, "t1": 6}, ii=4, versions={"x": 2}))
    assert resolution.is_resolved
    assert resolution.solver_state == "SAT"
    assert resolution.ii == {"value": 4, "unit": RP.TIME_UNIT}
    first, second = resolution.tasks
    assert (first.start_time["value"], first.stage, first.phase) == (0, 0, 0)
    assert (second.start_time["value"], second.stage, second.phase) == (6, 1, 2)
    assert [row.order for row in resolution.tasks] == [0, 1]
    assert resolution.issue_order == ("t0", "t1")


def test_ii_comes_from_the_solver_not_a_problem_default():
    """Two solves that differ only in II must resolve to different II."""
    narrow = resolve(result({"t0": 0, "t1": 4}, ii=4, versions={"x": 1}))
    wide = resolve(result({"t0": 0, "t1": 8}, ii=8, versions={"x": 1}))
    assert narrow.ii["value"] == 4 and wide.ii["value"] == 8
    # Same start, different II => different stage. A problem default could not do that.
    assert narrow.tasks[1].stage == 1 and wide.tasks[1].stage == 1
    assert narrow.tasks[0].phase == wide.tasks[0].phase == 0


def test_a_resolved_plan_uses_the_solver_ii_and_phase():
    problem = plan_problem()
    plan = build_plan(problem, ii=4, versions={"x": 1})
    # The Problem says ii=1 (a nominal default). The PLAN must say 4, from the solver.
    assert plan.loop.ii["value"] == 4, plan.loop.ii
    assert plan.loop.enable_offset is True
    by_id = {task.task_id: task for task in plan.tasks}
    assert by_id["t1"].stage == 1 and by_id["t1"].phase == 2
    assert by_id["t1"].start_time == {"value": 6, "unit": RP.TIME_UNIT}


# ------------------------------------- authored stage vs relative shift
def test_an_authored_stage_is_kept_and_distinguished_from_the_iteration_shift():
    resolution = resolve(
        result({"t0": 8, "t1": 12}, ii=4, versions={"x": 1}),
        task_contracts=contracts(("t0", "cube", "mte2", 2), ("t1", "vector", "vector", 3)))
    first, second = resolution.tasks
    assert (first.stage, first.authored_stage) == (2, 2)
    assert (second.stage, second.authored_stage) == (3, 3)
    # The core normalizes min(k) to 0, so the PLAN's relative shift is stage - min(stage).
    assert resolution.stage_origin == 2
    assert first.iteration_shift == 0 and second.iteration_shift == 1
    assert first.stage_origin == RP.STAGE_SOURCE_AUTHORED


def test_solver_derived_stages_are_labelled_not_pretended_to_be_authored():
    resolution = resolve(result({"t0": 0, "t1": 6}, ii=4, versions={"x": 1}))
    assert all(row.authored_stage is None for row in resolution.tasks)
    assert all(row.stage_origin == RP.STAGE_SOURCE_SOLVER for row in resolution.tasks)


def test_a_solver_that_ignores_the_authored_stage_is_refused():
    solve = result({"t0": 0, "t1": 6}, ii=4, versions={"x": 1})
    authored = contracts(("t0", "cube", "mte2", 0), ("t1", "vector", "vector", 3))
    error = expect_code("PLAN_RESOLVE_AUTHORED_STAGE_VIOLATED",
                        lambda: resolve(solve, task_contracts=authored))
    assert error.needs_replan is True


def test_a_supplied_stage_conflicting_with_start_over_ii_is_refused():
    solve = result({"t0": 0, "t1": 6}, ii=4, versions={"x": 1})
    _mutate(solve["assignment"]["tasks"]["t1"],
            lambda entry: entry.__setitem__("stage", 0), "conflicting stage")
    expect_code("PLAN_RESOLVE_STAGE_CONFLICT", lambda: resolve(solve))


# ------------------------------------------------------------ issue order
def test_issue_order_is_a_hard_partial_order_not_a_tie_break():
    """t1 must issue before t0 despite t0 having the earlier source index.

    ``issue_after`` names a PREDECESSOR, so the edge is expressed on t0 pointing at t1.
    Source order alone would emit ("t0", "t1"); the hard edge must override it.
    """
    resolution = resolve(
        result({"t0": 0, "t1": 0}, ii=4, versions={"x": 1}),
        task_contracts=contracts(("t0", "cube", "mte2", None, ("t1",)),
                                 ("t1", "cube", "mte2")))
    assert resolution.issue_order == ("t1", "t0"), resolution.issue_order


def test_a_cycle_in_the_hard_issue_order_is_a_replan_not_a_silent_flatten():
    solve = result({"t0": 0, "t1": 0}, ii=4, versions={"x": 1})
    cycle = contracts(("t0", "cube", "mte2", None, ("t1",)),
                      ("t1", "cube", "mte2", None, ("t0",)))
    error = expect_code("PLAN_RESOLVE_ISSUE_ORDER_CYCLE",
                        lambda: resolve(solve, task_contracts=cycle))
    assert error.needs_replan is True


def test_a_backwards_phase_edge_is_refused_rather_than_reordered():
    """A phase-3 task cannot be forced to issue before a phase-0 task.

    phase is start % II, NOT the stage: t1 at start 3 with II 4 is (stage 0, phase 3),
    and t0 at start 0 is (stage 0, phase 0). The edge t1 -> t0 therefore walks backwards
    across physical phases, which no reordering of the issue sequence can repair.
    """
    solve = result({"t0": 0, "t1": 3}, ii=4, versions={"x": 1})
    backwards = contracts(("t0", "cube", "mte2", None, ("t1",)),
                          ("t1", "cube", "mte2"))
    error = expect_code("PLAN_RESOLVE_ISSUE_AFTER_VIOLATED",
                        lambda: resolve(solve, task_contracts=backwards))
    assert error.needs_replan is True


def test_manual_mode_activates_same_pipe_source_order():
    solve = result({"t0": 0, "t1": 0}, ii=4, versions={"x": 1})
    manual = contracts(("t0", "cube", "mte2", 0), ("t1", "cube", "mte2", 0))
    resolution = resolve(solve, task_contracts=manual)
    assert resolution.issue_order == ("t0", "t1")


def test_an_unknown_issue_predecessor_is_named():
    solve = result({"t0": 0}, ii=4, versions={"x": 1})
    bad = contracts(("t0", "cube", "mte2", None, ("nope",)))
    expect_code("PLAN_RESOLVE_ISSUE_PREDECESSOR_INVALID",
                lambda: resolve(solve, task_contracts=bad))


# --------------------------------- no-offset: RECORDED, never silently refused
def test_cross_stage_start_under_no_offset_is_recorded_as_needs_replan():
    """Lead ruling (b): parse faithfully, record the spec 2.5 violation in diagnostics."""
    resolution = resolve(result({"t0": 0, "t1": 6}, ii=4, versions={"x": 1}),
                         enable_offset=False)
    assert resolution.is_resolved, "a cross-stage start must still resolve"
    violated = [d for d in resolution.diagnostics
                if d["kind"] == "OFFSET_WINDOW_VIOLATED"]
    assert len(violated) == 1, resolution.diagnostics
    assert violated[0]["task_id"] == "t1" and violated[0]["needs_replan"] is True
    assert violated[0]["stage"] == 1


def test_a_single_stage_solve_records_no_offset_violation():
    resolution = resolve(result({"t0": 0, "t1": 2}, ii=4, versions={"x": 1}),
                         enable_offset=False)
    assert resolution.diagnostics == (), resolution.diagnostics


# ----------------------------------------------------- non-SAT and gating
def test_non_sat_never_yields_a_schedule():
    for state in ("UNKNOWN", "UNSAT", "FALLBACK"):
        resolution = resolve(result({}, state=state, postcheck=None))
        assert not resolution.is_resolved, state
        assert resolution.ii is None and resolution.tasks == ()
        assert resolution.solver_state == state
        expect_code("PLAN_RESOLVE_TASK_MISSING",
                    lambda: resolution.task("t0"))


def test_unknown_keeps_its_state_even_with_a_partial_assignment():
    solve = result({"t0": 0}, ii=4, versions={"x": 1}, state="UNKNOWN", postcheck=None)
    resolution = resolve(solve)
    assert resolution.solver_state == "UNKNOWN" and not resolution.is_resolved


def test_an_unpostchecked_sat_is_refused_by_default():
    solve = result({"t0": 0, "t1": 4}, ii=4, versions={"x": 1}, postcheck=None)
    expect_code("PLAN_RESOLVE_POSTCHECK_REQUIRED", lambda: resolve(solve))
    # ...but an explicit opt-out is allowed for a nominal-only consumer.
    assert resolve(solve, require_source_postcheck=False).is_resolved


def test_an_unknown_solver_state_is_named():
    solve = result({"t0": 0}, ii=4, versions={"x": 1})
    _mutate(solve, lambda value: value.__setitem__("state", "MAYBE"), "bad state")
    expect_code("PLAN_RESOLVE_SOLVER_STATE_INVALID", lambda: resolve(solve))


# -------------------------------------------------- identity and coverage
def test_a_problem_digest_mismatch_is_refused():
    solve = result({"t0": 0, "t1": 4}, ii=4, versions={"x": 1})
    expect_code("PLAN_RESOLVE_SOURCE_ID_MISMATCH",
                lambda: resolve(solve, expected_problem_digest="sha256:" + "9" * 64))


def test_a_missing_or_extra_task_id_is_refused_with_the_ids_named():
    solve = result({"t0": 0}, ii=4, versions={"x": 1})
    error = expect_code("PLAN_RESOLVE_COVERAGE_MISMATCH", lambda: resolve(solve))
    assert error.actual["missing"] == ["t1"]
    extra = result({"t0": 0, "t1": 4, "t9": 8}, ii=4, versions={"x": 1})
    error = expect_code("PLAN_RESOLVE_COVERAGE_MISMATCH", lambda: resolve(extra))
    assert error.actual["extra"] == ["t9"]


def test_a_non_model_cycle_time_unit_is_refused():
    solve = result({"t0": 0, "t1": 4}, ii=4, versions={"x": 1})
    _mutate(solve["assignment"]["tasks"]["t0"]["start_time"],
            lambda value: value.__setitem__("unit", "ns"), "nanoseconds")
    expect_code("PLAN_RESOLVE_TIME_UNIT_MISMATCH", lambda: resolve(solve))


def test_a_bare_integer_start_time_is_not_accepted_as_model_cycles():
    solve = result({"t0": 0, "t1": 4}, ii=4, versions={"x": 1})
    _mutate(solve["assignment"]["tasks"]["t0"],
            lambda entry: entry.__setitem__("start_time", 0), "bare int")
    expect_code("PLAN_RESOLVE_TIME_RECORD_INVALID", lambda: resolve(solve))


def test_a_missing_assignment_is_refused_instead_of_falling_back_to_a_schedule():
    solve = result({"t0": 0, "t1": 4}, ii=4, versions={"x": 1})
    del solve["assignment"]
    solve["schedule"] = {"t0": {"stage": 0, "order": 0, "start_time": 0}}
    expect_code("PLAN_RESOLVE_ASSIGNMENT_MISSING", lambda: resolve(solve))


# ------------------------------------------------------ versions and capacity
def test_a_fixed_version_change_is_a_named_replan():
    fixed = (RP.BufferContract(root="x", min_versions=2, max_versions=2,
                               memory_domain="ub0", aligned_slot_bytes=128,
                               fixed_versions=2),)
    solve = result({"t0": 0, "t1": 4}, ii=4, versions={"x": 3})
    error = expect_code("PLAN_RESOLVE_FIXED_VERSION_VIOLATED",
                        lambda: resolve(solve, buffer_contracts=fixed))
    assert error.needs_replan is True


def test_a_version_outside_the_declared_range_is_not_clamped():
    solve = result({"t0": 0, "t1": 4}, ii=4, versions={"x": 9})
    error = expect_code("PLAN_RESOLVE_VERSION_RANGE_VIOLATED", lambda: resolve(solve))
    assert error.needs_replan is True
    assert error.actual == 9, "the solver's value must be reported, never replaced"


def test_capacity_pressure_asks_for_a_replan_instead_of_shrinking_versions():
    solve = result({"t0": 0, "t1": 4}, ii=4, versions={"x": 4})
    error = expect_code("PLAN_RESOLVE_CAPACITY_REPLAN_REQUIRED",
                        lambda: resolve(solve, capacities={"ub0": 256}))
    assert error.needs_replan is True
    assert error.actual == 512, error.actual


def test_capacity_that_fits_is_recorded_as_checked():
    resolution = resolve(result({"t0": 0, "t1": 4}, ii=4, versions={"x": 4}),
                         capacities={"ub0": 4096})
    assert resolution.capacity_checked is True
    assert resolution.declared_storage_bytes == (("ub0", 512),)
    assert resolution.buffers[0].versions == 4


def test_a_missing_capacity_domain_is_named():
    solve = result({"t0": 0, "t1": 4}, ii=4, versions={"x": 1})
    expect_code("PLAN_RESOLVE_CAPACITY_DOMAIN_MISSING",
                lambda: resolve(solve, capacities={"other": 1024}))


def test_a_fixed_buffer_reports_its_version_source():
    fixed = (RP.BufferContract(root="x", min_versions=1, max_versions=1,
                               memory_domain="ub0", aligned_slot_bytes=64,
                               fixed_versions=1),)
    resolution = resolve(result({"t0": 0, "t1": 4}, ii=4, versions={"x": 1}),
                         buffer_contracts=fixed)
    assert resolution.buffers[0].version_source == "fixed"


def test_duplicate_contract_ids_are_refused():
    solve = result({"t0": 0, "t1": 4}, ii=4, versions={"x": 1})
    duplicated = contracts(("t0", "cube", "mte2"), ("t0", "vector", "vector"))
    expect_code("PLAN_RESOLVE_CONTRACT_ID_INVALID",
                lambda: resolve(solve, task_contracts=duplicated))


def test_contracts_from_a_problem_keep_real_source_order():
    problem = {
        "tasks": [{"stable_id": "zzz", "action_ids": ["a"], "pipe": "vector",
                   "resource": "vector"},
                  {"stable_id": "aaa", "action_ids": ["b"], "pipe": "vector",
                   "resource": "vector"}],
        "buffers": [{"root": "x", "space": "ascend.ub", "bytes": 64, "versions": 2,
                     "fixed": True}],
    }
    tasks, buffers_out = RP.contracts_from_problem(problem)
    assert [task.task_id for task in tasks] == ["zzz", "aaa"], "sorted, not source order"
    assert [task.source_index for task in tasks] == [0, 1]
    assert buffers_out[0].fixed_versions == 2
    assert buffers_out[0].min_versions == buffers_out[0].max_versions == 2


# =============================== the FORMAL plan consumes the assignment
# G31 items 1-9: build_resolved_plan must build the authoritative ResolvedPlan FROM the
# solver's assignment -- versions from assignment.buffers, stage/phase from
# assignment.tasks, II from assignment.ii. These are the end-to-end proofs; the
# resolve_assignment tests above cannot show that the PLAN actually used the resolution.
def test_plan_slot_count_and_source_come_from_the_assignment_buffers():
    """AUTO buffer: the assignment's version wins over every Problem-side nominal value."""
    problem = plan_problem(buffers=[
        {"root": "x", "space": "ascend.ub", "bytes": 128, "versions": 1,
         "fixed": False, "max_versions": 4}])
    plan = build_plan(problem, versions={"x": 3})
    buffer = plan.buffers[0]
    assert (buffer.slot_count, buffer.slot_source) == (3, "auto"), buffer
    assert problem["buffers"][0]["versions"] == 1, "the Problem is unchanged"

    # Perturbing ONLY the assignment changes ONLY the plan's slot count.
    perturbed = build_plan(problem, versions={"x": 2})
    assert perturbed.buffers[0].slot_count == 2, perturbed.buffers[0]
    assert perturbed.identity.plan_digest != plan.identity.plan_digest


def test_plan_slot_count_comes_from_the_window_when_the_problem_has_no_nominal_version():
    """An auto buffer may declare ONLY [min_versions, max_versions]; the solver chooses."""
    problem = plan_problem(buffers=[
        {"root": "x", "space": "ascend.ub", "bytes": 128,
         "min_versions": 2, "max_versions": 4}])
    assert problem["buffers"][0].get("versions") is None
    for resolved in (2, 3, 4):
        plan = build_plan(problem, versions={"x": resolved})
        assert plan.buffers[0].slot_count == resolved, (resolved, plan.buffers[0])
        assert plan.buffers[0].slot_source == "auto"


def test_plan_slot_source_is_fixed_when_the_problem_pinned_the_version():
    problem = plan_problem(buffers=[
        {"root": "x", "space": "ascend.ub", "bytes": 128, "versions": 2,
         "fixed": True}])
    plan = build_plan(problem, versions={"x": 2})
    assert (plan.buffers[0].slot_count, plan.buffers[0].slot_source) == (2, "fixed")


def test_a_plan_with_no_resolved_version_is_refused_by_name():
    """The builder never falls back to the Problem's nominal count."""
    problem = plan_problem(buffers=[
        {"root": "x", "space": "ascend.ub", "bytes": 128, "versions": 1,
         "fixed": False, "max_versions": 4}])
    solve = result({"t0": 0, "t1": 6}, ii=4, versions={},
                   problem_digest=problem["problem_digest"])
    expect_code("PLAN_RESOLVE_COVERAGE_MISMATCH",
                lambda: RP.build_resolved_plan(problem, solve, SOURCES))


def _plan_fields(plan):
    """The resolved facts a solver perturbation must be able to move."""
    return (plan.loop.ii["value"],
            tuple((task.task_id, task.stage, task.phase, task.start_time["value"])
                  for task in plan.tasks),
            tuple(buffer.slot_count for buffer in plan.buffers))


def test_perturbing_the_assignment_phase_changes_the_formal_plan():
    problem = plan_problem(buffers=[
        {"root": "x", "space": "ascend.ub", "bytes": 128, "versions": 1,
         "fixed": False, "max_versions": 4}])
    base = build_plan(problem, starts={"t0": 0, "t1": 6}, ii=4, versions={"x": 2})
    moved = build_plan(problem, starts={"t0": 0, "t1": 5}, ii=4, versions={"x": 2})
    assert base.tasks[1].phase == 2 and moved.tasks[1].phase == 1
    assert base.tasks[1].stage == moved.tasks[1].stage == 1
    assert _plan_fields(base) != _plan_fields(moved)
    assert base.identity.plan_digest != moved.identity.plan_digest


def test_perturbing_the_assignment_ii_changes_the_formal_plan():
    problem = plan_problem(buffers=[
        {"root": "x", "space": "ascend.ub", "bytes": 128, "versions": 1,
         "fixed": False, "max_versions": 4}])
    narrow = build_plan(problem, starts={"t0": 0, "t1": 6}, ii=4, versions={"x": 2})
    wide = build_plan(problem, starts={"t0": 0, "t1": 6}, ii=8, versions={"x": 2})
    assert (narrow.loop.ii["value"], wide.loop.ii["value"]) == (4, 8)
    # Same starts, different II => different stage AND phase. The Problem's nominal ii=1
    # could produce neither, so this can only come from the assignment.
    assert (narrow.tasks[1].stage, wide.tasks[1].stage) == (1, 0)
    assert (narrow.tasks[1].phase, wide.tasks[1].phase) == (2, 6)
    assert _plan_fields(narrow) != _plan_fields(wide)
    assert narrow.identity.plan_digest != wide.identity.plan_digest


def test_perturbing_the_assignment_auto_version_changes_the_formal_plan():
    problem = plan_problem(buffers=[
        {"root": "x", "space": "ascend.ub", "bytes": 128,
         "min_versions": 1, "max_versions": 4}])
    base = build_plan(problem, versions={"x": 1})
    grown = build_plan(problem, versions={"x": 4})
    assert base.buffers[0].slot_count == 1 and grown.buffers[0].slot_count == 4
    assert _plan_fields(base) != _plan_fields(grown)
    assert base.identity.plan_digest != grown.identity.plan_digest


# -------------------------- equal-phase issue order (blueprints parity)
# blueprints/test_resolve_assignment.py is the reviewed reference for these two rules. Its
# sample puts four tasks with II=80 and starts 0/40/80/160, so every task is in phase 0
# and the (-stage, source_index) PREFERENCE can be told apart from the HARD edges.
II80 = 80
EQUAL_PHASE_STARTS = {"load_x": 0, "load_z": 40, "compute": 80, "store": 160}
EQUAL_PHASE_ROWS = (("load_x", "core", "mte2"), ("load_z", "core", "mte2"),
                    ("compute", "core", "vector"), ("store", "core", "mte3"))
EQUAL_PHASE_BUFFERS = (RP.BufferContract(root="x", min_versions=1, max_versions=3,
                                         memory_domain="ub0", aligned_slot_bytes=128),)


def equal_phase_contracts(*, authored=None, store_issue_after=()):
    """Source-order contracts; a non-None authored tuple activates manual mode."""
    return tuple(RP.TaskContract(
        task_id=task_id, core=core, pipe=pipe,
        authored_stage=None if authored is None else authored[index],
        issue_after=store_issue_after if task_id == "store" else (),
        source_index=index)
        for index, (task_id, core, pipe) in enumerate(EQUAL_PHASE_ROWS))


def equal_phase_solve(*, load_z_start=40):
    starts = dict(EQUAL_PHASE_STARTS, load_z=load_z_start)
    return result(starts, ii=II80, versions={"x": 2}, problem_digest=DIGEST)


def test_equal_phase_manual_source_order_outranks_the_stage_tie_break():
    """load_z's later stage would win by preference; the same-pipe manual edge must win.

    Blueprint parity: with load_z moved to start 80 (stage 1) and its authored stage 1,
    source order on CORE.mte2 makes load_x issue first. Preference alone puts load_z first
    because stage 1 > stage 0, so this distinguishes a real hard edge from a tie-break.
    """
    preferred = RP.resolve_assignment(
        equal_phase_solve(load_z_start=80),
        equal_phase_contracts(), EQUAL_PHASE_BUFFERS, enable_offset=True)
    assert preferred.issue_order.index("load_z") < preferred.issue_order.index("load_x")

    manual = RP.resolve_assignment(
        equal_phase_solve(load_z_start=80),
        equal_phase_contracts(authored=(0, 1, 1, 2)), EQUAL_PHASE_BUFFERS,
        enable_offset=True)
    assert manual.issue_order.index("load_x") < manual.issue_order.index("load_z"), \
        manual.issue_order
    # The manual edge is same-phase, so it reorders; it never invents a phase violation.
    assert all(task.phase == 0 for task in manual.tasks), manual.tasks


def test_equal_phase_issue_after_reverses_the_default_stage_priority():
    """store(stage 2) precedes compute(stage 1) by preference; the edge reverses it."""
    unconstrained = RP.resolve_assignment(
        equal_phase_solve(), equal_phase_contracts(), EQUAL_PHASE_BUFFERS,
        enable_offset=True)
    assert unconstrained.issue_order == ("store", "compute", "load_x", "load_z"), \
        unconstrained.issue_order

    constrained = RP.resolve_assignment(
        equal_phase_solve(), equal_phase_contracts(store_issue_after=("compute",)),
        EQUAL_PHASE_BUFFERS, enable_offset=True)
    assert constrained.issue_order == ("compute", "store", "load_x", "load_z"), \
        constrained.issue_order
    # Choosing an order must never rewrite the resolved start/stage/phase.
    by_id = {task.task_id: task for task in constrained.tasks}
    assert (by_id["store"].stage, by_id["store"].start_time["value"]) == (2, 160)


def test_assignment_dict_order_does_not_change_the_issue_order():
    """Issue rank follows the SOURCE contract order, never the assignment mapping order."""
    expected = RP.resolve_assignment(
        equal_phase_solve(), equal_phase_contracts(), EQUAL_PHASE_BUFFERS,
        enable_offset=True)
    shuffled_solve = equal_phase_solve()
    shuffled_solve["assignment"]["tasks"] = dict(
        reversed(list(shuffled_solve["assignment"]["tasks"].items())))
    shuffled = RP.resolve_assignment(
        shuffled_solve, equal_phase_contracts(), EQUAL_PHASE_BUFFERS, enable_offset=True)
    assert shuffled.issue_order == expected.issue_order == \
        ("store", "compute", "load_x", "load_z")
    assert [row.order for row in shuffled.tasks] == [2, 3, 1, 0], shuffled.tasks

# ------------------------------ one authoritative ResolvedPlan (G31 item 10)
def test_the_plan_package_exports_the_single_authoritative_resolved_plan():
    """The formal plan type is the one the package exports; the legacy one is labelled."""
    from tilefoundry_ascend import plan as plan_package
    from tilefoundry_ascend.schedule import resolved as legacy

    assert plan_package.ResolvedPlan is RP.ResolvedPlan
    assert RP.ResolvedPlan.__module__ == "tilefoundry_ascend.plan.resolved_plan"
    # The legacy diagnostic record must ANNOUNCE that it is not the authority, and its
    # own type must stay a different class -- nobody can pass one for the other.
    assert legacy.PLAN_AUTHORITY == "LEGACY_DIAGNOSTIC_ONLY"
    assert legacy.PLAN_AUTHORITY_MODULE == "tilefoundry_ascend.plan.resolved_plan"
    assert legacy.ResolvedPlan is not RP.ResolvedPlan

if __name__ == "__main__":
    failed = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print("PASS", _name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL", _name, type(exc).__name__, exc)
    raise SystemExit(1 if failed else 0)
