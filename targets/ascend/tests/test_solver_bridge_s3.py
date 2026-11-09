"""S3 solver-bridge tests: independent source postcheck + the sat_scope gate.

The postcheck under test re-derives every obligation from the Problem SOURCE fields
(tasks/buffers/dependencies/resource_deps/owner_exclusion_deps/pipe_order_deps/
stage_order_deps + the MODEL stage budget and capacity groups). These tests therefore
mutation-test the ASSIGNMENT and the PROBLEM, never the ConstraintTrace:

  * mutating an assignment must be caught by a specific, named code;
  * deleting the whole trace must NOT change a PASS (no self-certification);
  * tampering a Problem source edge must make a previously-passing assignment fail
    (the expected value really comes from the Problem).

Every rejection is a raised SolverError with a named code. There is no path that turns
a failed postcheck into a PASS, into a weaker state, or into a silent
"NOMINAL_MODEL_ONLY" write-off: sat_scope is a derived property that only a PASSing
report can raise.

CPU only; the real shared Z3 core is used through adapters.tf_loop_adapter (no device).
Run: PYTHONPATH=src:targets/ascend python targets/ascend/tests/test_solver_bridge_s3.py
"""
from __future__ import annotations

import copy
import dataclasses
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "targets" / "ascend"))

from tilefoundry_ascend.solver import bridge as B

try:  # z3 is required by the shared core; no-z3 interpreters must SKIP, not fail.
    import z3 as _z3
    _Z3_AVAILABLE = True
except ImportError:
    _z3 = None
    _Z3_AVAILABLE = False

SOLVER_TESTS_NEED_Z3 = "SOLVER_TESTS_NEED_Z3_INTERPRETER: run tools/run_solver_tests.sh with a z3 interpreter"
try:  # pytest: skip the whole module instead of reporting an AssertionError
    import pytest as _pytest
    pytestmark = _pytest.mark.skipif(not _Z3_AVAILABLE, reason=SOLVER_TESTS_NEED_Z3)
except ImportError:  # plain direct runner; handled in __main__ below
    _pytest = None


def expect_code(code, fn):
    """Assert a named refusal and return the error so callers can inspect fields."""
    try:
        fn()
    except Exception as error:  # noqa: BLE001
        got = getattr(error, "code", None)
        if got != code:
            raise AssertionError("expected %s, got %r: %s" % (code, got, error))
        return error
    raise AssertionError("expected %s, but nothing was refused" % code)


def expect_pass(fn):
    try:
        return fn()
    except Exception as error:  # noqa: BLE001
        raise AssertionError("expected POSTCHECK_PASS, got %r: %s"
                             % (getattr(error, "code", None), error))


# ------------------------------------------------------------------ source fixtures
DIGEST_PROBLEM = "sha256:" + "1" * 64
DIGEST_SOURCE = "sha256:" + "2" * 64
DIGEST_LEDGER = "sha256:" + "3" * 64


def task(stable_id, pipe, latency, issue_interval, stage=0):
    return {"stable_id": stable_id, "action_ids": ["a-" + stable_id], "pipe": pipe,
            "resource": pipe, "latency": latency, "issue_interval": issue_interval,
            "loop": "main", "stage": stage}


def buffer(root="x_ub", size=64, versions=1, fixed=True):
    return {"root": root, "space": "ascend.ub", "bytes": size,
            "versions": versions, "fixed": fixed}


def make_problem(tasks, buffers=None, dependencies=None, num_stages=2,
                 enable_offset=False, loop_id="main", **extra):
    problem = {
        "schema": "real-problem/1",
        "problem_digest": DIGEST_PROBLEM,
        "source_digest": DIGEST_SOURCE,
        "ledger_sha": DIGEST_LEDGER,
        "loop_id": loop_id,
        "requested_num_stages": num_stages,
        "enable_offset": enable_offset,
        "tasks": tasks,
        "buffers": buffers if buffers is not None else [buffer()],
        "dependencies": dependencies or [],
    }
    problem.update(extra)
    return problem


def make_model(capacity_bytes=1024, roots=("x_ub",)):
    return {"model_id": "ascend-pipe-model", "model_version": "v1",
            "memory_groups": [{"id": "ub0", "capacity_bytes": capacity_bytes,
                               "buffers": list(roots)}]}


#: The S0-S2 fixture. Its default-path result_digest is pinned below: S3 must not
#: drift it, because published single-run artifacts already record this digest.
S02_PROBLEM = make_problem(
    [task("t0", "mte2", 4, 1), task("t1", "vector", 8, 2), task("t2", "mte3", 4, 1)],
    dependencies=[
        {"source_id": "dep-raw-0", "producer": "t0", "consumer": "t1",
         "distance": 0, "kind": "RAW", "latency": 4},
        {"source_id": "dep-raw-1", "producer": "t1", "consumer": "t2",
         "distance": 0, "kind": "RAW", "latency": 8},
    ],
    resource_deps=[["t0", "t1"], ["t1", "t2"]])
S02_MODEL = make_model()
S02_OPTIONS = {"requested_num_stages": 2, "seed": 0}

#: Default-path digest of S02_PROBLEM measured on the pre-S3 bridge (HEAD 0d69b8f).
S02_RESULT_DIGEST_BEFORE_S3 = (
    "6583b1fc93a60fed825ba070fb0a7fe225e0c3aeb993ee5bd2b3b7569adff806")


def solve(problem, model=None, options=None):
    model = S02_MODEL if model is None else model
    options = {} if options is None else options
    result = B.solve_real_problem(problem, model, options)
    request = B.lower_problem_to_request(problem, model, options)
    return problem, request, result


def with_assignment(result, mutate):
    assignment = copy.deepcopy(result.assignment)
    mutate(assignment)
    return dataclasses.replace(result, assignment=assignment)


def set_start(assignment, tid, value):
    assignment["tasks"][tid]["start_time"]["value"] = value


def set_version(assignment, root, value):
    assignment["buffers"][root] = value


# ------------------------------------------------------------- positive control
def test_s3_real_core_sat_passes_independent_postcheck():
    problem, request, result = solve(S02_PROBLEM, S02_MODEL, S02_OPTIONS)
    assert result.state == B.SAT

    report = expect_pass(lambda: B.verify_solution(problem, request, result))
    assert report["schema"] == "solver-postcheck/1"
    assert report["verdict"] == B.POSTCHECK_VERDICT_PASS
    assert report["state"] == B.SAT
    assert report["sat_scope"] == B.SAT_SCOPE_POSTCHECKED
    # Capacity is a MODEL bound, not a measurement: a source postcheck cannot upgrade it.
    assert report["capacity_scope"] == "NOMINAL_MODEL_ONLY"
    assert report["source_authority"] == "problem-source-fields"
    assert report["trace_used"] is False
    assert report["problem_digest"] == problem["problem_digest"] == request.problem_digest
    assert report["request_digest"] == request.request_digest
    assert report["result_digest"] == result.result_digest
    assert report["core_source_hash"] == request.core_source_hash
    # Counts are re-derived from the source, so they must equal the source cardinality.
    checked = report["checked"]
    assert checked["tasks"] == 3 and checked["buffers"] == 1
    assert checked["starts"] == 3 and checked["ii"] == 1
    assert checked["data_deps"] == 2
    assert checked["resource_deps"] == 2
    assert checked["fixed_versions"] == 1
    assert checked["memory_groups"] == 1
    assert checked["request_source_consistency"] == 1
    # The report digest is recomputable from the report itself.
    assert report["postcheck_digest"] == B._canonical(
        {key: value for key, value in report.items() if key != "postcheck_digest"})


def test_s3_postcheck_and_scope_raises_scope_only_on_pass():
    problem, request, result = solve(S02_PROBLEM, S02_MODEL, S02_OPTIONS)
    report, sat_scope = B.postcheck_and_scope(problem, request, result)
    assert report["verdict"] == B.POSTCHECK_VERDICT_PASS
    assert sat_scope == B.SAT_SCOPE_POSTCHECKED

    # A default-path SAT result carries no postcheck and stays NOMINAL.
    assert result.postcheck is None
    assert result.sat_scope == B.SAT_SCOPE_NOMINAL
    # ...and its scope is derived, not a stored field that could be forged.
    forged = dataclasses.replace(result, postcheck={"verdict": "SOMETHING_ELSE"})
    assert forged.sat_scope == B.SAT_SCOPE_NOMINAL
    forged_ok = dataclasses.replace(result, state=B.FALLBACK,
                                    postcheck={"verdict": B.POSTCHECK_VERDICT_PASS})
    assert forged_ok.sat_scope == B.SAT_SCOPE_POSTCHECKED  # property is report-driven
    # But attaching a PASS to a non-SAT result is itself refused.
    expect_code(B.POSTCHECK_NOT_APPLICABLE,
                lambda: B._attach_postcheck(forged_ok, {"verdict": B.POSTCHECK_VERDICT_PASS}))


def test_s3_require_source_postcheck_attaches_report_and_upgrades_scope():
    result = B.solve_real_problem(S02_PROBLEM, S02_MODEL, S02_OPTIONS,
                                  require_source_postcheck=True)
    assert result.state == B.SAT
    assert result.postcheck is not None
    assert result.postcheck["verdict"] == B.POSTCHECK_VERDICT_PASS
    assert result.sat_scope == B.SAT_SCOPE_POSTCHECKED
    assert result.capacity_scope == "NOMINAL_MODEL_ONLY"
    # The assignment is not rewritten by postchecking.
    plain = B.solve_real_problem(S02_PROBLEM, S02_MODEL, S02_OPTIONS)
    assert result.assignment == plain.assignment
    assert result.state == plain.state == B.SAT


# ------------------------------------------------------------ assignment controls
def test_s3_data_dependency_violation_is_named():
    problem, request, result = solve(S02_PROBLEM, S02_MODEL, S02_OPTIONS)
    # t1 must start >= t0 + latency(4). Pulling t1 level with t0 breaks it.
    bad = with_assignment(result, lambda a: set_start(a, "t1", 0))
    error = expect_code(B.POSTCHECK_DEPENDENCY_VIOLATED,
                        lambda: B.verify_solution(problem, request, bad))
    assert error.source_id == "dep-raw-0"
    assert error.actual["gap"] < error.expected["required_gap"]


def test_s3_reuse_data_dependency_uses_buffer_version():
    """A negative distance is the core's buffer-version encoding, re-derived here."""
    problem = make_problem(
        [task("t0", "mte2", 4, 1), task("t1", "vector", 8, 2)],
        buffers=[buffer(versions=2, fixed=True)],
        dependencies=[
            {"source_id": "dep-raw-0", "producer": "t0", "consumer": "t1",
             "distance": 0, "kind": "RAW", "latency": 4},
            {"source_id": "dep-reuse-0", "producer": "t1", "consumer": "t0",
             "distance": -1, "kind": "reuse", "buffer": "x_ub", "latency": 0},
        ])
    problem, request, result = solve(problem)
    assert result.state == B.SAT
    report = expect_pass(lambda: B.verify_solution(problem, request, result))
    assert report["checked"]["data_deps"] == 2

    starts = result.assignment["tasks"]
    ii = result.assignment["ii"]["value"]
    versions = result.assignment["buffers"]["x_ub"]
    latency = 0  # the reuse edge's own latency, from the Problem source
    required = latency - ii * versions
    gap = (starts["t0"]["start_time"]["value"] - starts["t1"]["start_time"]["value"])
    assert gap >= required

    # Push t1 one cycle past the version-derived bound. The bound depends on the
    # returned version count, so a version-blind check would not catch this.
    def break_reuse(assignment):
        set_start(assignment, "t1",
                  assignment["tasks"]["t0"]["start_time"]["value"] - required + 1)
    bad = with_assignment(result, break_reuse)
    error = expect_code(B.POSTCHECK_DEPENDENCY_VIOLATED,
                        lambda: B.verify_solution(problem, request, bad))
    assert error.source_id == "dep-reuse-0"
    assert error.expected["versions"] == versions
    assert error.actual["gap"] < error.expected["required_gap"]


def test_s3_fixed_buffer_version_drift_is_named():
    problem, request, result = solve(S02_PROBLEM, S02_MODEL, S02_OPTIONS)
    assert result.assignment["buffers"] == {"x_ub": 1}
    # 2 is a legal value inside [1, num_stages=2] but it is NOT the Problem's fixed value.
    bad = with_assignment(result, lambda a: set_version(a, "x_ub", 2))
    error = expect_code(B.POSTCHECK_FIXED_DRIFT,
                        lambda: B.verify_solution(problem, request, bad))
    assert error.source_id == "x_ub"
    assert error.expected == 1 and error.actual == 2


def test_s3_fixed_version_drift_from_core_recalculation_is_caught():
    """The shared core recomputes versions from the schedule; the postcheck must not
    trust that label. A Problem fixed at 3 with no reuse edge comes back as 1."""
    problem = make_problem(
        [task("t0", "mte2", 4, 1), task("t1", "vector", 8, 2)],
        buffers=[buffer(versions=3, fixed=True)], num_stages=3)
    model = make_model()
    result = B.solve_real_problem(problem, model, {})
    assert result.state == B.SAT, result.reason
    assert result.assignment["buffers"] == {"x_ub": 1}  # core silently recalculated
    request = B.lower_problem_to_request(problem, model, {})
    expect_code(B.POSTCHECK_FIXED_DRIFT,
                lambda: B.verify_solution(problem, request, result))
    expect_code(B.POSTCHECK_FIXED_DRIFT,
                lambda: B.solve_real_problem(problem, model, {},
                                             require_source_postcheck=True))


def test_s3_ii_zero_or_below_source_lower_bound_is_named():
    problem, request, result = solve(S02_PROBLEM, S02_MODEL, S02_OPTIONS)
    expect_code(B.POSTCHECK_II_INVALID,
                lambda: B.verify_solution(
                    problem, request,
                    with_assignment(result, lambda a: a["ii"].__setitem__("value", 0))))
    # max(issue_interval) over the source is 2, so II=1 is below the lower bound.
    expect_code(B.POSTCHECK_II_INVALID,
                lambda: B.verify_solution(
                    problem, request,
                    with_assignment(result, lambda a: a["ii"].__setitem__("value", 1))))
    expect_code(B.POSTCHECK_II_INVALID,
                lambda: B.verify_solution(
                    problem, request,
                    with_assignment(result, lambda a: a["ii"].__setitem__("value", "3"))))


def test_s3_capacity_exceeded_is_named():
    problem = make_problem([task("t0", "mte2", 1, 1), task("t1", "vector", 1, 1)],
                           buffers=[buffer(size=600, versions=1, fixed=False)],
                           num_stages=2)
    problem, request, result = solve(problem)
    assert result.state == B.SAT
    report = expect_pass(lambda: B.verify_solution(problem, request, result))
    assert report["checked"]["memory_groups"] == 1
    # 2 versions * 600 B = 1200 B > 1024 B MODEL capacity; 2 is still inside [1, 2].
    bad = with_assignment(result, lambda a: set_version(a, "x_ub", 2))
    error = expect_code(B.POSTCHECK_CAPACITY_EXCEEDED,
                        lambda: B.verify_solution(problem, request, bad))
    assert error.expected["capacity_bytes"] == 1024
    assert error.actual["used_bytes"] == 1200


def test_s3_offset_window_violation_is_named():
    """enable_offset=False keeps every resource's tasks inside one II window."""
    problem = make_problem([task("t0", "vector", 1, 1), task("t1", "vector", 1, 1)])
    problem, request, result = solve(problem)
    assert result.state == B.SAT and result.assignment["ii"]["value"] == 1
    report = expect_pass(lambda: B.verify_solution(problem, request, result))
    assert report["checked"]["offset_groups"] == 1
    # Same pipe bit, span == II: the source forbids spreading a resource over stages.
    ii = result.assignment["ii"]["value"]
    bad = with_assignment(result, lambda a: set_start(a, "t1", ii))
    error = expect_code(B.POSTCHECK_OFFSET_VIOLATED,
                        lambda: B.verify_solution(problem, request, bad))
    assert error.source_id == "resource_bit:16"
    assert error.actual["span"] >= ii


def test_s3_offset_window_not_checked_when_offset_is_enabled():
    problem = make_problem([task("t0", "vector", 1, 1), task("t1", "vector", 1, 1)],
                           enable_offset=True)
    problem, request, result = solve(problem)
    assert result.state == B.SAT
    report = expect_pass(lambda: B.verify_solution(problem, request, result))
    assert report["checked"]["offset_groups"] == 0
    assert report["enable_offset"] is True


def test_s3_assignment_missing_task_or_buffer_is_named():
    problem, request, result = solve(S02_PROBLEM, S02_MODEL, S02_OPTIONS)
    missing_task = with_assignment(result, lambda a: a["tasks"].pop("t2"))
    error = expect_code(B.POSTCHECK_ASSIGNMENT_INCOMPLETE,
                        lambda: B.verify_solution(problem, request, missing_task))
    assert "t2" not in error.actual and "t2" in error.expected
    missing_buffer = with_assignment(result, lambda a: a["buffers"].pop("x_ub"))
    expect_code(B.POSTCHECK_ASSIGNMENT_INCOMPLETE,
                lambda: B.verify_solution(problem, request, missing_buffer))
    # An extra task is just as wrong as a missing one.
    extra_task = with_assignment(
        result, lambda a: a["tasks"].__setitem__(
            "intruder", {"start_time": {"value": 0, "unit": "MODEL cycles"}, "stage": None}))
    expect_code(B.POSTCHECK_ASSIGNMENT_INCOMPLETE,
                lambda: B.verify_solution(problem, request, extra_task))


def test_s3_out_of_range_values_are_named():
    problem, request, result = solve(S02_PROBLEM, S02_MODEL, S02_OPTIONS)
    expect_code(B.POSTCHECK_RANGE_INVALID,
                lambda: B.verify_solution(
                    problem, request,
                    with_assignment(result, lambda a: set_start(a, "t0", -1))))
    expect_code(B.POSTCHECK_RANGE_INVALID,
                lambda: B.verify_solution(
                    problem, request,
                    with_assignment(result, lambda a: set_version(a, "x_ub", 0))))
    expect_code(B.POSTCHECK_RANGE_INVALID,
                lambda: B.verify_solution(
                    problem, request,
                    with_assignment(result, lambda a: set_version(a, "x_ub", 3))))
    expect_code(B.POSTCHECK_ASSIGNMENT_INVALID,
                lambda: B.verify_solution(
                    problem, request,
                    with_assignment(result, lambda a: set_start(a, "t0", "0"))))


def test_s3_resource_conflict_is_named():
    problem = make_problem([task("t0", "mte2", 1, 1), task("t1", "vector", 1, 1)],
                           resource_deps=[["t0", "t1"]])
    problem, request, result = solve(problem)
    assert result.state == B.SAT
    expect_pass(lambda: B.verify_solution(problem, request, result))
    # Same modulo residue: the two tasks would overlap on the shared resource.
    bad = with_assignment(result, lambda a: set_start(a, "t1", 0))
    expect_code(B.POSTCHECK_RESOURCE_VIOLATED,
                lambda: B.verify_solution(problem, request, bad))


def test_s3_reversed_resource_pair_exposes_a_core_that_skipped_it():
    """Known core gap, pinned so it can never be silently certified.

    core.z3_schedule_loop_python only enforces a resource pair once, in `i < j` index
    order. A Problem declaring the pair in the other order therefore reaches the core
    as a constraint that is never asserted, and the core happily returns a schedule
    that overlaps the two tasks. The source postcheck treats the pair as UNORDERED (as
    the source semantics say) and refuses. That is the intended direction: the
    postcheck refuses rather than weakening itself to match the core's shortcut.
    """
    reversed_pair = make_problem([task("t0", "mte2", 1, 1), task("t1", "vector", 1, 1)],
                                 resource_deps=[["t1", "t0"]])
    problem, request, result = solve(reversed_pair)
    assert result.state == B.SAT, result.reason
    # The core ignored the reversed pair, so both tasks sit at the same residue.
    assert (result.assignment["tasks"]["t0"]["start_time"]["value"]
            == result.assignment["tasks"]["t1"]["start_time"]["value"])
    expect_code(B.POSTCHECK_RESOURCE_VIOLATED,
                lambda: B.verify_solution(problem, request, result))
    expect_code(B.POSTCHECK_RESOURCE_VIOLATED,
                lambda: B.solve_real_problem(reversed_pair, S02_MODEL, {},
                                             require_source_postcheck=True))



def test_s3_cyclic_nonoverlap_matches_a_set_intersection_model():
    """The resource test must mean "the modulo issue windows do not intersect".

    A tempting shortcut is to compare the wrap-around distance to the two issue
    intervals; that form is NOT equivalent (it accepts overlapping windows). Pin the
    exact model so a future simplification cannot quietly weaken the check.
    """
    def brute(producer_residue, producer_ii, consumer_residue, consumer_ii, ii):
        occupied = {(producer_residue + offset) % ii for offset in range(producer_ii)}
        other = {(consumer_residue + offset) % ii for offset in range(consumer_ii)}
        return not (occupied & other)

    for ii in range(2, 9):
        for producer_ii in range(1, ii + 1):
            for consumer_ii in range(1, ii + 1):
                for producer_residue in range(ii):
                    for consumer_residue in range(ii):
                        assert B._cyclic_nonoverlap(
                            producer_residue, producer_ii,
                            consumer_residue, consumer_ii, ii) == brute(
                                producer_residue, producer_ii,
                                consumer_residue, consumer_ii, ii), (
                            ii, producer_ii, consumer_ii,
                            producer_residue, consumer_residue)


def test_s3_owner_exclusion_uses_residues_not_absolute_starts():
    """Owners hand off one physical slot at their issue phases.

    Shifting BOTH starts by a whole II leaves every residue change-free, so the source
    relation still holds; comparing absolute starts would wrongly reject it.
    """
    problem = make_problem([task("t0", "mte2", 1, 1), task("t1", "vector", 1, 1)],
                           owner_exclusion_deps=[["t0", "t1", 1, 1]])
    problem, request, result = solve(problem)
    assert result.state == B.SAT
    expect_pass(lambda: B.verify_solution(problem, request, result))
    ii = result.assignment["ii"]["value"]

    def shift_one_iteration(assignment):
        for tid in ("t0", "t1"):
            entry = assignment["tasks"][tid]["start_time"]
            entry["value"] = entry["value"] + ii

    shifted = with_assignment(result, shift_one_iteration)
    expect_pass(lambda: B.verify_solution(problem, request, shifted))


def test_s3_owner_exclusion_is_named():
    problem = make_problem([task("t0", "mte2", 1, 1), task("t1", "vector", 1, 1)],
                           owner_exclusion_deps=[["t0", "t1", 1, 1]])
    problem, request, result = solve(problem)
    assert result.state == B.SAT
    report = expect_pass(lambda: B.verify_solution(problem, request, result))
    assert report["checked"]["owner_exclusion_deps"] == 1
    bad = with_assignment(result, lambda a: set_start(a, "t1", 0))
    expect_code(B.POSTCHECK_OWNER_VIOLATED,
                lambda: B.verify_solution(problem, request, bad))


def test_s3_pipe_order_is_named():
    problem = make_problem([task("t0", "mte2", 4, 2), task("t1", "mte2", 6, 2)],
                           pipe_order_deps=[["t0", "t1"]])
    problem, request, result = solve(problem)
    assert result.state == B.SAT
    report = expect_pass(lambda: B.verify_solution(problem, request, result))
    assert report["checked"]["pipe_order_deps"] == 1
    # r_u <= r_v must hold in modulo time. Put t0 later in the modulo window.
    def invert(assignment):
        set_start(assignment, "t0", 1)
        set_start(assignment, "t1", 0)
    expect_code(B.POSTCHECK_PIPE_ORDER_VIOLATED,
                lambda: B.verify_solution(problem, request,
                                          with_assignment(result, invert)))


def test_s3_stage_order_is_named():
    problem = make_problem([task("t0", "mte2", 1, 1), task("t1", "vector", 1, 1)],
                           stage_order_deps=[["t0", "t1"]])
    problem, request, result = solve(problem)
    assert result.state == B.SAT
    report = expect_pass(lambda: B.verify_solution(problem, request, result))
    assert report["checked"]["stage_order_deps"] == 1
    ii = result.assignment["ii"]["value"]
    # k = start // ii, so a full extra iteration puts t0 in a later stage than t1.
    bad = with_assignment(result, lambda a: set_start(a, "t0", ii))
    expect_code(B.POSTCHECK_STAGE_ORDER_VIOLATED,
                lambda: B.verify_solution(problem, request, bad))


# --------------------------------------------------------- state / provenance gate
def test_s3_non_sat_results_are_not_applicable():
    problem, request, _ = solve(S02_PROBLEM, S02_MODEL, S02_OPTIONS)
    fallback_raw = {"status": "fallback", "fallback_reason": "all II candidates unsat",
                    "start_times": [0, 4, 12], "buffer_versions": [1],
                    "minimal_II": 3, "check_results": []}
    fallback = B.classify_core_result(request, fallback_raw)
    assert fallback.state == B.FALLBACK
    error = expect_code(B.POSTCHECK_NOT_APPLICABLE,
                        lambda: B.verify_solution(problem, request, fallback))
    assert error.actual == B.FALLBACK
    expect_code(B.POSTCHECK_NOT_APPLICABLE,
                lambda: B.postcheck_and_scope(problem, request, fallback))

    unknown = B.classify_core_result(request, {"status": "unknown", "reason": "timeout"})
    assert unknown.state == B.UNKNOWN
    expect_code(B.POSTCHECK_NOT_APPLICABLE,
                lambda: B.verify_solution(problem, request, unknown))

    unsat = B.classify_core_result(request, {"status": "unsat",
                                             "search_scope_complete": True})
    assert unsat.state == B.UNSAT
    expect_code(B.POSTCHECK_NOT_APPLICABLE,
                lambda: B.verify_solution(problem, request, unsat))

    # A really-fallback Problem cannot satisfy require_source_postcheck either.
    cyclic = make_problem(
        [task("t0", "mte2", 1, 1), task("t1", "vector", 1, 1)],
        dependencies=[
            {"source_id": "c0", "producer": "t0", "consumer": "t1",
             "distance": 0, "kind": "RAW", "latency": 5},
            {"source_id": "c1", "producer": "t1", "consumer": "t0",
             "distance": 0, "kind": "RAW", "latency": 5},
        ])
    model = make_model()
    real_fallback = B.solve_real_problem(cyclic, model, {})
    assert real_fallback.state == B.FALLBACK, real_fallback.reason
    expect_code(B.POSTCHECK_NOT_APPLICABLE,
                lambda: B.solve_real_problem(cyclic, model, {},
                                             require_source_postcheck=True))


def test_s3_problem_digest_mismatch_is_named():
    problem, request, result = solve(S02_PROBLEM, S02_MODEL, S02_OPTIONS)
    other = dict(problem, problem_digest="sha256:" + "9" * 64)
    expect_code(B.POSTCHECK_SOURCE_MISMATCH,
                lambda: B.verify_solution(other, request, result))
    # A result that claims another Problem is not attributable either.
    other_result = dataclasses.replace(result, problem_digest="sha256:" + "9" * 64)
    expect_code(B.POSTCHECK_SOURCE_MISMATCH,
                lambda: B.verify_solution(problem, request, other_result))
    # No request at all is a provenance failure, never a vacuous PASS.
    expect_code(B.POSTCHECK_SOURCE_MISMATCH,
                lambda: B.verify_solution(problem, None, result))


def test_s3_request_that_does_not_match_source_is_named():
    """A request whose core arguments drifted from the Problem cannot certify it."""
    problem, request, result = solve(S02_PROBLEM, S02_MODEL, S02_OPTIONS)
    args = dict(request.core_args)
    args["latencies"] = [4, 99, 4]
    drifted = dataclasses.replace(request, core_args=args)
    expect_code(B.POSTCHECK_SOURCE_MISMATCH,
                lambda: B.verify_solution(problem, drifted, result))

    # Dropping the only buffer out of every memory group would hide its bytes from the
    # capacity sum; group membership must cover the whole source buffer set.
    args = dict(request.core_args)
    args["memory_groups"] = [[1024]]
    uncovered = dataclasses.replace(request, core_args=args)
    expect_code(B.POSTCHECK_SOURCE_MISMATCH,
                lambda: B.verify_solution(problem, uncovered, result))

    args = dict(request.core_args)
    args["buffer_sizes"] = [32]
    shrunk = dataclasses.replace(request, core_args=args)
    expect_code(B.POSTCHECK_SOURCE_MISMATCH,
                lambda: B.verify_solution(problem, shrunk, result))
    # Capacity ITSELF is a MODEL input that reaches the postcheck only through the
    # request; it is not re-derivable from the Problem, so it is deliberately not
    # asserted here as tamper-detectable. It IS enforced: see the CAPACITY_EXCEEDED test.



# ------------------------------------------------------- anti-self-certification
def test_s3_deleting_the_constraint_trace_does_not_change_the_verdict():
    """The trace records what the bridge ASKED the core. If the postcheck used it as
    'expected', this test would pass vacuously. Expected comes fromthe Problem."""
    problem, request, result = solve(S02_PROBLEM, S02_MODEL, S02_OPTIONS)
    empty_trace = dataclasses.replace(request, constraint_trace=())
    report = expect_pass(lambda: B.verify_solution(problem, empty_trace, result))
    assert report["verdict"] == B.POSTCHECK_VERDICT_PASS
    assert report["trace_used"] is False

    # Same verdict with a wholly forged trace: a trace cannot rescue a bad assignment.
    forged_entries = tuple(
        dataclasses.replace(entry, core_field="FORGED", core_index=-1, status="CONSUMED")
        for entry in request.constraint_trace)
    forged_trace = dataclasses.replace(request, constraint_trace=forged_entries)
    expect_pass(lambda: B.verify_solution(problem, forged_trace, result))
    bad = with_assignment(result, lambda a: set_version(a, "x_ub", 2))
    expect_code(B.POSTCHECK_FIXED_DRIFT,
                lambda: B.verify_solution(problem, forged_trace, bad))


def test_s3_tampering_a_problem_source_edge_rejects_a_previously_passing_assignment():
    """If the expected values came from the trace, this mutation would be invisible."""
    problem, request, result = solve(S02_PROBLEM, S02_MODEL, S02_OPTIONS)
    expect_pass(lambda: B.verify_solution(problem, request, result))

    # Raise dep-raw-0's latency past the real gap (4). The trace still records the old
    # relation; only the Problem source changes. A trace-driven check would PASS here.
    deps = [dict(dep) for dep in problem["dependencies"]]
    deps[0]["latency"] = 100
    tampered = dict(problem, dependencies=deps)
    error = expect_code(B.POSTCHECK_DEPENDENCY_VIOLATED,
                        lambda: B.verify_solution(tampered, request, result))
    assert error.source_id == "dep-raw-0"
    assert error.expected["required_gap"] == 100

    # Tightening a resource/reserved pair the assignment cannot satisfy is caught too:
    # add a resource pair the real schedule runs concurrently.
    concurrent = make_problem([task("t0", "mte2", 1, 1), task("t1", "vector", 1, 1)])
    concurrent, concurrent_request, concurrent_result = solve(concurrent)
    assert concurrent_result.assignment["tasks"]["t0"]["start_time"]["value"] == \
        concurrent_result.assignment["tasks"]["t1"]["start_time"]["value"]
    tampered_pairs = dict(concurrent, resource_deps=[["t0", "t1"]])
    expect_code(B.POSTCHECK_RESOURCE_VIOLATED,
                lambda: B.verify_solution(tampered_pairs, concurrent_request,
                                          concurrent_result))


# ------------------------------------------------------------- no-downgrade gate
def test_s3_a_failed_postcheck_never_produces_a_scope_or_a_pass():
    problem, request, result = solve(S02_PROBLEM, S02_MODEL, S02_OPTIONS)
    bad = with_assignment(result, lambda a: set_version(a, "x_ub", 2))

    # 1. verify_solution refuses; it does not return a dict with ok=False.
    error = expect_code(B.POSTCHECK_FIXED_DRIFT,
                        lambda: B.verify_solution(problem, request, bad))
    assert error.as_dict()["code"] == B.POSTCHECK_FIXED_DRIFT

    # 2. postcheck_and_scope refuses instead of returning NOMINAL_MODEL_ONLY.
    try:
        scope = B.postcheck_and_scope(problem, request, bad)
    except Exception:  # noqa: BLE001 - only the refusal kind matters here
        pass
    else:  # pragma: no cover - guard against a silent downgrade regression
        raise AssertionError("postcheck_and_scope returned %r for a failed postcheck"
                             % (scope,))

    # 3. The failed SAT result is NOT downgraded: state stays SAT, scope stays NOMINAL,
    #    and its digest is untouched (nothing was rewritten behind the caller's back).
    assert bad.state == B.SAT
    assert bad.sat_scope == B.SAT_SCOPE_NOMINAL
    assert bad.postcheck is None
    assert bad.result_digest == result.result_digest

    # 4. The gated entry point refuses, and no result object escapes with the raised scope.
    expect_code(B.POSTCHECK_FIXED_DRIFT,
                lambda: B.solve_real_problem(
                    make_problem([task("t0", "mte2", 4, 1), task("t1", "vector", 8, 2)],
                                 buffers=[buffer(versions=3, fixed=True)], num_stages=3),
                    make_model(), {}, require_source_postcheck=True))

    # 5. NOMINAL is a weaker scope than POSTCHECKED, never the other way around: a PASS
    #    report is the ONLY producer of the stronger scope, and it is derived.
    assert B.SAT_SCOPE_NOMINAL == "NOMINAL_MODEL_ONLY"
    assert B.SAT_SCOPE_POSTCHECKED == "SAT_SOURCE_POSTCHECKED"


# ------------------------------------------------- digest / compatibility pinning
def test_s3_default_path_result_digest_is_bit_identical_to_pre_s3():
    """Published single-run artifacts record this digest; S3 must not drift it."""
    result = B.solve_real_problem(S02_PROBLEM, S02_MODEL, S02_OPTIONS)
    assert result.result_digest == S02_RESULT_DIGEST_BEFORE_S3
    # The pre-S3 digest was sha256 over as_dict() minus result_digest; that formula is
    # unchanged AND sat_scope is not a stored field, which is why it cannot drift.
    assert result.result_digest == B._canonical(
        {key: value for key, value in result.as_dict().items() if key != "result_digest"})
    assert "sat_scope" not in result.as_dict()
    assert result.postcheck is None
    assert result.capacity_scope == "NOMINAL_MODEL_ONLY"


def test_s3_postchecked_result_digest_accounts_for_the_report():
    """Attaching the report DOES change the digest, because postcheck is a stored field.
    That is correct: a postchecked result is a different artifact than a nominal one."""
    plain = B.solve_real_problem(S02_PROBLEM, S02_MODEL, S02_OPTIONS)
    checked = B.solve_real_problem(S02_PROBLEM, S02_MODEL, S02_OPTIONS,
                                   require_source_postcheck=True)
    assert checked.result_digest != plain.result_digest
    assert checked.result_digest == B._canonical(
        {key: value for key, value in checked.as_dict().items() if key != "result_digest"})
    assert checked.as_dict()["postcheck"]["verdict"] == B.POSTCHECK_VERDICT_PASS


def test_s3_bridge_still_contains_no_second_solver():
    source = (REPO / "targets" / "ascend" / "tilefoundry_ascend" / "solver" /
              "bridge.py").read_text()
    for token in ("import z3", "z3_schedule_loop_python(", "z3.Solver(", "solver.check()"):
        assert token not in source, token
    assert "verify_solution" in source and "postcheck_and_scope" in source
    # The postcheck must never READ the trace as its expected values. Only the docstring
    # is allowed to spell the word "constraint_trace"; no executable line may touch it.
    code_lines = [line for line in source.splitlines()
                  if line.strip() and not line.strip().startswith("#")]
    in_docstring = False
    executable = []
    for line in code_lines:
        if line.count('"""') % 2 == 1:
            in_docstring = not in_docstring
            continue
        if not in_docstring:
            executable.append(line)
    postcheck_region = "\n".join(executable).split("def _derive_source_expectation", 1)[1]
    postcheck_region = postcheck_region.split("def postcheck_and_scope", 1)[0]
    assert "constraint_trace" not in postcheck_region, "postcheck reads the trace"
    assert "request.constraint_trace" not in executable


if __name__ == "__main__":
    if not _Z3_AVAILABLE:
        print("SKIP", SOLVER_TESTS_NEED_Z3)
        raise SystemExit(0)
    print("z3", _z3.get_version_string())
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
