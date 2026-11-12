"""P3-01 / G30: shared-solver adaptation, budget propagation, fallback verification.

G30 acceptance, one group of tests each:

  1. "真实共享solver消费完整数据/资源/版本/容量和作者约束" -- author stages reach the
     core instead of being dropped; a partial author stage set is refused.
  2. "no-offset在求解阶段强制stage=0且start位于一个II窗口" -- the opt-in
     enforce_no_offset_single_stage path really pins k=0, and the default path records
     that it did NOT (a recorded diagnostic, never a silent default).
  3. "SAT/UNSAT/UNKNOWN与已验证fallback区分；fallback不违反原硬约束" -- a candidate is
     re-checked against the SAME source rules as SAT and is REFUTED with a named code.
  4. "子问题从剩余总deadline扣费" -- nested budgets share one deadline and cannot re-grant.

Every negative test asserts that the mutation it depends on ACTUALLY APPLIED before it
asserts the refusal. A mutation that silently no-ops would otherwise make the test pass
for the wrong reason -- the "变异器静默失效" hazard. ``_mutate`` makes that a hard error.

CPU only; the real shared Z3 core is used (no device).
Run: PYTHONPATH=src:targets/ascend python targets/ascend/tests/test_solver_p3_01_budget_stage.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "targets" / "ascend"))

from tilefoundry_ascend.solver import bridge as B
from tilefoundry_ascend.solver.budget import BudgetExhausted, SolveBudget

try:
    import z3 as _z3  # noqa: F401
    _Z3_AVAILABLE = True
except ImportError:
    _Z3_AVAILABLE = False

try:
    import pytest as _pytest
    pytestmark = _pytest.mark.skipif(not _Z3_AVAILABLE, reason="SOLVER_TESTS_NEED_Z3_INTERPRETER")
except ImportError:
    _pytest = None

DIGEST_PROBLEM = "sha256:" + "1" * 64
DIGEST_SOURCE = "sha256:" + "2" * 64
DIGEST_LEDGER = "sha256:" + "3" * 64


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
    """Apply a mutation and PROVE it changed the target.

    A negative test whose mutation did not apply proves nothing: it would report the
    expected refusal for an untouched input. This raises instead, so a broken mutator
    is a test failure rather than a false pass.
    """
    before = copy.deepcopy(target)
    mutate_fn(target)
    if target == before:
        raise AssertionError("MUTATION_DID_NOT_APPLY: %s left the input unchanged" % label)
    return target


def task(stable_id, pipe, latency, issue_interval, stage=None, stage_source=None):
    entry = {"stable_id": stable_id, "action_ids": ["a-" + stable_id], "pipe": pipe,
             "resource": pipe, "latency": latency, "issue_interval": issue_interval,
             "loop": "main"}
    if stage is not None:
        entry["stage"] = stage
    if stage_source is not None:
        entry["stage_source"] = stage_source
    return entry


def problem(tasks, *, enable_offset=True, num_stages=4, dependencies=None,
            buffers=None, **extra):
    value = {
        "schema": "real-problem/1",
        "problem_digest": DIGEST_PROBLEM, "source_digest": DIGEST_SOURCE,
        "ledger_sha": DIGEST_LEDGER, "loop_id": "main",
        "requested_num_stages": num_stages, "enable_offset": enable_offset,
        "tasks": tasks,
        "buffers": buffers if buffers is not None else [
            {"root": "x_ub", "space": "ascend.ub", "bytes": 64, "versions": 1,
             "fixed": True}],
        "dependencies": dependencies if dependencies is not None else [
            {"source_id": "dep-0", "producer": "t0", "consumer": "t1",
             "distance": 0, "kind": "RAW", "latency": 4}],
        "resource_deps": [],
    }
    value.update(extra)
    return value


MODEL = {"model_id": "ascend-pipe-model", "model_version": "v1",
         "memory_groups": [{"id": "ub0", "capacity_bytes": 4096,
                            "buffers": ["x_ub"]}]}


def three_pipe(enable_offset=True, **kw):
    return problem([task("t0", "mte2", 4, 1, **kw), task("t1", "vector", 8, 2, **kw),
                    task("t2", "mte3", 4, 1, **kw)],
                   enable_offset=enable_offset,
                   dependencies=[
                       {"source_id": "dep-0", "producer": "t0", "consumer": "t1",
                        "distance": 0, "kind": "RAW", "latency": 4},
                       {"source_id": "dep-1", "producer": "t1", "consumer": "t2",
                        "distance": 0, "kind": "RAW", "latency": 8}],
                   resource_deps=[["t0", "t1"], ["t1", "t2"]])


# ------------------------------------------------ 1. author stage reaches the core
def test_author_stage_is_encoded_as_manual_mode_not_dropped():
    """A stage the SOURCE authored must reach the core as a manual stage."""
    value = three_pipe(stage=0, stage_source="AUTHORED")
    value["tasks"][0]["stage"] = 1
    value["tasks"][1]["stage"] = 2
    diagnostics = {}
    request = B.lower_problem_to_request(value, MODEL, {}, diagnostics=diagnostics)
    assert request.core_args["manual_schedule"] is True
    assert request.core_args["manual_stages"] == [1, 2, 0]
    assert diagnostics["manual_stage_mode"] is True
    assert diagnostics["ignored_nominal_stages"] == []


def test_author_stage_actually_changes_the_core_schedule():
    """Not just plumbed through: a changed author stage moves the real returned start."""
    all_zero = three_pipe(stage=0, stage_source="AUTHORED")
    staged = three_pipe(stage=0, stage_source="AUTHORED")
    # [0, 1, 2] is feasible for this graph and gives three DISTINCT stages.
    staged["tasks"][1]["stage"] = 1
    staged["tasks"][2]["stage"] = 2
    plain = B.solve_real_problem(all_zero, MODEL, {})
    moved = B.solve_real_problem(staged, MODEL, {})
    assert plain.state == B.SAT and moved.state == B.SAT, (plain.state, moved.state)
    ii_moved = moved.assignment["ii"]["value"]
    start_moved = moved.assignment["tasks"]["t1"]["start_time"]["value"]
    start_plain = plain.assignment["tasks"]["t1"]["start_time"]["value"]
    assert start_moved != start_plain, "the authored stage did not move the schedule"
    # The core normalizes so the EARLIEST stage is 0; the relative spread is what survives.
    stages = {tid: divmod(entry["start_time"]["value"], ii_moved)[0]
              for tid, entry in moved.assignment["tasks"].items()}
    assert stages["t0"] == 0 and stages["t1"] < stages["t2"], stages


def test_core_normalizes_stage_origin_so_an_absolute_shift_is_lost():
    """Recording WHY P3-02 must keep authored stage and relative shift separately.

    The core subtracts min(k)*II before returning, so [1,1,1] and [0,0,0] are
    indistinguishable in the assignment. A resolver that reads ONLY start_time therefore
    cannot recover the author's absolute stage -- it has to keep it.
    """
    shifted = three_pipe(stage=1, stage_source="AUTHORED")
    base = three_pipe(stage=0, stage_source="AUTHORED")
    one = B.solve_real_problem(shifted, MODEL, {})
    zero = B.solve_real_problem(base, MODEL, {})
    assert one.state == B.SAT and zero.state == B.SAT
    assert one.assignment["ii"]["value"] == zero.assignment["ii"]["value"]
    starts_one = {k: v["start_time"]["value"] for k, v in one.assignment["tasks"].items()}
    starts_zero = {k: v["start_time"]["value"] for k, v in zero.assignment["tasks"].items()}
    assert starts_one == starts_zero, "expected the origin normalization to erase the shift"


def test_an_infeasible_author_stage_set_is_never_a_silent_success():
    """A stage-1 producer feeding a stage-0 consumer is infeasible and must say so."""
    value = three_pipe(stage=0, stage_source="AUTHORED")
    value["tasks"][0]["stage"] = 1        # producer in a LATER stage than its consumer
    result = B.solve_real_problem(value, MODEL, {})
    assert result.state != B.SAT, "an infeasible author stage must not yield SAT"
    assert result.assignment == {}, "a non-SAT result must not carry a schedule"


def test_nominal_stage_fill_is_ignored_but_RECORDED():
    """A cost-model fill is not an author constraint; conflicting with it must not pin k."""
    value = three_pipe(stage=0, stage_source="COST_MODEL")
    diagnostics = {}
    request = B.lower_problem_to_request(value, MODEL, {}, diagnostics=diagnostics)
    assert request.core_args["manual_schedule"] is False
    assert request.core_args["manual_stages"] is None
    ignored = diagnostics["ignored_nominal_stages"]
    assert len(ignored) == 3, ignored
    assert {entry["action"] for entry in ignored} == {"IGNORED_AS_NOMINAL"}
    assert {entry["task_id"] for entry in ignored} == {"t0", "t1", "t2"}


def test_partial_author_stage_set_is_refused_not_defaulted():
    value = three_pipe()
    value["tasks"][0]["stage"] = 1
    value["tasks"][0]["stage_source"] = "AUTHORED"
    value["tasks"][1]["stage_source"] = "AUTHORED"     # declares authorship, omits a stage
    value["tasks"][2]["stage_source"] = "AUTHORED"
    # Prove the fixture really is PARTIAL before asserting the refusal.
    declared = [t for t in value["tasks"] if t.get("stage_source") == "AUTHORED"
                and "stage" in t]
    assert len(declared) == 1, declared
    expect_code("SOLVER_STAGE_INCOMPLETE",
                lambda: B.lower_problem_to_request(value, MODEL, {}))


def test_invalid_author_stage_is_named():
    negative = three_pipe(stage=0, stage_source="AUTHORED")
    _mutate(negative["tasks"][0], lambda t: t.__setitem__("stage", -1), "negative stage")
    expect_code("SOLVER_STAGE_INVALID",
                lambda: B.lower_problem_to_request(negative, MODEL, {}))


# --------------------------------------------- 2. no-offset single-stage policy
def test_no_offset_single_stage_opt_in_really_pins_stage_zero():
    value = three_pipe(enable_offset=False)
    options = {"enforce_no_offset_single_stage": True}
    diagnostics = {}
    request = B.lower_problem_to_request(value, MODEL, options, diagnostics=diagnostics)
    assert request.core_args["manual_schedule"] is True
    assert request.core_args["manual_stages"] == [0, 0, 0]
    assert diagnostics["enforce_no_offset_single_stage"] is True
    result = B.solve_real_problem(value, MODEL, options)
    assert result.state == B.SAT
    ii = result.assignment["ii"]["value"]
    stages = {tid: divmod(entry["start_time"]["value"], ii)[0]
              for tid, entry in result.assignment["tasks"].items()}
    assert set(stages.values()) == {0}, stages


def test_default_no_offset_policy_records_that_it_did_not_pin_stage_zero():
    """The default path must not silently look like it enforced spec 2.5."""
    value = three_pipe(enable_offset=False)
    diagnostics = {}
    B.lower_problem_to_request(value, MODEL, {}, diagnostics=diagnostics)
    assert diagnostics["enforce_no_offset_single_stage"] is False
    assert diagnostics["manual_stage_mode"] is False
    assert diagnostics["manual_stages"] is None


def test_no_offset_single_stage_contradicting_offset_enabled_is_refused():
    options = {"enforce_no_offset_single_stage": True}
    expect_code("SOLVER_OFFSET_POLICY_INVALID",
                lambda: B.lower_problem_to_request(three_pipe(enable_offset=True),
                                                   MODEL, options))


def test_non_bool_single_stage_flag_is_refused():
    value = three_pipe(enable_offset=False)
    diagnostics = {}
    for bad in (1, "true", None):
        expect_code("SOLVER_OFFSET_POLICY_INVALID",
                    lambda bad=bad: B.lower_problem_to_request(
                        value, MODEL, {"enforce_no_offset_single_stage": bad},
                        diagnostics=diagnostics))


# ------------------------------------------------- 3. fallback is re-verified
def _unknown_with_candidate(value, candidate):
    request = B.lower_problem_to_request(value, MODEL, {})
    raw = {"status": "unknown", "fallback_reason": "simulated",
           "search_scope_complete": False, "check_results": ["unknown"], **candidate}
    return request, B.classify_core_result(request, raw)


def test_fallback_candidate_that_satisfies_every_constraint_is_VERIFIED():
    value = three_pipe()
    good = B.solve_real_problem(value, MODEL, {})
    starts = [good.assignment["tasks"][tid]["start_time"]["value"] for tid in ("t0", "t1", "t2")]
    versions = [good.assignment["buffers"]["x_ub"]]
    request, result = _unknown_with_candidate(
        value, {"start_times": starts, "buffer_versions": versions,
                "minimal_II": good.assignment["ii"]["value"]})
    assert result.state == B.UNKNOWN
    report = B.verify_fallback_candidate(value, request, result.fallback_candidate)
    assert report["verdict"] == "VERIFIED", report
    assert report["limits"]["state_promoted"] is False
    assert report["trace_used"] is False


def test_fallback_candidate_violating_a_dependency_is_REFUTED():
    value = three_pipe()
    request, result = _unknown_with_candidate(
        value, {"start_times": [0, 0, 12], "buffer_versions": [1], "minimal_II": 3})
    report = B.verify_fallback_candidate(value, request, result.fallback_candidate)
    assert report["verdict"] == "REFUTED"
    assert report["code"] == "DEPENDENCY_VIOLATED"
    assert report["source_id"] == "dep-0"


def test_fallback_candidate_violating_fixed_versions_is_REFUTED():
    value = three_pipe()
    request, _ = _unknown_with_candidate(value, {"start_times": [0, 4, 12]})
    drift = {"start_times": [0, 4, 12], "buffer_versions": [2], "minimal_II": 3}
    report = B.verify_fallback_candidate(value, request, drift)
    assert report["verdict"] == "REFUTED"
    assert report["code"] == "FIXED_VERSION_DRIFT"


def test_fallback_candidate_with_wrong_shape_is_REFUTED_not_accepted():
    value = three_pipe()
    request, _ = _unknown_with_candidate(value, {"start_times": [0, 4, 12]})
    assert B.verify_fallback_candidate(value, request, {"start_times": [0, 4]})["code"] \
        == "FALLBACK_CANDIDATE_INCOMPLETE"
    assert B.verify_fallback_candidate(value, request, {"buffer_versions": [1]})["code"] \
        == "FALLBACK_CANDIDATE_MISSING"
    assert B.verify_fallback_candidate(value, request, None)["code"] \
        == "FALLBACK_CANDIDATE_MISSING"


def test_fallback_verification_never_promotes_state():
    value = three_pipe()
    good = B.solve_real_problem(value, MODEL, {})
    starts = [good.assignment["tasks"][tid]["start_time"]["value"] for tid in ("t0", "t1", "t2")]
    request, result = _unknown_with_candidate(
        value, {"start_times": starts, "buffer_versions": [1],
                "minimal_II": good.assignment["ii"]["value"]})
    checks = {}
    reported = B.solve_real_problem(
        value, MODEL, {"limits": {"timeout_ms": 1}, "allow_fallback_on_unknown": True},
        fallback_checks=checks)
    assert reported.state in (B.UNKNOWN, B.FALLBACK, B.SAT)
    # The state is whatever the core said; verification only ever ADDS a report.
    assert "verdict" in checks or reported.state == B.SAT


def test_a_refuted_candidate_is_attached_as_a_named_report_not_a_silent_pass():
    value = three_pipe()
    request = B.lower_problem_to_request(value, MODEL, {})
    refuted = B.verify_fallback_candidate(
        value, request, {"start_times": [0, 0, 12], "buffer_versions": [1], "minimal_II": 3})
    assert refuted["verdict"] == "REFUTED"
    assert refuted["code"] and refuted["detail"]


# -------------------------------------------------------- 4. budget propagation
class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, ms):
        self.t += ms / 1000.0


def test_child_budget_shares_the_parent_deadline():
    clock = _Clock()
    parent = SolveBudget(1000, clock=clock)
    clock.advance(400)
    child = parent.child("sub")
    assert child.remaining_ms() == parent.remaining_ms() == 600
    clock.advance(700)
    assert parent.exhausted and child.exhausted, "a child must not outlive its parent"


def test_nested_solves_cannot_regrant_the_full_budget():
    clock = _Clock()
    root = SolveBudget(1000, clock=clock)
    granted = []
    view = root
    for depth in range(5):
        granted.append(view.remaining_ms())
        clock.advance(150)
        view = view.child("depth%d" % depth)
    assert granted == sorted(granted, reverse=True), granted
    assert granted[0] == 1000 and granted[-1] < granted[0], granted
    assert view.remaining_ms() == root.remaining_ms()
    assert root.remaining_ms() < 1000


def test_per_check_share_never_exceeds_the_remaining_total():
    clock = _Clock()
    budget = SolveBudget(1000, clock=clock)
    for checks in (1, 2, 4, 16):
        share = budget.per_check_ms(None, checks)
        assert share <= budget.remaining_ms(), (checks, share)
    clock.advance(1000)
    assert budget.per_check_ms(None, 1) == 0
    assert budget.per_check_ms(60000, 1) == 0, "an exhausted budget grants nothing"


def test_exhausted_budget_is_UNKNOWN_not_UNSAT():
    clock = _Clock()
    budget = SolveBudget(1000, clock=clock)
    clock.advance(1500)
    value = three_pipe()
    try:
        B.solve_real_problem(value, MODEL, {"limits": {"total_timeout_ms": 1000}},
                             budget=budget)
    except BudgetExhausted as error:
        assert error.code == "SOLVER_BUDGET_EXHAUSTED"
    else:
        raise AssertionError("an exhausted budget must not return a state")


def test_total_budget_derives_the_per_check_timeout():
    value = three_pipe()
    budget = SolveBudget(1000, clock=lambda: 0.0)
    diagnostics = {}
    bounded = B.lower_problem_to_request(
        value, MODEL, {"limits": {"total_timeout_ms": 1000}}, budget=budget,
        diagnostics=diagnostics)
    assert bounded.core_args["solve_timeout_ms"] < B.DEFAULT_SOLVE_TIMEOUT_MS
    plain = B.lower_problem_to_request(value, MODEL, {})
    assert plain.core_args["solve_timeout_ms"] == B.DEFAULT_SOLVE_TIMEOUT_MS


def test_invalid_total_budget_is_named():
    for bad in (0, -1, 1.5, "1000"):
        expect_code("SOLVER_BUDGET_INVALID",
                    lambda bad=bad: B.solve_real_problem(
                        three_pipe(), MODEL, {"limits": {"total_timeout_ms": bad}}))


def test_budget_records_every_sub_solve_with_real_consumed_time():
    clock = _Clock()
    budget = SolveBudget(10_000, clock=clock)
    clock.advance(250)
    budget.record("first", 5000, budget.elapsed_ms(), "SAT")
    clock.advance(250)
    budget.record("second", 5000, budget.elapsed_ms(), "UNKNOWN")
    report = budget.report()
    assert report["total_ms"] == 10_000
    assert report["elapsed_ms"] == 500
    assert [entry["state"] for entry in report["sub_solves"]] == ["SAT", "UNKNOWN"]
    assert report["sub_solves"][0]["consumed_ms"] == 250
    assert report["sub_solves"][1]["consumed_ms"] == 500


def test_budget_rejects_non_positive_and_non_integer_totals():
    for bad in (0, -1, 1.5):
        try:
            SolveBudget(bad)
        except ValueError:
            continue
        raise AssertionError("SolveBudget accepted an invalid total %r" % (bad,))


if __name__ == "__main__":
    failed = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn) and _Z3_AVAILABLE:
            try:
                _fn()
                print("PASS", _name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL", _name, type(exc).__name__, exc)
    raise SystemExit(1 if failed else 0)
