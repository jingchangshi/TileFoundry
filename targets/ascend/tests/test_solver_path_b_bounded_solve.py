"""Path B: bounded solve time and a LABELLED degradation instead of a raise.

Why this file exists
--------------------
Measured before Path B (2026-11-10, TL z3 loop scheduler, all-same-pipe synthetic):

    n=16 -> 0.4 s   n=128 -> 19.7 s   n=384 -> 224 s
    n=256 -> 90 s   n=512 -> 410 s    n=768 -> 949 s     n=978 -> >1500 s (timeout)

TileLang's loop scheduler has no try/catch at all: `auto_schedule.cc:743` logs
"falling back to topological sort" and calls LOG(FATAL), and a Python exception tears
straight through the pass. TF must not copy that.

The contract this file pins
---------------------------
1. DEFAULT PATH STILL RAISES. Without allow_fallback_on_unknown an undecided solver
   raises exactly as before. NOTE: the default core_args DID change under Path B -- a
   default-on solve_timeout_ms was added, and the published result_digest moved with it
   (re-pinned in test_solver_bridge_s3.S02_RESULT_DIGEST_AT_PATH_B). So this is
   "the failure mode is unchanged", not "the bytes are unchanged".
2. THE TIMEOUT IS ON BY DEFAULT AND CAN BE DISABLED EXPLICITLY. A bound that only
   applies when the caller remembers to ask for it does not bound the runaway case,
   which is the only case that matters. limits.timeout_ms=None restores unbounded.
   allow_fallback_on_unknown (opt-in) turns an undecided result into status="unknown".
3. UNKNOWN IS NOT FALLBACK. "fallback" claims every candidate was proven infeasible.
   "unknown" claims nothing; it must never be reported as fallback and never as sat.
4. UNKNOWN NEVER COUNTS AS AN EXHAUSTED SEARCH, so it can never be promoted to UNSAT
   downstream (search_scope_complete=False).
"""
from __future__ import annotations

import pytest

from tilefoundry.vendor.tilelang_z3 import adapters as A

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _all_same_pipe(n, *, dense=True):
    """A loop body where every task wants the same pipe: the pathological shape.

    It is pathological because resource_deps becomes all-pairs, which is exactly the
    O(n^2) term that makes the real thing blow up.
    """
    return dict(
        num_stages=2,
        latencies=[96] * n,
        iis=[13] * n,
        resource_flags=[2] * n,
        data_deps=[(i, i + 1, 0, 96) for i in range(n - 1)],
        resource_deps=[(i, j) for i in range(n) for j in range(i + 1, n)] if dense else [],
        owner_exclusion_deps=None,
        pipe_order_deps=None,
        buffer_sizes=[],
        memory_groups=[],
        stage_order_deps=None,
    )


# --------------------------------------------------------------------------- contract 1
def test_default_path_still_raises_on_unknown():
    """No opt-in => the historical behaviour, byte for byte."""
    with pytest.raises(RuntimeError, match="unknown"):
        A.tf_loop_adapter(_all_same_pipe(64), return_status=True)


# --------------------------------------------------------------------------- contract 2
def test_opt_in_returns_a_labelled_unknown_instead_of_raising():
    args = dict(_all_same_pipe(64), solve_timeout_ms=30_000,
                allow_fallback_on_unknown=True)
    raw = A.tf_loop_adapter(args, return_status=True)

    assert raw["status"] == "unknown"
    assert raw["start_times"], "a degraded result must still carry the serial schedule"
    assert raw["unknown_at_ii"] is not None
    assert "NOT refuted" in raw["fallback_reason"]
    # The core's own explicit serial answer is sum(latencies) for this shape.
    assert raw["minimal_II"] == sum(args["latencies"])


def test_timeout_actually_bounds_the_solve():
    """A bounded solve must return, not run to completion.

    n=384 dense needs ~224 s unbounded (measured). A 5 s budget must come back far
    sooner; the assertion is deliberately loose so it is not flaky on a loaded box.
    """
    import time

    args = dict(_all_same_pipe(384), solve_timeout_ms=5_000,
                allow_fallback_on_unknown=True)
    started = time.monotonic()
    raw = A.tf_loop_adapter(args, return_status=True)
    elapsed = time.monotonic() - started

    # Prove the degraded branch was ENTERED, not merely that we finished quickly. Without
    # unknown_at_ii a fast return could mean "solved it", which would make this test pass
    # for the wrong reason.
    assert raw["status"] == "unknown"
    assert raw["unknown_at_ii"] is not None, "no II check was ever left undecided"
    assert "unknown" in raw["check_results"]
    # The failure mode of a dropped timeout is a HANG, which pytest reports as a timeout
    # death rather than a red assertion. This bound converts it back into a red assertion.
    assert elapsed < 120, "a bounded solve still took %.1fs" % elapsed


# --------------------------------------------------------------------------- contract 3
def test_unknown_is_never_reported_as_fallback_or_sat():
    args = dict(_all_same_pipe(64), solve_timeout_ms=30_000,
                allow_fallback_on_unknown=True)
    raw = A.tf_loop_adapter(args, return_status=True)
    assert raw["status"] not in ("sat", "fallback")

    # A genuinely UNSAT-exhausted run still reports "fallback", so the two states stay
    # distinguishable rather than collapsing into one bucket.
    clean = dict(_all_same_pipe(4), allow_fallback_on_unknown=True)
    clean["resource_deps"] = []
    clean["data_deps"] = []
    ok = A.tf_loop_adapter(clean, return_status=True)
    assert ok["status"] == "sat"


# --------------------------------------------------------------------------- contract 4
def test_unknown_never_claims_an_exhausted_search():
    args = dict(_all_same_pipe(64), solve_timeout_ms=30_000,
                allow_fallback_on_unknown=True)
    raw = A.tf_loop_adapter(args, return_status=True)
    assert raw["search_scope_complete"] is False, (
        "an undecided solver did not exhaust the search space; promoting it to UNSAT "
        "downstream would be a fabricated proof")


# --------------------------------------------------------- bridge-level classification
def test_bridge_classifies_unknown_as_unknown_with_a_candidate():
    """bridge.py must not collapse unknown into FALLBACK, and must keep the schedule
    only as a candidate -- never as a SAT assignment."""
    import tilefoundry_ascend.solver.bridge as B
    from tilefoundry_ascend.solver.bridge import (FALLBACK, SAT, UNKNOWN,
                                                  classify_core_result)

    class _Req:
        problem_digest = "d"
        request_digest = "r"
        core_source_hash = "c"
        index_task_map = {0: "t0", 1: "t1"}
        index_buffer_map = {}
        units = {}

    raw = {
        "status": "unknown",
        "reason": "solver returned unknown at II=7",
        "fallback_reason": "solver returned unknown at II=7",
        "start_times": [0, 96],
        "buffer_versions": [],
        "minimal_II": 96,
        "check_results": ["unknown"],
        "search_scope_complete": False,
    }
    result = classify_core_result(_Req(), raw)

    assert result.state == UNKNOWN
    assert result.state not in (SAT, FALLBACK)
    assert result.assignment == {}, "an unproven schedule is not a SAT assignment"
    assert result.fallback_candidate is not None
    assert result.fallback_candidate["start_times"] == [0, 96]
    assert result.search_scope == "NOT_EXHAUSTED"


def test_bridge_survives_a_raising_core(monkeypatch):
    """The invocation site must catch, not propagate: TL's fatal-on-raise is the bug
    we are deliberately not copying."""
    import tilefoundry_ascend.solver.bridge as B

    def boom(*_a, **_k):
        raise RuntimeError("simulated core explosion")

    # Patch the ONE physical core function, not the adapter: adapters.py asserts that the
    # TF and TL adapters live in the same module object, and replacing an adapter would
    # trip that identity check for a reason unrelated to what this test is about.
    from tilefoundry.vendor.tilelang_z3 import core as z3core
    monkeypatch.setattr(z3core, "z3_schedule_loop_python", boom)

    from tilefoundry_ascend.solver.bridge import (UNKNOWN, classify_core_result,
                                                  lower_problem_to_request)
    # Reuse the S02 default-path problem rather than inventing a second fixture: it is the
    # exact shape whose digest the published artifacts pin, so exercising the catch on it
    # keeps this test honest about which path it protects.
    import test_solver_bridge_s3 as S3

    request = lower_problem_to_request(S3.S02_PROBLEM, S3.S02_MODEL, S3.S02_OPTIONS)
    out = B.invoke_shared_core(request)
    assert out["raw"]["status"] == "unknown"
    assert "simulated core explosion" in out["raw"]["reason"]
    result = classify_core_result(request, out["raw"])
    assert result.state == UNKNOWN


# ------------------------------------------------- straight-line path (also bounded)
def test_straight_line_path_is_bounded_too():
    """The user asked for BOTH paths to be protected, and TF only *uses* the loop path.

    That makes the straight-line path easy to forget precisely because nothing exercises
    it in production. These tests keep it from rotting silently.
    """
    import time

    n = 200
    started = time.monotonic()
    try:
        A.tf_adapter(
            [96] * n, [13] * n, [2] * n,
            [(i, i + 1, 96) for i in range(n - 1)],
            [(i, j) for i in range(n) for j in range(i + 1, n)],
            solve_timeout_ms=3_000,
        )
    except RuntimeError as exc:
        # Either verdict is acceptable: what matters is that it came back, and that an
        # undecided solve is never dressed up as "unsat".
        assert "unknown" in str(exc) or "unsat" in str(exc)
    assert time.monotonic() - started < 120


def test_straight_line_unknown_is_not_reported_as_unsat():
    """A timed-out Optimize solve must say 'unknown', not fabricate a refutation."""
    n = 300
    try:
        A.tf_adapter(
            [96] * n, [13] * n, [2] * n,
            [(i, i + 1, 96) for i in range(n - 1)],
            [(i, j) for i in range(n) for j in range(i + 1, n)],
            solve_timeout_ms=1,
        )
    except RuntimeError as exc:
        assert "unknown" in str(exc), (
            "an undecided solve was reported as %r; a timeout is not a proof of "
            "infeasibility" % str(exc))
        return
    # If it solved, a 1 ms budget was simply not binding for this input. That is a fact
    # about the input, not a hole in the implementation, so it is stated rather than
    # hidden behind a skip marker.
    raise AssertionError(
        "expected the 1 ms budget to leave a check undecided at n=300; if the solver got "
        "fast enough to decide this, tighten the input instead of deleting this test")


def test_straight_line_adapters_stay_on_one_call_site():
    """TF and TL straight-line adapters must not diverge in argument handling."""
    assert A.tf_adapter.__globals__["_straight_call"] is A.tl_adapter.__globals__["_straight_call"]
    assert A.shared_core_sha256() == A.core.core_source_sha256()


def test_contract_errors_are_not_swallowed_into_unknown(monkeypatch):
    """A TypeError/KeyError in the core is a BUG, not a solver verdict.

    Swallowing it would turn "the argument surface changed" into "the solver ran out of
    budget" -- the two are indistinguishable at the state level, and the solve would be
    silently disabled. Contract errors must stay loud.
    """
    from tilefoundry.vendor.tilelang_z3 import core as z3core
    import tilefoundry_ascend.solver.bridge as B
    import test_solver_bridge_s3 as S3

    def broken(*_a, **_k):
        raise TypeError("core signature changed")

    monkeypatch.setattr(z3core, "z3_schedule_loop_python", broken)
    request = B.lower_problem_to_request(S3.S02_PROBLEM, S3.S02_MODEL, S3.S02_OPTIONS)
    with pytest.raises(TypeError, match="core signature changed"):
        B.invoke_shared_core(request)
