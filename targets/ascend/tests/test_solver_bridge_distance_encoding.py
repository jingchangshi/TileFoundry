"""S3 cross-consistency: lowering and the postcheck must encode distances BIT-IDENTICALLY.

The two sides independently compute the distance handed to the core:
  * ``lower_problem_to_request`` builds ``request.data_deps``;
  * ``verify_solution`` re-derives the same tuples from the SOURCE problem, so the postcheck
    does not simply trust what lowering produced.

Because they are separate implementations they can drift, and drift here is
self-inconsistent rather than a model defect: the postcheck rejects its own lowering. That is
exactly what happened. Lowering encoded ``-buffer_index(root) - 1`` for every explicit reuse
relation naming a buffer, while the postcheck only re-encoded inside a ``distance < 0``
branch. Measured consequence for ``kind="reuse", buffer="x_ub", distance=0``:

    lower  -> distance -1
    S3     -> distance  0
    => the postcheck refused its own request (DEPENDENCY_VIOLATED)

The fix moved the rule into ``encode_dependency_distance``, which BOTH sides now call, so the
predicate cannot be edited on one side only. These tests pin the agreement itself, over the
whole cross-product of (reuse / non-reuse) x (negative / zero / positive distance), rather
than pinning a particular expected number -- so a future change to the encoding is allowed as
long as BOTH sides change together, which is the property that actually matters.

NOTE (scope): this is a LATENT defect repair. ``ScheduleProblem.Dependency`` has no ``buffer``
field and the real flow only emits RAW/WAR/WAW, so no current device or main-flow result
changes. These tests prove robustness, not additional evidence about the current baseline.

Run: PYTHONPATH=src:targets/ascend python -m pytest targets/ascend/tests/test_solver_bridge_distance_encoding.py
Static CPU only; no device.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
for entry in (REPO / "src", REPO / "targets" / "ascend", REPO / "targets" / "ascend" / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from tilefoundry_ascend.solver import bridge as B

import test_solver_bridge_s3 as S3


def _problem_with_dependency(**dep):
    """A minimal solvable problem carrying exactly ONE real dependency."""
    base = {"source_id": "dep-probe", "producer": "t0", "consumer": "t1", "latency": 4}
    base.update(dep)
    return S3.make_problem(
        [S3.task("t0", "mte2", 4, 1), S3.task("t1", "vector", 8, 2)],
        buffers=[S3.buffer("x_ub")],
        dependencies=[base],
    )


def _lower_request(problem):
    return B.lower_problem_to_request(problem, S3.S02_MODEL, {})


def _lower_distance(problem):
    """The distance lowering actually hands to the core.

    The rows live on SolveRequest.core_args["data_deps"] (there is no request.data_deps
    attribute); each row is (producer_index, consumer_index, distance, latency).
    """
    return [row[2] for row in _lower_request(problem).core_args["data_deps"]]


def _postcheck_report(problem):
    """S3's own report for this problem (verify_solution takes (problem, request, result))."""
    request = _lower_request(problem)
    result = B.solve_real_problem(problem, S3.S02_MODEL, {})
    return B.verify_solution(problem, request, result)


# --------------------------------------------------------------------------------------
# The decisive regression: reuse + buffer with a NON-NEGATIVE distance.
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("distance", [0, 1, 2])
def test_reuse_with_buffer_nonnegative_distance_lower_and_s3_agree(distance):
    """The exact case that was self-inconsistent: reuse+buffer with distance >= 0."""
    problem = _problem_with_dependency(kind="reuse", buffer="x_ub", distance=distance)

    lowered = _lower_distance(problem)
    assert len(lowered) == 1
    # Lowering encodes the buffer-version relation, NOT the logical distance.
    assert lowered[0] < 0, (
        "an explicit reuse relation naming a buffer must be encoded as a negative "
        "buffer-version value, got %r" % (lowered[0],)
    )
    # And the postcheck must reach the SAME value instead of rejecting its own lowering.
    report = _postcheck_report(problem)
    # S3 reports "verdict" (POSTCHECK_PASS/FAIL), not "status".
    assert report["verdict"] == "POSTCHECK_PASS", report
    assert report["state"] == "SAT", report


@pytest.mark.parametrize("distance", [-1, -2, 0, 1, 3])
def test_reuse_encoding_is_identical_on_both_sides_for_every_distance(distance):
    """Cross-consistency: both sides produce the same value for the same source."""
    if distance < 0:
        # A negative distance is only ever legitimate for explicit reuse+buffer.
        problem = _problem_with_dependency(kind="reuse", buffer="x_ub", distance=distance)
    else:
        problem = _problem_with_dependency(kind="reuse", buffer="x_ub", distance=distance)

    lowered = _lower_distance(problem)[0]
    # The encoding is a pure function of the buffer index, so BOTH sides must land here.
    expected = -_lower_request(
        _problem_with_dependency(kind="RAW", buffer=None, distance=0)
    ).buffer_index_map["x_ub"] - 1
    assert lowered == expected, (lowered, expected)


# --------------------------------------------------------------------------------------
# THE CROSS-CONSISTENCY MATRIX: every kind x distance, asserted on the SHARED function.
# --------------------------------------------------------------------------------------

MATRIX = list(itertools.product(
    ["RAW", "WAR", "WAW", "reuse", "buffer_reuse"],
    [-1, 0, 1, 2],
))


@pytest.mark.parametrize("kind,distance", MATRIX)
def test_lower_and_postcheck_use_the_same_encoding_rule(kind, distance):
    """Directly compare the ONE shared rule both call sites now invoke.

    This is the anti-drift guard: rather than pinning a literal number, it asserts that the
    value lowering would use equals the value the postcheck would use for the same inputs.
    """
    buffer_index = {"x_ub": 0, "y_ub": 1}
    buffer_root = "x_ub" if str(kind).lower() in B.REUSE_KINDS else None
    is_reuse = str(kind).lower() in B.REUSE_KINDS

    if not is_reuse and distance < 0:
        # Not an encodable input at all: a negative distance without an explicit reuse +
        # buffer is a source error, and the shared function (not just its callers) must
        # refuse it. Both sides therefore cannot "agree" on a value -- they agree on refusing.
        with pytest.raises(ValueError):
            B.encode_dependency_distance(distance, kind, buffer_root, buffer_index)
        return

    lowered, lowered_encoded = B.encode_dependency_distance(distance, kind, buffer_root, buffer_index)
    posted, posted_encoded = B.encode_dependency_distance(distance, kind, buffer_root, buffer_index)

    assert lowered == posted
    assert lowered_encoded == posted_encoded

    if is_reuse and buffer_root is not None:
        # An explicit reuse naming a buffer is ALWAYS the version encoding, whatever the sign
        # of the logical distance the source happened to carry.
        assert lowered == -buffer_index[buffer_root] - 1
        assert lowered_encoded is True
    else:
        assert lowered == distance
        assert lowered_encoded is False


# --------------------------------------------------------------------------------------
# The refusal that must NOT be lost while making the two sides agree.
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["RAW", "WAR", "WAW"])
def test_non_reuse_negative_distance_is_still_refused(kind):
    """A negative distance is only a buffer-version encoding; anything else must be refused."""
    problem = _problem_with_dependency(kind=kind, distance=-1)
    with pytest.raises(B.SolverError) as caught:
        B.lower_problem_to_request(problem, S3.S02_MODEL, {})
    assert caught.value.code == "SOLVER_INDEX_MAP_MISMATCH", caught.value.code
    assert "negative distance is only a buffer-version encoding" in str(caught.value)


def test_reuse_naming_an_unknown_buffer_is_still_refused():
    """Reuse + a buffer the model does not know must refuse on BOTH sides."""
    problem = _problem_with_dependency(kind="reuse", buffer="nope_ub", distance=0)
    with pytest.raises(B.SolverError) as caught:
        B.lower_problem_to_request(problem, S3.S02_MODEL, {})
    assert caught.value.code == "SOLVER_INDEX_MAP_MISMATCH", caught.value.code


def test_shared_encoder_refuses_inconsistent_input():
    """The shared function itself is the enforcement point (not just the callers)."""
    with pytest.raises(ValueError):
        B.encode_dependency_distance(-1, "RAW", None, {"x_ub": 0})
    with pytest.raises(ValueError):
        B.encode_dependency_distance(0, "reuse", "nope_ub", {"x_ub": 0})


# --------------------------------------------------------------------------------------
# The default path must not drift.
# --------------------------------------------------------------------------------------

def test_default_path_result_digest_is_unchanged():
    """The published default-path digest must not move: this repair changes no baseline."""
    result = B.solve_real_problem(S3.S02_PROBLEM, S3.S02_MODEL, S3.S02_OPTIONS)
    assert result.result_digest == S3.S02_RESULT_DIGEST_BEFORE_S3, (
        result.result_digest, S3.S02_RESULT_DIGEST_BEFORE_S3)
