"""Cross-module parity: C's and R's loop-execution predicates must agree.

Why this test exists
--------------------
The "static presence != execution reachability" defect appeared three times in a row in
this package, each time as "the predicate exists but one consumer did not route through
it":

  1. C: ``_walk`` recursed into both ``If`` arms unconditionally while only ``Drain``
     consulted reachability.
  2. C: the same walk treated ``While``/``For`` as plain bodies, so ``While(False)`` and a
     zero-trip ``For`` re-opened the gap for a different statement class.
  3. R: ``index_statements_ledger`` had the identical blind spot, so a rewritten write
     inside a non-executing loop still counted as ``applied``.

C and R now each hold a loop-execution predicate:

  * ``tilefoundry_ascend.sync._loop_body_reachability(stmt, reachable)``
  * ``tilefoundry_ascend.plan.materialize._loop_body_can_execute(node)``

They are deliberately NOT shared: ``sync`` and ``plan.materialize`` do not import each
other at all, and collapsing them would introduce a horizontal dependency between the
completion layer and the plan layer. Instead of forcing one implementation, this test
pins the PROPERTY that matters -- the two predicates must give the same answer.

The assertions deliberately do NOT hard-code expected booleans. Pinning literal values
would freeze today's rule and would have to be rewritten whenever the rule legitimately
changes on both sides. What must hold for all time is "the two modules agree", exactly
as the S3 cross-encoding test asserts "lower and postcheck agree" rather than pinning a
literal distance. Either side drifting alone turns this red.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
for entry in (REPO / "src", REPO / "targets" / "ascend", REPO / "targets" / "ascend" / "tests"):
    sys.path.insert(0, str(entry))

from tilefoundry.ir.core import Constant, Var
from tilefoundry.ir.tir.stmts import For, While
from tilefoundry.ir.types import make_tensor_type
from tilefoundry_ascend.plan.materialize import _loop_body_can_execute
from tilefoundry_ascend.sync import _loop_body_reachability

_U32 = make_tensor_type((), dtype="int32", storage="umat")
_BOOL = make_tensor_type((), dtype="bool", storage="umat")
_IV = Var(name="i", type=_U32)


def _i32(value: int) -> Constant:
    return Constant(type=_U32, value=value)


def _bool(value: bool) -> Constant:
    return Constant(type=_BOOL, value=value)


def _for(start: int, stop: int, step: int) -> For:
    return For(induction_var=_IV, start=_i32(start), stop=_i32(stop), step=_i32(step),
               body=None)


def _while(value: bool) -> While:
    return While(cond=_bool(value), body=None)


def _c_side(stmt) -> bool:
    return bool(_loop_body_reachability(stmt, True))


def _r_side(stmt) -> bool:
    return bool(_loop_body_can_execute(stmt))


# Bounds chosen to straddle every sign/empty case with small numbers:
# negative steps, start > stop in both directions, single-trip, and exact multiples.
BOUNDS = range(-3, 5)
STEPS = (-2, -1, 0, 1, 2, 3)


def test_constant_for_predicates_agree_on_every_sign_combination():
    """For all constant (start, stop, step) shapes C and R must agree.

    Includes the two rules that look obvious and are both WRONG:
      * "step must be 1"          -> would mark every descending loop dead
      * "start >= stop is empty"  -> would mark every descending loop dead
    A descending loop such as For(4, 0, -1) really does run four times.
    """
    checked = 0
    mismatches = []
    for start, stop, step in itertools.product(BOUNDS, BOUNDS, STEPS):
        stmt = _for(start, stop, step)
        c_side, r_side = _c_side(stmt), _r_side(stmt)
        checked += 1
        if c_side != r_side:
            mismatches.append((start, stop, step, c_side, r_side))
    assert checked == len(BOUNDS) * len(BOUNDS) * len(STEPS), checked
    assert not mismatches, (
        "C (_loop_body_reachability) and R (_loop_body_can_execute) disagree on "
        "For(start, stop, step); the two predicates have drifted apart: %r" % (mismatches,)
    )


def test_while_predicates_agree():
    for value in (False, True):
        stmt = _while(value)
        assert _c_side(stmt) == _r_side(stmt), (
            "predicates disagree on While(%s)" % value
        )


def test_descending_constant_for_is_reachable_on_both_sides():
    """For(4, 0, -1) executes (i = 4, 3, 2, 1) and must not be treated as dead.

    Stated as an explicit, named guard in addition to the parity sweep: a future "fix"
    that declares every descending loop dead would keep the two sides in agreement while
    silently discarding real applied writes. Parity alone cannot catch an error made
    identically on both sides, so the sign semantics are also pinned once, here.
    """
    assert _c_side(_for(4, 0, -1)) is True
    assert _r_side(_for(4, 0, -1)) is True


def test_zero_trip_and_false_while_are_dead_on_both_sides():
    """The genuinely non-executing shapes must be dead on both sides."""
    for stmt in (_for(0, 0, 1), _for(4, 0, 1), _for(0, 4, -1), _while(False)):
        assert _c_side(stmt) is False, stmt
        assert _r_side(stmt) is False, stmt


def test_parity_guard_is_falsifiable():
    """Sanity: the sweep would actually catch a divergence.

    Proves the comparison is not vacuous -- if one side were replaced by a constant the
    sweep would report mismatches. We simulate that by comparing against an intentionally
    wrong predicate over the same domain and asserting it DOES disagree.
    """
    divergent = 0
    for start, stop, step in itertools.product(BOUNDS, BOUNDS, STEPS):
        truth = _c_side(_for(start, stop, step))
        if truth != True:  # a plausible wrong rule: "all loops execute"
            divergent += 1
    assert divergent > 0, (
        "the sweep domain contains no shape where a constant-True predicate would differ, "
        "so the parity test would not be able to detect drift"
    )
