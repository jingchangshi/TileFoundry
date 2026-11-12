"""The ONE total time budget shared by a tree of nested sub-solves (P3-01 / G30).

Why this exists
---------------
``bridge._timeout_ms`` bounds ONE z3 feasibility check. That is not a bound on a SOLVE:

  * the core binary-searches II, so one solve performs O(log(range)) checks, each of
    which may consume the whole per-check timeout;
  * a hierarchical schedule solves several sub-problems, and each layer historically
    received a FRESH full budget. A tree of depth d could therefore run for d x the
    intended total, which is exactly the "每层重新获得完整预算" defect P3-01 must close.

A ``SolveBudget`` is a deadline, not a duration. A child view SHARES the parent's
deadline, so nesting cannot re-grant time. Every observation of the remaining budget is
clamped to the parent's, and a child can never outlive its parent.
"""
from __future__ import annotations

import time

#: Default TOTAL wall-clock budget for one solve tree, in milliseconds.
#:
#: This is a SOLVE bound, not a per-check bound: the core's binary search and any nested
#: sub-problems all draw from it. The historical per-check bound stays
#: bridge.DEFAULT_SOLVE_TIMEOUT_MS and is unaffected unless a caller asks for a total.
DEFAULT_TOTAL_BUDGET_MS = 900_000


class BudgetExhausted(RuntimeError):
    """The shared deadline is spent; the correct state is UNKNOWN, never UNSAT."""

    code = "SOLVER_BUDGET_EXHAUSTED"

    def __init__(self, consumed_ms, total_ms):
        self.consumed_ms = consumed_ms
        self.total_ms = total_ms
        super().__init__("%s: consumed %d ms of a %d ms total budget"
                         % (self.code, consumed_ms, total_ms))


class SolveBudget:
    """A monotonically shrinking deadline shared by every nested sub-solve.

    ``clock`` is injectable so a test can drive time deterministically instead of
    sleeping. It defaults to ``time.monotonic``, which cannot go backwards.
    """

    def __init__(self, total_ms=None, *, clock=time.monotonic):
        if total_ms is None:
            total_ms = DEFAULT_TOTAL_BUDGET_MS
        if type(total_ms) is not int or total_ms <= 0:
            raise ValueError("total budget must be a positive integer of milliseconds, got %r"
                             % (total_ms,))
        self._clock = clock
        self.total_ms = int(total_ms)
        self._started = clock()
        self._deadline = self._started + self.total_ms / 1000.0
        #: Sub-solve records, in completion order: (label, granted_ms, consumed_ms, state).
        self.sub_solves = []

    # ---------------------------------------------------------------- deadline
    def remaining_ms(self):
        """Whole milliseconds left. 0 means the deadline is spent (never negative)."""
        left = self._deadline - self._clock()
        return max(0, int(left * 1000.0))

    def elapsed_ms(self):
        return max(0, int((self._clock() - self._started) * 1000.0))

    @property
    def exhausted(self):
        return self.remaining_ms() <= 0

    def child(self, label=""):
        """A view sharing THIS deadline. Never a fresh budget."""
        view = object.__new__(SolveBudget)
        view._clock = self._clock
        view.total_ms = self.total_ms
        view._started = self._started
        view._deadline = self._deadline      # the SAME absolute deadline
        view.sub_solves = self.sub_solves    # one shared ledger
        view._label = label
        return view

    # -------------------------------------------------------------- per-check
    def per_check_ms(self, requested_ms, checks_expected):
        """Split the remaining budget across the checks a solve still has to make.

        ``requested_ms`` is the caller's per-check bound (None disables the per-check
        bound historically). The return value is the smaller of that bound and an even
        share of what is actually left, so the SUM of the checks stays inside the total.
        Returns 0 when nothing is left, which callers must treat as "do not start".
        """
        remaining = self.remaining_ms()
        if remaining <= 0:
            return 0
        share = max(1, remaining // max(1, int(checks_expected)))
        if requested_ms is None:
            return share
        return min(int(requested_ms), share)

    # ---------------------------------------------------------------- records
    def record(self, label, granted_ms, consumed_ms, state):
        """Record ONE sub-solve. Consumed time is real elapsed time, not the grant."""
        self.sub_solves.append({"label": str(label), "granted_ms": int(granted_ms),
                                "consumed_ms": int(consumed_ms), "state": str(state),
                                "remaining_ms": self.remaining_ms()})

    def report(self):
        return {
            "total_ms": self.total_ms,
            "elapsed_ms": self.elapsed_ms(),
            "remaining_ms": self.remaining_ms(),
            "exhausted": self.exhausted,
            "sub_solves": [dict(entry) for entry in self.sub_solves],
        }


__all__ = ["BudgetExhausted", "DEFAULT_TOTAL_BUDGET_MS", "SolveBudget"]
