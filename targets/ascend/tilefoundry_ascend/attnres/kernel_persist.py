"""P5-03/G51: the persistent g += P kernel, with the residency fix that made it correct.

THE FIX THIS FILE EXISTS FOR. A shared (UB) buffer filled BEFORE a persistent T.Pipelined loop
does NOT hold its contents in later iterations. The resident gamma/w staging established outside
the loop is stale from iteration 1 onward, and the symptom is not a crash: it is plausible but
wrong numbers on every iteration after the first. Establishing residency INSIDE the iteration is
what makes the persistent schedule correct.

HOW THAT WAS ESTABLISHED, because the route matters more than the conclusion:
  * two reducers over the same quantity in one region PASS -> the second reducer was never it
  * a second reducer reading g_ub FAILS, reading w_ub PASSES -> g_ub is the FIRST buffer written
    before the loop, so the LEADING residency region is what goes stale
  * g_ub staged into a fragment first STILL FAILS -> it is g_ub's VALUE, not how it is read
  * residency re-established per iteration PASSES; residency kept with versions pinned to 1
    FAILS -> it is loop-carried residency, not buffer versioning

The scheduling claim and its cost are recorded separately (see the evidence record), because a
correct schedule is not automatically a fast one and this module makes no timing claim.
"""
from __future__ import annotations

PROGRAMS_DEFAULT = None   # None means one group per program (the G50 configuration)


def persistent_groups(tokens: int, programs: int) -> int:
    """ceil(M / P) -- how many iterations each program performs."""
    if programs < 1:
        raise ValueError("programs must be >= 1, got %r" % (programs,))
    return (tokens + programs - 1) // programs


def group_index(pid: int, iteration: int, programs: int) -> int:
    """The token group a program handles at one iteration: g = pid + it * P.

    Written as a function because this single expression IS the persistent schedule, and an
    off-by-one here silently computes a different group rather than failing.
    """
    if pid < 0:
        raise ValueError("pid must be >= 0, got %r" % (pid,))
    if iteration < 0:
        raise ValueError("iteration must be >= 0, got %r" % (iteration,))
    return pid + iteration * programs


def next_group(pid: int, iteration: int, programs: int) -> int:
    """The group the outer prefetch should target: g + P, i.e. this program's next iteration."""
    return group_index(pid, iteration + 1, programs)


def residency_placement() -> dict:
    """Where the persistent residency must live, and why -- stated once so it cannot drift.

    This is the module's central fact. A reviewer who moves the residency back outside the loop
    for efficiency would reintroduce exactly the defect this gate found.
    """
    return {
        "gamma": "INSIDE the token-group iteration",
        "w": "INSIDE the token-group iteration",
        "reason": ("a shared (UB) buffer written before a persistent loop does not hold its "
                   "contents in later iterations; residency outside the loop is stale from "
                   "iteration 1 onward"),
        "state_m_l_acc": ("declared outside the iteration, because it is per-group state that is "
                          "INITIALISED at the start of every group -- carrying it across groups "
                          "was never the defect and must not be 'fixed'"),
        "measured": "niter=2,3,4 at P=64/56/32/43 all correct; 20/20 official cases at P=32",
    }
