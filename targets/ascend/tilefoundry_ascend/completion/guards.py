"""Guard algebra for conditional completion edges.

A guard decides WHETHER a statement executes. For a completion edge this is not cosmetic: a
WAIT inside a conditional that the SET does not share is a hang, and a SET that fires while
its WAIT is skipped leaves a slot live at region exit. Therefore the checker needs to decide
IMPLICATION (can the waiter run when the setter does not?) rather than compare counts.

The algebra is deliberately small and conservative:

  ALWAYS   the statement executes unconditionally.
  NEVER    the statement cannot execute (a statically false branch).
  COND     it executes when a predicate holds, identified by a structural key.

IMPLICATION is answered exactly for ALWAYS/NEVER and by structural identity for COND. Two
DIFFERENT predicates are NOT assumed to imply one another: an unproven implication is refused
by name instead of being optimistically accepted, because "the predicates look similar" is
how a conditional deadlock reaches hardware.
"""
from __future__ import annotations

from dataclasses import dataclass

from .model import digest

ALWAYS = "always"
NEVER = "never"
COND = "cond"


@dataclass(frozen=True)
class Guard:
    kind: str
    key: str = ""
    source: object = None

    @property
    def is_conditional(self) -> bool:
        return self.kind == COND

    def as_dict(self):
        return {"kind": self.kind, "key": self.key}


GUARD_ALWAYS = Guard(ALWAYS)
GUARD_NEVER = Guard(NEVER)


def normalize_guard(guard) -> Guard:
    """A comparable Guard for any of the guard shapes the plan and TIR can carry."""
    if guard is None:
        return GUARD_ALWAYS
    if isinstance(guard, Guard):
        return guard
    if isinstance(guard, bool):
        return GUARD_ALWAYS if guard else GUARD_NEVER
    # tilefoundry Constant
    value = getattr(guard, "value", None)
    if isinstance(value, bool) and not isinstance(guard, (dict,)):
        return GUARD_ALWAYS if value else GUARD_NEVER
    # PExpr / our own dict expressions: a const is decidable, anything else is COND.
    op = getattr(guard, "op", None) or (guard.get("op") if isinstance(guard, dict) else None)
    if op == "const":
        const = getattr(guard, "value", None)
        if const is None and isinstance(guard, dict):
            const = guard.get("value")
        if isinstance(const, int):
            return GUARD_NEVER if const == 0 else GUARD_ALWAYS
    key = guard_key(guard)
    return Guard(COND, key=key, source=guard)


def guard_key(guard) -> str:
    """A stable structural key. Used ONLY for identity, never as a proof of implication."""
    if isinstance(guard, Guard):
        return guard.key or "always"
    if guard is None:
        return "always"
    if hasattr(guard, "to_dict"):
        return digest(guard.to_dict())
    return digest(repr(guard))


def guard_implies(waiter: Guard, setter: Guard) -> tuple:
    """Does 'waiter executes' imply 'setter executes'? Returns (ok, reason).

    This is the deadlock question for one edge: if the waiter can run while the setter is
    skipped, the WAIT can never be satisfied.
    """
    if waiter.kind == NEVER:
        return True, "the waiting statement cannot execute, so the edge is vacuous"
    if setter.kind == ALWAYS:
        return True, "the setter executes unconditionally"
    if waiter.kind == ALWAYS:
        return False, "the waiter executes unconditionally while the setter is conditional"
    if setter.kind == NEVER:
        return False, "the setter is statically dead while the waiter is live"
    if waiter.key == setter.key:
        return True, "both sides use the SAME predicate"
    return False, ("the waiter's predicate differs from the setter's; implication is not"
                   " proven from two different predicates")


def guards_agree(left: Guard, right: Guard) -> tuple:
    """Do two SIDES of one cross-core handoff use a compatible group predicate?"""
    if left.kind == ALWAYS and right.kind == ALWAYS:
        return True, "both sides are unconditional"
    if left.kind == right.kind and left.key == right.key:
        return True, "both sides use the SAME predicate"
    return False, ("the two sides of the handoff disagree about the group predicate"
                   " (%s:%s vs %s:%s); one side could skip its Set while the other Waits"
                   % (left.kind, left.key, right.kind, right.key))


__all__ = ["ALWAYS", "NEVER", "COND", "Guard", "GUARD_ALWAYS", "GUARD_NEVER",
           "normalize_guard", "guard_key", "guard_implies", "guards_agree"]
