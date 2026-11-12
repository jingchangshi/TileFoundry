"""Native symbolic completion surface.

The old unrolled trace/reemit path is archived. MAIN-B supplies the finite
CompletionSite/Channel implementation; this package refuses by name until then.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..plan.mainline import MainlineNotImplemented

__all__ = ["CompletionPlan", "CompletionError", "build_symbolic_completion",
           "require_completion_verified", "verify_completion"]


class CompletionError(RuntimeError):
    """A named completion refusal."""


@dataclass(frozen=True)
class CompletionPlan:
    sites: tuple = ()
    channels: tuple = ()
    verified: bool = False
    plan_iteration: int = 0
    identity: str = ""


def build_symbolic_completion(plan, target, effects=()):
    raise MainlineNotImplemented("COMPLETION_SYMBOLIC_NOT_IMPLEMENTED")


def verify_completion(plan, completion, target=None):
    raise MainlineNotImplemented("COMPLETION_VERIFY_NOT_IMPLEMENTED")


def require_completion_verified(completion):
    raise MainlineNotImplemented("COMPLETION_REQUIRED_NOT_VERIFIED")
