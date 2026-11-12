"""Native symbolic completion: the finite READY/FREE template of one plan.

The public surface is the frozen one:

    build_symbolic_completion(plan, target, effects) -> CompletionPlan
    verify_completion(plan, completion, target)     -> CompletionPlan (receipt attached)
    require_completion_verified(completion)         -> CompletionPlan (gate)

A completion states which generation of a buffer is safe to read (READY), when a
slot may be overwritten (FREE, every maximal reader publishes, writer k waits
k-(V+offset)), the region-exit tail as a SYMBOLIC window over the plan's own loop
bounds, and the physical hard-event namespace each channel occupies. The number
of sites and channels is decided by the plan's tasks and maximal readers, never
by the trip count.

completion/model.py stays the target-fact vocabulary (event pools, pipe facts,
physical namespaces); plan.py is the template builder; verify.py is the
independent recheck. There is one CompletionError and one CompletionPlan type.
"""
from __future__ import annotations

from .model import ASCEND_EVENT_FACTS, CompletionError
from .plan import (
    AFTER_TASK,
    BEFORE_TASK,
    CROSS_EVENT,
    DRAIN,
    EXIT,
    FIFO,
    FREE,
    LOCAL_EVENT,
    PIPE_BARRIER,
    READY,
    REGION_EXIT,
    STATE,
    CompletionChannel,
    CompletionPlan,
    CompletionSite,
    CompletionVerification,
    EventNamespaceKey,
    Mechanism,
    build_symbolic_completion,
    completion_identity,
    free_wait_generation,
    free_wait_guard,
    region_ordinal,
    region_tail_guard,
    region_tail_ordinal,
    region_trip_count,
    tail_window,
    task_domain_guard,
)
from .verify import require_completion_verified, verify_completion

__all__ = [
    "AFTER_TASK", "ASCEND_EVENT_FACTS", "BEFORE_TASK", "CROSS_EVENT",
    "CompletionChannel", "CompletionError", "CompletionPlan", "CompletionSite",
    "CompletionVerification", "DRAIN", "EventNamespaceKey", "EXIT", "FIFO", "FREE",
    "LOCAL_EVENT", "Mechanism", "PIPE_BARRIER", "READY", "REGION_EXIT", "STATE",
    "build_symbolic_completion", "completion_identity", "free_wait_generation",
    "free_wait_guard", "region_ordinal", "region_tail_guard", "region_tail_ordinal",
    "region_trip_count", "require_completion_verified", "tail_window",
    "task_domain_guard", "verify_completion",
]
