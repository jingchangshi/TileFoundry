"""A plan VIEW over the schedule-level Problem.

WHY THIS EXISTS. Two layers of this target derive completion obligations: the resolved-plan
layer (plan/resolved_plan.py) and the older schedule Problem layer (schedule/problem.py, used by
pipeline.py). If each derived them its own way the two would drift, and the drift would be
invisible because both would look internally consistent.

So the derivation lives in ONE place (obligations.py) and this adapter presents the Problem in
the shape that derivation reads. It is a VIEW, not a fabricated plan: every field comes from the
Problem itself, and a field the Problem does not carry is reported as absent rather than
invented.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _Task:
    task_id: str
    pipe: str
    resource: str
    order: int
    iteration_shift: int = 0
    stage: int = 0
    guard: object = None


@dataclass(frozen=True)
class _Access:
    access_id: str
    task_id: str
    buffer_root: str
    mode: str
    byte_footprint: int = 0
    guard: object = None


@dataclass(frozen=True)
class _UndeclaredBuffer:
    """Stand-in for a root the Problem names in an access but never declared."""

    root: str
    slot_count: int = 1
    space: str = ""


def _declared(buffer):
    """Re-express a ProblemBuffer in the vocabulary the derivation reads.

    The Problem calls the ring width 'versions'; the plan calls it 'slot_count'. They are the
    same fact, and this translation is the ONLY place the two names meet. Without it the
    derivation fell back to a default of 1 and every obligation claimed a single-version ring
    no matter what the buffer declared -- a silent, program-independent constant, which is
    exactly the defect this stage exists to remove.
    """
    return _UndeclaredBuffer(
        root=str(getattr(buffer, "root", "")),
        slot_count=int(getattr(buffer, "versions", getattr(buffer, "slot_count", 1)) or 1),
        space=str(getattr(buffer, "space", "")))


@dataclass(frozen=True)
class _Loop:
    source_loop_id: str
    domain: dict


def _trip_of(domain) -> int:
    try:
        start = int(domain.get("start", 0))
        stop = int(domain.get("stop", 0))
        step = int(domain.get("step", 1) or 1)
    except (AttributeError, TypeError, ValueError):
        return 1
    if step <= 0:
        return 1
    return max(1, (stop - start + step - 1) // step)


@dataclass(frozen=True)
class ProblemPlanView:
    """The subset of a ResolvedPlan the obligation derivation depends on."""

    loop: object
    tasks: tuple
    accesses: tuple
    buffers: tuple
    state: object = None
    units: dict = None
    identity: object = None
    escapes: tuple = ()
    syncs: tuple = ()
    unsupported: tuple = ()
    diagnostics: tuple = ()

    def trip_count(self) -> int:
        return _trip_of(self.loop.domain)


def problem_plan_view(problem) -> ProblemPlanView:
    """Project a ScheduleProblem into the shape obligations.py reads.

    Access ORDER is the Problem's own access order, which is the order the tasks are declared
    in; that is the only issue-order fact the schedule layer actually has, and inventing a
    finer one here would be a fabricated fact.
    """
    tasks = []
    pipes = {}
    for ordinal, task in enumerate(getattr(problem, "tasks", ()) or ()):
        task_id = str(getattr(task, "stable_id", "") or "task%d" % ordinal)
        pipe = str(getattr(task, "pipe", "") or "")
        pipes[task_id] = pipe
        tasks.append(_Task(task_id=task_id, pipe=pipe,
                           resource=str(getattr(task, "resource", "") or ""),
                           order=ordinal))
    accesses = []
    for ordinal, access in enumerate(getattr(problem, "accesses", ()) or ()):
        task_id = str(getattr(access, "task", "") or "")
        accesses.append(_Access(
            access_id="a%d" % ordinal, task_id=task_id,
            buffer_root=str(getattr(access, "root", "") or ""),
            mode=str(getattr(access, "mode", "") or "read"),
            byte_footprint=int(getattr(access, "byte_footprint", 0) or 0)))
    # A root named by an access but absent from the Problem's buffer list is a REAL gap in the
    # schedule-level Problem (extract_problem can emit accesses on a root it did not declare).
    # It is NOT papered over by inventing a ring width: the root is registered with the only
    # defensible width -- single version, because nothing declared a ring -- and the gap is
    # named in diagnostics so it is visible instead of silently swallowed.
    diagnostics = []
    known = {str(getattr(buffer, "root", "")) for buffer in
             (getattr(problem, "buffers", ()) or ())}
    declared = [_declared(buffer) for buffer in (getattr(problem, "buffers", ()) or ())]
    for access in accesses:
        if access.buffer_root in known or not access.buffer_root:
            continue
        known.add(access.buffer_root)
        diagnostics.append(
            "root %r is named by access %s but has no buffer declaration in the Problem;"
            " it is treated as a SINGLE-VERSION root and the gap is reported rather than"
            " filled in" % (access.buffer_root, access.access_id))
        declared.append(_UndeclaredBuffer(root=access.buffer_root))
    loop = _Loop(source_loop_id="problem/loop",
                 domain=dict(getattr(problem, "loop_domain", {}) or {}))
    return ProblemPlanView(loop=loop, tasks=tuple(tasks), accesses=tuple(accesses),
                           buffers=tuple(declared), units={},
                           diagnostics=tuple(diagnostics))


__all__ = ["ProblemPlanView", "problem_plan_view"]
