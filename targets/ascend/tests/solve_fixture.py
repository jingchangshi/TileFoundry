"""Test helper: build a REAL solver-result shape for the plan tests.

Before P3-02 the plan builder read a fabricated ``schedule`` dict -- a shape the solver
never produces -- so a "passing" plan test proved nothing about the production path. The
builder now consumes the real ``solve-result/1` fields:

    state, problem_digest, postcheck.verdict,
    assignment.tasks[task_id].start_time = {value, unit},
    assignment.ii = {value, unit},
    assignment.buffers[root] = resolved version count,
    assignment.units = {ii, start_time}

This module converts the old compact ``{task: {...}}`` description into exactly that
shape, so a test keeps its readable fixture while the code under test takes the real
route. Nothing here fabricates a ``schedule`` key; if a test needs one, the plan builder
will (correctly) refuse it.
"""
from __future__ import annotations

TIME_UNIT = "MODEL cycles"
POSTCHECK_PASS = "POSTCHECK_PASS"


def solve_result(schedule, *, problem_digest="", ii=1, buffers=None,
                 state="SAT", postcheck_verdict=POSTCHECK_PASS, reason="test fixture",
                 extra_tasks=None):
    """Build a real solve-result/1 mapping from a compact schedule description.

    ``schedule`` maps task_id -> {"stage": k, "start_time": cycles} (the compact form the
    older tests use). ``start`` is authoritative: stage is derived as start // ii, and a
    supplied stage that disagrees is an error rather than a silent override.
    """
    if state != "SAT":
        return {"state": state, "problem_digest": problem_digest,
                "postcheck": None if postcheck_verdict is None
                else {"verdict": postcheck_verdict},
                "assignment": {}, "reason": reason, "result_digest": "sha256:" + "5" * 64}
    tasks = {}
    for task_id, placed in schedule.items():
        start = int(placed["start_time"])
        if "stage" in placed and int(placed["stage"]) != start // ii:
            raise AssertionError(
                "fixture is inconsistent: stage %r != start//ii %d for %r"
                % (placed["stage"], start // ii, task_id))
        tasks[task_id] = {"start_time": {"value": start, "unit": TIME_UNIT},
                          "stage": None, "stage_source": "NOT_RETURNED_BY_CORE"}
    for task_id, start in (extra_tasks or {}).items():
        tasks[task_id] = {"start_time": {"value": int(start), "unit": TIME_UNIT},
                          "stage": None, "stage_source": "NOT_RETURNED_BY_CORE"}
    return {
        "state": "SAT",
        "problem_digest": problem_digest,
        "reason": reason,
        "postcheck": None if postcheck_verdict is None
        else {"verdict": postcheck_verdict, "limits": {"physical_feasibility": "UNPROVEN"}},
        "assignment": {
            "tasks": tasks,
            "buffers": dict(buffers or {}),
            "ii": {"value": int(ii), "unit": TIME_UNIT},
            "units": {"ii": TIME_UNIT, "start_time": TIME_UNIT},
        },
        "result_digest": "sha256:" + "5" * 64,
    }


def fixed_buffer_versions(problem, defaults=None):
    """Resolved version counts that match the Problem's own fixed/declared versions."""
    versions = dict(defaults or {})
    for buffer in problem.get("buffers", []) or []:
        root = str(buffer.get("root"))
        if root in versions:
            continue
        declared = buffer.get("versions")
        if declared is None:
            declared = buffer.get("max_versions") or 1
        versions[root] = int(declared)
    return versions
