"""LEGACY DIAGNOSTIC / PROVENANCE HELPER -- NOT an authoritative ResolvedPlan.

STATUS: legacy, kept only for the 05-01/05-02 provenance diagnostics. It used to be the
shape R was to grow into, and it still builds a small ``ResolvedPlan`` dataclass of its
own. That name is now AMBIGUOUS and must not be used as the scheduling authority.

R-06 / G31 item 10 -- ONE authoritative plan type:
    tilefoundry_ascend.plan.resolved_plan.ResolvedPlan
is the SINGLE per-kernel ResolvedPlan for this target. It carries the real solver
assignment (stage/phase/order/start_time/iteration_shift), the resolved version counts
with their source, typed byte offsets, the loop/phase domains and the plan identity
digests. Any scheduling, materialization, code generation or verification decision must
be taken from THAT object (``build_resolved_plan`` in plan/resolved_plan.py), never from
the diagnostic object below.

What this module still legitimately provides, and only that:
  * ``build_provenance`` -- TIR statement -> Problem task -> ledger action_ids coverage,
    used to audit a materialized skeleton statement by statement;
  * ``assert_resolved_plan`` -- the 05-01 "N/A stays N/A / fallback is never solved"
    status assertions for that diagnostic view.

What this module must NOT be used for:
  * deciding stages, versions, generations or offsets (they come from the real
    assignment via plan/resolved_plan.py);
  * reporting a solved/pipelined kernel. Its fields are N/A by construction and there is
    no resolution behind them.

The classes below deliberately keep their historical names so the diagnostic tests and
callers keep working; treat every one of them as diagnostic-only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

from tilefoundry.ir.tir.memory import Copy, Fill
from tilefoundry.ir.tir.stmts import Evaluate, ExecutionScope, For, If, LetStmt, MeshScope, Sequential

NA = "N/A_NOT_IN_PROBLEM"
MODEL = "MODEL"
OFFSET_UNKNOWN = "OFFSET_UNKNOWN"

#: Machine-readable provenance/authority label for every object this module produces.
#: G31 item 10: the authoritative per-kernel ResolvedPlan lives in
#: ``tilefoundry_ascend.plan.resolved_plan``; this module is a diagnostic view only.
PLAN_AUTHORITY = "LEGACY_DIAGNOSTIC_ONLY"
PLAN_AUTHORITY_MODULE = "tilefoundry_ascend.plan.resolved_plan"


@dataclass(frozen=True)
class ResolvedTask:
    stable_id: str
    action_ids: list
    pipe: str
    resource: str
    latency: Optional[int]
    issue_interval: Optional[int]
    cost_model: str = MODEL
    stage: str = NA
    version: str = NA
    generation: str = NA


@dataclass(frozen=True)
class ResolvedAccess:
    task: str
    action_id: str
    root: str
    space: str
    mode: str
    byte_offset: Optional[int]
    offset_status: str


@dataclass(frozen=True)
class ResolvedPlan:
    """LEGACY DIAGNOSTIC VIEW -- not the scheduling authority (see module docstring).

    For the per-kernel plan R/S/X consume use
    ``tilefoundry_ascend.plan.resolved_plan.ResolvedPlan``. This record exists so the
    05-01 schema/provenance diagnostics keep their stable shape; its stage/version/
    generation fields are ``N/A_NOT_IN_PROBLEM`` by construction because it has no
    solver assignment behind it.
    """

    problem_digest: str
    source_digest: str
    ledger_sha: str
    status: str
    num_stages: str
    x_versions: str
    y_versions: str
    tasks: list
    accesses: list
    buffers: list
    unsupported: list
    units: dict
    provenance_digest: str = ""
    #: Always ``PLAN_AUTHORITY``: a machine-readable reminder that this record is a
    #: diagnostic view, and that the authoritative plan is
    #: ``tilefoundry_ascend.plan.resolved_plan.ResolvedPlan`` (G31 item 10).
    plan_authority: str = PLAN_AUTHORITY

    def as_dict(self):
        return asdict(self)


def build_resolved_plan(problem, solver_result=None) -> ResolvedPlan:
    """LEGACY DIAGNOSTIC: Problem -> diagnostic outline. N/A stays N/A.

    This is NOT the ResolveSchedule product. It performs no resolution: it copies the
    Problem's ids and marks every resolved field N/A. ``solver_result`` is only consulted
    for its coarse ``status`` string so that a fallback stays labelled a fallback.
    Use ``tilefoundry_ascend.plan.resolved_plan.build_resolved_plan`` for the single
    authoritative per-kernel plan (G31 item 10).
    """
    status = "PLANNING_ONLY_NO_SOLVER"
    if solver_result is not None:
        result_status = str(getattr(solver_result, "status", "unknown"))
        if result_status == "fallback":
            status = "fallback"           # never 'solved'
        elif result_status == "solved":
            status = "solved"
        else:
            status = result_status or "unknown"
    tasks = [
        ResolvedTask(
            stable_id=task.stable_id,
            action_ids=list(task.action_ids),
            pipe=task.pipe,
            resource=task.resource,
            latency=task.latency,
            issue_interval=task.issue_interval,
        )
        for task in problem.tasks
    ]
    accesses = [
        ResolvedAccess(
            task=access.task,
            action_id=access.task,
            root=access.root,
            space=access.space,
            mode=access.mode,
            byte_offset=access.byte_offset,
            offset_status=access.offset_status,
        )
        for access in problem.accesses
    ]
    unsupported = [
        "plan_authority: " + PLAN_AUTHORITY + " (see " + PLAN_AUTHORITY_MODULE + ")",
        "num_stages: " + NA,
        "x_versions: " + NA,
        "y_versions: " + NA,
        "L2 read/write policy: " + NA,
        "enable_offset: " + NA,
        "byte_offset: " + OFFSET_UNKNOWN + " when the TIR window is symbolic (never 0)",
    ]
    return ResolvedPlan(
        problem_digest=problem.problem_digest,
        source_digest=problem.source_digest,
        ledger_sha=problem.ledger_sha,
        status=status,
        num_stages=NA,
        x_versions=NA,
        y_versions=NA,
        tasks=tasks,
        accesses=accesses,
        buffers=list(problem.buffers),
        unsupported=unsupported,
        units={
            "byte_offset": "bytes", "byte_footprint": "bytes", "distance": "iterations",
            "latency": MODEL + " cycles", "issue_interval": MODEL + " cycles",
        },
    )


def assert_resolved_plan(plan: ResolvedPlan) -> None:
    """Named refusals: fallback is never solved; N/A stages stay N/A; offsets stay UNKNOWN."""
    if plan.status == "solved" and plan.num_stages == NA:
        raise ValueError("FALLBACK_AS_SOLVED: a solved plan may not claim N/A stages without a solver")
    if plan.status != "solved" and plan.status != "fallback" and plan.status != "PLANNING_ONLY_NO_SOLVER":
        raise ValueError("UNKNOWN_STATUS: " + plan.status)


def _stmt_paths(function):
    """Walk the TIR with the SAME path convention as the ledger's instruction ids."""
    found = {}

    def walk(stmt, path):
        if isinstance(stmt, Sequential):
            for index, child in enumerate(stmt.body):
                walk(child, path + (f"s{index}",))
            return
        if isinstance(stmt, For):
            name = str(getattr(stmt.induction_var, "name", "i"))
            walk(stmt.body, path + (f"for:{name}",))
            return
        if isinstance(stmt, (MeshScope, ExecutionScope)):
            walk(stmt.body, path + (type(stmt).__name__,))
            return
        if isinstance(stmt, If):
            walk(stmt.then_body, path + ("then",))
            walk(stmt.else_body, path + ("else",))
            return
        if isinstance(stmt, Evaluate):
            found["/".join(path)] = stmt
            return
        if isinstance(stmt, LetStmt):
            found["/".join(path)] = stmt

    walk(function.body, ())
    return found


def build_provenance(problem, function) -> dict:
    """TIR statement -> Problem task -> ledger action_ids, with full coverage checks."""
    ledger_by_instruction = {}
    for task in problem.tasks:
        for action_id in task.action_ids:
            ledger_by_instruction.setdefault(task.stable_id, []).append(action_id)
    statements = _stmt_paths(function)
    function_name = str(getattr(function, "name", "function"))
    rows = []
    covered = set()
    for path, stmt in statements.items():
        instruction = "/".join((function_name, path)) if path else function_name
        action_ids = ledger_by_instruction.get(instruction, [])
        kind = type(stmt).__name__
        if isinstance(stmt, Evaluate):
            callable_ = getattr(stmt, "callable", None)
            target = getattr(callable_, "target", callable_)
            op = type(target).__name__
        else:
            op = kind
        covered.update(action_ids)
        rows.append({"stmt_path": path, "kind": kind, "op": op,
                     "task": instruction if action_ids else NA,
                     "action_ids": action_ids,
                     "problem_digest": problem.problem_digest})
    all_actions = {action for task in problem.tasks for action in task.action_ids}
    missing = sorted(all_actions - covered)
    if missing:
        raise ValueError(f"PROVENANCE_MISSING: {len(missing)} action(s) not mapped to any TIR statement")
    return {"function": function_name, "rows": rows, "action_coverage": len(covered),
            "action_total": len(all_actions), "complete": True}
