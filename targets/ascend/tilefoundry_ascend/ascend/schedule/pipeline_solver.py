"""Ascend950PR Z3 modulo pipeline constraint projection and solver."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from tilefoundry.vendor.tilelang_z3 import (
    SAT,
    Assignment,
    Buffer,
    Capacity,
    Edge,
    Problem,
    Resource,
    Reuse,
    Task,
    lower_bound_ii,
    solve,
    verify_assignment,
)
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, NodeResourceMask

_CORE_ALIASES: dict[str, str] = {
    "aic": "aic",
    "aic0": "aic",
    "aiv": "aiv0",
    "aiv0": "aiv0",
    "aiv1": "aiv1",
}

_PIPE_TO_LEDGER: dict[str, str] = {
    "MTE1": "PIPE_MTE1",
    "MTE2": "PIPE_MTE2",
    "MTE3": "PIPE_MTE3",
    "CUBE": "PIPE_M",
    "M": "PIPE_M",
    "VECTOR": "PIPE_V",
    "V": "PIPE_V",
    "FIXPIPE": "PIPE_FIX",
    "FIX": "PIPE_FIX",
    "SCALAR": "PIPE_S",
    "S": "PIPE_S",
}


class PipelineError(Exception):
    """Error raised when Ascend modulo pipeline constraints are invalid or unsat."""

    def __init__(self, code: str, message: str = "") -> None:
        self.code = str(code)
        self.message = str(message)
        super().__init__(f"{self.code}: {self.message}" if self.message else self.code)


def read_from_sil(path: str | Path) -> dict | None:
    """Read and parse a SIL packet from a file, returning None if missing or invalid."""
    p = Path(path)
    if not p.is_file():
        logging.warning("SIL packet not found at %s", p)
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logging.error("Failed to read/parse SIL packet from %s: %s", p, exc)
        return None
    if not isinstance(payload, dict) or not payload.get("roofline"):
        return None
    return payload


def format_pipe_key(core: str, pipe: str) -> str:
    """Return canonical ledger pipe identifier such as 'AIC.PIPE_M' or 'AIV0.PIPE_V'."""
    norm_core = _CORE_ALIASES.get(core.strip().lower(), core.strip().lower()).upper()
    norm_pipe = pipe.strip().upper()
    if norm_pipe.startswith("PIPE_"):
        norm_pipe = norm_pipe[5:]
    ledger_pipe = _PIPE_TO_LEDGER.get(norm_pipe, f"PIPE_{norm_pipe}")
    return f"{norm_core}.{ledger_pipe}"


def task_resource_mask(task_or_resources: Task | Sequence[tuple]) -> NodeResourceMask:
    """Build a 24-bit NodeResourceMask from a Task or solver resource tuple sequence."""
    resources = (
        task_or_resources.resources
        if isinstance(task_or_resources, Task)
        else task_or_resources
    )
    mask = NodeResourceMask(0)
    for use in resources:
        core_raw = str(use[1]).strip().lower()
        pipe_raw = str(use[2]).strip()
        norm_core = _CORE_ALIASES.get(core_raw, core_raw)
        mask = mask | NodeResourceMask.for_core(norm_core, pipe_raw)
    return mask


def masks_conflict(lhs: NodeResourceMask, rhs: NodeResourceMask) -> bool:
    """Return whether two 24-bit core/pipe resource masks overlap."""
    return (lhs.bits & rhs.bits) != 0


def bitwise_conflict_pairs(tasks: Sequence[Task]) -> tuple[tuple[int, int], ...]:
    """Return sorted index pairs (i, j) whose 24-bit resource masks overlap."""
    masks = [task_resource_mask(task) for task in tasks]
    pairs: list[tuple[int, int]] = []
    for i, m_i in enumerate(masks):
        for j in range(i + 1, len(masks)):
            if m_i.conflicts_with(masks[j]):
                pairs.append((i, j))
    return tuple(pairs)


def pipe_cycles_from_problem(
    problem: Problem,
    assignment: Assignment | None = None,
) -> dict[str, int]:
    """Accumulate per-pipe cycle budgets from a Problem's 24-bit task masks."""
    del assignment
    cycles: dict[str, int] = {}
    for task in problem.tasks:
        mask = task_resource_mask(task)
        dur = max(1, int(task.latency))
        for core_key, pipe_name in mask.to_core_pipe_pairs():
            key = format_pipe_key(core_key, pipe_name)
            cycles[key] = cycles.get(key, 0) + dur
    return dict(sorted(cycles.items()))


@dataclass(frozen=True)
class AscendPipelineResult:
    """Solved Ascend950PR modulo pipeline assignment alongside the projected Z3 problem."""

    problem: Problem
    assignment: Assignment
    task_stages: dict[str, int]
    buffer_versions: dict[str, int]
    fallback_used: bool = False
    fallback_reason: str = ""
    applied_sil_policy: dict[str, object] | None = None

    @property
    def task_masks(self) -> dict[str, NodeResourceMask]:
        """Return the 24-bit NodeResourceMask per task name."""
        return {task.name: task_resource_mask(task) for task in self.problem.tasks}


def _capacities_from_facts(
    facts: AscendSpaceFacts,
    *,
    has_simtvf: bool = False,
    unreserved_ubuf: bool = False,
) -> tuple[Capacity, ...]:
    if has_simtvf:
        ub_bytes = facts.ub_limit_bytes("simtvf")
    else:
        ub_bytes = facts.ub_limit_bytes("simdvf", unreserved_ubuf=unreserved_ubuf)
    return (
        Capacity("UB", "aiv", ub_bytes),
        Capacity("L1", "aic", facts.l1_bytes),
        Capacity("L0A", "aic", facts.l0a_bytes),
        Capacity("L0B", "aic", facts.l0b_bytes),
        Capacity("L0C", "aic", facts.l0c_bytes),
    )


def _resources_for_tasks(tasks: list[Task]) -> tuple[Resource, ...]:
    seen = {
        (owner, core, pipe)
        for task in tasks
        for owner, core, pipe, _dur, _off in task.resources
    }
    return tuple(Resource(owner, core, pipe, 1) for owner, core, pipe in sorted(seen))


def _recurrence_cycle_pairs(names: list[str], edges: list[Edge]) -> set[tuple[str, str]]:
    """Return ``(u, v)`` pairs sharing a loop-carried recurrence cycle."""
    fwd: dict[str, set[str]] = {n: set() for n in names}
    rev: dict[str, set[str]] = {n: set() for n in names}
    for e in edges:
        if e.distance == 0:
            fwd.setdefault(e.src, set()).add(e.dst)
            rev.setdefault(e.dst, set()).add(e.src)

    def _reach(start: str, adj: dict[str, set[str]]) -> set[str]:
        seen = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            for nxt in adj.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    same_cycle: set[tuple[str, str]] = set()
    for e in edges:
        if e.distance < 1:
            continue
        from_dst = _reach(e.dst, fwd)
        to_src = _reach(e.src, rev)
        cycle_nodes = from_dst & to_src
        for u in cycle_nodes:
            for v in cycle_nodes:
                same_cycle.add((u, v))
    return same_cycle


def _assign_dag_stages(
    names: list[str],
    edges: list[Edge],
    authored_stages: dict[str, int | None],
    max_stage: int,
) -> dict[str, int]:
    stages: dict[str, int] = {}
    for name in names:
        authored = authored_stages.get(name)
        stages[name] = authored if authored is not None else 0
    if any(stage is not None for stage in authored_stages.values()):
        return stages
    cycle_pairs = _recurrence_cycle_pairs(names, edges)
    for _ in names:
        updated = False
        for edge in edges:
            if edge.distance == 0:
                step = 0 if (edge.src, edge.dst) in cycle_pairs else 1
                candidate = min(max_stage, stages[edge.src] + step)
                if candidate > stages[edge.dst]:
                    stages[edge.dst] = candidate
                    updated = True
        if not updated:
            break
    return stages


def _verify_bitwise_mask_schedule(problem: Problem, assignment: Assignment) -> list[str]:
    """Verify no two tasks with overlapping 24-bit resource masks share a modulo slot."""
    problems: list[str] = []
    ii = assignment.ii
    if ii < 1:
        return [f"INVALID_II {ii}"]
    for i, j in bitwise_conflict_pairs(problem.tasks):
        t_i = problem.tasks[i]
        t_j = problem.tasks[j]
        s_i = assignment.starts.get(t_i.name)
        s_j = assignment.starts.get(t_j.name)
        if s_i is None or s_j is None:
            continue
        if (s_i - s_j) % ii == 0:
            problems.append(f"BITWISE_MASK_OVERLAP {t_i.name}/{t_j.name} at II={ii}")
    return problems


def _coerce_sil_policy(sil_policy: Mapping[str, object] | object | None) -> dict[str, object] | None:
    """Normalize a SIL SchedulePolicy object or Mapping into a plain dictionary."""
    if sil_policy is None:
        return None
    if isinstance(sil_policy, Mapping):
        return dict(sil_policy)
    to_dict = getattr(sil_policy, "to_dict", None)
    if callable(to_dict):
        res = to_dict()
        if isinstance(res, Mapping):
            return dict(res)
    keys = (
        "regime_tag",
        "topology_axes",
        "tile_mnk",
        "l1_stages",
        "l1_half_bank_split",
        "l1_slot_aliasing",
        "l0c_buffers",
        "ub_256kb_unlock",
        "vector_mode",
    )
    out: dict[str, object] = {}
    for key in keys:
        if hasattr(sil_policy, key):
            out[key] = getattr(sil_policy, key)
    return out


def _write_shadow_dump(shadow_dump_path: str | Path | None, payload: dict[str, object]) -> None:
    if shadow_dump_path is None:
        return
    path = Path(shadow_dump_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_aliased_l1_name(query: str, l1_names: set[str]) -> str | None:
    if query in l1_names:
        return query
    q_lower = query.lower()
    exact_ci = [n for n in l1_names if n.lower() == q_lower]
    if len(exact_ci) == 1:
        return exact_ci[0]
    suffix = [n for n in l1_names if n.lower().endswith(f"_{q_lower}") or q_lower in n.lower()]
    if len(suffix) == 1:
        return suffix[0]
    return None


def _apply_sil_policy_to_problem(problem: Problem, policy: Mapping[str, object]) -> Problem:
    """Project SIL policy overrides (L1 stages, L0C buffers, 256KB UB, L1 aliasing) onto Problem."""
    l1_stages = policy.get("l1_stages")
    l0c_buffers = policy.get("l0c_buffers")
    ub_256kb_unlock = bool(policy.get("ub_256kb_unlock", False))
    raw_aliasing = policy.get("l1_slot_aliasing") or {}

    if l1_stages is not None:
        if not isinstance(l1_stages, int) or isinstance(l1_stages, bool) or l1_stages < 1:
            raise PipelineError("ASCEND_SIL_POLICY_INVALID", f"invalid l1_stages={l1_stages!r}")
    if l0c_buffers is not None:
        if not isinstance(l0c_buffers, int) or isinstance(l0c_buffers, bool) or l0c_buffers < 1:
            raise PipelineError("ASCEND_SIL_POLICY_INVALID", f"invalid l0c_buffers={l0c_buffers!r}")

    new_buffers: list[Buffer] = []
    for buf in problem.buffers:
        if buf.space == "L1" and l1_stages is not None:
            new_buffers.append(
                Buffer(
                    name=buf.name,
                    slot_bytes=buf.slot_bytes,
                    space=buf.space,
                    owner=buf.owner,
                    fixed=int(l1_stages),
                )
            )
        elif buf.space == "L0C" and l0c_buffers is not None:
            new_buffers.append(
                Buffer(
                    name=buf.name,
                    slot_bytes=buf.slot_bytes,
                    space=buf.space,
                    owner=buf.owner,
                    fixed=int(l0c_buffers),
                )
            )
        else:
            new_buffers.append(buf)

    buf_by_name = {b.name: b for b in new_buffers}
    l1_names = {b.name for b in new_buffers if b.space == "L1"}
    task_index = {t.name: idx for idx, t in enumerate(problem.tasks)}

    writers: dict[str, set[str]] = {}
    readers: dict[str, set[str]] = {}
    for reuse in problem.reuses:
        writers.setdefault(reuse.buffer, set()).add(reuse.next_writer)
        readers.setdefault(reuse.buffer, set()).add(reuse.last_reader)

    new_edges = list(problem.edges)
    edge_keys = {(e.src, e.dst, e.distance, e.latency) for e in new_edges}
    new_reuses = list(problem.reuses)
    reuse_keys = {(r.buffer, r.last_reader, r.next_writer, r.delta) for r in new_reuses}

    if isinstance(raw_aliasing, Mapping):
        for raw_a, raw_b in raw_aliasing.items():
            name_a = _resolve_aliased_l1_name(str(raw_a), l1_names)
            name_b = _resolve_aliased_l1_name(str(raw_b), l1_names)
            if not name_a or not name_b or name_a == name_b:
                continue
            order_a = min((task_index.get(w, 10**9) for w in writers.get(name_a, ())), default=10**9)
            order_b = min((task_index.get(w, 10**9) for w in writers.get(name_b, ())), default=10**9)
            if order_b < order_a:
                first_name, second_name = name_b, name_a
            else:
                first_name, second_name = name_a, name_b
            first_buf = buf_by_name[first_name]
            second_buf = buf_by_name[second_name]
            merged_bytes = max(first_buf.slot_bytes, second_buf.slot_bytes)
            buf_by_name[first_name] = replace(first_buf, slot_bytes=merged_bytes)
            buf_by_name[second_name] = replace(second_buf, slot_bytes=0)

            for r_task in sorted(readers.get(first_name, ())):
                for w_task in sorted(writers.get(second_name, ())):
                    if r_task != w_task:
                        ek = (r_task, w_task, 0, 1)
                        if ek not in edge_keys:
                            edge_keys.add(ek)
                            new_edges.append(Edge(src=r_task, dst=w_task, distance=0, latency=1))
            for r_task in sorted(readers.get(second_name, ())):
                for w_task in sorted(writers.get(first_name, ())):
                    if r_task != w_task:
                        rk = (first_name, r_task, w_task, 0)
                        if rk not in reuse_keys:
                            reuse_keys.add(rk)
                            new_reuses.append(
                                Reuse(
                                    buffer=first_name,
                                    last_reader=r_task,
                                    next_writer=w_task,
                                    delta=0,
                                    read_latency=1,
                                )
                            )

    ordered_buffers = tuple(buf_by_name[b.name] for b in new_buffers)
    new_capacities = []
    for cap in problem.capacities:
        if ub_256kb_unlock and cap.space == "UB" and cap.owner == "aiv" and cap.bytes < 256 * 1024:
            new_capacities.append(Capacity(space=cap.space, owner=cap.owner, bytes=256 * 1024))
        else:
            new_capacities.append(cap)

    return Problem(
        tasks=problem.tasks,
        edges=tuple(new_edges),
        buffers=ordered_buffers,
        reuses=tuple(new_reuses),
        resources=problem.resources,
        capacities=tuple(new_capacities),
        max_stage=problem.max_stage,
        manual=problem.manual,
        issue_after=problem.issue_after,
    )


def _solve_problem_core(
    problem: Problem,
    *,
    fixed_ii: int | None = None,
    max_ii: int = 64,
    timeout_ms: int = 1000,
) -> Assignment:
    lb = lower_bound_ii(problem)
    if fixed_ii is not None:
        if fixed_ii < lb:
            raise PipelineError(
                "ASCEND_PIPELINE_UNSAT",
                f"fixed_ii={fixed_ii} is below resource lower bound {lb}",
            )
        candidates = (fixed_ii,)
    else:
        seq_lat = sum(t.latency for t in problem.tasks) + 4
        upper = max(lb + max_ii, seq_lat)
        candidates = range(lb, upper + 1)

    last_detail = "UNSAT"
    unknown_streak = 0
    for ii in candidates:
        status, assignment, detail = solve(problem, ii, timeout_ms=timeout_ms)
        last_detail = detail
        if status == SAT and assignment is not None:
            problems = verify_assignment(problem, assignment)
            problems.extend(_verify_bitwise_mask_schedule(problem, assignment))
            if problems:
                raise PipelineError(
                    "ASCEND_PIPELINE_VERIFY_FAILED",
                    "; ".join(problems),
                )
            return assignment
        if status == "UNSAT":
            unknown_streak = 0
        else:
            unknown_streak += 1
            if unknown_streak >= 2:
                break
    raise PipelineError(
        "ASCEND_PIPELINE_UNSAT",
        f"no feasible Ascend modulo schedule found ({last_detail})",
    )


def solve_ascend_pipeline(
    problem: Problem,
    *,
    fixed_ii: int | None = None,
    max_ii: int = 64,
    timeout_ms: int = 1000,
    sil_policy: Mapping[str, object] | object | None = None,
    shadow_dump_path: str | Path | None = None,
) -> AscendPipelineResult:
    """Solve for the minimum feasible initiation interval II and verify the model."""
    policy_dict = _coerce_sil_policy(sil_policy)
    solved_problem = problem
    fallback_used = False
    fallback_reason = ""

    if policy_dict is not None:
        try:
            policy_problem = _apply_sil_policy_to_problem(problem, policy_dict)
            assignment = _solve_problem_core(
                policy_problem,
                fixed_ii=fixed_ii,
                max_ii=max_ii,
                timeout_ms=timeout_ms,
            )
            solved_problem = policy_problem
        except PipelineError as exc:
            fallback_used = True
            fallback_reason = str(exc)
            assignment = _solve_problem_core(
                problem,
                fixed_ii=fixed_ii,
                max_ii=max_ii,
                timeout_ms=timeout_ms,
            )
            solved_problem = problem
    else:
        assignment = _solve_problem_core(
            problem,
            fixed_ii=fixed_ii,
            max_ii=max_ii,
            timeout_ms=timeout_ms,
        )

    if shadow_dump_path is not None:
        _write_shadow_dump(
            shadow_dump_path,
            {
                "kind": "solved_pipeline",
                "ii": assignment.ii,
                "starts": assignment.starts,
                "stages": assignment.stages,
                "versions": assignment.versions,
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
                "sil_policy": policy_dict,
                "tasks": [
                    {"name": t.name, "latency": t.latency, "resources": str(t.resources)}
                    for t in solved_problem.tasks
                ],
                "buffers": [
                    {"name": b.name, "space": b.space, "slot_bytes": b.slot_bytes}
                    for b in solved_problem.buffers
                ],
            },
        )

    return AscendPipelineResult(
        problem=solved_problem,
        assignment=assignment,
        task_stages=dict(assignment.stages),
        buffer_versions=dict(assignment.versions),
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        applied_sil_policy=policy_dict,
    )


def solve_pipeline_with_monotone_backoff(
    problem: Problem,
    *,
    fixed_ii: int | None = None,
    max_ii: int = 64,
    timeout_ms: int = 1000,
    sil_policy: Mapping[str, object] | object | None = None,
    shadow_dump_path: str | Path | None = None,
    max_retries: int = 4,
) -> AscendPipelineResult:
    """Solve Z3 modulo schedule with monotone tile-byte backoff before raising PipelineError."""
    cur_problem = problem
    for step in range(max(0, int(max_retries)) + 1):
        try:
            return solve_ascend_pipeline(
                cur_problem,
                fixed_ii=fixed_ii,
                max_ii=max_ii,
                timeout_ms=timeout_ms,
                sil_policy=sil_policy,
                shadow_dump_path=shadow_dump_path,
            )
        except PipelineError as exc:
            if step >= max_retries:
                raise exc
            shrunk_buffers: list[Buffer] = []
            changed = False
            for buf in cur_problem.buffers:
                if buf.slot_bytes > 32:
                    new_bytes = max(32, ((buf.slot_bytes // 2) // 32) * 32)
                    if new_bytes < buf.slot_bytes:
                        changed = True
                        shrunk_buffers.append(replace(buf, slot_bytes=new_bytes))
                        continue
                shrunk_buffers.append(buf)
            if not changed:
                single_ver_buffers: list[Buffer] = []
                for buf in shrunk_buffers:
                    if buf.fixed is None and (buf.auto_max is None or buf.auto_max > 1):
                        changed = True
                        single_ver_buffers.append(replace(buf, fixed=1, auto_min=1, auto_max=1))
                    else:
                        single_ver_buffers.append(buf)
                shrunk_buffers = single_ver_buffers
            if not changed:
                raise exc
            cur_problem = replace(cur_problem, buffers=tuple(shrunk_buffers))
    raise PipelineError("ASCEND_PIPELINE_UNSAT", "monotone backoff exhausted")


__all__ = [
    "AscendPipelineResult",
    "PipelineError",
    "bitwise_conflict_pairs",
    "format_pipe_key",
    "masks_conflict",
    "pipe_cycles_from_problem",
    "read_from_sil",
    "solve_ascend_pipeline",
    "solve_pipeline_with_monotone_backoff",
    "task_resource_mask",
]
