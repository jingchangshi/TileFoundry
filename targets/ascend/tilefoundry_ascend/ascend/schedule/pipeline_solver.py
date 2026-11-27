"""Ascend950PR Z3 modulo pipeline constraint projection and solver."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from tilefoundry.vendor.tilelang_z3 import (
    SAT, Assignment, Buffer, Capacity, Edge, Problem, Resource, Reuse, Task,
    lower_bound_ii, solve, verify_assignment,
)
from tilefoundry_ascend.ascend.json_record import atomic_write_text
from tilefoundry_ascend.ascend.sil.schema import DTYPE_BYTES
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, NodeResourceMask

_CORE_ALIASES = {"aic": "aic", "aic0": "aic", "aiv": "aiv0", "aiv0": "aiv0", "aiv1": "aiv1"}
_PIPE_TO_LEDGER = {
    "MTE1": "PIPE_MTE1", "MTE2": "PIPE_MTE2", "MTE3": "PIPE_MTE3", "CUBE": "PIPE_M",
    "M": "PIPE_M", "VECTOR": "PIPE_V", "V": "PIPE_V", "FIXPIPE": "PIPE_FIX",
    "FIX": "PIPE_FIX", "SCALAR": "PIPE_S", "S": "PIPE_S",
}
_SIL_POLICY_KEYS = (
    "regime_tag", "topology_axes", "tile_mnk", "l1_stages", "l1_half_bank_split",
    "l1_slot_aliasing", "l0c_buffers", "ub_256kb_unlock", "vector_mode",
)


class PipelineError(Exception):
    """Error raised when Ascend modulo pipeline constraints are invalid or unsat."""

    def __init__(self, code: str, message: str = "") -> None:
        self.code, self.message = str(code), str(message)
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
    return payload if isinstance(payload, dict) and payload.get("roofline") else None


def format_pipe_key(core: str, pipe: str) -> str:
    """Return canonical ledger pipe identifier such as 'AIC.PIPE_M' or 'AIV0.PIPE_V'."""
    norm_core = _CORE_ALIASES.get(core.strip().lower(), core.strip().lower()).upper()
    norm_pipe = pipe.strip().upper().removeprefix("PIPE_")
    return f"{norm_core}.{_PIPE_TO_LEDGER.get(norm_pipe, f'PIPE_{norm_pipe}')}"


def task_resource_mask(task_or_resources: Task | Sequence[tuple]) -> NodeResourceMask:
    """Build a 24-bit NodeResourceMask from a Task or solver resource tuple sequence."""
    resources = task_or_resources.resources if isinstance(task_or_resources, Task) else task_or_resources
    mask = NodeResourceMask(0)
    for use in resources:
        core_raw = str(use[1]).strip().lower()
        mask |= NodeResourceMask.for_core(_CORE_ALIASES.get(core_raw, core_raw), str(use[2]).strip())
    return mask


def masks_conflict(lhs: NodeResourceMask, rhs: NodeResourceMask) -> bool:
    """Return whether two 24-bit core/pipe resource masks overlap."""
    return (lhs.bits & rhs.bits) != 0


def bitwise_conflict_pairs(tasks: Sequence[Task]) -> tuple[tuple[int, int], ...]:
    """Return sorted index pairs (i, j) whose 24-bit resource masks overlap."""
    masks = [task_resource_mask(t) for t in tasks]
    return tuple((i, j) for i, m_i in enumerate(masks) for j in range(i + 1, len(masks)) if m_i.conflicts_with(masks[j]))


def pipe_cycles_from_problem(problem: Problem, assignment: Assignment | None = None) -> dict[str, int]:
    """Accumulate per-pipe cycle budgets from a Problem's 24-bit task masks."""
    del assignment
    cycles: dict[str, int] = {}
    for task in problem.tasks:
        dur = max(1, int(task.latency))
        for core_key, pipe_name in task_resource_mask(task).to_core_pipe_pairs():
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
    facts: AscendSpaceFacts, *, has_simtvf: bool = False, unreserved_ubuf: bool = False,
) -> tuple[Capacity, ...]:
    ub_bytes = facts.ub_limit_bytes("simtvf") if has_simtvf else facts.ub_limit_bytes("simdvf", unreserved_ubuf=unreserved_ubuf)
    return (
        Capacity("UB", "aiv", ub_bytes), Capacity("L1", "aic", facts.l1_bytes),
        Capacity("L0A", "aic", facts.l0a_bytes), Capacity("L0B", "aic", facts.l0b_bytes),
        Capacity("L0C", "aic", facts.l0c_bytes),
    )


def _resources_for_tasks(tasks: list[Task]) -> tuple[Resource, ...]:
    seen = {(owner, core, pipe) for t in tasks for owner, core, pipe, _d, _o in t.resources}
    return tuple(Resource(owner, core, pipe, 1) for owner, core, pipe in sorted(seen))


def _recurrence_cycle_pairs(names: list[str], edges: list[Edge]) -> set[tuple[str, str]]:
    """Return ``(u, v)`` pairs sharing a loop-carried recurrence cycle."""
    fwd, rev = {n: set() for n in names}, {n: set() for n in names}
    for e in edges:
        if e.distance == 0:
            fwd.setdefault(e.src, set()).add(e.dst)
            rev.setdefault(e.dst, set()).add(e.src)

    def _reach(start: str, adj: dict[str, set[str]]) -> set[str]:
        seen, stack = {start}, [start]
        while stack:
            for nxt in adj.get(stack.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    same_cycle: set[tuple[str, str]] = set()
    for e in edges:
        if e.distance >= 1:
            nodes = _reach(e.dst, fwd) & _reach(e.src, rev)
            same_cycle.update((u, v) for u in nodes for v in nodes)
    return same_cycle


def _assign_dag_stages(
    names: list[str], edges: list[Edge], authored_stages: dict[str, int | None], max_stage: int,
) -> dict[str, int]:
    stages = {name: (authored_stages[name] if authored_stages.get(name) is not None else 0) for name in names}
    if any(stage is not None for stage in authored_stages.values()):
        return stages
    cycle_pairs = _recurrence_cycle_pairs(names, edges)
    for _ in names:
        updated = False
        for e in edges:
            if e.distance == 0:
                cand = min(max_stage, stages[e.src] + (0 if (e.src, e.dst) in cycle_pairs else 1))
                if cand > stages[e.dst]:
                    stages[e.dst], updated = cand, True
        if not updated:
            break
    return stages


def _verify_bitwise_mask_schedule(problem: Problem, assignment: Assignment) -> list[str]:
    if (ii := assignment.ii) < 1:
        return [f"INVALID_II {ii}"]
    problems: list[str] = []
    for i, j in bitwise_conflict_pairs(problem.tasks):
        t_i, t_j = problem.tasks[i], problem.tasks[j]
        s_i, s_j = assignment.starts.get(t_i.name), assignment.starts.get(t_j.name)
        if s_i is not None and s_j is not None and (s_i - s_j) % ii == 0:
            problems.append(f"BITWISE_MASK_OVERLAP {t_i.name}/{t_j.name} at II={ii}")
    return problems


def _coerce_sil_policy(sil_policy: Mapping[str, object] | object | None) -> dict[str, object] | None:
    if sil_policy is None:
        return None
    if isinstance(sil_policy, Mapping):
        return dict(sil_policy)
    to_dict = getattr(sil_policy, "to_dict", None)
    if callable(to_dict) and isinstance(res := to_dict(), Mapping):
        return dict(res)
    return {k: getattr(sil_policy, k) for k in _SIL_POLICY_KEYS if hasattr(sil_policy, k)}


def _resolve_aliased_l1_name(query: str, l1_names: set[str]) -> str | None:
    if query in l1_names:
        return query
    q_low = query.lower()
    for matches in ([n for n in l1_names if n.lower() == q_low], [n for n in l1_names if n.lower().endswith(f"_{q_low}") or q_low in n.lower()]):
        if len(matches) == 1:
            return matches[0]
    return None


def _apply_sil_policy_to_problem(problem: Problem, policy: Mapping[str, object]) -> Problem:
    """Project SIL policy overrides (L1 stages, L0C buffers, 256KB UB, L1 aliasing) onto Problem."""
    l1_stages, l0c_buffers = policy.get("l1_stages"), policy.get("l0c_buffers")
    ub_256kb_unlock, raw_aliasing = bool(policy.get("ub_256kb_unlock", False)), policy.get("l1_slot_aliasing") or {}

    for label, val in (("l1_stages", l1_stages), ("l0c_buffers", l0c_buffers)):
        if val is not None and (not isinstance(val, int) or isinstance(val, bool) or val < 1):
            raise PipelineError("ASCEND_SIL_POLICY_INVALID", f"invalid {label}={val!r}")

    new_buffers = [
        replace(buf, fixed=int(l1_stages), auto_min=None, auto_max=None) if buf.space == "L1" and l1_stages is not None
        else (replace(buf, fixed=int(l0c_buffers), auto_min=None, auto_max=None) if buf.space == "L0C" and l0c_buffers is not None else buf)
        for buf in problem.buffers
    ]
    buf_by_name = {b.name: b for b in new_buffers}
    l1_names = {b.name for b in new_buffers if b.space == "L1"}
    task_index = {t.name: idx for idx, t in enumerate(problem.tasks)}

    writers, readers = {}, {}
    for r in problem.reuses:
        writers.setdefault(r.buffer, set()).add(r.next_writer)
        readers.setdefault(r.buffer, set()).add(r.last_reader)

    new_edges, new_reuses = list(problem.edges), list(problem.reuses)
    edge_keys = {(e.src, e.dst, e.distance, e.latency) for e in new_edges}
    reuse_keys = {(r.buffer, r.last_reader, r.next_writer, r.delta) for r in new_reuses}

    if isinstance(raw_aliasing, Mapping):
        for raw_a, raw_b in raw_aliasing.items():
            name_a, name_b = _resolve_aliased_l1_name(str(raw_a), l1_names), _resolve_aliased_l1_name(str(raw_b), l1_names)
            if not name_a or not name_b or name_a == name_b:
                continue
            ord_a = min((task_index.get(w, 10**9) for w in writers.get(name_a, ())), default=10**9)
            ord_b = min((task_index.get(w, 10**9) for w in writers.get(name_b, ())), default=10**9)
            first_name, second_name = (name_b, name_a) if ord_b < ord_a else (name_a, name_b)
            first_buf, second_buf = buf_by_name[first_name], buf_by_name[second_name]
            buf_by_name[first_name] = replace(first_buf, slot_bytes=max(first_buf.slot_bytes, second_buf.slot_bytes))
            buf_by_name[second_name] = replace(second_buf, slot_bytes=0)

            for r_task in sorted(readers.get(first_name, ())):
                for w_task in sorted(writers.get(second_name, ())):
                    if r_task != w_task and (ek := (r_task, w_task, 0, 1)) not in edge_keys:
                        edge_keys.add(ek)
                        new_edges.append(Edge(src=r_task, dst=w_task, distance=0, latency=1))
            for r_task in sorted(readers.get(second_name, ())):
                for w_task in sorted(writers.get(first_name, ())):
                    if r_task != w_task and (rk := (first_name, r_task, w_task, 0)) not in reuse_keys:
                        reuse_keys.add(rk)
                        new_reuses.append(Reuse(buffer=first_name, last_reader=r_task, next_writer=w_task, delta=0, read_latency=1))

    new_capacities = [
        Capacity(space=cap.space, owner=cap.owner, bytes=256 * 1024)
        if ub_256kb_unlock and cap.space == "UB" and cap.owner == "aiv" and cap.bytes < 256 * 1024 else cap
        for cap in problem.capacities
    ]
    return replace(
        problem, edges=tuple(new_edges), buffers=tuple(buf_by_name[b.name] for b in new_buffers),
        reuses=tuple(new_reuses), capacities=tuple(new_capacities),
    )


def _solve_problem_core(problem: Problem, *, fixed_ii: int | None = None, max_ii: int = 64, timeout_ms: int = 1000) -> Assignment:
    lb = lower_bound_ii(problem)
    if fixed_ii is not None:
        if fixed_ii < lb:
            raise PipelineError("ASCEND_PIPELINE_UNSAT", f"fixed_ii={fixed_ii} is below resource lower bound {lb}")
        candidates: Sequence[int] = (fixed_ii,)
    else:
        candidates = range(lb, max(lb + max_ii, sum(t.latency for t in problem.tasks) + 4) + 1)

    last_detail, unknown_streak = "UNSAT", 0
    for ii in candidates:
        status, assignment, detail = solve(problem, ii, timeout_ms=timeout_ms)
        last_detail = detail
        if status == SAT and assignment is not None:
            if problems := verify_assignment(problem, assignment) + _verify_bitwise_mask_schedule(problem, assignment):
                raise PipelineError("ASCEND_PIPELINE_VERIFY_FAILED", "; ".join(problems))
            return assignment
        unknown_streak = 0 if status == "UNSAT" else unknown_streak + 1
        if unknown_streak >= 2:
            break
    raise PipelineError("ASCEND_PIPELINE_UNSAT", f"no feasible Ascend modulo schedule found ({last_detail})")


def solve_ascend_pipeline(
    problem: Problem, *, fixed_ii: int | None = None, max_ii: int = 64, timeout_ms: int = 1000,
    sil_policy: Mapping[str, object] | object | None = None, shadow_dump_path: str | Path | None = None,
) -> AscendPipelineResult:
    """Solve for the minimum feasible initiation interval II and verify the model."""
    policy_dict, solved_problem = _coerce_sil_policy(sil_policy), problem
    fallback_used, fallback_reason = False, ""
    solve_kw = {"fixed_ii": fixed_ii, "max_ii": max_ii, "timeout_ms": timeout_ms}

    if policy_dict is not None:
        try:
            policy_problem = _apply_sil_policy_to_problem(problem, policy_dict)
            assignment, solved_problem = _solve_problem_core(policy_problem, **solve_kw), policy_problem
        except PipelineError as exc:
            fallback_used, fallback_reason = True, str(exc)
            assignment = _solve_problem_core(problem, **solve_kw)
    else:
        assignment = _solve_problem_core(problem, **solve_kw)

    if shadow_dump_path is not None:
        payload = {
            "kind": "solved_pipeline", "ii": assignment.ii, "starts": assignment.starts,
            "stages": assignment.stages, "versions": assignment.versions,
            "fallback_used": fallback_used, "fallback_reason": fallback_reason, "sil_policy": policy_dict,
            "tasks": [{"name": t.name, "latency": t.latency, "resources": str(t.resources)} for t in solved_problem.tasks],
            "buffers": [{"name": b.name, "space": b.space, "slot_bytes": b.slot_bytes} for b in solved_problem.buffers],
        }
        atomic_write_text(shadow_dump_path, json.dumps(payload, indent=2, sort_keys=True))

    return AscendPipelineResult(
        problem=solved_problem, assignment=assignment, task_stages=dict(assignment.stages),
        buffer_versions=dict(assignment.versions), fallback_used=fallback_used,
        fallback_reason=fallback_reason, applied_sil_policy=policy_dict,
    )


def solve_pipeline_with_monotone_backoff(
    problem: Problem, *, fixed_ii: int | None = None, max_ii: int = 64, timeout_ms: int = 1000,
    sil_policy: Mapping[str, object] | object | None = None, shadow_dump_path: str | Path | None = None,
    max_retries: int = 4,
) -> AscendPipelineResult:
    """Solve Z3 modulo schedule with monotone tile-byte backoff before raising PipelineError."""
    cur_problem = problem
    for step in range(max(0, int(max_retries)) + 1):
        try:
            return solve_ascend_pipeline(
                cur_problem, fixed_ii=fixed_ii, max_ii=max_ii, timeout_ms=timeout_ms,
                sil_policy=sil_policy, shadow_dump_path=shadow_dump_path,
            )
        except PipelineError as exc:
            if step >= max_retries:
                raise exc
            changed, shrunk = False, []
            for buf in cur_problem.buffers:
                if buf.slot_bytes > 32 and (nb := max(32, ((buf.slot_bytes // 2) // 32) * 32)) < buf.slot_bytes:
                    changed = True
                    shrunk.append(replace(buf, slot_bytes=nb))
                else:
                    shrunk.append(buf)
            if not changed:
                next_bufs = []
                for buf in shrunk:
                    if buf.fixed is None and (buf.auto_max is None or buf.auto_max > 1):
                        changed = True
                        next_bufs.append(replace(buf, fixed=1, auto_min=1, auto_max=1))
                    else:
                        next_bufs.append(buf)
                shrunk = next_bufs
            if not changed:
                raise exc
            cur_problem = replace(cur_problem, buffers=tuple(shrunk))
    raise PipelineError("ASCEND_PIPELINE_UNSAT", "monotone backoff exhausted")


class _DTypeLike(Protocol):
    bits: int
    lanes: int


def dtype_bytes(
    dtype_obj: object, data_type_cls: Callable[[str], _DTypeLike] | None = None
) -> float:
    """Return physical storage bytes per element or vector lane group for a dtype."""
    if data_type_cls is not None:
        dt = data_type_cls(str(dtype_obj))
        return (dt.bits * dt.lanes) / 8.0
    key = str(dtype_obj).strip().lower()
    if key in ("u16", "u32", "u64"):
        return float({"u16": 2, "u32": 4, "u64": 8}[key])
    raw = DTYPE_BYTES.get(key)
    return float(raw) if raw is not None else 2.0


__all__ = [
    "AscendPipelineResult", "PipelineError", "bitwise_conflict_pairs", "dtype_bytes",
    "format_pipe_key", "masks_conflict", "pipe_cycles_from_problem", "read_from_sil",
    "solve_ascend_pipeline", "solve_pipeline_with_monotone_backoff", "task_resource_mask",
]
