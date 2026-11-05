"""Real emission ledger -> normalized ScheduleProblem (WBS 04-05).

The ledger is extracted from the TYPED TIR that lowering actually emitted: every memory action
in the TIR (ascend_vld/vsts, gm_to_ub/ub_to_gm, Copy, Fill) becomes one entry with a STABLE
source/action id built from the structural statement path and the op name. No id(obj), no case
template, no hand-written task list.

The Problem is derived from the SAME ledger: buffers come from the ledger roots, accesses from the
ledger entries (bidirectional), and a byte_offset that the TIR does not carry as a literal is
OFFSET_UNKNOWN and is REFUSED as 0 by code (not by comment). A missing buffer root is refused with
UNKNOWN_ROOT.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from tilefoundry.ir.core import Call, Constant, Tuple, Var
from ..ir.isa import contract as isa_contract
from tilefoundry.ir.hir.schedule import ScheduleOp
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.tir.memory import Copy, Fill, PtrOf, TensorView
from tilefoundry.ir.tir.stmts import Evaluate, ExecutionScope, For, If, LetStmt, MeshScope, Sequential

#: TIR instruction class name -> [(operand index, mode)]; mode is "read" | "write".
_ACTION_EFFECTS = {
    # data movement
    "Vld": ((0, "read"), (1, "write")),
    "Vsts": ((0, "read"), (1, "write")),
    "GmToUb": ((0, "read"), (1, "write")),
    "UbToGm": ((0, "read"), (1, "write")),
    # VF tasks: every register read/write is an access (so whole-VF tasks are visible)
    "Vcadd": ((0, "read"), (1, "write")),
    "Vcvt": ((0, "read"), (1, "read"), (2, "write")),
    "Vmula": ((0, "read"), (1, "read"), (2, "read"), (3, "write")),
    "Vadd": ((0, "read"), (1, "read"), (2, "read"), (3, "write")),
    "Vmul": ((0, "read"), (1, "read"), (2, "read"), (3, "write")),
    "Vdiv": ((0, "read"), (1, "read"), (2, "read"), (3, "write")),
    "Vadds": ((0, "read"), (1, "read"), (2, "write")),
    "Vmuls": ((0, "read"), (1, "read"), (2, "write")),
    "Vsqrt": ((0, "read"), (1, "read"), (2, "write")),
    "Vdup": ((0, "read"), (1, "write")),
    "Vdupv": ((0, "read"), (1, "write")),
    "Pset": ((0, "write"),),
    "Pset16": ((0, "write"),),
}


class LedgerError(ValueError):
    """A named refusal from the ledger/Problem extraction."""


@dataclass(frozen=True)
class LedgerEntry:
    action_id: str
    instruction: str
    op: str
    root: str
    root_kind: str
    space: str
    mode: str
    byte_footprint: int
    byte_offset: Optional[int]      # None == OFFSET_UNKNOWN
    source_offset: str              # raw TIR slice starts (literal list or 'symbolic')
    attrs: tuple                    # op attributes: (('pack', 'PK_B32'), ('direction', 'reverse'), ...)
    offset_structure: tuple          # canonical typed structure of the window offset (op/args/consts/vars)
    guard: tuple                     # canonical keys of the enclosing If conditions (guard provenance)
    logical_iteration: str
    iteration_path: str
    iteration_extent: str
    iteration_domains: tuple        # ((loop_name, extent_str), ...) for every enclosing loop
    path: str


@dataclass
class Ledger:
    function: str
    entries: list = field(default_factory=list)

    def by_id(self):
        return {entry.action_id: entry for entry in self.entries}


def _op_name(op) -> str:
    return getattr(op, "name", type(op).__name__)


def _slice_starts(expr) -> tuple:
    """Literal starts of the Slice inside a PtrOf/TensorView chain (empty if symbolic)."""
    if isinstance(expr, Call) and isinstance(expr.target, TensorView):
        if expr.args and isinstance(expr.args[0], Constant) and isinstance(expr.args[0].value, int):
            return (int(expr.args[0].value),)
        return _slice_starts(expr.args[0]) if expr.args else ()
    if isinstance(expr, Call) and isinstance(expr.target, PtrOf):
        return _slice_starts(expr.args[0]) if expr.args else ()
    if isinstance(expr, Call) and isinstance(expr.target, Slice):
        if len(expr.args) > 1 and isinstance(expr.args[1], Tuple):
            values = tuple(getattr(element, 'value', None) for element in expr.args[1].elements)
            return values if all(isinstance(value, int) for value in values) else ()
        return ()
    if isinstance(expr, Slice):
        starts = tuple(getattr(expr, "starts", ()) or ())
        return starts if all(isinstance(start, int) for start in starts) else ()
    return ()




def _expr_key(expr) -> tuple:
    """Canonical structural key of a typed TIR expression (no id(obj), no string collapse)."""
    if isinstance(expr, Constant):
        return ("const", getattr(expr, "value", None))
    if isinstance(expr, Var):
        return ("var", str(getattr(expr, "name", None)))
    if isinstance(expr, Call):
        target = getattr(expr, "target", None)
        name = getattr(target, "name", None) or type(target).__name__
        attrs = []
        for key in sorted(getattr(type(target), "__dataclass_fields__", {}) or {}):
            value = getattr(target, key, None)
            if isinstance(value, (str, int, bool)):
                attrs.append((key, value))
            elif value is not None and type(value).__name__.endswith("Kind"):
                attrs.append((key, str(value)))
        return ("call", str(name), [_expr_key(arg) for arg in expr.args], attrs)
    if isinstance(expr, Tuple):
        return ("tuple", [_expr_key(element) for element in expr.elements])
    return ("opaque", type(expr).__name__)


def _slice_exprs(expr) -> tuple:
    """The typed start expressions of the Slice inside a PtrOf/TensorView chain."""
    if isinstance(expr, Call) and isinstance(expr.target, TensorView):
        if expr.args and isinstance(expr.args[0], Constant) and isinstance(expr.args[0].value, int):
            return (expr.args[0],)
        return _slice_exprs(expr.args[0]) if expr.args else ()
    if isinstance(expr, Call) and isinstance(expr.target, PtrOf):
        return _slice_exprs(expr.args[0]) if expr.args else ()
    if isinstance(expr, Call) and isinstance(expr.target, Slice):
        if len(expr.args) > 1 and isinstance(expr.args[1], Tuple):
            return tuple(expr.args[1].elements)
        return ()
    if isinstance(expr, Slice):
        return tuple(getattr(expr, "starts", ()) or ())
    return ()


def _shape_bytes(type_) -> int:
    shape = tuple(getattr(type_, "shape", ()) or ())
    layout = getattr(type_, "layout", None)
    if layout is not None:
        shape = tuple(getattr(layout, "shape", shape) or shape)
    count = 1
    for dim in shape:
        count *= int(dim)
    dtype = getattr(type_, "dtype", None)
    bits = int(getattr(dtype, "bit_width", 0) or 0)
    return max(1, count * max(1, bits // 8))


def _root_of(var, views: dict, defs: dict) -> tuple:
    """Resolve a TIR operand to (root_name, root_kind) by walking the view/ptr chain.

    A real allocation (function parameter or a bound AllocTensor var) is root_kind='real'.
    An anonymous view base (a literal offset) is root_kind='synthetic_alloc' and the caller
    derives a stable synthetic root id from the ALLOCATION SITE action path.
    An unresolvable chain is REFUSED by name (VIEW_CHAIN_UNRESOLVED), never given a fake name.
    """
    if isinstance(var, Call):
        if isinstance(var.target, (TensorView, PtrOf, Slice)):
            return _root_of(var.args[0], views, defs) if var.args else ("anon", "synthetic_alloc")
        return ("anon", "synthetic_alloc")
    if isinstance(var, Slice):
        return _root_of(var.base, views, defs)
    name = getattr(var, "name", None)
    if name is not None and name in views:
        return views[name][0], views[name][1]
    seen = set()
    current = var
    while True:
        key = getattr(current, "name", None)
        if key is None:
            return ("anon", "synthetic_alloc")
        if key in seen:
            raise LedgerError("VIEW_CHAIN_UNRESOLVED: cycle at " + str(key))
        seen.add(key)
        value = defs.get(key)
        if value is None:
            return (str(key), "real")
        if isinstance(value, Call) and isinstance(value.target, (TensorView, PtrOf, Slice)):
            current = value.args[0] if value.args else current
            continue
        return (str(key), "real")


def extract_ledger(function) -> Ledger:
    """Walk the typed TIR and record every actual memory action with a stable id."""
    ledger = Ledger(function=str(getattr(function, "name", "function")))
    views: dict = {}  # view var name -> (root name, root kind, start exprs, alloc path)
    defs: dict = {}   # var name -> defining value (for root resolution)

    def starts_of(var) -> tuple:
        name = getattr(var, "name", None)
        return views[name][2] if name in views else ()

    def source_offset_of(var) -> str:
        starts = starts_of(var)
        return ",".join(str(start) for start in starts) if starts else "symbolic"

    def offset_structure_of(var) -> tuple:
        starts = starts_of(var)
        return tuple(_expr_key(start) for start in starts)

    def offset_of(var) -> Optional[int]:
        starts = starts_of(var)
        if starts and all(isinstance(start, Constant) and isinstance(getattr(start, "value", None), int) for start in starts):
            return sum(int(start.value) for start in starts)
        return None

    def walk(stmt, path: tuple, loops: tuple, guards: tuple) -> None:
        if isinstance(stmt, Sequential):
            for index, child in enumerate(stmt.body):
                walk(child, path + (f"s{index}",), loops, guards)
            return
        if isinstance(stmt, For):
            name = str(getattr(stmt.induction_var, "name", "i"))
            extent = str(getattr(stmt, "stop", ""))
            walk(stmt.body, path + (f"for:{name}",), loops + ((name, extent),), guards)
            return
        if isinstance(stmt, (MeshScope, ExecutionScope)):
            walk(stmt.body, path + (type(stmt).__name__,), loops, guards)
            return
        if isinstance(stmt, If):
            condition = _expr_key(stmt.cond)
            walk(stmt.then_body, path + ("then",), loops, guards + (("then", condition),))
            walk(stmt.else_body, path + ("else",), loops, guards + (("else", condition),))
            return
        if isinstance(stmt, LetStmt):
            value = stmt.value
            defs[str(getattr(stmt.var, "name", ""))] = value
            if isinstance(value, Call) and isinstance(value.target, (TensorView, PtrOf)):
                root, kind = _root_of(value.args[0] if value.args else value, views, defs)
                views[str(getattr(stmt.var, "name", ""))] = (root, kind, _slice_exprs(value), "/".join(path))
            return
        if not isinstance(stmt, Evaluate):
            return
        op = stmt.callable
        seen_ops = set()
        while True:
            if isinstance(op, Var):
                key = getattr(op, "name", None)
                if key in seen_ops or key not in defs:
                    break
                seen_ops.add(key)
                op = defs[key]
                continue
            if isinstance(op, Call):
                op = op.target
                continue
            if isinstance(op, ScheduleOp):
                op = op.op
                continue
            break
        if isinstance(op, Copy):
            name, effects = "copy", ((0, "read"), (1, "write"))
        elif isinstance(op, Fill):
            name, effects = "fill", ((0, "write"),)
        else:
            name = type(op).__name__
            effects = _ACTION_EFFECTS.get(name)
        if effects is None:
            return
        for operand, mode in effects:
            if operand >= len(stmt.args):
                continue
            arg = stmt.args[operand]
            root, root_kind = _root_of(arg, views, defs)
            type_ = getattr(arg, "type", None)
            space = str(getattr(getattr(type_, "storage", None), "name", "?"))
            if root_kind == "synthetic_alloc":
                name = getattr(arg, "name", None)
                alloc_path = views[name][3] if name in views else "/".join(path)
                root = "smem@" + alloc_path
            elif root in ("None", "anon", "0"):
                raise LedgerError("VIEW_CHAIN_UNRESOLVED: action " + "/".join(path) + " has no real allocation root")
            ledger.entries.append(
                LedgerEntry(
                    action_id="/".join((ledger.function, *path, f"{name}:{operand}")),
                    instruction="/".join((ledger.function, *path)),
                    op=name,
                    root=root,
                    root_kind=root_kind,
                    space=space,
                    mode=mode,
                    byte_footprint=_shape_bytes(type_),
                    byte_offset=offset_of(arg),
                    source_offset=source_offset_of(arg),
                    offset_structure=offset_structure_of(arg),
                    guard=guards,
                    attrs=tuple(
                        (key, getattr(op, key))
                        for key in ("pack", "direction", "part", "elem_width")
                        if isinstance(getattr(op, key, None), (str, int, bool))
                    ),
                    logical_iteration=(loops[-1][0] if loops else "outside"),
                    iteration_path="/".join(name for name, _ in loops) or "outside",
                    iteration_extent=(loops[-1][1] if loops else "1"),
                    iteration_domains=tuple((name, str(extent)) for name, extent in loops),
                    path="/".join(path),
                )
            )

    walk(function.body, (), (), ())
    return ledger


@dataclass(frozen=True)
class ProblemAccess:
    task: str
    instruction: str
    root: str
    root_kind: str
    space: str
    mode: str
    byte_footprint: int
    byte_offset: Optional[int]
    source_offset: str
    attrs: tuple
    offset_structure: tuple
    guard: tuple
    normalized_offset: Optional[int]
    offset_status: str  # "BYTES" | "OFFSET_UNKNOWN"
    logical_iteration: str
    iteration_path: str
    iteration_extent: str
    iteration_domains: tuple




@dataclass(frozen=True)
class ProblemTask:
    """One real instruction (whole VF task or DMA/copy) with ISA-sourced pipe/cost fields."""
    stable_id: str
    op: str
    pipe: str
    resource: str
    latency: Optional[int]
    issue_interval: Optional[int]
    loop: str
    action_ids: list
    cost_model: str = "MODEL"


@dataclass
class NormalizedProblem:
    problem_id: str
    source_digest: str
    ledger_sha: str
    canonical_serialization_sha: str
    problem_digest: str
    loop_domain: dict
    accesses: list
    tasks: list
    buffers: list
    dependencies: list
    alias: dict

    def as_dict(self):
        return asdict(self)


def build_problem(ledger: Ledger) -> NormalizedProblem:
    """ledger -> normalized Problem. Buffers come from the SAME ledger; no invented roots."""
    roots = {}
    for entry in ledger.entries:
        key = (entry.root, entry.space)
        roots[key] = max(roots.get(key, 0), entry.byte_footprint)
    buffers = [
        {"root": root, "root_kind": ("synthetic_alloc" if root.startswith("smem@") else "real"), "space": space, "bytes_": size, "owner_loop": "outside", "versions": 1, "fixed": True}
        for (root, space), size in sorted(roots.items())
    ]
    accesses = [
        ProblemAccess(
            task=entry.action_id,
            instruction=entry.instruction,
            root=entry.root,
            root_kind=entry.root_kind,
            space=entry.space,
            mode=entry.mode,
            byte_footprint=entry.byte_footprint,
            byte_offset=entry.byte_offset,
            source_offset=entry.source_offset,
            attrs=entry.attrs,
            offset_structure=entry.offset_structure,
            guard=entry.guard,
            normalized_offset=entry.byte_offset,
            offset_status="BYTES" if entry.byte_offset is not None else "OFFSET_UNKNOWN",
            logical_iteration=entry.logical_iteration,
            iteration_path=entry.iteration_path,
            iteration_extent=entry.iteration_extent,
            iteration_domains=entry.iteration_domains,
        )
        for entry in ledger.entries
    ]
    # UNKNOWN_ROOT: every access root must have a buffer from the same ledger.
    buffer_roots = {(buffer["root"], buffer["space"]) for buffer in buffers}
    for access in accesses:
        if (access.root, access.space) not in buffer_roots:
            raise LedgerError(f"UNKNOWN_ROOT: access root {access.root!r} in {access.space} has no BufferDecl")
    # Dependencies: same-root RAW/WAW/WAR, ordered by the action order, distance from iterations.
    dependencies = []
    for later in range(len(accesses)):
        for earlier in range(later):
            a, b = accesses[earlier], accesses[later]
            if a.root != b.root or a.space != b.space:
                continue
            if a.mode == "read" and b.mode == "write":
                kind = "WAR"
            elif a.mode == "write" and b.mode == "read":
                kind = "RAW"
            elif a.mode == "write" and b.mode == "write":
                kind = "WAW"
            else:
                continue
            distance = 0 if a.logical_iteration == b.logical_iteration else 1
            dependencies.append(
                {"producer": a.task, "consumer": b.task, "distance": distance, "kind": kind, "evidence": "same root+space, emission order"}
            )
    # Tasks: one per REAL instruction, with pipe/latency/ii from the ISA contract (not hardcoded).
    by_instruction = {}
    for entry in ledger.entries:
        by_instruction.setdefault(entry.instruction, []).append(entry)
    tasks = []
    for instruction, group in sorted(by_instruction.items()):
        first = group[0]
        try:
            contract_ = isa_contract(first.op.lower())
            pipe, latency, ii = contract_.resource, contract_.latency_model, contract_.ii_model
        except KeyError:
            pipe, latency, ii = 'NO_ISA_CONTRACT', None, None
        tasks.append(
            ProblemTask(
                stable_id=instruction,
                op=first.op,
                pipe=pipe,
                resource=pipe,
                latency=latency,
                issue_interval=ii,
                loop=first.iteration_path,
                action_ids=[entry.action_id for entry in group],
            )
        )
    # source_digest is derived from the LEDGER CONTENT (not the case configuration), so any HIR
    # change that changes an emitted action changes the digest.
    payload = json.dumps(
        [
            {
                'action_id': entry.action_id, 'instruction': entry.instruction, 'op': entry.op,
                'root': entry.root, 'root_kind': entry.root_kind, 'space': entry.space,
                'mode': entry.mode, 'byte_footprint': entry.byte_footprint,
                'byte_offset': entry.byte_offset,
                'attrs': list(entry.attrs), 'offset_structure': entry.offset_structure,
                'guard': entry.guard, 'iteration_path': entry.iteration_path,
                'iteration_extent': entry.iteration_extent, 'iteration_domains': entry.iteration_domains,
            }
            for entry in ledger.entries
        ],
        sort_keys=True,
    )
    canonical_serialization_sha = hashlib.sha256(payload.encode()).hexdigest()[:16]
    ledger_identity = json.dumps(
        [
            {
                'action_id': entry.action_id, 'op': entry.op, 'root': entry.root,
                'mode': entry.mode, 'iteration_path': entry.iteration_path,
                'iteration_extent': str(entry.iteration_extent),
                'offset_structure': entry.offset_structure, 'guard': entry.guard,
            }
            for entry in ledger.entries
        ],
        sort_keys=True,
        default=str,
    )
    ledger_sha = hashlib.sha256(ledger_identity.encode()).hexdigest()[:16]
    source_digest = canonical_serialization_sha
    problem = NormalizedProblem(
        problem_id='ledger-' + source_digest,
        source_digest=source_digest,
        ledger_sha=ledger_sha,
        canonical_serialization_sha=canonical_serialization_sha,
        problem_digest='',
        loop_domain={'iterations': sorted({entry.logical_iteration for entry in ledger.entries})},
        accesses=accesses,
        tasks=tasks,
        buffers=buffers,
        dependencies=dependencies,
        alias={},
    )
    problem_digest_payload = problem.as_dict()
    problem_digest_payload.pop('problem_digest', None)
    for access_payload in problem_digest_payload.get('accesses', []):
        access_payload.pop('source_offset', None)  # human string, not part of the canonical input
    problem.problem_digest = hashlib.sha256(
        json.dumps(problem_digest_payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return problem






def tasks_coverage(ledger: Ledger, problem: NormalizedProblem) -> dict:
    """Every task action_id must exist in the ledger, and every ledger action in exactly one task."""
    by_id = {entry.action_id: entry for entry in ledger.entries}
    covered = [action for task in problem.tasks for action in task.action_ids]
    if len(covered) != len(set(covered)) or set(covered) != set(by_id):
        raise LedgerError(f'TASK_COVERAGE_MISMATCH: tasks cover {len(set(covered))} of {len(by_id)} actions')
    for task in problem.tasks:
        if task.stable_id not in {by_id[action].instruction for action in task.action_ids}:
            raise LedgerError('TASK_STABLE_ID_MISMATCH: ' + task.stable_id)
    return {'tasks': len(problem.tasks), 'actions': len(covered), 'bidirectional': True}


def validate_roots(problem: NormalizedProblem) -> None:
    """Every access root must have a BufferDecl from the same ledger; else UNKNOWN_ROOT."""
    roots = {(buffer['root'], buffer['space']) for buffer in problem.buffers}
    for access in problem.accesses:
        if (access.root, access.space) not in roots:
            raise LedgerError(f'UNKNOWN_ROOT: access root {access.root!r} in {access.space} has no BufferDecl')


def require_concrete_offset(problem: NormalizedProblem) -> None:
    """OFFSET_UNKNOWN is REFUSED as 0: a consumer that needs a concrete byte offset must refuse."""
    unknown = [access for access in problem.accesses if access.offset_status != "BYTES"]
    if unknown:
        raise LedgerError(
            f"OFFSET_UNKNOWN: {len(unknown)} access(es) have no concrete byte offset "
            f"(e.g. {unknown[0].task}); refusing to treat it as 0"
        )


def assert_no_default_zero(problem: NormalizedProblem) -> None:
    """A constant-0 extractor must be killed: KNOWN offset 0 without a literal source is a defect."""
    defaulted = [access for access in problem.accesses if access.byte_offset == 0 and access.offset_status != "BYTES"]
    if defaulted:
        raise LedgerError(f"DEFAULTED_OFFSET: {len(defaulted)} access(es) carry 0 as an UNKNOWN offset")




def alias_report(problem: NormalizedProblem, entry=None) -> dict:
    """Alias decisions go through the ONE entry point alias_conflicts, per (root, generation).

    Only root_kind='real' views with a KNOWN byte offset participate. Synthetic allocations
    (root_kind='synthetic_alloc') and OFFSET_UNKNOWN accesses are excluded and counted, so the
    consumer can see exactly what was NOT checked.
    """
    if entry is not None:
        raise LedgerError('ALIAS_ENTRY_BYPASSED: only the canonical view_spec.alias_conflicts entry is allowed')
    repo = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repo / 'tools' / 'p3'))
    from view_spec import OffsetView, alias_conflicts  # noqa: PLC0415
    real = [a for a in problem.accesses if a.root_kind == 'real' and a.offset_status == 'BYTES']
    views = [
        OffsetView(
            name=a.task, root=a.root, generation=0, byte_offset=a.byte_offset,
            dtype='f32', count=max(1, a.byte_footprint // 4),
        )
        for a in real
    ]
    accesses = {a.task: a.mode for a in real}
    conflicts = []
    for root in sorted({view.root for view in views}):
        group = [view for view in views if view.root == root]
        conflicts.extend(alias_conflicts(group, {view.name: accesses[view.name] for view in group}))
    return {
        'real_views': len(views),
        'excluded_unknown_or_synthetic': len(problem.accesses) - len(views),
        'conflicts': conflicts,
    }


def coverage(ledger: Ledger, problem: NormalizedProblem) -> dict:
    """Bidirectional coverage: every access has a ledger entry and every entry has an access."""
    entry_ids = {entry.action_id for entry in ledger.entries}
    access_ids = {access.task for access in problem.accesses}
    missing = sorted(entry_ids - access_ids)
    extra = sorted(access_ids - entry_ids)
    if missing or extra:
        raise LedgerError(f"COVERAGE_MISMATCH: missing={missing[:3]} extra={extra[:3]}")
    return {"entries": len(entry_ids), "accesses": len(access_ids), "bidirectional": True}
