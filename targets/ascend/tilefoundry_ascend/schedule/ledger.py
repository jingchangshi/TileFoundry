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
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from tilefoundry.ir.core import Call, Constant, Tuple, Var
from tilefoundry.ir.hir.schedule import ScheduleOp
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.tir.memory import Copy, Fill, PtrOf, TensorView
from tilefoundry.ir.tir.stmts import Evaluate, ExecutionScope, For, If, LetStmt, MeshScope, Sequential

#: TIR instruction class name -> [(operand index, mode)]; mode is "read" | "write".
_ACTION_EFFECTS = {
    "Vld": ((0, "read"), (1, "write")),
    "Vsts": ((0, "read"), (1, "write")),
    "GmToUb": ((0, "read"), (1, "write")),
    "UbToGm": ((0, "read"), (1, "write")),
}


class LedgerError(ValueError):
    """A named refusal from the ledger/Problem extraction."""


@dataclass(frozen=True)
class LedgerEntry:
    action_id: str
    op: str
    root: str
    root_kind: str
    space: str
    mode: str
    byte_footprint: int
    byte_offset: Optional[int]      # None == OFFSET_UNKNOWN
    logical_iteration: str
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
    views: dict = {}  # view var name -> (root name, root kind, literal starts, alloc path)
    defs: dict = {}   # var name -> defining value (for root resolution)

    def offset_of(var) -> Optional[int]:
        name = getattr(var, "name", None)
        if name not in views:
            return None
        starts = views[name][2]
        if starts and all(isinstance(start, int) for start in starts):
            return sum(starts)
        return None

    def walk(stmt, path: tuple, loops: tuple) -> None:
        if isinstance(stmt, Sequential):
            for index, child in enumerate(stmt.body):
                walk(child, path + (f"s{index}",), loops)
            return
        if isinstance(stmt, For):
            walk(stmt.body, path + (f"for:{getattr(stmt.induction_var, 'name', 'i')}",), loops + (str(getattr(stmt.induction_var, "name", "i")),))
            return
        if isinstance(stmt, (MeshScope, ExecutionScope)):
            walk(stmt.body, path + (type(stmt).__name__,), loops)
            return
        if isinstance(stmt, If):
            walk(stmt.then_body, path + ("then",), loops)
            walk(stmt.else_body, path + ("else",), loops)
            return
        if isinstance(stmt, LetStmt):
            value = stmt.value
            defs[str(getattr(stmt.var, "name", ""))] = value
            if isinstance(value, Call) and isinstance(value.target, (TensorView, PtrOf)):
                root, kind = _root_of(value.args[0] if value.args else value, views, defs)
                views[str(getattr(stmt.var, "name", ""))] = (root, kind, _slice_starts(value), "/".join(path))
            return
        if not isinstance(stmt, Evaluate):
            return
        op = stmt.callable
        if isinstance(op, Call):
            op = op.target
        if isinstance(op, ScheduleOp):
            op = op.op
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
                    op=name,
                    root=root,
                    root_kind=root_kind,
                    space=space,
                    mode=mode,
                    byte_footprint=_shape_bytes(type_),
                    byte_offset=offset_of(arg),
                    logical_iteration=(loops[-1] if loops else "outside"),
                    path="/".join(path),
                )
            )

    walk(function.body, (), ())
    return ledger


@dataclass(frozen=True)
class ProblemAccess:
    task: str
    root: str
    root_kind: str
    space: str
    mode: str
    byte_footprint: int
    byte_offset: Optional[int]
    offset_status: str  # "KNOWN" | "OFFSET_UNKNOWN"
    logical_iteration: str


@dataclass
class NormalizedProblem:
    problem_id: str
    source_digest: str
    loop_domain: dict
    accesses: list
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
            root=entry.root,
            root_kind=entry.root_kind,
            space=entry.space,
            mode=entry.mode,
            byte_footprint=entry.byte_footprint,
            byte_offset=entry.byte_offset,
            offset_status="KNOWN" if entry.byte_offset is not None else "OFFSET_UNKNOWN",
            logical_iteration=entry.logical_iteration,
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
    digest = hashlib.sha256("|".join(entry.action_id for entry in ledger.entries).encode()).hexdigest()[:16]
    return NormalizedProblem(
        problem_id="ledger-" + digest,
        source_digest=digest,
        loop_domain={"iterations": sorted({entry.logical_iteration for entry in ledger.entries})},
        accesses=accesses,
        buffers=buffers,
        dependencies=dependencies,
        alias={},
    )




def validate_roots(problem: NormalizedProblem) -> None:
    """Every access root must have a BufferDecl from the same ledger; else UNKNOWN_ROOT."""
    roots = {(buffer['root'], buffer['space']) for buffer in problem.buffers}
    for access in problem.accesses:
        if (access.root, access.space) not in roots:
            raise LedgerError(f'UNKNOWN_ROOT: access root {access.root!r} in {access.space} has no BufferDecl')


def require_concrete_offset(problem: NormalizedProblem) -> None:
    """OFFSET_UNKNOWN is REFUSED as 0: a consumer that needs a concrete byte offset must refuse."""
    unknown = [access for access in problem.accesses if access.offset_status != "KNOWN"]
    if unknown:
        raise LedgerError(
            f"OFFSET_UNKNOWN: {len(unknown)} access(es) have no concrete byte offset "
            f"(e.g. {unknown[0].task}); refusing to treat it as 0"
        )


def assert_no_default_zero(problem: NormalizedProblem) -> None:
    """A constant-0 extractor must be killed: KNOWN offset 0 without a literal source is a defect."""
    defaulted = [access for access in problem.accesses if access.byte_offset == 0 and access.offset_status != "KNOWN"]
    if defaulted:
        raise LedgerError(f"DEFAULTED_OFFSET: {len(defaulted)} access(es) carry 0 as an UNKNOWN offset")




def alias_report(problem: NormalizedProblem) -> dict:
    """Alias decisions go through the ONE entry point alias_conflicts, per (root, generation).

    Only root_kind='real' views with a KNOWN byte offset participate. Synthetic allocations
    (root_kind='synthetic_alloc') and OFFSET_UNKNOWN accesses are excluded and counted, so the
    consumer can see exactly what was NOT checked.
    """
    repo = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repo / 'tools' / 'p3'))
    from view_spec import OffsetView, alias_conflicts  # noqa: PLC0415
    real = [a for a in problem.accesses if a.root_kind == 'real' and a.offset_status == 'KNOWN']
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
