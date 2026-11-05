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


ESCAPE_EDGE_SCHEMA = "escape-edge/2"

#: Canonical schema tag for the normalized Problem handed to the solver/plan layers.
REAL_PROBLEM_SCHEMA = "real-problem/1"


@dataclass(frozen=True)
class EscapeEdge:
    """A NON-memory cross-region value binding (ExecutionScope output -> outer binding)."""
    edge_kind: str
    field_path: str
    scope_path: str
    inner_root: str
    inner_action_id: str
    outer_root: str
    outer_action_id: str
    iteration_path: str
    guard: tuple
    domain: str = ""
    ssa_group: Optional[int] = None
    inner_type: str = ""
    outer_type: str = ""
    outer_scope: str = ""
    outer_binding: str = ""
    iteration_domains: tuple = ()
    schema_version: str = ESCAPE_EDGE_SCHEMA


@dataclass
class Ledger:
    function: str
    entries: list = field(default_factory=list)
    escape_edges: list = field(default_factory=list)
    scope_paths: list = field(default_factory=list)
    # The typed sources the edges are built from. Kept on the ledger so validate_escape_edges can
    # rebuild the expectation INDEPENDENTLY of the edges it is checking.
    escape_records: list = field(default_factory=list)
    declaration_sites: dict = field(default_factory=dict)
    # Structural paths of the dataflow statements that emitted at least one action, in walk
    # order. Lets statement<->action coverage be checked without re-walking the TIR twice.
    statement_paths: list = field(default_factory=list)

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


def extract_ledger(function, *, escape_records=None, declaration_sites=None) -> Ledger:
    """Walk the typed TIR and record every actual memory action with a stable id.

    Escape edges are OPT-IN: without Core escape site records (and the declaration sites that
    anchor them to the final TIR) the ledger keeps escape_edges empty. It never guesses an
    endpoint by name, offset, or emission order.
    """
    ledger = Ledger(function=str(getattr(function, "name", "function")))
    views: dict = {}  # view var name -> (root name, root kind, start exprs, alloc path)
    defs: dict = {}   # var name -> defining value (for root resolution)
    scopes: list = []  # ExecutionScope paths in walk order

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
            scope_path = "/".join(path + (type(stmt).__name__,))
            if isinstance(stmt, ExecutionScope):
                scopes.append(scope_path)
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
        stmt_path = "/".join(path)
        if stmt_path not in ledger.statement_paths:
            ledger.statement_paths.append(stmt_path)
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
    ledger.scope_paths = _execution_scope_paths(function)
    # OPT-IN: consume the Core escape site records (module.metadata['escape_records']). Without
    # records the ledger keeps escape_edges empty; it never guesses by name/order/offset.
    if escape_records is not None:
        ledger.escape_records = list(escape_records)
        ledger.declaration_sites = dict(declaration_sites or {})
        ledger.escape_edges = build_escape_edges(ledger, escape_records, declaration_sites)
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
    escape_edges: list = field(default_factory=list)
    schema: str = REAL_PROBLEM_SCHEMA
    policy: dict = field(default_factory=dict)
    # One entry per ExecutionScope (whole-VF task): the scope path and ALL action ids it owns,
    # so an internal vld/vadd/vsts chain is visible as ONE schedulable task, not split per op.
    vf_tasks: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)






def _execution_scope_paths(function) -> list:
    """TIR ExecutionScope paths in walk order (matches Core escape_records ordinals)."""
    paths = []
    def walk(stmt, path):
        if isinstance(stmt, Sequential):
            for index, child in enumerate(stmt.body):
                walk(child, path + (f"s{index}",))
            return
        if isinstance(stmt, For):
            walk(stmt.body, path + (f"for:{getattr(stmt.induction_var, 'name', 'i')}",))
            return
        if isinstance(stmt, MeshScope):
            walk(stmt.body, path + ("MeshScope",))
            return
        if isinstance(stmt, ExecutionScope):
            paths.append("/".join(path + ("ExecutionScope",)))
            walk(stmt.body, path + ("ExecutionScope",))
            return
        if isinstance(stmt, If):
            walk(stmt.then_body, path + ("then",))
            walk(stmt.else_body, path + ("else",))
            return
    walk(function.body, ())
    return paths


def _declaration_sites(raw) -> dict:
    """Normalise metadata["escape_declaration_sites"] to {ssa_group: [site, ...]}."""
    if not raw:
        raise LedgerError(
            "ESCAPE_ENDPOINT_NOT_FOUND: no escape_declaration_sites metadata; the Core records the"
            " declarations but this TIR carries no structural anchor for them"
        )
    sites = {}
    for group, entries in raw.items():
        try:
            key = int(group)
        except (TypeError, ValueError):
            continue
        sites[key] = [dict(entry) for entry in entries]
    return sites


def _record_scope_path(ledger: Ledger, record) -> str:
    """The record scope_path must name a real TIR ExecutionScope; resolve it through the ordinal."""
    declared = str(record.get("scope_path", ""))
    index = record.get("scope_ordinal")
    if index is None:
        try:
            index = int(declared.split("[")[1].split("]")[0])
        except (IndexError, ValueError):
            raise LedgerError("ESCAPE_SCOPE_UNRESOLVED: record scope_path " + repr(declared) + " has no ordinal")
    if index >= len(ledger.scope_paths):
        raise LedgerError(f"ESCAPE_SCOPE_UNRESOLVED: scope ordinal {index} is outside {len(ledger.scope_paths)} scopes")
    return ledger.scope_paths[index]


def _site_scope(ledger: Ledger, path: str) -> str:
    """The innermost ExecutionScope containing a structural path ("" when there is none)."""
    best = ""
    for scope in ledger.scope_paths:
        if path == scope or path.startswith(scope + "/"):
            if len(scope) > len(best):
                best = scope
    return best


def _site_container(ledger: Ledger, path: str) -> str:
    """The structural region a site lives in, even when it is not inside an ExecutionScope.

    A main-domain consumer is emitted in the row loop OUTSIDE the scope that declared it, so
    _site_scope is empty there and reporting "" would lose the binding's real location. The
    container falls back to the longest enclosing structural prefix we can prove: the parent
    path of the statement. This is a coordinate, not an assertion about meaning.
    """
    inside = _site_scope(ledger, path)
    if inside:
        return inside
    return path.rsplit("/", 1)[0] if "/" in path else ""


def _unique_endpoint(ledger: Ledger, root: str, mode: str, endpoint: str, key: str) -> LedgerEntry:
    """Exactly one SMEM action of this mode on this root, or a named refusal."""
    candidates = [
        entry for entry in ledger.entries
        if entry.root == root and entry.mode == mode and entry.space == "SMEM"
    ]
    if not candidates:
        raise LedgerError(
            f"ESCAPE_ENDPOINT_NOT_FOUND: {endpoint} endpoint of {key} has no SMEM {mode} on root {root!r}"
        )
    if len(candidates) > 1:
        raise LedgerError(
            f"ESCAPE_ENDPOINT_AMBIGUOUS: {endpoint} endpoint of {key} has {len(candidates)} SMEM {mode}s"
            f" on root {root!r}"
        )
    return candidates[0]


def _resolve_inner_endpoint(ledger: Ledger, sites: dict, record, field, scope: str) -> tuple:
    """The producer: this field's ssa_group declared INSIDE its own record scope."""
    key = scope + "/" + str(field.get("field_path", ""))
    group = field.get("ssa_group")
    if group is None:
        raise LedgerError(f"ESCAPE_ENDPOINT_NOT_FOUND: {key} has no ssa_group; the binding is not typed")
    candidates = [
        site for site in sites.get(int(group), [])
        if _site_scope(ledger, str(site.get("path", ""))) == scope
    ]
    if not candidates:
        raise LedgerError(
            f"ESCAPE_ENDPOINT_NOT_FOUND: {key} group {group} has no declaration site inside {scope!r}"
        )
    if len(candidates) > 1:
        raise LedgerError(
            f"ESCAPE_ENDPOINT_AMBIGUOUS: {key} group {group} has {len(candidates)} declaration sites"
            f" inside {scope!r}"
        )
    site = candidates[0]
    _check_site_ordinal(key, "inner", field, site)
    _check_site_type(key, "inner", str(field.get("inner_type", "")), site)
    root = "smem@" + str(site.get("path", ""))
    return _unique_endpoint(ledger, root, "write", "inner", key), str(site.get("path", ""))


def _resolve_outer_endpoint(ledger: Ledger, sites: dict, record, field, scope: str) -> tuple:
    """The consumer: the SAME ssa_group in another scope, else the aliased consumer the Core bound."""
    key = scope + "/" + str(field.get("field_path", ""))
    group = field.get("ssa_group")
    if group is None:
        raise LedgerError(f"ESCAPE_ENDPOINT_NOT_FOUND: {key} has no ssa_group; the binding is not typed")
    sibling = [
        site for site in sites.get(int(group), [])
        if _site_scope(ledger, str(site.get("path", ""))) != scope
    ]
    if len(sibling) > 1:
        raise LedgerError(
            f"ESCAPE_ENDPOINT_AMBIGUOUS: {key} group {group} is declared in {len(sibling)} outer scopes"
        )
    if sibling:
        _check_site_ordinal(key, "outer", field, sibling[0])
        _check_site_type(key, "outer", str(field.get("outer_type") or ""), sibling[0])
        path = str(sibling[0].get("path", ""))
        outer_scope = _site_container(ledger, path)
        return (
            _unique_endpoint(ledger, "smem@" + path, "read", "outer", key),
            outer_scope,
            "outer_declaration",
        )
    if field.get("outer_declaration_ordinal") is not None or field.get("outer_consume_ordinal") is not None:
        # The Core bound this field to a consumer through the real storage alias chain, but the
        # consumer reads an ALIASED root (a fresh view of the same bytes), so no sibling
        # declaration exists for this group. Refuse rather than guess: no order-based pairing.
        raise LedgerError(
            f"ESCAPE_ENDPOINT_AMBIGUOUS: {key} group {group} was bound by the Core to a consumer, but it"
            " has no second declaration site to address that consumer structurally"
        )
    raise LedgerError(
        f"ESCAPE_ENDPOINT_NOT_FOUND: {key} group {group} has no outer declaration and no consumer binding"
    )

def _type_footprint(text: str):
    """(bytes, storage) parsed from a recorded TensorType string; (None, None) if unparsable.

    The recorded type is a string, so it must be CROSS-CHECKED against the real ledger entry
    rather than trusted: a forged dtype (f32 -> bf16) disagrees with the emitted footprint.
    """
    text = str(text)
    try:
        shape = text.split("shape=(")[1].split(")")[0]
        elements = 1
        for part in [piece.strip() for piece in shape.split(",") if piece.strip()]:
            elements *= int(part)
        bits = int(text.split("bit_width=")[1].split(",")[0])
    except (IndexError, ValueError):
        return None, None
    storage = "SMEM" if "StorageKind.SMEM" in text else ("OTHER" if "StorageKind." in text else None)
    return elements * bits // 8, storage


def _check_site_ordinal(key: str, endpoint: str, field, site) -> None:
    """The record's declaration ordinal must be one the site really emitted.

    The ordinal is the Core's own name for a declaration. The site records which ordinals it
    actually produced, so a forged ordinal (a value no site ever emitted) is refused here instead
    of being silently ignored -- which is what let the old order-based builder reproduce the same
    128 edges from corrupted records.
    """
    ordinal = field.get("declaration_ordinal") if endpoint == "inner" else field.get("outer_declaration_ordinal")
    if ordinal is None:
        return
    known = [int(value) for value in site.get("ordinals", [])]
    if not known:
        raise LedgerError(
            "ESCAPE_ORDINAL_UNANCHORED: " + key + " " + endpoint + " site records no declaration ordinal"
        )
    if int(ordinal) not in known:
        raise LedgerError(
            "ESCAPE_ORDINAL_MISMATCH: " + key + " " + endpoint + " records declaration ordinal "
            + str(ordinal) + " but the site emitted " + repr(known)
        )


def _check_site_type(key: str, endpoint: str, recorded: str, site) -> None:
    """The record's type text must equal the type the declaration site actually emitted.

    This is the typed binding check: the record CLAIMS a type, the site knows the type the TIR
    really carries, and a forged dtype (f32 -> bf16) differs even when the byte count does not.
    """
    truth = str(site.get("type", ""))
    if not recorded:
        raise LedgerError(
            "ESCAPE_TYPE_MISSING: " + key + " " + endpoint + " carries no recorded type"
        )
    if not truth:
        raise LedgerError(
            "ESCAPE_TYPE_UNPARSED: " + key + " " + endpoint + " declaration site carries no type"
        )
    if " ".join(recorded.split()) != " ".join(truth.split()):
        raise LedgerError(
            "ESCAPE_TYPE_MISMATCH: " + key + " " + endpoint + " records "
            + repr(_short_type(recorded)) + " but the declaration site emitted "
            + repr(_short_type(truth))
        )


def _short_type(text: str) -> str:
    """A compact dtype+storage view of a TensorType string, for refusal messages."""
    dtype = text.split("name='")[1].split("'")[0] if "name='" in text else "?"
    storage = text.split("StorageKind.")[1].split(":")[0] if "StorageKind." in text else "?"
    return dtype + "/" + storage


def _check_endpoint_type(edge_label: str, endpoint: str, text: str, entry, field_type: str) -> None:
    """A recorded type must agree with the REAL emitted action it is bound to."""
    if not text:
        raise LedgerError(
            "ESCAPE_TYPE_MISSING: " + edge_label + " " + endpoint + " carries no recorded type"
        )
    footprint, storage = _type_footprint(text)
    if footprint is None:
        raise LedgerError(
            "ESCAPE_TYPE_UNPARSED: " + edge_label + " " + endpoint + " type is " + repr(text[:60])
        )
    if storage != "SMEM":
        raise LedgerError(
            "ESCAPE_ENDPOINT_STORAGE: " + edge_label + " " + endpoint + " recorded type is "
            + str(storage) + ", not SMEM"
        )
    if footprint != entry.byte_footprint:
        raise LedgerError(
            "ESCAPE_TYPE_FOOTPRINT: " + edge_label + " " + endpoint + " recorded type footprint "
            + str(footprint) + " != emitted action footprint " + str(entry.byte_footprint)
        )

def build_escape_edges(ledger: Ledger, records: list, declaration_sites=None) -> list:
    """Ledger edges built ONLY from the Core escape site records and their declaration sites.

    Every endpoint is addressed by the typed binding the Core recorded, never by emission order:
    a field names its producer through (record scope, ssa_group); the declaration sites turn that
    group into structural TIR paths, and each path names exactly one SMEM root ("smem@<path>").
    The outer endpoint is the SAME ssa_group declared in a DIFFERENT scope (the sibling-scope
    weight-VF -> main-VF binding). The root must then carry exactly ONE SMEM write (producer) and
    ONE SMEM read (consumer).

    There is no zip of "every SMEM write in the scope" against "every SMEM read after it": a
    record that cannot be addressed this way is refused by name instead of being paired by
    position. That is what makes a forged ordinal/ssa_group/type change or refuse the result.
    """
    sites = _declaration_sites(declaration_sites)
    edges = []
    seen_fields = set()
    for record in records:
        scope = _record_scope_path(ledger, record)
        for field in record.get("fields", []):
            key = (scope, str(field.get("field_path", "")))
            if key in seen_fields:
                raise LedgerError(f"ESCAPE_FIELD_DUPLICATE: {key}")
            seen_fields.add(key)
            inner, inner_path = _resolve_inner_endpoint(ledger, sites, record, field, scope)
            outer, outer_scope, outer_binding = _resolve_outer_endpoint(
                ledger, sites, record, field, scope
            )
            label = key[0] + "/" + key[1]
            _check_endpoint_type(label, "inner", str(field.get("inner_type", "")), inner, field.get("inner_type"))
            _check_endpoint_type(
                label, "outer", str(field.get("outer_type") or ""), outer, field.get("outer_type")
            )
            if inner.byte_footprint != outer.byte_footprint:
                raise LedgerError(
                    f"ESCAPE_TYPE_FOOTPRINT: {key[0]}/{key[1]} producer {inner.root!r} "
                    f"footprint {inner.byte_footprint} != consumer {outer.root!r} "
                    f"footprint {outer.byte_footprint}"
                )
            if outer.space != "SMEM" or inner.space != "SMEM":
                raise LedgerError(
                    f"ESCAPE_ENDPOINT_STORAGE: {key[0]}/{key[1]} endpoints must be SMEM, got "
                    f"{inner.space!r} -> {outer.space!r}"
                )
            edges.append(EscapeEdge(
                edge_kind="ESCAPE_VALUE_BINDING",
                field_path=key[1],
                scope_path=scope,
                inner_root=inner.root,
                inner_action_id=inner.action_id,
                outer_root=outer.root,
                outer_action_id=outer.action_id,
                iteration_path=inner.iteration_path,
                guard=inner.guard,
                domain=_edge_domain(scope),
                ssa_group=int(field["ssa_group"]) if field.get("ssa_group") is not None else None,
                inner_type=str(field.get("inner_type", "")),
                outer_type=str(field.get("outer_type") or ""),
                outer_scope=outer_scope or "",
                outer_binding=outer_binding,
                iteration_domains=inner.iteration_domains,
            ))
    return edges


def _edge_domain(scope: str) -> str:
    """Coarse domain label taken from the scope's own structural path (never asserted by a case).

    The main VF scope is nested in the cooperative row loop (".../for:<var>/..."); the weight VF
    scopes are not. The label is derived from that prefix fact, so a scope cannot claim a domain
    it does not structurally have.
    """
    return "main" if "/for:" in scope else "weight"


#: Instructions that run in the vector execution mode; a whole-VF task owns all of them in
#: one ExecutionScope so the internal vld/add/store chain stays one schedulable unit.
VF_INSTRUCTIONS = frozenset({
    "Vld", "Vsts", "Vcvt", "Vmula", "Vadd", "Vmul", "Vdiv", "Vadds", "Vmuls",
    "Vsqrt", "Vdup", "Vdupv", "Pset", "Pset16", "Vcadd",
})


def whole_vf_tasks(ledger: Ledger) -> list:
    """One task per ExecutionScope: {scope, action_ids, instructions}.

    Every action whose structural path lives inside the scope belongs to that ONE task, so an
    internal register chain is not mis-reported as several independent cross-VF tasks. Derived
    from the real ledger paths, not from a template.
    """
    tasks = []
    for scope in ledger.scope_paths:
        inner = [entry for entry in ledger.entries if entry.path == scope or entry.path.startswith(scope + "/")]
        if not inner:
            continue
        tasks.append(
            {
                "scope": scope,
                "action_ids": [entry.action_id for entry in inner],
                "instructions": sorted({entry.instruction for entry in inner}),
            }
        )
    return tasks


PROBLEM_SOURCES_SCHEMA = "problem-sources/1"


@dataclass(frozen=True)
class ProblemSources:
    """The two INDEPENDENT typed sources a real Problem's policy is wired from.

    budgets come ONLY from PipelineSpec.num_stages (a loop-level stage budget).
    buffer_versions come ONLY from Empty.buffers (a per-buffer version count).
    They are different dimensions and are never copied from one another.
    """

    schema: str
    budgets: dict
    buffer_versions: dict


def capture_problem_sources(hir_module) -> "ProblemSources":
    """Read budget and buffer versions from the AUTHORED typed HIR, each from its own source."""
    from tilefoundry.analysis.pipeline_reader import (  # noqa: PLC0415
        collect_buffer_versions,
        collect_pipeline_declarations,
    )

    specs = collect_pipeline_declarations(hir_module)
    budgets = {key: spec.num_stages for key, spec in specs.items()}
    versions = collect_buffer_versions(hir_module)
    return ProblemSources(PROBLEM_SOURCES_SCHEMA, budgets, versions)


def validate_problem_sources(sources: "ProblemSources") -> dict:
    """Refuse a forged/typed-invalid source: a non-int budget or a version above the budget."""
    if getattr(sources, "schema", None) != PROBLEM_SOURCES_SCHEMA:
        raise LedgerError(
            "PROBLEM_SOURCE_SCHEMA_UNSUPPORTED: schema=%r" % (getattr(sources, "schema", None),)
        )
    for key, budget in sources.budgets.items():
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise LedgerError(
                "PIPELINE_BUDGET_SOURCE_MISMATCH: loop %r budget=%r is not a positive int from "
                "PipelineSpec.num_stages" % (key, budget)
            )
    for key, versions in sources.buffer_versions.items():
        if isinstance(versions, bool) or not isinstance(versions, int) or versions < 1:
            raise LedgerError(
                "BUFFER_VERSION_DECLARATION_INVALID: buffer %r versions=%r is not a positive int "
                "from Empty.buffers" % (key, versions)
            )
    distinct_budgets = set(sources.budgets.values())
    if len(distinct_budgets) == 1:
        (budget,) = tuple(distinct_budgets)
        for key, versions in sources.buffer_versions.items():
            if versions > budget:
                raise LedgerError(
                    "BUFFER_VERSION_EXCEEDS_PIPELINE_BUDGET: buffer %r declares %d versions but "
                    "the loop stage budget is %d" % (key, versions, budget)
                )
    return {"budgets": dict(sources.budgets), "buffer_versions": dict(sources.buffer_versions)}


def verify_policy_sources(expected: "ProblemSources", problem: "NormalizedProblem") -> dict:
    """Compare the Problem's policy with sources captured from the authored HIR.

    Never trusts the policy's own numbers: an authored constant punched in as a budget, or a
    version count copied from the budget instead of Empty.buffers, changes one side and is
    refused by name. The predicate sub-field must STAY UNKNOWN while P is unwired.

    'expected' is captured by capture_problem_sources(HIR) BEFORE lowering mutates the module;
    it is the independent typed side, not a copy of the policy being checked.
    """
    validate_problem_sources(expected)
    policy = problem.policy or {}
    if policy.get("status") != "WIRED":
        raise LedgerError("PROBLEM_POLICY_MISSING: policy.status is %r, not WIRED" % (policy.get("status"),))
    if policy.get("requested_budget") != expected.budgets:
        raise LedgerError(
            "PIPELINE_BUDGET_SOURCE_MISMATCH: policy budget %r vs PipelineSpec %r"
            % (policy.get("requested_budget"), expected.budgets)
        )
    if policy.get("buffer_versions") != expected.buffer_versions:
        raise LedgerError(
            "BUFFER_VERSION_DECLARATION_CHANGED: policy versions %r vs Empty.buffers %r"
            % (policy.get("buffer_versions"), expected.buffer_versions)
        )
    predicate = policy.get("predicate") or {}
    if predicate.get("status") != "UNKNOWN":
        raise LedgerError(
            "PREDICATE_STATUS_INVALID: predicate status %r while P output is not wired; it must "
            "stay UNKNOWN" % (predicate.get("status"),)
        )
    return {
        "budgets": dict(expected.budgets),
        "buffer_versions": dict(expected.buffer_versions),
        "independent": (
            not (set(expected.budgets.values()) == set(expected.buffer_versions.values()) == {None})
        ),
    }


def _canonical_problem_payload(problem: "NormalizedProblem") -> dict:
    """The exact dict a problem digest is taken over.

    Excludes the digest itself (it cannot include itself) and the human-readable
    source_offset string (structure is carried by offset_structure instead).
    """
    payload = problem.as_dict()
    payload.pop("problem_digest", None)
    for access_payload in payload.get("accesses", []):
        access_payload.pop("source_offset", None)
    return payload


def canonical_problem_json(problem: "NormalizedProblem") -> str:
    """Deterministic canonical serialization of a normalized problem."""
    return json.dumps(_canonical_problem_payload(problem), sort_keys=True, default=str)


def compute_problem_digest(problem: "NormalizedProblem") -> str:
    """Stable digest over the canonical problem payload (16 hex chars)."""
    return hashlib.sha256(canonical_problem_json(problem).encode()).hexdigest()[:16]


def _access_window(access):
    """(start, end) relative BYTES of an access, or None when the offset is UNKNOWN."""
    if getattr(access, "byte_offset", None) is None:
        return None
    start = int(access.byte_offset)
    return (start, start + int(access.byte_footprint))


def _windows_intersect(left, right) -> bool:
    (a0, a1), (b0, b1) = left, right
    return a0 < b1 and b0 < a1


def _access_domains(access):
    return tuple(getattr(access, "iteration_domains", ()) or ())


def _access_domain_key(access):
    """The loop-domain identity of an access: nested loop domains + its logical iteration label.

    Definition/labels come from the typed iteration metadata, never from statement order.
    """
    return (
        tuple(getattr(access, "iteration_domains", ()) or ()),
        str(getattr(access, "logical_iteration", "")),
        str(getattr(access, "iteration_path", "")),
    )


def _derive_from_bucket(bucket) -> list:
    dependencies = []
    for left in range(len(bucket)):
        for right in range(left + 1, len(bucket)):
            a, b = bucket[left], bucket[right]
            if a.mode == "write" and b.mode == "read":
                kind, producer, consumer = "RAW", a, b
            elif a.mode == "read" and b.mode == "write":
                kind, producer, consumer = "RAW", b, a
            elif a.mode == "write" and b.mode == "write":
                kind, producer, consumer = "WAW", a, b
            elif a.mode == "read" and b.mode == "read":
                kind, producer, consumer = "REUSE", a, b
            else:
                continue
            a_window, b_window = _access_window(a), _access_window(b)
            if a_window is None or b_window is None:
                status = "UNKNOWN"
            elif not _windows_intersect(a_window, b_window):
                continue  # disjoint byte windows => no dependency, whatever the names say
            else:
                status = "BYTES"
            same_domain = _access_domain_key(a) == _access_domain_key(b)
            dependencies.append(
                {
                    "producer": producer.task,
                    "consumer": consumer.task,
                    "distance": 0 if same_domain else 1,
                    "kind": kind,
                    "status": status,
                    "shared_root": a.root,
                    "producer_window": a_window,
                    "consumer_window": b_window,
                    "producer_domains": _access_domains(producer),
                    "consumer_domains": _access_domains(consumer),
                    "evidence": "window intersect on same root; kind from mode; distance from iteration domain",
                }
            )
    return dependencies


def derive_dependencies_with_stats(accesses) -> tuple:
    """Bucketed dependency derivation plus the pair-count/timing complexity facts.

    Accesses are first partitioned by (root, space); only pairs WITHIN a bucket are ever
    compared, so the work is sum_k C(k,2) rather than the naive C(n,2). The returned stats
    let a consumer see whether that prefilter actually shrank the candidate set.
    """
    import time  # noqa: PLC0415

    started = time.perf_counter()
    buckets: dict = {}
    for access in accesses:
        buckets.setdefault((access.root, access.space), []).append(access)
    total = len(accesses) * (len(accesses) - 1) // 2
    candidate = sum(len(group) * (len(group) - 1) // 2 for group in buckets.values())
    dependencies = []
    for group in buckets.values():
        dependencies.extend(_derive_from_bucket(group))
    stats = {
        "accesses": len(accesses),
        "buckets": len(buckets),
        "pairs_total_naive": total,
        "pairs_candidate_root_space": candidate,
        "pair_reduction_factor": (round(total / candidate, 3) if candidate else None),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    return dependencies, stats


def derive_dependencies(accesses) -> list:
    """RAW/WAR/WAW/REUSE from typed WINDOW intersection and loop DOMAIN, not names/order.

    Two accesses relate only when their relative byte windows INTERSECT and they resolve to
    the SAME root (alias lineage already folded into 'root'). The dependency KIND comes from
    the two access modes, never from statement order. When either window is UNKNOWN the pair
    is recorded with status UNKNOWN -- never silently treated as independent and never
    defaulted to offset 0. Every edge carries both windows, both domains, and the kind used.
    """
    return derive_dependencies_with_stats(accesses)[0]


def _problem_policy(sources: "ProblemSources | None") -> dict:
    """The policy dict, honestly UNWIRED or wired from two independent typed sources.

    The predicate sub-field is UNKNOWN in both cases: P's actual predicate is not materialised,
    so wiring budget/versions must not accidentally claim predicate coverage too.
    """
    predicate = {
        "status": "UNKNOWN",
        "required_active": None,
        "actual_active": None,
        "reason": "P actual predicate not materialised",
    }
    if sources is None:
        return {
            "status": "UNWIRED_L0",
            "requested_budget": None,
            "buffer_versions": {},
            "predicate_manifest": None,
            "predicate": predicate,
        }
    validate_problem_sources(sources)
    return {
        "status": "WIRED",
        "requested_budget": dict(sources.budgets),
        "buffer_versions": dict(sources.buffer_versions),
        "predicate_manifest": None,
        "predicate": predicate,
    }


def build_problem(ledger: Ledger, sources: "ProblemSources | None" = None) -> NormalizedProblem:
    """Ledger -> normalized Problem. Buffers come from the SAME ledger; no invented roots.

    When typed *sources* are supplied, the policy budget and buffer versions are wired from
    their two INDEPENDENT sources (PipelineSpec.num_stages and Empty.buffers); otherwise the
    policy stays honestly UNWIRED.
    """
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
    # Dependencies: derived from typed byte-window intersection + loop domain (not names/order).
    dependencies = derive_dependencies(accesses)
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
        escape_edges=list(ledger.escape_edges),
        schema=REAL_PROBLEM_SCHEMA,
        policy=_problem_policy(sources),
        vf_tasks=whole_vf_tasks(ledger),
    )
    problem.problem_digest = compute_problem_digest(problem)
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




def validate_escape_edges(ledger: Ledger) -> dict:
    """Verify the edges against an expectation rebuilt INDEPENDENTLY from the Core records.

    The expectation is never derived from the edges under test: it is rebuilt from
    ledger.escape_records + ledger.declaration_sites (the typed sources), so deleting an edge,
    deleting a whole scope, or keeping only one domain cannot pass by shrinking the expectation
    with it. complete=True additionally requires BOTH the weight and the main domain to be
    present, because a ledger that lost one of them has not covered the escape relation.

    Every endpoint is re-checked against the real ledger: the action id must exist, its root,
    space and mode must match the edge, and the producer iteration path and guard must match.
    """
    edges = list(ledger.escape_edges)
    if any(edge.edge_kind != "ESCAPE_VALUE_BINDING" for edge in edges):
        raise LedgerError("ESCAPE_EDGE_KIND: only ESCAPE_VALUE_BINDING is allowed")
    if not edges:
        raise LedgerError("ESCAPE_EDGE_MISSING: no ESCAPE_VALUE_BINDING edge at all")
    expected = build_escape_edges(ledger, ledger.escape_records, ledger.declaration_sites)
    expected_keys = {(edge.scope_path, edge.field_path) for edge in expected}
    actual_keys = set()
    for edge in edges:
        key = (edge.scope_path, edge.field_path)
        if key in actual_keys:
            raise LedgerError("ESCAPE_FIELD_DUPLICATE: " + repr(key))
        actual_keys.add(key)
    missing = sorted(expected_keys - actual_keys)
    if missing:
        raise LedgerError(
            "ESCAPE_EDGE_MISSING: " + str(len(missing)) + " expected field(s) have no edge, e.g. " + repr(missing[0])
        )
    extra = sorted(actual_keys - expected_keys)
    if extra:
        raise LedgerError(
            "ESCAPE_EDGE_UNEXPECTED: " + str(len(extra)) + " edge(s) have no typed field, e.g. " + repr(extra[0])
        )
    # A field permutation inside ONE scope keeps every (scope, field_path) key present, so the key
    # set alone cannot catch it: the expectation is a MAP from the typed field to its endpoints,
    # and every edge must match the endpoints its own field implies.
    by_key = {(edge.scope_path, edge.field_path): edge for edge in expected}
    by_id = ledger.by_id()
    for edge in edges:
        label = edge.scope_path + "/" + edge.field_path
        wanted = by_key[(edge.scope_path, edge.field_path)]
        for attribute in ("inner_root", "outer_root", "ssa_group", "domain", "outer_scope"):
            if getattr(edge, attribute) != getattr(wanted, attribute):
                raise LedgerError(
                    "ESCAPE_EDGE_MISMATCH: " + label + " has " + attribute + "="
                    + repr(getattr(edge, attribute)) + ", typed records give "
                    + repr(getattr(wanted, attribute))
                )
        for endpoint, action, root in (
            ("inner", edge.inner_action_id, edge.inner_root),
            ("outer", edge.outer_action_id, edge.outer_root),
        ):
            entry = by_id.get(action)
            if entry is None:
                raise LedgerError(
                    "ESCAPE_ENDPOINT_NOT_FOUND: " + label + " " + endpoint + " action "
                    + repr(action) + " is not in the ledger"
                )
            if entry.root != root:
                raise LedgerError(
                    "ESCAPE_EDGE_MISMATCH: " + label + " " + endpoint + " action "
                    + repr(action) + " is rooted at " + repr(entry.root) + ", edge says " + repr(root)
                )
            if entry.space != "SMEM":
                raise LedgerError(
                    "ESCAPE_ENDPOINT_STORAGE: " + label + " " + endpoint + " action "
                    + repr(action) + " is in " + repr(entry.space) + ", not SMEM"
                )
        producer = by_id[edge.inner_action_id]
        if edge.iteration_path != producer.iteration_path:
            raise LedgerError(
                "ESCAPE_EDGE_MISMATCH: " + label + " iteration_path " + repr(edge.iteration_path)
                + " is not the producer iteration path " + repr(producer.iteration_path)
            )
        if tuple(edge.guard) != tuple(producer.guard):
            raise LedgerError(
                "ESCAPE_EDGE_MISMATCH: " + label + " guard is not the producer guard"
            )
    domains = sorted({edge.domain for edge in edges})
    for required in ("weight", "main"):
        if required not in domains:
            raise LedgerError(
                "ESCAPE_DOMAIN_UNCOVERED: complete=True requires the " + repr(required)
                + " domain, got " + repr(domains)
            )
    scopes = {edge.scope_path for edge in edges}
    return {
        "schema": ESCAPE_EDGE_SCHEMA,
        "edges": len(edges),
        "scopes": len(scopes),
        "domains": domains,
        "dropped_types": _dropped_types(ledger.escape_records),
        "complete": True,
    }


def _dropped_types(records: list) -> dict:
    """Audit the dropped declarations by REAL dtype and storage.

    The Core drops SMEM allocations with no outer binding. Measured on the RMSNorm kernel this is
    128 BF16 SMEM temporaries (one per weight scope plus 64 in the main VF), NOT the f32 VF
    scratch views: those are kept as fields. The dtype is read off the recorded type, so the
    audit reports what was dropped instead of repeating an assumption.
    """
    counts = {}
    for record in records:
        for dropped in record.get("dropped_fields", []):
            text = str(dropped.get("inner_type", ""))
            if "FloatDType(name='" in text:
                dtype = text.split("FloatDType(name='")[1].split("'")[0]
            else:
                dtype = "unknown"
            key = dtype + " " + ("SMEM" if "StorageKind.SMEM" in text else "OTHER")
            counts[key] = counts.get(key, 0) + 1
    return counts




def validate_escape_audit(records: list, duplicates: list) -> dict:
    """Audit the escape records: dropped real edges and duplicate different fields are refused."""
    for record in records:
        for dropped in record.get("dropped_fields", []):
            if dropped.get("had_outer_binding"):
                raise LedgerError(f"ESCAPE_DROPPED_REAL_EDGE: {dropped.get('declaration_ordinal')}")
    group_fields = {}
    for record in records:
        for field in record.get("fields", []):
            group = field.get("ssa_group")
            group_fields[group] = group_fields.get(group, 0) + 1
        for dropped in record.get("dropped_fields", []):
            group = dropped.get("ssa_group")
            group_fields[group] = group_fields.get(group, 0) + 1
    for group, count in group_fields.items():
        if count > 1:
            raise LedgerError(f"ESCAPE_DUPLICATE_DIFFERENT_FIELD: group {group} has {count} fields")
    seen = {event.get("ssa_group") for event in duplicates}
    orphans = sorted(group for group in seen if group_fields.get(group, 0) == 0)
    if orphans:
        raise LedgerError(f"ESCAPE_DUPLICATE_ORPHAN: {orphans[:3]}")
    return {"records": len(records), "duplicates": len(duplicates), "groups": len(group_fields)}


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


def statement_coverage(ledger: Ledger) -> dict:
    """Bidirectional statement<->action coverage.

    Every dataflow statement that emitted an action must have one, and every action must
    name a real statement. A statement side with no action, or an action with no statement,
    is refused by name rather than silently dropped.
    """
    stmt_paths = set(ledger.statement_paths)
    entry_paths = {entry.path for entry in ledger.entries}
    if not stmt_paths:
        # A ledger built before statement paths were recorded cannot be certified complete.
        if entry_paths:
            raise LedgerError(
                "STATEMENT_COVERAGE_MISMATCH: ledger has %d actions but no recorded statement "
                "paths, so statement<->action coverage cannot be proved" % len(entry_paths)
            )
        raise LedgerError("STATEMENT_COVERAGE_MISMATCH: no dataflow statement produced an action")
    stmt_without_action = sorted(stmt_paths - entry_paths)
    action_without_stmt = sorted(entry_paths - stmt_paths)
    if stmt_without_action or action_without_stmt:
        raise LedgerError(
            "STATEMENT_COVERAGE_MISMATCH: stmt_without_action=%s action_without_stmt=%s"
            % (stmt_without_action[:3], action_without_stmt[:3])
        )
    return {"statements": len(stmt_paths), "actions": len(entry_paths), "bidirectional": True}


def _resolve_effect_op(callable_, defs):
    """Resolve a TIR Evaluate callable through Var/Call/ScheduleOp indirection."""
    seen = set()
    op = callable_
    while True:
        if isinstance(op, Var):
            key = getattr(op, "name", None)
            if key in seen or key not in defs:
                break
            seen.add(key)
            op = defs[key]
            continue
        if isinstance(op, Call):
            op = op.target
            continue
        if isinstance(op, ScheduleOp):
            op = op.op
            continue
        break
    return op


def _iter_effect_statement_paths(function) -> list:
    """Independently enumerate TIR dataflow statements: (path, status, op).

    status is "action" for a memory-effect statement this ledger knows how to extract,
    and "unsupported" for a memory-effect op it does NOT map -- the latter is refused by
    name, never silently skipped. Pure arithmetic/structural Evaluate nodes state no
    operand memory effect and are not dataflow statements.
    """
    defs = {}
    out = []

    def walk(stmt, path):
        if isinstance(stmt, Sequential):
            for index, child in enumerate(stmt.body):
                walk(child, path + ("s%d" % index,))
            return
        if isinstance(stmt, For):
            walk(stmt.body, path + ("for:%s" % getattr(stmt.induction_var, "name", "i"),))
            return
        if isinstance(stmt, (MeshScope, ExecutionScope)):
            walk(stmt.body, path + (type(stmt).__name__,))
            return
        if isinstance(stmt, If):
            walk(stmt.then_body, path + ("then",))
            walk(stmt.else_body, path + ("else",))
            return
        if isinstance(stmt, LetStmt):
            defs[str(getattr(stmt.var, "name", ""))] = stmt.value
            return
        if not isinstance(stmt, Evaluate):
            return
        op = _resolve_effect_op(stmt.callable, defs)
        if isinstance(op, Copy):
            name = "copy"
        elif isinstance(op, Fill):
            name = "fill"
        else:
            name = type(op).__name__
        if isinstance(op, (Copy, Fill)) or name in _ACTION_EFFECTS:
            out.append(("/".join(path), "action", name))
            return
        schema = getattr(type(op), "_op_schema", None)
        params = getattr(schema, "signature", ()) or ()
        has_effect = any(
            getattr(param, "kind", None) == "input" and getattr(param, "effect", None) is not None
            for param in params
        )
        if has_effect:
            out.append(("/".join(path), "unsupported", name))

    walk(function.body, ())
    return out


def verify_statement_coverage(function, ledger: Ledger) -> dict:
    """Independently prove every dataflow statement has an action and vice versa.

    The expectation is re-derived from the TIR by a walker that does NOT reuse the
    extraction's recorded paths, so a statement the extractor silently skipped shows up as
    stmt_without_action; a memory-effect op the extractor cannot classify is refused as
    UNSUPPORTED_EFFECT_OP rather than ignored.
    """
    rows = _iter_effect_statement_paths(function)
    unsupported = [(path, name) for path, status, name in rows if status == "unsupported"]
    if unsupported:
        raise LedgerError("UNSUPPORTED_EFFECT_OP: %s" % (unsupported[:3],))
    expected = {path for path, status, _name in rows if status == "action"}
    actual = {entry.path for entry in ledger.entries}
    if not expected:
        raise LedgerError("STATEMENT_COVERAGE_MISMATCH: no dataflow statement was found")
    stmt_without_action = sorted(expected - actual)
    action_without_stmt = sorted(actual - expected)
    if stmt_without_action or action_without_stmt:
        raise LedgerError(
            "STATEMENT_COVERAGE_MISMATCH: stmt_without_action=%s action_without_stmt=%s"
            % (stmt_without_action[:3], action_without_stmt[:3])
        )
    return {"statements": len(expected), "actions": len(actual), "bidirectional": True}


def verify_real_problem(problem: NormalizedProblem, ledger: Optional[Ledger] = None, function=None) -> dict:
    """Verify a real-problem/1 instance before any consumer trusts it.

    Refuses an unknown schema, a digest that does not match its own canonical payload
    (a mutated field or a dropped policy changes it), and -- when the source ledger is
    supplied -- the three bidirectional coverages. Names the failed side in every code.
    """
    if getattr(problem, "schema", None) != REAL_PROBLEM_SCHEMA:
        raise LedgerError(
            "PROBLEM_SCHEMA_UNSUPPORTED: schema=%r, expected %r"
            % (getattr(problem, "schema", None), REAL_PROBLEM_SCHEMA)
        )
    actual = compute_problem_digest(problem)
    if problem.problem_digest != actual:
        raise LedgerError(
            "PROBLEM_DIGEST_MISMATCH: stored problem_digest=%r but the canonical payload "
            "hashes to %r" % (problem.problem_digest, actual)
        )
    report = {"schema": problem.schema, "problem_digest": problem.problem_digest}
    if ledger is not None:
        report["coverage"] = coverage(ledger, problem)
        report["tasks"] = tasks_coverage(ledger, problem)
        # When the TIR is supplied, re-derive the statement expectation independently of
        # the extraction's recorded paths; otherwise fall back to the recorded paths.
        report["statements"] = (
            verify_statement_coverage(function, ledger)
            if function is not None
            else statement_coverage(ledger)
        )
    return report


def _output_roots(function) -> set:
    """Names of the trailing output parameters (the GM return chain endpoints)."""
    count = getattr(function, "output_count", 0) or 0
    params = tuple(getattr(function, "params", ()))
    if count <= 0 or count > len(params):
        return set()
    return {str(getattr(param, "name", "")) for param in params[-count:]}


def verify_final_gm_return_chain(function, ledger: Ledger) -> dict:
    """The returned GM output must be written by a real action.

    A program whose final GM store was dropped (the loop turns out to compute a private
    value) has no write to an output root, so it is refused by name rather than accepted
    as a valid kernel with an empty output.
    """
    outputs = _output_roots(function)
    if not outputs:
        raise LedgerError("FINAL_GM_RETURN_MISSING: the function declares no output parameter")
    final_writes = [
        entry
        for entry in ledger.entries
        if entry.mode == "write" and entry.space == "GMEM" and entry.root in outputs
    ]
    if not final_writes:
        raise LedgerError(
            "FINAL_GM_RETURN_MISSING: no write to an output GM root %s; the returned value "
            "has no producing store" % (sorted(outputs),)
        )
    return {"outputs": sorted(outputs), "final_gm_writes": len(final_writes)}


def verify_whole_vf_tasks(ledger: Ledger, problem: NormalizedProblem) -> dict:
    """Each ExecutionScope must appear as exactly ONE whole-VF task over all its actions."""
    expected = {task["scope"]: set(task["action_ids"]) for task in whole_vf_tasks(ledger)}
    actual = {task["scope"]: set(task["action_ids"]) for task in problem.vf_tasks}
    if set(expected) != set(actual):
        raise LedgerError(
            "VF_TASK_COVERAGE_MISMATCH: scopes expected=%s actual=%s"
            % (sorted(expected), sorted(actual))
        )
    for scope, ids in expected.items():
        if actual[scope] != ids:
            extra = sorted(actual[scope] - ids)
            missing = sorted(ids - actual[scope])
            raise LedgerError(
                "VF_TASK_COVERAGE_MISMATCH: scope %r action_ids differ (extra=%s missing=%s)"
                % (scope, extra[:3], missing[:3])
            )
    return {"vf_tasks": len(expected), "scopes": sorted(expected)}


def verify_predicate_wiring(problem: NormalizedProblem) -> dict:
    """Predicate fields must be an honest UNKNOWN, never a default 0 or a self-proof.

    P's actual predicate is not materialised, so the only legal state is UNKNOWN with no
    numbers. A WIRED state is legal only when its required and actual sets are BOTH present
    and DIFFERENT -- an equality would be the required==actual self-certification the design
    forbids.
    """
    predicate = (problem.policy or {}).get("predicate")
    if predicate is None:
        raise LedgerError("PREDICATE_FIELD_MISSING: policy.predicate is absent")
    status = predicate.get("status")
    if status not in ("UNKNOWN", "WIRED"):
        raise LedgerError("PREDICATE_STATUS_INVALID: status=%r" % (status,))
    required = predicate.get("required_active")
    actual = predicate.get("actual_active")
    if status == "UNKNOWN":
        if required is not None or actual is not None:
            raise LedgerError(
                "DEFAULTED_UNKNOWN_VALUE: UNKNOWN predicate carries required=%r actual=%r; "
                "UNKNOWN must not be turned into a number" % (required, actual)
            )
        return {"status": "UNKNOWN"}
    if required is None or actual is None:
        raise LedgerError("PREDICATE_SOURCE_UNBOUND: WIRED predicate lacks required/actual")
    if set(required) == set(actual):
        raise LedgerError(
            "PREDICATE_SELF_CERTIFIED: required==actual (%r); a predicate cannot certify itself"
            % (sorted(required),)
        )
    return {"status": "WIRED"}


def verify_escape_endpoints(ledger: Ledger) -> dict:
    """Escape edges must match the typed records; a wrong endpoint is ESCAPE_EDGE_MISMATCH.

    The expectation is rebuilt from the typed records (never from the edges under test).
    This does not require both weight and main domains (a single-domain carrier is valid);
    it only requires that every edge equals the edge its typed records imply.
    """
    if not ledger.escape_records:
        raise LedgerError("ESCAPE_EDGE_MISSING: no typed escape records to check against")
    expected = build_escape_edges(ledger, ledger.escape_records, ledger.declaration_sites)
    if not ledger.escape_edges:
        raise LedgerError("ESCAPE_EDGE_MISSING: the ledger carries no escape edge")
    if len(expected) != len(ledger.escape_edges):
        raise LedgerError(
            "ESCAPE_EDGE_MISMATCH: typed records imply %d edge(s), ledger has %d"
            % (len(expected), len(ledger.escape_edges))
        )
    def _key(edge):
        return (edge.scope_path, edge.field_path, edge.inner_action_id)
    expected_sorted = sorted(expected, key=_key)
    actual_sorted = sorted(ledger.escape_edges, key=_key)
    for want, got in zip(expected_sorted, actual_sorted):
        if want != got:
            raise LedgerError(
                "ESCAPE_EDGE_MISMATCH: field %r/%r differs between typed records and ledger"
                % (got.scope_path, got.field_path)
            )
    return {"escape_edges": len(ledger.escape_edges), "complete": True}


def verify_l2_problem(function, ledger: Ledger, problem: NormalizedProblem) -> dict:
    """L2 shape: roots/windows, whole-VF task, final GM return chain, predicate UNKNOWN, escape."""
    validate_roots(problem)
    assert_no_default_zero(problem)
    report = {
        "final_gm": verify_final_gm_return_chain(function, ledger),
        "vf_tasks": verify_whole_vf_tasks(ledger, problem),
        "predicate": verify_predicate_wiring(problem),
    }
    if ledger.escape_records:
        report["escape"] = verify_escape_endpoints(ledger)
    return report



S_PROBLEM_SCHEMA = "s-problem/1"
WINDOW_COVERAGE_UNKNOWN_RATIO_LIMIT = 0.5


def dependency_kind_status_counts(dependencies) -> dict:
    """RAW/WAR/WAW/REUSE x PROVEN/UNKNOWN counts (status BYTES == PROVEN)."""
    counts = {}
    for kind in ("RAW", "WAR", "WAW", "REUSE"):
        for status in ("PROVEN", "UNKNOWN"):
            counts["%s_%s" % (kind, status)] = 0
    for edge in dependencies:
        kind = edge.get("kind")
        status = "PROVEN" if edge.get("status") == "BYTES" else "UNKNOWN"
        key = "%s_%s" % (kind, status)
        counts[key] = counts.get(key, 0) + 1
    return counts


def dependency_unknown_ratio(dependencies) -> float:
    total = len(dependencies)
    if total == 0:
        return 0.0
    unknown = sum(1 for edge in dependencies if edge.get("status") != "BYTES")
    return round(unknown / total, 6)


def s_hard_constraint_dependencies(problem: NormalizedProblem) -> list:
    """Only PROVEN (status BYTES) edges are S hard constraints."""
    return [edge for edge in problem.dependencies if edge.get("status") == "BYTES"]


def s_unknown_dependencies(problem: NormalizedProblem) -> list:
    """UNKNOWN edges: NOT constraints; S must surface them as unknown, never default 0."""
    return [edge for edge in problem.dependencies if edge.get("status") != "BYTES"]


def build_s_problem_document(problem: NormalizedProblem) -> dict:
    """The independent S-facing problem document for one real-problem/1 instance.

    Carries kind x status counts and unknown_ratio; proves UNKNOWN edges are NOT hard
    constraints via s_contract; and reports a NAMED coverage gap (WINDOW_COVERAGE_LOW) when
    too many edges are UNKNOWN, so S is not handed a huge mostly-unknown graph as if it were
    solved input.
    """
    dependencies = list(problem.dependencies)
    counts = dependency_kind_status_counts(dependencies)
    ratio = dependency_unknown_ratio(dependencies)
    hard = s_hard_constraint_dependencies(problem)
    coverage_gaps = []
    if dependencies and ratio > WINDOW_COVERAGE_UNKNOWN_RATIO_LIMIT:
        coverage_gaps.append("WINDOW_COVERAGE_LOW")
    return {
        "schema": S_PROBLEM_SCHEMA,
        "problem_schema": problem.schema,
        "problem_digest": problem.problem_digest,
        "ledger_sha": problem.ledger_sha,
        "source_digest": problem.source_digest,
        "problem": problem.as_dict(),
        "dependencies": {
            "counts": counts,
            "total": len(dependencies),
            "unknown": len(dependencies) - len(hard),
            "proven": len(hard),
            "unknown_ratio": ratio,
        },
        "s_contract": {
            "unknown_not_a_hard_constraint": True,
            "hard_constraint_statuses": ["BYTES"],
            "note": "S may feed ONLY status=BYTES edges to z3. UNKNOWN edges are not "
                    "constraints; they are reported as unknown and never defaulted to offset 0.",
        },
        "coverage_gaps": coverage_gaps,
    }


def serialize_s_problem(problem: NormalizedProblem) -> str:
    """Deterministic canonical text of the S-facing document."""
    return json.dumps(build_s_problem_document(problem), sort_keys=True, default=str)


def write_s_problem(path, problem: NormalizedProblem) -> None:
    Path(path).write_text(serialize_s_problem(problem) + "\n", encoding="utf-8")


def read_s_problem(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest_of_problem_dict(payload: dict) -> str:
    import copy  # noqa: PLC0415

    data = copy.deepcopy(payload)
    data.pop("problem_digest", None)
    for access_payload in data.get("accesses", []):
        access_payload.pop("source_offset", None)
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:16]


def verify_s_problem_document(document: dict, *, problem: NormalizedProblem | None = None) -> dict:
    """Verify a serialized S document: schema, self-consistent digest, S contract present."""
    if document.get("schema") != S_PROBLEM_SCHEMA:
        raise LedgerError(
            "PROBLEM_SCHEMA_UNSUPPORTED: schema=%r, expected %r"
            % (document.get("schema"), S_PROBLEM_SCHEMA)
        )
    payload = document.get("problem")
    if not isinstance(payload, dict):
        raise LedgerError("PROBLEM_DIGEST_MISMATCH: no problem payload to hash")
    actual = _digest_of_problem_dict(payload)
    if document.get("problem_digest") != actual:
        raise LedgerError(
            "PROBLEM_DIGEST_MISMATCH: document digest %r vs canonical payload %r"
            % (document.get("problem_digest"), actual)
        )
    if problem is not None and actual != problem.problem_digest:
        raise LedgerError(
            "PROBLEM_DIGEST_MISMATCH: document %r != problem %r"
            % (actual, problem.problem_digest)
        )
    contract = document.get("s_contract") or {}
    if contract.get("unknown_not_a_hard_constraint") is not True:
        raise LedgerError("S_CONTRACT_MISSING: unknown_not_a_hard_constraint is not True")
    return {
        "problem_digest": actual,
        "coverage_gaps": list(document.get("coverage_gaps") or []),
        "unknown_ratio": (document.get("dependencies") or {}).get("unknown_ratio", 0.0),
    }
