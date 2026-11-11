"""Emit the completion trace as REAL typed TIR, and read it back.

WHY EMIT AT ALL. A completion plan that exists only as a Python list cannot be verified
against the program that will run: the verifier would be reading the generator's own
intention. This module materialises FlagSet / FlagWait / Drain statements -- the CORE's typed
ops, so the standard verifier re-checks slot/generation/pipe legality -- and reads them back
out of the finished TIR. Verification therefore reads the ACTUAL program, and the structural
checks (slot == generation % versions, non-empty channel, legal pipe pair) run twice: once in
the core verifier, once here.

THE EMITTER PRESERVES ORDER, and that is the point. Statements are appended in trace order
inside the function's own mesh scope, so a Set that must precede its Wait still does, and a
FREE Set still lands after the last consumer's read_end. Records that carry a CONDITIONAL
guard are wrapped in an If around the single statement, so a guarded Set and a guarded Wait
become real control flow rather than an attribute the checker has to trust.

The real ACCESS statements of the program are NOT emitted here: they already exist in the
module. The trace's access roles are the references the checker uses to decide "did the FREE
happen after the last read", and they are resolved to the module's own statement order.
"""
from __future__ import annotations

from dataclasses import replace

from tilefoundry.ir.core import Constant, Var
from tilefoundry.ir.tir.flag import Drain, FlagSet, FlagWait
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, ExecutionScope, For, If, MeshScope, Sequential
from tilefoundry.ir.types import DType, make_tensor_type

from .guards import ALWAYS, COND, NEVER, normalize_guard
from .model import CompletionError
from .schedule import ROLE_DRAIN, ROLE_SET, ROLE_WAIT, TraceRecord

_SYNC_KEY = "ascend.completion_sync"


def _bool_const(value: bool) -> Constant:
    return Constant(type=make_tensor_type((), dtype=DType.bool), value=bool(value))


def _flag_kind(kind) -> str:
    """The CORE completion kind for one obligation kind.

    The core knows ready/free/once. STATE and EXIT edges are ordering edges between two
    writes (or the region exit), which is 'once' in the core's vocabulary, so they are not
    mislabelled as a data-ready edge.
    """
    if str(kind).lower() in ("ready",):
        return "ready"
    if str(kind).lower() in ("free",):
        return "free"
    return "once"


def _flag_node(cls, record):
    return cls(channel_name=record.channel, kind=_flag_kind(record.kind),
               slot=int(record.slot), generation=int(record.generation),
               versions=max(1, int(record.versions)), flag_id=int(record.event_id),
               producer_pipe=str(record.producer_pipe or "device"),
               consumer_pipe=str(record.consumer_pipe or "device"))


def _wrap(statement, guard, induction_name="i"):
    """Wrap one statement in its guard, so conditional completion is REAL control flow."""
    normalized = normalize_guard(guard)
    if normalized.kind == ALWAYS:
        return statement
    if normalized.kind == NEVER:
        return If(cond=_bool_const(False), then_body=Sequential(body=()),
                  else_body=Sequential(body=(statement,)))
    source = normalized.source
    if isinstance(source, For):
        raise CompletionError("COMPLETION_GUARD_INVALID",
                              "a For cannot be a statement guard", "", "expression", "For")
    cond = source if source is not None else Var(name=induction_name,
                                                type=make_tensor_type((), dtype=DType.bool))
    return If(cond=cond, then_body=Sequential(body=(statement,)),
              else_body=Sequential(body=()))


def trace_nodes(trace) -> list:
    """The typed TIR statements for one completion trace, in execution order."""
    nodes = []
    for record in trace.records:
        if record.role == ROLE_SET:
            node = Evaluate(callable=_flag_node(FlagSet, record), args=())
        elif record.role == ROLE_WAIT:
            node = Evaluate(callable=_flag_node(FlagWait, record), args=())
        elif record.role == ROLE_DRAIN:
            node = Evaluate(callable=Drain(), args=())
        else:
            # An access role is NOT a completion statement: the real access already exists in
            # the module, and re-emitting a copy would double the program.
            continue
        nodes.append(_wrap(node, record.guard))
    return nodes


def _mesh_scopes(function):
    """The innermost mesh/execution scope chain, so a drain lands where threads execute."""
    def find(node, path):
        if isinstance(node, Sequential):
            for index, child in enumerate(node.body):
                found = find(child, path + ("s%d" % index,))
                if found is not None:
                    return found
            return None
        if isinstance(node, (MeshScope, ExecutionScope)):
            deeper = find(node.body, path + (type(node).__name__,))
            return deeper if deeper is not None else ("/".join(path), type(node).__name__)
        return None

    return find(function.body, ())


def emit_trace_tir(function, trace, *, events=None) -> PrimFunction:
    """Return 'function' with the completion statements appended INSIDE its mesh scope.

    'events' is the EventAssignment; when supplied, each record's flag id is taken from it so
    the emitted program and the allocation cannot silently disagree.
    """
    allocation = None
    if events is not None:
        allocation = {item.channel: item.event_id for item in events.allocations}
    records = []
    for record in trace.records:
        if allocation and record.channel in allocation:
            record = record.with_event(allocation[record.channel])
        elif allocation and record.role in (ROLE_SET, ROLE_WAIT):
            # A traced channel with no allocation is a real gap: inventing an id here would
            # hide an exhausted/unallocated namespace.
            keyed = [name for name in allocation
                     if name.split("#")[0] == record.channel]
            if not keyed:
                raise CompletionError(
                    "COMPLETION_CHANNEL_UNBOUND",
                    "the completion trace contains a channel with no physical event"
                    " allocation, so it cannot be emitted",
                    record.channel, "an allocated event id", None)
            record = record.with_event(allocation["%s#s%d" % (record.channel, record.slot)])
        records.append(record)
    resolved = replace(trace, records=tuple(records))
    nodes = trace_nodes(resolved)
    if not nodes:
        return function, resolved
    scope = _mesh_scopes(function)
    if scope is None:
        # No mesh scope: appending at function top level is the only structurally valid
        # placement left, and it is recorded by being what it is (the drain is function-level).
        body = Sequential(body=tuple(function.body.body) + tuple(nodes))
        return replace(function, body=body), resolved
    path, _kind = scope
    indices = [int(part[1:]) for part in path.split("/") if part.startswith("s")]
    if not indices:
        body = Sequential(body=tuple(function.body.body) + tuple(nodes))
        return replace(function, body=body), resolved

    def insert(node, remaining):
        if not remaining:
            if isinstance(node, Sequential):
                return Sequential(body=tuple(node.body) + tuple(nodes))
            body = getattr(node, "body", None)
            if isinstance(body, Sequential):
                return replace(node, body=Sequential(body=tuple(body.body) + tuple(nodes)))
            return node
        index = remaining[0]
        if isinstance(node, Sequential):
            children = list(node.body)
            children[index] = insert(children[index], remaining[1:])
            return Sequential(body=tuple(children))
        body = getattr(node, "body", None)
        if isinstance(body, Sequential):
            return replace(node, body=insert(body, remaining))
        return node

    return replace(function, body=insert(function.body, indices)), resolved


def read_trace_from_tir(function) -> list:
    """Read the completion statements BACK out of a finished TIR, in tree order.

    The reader is deliberately independent of the trace that produced them: it walks the real
    tree, honours reachability (a Set inside a statically false arm cannot execute and is not
    coverage), and returns records with the ATTRIBUTES THE PROGRAM ACTUALLY CARRIES. A verifier
    built on this cannot be satisfied by a map or a flag.
    """
    from .guards import ALWAYS, NEVER

    records = []
    counter = {"n": 0}

    def walk(node, reachable=True, guard=None):
        if node is None:
            return
        if isinstance(node, If):
            cond = getattr(node, "cond", None)
            static = _static_bool(cond)
            then_guard = _guard_of(cond, guard)
            if static is False:
                walk(node.else_body, reachable, guard)
            elif static is True:
                walk(node.then_body, reachable, guard)
            else:
                walk(node.then_body, reachable, then_guard)
                walk(node.else_body, reachable, guard)
            return
        if isinstance(node, Sequential):
            for child in node.body:
                walk(child, reachable, guard)
            return
        if isinstance(node, (MeshScope, ExecutionScope, For)):
            body = getattr(node, "body", None)
            if isinstance(node, For) and _trip(node) == 0:
                walk(body, False, guard)
            else:
                walk(body, reachable, guard)
            return
        if isinstance(node, Evaluate):
            target = node.callable
            if isinstance(target, FlagSet):
                records.append(_record(counter, "set", target, reachable, guard))
            elif isinstance(target, FlagWait):
                records.append(_record(counter, "wait", target, reachable, guard))
            elif isinstance(target, Drain):
                records.append(_record(counter, "drain", target, reachable, guard))
            return

    walk(function.body)
    return records


def _record(counter, role, target, reachable, guard):
    record = TraceRecord(
        index=counter["n"], role=role,
        channel=str(getattr(target, "channel_name", "")),
        obligation_id="", kind=str(getattr(target, "kind", "")),
        generation=int(getattr(target, "generation", -1)),
        slot=int(getattr(target, "slot", -1)),
        versions=int(getattr(target, "versions", 1)),
        task="", core="", pipe="",
        producer_pipe=str(getattr(target, "producer_pipe", "")),
        consumer_pipe=str(getattr(target, "consumer_pipe", "")),
        guard=guard, event_id=int(getattr(target, "flag_id", -1)),
        sweep=0, issue_order=counter["n"])
    counter["n"] += 1
    return _Annotated(record, reachable)


class _Annotated:
    """A read-back record plus the two facts only the READER can know."""

    __slots__ = ("record", "reachable")

    def __init__(self, record, reachable):
        self.record = record
        self.reachable = reachable

    def __getattr__(self, name):
        return getattr(self.record, name)

    def as_dict(self):
        return {**self.record.as_dict(), "reachable": self.reachable}


def _static_bool(cond):
    value = getattr(cond, "value", None)
    if isinstance(value, bool):
        return value
    return None


def _guard_of(cond, outer):
    from .guards import Guard

    static = _static_bool(cond)
    if static is True:
        return outer
    if static is False:
        return Guard(NEVER, key="never")
    return Guard(COND, key=_key(cond), source=cond)


def _key(cond):
    from .guards import guard_key

    return guard_key(cond)


def _trip(loop):
    values = []
    for bound in (getattr(loop, "start", None), getattr(loop, "stop", None),
                  getattr(loop, "step", None)):
        value = getattr(bound, "value", None)
        if not isinstance(value, int):
            return None
        values.append(value)
    start, stop, step = values
    if step <= 0:
        return 0
    return max(0, (stop - start + step - 1) // step)


__all__ = ["trace_nodes", "emit_trace_tir", "read_trace_from_tir"]
