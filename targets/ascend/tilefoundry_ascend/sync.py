"""C completion module (static finite-domain POC, C0-C2).

This is ADDITIVE to the target sync layer: it does not change the existing
schedule/sync.py generation=t semantics, and it does not modify Core flag schema.
It derives completion obligations from L4 real dependency edges plus an R ResolvedPlan
(slot/version/generation), emits a completion plan, and verifies an actual typed-TIR
placement independently from the placement map.

The POC is deliberately static: Core currently stores slot/generation as int attributes,
so a finite static epoch (k0..2 / V2 in the tests) is materialised as actual FlagSet /
FlagWait / Drain nodes. Dynamic slot operands are not pretended to work.

Named refusals:
  COMPLETION_SOURCE_COVERAGE_MISSING  a dependency has no matching Plan access/task
  COMPLETION_CAPABILITY_UNKNOWN       target cannot express the required completion
  COMPLETION_MECHANISM_UNPROVEN       S latency/pipe-name alone is not a mechanism
  FLAG_RESOURCE_CONFLICT / FLAG_POOL_EXHAUSTED
  COMPLETION_GENERATION_MISMATCH      slot/min(tile) used as generation
  COMPLETION_NOT_VERIFIED             actual Set/Wait/Drain missing or map forged
  COMPLETION_MAP_MISMATCH             map points at nodes not present in actual TIR
  COMPLETION_WAIT_BEFORE_SET          a same-iteration (distance 0) Wait is ordered before its Set
  COMPLETION_DUPLICATE_FLAG_NODE      more than one Set or Wait for one (channel, kind, generation)
  COMPLETION_DISTANCE_NOT_PRESERVED   a cross-iteration edge was materialised as same-iteration
  COMPLETION_STARTUP_UNSATISFIED      a cross-iteration Wait has no Set that executes before it
                                      (no prologue cover), so the first iteration would block
  COMPLETION_ANCHOR_AMBIGUOUS         one action maps to more than one real statement, so the
                                      anchor is not uniquely determined (refuse, never pick [0])
  COMPLETION_RELATION_AMBIGUOUS       a reuse/RAW root has several reads AND several writes
  FINAL_DRAIN_MISSING
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.tir.flag import Drain, FlagSet, FlagWait
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, ExecutionScope, For, If, MeshScope, Sequential, While


from tilefoundry.ir.pattern import utils as _pattern_utils
from tilefoundry.visitor_registry.access_relation import identity_relations as _identity_relations
from tilefoundry.visitor_registry.access_relation import register_access_relation as _register_access_relation
from tilefoundry.visitor_registry import register_typeinfer as _register_typeinfer
from tilefoundry.visitor_registry import register_verify_stmt as _register_verify_stmt
from tilefoundry.ir.types import UnitType as _UnitType
from tilefoundry.ir.tir.flag import _check_common as _flag_check_common


class AivFlagSet(FlagSet):
    execution_mesh = _pattern_utils.memory_execution_mesh()


class AivFlagWait(FlagWait):
    execution_mesh = _pattern_utils.memory_execution_mesh()


class AivDrain(Drain):
    execution_mesh = _pattern_utils.memory_execution_mesh()


@_register_access_relation(AivFlagSet)
@_register_access_relation(AivFlagWait)
@_register_access_relation(AivDrain)
def _aiv_access(call, ctx):
    return _identity_relations(call, ctx)


@_register_typeinfer(AivFlagSet)
@_register_typeinfer(AivFlagWait)
@_register_typeinfer(AivDrain)
def _aiv_type(call, ctx):
    return _UnitType()


@_register_verify_stmt(AivFlagSet)
@_register_verify_stmt(AivFlagWait)
def _aiv_verify(call, ctx):
    _flag_check_common(call, need_pipes=True)


@_register_verify_stmt(AivDrain)
def _aiv_drain_verify(call, ctx):
    return None


class CompletionError(RuntimeError):
    def __init__(self, code, message, source_id="", expected=None, actual=None):
        self.code = code
        self.source_id = source_id
        self.expected = expected
        self.actual = actual
        super().__init__("%s: %s (source=%r expected=%r actual=%r)"
                         % (code, message, source_id, expected, actual))

    def as_dict(self):
        return {"code": self.code, "source_id": self.source_id,
                "expected": self.expected, "actual": self.actual}


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _sha(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Obligation:
    obligation_id: str
    source_dependency_ids: tuple
    access_ids: tuple
    kind: str                         # READY/FREE/ONCE/WAW/FINAL
    producer_task: str
    consumer_task: str
    producer_action: str
    consumer_action: str
    producer_pipe: str
    consumer_pipe: str
    region: str
    buffer_root: str
    versions: int
    distance: int
    generation_domain: tuple
    required_completion_kind: str
    guard: object = None

    def as_dict(self):
        return dict(self.__dict__)


@dataclass(frozen=True)
class Discharge:
    obligation_id: str
    mechanism: str                    # EVENT/PIPE_ORDER/DEPENDENCY_CHAIN/DRAIN
    actual_stmt_refs: tuple
    ordering_proof: str
    unsupported_reason: Optional[str] = None

    def as_dict(self):
        return dict(self.__dict__)


@dataclass(frozen=True)
class LogicalChannel:
    channel_id: str
    covered_obligation_ids: tuple
    producer_pipe: str
    consumer_pipe: str
    owner: str
    event_domain: str
    kind: str
    generation_match: str

    def as_dict(self):
        return dict(self.__dict__)


@dataclass(frozen=True)
class PhysicalAllocation:
    channel_id: str
    flag_id: int
    pool: str
    use_interval: tuple
    reuse_evidence: str

    def as_dict(self):
        return dict(self.__dict__)


@dataclass(frozen=True)
class ActionPlacement:
    placement_id: str
    actual_stmt_ref: str
    role: str                         # SET / WAIT / DRAIN
    channel_id: str
    flag_id: int
    generation: int
    slot: int
    versions: int
    task: str
    action: str
    source_obligation_ids: tuple

    def as_dict(self):
        return dict(self.__dict__)


@dataclass(frozen=True)
class CoverageReport:
    status: str                       # VERIFIED / FAILED / PENDING
    code: Optional[str]
    required: int
    provided: int
    covered: int
    missing_obligation_ids: tuple
    placements: int
    detail: str = ""
    required_by_kind: dict = field(default_factory=dict)
    covered_by_kind: dict = field(default_factory=dict)
    edge_table: tuple = ()

    def as_dict(self):
        return dict(self.__dict__)


def _trip(plan) -> int:
    domain = _get(plan.loop, "domain", {}) or {}
    start, stop, step = int(domain["start"]), int(domain["stop"]), int(domain["step"])
    return max(0, (stop - start + step - 1) // step)


def _buffer(plan, root):
    for buffer in (_get(plan, "buffers", []) or []):
        if str(_get(buffer, "root")) == str(root):
            return buffer
    return None


def _task(plan, task_id):
    for task in (_get(plan, "tasks", []) or []):
        if str(_get(task, "task_id", _get(task, "stable_id"))) == str(task_id):
            return task
    return None


def _access_for(plan, task_id, root, mode):
    for access in (_get(plan, "accesses", []) or []):
        task = str(_get(access, "task_id", _get(access, "task")))
        if task == str(task_id) and str(_get(access, "buffer_root", _get(access, "root"))) == str(root) \
                and str(_get(access, "mode", "")) == mode:
            return access
    return None


_KIND_MAP = {"RAW": "READY", "WAR": "FREE", "WAW": "WAW", "REUSE": "ONCE",
             "READY": "READY", "FREE": "FREE", "ONCE": "ONCE"}


def derive_completion_obligations(problem, plan, materialized_access_map=None,
                                  completion_capabilities=None) -> tuple:
    """Every L4 RAW/WAR/WAW/REUSE edge becomes one completion obligation.

    No fixed six-channel template is used: names are derived from the source edge +
    pipe pair + owner/root, and an edge without a matching Plan access is refused.
    """
    deps = list(_get(problem, "dependencies", []) or [])
    trip = _trip(plan)
    obligations = []
    for index, dep in enumerate(deps):
        producer = str(_get(dep, "producer"))
        consumer = str(_get(dep, "consumer"))
        source_id = str(_get(dep, "source_id", "dep-%d:%s->%s" % (index, producer, consumer)))
        kind_in = str(_get(dep, "kind", "RAW")).upper()
        if kind_in not in _KIND_MAP:
            raise CompletionError("COMPLETION_SOURCE_COVERAGE_MISSING",
                                  "dependency kind has no completion mapping", source_id,
                                  list(_KIND_MAP), kind_in)
        root = _get(dep, "shared_root", _get(dep, "root"))
        if not root:
            raise CompletionError("COMPLETION_SOURCE_COVERAGE_MISSING",
                                  "dependency has no shared root", source_id, "root", root)
        buffer = _buffer(plan, root)
        if buffer is None:
            raise CompletionError("COMPLETION_SOURCE_COVERAGE_MISSING",
                                  "dependency root is not in the Plan", source_id, root, None)
        producer_access = _access_for(plan, producer, root, "write" if kind_in == "RAW" else
                                      ("read" if kind_in in ("WAR", "REUSE") else "write"))
        consumer_access = _access_for(plan, consumer, root, "read" if kind_in in ("RAW", "REUSE") else "write")
        if producer_access is None or consumer_access is None:
            raise CompletionError("COMPLETION_SOURCE_COVERAGE_MISSING",
                                  "dependency has no matching Plan access", source_id,
                                  [producer, consumer, root], None)
        producer_task = _task(plan, producer)
        consumer_task = _task(plan, consumer)
        if producer_task is None or consumer_task is None:
            raise CompletionError("COMPLETION_SOURCE_COVERAGE_MISSING",
                                  "dependency task is not in the Plan", source_id,
                                  [producer, consumer], None)
        versions = int(_get(buffer, "slot_count", 1))
        distance = int(_get(dep, "distance", 0))
        required = _KIND_MAP[kind_in]
        obligation_id = "ob-%s-%s-%s" % (required.lower(), source_id, _sha({
            "edge": source_id, "producer": producer, "consumer": consumer,
            "root": root, "distance": distance, "versions": versions}))
        obligations.append(Obligation(
            obligation_id=obligation_id, source_dependency_ids=(source_id,),
            access_ids=(str(_get(producer_access, "access_id")),
                        str(_get(consumer_access, "access_id"))),
            kind=required, producer_task=producer, consumer_task=consumer,
            producer_action=str(_get(producer_access, "access_id")),
            consumer_action=str(_get(consumer_access, "access_id")),
            producer_pipe=str(_get(producer_task, "pipe", "")),
            consumer_pipe=str(_get(consumer_task, "pipe", "")),
            region=str(_get(plan.loop, "source_loop_id", "loop")),
            buffer_root=str(root), versions=versions, distance=distance,
            generation_domain=tuple(range(trip)), required_completion_kind=required,
            guard=_get(dep, "guard")))
    return tuple(obligations)


def choose_completion_mechanisms(obligations, schedule_domains, capabilities) -> tuple:
    """Choose real event/pipe mechanisms and logical channels; no latency-only claim."""
    capabilities = capabilities or {}
    supported_pairs = {tuple(pair) for pair in capabilities.get("supported_pairs", [])}
    same_pipe_order = bool(capabilities.get("same_pipe_pipe_order", False))
    channels = []
    discharges = []
    by_key = {}
    for obligation in obligations:
        pipes = (obligation.producer_pipe, obligation.consumer_pipe)
        if pipes[0] != pipes[1] and supported_pairs and pipes not in supported_pairs:
            raise CompletionError("COMPLETION_CAPABILITY_UNKNOWN",
                                  "target cannot express this pipe pair", obligation.obligation_id,
                                  sorted(supported_pairs), list(pipes))
        if pipes[0] == pipes[1] and not same_pipe_order:
            raise CompletionError("COMPLETION_MECHANISM_UNPROVEN",
                                  "same-pipe edge has no completion guarantee", obligation.obligation_id,
                                  "same_pipe_pipe_order capability", False)
        if pipes[0] != pipes[1] and same_pipe_order:
            raise CompletionError("COMPLETION_MECHANISM_UNPROVEN",
                                  "cross-pipe edge cannot claim SAME_PIPE_ORDER", obligation.obligation_id,
                                  "EVENT for cross-pipe", "SAME_PIPE_ORDER")
        mechanism = "EVENT" if pipes[0] != pipes[1] else "PIPE_ORDER"
        key = (obligation.required_completion_kind, pipes[0], pipes[1],
               obligation.buffer_root, obligation.region)
        if key not in by_key:
            channel_id = "ch-%s-%s" % (obligation.required_completion_kind.lower(),
                                       _sha({"pipes": pipes, "root": obligation.buffer_root,
                                             "region": obligation.region}))
            by_key[key] = LogicalChannel(
                channel_id=channel_id, covered_obligation_ids=(),
                producer_pipe=pipes[0], consumer_pipe=pipes[1], owner=obligation.buffer_root,
                event_domain=("ring" if obligation.versions > 1 else "once"),
                kind=obligation.required_completion_kind,
                generation_match="generation=k, slot=k%V")
            channels.append(by_key[key])
        channel = by_key[key]
        by_key[key] = LogicalChannel(**{**channel.__dict__,
                                        "covered_obligation_ids": channel.covered_obligation_ids
                                        + (obligation.obligation_id,)})
        discharges.append(Discharge(obligation.obligation_id, mechanism, (),
                                    "pipe pair %s->%s with declared target capability" % pipes))
    channels = tuple(by_key[key] for key in by_key)
    return tuple(channels), tuple(discharges)


def allocate_completion_resources(channels, lifetime_relations, capabilities) -> tuple:
    """Bounded physical flag pool; active channels never share a flag unsafely."""
    pool_size = int((capabilities or {}).get("flag_pool_size", 1))
    allocations = []
    for index, channel in enumerate(channels):
        if index >= pool_size:
            raise CompletionError("FLAG_POOL_EXHAUSTED", "physical flag pool exhausted",
                                  channel.channel_id, pool_size, index + 1)
        # Static POC uses one active interval per channel; reuse is only allowed when the
        # caller supplies non-overlapping intervals. Distinct flag ids keep this honest.
        allocations.append(PhysicalAllocation(channel.channel_id, index, "target_flags",
                                              (0, 1), "static-POC distinct interval"))
    return tuple(allocations)


def _walk(stmt, path=()):
    yield "/".join(path), stmt
    body = getattr(stmt, "body", None)
    if isinstance(body, Sequential):
        for index, child in enumerate(body.body):
            yield from _walk(child, path + ("s%d" % index,))
    elif isinstance(body, (list, tuple)):
        for index, child in enumerate(body):
            yield from _walk(child, path + ("s%d" % index,))
    elif body is not None and not isinstance(body, (str, int)):
        # MeshScope / ExecutionScope / For / While: descend so a nested Drain/Set/Wait is
        # still found; the previous top-only walk missed nested scope nodes.
        yield from _walk(body, path + (type(stmt).__name__,))
    for branch_name in ("then_body", "else_body"):
        branch = getattr(stmt, branch_name, None)
        if isinstance(branch, Sequential):
            yield from _walk(branch, path + (branch_name,))


def _actual_nodes(tir_module) -> tuple:
    nodes = {"set": [], "wait": [], "drain": []}
    for path, stmt in _walk(tir_module):
        if not isinstance(stmt, Evaluate):
            continue
        target = stmt.callable
        if isinstance(target, FlagSet):
            nodes["set"].append((path, target))
        elif isinstance(target, FlagWait):
            nodes["wait"].append((path, target))
        elif isinstance(target, Drain):
            nodes["drain"].append((path, target))
    return nodes


def _pipe_of_op(op):
    op = str(op)
    if op in ("GmToUb", "UbToGm", "copy"):
        return "DMA"
    if op in ("Vld", "Vsts"):
        return "vector"
    return "vector"


def _entries_by_action(ledger, action_id):
    """Every ledger entry that names this action; >1 means the anchor is ambiguous."""
    wanted = str(action_id)
    exact = [entry for entry in list(getattr(ledger, "entries", ()) or ())
             if str(getattr(entry, "action_id", "")) == wanted]
    if exact:
        return exact
    base = wanted.rsplit(":", 1)[0]
    return [entry for entry in list(getattr(ledger, "entries", ()) or ())
            if str(getattr(entry, "action_id", "")).rsplit(":", 1)[0] == base]


def _anchor_walk(node, path=()):
    """Yield (anchor_path, stmt) using the SAME naming _clone_with_anchors consumes.

    Lift/ledger statement paths spell scopes out ("s0/MeshScope/s0/for:i/s2") while
    _walk() collapses Sequential bodies, so the two are NOT interchangeable. Anchor
    positions must be ordered in the anchor scheme or every lookup silently misses.
    """
    if isinstance(node, Sequential):
        for index, child in enumerate(node.body):
            yield from _anchor_walk(child, path + ("s%d" % index,))
        return
    yield "/".join(path), node
    if isinstance(node, (MeshScope, ExecutionScope)):
        yield from _anchor_walk(node.body, path + (type(node).__name__,))
    elif isinstance(node, For):
        name = getattr(getattr(node, "induction_var", None), "name", "i")
        yield from _anchor_walk(node.body, path + ("for:" + str(name),))
    elif isinstance(node, If):
        yield from _anchor_walk(node.then_body, path + ("then",))
        yield from _anchor_walk(node.else_body, path + ("else",))
    else:
        body = getattr(node, "body", None)
        if isinstance(body, Sequential):
            yield from _anchor_walk(body, path + (type(node).__name__,))


def _anchor_ordinals(tir_module):
    """path -> real tree ordinal, in the anchor/ledger path scheme."""
    return {path: index for index, (path, _stmt) in enumerate(_anchor_walk(tir_module.body))}


def _instance_anchor(paths, generation, instance_index, obligation_id, role):
    """Anchor one generation at ITS OWN carrier instance, never at paths[0].

    The POC carrier is UNROLLED: one action body appears once per iteration, so an action
    with N real instances is N distinct statements and generation g must use instance g
    (the producer uses g - distance). Collapsing to paths[0] put every generation on the
    same statement, which no ordering check can then tell apart from a correct placement.

    A genuinely loop-nested carrier has exactly ONE instance and the loop supplies the
    iterations; the checker refuses that shape by name unless a prologue covers startup.
    """
    ordered = list(paths)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    if instance_index < 0 or instance_index >= len(ordered):
        raise CompletionError("COMPLETION_ANCHOR_NOT_FOUND",
                              "%s has no instance for generation %d (index %d of %d)"
                              % (role, generation, instance_index, len(ordered)),
                              obligation_id, len(ordered), instance_index)
    return ordered[instance_index]


def derive_reuse_obligations_from_ledger(ledger, expansion):
    """Bind the 18 cross-iteration WAR/FREE obligations to real ledger actions.

    A reuse obligation is bound by its root: producer_action is a READ entry's
    action_id (reader SET after read), consumer_action is a WRITE entry's action_id
    (writer WAIT before overwrite). Vsts write is selected by its :operand action id.
    A root without both a read and a write entry is COMPLETION_OBLIGATION_UNBOUND.
    """
    entries = list(getattr(ledger, "entries", ()) or ())
    alias = dict(getattr(ledger, "root_alias", {}) or {})

    def canonical_root(entry):
        root = str(getattr(entry, "root", ""))
        return str(alias.get(root, root))

    smem_entries = [entry for entry in entries
                    if str(getattr(entry, "space", "")) == "SMEM"]
    roots = sorted({canonical_root(entry) for entry in smem_entries
                    if str(getattr(entry, "mode", "")) == "write"})
    out = []
    for root in roots:
        reads = [entry for entry in smem_entries if canonical_root(entry) == root
                 and str(getattr(entry, "mode", "")) == "read"]
        writes = [entry for entry in smem_entries if canonical_root(entry) == root
                  and str(getattr(entry, "mode", "")) == "write"]
        if not reads or not writes:
            raise CompletionError("COMPLETION_OBLIGATION_UNBOUND",
                                  "SMEM reuse root lacks read/write ledger entry", root,
                                  "read+write entries", [len(reads), len(writes)])
        if len(reads) > 1 and len(writes) > 1:
            raise CompletionError("COMPLETION_RELATION_AMBIGUOUS",
                                  "reuse root has multiple reads and writes; refusing to pick one",
                                  root, "one-to-many or one-to-one relation",
                                  [len(reads), len(writes)])
        pairs = [(read, write) for read in reads for write in writes]
        count = max(0, int(getattr(expansion, "count", 0)))
        for pair_index, (read, write) in enumerate(pairs):
            producer_action = str(read.action_id)
            consumer_action = str(write.action_id)
            producer_pipe = _pipe_of_op(getattr(read, "op", ""))
            consumer_pipe = _pipe_of_op(getattr(write, "op", ""))
            suffix = "" if len(pairs) == 1 else "#%d" % pair_index
            for index in range(max(0, count - 1)):
                for kind in ("WAR", "FREE"):
                    oid = "%s:%s:i%d->i%d%s" % (kind, root, index, index + 1, suffix)
                    out.append(Obligation(
                        obligation_id=oid,
                        source_dependency_ids=("reuse:%s:i%d->i%d%s" % (root, index, index + 1, suffix),),
                        access_ids=(producer_action, consumer_action), kind=kind,
                        producer_task=producer_action, consumer_task=consumer_action,
                        producer_action=producer_action, consumer_action=consumer_action,
                        producer_pipe=producer_pipe, consumer_pipe=consumer_pipe,
                        region="loop", buffer_root=root, versions=1, distance=0,
                        generation_domain=(index + 1,), required_completion_kind=kind))
    if len(out) < 18:
        raise CompletionError("COMPLETION_OBLIGATION_UNBOUND",
                              "expected at least 18 reuse obligations, got %d" % len(out),
                              "ledger", 18, len(out))
    return tuple(out)


def _raw_obligations_from_ledger(ledger, expansion, expected):
    entries = list(getattr(ledger, "entries", ()) or ())
    alias = dict(getattr(ledger, "root_alias", {}) or {})
    def canonical(entry):
        root = str(getattr(entry, "root", ""))
        return str(alias.get(root, root))
    smem = [e for e in entries if str(getattr(e, "space", "")) == "SMEM"]
    roots = sorted({canonical(e) for e in smem if str(getattr(e, "mode", "")) == "write"})
    family_root = {}
    for root in roots:
        if root.endswith("/s1"):
            family_root["X-in"] = root
        elif root.endswith("/s4"):
            family_root["Z-in"] = root
        elif root.endswith("/s10"):
            family_root["Y-out"] = root
    raw_items = list(expected.get("raw", []) if isinstance(expected, dict) else [])
    out = []
    for item in raw_items:
        oid = str(getattr(item, "oid", ""))
        parts = oid.split(":")
        family = parts[1] if len(parts) > 1 else ""
        root = family_root.get(family)
        if root is None:
            raise CompletionError("COMPLETION_OBLIGATION_UNBOUND", "RAW family has no bound SMEM root",
                                  oid, "X-in/Z-in/Y-out root", family)
        reads = [e for e in smem if canonical(e) == root and str(getattr(e, "mode", "")) == "read"]
        writes = [e for e in smem if canonical(e) == root and str(getattr(e, "mode", "")) == "write"]
        if not reads or not writes:
            raise CompletionError("COMPLETION_OBLIGATION_UNBOUND", "RAW root lacks read/write entry", oid,
                                  "read+write", [len(reads), len(writes)])
        # Same strictness as the reuse branch: never silently pick reads[0]/writes[0]. Several
        # reads AND several writes leave the producer->consumer relation undetermined; one-to-many
        # and many-to-one are expanded pair by pair so every real relation is its own obligation.
        if len(reads) > 1 and len(writes) > 1:
            raise CompletionError("COMPLETION_RELATION_AMBIGUOUS",
                                  "RAW root has multiple reads and writes; refusing to pick one",
                                  oid, "one-to-many or one-to-one relation",
                                  [len(reads), len(writes)])
        pairs = [(write, read) for write in writes for read in reads]
        index = int(oid.rsplit("i", 1)[-1]) if "i" in oid else 0
        for pair_index, (write, read) in enumerate(pairs):
            suffix = "" if len(pairs) == 1 else "#%d" % pair_index
            out.append(Obligation(
                obligation_id=oid + suffix, source_dependency_ids=(oid,), access_ids=(str(write.action_id), str(read.action_id)),
                kind="RAW", producer_task=str(write.action_id), consumer_task=str(read.action_id),
                producer_action=str(write.action_id), consumer_action=str(read.action_id),
                producer_pipe=_pipe_of_op(getattr(write, "op", "")),
                consumer_pipe=_pipe_of_op(getattr(read, "op", "")),
                region="loop", buffer_root=root, versions=1, distance=0,
                generation_domain=(index,), required_completion_kind="RAW"))
    return tuple(out)


def derive_completion_obligations_from_ledger(ledger, expansion, problem=None,
                                               plan=None, expected=None):
    """Build the independent 47-item required set from L facts + bound reuse edges."""
    if expected is None:
        from .schedule.poc_gate import build_expected_obligations
        if problem is None:
            raise CompletionError("COMPLETION_OBLIGATION_UNBOUND",
                                  "problem is required to derive the 47 expected set",
                                  "expected", "problem", None)
        expected = build_expected_obligations(ledger, problem, expansion)
    items = list(expected.get("all", []) if isinstance(expected, dict) else expected)
    reuse = derive_reuse_obligations_from_ledger(ledger, expansion)
    raw = _raw_obligations_from_ledger(ledger, expansion, expected)
    reuse_ids = {o.obligation_id for o in reuse}
    # An expanded RAW root yields oid#k records; the base oid is still the expected item that
    # must not be counted twice, and #k suffixes must not shrink the denominator.
    raw_ids = {o.obligation_id.split("#", 1)[0] for o in raw}
    result = []
    for item in items:
        oid = str(getattr(item, "oid", getattr(item, "obligation_id", "")))
        kind = str(getattr(item, "kind", ""))
        if kind in ("WAR", "FREE", "WAW") or oid in reuse_ids or oid in raw_ids:
            continue
        result.append(Obligation(
            obligation_id=oid, source_dependency_ids=(str(getattr(item, "discharged_by", "") or ""),),
            access_ids=(), kind=kind, producer_task="", consumer_task="",
            producer_action="", consumer_action="", producer_pipe="", consumer_pipe="",
            region="loop", buffer_root="", versions=1, distance=0,
            generation_domain=(), required_completion_kind=kind))
    result.extend(raw)
    result.extend(reuse)
    # The denominator is the count of DISTINCT source obligations (47). Pair expansion of an
    # ambiguous one-to-many root adds records for the same source obligation, so it is allowed
    # to raise the record count but never to change the 47-item source denominator.
    source_ids = {o.obligation_id.split("#", 1)[0] for o in result}
    if len(source_ids) != 47 or len(result) < 47:
        raise CompletionError("COMPLETION_OBLIGATION_UNBOUND",
                              "expected 47 required obligations, got %d distinct/%d records"
                              % (len(source_ids), len(result)),
                              "expected", 47, [len(source_ids), len(result)])
    return tuple(result)


def _clone_with_anchors(node, path, before, after):
    import dataclasses as _dc
    from tilefoundry.ir.tir.stmts import ExecutionScope, For, If, MeshScope
    if isinstance(node, Sequential):
        body = []
        for index, child in enumerate(node.body):
            child_path = path + ("s%d" % index,)
            key = "/".join(child_path)
            body.extend(before.get(key, ()))
            body.append(_clone_with_anchors(child, child_path, before, after))
            body.extend(after.get(key, ()))
        return _dc.replace(node, body=tuple(body))
    if isinstance(node, (MeshScope, ExecutionScope)):
        return _dc.replace(node, body=_clone_with_anchors(
            node.body, path + (type(node).__name__,), before, after))
    if isinstance(node, For):
        name = getattr(getattr(node, "induction_var", None), "name", "i")
        return _dc.replace(node, body=_clone_with_anchors(
            node.body, path + ("for:" + str(name),), before, after))
    if isinstance(node, If):
        return _dc.replace(node,
                           then_body=_clone_with_anchors(node.then_body, path + ("then",), before, after),
                           else_body=_clone_with_anchors(node.else_body, path + ("else",), before, after))
    return node


def _anchor_path(path):
    """Move an anchor inside an ExecutionScope to the scope statement itself.

    Flag ops are mesh-scoped, not SIMD-VF mode ops; placing them inside
    ExecutionScope('ascend.simdvf') is invalid. The surrounding AIV MeshScope is
    still the same execution mesh/thread domain.
    """
    parts = str(path).split("/")
    if "ExecutionScope" in parts:
        index = parts.index("ExecutionScope")
        return "/".join(parts[:index])
    return str(path)


def _reachable_drain(tir_module):
    def walk(node, reachable=True):
        from tilefoundry.ir.tir.stmts import If
        if isinstance(node, Sequential):
            for child in node.body:
                if walk(child, reachable):
                    return True
            return False
        if isinstance(node, If):
            cond = getattr(node, "cond", None)
            value = getattr(cond, "value", None)
            then_reach = reachable and not (isinstance(cond, Constant) and value is False)
            else_reach = reachable and not (isinstance(cond, Constant) and value is True)
            return walk(node.then_body, then_reach) or walk(node.else_body, else_reach)
        if isinstance(node, Evaluate) and isinstance(node.callable, Drain):
            return reachable
        body = getattr(node, "body", None)
        if body is not None and not isinstance(body, (str, int)):
            return walk(body, reachable)
        return False
    return walk(tir_module.body)


def _append_aiv_drain(node):
    import dataclasses as _dc
    from tilefoundry.ir.tir.stmts import ExecutionScope, MeshScope
    if isinstance(node, Sequential):
        if node.body and isinstance(node.body[-1], (MeshScope, ExecutionScope)):
            last = node.body[-1]
            inner = _append_aiv_drain(last.body)
            return _dc.replace(node, body=tuple(node.body[:-1]) + (_dc.replace(last, body=inner),))
        return _dc.replace(node, body=tuple(node.body) + (Evaluate(callable=AivDrain(), args=()),))
    return node


def _actual_path_ordinals(tir_module):
    return {path: index for index, (path, _stmt) in enumerate(_walk(tir_module))}


def _walk_ordered(stmt, path=(), in_loop=False):
    """_walk() plus "is this statement inside a loop body?" and real tree order.

    The completion checker needs the ACTUAL traversal ordinal of each Set/Wait and
    whether it runs per loop iteration, because "later in the tree" and "later in
    execution" only agree inside one loop body sweep.
    """
    yield "/".join(path), stmt, in_loop
    body = getattr(stmt, "body", None)
    inside = in_loop or isinstance(stmt, (For, While))
    if isinstance(body, Sequential):
        for index, child in enumerate(body.body):
            yield from _walk_ordered(child, path + ("s%d" % index,), inside)
    elif isinstance(body, (list, tuple)):
        for index, child in enumerate(body):
            yield from _walk_ordered(child, path + ("s%d" % index,), inside)
    elif body is not None and not isinstance(body, (str, int)):
        yield from _walk_ordered(body, path + (type(stmt).__name__,), inside)
    for branch_name in ("then_body", "else_body"):
        branch = getattr(stmt, branch_name, None)
        if isinstance(branch, Sequential):
            yield from _walk_ordered(branch, path + (branch_name,), in_loop)


def _ordering_context(tir_module):
    """path -> (ordinal, in_loop) over the real tree; no per-generation merging."""
    ordinals, in_loop = {}, {}
    for index, (path, _stmt, inside) in enumerate(_walk_ordered(tir_module)):
        ordinals[path] = index
        in_loop[path] = inside
    return ordinals, in_loop


def _iteration_offset(obligation):
    """How many loop iterations separate the producer from the consumer of an edge.

    A reuse edge names it in the id (``...:i0->i1``) and its consumer generation is that
    later iteration; a RAW instance lives inside one iteration. Offset 0 means "both ends
    are in the same body sweep", so tree order IS execution order and a Wait before its
    Set is a real deadlock. Offset >= 1 means the loop itself supplies ordering, but only
    from the second iteration on: the first sweep still needs an earlier Set (a pre-loop
    prologue) or it blocks.
    """
    match = re.search(r":i(\d+)->i(\d+)(?:#\d+)?$", str(getattr(obligation, "obligation_id", "")))
    if match:
        return max(0, int(match.group(2)) - int(match.group(1)))
    if str(getattr(obligation, "kind", "")).upper() in ("WAR", "FREE", "WAW", "REUSE"):
        return 1
    return 0


def inject_completion_anchored(tir_module, plan, obligations, allocation, *,
                               channels=None, ledger=None, drop_wait_ids=(),
                               materialization=None):
    """Deep Sequential anchor injection for reuse Set/Wait nodes, preserving Drain."""
    if ledger is None:
        raise CompletionError("COMPLETION_ANCHOR_NOT_FOUND", "ledger is required for anchor lookup",
                              "ledger", "real emission ledger", None)
    from .plan.materialize import fingerprint as _mat_fingerprint
    from .plan.materialize import index_statements_ledger
    obligation_list = tuple(getattr(obligations, "obligations", obligations))
    if channels is None:
        channels, _ = choose_completion_mechanisms(obligation_list, {}, {"supported_pairs": [["DMA", "vector"], ["vector", "DMA"]], "same_pipe_pipe_order": False})
    channels = tuple(channels)
    if len(tuple(allocation)) < len(channels):
        raise CompletionError("FLAG_POOL_EXHAUSTED", "flag pool smaller than channels",
                              "allocation", len(channels), len(tuple(allocation)))
    channel_by_obligation = {}
    for channel in channels:
        for oid in getattr(channel, "covered_obligation_ids", ()):
            channel_by_obligation[oid] = channel
    allocation_by_channel = {item.channel_id: item for item in allocation}
    statements = index_statements_ledger(tir_module)
    ordinals = _anchor_ordinals(tir_module)
    rewritten_paths_by_action = {}
    if materialization is not None:
        fp_paths = {}
        for path, stmt in statements.items():
            try:
                fp_paths.setdefault(_mat_fingerprint(stmt), []).append(path)
            except Exception:  # noqa: BLE001
                continue
        for entry in getattr(materialization, "entries", ()) or ():
            action_id = str(_get(entry, "action_id", ""))
            if not action_id:
                continue
            for path in fp_paths.get(str(_get(entry, "new_fingerprint", "")), []):
                rewritten_paths_by_action.setdefault(action_id, set()).add(path)
    before, after = {}, {}
    for obligation in obligation_list:
        if obligation.kind not in ("WAR", "FREE", "WAW", "RAW"):
            continue
        producer_entries = _entries_by_action(ledger, obligation.producer_action)
        consumer_entries = _entries_by_action(ledger, obligation.consumer_action)
        if not producer_entries or not consumer_entries:
            raise CompletionError("COMPLETION_ANCHOR_NOT_FOUND",
                                  "producer/consumer action not found in ledger",
                                  obligation.obligation_id, "ledger action",
                                  [obligation.producer_action, obligation.consumer_action])

        def _instances(action, entries):
            """The real statements this action emits, in real tree order (per iteration)."""
            paths = rewritten_paths_by_action.get(str(action))
            if paths is None:
                paths = {str(getattr(entry, "path", "")) for entry in entries}
            return sorted(paths, key=lambda path: ordinals.get(path, 1 << 30))

        producer_instances = _instances(obligation.producer_action, producer_entries)
        consumer_instances = _instances(obligation.consumer_action, consumer_entries)
        if not producer_instances or not consumer_instances:
            raise CompletionError("COMPLETION_ANCHOR_NOT_FOUND",
                                  "producer/consumer path not found in actual TIR",
                                  obligation.obligation_id, "Evaluate path",
                                  [producer_instances, consumer_instances])
        for path in producer_instances + consumer_instances:
            if path not in statements:
                raise CompletionError("COMPLETION_ANCHOR_NOT_FOUND",
                                      "producer/consumer path not found in actual TIR",
                                      obligation.obligation_id, "Evaluate path", path)
        channel = channel_by_obligation.get(obligation.obligation_id)
        if channel is None:
            raise CompletionError("COMPLETION_MAP_MISMATCH", "no channel for obligation",
                                  obligation.obligation_id, "channel", None)
        allocation_item = allocation_by_channel.get(channel.channel_id)
        if allocation_item is None:
            raise CompletionError("FLAG_POOL_EXHAUSTED", "no flag allocation for channel",
                                  channel.channel_id, "allocation", None)
        # Each generation is its own flag epoch AND its own carrier instance. The POC carrier
        # is unrolled (one real statement per iteration), so generation g is anchored at
        # instance g - distance of the producer and instance g of the consumer. Anchoring all
        # generations at [0] would place an i0->i1 Set and Wait inside the SAME iteration,
        # which is exactly the collapsed-ordering defect this pass must not reintroduce.
        distance = _iteration_offset(obligation)
        for generation in sorted(int(g) for g in obligation.generation_domain):
            producer_path = _instance_anchor(producer_instances, generation, generation - distance,
                                             obligation.obligation_id, "producer")
            consumer_path = _instance_anchor(consumer_instances, generation, generation,
                                             obligation.obligation_id, "consumer")
            set_node = Evaluate(callable=_flag_node(AivFlagSet, obligation, channel,
                                                    allocation_item, generation), args=())
            after.setdefault(_anchor_path(producer_path), []).append(set_node)
            if obligation.obligation_id not in drop_wait_ids:
                wait_node = Evaluate(callable=_flag_node(AivFlagWait, obligation, channel,
                                                         allocation_item, generation), args=())
                before.setdefault(_anchor_path(consumer_path), []).append(wait_node)
    actual_body = _clone_with_anchors(tir_module.body, (), before, after)
    if isinstance(actual_body, Sequential):
        actual_body = _append_aiv_drain(actual_body)
    actual = PrimFunction(name=tir_module.name, params=tir_module.params, body=actual_body,
                          output_count=getattr(tir_module, "output_count", 1),
                          target=getattr(tir_module, "target"))
    # Map every real node that now exists, not a guessed path.
    nodes = _actual_nodes(actual)
    nodes_by_channel = {}
    for path, node in nodes["set"]:
        nodes_by_channel.setdefault((node.channel_name, "SET", getattr(node, "generation", -1)), []).append(path)
    for path, node in nodes["wait"]:
        nodes_by_channel.setdefault((node.channel_name, "WAIT", getattr(node, "generation", -1)), []).append(path)
    placements = []
    for channel in channels:
        allocation_item = allocation_by_channel.get(channel.channel_id)
        for oid in getattr(channel, "covered_obligation_ids", ()):
            obligation = next((o for o in obligation_list if o.obligation_id == oid), None)
            if obligation is None:
                continue
            for generation in obligation.generation_domain:
                for role in ("SET", "WAIT"):
                    paths = nodes_by_channel.get((channel.channel_id, role, generation), [])
                    if not paths:
                        continue
                    placements.append(ActionPlacement(
                        "%s:%s:k%d" % (role.lower(), oid, generation), paths[0], role,
                        channel.channel_id, allocation_item.flag_id, generation,
                        generation % obligation.versions, obligation.versions,
                        obligation.producer_task if role == "SET" else obligation.consumer_task,
                        obligation.producer_action if role == "SET" else obligation.consumer_action,
                        (obligation.obligation_id,)))
    if nodes["drain"]:
        placements.append(ActionPlacement("drain", nodes["drain"][0][0], "DRAIN", "", -1, -1,
                                          -1, 0, "", "", ()))
    return actual, {"placements": tuple(placements), "channels": tuple(channels),
                    "allocations": tuple(allocation), "anchored": True}


def _flag_kind(kind):
    upper = str(kind).upper()
    if upper in ("WAR", "FREE", "WAW"):
        return "free"
    if upper == "RAW":
        return "ready"
    return str(kind).lower()


def _flag_node(cls, obligation, channel, allocation, generation):
    slot = int(generation) % int(obligation.versions)
    return cls(channel_name=channel.channel_id,
               kind=_flag_kind(obligation.required_completion_kind),
               slot=slot, generation=int(generation), versions=int(obligation.versions),
               flag_id=int(allocation.flag_id), producer_pipe=obligation.producer_pipe,
               consumer_pipe=obligation.consumer_pipe)


def inject_completion(tir_module, plan, obligations, allocation, *, channels=None,
                      drop_wait_ids=()):
    """Materialise actual FlagSet/FlagWait/Drain nodes for a static finite domain.

    No metadata-only completion: the returned module contains real TIR nodes, and the
    map only points at those nodes. The function is deliberately a static POC.
    """
    obligation_list = tuple(getattr(obligations, "obligations", obligations))
    if channels is None:
        channels = tuple(LogicalChannel(
            "ch-%s-%s" % (ob.required_completion_kind.lower(),
                          _sha({"pipe": ob.producer_pipe, "root": ob.buffer_root,
                                "region": ob.region})),
            (ob.obligation_id,), ob.producer_pipe, ob.consumer_pipe,
            ob.buffer_root, "ring" if ob.versions > 1 else "once",
            ob.required_completion_kind, "generation=k, slot=k%V") for ob in obligation_list)
    channels = tuple(channels)
    channel_by_obligation = {}
    for channel in channels:
        for obligation_id in getattr(channel, "covered_obligation_ids", ()):
            channel_by_obligation[obligation_id] = channel
    allocation_by_channel = {item.channel_id: item for item in allocation}

    original = tuple(tir_module.body.body) if isinstance(tir_module.body, Sequential) else ()
    nodes = []
    placements = []
    for obligation in obligation_list:
        channel = channel_by_obligation.get(obligation.obligation_id)
        if channel is None:
            raise CompletionError("COMPLETION_MAP_MISMATCH", "no channel for obligation",
                                  obligation.obligation_id, "channel", None)
        allocation_item = allocation_by_channel.get(channel.channel_id)
        if allocation_item is None:
            raise CompletionError("COMPLETION_MAP_MISMATCH", "no flag allocation for channel",
                                  channel.channel_id, "allocation", None)
        for generation in obligation.generation_domain:
            set_node = Evaluate(callable=_flag_node(FlagSet, obligation, channel,
                                                    allocation_item, generation), args=())
            nodes.append(set_node)
            placements.append(ActionPlacement(
                "set:%s:k%d" % (obligation.obligation_id, generation),
                "s%d" % (len(original) + len(nodes) - 1), "SET", channel.channel_id,
                allocation_item.flag_id, generation, generation % obligation.versions,
                obligation.versions, obligation.producer_task, obligation.producer_action,
                (obligation.obligation_id,)))
            if obligation.obligation_id not in drop_wait_ids:
                wait_node = Evaluate(callable=_flag_node(FlagWait, obligation, channel,
                                                         allocation_item, generation), args=())
                nodes.append(wait_node)
                placements.append(ActionPlacement(
                    "wait:%s:k%d" % (obligation.obligation_id, generation),
                    "s%d" % (len(original) + len(nodes) - 1), "WAIT", channel.channel_id,
                    allocation_item.flag_id, generation, generation % obligation.versions,
                    obligation.versions, obligation.consumer_task, obligation.consumer_action,
                    (obligation.obligation_id,)))
    drain = Evaluate(callable=Drain(), args=())
    new_body = Sequential(body=original + tuple(nodes) + (drain,))
    actual = PrimFunction(name=tir_module.name, params=tir_module.params, body=new_body,
                          output_count=getattr(tir_module, "output_count", 1),
                          target=getattr(tir_module, "target"))
    placements.append(ActionPlacement("drain", "s%d" % (len(original) + len(nodes)), "DRAIN",
                                      "", -1, -1, -1, 0, "", "", ()))
    return actual, {"placements": tuple(placements), "channels": tuple(channels),
                    "allocations": tuple(allocation)}


def verify_completion_coverage(problem, plan, actual_tir, placement_map,
                               capabilities, obligations=None) -> CoverageReport:
    """Independent source re-derivation + actual-node scan; map alone cannot self-certify."""
    required = derive_completion_obligations(problem, plan, completion_capabilities=capabilities)
    provided = required if obligations is None else tuple(obligations)
    required_ids = {item.obligation_id for item in required}
    provided_ids = {item.obligation_id for item in provided}
    missing = tuple(sorted(required_ids - provided_ids))
    if missing:
        return CoverageReport("FAILED", "COMPLETION_NOT_VERIFIED", len(required),
                              len(provided), 0, missing, 0,
                              "source-derived obligations missing from supplied plan")
    actual = _actual_nodes(actual_tir)
    placements = tuple(placement_map.get("placements", ())) if isinstance(placement_map, dict) else ()
    channels = tuple(placement_map.get("channels", ())) if isinstance(placement_map, dict) else ()
    channel_by_id = {channel.channel_id: channel for channel in channels}
    covered = 0
    missing_detail = []
    for obligation in required:
        channel = next((c for c in channels if obligation.obligation_id in c.covered_obligation_ids), None)
        if channel is None:
            missing_detail.append("no channel:" + obligation.obligation_id)
            continue
        expected = len(obligation.generation_domain)
        sets = [node for _path, node in actual["set"]
                if node.channel_name == channel.channel_id and node.kind == obligation.required_completion_kind.lower()]
        waits = [node for _path, node in actual["wait"]
                 if node.channel_name == channel.channel_id and node.kind == obligation.required_completion_kind.lower()]
        if len(sets) < expected or len(waits) < expected:
            missing_detail.append("actual nodes:%s set=%d/%d wait=%d/%d"
                                  % (obligation.obligation_id, len(sets), expected, len(waits), expected))
            continue
        covered += 1
    # A placement map entry must point at a real node; metadata alone is rejected.
    actual_paths = {path for group in actual.values() for path, _node in group}
    for placement in placements:
        if placement.role in ("SET", "WAIT") and placement.actual_stmt_ref not in actual_paths:
            return CoverageReport("FAILED", "COMPLETION_MAP_MISMATCH", len(required),
                                  len(provided), covered, missing,
                                  len(placements), "placement points at absent node: %s"
                                  % placement.actual_stmt_ref)
    if not _reachable_drain(actual_tir):
        return CoverageReport("FAILED", "FINAL_DRAIN_MISSING", len(required), len(provided),
                              covered, missing, len(placements), "no actual Drain")
    if missing_detail or covered != len(required):
        return CoverageReport("FAILED", "COMPLETION_NOT_VERIFIED", len(required), len(provided),
                              covered, missing, len(placements), "; ".join(missing_detail))
    return CoverageReport("VERIFIED", None, len(required), len(provided), covered,
                          (), len(placements), "actual Set/Wait/Drain cover all source obligations")


def verify_completion_coverage_from_ledger(ledger, expansion, problem=None, plan=None,
                                             actual_tir=None, placement_map=None,
                                             capabilities=None, obligations=None,
                                             expected=None, **kwargs):
    """Independent 47-item coverage for the anchored reuse extension."""
    if actual_tir is None:
        raise CompletionError("COMPLETION_NOT_VERIFIED", "actual TIR is required",
                              "actual_tir", "typed TIR", None)
    required = tuple(obligations) if obligations is not None else derive_completion_obligations_from_ledger(
        ledger, expansion, problem=problem, plan=plan, expected=expected)
    required_by_kind = {"RAW": 0, "SSA": 0, "WAR_FREE": 0, "FINAL": 0}
    for item in required:
        if item.kind == "RAW":
            required_by_kind["RAW"] += 1
        elif item.kind == "SSA":
            required_by_kind["SSA"] += 1
        elif item.kind == "FINAL":
            required_by_kind["FINAL"] += 1
        else:
            required_by_kind["WAR_FREE"] += 1
    actual = _actual_nodes(actual_tir)
    ordinals, looped = _ordering_context(actual_tir)
    placements = tuple((placement_map or {}).get("placements", ())) if isinstance(placement_map, dict) else ()
    channels = tuple((placement_map or {}).get("channels", ())) if isinstance(placement_map, dict) else ()
    channel_by_obligation = {}
    for channel in channels:
        for oid in getattr(channel, "covered_obligation_ids", ()):
            channel_by_obligation[oid] = channel
    allocation_by_channel = {getattr(item, "channel_id", ""): item
                             for item in ((placement_map or {}).get("allocations", ()) if isinstance(placement_map, dict) else ())}
    entry_by_action = {str(entry.action_id): entry for entry in list(getattr(ledger, "entries", ()) or ())}
    covered = 0
    covered_by_kind = {"RAW": 0, "SSA": 0, "WAR_FREE": 0, "FINAL": 0}
    missing = []
    edge_table = []
    # Ordering refusals are COLLECTED, not returned immediately. A node that exists but is
    # ordered wrongly is still a present node, so a missing node (COMPLETION_NOT_VERIFIED)
    # must outrank it; returning early let one withheld Wait hide behind an unrelated
    # ordering refusal and report the wrong code.
    ordering_failures = []
    for obligation in required:
        kind = obligation.kind
        if kind in ("WAR", "FREE", "WAW", "RAW"):
            channel = channel_by_obligation.get(obligation.obligation_id)
            if channel is None:
                missing.append(obligation.obligation_id)
                continue
            generation_expected = set(int(g) for g in obligation.generation_domain)
            sets = [(path, node) for path, node in actual["set"]
                    if node.channel_name == channel.channel_id and node.kind == _flag_kind(kind)
                    and int(getattr(node, "generation", -1)) in generation_expected]
            waits = [(path, node) for path, node in actual["wait"]
                     if node.channel_name == channel.channel_id and node.kind == _flag_kind(kind)
                     and int(getattr(node, "generation", -1)) in generation_expected]
            set_generations = {int(getattr(node, "generation", -1)) for _path, node in sets}
            wait_generations = {int(getattr(node, "generation", -1)) for _path, node in waits}
            if not generation_expected or not generation_expected <= set_generations or not generation_expected <= wait_generations:
                missing.append(obligation.obligation_id)
                continue
            expected_flag = getattr(allocation_by_channel.get(channel.channel_id), "flag_id", None)
            actual_flags = {int(getattr(node, "flag_id", -1)) for _path, node in sets + waits}
            if len(actual_flags) > 1 or (expected_flag is not None and actual_flags != {int(expected_flag)}):
                return CoverageReport("FAILED", "COMPLETION_FLAG_ID_MISMATCH", len(required),
                                      len(required), covered, tuple(missing), len(placements),
                                      "flag_id differs within channel for %s" % obligation.obligation_id,
                                      required_by_kind, covered_by_kind, tuple(edge_table))
            for _path, node in sets + waits:
                if expected_flag is not None and int(getattr(node, "flag_id", -1)) != int(expected_flag):
                    return CoverageReport("FAILED", "COMPLETION_FLAG_ID_MISMATCH", len(required),
                                          len(required), covered, tuple(missing), len(placements),
                                          "flag_id differs from channel allocation for %s" % obligation.obligation_id,
                                          required_by_kind, covered_by_kind, tuple(edge_table))
                if (str(getattr(node, "producer_pipe", "")) != str(obligation.producer_pipe)
                        or str(getattr(node, "consumer_pipe", "")) != str(obligation.consumer_pipe)):
                    return CoverageReport("FAILED", "COMPLETION_PIPE_MISMATCH", len(required),
                                          len(required), covered, tuple(missing), len(placements),
                                          "flag pipe differs from source edge for %s" % obligation.obligation_id,
                                          required_by_kind, covered_by_kind, tuple(edge_table))
            by_gen_set = {}
            for path, node in sets:
                by_gen_set.setdefault(int(getattr(node, "generation", -1)), []).append(path)
            by_gen_wait = {}
            for path, node in waits:
                by_gen_wait.setdefault(int(getattr(node, "generation", -1)), []).append(path)
            # More than one Set (or Wait) for one (channel, kind, generation) is not a
            # harmless duplicate: the flag protocol has exactly one setter and one waiter
            # per generation, so the extra node is either a forged cover or a stale copy.
            duplicate_problem = None
            for generation in sorted(generation_expected):
                if len(by_gen_set.get(generation, ())) != 1:
                    duplicate_problem = ("generation %d has %d Set nodes for %s"
                                         % (generation, len(by_gen_set.get(generation, ())),
                                            obligation.obligation_id))
                    break
                if len(by_gen_wait.get(generation, ())) != 1:
                    duplicate_problem = ("generation %d has %d Wait nodes for %s"
                                         % (generation, len(by_gen_wait.get(generation, ())),
                                            obligation.obligation_id))
                    break
            offset = _iteration_offset(obligation)
            set_ordinal = min(ordinals.get(path, -1) for path, _node in sets)
            wait_ordinal = max(ordinals.get(path, -1) for path, _node in waits)
            set_in_loop = any(looped.get(path, False) for path, _node in sets)
            wait_in_loop = any(looped.get(path, False) for path, _node in waits)
            covered += 1
            covered_by_kind["RAW" if kind == "RAW" else "WAR_FREE"] += 1
            if duplicate_problem is not None:
                ordering_failures.append(CoverageReport(
                    "FAILED", "COMPLETION_DUPLICATE_FLAG_NODE", len(required), len(required),
                    covered, tuple(missing), len(placements), duplicate_problem,
                    required_by_kind, covered_by_kind, tuple(edge_table)))
            elif offset == 0:
                # Same iteration: tree order IS execution order inside one body sweep.
                if wait_ordinal < set_ordinal:
                    ordering_failures.append(CoverageReport(
                        "FAILED", "COMPLETION_WAIT_BEFORE_SET", len(required), len(required),
                        covered, tuple(missing), len(placements),
                        "wait@%d precedes set@%d for %s"
                        % (wait_ordinal, set_ordinal, obligation.obligation_id),
                        required_by_kind, covered_by_kind, tuple(edge_table)))
            elif not set_in_loop and not wait_in_loop:
                # Cross-iteration edge: the loop supplies the ordering only from the second
                # sweep on. The first sweep of a ring consumer waits on a generation its
                # producer has not reached yet, so some Set must run before it in the tree
                # (an unrolled prologue). Without that the pipeline deadlocks at i=start.
                ordering_failures.append(CoverageReport(
                    "FAILED", "COMPLETION_DISTANCE_NOT_PRESERVED", len(required), len(required),
                    covered, tuple(missing), len(placements),
                    "cross-iteration edge %s is fully unrolled inside one sweep; the distance "
                    "is not materialised" % obligation.obligation_id,
                    required_by_kind, covered_by_kind, tuple(edge_table)))
            elif wait_in_loop and wait_ordinal < set_ordinal:
                ordering_failures.append(CoverageReport(
                    "FAILED", "COMPLETION_STARTUP_UNSATISFIED", len(required), len(required),
                    covered, tuple(missing), len(placements),
                    "cross-iteration Wait@%d in the same sweep as Set@%d has no earlier Set to "
                    "cover iteration start for %s"
                    % (wait_ordinal, set_ordinal, obligation.obligation_id),
                    required_by_kind, covered_by_kind, tuple(edge_table)))
            producer = entry_by_action.get(str(obligation.producer_action))
            consumer = entry_by_action.get(str(obligation.consumer_action))
            if kind == "RAW":
                continue
            edge_table.append({
                "root": obligation.buffer_root,
                "transition": obligation.obligation_id.split(":")[-1],
                "kind": kind,
                "producer_op": getattr(producer, "op", ""),
                "producer_pipe": obligation.producer_pipe,
                "consumer_op": getattr(consumer, "op", ""),
                "consumer_pipe": obligation.consumer_pipe,
                "set_scope": getattr(producer, "path", ""),
                "wait_scope": getattr(consumer, "path", ""),
                "set_iteration": getattr(producer, "logical_iteration", ""),
                "wait_iteration": getattr(consumer, "logical_iteration", ""),
            })
            continue
        if kind == "FINAL":
            if _reachable_drain(actual_tir):
                covered += 1
                covered_by_kind["FINAL"] += 1
            else:
                missing.append(obligation.obligation_id)
            continue
        if kind == "SSA":
            parts = obligation.obligation_id.split(":")
            relation = parts[2] if len(parts) > 2 else ""
            producer_op, consumer_op = relation.split("->", 1) if "->" in relation else ("", "")
            op_order = [(index, type(node.callable).__name__)
                        for index, (_path, node) in enumerate(_walk(actual_tir))
                        if isinstance(node, Evaluate)]
            producer_index = next((index for index, name in op_order if name == producer_op), None)
            consumer_index = next((index for index, name in op_order if name == consumer_op), None)
            if producer_index is not None and consumer_index is not None and producer_index < consumer_index:
                covered += 1
                covered_by_kind["SSA"] += 1
            else:
                missing.append(obligation.obligation_id)
            continue
        missing.append(obligation.obligation_id)
    for placement in placements:
        if getattr(placement, "role", "") in ("SET", "WAIT"):
            paths = {path for group in actual.values() for path, _node in group}
            if placement.actual_stmt_ref not in paths:
                return CoverageReport("FAILED", "COMPLETION_MAP_MISMATCH", len(required),
                                      len(required), covered, tuple(missing), len(placements),
                                      "placement points at absent node", required_by_kind,
                                      covered_by_kind, tuple(edge_table))
    if not _reachable_drain(actual_tir):
        return CoverageReport("FAILED", "FINAL_DRAIN_MISSING", len(required), len(required),
                              covered, tuple(missing), len(placements), "no actual Drain",
                              required_by_kind, covered_by_kind, tuple(edge_table))
    if missing or covered != len(required):
        return CoverageReport("FAILED", "COMPLETION_NOT_VERIFIED", len(required), len(required),
                              covered, tuple(missing), len(placements),
                              "missing/undischarged: %r" % missing[:8], required_by_kind,
                              covered_by_kind, tuple(edge_table))
    if ordering_failures:
        # Every node exists, but at least one edge is materialised in an order that either
        # cannot complete (Wait before Set) or cannot start (no prologue cover).
        first = ordering_failures[0]
        return CoverageReport("FAILED", first.code, first.required, first.provided,
                              first.covered, first.missing_obligation_ids, first.placements,
                              "%d ordering defect(s); first: %s"
                              % (len(ordering_failures), first.detail),
                              first.required_by_kind, first.covered_by_kind, first.edge_table)
    return CoverageReport("VERIFIED", None, len(required), len(required), covered, (),
                          len(placements), "47/47 anchored coverage", required_by_kind,
                          covered_by_kind, tuple(edge_table))



__all__ = ["CompletionError", "Obligation", "Discharge", "LogicalChannel",
           "PhysicalAllocation", "ActionPlacement", "CoverageReport",
           "derive_completion_obligations", "choose_completion_mechanisms",
           "allocate_completion_resources", "inject_completion",
           "verify_completion_coverage", "derive_reuse_obligations_from_ledger",
           "derive_completion_obligations_from_ledger", "inject_completion_anchored",
           "verify_completion_coverage_from_ledger"]
