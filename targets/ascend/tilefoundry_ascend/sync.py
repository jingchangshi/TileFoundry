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
  FINAL_DRAIN_MISSING
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.tir.flag import Drain, FlagSet, FlagWait
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, Sequential


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


def _flag_node(cls, obligation, channel, allocation, generation):
    slot = int(generation) % int(obligation.versions)
    return cls(channel_name=channel.channel_id,
               kind=obligation.required_completion_kind.lower(),
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
    if not actual["drain"]:
        return CoverageReport("FAILED", "FINAL_DRAIN_MISSING", len(required), len(provided),
                              covered, missing, len(placements), "no actual Drain")
    if missing_detail or covered != len(required):
        return CoverageReport("FAILED", "COMPLETION_NOT_VERIFIED", len(required), len(provided),
                              covered, missing, len(placements), "; ".join(missing_detail))
    return CoverageReport("VERIFIED", None, len(required), len(provided), covered,
                          (), len(placements), "actual Set/Wait/Drain cover all source obligations")


__all__ = ["CompletionError", "Obligation", "Discharge", "LogicalChannel",
           "PhysicalAllocation", "ActionPlacement", "CoverageReport",
           "derive_completion_obligations", "choose_completion_mechanisms",
           "allocate_completion_resources", "inject_completion",
           "verify_completion_coverage"]
