"""resolved-plan/1 — R0 schema, S/L/C source interfaces, typed addresses.

Approved design: artifacts/run02/review/architect_r_plan_materialization_module_design_v1.md
(Lead-approved resolved-plan/1). R own this schema; it does NOT redefine the Problem,
the ledger or the F/P interfaces. L/S/C are consumed through duck-typed read-only views
so this module does not import (or modify) schedule/problem.py, schedule/ledger.py, or
passes/pipeline.py.

Non-negotiables encoded here:
  * unknown / offset are NEVER 0. OFFSET_UNKNOWN is a sentinel; a missing stage/order is
    a refusal (PLAN_SOURCE_MISMATCH), never an N/A masquerading as resolved.
  * a fallback/unknown solve is never SAT (PLAN_REQUIRES_SAT).
  * the generation token is the FULL execution ordinal k=(i-start)/step. Passing a slot
    or a recompute min(tile) index as the token is GENERATION_MAPPING_INVALID.
  * addresses are typed expressions (PExpr: Var/Const/Add/Mul/FloorDiv) with ranges and
    a CPU replay; they are never proven by a string containing "modulo".
  * the first increment is SKELETON_NOT_APPLIED until materialization and independent
    application verification succeed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

SCHEMA = "resolved-plan/1"
OFFSET_UNKNOWN = "OFFSET_UNKNOWN"
NA = "N/A_NOT_IN_PROBLEM"
MODEL = "MODEL"
SAT_POSTCHECKED = "SAT_POSTCHECKED"
STATE_SKELETON = "SKELETON_NOT_APPLIED"
STATE_APPLIED = "APPLIED"
COMPLETION_PENDING = "PENDING"
COMPLETION_VERIFIED = "VERIFIED"


class PlanError(RuntimeError):
    def __init__(self, code, message, source_id="", expected=None, actual=None):
        self.code = code
        self.source_id = source_id
        self.expected = expected
        self.actual = actual
        super().__init__("%s: %s (source=%r expected=%r actual=%r)"
                         % (code, message, source_id, expected, actual))


# --------------------------------------------------------------- typed expr
@dataclass(frozen=True)
class PExpr:
    """A small TYPED address expression: Var / Const / Add / Mul / FloorDiv."""
    op: str
    args: tuple = ()
    value: Optional[int] = None
    name: Optional[str] = None

    def evaluate(self, env: dict) -> int:
        if self.op == "const":
            return int(self.value)
        if self.op == "var":
            if self.name not in env:
                raise PlanError("PLAN_ADDRESS_SYMBOL_UNBOUND", "unbound symbol in address",
                                self.name)
            return int(env[self.name])
        values = [arg.evaluate(env) for arg in self.args]
        if self.op == "add":
            return values[0] + values[1]
        if self.op == "mul":
            return values[0] * values[1]
        if self.op == "floordiv":
            return values[0] // values[1]
        raise PlanError("PLAN_ADDRESS_EXPR_INVALID", "unknown expression op", self.op)

    def symbols(self) -> set:
        if self.op == "var":
            return {self.name}
        out = set()
        for arg in self.args:
            out |= arg.symbols()
        return out

    def to_dict(self):
        out = {"op": self.op}
        if self.value is not None:
            out["value"] = int(self.value)
        if self.name is not None:
            out["name"] = self.name
        if self.args:
            out["args"] = [arg.to_dict() for arg in self.args]
        return out


def C(value: int) -> PExpr:
    return PExpr("const", value=int(value))


def V(name: str) -> PExpr:
    return PExpr("var", name=name)


def Add(a, b):
    return PExpr("add", (a, b))


def Mul(a, b):
    return PExpr("mul", (a, b))


def FloorDiv(a, b):
    return PExpr("floordiv", (a, b))


def _jsonable(value):
    if isinstance(value, PExpr):
        return value.to_dict()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _digest(payload) -> str:
    return hashlib.sha256(json.dumps(_jsonable(payload), sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()


# ------------------------------------------------------------------ records
@dataclass(frozen=True)
class PlanIdentity:
    schema: str
    source_digest: str
    problem_digest: str
    ledger_sha: str
    s_request_digest: str
    s_result_digest: str
    s_core_digest: str
    f_refs: dict
    p_refs: dict
    structural_model_scope: str
    plan_digest: str = ""


@dataclass(frozen=True)
class PlanState:
    planning_status: str = SAT_POSTCHECKED
    application_status: str = STATE_SKELETON
    completion_status: str = COMPLETION_PENDING
    fallback: Optional[str] = None
    unknown: Optional[str] = None


@dataclass(frozen=True)
class LoopPlan:
    source_loop_id: str
    domain: dict                 # start/stop/step, concrete ints for the R2 static fixture
    requested_budget: int
    enable_offset: object        # False | PExpr | OFFSET_UNKNOWN
    ii: dict                     # {"value": int, "unit": "MODEL cycles"}
    latency: dict
    execution_ordinal_def: str   # "k=(i-start)/step"
    prologue_domain: list
    steady_domain: list
    drain_domain: list
    application_mode: str = "SERIAL"
    static: bool = True


@dataclass(frozen=True)
class TaskPlan:
    task_id: str
    source: str
    action_ids: tuple
    pipe: str
    resource: str
    start_time: dict
    stage: int
    order: int
    iteration_shift: int
    guard: object


@dataclass(frozen=True)
class BufferPlan:
    source_decl: str
    root: str
    owner: str
    space: str
    dtype: str
    shape: tuple
    single_version_bytes: int
    alignment: int
    slot_count: int
    slot_source: str             # fixed | auto
    version_stride_bytes: Optional[int] = None
    arena_id: Optional[str] = None
    base_offset_bytes: Optional[int] = None
    allocation_bytes: Optional[int] = None
    lifetime: str = "loop"


@dataclass(frozen=True)
class AccessPlan:
    access_id: str
    action_id: str
    task_id: str
    source_stmt: str
    buffer_root: str
    logical_index: object        # PExpr
    execution_ordinal: object    # PExpr (k), computed at materialization time
    slot: object                 # PExpr, k % V
    generation: object           # FULL execution ordinal k (never slot/min-tile)
    relative_byte_offset: object  # PExpr | OFFSET_UNKNOWN
    guard: object                 # PExpr | None
    byte_footprint: int
    mode: str = "read"


@dataclass(frozen=True)
class EscapePlan:
    field: str
    ssa: int
    region: str
    source: str
    outer_consumer: str
    owner: str
    version_relation: str


@dataclass(frozen=True)
class SyncRef:
    obligation: str
    channel: str
    event: Optional[str] = None
    status: str = COMPLETION_PENDING


@dataclass(frozen=True)
class MaterializationMap:
    entries: tuple = ()
    allocs: tuple = ()
    controls: tuple = ()
    original_digest: str = ""
    rewritten_digest: str = ""


@dataclass(frozen=True)
class ResolvedPlan:
    identity: PlanIdentity
    state: PlanState
    loop: LoopPlan
    tasks: tuple
    buffers: tuple
    accesses: tuple
    escapes: tuple
    syncs: tuple
    unsupported: tuple
    units: dict
    materialization: MaterializationMap = field(default_factory=MaterializationMap)

    def to_dict(self):
        return _jsonable({
            "schema": SCHEMA, "identity": self.identity.__dict__,
            "state": self.state.__dict__, "loop": self.loop.__dict__,
            "tasks": [task.__dict__ for task in self.tasks],
            "buffers": [buffer.__dict__ for buffer in self.buffers],
            "accesses": [access.__dict__ for access in self.accesses],
            "escapes": [escape.__dict__ for escape in self.escapes],
            "syncs": [sync.__dict__ for sync in self.syncs],
            "unsupported": list(self.unsupported), "units": self.units,
            "materialization": {
                "entries": list(self.materialization.entries),
                "allocs": list(self.materialization.allocs),
                "controls": list(self.materialization.controls),
                "original_digest": self.materialization.original_digest,
                "rewritten_digest": self.materialization.rewritten_digest,
            },
        })


# ------------------------------------------------------ ordinal / generation
def loop_trip_count(loop: LoopPlan) -> int:
    start, stop, step = (int(loop.domain[k]) for k in ("start", "stop", "step"))
    if step <= 0:
        raise PlanError("LOOP_DOMAIN_INVALID", "step must be positive", loop.source_loop_id,
                        ">0", step)
    return max(0, (stop - start + step - 1) // step)


def execution_ordinal(loop: LoopPlan, i: int) -> int:
    start, step = int(loop.domain["start"]), int(loop.domain["step"])
    stop = int(loop.domain["stop"])
    if not (start <= i < stop) or (i - start) % step != 0:
        raise PlanError("LOOP_ITERATION_OUT_OF_DOMAIN", "iteration outside loop domain",
                        loop.source_loop_id, (start, stop, step), i)
    return (i - start) // step


def generation_token(loop: LoopPlan, i: int, *, supplied=None,
                     slot: Optional[int] = None, tile_index: Optional[int] = None) -> int:
    """The generation token is the FULL execution ordinal k.

    A slot index or a recompute min(tile) index is a different quantity and must not
    stand in for k; supplying one that disagrees is GENERATION_MAPPING_INVALID.
    """
    k = execution_ordinal(loop, i)
    if supplied is not None and int(supplied) != k:
        raise PlanError("GENERATION_MAPPING_INVALID",
                        "generation token is not the full execution ordinal",
                        loop.source_loop_id, k, supplied)
    if slot is not None and int(slot) == int(supplied if supplied is not None else k) and k != int(slot):
        raise PlanError("GENERATION_MAPPING_INVALID",
                        "slot was substituted for the generation token",
                        loop.source_loop_id, k, slot)
    if tile_index is not None and int(tile_index) == int(supplied if supplied is not None else k) and k != int(tile_index):
        raise PlanError("GENERATION_MAPPING_INVALID",
                        "recompute min(tile) index was substituted for the token",
                        loop.source_loop_id, k, tile_index)
    return k


# --------------------------------------------------------------- R0 builders
def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _solve_status(solve_result) -> str:
    return str(_get(solve_result, "status", "unknown"))


def build_resolved_plan(problem, solve_result, source_manifests) -> ResolvedPlan:
    """R0/R1: build the plan from L's Problem + a REAL post-checked SAT result.

    Refuses a fallback/unknown solve (PLAN_REQUIRES_SAT) and any task/access/action id
    that the solve did not actually resolve (PLAN_SOURCE_MISMATCH). Nothing is filled
    with N/A or 0 to look resolved.
    """
    status = _solve_status(solve_result)
    if status != SAT_POSTCHECKED:
        raise PlanError("PLAN_REQUIRES_SAT", "plan needs a post-checked SAT solve",
                        "solve_result", SAT_POSTCHECKED, status)
    source_manifests = source_manifests or {}
    for key in ("L", "S", "C", "F", "P"):
        if not source_manifests.get(key):
            raise PlanError("PLAN_SOURCE_MISMATCH", "missing source manifest: " + key,
                            key, "digest", None)

    tasks_in = list(_get(problem, "tasks", []))
    accesses_in = list(_get(problem, "accesses", []))
    buffers_in = list(_get(problem, "buffers", []))
    problem_digest = _get(problem, "problem_digest") or _get(problem, "problem_id")
    source_digest = _get(problem, "source_digest")
    ledger_sha = _get(problem, "ledger_sha")
    if not problem_digest or not source_digest or not ledger_sha:
        raise PlanError("PLAN_SOURCE_MISMATCH", "problem lacks stable digests",
                        "problem", "problem_digest/source_digest/ledger_sha",
                        [problem_digest, source_digest, ledger_sha])

    schedule = _get(solve_result, "schedule", {})
    resolved_ids = set(schedule)
    task_ids = [str(_get(task, "stable_id") or _get(task, "task_id")) for task in tasks_in]
    if not task_ids or any(not tid or tid == "None" for tid in task_ids):
        raise PlanError("PLAN_SOURCE_MISMATCH", "task without a stable id", "tasks")
    unknown = sorted(set(task_ids) - resolved_ids)
    if unknown:
        raise PlanError("PLAN_SOURCE_MISMATCH", "solve did not resolve these task ids",
                        "solve.schedule", task_ids, unknown)

    tasks = []
    for index, task in enumerate(tasks_in):
        tid = str(_get(task, "stable_id") or _get(task, "task_id"))
        placed = schedule[tid]
        stage = _get(placed, "stage")
        order = _get(placed, "order")
        start = _get(placed, "start_time")
        if stage is None or order is None or start is None:
            raise PlanError("PLAN_SOURCE_MISMATCH", "task resolution incomplete",
                            tid, "stage/order/start_time", placed)
        action_ids = tuple(str(a) for a in _get(task, "action_ids", []))
        if not action_ids:
            raise PlanError("PLAN_SOURCE_MISMATCH", "task has no action ids", tid,
                            "nonempty", action_ids)
        tasks.append(TaskPlan(
            task_id=tid, source=str(_get(task, "source", tid)),
            action_ids=action_ids, pipe=str(_get(task, "pipe", "")),
            resource=str(_get(task, "resource", "")),
            start_time={"value": int(start), "unit": MODEL + " cycles"},
            stage=int(stage), order=int(order),
            iteration_shift=int(_get(placed, "iteration_shift", 0)),
            guard=_get(placed, "guard")))

    loop_domain = _get(problem, "loop_domain", {}) or {}
    for key in ("start", "stop", "step"):
        if not isinstance(loop_domain.get(key), int):
            raise PlanError("PLAN_SOURCE_MISMATCH", "static loop domain required for R2",
                            "loop_domain", "int start/stop/step", loop_domain)
    domain = {k: int(loop_domain[k]) for k in ("start", "stop", "step")}
    trip = max(0, (domain["stop"] - domain["start"] + domain["step"] - 1) // domain["step"])
    loop = LoopPlan(
        source_loop_id=str(_get(problem, "loop_id", "row_pipeline")),
        domain=domain,
        requested_budget=int(_get(problem, "requested_num_stages", 1) or 1),
        enable_offset=False,
        ii={"value": int(_get(problem, "ii", 1) or 1), "unit": MODEL + " cycles"},
        latency={"value": int(_get(problem, "latency", 1) or 1), "unit": MODEL + " cycles"},
        execution_ordinal_def="k=(i-start)/step",
        prologue_domain=[], steady_domain=list(range(trip)), drain_domain=[])

    buffers = []
    for buffer in buffers_in:
        root = str(_get(buffer, "root"))
        if not root:
            raise PlanError("PLAN_SOURCE_MISMATCH", "buffer without root", "buffers")
        versions = _get(buffer, "versions")
        fixed = bool(_get(buffer, "fixed", False))
        if versions is None or int(versions) <= 0:
            raise PlanError("PLAN_SOURCE_MISMATCH", "buffer versions unknown/zero", root,
                            ">0", versions)
        buffers.append(BufferPlan(
            source_decl=str(_get(buffer, "source_decl", root)), root=root,
            owner=str(_get(buffer, "display_owner", _get(buffer, "owner", "row_pipeline"))),
            space=str(_get(buffer, "space")), dtype=str(_get(buffer, "dtype", "f32")),
            shape=tuple(_get(buffer, "shape", ()) or ()),
            single_version_bytes=int(_get(buffer, "bytes", _get(buffer, "single_version_bytes"))),
            alignment=int(_get(buffer, "alignment", 64)),
            slot_count=int(versions), slot_source="fixed" if fixed else "auto"))

    accesses = []
    for access in accesses_in:
        access_id = str(_get(access, "access_id") or _get(access, "task"))
        task_id = str(_get(access, "task"))
        action_id = str(_get(access, "action_id") or "")
        if not access_id or not task_id:
            raise PlanError("PLAN_SOURCE_MISMATCH", "access without stable ids", "accesses")
        # R1: action must be one of the ids the resolved task owns.
        owner = next((t for t in tasks if t.task_id == task_id), None)
        if owner is None or (action_id and action_id not in owner.action_ids):
            raise PlanError("PLAN_SOURCE_MISMATCH", "access refers to an unresolved action",
                            access_id, "action in resolved task", action_id)
        offset = _get(access, "relative_byte_offset")
        if offset is None:
            offset = OFFSET_UNKNOWN      # never 0
        elif isinstance(offset, int):
            offset = C(offset)
        elif not isinstance(offset, PExpr):
            raise PlanError("PLAN_ADDRESS_EXPR_INVALID", "offset must be PExpr/OFFSET_UNKNOWN",
                            access_id, "PExpr", type(offset).__name__)
        accesses.append(AccessPlan(
            access_id=access_id, action_id=action_id, task_id=task_id,
            source_stmt=str(_get(access, "source_stmt")),
            buffer_root=str(_get(access, "root")),
            logical_index=_get(access, "logical_index", V("i")),
            execution_ordinal=V("k"), slot=V("slot"),
            generation=V("k"),
            relative_byte_offset=offset,
            guard=_get(access, "guard"), mode=str(_get(access, "mode", "read")),
            byte_footprint=int(_get(access, "byte_footprint", 0))))

    identity = PlanIdentity(
        schema=SCHEMA, source_digest=str(source_digest),
        problem_digest=str(problem_digest), ledger_sha=str(ledger_sha),
        s_request_digest=str(source_manifests["S"]),
        s_result_digest=str(_get(solve_result, "result_digest", _digest(_get(solve_result, "schedule", {})))),
        s_core_digest=str(_get(solve_result, "core_digest", source_manifests["S"])),
        f_refs={"digest": source_manifests["F"]},
        p_refs={"digest": source_manifests["P"]},
        structural_model_scope="SERIAL_STATIC_R2")
    temp = ResolvedPlan(
        identity=identity, state=PlanState(), loop=loop, tasks=tuple(tasks),
        buffers=tuple(buffers), accesses=tuple(accesses), escapes=(),
        syncs=(SyncRef("completion", "C", None, COMPLETION_PENDING),),
        unsupported=(), units={"byte_offset": "bytes", "byte_footprint": "bytes",
                               "version_stride_bytes": "bytes",
                               "latency": MODEL + " cycles", "ii": MODEL + " cycles",
                               "start_time": MODEL + " cycles"})
    digest = _digest(temp.to_dict())
    return ResolvedPlan(
        identity=PlanIdentity(**{**identity.__dict__, "plan_digest": digest}),
        **{k: v for k, v in temp.__dict__.items() if k != "identity"})


def verify_plan_sources(plan: ResolvedPlan, problem, source_manifests) -> None:
    """All ids/digests still agree with the Problem and source manifests."""
    source_manifests = source_manifests or {}
    if plan.identity.problem_digest != str(_get(problem, "problem_digest")):
        raise PlanError("PLAN_SOURCE_MISMATCH", "problem digest changed",
                        "problem_digest", plan.identity.problem_digest,
                        _get(problem, "problem_digest"))
    if plan.identity.ledger_sha != str(_get(problem, "ledger_sha")):
        raise PlanError("PLAN_SOURCE_MISMATCH", "ledger sha changed", "ledger_sha",
                        plan.identity.ledger_sha, _get(problem, "ledger_sha"))
    if plan.state.planning_status != SAT_POSTCHECKED:
        raise PlanError("PLAN_REQUIRES_SAT", "plan is not post-checked SAT",
                        "planning_status", SAT_POSTCHECKED, plan.state.planning_status)
    for task in plan.tasks:
        if task.stage is None or task.order is None:
            raise PlanError("PLAN_SOURCE_MISMATCH", "task has N/A stage/order",
                            task.task_id, "real stage/order", None)
    for access in plan.accesses:
        if access.action_id in ("", "N/A", NA):
            raise PlanError("PLAN_SOURCE_MISMATCH", "access action is N/A", access.access_id)


def resolve_buffer_layout(plan: ResolvedPlan, allocation_facts) -> ResolvedPlan:
    """R0: versions/stride/arena ranges; fixed versions must not be silently re-budgeted."""
    allocation_facts = allocation_facts or {}
    buffers = []
    arenas = {}
    for buffer in plan.buffers:
        fact = allocation_facts.get(buffer.root, {}) or {}
        versions = int(fact.get("versions", buffer.slot_count))
        if buffer.slot_source == "fixed" and versions != buffer.slot_count:
            raise PlanError("FIXED_VERSION_MISMATCH",
                            "fixed buffer version count was changed",
                            buffer.root, buffer.slot_count, versions)
        alignment = int(fact.get("alignment", buffer.alignment))
        if alignment <= 0 or (alignment & (alignment - 1)) != 0:
            raise PlanError("ALIGNMENT_INVALID", "alignment must be a power of two",
                            buffer.root, "power-of-two", alignment)
        stride = _align_up(buffer.single_version_bytes, alignment)
        if stride < buffer.single_version_bytes:
            raise PlanError("ALIGNMENT_INVALID", "stride smaller than one version",
                            buffer.root, ">=single_version_bytes", stride)
        arena_id = str(fact.get("arena_id", buffer.arena_id or "arena0"))
        base = fact.get("base_offset_bytes", buffer.base_offset_bytes)
        if base is None:
            raise PlanError("PLAN_ADDRESS_RANGE_INVALID", "missing arena base offset",
                            buffer.root, "explicit base", None)
        base = int(base)
        allocation = int(versions) * stride
        buffers.append(BufferPlan(**{**buffer.__dict__, "slot_count": versions,
                                     "alignment": alignment, "version_stride_bytes": stride,
                                     "arena_id": arena_id, "base_offset_bytes": base,
                                     "allocation_bytes": allocation}))
        arenas.setdefault(arena_id, []).append((base, base + allocation, buffer.root))
    for arena_id, ranges in arenas.items():
        ranges.sort()
        for (base_a, end_a, root_a), (base_b, end_b, root_b) in zip(ranges, ranges[1:]):
            if base_b < end_a and root_a != root_b:
                raise PlanError("PLAN_ADDRESS_OVERLAP", "two live buffers overlap in arena",
                                arena_id, "disjoint", [(root_a, base_a, end_a),
                                                       (root_b, base_b, end_b)])
    return ResolvedPlan(**{**plan.__dict__, "buffers": tuple(buffers)})


def attach_completion(plan: ResolvedPlan, verified_C_plan) -> ResolvedPlan:
    """R5 hook: only accept a C-verified completion plan; keep PENDING otherwise."""
    verified = bool(_get(verified_C_plan, "verified", False))
    digest = _get(verified_C_plan, "digest")
    if not verified or not digest:
        raise PlanError("COMPLETION_NOT_VERIFIED", "C completion plan is not verified",
                        "C", "verified digest", verified_C_plan)
    syncs = tuple(SyncRef(s.obligation, s.channel, s.event, COMPLETION_VERIFIED)
                  for s in plan.syncs)
    return ResolvedPlan(**{**plan.__dict__, "syncs": syncs,
                           "state": PlanState(plan.state.planning_status,
                                              plan.state.application_status,
                                              COMPLETION_VERIFIED)})


def address_range(plan: ResolvedPlan, access: AccessPlan, env: dict):
    """CPU replay of one access address: (start, end) and the containing buffer range."""
    buffer = next((b for b in plan.buffers if b.root == access.buffer_root), None)
    if buffer is None:
        raise PlanError("PLAN_ADDRESS_RANGE_INVALID", "access buffer not in plan",
                        access.access_id, access.buffer_root, None)
    if access.relative_byte_offset == OFFSET_UNKNOWN:
        raise PlanError("PLAN_ADDRESS_OFFSET_UNKNOWN", "offset is UNKNOWN (not 0)",
                        access.access_id, "typed expression", OFFSET_UNKNOWN)
    relative = access.relative_byte_offset.evaluate(env)
    slot = access.slot.evaluate(env)
    start = (buffer.base_offset_bytes + slot * buffer.version_stride_bytes + relative)
    end = start + access.byte_footprint
    if start < buffer.base_offset_bytes or end > buffer.base_offset_bytes + buffer.allocation_bytes:
        raise PlanError("PLAN_ADDRESS_RANGE_INVALID", "access outside buffer allocation",
                        access.access_id, (buffer.base_offset_bytes,
                                           buffer.base_offset_bytes + buffer.allocation_bytes),
                        (start, end))
    return start, end


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment
