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
import types
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

# ---------------------------------------------------------------- resolve (P3-02)
#: Model-cycle unit tag the solver bridge uses. A value in any other unit (ns, seconds,
#: a bare integer) is NOT a model cycle and must be refused rather than rescaled.
TIME_UNIT = MODEL + " cycles"

#: Where a task's stage came from. Kept separate because the core NORMALIZES the stage
#: origin (it subtracts min(k)*II), so an absolute authored stage cannot be recovered from
#: start_time alone.
STAGE_SOURCE_AUTHORED = "AUTHORED"
STAGE_SOURCE_SOLVER = "SOLVER_DERIVED"
STAGE_SOURCE_UNKNOWN = "UNKNOWN"

#: Solver states that carry no usable schedule. Anything not SAT resolves to no plan.
RESOLVE_NON_SAT = frozenset({"UNSAT", "UNKNOWN", "FALLBACK"})

#: Distinguishes "field absent" from "field present but null" when reading a result.
_MISSING = object()


class PlanError(RuntimeError):
    """A named plan refusal. ``needs_replan`` says the SOLVER must be asked again.

    P3-02 distinguishes "this assignment is unusable" from "the search must be redone":
    an incomplete/inconsistent assignment (a missing task, a unit mismatch, a fixed-version
    disagreement) is a plain refusal, while a violated hard order or a stage outside the
    offset policy is a replanning request. The distinction is how a caller knows not to
    re-run the same solve, so it is carried on the error rather than inferred from the text.
    """

    def __init__(self, code, message, source_id="", expected=None, actual=None,
                 needs_replan=False):
        self.code = code
        self.source_id = source_id
        self.expected = expected
        self.actual = actual
        self.needs_replan = bool(needs_replan)
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
    prologue_domain: dict        # explicit start/stop/step (empty when no prologue)
    steady_domain: dict          # explicit start/stop/step
    drain_domain: dict           # explicit start/stop/step (empty when no drain)
    application_mode: str = "SERIAL"
    static: bool = True
    stage_shift: int = 0         # stage shift applied to producer/consumer accesses
    stage_depth: int = 1


@dataclass(frozen=True)
class TaskPlan:
    task_id: str
    source: str
    action_ids: tuple
    pipe: str
    resource: str
    start_time: dict
    stage: int                    # AUTHORED/absolute stage the solver pinned
    order: int                    # physical issue rank inside the loop body
    iteration_shift: int          # stage - min(stage); the RELATIVE lowering shift
    guard: object
    phase: Optional[int] = None   # start % II: the modulo slot inside one II window
    authored_stage: Optional[int] = None  # stage as authored, before any normalization
    stage_origin: str = STAGE_SOURCE_UNKNOWN


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
    """The PLANNING status: SAT_POSTCHECKED, or the raw solver state verbatim.

    A real ``solve-result/1`` has no ``status`` key: it carries ``state`` plus a
    ``postcheck``. Deriving the planning status from those two is the production path, so
    it is what runs when ``status`` is absent. A caller-supplied ``status`` is still
    honoured for the pre-planning views, but it can never UPGRADE a non-SAT solve: the
    postchecked label additionally requires the real postcheck verdict.
    """
    declared = _get(solve_result, "status", None)
    state = str(_get(solve_result, "state", "") or "").upper()
    postcheck = _get(solve_result, "postcheck", None)
    verdict = _get(postcheck, "verdict", None) if postcheck is not None else None
    if state == "SAT" and str(verdict) == "POSTCHECK_PASS":
        return SAT_POSTCHECKED
    if state:
        # SAT without a passing postcheck is NOT a postchecked plan source.
        return state if state != "SAT" else "SAT_NOT_POSTCHECKED"
    return str(declared) if declared is not None else "unknown"


def _empty_domain(start, step):
    return {"start": int(start), "stop": int(start), "step": int(step)}


def phase_domains(domain, stage_depth):
    """Explicit prologue/steady/drain for a static loop.

    All three ranges lie inside the original loop domain and are disjoint:
      prologue = first (depth-1) iterations
      steady   = middle until the last (depth-1) iterations
      drain    = last (depth-1) iterations
    For stage_depth=1 the whole loop is steady and prologue/drain are empty.
    """
    start, stop, step = int(domain["start"]), int(domain["stop"]), int(domain["step"])
    depth = max(1, int(stage_depth))
    shift = depth - 1
    if start >= stop:
        empty = _empty_domain(start, step)
        return {"prologue": empty, "steady": empty, "drain": empty}
    pro_end = min(stop, start + shift * step)
    drain_start = max(pro_end, stop - shift * step)
    return {
        "prologue": {"start": start, "stop": pro_end, "step": step},
        "steady": {"start": pro_end, "stop": drain_start, "step": step},
        "drain": {"start": drain_start, "stop": stop, "step": step},
    }


def validate_phase_domains(loop: "LoopPlan") -> dict:
    """Reject overlapping/out-of-bounds prologue/steady/drain domains."""
    domains = {"prologue": loop.prologue_domain, "steady": loop.steady_domain,
               "drain": loop.drain_domain}
    for name, domain in domains.items():
        for key in ("start", "stop", "step"):
            if not isinstance(domain.get(key), int):
                raise PlanError("PLAN_PHASE_DOMAIN_INVALID", "%s.%s must be int" % (name, key),
                                loop.source_loop_id, "int", domain.get(key))
        if domain["step"] <= 0 or domain["stop"] < domain["start"]:
            raise PlanError("PLAN_PHASE_DOMAIN_INVALID", "%s domain is invalid" % name,
                            loop.source_loop_id, "start<=stop, step>0", domain)
        if domain["start"] < loop.domain["start"] or domain["stop"] > loop.domain["stop"]:
            raise PlanError("PLAN_PHASE_DOMAIN_INVALID", "%s leaves loop domain" % name,
                            loop.source_loop_id, loop.domain, domain)
    ordered = [("prologue", domains["prologue"]), ("steady", domains["steady"]),
               ("drain", domains["drain"])]
    for (name_a, a), (name_b, b) in zip(ordered, ordered[1:]):
        if a["stop"] > b["start"]:
            raise PlanError("PLAN_PHASE_DOMAIN_OVERLAP", "phase domains overlap",
                            loop.source_loop_id, "%s.stop<=%s.start" % (name_a, name_b),
                            [a, b])
    if loop.stage_shift != max(0, int(loop.stage_depth) - 1):
        raise PlanError("PLAN_STAGE_SHIFT_INVALID", "stage_shift must be stage_depth-1",
                        loop.source_loop_id, int(loop.stage_depth) - 1, loop.stage_shift)
    return {"valid": True, "prologue": loop.prologue_domain,
            "steady": loop.steady_domain, "drain": loop.drain_domain}


# =========================================================== ResolveSchedule (P3-02)
# The ONE consumer of the real solver assignment. spec 2.6: "输入是原问题、真实 IR 和
# solver 的 assignment.tasks[task_id].start_time、assignment.buffers 的 root→resolved-version
# 映射、assignment.ii. 由唯一 adapter 规范化. 不要依赖测试伪造的 schedule 字段."
#
# The solver bridge returns SolveResult(state, assignment, postcheck). It has NO
# "schedule" field; the previous builder read one anyway, so a real result could never
# become a plan. Everything below reads the assignment that actually exists.
#
# Non-negotiables:
#   * II and auto versions come from the ASSIGNMENT, never from Problem defaults.
#   * stage/phase derive from one consistent origin, and BOTH the authored stage and the
#     relative shift are kept (the core normalizes min(k) away, see P3-01).
#   * a solver-returned stage of None is not "stage 0": it is derived here and labelled.
#   * issue rank is a hard-order topological extension; a cycle or a backwards phase edge
#     is a replan, not a tie-break.

@dataclass(frozen=True)
class TaskContract:
    """What the SOURCE says about one task, independent of any solve.

    ``core``/``pipe`` are the execution domain the writer must run on. ``authored_stage``
    is the stage the source pinned (None when the solver is free to choose).
    ``issue_after`` names same-physical-step issue predecessors: an ORDERING fact, not a
    zero-distance data dependency.
    """

    task_id: str
    core: str
    pipe: str
    authored_stage: Optional[int] = None
    issue_after: tuple = ()
    source_index: int = 0


@dataclass(frozen=True)
class BufferContract:
    """What the SOURCE says about one buffer's versioning and physical capacity.

    ``aligned_slot_bytes`` already includes layout, padding and alignment for ONE version.
    ``memory_domain`` is the target-defined capacity scope (a per-owner UB/L1/L0 domain),
    NOT a sum across cores that merely share a name.
    """

    root: str
    min_versions: int
    max_versions: int
    memory_domain: str
    aligned_slot_bytes: int
    fixed_versions: Optional[int] = None


@dataclass(frozen=True)
class TaskResolution:
    task_id: str
    core: str
    pipe: str
    start_time: dict              # {"value": int, "unit": "MODEL cycles"}
    stage: int
    phase: int
    order: int                    # physical issue rank
    iteration_shift: int          # stage - min(stage)
    authored_stage: Optional[int]
    stage_origin: str


@dataclass(frozen=True)
class BufferResolution:
    root: str
    versions: int
    version_source: str           # "fixed" | "auto"
    memory_domain: str
    declared_allocation_bytes: int


@dataclass(frozen=True)
class Resolution:
    """The normalized result of consuming ONE real solver assignment."""

    solver_state: str
    reason: str
    problem_digest: str
    ii: Optional[dict] = None
    tasks: tuple = ()             # source order; each row carries its issue rank
    buffers: tuple = ()
    issue_order: tuple = ()       # task ids in physical issue rank order
    stage_origin: Optional[int] = None
    source_postchecked: bool = False
    declared_storage_bytes: tuple = ()
    capacity_checked: bool = False
    diagnostics: tuple = ()       # recorded decisions/violations, never silent

    @property
    def is_resolved(self):
        return self.solver_state == "SAT" and self.ii is not None

    def task(self, task_id):
        for row in self.tasks:
            if row.task_id == task_id:
                return row
        raise PlanError("PLAN_RESOLVE_TASK_MISSING", "no resolved task with this id",
                        task_id)


def _resolve_get(value, name, default=None):
    """Read a field from a mapping OR an attribute-style result object."""
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _resolve_integer(value, label, minimum):
    """A real integer (bool is NOT an int here) at or above ``minimum``."""
    if type(value) is not int or value < minimum:
        raise PlanError("PLAN_RESOLVE_INTEGER_INVALID",
                        "expected an integer >= %d" % minimum, label, "int >= %d" % minimum,
                        value)
    return value


def _resolve_time(value, label, minimum=0):
    """Normalize a {value, unit} MODEL-cycle record. A bare int is NOT a model cycle."""
    if not isinstance(value, dict) or "value" not in value or "unit" not in value:
        raise PlanError("PLAN_RESOLVE_TIME_RECORD_INVALID",
                        "expected a {value, unit} time record", label,
                        "{'value': int, 'unit': %r}" % TIME_UNIT, value)
    if value["unit"] != TIME_UNIT:
        raise PlanError("PLAN_RESOLVE_TIME_UNIT_MISMATCH",
                        "time is not in model cycles", label, TIME_UNIT, value["unit"])
    return {"value": _resolve_integer(value["value"], label, minimum), "unit": TIME_UNIT}


def _resolve_exact_keys(actual, expected, label):
    """Coverage is EXACT: a missing or an extra id both refuse, no partial plan."""
    if not isinstance(actual, dict):
        raise PlanError("PLAN_RESOLVE_ASSIGNMENT_SHAPE_INVALID",
                        "expected a mapping", label, "dict", type(actual).__name__)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise PlanError("PLAN_RESOLVE_COVERAGE_MISMATCH",
                        "assignment does not cover the source exactly", label,
                        {"missing": [], "extra": []}, {"missing": missing, "extra": extra})


def _resolve_unique(items, field, label):
    seen = {}
    for item in items:
        name = getattr(item, field)
        if not isinstance(name, str) or not name:
            raise PlanError("PLAN_RESOLVE_CONTRACT_ID_INVALID",
                            "empty or non-string contract id", label, "non-empty str", name)
        if name in seen:
            raise PlanError("PLAN_RESOLVE_CONTRACT_ID_INVALID",
                            "duplicate contract id", label, "unique ids", name)
        seen[name] = item
    return seen


def _issue_rank_order(rows, manual):
    """Physical issue order: phase first, then a HARD-order linear extension.

    Two kinds of hard order exist and neither is a data dependency:
      * manual: authored stages activate same-(core, pipe) SOURCE order, because the
        writer's source sequence on one pipe is a real ordering fact;
      * issue_after: an explicit same-physical-step issue predecessor.
    Within the ready set, a later stage and then source position are PREFERENCES. They
    can be reordered by a hard edge -- that is the point -- but they can never create one.
    """
    by_id = {row.task_id: row for row in rows}
    successors = {task_id: set() for task_id in by_id}
    indegree = {task_id: 0 for task_id in by_id}

    def add_edge(before, after, code):
        if before not in by_id or after not in by_id:
            raise PlanError("PLAN_RESOLVE_ORDER_EDGE_INVALID",
                            "hard-order edge names an unresolved task", code,
                            sorted(by_id), [before, after])
        before_phase, after_phase = by_id[before].phase, by_id[after].phase
        if before_phase > after_phase:
            # A phase-ordered edge pointing backwards cannot be fixed by reordering.
            raise PlanError(code, "an earlier-phase task cannot issue after a later-phase one",
                            "%s->%s" % (before, after),
                            "phase(before) <= phase(after)",
                            [before_phase, after_phase], needs_replan=True)
        if before_phase == after_phase and after not in successors[before]:
            successors[before].add(after)
            indegree[after] += 1
        # A strictly earlier phase already satisfies the edge; no topo edge is needed.

    if manual:
        previous = {}
        for row in rows:                      # REAL source order, never dict order
            domain = (row.core, row.pipe)
            if domain in previous:
                add_edge(previous[domain], row.task_id, "PLAN_RESOLVE_MANUAL_ORDER_VIOLATED")
            previous[domain] = row.task_id
    for row in rows:
        for predecessor in row.issue_after:
            add_edge(predecessor, row.task_id, "PLAN_RESOLVE_ISSUE_AFTER_VIOLATED")

    ordered = []
    for phase in sorted({row.phase for row in rows}):
        group = [row for row in rows if row.phase == phase]
        # Preferences only: (-stage, source_index) among tasks that are ALREADY ready.
        ready = sorted(((-row.stage, row.source_index, row.task_id) for row in group
                        if indegree[row.task_id] == 0))
        emitted = 0
        while ready:
            _, _, task_id = ready.pop(0)
            ordered.append(by_id[task_id])
            emitted += 1
            for successor in successors[task_id]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    row = by_id[successor]
                    ready.append((-row.stage, row.source_index, successor))
                    ready.sort()
        if emitted != len(group):
            blocked = [row.task_id for row in group if indegree[row.task_id]]
            raise PlanError("PLAN_RESOLVE_ISSUE_ORDER_CYCLE",
                            "cyclic hard issue order within one phase", "phase %d" % phase,
                            "a linear extension", blocked, needs_replan=True)
    return ordered


def resolve_assignment(solve_result, task_contracts, buffer_contracts, *, enable_offset,
                       expected_problem_digest=None, capacities=None,
                       require_source_postcheck=True) -> Resolution:
    """Normalize ONE real solver assignment into a Resolution. Never mutates the input.

    ``task_contracts`` MUST be in real source order for one scheduled loop body: the
    same-pipe source-order rule and the tie-break depend on it, and the core's task index
    is sorted by id, so assignment dict order is NOT a substitute.

    ``capacities`` maps each memory domain to usable bytes. Nothing is clamped: a version
    or capacity conflict returns ``needs_replan=True``, it never shrinks a version count.
    This is a NAMED-declaration check, not the final asynchronous physical-storage proof.
    """
    state = _resolve_get(solve_result, "state")
    if state not in ("SAT",) + tuple(RESOLVE_NON_SAT):
        raise PlanError("PLAN_RESOLVE_SOLVER_STATE_INVALID",
                        "unknown or missing solver state", "solve_result",
                        ["SAT"] + sorted(RESOLVE_NON_SAT), state)
    digest = _resolve_get(solve_result, "problem_digest", "")
    reason = _resolve_get(solve_result, "reason", "")
    if not isinstance(digest, str) or not digest:
        raise PlanError("PLAN_RESOLVE_SOURCE_ID_MISSING",
                        "the result carries no problem_digest", "solve_result",
                        "non-empty problem_digest", digest)
    if expected_problem_digest is not None and digest != expected_problem_digest:
        raise PlanError("PLAN_RESOLVE_SOURCE_ID_MISMATCH",
                        "result and source describe different problems", "problem_digest",
                        expected_problem_digest, digest)
    if state != "SAT":
        # UNKNOWN stays UNKNOWN even when a partial assignment is present. A fallback
        # candidate is a CANDIDATE: it must be independently verified (P3-01
        # verify_fallback_candidate) and is never turned into a plan here.
        return Resolution(state, str(reason), digest)

    if type(enable_offset) is not bool:
        raise PlanError("PLAN_RESOLVE_OFFSET_POLICY_INVALID",
                        "enable_offset must be a bool", "enable_offset", "bool", enable_offset)
    postchecked = (_resolve_get(_resolve_get(solve_result, "postcheck"), "verdict")
                   == "POSTCHECK_PASS")
    if require_source_postcheck and not postchecked:
        raise PlanError("PLAN_RESOLVE_POSTCHECK_REQUIRED",
                        "run the source-constraint postcheck before resolving", "postcheck",
                        "POSTCHECK_PASS", _resolve_get(solve_result, "postcheck"))

    tasks_in = tuple(task_contracts)
    buffers_in = tuple(buffer_contracts)
    by_task = _resolve_unique(tasks_in, "task_id", "tasks")
    by_buffer = _resolve_unique(buffers_in, "root", "buffers")
    if not tasks_in:
        raise PlanError("PLAN_RESOLVE_EMPTY_TASK_LIST",
                        "a scheduled body must contain tasks", "tasks", "non-empty", 0)
    manual = any(t.authored_stage is not None for t in tasks_in)
    for contract in tasks_in:
        if not contract.core or not contract.pipe:
            raise PlanError("PLAN_RESOLVE_DOMAIN_MISSING",
                            "task has no execution domain", contract.task_id,
                            "core and pipe", [contract.core, contract.pipe])
        if contract.authored_stage is not None:
            _resolve_integer(contract.authored_stage, contract.task_id + ".authored_stage", 0)
        for predecessor in contract.issue_after:
            if predecessor not in by_task or predecessor == contract.task_id:
                raise PlanError("PLAN_RESOLVE_ISSUE_PREDECESSOR_INVALID",
                                "issue_after names an unknown or self task", contract.task_id,
                                "another resolved task", predecessor)
    for contract in buffers_in:
        low = _resolve_integer(contract.min_versions, contract.root + ".min_versions", 1)
        high = _resolve_integer(contract.max_versions, contract.root + ".max_versions", 1)
        if low > high or not contract.memory_domain:
            raise PlanError("PLAN_RESOLVE_BUFFER_CONTRACT_INVALID",
                            "buffer contract has an empty range or domain", contract.root,
                            "min<=max and a memory domain",
                            [low, high, contract.memory_domain])
        _resolve_integer(contract.aligned_slot_bytes, contract.root + ".aligned_slot_bytes", 1)
        if contract.fixed_versions is not None:
            fixed = _resolve_integer(contract.fixed_versions,
                                     contract.root + ".fixed_versions", 1)
            if not low <= fixed <= high:
                raise PlanError("PLAN_RESOLVE_BUFFER_CONTRACT_INVALID",
                                "fixed version count is outside the declared range",
                                contract.root, [low, high], fixed)

    assignment = _resolve_get(solve_result, "assignment", _MISSING)
    if assignment is _MISSING or assignment is None:
        # ABSENT is a different fact from WRONG SHAPE: a legacy result may still carry a
        # hand-written "schedule", and calling that a shape error would suggest that
        # reshaping it would help. It must be named as "not an authoritative input".
        raise PlanError("PLAN_RESOLVE_ASSIGNMENT_MISSING",
                        "the solve result carries no assignment; a fabricated 'schedule'"
                        " dict is not an authoritative input", "assignment",
                        "assignment mapping", sorted(_resolve_dict_keys(solve_result)))
    if not isinstance(assignment, dict):
        raise PlanError("PLAN_RESOLVE_ASSIGNMENT_SHAPE_INVALID",
                        "a SAT result must carry an assignment mapping", "assignment",
                        "dict", type(assignment).__name__)
    starts = assignment.get("tasks")
    versions = assignment.get("buffers")
    _resolve_exact_keys(starts, by_task, "assignment.tasks")
    _resolve_exact_keys(versions, by_buffer, "assignment.buffers")
    units = assignment.get("units")
    if not isinstance(units, dict) or any(units.get(key) != TIME_UNIT
                                          for key in ("ii", "start_time")):
        raise PlanError("PLAN_RESOLVE_TIME_UNIT_MISMATCH",
                        "assignment.units must declare ii/start_time as model cycles",
                        "assignment.units", {"ii": TIME_UNIT, "start_time": TIME_UNIT},
                        units)
    ii = _resolve_time(assignment.get("ii"), "assignment.ii", minimum=1)
    ii_value = ii["value"]

    diagnostics = []
    rows = []
    for source_index, contract in enumerate(tasks_in):
        raw = starts[contract.task_id]
        start = _resolve_time(_resolve_get(raw, "start_time"),
                              contract.task_id + ".start_time", minimum=0)
        stage, phase = divmod(start["value"], ii_value)
        supplied = _resolve_get(raw, "stage")
        if supplied is not None:
            # The core returns stage=None. A caller that supplies one must agree with
            # start//II, otherwise two "authoritative" stage sources disagree.
            if type(supplied) is not int or supplied != stage:
                raise PlanError("PLAN_RESOLVE_STAGE_CONFLICT",
                                "supplied stage disagrees with start//II", contract.task_id,
                                stage, supplied)
        expected_stage = contract.authored_stage if contract.authored_stage is not None else 0
        if manual and stage != expected_stage:
            raise PlanError("PLAN_RESOLVE_AUTHORED_STAGE_VIOLATED",
                            "solver did not honour the authored stage", contract.task_id,
                            expected_stage, stage, needs_replan=True)
        if not enable_offset and stage != 0:
            # spec 2.5 wants k=0 here. Recorded as a replan request, not silently folded:
            # folding it would change the meaning of the start time.
            diagnostics.append({
                "kind": "OFFSET_WINDOW_VIOLATED", "task_id": contract.task_id,
                "stage": stage, "ii": ii_value, "needs_replan": True,
                "detail": "enable_offset=False requires stage 0; the solver returned %d"
                          % stage})
        rows.append({"contract": contract, "start": start, "stage": stage,
                     "phase": phase, "source_index": source_index})

    ordered = _issue_rank_order(
        [types.SimpleNamespace(task_id=row["contract"].task_id,
                               core=row["contract"].core,
                               pipe=row["contract"].pipe,
                               stage=row["stage"],
                               phase=row["phase"],
                               source_index=row["source_index"],
                               issue_after=row["contract"].issue_after)
         for row in rows],
        manual)
    rank = {row.task_id: index for index, row in enumerate(ordered)}
    minimum_stage = min(row["stage"] for row in rows)
    task_results = tuple(
        TaskResolution(
            task_id=row["contract"].task_id, core=row["contract"].core,
            pipe=row["contract"].pipe, start_time=row["start"], stage=row["stage"],
            phase=row["phase"], order=rank[row["contract"].task_id],
            iteration_shift=row["stage"] - minimum_stage,
            authored_stage=row["contract"].authored_stage,
            stage_origin=(STAGE_SOURCE_AUTHORED if row["contract"].authored_stage is not None
                          else STAGE_SOURCE_SOLVER))
        for row in rows)

    buffer_results = []
    usage = {}
    for contract in buffers_in:
        version = _resolve_integer(versions[contract.root],
                                   contract.root + ".resolved_versions", 1)
        # A FIXED count is checked FIRST. It is the more specific obligation: when the
        # source pinned the version, "the solver changed a fixed value" is the true
        # diagnosis, and reporting the generic range breach instead would hide the fact
        # that an AUTHORED decision was overridden.
        if contract.fixed_versions is not None and version != contract.fixed_versions:
            raise PlanError("PLAN_RESOLVE_FIXED_VERSION_VIOLATED",
                            "solver changed a FIXED version count", contract.root,
                            contract.fixed_versions, version, needs_replan=True)
        if not contract.min_versions <= version <= contract.max_versions:
            raise PlanError("PLAN_RESOLVE_VERSION_RANGE_VIOLATED",
                            "resolved version count is outside the declared range",
                            contract.root, [contract.min_versions, contract.max_versions],
                            version, needs_replan=True)
        allocation = version * contract.aligned_slot_bytes
        usage[contract.memory_domain] = usage.get(contract.memory_domain, 0) + allocation
        buffer_results.append(BufferResolution(
            root=contract.root, versions=version,
            version_source=("fixed" if contract.fixed_versions is not None else "auto"),
            memory_domain=contract.memory_domain, declared_allocation_bytes=allocation))

    if capacities is not None:
        if not isinstance(capacities, dict) or any(domain not in capacities for domain in usage):
            raise PlanError("PLAN_RESOLVE_CAPACITY_DOMAIN_MISSING",
                            "capacities must cover every declared memory domain",
                            "capacities", sorted(usage), sorted(capacities or {}))
        for domain, allocation in usage.items():
            capacity = _resolve_integer(capacities[domain], domain + ".capacity", 0)
            if allocation > capacity:
                raise PlanError("PLAN_RESOLVE_CAPACITY_REPLAN_REQUIRED",
                                "declared storage exceeds capacity; replan instead of"
                                " clamping versions", domain, capacity, allocation,
                                needs_replan=True)

    return Resolution(
        solver_state="SAT", reason=str(reason), problem_digest=digest, ii=ii,
        tasks=task_results, buffers=tuple(buffer_results),
        issue_order=tuple(row.task_id for row in ordered),
        stage_origin=minimum_stage, source_postchecked=postchecked,
        declared_storage_bytes=tuple(sorted(usage.items())),
        capacity_checked=capacities is not None, diagnostics=tuple(diagnostics))


def contracts_from_problem(problem):
    """Build the Resolver's SOURCE contracts from the Problem, in REAL source order.

    Task order follows the Problem's own task list, not a sorted id order: the same-pipe
    source-order rule and the equal-phase tie-break both depend on it.
    """
    tasks = []
    for index, task in enumerate(list(_get(problem, "tasks", []) or [])):
        task_id = str(_get(task, "stable_id") or _get(task, "task_id"))
        if not task_id:
            raise PlanError("PLAN_RESOLVE_CONTRACT_ID_INVALID", "task without a stable id",
                            "tasks[%d]" % index, "non-empty str", task_id)
        pipe = str(_get(task, "pipe") or _get(task, "resource") or "")
        resource = str(_get(task, "resource") or pipe)
        authored = _get(task, "authored_stage")
        if authored is None and str(_get(task, "stage_source", "")).upper() == "AUTHORED":
            authored = _get(task, "stage")
        issue_after = tuple(str(name) for name in (_get(task, "issue_after", ()) or ()))
        tasks.append(TaskContract(task_id=task_id, core=resource or pipe, pipe=pipe,
                                 authored_stage=authored, issue_after=issue_after,
                                 source_index=index))
    buffers = []
    for buffer in list(_get(problem, "buffers", []) or []):
        root = str(_get(buffer, "root"))
        if not root:
            raise PlanError("PLAN_RESOLVE_CONTRACT_ID_INVALID", "buffer without a root",
                            "buffers", "non-empty str", root)
        versions = _get(buffer, "versions")
        fixed = bool(_get(buffer, "fixed", False))
        declared_max = _get(buffer, "max_versions")
        if fixed:
            if versions is None or int(versions) <= 0:
                raise PlanError("PLAN_RESOLVE_BUFFER_CONTRACT_INVALID",
                                "a FIXED buffer needs a version count", root, ">0", versions)
            min_versions = max_versions = int(versions)
        else:
            # AUTO: ``versions`` is a NOMINAL default, not a cap. The real freedom is the
            # declared window [min_versions, max_versions], and the solver may pick any
            # count in it. Reading the nominal value as the whole range (what this used to
            # do) turned every auto buffer into a fixed one, so a solver-returned version
            # was always "out of range" unless it repeated the default. The upper bound is
            # the widest of: an explicit max_versions, the pipeline stage budget, the
            # nominal declaration, or a conservative fallback. It is never shrunk below a
            # value the source already declared.
            min_versions = int(_get(buffer, "min_versions", 1) or 1)
            nominal = int(versions) if versions is not None else None
            upper = declared_max
            if upper is None:
                upper = _get(problem, "requested_num_stages", None)
            if upper is None:
                upper = nominal
            max_versions = int(upper if upper is not None else 4)
            if nominal is not None and nominal > max_versions:
                max_versions = nominal
            if max_versions < min_versions:
                max_versions = min_versions
        buffers.append(BufferContract(
            root=root, min_versions=min_versions, max_versions=max_versions,
            memory_domain=str(_get(buffer, "memory_domain")
                              or _get(buffer, "space") or "ascend.ub"),
            aligned_slot_bytes=int(_get(buffer, "aligned_slot_bytes")
                                   or _get(buffer, "bytes")
                                   or _get(buffer, "single_version_bytes") or 0),
            fixed_versions=int(versions) if fixed and versions is not None else None))
    return tuple(tasks), tuple(buffers)


def _resolve_real_assignment(problem, solve_result, tasks_in, buffers_in, *,
                             require_source_postcheck, capacities, enable_offset):
    """Resolve the solver's REAL assignment, or adapt a pre-resolved view.

    Two input shapes are accepted and nothing else:
      * a real ``SolveResult`` with ``state``/``assignment``/``postcheck`` -> the
        authoritative path, normalized by ``resolve_assignment``;
      * an object already exposing a ``resolution`` from THIS module -> re-used as-is,
        so a caller cannot hand over a hand-written schedule dict and bypass coverage,
        unit and version checks.
    A legacy fabricated ``schedule`` dict is NOT a supported input any more.
    """
    existing = _get(solve_result, "resolution", None)
    if isinstance(existing, Resolution):
        return existing
    if _get(solve_result, "assignment", None) is None:
        raise PlanError("PLAN_RESOLVE_ASSIGNMENT_MISSING",
                        "the solve result carries no assignment; a fabricated 'schedule'"
                        " dict is not an authoritative input", "solve_result",
                        "assignment mapping", sorted(_resolve_dict_keys(solve_result)))
    contracts, buffer_contracts = contracts_from_problem(problem)
    return resolve_assignment(
        solve_result, contracts, buffer_contracts, enable_offset=enable_offset,
        expected_problem_digest=_get(problem, "problem_digest"),
        capacities=capacities, require_source_postcheck=require_source_postcheck)


def _resolve_dict_keys(value):
    if isinstance(value, dict):
        return value
    return getattr(value, "__dict__", {}) or {}



def build_resolved_plan(problem, solve_result, source_manifests, *,
                        require_source_postcheck=True, capacities=None,
                        enable_offset=None) -> ResolvedPlan:
    """R0/R1: build the plan from L's Problem + a REAL post-checked SAT result.

    Refuses a fallback/unknown solve (PLAN_REQUIRES_SAT) and any task/access/action id
    that the solve did not actually resolve (PLAN_SOURCE_MISMATCH). Nothing is filled
    with N/A or 0 to look resolved.

    Every RESOLVED value comes from the solver's assignment (spec 2.6): stage/phase/order
    from ``assignment.tasks``, II from ``assignment.ii``, and each buffer's ``slot_count``
    plus ``slot_source`` from ``assignment.buffers``. The Problem supplies only the
    SOURCE facts (ids, shapes, byte sizes, the version WINDOW) -- it never supplies the
    answer. In particular an auto buffer's nominal ``versions`` value is not read back as
    the resolved count, so perturbing the assignment always moves the plan.
    """
    status = _solve_status(solve_result)
    if status != SAT_POSTCHECKED:
        raise PlanError("PLAN_REQUIRES_SAT", "need post-checked SAT", "solve_result",
                        SAT_POSTCHECKED, status)
    source_manifests = source_manifests or {}
    # Each source manifest here is a SCHEMA/CAPABILITY reference (a non-empty descriptor/digest),
    # NOT proof that the source has finished. In particular C's completion stays PENDING; R must
    # not block itself waiting for a final C VERIFIED result (that would be circular).
    for key in ("L", "S", "C", "F", "P"):
        if not source_manifests.get(key):
            raise PlanError("PLAN_SOURCE_MISMATCH", "missing source ref: " + key,
                            key, "descriptor", None)

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

    task_ids = [str(_get(task, "stable_id") or _get(task, "task_id")) for task in tasks_in]
    if not task_ids or any(not tid or tid == "None" for tid in task_ids):
        raise PlanError("PLAN_SOURCE_MISMATCH", "task without a stable id", "tasks")

    # P3-02: consume the REAL solver assignment. The bridge's SolveResult has no
    # "schedule" field; reading one (as this function used to) meant a real result could
    # never become a plan, and a test-fabricated one silently could.
    if enable_offset is None:
        enable_offset = bool(_get(problem, "enable_offset", False))
    resolution = _resolve_real_assignment(
        problem, solve_result, tasks_in, buffers_in,
        require_source_postcheck=require_source_postcheck,
        capacities=capacities, enable_offset=enable_offset)
    placed_rows = {row.task_id: row for row in resolution.tasks}

    tasks = []
    for index, task in enumerate(tasks_in):
        tid = str(_get(task, "stable_id") or _get(task, "task_id"))
        placed = placed_rows[tid]
        action_ids = tuple(str(a) for a in _get(task, "action_ids", []))
        if not action_ids:
            raise PlanError("PLAN_SOURCE_MISMATCH", "task has no action ids", tid,
                            "nonempty", action_ids)
        tasks.append(TaskPlan(
            task_id=tid, source=str(_get(task, "source", tid)),
            action_ids=action_ids, pipe=str(_get(task, "pipe", "")),
            resource=str(_get(task, "resource", "")),
            start_time=dict(placed.start_time), stage=placed.stage, order=placed.order,
            iteration_shift=placed.iteration_shift, guard=_get(task, "guard"),
            phase=placed.phase, authored_stage=placed.authored_stage,
            stage_origin=placed.stage_origin))

    loop_domain = _get(problem, "loop_domain", {}) or {}
    for key in ("start", "stop", "step"):
        if not isinstance(loop_domain.get(key), int):
            raise PlanError("PLAN_SOURCE_MISMATCH", "static loop domain required for R2",
                            "loop_domain", "int start/stop/step", loop_domain)
    domain = {k: int(loop_domain[k]) for k in ("start", "stop", "step")}
    trip = max(0, (domain["stop"] - domain["start"] + domain["step"] - 1) // domain["step"])
    stage_depth = int(_get(problem, "stage_depth", 1) or 1)
    if stage_depth < 1:
        raise PlanError("PLAN_STAGE_SHIFT_INVALID", "stage_depth must be >= 1",
                        "stage_depth", ">=1", stage_depth)
    phases = phase_domains(domain, stage_depth)
    loop = LoopPlan(
        source_loop_id=str(_get(problem, "loop_id", "row_pipeline")),
        domain=domain,
        requested_budget=int(_get(problem, "requested_num_stages", 1) or 1),
        # The RESOLVED offset policy, not a constant False: a plan must say what the solve
        # was actually allowed to do, not what a previous default happened to be.
        enable_offset=bool(enable_offset),
        # spec 2.6: use the II the SOLVER returned. Reading problem["ii"] back here would
        # substitute a nominal default for the authoritative value and hide a changed
        # schedule.
        ii=dict(resolution.ii),
        latency={"value": int(_get(problem, "latency", 1) or 1), "unit": MODEL + " cycles"},
        execution_ordinal_def="k=(i-start)/step",
        prologue_domain=phases["prologue"], steady_domain=phases["steady"],
        drain_domain=phases["drain"], stage_shift=stage_depth - 1, stage_depth=stage_depth)
    validate_phase_domains(loop)

    # spec 2.6: the SLOT COUNT is the solver-assigned version, and the slot SOURCE says
    # whether the source pinned it or the solver chose it. Reading problem["buffers"][*]
    # ["versions"] back here (what this used to do) discarded the assignment: an auto
    # buffer would report its NOMINAL default as if it were the resolved count, and a
    # perturbed assignment.buffers could not change the plan at all.
    placed_buffers = {row.root: row for row in resolution.buffers}
    buffers = []
    for buffer in buffers_in:
        root = str(_get(buffer, "root"))
        if not root:
            raise PlanError("PLAN_SOURCE_MISMATCH", "buffer without root", "buffers")
        placed = placed_buffers.get(root)
        if placed is None:
            # The resolver checks exact coverage first, so this is unreachable unless a
            # caller hands over a pre-resolved view. Name the gap instead of inventing a
            # version count from the Problem.
            raise PlanError("PLAN_RESOLVE_COVERAGE_MISMATCH",
                            "the resolution covers no version for this buffer", root,
                            "root in resolution.buffers", sorted(placed_buffers))
        buffers.append(BufferPlan(
            source_decl=str(_get(buffer, "source_decl", root)), root=root,
            owner=str(_get(buffer, "display_owner", _get(buffer, "owner", "row_pipeline"))),
            space=str(_get(buffer, "space")), dtype=str(_get(buffer, "dtype", "f32")),
            shape=tuple(_get(buffer, "shape", ()) or ()),
            single_version_bytes=int(_get(buffer, "bytes", _get(buffer, "single_version_bytes"))),
            alignment=int(_get(buffer, "alignment", 64)),
            slot_count=int(placed.versions), slot_source=str(placed.version_source)))

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
        ordinal = _get(access, "execution_ordinal", None)
        if ordinal is None:
            shift = int(_get(access, "stage_shift", 0) or 0)
            ordinal = V("k") if shift == 0 else Add(V("i"), C(-shift))
        # Stage-shifted accesses carry their own ordinal expression; it is evaluated
        # per access, never replaced by the materializer's current loop i.
        accesses.append(AccessPlan(
            access_id=access_id, action_id=action_id, task_id=task_id,
            source_stmt=str(_get(access, "source_stmt")),
            buffer_root=str(_get(access, "root")),
            logical_index=_get(access, "logical_index", V("i")),
            execution_ordinal=ordinal, slot=V("slot"),
            generation=V("k"),
            relative_byte_offset=offset,
            guard=_get(access, "guard"), mode=str(_get(access, "mode", "read")),
            byte_footprint=int(_get(access, "byte_footprint", 0))))

    identity = PlanIdentity(
        schema=SCHEMA, source_digest=str(source_digest),
        problem_digest=str(problem_digest), ledger_sha=str(ledger_sha),
        s_request_digest=str(source_manifests["S"]),
        # The legacy fabricated "schedule" field is gone from the ABI: fall back to the
        # ASSIGNMENT the solve actually carries, so the digest names the real input.
        s_result_digest=str(_get(solve_result, "result_digest",
                                _digest(_get(solve_result, "assignment", {})))),
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
    """R5: attach a REAL C verify_completion_coverage result.

    The C report must be VERIFIED and must cover every source-derived obligation; a
    local metadata flag or an empty report is not a completion proof. C derives its
    obligations from the real L/R dependency edges; R does not manufacture them.
    """
    if verified_C_plan is None:
        raise PlanError("COMPLETION_NOT_VERIFIED", "no C verification result supplied",
                        "C", "VERIFIED CoverageReport", None)
    status = str(_get(verified_C_plan, "status", "") or "").upper()
    required = int(_get(verified_C_plan, "required", 0) or 0)
    covered = int(_get(verified_C_plan, "covered", 0) or 0)
    missing = tuple(_get(verified_C_plan, "missing_obligation_ids", ()) or ())
    if status != "VERIFIED" or required <= 0 or covered != required or missing:
        raise PlanError("COMPLETION_NOT_VERIFIED",
                        "C completion coverage is not VERIFIED/complete",
                        "C", {"status": "VERIFIED", "required": covered,
                              "missing": 0}, verified_C_plan)
    digest = _get(verified_C_plan, "result_digest", _get(verified_C_plan, "digest", None))
    if not digest:
        digest = _digest({"status": status, "required": required, "covered": covered})
    syncs = tuple(SyncRef(s.obligation, s.channel, s.event, COMPLETION_VERIFIED)
                  for s in plan.syncs)
    return ResolvedPlan(**{**plan.__dict__, "syncs": syncs,
                           "state": PlanState(plan.state.planning_status,
                                              plan.state.application_status,
                                              COMPLETION_VERIFIED),
                           "units": {**plan.units, "completion_digest": str(digest)}})


def emit_for_codegen(plan: ResolvedPlan, codegen_binding=None) -> dict:
    """The ONLY X boundary. It refuses until C is VERIFIED and X exists.

    There is no silent success and no fabricated source/CCE/binary/load artifact.
    """
    if plan.state.completion_status != COMPLETION_VERIFIED:
        raise PlanError("COMPLETION_NOT_VERIFIED",
                        "X codegen requires a C-VERIFIED completion plan",
                        "completion_status", COMPLETION_VERIFIED,
                        plan.state.completion_status)
    # X is not implemented yet. Even a caller-supplied binding is not accepted as proof:
    # this R module never invents source/CCE/binary/load artifacts.
    raise PlanError("CODEGEN_CONTRACT_UNIMPLEMENTED",
                    "X codegen contract is not implemented; no source/binary emitted",
                    "emit_for_codegen", "real X codegen implementation", codegen_binding)


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
    if not 0 <= int(slot) < int(buffer.slot_count):
        raise PlanError("PLAN_ADDRESS_RANGE_INVALID", "slot outside [0,V)",
                        access.access_id, (0, int(buffer.slot_count) - 1), slot)
    generation = access.generation.evaluate(env)
    if "k" in env and int(generation) != int(env["k"]):
        raise PlanError("GENERATION_MAPPING_INVALID",
                        "generation token is not the full execution ordinal",
                        access.access_id, int(env["k"]), generation)
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


# ------------------------------------------------------------ structured L (/2)
# D6: the structured layer is an increment on THIS authority, not a parallel framework.
# The tree and the single-region lift live in `unit_tree` and are re-exported here so that
# consumers keep importing one place. §5 step 1 stops at structure + one-region lift; the
# /2 plan, hierarchical S and structured R/C are later steps.
from .unit_tree import (  # noqa: E402  (deliberate: keeps one public import surface)
    KIND_DMA as UNIT_KIND_DMA,
    KIND_IF as UNIT_KIND_IF,
    KIND_LOOP as UNIT_KIND_LOOP,
    KIND_VF as UNIT_KIND_VF,
    SCHEMA as UNIT_TREE_SCHEMA,
    RegionLift,
    RegionTaskBinding,
    Unit,
    UnitTree,
    UnitTreeError,
    build_unit_tree,
    lift_region,
)
