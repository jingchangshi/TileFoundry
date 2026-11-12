"""Finite symbolic completion: READY/FREE channels over the REAL resolved plan.

WHAT THIS MODULE IS. Completion is the one place that says which generation of a
buffer is safe to read, and when a slot may be overwritten. It is a FINITE
TEMPLATE: the number of sites and channels is decided by the plan's tasks and by
the MAXIMAL READERS of each buffer, never by the trip count N. One site covers
every generation of a buffer; the generation itself is a symbolic Expr.

WHAT IT CONSUMES. Exactly the frozen surface: build_symbolic_completion(plan,
target, effects).

- plan is the single ResolvedPlan (plan/resolved.py): every region's real
  lower/stop/step Expr, II/origin/max_shift, each task's stage/phase/start/
  issue_rank, each buffer's Fixed/Auto domain and resolved_versions.
- target is the real Target; its declared event surface is read through
  target_event_facts(), never defaulted.
- effects is the problem's REAL effect/reuse face: the RegionProblem records the
  plan was resolved from (or the NormalizedProblem that holds them). Those are
  the only records that carry both the effect table and the reuse edge
  (last_reader/next_writer/delay/distance_offset); a bare Effect tuple states no
  reuse edge and no access->buffer link, and is refused by name.

WHAT IT DERIVES, AND FROM WHICH REAL OBJECT.

- writers/readers of a buffer: the RegionProblem's AccessFact records, matched to
  the plan's ScheduledBuffer by root identity (not by name or shape).
- the pipe of an access: the RegionProblem's Effect that names that access id
  (an access with no effect is a refusal, not a guess).
- the maximal readers of a buffer: for each consumer pipe, the reader task with
  the highest plan issue_rank. Same-pipe readers merge into one FREE; different
  pipes get DIFFERENT channels, because a hardware flag is a consumed token and
  two pipes must not share one.
- the FREE generation a writer waits for: task_ordinal - (versions +
  distance_offset). The ordinal is the task's own logical generation; the
  generation it waits for is DERIVED from it, never the same field. versions is
  the plan's resolved_versions, and distance_offset is the problem's reuse
  offset, so no version or offset is re-estimated here.
- the tail: every FREE channel carries ONE region-exit site whose guard is the
  symbolic window max(0, N-(V+offset)) <= g < N, with N built from the plan's
  region bounds. It is never a per-generation list.

WHAT THIS SLICE DOES NOT DO (named refusals, not silent defaults):
- a cross-core handoff (the slice uses the same-core hard-event namespaces);
- a nonzero logical generation_delta (a value-generation mapping this slice
  cannot state);
- STATE effects (no state version fact exists yet);
- more than one writer task of a ring buffer per region.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from tilefoundry.ir.core import Call, Constant, Expr, Var
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.dim import (
    DimAdd,
    DimFloorDiv,
    DimMax,
    DimMin,
    DimMod,
    DimMul,
    DimSub,
    ceildiv,
    dim_expr,
    dim_max,
    simplify_dim,
)

from ..plan.problem import NormalizedProblem, RegionProblem
from ..plan.resolved import ResolvedPlan
from ..plan.units import Auto, Fixed, OwnerKey

from .model import (AIC, AIV, CompletionError, core_for_pipe, declared_core,
                    digest, event_facts_for)

__all__ = [
    "AFTER_TASK",
    "expr_equal",
    "site_domain_guard",
    "BEFORE_TASK",
    "CompletionChannel",
    "CompletionPlan",
    "CompletionSite",
    "CompletionVerification",
    "DRAIN",
    "EventNamespaceKey",
    "EXIT",
    "FREE",
    "LOCAL_EVENT",
    "Mechanism",
    "READY",
    "REGION_EXIT",
    "STATE",
    "build_symbolic_completion",
    "completion_identity",
    "free_wait_generation",
    "free_wait_guard",
    "region_ordinal",
    "region_tail_guard",
    "region_tail_ordinal",
    "region_trip_count",
    "tail_window",
    "task_domain_guard",
]

READY = "READY"
FREE = "FREE"
STATE = "STATE"
EXIT = "EXIT"
REASONS = (READY, FREE, STATE, EXIT)

BEFORE_TASK = "before_task"
AFTER_TASK = "after_task"
REGION_EXIT = "region_exit"
ANCHORS = (BEFORE_TASK, AFTER_TASK, REGION_EXIT)

LOCAL_EVENT = "LOCAL_EVENT"
CROSS_EVENT = "CROSS_EVENT"
FIFO = "FIFO"
PIPE_BARRIER = "PIPE_BARRIER"
DRAIN = "DRAIN"
MECHANISM_KINDS = (LOCAL_EVENT, CROSS_EVENT, FIFO, PIPE_BARRIER, DRAIN)

_READ_MODES = ("READ", "READ_WRITE")
_WRITE_MODES = ("WRITE", "READ_WRITE")

I64 = TensorType.umat_scalar()
BOOL = TensorType.scalar(DType.bool)

_DIM_OPS = {DimAdd, DimSub, DimMul, DimFloorDiv, DimMod, DimMin, DimMax}
_LOGIC_OPS = {
    BinaryKind.EQ: lambda a, b: a == b,
    BinaryKind.NE: lambda a, b: a != b,
    BinaryKind.LT: lambda a, b: a < b,
    BinaryKind.LE: lambda a, b: a <= b,
    BinaryKind.GT: lambda a, b: a > b,
    BinaryKind.GE: lambda a, b: a >= b,
    BinaryKind.AND: lambda a, b: bool(a) and bool(b),
    BinaryKind.OR: lambda a, b: bool(a) or bool(b),
}
_DIM_FOLD = {
    DimAdd: lambda a, b: a + b,
    DimSub: lambda a, b: a - b,
    DimMul: lambda a, b: a * b,
    DimFloorDiv: lambda a, b: a // b,
    DimMod: lambda a, b: a % b,
    DimMin: lambda a, b: min(a, b),
    DimMax: lambda a, b: max(a, b),
}


# --------------------------------------------------------------- plan records
@dataclass(frozen=True)
class EventNamespaceKey:
    """A resolved HARDWARE event resource identity, READY/FREE never part of it.

    The key is (owner, core, normalized producer/consumer pipe pair, bank). Two
    channels with the same key compete for the same finite ids, which is why the
    allocator numbers them together; channels in different keys are independent.
    """

    owner: OwnerKey
    core: str
    pipe_pair: tuple | None
    bank: int = 0

    def key(self) -> str:
        pipes = "-".join(self.pipe_pair) if self.pipe_pair else "*"
        return "%s/%s/%s/bank%d" % (self.core, pipes, _owner_text(self.owner), self.bank)


@dataclass(frozen=True)
class Mechanism:
    """How one channel is physically signalled.

    EVENT mechanisms occupy a finite id range [base_id, base_id + versions) in
    every declared namespace at once (a cross-core channel reserves both banks).
    FIFO/PIPE_BARRIER/DRAIN carry no ids and are only used where the target's
    facts state the ordering contract.
    """

    kind: str
    namespaces: tuple = ()
    base_id: int | None = None
    proof_source: str = ""


@dataclass(frozen=True)
class CompletionSite:
    """One FINITE completion position inside a region.

    task_ref is None only for REGION_EXIT (the exit envelope). task_ordinal is
    the region iteration the site belongs to; generation is the DATA generation
    the channel token protects. They are separate fields on purpose: the writer's
    FREE wait has the ordinal of its own iteration and the generation it consumes.
    """

    site_id: str
    region_id: str
    task_ref: object | None
    anchor: str
    task_ordinal: Expr
    generation: Expr
    guard: Expr
    owner: OwnerKey
    pipe: str


@dataclass(frozen=True)
class CompletionChannel:
    """One READY/FREE/STATE/EXIT obligation between real sites.

    buffer_ref is the plan's ScheduledBuffer.root_ref (object identity), versions
    is that buffer's resolved_versions, and distance_offset is the problem's reuse
    offset for it. producer and consumers are site ids; the tail consumer of a
    FREE channel is the region-exit site carrying the symbolic window.
    """

    channel_id: str
    buffer_ref: object | None
    reason: str
    producer: str
    consumers: tuple
    versions: int
    physical_mechanism: Mechanism
    state_ref: object | None = None
    distance_offset: int = 0


@dataclass(frozen=True)
class CompletionVerification:
    """The receipt verify_completion attaches; never a hand-filled flag."""

    status: str
    completion_identity: str
    plan_identity: str
    target_identity: str


@dataclass(frozen=True)
class CompletionPlan:
    """The finite completion template of one plan iteration."""

    identity: str
    plan_identity: str
    effects_identity: str
    target_identity: str
    plan_iteration: int
    sites: tuple
    channels: tuple
    exit_site_ids: tuple
    verification: CompletionVerification | None = None


# ------------------------------------------------------------ symbolic helpers
def _int(value) -> Constant:
    return dim_expr(int(value))


def _truth() -> Constant:
    return Constant(type=BOOL, value=True)


def _binary(kind: BinaryKind, lhs, rhs) -> Call:
    return Call(type=BOOL, target=Binary(kind=kind), args=(lhs, rhs))


def _conjunction(*parts) -> Expr:
    exprs = [part for part in parts if part is not None]
    if not exprs:
        return _truth()
    result = exprs[0]
    for part in exprs[1:]:
        result = _binary(BinaryKind.AND, result, part)
    return result


def _ordinal_var(name: str) -> Var:
    return Var(name=name, type=I64)


def _static_int(expr):
    """The exact integer an Expr denotes, or None when it is symbolic."""
    if isinstance(expr, Constant):
        value = expr.value
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None
    if isinstance(expr, Call):
        op = type(expr.target)
        if op not in _DIM_OPS:
            return None
        values = [_static_int(arg) for arg in expr.args]
        if any(value is None for value in values):
            return None
        return _DIM_FOLD[op](*values)
    return None


def _static_bool(expr):
    """The exact boolean a predicate denotes, or None when it is symbolic."""
    if isinstance(expr, Constant):
        return bool(expr.value) if isinstance(expr.value, bool) else None
    if isinstance(expr, Call) and isinstance(expr.target, Binary):
        fold = _LOGIC_OPS.get(expr.target.kind)
        if fold is None or len(expr.args) != 2:
            return None
        values = [_static_bool(arg) if _is_bool_expr(arg) else _static_int(arg)
                  for arg in expr.args]
        if any(value is None for value in values):
            return None
        try:
            return bool(fold(*values))
        except TypeError:
            return None
    return None


def _is_bool_expr(expr) -> bool:
    dtype = getattr(getattr(expr, "type", None), "dtype", None)
    return dtype == DType.bool


def _shift(expr: Expr, delta: int) -> Expr:
    return expr if delta == 0 else simplify_dim(DimAdd, (expr, delta))


def _offset(expr: Expr, delta: int) -> Expr:
    return expr if delta == 0 else simplify_dim(DimSub, (expr, delta))


def _expr_key(expr):
    """Canonical structural key of an Expr: content only, never repr/address."""
    if isinstance(expr, Constant):
        return ["constant", _type_key(expr.type), expr.value]
    if isinstance(expr, Var):
        return ["var", str(expr.name), _type_key(expr.type)]
    if isinstance(expr, Call):
        attributes = []
        for parameter in type(expr.target).params():
            if parameter.kind == "attribute":
                value = getattr(expr.target, parameter.name, None)
                attributes.append([parameter.name, getattr(value, "value", str(value))])
        return ["call", type(expr.target).__name__, attributes,
                [_expr_key(argument) for argument in expr.args]]
    raise CompletionError(
        "COMPLETION_EXPR_UNSUPPORTED",
        "a completion expression is not a Constant/Var/Call, so its content cannot be stated",
        type(expr).__name__,
    )


def _type_key(tensor_type) -> str:
    return str(tensor_type)


def expr_equal(left, right) -> bool:
    """Structural equality of two completion Exprs: content, not object identity.

    TensorType carries no value equality, so two structurally identical constants
    built at different moments are different objects. Everything that compares a
    site's expression with a recomputed one goes through this function.
    """
    try:
        return _expr_key(left) == _expr_key(right)
    except CompletionError:
        return False


def _owner_text(owner) -> str:
    group = getattr(owner, "program_group", None)
    name = getattr(group, "name", None) if group is not None else None
    sid = getattr(owner, "sid", None)
    return "%s|%s|%s" % (getattr(owner, "core_kind", "?"), name, sid)


# ------------------------------------------------------------- region symbols
def region_trip_count(region_plan) -> Expr:
    """N = max(0, ceil((stop - lower) / step)) as a symbolic Expr of the plan bounds."""
    lower = getattr(region_plan, "lower_expr", None)
    stop = getattr(region_plan, "stop_expr", None)
    step = getattr(region_plan, "step_expr", None)
    if lower is None or stop is None or step is None:
        raise CompletionError(
            "COMPLETION_BOUNDS_MISSING",
            "a region plan carries no lower/stop/step Expr, so the trip domain is unknown",
            str(getattr(getattr(region_plan, "region_ref", None), "region_id", "")),
        )
    span = simplify_dim(DimSub, (stop, lower))
    return dim_max(0, ceildiv(span, step))


def region_ordinal(region_id: str) -> Var:
    """The symbolic iteration index of a loop region's tasks."""
    return _ordinal_var("k[%s]" % (region_id,))


def region_tail_ordinal(region_id: str) -> Var:
    """The symbolic generation consumed by a region's exit tail loop."""
    return _ordinal_var("g[%s]" % (region_id,))


def task_domain_guard(region_id: str, ordinal: Expr, n_expr: Expr) -> Expr:
    """The notification envelope: the site exists exactly on the region's iterations.

    The original compute predicate is NOT part of this guard; the balanced
    envelope notifies once per in-domain generation even when the computation is
    predicate-disabled (docs spec 04 section 4.3).
    """
    return _conjunction(_binary(BinaryKind.LE, _int(0), ordinal),
                        _binary(BinaryKind.LT, ordinal, n_expr))


def site_domain_guard(region_plan, ordinal: Expr) -> Expr:
    """The notification envelope a task-anchored site carries.

    Inside a loop region it is the iteration domain 0 <= ordinal < N; a straight
    line region runs once, so its envelope is the region's own activation 0 < N.
    The compute predicate stays outside this guard (balanced envelope).
    """
    n_expr = region_trip_count(region_plan)
    if getattr(region_plan.region_ref, "induction_var_ref", None) is None:
        return _binary(BinaryKind.LT, _int(0), n_expr)
    return task_domain_guard(region_plan.region_ref.region_id, ordinal, n_expr)


def free_wait_generation(region_id: str, versions: int, distance_offset: int) -> Expr:
    """The FREE generation writer k consumes: ordinal - (V + offset), derived."""
    return _offset(region_ordinal(region_id), int(versions) + int(distance_offset))


def free_wait_guard(region_id: str, versions: int, distance_offset: int,
                    n_expr: Expr) -> Expr:
    """The wait envelope: in domain AND the first V+offset generations are exempt."""
    ordinal = region_ordinal(region_id)
    return _conjunction(
        task_domain_guard(region_id, ordinal, n_expr),
        _binary(BinaryKind.LE, _int(int(versions) + int(distance_offset)), ordinal),
    )


def region_tail_guard(region_id: str, n_expr: Expr, versions: int,
                      distance_offset: int) -> Expr:
    """The symbolic tail window max(0, N-(V+offset)) <= g < N."""
    generation = region_tail_ordinal(region_id)
    lower = dim_max(0, _offset(n_expr, int(versions) + int(distance_offset)))
    return _conjunction(_binary(BinaryKind.LE, lower, generation),
                        _binary(BinaryKind.LT, generation, n_expr))


def tail_window(site) -> tuple | None:
    """(lower Expr, upper Expr, generation Var) of a symbolic tail guard, or None.

    This reads the window back out of the guard rather than storing a parallel
    field, so a caller (storage LiveWindow, tests) consumes the same expression
    the emitted program will test. A guard that is not the window shape yields
    None: an expanded per-generation list has no window to return.
    """
    guard = getattr(site, "guard", None)
    if not isinstance(guard, Call) or not isinstance(guard.target, Binary):
        return None
    if guard.target.kind is not BinaryKind.AND or len(guard.args) != 2:
        return None
    left, right = guard.args
    if (isinstance(left, Call) and isinstance(left.target, Binary)
            and left.target.kind is BinaryKind.LE and len(left.args) == 2):
        lower = left.args[0]
        generation = left.args[1]
    else:
        return None
    if (isinstance(right, Call) and isinstance(right.target, Binary)
            and right.target.kind is BinaryKind.LT and len(right.args) == 2):
        inside = right.args[0]
        upper = right.args[1]
    else:
        return None
    if not isinstance(generation, Var) or not expr_equal(inside, generation):
        return None
    return (lower, upper, generation)


# ---------------------------------------------------------------- input checks
def _event_facts(target):
    hook = getattr(target, "target_event_facts", None)
    if callable(hook):
        facts = hook()
        if facts is None:
            raise CompletionError(
                "COMPLETION_TARGET_UNKNOWN",
                "the target answers its event surface with None",
                type(target).__name__,
            )
        return facts
    return event_facts_for(getattr(target, "name", None))


# _problem_regions() is DELETED, not kept as a compatibility path. It accepted a
# NormalizedProblem, a single RegionProblem, or a tuple of regions, which meant
# the argument's TYPE decided which obligations existed. completion/bind.py
# replaces it with exactly one shape: the plan's own problem, with the supplied
# effects verified to BE that problem's own records.


def _require_plan(plan) -> ResolvedPlan:
    if not isinstance(plan, ResolvedPlan):
        raise CompletionError("COMPLETION_PLAN_INVALID",
                              "completion is built from the single ResolvedPlan",
                              type(plan).__name__)
    if getattr(plan, "feasibility", None) != "SAT":
        raise CompletionError("COMPLETION_PLAN_NOT_SAT",
                              "only a SAT plan states a runnable schedule",
                              "%r" % (getattr(plan, "feasibility", None),))
    if not getattr(plan, "plan_identity", ""):
        raise CompletionError("COMPLETION_PLAN_INVALID", "the plan carries no identity")
    return plan


def _pair(plan, regions) -> tuple:
    by_id = {region.region_ref.region_id: region for region in regions}
    if len(by_id) != len(regions):
        raise CompletionError("COMPLETION_REGION_DUPLICATE",
                              "two problem regions share one region id")
    planned = [region_plan.region_ref.region_id for region_plan in plan.regions]
    if len(set(planned)) != len(planned):
        raise CompletionError("COMPLETION_REGION_DUPLICATE",
                              "two plan regions share one region id")
    for region_id in planned:
        if region_id not in by_id:
            raise CompletionError(
                "COMPLETION_REGION_UNRESOLVED",
                "the plan has a region with no problem record, so its effects and reuse "
                "edges are not in the completion input", region_id)
    for region_id in by_id:
        if region_id not in set(planned):
            raise CompletionError(
                "COMPLETION_REGION_UNPLANNED",
                "a problem region is not resolved by this plan, so it belongs to another "
                "plan iteration", region_id)
    return tuple((region_plan, by_id[region_plan.region_ref.region_id])
                 for region_plan in plan.regions)


def _effects_identity(regions) -> str:
    payload = []
    for region in regions:
        for effect in region.effects:
            payload.append([
                getattr(region.region_ref, "region_id", ""),
                effect.effect_id,
                effect.task_id,
                effect.kind,
                effect.completion,
                effect.pipe,
                list(effect.access_ids),
            ])
    return digest(payload)


def _declared_leaves(payload, prefix: str) -> tuple:
    """Enumerate every leaf path of a fact payload BEFORE prose is removed.

    The point is ordering: identity is defined by what the records declare, so a
    field added to a fact record is hashed by default. A blacklist would make a
    forgotten entry silently stop carrying identity - the failure mode this
    replaces.
    """
    if isinstance(payload, dict):
        paths = []
        for key in sorted(payload):
            child = "%s.%s" % (prefix, key) if prefix else key
            paths.extend(_declared_leaves(payload[key], child))
        return tuple(paths)
    if isinstance(payload, (list, tuple)):
        if payload and isinstance(payload[0], (dict, list, tuple)):
            paths = []
            for item in payload:
                paths.extend(_declared_leaves(item, prefix + "[]"))
            return tuple(paths)
        return (prefix,)
    return (prefix,)


def _identity_payload(payload):
    """Keep every declared leaf except the ones explicitly declared PROSE.

    A path is prose only if it appears in PROSE_PATHS. That is a positive
    declaration per record: ADDING a field to a fact class makes it participate
    in identity immediately (fail-safe direction), and removing an entry from
    PROSE_PATHS is a deliberate act that makes that sentence identity-bearing.
    """
    def prune(node, path):
        if isinstance(node, dict):
            def child(key):
                return "%s.%s" % (path, key) if path else key

            return {key: prune(node[key], child(key))
                    for key in sorted(node) if child(key) not in PROSE_PATHS}
        if isinstance(node, (list, tuple)):
            if node and isinstance(node[0], (dict, list, tuple)):
                return [prune(item, path + "[]") for item in node]
            return list(node)
        return node

    return prune(payload, "")


#: The ONLY paths treated as documentation. Every other declared leaf of these
#: records participates in target identity.
PROSE_PATHS = frozenset({
    "event_facts.pools[].source",
    "event_facts.hard_event_pools[].source",
    "event_facts.source",
})


def _target_identity(plan, facts) -> str:
    """The target's declared event surface, with prose pruned BEFORE the digest.

    Ordering matters and an earlier version got it wrong: pruning a digest is a
    no-op, because the prose was already hashed. So the payload is built whole,
    declared prose paths are pruned, and ONLY THEN is it digested.

    Identity is defined by what the records DECLARE. Every leaf of these fact
    records participates except those listed in PROSE_PATHS, so a field added to
    EventPool/HardEventPoolFacts/TargetEventFacts is identity-bearing without any
    change here (fail-safe), while leaving a sentence out takes a deliberate edit.

    Covers the facts a completion allocates against: which core domains exist,
    which cross-core mechanism is declared, and each pool's capacity, banks,
    scope, mode, status and pipe pair. Ablation-tested both ways in
    test_rewording_target_prose_cannot_drift_completion_identity and
    test_new_model_field_participates_in_identity_by_default.
    """
    payload = {
        "plan_target": plan.target_identity,
        "event_facts": {
            "profile_id": facts.profile_id,
            "core_domains": sorted(str(domain) for domain in facts.core_domains),
            "cross_core_mechanism": facts.cross_core_mechanism,
            "source": facts.source,
            "pools": [pool.as_dict() for pool in facts.pools],
            "hard_event_pools": [pool.as_dict() for pool in facts.hard_event_pools],
        },
    }
    return digest(_identity_payload(payload))


# ------------------------------------------------------------------- the build
class _RegionBuilder:
    """Derive one region's finite obligations from its real records."""

    def __init__(self, region_plan, region_problem, facts, sites, channels) -> None:
        self.plan = region_plan
        self.problem = region_problem
        self.facts = facts
        self.sites = sites
        self.channels = channels
        self.region_id = region_plan.region_ref.region_id
        self.loop = getattr(region_plan.region_ref, "induction_var_ref", None) is not None
        self.n = region_trip_count(region_plan)
        self.n_static = _static_int(self.n)
        self.ordinal = region_ordinal(self.region_id) if self.loop else _int(0)
        self.pipe_of = {}
        for effect in region_problem.effects:
            if effect.kind != "MEMORY":
                raise CompletionError(
                    "COMPLETION_STATE_NOT_SOURCED",
                    "this slice derives READY/FREE from MEMORY effects only; a %s effect "
                    "has no state version fact to protect" % (effect.kind,),
                    effect.effect_id)
            for access_id in effect.access_ids:
                existing = self.pipe_of.get(access_id)
                if existing is not None and existing != effect.pipe:
                    raise CompletionError(
                        "COMPLETION_ACCESS_PIPE_CONFLICT",
                        "one access is claimed by two pipes", access_id,
                        existing, effect.pipe)
                self.pipe_of[access_id] = effect.pipe
        self.completion_of = {}
        for effect in region_problem.effects:
            for access_id in effect.access_ids:
                self.completion_of[access_id] = effect.completion
        self.task_by_id = {task.task_id: task for task in region_problem.tasks}
        plan_tasks = {task.task_id: task for task in region_plan.tasks}
        self.plan_task = plan_tasks
        if set(plan_tasks) != set(self.task_by_id):
            raise CompletionError(
                "COMPLETION_TASK_SET_MISMATCH",
                "the plan's tasks and the problem's tasks are not the same set",
                self.region_id,
                sorted(plan_tasks), sorted(self.task_by_id))
        self.reuse_of = {}
        for reuse in region_problem.reuse:
            if reuse.buffer in self.reuse_of:
                raise CompletionError(
                    "COMPLETION_MULTIPLE_REUSE_UNSUPPORTED",
                    "two reuse edges protect one buffer id, so the distance offset and "
                    "delay would be ambiguous", reuse.buffer)
            self.reuse_of[reuse.buffer] = reuse
        self.domain_guard = site_domain_guard(region_plan, self.ordinal)

    # -- small accessors ----------------------------------------------------
    def _task_owner(self, task_id: str) -> OwnerKey:
        task = self.task_by_id.get(task_id)
        if task is None:
            raise CompletionError("COMPLETION_TASK_UNKNOWN",
                                  "a site names a task that is not in this region", task_id)
        for use in task.uses:
            return use.resource.owner
        raise CompletionError("COMPLETION_TASK_OWNER_MISSING",
                              "a task states no resource owner", task_id)

    def _task_pipe(self, task_id: str) -> str:
        task = self.task_by_id.get(task_id)
        if task is None or not task.uses:
            raise CompletionError("COMPLETION_TASK_PIPE_MISSING",
                                  "a task states no pipe", task_id)
        return task.uses[0].resource.pipe

    def _pipe_for_access(self, access) -> str:
        pipe = self.pipe_of.get(access.access_id)
        if pipe is None:
            raise CompletionError(
                "COMPLETION_ACCESS_UNSOURCED",
                "an access is named by no effect, so its pipe and completion are unknown",
                access.access_id)
        return pipe

    def _address_key(self, region_id: str, role: str, detail: str) -> str:
        return "site:%s:%s:%s" % (region_id, role, detail)

    def _site(self, site_id, task_id, anchor, ordinal, generation, guard, owner, pipe):
        site = CompletionSite(
            site_id=site_id,
            region_id=self.region_id,
            task_ref=None if task_id is None else self.plan_task[task_id].task_ref,
            anchor=anchor,
            task_ordinal=ordinal,
            generation=generation,
            guard=guard,
            owner=owner,
            pipe=pipe,
        )
        self.sites.append(site)
        return site

    # -- channel machinery --------------------------------------------------
    def _mechanism(self, producer_core, consumer_core, producer_pipe, consumer_pipe,
                   owner_key, versions, reason) -> Mechanism:
        if producer_core != consumer_core:
            raise CompletionError(
                "COMPLETION_CROSS_CORE_NOT_SUPPORTED",
                "this slice only builds same-core channels; a %s->%s hand-off needs the "
                "cross-core bank contract" % (producer_core, consumer_core),
                self.region_id, producer_core, consumer_core)
        namespace = self.facts.physical_namespace_for(
            producer_core, consumer_core, producer_pipe, consumer_pipe,
            owner_key=owner_key)
        key = EventNamespaceKey(
            owner=owner_key[0] if owner_key else OwnerKey(None, producer_core.upper()),
            core=namespace.core_domain.upper(),
            pipe_pair=tuple(namespace.pipe_pair),
            bank=0,
        )
        proof = ("%s %s capacity=%d mode=%s bank_status=%s source=%s"
                 % (namespace.scope, namespace.key(), namespace.capacity,
                    namespace.mode_id, namespace.bank_status, namespace.source))
        return Mechanism(kind=LOCAL_EVENT, namespaces=(key,), base_id=None, proof_source=proof)

    def _add_channel(self, channel_id, buffer_ref, reason, producer_id, consumer_ids,
                     versions, mechanism, state_ref=None, distance_offset=0):
        channel = CompletionChannel(
            channel_id=channel_id,
            buffer_ref=buffer_ref,
            reason=reason,
            producer=producer_id,
            consumers=tuple(consumer_ids),
            versions=int(versions),
            physical_mechanism=mechanism,
            state_ref=state_ref,
            distance_offset=int(distance_offset),
        )
        self.channels.append(channel)
        return channel

    # -- per buffer ---------------------------------------------------------
    def _buffer_channels(self, buffer_plan, logical) -> None:
        accesses = [access for access in self.problem.accesses
                    if access.root_ref is logical.root_ref]
        writers = [access for access in accesses if access.mode in _WRITE_MODES]
        readers = [access for access in accesses
                   if access.mode in _READ_MODES
                   and access.task_id not in {writer.task_id for writer in writers}]
        if not writers or not readers:
            return
        if getattr(logical, "generation_delta", 0):
            raise CompletionError(
                "COMPLETION_GENERATION_DELTA_UNSUPPORTED",
                "a nonzero logical generation offset needs a value-generation map this "
                "slice cannot state", logical.buffer_id, 0, logical.generation_delta)
        versions = int(buffer_plan.resolved_versions)
        if versions < 1:
            raise CompletionError("COMPLETION_VERSIONS_INVALID",
                                  "a scheduled buffer needs at least one version",
                                  logical.buffer_id, ">= 1", versions)
        domain = buffer_plan.domain
        if _domain_shape(domain) != _domain_shape(logical.domain):
            raise CompletionError(
                "COMPLETION_BUFFER_DOMAIN_MISMATCH",
                "the plan's version domain is not the problem buffer's declared domain",
                logical.buffer_id, _domain_shape(logical.domain), _domain_shape(domain))
        if isinstance(logical.domain, Fixed) and int(logical.domain.value) != versions:
            raise CompletionError(
                "COMPLETION_VERSIONS_OUT_OF_DOMAIN",
                "the resolved version count is not the problem's Fixed declaration",
                logical.buffer_id, int(logical.domain.value), versions)
        if isinstance(domain, Fixed) and int(domain.value) != versions:
            raise CompletionError(
                "COMPLETION_VERSIONS_OUT_OF_DOMAIN",
                "a Fixed domain states an equality the resolved version count must hold",
                logical.buffer_id, int(domain.value), versions)
        if isinstance(domain, Auto) and not (
                int(domain.minimum) <= versions <= int(domain.maximum)):
            raise CompletionError(
                "COMPLETION_VERSIONS_OUT_OF_DOMAIN",
                "the resolved version count is outside the buffer's Auto domain",
                logical.buffer_id, [int(domain.minimum), int(domain.maximum)], versions)
        reuse = self.reuse_of.get(logical.buffer_id)
        if self.loop and reuse is None:
            raise CompletionError(
                "COMPLETION_REUSE_UNPROVEN",
                "a loop buffer with readers has no explicit reuse obligation",
                logical.buffer_id)
        distance_offset = 0 if reuse is None else int(reuse.distance_offset)
        if distance_offset != 0:
            raise CompletionError(
                "COMPLETION_REUSE_OFFSET_UNSUPPORTED",
                "nonzero reuse offsets require a proven generation and tail mapping",
                logical.buffer_id, 0, distance_offset)
        if versions + distance_offset < 1:
            raise CompletionError("COMPLETION_VERSION_DELTA",
                                  "versions + offset must stay positive",
                                  logical.buffer_id, 1, versions + distance_offset)
        writer_tasks = []
        for task in self.plan.tasks:
            if any(access.task_id == task.task_id for access in writers) \
                    and task.task_id not in writer_tasks:
                writer_tasks.append(task.task_id)
        if len(writer_tasks) > 1:
            raise CompletionError(
                "COMPLETION_MULTI_WRITER_RING",
                "a rewritten buffer with more than one writer task needs per-writer "
                "tokens this slice does not state", logical.buffer_id,
                "one writer task", writer_tasks)
        if reuse is not None and reuse.next_writer not in {
                access.access_id for access in writers}:
            raise CompletionError(
                "COMPLETION_REUSE_ENDPOINT_MISMATCH",
                "the reuse edge names a next_writer that is not a writer of this buffer",
                logical.buffer_id, sorted(access.access_id for access in writers),
                reuse.next_writer)
        generation = _shift(self.ordinal, int(getattr(logical, "generation_delta", 0)))

        # READY: one channel per declared RAW dependence, so the set of channels
        # is the dependence table and nothing else. Consumer pipes still may not
        # share one flag: a dependence whose consumers span pipes becomes one
        # channel per consumer pipe, each carrying that pipe's consumers.
        reader_groups = {}
        for access in readers:
            pipe = self._pipe_for_access(access)
            reader_groups.setdefault(pipe, []).append(access)
        for pipe, group in reader_groups.items():
            if len({access.task_id for access in group}) > 1:
                raise CompletionError(
                    "COMPLETION_READER_ORDER_UNPROVEN",
                    "issue rank does not prove completion order or permit sharing a flag",
                    logical.buffer_id, "one reader task per pipe", pipe)
        # The dependences that made this buffer's writer a producer at all. A
        # reader with no declared RAW edge here is not protected by a flag, and
        # saying so is better than inventing an obligation the analysis denied.
        writer_task_ids = set(writer_tasks)
        edges = [edge for edge in self.problem.edges
                 if edge.producer in writer_task_ids
                 and any(access.task_id == edge.consumer and access.root_ref is logical.root_ref
                         for access in readers)]
        reader_task_ids = {access.task_id for access in readers}
        covered = {edge.consumer for edge in edges}
        if reader_task_ids and not covered:
            raise CompletionError(
                "COMPLETION_DEPENDENCE_MISSING",
                "%s is read by %s but the problem declares no RAW dependence from the "
                "writer, so no flag obligation is sourced for it"
                % (logical.buffer_id, sorted(reader_task_ids)), self.region_id)
        uncovered = sorted(reader_task_ids - covered)
        if uncovered:
            raise CompletionError(
                "COMPLETION_DEPENDENCE_MISSING",
                "readers of %s appear in no RAW dependence of this buffer: %s"
                % (logical.buffer_id, uncovered), self.region_id, uncovered)
        for writer_task in writer_tasks:
            writer_owner = self._task_owner(writer_task)
            writer_pipe = self._task_pipe(writer_task)
            produce_id = self._address_key(
                self.region_id, "ready-produce", "%s:%s" % (logical.buffer_id, writer_task))
            self._site(produce_id, writer_task, AFTER_TASK, self.ordinal, generation,
                       self.domain_guard, writer_owner, writer_pipe)
            for pipe in sorted(reader_groups):
                consumer_tasks = []
                for task_id, _pipe in self._reader_tasks(reader_groups[pipe]):
                    if task_id not in consumer_tasks:
                        consumer_tasks.append(task_id)
                consumer_ids = []
                consumer_owner = None
                for task_id in consumer_tasks:
                    owner = self._task_owner(task_id)
                    if consumer_owner is None:
                        consumer_owner = owner
                    elif consumer_owner != owner:
                        raise CompletionError(
                            "COMPLETION_SHARED_FLAG_ACROSS_OWNERS",
                            "one channel would notify two execution owners on the same pipe",
                            logical.buffer_id, consumer_owner, owner)
                    consumer_id = self._address_key(
                        self.region_id, "ready-consume",
                        "%s:%s:%s" % (logical.buffer_id, pipe, task_id))
                    self._site(consumer_id, task_id, BEFORE_TASK, self.ordinal, generation,
                               self.domain_guard, owner, self._task_pipe(task_id))
                    consumer_ids.append(consumer_id)
                mechanism = self._mechanism(
                    core_for_pipe(writer_pipe), core_for_pipe(pipe), writer_pipe, pipe,
                    (consumer_owner,), versions, READY)
                self._add_channel(
                    "channel:ready:%s:%s:%s" % (self.region_id, logical.buffer_id, pipe),
                    logical.root_ref, READY, produce_id, consumer_ids, versions, mechanism)

        # FREE: every MAXIMAL reader publishes; the writer waits k-(V+offset).
        if not self.loop or self.n_static == 0:
            return
        wait_guard = free_wait_guard(self.region_id, versions, distance_offset, self.n)
        wait_generation = free_wait_generation(self.region_id, versions, distance_offset)
        tail_guard = region_tail_guard(self.region_id, self.n, versions, distance_offset)
        maximal = self._maximal_readers(logical)
        if reuse is not None and reuse.last_reader not in {
                access_id for _pipe, _task, access_id in maximal}:
            raise CompletionError(
                "COMPLETION_REUSE_ENDPOINT_MISMATCH",
                "the reuse edge names a last_reader that is not a maximal reader of this "
                "buffer", logical.buffer_id, [item[2] for item in maximal], reuse.last_reader)
        writer_task = writer_tasks[0]
        writer_owner = self._task_owner(writer_task)
        writer_pipe = self._task_pipe(writer_task)
        for pipe, task_id, access_id in maximal:
            produce_id = self._address_key(
                self.region_id, "free-publish", "%s:%s" % (logical.buffer_id, task_id))
            self._site(produce_id, task_id, AFTER_TASK, self.ordinal, generation,
                       self.domain_guard, self._task_owner(task_id), pipe)
            wait_id = self._address_key(
                self.region_id, "free-wait", "%s:%s" % (logical.buffer_id, task_id))
            self._site(wait_id, writer_task, BEFORE_TASK, self.ordinal, wait_generation,
                       wait_guard, writer_owner, writer_pipe)
            tail_id = self._address_key(
                self.region_id, "free-tail", "%s:%s" % (logical.buffer_id, task_id))
            self._site(tail_id, None, REGION_EXIT, region_tail_ordinal(self.region_id),
                       region_tail_ordinal(self.region_id), tail_guard, writer_owner,
                       writer_pipe)
            mechanism = self._mechanism(
                core_for_pipe(pipe), core_for_pipe(writer_pipe), pipe, writer_pipe,
                (writer_owner,), versions, FREE)
            self._add_channel(
                "channel:free:%s:%s:%s:%s" % (self.region_id, logical.buffer_id, task_id,
                                              access_id),
                logical.root_ref, FREE, produce_id, (wait_id, tail_id), versions, mechanism,
                distance_offset=distance_offset)

    def _reader_tasks(self, accesses) -> tuple:
        ordered = []
        for access in accesses:
            ordered.append((access.task_id, self._pipe_for_access(access)))
        return tuple(ordered)

    def _maximal_readers(self, logical) -> tuple:
        """(pipe, task_id, access_id) of every MAXIMAL TERMINAL READER.

        Taken from the problem's own terminal_readers fact, resolved through the
        effect table to a pipe - NOT from the highest plan issue_rank. issue_rank
        orders emission; it does not prove that one reader's read has landed
        before another's, and collapsing two readers into the reader that happens
        to be scheduled last would let the writer overwrite a slot whose earlier
        read is still in flight. Two terminal readers on the SAME pipe still get
        TWO channels: merging them requires a proven completion partial order,
        which is exactly what this slice cannot state.
        """
        from .bind import terminal_readers

        readers = terminal_readers(self.problem, logical)
        seen = set()
        ordered = []
        for pipe, task_id, access_id in readers:
            key = (task_id, access_id)
            if key in seen:
                continue
            seen.add(key)
            ordered.append((pipe, task_id, access_id))
        return tuple(ordered)

    # -- region exit drains -------------------------------------------------
    def _drains(self) -> None:
        if self.n_static == 0:
            return
        readers = {access.root_ref for access in self.problem.accesses
                   if access.mode in _READ_MODES}
        handled = set()
        for task in self.plan.tasks:
            task_uses = self.task_by_id.get(task.task_id)
            if task_uses is None:
                continue
            writes = [access for access in self.problem.accesses
                      if access.task_id == task.task_id and access.mode in _WRITE_MODES]
            boundary = []
            for access in writes:
                if self.completion_of.get(access.access_id) != "WRITE_VISIBLE":
                    continue
                logical = self._logical_of(access.root_ref)
                if logical is None or logical.space != "gmem":
                    continue
                if access.root_ref in readers:
                    continue
                boundary.append(logical)
            if not boundary:
                continue
            if task.task_id in handled:
                continue
            handled.add(task.task_id)
            owner = self._task_owner(task.task_id)
            pipe = self._task_pipe(task.task_id)
            produce_id = self._address_key(
                self.region_id, "drain-produce", task.task_id)
            self._site(produce_id, task.task_id, AFTER_TASK, self.ordinal, self.ordinal,
                       self.domain_guard, owner, pipe)
            exit_id = self._address_key(self.region_id, "drain-exit", task.task_id)
            self._site(exit_id, None, REGION_EXIT, self.n, _int(0),
                       _binary(BinaryKind.LT, _int(0), self.n), owner, pipe)
            mechanism = Mechanism(
                kind=DRAIN, namespaces=(), base_id=None,
                proof_source="region exit drain for an async boundary write with no "
                             "consumer in the plan")
            self._add_channel(
                "channel:exit:%s:%s" % (self.region_id, task.task_id),
                boundary[0].root_ref, EXIT, produce_id, (exit_id,), 1, mechanism)

    def _logical_of(self, root):
        for logical in self.problem.buffers:
            if logical.root_ref is root:
                return logical
        return None

    # -- entry --------------------------------------------------------------
    def build(self) -> None:
        if self.n_static == 0:
            # A statically empty trip domain executes nothing and states no
            # obligation; the tail window is empty by construction.
            return
        logical_by_root = {}
        for logical in self.problem.buffers:
            logical_by_root[id(logical.root_ref)] = logical
        for buffer_plan in self.plan.buffers:
            logical = logical_by_root.get(id(buffer_plan.buffer_ref))
            if logical is None or logical.buffer_id != buffer_plan.buffer_id:
                raise CompletionError(
                    "COMPLETION_BUFFER_UNRESOLVED",
                    "a scheduled buffer has no problem record with the same root/buffer id",
                    buffer_plan.buffer_id)
            self._buffer_channels(buffer_plan, logical)
        self._drains()


def build_symbolic_completion(plan, target, effects) -> CompletionPlan:
    """Build the finite template from THE PLAN'S OWN problem, or refuse by name.

    The three arguments are not interchangeable:

      plan     the single ResolvedPlan. Its problem_ref is the only source of
               structural truth, because the obligations protected here must be
               the ones the solver and the schedule were built against.
      target   the real Target; its declared event surface is read here and
               becomes part of the completion's target identity.
      effects  must BE plan.problem_ref's own Effect records, concatenated in the
               problem's region order, object for object. This is a binding
               check, not an alternative input: a second, friendlier effect table
               may not decide which obligations exist.

    The result carries verification=None. Construction is not certification:
    only verify_completion() re-enumerates the obligations independently and
    attaches a receipt, and require_completion_verified() re-runs that against
    the saved real objects before anything consumes the template.
    """
    plan = _require_plan(plan)
    facts = _event_facts(target)
    from .bind import bound_regions

    regions = bound_regions(plan, effects)
    pairs = _pair(plan, regions)
    # The same bounds provenance verify_completion applies, applied HERE too: a
    # hand-assembled plan records no canonical bytes, so plan_content_verified
    # has nothing to compare and the build path alone would happily emit an
    # empty template for bounds somebody edited. Refusing on both sides keeps
    # "build produced 0/0" from ever being the first answer a caller hears.
    from .bind import region_bounds

    for planned, region_problem in pairs:
        region_bounds(planned, region_problem)
    sites = []
    channels = []
    for region_plan, region_problem in pairs:
        _RegionBuilder(region_plan, region_problem, facts, sites, channels).build()
    channels = _assign_base_ids(channels, facts)
    exit_site_ids = tuple(site.site_id for site in sites if site.anchor == REGION_EXIT)
    completion = CompletionPlan(
        identity="",
        plan_identity=plan.plan_identity,
        effects_identity=_effects_identity(regions),
        target_identity=_target_identity(plan, facts),
        plan_iteration=0,
        sites=tuple(sites),
        channels=tuple(channels),
        exit_site_ids=exit_site_ids,
    )
    return replace(completion, identity=completion_identity(completion))


def _domain_shape(domain) -> tuple:
    """The declared version domain as a comparable value (Fixed or Auto, never both)."""
    if isinstance(domain, Fixed):
        return ("fixed", int(domain.value))
    if isinstance(domain, Auto):
        return ("auto", int(domain.minimum), int(domain.maximum))
    return ("unknown", repr(domain))


def _assign_base_ids(channels, facts) -> tuple:
    """Number every EVENT channel inside its declared namespace, or refuse.

    The namespace is a finite hardware resource: a channel occupies
    [base, base + versions) in every namespace it declares. Two channels in one
    namespace must not overlap, and the range may not exceed the declared
    capacity. Nothing here is a default: an undeclared namespace is a refusal.
    """
    capacity = {}
    for channel in channels:
        mechanism = channel.physical_mechanism
        if mechanism.kind != LOCAL_EVENT:
            continue
        for namespace in mechanism.namespaces:
            capacity.setdefault(namespace.key(), _namespace_capacity(facts, namespace))
    used = {}
    numbered = []
    for channel in channels:
        mechanism = channel.physical_mechanism
        if mechanism.kind != LOCAL_EVENT:
            numbered.append(channel)
            continue
        base = 0
        for namespace in mechanism.namespaces:
            occupied = used.setdefault(namespace.key(), [])
            base = max(base, _next_free(occupied, channel.versions))
        for namespace in mechanism.namespaces:
            used[namespace.key()].append((base, base + channel.versions))
            limit = capacity[namespace.key()]
            if base + channel.versions > limit:
                raise CompletionError(
                    "COMPLETION_EVENT_POOL_EXHAUSTED",
                    "the channel needs %d event id(s) in namespace %s, whose declared "
                    "capacity is %d" % (channel.versions, namespace.key(), limit),
                    channel.channel_id, limit, base + channel.versions)
        numbered.append(replace(
            channel, physical_mechanism=replace(mechanism, base_id=base)))
    return tuple(numbered)


def _next_free(occupied, size: int) -> int:
    base = 0
    for start, stop in sorted(occupied):
        if base + size <= start:
            return base
        base = max(base, stop)
    return base


def _namespace_capacity(facts, namespace) -> int:
    # The core token arrives from the mechanism key (upper-cased at construction)
    # and the pool is declared under the target's own spelling, so the lookup
    # resolves through declared_core rather than hand-casing: a .lower() here
    # silently returns None the moment a declared domain is spelled differently,
    # and None read as "no limit" would emit flag ids the hardware does not have.
    core = declared_core(facts, namespace.core, "_namespace_capacity")
    scope = "intra_core" if core in (AIV, AIC) else "cross_core"
    pool = facts.hard_event_pool(core, scope)
    if pool is None:
        raise CompletionError(
            "COMPLETION_EVENT_NAMESPACE_MISSING",
            "the target declares no physical pool for this namespace",
            namespace.key())
    return int(pool.capacity)


def completion_identity(completion) -> str:
    """Content identity of a completion template: every site and channel, not counts."""
    payload = {
        "plan": completion.plan_identity,
        "effects": completion.effects_identity,
        "target": completion.target_identity,
        "iteration": completion.plan_iteration,
        "sites": [
            [site.site_id, site.region_id, site.anchor, site.pipe,
             _owner_text(site.owner), _expr_key(site.task_ordinal),
             _expr_key(site.generation), _expr_key(site.guard)]
            for site in completion.sites
        ],
        "channels": [
            [channel.channel_id, channel.reason,
             "" if channel.buffer_ref is None else str(getattr(channel.buffer_ref, "name", "")),
             channel.producer, list(channel.consumers), channel.versions,
             channel.distance_offset, channel.physical_mechanism.kind,
             channel.physical_mechanism.base_id,
             [namespace.key() for namespace in channel.physical_mechanism.namespaces]]
            for channel in completion.channels
        ],
        "exit_sites": list(completion.exit_site_ids),
    }
    return digest(payload)
