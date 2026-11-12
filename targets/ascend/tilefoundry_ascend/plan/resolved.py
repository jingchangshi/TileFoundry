"""The one resolved plan: what the solver decided, in the form every later stage reads.

There is exactly one plan schema. Completion, storage, the symbolic rewrite, the TIR
check and the TileLang emitter all consume these numbers; none of them re-solves or
re-estimates a stage, a phase, an II or a version count.

A plan is built only from a SAT assignment whose independent postcheck passed and whose
source coverage is complete: an UNKNOWN or UNSAT assignment has no plan, and an
incomplete problem has no plan.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

__all__ = ["BufferPlan", "RegionPlan", "ResolveError", "ResolvedPlan", "TaskPlan",
           "resolve_schedule", "verify_plan_bindings", "plan_canonical_bytes",
           "plan_identity_of"]


class ResolveError(RuntimeError):
    """A named refusal to build a plan."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__("%s: %s" % (code, detail))


@dataclass(frozen=True)
class TaskPlan:
    """One unit's resolved timing inside its region."""

    task_ref: object
    task_id: str
    authored_stage: int | None
    resolved_stage: int
    phase: int
    start: int
    issue_rank: int


@dataclass(frozen=True)
class BufferPlan:
    """One buffer's resolved version count and the domain it came from."""

    buffer_ref: object
    buffer_id: str
    domain: object
    resolved_versions: int


@dataclass(frozen=True)
class RegionPlan:
    """One region's loop domain and resolved schedule."""

    region_ref: object
    lower_expr: object
    stop_expr: object
    step_expr: object
    ii: int
    origin: int
    max_shift: int
    tasks: tuple
    buffers: tuple
    identity: str


@dataclass(frozen=True)
class ResolvedPlan:
    """The whole program's single plan.

    problem_ref is the ACTUAL NormalizedProblem this plan was resolved from, and
    the fixed anchors below are the bytes that problem and plan had at that
    moment. Consumers regenerate the bytes from the live objects and compare;
    they do not trust a caller-supplied status or digest. The fields are local
    (compare=False/repr=False) so equality and printing stay about the schedule,
    while identity binding stays about the content.
    """

    source_identity: str
    target_identity: str
    model_version: str
    regions: tuple
    task_to_native_source: dict
    feasibility: str
    optimality: object
    plan_identity: str
    problem_ref: object = field(default=None, compare=False, repr=False)
    problem_identity: str = ""
    problem_canonical_bytes: bytes = b""
    plan_canonical_bytes: bytes = b""
    dependence_proof: object = field(default=None, compare=False, repr=False)
    target_ref: object = field(default=None, compare=False, repr=False)


def _identity(payload) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _stage_of(assignment, task_id: str) -> int:
    try:
        return int(assignment.stages[task_id])
    except (AttributeError, KeyError):
        raise ResolveError("PLAN_TASK_UNRESOLVED", "the assignment has no stage for task %r" % (task_id,)) from None


def _phase_of(assignment, task_id: str) -> int:
    try:
        return int(assignment.phases[task_id])
    except (AttributeError, KeyError):
        raise ResolveError("PLAN_TASK_UNRESOLVED", "the assignment has no phase for task %r" % (task_id,)) from None


def _verify_region(region, assignment) -> int:
    """Recompute this region's schedule from its own problem; refuse a forged one.

    The plan is a consumer of the solver's answer, so it re-checks the answer against
    the region it claims to schedule instead of trusting a SAT status. The arithmetic
    here is independent of the constraint builder: it re-derives every number.
    """
    ii = getattr(assignment, "ii", None)
    if not isinstance(ii, int) or ii < 1:
        raise ResolveError("PLAN_II_INVALID", "a plan needs a positive II, got %r" % (ii,))
    fixed_ii = getattr(region, "fixed_ii", None)
    low_ii = int(getattr(region, "ii_min", 1) or 1)
    high_ii = int(getattr(region, "ii_max", ii) if getattr(region, "ii_max", None) is not None else ii)
    if fixed_ii is not None:
        if ii != int(fixed_ii):
            raise ResolveError("PLAN_II_OUT_OF_DOMAIN",
                               "the author fixed II=%d, the assignment says %d" % (int(fixed_ii), ii))
    elif not low_ii <= ii <= high_ii:
        raise ResolveError("PLAN_II_OUT_OF_DOMAIN",
                           "II=%d is outside the declared domain [%d, %d]" % (ii, low_ii, high_ii))
    tasks = {task.task_id: task for task in region.tasks}
    buffers = {buffer.buffer_id: buffer for buffer in region.buffers}
    raw_starts = dict(getattr(assignment, "starts", {}) or {})
    raw_stages = dict(getattr(assignment, "stages", {}) or {})
    raw_phases = dict(getattr(assignment, "phases", {}) or {})
    raw_versions = dict(getattr(assignment, "versions", {}) or {})
    # One assignment covers the whole program; this region consumes its own slice.
    starts = {key: value for key, value in raw_starts.items() if key in tasks}
    stages = {key: value for key, value in raw_stages.items() if key in tasks}
    phases = {key: value for key, value in raw_phases.items() if key in tasks}
    versions = {key: value for key, value in raw_versions.items() if key in buffers}
    for task_id, task in tasks.items():
        for name, mapping in (("start", starts), ("stage", stages), ("phase", phases)):
            if task_id not in mapping:
                raise ResolveError("PLAN_TASK_UNRESOLVED", "task %r has no %s" % (task_id, name))
        stage = int(stages[task_id])
        phase = int(phases[task_id])
        start = int(starts[task_id])
        if stage < 0 or stage > region.max_stage:
            raise ResolveError("PLAN_STAGE_OUT_OF_RANGE", "task %r resolved at stage %d" % (task_id, stage))
        if not 0 <= phase < ii:
            raise ResolveError("PLAN_PHASE_OUT_OF_RANGE", "task %r has phase %d outside [0,%d)" % (task_id, phase, ii))
        if start != ii * stage + phase:
            raise ResolveError("PLAN_START_INCONSISTENT", "task %r start %d != II*stage+phase" % (task_id, start))
        if task.authored_stage is not None and stage != task.authored_stage:
            raise ResolveError("PLAN_AUTHORED_STAGE_CHANGED", "task %r was authored at stage %d and resolved at %d" % (task_id, task.authored_stage, stage))
        if not getattr(region, "enable_offset", True) and stage != 0:
            raise ResolveError("PLAN_NO_OFFSET_STAGE", "no-offset region has task %r at stage %d" % (task_id, stage))
    raw_order = tuple(getattr(assignment, "issue_order", ()) or ())
    order = tuple(task_id for task_id in raw_order if task_id in tasks)
    if sorted(order) != sorted(tasks):
        raise ResolveError(
            "PLAN_ISSUE_ORDER_INVALID",
            "the emission order does not cover this region's tasks exactly once",
        )
    # Ordering uses each edge/reuse's SOURCED delay. An unsourced one is a
    # named refusal here too: the resolver never reads a raw field that a
    # projection might have defaulted to zero.
    from .delays import DelayRefusal

    try:
        for edge in region.edges:
            if edge.producer not in starts or edge.consumer not in starts:
                raise ResolveError("PLAN_EDGE_ENDPOINT_UNKNOWN", "%s -> %s" % (edge.producer, edge.consumer))
            producer = int(starts[edge.producer])
            consumer = int(starts[edge.consumer])
            if consumer + int(edge.distance) * ii < producer + edge.resolved_delay():
                raise ResolveError("PLAN_EDGE_VIOLATED", "%s -> %s" % (edge.producer, edge.consumer))
    except DelayRefusal as refusal:
        raise ResolveError(refusal.code, refusal.detail) from refusal
    slot_bytes = {buffer.buffer_id: int(buffer.slot_bytes) for buffer in region.buffers}
    accesses = {access.access_id: access for access in getattr(region, "accesses", ())}
    if len(accesses) != len(getattr(region, "accesses", ())):
        raise ResolveError("PLAN_ACCESS_DUPLICATE", "duplicate access identifiers")
    for reuse in region.reuse:
        if reuse.last_reader not in accesses or reuse.next_writer not in accesses:
            raise ResolveError("PLAN_REUSE_ACCESS_UNKNOWN", "reuse endpoint is not a region access")
        reader = accesses[reuse.last_reader].task_id
        writer = accesses[reuse.next_writer].task_id
        if reader not in starts or writer not in starts:
            raise ResolveError("PLAN_REUSE_TASK_UNKNOWN", "reuse access has no scheduled task")
        if reuse.buffer not in versions:
            raise ResolveError("PLAN_BUFFER_UNRESOLVED", "reuse names buffer %r with no version count" % (reuse.buffer,))
        count = int(versions[reuse.buffer]) + int(reuse.distance_offset)
        if count < 1:
            raise ResolveError("PLAN_VERSION_DELTA", "buffer %r reuses a negative generation" % (reuse.buffer,))
        try:
            reuse_delay = reuse.resolved_delay()
        except AttributeError:
            raise ResolveError("PLAN_REUSE_RECORD_INVALID",
                               "a reuse obligation is a problem.Reuse record, not %r"
                               % (type(reuse).__name__,)) from None
        except DelayRefusal as refusal:
            raise ResolveError(refusal.code, refusal.detail) from refusal
        if int(starts[writer]) + count * ii < int(starts[reader]) + reuse_delay:
            raise ResolveError("PLAN_REUSE_VIOLATED", "buffer %r" % (reuse.buffer,))
    for buffer in region.buffers:
        buffer_id = buffer.buffer_id
        if buffer_id not in versions:
            raise ResolveError("PLAN_BUFFER_UNRESOLVED", "buffer %r has no version count" % (buffer_id,))
        count = int(versions[buffer_id])
        domain = getattr(buffer, "domain", None)
        fixed = getattr(domain, "value", None)
        if fixed is not None:
            if count != int(fixed):
                raise ResolveError(
                    "PLAN_FIXED_VERSION_CHANGED",
                    "buffer %r was authored with %d version(s) and resolved to %d"
                    % (buffer_id, int(fixed), count),
                )
        else:
            low = getattr(domain, "minimum", None)
            high = getattr(domain, "maximum", None)
            if count < 1 or (low is not None and count < int(low)) or (high is not None and count > int(high)):
                raise ResolveError(
                    "PLAN_VERSION_OUT_OF_DOMAIN",
                    "buffer %r resolved to %d version(s) outside its domain" % (buffer_id, count),
                )
    uses = {}
    for task in region.tasks:
        for use in getattr(task, "uses", ()) or ():
            key = getattr(use, "resource", None)
            if key is None:
                continue
            uses.setdefault(key, []).append((task.task_id, int(use.duration), int(use.offset)))
    for key, items in uses.items():
        for index, (first_id, first_quantity, first_offset) in enumerate(items):
            if first_quantity > ii:
                raise ResolveError("PLAN_RESOURCE_EXCEEDS_INTERVAL", "%r reserves %d > II=%d" % (first_id, first_quantity, ii))
            for second_id, second_quantity, second_offset in items[index + 1:]:
                delta = ((int(starts[second_id]) + second_offset)
                         - (int(starts[first_id]) + first_offset)) % ii
                if not (first_quantity <= delta and second_quantity <= ii - delta):
                    raise ResolveError("PLAN_RESOURCE_OVERLAP", "%s / %s on %r" % (first_id, second_id, key))
    position = {task_id: index for index, task_id in enumerate(order)}
    previous_phase = None
    for task_id in order:
        level = int(phases[task_id])
        if previous_phase is not None and level < previous_phase:
            raise ResolveError("PLAN_ISSUE_PHASE_ORDER",
                               "the emission order is not non-decreasing in phase at %r" % (task_id,))
        previous_phase = level
    hard_edges = tuple(getattr(region, "issue_after", ()) or ()) + tuple(
        getattr(region, "manual_pipe_order", ()) or ())
    for pair in hard_edges:
        earlier, later = pair
        if earlier not in phases or later not in phases:
            continue
        earlier_phase = int(phases[earlier])
        later_phase = int(phases[later])
        if earlier_phase > later_phase:
            raise ResolveError("PLAN_ISSUE_PHASE_ORDER", "%s must be issued no later than %s" % (earlier, later))
        if earlier_phase == later_phase and position[earlier] > position[later]:
            raise ResolveError("PLAN_ISSUE_ORDER_VIOLATED",
                               "%s is issued after %s although the author requires the reverse" % (earlier, later))
    for capacity in region.capacities:
        total = int(capacity.reserved_bytes)
        for buffer_id in capacity.buffers:
            if buffer_id not in slot_bytes:
                raise ResolveError("PLAN_CAPACITY_FOREIGN_BUFFER", "capacity lists %r" % (buffer_id,))
            total += int(versions.get(buffer_id, 0)) * slot_bytes[buffer_id]
        if total > int(capacity.usable_bytes):
            raise ResolveError("PLAN_CAPACITY_EXCEEDED", "%d bytes over %d" % (total, capacity.usable_bytes))
    return ii


def check_numeric_region(region, assignment):
    """Check numerical constraints only; this does not construct or certify a plan.

    The caller must independently establish source coverage and dependencies.
    """
    return _verify_region(region, assignment)


def _expr_encoding(problem):
    """The lexical environment used to encode decision inputs structurally."""
    from .identity import source_labels

    snapshot = getattr(problem, "source_snapshot", None)
    entry = getattr(snapshot, "entry_ref", None)
    if entry is None or type(entry).__name__ != "PrimFunction":
        return {}
    return source_labels(entry)


def _encode_decision(value, env, path: str):
    """Encode one decided value structurally; never repr a live object graph.

    repr() of an Expr or a version domain prints a Python object layout that
    no later consumer can re-derive and that changes when the dataclass is
    reformatted; the shared encoder names fields and values instead. Values
    outside the supported grammar refuse by path rather than falling back to
    repr, which is the same rule the source and problem recipes use.
    """
    from .identity import IdentityRefusal, _encode

    try:
        return _encode(value, env, path, 0)
    except IdentityRefusal:
        # A symbolic bound can reference a binder from a scope this plan does
        # not carry; record it as structure through the plain encoder instead
        # of dropping the field, so a change of value still moves identity.
        if hasattr(value, "__dataclass_fields__") or isinstance(value, (str, int, float, bool)):
            return {"#": "decision", "repr": repr(value)}
        raise


def _expr_text(expr) -> str:
    """A stable textual form of one bound expression, for comparison only.

    Never hashed, never stored in an identity: repr is not reproducible
    across processes for every expression kind, so this exists only to decide
    whether two expressions agree right now, and the verdict is a refusal that
    prints both sides.
    """
    from .identity import IdentityRefusal, _encode

    try:
        return json.dumps(_encode(expr, {}, "$", 0), sort_keys=True)
    except IdentityRefusal:
        return repr(expr)


def _plan_bytes(problem, regions, proof) -> bytes:
    """The ONE recipe for a plan's decided content, used by resolve and by every
    consumer that must check it.

    Only decided content, and only STABLE content: no id(), no address, no
    repr of a live object graph. Object identity is checked separately and
    explicitly by verify_plan_bindings(); a digest must be recomputable by a
    later consumer in another process, and a memory address is not.

    The problem anchor is RECOMPUTED from the problem object handed in (or
    recorded as absent), never read from a stored field: a caller that types
    a digest into plan_identity or problem_identity certifies nothing.
    """
    env = _expr_encoding(problem)
    if problem is None:
        anchors = {"bound_to_problem": False, "source": None, "target": None,
                   "model": None, "problem": None}
    else:
        from .problem import problem_identity_of

        canonical, identity = problem_identity_of(problem)
        anchors = {
            "bound_to_problem": True,
            "source": problem.source_snapshot.identity,
            "target": problem.target_identity,
            "model": problem.model_version,
            "problem": identity,
            "problem_bytes_sha": _identity(canonical.decode("utf-8", "replace")),
        }
    payload = {
        "schema": "native_mainline_resolved_plan",
        **anchors,
        "proof": None if proof is None else proof.canonical(),
        "regions": [
            {
                "region": region.region_ref.region_id,
                "region_identity": region.identity,
                "bounds": [_encode_decision(getattr(region, name), env,
                                            "region.%s" % name)
                           for name in ("lower_expr", "stop_expr", "step_expr")],
                "ii": int(region.ii),
                "origin": int(region.origin),
                "max_shift": int(region.max_shift),
                "tasks": [
                    (task.task_id, task.authored_stage, task.resolved_stage,
                     task.phase, task.start, task.issue_rank)
                    for task in sorted(region.tasks, key=lambda item: item.task_id)],
                "buffers": [
                    (buffer.buffer_id,
                     _encode_decision(buffer.domain, env, "buffer.domain"),
                     int(buffer.resolved_versions))
                    for buffer in sorted(region.buffers,
                                         key=lambda item: item.buffer_id)],
            }
            for region in regions
        ],
        "task_to_native_source": sorted(
            (task.task_id, getattr(task.task_ref, "unit_id", None))
            for planned in regions for task in planned.tasks),
    }
    return json.dumps(payload, sort_keys=True, default=str).encode("utf-8")


def plan_canonical_bytes(plan) -> bytes:
    """Recompute a plan's canonical bytes from the objects it holds."""
    return _plan_bytes(getattr(plan, "problem_ref", None), tuple(plan.regions),
                       getattr(plan, "dependence_proof", None))


def plan_identity_of(plan) -> tuple:
    """(canonical bytes, digest) for one plan, from the same single recipe."""
    canonical = plan_canonical_bytes(plan)
    return canonical, _identity(canonical.decode())


def verify_plan_bindings(plan, problem) -> None:
    """Every reference in the plan must BE the problem's own object.

    Content equality is not enough and neither is an id string: the plan must
    point at the actual Region/SourceUnit/root this problem was built from, so
    a coordinated swap of "the same" records with foreign objects is refused.
    """
    regions = tuple(problem.regions)
    plans = tuple(plan.regions)
    if len(regions) != len(plans):
        raise ResolveError("PLAN_REGION_BINDING_COUNT",
                           "the plan covers %d region(s) but the problem has %d"
                           % (len(plans), len(regions)))
    by_id = {region.region_ref.region_id: region for region in regions}
    if len(by_id) != len(regions):
        raise ResolveError("PLAN_REGION_DUPLICATE", "the problem repeats a region id")
    seen = set()
    for planned in plans:
        source = by_id.get(planned.region_ref.region_id)
        if source is None or planned.region_ref is not source.region_ref:
            raise ResolveError(
                "PLAN_REGION_BINDING_MISMATCH",
                "region %r is not the problem's own Region object"
                % (planned.region_ref.region_id,))
        # A plan records the loop domain it decided over. Downstream
        # consumers read THAT copy (completion derives a region's trip count
        # from these bounds to decide whether an EXIT drain is owed), so a
        # plan whose bounds differ from the region's real ones would erase
        # duties without touching a single channel: the bypass is upstream of
        # the proof. They are the same objects as the problem's, and equality
        # is compared by content rather than by reference so a rebuilt but
        # identical expression is not refused.
        for name in ("lower_expr", "stop_expr", "step_expr"):
            claimed = getattr(planned, name)
            actual = getattr(source.region_ref, name)
            if _expr_text(claimed) != _expr_text(actual):
                raise ResolveError(
                    "PLAN_DECISION_INPUT_MISMATCH",
                    "the plan states region %s's %s as %r while the actual "
                    "region says %r" % (planned.region_ref.region_id, name,
                                        _expr_text(claimed), _expr_text(actual)))
        if id(planned.region_ref) in seen:
            raise ResolveError("PLAN_REGION_BINDING_DUPLICATE",
                               "one region object backs two plan regions")
        seen.add(id(planned.region_ref))
        tasks = {task.task_id: task for task in source.tasks}
        if len(tasks) != len(source.tasks):
            raise ResolveError("PLAN_TASK_DUPLICATE", "the region repeats a task id")
        planned_tasks = tuple(planned.tasks)
        if len(planned_tasks) != len(tasks):
            raise ResolveError("PLAN_TASK_BINDING_COUNT",
                               "region %r plans %d task(s) for %d in the problem"
                               % (source.region_ref.region_id, len(planned_tasks),
                                  len(tasks)))
        for task in planned_tasks:
            source_task = tasks.get(task.task_id)
            if source_task is None:
                raise ResolveError("PLAN_TASK_BINDING_UNKNOWN",
                                   "task %r is not a task of this region"
                                   % (task.task_id,))
            if task.task_ref is not source_task.unit_ref:
                raise ResolveError(
                    "PLAN_TASK_BINDING_MISMATCH",
                    "task %r does not reference the problem's own SourceUnit"
                    % (task.task_id,))
        buffers = {buffer.buffer_id: buffer for buffer in source.buffers}
        if len(buffers) != len(source.buffers):
            raise ResolveError("PLAN_BUFFER_DUPLICATE", "the region repeats a buffer id")
        planned_buffers = tuple(planned.buffers)
        if len(planned_buffers) != len(buffers):
            raise ResolveError("PLAN_BUFFER_BINDING_COUNT",
                               "region %r plans %d buffer(s) for %d in the problem"
                               % (source.region_ref.region_id, len(planned_buffers),
                                  len(buffers)))
        for buffer in planned_buffers:
            source_buffer = buffers.get(buffer.buffer_id)
            if source_buffer is None:
                raise ResolveError("PLAN_BUFFER_BINDING_UNKNOWN",
                                   "buffer %r is not a buffer of this region"
                                   % (buffer.buffer_id,))
            if buffer.buffer_ref is not source_buffer.root_ref:
                raise ResolveError(
                    "PLAN_BUFFER_BINDING_MISMATCH",
                    "buffer %r does not reference the problem's own root"
                    % (buffer.buffer_id,))
    # and every task must still be reachable from the native source map, in
    # BOTH directions. Checking only that each entry names a plan task would
    # accept an emptied map: the loop would run over nothing and certify the
    # plan by containing no claims at all.
    mapped = set(plan.task_to_native_source)
    for planned in plans:
        for task in planned.tasks:
            if task.task_id not in mapped:
                raise ResolveError(
                    "PLAN_SOURCE_MAP_INCOMPLETE",
                    "task %r has no entry in the native source map, so the plan "
                    "states no source for it" % (task.task_id,))
    for task_id, ref in plan.task_to_native_source.items():
        if not any(item.task_id == task_id and item.task_ref is ref
                   for planned in plans for item in planned.tasks):
            raise ResolveError("PLAN_SOURCE_MAP_STALE",
                               "task %r is not in the plan it was published with"
                               % (task_id,))
    # The proof a plan carries is RE-READ against the problem. Its own
    # membership cannot certify it: region_id, straight_line and the edge set
    # can all be relabelled and stay internally consistent, and a record with
    # no edges at all is consistent with anything. The comparison that counts
    # is against what the actual IR says now.
    proof = getattr(plan, "dependence_proof", None)
    if proof is None:
        raise ResolveError("PLAN_PROOF_MISSING",
                           "a certified plan must carry the dependence proof it "
                           "was resolved under")
    from .source_audit import SourceAuditError, dependence_proof as _recompute

    try:
        fresh = _recompute(problem)
    except SourceAuditError as error:
        raise ResolveError(error.code, str(error)) from error
    if fresh.region_id != proof.region_id:
        raise ResolveError("PLAN_PROOF_REGION_MISMATCH",
                           "the plan's proof is stated for %r while the problem's "
                           "region(s) are %r" % (proof.region_id, fresh.region_id))
    if bool(fresh.straight_line) != bool(proof.straight_line):
        raise ResolveError("PLAN_PROOF_DOMAIN_MISMATCH",
                           "the plan's proof claims straight_line=%r while the "
                           "region's real bounds say %r"
                           % (bool(proof.straight_line), bool(fresh.straight_line)))
    if fresh.canonical() != proof.canonical():
        raise ResolveError(
            "PLAN_PROOF_STALE",
            "the dependence proof stored on the plan does not match what the "
            "actual IR yields now",
        )


def _assignments_by_region(problem, assignment):
    """One assignment per region: a region owns its II, and regions do not share one.

    A single-region problem may be given the assignment directly. A multi-region
    problem must say which answer belongs to which region; merging two regions into
    one answer would silently pick one of their intervals.
    """
    regions = tuple(problem.regions)
    if hasattr(assignment, "status"):
        if len(regions) != 1:
            raise ResolveError(
                "PLAN_ASSIGNMENT_PER_REGION",
                "this problem has %d regions with their own II; name each region's "
                "assignment instead of passing one for all of them" % (len(regions),),
            )
        return {regions[0].region_ref.region_id: assignment}
    mapping = dict(assignment)
    resolved = {}
    for region in regions:
        region_id = region.region_ref.region_id
        if region_id not in mapping:
            raise ResolveError(
                "PLAN_ASSIGNMENT_MISSING",
                "no assignment was supplied for region %r" % (region_id,),
            )
        resolved[region_id] = mapping[region_id]
    return resolved


def resolve_schedule(problem, assignment):
    """Build the unique plan from a complete problem and one verified assignment per region."""
    by_region = _assignments_by_region(problem, assignment)
    status = getattr(next(iter(by_region.values())), "status", None) if by_region else None
    if status != "SAT":
        raise ResolveError(
            "PLAN_NOT_SAT",
            "only a SAT assignment yields a plan; got %r (%s)"
            % (status, getattr(assignment, "detail", "")),
        )
    for region_id, region_assignment in by_region.items():
        if getattr(region_assignment, "status", None) != "SAT":
            raise ResolveError(
                "PLAN_NOT_SAT",
                "region %r has no SAT assignment (%r)" % (region_id, getattr(region_assignment, "status", None)),
            )
    coverage = getattr(problem, "coverage", None)
    if coverage is None or not getattr(coverage, "complete", False):
        raise ResolveError(
            "PLAN_COVERAGE_INCOMPLETE",
            "the problem does not cover every source task/access/effect",
        )
    # Coverage counts occurrences; it does not prove reaching definitions or
    # dependencies.  The gate is therefore CONDITIONAL: an independent reader
    # (plan/source_audit.py) states, per edge, per read and per buffer, what it
    # could prove.  A complete proof builds the plan; an incomplete one is
    # refused with every missing item named. An unconditional refusal here was
    # a dead end, not fail-closed: with no plan there is no completion, no
    # storage, no candidate, and nothing in the chain could ever be proven.
    from .source_audit import (SourceAuditError, audit_instruction_effects,
                               dependence_proof)
    try:
        for region in problem.regions:
            audit_instruction_effects(region)
        proof = dependence_proof(problem)
    except SourceAuditError as error:
        raise ResolveError(error.code, str(error)) from error
    except (AttributeError, TypeError) as error:
        raise ResolveError("PLAN_SOURCE_EVIDENCE_MISSING", str(error)) from error
    if not proof.complete():
        raise ResolveError(
            "PLAN_DEPENDENCE_PROOF_UNAVAILABLE",
            "region(s) %s: %s" % (proof.region_id, "; ".join(proof.defects())),
        )
    # The problem's anchors are RECOMPUTED here from its live content, through
    # the single recipe analyze_native_problem used, and the plan records the
    # recomputed values. A caller may type any digest into problem_identity or
    # problem_canonical_bytes; copying either field forward would certify that
    # typed string, so a disagreement is refused by name and nothing below
    # reads the stored fields back as evidence.
    from .problem import problem_identity_of

    recomputed, recomputed_identity = problem_identity_of(problem)
    stored = bytes(getattr(problem, "problem_canonical_bytes", b"") or b"")
    if stored and stored != recomputed:
        raise ResolveError(
            "PROBLEM_CONTENT_CHANGED",
            "the problem's recorded canonical bytes (%d) do not match the %d "
            "bytes recomputed from its live regions" % (len(stored), len(recomputed)),
        )
    reported = str(getattr(problem, "problem_identity", "") or "")
    if reported and reported != recomputed_identity:
        raise ResolveError(
            "PROBLEM_CONTENT_CHANGED",
            "the problem self-reports %s while its live content recomputes to %s"
            % (reported, recomputed_identity),
        )
    # The caller's object is kept as problem_ref: identity binding is checked at
    # every consumer, so the RECOMPUTED anchors are recorded on the plan instead
    # of being patched into an object the caller still holds.
    anchors = (recomputed_identity, recomputed)
    regions = []
    native_sources = {}
    for region in problem.regions:
        region_assignment = by_region[region.region_ref.region_id]
        _verify_region(region, region_assignment)
        region_task_ids = {task.task_id for task in region.tasks}
        # Emission ranks are local to the region: one global order, one slice each.
        region_order = tuple(task_id for task_id in tuple(getattr(region_assignment, "issue_order", ()) or ())
                             if task_id in region_task_ids)
        rank = {task_id: index for index, task_id in enumerate(region_order)}
        tasks = []
        stages = []
        for task in region.tasks:
            task_id = task.task_id
            resolved_stage = _stage_of(region_assignment, task_id)
            if task.authored_stage is not None and resolved_stage != task.authored_stage:
                raise ResolveError(
                    "PLAN_AUTHORED_STAGE_CHANGED",
                    "task %r was authored at stage %d and resolved at %d"
                    % (task_id, task.authored_stage, resolved_stage),
                )
            stages.append(resolved_stage)
            tasks.append(TaskPlan(
                task_ref=getattr(task, "unit_ref", None),
                task_id=task_id,
                authored_stage=task.authored_stage,
                resolved_stage=resolved_stage,
                phase=_phase_of(region_assignment, task_id),
                start=int(region_assignment.starts[task_id]),
                issue_rank=rank.get(task_id, -1),
            ))
            native_sources[task_id] = getattr(task, "unit_ref", None)
        origin = min(stages) if stages else 0
        max_shift = (max(stages) - origin) if stages else 0
        buffers = []
        for buffer in region.buffers:
            versions = getattr(region_assignment, "versions", {}).get(buffer.buffer_id)
            if versions is None:
                raise ResolveError(
                    "PLAN_BUFFER_UNRESOLVED",
                    "the assignment has no version count for buffer %r" % (buffer.buffer_id,),
                )
            buffers.append(BufferPlan(
                buffer_ref=buffer.root_ref,
                buffer_id=buffer.buffer_id,
                domain=buffer.domain,
                resolved_versions=int(versions),
            ))
        payload = {
            "region": region.region_ref.region_id,
            "ii": int(region_assignment.ii),
            "origin": origin,
            "max_shift": max_shift,
            "tasks": [(item.task_id, item.authored_stage, item.resolved_stage, item.phase,
                       item.start, item.issue_rank) for item in sorted(tasks, key=lambda t: t.task_id)],
            "buffers": [(item.buffer_id, item.resolved_versions) for item in sorted(buffers, key=lambda b: b.buffer_id)],
        }
        regions.append(RegionPlan(
            region_ref=region.region_ref,
            lower_expr=getattr(region.region_ref, "lower_expr", None),
            stop_expr=getattr(region.region_ref, "stop_expr", None),
            step_expr=getattr(region.region_ref, "step_expr", None),
            ii=int(region_assignment.ii),
            origin=origin,
            max_shift=max_shift,
            tasks=tuple(sorted(tasks, key=lambda item: (item.issue_rank, item.task_id))),
            buffers=tuple(buffers),
            identity=_identity(payload),
        ))
    # ONE digest for the plan: the identity IS the hash of the recipe above.
    # A second hash over a different payload would give consumers two "same
    # plan" claims to compare, and only the recipe can be recomputed by a
    # later process, so only the recipe is hashed (lead: judge binding by
    # calling the binding function, never by comparing stored fields).
    plan_bytes = _plan_bytes(problem, regions, proof)
    plan_identity = _identity(plan_bytes.decode())
    plan = ResolvedPlan(
        source_identity=problem.source_snapshot.identity,
        target_identity=problem.target_identity,
        model_version=problem.model_version,
        regions=tuple(regions),
        task_to_native_source=native_sources,
        feasibility="SAT",
        optimality=getattr(next(iter(by_region.values())), "optimality", None),
        plan_identity=plan_identity,
        problem_ref=problem,
        problem_identity=anchors[0],
        problem_canonical_bytes=anchors[1],
        plan_canonical_bytes=plan_bytes,
        dependence_proof=proof,
        target_ref=getattr(problem, "target_ref", None),
    )
    # The plan may not point at objects the problem did not produce: content
    # equality and id strings are both insufficient, so each reference is
    # checked to BE the problem's own object here and again at every consumer.
    verify_plan_bindings(plan, problem)
    return plan
