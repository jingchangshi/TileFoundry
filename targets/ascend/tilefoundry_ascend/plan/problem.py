"""Complete NormalizedProblem over the typed UnitTree (schema section 2).

This is the second half of the S0 source-analysis boundary: it consumes the real
UnitTree built from the lowered TIR and produces the complete finite problem the
shared solver will receive. It does not solve, resolve, place or emit.

EVERY RECORD IS ROOTED IN A REAL OBJECT. Tasks point at SourceUnits, accesses at
the real instruction statement and operand, effects at the instruction schema and
the target instruction contract, buffers at the real allocation/function-parameter
Var, dependences at two real accesses. IDs are only stable indices; no lookup in
this module matches by name, shape or printed path.

WHAT THIS SLICE CANNOT SOURCE, and therefore reports instead of inventing:
- Dependence delay and distance are 0: the lowered program states no cross-
  iteration distance and this slice owns no cost model for a real completion
  delay. The solver model consumes the resource/latency facts and assigns time.
- Reuse entries exist only where a real later writer follows a maximal reader in
  one region; no version schedule exists yet, so distance_offset is 0.
- RegionProblem.issue_after / manual_pipe_order contain same-pipe source pairs;
  the S0 sample has no same-pipe siblings, so both are empty.
- LogicalBuffer.generation_delta is 0 (no versioned schedule exists); GM roots and
  non-loop roots are Fixed(1), a non-GM root accessed inside a loop with a
  PipelineSpec gets Auto(1, max_buffer_versions).
- Capacity records exist only for spaces whose byte budget is a current Target
  fact (UB for AIV). Register spaces have no sourced capacity and are omitted
  rather than backfilled.
- ii_max for an auto-II loop is the serial per-unit latency sum: an II above the
  serial body cannot be needed, so the search bound is finite without inventing a
  measured cost profile. fixed_ii keeps min=max=fixed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, is_dataclass, replace

from tilefoundry.ir.core import Constant
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.tir.memory.alloc_tensor import AllocTensor
from tilefoundry.ir.tir.memory.tensor_view import TensorView
from tilefoundry.ir.tir.stmts import Evaluate, For, LetStmt, MeshScope

from .units import (
    AccessFact,
    Auto,
    Fixed,
    OwnerKey,
    Region,
    SourceRefusal,
    SourceUnit,
    StorageKey,
    UnitTree,
    unit_accesses,
)

from tilefoundry_ascend.tf_target import UB_BYTES

__all__ = [
    "Capacity",
    "CoverageVerdict",
    "Dependence",
    "LogicalBuffer",
    "NormalizedProblem",
    "ProblemError",
    "RegionProblem",
    "ResourceFact",
    "Reuse",
    "Task",
    "analyze_native_problem",
    "problem_canonical_bytes",
    "problem_identity_of",
]

REFERENCE = "REFERENCE_IMPLEMENTATION"
_READ_MODES = ("READ", "READ_WRITE")
_WRITE_MODES = ("WRITE", "READ_WRITE")


class ProblemError(RuntimeError):
    """A named refusal from the normalized problem boundary."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__("%s: %s" % (code, detail))


@dataclass(frozen=True)
class Task:
    """One whole scheduling unit as the solver task."""

    task_id: str
    unit_ref: SourceUnit
    uses: tuple
    latency: int
    source_order: int
    authored_stage: int | None


@dataclass(frozen=True)
class ResourceFact:
    key: object
    capacity: int
    source: str
    status: str


@dataclass(frozen=True)
class LogicalBuffer:
    buffer_id: str
    root_ref: object
    allocation_ref: object
    owner: OwnerKey
    space: str
    dtype: object
    shape: tuple
    layout: object
    domain: object
    generation_delta: int
    slot_bytes: int
    alignment_bytes: int
    capacity_key: StorageKey | None
    initial_values: tuple
    terminal_readers: tuple


@dataclass(frozen=True)
class Dependence:
    producer: str
    consumer: str
    distance: int
    delay: int
    kind: str
    effect_ids: tuple
    #: Where the number above came from. None means UNSTATED: a bare 0 that no
    #: consumer may project (plan/delays.py records the only legal states).
    delay_fact: object | None = None

    def resolved_delay(self) -> int:
        """The number this edge's ordering must use, or a named refusal.

        A DelayFact resolves through its own state/unit rules; an explicitly
        stated integer (a caller's own declaration, e.g. a synthetic fixture) is
        used as given. None is never read as zero.
        """
        from .delays import DelayFact

        if isinstance(self.delay_fact, DelayFact):
            return self.delay_fact.resolve()
        if self.delay is None:
            from .delays import DelayRefusal

            raise DelayRefusal(
                "DELAY_UNSOURCED",
                "edge %s -> %s (%s) states neither a delay fact nor a number; an "
                "unstated delay is never projected into 0"
                % (self.producer, self.consumer, self.kind),
                "%s->%s" % (self.producer, self.consumer))
        return int(self.delay)


@dataclass(frozen=True)
class Reuse:
    last_reader: str
    next_writer: str
    buffer: str
    delay: int
    distance_offset: int
    delay_fact: object | None = None

    def resolved_delay(self) -> int:
        """A DelayFact resolves by its own rules; a stated number is used as
        given; None is never read as zero."""
        from .delays import DelayFact

        if isinstance(self.delay_fact, DelayFact):
            return self.delay_fact.resolve()
        if self.delay is None:
            from .delays import DelayRefusal

            raise DelayRefusal(
                "DELAY_UNSOURCED",
                "reuse of buffer %r states neither a delay fact nor a number; an "
                "unstated delay is never projected into 0" % (self.buffer,),
                self.buffer)
        return int(self.delay)


@dataclass(frozen=True)
class Capacity:
    storage_key: StorageKey
    buffers: tuple
    usable_bytes: int
    reserved_bytes: int
    source: str
    status: str


@dataclass(frozen=True)
class CoverageVerdict:
    source_identity: str
    problem_identity: str
    expected_tasks: frozenset
    covered_tasks: frozenset
    expected_accesses: frozenset
    covered_accesses: frozenset
    expected_effects: frozenset
    covered_effects: frozenset
    refusals: tuple

    @property
    def complete(self) -> bool:
        return (
            not self.refusals
            and self.expected_tasks == self.covered_tasks
            and self.expected_accesses == self.covered_accesses
            and self.expected_effects == self.covered_effects
        )


@dataclass(frozen=True)
class RegionProblem:
    region_ref: Region
    tasks: tuple
    buffers: tuple
    accesses: tuple
    effects: tuple
    resources: tuple
    owners: tuple
    pipes: tuple
    edges: tuple
    reuse: tuple
    capacities: tuple
    issue_after: tuple
    manual_pipe_order: tuple
    enable_offset: bool
    max_stage: int
    fixed_ii: int | None
    ii_min: int
    ii_max: int


@dataclass(frozen=True)
class NormalizedProblem:
    source_snapshot: object
    target_identity: str
    model_version: str
    unit_tree: UnitTree
    regions: tuple
    coverage: CoverageVerdict
    #: Content identity over the source snapshot, the actual target facts, the
    #: delay surface and every relation, edge and reuse obligation below.
    problem_identity: str = ""
    #: The canonical bytes problem_identity was computed over: consumers
    #: regenerate and compare, they do not trust a caller-supplied string.
    problem_canonical_bytes: bytes = b""
    #: The ACTUAL target instance the facts were queried from. A same-class
    #: replacement must re-enter analysis, not reuse this plan's binding.
    target_ref: object = None
    #: The delay table this problem was built against, hashed so an edit to any
    #: tick/basis/unit/status changes problem_identity.
    delay_table_digest: str = ""


def _identity(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _target_bytes(target):
    """Canonical bytes of the ACTUAL target, or a named refusal.

    There is no repr fallback: a target that publishes none of the queried
    capabilities cannot have its facts stated here, and inventing an identity
    from its class name would let two different machines share one plan.
    """
    if target is None:
        raise ProblemError("PROBLEM_TARGET_MISSING", "analyze_native_problem needs the target")
    from .identity import target_bytes

    return target_bytes(target)


def _target_identity(target) -> str:
    return _identity(_target_bytes(target).decode())


def _isa_rows(unit_tree: UnitTree) -> dict:
    from tilefoundry_ascend.ir.isa import ISA

    rows = {}
    for unit in unit_tree.units:
        for effect in unit.effects:
            schema = getattr(type(effect.source_ref.callable), "_op_schema", None)
            if schema is None or not schema.name.startswith("ascend_"):
                raise ProblemError(
                    "PROBLEM_COST_MODEL_UNSOURCED",
                    "an instruction has no target contract to state a cost model",
                )
            entry = ISA[schema.name[len("ascend_"):]]
            rows[entry.name] = {
                "resource": entry.resource,
                "completion": entry.completion,
                "core_affinity": entry.core_affinity,
                "latency_model": entry.latency_model,
                "ii_model": entry.ii_model,
            }
    return rows


def _model_version(target, unit_tree: UnitTree) -> str:
    return _identity(json.dumps({
        "target": getattr(target, "name", type(target).__name__),
        "isa": _isa_rows(unit_tree),
    }, sort_keys=True))


def _static_shape(tensor_type) -> tuple:
    from tilefoundry.ir.types.utils import static_dim_value

    shape = []
    for dim in tuple(getattr(tensor_type, "shape", ()) or ()):
        value = static_dim_value(dim)
        if value is None:
            raise ProblemError(
                "PROBLEM_BUFFER_DOMAIN_UNSUPPORTED",
                "a buffer shape dimension is not a static integer",
            )
        shape.append(int(value))
    return tuple(shape)


def _dtype_bytes(dtype) -> int:
    return max(1, int(getattr(dtype, "bit_width", 1)) // 8)


def _slot_bytes(tensor_type) -> int:
    count = 1
    for dim in _static_shape(tensor_type):
        count *= dim
    bits = int(getattr(tensor_type.dtype, "bit_width", 1))
    return max(1, (count * bits + 7) // 8)


def _space_of(tensor_type) -> str:
    storage = getattr(tensor_type, "storage", None)
    return getattr(storage, "name", str(storage)).lower()


def _binder_index(entry) -> dict:
    index = {}
    for parameter in entry.params:
        index[id(parameter)] = ("param", parameter)
    for item in _statements(entry.body):
        if isinstance(item, LetStmt):
            index[id(item.var)] = ("let", item)
        elif isinstance(item, For):
            index[id(item.induction_var)] = ("for", item)
        elif isinstance(item, MeshScope):
            index[id(item.binding)] = ("mesh", item)
    return index


def _statements(node):
    """Every statement in document order, itself included."""
    yield node
    if isinstance(node, (tuple, list)):
        for item in node:
            yield from _statements(item)
        return
    body = getattr(node, "body", None)
    if body is not None and node is not body:
        yield from _statements(body)
    for name in ("then_body", "else_body"):
        branch = getattr(node, name, None)
        if branch is not None:
            yield from _statements(branch)


def _region_chain(tree: UnitTree, region: Region) -> tuple:
    by_id = {item.region_id: item for item in tree.regions}
    chain = [region]
    parent = by_id.get(region.parent_id)
    while parent is not None:
        chain.append(parent)
        parent = by_id.get(parent.parent_id)
    return tuple(chain)


def _pipeline_of(region: Region):
    node = region.node_ref
    return getattr(node, "pipeline", None) if isinstance(node, For) else None


def _domain_region(tree: UnitTree, use_regions: tuple, fallback: Region) -> Region:
    """The region whose PipelineSpec governs a root: the deepest pipelined user."""
    for region in use_regions:
        if _pipeline_of(region) is not None:
            return region
    for region in use_regions:
        for item in _region_chain(tree, region):
            if _pipeline_of(item) is not None:
                return item
    return fallback


def _author_schedule(spec, root_var, buffer_id):
    """The author's published version domain for this buffer root, if any."""
    for schedule in getattr(spec, "buffer_schedules", ()) or ():
        reference = getattr(schedule, "buffer_ref", None)
        if reference is not None and reference is root_var:
            return schedule
        name = getattr(root_var, "name", None)
        if getattr(schedule, "root", None) in (buffer_id, name):
            return schedule
    return None


def _version_domain(tree: UnitTree, region: Region, space: str, *,
                    root_var=None, buffer_id="", matched=None):
    """The buffer's version domain: the author's published one, or the default.

    A published BufferSchedule is a hard author constraint. It is consumed here or
    refused by name; it is never dropped in favour of max_buffer_versions.
    """
    spec = None
    for item in _region_chain(tree, region):
        candidate = _pipeline_of(item)
        if candidate is not None:
            spec = candidate
            break
    if spec is not None:
        schedule = _author_schedule(spec, root_var, buffer_id)
        if schedule is not None:
            if matched is not None:
                matched.add(id(schedule))
            if space == "gmem":
                raise ProblemError(
                    "PROBLEM_BUFFER_SCHEDULE_UNSUPPORTED",
                    "buffer %r is a GM boundary tensor; a version ring is not published "
                    "for device-visible storage" % (buffer_id,),
                )
            if schedule.versions[0] == "fixed":
                return Fixed(int(schedule.versions[1]))
            return Auto(int(schedule.versions[1]), int(schedule.versions[2]))
    if space == "gmem":
        return Fixed(1)
    if spec is not None:
        return Auto(1, int(spec.max_buffer_versions))
    return Fixed(1)


def _buffer_facts(tree: UnitTree, binders: dict, root_var, region: Region,
                  owner: OwnerKey, buffer_id: str, matched=None) -> LogicalBuffer:
    binder = binders.get(id(root_var))
    if binder is None:
        raise ProblemError(
            "PROBLEM_BUFFER_ROOT_UNRESOLVED",
            "a buffer root Var has no binder in the lowered entry",
        )
    kind, node = binder
    if kind == "param":
        allocation_ref = root_var
    elif kind == "let":
        allocation_ref = node.value
    else:
        raise ProblemError(
            "PROBLEM_BUFFER_ROOT_UNRESOLVED",
            "a buffer root binds to %s rather than an allocation" % (kind,),
        )
    tensor_type = root_var.type
    space = _space_of(tensor_type)
    return LogicalBuffer(
        buffer_id=buffer_id,
        root_ref=root_var,
        allocation_ref=allocation_ref,
        owner=owner,
        space=space,
        dtype=tensor_type.dtype,
        shape=_static_shape(tensor_type),
        layout=getattr(tensor_type, "layout", None),
        domain=_version_domain(tree, region, space, root_var=root_var,
                              buffer_id=buffer_id, matched=matched),
        generation_delta=0,
        slot_bytes=_slot_bytes(tensor_type),
        alignment_bytes=_dtype_bytes(tensor_type.dtype),
        capacity_key=None if space == "gmem" else StorageKey(space, owner),
        initial_values=(),
        terminal_readers=(),
    )


def _grid_instances(entry) -> int:
    """How many program instances the entry's real MeshScope launches.

    Read from the lowered TIR, never assumed: same-pipe FIFO order is a
    per-instance guarantee, so a grid of N instances cannot be proven ordered
    from the pipe facts alone. A missing or multi-level mesh is a named
    refusal at this boundary rather than a silent 1.
    """
    from tilefoundry.ir.tir.stmts import MeshScope

    frames = []

    def visit(node):
        if node is None:
            return
        if isinstance(node, MeshScope):
            frames.append(node)
            visit(node.body)
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                visit(item)
            return
        for name in getattr(node, "__dataclass_fields__", ()) or ():
            if name in ("type", "metadata", "origin", "annotation", "loc"):
                continue
            visit(getattr(node, name, None))

    visit(getattr(entry, "body", None))
    if not frames:
        raise ProblemError(
            "PROBLEM_MESH_MISSING",
            "the entry launches no mesh scope, so no grid instance count exists "
            "and same-pipe ordering cannot be proven",
        )
    levels = tuple(getattr(frames[0].mesh, "topologies", ()) or ())
    if len(levels) != 1:
        raise ProblemError(
            "PROBLEM_MESH_UNSUPPORTED",
            "the mesh states %d level(s); this slice proves ordering for one "
            "level only" % (len(levels),),
        )
    try:
        return int(levels[0].size)
    except (AttributeError, TypeError, ValueError):
        raise ProblemError(
            "PROBLEM_MESH_EXTENT_UNSUPPORTED",
            "the mesh level extent %r is not an explicit integer, so the "
            "instance count is unknown" % (getattr(levels[0], "size", None),),
        ) from None


def _same_generation(left, right) -> bool:
    """Two accesses are the same generation only when provably identical."""
    from tilefoundry.ir.core import Constant, Var

    a, b = left.generation, right.generation
    if isinstance(a, Constant) and isinstance(b, Constant):
        return int(a.value) == int(b.value)
    if isinstance(a, Var) and isinstance(b, Var):
        # The loop induction variable is the generation of its own region: the
        # same object means the same iteration. Two different Vars do not.
        return a is b
    return False


def _generation_distance(left: AccessFact, right: AccessFact) -> int:
    """The ordinal difference between a producer and a consumer generation.

    Today only distance 0 is derivable from the facts the unit tree records. A
    carried relation (k -> k+1) needs the region's yield/carry identity, which
    this slice does not have, so it is refused by name rather than filled in.
    """
    if _same_generation(left, right):
        return 0
    raise ProblemError(
        "PROBLEM_DEPENDENCE_DISTANCE_UNSUPPORTED",
        "access %s (generation %r) and access %s (generation %r) are not the same "
        "generation, and this slice derives no carried distance; refusing instead "
        "of recording distance 0"
        % (left.access_id, getattr(left.generation, "value", left.generation),
           right.access_id, getattr(right.generation, "value", right.generation)),
    )


def _access_key(task_order: dict, access: AccessFact) -> tuple:
    """Program order of one access. An unknown task is refused, never ordered at 0.

    The default was a silent backfill: an access naming a task the region does
    not declare would sort before everything, so it could become no edge at all
    and nothing would say so. It is reachable - a caller-supplied
    RegionProblem whose accesses were replaced while tasks were not - and the
    same shape was already removed twice (reuse endpoints), so it is named here.
    """
    if access.task_id not in task_order:
        raise ProblemError(
            "PROBLEM_ACCESS_TASK_UNKNOWN",
            "access %r names task %r, which this region does not declare, so its "
            "place in program order cannot be established"
            % (access.access_id, access.task_id))
    return (task_order[access.task_id], access.operand_index)


def _dependences(accesses: tuple, effects: tuple, task_order: dict, *,
                 straight_line: bool, grid_instances: int) -> tuple:
    """RAW/WAR/WAW from real accesses; a read/read pair carries no data edge.

    The delay is never a literal 0 here: it comes from plan/delays.py, which
    states its state, basis, unit and source. The direction filter stays what it
    was (program order); proving that filter is the source audit's job, and the
    resolver refuses until that proof exists.
    """
    from .delays import declared_occupancy, edge_delay_fact

    effect_of = {access_id: effect for effect in effects for access_id in effect.access_ids}
    edges = []
    edge_index = {}
    for left in accesses:
        for right in accesses:
            if left.task_id == right.task_id or left.root_ref is not right.root_ref:
                continue
            left_write = left.mode in _WRITE_MODES
            right_write = right.mode in _WRITE_MODES
            if left_write and right_write:
                kind = "WAW"
            elif left_write and right.mode in _READ_MODES:
                kind = "RAW"
            elif left.mode in _READ_MODES and right_write:
                kind = "WAR"
            else:
                continue
            if _access_key(task_order, left) > _access_key(task_order, right):
                continue
            # The distance is DERIVED, never assumed: two accesses agree on a
            # generation only when their generation values are provably the same
            # instance (the same constant, or literally the same induction Var).
            # Anything else (k vs k+1, a carry, another region) is refused here by
            # name instead of being recorded as distance 0.
            distance = _generation_distance(left, right)
            key = (left.task_id, right.task_id, kind)
            pair_effects = {effect_of[left.access_id].effect_id,
                            effect_of[right.access_id].effect_id}
            if key in edge_index:
                index = edge_index[key]
                edge = edges[index]
                if int(edge.distance) != distance:
                    raise ProblemError(
                        "PROBLEM_DEPENDENCE_DISTANCE_INCONSISTENT",
                        "tasks %s -> %s mix generation distances %s and %d"
                        % (edge.producer, edge.consumer, edge.distance, distance))
                edges[index] = replace(edge, effect_ids=tuple(sorted(set(edge.effect_ids) | pair_effects)))
                continue
            producer = effect_of[left.access_id]
            consumer = effect_of[right.access_id]
            fact = edge_delay_fact(
                producer_instruction=producer.instruction,
                producer_pipe=producer.pipe,
                producer_completion=producer.completion,
                consumer_pipe=consumer.pipe,
                producer_occupancy=declared_occupancy(producer.instruction),
                same_region=True, straight_line=straight_line,
                # Both owners travel with the edge: same core KIND is not the
                # same instance, and only a shared instance can be ordered.
                producer_owner=left.owner, consumer_owner=right.owner,
                grid_instances=grid_instances, distance=0)
            edge_index[key] = len(edges)
            edges.append(Dependence(
                producer=left.task_id, consumer=right.task_id,
                distance=distance, delay=fact.ticks,
                kind=kind, effect_ids=tuple(sorted(pair_effects)),
                delay_fact=fact,
            ))
    return tuple(edges)


def _reuse(buffers: tuple, accesses: tuple, effects: tuple, task_order: dict) -> tuple:
    """Slot-reuse obligations, each carrying its own sourced read-completion delay.

    The number is the LAST READER's read completion: a slot may only be
    overwritten once that read has actually landed. Version wraparound is never
    in the proven-zero family (plan/delays.py).
    """
    from .delays import declared_occupancy, reuse_delay_fact

    effect_of = {access_id: effect for effect in effects for access_id in effect.access_ids}
    reuse = []
    for buffer in buffers:
        readers = [access for access in accesses
                   if access.root_ref is buffer.root_ref and access.mode in _READ_MODES]
        writers = [access for access in accesses
                   if access.root_ref is buffer.root_ref and access.mode in _WRITE_MODES]
        # "no reader was found" and "this root provably has no reader" used to
        # be one branch, which let an unexamined root drop its obligation
        # silently. The three cases are separated and only the one the actual
        # accesses prove empty is allowed to continue.
        if not readers and not writers:
            raise ProblemError(
                "PROBLEM_BUFFER_UNACCESSED",
                "buffer %r (root %r) is named by no access of this region, so "
                "neither its readers nor its writers were read from the actual IR"
                % (buffer.buffer_id, buffer.root_ref.name))
        if not readers and not writers:
            raise ProblemError(
                "PROBLEM_BUFFER_UNACCESSED",
                "buffer %r (root %r) is named by no access of this region, so "
                "neither its readers nor its writers were read from the actual IR"
                % (buffer.buffer_id, buffer.root_ref.name))
        if not readers or not writers:
            # One side is provably empty FROM THE REGION'S OWN ACCESS SET, not
            # merely unfound: a read-only root is an input whose definition the
            # reaching-definition audit traces to ENTRY (and refuses if it is a
            # local), a write-only root is an output nobody reads back, so no
            # slot can wrap around. The conclusion is derivable because every
            # access's root and mode is bound into the problem identity; the
            # case above is the one where nothing was examined at all.
            continue
        last_reader = max(readers, key=lambda access: _access_key(task_order, access))
        later = [access for access in writers
                 if access.task_id != last_reader.task_id
                 and _access_key(task_order, access) > _access_key(task_order, last_reader)]
        if not later:
            continue
        next_writer = min(later, key=lambda access: _access_key(task_order, access))
        reader = effect_of[last_reader.access_id]
        fact = reuse_delay_fact(
            producer_instruction=reader.instruction,
            producer_pipe=reader.pipe,
            producer_completion=reader.completion,
            producer_occupancy=declared_occupancy(reader.instruction),
            distance_offset=0)
        reuse.append(Reuse(
            last_reader=last_reader.access_id,
            next_writer=next_writer.access_id,
            buffer=buffer.buffer_id,
            delay=fact.ticks,
            distance_offset=0,
            delay_fact=fact,
        ))
    return tuple(reuse)


def _source_occurrences(entry) -> dict:
    occurrences = [statement for statement in _statements(entry.body)
                   if isinstance(statement, Evaluate)]
    return {id(statement): index for index, statement in enumerate(occurrences)}


def _coverage(tree: UnitTree, regions: tuple, problem_identity: str) -> CoverageVerdict:
    entry = tree.entry_ref
    occurrence_of = _source_occurrences(entry)
    expected_tasks = frozenset("src:task:%d" % index for index in occurrence_of.values())
    expected_accesses = set()
    expected_effects = set()
    for statement in _statements(entry.body):
        if not isinstance(statement, Evaluate):
            continue
        index = occurrence_of.get(id(statement))
        if index is None:
            continue
        schema = getattr(type(statement.callable), "_op_schema", None)
        parameters = tuple(param for param in schema.signature if param.kind == "input")
        for position, parameter in enumerate(parameters):
            if parameter.effect is not None:
                expected_accesses.add("src:task:%d:acc:%d" % (index, position))
        # Every lowered instruction occurrence owes exactly ONE effect record,
        # whether or not it has input operands. Deriving the obligation from the
        # parameter list silently exempts an attribute-only op: T.flag_set and
        # T.flag_wait carry all their facts as attributes and T.drain carries
        # none, so a parameter rule would count such a statement as a TASK but
        # never as an EFFECT - a dropped event effect would then leave coverage
        # saying complete. The obligation is stated from the statement itself
        # (lead ruling D1, task-47); it is the same shape as the delay rule
        # "an unstated fact is refused, never defaulted".
        expected_effects.add("src:task:%d" % index)

    covered_tasks = set()
    covered_accesses = set()
    covered_effects = set()
    covered_units = {}
    unit_by_identity = {id(unit): unit for unit in tree.units}
    foreign_units = []
    misplaced_units = []
    duplicate_units = []
    for region in regions:
        member_units = {id(child) for child in getattr(region.region_ref, "direct_children", ())}
        for task in region.tasks:
            unit = getattr(task, "unit_ref", None)
            key = id(unit)
            if key not in unit_by_identity:
                foreign_units.append(getattr(unit, "unit_id", repr(unit)))
                continue
            if key not in member_units:
                misplaced_units.append(unit.unit_id)
                continue
            covered_units[key] = covered_units.get(key, 0) + 1
            if covered_units[key] > 1:
                duplicate_units.append(unit.unit_id)
            # A unit covers the source occurrences it actually carries: the
            # instruction node itself and every effect statement inside a whole VF.
            statements = [getattr(unit, "node_ref", None)]
            statements.extend(effect.source_ref for effect in unit.effects)
            for statement in statements:
                index = occurrence_of.get(id(statement))
                if index is not None:
                    covered_tasks.add("src:task:%d" % index)
        for access in region.accesses:
            index = occurrence_of.get(id(access.source_ref))
            if index is not None:
                covered_accesses.add("src:task:%d:acc:%d" % (index, access.operand_index))
        for effect in region.effects:
            index = occurrence_of.get(id(effect.source_ref))
            if index is not None:
                covered_effects.add("src:task:%d" % index)

    refusals = []
    expected_unit_keys = set(unit_by_identity)
    covered_keys = set(covered_units)
    if expected_unit_keys - covered_keys:
        refusals.append(SourceRefusal(
            "SOURCE_COVERAGE_UNITS_UNMAPPED", entry,
            "%d whole-task unit(s) are not scheduling tasks of any region"
            % (len(expected_unit_keys - covered_keys),),
        ))
    if foreign_units:
        refusals.append(SourceRefusal(
            "SOURCE_COVERAGE_TASK_FOREIGN", entry,
            "%d task(s) reference a unit that is not in this tree: %s"
            % (len(foreign_units), ", ".join(sorted(foreign_units)[:4])),
        ))
    if misplaced_units:
        refusals.append(SourceRefusal(
            "SOURCE_COVERAGE_TASK_REGION_MISMATCH", entry,
            "%d task(s) place a unit outside its own region: %s"
            % (len(misplaced_units), ", ".join(sorted(misplaced_units)[:4])),
        ))
    if duplicate_units:
        refusals.append(SourceRefusal(
            "SOURCE_COVERAGE_TASK_DUPLICATE", entry,
            "%d unit(s) are scheduled more than once: %s"
            % (len(duplicate_units), ", ".join(sorted(duplicate_units)[:4])),
        ))
    if expected_tasks - covered_tasks:
        refusals.append(SourceRefusal(
            "SOURCE_COVERAGE_TASKS_UNMAPPED", entry,
            "%d source instruction(s) have no whole-task unit"
            % (len(expected_tasks - covered_tasks),),
        ))
    if expected_accesses - covered_accesses:
        refusals.append(SourceRefusal(
            "SOURCE_COVERAGE_ACCESSES_UNMAPPED", entry,
            "%d source access occurrence(s) are not in the problem"
            % (len(expected_accesses - covered_accesses),),
        ))
    if expected_effects - covered_effects:
        refusals.append(SourceRefusal(
            "SOURCE_COVERAGE_EFFECTS_UNMAPPED", entry,
            "%d source effect occurrence(s) are not in the problem"
            % (len(expected_effects - covered_effects),),
        ))
    return CoverageVerdict(
        source_identity=tree.source_snapshot.identity,
        problem_identity=problem_identity,
        expected_tasks=expected_tasks,
        covered_tasks=frozenset(covered_tasks),
        expected_accesses=frozenset(expected_accesses),
        covered_accesses=frozenset(covered_accesses),
        expected_effects=frozenset(expected_effects),
        covered_effects=frozenset(covered_effects),
        refusals=tuple(refusals),
    )


def _pipeline_fields(region: Region, tasks: tuple) -> dict:
    spec = _pipeline_of(region)
    serial = max(1, sum(task.latency for task in tasks)) if tasks else 1
    if spec is None:
        # A straight-line region runs once: its interval is its serial span, not a
        # cyclic slot, so the ring capacity model must not force II to 1.
        return {"enable_offset": False, "max_stage": 0, "fixed_ii": None,
                "ii_min": serial, "ii_max": serial}
    fixed = getattr(spec, "fixed_ii", None)
    return {
        "enable_offset": bool(spec.enable_offset),
        "max_stage": int(spec.max_stage),
        "fixed_ii": None if fixed is None else int(fixed),
        "ii_min": 1 if fixed is None else int(fixed),
        "ii_max": serial if fixed is None else int(fixed),
    }


def _same_pipe_pairs(tasks: tuple, *, consecutive: bool) -> tuple:
    pairs = []
    for position, (left, right) in enumerate(zip(tasks, tasks[1:])):
        if not left.uses or not right.uses:
            continue
        pipe = left.uses[0].resource.pipe
        if right.uses[0].resource.pipe != pipe:
            continue
        if consecutive:
            pairs.append((left.task_id, right.task_id))
        elif any(task.uses and task.uses[0].resource.pipe == pipe
                 for task in tasks[:position]):
            pairs.append((left.task_id, right.task_id))
    return tuple(pairs)


def _buffer_owner(accesses: tuple) -> OwnerKey:
    return accesses[0].owner if accesses else OwnerKey(None, "DEVICE", None)


def analyze_native_problem(units: UnitTree, effects=None, alias=None, target=None
                           ) -> NormalizedProblem:
    """Build the complete NormalizedProblem and its independent source coverage.

    effects is the ordered unit-effect tuple (None takes them from the tree; a
    supplied value must be the same real records). alias is reserved for declared
    alias facts: this slice resolves lowered buffer roots from their real
    address/binder definitions, so None is the honest value and a non-None value
    must be a tuple. target is the real Target instance.
    """
    if not isinstance(units, UnitTree):
        raise TypeError("analyze_native_problem expects a UnitTree, got %s"
                        % (type(units).__name__,))
    unit_effects = units.all_effects()
    effect_records = unit_effects if effects is None else tuple(effects)
    if tuple(id(effect) for effect in effect_records) != tuple(id(effect) for effect in unit_effects):
        raise ProblemError(
            "PROBLEM_EFFECTS_MISMATCH",
            "the supplied effects are not the unit tree's real effect records",
        )
    if alias is not None and not isinstance(alias, tuple):
        raise ProblemError(
            "PROBLEM_ALIAS_INVALID",
            "alias facts are supplied as a tuple of real records, not a dict or string",
        )
    target_identity = _target_identity(target)
    model_version = _model_version(target, units)
    binders = _binder_index(units.entry_ref)

    region_accesses = {}
    region_effects = {}
    for region in units.regions:
        direct = tuple(child for child in region.direct_children
                       if isinstance(child, SourceUnit))
        accesses = tuple(access for unit in direct for access in unit_accesses(unit))
        region_accesses[region.region_id] = (direct, accesses,
                                             tuple(effect for unit in direct
                                                   for effect in unit.effects))

    # One LogicalBuffer per real root, with the version domain of its pipelined use.
    use_regions = {}
    for region in units.regions:
        _direct, accesses, _effects = region_accesses[region.region_id]
        for access in accesses:
            use_regions.setdefault(id(access.root_ref), []).append(region)
    buffer_of = {}
    buffer_records = []
    matched_schedules = set()
    for region in units.regions:
        direct, accesses, _effects = region_accesses[region.region_id]
        for access in accesses:
            key = id(access.root_ref)
            if key in buffer_of:
                continue
            domain_region = _domain_region(units, tuple(use_regions[key]), region)
            record = _buffer_facts(units, binders, access.root_ref, domain_region,
                                   _buffer_owner(accesses),
                                   "buffer:%d" % len(buffer_records),
                                   matched=matched_schedules)
            buffer_of[key] = record
            buffer_records.append(record)

    for region in units.regions:
        spec = _pipeline_of(region)
        if spec is None:
            continue
        for schedule in getattr(spec, "buffer_schedules", ()) or ():
            if id(schedule) not in matched_schedules:
                raise ProblemError(
                    "PROBLEM_BUFFER_SCHEDULE_UNMATCHED",
                    "an authored BufferSchedule(root=%r) names no buffer of this program"
                    % (getattr(schedule, "root", None),),
                )

    # The instance count is one program-wide fact, read once from the real
    # lowered mesh scope rather than re-derived per region.
    grid_instances = _grid_instances(units.entry_ref)
    initial_values = _carried_initials(units, binders)
    buffer_records = [
        replace(record, initial_values=initial_values.get(id(record.root_ref), ()))
        for record in buffer_records
    ]
    by_root = {id(record.root_ref): record for record in buffer_records}

    problems = []
    for region in units.regions:
        direct, accesses, effects = region_accesses[region.region_id]
        owner = _buffer_owner(accesses)
        task_order = {unit.unit_id: index for index, unit in enumerate(direct)}
        tasks = []
        for unit in direct:
            latency = max((use.offset + use.duration for use in unit.resources), default=1)
            authored_stage = None
            spec = _pipeline_of(region)
            if spec is not None:
                for intent in getattr(spec, "task_intents", ()):
                    if intent.anchor == unit.source_order:
                        authored_stage = intent.stage
            tasks.append(Task(
                task_id=unit.unit_id,
                unit_ref=unit,
                uses=unit.resources,
                latency=max(1, latency),
                source_order=unit.source_order,
                authored_stage=authored_stage,
            ))
        task_tuple = tuple(tasks)
        roots = []
        for access in accesses:
            if not any(access.root_ref is root for root in roots):
                roots.append(access.root_ref)
        terminal = {}
        for root in roots:
            readers = [access for access in accesses
                       if access.root_ref is root and access.mode in _READ_MODES]
            if readers:
                terminal[id(root)] = (readers[-1].access_id,)
        buffers = tuple(
            replace(by_root[id(root)], terminal_readers=terminal.get(id(root), ()))
            for root in roots
        )
        resources = []
        owners = []
        pipes = []
        for task in task_tuple:
            for use in task.uses:
                if not any(use.resource == item.key for item in resources):
                    resources.append(ResourceFact(
                        key=use.resource,
                        capacity=1,
                        source="tilefoundry_ascend.ir.isa.InstructionContract.resource",
                        status=REFERENCE,
                    ))
                if not any(use.resource.owner == item for item in owners):
                    owners.append(use.resource.owner)
                if use.resource.pipe not in pipes:
                    pipes.append(use.resource.pipe)
        capacities = []
        for buffer in buffers:
            if buffer.capacity_key is None or buffer.space != "smem":
                continue
            if any(item.storage_key == buffer.capacity_key for item in capacities):
                continue
            capacities.append(Capacity(
                storage_key=buffer.capacity_key,
                buffers=tuple(item.buffer_id for item in buffers
                              if item.capacity_key == buffer.capacity_key),
                usable_bytes=UB_BYTES,
                reserved_bytes=0,
                source="tilefoundry_ascend.tf_target.UB_BYTES",
                status=REFERENCE,
            ))
        problems.append(RegionProblem(
            region_ref=region,
            tasks=task_tuple,
            buffers=buffers,
            accesses=accesses,
            effects=effects,
            resources=tuple(resources),
            owners=tuple(owners),
            pipes=tuple(pipes),
            # A loop region is not a straight line across iterations, so its
            # intra-iteration edges never enter the proven-zero family; they
            # resolve through the registered model instead of a literal 0.
            edges=_dependences(accesses, effects, task_order,
                               straight_line=region.induction_var_ref is None,
                               grid_instances=grid_instances),
            reuse=_reuse(buffers, accesses, effects, task_order),
            capacities=tuple(capacities),
            issue_after=_same_pipe_pairs(task_tuple, consecutive=True),
            manual_pipe_order=_same_pipe_pairs(task_tuple, consecutive=False),
            **_pipeline_fields(region, task_tuple),
        ))

    region_profiles = tuple(problems)
    from .delays import delay_table_bytes

    # One recipe. The anchors are computed from the LIVE content through the
    # same function every consumer calls; nothing here reads a stored digest
    # back, so a field a caller typed cannot certify itself.
    bound = NormalizedProblem(
        source_snapshot=units.source_snapshot,
        target_identity=target_identity,
        model_version=model_version,
        unit_tree=units,
        regions=region_profiles,
        coverage=_coverage(units, region_profiles, ""),
        problem_identity="",
        problem_canonical_bytes=b"",
        target_ref=target,
        delay_table_digest=_identity(delay_table_bytes().decode("utf-8", "replace")),
    )
    canonical = problem_canonical_bytes(bound)
    problem_identity = problem_identity_of(bound)[1]
    return replace(
        bound,
        coverage=_coverage(units, region_profiles, problem_identity),
        problem_identity=problem_identity,
        problem_canonical_bytes=canonical,
    )



def _dataclass_fields(value):
    from dataclasses import fields as _fields

    return _fields(value) if is_dataclass(value) else ()


def _source_values(values, names: dict, require_named, path: str) -> list:
    """Locate, then encode, source-side expressions in the merged name table."""
    from .identity import _encode

    for index, value in enumerate(values):
        require_named(value, "%s[%d]" % (path, index))
    return [_encode(value, names, "%s[%d]" % (path, index), 0)
            for index, value in enumerate(values)]


def problem_canonical_bytes(problem) -> bytes:
    """The ONE recipe binding a problem's full content, per SOURCE_SNAPSHOT_DESIGN §4.

    Every data field of every record is encoded, and object references become
    POSITIONAL labels taken from the same traversal that produces source_bytes
    (identity.source_labels): the identity points at statements and binders in
    the actual TIR instead of at Python addresses, and instead of merely
    listing ids. Nothing here reads problem_identity or the stored bytes, so a
    field a caller typed can never certify itself.

    A value this encoder cannot represent raises IdentityRefusal naming the
    field path: that is a content-binding gap, never something to skip over.
    """
    from .delays import delay_table_bytes
    from .identity import IdentityRefusal, _encode, source_labels

    entry = problem.source_snapshot.entry_ref
    env = source_labels(entry)

    def ref(node, path: str):
        """Locate one TIR object or source-side binder by position; refuse absence.

        The merged table is what makes this single lookup meaningful: a
        lowered statement occurrence and a name a source loop declares are
        different origins ("s<n>" vs "b<n>") and a value from one table can
        never be mistaken for a value from the other.
        """
        if node is None:
            return {"#": "absent", "path": path}
        label = names.get(id(node))
        if label is None:
            raise IdentityRefusal(path, "the object is bound by neither a "
                                        "statement of this entry's actual TIR nor "
                                        "a binder table of its loop summaries, so "
                                        "the identity cannot locate it")
        return {"#": "at", "label": label}

    def side(value, path: str) -> dict:
        """Locate a NAME, describe a VALUE.

        A Var only means something at the position that binds it, so it must
        be in the merged table or the identity refuses it. An address recipe
        (View) or a constant carries its own content and lowering gives it no
        TIR position to point at, so it is encoded structurally. Refusing both
        ways would be wrong: for a name it would hide an unbound binder, for
        a value it would demand a position that does not exist.
        """
        from tilefoundry.ir.core import Var

        if isinstance(value, Var):
            return ref(value, path)
        return {"#": "content", "value": enc(value, path)}

    def _pair_rows(rows, path: str) -> list:
        """(source -> actual) binding pairs, each side by the rule above."""
        return sorted([[side(key, "%s.key" % path), side(value, "%s.value" % path)]
                       for key, value in rows],
                      key=lambda row: json.dumps(row, sort_keys=True))

    def _item_rows(values, path: str) -> list:
        """A plain sequence of definitions or allocations, one row per item.

        actual_defs and actual_allocations are sequences of OBJECTS, not of
        pairs; only binding_map pairs source with actual. Reading them as
        pairs raised TypeError on the first bare Var, so the shapes are
        stated apart here rather than assumed to agree.
        """
        return [side(value, "%s[%d]" % (path, index))
                for index, value in enumerate(values)]

    def enc(value, path: str):
        # ONE merged table: lowered occurrences (s<n>) and source-side binders
        # (b<n>). A field that belongs to the source loop encodes through this
        # same function; nothing here needs to know which side a value came
        # from, which is why one table beats one table per region.
        return _encode(value, names, path, 0)

    # ------------------------------------------------------------------
    # ONE name table for the whole problem, with two origins.
    #
    # Lowered TIR objects get a positional label from identity.source_labels.
    # A carried initial value, however, is an expression of the SOURCE loop:
    # the lowered TIR binds nothing for it, and the authored dsl Function is
    # not a PrimFunction, so no TIR traversal can label it at all (measured:
    # summary.start/stop/step/carried_params/carried_initials/actual_bounds
    # appear in no lowered label table). Such a value only means something
    # inside the binder table its own summary declares, so those names are
    # given positional labels too - "b<n>", ordered by first appearance - and
    # that table is encoded INTO the problem bytes, which is what makes the
    # labels interpretable rather than assumed. (lead ruling: option (a), with
    # the binder table bound into identity.)
    #
    # The two prefixes never collide, so one merged table serves every field,
    # and task-46's carried-distance derivation reads the same table.
    # ------------------------------------------------------------------
    from tilefoundry.ir.core import Var

    source_rows = []
    source_env = {}

    def declare(node, origin: str):
        if not isinstance(node, Var) or id(node) in source_env or id(node) in env:
            return
        label = "b%d" % len(source_rows)
        source_env[id(node)] = label
        source_rows.append([label, origin,
                            _encode(node.type, source_env, "$.binder.type", 0),
                            bool(node.is_const)])

    def collect_vars(value, origin: str, seen: set):
        """Register every Var a source-side object binds or reads.

        yield_values are ExecutionRegion objects whose own parameters are the
        names the loop yields, so declaration has to walk into them: a name
        that is not in the table cannot be located, and a value whose parts
        are located inconsistently would bind half a fact.
        """
        if id(value) in seen:
            return
        if isinstance(value, Var):
            seen.add(id(value))
            declare(value, origin)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            seen.add(id(value))
            for item in value:
                collect_vars(item, origin, seen)
            return
        fields = _dataclass_fields(value)
        if fields:
            seen.add(id(value))
            for field in fields:
                collect_vars(getattr(value, field.name, None), origin, seen)

    def collect_source(summary):
        seen: set = set()
        declare(summary.induction_var, "induction_var")
        for name in ("carried_params", "yield_values", "captures"):
            for node in getattr(summary, name, ()) or ():
                collect_vars(node, name, seen)

    def collect_units(unit):
        """Declare the names a scheduling unit reads from its source loop.

        carried_values are (source parameter, actual definition) PAIRS, not
        bare Vars, and yields are source-side objects the lowered TIR never
        binds, so the same recursive walk is used for them. Without this the
        identity would carry a name it cannot locate.
        """
        seen: set = set()
        for name in ("carried_values", "yields", "captured_values"):
            for node in getattr(unit, name, ()) or ():
                collect_vars(node, name, seen)

    for region in problem.regions:
        if region.region_ref.binding_summary is not None:
            collect_source(region.region_ref.binding_summary)
    for unit in problem.unit_tree.units:
        collect_units(unit)

    names = dict(env)
    names.update(source_env)

    def require_named(value, path: str):
        """Every Var anywhere in a bound value must be in the name table.

        Measured today: the source pools cover every Var a carried initial
        reads. That must not become an assumption - an unnamed Var would
        either be refused with a bare 'outside its lexical scope' or, worse,
        go unbound, which is the silent-loss shape (D-class) this task exists
        to close. So it is refused here, naming the Var and the path that
        reached it.
        """
        if isinstance(value, Var):
            if id(value) not in names:
                raise IdentityRefusal(
                    path, "the problem states %r, which is bound by neither the "
                          "actual TIR nor a loop summary's binder table"
                          % (value.name,))
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for index, item in enumerate(value):
                require_named(item, "%s[%d]" % (path, index))
            return
        fields = _dataclass_fields(value)
        for field in fields:
            if field.name in ("loc", "metadata", "annotation"):
                continue
            require_named(getattr(value, field.name, None), "%s.%s" % (path, field.name))

    rows = []
    for region in problem.regions:
        locator = region.region_ref
        binding = locator.binding_summary
        rows.append({
            "region": locator.region_id,
            "parent": locator.parent_id,
            # the region's own source node. It is a FUNCTION, not a statement:
            # the entry's own body is where a top-level region comes from, and
            # a nested region's node is the statement that opened it. Both are
            # positional, so pointing a region at another node moves identity.
            "node": ({"#": "entry"} if locator.node_ref is entry
                     else ref(locator.node_ref, "region.node_ref")),
            "induction_var": ref(locator.induction_var_ref, "region.induction_var_ref"),
            "lower": enc(locator.lower_expr, "region.lower_expr"),
            "stop": enc(locator.stop_expr, "region.stop_expr"),
            "step": enc(locator.step_expr, "region.step_expr"),
            "activation_guard": enc(locator.activation_guard, "region.activation_guard"),
            # binding_summary is None for a straight-line region; for a loop it
            # describes the SOURCE loop the actual For was built from. Its
            # binding_map/actual_defs are sequences of pairs, not dicts.
            "binding": (None if binding is None else {
                "induction_var": ref(binding.induction_var, "binding.induction_var"),
                "start": enc(binding.start, "binding.start"),
                "stop": enc(binding.stop, "binding.stop"),
                "step": enc(binding.step, "binding.step"),
                "carried_params": enc(binding.carried_params, "binding.carried_params"),
                "carried_initials": enc(binding.carried_initials, "binding.carried_initials"),
                "yield_values": enc(binding.yield_values, "binding.yield_values"),
                "captures": enc(binding.captures, "binding.captures"),
                "actual_loop": ref(binding.actual_loop, "binding.actual_loop"),
                "actual_induction_var": ref(binding.actual_induction_var,
                                            "binding.actual_induction_var"),
                "binding_map": _pair_rows(binding.binding_map, "binding.map"),
                "actual_defs": _item_rows(binding.actual_defs, "binding.defs"),
                "actual_allocations": _item_rows(binding.actual_allocations,
                                                  "binding.allocations"),
                "actual_bounds": enc(binding.actual_bounds, "binding.actual_bounds"),
            }),
            "policy": {
                "enable_offset": region.enable_offset,
                "max_stage": region.max_stage,
                "fixed_ii": region.fixed_ii,
                "ii_min": region.ii_min,
                "ii_max": region.ii_max,
            },
            "tasks": [
                {
                    "task_id": task.task_id,
                    "unit_id": task.unit_ref.unit_id,
                    "unit_statement": ref(task.unit_ref.node_ref, "task.unit_ref.node_ref"),
                    "whole_body": ref(task.unit_ref.whole_body, "task.unit_ref.whole_body"),
                    "kind": task.unit_ref.kind,
                    "source_order": task.source_order,
                    "authored_stage": task.authored_stage,
                    "latency": task.latency,
                    "owner": enc(task.unit_ref.execution_owner, "task.execution_owner"),
                    "guard": enc(task.unit_ref.original_guard, "task.original_guard"),
                    "uses": enc(task.uses, "task.uses"),
                    "captured": [side(v, "task.captured_values")
                                 for v in task.unit_ref.captured_values],
                    "carried": [side(v, "task.carried_values")
                                for v in task.unit_ref.carried_values],
                    "yields": [side(v, "task.yields") for v in task.unit_ref.yields],
                    "effects": sorted(effect.effect_id for effect in task.unit_ref.effects),
                    "intent": task.unit_ref.intent,
                }
                for task in sorted(region.tasks, key=lambda item: item.task_id)
            ],
            "buffers": [
                {
                    "buffer_id": buffer.buffer_id,
                    "root": ref(buffer.root_ref, "buffer.root_ref"),
                    "allocation": enc(buffer.allocation_ref, "buffer.allocation_ref"),
                    "owner": enc(buffer.owner, "buffer.owner"),
                    "space": buffer.space,
                    "dtype": enc(buffer.dtype, "buffer.dtype"),
                    "shape": enc(buffer.shape, "buffer.shape"),
                    "layout": enc(buffer.layout, "buffer.layout"),
                    "domain": enc(buffer.domain, "buffer.domain"),
                    "generation_delta": buffer.generation_delta,
                    "slot_bytes": buffer.slot_bytes,
                    "alignment_bytes": buffer.alignment_bytes,
                    "capacity_key": enc(buffer.capacity_key, "buffer.capacity_key"),
                    # a carried initial is a SOURCE-side expression: it is
                    # located in the loop summary's binder table, never in the
                    # lowered TIR table, and each Var it reads is checked
                    # against that table before anything is encoded
                    "initial_values": _source_values(buffer.initial_values, names,
                                                     require_named,
                                                     "buffer.initial_values"),
                    "terminal_readers": sorted(buffer.terminal_readers),
                }
                for buffer in sorted(region.buffers, key=lambda item: item.buffer_id)
            ],
            "accesses": [
                {
                    "access_id": access.access_id,
                    "task_id": access.task_id,
                    "source_statement": ref(access.source_ref, "access.source_ref"),
                    "operand_index": access.operand_index,
                    "root": ref(access.root_ref, "access.root_ref"),
                    "view": ref(access.view_ref, "access.view_ref"),
                    "relation": enc(access.relation, "access.relation"),
                    "mode": access.mode,
                    "owner": enc(access.owner, "access.owner"),
                    "guard": enc(access.guard, "access.guard"),
                    "generation": enc(access.generation, "access.generation"),
                }
                for access in sorted(region.accesses, key=lambda item: item.access_id)
            ],
            "effects": [
                {
                    "effect_id": effect.effect_id,
                    "task_id": effect.task_id,
                    "source_statement": ref(effect.source_ref, "effect.source_ref"),
                    "kind": effect.kind,
                    "access_ids": sorted(effect.access_ids),
                    "state_ref": side(effect.state_ref, "effect.state_ref"),
                    "generation": enc(effect.generation, "effect.generation"),
                    "guard": enc(effect.guard, "effect.guard"),
                    "completion": effect.completion,
                    "pipe": effect.pipe,
                    "instruction": effect.instruction,
                }
                for effect in sorted(region.effects, key=lambda item: item.effect_id)
            ],
            "resources": enc(sorted(region.resources, key=str), "region.resources"),
            "owners": enc(sorted(region.owners, key=str), "region.owners"),
            "pipes": enc(sorted(region.pipes, key=str), "region.pipes"),
            "capacities": enc(sorted(region.capacities, key=str), "region.capacities"),
            "issue_after": enc(region.issue_after, "region.issue_after"),
            "manual_pipe_order": enc(region.manual_pipe_order, "region.manual_pipe_order"),
            "edges": [
                {
                    "producer": edge.producer,
                    "consumer": edge.consumer,
                    "kind": edge.kind,
                    "distance": int(edge.distance),
                    "delay": edge.delay,
                    "effect_ids": sorted(edge.effect_ids),
                    "fact": edge.delay_fact.canonical(),
                }
                for edge in sorted(region.edges,
                                   key=lambda item: (item.producer, item.consumer,
                                                     item.kind))
            ],
            "reuse": [
                {
                    "last_reader": item.last_reader,
                    "next_writer": item.next_writer,
                    "buffer": item.buffer,
                    "distance_offset": int(item.distance_offset),
                    "delay": item.delay,
                    "fact": item.delay_fact.canonical(),
                }
                for item in sorted(region.reuse,
                                   key=lambda item: (item.buffer, item.last_reader,
                                                     item.next_writer))
            ],
        })
    coverage = problem.coverage
    # The three echoes a caller may type are RECOMPUTED from the objects they
    # describe and compared: hashing the stored string instead would let a
    # caller certify any content by typing any digest, and ignoring it
    # altogether would leave the field unbound and unrefused (the silent-loss
    # shape). Agreement is required, and disagreement names which echo lied.
    target_ref = getattr(problem, "target_ref", None)
    if target_ref is not None:
        recomputed_target = _target_identity(target_ref)
        stated_target = str(getattr(problem, "target_identity", "") or "")
        if stated_target and stated_target != recomputed_target:
            raise ProblemError(
                "PROBLEM_CONTENT_CHANGED",
                "the problem self-reports target %s while its own target_ref "
                "recomputes to %s" % (stated_target, recomputed_target))
    else:
        recomputed_target = problem.target_identity
    unit_tree = getattr(problem, "unit_tree", None)
    if target_ref is not None and unit_tree is not None:
        recomputed_model = _model_version(target_ref, unit_tree)
        stated_model = str(getattr(problem, "model_version", "") or "")
        if stated_model and stated_model != recomputed_model:
            raise ProblemError(
                "PROBLEM_CONTENT_CHANGED",
                "the problem self-reports model %s while its target and unit "
                "tree recompute to %s" % (stated_model, recomputed_model))
    else:
        recomputed_model = problem.model_version
    delays_digest = hashlib.sha256(delay_table_bytes()).hexdigest()
    stated_delays = str(getattr(problem, "delay_table_digest", "") or "")
    if stated_delays:
        recomputed_delays = _identity(delay_table_bytes().decode("utf-8", "replace"))
        if stated_delays != recomputed_delays:
            raise ProblemError(
                "PROBLEM_CONTENT_CHANGED",
                "the problem self-reports a delay table digest that the live "
                "delay table does not reproduce")
    return json.dumps({
        # Every bound value carries the encoder version through _tag(); the
        # recipe itself does not restate it, because a version number written
        # twice is a version number that can drift out of agreement with the
        # one the encoder stamps.
        "source": problem.source_snapshot.identity,
        "target": recomputed_target,
        # Design section 4: the cost model is part of the problem's content.
        "model": recomputed_model,
        "delays": delays_digest,
        # the binder table that gives every b<n> label above its meaning: bound
        # into the identity, so dropping, adding or reordering one binder moves
        # the digest and a label can never be reinterpreted silently
        "source_binders": source_rows,
        "regions": rows,
        "coverage": {
            "source_identity": coverage.source_identity,
            # coverage.problem_identity is deliberately NOT bound: it is the
            # digest of the very content being encoded, so embedding it would
            # make the recipe self-referential and no honest recompute could
            # reproduce it. A verdict naming another problem is refused by the
            # consumer that recomputes THIS digest (PROBLEM_CONTENT_CHANGED);
            # the sets below are what the verdict actually decides.
            # the problem anchor the verdict was computed for: recomputed here,
            # never read back from the verdict, so a verdict that names another
            # problem is bound (and refused) rather than trusted
            "expected_tasks": sorted(coverage.expected_tasks),
            "covered_tasks": sorted(coverage.covered_tasks),
            "expected_accesses": sorted(coverage.expected_accesses),
            "covered_accesses": sorted(coverage.covered_accesses),
            "expected_effects": sorted(coverage.expected_effects),
            "covered_effects": sorted(coverage.covered_effects),
            # a refusal is a verdict about coverage, so each one is bound as
            # (code, detail) rather than dropped: an appended refusal used to
            # change nothing, and a bound set of obligations with no record of
            # what failed is half a verdict
            "refusals": sorted([str(item.code), str(item.detail),
                                ref(getattr(item, "source_ref", None),
                                    "coverage.refusal.source_ref")]
                               for item in coverage.refusals),
        },
    }, sort_keys=True).encode("utf-8")


def problem_identity_of(problem) -> tuple:
    """(canonical bytes, digest) for one problem, from ONE recipe.

    There is deliberately a single place that hashes a problem. Two callers
    hashing with two different digest helpers is how an honest recompute ends
    up disagreeing with the value analyze recorded, so every consumer that
    must check a self-reported anchor calls this.
    """
    canonical = problem_canonical_bytes(problem)
    return canonical, _identity(canonical.decode())

def _carried_initials(units: UnitTree, binders: dict) -> dict:
    """root Var id -> the source loop's real carried initial value."""
    initials = {}
    for region in units.regions:
        summary = region.binding_summary
        if summary is None:
            continue
        for index, parameter in enumerate(summary.carried_params):
            actual = summary.binding_map[index + 1][1] if index + 1 < len(summary.binding_map) else None
            address = _address_of_actual(actual)
            if address is None:
                continue
            for root in binders:
                binder = binders[root]
                if _address_of_binder(binder) == address:
                    initials[root] = (summary.carried_initials[index],)
    return initials


def _storage_address(value) -> tuple | None:
    """(space, address, shape) of a real address recipe, or None when unnamed.

    The lowering states a non-GM window either as a TensorView Call in the TIR or
    as its own View record in region_bindings; both are address recipes over a
    pointer Constant, and that address is what two spellings of one buffer share.
    """
    target = getattr(value, "target", None)
    if isinstance(target, TensorView):
        base = value.args[0] if value.args else None
        tensor_type = value.type
    elif isinstance(value, TensorView):
        base = value.args[0] if getattr(value, "args", ()) else None
        tensor_type = value.type
    else:
        base = getattr(value, "base", None)
        tensor_type = getattr(value, "type", None)
    if base is None or not isinstance(base, Constant):
        return None
    if not isinstance(tensor_type, TensorType):
        return None
    return (_space_of(base.type), int(base.value), _static_shape(tensor_type))


def _address_of_actual(actual) -> tuple | None:
    return _storage_address(actual)


def _address_of_binder(binder) -> tuple | None:
    kind, node = binder
    if kind != "let":
        return None
    return _storage_address(getattr(node, "value", None))
