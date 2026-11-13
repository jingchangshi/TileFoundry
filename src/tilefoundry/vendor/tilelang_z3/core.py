"""Fixed-interval modulo scheduling: the single constraint path.

One core; the Tensor-Foundry and TileLang adapters only project into it. An authored
stage is a constant, an authored Fixed(V) is an equality, and the model value is
exported unchanged.

    t_u = II * stage_u + phase_u,  0 <= phase_u < II
    t_dst + distance * II >= t_src + latency
    t_next_writer + (V + delta) * II >= t_last_reader + read_latency
    delta_slot = ((t_v + offset_v) - (t_u + offset_u)) mod II
    q_u <= delta_slot AND q_v <= II - delta_slot
    sum(V_b * slot_bytes_b) <= usable_bytes   per (space, owner)

The resource condition is a conjunction: two cyclic intervals on one capacity-1
resource must not overlap, so each has to fit ahead of the other. A disjunction
would accept II=1 with two unit tasks, which is an overlapping schedule.
"""
from __future__ import annotations

from dataclasses import dataclass

SAT = "SAT"
UNSAT = "UNSAT"
UNKNOWN = "UNKNOWN"

__all__ = [
    "Assignment", "Buffer", "Capacity", "Edge", "Problem", "Resource", "Reuse",
    "SAT", "SolverError", "Task", "UNKNOWN", "UNSAT", "lower_bound_ii", "solve",
    "verify_assignment",
]


class SolverError(ValueError):
    """A malformed problem or assignment, with a stable named code."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__("%s: %s" % (code, detail))


@dataclass(frozen=True)
class Task:
    """One task with resource uses (owner, core, pipe, duration, offset).

    Four-element uses specify offset zero and are normalized on construction.
    """

    name: str
    resources: tuple = ()
    latency: int = 1
    stage: int | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise SolverError("TASK_NAME", "a task needs a non-empty name")
        if not isinstance(self.latency, int) or self.latency < 0:
            raise SolverError("TASK_LATENCY", "latency must be a non-negative int")
        if self.stage is not None and (not isinstance(self.stage, int) or self.stage < 0):
            raise SolverError("TASK_STAGE", "an authored stage must be a non-negative int")
        normalized = []
        for use in self.resources:
            if len(use) not in (4, 5):
                raise SolverError("TASK_RESOURCE", "a resource use is (owner, core, pipe, duration[, offset])")
            duration = use[3]
            offset = use[4] if len(use) == 5 else 0
            if type(duration) is not int or duration < 1:
                raise SolverError("TASK_QUANTITY", "duration must be a positive int")
            if type(offset) is not int or offset < 0:
                raise SolverError("TASK_RESOURCE_OFFSET", "offset must be a non-negative int")
            normalized.append((*use[:4], offset))
        object.__setattr__(self, "resources", tuple(normalized))


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    distance: int = 0
    latency: int = 1

    def __post_init__(self):
        if not isinstance(self.distance, int) or self.distance < 0:
            raise SolverError("EDGE_DISTANCE", "distance must be a non-negative int")
        if not isinstance(self.latency, int) or self.latency < 0:
            raise SolverError("EDGE_LATENCY", "latency must be a non-negative int")


@dataclass(frozen=True)
class Buffer:
    name: str
    slot_bytes: int
    space: str = ""
    owner: str = ""
    fixed: int | None = None
    auto_min: int | None = None
    auto_max: int | None = None

    def __post_init__(self):
        if not isinstance(self.slot_bytes, int) or self.slot_bytes < 0:
            raise SolverError("BUFFER_BYTES", "slot_bytes must be a non-negative int")
        if self.fixed is not None:
            if not isinstance(self.fixed, int) or self.fixed < 1:
                raise SolverError("BUFFER_FIXED", "a fixed version count must be a positive int")
            if self.auto_min is not None or self.auto_max is not None:
                raise SolverError("BUFFER_DOMAIN", "a buffer is Fixed or Auto, never both")
            return
        low = 1 if self.auto_min is None else self.auto_min
        high = low if self.auto_max is None else self.auto_max
        if not isinstance(low, int) or not isinstance(high, int) or low < 1 or high < low:
            raise SolverError("BUFFER_AUTO", "an auto domain needs 1 <= min <= max")


@dataclass(frozen=True)
class Reuse:
    buffer: str
    last_reader: str
    next_writer: str
    delta: int = 0
    read_latency: int = 1


@dataclass(frozen=True)
class Resource:
    owner: str
    core: str
    pipe: str
    capacity: int = 1

    def __post_init__(self):
        if self.capacity != 1:
            raise SolverError(
                "RESOURCE_CAPACITY_UNSUPPORTED",
                "this core models capacity 1 per (owner, core, pipe); a multi-slot "
                "resource must be registered as distinct provable resources",
            )


@dataclass(frozen=True)
class Capacity:
    space: str
    owner: str
    bytes: int


@dataclass(frozen=True)
class Problem:
    tasks: tuple
    edges: tuple = ()
    buffers: tuple = ()
    reuses: tuple = ()
    resources: tuple = ()
    capacities: tuple = ()
    max_stage: int = 0
    manual: bool = False
    issue_after: tuple = ()


@dataclass(frozen=True)
class Assignment:
    ii: int
    starts: dict
    stages: dict
    phases: dict
    versions: dict
    issue_order: tuple
    detail: str = ""


def _task_map(problem):
    by_name = {}
    for task in problem.tasks:
        if task.name in by_name:
            raise SolverError("TASK_DUPLICATE", "two tasks share the name %r" % (task.name,))
        by_name[task.name] = task
    for edge in problem.edges:
        for endpoint in (edge.src, edge.dst):
            if endpoint not in by_name:
                raise SolverError("EDGE_ENDPOINT", "edge names an unknown task %r" % (endpoint,))
    for reuse in problem.reuses:
        for endpoint in (reuse.last_reader, reuse.next_writer):
            if endpoint not in by_name:
                raise SolverError("REUSE_ENDPOINT", "reuse names an unknown task %r" % (endpoint,))
    for pair in problem.issue_after:
        for endpoint in pair:
            if endpoint not in by_name:
                raise SolverError("ISSUE_ENDPOINT", "issue_after names an unknown task %r" % (endpoint,))
    return by_name


def _buffer_map(problem):
    by_name = {buffer.name: buffer for buffer in problem.buffers}
    for reuse in problem.reuses:
        if reuse.buffer not in by_name:
            raise SolverError("REUSE_BUFFER", "reuse names an unknown buffer %r" % (reuse.buffer,))
    return by_name


def lower_bound_ii(problem):
    """A necessary lower bound on II: it narrows the search and never decides it."""
    bound = 1
    per_resource = {}
    for task in problem.tasks:
        for owner, core, pipe, quantity, offset in task.resources:
            key = (owner, core, pipe)
            per_resource[key] = per_resource.get(key, 0) + quantity
    if per_resource:
        bound = max(bound, max(per_resource.values()))
    for edge in problem.edges:
        if edge.src != edge.dst or edge.distance == 0:
            # A zero-distance self edge is t >= t + latency: trivially true for
            # latency 0 and infeasible otherwise (decided in solve, not by a bound).
            continue
        bound = max(bound, -(-edge.latency // edge.distance))
    return bound


def _issue_order(problem, phases, by_name):
    """Phase order first, then the same-phase linear extension.

    The physical emission order is the phase; only tasks sharing a phase are
    ordered by the author's issue edges and by same-pipe source order. Sorting
    the whole task list topologically would let a later-phase task claim an
    earlier issue slot.
    """
    edges = [(a, b) for a, b in problem.issue_after]
    groups = {}
    for task in problem.tasks:
        groups.setdefault(phases[task.name], []).append(task.name)
    for names in groups.values():
        by_pipe = {}
        for name in names:
            task = by_name[name]
            for owner, core, pipe, _quantity, _offset in task.resources:
                by_pipe.setdefault((owner, core, pipe), []).append(name)
        for names_in_pipe in by_pipe.values():
            for earlier, later in zip(names_in_pipe, names_in_pipe[1:]):
                edges.append((earlier, later))
    order = []
    for phase in sorted(groups):
        names = groups[phase]
        adjacency = {name: [] for name in names}
        indegree = {name: 0 for name in names}
        seen = set()
        for earlier, later in edges:
            if earlier == later or earlier not in indegree or later not in indegree:
                continue
            if (earlier, later) in seen:
                continue
            seen.add((earlier, later))
            adjacency[earlier].append(later)
            indegree[later] += 1
        ready = sorted(name for name, degree in indegree.items() if degree == 0)
        placed = 0
        while ready:
            name = ready.pop(0)
            order.append(name)
            placed += 1
            for later in sorted(adjacency[name]):
                indegree[later] -= 1
                if indegree[later] == 0:
                    ready.append(later)
            ready.sort()
        if placed != len(names):
            return None
    return tuple(order) if len(order) == len(problem.tasks) else None


def solve(problem, ii, timeout_ms=None):
    """Decide one fixed II. Returns (status, Assignment | None, detail)."""
    if not isinstance(ii, int) or ii < 1:
        raise SolverError("SOLVER_II", "II must be a positive int")
    by_name = _task_map(problem)
    _buffer_map(problem)
    for edge in problem.edges:
        if edge.src == edge.dst and edge.distance == 0 and edge.latency > 0:
            return UNSAT, None, "SELF_EDGE_INFEASIBLE: %r needs latency %d with no distance" % (
                edge.src, edge.latency)
    if not problem.manual:
        for task in problem.tasks:
            if task.stage not in (None, 0):
                return UNSAT, None, "NO_OFFSET_AUTHORED_STAGE: %r has stage %r" % (task.name, task.stage)
    if not problem.tasks:
        return SAT, Assignment(ii=ii, starts={}, stages={}, phases={}, versions={},
                               issue_order=(), detail="EMPTY_PROBLEM"), "EMPTY_PROBLEM"
    try:
        import z3
    except ImportError:
        return UNKNOWN, None, "Z3_UNAVAILABLE"
    solver = z3.Solver()
    if timeout_ms is not None:
        solver.set("timeout", max(1, int(timeout_ms)))
    stage, phase = {}, {}
    for task in problem.tasks:
        stage[task.name] = z3.Int("stage_%s" % task.name)
        phase[task.name] = z3.Int("phase_%s" % task.name)
        solver.add(phase[task.name] >= 0, phase[task.name] < ii)
        if task.stage is not None and task.stage > problem.max_stage:
            return UNSAT, None, "AUTHORED_STAGE_EXCEEDS_MAX: %r" % (task.name,)
        if not problem.manual:
            solver.add(stage[task.name] == 0)
        elif task.stage is not None:
            solver.add(stage[task.name] == task.stage)
        else:
            # A manual list pins unannotated siblings to stage 0: they are not
            # free to drift because some sibling declared a stage.
            solver.add(stage[task.name] == 0)

    def start(name):
        return ii * stage[name] + phase[name]

    for edge in problem.edges:
        solver.add(start(edge.dst) + edge.distance * ii >= start(edge.src) + edge.latency)
    version = {}
    for buffer in problem.buffers:
        version[buffer.name] = z3.Int("version_%s" % buffer.name)
        if buffer.fixed is not None:
            solver.add(version[buffer.name] == buffer.fixed)
        else:
            low = 1 if buffer.auto_min is None else buffer.auto_min
            high = low if buffer.auto_max is None else buffer.auto_max
            solver.add(version[buffer.name] >= low, version[buffer.name] <= high)
    for reuse in problem.reuses:
        offset = version[reuse.buffer] + reuse.delta
        solver.add(offset >= 1)
        solver.add(start(reuse.next_writer) + offset * ii >= start(reuse.last_reader) + reuse.read_latency)
    per_resource = {}
    for task in problem.tasks:
        for owner, core, pipe, quantity, offset in task.resources:
            if quantity > ii:
                return UNSAT, None, "TASK_EXCEEDS_INTERVAL: %r reserves %d > II=%d" % (
                    task.name, quantity, ii)
            per_resource.setdefault((owner, core, pipe), []).append((task.name, quantity, offset))
    for uses in per_resource.values():
        for index, (first_name, first_quantity, first_offset) in enumerate(uses):
            for second_name, second_quantity, second_offset in uses[index + 1:]:
                delta = ((start(second_name) + second_offset)
                         - (start(first_name) + first_offset)) % ii
                solver.add(first_quantity <= delta)
                solver.add(second_quantity <= ii - delta)
    for earlier, later in problem.issue_after:
        # Emission order is physical: within one cycle the earlier-issued task may
        # not sit at a later phase. Same-phase ties are settled by the linear
        # extension recorded in issue_order.
        solver.add(phase[earlier] <= phase[later])
    # Capacity constraints exist only for a space that publishes a usable size.
    # A space without a fact is unconstrained here and must stay UNRESOLVED in the
    # storage plan; treating it as infinite would be the backfill this core refuses.
    for capacity in problem.capacities:
        terms = [version[buffer.name] * buffer.slot_bytes for buffer in problem.buffers
                 if buffer.space == capacity.space and buffer.owner == capacity.owner]
        if terms:
            solver.add(sum(terms) <= capacity.bytes)
    result = solver.check()
    if result == z3.unsat:
        return UNSAT, None, "UNSAT_FOR_II"
    if result != z3.sat:
        return UNKNOWN, None, "SOLVER_UNKNOWN_OR_TIMEOUT"
    model = solver.model()
    stages = {task.name: model.eval(stage[task.name], model_completion=True).as_long()
              for task in problem.tasks}
    phases = {task.name: model.eval(phase[task.name], model_completion=True).as_long()
              for task in problem.tasks}
    versions = {buffer.name: model.eval(version[buffer.name], model_completion=True).as_long()
                for buffer in problem.buffers}
    starts = {name: ii * stages[name] + phases[name] for name in stages}
    order = _issue_order(problem, phases, by_name)
    if order is None:
        return UNSAT, None, "ISSUE_ORDER_CYCLE"
    return SAT, Assignment(ii=ii, starts=starts, stages=stages, phases=phases,
                           versions=versions, issue_order=order, detail="SAT"), "SAT"


def verify_assignment(problem, assignment):
    """Independent integer postcheck: recompute every constraint without z3."""
    by_name = _task_map(problem)
    _buffer_map(problem)
    problems = []
    ii = assignment.ii
    known_buffers = {buffer.name for buffer in problem.buffers}
    for label, mapping in (("STARTS", assignment.starts), ("STAGES", assignment.stages),
                           ("PHASES", assignment.phases)):
        for key in mapping:
            if key not in by_name:
                problems.append("FOREIGN_TASK_%s %r" % (label, key))
    for key in assignment.versions:
        if key not in known_buffers:
            problems.append("FOREIGN_BUFFER_VERSION %r" % (key,))
    for name, task in by_name.items():
        if name not in assignment.stages or name not in assignment.phases:
            problems.append("MISSING_TASK %r" % (name,))
            continue
        level = assignment.stages[name]
        phase = assignment.phases[name]
        if level < 0 or level > problem.max_stage:
            problems.append("STAGE_OUT_OF_RANGE %r=%d" % (name, level))
        if not 0 <= phase < ii:
            problems.append("PHASE_OUT_OF_RANGE %r=%d" % (name, phase))
        if assignment.starts.get(name) != ii * level + phase:
            problems.append("START_FORMULA %r" % (name,))
        if task.stage is not None and level != task.stage:
            problems.append("AUTHORED_STAGE_CHANGED %r=%d authored=%d" % (name, level, task.stage))
        if not problem.manual and level != 0:
            problems.append("NO_OFFSET_STAGE %r=%d" % (name, level))
        if problem.manual and task.stage is None and level != 0:
            problems.append("MANUAL_DEFAULT_STAGE %r=%d" % (name, level))
        for use in task.resources:
            if use[3] > ii:
                problems.append("TASK_EXCEEDS_INTERVAL %r reserves %d > II=%d" % (name, use[3], ii))
    for edge in problem.edges:
        if assignment.starts.get(edge.dst, 0) + edge.distance * ii < assignment.starts.get(edge.src, 0) + edge.latency:
            problems.append("EDGE_VIOLATED %s->%s" % (edge.src, edge.dst))
    for reuse in problem.reuses:
        versions = assignment.versions.get(reuse.buffer)
        if versions is None:
            problems.append("MISSING_BUFFER %r" % (reuse.buffer,))
            continue
        if versions + reuse.delta < 1:
            problems.append("VERSION_DELTA %r" % (reuse.buffer,))
        if (assignment.starts.get(reuse.next_writer, 0) + (versions + reuse.delta) * ii
                < assignment.starts.get(reuse.last_reader, 0) + reuse.read_latency):
            problems.append("REUSE_VIOLATED %r" % (reuse.buffer,))
    for buffer in problem.buffers:
        versions = assignment.versions.get(buffer.name)
        if versions is None:
            problems.append("MISSING_VERSIONS %r" % (buffer.name,))
            continue
        if buffer.fixed is not None and versions != buffer.fixed:
            problems.append("FIXED_VERSIONS_CHANGED %r=%d authored=%d" % (buffer.name, versions, buffer.fixed))
        low = 1 if buffer.auto_min is None else buffer.auto_min
        high = low if buffer.auto_max is None else buffer.auto_max
        if buffer.fixed is None and not low <= versions <= high:
            problems.append("AUTO_VERSIONS_OUT_OF_DOMAIN %r=%d" % (buffer.name, versions))
    per_resource = {}
    for task in problem.tasks:
        for owner, core, pipe, quantity, offset in task.resources:
            per_resource.setdefault((owner, core, pipe), []).append((task.name, quantity, offset))
    for uses in per_resource.values():
        for index, (first_name, first_quantity, first_offset) in enumerate(uses):
            for second_name, second_quantity, second_offset in uses[index + 1:]:
                if first_name not in assignment.starts or second_name not in assignment.starts:
                    continue
                delta = ((assignment.starts[second_name] + second_offset)
                         - (assignment.starts[first_name] + first_offset)) % ii
                if not (first_quantity <= delta and second_quantity <= ii - delta):
                    problems.append("RESOURCE_OVERLAP %s/%s" % (first_name, second_name))
    for capacity in problem.capacities:
        total = sum(assignment.versions.get(buffer.name, 0) * buffer.slot_bytes
                    for buffer in problem.buffers
                    if buffer.space == capacity.space and buffer.owner == capacity.owner)
        if total > capacity.bytes:
            problems.append("CAPACITY_EXCEEDED %s/%s" % (capacity.space, capacity.owner))
    order = tuple(assignment.issue_order)
    if sorted(order) != sorted(by_name):
        problems.append("ISSUE_ORDER_NOT_A_PERMUTATION")
    else:
        previous_phase = None
        for name in order:
            phase = assignment.phases.get(name, 0)
            if previous_phase is not None and phase < previous_phase:
                problems.append("ISSUE_ORDER_PHASE %r" % (name,))
            previous_phase = phase
        position = {name: index for index, name in enumerate(order)}
        for earlier, later in problem.issue_after:
            if assignment.phases.get(earlier, 0) > assignment.phases.get(later, 0):
                problems.append("ISSUE_ORDER_BEFORE %s->%s" % (earlier, later))
            if assignment.phases.get(earlier) == assignment.phases.get(later) and position[earlier] > position[later]:
                problems.append("ISSUE_ORDER_VIOLATED %s->%s" % (earlier, later))
    return problems


def core_source_sha256() -> str:
    """Content identity of this core, for provenance without a second digest."""
    import hashlib
    from pathlib import Path

    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _z3_available() -> bool:
    from importlib.util import find_spec

    return find_spec("z3") is not None


Z3_AVAILABLE = _z3_available()

