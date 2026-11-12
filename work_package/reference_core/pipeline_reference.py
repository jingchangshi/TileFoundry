"""Small executable reference for ResolveSchedule and symbolic loop lowering.

This is not TF IR, a solver, a source translator, or a production driver. Port
the validation/mapping into TF's existing passes and typed expressions. Only
trace() expands runtime instances, deliberately for tiny reference tests.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Symbol:
    name: str

    def __post_init__(self):
        if not self.name.isidentifier():
            raise ValueError("symbol must be an identifier")


def bind(value, parameters):
    actual = parameters[value.name] if isinstance(value, Symbol) else value
    if type(actual) is not int:
        raise ValueError("loop bounds must bind to integers")
    return actual


@dataclass(frozen=True)
class Loop:
    start: int | Symbol
    stop: int | Symbol
    step: int = 1

    def __post_init__(self):
        if type(self.step) is not int or self.step <= 0:
            raise ValueError("reference supports positive static steps")

    def trip_count(self, parameters):
        extent = bind(self.stop, parameters) - bind(self.start, parameters)
        return max(0, (extent + self.step - 1) // self.step)


@dataclass(frozen=True)
class Task:
    name: str
    pipe: str
    start: int
    buffers: tuple[str, ...] = ()
    authored_stage: int | None = None


@dataclass(frozen=True)
class ResolvedTask:
    task: Task
    stage: int
    phase: int
    source_order: int


@dataclass(frozen=True)
class Schedule:
    ii: int
    tasks: tuple[ResolvedTask, ...]  # Physical-step issue order.

    @classmethod
    def resolve(cls, tasks, ii, *, enable_offset=True):
        tasks = tuple(tasks)
        if type(ii) is not int or ii <= 0 or not tasks:
            raise ValueError("positive II and nonempty task list required")
        if len({t.name for t in tasks}) != len(tasks):
            raise ValueError("duplicate task names")
        manual, resolved = any(t.authored_stage is not None for t in tasks), []
        for index, task in enumerate(tasks):
            if type(task.start) is not int or task.start < 0:
                raise ValueError("start must be a nonnegative integer")
            stage, phase = divmod(task.start, ii)
            if task.authored_stage is not None and (type(task.authored_stage) is not int or task.authored_stage < 0):
                raise ValueError("authored stage must be a nonnegative integer")
            if manual and stage != (task.authored_stage if task.authored_stage is not None else 0):
                raise ValueError(f"fixed/default-zero stage violated: {task.name}")
            if not enable_offset and stage != 0:
                raise ValueError("offset=False requires all starts within one II")
            resolved.append(ResolvedTask(task, stage, phase, index))
        if manual:
            for pipe in {t.pipe for t in tasks}:
                phases = [r.phase for r in resolved if r.task.pipe == pipe]
                if phases != sorted(phases):
                    raise ValueError(f"manual same-pipe source order violated: {pipe}")
        ordered = []
        for phase in sorted({r.phase for r in resolved}):
            pending = [r for r in resolved if r.phase == phase]
            while pending:
                # Same-phase manual order is a hard edge. Stage priority only
                # breaks ties between tasks whose predecessors already issued.
                ready = [r for r in pending if not manual or not any(
                    q.task.pipe == r.task.pipe and q.source_order < r.source_order
                    for q in pending)]
                chosen = min(ready, key=lambda r: (-r.stage, r.source_order))
                ordered.append(chosen)
                pending.remove(chosen)
        return cls(ii, tuple(ordered))


@dataclass(frozen=True)
class Instance:
    task: str
    physical_step: int
    ordinal: int
    source_index: int
    slots: tuple[tuple[str, int, int], ...]  # buffer, slot, complete generation


@dataclass(frozen=True)
class PipelineTemplate:
    loop: Loop
    schedule: Schedule
    versions: tuple[tuple[str, int], ...]

    def __post_init__(self):
        counts = dict(self.versions)
        if len(counts) != len(self.versions) or any(type(v) is not int or v <= 0 for v in counts.values()):
            raise ValueError("unique positive buffer version counts required")
        for r in self.schedule.tasks:
            if any(b not in counts for b in r.task.buffers):
                raise ValueError(f"missing resolved buffer versions: {r.task.name}")

    @property
    def shifts(self):
        minimum = min(r.stage for r in self.schedule.tasks)
        return tuple((r.task.name, r.stage - minimum) for r in self.schedule.tasks)

    def render(self):
        """Diagnostic pseudocode, not fake executable TileLang/TF syntax."""
        name = lambda b: b.name if isinstance(b, Symbol) else str(b)
        a, b, h = name(self.loop.start), name(self.loop.stop), self.loop.step
        shifts, versions = dict(self.shifts), dict(self.versions)
        span = max(shifts.values())
        lines = [f"N = max(0, ceil_div({b} - ({a}), {h}))",
                 f"if N > 0: for p in range(N + {span}):"]
        for r in self.schedule.tasks:
            k = f"p - {shifts[r.task.name]}"
            slots = ", ".join(f"{v}[({k}) % {versions[v]}]" for v in r.task.buffers)
            lines.append(f"  if 0 <= ({k}) < N: {r.task.name}(i={a}+({k})*{h}; {slots})")
        return "\n".join(lines)

    def trace(self, parameters=None):
        """Enumerate a tiny concrete execution for checking, not for compiling."""
        parameters = {} if parameters is None else parameters
        count, start = self.loop.trip_count(parameters), bind(self.loop.start, parameters)
        shifts, versions = dict(self.shifts), dict(self.versions)
        for p in range(count + max(shifts.values()) if count else 0):
            for r in self.schedule.tasks:
                k = p - shifts[r.task.name]
                if 0 <= k < count:
                    yield Instance(r.task.name, p, k, start + k * self.loop.step,
                                   tuple((b, k % versions[b], k) for b in r.task.buffers))
