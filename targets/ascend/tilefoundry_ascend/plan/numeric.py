"""The numeric projection: one region problem as the plain numbers the solver decides on.

The native NormalizedProblem stays authoritative: it holds the object references, the
effects and the coverage verdict. This module only states the arithmetic the shared
constraint core consumes, so a plan is never built from a second description of the
program.
"""
from __future__ import annotations

from tilefoundry.vendor.tilelang_z3 import Buffer, Capacity, Edge, Problem, Reuse, Task
from tilefoundry.vendor.tilelang_z3.core import SolverError

from ..solver import ScheduleProblem

__all__ = ["to_core", "to_schedule_problem"]


def _owner_key(owner):
    if owner is None:
        return ("", "", "")
    return (str(getattr(owner, "program_group", "")),
            str(getattr(owner, "core_kind", "")),
            str(getattr(owner, "sid", "")))


def _storage_owner(buffer):
    key = getattr(buffer, "capacity_key", None)
    if key is not None:
        return _owner_key(getattr(key, "owner", None))
    return _owner_key(getattr(buffer, "owner", None))


def _domain(buffer):
    domain = getattr(buffer, "domain", None)
    fixed = getattr(domain, "value", None)
    if fixed is not None:
        return int(fixed), None, None
    low = getattr(domain, "minimum", None)
    high = getattr(domain, "maximum", None)
    return None, (None if low is None else int(low)), (None if high is None else int(high))


def to_core(region):
    """Project one RegionProblem onto the core data view."""
    tasks = tuple(
        Task(
            name=task.task_id,
            resources=tuple(
                (_owner_key(use.resource.owner), str(use.resource.pipe),
                 int(getattr(use.resource, "instance", 0)), int(use.duration), int(use.offset))
                for use in getattr(task, "uses", ()) or ()
            ),
            latency=int(task.latency),
            stage=(None if task.authored_stage is None else int(task.authored_stage)),
        )
        for task in region.tasks
    )
    # The number is the edge's sourced delay fact resolved through the delay
    # table. A projection never reads a raw attribute and never defaults an
    # unsourced edge to 0 or 1: resolved_delay() refuses by name instead.
    edges = tuple(Edge(item.producer, item.consumer, int(item.distance),
                       item.resolved_delay())
                  for item in region.edges)
    buffers = []
    for buffer in region.buffers:
        fixed, low, high = _domain(buffer)
        buffers.append(Buffer(
            name=buffer.buffer_id,
            slot_bytes=int(buffer.slot_bytes),
            space=str(buffer.space),
            owner=_storage_owner(buffer),
            fixed=fixed, auto_min=low, auto_max=high,
        ))
    accesses = {}
    for access in region.accesses:
        if access.access_id in accesses:
            raise SolverError("PROJECTION_ACCESS_DUPLICATE",
                              "duplicate access id %r" % (access.access_id,))
        accesses[access.access_id] = access
    task_ids = {task.task_id for task in region.tasks}

    buffers_by_id = {buffer.buffer_id: buffer for buffer in region.buffers}
    if len(buffers_by_id) != len(region.buffers):
        raise SolverError("PROJECTION_BUFFER_DUPLICATE",
                          "duplicate buffer id in the region")
    READ_MODES = ("READ", "READ_WRITE")
    WRITE_MODES = ("WRITE", "READ_WRITE")

    def reuse_task(access_id, role, buffer_id, modes):
        """Map one reuse endpoint to its task, proving the pairing is real.

        A reuse obligation says "this slot may be overwritten once the LAST
        READ of this buffer has landed". An endpoint is therefore only
        admissible when it is an access OF THAT BUFFER (same root) with the
        mode its role requires: a reader for last_reader, a writer for
        next_writer. Otherwise a caller could pair any two accesses of any
        two buffers and get a constraint over data that never existed.
        """
        if access_id not in accesses:
            raise SolverError("PROJECTION_REUSE_ACCESS_UNKNOWN",
                              "reuse %s %r is not a region access" % (role, access_id))
        access = accesses[access_id]
        task_id = access.task_id
        if task_id not in task_ids:
            raise SolverError("PROJECTION_REUSE_TASK_UNKNOWN",
                              "reuse endpoint %r names unknown task %r"
                              % (access_id, task_id))
        if buffer_id not in buffers_by_id:
            raise SolverError("PROJECTION_REUSE_BUFFER_UNKNOWN",
                              "reuse names buffer %r which the region does not have"
                              % (buffer_id,))
        if access.root_ref is not buffers_by_id[buffer_id].root_ref:
            raise SolverError(
                "PROJECTION_REUSE_UNPAIRED",
                "reuse %s %r touches root %r, not the root of buffer %r; a reuse "
                "obligation must be derived from a real read/write pair of the "
                "same buffer" % (role, access_id, getattr(access.root_ref, "name", None),
                                 buffer_id))
        if access.mode not in modes:
            raise SolverError("PROJECTION_REUSE_MODE",
                              "reuse %s %r has mode %r, which is not a %s of buffer %r"
                              % (role, access_id, access.mode, buffer_id,
                                 "reader" if role == "last_reader" else "writer"))
        return task_id

    reuses = []
    for item in region.reuse:
        reader_task = reuse_task(item.last_reader, "last_reader", item.buffer, READ_MODES)
        writer_task = reuse_task(item.next_writer, "next_writer", item.buffer, WRITE_MODES)
        if reader_task == writer_task:
            raise SolverError(
                "PROJECTION_REUSE_SAME_TASK",
                "reuse of buffer %r pairs two endpoints inside task %r; a slot is "
                "only reused by a LATER covering write, never by the task that "
                "produced the value" % (item.buffer, writer_task))
        reuses.append(Reuse(item.buffer, reader_task, writer_task,
                            int(item.distance_offset), item.resolved_delay()))
    reuses = tuple(reuses)
    capacities = tuple(
        Capacity(str(item.storage_key.space), _owner_key(item.storage_key.owner),
                 int(item.usable_bytes) - int(item.reserved_bytes))
        for item in region.capacities
    )
    manual = bool(getattr(region, "enable_offset", False)) or any(
        task.authored_stage is not None for task in region.tasks)
    return Problem(
        tasks=tasks, edges=edges, buffers=tuple(buffers), reuses=reuses,
        capacities=capacities, max_stage=int(region.max_stage), manual=manual,
        issue_after=tuple(region.issue_after),
    )


def to_schedule_problem(region):
    """The finite II domain the author declared, over the projected numbers."""
    return ScheduleProblem(
        core=to_core(region),
        fixed_ii=(None if region.fixed_ii is None else int(region.fixed_ii)),
        ii_min=int(region.ii_min),
        ii_max=int(region.ii_max),
    )
