"""Extract a ScheduleProblem from the Scheduled HIR program.

Tasks and affine access footprints are derived from the program's buffers and
VF regions (the emission slice), not hard-coded as a 3-task graph. Dependency
distances follow the ring-buffer semantics (X free at +Bx, Y free at +By), the
weight live-in, and the recompute WAW on the output window.
"""
from __future__ import annotations

import hashlib, json
from dataclasses import dataclass, field, asdict
from typing import Optional

from .program import ScheduledProgram, CaseConfig
from ..ir.isa import ISA


@dataclass(frozen=True)
class Access:
    task: str
    root: str
    space: str
    mode: str            # read | write
    byte_footprint: int
    logical_iteration: str   # "t" | "outside"


@dataclass(frozen=True)
class Dependency:
    producer: str
    consumer: str
    distance: int
    kind: str            # RAW | WAR | WAW | reuse
    evidence: str


@dataclass(frozen=True)
class Task:
    stable_id: str
    pipe: str            # mte2 | vector | mte3
    resource: str
    latency: int
    issue_interval: int
    region: str
    loop: str            # "row_pipeline" | "outside"

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ProblemBuffer:
    root: str
    space: str
    bytes_: int
    alignment: int
    owner_loop: str
    versions: int
    fixed: bool

    def as_dict(self):
        return asdict(self)


@dataclass
class ScheduleProblem:
    problem_id: str
    target_profile: str
    source_digest: str
    loop_domain: dict
    tasks: list
    accesses: list
    dependencies: list
    buffers: list
    policy: dict

    def as_dict(self):
        return {
            "problem_id": self.problem_id, "target_profile": self.target_profile,
            "source_digest": self.source_digest, "loop_domain": self.loop_domain,
            "tasks": [t.as_dict() for t in self.tasks],
            "accesses": [asdict(a) for a in self.accesses],
            "dependencies": [asdict(d) for d in self.dependencies],
            "buffers": [b.as_dict() for b in self.buffers],
            "policy": self.policy,
        }


def _latency_vector(instructions) -> int:
    return sum(ISA[i].latency_model or 1 for i in instructions)


def _ii_resource(instructions, resource) -> int:
    return sum(ISA[i].ii_model or 1 for i in instructions if ISA[i].resource == resource)


def extract_problem(program: ScheduledProgram) -> ScheduleProblem:
    c = program.config
    # tasks derived from regions + loop structure
    tasks = []
    tasks.append(Task("w_load", "mte2", "mte2", ISA["gm_to_ub"].latency_model, ISA["gm_to_ub"].ii_model,
                      "weight_convert", "outside"))
    tasks.append(Task("w_convert", "vector", "vector",
                      _latency_vector(program.regions[0].instructions),
                      max(1, _ii_resource(program.regions[0].instructions, "vector")),
                      "weight_convert", "outside"))
    tasks.append(Task("x_load", "mte2", "mte2", ISA["gm_to_ub"].latency_model, ISA["gm_to_ub"].ii_model,
                      "main_compute", "row_pipeline"))
    tasks.append(Task("main_vf", "vector", "vector",
                      _latency_vector(program.regions[1].instructions),
                      max(1, _ii_resource(program.regions[1].instructions, "vector")),
                      "main_compute", "row_pipeline"))
    tasks.append(Task("y_store", "mte3", "mte3", ISA["ub_to_gm"].latency_model, ISA["ub_to_gm"].ii_model,
                      "main_compute", "row_pipeline"))

    xb = c.rows * c.d * 2
    accesses = [
        Access("w_load", "w_gm", "ascend.gm", "read", c.d * 2, "outside"),
        Access("w_load", "w_ub", "ascend.ub", "write", c.d * 2, "outside"),
        Access("w_convert", "w_ub", "ascend.ub", "read", c.d * 2, "outside"),
        Access("w_convert", "wf", "ascend.ub", "write", c.d * 4, "outside"),
        Access("x_load", "x_gm", "ascend.gm", "read", xb, "t"),
        Access("x_load", "x_ub", "ascend.ub", "write", xb, "t"),
        Access("main_vf", "x_ub", "ascend.ub", "read", xb, "t"),
        Access("main_vf", "wf", "ascend.ub", "read", c.d * 4, "t"),
        Access("main_vf", "y_ub", "ascend.ub", "write", xb, "t"),
        Access("y_store", "y_ub", "ascend.ub", "read", xb, "t"),
        Access("y_store", "y_gm", "ascend.gm", "write", xb, "t"),
    ]

    deps = [
        Dependency("w_load", "w_convert", 0, "RAW", "weight live-in load then convert"),
        Dependency("w_convert", "main_vf", 0, "RAW", "converted weights live-in to compute"),
        Dependency("x_load", "main_vf", 0, "RAW", "X ready: complete load before VF reads"),
        Dependency("main_vf", "y_store", 0, "RAW", "Y ready: VF write before store reads"),
        Dependency("main_vf", "x_load", c.x_versions, "WAR", "X free: last VF read before slot rewritten at t+Bx"),
        Dependency("y_store", "main_vf", c.y_versions, "WAR", "Y free: store read before next VF write at t+By"),
        Dependency("y_store", "y_store", 1, "WAW", "recompute rewrites the same GM window (idempotent, must not be dropped)"),
    ]

    buffers = [ProblemBuffer(b.root, b.space, b.bytes_, b.alignment, b.owner_loop, b.versions, b.fixed)
               for b in program.buffers.values()]

    # effective offset: source_default -> backend default (False); explicit value preserved
    eff_offset = {"source_default": False, "default": False}.get(program.policy.offset_policy, program.policy.offset_policy)
    policy = {"requested_num_stages": program.policy.requested_num_stages,
              "offset_policy": program.policy.offset_policy,
              "effective_offset": eff_offset,
              "x_versions": c.x_versions, "y_versions": c.y_versions,
              "loop_mode": c.loop_mode}

    loop_domain = {"start": "core*nbase+min(core,rem)", "count_expr": c.loop_stop(),
                   "count_dyn": "max(cnt,1)", "count_recompute": c.iters,
                   "nbase": c.nbase, "rem": c.rem, "tiles": c.tiles,
                   "induction_id": "step", "guards": []}

    digest = hashlib.sha256(json.dumps(program.as_dict(), sort_keys=True).encode()).hexdigest()[:16]
    pid = f"{c.case_id}:{digest}"
    return ScheduleProblem(pid, c.target_profile, digest, loop_domain, tasks, accesses, deps, buffers, policy)


__all__ = ["Access", "Dependency", "Task", "ProblemBuffer", "ScheduleProblem", "extract_problem"]
