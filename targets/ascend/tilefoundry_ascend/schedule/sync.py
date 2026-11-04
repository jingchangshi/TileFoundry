"""Completion obligations -> logical channels -> physical Ascend flags.

ready/free obligations are derived from the Problem dependencies and the ring
buffer versions. A logical channel is NOT a physical flag id: physical flags are
allocated per channel from a bounded pool, and slot generation is the *logical
iteration ordinal* t (not the recompute tile index).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Obligation:
    channel: str
    producer: str          # task id
    producer_pipe: str
    consumer: str
    consumer_pipe: str
    event_domain: str      # "x_slot" | "y_slot" | "once"
    versions: int
    distance: int
    meaning: str

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SyncAction:
    channel: str
    logical_channel: str
    physical_flag: int
    producer_pipe: str
    consumer_pipe: str
    direction: str         # "set_wait"

    def as_dict(self):
        return asdict(self)


def build_obligations(problem) -> list:
    Vx = problem.policy["x_versions"]; Vy = problem.policy["y_versions"]
    return [
        Obligation("x_ready", "x_load", "mte2", "main_vf", "vector", "x_slot", Vx, 0,
                   "complete L(t) precedes read C(t)"),
        Obligation("x_free", "main_vf", "vector", "x_load", "mte2", "x_slot", Vx, Vx,
                   "last read C(t) precedes write L(t+Bx)"),
        Obligation("y_ready", "main_vf", "vector", "y_store", "mte3", "y_slot", Vy, 0,
                   "complete write C(t) precedes read S(t)"),
        Obligation("y_free", "y_store", "mte3", "main_vf", "vector", "y_slot", Vy, Vy,
                   "complete read S(t) precedes write C(t+By)"),
        Obligation("w_ready", "w_load", "mte2", "w_convert", "vector", "once", 1, 0,
                   "weight tile loaded before convert"),
        Obligation("w_conv", "w_convert", "vector", "main_vf", "vector", "once", 1, 0,
                   "converted weight live-in to compute VF"),
    ]


def allocate_flags(obligations) -> dict:
    """Assign a distinct physical flag id per logical channel (bounded pool)."""
    flags = {}
    nxt = 0
    for ob in obligations:
        if ob.channel not in flags:
            flags[ob.channel] = nxt
            nxt += 1
    return flags


def build_plan(problem, solve_result) -> dict:
    obls = build_obligations(problem)
    flags = allocate_flags(obls)
    actions = [SyncAction(ob.channel, ob.channel, flags[ob.channel], ob.producer_pipe,
                          ob.consumer_pipe, "set_wait") for ob in obls]
    Vx = problem.policy["x_versions"]; Vy = problem.policy["y_versions"]
    buffers = {}
    for b in problem.buffers:
        if b.root in ("x_ub", "y_ub"):
            V = b.versions
            buffers[b.root] = {"root": b.root, "space": b.space, "versions": V,
                               "owner_loop": b.owner_loop, "slot_rule": f"t % {V}",
                               "generation_rule": "t (logical iteration ordinal)",
                               "bytes": b.bytes_}
        else:
            buffers[b.root] = {"root": b.root, "space": b.space, "versions": 1,
                               "owner_loop": b.owner_loop, "slot_rule": "static",
                               "generation_rule": "-", "bytes": b.bytes_}
    return {
        "status": solve_result.status,
        "ii": solve_result.ii,
        "stages": dict(solve_result.stages),
        "model_version": solve_result.model_version,
        "solver_version": solve_result.solver_version,
        "seed": solve_result.seed,
        "buffers": buffers,
        "obligations": [o.as_dict() for o in obls],
        "sync_actions": [a.as_dict() for a in actions],
        "flag_pool": flags,
        "diagnostics": list(solve_result.diagnostics),
    }


__all__ = ["Obligation", "SyncAction", "build_obligations", "allocate_flags", "build_plan"]
