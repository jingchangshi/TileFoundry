"""Scheduled HIR program model for Ascend950PR RMSNorm.

The program is a real structured HIR: buffers with owner loops and fixed
buffer versions, two SIMD VF execution regions (weight-convert VF outside the
pipeline, main compute VF inside it), and a declarative PipelinePolicy on the
row loop. Tasks and their affine access footprints are *derived* from these
declarations (not a hand-written 3-task graph).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass(frozen=True)
class CaseConfig:
    case_id: str
    batch: int
    d: int
    dtype: str
    eps: float
    core_count: int
    rows: int
    loop_mode: str
    requested_num_stages: int
    x_versions: int
    y_versions: int
    offset_policy: str
    l2_read: str
    l2_write: str
    target_profile: str = "ascend950pr_9579"

    @property
    def tiles(self) -> int:
        return self.batch // self.rows

    @property
    def nbase(self) -> int:
        return self.tiles // self.core_count

    @property
    def rem(self) -> int:
        return self.tiles % self.core_count

    @property
    def iters(self) -> int:
        return self.nbase + (1 if self.rem else 0)

    def loop_stop(self) -> str:
        if self.loop_mode == "dyn":
            return "max(cnt, 1)"
        return "iters"

    def as_dict(self):
        d = asdict(self)
        d.update(tiles=self.tiles, nbase=self.nbase, rem=self.rem, iters=self.iters)
        return d


@dataclass(frozen=True)
class BufferDecl:
    root: str
    space: str            # ascend.gm | ascend.ub
    bytes_: int
    alignment: int
    owner_loop: str       # "row_pipeline" | "outside" | "none"
    versions: int         # fixed physical version count (1 for non-ring)
    fixed: bool = True

    def as_dict(self):
        return {"root": self.root, "space": self.space, "bytes": self.bytes_,
                "alignment": self.alignment, "owner_loop": self.owner_loop,
                "versions": self.versions, "fixed": self.fixed}


@dataclass(frozen=True)
class VFRegion:
    kind: str             # "weight_convert" | "main_compute"
    inside_loop: bool
    instructions: tuple   # tuple of instruction contract names (emission order)

    def as_dict(self):
        return {"kind": self.kind, "inside_loop": self.inside_loop,
                "instructions": list(self.instructions)}


@dataclass(frozen=True)
class PipelinePolicy:
    requested_num_stages: int
    offset_policy: str
    owner_loop: str = "row_pipeline"


@dataclass
class ScheduledProgram:
    config: CaseConfig
    buffers: dict
    regions: tuple
    policy: PipelinePolicy

    def buffer_bytes_total(self) -> int:
        # per-AIV UB footprint (sum of ring versions of UB buffers + fixed)
        total = 0
        for b in self.buffers.values():
            if b.space == "ascend.ub":
                total += b.bytes_ * b.versions
        return total

    def as_dict(self):
        return {"config": self.config.as_dict(),
                "buffers": {k: v.as_dict() for k, v in self.buffers.items()},
                "regions": [r.as_dict() for r in self.regions],
                "policy": asdict(self.policy),
                "ub_footprint_bytes": self.buffer_bytes_total()}


MAIN_VF_INSTRUCTIONS = (
    # one representative emission slice of the main compute VF (per row):
    "pset", "vld", "vcvt", "vmula", "vmula", "vmula", "vmula", "vadd", "vadd",
    "vcadd", "vcadd", "vcadd", "vcadd", "vcadd", "vcadd", "vdupv",
    "vmuls", "vadds", "vsqrt", "vdiv", "vdup",
    "vld", "vcvt", "vmuls", "vmul", "vsts",
)
WEIGHT_VF_INSTRUCTIONS = ("pset", "vld", "vcvt", "vsts")


def build_program(config: CaseConfig) -> ScheduledProgram:
    CHB = 2  # bf16 bytes
    x_bytes = config.rows * config.d * CHB
    y_bytes = config.rows * config.d * CHB
    w_bf16 = config.d * CHB
    w_f32 = config.d * 4
    buffers = {
        "x_ub":  BufferDecl("x_ub", "ascend.ub", x_bytes, 32, "row_pipeline", config.x_versions, True),
        "y_ub":  BufferDecl("y_ub", "ascend.ub", y_bytes, 32, "row_pipeline", config.y_versions, True),
        "w_ub":  BufferDecl("w_ub", "ascend.ub", w_bf16, 32, "outside", 1, True),
        "wf":    BufferDecl("wf",   "ascend.ub", w_f32, 32, "outside", 1, True),
        "x_gm":  BufferDecl("x_gm", "ascend.gm", config.batch * config.d * CHB, 512, "none", 1, True),
        "y_gm":  BufferDecl("y_gm", "ascend.gm", config.batch * config.d * CHB, 512, "none", 1, True),
    }
    regions = (
        VFRegion("weight_convert", False, WEIGHT_VF_INSTRUCTIONS),
        VFRegion("main_compute", True, MAIN_VF_INSTRUCTIONS),
    )
    policy = PipelinePolicy(config.requested_num_stages, config.offset_policy, "row_pipeline")
    return ScheduledProgram(config, buffers, regions, policy)


def case_from_dict(d: dict) -> CaseConfig:
    return CaseConfig(
        case_id=d["case_id"], batch=d["shape"][0], d=d["shape"][1], dtype=d.get("dtype", "bfloat16"),
        eps=d.get("eps", 1e-6), core_count=d.get("core_count", 56), rows=d.get("rows", 4),
        loop_mode=d.get("loop_mode", "dyn"), requested_num_stages=d.get("requested_num_stages", 4),
        x_versions=d.get("x_versions", 4), y_versions=d.get("y_versions", 2),
        offset_policy=d.get("offset_policy", "source_default"),
        l2_read=d.get("l2_read", "NOTALLOC_CLEAN"), l2_write=d.get("l2_write", "NORMAL_FV"))


__all__ = ["CaseConfig", "BufferDecl", "VFRegion", "PipelinePolicy", "ScheduledProgram",
           "build_program", "case_from_dict", "MAIN_VF_INSTRUCTIONS", "WEIGHT_VF_INSTRUCTIONS"]
