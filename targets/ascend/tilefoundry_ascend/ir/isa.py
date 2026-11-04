"""Minimal Ascend950PR AIV Reg-SIMD instruction contract table.

Each contract states the operand schema, compute domain, required execution
region, core affinity, resource, completion rule and reference semantics. It is
a *contract*, not a performance claim; latency/II are optional cost-model inputs
and never stand in for a completion protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable

READ = "READ"; WRITE = "WRITE"; RW = "READ|WRITE"


@dataclass(frozen=True)
class Operand:
    name: str
    effect: str
    space_class: str           # "ub" | "vreg" | "preg" | "scalar" | "gm"


@dataclass(frozen=True)
class InstructionContract:
    name: str
    category: str              # dma | predicate | convert | reduce | elementwise | store
    operands: tuple
    compute_domain: str        # e.g. "64 x fp32 lanes"
    elem_width_bits: Optional[int]
    required_region: str       # "simdvf" | "outside" | "either"
    core_affinity: str         # "aiv" | "aic"
    resource: str              # "vector" | "mte2" | "mte3" | "scalar"
    completion: str
    result: str
    latency_model: Optional[int] = None
    ii_model: Optional[int] = None
    semantics: str = ""
    # P2: lane/predicate/element-width surface, explicit rather than implied.
    lane_count: int = 64
    # BF16 lanes are packed two per 32-bit carrier; 16 names the packed width, 32 f32.
    element_widths: tuple = (32,)
    predicate_rule: str = (
        "one predicate bit per lane; an inactive lane keeps the destination bits unchanged"
    )
    rounding: str = "per-element round-to-nearest-even to the element width (see bf16)"
    reduction_order: str = ""


def _c(**kw):
    return InstructionContract(**kw)


ISA = {
    "pset": _c(name="pset", category="predicate",
               operands=(), compute_domain="predicate bits over 64 lanes",
               elem_width_bits=32, required_region="simdvf", core_affinity="aiv",
               resource="vector", completion="issue", result="preg(64 bits)",
               semantics="pset(32) sets element width 32 (bits), NOT a lane count; PAT_ALL => all lanes active."),
    "vld": _c(name="vld", category="dma",
              operands=(Operand("src", READ, "ub"),), compute_domain="64 lanes",
              elem_width_bits=None, required_region="simdvf", core_affinity="aiv",
              resource="vector", completion="issue+data_ready", result="vreg",
              latency_model=4, ii_model=1,
              semantics="vld(ub, off, dist): NORM loads f32 lanes; UNPK_B16 widens bf16 to f32 lanes; BRC_* broadcasts."),
    "vcvt": _c(name="vcvt", category="convert",
               operands=(Operand("src", READ, "vreg"),), compute_domain="64 lanes",
               elem_width_bits=None, required_region="simdvf", core_affinity="aiv",
               resource="vector", completion="issue", result="vreg",
               latency_model=2, ii_model=1,
               semantics="vcvt(src, dtype, part=0): convert lanes to dtype; part selects register half."),
    "vmula": _c(name="vmula", category="reduce",
                operands=(Operand("acc", RW, "vreg"), Operand("a", READ, "vreg"), Operand("b", READ, "vreg")),
                compute_domain="64 x fp32 lanes", elem_width_bits=32,
                required_region="simdvf", core_affinity="aiv", resource="vector",
                completion="issue", result="vreg(acc)",
                latency_model=4, ii_model=1,
                semantics="acc = acc + a*b (fused multiply-add into acc); loop-carried.",
                element_widths=(16, 32)),
    "vadd": _c(name="vadd", category="elementwise",
               operands=(Operand("a", READ, "vreg"), Operand("b", READ, "vreg")),
               compute_domain="64 lanes", elem_width_bits=32, required_region="simdvf",
               core_affinity="aiv", resource="vector", completion="issue", result="vreg",
               latency_model=3, ii_model=1, semantics="dst = a + b",
               element_widths=(16, 32)),
    "vadds": _c(name="vadds", category="elementwise",
                operands=(Operand("a", READ, "vreg"), Operand("s", READ, "scalar")),
                compute_domain="64 lanes", elem_width_bits=32, required_region="simdvf",
                core_affinity="aiv", resource="vector", completion="issue", result="vreg",
                latency_model=3, ii_model=1, semantics="dst = a + scalar"),
    "vmuls": _c(name="vmuls", category="elementwise",
                operands=(Operand("a", READ, "vreg"), Operand("s", READ, "scalar")),
                compute_domain="64 lanes", elem_width_bits=32, required_region="simdvf",
                core_affinity="aiv", resource="vector", completion="issue", result="vreg",
                latency_model=3, ii_model=1, semantics="dst = a * scalar"),
    "vmul": _c(name="vmul", category="elementwise",
               operands=(Operand("a", READ, "vreg"), Operand("b", READ, "vreg")),
               compute_domain="64 lanes", elem_width_bits=32, required_region="simdvf",
               core_affinity="aiv", resource="vector", completion="issue", result="vreg",
               latency_model=3, ii_model=1, semantics="dst = a * b",
               element_widths=(16, 32)),
    "vcadd": _c(name="vcadd", category="reduce",
                operands=(Operand("src", READ, "vreg"),), compute_domain="64 lanes",
                elem_width_bits=32, required_region="simdvf", core_affinity="aiv",
                resource="vector", completion="issue", result="vreg",
                latency_model=3, ii_model=1,
                semantics="PAIRWISE add reduction: dst[i] = src[2i]+src[2i+1] in the low half. 6 calls reduce 64->1.",
                element_widths=(16, 32),
                reduction_order="pairwise tree: bf16(a+b) at each level; six calls take 64 lanes to 1"),
    "vdupv": _c(name="vdupv", category="elementwise",
                operands=(Operand("src", READ, "vreg"),), compute_domain="64 lanes",
                elem_width_bits=32, required_region="simdvf", core_affinity="aiv",
                resource="vector", completion="issue", result="vreg",
                latency_model=3, ii_model=1, semantics="broadcast lane pos (POS_LOWEST=0) to all lanes"),
    "vdup": _c(name="vdup", category="elementwise",
               operands=(Operand("s", READ, "scalar"),), compute_domain="64 lanes",
               elem_width_bits=32, required_region="simdvf", core_affinity="aiv",
               resource="vector", completion="issue", result="vreg",
               latency_model=2, ii_model=1, semantics="broadcast scalar to all lanes"),
    "vsqrt": _c(name="vsqrt", category="elementwise",
                operands=(Operand("a", READ, "vreg"),), compute_domain="64 lanes",
                elem_width_bits=32, required_region="simdvf", core_affinity="aiv",
                resource="vector", completion="issue", result="vreg",
                latency_model=8, ii_model=4, semantics="dst = sqrt(a); NOT replaced by approximate rsqrt"),
    "vdiv": _c(name="vdiv", category="elementwise",
               operands=(Operand("a", READ, "vreg"), Operand("b", READ, "vreg")),
               compute_domain="64 lanes", elem_width_bits=32, required_region="simdvf",
               core_affinity="aiv", resource="vector", completion="issue", result="vreg",
               latency_model=8, ii_model=4, semantics="dst = a / b"),
    "vsts": _c(name="vsts", category="store",
               operands=(Operand("dst", WRITE, "ub"), Operand("src", READ, "vreg")),
               compute_domain="64 lanes", elem_width_bits=32, required_region="simdvf",
               core_affinity="aiv", resource="vector", completion="issue+write_visible",
               result="ub window", latency_model=4, ii_model=1,
               semantics="vsts(ub, off, src, PK_B32): pack f32 lanes to bf16 into UB window [off, off+64)."),
    # DMA / outside-VF ops
    "gm_to_ub": _c(name="gm_to_ub", category="dma",
                   operands=(Operand("src", READ, "gm"), Operand("dst", WRITE, "ub")),
                   compute_domain="-", elem_width_bits=None, required_region="outside",
                   core_affinity="aiv", resource="mte2", completion="issue+visible", result="ub",
                   latency_model=64, ii_model=8, semantics="GM->UB tile load (MTE2) with L2 cache policy"),
    "ub_to_gm": _c(name="ub_to_gm", category="dma",
                   operands=(Operand("src", READ, "ub"), Operand("dst", WRITE, "gm")),
                   compute_domain="-", elem_width_bits=None, required_region="outside",
                   core_affinity="aiv", resource="mte3", completion="issue+visible", result="gm",
                   latency_model=64, ii_model=8, semantics="UB->GM tile store (MTE3) with L2 cache policy"),
}


def contract(name: str) -> InstructionContract:
    if name not in ISA:
        raise KeyError(f"no instruction contract {name!r}")
    return ISA[name]


__all__ = ["InstructionContract", "Operand", "ISA", "contract", "READ", "WRITE", "RW"]
