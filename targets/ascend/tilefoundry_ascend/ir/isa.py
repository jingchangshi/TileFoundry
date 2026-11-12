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
    # --- accumulator contract -------------------------------------------------
    # accumulator_count is deliberately OPTIONAL and defaults to None: None means this
    # instruction does NOT state a count, so no downstream claim may assume one.
    #
    # RETRACTED (kept for the record, NOT in force): this comment previously declared "the
    # vcadd entry states TWO as the CLAIM BASELINE only; PROVISIONAL Lead ruling; architect
    # confirmation pending". The Lead RETRACTED that statement. reference/kernel_v33.py
    # (SHA c6c37fb751d0afe439006a953d5518724166a7375bc5201bfdb9acb61b93d909, verified
    # against the P0 pin) uses FOUR chains (acc0_0..acc0_3, balanced merge ((0+1)+(2+3)))
    # on the rows=4 path and TWO (merge (0+1)) only in the rows<=2 fallback. The count is a
    # PER-CASE algorithm / scheduled-HIR contract, so NO ISA entry declares one.
    accumulator_count: Optional[int] = None
    accumulator_contract: str = ""
    reduction_depends_on: tuple = ()
    # --- native-evidence status ----------------------------------------------
    # What (if anything) the current BUILD proves about this instruction's native tree
    # shape / output layout. Empty means "not examined". "UNSOURCED" means a frozen
    # description exists but the current C++ contradicts it, so it is NOT evidence.
    native_evidence_status: str = ""
    native_evidence: str = ""
    baseline_sequence: str = ""


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
                reduction_order=(
                    "six vcadd calls reduce 64 lanes to one scalar (BASELINE SEQUENCE, kept);"
                    " the native reduction TREE and OUTPUT LAYOUT are UNSOURCED against the"
                    " current build, and the NUMBER OF CHAINS is PER CASE"
                    " (PER_CASE_ACCUMULATOR_CONTRACT), not an ISA constant. Accumulation is"
                    " FP32 after BF16 conversion; a per-level BF16 reducer is NOT v33"
                    " semantics"
                ),
                baseline_sequence=(
                    "six vcadd calls (64 lanes -> one scalar), kept as the baseline"
                    " sequence; the six calls do NOT prove the native tree shape"
                ),
                native_evidence_status=(
                    "UNSOURCED against the current build: tilelang 0.1.15+cuda.git8e2d3c82, "
                    "src/tl_templates/ascend/simd_inst.h sha256 "
                    "979f4ff76eee82bf092addcc069b238e3084af890f11660498f00640a00f20a1"
                ),
                native_evidence=(
                    "The frozen-518 Python vcadd docstring ('low-half pairwise / six "
                    "halvings / BF16 at each level') CONTRADICTS the C++ actually built: "
                    "simd_inst.h lines 688-703 vcadd calls asc_reduce_sum(dst, src, mask) "
                    "and merges with reduction_result_mask(PAT_VL1); vcpadd at lines 681-684 "
                    "calls asc_pair_reduce_sum. It is therefore NOT evidence of native tree "
                    "shape or output layout and must not be cited as proof."
                ),
                # RETRACTED, kept for the record and NOT in force.
                #
                # This entry once declared a global accumulator_count=2 as a "PROVISIONAL
                # Lead ruling". That was WRONG and the Lead retracted it. reference/
                # kernel_v33.py (SHA c6c37fb751d0afe439006a953d5518724166a7375bc5201bfdb9acb61b93d909,
                # verified against the P0 pin) declares FOUR acc0_0..acc0_3 chains per row
                # on the rows=4 path at lines 127-139 with a balanced merge ((0+1)+(2+3))
                # at line 140; TWO chains with merge (0+1) appear only in the rows<=2
                # fallback at lines 253-260. "Two is the baseline" mistook the fallback for
                # the main path. The count is therefore a PER-CASE algorithm / scheduled-HIR
                # numerical contract, not a universal primitive field, so NO accumulator_count
                # is declared here and nothing downstream can inherit "two" from the ISA.
                accumulator_count=None,
                accumulator_contract="",
                reduction_depends_on=()),
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


#: The exact build this file's native claims are judged against. Cited so a microtest can
#: verify the citation instead of trusting prose.
NATIVE_BUILD = {
    "tilelang_version": "0.1.15+cuda.git8e2d3c82",
    "git_head": "8e2d3c823b0c3a01d2ce68d47dabfe98e35cc22b",
    "simd_inst_path": "src/tl_templates/ascend/simd_inst.h",
    "simd_inst_sha256": "979f4ff76eee82bf092addcc069b238e3084af890f11660498f00640a00f20a1",
    "credit": "CPU-only software quantizer; the full P2 native-ISA claim is NOT established",
}


#: PER-CASE accumulator contract. This is an ALGORITHM / scheduled-HIR numerical
#: contract, NOT an ISA field: how many accumulator chains a case uses is a property of
#: the v33 program for that row configuration, not of a primitive instruction.
#:
#: Verified against reference/kernel_v33.py, SHA
#: c6c37fb751d0afe439006a953d5518724166a7375bc5201bfdb9acb61b93d909 (the P0 pin) on
#: 2026-11-04: rows=4 uses acc0_0..acc0_3 (four vmula chains) merged ((0+1)+(2+3)); the
#: rows<=2 fallback uses acc0/acc1 (two chains) merged (0+1).
#:
#: The RETRACTED draft stated "the claim baseline accumulator count = TWO" as a universal
#: ISA value. That is kept above only as a marked retraction, not as a contract.
PER_CASE_ACCUMULATOR_CONTRACT = {
    "rows=4": {
        "chains": 4,
        "merge": "((0+1)+(2+3))",
        "evidence": "reference/kernel_v33.py lines 127-140 (rows==4 path)",
    },
    "rows<=2": {
        "chains": 2,
        "merge": "(0+1)",
        "evidence": "reference/kernel_v33.py lines 253-260 (rows<=2 fallback)",
    },
}


#: --------------------------------------------------------- event-domain contracts
#:
#: The TF-side completion ops are declared in src/tilefoundry/ir/tir/flag.py:
#: FlagSet ("T.flag_set"), FlagWait ("T.flag_wait") and Drain ("T.drain"), whose
#: parameters are ALL attributes (channel_name/kind/slot/generation/versions/
#: flag_id/producer_pipe/consumer_pipe); Drain carries none. pipe_barrier and
#: threadfence have NO TF op in this tree - only the TileLang language surface
#: exposes them (ascend_pipe_barrier/ascend_threadfence), so their rows are
#: declared here as the contract the emitter must satisfy, and step 2 of the
#: event slice decides whether they are reachable at all.
#:
#: They carry NO data operands, so an event contract's resource is the EVENT
#: NAMESPACE, named by obligation kind exactly as the target declares its pools
#: (completion/model.py: EventPool(core_domain, kind, pipe_pair, banks)). The
#: HARD per-pipe-pair flags are declared separately in hard_event_pools with
#: bank_status=UNRESOLVED, so they are NOT used as a capacity here.
#:
#: latency_model/ii_model are deliberately None. The target declares no event
#: cost model; a None is a refusal in plan/delays.py, never a silent 0 or 1,
#: and none of these rows is a performance claim.
_EVENT_ROWS = (
    ("flag_set", "event_ready", "issue+state_complete",
     "publish one slot's data generation (READY) or one maximal last reader (FREE)"),
    ("flag_wait", "event_ready", "issue+data_ready",
     "consume one declared handoff before reading or before overwriting"),
    ("pipe_barrier", "event_state", "issue+state_complete",
     "order the rest of one pipe inside this instance (TF op MISSING)"),
    ("threadfence", "event_state", "issue+state_complete",
     "order writes for visibility across pipes (TF op MISSING)"),
    ("drain", "event_exit", "issue+state_complete",
     "region exit: nothing may remain outstanding past this point"),
)

for _name, _resource, _completion, _domain in _EVENT_ROWS:
    ISA[_name] = _c(
        name=_name, category="sync", operands=(),
        compute_domain=_domain, elem_width_bits=None, required_region="outside",
        core_affinity="aiv", resource=_resource, completion=_completion,
        result="event",
        semantics="completion signalling only; no data operand, no performance "
                  "figure; channel/slot/generation are attributes the completion "
                  "plan resolves from the same ResolvedPlan",
    )

#: The event-domain names, so a consumer can tell an event contract from a data
#: contract without re-reading the table.
EVENT_CONTRACT_NAMES = tuple(row[0] for row in _EVENT_ROWS)


def contract(name: str) -> InstructionContract:
    if name not in ISA:
        raise KeyError(f"no instruction contract {name!r}")
    return ISA[name]


def event_contract_of(op) -> InstructionContract:
    """The event-domain contract of a REAL sync Op instance or class.

    The data-side helper isa_contract_of() keys on the ascend_* instruction
    names; a sync op's registered schema name is flag_set/flag_wait/drain, so
    it would KeyError there. This is the matching accessor for the event rows:
    the schema name is looked up, and only a SYNC-category op is accepted - a
    data instruction with an event-shaped name is refused rather than merged.
    """
    schema = getattr(op, "_op_schema", None) or getattr(type(op), "_op_schema", None)
    name = getattr(schema, "name", "") if schema is not None else ""
    category = getattr(schema, "category", "") if schema is not None else ""
    if not name:
        raise KeyError("op carries no registered schema name, so no event contract "
                       "can be stated for it")
    if category != "sync":
        raise KeyError("instruction %r is category %r; only a sync op carries an "
                       "event contract" % (name, category))
    return contract(name)


def reduction_order_for_contract(instruction: InstructionContract) -> str:
    """The reduction order a contract states, refusing to omit a count it depends on."""
    if "accumulator_count" in instruction.reduction_depends_on:
        if instruction.accumulator_count is None:
            raise ValueError(
                f"{instruction.name}: reduction order depends on accumulator_count, which "
                "this instruction does not state"
            )
        return (
            f"{instruction.reduction_order} "
            f"(depends on accumulator_count={instruction.accumulator_count})"
        )
    return instruction.reduction_order


def reduction_order_of(name: str) -> str:
    """The reduction order *name* states, with the count it depends on made explicit."""
    return reduction_order_for_contract(contract(name))


__all__ = [
    "InstructionContract", "Operand", "ISA", "contract", "READ", "WRITE", "RW",
    "reduction_order_for_contract", "reduction_order_of", "PER_CASE_ACCUMULATOR_CONTRACT",
    "NATIVE_BUILD",
]
