"""Memory-space identities, transfer legality and alias rules for Ascend 950PR.

Storage identity is *not* just an offset. A UB value owned by AIV core 3 and a
UB value owned by AIV core 7 at the same offset are different storage. GM is
shared across cores. VREG/PREG are register classes that never escape a
SIMD VF execution region. L2 is an implicit cache and cannot be allocated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Categories
GLOBAL = "global"
LOCAL = "local"
REGISTER = "register"
AUX = "auxiliary"
IMPLICIT_CACHE = "implicit_cache"
UNMATERIALIZED = "unmaterialized"

# Provenance for every declared target fact. DEVICE_VALIDATED requires real 950PR
# compile/run evidence; a local nominal record is REFERENCE_IMPLEMENTATION only.
STATUS_REFERENCE_IMPLEMENTATION = "REFERENCE_IMPLEMENTATION"
STATUS_DEVICE_VALIDATED = "DEVICE_VALIDATED"
STATUS_UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class MemorySpaceFacts:
    name: str
    category: str
    owner: str                    # "device" | "aiv" | "aic" | "vf"
    nominal_bytes: Optional[int]
    usable_bytes: Optional[int]
    alignment: Optional[int]
    reachable_domains: tuple      # execution domains that may address it
    register_encoding: Optional[str] = None
    source: str = ""
    allocatable: bool = True
    #: Provenance of the declared nominal/alignment values.
    status: str = STATUS_REFERENCE_IMPLEMENTATION
    #: Provenance of the usable capacity; None usable is UNRESOLVED, never a measurement.
    usable_status: str = STATUS_UNRESOLVED

    def as_dict(self) -> dict:
        return {
            "name": self.name, "category": self.category, "owner": self.owner,
            "nominal_bytes": self.nominal_bytes, "usable_bytes": self.usable_bytes,
            "alignment": self.alignment, "reachable_domains": list(self.reachable_domains),
            "register_encoding": self.register_encoding, "source": self.source,
            "allocatable": self.allocatable, "status": self.status,
            "usable_status": self.usable_status,
        }


@dataclass(frozen=True)
class MemorySpaceRef:
    """A typed reference to a memory space, owned by an execution instance."""
    space: str
    owner_instance: Optional[str] = None   # e.g. "aiv:3"; None for device-global

    def alias_key(self) -> tuple:
        """Two refs alias only if their full identity (space, owner) matches.

        GM is device-global so its owner_instance is normalised to None; all
        local/register spaces are per-instance.
        """
        if self.space in ("ascend.gm", "ascend.l2"):
            return (self.space, None)
        return (self.space, self.owner_instance)

    def __str__(self) -> str:
        return self.space if self.owner_instance is None else f"{self.space}@{self.owner_instance}"


KB = 1024
SPACE_FACTS = {
    "ascend.gm": MemorySpaceFacts("ascend.gm", GLOBAL, "device", None, None, 512,
                                  ("aiv", "aic", "device"), source="device HBM"),
    "ascend.l2": MemorySpaceFacts("ascend.l2", IMPLICIT_CACHE, "device", None, None, 128,
                                  ("aiv", "aic"), source="implicit cache; not allocatable",
                                  allocatable=False),
    "ascend.ub": MemorySpaceFacts("ascend.ub", LOCAL, "aiv", 256 * KB, None, 32,
                                  ("aiv",), source="3510 nominal 256KiB"),
    "ascend.vreg": MemorySpaceFacts("ascend.vreg", REGISTER, "vf", 256, None, 256,
                                    ("aiv",), register_encoding="vreg256B", source="256B/vector register"),
    "ascend.preg": MemorySpaceFacts("ascend.preg", REGISTER, "vf", None, None, 32,
                                    ("aiv",), register_encoding="predicate_bits", source="predicate register"),
    "ascend.scalar": MemorySpaceFacts("ascend.scalar", REGISTER, "aiv", None, None, 4,
                                      ("aiv",), register_encoding="scalar", source="scalar regs"),
    "ascend.l1": MemorySpaceFacts("ascend.l1", LOCAL, "aic", 512 * KB, None, 512,
                                  ("aic",), source="3510 nominal 512KiB"),
    "ascend.l0a": MemorySpaceFacts("ascend.l0a", LOCAL, "aic", 64 * KB, None, 512,
                                   ("aic",), source="3510 nominal 64KiB, 512B align"),
    "ascend.l0b": MemorySpaceFacts("ascend.l0b", LOCAL, "aic", 64 * KB, None, 512,
                                   ("aic",), source="3510 nominal 64KiB, 512B align"),
    "ascend.l0c": MemorySpaceFacts("ascend.l0c", AUX, "aic", 256 * KB, None, 64,
                                   ("aic",), source="3510 nominal 256KiB, 64B align"),
    # SS (shared scalar storage) is NOT in the TF local3510 record and deployed
    # TileLang 0e687c17 has no alloc_ss; capacity/visibility/instructions are UNRESOLVED.
    # It is registered as non-allocatable so no candidate can silently claim it.
    "ascend.ss": MemorySpaceFacts("ascend.ss", LOCAL, "device", None, None, None,
                                  ("aiv", "aic"),
                                  source="SS: not in TF local3510 record; no alloc_ss in TL "
                                         "0e687c17; capacity/visibility/instructions UNRESOLVED",
                                  allocatable=False, status=STATUS_UNRESOLVED,
                                  usable_status=STATUS_UNRESOLVED),
    "ascend.bias": MemorySpaceFacts("ascend.bias", AUX, "aic", None, None, 64,
                                    ("aic",), source="auxiliary fixpipe buffer; registered only"),
}


def space_facts_table() -> dict:
    return dict(SPACE_FACTS)


# --------------------------------------------------------------------- registers
# Register spaces are counted, not sized. A vector register is 256 B WIDE, but how
# many registers an AIV has is a separate fact that this build does NOT state, and
# the two must never be conflated: dividing a nominal byte figure by a register
# width would invent a count, and that is exactly the "requirement backfilled as a
# capacity" the storage gate exists to refuse.
#
# Not to be confused with MemorySpaceFacts.nominal_bytes above, which is a per-value
# WIDTH (256 B for one vreg); a budget needs a COUNT of registers per execution
# domain, which is a different fact with a different provenance.
STATUS_REGISTER_UNSOURCED = "UNSOURCED"


@dataclass(frozen=True)
class RegisterBudgetFacts:
    """How many registers one execution domain can use, and where that came from.

    value/source/status are deliberately separate. status UNSOURCED with value None
    means "this build does not state a budget"; a caller must treat that as a
    refusal, never as "unlimited". A number without a source is not a fact either,
    so status must be DEVICE_VALIDATED or REFERENCE_IMPLEMENTATION for the value to
    count, and the reason string names what is missing either way.
    """

    domain: str
    register_class: str
    value: Optional[int]
    source: str = ""
    status: str = STATUS_REGISTER_UNSOURCED
    reason: str = ""

    def is_sourced(self) -> bool:
        """A budget counts only when a positive count AND its provenance exist."""
        return (isinstance(self.value, int) and not isinstance(self.value, bool)
                and self.value > 0
                and self.status in (STATUS_DEVICE_VALIDATED,
                                    STATUS_REFERENCE_IMPLEMENTATION)
                and bool(self.source))

    def as_dict(self) -> dict:
        return {"domain": self.domain, "register_class": self.register_class,
                "value": self.value, "source": self.source,
                "status": self.status, "reason": self.reason}


#: Declared register budgets, keyed by (domain, register_class).
#:
#: THIS IS A FACT-REGISTRATION SLOT, NOT AN UNFINISHED TODO. Every entry is UNSOURCED
#: because the current build has NO Ascend source for a per-domain register COUNT:
#:   - the target states a vreg WIDTH (256 B per register), which is a different
#:     fact and must never be divided into a count;
#:   - the deployed TileLang build resource capture is HIP-only
#:     (tilelang/jit/kernel.py:264, capture_hip_resource_usage = is_hip_target), so
#:     n_regs/n_spills are not merely absent on Ascend - they are not provided.
#: Writing a plausible number here would let the storage gate PASS on a number
#: nobody sourced, which is worse than refusing. The register-budget gate therefore
#: stays closed by FACT, and a positive register example is explicitly out of scope
#: until a device or documentation source exists.
#:
#: The lowering-side report (storage.RegisterOccupancy) IS measurable and is kept: it
#: is the half that can be filled in today. One side measurable plus one side
#: UNSOURCED still refuses - that is the gate working, not a gap in the gate.
REGISTER_BUDGETS = {
    ("aiv", "vreg"): RegisterBudgetFacts(
        "aiv", "vreg", None,
        reason="the target states a 256 B register WIDTH, not how many vector "
               "registers one AIV has; a width is not a budget"),
    ("aiv", "preg"): RegisterBudgetFacts(
        "aiv", "preg", None,
        reason="no predicate-register count is stated by this build"),
    ("aiv", "carrier"): RegisterBudgetFacts(
        "aiv", "carrier", None,
        reason="carrier is a bf16 transport register class whose aggregate count is "
               "UNSOURCED, exactly as for vreg"),
    ("aic", "vreg"): RegisterBudgetFacts(
        "aic", "vreg", None,
        reason="no AIC register count is stated by this build"),
}


def register_budget(domain: str, register_class: str) -> RegisterBudgetFacts:
    """The declared budget for one (domain, register class), never a default.

    An unknown pair returns an UNSOURCED record rather than raising, so a caller
    reports a named gap with a reason instead of crashing on a lookup.
    """
    found = REGISTER_BUDGETS.get((str(domain), str(register_class)))
    if found is not None:
        return found
    return RegisterBudgetFacts(
        str(domain), str(register_class), None,
        reason="this build declares no register budget for domain %r class %r"
               % (domain, register_class))


def register_budgets_table() -> dict:
    return dict(REGISTER_BUDGETS)



# ---- Transfer legality (3510 data paths; S17-S19) ----
# (src, dst) -> (issuer_pipe, legal, note)
TRANSFERS = {
    ("ascend.gm", "ascend.ub"):  ("aiv.mte2", True,  "GM->UB AIV load"),
    ("ascend.ub", "ascend.gm"):  ("aiv.mte3", True,  "UB->GM AIV store"),
    ("ascend.ub", "ascend.vreg"): ("aiv.vector", True, "vld UB->VREG"),
    ("ascend.vreg", "ascend.ub"): ("aiv.vector", True, "vsts VREG->UB"),
    ("ascend.gm", "ascend.l1"):  ("aic.mte2", True,  "GM->L1 AIC load"),
    ("ascend.l1", "ascend.l0a"): ("aic.mte1", True,  "L1->L0A"),
    ("ascend.l1", "ascend.l0b"): ("aic.mte1", True,  "L1->L0B"),
    ("ascend.l0c", "ascend.ub"): ("fix", True,       "L0C->UB via FIX"),
    ("ascend.l0c", "ascend.l1"): ("fix", True,       "L0C->L1 via FIX"),
    ("ascend.l0c", "ascend.gm"): ("fix", True,       "L0C->GM via FIX"),
    ("ascend.l1", "ascend.ub"):  ("aic.mte3", True,  "L1->UB (AIC/AIV copy path)"),
    # 3510 new path: UB->L1 is a software per-AIV copy followed by InsertNd2Nz
    # (REFERENCE_IMPLEMENTATION; no device validation in this round).
    ("ascend.ub", "ascend.l1"):  ("aiv.mte3", True,  "UB->L1 software per-AIV copy + InsertNd2Nz"),
    # explicitly illegal direct paths (3510 has no such edges)
    ("ascend.gm", "ascend.l0a"): ("-", False, "no GM->L0A direct path on 3510"),
    ("ascend.gm", "ascend.l0b"): ("-", False, "no GM->L0B direct path on 3510"),
    ("ascend.l1", "ascend.gm"):  ("-", False, "no L1->GM direct path on 3510"),
}


@dataclass(frozen=True)
class TransferContract:
    src: str
    dst: str
    issuer: str
    resource: str
    legal: bool
    note: str = ""
    status: str = STATUS_REFERENCE_IMPLEMENTATION

    def as_dict(self) -> dict:
        return {"src": self.src, "dst": self.dst, "issuer": self.issuer,
                "resource": self.resource, "legal": self.legal, "note": self.note,
                "status": self.status}


def transfer_contract(src: str, dst: str) -> TransferContract:
    if src == "ascend.l2" or dst == "ascend.l2":
        return TransferContract(src, dst, "-", "-", False, "L2 is an implicit cache, not allocatable")
    entry = TRANSFERS.get((src, dst))
    if entry is None:
        return TransferContract(src, dst, "-", "-", False, "no declared data path")
    pipe, legal, note = entry
    return TransferContract(src, dst, pipe, pipe, legal, note)


def transfer_table() -> dict:
    """Every declared transfer edge as a TransferContract (REFERENCE_IMPLEMENTATION)."""
    return {pair: TransferContract(pair[0], pair[1], entry[0], entry[0], entry[1], entry[2])
            for pair, entry in TRANSFERS.items()}


def assert_transfer_legal(src: str, dst: str) -> None:
    tc = transfer_contract(src, dst)
    if not tc.legal:
        raise ValueError(f"illegal transfer {src}->{dst}: {tc.note}")


__all__ = [
    "GLOBAL", "LOCAL", "REGISTER", "AUX", "IMPLICIT_CACHE", "UNMATERIALIZED",
    "MemorySpaceFacts", "MemorySpaceRef", "SPACE_FACTS", "space_facts_table",
    "RegisterBudgetFacts", "REGISTER_BUDGETS", "register_budget",
    "register_budgets_table", "STATUS_REGISTER_UNSOURCED",
    "TransferContract", "transfer_contract", "transfer_table", "assert_transfer_legal",
    "STATUS_REFERENCE_IMPLEMENTATION", "STATUS_DEVICE_VALIDATED", "STATUS_UNRESOLVED",
]
