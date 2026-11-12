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


@dataclass(frozen=True)
class MemorySpaceFacts:
    name: str
    category: str
    owner: str                    # "device" | "aiv" | "aic" | "vf"
    nominal_bytes: Optional[int]
    usable_bytes: Optional[int]
    alignment: int
    reachable_domains: tuple      # execution domains that may address it
    register_encoding: Optional[str] = None
    source: str = ""
    allocatable: bool = True

    def as_dict(self) -> dict:
        return {
            "name": self.name, "category": self.category, "owner": self.owner,
            "nominal_bytes": self.nominal_bytes, "usable_bytes": self.usable_bytes,
            "alignment": self.alignment, "reachable_domains": list(self.reachable_domains),
            "register_encoding": self.register_encoding, "source": self.source,
            "allocatable": self.allocatable,
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
    "ascend.bias": MemorySpaceFacts("ascend.bias", AUX, "aic", None, None, 64,
                                    ("aic",), source="auxiliary fixpipe buffer; registered only"),
}


def space_facts_table() -> dict:
    return dict(SPACE_FACTS)


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


def transfer_contract(src: str, dst: str) -> TransferContract:
    if src == "ascend.l2" or dst == "ascend.l2":
        return TransferContract(src, dst, "-", "-", False, "L2 is an implicit cache, not allocatable")
    entry = TRANSFERS.get((src, dst))
    if entry is None:
        return TransferContract(src, dst, "-", "-", False, "no declared data path")
    pipe, legal, note = entry
    return TransferContract(src, dst, pipe, pipe, legal, note)


def assert_transfer_legal(src: str, dst: str) -> None:
    tc = transfer_contract(src, dst)
    if not tc.legal:
        raise ValueError(f"illegal transfer {src}->{dst}: {tc.note}")


__all__ = [
    "GLOBAL", "LOCAL", "REGISTER", "AUX", "IMPLICIT_CACHE", "UNMATERIALIZED",
    "MemorySpaceFacts", "MemorySpaceRef", "SPACE_FACTS", "space_facts_table",
    "TransferContract", "transfer_contract", "assert_transfer_legal",
]
