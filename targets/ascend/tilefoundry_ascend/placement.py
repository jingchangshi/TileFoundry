"""P1: memory-space semantics, capacity model, and access->space classification.

This replaces informal placement. Instead of "this looks like smem", every space states
what LIVES there, who OWNS it, what it can HOLD, and a classified access decides which
space each buffer lands in. A buffer that fits no space is REFUSED with a named reason
rather than dropped into a default.

The five semantic spaces the round needs are derived from the registered physical facts
(memory.SPACE_FACTS), not invented beside them:

  gmem  -> ascend.gm      device-global HBM; function inputs/outputs, cross-core shared
  smem  -> ascend.ub      per-AIV UB; tiles staged by MTE, live across a VF boundary
  rmem  -> ascend.vreg    VF-private registers; compute temps that must not escape
  L1    -> ascend.l1      AIC-local; registered for legality, not allocated in AIV work
  slab  -> ascend.ub      a versioned staging region inside UB (ring of V versions)

L2 is an implicit cache and is NOT a placement: it is refused, not modelled as a buffer.

Capacity is stated as nominal vs usable with provenance, because they differ: UB is 256
KiB nominal but the usable figure depends on reservation/compile options/execution mode,
and the aggregate register budget is NOT sourced, so rmem usable capacity is UNKNOWN and
is reported as unknown rather than guessed. A placement that would need that number is
refused as unknown, never admitted on a fabricated value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .memory import REGISTER, SPACE_FACTS, MemorySpaceFacts

GMEM = "gmem"
SMEM = "smem"
RMEM = "rmem"
L1 = "L1"
SLAB = "slab"

SPACE_OF = {"gmem": "ascend.gm", "smem": "ascend.ub", "rmem": "ascend.vreg", "L1": "ascend.l1", "slab": "ascend.ub"}
# An explicit physical request maps to the PHYSICAL space's default semantic space.
# ascend.ub maps to smem (not slab): slab is a ROLE (versioned ring), never a physical id.
PHYSICAL_TO_SPACE = {"ascend.gm": "gmem", "ascend.ub": "smem", "ascend.vreg": "rmem", "ascend.l1": "L1"}


class PlacementRefusal(ValueError):
    """A buffer that no space can hold, with the reason it fits none."""


@dataclass(frozen=True)
class SpaceSemantics:
    space: str
    physical: str
    role: str
    owner: str
    allocatable: bool
    capacity_rule: str
    alignment: int
    nominal_bytes: Optional[int]
    usable_bytes: Optional[int]
    capacity_known: bool
    source: str
    # For register classes this is the WIDTH of one register, NOT an aggregate bound.
    register_width_bytes: Optional[int] = None

    def as_dict(self) -> dict:
        return {
            "space": self.space, "physical": self.physical, "role": self.role,
            "owner": self.owner, "allocatable": self.allocatable,
            "capacity_rule": self.capacity_rule, "alignment": self.alignment,
            "nominal_bytes": self.nominal_bytes, "usable_bytes": self.usable_bytes,
            "capacity_known": self.capacity_known, "source": self.source,
            "register_width_bytes": self.register_width_bytes,
        }


def _facts(physical: str) -> MemorySpaceFacts:
    facts = SPACE_FACTS.get(physical)
    if facts is None:
        raise PlacementRefusal(f"no registered memory-space facts for {physical!r}")
    return facts


def space_semantics(space: str) -> SpaceSemantics:
    """The semantics of one semantic space, derived from its physical facts."""
    physical = SPACE_OF.get(space)
    if physical is None:
        raise PlacementRefusal(f"unknown semantic space {space!r}; known: {sorted(SPACE_OF)}")
    facts = _facts(physical)
    capacity_known = facts.usable_bytes is not None
    if space == SLAB:
        return SpaceSemantics(
            SLAB, physical, "versioned staging slot ring inside UB", facts.owner, True,
            "parent usable_bytes / versions, minus reserved; one slot must hold a tile",
            facts.alignment, facts.nominal_bytes, facts.usable_bytes, capacity_known, facts.source,
        )
    if space == RMEM:
        # CATEGORY CORRECTION: 256 B is the WIDTH of ONE register, not an aggregate bound.
        # Using it as the rmem capacity was a category error; the register COUNT is
        # UNSOURCED, so rmem states NO aggregate capacity and NO nominal total.
        return SpaceSemantics(
            RMEM, physical, "VF-private vector/predicate registers", facts.owner, True,
            "one register is 256 B; the register COUNT is UNSOURCED, so no aggregate rmem "
            "capacity is claimed (per-register width is not a bound)",
            facts.alignment, None, None, False, facts.source,
            register_width_bytes=facts.nominal_bytes,
        )
    role = {
        GMEM: "device-global shared storage; function inputs and outputs",
        SMEM: "per-AIV UB; DMA-staged tiles live across a VF boundary",
        L1: "AIC-local storage; registered for legality, not allocated in AIV work",
    }.get(space, "")
    return SpaceSemantics(
        space, physical, role, facts.owner, facts.allocatable,
        "usable_bytes from registered facts" if capacity_known else "usable capacity UNSOURCED",
        facts.alignment, facts.nominal_bytes, facts.usable_bytes, capacity_known, facts.source,
    )


def space_table() -> dict:
    return {space: space_semantics(space).as_dict() for space in (GMEM, SMEM, RMEM, L1, SLAB)}


ROLE_PARAM = "function-param"
ROLE_RESULT = "function-result"
ROLE_STAGED_TILE = "staged-tile"
ROLE_PRIVATE_TEMP = "private-temp"
ROLE_VERSIONED_SLOT = "versioned-slot"
ROLE_AIC_OPERAND = "aic-operand"
ROLE_SCALAR = "scalar"


@dataclass(frozen=True)
class AccessSpec:
    """One buffer placement request, classified rather than assumed."""
    name: str
    role: str
    bytes: int
    versions: int = 1
    domain: str = "aiv"
    inside_vf: bool = False
    wants: Optional[str] = None


def classify(access: AccessSpec) -> str:
    """The semantic space an access lands in, or PlacementRefusal when none fits."""
    if access.bytes < 0:
        raise PlacementRefusal(f"{access.name}: negative byte footprint")
    if access.wants is not None:
        if access.wants == "ascend.l2":
            raise PlacementRefusal(
                f"{access.name}: ascend.l2 is an implicit cache, not an allocatable space"
            )
        facts = _facts(access.wants)
        if not facts.allocatable:
            raise PlacementRefusal(f"{access.name}: {access.wants} is declared non-allocatable")
        # An explicit request does NOT buy an exemption from legality: it must still be a
        # legal home for THIS access. Asking for rmem outside a VF, or L1 in an AIV
        # program, is refused even though the caller named the space.
        space = PHYSICAL_TO_SPACE.get(access.wants)
        if space is None:
            raise PlacementRefusal(f"{access.name}: {access.wants} has no semantic space")
        if space == RMEM and not access.inside_vf:
            raise PlacementRefusal(
                f"{access.name}: {access.wants} is VF-private, and an explicit request does "
                "not make it legal outside a VF"
            )
        if space == L1 and access.domain != "aic":
            raise PlacementRefusal(
                f"{access.name}: {access.wants} is AIC-local, and an explicit request does "
                f"not make it legal in an {access.domain} program"
            )
        # An explicit PHYSICAL space must not erase VERSIONED semantics: a versioned slot
        # asked for as ascend.ub is still a slab ring (so place() multiplies by versions),
        # and a versioned slot cannot be forced into a space that has no slab.
        if access.role == ROLE_VERSIONED_SLOT or access.versions > 1:
            if access.wants == "ascend.ub":
                return SLAB
            raise PlacementRefusal(
                f"{access.name}: versioned slot requires UB/slab, but {access.wants} was "
                "requested explicitly"
            )
        return space
    if access.role in (ROLE_PARAM, ROLE_RESULT):
        return GMEM
    if access.role == ROLE_AIC_OPERAND:
        if access.domain != "aic":
            raise PlacementRefusal(
                f"{access.name}: AIC operand requested in an {access.domain} program"
            )
        return L1
    if access.role == ROLE_VERSIONED_SLOT or access.versions > 1:
        return SLAB
    if access.role == ROLE_STAGED_TILE:
        return SMEM
    if access.role in (ROLE_PRIVATE_TEMP, ROLE_SCALAR):
        if not access.inside_vf:
            raise PlacementRefusal(
                f"{access.name}: {access.role} storage is VF-private and this access is outside a VF"
            )
        return RMEM
    raise PlacementRefusal(
        f"{access.name}: role {access.role!r} fits no declared space; refusing rather than defaulting"
    )


@dataclass
class PlacementResult:
    space: str
    accesses: list
    bytes: int
    capacity: Optional[int]
    capacity_known: bool
    slots: int
    bound: Optional[int] = None

    def as_dict(self) -> dict:
        return {
            "space": self.space, "accesses": [a.name for a in self.accesses],
            "bytes": self.bytes, "capacity": self.capacity,
            "capacity_known": self.capacity_known, "slots": self.slots,
            "bound": self.bound,
        }


def place(accesses, *, include_physical: bool = False) -> dict:
    """Classify every access and enforce capacity; refuse what fits nothing.

    Capacity is enforced BOTH per semantic space and per PHYSICAL space. smem and slab are
    both ascend.ub, so enforcing the semantic spaces separately would let 200 KiB smem plus
    200 KiB slab pass while overcommitting the same 256 KiB UB.
    """
    results = {}
    for access in accesses:
        space = classify(access)
        semantics = space_semantics(space)
        result = results.get(space)
        if result is None:
            # The BOUND is the usable figure when it is sourced, otherwise the nominal
            # capability, which is a hard ceiling a buffer can never exceed. Usable being
            # unknown does not make the ceiling infinite; it only means the usable
            # fraction is unstated (reported via capacity_known / require_known_capacity).
            bound = semantics.usable_bytes if semantics.usable_bytes is not None else semantics.nominal_bytes
            result = results[space] = PlacementResult(
                space, [], 0, semantics.usable_bytes, semantics.capacity_known, 1, bound
            )
        result.accesses.append(access)
        if space == SLAB:
            versions = max(1, access.versions)
            result.slots = max(result.slots, versions)
            result.bytes += access.bytes * versions
        else:
            result.bytes += access.bytes
    for result in results.values():
        if result.bound is not None and result.bytes > result.bound:
            raise PlacementRefusal(
                f"{result.space}: {result.bytes} bytes requested exceeds capacity bound {result.bound}"
            )
    physical_totals: dict = {}
    for space, result in results.items():
        physical = space_semantics(space).physical
        entry = physical_totals.setdefault(physical, {"bytes": 0, "spaces": []})
        entry["bytes"] += result.bytes
        entry["spaces"].append(space)
    for physical, entry in physical_totals.items():
        facts = _facts(physical)
        if facts.category == REGISTER:
            # Register "nominal" is ONE register's width, not an aggregate budget: using it
            # as a physical bound is the same category error the rmem semantics fix. An
            # aggregate register budget needs a register COUNT, which is unsourced.
            bound = None
        else:
            bound = facts.usable_bytes if facts.usable_bytes is not None else facts.nominal_bytes
        if bound is not None and entry["bytes"] > bound:
            raise PlacementRefusal(
                f"physical {physical}: {entry['bytes']} bytes across {sorted(entry['spaces'])} "
                f"exceeds capacity bound {bound}; summing semantic spaces separately would "
                "double-count the same physical memory"
            )
    spaces = {space: result.as_dict() for space, result in results.items()}
    if include_physical:
        return {"spaces": spaces, "physical": physical_totals}
    return spaces


def require_known_capacity(accesses) -> dict:
    """place() plus a refusal if any chosen space has no sourced usable capacity."""
    placed = place(accesses)
    for space, result in placed.items():
        if not result["capacity_known"]:
            raise PlacementRefusal(
                f"{space}: usable capacity is UNSOURCED; refusing a bound check rather than assuming one"
            )
    return placed


__all__ = [
    "GMEM", "SMEM", "RMEM", "L1", "SLAB", "SPACE_OF", "PlacementRefusal",
    "SpaceSemantics", "AccessSpec", "PlacementResult", "space_semantics", "space_table",
    "classify", "place", "require_known_capacity",
    "ROLE_PARAM", "ROLE_RESULT", "ROLE_STAGED_TILE", "ROLE_PRIVATE_TEMP",
    "ROLE_VERSIONED_SLOT", "ROLE_AIC_OPERAND", "ROLE_SCALAR",
]
