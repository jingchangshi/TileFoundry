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

from .memory import SPACE_FACTS, MemorySpaceFacts

GMEM = "gmem"
SMEM = "smem"
RMEM = "rmem"
L1 = "L1"
SLAB = "slab"

SPACE_OF = {"gmem": "ascend.gm", "smem": "ascend.ub", "rmem": "ascend.vreg", "L1": "ascend.l1", "slab": "ascend.ub"}


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

    def as_dict(self) -> dict:
        return {
            "space": self.space, "physical": self.physical, "role": self.role,
            "owner": self.owner, "allocatable": self.allocatable,
            "capacity_rule": self.capacity_rule, "alignment": self.alignment,
            "nominal_bytes": self.nominal_bytes, "usable_bytes": self.usable_bytes,
            "capacity_known": self.capacity_known, "source": self.source,
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
        return SpaceSemantics(
            RMEM, physical, "VF-private vector/predicate registers", facts.owner, True,
            "per-register 256 B; aggregate register budget UNSOURCED",
            facts.alignment, facts.nominal_bytes, facts.usable_bytes, capacity_known, facts.source,
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
        if access.wants == "ascend.l1" and access.domain != "aic":
            raise PlacementRefusal(
                f"{access.name}: L1 is AIC-local and cannot hold an {access.domain} buffer"
            )
        reverse = {physical: space for space, physical in SPACE_OF.items()}
        return reverse.get(access.wants, access.wants)
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


def place(accesses) -> dict:
    """Classify every access and enforce capacity per space; refuse what fits nothing."""
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
    return {space: result.as_dict() for space, result in results.items()}


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
