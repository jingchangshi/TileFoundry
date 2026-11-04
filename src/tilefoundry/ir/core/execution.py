"""Declare execution regions and the placement law of the operations inside them.

An execution region is not a mesh. A mesh answers *which participants run this
work*; an execution region answers *in which execution mode the work runs*, and
therefore which values may cross its boundary at all. A vector-register value
has no meaning outside the region that gives it one, so the boundary rule is a
semantic rule, not a convention.

This module owns the vocabulary only:

* ExecutionKind -- one target-declared execution mode: its surface spelling
  (T.SimdVF), the storage classes that are private to it, the mask width its
  instructions expect, and the topology it must be entered under.
* RegionRequirement -- what one operation states about where it may appear.
  None means the operation states nothing, which is why every pre-existing
  operation keeps its behaviour unchanged.
* the registry that maps a surface marker to its declaration.

The checks live in tilefoundry.visitor_registry.verify; this module only states
the facts they read, so one declaration drives parse, analysis and diagnosis
instead of three copies drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from tilefoundry.ir.core.op_registry import _register
from tilefoundry.ir.types.storage import StorageKind

Placement = Literal["inside", "outside"]

PLACEMENTS: tuple[str, ...] = ("inside", "outside")


@dataclass(frozen=True)
class RegionRequirement:
    """Where one operation may appear, relative to one execution region kind.

    "inside" means the operation is only meaningful with that region in force;
    "outside" means entering that region makes the operation invalid. Both
    directions are needed: a register instruction outside its region and a
    direct-memory transfer inside one are different defects of the same
    boundary.
    """

    kind: str
    placement: Placement

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("RegionRequirement: kind must be a non-empty string")
        if self.placement not in PLACEMENTS:
            raise ValueError(
                f"RegionRequirement: placement must be one of {PLACEMENTS}, "
                f"got {self.placement!r}"
            )


@dataclass(frozen=True)
class ExecutionKind:
    """One target-declared execution mode and the boundary it draws.

    register_storages are the storage classes that exist only inside the
    region: a value held there may not be captured, yielded, or read outside.
    mask_width_bits is the element width the region's instructions reduce
    their predicates to; an instruction that states a different width is
    addressing lanes the region does not have. required_topologies names the
    hardware levels the region must be entered under -- a region is not a
    mesh, so it does not open one; it requires one.
    """

    name: str
    surface: str
    register_storages: tuple[StorageKind, ...] = ()
    mask_width_bits: int | None = None
    required_topologies: tuple[str, ...] = ()
    nestable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not all(
            part.isidentifier() for part in self.name.split(".")
        ):
            raise ValueError(
                f"ExecutionKind: name must be a dotted identifier, got {self.name!r}"
            )
        if not isinstance(self.surface, str) or not self.surface.isidentifier():
            raise ValueError(
                f"ExecutionKind: surface must be a single identifier, got {self.surface!r}"
            )
        if any(not isinstance(storage, StorageKind) for storage in self.register_storages):
            raise TypeError("ExecutionKind: register_storages must be StorageKind values")
        width = self.mask_width_bits
        if width is not None and (isinstance(width, bool) or not isinstance(width, int) or width < 1):
            raise ValueError(
                f"ExecutionKind {self.name!r}: mask_width_bits must be a positive int, "
                f"got {width!r}"
            )
        if any(not isinstance(level, str) or not level for level in self.required_topologies):
            raise ValueError(
                f"ExecutionKind {self.name!r}: required_topologies must be non-empty names"
            )
        if not isinstance(self.nestable, bool):
            raise TypeError(f"ExecutionKind {self.name!r}: nestable must be a bool")

    def owns_storage(self, storage: object) -> bool:
        """Whether *storage* is private to this region kind."""
        return storage in self.register_storages


_KINDS: dict[str, ExecutionKind] = {}
_SURFACES: dict[tuple[str, str], ExecutionKind] = {}


def register_execution_kind(kind: ExecutionKind, *, dialect: str = "T") -> type:
    """Publish *kind* as a surface marker in *dialect* and index its name.

    The returned marker class is what a with-statement surface resolves to
    through the dialect tree; the parser reads its declaration rather than
    hard-coding a spelling. A repeated name or surface is refused instead of
    silently replacing the first registration.
    """
    if not isinstance(kind, ExecutionKind):
        raise TypeError(
            f"register_execution_kind expects an ExecutionKind, got {type(kind).__name__}"
        )
    if kind.name in _KINDS:
        raise ValueError(f"execution kind {kind.name!r} is already registered")
    surface_key = (dialect, kind.surface)
    if surface_key in _SURFACES:
        raise ValueError(
            f"{dialect}.{kind.surface} is already the surface of execution kind "
            f"{_SURFACES[surface_key].name!r}"
        )
    marker = type(
        kind.surface,
        (),
        {
            "__doc__": f"Surface marker for execution kind {kind.name!r}.",
            "__module__": __name__,
            "execution_kind": kind,
        },
    )
    _register(dialect, (), kind.surface, marker)
    _KINDS[kind.name] = kind
    _SURFACES[surface_key] = kind
    return marker


def execution_kind(name: str) -> ExecutionKind | None:
    """The declaration named *name*, or None when nothing declares it."""
    return _KINDS.get(name)


def execution_kind_of(surface: object) -> ExecutionKind | None:
    """The declaration whose marker is *surface*, or None."""
    declared = getattr(surface, "execution_kind", None)
    return declared if isinstance(declared, ExecutionKind) else None


def declared_kinds() -> tuple[ExecutionKind, ...]:
    """Every registered declaration, in registration order."""
    return tuple(_KINDS.values())


def region_requirement(op: Any) -> RegionRequirement | None:
    """The placement *op* states, or None when it states none."""
    declaration = op if isinstance(op, type) else type(op)
    stated = getattr(declaration, "execution_region", None)
    if stated is None:
        return None
    if not isinstance(stated, RegionRequirement):
        raise ValueError(
            f"{declaration.__name__}.execution_region must be a RegionRequirement or None"
        )
    return stated


def declared_mask_width(op: Any) -> int | None:
    """The mask width *op* states through its declared attribute, or None."""
    declaration = op if isinstance(op, type) else type(op)
    attribute = getattr(declaration, "mask_width_attribute", None)
    if attribute is None:
        return None
    if not isinstance(attribute, str) or not attribute:
        raise ValueError(
            f"{declaration.__name__}.mask_width_attribute must be an attribute name or None"
        )
    if isinstance(op, type):
        return None
    value = getattr(op, attribute, None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{declaration.__name__}.{attribute} must be an int bit width, got {value!r}"
        )
    return value


__all__ = [
    "ExecutionKind",
    "Placement",
    "PLACEMENTS",
    "RegionRequirement",
    "declared_kinds",
    "declared_mask_width",
    "execution_kind",
    "execution_kind_of",
    "register_execution_kind",
    "region_requirement",
]
