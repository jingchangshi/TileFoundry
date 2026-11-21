"""Explicit dtype-view aliasing: a root read at a different element size.

A view is not a copy or a conversion: it addresses a byte range of an existing
root as elements of another dtype, carrying element size, byte offset, extent,
alignment, and the aliased root. The verifier refuses misalignment,
out-of-bounds ranges, and overlapping writes with named reasons; overlap is
decided by access, so read/read is harmless and a write on either side is not.
Staleness and element-count caps belong to view_spec, not this module.
"""
from __future__ import annotations

from dataclasses import dataclass


class DtypeViewError(ValueError):
    """A dtype view that is not sound, with a named reason."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason


@dataclass(frozen=True)
class DtypeViewSpec:
    """One dtype-view request with its aliasing relation stated explicitly.

    ``access`` is required and has no default: defaulting to "read" would let a
    caller's write become invisible to the access-based overlap rule. Overlap
    legality is decided by access, never by element size.
    """

    root: str
    dtype: str
    element_bytes: int
    byte_offset: int
    extent: int
    alignment: int
    root_bytes: int
    access: str

    @property
    def nbytes(self) -> int:
        return self.extent * self.element_bytes

    @property
    def end_byte(self) -> int:
        return self.byte_offset + self.nbytes

    def overlaps(self, other: "DtypeViewSpec") -> bool:
        if self.root != other.root:
            return False
        return self.byte_offset < other.end_byte and other.byte_offset < self.end_byte

    def as_dict(self) -> dict:
        return {
            "root": self.root, "dtype": self.dtype, "element_bytes": self.element_bytes,
            "byte_offset": self.byte_offset, "extent": self.extent,
            "alignment": self.alignment, "root_bytes": self.root_bytes,
            "nbytes": self.nbytes, "end_byte": self.end_byte, "access": self.access,
        }


def validate_view(view: DtypeViewSpec) -> DtypeViewSpec:
    """Refuse an unsound single view with a named reason."""
    if view.element_bytes <= 0:
        raise DtypeViewError("invalid-element-size", f"{view.root}: {view.element_bytes}")
    if view.extent < 0:
        raise DtypeViewError("invalid-extent", f"{view.root}: extent {view.extent}")
    if view.alignment <= 0 or view.byte_offset % view.alignment:
        raise DtypeViewError(
            "misaligned",
            f"{view.root}: byte offset {view.byte_offset} is not a multiple of alignment "
            f"{view.alignment}",
        )
    if view.end_byte > view.root_bytes:
        raise DtypeViewError(
            "out-of-bounds",
            f"{view.root}: bytes [{view.byte_offset}, {view.end_byte}) exceed the root's "
            f"{view.root_bytes} bytes ({view.extent} x {view.element_bytes} B elements)",
        )
    return view


def validate_views(views) -> tuple:
    """Validate each view, then refuse intersecting byte ranges with a WRITE on either side.

    ACCESS-BASED, not dtype-based: read/read overlap is allowed regardless of dtype; any
    write overlapping any other view of the same root is forbidden. Each view's access is
    REQUIRED, so a write cannot be silently treated as a read.
    """
    checked = [validate_view(view) for view in views]
    for view in checked:
        if view.access not in ("read", "write"):
            raise DtypeViewError(
                "invalid-access", f"{view.root}: access {view.access!r} is not read/write"
            )
    for index, first in enumerate(checked):
        for second in checked[index + 1:]:
            if not first.overlaps(second):
                continue
            if first.access != "read" or second.access != "read":
                raise DtypeViewError(
                    "overlapping-write",
                    f"{first.root}: {first.dtype} {first.access} "
                    f"[{first.byte_offset}, {first.end_byte}) overlaps {second.dtype} "
                    f"{second.access} [{second.byte_offset}, {second.end_byte}); a write on "
                    "either side over intersecting bytes is forbidden",
                )
    return tuple(checked)


__all__ = ["DtypeViewSpec", "DtypeViewError", "validate_view", "validate_views"]
