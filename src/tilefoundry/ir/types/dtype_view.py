"""Explicit dtype-view aliasing: a root read at a different element size via byte offset.

A view is NOT a copy and NOT a conversion. It states that a byte range of an existing root
is addressed as elements of another dtype. To be sound it must carry:

  * element size (bytes per element of the viewed dtype),
  * byte offset from the root (not an element index: element size changes),
  * extent in viewed elements,
  * alignment the offset must satisfy,
  * the aliasing relation (which root it aliases).

The verifier refuses, with a NAMED reason:
  * misaligned        - byte offset not a multiple of the declared alignment,
  * out-of-bounds     - offset + extent*element_size exceeds the root's byte size,
  * overlapping-write - two distinct views of one root intersect in bytes and EITHER side
    is written. This is ACCESS-BASED, not dtype-based: the aliasing hazard is an unordered
    write against a read (or another write) over the same bytes, and element width does not
    change that hazard. Same-dtype write/read overlap is therefore just as forbidden as a
    cross-dtype one, and a cross-dtype READ/READ overlap is harmless and allowed.

SCOPE LIMITS, stated so nothing is silently considered checked: this validator does NOT
carry an allocation GENERATION and does NOT carry an element-COUNT cap. Staleness and
extent limits are enforced at the spec/plan level by view_spec, not here; a caller must not
treat those fields as validated by this function.

Consequence worth stating plainly: a bf16 root viewed as f32 with the SAME element count
does NOT fit -- it needs twice the bytes -- so that view is out-of-bounds and is refused.
Reading bf16 lanes as f32 requires a CONVERSION, not an alias; refusing here is what stops
a fabricated in-place reinterpret from masquerading as a free conversion.
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
    root: str
    dtype: str
    element_bytes: int
    byte_offset: int
    extent: int
    alignment: int
    root_bytes: int
    # "read" or "write", REQUIRED: there is deliberately no default. A default of "read"
    # would let a caller's write become invisible to the access-based overlap rule, which
    # is exactly the hole this field exists to close. Overlap legality is decided by
    # ACCESS, never by element size.
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
            # ACCESS-BASED: a write on either side is an unordered hazard over the bytes,
            # whatever the element sizes. Read/read overlap is harmless.
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
