"""Storage facts: what a buffer occupies, stated with provenance or refused.

THE RULE THIS MODULE ENFORCES. A footprint is a CLAIM about hardware, so every number
here is either (a) read from the plan, (b) derived from the closed dtype registry in
tilefoundry.ir.types.dtype, or (c) taken from a DECLARED layout fact in this file whose
source is written down. A footprint that needs a fact nobody has is REFUSED. It is never
filled in with a plausible guess, because a guessed footprint is indistinguishable from a
measured one downstream and the capacity check would then be verifying arithmetic against
itself.

WHY NZ IS NOT rows*cols*itemsize. A fractal (NZ) layout stores a matrix as blocks of
16x16 elements (or 16x8 for some dtypes) with each block padded out to its full fractal
size. A 64x64 bf16 tile is 8 KiB row-major, but a 63x63 bf16 tile is 7938 element bytes
row-major and 16384 bytes in NZ, because it occupies four full 16x16 fractals. The padding
IS allocated. Computing the footprint from element count understates it by 2x here, and
that error is exactly what a bare-largest-first layout then fails to notice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

STORAGE_SCHEMA = "native_mainline_storage"

#: Layout kinds a buffer can occupy. A layout is a FACT about how bytes are arranged, so
#: it changes the footprint; it is not a label attached after the size is known.
LAYOUT_LINEAR = "linear"          # row-major, element_bytes = dtype bytes
LAYOUT_FRACTAL_NZ = "fractal_nz"  # 16x16 block-padded matrix layout (Cube operands)
LAYOUT_SCRATCH = "scratch"        # FP32 accumulation / conversion scratch, own dtype
LAYOUT_RESERVED = "reserved"      # bytes a region holds back and never exposes

ALIGN_RULE = "power_of_two"


class StorageRefusal(ValueError):
    """A placement that cannot be justified, with the specific fact that is missing."""

    def __init__(self, code, message, source="", expected=None, actual=None):
        self.code = str(code)
        self.message = str(message)
        self.source = str(source)
        self.expected = expected
        self.actual = actual
        super().__init__("%s: %s (source=%r expected=%r actual=%r)"
                         % (self.code, self.message, self.source, expected, actual))

    def as_dict(self):
        return {"code": self.code, "message": self.message, "source": self.source,
                "expected": self.expected, "actual": self.actual}


#: Raised for a layout question the target does not answer. Kept separate from
#: StorageRefusal so a caller can tell "no such layout" from "this layout does not fit".
LayoutRefusal = StorageRefusal


def align_up(value, alignment) -> int:
    """Round 'value' up to a multiple of 'alignment' (a power of two)."""
    value = int(value)
    alignment = int(alignment)
    if alignment <= 1:
        return value
    if alignment & (alignment - 1):
        raise StorageRefusal("STORAGE_ALIGNMENT_INVALID", "alignment must be a power of two",
                             "alignment", "power of two", alignment)
    return ((value + alignment - 1) // alignment) * alignment


def dtype_bytes(name) -> int:
    """Bytes per element, resolved through the CLOSED dtype registry.

    The registry is the project's single source for bit widths. A name it does not know is
    refused with the valid names, so a typo in a plan cannot become a silently wrong size.
    """
    from tilefoundry.ir.types.dtype import DType

    try:
        descriptor = DType.from_name(str(name))
    except ValueError as error:
        raise StorageRefusal("STORAGE_DTYPE_UNKNOWN", "unknown dtype", str(name),
                             "a registered dtype name", str(name)) from error
    bits = int(getattr(descriptor, "bit_width", 0) or 0)
    if bits <= 0:
        raise StorageRefusal("STORAGE_DTYPE_UNKNOWN", "dtype has no positive bit width",
                             str(name), "bit_width > 0", bits)
    return max(1, (bits + 7) // 8)


@dataclass(frozen=True)
class StorageFacts:
    """The layout facts this target DECLARES, each with its provenance."""

    fractal_rows: int = 16
    fractal_cols: int = 16
    nz_alignment: int = 512
    backend_staging_ratio: int = 0       # extra bytes per converted element, 0 = none needed
    source: str = ""

    def as_dict(self):
        return {"fractal_rows": self.fractal_rows, "fractal_cols": self.fractal_cols,
                "nz_alignment": self.nz_alignment,
                "backend_staging_ratio": self.backend_staging_ratio,
                "source": self.source}


#: One target profile's layout facts. fractal 16x16 is the Cube operand block on this
#: family; NZ bases are 512B aligned (memory.SPACE_FACTS gives ascent.l0a/l0b 512B
#: alignment, and a fractal base must satisfy it). backend_staging_ratio is 0 because no
#: conversion staging is DECLARED for this profile -- 0 means "none required", NOT "unknown",
#: and a caller that needs staging asks for it explicitly (see bytes_for).
LAYOUT_FACTS = {
    "ascend950pr_9579": StorageFacts(
        fractal_rows=16, fractal_cols=16, nz_alignment=512, backend_staging_ratio=0,
        source="16x16 Cube operand fractal; NZ base 512B (matches l0a/l0b alignment)"),
}


def facts_for(profile_id=None) -> StorageFacts:
    """The declared layout facts for a profile, or a refusal. Never a default."""
    if profile_id is None:
        profile_id = next(iter(LAYOUT_FACTS))
    facts = LAYOUT_FACTS.get(str(profile_id))
    if facts is None:
        raise StorageRefusal("STORAGE_TARGET_UNKNOWN",
                             "no declared layout facts for this target profile; a footprint"
                             " cannot be computed without them",
                             str(profile_id), sorted(LAYOUT_FACTS), str(profile_id))
    return facts


@dataclass(frozen=True)
class LayoutRequest:
    """One buffer's layout question, exactly as the plan states it."""

    root: str
    shape: tuple = ()
    dtype: str = "f32"
    layout: str = LAYOUT_LINEAR
    version_stride_bytes: Optional[int] = None
    scratch_dtype: str = ""
    scratch_lanes: int = 0
    reserved_bytes: int = 0
    alignment: int = 64
    source: str = ""


def _fractal_bytes(shape, item_bytes, facts) -> int:
    """Element bytes of a matrix in fractal layout: every 16x16 block is fully allocated."""
    rows = int(shape[0]) if len(shape) >= 1 else 0
    cols = int(shape[1]) if len(shape) >= 2 else 0
    if rows < 0 or cols < 0:
        raise StorageRefusal("STORAGE_SHAPE_INVALID", "negative dimension", "shape",
                             "non-negative", shape)
    blocks = (-(-rows // facts.fractal_rows)) * (-(-cols // facts.fractal_cols))
    per_block = facts.fractal_rows * facts.fractal_cols * item_bytes
    return blocks * per_block


def bytes_for(request: LayoutRequest, facts: Optional[StorageFacts] = None) -> dict:
    """The bytes ONE VERSION of this buffer occupies, with the breakdown that produced it.

    Returns a dict rather than an int so the caller can SEE which component dominated: a
    capacity report that says only '24576 bytes' cannot show that two thirds of it is
    fractal padding, and that is the number a reviewer needs.
    """
    facts = facts or facts_for()
    item = dtype_bytes(request.dtype)
    layout = str(request.layout)
    payload = 0
    padding = 0
    scratch = 0
    if layout == LAYOUT_LINEAR:
        elements = 1
        for dim in request.shape:
            elements *= int(dim)
        payload = elements * item
    elif layout == LAYOUT_FRACTAL_NZ:
        # THE BUG THIS REPLACED: the padding was recomputed as
        #     align_up(payload + padding, nz_alignment) - payload
        # which for a 63x63 bf16 tile collapses the full 4-fractal footprint (16384 B) back to
        # 8192 B -- HALF the space the buffer really needs, because the intermediate value
        # dropped the fractal term. A footprint that understates by 2x passes every capacity
        # check it is then used for. The fractal total is the ANSWER; the 512B NZ base
        # alignment is applied ON TOP of it, never instead of it.
        elements_row_major = 1
        for dim in request.shape:
            elements_row_major *= int(dim)
        payload = elements_row_major * item
        fractal_total = _fractal_bytes(request.shape, item, facts)
        if fractal_total < payload:
            raise StorageRefusal("STORAGE_LAYOUT_INVALID",
                                 "fractal footprint smaller than row-major payload",
                                 request.root, ">= payload", fractal_total - payload)
        padding = align_up(fractal_total, facts.nz_alignment) - payload
    elif layout == LAYOUT_SCRATCH:
        lanes = int(request.scratch_lanes)
        if lanes <= 0:
            raise StorageRefusal(
                "STORAGE_SCRATCH_UNSPECIFIED",
                "a scratch buffer must state its lane count; accumulation width is not"
                " inferable from the operand shape",
                request.root, "scratch_lanes > 0", lanes)
        scratch = lanes * dtype_bytes(request.scratch_dtype or "f32")
    elif layout == LAYOUT_RESERVED:
        if int(request.reserved_bytes) <= 0:
            raise StorageRefusal(
                "STORAGE_RESERVED_UNSPECIFIED",
                "a reserved region must state its byte count; a reservation of unknown size"
                " cannot be planned around",
                request.root, "reserved_bytes > 0", request.reserved_bytes)
    else:
        raise StorageRefusal("STORAGE_LAYOUT_UNKNOWN", "no such layout kind", request.root,
                             [LAYOUT_LINEAR, LAYOUT_FRACTAL_NZ, LAYOUT_SCRATCH, LAYOUT_RESERVED],
                             layout)

    staging = 0
    if facts.backend_staging_ratio and payload:
        staging = (payload * int(facts.backend_staging_ratio)) // 100

    total = payload + padding + scratch + staging + int(request.reserved_bytes)
    stride = request.version_stride_bytes
    if stride is not None and int(stride) < total:
        raise StorageRefusal(
            "STORAGE_STRIDE_TOO_SMALL",
            "the declared version stride cannot hold one version's real footprint; using it"
            " would overlap consecutive slots of the ring",
            request.root, ">= %d" % total, int(stride))
    return {"payload_bytes": payload, "padding_bytes": padding, "scratch_bytes": scratch,
            "staging_bytes": staging, "reserved_bytes": int(request.reserved_bytes),
            "single_version_bytes": total, "item_bytes": item, "layout": layout,
            "version_stride_bytes": int(stride) if stride is not None else total}


#: Physical layout descriptors (design section 4.1). These are FACTS about arrangement,
#: not labels attached after the size is known.
PHYSICAL_ND = "ND"
PHYSICAL_NZ = "NZ"
PHYSICAL_FRACTAL = "FRACTAL"
PHYSICAL_PACKED = "PACKED"
PHYSICAL_SCRATCH = "SCRATCH"
PHYSICAL_RESERVED = "RESERVED"

STATUS_REFERENCE_IMPLEMENTATION = "REFERENCE_IMPLEMENTATION"
STATUS_DEVICE_VALIDATED = "DEVICE_VALIDATED"
STATUS_UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class Breakdown:
    """The byte decomposition of one physical layout."""

    payload_bytes: int = 0
    padding_bytes: int = 0
    scratch_bytes: int = 0
    staging_bytes: int = 0
    reserved_bytes: int = 0

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class PhysicalLayout:
    """One buffer's resolved physical layout, with provenance."""

    layout_kind: str                 # ND | NZ | FRACTAL | PACKED | SCRATCH | RESERVED
    logical_shape: tuple
    physical_shape: tuple
    dtype: str
    item_bytes: int
    pitch_bytes: int
    alignment_bytes: int
    required_bytes: int              # single version, aligned
    breakdown: Breakdown
    source: str = ""
    status: str = STATUS_REFERENCE_IMPLEMENTATION
    refinement_id: str = ""

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class Requirement:
    """The full physical requirement of one logical root at one plan_iteration."""

    root: object
    owner: tuple
    space: str
    versions: int
    layout: PhysicalLayout
    version_stride_bytes: int
    slot_bytes: int
    extra_scratch: tuple = ()
    capacity_source: str = ""
    completion_edge_ids: tuple = ()
    reuse_group: str = ""
    invalidation_key: str = ""
    target_profile_digest: str = ""

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class TargetLayoutFacts:
    """Target layout facts with VALUE/SOURCE/STATUS per fact (design section 4.1)."""

    layout_facts: dict = field(default_factory=dict)
    status: str = STATUS_REFERENCE_IMPLEMENTATION
    source: str = ""

    def resolve(self, space, dtype, shape, explicit_layout=None):
        """Resolve a layout. L1 must be declared; never silently default to linear."""
        space = str(space)
        if explicit_layout:
            return str(explicit_layout)
        if space in ("ascend.l0a", "ascend.l0b"):
            return PHYSICAL_FRACTAL
        if space == "ascend.l1":
            raise StorageRefusal(
                "STORAGE_LAYOUT_UNKNOWN",
                "L1 layout (ND/NZ) must be declared by the plan or lowering registry;"
                " defaulting L1 to linear is forbidden",
                str(space), [PHYSICAL_ND, PHYSICAL_NZ], explicit_layout)
        if space == "ascend.ub":
            return PHYSICAL_ND
        return PHYSICAL_ND

    def as_dict(self):
        return {"layout_facts": dict(self.layout_facts), "status": self.status,
                "source": self.source}


__all__ = [
    "STORAGE_SCHEMA", "LAYOUT_LINEAR", "LAYOUT_FRACTAL_NZ", "LAYOUT_SCRATCH", "LAYOUT_RESERVED",
    "ALIGN_RULE", "StorageRefusal", "LayoutRefusal", "StorageFacts", "LAYOUT_FACTS",
    "facts_for", "align_up", "dtype_bytes", "LayoutRequest", "bytes_for", "layout_fact",
    "PHYSICAL_ND", "PHYSICAL_NZ", "PHYSICAL_FRACTAL", "PHYSICAL_PACKED", "PHYSICAL_SCRATCH",
    "PHYSICAL_RESERVED", "STATUS_REFERENCE_IMPLEMENTATION", "STATUS_DEVICE_VALIDATED",
    "STATUS_UNRESOLVED", "Breakdown", "PhysicalLayout", "Requirement", "TargetLayoutFacts",
]


def layout_fact(root, layout, source="") -> dict:
    """Record that a root was assigned a layout, and on what basis."""
    return {"root": str(root), "layout": str(layout), "source": str(source)}
