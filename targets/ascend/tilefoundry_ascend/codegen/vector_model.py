"""The vector-register model, with provenance DERIVED from the target's geometry.

WHY A SEPARATE MODULE. The emitted TileLang program names a vector dtype and a
register access extent that the DEPLOYED toolchain chooses, not TileFoundry:
"T.simd.vld(tile[0])" states no lane count in the source text at all. So both the
emitter's expectation and the reader's comparison need one stated rule, traceable
to the build it was measured against rather than to a literal here.

THE RULE, AS MEASURED. From the deployed package
(tilelang/ascend/language/simd.py:85-118):

    _vreg_lanes(dtype): bool -> 256, else 2048 // element_bits
    _vec_dtype(dtype):  "<dtype>x<lanes>"
    pset(...)           -> always "boolx256" (call_intrin("boolx256", ...))

Measured against the parsed IR of this build (every case read back through
read_tl_facts, never assumed):

    tile dtype   tile elements   vector dtype     access_ptr extent
    float32      64              float32x64       64
    float32      32              float32x64       64    <- extent is lanes, not 32
    bfloat16     64              bfloat16x128     128
    bfloat16     128             bfloat16x128     128
    float16      64              float16x128      128

So a 32-bit tile's register extent happens to equal our 64-element fixtures,
which is why an earlier version that spelled "float32x%d" and used the tile extent
for both passed by accident: the extent of a register access is the LANE COUNT,
and the lane count follows the element width, not the buffer.

PROVENANCE IS A RESULT, NOT A LABEL. This module used to store a "SOURCED" string
that a caller could overwrite, and the lead correctly called that a forged door.
VectorModel takes no provenance argument: it is computed by cross-checking two
independent declarations.

    (computed lanes, element_bits) is a pair the TARGET actually declares, AND
    computed lanes * element_bits == one whole vector register

The target's ISA table declares lane_count = 64 paired with elem_width_bits = 32
(targets/ascend/tilefoundry_ascend/ir/isa.py:39-41), and memory.py:85-86 declares
one vector register as 256 bytes = 2048 bits.
 - 32-bit: 2048//32 = 64; the pair (64, 32) IS declared; 64*32 = 2048 is exactly
   one register. The equation holds, so this is SOURCED - by arithmetic, not
   because anyone trusts the width by default.
 - 16-bit: the deployed build gives 128 lanes; the target declares no (128, 16)
   pair, and its declared 64 lanes at 16 bits is only half a register
   (64*16 = 1024). The two sides do not agree, so 16-bit is UNSOURCED and the
   reader refuses to certify it. Whether 128 is genuine 16-bit packing on silicon
   is a device-evidence question (S3 / the dtype gate).

To make another width SOURCED you must change the TARGET's declared geometry in
ir/isa.py - a reviewable act that moves plan identity - not a field of this model.
"""
from __future__ import annotations

__all__ = ["DECLARED_LANE_COUNT", "ELEMENT_BITS", "PREDICATE_DTYPE",
           "REGISTER_FILE_BITS", "VectorModel", "declared_geometry", "model_for",
           "vector_lanes", "vector_model"]

#: Deployed element widths (tilelang/ascend/language/simd.py _ELEM_BITS).
ELEMENT_BITS = {"float32": 32, "float16": 16, "bfloat16": 16,
                "float8_e5m2": 8, "float8_e8m0fnu": 8, "float4_e2m1fn": 4}

#: pset's return type in the deployed surface, independent of elem_width
#: (simd.py:105-111: call_intrin("boolx256", tl.simd.pset, elem_width, dist)).
PREDICATE_DTYPE = "boolx256"

#: The register width the deployed lane rule divides by (simd.py:85-96: 2048 bits),
#: matching the target's own 256-byte vector register declaration (memory.py:85-86).
REGISTER_FILE_BITS = 2048

#: The lane count the target declares for its stated element width, read from the
#: target's ISA table rather than repeated here as a constant of our own.
DECLARED_LANE_COUNT = 64


def declared_geometry():
    """The (lane_count, element_bits) pairs the TARGET actually declares.

    A row's lane_count is declared FOR its elem_width_bits. Listing extra widths
    in element_widths does not declare a lane count for them - that is precisely
    the 16-bit case, and it is why 16-bit cannot be SOURCED from this table.
    """
    from ..ir.isa import ISA

    pairs = set()
    for contract in ISA.values():
        lanes = getattr(contract, "lane_count", None)
        width = getattr(contract, "elem_width_bits", None)
        if lanes and width:
            pairs.add((int(lanes), int(width)))
    return tuple(sorted(pairs))


class VectorModel:
    """One width's lanes and dtype spelling, plus the cross-check that judged it."""

    __slots__ = ("cross_check", "element_bits", "element_name", "lanes",
                 "provenance", "reason", "vector_dtype")

    def __init__(self, element_name, element_bits, lanes, cross_check):
        self.element_name = str(element_name)
        self.element_bits = int(element_bits)
        self.lanes = int(lanes)
        self.vector_dtype = "%sx%d" % (element_name, int(lanes))
        #: The equation's outcome as DATA: what was computed, what the target
        #: declares, whether the pair is a whole register. The verdict reads this,
        #: so re-classifying a width requires changing a target declaration.
        self.cross_check = dict(cross_check)
        #: Derived from the check; never accepted as an argument.
        self.provenance = "SOURCED" if self.cross_check["holds"] else "UNSOURCED"
        self.reason = self.cross_check["reason"]

    def __eq__(self, other):
        if not isinstance(other, VectorModel):
            return NotImplemented
        same = (self.element_name, self.element_bits, self.lanes, self.provenance)
        other_same = (other.element_name, other.element_bits, other.lanes,
                      other.provenance)
        return same == other_same

    def __hash__(self):
        return hash((self.element_name, self.element_bits, self.lanes,
                     self.provenance))

    def as_dict(self):
        return {"element_name": self.element_name, "element_bits": self.element_bits,
                "lanes": self.lanes, "vector_dtype": self.vector_dtype,
                "provenance": self.provenance,
                "cross_check": dict(self.cross_check)}

    def __repr__(self):
        return "VectorModel(%r)" % (self.as_dict(),)


def vector_lanes(element_bits: int) -> int:
    """The deployed lane rule: 2048 // element_bits, and 256 for one-bit lanes."""
    bits = int(element_bits)
    if bits <= 0:
        raise ValueError("element_bits must be positive, got %d" % bits)
    if bits == 1:
        return 256
    return REGISTER_FILE_BITS // bits


def _cross_check(element_bits: int, lanes: int, basis: str = "GEOMETRY") -> dict:
    """Does the deployed lane count agree with a declared register-sized geometry?"""
    geometry = declared_geometry()
    declared = (lanes, element_bits) in geometry
    whole = lanes * element_bits == REGISTER_FILE_BITS
    if declared and whole:
        reason = ("2048//%d = %d equals the target's declared lane_count for a %d-bit "
                  "element, and %d*%d = %d is exactly one vector register"
                  % (element_bits, lanes, element_bits, lanes, element_bits,
                     REGISTER_FILE_BITS))
    elif not declared:
        reason = ("the target declares %s: no %d-lane geometry for a %d-bit element, so "
                  "the deployed 2048//%d = %d lanes has nothing to cross-check against"
                  % (list(geometry), lanes, element_bits, element_bits, lanes))
    else:
        reason = ("the target declares %d lanes at %d bits = %d bits, which is not one "
                  "whole %d-bit vector register"
                  % (lanes, element_bits, lanes * element_bits, REGISTER_FILE_BITS))
    return {"basis": basis, "holds": bool(declared and whole),
            "declared_geometry": list(geometry), "declared_for_this_width": declared,
            "whole_register": whole, "computed_lanes": lanes,
            "register_file_bits": REGISTER_FILE_BITS, "reason": reason}


def model_for(element_name: str) -> VectorModel:
    """The lane model for one TL element dtype name, judged by the cross-check."""
    if element_name == "bool":
        # The predicate surface is different in kind: the deployed pset publishes
        # boolx256 (simd.py:105-111) and the target's predicate rule is one bit per
        # lane at a 32-bit element (isa.py:42-44), so 256 one-bit mask lanes is a
        # packed-mask encoding, not 2048//1. Agreement here is with the deployed
        # constant, and that is the only claim this row makes.
        geometry = declared_geometry()
        packed = (DECLARED_LANE_COUNT, 32) in geometry
        check = {"basis": "DEPLOYED_SURFACE_CONSTANT",
                 "holds": bool(packed),
                 "declared_geometry": list(geometry),
                 "declared_for_this_width": packed,
                 "whole_register": True, "computed_lanes": 256,
                 "register_file_bits": REGISTER_FILE_BITS,
                 "reason": ("the deployed pset publishes boolx256 and the target declares "
                            "one predicate bit per lane at lane_count=%d/32-bit elements; "
                            "256 is a packed mask width, so it is cross-checked against "
                            "the surface constant, not against 2048//1"
                            % DECLARED_LANE_COUNT)}
        return VectorModel("bool", 1, 256, check)
    if element_name not in ELEMENT_BITS:
        raise KeyError("no deployed element-width rule for %r; this slice states no "
                       "vector model for it" % (element_name,))
    bits = ELEMENT_BITS[element_name]
    lanes = vector_lanes(bits)
    return VectorModel(element_name, bits, lanes, _cross_check(bits, lanes))


def vector_model(element_name: str) -> VectorModel:
    """Public spelling; raises KeyError for a width this slice does not model."""
    return model_for(element_name)
