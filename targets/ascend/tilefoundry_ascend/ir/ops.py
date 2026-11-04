"""Ascend950PR AIV instruction declarations for the vector-fetch mode.

Each operation here states the three things the verifier needs: which storage
classes its operands live in, whether it runs inside the execution mode or
outside it, and what its own operands must agree on. The placement statement is
the load-bearing one. A vector instruction outside the mode has registers with no
meaning, and a direct-memory transfer inside the mode splits the issue stream the
mode exists to keep whole. Neither is a style preference, so neither is left to
convention.

The mask is an operand, not an attribute. A predicate that no node carries is a
claim about a register the program never mentions, and nothing can check it
against the data it selects. ``ascend_pset`` produces a real predicate register
value and ``ascend_vadd`` consumes it, so the mask is a value a reader can point
at, verify against the data shape, and evaluate.

Element width is not a lane count. ``pset(32)`` selects 32-bit elements; the AIV
vector unit still computes over 64 lanes. Those are different numbers, so they
are different facts: ``elem_width`` is checked against the mode declaration, and
nothing derives a lane count from it.
"""
from __future__ import annotations

from tilefoundry.evaluator.registry import register_schedule_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op, OpCapability
from tilefoundry.ir.core.execution import RegionRequirement
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import Constraint, OrPattern, SameConstraint, TensorPattern, utils
from tilefoundry.ir.types import DType, StorageKind, UnitType
from tilefoundry.utils.lazy import torch
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt
from tilefoundry.visitor_registry.access_relation import (
    identity_relations,
    register_access_relation,
)

from .execution import SIMD_VF_KIND

READ = MemoryEffect.READ
WRITE = MemoryEffect.WRITE

_AIV = utils.memory_execution_mesh()

#: The dtypes these instructions compute on. The set is stated once, and the
#: mode declaration states the element width it must agree with; a dtype that
#: happens to be 32 bits wide but is not f32 is a different quantity, not an
#: equally good substitute. Naming the set here is what makes "vadd is an f32
#: add" a checked claim instead of a docstring.
VECTOR_DTYPES: tuple[DType, ...] = (DType.f32,)

#: Staging loads also carry bf16, v33's input format: GM/UB hold packed bf16 and vld with
#: dist="UNPK_B16" unpacks it. This does NOT widen the COMPUTE dtype set -- arithmetic
#: instructions still take f32 and the mode's element width still checks them.
LOAD_DTYPES: tuple[DType, ...] = (DType.f32, DType.bf16)


def _operands_in(storage: StorageKind, *dtypes: DType) -> TensorPattern:
    """A tensor in *storage* whose dtype is one of *dtypes*.

    Parameterizing the operand pattern by dtype is the difference between a
    rule the matcher enforces and a rule a verifier remembers to restate. An
    i32 register is still 32 bits wide, so a width-only check would admit it;
    stating the dtypes admits exactly the arithmetic the instruction performs.
    """
    return TensorPattern(dtype=OrPattern(*dtypes) if len(dtypes) > 1 else dtypes[0], storage=storage)


_IN_GMEM = utils.tensor_in(StorageKind.GMEM)
_IN_UB = _operands_in(StorageKind.SMEM, *VECTOR_DTYPES)
_IN_VREG = _operands_in(StorageKind.RMEM, *VECTOR_DTYPES)
_IN_PREG = TensorPattern(dtype=DType.bool, storage=StorageKind.RMEM)
_IN_GMEM_LOAD = TensorPattern(dtype=OrPattern(DType.f32, DType.bf16), storage=StorageKind.GMEM)
_IN_UB_LOAD = _operands_in(StorageKind.SMEM, *LOAD_DTYPES)
_IN_VREG_LOAD = _operands_in(StorageKind.RMEM, *LOAD_DTYPES)
_IN_UB_STAGE = _operands_in(StorageKind.SMEM, *LOAD_DTYPES)
#: A vld destination is either an f32 element register (plain load) or the opaque carrier
#: register class (UNPK_B16): a union of exact signatures, not a dtype wildcard.
_IN_VREG_OR_CARRIER = OrPattern(
    TensorPattern(dtype=DType.f32, storage=StorageKind.RMEM),
    TensorPattern(dtype=DType.bf16, storage=StorageKind.CARRIER),
)

#: Exact signatures for the two vcvt directions. The carrier is a DISTINCT register
#: class: a bf16 value in CARRIER is a transport window, not an f32 element register.
#: The union is written as exact alternatives, and _VcvtSignature below is the rule that
#: actually checks which alternative each operand took (an OrPattern alone is not a
#: cross-operand proof).
_IN_CARRIER = TensorPattern(dtype=DType.bf16, storage=StorageKind.CARRIER)
_IN_F32_RMEM = TensorPattern(dtype=DType.f32, storage=StorageKind.RMEM)
_IN_CARRIER_OR_F32 = OrPattern(_IN_CARRIER, _IN_F32_RMEM)
_IN_F32_OR_CARRIER = OrPattern(_IN_F32_RMEM, _IN_CARRIER)


class _VldDist(Constraint):
    """Tie dist to the exact load signature: UNPK_B16 <-> bf16 src / carrier dst.

    The attribute reaches this rule through the constraint map (instruction.py), so the
    rule really discriminates; it is not a comment. An f32 load states no dist; a bf16 load
    must state UNPK_B16 and land in the carrier class.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "field", "dtype")
        object.__setattr__(self, "left", "src")
        object.__setattr__(self, "right", "dst")

    def holds(self, operands: dict) -> bool:
        src, dst = operands.get("src"), operands.get("dst")
        if src is None or dst is None:
            return True
        dist = operands.get("dist")
        if getattr(src, "dtype", None) != getattr(dst, "dtype", None):
            return False
        if getattr(src, "dtype", None) == DType.bf16:
            return dist == "UNPK_B16" and getattr(dst, "storage", None) is StorageKind.CARRIER
        return dist is None and getattr(dst, "storage", None) is StorageKind.RMEM

    def written(self) -> str:
        return (
            "bf16 loads state dist=UNPK_B16 and land in the carrier class; f32 loads state "
            "no dist and land in an element register"
        )


class _VcvtSignature(Constraint):
    """The two exact vcvt signatures, checked across src, dst AND the attributes.

    A carrier window is not an f32 element register and an f32 register is not a
    carrier. Direction is EXPLICIT (no inference from operand shape): forward reads a
    carrier and writes f32, reverse reads f32 and writes a carrier. part is explicit
    and must be 0; elem_width is the declared CONVERSION mask granularity, 16 =
    min(bf16, f32), never a number inferred from a bool tensor's shape.

    The attributes reach this rule through the same constraint map as vld's dist
    (instruction.py: check_call adds every declared attribute), so a missing or wrong
    part/width/direction is refused rather than defaulted.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "field", "dtype")
        object.__setattr__(self, "left", "src")
        object.__setattr__(self, "right", "dst")

    def holds(self, operands: dict) -> bool:
        src, dst = operands.get("src"), operands.get("dst")
        if src is None or dst is None:
            return False
        direction = operands.get("direction")
        part = operands.get("part")
        width = operands.get("elem_width")
        if direction not in ("forward", "reverse"):
            return False
        if isinstance(part, bool) or not isinstance(part, int) or part != 0:
            return False
        if isinstance(width, bool) or not isinstance(width, int) or width != 16:
            return False
        src_dtype = getattr(src, "dtype", None)
        dst_dtype = getattr(dst, "dtype", None)
        src_storage = getattr(src, "storage", None)
        dst_storage = getattr(dst, "storage", None)
        if direction == "forward":
            return (
                src_dtype == DType.bf16
                and src_storage is StorageKind.CARRIER
                and dst_dtype == DType.f32
                and dst_storage is StorageKind.RMEM
            )
        return (
            src_dtype == DType.f32
            and src_storage is StorageKind.RMEM
            and dst_dtype == DType.bf16
            and dst_storage is StorageKind.CARRIER
        )

    def written(self) -> str:
        return (
            "ascend_vcvt forward reads a bf16 CARRIER and writes an f32 RMEM, reverse reads "
            "an f32 RMEM and writes a bf16 CARRIER; direction must be stated, part must be "
            "STATED as the int 0 (a missing part is refused by name here), and the conversion "
            "mask granularity elem_width must be 16"
        )

    def refused(self, operands: dict) -> str:
        src, dst = operands.get("src"), operands.get("dst")

        def describe(value):
            if value is None:
                return "None"
            dtype = getattr(getattr(value, "dtype", None), "name", None)
            storage = getattr(getattr(value, "storage", None), "name", None)
            return f"{dtype}/{storage}"

        return (
            f"ascend_vcvt: src={describe(src)} dst={describe(dst)} "
            f"direction={operands.get('direction')!r} part={operands.get('part')!r} "
            f"elem_width={operands.get('elem_width')!r}; reads {self.written()}"
        )


class _VstsPack(Constraint):
    """Tie the store's pack to the exact signature: carrier <-> PK_B32, f32 <-> no pack."""

    def __init__(self) -> None:
        object.__setattr__(self, "field", "dtype")
        object.__setattr__(self, "left", "src")
        object.__setattr__(self, "right", "dst")

    def holds(self, operands: dict) -> bool:
        src = operands.get("src")
        if src is None:
            return False
        pack = operands.get("pack")
        if getattr(src, "storage", None) is StorageKind.CARRIER:
            return getattr(src, "dtype", None) == DType.bf16 and pack == "PK_B32"
        return pack is None

    def written(self) -> str:
        return (
            "a carrier window store must state pack=PK_B32 and land in bf16 UB; a plain f32 "
            "store states no pack"
        )


INSIDE = RegionRequirement(kind=SIMD_VF_KIND, placement="inside")
OUTSIDE = RegionRequirement(kind=SIMD_VF_KIND, placement="outside")


@register_op(dialect="T", category="ascend", name="ascend_gm_to_ub")
class GmToUb(Op):
    """GM->UB tile load (MTE2): a memory instruction, so outside the mode."""

    capability = OpCapability(None)
    execution_mesh = _AIV
    execution_region = OUTSIDE

    src = ParamDef(kind="input", effect=READ, pattern=_IN_GMEM_LOAD)
    dst = ParamDef(kind="input", effect=WRITE, pattern=_IN_UB_LOAD)

    between = (SameConstraint("shape", "src", "dst"), SameConstraint("dtype", "src", "dst"))


@register_op(dialect="T", category="ascend", name="ascend_ub_to_gm")
class UbToGm(Op):
    """UB->GM tile store (MTE3): likewise outside the mode."""

    capability = OpCapability(None)
    execution_mesh = _AIV
    execution_region = OUTSIDE

    src = ParamDef(kind="input", effect=READ, pattern=_IN_UB_LOAD)
    dst = ParamDef(kind="input", effect=WRITE, pattern=_IN_GMEM_LOAD)

    between = (SameConstraint("shape", "src", "dst"), SameConstraint("dtype", "src", "dst"))


@register_op(dialect="T", category="ascend", name="ascend_vld")
class Vld(Op):
    """UB->VREG lane load, instruction ``vld``."""

    capability = OpCapability(None)
    execution_mesh = _AIV
    execution_region = INSIDE

    src = ParamDef(kind="input", effect=READ, pattern=_IN_UB_STAGE)
    dst = ParamDef(kind="input", effect=WRITE, pattern=_IN_VREG_OR_CARRIER)
    dist = ParamDef(kind="attribute", annotation=str, optional=True, default=None)

    #: Narrow transport capability: bf16 carriers are admitted for THIS instruction only.
    #: Arithmetic signatures stay f32.
    transport_dtypes = (DType.bf16,)

    between = (SameConstraint("shape", "src", "dst"), _VldDist())


@register_op(dialect="T", category="ascend", name="ascend_vsts")
class Vsts(Op):
    """VREG->UB lane store, instruction ``vsts``."""

    capability = OpCapability(None)
    execution_mesh = _AIV
    execution_region = INSIDE

    src = ParamDef(kind="input", effect=READ, pattern=_IN_VREG_OR_CARRIER)
    # The reverse conversion path stores a bf16 carrier window into a bf16 UB tile, so the
    # destination admits bf16 as well as f32 (the dtype constraint still forces src==dst).
    dst = ParamDef(kind="input", effect=WRITE, pattern=_IN_UB_STAGE)
    #: pack states how a carrier window is written back to UB. The reverse conversion
    #: path packs two bf16 halves per 32-bit word (PK_B32); a plain f32 store states no
    #: pack. An optional attribute checked by the rule below, so it cannot default.
    pack = ParamDef(kind="attribute", annotation=str, optional=True, default=None)

    between = (
        SameConstraint("shape", "src", "dst"),
        SameConstraint("dtype", "src", "dst"),
        _VstsPack(),
    )


@register_op(dialect="T", category="ascend", name="ascend_pset")
class Pset(Op):
    """Predicate set, instruction ``pset``: produce the lane mask.

    ``elem_width`` is the element width the predicate selects, in bits. It is
    checked against the mode declaration and never converted into a lane count:
    the mask covers one bit per lane whatever the element width is.
    """

    capability = OpCapability(None)
    execution_mesh = _AIV
    execution_region = INSIDE
    mask_width_attribute = "elem_width"

    dst = ParamDef(kind="input", effect=WRITE, pattern=_IN_PREG)
    elem_width = ParamDef(kind="attribute", annotation=int)


@register_op(dialect="T", category="ascend", name="ascend_pset16")
class Pset16(Op):
    """Conversion-granularity predicate set: a mask produced at 16-bit granularity.

    pset32 stays reserved for FP32 arithmetic. A bf16<->f32 conversion selects
    elements at min(src, dst) width = 16, so this is a NEW producer form with the
    granularity declared on the instruction; it is not a second compute mode and the
    bool tensor shape still does not encode the predicate encoding. Declaring 16 here
    is checked against conversion_mask_width_bits, so a pset16 stating 32 is refused
    by name at placement.
    """

    capability = OpCapability(None)
    execution_mesh = _AIV
    execution_region = INSIDE
    mask_width_attribute = "elem_width"
    mask_class = "conversion"

    dst = ParamDef(kind="input", effect=WRITE, pattern=_IN_PREG)
    elem_width = ParamDef(kind="attribute", annotation=int)


@register_op(dialect="T", category="ascend", name="ascend_vcvt")
class Vcvt(Op):
    """Carrier-window <-> f32 conversion, predicated at conversion granularity.

    The forward path is the only consumer of an UNPK_B16 carrier window; a plain f32
    register is NOT a carrier. part is an explicit attribute (0), direction states
    which of the two exact signatures applies, and elem_width is the declared 16-bit
    conversion mask granularity. The mask is a VALUE produced by a pset in the same
    region instance before this instruction, not an attribute: the predicate-producer
    rule checks its dominance and its declared granularity.

    This is a CPU software model of the declared contract. The native PREG encoding,
    the physical part selection and the rounding/layout of a real 950PR issue are
    UNSOURCED until 03-20; nothing here claims a native instruction's exact behaviour.
    """

    capability = OpCapability(None)
    execution_mesh = _AIV
    execution_region = INSIDE
    mask_width_attribute = "elem_width"
    mask_class = "conversion"
    transport_dtypes = (DType.bf16,)

    src = ParamDef(kind="input", effect=READ, pattern=_IN_CARRIER_OR_F32)
    mask = ParamDef(kind="input", effect=READ, pattern=_IN_PREG)
    dst = ParamDef(kind="input", effect=WRITE, pattern=_IN_F32_OR_CARRIER)

    direction = ParamDef(kind="attribute", annotation=str)
    # part is OPTIONAL at construction on purpose: a missing part must reach the same
    # checked contract as part=1 or a wrong type and be REFUSED BY NAME there, rather
    # than crashing in Op.__init__ with a TypeError that is no verdict at all.
    part = ParamDef(kind="attribute", annotation=int, optional=True, default=None)
    elem_width = ParamDef(kind="attribute", annotation=int)

    between = (
        SameConstraint("shape", "src", "dst"),
        SameConstraint("shape", "src", "mask"),
        _VcvtSignature(),
    )


@register_op(dialect="T", category="ascend", name="ascend_vmula")
class Vmula(Op):
    """Predicated FP32 multiply-accumulate into acc: acc = acc + a * b.

    acc is READ|WRITE, not WRITE: an inactive lane keeps the value it carried into this
    issue, which is what makes the predicate observable and what makes a chain of vmula
    calls a loop-carried accumulation rather than a rename. The model is an UNFUSED
    two-step FP32 multiply then add. Whether the native issue fuses the multiply and the
    add (one rounding) is ASSUMED-UNKNOWN and UNSOURCED against the selected build: no
    native credit is claimed, and the model states which one it implements.
    """

    capability = OpCapability(None)
    execution_mesh = _AIV
    execution_region = INSIDE

    a = ParamDef(kind="input", effect=READ, pattern=_IN_VREG)
    b = ParamDef(kind="input", effect=READ, pattern=_IN_VREG)
    mask = ParamDef(kind="input", effect=READ, pattern=_IN_PREG)
    acc = ParamDef(kind="input", effect=READ | WRITE, pattern=_IN_VREG)

    between = (
        SameConstraint("shape", "a", "b"),
        SameConstraint("shape", "a", "acc"),
        SameConstraint("dtype", "a", "b"),
        SameConstraint("dtype", "a", "acc"),
    )


@register_op(dialect="T", category="ascend", name="ascend_vdup")
class Vdup(Op):
    """Broadcast one FP32 scalar into every ACTIVE lane of dst.

    dst is READ|WRITE: an inactive lane keeps what it held, so the predicate is
    observable here exactly as in vadd. The scalar is an instruction attribute in this
    model (ABI: a Python float rendered into the TIR call), never a lane count.
    """

    capability = OpCapability(None)
    execution_mesh = _AIV
    execution_region = INSIDE

    mask = ParamDef(kind="input", effect=READ, pattern=_IN_PREG)
    dst = ParamDef(kind="input", effect=READ | WRITE, pattern=_IN_VREG)
    value = ParamDef(kind="attribute", annotation=float)


@register_op(dialect="T", category="ascend", name="ascend_vadds")
class Vadds(Op):
    """Predicated lane add of an FP32 scalar: dst = a + value on active lanes."""

    capability = OpCapability(None)
    execution_mesh = _AIV
    execution_region = INSIDE

    a = ParamDef(kind="input", effect=READ, pattern=_IN_VREG)
    mask = ParamDef(kind="input", effect=READ, pattern=_IN_PREG)
    dst = ParamDef(kind="input", effect=READ | WRITE, pattern=_IN_VREG)
    value = ParamDef(kind="attribute", annotation=float)

    between = (
        SameConstraint("shape", "a", "dst"),
        SameConstraint("dtype", "a", "dst"),
    )


@register_op(dialect="T", category="ascend", name="ascend_vmuls")
class Vmuls(Op):
    """Predicated lane multiply by an FP32 scalar: dst = a * value on active lanes."""

    capability = OpCapability(None)
    execution_mesh = _AIV
    execution_region = INSIDE

    a = ParamDef(kind="input", effect=READ, pattern=_IN_VREG)
    mask = ParamDef(kind="input", effect=READ, pattern=_IN_PREG)
    dst = ParamDef(kind="input", effect=READ | WRITE, pattern=_IN_VREG)
    value = ParamDef(kind="attribute", annotation=float)

    between = (
        SameConstraint("shape", "a", "dst"),
        SameConstraint("dtype", "a", "dst"),
    )


@register_op(dialect="T", category="ascend", name="ascend_vmul")
class Vmul(Op):
    """Predicated FP32 multiply: dst = a * b on active lanes."""

    capability = OpCapability(None)
    execution_mesh = _AIV
    execution_region = INSIDE

    a = ParamDef(kind="input", effect=READ, pattern=_IN_VREG)
    b = ParamDef(kind="input", effect=READ, pattern=_IN_VREG)
    mask = ParamDef(kind="input", effect=READ, pattern=_IN_PREG)
    dst = ParamDef(kind="input", effect=READ | WRITE, pattern=_IN_VREG)

    between = (
        SameConstraint("shape", "a", "b"),
        SameConstraint("shape", "a", "dst"),
        SameConstraint("dtype", "a", "b"),
        SameConstraint("dtype", "a", "dst"),
    )


@register_op(dialect="T", category="ascend", name="ascend_vsqrt")
class Vsqrt(Op):
    """Predicated FP32 square root: dst = sqrt(a). NOT a reciprocal-square-root.

    The body needs rstd = 1 / sqrt(mean(x^2) + eps), written as an exact vsqrt followed by
    an explicit vdiv, never as an rsqrt approximation. Whether the native sqrt is
    correctly rounded is UNSOURCED; this model uses IEEE sqrt.
    """

    capability = OpCapability(None)
    execution_mesh = _AIV
    execution_region = INSIDE

    a = ParamDef(kind="input", effect=READ, pattern=_IN_VREG)
    mask = ParamDef(kind="input", effect=READ, pattern=_IN_PREG)
    dst = ParamDef(kind="input", effect=READ | WRITE, pattern=_IN_VREG)

    between = (SameConstraint("shape", "a", "dst"), SameConstraint("dtype", "a", "dst"))


@register_op(dialect="T", category="ascend", name="ascend_vdiv")
class Vdiv(Op):
    """Predicated FP32 divide: dst = a / b on active lanes."""

    capability = OpCapability(None)
    execution_mesh = _AIV
    execution_region = INSIDE

    a = ParamDef(kind="input", effect=READ, pattern=_IN_VREG)
    b = ParamDef(kind="input", effect=READ, pattern=_IN_VREG)
    mask = ParamDef(kind="input", effect=READ, pattern=_IN_PREG)
    dst = ParamDef(kind="input", effect=READ | WRITE, pattern=_IN_VREG)

    between = (
        SameConstraint("shape", "a", "b"),
        SameConstraint("shape", "a", "dst"),
        SameConstraint("dtype", "a", "b"),
        SameConstraint("dtype", "a", "dst"),
    )


@register_op(dialect="T", category="ascend", name="ascend_vcadd")
class Vcadd(Op):
    """One step of a pairwise reduction tree: dst[i] = src[2i] + src[2i+1], i < lanes/2.

    The reduction WIDTH is an explicit attribute (lanes), so the six source calls
    64->32->16->8->4->2 are a real SSA chain and dropping one leaves a partial sum. This
    model does NOT broadcast: only the first lanes/2 lanes of dst are written and the rest
    keep their value, so a broadcast is an explicit ascend_vdupv (POS_LOWEST). The native
    reduction tree and output layout are UNSOURCED / UNVALIDATED -- this is the software
    model of the DECLARED shape of the reduction, not evidence about the hardware.
    """

    capability = OpCapability(None)
    execution_mesh = _AIV
    execution_region = INSIDE

    src = ParamDef(kind="input", effect=READ, pattern=_IN_VREG)
    dst = ParamDef(kind="input", effect=READ | WRITE, pattern=_IN_VREG)
    lanes = ParamDef(kind="attribute", annotation=int)

    between = (SameConstraint("shape", "src", "dst"), SameConstraint("dtype", "src", "dst"))


@register_op(dialect="T", category="ascend", name="ascend_vdupv")
class Vdupv(Op):
    """Broadcast one LANE POSITION of src to every lane of dst (POS_LOWEST = position 0).

    The broadcast a reduction needs is THIS explicit instruction, never baked into vcadd.
    """

    capability = OpCapability(None)
    execution_mesh = _AIV
    execution_region = INSIDE

    src = ParamDef(kind="input", effect=READ, pattern=_IN_VREG)
    dst = ParamDef(kind="input", effect=WRITE, pattern=_IN_VREG)
    position = ParamDef(kind="attribute", annotation=int)

    between = (SameConstraint("shape", "src", "dst"), SameConstraint("dtype", "src", "dst"))


@register_op(dialect="T", category="ascend", name="ascend_vadd")
class Vadd(Op):
    """Predicated lane add, instruction ``vadd``: active lanes take ``a + b``.

    ``dst`` is READ|WRITE rather than WRITE because the predicate is per lane:
    an inactive lane keeps what it held, so this is an accumulate and not a
    rename. Stating that is what makes the mask observable -- in the evaluator, in
    the reference, and to a reader checking that the mask does anything at all.
    """

    capability = OpCapability(None)
    execution_mesh = _AIV
    execution_region = INSIDE

    a = ParamDef(kind="input", effect=READ, pattern=_IN_VREG)
    b = ParamDef(kind="input", effect=READ, pattern=_IN_VREG)
    mask = ParamDef(kind="input", effect=READ, pattern=_IN_PREG)
    dst = ParamDef(kind="input", effect=READ | WRITE, pattern=_IN_VREG)

    between = (
        SameConstraint("shape", "a", "b"),
        SameConstraint("shape", "a", "dst"),
        SameConstraint("dtype", "a", "b"),
        SameConstraint("dtype", "a", "dst"),
    )


for _op in (GmToUb, UbToGm, Vld, Vsts, Pset, Pset16, Vcvt, Vadd, Vmula, Vdup, Vadds, Vmuls, Vmul, Vsqrt, Vdiv, Vcadd, Vdupv):
    register_access_relation(_op)(identity_relations)


# Every instruction here is invoked as a statement, so its own type is unit. The
# value a caller sees comes from the operand it writes: the HIR schedule form
# reads its result type off the written operand, which is why a TIR statement op
# must not claim a tensor type of its own -- doing so makes the emitted program
# unparseable as a statement.
@register_typeinfer(GmToUb)
def _infer_gm_to_ub(call, ctx):
    return UnitType()


@register_typeinfer(UbToGm)
def _infer_ub_to_gm(call, ctx):
    return UnitType()


@register_typeinfer(Vld)
def _infer_vld(call, ctx):
    return UnitType()


@register_typeinfer(Vsts)
def _infer_vsts(call, ctx):
    return UnitType()


@register_typeinfer(Pset)
def _infer_pset(call, ctx):
    return UnitType()


@register_typeinfer(Pset16)
def _infer_pset16(call, ctx):
    return UnitType()


@register_typeinfer(Vcvt)
def _infer_vcvt(call, ctx):
    return UnitType()


@register_typeinfer(Vmula)
def _infer_vmula(call, ctx):
    return UnitType()


@register_typeinfer(Vdup)
def _infer_vdup(call, ctx):
    return UnitType()


@register_typeinfer(Vadds)
def _infer_vadds(call, ctx):
    return UnitType()


@register_typeinfer(Vmuls)
def _infer_vmuls(call, ctx):
    return UnitType()


@register_typeinfer(Vmul)
def _infer_vmul(call, ctx):
    return UnitType()


@register_typeinfer(Vsqrt)
def _infer_vsqrt(call, ctx):
    return UnitType()


@register_typeinfer(Vdiv)
def _infer_vdiv(call, ctx):
    return UnitType()


@register_typeinfer(Vcadd)
def _infer_vcadd(call, ctx):
    return UnitType()


@register_typeinfer(Vdupv)
def _infer_vdupv(call, ctx):
    return UnitType()


@register_typeinfer(Vadd)
def _infer_vadd(call, ctx):
    return UnitType()


@register_verify_stmt(Pset)
def _verify_pset(call, ctx):
    """The predicate this produces must be one bit per lane of the mask it fills."""
    dst = ctx.type_of(call.args[0])
    width = call.target.elem_width
    if isinstance(width, bool) or not isinstance(width, int):
        ctx.error(call, f"Pset: elem_width must be an int bit width, got {width!r}")
    if dst.dtype != DType.bool:
        ctx.error(call, f"Pset: dst must hold a predicate (bool), got {dst.dtype.name}")


@register_verify_stmt(Vcvt)
def _verify_vcvt(call, ctx):
    """The mask is a real bool predicate covering exactly the converted lanes."""
    src, mask, dst = (ctx.type_of(arg) for arg in call.args)
    if mask.dtype != DType.bool:
        ctx.error(call, f"Vcvt: mask must be a predicate (bool), got {mask.dtype.name}")
    if tuple(mask.shape) != tuple(src.shape):
        ctx.error(
            call,
            f"Vcvt: mask covers {tuple(mask.shape)} lanes but the source has {tuple(src.shape)}",
        )
    if tuple(dst.shape) != tuple(src.shape):
        ctx.error(call, f"Vcvt: dst {tuple(dst.shape)} does not cover {tuple(src.shape)}")


@register_verify_stmt(Pset16)
def _verify_pset16(call, ctx):
    """The producer states the conversion granularity explicitly: 16, not a shape."""
    dst = ctx.type_of(call.args[0])
    width = call.target.elem_width
    if isinstance(width, bool) or not isinstance(width, int) or width != 16:
        ctx.error(
            call,
            f"Pset16: elem_width must be the int 16 for a conversion mask, got {width!r}",
        )
    if dst.dtype != DType.bool:
        ctx.error(call, f"Pset16: dst must hold a predicate (bool), got {dst.dtype.name}")


@register_verify_stmt(Vmula)
def _verify_vmula(call, ctx):
    a, b, mask, acc = (ctx.type_of(arg) for arg in call.args)
    if mask.dtype != DType.bool:
        ctx.error(call, f"Vmula: mask must be a predicate (bool), got {mask.dtype.name}")
    if tuple(mask.shape) != tuple(a.shape):
        ctx.error(call, f"Vmula: mask covers {tuple(mask.shape)} lanes but the data has {tuple(a.shape)}")
    if b.shape != a.shape or b.dtype != a.dtype or acc.shape != a.shape or acc.dtype != a.dtype:
        ctx.error(call, "Vmula: a, b and acc must have one shape and dtype")


@register_verify_stmt(Vdup)
def _verify_vdup(call, ctx):
    mask, dst = (ctx.type_of(arg) for arg in call.args)
    if mask.dtype != DType.bool:
        ctx.error(call, f"Vdup: mask must be a predicate (bool), got {mask.dtype.name}")
    if tuple(mask.shape) != tuple(dst.shape):
        ctx.error(call, f"Vdup: mask covers {tuple(mask.shape)} lanes but dst has {tuple(dst.shape)}")
    if isinstance(call.target.value, bool) or not isinstance(call.target.value, (int, float)):
        ctx.error(call, f"Vdup: value must be a real scalar, got {call.target.value!r}")


@register_verify_stmt(Vadds)
def _verify_vadds(call, ctx):
    a, mask, dst = (ctx.type_of(arg) for arg in call.args)
    if mask.dtype != DType.bool or tuple(mask.shape) != tuple(a.shape):
        ctx.error(call, "Vadds: the mask must be a bool predicate covering the data lanes")
    if dst.shape != a.shape or dst.dtype != a.dtype:
        ctx.error(call, "Vadds: dst must match a")
    if isinstance(call.target.value, bool) or not isinstance(call.target.value, (int, float)):
        ctx.error(call, f"Vadds: value must be a real scalar, got {call.target.value!r}")


@register_verify_stmt(Vmuls)
def _verify_vmuls(call, ctx):
    a, mask, dst = (ctx.type_of(arg) for arg in call.args)
    if mask.dtype != DType.bool or tuple(mask.shape) != tuple(a.shape):
        ctx.error(call, "Vmuls: the mask must be a bool predicate covering the data lanes")
    if dst.shape != a.shape or dst.dtype != a.dtype:
        ctx.error(call, "Vmuls: dst must match a")
    if isinstance(call.target.value, bool) or not isinstance(call.target.value, (int, float)):
        ctx.error(call, f"Vmuls: value must be a real scalar, got {call.target.value!r}")


def _check_binary_mask(call, ctx, a, b, mask, dst, label):
    if mask.dtype != DType.bool or tuple(mask.shape) != tuple(a.shape):
        ctx.error(call, f"{label}: the mask must be a bool predicate covering the data lanes")
    if b.shape != a.shape or b.dtype != a.dtype or dst.shape != a.shape or dst.dtype != a.dtype:
        ctx.error(call, f"{label}: a, b and dst must have one shape and dtype")


@register_verify_stmt(Vmul)
def _verify_vmul(call, ctx):
    a, b, mask, dst = (ctx.type_of(arg) for arg in call.args)
    _check_binary_mask(call, ctx, a, b, mask, dst, "Vmul")


@register_verify_stmt(Vdiv)
def _verify_vdiv(call, ctx):
    a, b, mask, dst = (ctx.type_of(arg) for arg in call.args)
    _check_binary_mask(call, ctx, a, b, mask, dst, "Vdiv")


@register_verify_stmt(Vsqrt)
def _verify_vsqrt(call, ctx):
    a, mask, dst = (ctx.type_of(arg) for arg in call.args)
    if mask.dtype != DType.bool or tuple(mask.shape) != tuple(a.shape):
        ctx.error(call, "Vsqrt: the mask must be a bool predicate covering the data lanes")
    if dst.shape != a.shape or dst.dtype != a.dtype:
        ctx.error(call, "Vsqrt: dst must match a")


@register_verify_stmt(Vcadd)
def _verify_vcadd(call, ctx):
    src, dst = (ctx.type_of(arg) for arg in call.args)
    lanes = call.target.lanes
    if isinstance(lanes, bool) or not isinstance(lanes, int) or lanes < 2 or lanes > 64 or lanes & (lanes - 1):
        ctx.error(call, f"Vcadd: lanes must be a power of two in [2, 64], got {lanes!r}")
    if dst.shape != src.shape or dst.dtype != src.dtype:
        ctx.error(call, "Vcadd: dst must match src")


@register_verify_stmt(Vdupv)
def _verify_vdupv(call, ctx):
    src, dst = (ctx.type_of(arg) for arg in call.args)
    position = call.target.position
    if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position < 64:
        ctx.error(call, f"Vdupv: position must be an int lane in [0, 64), got {position!r}")
    if dst.shape != src.shape or dst.dtype != src.dtype:
        ctx.error(call, "Vdupv: dst must match src")


@register_verify_stmt(Vadd)
def _verify_vadd(call, ctx):
    a, b, mask, dst = (ctx.type_of(arg) for arg in call.args)
    if tuple(mask.shape) != tuple(a.shape):
        ctx.error(
            call,
            f"Vadd: mask covers {tuple(mask.shape)} lanes but the data has {tuple(a.shape)}",
        )
    if mask.dtype != DType.bool:
        ctx.error(call, f"Vadd: mask must be a predicate (bool), got {mask.dtype.name}")
    if dst.shape != a.shape or dst.dtype != a.dtype:
        ctx.error(call, f"Vadd: dst {dst.shape}/{dst.dtype.name} does not match {a.shape}/{a.dtype.name}")


@register_schedule_eval(GmToUb)
def _eval_gm_to_ub(ctx):
    return TensorValue(data=ctx.args[0].data, type=ctx.result_type)


@register_schedule_eval(UbToGm)
def _eval_ub_to_gm(ctx):
    return TensorValue(data=ctx.args[0].data, type=ctx.result_type)


@register_schedule_eval(Vld)
def _eval_vld(ctx):
    return TensorValue(data=ctx.args[0].data, type=ctx.result_type)


@register_schedule_eval(Vsts)
def _eval_vsts(ctx):
    return TensorValue(data=ctx.args[0].data, type=ctx.result_type)


@register_schedule_eval(Pset)
def _eval_pset(ctx):
    shape = tuple(ctx.result_type.shape)
    return TensorValue(data=torch.ones(shape, dtype=torch.bool), type=ctx.result_type)


@register_schedule_eval(Vcvt)
def _eval_vcvt(ctx):
    """Software model of the declared conversion, with an explicit part mapping.

    The carrier window is 64 words of 32 bits; each word holds two bf16 halves. This
    model carries ONE bf16 value per word -- the half part=0 consumes -- as the logical
    element of a (64,) carrier tensor. The unconsumed half is NOT simulated: its native
    encoding is UNSOURCED. part is checked by _VcvtSignature, so part=1 never reaches
    here. The f32 result is built from the 16 bf16 BITS by a shift into the high half of
    an f32 word, not by an identity copy. Inactive lanes keep what dst held, exactly as
    the predicated vadd does, so the mask is load-bearing here too.
    """
    src, mask, dst = (value.data for value in ctx.args)
    active = mask.to(torch.bool)
    if ctx.op.direction == "forward":
        bits = src.reshape(-1).view(torch.int16).to(torch.int32)
        converted = (bits << 16).view(torch.float32).reshape(src.shape)
    else:
        # Reverse: f32 -> bf16, RNE model (torch CPU), typed as a carrier in the IR.
        # Native packing/rounding is UNSOURCED until 03-20.
        converted = src.to(torch.bfloat16)
    return TensorValue(data=torch.where(active, converted, dst), type=ctx.result_type)


@register_schedule_eval(Pset16)
def _eval_pset16(ctx):
    shape = tuple(ctx.result_type.shape)
    return TensorValue(data=torch.ones(shape, dtype=torch.bool), type=ctx.result_type)


@register_schedule_eval(Vmula)
def _eval_vmula(ctx):
    a, b, mask, acc = (value.data for value in ctx.args)
    active = mask.to(torch.bool)
    # UNFUSED model: multiply rounds, then add rounds. Native FMA fusion ASSUMED-UNKNOWN.
    return TensorValue(data=torch.where(active, acc + a * b, acc), type=ctx.result_type)


@register_schedule_eval(Vdup)
def _eval_vdup(ctx):
    mask, dst = (value.data for value in ctx.args)
    active = mask.to(torch.bool)
    filled = torch.full_like(dst, float(ctx.op.value))
    return TensorValue(data=torch.where(active, filled, dst), type=ctx.result_type)


@register_schedule_eval(Vadds)
def _eval_vadds(ctx):
    a, mask, dst = (value.data for value in ctx.args)
    active = mask.to(torch.bool)
    return TensorValue(data=torch.where(active, a + float(ctx.op.value), dst), type=ctx.result_type)


@register_schedule_eval(Vmuls)
def _eval_vmuls(ctx):
    a, mask, dst = (value.data for value in ctx.args)
    active = mask.to(torch.bool)
    return TensorValue(data=torch.where(active, a * float(ctx.op.value), dst), type=ctx.result_type)


@register_schedule_eval(Vmul)
def _eval_vmul(ctx):
    a, b, mask, dst = (value.data for value in ctx.args)
    active = mask.to(torch.bool)
    return TensorValue(data=torch.where(active, a * b, dst), type=ctx.result_type)


@register_schedule_eval(Vdiv)
def _eval_vdiv(ctx):
    a, b, mask, dst = (value.data for value in ctx.args)
    active = mask.to(torch.bool)
    return TensorValue(data=torch.where(active, a / b, dst), type=ctx.result_type)


@register_schedule_eval(Vsqrt)
def _eval_vsqrt(ctx):
    a, mask, dst = (value.data for value in ctx.args)
    active = mask.to(torch.bool)
    return TensorValue(data=torch.where(active, torch.sqrt(a), dst), type=ctx.result_type)


@register_schedule_eval(Vcadd)
def _eval_vcadd(ctx):
    src, dst = (value.data for value in ctx.args)
    n = int(ctx.op.lanes)
    half = n // 2
    out = dst.clone()
    # Pairwise reduction of the DECLARED width only. No broadcast here.
    out[:half] = src[0:n:2] + src[1:n:2]
    return TensorValue(data=out, type=ctx.result_type)


@register_schedule_eval(Vdupv)
def _eval_vdupv(ctx):
    src, dst = (value.data for value in ctx.args)
    position = int(ctx.op.position)
    return TensorValue(data=torch.full_like(dst, src[position]), type=ctx.result_type)


@register_schedule_eval(Vadd)
def _eval_vadd(ctx):
    a, b, mask, dst = (value.data for value in ctx.args)
    active = mask.to(torch.bool)
    return TensorValue(data=torch.where(active, a + b, dst), type=ctx.result_type)

# --- target-scoped automatic selection for the generic copy / reshard ---
#
# A generic storage-changing Reshard (a copy with no tf.schedule) has to become the
# RIGHT Ascend move for its direction. These four are offered ONLY when the
# Ascend950PR target is the one lowering the program; they are not added to the global
# candidate list, so CUDA keeps its single global Copy carrier and sole_candidate stays
# unambiguous there.
#
# Resolution is by PLACEMENT then declared operand storages:
#   outside a mode  gm->smem  -> ascend_gm_to_ub        smem->gmem -> ascend_ub_to_gm
#   inside a SimdVF smem->rmem -> ascend_vld            rmem->smem -> ascend_vsts
# More than one fit is refused as ambiguous; none fitting is refused with the list. That
# logic lives in tilefoundry.visitor_registry.candidates.select_instruction, shared by
# lowering and reporting, rather than being a late substitution in the converter.
from tilefoundry.ir.hir.sharding.reshard import Reshard as _HirReshard  # noqa: E402
from tilefoundry.visitor_registry.candidates import (  # noqa: E402
    register_target_candidates as _register_target_candidates,
)

_register_target_candidates(
    "ascend950pr",
    _HirReshard,
    (GmToUb, UbToGm, Vld, Vsts),
)