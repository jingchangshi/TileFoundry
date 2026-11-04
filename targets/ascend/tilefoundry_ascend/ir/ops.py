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
from tilefoundry.ir.pattern import SameConstraint, TensorPattern, utils
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
_IN_GMEM = utils.tensor_in(StorageKind.GMEM)
_IN_UB = utils.tensor_in(StorageKind.SMEM)
_IN_VREG = utils.tensor_in(StorageKind.RMEM)
_IN_PREG = TensorPattern(dtype=DType.bool, storage=StorageKind.RMEM)

INSIDE = RegionRequirement(kind=SIMD_VF_KIND, placement="inside")
OUTSIDE = RegionRequirement(kind=SIMD_VF_KIND, placement="outside")


@register_op(dialect="T", category="ascend", name="ascend_gm_to_ub")
class GmToUb(Op):
    """GM->UB tile load (MTE2): a memory instruction, so outside the mode."""

    capability = OpCapability(None)
    execution_mesh = _AIV
    execution_region = OUTSIDE

    src = ParamDef(kind="input", effect=READ, pattern=_IN_GMEM)
    dst = ParamDef(kind="input", effect=WRITE, pattern=_IN_UB)

    between = (SameConstraint("shape", "src", "dst"), SameConstraint("dtype", "src", "dst"))


@register_op(dialect="T", category="ascend", name="ascend_ub_to_gm")
class UbToGm(Op):
    """UB->GM tile store (MTE3): likewise outside the mode."""

    capability = OpCapability(None)
    execution_mesh = _AIV
    execution_region = OUTSIDE

    src = ParamDef(kind="input", effect=READ, pattern=_IN_UB)
    dst = ParamDef(kind="input", effect=WRITE, pattern=_IN_GMEM)

    between = (SameConstraint("shape", "src", "dst"), SameConstraint("dtype", "src", "dst"))


@register_op(dialect="T", category="ascend", name="ascend_vld")
class Vld(Op):
    """UB->VREG lane load, instruction ``vld``."""

    capability = OpCapability(None)
    execution_mesh = _AIV
    execution_region = INSIDE

    src = ParamDef(kind="input", effect=READ, pattern=_IN_UB)
    dst = ParamDef(kind="input", effect=WRITE, pattern=_IN_VREG)

    between = (SameConstraint("shape", "src", "dst"), SameConstraint("dtype", "src", "dst"))


@register_op(dialect="T", category="ascend", name="ascend_vsts")
class Vsts(Op):
    """VREG->UB lane store, instruction ``vsts``."""

    capability = OpCapability(None)
    execution_mesh = _AIV
    execution_region = INSIDE

    src = ParamDef(kind="input", effect=READ, pattern=_IN_VREG)
    dst = ParamDef(kind="input", effect=WRITE, pattern=_IN_UB)

    between = (SameConstraint("shape", "src", "dst"), SameConstraint("dtype", "src", "dst"))


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


for _op in (GmToUb, UbToGm, Vld, Vsts, Pset, Vadd):
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


@register_schedule_eval(Vadd)
def _eval_vadd(ctx):
    a, b, mask, dst = (value.data for value in ctx.args)
    active = mask.to(torch.bool)
    return TensorValue(data=torch.where(active, a + b, dst), type=ctx.result_type)