"""Ascend950PR TIR instruction declarations and semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import isl

from tilefoundry.evaluator.registry import register_schedule_eval
from tilefoundry.evaluator.value import TensorValue, to_torch_dtype
from tilefoundry.ir.core import Call, Op, OpCapability
from tilefoundry.ir.core.execution import RegionRequirement
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import (
    ComposedLayoutPattern,
    DistinctConstraint,
    LayoutPattern,
    MeshPattern,
    SameConstraint,
    StarPattern,
    TensorPattern,
    WildcardPattern,
    utils,
)
from tilefoundry.ir.pattern import predicates as P
from tilefoundry.ir.tir.memory.copy import register_copy_semantics
from tilefoundry.ir.types import DType, StorageKind, UnitType
from tilefoundry.utils.lazy import torch
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    broadcast_shapes,
    identity_access,
    identity_relations,
    iterating,
    matmul_relations,
    register_access_relation,
)

_EXEC_LAYOUT = ComposedLayoutPattern(
    inner=None,
    offset=WildcardPattern("p0"),
    outer=LayoutPattern(
        (StarPattern(WildcardPattern("execution_shape")),),
        (StarPattern(WildcardPattern("execution_strides")),),
    ),
)
_MEM_MESH = utils.memory_execution_mesh()
_AIC_MESH = MeshPattern(("aic",), _EXEC_LAYOUT)
_AIV_MESH = MeshPattern(("aiv",), _EXEC_LAYOUT)

_GMEM = utils.tensor_in(StorageKind.GMEM)
_SMEM = utils.tensor_in(StorageKind.SMEM)
_TMEM = utils.tensor_in(StorageKind.TMEM)
_RMEM_AIV = utils.tensor_in(StorageKind.RMEM, _AIV_MESH)

_DUAL_STORAGES = (StorageKind.SMEM, StorageKind.TMEM)
_DUAL_SRC = TensorPattern(
    storage=WildcardPattern(
        "dual_src_storage",
        predicates=(P.In(WildcardPattern("dual_src_storage"), _DUAL_STORAGES),),
    )
)
_DUAL_DST = TensorPattern(
    storage=WildcardPattern(
        "dual_dst_storage",
        predicates=(P.In(WildcardPattern("dual_dst_storage"), _DUAL_STORAGES),),
    )
)

_OUTSIDE_VF = RegionRequirement("ascend.simdvf", "outside")
_INSIDE_VF = RegionRequirement("ascend.simdvf", "inside")

_SAME_SHAPE_DTYPE = (
    SameConstraint("shape", "src", "dst"),
    SameConstraint("dtype", "src", "dst"),
)


@dataclass(frozen=True)
class AscendOpSpec:
    """Metadata describing an Ascend950PR instruction for schedule and pipeline tools."""

    op_cls: type[Op]
    name: str
    resource: str
    capability: str
    placement: str


def _register_binary_semantics(op_cls: type[Op], torch_fn) -> type[Op]:
    """Attach unit typeinfer, identity access, eval, and broadcast verify to binary op."""

    @register_typeinfer(op_cls)
    def _infer(call: Call, ctx) -> UnitType:
        return UnitType()

    register_access_relation(op_cls)(identity_relations)

    @register_schedule_eval(op_cls)
    def _eval(ctx):
        return TensorValue(
            data=torch_fn(ctx.args[0].data, ctx.args[1].data),
            type=ctx.result_type,
        )

    @register_verify_stmt(op_cls)
    def _verify(call: Call, ctx) -> None:
        lty, rty, dty = (ctx.type_of(arg) for arg in call.args)
        shape = broadcast_shapes(lty.shape, rty.shape, raising=False)
        if shape != tuple(dty.shape):
            ctx.error(
                call,
                f"{op_cls.__name__}: dst {dty.shape} != broadcast({lty.shape}, {rty.shape})",
            )

    return op_cls


def _register_unary_semantics(op_cls: type[Op], torch_fn) -> type[Op]:
    """Attach unit typeinfer, identity access, eval, and shape verify to unary op."""

    @register_typeinfer(op_cls)
    def _infer(call: Call, ctx) -> UnitType:
        return UnitType()

    register_access_relation(op_cls)(identity_relations)

    @register_schedule_eval(op_cls)
    def _eval(ctx):
        return TensorValue(data=torch_fn(ctx.args[0].data), type=ctx.result_type)

    @register_verify_stmt(op_cls)
    def _verify(call: Call, ctx) -> None:
        sty, dty = (ctx.type_of(arg) for arg in call.args)
        if tuple(sty.shape) != tuple(dty.shape):
            ctx.error(call, f"{op_cls.__name__}: shape mismatch {sty.shape} vs {dty.shape}")

    return op_cls


def _reduce_axes(op, rank: int) -> tuple[int, ...]:
    axes = tuple(axis + rank if axis < 0 else axis for axis in op.axes)
    if any(axis < 0 or axis >= rank for axis in axes):
        raise ValueError(f"{type(op).__name__} axes {op.axes} exceed source rank {rank}")
    return axes


def _register_reduce_semantics(op_cls: type[Op], reduce_fn) -> type[Op]:
    """Attach unit typeinfer, collapse access, eval, and shape verify to reduce op."""

    @register_typeinfer(op_cls)
    def _infer(call: Call, ctx) -> UnitType:
        return UnitType()

    @register_access_relation(op_cls)
    def _access(call: Call, ctx) -> tuple[AccessRelation, ...]:
        op = call.target
        source = ctx.type_of(call.args[0])
        rank = len(source.shape)
        axes = _reduce_axes(op, rank)
        out_shape = tuple(
            1 if axis in axes else extent
            for axis, extent in enumerate(source.shape)
            if op.keepdim or axis not in axes
        )
        surviving = [axis for axis in range(rank) if axis not in axes]
        came_from = (
            {axis: axis for axis in range(rank)}
            if op.keepdim
            else dict(enumerate(surviving))
        )
        writes_at = [
            "0" if axis not in came_from or came_from[axis] in axes else f"d{came_from[axis]}"
            for axis in range(len(out_shape))
        ]
        domain = ", ".join(f"d{axis}" for axis in range(rank))
        dst_access = AccessRelation(isl.map(f"{{ [{domain}] -> [{', '.join(writes_at)}] }}"))
        return iterating(source.shape, (identity_access(rank), dst_access, dst_access))

    @register_schedule_eval(op_cls)
    def _eval(ctx):
        src = ctx.args[0].data
        axes = _reduce_axes(ctx.op, src.ndim)
        out = reduce_fn(src, dim=axes, keepdim=ctx.op.keepdim)
        return TensorValue(data=out, type=ctx.result_type)

    return op_cls


@register_copy_semantics
@register_op(dialect="T", category="ascend", name="ascend_gm_to_ub")
class AscendGmToUb(Op):
    """Copy a tile from global HBM to AIV Unified Buffer."""

    capability = OpCapability("ascend.dma")
    resource = "mte2_engine"
    execution_mesh = _MEM_MESH
    execution_region = _OUTSIDE_VF
    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_GMEM)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_SMEM)
    between = _SAME_SHAPE_DTYPE


@register_copy_semantics
@register_op(dialect="T", category="ascend", name="ascend_ub_to_gm")
class AscendUbToGm(Op):
    """Copy a tile from AIV Unified Buffer to global HBM."""

    capability = OpCapability("ascend.dma")
    resource = "mte3_engine"
    execution_mesh = _MEM_MESH
    execution_region = _OUTSIDE_VF
    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_SMEM)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_GMEM)
    between = _SAME_SHAPE_DTYPE


@register_copy_semantics
@register_op(dialect="T", category="ascend", name="ascend_gm_to_l1")
class AscendGmToL1(Op):
    """Copy a tile from global HBM to AIC L1 buffer."""

    capability = OpCapability("ascend.dma")
    resource = "mte2_engine"
    execution_mesh = _MEM_MESH
    execution_region = _OUTSIDE_VF
    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_GMEM)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_TMEM)
    between = _SAME_SHAPE_DTYPE


@register_copy_semantics
@register_op(dialect="T", category="ascend", name="ascend_l1_to_l0a")
class AscendL1ToL0A(Op):
    """Load an L1 matrix tile into Cube L0A."""

    capability = OpCapability("ascend.dma")
    resource = "mte1_engine"
    execution_mesh = _MEM_MESH
    execution_region = _OUTSIDE_VF
    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_TMEM)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_TMEM)
    between = _SAME_SHAPE_DTYPE


@register_copy_semantics
@register_op(dialect="T", category="ascend", name="ascend_l1_to_l0b")
class AscendL1ToL0B(Op):
    """Load an L1 matrix tile into Cube L0B."""

    capability = OpCapability("ascend.dma")
    resource = "mte1_engine"
    execution_mesh = _MEM_MESH
    execution_region = _OUTSIDE_VF
    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_TMEM)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_TMEM)
    between = _SAME_SHAPE_DTYPE


@register_copy_semantics
@register_op(dialect="T", category="ascend", name="ascend_l0c_to_gm")
class AscendL0cToGm(Op):
    """Fixpipe store from Cube L0C accumulator to global HBM."""

    capability = OpCapability("ascend.dma")
    resource = "fixpipe_engine"
    execution_mesh = _MEM_MESH
    execution_region = _OUTSIDE_VF
    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_TMEM)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_GMEM)
    between = _SAME_SHAPE_DTYPE


@register_copy_semantics
@register_op(dialect="T", category="ascend", name="ascend_dual_copy")
class AscendDualCopy(Op):
    """Cross-core AIC<->AIV tensor exchange between TMEM and UB."""

    capability = OpCapability("ascend.dual_copy")
    resource = "dual_copy_engine"
    execution_mesh = _MEM_MESH
    execution_region = _OUTSIDE_VF
    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_DUAL_SRC)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_DUAL_DST)
    between = (*_SAME_SHAPE_DTYPE, DistinctConstraint("storage", "src", "dst"))


@register_copy_semantics
@register_op(dialect="T", category="ascend", name="ascend_ub_to_reg")
class AscendUbToReg(Op):
    """Load a vector tile from AIV UB into vector registers inside SimdVF."""

    capability = OpCapability("ascend.vector")
    resource = "vector_engine"
    execution_mesh = _AIV_MESH
    execution_region = _INSIDE_VF
    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_SMEM)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_RMEM_AIV)
    between = _SAME_SHAPE_DTYPE


@register_copy_semantics
@register_op(dialect="T", category="ascend", name="ascend_reg_to_ub")
class AscendRegToUb(Op):
    """Store a vector register tile back to AIV UB inside SimdVF."""

    capability = OpCapability("ascend.vector")
    resource = "vector_engine"
    execution_mesh = _AIV_MESH
    execution_region = _INSIDE_VF
    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_RMEM_AIV)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_SMEM)
    between = _SAME_SHAPE_DTYPE


@register_op(dialect="T", category="ascend", name="ascend_mmad")
class AscendMmad(Op):
    """Execute one Ascend Cube matrix multiply-accumulate on L0A/L0B -> L0C."""

    capability = OpCapability("ascend.mmad")
    resource = "cube_engine"
    execution_mesh = _AIC_MESH
    execution_region = _OUTSIDE_VF

    acc = ParamDef(kind="input", effect=MemoryEffect.READ | MemoryEffect.WRITE, pattern=_TMEM)
    lhs = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_TMEM)
    rhs = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_TMEM)
    a_layout = ParamDef(kind="attribute", annotation=Literal["MK", "KM"], default="MK")
    b_layout = ParamDef(kind="attribute", annotation=Literal["NK", "KN"], default="KN")
    out_dtype = ParamDef(kind="attribute", annotation=DType, optional=True, default=None)
    init_zero = ParamDef(kind="attribute", annotation=bool, default=False)

    def __init__(self, **attrs) -> None:
        out_dtype = attrs.get("out_dtype")
        if isinstance(out_dtype, str):
            attrs["out_dtype"] = DType.from_name(out_dtype)
        super().__init__(**attrs)


@register_typeinfer(AscendMmad)
def _infer_mmad(call: Call, ctx) -> UnitType:
    return UnitType()


@register_access_relation(AscendMmad)
def _access_mmad(call: Call, ctx) -> tuple[AccessRelation, ...]:
    op = call.target
    lhs = ctx.type_of(call.args[1])
    rhs = ctx.type_of(call.args[2])
    a_axes = (-2, -1) if op.a_layout == "MK" else (-1, -2)
    b_axes = (-2, -1) if op.b_layout == "NK" else (-1, -2)
    a_rel, b_rel, acc_rel = matmul_relations(lhs.shape, rhs.shape, (*a_axes, *b_axes))
    return acc_rel, a_rel, b_rel, acc_rel


@register_schedule_eval(AscendMmad)
def _eval_mmad(ctx):
    acc, lhs, rhs = (arg.data for arg in ctx.args)
    lhs_m = lhs if ctx.op.a_layout == "MK" else lhs.transpose(-2, -1)
    rhs_m = rhs if ctx.op.b_layout == "KN" else rhs.transpose(-2, -1)
    prod = torch.matmul(lhs_m.to(torch.float32), rhs_m.to(torch.float32))
    base = torch.zeros_like(prod) if ctx.op.init_zero else acc.to(torch.float32)
    return TensorValue(data=(base + prod).to(acc.dtype), type=ctx.result_type)


class _VectorBinaryBase(Op):
    """Shared structure for elementwise AIV binary vector instructions."""

    capability = OpCapability("ascend.vector")
    resource = "vector_engine"
    execution_mesh = _AIV_MESH
    execution_region = _INSIDE_VF

    lhs = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_RMEM_AIV)
    rhs = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_RMEM_AIV)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_RMEM_AIV)
    between = (SameConstraint("dtype", "lhs", "dst"), SameConstraint("dtype", "rhs", "dst"))


@register_op(dialect="T", category="ascend", name="ascend_vadd")
class AscendVadd(_VectorBinaryBase):
    """AIV vector addition inside SimdVF."""

    capability = OpCapability("ascend.vector")


@register_op(dialect="T", category="ascend", name="ascend_vsub")
class AscendVsub(_VectorBinaryBase):
    """AIV vector subtraction inside SimdVF."""

    capability = OpCapability("ascend.vector")


@register_op(dialect="T", category="ascend", name="ascend_vmul")
class AscendVmul(_VectorBinaryBase):
    """AIV vector multiplication inside SimdVF."""

    capability = OpCapability("ascend.vector")


@register_op(dialect="T", category="ascend", name="ascend_vdiv")
class AscendVdiv(_VectorBinaryBase):
    """AIV vector division inside SimdVF."""

    capability = OpCapability("ascend.vector")


@register_op(dialect="T", category="ascend", name="ascend_vmax")
class AscendVmax(_VectorBinaryBase):
    """AIV vector elementwise maximum inside SimdVF."""

    capability = OpCapability("ascend.vector")


_register_binary_semantics(AscendVadd, torch.add)
_register_binary_semantics(AscendVsub, torch.sub)
_register_binary_semantics(AscendVmul, torch.mul)
_register_binary_semantics(AscendVdiv, torch.div)
_register_binary_semantics(AscendVmax, torch.maximum)


class _VectorUnaryBase(Op):
    """Shared structure for elementwise AIV unary vector instructions."""

    capability = OpCapability("ascend.vector")
    resource = "vector_engine"
    execution_mesh = _AIV_MESH
    execution_region = _INSIDE_VF

    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_RMEM_AIV)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_RMEM_AIV)
    between = _SAME_SHAPE_DTYPE


@register_op(dialect="T", category="ascend", name="ascend_vexp")
class AscendVexp(_VectorUnaryBase):
    """AIV vector exponential inside SimdVF."""

    capability = OpCapability("ascend.vector")


@register_op(dialect="T", category="ascend", name="ascend_vrsqrt")
class AscendVrsqrt(_VectorUnaryBase):
    """AIV vector reciprocal square root inside SimdVF."""

    capability = OpCapability("ascend.vector")


_register_unary_semantics(AscendVexp, torch.exp)
_register_unary_semantics(AscendVrsqrt, torch.rsqrt)


@register_op(dialect="T", category="ascend", name="ascend_vcast")
class AscendVcast(Op):
    """AIV vector register dtype conversion inside SimdVF."""

    capability = OpCapability("ascend.vector")
    resource = "vector_engine"
    execution_mesh = _AIV_MESH
    execution_region = _INSIDE_VF

    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_RMEM_AIV)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_RMEM_AIV)
    dtype = ParamDef(kind="attribute", annotation=DType)
    between = (SameConstraint("shape", "src", "dst"), DistinctConstraint("dtype", "src", "dst"))

    def __init__(self, **attrs) -> None:
        dtype = attrs.get("dtype")
        if isinstance(dtype, str):
            attrs["dtype"] = DType.from_name(dtype)
        super().__init__(**attrs)


@register_typeinfer(AscendVcast)
def _infer_vcast(call: Call, ctx) -> UnitType:
    return UnitType()


register_access_relation(AscendVcast)(identity_relations)


@register_schedule_eval(AscendVcast)
def _eval_vcast(ctx):
    return TensorValue(
        data=ctx.args[0].data.to(to_torch_dtype(ctx.op.dtype)),
        type=ctx.result_type,
    )


@register_verify_stmt(AscendVcast)
def _verify_vcast(call: Call, ctx) -> None:
    dty = ctx.type_of(call.args[1])
    if dty.dtype != call.target.dtype:
        ctx.error(call, f"AscendVcast: dst dtype {dty.dtype} != {call.target.dtype}")


class _VectorReduceBase(Op):
    """Shared structure for AIV vector reductions inside SimdVF."""

    capability = OpCapability("ascend.vector")
    resource = "vector_engine"
    execution_mesh = _AIV_MESH
    execution_region = _INSIDE_VF

    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_RMEM_AIV)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_RMEM_AIV)
    axes = ParamDef(kind="attribute", annotation=tuple)
    keepdim = ParamDef(kind="attribute", annotation=bool, default=True)


@register_op(dialect="T", category="ascend", name="ascend_vreduce_sum")
class AscendVreduceSum(_VectorReduceBase):
    """AIV vector sum reduction inside SimdVF."""

    capability = OpCapability("ascend.vector")


@register_op(dialect="T", category="ascend", name="ascend_vreduce_max")
class AscendVreduceMax(_VectorReduceBase):
    """AIV vector max reduction inside SimdVF."""

    capability = OpCapability("ascend.vector")


_register_reduce_semantics(AscendVreduceSum, torch.sum)
_register_reduce_semantics(AscendVreduceMax, torch.amax)

ASCEND_OP_SPECS: tuple[AscendOpSpec, ...] = tuple(
    AscendOpSpec(
        op_cls=cls,
        name=cls._op_schema.name,
        resource=cls.resource,
        capability=cls.capability.name,
        placement=cls.execution_region.placement,
    )
    for cls in (
        AscendGmToUb,
        AscendUbToGm,
        AscendGmToL1,
        AscendL1ToL0A,
        AscendL1ToL0B,
        AscendL0cToGm,
        AscendDualCopy,
        AscendUbToReg,
        AscendRegToUb,
        AscendMmad,
        AscendVadd,
        AscendVsub,
        AscendVmul,
        AscendVdiv,
        AscendVmax,
        AscendVexp,
        AscendVrsqrt,
        AscendVcast,
        AscendVreduceSum,
        AscendVreduceMax,
    )
)

__all__ = [
    "ASCEND_OP_SPECS",
    "AscendDualCopy",
    "AscendGmToL1",
    "AscendGmToUb",
    "AscendL0cToGm",
    "AscendL1ToL0A",
    "AscendL1ToL0B",
    "AscendMmad",
    "AscendOpSpec",
    "AscendRegToUb",
    "AscendUbToGm",
    "AscendUbToReg",
    "AscendVadd",
    "AscendVcast",
    "AscendVdiv",
    "AscendVexp",
    "AscendVmax",
    "AscendVmul",
    "AscendVreduceMax",
    "AscendVreduceSum",
    "AscendVrsqrt",
    "AscendVsub",
]
