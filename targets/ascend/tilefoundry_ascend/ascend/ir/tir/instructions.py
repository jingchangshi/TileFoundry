"""Ascend950PR TIR instruction declarations and semantics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import isl

from tilefoundry.evaluator.kernels import binary as eval_binary_kernel
from tilefoundry.evaluator.registry import register_schedule_eval
from tilefoundry.evaluator.value import TensorValue, to_torch_dtype
from tilefoundry.ir.core import Call, Op, OpCapability
from tilefoundry.ir.core.execution import RegionRequirement
from tilefoundry.ir.core.kinds import BinaryKind, ReduceKind, UnaryKind
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
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.utils.lazy import torch
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    broadcast_access,
    broadcast_all,
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
_BOOL_RMEM_AIV = replace(_RMEM_AIV, dtype=DType.bool)

def _cand_storage(name: str, base: TensorPattern, storage: StorageKind) -> TensorPattern:
    """Wrap *storage* in a constrained wildcard for candidate-only Reshard matching."""
    return replace(
        base,
        storage=WildcardPattern(
            name,
            predicates=(P.In(WildcardPattern(name), (storage,)),),
        ),
    )


_GMEM_CAND = _cand_storage("cand_gmem_storage", _GMEM, StorageKind.GMEM)
_SMEM_CAND = _cand_storage("cand_smem_storage", _SMEM, StorageKind.SMEM)
_TMEM_CAND = _cand_storage("cand_tmem_storage", _TMEM, StorageKind.TMEM)
_RMEM_AIV_CAND = _cand_storage("cand_rmem_storage", _RMEM_AIV, StorageKind.RMEM)

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


def _register_unary_semantics(
    op_cls: type[Op], torch_fn=None, *, register_eval: bool = True
) -> type[Op]:
    """Attach unit typeinfer, identity access, eval, and shape verify to unary op."""

    @register_typeinfer(op_cls)
    def _infer(call: Call, ctx) -> UnitType:
        return UnitType()

    register_access_relation(op_cls)(identity_relations)

    if register_eval:
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


def _register_reduce_semantics(
    op_cls: type[Op], reduce_fn=None, *, register_eval: bool = True
) -> type[Op]:
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

    if register_eval:
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

_CMP_FNS = {
    "eq": torch.eq,
    "ne": torch.ne,
    "lt": torch.lt,
    "le": torch.le,
    "gt": torch.gt,
    "ge": torch.ge,
}


def _reduce_dispatch(src, axes: tuple[int, ...], keepdim: bool, kind: ReduceKind):
    if kind is ReduceKind.SUM:
        return src.sum(dim=axes, keepdim=keepdim)
    if kind is ReduceKind.MEAN:
        return src.mean(dim=axes, keepdim=keepdim)
    if kind is ReduceKind.MAX:
        return src.amax(dim=axes, keepdim=keepdim)
    if kind is ReduceKind.MIN:
        return src.amin(dim=axes, keepdim=keepdim)
    if kind is ReduceKind.ABS_MAX:
        return src.abs().amax(dim=axes, keepdim=keepdim)
    raise ValueError(f"AscendVreduce: unsupported ReduceKind {kind!r}")


def _unary_dispatch(kind: UnaryKind, src):
    fns = {
        UnaryKind.NEG: torch.neg,
        UnaryKind.ABS: torch.abs,
        UnaryKind.RSQRT: torch.rsqrt,
        UnaryKind.NOT: torch.logical_not,
        UnaryKind.RELU: torch.relu,
        UnaryKind.SQUARE: torch.square,
        UnaryKind.EXP: torch.exp,
        UnaryKind.LOG: torch.log,
        UnaryKind.CEIL: torch.ceil,
        UnaryKind.ROUND: torch.round,
        UnaryKind.EXP2: torch.exp2,
        UnaryKind.LOG2: torch.log2,
    }
    if kind not in fns:
        raise ValueError(f"AscendVunary: unsupported UnaryKind {kind!r}")
    return fns[kind](src)


def _register_broadcast_binary_semantics(op_cls: type[Op], eval_fn) -> type[Op]:
    """Attach unit typeinfer, broadcast access, eval, and shape verify to binary op."""

    @register_typeinfer(op_cls)
    def _infer(call: Call, ctx) -> UnitType:
        return UnitType()

    @register_access_relation(op_cls)
    def _access(call: Call, ctx) -> tuple[AccessRelation, ...]:
        shapes = tuple(ctx.type_of(arg).shape for arg in call.args[:2])
        out_shape = broadcast_all(shapes)
        out = identity_access(len(out_shape))
        return iterating(
            out_shape,
            (*(broadcast_access(out_shape, shape) for shape in shapes), out, out),
        )

    @register_schedule_eval(op_cls)
    def _eval(ctx):
        return TensorValue(data=eval_fn(ctx), type=ctx.result_type)

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


def _register_dst_only_semantics(op_cls: type[Op], eval_fn) -> type[Op]:
    """Attach unit typeinfer, identity write access, and eval for dst-only ops."""

    @register_typeinfer(op_cls)
    def _infer(call: Call, ctx) -> UnitType:
        return UnitType()

    @register_access_relation(op_cls)
    def _access(call: Call, ctx) -> tuple[AccessRelation, ...]:
        dst = ctx.type_of(call.args[0])
        rel = identity_access(len(dst.shape))
        return iterating(dst.shape, (rel, rel))

    @register_schedule_eval(op_cls)
    def _eval(ctx):
        return TensorValue(data=eval_fn(ctx), type=ctx.result_type)

    return op_cls


@register_op(dialect="T", category="ascend", name="ascend_vbinary")
class AscendVbinary(Op):
    """Kind-tagged AIV vector binary instruction matching HIR Binary inside SimdVF."""

    capability = OpCapability("ascend.vector")
    resource = "vector_engine"
    execution_mesh = _AIV_MESH
    execution_region = _INSIDE_VF
    mask_width_attribute = "mask_width"

    lhs = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_RMEM_AIV)
    rhs = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_RMEM_AIV)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_RMEM_AIV)
    kind = ParamDef(kind="attribute", annotation=BinaryKind)
    mask_width = ParamDef(kind="attribute", annotation=int, default=32)
    between = (SameConstraint("dtype", "lhs", "rhs"),)


_register_broadcast_binary_semantics(
    AscendVbinary,
    lambda ctx: eval_binary_kernel(ctx.op.kind, ctx.args[0].data, ctx.args[1].data),
)


@register_op(dialect="T", category="ascend", name="ascend_vunary")
class AscendVunary(_VectorUnaryBase):
    """Kind-tagged AIV vector unary instruction matching HIR Unary inside SimdVF."""

    capability = OpCapability("ascend.vector")
    mask_width_attribute = "mask_width"

    kind = ParamDef(kind="attribute", annotation=UnaryKind)
    mask_width = ParamDef(kind="attribute", annotation=int, default=32)


_register_unary_semantics(AscendVunary, register_eval=False)


@register_schedule_eval(AscendVunary)
def _eval_vunary(ctx):
    return TensorValue(
        data=_unary_dispatch(ctx.op.kind, ctx.args[0].data),
        type=ctx.result_type,
    )


@register_op(dialect="T", category="ascend", name="ascend_vreduce")
class AscendVreduce(_VectorReduceBase):
    """Kind-tagged AIV vector reduction instruction matching HIR Reduce inside SimdVF."""

    capability = OpCapability("ascend.vector")
    keepdim = ParamDef(kind="attribute", annotation=bool, default=False)
    kind = ParamDef(kind="attribute", annotation=ReduceKind, default=ReduceKind.SUM)
    between = (SameConstraint("dtype", "src", "dst"),)


_register_reduce_semantics(AscendVreduce, register_eval=False)


@register_schedule_eval(AscendVreduce)
def _eval_vreduce(ctx):
    src = ctx.args[0].data
    axes = _reduce_axes(ctx.op, src.ndim)
    return TensorValue(
        data=_reduce_dispatch(src, axes, ctx.op.keepdim, ctx.op.kind),
        type=ctx.result_type,
    )


@register_op(dialect="T", category="ascend", name="ascend_vwhere")
class AscendVwhere(Op):
    """Predicated AIV elementwise selection inside SimdVF (dst = where(cond, lhs, rhs))."""

    capability = OpCapability("ascend.vector")
    resource = "vector_engine"
    execution_mesh = _AIV_MESH
    execution_region = _INSIDE_VF

    cond = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_BOOL_RMEM_AIV)
    lhs = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_RMEM_AIV)
    rhs = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_RMEM_AIV)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_RMEM_AIV)
    between = (SameConstraint("dtype", "lhs", "rhs"), SameConstraint("dtype", "lhs", "dst"))


def _register_where_semantics(op_cls: type[Op]) -> None:
    @register_typeinfer(op_cls)
    def _infer(call: Call, ctx) -> UnitType:
        return UnitType()

    @register_access_relation(op_cls)
    def _access(call: Call, ctx) -> tuple[AccessRelation, ...]:
        shapes = tuple(ctx.type_of(arg).shape for arg in call.args[:3])
        out_shape = broadcast_all(shapes)
        out = identity_access(len(out_shape))
        return iterating(
            out_shape,
            (*(broadcast_access(out_shape, shape) for shape in shapes), out, out),
        )

    @register_schedule_eval(op_cls)
    def _eval(ctx):
        cond, lhs, rhs = ctx.args[0].data, ctx.args[1].data, ctx.args[2].data
        return TensorValue(data=torch.where(cond, lhs, rhs), type=ctx.result_type)

    @register_verify_stmt(op_cls)
    def _verify(call: Call, ctx) -> None:
        cond, lhs, rhs, dst = (ctx.type_of(arg) for arg in call.args)
        if broadcast_all((cond.shape, lhs.shape, rhs.shape)) != tuple(dst.shape):
            ctx.error(call, f"{op_cls.__name__}: dst shape != broadcast of inputs")


_register_where_semantics(AscendVwhere)


@register_op(dialect="T", category="ascend", name="ascend_vsel")
class AscendVsel(AscendVwhere):
    """Explicit Ascend SimdVF vsel instruction selecting lanes under predicate cond."""

    capability = OpCapability("ascend.vector")


_register_where_semantics(AscendVsel)


@register_op(dialect="T", category="ascend", name="ascend_vclamp")
class AscendVclamp(_VectorUnaryBase):
    """AIV vector elementwise clamp inside SimdVF."""

    capability = OpCapability("ascend.vector")
    min_val = ParamDef(kind="attribute", annotation=float)
    max_val = ParamDef(kind="attribute", annotation=float)


_register_unary_semantics(AscendVclamp, register_eval=False)


@register_schedule_eval(AscendVclamp)
def _eval_vclamp(ctx):
    return TensorValue(
        data=torch.clamp(ctx.args[0].data, min=ctx.op.min_val, max=ctx.op.max_val),
        type=ctx.result_type,
    )


def _eval_pset_data(ctx):
    dst_ty = ctx.result_type
    shape = tuple(int(d) for d in dst_ty.shape)
    pat = ctx.op.pat
    if pat == "PAT_NONE":
        return torch.zeros(shape, dtype=torch.bool)
    if pat == "PAT_HALF" and shape and shape[-1] > 1:
        idx = torch.arange(shape[-1]) < (shape[-1] // 2)
        return idx.expand(shape).clone()
    return torch.ones(shape, dtype=torch.bool)


@register_op(dialect="T", category="ascend", name="ascend_pset")
class AscendPset(Op):
    """Initialize an AIV vector predicate register at declared element width."""

    capability = OpCapability("ascend.vector")
    resource = "vector_engine"
    execution_mesh = _AIV_MESH
    execution_region = _INSIDE_VF
    mask_width_attribute = "elem_width"

    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_BOOL_RMEM_AIV)
    elem_width = ParamDef(kind="attribute", annotation=int, default=32)
    pat = ParamDef(kind="attribute", annotation=str, default="PAT_ALL")


_register_dst_only_semantics(AscendPset, _eval_pset_data)


def _eval_update_mask_data(ctx):
    shape = tuple(int(d) for d in ctx.result_type.shape)
    if not shape:
        return torch.tensor(ctx.op.count > 0, dtype=torch.bool)
    idx = torch.arange(shape[-1]) < ctx.op.count
    return idx.expand(shape).clone()


@register_op(dialect="T", category="ascend", name="ascend_update_mask")
class AscendUpdateMask(Op):
    """Tail-safe active-lane predicate generator (lanes < count set to True)."""

    capability = OpCapability("ascend.vector")
    resource = "vector_engine"
    execution_mesh = _AIV_MESH
    execution_region = _INSIDE_VF
    mask_width_attribute = "elem_width"

    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_BOOL_RMEM_AIV)
    count = ParamDef(kind="attribute", annotation=int)
    elem_width = ParamDef(kind="attribute", annotation=int, default=32)


_register_dst_only_semantics(AscendUpdateMask, _eval_update_mask_data)


class _PredicateBinaryBase(Op):
    """Shared structure for AIV predicate bitwise/logical combine instructions."""

    capability = OpCapability("ascend.vector")
    resource = "vector_engine"
    execution_mesh = _AIV_MESH
    execution_region = _INSIDE_VF
    mask_width_attribute = "elem_width"

    lhs = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_BOOL_RMEM_AIV)
    rhs = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_BOOL_RMEM_AIV)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_BOOL_RMEM_AIV)
    elem_width = ParamDef(kind="attribute", annotation=int, default=32)


@register_op(dialect="T", category="ascend", name="ascend_pand")
class AscendPand(_PredicateBinaryBase):
    """AIV predicate logical AND inside SimdVF."""

    capability = OpCapability("ascend.vector")


@register_op(dialect="T", category="ascend", name="ascend_por")
class AscendPor(_PredicateBinaryBase):
    """AIV predicate logical OR inside SimdVF."""

    capability = OpCapability("ascend.vector")


@register_op(dialect="T", category="ascend", name="ascend_pxor")
class AscendPxor(_PredicateBinaryBase):
    """AIV predicate logical XOR inside SimdVF."""

    capability = OpCapability("ascend.vector")


_register_broadcast_binary_semantics(
    AscendPand, lambda ctx: torch.logical_and(ctx.args[0].data, ctx.args[1].data)
)
_register_broadcast_binary_semantics(
    AscendPor, lambda ctx: torch.logical_or(ctx.args[0].data, ctx.args[1].data)
)
_register_broadcast_binary_semantics(
    AscendPxor, lambda ctx: torch.logical_xor(ctx.args[0].data, ctx.args[1].data)
)


@register_op(dialect="T", category="ascend", name="ascend_pnot")
class AscendPnot(Op):
    """AIV predicate logical NOT inside SimdVF."""

    capability = OpCapability("ascend.vector")
    resource = "vector_engine"
    execution_mesh = _AIV_MESH
    execution_region = _INSIDE_VF
    mask_width_attribute = "elem_width"

    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_BOOL_RMEM_AIV)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_BOOL_RMEM_AIV)
    elem_width = ParamDef(kind="attribute", annotation=int, default=32)


_register_unary_semantics(AscendPnot, torch.logical_not)


@register_op(dialect="T", category="ascend", name="ascend_vcmp")
class AscendVcmp(Op):
    """AIV vector-vector comparison producing a bool predicate register."""

    capability = OpCapability("ascend.vector")
    resource = "vector_engine"
    execution_mesh = _AIV_MESH
    execution_region = _INSIDE_VF
    mask_width_attribute = "elem_width"

    lhs = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_RMEM_AIV)
    rhs = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_RMEM_AIV)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_BOOL_RMEM_AIV)
    cmp_op = ParamDef(
        kind="attribute",
        annotation=Literal["eq", "ne", "lt", "le", "gt", "ge"],
        default="lt",
    )
    elem_width = ParamDef(kind="attribute", annotation=int, default=32)
    between = (SameConstraint("dtype", "lhs", "rhs"),)


_register_broadcast_binary_semantics(
    AscendVcmp, lambda ctx: _CMP_FNS[ctx.op.cmp_op](ctx.args[0].data, ctx.args[1].data)
)


@register_op(dialect="T", category="ascend", name="ascend_vcmps")
class AscendVcmps(Op):
    """AIV vector-scalar comparison producing a bool predicate register."""

    capability = OpCapability("ascend.vector")
    resource = "vector_engine"
    execution_mesh = _AIV_MESH
    execution_region = _INSIDE_VF
    mask_width_attribute = "elem_width"

    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_RMEM_AIV)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_BOOL_RMEM_AIV)
    scalar = ParamDef(kind="attribute", annotation=float)
    cmp_op = ParamDef(
        kind="attribute",
        annotation=Literal["eq", "ne", "lt", "le", "gt", "ge"],
        default="lt",
    )
    elem_width = ParamDef(kind="attribute", annotation=int, default=32)


_register_unary_semantics(AscendVcmps, register_eval=False)


@register_schedule_eval(AscendVcmps)
def _eval_vcmps(ctx):
    return TensorValue(
        data=_CMP_FNS[ctx.op.cmp_op](ctx.args[0].data, ctx.op.scalar),
        type=ctx.result_type,
    )


class _ScalarVectorBase(_VectorUnaryBase):
    """Shared structure for AIV scalar-vector arithmetic instructions."""

    scalar = ParamDef(kind="attribute", annotation=float)


def _register_scalar_vector_semantics(op_cls: type[Op], fn) -> None:
    _register_unary_semantics(op_cls, register_eval=False)

    @register_schedule_eval(op_cls)
    def _eval(ctx):
        data = ctx.args[0].data
        scalar_tensor = torch.tensor(ctx.op.scalar, dtype=data.dtype, device=data.device)
        return TensorValue(data=fn(data, scalar_tensor), type=ctx.result_type)


@register_op(dialect="T", category="ascend", name="ascend_vadds")
class AscendVadds(_ScalarVectorBase):
    """AIV vector + immediate scalar inside SimdVF."""

    capability = OpCapability("ascend.vector")


@register_op(dialect="T", category="ascend", name="ascend_vmuls")
class AscendVmuls(_ScalarVectorBase):
    """AIV vector * immediate scalar inside SimdVF."""

    capability = OpCapability("ascend.vector")


@register_op(dialect="T", category="ascend", name="ascend_vmaxs")
class AscendVmaxs(_ScalarVectorBase):
    """AIV elementwise max(vector, scalar) inside SimdVF."""

    capability = OpCapability("ascend.vector")


@register_op(dialect="T", category="ascend", name="ascend_vmins")
class AscendVmins(_ScalarVectorBase):
    """AIV elementwise min(vector, scalar) inside SimdVF."""

    capability = OpCapability("ascend.vector")


_register_scalar_vector_semantics(AscendVadds, torch.add)
_register_scalar_vector_semantics(AscendVmuls, torch.mul)
_register_scalar_vector_semantics(AscendVmaxs, torch.maximum)
_register_scalar_vector_semantics(AscendVmins, torch.minimum)


@register_op(dialect="T", category="ascend", name="ascend_vdup")
class AscendVdup(Op):
    """Broadcast a scalar constant into an AIV vector register inside SimdVF."""

    capability = OpCapability("ascend.vector")
    resource = "vector_engine"
    execution_mesh = _AIV_MESH
    execution_region = _INSIDE_VF

    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_RMEM_AIV)
    value = ParamDef(kind="attribute", annotation=float)


_register_dst_only_semantics(
    AscendVdup,
    lambda ctx: torch.full(
        tuple(int(d) for d in ctx.result_type.shape),
        ctx.op.value,
        dtype=to_torch_dtype(ctx.result_type.dtype),
    ),
)


@register_op(dialect="T", category="ascend", name="ascend_vdupv")
class AscendVdupv(_VectorUnaryBase):
    """Broadcast one lane value (src[..., pos]) across the last axis inside SimdVF."""

    capability = OpCapability("ascend.vector")
    pos = ParamDef(kind="attribute", annotation=int, default=0)


_register_unary_semantics(AscendVdupv, register_eval=False)


@register_schedule_eval(AscendVdupv)
def _eval_vdupv(ctx):
    src = ctx.args[0].data
    pos = ctx.op.pos
    if src.ndim == 0:
        return TensorValue(data=src.clone(), type=ctx.result_type)
    lane = src[..., pos : pos + 1].expand_as(src).clone()
    return TensorValue(data=lane, type=ctx.result_type)


def _eval_vci_data(ctx):
    dst_ty = ctx.result_type
    shape = tuple(int(d) for d in dst_ty.shape)
    dtype = to_torch_dtype(dst_ty.dtype)
    if not shape:
        return torch.tensor(ctx.op.start, dtype=dtype)
    offsets = torch.arange(shape[-1], dtype=torch.int64)
    vals = (
        ctx.op.start - offsets
        if ctx.op.order == "DEC_ORDER"
        else ctx.op.start + offsets
    ).to(dtype)
    return vals.expand(shape).clone()


@register_op(dialect="T", category="ascend", name="ascend_vci")
class AscendVci(Op):
    """Generate an arithmetic index vector (start +/- lane_id) inside SimdVF."""

    capability = OpCapability("ascend.vector")
    resource = "vector_engine"
    execution_mesh = _AIV_MESH
    execution_region = _INSIDE_VF

    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_RMEM_AIV)
    start = ParamDef(kind="attribute", annotation=int, default=0)
    order = ParamDef(
        kind="attribute",
        annotation=Literal["INC_ORDER", "DEC_ORDER"],
        default="INC_ORDER",
    )


_register_dst_only_semantics(AscendVci, _eval_vci_data)


@register_op(dialect="T", category="ascend", name="ascend_vmula")
class AscendVmula(Op):
    """Fused AIV vector multiply-accumulate (acc = acc + lhs * rhs) inside SimdVF."""

    capability = OpCapability("ascend.vector")
    resource = "vector_engine"
    execution_mesh = _AIV_MESH
    execution_region = _INSIDE_VF

    acc = ParamDef(
        kind="input",
        effect=MemoryEffect.READ | MemoryEffect.WRITE,
        pattern=_RMEM_AIV,
    )
    lhs = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_RMEM_AIV)
    rhs = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_RMEM_AIV)
    between = (SameConstraint("dtype", "acc", "lhs"), SameConstraint("dtype", "acc", "rhs"))


@register_typeinfer(AscendVmula)
def _infer_vmula(call: Call, ctx) -> UnitType:
    return UnitType()


register_access_relation(AscendVmula)(identity_relations)


@register_schedule_eval(AscendVmula)
def _eval_vmula(ctx):
    acc, lhs, rhs = (arg.data for arg in ctx.args)
    return TensorValue(data=acc + lhs * rhs, type=ctx.result_type)


@register_verify_stmt(AscendVmula)
def _verify_vmula(call: Call, ctx) -> None:
    aty, lty, rty = (ctx.type_of(arg) for arg in call.args)
    shape = broadcast_all((aty.shape, lty.shape, rty.shape))
    if shape != tuple(aty.shape):
        ctx.error(call, f"AscendVmula: acc shape {aty.shape} != broadcast {shape}")


def _dst_shape_from_ctx(ctx) -> tuple[int, ...]:
    if len(ctx.args) >= 2 and getattr(ctx.args[1], "data", None) is not None:
        return tuple(ctx.args[1].data.shape)
    return tuple(int(d) for d in ctx.result_type.shape)


def _atomic_reduce(acc: torch.Tensor, src: torch.Tensor, atomic_op: str) -> torch.Tensor:
    if atomic_op == "add":
        return acc + src
    if atomic_op == "max":
        return torch.maximum(acc, src)
    if atomic_op == "min":
        return torch.minimum(acc, src)
    raise ValueError(f"unsupported atomic_op {atomic_op!r}")


@register_copy_semantics
@register_op(dialect="T", category="ascend", name="ascend_dual_copy_l0c_to_ub")
class AscendDualCopyL0cToUb(Op):
    """Hardware FixPipe dual-destination L0C->UB redistribution (T.dual_copy)."""

    capability = OpCapability("ascend.dual_copy")
    resource = "dual_copy_engine"
    execution_mesh = _MEM_MESH
    execution_region = _OUTSIDE_VF

    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_TMEM)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_SMEM)
    split_mode = ParamDef(
        kind="attribute",
        annotation=Literal["m_split", "n_split"],
        default="m_split",
    )
    between = _SAME_SHAPE_DTYPE


@register_copy_semantics
@register_op(dialect="T", category="ascend", name="ascend_dual_copy_ub_to_l1")
class AscendDualCopyUbToL1(Op):
    """Coupled dual-AIV UB->L1 tile assembly with optional ND-to-NZ conversion."""

    capability = OpCapability("ascend.dual_copy")
    resource = "dual_copy_engine"
    execution_mesh = _MEM_MESH
    execution_region = _OUTSIDE_VF

    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_SMEM)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_TMEM)
    nd2nz = ParamDef(kind="attribute", annotation=bool, default=True)
    between = _SAME_SHAPE_DTYPE


@register_op(dialect="T", category="ascend", name="ascend_gm_to_l1_nz")
class AscendGmToL1Nz(Op):
    """MTE2 GM->L1 load with on-the-fly ND-to-NZ / DN-to-NZ layout conversion."""

    capability = OpCapability("ascend.dma")
    resource = "mte2_engine"
    execution_mesh = _MEM_MESH
    execution_region = _OUTSIDE_VF

    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_GMEM_CAND)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_TMEM_CAND)
    nz_mode = ParamDef(
        kind="attribute",
        annotation=Literal["nd2nz", "dn2nz"],
        default="nd2nz",
    )
    transpose = ParamDef(kind="attribute", annotation=bool, default=False)
    between = (SameConstraint("dtype", "src", "dst"),)


@register_typeinfer(AscendGmToL1Nz)
def _infer_gm_to_l1_nz(call: Call, ctx) -> UnitType:
    return UnitType()


@register_access_relation(AscendGmToL1Nz)
def _access_gm_to_l1_nz(call: Call, ctx) -> tuple[AccessRelation, ...]:
    if not call.target.transpose:
        return identity_relations(call, ctx)
    dty = ctx.type_of(call.args[1])
    rank = len(dty.shape)
    dims = [f"d{i}" for i in range(rank)]
    src_coords = [*dims[:-2], dims[-1], dims[-2]]
    src_rel = AccessRelation(isl.map(f"{{ [{', '.join(dims)}] -> [{', '.join(src_coords)}] }}"))
    id_rel = identity_access(rank)
    return iterating(dty.shape, (src_rel, id_rel, id_rel))


@register_schedule_eval(AscendGmToL1Nz)
def _eval_gm_to_l1_nz(ctx):
    src = ctx.args[0].data
    out = src.transpose(-2, -1).contiguous() if ctx.op.transpose else src.clone()
    return TensorValue(data=out, type=ctx.result_type)


@register_verify_stmt(AscendGmToL1Nz)
def _verify_gm_to_l1_nz(call: Call, ctx) -> None:
    sty, dty = (ctx.type_of(arg) for arg in call.args)
    if call.target.transpose:
        if len(sty.shape) < 2 or tuple(dty.shape) != (*sty.shape[:-2], sty.shape[-1], sty.shape[-2]):
            ctx.error(
                call,
                f"AscendGmToL1Nz(transpose=True): expected transposed shape from {sty.shape}, got {dty.shape}",
            )
    elif tuple(sty.shape) != tuple(dty.shape):
        ctx.error(call, f"AscendGmToL1Nz: shape mismatch {sty.shape} vs {dty.shape}")


@register_op(dialect="T", category="ascend", name="ascend_gm_to_ub_pad")
class AscendGmToUbPad(Op):
    """MTE2 GM->UB copy with hardware tail right-padding (pad_value)."""

    capability = OpCapability("ascend.dma")
    resource = "mte2_engine"
    execution_mesh = _MEM_MESH
    execution_region = _OUTSIDE_VF

    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_GMEM_CAND)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_SMEM_CAND)
    pad_value = ParamDef(kind="attribute", annotation=float, default=0.0)
    between = (SameConstraint("dtype", "src", "dst"),)


@register_typeinfer(AscendGmToUbPad)
def _infer_gm_to_ub_pad(call: Call, ctx) -> UnitType:
    return UnitType()


@register_access_relation(AscendGmToUbPad)
def _access_gm_to_ub_pad(call: Call, ctx) -> tuple[AccessRelation, ...]:
    sty, dty = ctx.type_of(call.args[0]), ctx.type_of(call.args[1])
    if tuple(sty.shape) == tuple(dty.shape):
        return identity_relations(call, ctx)
    rank = len(dty.shape)
    dims = ", ".join(f"d{i}" for i in range(rank))
    k_src = static_dim_value(sty.shape[-1])
    if k_src is None:
        return identity_relations(call, ctx)
    src_rel = AccessRelation(isl.map(f"{{ [{dims}] -> [{dims}] : d{rank - 1} < {k_src} }}"))
    id_rel = identity_access(rank)
    return iterating(dty.shape, (src_rel, id_rel, id_rel))


@register_schedule_eval(AscendGmToUbPad)
def _eval_gm_to_ub_pad(ctx):
    src = ctx.args[0].data
    dst_shape = _dst_shape_from_ctx(ctx)
    out = torch.full(dst_shape, float(ctx.op.pad_value), dtype=src.dtype, device=src.device)
    out[..., : src.shape[-1]] = src
    return TensorValue(data=out, type=ctx.result_type)


@register_verify_stmt(AscendGmToUbPad)
def _verify_gm_to_ub_pad(call: Call, ctx) -> None:
    sty, dty = (ctx.type_of(arg) for arg in call.args)
    if len(sty.shape) != len(dty.shape) or not sty.shape:
        ctx.error(call, f"AscendGmToUbPad: rank mismatch {sty.shape} vs {dty.shape}")
        return
    if sty.shape[:-1] != dty.shape[:-1]:
        ctx.error(call, f"AscendGmToUbPad: prefix shape mismatch {sty.shape} vs {dty.shape}")
        return
    k_src, k_dst = static_dim_value(sty.shape[-1]), static_dim_value(dty.shape[-1])
    if k_src is not None and k_dst is not None and k_src > k_dst:
        ctx.error(call, f"AscendGmToUbPad: src inner extent {k_src} > dst {k_dst}")


@register_op(dialect="T", category="ascend", name="ascend_vld_dist")
class AscendVldDist(Op):
    """Structured SimdVF UB->Reg vector load supporting lane broadcast (BRC_B32)."""

    capability = OpCapability("ascend.vector")
    resource = "vector_engine"
    execution_mesh = _AIV_MESH
    execution_region = _INSIDE_VF

    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_SMEM_CAND)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_RMEM_AIV_CAND)
    dist = ParamDef(
        kind="attribute",
        annotation=Literal["NORM_B32", "BRC_B32", "US_B16"],
        default="NORM_B32",
    )
    between = (SameConstraint("dtype", "src", "dst"),)


@register_typeinfer(AscendVldDist)
def _infer_vld_dist(call: Call, ctx) -> UnitType:
    return UnitType()


register_access_relation(AscendVldDist)(identity_relations)


@register_schedule_eval(AscendVldDist)
def _eval_vld_dist(ctx):
    dst_shape = _dst_shape_from_ctx(ctx)
    return TensorValue(
        data=torch.broadcast_to(ctx.args[0].data, dst_shape).clone(),
        type=ctx.result_type,
    )


@register_verify_stmt(AscendVldDist)
def _verify_vld_dist(call: Call, ctx) -> None:
    sty, dty = (ctx.type_of(arg) for arg in call.args)
    if call.target.dist == "BRC_B32":
        if broadcast_shapes(sty.shape, dty.shape, raising=False) != tuple(dty.shape):
            ctx.error(
                call,
                f"AscendVldDist(BRC_B32): dst {dty.shape} != broadcast from {sty.shape}",
            )
    elif tuple(sty.shape) != tuple(dty.shape):
        ctx.error(call, f"AscendVldDist: shape mismatch {sty.shape} vs {dty.shape}")


@register_op(dialect="T", category="ascend", name="ascend_vsts_dist")
class AscendVstsDist(Op):
    """Structured SimdVF Reg->UB vector store supporting ONEPT_B32 and PK_B32."""

    capability = OpCapability("ascend.vector")
    resource = "vector_engine"
    execution_mesh = _AIV_MESH
    execution_region = _INSIDE_VF

    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_RMEM_AIV_CAND)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_SMEM_CAND)
    dist = ParamDef(
        kind="attribute",
        annotation=Literal["NORM_B32", "ONEPT_B32", "PK_B32"],
        default="NORM_B32",
    )
    between = (SameConstraint("dtype", "src", "dst"),)


@register_typeinfer(AscendVstsDist)
def _infer_vsts_dist(call: Call, ctx) -> UnitType:
    return UnitType()


@register_access_relation(AscendVstsDist)
def _access_vsts_dist(call: Call, ctx) -> tuple[AccessRelation, ...]:
    sty, dty = ctx.type_of(call.args[0]), ctx.type_of(call.args[1])
    if call.target.dist == "ONEPT_B32" and tuple(sty.shape) != tuple(dty.shape):
        rank = len(dty.shape)
        dims = [f"d{i}" for i in range(rank)]
        src_coords = [*dims[:-1], "0"]
        src_rel = AccessRelation(isl.map(f"{{ [{', '.join(dims)}] -> [{', '.join(src_coords)}] }}"))
        id_rel = identity_access(rank)
        return iterating(dty.shape, (src_rel, id_rel, id_rel))
    return identity_relations(call, ctx)


@register_schedule_eval(AscendVstsDist)
def _eval_vsts_dist(ctx):
    src = ctx.args[0].data
    dst_shape = _dst_shape_from_ctx(ctx)
    if ctx.op.dist == "ONEPT_B32" and dst_shape and dst_shape[-1] == 1 and src.shape[-1] > 1:
        out = src[..., :1].clone()
    else:
        out = src.clone()
    return TensorValue(data=out, type=ctx.result_type)


@register_verify_stmt(AscendVstsDist)
def _verify_vsts_dist(call: Call, ctx) -> None:
    sty, dty = (ctx.type_of(arg) for arg in call.args)
    if call.target.dist == "ONEPT_B32" and tuple(sty.shape) != tuple(dty.shape):
        if len(sty.shape) != len(dty.shape) or not dty.shape:
            ctx.error(call, f"AscendVstsDist(ONEPT_B32): rank mismatch {sty.shape} vs {dty.shape}")
            return
        if sty.shape[:-1] != dty.shape[:-1] or static_dim_value(dty.shape[-1]) != 1:
            ctx.error(
                call,
                f"AscendVstsDist(ONEPT_B32): expected trailing 1 from {sty.shape}, got {dty.shape}",
            )
    elif tuple(sty.shape) != tuple(dty.shape):
        ctx.error(call, f"AscendVstsDist: shape mismatch {sty.shape} vs {dty.shape}")


@register_op(dialect="T", category="ascend", name="ascend_ub_to_gm_atomic")
class AscendUbToGmAtomic(Op):
    """Writeback-stage MTE3 UB->GM transfer with hardware atomic accumulation."""

    capability = OpCapability("ascend.dma")
    resource = "mte3_engine"
    execution_mesh = _MEM_MESH
    execution_region = _OUTSIDE_VF

    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_SMEM_CAND)
    dst = ParamDef(kind="input", effect=MemoryEffect.READ | MemoryEffect.WRITE, pattern=_GMEM_CAND)
    atomic_op = ParamDef(
        kind="attribute",
        annotation=Literal["add", "max", "min"],
        default="add",
    )
    between = _SAME_SHAPE_DTYPE


@register_typeinfer(AscendUbToGmAtomic)
def _infer_ub_to_gm_atomic(call: Call, ctx) -> UnitType:
    return UnitType()


register_access_relation(AscendUbToGmAtomic)(identity_relations)


@register_schedule_eval(AscendUbToGmAtomic)
def _eval_ub_to_gm_atomic(ctx):
    out = _atomic_reduce(ctx.args[1].data, ctx.args[0].data, ctx.op.atomic_op)
    return TensorValue(data=out, type=ctx.result_type)


@register_verify_stmt(AscendUbToGmAtomic)
def _verify_ub_to_gm_atomic(call: Call, ctx) -> None:
    sty, dty = (ctx.type_of(arg) for arg in call.args)
    if tuple(sty.shape) != tuple(dty.shape) or sty.dtype != dty.dtype:
        ctx.error(call, f"AscendUbToGmAtomic: operand mismatch {sty} vs {dty}")


@register_op(dialect="T", category="ascend", name="ascend_l0c_to_gm_atomic")
class AscendL0cToGmAtomic(Op):
    """Writeback-stage FixPipe L0C->GM transfer with quant_pre cast and atomic accumulation."""

    capability = OpCapability("ascend.dma")
    resource = "fixpipe_engine"
    execution_mesh = _MEM_MESH
    execution_region = _OUTSIDE_VF

    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_TMEM_CAND)
    dst = ParamDef(kind="input", effect=MemoryEffect.READ | MemoryEffect.WRITE, pattern=_GMEM_CAND)
    atomic_op = ParamDef(
        kind="attribute",
        annotation=Literal["add", "max", "min"],
        default="add",
    )
    between = (SameConstraint("shape", "src", "dst"),)


@register_typeinfer(AscendL0cToGmAtomic)
def _infer_l0c_to_gm_atomic(call: Call, ctx) -> UnitType:
    return UnitType()


register_access_relation(AscendL0cToGmAtomic)(identity_relations)


@register_schedule_eval(AscendL0cToGmAtomic)
def _eval_l0c_to_gm_atomic(ctx):
    src = ctx.args[0].data.to(ctx.args[1].data.dtype)
    out = _atomic_reduce(ctx.args[1].data, src, ctx.op.atomic_op)
    return TensorValue(data=out, type=ctx.result_type)


@register_verify_stmt(AscendL0cToGmAtomic)
def _verify_l0c_to_gm_atomic(call: Call, ctx) -> None:
    sty, dty = (ctx.type_of(arg) for arg in call.args)
    if tuple(sty.shape) != tuple(dty.shape):
        ctx.error(call, f"AscendL0cToGmAtomic: shape mismatch {sty.shape} vs {dty.shape}")


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
        AscendDualCopyL0cToUb,
        AscendDualCopyUbToL1,
        AscendGmToL1Nz,
        AscendGmToUbPad,
        AscendUbToReg,
        AscendRegToUb,
        AscendVldDist,
        AscendVstsDist,
        AscendUbToGmAtomic,
        AscendL0cToGmAtomic,
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
        AscendVbinary,
        AscendVunary,
        AscendVreduce,
        AscendVwhere,
        AscendVsel,
        AscendVclamp,
        AscendPset,
        AscendUpdateMask,
        AscendPand,
        AscendPor,
        AscendPxor,
        AscendPnot,
        AscendVcmp,
        AscendVcmps,
        AscendVadds,
        AscendVmuls,
        AscendVmaxs,
        AscendVmins,
        AscendVdup,
        AscendVdupv,
        AscendVci,
        AscendVmula,
    )
)

__all__ = [
    "ASCEND_OP_SPECS",
    "AscendDualCopy",
    "AscendDualCopyL0cToUb",
    "AscendDualCopyUbToL1",
    "AscendGmToL1",
    "AscendGmToL1Nz",
    "AscendGmToUb",
    "AscendGmToUbPad",
    "AscendL0cToGm",
    "AscendL0cToGmAtomic",
    "AscendL1ToL0A",
    "AscendL1ToL0B",
    "AscendMmad",
    "AscendOpSpec",
    "AscendPand",
    "AscendPnot",
    "AscendPor",
    "AscendPset",
    "AscendPxor",
    "AscendRegToUb",
    "AscendUbToGm",
    "AscendUbToGmAtomic",
    "AscendUbToReg",
    "AscendUpdateMask",
    "AscendVadd",
    "AscendVadds",
    "AscendVbinary",
    "AscendVcast",
    "AscendVci",
    "AscendVclamp",
    "AscendVcmp",
    "AscendVcmps",
    "AscendVdiv",
    "AscendVdup",
    "AscendVdupv",
    "AscendVexp",
    "AscendVldDist",
    "AscendVmax",
    "AscendVmaxs",
    "AscendVmins",
    "AscendVmul",
    "AscendVmula",
    "AscendVmuls",
    "AscendVreduce",
    "AscendVreduceMax",
    "AscendVreduceSum",
    "AscendVrsqrt",
    "AscendVsel",
    "AscendVstsDist",
    "AscendVsub",
    "AscendVunary",
    "AscendVwhere",
]
