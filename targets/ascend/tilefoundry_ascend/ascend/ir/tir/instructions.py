"""Canonical pipe- and resource-defining TIR primitives for Ascend 950."""

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
    """Hardware pipe and placement descriptor for one canonical Ascend TIR op."""

    op_cls: type[Op]
    name: str
    resource: str
    capability: str
    placement: str


def _coerce_enum(enum_cls, value):
    """Coerce string names or values into *enum_cls* instances."""
    if isinstance(value, enum_cls) or value is None:
        return value
    if isinstance(value, str):
        for member in enum_cls:
            if value == member.value or value.upper() == member.name:
                return member
    return value


def _reduce_axes(op, rank: int) -> tuple[int, ...]:
    """Normalize negative reduction axes against *rank*."""
    axes = tuple(axis + rank if axis < 0 else axis for axis in op.axes)
    if any(axis < 0 or axis >= rank for axis in axes):
        raise ValueError(f"{type(op).__name__} axes {op.axes} exceed source rank {rank}")
    return axes


def _reduce_dispatch(src, axes: tuple[int, ...], keepdim: bool, kind: ReduceKind):
    """Evaluate one vector reduction over *src* on the host reference backend."""
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
    """Evaluate one unary vector operation lazily without import-time torch lookup."""
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


@register_copy_semantics
@register_op(dialect="T", category="ascend", name="ascend_gm_to_ub")
class AscendGmToUb(Op):
    """Copy a tile from global HBM to AIV Unified Buffer over MTE2."""

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
    """Copy a tile from AIV Unified Buffer to global HBM over MTE3."""

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
    """Copy a tile from global HBM to AIC L1 buffer over MTE2."""

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
    """Load an L1 matrix tile into Cube L0A over MTE1."""

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
    """Load an L1 matrix tile into Cube L0B over MTE1."""

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
    """Store a Cube L0C accumulator tile to global HBM over FixPipe."""

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
    """Bidirectional cross-core AIC<->AIV tensor handoff between TMEM and UB."""

    capability = OpCapability("ascend.dual_copy")
    resource = "dual_copy_engine"
    execution_mesh = _MEM_MESH
    execution_region = _OUTSIDE_VF
    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_DUAL_SRC)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_DUAL_DST)
    between = (*_SAME_SHAPE_DTYPE, DistinctConstraint("storage", "src", "dst"))


@register_copy_semantics
@register_op(dialect="T", category="ascend", name="ascend_dual_copy_l0c_to_ub")
class AscendDualCopyL0cToUb(Op):
    """Hardware FixPipe dual-destination L0C->UB redistribution across coupled AIVs."""

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
    """Coupled dual-AIV UB->L1 tile assembly with optional ND-to-NZ layout conversion."""

    capability = OpCapability("ascend.dual_copy")
    resource = "dual_copy_engine"
    execution_mesh = _MEM_MESH
    execution_region = _OUTSIDE_VF
    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_SMEM)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_TMEM)
    nd2nz = ParamDef(kind="attribute", annotation=bool, default=True)
    between = _SAME_SHAPE_DTYPE


@register_op(dialect="T", category="ascend", name="ascend_cross_core_sync")
class AscendCrossCoreSync(Op):
    """Cross-core AIC<->AIV Mode-4 flag or hardware mutex synchronization primitive."""

    capability = OpCapability("ascend.dual_copy")
    resource = "dual_copy_engine"
    execution_mesh = _MEM_MESH
    execution_region = _OUTSIDE_VF

    flag = ParamDef(
        kind="input",
        effect=MemoryEffect.READ | MemoryEffect.WRITE,
        pattern=TensorPattern(),
    )
    mode = ParamDef(
        kind="attribute",
        annotation=Literal["mode4", "mutex", "flag_set", "flag_wait"],
        default="mode4",
    )
    flag_id = ParamDef(kind="attribute", annotation=int, default=0)
    sender = ParamDef(
        kind="attribute",
        annotation=Literal["aic", "aiv", "both"],
        default="aic",
    )
    receiver = ParamDef(
        kind="attribute",
        annotation=Literal["aic", "aiv", "both"],
        default="aiv",
    )
    between = ()


@register_typeinfer(AscendCrossCoreSync)
def _infer_cross_core_sync(call: Call, ctx) -> UnitType:
    return UnitType()


register_access_relation(AscendCrossCoreSync)(identity_relations)


@register_schedule_eval(AscendCrossCoreSync)
def _eval_cross_core_sync(ctx):
    return TensorValue(data=ctx.args[0].data.clone(), type=ctx.result_type)


@register_verify_stmt(AscendCrossCoreSync)
def _verify_cross_core_sync(call: Call, ctx) -> None:
    fid = call.target.flag_id
    if call.target.mode == "mutex":
        if fid < 0 or fid >= 28:
            ctx.error(call, f"AscendCrossCoreSync(mutex): flag_id {fid} out of range [0, 28)")
    elif fid < 0 or fid > 15:
        ctx.error(call, f"AscendCrossCoreSync({call.target.mode}): flag_id {fid} exceeds depth 15")


@register_op(dialect="T", category="ascend", name="ascend_mmad")
class AscendMmad(Op):
    """Cube matrix multiply-accumulate on L0A/L0B into L0C."""

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


@register_op(dialect="T", category="ascend", name="ascend_vcast")
class AscendVcast(Op):
    """Convert vector register element precision inside SimdVF."""

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


@register_op(dialect="T", category="ascend", name="ascend_vbinary")
class AscendVbinary(Op):
    """Kind-tagged AIV binary vector primitive matching core HIR Binary."""

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

    def __init__(self, **attrs) -> None:
        if "kind" in attrs:
            attrs["kind"] = _coerce_enum(BinaryKind, attrs["kind"])
        super().__init__(**attrs)


@register_typeinfer(AscendVbinary)
def _infer_vbinary(call: Call, ctx) -> UnitType:
    return UnitType()


@register_access_relation(AscendVbinary)
def _access_vbinary(call: Call, ctx) -> tuple[AccessRelation, ...]:
    shapes = tuple(ctx.type_of(arg).shape for arg in call.args[:2])
    out_shape = broadcast_all(shapes)
    out = identity_access(len(out_shape))
    return iterating(
        out_shape,
        (*(broadcast_access(out_shape, shape) for shape in shapes), out, out),
    )


@register_schedule_eval(AscendVbinary)
def _eval_vbinary(ctx):
    return TensorValue(
        data=eval_binary_kernel(ctx.op.kind, ctx.args[0].data, ctx.args[1].data),
        type=ctx.result_type,
    )


@register_verify_stmt(AscendVbinary)
def _verify_vbinary(call: Call, ctx) -> None:
    lty, rty, dty = (ctx.type_of(arg) for arg in call.args)
    shape = broadcast_shapes(lty.shape, rty.shape, raising=False)
    if shape != tuple(dty.shape):
        ctx.error(
            call,
            f"AscendVbinary: dst {dty.shape} != broadcast({lty.shape}, {rty.shape})",
        )


class _VectorUnaryBase(Op):
    """Shared AIV unary vector structure inside SimdVF."""

    capability = OpCapability("ascend.vector")
    resource = "vector_engine"
    execution_mesh = _AIV_MESH
    execution_region = _INSIDE_VF

    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_RMEM_AIV)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_RMEM_AIV)
    between = _SAME_SHAPE_DTYPE


def _register_unary_common(op_cls: type[Op]) -> None:
    """Register typeinfer, identity access, and shape verification on *op_cls*."""

    @register_typeinfer(op_cls)
    def _infer(call: Call, ctx) -> UnitType:
        return UnitType()

    register_access_relation(op_cls)(identity_relations)

    @register_verify_stmt(op_cls)
    def _verify(call: Call, ctx) -> None:
        sty, dty = (ctx.type_of(arg) for arg in call.args)
        if tuple(sty.shape) != tuple(dty.shape):
            ctx.error(call, f"{op_cls.__name__}: shape mismatch {sty.shape} vs {dty.shape}")


@register_op(dialect="T", category="ascend", name="ascend_vunary")
class AscendVunary(_VectorUnaryBase):
    """Kind-tagged AIV unary vector primitive matching core HIR Unary."""

    capability = OpCapability("ascend.vector")
    mask_width_attribute = "mask_width"

    kind = ParamDef(kind="attribute", annotation=UnaryKind)
    mask_width = ParamDef(kind="attribute", annotation=int, default=32)

    def __init__(self, **attrs) -> None:
        if "kind" in attrs:
            attrs["kind"] = _coerce_enum(UnaryKind, attrs["kind"])
        super().__init__(**attrs)


_register_unary_common(AscendVunary)


@register_schedule_eval(AscendVunary)
def _eval_vunary(ctx):
    return TensorValue(
        data=_unary_dispatch(ctx.op.kind, ctx.args[0].data),
        type=ctx.result_type,
    )


@register_op(dialect="T", category="ascend", name="ascend_vreduce")
class AscendVreduce(Op):
    """Kind-tagged AIV vector reduction primitive matching core HIR Reduce."""

    capability = OpCapability("ascend.vector")
    resource = "vector_engine"
    execution_mesh = _AIV_MESH
    execution_region = _INSIDE_VF

    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_RMEM_AIV)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_RMEM_AIV)
    axes = ParamDef(kind="attribute", annotation=tuple, default=(-1,))
    keepdim = ParamDef(kind="attribute", annotation=bool, default=False)
    kind = ParamDef(kind="attribute", annotation=ReduceKind, default=ReduceKind.SUM)
    between = (SameConstraint("dtype", "src", "dst"),)

    def __init__(self, **attrs) -> None:
        if "kind" in attrs:
            attrs["kind"] = _coerce_enum(ReduceKind, attrs["kind"])
        super().__init__(**attrs)


@register_typeinfer(AscendVreduce)
def _infer_vreduce(call: Call, ctx) -> UnitType:
    return UnitType()


@register_access_relation(AscendVreduce)
def _access_vreduce(call: Call, ctx) -> tuple[AccessRelation, ...]:
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
        {axis: axis for axis in range(rank)} if op.keepdim else dict(enumerate(surviving))
    )
    writes_at = [
        "0" if axis not in came_from or came_from[axis] in axes else f"d{came_from[axis]}"
        for axis in range(len(out_shape))
    ]
    domain = ", ".join(f"d{axis}" for axis in range(rank))
    dst_access = AccessRelation(isl.map(f"{{ [{domain}] -> [{', '.join(writes_at)}] }}"))
    return iterating(source.shape, (identity_access(rank), dst_access, dst_access))


@register_schedule_eval(AscendVreduce)
def _eval_vreduce(ctx):
    src = ctx.args[0].data
    axes = _reduce_axes(ctx.op, src.ndim)
    return TensorValue(
        data=_reduce_dispatch(src, axes, ctx.op.keepdim, ctx.op.kind),
        type=ctx.result_type,
    )


@register_verify_stmt(AscendVreduce)
def _verify_vreduce(call: Call, ctx) -> None:
    op = call.target
    sty, dty = (ctx.type_of(arg) for arg in call.args)
    rank = len(sty.shape)
    try:
        axes = _reduce_axes(op, rank)
    except ValueError as exc:
        ctx.error(call, str(exc))
        return
    expected = tuple(
        1 if axis in axes else extent
        for axis, extent in enumerate(sty.shape)
        if op.keepdim or axis not in axes
    )
    if tuple(dty.shape) != expected:
        ctx.error(call, f"AscendVreduce: dst shape {dty.shape} != expected {expected}")


@register_op(dialect="T", category="ascend", name="ascend_vwhere")
class AscendVwhere(Op):
    """Predicated AIV elementwise selection matching core HIR Where."""

    capability = OpCapability("ascend.vector")
    resource = "vector_engine"
    execution_mesh = _AIV_MESH
    execution_region = _INSIDE_VF

    cond = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_BOOL_RMEM_AIV)
    lhs = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_RMEM_AIV)
    rhs = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_RMEM_AIV)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_RMEM_AIV)
    between = (SameConstraint("dtype", "lhs", "rhs"), SameConstraint("dtype", "lhs", "dst"))


@register_typeinfer(AscendVwhere)
def _infer_vwhere(call: Call, ctx) -> UnitType:
    return UnitType()


@register_access_relation(AscendVwhere)
def _access_vwhere(call: Call, ctx) -> tuple[AccessRelation, ...]:
    shapes = tuple(ctx.type_of(arg).shape for arg in call.args[:3])
    out_shape = broadcast_all(shapes)
    out = identity_access(len(out_shape))
    return iterating(
        out_shape,
        (*(broadcast_access(out_shape, shape) for shape in shapes), out, out),
    )


@register_schedule_eval(AscendVwhere)
def _eval_vwhere(ctx):
    cond, lhs, rhs = ctx.args[0].data, ctx.args[1].data, ctx.args[2].data
    return TensorValue(data=torch.where(cond, lhs, rhs), type=ctx.result_type)


@register_verify_stmt(AscendVwhere)
def _verify_vwhere(call: Call, ctx) -> None:
    cond, lhs, rhs, dst = (ctx.type_of(arg) for arg in call.args)
    if broadcast_all((cond.shape, lhs.shape, rhs.shape)) != tuple(dst.shape):
        ctx.error(call, "AscendVwhere: dst shape != broadcast of inputs")


@register_op(dialect="T", category="ascend", name="ascend_vclamp")
class AscendVclamp(_VectorUnaryBase):
    """Elementwise AIV vector clamp primitive matching core HIR Clamp."""

    capability = OpCapability("ascend.vector")
    min_val = ParamDef(kind="attribute", annotation=float)
    max_val = ParamDef(kind="attribute", annotation=float)


_register_unary_common(AscendVclamp)


@register_schedule_eval(AscendVclamp)
def _eval_vclamp(ctx):
    return TensorValue(
        data=torch.clamp(ctx.args[0].data, min=ctx.op.min_val, max=ctx.op.max_val),
        type=ctx.result_type,
    )


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
        AscendCrossCoreSync,
        AscendMmad,
        AscendUbToReg,
        AscendRegToUb,
        AscendVcast,
        AscendVbinary,
        AscendVunary,
        AscendVreduce,
        AscendVwhere,
        AscendVclamp,
    )
)

__all__ = [
    "ASCEND_OP_SPECS",
    "AscendCrossCoreSync",
    "AscendDualCopy",
    "AscendDualCopyL0cToUb",
    "AscendDualCopyUbToL1",
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
    "AscendVbinary",
    "AscendVcast",
    "AscendVclamp",
    "AscendVreduce",
    "AscendVunary",
    "AscendVwhere",
]
