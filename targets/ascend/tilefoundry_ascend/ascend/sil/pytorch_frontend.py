"""PyTorch-golden semantic frontend for physics ratios and analytical HIR blueprints."""

from __future__ import annotations

import ast
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from tilefoundry.ir.types import DType
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import (
    BlueprintProvenance,
    BlueprintScheduleDelta,
    HirBlueprintPacket,
    HirRooflineSummary,
    compile_hir_schedule_delta,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger
from tilefoundry_ascend.ascend.sil.schema import (
    CasePhysicsKind,
    CasePhysicsRatios,
    TopologyAxes,
    build_case_physics_context,
    compute_case_physics_ratios,
    derive_case_physics_input,
    dtype_bytes_for,
    flatten_int_shapes,
    flatten_str_values,
    has_empty_group_attrs,
    infer_case_dims_structural,
    infer_case_physics_kind,
)
from tilefoundry_ascend.ascend.target import Ascend950PR, AscendSpaceFacts, AscendTarget

_CONV_CALL_NAMES = frozenset(
    {
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    }
)
_MATMUL_CALL_NAMES = frozenset(
    {
        "addmm",
        "addmv",
        "baddbmm",
        "bmm",
        "einsum",
        "grouped_matmul",
        "linear",
        "matmul",
        "matmul_",
        "mm",
        "mv",
        "scaled_dot_product_attention",
        *_CONV_CALL_NAMES,
    }
)
_REDUCTION_CALL_NAMES = frozenset(
    {
        "all",
        "amax",
        "amin",
        "any",
        "argmax",
        "argmin",
        "cross_entropy",
        "cumprod",
        "cumsum",
        "group_norm",
        "layer_norm",
        "log_softmax",
        "logsumexp",
        "max",
        "mean",
        "norm",
        "prod",
        "rms_norm",
        "scaled_dot_product_attention",
        "softmax",
        "std",
        "sum",
        "var",
    }
)
_TRANSCENDENTAL_CALL_NAMES = frozenset(
    {
        "rsqrt",
        "silu",
        "gelu",
        "sigmoid",
        "tanh",
        "sin",
        "cos",
        "erf",
        "exp",
        "exp2",
        "expm1",
        "log",
        "log2",
        "log10",
        "softmax",
        "log_softmax",
        "pow",
        "sqrt",
    }
)
_SHAPE_PRESERVING_CALL_NAMES = frozenset(
    {
        "cumprod",
        "cumsum",
        "group_norm",
        "layer_norm",
        "log_softmax",
        "rms_norm",
        "softmax",
    }
)
_EXP_CALL_NAMES = frozenset({"exp", "exp2", "expm1"})
_MAX_CALL_NAMES = frozenset(
    {"amax", "clamp", "clamp_", "log_softmax", "max", "maximum", "min", "minimum", "softmax"}
)
_INDIRECT_CALL_NAMES = frozenset(
    {
        "bucketize",
        "embedding",
        "gather",
        "index",
        "index_add",
        "index_select",
        "scatter",
        "scatter_",
        "scatter_add",
        "searchsorted",
        "take",
        "topk",
    }
)
_SUBBYTE_OR_QUANT_CALL_TOKENS = (
    "dequant",
    "antiquant",
    "int4",
    "uint4",
    "quint4x2",
    "q4",
    "w4",
    "awq",
    "gptq",
    "nf4",
    "fp4",
    "mxfp",
    "block_quant",
)
_SUBBYTE_OR_QUANT_IDENT_TOKENS = (
    "dequant",
    "antiquant",
    "qweight",
    "mxscale",
    "block_size",
    "blocksize",
    "group_size",
    "groupsize",
    "weight_scale",
    "activation_scale",
    "int4",
    "uint4",
    "quint4x2",
    "fp4",
    "mxfp",
    "q4_",
    "awq",
    "gptq",
    "nf4",
)
_SUBBYTE_OR_QUANT_DTYPE_ATTRS = frozenset(
    {
        "int4",
        "uint4",
        "quint4x2",
        "qint8",
        "float4_e2m1",
        "float4_e1m2",
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e8m0fnu",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
    }
)
_STANDARD_8BIT_DTYPE_ATTRS = frozenset({"int8", "uint8"})
_BLOCK_SCALE_IDENT_NAMES = frozenset(
    {
        "scales",
        "scale_inv",
        "zero_point",
        "zeros",
        "qzeros",
        "block_size",
        "blocksize",
        "group_size",
        "groupsize",
    }
)
_DIM_KEYWORDS = frozenset({"dim", "axis"})
_TRANSCENDENTAL_OP_EQUIV = 8
TrafficBasis = Literal["declared_io_exact", "estimated_shape_dtype"]
ComputeBasis = Literal["exact_work_count", "estimated"]
_ProvenanceFlag = Literal["input", "reduction"]
_PROVENANCE_INPUT: _ProvenanceFlag = "input"
_PROVENANCE_REDUCTION: _ProvenanceFlag = "reduction"
_SHAPE_TRANSPARENT_METHODS = frozenset(
    {
        "bfloat16",
        "clone",
        "contiguous",
        "cpu",
        "cuda",
        "detach",
        "double",
        "float",
        "half",
        "npu",
        "to",
        "type",
    }
)


@dataclass(frozen=True, kw_only=True)
class PyTorchOpSemantics:
    """Structural mathematical semantics recovered from PyTorch golden source."""

    entry_name: str
    has_matmul: bool
    has_reduction: bool
    reduction_axes: tuple[int, ...]
    has_online_exp_max: bool
    has_indirect_index: bool
    has_subbyte_or_block_quant: bool
    has_recurrent_loop: bool
    mutates_inputs: bool
    preserves_input_shape: bool = False
    has_conv: bool = False
    transcendental_op_count: int = 0

    def __post_init__(self) -> None:
        if not self.entry_name.strip():
            raise ValueError("PyTorchOpSemantics.entry_name must be a non-empty string")
        axes = tuple(self.reduction_axes)
        for axis in axes:
            if isinstance(axis, bool) or not isinstance(axis, int):
                raise ValueError("PyTorchOpSemantics.reduction_axes must contain only ints")
        if self.transcendental_op_count < 0:
            raise ValueError("PyTorchOpSemantics.transcendental_op_count must be non-negative")
        object.__setattr__(self, "reduction_axes", axes)

    @property
    def base_axes(self) -> TopologyAxes:
        """Return the base topology axes implied by the recovered semantics."""
        coupled = self.has_matmul and (
            self.has_online_exp_max
            or self.has_subbyte_or_block_quant
            or self.has_indirect_index
            or self.has_recurrent_loop
        )
        if coupled:
            compute_engine = "coupled_1c2v"
        elif self.has_matmul:
            compute_engine = "aic_only"
        else:
            compute_engine = "aiv_only"

        if self.has_recurrent_loop and not self.has_matmul:
            memory_ring = "persistent_onchip_state"
        elif coupled and self.has_online_exp_max:
            memory_ring = "bidirectional_cv_ring"
        elif coupled:
            memory_ring = "l0c_ub_epilogue"
        elif self.has_matmul:
            memory_ring = "gm_l1_l0_fixpipe"
        elif self.has_indirect_index:
            memory_ring = "ub_random_atomic"
        else:
            memory_ring = "gm_ub_stream"

        iteration_regularity = (
            "irregular_simt" if self.has_indirect_index else "affine_simd"
        )
        return TopologyAxes(
            compute_engine=compute_engine,
            memory_ring=memory_ring,
            iteration_regularity=iteration_regularity,
        )


@dataclass(frozen=True, kw_only=True)
class RepresentativeCase:
    """Declared inputs and outputs of one official case for analytical HBM pricing."""

    case_id: int
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str | None, ...]
    output_dtype: str | None = None
    output_dtypes: tuple[str | None, ...] = ()
    output_shapes: tuple[tuple[int, ...], ...] = ()
    attrs: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.case_id, bool) or not isinstance(self.case_id, int):
            raise ValueError("RepresentativeCase.case_id must be a positive integer")
        if self.case_id < 1:
            raise ValueError("RepresentativeCase.case_id must be a positive integer")
        shapes = tuple(tuple(int(dim) for dim in shape) for shape in self.input_shapes)
        dtypes = tuple(self.input_dtypes)
        out_dtypes = tuple(self.output_dtypes)
        out_shapes = tuple(tuple(int(dim) for dim in shape) for shape in self.output_shapes)
        if not shapes:
            raise ValueError("RepresentativeCase.input_shapes must not be empty")
        if len(shapes) != len(dtypes):
            raise ValueError(
                "RepresentativeCase.input_shapes and input_dtypes must have equal length"
            )
        if any(dim < 0 for shape in shapes for dim in shape):
            raise ValueError("RepresentativeCase.input_shapes must not contain negative dims")
        if any(dtype is not None and not dtype.strip() for dtype in dtypes):
            raise ValueError("RepresentativeCase.input_dtypes must be non-empty strings or None")
        if self.output_dtype is not None and not self.output_dtype.strip():
            raise ValueError(
                "RepresentativeCase.output_dtype must be a non-empty string or None"
            )
        if any(dtype is not None and not dtype.strip() for dtype in out_dtypes):
            raise ValueError("RepresentativeCase.output_dtypes must be non-empty strings or None")
        object.__setattr__(self, "input_shapes", shapes)
        object.__setattr__(self, "input_dtypes", dtypes)
        object.__setattr__(self, "output_dtypes", out_dtypes)
        object.__setattr__(self, "output_shapes", out_shapes)
        object.__setattr__(self, "attrs", dict(self.attrs))

    @classmethod
    def from_case_entry(
        cls,
        case_entry: Mapping[str, object],
        proto: Mapping[str, object] | None = None,
    ) -> RepresentativeCase:
        """Pair declared input/output shapes and dtypes from case and proto metadata."""
        case_id = case_entry.get("case_id")
        if isinstance(case_id, bool) or not isinstance(case_id, int) or case_id < 1:
            raise ValueError("case entry requires a positive integer case_id")
        shapes, dtypes = _pair_case_shapes_and_dtypes(
            case_entry.get("input_shape"),
            case_entry.get("dtype"),
        )
        if not shapes:
            raise ValueError(f"case {case_id} requires at least one integer input shape")
        raw_attrs = case_entry.get("attrs")
        attrs = dict(raw_attrs) if isinstance(raw_attrs, Mapping) else {}
        output_dtype, output_dtypes = _resolve_proto_output_dtypes(proto, dtypes)
        return cls(
            case_id=case_id,
            input_shapes=shapes,
            input_dtypes=dtypes,
            output_dtype=output_dtype,
            output_dtypes=output_dtypes,
            attrs=attrs,
        )


def _is_int_sequence(raw: object) -> bool:
    return (
        isinstance(raw, (list, tuple))
        and len(raw) > 0
        and all(isinstance(dim, int) and not isinstance(dim, bool) for dim in raw)
    )


def _pair_case_shapes_and_dtypes(
    raw_shapes: object,
    raw_dtypes: object,
) -> tuple[tuple[tuple[int, ...], ...], tuple[str | None, ...]]:
    if not isinstance(raw_shapes, (list, tuple)):
        return (), ()
    if _is_int_sequence(raw_shapes):
        shape = tuple(int(dim) for dim in raw_shapes)
        if isinstance(raw_dtypes, str) and raw_dtypes.strip():
            return (shape,), (raw_dtypes.strip(),)
        flat_dtypes = flatten_str_values(raw_dtypes)
        if len(flat_dtypes) == 1:
            return (shape,), (flat_dtypes[0],)
        return (shape,), (None,)

    slots = list(raw_shapes)
    shapes: list[tuple[int, ...]] = []
    paired_dtypes: list[str | None] = []
    broadcast_dtype: str | None = None
    slot_dtypes: list[object] | None = None

    if isinstance(raw_dtypes, str) and raw_dtypes.strip():
        broadcast_dtype = raw_dtypes.strip()
    elif isinstance(raw_dtypes, (list, tuple)):
        if len(raw_dtypes) == len(slots):
            slot_dtypes = list(raw_dtypes)
        elif (
            len(raw_dtypes) == 1
            and isinstance(raw_dtypes[0], str)
            and raw_dtypes[0].strip()
        ):
            broadcast_dtype = raw_dtypes[0].strip()

    desynced = False
    for idx, slot in enumerate(slots):
        if slot is None:
            continue
        dt_slot = slot_dtypes[idx] if slot_dtypes is not None else broadcast_dtype
        if _is_int_sequence(slot):
            shapes.append(tuple(int(dim) for dim in slot))
            if isinstance(dt_slot, str) and dt_slot.strip():
                paired_dtypes.append(dt_slot.strip())
            elif (
                isinstance(dt_slot, (list, tuple))
                and len(dt_slot) == 1
                and isinstance(dt_slot[0], str)
                and dt_slot[0].strip()
            ):
                paired_dtypes.append(dt_slot[0].strip())
            elif slot_dtypes is None and broadcast_dtype is not None:
                paired_dtypes.append(broadcast_dtype)
            else:
                desynced = True
        elif isinstance(slot, (list, tuple)):
            sub_shapes = [
                tuple(int(dim) for dim in sub)
                for sub in slot
                if _is_int_sequence(sub)
            ]
            if not sub_shapes:
                continue
            shapes.extend(sub_shapes)
            if isinstance(dt_slot, str) and dt_slot.strip():
                paired_dtypes.extend([dt_slot.strip()] * len(sub_shapes))
            elif isinstance(dt_slot, (list, tuple)):
                sub_dt = [str(item).strip() for item in dt_slot if item is not None]
                if len(sub_dt) == len(sub_shapes) and all(sub_dt):
                    paired_dtypes.extend(sub_dt)
                elif len(sub_dt) == 1 and sub_dt[0]:
                    paired_dtypes.extend([sub_dt[0]] * len(sub_shapes))
                else:
                    desynced = True
            elif slot_dtypes is None and broadcast_dtype is not None:
                paired_dtypes.extend([broadcast_dtype] * len(sub_shapes))
            else:
                desynced = True

    if not shapes:
        return (), ()
    if desynced or len(paired_dtypes) != len(shapes):
        return tuple(shapes), (None,) * len(shapes)
    return tuple(shapes), tuple(paired_dtypes)


def _resolve_proto_output_dtypes(
    proto: Mapping[str, object] | None,
    input_dtypes: Sequence[str | None],
) -> tuple[str | None, tuple[str | None, ...]]:
    if not isinstance(proto, Mapping):
        return None, ()
    op_block = (
        proto["operator"]
        if "outputs" not in proto and isinstance(proto.get("operator"), Mapping)
        else proto
    )
    raw_outputs = op_block.get("outputs") if isinstance(op_block, Mapping) else None
    if not isinstance(raw_outputs, Sequence) or isinstance(raw_outputs, (str, bytes)):
        return None, ()
    declared_inputs = [dt for dt in input_dtypes if dt is not None]
    resolved: list[str | None] = []
    for out_item in raw_outputs:
        if not isinstance(out_item, Mapping):
            continue
        raw_dt = out_item.get("dtype")
        if isinstance(raw_dt, str) and raw_dt.strip():
            resolved.append(raw_dt.strip())
            continue
        if isinstance(raw_dt, Sequence) and not isinstance(raw_dt, (str, bytes)):
            cands = [str(item).strip() for item in raw_dt if item is not None and str(item).strip()]
            if len(cands) == 1:
                resolved.append(cands[0])
            elif cands:
                matched = next((dt for dt in declared_inputs if dt in cands), None)
                resolved.append(matched)
            else:
                resolved.append(None)
        else:
            resolved.append(None)
    if not resolved:
        return None, ()
    first_non_null = next((dt for dt in resolved if dt is not None), None)
    return first_non_null, tuple(resolved)


@dataclass(frozen=True, kw_only=True)
class _AnalyticalWork:
    cube_flops: int
    vector_flops: int
    min_gm_bytes: int
    special_ops: int
    resolved_dims: dict[str, int]
    work_model: str
    traffic_basis: TrafficBasis
    compute_basis: ComputeBasis


@dataclass(frozen=True, kw_only=True)
class _TrafficCharge:
    """One HBM tensor charge priced at a declared dtype or a marked fallback width."""

    element_count: int
    element_bytes: float
    dtype_declared: bool


def _call_leaf_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
_CallResolver = Callable[[str, _FunctionNode], _FunctionNode | None]


def _entry_parameters(entry: _FunctionNode) -> frozenset[str]:
    args = entry.args
    names = {arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    if args.vararg is not None:
        names.add(args.vararg.arg)
    if args.kwarg is not None:
        names.add(args.kwarg.arg)
    return frozenset(names)


def _direct_nested_functions(node: _FunctionNode) -> dict[str, _FunctionNode]:
    nested: dict[str, _FunctionNode] = {}
    pending: list[ast.AST] = list(ast.iter_child_nodes(node))
    while pending:
        child = pending.pop()
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nested.setdefault(child.name, child)
            continue
        pending.extend(ast.iter_child_nodes(child))
    return nested


def _function_scopes(
    tree: ast.Module,
) -> tuple[
    dict[str, _FunctionNode],
    dict[_FunctionNode, _FunctionNode | None],
    dict[_FunctionNode, dict[str, _FunctionNode]],
    dict[str, dict[str, _FunctionNode]],
    dict[str, str],
    dict[str, str],
    dict[_FunctionNode, str],
]:
    top_level: dict[str, _FunctionNode] = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    class_methods: dict[str, dict[str, _FunctionNode]] = {}
    owner_class: dict[_FunctionNode, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = {
                item.name: item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if methods:
                class_methods[node.name] = methods
                for method in methods.values():
                    owner_class[method] = node.name

    func_aliases: dict[str, str] = {}
    class_instances: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if isinstance(value, ast.Name):
                func_aliases[target.id] = value.id
            elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                if value.func.id in class_methods:
                    class_instances[target.id] = value.func.id

    parents: dict[_FunctionNode, _FunctionNode | None] = {}
    nested: dict[_FunctionNode, dict[str, _FunctionNode]] = {}

    def register(node: _FunctionNode, parent: _FunctionNode | None) -> None:
        parents[node] = parent
        child_functions = _direct_nested_functions(node)
        nested[node] = child_functions
        for child in child_functions.values():
            register(child, node)

    for node in top_level.values():
        register(node, None)
    for methods in class_methods.values():
        for method in methods.values():
            register(method, None)
    return (
        top_level,
        parents,
        nested,
        class_methods,
        func_aliases,
        class_instances,
        owner_class,
    )


def _resolve_alias_target(
    name: str,
    func_aliases: Mapping[str, str],
) -> str:
    current = name
    seen: set[str] = set()
    while current in func_aliases and current not in seen:
        seen.add(current)
        current = func_aliases[current]
    return current


def _local_bindings(function: _FunctionNode) -> tuple[dict[str, str], dict[str, str]]:
    local_aliases: dict[str, str] = {}
    local_calls: dict[str, str] = {}
    for node in _walk_executed_nodes(function):
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if isinstance(value, ast.Name):
                local_aliases[target.id] = value.id
            elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                local_calls[target.id] = value.func.id
    return local_aliases, local_calls


def _resolve_function(
    name: str,
    scope: _FunctionNode,
    top_level: Mapping[str, _FunctionNode],
    parents: Mapping[_FunctionNode, _FunctionNode | None],
    nested: Mapping[_FunctionNode, Mapping[str, _FunctionNode]],
    func_aliases: Mapping[str, str] | None = None,
) -> _FunctionNode | None:
    current: _FunctionNode | None = scope
    while current is not None:
        target = nested[current].get(name)
        if target is not None:
            return target
        current = parents[current]
    if name in top_level:
        return top_level[name]
    if func_aliases:
        resolved_name = _resolve_alias_target(name, func_aliases)
        return top_level.get(resolved_name)
    return None


def _walk_executed_nodes(node: _FunctionNode) -> Iterable[ast.AST]:
    pending: list[ast.AST] = list(ast.iter_child_nodes(node))
    while pending:
        child = pending.pop()
        yield child
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        pending.extend(ast.iter_child_nodes(child))


def _class_entry_targets(
    cls_name: str,
    class_methods: Mapping[str, Mapping[str, _FunctionNode]],
) -> list[_FunctionNode]:
    methods = class_methods.get(cls_name, {})
    targets: list[_FunctionNode] = []
    for method_name in ("__call__", "forward"):
        fn = methods.get(method_name)
        if fn is not None:
            targets.append(fn)
    return targets


def _reachable_functions(
    entry: _FunctionNode,
    *,
    top_level: Mapping[str, _FunctionNode],
    parents: Mapping[_FunctionNode, _FunctionNode | None],
    nested: Mapping[_FunctionNode, Mapping[str, _FunctionNode]],
    class_methods: Mapping[str, Mapping[str, _FunctionNode]] | None = None,
    func_aliases: Mapping[str, str] | None = None,
    class_instances: Mapping[str, str] | None = None,
    owner_class: Mapping[_FunctionNode, str] | None = None,
) -> tuple[_FunctionNode, ...]:
    cls_methods = class_methods or {}
    aliases = func_aliases or {}
    cls_instances = class_instances or {}
    fn_owner = owner_class or {}
    reachable: list[_FunctionNode] = []
    seen: set[int] = set()
    pending: list[_FunctionNode] = [entry]
    while pending:
        function = pending.pop()
        if id(function) in seen:
            continue
        seen.add(id(function))
        reachable.append(function)
        local_aliases, local_calls = _local_bindings(function)
        merged_aliases = {**aliases, **local_aliases}
        current_cls = fn_owner.get(function)
        for node in _walk_executed_nodes(function):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                call_id = node.func.id
                target = _resolve_function(
                    call_id,
                    function,
                    top_level,
                    parents,
                    nested,
                    merged_aliases,
                )
                if target is not None:
                    pending.append(target)
                    continue
                resolved_id = _resolve_alias_target(call_id, merged_aliases)
                inst_cls = (
                    local_calls.get(call_id)
                    or cls_instances.get(call_id)
                    or local_calls.get(resolved_id)
                    or cls_instances.get(resolved_id)
                    or (resolved_id if resolved_id in cls_methods else None)
                )
                if inst_cls is not None:
                    pending.extend(_class_entry_targets(inst_cls, cls_methods))
            elif isinstance(node.func, ast.Call) and isinstance(node.func.func, ast.Name):
                cls_name = _resolve_alias_target(node.func.func.id, merged_aliases)
                if cls_name in cls_methods:
                    pending.extend(_class_entry_targets(cls_name, cls_methods))
            elif isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                recv = node.func.value
                if isinstance(recv, ast.Name):
                    if recv.id == "self" and current_cls in cls_methods:
                        method = cls_methods[current_cls].get(attr)
                        if method is not None:
                            pending.append(method)
                            continue
                    recv_cls = (
                        local_calls.get(recv.id)
                        or cls_instances.get(recv.id)
                        or (_resolve_alias_target(recv.id, merged_aliases))
                    )
                    if recv_cls in cls_methods:
                        method = cls_methods[recv_cls].get(attr)
                        if method is not None:
                            pending.append(method)
                elif isinstance(recv, ast.Call) and isinstance(recv.func, ast.Name):
                    recv_cls = _resolve_alias_target(recv.func.id, merged_aliases)
                    if recv_cls in cls_methods:
                        method = cls_methods[recv_cls].get(attr)
                        if method is not None:
                            pending.append(method)
    return tuple(reachable)


def _detect_input_mutation(
    nodes: Iterable[ast.AST],
    input_names: frozenset[str],
) -> bool:
    if not input_names:
        return False
    for node in nodes:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr.endswith("_")
            and not node.func.attr.startswith("__")
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in input_names
        ):
            return True
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if (
                    keyword.arg == "out"
                    and isinstance(keyword.value, ast.Name)
                    and keyword.value.id in input_names
                ):
                    return True
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in input_names:
                return True
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                if target.value.id in input_names:
                    return True
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                if target.value.id in input_names:
                    return True
    return False


def _has_keepdim_true(node: ast.Call) -> bool:
    for keyword in node.keywords:
        if keyword.arg == "keepdim":
            return isinstance(keyword.value, ast.Constant) and keyword.value.value is True
    return False


def _expression_provenance(
    node: ast.expr | ast.keyword,
    assignments: Mapping[str, frozenset[_ProvenanceFlag]],
) -> frozenset[_ProvenanceFlag]:
    """Propagate entry-input and keepdim-reduction provenance through one expression."""
    if isinstance(node, ast.keyword):
        return _expression_provenance(node.value, assignments)
    if isinstance(node, ast.Name):
        return assignments.get(node.id, frozenset())
    if isinstance(node, ast.Call):
        leaf = _call_leaf_name(node.func)
        if leaf in _REDUCTION_CALL_NAMES and _has_keepdim_true(node):
            return frozenset({_PROVENANCE_REDUCTION})
    if isinstance(node, ast.Lambda):
        return frozenset()
    flags: set[_ProvenanceFlag] = set()
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.expr, ast.keyword)):
            flags |= _expression_provenance(child, assignments)
    return frozenset(flags)


def _returned_expressions(function: _FunctionNode) -> tuple[ast.expr, ...]:
    """Return the value expressions of every executed return statement in *function*."""
    return tuple(
        node.value
        for node in _walk_executed_nodes(function)
        if isinstance(node, ast.Return) and node.value is not None
    )


def _assignment_target_names(target: ast.expr) -> frozenset[str]:
    """Collect the local names one assignment target binds or writes."""
    if isinstance(target, ast.Name):
        return frozenset({target.id})
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in target.elts:
            names |= _assignment_target_names(element)
        return frozenset(names)
    if isinstance(target, (ast.Subscript, ast.Attribute)) and isinstance(target.value, ast.Name):
        return frozenset({target.value.id})
    return frozenset()


def _function_assignments(function: _FunctionNode) -> dict[str, tuple[ast.expr, ...]]:
    """Map every locally assigned name to the values written to it."""
    definitions: dict[str, list[ast.expr]] = {}
    for node in _walk_executed_nodes(function):
        targets: Sequence[ast.expr]
        value: ast.expr
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None:
            targets = (node.target,)
            value = node.value
        else:
            continue
        for target in targets:
            for name in _assignment_target_names(target):
                definitions.setdefault(name, []).append(value)
    return {name: tuple(values) for name, values in definitions.items()}


def _shape_transparent_base(node: ast.expr) -> ast.expr:
    """Peel dtype and storage-view methods that never change tensor shape."""
    current = node
    while isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
        if current.func.attr not in _SHAPE_TRANSPARENT_METHODS:
            break
        current = current.func.value
    return current


def _resolved_return_head(
    node: ast.expr,
    definitions: Mapping[str, tuple[ast.expr, ...]],
    seen: set[str],
) -> ast.expr:
    """Resolve single-assignment names on the return path down to the defining expression."""
    current = _shape_transparent_base(node)
    while isinstance(current, ast.Name):
        values = definitions.get(current.id, ())
        if len(values) != 1 or current.id in seen:
            break
        seen.add(current.id)
        current = _shape_transparent_base(values[0])
    return current


def _return_value_is_shape_reduced(
    node: ast.expr,
    definitions: Mapping[str, tuple[ast.expr, ...]],
    seen: set[str],
) -> bool:
    """Return True when evaluating *node* ends in a non-keepdim reduction."""
    if isinstance(node, (ast.Tuple, ast.List)):
        elements = tuple(
            element for element in node.elts if not isinstance(element, ast.Starred)
        )
        return bool(elements) and all(
            _return_value_is_shape_reduced(element, definitions, set(seen))
            for element in elements
        )
    head = _resolved_return_head(node, definitions, seen)
    if isinstance(head, ast.BinOp):
        return _return_value_is_shape_reduced(
            head.left, definitions, set(seen)
        ) or _return_value_is_shape_reduced(head.right, definitions, set(seen))
    if not isinstance(head, ast.Call):
        return False
    leaf = _call_leaf_name(head.func)
    if (
        leaf in _REDUCTION_CALL_NAMES
        and leaf not in _SHAPE_PRESERVING_CALL_NAMES
        and not _has_keepdim_true(head)
    ):
        return True
    children: list[ast.expr] = [
        *head.args,
        *(keyword.value for keyword in head.keywords),
    ]
    if isinstance(head.func, ast.Attribute):
        children.append(head.func.value)
    return any(
        _return_value_is_shape_reduced(child, definitions, set(seen))
        for child in children
    )


def _return_path_groups(
    function: _FunctionNode,
    resolve: _CallResolver,
    seen_functions: set[int],
) -> dict[int, tuple[_FunctionNode, list[ast.AST]]]:
    """Collect return-path AST nodes per function, following local helper calls."""
    groups: dict[int, tuple[_FunctionNode, list[ast.AST]]] = {}
    if id(function) in seen_functions:
        return groups
    seen_functions.add(id(function))
    definitions = _function_assignments(function)
    expanded: set[str] = set()
    nodes: list[ast.AST] = []
    pending: list[ast.expr] = list(_returned_expressions(function))
    while pending:
        expression = pending.pop()
        for node in ast.walk(expression):
            nodes.append(node)
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id not in expanded
            ):
                expanded.add(node.id)
                pending.extend(definitions.get(node.id, ()))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                target = resolve(node.func.id, function)
                if target is not None:
                    for key, value in _return_path_groups(
                        target, resolve, seen_functions
                    ).items():
                        groups[key] = value
    groups[id(function)] = (function, nodes)
    return groups


def _closure_has_keepdim_broadcast(
    function: _FunctionNode,
    nodes: Sequence[ast.AST],
) -> bool:
    """Detect input-shaped data combined with a keepdim reduction of input data."""
    definitions = _function_assignments(function)
    assignments: dict[str, frozenset[_ProvenanceFlag]] = {
        name: frozenset({_PROVENANCE_INPUT})
        for name in _entry_parameters(function)
    }
    changed = True
    while changed:
        changed = False
        for name, values in definitions.items():
            flags: frozenset[_ProvenanceFlag] = frozenset()
            for value in values:
                flags |= _expression_provenance(value, assignments)
            current = assignments.get(name, frozenset())
            merged = current | flags
            if merged != current:
                assignments[name] = merged
                changed = True
    for node in nodes:
        if not isinstance(node, ast.BinOp):
            continue
        left = _expression_provenance(node.left, assignments)
        right = _expression_provenance(node.right, assignments)
        if _PROVENANCE_INPUT in left and _PROVENANCE_REDUCTION in right:
            return True
        if _PROVENANCE_REDUCTION in left and _PROVENANCE_INPUT in right:
            return True
    return False


def _detect_return_shape_preservation(
    entry: _FunctionNode,
    resolve: _CallResolver,
) -> bool:
    """Decide shape preservation from the values the entry function actually returns."""
    groups = tuple(_return_path_groups(entry, resolve, set()).values())
    returns = _returned_expressions(entry)
    definitions = _function_assignments(entry)
    if not returns or all(
        _return_value_is_shape_reduced(value, definitions, set()) for value in returns
    ):
        return False
    node_groups = tuple(group[1] for group in groups)
    for nodes in node_groups:
        for node in nodes:
            if (
                isinstance(node, ast.Call)
                and _call_leaf_name(node.func) in _SHAPE_PRESERVING_CALL_NAMES
            ):
                return True
    reductions_keepdim = all(
        _has_keepdim_true(node)
        for nodes in node_groups
        for node in nodes
        if isinstance(node, ast.Call)
        and _call_leaf_name(node.func) in _REDUCTION_CALL_NAMES
        and _call_leaf_name(node.func) not in _SHAPE_PRESERVING_CALL_NAMES
    )
    if not reductions_keepdim:
        return False
    return any(
        _closure_has_keepdim_broadcast(function, nodes) for function, nodes in groups
    )


def _constant_int(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        if isinstance(node.value, bool):
            return None
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _constant_int(node.operand)
        if operand is None:
            return None
        return -operand if isinstance(node.op, ast.USub) else operand
    return None


def _has_quant_evidence(
    nodes: Iterable[ast.AST],
    call_names: Iterable[str],
) -> bool:
    for call_name in call_names:
        lowered = call_name.lower()
        if any(marker in lowered for marker in _SUBBYTE_OR_QUANT_CALL_TOKENS):
            return True
    has_8bit_dtype = False
    has_block_scale_ident = False
    for node in nodes:
        if isinstance(node, ast.Attribute):
            if node.attr in _SUBBYTE_OR_QUANT_DTYPE_ATTRS:
                return True
            if node.attr in _STANDARD_8BIT_DTYPE_ATTRS:
                has_8bit_dtype = True
        elif isinstance(node, (ast.Name, ast.arg)):
            ident = node.id if isinstance(node, ast.Name) else node.arg
            lowered_ident = ident.lower()
            if any(marker in lowered_ident for marker in _SUBBYTE_OR_QUANT_IDENT_TOKENS):
                return True
            if lowered_ident in _BLOCK_SCALE_IDENT_NAMES:
                has_block_scale_ident = True
    return has_8bit_dtype and has_block_scale_ident


def _positive_dim(dims: Mapping[str, int], names: Sequence[str]) -> int | None:
    for name in names:
        value = dims.get(name)
        if value is not None and value > 0:
            return value
    return None


def _require_positive_dim(dims: Mapping[str, int], names: Sequence[str], what: str) -> int:
    value = _positive_dim(dims, names)
    if value is None:
        raise ValueError(f"{what} requires one of anchor dims {tuple(names)} to be positive")
    return value


def _dim_or(dims: Mapping[str, int], names: Sequence[str], default: int) -> int:
    value = _positive_dim(dims, names)
    return default if value is None else value


def _shape_leading_dim_product(dims: Mapping[str, int], prefix: str) -> int | None:
    factors: list[int] = []
    index = 0
    while f"{prefix}_dim_{index}" in dims:
        value = dims[f"{prefix}_dim_{index}"]
        if value <= 0:
            return None
        factors.append(value)
        index += 1
    if not factors:
        return None
    leading = 1
    for value in factors[:-1]:
        leading *= value
    return leading


def _validated_dtype_bytes(dtype_bytes: float) -> float:
    value = float(dtype_bytes)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("dtype_bytes must be finite and positive")
    return value


def _declared_element_bytes(dtype_name: str | None) -> float | None:
    """Return a declared element width, or None when absent, unknown, or sub-byte packed."""
    if dtype_name is None:
        return None
    try:
        element_bytes = dtype_bytes_for(dtype_name)
    except ValueError:
        return None
    return element_bytes if element_bytes >= 1.0 else None


def _declared_input_charges(
    representative_case: RepresentativeCase,
    fallback_dtype_bytes: float,
) -> list[_TrafficCharge]:
    charges: list[_TrafficCharge] = []
    for shape, dtype_name in zip(
        representative_case.input_shapes, representative_case.input_dtypes
    ):
        element_bytes = _declared_element_bytes(dtype_name)
        charges.append(
            _TrafficCharge(
                element_count=math.prod(shape),
                element_bytes=(
                    fallback_dtype_bytes if element_bytes is None else element_bytes
                ),
                dtype_declared=element_bytes is not None,
            )
        )
    return charges


def _declared_output_charges(
    representative_case: RepresentativeCase,
    *,
    element_count: int,
    fallback_dtype_bytes: float,
) -> list[_TrafficCharge]:
    if len(representative_case.output_dtypes) > 1:
        charges: list[_TrafficCharge] = []
        for idx, dtype_name in enumerate(representative_case.output_dtypes):
            out_elems = (
                math.prod(representative_case.output_shapes[idx])
                if idx < len(representative_case.output_shapes)
                else element_count
            )
            element_bytes = _declared_element_bytes(dtype_name)
            charges.append(
                _TrafficCharge(
                    element_count=out_elems,
                    element_bytes=(
                        fallback_dtype_bytes if element_bytes is None else element_bytes
                    ),
                    dtype_declared=element_bytes is not None,
                )
            )
        return charges
    dtype_name = (
        representative_case.output_dtypes[0]
        if representative_case.output_dtypes
        else representative_case.output_dtype
    )
    element_bytes = _declared_element_bytes(dtype_name)
    return [
        _TrafficCharge(
            element_count=element_count,
            element_bytes=fallback_dtype_bytes if element_bytes is None else element_bytes,
            dtype_declared=element_bytes is not None,
        )
    ]


def _priced_traffic(
    charges: Sequence[_TrafficCharge],
    *,
    packed: bool,
) -> tuple[int, TrafficBasis]:
    total_bytes = int(
        math.ceil(sum(charge.element_count * charge.element_bytes for charge in charges))
    )
    traffic_basis: TrafficBasis = (
        "declared_io_exact"
        if not packed and all(charge.dtype_declared for charge in charges)
        else "estimated_shape_dtype"
    )
    return total_bytes, traffic_basis


class PyTorchSemanticAnalyzer:
    """Recover PyTorch semantics, physics ratios, and analytical HIR blueprints."""

    def __init__(self, target: AscendTarget | None = None) -> None:
        self.target = AscendTarget("ascend.950pr") if target is None else target
        device = self.target.device
        if not isinstance(device, Ascend950PR):
            raise RuntimeError(
                f"PyTorchSemanticAnalyzer requires Ascend950PR, got {type(device).__name__}"
            )
        self.device = device
        facts = self.target.get_facts(AscendSpaceFacts)
        if not isinstance(facts, AscendSpaceFacts):
            raise RuntimeError(
                f"AscendTarget.get_facts returned {type(facts).__name__}, expected AscendSpaceFacts"
            )
        self.facts = facts
        unit_rates = device.unit_flops_per_second
        aic_rates = unit_rates.get("aic", {})
        aiv_rates = unit_rates.get("aiv", {})
        cube_peak = (
            aic_rates.get(DType.from_name("bf16"))
            or aic_rates.get(DType.from_name("f16"))
            or 0
        )
        vector_peak = (
            aiv_rates.get(DType.from_name("f32"))
            or max(aiv_rates.values(), default=0)
            or device.aiv_lane_events_per_second
        )
        if cube_peak <= 0 or vector_peak <= 0:
            raise RuntimeError("Ascend950PR must declare positive AIC and AIV peak rates")
        if device.hbm_bandwidth_bytes_per_second <= 0:
            raise RuntimeError("Ascend950PR must declare a positive HBM bandwidth")
        self.cube_peak_flops = float(cube_peak)
        self.vector_peak_flops = float(vector_peak)
        self.hbm_bytes_per_second = float(device.hbm_bandwidth_bytes_per_second)

    def analyze_source(
        self,
        source: str,
        *,
        entry_name: str,
        proto: object | None = None,
    ) -> PyTorchOpSemantics:
        """Recover semantics from the entry function and its statically reachable helpers."""
        if not entry_name.strip():
            raise ValueError("entry_name must be a non-empty string")
        if proto is not None and not isinstance(proto, Mapping):
            raise ValueError("proto must be a mapping")
        tree = ast.parse(source)
        (
            top_level,
            parents,
            nested,
            class_methods,
            func_aliases,
            class_instances,
            owner_class,
        ) = _function_scopes(tree)
        entry = top_level.get(entry_name)
        if entry is None:
            resolved_entry = _resolve_alias_target(entry_name, func_aliases)
            entry = top_level.get(resolved_entry)
        if entry is None and entry_name in class_instances:
            targets = _class_entry_targets(class_instances[entry_name], class_methods)
            if targets:
                entry = targets[0]
        if entry is None and entry_name in class_methods:
            targets = _class_entry_targets(entry_name, class_methods)
            if targets:
                entry = targets[0]
        if entry is None and len(top_level) == 1:
            entry = next(iter(top_level.values()))
        if entry is None:
            raise ValueError(f"entry function {entry_name!r} not found in PyTorch source")
        reachable = _reachable_functions(
            entry,
            top_level=top_level,
            parents=parents,
            nested=nested,
            class_methods=class_methods,
            func_aliases=func_aliases,
            class_instances=class_instances,
            owner_class=owner_class,
        )
        nodes = tuple(
            node for function in reachable for node in _walk_executed_nodes(function)
        )
        call_names: set[str] = set()
        reduction_axes: set[int] = set()
        has_matmult = False
        has_loop = False
        transcendental_op_count = 0
        for node in nodes:
            if isinstance(node, ast.MatMult):
                has_matmult = True
            if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                has_loop = True
            if isinstance(node, ast.Call):
                call_name = _call_leaf_name(node.func)
                if call_name is not None:
                    call_names.add(call_name)
                    if call_name in _TRANSCENDENTAL_CALL_NAMES:
                        transcendental_op_count += 1
                if call_name in _REDUCTION_CALL_NAMES:
                    for keyword in node.keywords:
                        if keyword.arg in _DIM_KEYWORDS:
                            axis = _constant_int(keyword.value)
                            if axis is not None:
                                reduction_axes.add(axis)
        has_online_normalization = bool(
            call_names & {"log_softmax", "softmax", "scaled_dot_product_attention"}
        )
        has_exp = bool(call_names & _EXP_CALL_NAMES) or has_online_normalization
        has_max = bool(call_names & _MAX_CALL_NAMES) or (
            "scaled_dot_product_attention" in call_names
        )

        def resolve_call(name: str, scope: _FunctionNode) -> _FunctionNode | None:
            return _resolve_function(
                name, scope, top_level, parents, nested, func_aliases
            )

        preserves_input_shape = _detect_return_shape_preservation(entry, resolve_call)
        return PyTorchOpSemantics(
            entry_name=entry_name,
            has_matmul=has_matmult or bool(call_names & _MATMUL_CALL_NAMES),
            has_reduction=bool(call_names & _REDUCTION_CALL_NAMES),
            reduction_axes=tuple(sorted(reduction_axes)),
            has_online_exp_max=has_exp and has_max,
            has_indirect_index=bool(call_names & _INDIRECT_CALL_NAMES),
            has_subbyte_or_block_quant=_has_quant_evidence(nodes, call_names),
            has_recurrent_loop=has_loop,
            mutates_inputs=_detect_input_mutation(nodes, _entry_parameters(entry)),
            preserves_input_shape=preserves_input_shape,
            has_conv=bool(call_names & _CONV_CALL_NAMES),
            transcendental_op_count=transcendental_op_count,
        )

    def compute_case_ratios(
        self,
        semantics: PyTorchOpSemantics,
        case_entry: Mapping[str, object],
        *,
        dtype_bytes: float = 2.0,
    ) -> CasePhysicsRatios:
        """Compute the six dimensionless physics ratios for one concrete case entry."""
        fallback_dtype_bytes = _validated_dtype_bytes(dtype_bytes)
        case_id = case_entry.get("case_id")
        if isinstance(case_id, bool) or not isinstance(case_id, int) or case_id < 1:
            raise ValueError("case entry requires a positive integer case_id")
        shapes = flatten_int_shapes(case_entry.get("input_shape"))
        if not shapes:
            raise ValueError(f"case {case_id} requires at least one integer input shape")
        raw_attrs = case_entry.get("attrs")
        if raw_attrs is None:
            attrs: Mapping[str, object] = {}
        elif isinstance(raw_attrs, Mapping):
            attrs = raw_attrs
        else:
            raise ValueError(f"case {case_id} attrs must be a mapping")
        dtypes = flatten_str_values(case_entry.get("dtype"))
        element_bytes = (
            dtype_bytes_for(dtypes[0]) if dtypes else fallback_dtype_bytes
        )
        axes = semantics.base_axes
        kind = infer_case_physics_kind(axes, has_reduction=semantics.has_reduction)
        dims = infer_case_dims_structural(
            shapes,
            attrs,
            axes=axes,
            uses_cube=axes.uses_cube,
        )
        reduction_axis = semantics.reduction_axes[0] if semantics.reduction_axes else -1
        context = build_case_physics_context(
            kind,
            element_bytes,
            self.facts,
            reduction_axis=reduction_axis,
            strided_gather=semantics.has_indirect_index,
            has_empty_group=has_empty_group_attrs(attrs),
        )
        raw_case = derive_case_physics_input(case_id, shapes, attrs, dims, context)
        return compute_case_physics_ratios(raw_case)

    def build_universal_blueprint(
        self,
        semantics: PyTorchOpSemantics,
        anchor_dims: Mapping[str, int],
        *,
        authored_hir_source: str | None = None,
        scheduled_hir_source: str | None = None,
        dtype_bytes: float = 2.0,
        representative_case: RepresentativeCase | None = None,
    ) -> HirBlueprintPacket:
        """Build a semantic-first blueprint anchored to real dimensions and optional HIR delta.

        Authoritative Roofline floors always derive from *semantics*, *anchor_dims*, and
        *representative_case*. When *authored_hir_source* is supplied, optional schedule
        enrichment attaches via ``schedule_delta`` without overwriting ``roofline``.
        """
        element_bytes = _validated_dtype_bytes(dtype_bytes)
        dims = {str(key): int(value) for key, value in anchor_dims.items()}
        work = self._estimate_analytical_work(
            semantics, dims, element_bytes, representative_case
        )
        compute_flops = work.cube_flops + work.vector_flops
        cube_floor_s = (
            work.cube_flops / self.cube_peak_flops if work.cube_flops > 0 else 0.0
        )
        vec_floor_s = (
            work.vector_flops / self.vector_peak_flops if work.vector_flops > 0 else 0.0
        )
        cube_floor_us = cube_floor_s * 1e6
        vec_floor_us = vec_floor_s * 1e6
        compute_floor_us = (
            vec_floor_us if work.cube_flops == 0 else max(cube_floor_us, vec_floor_us)
        )
        memory_floor_us = work.min_gm_bytes / self.hbm_bytes_per_second * 1e6
        ideal_roofline_us = max(compute_floor_us, memory_floor_us)
        if ideal_roofline_us <= 0.0:
            dominant_bound = "none"
        elif abs(compute_floor_us - memory_floor_us) <= 1e-12 * max(1.0, ideal_roofline_us):
            dominant_bound = "balanced"
        elif memory_floor_us > compute_floor_us:
            dominant_bound = "memory"
        else:
            dominant_bound = "compute"
        arithmetic_intensity = (
            round(compute_flops / float(work.min_gm_bytes), 4)
            if work.min_gm_bytes > 0
            else 0.0
        )
        roofline = HirRooflineSummary(
            anchor_dims=dict(work.resolved_dims),
            compute_flops=compute_flops,
            min_gm_bytes=work.min_gm_bytes,
            arithmetic_intensity=arithmetic_intensity,
            compute_floor_us=compute_floor_us,
            memory_floor_us=memory_floor_us,
            ideal_roofline_us=ideal_roofline_us,
            dominant_bound=dominant_bound,
            requested_anchor_dims=dict(dims),
            resolved_hir_dims=dict(work.resolved_dims),
            shape_exact_match=True,
            fallback_reason=None,
            cube_flops=work.cube_flops,
            vector_flops=work.vector_flops,
            special_ops=work.special_ops,
            cube_floor_us=cube_floor_us,
            vec_floor_us=vec_floor_us,
        )
        has_authored_hir = bool(authored_hir_source is not None and authored_hir_source.strip())
        has_scheduled_hir = bool(
            has_authored_hir
            and scheduled_hir_source is not None
            and scheduled_hir_source.strip()
        )
        schedule_delta: BlueprintScheduleDelta | None = None
        reference_ledger: ObjectivePhysicalLedger | None = None
        diagnostics: tuple[str, ...] = ("analytical_pytorch_frontend",)
        if has_authored_hir:
            assert authored_hir_source is not None
            schedule_delta, reference_ledger, hir_diags = compile_hir_schedule_delta(
                authored_hir_source,
                scheduled_hir_source=scheduled_hir_source if has_scheduled_hir else None,
                anchor_dims=dims,
                target=self.target,
            )
            if hir_diags:
                diagnostics = (*diagnostics, *hir_diags)
        provenance = BlueprintProvenance(
            semantic_source=(
                "pytorch_golden_plus_hir" if has_authored_hir else "pytorch_golden"
            ),
            traffic_basis=work.traffic_basis,
            coordinate_scope="representative_anchor",
            has_authored_hir=has_authored_hir,
            has_scheduled_hir=has_scheduled_hir,
        )
        return HirBlueprintPacket(
            entry_name=semantics.entry_name,
            roofline=roofline,
            reference_ledger=reference_ledger,
            lowering_diagnostics=diagnostics,
            requested_anchor_dims=dict(dims),
            resolved_hir_dims=dict(work.resolved_dims),
            shape_exact_match=True,
            fallback_reason=None,
            provenance=provenance,
            schedule_delta=schedule_delta,
        )

    build_blueprint = build_universal_blueprint

    def _estimate_analytical_work(
        self,
        semantics: PyTorchOpSemantics,
        dims: Mapping[str, int],
        dtype_bytes: float,
        representative_case: RepresentativeCase | None,
    ) -> _AnalyticalWork:
        kind = infer_case_physics_kind(
            semantics.base_axes,
            has_reduction=semantics.has_reduction,
        )
        if kind == "attention":
            return _attention_work(semantics, dims, dtype_bytes, representative_case)
        if kind == "matmul":
            if semantics.has_indirect_index and _has_attention_anchor_dims(dims):
                return _attention_work(
                    semantics, dims, dtype_bytes, representative_case
                )
            return _matmul_work(semantics, dims, dtype_bytes, representative_case)
        return _stream_work(
            semantics, dims, kind, dtype_bytes, representative_case
        )


def _has_attention_anchor_dims(dims: Mapping[str, int]) -> bool:
    sequence = ("S_q", "S", "s_dim")
    return (
        _positive_dim(dims, sequence) is not None
        and _positive_dim(dims, ("N_q", "n_q", "H_q", "h_q", "N_h", "n_h")) is not None
        and _positive_dim(dims, ("N_kv", "n_kv")) is not None
    )


def _attention_work(
    semantics: PyTorchOpSemantics,
    dims: Mapping[str, int],
    dtype_bytes: float,
    representative_case: RepresentativeCase | None,
) -> _AnalyticalWork:
    layout = str(dims.get("inputLayout", dims.get("layout", "BSND"))).upper()
    batch = _dim_or(dims, ("B", "b_dim"), 1)
    num_q_heads = _dim_or(dims, ("N_q", "n_q", "H_q", "h_q", "N_h", "n_h"), 1)
    num_kv_heads = _dim_or(dims, ("N_kv", "n_kv"), num_q_heads)
    s_q = _require_positive_dim(dims, ("S_q", "S", "s_dim", "M", "m_rows"), "attention")
    s_kv = _positive_dim(dims, ("S_kv", "s_kv"))
    if s_kv is None:
        sequence_index = 2 if layout == "BNSD" else 1
        s_kv = _positive_dim(
            dims,
            (f"shape_1_dim_{sequence_index}", "shape_1_dim_1", "shape_1_dim_0"),
        )
    if s_kv is None:
        s_kv = s_q
    d_nope = _positive_dim(dims, ("D_nope", "d_nope"))
    d_rope = _positive_dim(dims, ("D_rope", "d_rope"))
    d_qk = _positive_dim(dims, ("D_qk", "D", "d_dim", "dk"))
    if d_qk is None and d_nope is not None and d_rope is not None:
        d_qk = d_nope + d_rope
    if d_qk is None:
        raise ValueError(
            "attention analytical Roofline requires D_qk, D, or D_nope+D_rope anchor dims"
        )
    d_v = _positive_dim(dims, ("D_v", "dv", "d_v")) or d_nope or d_qk
    score_pairs = batch * num_q_heads * s_q * s_kv
    cube_flops = 2 * score_pairs * (d_qk + d_v)
    special_ops = score_pairs if semantics.has_online_exp_max else 0
    vector_flops = 5 * score_pairs + special_ops * _TRANSCENDENTAL_OP_EQUIV
    output_elements = batch * num_q_heads * s_q * d_v
    gm_elements = batch * (
        num_q_heads * s_q * d_qk
        + num_kv_heads * s_kv * d_qk
        + num_kv_heads * s_kv * d_v
        + num_q_heads * s_q * d_v
    )
    if representative_case is None:
        charges = [
            _TrafficCharge(
                element_count=gm_elements,
                element_bytes=dtype_bytes,
                dtype_declared=False,
            )
        ]
    else:
        charges = _declared_input_charges(representative_case, dtype_bytes)
        charges.extend(
            _declared_output_charges(
                representative_case,
                element_count=output_elements,
                fallback_dtype_bytes=dtype_bytes,
            )
        )
    min_gm_bytes, traffic_basis = _priced_traffic(
        charges, packed=semantics.has_subbyte_or_block_quant
    )
    return _AnalyticalWork(
        cube_flops=cube_flops,
        vector_flops=vector_flops,
        min_gm_bytes=max(1, min_gm_bytes),
        special_ops=special_ops,
        resolved_dims={
            "B": batch,
            "S_q": s_q,
            "S_kv": s_kv,
            "N_q": num_q_heads,
            "N_kv": num_kv_heads,
            "D_qk": d_qk,
            "D_v": d_v,
        },
        work_model="analytical_attention_online" if special_ops else "analytical_attention",
        traffic_basis=traffic_basis,
        compute_basis="estimated",
    )


def _conv_attr_item(
    attrs: Mapping[str, object],
    keys: Sequence[str],
    axis: int,
    default: int,
) -> int:
    for key in keys:
        if key not in attrs:
            continue
        raw = attrs[key]
        if isinstance(raw, int) and not isinstance(raw, bool):
            return max(1, int(raw))
        if isinstance(raw, (list, tuple)) and axis < len(raw):
            val = raw[axis]
            if isinstance(val, int) and not isinstance(val, bool):
                return max(1, int(val))
    return default


def _conv_pad_pair(
    attrs: Mapping[str, object],
    keys: Sequence[str],
    axis: int,
    spatial_rank: int,
) -> int:
    for key in keys:
        if key not in attrs:
            continue
        raw = attrs[key]
        if isinstance(raw, int) and not isinstance(raw, bool):
            return max(0, int(raw) * 2)
        if isinstance(raw, (list, tuple)):
            if len(raw) == 2 * spatial_rank and 2 * axis + 1 < len(raw):
                lo, hi = raw[2 * axis], raw[2 * axis + 1]
                if isinstance(lo, int) and isinstance(hi, int):
                    return max(0, int(lo) + int(hi))
            if axis < len(raw) and isinstance(raw[axis], int):
                return max(0, int(raw[axis]) * 2)
    return 0


def _derive_matmul_dims_from_case(
    semantics: PyTorchOpSemantics,
    dims: Mapping[str, int],
    representative_case: RepresentativeCase,
) -> tuple[int, int, int, int, int, int] | None:
    shapes = representative_case.input_shapes
    if len(shapes) < 2:
        return None
    attrs = representative_case.attrs
    s0, s1 = shapes[0], shapes[1]
    transpose_a = bool(attrs.get("transpose_a", attrs.get("trans_a", False)))
    transpose_b = bool(
        attrs.get("transpose_b", attrs.get("trans_b", attrs.get("transpose_weight", False)))
    )
    group_list = attrs.get("group_list")

    if semantics.has_conv and len(s0) >= 3 and len(s1) >= 3:
        batch = max(1, s0[0])
        c_out = max(1, s1[0])
        spatial_rank = len(s0) - 2
        if len(s1) == len(s0):
            c_in_g = max(1, s1[1])
            k_spatial = s1[2:]
        else:
            c_in_g = 1
            k_spatial = s1[1:]
        spatial_out: list[int] = []
        for ax in range(spatial_rank):
            stride = _conv_attr_item(attrs, ("strides", "stride"), ax, 1)
            pad = _conv_pad_pair(attrs, ("pads", "padding"), ax, spatial_rank)
            dilation = _conv_attr_item(attrs, ("dilations", "dilation"), ax, 1)
            k_ext = k_spatial[ax] if ax < len(k_spatial) else 1
            out_ext = max(
                1,
                (s0[2 + ax] + pad - dilation * (k_ext - 1) - 1) // stride + 1,
            )
            spatial_out.append(out_ext)
        m_dim = math.prod(spatial_out)
        n_dim = c_out
        k_dim = max(1, c_in_g * math.prod(k_spatial))
        return batch, 1, m_dim, n_dim, k_dim, batch * m_dim * n_dim

    if group_list is not None or (len(s0) == 2 and len(s1) == 3):
        m_dim, k_dim = (s0[1], s0[0]) if transpose_a else (s0[0], s0[1])
        groups = (
            len(group_list)
            if isinstance(group_list, (list, tuple)) and len(group_list) > 0
            else max(1, s1[0])
        )
        if len(s1) >= 3:
            if transpose_b or (s1[2] == k_dim and s1[1] != k_dim):
                n_dim = s1[1]
            else:
                n_dim = s1[2]
        else:
            n_dim = s1[0] if transpose_b else s1[-1]
        return 1, groups, m_dim, n_dim, k_dim, m_dim * n_dim

    if len(s0) == 3 and len(s1) == 3:
        batch = max(1, s0[0], s1[0])
        m_dim, k_dim = (s0[2], s0[1]) if transpose_a else (s0[1], s0[2])
        n_dim = s1[1] if transpose_b else s1[2]
        return batch, 1, m_dim, n_dim, k_dim, batch * m_dim * n_dim

    if len(s0) == 2 and len(s1) == 2:
        if (
            semantics.has_subbyte_or_block_quant
            and "in_features" in dims
            and "out_features" in dims
        ):
            m_dim = s0[0]
            k_dim = dims["in_features"]
            n_dim = dims["out_features"]
        else:
            m_dim, k_dim = (s0[1], s0[0]) if transpose_a else (s0[0], s0[1])
            n_dim = s1[0] if transpose_b else s1[1]
        return 1, 1, m_dim, n_dim, k_dim, m_dim * n_dim

    if len(s0) >= 3 and len(s1) == 2:
        batch = math.prod(s0[:-2])
        m_dim = s0[-2]
        if "in_features" in dims and "out_features" in dims:
            k_dim = dims["in_features"]
            n_dim = dims["out_features"]
        else:
            k_dim = s0[-1]
            n_dim = (
                s1[0] if (transpose_b or (s1[1] == k_dim and s1[0] != k_dim)) else s1[1]
            )
        return batch, 1, m_dim, n_dim, k_dim, batch * m_dim * n_dim

    if len(s0) >= 4 and len(s1) == len(s0):
        batch = math.prod(max(a, b) for a, b in zip(s0[:-2], s1[:-2]))
        m_dim, k_dim = (s0[-1], s0[-2]) if transpose_a else (s0[-2], s0[-1])
        n_dim = s1[-2] if transpose_b else s1[-1]
        return batch, 1, m_dim, n_dim, k_dim, batch * m_dim * n_dim

    return None


def _matmul_work(
    semantics: PyTorchOpSemantics,
    dims: Mapping[str, int],
    dtype_bytes: float,
    representative_case: RepresentativeCase | None,
) -> _AnalyticalWork:
    if (
        not semantics.has_conv
        and representative_case is not None
        and len(representative_case.input_shapes) >= 3
        and all(len(sh) == 4 for sh in representative_case.input_shapes[:3])
        and _has_attention_anchor_dims(dims)
    ):
        return _attention_work(semantics, dims, dtype_bytes, representative_case)

    derived = (
        _derive_matmul_dims_from_case(semantics, dims, representative_case)
        if representative_case is not None
        else None
    )
    if derived is not None:
        batch, groups, m_dim, n_dim, k_dim, output_elements = derived
    else:
        k_dim = _require_positive_dim(
            dims,
            ("K", "in_features", "k_dim", "shape_0_dim_1"),
            "matmul",
        )
        m_dim = _positive_dim(dims, ("M", "m_rows", "rows", "T"))
        if m_dim is None:
            activation_elements = _positive_dim(dims, ("total_elements", "total_elems"))
            if activation_elements is not None and activation_elements % k_dim == 0:
                m_dim = activation_elements // k_dim
        if m_dim is None:
            m_dim = _shape_leading_dim_product(dims, "shape_0")
        if m_dim is None:
            raise ValueError(
                "matmul analytical Roofline requires M, rows, total_elements divisible by K, "
                "or a rank-2 activation shape"
            )
        n_dim = _positive_dim(dims, ("out_features", "N", "n_dim", "cols", "D_v", "dv"))
        if n_dim is None:
            if "shape_1_dim_2" in dims:
                n_dim = _positive_dim(dims, ("shape_1_dim_2",))
            else:
                n_dim = _positive_dim(dims, ("shape_1_dim_1", "shape_1_dim_0"))
        if n_dim is None:
            raise ValueError(
                "matmul analytical Roofline requires N, out_features, or weight N dims"
            )
        batch = _dim_or(dims, ("B", "b_dim"), 1)
        groups = _dim_or(dims, ("E", "num_groups"), 1)
        if groups == 1 and "shape_1_dim_2" in dims:
            groups = _dim_or(dims, ("shape_1_dim_0",), 1)
        output_elements = batch * m_dim * n_dim

    activation_elements = batch * m_dim * k_dim
    weight_elements = groups * k_dim * n_dim
    cube_flops = 2 * batch * m_dim * n_dim * k_dim
    vector_flops = 0
    special_ops = 0
    if semantics.has_subbyte_or_block_quant:
        padding_elements = max(1, k_dim // 32) * n_dim * groups
        vector_flops += 2 * (weight_elements + padding_elements)
    if semantics.has_online_exp_max:
        special_ops = output_elements
        vector_flops += special_ops * _TRANSCENDENTAL_OP_EQUIV
    if representative_case is None:
        weight_dtype_bytes = 0.5 if semantics.has_subbyte_or_block_quant else dtype_bytes
        charges = [
            _TrafficCharge(
                element_count=activation_elements,
                element_bytes=dtype_bytes,
                dtype_declared=False,
            ),
            _TrafficCharge(
                element_count=weight_elements,
                element_bytes=weight_dtype_bytes,
                dtype_declared=False,
            ),
            _TrafficCharge(
                element_count=output_elements,
                element_bytes=dtype_bytes,
                dtype_declared=False,
            ),
        ]
    else:
        charges = _declared_input_charges(representative_case, dtype_bytes)
        charges.extend(
            _declared_output_charges(
                representative_case,
                element_count=output_elements,
                fallback_dtype_bytes=dtype_bytes,
            )
        )
    gm_bytes, traffic_basis = _priced_traffic(
        charges, packed=semantics.has_subbyte_or_block_quant
    )
    return _AnalyticalWork(
        cube_flops=cube_flops,
        vector_flops=vector_flops,
        min_gm_bytes=max(1, gm_bytes),
        special_ops=special_ops,
        resolved_dims={
            "M": m_dim,
            "N": n_dim,
            "K": k_dim,
            "B": batch,
            "E": groups,
        },
        work_model=(
            "analytical_matmul_quant"
            if semantics.has_subbyte_or_block_quant
            else "analytical_matmul"
        ),
        traffic_basis=traffic_basis,
        compute_basis="exact_work_count" if vector_flops == 0 else "estimated",
    )


def _reduction_extent(
    reduction_axes: Sequence[int],
    dims: Mapping[str, int],
    representative_case: RepresentativeCase | None,
) -> int | None:
    """Return the product of reduced-axis extents, or None for a full-tensor reduction."""
    if not reduction_axes:
        return None
    if representative_case is not None:
        for shape in representative_case.input_shapes:
            normalized: list[int] = []
            valid = True
            for axis in reduction_axes:
                index = axis + len(shape) if axis < 0 else axis
                if index < 0 or index >= len(shape):
                    valid = False
                    break
                normalized.append(index)
            if not valid:
                continue
            extent = math.prod(shape[index] for index in sorted(set(normalized)))
            if extent > 0:
                return extent
    return _positive_dim(dims, ("d_dim", "D", "cols", "inner"))


def _stream_output_elements(
    semantics: PyTorchOpSemantics,
    elements: int,
    kind: CasePhysicsKind,
    dims: Mapping[str, int],
    representative_case: RepresentativeCase | None,
) -> int:
    """Price the result tensor written back by a stream or reduction work model."""
    if kind != "reduction" or semantics.preserves_input_shape:
        return elements
    extent = _reduction_extent(semantics.reduction_axes, dims, representative_case)
    if extent is None:
        return 1
    return max(1, elements // extent)


def _stream_work(
    semantics: PyTorchOpSemantics,
    dims: Mapping[str, int],
    kind: CasePhysicsKind,
    dtype_bytes: float,
    representative_case: RepresentativeCase | None,
) -> _AnalyticalWork:
    elements = _positive_dim(dims, ("total_elements", "total_elems"))
    if elements is None and "M" in dims and "D" in dims:
        m_dim = dims["M"]
        d_dim = dims["D"]
        if m_dim > 0 and d_dim > 0:
            elements = m_dim * d_dim
    if elements is None and "rows" in dims and "cols" in dims:
        rows = dims["rows"]
        cols = dims["cols"]
        if rows > 0 and cols > 0:
            elements = rows * cols
    if elements is None:
        raise ValueError(
            "analytical Roofline requires total_elements, M x D, or rows x cols anchor dims"
        )
    resolved_dims: dict[str, int] = {"elements": elements}
    if "M" in dims and "D" in dims and dims["M"] > 0 and dims["D"] > 0:
        resolved_dims["M"] = dims["M"]
        resolved_dims["D"] = dims["D"]
    elif "rows" in dims and "cols" in dims and dims["rows"] > 0 and dims["cols"] > 0:
        resolved_dims["rows"] = dims["rows"]
        resolved_dims["cols"] = dims["cols"]
    elif representative_case is not None and representative_case.input_shapes:
        first_shape = representative_case.input_shapes[0]
        if len(first_shape) >= 2 and all(dim > 0 for dim in first_shape):
            resolved_dims["rows"] = math.prod(first_shape[:-1])
            resolved_dims["cols"] = first_shape[-1]
    ops_per_element = 3 if kind == "reduction" else 1
    sfu_multiplier = max(
        1 if semantics.has_online_exp_max else 0,
        semantics.transcendental_op_count,
    )
    special_ops = elements * sfu_multiplier
    vector_flops = ops_per_element * elements + special_ops * _TRANSCENDENTAL_OP_EQUIV
    output_elements = _stream_output_elements(
        semantics, elements, kind, dims, representative_case
    )
    if representative_case is None:
        charges = [
            _TrafficCharge(
                element_count=elements + output_elements,
                element_bytes=dtype_bytes,
                dtype_declared=False,
            )
        ]
    else:
        charges = _declared_input_charges(representative_case, dtype_bytes)
        charges.extend(
            _declared_output_charges(
                representative_case,
                element_count=output_elements,
                fallback_dtype_bytes=dtype_bytes,
            )
        )
    min_gm_bytes, traffic_basis = _priced_traffic(
        charges, packed=semantics.has_subbyte_or_block_quant
    )
    return _AnalyticalWork(
        cube_flops=0,
        vector_flops=vector_flops,
        min_gm_bytes=max(1, min_gm_bytes),
        special_ops=special_ops,
        resolved_dims=resolved_dims,
        work_model=f"analytical_{kind}",
        traffic_basis=traffic_basis,
        compute_basis="estimated",
    )


__all__ = [
    "CasePhysicsRatios",
    "PyTorchOpSemantics",
    "PyTorchSemanticAnalyzer",
    "RepresentativeCase",
]
