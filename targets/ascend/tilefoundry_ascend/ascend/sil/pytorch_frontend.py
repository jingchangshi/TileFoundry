"""PyTorch-golden semantic frontend for physics ratios and analytical HIR blueprints."""

from __future__ import annotations

import ast
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from tilefoundry_ascend.ascend.passes.hir_tir_bridge import (
    BlueprintProvenance,
    HirBlueprintPacket,
    HirRooflineSummary,
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
    {"conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d"}
)
_MATMUL_CALL_NAMES = frozenset(
    {
        "addmm", "addmv", "baddbmm", "bmm", "einsum", "grouped_matmul",
        "linear", "matmul", "matmul_", "mm", "mv", "scaled_dot_product_attention",
        *_CONV_CALL_NAMES,
    }
)
_REDUCTION_CALL_NAMES = frozenset(
    {
        "all", "amax", "amin", "any", "argmax", "argmin", "cross_entropy",
        "cumprod", "cumsum", "group_norm", "layer_norm", "log_softmax", "logsumexp",
        "max", "mean", "norm", "prod", "rms_norm", "scaled_dot_product_attention",
        "softmax", "std", "sum", "var",
    }
)
_TRANSCENDENTAL_CALL_NAMES = frozenset(
    {
        "rsqrt", "silu", "gelu", "sigmoid", "tanh", "sin", "cos", "erf",
        "exp", "exp2", "expm1", "log", "log2", "log10", "softmax", "log_softmax", "pow", "sqrt",
    }
)
_SHAPE_PRESERVING_CALL_NAMES = frozenset(
    {"cumprod", "cumsum", "group_norm", "layer_norm", "log_softmax", "rms_norm", "softmax"}
)
_EXP_CALL_NAMES = frozenset({"exp", "exp2", "expm1"})
_MAX_CALL_NAMES = frozenset(
    {"amax", "clamp", "clamp_", "log_softmax", "max", "maximum", "min", "minimum", "softmax"}
)
_INDIRECT_CALL_NAMES = frozenset(
    {
        "bucketize", "embedding", "gather", "index", "index_add", "index_select",
        "scatter", "scatter_", "scatter_add", "searchsorted", "take", "topk",
    }
)
_SUBBYTE_OR_QUANT_CALL_TOKENS = (
    "dequant", "antiquant", "int4", "uint4", "quint4x2", "q4", "w4",
    "awq", "gptq", "nf4", "fp4", "mxfp", "block_quant",
)
_SUBBYTE_OR_QUANT_IDENT_TOKENS = (
    "dequant", "antiquant", "qweight", "mxscale", "block_size", "blocksize",
    "group_size", "groupsize", "weight_scale", "activation_scale", "int4",
    "uint4", "quint4x2", "fp4", "mxfp", "q4_", "awq", "gptq", "nf4",
)
_SUBBYTE_OR_QUANT_DTYPE_ATTRS = frozenset(
    {
        "int4", "uint4", "quint4x2", "qint8", "float4_e2m1", "float4_e1m2",
        "float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu", "float8_e4m3fnuz", "float8_e5m2fnuz",
    }
)
_STANDARD_8BIT_DTYPE_ATTRS = frozenset({"int8", "uint8"})
_BLOCK_SCALE_IDENT_NAMES = frozenset(
    {"scales", "scale_inv", "zero_point", "zeros", "qzeros", "block_size", "blocksize", "group_size", "groupsize"}
)
_DIM_KEYWORDS = frozenset({"dim", "axis"})
_TRANSCENDENTAL_OP_EQUIV = 8
TrafficBasis = Literal["declared_io_exact", "estimated_shape_dtype"]
ComputeBasis = Literal["exact_work_count", "estimated"]
_ProvenanceFlag = Literal["input", "reduction"]
_PROVENANCE_INPUT: _ProvenanceFlag = "input"
_PROVENANCE_REDUCTION: _ProvenanceFlag = "reduction"
_SHAPE_TRANSPARENT_METHODS = frozenset(
    {"bfloat16", "clone", "contiguous", "cpu", "cuda", "detach", "double", "float", "half", "npu", "to", "type"}
)
_NONLINEAR_STATE_WRAPPERS = frozenset({"tanh", "sigmoid", "relu", "gelu", "silu", "softmax", "exp", "rsqrt"})


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
    has_gemm_contraction: bool | None = None
    has_associative_outer_recurrence: bool = False

    def __post_init__(self) -> None:
        if not self.entry_name.strip():
            raise ValueError("PyTorchOpSemantics.entry_name must be a non-empty string")
        axes = tuple(self.reduction_axes)
        if any(isinstance(a, bool) or not isinstance(a, int) for a in axes):
            raise ValueError("PyTorchOpSemantics.reduction_axes must contain only ints")
        if self.transcendental_op_count < 0:
            raise ValueError("PyTorchOpSemantics.transcendental_op_count must be non-negative")
        object.__setattr__(self, "reduction_axes", axes)
        if self.has_gemm_contraction is None:
            object.__setattr__(self, "has_gemm_contraction", bool(self.has_matmul))

    @property
    def base_axes(self) -> TopologyAxes:
        """Return the base topology axes implied by the recovered semantics."""
        is_gemm = bool(self.has_gemm_contraction)
        coupled = is_gemm and (
            self.has_online_exp_max
            or self.has_subbyte_or_block_quant
            or self.has_indirect_index
            or self.has_recurrent_loop
        )
        compute_engine = "coupled_1c2v" if coupled else ("aic_only" if is_gemm else "aiv_only")
        if self.has_recurrent_loop and not is_gemm:
            memory_ring = "persistent_onchip_state"
        elif coupled and self.has_online_exp_max:
            memory_ring = "bidirectional_cv_ring"
        elif coupled:
            memory_ring = "l0c_ub_epilogue"
        elif is_gemm:
            memory_ring = "gm_l1_l0_fixpipe"
        elif self.has_indirect_index:
            memory_ring = "ub_random_atomic"
        else:
            memory_ring = "gm_ub_stream"
        return TopologyAxes(
            compute_engine=compute_engine,
            memory_ring=memory_ring,
            iteration_regularity="irregular_simt" if self.has_indirect_index else "affine_simd",
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
        if isinstance(self.case_id, bool) or not isinstance(self.case_id, int) or self.case_id < 1:
            raise ValueError("RepresentativeCase.case_id must be a positive integer")
        shapes = tuple(tuple(int(d) for d in sh) for sh in self.input_shapes)
        dtypes = tuple(self.input_dtypes)
        out_dtypes = tuple(self.output_dtypes)
        out_shapes = tuple(tuple(int(d) for d in sh) for sh in self.output_shapes)
        if not shapes:
            raise ValueError("RepresentativeCase.input_shapes must not be empty")
        if len(shapes) != len(dtypes):
            raise ValueError("RepresentativeCase.input_shapes and input_dtypes must have equal length")
        if any(d < 0 for sh in shapes for d in sh):
            raise ValueError("RepresentativeCase.input_shapes must not contain negative dims")
        if any(dt is not None and not dt.strip() for dt in dtypes):
            raise ValueError("RepresentativeCase.input_dtypes must be non-empty strings or None")
        if self.output_dtype is not None and not self.output_dtype.strip():
            raise ValueError("RepresentativeCase.output_dtype must be a non-empty string or None")
        if any(dt is not None and not dt.strip() for dt in out_dtypes):
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
        shapes, dtypes = _pair_case_shapes_and_dtypes(case_entry.get("input_shape"), case_entry.get("dtype"))
        if not shapes:
            raise ValueError(f"case {case_id} requires at least one integer input shape")
        raw_attrs = case_entry.get("attrs")
        output_dtype, output_dtypes = _resolve_proto_output_dtypes(proto, dtypes)
        return cls(
            case_id=case_id,
            input_shapes=shapes,
            input_dtypes=dtypes,
            output_dtype=output_dtype,
            output_dtypes=output_dtypes,
            attrs=dict(raw_attrs) if isinstance(raw_attrs, Mapping) else {},
        )


def _is_int_sequence(raw: object) -> bool:
    return isinstance(raw, (list, tuple)) and len(raw) > 0 and all(isinstance(d, int) and not isinstance(d, bool) for d in raw)


def _pair_case_shapes_and_dtypes(
    raw_shapes: object,
    raw_dtypes: object,
) -> tuple[tuple[tuple[int, ...], ...], tuple[str | None, ...]]:
    if not isinstance(raw_shapes, (list, tuple)):
        return (), ()
    if _is_int_sequence(raw_shapes):
        shape = tuple(int(d) for d in raw_shapes)
        if isinstance(raw_dtypes, str) and raw_dtypes.strip():
            return (shape,), (raw_dtypes.strip(),)
        flat = flatten_str_values(raw_dtypes)
        return (shape,), (flat[0],) if len(flat) == 1 else (None,)

    slots = list(raw_shapes)
    shapes: list[tuple[int, ...]] = []
    paired: list[str | None] = []
    broadcast = raw_dtypes.strip() if isinstance(raw_dtypes, str) and raw_dtypes.strip() else None
    slot_dtypes: list[object] | None = None
    if isinstance(raw_dtypes, (list, tuple)):
        if len(raw_dtypes) == len(slots):
            slot_dtypes = list(raw_dtypes)
        elif len(raw_dtypes) == 1 and isinstance(raw_dtypes[0], str) and raw_dtypes[0].strip():
            broadcast = raw_dtypes[0].strip()

    desynced = False
    for idx, slot in enumerate(slots):
        if slot is None:
            continue
        dt_slot = slot_dtypes[idx] if slot_dtypes is not None else broadcast
        if _is_int_sequence(slot):
            shapes.append(tuple(int(d) for d in slot))
            if isinstance(dt_slot, str) and dt_slot.strip():
                paired.append(dt_slot.strip())
            elif isinstance(dt_slot, (list, tuple)) and len(dt_slot) == 1 and isinstance(dt_slot[0], str) and dt_slot[0].strip():
                paired.append(dt_slot[0].strip())
            elif slot_dtypes is None and broadcast is not None:
                paired.append(broadcast)
            else:
                desynced = True
        elif isinstance(slot, (list, tuple)):
            subs = [tuple(int(d) for d in sub) for sub in slot if _is_int_sequence(sub)]
            if not subs:
                continue
            shapes.extend(subs)
            if isinstance(dt_slot, str) and dt_slot.strip():
                paired.extend([dt_slot.strip()] * len(subs))
            elif isinstance(dt_slot, (list, tuple)):
                sub_dt = [str(item).strip() for item in dt_slot if item is not None]
                if len(sub_dt) == len(subs) and all(sub_dt):
                    paired.extend(sub_dt)
                elif len(sub_dt) == 1 and sub_dt[0]:
                    paired.extend([sub_dt[0]] * len(subs))
                else:
                    desynced = True
            elif slot_dtypes is None and broadcast is not None:
                paired.extend([broadcast] * len(subs))
            else:
                desynced = True

    if not shapes:
        return (), ()
    if desynced or len(paired) != len(shapes):
        return tuple(shapes), (None,) * len(shapes)
    return tuple(shapes), tuple(paired)


def _resolve_proto_output_dtypes(
    proto: Mapping[str, object] | None,
    input_dtypes: Sequence[str | None],
) -> tuple[str | None, tuple[str | None, ...]]:
    if not isinstance(proto, Mapping):
        return None, ()
    op_block = proto["operator"] if "outputs" not in proto and isinstance(proto.get("operator"), Mapping) else proto
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
        elif isinstance(raw_dt, Sequence) and not isinstance(raw_dt, (str, bytes)):
            cands = [str(item).strip() for item in raw_dt if item is not None and str(item).strip()]
            if len(cands) == 1:
                resolved.append(cands[0])
            elif cands:
                resolved.append(next((dt for dt in declared_inputs if dt in cands), None))
            else:
                resolved.append(None)
        else:
            resolved.append(None)
    if not resolved:
        return None, ()
    return next((dt for dt in resolved if dt is not None), None), tuple(resolved)


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
    element_count: int
    element_bytes: float
    dtype_declared: bool


def _call_leaf_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    return func.attr if isinstance(func, ast.Attribute) else None


_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
_CallResolver = Callable[[str, _FunctionNode], _FunctionNode | None]


def _iter_assign_pairs(node: ast.AST) -> tuple[Sequence[ast.expr], ast.expr] | None:
    if isinstance(node, ast.Assign):
        return node.targets, node.value
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None:
        return (node.target,), node.value
    return None


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
        else:
            pending.extend(ast.iter_child_nodes(child))
    return nested


def _function_scopes(
    tree: ast.Module,
):
    top_level: dict[str, _FunctionNode] = {
        n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    class_methods: dict[str, dict[str, _FunctionNode]] = {}
    owner_class: dict[_FunctionNode, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = {
                item.name: item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if methods:
                class_methods[node.name] = methods
                for method in methods.values():
                    owner_class[method] = node.name

    func_aliases: dict[str, str] = {}
    class_instances: dict[str, str] = {}
    for node in tree.body:
        if pair := _iter_assign_pairs(node):
            targets, value = pair
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                if isinstance(value, ast.Name):
                    func_aliases[target.id] = value.id
                elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in class_methods:
                    class_instances[target.id] = value.func.id

    parents: dict[_FunctionNode, _FunctionNode | None] = {}
    nested: dict[_FunctionNode, dict[str, _FunctionNode]] = {}

    def register(node: _FunctionNode, parent: _FunctionNode | None) -> None:
        parents[node] = parent
        nested[node] = (child_fns := _direct_nested_functions(node))
        for child in child_fns.values():
            register(child, node)

    for node in top_level.values():
        register(node, None)
    for methods in class_methods.values():
        for method in methods.values():
            register(method, None)
    return top_level, parents, nested, class_methods, func_aliases, class_instances, owner_class


def _resolve_alias_target(name: str, func_aliases: Mapping[str, str]) -> str:
    current, seen = name, set()
    while current in func_aliases and current not in seen:
        seen.add(current)
        current = func_aliases[current]
    return current


def _local_bindings(function: _FunctionNode) -> tuple[dict[str, str], dict[str, str]]:
    local_aliases: dict[str, str] = {}
    local_calls: dict[str, str] = {}
    for node in _walk_executed_nodes(function):
        if pair := _iter_assign_pairs(node):
            targets, value = pair
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
        if (target := nested[current].get(name)) is not None:
            return target
        current = parents[current]
    if name in top_level:
        return top_level[name]
    return top_level.get(_resolve_alias_target(name, func_aliases)) if func_aliases else None


def _walk_executed_nodes(node: _FunctionNode) -> Iterable[ast.AST]:
    pending: list[ast.AST] = list(ast.iter_child_nodes(node))
    while pending:
        child = pending.pop()
        yield child
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            pending.extend(ast.iter_child_nodes(child))


def _class_entry_targets(cls_name: str, class_methods: Mapping[str, Mapping[str, _FunctionNode]]) -> list[_FunctionNode]:
    methods = class_methods.get(cls_name, {})
    return [methods[m] for m in ("__call__", "forward") if m in methods]


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
                cid = node.func.id
                target = _resolve_function(cid, function, top_level, parents, nested, merged_aliases)
                if target is not None:
                    pending.append(target)
                    continue
                rid = _resolve_alias_target(cid, merged_aliases)
                inst_cls = local_calls.get(cid) or cls_instances.get(cid) or local_calls.get(rid) or cls_instances.get(rid) or (rid if rid in cls_methods else None)
                if inst_cls is not None:
                    pending.extend(_class_entry_targets(inst_cls, cls_methods))
            elif isinstance(node.func, ast.Call) and isinstance(node.func.func, ast.Name):
                cls_name = _resolve_alias_target(node.func.func.id, merged_aliases)
                if cls_name in cls_methods:
                    pending.extend(_class_entry_targets(cls_name, cls_methods))
            elif isinstance(node.func, ast.Attribute):
                attr, recv = node.func.attr, node.func.value
                if isinstance(recv, ast.Name):
                    if recv.id == "self" and current_cls in cls_methods and (m := cls_methods[current_cls].get(attr)) is not None:
                        pending.append(m)
                        continue
                    recv_cls = local_calls.get(recv.id) or cls_instances.get(recv.id) or _resolve_alias_target(recv.id, merged_aliases)
                elif isinstance(recv, ast.Call) and isinstance(recv.func, ast.Name):
                    recv_cls = _resolve_alias_target(recv.func.id, merged_aliases)
                else:
                    recv_cls = None
                if recv_cls in cls_methods and (m := cls_methods[recv_cls].get(attr)) is not None:
                    pending.append(m)
    return tuple(reachable)


def _detect_input_mutation(nodes: Iterable[ast.AST], input_names: frozenset[str]) -> bool:
    if not input_names:
        return False
    for node in nodes:
        if isinstance(node, ast.Call):
            fn = node.func
            if (
                isinstance(fn, ast.Attribute)
                and fn.attr.endswith("_")
                and not fn.attr.startswith("__")
                and isinstance(fn.value, ast.Name)
                and fn.value.id in input_names
            ):
                return True
            if any(
                kw.arg == "out" and isinstance(kw.value, ast.Name) and kw.value.id in input_names
                for kw in node.keywords
            ):
                return True
        if pair := _iter_assign_pairs(node):
            for target in pair[0]:
                if isinstance(target, ast.Name) and target.id in input_names:
                    return True
                if isinstance(target, (ast.Subscript, ast.Attribute)) and isinstance(target.value, ast.Name) and target.value.id in input_names:
                    return True
    return False


def _has_keepdim_true(node: ast.Call) -> bool:
    return any(kw.arg == "keepdim" and isinstance(kw.value, ast.Constant) and kw.value.value is True for kw in node.keywords)


def _expression_provenance(
    node: ast.expr | ast.keyword,
    assignments: Mapping[str, frozenset[_ProvenanceFlag]],
) -> frozenset[_ProvenanceFlag]:
    if isinstance(node, ast.keyword):
        return _expression_provenance(node.value, assignments)
    if isinstance(node, ast.Name):
        return assignments.get(node.id, frozenset())
    if isinstance(node, ast.Call) and _call_leaf_name(node.func) in _REDUCTION_CALL_NAMES and _has_keepdim_true(node):
        return frozenset({_PROVENANCE_REDUCTION})
    if isinstance(node, ast.Lambda):
        return frozenset()
    flags: set[_ProvenanceFlag] = set()
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.expr, ast.keyword)):
            flags |= _expression_provenance(child, assignments)
    return frozenset(flags)


def _returned_expressions(function: _FunctionNode) -> tuple[ast.expr, ...]:
    return tuple(n.value for n in _walk_executed_nodes(function) if isinstance(n, ast.Return) and n.value is not None)


def _assignment_target_names(target: ast.expr) -> frozenset[str]:
    if isinstance(target, ast.Name):
        return frozenset({target.id})
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for elt in target.elts:
            names |= _assignment_target_names(elt)
        return frozenset(names)
    if isinstance(target, (ast.Subscript, ast.Attribute)) and isinstance(target.value, ast.Name):
        return frozenset({target.value.id})
    return frozenset()


def _function_assignments(function: _FunctionNode) -> dict[str, tuple[ast.expr, ...]]:
    defs: dict[str, list[ast.expr]] = {}
    for node in _walk_executed_nodes(function):
        if pair := _iter_assign_pairs(node):
            for target in pair[0]:
                for name in _assignment_target_names(target):
                    defs.setdefault(name, []).append(pair[1])
    return {name: tuple(vals) for name, vals in defs.items()}


def _shape_transparent_base(node: ast.expr) -> ast.expr:
    cur = node
    while isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute) and cur.func.attr in _SHAPE_TRANSPARENT_METHODS:
        cur = cur.func.value
    return cur


def _resolved_return_head(
    node: ast.expr,
    definitions: Mapping[str, tuple[ast.expr, ...]],
    seen: set[str],
) -> ast.expr:
    cur = _shape_transparent_base(node)
    while isinstance(cur, ast.Name):
        vals = definitions.get(cur.id, ())
        if len(vals) != 1 or cur.id in seen:
            break
        seen.add(cur.id)
        cur = _shape_transparent_base(vals[0])
    return cur


def _return_value_is_shape_reduced(
    node: ast.expr,
    definitions: Mapping[str, tuple[ast.expr, ...]],
    seen: set[str],
) -> bool:
    if isinstance(node, (ast.Tuple, ast.List)):
        elts = tuple(e for e in node.elts if not isinstance(e, ast.Starred))
        return bool(elts) and all(_return_value_is_shape_reduced(e, definitions, set(seen)) for e in elts)
    head = _resolved_return_head(node, definitions, seen)
    if isinstance(head, ast.BinOp):
        return _return_value_is_shape_reduced(head.left, definitions, set(seen)) or _return_value_is_shape_reduced(
            head.right, definitions, set(seen)
        )
    if not isinstance(head, ast.Call):
        return False
    leaf = _call_leaf_name(head.func)
    if leaf in _REDUCTION_CALL_NAMES and leaf not in _SHAPE_PRESERVING_CALL_NAMES and not _has_keepdim_true(head):
        return True
    children: list[ast.expr] = [*head.args, *(kw.value for kw in head.keywords)]
    if isinstance(head.func, ast.Attribute):
        children.append(head.func.value)
    return any(_return_value_is_shape_reduced(c, definitions, set(seen)) for c in children)


def _return_path_groups(
    function: _FunctionNode,
    resolve: _CallResolver,
    seen_functions: set[int],
) -> dict[int, tuple[_FunctionNode, list[ast.AST]]]:
    groups: dict[int, tuple[_FunctionNode, list[ast.AST]]] = {}
    if id(function) in seen_functions:
        return groups
    seen_functions.add(id(function))
    defs = _function_assignments(function)
    expanded: set[str] = set()
    nodes: list[ast.AST] = []
    pending: list[ast.expr] = list(_returned_expressions(function))
    while pending:
        for node in ast.walk(pending.pop()):
            nodes.append(node)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in expanded:
                expanded.add(node.id)
                pending.extend(defs.get(node.id, ()))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if (target := resolve(node.func.id, function)) is not None:
                    groups.update(_return_path_groups(target, resolve, seen_functions))
    groups[id(function)] = (function, nodes)
    return groups


def _closure_has_keepdim_broadcast(function: _FunctionNode, nodes: Sequence[ast.AST]) -> bool:
    defs = _function_assignments(function)
    assignments: dict[str, frozenset[_ProvenanceFlag]] = {
        name: frozenset({_PROVENANCE_INPUT}) for name in _entry_parameters(function)
    }
    changed = True
    while changed:
        changed = False
        for name, values in defs.items():
            flags: frozenset[_ProvenanceFlag] = frozenset()
            for v in values:
                flags |= _expression_provenance(v, assignments)
            cur = assignments.get(name, frozenset())
            if (merged := cur | flags) != cur:
                assignments[name] = merged
                changed = True
    for node in nodes:
        if isinstance(node, ast.BinOp):
            left = _expression_provenance(node.left, assignments)
            right = _expression_provenance(node.right, assignments)
            if (_PROVENANCE_INPUT in left and _PROVENANCE_REDUCTION in right) or (
                _PROVENANCE_REDUCTION in left and _PROVENANCE_INPUT in right
            ):
                return True
    return False


def _detect_return_shape_preservation(entry: _FunctionNode, resolve: _CallResolver) -> bool:
    groups = tuple(_return_path_groups(entry, resolve, set()).values())
    returns, defs = _returned_expressions(entry), _function_assignments(entry)
    if not returns or all(_return_value_is_shape_reduced(v, defs, set()) for v in returns):
        return False
    node_groups = tuple(g[1] for g in groups)
    if any(
        isinstance(n, ast.Call) and _call_leaf_name(n.func) in _SHAPE_PRESERVING_CALL_NAMES
        for nodes in node_groups
        for n in nodes
    ):
        return True
    if not all(
        _has_keepdim_true(n)
        for nodes in node_groups
        for n in nodes
        if isinstance(n, ast.Call)
        and (leaf := _call_leaf_name(n.func)) in _REDUCTION_CALL_NAMES
        and leaf not in _SHAPE_PRESERVING_CALL_NAMES
    ):
        return False
    return any(_closure_has_keepdim_broadcast(fn, nodes) for fn, nodes in groups)


def _constant_int(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        op_val = _constant_int(node.operand)
        return None if op_val is None else (-op_val if isinstance(node.op, ast.USub) else op_val)
    return None


def _has_quant_evidence(nodes: Iterable[ast.AST], call_names: Iterable[str]) -> bool:
    if any(tok in cn.lower() for cn in call_names for tok in _SUBBYTE_OR_QUANT_CALL_TOKENS):
        return True
    has_8bit = has_scale = False
    for node in nodes:
        if isinstance(node, ast.Attribute):
            if node.attr in _SUBBYTE_OR_QUANT_DTYPE_ATTRS:
                return True
            if node.attr in _STANDARD_8BIT_DTYPE_ATTRS:
                has_8bit = True
        elif isinstance(node, (ast.Name, ast.arg)):
            ident = (node.id if isinstance(node, ast.Name) else node.arg).lower()
            if any(m in ident for m in _SUBBYTE_OR_QUANT_IDENT_TOKENS):
                return True
            if ident in _BLOCK_SCALE_IDENT_NAMES:
                has_scale = True
    return has_8bit and has_scale


def _positive_dim(dims: Mapping[str, int], names: Sequence[str]) -> int | None:
    return next((v for n in names if (v := dims.get(n)) is not None and v > 0), None)


def _require_positive_dim(dims: Mapping[str, int], names: Sequence[str], what: str) -> int:
    if (val := _positive_dim(dims, names)) is None:
        raise ValueError(f"{what} requires one of anchor dims {tuple(names)} to be positive")
    return val


def _dim_or(dims: Mapping[str, int], names: Sequence[str], default: int) -> int:
    return default if (val := _positive_dim(dims, names)) is None else val


def _shape_leading_dim_product(dims: Mapping[str, int], prefix: str) -> int | None:
    factors: list[int] = []
    idx = 0
    while f"{prefix}_dim_{idx}" in dims:
        if (val := dims[f"{prefix}_dim_{idx}"]) <= 0:
            return None
        factors.append(val)
        idx += 1
    return math.prod(factors[:-1]) if factors else None


def _validated_dtype_bytes(dtype_bytes: float) -> float:
    value = float(dtype_bytes)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("dtype_bytes must be finite and positive")
    return value


def _declared_element_bytes(dtype_name: str | None) -> float | None:
    if dtype_name is None:
        return None
    try:
        eb = dtype_bytes_for(dtype_name)
    except ValueError:
        return None
    return eb if eb >= 1.0 else None


def _make_charge(count: int, dtype_name: str | None, fallback: float) -> _TrafficCharge:
    eb = _declared_element_bytes(dtype_name)
    return _TrafficCharge(
        element_count=count,
        element_bytes=fallback if eb is None else eb,
        dtype_declared=eb is not None,
    )


def _build_case_charges(
    representative_case: RepresentativeCase | None,
    *,
    dtype_bytes: float,
    output_elements: int,
    fallback_charges: Sequence[_TrafficCharge],
) -> list[_TrafficCharge]:
    if representative_case is None:
        return list(fallback_charges)
    charges = [
        _make_charge(math.prod(sh), dt, dtype_bytes)
        for sh, dt in zip(representative_case.input_shapes, representative_case.input_dtypes)
    ]
    if len(representative_case.output_dtypes) > 1:
        for idx, dt in enumerate(representative_case.output_dtypes):
            out_elems = (
                math.prod(representative_case.output_shapes[idx])
                if idx < len(representative_case.output_shapes)
                else output_elements
            )
            charges.append(_make_charge(out_elems, dt, dtype_bytes))
    else:
        dt = representative_case.output_dtypes[0] if representative_case.output_dtypes else representative_case.output_dtype
        charges.append(_make_charge(output_elements, dt, dtype_bytes))
    return charges


def _priced_traffic(charges: Sequence[_TrafficCharge], *, packed: bool) -> tuple[int, TrafficBasis]:
    total = int(math.ceil(sum(c.element_count * c.element_bytes for c in charges)))
    basis: TrafficBasis = (
        "declared_io_exact" if not packed and all(c.dtype_declared for c in charges) else "estimated_shape_dtype"
    )
    return total, basis


def _einsum_is_gemm_contraction(call_node: ast.Call) -> bool:
    if not call_node.args or not (isinstance(first := call_node.args[0], ast.Constant) and isinstance(first.value, str)):
        return True
    eq = first.value.replace(" ", "").replace("...", "")
    if "->" in eq:
        lhs_str, rhs_str = eq.split("->", 1)
    else:
        lhs_str = eq
        letters = [ch for ch in lhs_str if ch.isalpha()]
        rhs_str = "".join(ch for ch in letters if letters.count(ch) == 1)
    terms = lhs_str.split(",")
    if len(terms) != 2:
        return True
    a0, a1, out = ({ch for ch in s if ch.isalpha()} for s in (terms[0], terms[1], rhs_str))
    return bool((a0 & a1) - out) and bool((a0 - a1) & out) and bool((a1 - a0) & out)


def _loop_has_linear_outer_state_recurrence(nodes: Sequence[ast.AST]) -> bool:
    """Detect loop-carried linear state updates with bilinear outer-product terms."""
    for node in nodes:
        if not isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            continue
        body_nodes = [sub for stmt in node.body for sub in ast.walk(stmt)]
        has_add = any(isinstance(s, ast.AugAssign) and isinstance(s.op, (ast.Add, ast.Sub)) for s in body_nodes) or any(
            isinstance(s, ast.Assign) and any(isinstance(op, (ast.Add, ast.Sub)) for op in ast.walk(s.value))
            for s in body_nodes
        )
        if not has_add:
            continue
        loop_calls = {name for s in body_nodes if isinstance(s, ast.Call) and (name := _call_leaf_name(s.func)) is not None}
        has_bilinear = (
            any(isinstance(s, ast.MatMult) for s in body_nodes)
            or bool(loop_calls & {"outer", "einsum", "matmul", "bmm", "mm", "addmm"})
            or ("unsqueeze" in loop_calls and any(isinstance(s, ast.BinOp) and isinstance(s.op, ast.Mult) for s in body_nodes))
        )
        if not has_bilinear:
            continue
        wrapped = False
        for sub in body_nodes:
            if not isinstance(sub, ast.Assign):
                continue
            base_val = _shape_transparent_base(sub.value)
            if not isinstance(base_val, ast.Call) or _call_leaf_name(base_val.func) not in _NONLINEAR_STATE_WRAPPERS:
                continue
            target_names: set[str] = set()
            for t in sub.targets:
                target_names |= _assignment_target_names(t)
            loaded = {i.id for i in ast.walk(base_val) if isinstance(i, ast.Name) and isinstance(i.ctx, ast.Load)}
            has_carry = any(isinstance(i, (ast.Add, ast.Sub, ast.MatMult)) for i in ast.walk(base_val))
            if (target_names & loaded) or has_carry:
                wrapped = True
                break
        if not wrapped:
            return True
    return False


class PyTorchSemanticAnalyzer:
    """Recover PyTorch semantics, physics ratios, and analytical HIR blueprints."""

    def __init__(self, target: AscendTarget | None = None) -> None:
        self.target = AscendTarget("ascend.950pr") if target is None else target
        device = self.target.device
        if not isinstance(device, Ascend950PR):
            raise RuntimeError(f"PyTorchSemanticAnalyzer requires Ascend950PR, got {type(device).__name__}")
        self.device = device
        facts = self.target.get_facts(AscendSpaceFacts)
        if not isinstance(facts, AscendSpaceFacts):
            raise RuntimeError(f"AscendTarget.get_facts returned {type(facts).__name__}, expected AscendSpaceFacts")
        self.facts = facts
        unit_rates = device.unit_flops_per_second
        aic_rates, aiv_rates = unit_rates.get("aic", {}), unit_rates.get("aiv", {})
        aic_by_name = {getattr(k, "name", str(k)): v for k, v in aic_rates.items()}
        aiv_by_name = {getattr(k, "name", str(k)): v for k, v in aiv_rates.items()}
        cube_peak = aic_by_name.get("bf16") or aic_by_name.get("f16") or 0
        vector_peak = aiv_by_name.get("f32") or max(aiv_rates.values(), default=0) or device.aiv_lane_events_per_second
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
        top_level, parents, nested, class_methods, func_aliases, class_instances, owner_class = _function_scopes(
            ast.parse(source)
        )
        entry = top_level.get(entry_name) or top_level.get(_resolve_alias_target(entry_name, func_aliases))
        if entry is None and entry_name in class_instances:
            targets = _class_entry_targets(class_instances[entry_name], class_methods)
            entry = targets[0] if targets else None
        if entry is None and entry_name in class_methods:
            targets = _class_entry_targets(entry_name, class_methods)
            entry = targets[0] if targets else None
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
        nodes = tuple(n for fn in reachable for n in _walk_executed_nodes(fn))
        call_names: set[str] = set()
        reduction_axes: set[int] = set()
        has_matmult = has_loop = has_einsum_gemm = False
        transcendental_op_count = 0
        for node in nodes:
            if isinstance(node, ast.MatMult):
                has_matmult = True
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                has_loop = True
            elif isinstance(node, ast.Call):
                if (cn := _call_leaf_name(node.func)) is not None:
                    call_names.add(cn)
                    if cn in _TRANSCENDENTAL_CALL_NAMES:
                        transcendental_op_count += 1
                    if cn == "einsum" and _einsum_is_gemm_contraction(node):
                        has_einsum_gemm = True
                if cn in _REDUCTION_CALL_NAMES:
                    for kw in node.keywords:
                        if kw.arg in _DIM_KEYWORDS and (ax := _constant_int(kw.value)) is not None:
                            reduction_axes.add(ax)

        has_online_norm = bool(call_names & {"log_softmax", "softmax", "scaled_dot_product_attention"})
        has_exp = bool(call_names & _EXP_CALL_NAMES) or has_online_norm
        has_max = bool(call_names & _MAX_CALL_NAMES) or ("scaled_dot_product_attention" in call_names)

        def resolve_call(name: str, scope: _FunctionNode) -> _FunctionNode | None:
            return _resolve_function(name, scope, top_level, parents, nested, func_aliases)

        has_gemm = has_matmult or bool(call_names & (_MATMUL_CALL_NAMES - {"einsum", "mv", "addmv"})) or has_einsum_gemm
        return PyTorchOpSemantics(
            entry_name=entry_name,
            has_matmul=has_matmult or bool(call_names & _MATMUL_CALL_NAMES),
            has_gemm_contraction=has_gemm,
            has_reduction=bool(call_names & _REDUCTION_CALL_NAMES),
            reduction_axes=tuple(sorted(reduction_axes)),
            has_online_exp_max=has_exp and has_max,
            has_indirect_index=bool(call_names & _INDIRECT_CALL_NAMES),
            has_subbyte_or_block_quant=_has_quant_evidence(nodes, call_names),
            has_recurrent_loop=has_loop,
            has_associative_outer_recurrence=bool(has_loop and _loop_has_linear_outer_state_recurrence(nodes)),
            mutates_inputs=_detect_input_mutation(nodes, _entry_parameters(entry)),
            preserves_input_shape=_detect_return_shape_preservation(entry, resolve_call),
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
        fallback_eb = _validated_dtype_bytes(dtype_bytes)
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
        elem_b = dtype_bytes_for(dtypes[0]) if dtypes else fallback_eb
        axes = semantics.base_axes
        kind = infer_case_physics_kind(axes, has_reduction=semantics.has_reduction)
        dims = infer_case_dims_structural(shapes, attrs, axes=axes, uses_cube=axes.uses_cube)
        ctx = build_case_physics_context(
            kind,
            elem_b,
            self.facts,
            reduction_axis=semantics.reduction_axes[0] if semantics.reduction_axes else -1,
            strided_gather=semantics.has_indirect_index,
            has_empty_group=has_empty_group_attrs(attrs),
        )
        return compute_case_physics_ratios(derive_case_physics_input(case_id, shapes, attrs, dims, ctx))

    def build_universal_blueprint(
        self,
        semantics: PyTorchOpSemantics,
        anchor_dims: Mapping[str, int],
        *,
        dtype_bytes: float = 2.0,
        representative_case: RepresentativeCase | None = None,
    ) -> HirBlueprintPacket:
        """Build a semantic-first blueprint anchored to real dimensions and PyTorch semantics."""
        eb = _validated_dtype_bytes(dtype_bytes)
        dims = {str(k): int(v) for k, v in anchor_dims.items()}
        work = self._estimate_analytical_work(semantics, dims, eb, representative_case)
        compute_flops = work.cube_flops + work.vector_flops
        cube_us = (work.cube_flops / self.cube_peak_flops * 1e6) if work.cube_flops > 0 else 0.0
        vec_us = (work.vector_flops / self.vector_peak_flops * 1e6) if work.vector_flops > 0 else 0.0
        compute_floor_us = vec_us if work.cube_flops == 0 else max(cube_us, vec_us)
        memory_floor_us = work.min_gm_bytes / self.hbm_bytes_per_second * 1e6
        ideal_us = max(compute_floor_us, memory_floor_us)
        if ideal_us <= 0.0:
            bound = "none"
        elif abs(compute_floor_us - memory_floor_us) <= 1e-12 * max(1.0, ideal_us):
            bound = "balanced"
        else:
            bound = "memory" if memory_floor_us > compute_floor_us else "compute"
        ai = round(compute_flops / float(work.min_gm_bytes), 4) if work.min_gm_bytes > 0 else 0.0
        roofline = HirRooflineSummary(
            anchor_dims=dict(work.resolved_dims),
            compute_flops=compute_flops,
            min_gm_bytes=work.min_gm_bytes,
            arithmetic_intensity=ai,
            compute_floor_us=compute_floor_us,
            memory_floor_us=memory_floor_us,
            ideal_roofline_us=ideal_us,
            dominant_bound=bound,
            requested_anchor_dims=dict(dims),
            resolved_hir_dims=dict(work.resolved_dims),
            shape_exact_match=True,
            fallback_reason=None,
            cube_flops=work.cube_flops,
            vector_flops=work.vector_flops,
            special_ops=work.special_ops,
            cube_floor_us=cube_us,
            vec_floor_us=vec_us,
        )
        return HirBlueprintPacket(
            entry_name=semantics.entry_name,
            roofline=roofline,
            lowering_diagnostics=("analytical_pytorch_frontend",),
            requested_anchor_dims=dict(dims),
            resolved_hir_dims=dict(work.resolved_dims),
            shape_exact_match=True,
            fallback_reason=None,
            provenance=BlueprintProvenance(
                semantic_source="pytorch_golden",
                traffic_basis=work.traffic_basis,
                coordinate_scope="representative_anchor",
            ),
        )

    build_blueprint = build_universal_blueprint

    def _estimate_analytical_work(
        self,
        semantics: PyTorchOpSemantics,
        dims: Mapping[str, int],
        dtype_bytes: float,
        representative_case: RepresentativeCase | None,
    ) -> _AnalyticalWork:
        kind = infer_case_physics_kind(semantics.base_axes, has_reduction=semantics.has_reduction)
        if kind == "attention":
            return _attention_work(semantics, dims, dtype_bytes, representative_case)
        if kind == "matmul":
            if semantics.has_indirect_index and _has_attention_anchor_dims(dims):
                return _attention_work(semantics, dims, dtype_bytes, representative_case)
            return _matmul_work(semantics, dims, dtype_bytes, representative_case)
        return _stream_work(semantics, dims, kind, dtype_bytes, representative_case)


def _has_attention_anchor_dims(dims: Mapping[str, int]) -> bool:
    return (
        _positive_dim(dims, ("S_q", "S", "s_dim")) is not None
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
    seq_idx = 2 if layout == "BNSD" else 1
    s_kv = _positive_dim(dims, ("S_kv", "s_kv")) or _positive_dim(
        dims, (f"shape_1_dim_{seq_idx}", "shape_1_dim_1", "shape_1_dim_0")
    ) or s_q
    d_nope, d_rope = _positive_dim(dims, ("D_nope", "d_nope")), _positive_dim(dims, ("D_rope", "d_rope"))
    d_qk = _positive_dim(dims, ("D_qk", "D", "d_dim", "dk"))
    if d_qk is None and d_nope is not None and d_rope is not None:
        d_qk = d_nope + d_rope
    if d_qk is None:
        raise ValueError("attention analytical Roofline requires D_qk, D, or D_nope+D_rope anchor dims")
    d_v = _positive_dim(dims, ("D_v", "dv", "d_v")) or d_nope or d_qk
    score_pairs = batch * num_q_heads * s_q * s_kv
    cube_flops = 2 * score_pairs * (d_qk + d_v)
    special_ops = score_pairs if semantics.has_online_exp_max else 0
    vector_flops = 5 * score_pairs + special_ops * _TRANSCENDENTAL_OP_EQUIV
    out_elems = batch * num_q_heads * s_q * d_v
    gm_elems = batch * (num_q_heads * s_q * d_qk + num_kv_heads * s_kv * (d_qk + d_v) + num_q_heads * s_q * d_v)
    charges = _build_case_charges(
        representative_case,
        dtype_bytes=dtype_bytes,
        output_elements=out_elems,
        fallback_charges=[_TrafficCharge(element_count=gm_elems, element_bytes=dtype_bytes, dtype_declared=False)],
    )
    min_gm_bytes, traffic_basis = _priced_traffic(charges, packed=semantics.has_subbyte_or_block_quant)
    return _AnalyticalWork(
        cube_flops=cube_flops,
        vector_flops=vector_flops,
        min_gm_bytes=max(1, min_gm_bytes),
        special_ops=special_ops,
        resolved_dims={"B": batch, "S_q": s_q, "S_kv": s_kv, "N_q": num_q_heads, "N_kv": num_kv_heads, "D_qk": d_qk, "D_v": d_v},
        work_model="analytical_attention_online" if special_ops else "analytical_attention",
        traffic_basis=traffic_basis,
        compute_basis="estimated",
    )


def _conv_attr_item(attrs: Mapping[str, object], keys: Sequence[str], axis: int, default: int) -> int:
    for key in keys:
        if key not in attrs:
            continue
        raw = attrs[key]
        if isinstance(raw, int) and not isinstance(raw, bool):
            return max(1, int(raw))
        if isinstance(raw, (list, tuple)) and axis < len(raw) and isinstance(raw[axis], int) and not isinstance(raw[axis], bool):
            return max(1, int(raw[axis]))
    return default


def _conv_pad_pair(attrs: Mapping[str, object], keys: Sequence[str], axis: int, spatial_rank: int) -> int:
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
    attrs, s0, s1 = representative_case.attrs, shapes[0], shapes[1]
    ta = bool(attrs.get("transpose_a", attrs.get("trans_a", False)))
    tb = bool(attrs.get("transpose_b", attrs.get("trans_b", attrs.get("transpose_weight", False))))
    group_list = attrs.get("group_list")

    if semantics.has_conv and len(s0) >= 3 and len(s1) >= 3:
        batch, c_out, rank = max(1, s0[0]), max(1, s1[0]), len(s0) - 2
        c_in_g, k_sp = (max(1, s1[1]), s1[2:]) if len(s1) == len(s0) else (1, s1[1:])
        sp_out = [
            max(
                1,
                (
                    s0[2 + ax]
                    + _conv_pad_pair(attrs, ("pads", "padding"), ax, rank)
                    - _conv_attr_item(attrs, ("dilations", "dilation"), ax, 1) * ((k_sp[ax] if ax < len(k_sp) else 1) - 1)
                    - 1
                )
                // _conv_attr_item(attrs, ("strides", "stride"), ax, 1)
                + 1,
            )
            for ax in range(rank)
        ]
        m_dim = math.prod(sp_out)
        return batch, 1, m_dim, c_out, max(1, c_in_g * math.prod(k_sp)), batch * m_dim * c_out

    if group_list is not None or (len(s0) == 2 and len(s1) == 3):
        m_dim, k_dim = (s0[1], s0[0]) if ta else (s0[0], s0[1])
        groups = len(group_list) if isinstance(group_list, (list, tuple)) and group_list else max(1, s1[0])
        if len(s1) >= 3:
            n_dim = s1[1] if (tb or (s1[2] == k_dim and s1[1] != k_dim)) else s1[2]
        else:
            n_dim = s1[0] if tb else s1[-1]
        return 1, groups, m_dim, n_dim, k_dim, m_dim * n_dim

    if len(s0) == 3 and len(s1) == 3:
        batch = max(1, s0[0], s1[0])
        m_dim, k_dim = (s0[2], s0[1]) if ta else (s0[1], s0[2])
        n_dim = s1[1] if tb else s1[2]
        return batch, 1, m_dim, n_dim, k_dim, batch * m_dim * n_dim

    if len(s0) == 2 and len(s1) == 2:
        if semantics.has_subbyte_or_block_quant and "in_features" in dims and "out_features" in dims:
            m_dim, k_dim, n_dim = s0[0], dims["in_features"], dims["out_features"]
        else:
            m_dim, k_dim = (s0[1], s0[0]) if ta else (s0[0], s0[1])
            n_dim = s1[0] if tb else s1[1]
        return 1, 1, m_dim, n_dim, k_dim, m_dim * n_dim

    if len(s0) >= 3 and len(s1) == 2:
        batch, m_dim = math.prod(s0[:-2]), s0[-2]
        if "in_features" in dims and "out_features" in dims:
            k_dim, n_dim = dims["in_features"], dims["out_features"]
        else:
            k_dim = s0[-1]
            n_dim = s1[0] if (tb or (s1[1] == k_dim and s1[0] != k_dim)) else s1[1]
        return batch, 1, m_dim, n_dim, k_dim, batch * m_dim * n_dim

    if len(s0) >= 4 and len(s1) == len(s0):
        batch = math.prod(max(a, b) for a, b in zip(s0[:-2], s1[:-2]))
        m_dim, k_dim = (s0[-1], s0[-2]) if ta else (s0[-2], s0[-1])
        return batch, 1, m_dim, s1[-2] if tb else s1[-1], k_dim, batch * m_dim * (s1[-2] if tb else s1[-1])

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

    derived = _derive_matmul_dims_from_case(semantics, dims, representative_case) if representative_case else None
    if derived is not None:
        batch, groups, m_dim, n_dim, k_dim, out_elems = derived
    else:
        k_dim = _require_positive_dim(dims, ("K", "in_features", "k_dim", "shape_0_dim_1"), "matmul")
        m_dim = _positive_dim(dims, ("M", "m_rows", "rows", "T"))
        if m_dim is None and (act_e := _positive_dim(dims, ("total_elements", "total_elems"))) and act_e % k_dim == 0:
            m_dim = act_e // k_dim
        if m_dim is None:
            m_dim = _shape_leading_dim_product(dims, "shape_0")
        if m_dim is None:
            raise ValueError(
                "matmul analytical Roofline requires M, rows, total_elements divisible by K, or a rank-2 activation shape"
            )
        n_dim = _positive_dim(dims, ("out_features", "N", "n_dim", "cols", "D_v", "dv")) or (
            _positive_dim(dims, ("shape_1_dim_2",)) if "shape_1_dim_2" in dims else _positive_dim(dims, ("shape_1_dim_1", "shape_1_dim_0"))
        )
        if n_dim is None:
            raise ValueError("matmul analytical Roofline requires N, out_features, or weight N dims")
        batch = _dim_or(dims, ("B", "b_dim"), 1)
        groups = _dim_or(dims, ("E", "num_groups"), 1)
        if groups == 1 and "shape_1_dim_2" in dims:
            groups = _dim_or(dims, ("shape_1_dim_0",), 1)
        out_elems = batch * m_dim * n_dim

    act_elems, w_elems = batch * m_dim * k_dim, groups * k_dim * n_dim
    cube_flops = 2 * batch * m_dim * n_dim * k_dim
    vector_flops, special_ops = 0, 0
    if semantics.has_subbyte_or_block_quant:
        vector_flops += 2 * (w_elems + max(1, k_dim // 32) * n_dim * groups)
    if semantics.has_online_exp_max:
        special_ops = out_elems
        vector_flops += special_ops * _TRANSCENDENTAL_OP_EQUIV
    w_eb = 0.5 if semantics.has_subbyte_or_block_quant else dtype_bytes
    charges = _build_case_charges(
        representative_case,
        dtype_bytes=dtype_bytes,
        output_elements=out_elems,
        fallback_charges=[
            _TrafficCharge(element_count=act_elems, element_bytes=dtype_bytes, dtype_declared=False),
            _TrafficCharge(element_count=w_elems, element_bytes=w_eb, dtype_declared=False),
            _TrafficCharge(element_count=out_elems, element_bytes=dtype_bytes, dtype_declared=False),
        ],
    )
    gm_bytes, traffic_basis = _priced_traffic(charges, packed=semantics.has_subbyte_or_block_quant)
    return _AnalyticalWork(
        cube_flops=cube_flops,
        vector_flops=vector_flops,
        min_gm_bytes=max(1, gm_bytes),
        special_ops=special_ops,
        resolved_dims={"M": m_dim, "N": n_dim, "K": k_dim, "B": batch, "E": groups},
        work_model="analytical_matmul_quant" if semantics.has_subbyte_or_block_quant else "analytical_matmul",
        traffic_basis=traffic_basis,
        compute_basis="exact_work_count" if vector_flops == 0 else "estimated",
    )


def _reduction_extent(
    reduction_axes: Sequence[int],
    dims: Mapping[str, int],
    representative_case: RepresentativeCase | None,
) -> int | None:
    if not reduction_axes:
        return None
    if representative_case is not None:
        for shape in representative_case.input_shapes:
            norm = [ax + len(shape) if ax < 0 else ax for ax in reduction_axes]
            if all(0 <= idx < len(shape) for idx in norm):
                if (extent := math.prod(shape[idx] for idx in sorted(set(norm)))) > 0:
                    return extent
    return _positive_dim(dims, ("d_dim", "D", "cols", "inner"))


def _stream_output_elements(
    semantics: PyTorchOpSemantics,
    elements: int,
    kind: CasePhysicsKind,
    dims: Mapping[str, int],
    representative_case: RepresentativeCase | None,
) -> int:
    if kind != "reduction" or semantics.preserves_input_shape:
        return elements
    extent = _reduction_extent(semantics.reduction_axes, dims, representative_case)
    return 1 if extent is None else max(1, elements // extent)


def _stream_work(
    semantics: PyTorchOpSemantics,
    dims: Mapping[str, int],
    kind: CasePhysicsKind,
    dtype_bytes: float,
    representative_case: RepresentativeCase | None,
) -> _AnalyticalWork:
    elements = _positive_dim(dims, ("total_elements", "total_elems"))
    if elements is None and dims.get("M", 0) > 0 and dims.get("D", 0) > 0:
        elements = dims["M"] * dims["D"]
    if elements is None and dims.get("rows", 0) > 0 and dims.get("cols", 0) > 0:
        elements = dims["rows"] * dims["cols"]
    d_col = _positive_dim(dims, ("D", "d_dim", "cols"))
    seq_or_m = _positive_dim(dims, ("S", "s_dim", "T", "t_dim", "seq_len", "M", "m_rows", "rows"))
    n_depth, batch = _positive_dim(dims, ("N", "n_depth", "n_layers")), _positive_dim(dims, ("B", "batch"))
    if elements is None and d_col is not None and seq_or_m is not None:
        elements = (n_depth or 1) * (batch or 1) * seq_or_m * d_col
    if elements is None:
        raise ValueError("analytical Roofline requires total_elements, M x D, or rows x cols anchor dims")
    resolved_dims: dict[str, int] = {"elements": elements}
    if dims.get("M", 0) > 0 and dims.get("D", 0) > 0:
        resolved_dims.update({"M": dims["M"], "D": dims["D"]})
    elif dims.get("rows", 0) > 0 and dims.get("cols", 0) > 0:
        resolved_dims.update({"rows": dims["rows"], "cols": dims["cols"]})
    elif representative_case is not None and representative_case.input_shapes:
        sh0 = representative_case.input_shapes[0]
        if len(sh0) >= 2 and all(d > 0 for d in sh0):
            resolved_dims.update({"rows": math.prod(sh0[:-1]), "cols": sh0[-1]})
    elif d_col is not None and seq_or_m is not None:
        resolved_dims.update({"rows": (n_depth or 1) * (batch or 1) * seq_or_m, "cols": d_col})

    sfu_mult = max(1 if semantics.has_online_exp_max else 0, semantics.transcendental_op_count)
    special_ops = elements * sfu_mult
    vector_flops = (3 if kind == "reduction" else 1) * elements + special_ops * _TRANSCENDENTAL_OP_EQUIV
    out_elems = _stream_output_elements(semantics, elements, kind, dims, representative_case)
    charges = _build_case_charges(
        representative_case,
        dtype_bytes=dtype_bytes,
        output_elements=out_elems,
        fallback_charges=[_TrafficCharge(element_count=elements + out_elems, element_bytes=dtype_bytes, dtype_declared=False)],
    )
    min_gm_bytes, traffic_basis = _priced_traffic(charges, packed=semantics.has_subbyte_or_block_quant)
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


__all__ = ["CasePhysicsRatios", "PyTorchOpSemantics", "PyTorchSemanticAnalyzer", "RepresentativeCase"]
