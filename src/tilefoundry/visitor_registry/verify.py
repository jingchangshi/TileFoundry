"""Verify HIR and TIR functions, bindings, calls, and contextual placement.

The checked rule set is defined by
[tir §1.3](docs/spec/tir.md#13-primfunction).
"""

from __future__ import annotations

import dataclasses
from contextlib import ExitStack, contextmanager
from typing import Iterable

from tilefoundry.inspection.tir_printer import TirPrinter
from tilefoundry.ir.core import Expr, Op, Var, VerifyError
from tilefoundry.ir.core.execution import declared_mask_width, is_structural
from tilefoundry.ir.core.expr import Call, Constant
from tilefoundry.ir.core.metadata import diagnostic_location
from tilefoundry.ir.core.module import Module, ModuleList, resolve_symbol
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.core.param_def import MemoryEffect
from tilefoundry.ir.core.execution import (
    declared_kinds,
    execution_kind,
    region_requirement,
)
from tilefoundry.ir.hir.execution_region import ExecutionRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.hir.specialize import canonical_specialization_signature
from tilefoundry.ir.module_utils import list_call_paths
from tilefoundry.ir.pattern import RangePattern, locate_dim_var
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.memory import AllocTensor as AllocTensorOp
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.shape import ShapeOf
from tilefoundry.ir.tir.stmt import Stmt
from tilefoundry.ir.tir.stmts import (
    ExecutionScope,
    Evaluate,
    For,
    If,
    LetStmt,
    MeshScope,
    Return,
    Sequential,
    While,
)
from tilefoundry.ir.tir.symbol_ref import SymbolRef
from tilefoundry.ir.types import DType, StorageKind, TensorType, UnitType
from tilefoundry.ir.types.callable_type import callable_type_for_prim_function
from tilefoundry.ir.types.dim import (
    DimAdd,
    DimFloorDiv,
    DimMax,
    DimMin,
    DimMod,
    DimMul,
    DimSub,
    DimVar,
)
from tilefoundry.ir.types.layout import ComposedLayout, LayoutBase, flatten
from tilefoundry.ir.types.mesh import Mesh, make_mesh
from tilefoundry.ir.types.shard_layout import ShardLayout
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.types.tensor_type import TupleType
from tilefoundry.ir.types.utils import static_dim_value, tensor_types
from tilefoundry.ir.visitor import ExprVisitor, collect_exprs, expr_children
from tilefoundry.target import CpuTarget
from tilefoundry.utils.spec_ref import spec_ref_render
from tilefoundry.visitor_registry import verify_stmt_registry
from tilefoundry.visitor_registry.contexts import TypeInferContext, VerifyContext
from tilefoundry.visitor_registry.instruction import ScheduleRefusal, check_call
from tilefoundry.visitor_registry.typeinfer import inference_type

_PRIM_FUNCTION = "[tir §1.3](docs/spec/tir.md#13-primfunction)"


def verify_region_isolated(region: LoopRegion | MeshRegion, ctx=None) -> None:
    """Ensure a region body reaches captured values only through its params."""
    if len(region.params) != len(region.args):
        message = f"region has {len(region.params)} params but {len(region.args)} args"
        if ctx is not None:
            ctx.error(region, message)
        raise VerifyError(f"{type(region).__name__}: {message}")
    arg_ids = {id(arg) for arg in region.args}
    roots = (
        (region.body, *region.yield_values) if isinstance(region, LoopRegion) else (region.body,)
    )
    leaked = arg_ids.intersection(id(expr) for root in roots for expr in collect_exprs(root))
    if not leaked:
        return
    message = (
        "region is not isolated: body reads an args value directly; reference its param instead"
    )
    if ctx is not None:
        ctx.error(region, message)
    raise VerifyError(f"{type(region).__name__}: {message}")




def execution_kind_declaration(kind: str):
    """The declaration named *kind*, or a precise refusal when none declares it.

    A region that names no declared mode has no boundary to check, so an unknown
    kind is refused rather than treated as an unconstrained region. The same is
    true of a declaration that states no mask width: a mode whose instructions
    cannot be bounded cannot be checked, and passing it would be a silent
    admission of every width.
    """
    declared = execution_kind(kind)
    if declared is None:
        raise VerifyError(
            f"execution region names undeclared kind {kind!r}; declared kinds are "
            f"{tuple(item.name for item in declared_kinds())!r}"
        )
    return declared


def current_execution_kinds(ctx) -> tuple:
    """The execution modes in force at *ctx*, innermost first."""
    if ctx is None:
        return ()
    held = [
        ctx_.owner.declaration()
        for ctx_ in ctx.ancestors()
        if isinstance(ctx_.owner, ExecutionRegion)
    ]
    return tuple(held)


def _mesh_topology_names(mesh) -> tuple[str, ...]:
    """The level names of *mesh*, or an empty tuple when it states none."""
    if mesh is None:
        return ()
    return tuple(
        str(getattr(topology, "name", topology)) for topology in getattr(mesh, "topologies", ())
    )


def verify_execution_boundary(region, ctx) -> None:
    """Check one execution region states a mode that may be entered here.

    Everything here is about the boundary; the body is the body walker s
    question. Four refusals, each of which would otherwise be a silent yes:

    - a kind nothing declares, which has no rules to hold the region to;
    - a declaration with no mask width, whose instructions cannot be bounded;
    - a declaration with no required topology, which names no domain at all;
    - an enclosing mode that is not nestable, which for the parser and for
      this check alike means no execution region may be entered inside it.

    The domain itself is read off the mesh *in force at the region*, never off
    the module declaration: a module that declares an AIV level says what the
    program may use, not that every region in it runs there. A region with no
    mesh in force therefore states a mode with nowhere to run and is refused,
    which is also why a module declaring no topologies cannot slip through as
    an empty set that trivially contains the requirement.
    """
    if ctx is None:
        raise VerifyError(
            "ExecutionRegion: this boundary cannot be checked without a lexical "
            "context; the kind, the domain in force and the enclosing modes are all "
            "properties of where the region sits. Verify it from a walk that has one."
        )
    declared = execution_kind_declaration(region.kind)
    if declared.mask_width_bits is None:
        ctx.error(
            region,
            f"execution region {declared.name!r} declares no mask width; "
            "a mode whose element width is unstated cannot bound its instructions",
        )
    if not declared.required_topologies:
        ctx.error(
            region,
            f"execution region {declared.name!r} declares no required topology; "
            "a mode with no domain is not a mode this program can be checked against",
        )
    for outer in current_execution_kinds(ctx):
        if not outer.nestable:
            ctx.error(
                region,
                f"execution region {declared.name!r} is entered inside "
                f"{outer.name!r}, which is not nestable; no execution region may be "
                "opened inside it",
            )
    in_force = _mesh_topology_names(getattr(ctx, "current_mesh", None))
    if not in_force:
        ctx.error(
            region,
            f"execution region {declared.name!r} requires topology "
            f"{declared.required_topologies!r} but no mesh is in force at the region; "
            "a mode needs a domain to run in, not only a declaration",
        )
    for topology in declared.required_topologies:
        if topology not in in_force:
            ctx.error(
                region,
                f"execution region {declared.name!r} requires topology {topology!r}, "
                f"which the mesh in force does not have; it has {in_force!r}",
            )


def verify_value_layouts(expr):
    """Require concrete tensor layouts on values, independently of annotations."""

    def layout(value, node):
        if value is None:
            where = diagnostic_location(node)
            raise VerifyError(
                f"value {getattr(node, 'name', type(node).__name__)!r} has no tensor layout"
                + (f"\n  at {where}" if where else "")
            )
        if isinstance(value, ShardLayout):
            layout(value.layout, node)
        elif isinstance(value, ComposedLayout):
            for child in (value.outer, value.inner):
                if isinstance(child, LayoutBase):
                    layout(child, node)

    for value in (expr, *collect_exprs(expr)):
        bound = value.params if isinstance(value, (LoopRegion, MeshRegion)) else ()
        if isinstance(value, LoopRegion):
            bound = (*bound, value.induction_var)
        for node in (value, *bound):
            for tensor in tensor_types(node.type):
                layout(tensor.layout, node)


def verify_function(fn: Function, *, module=None) -> None:
    """Verify HIR parameters, symbolic dimensions, body, and variants."""
    for p in fn.params:
        if not isinstance(p, Var):
            raise VerifyError(f"hir Function {fn.name!r}: params must be Vars")
    _verify_signature_dim_vars(fn)
    if fn.variants:
        if fn.body is not None:
            raise VerifyError(
                f"hir Function {fn.name!r}: a function with variants must have "
                f"no body (a dispatch prototype's body is None / `pass`)"
            )
        _verify_variants(fn, module=module)
        return
    if fn.body is None:
        return
    if not isinstance(fn.body, Expr):
        raise VerifyError(
            f"hir Function {fn.name!r}: body must be an Expr or None, got {type(fn.body).__name__}"
        )
    _reject_stmt_nodes(fn.body)
    for value in (*fn.params, fn.body):
        verify_value_layouts(value)


def _verify_variants(base: Function, *, module=None) -> None:
    """Verify a dispatch prototype's variants and their envelope partition."""
    base_param_types = tuple(p.annotation for p in base.params)
    sigs: dict[str, Function] = {}
    for v in base.variants:
        if v.variants:
            raise VerifyError(
                f"hir Function {v.name!r}: a variant must not carry variants "
                f"(specialization nesting is one level)"
            )
        if not v.specializations:
            raise VerifyError(
                f"hir Function {base.name!r}: a variant must carry a specialization pattern"
            )
        if v.body is None:
            raise VerifyError(
                f"hir Function {base.name!r}: a variant must have a real body, not `pass`"
            )
        if v.name != base.name:
            raise VerifyError(f"variant name {v.name!r} does not match base {base.name!r}")
        if tuple(p.annotation for p in v.params) != base_param_types or (
            v.return_annotation != base.return_annotation
        ):
            raise VerifyError(
                f"hir Function {base.name!r}: variant signature must match the base signature"
            )
        sig = canonical_specialization_signature(v.specializations)
        if sig in sigs:
            raise VerifyError(
                f"hir Function {base.name!r}: duplicate variant canonical signature {sig!r}"
            )
        sigs[sig] = v
        verify_function(v, module=module)
    _verify_partition(base)


def _verify_partition(base: Function) -> None:
    """Verify partition.

    Check the variants' ranges partition the base DimVar envelope —
    pairwise disjoint and jointly complete over closed specialization ranges.
    """
    dim_vars: set[str] = set()
    ranges: list[tuple[int, int]] = []
    for v in base.variants:
        for pat in v.specializations:
            if not isinstance(pat, RangePattern):
                raise VerifyError(
                    f"hir Function {base.name!r}: only RangePattern is "
                    f"supported for dispatch (got {type(pat).__name__})"
                )
            dim_vars.add(pat.dim_var)
            ranges.append((pat.lo, pat.hi))
    if len(dim_vars) != 1:
        raise VerifyError(
            f"hir Function {base.name!r}: variants must dispatch on a single "
            f"DimVar, got {sorted(dim_vars)}"
        )
    envelope = _collect_param_dim_vars(base).get(next(iter(dim_vars)))
    if envelope is None:
        raise VerifyError(
            f"hir Function {base.name!r}: dispatch DimVar "
            f"{next(iter(dim_vars))!r} is not reachable from an input parameter"
        )
    lo, hi = envelope

    cursor = lo
    for rlo, rhi in sorted(ranges):
        if rlo != cursor:
            raise VerifyError(
                f"hir Function {base.name!r}: variant ranges do not partition "
                f"envelope [{lo}, {hi}] — gap or overlap at {rlo} (expected "
                f"{cursor})"
            )
        cursor = rhi + 1
    if cursor != hi + 1:
        raise VerifyError(
            f"hir Function {base.name!r}: variant ranges cover "
            f"[{lo}, {cursor - 1}] but the envelope is [{lo}, {hi}]"
        )


def _ingest_dim_vars(ty: object, bounds: dict[str, tuple[int, int]]) -> None:
    """Collect consistent ``DimVar`` bounds recursively from a signature type.

    Tensor tuples and dimension-expression calls are traversed. Reusing a name
    with different bounds raises ``VerifyError``.

    See [types §4](docs/spec/types.md#4-dim--symbolic-shape-dimensions).
    """
    if isinstance(ty, TensorType):
        for entry in ty.shape:
            _ingest_shape_entry(entry, bounds)
    elif isinstance(ty, TupleType):
        for field in ty.fields:
            _ingest_dim_vars(field, bounds)


class _SignatureDimVisitor(ExprVisitor[None]):
    def __init__(self, bounds: dict[str, tuple[int, int]]) -> None:
        super().__init__()
        self.bounds = bounds

    def visit_DimVar(self, entry: DimVar, ctx=None) -> None:
        prior = self.bounds.get(entry.name)
        if prior is None:
            self.bounds[entry.name] = (entry.lo, entry.hi)
        elif prior != (entry.lo, entry.hi):
            raise VerifyError(
                f"inconsistent DimVar bounds for {entry.name!r} within "
                f"function signature: [{prior[0]}, {prior[1]}] vs "
                f"[{entry.lo}, {entry.hi}]"
            )

    def visit_Call(self, entry: Call, ctx=None) -> None:
        for arg in entry.args:
            self.visit(arg, ctx)

    def default_visit(self, entry, ctx=None) -> None:
        return None


def _ingest_shape_entry(entry: object, bounds: dict[str, tuple[int, int]]) -> None:
    if isinstance(entry, (DimVar, Call)):
        _SignatureDimVisitor(bounds).visit(entry)


def _collect_param_dim_vars(fn: Function) -> dict[str, tuple[int, int]]:
    """Bounds from ``fn.params`` only.

    Drives envelope ⊆ and unknown-name checks: specializations must
    anchor to a ``DimVar`` reachable from an *input* param, because
    Dispatch subjects lower to ``ShapeOf(param, axis)`` and
    can only reference a value the caller provides.
    """
    bounds: dict[str, tuple[int, int]] = {}
    for p in fn.params:
        _ingest_dim_vars(p.type, bounds)
    return bounds


def _check_signature_dim_var_consistency(fn: Function) -> None:
    """Scan params + return_type for same-name ``DimVar`` consistency.

    Recurses through ``TensorType`` and ``TupleType``. Raises on any
    same-name disagreement anywhere in the signature.
    """
    bounds: dict[str, tuple[int, int]] = {}
    for p in fn.params:
        _ingest_dim_vars(p.type, bounds)
    _ingest_dim_vars(getattr(fn, "return_type", None), bounds)


def _verify_signature_dim_vars(fn: Function) -> None:

    _check_signature_dim_var_consistency(fn)
    param_bounds = _collect_param_dim_vars(fn)
    for pat in fn.specializations:
        if not isinstance(pat, RangePattern):
            continue
        dv_bounds = param_bounds.get(pat.dim_var)
        if dv_bounds is None:
            raise VerifyError(
                f"specialization RangePattern({pat.dim_var!r}, {pat.lo}, "
                f"{pat.hi}) references unknown DimVar (specializations must "
                f"anchor to a DimVar reachable from an input parameter)"
            )
        lo, hi = dv_bounds
        if not (lo <= pat.lo and pat.hi <= hi):
            raise VerifyError(
                f"RangePattern ({pat.dim_var!r}, {pat.lo}, {pat.hi}) is not "
                f"contained in DimVar envelope [{lo}, {hi}]"
            )


class _StmtRejectingVisitor(ExprVisitor[None]):
    def _visit_region(self, expr: LoopRegion | MeshRegion, ctx=None) -> None:
        verify_region_isolated(expr)
        for operand in expr_children(expr):
            self.visit(operand, ctx)

    visit_LoopRegion = _visit_region
    visit_MeshRegion = _visit_region

    def visit_Call(self, expr: Call, ctx=None) -> None:
        for arg in expr.args:
            if isinstance(arg, Stmt):
                raise VerifyError(
                    f"hir body contains a Stmt node {type(arg).__name__}; hir is expr-only"
                )
            self.visit(arg, ctx)

    def default_visit(self, expr, ctx=None) -> None:
        if isinstance(expr, Stmt):
            raise VerifyError(
                f"hir body contains a Stmt node {type(expr).__name__}; hir is expr-only"
            )


def _reject_stmt_nodes(expr: Expr) -> None:
    _StmtRejectingVisitor().visit(expr)


def input_params(op_type: type) -> tuple:
    return tuple(param for param in op_type._op_schema.signature if param.kind == "input")


def verify_prim_function(
    fn: PrimFunction, *, module_fns: Iterable[PrimFunction] | Module = ()
) -> None:
    """Per [tir §1.3](docs/spec/tir.md#13-primfunction)'s rule list."""
    for param in fn.params:
        verify_value_layouts(param)
    _check_param_homogeneity(fn)
    module_context = module_fns if isinstance(module_fns, Module) else tuple(module_fns)
    for variant in fn.variants:
        if len(variant.specializations) != 1 or not isinstance(
            variant.specializations[0], RangePattern
        ):
            raise VerifyError(f"PrimFunction {fn.name!r}: each variant must have one RangePattern")
        pat = variant.specializations[0]
        if locate_dim_var(fn.params, pat.dim_var) is None:
            raise VerifyError(
                f"PrimFunction {fn.name!r}: specialization subject {pat.dim_var!r} "
                "cannot be derived from parameters"
            )
        verify_prim_function(variant, module_fns=module_context)
    ctx = VerifyContext()
    scope: list[Mesh] = []

    if isinstance(module_fns, Module):
        environment = next(
            (
                frame
                for frame in list_call_paths(module_fns).values()
                if fn is frame.function or any(fn is v for v in frame.function.variants)
            ),
            None,
        )

        if environment is not None:
            ctx = VerifyContext(module=environment.module, function=fn,
                                target=environment.target, topologies=environment.topologies)

    verify_private_parameters(fn, _modes_used_by(fn))
    _walk_stmt(fn.body, ctx, scope, fn, module_context, _Binders(fn.params))


def _check_param_homogeneity(fn: PrimFunction) -> None:
    """Param homogeneity ([tir §1.3](docs/spec/tir.md#13-primfunction)).

    Param homogeneity ([tir §1.3](docs/spec/tir.md#13-primfunction)): all params' layouts must be
    consistently ShardLayout or all non-ShardLayout. Mixed is error.
    """
    kinds = []
    for p in fn.params:
        ty = p.type
        if not isinstance(ty, TensorType):
            continue
        kinds.append(isinstance(ty.layout, ShardLayout))
    if kinds and len(set(kinds)) > 1:
        raise VerifyError(
            f"PrimFunction {fn.name!r}: {spec_ref_render(_PRIM_FUNCTION)} param layouts "
            f"must be homogeneously ShardLayout or non-ShardLayout, got mixed"
        )


class _Binders:
    def __init__(self, params):
        self.visible = {id(param) for param in params}
        self.ever_bound = set(self.visible)

    def uses(self, value, stmt):
        if isinstance(value, Var):
            if id(value) not in self.visible:
                raise VerifyError(f"{value.name} is used outside the scope that binds it")
        elif isinstance(value, (Op, PrimFunction, SymbolRef, ModuleList)):
            return
        elif isinstance(value, Call):
            self.uses((value.target, value.args), stmt)
        elif isinstance(value, (Expr, tuple, list)):
            children = value if isinstance(value, (tuple, list)) else expr_children(value)
            for child in children:
                self.uses(child, stmt)
        else:
            raise VerifyError(f"non-IR node {type(value).__name__} in {type(stmt).__name__}")

    @contextmanager
    def bind(self, var, stmt):
        if id(var) in self.ever_bound:
            raise VerifyError(
                f"{type(stmt).__name__} binding {var.name!r}: "
                f"{spec_ref_render(_PRIM_FUNCTION)} var must be a fresh Var; this Var "
                "is already bound in an outer scope"
            )
        self.ever_bound.add(id(var))
        self.visible.add(id(var))
        try:
            yield
        finally:
            self.visible.remove(id(var))


def _walk_stmt(stmt, ctx, scope, fn, module_context, binders: _Binders, modes: tuple = ()):
    for _, value in _iter_stmt_expr_fields(stmt):
        if isinstance(value, Expr):
            verify_value_layouts(value)
    match stmt:
        case Sequential():
            with ExitStack() as bound:
                for s in stmt.body:
                    _walk_stmt(s, ctx, scope, fn, module_context, binders, modes)
                    if isinstance(s, LetStmt):
                        bound.enter_context(binders.bind(s.var, s))
            return
        case LetStmt():
            verify_expression_placement(stmt.value, modes, where="ExecutionScope")
            verify_allocation_placement(stmt.value, modes, _modes_used_by(fn))
            binders.uses(stmt.value, stmt)
            value_ty = ctx.type_of(stmt.value)
            if stmt.var.type != value_ty:
                raise VerifyError(
                    f"LetStmt binding {stmt.var.name!r}: var.type {stmt.var.type} "
                    f"!= value.type {value_ty}"
                )

            _reject_nested_alloc_tensor(stmt.value, at_letstmt_value=True)
            _check_embedded_sharding(stmt.value, scope, fn)
            return
        case For():
            binders.uses((stmt.start, stmt.stop, stmt.step), stmt)
            _check_rank0_int(ctx, stmt, stmt.start, "For.start")
            _check_rank0_int(ctx, stmt, stmt.stop, "For.stop")
            _check_rank0_int(ctx, stmt, stmt.step, "For.step")
            for field, bound in (
                ("start", stmt.start),
                ("stop", stmt.stop),
                ("step", stmt.step),
            ):
                _check_bound_coordinates(field, bound, scope)
            if isinstance(stmt.step, Constant) and stmt.step.value == 0:
                raise VerifyError("For.step must not be 0")
            iv_ty = stmt.induction_var.type
            if not (
                isinstance(iv_ty, TensorType)
                and iv_ty.shape == ()
                and iv_ty.dtype in (DType.i32, DType.i64)
            ):
                raise VerifyError("For.induction_var must be rank-0 integer")
            with binders.bind(stmt.induction_var, stmt):
                _walk_stmt(stmt.body, ctx, scope, fn, module_context, binders, modes)
            return
        case While():
            binders.uses(stmt.cond, stmt)
            _check_rank0_bool(ctx, stmt, stmt.cond)
            _walk_stmt(stmt.body, ctx, scope, fn, module_context, binders, modes)
            return
        case If():
            binders.uses(stmt.cond, stmt)
            _check_rank0_bool(ctx, stmt, stmt.cond)
            _walk_stmt(stmt.then_body, ctx, scope, fn, module_context, binders, modes)
            _walk_stmt(stmt.else_body, ctx, scope, fn, module_context, binders, modes)
            return
        case MeshScope():
            scope.append(stmt.mesh)
            with binders.bind(stmt.binding, stmt):
                _walk_stmt(stmt.body, ctx, scope, fn, module_context, binders, modes)
            scope.pop()
            return
        case ExecutionScope():
            _verify_execution_scope(stmt, ctx, scope, modes)
            _walk_stmt(stmt.body, ctx, scope, fn, module_context, binders, (*modes, stmt.kind))
            # AFTER the body, deliberately. The producer rule presumes a well-formed
            # predicate: if the operand in the mask slot is the wrong TYPE (i32, say),
            # that is an operand-type refusal and must remain the reported layer. Only
            # a mask that is otherwise well formed -- a genuine bool register -- can be
            # "unproduced". Running this before the walk made every unproduced mask
            # preempt the dtype/lane rules and stole their negatives; deferring keeps
            # each rule owning its own case and still catches the mask that is correct
            # in every respect except that nothing wrote it.
            verify_predicate_producers(stmt)
            return
        case Return():
            for _name, value in _iter_stmt_expr_fields(stmt):
                binders.uses(value, stmt)
            return
        case Evaluate():
            binders.uses((stmt.callable, stmt.args), stmt)
            # Every operand is scanned under the modes in force, whichever route
            # the callable itself takes. Doing this per-route meant a computed
            # instruction target skipped the scan entirely.
            for argument in stmt.args:
                verify_expression_placement(argument, modes, where="ExecutionScope")
                # An inline view or allocation written straight into an operand
                # position is the same claim as one bound with LetStmt, so it is
                # checked by the same rule rather than only where it is spelled
                # with a name.
                verify_allocation_placement(argument, modes, _modes_used_by(fn))
            verify_operand_lanes(stmt, modes)
            verify_evaluate_callable(stmt.callable, modes)
            if isinstance(stmt.callable, Launch):
                _verify_launch(stmt, fn, module_context, ctx)
            elif isinstance(stmt.callable, SymbolRef):
                _verify_symbol_call(stmt, fn, module_context, ctx)
            elif isinstance(stmt.callable, (PrimFunction, ModuleList, Call)):
                _verify_retained_call(stmt, fn, module_context, ctx)
            else:
                op = stmt.callable
                verify_op_placement(op, modes, where="ExecutionScope")
                ctx.mesh_scope = tuple(scope)
                ctx.current_mesh = None
                for mesh in scope:
                    ctx.current_mesh = (
                        make_mesh(ctx.current_mesh, mesh) if ctx.current_mesh else mesh
                    )
                try:
                    check_call(op, stmt.args, ctx.current_mesh, frame=ctx)
                except ScheduleRefusal as error:
                    raise VerifyError(str(error)) from error

            for arg in stmt.args:
                _reject_nested_alloc_tensor(arg, at_letstmt_value=False)
                _check_embedded_sharding(arg, scope, fn)
            return

    fn_verify = verify_stmt_registry.lookup(type(stmt))
    if fn_verify is None:
        raise VerifyError(f"no verify_stmt registered for {type(stmt).__name__}")
    fn_verify(stmt, ctx)
    for field_name, field_val in _iter_stmt_expr_fields(stmt):
        binders.uses(field_val, stmt)
        _reject_nested_alloc_tensor(field_val, at_letstmt_value=False)
        _check_embedded_sharding(field_val, scope, fn)


def _call_like(value) -> bool:
    """Whether *value* reaches a callable other than a direct declared operation."""
    return isinstance(value, (SymbolRef, PrimFunction, ModuleList, Launch)) or (
        isinstance(value, Call) and not isinstance(value.target, Op)
    )


def verify_expression_placement(value, modes: tuple, *, where: str) -> None:
    """Check every operation reachable from one expression under the modes in force.

    Placement is a property of the whole statement, not only of the operation a
    statement happens to evaluate. An operation can appear in a LetStmt value, in
    a call argument, in a tensor view chain or behind a retained callee, and each
    of those positions is an equally good hiding place for something the mode
    does not admit. The whole expression is walked so that "inside the with-block"
    means the same thing as "checked".

    A callable that is not a direct operation is refused inside a mode while
    callee verification is not mode-aware: a callee is verified where it is
    defined, which says nothing about the mode at the call site, so admitting it
    here would be a loophole rather than a check.
    """
    if not modes:
        for reachable in collect_exprs(value) if isinstance(value, Expr) else ():
            if isinstance(reachable, Call):
                verify_op_placement(reachable.target, modes, where=where)
        return
    for reachable in collect_exprs(value) if isinstance(value, Expr) else ():
        if not isinstance(reachable, Call):
            continue
        target = reachable.target
        if is_structural(target):
            continue
        if _call_like(target):
            raise VerifyError(
                f"call to {getattr(target, 'name', type(target).__name__)!r} inside "
                f"{where} {modes[-1]!r}: a call inside an execution region needs "
                "callee verification that knows the mode, which this build does not "
                "have yet; inline the computation instead"
            )
        verify_op_placement(target, modes, where=where)


def _modes_used_by(fn) -> frozenset:
    """Every execution mode this function enters, anywhere in its body."""
    return frozenset(
        stmt.kind for stmt in _iter_all_stmts(fn.body) if isinstance(stmt, ExecutionScope)
    )


def verify_allocation_placement(value, modes: tuple, enclosing: frozenset = frozenset()) -> None:
    """Refuse a region-private allocation made outside the region that owns it.

    A register value is private to its mode: it exists because the mode has a
    register file, and it has no meaning in an issue stream that is not the vector
    one. So an allocation of register storage *outside* the mode is not a
    near-miss to be repaired later -- it is a buffer that should not exist, and a
    later pass that hoists one "for reuse" would be manufacturing exactly the
    value this rule forbids.

    Checked at the allocation rather than at its uses because the allocation is
    where the lifetime is decided. A use outside the region is then impossible
    without also crossing the binding scope, which the binder check already
    refuses.

    The rule is stated relative to the modes *this function* enters, not to every
    kind any loaded target declares. Register storage is not globally private: it
    is private to a mode, and a program with no mode has none. Keying off the
    registry instead would make one target program unverifiable because another
    target happened to be imported into the same process, which is a property of
    the process rather than of the program.
    """
    for reachable in collect_exprs(value) if isinstance(value, Expr) else ():
        if not isinstance(reachable, Call):
            continue
        type_ = getattr(reachable, "type", None)
        storage = getattr(type_, "storage", None)
        if storage is None:
            continue
        # Which declarations treat this storage as theirs, and which of those the
        # function actually enters. Ownership must be decided over BOTH sets: an
        # allocation is legal as soon as ANY mode in force owns the storage, and is
        # a violation only when a mode this function enters owns it while none of
        # the modes in force does.
        #
        # Iterating the declarations and refusing on the first owner that is "in
        # enclosing but not in modes" would refuse a legal allocation whenever two
        # modes own the same storage: standing inside the first, the second is
        # entered somewhere in the function and always looks violated. That shape
        # cannot occur today -- exactly one declared kind claims register storage --
        # but the rule should not depend on that remaining true, so it is written to
        # be correct for any number of owners rather than for the one we happen to
        # have.
        owners = [declared for declared in declared_kinds() if declared.owns_storage(storage)]
        if not owners:
            continue
        if any(declared.name in modes for declared in owners):
            continue
        entered = [declared for declared in owners if declared.name in enclosing]
        if not entered:
            continue  # private to a mode this function never enters: not this rule
        holder = modes[-1] if modes else None
        where = (
            f"runs inside {holder!r}"
            if holder is not None
            else "runs outside every execution region"
        )
        names = ", ".join(repr(declared.name) for declared in entered)
        raise VerifyError(
            f"allocation of {storage.name.lower()} storage for "
            f"{getattr(type_, 'shape', ())!s} {where}: {storage.name.lower()} is "
            f"private to execution region {names}, so it may only be "
            "allocated inside that region"
        )

def verify_private_parameters(fn, enclosing: frozenset) -> None:
    """Refuse an externally supplied value in storage a mode owns.

    A register value exists because the region has a register file, so it cannot
    also be handed in from outside: the caller has no register file to supply it
    from. A parameter whose storage is private to a mode *this function enters*
    therefore has no possible producer, which is why the rule is stated over the
    modes the function enters rather than over every declared kind. In a program
    with no mode, register storage is ordinary CUDA thread-local memory and the
    rule does not apply -- the same reasoning as the allocation check above.

    This is the fail-closed half of the capture rule. The allocation check refuses
    making a private value outside the region; this refuses arriving with one, so
    there is no shape of private value whose owner is unaccounted for: allocated
    in the region, or handed in (refused here), or viewed inline (refused by the
    same expression walk, since a view call carries its result type).
    """
    if not enclosing:
        return
    for param in fn.params:
        type_ = getattr(param, "type", None)
        storage = getattr(type_, "storage", None)
        if storage is None:
            continue
        # Named over EVERY declaring mode the function enters, for the reason given
        # on the allocation check: with more than one owner, reporting only the first
        # found would name a mode that is not the one the reader was thinking of, and
        # the message would be right for the wrong place. The rule is the same either
        # way -- such a parameter has no possible producer -- so the parameter is
        # refused once, with every owner named.
        entered = [
            declared
            for declared in declared_kinds()
            if declared.owns_storage(storage) and declared.name in enclosing
        ]
        if not entered:
            continue
        names = ", ".join(repr(declared.name) for declared in entered)
        raise VerifyError(
            f"PrimFunction {fn.name!r}: parameter {param.name!r} is "
            f"{storage.name.lower()} storage, which is private to execution "
            f"region {names}; a private value is created inside its "
            "region, so it cannot be supplied from outside one"
        )


def verify_operand_lanes(stmt, modes: tuple) -> None:
    '''Refuse an instruction operand whose extent is not the declared lane count.

    A lane count that only appears in a constant is documentation. The mode
    states how many lanes one issue covers, and an instruction in that mode
    operates on exactly that many: an operand of a different extent is a
    quantity this mode cannot issue, whether it is shorter (a partial issue
    nothing here describes) or longer (tiling this build has no rule for).
    Stating the check makes the extent a contract; leaving it out would make a
    64-lane add and a 32-lane add equally acceptable.

    Only register operands are measured. Memory operands are tiles staged by the
    transfers, whose extent rules belong to the instruction that moves them, and
    the predicate is measured by the instruction that consumes it, so the mask
    keeps one owner instead of two that could disagree.
    '''
    if not modes:
        return
    declared = execution_kind(modes[-1])
    if declared is None or declared.lane_count is None:
        return
    for argument in stmt.args:
        type_ = getattr(argument, "type", None)
        storage = getattr(type_, "storage", None)
        if storage is None or not declared.owns_storage(storage):
            continue
        name = getattr(stmt.callable, "name", type(stmt.callable).__name__)
        dtype_name = getattr(getattr(type_, "dtype", None), "name", None)
        if dtype_name == "bool":
            # the predicate: owned by the instruction that consumes it
            continue
        if storage is StorageKind.CARRIER:
            # The carrier register class is BF16 TRANSPORT/CONVERSION, not a compute
            # register. It is exempt from the compute-dtype rule exactly as the predicate
            # is, and stays fully checked for lane extent below. This ADDS a capability; it
            # does not disable the check for compute registers.
            pass
        allowed = tuple(getattr(dtype, "name", None) for dtype in declared.element_dtypes) if storage is not StorageKind.CARRIER else ()
        if allowed and dtype_name not in allowed:
            raise VerifyError(
                f"{name}: operand of type {dtype_name} in "
                f"{storage.name.lower()} is not an element type execution region "
                f"{declared.name!r} computes on ({allowed!r}); an i32 register is "
                "also 32 bits wide, which is why agreeing widths is not the rule"
            )
        shape = tuple(getattr(type_, "shape", ()))
        if len(shape) != 1 or shape[0] != declared.lane_count:
            raise VerifyError(
                f"{name}: operand of shape {shape} is not the "
                f"{declared.lane_count} lanes execution region {declared.name!r} "
                "declares; one issue covers exactly that many lanes, and this build "
                "has no tiling rule that would make a different extent mean something"
            )


def verify_operand_dtypes(op, declared, name: str) -> None:
    """Refuse an operand whose element type the mode does not compute on.

    The width check above is necessary but not sufficient: i32 is also 32 bits
    wide, so "the widths agree" admits a register holding integers to an add
    that only performs floating-point arithmetic. The mode therefore states its
    element types, and every register operand must be one of them. Rows are read
    off the operation's own operand types rather than from a list of parameter
    names, so an instruction added later is covered without editing this check.
    """
    allowed = tuple(getattr(dtype, "name", None) for dtype in declared.element_dtypes)
    if not allowed:
        return
    for param in input_params(type(op)):
        pattern = getattr(param, "pattern", None)
        if getattr(pattern, "storage", None) is None:
            continue
        if not declared.owns_storage(pattern.storage):
            continue
        dtype = getattr(pattern, "dtype", None)
        if dtype is None:
            continue
        names = tuple(
            getattr(option, "name", None)
            for option in getattr(dtype, "patterns", (dtype,))
        )
        if names == ("bool",):
            # The predicate register is a mask, not an element: it is one bit per
            # lane and no arithmetic is performed on it, so the element-type rule
            # does not apply to it. Without this exemption every instruction that
            # WRITES the predicate -- pset -- would be refused by a rule about
            # elements, which is the positive part of the chain failing its own
            # check. The extent rule above already exempts it for the same reason;
            # the two must agree or the chain is unverifiable.
            continue
        if getattr(pattern, "storage", None) is StorageKind.CARRIER:
            # Same capability split at the pattern level: a carrier operand is transport/
            # conversion, not a compute element. Every other operand is still checked, so
            # bf16 in an element register stays refused.
            continue
        if any(name_ not in allowed for name_ in names):
            raise VerifyError(
                f"{name}: operand {param.name!r} is declared as {names!r} but execution "
                f"region {declared.name!r} computes on {allowed!r}; an element type the "
                "mode does not perform arithmetic on cannot be issued there, even when "
                "its width matches"
            )


def verify_predicate_producers(scope_stmt) -> None:
    """Refuse a predicate operand whose producing instruction does not DOMINATE it.

    Two failures, one rule:
      * a mask with NO producing instruction in the region (the pset was dropped), and
      * a mask whose producing instruction comes AFTER the instruction that reads it
        (the pset was placed after the vadd).
    Both leave the predicated result reading the mask register as it stood before any
    mask was written.

    The second is why this cannot be a set-membership test over the whole scope. The
    resolved TIR is a Sequential and execution order IS statement order, so "a writer
    exists somewhere in this scope" does not mean the writer RAN before the reader. A
    later writer is not legitimate for a mask: nothing in this IR carries a mask write
    across a later read, there is no reordering pass, and no semantic basis for one --
    pset writes a value register, not a deferred promise. The producer must PRECEDE the
    consumer in the resolved statement order. This is dominance, not presence.

    A mask is just a bool register, so an allocated-but-never-written mask type-checks,
    matches the bool-in-rmem operand pattern, passes the lane rule (bool exempt) and
    the dtype rule (bool is a mask, not an element). Without this rule a dropped or
    misordered pset is ACCEPTED and the predicated instruction reads garbage.

    Same region instance means same ExecutionScope body: a writer in a sibling or
    enclosing scope does not make this instance's mask produced, and nesting is refused
    separately (a non-nestable mode cannot be entered inside itself).

    A producer is an instruction (an Evaluate of a declared operation) with declared
    WRITE effect on that operand position that appears EARLIER in the walk. An
    allocation that merely binds the name is not a producer. Only predicate operands the
    instruction READS are subjects: pset's own dst is WRITE and is not a read, so pset
    is not asked to have produced the mask it produces.

    Ordering under control flow is conservative: a write inside an If branch or a loop
    body does not dominate statements after the construct (it may not have executed),
    so it is not carried out of the construct. A write before the construct does
    dominate inside it. Region bodies here are straight-line, so this is exact for the
    programs this mode admits.
    """
    declared = execution_kind(scope_stmt.kind)
    if declared is None:
        return  # an undeclared kind is refused by the scope check; not this rule

    produced = set()  # ids of Vars WRITTEN by a statement already passed in this chain

    def writes_of(stmt):
        """(param, arg) pairs of *stmt* whose operand carries the WRITE effect."""
        op = stmt.callable
        try:
            params = input_params(type(op))
        except Exception:  # noqa: BLE001 -- a non-operation callable has no schema
            return ()
        pairs = []
        for param, arg in zip(params, stmt.args):
            effect = getattr(param, "effect", None)
            if effect is not None and (effect & MemoryEffect.WRITE):
                pairs.append((param, arg))
        return tuple(pairs)

    def is_predicate_read(param):
        """True when *param* is a bool mask the instruction READS, not a write target."""
        pattern = getattr(param, "pattern", None)
        storage = getattr(pattern, "storage", None)
        if storage is None or not declared.owns_storage(storage):
            return False
        dtype = getattr(pattern, "dtype", None)
        names = tuple(
            getattr(option, "name", None)
            for option in getattr(dtype, "patterns", (dtype,))
        )
        if names != ("bool",):
            return False
        effect = getattr(param, "effect", None)
        return effect is not None and bool(effect & MemoryEffect.READ)

    def visit(stmt):
        if isinstance(stmt, Sequential):
            for child in stmt.body:
                visit(child)
            return
        if isinstance(stmt, MeshScope):
            visit(stmt.body)
            return
        if isinstance(stmt, Evaluate):
            op = stmt.callable
            requirement = region_requirement(op)
            try:
                params = input_params(type(op))
            except Exception:  # noqa: BLE001
                params = ()
            if (
                requirement is not None
                and requirement.placement == "inside"
                and requirement.kind == scope_stmt.kind
            ):
                name = getattr(op, "name", type(op).__name__)
                for param, arg in zip(params, stmt.args):
                    if not is_predicate_read(param):
                        continue
                    if isinstance(arg, Var) and id(arg) in produced:
                        continue
                    operand = getattr(arg, "name", type(arg).__name__)
                    raise VerifyError(
                        f"{name}: predicate operand {operand!r} has no producing "
                        f"instruction before it in execution region {declared.name!r}; "
                        "a mask must be written by an instruction that precedes its use "
                        "in the same region instance (pset before the predicated op), or "
                        "the predicated instruction reads a register nothing produced"
                    )
            for _param, arg in writes_of(stmt):
                if isinstance(arg, Var):
                    produced.add(id(arg))
            return
        if isinstance(stmt, (For, While)):
            saved = set(produced)
            visit(stmt.body)
            produced.clear()
            produced.update(saved)
            return
        if isinstance(stmt, If):
            saved = set(produced)
            visit(stmt.then_body)
            produced.clear()
            produced.update(saved)
            visit(stmt.else_body)
            produced.clear()
            produced.update(saved)
            return
        # LetStmt / anything else: an allocation binds a name, it does not produce a
        # mask, so it has no ordering effect here.

    visit(scope_stmt.body)
def verify_op_placement(op, modes: tuple, *, where: str = "execution region") -> None:
    """Check one operation is legal in the execution modes in force.

    A mode is not a hint. Every operation states its placement, and an
    operation that states none may not run inside a mode: the alternative --
    treating "no declaration" as "allowed anywhere" -- is exactly how a legacy
    CUDA-thread binary becomes a vector instruction because it happened to be
    written inside the with-block. Outside a mode the same rule refuses an
    operation that says it belongs inside one, so the declaration is enforced
    in both directions rather than only where it is convenient.
    """
    requirement = region_requirement(op)
    name = getattr(op, "name", type(op).__name__)
    if not modes:
        if requirement is not None and requirement.placement == "inside":
            raise VerifyError(
                f"{name}: this operation declares it belongs inside execution region "
                f"{requirement.kind!r}, but no execution region is in force at it"
            )
        return
    if requirement is None:
        raise VerifyError(
            f"{name}: this operation declares no execution placement and may not run "
            f"inside {where} {modes[-1]!r}; a mode admits only operations written for it"
        )
    if requirement.placement == "outside":
        raise VerifyError(
            f"{name}: this operation declares it belongs outside an execution region "
            f"but runs inside {where} {modes[-1]!r}"
        )
    if requirement.kind != modes[-1]:
        raise VerifyError(
            f"{name}: this operation belongs inside execution region "
            f"{requirement.kind!r} but runs inside {modes[-1]!r}"
        )
    declared = execution_kind(requirement.kind)
    width = declared_mask_width(op)
    if width is not None and declared is not None:
        # ONE mode identity, TWO independently checked widths: an ARITHMETIC instruction is
        # bound to mask_width_bits, a CONVERSION instruction to conversion_mask_width_bits.
        # Neither check may be satisfied by defaulting.
        mask_class = getattr(type(op), "mask_class", "arithmetic")
        if mask_class not in ("arithmetic", "conversion"):
            raise VerifyError(
                f"{name}: mask_class {mask_class!r} is not 'arithmetic' or 'conversion'"
            )
        expected = (
            declared.mask_width_bits
            if mask_class == "arithmetic"
            else declared.conversion_mask_width_bits
        )
        if expected is None:
            raise VerifyError(
                f"{name}: a {mask_class} instruction needs execution region "
                f"{declared.name!r} to declare a {mask_class} mask width, and it does not"
            )
        if width != expected:
            raise VerifyError(
                f"{name}: states element width {width} but execution region "
                f"{declared.name!r} declares {expected} for {mask_class} instructions; an "
                "operation addressing a different element width is addressing lanes this "
                "mode does not have"
            )
    if declared is not None:
        verify_operand_dtypes(op, declared, name)


def verify_evaluate_callable(callable_value, modes: tuple) -> None:
    """Check the callable slot of one Evaluate under the modes in force.

    A computed instance target is a ``Call`` here, and it reaches the retained
    route -- the same route as a direct reference to a PrimFunction. Listing the
    rejection set by the shapes an author might write rather than by the routes
    the walker actually takes is what left ``Call`` out: the guard named four
    types, the dispatch handled five, and an instruction selected at runtime
    walked straight past the mode.

    A declared operation is the one callable this walker *can* judge inside a
    mode, and it is judged on its own terms by verify_op_placement immediately
    afterwards. It must therefore pass through here untouched: refusing it would
    replace a placement verdict with "not verified yet", which is a weaker
    statement about a case that has a real answer.

    While no callee verification carries a mode stack, there is no honest answer
    for a callee inside a mode, so every non-operation callable is refused rather
    than admitted by the route it happens to arrive on.
    """
    if not modes or isinstance(callable_value, Op):
        return
    if isinstance(callable_value, Call):
        name = getattr(callable_value.target, "name", type(callable_value.target).__name__)
    else:
        name = getattr(callable_value, "name", type(callable_value).__name__)
    raise VerifyError(
        f"call to {name!r} inside ExecutionScope {modes[-1]!r}: a call inside an "
        "execution region needs callee verification that knows the mode, which this "
        "build does not have yet; inline the computation instead"
    )


def _verify_execution_scope(stmt, ctx, scope, modes: tuple) -> None:
    """Check a TIR mode scope states a mode this point in the program allows.

    The same four boundary questions the HIR region answers, asked of the TIR
    node, because a lowered scope is a second entry point into the same claims
    and a check that only ran before lowering is not a check on the program that
    is shipped. The TIR scope carries its kind as a string and opens no
    participants, so the mesh is read off the lexical scopes already open.
    """
    declared = execution_kind(stmt.kind)
    if declared is None:
        raise VerifyError(
            f"ExecutionScope names undeclared kind {stmt.kind!r}; declared kinds are "
            f"{tuple(item.name for item in declared_kinds())!r}"
        )
    if declared.mask_width_bits is None:
        raise VerifyError(
            f"ExecutionScope {declared.name!r} declares no mask width; a mode whose "
            "element width is unstated cannot bound its instructions"
        )
    if not declared.required_topologies:
        raise VerifyError(
            f"ExecutionScope {declared.name!r} declares no required topology; "
            "a mode with no domain cannot be checked against this program"
        )
    for outer in modes:
        outer_declared = execution_kind(outer)
        if outer_declared is None:
            raise VerifyError(f"ExecutionScope encloses undeclared kind {outer!r}")
        if not outer_declared.nestable:
            raise VerifyError(
                f"ExecutionScope {declared.name!r} is nested inside {outer_declared.name!r}, "
                "which is not nestable; no execution scope may be opened inside it"
            )
    in_force = tuple(
        str(getattr(level, "name", level))
        for mesh in scope
        for level in getattr(mesh, "topologies", ())
    )
    if not in_force:
        raise VerifyError(
            f"ExecutionScope {declared.name!r} requires topology "
            f"{declared.required_topologies!r} but no MeshScope encloses it; "
            "a mode needs a domain to run in, not only a declaration"
        )
    for topology in declared.required_topologies:
        if topology not in in_force:
            raise VerifyError(
                f"ExecutionScope {declared.name!r} requires topology {topology!r}, "
                f"which the enclosing MeshScope does not have; it has {in_force!r}"
            )


def _iter_stmt_expr_fields(stmt):
    if not dataclasses.is_dataclass(stmt):
        return
    for f in dataclasses.fields(stmt):
        val = getattr(stmt, f.name)
        if f.type in (Expr, "Expr") or isinstance(val, Expr):
            yield f.name, val


def _check_rank0_int(ctx, stmt, expr, field: str):
    t = ctx.type_of(expr)
    if not (isinstance(t, TensorType) and t.shape == () and t.dtype in (DType.i32, DType.i64)):
        raise VerifyError(f"{field} must be rank-0 integer, got {t}")


def _check_rank0_bool(ctx, stmt, expr):
    t = ctx.type_of(expr)
    if not (isinstance(t, TensorType) and t.shape == () and t.dtype == DType.bool):
        raise VerifyError(f"condition must be rank-0 bool, got {t}")


class _AllocTensorRejectingVisitor(ExprVisitor[None]):
    def __init__(self, *, at_letstmt_value: bool) -> None:
        super().__init__()
        self.at_letstmt_value = at_letstmt_value

    def visit_Call(self, expr: Call, ctx=None) -> None:
        if isinstance(expr.target, AllocTensorOp):
            if not self.at_letstmt_value:
                raise VerifyError(
                    f"{spec_ref_render(_PRIM_FUNCTION)}: Call(AllocTensor, ...) may only appear "
                    f"as a direct LetStmt.value; found nested inside another Expr"
                )
            self.at_letstmt_value = False
            self.visit_operands(expr, ctx)
            return
        self.at_letstmt_value = False
        self.visit_operands(expr, ctx)

    def default_visit(self, expr, ctx=None) -> None:
        return None


def _reject_nested_alloc_tensor(expr: Expr, *, at_letstmt_value: bool) -> None:
    """Reject an ``AllocTensor`` call nested outside a ``LetStmt`` value.

    ``Call(AllocTensor, ...)`` may appear only as ``LetStmt.value``.

    See [tir §1.3](docs/spec/tir.md#13-primfunction).
    """
    visitor = _AllocTensorRejectingVisitor(at_letstmt_value=at_letstmt_value)
    visitor.visit(expr)


def _check_bound_coordinates(field: str, bound, scope) -> None:
    """Hold a loop bound's mesh coordinates to the scopes around the loop.

    A coordinate is a coordinate of one scope, read through that scope's
    binding, so a bound naming one has to stand inside a ``MeshScope`` of the
    mesh it names -- the enclosing scope is what says which unit it is about.
    """
    for expr in collect_exprs(bound):
        if not (isinstance(expr, Call) and isinstance(expr.target, MeshCoord)):
            continue
        mesh = expr.target.mesh
        axis = static_dim_value(expr.args[0]) if expr.args else None
        if axis is None or not 0 <= axis < len(flatten(mesh.layout).shape):
            raise VerifyError(
                f"For.{field} reads axis {axis!r} of a mesh of rank "
                f"{len(flatten(mesh.layout).shape)}"
            )
        if not any(held is mesh or held == mesh for held in scope):
            raise VerifyError(
                f"For.{field} reads a coordinate of {mesh!r}, which no enclosing MeshScope binds"
            )


def _check_embedded_sharding(expr: Expr, scope, fn):
    """Require embedded shard layouts to use a scoped or parameter mesh.

    See [tir §1.3](docs/spec/tir.md#13-primfunction).
    """
    to_visit = [expr]
    while to_visit:
        e = to_visit.pop()
        if isinstance(e, Call):
            op = e.target
            for attr_name, attr_val in _iter_op_attrs(op):
                if isinstance(attr_val, ShardLayout):
                    _assert_mesh_in_scope(attr_val.mesh, scope, fn)
            to_visit.extend(e.args)


def _iter_op_attrs(op):
    for info in type(op).params():
        if info.kind == "attribute":
            yield info.name, getattr(op, info.name, None)


def _assert_mesh_in_scope(mesh: Mesh, scope, fn):
    if any(mesh == m for m in scope) or (scope and mesh == make_mesh(*scope)):
        return
    for p in fn.params:
        if isinstance(p.type, TensorType) and isinstance(p.type.layout, ShardLayout):
            if p.type.layout.mesh == mesh:
                return
    raise VerifyError(
        f"{spec_ref_render(_PRIM_FUNCTION)}: ShardLayout references mesh {mesh!r} that is "
        f"not in current MeshScope stack nor bound by any param"
    )


def _verify_shape_of(expr: ShapeOf) -> None:
    if not isinstance(expr.param, Var):
        raise VerifyError(f"ShapeOf.param must be a Var, got {type(expr.param).__name__}")
    if not isinstance(expr.axis, int) or isinstance(expr.axis, bool) or expr.axis < 0:
        raise VerifyError(f"ShapeOf.axis must be a non-negative int, got {expr.axis!r}")
    expected = TensorType.scalar(dtype=DType.i32, storage=StorageKind.RMEM)
    if expr.type != expected:
        raise VerifyError(f"ShapeOf.type must be rank-0 i32 scalar TensorType, got {expr.type}")


def _resolve_symbol_ref(ref, module_context, caller, frame=None):
    """Resolve relative to the caller's slot context, through the shared reader."""
    scope = module_context
    if frame is not None and frame.module is not None:
        scope = frame.module
    elif isinstance(scope, Module):
        owner = next(
            (
                item
                for item in list_call_paths(scope).values()
                if caller is item.function or any(caller is v for v in item.function.variants)
            ),
            None,
        )
        if owner is None:
            path = ".".join((*ref.nested, ref.name))
            raise VerifyError(f"symbol {path!r}: caller {caller.name!r} has no owning Module")
        scope = owner.module
    try:
        return resolve_symbol(scope, ref.name, nested=ref.nested)
    except ValueError as error:
        raise VerifyError(str(error)) from error


def _verify_symbol_call(stmt: Evaluate, fn, module_context, ctx):
    ref = stmt.callable
    if not module_context:
        raise VerifyError(
            f"Evaluate(SymbolRef {ref.name!r}) requires module context; "
            f"call verify_prim_function(fn, module_fns=...) or verify_module([...])"
        )
    callee = _resolve_symbol_ref(ref, module_context, fn, ctx)
    expected_type = callable_type_for_prim_function(callee)
    if ref.type != expected_type:
        raise VerifyError(
            f"SymbolRef {ref.name!r}: type {ref.type} != resolved callee "
            f"CallableType {expected_type}"
        )
    if callee is fn:
        raise VerifyError("Evaluate(SymbolRef): recursion not allowed")
    _assert_no_path_back(callee, fn, module_context, visited=set(), frame=ctx)
    _verify_call_arguments(stmt, callee.params, f"Evaluate(SymbolRef {ref.name!r})", ctx)


def _verify_call_arguments(stmt, params, label, ctx):
    if len(stmt.args) != len(params):
        raise VerifyError(f"{label}: arg count {len(stmt.args)} != param count {len(params)}")
    for i, (arg, param) in enumerate(zip(stmt.args, params)):
        arg_ty = ctx.type_of(arg)
        param_ty = param.type
        if type(arg_ty) is not type(param_ty):
            raise VerifyError(
                f"{label}: arg[{i}] type kind "
                f"{type(arg_ty).__name__} != param type kind {type(param_ty).__name__}"
            )
        if arg_ty != param_ty:
            raise VerifyError(
                f"{label}: arg[{i}] type {arg_ty} != param {param.name!r} type {param_ty}"
            )


def _retained_callee(target, frame):
    if not isinstance(target, (PrimFunction, ModuleList, Call)):
        raise VerifyError(f"unsupported TIR call target {type(target).__name__}")
    if not isinstance(target, PrimFunction):
        if frame is None:
            raise VerifyError("instance TIR call requires module context")
        inference_type(target, TypeInferContext(module=frame.module, function=frame.function, target=frame.target, topologies=frame.topologies))
    inner = None if frame is None else frame._callee_context(Call(target, (), type=UnitType()))
    callee = inner.function if inner is not None else target
    if not isinstance(callee, PrimFunction):
        # The label has to survive the failure it reports: a computed instance
        # target resolves to whatever the projection returns, so reading `.name`
        # off it unconditionally turned this diagnostic into an AttributeError.
        label = getattr(callee, "name", None) or type(callee).__name__
        raise VerifyError(f"TIR instance entry {label!r} is not a PrimFunction")
    return callee, inner


def _verify_retained_call(stmt, fn, module_context, ctx):
    target = stmt.callable
    callee, frame = _retained_callee(target, ctx)
    direct = isinstance(target, PrimFunction)
    label = (
        target.name
        if direct
        else (
            target.key
            if isinstance(target, ModuleList)
            else f"{target.args[0].key}[{TirPrinter().visit(target.args[1])}]"
        )
    )
    prefix = f"Evaluate({label})"
    _assert_no_path_back(
        callee,
        fn,
        module_context,
        visited=set(),
        frame=frame,
        recursion=f"{prefix}: recursion not allowed",
    )
    params = tuple(param for param in callee.params if direct or not param.is_const)
    _verify_call_arguments(stmt, params, prefix, ctx)


_LAUNCH_EXTENT_OPS = (DimAdd, DimSub, DimMul, DimFloorDiv, DimMod, DimMin, DimMax)


class _LaunchExtentVisitor(ExprVisitor[None]):
    def __init__(self, extent_params, ref_name) -> None:
        super().__init__()
        self.extent_params = extent_params
        self.ref_name = ref_name

    def visit_Constant(self, extent: Constant, ctx=None) -> None:
        if isinstance(extent.value, bool) or not isinstance(extent.value, int):
            raise VerifyError(
                f"Launch of {self.ref_name!r}: grid/block extent Constant must be an "
                f"int, got {extent.value!r}"
            )

    def visit_ShapeOf(self, extent: ShapeOf, ctx=None) -> None:
        param = self.extent_params.get(id(extent.param))
        if param is None:
            raise VerifyError(
                f"Launch of {self.ref_name!r}: grid/block extent ShapeOf references "
                f"{extent.param.name!r}, which is not a forwarded launch "
                f"argument or host parameter"
            )
        rank = len(param.type.shape) if isinstance(param.type, TensorType) else 0
        if not isinstance(param.type, TensorType) or not (0 <= extent.axis < rank):
            raise VerifyError(
                f"Launch of {self.ref_name!r}: grid/block extent ShapeOf axis "
                f"{extent.axis} is out of range for {extent.param.name!r} "
                f"(rank {rank})"
            )

    def visit_Call(self, extent: Call, ctx=None) -> None:
        if not (isinstance(extent.target, _LAUNCH_EXTENT_OPS)):
            self.default_visit(extent, ctx)
            return
        self.visit_operands(extent, ctx)

    def default_visit(self, extent, ctx=None) -> None:
        raise VerifyError(
            f"Launch of {self.ref_name!r}: grid/block extent must be an integer "
            f"Constant, ShapeOf, or dim-arithmetic Call, got {type(extent).__name__}"
        )


def _verify_launch_extent(extent, extent_params, ref_name) -> None:
    """Verify launch extent.

    A grid/block launch extent Expr is an integer ``Constant``, a
    ``ShapeOf`` of a forwarded / entry tensor parameter, or a dim-arithmetic
    ``Call`` whose operands are themselves valid extents.
    """
    if isinstance(extent, int) and not isinstance(extent, bool):
        return
    _LaunchExtentVisitor(extent_params, ref_name).visit(extent)


def _verify_launch(stmt: Evaluate, fn, module_context, ctx):
    """Check host launches before target-specific placement validation.

    The launch is ``Evaluate(Launch(...), (SymbolRef(callee), grid_x, grid_y,
    grid_z, block_x, block_y, block_z, *forwarded))``. An HIR callee's arguments
    were written and checked by the host-entry pass that built the Launch
    ([passes §7.3](docs/spec/passes.md#73-insert_default_host_entry)), so only
    its selector is resolved here.
    """
    ref = stmt.args[0]
    if not isinstance(ref, SymbolRef):
        raise VerifyError(f"Launch: args[0] must be a SymbolRef callee, got {type(ref).__name__}")
    if len(stmt.args) < 7:
        raise VerifyError(
            f"Launch of {ref.name!r}: expected at least 7 args (callee + six "
            f"grid/block extents), got {len(stmt.args)}"
        )
    forwarded = stmt.args[7:]

    extent_params = {id(p): p for p in fn.params}
    for a in forwarded:
        if isinstance(a, Var):
            extent_params.setdefault(id(a), a)
    for extent in stmt.args[1:7]:
        _verify_launch_extent(extent, extent_params, ref.name)
    for i, arg in enumerate(forwarded):
        if not isinstance(ctx.type_of(arg), TensorType):
            raise VerifyError(f"Launch of {ref.name!r}: forwarded arg[{i}] must be a tensor")
    if not module_context:
        return
    callee = _resolve_symbol_ref(ref, module_context, fn, ctx)
    if isinstance(callee, Function):
        return
    if isinstance(callee.target, CpuTarget):
        raise VerifyError(f"Launch callee {callee.name!r} must target a device, not the CPU")
    expected = callable_type_for_prim_function(callee)
    if ref.type != expected:
        raise VerifyError(
            f"Launch callee SymbolRef {ref.name!r}: type {ref.type} != resolved "
            f"callee CallableType {expected}"
        )

    if len(forwarded) != len(callee.params):
        raise VerifyError(
            f"Launch of {callee.name!r}: forwarded arg count {len(forwarded)} != "
            f"param count {len(callee.params)}"
        )
    for i, (arg, param) in enumerate(zip(forwarded, callee.params)):
        arg_ty = ctx.type_of(arg)
        if arg_ty != param.type:
            raise VerifyError(
                f"Launch of {callee.name!r}: forwarded arg[{i}] type {arg_ty} != "
                f"param {param.name!r} type {param.type}"
            )


def _assert_no_path_back(callee, root, module_context, visited, frame=None, recursion=None):
    if callee is root:
        raise VerifyError(
            recursion or f"call DAG violation: {callee.name!r} calls back into {root.name!r}"
        )
    if isinstance(module_context, Module):
        frame = next(
            (
                item
                for item in list_call_paths(module_context).values()
                if callee is item.function
                or any(callee is variant for variant in item.function.variants)
            ),
            frame,
        )
    key = (id(callee), id(frame.module) if frame is not None else None)
    if key in visited:
        return
    visited.add(key)
    for stmt in _iter_all_stmts(callee.body):
        if not isinstance(stmt, Evaluate):
            continue
        target = stmt.callable
        reached, inner = None, frame
        if isinstance(target, SymbolRef):
            reached = _resolve_symbol_ref(target, module_context, callee, frame)
            if frame is not None:
                inner = dataclasses.replace(frame, function=reached)
        elif isinstance(target, (PrimFunction, ModuleList, Call)):
            reached, inner = _retained_callee(target, frame)
        if reached is not None:
            _assert_no_path_back(reached, root, module_context, visited, inner, recursion)


def _iter_all_stmts(body):
    """Yield every Stmt reachable from a Sequential / any Stmt tree."""
    stack = [body]
    while stack:
        s = stack.pop()
        yield s
        if isinstance(s, Sequential):
            stack.extend(s.body)
        elif isinstance(s, (For, While, MeshScope, ExecutionScope)):
            stack.append(s.body)
        elif isinstance(s, If):
            stack.append(s.then_body)
            stack.append(s.else_body)


def verify_module(fns) -> None:
    """Verify a Module tree or a legacy flat function iterable.

    Names are unique within each Module. Variants live on their prototype's
    `variants` field, not as separate same-name entries. A top-level entry is
    never an HIR variant and remains callable when it has no body.
    See [hir §1.1](docs/spec/hir.md#11-function).
    """
    root = fns if isinstance(fns, Module) else None
    if root is not None:
        frames = list_call_paths(root)
        owners = {id(root): root, **{id(frame.module): frame.module for frame in frames.values()}}
        groups = tuple(node.functions for node in owners.values())
        fns = tuple(frame.function for frame in frames.values())
    else:
        groups = (tuple(fns),)
        fns = groups[0]
    prim_fns = [f for f in fns if isinstance(f, PrimFunction)]
    for f in fns:
        if isinstance(f, Function):
            verify_function(f)
            if f.specializations:
                raise VerifyError(
                    f"verify_module: {f.name!r} is a specialization variant at "
                    f"top level; variants live on the base function's "
                    f"`variants`, not as Module.functions entries"
                )
            if f.body is None and not f.variants:
                raise VerifyError(
                    f"verify_module: {f.name!r} has no body and no variants; a "
                    f"module function must be callable"
                )
        elif isinstance(f, PrimFunction):
            verify_prim_function(
                f,
                module_fns=root
                if root is not None
                else tuple(each for f in prim_fns for each in (f, *f.variants)),
            )
        else:
            raise VerifyError(f"verify_module: unknown function type {type(f).__name__}")

    for functions in groups:
        by_name: dict[str, list] = {}
        for f in functions:
            by_name.setdefault(f.name, []).append(f)
        for name, group in by_name.items():
            if len(group) > 1:
                raise VerifyError(
                    f"verify_module: {name!r} appears {len(group)} times; one name "
                    f"maps to one local function (specialization variants live on "
                    f"Function.variants)"
                )


__all__ = [
    "input_params",
    "verify_function",
    "verify_module",
    "verify_prim_function",
    "verify_region_isolated",
    "verify_value_layouts",
]
