"""Typed UnitTree over the LOWERED native TIR.

This is the source-analysis boundary of the S0 slice: it turns the actual
PrimFunction produced by the normal entry (checked_program -> analyze ->
finalize) into whole-task scheduling units. It does not build TIR, does not
solve, and does not emit.

UNIT GRANULARITY. One whole VF (an ExecutionScope) is ONE unit, whatever number
of vector instructions run inside it: the scheduler does not split a whole vector
function. A DMA/Cube instruction outside any execution region is its own unit.
Nested loops are Regions, not units; a unit occurs exactly once in the
direct_children of its Region.

FIELD SOURCES. Field names follow docs/handoff/TYPED_SCHEMAS.md section 1.
- node/region/IV/bounds/captures/carries: the actual lowered TIR plus the
  lowering binding record on module.metadata["region_bindings"]; every link is an
  object identity, never a name lookup.
- effects: ParamDef.MemoryEffect from the real instruction schema; an undeclared
  effect (None) is a named refusal, not an empty tuple.
- resources/pipe/completion: the target's instruction contract table
  (tilefoundry_ascend.ir.isa). An instruction with no contract is a named refusal;
  it is never guessed from its class name.
- owner: the enclosing MeshScope binding plus the mesh topology level.

SCHEMA DEVIATIONS reported to the Lead (not silently filled):
- RegionBindingSummary.binding_map maps the occurrences the lowering record
  actually binds (IV, carried params, yields, captures through authored function
  parameters). A loop initial value has no Expr-level actual counterpart in that
  record, so it stays source-side in carried_initials.
- SourceUnit.yields is empty: a TIR ExecutionScope has no result slot, so there is
  no region yield to point at. Values that escape are reachable through the later
  statements that consume the stored buffer.
- SourceUnit.intent is None: this slice has no authored TaskIntent; None means
  "the author declared none", exactly as the schema says.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, is_dataclass

import isl

from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.core.param_def import MemoryEffect
from tilefoundry.ir.isl_interop import shape_to_isl_set
from tilefoundry.ir.tir.memory.alloc_tensor import AllocTensor
from tilefoundry.ir.tir.memory.tensor_view import TensorView
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import (
    Evaluate,
    ExecutionScope,
    For,
    If,
    LetStmt,
    MeshScope,
    Sequential,
    While,
)
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.visitor_registry.access_relation import AccessRelation

from .delays import occupancy_ticks
from .identity import function_identity, node_identity_payload

from tilefoundry_ascend.ir.execution import SIMD_VF_KIND
from tilefoundry_ascend.ir.isa import ISA

__all__ = [
    "AccessFact", "Auto", "COMPLETION_ISSUE", "COMPLETION_READ_COMPLETE",
    "COMPLETION_STATE_COMPLETE", "COMPLETION_WRITE_VISIBLE", "COMPOSITE", "CUBE",
    "DMA", "EFFECT_CONTROL", "EFFECT_MEMORY", "EFFECT_STATE", "Effect", "Fixed",
    "MODE_READ", "MODE_READ_WRITE", "MODE_WRITE", "OwnerKey", "READ", "Region",
    "RegionBindingSummary", "ResourceKey", "ResourceUse", "SourceRefusal",
    "SourceSnapshot", "SourceUnit", "StorageKey", "TRUE_GUARD", "UnitTree",
    "UnitTreeError", "VF", "WRITE", "build_native_unit_tree", "isa_contract_of",
    "unit_accesses",
]

VF = "VF"
DMA = "DMA"
CUBE = "CUBE"
COMPOSITE = "COMPOSITE"
KINDS = (VF, DMA, CUBE, COMPOSITE)

READ = "READ"
WRITE = "WRITE"
READ_WRITE = "READ_WRITE"
MODE_READ = READ
MODE_WRITE = WRITE
MODE_READ_WRITE = READ_WRITE

EFFECT_MEMORY = "MEMORY"
EFFECT_STATE = "STATE"
EFFECT_CONTROL = "CONTROL"

COMPLETION_ISSUE = "ISSUE"
COMPLETION_WRITE_VISIBLE = "WRITE_VISIBLE"
COMPLETION_READ_COMPLETE = "READ_COMPLETE"
COMPLETION_STATE_COMPLETE = "STATE_COMPLETE"

_AIV = "AIV"
_AIC = "AIC"
_DEVICE = "DEVICE"


class UnitTreeError(RuntimeError):
    """A source located refusal from the unit-tree boundary.

    The refusal names the reason by a stable code and keeps the REAL source node.
    It is raised instead of producing a unit whose effects/resources would be a
    guess.
    """

    def __init__(self, code: str, source_ref, detail: str) -> None:
        self.code = code
        self.source_ref = source_ref
        self.detail = detail
        super().__init__("%s: %s" % (code, detail))


@dataclass(frozen=True)
class OwnerKey:
    """(logical program group, engine kind, side id) of one execution owner."""

    program_group: object | None
    core_kind: str
    sid: object | None = None


@dataclass(frozen=True)
class StorageKey:
    space: str
    owner: OwnerKey


@dataclass(frozen=True)
class ResourceKey:
    owner: OwnerKey
    pipe: str
    instance: int = 0


@dataclass(frozen=True)
class ResourceUse:
    resource: ResourceKey
    duration: int
    offset: int = 0


@dataclass(frozen=True)
class Fixed:
    """Authored fixed version count."""

    value: int


@dataclass(frozen=True)
class Auto:
    """Authored auto version domain."""

    minimum: int
    maximum: int


@dataclass(frozen=True)
class SourceRefusal:
    """A named gap with the real source node it belongs to."""

    code: str
    source_ref: object
    detail: str


@dataclass(frozen=True)
class SourceSnapshot:
    """The frozen program the analysis records bind to."""

    module_ref: object
    entry_ref: object
    canonical_bytes: bytes
    identity: str


@dataclass(frozen=True)
class AccessFact:
    """One real operand boundary of one instruction."""

    access_id: str
    task_id: str
    source_ref: object
    operand_index: int
    root_ref: object
    view_ref: object
    relation: AccessRelation
    mode: str
    owner: OwnerKey
    guard: object
    generation: object


@dataclass(frozen=True)
class Effect:
    """One instruction's declared memory effect, bound to its real accesses."""

    effect_id: str
    task_id: str
    source_ref: object
    kind: str
    access_ids: tuple
    state_ref: object | None
    generation: object
    guard: object
    completion: str
    pipe: str
    #: The instruction this effect was declared by, so a consumer can look its
    #: registered model cost up instead of inventing one.
    instruction: str = ""


@dataclass(frozen=True)
class RegionBindingSummary:
    """Typed view of one source loop's lowering record (schema section 1)."""

    source_region: object
    induction_var: object
    start: object
    stop: object
    step: object
    carried_params: tuple
    carried_initials: tuple
    yield_values: tuple
    captures: tuple
    actual_loop: object
    actual_induction_var: object
    binding_map: tuple
    actual_defs: tuple
    actual_allocations: tuple
    actual_bounds: tuple


@dataclass(frozen=True)
class Region:
    """One scheduling region: the function root or a real loop."""

    region_id: str
    node_ref: object
    parent_id: str | None
    direct_children: tuple
    induction_var_ref: object | None
    lower_expr: object
    stop_expr: object
    step_expr: object
    binding_summary: RegionBindingSummary | None
    activation_guard: object


@dataclass(frozen=True)
class SourceUnit:
    """One whole task: a whole VF, or one DMA/Cube/composite instruction."""

    unit_id: str
    node_ref: object
    region_ref: Region
    source_order: int
    whole_body: object
    kind: str
    execution_owner: OwnerKey
    resources: tuple
    original_guard: object
    captured_values: tuple
    carried_values: tuple
    yields: tuple
    effects: tuple
    intent: object | None


@dataclass(frozen=True)
class UnitTree:
    """The whole-task tree plus its flat ordered unit index."""

    source_snapshot: SourceSnapshot
    entry_ref: PrimFunction
    roots: tuple
    units: tuple

    @property
    def regions(self) -> tuple:
        """Every Region in document order, root function region included."""
        found = []

        def walk(region):
            found.append(region)
            for child in region.direct_children:
                if isinstance(child, Region):
                    walk(child)

        for root in self.roots:
            walk(root)
        return tuple(found)

    def all_effects(self) -> tuple:
        """Every unit effect in source order (the analyze_native_problem input)."""
        return tuple(effect for unit in self.units for effect in unit.effects)

    def region_of(self, unit: SourceUnit) -> Region:
        return unit.region_ref


def _true() -> Constant:
    return Constant(type=TensorType.scalar(DType.bool), value=True)


def _int(value: int) -> Constant:
    return Constant(type=TensorType.scalar(DType.i64), value=value)


TRUE_GUARD = _true()


def _static_shape(tensor_type) -> tuple:
    shape = tuple(getattr(tensor_type, "shape", ()) or ())
    out = []
    for dim in shape:
        value = static_dim_value(dim)
        if value is None:
            raise UnitTreeError(
                "UNIT_TREE_ACCESS_DOMAIN_UNSUPPORTED", tensor_type,
                "a tensor shape dimension is not a static integer, so this access "
                "domain cannot be stated at the source boundary",
            )
        out.append(value)
    return tuple(out)


def _identity_relation(tensor_type) -> AccessRelation:
    """Whole-tensor identity relation over the REAL operand type."""
    shape = _static_shape(tensor_type)
    box = shape_to_isl_set(shape, {})
    return AccessRelation(
        isl.map.identity(box.get_space().map_from_set()).intersect_domain(box)
    )


def _mode_of(effect: MemoryEffect) -> str:
    read = bool(effect & MemoryEffect.READ)
    write = bool(effect & MemoryEffect.WRITE)
    if read and write:
        return MODE_READ_WRITE
    if write:
        return MODE_WRITE
    if read:
        return MODE_READ
    raise ValueError("MemoryEffect(0) is not a declared operand effect")


_COMPLETIONS = {
    "issue": COMPLETION_ISSUE,
    "issue+visible": COMPLETION_WRITE_VISIBLE,
    "issue+write_visible": COMPLETION_WRITE_VISIBLE,
    "issue+data_ready": COMPLETION_READ_COMPLETE,
    "issue+state_complete": COMPLETION_STATE_COMPLETE,
}


def isa_contract_of(op):
    """The target instruction contract of a REAL registered Op instance."""
    schema = getattr(type(op), "_op_schema", None)
    if schema is None or not getattr(schema, "name", ""):
        raise KeyError(type(op).__name__)
    name = schema.name
    prefix = "ascend_"
    if not name.startswith(prefix):
        raise KeyError(name)
    return ISA[name[len(prefix):]]


def _walk(node, *, skip=("type", "metadata", "origin")):
    yield node
    if isinstance(node, (tuple, list)):
        for item in node:
            yield from _walk(item, skip=skip)
        return
    if isinstance(node, Var):
        return
    for name in getattr(node, "__dataclass_fields__", ()) or ():
        if name in skip:
            continue
        yield from _walk(getattr(node, name), skip=skip)


def _vars(node) -> tuple:
    """Var references inside an expression/statement subtree, first use order."""
    found = []
    seen = set()

    def walk(item):
        if isinstance(item, Var):
            if id(item) not in seen:
                seen.add(id(item))
                found.append(item)
            return
        if isinstance(item, (tuple, list)):
            for child in item:
                walk(child)
            return
        if is_dataclass(item):
            for name in getattr(item, "__dataclass_fields__", ()) or ():
                if name in ("type", "metadata", "origin"):
                    continue
                walk(getattr(item, name))

    walk(node)
    return tuple(found)


def _core_kind(mesh) -> str:
    names = {getattr(topology, "name", None) for topology in getattr(mesh, "topologies", ())}
    if "aic" in names and "aiv" not in names:
        return _AIC
    if "aiv" in names:
        return _AIV
    raise UnitTreeError(
        "UNIT_TREE_MESH_UNSUPPORTED", mesh,
        "the mesh does not declare an AIV or AIC topology level this target owns",
    )


@dataclass
class _Root:
    var: Var
    allocation_ref: object
    space: str
    tensor_type: TensorType
    kind: str  # "param" | "alloc" | "addr"


@dataclass
class _RegionState:
    region_id: str
    node_ref: object
    parent_id: str | None
    loop: For | None
    children: list = field(default_factory=list)
    region: Region | None = None


class _Builder:
    """One pass over the REAL lowered statement tree."""

    def __init__(self, snapshot: SourceSnapshot, module, target) -> None:
        self.snapshot = snapshot
        self.module = module
        self.entry = module.entry_function()
        if not isinstance(self.entry, PrimFunction):
            raise UnitTreeError(
                "UNIT_TREE_ENTRY_NOT_PRIM_FUNCTION", self.entry,
                "the unit tree is built over the lowered TIR entry PrimFunction",
            )
        self.target = target if target is not None else getattr(self.entry, "target", None)
        raw = getattr(module, "metadata", {}).get("region_bindings", ())
        self.bindings = {id(row["actual_loop"]): row for row in raw}
        self._authored = None
        self.binders = {}
        self.roots = {}
        self.units = []
        self.regions = []
        self._unit_index = 0
        self._region_index = 0
        self._access_index = 0
        self._effect_index = 0

    # -- identity rooted buffer/root resolution -----------------------------
    def _bind_params(self) -> None:
        for parameter in self.entry.params:
            self.binders[id(parameter)] = ("param", parameter)

    def _register_definition(self, let: LetStmt, source_ref) -> None:
        value = let.value
        target = getattr(value, "target", None)
        if isinstance(target, TensorView):
            base = value.args[0]
            if isinstance(base, Constant) and getattr(base, "value", None) is not None:
                storage = getattr(getattr(base, "type", None), "storage", None)
                if storage is None:
                    raise UnitTreeError(
                        "UNIT_TREE_OPERAND_ROOT_UNRESOLVED", source_ref,
                        "a tensor view base carries no storage kind",
                    )
                space = getattr(storage, "name", str(storage)).lower()
                key = ("addr", space, int(base.value), _static_shape(let.var.type))
                self._root_for(key, let.var, value, space, "addr", source_ref)
                return
            if isinstance(base, Var):
                root = self._root_of(base, source_ref)
                self._root_for(("alias", id(root.var)), root.var, value,
                               root.space, root.kind, source_ref)
                return
            raise UnitTreeError(
                "UNIT_TREE_OPERAND_ROOT_UNRESOLVED", source_ref,
                "a tensor view base is neither a pointer constant nor a bound Var",
            )
        if isinstance(target, AllocTensor):
            self._root_for(("alloc", id(let.var)), let.var, value,
                           _space_of(let.var.type), "alloc", source_ref)
            return
        raise UnitTreeError(
            "UNIT_TREE_DEFINITION_UNSUPPORTED", source_ref,
            "a lowered LetStmt binds neither an allocation nor a tensor view",
        )

    def _root_for(self, key, var, allocation_ref, space, kind, source_ref) -> _Root:
        existing = self.roots.get(key)
        if existing is not None:
            return existing
        root = _Root(var=var, allocation_ref=allocation_ref, space=space,
                     tensor_type=var.type, kind=kind)
        self.roots[key] = root
        return root

    def _root_of(self, var: Var, source_ref) -> _Root:
        for root in self.roots.values():
            if root.var is var:
                return root
        binder = self.binders.get(id(var))
        if binder is None:
            raise UnitTreeError(
                "UNIT_TREE_OPERAND_ROOT_UNRESOLVED", source_ref,
                "an operand Var has no binder in this program, so its buffer root is unknown",
            )
        kind, node = binder
        if kind == "param":
            return self._root_for(("param", id(var)), var, var, _space_of(var.type),
                                  "param", source_ref)
        if kind == "let":
            target = getattr(node.value, "target", None)
            if isinstance(target, TensorView):
                base = node.value.args[0]
                if isinstance(base, Constant) and getattr(base, "value", None) is not None:
                    storage = getattr(getattr(base, "type", None), "storage", None)
                    space = getattr(storage, "name", str(storage)).lower()
                    return self._root_for(
                        ("addr", space, int(base.value), _static_shape(var.type)),
                        var, node.value, space, "addr", source_ref,
                    )
                if isinstance(base, Var):
                    return self._root_of(base, source_ref)
            if isinstance(target, AllocTensor):
                return self._root_for(("alloc", id(var)), var, node.value,
                                      _space_of(var.type), "alloc", source_ref)
        raise UnitTreeError(
            "UNIT_TREE_OPERAND_ROOT_UNRESOLVED", source_ref,
            "the operand's definition does not state a buffer root this slice can bind",
        )

    # -- access / effect records -------------------------------------------
    def _accesses(self, stmt: Evaluate, task_id: str, owner: OwnerKey,
                  generation) -> tuple:
        op = stmt.callable
        schema = getattr(type(op), "_op_schema", None)
        if schema is None:
            raise UnitTreeError(
                "UNIT_TREE_INSTRUCTION_UNREGISTERED", stmt,
                "an Evaluate carries an Op that is not registered, so its operands have no schema",
            )
        parameters = tuple(param for param in schema.signature if param.kind == "input")
        if len(parameters) != len(stmt.args):
            raise UnitTreeError(
                "UNIT_TREE_OPERAND_ARITY_MISMATCH", stmt,
                "the lowered instruction issues %d operand(s) but its schema declares %d"
                % (len(stmt.args), len(parameters)),
            )
        accesses = []
        for index, (parameter, operand) in enumerate(zip(parameters, stmt.args, strict=True)):
            if parameter.effect is None:
                raise UnitTreeError(
                    "UNIT_TREE_EFFECT_UNDECLARED", stmt,
                    "operand %r declares no MemoryEffect; an undeclared effect is not a "
                    "pure operation" % (parameter.name,),
                )
            if not isinstance(getattr(operand, "type", None), TensorType):
                raise UnitTreeError(
                    "UNIT_TREE_ACCESS_DOMAIN_UNSUPPORTED", stmt,
                    "operand %r is not a tensor, so no access relation can be stated"
                    % (parameter.name,),
                )
            if not isinstance(operand, Var):
                raise UnitTreeError(
                    "UNIT_TREE_OPERAND_ROOT_UNRESOLVED", stmt,
                    "operand %r is not a bound Var, so its buffer root is unknown"
                    % (parameter.name,),
                )
            root = self._root_of(operand, stmt)
            access_id = "access:%d" % self._access_index
            self._access_index += 1
            accesses.append(AccessFact(
                access_id=access_id,
                task_id=task_id,
                source_ref=stmt,
                operand_index=index,
                root_ref=root.var,
                view_ref=operand,
                relation=_identity_relation(operand.type),
                mode=_mode_of(parameter.effect),
                owner=owner,
                guard=TRUE_GUARD,
                generation=generation,
            ))
        return tuple(accesses)

    def _effect(self, stmt: Evaluate, task_id: str, contract, accesses, generation) -> Effect:
        completion = _COMPLETIONS.get(contract.completion)
        if completion is None:
            raise UnitTreeError(
                "UNIT_TREE_COMPLETION_UNSUPPORTED", stmt,
                "target completion %r is not one of the typed completion kinds"
                % (contract.completion,),
            )
        effect_id = "effect:%d" % self._effect_index
        self._effect_index += 1
        return Effect(
            effect_id=effect_id,
            task_id=task_id,
            source_ref=stmt,
            kind=EFFECT_MEMORY,
            access_ids=tuple(access.access_id for access in accesses),
            state_ref=None,
            generation=generation,
            guard=TRUE_GUARD,
            completion=completion,
            pipe=contract.resource,
            instruction=contract.name,
        )

    # -- instruction units --------------------------------------------------
    def _instruction_unit(self, region: _RegionState, stmt: Evaluate,
                          owner: OwnerKey, generation, order: int) -> SourceUnit:
        contract = self._contract(stmt)
        unit_id = "unit:%d" % self._unit_index
        self._unit_index += 1
        accesses = self._accesses(stmt, unit_id, owner, generation)
        effect = self._effect(stmt, unit_id, contract, accesses, generation)
        kind = DMA if contract.category == "dma" else (
            CUBE if contract.core_affinity == "aic" else COMPOSITE
        )
        # The window is the REGISTERED model cost, never max(1, latency or 0):
        # an instruction the table does not model refuses by name instead of
        # silently acquiring a tick.
        duration = occupancy_ticks(contract.name)
        use = ResourceUse(ResourceKey(owner, contract.resource, 0), duration, 0)
        unit = SourceUnit(
            unit_id=unit_id,
            node_ref=stmt,
            region_ref=region.region,
            source_order=order,
            whole_body=stmt,
            kind=kind,
            execution_owner=owner,
            resources=(use,),
            original_guard=TRUE_GUARD,
            captured_values=_captured(stmt),
            carried_values=_carried(self, region),
            yields=(),
            effects=(effect,),
            intent=None,
        )
        object.__setattr__(unit, "_accesses", tuple(accesses))
        self.units.append(unit)
        return unit

    def _contract(self, stmt: Evaluate):
        try:
            return isa_contract_of(stmt.callable)
        except KeyError:
            raise UnitTreeError(
                "UNIT_TREE_INSTRUCTION_UNREGISTERED", stmt,
                "instruction %r has no contract in the target instruction table, so its "
                "resource and completion facts cannot be stated"
                % (type(stmt.callable).__name__,),
            ) from None

    def _vf_unit(self, region: _RegionState, scope: ExecutionScope,
                 owner: OwnerKey, generation, order: int) -> SourceUnit:
        if scope.kind != SIMD_VF_KIND:
            raise UnitTreeError(
                "UNIT_TREE_EXECUTION_MODE_UNSUPPORTED", scope,
                "execution region %r is not the declared SimdVF whole-task kind" % (scope.kind,),
            )
        unit_id = "unit:%d" % self._unit_index
        self._unit_index += 1
        members = []
        self._collect_region_body(scope.body, members)
        if not members:
            raise UnitTreeError(
                "UNIT_TREE_EMPTY_VF", scope,
                "the execution region computes nothing, so it has no resources or effects",
            )
        accesses = []
        effects = []
        resources = {}
        for member in members:
            contract = self._contract(member)
            member_accesses = self._accesses(member, unit_id, owner, generation)
            accesses.extend(member_accesses)
            effects.append(self._effect(member, unit_id, contract, member_accesses, generation))
            resources[contract.resource] = (
                resources.get(contract.resource, 0) + occupancy_ticks(contract.name)
            )
        uses = tuple(
            ResourceUse(ResourceKey(owner, pipe, 0), duration, 0)
            for pipe, duration in resources.items()
        )
        unit = SourceUnit(
            unit_id=unit_id,
            node_ref=scope,
            region_ref=region.region,
            source_order=order,
            whole_body=scope,
            kind=VF,
            execution_owner=owner,
            resources=uses,
            original_guard=TRUE_GUARD,
            captured_values=_captured(scope),
            carried_values=_carried(self, region),
            yields=(),
            effects=tuple(effects),
            intent=None,
        )
        object.__setattr__(unit, "_accesses", tuple(accesses))
        self.units.append(unit)
        return unit

    def _collect_region_body(self, node, members: list) -> None:
        if isinstance(node, Sequential):
            for item in node.body:
                self._collect_region_body(item, members)
            return
        if isinstance(node, Evaluate):
            members.append(node)
            return
        if isinstance(node, LetStmt):
            self.binders[id(node.var)] = ("let", node)
            self._register_definition(node, node)
            return
        raise UnitTreeError(
            "UNIT_TREE_VF_BODY_UNSUPPORTED", node,
            "%s inside a whole-VF region would split the scheduling unit" % (type(node).__name__,),
        )

    # -- regions ------------------------------------------------------------
    def _new_region(self, node_ref, parent: _RegionState | None, loop: For | None) -> _RegionState:
        region_id = "region:%d" % self._region_index
        self._region_index += 1
        state = _RegionState(
            region_id=region_id,
            node_ref=node_ref,
            parent_id=None if parent is None else parent.region_id,
            loop=loop,
        )
        self.regions.append(state)
        return state

    def _binding_summary(self, loop: For) -> RegionBindingSummary:
        row = self.bindings.get(id(loop))
        if row is None:
            raise UnitTreeError(
                "UNIT_TREE_REGION_BINDING_UNRESOLVED", loop,
                "the lowered loop has no source binding record",
            )
        pairs = [(row["induction_var"], row["actual_induction_var"])]
        for parameter, actual in zip(row["carried_params"], row["carried_actual_defs"], strict=True):
            pairs.append((parameter, actual))
        for value, actual in zip(row["yield_values"], row["yields_actual"], strict=True):
            pairs.append((value, actual))
        source_function = getattr(self, "_source_function", None) or self._authored
        bounds = tuple(row["actual_bounds"])
        return RegionBindingSummary(
            source_region=row["source_region"],
            induction_var=row["induction_var"],
            start=bounds[0],
            stop=bounds[1],
            step=bounds[2],
            carried_params=tuple(row["carried_params"]),
            carried_initials=tuple(row["carried_initials"]),
            yield_values=tuple(row["yield_values"]),
            captures=tuple(row["captures"]),
            actual_loop=loop,
            actual_induction_var=loop.induction_var,
            binding_map=tuple(pairs),
            actual_defs=tuple(row["actual_defs"]),
            actual_allocations=tuple(row["actual_allocations"]),
            actual_bounds=bounds,
        )

    def _freeze_region(self, state: _RegionState) -> Region:
        if state.loop is None:
            region = Region(
                region_id=state.region_id,
                node_ref=self.entry,
                parent_id=state.parent_id,
                direct_children=(),
                induction_var_ref=None,
                lower_expr=_int(0),
                stop_expr=_int(1),
                step_expr=_int(1),
                binding_summary=None,
                activation_guard=TRUE_GUARD,
            )
        else:
            loop = state.loop
            region = Region(
                region_id=state.region_id,
                node_ref=loop,
                parent_id=state.parent_id,
                direct_children=(),
                induction_var_ref=loop.induction_var,
                lower_expr=loop.start,
                stop_expr=loop.stop,
                step_expr=loop.step,
                binding_summary=self._binding_summary(loop),
                activation_guard=TRUE_GUARD,
            )
        state.region = region
        return region

    def _walk(self, state: _RegionState, node, owner: OwnerKey, generation) -> None:
        if isinstance(node, Sequential):
            for item in node.body:
                self._walk(state, item, owner, generation)
            return
        if isinstance(node, MeshScope):
            new_owner = OwnerKey(program_group=node.binding,
                                 core_kind=_core_kind(node.mesh), sid=None)
            self.binders[id(node.binding)] = ("mesh", node)
            self._walk(state, node.body, new_owner, generation)
            return
        if isinstance(node, LetStmt):
            self.binders[id(node.var)] = ("let", node)
            self._register_definition(node, node)
            return
        if isinstance(node, For):
            child = self._new_region(node, state, node)
            self._freeze_region(child)
            state.children.append(child.region)
            self.binders[id(node.induction_var)] = ("for", node)
            self._walk(child, node.body, owner, node.induction_var)
            return
        if isinstance(node, ExecutionScope):
            state.children.append(
                self._vf_unit(state, node, owner, generation, len(state.children))
            )
            return
        if isinstance(node, Evaluate):
            state.children.append(
                self._instruction_unit(state, node, owner, generation, len(state.children))
            )
            return
        if isinstance(node, (If, While)):
            raise UnitTreeError(
                "UNIT_TREE_CONTROL_UNSUPPORTED", node,
                "%s is outside this slice's unit boundary" % (type(node).__name__,),
            )
        raise UnitTreeError(
            "UNIT_TREE_STATEMENT_UNSUPPORTED", node,
            "%s is not a statement this unit boundary represents" % (type(node).__name__,),
        )

    def build(self, authored=None, source_function=None) -> UnitTree:
        self._authored = authored
        self._source_function = source_function
        self._bind_params()
        root = self._new_region(self.entry, None, None)
        self._freeze_region(root)
        self._walk(root, self.entry.body, OwnerKey(None, _DEVICE, None), _int(0))
        for state in self.regions:
            object.__setattr__(state.region, "direct_children", tuple(state.children))
        return UnitTree(
            source_snapshot=self.snapshot,
            entry_ref=self.entry,
            roots=(self.regions[0].region,),
            units=tuple(self.units),
        )


def unit_accesses(unit: SourceUnit) -> tuple:
    """The real AccessFact records of one unit (builder handoff, not a schema field)."""
    return tuple(getattr(unit, "_accesses", ()))


def _space_of(tensor_type) -> str:
    storage = getattr(tensor_type, "storage", None)
    if storage is None:
        raise UnitTreeError(
            "UNIT_TREE_OPERAND_ROOT_UNRESOLVED", tensor_type,
            "a tensor type carries no storage kind, so its space is unknown",
        )
    return getattr(storage, "name", str(storage)).lower()


def _carried(builder: _Builder, region: _RegionState) -> tuple:
    if region.loop is None:
        return ()
    row = builder.bindings.get(id(region.loop))
    if row is None:
        return ()
    return tuple(
        (row["carried_params"][index], row["carried_initials"][index],
         row["yield_values"][index])
        for index in range(len(row["carried_params"]))
    )


def _captured(whole_body) -> tuple:
    """Values a unit consumes that are bound outside its own body."""
    local = set()
    for item in _walk(whole_body):
        if isinstance(item, LetStmt):
            local.add(id(item.var))
        elif isinstance(item, For):
            local.add(id(item.induction_var))
        elif isinstance(item, MeshScope):
            local.add(id(item.binding))
    out = []
    seen = set()
    for var in _vars(whole_body):
        if id(var) in local or id(var) in seen:
            continue
        seen.add(id(var))
        out.append(var)
    return tuple(out)


def build_native_unit_tree(checked_module, region_bindings=None, *, target=None,
                           authored=None, source_function=None) -> UnitTree:
    """Build the typed UnitTree over the REAL lowered module.

    checked_module is the Module returned by finalize; region_bindings is the
    lowering record (defaults to module.metadata["region_bindings"]). authored is
    the authored HIR function, used only to bind source capture Vars to their
    actual function parameter; the parameter order is a declaration fact, not a
    name match.
    """
    module = checked_module
    entry = module.entry_function()
    payload = json.dumps(node_identity_payload(entry), sort_keys=True).encode()
    snapshot = SourceSnapshot(
        module_ref=module,
        entry_ref=entry,
        canonical_bytes=payload,
        identity=function_identity(entry),
    )
    builder = _Builder(snapshot, module, target)
    if region_bindings is not None:
        builder.bindings = {id(row["actual_loop"]): row for row in region_bindings}
    return builder.build(authored=authored, source_function=source_function)
