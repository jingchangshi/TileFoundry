"""structured-l/1 — ScheduleUnitTree: the real TIR structure as a schedulable hierarchy.

Approved design: artifacts/run03/w45_vadd_pair/RESOLVEDPLAN_BLOCKED_DESIGN.md
(sha256 3fef088cbb03f268e16761d3d0d391515fd392751994467b6b5868e7f63cb6a5).
This module implements §5 step 1 only: build the structure and lift one region.
No new z3 solve, no R rewrite, no C change.

WHAT THIS IS
------------
D1: the tree comes from the FINALIZED REAL TIR (Sequential / MeshScope / For /
    ExecutionScope / If / LetStmt / Evaluate), never from string prefixes guessed back out
    of the flat Problem. Source identity is bound from the ledger, which already records
    `iteration_path`, `scope_paths` and per-entry `path`. The path convention below is the
    ledger's own (`ledger.py` walk): a Sequential child appends `s<index>`, a loop appends
    `for:<var>`, Mesh/ExecutionScope append their type name, an If appends then/else.
D2: a schedulable leaf is a COMPLETE task, not one access:
      * one ExecutionScope -> one whole-VF leaf, owning every action inside it exactly
        once (the ledger's whole_vf_tasks() is that mapping);
      * one Evaluate       -> one DMA/copy leaf, owning the actions at that statement;
      * LetStmt is a binding PRELUDE, not a schedulable unit;
      * If is kept as a compound unit with both branches (D1: deliberately more
        conservative than TL's guarded-sibling expansion).

TRANSPARENT WRAPPERS
--------------------
A plain `Sequential` is a LIST, not a barrier: an extra pair of parentheses must not change
the structure (D1). `MeshScope` supplies the region's participants/execution mode, and the
design's root region IS its body (`RootRegion` inside `MeshScope(AIV)`, §3.2), so it is
transparent too, with its mesh recorded in `UnitTree.mesh`. This is what makes vadd come out
as `RootRegion(n=1) -> LoopUnit(i)` rather than with two extra wrapper layers.

WHAT THIS IS NOT
----------------
The tree is STRUCTURE, not a schedule. It carries no time, no II, no start order beyond
source order. A leaf list being small says nothing about performance, and nothing here is
evidence about hardware.

ACTION INVENTORY IS PRESERVED
-----------------------------
D2 requires that shrinking a VF into one unit must NOT delete its accesses: C validates
against them. Every ledger entry is therefore bound to exactly one leaf, and
`UnitTree.unbound_entries()` must be empty; tests assert the total is unchanged
(vadd 17 entries, decode_dyn 2476).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

SCHEMA = "structured-l/1"

KIND_LOOP = "LOOP_UNIT"
KIND_VF = "VF_LEAF"
KIND_DMA = "DMA_LEAF"
KIND_IF = "IF_UNIT"

#: Statements that bind a value and are therefore a prelude, never a schedulable unit.
_BINDING_STATEMENTS = frozenset({"LetStmt", "Pass", "AssertStmt", "AssessStmt"})


class UnitTreeError(RuntimeError):
    """A structure the tree cannot represent is refused by name, never silently dropped."""

    def __init__(self, code, message, source_id="", expected=None, actual=None):
        self.code = code
        self.source_id = source_id
        self.expected = expected
        self.actual = actual
        super().__init__("%s: %s (source=%r expected=%r actual=%r)"
                         % (code, message, source_id, expected, actual))


# --------------------------------------------------------------------------- records
@dataclass(frozen=True)
class Unit:
    """One node of the structure. A control unit has children; a leaf has actions."""

    unit_id: str
    kind: str
    source_order: int
    parent_id: Optional[str]
    leaf_action_ids: tuple = ()
    child_ids: tuple = ()
    loop_var: Optional[str] = None
    loop_domain: Optional[dict] = None
    scope_path: Optional[str] = None
    tir_node: Optional[str] = None
    ledger_path: Optional[str] = None

    @property
    def is_leaf(self) -> bool:
        return self.kind in (KIND_VF, KIND_DMA)

    @property
    def action_count(self) -> int:
        return len(self.leaf_action_ids)


@dataclass(frozen=True)
class Region:
    """A direct-child list solved in one activation context (D3).

    `kind` is 'sequence' for a straight list (root/branch) and 'loop' for a loop body,
    where the loop's own iteration coordinate is the region's time base.
    """

    region_id: str
    owner_unit: str
    kind: str
    direct_child_ids: tuple
    loop_var: Optional[str] = None
    loop_domain: Optional[dict] = None


@dataclass(frozen=True)
class UnitTree:
    """The structure plus the region index. Immutable; built once from the real TIR."""

    schema: str
    root_region_id: str
    units: dict
    regions: dict
    action_to_unit: dict
    mesh: dict = field(default_factory=dict)
    source_identity: dict = field(default_factory=dict)

    def unit(self, unit_id) -> Unit:
        return self.units[unit_id]

    def region(self, region_id) -> Region:
        return self.regions[region_id]

    def children(self, unit_id) -> list:
        return [self.units[c] for c in self.units[unit_id].child_ids]

    def region_children(self, region_id) -> list:
        return [self.units[c] for c in self.regions[region_id].direct_child_ids]

    def leaves(self) -> list:
        return [u for u in self.units.values() if u.is_leaf]

    def leaves_in_source_order(self) -> list:
        return sorted(self.leaves(), key=lambda u: u.source_order)

    def bound_action_ids(self) -> set:
        out = set()
        for u in self.units.values():
            out.update(u.leaf_action_ids)
        return out

    def unbound_entries(self, ledger) -> list:
        """Ledger actions with no leaf owner. Must be empty; D2 forbids dropping them."""
        known = self.bound_action_ids()
        return [e.action_id for e in ledger.entries if e.action_id not in known]

    def actions_of(self, unit_id) -> tuple:
        return self.units[unit_id].leaf_action_ids

    def unit_ids_in(self, region_id) -> list:
        return list(self.regions[region_id].direct_child_ids)

    def loop_units(self) -> list:
        return [u for u in self.units.values() if u.kind == KIND_LOOP]


# ------------------------------------------------------------------- path helpers
def _join(path: tuple) -> str:
    return "/".join(path)


def _stmts(stmt):
    """Statement list of a container, or None when `stmt` is not a container."""
    body = getattr(stmt, "body", None)
    if body is None:
        return None
    return list(body) if isinstance(body, (list, tuple)) else [body]


def _kind_of(stmt) -> str:
    return type(stmt).__name__


def _as_int(x):
    if isinstance(x, int) and not isinstance(x, bool):
        return x
    v = getattr(x, "value", None)
    return v if isinstance(v, int) and not isinstance(v, bool) else None


class _Builder:
    def __init__(self, ledger, function):
        self.ledger = ledger
        self.function = function
        self.units = {}
        self.regions = {}
        self.action_to_unit = {}
        self.mesh = {}
        self._order = 0
        self._vf_by_scope = {t["scope"]: t for t in _whole_vf_tasks(ledger)}
        self._entry_by_path = {}
        for e in ledger.entries:
            self._entry_by_path.setdefault(e.path, []).append(e)

    # -- public ------------------------------------------------------------
    def build(self) -> UnitTree:
        body = _stmts(self.function.body)
        if body is None:
            raise UnitTreeError("TIR_BODY_NOT_A_SEQUENCE", "function body is not a container",
                                "", "a statement list", _kind_of(self.function.body))
        # The function body is the outermost Sequential; the ledger numbers its children
        # from s0 without indexing the body itself.
        root_children = self._visit(self.function.body, ())
        self.regions["root"] = Region(region_id="root", owner_unit="root", kind="sequence",
                                      direct_child_ids=tuple(root_children))
        tree = UnitTree(
            schema=SCHEMA,
            root_region_id="root",
            units=dict(self.units),
            regions=dict(self.regions),
            action_to_unit=dict(self.action_to_unit),
            mesh=dict(self.mesh),
            source_identity={
                "function": getattr(self.ledger, "function", None),
                "scope_paths": tuple(getattr(self.ledger, "scope_paths", ()) or ()),
                "entry_count": len(self.ledger.entries),
            },
        )
        self._check_vf_leaves_match_ledger(tree)
        self._check_every_entry_bound(tree)
        return tree

    # -- the walk ----------------------------------------------------------
    def _visit(self, stmt, own_path):
        """Collect schedulable children of `stmt`, splicing transparent wrappers.

        `own_path` is the ledger path OF `stmt`. Path construction mirrors the ledger walk
        exactly, while the returned list is flattened for the tree's child list.
        """
        kind = _kind_of(stmt)
        if kind == "Sequential":
            out = []
            for index, child in enumerate(_stmts(stmt) or []):
                out.extend(self._visit(child, own_path + ("s%d" % index,)))
            return out
        if kind == "MeshScope":
            self.mesh.setdefault("scopes", []).append(
                {"ledger_path": _join(own_path), "tir_node": kind,
                 "display": _describe_mesh(stmt)})
            if _stmts(stmt) is None:
                raise UnitTreeError("MESH_WITHOUT_BODY", "MeshScope has no body",
                                    _join(own_path), "a body", None)
            return self._visit(stmt.body, own_path + ("MeshScope",))
        if kind in _BINDING_STATEMENTS:
            return []
        if kind == "For":
            return [self._loop(stmt, own_path)]
        if kind == "If":
            return [self._if(stmt, own_path)]
        if kind == "ExecutionScope":
            return [self._vf_leaf(stmt, own_path)]
        if kind == "Evaluate":
            return [self._dma_leaf(own_path)]
        raise UnitTreeError("UNREPRESENTED_TIR_NODE",
                            "the structure has a node this tree does not classify",
                            _join(own_path), "known TIR node", kind)

    def _loop(self, stmt, own_path):
        var = getattr(getattr(stmt, "induction_var", None), "name", None)
        domain = {"start": _as_int(getattr(stmt, "start", None)),
                  "stop": _as_int(getattr(stmt, "stop", None)),
                  "step": _as_int(getattr(stmt, "step", None)) or 1}
        if _stmts(stmt) is None:
            raise UnitTreeError("LOOP_WITHOUT_BODY", "loop has no body",
                                _join(own_path), "a body", None)
        uid = _uid(own_path, KIND_LOOP)
        child_ids = self._visit(stmt.body, own_path + ("for:%s" % var,))
        self.regions[uid] = Region(region_id=uid, owner_unit=uid, kind="loop",
                                   direct_child_ids=tuple(child_ids), loop_var=var,
                                   loop_domain=domain)
        return self._add(Unit(unit_id=uid, kind=KIND_LOOP, source_order=self._take_order(),
                              parent_id=None, child_ids=tuple(child_ids), loop_var=var,
                              loop_domain=domain, tir_node="For",
                              ledger_path=_join(own_path)))

    def _if(self, stmt, own_path):
        uid = _uid(own_path, KIND_IF)
        child_ids = []
        for br_name in ("then", "else"):
            br = getattr(stmt, "%s_body" % br_name, None)
            if br is None:
                continue
            kid = self._visit(br, own_path + (br_name,))
            self.regions["%s/%s" % (uid, br_name)] = Region(
                region_id="%s/%s" % (uid, br_name), owner_unit=uid, kind="sequence",
                direct_child_ids=tuple(kid))
            child_ids.extend(kid)
        return self._add(Unit(unit_id=uid, kind=KIND_IF, source_order=self._take_order(),
                              parent_id=None, child_ids=tuple(child_ids), tir_node="If",
                              ledger_path=_join(own_path)))

    def _vf_leaf(self, stmt, own_path):
        scope = _join(own_path + ("ExecutionScope",))
        task = self._vf_by_scope.get(scope)
        if task is None:
            raise UnitTreeError("VF_SCOPE_WITHOUT_TASK",
                                "ExecutionScope with no whole_vf_tasks entry",
                                scope, "one task per scope", None)
        uid = _uid(own_path, KIND_VF)
        action_ids = tuple(task["action_ids"])
        for aid in action_ids:
            self.action_to_unit[aid] = uid
        return self._add(Unit(unit_id=uid, kind=KIND_VF, source_order=self._take_order(),
                              parent_id=None, leaf_action_ids=action_ids,
                              scope_path=scope, tir_node="ExecutionScope",
                              ledger_path=_join(own_path)))

    def _dma_leaf(self, own_path):
        path = _join(own_path)
        entries = self._entry_by_path.get(path) or []
        if not entries:
            raise UnitTreeError("DMA_STATEMENT_WITHOUT_ACTION",
                                "an Evaluate statement owns no ledger action",
                                path, ">=1 action", 0)
        uid = _uid(own_path, KIND_DMA)
        action_ids = tuple(e.action_id for e in entries)
        for aid in action_ids:
            self.action_to_unit[aid] = uid
        return self._add(Unit(unit_id=uid, kind=KIND_DMA, source_order=self._take_order(),
                              parent_id=None, leaf_action_ids=action_ids,
                              tir_node="Evaluate", ledger_path=path))

    # -- checks ------------------------------------------------------------
    def _check_vf_leaves_match_ledger(self, tree):
        """Every ledger VF scope must have become a leaf, and no leaf may invent one."""
        leaf_scopes = {u.scope_path for u in tree.leaves() if u.kind == KIND_VF}
        declared = set(getattr(self.ledger, "scope_paths", ()) or ())
        if leaf_scopes != declared:
            raise UnitTreeError(
                "VF_LEAVES_DISAGREE_WITH_LEDGER",
                "ExecutionScope leaves and ledger scope_paths differ",
                "", sorted(declared), sorted(leaf_scopes))

    def _check_every_entry_bound(self, tree):
        """D2: shrinking a scope into one unit must not delete its accesses."""
        missing = tree.unbound_entries(self.ledger)
        if missing:
            raise UnitTreeError(
                "LEDGER_ACTION_UNBOUND", "ledger actions belong to no leaf",
                missing[0], "all entries bound", len(missing))

    def _add(self, unit: Unit) -> str:
        self.units[unit.unit_id] = unit
        return unit.unit_id

    def _take_order(self) -> int:
        order = self._order
        self._order += 1
        return order


def _uid(own_path, kind):
    return "%s#%s" % (_join(own_path), kind)


def _describe_mesh(stmt):
    """Mesh participants, read from the node itself rather than from a name in the path."""
    mesh = getattr(stmt, "mesh", None)
    if mesh is None:
        return None
    topo = getattr(mesh, "topologies", None)
    names = getattr(mesh, "names", None)
    return {"topologies": [getattr(t, "name", None) for t in topo] if topo else None,
            "names": list(names) if names else None}


def _whole_vf_tasks(ledger):
    from tilefoundry_ascend.schedule.ledger import whole_vf_tasks
    return whole_vf_tasks(ledger)


def build_unit_tree(ledger, function) -> UnitTree:
    """Build the unit tree from the FINALIZED function plus its ledger.

    Both inputs are required: the TIR gives the true nesting and source order, the ledger
    gives the action identity and the whole-VF grouping. Neither alone is sufficient, and
    guessing the structure from the flat Problem is exactly what D1 forbids.
    """
    return _Builder(ledger, function).build()


# ------------------------------------------------------- single-region lift
# Design D6: this is an increment on the SAME authority, not a parallel framework.
# §5 step 1 only needs the lift to work for ONE region; a full /2 migration is later.
SCHEMA_LIFT = "structured-l/1-lift"


@dataclass(frozen=True)
class RegionTaskBinding:
    """One flat Problem task, bound to the leaf unit that owns its actions.

    `source_task_id` and the action ids are kept verbatim: the lift is a re-projection of
    the existing inventory onto the structure, NOT a renumbering. D1 forbids inventing a
    new dense index in place of the source identity.
    """

    source_task_id: str
    unit_id: str
    op: str
    pipe: str
    action_ids: tuple
    loop: Optional[str]


@dataclass(frozen=True)
class RegionLift:
    """A single region projected from the flat Problem, still carrying source identity.

    `optimisation_variable_count` is the number of DISTINCT leaf units the region schedules
    (vadd: 4, not the 8 instruction tasks). `source_task_count` is unchanged, so no source
    obligation is lost and none is invented.
    """

    schema: str
    region_id: str
    region_kind: str
    unit_ids: tuple
    task_bindings: tuple
    source_task_count: int
    source_access_count: int
    optimised_variable_ids: tuple
    loop_var: Optional[str] = None
    loop_domain: Optional[dict] = None

    def to_dict(self):
        return {
            "schema": self.schema, "region_id": self.region_id,
            "region_kind": self.region_kind, "unit_ids": list(self.unit_ids),
            "task_bindings": [
                {"source_task_id": b.source_task_id, "unit_id": b.unit_id, "op": b.op,
                 "pipe": b.pipe, "action_ids": list(b.action_ids), "loop": b.loop}
                for b in self.task_bindings],
            "source_task_count": self.source_task_count,
            "source_access_count": self.source_access_count,
            "optimised_variable_ids": list(self.optimised_variable_ids),
            "loop_var": self.loop_var, "loop_domain": self.loop_domain,
        }


def lift_region(tree: UnitTree, problem, region_id) -> RegionLift:
    """Project the flat Problem onto ONE region of the tree.

    The point of the lift (§5 step 1, acceptance 4) is that the region's optimisation
    variables are its DIRECT CHILD UNITS, using the same unit/action identity as the tree.
    It must not re-flatten back to the instruction-level task list: for vadd that is 4
    units (LoadX, LoadZ, WholeVF, StoreY) rather than 8 instruction tasks, while all 8
    tasks and 17 actions remain accounted for in `task_bindings`.
    """
    region = tree.regions.get(region_id)
    if region is None:
        raise UnitTreeError("REGION_UNKNOWN", "no such region in the unit tree",
                            region_id, "an existing region", None)
    tasks_in = list(getattr(problem, "tasks", ()) or ())
    accesses_in = list(getattr(problem, "accesses", ()) or ())
    child_units = set(region.direct_child_ids)

    bindings = []
    for task in tasks_in:
        source_task_id = str(getattr(task, "stable_id", None) or getattr(task, "task_id", ""))
        action_ids = tuple(getattr(task, "action_ids", ()) or ())
        owners = {tree.action_to_unit.get(aid) for aid in action_ids}
        owners.discard(None)
        if not owners:
            raise UnitTreeError("TASK_HAS_NO_OWNING_UNIT",
                                "a flat task's actions belong to no unit",
                                source_task_id, "1 owner", owners)
        if len(owners) > 1:
            raise UnitTreeError("TASK_SPANS_SEVERAL_UNITS",
                                "a flat task's actions span more than one leaf unit; the "
                                "leaf boundary is wrong",
                                source_task_id, 1, sorted(owners))
        owner = next(iter(owners))
        bindings.append(RegionTaskBinding(
            source_task_id=source_task_id, unit_id=owner, op=getattr(task, "op", None),
            pipe=getattr(task, "pipe", None), action_ids=action_ids,
            loop=getattr(task, "loop", None)))

    # The lift covers a whole tree, so it binds every task; the REGION's optimisation
    # variables are only the direct children, which is the difference that matters.
    bound_units = {b.unit_id for b in bindings}
    if bound_units - set(tree.units):
        raise UnitTreeError("BINDING_POINTS_TO_UNKNOWN_UNIT",
                            "lift produced a unit id that is not in the tree",
                            "", sorted(tree.units), sorted(bound_units))

    return RegionLift(
        schema=SCHEMA_LIFT, region_id=region_id, region_kind=region.kind,
        unit_ids=tuple(region.direct_child_ids),
        task_bindings=tuple(bindings),
        source_task_count=len(tasks_in),
        source_access_count=len(accesses_in),
        optimised_variable_ids=tuple(region.direct_child_ids),
        loop_var=region.loop_var, loop_domain=region.loop_domain)
