"""Whole-task symbolic materialization: the ONE plan becomes a private candidate.

This is the apply stage of the frozen transaction. It consumes the real entry
PrimFunction, the SAT ResolvedPlan and (when the later stages exist) the
CompletionPlan/StoragePlan, and returns a PRIVATE candidate Module plus a typed
SemanticSourceMap. The original module, its entry and every native child object
stay untouched: the candidate owns the fresh functions, fresh binders and the
source map.

SCOPE OF THIS SLICE (S0), stated so it cannot be mistaken for the whole spec:
- every loop that the plan owns is materialized as ONE symbolic p loop:
  p in [0, N+H), i = a + k*h, k = p - shift, with N = max(0, ceil((b-a)/h)).
  For the S0 plan all shifts are 0 and a=0, h=1, so k=i=p by pure constant
  folding; the body is cloned once and is never unrolled over N.
- a loop whose plan needs a non-foldable stage shift, non-zero lower bound or
  non-unit step is refused by name; that rewrite belongs to S1/S2 and is not
  faked here.
- completion is not implemented yet and the S0 plan's storage cannot cover the
  register-space buffers, so apply consumes completion/storage when they are
  given and records their absence instead of inventing sync or slot views.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, field

from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.core.module import Module
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
    Return,
    Sequential,
    While,
)
from tilefoundry.ir.types.utils import static_dim_value

from .identity import function_identity

__all__ = [
    "APPLY_ROLE_ALLOCATION",
    "APPLY_ROLE_COMPUTE",
    "APPLY_ROLE_CONTROL",
    "APPLY_ROLE_SYNC",
    "SemanticSourceMap",
    "SourceMapEntry",
    "SymbolicApplyError",
    "apply_symbolic_pipeline",
]

def mesh_frames(function) -> list:
    """Every real MeshScope node of one function, in traversal order."""
    found: list = []

    def visit(node):
        if node is None:
            return
        if isinstance(node, MeshScope):
            found.append(node)
            visit(node.body)
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                visit(item)
            return
        for name in getattr(node, "__dataclass_fields__", ()) or ():
            if name in ("type", "metadata", "origin", "annotation", "loc"):
                continue
            visit(getattr(node, name, None))

    visit(getattr(function, "body", None))
    return found


#: The emitter's OWN codes (codegen/emit.py:547-557, 673), reused here so the
#: front end and the backend never carry two names for one failure: a candidate
#: that cannot be emitted is refused with the code the emitter would raise.
#: TL_ALLOC_OUTSIDE_KERNEL_FRAME has no emitter counterpart yet - it is the
#: case the packed-API lowering crashes on (make_packed_api.cc: "variables
#: (buf_dyn_shmem,) are used, but not passed in as API arguments"), and naming
#: it here is what keeps that crash out of the backend.
KERNEL_FRAME_CODES = {
    "missing": "TL_MISSING_MESH_SCOPE",
    "multiple": "TL_MULTIPLE_MESH_SCOPES",
    "levels": "TL_UNSUPPORTED_MESH",
    "level": "TL_UNSUPPORTED_MESH_LEVEL",
    "alloc": "TL_ALLOC_OUTSIDE_KERNEL_FRAME",
}


def alloc_nodes(function) -> list:
    """Every AllocTensor definition in traversal order (outer-to-inner)."""
    found: list = []

    def visit(node, inside_frame):
        if node is None:
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                visit(item, inside_frame)
            return
        if isinstance(node, MeshScope):
            visit(node.body, True)
            return
        if isinstance(node, LetStmt) and isinstance(node.value, Call) \
                and isinstance(node.value.target, AllocTensor):
            found.append((inside_frame, node))
            return
        for name in getattr(node, "__dataclass_fields__", ()) or ():
            if name in ("type", "metadata", "origin", "annotation", "loc"):
                continue
            visit(getattr(node, name, None), inside_frame)

    visit(getattr(function, "body", None), False)
    return found


def require_kernel_frame(function):
    """A candidate the backend cannot lower/emit is not a candidate.

    The compiled path needs a real grid/kernel frame: lowering emits
    T.Kernel and the alloc_shared statements from an actual MeshScope, and a
    body without one fails inside the backend (make_packed_api.cc) rather than
    here. Every allocation must also sit INSIDE that frame, because hoisting an
    alloc out of the kernel is exactly the case the backend reports as
    "used, but not passed in as API arguments". The codes are the emitter's
    own (KERNEL_FRAME_CODES): one failure has one name in both stages.
    """
    frames = mesh_frames(function)
    if not frames:
        raise SymbolicApplyError(
            KERNEL_FRAME_CODES["missing"],
            "the candidate carries no real MeshScope, so lowering cannot produce "
            "a kernel frame and the backend cannot emit it",
        )
    if len(frames) > 1:
        raise SymbolicApplyError(
            KERNEL_FRAME_CODES["multiple"],
            "the candidate nests %d MeshScope frames; this slice emits one kernel "
            "per entry and a nested grid is not a declared shape" % (len(frames),),
        )
    mesh = frames[0].mesh
    levels = tuple(getattr(mesh, "topologies", ()) or ())
    if len(levels) != 1:
        raise SymbolicApplyError(
            KERNEL_FRAME_CODES["levels"],
            "the mesh states %d level(s); this slice emits one AIV instance"
            % (len(levels),),
        )
    level = levels[0]
    if getattr(level, "name", "") != "aiv":
        raise SymbolicApplyError(
            KERNEL_FRAME_CODES["level"],
            "the mesh level is %r; this slice emits the AIV vector engine"
            % (getattr(level, "name", level),),
        )
    raw_extent = getattr(level, "size", None)
    try:
        extent = int(raw_extent)
    except (TypeError, ValueError):
        extent = -1
    if extent < 1:
        raise SymbolicApplyError(
            KERNEL_FRAME_CODES["level"],
            "the mesh level declares grid extent %r, which is not a launch shape"
            % (getattr(level, "size", None),),
        )
    outside = [node for inside, node in alloc_nodes(function) if not inside]
    if outside:
        raise SymbolicApplyError(
            KERNEL_FRAME_CODES["alloc"],
            "the candidate allocates %d tensor(s) outside its kernel frame; the "
            "backend cannot pass such a buffer through the packed API"
            % (len(outside),),
        )
    return frames[0]


APPLY_ROLE_COMPUTE = "COMPUTE"
APPLY_ROLE_CONTROL = "CONTROL"
APPLY_ROLE_ALLOCATION = "ALLOCATION"
APPLY_ROLE_SYNC = "SYNC"
APPLY_ROLE_STORAGE = "STORAGE"


class SymbolicApplyError(RuntimeError):
    """A named refusal from symbolic materialization; the original stays untouched."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__("%s: %s" % (code, detail))


@dataclass(frozen=True)
class SourceMapEntry:
    """One source occurrence and the candidate nodes/bindings it became."""

    entry_id: str
    source_ref: object
    candidate_refs: tuple
    role: str
    binder_pairs: tuple = ()
    substitutions: tuple = ()
    unit_id: str = ""


@dataclass(frozen=True)
class SemanticSourceMap:
    """The typed correspondence; it records what apply did, it is not a program."""

    source_identity: str
    plan_identity: str
    candidate_identity: str
    entries: tuple = ()

    def by_role(self, role: str) -> tuple:
        return tuple(entry for entry in self.entries if entry.role == role)


def _int_dim(expr, what: str) -> int:
    value = static_dim_value(expr)
    if value is None:
        raise SymbolicApplyError(
            "APPLY_DYNAMIC_BOUNDS_UNIMPLEMENTED",
            "loop %s is not a static integer; symbolic bounds are S1, not faked here"
            % (what,),
        )
    return int(value)


def _is_allocation(value) -> bool:
    target = getattr(value, "target", None)
    return isinstance(target, (AllocTensor, TensorView))


def _count_schedule_units(stmt) -> int:
    """Whole scheduling statements of ONE region (nested loops belong to their own).

    Sync instructions are NOT scheduling units: they are events the completion
    decided to place around a unit, so counting them would make the candidate's
    unit count disagree with the plan's the moment an event is emitted.
    """
    if isinstance(stmt, Evaluate):
        schema = getattr(type(getattr(stmt, "callable", None)), "_op_schema", None)
        if schema is not None and getattr(schema, "category", "") == "sync":
            return 0
        return 1
    if isinstance(stmt, ExecutionScope):
        return 1
    if isinstance(stmt, Sequential):
        return sum(_count_schedule_units(item) for item in stmt.body)
    if isinstance(stmt, MeshScope):
        return _count_schedule_units(stmt.body)
    if isinstance(stmt, LetStmt):
        return 0
    if isinstance(stmt, (For, If, While)):
        return 0
    return 0




# ------------------------------------------------------- event site placement
def _sites_by_unit(plan, completion):
    """(region_id, unit_id) -> [(anchor, key, site_id)] from the REAL completion.

    The plan and completion are the source of truth: a site whose channel the
    completion does not carry is refused here rather than placed on a guess.
    """
    channels = {channel.channel_id: channel
                for channel in getattr(completion, "channels", ()) or ()}
    by_consumer = {}
    for channel in channels.values():
        for consumer in getattr(channel, "consumers", ()) or ():
            by_consumer[consumer] = channel
    by_producer = {getattr(channel, "producer", None): channel
                   for channel in channels.values()}
    placement = {}
    for site in getattr(completion, "sites", ()) or ():
        site_id = getattr(site, "site_id", "")
        channel = by_consumer.get(site_id) or by_producer.get(site_id)
        if channel is None:
            raise SymbolicApplyError(
                "APPLY_EVENT_CHANNEL_MISSING",
                "completion site %s belongs to no channel, so no instruction can "
                "be placed for it" % (site_id,),
            )
        key = event_key_of_site(site, channel)
        unit = getattr(site, "task_ref", None)
        unit_id = str(getattr(unit, "unit_id", "") or "")
        region_id = str(getattr(site, "region_id", "") or "")
        placement.setdefault((region_id, unit_id), []).append(
            (str(getattr(site, "anchor", "")), key, site_id))
    return placement


def _sync_statement(key, site_id: str):
    from tilefoundry.ir.tir.stmts import Evaluate

    return Evaluate(callable=event_instruction(key), args=())


class _Cloner:
    """Fresh-binder whole-task clone over one real entry; external Vars keep identity."""

    def __init__(self, entry: PrimFunction, plan, completion=None) -> None:
        self.entry = entry
        self.plan = plan
        self.entries = []
        self.region_by_loop = {}
        self.region_id_by_loop = {}
        for region in getattr(plan, "regions", ()) or ():
            node = getattr(region.region_ref, "node_ref", None)
            if isinstance(node, For):
                self.region_by_loop[id(node)] = region
                self.region_id_by_loop[id(node)] = str(
                    getattr(region.region_ref, "region_id", ""))
        self._index = 0
        self.candidate_for_by_source = {}
        # Event placement, derived from the REAL completion: (region, unit) ->
        # the anchors that must appear around that unit's statement, and the
        # exits that must appear at the end of the region's body.
        self.sites_by_unit = {}
        self.sites_at_exit = {}
        self.placed_sites = []
        if completion is not None:
            for (region_id, unit_id), rows in _sites_by_unit(plan, completion).items():
                for anchor, key, site_id in rows:
                    if anchor == SITE_EXIT:
                        self.sites_at_exit.setdefault(region_id, []).append((key, site_id))
                    else:
                        self.sites_by_unit.setdefault((region_id, unit_id), []).append(
                            (anchor, key, site_id))
        # Which unit each source statement belongs to, from the plan's own tasks.
        self.unit_of_statement = {}
        for region in getattr(plan, "regions", ()) or ():
            region_id = str(getattr(region.region_ref, "region_id", ""))
            for task in getattr(region, "tasks", ()) or ():
                unit = getattr(task, "task_ref", None)
                node = getattr(unit, "node_ref", None)
                if node is not None:
                    self.unit_of_statement[id(node)] = (
                        region_id, str(getattr(unit, "unit_id", "")))
        self._region_stack = []
        self._current_region = ""
        root = (getattr(plan, "regions", ()) or (None,))[0]
        self._root_region_id = ("" if root is None else
                                str(getattr(root.region_ref, "region_id", "")))

    # -- records ------------------------------------------------------------
    def _record(self, source_ref, candidate_refs, role, *, binder_pairs=(),
                substitutions=(), unit_id=""):
        self.entries.append(SourceMapEntry(
            entry_id="source-map:%d" % self._index,
            source_ref=source_ref,
            candidate_refs=tuple(candidate_refs),
            role=role,
            binder_pairs=tuple(binder_pairs),
            substitutions=tuple(substitutions),
            unit_id=unit_id,
        ))
        self._index += 1

    # -- expressions --------------------------------------------------------
    def expression(self, expr, env):
        if isinstance(expr, Var):
            return env.get(id(expr), expr)
        if isinstance(expr, Constant):
            return expr
        if isinstance(expr, Call):
            args = tuple(self.expression(arg, env) for arg in expr.args)
            return Call(type=expr.type, target=expr.target, args=args)
        if isinstance(expr, tuple):
            return tuple(self.expression(item, env) for item in expr)
        raise SymbolicApplyError(
            "APPLY_EXPR_UNSUPPORTED",
            "expression %s is outside the whole-task clone domain" % (type(expr).__name__,),
        )


    # -- event emission -----------------------------------------------------
    def _unit_of(self, stmt):
        """(region_id, unit_id) for a statement the plan schedules, else None."""
        found = self.unit_of_statement.get(id(stmt))
        if found is not None:
            return found
        # A whole VF unit's node is the enclosing ExecutionScope; its inner
        # Evaluate statements belong to that same unit, so an event anchored
        # "before_task" for the VF lands before the scope, not inside it.
        return None

    def _emit(self, rows, body):
        from tilefoundry.ir.tir.stmts import Evaluate

        for key, site_id in rows:
            statement = Evaluate(callable=event_instruction(key), args=())
            body.append(statement)
            # The SYNC record names the SITE, not a source statement: the site
            # is what the completion decided, and the correspondence the
            # verifier checks is plan <-> candidate, never this map.
            self._record(site_id, (statement,), APPLY_ROLE_SYNC, unit_id=site_id)
            self.placed_sites.append((site_id, key))

    def _before_events(self, stmt, env):
        unit = self._unit_of(stmt)
        if unit is None:
            return []
        body = []
        rows = [(key, site_id) for anchor, key, site_id in
                self.sites_by_unit.get(unit, ()) if anchor == SITE_BEFORE]
        self._emit(rows, body)
        return body

    def _after_events(self, stmt, env):
        unit = self._unit_of(stmt)
        if unit is None:
            return []
        body = []
        rows = [(key, site_id) for anchor, key, site_id in
                self.sites_by_unit.get(unit, ()) if anchor == SITE_AFTER]
        self._emit(rows, body)
        return body

    def _exit_events(self, region_id: str):
        body = []
        self._emit(self.sites_at_exit.get(region_id, ()), body)
        return body

    # -- statements ---------------------------------------------------------
    def statement(self, stmt, env):
        if isinstance(stmt, Sequential):
            body = []
            # Siblings share the evolving environment, as their definitions do.
            for item in stmt.body:
                body.extend(self._before_events(item, env))
                candidate, env = self.statement(item, env)
                body.append(candidate)
                body.extend(self._after_events(item, env))
            return Sequential(tuple(body)), env
        if isinstance(stmt, LetStmt):
            value = self.expression(stmt.value, env)
            fresh = Var(name=stmt.var.name, type=stmt.var.type, is_const=stmt.var.is_const)
            inner = dict(env)
            inner[id(stmt.var)] = fresh
            candidate = LetStmt(var=fresh, value=value)
            role = APPLY_ROLE_ALLOCATION if _is_allocation(value) else APPLY_ROLE_CONTROL
            self._record(stmt, (candidate,), role,
                         binder_pairs=((stmt.var, fresh),))
            return candidate, inner
        if isinstance(stmt, For):
            region = self.region_by_loop.get(id(stmt))
            if region is None:
                raise SymbolicApplyError(
                    "APPLY_REGION_UNRESOLVED",
                    "loop has no region in the plan, so it has no schedule to materialize",
                )
            return self._symbolic_for(stmt, region, env), env
        if isinstance(stmt, MeshScope):
            inner = dict(env)
            fresh = Var(name=stmt.binding.name, type=stmt.binding.type,
                        is_const=stmt.binding.is_const)
            inner[id(stmt.binding)] = fresh
            body, _ = self.statement(stmt.body, inner)
            candidate = MeshScope(mesh=stmt.mesh, binding=fresh, body=body)
            self._record(stmt, (candidate,), APPLY_ROLE_CONTROL,
                         binder_pairs=((stmt.binding, fresh),))
            return candidate, env
        if isinstance(stmt, ExecutionScope):
            body, _ = self.statement(stmt.body, dict(env))
            candidate = ExecutionScope(kind=stmt.kind, body=body)
            self._record(stmt, (candidate,), APPLY_ROLE_COMPUTE)
            return candidate, env
        if isinstance(stmt, If):
            cond = self.expression(stmt.cond, env)
            then_body, _ = self.statement(stmt.then_body, dict(env))
            else_body, _ = self.statement(stmt.else_body, dict(env))
            candidate = If(cond=cond, then_body=then_body, else_body=else_body)
            self._record(stmt, (candidate,), APPLY_ROLE_CONTROL)
            return candidate, env
        if isinstance(stmt, Evaluate):
            args = tuple(self.expression(arg, env) for arg in stmt.args)
            candidate = Evaluate(callable=stmt.callable, args=args)
            self._record(stmt, (candidate,), APPLY_ROLE_COMPUTE)
            return candidate, env
        if isinstance(stmt, Return):
            candidate = Return()
            self._record(stmt, (candidate,), APPLY_ROLE_CONTROL)
            return candidate, env
        if isinstance(stmt, While):
            raise SymbolicApplyError(
                "APPLY_CONTROL_UNSUPPORTED",
                "a While statement is outside the native symbolic loop domain",
            )
        raise SymbolicApplyError(
            "APPLY_STATEMENT_UNSUPPORTED",
            "statement %s is outside the whole-task clone domain" % (type(stmt).__name__,),
        )

    def _symbolic_for(self, stmt: For, region, env):
        lower = _int_dim(stmt.start, "lower")
        stop = _int_dim(stmt.stop, "stop")
        step = _int_dim(stmt.step, "step")
        if step <= 0:
            raise SymbolicApplyError("APPLY_LOOP_STEP", "a symbolic loop needs a positive step")
        iterations = max(0, -(-(stop - lower) // step))
        tasks = getattr(region, "tasks", ()) or ()
        if not tasks:
            raise SymbolicApplyError(
                "APPLY_EMPTY_REGION",
                "the plan region has no whole-task unit; empty-region semantics are not faked",
            )
        origin = int(getattr(region, "origin", 0) or 0)
        shifts = [int(task.resolved_stage) - origin for task in tasks]
        if (max(shifts) if shifts else 0) != int(getattr(region, "max_shift", 0) or 0):
            raise SymbolicApplyError(
                "APPLY_PLAN_SHIFT_MISMATCH",
                "the plan's stage shift and its max_shift disagree",
            )
        if lower != 0 or step != 1 or any(shifts):
            raise SymbolicApplyError(
                "APPLY_SYMBOLIC_FORM_UNIMPLEMENTED",
                "stage-shifted / non-zero-lower / step>1 loops need the S1/S2 symbolic form",
            )
        p = Var(name="p", type=stmt.induction_var.type,
                is_const=stmt.induction_var.is_const)
        body_env = dict(env)
        body_env[id(stmt.induction_var)] = p
        region_id = self.region_id_by_loop.get(id(stmt), "")
        body, _ = self.statement(stmt.body, body_env)
        exit_body = self._exit_events(region_id)
        if exit_body:
            body = Sequential(tuple(body.body if isinstance(body, Sequential) else (body,))
                              + tuple(exit_body))
        candidate = For(
            induction_var=p,
            start=Constant(type=stmt.start.type, value=0),
            stop=Constant(type=stmt.stop.type, value=iterations),
            step=Constant(type=stmt.step.type, value=1),
            body=body,
            pipeline=None,
        )
        self.candidate_for_by_source[id(stmt)] = candidate
        self._record(
            stmt, (candidate,), APPLY_ROLE_CONTROL,
            binder_pairs=((stmt.induction_var, p),),
            substitutions=(
                ("lower", str(lower)), ("stop", str(stop)), ("step", str(step)),
                ("N", str(iterations)), ("origin", str(origin)),
                ("k", "p"), ("i", "p"),
            ),
            unit_id=str(getattr(region.region_ref, "region_id", "")),
        )
        return candidate




# --------------------------------------------------------------------- events
#: The instruction names the frozen interface keys on, taken from each op's
#: schema (dialect + name), never from a class name: the schema is what the
#: emitted TL surface and the reader both agree on.
EVENT_OP_NAMES = ("flag_set", "flag_wait", "drain")

#: The attributes a flag instruction declares, in schema order. A key that
#: listed them in a different order would compare equal to a different
#: instruction, so the order is stated once, here.
_FLAG_ATTRIBUTES = ("channel_name", "kind", "slot", "generation", "versions",
                    "flag_id", "producer_pipe", "consumer_pipe")

#: Which anchor means "this site belongs before/after the task" and which means
#: "the whole region has ended".
SITE_BEFORE = "before_task"
SITE_AFTER = "after_task"
SITE_EXIT = "region_exit"


def event_op_name(node) -> str:
    """The schema name of an event instruction, or "" for a non-event op."""
    callable_ = getattr(node, "callable", node)
    schema = getattr(type(callable_), "_op_schema", None)
    if schema is None or getattr(schema, "category", "") != "sync":
        return ""
    return str(schema.name)


def event_statement_key(stmt) -> tuple:
    """The identity of ONE event statement, as declared attributes.

    This is the frozen binding (task-54): a statement is identified by what its
    instruction DECLARES, in schema order, with no string formatting anywhere.
    The same function is used by apply, by verify (reading the candidate TIR)
    and by the reader that parses the emitted TL, so the three cannot drift.

    A flag instruction without a static attribute value is refused rather than
    keyed with a placeholder: a non-static slot or generation is not a fact
    this layer can state.
    """
    name = event_op_name(stmt)
    if not name:
        return ()
    if name == "drain":
        return ("drain",)
    attributes = []
    for attribute in _FLAG_ATTRIBUTES:
        value = getattr(stmt.callable, attribute, None)
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise SymbolicApplyError(
                "APPLY_EVENT_ATTRIBUTE_UNSOURCED",
                "event instruction %r states %s=%r, which is not a static value; "
                "a slot or generation is never guessed" % (name, attribute, value),
            )
        attributes.append(value)
    return (name, *attributes)


def _mechanism_of(channel):
    return getattr(channel, "physical_mechanism", None)


def event_key_of_site(site, channel) -> tuple:
    """The key a completion site must produce in the candidate program.

    RE-DERIVED from the site and its channel, never read from a stored string:
    a site that names a slot the channel never allocated cannot key itself
    into existence.
    """
    anchor = str(getattr(site, "anchor", ""))
    mechanism = _mechanism_of(channel)
    if mechanism is None:
        raise SymbolicApplyError(
            "APPLY_EVENT_CHANNEL_MISSING",
            "site %s names no channel, so its instruction cannot be stated"
            % (getattr(site, "site_id", "?"),),
        )
    generation = getattr(site, "generation", None)
    static = static_dim_value(generation) if generation is not None else None
    if static is None:
        raise SymbolicApplyError(
            "APPLY_EVENT_GENERATION_UNSOURCED",
            "site %s states a generation that is not a static integer, so no "
            "flag generation can be emitted for it" % (getattr(site, "site_id", "?"),),
        )
    if _mechanism_is_drain(mechanism):
        return ("drain",)
    base = getattr(mechanism, "base_id", None)
    if base is None:
        raise SymbolicApplyError(
            "APPLY_EVENT_SLOT_UNALLOCATED",
            "channel %s declares no base id, so its flag slot was never allocated"
            % (getattr(channel, "channel_id", "?"),),
        )
    namespaces = tuple(getattr(mechanism, "namespaces", ()) or ())
    if len(namespaces) != 1:
        raise SymbolicApplyError(
            "APPLY_EVENT_NAMESPACE_UNSUPPORTED",
            "channel %s declares %d event namespace(s); one instruction can only "
            "occupy one pipe pair" % (getattr(channel, "channel_id", "?"),
                                      len(namespaces)),
        )
    producer_pipe, consumer_pipe = namespaces[0].pipe_pair
    return (("flag_wait" if anchor == SITE_BEFORE else "flag_set"),
            str(getattr(channel, "channel_id", "")),
            str(getattr(channel, "reason", "")).lower(),
            int(base), int(static), int(getattr(channel, "versions", 1)),
            int(base), str(producer_pipe), str(consumer_pipe))


def _mechanism_is_drain(mechanism) -> bool:
    return str(getattr(mechanism, "kind", "")) == "DRAIN"


def event_instruction(key) -> object:
    """Build the real TileFoundry op for one key; never a string stand-in."""
    from tilefoundry.ir.tir.flag import Drain, FlagSet, FlagWait

    if not key:
        raise SymbolicApplyError("APPLY_EVENT_KEY_EMPTY",
                                 "an empty key names no event instruction")
    if key[0] == "drain":
        return Drain()
    _name, channel_name, kind, slot, generation, versions, flag_id, producer, consumer = key
    attributes = dict(channel_name=channel_name, kind=kind, slot=int(slot),
                      generation=int(generation), versions=int(versions),
                      flag_id=int(flag_id), producer_pipe=producer,
                      consumer_pipe=consumer)
    if key[0] == "flag_set":
        return FlagSet(**attributes)
    if key[0] == "flag_wait":
        return FlagWait(**attributes)
    raise SymbolicApplyError("APPLY_EVENT_OP_UNKNOWN",
                             "key names an instruction this layer cannot build: %r"
                             % (key[0],))



def _event_statements(function) -> tuple:
    """Every event statement in a program, in program order."""
    found = []

    def visit(node):
        if isinstance(node, Evaluate) and event_op_name(node):
            found.append(node)
            return
        if isinstance(node, (Sequential,)):
            for item in node.body:
                visit(item)
            return
        if isinstance(node, (MeshScope, ExecutionScope)):
            visit(node.body)
            return
        if isinstance(node, For):
            visit(node.body)
            return
        if isinstance(node, If):
            visit(node.then_body)
            if node.else_body is not None:
                visit(node.else_body)

    visit(getattr(function, "body", None))
    return tuple(found)


def _require_all_sites_placed(completion, placed, candidate_entry) -> None:
    """apply fails if a decided site is missing, or an extra event appeared."""
    expected = sorted(getattr(site, "site_id", "")
                      for site in getattr(completion, "sites", ()) or ())
    if not expected:
        return
    actual_sites = sorted(site_id for site_id, _key in placed)
    if actual_sites != expected:
        missing = sorted(set(expected) - set(actual_sites))
        extra = sorted(set(actual_sites) - set(expected))
        raise SymbolicApplyError(
            "APPLY_EVENT_SITE_MISMATCH",
            "the candidate carries %d event statement(s) for %d site(s); "
            "missing=%s extra=%s" % (len(actual_sites), len(expected), missing, extra),
        )
    statements = _event_statements(candidate_entry)
    if len(statements) != len(expected):
        raise SymbolicApplyError(
            "APPLY_EVENT_SITE_MISMATCH",
            "the candidate TIR holds %d event statement(s) for %d site(s)"
            % (len(statements), len(expected)),
        )





# ------------------------------------------------------- versioned slot/view
#: The generation the S0 candidate addresses. A straight-line region runs once,
#: so every buffer is touched in generation 0; a loop's symbolic generation needs
#: the control envelope (task-46), and pretending otherwise here would place
#: bytes at an offset nothing proved.
STORAGE_SLOT_GENERATION = 0


def storage_offset_of(entry) -> int:
    """The byte offset the storage plan places one buffer at, for this form."""
    slots = int(getattr(entry, "slot_count", 1) or 1)
    if slots != 1:
        raise SymbolicApplyError(
            "APPLY_STORAGE_MULTI_SLOT_UNIMPLEMENTED",
            "buffer %r is placed in %d slots; a multi-slot ring needs the loop's "
            "control envelope, so its offset is not derived here"
            % (str(getattr(getattr(entry, "root", None), "name", "?")), slots))
    slot = STORAGE_SLOT_GENERATION % slots
    return int(getattr(entry, "base_offset_bytes", 0)) + \
        slot * int(getattr(entry, "version_stride_bytes", 0))


def _inject_storage_windows(function, storage, records):
    """Rewrite every view's address from the placements; returns a new entry.

    This is a real transformation, not an assertion: the offset in the candidate
    comes from the storage plan's placement, so a plan that places a buffer
    somewhere else produces a different candidate. For the S0 programs the plan
    and the lowered window coincide (one slot, generation 0), which is why the
    numbers do not change - the substitution is still recorded and the verifier
    still recomputes it, so a placement that disagrees is refused rather than
    silently kept.
    """
    from tilefoundry.ir.core import Call, Constant
    from tilefoundry.ir.tir.stmts import (ExecutionScope, For, LetStmt, MeshScope,
                                          Sequential)

    if storage is None:
        return function, records
    placement = {}
    for entry in getattr(storage, "entries", ()) or ():
        name = str(getattr(getattr(entry, "root", None), "name", ""))
        if name:
            placement[name] = entry
    if not placement:
        return function, records

    def rewrite(node):
        if isinstance(node, Sequential):
            return Sequential(tuple(rewrite(item) for item in node.body))
        if isinstance(node, (MeshScope, ExecutionScope)):
            return replace(node, body=rewrite(node.body))
        if isinstance(node, For):
            return replace(node, body=rewrite(node.body))
        if isinstance(node, LetStmt):
            value = node.value
            target = getattr(value, "target", None)
            if type(target).__name__ != "TensorView" or not isinstance(value, Call):
                return node
            entry = placement.get(str(getattr(node.var, "name", "")))
            if entry is None or not value.args:
                return node
            wanted = storage_offset_of(entry)
            address = value.args[0]
            if not isinstance(address, Constant):
                raise SymbolicApplyError(
                    "APPLY_STORAGE_WINDOW_UNSOURCED",
                    "view %r has no static address Constant, so its placement "
                    "cannot be applied" % (str(getattr(node.var, "name", "")),))
            if int(getattr(address, "value", 0)) == wanted:
                return node
            new_value = replace(value, args=(Constant(type=address.type, value=wanted),)
                                + tuple(value.args[1:]))
            records.append((str(getattr(node.var, "name", "")),
                            int(getattr(address, "value", 0)), wanted))
            return replace(node, value=new_value)
        return node

    return replace(function, body=rewrite(function.body)), records

def _append_inside_kernel_frame(body, extra):
    """Append statements to the END of the kernel frame's body, not beside it."""
    from tilefoundry.ir.tir.stmts import MeshScope, Sequential

    if not extra:
        return body

    def append(node):
        if isinstance(node, MeshScope):
            inner = node.body
            items = inner.body if isinstance(inner, Sequential) else (inner,)
            return MeshScope(mesh=node.mesh, binding=node.binding,
                             body=Sequential(tuple(items) + tuple(extra)))
        if isinstance(node, Sequential):
            if not node.body:
                raise SymbolicApplyError(
                    "APPLY_KERNEL_FRAME_MISSING",
                    "no kernel frame exists to carry the region's exit events")
            head = node.body[:-1]
            tail = append(node.body[-1])
            return Sequential(tuple(head) + (tail,))
        raise SymbolicApplyError(
            "APPLY_KERNEL_FRAME_MISSING",
            "the entry body carries no kernel frame, so an exit event would run "
            "in no execution domain")
    return append(body)

def _candidate_module(entry, candidate_entry, original):
    if original is None:
        functions = (candidate_entry,)
        module = Module(
            name=entry.name + "#candidate",
            functions=functions,
            entry=entry.name,
            target=entry.target,
        )
    else:
        functions = tuple(
            candidate_entry if function is entry else function
            for function in getattr(original, "functions", ()) or ()
        )
        module = Module(
            name=getattr(original, "name", entry.name) + "#candidate",
            functions=functions,
            entry=getattr(original, "entry", entry.name),
            children=dict(getattr(original, "children", {}) or {}),
            target=getattr(original, "target", entry.target),
            topologies=getattr(original, "topologies", None),
            metadata=dict(getattr(original, "metadata", {}) or {}),
            methods=dict(getattr(original, "methods", {}) or {}),
        )
    module.metadata["ascend.candidate"] = {
        "source_identity": function_identity(entry),
        "candidate_identity": function_identity(candidate_entry),
    }
    return module


def apply_symbolic_pipeline(entry, plan, completion, storage, *, original=None):
    """Materialize the plan into a PRIVATE candidate module; never installs or exports."""
    if not isinstance(entry, PrimFunction):
        raise SymbolicApplyError(
            "APPLY_ENTRY_NOT_PRIM_FUNCTION",
            "apply needs the real entry PrimFunction, got %s" % (type(entry).__name__,),
        )
    if plan is None:
        raise SymbolicApplyError("APPLY_PLAN_MISSING", "apply needs the resolved plan")
    cloner = _Cloner(entry, plan, completion)
    body, _env = cloner.statement(entry.body, {})
    # A top-level region has no For body to append to, so its exit events go at
    # the end of the entry body - still inside the kernel frame, which
    # require_kernel_frame() checks afterwards.
    root_id = cloner._root_region_id
    exit_body = cloner._exit_events(root_id) if root_id else []
    if exit_body:
        # A region's exit events belong INSIDE its kernel frame: an event
        # statement has an execution domain (the target's, per its facts), so a
        # top-level append would place it outside every MeshScope and the TIR
        # validator refuses it - correctly, because such a statement would
        # execute in no declared domain.
        body = _append_inside_kernel_frame(body, exit_body)
    candidate_entry = PrimFunction(
        name=entry.name,
        params=entry.params,
        body=body,
        output_count=entry.output_count,
        target=entry.target,
    )
    for region in getattr(plan, "regions", ()) or ():
        node = getattr(region.region_ref, "node_ref", None)
        if isinstance(node, For):
            candidate = cloner.candidate_for_by_source.get(id(node))
            if candidate is None:
                raise SymbolicApplyError(
                    "APPLY_REGION_UNRESOLVED",
                    "a plan region's loop was not materialized",
                )
            actual = _count_schedule_units(candidate.body)
            expected = len(getattr(region, "tasks", ()) or ())
            if actual != expected:
                raise SymbolicApplyError(
                    "APPLY_MULTIPLICITY_MISMATCH",
                    "the materialized loop has %d unit(s) but the plan schedules %d"
                    % (actual, expected),
                )
    root_region = (getattr(plan, "regions", ()) or (None,))[0]
    if root_region is not None:
        actual = _count_schedule_units(candidate_entry.body)
        expected = len(getattr(root_region, "tasks", ()) or ())
        if actual != expected:
            raise SymbolicApplyError(
                "APPLY_MULTIPLICITY_MISMATCH",
                "the materialized entry has %d top-level unit(s) but the plan schedules %d"
                % (actual, expected),
            )
    require_kernel_frame(candidate_entry)
    # Every site the completion decided must exist as a statement, and no
    # statement may exist that no site asked for. This is apply's own
    # obligation; verify checks it again from the TIR, never from this map.
    _require_all_sites_placed(completion, cloner.placed_sites, candidate_entry)
    # Versioned windows come from the storage plan, in the same pass: injecting
    # events while leaving the addresses as lowered would make the candidate and
    # the plan disagree about where the bytes live, which is exactly what the
    # verifier is supposed to be able to check.
    storage_records = []
    candidate_entry, storage_records = _inject_storage_windows(
        candidate_entry, storage, storage_records)
    for name, before, after in storage_records:
        cloner._record("storage-window:%s" % (name,), (candidate_entry,),
                       APPLY_ROLE_STORAGE, unit_id=name,
                       substitutions=(("offset", str(before)), ("placed", str(after))))
    module = _candidate_module(entry, candidate_entry, original)
    source_map = SemanticSourceMap(
        source_identity=function_identity(entry),
        plan_identity=str(getattr(plan, "plan_identity", "")),
        candidate_identity=function_identity(candidate_entry),
        entries=tuple(cloner.entries),
    )
    return module, source_map
