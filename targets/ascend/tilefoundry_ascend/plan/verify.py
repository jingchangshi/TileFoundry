"""verify_actual_program: correspondence proof over the real source and candidate.

The verifier reads the FROZEN source module and the PRIVATE candidate module. It
does not trust apply's source map as evidence: it walks both statement trees,
rebuilds the binder environment, and compares every operand, every Op instance
(attributes included), every type, the loop bounds the plan requires, and the
whole-task multiplicity against the plan's task lists. A mismatch becomes a
named verdict code; it is never a silent pass.

The returned NativeVerdict binds the REAL objects (original module, candidate
module, candidate entry) and one private receipt, so install_entry can require
object identity plus the registered receipt, not a hash or a status string.
Only this function can create a receipt; a hand-made verdict is not installable.

STATED GAPS (S0 slice; recorded in the verdict, never asserted as done):
- completion is not implemented, so no wait/set/tail/drain node exists to check;
- the S0 plan's storage cannot cover the register-space buffers, so the
  candidate keeps the real lowered window addresses. When a storage plan is
  supplied the verifier compares its placed addresses against the candidate and
  refuses on disagreement instead of pretending the slots were injected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

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
from tilefoundry.visitor_registry.verify import verify_prim_function

from .apply import (
    APPLY_ROLE_ALLOCATION,
    APPLY_ROLE_COMPUTE,
    APPLY_ROLE_CONTROL,
    SymbolicApplyError,
    require_kernel_frame,
)
from .identity import function_identity

__all__ = ["NativeVerdict", "VERDICT_REFUSED", "VERDICT_VERIFIED",
           "verify_actual_program"]

VERDICT_VERIFIED = "VERIFIED"
VERDICT_REFUSED = "REFUSED"

_CHECK_OK = "OK"
_CHECK_GAP = "GAP"
_CHECK_FAIL = "FAIL"


class _Mismatch(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__("%s: %s" % (code, detail))


@dataclass(frozen=True)
class VerificationCheck:
    """One check the verifier actually ran, with the inputs it consumed."""

    name: str
    status: str
    detail: str
    evidence: tuple = ()


@dataclass(frozen=True)
class _Receipt:
    """The process-local proof that THIS verifier run succeeded on THIS object.

    A receipt is not constructible into an installable capability: it only exists
    because verify_actual_program put it in the module-private registry below,
    together with the real candidate module object it proved.
    """

    candidate_module_ref: object
    candidate_entry_ref: object
    candidate_identity: str
    plan_identity: str
    source_identity: str
    entry_identity: str


#: (id(candidate_module), candidate_identity) -> _Receipt. Only verify_actual_program
#: writes here; install_entry reads it and requires object identity of the receipt.
_VERIFIED_RECEIPTS: dict = {}


def _registered_receipt(candidate_module, candidate_identity: str):
    return _VERIFIED_RECEIPTS.get((id(candidate_module), str(candidate_identity)))


@dataclass(frozen=True)
class NativeVerdict:
    """The verifier's complete record; refs are compare=False and never hashed.

    A hand-built instance of this dataclass has no installable capability: the only
    usable evidence is the private receipt, which only verify_actual_program can put
    into the module-private registry for a real candidate object.
    """

    status: str
    code: str = ""
    detail: str = ""
    source_identity: str = ""
    plan_identity: str = ""
    original_identity: str = ""
    candidate_identity: str = ""
    original_entry_identity: str = ""
    candidate_entry_identity: str = ""
    entry_identity: str = ""
    checks: tuple = ()
    gaps: tuple = ()
    source_map: object | None = None
    receipt: object | None = field(default=None, compare=False, repr=False)
    original_module_ref: Module | None = field(default=None, compare=False, repr=False)
    candidate_module_ref: Module | None = field(default=None, compare=False, repr=False)
    candidate_entry_ref: PrimFunction | None = field(default=None, compare=False, repr=False)
    original_entry_ref: PrimFunction | None = field(default=None, compare=False, repr=False)

    @property
    def verified(self) -> bool:
        return self.status == VERDICT_VERIFIED

    def as_dict(self):
        return {
            "status": self.status,
            "code": self.code,
            "detail": self.detail,
            "source_identity": self.source_identity,
            "plan_identity": self.plan_identity,
            "original_identity": self.original_identity,
            "candidate_identity": self.candidate_identity,
            "entry_identity": self.entry_identity,
            "checks": [
                {"name": item.name, "status": item.status, "detail": item.detail,
                 "evidence": [list(pair) for pair in item.evidence]}
                for item in self.checks
            ],
            "gaps": list(self.gaps),
        }


def _identity(function) -> str:
    return function_identity(function)


class _Compare:
    def __init__(self, original_entry, candidate_entry, plan):
        self.source = original_entry
        self.candidate = candidate_entry
        self.plan = plan
        self.env = {}
        self.statement_of = {}
        self.candidate_binders = set()
        self.evaluates = 0
        self.execution_scopes = 0
        self.ifs = 0
        self.region_by_loop = {}
        for region in getattr(plan, "regions", ()) or ():
            node = getattr(region.region_ref, "node_ref", None)
            if isinstance(node, For):
                self.region_by_loop[id(node)] = region
        self.regions_seen = set()
        # Every sync instruction the candidate added, collected while walking so
        # the correspondence check compares the ACTUAL TIR against the plan.
        self.sync_statements = []

    # -- expressions --------------------------------------------------------
    def expression(self, source, candidate, where: str):
        if isinstance(source, Var):
            if id(source) in self.env:
                expected = self.env[id(source)]
                if candidate is not expected:
                    raise _Mismatch(
                        "VERIFY_BINDER_MISMATCH",
                        "%s references %r but the candidate uses a different binder"
                        % (where, source.name),
                    )
                return
            if candidate is not source:
                raise _Mismatch(
                    "VERIFY_EXTERNAL_VAR_CLONED",
                    "%s must keep the external Var %r by identity" % (where, source.name),
                )
            return
        if isinstance(source, Constant):
            if not isinstance(candidate, Constant):
                raise _Mismatch("VERIFY_EXPR_KIND", "%s changed a constant into %s"
                                % (where, type(candidate).__name__))
            if candidate.value != source.value or candidate.type != source.type:
                raise _Mismatch("VERIFY_CONSTANT_MISMATCH", "%s changed a constant" % (where,))
            return
        if isinstance(source, Call):
            if not isinstance(candidate, Call):
                raise _Mismatch("VERIFY_EXPR_KIND", "%s changed a Call into %s"
                                % (where, type(candidate).__name__))
            if candidate.target is not source.target:
                raise _Mismatch(
                    "VERIFY_OPERATION_CHANGED",
                    "%s was rewritten to a different operation object" % (where,),
                )
            if candidate.type != source.type:
                raise _Mismatch("VERIFY_TYPE_MISMATCH", "%s changed the value type" % (where,))
            if len(candidate.args) != len(source.args):
                raise _Mismatch("VERIFY_ARITY_MISMATCH", "%s changed operand count" % (where,))
            for index, (left, right) in enumerate(zip(source.args, candidate.args, strict=True)):
                self.expression(left, right, "%s.arg%d" % (where, index))
            return
        if isinstance(source, tuple):
            if not isinstance(candidate, tuple) or len(candidate) != len(source):
                raise _Mismatch("VERIFY_EXPR_KIND", "%s changed a tuple" % (where,))
            for index, (left, right) in enumerate(zip(source, candidate, strict=True)):
                self.expression(left, right, "%s[%d]" % (where, index))
            return
        raise _Mismatch("VERIFY_EXPR_UNSUPPORTED",
                        "%s: %s is outside the verifier domain" % (where, type(source).__name__))

    # -- binders ------------------------------------------------------------
    def _fresh(self, source_var, candidate_var, where: str):
        if not isinstance(candidate_var, Var):
            raise _Mismatch("VERIFY_BINDER_MISSING", "%s did not bind a Var" % (where,))
        if candidate_var is source_var:
            raise _Mismatch("VERIFY_BINDER_NOT_FRESH",
                            "%s reused the source binder %r" % (where, source_var.name))
        if candidate_var.type != source_var.type:
            raise _Mismatch("VERIFY_BINDER_TYPE", "%s changed the binder type" % (where,))
        if id(candidate_var) in self.candidate_binders:
            raise _Mismatch("VERIFY_BINDER_DUPLICATE",
                            "%s reuses one candidate binder for two definitions" % (where,))
        self.candidate_binders.add(id(candidate_var))

    # -- statements ---------------------------------------------------------
    def statement(self, source, candidate, where: str):
        if id(source) in self.statement_of:
            raise _Mismatch("VERIFY_DUPLICATE_SOURCE",
                            "source statement %s appears twice" % (where,))
        self.statement_of[id(source)] = candidate
        if isinstance(source, Sequential):
            if not isinstance(candidate, Sequential):
                raise _Mismatch("VERIFY_STRUCTURE_MISMATCH",
                                "%s is not a Sequential in the candidate" % (where,))
            # Sync instructions are additions the completion decided on, not
            # clones of a source statement: they are skipped here and checked
            # as a set against the plan below (VERIFY_EVENT_CORRESPONDENCE),
            # which is the only place their identity is judged.
            candidate_body = tuple(item for item in candidate.body
                                   if not _is_event_statement(item))
            self.sync_statements.extend(item for item in candidate.body
                                        if _is_event_statement(item))
            if len(candidate_body) != len(source.body):
                raise _Mismatch("VERIFY_STRUCTURE_MISMATCH",
                                "%s has %d statements, source has %d"
                                % (where, len(candidate_body), len(source.body)))
            for index, (left, right) in enumerate(zip(source.body, candidate_body,
                                                      strict=True)):
                self.statement(left, right, "%s/%d" % (where, index))
            return
        if isinstance(source, LetStmt):
            if not isinstance(candidate, LetStmt):
                raise _Mismatch("VERIFY_STRUCTURE_MISMATCH",
                                "%s is not a LetStmt in the candidate" % (where,))
            self.expression(source.value, candidate.value, "%s.value" % (where,))
            self._fresh(source.var, candidate.var, "%s.binder" % (where,))
            self.env[id(source.var)] = candidate.var
            return
        if isinstance(source, For):
            self._for(source, candidate, where)
            return
        if isinstance(source, MeshScope):
            if not isinstance(candidate, MeshScope):
                raise _Mismatch("VERIFY_STRUCTURE_MISMATCH",
                                "%s is not a MeshScope in the candidate" % (where,))
            if candidate.mesh != source.mesh:
                raise _Mismatch("VERIFY_MESH_MISMATCH", "%s changed the mesh" % (where,))
            self._fresh(source.binding, candidate.binding, "%s.binding" % (where,))
            self.env[id(source.binding)] = candidate.binding
            self.statement(source.body, candidate.body, "%s.body" % (where,))
            return
        if isinstance(source, ExecutionScope):
            if not isinstance(candidate, ExecutionScope):
                raise _Mismatch("VERIFY_STRUCTURE_MISMATCH",
                                "%s is not an ExecutionScope in the candidate" % (where,))
            if candidate.kind != source.kind:
                raise _Mismatch("VERIFY_EXECUTION_MODE",
                                "%s changed execution mode %r" % (where, source.kind))
            self.execution_scopes += 1
            self.statement(source.body, candidate.body, "%s.body" % (where,))
            return
        if isinstance(source, Evaluate):
            if not isinstance(candidate, Evaluate):
                raise _Mismatch("VERIFY_STRUCTURE_MISMATCH",
                                "%s is not an Evaluate in the candidate" % (where,))
            if candidate.callable is not source.callable:
                raise _Mismatch("VERIFY_OPERATION_CHANGED",
                                "%s changed the instruction object" % (where,))
            if len(candidate.args) != len(source.args):
                raise _Mismatch("VERIFY_ARITY_MISMATCH",
                                "%s changed the instruction operand count" % (where,))
            for index, (left, right) in enumerate(zip(source.args, candidate.args, strict=True)):
                self.expression(left, right, "%s.arg%d" % (where, index))
            self.evaluates += 1
            return
        if isinstance(source, If):
            if not isinstance(candidate, If):
                raise _Mismatch("VERIFY_STRUCTURE_MISMATCH",
                                "%s is not an If in the candidate" % (where,))
            self.ifs += 1
            self.expression(source.cond, candidate.cond, "%s.cond" % (where,))
            self.statement(source.then_body, candidate.then_body, "%s.then" % (where,))
            self.statement(source.else_body, candidate.else_body, "%s.else" % (where,))
            return
        if isinstance(source, Return):
            if not isinstance(candidate, Return):
                raise _Mismatch("VERIFY_STRUCTURE_MISMATCH",
                                "%s is not a Return in the candidate" % (where,))
            return
        if isinstance(source, While):
            raise _Mismatch("VERIFY_CONTROL_UNSUPPORTED", "%s is a While" % (where,))
        raise _Mismatch("VERIFY_STATEMENT_UNSUPPORTED",
                        "%s: %s" % (where, type(source).__name__))

    def _strip_sync(self, body):
        """Split one body into (cloned statements, sync statements)."""
        cloned, sync = [], []
        for item in body:
            (sync if _is_event_statement(item) else cloned).append(item)
        self.sync_statements.extend(sync)
        return tuple(cloned)

    def _for(self, source: For, candidate: For, where: str):
        if not isinstance(candidate, For):
            raise _Mismatch("VERIFY_STRUCTURE_MISMATCH", "%s is not a For in the candidate" % (where,))
        region = self.region_by_loop.get(id(source))
        if region is None:
            raise _Mismatch("VERIFY_REGION_UNRESOLVED",
                            "%s has no plan region, so its bounds cannot be checked" % (where,))
        self.regions_seen.add(str(getattr(region.region_ref, "region_id", "")))
        lower = static_dim_value(source.start)
        stop = static_dim_value(source.stop)
        step = static_dim_value(source.step)
        if lower is None or stop is None or step is None or int(step) <= 0:
            raise _Mismatch("VERIFY_LOOP_DOMAIN",
                            "%s has no static loop domain" % (where,))
        iterations = max(0, -(-(int(stop) - int(lower)) // int(step)))
        expected = (0, iterations, 1)
        actual = (static_dim_value(candidate.start), static_dim_value(candidate.stop),
                  static_dim_value(candidate.step))
        if actual != expected:
            raise _Mismatch("VERIFY_LOOP_BOUNDS",
                            "%s bounds %r are not the symbolic domain %r"
                            % (where, actual, expected))
        origin = int(getattr(region, "origin", 0) or 0)
        tasks = getattr(region, "tasks", ()) or ()
        shifts = sorted(int(task.resolved_stage) - origin for task in tasks)
        if not tasks:
            raise _Mismatch("VERIFY_REGION_EMPTY", "%s plan region has no task" % (where,))
        if shifts and (shifts[0] != 0 or shifts[-1] != int(getattr(region, "max_shift", 0) or 0)):
            raise _Mismatch("VERIFY_STAGE_SHIFT_MISMATCH",
                            "%s plan stage shifts disagree with max_shift" % (where,))
        if any(shifts):
            raise _Mismatch("VERIFY_SYMBOLIC_FORM_UNSUPPORTED",
                            "%s needs the stage-shifted S1/S2 form" % (where,))
        self._fresh(source.induction_var, candidate.induction_var, "%s.iv" % (where,))
        self.env[id(source.induction_var)] = candidate.induction_var
        self.statement(source.body, candidate.body, "%s.body" % (where,))


def _count_statement_units(stmt) -> int:
    """Scheduling units in one body; sync instructions are not units."""
    if isinstance(stmt, Evaluate):
        return 0 if _is_event_statement(stmt) else 1
    if isinstance(stmt, ExecutionScope):
        return 1
    if isinstance(stmt, Sequential):
        return sum(_count_statement_units(item) for item in stmt.body)
    if isinstance(stmt, MeshScope):
        return _count_statement_units(stmt.body)
    return 0


def _is_event_statement(stmt) -> bool:
    from .apply import event_op_name

    from tilefoundry.ir.tir.stmts import Evaluate

    return isinstance(stmt, Evaluate) and bool(event_op_name(stmt))


def _expected_event_keys(plan, completion) -> list:
    """The event instructions the plan+completion REQUIRE, re-derived.

    Derived from the same functions apply used, never from what apply reported:
    a candidate that lost a wait cannot be certified by a source map that still
    lists it.
    """
    from .apply import _sites_by_unit, event_key_of_site

    channels = {channel.channel_id: channel
                for channel in getattr(completion, "channels", ()) or ()}
    by_consumer = {}
    for channel in channels.values():
        for consumer in getattr(channel, "consumers", ()) or ():
            by_consumer[consumer] = channel
    by_producer = {getattr(channel, "producer", None): channel
                   for channel in channels.values()}
    keys = []
    for site in getattr(completion, "sites", ()) or ():
        site_id = getattr(site, "site_id", "")
        channel = by_consumer.get(site_id) or by_producer.get(site_id)
        if channel is None:
            raise SymbolicApplyError(
                "APPLY_EVENT_CHANNEL_MISSING",
                "completion site %s belongs to no channel" % (site_id,))
        keys.append(event_key_of_site(site, channel))
    return keys


def _refuse(code: str, detail: str, **refs) -> NativeVerdict:
    return NativeVerdict(status=VERDICT_REFUSED, code=code, detail=detail, **refs)




def _window_of(view) -> tuple:
    """(offset, bytes) of one lowered view, from its real address Constant.

    The offset lives on the TensorView's ARGUMENT (the lowered form is
    Call(target=TensorView(...), args=(Constant(PointerType, offset),))), not on
    the target, so reading it from the target found nothing and the check
    refused instead of comparing.
    """
    from tilefoundry.ir.types.utils import static_dim_value

    args = tuple(getattr(view, "args", ()) or ())
    offset = static_dim_value(args[0]) if args else None
    if offset is None:
        raise SymbolicApplyError(
            "APPLY_STORAGE_WINDOW_UNSOURCED",
            "a lowered view states no static offset, so its placement cannot be "
            "compared with the plan")
    sizes = tuple(getattr(getattr(view, "target", None), "shape", ()) or ())
    elements = 1
    for size in sizes:
        elements *= int(size)
    return int(offset), elements * _dtype_bytes(view)


def _dtype_bytes(view) -> int:
    dtype = getattr(getattr(view, "type", None), "dtype", None)
    bits = getattr(dtype, "bit_width", None)
    if not bits:
        raise SymbolicApplyError(
            "APPLY_STORAGE_WINDOW_UNSOURCED",
            "a lowered view states no dtype width, so its byte window cannot be "
            "derived")
    return max(1, int(bits) // 8)


def _views_by_root(function) -> dict:
    """root Var name -> its lowered view, read from the actual TIR."""
    from tilefoundry.ir.tir.stmts import LetStmt, Sequential, Evaluate, MeshScope, ExecutionScope, For

    found = {}

    def visit(node):
        from tilefoundry.ir.tir.stmts import ExecutionScope as ES
        from tilefoundry.ir.tir.stmts import For as F
        from tilefoundry.ir.tir.stmts import MeshScope as MS
        from tilefoundry.ir.tir.stmts import Sequential as SQ

        if isinstance(node, SQ):
            for item in node.body:
                visit(item)
            return
        if isinstance(node, (MS, ES)):
            visit(node.body)
            return
        if isinstance(node, F):
            visit(node.body)
            return
        if isinstance(node, LetStmt):
            value = node.value
            target = getattr(value, "target", None)
            if type(target).__name__ == "TensorView":
                # the CALL, not the bound Var: the address Constant rides on
                # Call.args, so storing the Var loses the offset
                found[str(getattr(node.var, "name", ""))] = value
    visit(getattr(function, "body", None))
    return found


def _storage_disagreements(candidate_entry, original_entry, storage) -> list:
    """Where the plan's placement and the candidate's own windows disagree."""
    notes = []
    for entry in getattr(storage, "entries", ()) or ():
        root = getattr(entry, "root", None)
        name = str(getattr(root, "name", ""))
        view = _views_by_root(candidate_entry).get(name)
        if view is None:
            notes.append("buffer %r is placed by the plan but the candidate holds no "
                         "view for it" % (name,))
            continue
        offset, payload = _window_of(view)
        if int(getattr(entry, "slot_count", 1)) != 1:
            notes.append("buffer %r is placed in %d slots; the S0 candidate addresses "
                         "one window" % (name, int(entry.slot_count)))
        if int(getattr(entry, "base_offset_bytes", 0)) != offset:
            notes.append("buffer %r is placed at %d but the candidate addresses %d"
                         % (name, int(entry.base_offset_bytes), offset))
        if int(getattr(entry, "version_stride_bytes", 0)) < payload:
            notes.append("buffer %r has a %d-byte stride for a %d-byte payload"
                         % (name, int(entry.version_stride_bytes), payload))
    for region in getattr(original_entry, "params", ()) or ():
        pass
    return notes

def _walk_events(function) -> tuple:
    """Every event statement in a program, in program order."""
    from .apply import event_op_name

    found = []

    def visit(node):
        from tilefoundry.ir.tir.stmts import (Evaluate, ExecutionScope, For, If,
                                              MeshScope, Sequential)

        if isinstance(node, Evaluate) and event_op_name(node):
            found.append(node)
            return
        if isinstance(node, Sequential):
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
            return

    visit(getattr(function, "body", None))
    return tuple(found)


def _difference(left, right) -> list:
    """Multiset difference, so one duplicated key shows up as one extra."""
    remaining = list(right)
    gone = []
    for item in left:
        if item in remaining:
            remaining.remove(item)
        else:
            gone.append(item)
    return gone


def verify_actual_program(original, rewritten, plan, completion, storage, *, source_map=None):
    """Compare the frozen source with the private candidate; emit a bound verdict."""
    refs = {
        "original_module_ref": original if isinstance(original, Module) else None,
        "candidate_module_ref": rewritten if isinstance(rewritten, Module) else None,
    }
    if not isinstance(original, Module) or not isinstance(rewritten, Module):
        return _refuse("VERIFY_MODULE_TYPE", "verify needs two real Modules", **refs)
    if original is rewritten:
        return _refuse("VERIFY_SAME_MODULE",
                       "the candidate module must be a different object from the source",
                       **refs)
    try:
        original_entry = original.entry_function()
        candidate_entry = rewritten.entry_function()
    except Exception as error:
        return _refuse("VERIFY_ENTRY_UNRESOLVED", str(error), **refs)
    refs["original_entry_ref"] = original_entry
    refs["candidate_entry_ref"] = candidate_entry
    if candidate_entry is original_entry:
        return _refuse("VERIFY_ENTRY_NOT_CLONED",
                       "the candidate entry is the source entry object", **refs)
    # Emittability is a property of the CANDIDATE, not of the plan: a body the
    # backend cannot lower is refused before anything else is consulted. apply
    # raises the emitter's own codes (codegen/emit.py:547-557, 673) and they
    # are reused verbatim here, so one failure never carries two names across
    # the two stages.
    try:
        require_kernel_frame(candidate_entry)
    except SymbolicApplyError as refusal:
        return _refuse(refusal.code, refusal.detail, **refs)
    if plan is None:
        return _refuse("VERIFY_PLAN_MISSING", "verify needs the resolved plan", **refs)
    plan_identity = str(getattr(plan, "plan_identity", "") or "")
    if not plan_identity:
        return _refuse("VERIFY_PLAN_IDENTITY_MISSING",
                       "the plan carries no identity", **refs)

    source_identity = _identity(original_entry)
    if str(getattr(plan, "source_identity", "")) != source_identity:
        return _refuse(
            "VERIFY_SOURCE_IDENTITY_MISMATCH",
            "the plan was not built from this frozen source",
            **refs,
        )

    checks = []
    try:
        compare = _Compare(original_entry, candidate_entry, plan)
        compare.statement(original_entry.body, candidate_entry.body, "entry")
    except _Mismatch as mismatch:
        return _refuse(mismatch.code, mismatch.detail, **refs)
    except Exception as error:  # a structural surprise is still a named refusal
        return _refuse("VERIFY_INTERNAL", str(error), **refs)

    for region in getattr(plan, "regions", ()) or ():
        node = getattr(region.region_ref, "node_ref", None)
        expected = len(getattr(region, "tasks", ()) or ())
        if isinstance(node, For):
            candidate = compare.statement_of.get(id(node))
            if candidate is None:
                return _refuse("VERIFY_REGION_UNRESOLVED",
                               "a plan loop was not found in the candidate", **refs)
            actual = _count_statement_units(candidate.body)
        else:
            actual = _count_statement_units(candidate_entry.body)
        if actual != expected:
            return _refuse(
                "VERIFY_MULTIPLICITY_MISMATCH",
                "plan schedules %d unit(s) but the candidate materialized %d"
                % (expected, actual), **refs)
    checks.append(("multiplicity", _CHECK_OK,
                   "%d plan regions, one materialization per task"
                   % (len(getattr(plan, "regions", ()) or ()),)))

    # Regular TIR legality of the actual candidate (type/mesh/SSA/ExecutionScope).
    try:
        verify_prim_function(candidate_entry, module_fns=rewritten.functions)
    except Exception as error:
        return _refuse("VERIFY_TIR_INVALID", str(error), **refs)
    checks.append(("tir_valid", _CHECK_OK, "verify_prim_function passed"))

    # The candidate keeps every non-entry function of the source by identity.
    for function in original.functions:
        if function is original_entry:
            continue
        if not any(item is function for item in rewritten.functions):
            return _refuse("VERIFY_FUNCTION_DROPPED",
                           "the candidate dropped a non-entry function", **refs)
    checks.append(("other_functions", _CHECK_OK, "non-entry functions preserved by identity"))

    checks.append(("operands", _CHECK_OK,
                   "%d instructions/operands compared in order" % (compare.evaluates,)))
    checks.append(("execution_scopes", _CHECK_OK,
                   "%d SimdVF region(s) compared" % (compare.execution_scopes,)))
    checks.append(("guards", _CHECK_OK,
                   "%d If statement(s) compared" % (compare.ifs,)))
    checks.append(("binders", _CHECK_OK,
                   "%d fresh binder(s) checked" % (len(compare.candidate_binders),)))
    plan_slots = []
    for region in getattr(plan, "regions", ()) or ():
        for buffer in getattr(region, "buffers", ()) or ():
            versions = int(getattr(buffer, "resolved_versions", 0) or 0)
            if versions < 1:
                return _refuse("VERIFY_SLOT_INVALID",
                               "a plan buffer has no positive resolved versions", **refs)
            plan_slots.append((str(buffer.buffer_id), versions))
    checks.append(("plan_slots", _CHECK_OK,
                   "%d plan buffer(s) with positive resolved versions" % (len(plan_slots),)))

    # The completion's obligations, checked as a SET against the actual TIR in
    # BOTH directions: a decided site with no statement is a missing wait (the
    # program would race), and a statement no site asked for is an invented
    # sync (the program would claim an ordering nothing proved). Endpoints,
    # slot, generation, versions and pipes are all part of the key, so editing
    # any of them is a mismatch too. One existing code covers all of it.
    if completion is None:
        checks.append(("completion", _CHECK_GAP,
                       "no completion plan supplied; the candidate carries no sync "
                       "obligations to check"))
    else:
        from .apply import event_statement_key

        try:
            expected_keys = _expected_event_keys(plan, completion)
        except SymbolicApplyError as refusal:
            return _refuse(refusal.code, refusal.detail, **refs)
        actual_keys = [event_statement_key(item)
                       for item in _walk_events(candidate_entry)]
        if sorted(actual_keys) != sorted(expected_keys):
            missing = _difference(expected_keys, actual_keys)
            extra = _difference(actual_keys, expected_keys)
            return _refuse(
                "NATIVE_COMPLETION_MISMATCH",
                "the candidate's sync instructions do not correspond to the "
                "planned ones: %d expected, %d present, missing=%s extra=%s"
                % (len(expected_keys), len(actual_keys), missing, extra), **refs)
        checks.append(("completion", _CHECK_OK,
                       "%d sync instruction(s) correspond to the planned sites, "
                       "both directions" % (len(expected_keys),)))

    if storage is None:
        checks.append(("storage", _CHECK_GAP,
                       "no storage plan supplied; the candidate's windows are not "
                       "checked against a placement"))
    else:
        # The placement is checked against the CANDIDATE's actual windows rather
        # than asserted. For the S0 programs the plan's answers coincide with the
        # lowered addresses (measured: copy_only's loaded buffer is placed at 0
        # with stride 256 in one slot, which is exactly its lowered TensorView),
        # so what this check establishes is that the plan and the program agree -
        # not that nothing was looked at. A placement that disagrees is refused.
        try:
            disagreements = _storage_disagreements(candidate_entry, original_entry,
                                                   storage)
        except SymbolicApplyError as refusal:
            return _refuse(refusal.code, refusal.detail, **refs)
        if disagreements:
            return _refuse("NATIVE_COMPLETION_MISMATCH",
                           "the candidate's storage windows do not match the plan's "
                           "placement: " + "; ".join(disagreements[:4]), **refs)
        checks.append(("storage", _CHECK_OK,
                       "%d placed buffer(s) agree with the candidate's windows"
                       % (len(getattr(storage, "entries", ()) or ()),)))
    checks.append(("slot_generation", _CHECK_OK,
                   "every plan buffer states a positive resolved version count and "
                   "the candidate's windows carry the plan's stride"))

    source_map_identity = getattr(source_map, "candidate_identity", None)
    if source_map is not None and source_map_identity != _identity(candidate_entry):
        return _refuse("VERIFY_SOURCE_MAP_STALE",
                       "the supplied source map was not produced for this candidate", **refs)

    candidate_identity = _identity(candidate_entry)
    evidence = (
        ("source_identity", source_identity),
        ("plan_identity", plan_identity),
        ("candidate_identity", candidate_identity),
        ("entry_identity", candidate_identity),
    )
    check_records = tuple(
        VerificationCheck(name, status, detail, evidence)
        for name, status, detail in checks if status == _CHECK_OK
    )
    if len(check_records) != len([item for item in checks if item[1] == _CHECK_OK]):
        return _refuse("VERIFY_CHECKS_INCONSISTENT", "a check record is malformed", **refs)
    gaps = tuple("%s: %s" % (name, detail)
                 for name, status, detail in checks if status == _CHECK_GAP)
    if gaps:
        return _refuse("VERIFY_INCOMPLETE", "native verification has unresolved gaps: " + "; ".join(gaps), **refs)
    receipt = _Receipt(
        candidate_module_ref=rewritten,
        candidate_entry_ref=candidate_entry,
        candidate_identity=candidate_identity,
        plan_identity=plan_identity,
        source_identity=source_identity,
        entry_identity=candidate_identity,
    )
    _VERIFIED_RECEIPTS[(id(rewritten), candidate_identity)] = receipt
    return NativeVerdict(
        status=VERDICT_VERIFIED,
        code="",
        detail="source and candidate correspond on structure, operands, operations, "
               "binders, plan bounds and whole-task multiplicity",
        source_identity=source_identity,
        plan_identity=plan_identity,
        original_identity=_identity(original_entry),
        candidate_identity=candidate_identity,
        original_entry_identity=_identity(original_entry),
        candidate_entry_identity=candidate_identity,
        entry_identity=candidate_identity,
        checks=check_records,
        gaps=gaps,
        source_map=source_map,
        receipt=receipt,
        **refs,
    )


def _walk(node):
    yield node
    if isinstance(node, (tuple, list)):
        for item in node:
            yield from _walk(item)
        return
    for name in getattr(node, "__dataclass_fields__", ()) or ():
        if name in ("type", "metadata", "origin"):
            continue
        yield from _walk(getattr(node, name))
