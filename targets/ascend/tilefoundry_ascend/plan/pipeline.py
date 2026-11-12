"""P3-03 / G32: symbolic software-pipeline materialization on REAL TIR.

The R2/R3 materializers in materialize.py UNROLL the loop body once per logical iteration.
That is a static fixture technique, not a pipeline: the IR grows with the trip count, so it
can neither express a dynamic trip nor state a schedule. This module performs the transform
specs/03 section 3.2..3.4 asks for and rewrites the ORIGINAL loop into ONE symbolic guarded
loop:

    p in [0, N + H)                     physical step; H = max over tasks of shift[u]
      task u in resolved physical order:
        k_u = p - shift[u]              shift[u] = stage[u] - min(stage)
        if 0 <= k_u < N and <source guard>:
          i = start + k_u * step        the ORIGINAL induction value for k_u
          <u's own statements, with i and its ring slot substituted>

The loop body therefore holds ONE instance of each task no matter what the trip count is,
the trip may stay symbolic, and every resolved (task, k) executes exactly once. Buffer
accesses use the ring mapping slot = k % V with generation = k (spec 3.4). AccessPlan
already carries the ordinal expression, so a stage-shifted producer and consumer of one
buffer resolve to their OWN k and therefore their own slot instead of a shared guess.

SSA / Let / captured values (spec 3.3). Value-form TIR binds a window with a LetStmt
inside the loop body and consumes it in a task. Such a binding depends on the induction
value, so it cannot stay shared: each task receives a FRESH copy of every body-level
binding it transitively uses, rewritten against that task's own i. Renaming per task is
what keeps the value chain intact rather than re-reading the buffer. The companion
verify_pipeline_application() re-derives all of this from the REWRITTEN TIR and refuses a
block that uses a value nothing inside it (or the enclosing function) binds -- the
dropped-carried-value failure -- and refuses a wrong shift or a wrong guard by
re-simulating the emitted conditions.

Nothing here fabricates a schedule. Shifts, orders, ordinals and version counts all come
from a ResolvedPlan built by ResolveSchedule, so perturbing the solver assignment changes
this transform's output.
"""
from __future__ import annotations

import dataclasses
import hashlib
import types

from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.core.expr import Tuple as TupleExpr
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.tir.arith import Binary
from tilefoundry.ir.tir.flag import Drain
from tilefoundry.ir.tir.memory import AllocTensor
from tilefoundry.ir.tir.stmts import (Evaluate, ExecutionScope, For, If, LetStmt,
                                      MeshScope, Sequential)
from tilefoundry.ir.types import (DType, TupleType, make_tensor_type)
from tilefoundry.ir.types.storage import StorageKind

from .resolved_plan import OFFSET_UNKNOWN, PlanError, ResolvedPlan

PIPELINE_SCHEMA = "software-pipeline/1"

# STABLE names. The verifier re-derives the schedule from the rewritten IR by these, so a
# silently renamed loop would otherwise be unverifiable rather than merely renamed.
P_STEP_VAR = "p__pipeline"
N_TRIP_VAR = "n_trip__pipeline"
K_VAR_PREFIX = "k__pipeline__"
I_VAR_PREFIX = "i__pipeline__"
RING_SUFFIX = "__ring"
RENAME_SUFFIX = "__pl"

MAX_PIPELINE_DEPTH = 64
MAX_SIMULATED_TRIP = 4096

_ELEMENT_BYTES = {"f32": 4, "f16": 2, "bf16": 2, "i32": 4, "i16": 2, "i8": 1,
                  "u8": 1, "i64": 8, "f64": 8, "bool": 1}


def _scalar(dtype=DType.i32):
    return make_tensor_type((), dtype=dtype, storage=StorageKind.UMAT)


def _const(value, dtype=DType.i32):
    return Constant(type=_scalar(dtype), value=int(value))


def _bin(kind, lhs, rhs, dtype=DType.i32):
    return Call(type=_scalar(dtype), target=Binary(kind=kind), args=(lhs, rhs))


def _dtype(value):
    """Resolve a plan DATA-TYPE NAME to a real DType.

    The plan stores dtypes as names ("f32") because it is a portable record. TIR needs the
    DType OBJECT: a string type leaves the operand pattern unable to bind a dtype, which
    surfaces far from the cause as "dtype binding missing".
    """
    if isinstance(value, DType):
        return value
    resolved = getattr(DType, str(value), None)
    if resolved is None:
        raise PlanError("PLAN_PIPELINE_UNSUPPORTED_DTYPE",
                        "unknown element type for a ring view", "buffer.dtype",
                        "a known DType name", str(value))
    return resolved

def _element_bytes(dtype):
    size = _ELEMENT_BYTES.get(str(dtype))
    if size is None:
        raise PlanError("PLAN_PIPELINE_UNSUPPORTED_DTYPE",
                        "cannot size one element of this dtype", "buffer.dtype",
                        sorted(_ELEMENT_BYTES), str(dtype))
    return size


def _product(shape):
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


@dataclasses.dataclass(frozen=True)
class TaskBlock:
    """One resolved task's guarded block, as EMITTED into the symbolic loop."""

    task_id: str
    shift: int
    k_var: str
    i_var: str
    guard: str
    source_stmts: tuple = ()
    captured: tuple = ()


@dataclasses.dataclass(frozen=True)
class PipelineMap:
    """What the transform did, in a form the verifier re-checks against the IR itself."""

    schema: str
    p_var: str
    trip_var: str
    trip: object
    depth: int
    shifts: tuple
    versions: tuple
    blocks: tuple
    hoisted_allocs: tuple
    rewritten_index: tuple
    decisions: tuple = ()
    original_digest: str = ""
    rewritten_digest: str = ""

def _subst(node, mapping):
    """Replace Var leaves by name; everything else is rebuilt structurally."""
    if isinstance(node, Var):
        return mapping.get(node.name, node)
    if isinstance(node, Call):
        return dataclasses.replace(
            node, args=tuple(_subst(arg, mapping) for arg in node.args))
    if isinstance(node, TupleExpr):
        return dataclasses.replace(
            node, elements=tuple(_subst(item, mapping) for item in node.elements))
    return node


def _vars_used(node, out=None):
    """Every Var NAME reachable from node, statement fields included."""
    out = set() if out is None else out
    if isinstance(node, Var):
        out.add(node.name)
        return out
    if isinstance(node, Call):
        for arg in node.args:
            _vars_used(arg, out)
        return out
    if isinstance(node, TupleExpr):
        for item in node.elements:
            _vars_used(item, out)
        return out
    if isinstance(node, (tuple, list)):
        for item in node:
            _vars_used(item, out)
        return out
    if dataclasses.is_dataclass(node):
        for field in dataclasses.fields(node):
            if field.name in ("type", "metadata"):
                continue
            _vars_used(getattr(node, field.name, None), out)
    return out


def _bound_by(node, out=None):
    """Names a statement subtree BINDS (LetStmt vars, loop induction vars)."""
    out = set() if out is None else out
    if isinstance(node, LetStmt):
        out.add(node.var.name)
        _bound_by(node.value, out)
    elif isinstance(node, Sequential):
        for child in node.body:
            _bound_by(child, out)
    elif isinstance(node, If):
        _bound_by(node.cond, out)
        _bound_by(node.then_body, out)
        _bound_by(node.else_body, out)
    elif isinstance(node, (MeshScope, ExecutionScope)):
        name = getattr(getattr(node, "binding", None), "name", None)
        if name:
            out.add(name)
        _bound_by(node.body, out)
    elif isinstance(node, For):
        out.add(node.induction_var.name)
        for part in (node.start, node.stop, node.step, node.body):
            _bound_by(part, out)
    return out


def _digest_module(module):
    try:
        payload = repr(module)
    except Exception:
        payload = str(id(module))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()


def _as_int(expr):
    value = getattr(expr, "value", None)
    if isinstance(expr, Constant) and isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return None


_SYMBOLS = {BinaryKind.ADD: "+", BinaryKind.SUB: "-", BinaryKind.MUL: "*",
            BinaryKind.MOD: "%", BinaryKind.FLOOR_DIV: "//",
            BinaryKind.GE: ">=", BinaryKind.LT: "<", BinaryKind.LE: "<=",
            BinaryKind.GT: ">", BinaryKind.EQ: "==", BinaryKind.NE: "!=",
            BinaryKind.AND: " and ", BinaryKind.OR: " or "}


def _render(expr):
    """A stable printable form of a guard. Diagnostics only, never an evaluation."""
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Constant):
        return str(expr.value)
    if isinstance(expr, Call):
        symbol = _SYMBOLS.get(getattr(expr.target, "kind", None))
        if symbol is not None and len(expr.args) == 2:
            return "(%s%s%s)" % (_render(expr.args[0]), symbol, _render(expr.args[1]))
        return "%s(%s)" % (type(expr.target).__name__,
                           ", ".join(_render(arg) for arg in expr.args))
    return type(expr).__name__


def _iter_children(stmt):
    """(label, child) for every nested statement container, in source order."""
    if isinstance(stmt, Sequential):
        for index, child in enumerate(stmt.body):
            yield "s%d" % index, child
    elif isinstance(stmt, (MeshScope, ExecutionScope)):
        yield "scope", stmt.body
    elif isinstance(stmt, If):
        yield "then", stmt.then_body
        yield "else", stmt.else_body


def _first_pipeline_loop(module):
    """The For this transform emits, located by its STABLE induction-variable name."""
    found = []

    def walk(node):
        if isinstance(node, For):
            if getattr(node.induction_var, "name", None) == P_STEP_VAR:
                found.append(node)
            walk(node.body)
            return
        for _label, child in _iter_children(node):
            walk(child)

    walk(module.body)
    return found[0] if found else None


def _source_loops(module):
    """Every For in the module this transform has NOT already rewritten."""
    found = []

    def walk(node):
        if isinstance(node, For):
            if getattr(node.induction_var, "name", None) != P_STEP_VAR:
                found.append(node)
            walk(node.body)
            return
        for _label, child in _iter_children(node):
            walk(child)

    walk(module.body)
    return found


def _statement_index(node, path=()):
    """source_stmt path -> statement, using the ledger's structural path spelling."""
    found = {}

    def walk(item, here):
        if isinstance(item, Sequential):
            for index, child in enumerate(item.body):
                walk(child, here + ("s%d" % index,))
            return
        if isinstance(item, (MeshScope, ExecutionScope)):
            walk(item.body, here + (type(item).__name__,))
            return
        if isinstance(item, If):
            walk(item.then_body, here + ("then",))
            walk(item.else_body, here + ("else",))
            return
        if isinstance(item, For):
            name = str(getattr(getattr(item, "induction_var", None), "name", "i"))
            walk(item.body, here + ("for:" + name,))
            return
        found["/".join(here)] = item

    walk(node, tuple(path))
    return found


def _top_bindings(body):
    """The body-level LetStmts, in source order, keyed by the name they bind."""
    out = {}
    for child in body.body:
        if isinstance(child, LetStmt):
            out[child.var.name] = child
    return out


def _is_allocation(stmt):
    return (isinstance(stmt, LetStmt) and isinstance(stmt.value, Call)
            and isinstance(stmt.value.target, AllocTensor))


def _uses_transitively(names, bindings):
    """Closure of body-level bindings reached from names, in SOURCE order."""
    needed, frontier = set(), list(names)
    while frontier:
        name = frontier.pop()
        let = bindings.get(name)
        if let is None or name in needed:
            continue
        needed.add(name)
        frontier.extend(_vars_used(let.value))
    return [name for name in bindings if name in needed]


def _operand_position(access, index):
    """Explicit operand position from action_id 'op:N', else the declared index."""
    tail = str(access.action_id).rsplit(":", 1)
    if len(tail) == 2 and tail[1].isdigit():
        return int(tail[1])
    return int(index)


def _operand_map(statement, accesses):
    """Operand position -> access, per statement. Ambiguity is refused, never guessed."""
    positions = {}
    for index, access in enumerate(accesses):
        position = _operand_position(access, index)
        if position in positions:
            raise PlanError("PLAN_PIPELINE_OPERAND_UNMAPPED",
                            "two accesses claim one operand position",
                            type(getattr(statement, "callable", None)).__name__,
                            "unique positions", position)
        if position >= len(getattr(statement, "args", ())):
            raise PlanError("PLAN_PIPELINE_OPERAND_UNMAPPED",
                            "operand position is outside the statement's operands",
                            type(getattr(statement, "callable", None)).__name__,
                            "position < %d" % len(statement.args), position)
        positions[position] = access
    return positions


def _divide_expr(expr, factor, *, what, source):
    """Prove expr is a whole number of elements and return expr / factor as a PExpr."""
    from .resolved_plan import Add as PAdd
    from .resolved_plan import C as PC
    from .resolved_plan import Mul as PMul
    from .resolved_plan import PExpr

    if not isinstance(expr, PExpr):
        raise PlanError("PLAN_PIPELINE_ADDRESS_UNSUPPORTED",
                        "address is not a typed PExpr", source, "PExpr",
                        type(expr).__name__)
    if expr.op == "const":
        value = int(expr.value)
        if value % factor:
            raise PlanError("PLAN_PIPELINE_ADDRESS_UNALIGNED",
                            what + " is not a whole number of elements", source,
                            "multiple of %d" % factor, value)
        return PC(value // factor)
    if expr.op == "mul":
        left, right = expr.args
        if left.op == "const" and int(left.value) % factor == 0:
            return PMul(PC(int(left.value) // factor), right)
        if right.op == "const" and int(right.value) % factor == 0:
            return PMul(left, PC(int(right.value) // factor))
        raise PlanError("PLAN_PIPELINE_ADDRESS_UNALIGNED",
                        what + " is not provably a whole number of elements", source,
                        "a factor divisible by %d" % factor, expr.to_dict())
    if expr.op == "add":
        return PAdd(_divide_expr(expr.args[0], factor, what=what, source=source),
                    _divide_expr(expr.args[1], factor, what=what, source=source))
    raise PlanError("PLAN_PIPELINE_ADDRESS_UNSUPPORTED",
                    what + " cannot be shown element-aligned", source,
                    "const/mul/add", expr.to_dict())


def _canonical_path(path):
    """One spelling per structural path: a segment "for:NAME" becomes "for".

    The ledger names a loop segment after its induction variable while a Problem may
    name it by position. Both denote the same structural position, so the comparison
    must not depend on which spelling a producer happened to choose.
    """
    parts = []
    for part in str(path).split("/"):
        parts.append("for" if part.startswith("for:") else part)
    return "/".join(parts)


def _normalize_path(path, statements):
    """Resolve a plan source_stmt against the module's statements.

    A Producer may spell a path with its scope prefixes omitted (the ledger records the
    full structural path; a Problem may record it relative to the loop). So: exact match,
    then exact match on the canonical spelling, then a UNIQUE suffix match. The omitted
    part may only be a PREFIX -- a plan path is never a prefix of a statement path, since
    that would let a shorter spelling claim a deeper statement. Ambiguity is refused by
    name; picking one candidate silently is how a rewrite lands on the wrong statement.
    """
    if path in statements:
        return path
    wanted = _canonical_path(path)
    tier = [key for key in statements if _canonical_path(key) == wanted] or            [key for key in statements if _canonical_path(key).endswith("/" + wanted)]
    if len(tier) == 1:
        return tier[0]
    raise PlanError("PLAN_PIPELINE_SOURCE_MISMATCH",
                    "the plan names a statement the module does not have" if not tier
                    else "the plan's statement path is ambiguous in this module",
                    path, "exactly one matching statement", tier[:4])


def _plan_buffer(plan, root):
    for buffer in plan.buffers:
        if buffer.root == root:
            return buffer
    raise PlanError("PLAN_PIPELINE_UNSUPPORTED", "the access names a buffer the plan has no"
                    " record for", root, "root in plan.buffers",
                    [buffer.root for buffer in plan.buffers])


def _ring_view(ring, buffer, index_expr, shape=None):
    """A real TIR view of one ring slot: ring[index : index + extent]."""
    extent = tuple(shape if shape is not None else buffer.shape)
    if any(int(dim) <= 0 for dim in extent):
        raise PlanError("PLAN_PIPELINE_ADDRESS_UNSUPPORTED",
                        "a dynamic extent is not supported by a symbolic ring view",
                        buffer.root, "static positive extents", extent)
    tensor_type = make_tensor_type(extent, dtype=_dtype(buffer.dtype),
                                   storage=StorageKind.SMEM)
    keys = TupleExpr(elements=(index_expr,), type=TupleType((_scalar(),)))
    view = Call(type=tensor_type,
                target=Slice(sizes=extent, strides=(1,) * len(extent)),
                args=(ring, keys))
    return view


def _slot_index(buffer, access, symbols):
    """element index of this access's slot: (k % V) * stride + relative/element."""
    element = _element_bytes(buffer.dtype)
    relative = access.relative_byte_offset
    if relative == OFFSET_UNKNOWN:
        raise PlanError("PLAN_PIPELINE_ADDRESS_UNSUPPORTED",
                        "OFFSET_UNKNOWN cannot be materialized", access.access_id,
                        "typed offset", "OFFSET_UNKNOWN")
    relative_elements = _divide_expr(relative, element,
                                     what="the relative byte offset",
                                     source=access.access_id)
    stride_elements = int(buffer.version_stride_bytes) // element
    versions = int(buffer.slot_count)
    ordinal = _pexpr_to_tir(access.execution_ordinal, symbols)
    index = ordinal
    if versions > 1:
        index = _bin(BinaryKind.MOD, index, _const(versions))
    else:
        index = _const(0)
    if stride_elements != 1:
        index = _bin(BinaryKind.MUL, index, _const(stride_elements))
    offset = _pexpr_to_tir(relative_elements, symbols)
    if _as_int(offset) != 0:
        index = _bin(BinaryKind.ADD, index, offset)
    return index


def _pexpr_to_tir(expr, symbols):
    """Translate a plan PExpr into a REAL rank-0 integer TIR expression."""
    from .resolved_plan import PExpr

    if not isinstance(expr, PExpr):
        raise PlanError("PLAN_PIPELINE_ADDRESS_UNSUPPORTED",
                        "plan expression is not a typed PExpr", "expr", "PExpr",
                        type(expr).__name__)
    if expr.op == "const":
        return _const(int(expr.value))
    if expr.op == "var":
        if expr.name not in symbols:
            raise PlanError("PLAN_PIPELINE_SYMBOL_UNBOUND",
                            "the plan names a symbol this pipeline binds nothing for;"
                            " refusing to guess a value", str(expr.name),
                            sorted(symbols), expr.name)
        return symbols[expr.name]
    left = _pexpr_to_tir(expr.args[0], symbols)
    right = _pexpr_to_tir(expr.args[1], symbols)
    if expr.op == "add":
        kind = BinaryKind.ADD
    elif expr.op == "mul":
        kind = BinaryKind.MUL
    elif expr.op == "floordiv":
        kind = BinaryKind.FLOOR_DIV
    else:
        raise PlanError("PLAN_PIPELINE_ADDRESS_UNSUPPORTED", "unknown plan expression op",
                        "expr", "add/mul/floordiv", expr.op)
    return _bin(kind, left, right)


# The verifier binds by Var IDENTITY, not by name, so these two objects must be the
# SAME instances everywhere in one emitted function: a fresh Var of the same name is a
# different variable and is correctly refused as out of scope.
P_VAR = Var(name=P_STEP_VAR, type=_scalar())
TRIP_VAR = Var(name=N_TRIP_VAR, type=_scalar())


def _p_step():
    return P_VAR


def _trip_var():
    return TRIP_VAR


def _path_key(path):
    """Source order along a structural path: s0 < s1 < ... < then < else."""
    return tuple((0, int(part[1:])) if part.startswith("s") and part[1:].isdigit()
                 else (1, part) for part in str(path).split("/"))


def _allocations(module):
    """Every allocation LetStmt in the module, keyed by the name it binds."""
    found = {}
    for _path, statement in _statement_index(module.body).items():
        if _is_allocation(statement):
            found.setdefault(statement.var.name, statement)
    return found


def _views(module):
    """Every let-bound memory VIEW in the module, keyed by the name it binds.

    A view is how a program names a slice of an arena it does not own: it carries an
    absolute byte address instead of an allocation. Recognising them separately lets the
    hoist refuse such a buffer with a precise reason."""
    found = {}
    for _path, statement in _statement_index(module.body).items():
        if not isinstance(statement, LetStmt):
            continue
        value = statement.value
        if not isinstance(value, Call):
            continue
        if type(getattr(value, "target", None)).__name__ == "TensorView":
            found.setdefault(statement.var.name, statement)
    return found


def _declaration(module, buffer, statements):
    """Find the LetStmt that DECLARES this buffer, and say how it was declared.

    A Problem names a buffer either by the variable it is bound to or by the PATH of the
    declaring statement (the ledger spells roots as "smem@<path>"). Both are honoured,
    because guessing from the shape could attribute a ring to the wrong buffer. Returns
    (statement, kind), where kind is "allocation", "view" or "other", or (None, None).
    """
    for candidate in (buffer.source_decl, buffer.root):
        if not candidate:
            continue
        text = str(candidate)
        if "@" in text:
            text = text.rsplit("@", 1)[1]
        try:
            resolved = _normalize_path(text, statements)
        except PlanError:
            resolved = None
        if resolved is not None and isinstance(statements[resolved], LetStmt):
            statement = statements[resolved]
            return statement, _declaration_kind(statement)
        for statement in statements.values():
            if isinstance(statement, LetStmt) and statement.var.name == text:
                return statement, _declaration_kind(statement)
    return None, None


def _declaration_kind(statement):
    """How a LetStmt declares memory: a real allocation, or a view of someone else's."""
    target = getattr(statement.value, "target", None)
    if isinstance(target, AllocTensor):
        return "allocation"
    if type(target).__name__ == "TensorView":
        return "view"
    return "other"


def _hoist_rings(module, plan, decisions):
    """One ring allocation per buffer, replacing the per-iteration declaration.

    A ring is only possible if all versions live in ONE allocation that outlives the loop,
    and only if that allocation can be ADDRESSED BY A SYMBOL (k % V). A declaration
    written as an absolute-address view cannot, so it is refused by name. The declaration
    is located by path or by name across the whole function, because a buffer may be
    declared just inside the loop (the per-iteration case) or already at function scope.
    The ring SIZING is authoritative from the plan, so a declaration written for one
    version is correctly enlarged to V.
    """
    statements = _statement_index(module.body)
    hoists, drop = {}, {}
    for buffer in plan.buffers:
        statement, kind = _declaration(module, buffer, statements)
        if statement is None:
            raise PlanError(
                "PLAN_PIPELINE_UNSUPPORTED",
                "no declaration in this function carries the buffer's declared name; the"
                " ring cannot be attributed without guessing", buffer.root,
                "a declaration named " + str(buffer.source_decl or buffer.root),
                sorted(name for name, row in statements.items()
                       if isinstance(row, LetStmt)))
        declared = statement.var.name
        if kind == "view":
            raise PlanError(
                "PLAN_PIPELINE_BUFFER_NOT_ALLOCATED",
                "this buffer is declared as an absolute-address VIEW, not as an allocation;"
                " a ring is ONE allocation indexed by k % V, and an absolute address cannot"
                " carry that symbol", buffer.root, "an AllocTensor-declared buffer",
                {"declaration": declared, "kind": kind})
        if kind != "allocation":
            raise PlanError(
                "PLAN_PIPELINE_UNSUPPORTED",
                "this buffer's declaration is neither an allocation nor an address view,"
                " so the transform does not know how to give it versions", buffer.root,
                "an AllocTensor declaration", {"declaration": declared, "kind": kind})
        versions = int(buffer.slot_count)
        if versions < 1:
            raise PlanError("PLAN_PIPELINE_VERSION_INVALID",
                            "a ring needs at least one version", buffer.root, ">=1",
                            versions)
        if buffer.version_stride_bytes is None or buffer.base_offset_bytes is None:
            raise PlanError("PLAN_LAYOUT_UNRESOLVED",
                            "resolve_buffer_layout must run before pipelining",
                            buffer.root, "resolved layout", None)
        element = _element_bytes(buffer.dtype)
        if int(buffer.version_stride_bytes) % element:
            raise PlanError("PLAN_PIPELINE_ADDRESS_UNALIGNED",
                            "the version stride is not a whole number of elements",
                            buffer.root, "multiple of %d" % element,
                            buffer.version_stride_bytes)
        elements = int(buffer.allocation_bytes) // element
        needed = _product(buffer.shape) * versions
        if elements < needed:
            raise PlanError("PLAN_PIPELINE_VERSION_INVALID",
                            "the resolved allocation cannot hold every version",
                            buffer.root, needed, elements)
        ring_type = make_tensor_type((elements,), dtype=_dtype(buffer.dtype),
                                     storage=StorageKind.SMEM)
        ring = Var(name="%s%s" % (declared, RING_SUFFIX), type=ring_type)
        hoists[buffer.root] = ring
        drop[declared] = buffer.root
        decisions.append({"kind": "RING_HOISTED", "root": buffer.root, "from": declared,
                          "to": ring.name, "versions": versions, "elements": elements})
    return hoists, drop

def _slot_base(buffer, symbols, ordinal=None):
    """element index of the current slot inside the ring: (k % V) * stride."""
    element = _element_bytes(buffer.dtype)
    stride_elements = int(buffer.version_stride_bytes) // element
    versions = int(buffer.slot_count)
    index = (_pexpr_to_tir(ordinal, symbols) if ordinal is not None
             else symbols["k"])
    if versions > 1:
        index = _bin(BinaryKind.MOD, index, _const(versions))
    else:
        index = _const(0)
    if stride_elements != 1:
        index = _bin(BinaryKind.MUL, index, _const(stride_elements))
    return index


def _slot_index(buffer, access, symbols):
    """Element index of this access's slot: (k % V) * stride + relative/element.

    k here is the OWNING TASK's ordinal, because that is what symbols["k"] is bound to,
    so a stage-shifted producer and consumer of one buffer get their own slots.
    """
    element = _element_bytes(buffer.dtype)
    relative = access.relative_byte_offset
    if relative == OFFSET_UNKNOWN:
        raise PlanError("PLAN_PIPELINE_ADDRESS_UNSUPPORTED",
                        "OFFSET_UNKNOWN cannot be materialized", access.access_id,
                        "typed offset", "OFFSET_UNKNOWN")
    index = _slot_base(buffer, symbols, access.execution_ordinal)
    offset = _pexpr_to_tir(_divide_expr(relative, element,
                                        what="the relative byte offset",
                                        source=access.access_id), symbols)
    if _as_int(offset) != 0:
        index = _bin(BinaryKind.ADD, index, offset)
    return index


def _rebuild_statement(statement, accesses, plan, symbols, hoists):
    """Replace every mapped operand by its ring-slot view; keep the rest untouched."""
    args = list(statement.args)
    for position, access in _operand_map(statement, accesses).items():
        buffer = _plan_buffer(plan, access.buffer_root)
        ring = hoists.get(access.buffer_root)
        if ring is None:
            raise PlanError("PLAN_PIPELINE_UNSUPPORTED",
                            "this operand's buffer has no ring in this loop",
                            access.access_id, "a hoisted ring", access.buffer_root)
        # The view must have the REPLACED OPERAND's type, not the whole buffer's: an
        # instruction operand is one tile of the buffer, and widening it to the buffer
        # extent both mis-addresses the tile and breaks the op's own operand pattern.
        original = statement.args[position]
        extent = tuple(getattr(getattr(original, "type", None), "shape", ()) or ())
        args[position] = _ring_view(ring, buffer, _slot_index(buffer, access, symbols),
                                    shape=extent if extent else None)
    return Evaluate(callable=statement.callable, args=tuple(args))



def _captured_lets(owned_nodes, body, plan, symbols, hoists, suffix):
    """Fresh per-task copies of the value-form bindings the owned statements use.

    A window bound once per iteration depends on the induction value, so it cannot stay
    shared across tasks: each task re-binds its own copy against its own i. Copying the
    binding rather than re-reading the buffer is what keeps the SSA chain intact.
    """
    bindings = _top_bindings(body)
    used = set()
    for node in owned_nodes:
        used |= _vars_used(node)
    captured, local = [], {}
    for name in _uses_transitively(used, bindings):
        let = bindings[name]
        value = _subst(let.value, local)
        value = _ring_route(value, plan, symbols, hoists)
        fresh = Var(name=name + suffix, type=let.var.type)
        captured.append(LetStmt(var=fresh, value=value))
        local[name] = fresh
    return captured, local


def _ring_route(value, plan, symbols, hoists):
    """Route a captured PointerOf/window over a ring buffer through a slot view.

    A captured value reads a TensorView whose address the plan owns; leaving it pointing
    at the per-iteration allocation would read whichever version happened to be there.
    """
    from tilefoundry.ir.tir.memory.ptr_of import PtrOf

    if not isinstance(value, Call):
        return value
    if not isinstance(value.target, PtrOf):
        return _subst(value, symbols)
    base = value.args[0]
    if not isinstance(base, Call) or not isinstance(base.target, TensorView):
        return _subst(value, symbols)
    address = base.args[0]
    for root, ring in hoists.items():
        buffer = _plan_buffer(plan, root)
        if tuple(getattr(base.type, "shape", ())) != tuple(buffer.shape):
            continue
        return _ring_view(ring, buffer, _slot_base(buffer, symbols))
    return _subst(value, symbols)


def _clone_task_statements(node, path, accessible, plan, symbols, hoists, accesses_by_path,
                           rename=None):
    """One task's own copy of a statement subtree, with per-task fresh bindings.

    A LetStmt is the shared value chain: its copy is renamed so several tasks can each hold
    an instance. An If keeps both arms because the source guard is still a runtime
    predicate; what the pipeline adds is the ordinal guard AROUND the whole block.
    """
    rename = {} if rename is None else rename
    if isinstance(node, Sequential):
        local = dict(rename)
        out = []
        for index, child in enumerate(node.body):
            here = "/".join(path + ("s%d" % index,))
            if here in accesses_by_path:
                rebuilt = _rebuild_statement(child, accesses_by_path[here], plan, symbols,
                                             hoists)
                out.append(dataclasses.replace(
                    rebuilt, args=tuple(_subst(arg, local) for arg in rebuilt.args)))
                continue
            copied = _clone_task_statements(child, path + ("s%d" % index,), accessible,
                                            plan, symbols, hoists, accesses_by_path, local)
            out.append(copied)
            if isinstance(copied, LetStmt):
                local[child.var.name] = copied.var
        return Sequential(body=tuple(out))
    if isinstance(node, (MeshScope, ExecutionScope)):
        return dataclasses.replace(
            node, body=_clone_task_statements(node.body, path + (type(node).__name__,),
                                              accessible, plan, symbols, hoists,
                                              accesses_by_path, dict(rename)))
    if isinstance(node, If):
        return dataclasses.replace(
            node, cond=_subst(node.cond, rename),
            then_body=_clone_task_statements(node.then_body, path + ("then",), accessible,
                                             plan, symbols, hoists, accesses_by_path,
                                             dict(rename)),
            else_body=_clone_task_statements(node.else_body, path + ("else",), accessible,
                                             plan, symbols, hoists, accesses_by_path,
                                             dict(rename)))
    if isinstance(node, LetStmt):
        fresh = Var(name=node.var.name + RENAME_SUFFIX, type=node.var.type)
        return LetStmt(var=fresh, value=_subst(node.value, rename))
    if isinstance(node, Evaluate):
        return dataclasses.replace(node, args=tuple(_subst(arg, rename)
                                                    for arg in node.args))
    return node



def _scope_bindings(body):
    """Every LetStmt visible at any depth of the loop body, keyed by the name it binds.

    A task block must be able to re-bind a value from ITS OWN scope, not only from the
    outermost level: an rmem value bound inside an ExecutionScope is exactly as
    induction-dependent as a top-level window.
    """
    found = {}

    def walk(node):
        if isinstance(node, Sequential):
            for child in node.body:
                walk(child)
            return
        if isinstance(node, LetStmt):
            found.setdefault(node.var.name, node)
            walk(node.value)
            return
        if isinstance(node, If):
            walk(node.then_body)
            walk(node.else_body)
            return
        if isinstance(node, (MeshScope, ExecutionScope)):
            walk(node.body)

    walk(body)
    return found


def _captured_for(statements_owned, bindings, suffix, plan, symbols, hoists, already):
    """Fresh per-block copies of every binding this block transitively reads.

    Copying the BINDING (renamed) rather than re-reading the buffer is what preserves the
    SSA chain; dropping it is the carried-value failure the verifier refuses by name.
    """
    used = set()
    for node in statements_owned:
        used |= _vars_used(node)
    out, local = [], dict(already)
    for name in _uses_transitively(used, bindings):
        let = bindings[name]
        value = _subst(let.value, local)
        value = _ring_route(value, plan, symbols, hoists)
        fresh = Var(name=name + suffix, type=let.var.type)
        out.append(LetStmt(var=fresh, value=value))
        local[name] = fresh
    return out, local


def _block_for(task, owned_paths, bindings, plan, step_value, start_expr, shift,
               accesses_by_path, hoists, suffix):
    """One task block: the k/i BINDINGS, then the statements they index.

    The bindings are emitted OUTSIDE the guard because the guard itself is written in
    terms of k; a guard that reads a name its own arm binds is unverifiable. Returns
    (bindings, guarded_if, guard_text, captured_names).
    """
    k_var = Var(name=K_VAR_PREFIX + task.task_id, type=_scalar())
    i_var = Var(name=I_VAR_PREFIX + task.task_id, type=_scalar())
    # The binder and every use must be the SAME object: the verifier binds by identity,
    # so a second Var with the same name is a different variable and is refused.
    k_use = k_var
    k_value = (_p_step() if shift == 0
               else _bin(BinaryKind.SUB, _p_step(), _const(shift)))
    i_value = _bin(BinaryKind.ADD, start_expr,
                   _bin(BinaryKind.MUL, k_use, _const(step_value)))
    symbols = {"k": k_use, "p": _p_step(), "i": i_var}
    owned_nodes = [accesses_by_path["__nodes__"][path] for path in owned_paths]
    captured, local = _captured_for(owned_nodes, bindings, suffix, plan, symbols, hoists,
                                    {})
    statements = []
    for path, node in zip(owned_paths, owned_nodes, strict=True):
        rebuilt = _rebuild_statement(node, accesses_by_path[path], plan, symbols, hoists)
        statements.append(dataclasses.replace(
            rebuilt, args=tuple(_subst(arg, local) for arg in rebuilt.args)))
    cond = _bin(BinaryKind.AND,
                _bin(BinaryKind.GE, k_use, _const(0), DType.bool),
                _bin(BinaryKind.LT, k_use, _trip_var(), DType.bool), DType.bool)
    source_guard = _source_guard(task, symbols)
    if source_guard is not None:
        cond = _bin(BinaryKind.AND, cond, source_guard, DType.bool)
    bindings_out = [LetStmt(var=k_var, value=k_value),
                    LetStmt(var=i_var, value=i_value)]
    bindings_out.extend(captured)
    guarded = If(cond=cond, then_body=Sequential(body=tuple(statements)),
                 else_body=Sequential(body=()))
    return (bindings_out, guarded, _render(cond),
            tuple(getattr(let.var, "name", "?") for let in captured))

def _has_drain(body):
    """Does this subtree already carry a completion barrier?"""
    if isinstance(body, Sequential):
        for child in body.body:
            if isinstance(child, Evaluate) and isinstance(child.callable, Drain):
                return True
            if isinstance(child, (MeshScope, ExecutionScope)) and _has_drain(child.body):
                return True
    return False


def _source_guard(task, symbols):
    """The plan's own guard for this task, or None. Never invented, never ignored."""
    guard = getattr(task, "guard", None)
    if guard is None:
        return None
    from .resolved_plan import PExpr
    if isinstance(guard, bool):
        return _const(1 if guard else 0, DType.bool) if not guard else None
    if not isinstance(guard, PExpr):
        raise PlanError("PLAN_PIPELINE_UNSUPPORTED",
                        "the task guard is not a typed PExpr", task.task_id, "PExpr",
                        type(guard).__name__)
    tir = _pexpr_to_tir(guard, symbols)
    dtype = getattr(getattr(tir, "type", None), "dtype", None)
    if dtype is not DType.bool:
        tir = _bin(BinaryKind.NE, tir, _const(0))
    return tir



def _replace_loop(module, loop_stmt, replacement):
    """Swap the original For by the pipelined statement, keeping every enclosing scope."""
    def walk(node):
        if node is loop_stmt:
            return replacement
        if isinstance(node, Sequential):
            return Sequential(body=tuple(walk(child) for child in node.body))
        if isinstance(node, (MeshScope, ExecutionScope)):
            return dataclasses.replace(node, body=walk(node.body))
        if isinstance(node, If):
            return dataclasses.replace(node, then_body=walk(node.then_body),
                                       else_body=walk(node.else_body))
        if isinstance(node, For):
            return dataclasses.replace(node, body=walk(node.body))
        return node

    return dataclasses.replace(module, body=walk(module.body))


def apply_software_pipeline(tir_module, plan, *, sidecar_only=False):
    """Rewrite the ORIGINAL loop into the symbolic guarded software pipeline.

    Returns (rewritten_module, PipelineMap). The loop body holds ONE instance of each
    resolved task; the trip may stay symbolic; every resolved (task, k) executes exactly
    once. With sidecar_only the module is returned unchanged so a caller can inspect the
    decision without the verifier mistaking a sidecar for an application.
    """
    from .resolved_plan import SAT_POSTCHECKED

    if not isinstance(plan, ResolvedPlan):
        raise PlanError("PLAN_PIPELINE_REQUIRES_PLAN",
                        "the transform consumes a ResolvedPlan", "plan", "ResolvedPlan",
                        type(plan).__name__)
    if plan.state.planning_status != SAT_POSTCHECKED:
        raise PlanError("PLAN_PIPELINE_REQUIRES_SAT",
                        "a pipeline may only be applied to a post-checked SAT plan",
                        "planning_status", SAT_POSTCHECKED, plan.state.planning_status)
    # A plan that carries a RECORDED replan request must never be pipelined: the schedule
    # it carries is the one the resolver already judged unusable (for example a
    # cross-stage start under enable_offset=False). Refusing here means this transform
    # cannot launder an unusable schedule into executable IR.
    replan = [row for row in (getattr(plan, "resolution_diagnostics", ()) or ())
              if row.get("needs_replan")]
    if replan:
        first = replan[0]
        raise PlanError(
            "PLAN_RESOLVE_OFFSET_WINDOW_VIOLATED"
            if first.get("kind") == "OFFSET_WINDOW_VIOLATED"
            else "PLAN_PIPELINE_REPLAN_REQUIRED",
            "the plan records a hard-constraint violation, so it must be replanned"
            " rather than pipelined: " + str(first.get("detail", first.get("kind"))),
            str(first.get("task_id") or first.get("root") or "plan"),
            "no needs_replan diagnostic", first, needs_replan=True)

    loops = _source_loops(tir_module)
    if not loops:
        raise PlanError("PLAN_PIPELINE_UNSUPPORTED", "no loop to pipeline", "tir", "For",
                        None)
    if len(loops) > 1:
        raise PlanError("PLAN_PIPELINE_UNSUPPORTED",
                        "this transform pipelines exactly one loop per function", "tir",
                        "one For", [str(getattr(loop.induction_var, "name", "?"))
                                    for loop in loops])
    loop_stmt = loops[0]

    step = _as_int(loop_stmt.step)
    if step is None:
        raise PlanError("PLAN_PIPELINE_UNSUPPORTED",
                        "a symbolic loop step is not supported: the guard's lower bound"
                        " would rest on an unchecked assumption", "loop.step",
                        "positive int", _render(loop_stmt.step))
    if step <= 0:
        raise PlanError("PLAN_PIPELINE_UNSUPPORTED", "only a positive step is a pipeline",
                        "loop.step", ">0", step)
    start_constant = _as_int(loop_stmt.start)
    stop_constant = _as_int(loop_stmt.stop)
    for key, live in (("start", start_constant), ("stop", stop_constant), ("step", step)):
        stated = (plan.loop.domain or {}).get(key)
        if isinstance(stated, int) and live is not None and int(stated) != live:
            raise PlanError("PLAN_PIPELINE_DOMAIN_MISMATCH",
                            "the TIR loop does not match the plan's domain", key,
                            int(stated), live)

    if None not in (start_constant, stop_constant):
        trip = max(0, (stop_constant - start_constant + step - 1) // step)
        trip_expr, trip_record = _const(trip), trip
    else:
        span = _bin(BinaryKind.SUB, loop_stmt.stop, loop_stmt.start)
        span = _bin(BinaryKind.ADD, span, _const(step - 1))
        trip_expr = _bin(BinaryKind.MAX, _const(0),
                         _bin(BinaryKind.FLOOR_DIV, span, _const(step)))
        trip_record = None

    decisions = []
    statements = _statement_index(tir_module.body)
    accesses_by_path = {}
    for access in plan.accesses:
        path = _normalize_path(access.source_stmt, statements)
        node = statements[path]
        if not isinstance(node, Evaluate):
            raise PlanError("PLAN_PIPELINE_UNSUPPORTED",
                            "a mapped statement is not an effect statement", path,
                            "Evaluate", type(node).__name__)
        accesses_by_path.setdefault(path, []).append(access)

    body = loop_stmt.body
    hoists, drop = _hoist_rings(tir_module, plan, decisions)
    bindings = _scope_bindings(body)

    minimum = min((int(task.stage) for task in plan.tasks), default=0)
    shifts = {task.task_id: int(task.stage) - minimum for task in plan.tasks}
    depth = max(shifts.values()) if shifts else 0
    if depth > MAX_PIPELINE_DEPTH:
        raise PlanError("PLAN_PIPELINE_UNSUPPORTED", "pipeline depth beyond the cap",
                        "depth", "<=%d" % MAX_PIPELINE_DEPTH, depth)

    nodes = dict(accesses_by_path)
    nodes["__nodes__"] = {path: statements[path] for path in accesses_by_path}
    blocks, records, covered = [], [], set()
    for task in sorted(plan.tasks, key=lambda row: int(row.order)):
        owned = sorted({path for path, rows in accesses_by_path.items()
                        if any(row.task_id == task.task_id for row in rows)},
                       key=_path_key)
        if not owned:
            decisions.append({"kind": "TASK_WITHOUT_STATEMENT", "task_id": task.task_id,
                              "reason": "no access maps this task to a statement, so it"
                                        " has no executable body in this loop"})
            continue
        for path in owned:
            symbols = {"k": _p_step(), "p": _p_step()}
            for name in _expr_symbols(*(row.execution_ordinal
                                        for row in accesses_by_path[path])):
                if name in ("k", "i", "p"):
                    continue
                raise PlanError("PLAN_PIPELINE_SYMBOL_UNBOUND",
                                "an access ordinal uses a symbol the pipeline has no"
                                " binding for", task.task_id, ["i", "k", "p"], name)
        task_bindings, guarded, guard_text, captured = _block_for(
            task, owned, bindings, plan, step, loop_stmt.start,
            shifts[task.task_id], nodes, hoists, "__" + task.task_id)
        blocks.extend(task_bindings)
        blocks.append(guarded)
        covered.update(owned)
        records.append(TaskBlock(task_id=task.task_id, shift=shifts[task.task_id],
                                 k_var=K_VAR_PREFIX + task.task_id,
                                 i_var=I_VAR_PREFIX + task.task_id, guard=guard_text,
                                 source_stmts=tuple(owned), captured=captured))

    for path, statement in statements.items():
        if path in covered or statement is loop_stmt:
            continue
        if not _inside(statement, body, statements):
            continue
        name = getattr(getattr(statement, "var", None), "name", None)
        if isinstance(statement, Evaluate):
            raise PlanError("PLAN_PIPELINE_UNSUPPORTED",
                            "a loop-body effect statement is not owned by any resolved"
                            " task; the pipeline will not drop it silently",
                            type(statement.callable).__name__,
                            "every effect mapped to a task", path)
        if _is_allocation(statement):
            if name in drop:
                decisions.append({"kind": "ALLOCATION_REPLACED_BY_RING", "var": name})
                continue
            decisions.append({"kind": "ALLOCATION_KEPT", "var": name})
            continue
        if isinstance(statement, LetStmt):
            decisions.append({"kind": "VALUE_BOUND_PER_BLOCK", "var": name})
            continue
        raise PlanError("PLAN_PIPELINE_UNSUPPORTED",
                        "an unowned statement kind would be dropped silently",
                        type(statement).__name__, "a mapped statement", path)

    # The physical step sweeps [0, N + H): H = depth is the largest shift, so the last
    # task's last valid instance (k = N - 1) is reached at p = N - 1 + H. A shorter bound
    # would silently drop the drain instances.
    pipelined = For(
        induction_var=_p_step(), start=_const(0),
        stop=(_const(trip + depth) if trip_record is not None
              else _bin(BinaryKind.ADD, _trip_var(), _const(depth))),
        step=_const(1),
        body=Sequential(body=tuple(blocks)))
    # The trip binding goes BEFORE the loop, never inside it: the loop BOUND reads it,
    # and a binder inside the body is not in scope at the stop of its own loop.
    sequence = [LetStmt(var=_trip_var(), value=trip_expr), pipelined]
    # The pipeline does NOT invent a completion barrier. A Drain is MESH-scoped (it must
    # sit where the executing threads are), so completion injection owns its placement;
    # adding one here would smuggle a completion claim into a schedule transform.
    if _has_drain(loop_stmt.body):
        sequence.append(Evaluate(callable=Drain(), args=()))
        decisions.append({"kind": "DRAIN_PRESERVED",
                          "reason": "the source loop already carried the barrier"})

    original_digest = _digest_module(tir_module)
    mapping = PipelineMap(
        schema=PIPELINE_SCHEMA, p_var=P_STEP_VAR, trip_var=N_TRIP_VAR,
        trip=trip_record, depth=depth,
        shifts=tuple(sorted(shifts.items(), key=lambda row: (row[1], row[0]))),
        versions=tuple(sorted((buffer.root, int(buffer.slot_count))
                              for buffer in plan.buffers)),
        blocks=tuple(records),
        hoisted_allocs=tuple({"root": root, "var": ring.name}
                             for root, ring in sorted(hoists.items())),
        rewritten_index=tuple(sorted((row.access_id, row.buffer_root, row.task_id)
                                     for row in plan.accesses)),
        decisions=tuple(decisions), original_digest=original_digest,
        rewritten_digest=original_digest)
    if sidecar_only:
        return tir_module, mapping
    rewritten = _install(tir_module, loop_stmt, sequence,
                         tuple(LetStmt(var=ring,
                                      value=Call(type=ring.type, args=(),
                                                 target=AllocTensor(tensor_type=ring.type)))
                               for ring in hoists.values()),
                         tuple(drop))
    return rewritten, dataclasses.replace(mapping,
                                          rewritten_digest=_digest_module(rewritten))


def _inside(statement, body, statements):
    """Does this statement live inside the loop body (not merely anywhere in the module)?"""
    keys = [key for key, value in statements.items() if value is statement]
    if not keys:
        return False
    inner = {key for key, value in statements.items() if value is body}
    if not inner:
        return False
    prefix = next(iter(inner))
    return any(key == prefix or key.startswith(prefix + "/") for key in keys)



def _expr_symbols(*exprs):
    """Every symbol name a group of plan expressions reads."""
    names = set()
    for expr in exprs:
        if expr is not None:
            names |= set(expr.symbols())
    return names



# ------------------------------------------------------------------ verifying
def _eval_guard(expr, env):
    """Evaluate an emitted guard in a concrete environment. Unknown -> unknown."""
    if isinstance(expr, Constant):
        return expr.value
    if isinstance(expr, Var):
        return env.get(expr.name, _UNKNOWN)
    if isinstance(expr, Call):
        kind = getattr(expr.target, "kind", None)
        values = [_eval_guard(arg, env) for arg in expr.args]
        if any(value is _UNKNOWN for value in values):
            return _UNKNOWN
        left, right = (values + [None, None])[:2]
        try:
            if kind == BinaryKind.GE:
                return left >= right
            if kind == BinaryKind.GT:
                return left > right
            if kind == BinaryKind.LE:
                return left <= right
            if kind == BinaryKind.LT:
                return left < right
            if kind == BinaryKind.EQ:
                return left == right
            if kind == BinaryKind.NE:
                return left != right
            if kind == BinaryKind.AND:
                return bool(left) and bool(right)
            if kind == BinaryKind.OR:
                return bool(left) or bool(right)
            if kind == BinaryKind.ADD:
                return left + right
            if kind == BinaryKind.SUB:
                return left - right
            if kind == BinaryKind.MUL:
                return left * right
            if kind == BinaryKind.MOD:
                return left % right if right else _UNKNOWN
            if kind == BinaryKind.FLOOR_DIV:
                return left // right if right else _UNKNOWN
            if kind == BinaryKind.MAX:
                return max(left, right)
        except (TypeError, ZeroDivisionError):
            return _UNKNOWN
    return _UNKNOWN


class _Unknown:
    def __repr__(self):
        return "<unknown>"


_UNKNOWN = _Unknown()


def verify_pipeline_application(original_module, plan, rewritten_module, mapping, *,
                                require_trip_folding=True):
    """Re-derive the pipeline from the REWRITTEN TIR and judge it.

    Independent of the emission path: it locates the loop by its stable induction name,
    re-reads every block's guard and shift from the IR, and re-simulates the schedule for
    a concrete trip. It refuses (a) a module that is unchanged or not typed-valid,
    (b) a wrong shift, (c) a guard that does not admit exactly the valid instances,
    (d) a block that reads a value nothing in scope binds, and (e) a schedule whose
    resolved instances are not covered exactly once.
    """
    report = {"status": "FAIL", "code": None, "schema": PIPELINE_SCHEMA,
              "blocks": 0, "instances": 0, "expected_instances": 0,
              "trip": None, "depth": 0, "shifts": [], "slots": [],
              "original_digest": getattr(mapping, "original_digest", ""),
              "rewritten_digest": getattr(mapping, "rewritten_digest", "")}
    if rewritten_module is None or rewritten_module is original_module:
        report["code"] = "PIPELINE_NOT_APPLIED"
        return report
    if getattr(mapping, "rewritten_digest", "") == getattr(mapping, "original_digest", ""):
        report["code"] = "PIPELINE_NOT_APPLIED"
        return report
    loop = _first_pipeline_loop(rewritten_module)
    if loop is None:
        report["code"] = "PIPELINE_LOOP_MISSING"
        return report
    report["loops"] = 1
    report["shifts"] = [[task_id, shift] for task_id, shift in mapping.shifts]
    report["depth"] = int(mapping.depth)

    # The shift TABLE is part of the published contract, not a convenience view of the
    # blocks: a consumer reads it to reconstruct the schedule. So it is checked against
    # the IR too, instead of only checking the per-block records. Without this, a map
    # whose table disagreed with its own blocks verified clean.
    declared = {task_id: int(shift) for task_id, shift in mapping.shifts}
    from_blocks = {record.task_id: int(record.shift) for record in mapping.blocks}
    if declared != from_blocks:
        report["code"] = "PIPELINE_SHIFT_TABLE_MISMATCH"
        report["detail"] = {"table": declared, "blocks": from_blocks}
        return report

    trip = mapping.trip
    if trip is None:
        trip_bindings = {child.var.name: child.value
                         for child in _binders_around(rewritten_module, loop)
                         if isinstance(child, LetStmt)}
        if N_TRIP_VAR not in trip_bindings:
            report["code"] = "PIPELINE_TRIP_MISSING"
            return report
        evaluated = _eval_guard(trip_bindings[N_TRIP_VAR], {})
        trip = evaluated if isinstance(evaluated, int) else None
        if trip is None and require_trip_folding:
            report["code"] = "PIPELINE_TRIP_SYMBOLIC"
            return report
    report["trip"] = trip

    stop = _eval_guard(loop.stop, {N_TRIP_VAR: trip} if trip is not None else {})
    if trip is not None and stop != trip + int(mapping.depth):
        report["code"] = "PIPELINE_TRIP_BOUND_MISMATCH"
        report["detail"] = {"expected_stop": trip + int(mapping.depth), "actual": stop}
        return report

    emitted = list(loop.body.body)
    block_ids = [child for child in emitted if isinstance(child, If)]
    report["blocks"] = len(block_ids)
    if len(block_ids) != len(mapping.blocks):
        report["code"] = "PIPELINE_BLOCK_COUNT_MISMATCH"
        report["detail"] = {"mapped": len(mapping.blocks), "emitted": len(block_ids)}
        return report

    # Re-read each block's shift from the IR: k = p - shift, where the binding sits
    # OUTSIDE the guard because the guard itself is written in terms of k.
    shift_bindings = {child.var.name: child.value for child in emitted
                      if isinstance(child, LetStmt)}
    observed = []
    for index, block in enumerate(block_ids):
        record = mapping.blocks[index]
        k_name = getattr(record, "k_var", None)
        if k_name not in shift_bindings:
            report["code"] = "PIPELINE_SHIFT_UNREADABLE"
            report["detail"] = {"block": index, "k_var": k_name,
                                "bindings": sorted(shift_bindings)}
            return report
        k_binding = types.SimpleNamespace(value=shift_bindings[k_name])
        shift = _shift_of(k_binding.value)
        if shift is None:
            report["code"] = "PIPELINE_SHIFT_UNREADABLE"
            report["detail"] = {"block": index, "expr": _render(k_binding.value)}
            return report
        if shift != int(record.shift):
            report["code"] = "PIPELINE_SHIFT_MISMATCH"
            report["detail"] = {"block": record.task_id, "mapped": int(record.shift),
                                "emitted": shift}
            return report
        observed.append((record, shift))

    # Exact coverage: simulate p over [0, N + H) and count each task's admitted instances.
    if trip is not None:
        if trip > MAX_SIMULATED_TRIP:
            report["code"] = "PIPELINE_TRIP_TOO_LARGE"
            report["detail"] = {"trip": trip, "cap": MAX_SIMULATED_TRIP}
            return report
        admitted = {record.task_id: 0 for record, _shift in observed}
        for p in range(trip + int(mapping.depth)):
            env = {P_STEP_VAR: p, N_TRIP_VAR: trip}
            for record, shift in observed:
                env[record.k_var] = p - shift
                env[record.i_var] = p - shift
            for block, (record, _shift) in zip(block_ids, observed, strict=True):
                if _eval_guard(block.cond, env) is True:
                    admitted[record.task_id] += 1
        report["instances"] = sum(admitted.values())
        report["expected_instances"] = trip * len(observed)
        if any(count != trip for count in admitted.values()) or not observed:
            report["code"] = "PIPELINE_COVERAGE_MISMATCH"
            report["detail"] = {"expected_each": trip, "admitted": admitted}
            return report

    # Nothing may read a value no enclosing binder provides.
    for block, (record, _shift) in zip(block_ids, observed, strict=True):
        missing = _unbound_reads(block, rewritten_module)
        if missing:
            report["code"] = "PIPELINE_VALUE_UNBOUND"
            report["detail"] = {"block": record.task_id, "missing": sorted(missing)}
            return report

    report["status"] = "PASS"
    return report


def _binders_around(module, loop):
    """Every statement that DOMINATES the pipelined loop, outermost first.

    The trip binding is emitted beside the loop, not inside it, because the loop bound
    reads it. So the verifier reads binders from the enclosing path: every sibling
    BEFORE the loop in any block that contains it, plus every enclosing wrapper.
    A binder that merely precedes the loop in the same block is in scope at it; only
    binders introduced INSIDE the loop body are not.
    """
    found = []

    def walk(node):
        if isinstance(node, Sequential):
            for child in node.body:
                if child is loop or walk(child):
                    return True
                found.append(child)
            return False
        if isinstance(node, (MeshScope, ExecutionScope)):
            return walk(node.body)
        return False

    walk(module.body)
    return tuple(found)

def _shift_of(expr):
    """Recover the shift from an emitted k = p or k = p - shift binding."""
    if isinstance(expr, Var):
        return 0 if expr.name == P_STEP_VAR else None
    if isinstance(expr, Call):
        if getattr(expr.target, "kind", None) != BinaryKind.SUB:
            return None
        left, right = expr.args
        if isinstance(left, Var) and left.name == P_STEP_VAR:
            value = _as_int(right)
            if value is not None:
                return int(value)
    return None


def _unbound_reads(block, module):
    """Names the block reads that nothing inside it (or the function) binds."""
    from tilefoundry.ir.tir.prim_function import PrimFunction  # noqa: F401

    parameters = {getattr(param, "name", None) for param in getattr(module, "params", ())}
    available = set(parameters) | _bound_by(module.body)
    available.add(N_TRIP_VAR)
    read = _vars_used(block)
    missing = read - available
    return {name for name in missing if name and not _is_external(name)}


def _is_external(name):
    """A symbol deliberately provided by an enclosing construct we do not own."""
    return name.startswith("__")



def _install(module, loop_stmt, sequence, ring_allocs, drop_names):
    """Splice the pipeline in: rings at FUNCTION scope, the loop swapped, old allocs gone.

    The ring must outlive the loop, so its allocation goes where the loop's own scope is
    entered, before the loop. The original declaration is removed wherever it sits;
    otherwise the body would still declare a single version that no ring view reads.
    """
    rings = tuple(ring_allocs)

    def splice(node):
        if node is loop_stmt:
            return tuple(sequence)
        if isinstance(node, Sequential):
            out = []
            for child in node.body:
                if (isinstance(child, LetStmt) and _is_allocation(child)
                        and child.var.name in drop_names):
                    continue
                spliced = splice(child)
                if spliced is None:
                    out.append(child)
                elif isinstance(spliced, tuple):
                    out.extend(spliced)
                else:
                    out.append(spliced)
            return Sequential(body=tuple(out))
        if isinstance(node, (MeshScope, ExecutionScope)):
            inner = splice(node.body)
            return dataclasses.replace(node, body=inner)
        if isinstance(node, If):
            return dataclasses.replace(node,
                                       then_body=splice(node.then_body) or node.then_body,
                                       else_body=splice(node.else_body) or node.else_body)
        if isinstance(node, For):
            inner = splice(node.body)
            return dataclasses.replace(node, body=inner)
        return None

    body = splice(module.body)
    if rings:
        body = _place_rings(body, rings)
    return dataclasses.replace(module, body=body)


def _place_rings(node, rings):
    """Put the ring allocations in the Sequential that DOMINATES the pipelined loop."""
    def has_pipeline(item):
        if isinstance(item, For) and item.induction_var is P_VAR:
            return True
        if isinstance(item, Sequential):
            return any(has_pipeline(child) for child in item.body)
        if isinstance(item, (MeshScope, ExecutionScope)):
            return has_pipeline(item.body)
        return False

    if isinstance(node, Sequential):
        already = any(isinstance(child, LetStmt) and child.var.name in
                      {ring.name for ring in rings} for child in node.body)
        out = []
        for child in node.body:
            if has_pipeline(child):
                if not already:
                    out.extend(rings)
                out.append(child)
                continue
            out.append(child)
        return Sequential(body=tuple(out))
    if isinstance(node, (MeshScope, ExecutionScope)):
        return dataclasses.replace(node, body=_place_rings(node.body, rings))
    return node

