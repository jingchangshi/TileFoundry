"""R2 actual-TIR materialization for resolved-plan/1 (single-version SERIAL).

The materializer changes REAL typed TIR nodes -- allocation types and Copy operands --
and inserts a real guard statement. It never writes a sidecar and calls that applied.
verify_plan_application independently walks the REWRITTEN module and refuses:

  PLAN_NOT_APPLIED               no actual node changed (sidecar-only, or identity)
  APPLICATION_COVERAGE_MISMATCH  a mapped write/access was not applied for every
                                 execution ordinal, or an original effect statement
                                 survived unchanged
  MATERIALIZATION_MAP_MISMATCH   the map points at a statement whose fingerprint is
                                 not the one recorded (raw TIR changed after the map)
  GENERATION_MAPPING_INVALID     generation token is not the full execution ordinal k

R2 scope is deliberately narrow: static SERIAL, one version (slot 0). Dynamic or
multi-version loops are refused with PLAN_MATERIALIZATION_UNSUPPORTED (R3).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import fields, is_dataclass

from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.tir.memory import AllocTensor
from tilefoundry.ir.tir.memory.tensor_view import TensorView
from tilefoundry.ir.tir.memory.copy import Copy
from tilefoundry.ir.tir.flag import Drain
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, For, If, LetStmt, Return, Sequential, While
from tilefoundry.ir.types import Layout, PointerType, make_tensor_type
from tilefoundry.ir.types.storage import StorageKind

from .resolved_plan import (OFFSET_UNKNOWN, STATE_APPLIED, STATE_SKELETON,
                            MaterializationMap, PlanError, PlanState, ResolvedPlan,
                            generation_token, loop_trip_count)

MAX_STATIC_TRIP = 64


def _simple(value):
    if isinstance(value, Constant):
        return ("Constant", value.value)
    if isinstance(value, Var):
        return ("Var", value.name, _simple(value.type))
    if isinstance(value, (tuple, list)):
        return tuple(_simple(v) for v in value)
    if is_dataclass(value) and not isinstance(value, type):
        out = [type(value).__name__]
        for field in fields(value):
            if field.name in ("metadata", "type"):
                continue
            out.append((field.name, _simple(getattr(value, field.name))))
        return tuple(out)
    try:
        if type(value).__name__ in ("TensorType", "Layout", "PointerType"):
            return repr(value)
    except Exception:  # noqa: BLE001
        pass
    return value


def fingerprint(node) -> str:
    payload = json.dumps(_simple(node), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def tir_digest(module) -> str:
    return fingerprint(module)


def _record(found, path, node):
    """File a statement under its one canonical (ledger) path.

    Aliases are NOT stored here. A dict that files one node under several keys makes
    len(index) count the same statement repeatedly, which silently broke a caller that counts
    rewrites ("2 * ROWS copies"). Dialect tolerance belongs in the LOOKUP, not in the index.
    """
    found.setdefault("/".join(path), node)


def _segments_equal(want, have):
    """Do two path segments name the same structural step?

    The loop segment has several legitimate spellings in this tree: the ledger writes
    "for:<var>" (it carries the induction variable) and "for"; older fixtures also wrote
    "for" without a name. They denote the same step, so a lookup must treat them as equal
    rather than declaring one dialect correct and silently missing the other.
    """
    if want == have:
        return True
    want_loop = want.startswith("for") or want.startswith("while")
    have_loop = have.startswith("for") or have.startswith("while")
    return want_loop and have_loop


def lookup_statement(statements, path):
    """Resolve a plan/fixture path against an index that speaks the LEDGER dialect.

    Resolution is a LOOKUP, not a second index: exact hit first, then segment-wise equality
    (which absorbs the for/<var> spelling difference), then a unique suffix. Ambiguity is an
    error -- picking one candidate silently is how a rewrite lands on the wrong statement.
    """
    if path in statements:
        return statements[path]
    wanted = path.split("/") if path else []
    if not wanted:
        return None

    # TRY PROGRESSIVELY SHORTER SUFFIXES, LONGEST FIRST.
    #
    # The index is unprefixed ("s0/MeshScope/s0/for:i/s2") while a plan path carries the
    # function name ("k/s0/MeshScope/s0/for:i/s2"), so the two differ by a PREFIX. The previous
    # version only compared EQUAL-length suffixes, so it required the lengths to match and then
    # compared the wrong segments: it returned None for every plan path, the caller read None as
    # "not that shape", and materialisation fell through to a branch that refused with
    # PLAN_MATERIALIZATION_UNSUPPORTED on a statement that was present all along.
    #
    # Longest-first matters: stopping at the first length that matches anything prefers the most
    # specific answer, so a short suffix that appears in several places cannot win.
    for length in range(len(wanted), 0, -1):
        tail = wanted[-length:]
        matches = [node for key, node in statements.items()
                   if len(key.split("/")) >= length
                   and all(_segments_equal(want, have)
                           for want, have in zip(tail, key.split("/")[-length:]))]
        if len(matches) == 1:
            return matches[0]
        if matches:
            # Ambiguous at this length. A shorter key cannot disambiguate it, and picking one
            # silently is how a rewrite lands on the wrong statement.
            return None
    return None


def index_statements(module) -> dict:
    """Path -> statement, using the LEDGER's structural spelling (see index_statements_ledger).

    This index exists to let a caller FIND a statement; every caller then compares what it
    finds against a plan path that came from the ledger. So it must speak the ledger's
    dialect exactly: function-name prefix, Sequential sN, MeshScope, ExecutionScope,
    "for:<var>". It previously emitted neither the function prefix nor the scope segments,
    which made it useless for its one purpose while still looking like a working index.
    """
    from tilefoundry.ir.tir.stmts import ExecutionScope, MeshScope

    found = {}
    root_name = str(getattr(module, "name", "") or "")

    def walk(stmt, path):
        if isinstance(stmt, Sequential):
            for index, child in enumerate(stmt.body):
                walk(child, path + ("s%d" % index,))
            return
        if isinstance(stmt, (MeshScope, ExecutionScope)):
            # A mesh- or execution-scoped region is a STRUCTURAL segment, not a leaf. Not
            # descending it is what hid the loop inside MeshScope: the kernel's only For sits
            # at k/s0/MeshScope/s0/for:i, and a walk that stops at MeshScope reports that the
            # program has no loop at all.
            walk(stmt.body, path + (type(stmt).__name__,))
            return
        if isinstance(stmt, For):
            # The loop itself is a statement too, and it is filed under BOTH spellings of its
            # own segment: the ledger writes "for:<var>", the R2/R3 fixtures write "for".
            _record(found, path, stmt)
            name = str(getattr(getattr(stmt, "induction_var", None), "name", "i"))
            walk(stmt.body, path + ("for:" + name,))
            return
        if isinstance(stmt, If):
            _record(found, path, stmt)
            walk(stmt.then_body, path + ("then",))
            walk(stmt.else_body, path + ("else",))
            return
        _record(found, path, stmt)

    walk(module.body, ((root_name,) if root_name else ()))
    return found


def _loop_body_can_execute(node) -> bool:
    """Can this loop's body run at least once?

    Only a PROVABLY empty loop is dead; anything unprovable is left reachable. Assuming a
    loop runs zero times when its trip count is merely unknown would discard real applied
    writes, which is the opposite failure and just as wrong.

      While(False)              -> False. While(True)/runtime condition -> True.
      For with a runtime bound  -> True (unprovable).
      For, all bounds constant  -> the true trip count, for BOTH step signs:
                                     step > 0 : ceil((stop - start) / step)
                                     step < 0 : ceil((start - stop) / -step)
                                     step == 0 or span <= 0 : 0
    """
    if isinstance(node, While):
        cond = getattr(node, "cond", None)
        return not (isinstance(cond, Constant) and getattr(cond, "value", None) is False)
    values = []
    for bound in (getattr(node, "start", None), getattr(node, "stop", None),
                  getattr(node, "step", None)):
        if isinstance(bound, Constant) and isinstance(getattr(bound, "value", None), int):
            values.append(int(bound.value))
        else:
            return True
    start, stop, step = values
    if step == 0:
        return False
    span, magnitude = (stop - start, step) if step > 0 else (start - stop, -step)
    if span <= 0:
        return False
    return -(-span // magnitude) > 0


def _terminated_by_return(children, index) -> bool:
    """Are the statements from ``index`` on dead because an earlier sibling returns?

    Conservative: only a DIRECT earlier Return counts. A Return inside an If arm does not
    necessarily terminate the block, so it must not mark live statements dead.
    """
    for earlier in children[:index]:
        if isinstance(earlier, Return):
            return True
    return False


def _alias_unprefixed(found, root_name):
    """Also index every path WITHOUT the function-name prefix.

    TWO PRODUCERS SPELL PATHS DIFFERENTLY, and both are legitimate. The ledger builds an
    instruction path as "/".join((ledger.function, *structural_path)) because a plan can span
    functions; the R2/R3 fixtures name a statement relative to the loop ("s1/for/s0") because
    they were authored against a single-function program. A lookup index that speaks only one
    dialect silently fails the other -- returning None, which a caller reads as "not that
    shape" rather than as an error.

    Both spellings are registered for the SAME node, and they cannot collide because one
    begins with the function name and the other does not. An exact lookup therefore always
    resolves, and a suffix match is only reached when there was no exact hit.
    """
    if not root_name:
        return
    prefix = root_name + "/"
    for key, node in list(found.items()):
        if key.startswith(prefix):
            found.setdefault(key[len(prefix):], node)


def index_statements_ledger(module, *, reachable_only=False) -> dict:
    """Path index that MATCHES the ledger's structural paths (Sequential sN, MeshScope,
    ExecutionScope, for:<var>, If then/else), so R2 can find real instruction statements.

    ``reachable_only`` excludes statements sealed inside a statically dead branch. Without
    it, If(False).then and If(True).else were walked unconditionally, so a real rewritten
    WRITE moved into a dead branch still counted as applied and verify_plan_application
    still reported PASS with applied_accesses unchanged. Same defect class as C's Set/Wait
    reachability: a predicate granted to some consumers and silently withheld from others.
    """
    from tilefoundry.ir.tir.stmts import ExecutionScope, MeshScope

    found = {}

    # THIS INDEX IS UNPREFIXED, AND IT MUST STAY THAT WAY.
    #
    # The ledger spells an instruction path as "/".join((ledger.function, *path)), but the
    # callers of THIS index are not all ledger callers: C's rewritten_paths_by_action and
    # _anchor_ordinals address statements relative to the function. Prefixing here made the
    # keys and those callers disagree in a way that fails silently -- a miss returns None, and
    # None reads as "not this shape" rather than as an error.
    #
    # Callers that DO hold a plan path get it resolved by lookup_statement, which tries the
    # exact key, then a unique suffix. That keeps the index canonical (one node, one key, so a
    # fingerprint maps to exactly one path) while still accepting either spelling.
    root_name = ""

    def walk(node, path, reachable=True):
        if isinstance(node, Sequential):
            for index, child in enumerate(node.body):
                # Statements after a Return in the same Sequential never execute. Same
                # fourth-form rule as C's _walk; kept in step via the parity test so one
                # side cannot drift.
                child_reachable = reachable and not _terminated_by_return(node.body, index)
                walk(child, path + ("s%d" % index,), child_reachable)
            return
        if isinstance(node, (MeshScope, ExecutionScope)):
            walk(node.body, path + (type(node).__name__,), reachable)
            return
        if isinstance(node, (For, While)):
            if reachable or not reachable_only:
                found["/".join(path)] = node
            # A loop that provably runs zero times executes nothing inside it, so a WRITE
            # sealed in its body is not an applied write. The census formerly descended into
            # every loop body unconditionally: reachability was taught about If but withheld
            # from loops, which is the same defect class one statement kind over.
            body_reachable = reachable and _loop_body_can_execute(node)
            if isinstance(node, While):
                walk(node.body, path + ("while",), body_reachable)
            else:
                name = getattr(getattr(node, "induction_var", None), "name", "i")
                walk(node.body, path + ("for:" + str(name),), body_reachable)
            return
        if isinstance(node, If):
            if reachable_only:
                cond = getattr(node, "cond", None)
                value = getattr(cond, "value", None)
                # A constant condition statically kills one arm. A RUNTIME condition decides
                # nothing, so both arms stay reachable -- the taken branch is not knowable
                # statically and dropping it would discard real applications.
                then_reachable = reachable and not (isinstance(cond, Constant) and value is False)
                else_reachable = reachable and not (isinstance(cond, Constant) and value is True)
            else:
                then_reachable = else_reachable = reachable
            walk(node.then_body, path + ("then",), then_reachable)
            walk(node.else_body, path + ("else",), else_reachable)
            return
        if reachable or not reachable_only:
            found["/".join(path)] = node

    walk(module.body, ((root_name,) if root_name else ()))
    # NO ALIASES. ONE NODE, ONE KEY -- BECAUSE A DUPLICATED KEY BREAKS FINGERPRINT GROUPING.
    #
    # Storing the same node under both the prefixed and unprefixed spelling makes a
    # fingerprint lookup return TWO paths for one statement. C's rewritten_paths_by_action
    # groups by fingerprint and then anchors on the paths it collected, so the duplicate made
    # it anchor on a spelling whose descendants were the OTHER spelling's: 24 materialisation
    # entries, 0 fingerprint hits, and COMPLETION_ANCHOR_NOT_FOUND for a path that was there.
    #
    # Dialect tolerance belongs in lookup_statement (exact key first, then a unique suffix).
    # A LOOKUP cannot make one statement look like two; an INDEX can.
    return found


def _first_for(module):
    for stmt in index_statements(module).values():
        if isinstance(stmt, For):
            return stmt
    return None


def _static_int(expr):
    return int(expr.value) if isinstance(expr, Constant) and isinstance(expr.value, int) else None


def _view(access, plan, env):
    buffer = next(b for b in plan.buffers if b.root == access.buffer_root)
    if access.relative_byte_offset == OFFSET_UNKNOWN:
        raise PlanError("PLAN_MATERIALIZATION_UNSUPPORTED",
                        "cannot materialize OFFSET_UNKNOWN in R2", access.access_id,
                        "typed offset", OFFSET_UNKNOWN)
    slot = int(access.slot.evaluate(env))
    relative = int(access.relative_byte_offset.evaluate(env))
    start = int(buffer.base_offset_bytes) + slot * int(buffer.version_stride_bytes) + relative
    shape = tuple(buffer.shape)
    from tilefoundry.ir.types import DType  # noqa: PLC0415

    storage = StorageKind.SMEM if buffer.space in ("ascend.ub", "smem", "ub") else StorageKind.GMEM
    dtype = getattr(DType, str(buffer.dtype), buffer.dtype)
    tensor_type = make_tensor_type(shape, dtype=dtype, storage=storage)
    pointer_type = PointerType(tensor_type.dtype, tensor_type.storage)
    address = Constant(type=pointer_type, value=start)
    return Call(type=tensor_type,
                target=TensorView(storage=tensor_type.storage, dtype=str(buffer.dtype),
                                  layout=Layout(shape), shape=shape),
                args=(address,))


def _access_env(access, plan, env, k):
    """Evaluate the access's OWN ordinal, then derive slot(k)=k%V.

    A stage-shifted producer/consumer must use the AccessPlan ordinal expression,
    not the materializer's current loop iteration i/k.
    """
    buffer = next(b for b in plan.buffers if b.root == access.buffer_root)
    scoped = dict(env)
    ordinal = getattr(access, "execution_ordinal", None)
    if ordinal is None:
        raise PlanError("PLAN_STAGE_MAPPING_INVALID", "access has no ordinal expression",
                        access.access_id, "typed ordinal", None)
    try:
        access_k = int(ordinal.evaluate({**scoped, "k": int(k), "i": env.get("i", int(k))}))
    except PlanError:
        raise PlanError("PLAN_STAGE_MAPPING_INVALID", "access ordinal cannot be evaluated",
                        access.access_id, "evaluable PExpr", ordinal)
    scoped["k"] = access_k
    scoped["slot"] = access_k % int(buffer.slot_count)
    return scoped


def _phase_of(loop, iteration):
    for name in ("prologue", "steady", "drain"):
        domain = getattr(loop, name + "_domain", None) or {}
        if domain.get("start", 0) <= iteration < domain.get("stop", 0):
            return name
    return "steady"


def _bool_type():
    return make_tensor_type((), dtype="bool", storage="umat")


def materialize_plan(tir_module, plan: ResolvedPlan, *, sidecar_only=False,
                     skip_access_ids=()):
    """Apply the plan to the actual TIR. Returns (rewritten_module, MaterializationMap)."""
    map_entries = []
    _peek = index_statements_ledger(tir_module)
    _groups = {a.source_stmt for a in plan.accesses}

    def _peek_at(path):
        # THE LEDGER INDEX IS UNPREFIXED; PLAN PATHS ARE PREFIXED WITH THE FUNCTION NAME.
        # A bare .get() therefore misses every single one and the branch below was reading
        # None, so the instruction-plan path was never taken and R2 then refused with
        # PLAN_MATERIALIZATION_UNSUPPORTED on a source_stmt it could have found. Resolve
        # through lookup_statement, which accepts either spelling.
        return lookup_statement(_peek, path)

    if any(isinstance(_peek_at(p), Evaluate)
           and type(_peek_at(p).callable).__name__ != "Copy"
           for p in _groups):
        return materialize_instruction_plan(tir_module, plan, sidecar_only=sidecar_only)
    if plan.state.application_status != STATE_SKELETON:
        raise PlanError("PLAN_ALREADY_APPLIED", "plan is not in SKELETON_NOT_APPLIED",
                        "application_status", STATE_SKELETON, plan.state.application_status)
    if plan.loop.application_mode != "SERIAL":
        raise PlanError("PLAN_MATERIALIZATION_UNSUPPORTED", "R2 supports SERIAL only",
                        "application_mode", "SERIAL", plan.loop.application_mode)
    for buffer in plan.buffers:
        # R3 supports static V1 or V2 (the latter is the ring case: slot=k%V).
        # More than two versions, or any dynamic loop, remains explicitly unsupported.
        if int(buffer.slot_count) not in (1, 2):
            raise PlanError("PLAN_MATERIALIZATION_UNSUPPORTED",
                            "R3 supports static V1/V2 only; >2 versions is later",
                            buffer.root, [1, 2], buffer.slot_count)
        if buffer.version_stride_bytes is None or buffer.base_offset_bytes is None:
            raise PlanError("PLAN_LAYOUT_UNRESOLVED", "resolve_buffer_layout was not run",
                            buffer.root, "resolved layout", None)
    loop_stmt = _first_for(tir_module)
    if loop_stmt is None:
        raise PlanError("PLAN_MATERIALIZATION_UNSUPPORTED", "no loop to materialize",
                        "tir", "For", None)
    start = _static_int(loop_stmt.start)
    stop = _static_int(loop_stmt.stop)
    step = _static_int(loop_stmt.step)
    if None in (start, stop, step):
        raise PlanError("PLAN_MATERIALIZATION_UNSUPPORTED", "dynamic loop is R4",
                        "loop", "static ints", [start, stop, step])
    trip = loop_trip_count(plan.loop)
    if trip > MAX_STATIC_TRIP:
        raise PlanError("PLAN_MATERIALIZATION_UNSUPPORTED", "static unroll cap exceeded",
                        "trip", "<=%d" % MAX_STATIC_TRIP, trip)

    statements = index_statements(tir_module)
    groups = {}
    for access in plan.accesses:
        groups.setdefault(access.source_stmt, []).append(access)
    originals = {}
    for path in groups:
        stmt = lookup_statement(statements, path)
        if not isinstance(stmt, Evaluate) or not isinstance(stmt.callable, Copy):
            raise PlanError("PLAN_MATERIALIZATION_UNSUPPORTED",
                            "source_stmt is not an Evaluate(Copy)", path,
                            "Evaluate(Copy)", type(stmt).__name__)
        originals[path] = stmt

    body = []
    allocs = []
    for stmt in tir_module.body.body:
        if (isinstance(stmt, LetStmt) and isinstance(stmt.value, Call)
                and isinstance(stmt.value.target, AllocTensor)):
            match = next((b for b in plan.buffers if b.root == stmt.var.name), None)
            if match is not None:
                per = {"f32": 4, "bf16": 2, "f16": 2}.get(str(match.dtype), 4)
                number = max(1, int(match.allocation_bytes) // per)
                new_type = make_tensor_type((number,), dtype=match.dtype,
                                            storage=StorageKind.SMEM)
                new_var = Var(name=stmt.var.name, type=new_type)
                new_stmt = LetStmt(var=new_var,
                                   value=Call(type=new_type,
                                              target=AllocTensor(tensor_type=new_type),
                                              args=()))
                allocs.append({"root": match.root, "var": stmt.var.name,
                               "old_fingerprint": fingerprint(stmt),
                               "new_fingerprint": fingerprint(new_stmt),
                               "allocation_bytes": int(match.allocation_bytes)})
                body.append(new_stmt)
                continue
        if stmt is loop_stmt:
            phase_blocks = {"prologue": [], "steady": [], "drain": []}
            for k in range(trip):
                iteration = start + k * step
                env = {"i": iteration, "k": k, "slot": 0}
                generation_token(plan.loop, iteration, supplied=k)
                phase = _phase_of(plan.loop, iteration)
                for path, accesses in groups.items():
                    original = originals[path]
                    if any(a.access_id in skip_access_ids for a in accesses):
                        phase_blocks[phase].append(original)  # deliberately un-applied
                        continue
                    if len(accesses) != len(original.args):
                        raise PlanError("APPLICATION_COVERAGE_MISMATCH",
                                        "one access per operand is required", path,
                                        len(original.args), len(accesses))
                    scoped = [_access_env(access, plan, env, k) for access in accesses]
                    new_args = tuple(_view(access, plan, scoped[index])
                                     for index, access in enumerate(accesses))
                    rewritten = Evaluate(callable=Copy(), args=new_args)
                    phase_blocks[phase].append(If(
                        cond=Constant(type=_bool_type(), value=bool(0 <= k < trip)),
                        then_body=Sequential(body=(rewritten,)),
                        else_body=Sequential(body=())))
                    for access, access_env in zip(accesses, scoped):
                        access_k = int(access_env["k"])
                        map_entries.append({
                            "access_id": access.access_id,
                            "action_id": access.action_id,
                            "task_id": access.task_id,
                            "mode": access.mode, "k": k, "loop_k": k,
                            "generation": access_k,
                            "old_stmt": path,
                            "old_fingerprint": fingerprint(original),
                            "new_fingerprint": fingerprint(rewritten),
                            "new_stmt": "unrolled:k%d:%s" % (k, access.access_id)})
            for phase_name in ("prologue", "steady", "drain"):
                if phase_blocks[phase_name]:
                    body.append(If(cond=Constant(type=_bool_type(), value=True),
                                   then_body=Sequential(body=tuple(phase_blocks[phase_name])),
                                   else_body=Sequential(body=())))
            continue
        body.append(stmt)

    # R4 terminal drain: an actual Drain node is part of the materialized TIR.
    body.append(Evaluate(callable=Drain(), args=()))

    original_digest = tir_digest(tir_module)
    if sidecar_only:
        return tir_module, MaterializationMap(entries=tuple(map_entries),
                                              allocs=tuple(allocs), controls=(),
                                              original_digest=original_digest,
                                              rewritten_digest=original_digest)
    rewritten = PrimFunction(name=tir_module.name, params=tir_module.params,
                             body=Sequential(body=tuple(body)),
                             output_count=tir_module.output_count,
                             target=tir_module.target)
    materialization = MaterializationMap(
        entries=tuple(map_entries), allocs=tuple(allocs),
        controls=tuple({"k": k, "kind": "guard"} for k in range(trip)),
        original_digest=original_digest, rewritten_digest=tir_digest(rewritten))
    return rewritten, materialization


def apply_plan(plan: ResolvedPlan, materialization: MaterializationMap) -> ResolvedPlan:
    """Mark the plan APPLIED only after verify_plan_application has accepted it."""
    return ResolvedPlan(**{**plan.__dict__,
                           "state": PlanState(plan.state.planning_status, STATE_APPLIED,
                                              plan.state.completion_status),
                           "materialization": materialization})


def _expected_generation(plan, access_id, loop_k, start, step):
    access = next((a for a in plan.accesses if a.access_id == access_id), None)
    if access is None:
        return None
    ordinal = getattr(access, "execution_ordinal", None)
    if ordinal is None:
        return None
    iteration = start + int(loop_k) * step
    try:
        return int(ordinal.evaluate({"i": iteration, "k": int(loop_k),
                                     "slot": int(loop_k) % 1024}))
    except Exception:  # noqa: BLE001
        return None


def _combined_index(module) -> dict:
    merged = dict(index_statements(module))
    merged.update(index_statements_ledger(module))
    return merged


def verify_plan_application(original_module, plan: ResolvedPlan, rewritten_module,
                            materialization: MaterializationMap, *, require_drain=True) -> dict:
    """Independent walk of the REWRITTEN TIR: real change + full coverage + map fidelity."""
    report = {"status": "FAIL", "code": None, "applied_accesses": 0,
              "expected_applications": 0, "allocations_applied": 0,
              "controls_applied": 0, "original_digest": materialization.original_digest,
              "rewritten_digest": materialization.rewritten_digest}
    # ONE local index per module for this verification pass (no repeated full-tree walks).
    original_index = _combined_index(original_module)
    rewritten_index = _combined_index(rewritten_module)
    # The reachable-only views are what "the plan was applied" must be judged on: a rewritten
    # WRITE sealed in a dead branch executes never, so its presence in the rewritten tree is
    # not evidence of application. Keeping a separate full index preserves the map-fidelity
    # checks (a fingerprint may legitimately live anywhere) while coverage uses only reachable
    # statements.
    rewritten_reachable = index_statements_ledger(rewritten_module, reachable_only=True)
    rewritten_reachable_ids = {id(stmt) for stmt in rewritten_reachable.values()}
    if (rewritten_module is None or rewritten_module is original_module
            or materialization.rewritten_digest == materialization.original_digest):
        report["code"] = "PLAN_NOT_APPLIED"
        return report
    if require_drain:
        actual_has_drain = any(
            isinstance(stmt, Evaluate) and isinstance(stmt.callable, Drain)
            and id(stmt) in rewritten_reachable_ids
            for stmt in rewritten_index.values())
        if not actual_has_drain:
            report["code"] = "PLAN_DRAIN_MISSING"
            return report
    writes = [a for a in plan.accesses if a.mode == "write"]
    trip = loop_trip_count(plan.loop)
    report["expected_applications"] = len(writes) * trip
    counts = {}
    for entry in materialization.entries:
        counts[entry["access_id"]] = counts.get(entry["access_id"], 0) + 1
    for access in writes:
        if counts.get(access.access_id, 0) != trip:
            report["code"] = "APPLICATION_COVERAGE_MISMATCH"
            report["detail"] = {"access_id": access.access_id,
                                "expected": trip, "actual": counts.get(access.access_id, 0)}
            return report
    # Real missing-pointer mutation: if a WRITE statement the plan must rewrite is still the
    # IDENTICAL original object, its operand/address was not applied -> coverage mismatch.
    # (Reads that already satisfy the layout are allowed to be preserved; this is NOT the old
    # "every covered node must disappear" proxy.)
    rewritten_ids = {id(stmt) for stmt in rewritten_index.values()}
    for path in {access.source_stmt for access in plan.accesses if access.mode == "write"}:
        original = original_index.get(path)
        if id(original) in rewritten_ids:
            report["code"] = "APPLICATION_COVERAGE_MISMATCH"
            report["detail"] = {"unrewritten_write": path}
            return report
    # A rewritten WRITE that exists but cannot execute is NOT an applied write. Fingerprints
    # are compared against the REACHABLE set only, so sealing a rewritten write in a dead
    # branch -- or inside a loop whose body provably never runs (While(False), zero-trip For)
    # -- is refused instead of passing on map fidelity alone.
    reachable_fingerprints = {fingerprint(stmt) for stmt in rewritten_reachable.values()}
    for entry in materialization.entries:
        if entry["new_fingerprint"] in reachable_fingerprints:
            continue
        if entry["new_fingerprint"] in {fingerprint(stmt) for stmt in rewritten_index.values()}:
            report["code"] = "PLAN_WRITE_UNREACHABLE"
            report["detail"] = {"access_id": entry["access_id"], "k": entry["k"],
                                "reason": "rewritten statement is present but statically "
                                          "unreachable (dead branch arm, While(False), or a "
                                          "zero-trip counted loop), so it never executes"}
            return report
    # Map fidelity: recorded fingerprints must equal actual rewritten statements.
    rewritten_stmts = list(rewritten_index.values())
    rewritten_fingerprints = {fingerprint(stmt) for stmt in rewritten_stmts}
    for entry in materialization.entries:
        if entry["new_fingerprint"] not in rewritten_fingerprints:
            report["code"] = "MATERIALIZATION_MAP_MISMATCH"
            report["detail"] = {"access_id": entry["access_id"], "k": entry["k"]}
            return report
        loop_k = entry.get("loop_k", entry["k"])
        expected_generation = _expected_generation(plan, entry["access_id"], loop_k,
                                                   int(plan.loop.domain["start"]),
                                                   int(plan.loop.domain["step"]))
        if expected_generation is None or int(entry["generation"]) != expected_generation:
            report["code"] = "GENERATION_MAPPING_INVALID"
            report["detail"] = {"access_id": entry["access_id"],
                                "generation": entry["generation"],
                                "expected": expected_generation, "loop_k": loop_k}
            return report
    # Allocation sizes are INFORMATIONAL: a buffer whose layout already matches must not be
    # forced to change. Only an alloc that claims to be applied but is absent from the rewritten
    # module is a real map error.
    rewritten_allocs = {fingerprint(s) for s in rewritten_index.values() if isinstance(s, LetStmt)}
    for alloc in materialization.allocs:
        if alloc["new_fingerprint"] not in rewritten_allocs:
            report["code"] = "MATERIALIZATION_MAP_MISMATCH"
            report["detail"] = {"alloc": alloc}
            return report
        report["allocations_applied"] += 1
    report["applied_accesses"] = sum(counts.get(a.access_id, 0) for a in writes)
    report["controls_applied"] = len(materialization.controls)
    report["status"] = "PASS"
    return report


def replay_plan_addresses(plan: ResolvedPlan):
    """CPU replay of every access address over the static serial loop.

    Proves ranges stay inside their allocation and the generation token is the full
    execution ordinal k for every iteration; never a string/modulo argument.
    """
    from .resolved_plan import address_range
    trip = loop_trip_count(plan.loop)
    max_versions = max((int(buffer.slot_count) for buffer in plan.buffers), default=1)
    if trip < max_versions + 1:
        raise PlanError("PLAN_REPLAY_INCOMPLETE",
                        "fewer than V+1 iterations cannot prove a ring reused a slot",
                        "loop", max_versions + 1, trip)
    start, step = int(plan.loop.domain["start"]), int(plan.loop.domain["step"])
    rows = []
    for k in range(trip):
        iteration = start + k * step
        for access in plan.accesses:
            buffer = next(b for b in plan.buffers if b.root == access.buffer_root)
            env = {"i": iteration, "k": k, "slot": k % int(buffer.slot_count)}
            ordinal = getattr(access, "execution_ordinal", None)
            if ordinal is None:
                raise PlanError("PLAN_STAGE_MAPPING_INVALID", "access has no ordinal expression",
                                access.access_id, "typed ordinal", None)
            access_k = int(ordinal.evaluate(env))
            env["k"] = access_k
            env["slot"] = access_k % int(buffer.slot_count)
            generation = int(access.generation.evaluate(env))
            if generation != access_k:
                raise PlanError("GENERATION_MAPPING_INVALID",
                                "generation token is not the access execution ordinal",
                                access.access_id, access_k, generation)
            slot = int(access.slot.evaluate(env))
            if slot != access_k % int(buffer.slot_count):
                raise PlanError("GENERATION_MAPPING_INVALID",
                                "slot is not k%V for this access",
                                access.access_id, access_k % int(buffer.slot_count), slot)
            begin, end = address_range(plan, access, env)
            rows.append({"access_id": access.access_id, "buffer": access.buffer_root,
                         "k": k, "loop_k": k, "execution_ordinal": access_k,
                         "generation": generation, "slot": slot,
                         "start": begin, "end": end})
    # Different live buffers in one arena must not overlap (all versions included).
    arenas = {}
    for buffer in plan.buffers:
        arenas.setdefault(buffer.arena_id, []).append(
            (int(buffer.base_offset_bytes),
             int(buffer.base_offset_bytes) + int(buffer.allocation_bytes), buffer.root))
    for arena_id, ranges in arenas.items():
        ranges.sort()
        for (base_a, end_a, root_a), (base_b, _end_b, root_b) in zip(ranges, ranges[1:]):
            if base_b < end_a and root_a != root_b:
                raise PlanError("PLAN_ADDRESS_OVERLAP", "live buffers overlap in arena",
                                arena_id, "disjoint", [(root_a, base_a, end_a), (root_b, base_b)])
    return rows



SUPPORTED_INSTRUCTION_OPS = ("Copy", "Fill", "GmToUb", "Vld", "Vsts")


def _operand_index(access):
    try:
        return int(str(access.action_id).rsplit(":", 1)[1])
    except (ValueError, IndexError):
        raise PlanError("MATERIALIZATION_OPERAND_UNMAPPED", "access has no operand index",
                        access.access_id, "action_id:operand", access.action_id)


def _rebuild_eval(stmt, accesses, plan, env):
    indices = {}
    for access in accesses:
        idx = _operand_index(access)
        if idx in indices or idx >= len(stmt.args):
            raise PlanError("APPLICATION_COVERAGE_MISMATCH", "operand map is ambiguous/out of range",
                            getattr(stmt, "instruction", "?"), "unique operand indices", idx)
        indices[idx] = access
    new_args = list(stmt.args)
    for idx, access in indices.items():
        scoped = _access_env(access, plan, env, env["k"])
        new_args[idx] = _view(access, plan, scoped)
    return Evaluate(callable=stmt.callable, args=tuple(new_args))


def _rewrite_node(node, loop_stmt, blocks):
    from tilefoundry.ir.tir.stmts import ExecutionScope, MeshScope, While

    if node is loop_stmt:
        return Sequential(body=tuple(blocks))
    if isinstance(node, Sequential):
        return Sequential(body=tuple(_rewrite_node(c, loop_stmt, blocks) for c in node.body))
    if isinstance(node, (MeshScope, ExecutionScope)):
        return dataclasses.replace(node, body=_rewrite_node(node.body, loop_stmt, blocks))
    if isinstance(node, If):
        return dataclasses.replace(node, then_body=_rewrite_node(node.then_body, loop_stmt, blocks),
                                   else_body=_rewrite_node(node.else_body, loop_stmt, blocks))
    if isinstance(node, While):
        return dataclasses.replace(node, body=_rewrite_node(node.body, loop_stmt, blocks))
    return node


def materialize_instruction_plan(tir_module, plan: ResolvedPlan, *, sidecar_only=False):
    """R2 extension: materialize INSTRUCTION TIR (Copy/Fill/GmToUb/Vld/Vsts).

    Generic per-operand view rewrite: each access maps to a real operand index (from its
    action_id 'op:operand'); that operand's TensorView is replaced by a new view; other
    operands are kept. An unsupported op kind is refused PLAN_MATERIALIZATION_UNSUPPORTED_OP.
    """
    statements = index_statements_ledger(tir_module)
    groups = {}
    for access in plan.accesses:
        groups.setdefault(access.source_stmt, []).append(access)
    for path in groups:
        stmt = lookup_statement(statements, path)
        if not isinstance(stmt, Evaluate):
            raise PlanError("PLAN_MATERIALIZATION_UNSUPPORTED", "source_stmt is not an Evaluate",
                            path, "Evaluate", type(stmt).__name__)
        kind = type(stmt.callable).__name__
        if kind not in SUPPORTED_INSTRUCTION_OPS:
            raise PlanError("PLAN_MATERIALIZATION_UNSUPPORTED_OP",
                            "unsupported instruction op for R2 materialization", path,
                            SUPPORTED_INSTRUCTION_OPS, kind)
    loop_stmt = next((s for s in index_statements_ledger(tir_module).values() if isinstance(s, For)), None)
    if loop_stmt is None:
        raise PlanError("PLAN_MATERIALIZATION_UNSUPPORTED", "no loop to materialize", "tir", "For", None)
    start = _static_int(loop_stmt.start)
    stop = _static_int(loop_stmt.stop)
    step = _static_int(loop_stmt.step)
    if None in (start, stop, step):
        raise PlanError("PLAN_MATERIALIZATION_UNSUPPORTED", "dynamic loop is R4", "loop", "static ints", None)
    trip = loop_trip_count(plan.loop)

    blocks = []
    map_entries = []
    for k in range(trip):
        iteration = start + k * step
        env = {"i": iteration, "k": k, "slot": 0}
        for path, accesses in groups.items():
            original = lookup_statement(statements, path)
            rewritten = None
            for access in accesses:
                scoped = _access_env(access, plan, env, k)
                pass
            # all accesses for this path at this k
            rewritten = _rebuild_eval(original, accesses, plan, {"i": iteration, "k": k, "slot": 0})
            blocks.append(rewritten)
            for access in accesses:
                map_entries.append({
                    "access_id": access.access_id, "action_id": access.action_id,
                    "task_id": access.task_id, "mode": access.mode, "k": k, "loop_k": k,
                    "generation": int(access.execution_ordinal.evaluate({"i": iteration, "k": k})),
                    "old_stmt": path, "old_fingerprint": fingerprint(original),
                    "new_fingerprint": fingerprint(rewritten),
                    "new_stmt": "unrolled:k%d:%s" % (k, access.access_id),
                })
    new_body = _rewrite_node(tir_module.body, loop_stmt, blocks)
    new_body = Sequential(body=tuple(new_body.body) + (Evaluate(callable=Drain(), args=()),))
    original_digest = tir_digest(tir_module)
    if sidecar_only:
        return tir_module, MaterializationMap(entries=tuple(map_entries), allocs=(), controls=(),
                                              original_digest=original_digest,
                                              rewritten_digest=original_digest)
    rewritten = PrimFunction(name=tir_module.name, params=tir_module.params, body=new_body,
                             output_count=tir_module.output_count, target=tir_module.target)
    materialization = MaterializationMap(
        entries=tuple(map_entries), allocs=(),
        controls=tuple({"k": k, "kind": "guard"} for k in range(trip)),
        original_digest=original_digest, rewritten_digest=tir_digest(rewritten))
    return rewritten, materialization




def verify_typed_tir(module) -> dict:
    """Explicit typed structural check of a rewritten TIR; refuse by name on failure."""
    from tilefoundry.visitor_registry.verify import verify_prim_function
    try:
        verify_prim_function(module, module_fns={})
    except Exception as error:  # noqa: BLE001
        raise PlanError("PLAN_TYPED_VERIFY_FAILED", str(error)[:240],
                        "rewritten_tir", "valid typed PrimFunction", type(error).__name__)
    return {"valid": True}


def _rw_subst_expr(expr, mapping):
    from tilefoundry.ir.core import Var, Call
    if isinstance(expr, Var):
        return mapping.get(expr.name, expr)
    if isinstance(expr, Call):
        return dataclasses.replace(expr, args=tuple(_rw_subst_expr(a, mapping) for a in expr.args))
    return expr


def _rw_rewrite_body(node, path, groups, statements, plan, env, rename=None, fps=None):
    """One unrolled loop-body copy: KEEP every LetStmt binding, give each copy fresh Vars and
    substitute uses inside that copy. Only grouped Evaluate operands are replaced by new views."""
    from tilefoundry.ir.tir.stmts import ExecutionScope, MeshScope, While
    from tilefoundry.ir.core import Var

    rename = {} if rename is None else rename
    if isinstance(node, Sequential):
        local = dict(rename)
        out = []
        for index, child in enumerate(node.body):
            key = "/".join(path + ("s%d" % index,))
            if key in groups:
                stmt = _rebuild_eval(lookup_statement(statements, key) or statements[key], groups[key], plan, env)
                stmt = dataclasses.replace(stmt, args=tuple(_rw_subst_expr(a, local) for a in stmt.args))
                if fps is not None:
                    fps[key] = fingerprint(stmt)
                out.append(stmt)
            elif isinstance(child, LetStmt):
                fresh = Var(name="%s__k%d" % (child.var.name, env["k"]), type=child.var.type)
                out.append(LetStmt(var=fresh, value=_rw_subst_expr(child.value, local)))
                local[child.var.name] = fresh
            elif isinstance(child, Evaluate):
                out.append(dataclasses.replace(child, args=tuple(_rw_subst_expr(a, local) for a in child.args)))
            else:
                out.append(_rw_rewrite_body(child, path + ("s%d" % index,), groups, statements, plan, env, local, fps))
        return Sequential(body=tuple(out))
    if isinstance(node, (MeshScope, ExecutionScope)):
        return dataclasses.replace(node, body=_rw_rewrite_body(node.body, path + (type(node).__name__,), groups, statements, plan, env, dict(rename), fps))
    if isinstance(node, If):
        return dataclasses.replace(node, then_body=_rw_rewrite_body(node.then_body, path + ("then",), groups, statements, plan, env, dict(rename), fps),
                                   else_body=_rw_rewrite_body(node.else_body, path + ("else",), groups, statements, plan, env, dict(rename), fps))
    if isinstance(node, While):
        return dataclasses.replace(node, body=_rw_rewrite_body(node.body, path + ("while",), groups, statements, plan, env, dict(rename), fps))
    return node


def _rw_replace_loop(node, loop_stmt, blocks):
    from tilefoundry.ir.tir.stmts import ExecutionScope, MeshScope, While
    if node is loop_stmt:
        # Keep the For induction binding/scope; the body holds the static k copies. Executing
        # the outer For exactly once keeps the unrolled semantics finite and typed-legal.
        one = Constant(type=loop_stmt.stop.type, value=1)
        return dataclasses.replace(node, stop=one, body=Sequential(body=tuple(blocks)))
    if isinstance(node, Sequential):
        return Sequential(body=tuple(_rw_replace_loop(c, loop_stmt, blocks) for c in node.body))
    if isinstance(node, (MeshScope, ExecutionScope)):
        return dataclasses.replace(node, body=_rw_replace_loop(node.body, loop_stmt, blocks))
    if isinstance(node, If):
        return dataclasses.replace(node, then_body=_rw_replace_loop(node.then_body, loop_stmt, blocks),
                                   else_body=_rw_replace_loop(node.else_body, loop_stmt, blocks))
    if isinstance(node, While):
        return dataclasses.replace(node, body=_rw_replace_loop(node.body, loop_stmt, blocks))
    return node


def materialize_instruction_plan_v2(tir_module, plan: ResolvedPlan, *, sidecar_only=False):
    """Scope-preserving instruction-TIR materializer (task-86)."""
    statements = index_statements_ledger(tir_module)
    groups = {}
    for access in plan.accesses:
        groups.setdefault(access.source_stmt, []).append(access)
    for path in groups:
        stmt = lookup_statement(statements, path)
        if not isinstance(stmt, Evaluate):
            raise PlanError("PLAN_MATERIALIZATION_UNSUPPORTED", "source_stmt is not an Evaluate",
                            path, "Evaluate", type(stmt).__name__)
        kind = type(stmt.callable).__name__
        if kind not in SUPPORTED_INSTRUCTION_OPS:
            raise PlanError("PLAN_MATERIALIZATION_UNSUPPORTED_OP", "unsupported instruction op",
                            path, SUPPORTED_INSTRUCTION_OPS, kind)
    loop_stmt = next((s for s in statements.values() if isinstance(s, For)), None)
    if loop_stmt is None:
        raise PlanError("PLAN_MATERIALIZATION_UNSUPPORTED", "no loop to materialize", "tir", "For", None)
    start = _static_int(loop_stmt.start); stop = _static_int(loop_stmt.stop); step = _static_int(loop_stmt.step)
    if None in (start, stop, step):
        raise PlanError("PLAN_MATERIALIZATION_UNSUPPORTED", "dynamic loop is R4", "loop", "static ints", None)
    trip = loop_trip_count(plan.loop)
    loop_path = next((p for p, s in statements.items() if s is loop_stmt), None)
    # THE REWRITE PREFIX MUST SPEAK THE PLAN'S DIALECT.
    # 'statements' is the UNPREFIXED ledger index ("s0/MeshScope/s0"), but 'groups' is keyed by
    # PLAN paths, which carry the function name ("k/s0/MeshScope/s0/for:i/s2"). Building the
    # prefix from loop_path therefore produced "s0/MeshScope/s0/for:i/s2" against a group key of
    # "k/...", so the "key in groups" test never fired, no statement was ever rebuilt, 'fps'
    # stayed empty, and every map entry recorded the UNREWRITTEN fingerprint. Materialisation
    # then reported 20 entries and verify_plan_application rejected all 20 as
    # MATERIALIZATION_MAP_MISMATCH -- the map was honest, the program had not been rewritten.
    # THE PREFIX MUST SPEAK WHATEVER DIALECT 'groups' SPEAKS.
    #
    # 'statements' is the unprefixed ledger index ("s0/MeshScope/s0"), but 'groups' is keyed by
    # PLAN paths, and those are NOT always prefixed: the production chain writes
    # "k/s0/MeshScope/s0/for:i/s2" while the L/S/C fixtures write the same statement as
    # "s0/MeshScope/s0/for:i/s2". Hardcoding either spelling silently disables the rewrite for
    # the other -- the "key in groups" test simply never fires, 'fps' stays empty, every map
    # entry records the UNREWRITTEN fingerprint, and verification refuses with
    # MATERIALIZATION_MAP_MISMATCH on a materialization that did nothing.
    #
    # So: derive the prefix from an actual group key when one extends loop_path, and fall back
    # to the bare ledger spelling when none does. The prefix is a NAMING choice, not a fact
    # about the program, so it must adapt rather than assert.
    _base = tuple(loop_path.split("/")) + ("for:" + str(getattr(loop_stmt.induction_var, "name", "i")),)
    prefix = _base
    for _key in groups:
        _segments = tuple(_key.split("/"))
        _offset = len(_segments) - len(_base) - 1
        if _offset >= 0 and _segments[_offset:_offset + len(_base)] == _base:
            prefix = _segments[:-1]
            break
    blocks, map_entries = [], []
    for k in range(trip):
        iteration = start + k * step
        env = {"i": iteration, "k": k, "slot": 0}
        fps = {}
        blocks.append(_rw_rewrite_body(loop_stmt.body, prefix, groups, statements, plan, env, None, fps))
        for path, accesses in groups.items():
            original = lookup_statement(statements, path)
            for access in accesses:
                map_entries.append({
                    "access_id": access.access_id, "action_id": access.action_id,
                    "task_id": access.task_id, "mode": access.mode, "k": k, "loop_k": k,
                    "generation": int(access.execution_ordinal.evaluate({"i": iteration, "k": k})),
                    "old_stmt": path, "old_fingerprint": fingerprint(original),
                    "new_fingerprint": fps.get(path, fingerprint(_rebuild_eval(original, accesses, plan, env))),
                    "new_stmt": "unrolled:k%d:%s" % (k, access.access_id),
                })
    new_body = _rw_replace_loop(tir_module.body, loop_stmt, blocks)
    original_digest = tir_digest(tir_module)
    if sidecar_only:
        return tir_module, MaterializationMap(entries=tuple(map_entries), allocs=(), controls=(),
                                              original_digest=original_digest, rewritten_digest=original_digest)
    rewritten = PrimFunction(name=tir_module.name, params=tir_module.params, body=new_body,
                             output_count=tir_module.output_count, target=tir_module.target)
    verify_typed_tir(rewritten)
    materialization = MaterializationMap(
        entries=tuple(map_entries), allocs=(),
        controls=tuple({"k": k, "kind": "guard"} for k in range(trip)),
        original_digest=original_digest, rewritten_digest=tir_digest(rewritten))
    return rewritten, materialization


materialize_instruction_plan = materialize_instruction_plan_v2

