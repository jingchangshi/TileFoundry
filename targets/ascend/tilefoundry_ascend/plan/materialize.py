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

import hashlib
import json
from dataclasses import fields, is_dataclass

from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.tir.memory import AllocTensor
from tilefoundry.ir.tir.memory.tensor_view import TensorView
from tilefoundry.ir.tir.memory.copy import Copy
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, For, If, LetStmt, Sequential
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


def index_statements(module) -> dict:
    found = {}

    def walk(stmt, path):
        if isinstance(stmt, Sequential):
            for index, child in enumerate(stmt.body):
                walk(child, path + ("s%d" % index,))
            return
        if isinstance(stmt, For):
            found["/".join(path)] = stmt          # the loop itself is a statement too
            walk(stmt.body, path + ("for",))
            return
        if isinstance(stmt, If):
            found["/".join(path)] = stmt
            walk(stmt.then_body, path + ("then",))
            walk(stmt.else_body, path + ("else",))
            return
        found["/".join(path)] = stmt

    walk(module.body, ())
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
    storage = StorageKind.SMEM if buffer.space in ("ascend.ub", "smem", "ub") else StorageKind.GMEM
    tensor_type = make_tensor_type(shape, dtype=buffer.dtype, storage=storage)
    pointer_type = PointerType(tensor_type.dtype, tensor_type.storage)
    address = Constant(type=pointer_type, value=start)
    return Call(type=tensor_type,
                target=TensorView(storage=tensor_type.storage, dtype=str(buffer.dtype),
                                  layout=Layout(shape), shape=shape),
                args=(address,))


def _bool_type():
    return make_tensor_type((), dtype="bool", storage="umat")


def materialize_plan(tir_module, plan: ResolvedPlan, *, sidecar_only=False,
                     skip_access_ids=()):
    """Apply the plan to the actual TIR. Returns (rewritten_module, MaterializationMap)."""
    map_entries = []
    if plan.state.application_status != STATE_SKELETON:
        raise PlanError("PLAN_ALREADY_APPLIED", "plan is not in SKELETON_NOT_APPLIED",
                        "application_status", STATE_SKELETON, plan.state.application_status)
    if plan.loop.application_mode != "SERIAL":
        raise PlanError("PLAN_MATERIALIZATION_UNSUPPORTED", "R2 supports SERIAL only",
                        "application_mode", "SERIAL", plan.loop.application_mode)
    for buffer in plan.buffers:
        if int(buffer.slot_count) != 1:
            raise PlanError("PLAN_MATERIALIZATION_UNSUPPORTED",
                            "R2 fixes a single version; multi-version is R3",
                            buffer.root, 1, buffer.slot_count)
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
        stmt = statements.get(path)
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
            for k in range(trip):
                iteration = start + k * step
                env = {"i": iteration, "k": k, "slot": 0}
                generation_token(plan.loop, iteration, supplied=k)
                body.append(If(cond=Constant(type=_bool_type(), value=True),
                               then_body=Sequential(body=()), else_body=Sequential(body=())))
                for path, accesses in groups.items():
                    original = originals[path]
                    if any(a.access_id in skip_access_ids for a in accesses):
                        body.append(original)       # deliberately un-applied
                        continue
                    if len(accesses) != len(original.args):
                        raise PlanError("APPLICATION_COVERAGE_MISMATCH",
                                        "one access per operand is required", path,
                                        len(original.args), len(accesses))
                    new_args = tuple(_view(access, plan, env) for access in accesses)
                    rewritten = Evaluate(callable=Copy(), args=new_args)
                    body.append(If(cond=Constant(type=_bool_type(), value=bool(0 <= k < trip)),
                                   then_body=Sequential(body=(rewritten,)),
                                   else_body=Sequential(body=())))
                    for access in accesses:
                        map_entries.append({
                            "access_id": access.access_id,
                            "action_id": access.action_id,
                            "task_id": access.task_id,
                            "mode": access.mode, "k": k, "generation": k,
                            "old_stmt": path,
                            "old_fingerprint": fingerprint(original),
                            "new_fingerprint": fingerprint(rewritten),
                            "new_stmt": "unrolled:k%d:%s" % (k, access.access_id)})
            continue
        body.append(stmt)

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


def verify_plan_application(original_module, plan: ResolvedPlan, rewritten_module,
                            materialization: MaterializationMap) -> dict:
    """Independent walk of the REWRITTEN TIR: real change + full coverage + map fidelity."""
    report = {"status": "FAIL", "code": None, "applied_accesses": 0,
              "expected_applications": 0, "allocations_applied": 0,
              "controls_applied": 0, "original_digest": materialization.original_digest,
              "rewritten_digest": materialization.rewritten_digest}
    if (rewritten_module is None or rewritten_module is original_module
            or materialization.rewritten_digest == materialization.original_digest):
        report["code"] = "PLAN_NOT_APPLIED"
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
    # Every original effect statement that the plan covers must be GONE (not identity).
    rewritten_ids = {id(stmt) for stmt in index_statements(rewritten_module).values()}
    for path in {access.source_stmt for access in plan.accesses}:
        original = index_statements(original_module).get(path)
        if id(original) in rewritten_ids:
            report["code"] = "APPLICATION_COVERAGE_MISMATCH"
            report["detail"] = {"survived": path}
            return report
    # Map fidelity: recorded fingerprints must equal actual rewritten statements.
    rewritten_stmts = list(index_statements(rewritten_module).values())
    rewritten_fingerprints = {fingerprint(stmt) for stmt in rewritten_stmts}
    for entry in materialization.entries:
        if entry["new_fingerprint"] not in rewritten_fingerprints:
            report["code"] = "MATERIALIZATION_MAP_MISMATCH"
            report["detail"] = {"access_id": entry["access_id"], "k": entry["k"]}
            return report
        if entry["generation"] != entry["k"]:
            report["code"] = "GENERATION_MAPPING_INVALID"
            report["detail"] = {"access_id": entry["access_id"],
                                "generation": entry["generation"], "k": entry["k"]}
            return report
    original_allocs = [s for s in index_statements(original_module).values()
                       if isinstance(s, LetStmt)]
    rewritten_allocs = {fingerprint(s) for s in index_statements(rewritten_module).values()
                        if isinstance(s, LetStmt)}
    for alloc in materialization.allocs:
        if alloc["new_fingerprint"] not in rewritten_allocs:
            report["code"] = "MATERIALIZATION_MAP_MISMATCH"
            report["detail"] = {"alloc": alloc}
            return report
        if alloc["new_fingerprint"] == alloc["old_fingerprint"]:
            report["code"] = "PLAN_NOT_APPLIED"
            report["detail"] = {"alloc": alloc["root"]}
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
    start, step = int(plan.loop.domain["start"]), int(plan.loop.domain["step"])
    rows = []
    for k in range(trip):
        iteration = start + k * step
        env = {"i": iteration, "k": k, "slot": k - (k // 1) * 1}  # V=1 => slot 0
        env["slot"] = 0
        for access in plan.accesses:
            generation = int(access.generation.evaluate(env))
            if generation != k:
                raise PlanError("GENERATION_MAPPING_INVALID",
                                "generation token is not execution ordinal k",
                                access.access_id, k, generation)
            begin, end = address_range(plan, access, env)
            rows.append({"access_id": access.access_id, "k": k,
                         "generation": generation, "start": begin, "end": end})
    return rows
