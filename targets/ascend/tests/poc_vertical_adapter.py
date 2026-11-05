"""POC_GLUE_NOT_PRODUCTION -- test-only adapter, NOT a production contract.

Projects L's accepted real-problem/1 into the shapes S/R expect WITHOUT changing the
production schema. Every projected field is recorded with projected=True and its source.
No value is invented: loop_domain is derived from the static TIR For and rejected if not a
contiguous arithmetic progression (POC_LOOP_DOMAIN_NOT_AFFINE); model quantities are marked
MODEL/NOMINAL with units.

SCHEMA-EVOLUTION ITEM (NOT DONE HERE): real-problem/1 lacks loop_domain(start/stop/step) and
per-task latency/ii/stage. Those belong in a future real-problem/2 or a dedicated adapter
contract; this glue must not be mistaken for it.
"""
from __future__ import annotations

POC_GLUE_NOT_PRODUCTION = True

from tilefoundry_ascend.plan.resolved_plan import V


class VerticalError(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        super().__init__("%s: %s" % (code, message))


def _walk_for(node):
    from tilefoundry.ir.tir.stmts import ExecutionScope, For, If, MeshScope, Sequential, While

    if node is None:
        return None
    if isinstance(node, (tuple, list)):
        for child in node:
            found = _walk_for(child)
            if found is not None:
                return found
        return None
    if isinstance(node, For):
        return node
    if isinstance(node, (Sequential, MeshScope, ExecutionScope)):
        return _walk_for(node.body)
    if isinstance(node, While):
        return _walk_for(node.body)
    if isinstance(node, If):
        return _walk_for(node.then_body) or _walk_for(node.else_body)
    return None


def _static_int(expr):
    from tilefoundry.ir.core import Constant

    if isinstance(expr, Constant):
        return int(expr.value)
    return None


def derive_loop_domain(tir_function, expansion_count):
    loop = _walk_for(getattr(tir_function, "body", None))
    if loop is None:
        raise VerticalError("POC_LOOP_DOMAIN_NOT_AFFINE", "no For loop to derive a domain from")
    start = _static_int(loop.start)
    stop = _static_int(loop.stop)
    step = _static_int(loop.step)
    if None in (start, stop, step) or step <= 0:
        raise VerticalError("POC_LOOP_DOMAIN_NOT_AFFINE", "loop bounds are not static ints")
    values = list(range(start, stop, step))
    if len(values) != expansion_count or any(
        values[k + 1] - values[k] != step for k in range(len(values) - 1)
    ):
        raise VerticalError("POC_LOOP_DOMAIN_NOT_AFFINE", "iteration set is not a contiguous AP")
    return {"start": start, "stop": stop, "step": step,
            "projected": True, "source": "TIR For start/stop/step (verified AP, count=%d)" % len(values)}


def project_problem(problem, tir_function, expansion_count, sources=None, obligation_access_ids=None):
    """Return (r_problem, projected_fields). All values derived or explicitly MODEL/NOMINAL."""
    loop_domain = derive_loop_domain(tir_function, expansion_count)
    tasks_in = list(getattr(problem, "tasks", []) or [])
    loop_id = next((t.get("loop", None) if isinstance(t, dict) else getattr(t, "loop", None)
                    for t in tasks_in), None) or "i"
    budget = 1
    if sources is not None:
        budget = max(list(sources.budgets.values()) or [1])
    r_tasks = []
    for task in tasks_in:
        tid = task.get("stable_id") if isinstance(task, dict) else task.stable_id
        action_ids = task.get("action_ids") if isinstance(task, dict) else task.action_ids
        r_tasks.append({
            "stable_id": tid, "action_ids": list(action_ids),
            "pipe": "vector", "resource": "vector", "source": tid, "loop": loop_id,
            "latency": 1, "issue_interval": 1, "stage": 0, "cost_model": "MODEL",
        })
    buffers = []
    dropped_buffers = []
    for buf in getattr(problem, "buffers", []) or []:
        get = buf.get if isinstance(buf, dict) else lambda k, b=buf: getattr(b, k, None)
        root = get("root")
        if not str(root).startswith("smem@"):
            dropped_buffers.append({"root": root, "reason": "not a synthetic SMEM allocation"})
            continue
        buffers.append({"root": root, "source_decl": root, "owner": loop_id,
                        "space": "ascend.ub", "dtype": "f32", "shape": [int(get("bytes_") or 0) // 4],
                        "bytes": int(get("bytes_") or 0), "alignment": 64,
                        "versions": 1, "fixed": True})
    action_owner = {}
    for task in tasks_in:
        tid = task.get("stable_id") if isinstance(task, dict) else task.stable_id
        ids = task.get("action_ids") if isinstance(task, dict) else task.action_ids
        for action_id in ids:
            action_owner[str(action_id)] = tid
    root_base = {}
    for access in getattr(problem, "accesses", []) or []:
        g = access.get if isinstance(access, dict) else lambda k, a=access: getattr(a, k, None)
        root_i = g("root")
        off_i = g("byte_offset")
        if str(root_i).startswith("smem@") and off_i is not None:
            root_base[root_i] = min(root_base.get(root_i, int(off_i)), int(off_i))
    accesses = []
    dropped_accesses = []
    obligation_ids = set(str(x) for x in (obligation_access_ids or ()))
    for index, access in enumerate(getattr(problem, "accesses", []) or []):
        get = access.get if isinstance(access, dict) else lambda k, a=access: getattr(a, k, None)
        root = get("root")
        if not str(root).startswith("smem@"):
            dropped_accesses.append({"task": get("task"), "reason": "non-SMEM (GM/register) access"})
            if str(get("task")) in obligation_ids:
                raise VerticalError("POC_ADAPTER_UNSUPPORTED_ACCESS",
                                    "obligation access %r would be filtered out" % (get("task"),))
            continue
        if get("byte_offset") is None:
            dropped_accesses.append({"task": get("task"), "reason": "unknown byte_offset"})
            if str(get("task")) in obligation_ids:
                raise VerticalError("POC_ADAPTER_UNSUPPORTED_ACCESS",
                                    "obligation access %r has unknown offset" % (get("task"),))
            continue
        accesses.append({
            "access_id": "%s#%d" % (get("task"), index),
            "task": action_owner.get(str(get("task")), get("task")), "action_id": get("task"),
            "source_stmt": str(get("instruction")).split("/", 1)[1] if "/" in str(get("instruction")) else get("instruction"),
            "root": root, "space": "ascend.ub",
            "mode": get("mode"),
            "relative_byte_offset": int(get("byte_offset")) - root_base.get(root, 0),
            "byte_footprint": int(get("byte_footprint")), "logical_index": V("i"),
        })
    r_problem = {
        "problem_digest": problem.problem_digest, "source_digest": problem.source_digest,
        "ledger_sha": problem.ledger_sha, "loop_id": loop_id, "loop_domain": loop_domain,
        "requested_num_stages": budget, "stage_depth": 1, "ii": 1, "latency": 1,
        "tasks": r_tasks, "buffers": buffers, "accesses": accesses,
    }
    projected = {
        "loop_domain": loop_domain,
        "requested_num_stages": {"value": budget, "unit": "stages", "cost_model": "MODEL", "projected": True},
        "stage_depth": {"value": 1, "cost_model": "MODEL", "projected": True},
        "ii": {"value": 1, "unit": "MODEL cycles", "cost_model": "MODEL", "projected": True},
        "task_latency_ii_stage": {"cost_model": "MODEL", "projected": True,
                                   "note": "real-problem/1 has no latency/ii/stage; nominal 1/1/0"},
        "task_pipe_resource": {"value": "vector", "cost_model": "MODEL", "projected": True},
        "access_source_stmt": {"source": "ProblemAccess.instruction (typed stmt path)", "projected": True},
        "access_relative_offset": {"source": "ProblemAccess.byte_offset (typed window)", "projected": True},
        "access_logical_index": {"source": "PExpr V('i')", "projected": True},
        "buffers_filtered": {"note": "only synthetic SMEM roots with known byte_offset are planned",
                             "projected": True, "count": len(dropped_buffers),
                             "dropped": dropped_buffers[:8]},
        "accesses_filtered": {"count": len(dropped_accesses), "dropped": dropped_accesses[:8],
                              "projected": True},
    }
    return r_problem, projected


def nominal_solve_result(core_result, r_problem):
    """Adapter: expose the REAL core assignment as R's SAT_POSTCHECKED view, NOMINAL only.

    This is NOT a physical proof and NOT an S3 source postcheck (S3 is unimplemented).
    Returns a SimpleNamespace with status='SAT_POSTCHECKED', sat_scope='NOMINAL_MODEL_ONLY'.
    """
    import types

    assignment = getattr(core_result, "assignment", {}) or {}
    tasks = assignment.get("tasks", {}) if isinstance(assignment, dict) else {}
    schedule = {}
    for order, tid in enumerate(assignment.get("order", list(tasks)) if isinstance(assignment, dict) else list(tasks)):
        entry = tasks.get(tid, {}) if isinstance(tasks, dict) else {}
        start = (entry.get("start_time") or {}).get("value", 0) if isinstance(entry, dict) else 0
        schedule[tid] = {"stage": 0, "order": order, "start_time": int(start), "iteration_shift": 0,
                         "cost_model": "MODEL"}
    # NOT SAT_POSTCHECKED: S3 source postcheck is unimplemented. This is a NOMINAL_SAT view
    # for the R adapter only; the S result itself is state=SAT / sat_scope=NOMINAL_MODEL_ONLY.
    return types.SimpleNamespace(status="NOMINAL_SAT", schedule=schedule,
                                 sat_scope="NOMINAL_MODEL_ONLY",
                                 postcheck="SOLVER_POSTCHECK_NOT_IMPLEMENTED",
                                 core_source_hash=getattr(core_result, "core_source_hash", ""))



def build_c_plan_view(plan, problem, loop_domain):
    """POC_GLUE: action-id-level plan view + real GM buffers/accesses for C coverage only.

    R materialization uses the SMEM plan; this view is NOT used by R. It exposes each action
    as its own task_id (so C's edge producer/consumer action ids resolve) and adds REAL GM
    buffers/accesses (x,z,out) with symbolic per-iteration offsets derived from the loop.
    """
    import dataclasses
    from tilefoundry_ascend.plan.resolved_plan import AccessPlan, BufferPlan, TaskPlan, V, Mul, C

    tasks = []
    for task in plan.tasks:
        for action_id in task.action_ids:
            tasks.append(dataclasses.replace(task, task_id=str(action_id), action_ids=(str(action_id),)))
    buffers = list(plan.buffers)
    accesses = []
    for access in plan.accesses:
        accesses.append(dataclasses.replace(access, task_id=str(access.action_id), action_id=str(access.action_id)))
    # real GM buffers from L's problem
    for buf in getattr(problem, "buffers", []) or []:
        g = buf.get if isinstance(buf, dict) else lambda k, b=buf: getattr(b, k, None)
        root = g("root")
        if str(root).startswith("smem@"):
            continue
        buffers.append(BufferPlan(
            source_decl=str(root), root=str(root), owner="row_pipeline",
            space=str(g("space") or "GMEM"), dtype=str(g("dtype") or "f32"),
            shape=tuple(g("shape") or (int(g("bytes_") or 0) // 4,)),
            single_version_bytes=int(g("bytes_") or 0), alignment=64,
            slot_count=1, slot_source="fixed"))
    for index, access in enumerate(getattr(problem, "accesses", []) or []):
        g = access.get if isinstance(access, dict) else lambda k, a=access: getattr(a, k, None)
        root = g("root")
        if str(root).startswith("smem@"):
            continue
        stride = 256
        if g("byte_offset") is None:
            offset = Mul(V("i"), C(stride))
            fmt = "symbolic"
        else:
            offset = C(int(g("byte_offset")))
            fmt = "literal"
        accesses.append(AccessPlan(
            access_id=str(g("task")) + "#gm%d" % index, action_id=str(g("task")),
            task_id=str(g("task")), source_stmt=str(g("instruction")).split("/", 1)[1]
            if "/" in str(g("instruction")) else str(g("instruction")),
            buffer_root=str(root), logical_index=V("i"), execution_ordinal=V("k"),
            slot=V("slot"), generation=V("k"), relative_byte_offset=offset,
            guard=None, mode=str(g("mode")), byte_footprint=int(g("byte_footprint") or stride)))
    return dataclasses.replace(plan, tasks=tuple(tasks), buffers=tuple(buffers), accesses=tuple(accesses))

