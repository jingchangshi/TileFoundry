"""solver-request/1 / solver-result/1 bridge (S0-S2).

Approved design: artifacts/run02/review/architect_s_solver_bridge_module_design_v1.md.
This module is a BRIDGE, not a solver. It:

  * accepts a real L Problem through a read-only duck-typed view (never rebuilding
    the business program from a test CaseConfig);
  * lowers it to the arguments of the ONE shared physical core
    (tilefoundry.vendor.tilelang_z3.adapters.tf_loop_adapter -> core.py), with a dense
    task/buffer index map and a ConstraintTrace for every required source constraint;
  * classifies the core result as SAT / UNSAT / UNKNOWN / FALLBACK without ever
    promoting fallback/unknown to SAT.

It imports no z3 directly and contains no second solver: the only call site is the
shared adapter. Real-POC coverage is blocked by L5; this is the pure bridge.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Optional

from tilefoundry.vendor.tilelang_z3 import adapters as shared_adapters

SCHEMA_REQUEST = "solver-request/1"
SCHEMA_RESULT = "solver-result/1"
PROBLEM_SCHEMA = "real-problem/1"
CORE_SOURCE_HASH = "tilelang_z3/core.py"

SAT, UNSAT, UNKNOWN, FALLBACK = "SAT", "UNSAT", "UNKNOWN", "FALLBACK"

#: pipe/resource names used by the Problem -> the core's resource bitmask.
RESOURCE_BITS = {"mte1": 1, "mte2": 2, "mte3": 4, "cube": 8,
                 "vector": 16, "fixpipe": 32, "scalar": 64}


class SolverError(RuntimeError):
    def __init__(self, code, message, source_id="", expected=None, actual=None, phase="bridge"):
        self.code = code
        self.phase = phase
        self.source_id = source_id
        self.expected = expected
        self.actual = actual
        super().__init__("%s [%s] %s (source=%r expected=%r actual=%r)"
                         % (code, phase, message, source_id, expected, actual))

    def as_dict(self):
        return {"code": self.code, "phase": self.phase, "source_id": self.source_id,
                "expected": self.expected, "actual": self.actual}


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _canonical(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                      default=str).encode()).hexdigest()


@dataclass(frozen=True)
class ConstraintTrace:
    kind: str                 # data_dep | resource_dep | owner_exclusion | pipe_order | stage_order | memory_group | fixed_version
    source_id: str
    core_field: str
    core_index: object
    status: str               # CONSUMED | UNSUPPORTED
    note: str = ""

    def as_dict(self):
        return self.__dict__


@dataclass(frozen=True)
class SolveRequest:
    schema: str
    problem_digest: str
    source_digest: str
    ledger_digest: str
    loop_source_id: str
    core_source_hash: str
    model_id: str
    model_version: str
    solver_package_version: str
    seed: int
    limits: dict
    units: dict
    task_index_map: dict
    index_task_map: dict
    buffer_index_map: dict
    index_buffer_map: dict
    constraint_trace: tuple
    core_args: dict
    unsupported: tuple
    unknown: tuple
    request_digest: str = ""

    @property
    def constraint_source_map(self):
        """Design-table alias: source constraint ID -> core field/index (the trace)."""
        return self.constraint_trace

    def as_dict(self):
        out = dict(self.__dict__)
        out["constraint_trace"] = [entry.as_dict() if isinstance(entry, ConstraintTrace) else entry
                                  for entry in self.constraint_trace]
        return out


@dataclass(frozen=True)
class SolveResult:
    schema: str
    problem_digest: str
    request_digest: str
    core_source_hash: str
    state: str
    reason: str
    raw_core_status: str
    per_ii_checks: tuple
    search_scope: str
    assignment: dict
    fallback_candidate: object
    postcheck: object
    capacity_scope: str
    result_digest: str = ""

    def as_dict(self):
        return dict(self.__dict__)


# ------------------------------------------------------------ Problem -> request
def _task_id(task):
    return str(_get(task, "stable_id") or _get(task, "task_id"))


def _is_loop_task(task, loop_id):
    loop = _get(task, "loop", None)
    return loop in (None, "", loop_id, "main_compute", "row_pipeline")


def lower_problem_to_request(problem, model_facts, solve_options) -> SolveRequest:
    """S1/S2: a real Problem -> the shared core's normalized argument set.

    Every required constraint gets a trace entry; a constraint the core cannot
    express is named UNSUPPORTED, never silently dropped or defaulted to 0/1.
    """
    model_facts = model_facts or {}
    solve_options = solve_options or {}
    schema = _get(problem, "schema", None)
    if str(schema) != PROBLEM_SCHEMA:
        raise SolverError("SOLVER_PROBLEM_SCHEMA_UNSUPPORTED",
                          "only canonical real-problem/1 input is accepted; legacy shapes are retired",
                          "schema", PROBLEM_SCHEMA, schema, phase="lower")
    problem_digest = str(_get(problem, "problem_digest") or "")
    source_digest = str(_get(problem, "source_digest") or "")
    ledger_digest = str(_get(problem, "ledger_sha") or "")
    loop_id = _get(problem, "loop_id", None)
    if not problem_digest or not source_digest or not ledger_digest or loop_id is None:
        raise SolverError("SOLVER_PROBLEM_SCHEMA_UNSUPPORTED",
                          "canonical problem must carry problem_digest/source_digest/ledger_sha/loop_id",
                          "problem", "real-problem/1 stable fields",
                          [problem_digest, source_digest, ledger_digest, loop_id], phase="lower")
    loop_id = str(loop_id)

    tasks_in = list(_get(problem, "tasks", []) or [])
    if not tasks_in:
        raise SolverError("SOLVER_CONSTRAINT_COVERAGE_MISSING", "problem has no tasks",
                          "tasks", "nonempty", 0, phase="lower")
    loop_tasks = [task for task in tasks_in if _is_loop_task(task, loop_id)]
    outside = [task for task in tasks_in if task not in loop_tasks]
    unsupported = []
    unknown = []
    if outside:
        # Do not stuff an outside-loop task into per-iteration arrays as if it ran
        # every iteration. It is a live-in/partition obligation the core cannot state here.
        raise SolverError("UNSUPPORTED_CONSTRAINT", "outside-loop task cannot be expressed",
                          _task_id(outside[0]), "loop task", _get(outside[0], "loop"), phase="lower")

    loop_tasks = sorted(loop_tasks, key=_task_id)
    task_index_map = {_task_id(task): index for index, task in enumerate(loop_tasks)}
    index_task_map = {index: _task_id(task) for task, index in
                      ((task, task_index_map[_task_id(task)]) for task in loop_tasks)}

    buffers_in = sorted(list(_get(problem, "buffers", []) or []), key=lambda b: str(_get(b, "root")))
    buffer_index_map = {str(_get(buffer, "root")): index
                        for index, buffer in enumerate(buffers_in)}
    index_buffer_map = {index: root for root, index in buffer_index_map.items()}

    trace = []

    def trace_add(kind, source_id, core_field, core_index, note=""):
        trace.append(ConstraintTrace(kind, source_id, core_field, core_index, "CONSUMED", note))

    latencies, iis, resource_flags, stage_order = [], [], [], []
    for task in loop_tasks:
        tid = _task_id(task)
        resource = str(_get(task, "resource") or _get(task, "pipe") or "").lower()
        if resource not in RESOURCE_BITS:
            raise SolverError("UNSUPPORTED_CONSTRAINT", "unknown resource pipe",
                              tid, list(RESOURCE_BITS), resource, phase="lower")
        latencies.append(int(_get(task, "latency")))
        iis.append(int(_get(task, "issue_interval")))
        resource_flags.append(int(_get(task, "resource_flags", RESOURCE_BITS[resource])))
        stage_order.append(int(_get(task, "stage", 0)))
        trace_add("task", tid, "latencies/iis/resource_flags",
                  task_index_map[tid], "task row")

    # Dependencies: normal distance is the LOGICAL iteration distance. A negative
    # distance is only the core's buffer-version encoding (-buffer_index-1) and is
    # emitted only for an explicit reuse relation that names its buffer.
    data_deps = []
    deps = list(_get(problem, "dependencies", []) or [])
    for position, dep in enumerate(deps):
        producer = str(_get(dep, "producer"))
        consumer = str(_get(dep, "consumer"))
        if producer not in task_index_map or consumer not in task_index_map:
            raise SolverError("SOLVER_INDEX_MAP_MISMATCH", "dependency names an unknown task",
                              "%s->%s" % (producer, consumer), list(task_index_map), [producer, consumer],
                              phase="lower")
        distance = int(_get(dep, "distance", 0))
        kind = str(_get(dep, "kind", "RAW"))
        buffer_root = _get(dep, "buffer")
        if distance < 0 and not (kind.lower() in ("reuse", "buffer_reuse") and buffer_root):
            raise SolverError("SOLVER_INDEX_MAP_MISMATCH",
                              "a negative distance is only a buffer-version encoding",
                              "dep[%d]" % position, "explicit reuse + buffer", distance,
                              phase="lower")
        if kind.lower() in ("reuse", "buffer_reuse") and buffer_root:
            if str(buffer_root) not in buffer_index_map:
                raise SolverError("SOLVER_INDEX_MAP_MISMATCH", "reuse names unknown buffer",
                                  str(buffer_root), list(buffer_index_map), buffer_root, phase="lower")
            encoded = -buffer_index_map[str(buffer_root)] - 1
            trace_add("data_dep", str(_get(dep, "source_id", "dep%d" % position)),
                      "data_deps.distance", (producer, consumer), "buffer version encoding")
            distance = encoded
        else:
            trace_add("data_dep", str(_get(dep, "source_id", "dep%d" % position)),
                      "data_deps.distance", (producer, consumer), "logical distance")
        data_deps.append((task_index_map[producer], task_index_map[consumer], distance,
                          int(_get(dep, "latency", latencies[task_index_map[producer]]))))

    def _pairs(field_name, source_field):
        pairs = []
        for index, pair in enumerate(list(_get(problem, source_field, []) or [])):
            producer, consumer = str(pair[0]), str(pair[1])
            for name in (producer, consumer):
                if name not in task_index_map:
                    raise SolverError("SOLVER_INDEX_MAP_MISMATCH", "pair names unknown task",
                                      name, list(task_index_map), name, phase="lower")
            pairs.append((task_index_map[producer], task_index_map[consumer]))
            trace_add(field_name, "%s[%d]" % (source_field, index), field_name,
                      (producer, consumer))
        return pairs

    resource_deps = _pairs("resource_dep", "resource_deps")
    owner_deps = []
    for index, item in enumerate(list(_get(problem, "owner_exclusion_deps", []) or [])):
        producer, consumer = str(item[0]), str(item[1])
        if producer not in task_index_map or consumer not in task_index_map:
            raise SolverError("SOLVER_INDEX_MAP_MISMATCH", "owner exclusion names unknown task",
                              str(item), list(task_index_map), item, phase="lower")
        owner_deps.append((task_index_map[producer], task_index_map[consumer],
                           int(item[2]) if len(item) > 2 else 1,
                           int(item[3]) if len(item) > 3 else 1))
        trace_add("owner_exclusion", "owner[%d]" % index, "owner_exclusion_deps", item)
    pipe_order_deps = _pairs("pipe_order", "pipe_order_deps")
    stage_order_deps = _pairs("stage_order", "stage_order_deps")

    buffer_sizes = []
    fixed_versions = []
    for buffer in buffers_in:
        root = str(_get(buffer, "root"))
        size = _get(buffer, "bytes", _get(buffer, "size_bytes"))
        if size is None or int(size) <= 0:
            raise SolverError("DEFAULTED_UNKNOWN_VALUE", "buffer size must be explicit",
                              root, ">0 bytes", size, phase="lower")
        buffer_sizes.append(int(size))
        versions = _get(buffer, "versions")
        fixed = bool(_get(buffer, "fixed", False))
        if fixed:
            if versions is None or int(versions) <= 0:
                raise SolverError("FIXED_VERSION_MISMATCH", "fixed buffer needs a version count",
                                  root, ">0", versions, phase="lower")
            fixed_versions.append(int(versions))
        else:
            fixed_versions.append(None)
        trace_add("fixed_version", root, "fixed_buffer_versions",
                  buffer_index_map[root], "fixed" if fixed else "auto")

    # Memory capacity is MODEL bytes. UNKNOWN usable capacity must never be filled
    # with nominal or 0; the bridge refuses instead of inventing it.
    memory_groups = []
    facts_groups = list(model_facts.get("memory_groups", []) or [])
    if not facts_groups and buffers_in:
        facts_groups = [{"id": "group0", "capacity_bytes": None,
                         "buffers": [str(_get(b, "root")) for b in buffers_in]}]
    for group in facts_groups:
        capacity = group.get("capacity_bytes")
        members = [str(root) for root in group.get("buffers", [])]
        if capacity is None or str(capacity).upper() == "UNKNOWN":
            raise SolverError("DEFAULTED_UNKNOWN_VALUE", "usable capacity is UNKNOWN",
                              str(group.get("id")), "explicit MODEL capacity bytes", capacity,
                              phase="lower")
        for root in members:
            if root not in buffer_index_map:
                raise SolverError("SOLVER_INDEX_MAP_MISMATCH", "memory group names unknown buffer",
                                  root, list(buffer_index_map), root, phase="lower")
        memory_groups.append([int(capacity)] + [buffer_index_map[root] for root in members])
        trace_add("memory_group", str(group.get("id")), "memory_groups", group.get("id"))

    num_stages = solve_options.get("requested_num_stages",
                                   _get(problem, "requested_num_stages", 1))
    if num_stages is None or int(num_stages) < 1:
        raise SolverError("DEFAULTED_UNKNOWN_VALUE", "num_stages budget must be explicit",
                          "requested_num_stages", ">=1", num_stages, phase="lower")
    enable_offset = bool(solve_options.get("enable_offset",
                                           _get(problem, "enable_offset", False)))
    seed = int(solve_options.get("seed", 42))
    core_args = {
        "num_stages": int(num_stages),
        "latencies": latencies, "iis": iis, "resource_flags": resource_flags,
        "data_deps": data_deps, "resource_deps": resource_deps,
        "owner_exclusion_deps": owner_deps or None,
        "pipe_order_deps": pipe_order_deps or None,
        "buffer_sizes": buffer_sizes, "memory_groups": memory_groups,
        "stage_order_deps": stage_order_deps or [],
        "recalculate_buffer_versions": True,
        "enable_offset": enable_offset,
        "manual_stages": None, "manual_schedule": False,
        "seed": seed, "verbose": False,
        "fixed_buffer_versions": fixed_versions,
        "return_status": True,
    }
    core_hash = shared_adapters.shared_loop_core_sha256()
    units = {"latency": "MODEL cycles", "ii": "MODEL cycles",
             "buffer_sizes": "bytes", "memory_groups": "bytes",
             "distance": "logical iterations", "start_time": "MODEL cycles"}
    request = SolveRequest(
        schema=SCHEMA_REQUEST, problem_digest=problem_digest, source_digest=source_digest,
        ledger_digest=ledger_digest, loop_source_id=loop_id, core_source_hash=core_hash,
        model_id=str(model_facts.get("model_id", "ascend-pipe-model")),
        model_version=str(model_facts.get("model_version", "v1")),
        solver_package_version=str(getattr(shared_adapters, "__name__", "vendor")),
        seed=seed, limits=dict(solve_options.get("limits", {"timeout_ms": None}) or {}),
        units=units, task_index_map=task_index_map, index_task_map=index_task_map,
        buffer_index_map=buffer_index_map, index_buffer_map=index_buffer_map,
        constraint_trace=tuple(trace), core_args=core_args,
        unsupported=tuple(unsupported), unknown=tuple(unknown))
    digest = _canonical({key: value for key, value in request.as_dict().items()
                         if key != "request_digest"})
    return SolveRequest(**{**request.__dict__, "request_digest": digest})


# ---------------------------------------------------------- coverage / identity
def verify_request_coverage(problem, request: SolveRequest) -> dict:
    """Every required source constraint must be CONSUMED exactly once; indexes must
    be a bijection; fixed versions must agree with the Problem."""
    problems = []
    trace = list(request.constraint_trace)
    consumed = {}
    for entry in trace:
        item = entry.as_dict() if isinstance(entry, ConstraintTrace) else dict(entry)
        if item["status"] != "CONSUMED":
            problems.append(("SOLVER_CONSTRAINT_COVERAGE_MISSING", item["source_id"],
                             "UNSUPPORTED", item["status"]))
        key = (item["kind"], item["source_id"])
        consumed[key] = consumed.get(key, 0) + 1

    required = []
    for index, dep in enumerate(list(_get(problem, "dependencies", []) or [])):
        required.append(("data_dep", str(_get(dep, "source_id", "dep%d" % index))))
    for index, _ in enumerate(list(_get(problem, "resource_deps", []) or [])):
        required.append(("resource_dep", "resource_deps[%d]" % index))
    for index, _ in enumerate(list(_get(problem, "pipe_order_deps", []) or [])):
        required.append(("pipe_order", "pipe_order_deps[%d]" % index))
    for index, _ in enumerate(list(_get(problem, "stage_order_deps", []) or [])):
        required.append(("stage_order", "stage_order_deps[%d]" % index))
    for buffer in list(_get(problem, "buffers", []) or []):
        required.append(("fixed_version", str(_get(buffer, "root"))))
    missing = [item for item in required if consumed.get(item, 0) == 0]
    if missing:
        raise SolverError("SOLVER_CONSTRAINT_COVERAGE_MISSING",
                          "required source constraint has no CONSUMED trace",
                          str(missing[0]), "one CONSUMED entry", consumed, phase="coverage")
    for item in required:
        if consumed[item] > 1:
            raise SolverError("SOLVER_CONSTRAINT_COVERAGE_MISSING",
                              "a required constraint was consumed more than once",
                              str(item), 1, consumed[item], phase="coverage")

    task_ids = [_task_id(t) for t in list(_get(problem, "tasks", []) or [])
                if _is_loop_task(t, request.loop_source_id)]
    if set(task_ids) != set(request.task_index_map):
        raise SolverError("SOLVER_INDEX_MAP_MISMATCH", "task index map is not a bijection",
                          "task_index_map", sorted(task_ids), sorted(request.task_index_map),
                          phase="coverage")
    if sorted(request.task_index_map.values()) != list(range(len(task_ids))):
        raise SolverError("SOLVER_INDEX_MAP_MISMATCH", "task dense indices are not 0..n-1",
                          "task_index_map", list(range(len(task_ids))),
                          sorted(request.task_index_map.values()), phase="coverage")
    for tid, index in request.task_index_map.items():
        if request.index_task_map.get(index) != tid:
            raise SolverError("SOLVER_INDEX_MAP_MISMATCH", "task reverse map disagrees",
                              tid, tid, request.index_task_map.get(index), phase="coverage")
    roots = [str(_get(b, "root")) for b in list(_get(problem, "buffers", []) or [])]
    if set(roots) != set(request.buffer_index_map):
        raise SolverError("SOLVER_INDEX_MAP_MISMATCH", "buffer index map is not a bijection",
                          "buffer_index_map", sorted(roots), sorted(request.buffer_index_map),
                          phase="coverage")
    for root, index in request.buffer_index_map.items():
        if request.index_buffer_map.get(index) != root:
            raise SolverError("SOLVER_INDEX_MAP_MISMATCH", "buffer reverse map disagrees",
                              root, root, request.index_buffer_map.get(index), phase="coverage")

    expected_fixed = [None] * len(request.buffer_index_map)
    for buffer in list(_get(problem, "buffers", []) or []):
        root = str(_get(buffer, "root"))
        if root not in request.buffer_index_map:
            raise SolverError("SOLVER_INDEX_MAP_MISMATCH", "buffer missing from request map",
                              root, sorted(request.buffer_index_map), root, phase="coverage")
        index = request.buffer_index_map[root]
        expected_fixed[index] = int(_get(buffer, "versions")) if _get(buffer, "fixed", False) else None
    if expected_fixed != request.core_args["fixed_buffer_versions"]:
        raise SolverError("FIXED_VERSION_MISMATCH", "fixed buffer versions changed in the request",
                          "fixed_buffer_versions", expected_fixed,
                          request.core_args["fixed_buffer_versions"], phase="coverage")
    return {"consumed": sum(consumed.values()), "required": len(required), "complete": True}


def invoke_shared_core(request: SolveRequest) -> dict:
    """The ONLY core invocation: the TF loop adapter over the one physical core."""
    actual_hash = shared_adapters.shared_loop_core_sha256()
    if request.core_source_hash != actual_hash:
        raise SolverError("SOLVER_CORE_IDENTITY_MISMATCH",
                          "request was not built for the current shared core",
                          "core_source_hash", request.core_source_hash, actual_hash,
                          phase="invoke")
    raw = shared_adapters.tf_loop_adapter(dict(request.core_args), return_status=True)
    return {"raw": raw, "core_source_hash": actual_hash}


def classify_core_result(request: SolveRequest, raw, claimed_state=None) -> SolveResult:
    """Map raw core output to SAT/UNSAT/UNKNOWN/FALLBACK without promotion.

    Core currently returns only sat/fallback at the top level; per-II check_results
    are preserved. A real UNKNOWN needs timeout/unsupported facts (S4); a real UNSAT
    needs an exhausted finite search domain (S4). We never fabricate those.
    """
    status = str(raw.get("status", "unknown"))
    per_ii = tuple(raw.get("check_results", []) or [])
    if status == "sat":
        state = SAT
        reason = "core returned a sat schedule"
        assignment = _assignment(request, raw)
        fallback_candidate = None
    elif status == "fallback":
        state = FALLBACK
        reason = str(raw.get("fallback_reason") or "core serial fallback")
        assignment = {}
        fallback_candidate = {key: raw.get(key) for key in
                              ("start_times", "buffer_versions", "minimal_II", "check_results")}
    elif status in ("unsat", "unknown"):
        # UNSAT only with a declared exhausted finite domain; otherwise UNKNOWN.
        complete = bool(raw.get("search_scope_complete", False))
        if status == "unsat" and complete:
            state, reason = UNSAT, "finite search domain exhausted; all candidates unsat"
        else:
            state, reason = UNKNOWN, str(raw.get("reason") or "core did not prove unsat")
        assignment = {}
        fallback_candidate = None
    else:
        state, reason = UNKNOWN, "unrecognized raw core status: %s" % status
        assignment = {}
        fallback_candidate = None
    if claimed_state is not None and str(claimed_state).upper() != state:
        raise SolverError("SOLVER_STATE_PROMOTION_FORBIDDEN",
                          "raw core state cannot be promoted/relabelled",
                          "state", state, str(claimed_state).upper(), phase="classify")
    result = SolveResult(
        schema=SCHEMA_RESULT, problem_digest=request.problem_digest,
        request_digest=request.request_digest, core_source_hash=request.core_source_hash,
        state=state, reason=reason, raw_core_status=status, per_ii_checks=per_ii,
        search_scope=("CORE_MIN_II_SEARCH" if state == SAT else
                      "FALLBACK_ONLY" if state == FALLBACK else
                      "FINITE_EXHAUSTED" if state == UNSAT else "NOT_EXHAUSTED"),
        assignment=assignment, fallback_candidate=fallback_candidate, postcheck=None,
        capacity_scope="NOMINAL_MODEL_ONLY")
    return SolveResult(**{**result.__dict__, "result_digest": _canonical(
        {key: value for key, value in result.as_dict().items() if key != "result_digest"})})


def _assignment(request: SolveRequest, raw):
    starts = list(raw.get("start_times", []) or [])
    versions = list(raw.get("buffer_versions", []) or [])
    if len(starts) != len(request.index_task_map):
        raise SolverError("SOLVER_ASSIGNMENT_INVALID", "start_times length mismatch",
                          "start_times", len(request.index_task_map), len(starts), phase="classify")
    if len(versions) != len(request.index_buffer_map):
        raise SolverError("SOLVER_ASSIGNMENT_INVALID", "buffer_versions length mismatch",
                          "buffer_versions", len(request.index_buffer_map), len(versions),
                          phase="classify")
    tasks = {}
    for index, task_id in request.index_task_map.items():
        # core returns start times, not a stage label; do not invent a 0 stage.
        tasks[task_id] = {"start_time": {"value": int(starts[index]), "unit": "MODEL cycles"},
                          "stage": None, "stage_source": "NOT_RETURNED_BY_CORE"}
    buffers = {request.index_buffer_map[index]: int(version)
               for index, version in enumerate(versions)}
    return {"tasks": tasks, "buffers": buffers,
            "ii": {"value": int(raw.get("minimal_II")), "unit": "MODEL cycles"},
            "units": request.units}


def solve_real_problem(problem, model_facts, solve_options) -> SolveResult:
    request = lower_problem_to_request(problem, model_facts, solve_options)
    verify_request_coverage(problem, request)
    invoked = invoke_shared_core(request)
    return classify_core_result(request, invoked["raw"])


def verify_solution(problem, request, result) -> dict:
    """S3 placeholder: independent source postcheck is NOT implemented yet.

    Refuses rather than returning a vacuous PASS, so no caller can mistake this
    bridge for a postchecked solver.
    """
    raise SolverError("SOLVER_POSTCHECK_NOT_IMPLEMENTED",
                      "independent source postcheck arrives in S3", "postcheck",
                      "S3 implementation", None, phase="postcheck")
