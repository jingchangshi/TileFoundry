"""solver-request/1 / solver-result/1 bridge (S0-S3).

Approved design: artifacts/run02/review/architect_s_solver_bridge_module_design_v1.md.
This module is a BRIDGE, not a solver. It:

  * accepts a real L Problem through a read-only duck-typed view (never rebuilding
    the business program from a test CaseConfig);
  * lowers it to the arguments of the ONE shared physical core
    (tilefoundry.vendor.tilelang_z3.adapters.tf_loop_adapter -> core.py), with a dense
    task/buffer index map and a ConstraintTrace for every required source constraint;
  * classifies the core result as SAT / UNSAT / UNKNOWN / FALLBACK without ever
    promoting fallback/unknown to SAT;
  * (S3) re-derives every inequality/resource/version/capacity obligation from the
    Problem SOURCE fields and checks the returned assignment against that independent
    derivation. The trace is never used as the expected value here, so a stale trace,
    a deleted trace entry, or a core-side version recalculation cannot make the
    postcheck pass or fail for the wrong reason.

A postcheck failure is always a named SolverError; no path returns a PASS report for a
non-SAT state and no path rewrites SAT into something else. `solver-postcheck/1` PASS
only raises sat_scope to SAT_SOURCE_POSTCHECKED; capacity_scope stays
NOMINAL_MODEL_ONLY because capacity remains a MODEL quantity.

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
SCHEMA_POSTCHECK = "solver-postcheck/1"
PROBLEM_SCHEMA = "real-problem/1"
CORE_SOURCE_HASH = "tilelang_z3/core.py"

SAT, UNSAT, UNKNOWN, FALLBACK = "SAT", "UNSAT", "UNKNOWN", "FALLBACK"

#: sat_scope values. Only a PASSing S3 source postcheck may raise the scope; nothing
#: in this module ever lowers SAT to a weaker state or invents a stronger one.
SAT_SCOPE_NOMINAL = "NOMINAL_MODEL_ONLY"
SAT_SCOPE_POSTCHECKED = "SAT_SOURCE_POSTCHECKED"

#: Postcheck failure codes (named, one per re-derived obligation family).
POSTCHECK_NOT_APPLICABLE = "SOLVER_POSTCHECK_NOT_APPLICABLE"
POSTCHECK_SOURCE_MISMATCH = "SOLVER_RESULT_SOURCE_MISMATCH"
POSTCHECK_ASSIGNMENT_INVALID = "SOLVER_ASSIGNMENT_INVALID"
POSTCHECK_ASSIGNMENT_INCOMPLETE = "ASSIGNMENT_INCOMPLETE"
POSTCHECK_DEPENDENCY_VIOLATED = "DEPENDENCY_VIOLATED"
POSTCHECK_RESOURCE_VIOLATED = "RESOURCE_CONFLICT_VIOLATED"
POSTCHECK_OWNER_VIOLATED = "OWNER_EXCLUSION_VIOLATED"
POSTCHECK_PIPE_ORDER_VIOLATED = "PIPE_ORDER_VIOLATED"
POSTCHECK_STAGE_ORDER_VIOLATED = "STAGE_ORDER_VIOLATED"
POSTCHECK_FIXED_DRIFT = "FIXED_VERSION_DRIFT"
POSTCHECK_CAPACITY_EXCEEDED = "CAPACITY_EXCEEDED"
POSTCHECK_OFFSET_VIOLATED = "OFFSET_WINDOW_VIOLATED"
POSTCHECK_II_INVALID = "II_INVALID"
POSTCHECK_RANGE_INVALID = "VALUE_OUT_OF_RANGE"

POSTCHECK_VERDICT_PASS = "POSTCHECK_PASS"

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

    @property
    def sat_scope(self) -> str:
        """Derived, never stored: only a PASSing postcheck report can raise it.

        Keeping this a property (instead of a dataclass field) is deliberate: the
        historical result_digest is computed over as_dict(), and adding a field there
        would silently change every already-published single-run digest. A caller that
        wants the scope in a serialized artifact asks postcheck_and_scope().
        """
        report = self.postcheck
        if isinstance(report, dict) and report.get("verdict") == POSTCHECK_VERDICT_PASS:
            return SAT_SCOPE_POSTCHECKED
        return SAT_SCOPE_NOMINAL

    def as_dict(self):
        return dict(self.__dict__)


# ------------------------------------------------------------ Problem -> request
def _task_id(task):
    return str(_get(task, "stable_id") or _get(task, "task_id"))


def _is_loop_task(task, loop_id):
    loop = _get(task, "loop", None)
    return loop in (None, "", loop_id, "main_compute", "row_pipeline")


#: Relation kinds that carry a buffer-version encoding instead of a logical distance.
REUSE_KINDS = frozenset({"reuse", "buffer_reuse"})


def encode_dependency_distance(distance, kind, buffer_root, buffer_index):
    """The ONE reuse/version distance encoding shared by lowering and postcheck.

    A reuse relation that names a buffer encodes the buffer-version relationship as a
    negative value ``-buffer_index(root) - 1`` rather than a logical iteration distance.
    Lowering and the S3 postcheck previously decided WHEN to apply that encoding with
    different guards:

      lower:      kind in REUSE_KINDS and buffer_root           -> always encode
      postcheck:  distance < 0 and kind in REUSE_KINDS and buffer -> encode

    So a source dependency that already says ``kind="reuse", buffer=<root>, distance=0``
    was encoded as ``-index-1`` by lowering and left as ``0`` by the postcheck. The two
    would then disagree bit-for-bit on the same source, which is a silent mismatch: it
    either mis-refuses a legal schedule or hides a real one.

    The rule below is the UNION-safe reading: an explicit reuse relation naming a buffer
    defines the encoding unconditionally, and a negative distance is only ever legitimate
    for exactly that case. Both callers now share this single definition, so they cannot
    drift again.

    Returns ``(encoded_distance, encoded?)``. Raises ValueError when a negative distance is
    claimed by a relation that is not an explicit buffer reuse.
    """
    is_reuse_with_buffer = str(kind).lower() in REUSE_KINDS and buffer_root is not None
    if is_reuse_with_buffer:
        root = str(buffer_root)
        if root not in buffer_index:
            raise ValueError("reuse names unknown buffer: %r" % (root,))
        return -int(buffer_index[root]) - 1, True
    if int(distance) < 0:
        raise ValueError(
            "a negative distance is only a buffer-version encoding, but kind=%r buffer=%r"
            % (kind, buffer_root)
        )
    return int(distance), False


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
        # The encoding rule lives in ONE place (encode_dependency_distance) so lowering and
        # the S3 postcheck cannot disagree about when a reuse relation becomes a negative
        # buffer-version encoding.
        if (str(kind).lower() in REUSE_KINDS and buffer_root is not None
                and str(buffer_root) not in buffer_index_map):
            raise SolverError("SOLVER_INDEX_MAP_MISMATCH", "reuse names unknown buffer",
                              str(buffer_root), list(buffer_index_map), buffer_root, phase="lower")
        if distance < 0 and not (str(kind).lower() in REUSE_KINDS and buffer_root is not None):
            raise SolverError("SOLVER_INDEX_MAP_MISMATCH",
                              "a negative distance is only a buffer-version encoding",
                              "dep[%d]" % position, "explicit reuse + buffer", distance,
                              phase="lower")
        distance, encoded_reuse = encode_dependency_distance(
            distance, kind, buffer_root, buffer_index_map)
        if encoded_reuse:
            trace_add("data_dep", str(_get(dep, "source_id", "dep%d" % position)),
                      "data_deps.distance", (producer, consumer), "buffer version encoding")
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
    return SolveResult(**{**result.__dict__, "result_digest": _result_digest(result)})


def _result_digest(result) -> str:
    """Digest over the stored fields only (as_dict), exactly as S0-S2 computed it.

    sat_scope is a derived property and is deliberately NOT a stored field, so the
    historical default-path result_digest cannot drift when S3 is added.
    """
    return _canonical({key: value for key, value in result.as_dict().items()
                       if key != "result_digest"})


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


def solve_real_problem(problem, model_facts, solve_options,
                       require_source_postcheck: bool = False) -> SolveResult:
    """Lower -> one shared core -> classify; optionally gate on the S3 postcheck.

    ``require_source_postcheck=False`` is the historical path: byte-identical
    result_digest, postcheck=None, sat_scope=NOMINAL_MODEL_ONLY.

    ``require_source_postcheck=True`` demands an independently postchecked SAT. A
    non-SAT state is a named refusal (never a PASS) and a failing postcheck raises,
    so no caller can receive a result that merely *looks* postchecked.
    """
    request = lower_problem_to_request(problem, model_facts, solve_options)
    verify_request_coverage(problem, request)
    invoked = invoke_shared_core(request)
    result = classify_core_result(request, invoked["raw"])
    if not require_source_postcheck:
        return result
    if result.state != SAT:
        raise SolverError(POSTCHECK_NOT_APPLICABLE,
                          "source postcheck only applies to a SAT result",
                          result.problem_digest, SAT, result.state, phase="postcheck")
    report = verify_solution(problem, request, result)
    return _attach_postcheck(result, report)


# ------------------------------------------------- S3 independent source postcheck
# Everything below re-derives the expected obligations from the Problem SOURCE
# fields. request.constraint_trace is NEVER read here: a trace is a record of what
# the bridge *asked* for, so using it as "expected" would make the check circular.
def _sorted_task_ids(loop_tasks):
    return sorted(_task_id(task) for task in loop_tasks)


def _derive_source_expectation(problem, request) -> dict:
    """Re-derive every expected obligation from the Problem's own source fields."""
    schema = _get(problem, "schema", None)
    if str(schema) != PROBLEM_SCHEMA:
        raise SolverError(POSTCHECK_SOURCE_MISMATCH,
                          "source postcheck needs canonical real-problem/1",
                          "schema", PROBLEM_SCHEMA, schema, phase="postcheck")
    loop_id = _get(problem, "loop_id", None)
    if loop_id is None:
        raise SolverError(POSTCHECK_SOURCE_MISMATCH,
                          "canonical problem must carry loop_id",
                          "loop_id", "a loop id", loop_id, phase="postcheck")
    loop_id = str(loop_id)

    tasks_in = list(_get(problem, "tasks", []) or [])
    if not tasks_in:
        raise SolverError(POSTCHECK_SOURCE_MISMATCH, "problem has no tasks",
                          "tasks", "nonempty", 0, phase="postcheck")
    # Partition by POSITION, not by equality: two structurally identical task records
    # must not make one of them look "outside the loop".
    is_loop = [_is_loop_task(task, loop_id) for task in tasks_in]
    loop_tasks = [task for task, inside in zip(tasks_in, is_loop) if inside]
    outside = [task for task, inside in zip(tasks_in, is_loop) if not inside]
    if outside:
        raise SolverError("UNSUPPORTED_CONSTRAINT",
                          "outside-loop task has no per-iteration obligation to recheck",
                          _task_id(outside[0]), "loop task",
                          _get(outside[0], "loop"), phase="postcheck")
    if not loop_tasks:
        raise SolverError(POSTCHECK_SOURCE_MISMATCH, "problem has no loop task",
                          "tasks", ">=1 loop task", 0, phase="postcheck")

    task_ids = _sorted_task_ids(loop_tasks)
    if len(set(task_ids)) != len(task_ids):
        raise SolverError(POSTCHECK_SOURCE_MISMATCH, "duplicate task stable_id",
                          "tasks", "unique stable_id", task_ids, phase="postcheck")
    task_index = {tid: index for index, tid in enumerate(task_ids)}
    latencies, iis, resource_flags = {}, {}, {}
    for task in loop_tasks:
        tid = _task_id(task)
        resource = str(_get(task, "resource") or _get(task, "pipe") or "").lower()
        if resource not in RESOURCE_BITS:
            raise SolverError("UNSUPPORTED_CONSTRAINT", "unknown resource pipe",
                              tid, list(RESOURCE_BITS), resource, phase="postcheck")
        latency = _get(task, "latency", None)
        issue_interval = _get(task, "issue_interval", None)
        if latency is None or issue_interval is None:
            raise SolverError("DEFAULTED_UNKNOWN_VALUE",
                              "task latency/ii must be explicit to be postcheckable",
                              tid, "explicit latency/issue_interval",
                              [latency, issue_interval], phase="postcheck")
        latencies[tid] = int(latency)
        iis[tid] = int(issue_interval)
        resource_flags[tid] = int(_get(task, "resource_flags", RESOURCE_BITS[resource]))

    buffers_in = list(_get(problem, "buffers", []) or [])
    roots = sorted(str(_get(buffer, "root")) for buffer in buffers_in)
    if len(set(roots)) != len(roots):
        raise SolverError(POSTCHECK_SOURCE_MISMATCH, "duplicate buffer root",
                          "buffers", "unique root", roots, phase="postcheck")
    buffer_index = {root: index for index, root in enumerate(roots)}
    by_root = {str(_get(buffer, "root")): buffer for buffer in buffers_in}
    buffer_sizes, fixed_versions = [], []
    for root in roots:
        buffer = by_root[root]
        size = _get(buffer, "bytes", _get(buffer, "size_bytes"))
        if size is None or int(size) <= 0:
            raise SolverError("DEFAULTED_UNKNOWN_VALUE",
                              "buffer size must be explicit to be postcheckable",
                              root, ">0 bytes", size, phase="postcheck")
        buffer_sizes.append(int(size))
        if bool(_get(buffer, "fixed", False)):
            want = _get(buffer, "versions", None)
            if want is None or int(want) <= 0:
                raise SolverError(POSTCHECK_FIXED_DRIFT,
                                  "fixed buffer needs a positive version count",
                                  root, ">0", want, phase="postcheck")
            fixed_versions.append(int(want))
        else:
            fixed_versions.append(None)

    # Data dependencies: re-encode a negative distance here from the SOURCE reuse
    # relation, independently of what the lowering did.
    data_deps = []
    for position, dep in enumerate(list(_get(problem, "dependencies", []) or [])):
        producer = str(_get(dep, "producer"))
        consumer = str(_get(dep, "consumer"))
        source_id = str(_get(dep, "source_id", "dependencies[%d]" % position))
        if producer not in task_index or consumer not in task_index:
            raise SolverError(POSTCHECK_SOURCE_MISMATCH,
                              "dependency names an unknown task", source_id,
                              task_ids, [producer, consumer], phase="postcheck")
        distance = int(_get(dep, "distance", 0))
        kind = str(_get(dep, "kind", "RAW")).lower()
        buffer_root = _get(dep, "buffer", None)
        # Re-encoded here from the SOURCE relation, independently of what lowering did -- but
        # using the SAME rule as lowering (encode_dependency_distance). Previously this branch
        # was guarded by `distance < 0`, while lowering encoded unconditionally for an explicit
        # reuse + buffer. A source dep with kind=reuse, buffer=<root>, distance=0 was therefore
        # encoded as -index-1 by lowering and left as 0 here: the postcheck rejected its own
        # lowering (DEPENDENCY_VIOLATED), a self-inconsistency rather than a model defect.
        if str(buffer_root) not in buffer_index and str(kind) in REUSE_KINDS and buffer_root is not None:
            raise SolverError(POSTCHECK_SOURCE_MISMATCH,
                              "reuse names an unknown buffer", source_id,
                              roots, buffer_root, phase="postcheck")
        if distance < 0 and not (kind in REUSE_KINDS and buffer_root is not None):
            raise SolverError(POSTCHECK_SOURCE_MISMATCH,
                              "a negative distance is only a buffer-version encoding",
                              source_id, "explicit reuse + buffer", distance,
                              phase="postcheck")
        distance, _encoded = encode_dependency_distance(distance, kind, buffer_root, buffer_index)
        latency = _get(dep, "latency", None)
        latency = int(latency) if latency is not None else latencies[producer]
        data_deps.append((producer, consumer, distance, latency, source_id))

    def _source_pairs(field_name):
        pairs = []
        for index, pair in enumerate(list(_get(problem, field_name, []) or [])):
            producer, consumer = str(pair[0]), str(pair[1])
            if producer not in task_index or consumer not in task_index:
                raise SolverError(POSTCHECK_SOURCE_MISMATCH,
                                  "pair names an unknown task",
                                  "%s[%d]" % (field_name, index), task_ids,
                                  [producer, consumer], phase="postcheck")
            pairs.append((producer, consumer))
        return pairs

    resource_deps = _source_pairs("resource_deps")
    pipe_order_deps = _source_pairs("pipe_order_deps")
    stage_order_deps = _source_pairs("stage_order_deps")
    owner_deps = []
    for index, item in enumerate(list(_get(problem, "owner_exclusion_deps", []) or [])):
        producer, consumer = str(item[0]), str(item[1])
        if producer not in task_index or consumer not in task_index:
            raise SolverError(POSTCHECK_SOURCE_MISMATCH,
                              "owner exclusion names an unknown task",
                              "owner_exclusion_deps[%d]" % index, task_ids,
                              [producer, consumer], phase="postcheck")
        owner_deps.append((producer, consumer,
                           int(item[2]) if len(item) > 2 else 1,
                           int(item[3]) if len(item) > 3 else 1))

    # Stage budget / offset are POLICY inputs, not source edges. They may legitimately
    # be overridden per solve (solve_options wins over the Problem's own declaration in
    # lower_problem_to_request), so the REQUEST is the authority for them and the
    # postcheck does not pretend to re-derive them. Every other core argument IS
    # re-derived from the source and cross-checked in _verify_request_matches_source,
    # so a request cannot quietly widen an edge, a size, or a fixed version.
    core_args = dict(request.core_args or {})
    num_stages = core_args.get("num_stages", None)
    if num_stages is None:
        num_stages = _get(problem, "requested_num_stages", None)
    if num_stages is None or int(num_stages) < 1:
        raise SolverError(POSTCHECK_RANGE_INVALID,
                          "stage budget must be positive to bound buffer versions",
                          "num_stages", ">=1", num_stages, phase="postcheck")
    num_stages = int(num_stages)
    enable_offset = core_args.get("enable_offset", None)
    if enable_offset is None:
        enable_offset = _get(problem, "enable_offset", False)

    # Capacity is a MODEL quantity. It is only carried by model_facts, which reaches
    # this pure function through core_args (never through the trace). Group membership
    # is re-mapped onto the SOURCE buffer order and range/duplication checked, so a
    # request cannot drop a buffer from every group or double-count one to slip under
    # the byte bound.
    memory_groups = []
    grouped = set()
    for entry in list(core_args.get("memory_groups", []) or []):
        if not entry:
            raise SolverError(POSTCHECK_SOURCE_MISMATCH,
                              "empty memory group entry", "memory_groups", "[capacity, *indices]",
                              entry, phase="postcheck")
        capacity = int(entry[0])
        members = []
        for raw_index in list(entry[1:]):
            index = int(raw_index)
            if index < 0 or index >= len(roots):
                raise SolverError(POSTCHECK_SOURCE_MISMATCH,
                                  "memory group names a buffer index outside the source set",
                                  "memory_groups", list(range(len(roots))), index,
                                  phase="postcheck")
            members.append(roots[index])
            if index in grouped:
                raise SolverError(POSTCHECK_SOURCE_MISMATCH,
                                  "buffer is a member of more than one memory group",
                                  roots[index], "one group", [roots[i] for i in sorted(grouped)],
                                  phase="postcheck")
            grouped.add(index)
        memory_groups.append((capacity, members))
    if memory_groups and len(grouped) != len(roots):
        missing = [roots[i] for i in range(len(roots)) if i not in grouped]
        raise SolverError(POSTCHECK_SOURCE_MISMATCH,
                          "every buffer must appear in a declared memory group",
                          "memory_groups", roots, {"uncovered": missing},
                          phase="postcheck")

    return {
        "loop_id": loop_id, "task_ids": task_ids, "task_index": task_index,
        "latencies": latencies, "iis": iis, "resource_flags": resource_flags,
        "buffer_roots": roots, "buffer_index": buffer_index,
        "buffer_sizes": buffer_sizes, "fixed_versions": fixed_versions,
        "data_deps": data_deps, "resource_deps": resource_deps,
        "owner_exclusion_deps": owner_deps, "pipe_order_deps": pipe_order_deps,
        "stage_order_deps": stage_order_deps, "memory_groups": memory_groups,
        "num_stages": num_stages,
        "enable_offset": bool(enable_offset),
    }


def _verify_request_matches_source(request, expected) -> int:
    """The result is only attributable to this Problem if the core was asked the same
    thing the source says. Any drift here is a provenance reject, not a PASS."""
    core_args = dict(request.core_args or {})

    def mismatch(field_name, want, got):
        raise SolverError(POSTCHECK_SOURCE_MISMATCH,
                          "request does not match the Problem source it claims",
                          field_name, want, got, phase="postcheck")

    if request.task_index_map != expected["task_index"]:
        mismatch("task_index_map", expected["task_index"], request.task_index_map)
    if request.buffer_index_map != expected["buffer_index"]:
        mismatch("buffer_index_map", expected["buffer_index"], request.buffer_index_map)
    order = expected["task_ids"]
    want_latencies = [expected["latencies"][tid] for tid in order]
    want_iis = [expected["iis"][tid] for tid in order]
    want_flags = [expected["resource_flags"][tid] for tid in order]
    if list(core_args.get("latencies", []) or []) != want_latencies:
        mismatch("latencies", want_latencies, core_args.get("latencies"))
    if list(core_args.get("iis", []) or []) != want_iis:
        mismatch("iis", want_iis, core_args.get("iis"))
    if list(core_args.get("resource_flags", []) or []) != want_flags:
        mismatch("resource_flags", want_flags, core_args.get("resource_flags"))
    if list(core_args.get("buffer_sizes", []) or []) != expected["buffer_sizes"]:
        mismatch("buffer_sizes", expected["buffer_sizes"], core_args.get("buffer_sizes"))
    if list(core_args.get("fixed_buffer_versions", []) or []) != expected["fixed_versions"]:
        mismatch("fixed_buffer_versions", expected["fixed_versions"],
                 core_args.get("fixed_buffer_versions"))
    index = expected["task_index"]
    want_deps = [(index[u], index[v], distance, latency)
                 for u, v, distance, latency, _ in expected["data_deps"]]
    if [tuple(item) for item in list(core_args.get("data_deps", []) or [])] != want_deps:
        mismatch("data_deps", want_deps, core_args.get("data_deps"))
    want_resource = [(index[u], index[v]) for u, v in expected["resource_deps"]]
    if [tuple(item) for item in list(core_args.get("resource_deps", []) or [])] != want_resource:
        mismatch("resource_deps", want_resource, core_args.get("resource_deps"))
    want_owner = [(index[u], index[v], a, b)
                  for u, v, a, b in expected["owner_exclusion_deps"]]
    if [tuple(item) for item in list(core_args.get("owner_exclusion_deps", []) or [])] != want_owner:
        mismatch("owner_exclusion_deps", want_owner, core_args.get("owner_exclusion_deps"))
    want_pipe = [(index[u], index[v]) for u, v in expected["pipe_order_deps"]]
    if [tuple(item) for item in list(core_args.get("pipe_order_deps", []) or [])] != want_pipe:
        mismatch("pipe_order_deps", want_pipe, core_args.get("pipe_order_deps"))
    want_stage = [(index[u], index[v]) for u, v in expected["stage_order_deps"]]
    if [tuple(item) for item in list(core_args.get("stage_order_deps", []) or [])] != want_stage:
        mismatch("stage_order_deps", want_stage, core_args.get("stage_order_deps"))
    want_groups = [[capacity] + [expected["buffer_index"][root] for root in members]
                   for capacity, members in expected["memory_groups"]]
    got_groups = [[int(value) for value in group]
                  for group in list(core_args.get("memory_groups", []) or [])]
    if got_groups != want_groups:
        mismatch("memory_groups", want_groups, got_groups)
    return 1


def _assignment_tables(assignment):
    if not isinstance(assignment, dict):
        raise SolverError(POSTCHECK_ASSIGNMENT_INVALID, "assignment is not a mapping",
                          "assignment", "dict", type(assignment).__name__, phase="postcheck")
    tasks = assignment.get("tasks", None)
    buffers = assignment.get("buffers", None)
    ii = assignment.get("ii", None)
    if not isinstance(tasks, dict):
        raise SolverError(POSTCHECK_ASSIGNMENT_INVALID, "assignment has no task table",
                          "assignment.tasks", "dict of stable_id -> start_time",
                          type(tasks).__name__, phase="postcheck")
    if not isinstance(buffers, dict):
        raise SolverError(POSTCHECK_ASSIGNMENT_INVALID, "assignment has no buffer table",
                          "assignment.buffers", "dict of root -> version",
                          type(buffers).__name__, phase="postcheck")
    if not isinstance(ii, dict):
        raise SolverError(POSTCHECK_ASSIGNMENT_INVALID, "assignment has no II record",
                          "assignment.ii", "dict with value", type(ii).__name__,
                          phase="postcheck")
    return tasks, buffers, ii


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _check_value_ranges(tasks, buffers, expected, counters):
    num_stages = expected["num_stages"]
    starts = {}
    for tid in sorted(tasks):
        entry = tasks[tid]
        start = None
        if isinstance(entry, dict):
            start_time = entry.get("start_time", None)
            if isinstance(start_time, dict):
                start = start_time.get("value", None)
        if not _integer(start):
            raise SolverError(POSTCHECK_ASSIGNMENT_INVALID,
                              "start_time must be an integer MODEL cycle value",
                              tid, "integer start_time", start, phase="postcheck")
        if start < 0:
            raise SolverError(POSTCHECK_RANGE_INVALID,
                              "start_time cannot be negative", tid, ">= 0", start,
                              phase="postcheck")
        starts[tid] = int(start)
        counters["starts"] += 1
    versions = {}
    for root in sorted(buffers):
        version = buffers[root]
        if not _integer(version):
            raise SolverError(POSTCHECK_ASSIGNMENT_INVALID,
                              "buffer version must be an integer",
                              root, "integer version", version, phase="postcheck")
        if version < 1 or version > num_stages:
            raise SolverError(POSTCHECK_RANGE_INVALID,
                              "buffer version must stay inside [1, num_stages]",
                              root, "1..%d" % num_stages, version, phase="postcheck")
        versions[root] = int(version)
        counters["buffer_versions"] += 1
    return starts, versions


def _check_ii(ii_record, expected, counters):
    ii = ii_record.get("value", None)
    if not _integer(ii):
        raise SolverError(POSTCHECK_II_INVALID, "II must be an integer",
                          "assignment.ii", "integer > 0", ii, phase="postcheck")
    if ii <= 0:
        raise SolverError(POSTCHECK_II_INVALID, "II must be positive",
                          "assignment.ii", "> 0", ii, phase="postcheck")
    floor = max(expected["iis"].values()) if expected["iis"] else 1
    if ii < floor:
        raise SolverError(POSTCHECK_II_INVALID,
                          "II is below the source issue-interval lower bound",
                          "assignment.ii", ">= %d" % floor, ii, phase="postcheck")
    counters["ii"] += 1
    return int(ii)


def _check_fixed_versions(versions, expected, counters):
    for root, want in zip(expected["buffer_roots"], expected["fixed_versions"]):
        if want is None:
            continue
        got = versions[root]
        if got != want:
            raise SolverError(POSTCHECK_FIXED_DRIFT,
                              "fixed buffer version drifted from the Problem source",
                              root, want, got, phase="postcheck")
        counters["fixed_versions"] += 1


def _check_data_deps(starts, versions, ii, expected, counters):
    for producer, consumer, distance, latency, source_id in expected["data_deps"]:
        gap = starts[consumer] - starts[producer]
        if distance >= 0:
            required = latency - ii * distance
            rule = "start_v - start_u >= latency - ii*distance"
            bound = {"distance": distance, "latency": latency, "ii": ii,
                     "required_gap": required}
        else:
            buffer_id = -distance - 1
            root = expected["buffer_roots"][buffer_id]
            required = latency - ii * versions[root]
            rule = "start_v - start_u >= latency - ii*versions[buffer]"
            bound = {"buffer": root, "latency": latency, "ii": ii,
                     "versions": versions[root], "required_gap": required}
        if gap < required:
            raise SolverError(POSTCHECK_DEPENDENCY_VIOLATED,
                              "data dependency %s is violated" % rule,
                              source_id,
                              dict(bound, producer=producer, consumer=consumer),
                              {"start_producer": starts[producer],
                               "start_consumer": starts[consumer], "gap": gap},
                              phase="postcheck")
        counters["data_deps"] += 1


def _cyclic_nonoverlap(first_residue, first_ii, second_residue, second_ii, ii):
    """Two modulo issue windows on a ring of length ii; True iff they cannot overlap.

    This is the literal source semantics (each task occupies [r, r+ii_task) modulo ii)
    and is stated as the exact disjunction the two admissible modulo orders allow,
    rather than as an inequality on the wrap-around distance: those two forms differ,
    and only this one matches "the windows do not intersect".

    Checked exhaustively against a set-intersection model for ii in 2..9.
    """
    first_before = (second_residue - first_residue >= first_ii
                    and first_residue - second_residue + ii >= second_ii)
    second_before = (first_residue - second_residue >= second_ii
                     and second_residue - first_residue + ii >= first_ii)
    return first_before or second_before


def _check_resource_deps(starts, ii, expected, counters):
    for producer, consumer in expected["resource_deps"]:
        if producer == consumer:
            raise SolverError(POSTCHECK_SOURCE_MISMATCH,
                              "a resource pair must name two distinct tasks",
                              "resource_deps", "two distinct tasks",
                              [producer, consumer], phase="postcheck")
        producer_residue = starts[producer] % ii
        consumer_residue = starts[consumer] % ii
        if not _cyclic_nonoverlap(producer_residue, expected["iis"][producer],
                                  consumer_residue, expected["iis"][consumer], ii):
            raise SolverError(POSTCHECK_RESOURCE_VIOLATED,
                              "two tasks overlap on one resource in modulo time",
                              "%s<->%s" % (producer, consumer),
                              {"ii": ii, "issue_intervals": [expected["iis"][producer],
                                                            expected["iis"][consumer]]},
                              {"starts": [starts[producer], starts[consumer]],
                               "residues": [producer_residue, consumer_residue]},
                              phase="postcheck")
        counters["resource_deps"] += 1


def _check_owner_deps(starts, ii, expected, counters):
    for producer, consumer, latency_producer, latency_consumer in expected["owner_exclusion_deps"]:
        # Owners exchange one physical slot at their issue PHASES, so the hand-off is
        # constrained on residues, not on absolute start times (a whole extra iteration
        # must not change the verdict).
        producer_residue = starts[producer] % ii
        consumer_residue = starts[consumer] % ii
        producer_first = (consumer_residue - producer_residue >= latency_producer
                          and producer_residue - consumer_residue + ii >= latency_consumer)
        consumer_first = (producer_residue - consumer_residue >= latency_consumer
                          and consumer_residue - producer_residue + ii >= latency_producer)
        if not (producer_first or consumer_first):
            raise SolverError(POSTCHECK_OWNER_VIOLATED,
                              "shared-storage owners violate their cyclic hand-off",
                              "%s<->%s" % (producer, consumer),
                              {"ii": ii, "latencies": [latency_producer, latency_consumer],
                               "residues": [producer_residue, consumer_residue]},
                              {"starts": [starts[producer], starts[consumer]],
                               "residues": [producer_residue, consumer_residue]},
                              phase="postcheck")
        counters["owner_exclusion_deps"] += 1



def _check_pipe_order(starts, ii, expected, counters):
    for producer, consumer in expected["pipe_order_deps"]:
        if (starts[producer] % ii) > (starts[consumer] % ii):
            raise SolverError(POSTCHECK_PIPE_ORDER_VIOLATED,
                              "source issue order is inverted on one pipe",
                              "%s->%s" % (producer, consumer),
                              {"residue_producer": starts[producer] % ii,
                               "residue_order": "<= residue_consumer"},
                              {"residue_producer": starts[producer] % ii,
                               "starts": [starts[producer], starts[consumer]]},
                              phase="postcheck")
        counters["pipe_order_deps"] += 1


def _check_stage_order(starts, ii, expected, counters):
    for producer, consumer in expected["stage_order_deps"]:
        producer_stage = starts[producer] // ii
        consumer_stage = starts[consumer] // ii
        if producer_stage > consumer_stage:
            raise SolverError(POSTCHECK_STAGE_ORDER_VIOLATED,
                              "stage order k_u <= k_w is inverted",
                              "%s->%s" % (producer, consumer),
                              {"k_producer": "<= k_consumer"},
                              {"k_producer": producer_stage, "k_consumer": consumer_stage},
                              phase="postcheck")
        counters["stage_order_deps"] += 1


def _check_capacity(versions, expected, counters):
    for capacity, members in expected["memory_groups"]:
        total = 0
        for root in members:
            size = expected["buffer_sizes"][expected["buffer_index"][root]]
            total += versions[root] * size
        if total > capacity:
            raise SolverError(POSTCHECK_CAPACITY_EXCEEDED,
                              "MODEL capacity is exceeded by the returned versions",
                              "memory_group",
                              {"capacity_bytes": capacity, "members": members},
                              {"used_bytes": total,
                               "versions": {root: versions[root] for root in members}},
                              phase="postcheck")
        counters["memory_groups"] += 1


def _check_offset_window(starts, ii, expected, counters):
    if expected["enable_offset"]:
        return
    for bit in (1, 2, 4, 8, 16, 32, 64):
        group = [tid for tid in expected["task_ids"]
                 if expected["resource_flags"][tid] & bit]
        if len(group) <= 1:
            continue
        values = [starts[tid] for tid in group]
        span = max(values) - min(values)
        if span >= ii:
            raise SolverError(POSTCHECK_OFFSET_VIOLATED,
                              "enable_offset=False requires one II window per resource",
                              "resource_bit:%d" % bit,
                              {"ii": ii, "max_span": "< %d" % ii},
                              {"span": span, "starts": dict(zip(group, values))},
                              phase="postcheck")
        counters["offset_groups"] += 1


def _attach_postcheck(result, report):
    """Attach a PASSing report. sat_scope then derives to SAT_SOURCE_POSTCHECKED.

    Capacity stays NOMINAL_MODEL_ONLY: capacity is a MODEL bound, not a measured one.
    """
    if not isinstance(report, dict) or report.get("verdict") != POSTCHECK_VERDICT_PASS:
        raise SolverError(POSTCHECK_NOT_APPLICABLE,
                          "sat_scope can only be raised by a PASSing postcheck report",
                          getattr(result, "problem_digest", ""),
                          POSTCHECK_VERDICT_PASS, report, phase="postcheck")
    if str(_get(result, "state", "")) != SAT:
        raise SolverError(POSTCHECK_NOT_APPLICABLE,
                          "only a SAT result may carry a passing postcheck",
                          getattr(result, "problem_digest", ""), SAT,
                          _get(result, "state", None), phase="postcheck")
    updated = SolveResult(**{**result.__dict__, "postcheck": report})
    return SolveResult(**{**updated.__dict__, "result_digest": _result_digest(updated)})



def verify_solution(problem, request, result) -> dict:
    """S3: independent source postcheck of a SAT assignment.

    Expected values are re-derived from the Problem source fields
    (tasks/buffers/dependencies/resource_deps/owner_exclusion_deps/
    pipe_order_deps/stage_order_deps) plus the MODEL channels (stage budget, capacity
    groups). ``request.constraint_trace`` is never read, so deleting or rewriting it
    cannot change the verdict.

    PASS returns a ``solver-postcheck/1`` report. Every failure raises a named
    SolverError; a non-SAT state is refused, never postchecked and never promoted.
    """
    if request is None:
        raise SolverError(POSTCHECK_SOURCE_MISMATCH,
                          "source postcheck needs the request that produced the result",
                          "request", SCHEMA_REQUEST, None, phase="postcheck")
    state = str(_get(result, "state", None))
    if state != SAT:
        raise SolverError(POSTCHECK_NOT_APPLICABLE,
                          "source postcheck does not apply to a non-SAT result",
                          str(_get(result, "problem_digest", "")), SAT, state,
                          phase="postcheck")

    problem_digest = str(_get(problem, "problem_digest", "") or "")
    request_digest = str(_get(request, "problem_digest", "") or "")
    result_digest_source = str(_get(result, "problem_digest", "") or "")
    if not problem_digest or not (problem_digest == request_digest == result_digest_source):
        raise SolverError(POSTCHECK_SOURCE_MISMATCH,
                          "result/problem/request do not describe the same Problem",
                          "problem_digest", problem_digest,
                          {"request": request_digest, "result": result_digest_source},
                          phase="postcheck")

    expected = _derive_source_expectation(problem, request)
    tasks, buffers, ii_record = _assignment_tables(_get(result, "assignment", None))

    if set(tasks) != set(expected["task_ids"]):
        raise SolverError(POSTCHECK_ASSIGNMENT_INCOMPLETE,
                          "assignment task set is not the Problem loop task set",
                          "assignment.tasks", expected["task_ids"], sorted(tasks),
                          phase="postcheck")
    if set(buffers) != set(expected["buffer_roots"]):
        raise SolverError(POSTCHECK_ASSIGNMENT_INCOMPLETE,
                          "assignment buffer set is not the Problem buffer set",
                          "assignment.buffers", expected["buffer_roots"], sorted(buffers),
                          phase="postcheck")

    counters = {"tasks": len(expected["task_ids"]),
                "buffers": len(expected["buffer_roots"]),
                "starts": 0, "ii": 0, "buffer_versions": 0,
                "data_deps": 0, "resource_deps": 0, "owner_exclusion_deps": 0,
                "pipe_order_deps": 0, "stage_order_deps": 0,
                "fixed_versions": 0, "memory_groups": 0, "offset_groups": 0}
    ii = _check_ii(ii_record, expected, counters)
    starts, versions = _check_value_ranges(tasks, buffers, expected, counters)
    _check_fixed_versions(versions, expected, counters)
    _check_data_deps(starts, versions, ii, expected, counters)
    _check_resource_deps(starts, ii, expected, counters)
    _check_owner_deps(starts, ii, expected, counters)
    _check_pipe_order(starts, ii, expected, counters)
    _check_stage_order(starts, ii, expected, counters)
    _check_capacity(versions, expected, counters)
    _check_offset_window(starts, ii, expected, counters)
    counters["request_source_consistency"] = _verify_request_matches_source(request, expected)

    report = {
        "schema": SCHEMA_POSTCHECK,
        "verdict": POSTCHECK_VERDICT_PASS,
        "state": state,
        "sat_scope": SAT_SCOPE_POSTCHECKED,
        "capacity_scope": "NOMINAL_MODEL_ONLY",
        "source_authority": "problem-source-fields",
        "trace_used": False,
        "problem_digest": problem_digest,
        "request_digest": str(_get(request, "request_digest", "") or ""),
        "result_digest": str(_get(result, "result_digest", "") or ""),
        "core_source_hash": str(_get(request, "core_source_hash", "") or ""),
        "loop_source_id": expected["loop_id"],
        "ii": ii,
        "num_stages": expected["num_stages"],
        "enable_offset": expected["enable_offset"],
        "units": dict(_get(request, "units", {}) or {}),
        "checked": counters,
        "limits": {"physical_feasibility": "UNPROVEN", "capacity": "MODEL",
                   "units": "MODEL cycles"},
    }
    report["postcheck_digest"] = _canonical(
        {key: value for key, value in report.items() if key != "postcheck_digest"})
    return report


def postcheck_and_scope(problem, request, result):
    """Return (report, SAT_SCOPE_POSTCHECKED) only for a PASSing postcheck.

    Any failure raises: there is deliberately no "NOMINAL_MODEL_ONLY" return here,
    because a silent downgrade would let a caller treat a rejection as normal.
    """
    report = verify_solution(problem, request, result)
    if not isinstance(report, dict) or report.get("verdict") != POSTCHECK_VERDICT_PASS:
        raise SolverError(POSTCHECK_NOT_APPLICABLE,
                          "postcheck_and_scope requires a PASSing report",
                          str(_get(result, "problem_digest", "")),
                          POSTCHECK_VERDICT_PASS, report, phase="postcheck")
    return report, SAT_SCOPE_POSTCHECKED
