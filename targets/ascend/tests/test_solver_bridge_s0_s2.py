"""S0-S2 solver bridge tests: schema, Problem->args, one shared core, 4-state map.

CPU only; uses the real shared Z3 core through adapters.tf_loop_adapter (no device).
Run: PYTHONPATH=src:targets/ascend python targets/ascend/tests/test_solver_bridge_s0_s2.py
"""
from __future__ import annotations

import ast
import dataclasses
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "targets" / "ascend"))

from tilefoundry_ascend.solver import bridge as B

PROBLEM = {
    "problem_digest": "sha256:" + "1" * 64,
    "source_digest": "sha256:" + "2" * 64,
    "ledger_sha": "sha256:" + "3" * 64,
    "loop_id": "main",
    "requested_num_stages": 2,
    "enable_offset": False,
    "tasks": [
        {"stable_id": "t0", "action_ids": ["a0"], "pipe": "mte2", "resource": "mte2",
         "latency": 4, "issue_interval": 1, "loop": "main", "stage": 0},
        {"stable_id": "t1", "action_ids": ["a1"], "pipe": "vector", "resource": "vector",
         "latency": 8, "issue_interval": 2, "loop": "main", "stage": 0},
        {"stable_id": "t2", "action_ids": ["a2"], "pipe": "mte3", "resource": "mte3",
         "latency": 4, "issue_interval": 1, "loop": "main", "stage": 0},
    ],
    "buffers": [{"root": "x_ub", "space": "ascend.ub", "bytes": 64,
                 "versions": 1, "fixed": True}],
    "dependencies": [
        {"source_id": "dep-raw-0", "producer": "t0", "consumer": "t1",
         "distance": 0, "kind": "RAW", "latency": 4},
        {"source_id": "dep-raw-1", "producer": "t1", "consumer": "t2",
         "distance": 0, "kind": "RAW", "latency": 8},
    ],
    "resource_deps": [["t0", "t1"], ["t1", "t2"]],
}
MODEL = {"model_id": "ascend-pipe-model", "model_version": "v1",
         "memory_groups": [{"id": "ub0", "capacity_bytes": 1024,
                            "buffers": ["x_ub"]}]}
OPTIONS = {"requested_num_stages": 2, "seed": 0}


def expect_code(code, fn):
    try:
        fn()
    except Exception as error:  # noqa: BLE001
        got = getattr(error, "code", None)
        if got != code:
            raise AssertionError("expected %s, got %r: %s" % (code, got, error))
        return error
    raise AssertionError("expected %s, but nothing was refused" % code)


def make_request():
    return B.lower_problem_to_request(PROBLEM, MODEL, OPTIONS)


# ---------------------------------------------------------------- S0 schema
def test_s0_request_schema_is_canonical_and_carries_maps_trace_units():
    request = make_request()
    assert request.schema == "solver-request/1"
    assert request.core_source_hash and len(request.core_source_hash) == 64
    assert request.task_index_map == {"t0": 0, "t1": 1, "t2": 2}
    assert request.buffer_index_map == {"x_ub": 0}
    kinds = {entry.kind for entry in request.constraint_trace}
    assert {"data_dep", "resource_dep", "fixed_version", "memory_group", "task"} <= kinds
    assert request.core_args["fixed_buffer_versions"] == [1]
    assert request.core_args["memory_groups"] == [[1024, 0]]
    assert request.units["buffer_sizes"] == "bytes"
    assert json.loads(json.dumps(request.as_dict(), sort_keys=True))["schema"] == "solver-request/1"
    assert request.request_digest == B._canonical(
        {key: value for key, value in request.as_dict().items() if key != "request_digest"})


# ------------------------------------------------- S1/S2 Problem -> request
def test_s1_coverage_consumes_every_required_source_constraint():
    request = make_request()
    coverage = B.verify_request_coverage(PROBLEM, request)
    assert coverage["complete"] is True
    assert coverage["consumed"] >= coverage["required"] >= 1


def test_s1_missing_trace_entry_is_coverage_missing():
    request = make_request()
    bad = dataclasses.replace(request, constraint_trace=tuple(
        entry for entry in request.constraint_trace
        if not (entry.kind == "data_dep" and entry.source_id == "dep-raw-0")))
    expect_code("SOLVER_CONSTRAINT_COVERAGE_MISSING",
                lambda: B.verify_request_coverage(PROBLEM, bad))


def test_s1_swapped_index_is_index_map_mismatch():
    request = make_request()
    swapped = dict(request.task_index_map)
    swapped["t0"], swapped["t1"] = swapped["t1"], swapped["t0"]
    bad = dataclasses.replace(request, task_index_map=swapped)
    expect_code("SOLVER_INDEX_MAP_MISMATCH",
                lambda: B.verify_request_coverage(PROBLEM, bad))


def test_s1_unknown_capacity_or_size_is_never_defaulted():
    model = {"model_id": "m", "model_version": "v1",
             "memory_groups": [{"id": "ub0", "capacity_bytes": None,
                                "buffers": ["x_ub"]}]}
    expect_code("DEFAULTED_UNKNOWN_VALUE",
                lambda: B.lower_problem_to_request(PROBLEM, model, OPTIONS))
    problem = json.loads(json.dumps(PROBLEM))
    problem["buffers"][0].pop("bytes")
    expect_code("DEFAULTED_UNKNOWN_VALUE",
                lambda: B.lower_problem_to_request(problem, MODEL, OPTIONS))


def test_s1_fixed_versions_cannot_drift_from_the_problem():
    request = make_request()
    args = dict(request.core_args)
    args["fixed_buffer_versions"] = [2]
    bad = dataclasses.replace(request, core_args=args)
    expect_code("FIXED_VERSION_MISMATCH",
                lambda: B.verify_request_coverage(PROBLEM, bad))


def test_s1_negative_distance_requires_explicit_reuse_buffer():
    problem = json.loads(json.dumps(PROBLEM))
    problem["dependencies"][0]["distance"] = -1
    expect_code("SOLVER_INDEX_MAP_MISMATCH",
                lambda: B.lower_problem_to_request(problem, MODEL, OPTIONS))


# ------------------------------------------ S2 one shared core + state map
def test_s2_only_the_shared_adapter_is_used_and_hash_must_match():
    request = make_request()
    bad = dataclasses.replace(request, core_source_hash="0" * 64)
    expect_code("SOLVER_CORE_IDENTITY_MISMATCH",
                lambda: B.invoke_shared_core(bad))
    invoked = B.invoke_shared_core(request)
    assert invoked["core_source_hash"] == request.core_source_hash
    assert invoked["raw"]["status"] in ("sat", "fallback")


def test_s2_real_core_sat_maps_back_to_stable_ids():
    result = B.solve_real_problem(PROBLEM, MODEL, OPTIONS)
    assert result.schema == "solver-result/1"
    assert result.state == B.SAT, result.reason
    assert set(result.assignment["tasks"]) == {"t0", "t1", "t2"}
    assert result.assignment["buffers"] == {"x_ub": 1}
    assert result.assignment["ii"]["value"] >= 1
    assert result.assignment["tasks"]["t0"]["stage"] is None  # never invented as 0
    assert result.capacity_scope == "NOMINAL_MODEL_ONLY"


def test_s2_fallback_is_not_promoted_to_sat():
    request = make_request()
    raw = {"status": "fallback", "fallback_reason": "all II candidates unsat",
           "start_times": [0, 4, 12], "buffer_versions": [1], "minimal_II": 3,
           "check_results": []}
    result = B.classify_core_result(request, raw)
    assert result.state == B.FALLBACK and result.assignment == {}
    assert result.fallback_candidate is not None
    expect_code("SOLVER_STATE_PROMOTION_FORBIDDEN",
                lambda: B.classify_core_result(request, raw, claimed_state="SAT"))


def test_s2_unknown_and_unsat_are_distinct_and_never_promoted():
    request = make_request()
    unknown = {"status": "unknown", "reason": "timeout", "check_results": []}
    result = B.classify_core_result(request, unknown)
    assert result.state == B.UNKNOWN and result.search_scope == "NOT_EXHAUSTED"
    expect_code("SOLVER_STATE_PROMOTION_FORBIDDEN",
                lambda: B.classify_core_result(request, unknown, claimed_state="SAT"))
    unsat_core = {"status": "unsat", "search_scope_complete": True,
                  "check_results": [], "reason": "exhausted"}
    assert B.classify_core_result(request, unsat_core).state == B.UNSAT


def test_bridge_has_no_second_solver_and_no_caseconfig_rebuild():
    source = (REPO / "targets" / "ascend" / "tilefoundry_ascend" / "solver" / "bridge.py").read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert "z3" not in imported, imported
    for token in ("build_program(" , "CaseConfig(", "z3_schedule_loop_python(", "import z3"):
        assert token not in source, token
    assert "tf_loop_adapter" in source


if __name__ == "__main__":
    failed = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print("PASS", _name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL", _name, type(exc).__name__, exc)
    raise SystemExit(1 if failed else 0)
