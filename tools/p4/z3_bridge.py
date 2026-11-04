#!/usr/bin/env python3
"""WBS 04-07/04-08 CPU planning harness: two adapters, ONE physical Z3 core.

NO device. NO P5/performance/native credit. UNKNOWN / excluded Problem fields are passed
through unchanged, never filled with a convenient "known" value.

The harness builds normalized FULL constraint sets from real ScheduleProblem objects,
passes them through BOTH tf_loop_adapter and tl_loop_adapter to the SAME
core.z3_schedule_loop_python file, and records solver states sat / unsat / unknown /
fallback SEPARATELY. The core's explicit serial fallback is never reported as solved.

CLI:
  python tools/p4/z3_bridge.py --out <evidence-dir>
  python tools/p4/z3_bridge.py --out <evidence-dir> --mutation <name>
"""
from __future__ import annotations

import argparse
import ast
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
TF = HERE.parents[1]
SRC = TF / "src"
TARGETS = TF / "targets" / "ascend"
sys.path[:0] = [str(SRC), str(TARGETS)]

from tilefoundry.vendor.tilelang_z3 import adapters, core  # noqa: E402
from tilefoundry_ascend.memory import SPACE_FACTS  # noqa: E402
from tilefoundry_ascend.schedule.program import CaseConfig, build_program  # noqa: E402
from tilefoundry_ascend.schedule.problem import Task, extract_problem  # noqa: E402

UPSTREAM = Path("/home/s00653124/workspace/tilelang/tilelang/ascend/transform/z3_scheduler.py")
SEED = 42
MODEL_VERSION = "ascend-pipe-model-v2-040708"
PIPE_FLAGS = {"mte2": 2, "vector": 16, "mte3": 4}
STATES = ("sat", "unsat", "unknown", "fallback")
REQUIRED_CATEGORIES = {"ready": 2, "free": 2, "reuse": 1}

CASES = {
    "prefill_dyn": dict(case_id="prefill_dyn", batch=8192, d=4096, dtype="bfloat16",
                        eps=1e-6, core_count=56, rows=4, loop_mode="dyn",
                        requested_num_stages=4, x_versions=4, y_versions=2,
                        offset_policy="source_default", l2_read="NOTALLOC_CLEAN",
                        l2_write="NORMAL_FV", target_profile="ascend950pr_9579"),
    "prefill_recompute": dict(case_id="prefill_recompute", batch=8192, d=4096, dtype="bfloat16",
                              eps=1e-6, core_count=56, rows=4, loop_mode="recompute",
                              requested_num_stages=4, x_versions=4, y_versions=2,
                              offset_policy="source_default", l2_read="NOTALLOC_CLEAN",
                              l2_write="NORMAL_FV", target_profile="ascend950pr_9579"),
    "decode_dyn": dict(case_id="decode_dyn", batch=64, d=4096, dtype="bfloat16",
                       eps=1e-6, core_count=56, rows=1, loop_mode="dyn",
                       requested_num_stages=2, x_versions=2, y_versions=1,
                       offset_policy="source_default", l2_read="NOTALLOC_CLEAN",
                       l2_write="NORMAL_FV", target_profile="ascend950pr_9579"),
    "decode_recompute": dict(case_id="decode_recompute", batch=64, d=4096, dtype="bfloat16",
                             eps=1e-6, core_count=56, rows=1, loop_mode="recompute",
                             requested_num_stages=2, x_versions=2, y_versions=1,
                             offset_policy="source_default", l2_read="NOTALLOC_CLEAN",
                             l2_write="NORMAL_FV", target_profile="ascend950pr_9579"),
}


class BridgeError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__("%s: %s" % (code, message))
        self.code, self.message = code, message


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(obj) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_problem(case_id: str):
    return extract_problem(build_program(CaseConfig(**CASES[case_id])))


def capacity_facts() -> dict:
    f = SPACE_FACTS["ascend.ub"]
    return {
        "space": f.name,
        "category": f.category,
        "owner": f.owner,
        "nominal_bytes": f.nominal_bytes,
        "usable_bytes": f.usable_bytes,
        "alignment": f.alignment,
        "allocatable": f.allocatable,
        "source": f.source,
        "status": ("UNKNOWN_USABLE_NOMINAL_MODEL_ONLY" if f.usable_bytes is None
                   else "KNOWN_USABLE"),
        "usable_known": f.usable_bytes is not None,
        "constraint_uses": "nominal_bytes_MODEL_ONLY; usable_bytes=None is preserved",
    }


def exclusions(problem) -> list[dict]:
    out = []
    for name, f in sorted(SPACE_FACTS.items()):
        if name == "ascend.ub":
            continue
        if f.usable_bytes is None or not f.allocatable:
            out.append({"kind": "space_fact_excluded", "space": name,
                        "nominal_bytes": f.nominal_bytes, "usable_bytes": f.usable_bytes,
                        "allocatable": f.allocatable, "source": f.source,
                        "reason": ("not allocatable" if not f.allocatable
                                   else "usable capacity UNKNOWN")})
    for b in problem.buffers:
        if b.space == "ascend.gm":
            out.append({"kind": "problem_buffer_not_charged_to_UB_capacity",
                        "root": b.root, "space": b.space, "bytes": b.bytes_,
                        "versions": b.versions, "fixed": b.fixed})
    return out


def categorize(dep) -> str:
    ev = (dep.evidence or "").lower()
    if "ready" in ev:
        return "ready"
    if "free" in ev:
        return "free"
    if dep.kind == "WAW" or "recompute" in ev or "reuse" in ev:
        return "reuse"
    return "other"


def normalize_problem(problem, side: str) -> dict:
    """Build the SAME full normalized core input for either side.

    side is metadata ONLY: it MUST NOT change core_args. Both adapters receive identical
    normalized inputs. UNKNOWN usable capacity and excluded fields are preserved as-is.
    """
    tasks = list(problem.tasks)
    stable_order = [t.stable_id for t in tasks]
    idx = {name: i for i, name in enumerate(stable_order)}
    latencies = [int(t.latency) for t in tasks]
    iis = [int(t.issue_interval) for t in tasks]
    flags = [PIPE_FLAGS[t.pipe] for t in tasks]

    data_deps, manifest = [], []
    for dep in problem.dependencies:
        u, v = idx[dep.producer], idx[dep.consumer]
        latency = latencies[u]
        tup = [u, v, int(dep.distance), int(latency)]
        cat = categorize(dep)
        data_deps.append(tup)
        manifest.append({"kind": "data_dep", "category": cat, "tuple": tup,
                         "producer": dep.producer, "consumer": dep.consumer,
                         "distance": int(dep.distance), "dependency_kind": dep.kind,
                         "latency_model_cycles": int(latency), "evidence": dep.evidence})

    resource_deps = []
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            if tasks[i].resource == tasks[j].resource:
                resource_deps.append([i, j])
                manifest.append({"kind": "resource_dep", "category": "resource",
                                 "tuple": [i, j], "resource": tasks[i].resource})

    pipe_order_deps = []
    for pipe in ("mte2", "vector", "mte3"):
        group = [i for i, t in enumerate(tasks) if t.pipe == pipe]
        for a, b in zip(group, group[1:]):
            pipe_order_deps.append([a, b])
            manifest.append({"kind": "pipe_order_dep", "category": "pipe_order",
                             "tuple": [a, b], "pipe": pipe})

    owner_exclusion_deps = []
    roots = {}
    for a in problem.accesses:
        if a.space != "ascend.ub":
            continue
        d = roots.setdefault(a.root, {"write": set(), "read": set()})
        d["write" if a.mode == "write" else "read"].add(idx[a.task])
    for root, d in sorted(roots.items()):
        for w in sorted(d["write"]):
            for r in sorted(d["read"]):
                if w == r:
                    continue
                tup = [w, r, latencies[w], latencies[r]]
                owner_exclusion_deps.append(tup)
                manifest.append({"kind": "owner_exclusion_dep", "category": "owner_exclusion",
                                 "tuple": tup, "root": root})

    ub = [b for b in problem.buffers if b.space == "ascend.ub"]
    buffer_sizes = [int(b.bytes_) for b in ub]
    fixed_versions = [int(b.versions) if b.fixed else None for b in ub]
    cap = capacity_facts()
    if cap["nominal_bytes"] is None:
        raise BridgeError("CAPACITY_UNSOURCED", "ascend.ub nominal_bytes is None; no fake capacity")
    memory_groups = [[int(cap["nominal_bytes"]), *range(len(ub))]]
    manifest.append({"kind": "capacity", "category": "capacity", "tuple": memory_groups[0],
                     "usable_bytes": cap["usable_bytes"], "status": cap["status"]})
    for i, b in enumerate(ub):
        manifest.append({"kind": "fixed_version", "category": "fixed_versions",
                         "tuple": [i, fixed_versions[i]], "root": b.root,
                         "bytes": b.bytes_, "fixed": b.fixed})

    eff = bool(problem.policy.get("effective_offset", False))
    manifest.append({"kind": "offset", "category": "offset",
                     "tuple": [int(eff)], "offset_policy": problem.policy.get("offset_policy")})
    manifest.append({"kind": "stage_order_unknown", "category": "stage_order_unknown",
                     "tuple": [], "status": "NO_SOURCE_IN_PROBLEM; NOT FABRICATED"})

    core_args = {
        "num_stages": int(problem.policy["requested_num_stages"]),
        "latencies": latencies,
        "iis": iis,
        "resource_flags": flags,
        "data_deps": data_deps,
        "resource_deps": resource_deps,
        "owner_exclusion_deps": owner_exclusion_deps,
        "pipe_order_deps": pipe_order_deps,
        "buffer_sizes": buffer_sizes,
        "memory_groups": memory_groups,
        "stage_order_deps": [],
        "recalculate_buffer_versions": False,
        "enable_offset": eff,
        "manual_stages": [0] * len(tasks),
        "manual_schedule": True,
        "seed": SEED,
        "verbose": False,
        "fixed_buffer_versions": fixed_versions,
    }
    return {
        "side": side,
        "case_id": problem.problem_id.split(":", 1)[0],
        "problem_id": problem.problem_id,
        "source_digest": problem.source_digest,
        "problem_digest": digest(problem.as_dict()),
        "normalized_digest": digest(core_args),
        "stable_order": stable_order,
        "model_version": MODEL_VERSION,
        "solver_version": core.z3.get_version_string() if core.Z3_AVAILABLE else None,
        "seed": SEED,
        "policy": dict(problem.policy),
        "capacity": cap,
        "capacity_usable_known": cap["usable_known"],
        "excluded_unchanged": exclusions(problem),
        "stage_order_status": "NO_SOURCE_IN_PROBLEM; NOT FABRICATED",
        "constraint_manifest": manifest,
        "core_args": core_args,
        "core_file": str(Path(core.__file__).resolve()),
        "core_sha256": adapters.shared_loop_core_sha256(),
    }


def validate_full_constraints(norm: dict) -> None:
    a = norm["core_args"]
    cap = norm["capacity"]
    if cap["usable_bytes"] is None and not str(cap["status"]).startswith("UNKNOWN"):
        raise BridgeError("UNKNOWN_CAPACITY_FABRICATED", "usable_bytes=None must stay UNKNOWN")
    if not a["memory_groups"] or any(len(g) < 2 or g[0] is None or g[0] < 0
                                     for g in a["memory_groups"]):
        raise BridgeError("CAPACITY_CONSTRAINT_MISSING", "memory_groups absent or invalid")
    if not a["fixed_buffer_versions"] or any(v is None or v < 1
                                             for v in a["fixed_buffer_versions"]):
        raise BridgeError("FIXED_VERSION_CONSTRAINT_MISSING", "fixed versions not pinned")
    if not isinstance(a["enable_offset"], bool):
        raise BridgeError("OFFSET_POLICY_UNKNOWN", "enable_offset is not a bool")
    if int(a["num_stages"]) < 1:
        raise BridgeError("NUM_STAGES_INVALID", "num_stages < 1")
    manifest = norm["constraint_manifest"]
    if not manifest:
        raise BridgeError("CONSTRAINT_MANIFEST_EMPTY", "empty manifest cannot satisfy")
    cats = Counter(m["category"] for m in manifest)
    for cat, needed in REQUIRED_CATEGORIES.items():
        if cats[cat] < needed:
            raise BridgeError("READY_FREE_CONSTRAINT_MISSING",
                              "category %s has %d, need %d" % (cat, cats[cat], needed))
    data_entries = [m for m in manifest if m["kind"] == "data_dep"]
    if len(data_entries) != len(a["data_deps"]):
        raise BridgeError("READY_FREE_CONSTRAINT_MISSING",
                          "manifest has %d data deps, core_args has %d"
                          % (len(data_entries), len(a["data_deps"])))
    for m, tup in zip(data_entries, a["data_deps"]):
        if m["tuple"] != list(tup):
            raise BridgeError("CONSTRAINT_MANIFEST_MISMATCH",
                              "manifest tuple %r != core tuple %r" % (m["tuple"], list(tup)))
    if not a["resource_deps"]:
        raise BridgeError("RESOURCE_CONSTRAINT_MISSING", "resource_deps empty")
    if not a["owner_exclusion_deps"]:
        raise BridgeError("OWNER_EXCLUSION_CONSTRAINT_MISSING", "owner_exclusion_deps empty")
    if not a["pipe_order_deps"]:
        raise BridgeError("PIPE_ORDER_CONSTRAINT_MISSING", "pipe_order_deps empty")
    if norm.get("stage_order_status") != "NO_SOURCE_IN_PROBLEM; NOT FABRICATED":
        raise BridgeError("UNKNOWN_EXCLUSION_FABRICATED", "stage_order UNKNOWN must be preserved")


class _UnknownCheckSolver:
    def __init__(self, real):
        self._real = real

    def set(self, *args, **kwargs):
        return self._real.set(*args, **kwargs)

    def add(self, *args, **kwargs):
        return self._real.add(*args, **kwargs)

    def check(self):
        return self._real_z3.unknown

    def model(self):
        raise RuntimeError("simulated unknown has no model")


class _UnknownZ3:
    """Instrumentation ONLY: force the FIRST Solver.check to return z3.unknown."""
    def __init__(self, real):
        self._real = real
        self._solver_count = 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def Solver(self, ctx=None):
        self._solver_count += 1
        real = self._real.Solver(ctx=ctx)
        proxy = _UnknownCheckSolver(real)
        proxy._real_z3 = self._real
        return proxy


def nonempty_stage_ordinals(result) -> list[int] | None:
    if not result or "start_times" not in result:
        return None
    ii = int(result.get("minimal_II") or 0)
    if ii <= 0:
        return None
    return sorted({int(s) // ii for s in result["start_times"]})


def solve(norm: dict, side: str = "tf", *, z3_missing: bool = False,
          force_solved: bool = False, unknown_check: bool = False,
          skip_validate: bool = False) -> dict:
    if not skip_validate:
        validate_full_constraints(norm)
    if z3_missing and force_solved:
        raise BridgeError("Z3_MISSING_REPORTED_SOLVED",
                          "forged solved claim with z3 unavailable is forbidden")
    adapter = adapters.tf_loop_adapter if side == "tf" else adapters.tl_loop_adapter
    if adapter not in (adapters.tf_loop_adapter, adapters.tl_loop_adapter):
        raise BridgeError("UNKNOWN_ADAPTER", side)
    core_args = copy.deepcopy(norm["core_args"])
    saved_available, saved_z3 = core.Z3_AVAILABLE, core.z3
    z3_available = bool(saved_available) and not z3_missing
    raw, exc = None, None
    try:
        if z3_missing:
            core.Z3_AVAILABLE = False
        if unknown_check:
            core.z3 = _UnknownZ3(saved_z3)
        raw = adapter(core_args, return_status=True)
    except Exception as e:  # noqa: BLE001
        exc = e
    finally:
        core.Z3_AVAILABLE = saved_available
        core.z3 = saved_z3

    checks, core_status = [], None
    if exc is not None:
        msg = str(exc)
        if "z3-solver package is required" in msg or "Z3_AVAILABLE" in msg:
            state, reason = "unknown", "z3_unavailable"
            plan_status = "unavailable"
        elif "solver returned unknown" in msg:
            state, reason = "unknown", "z3_check_unknown_preserved"
            plan_status = "unavailable"
        elif ("Manual schedule constraints are infeasible" in msg
              or "fixed buffer version" in msg):
            state, reason = "unsat", "full_constraints_infeasible"
            plan_status = "refused"
        else:
            state, reason = "unknown", "core_exception"
            plan_status = "unavailable"
    else:
        core_status = raw.get("status")
        checks = list(raw.get("check_results") or [])
        if core_status == "fallback":
            if force_solved:
                raise BridgeError("FALLBACK_REPORTED_SOLVED",
                                  "explicit serial fallback cannot be reported as solved")
            state, reason = "fallback", raw.get("fallback_reason")
            plan_status = "fallback"
        elif core_status == "sat" and "sat" in checks:
            if not z3_available:
                raise BridgeError("Z3_MISSING_REPORTED_SOLVED",
                                  "sat without z3 available is a forged solved claim")
            state, reason = "sat", "z3_sat_on_full_constraints"
            plan_status = "solved"
        else:
            state, reason = "unknown", "unrecognised_core_outcome"
            plan_status = "unavailable"
    if not z3_available and (state == "sat" or plan_status == "solved" or force_solved):
        raise BridgeError("Z3_MISSING_REPORTED_SOLVED",
                          "no solved claim is allowed when z3 is unavailable")

    result_digest = digest(raw) if raw is not None else None
    trace = {
        "side": side,
        "adapter": adapter.__module__ + "." + adapter.__name__,
        "core_file": str(Path(core.__file__).resolve()),
        "core_sha256": adapters.shared_loop_core_sha256(),
        "core_function": core.z3_schedule_loop_python.__module__ + "."
                         + core.z3_schedule_loop_python.__name__,
        "args_sha256": digest(core_args),
        "z3_available": z3_available,
        "z3_version": saved_z3.get_version_string() if saved_available else None,
        "check_results": checks,
        "core_status": core_status,
        "exception": None if exc is None else type(exc).__name__ + ": " + str(exc)[:300],
        "forced_solved_requested": bool(force_solved),
        "simulated_unknown_check": bool(unknown_check),
    }
    return {
        "solver_state": state,
        "plan_status": plan_status,
        "state_reason": reason,
        "solved_claim": plan_status == "solved",
        "core_result": raw,
        "result_digest": result_digest,
        "check_results": checks,
        "requested_num_stages": core_args["num_stages"],
        "nonempty_stage_ordinals_observed": nonempty_stage_ordinals(raw),
        "stage_claim": "observed ordinals ONLY; NOT a claim of N non-empty stages",
        "trace": trace,
        "normalized_digest": norm["normalized_digest"],
    }


def core_def_paths() -> list[str]:
    out = []
    for base in ("src", "tools"):
        for f in (TF / base).rglob("*.py"):
            try:
                tree = ast.parse(f.read_text())
            except Exception:
                continue
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))                         and node.name == "z3_schedule_loop_python":
                    out.append(str(f.relative_to(TF)))
    return out


def upstream_stage0_equivalence(norm: dict) -> dict:
    """Compare the DEFAULT core path (no 04-07/08 extensions) with upstream TileLang."""
    args = copy.deepcopy(norm["core_args"])
    args.pop("fixed_buffer_versions", None)
    args.pop("return_status", None)
    args["recalculate_buffer_versions"] = True
    args["manual_schedule"] = True
    args["manual_stages"] = [0] * len(args["latencies"])
    args["enable_offset"] = False
    args["seed"] = SEED
    args["verbose"] = False
    vendor = core.z3_schedule_loop_python(**args)
    code = (
        "import importlib.util,json;"
        "spec=importlib.util.spec_from_file_location('upz3', %r);"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
        "r=m.z3_schedule_loop_python(**json.loads(%r));"
        "print('RES='+json.dumps([list(r[0]),list(r[1]),int(r[2])]))"
        % (str(UPSTREAM), json.dumps(args))
    )
    env = dict(os.environ, PYTHONPATH=str(SRC) + os.pathsep + str(TARGETS),
               TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=env, timeout=300)
    out = {"vendor_default_path": [list(vendor[0]), list(vendor[1]), int(vendor[2])],
           "upstream_path": str(UPSTREAM), "upstream_git_head": "8e2d3c823b0c3a01d2ce68d47dabfe98e35cc22b",
           "returncode": p.returncode, "match": None, "status": "UNRUN"}
    if p.returncode == 0:
        line = next((l for l in p.stdout.splitlines() if l.startswith("RES=")), None)
        if line:
            up = json.loads(line[4:])
            out["upstream_result"] = up
            out["match"] = out["vendor_default_path"] == up
            out["status"] = "PASS" if out["match"] else "MISMATCH"
        else:
            out["status"] = "NO_RESULT_LINE"
    else:
        out["status"] = "UNRUN"
        out["stderr_tail"] = p.stderr.strip()[-400:]
    return out


def mutate_problem(problem, kind: str, value):
    q = copy.deepcopy(problem)
    if kind == "num_stages":
        q.policy["requested_num_stages"] = int(value)
    elif kind == "x_versions":
        q.policy["x_versions"] = int(value)
        q.buffers = [dataclasses.replace(b, versions=int(value)) if b.root == "x_ub" else b
                     for b in q.buffers]
        q.dependencies = [dataclasses.replace(d, distance=int(value)) if "X free" in d.evidence
                          else d for d in q.dependencies]
    elif kind == "y_versions":
        q.policy["y_versions"] = int(value)
        q.buffers = [dataclasses.replace(b, versions=int(value)) if b.root == "y_ub" else b
                     for b in q.buffers]
        q.dependencies = [dataclasses.replace(d, distance=int(value)) if "Y free" in d.evidence
                          else d for d in q.dependencies]
    elif kind == "offset":
        q.policy["effective_offset"] = bool(value)
        q.policy["offset_policy"] = "explicit_true" if value else "explicit_false"
    else:
        raise BridgeError("UNKNOWN_MUTATION", kind)
    return q


def budget_variants(problem, base_norm: dict, baseline_res: dict) -> list[dict]:
    out = []
    baseline = base_norm["core_args"]
    for kind, value in (("num_stages", 3), ("x_versions", 3),
                        ("y_versions", 1), ("offset", True)):
        q = mutate_problem(problem, kind, value)
        norm = normalize_problem(q, "budget")
        res = solve(norm, "tf")
        old = baseline[{"num_stages": "num_stages",
                         "x_versions": "fixed_buffer_versions",
                         "y_versions": "fixed_buffer_versions",
                         "offset": "enable_offset"}[kind]]
        new = norm["core_args"][{"num_stages": "num_stages",
                                 "x_versions": "fixed_buffer_versions",
                                 "y_versions": "fixed_buffer_versions",
                                 "offset": "enable_offset"}[kind]]
        out.append({"field": kind, "from": old, "to": new,
                    "policy": norm["policy"],
                    "normalized_digest": norm["normalized_digest"],
                    "normalized_digest_changed": norm["normalized_digest"] != base_norm["normalized_digest"],
                    "solver_state": res["solver_state"], "plan_status": res["plan_status"],
                    "result_digest": res["result_digest"],
                    "result_changed": res["result_digest"] != baseline_res["result_digest"],
                    "nonempty_stage_ordinals_observed": res["nonempty_stage_ordinals_observed"],
                    "stage_claim": res["stage_claim"]})
    return out


def scenario_norm(norm: dict, *, capacity_zero: bool = False, manual: bool | None = None):
    q = copy.deepcopy(norm)
    if capacity_zero:
        for g in q["core_args"]["memory_groups"]:
            g[0] = 0
    if manual is not None:
        q["core_args"]["manual_schedule"] = bool(manual)
    return q


def state_table(norm: dict) -> list[dict]:
    rows = []
    sat = solve(norm, "tf")
    rows.append({"scenario": "sat_full_constraints", **_state_row(sat)})
    unsat = solve(scenario_norm(norm, capacity_zero=True, manual=True), "tf")
    rows.append({"scenario": "unsat_manual_infeasible", **_state_row(unsat)})
    unk = solve(norm, "tf", z3_missing=True)
    rows.append({"scenario": "unknown_z3_unavailable", **_state_row(unk)})
    unk2 = solve(norm, "tf", unknown_check=True)
    rows.append({"scenario": "unknown_check_unknown_preserved", **_state_row(unk2)})
    fb = solve(scenario_norm(norm, capacity_zero=True, manual=False), "tf")
    rows.append({"scenario": "fallback_explicit_serial", **_state_row(fb)})
    return rows


def _state_row(res: dict) -> dict:
    return {"solver_state": res["solver_state"], "plan_status": res["plan_status"],
            "solved_claim": res["solved_claim"], "check_results": res["check_results"],
            "state_reason": res["state_reason"],
            "result_digest": res["result_digest"],
            "trace": res["trace"]}


def run_mutation(name: str) -> dict:
    """One named mutation; returns a report. Process exit 0 means the mutation was KILLED."""
    problem = build_problem("prefill_dyn")
    base = normalize_problem(problem, "tf")
    baseline = solve(base, "tf")
    out = {"name": name, "killed": False, "named_code": None}

    def expect_refusal(fn, code):
        try:
            fn()
        except BridgeError as e:
            out["named_code"] = e.code
            out["killed"] = e.code == code
            out["detail"] = e.message
            return
        out["named_code"] = "NO_REFUSAL"
        out["killed"] = False

    if name == "baseline_positive":
        out["killed"] = baseline["solver_state"] == "sat" and baseline["solved_claim"]
        out["named_code"] = "PASS"
    elif name == "probe_can_fail":
        # Prove the validator itself can fail: with it neutralised, a dropped-capacity
        # input is no longer refused. This is a probe of the probe, not a solved claim.
        bad = copy.deepcopy(base)
        bad["core_args"]["memory_groups"] = []
        old = globals()["validate_full_constraints"]
        try:
            globals()["validate_full_constraints"] = lambda n: None
            try:
                solve(bad, "tf")
                out["killed"] = True
                out["named_code"] = "VALIDATOR_NEUTRALISED_ACCEPTS_DROPPED_CAPACITY"
            except BridgeError:
                out["killed"] = False
                out["named_code"] = "VALIDATOR_STILL_LIVE"
        finally:
            globals()["validate_full_constraints"] = old
    elif name == "probe_ready_free_can_fail":
        bad = copy.deepcopy(base)
        bad["core_args"]["data_deps"] = []
        old = globals()["validate_full_constraints"]
        try:
            globals()["validate_full_constraints"] = lambda n: None
            try:
                solve(bad, "tf")
                out["killed"] = True
                out["named_code"] = "READY_FREE_VALIDATOR_NEUTRALISED"
            except BridgeError:
                out["killed"] = False
                out["named_code"] = "READY_FREE_VALIDATOR_STILL_LIVE"
        finally:
            globals()["validate_full_constraints"] = old
    elif name == "empty_manifest":
        bad = copy.deepcopy(base)
        bad["constraint_manifest"] = []
        expect_refusal(lambda: validate_full_constraints(bad), "CONSTRAINT_MANIFEST_EMPTY")
    elif name == "drop_capacity":
        bad = copy.deepcopy(base)
        bad["core_args"]["memory_groups"] = []
        expect_refusal(lambda: validate_full_constraints(bad), "CAPACITY_CONSTRAINT_MISSING")
    elif name == "drop_ready_free":
        bad = copy.deepcopy(base)
        drop = next(i for i, m in enumerate(bad["constraint_manifest"])
                    if m["category"] == "free")
        # Remove the corresponding core tuple only; the manifest still demands it.
        bad["core_args"]["data_deps"].pop(drop)
        expect_refusal(lambda: validate_full_constraints(bad), "READY_FREE_CONSTRAINT_MISSING")
    elif name == "force_solved_missing_z3":
        expect_refusal(lambda: solve(base, "tf", z3_missing=True, force_solved=True),
                       "Z3_MISSING_REPORTED_SOLVED")
    elif name == "fallback_as_solved":
        fb = scenario_norm(base, capacity_zero=True, manual=False)
        expect_refusal(lambda: solve(fb, "tf", force_solved=True), "FALLBACK_REPORTED_SOLVED")
    elif name == "extra_task":
        q = copy.deepcopy(problem)
        q.tasks = list(q.tasks) + [Task("extra_vf", "vector", "vector", 7, 1,
                                        "main_compute", "row_pipeline")]
        norm2 = normalize_problem(q, "tf")
        res2 = solve(norm2, "tf")
        changed = (norm2["normalized_digest"] != base["normalized_digest"]
                   and res2["result_digest"] != baseline["result_digest"])
        out["killed"] = changed
        out["named_code"] = "EXTRA_TASK_CHANGED_SOLVING" if changed else "EXTRA_TASK_IGNORED"
        out["baseline_result_digest"] = baseline["result_digest"]
        out["mutated_result_digest"] = res2["result_digest"]
        out["mutated_state"] = res2["solver_state"]
    elif name == "distance_zero":
        q = copy.deepcopy(problem)
        q.dependencies = [dataclasses.replace(d, distance=0) if d.kind == "WAW" else d
                          for d in q.dependencies]
        norm2 = normalize_problem(q, "tf")
        res2 = solve(norm2, "tf")
        changed = (norm2["normalized_digest"] != base["normalized_digest"]
                   and (res2["result_digest"] != baseline["result_digest"]
                        or res2["solver_state"] != baseline["solver_state"]))
        out["killed"] = changed
        out["named_code"] = "DISTANCE_CHANGED_SOLVING" if changed else "DISTANCE_IGNORED"
        out["baseline_state"] = baseline["solver_state"]
        out["mutated_state"] = res2["solver_state"]
        out["baseline_result_digest"] = baseline["result_digest"]
        out["mutated_result_digest"] = res2["result_digest"]
    elif name == "capacity_zero_fallback":
        res = solve(scenario_norm(base, capacity_zero=True, manual=False), "tf")
        out["killed"] = res["solver_state"] == "fallback" and not res["solved_claim"]
        out["named_code"] = "FALLBACK_DISTINCT" if out["killed"] else "FALLBACK_COLLAPSED"
    elif name == "unknown_check_preserved":
        res = solve(base, "tf", unknown_check=True)
        out["killed"] = res["solver_state"] == "unknown" and not res["solved_claim"]
        out["named_code"] = "UNKNOWN_PRESERVED" if out["killed"] else "UNKNOWN_COLLAPSED"
    elif name == "missing_z3_not_solved":
        res = solve(base, "tf", z3_missing=True)
        out["killed"] = res["solver_state"] == "unknown" and not res["solved_claim"]
        out["named_code"] = "MISSING_Z3_UNKNOWN" if out["killed"] else "MISSING_Z3_SOLVED"
    elif name == "unsat_manual_not_solved":
        res = solve(scenario_norm(base, capacity_zero=True, manual=True), "tf")
        out["killed"] = res["solver_state"] == "unsat" and not res["solved_claim"]
        out["named_code"] = "UNSAT_DISTINCT" if out["killed"] else "UNSAT_COLLAPSED"
    else:
        raise BridgeError("UNKNOWN_MUTATION", name)
    return out


MUTATIONS = ("baseline_positive", "probe_can_fail", "probe_ready_free_can_fail",
             "empty_manifest", "drop_capacity", "drop_ready_free",
             "force_solved_missing_z3", "fallback_as_solved", "extra_task", "distance_zero",
             "capacity_zero_fallback", "unknown_check_preserved",
             "missing_z3_not_solved", "unsat_manual_not_solved")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + chr(10))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mutation", default=None)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.mutation:
        res = run_mutation(args.mutation)
        write_json(out / (args.mutation + ".json"), res)
        print(json.dumps(res, sort_keys=True))
        return 0 if res["killed"] else 1

    # Fresh evidence: no stale files are read as results.
    for child in out.iterdir():
        if child.is_dir():
            import shutil
            shutil.rmtree(child)
        else:
            child.unlink()

    core_sha = adapters.shared_loop_core_sha256()
    defs = core_def_paths()
    if defs != ["src/tilefoundry/vendor/tilelang_z3/core.py"]:
        raise BridgeError("MULTIPLE_SOLVER_DEFINITIONS", str(defs))
    meta = {"task": "04-07/04-08 two adapters, ONE physical Z3 core",
            "mode": "CPU_PLANNING_ONLY", "bound_04_10_commit": "2e938af",
            "core_file": str(Path(core.__file__).resolve()), "core_sha256": core_sha,
            "core_loop_definitions": defs,
            "tf_loop_adapter_module": adapters.tf_loop_adapter.__module__,
            "tl_loop_adapter_module": adapters.tl_loop_adapter.__module__,
            "shared_loop_core_sha256": core_sha,
            "seed": SEED, "model_version": MODEL_VERSION,
            "z3_version": core.z3.get_version_string() if core.Z3_AVAILABLE else None,
            "states": list(STATES),
            "no_device": True, "p4_only": True,
            "no_p5_or_performance_claim": True}
    write_json(out / "META.json", meta)

    case_reports = {}
    for case_id in CASES:
        problem = build_problem(case_id)
        tf_norm = normalize_problem(problem, "tf")
        tl_norm = normalize_problem(problem, "tl")
        if tf_norm["core_args"] != tl_norm["core_args"]:
            raise BridgeError("NORMALIZED_INPUT_DIVERGED", case_id)
        validate_full_constraints(tf_norm)
        write_json(out / ("normalized_inputs/tf_" + case_id + ".json"), tf_norm)
        write_json(out / ("normalized_inputs/tl_" + case_id + ".json"), tl_norm)
        tf_res = solve(tf_norm, "tf")
        tl_res = solve(tl_norm, "tl")
        repeat = solve(tf_norm, "tf")
        same_sides = (tf_res["core_result"] == tl_res["core_result"]
                      and tf_res["result_digest"] == tl_res["result_digest"])
        repeat_ok = (tf_res["core_result"] == repeat["core_result"]
                     and tf_res["result_digest"] == repeat["result_digest"])
        case_reports[case_id] = {
            "problem_id": problem.problem_id,
            "source_digest": problem.source_digest,
            "normalized_digest": tf_norm["normalized_digest"],
            "tf_normalized_digest": tf_norm["normalized_digest"],
            "tl_normalized_digest": tl_norm["normalized_digest"],
            "core_sha256": core_sha,
            "capacity_usable_known": tf_norm["capacity_usable_known"],
            "capacity_status": tf_norm["capacity"]["status"],
            "excluded_unchanged": tf_norm["excluded_unchanged"],
            "stage_order_status": tf_norm["stage_order_status"],
            "tf_result": tf_res, "tl_result": tl_res, "repeat_result": repeat,
            "both_adapters_identical": same_sides,
            "repeat_solve_identical": repeat_ok,
        }
        write_json(out / ("cases/" + case_id + ".json"), case_reports[case_id])

    base_problem = build_problem("prefill_dyn")
    base_norm = normalize_problem(base_problem, "tf")
    baseline = solve(base_norm, "tf")
    write_json(out / "state_table.json", state_table(base_norm))
    write_json(out / "budget_variants.json", budget_variants(base_problem, base_norm, baseline))
    write_json(out / "repeat_solves.json",
               {"baseline": baseline, "tf": case_reports["prefill_dyn"]["tf_result"],
                "tl": case_reports["prefill_dyn"]["tl_result"],
                "repeat": case_reports["prefill_dyn"]["repeat_result"],
                "both_adapters_identical": case_reports["prefill_dyn"]["both_adapters_identical"],
                "repeat_solve_identical": case_reports["prefill_dyn"]["repeat_solve_identical"]})
    write_json(out / "upstream_stage0_equivalence.json", upstream_stage0_equivalence(base_norm))
    write_json(out / "solver_call_trace.json",
               {cid: {"tf": r["tf_result"]["trace"], "tl": r["tl_result"]["trace"]}
                for cid, r in case_reports.items()})
    write_json(out / "normalized_digest_table.json",
               {cid: {"tf": r["tf_normalized_digest"], "tl": r["tl_normalized_digest"],
                      "identical": r["tf_normalized_digest"] == r["tl_normalized_digest"]}
                for cid, r in case_reports.items()})

    # Evidence self-checks: these are assertions produced by THIS run.
    checks = {
        "one_loop_core_definition": defs == ["src/tilefoundry/vendor/tilelang_z3/core.py"],
        "both_adapters_same_module": adapters.tf_loop_adapter.__module__
                                     == adapters.tl_loop_adapter.__module__,
        "shared_core_hash_matches_file": core_sha == sha_file(Path(core.__file__).resolve()),
        "all_cases_both_adapters_identical": all(r["both_adapters_identical"]
                                                for r in case_reports.values()),
        "all_cases_repeat_identical": all(r["repeat_solve_identical"]
                                         for r in case_reports.values()),
        "all_cases_tf_tl_normalized_equal": all(r["tf_normalized_digest"] == r["tl_normalized_digest"]
                                               for r in case_reports.values()),
        "capacity_usable_unknown_preserved": all(not r["capacity_usable_known"]
                                                 for r in case_reports.values()),
        "baseline_sat": baseline["solver_state"] == "sat",
    }
    write_json(out / "SELF_CHECKS.json", checks)
    print(json.dumps({"core_sha256": core_sha, "cases": len(case_reports),
                      "checks": checks}, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BridgeError as e:
        print("REFUSED " + e.code + ": " + e.message)
        raise SystemExit(2)
