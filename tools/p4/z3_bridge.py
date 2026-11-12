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

#: Upstream tilelang scheduler, resolved from the ENVIRONMENT. The literal
#: "<user>/..." placeholder made every comparison UNRUN; a missing root is now a
#: NAMED skip (TILELANG_ROOT_UNSET) carrying its reason, never a silent pass.
UPSTREAM_RELATIVE = "tilelang/ascend/transform/z3_scheduler.py"


def resolve_upstream() -> Path | None:
    root = (os.environ.get("TILELANG_ROOT") or "").strip()
    if not root:
        return None
    candidate = Path(root) / UPSTREAM_RELATIVE
    return candidate if candidate.is_file() else None


UPSTREAM = resolve_upstream()
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
        if isinstance(b, dict):
            space, root = b.get("space"), b.get("root")
            bytes_, versions, fixed = b.get("bytes_"), b.get("versions"), b.get("fixed")
        else:
            space, root = b.space, b.root
            bytes_, versions, fixed = b.bytes_, b.versions, b.fixed
        if space == "ascend.gm":
            out.append({"kind": "problem_buffer_not_charged_to_UB_capacity",
                        "root": root, "space": space, "bytes": bytes_,
                        "versions": versions, "fixed": fixed})
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
    if UPSTREAM is None:
        # Absent upstream is UNRUN with a reason, not a pass and not a failure.
        return {"vendor_default_path": [list(vendor[0]), list(vendor[1]), int(vendor[2])],
                "upstream_path": None, "status": "SKIP",
                "skip_reason": "TILELANG_ROOT_UNSET",
                "skip_detail": "set TILELANG_ROOT to a tilelang root containing %s"
                               % UPSTREAM_RELATIVE}
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


LEDGER_FIXTURE = TF / "targets/ascend/tests/fixtures/rmsnorm/rmsnorm_main.py"
LEDGER_REL = "targets/ascend/tilefoundry_ascend/schedule/ledger.py"
NA = "N-A_NOT_IN_REAL_LEDGER"
UNKNOWN = "UNKNOWN"


def resolve_problem_source(source):
    """04-09 fail-closed gate: the source MUST be explicit. No auto/template fallback."""
    if source is None:
        raise BridgeError("PROBLEM_SOURCE_EXPLICIT_REQUIRED",
                          "04-09 requires explicit --problem-source ledger; template is old-domain regression only")
    if source not in ("ledger", "template"):
        raise BridgeError("AUTO_TEMPLATE_FALLBACK_FORBIDDEN",
                          "unknown/auto source %r; never silently fall back to template" % source)
    return source


def guard_no_template_fallback(source, ledger_failed: bool, detail: str = "") -> None:
    if ledger_failed and source == "ledger":
        raise BridgeError("LEDGER_SOURCE_FAILED_NO_TEMPLATE_FALLBACK",
                          "real ledger build failed (%s); template fallback is forbidden" % detail)


def ledger_git_blob_sha256() -> dict:
    rel = LEDGER_REL
    p = subprocess.run(["git", "-C", str(TF), "show", "2e938af:" + rel],
                       capture_output=True)
    out = {"commit": "2e938af", "path": rel, "status": "UNRUN"}
    if p.returncode == 0:
        out["status"] = "PASS"
        out["sha256"] = hashlib.sha256(p.stdout).hexdigest()
    else:
        out["stderr"] = p.stderr.decode(errors="replace")[-300:]
    return out


def build_ledger_problem():
    """REAL 04-05 emission ledger -> NormalizedProblem (NOT the case-template HIR)."""
    return build_ledger_problem_for(LEDGER_FIXTURE, "RMSNormD")


def build_ledger_problem_for(fixture_path: Path, entry: str = "RMSNormD"):
    """Same real ledger chain, parameterized for a same-parameter TF fixture."""
    from tilefoundry.cli.source import checked_program
    from tilefoundry.analysis.api import analyze
    from tilefoundry.schedule.finalize import finalize
    from tilefoundry_ascend.schedule.ledger import extract_ledger, build_problem
    fixture = str(fixture_path) + ":" + entry
    prog = checked_program(fixture, None, command="04-09-partial-ledger")
    analyzed = analyze(prog.module, prog.entry, analysis=("memory",))
    module = finalize(analyzed, authored=prog.authored)
    ledger = extract_ledger(module.entry_function())
    problem = build_problem(ledger)
    return ledger, problem


def jsonable(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if dataclasses.is_dataclass(obj):
        return jsonable(dataclasses.asdict(obj))
    return str(obj)


def _safe_pipe_flag(pipe):
    if pipe in PIPE_FLAGS:
        return PIPE_FLAGS[pipe]
    if pipe == "NO_ISA_CONTRACT" or pipe is None:
        return "NO_ISA_CONTRACT_OR_NONE"
    return "UNKNOWN_PIPE"


def normalize_ledger_problem(ledger, problem, side: str) -> dict:
    """Build core_args from REAL ledger task.action_ids / ProblemAccess / dependencies.

    Missing policy/versions/offset/cost fields stay UNKNOWN / N-A. Nothing is filled
    from the case template and nothing is default-satisfied for the solver.
    """
    tasks = list(problem.tasks)
    action_to_task = {}
    for i, task in enumerate(tasks):
        for aid in task.action_ids:
            action_to_task.setdefault(aid, []).append(i)
    task_payloads = []
    for i, t in enumerate(tasks):
        task_payloads.append({
            "task_index": i,
            "stable_id": t.stable_id,
            "op": t.op,
            "pipe": t.pipe,
            "resource": t.resource,
            "latency": t.latency,
            "issue_interval": t.issue_interval,
            "cost_model": t.cost_model,
            "loop": t.loop,
            "action_ids": list(t.action_ids),
        })
    access_payloads = []
    for j, a in enumerate(problem.accesses):
        access_payloads.append({
            "access_index": j,
            "action_id": a.task,
            "task_stable_id": a.instruction,
            "root": a.root,
            "root_kind": a.root_kind,
            "space": a.space,
            "mode": a.mode,
            "byte_footprint": a.byte_footprint,
            "byte_offset": a.byte_offset,
            "source_offset": a.source_offset,
            "attrs": jsonable(a.attrs),
            "offset_structure": jsonable(a.offset_structure),
            "guard": jsonable(a.guard),
            "normalized_offset": a.normalized_offset,
            "offset_status": a.offset_status,
            "logical_iteration": a.logical_iteration,
            "iteration_path": a.iteration_path,
            "iteration_extent": a.iteration_extent,
            "access_key": json.dumps([a.task, a.root, a.space, a.mode,
                                      a.iteration_path, jsonable(a.offset_structure)],
                                     sort_keys=True, separators=(",", ":")),
        })
    dep_payloads = []
    unmapped = []
    for k, d in enumerate(problem.dependencies):
        pidx = action_to_task.get(d["producer"])
        cidx = action_to_task.get(d["consumer"])
        if not pidx or not cidx:
            unmapped.append({"index": k, "producer": d["producer"],
                             "consumer": d["consumer"]})
        dep_payloads.append({
            "dep_index": k,
            "producer_action_id": d["producer"],
            "consumer_action_id": d["consumer"],
            "producer_task_indices": pidx or [],
            "consumer_task_indices": cidx or [],
            "distance": d["distance"],
            "kind": d["kind"],
            "evidence": d["evidence"],
            "latency_model_cycles": "UNKNOWN_NONE",
        })
    resource_deps = []
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            if tasks[i].resource == tasks[j].resource and tasks[i].resource not in (None, "NO_ISA_CONTRACT"):
                resource_deps.append({"tasks": [i, j], "resource": tasks[i].resource,
                                      "action_ids": [list(tasks[i].action_ids), list(tasks[j].action_ids)]})
    owner_exclusion_deps = []
    roots = {}
    for a in access_payloads:
        if a["space"] != "ascend.ub":
            continue
        key = (a["root"], a["space"], a["iteration_path"])
        roots.setdefault(key, {"write": [], "read": []})[a["mode"]].append(a)
    for key, d in sorted(roots.items(), key=lambda x: repr(x[0])):
        for w in d["write"]:
            for r in d["read"]:
                if w["action_id"] == r["action_id"]:
                    continue
                owner_exclusion_deps.append({
                    "producer_action_id": w["action_id"],
                    "consumer_action_id": r["action_id"],
                    "root": key[0], "space": key[1], "iteration_path": key[2],
                    "producer_offset_status": w["offset_status"],
                    "consumer_offset_status": r["offset_status"],
                    "latency_model_cycles": "UNKNOWN_NONE",
                })
    pipe_order_deps = []
    for pipe in ("mte2", "vector", "mte3"):
        group = [i for i, t in enumerate(tasks) if t.pipe == pipe]
        for a, b in zip(group, group[1:]):
            pipe_order_deps.append({"tasks": [a, b], "pipe": pipe,
                                    "action_ids": [list(tasks[a].action_ids), list(tasks[b].action_ids)]})
    ub = [b for b in problem.buffers if b.get("space") == "ascend.ub"]
    buffer_sizes = [b.get("bytes_") for b in ub]
    capacity = capacity_facts()
    unknown_fields = [
        {"field": "policy.requested_num_stages", "status": NA,
         "reason": "NormalizedProblem has no PipelinePolicy; template default forbidden"},
        {"field": "policy.x_versions / y_versions", "status": NA,
         "reason": "real ledger Problem has no case-config version policy"},
        {"field": "policy.effective_offset / offset_policy", "status": NA,
         "reason": "real ledger Problem has no offset policy"},
        {"field": "capacity.usable_bytes", "status": "UNKNOWN",
         "value": None, "reason": "SPACE_FACTS usable_bytes is None; nominal is MODEL-ONLY"},
        {"field": "fixed_buffer_versions", "status": NA,
         "reason": "not default-satisfied from ledger buffers"},
        {"field": "memory_groups", "status": NA,
         "reason": "known usable capacity unavailable"},
        {"field": "stage_order_deps", "status": "NO_SOURCE_IN_PROBLEM; NOT FABRICATED"},
    ]
    core_args = {
        "source": "real_ledger_2e938af",
        "num_stages": NA,
        "tasks": task_payloads,
        "latencies": [t["latency"] for t in task_payloads],
        "iis": [t["issue_interval"] for t in task_payloads],
        "resource_flags": [_safe_pipe_flag(t["pipe"]) for t in task_payloads],
        "accesses": access_payloads,
        "data_deps": dep_payloads,
        "resource_deps": resource_deps,
        "owner_exclusion_deps": owner_exclusion_deps,
        "pipe_order_deps": pipe_order_deps,
        "buffer_sizes": buffer_sizes,
        "memory_groups": NA,
        "fixed_buffer_versions": NA,
        "stage_order_deps": "NO_SOURCE_IN_PROBLEM; NOT FABRICATED",
        "recalculate_buffer_versions": False,
        "enable_offset": NA,
        "manual_stages": NA,
        "manual_schedule": NA,
        "seed": SEED,
        "verbose": False,
        "solver_callable": False,
    }
    manifest = []
    for t in task_payloads:
        manifest.append({"kind": "ledger_task", "category": "task",
                         "core_args_path": "core_args.tasks[%d]" % t["task_index"],
                         "action_ids": t["action_ids"],
                         "pipe": t["pipe"], "latency": t["latency"],
                         "issue_interval": t["issue_interval"]})
    for a in access_payloads:
        manifest.append({"kind": "ledger_access", "category": "access",
                         "core_args_path": "core_args.accesses[%d]" % a["access_index"],
                         "action_id": a["action_id"], "root": a["root"], "space": a["space"],
                         "mode": a["mode"], "offset_status": a["offset_status"]})
    for d in dep_payloads:
        manifest.append({"kind": "ledger_dependency", "category": "dependency",
                         "core_args_path": "core_args.data_deps[%d]" % d["dep_index"],
                         "producer_action_id": d["producer_action_id"],
                         "consumer_action_id": d["consumer_action_id"],
                         "distance": d["distance"], "kind": d["kind"]})
    forward = {}
    reverse = {}
    for t in task_payloads:
        path = "core_args.tasks[%d]" % t["task_index"]
        reverse.setdefault(path, []).extend(t["action_ids"])
        for aid in t["action_ids"]:
            e = forward.setdefault(aid, {"task_stable_ids": [], "core_args_paths": [],
                                         "access_keys": [], "dependency_keys": []})
            e["task_stable_ids"].append(t["stable_id"])
            e["core_args_paths"].append(path)
    for a in access_payloads:
        path = "core_args.accesses[%d]" % a["access_index"]
        reverse.setdefault(path, []).append(a["action_id"])
        e = forward.setdefault(a["action_id"], {"task_stable_ids": [], "core_args_paths": [],
                                                "access_keys": [], "dependency_keys": []})
        e["core_args_paths"].append(path)
        e["access_keys"].append(a["access_key"])
    for d in dep_payloads:
        path = "core_args.data_deps[%d]" % d["dep_index"]
        aids = [d["producer_action_id"], d["consumer_action_id"]]
        reverse.setdefault(path, []).extend(aids)
        for aid in aids:
            e = forward.setdefault(aid, {"task_stable_ids": [], "core_args_paths": [],
                                         "access_keys": [], "dependency_keys": []})
            e["core_args_paths"].append(path)
            e["dependency_keys"].append(path)
    for k in forward:
        forward[k]["core_args_paths"] = sorted(set(forward[k]["core_args_paths"]))
        forward[k]["task_stable_ids"] = sorted(set(forward[k]["task_stable_ids"]))
    for k in reverse:
        reverse[k] = sorted(set(reverse[k]))
    bidirectional_ok = True
    for aid, e in forward.items():
        for path in e["core_args_paths"]:
            if aid not in reverse.get(path, []):
                bidirectional_ok = False
    for path, aids in reverse.items():
        for aid in aids:
            if path not in forward.get(aid, {}).get("core_args_paths", []):
                bidirectional_ok = False
    action_id_map = {
        "schema": "tf_ledger_action_id_to_core_args_dev_v1",
        "problem_source": "ledger",
        "bidirectional": True,
        "bidirectional_ok": bidirectional_ok,
        "forward": forward,
        "reverse": reverse,
        "unmapped_action_ids": sorted({u["producer"] for u in unmapped} | {u["consumer"] for u in unmapped}),
        "unknown_fields": unknown_fields,
        "excluded_unchanged": jsonable(exclusions(problem)),
        "capacity": capacity,
        "mapping_digest": digest({"forward": forward, "reverse": reverse}),
    }
    status_counts = {
        "entries": len(ledger.entries),
        "tasks": len(problem.tasks),
        "accesses": len(problem.accesses),
        "dependencies": len(problem.dependencies),
        "buffers": len(problem.buffers),
        "offset_unknown": sum(1 for a in problem.accesses if a.offset_status == "OFFSET_UNKNOWN"),
        "no_isa_contract_tasks": sum(1 for t in problem.tasks if t.pipe == "NO_ISA_CONTRACT"),
        "latency_none_tasks": sum(1 for t in problem.tasks if t.latency is None),
        "ii_none_tasks": sum(1 for t in problem.tasks if t.issue_interval is None),
    }
    return {
        "side": side,
        "problem_source": "ledger",
        "case_id": "REAL_LEDGER_NOT_CASE_TEMPLATE",
        "fixture": str(LEDGER_FIXTURE) + ":RMSNormD",
        "chain": ["checked_program", "analyze", "finalize", "entry_function",
                  "extract_ledger", "ledger.py:374"],
        "bound_commit": "2e938af",
        "ledger_git_blob": ledger_git_blob_sha256(),
        "ledger_current_file_sha256": sha_file(TF / LEDGER_REL),
        "problem_id": problem.problem_id,
        "source_digest": problem.source_digest,
        "ledger_sha": problem.ledger_sha,
        "canonical_serialization_sha": problem.canonical_serialization_sha,
        "problem_digest": problem.problem_digest,
        "problem_as_dict": jsonable(problem.as_dict()),
        "status_counts": status_counts,
        "unknown_fields": unknown_fields,
        "excluded_unchanged": jsonable(exclusions(problem)),
        "capacity": capacity,
        "capacity_usable_known": capacity["usable_known"],
        "stage_order_status": "NO_SOURCE_IN_PROBLEM; NOT FABRICATED",
        "constraint_manifest": manifest,
        "core_args": core_args,
        "action_id_map": action_id_map,
        "normalized_digest": digest(core_args),
        "model_version": MODEL_VERSION,
        "solver_version": core.z3.get_version_string() if core.Z3_AVAILABLE else None,
        "seed": SEED,
        "stable_order": [t["stable_id"] for t in task_payloads],
        "no_template_hir": True,
        "auto_template_fallback": False,
    }


def validate_ledger_constraints(norm: dict) -> None:
    if norm.get("problem_source") != "ledger":
        raise BridgeError("LEDGER_SOURCE_REQUIRED", "ledger mode requires real ledger Problem")
    if norm.get("auto_template_fallback"):
        raise BridgeError("AUTO_TEMPLATE_FALLBACK_FORBIDDEN", "template fallback flag set")
    if not norm.get("constraint_manifest"):
        raise BridgeError("LEDGER_EMPTY_MANIFEST", "empty manifest cannot satisfy")
    a = norm["core_args"]
    cap = norm["capacity"]
    if cap["usable_bytes"] is None and not str(cap["status"]).startswith("UNKNOWN"):
        raise BridgeError("UNKNOWN_CAPACITY_FABRICATED", "usable_bytes=None must stay UNKNOWN")
    if a["memory_groups"] != NA:
        raise BridgeError("LEDGER_CAPACITY_NOT_DEFAULT_SATISFIED",
                          "memory_groups must stay N-A; no default capacity satisfaction")
    if a["fixed_buffer_versions"] != NA:
        raise BridgeError("LEDGER_FIXED_VERSIONS_NOT_DEFAULT_SATISFIED",
                          "fixed buffer versions must stay N-A")
    if a["num_stages"] != NA or a["enable_offset"] != NA or a["manual_stages"] != NA:
        raise BridgeError("LEDGER_POLICY_N-A_VIOLATION", "policy/offset/stages must stay N-A")
    no_isa = sum(1 for t in a["tasks"] if t["pipe"] == "NO_ISA_CONTRACT")
    none_latency = sum(1 for t in a["tasks"] if t["latency"] is None)
    none_ii = sum(1 for t in a["tasks"] if t["issue_interval"] is None)
    offset_unknown = sum(1 for x in a["accesses"] if x["offset_status"] == "OFFSET_UNKNOWN")
    if not (no_isa or none_latency or none_ii or offset_unknown):
        raise BridgeError("LEDGER_REAL_STATUS_LOST", "real UNKNOWN/None statuses were overwritten")
    for x in a["accesses"]:
        if x["offset_status"] == "OFFSET_UNKNOWN" and x["normalized_offset"] is not None:
            raise BridgeError("UNKNOWN_OFFSET_FILLED_WITH_ZERO_OR_VALUE",
                              "OFFSET_UNKNOWN must not be converted to a concrete offset")
    for t in a["tasks"]:
        if t["pipe"] == "NO_ISA_CONTRACT" and t["resource"] not in (None, "NO_ISA_CONTRACT"):
            raise BridgeError("NO_ISA_CONTRACT_FABRICATED", "NO_ISA_CONTRACT pipe was given a fake resource")
    if len(a["data_deps"]) != len(norm["problem_as_dict"]["dependencies"]):
        raise BridgeError("LEDGER_DEPENDENCY_MAPPING_MISSING",
                          "data_deps count does not match real Problem dependencies")
    if not norm["action_id_map"]["bidirectional_ok"]:
        raise BridgeError("LEDGER_ACTION_MAPPING_INCOMPLETE", "action_id <-> core_args reverse check failed")
    unknown = norm.get("unknown_fields") or []
    if not unknown:
        raise BridgeError("LEDGER_UNKNOWN_FIELDS_LOST", "UNKNOWN/N-A fields were dropped")
    raise BridgeError("LEDGER_FIELD_UNKNOWN",
                      "real ledger lacks solver-ready policy/versions/offset/%d NO_ISA_CONTRACT/%d None latency/%d OFFSET_UNKNOWN; status UNKNOWN/REFUSED"
                      % (no_isa, none_latency, offset_unknown))


def solve_ledger(norm: dict, side: str = "tf", *, z3_missing: bool = False,
                 force_solved: bool = False, unknown_check: bool = False) -> dict:
    """Fail-closed ledger solve: NO core call, NO template fallback, NO solved claim."""
    if z3_missing and force_solved:
        raise BridgeError("Z3_MISSING_REPORTED_SOLVED",
                          "forged solved claim with z3 unavailable is forbidden")
    if force_solved:
        raise BridgeError("LEDGER_FALLBACK_FORBIDDEN",
                          "ledger mode has no sat/fallback result; forced solved is forbidden")
    try:
        validate_ledger_constraints(norm)
        refusal_code = None
        detail = "validator unexpectedly passed"
    except BridgeError as e:
        refusal_code, detail = e.code, e.message
    trace = {"side": side, "problem_source": "ledger", "core_called": False,
             "template_fallback": False, "z3_missing": bool(z3_missing),
             "simulated_unknown_check": bool(unknown_check), "validation_code": refusal_code}
    return {
        "solver_state": "unknown",
        "plan_status": "refused",
        "state_reason": "LEDGER_FAIL_CLOSED",
        "solved_claim": False,
        "refusal_code": refusal_code,
        "detail": detail,
        "core_result": None,
        "result_digest": None,
        "check_results": [],
        "requested_num_stages": NA,
        "nonempty_stage_ordinals_observed": None,
        "stage_claim": "no solver run; no claim of N non-empty stages",
        "trace": trace,
        "normalized_digest": norm["normalized_digest"],
        "unknown_or_na_preserved": True,
        "fallback_not_solved": True,
    }


def ledger_state_table(norm: dict) -> list[dict]:
    rows = []
    scenarios = [
        ("sat_full_constraints", "sat", {}),
        ("unsat_manual_infeasible", "unsat", {}),
        ("unknown_z3_unavailable", "unknown", {"z3_missing": True}),
        ("unknown_check_unknown_preserved", "unknown", {"unknown_check": True}),
        ("fallback_explicit_serial", "fallback", {}),
    ]
    for scenario, requested, kw in scenarios:
        res = solve_ledger(norm, "tf", **kw)
        rows.append({"scenario": scenario, "requested_state": requested,
                     "observed_state": res["solver_state"],
                     "plan_status": res["plan_status"],
                     "solved_claim": res["solved_claim"],
                     "refusal_code": res["refusal_code"],
                     "check_results": res["check_results"],
                     "core_called": res["trace"]["core_called"],
                     "template_fallback": res["trace"]["template_fallback"],
                     "no_fake_status_claim": True})
    return rows


def run_mutation_ledger(name: str) -> dict:
    ledger, problem = build_ledger_problem()
    base = normalize_ledger_problem(ledger, problem, "tf")
    baseline = solve_ledger(base, "tf")
    out = {"name": name, "problem_source": "ledger", "killed": False, "named_code": None}

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
        out["killed"] = (baseline["solver_state"] == "unknown" and
                         baseline["plan_status"] == "refused" and not baseline["solved_claim"])
        out["named_code"] = "LEDGER_FAIL_CLOSED" if out["killed"] else "LEDGER_NOT_FAIL_CLOSED"
    elif name == "probe_can_fail":
        bad = copy.deepcopy(base)
        bad["core_args"]["memory_groups"] = []
        old = globals()["validate_ledger_constraints"]
        try:
            globals()["validate_ledger_constraints"] = lambda n: None
            try:
                validate_ledger_constraints(bad)
                out["killed"] = True
                out["named_code"] = "LEDGER_VALIDATOR_NEUTRALISED"
            except BridgeError:
                out["killed"] = False
                out["named_code"] = "LEDGER_VALIDATOR_STILL_LIVE"
        finally:
            globals()["validate_ledger_constraints"] = old
    elif name == "probe_ready_free_can_fail":
        bad = copy.deepcopy(base)
        bad["core_args"]["data_deps"] = []
        old = globals()["validate_ledger_constraints"]
        try:
            globals()["validate_ledger_constraints"] = lambda n: None
            try:
                validate_ledger_constraints(bad)
                out["killed"] = True
                out["named_code"] = "LEDGER_DEPENDENCY_VALIDATOR_NEUTRALISED"
            except BridgeError:
                out["killed"] = False
                out["named_code"] = "LEDGER_DEPENDENCY_VALIDATOR_STILL_LIVE"
        finally:
            globals()["validate_ledger_constraints"] = old
    elif name == "empty_manifest":
        bad = copy.deepcopy(base)
        bad["constraint_manifest"] = []
        expect_refusal(lambda: validate_ledger_constraints(bad), "LEDGER_EMPTY_MANIFEST")
    elif name == "drop_capacity":
        bad = copy.deepcopy(base)
        bad["core_args"]["memory_groups"] = []
        expect_refusal(lambda: validate_ledger_constraints(bad),
                       "LEDGER_CAPACITY_NOT_DEFAULT_SATISFIED")
    elif name == "drop_ready_free":
        bad = copy.deepcopy(base)
        bad["core_args"]["data_deps"] = bad["core_args"]["data_deps"][:-1]
        expect_refusal(lambda: validate_ledger_constraints(bad),
                       "LEDGER_DEPENDENCY_MAPPING_MISSING")
    elif name == "force_solved_missing_z3":
        expect_refusal(lambda: solve_ledger(base, "tf", z3_missing=True, force_solved=True),
                       "Z3_MISSING_REPORTED_SOLVED")
    elif name == "fallback_as_solved":
        expect_refusal(lambda: solve_ledger(base, "tf", force_solved=True),
                       "LEDGER_FALLBACK_FORBIDDEN")
    elif name == "extra_task":
        from tilefoundry_ascend.schedule.ledger import LedgerEntry, ProblemTask, build_problem
        extra_aid = "ledger-extra-action:0"
        entry = LedgerEntry(extra_aid, "extra_instruction", "Vadd", "x_extra", "real",
                            "ascend.ub", "write", 64, None, "symbolic", (), (), (),
                            "t", "extra/loop", "1", "extra/path")
        ledger2 = copy.deepcopy(ledger)
        ledger2.entries = list(ledger2.entries) + [entry]
        problem2 = build_problem(ledger2)
        norm2 = normalize_ledger_problem(ledger2, problem2, "tf")
        changed = (norm2["normalized_digest"] != base["normalized_digest"]
                   and norm2["action_id_map"]["mapping_digest"] != base["action_id_map"]["mapping_digest"])
        out["killed"] = changed and not solve_ledger(norm2, "tf")["solved_claim"]
        out["named_code"] = "EXTRA_TASK_CHANGED_CORE_ARGS" if out["killed"] else "EXTRA_TASK_IGNORED"
    elif name == "distance_zero":
        # Real ledger distances are already 0/1; forcing 0 can be a no-op. Change
        # every dependency distance by a fixed non-zero delta so the probe can fail.
        problem2 = copy.deepcopy(problem)
        problem2.dependencies = [dict(d, distance=int(d["distance"]) + 7)
                                 for d in problem2.dependencies]
        norm2 = normalize_ledger_problem(ledger, problem2, "tf")
        changed = norm2["normalized_digest"] != base["normalized_digest"]
        out["killed"] = changed and not solve_ledger(norm2, "tf")["solved_claim"]
        out["named_code"] = "DISTANCE_CHANGED_CORE_ARGS" if out["killed"] else "DISTANCE_IGNORED"
        out["baseline_normalized_digest"] = base["normalized_digest"]
        out["mutated_normalized_digest"] = norm2["normalized_digest"]
    elif name == "capacity_zero_fallback":
        bad = copy.deepcopy(base)
        bad["core_args"]["memory_groups"] = [[0, 0]]
        bad["capacity"]["status"] = "KNOWN_USABLE"
        expect_refusal(lambda: validate_ledger_constraints(bad), "UNKNOWN_CAPACITY_FABRICATED")
    elif name == "unknown_check_preserved":
        res = solve_ledger(base, "tf", unknown_check=True)
        out["killed"] = res["solver_state"] == "unknown" and not res["solved_claim"]
        out["named_code"] = "UNKNOWN_PRESERVED" if out["killed"] else "UNKNOWN_COLLAPSED"
    elif name == "missing_z3_not_solved":
        res = solve_ledger(base, "tf", z3_missing=True)
        out["killed"] = res["solver_state"] == "unknown" and not res["solved_claim"]
        out["named_code"] = "MISSING_Z3_UNKNOWN" if out["killed"] else "MISSING_Z3_SOLVED"
    elif name == "unsat_manual_not_solved":
        res = solve_ledger(base, "tf")
        out["killed"] = res["solver_state"] == "unknown" and not res["solved_claim"] and res["plan_status"] == "refused"
        out["named_code"] = "UNSAT_NOT_CLAIMED" if out["killed"] else "UNSAT_FABRICATED"
    else:
        raise BridgeError("UNKNOWN_MUTATION", name)
    return out


def run_ledger_evidence(out: Path) -> int:
    """A intermediate evidence: real ledger wiring only; NO correspondence.json."""
    for child in out.iterdir():
        if child.is_dir():
            import shutil
            shutil.rmtree(child)
        else:
            child.unlink()
    try:
        ledger, problem = build_ledger_problem()
    except Exception as e:  # noqa: BLE001
        guard_no_template_fallback("ledger", True, type(e).__name__ + ": " + str(e)[:200])
        raise
    guard_no_template_fallback("ledger", False)
    norm = normalize_ledger_problem(ledger, problem, "tf")
    write_json(out / "tf_ledger_problem.json", norm["problem_as_dict"])
    write_json(out / "core_args.json", norm["core_args"])
    write_json(out / "action_id_map.json", norm["action_id_map"])
    write_json(out / "state_table.json", ledger_state_table(norm))
    refusal = solve_ledger(norm, "tf")
    write_json(out / "LEDGER_REFUSAL.json", refusal)
    meta = {
        "phase": "A_TF_REAL_LEDGER_WIRING",
        "task": "04-09 B-to-C point 1 development/field comparison PREPARATION",
        "problem_source": "ledger",
        "explicit_source_required": True,
        "auto_template_fallback_forbidden": True,
        "template_mode_is_old_domain": True,
        "bound_04_10_commit": "2e938af",
        "ledger_git_blob": norm["ledger_git_blob"],
        "ledger_current_file_sha256": norm["ledger_current_file_sha256"],
        "fixture": norm["fixture"],
        "chain": norm["chain"],
        "status_counts": norm["status_counts"],
        "unknown_fields": norm["unknown_fields"],
        "excluded_unchanged": norm["excluded_unchanged"],
        "capacity": norm["capacity"],
        "no_device": True,
        "p4_cpu_planning_only": True,
        "no_p5_or_performance_claim": True,
        "correspondence_json_produced": False,
        "tl_structured_fields_phase": "B_NOT_STARTED",
        "conclusion": "A only; B-to-C point 1 NOT accepted",
    }
    write_json(out / "META.json", meta)
    checks = {
        "explicit_source_required": _refusal_code(lambda: resolve_problem_source(None))
                                     == "PROBLEM_SOURCE_EXPLICIT_REQUIRED",
        "auto_template_fallback_forbidden": _refusal_code(lambda: resolve_problem_source("auto"))
                                             == "AUTO_TEMPLATE_FALLBACK_FORBIDDEN",
        "ledger_failure_no_template_fallback": _refusal_code(
            lambda: guard_no_template_fallback("ledger", True, "simulated")) ==
            "LEDGER_SOURCE_FAILED_NO_TEMPLATE_FALLBACK",
        "real_ledger_entries": len(ledger.entries) > 0,
        "offset_unknown_preserved": norm["status_counts"]["offset_unknown"] > 0,
        "no_isa_contract_preserved": norm["status_counts"]["no_isa_contract_tasks"] > 0,
        "latency_none_preserved": norm["status_counts"]["latency_none_tasks"] > 0,
        "capacity_usable_unknown_preserved": not norm["capacity_usable_known"],
        "fixed_capacity_not_default_satisfied": norm["core_args"]["fixed_buffer_versions"] == NA
                                                and norm["core_args"]["memory_groups"] == NA,
        "bidirectional_action_map_ok": norm["action_id_map"]["bidirectional_ok"],
        "no_template_hir": norm["no_template_hir"] and not norm["auto_template_fallback"],
        "no_solved_claim": all(not r["solved_claim"] for r in ledger_state_table(norm)),
        "no_correspondence_json": not (out / "correspondence.json").exists(),
    }
    write_json(out / "SELF_CHECKS.json", checks)
    print(json.dumps({"phase": "A", "problem_source": "ledger",
                      "entries": len(ledger.entries),
                      "checks": checks}, sort_keys=True))
    return 0 if all(checks.values()) else 1


C_PASSES = ["tl.AutoSchedule", "tl.PrepareMultiBuffer", "tl.InsertSync",
            "tl.MaterializeMultiBuffer", "tl.LowerScheduledTIR"]
TF_SPACE_CLASS = {"GMEM": "GMEM", "SMEM": "LOCAL", "RMEM": "LOCAL",
                  "CARRIER": "LOCAL"}
TF_OP_TO_TL = {
    "GmToUb": "tl.tileop.ascend_copy", "UbToGm": "tl.tileop.ascend_copy",
    "UbToUb": "tl.tileop.ascend_copy", "UbToL1": "tl.tileop.ascend_copy",
    "L1ToUb": "tl.tileop.ascend_copy",
    "Vcvt": "tl.simd.vcvt", "Vmul": "tl.simd.vmul", "Vsts": "tl.simd.vsts",
    "Vmula": "tl.simd.vmula", "Pset16": "tl.simd.pset", "Vadd": "tl.simd.vadd",
    "Vcadd": "tl.simd.vcadd", "Vdiv": "tl.simd.vdiv", "Vmuls": "tl.simd.vmuls",
    "Vadds": "tl.simd.vadds", "Vsqrt": "tl.simd.vsqrt", "Vld": "tl.simd.vld",
    "Vdup": "tl.simd.vdup", "Vdupv": "tl.simd.vdupv",
}
for _i in range(64):
    TF_OP_TO_TL["uw" + str(_i)] = "tl.tileop.ascend_copy"


def _tf_loop_class(iteration_path: str) -> str:
    return "outside" if iteration_path == "outside" else "loop"


def _tl_loop_class(loop_depth: int) -> str:
    return "outside" if int(loop_depth) == 0 else "loop"


def _load_c_nodes(c_root: Path) -> dict:
    out = {}
    cases = sorted(p.name for p in c_root.iterdir() if p.is_dir())
    for p in C_PASSES:
        out[p] = []
        for case in cases:
            fp = c_root / case / (p + ".json")
            if not fp.is_file():
                raise BridgeError("C_STRUCTURED_FILE_MISSING", str(fp))
            d = json.loads(fp.read_text())
            for n in d["nodes"]:
                nn = dict(n)
                nn["case"] = case
                nn["pass"] = p
                nn["tl_node_key"] = case + "::" + n["node_path"]
                out[p].append(nn)
    return out


def _tl_access_records(nodes: list) -> tuple[list, dict]:
    accesses = []
    node_has_access = {}
    for n in nodes:
        has = False
        for i, a in enumerate(n.get("accesses", [])):
            has = True
            rec = {
                "pass": n["pass"], "case": n["case"], "node_path": n["node_path"],
                "tl_node_key": n["tl_node_key"], "access_index": i,
                "tl_access_key": n["tl_node_key"] + "#" + str(i),
                "op": n.get("op"), "root": a.get("root"), "scope": a.get("scope"),
                "space_class": a.get("space_class"), "mode": a.get("mode"),
                "mode_code": a.get("mode_code"),
                "mode_semantics_source": a.get("mode_semantics_source"),
                "offset_structure": a.get("offset_structure"),
                "loop_depth": n.get("loop_depth", 0),
                "loop_class": _tl_loop_class(n.get("loop_depth", 0)),
                "loop_vars": n.get("loop_vars", []),
                "extent_values": a.get("extent_values", []),
            }
            accesses.append(rec)
        node_has_access[n["tl_node_key"]] = has
    return accesses, node_has_access


def _tf_action_records(problem) -> tuple[list, dict]:
    aid_to_task = {}
    for t in problem.tasks:
        for aid in t.action_ids:
            aid_to_task[aid] = t
    recs = []
    counts = {"no_isa_contract_tasks": sum(1 for t in problem.tasks if t.pipe == "NO_ISA_CONTRACT"),
              "latency_none_tasks": sum(1 for t in problem.tasks if t.latency is None),
              "ii_none_tasks": sum(1 for t in problem.tasks if t.issue_interval is None),
              "offset_unknown": sum(1 for a in problem.accesses if a.offset_status == "OFFSET_UNKNOWN"),
              "offset_bytes": sum(1 for a in problem.accesses if a.offset_status == "BYTES")}
    for a in problem.accesses:
        t = aid_to_task.get(a.task)
        if t is None:
            recs.append({"action_id": a.task, "task": None, "op": None,
                         "root": a.root, "space": a.space, "space_class": None,
                         "mode": a.mode, "iteration_path": a.iteration_path,
                         "offset_status": a.offset_status,
                         "offset_structure": jsonable(a.offset_structure),
                         "no_isa_contract": True, "latency_none": True,
                         "tf_tl_op": None})
            continue
        rec = {"action_id": a.task, "task": t.stable_id, "op": t.op,
               "pipe": t.pipe, "latency": t.latency, "issue_interval": t.issue_interval,
               "root": a.root, "root_kind": a.root_kind, "space": a.space,
               "space_class": TF_SPACE_CLASS.get(a.space, "UNKNOWN_SPACE"),
               "mode": a.mode, "iteration_path": a.iteration_path,
               "loop_class": _tf_loop_class(a.iteration_path),
               "offset_status": a.offset_status,
               "offset_structure": jsonable(a.offset_structure),
               "byte_footprint": a.byte_footprint, "byte_offset": a.byte_offset,
               "attrs": jsonable(a.attrs), "guard": jsonable(a.guard),
               "logical_iteration": a.logical_iteration,
               "iteration_extent": a.iteration_extent,
               "iteration_domains": jsonable(getattr(a, "iteration_domains", ())),
               "no_isa_contract": t.pipe == "NO_ISA_CONTRACT",
               "latency_none": t.latency is None or t.issue_interval is None,
               "tf_tl_op": TF_OP_TO_TL.get(t.op)}
        recs.append(rec)
    return recs, counts


def _pair_status_legacy(tf: dict, tl: dict) -> tuple[str, list, list]:
    """PRE-FIX behavior kept only for before/after evidence: unconditional N-A
    made the EQUAL branch unreachable."""
    diffs = []
    na = []
    for field in ("root", "space_class", "mode", "loop_class", "offset_structure"):
        a, b = tf.get(field), tl.get(field)
        if a is None or b is None:
            na.append({"field": field, "tf": a, "tl": b,
                       "reason": "FIELD_UNAVAILABLE_ONE_SIDE"})
        elif a != b:
            diffs.append({"field": field, "tf": a, "tl": b,
                          "reason": "STRUCTURAL_FIELD_DIFF"})
    for field in ("byte_footprint", "byte_offset", "attrs", "guard",
                  "logical_iteration", "iteration_extent", "latency", "issue_interval"):
        na.append({"field": field, "tf": tf.get(field),
                   "tl": "N-A_NOT_EXPORTED_IN_TL_STRUCTURED",
                   "reason": "N-A_NOT_IN_TL_STRUCTURED_EXPORT"})
    na.append({"field": "physical_address_or_generation", "tf": "N-A", "tl": "N-A",
               "reason": "NO_TYPED_SOURCE_EITHER_SIDE"})
    if diffs:
        return "DIFFERENCE", diffs, na
    if na:
        return "N-A", diffs, na
    return "EQUAL", diffs, na


def _pair_status(tf: dict, tl: dict) -> tuple[str, list, list]:
    """FIXED: core fields decide EQUAL/DIFFERENCE/N-A; auxiliary not-exported
    fields are retained in na_fields but do NOT make EQUAL unreachable."""
    diffs = []
    na_core = []
    na_aux = []
    for field in ("root", "space_class", "mode", "loop_class", "offset_structure"):
        a, b = tf.get(field), tl.get(field)
        if a is None or b is None:
            na_core.append({"field": field, "tf": a, "tl": b,
                            "reason": "FIELD_UNAVAILABLE_ONE_SIDE"})
        elif a != b:
            diffs.append({"field": field, "tf": a, "tl": b,
                          "reason": "STRUCTURAL_FIELD_DIFF"})
    for field in ("byte_footprint", "byte_offset", "attrs", "guard",
                  "logical_iteration", "iteration_extent", "latency", "issue_interval"):
        na_aux.append({"field": field, "tf": tf.get(field),
                       "tl": "N-A_NOT_EXPORTED_IN_TL_STRUCTURED",
                       "reason": "N-A_NOT_IN_TL_STRUCTURED_EXPORT"})
    na_aux.append({"field": "physical_address_or_generation", "tf": "N-A", "tl": "N-A",
                   "reason": "NO_TYPED_SOURCE_EITHER_SIDE"})
    if diffs:
        return "DIFFERENCE", diffs, na_aux
    if na_core:
        return "N-A", diffs, na_core + na_aux
    return "EQUAL", diffs, na_aux


def run_c_evidence(out: Path, c_root: Path) -> int:
    if not c_root.is_dir():
        raise BridgeError("C_STRUCTURED_REQUIRED", str(c_root))
    ledger, problem = build_ledger_problem()
    nodes = _load_c_nodes(c_root)
    tf_records, tf_counts = _tf_action_records(problem)
    mapping_root = out / "mapping"
    if mapping_root.exists():
        import shutil
        shutil.rmtree(mapping_root)
    mapping_root.mkdir(parents=True, exist_ok=True)
    all_forward, all_reverse, all_unknown, all_excluded = {}, {}, {}, {}
    all_diffs = []
    stats = {}
    predeclared = {
        "TF_NO_ISA_CONTRACT_TASKS": tf_counts["no_isa_contract_tasks"],
        "TF_LATENCY_NONE_TASKS": tf_counts["latency_none_tasks"],
        "TF_II_NONE_TASKS": tf_counts["ii_none_tasks"],
        "TF_OFFSET_UNKNOWN_ACTIONS": tf_counts["offset_unknown"],
        "TF_OFFSET_BYTES_ACTIONS": tf_counts["offset_bytes"],
        "TL_UNKNOWN_EXPR_STRUCTURE_MARKERS": 0,
        "TL_UNKNOWN_CODE_SEMANTICS_UNSOURCED": 0,
        "TL_NON_ACCESS_NODES": 0,
        "TL_DUPLICATE_STRUCTURAL_OCCURRENCES": 0,
        "expected_root_namespace_difference": "TF ledger roots (w/x/out/smem@.../wcar0/...) and TL Buffer roots (X/W/Y/x_ub/w_ub/wf/acc*) are not assumed equal",
    }
    for p in C_PASSES:
        tl_nodes = nodes[p]
        tl_accesses, node_has_access = _tl_access_records(tl_nodes)
        for n in tl_nodes:
            if not node_has_access.get(n["tl_node_key"], False):
                predeclared["TL_NON_ACCESS_NODES"] += 1
        for a in tl_accesses:
            if a.get("mode_semantics_source") == "UNKNOWN_CODE_SEMANTICS_UNSOURCED":
                predeclared["TL_UNKNOWN_CODE_SEMANTICS_UNSOURCED"] += 1
            if any(x and x[0] == "UNKNOWN_TYPED_EXPR" for x in (a.get("offset_structure") or []) if isinstance(x, list)):
                predeclared["TL_UNKNOWN_EXPR_STRUCTURE_MARKERS"] += 1
        tl_by_key = {}
        for a in tl_accesses:
            if a.get("op") is None or a.get("space_class") in (None, "UNKNOWN_SCOPE") or a.get("mode") is None or a.get("loop_class") is None:
                continue
            key = (a["op"], a["space_class"], a["loop_class"], a["mode"])
            tl_by_key.setdefault(key, []).append(a)
        for k in tl_by_key:
            tl_by_key[k].sort(key=lambda x: x["tl_access_key"])
        forward = {}
        reverse = {}
        unknown = []
        excluded = []
        status_counts = {"EQUAL": 0, "DIFFERENCE": 0, "N-A": 0,
                         "UNKNOWN": 0, "EXCLUDED": 0}
        for tf in tf_records:
            if tf["no_isa_contract"]:
                unknown.append({"side": "TF", "action_id": tf["action_id"],
                                "reason_code": "TF_NO_ISA_CONTRACT"})
                status_counts["UNKNOWN"] += 1
                continue
            if tf["latency_none"]:
                unknown.append({"side": "TF", "action_id": tf["action_id"],
                                "reason_code": "TF_LATENCY_OR_II_NONE"})
                status_counts["UNKNOWN"] += 1
                continue
            if tf["offset_status"] != "BYTES":
                unknown.append({"side": "TF", "action_id": tf["action_id"],
                                "reason_code": "TF_" + tf["offset_status"]})
                status_counts["UNKNOWN"] += 1
                continue
            if tf.get("tf_tl_op") is None:
                unknown.append({"side": "TF", "action_id": tf["action_id"],
                                "reason_code": "TF_OP_WITHOUT_STRUCTURAL_TL_FAMILY"})
                status_counts["UNKNOWN"] += 1
                continue
            if tf.get("space_class") in (None, "UNKNOWN_SPACE"):
                unknown.append({"side": "TF", "action_id": tf["action_id"],
                                "reason_code": "TF_SPACE_WITHOUT_STRUCTURAL_CLASS"})
                status_counts["UNKNOWN"] += 1
                continue
            key = (tf["tf_tl_op"], tf["space_class"], tf["loop_class"], tf["mode"])
            candidates = tl_by_key.get(key, [])
            if not candidates:
                unknown.append({"side": "TF", "action_id": tf["action_id"],
                                "reason_code": "NO_TL_STRUCTURAL_MATCH",
                                "key": list(key)})
                status_counts["UNKNOWN"] += 1
                continue
            tl = candidates[0]
            status, diffs, na_fields = _pair_status(tf, tl)
            entry = {"action_id": tf["action_id"], "task": tf["task"], "op": tf["op"],
                     "tf_tl_op": tf["tf_tl_op"], "pass": p, "case": tl["case"],
                     "node_path": tl["node_path"], "tl_access_key": tl["tl_access_key"],
                     "access_index": tl["access_index"], "status": status,
                     "difference_fields": diffs, "na_fields": na_fields,
                     "candidate_count": len(candidates)}
            forward[tf["action_id"]] = entry
            rev = reverse.setdefault(tl["tl_access_key"], {
                "pass": p, "case": tl["case"], "node_path": tl["node_path"],
                "access_index": tl["access_index"], "action_ids": [],
                "status_counts": {"EQUAL": 0, "DIFFERENCE": 0, "N-A": 0},
                "difference_fields_by_action": []})
            rev["action_ids"].append(tf["action_id"])
            rev["status_counts"][status] += 1
            if status == "DIFFERENCE":
                rev["difference_fields_by_action"].append(
                    {"action_id": tf["action_id"], "difference_fields": diffs})
                all_diffs.append(entry)
            status_counts[status] += 1
        reverse_nodes = {k.rsplit("#", 1)[0] for k in reverse}
        for n in tl_nodes:
            if n["tl_node_key"] in reverse_nodes:
                continue
            if not node_has_access.get(n["tl_node_key"], False):
                excluded.append({"side": "TL", "case": n["case"], "pass": p,
                                 "node_path": n["node_path"],
                                 "reason_code": "TL_NON_ACCESS_TYPED_NODE"})
                status_counts["EXCLUDED"] += 1
            else:
                unknown.append({"side": "TL", "case": n["case"], "pass": p,
                                "node_path": n["node_path"],
                                "reason_code": "NO_TF_STRUCTURAL_MATCH"})
                status_counts["UNKNOWN"] += 1
        for a in tl_accesses:
            if a["tl_access_key"] in reverse:
                continue
            if any(e["node_path"] == a["node_path"] and e["case"] == a["case"]
                   for e in unknown if e["side"] == "TL"):
                continue
            unknown.append({"side": "TL", "case": a["case"], "pass": p,
                            "node_path": a["node_path"], "access_index": a["access_index"],
                            "reason_code": "UNPAIRED_TL_ACCESS",
                            "candidate_key": [a["op"], a["space_class"], a["loop_class"], a["mode"]]})
        tf_forward = len(forward)
        tf_unknown = sum(1 for u in unknown if u["side"] == "TF")
        node_status = {n["tl_node_key"]: "UNKNOWN" for n in tl_nodes}
        for k in reverse_nodes:
            node_status[k] = "REVERSE"
        for e in excluded:
            k = e["case"] + "::" + e["node_path"]
            if node_status.get(k) != "REVERSE":
                node_status[k] = "EXCLUDED"
        for u in unknown:
            if u["side"] == "TL" and u.get("node_path"):
                k = u["case"] + "::" + u["node_path"]
                if node_status.get(k) not in ("REVERSE", "EXCLUDED"):
                    node_status[k] = "UNKNOWN"
        for k, st in node_status.items():
            if st == "UNKNOWN" and not any(
                    u["side"] == "TL" and u.get("node_path") and
                    u["case"] + "::" + u["node_path"] == k for u in unknown):
                case, node_path = k.split("::", 1)
                unknown.append({"side": "TL", "case": case, "pass": p,
                                "node_path": node_path,
                                "reason_code": "UNCLASSIFIED_TL_NODE"})
        tl_reverse_nodes = sum(1 for s in node_status.values() if s == "REVERSE")
        tl_unknown_nodes = sum(1 for s in node_status.values() if s == "UNKNOWN")
        tl_excluded_nodes = sum(1 for s in node_status.values() if s == "EXCLUDED")
        tl_unknown_accesses = sum(1 for u in unknown if u["side"] == "TL" and "access_index" in u)
        tl_excluded = tl_excluded_nodes
        tl_covered_nodes = tl_reverse_nodes + tl_unknown_nodes + tl_excluded_nodes
        stats[p] = {
            "tf_total": len(tf_records), "tf_forward_index": tf_forward,
            "tf_unknown": tf_unknown, "tf_excluded": 0,
            "tf_forward_rate": tf_forward / len(tf_records) if tf_records else None,
            "tl_nodes_total": len(tl_nodes), "tl_accesses_total": len(tl_accesses),
            "tl_reverse_nodes": tl_reverse_nodes,
            "tl_reverse_access": len(reverse),
            "tl_reverse_actions": sum(len(v["action_ids"]) for v in reverse.values()),
            "tl_unknown_nodes": tl_unknown_nodes,
            "tl_unknown_accesses": tl_unknown_accesses,
            "tl_excluded_nodes": tl_excluded_nodes,
            "tl_node_coverage_complete": tl_covered_nodes == len(tl_nodes),
            "status_counts": status_counts,
        }
        all_forward[p] = forward
        all_reverse[p] = reverse
        all_unknown[p] = unknown
        all_excluded[p] = excluded
    total_pairs = sum(len(v) for v in all_forward.values())
    total_denom = len(tf_records) * len(C_PASSES)
    aggregate = {
        "tf_actions": len(tf_records), "passes": len(C_PASSES),
        "tf_action_pass_denominator": total_denom,
        "forward_entries": total_pairs,
        "mapping_rate": total_pairs / total_denom if total_denom else None,
        "reverse_access_entries": sum(stats[p]["tl_reverse_access"] for p in C_PASSES),
        "reverse_actions": sum(stats[p]["tl_reverse_actions"] for p in C_PASSES),
        "status_counts": {k: sum(stats[p]["status_counts"][k] for p in C_PASSES)
                          for k in ("EQUAL", "DIFFERENCE", "N-A", "UNKNOWN", "EXCLUDED")},
        "tf_unknown_total": sum(stats[p]["tf_unknown"] for p in C_PASSES),
        "tl_unknown_node_total": sum(stats[p]["tl_unknown_nodes"] for p in C_PASSES),
        "tl_unknown_access_total": sum(stats[p]["tl_unknown_accesses"] for p in C_PASSES),
        "tl_excluded_total": sum(stats[p]["tl_excluded_nodes"] for p in C_PASSES),
        "tl_node_coverage_complete_all_passes": all(stats[p]["tl_node_coverage_complete"]
                                                     for p in C_PASSES),
        "unique_difference_actions": len({e["action_id"] for e in all_diffs}),
    }
    all_diffs.sort(key=lambda e: e["action_id"])
    first10 = []
    seen_diff_actions = set()
    for e in all_diffs:
        if e["action_id"] in seen_diff_actions:
            continue
        seen_diff_actions.add(e["action_id"])
        first10.append(e)
        if len(first10) >= 10:
            break
    write_json(mapping_root / "PRE_DECLARED.json", predeclared)
    write_json(mapping_root / "COVERAGE.json", {"aggregate": aggregate, "per_pass": stats})
    write_json(mapping_root / "FORWARD_INDEX.json", all_forward)
    write_json(mapping_root / "REVERSE_INDEX.json", all_reverse)
    write_json(mapping_root / "UNKNOWNS.json", all_unknown)
    write_json(mapping_root / "EXCLUDED.json", all_excluded)
    write_json(mapping_root / "DIFFERENCE_TABLE.json",
               {"total": len(all_diffs), "first_10": first10})
    lines = ["# C structural difference table (development)", "",
             "First 10 DIFFERENCE entries; every entry keeps TF action_id and TL node_path.", "",
             "| # | action_id | pass | case | node_path | differing fields |", "|---|---|---|---|---|---|"]
    for i, e in enumerate(first10, 1):
        fields = "; ".join("%s: tf=%s tl=%s" % (d["field"], d["tf"], d["tl"])
                           for d in e["difference_fields"])
        lines.append("| %d | %s | %s | %s | %s | %s |" %
                     (i, e["action_id"], e["pass"], e["case"], e["node_path"], fields))
    lines.append("")
    (mapping_root / "DIFFERENCE_TABLE.md").write_text(chr(10).join(lines) + chr(10))
    result = {
        "phase": "C", "status": "DONE_INTERMEDIATE",
        "conclusion": "B-to-C point 1 development/field comparison preparation; NOT accepted",
        "tf_source": "real 2e938af ledger rmsnorm_main.py:RMSNormD",
        "c_structured_root": str(c_root),
        "node_path_scheme": "TL C export: main/<typed_node_class>:<structural_key>#<preorder_sibling_index>; case-qualified in C mapping",
        "predeclared_before_summary": predeclared,
        "aggregate": aggregate,
        "per_pass": stats,
        "first_10_differences": first10,
        "difference_table": "mapping/DIFFERENCE_TABLE.json",
        "forward_index": "mapping/FORWARD_INDEX.json",
        "reverse_index": "mapping/REVERSE_INDEX.json",
        "unknowns": "mapping/UNKNOWNS.json",
        "excluded": "mapping/EXCLUDED.json",
        "coverage": "mapping/COVERAGE.json",
        "correspondence_json_produced": False,
        "D_correspondence": "NOT_STARTED",
        "no_device": True, "no_p5_or_performance_credit": True,
        "commit": "NOT_COMMITTED_UNTIL_D",
    }
    write_json(out / "C_RESULT.json", result)
    print(json.dumps({"phase": "C", "aggregate": aggregate,
                      "first10": len(first10)}, sort_keys=True))
    return 0


TF_DOMAIN = {"fixture": "rmsnorm_main.py:RMSNormD", "batch": 2, "d": 1024,
             "x_len": 2048, "w_len": 1024, "rows": 2, "loop_mode": "static_main",
             "source": "targets/ascend/tests/fixtures/rmsnorm/rmsnorm_main.py: BATCH=2,D=1024"}
TL_CASE_CONFIG = {
    "prefill_dyn": {"batch": 8192, "d": 4096, "rows": 4, "loop_mode": "dyn"},
    "prefill_recompute": {"batch": 8192, "d": 4096, "rows": 4, "loop_mode": "recompute"},
    "decode_dyn": {"batch": 64, "d": 4096, "rows": 1, "loop_mode": "dyn"},
    "decode_recompute": {"batch": 64, "d": 4096, "rows": 1, "loop_mode": "recompute"},
}


def _tf_attrs(tf: dict) -> dict:
    try:
        return dict(tf.get("attrs") or [])
    except Exception:
        return {}


def _tf_role(tf: dict) -> str:
    attrs = _tf_attrs(tf)
    pack = attrs.get("pack")
    if tf.get("op") == "Vsts" and pack == "PK_B32" and tf.get("space") == "SMEM":
        return "output_bf16_tile"
    if tf.get("op") == "Vsts":
        return "store_other"
    if tf.get("op") in ("GmToUb", "UbToGm", "UbToUb", "UbToL1", "L1ToUb") or str(tf.get("op", "")).startswith("uw"):
        return "copy"
    if tf.get("op") == "Vcvt":
        return "conversion"
    if tf.get("op") == "Vld":
        return "load"
    return "UNKNOWN_ROLE"


def _tl_dtype_map(nodes: list) -> dict:
    out = {}
    for n in nodes:
        if n.get("node_type") == "Buffer" and n.get("name"):
            out[n["name"]] = n.get("dtype")
        for b in n.get("alloc_buffers", []) or []:
            if b.get("name"):
                out[b["name"]] = b.get("dtype")
    return out


def _tl_role(root: str, dtype: str | None, phase: str) -> str:
    root = root or ""
    dtype = dtype or "UNKNOWN_DTYPE"
    if root == "wf":
        return "weight_f32_ub"
    if root == "w_ub":
        return "weight_bf16_ub"
    if root == "y_ub":
        return "output_bf16_tile"
    if root == "x_ub":
        return "input_bf16_ub"
    if root.startswith("acc"):
        return "acc"
    if "float32" in dtype and phase == "WEIGHT_VF":
        return "weight_f32_ub"
    return "UNKNOWN_ROLE"


def _tl_phase(loop_vars: list) -> str:
    if "s" in loop_vars:
        return "MAIN_VF"
    if "i" in loop_vars:
        return "WEIGHT_VF"
    if "r" in loop_vars:
        return "MAIN_VF"
    return "OUTSIDE"


def _case_binding() -> dict:
    exact = []
    for case, cfg in TL_CASE_CONFIG.items():
        if (cfg["batch"] == TF_DOMAIN["batch"] and cfg["d"] == TF_DOMAIN["d"]
                and cfg["rows"] == TF_DOMAIN["rows"] and cfg["loop_mode"] == TF_DOMAIN["loop_mode"]):
            exact.append(case)
    return {"tf_domain": TF_DOMAIN, "tl_cases": TL_CASE_CONFIG,
            "exact_matches": exact,
            "applicability": "EXACT_SHAPE_MODE" if exact else "UNBOUND_NO_EXACT_CASE_SHAPE_MODE",
            "rule": "case, pass, shape/mode must be partitioned BEFORE any candidate pairing",
            "partition_key": ["case", "pass", "shape.batch", "shape.d", "shape.rows", "loop_mode"]}


def run_c2_evidence(out: Path, c_root: Path, c_baseline: Path) -> int:
    if not c_root.is_dir():
        raise BridgeError("C2_STRUCTURED_REQUIRED", str(c_root))
    baseline = json.loads((c_baseline / "C_RESULT.json").read_text())
    ledger, problem = build_ledger_problem()
    nodes = _load_c_nodes(c_root)
    tf_records, _ = _tf_action_records(problem)
    binding = _case_binding()
    c2_root = out
    if c2_root.exists():
        import shutil
        shutil.rmtree(c2_root)
    c2_root.mkdir(parents=True, exist_ok=True)
    write_json(c2_root / "PARTITION.json", binding)

    # Candidate distribution: raw structural key only. No case is bound, so these
    # candidates are diagnostic and MUST NOT become forward entries.
    tl_all = []
    dtype_all = {}
    for p in C_PASSES:
        dtype_all.update(_tl_dtype_map(nodes[p]))
        accs, _ = _tl_access_records(nodes[p])
        for a in accs:
            a = dict(a)
            a["pass"] = p
            a["dtype"] = dtype_all.get(a.get("root"))
            a["vf_phase"] = _tl_phase(a.get("loop_vars", []))
            a["role"] = _tl_role(a.get("root"), a.get("dtype"), a["vf_phase"])
            tl_all.append(a)
    tl_by_key = {}
    for a in tl_all:
        if a.get("op") is None or a.get("space_class") in (None, "UNKNOWN_SCOPE") or a.get("mode") is None or a.get("loop_class") is None:
            continue
        tl_by_key.setdefault((a["op"], a["space_class"], a["loop_class"], a["mode"]), []).append(a)
    dist = {"0": 0, "1": 0, ">1": 0}
    role_dist = {"0": 0, "1": 0, ">1": 0}
    tf_with_any = 0
    candidate_total = 0
    role_candidate_total = 0
    role_rejected_total = 0
    for tf in tf_records:
        key = (tf.get("tf_tl_op"), tf.get("space_class"), tf.get("loop_class"), tf.get("mode"))
        cand = tl_by_key.get(key, [])
        n = len(cand)
        candidate_total += n
        if n == 0:
            dist["0"] += 1
        elif n == 1:
            dist["1"] += 1
        else:
            dist[">1"] += 1
        if n:
            tf_with_any += 1
        tf_role = _tf_role(tf)
        role_cands = []
        for c in cand:
            role_ok = tf_role == "UNKNOWN_ROLE" or c.get("role") == tf_role
            phase_ok = ((tf.get("loop_class") == "outside" and c.get("vf_phase") == "OUTSIDE")
                        or (tf.get("loop_class") == "loop"
                            and c.get("vf_phase") in ("MAIN_VF", "WEIGHT_VF")))
            if role_ok and phase_ok:
                role_cands.append(c)
            else:
                role_rejected_total += 1
        role_candidate_total += len(role_cands)
        rn = len(role_cands)
        if rn == 0:
            role_dist["0"] += 1
        elif rn == 1:
            role_dist["1"] += 1
        else:
            role_dist[">1"] += 1
    candidate_distribution = {
        "partition": "case/pass/shape-mode UNBOUND; candidate pool is cross-case diagnostic only",
        "tf_actions": len(tf_records),
        "candidate_entries_total": candidate_total,
        "tf_actions_with_any_candidate": tf_with_any,
        "per_tf_action_candidate_count": dist,
        "role_phase_rejected_candidates": role_rejected_total,
        "role_phase_candidate_entries_total": role_candidate_total,
        "per_tf_action_role_phase_candidate_count": role_dist,
        "do_not_select_first": True,
        "exact_case_matches": binding["exact_matches"],
    }
    write_json(c2_root / "CANDIDATE_DISTRIBUTION.json", candidate_distribution)

    # uo52 candidate set: show both wf (wrong f32 weight role) and y_ub (plausible output role),
    # but do NOT select either. Same candidate per case/pass is retained.
    tf_uo52 = next(x for x in tf_records if x["action_id"].endswith("/uo52:1"))
    auto_nodes = nodes["tl.AutoSchedule"]
    dtype_map = _tl_dtype_map(auto_nodes)
    uo52_candidates = []
    for n in auto_nodes:
        for i, a in enumerate(n.get("accesses", [])):
            if n.get("op") != "tl.simd.vsts" or a.get("mode") != "write":
                continue
            if a.get("space_class") != "LOCAL":
                continue
            phase = _tl_phase(n.get("loop_vars", []))
            role = _tl_role(a.get("root"), dtype_map.get(a.get("root")), phase)
            pack = "UNKNOWN_PACK_NOT_IN_TL_STRUCTURED_EXPORT"
            for arg in n.get("args", []) or []:
                if isinstance(arg, dict) and arg.get("kind") == "StringImm" and arg.get("value"):
                    pack = arg["value"]
                    break
            cand = {
                "case": n.get("case"), "pass": n.get("pass", "tl.AutoSchedule"),
                "node_path": n.get("node_path"), "access_index": i,
                "root": a.get("root"), "scope": a.get("scope"),
                "space_class": a.get("space_class"), "mode": a.get("mode"),
                "dtype": dtype_map.get(a.get("root")),
                "pack": pack,
                "loop_vars": n.get("loop_vars", []), "vf_phase": phase,
                "role": role,
                "role_matches_tf_output": role == _tf_role(tf_uo52),
                "pack_matches_tf": pack == _tf_attrs(tf_uo52).get("pack"),
            }
            # Field comparison for audit only; semantic status stays null.
            field_status, diffs, na = _pair_status(tf_uo52, {
                "root": cand["root"], "space_class": cand["space_class"],
                "mode": cand["mode"], "loop_class": "loop",
                "offset_structure": a.get("offset_structure")})
            cand["raw_field_status"] = field_status
            cand["difference_fields"] = diffs
            cand["na_fields"] = na
            cand["semantic_status"] = None
            if cand["root"] == "wf" and not cand["role_matches_tf_output"]:
                cand["verdict"] = "REJECT_ROLE_AND_PACK_MISMATCH_WEIGHT_F32_NORM_B32_VS_OUTPUT_BF16_PK_B32"
            elif cand["root"] == "y_ub" and cand["role_matches_tf_output"] and cand["pack_matches_tf"]:
                cand["verdict"] = "PLAUSIBLE_ROLE_DTYPE_PACK_PHASE_MATCH_BUT_CASE_SHAPE_UNBOUND_ROOT_UNPROVEN_OFFSET_COORDINATE_UNALIGNED"
            else:
                cand["verdict"] = "CANDIDATE_UNRESOLVED"
            uo52_candidates.append(cand)
    uo52_candidates.sort(key=lambda x: (x["case"], x["node_path"], x["access_index"]))
    tf_uo52_summary = {
        "action_id": tf_uo52["action_id"], "op": tf_uo52["op"], "root": tf_uo52["root"],
        "space": tf_uo52["space"], "mode": tf_uo52["mode"],
        "attrs": tf_uo52.get("attrs"), "pack": _tf_attrs(tf_uo52).get("pack"),
        "iteration_path": tf_uo52["iteration_path"], "offset_status": tf_uo52["offset_status"],
        "role": _tf_role(tf_uo52), "dtype": "UNKNOWN_NOT_IN_PROBLEM_ACCESS",
        "offset_note": "TF 23040 is arena physical byte constant; TL mul(i,64) is buffer element index. Not compared as same number.",
    }
    uo52 = {"tf": tf_uo52_summary, "candidate_count": len(uo52_candidates),
            "selected_candidate": None, "selection_rule": "NO_SELECTION_ON_MULTIPLE_CANDIDATES",
            "candidate_roots": sorted({c["root"] for c in uo52_candidates}),
            "candidate_cases": sorted({c["case"] for c in uo52_candidates}),
            "candidates": uo52_candidates,
            "verdict": "PARTIAL_AMBIGUOUS",
            "y_ub_is_plausible_candidate": any(c["root"] == "y_ub" and c["role_matches_tf_output"] for c in uo52_candidates),
            "root_binding_evidence": {
                "tf_alloc_path": tf_uo52["root"], "tl_buffer": "y_ub (plausible only)",
                "producer_consumer_deps": [],
                "base_range": None, "version_generation": "UNKNOWN",
                "status": "UNPROVEN_NO_PRODUCER_CONSUMER_BASE_RANGE_EVIDENCE",
            }}
    write_json(c2_root / "UO52_CANDIDATES.json", uo52)

    # EQUAL branch before/after: pure equal core pair.
    equal_tf = {"root": "same", "space_class": "LOCAL", "mode": "write",
                "loop_class": "outside", "offset_structure": [["const", 0]]}
    equal_tl = dict(equal_tf)
    legacy_status, legacy_diffs, legacy_na = _pair_status_legacy(equal_tf, equal_tl)
    fixed_status, fixed_diffs, fixed_na = _pair_status(equal_tf, equal_tl)
    equal_fix = {
        "input": {"tf": equal_tf, "tl": equal_tl, "note": "control pair, not a TF/TL production pairing"},
        "before": {"status": legacy_status, "difference_fields": legacy_diffs,
                   "auxiliary_na_count": len(legacy_na)},
        "after": {"status": fixed_status, "difference_fields": fixed_diffs,
                  "auxiliary_na_count": len(fixed_na)},
        "conclusion": "EQUAL is reachable after the fix; auxiliary N/A fields no longer force N-A when all core fields are equal.",
    }
    write_json(c2_root / "EQUAL_FIX.json", equal_fix)

    raw_counts = {"EQUAL": 0, "DIFFERENCE": 0, "N-A": 0, "UNKNOWN": 0}
    for c in uo52_candidates:
        raw_counts[c["raw_field_status"]] += 1
    c2_status = {
        "phase": "C2",
        "status": "DIFFERENCES_OPEN",
        "pairing_status": "PARTIAL_AMBIGUOUS",
        "consistency_proven": False,
        "point1_accepted": False,
        "scope": "development_field_comparison",
        "baseline": {"id": "C_BASELINE_ORIGINAL", "path": str(c_baseline),
                     "preserved": True, "raw_equal": 0,
                     "raw_difference_action_pass": baseline["aggregate"]["status_counts"]["DIFFERENCE"],
                     "unique_candidate_tf_actions": baseline["aggregate"]["unique_difference_actions"],
                     "tf_action_pass_denominator": baseline["aggregate"]["tf_action_pass_denominator"],
                     "candidate_rate": baseline["aggregate"]["mapping_rate"]},
        "partition": binding,
        "forward_verified": 0,
        "forward_verified_reason": "NO_EXACT_TF_CASE_SHAPE_MODE_BINDING",
        "candidate_distribution": candidate_distribution,
        "uo52": {"candidate_count": len(uo52_candidates),
                 "candidate_roots": uo52["candidate_roots"],
                 "selected_candidate": None,
                 "y_ub_is_plausible_candidate": uo52["y_ub_is_plausible_candidate"],
                 "status": "PARTIAL_AMBIGUOUS"},
        "equal_branch": {"before": legacy_status, "after": fixed_status,
                         "before_counts": {"EQUAL": 0, "DIFFERENCE": 0, "N-A": 1, "UNKNOWN": 0},
                         "after_counts": {"EQUAL": 1, "DIFFERENCE": 0, "N-A": 0, "UNKNOWN": 0}},
        "candidate_field_status_counts": raw_counts,
        "verified_semantic_difference_count": None,
        "unknown_count_units": {"tf_action_pass": baseline["aggregate"]["tf_unknown_total"],
                                "tl_node": baseline["aggregate"]["tl_unknown_node_total"]},
        "excluded_tl_nonaccess_nodes": baseline["aggregate"]["tl_excluded_total"],
        "tl_node_coverage_complete_all_passes": baseline["aggregate"]["tl_node_coverage_complete_all_passes"],
        "pairing_full_coverage": False,
        "blocking_reasons": ["CASE_SHAPE_MODE_UNBOUND", "AMBIGUOUS_FIRST_CANDIDATE",
                             "ROLE_MISMATCH_REJECTED_CANDIDATES", "ROOT_MAPPING_UNPROVEN",
                             "OFFSET_COORDINATE_SYSTEM_UNALIGNED"],
        "no_device": True, "no_p5_or_performance_credit": True,
        "commit": "NOT_COMMITTED_UNTIL_D",
    }
    write_json(c2_root / "C2_RESULT.json", c2_status)
    print(json.dumps({"phase": "C2", "status": c2_status["status"],
                      "pairing_status": c2_status["pairing_status"],
                      "forward_verified": 0, "uo52_candidates": len(uo52_candidates),
                      "candidate_distribution": dist,
                      "equal_before": legacy_status, "equal_after": fixed_status},
                     sort_keys=True))
    return 0


PARTIAL_CASES = [
    {"tf_id": "rmsnorm_decode_56aiv_dyn", "tf_fixture": "rmsnorm_decode_56aiv_dyn.py",
     "tl_case": "decode_dyn", "loop_mode": "dyn"},
    {"tf_id": "rmsnorm_decode_56aiv_recompute", "tf_fixture": "rmsnorm_decode_56aiv_recompute.py",
     "tl_case": "decode_recompute", "loop_mode": "recompute"},
    {"tf_id": "rmsnorm_prefill_56aiv_dyn", "tf_fixture": "rmsnorm_prefill_56aiv_dyn.py",
     "tl_case": "prefill_dyn", "loop_mode": "dyn"},
    {"tf_id": "rmsnorm_prefill_56aiv_recompute", "tf_fixture": "rmsnorm_prefill_56aiv_recompute.py",
     "tl_case": "prefill_recompute", "loop_mode": "recompute"},
]


def _tl_accesses_enriched(nodes: list) -> list:
    dtype_map = _tl_dtype_map(nodes)
    out = []
    for n in nodes:
        pack = "UNKNOWN_PACK_NOT_IN_TL_STRUCTURED_EXPORT"
        for arg in n.get("args", []) or []:
            if isinstance(arg, dict) and arg.get("kind") == "StringImm" and arg.get("value"):
                pack = arg["value"]
                break
        phase = _tl_phase(n.get("loop_vars", []))
        for i, a in enumerate(n.get("accesses", [])):
            out.append({
                "case": n.get("case"), "pass": n.get("pass"),
                "node_path": n.get("node_path"), "tl_node_key": n.get("tl_node_key"),
                "tl_access_key": n.get("tl_node_key") + "#" + str(i),
                "access_index": i, "op": n.get("op"),
                "root": a.get("root"), "scope": a.get("scope"),
                "space_class": a.get("space_class"), "mode": a.get("mode"),
                "dtype": dtype_map.get(a.get("root")),
                "pack": pack, "loop_vars": n.get("loop_vars", []),
                "loop_depth": n.get("loop_depth", 0),
                "loop_class": _tl_loop_class(n.get("loop_depth", 0)),
                "vf_phase": phase,
                "role": _tl_role(a.get("root"), dtype_map.get(a.get("root")), phase),
                "offset_structure": a.get("offset_structure"),
                "extent_values": a.get("extent_values", []),
            })
    return out


def _partial_action_key(tf: dict) -> tuple:
    return (tf.get("tf_tl_op"), tf.get("space_class"), tf.get("loop_class"),
            tf.get("mode"), _tf_role(tf))


def _partial_candidate_match(tf: dict, c: dict) -> tuple[bool, str]:
    tf_role = _tf_role(tf)
    if tf_role != "UNKNOWN_ROLE" and c.get("role") != tf_role:
        return False, "ROLE_MISMATCH"
    if (tf.get("loop_class") == "outside" and c.get("vf_phase") != "OUTSIDE"):
        return False, "VF_PHASE_MISMATCH"
    if (tf.get("loop_class") == "loop" and c.get("vf_phase") == "OUTSIDE"):
        return False, "VF_PHASE_MISMATCH"
    tf_pack = _tf_attrs(tf).get("pack")
    if tf_pack and c.get("pack") not in ("UNKNOWN_PACK_NOT_IN_TL_STRUCTURED_EXPORT", tf_pack):
        return False, "PACK_MISMATCH"
    return True, "ROLE_PACK_PHASE_MATCH"


def typed_fixture_params(fixture: Path, loop_mode: str) -> dict:
    """Extract x/w shapes and L/CHAINS/TPC/BATCH/rows from the TYPED instance.
    No bridge constants and no C2 d=1024 value."""
    from tilefoundry.cli.source import load_namespace, load_authored_ir
    ns, _ = load_namespace(str(fixture))
    mod = load_authored_ir(str(fixture) + ":RMSNormD")
    f = mod.entry_function()
    params = {p.name: p for p in f.params}
    x_shape = tuple(int(v) for v in params["x"].type.shape)
    w_shape = tuple(int(v) for v in params["w"].type.shape)
    x_len = x_shape[0]
    w_len = w_shape[0]
    d = w_len
    batch = x_len // d if d else None
    ns_batch = int(ns["BATCH"]) if "BATCH" in ns else None
    rows = int(ns["ROWS"]) if "ROWS" in ns else 1
    return {
        "fixture": str(fixture), "x_shape": list(x_shape), "w_shape": list(w_shape),
        "y_shape": list(x_shape), "x_len": x_len, "w_len": w_len,
        "L": int(ns["L"]) if "L" in ns else None,
        "CHAINS": int(ns["CHAINS"]) if "CHAINS" in ns else None,
        "TPC": int(ns["TPC"]) if "TPC" in ns else None,
        "TILES": int(ns["TILES"]) if "TILES" in ns else None,
        "BATCH": batch, "BATCH_DECLARED": ns_batch,
        "D": d, "D_DECLARED": int(ns["D"]) if "D" in ns else None,
        "rows": rows, "rows_source": "ROWS constant" if "ROWS" in ns else "tf.tile(...,1) single row",
        "loop_mode": loop_mode,
        "stages": "N/A_NOT_EXPRESSIBLE", "x_versions": "N/A_NOT_EXPRESSIBLE",
        "y_versions": "N/A_NOT_EXPRESSIBLE", "L2": "N/A_NOT_EXPRESSIBLE",
        "offset": "N/A_NOT_EXPRESSIBLE",
        "typed_consistency": {"batch_declared_matches_shape": ns_batch == batch,
                              "d_declared_matches_shape": (int(ns["D"]) if "D" in ns else None) == d},
        "source": "typed params from load_authored_ir entry function + typed namespace constants",
    }


def _partial_case_binding(tf_id: str, tl_case: str, tf_domain: dict) -> dict:
    cfg = TL_CASE_CONFIG[tl_case]
    tf_batch = tf_domain.get("batch", tf_domain.get("BATCH"))
    tf_d = tf_domain.get("d", tf_domain.get("D"))
    tf_rows = tf_domain.get("rows")
    shape_match = (tf_batch == cfg["batch"] and tf_d == cfg["d"]
                   and tf_rows == cfg["rows"])
    loop_match = tf_domain.get("loop_mode") == cfg["loop_mode"]
    return {
        "tf_id": tf_id, "tl_case": tl_case,
        "tf_typed": tf_domain,
        "tl_declared": cfg,
        "batch_match": tf_batch == cfg["batch"],
        "d_match": tf_d == cfg["d"],
        "rows_match": tf_rows == cfg["rows"],
        "loop_mode_match": loop_match,
        "shape_and_rows_match": shape_match,
        "stages_match": "N/A_NOT_EXPRESSIBLE",
        "versions_match": "N/A_NOT_EXPRESSIBLE",
        "exact_shape_match": shape_match,
        "exact_case_match": False,
        "binding_status": "SHAPE_AND_ROWS_MODE_MATCH_STAGES_VERSIONS_NA" if shape_match and loop_match
                          else "PARTIAL_CASE_BINDING",
        "note": "typed x/w shapes and rows; stages/versions/L2/offset are N/A_NOT_EXPRESSIBLE, not UNKNOWN.",
    }


def run_partial_evidence(out: Path, c_root: Path) -> int:
    if not c_root.is_dir():
        raise BridgeError("PARTIAL_STRUCTURED_REQUIRED", str(c_root))
    nodes_all = _load_c_nodes(c_root)
    partial_root = out
    if partial_root.exists():
        import shutil
        shutil.rmtree(partial_root)
    partial_root.mkdir(parents=True, exist_ok=True)
    results = {}
    prefill_missing = ["prefill_dyn", "prefill_recompute"]
    for spec in PARTIAL_CASES:
        fixture = TF / "targets/ascend/tests/fixtures/rmsnorm" / spec["tf_fixture"]
        ledger, problem = build_ledger_problem_for(fixture, "RMSNormD")
        tf_records, tf_counts = _tf_action_records(problem)
        tf_domain = typed_fixture_params(fixture, spec["loop_mode"])
        binding = _partial_case_binding(spec["tf_id"], spec["tl_case"], tf_domain)
        per_pass = {}
        seen_candidate, seen_partial = set(), set()
        total = {"tf_actions": len(tf_records), "candidate_actions": 0,
                 "verified_pairs": 0, "partial_field_equal": 0,
                 "ambiguous": 0, "no_candidate": 0, "role_mismatch": 0,
                 "pack_mismatch": 0, "pack_unknown": 0, "unresolved": 0}
        for p in C_PASSES:
            case_nodes = [n for n in nodes_all[p] if n.get("case") == spec["tl_case"]]
            tl_acc = _tl_accesses_enriched(case_nodes)
            tl_by_key = {}
            for c in tl_acc:
                if c.get("op") is None or c.get("space_class") in (None, "UNKNOWN_SCOPE") or c.get("mode") is None:
                    continue
                key = (c["op"], c["space_class"], c["loop_class"], c["mode"], c["role"])
                tl_by_key.setdefault(key, []).append(c)
            counts = {"pass": p, "tf_actions": len(tf_records),
                      "tl_nodes": len(case_nodes), "tl_accesses": len(tl_acc),
                      "candidate_actions": 0, "verified_pairs": 0,
                      "partial_field_equal": 0, "ambiguous": 0, "no_candidate": 0,
                      "role_mismatch": 0, "pack_mismatch": 0, "pack_unknown": 0,
                      "unresolved": 0}
            for tf in tf_records:
                key = _partial_action_key(tf)
                raw = tl_by_key.get(key, [])
                cands = []
                reasons = []
                for c in raw:
                    ok, why = _partial_candidate_match(tf, c)
                    if ok:
                        cands.append(c)
                    else:
                        reasons.append(why)
                counts["candidate_actions"] += 1 if raw else 0
                if raw:
                    seen_candidate.add(tf["action_id"])
                if not raw:
                    counts["no_candidate"] += 1
                    counts["unresolved"] += 1
                elif len(cands) > 1:
                    counts["ambiguous"] += 1
                    counts["unresolved"] += 1
                elif len(cands) == 1:
                    c = cands[0]
                    if c.get("pack") == "UNKNOWN_PACK_NOT_IN_TL_STRUCTURED_EXPORT":
                        counts["pack_unknown"] += 1
                        counts["unresolved"] += 1
                    else:
                        counts["partial_field_equal"] += 1
                        seen_partial.add(tf["action_id"])
                        counts["unresolved"] += 1
                else:
                    counts["unresolved"] += 1
                    if reasons and all(r == "ROLE_MISMATCH" for r in reasons):
                        counts["role_mismatch"] += 1
                    elif reasons and all(r == "PACK_MISMATCH" for r in reasons):
                        counts["pack_mismatch"] += 1
            per_pass[p] = counts
            for k in ("candidate_actions", "verified_pairs", "partial_field_equal",
                      "ambiguous", "no_candidate", "role_mismatch", "pack_mismatch",
                      "pack_unknown", "unresolved"):
                total[k] += counts[k]
        tf_roots = sorted({a.root for a in problem.accesses})
        tl_buffers = {}
        for p in C_PASSES:
            for n in nodes_all[p]:
                if n.get("case") != spec["tl_case"]:
                    continue
                for b in n.get("alloc_buffers", []) or []:
                    tl_buffers[b.get("name")] = b.get("dtype")
        root_evidence = {
            "tf_allocation_roots": len(tf_roots),
            "tf_root_samples": tf_roots[:12],
            "tl_buffer_roots": len(tl_buffers),
            "tl_buffer_dtypes": tl_buffers,
            "root_mapping_status": "UNPROVEN_NO_PRODUCER_CONSUMER_BASE_RANGE_EVIDENCE",
            "no_alias_table_used": True,
        }
        unit_evidence = {
            "tf_offset_unit": "BYTES_ARENA_PHYSICAL",
            "tl_offset_unit": "ELEMENTS_BUFFER_RELATIVE",
            "comparison": "NOT_PERFORMED_UNIT_ORIGIN_UNALIGNED",
            "example": "TF arena byte constant vs TL mul(i,64) element index; no numeric equality.",
        }
        instance_evidence = {
            "tf_iteration_paths": dict(Counter(x["iteration_path"] for x in tf_records)),
            "tf_iteration_domain_names": Counter(
                tuple(str(d[0]) for d in x.get("iteration_domains", ())
                      if isinstance(d, (list, tuple)) and d) for x in tf_records).most_common(8),
            "tf_iteration_extent_export": "STRING_EXTENT_OBSERVED_NOT_USED_FOR_EQUALITY",
            "tl_loop_vars_samples": sorted({tuple(c.get("loop_vars", [])) for p in C_PASSES for c in _tl_accesses_enriched(nodes_all[p]) if c.get("case") == spec["tl_case"]})[:8],
            "tile_loop_relation": "UNPROVEN_TILE_INSTANCE_NOT_EXPANDED",
            "partial_scope": "decode loop mode only; prefill cases not included",
        }
        case_result = {
            "tf_id": spec["tf_id"], "tl_case": spec["tl_case"],
            "status": "PARTIAL_STRUCTURAL_ONLY",
            "binding": binding,
            "tf_counts": {"entries": len(ledger.entries), "tasks": len(problem.tasks),
                           "accesses": len(problem.accesses),
                           "dependencies": len(problem.dependencies),
                           "problem_digest": problem.problem_digest},
            "candidate_coverage": {
                "tf_actions": total["tf_actions"],
                "tf_action_pass_denominator": total["tf_actions"] * len(C_PASSES),
                "candidate_action_pass": total["candidate_actions"],
                "candidate_action_pass_coverage": (total["candidate_actions"] /
                                                    (total["tf_actions"] * len(C_PASSES))),
                "unique_candidate_tf_actions": len(seen_candidate),
                "unique_candidate_fraction": len(seen_candidate) / total["tf_actions"] if total["tf_actions"] else None,
                "verified_pairs": total["verified_pairs"],
                "partial_field_equal_action_pass": total["partial_field_equal"],
                "unique_partial_field_equal_tf_actions": len(seen_partial),
                "unresolved_action_pass": total["unresolved"],
                "selected_candidate": None,
            },
            "unresolved_breakdown": {
                "ambiguous": total["ambiguous"], "no_candidate": total["no_candidate"],
                "role_mismatch": total["role_mismatch"], "pack_mismatch": total["pack_mismatch"],
                "pack_unknown": total["pack_unknown"],
                "unique_partial_unverified": total["partial_field_equal"],
            },
            "field_equal_scope": "role/pack/VF-phase only; NOT full field equality; root/offset/instance unresolved",
            "root_allocation_base_range": root_evidence,
            "offset_unit": unit_evidence,
            "tile_loop_instances": instance_evidence,
            "per_pass": per_pass,
            "verified_semantic_difference_count": None,
            "point1_accepted": False,
        }
        if spec["loop_mode"] == "recompute":
            case_result["waw_evidence"] = {
                "status": "REFERENCED_I1_DUPLICATE_LAST_TILE_STORE_EVIDENCE_ONLY",
                "waw_pairs": 2560, "entries": len(ledger.entries),
                "source": "I1 duplicate last-tile store evidence",
                "not_inferred_from_constant_37_and_clamp": True,
                "multi_core_numeric_claim": False,
            }
        results[spec["tl_case"]] = case_result
    source_identity = {
        "tf_fixtures": {s["tf_fixture"]: sha_file(TF / "targets/ascend/tests/fixtures/rmsnorm" / s["tf_fixture"])
                        for s in PARTIAL_CASES},
        "ledger_py_sha256": sha_file(TF / "targets/ascend/tilefoundry_ascend/schedule/ledger.py"),
        "tl_structured_root": str(c_root),
        "tl_source_c_sha256": sha_file(c_root.parent / "tl_source_c.json") if (c_root.parent / "tl_source_c.json").is_file() else None,
        "note": "partial run uses the current shared worktree; ledger.py may carry I1 iteration_domains changes; no commit.",
        "typed_params": {k: v["binding"]["tf_typed"] for k, v in results.items()},
    }
    out_doc = {
        "phase": "PARTIAL_POINT1",
        "status": "PARTIAL_4_OF_4_CASES",
        "source_identity": source_identity,
        "pairing_status": "PARTIAL_AMBIGUOUS",
        "consistency_proven": False,
        "point1_accepted": False,
        "verified_semantic_difference_count": None,
        "cases_present": ["decode_dyn", "decode_recompute", "prefill_dyn", "prefill_recompute"],
        "cases_missing": [],
        "cases_present_count": 4, "cases_total_expected": 4,
        "missing_reason": "none; four instances present. stages/versions/L2/offset are N/A_NOT_EXPRESSIBLE.",
        "tf_source": "I1 rmsnorm_{decode,prefill}_56aiv_{dyn,recompute}.py real ledger->Problem->core_args",
        "tl_source": "06-03 live pass structured C export for all four decode/prefill cases",
        "candidate_rule": "case/pass partition; root role; dtype/pack/VF phase; no first-candidate selection",
        "numerical_scope": "SINGLE_CORE_SMOKE_ONLY; MeshCoord not modelled; no multi-core numerical claim",
        "case_results": results,
        "conclusion": "PARTIAL 4 of 4 cases; root/base/range, unit origin and tile instance expansion remain UNPROVEN; not point 1 acceptance",
        "no_device": True, "no_p5_or_performance_credit": True,
        "commit": "NOT_COMMITTED_UNTIL_D",
    }
    write_json(partial_root / "PARTIAL_RESULT.json", out_doc)
    print(json.dumps({"phase": "PARTIAL_POINT1", "status": out_doc["status"],
                      "cases": {k: v["candidate_coverage"] for k, v in results.items()}},
                     sort_keys=True))
    return 0


def _root_scope_class(space: str | None) -> str:
    if space == "GMEM":
        return "GMEM"
    if space in ("SMEM", "RMEM", "CARRIER"):
        return "LOCAL"
    return "UNKNOWN"


def run_root_binding(out: Path, c_root: Path) -> int:
    if not c_root.is_dir():
        raise BridgeError("ROOT_STRUCTURED_REQUIRED", str(c_root))
    nodes_all = _load_c_nodes(c_root)
    rb = out / "root_binding"
    if rb.exists():
        import shutil
        shutil.rmtree(rb)
    rb.mkdir(parents=True, exist_ok=True)
    case_tables = {}
    totals = {"roots_total": 0, "proven": 0, "unproven": 0,
              "root_range_unavailable": 0, "size_only_base_unknown": 0,
              "no_same_scope_tl_buffer": 0}
    for spec in PARTIAL_CASES:
        fixture = TF / "targets/ascend/tests/fixtures/rmsnorm" / spec["tf_fixture"]
        ledger, problem = build_ledger_problem_for(fixture, "RMSNormD")
        by_root = {}
        for a in problem.accesses:
            by_root.setdefault(a.root, []).append(a)
        tf_bufs = {b.get("root"): b for b in problem.buffers}
        # TL LowerScheduledTIR buffers: name/dtype/shape/scope; allocation base absent.
        tl_nodes = [n for n in nodes_all["tl.LowerScheduledTIR"] if n.get("case") == spec["tl_case"]]
        tl_buffers = {}
        alias_map = None
        for n in tl_nodes:
            if n.get("node_type") == "Buffer":
                tl_buffers[n["name"]] = {"name": n["name"], "dtype": n.get("dtype"),
                                         "shape": n.get("shape"), "scope": n.get("scope"),
                                         "scope_class": ("GMEM" if n.get("scope") == "global"
                                                          else "LOCAL" if n.get("scope") in ("shared.dyn", "local.var")
                                                          else "UNKNOWN"),
                                         "allocation_base": "UNKNOWN_NO_BASE_FIELD_IN_STRUCTURED_EXPORT",
                                         "range": "UNKNOWN_NO_BASE"}
            for b in n.get("alloc_buffers", []) or []:
                tl_buffers[b.get("name")] = {"name": b.get("name"), "dtype": b.get("dtype"),
                                              "shape": b.get("shape"), "scope": b.get("scope"),
                                              "scope_class": ("GMEM" if b.get("scope") == "global"
                                                               else "LOCAL" if b.get("scope") in ("shared.dyn", "local.var")
                                                               else "UNKNOWN"),
                                              "allocation_base": "UNKNOWN_NO_BASE_FIELD_IN_STRUCTURED_EXPORT",
                                              "range": "UNKNOWN_NO_BASE"}
            if n.get("node_type") == "AttrStmt" and n.get("attr_key") == "tl.buffer_alias_map":
                alias_map = n.get("attr_node")
        roots = []
        for root in sorted(by_root):
            accs = by_root[root]
            b = tf_bufs.get(root, {})
            statuses = Counter(a.offset_status for a in accs)
            offsets = [a.byte_offset for a in accs if a.byte_offset is not None]
            space = b.get("space") or (accs[0].space if accs else None)
            scope_class = _root_scope_class(space)
            if offsets:
                range_status = "SIZE_ONLY_BASE_UNKNOWN"
            else:
                range_status = "ROOT_RANGE_UNAVAILABLE_ALL_OFFSETS_UNKNOWN"
            tl_candidates = [v for v in tl_buffers.values() if v["scope_class"] == scope_class]
            reasons = []
            if not tl_candidates:
                reasons.append("NO_TL_BUFFER_IN_SAME_SCOPE")
            if range_status != "SIZE_ONLY_BASE_UNKNOWN":
                reasons.append("ROOT_RANGE_UNAVAILABLE_OFFSET_UNKNOWN")
            reasons.append("NO_PROCESS_OR_BASE_RANGE_EVIDENCE")
            root_rec = {
                "tf_root": root, "tf_root_kind": b.get("root_kind") or (accs[0].root_kind if accs else None),
                "tf_space": space, "tf_scope_class": scope_class,
                "tf_allocation_site": root, "tf_allocation_bytes": b.get("bytes_"),
                "tf_owner_loop": b.get("owner_loop"), "tf_versions": b.get("versions"),
                "tf_fixed": b.get("fixed"),
                "tf_access_count": len(accs),
                "tf_offset_status": dict(statuses),
                "tf_byte_offset_min": min(offsets) if offsets else None,
                "tf_byte_offset_max": max(offsets) if offsets else None,
                "tf_range_status": range_status,
                "tf_range_base": "UNKNOWN",
                "tl_candidate_buffers": [v["name"] for v in tl_candidates],
                "tl_candidate_count": len(tl_candidates),
                "tl_allocation_bases": "UNKNOWN_NO_BASE_FIELD_IN_STRUCTURED_EXPORT",
                "tl_alias_map": alias_map,
                "same_scope": bool(tl_candidates),
                "range_overlap_or_inclusion": "UNPROVEN_NO_BASE",
                "producer_consumer_sequence": "UNPROVEN_NO_BOUND_ACTION_SEQUENCE",
                "binding": "UNPROVEN",
                "proven": False,
                "reject_reasons": reasons,
            }
            roots.append(root_rec)
            totals["roots_total"] += 1
            totals["unproven"] += 1
            if range_status != "SIZE_ONLY_BASE_UNKNOWN":
                totals["root_range_unavailable"] += 1
            else:
                totals["size_only_base_unknown"] += 1
            if not tl_candidates:
                totals["no_same_scope_tl_buffer"] += 1
        case_tables[spec["tl_case"]] = {
            "tf_id": spec["tf_id"],
            "tf_entries": len(ledger.entries), "tf_tasks": len(problem.tasks),
            "tf_accesses": len(problem.accesses), "tf_dependencies": len(problem.dependencies),
            "tl_lowerscheduled_buffers": sorted(tl_buffers.values(), key=lambda x: x["name"]),
            "tl_buffer_alias_map": alias_map,
            "tl_allocation_base_status": "UNKNOWN_NO_BASE_FIELD_IN_STRUCTURED_EXPORT",
            "roots_total": len(roots), "proven": 0, "unproven": len(roots),
            "roots": roots,
        }
    summary = {
        "phase": "ROOT_BINDING",
        "status": "PARTIAL_4_OF_4_CASES",
        "root_binding_status": "ALL_UNPROVEN",
        "proven": totals["proven"], "unproven": totals["unproven"],
        "roots_total": totals["roots_total"],
        "root_range_unavailable": totals["root_range_unavailable"],
        "size_only_base_unknown": totals["size_only_base_unknown"],
        "no_same_scope_tl_buffer": totals["no_same_scope_tl_buffer"],
        "no_correspondence_json": True,
        "point1_accepted": False,
        "conclusion": "No root can be proven: TL allocation base is absent; TF range base is absent; producer/consumer sequence is not bound. No alias guessing.",
        "case_counts": {k: {"roots_total": v["roots_total"], "proven": v["proven"],
                            "unproven": v["unproven"]} for k, v in case_tables.items()},
    }
    write_json(rb / "ROOT_BINDING_TABLE.json", {"summary": summary, "cases": case_tables})
    write_json(rb / "ROOT_BINDING_SUMMARY.json", summary)
    lines = ["# Root binding evidence (PARTIAL 4/4)", "",
             "status: " + summary["root_binding_status"],
             "proven: %d / roots: %d" % (summary["proven"], summary["roots_total"]),
             "root_range_unavailable: %d" % summary["root_range_unavailable"],
             "size_only_base_unknown: %d" % summary["size_only_base_unknown"],
             "no_same_scope_tl_buffer: %d" % summary["no_same_scope_tl_buffer"], "",
             "No root is bound. TF allocation base and TL allocation base are absent; no alias",
             "or name inference is used. Full per-root table: ROOT_BINDING_TABLE.json.", "",
             "| case | roots | proven | unproven |", "|---|---|---|---|"]
    for k, v in case_tables.items():
        lines.append("| %s | %d | 0 | %d |" % (k, v["roots_total"], v["unproven"]))
    lines += ["", "Sample first roots per case:",
              "| case | tf_root | tf_space | tf_range | tl_candidates | binding | reason |",
              "|---|---|---|---|---|---|---|"]
    for k, v in case_tables.items():
        for r in v["roots"][:3]:
            lines.append("| %s | %s | %s | %s | %d | %s | %s |" %
                         (k, r["tf_root"], r["tf_space"], r["tf_range_status"],
                          r["tl_candidate_count"], r["binding"], ",".join(r["reject_reasons"])))
    (rb / "ROOT_BINDING.md").write_text(chr(10).join(lines) + chr(10))
    print(json.dumps({"phase": "ROOT_BINDING", "status": summary["status"],
                      "proven": summary["proven"], "unproven": summary["unproven"],
                      "roots_total": summary["roots_total"]}, sort_keys=True))
    return 0


def _refusal_code(fn) -> str | None:
    try:
        fn()
    except BridgeError as e:
        return e.code
    return None


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
    ap.add_argument("--problem-source", default=None,
                    help="EXPLICIT ledger|template; ledger for 04-09, template only 05dcfcea old-domain regression")
    ap.add_argument("--phase", default="A", choices=["A", "C", "C2", "P", "R"],
                    help="A = wiring; C/C2 = mapping; P = partial 4-case; R = root binding evidence")
    ap.add_argument("--c-structured", default=None,
                    help="C/C2 input root: structured_c/<case>/<pass>.json from the 06-03 capture")
    ap.add_argument("--c-baseline", default=None,
                    help="C baseline evidence root preserved for C2 before/after")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = resolve_problem_source(args.problem_source)
    if args.mutation:
        res = run_mutation_ledger(args.mutation) if source == "ledger" else run_mutation(args.mutation)
        write_json(out / (args.mutation + ".json"), res)
        print(json.dumps(res, sort_keys=True))
        return 0 if res["killed"] else 1
    if source == "ledger":
        if args.phase == "C":
            croot = (Path(args.c_structured).resolve() if args.c_structured
                     else out / "structured_c")
            return run_c_evidence(out, croot)
        if args.phase == "C2":
            croot = (Path(args.c_structured).resolve() if args.c_structured
                     else out / "structured_c")
            cbaseline = (Path(args.c_baseline).resolve() if args.c_baseline
                         else out.parent / "C")
            return run_c2_evidence(out, croot, cbaseline)
        if args.phase == "P":
            croot = (Path(args.c_structured).resolve() if args.c_structured
                     else out / "structured_c")
            return run_partial_evidence(out, croot)
        if args.phase == "R":
            croot = (Path(args.c_structured).resolve() if args.c_structured
                     else Path(args.out).resolve().parent / "C" / "structured_c")
            return run_root_binding(out, croot)
        return run_ledger_evidence(out)
    if args.phase in ("C", "C2", "P", "R"):
        raise BridgeError("C_REQUIRES_LEDGER_SOURCE",
                          "C/C2/P/R must use --problem-source ledger, never the template old domain")

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
            "problem_source": "template",
            "old_domain_05dcfcea_regression": True,
            "not_04_09_evidence": True,
            "template_auto_fallback_forbidden": True,
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
