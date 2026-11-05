#!/usr/bin/env python3
"""06-02 Twin candidate parameter ABI audit (CPU only, read-only).

Checks:
  A. expected ABI from run_all_cases.sh: shape/rows/loop_mode/stages/x_versions/y_versions
     per case; L2 NOTALLOC_CLEAN(read)/NORMAL_FV(write); core_count 56; eps 1e-6;
     offset default False (from the TF plan policy).
  B. cache_key is sha256(canonical asdict)[:16]; a declared key must match; every ABI field
     changes the key (so a wrong config cannot silently reuse another candidate's cache).
  C. deliberately wrong rows / loop_mode / versions / L2 / shape must be REJECTED by the
     config audit with a named code; wrong rows additionally changes the numeric result
     (4-chain vs 2-chain model on the known cancellation vector: 0 vs 1).
  D. computation independence: AST imports of twin_rmsnorm.py contain no tilefoundry /
     kernel_v33 / evaluator; optional --import-probe runs a clean subprocess and records
     whether importing the twin drags in those modules.
  E. emits the "candidate param <-> TF HIR/plan" correspondence table with statuses.

Run: python tools/p4/audit_twin_abi.py --out <json> [--import-probe]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TF = Path(__file__).resolve().parents[2]
TWIN = TF / "examples/rmsnorm_950pr/kernels/twin_rmsnorm.py"
RUNTIME = TF / "examples/rmsnorm_950pr/runtime_model.py"
RUNALL = TF / "examples/rmsnorm_950pr/tools/run_all_cases.sh"

EXPECTED = {
    "prefill_dyn": dict(shape=(8192, 4096), rows=4, loop_mode="dyn", stages=4, xv=4, yv=2),
    "prefill_recompute": dict(shape=(8192, 4096), rows=4, loop_mode="recompute", stages=4, xv=4, yv=2),
    "decode_dyn": dict(shape=(64, 4096), rows=1, loop_mode="dyn", stages=2, xv=2, yv=1),
    "decode_recompute": dict(shape=(64, 4096), rows=1, loop_mode="recompute", stages=2, xv=2, yv=1),
}
ALLOWED_SHAPES = {(8192, 4096): 4, (64, 4096): 1}
L2_READ, L2_WRITE = "NOTALLOC_CLEAN", "NORMAL_FV"
CORE_COUNT, EPS = 56, 1e-6

CODES = ("SHAPE_UNKNOWN", "ROWS_MISMATCH", "LOOP_MODE_INVALID", "STAGES_MISMATCH",
         "XVERSIONS_MISMATCH", "YVERSIONS_MISMATCH", "L2_READ_INVALID", "L2_WRITE_INVALID",
         "CORE_COUNT_MISMATCH", "EPS_MISMATCH", "CACHE_KEY_MISMATCH", "OFFSET_POLICY_NOT_FALSE")


def canonical_key(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def audit(cfg: dict, case_id: str, declared_key: str | None = None,
          offset_policy: str = "source_default") -> list[str]:
    bad = []
    exp = EXPECTED[case_id]
    shape = tuple(cfg["shape"])
    if shape not in ALLOWED_SHAPES:
        bad.append("SHAPE_UNKNOWN")
    if cfg["rows"] != ALLOWED_SHAPES.get(shape, exp["rows"]):
        bad.append("ROWS_MISMATCH")
    if cfg["loop_mode"] not in ("dyn", "recompute"):
        bad.append("LOOP_MODE_INVALID")
    if cfg["num_stages"] != exp["stages"]:
        bad.append("STAGES_MISMATCH")
    if cfg["x_versions"] != exp["xv"]:
        bad.append("XVERSIONS_MISMATCH")
    if cfg["y_versions"] != exp["yv"]:
        bad.append("YVERSIONS_MISMATCH")
    if cfg.get("l2_read", L2_READ) != L2_READ:
        bad.append("L2_READ_INVALID")
    if cfg.get("l2_write", L2_WRITE) != L2_WRITE:
        bad.append("L2_WRITE_INVALID")
    if cfg.get("core_count", CORE_COUNT) != CORE_COUNT:
        bad.append("CORE_COUNT_MISMATCH")
    if float(cfg.get("eps", EPS)) != EPS:
        bad.append("EPS_MISMATCH")
    if declared_key is not None and declared_key != canonical_key(cfg):
        bad.append("CACHE_KEY_MISMATCH")
    eff = "False" if offset_policy in ("source_default", "False", False) else str(offset_policy)
    if eff != "False":
        bad.append("OFFSET_POLICY_NOT_FALSE")
    return bad


def base_cfg(case_id: str) -> dict:
    e = EXPECTED[case_id]
    return {"case_id": case_id, "batch": e["shape"][0], "d": e["shape"][1],
            "shape": list(e["shape"]), "rows": e["rows"], "loop_mode": e["loop_mode"],
            "num_stages": e["stages"], "x_versions": e["xv"], "y_versions": e["yv"],
            "core_count": CORE_COUNT, "eps": EPS, "l2_read": L2_READ, "l2_write": L2_WRITE}


def independence(twin_src: str) -> dict:
    """Code-level independence: imports and NAME/ATTRIBUTE identifiers, NOT docstrings.

    A docstring may legitimately say "the TF side has its own evaluator"; that is prose,
    not a dependency. Only real code references count.
    """
    tree = ast.parse(twin_src)
    imports, names = [], set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    forbidden_imports = [m for m in imports
                         if "tilefoundry" in m or "kernel_v33" in m or "evaluator" in m]
    forbidden_names = sorted(n for n in names
                             if n in ("tilefoundry", "kernel_v33", "v33_base", "evaluator"))
    docstring_only = [w for w in ("tilefoundry", "kernel_v33", "evaluator", "v33_base")
                      if re.search(r"\b" + re.escape(w) + r"\b", twin_src)
                      and w not in forbidden_names]
    return {"imports": sorted(set(imports)), "forbidden_imports": forbidden_imports,
            "forbidden_identifiers_in_code": forbidden_names,
            "mentioned_in_docstring_only": docstring_only,
            "independent": not forbidden_imports and not forbidden_names}


def numeric_rows_demo() -> dict:
    """Known cancellation vector: 4-chain (rows=4) vs 2-chain (rows<=2) differ."""
    import numpy as np
    f32 = np.float32
    def bits(x):
        return int(np.asarray(x, dtype=np.float32).view(np.uint32))
    vals = [f32(2.0 ** 24), f32(1.0), f32(-(2.0 ** 24)), f32(0.0)]
    two = [f32(0.0), f32(0.0)]
    for i, v in enumerate(vals):
        two[i % 2] = f32(two[i % 2] + v)
    two_merge = bits(f32(two[0] + two[1]))
    four = [f32(0.0)] * 4
    for i, v in enumerate(vals):
        four[i % 4] = f32(four[i % 4] + v)
    four_merge = bits(f32(f32(four[0] + four[1]) + f32(four[2] + four[3])))
    return {"vector": ["0x4B800000", "0x3F800000", "0xCB800000", "0x00000000"],
            "two_chain_merge": "0x%08X" % two_merge,
            "four_chain_merge": "0x%08X" % four_merge,
            "differs": two_merge != four_merge,
            "note": "model-level numeric difference for wrong rows policy; not device output"}


CORRESPONDENCE = [
    {"twin_field": "batch,d (shape)", "twin_values": "8192x4096 / 64x4096",
     "tf_side": "PROPOSED scheduled.py CaseConfig; today only examples case dict (program.py:150-157)",
     "status": "PROPOSED / no public HIR entry yet"},
    {"twin_field": "rows", "twin_values": "4 / 1",
     "tf_side": "HIR per-row VF region / per-core rows policy; today CaseConfig.rows (program.py:24) from case dict",
     "status": "PROPOSED (04-05 must derive from HIR)"},
    {"twin_field": "x_versions,y_versions", "twin_values": "4,2 / 2,1",
     "tf_side": "plan ProblemBuffer.versions (problem.py:51-59) fed by CaseConfig (program.py:27-28)",
     "status": "BOUND to plan, but HIR policy retention is 04-03 gap"},
    {"twin_field": "num_stages", "twin_values": "4 / 2",
     "tf_side": "tf.pipelined num_stages; NOT preserved today",
     "status": "NO_TF_SOURCE (04-03)"},
    {"twin_field": "loop_mode", "twin_values": "dyn / recompute",
     "tf_side": "HIR loop policy; today only a case-config label",
     "status": "NO_TF_SOURCE (04-03)"},
    {"twin_field": "l2_read,l2_write", "twin_values": "NOTALLOC_CLEAN / NORMAL_FV",
     "tf_side": "none in HIR/plan",
     "status": "NO_TF_SOURCE"},
    {"twin_field": "core_count", "twin_values": "56",
     "tf_side": "target topology facts (tf_target.aiv_count); HIR 56-AIV ownership pending 03-24",
     "status": "PARTIAL: facts yes, HIR ownership no"},
    {"twin_field": "eps", "twin_values": "1e-6",
     "tf_side": "frozen protocol P0 (not HIR)",
     "status": "BOUND to protocol, not HIR"},
    {"twin_field": "offset policy", "twin_values": "backend default False",
     "tf_side": "plan policy effective_offset (problem.py:144-150)",
     "status": "BOUND to plan; TwinConfig has NO offset field (ABI gap, default assumed)"},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--import-probe", action="store_true")
    args = ap.parse_args()
    twin_src = TWIN.read_text()
    rt_src = RUNTIME.read_text()
    report = {"tool": "audit_twin_abi (06-02)", "twin_sha256": hashlib.sha256(twin_src.encode()).hexdigest(),
              "runtime_sha256": hashlib.sha256(rt_src.encode()).hexdigest(),
              "expected_abi": EXPECTED, "cases": [], "negatives": [],
              "cache_key_sensitivity": {}, "independence": independence(twin_src),
              "runtime_forbidden_refs": sorted(
                  {n.id for n in ast.walk(ast.parse(rt_src)) if isinstance(n, ast.Name)
                   and n.id in ("tilefoundry", "kernel_v33", "v33_base", "evaluator")}),
              "numeric_rows_demo": numeric_rows_demo(), "correspondence": CORRESPONDENCE}
    for case_id in EXPECTED:
        cfg = base_cfg(case_id)
        violations = audit(cfg, case_id)
        report["cases"].append({"case_id": case_id, "cfg": cfg,
                                "cache_key": canonical_key(cfg), "violations": violations,
                                "pass": not violations})
    # cache key sensitivity over every ABI field
    c0 = base_cfg("prefill_dyn")
    base = canonical_key(c0)
    for field, val in (("rows", 1), ("loop_mode", "recompute"), ("num_stages", 2),
                       ("x_versions", 2), ("y_versions", 1), ("l2_read", "NORMAL_FV"),
                       ("l2_write", "NOTALLOC_CLEAN"), ("core_count", 55), ("eps", 1e-5)):
        c = dict(c0); c[field] = val
        report["cache_key_sensitivity"][field] = canonical_key(c) != base
    # negatives that MUST be rejected
    negs = [
        ("wrong_rows_prefill", dict(c0, rows=1), "ROWS_MISMATCH"),
        ("wrong_loop_mode", dict(c0, loop_mode="serial"), "LOOP_MODE_INVALID"),
        ("wrong_stages", dict(c0, num_stages=2), "STAGES_MISMATCH"),
        ("wrong_versions", dict(c0, x_versions=2), "XVERSIONS_MISMATCH"),
        ("wrong_l2_read", dict(c0, l2_read="NORMAL_FV"), "L2_READ_INVALID"),
        ("wrong_shape", dict(c0, shape=[4096, 8192], batch=4096, d=8192), "SHAPE_UNKNOWN"),
    ]
    for name, cfg, want in negs:
        got = audit(cfg, "prefill_dyn")
        report["negatives"].append({"name": name, "expected": want, "got": got,
                                    "killed": want in got, "violations": got})
    # wrong declared cache key
    got = audit(c0, "prefill_dyn", declared_key="deadbeefdeadbeef")
    report["negatives"].append({"name": "wrong_cache_key", "expected": "CACHE_KEY_MISMATCH",
                                "got": got, "killed": "CACHE_KEY_MISMATCH" in got})
    report["negatives"].append({"name": "offset_policy_true", "expected": "OFFSET_POLICY_NOT_FALSE",
                                "got": audit(c0, "prefill_dyn", offset_policy=True),
                                "killed": "OFFSET_POLICY_NOT_FALSE" in audit(c0, "prefill_dyn", offset_policy=True)})
    if args.import_probe:
        probe = ("import sys, importlib.util;"
                 "sys.path.insert(0, %r);"
                 "spec=importlib.util.spec_from_file_location('twin', %r);"
                 "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
                 "bad=[k for k in sys.modules if k.split('.')[0] in ('tilefoundry','kernel_v33')"
                 " or 'evaluator' in k.split('.')[0]];"
                 "print('BAD='+','.join(sorted(bad)))") % (str(TWIN.parent), str(TWIN))
        # The tilelang root comes from the ENVIRONMENT. The literal "<user>/..." was a
        # placeholder that put a nonexistent path on PYTHONPATH and made the probe look
        # configured while resolving nothing.
        tl_root = (os.environ.get("TILELANG_ROOT") or "").strip()
        import_path = [str(TF / "src"), str(TF / "targets/ascend")]
        if tl_root:
            import_path.append(tl_root)
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(import_path),
                   PYTHONDONTWRITEBYTECODE="1", TORCH_DEVICE_BACKEND_AUTOLOAD="0")
        p = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env)
        bad = next((l[4:] for l in p.stdout.splitlines() if l.startswith("BAD=")), None)
        report["import_probe"] = {"returncode": p.returncode, "bad": bad,
                                  "tilelang_root": tl_root or None,
                                  "stdout": p.stdout.strip(), "stderr_tail": p.stderr.strip()[-400:]}
        if not tl_root:
            report["import_probe"]["skip_reason"] = "TILELANG_ROOT_UNSET"
    Path(args.out).write_text(json.dumps(report, indent=2) + chr(10))
    fails = [c["case_id"] for c in report["cases"] if not c["pass"]]
    killed = all(n["killed"] for n in report["negatives"])
    print("cases:", len(report["cases"]), "fail:", fails)
    print("negatives killed:", killed, "of", len(report["negatives"]))
    print("independence:", report["independence"]["independent"])
    print("cache_key_sensitive_all:", all(report["cache_key_sensitivity"].values()))
    print("numeric_rows_differ:", report["numeric_rows_demo"]["differs"])
    return 0 if not fails and killed and all(report["cache_key_sensitivity"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
