#!/usr/bin/env python3
"""04-06 probes for the vendored pure-Z3 core.

Checks (each run-produced in this run):
  1. IMPORT PURITY: a fresh process importing ONLY the vendored core must not have
     torch / torch_npu / tilelang / tvm_ffi in sys.modules (imported by file path, and,
     if possible, as a package).
  2. SINGLE PHYSICAL CORE: exactly one file in src/ + tools/ defines z3_schedule_python,
     and both adapters resolve to that same module object + same core file SHA.
  3. BEHAVIOURAL EQUIVALENCE: the vendored core returns exactly the same
     (start_times, sorted_indices) as the upstream TileLang function on a synthetic
     problem (upstream import is done in a SEPARATE process; it has FFI side effects).
  4. license/provenance present.

Run: python tools/p4/test_vendor_z3.py   (exit 0 iff all checks hold or a skipped probe
is explicitly labelled UNRUN with a reason and does not count as PASS)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TF = HERE.parents[1]
SRC = TF / "src"
VEND = SRC / "tilefoundry/vendor/tilelang_z3"
UP = Path("<user>/workspace/tilelang/tilelang/ascend/transform/z3_scheduler.py")
FAILS = 0


def check(cond, label):
    global FAILS
    if not cond:
        FAILS += 1
    print("%-92s %s" % (label, "PASS" if cond else "FAIL"))


def run(cmd, env_extra=None, timeout=180):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)


def main() -> int:
    # 1. import purity via file path
    probe = (
        "import sys;"
        "import importlib.util;"
        "spec=importlib.util.spec_from_file_location('vcore', %r);"
        "m=importlib.util.module_from_spec(spec);"
        "spec.loader.exec_module(m);"
        "bad=[k for k in sys.modules if k.split('.')[0] in ('torch','torch_npu','tilelang','tvm_ffi')];"
        "print('BAD='+','.join(sorted(bad)));"
        "print('SHA='+m.core_source_sha256())" % str(VEND / "core.py")
    )
    p = run([sys.executable, "-c", probe], env_extra={"PYTHONPATH": str(SRC)})
    out = p.stdout.strip().splitlines()
    bad = next((l[4:] for l in out if l.startswith("BAD=")), None)
    sha = next((l[4:] for l in out if l.startswith("SHA=")), None)
    check(p.returncode == 0 and bad == "",
          "import purity (file path): no torch/torch_npu/tilelang/tvm_ffi in sys.modules (BAD=%r)" % bad)
    check(bool(sha) and len(sha) == 64, "core file SHA computed at import (%s)" % sha)

    # package import (may route through tilefoundry/__init__; record what it drags in)
    pkg_probe = ("import sys;import tilefoundry.vendor.tilelang_z3 as v;"
                 "bad=[k for k in sys.modules if k.split('.')[0] in ('torch','torch_npu','tilelang','tvm_ffi')];"
                 "print('BAD='+','.join(sorted(bad)));print('SHA='+v.shared_core_sha256())")
    p = run([sys.executable, "-c", pkg_probe], env_extra={"PYTHONPATH": str(SRC)})
    print("package-import probe stdout:", p.stdout.strip().replace(chr(10), " | "))
    if p.returncode != 0:
        print("package import UNRUN:", p.stderr.strip().splitlines()[-1] if p.stderr else "?")
        check(False, "package import purity (UNRUN is not PASS): see stderr")
    else:
        bad2 = next((l[4:] for l in p.stdout.strip().splitlines() if l.startswith("BAD=")), "?")
        check(bad2 == "", "import purity (package): no torch/torch_npu/tilelang/tvm_ffi (BAD=%r)" % bad2)

    # 2. single physical core + adapters
    import ast
    defs = []
    for base in ("src", "tools"):
        for f in (TF / base).rglob("*.py"):
            try:
                tree = ast.parse(f.read_text())
            except Exception:
                continue
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))                         and node.name == "z3_schedule_python":
                    defs.append(str(f.relative_to(TF)))
    check(defs == ["src/tilefoundry/vendor/tilelang_z3/core.py"],
          "exactly one z3_schedule_python definition (AST; found %r)" % defs)
    sys.path.insert(0, str(SRC))
    import tilefoundry.vendor.tilelang_z3 as v  # noqa: E402
    from tilefoundry.vendor.tilelang_z3 import core  # noqa: E402
    check(v.tf_adapter.__module__ == v.tl_adapter.__module__,
          "TF and TL adapters live in one module (shared, not duplicated)")
    check(v.shared_core_sha256() == core.core_source_sha256(),
          "both adapters report the SAME physical core SHA")
    a = v.tf_adapter([4, 2, 3, 1], [1, 1, 1, 1], [2, 16, 16, 4],
                     [(0, 1, 0), (1, 2, 0), (2, 3, 0)], [(1, 2)])
    b = v.tl_adapter([4, 2, 3, 1], [1, 1, 1, 1], [2, 16, 16, 4],
                     [(0, 1, 0), (1, 2, 0), (2, 3, 0)], [(1, 2)])
    check(a == b, "TF adapter == TL adapter on a synthetic problem (%r)" % (a,))
    loop_defs = []
    for base in ("src", "tools"):
        for f in (TF / base).rglob("*.py"):
            try:
                tree = ast.parse(f.read_text())
            except Exception:
                continue
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))                         and node.name == "z3_schedule_loop_python":
                    loop_defs.append(str(f.relative_to(TF)))
    check(loop_defs == ["src/tilefoundry/vendor/tilelang_z3/core.py"],
          "exactly one z3_schedule_loop_python definition (AST; found %r)" % loop_defs)
    check(v.tf_loop_adapter.__module__ == v.tl_loop_adapter.__module__,
          "TF and TL LOOP adapters live in one module (shared, not duplicated)")
    check(v.shared_loop_core_sha256() == core.core_source_sha256(),
          "both LOOP adapters report the SAME physical core SHA")

    # 3. behavioural equivalence to upstream (separate process)
    # Load the upstream file BY PATH: importing the tilelang package fails in this CPU env
    # (needs a configured TVM build), but the single module only needs tvm_ffi + z3.
    up_probe = (
        "import json,importlib.util;"
        "spec=importlib.util.spec_from_file_location('upz3', %r);"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
        "r=m.z3_schedule_python([4,2,3,1],[1,1,1,1],[2,16,16,4],[(0,1,0),(1,2,0),(2,3,0)],[(1,2)]);"
        "print('RES='+json.dumps([list(r[0]),list(r[1])]))" % str(UP)
    )
    p = run([sys.executable, "-c", up_probe], env_extra={"PYTHONPATH": str(SRC)})
    if p.returncode != 0:
        print("upstream equivalence UNRUN:", (p.stderr.strip().splitlines() or ["?"])[-1])
        check(False, "behavioural equivalence to upstream (UNRUN is not PASS)")
    else:
        res = json.loads(next(l[4:] for l in p.stdout.strip().splitlines() if l.startswith("RES=")))
        check([list(a[0]), list(a[1])] == res,
              "vendor core == upstream z3_schedule_python on synthetic input (%r vs %r)"
              % ([list(a[0]), list(a[1])], res))

    # 4. provenance / license
    prov = (VEND / "PROVENANCE.md").read_text()
    check("8e2d3c823b0c3a01d2ce68d47dabfe98e35cc22b" in prov, "PROVENANCE binds upstream git HEAD")
    check("baa797dbede33091a0fe9f0754ed62d5561c8e048724e9586c3d58a323946580" in prov,
          "PROVENANCE binds upstream file sha256")
    check((VEND / "LICENSE").is_file() and "MIT License" in (VEND / "LICENSE").read_text(),
          "MIT license copied verbatim")
    print("RESULT:", "PASS" if FAILS == 0 else "FAILURES=%d" % FAILS)
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
