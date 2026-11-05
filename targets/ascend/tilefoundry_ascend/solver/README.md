# Solver bridge tests — interpreter matrix

The S bridge (`bridge.py`) never imports z3 itself; the ONE shared vendor core
(`tilefoundry.vendor.tilelang_z3.core`) does. So the test runner must be chosen by
whether z3 is present, and a pytest run without z3 must **skip** this module, not fail.

| interpreter kind | z3 | pytest | how to run |
|---|---|---|---|
| `tfnpu-v2` (canonical solver env) | 4.15.4 | no | `SOLVER_PY=<tfnpu-v2>/bin/python tools/run_solver_tests.sh` (direct runner; prints `z3.get_version_string()`) |
| `tilefoundry-npu` conda CPU env | 5.3.3 | no | same direct runner via `SOLVER_PY=...`; used for CPU evidence |
| `.venv-explorer` (pytest env) | none | 9.2 | `python -m pytest -q targets/ascend/tests/test_solver_bridge_s0_s2.py` → all **skipped** with reason `SOLVER_TESTS_NEED_Z3_INTERPRETER` |
| any pytest venv without z3 | none | any | same: module-level `pytestmark = skipif(not z3)`, named skip |

Canonical command (no machine-specific path is baked into the repo):

```bash
SOLVER_PY=/path/to/tfnpu-v2/bin/python tools/run_solver_tests.sh
# z3: 4.15.4 python: 3.12.16
# ... PASS test_... (12 tests)
```

Direct runner also refuses to run tests without z3 (exit 0 with a `SKIP ...` line), so
CI never turns a missing dependency into a red test. The full pytest suite therefore
reports this file as skipped on a no-z3 venv.

Evidence for S0-S2 lives in the pack at `artifacts/run03/w40_s/`.
