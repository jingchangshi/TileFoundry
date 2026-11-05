# Solver bridge tests — interpreter matrix

The S bridge (`bridge.py`) never imports z3 itself; the ONE shared vendor core
(`tilefoundry.vendor.tilelang_z3.core`) does. So the test runner must be chosen by
whether z3 is present, and a pytest run without z3 must **skip** this module, not fail.

| interpreter kind | z3 | pytest | how to run |
|---|---|---|---|
| `tfnpu-v2` (ONLY judgment env) | 4.15.4 | no | `SOLVER_PY=<tfnpu-v2>/bin/python tools/run_solver_tests.sh` (direct runner; prints `z3.get_version_string()`) |
| `tfnpu-dev` (same ABI as v2, test overlay) | 4.15.4 | 9.2 | `envs/tfnpu-dev/bin/python -m pytest -q ...` or `SOLVER_PY=<tfnpu-dev>/bin/python tools/run_solver_tests.sh` |
| `.conda/envs/tilefoundry-npu` | 5.3.3 | no | **scratch/reference only**; never a judgment source |
| `.venv-explorer` | none | 9.2 | **scratch only**; pytest skips by `SOLVER_TESTS_NEED_Z3_INTERPRETER` |

Canonical command (no machine-specific path is baked into the repo):

```bash
SOLVER_PY=/path/to/envs/tfnpu-v2/bin/python tools/run_solver_tests.sh
# ABI_PREFLIGHT PASS {'tilelang': '0.1.15+cpu.git0e687c17', 'apache-tvm-ffi': '0.1.12', 'z3-solver': '4.15.4.0'} z3-runtime=4.15.4
# z3: 4.15.4 python: 3.12.16
# ... PASS test_... (12 tests)
```

`tools/run_solver_tests.sh` now calls `envs/tfnpu-dev/abi_preflight.py` first: a wrong
interpreter or mismatched tilelang/tvm_ffi/z3 triple dies by
`ABI_PREFLIGHT_INTERPRETER` / `ABI_PREFLIGHT_VERSION` / `ABI_PREFLIGHT_SHA`.
Direct runner also refuses to run tests without z3 (exit 0 with a `SKIP ...` line), so
CI never turns a missing dependency into a red test. The full pytest suite therefore
reports this file as skipped on a no-z3 venv.

Evidence for S0-S2 lives in the pack at `artifacts/run03/w40_s/`.
