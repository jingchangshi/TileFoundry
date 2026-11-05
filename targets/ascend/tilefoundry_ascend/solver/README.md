# Solver bridge tests — interpreter matrix

The S bridge (`bridge.py`) never imports z3 itself; the ONE shared vendor core
(`tilefoundry.vendor.tilelang_z3.core`) does. So the test runner must be chosen by
whether z3 is present, and a pytest run without z3 must **skip** this module, not fail.

| interpreter kind | z3 | pytest | how to run |
|---|---|---|---|
| `envs/tfnpu-v2` (ONLY formal env) | 4.15.4 | 9.2 (dev-only) | `SOLVER_PY=<envs/tfnpu-v2>/bin/python tools/run_solver_tests.sh` |
| `.conda/envs/tilefoundry-npu` | 5.3.3 | no | adapter 惯例位置，当前失效/遗留，暂停用；不删不改，待用户重构 |
| temporary one-off overlay | 4.15.4 inherited | 9.2 | derive from v2 with `venv --system-site-packages`, use once, then delete |

Canonical command (no machine-specific path is baked into the repo):

```bash
SOLVER_PY=/path/to/envs/tfnpu-v2/bin/python tools/run_solver_tests.sh
# ABI_PREFLIGHT PASS {'tilelang': '0.1.15+cpu.git0e687c17', 'apache-tvm-ffi': '0.1.12', 'z3-solver': '4.15.4.0'} z3-runtime=4.15.4
# z3: 4.15.4 python: 3.12.16
# ... PASS test_... (12 tests)
```

`tools/run_solver_tests.sh` calls `tools/abi_preflight.py` first: a wrong interpreter or
mismatched tilelang/tvm_ffi/z3 triple dies by
`ABI_PREFLIGHT_INTERPRETER` / `ABI_PREFLIGHT_VERSION` / `ABI_PREFLIGHT_SHA`.
A temporary need is a one-off `venv --system-site-packages` overlay from v2, deleted
after use; it never becomes a second formal environment.

Evidence for S0-S2 lives in the pack at `artifacts/run03/w40_s/`.
