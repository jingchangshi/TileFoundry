#!/usr/bin/env bash
# Run the S solver-bridge tests with an interpreter that HAS z3.
#
# The bridge itself never imports z3; the shared vendor core does. This script is a
# guard so a no-z3 pytest venv can SKIP the module instead of failing, while the
# canonical run below proves the shared core is exercised and prints its version.
#
# Usage:
#   SOLVER_PY=/path/to/tfnpu-v2/bin/python tools/run_solver_tests.sh
#   SOLVER_PY=/path/to/tfnpu-dev/bin/python tools/run_solver_tests.sh
set -euo pipefail
: "${SOLVER_PY:?set SOLVER_PY to a Python interpreter with z3 (for example tfnpu-v2/bin/python)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVS_ROOT="${ENVS_ROOT:-$(cd "${ROOT}/.." && pwd)/envs}"
export PYTHONPATH="${ROOT}/src:${ROOT}/targets/ascend${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# ABI preflight: only tfnpu-v2 (judgment) or tfnpu-dev (same ABI + pytest) may run.
# A wrong interpreter dies by name (ABI_PREFLIGHT_INTERPRETER/VERSION/SHA).
"${SOLVER_PY}" "${ENVS_ROOT}/tfnpu-dev/abi_preflight.py"
"${SOLVER_PY}" - <<'PY'
import sys
try:
    import z3
except ImportError as exc:
    raise SystemExit("SOLVER_TESTS_NEED_Z3_INTERPRETER: %s" % exc)
print("z3:", z3.get_version_string(), "python:", sys.version.split()[0])
PY
exec "${SOLVER_PY}" "${ROOT}/targets/ascend/tests/test_solver_bridge_s0_s2.py" "$@"
