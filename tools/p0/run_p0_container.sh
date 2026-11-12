#!/usr/bin/env bash
# Canonical P0 recapture harness (versioned). A fresh replay copies this runbook.
set -euo pipefail

RUN_DIR="${1:?usage: run_p0_container.sh <new-run-dir>}"
TF_ROOT="${TF_ROOT:-${HOME}/workspace/TileFoundry}"
TL_ROOT="${TILELANG_ROOT:-${HOME}/workspace/tilelang}"
PKG_ROOT="${PKG_ROOT:-${HOME}/workspace/TileFoundry_Ascend950PR_AgentTeam}"
PY="${PY:-${HOME}/.conda/envs/tilefoundry-npu-dev/bin/python}"

[ -e "$RUN_DIR" ] && { echo "REFUSE: run dir exists ($RUN_DIR)"; exit 64; }

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TF_ROOT/src:$TF_ROOT/targets/ascend:$TL_ROOT:$TL_ROOT/testing:$PKG_ROOT/reference"
export LD_LIBRARY_PATH="$TL_ROOT/build/lib:${LD_LIBRARY_PATH:-}"
export P0_PROTOCOL="$PKG_ROOT/artifacts/run01/p0/precision_protocol.json"
export P0_KERNEL="$PKG_ROOT/reference/kernel_v33.py"
export TF_ROOT TILELANG_ROOT="$TL_ROOT"
export TF_TARGET_PATH="$TF_ROOT/targets/ascend"

# CANN must be sourced for npu-smi / bisheng (best effort; capture still proceeds).
if [ -f ${CANN_ROOT}/ascend-toolkit/set_env.sh ]; then
  export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
  set +u
  # shellcheck disable=SC1091
  source ${CANN_ROOT}/ascend-toolkit/set_env.sh
  set -u
fi

echo "run_id: $(basename "$RUN_DIR")"
echo "python: $PY"
"$PY" "$TF_ROOT/tools/p0/capture_p0.py" "$RUN_DIR"
