#!/bin/bash
# Run all four Twin cases SERIALLY on device 0, one process each, under a HOST-WIDE
# flock so only one team measurement touches the NPU at a time. set -euo pipefail: a
# failing case stops the batch, because a failing case invalidates the comparison.
set -euo pipefail

# CANN must be sourced BEFORE torch_npu is imported (libhccl).
source /data/pri/cann-9.2.0/ascend-toolkit/set_env.sh >/dev/null 2>&1
# Device 0 only.
export ASCEND_RT_VISIBLE_DEVICES=0
# Device backend MUST load for device work: do NOT set TORCH_DEVICE_BACKEND_AUTOLOAD.
unset TORCH_DEVICE_BACKEND_AUTOLOAD || true
# Authorise run_case's device guard for this invocation only.
export TF_DEVICE_AUTH=1
export PYTHONUNBUFFERED=1

REPO=/home/s00653124/workspace/TileFoundry
TL=/home/s00653124/workspace/tilelang
export PYTHONPATH="$TL:$REPO/src:$REPO/targets/ascend:$REPO/examples/rmsnorm_950pr"
export LD_LIBRARY_PATH="$TL/build/lib:${LD_LIBRARY_PATH:-}"
export TILELANG_CACHE_DIR=/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/artifacts/run02/tl_cache
# Env of record for the device Twin (has tilelang + torch_npu); NOT the CPU env.
# Env of record: torch 2.12.0+cpu / torch_npu 2.12.0 (the CPU env of record), with
# apache-tvm-ffi 0.1.12 aligned to the in-tree tilelang build (0.1.15 mismatches the
# compiled libtvm_ffi.so and fails enum registration). twinfree at runtime.
PY=/home/s00653124/.conda/envs/tilefoundry-npu/bin/python

CASES_DIR="${1:?usage: run_all_cases.sh CASES_DIR}"
mkdir -p "$CASES_DIR"
cd "$REPO/examples/rmsnorm_950pr"

# Host-wide lock on a /home path (bind-mounted), so it serialises with any other run
# of ours across containers. The card is shared; one measurement at a time.
LOCK=/home/s00653124/.tf_device0.lock
exec 9>"$LOCK"
flock -x 9
echo "=== device-lock acquired $(date -Is) ==="

snap() { ( npu-smi info || /usr/local/sbin/npu-smi info ) > "$1" 2>&1 || echo "npu-smi failed" > "$1"; }

run() {
  local cid=$1; shift
  local d="$CASES_DIR/$cid"
  mkdir -p "$d"
  echo "=== $cid START $(date -Is) ==="
  snap "$d/npu-smi.before.txt"
  set +e
  timeout 1800 "$PY" tools/run_case.py --case-id "$cid" --out "$d" "$@"
  local rc=$?
  set -e
  snap "$d/npu-smi.after.txt"
  echo "$rc" > "$d/exit_code.sh.txt"
  echo "=== $cid EXIT $rc $(date -Is) ==="
  return $rc
}

run prefill_dyn       --shape 8192 4096 --rows 4 --loop-mode dyn       --stages 4 --xversions 4 --yversions 2
run prefill_recompute --shape 8192 4096 --rows 4 --loop-mode recompute --stages 4 --xversions 4 --yversions 2
run decode_dyn        --shape 64 4096   --rows 1 --loop-mode dyn       --stages 2 --xversions 2 --yversions 1
run decode_recompute  --shape 64 4096   --rows 1 --loop-mode recompute --stages 2 --xversions 2 --yversions 1
echo "ALL_TWIN_CASES_DONE"
