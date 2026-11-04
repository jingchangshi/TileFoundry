#!/bin/bash
# Serial four-case Twin device run on device 0.
#
# GOVERNANCE (do not weaken):
#  * Authorization is EXTERNAL. This script NEVER sets TF_DEVICE_AUTH; if it is not
#    already "1" in the environment, the script refuses. Authorization must be granted
#    by the caller (docker exec -e TF_DEVICE_AUTH=1), never self-granted.
#  * Occupancy is RECORDED whatever it shows, but the run REFUSES if the snapshot is
#    unreadable/failed or the device health is not OK. A BUSY card does NOT refuse
#    (shared card is allowed); it is recorded, not rejected.
#  * Evidence is append-only: CASES_DIR must NOT already exist. Each run writes a unique
#    RUN_ID. Reusing a directory is refused rather than overwritten.
set -euo pipefail

if [ "${TF_DEVICE_AUTH:-}" != "1" ]; then
  echo "REFUSE: TF_DEVICE_AUTH is not '1'. Authorization is supplied by the caller and is"
  echo "  never self-granted by this script. Invoke with:  docker exec -e TF_DEVICE_AUTH=1 ..."
  exit 75
fi

source /data/pri/cann-9.2.0/ascend-toolkit/set_env.sh >/dev/null 2>&1
export ASCEND_RT_VISIBLE_DEVICES=0
unset TORCH_DEVICE_BACKEND_AUTOLOAD || true
export PYTHONUNBUFFERED=1
REPO=/home/s00653124/workspace/TileFoundry
TL=/home/s00653124/workspace/tilelang
export PYTHONPATH="$TL:$REPO/src:$REPO/targets/ascend:$REPO/examples/rmsnorm_950pr"
export LD_LIBRARY_PATH="$TL/build/lib:${LD_LIBRARY_PATH:-}"
export TILELANG_CACHE_DIR=/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/artifacts/run02/tl_cache
# Dedicated DEVICE env (apache-tvm-ffi 0.1.12); the CPU toolchain keeps its own env.
PY=/home/s00653124/.conda/envs/tilefoundry-npu-dev/bin/python

CASES_DIR="${1:?usage: run_all_cases.sh CASES_DIR (must not already exist)}"
if [ -e "$CASES_DIR" ]; then
  echo "REFUSE: CASES_DIR already exists ($CASES_DIR)."
  echo "  Evidence is append-only; pass a NEW per-run directory instead of overwriting."
  exit 64
fi
mkdir -p "$CASES_DIR"
RUN_ID="$(date +%Y%m%dT%H%M%S)_$$"
echo "$RUN_ID" > "$CASES_DIR/RUN_ID.txt"
echo "RUN_ID=$RUN_ID"

# Occupancy: record whatever it shows; refuse only on FAILED/UNREADABLE/unhealthy.
snapshot() {
  local f="$1"
  if ! ( npu-smi info || /usr/local/sbin/npu-smi info ) > "$f" 2>&1; then
    echo "REFUSE: occupancy snapshot FAILED/UNREADABLE: $f"; return 3
  fi
  if ! grep -qi "Ascend950PR" "$f"; then
    echo "REFUSE: occupancy snapshot unparseable (no Ascend950PR row): $f"; return 3
  fi
  if ! awk '/Ascend950PR/{found=1; if ($0 ~ /OK/) ok=1} END{exit (found&&ok)?0:1}' "$f"; then
    echo "REFUSE: device health is not OK in $f"; return 4
  fi
  # Busy is allowed and recorded, not rejected.
  if grep -qi "No running processes" "$f"; then echo "  occupancy: device idle"; else echo "  occupancy: device BUSY (shared card allowed; recorded)"; fi
  return 0
}

LOCK=/home/s00653124/.tf_device0.lock
exec 9>"$LOCK"
flock -x 9
echo "=== device-lock acquired $(date -Is) ==="

# Preflight occupancy: must be readable + healthy before any case runs.
snapshot "$CASES_DIR/npu-smi.preflight.txt" || { echo "REFUSE: preflight occupancy check failed"; exit 4; }

cd "$REPO/examples/rmsnorm_950pr"

run() {
  local cid=$1; shift
  local d="$CASES_DIR/$cid"
  mkdir -p "$d"
  echo "=== $cid START $(date -Is) ==="
  if ! snapshot "$d/npu-smi.before.txt"; then
    echo "REFUSE: $cid occupancy (before) failed/unhealthy"; return 4
  fi
  set +e
  timeout 1800 "$PY" tools/run_case.py --case-id "$cid" --out "$d" "$@"
  local rc=$?
  set -e
  ( npu-smi info || /usr/local/sbin/npu-smi info ) > "$d/npu-smi.after.txt" 2>&1 || echo "after snapshot failed" > "$d/npu-smi.after.txt"
  echo "$rc" > "$d/exit_code.sh.txt"
  echo "=== $cid EXIT $rc $(date -Is) ==="
  return $rc
}

run prefill_dyn       --shape 8192 4096 --rows 4 --loop-mode dyn       --stages 4 --xversions 4 --yversions 2
run prefill_recompute --shape 8192 4096 --rows 4 --loop-mode recompute --stages 4 --xversions 4 --yversions 2
run decode_dyn        --shape 64 4096   --rows 1 --loop-mode dyn       --stages 2 --xversions 2 --yversions 1
run decode_recompute  --shape 64 4096   --rows 1 --loop-mode recompute --stages 2 --xversions 2 --yversions 1
echo "ALL_TWIN_CASES_DONE"
