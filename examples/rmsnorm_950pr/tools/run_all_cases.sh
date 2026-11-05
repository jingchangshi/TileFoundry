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

source "${CANN_ROOT:?CANN_ROOT must be set}/ascend-toolkit/set_env.sh" >/dev/null 2>&1
export ASCEND_RT_VISIBLE_DEVICES=0
unset TORCH_DEVICE_BACKEND_AUTOLOAD || true
export PYTHONUNBUFFERED=1
REPO="${HOME}/workspace/TileFoundry"
AG_ADAPTER="${REPO%/TileFoundry}/AgentGates/agent_gates/adapters/tilefoundry-npu/scripts/env_context.py"

# --- tilelang comes from the INSTALLED WHEEL in the shared prefix, NEVER a source
# checkout. With .../workspace/tilelang first on PYTHONPATH the device run imports the
# checkout, and then the prefix passing its four layers says nothing about what ran on
# the card. Resolution order is explicit and has NO source-tree fallback:
#   1. TILELANG_ROOT -- an ENVIRONMENT root that owns an installed wheel,
#   2. the adapter env context SHARED_PREFIX (one source of truth with configure.py),
#   3. refuse by name: TILELANG_ROOT_UNSET.
adapter_prefix() {
  [ -f "$AG_ADAPTER" ] || return 0
  local helper
  helper="$(command -v python3 || true)"
  [ -n "$helper" ] || return 0
  "$helper" - "$AG_ADAPTER" <<'PY' 2>/dev/null || true
import runpy, sys
print(str(runpy.run_path(sys.argv[1]).get("SHARED_PREFIX") or ""))
PY
}
TL="${TILELANG_ROOT:-}"
if [ -z "$TL" ]; then TL="$(adapter_prefix)"; fi
if [ -z "$TL" ]; then
  echo "REFUSE: TILELANG_ROOT_UNSET -- neither TILELANG_ROOT nor the adapter env context"
  echo "  ($AG_ADAPTER) names the prefix that owns the installed tilelang wheel."
  echo "  Falling back to a source checkout would measure code nobody deploys."
  exit 78
fi
export TILELANG_ROOT="$TL"
# NOTE: $TL is deliberately NOT on PYTHONPATH -- the wheel is imported from the
# prefix site-packages, and that is the artefact the four-layer check verified.
export PYTHONPATH="$REPO/src:$REPO/targets/ascend:$REPO/examples/rmsnorm_950pr"
export TILELANG_CACHE_DIR="${HOME}/workspace/TileFoundry_Ascend950PR_AgentTeam/artifacts/run02/tl_cache"
# The DEVICE interpreter is the one that OWNS the wheel. The old conda env has no
# tilelang at all, so its device env only ever worked because the source tree was on
# PYTHONPATH -- the exact decoupling this change removes. Override with TFNU_PYTHON.
PY="${TFNU_PYTHON:-$TL/bin/python}"
if [ ! -x "$PY" ]; then
  echo "REFUSE: no interpreter at $PY (the wheel prefix has none; set TFNU_PYTHON)"
  exit 78
fi

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

# CANONICAL lock path (Lead ruling D4): the ONE lock shared by every team device job,
# on the host and inside the container (container /home is the host /home). It is NOT
# a per-user lock under $HOME: two teammates keeping two different locks is the same as
# keeping none. Created once as root inside the container, mode 0666, so both the host
# user and root-in-container can open it. One holder per run: this script acquires it
# once and passes no nested acquisition down to run_case.py.
LOCK=/home/.tf_device0.lock
if [ ! -e "$LOCK" ]; then
  echo "REFUSE: canonical device lock $LOCK is missing; create it (root, mode 0666) instead of"
  echo "  silently falling back to a per-user lock."
  exit 78
fi
exec 9>"$LOCK"
flock -x 9
# One holder only: children must NOT re-acquire the same lock (run_case.py checks this).
export TF_DEVICE_LOCK_HELD=1
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
