#!/usr/bin/env bash
# Verify the TileFoundry Ascend950PR container from INSIDE it.
#
# Every check prints its command, its output and its exit code. A check that
# cannot run is reported NOT_RUN with its reason and contributes a non-zero exit;
# nothing here is upgraded to PASS because a neighbour passed.
#
# Device checks are read-only and take a snapshot; they never reset, restart or
# signal anything, and they never touch another user's process.
#
# Exit codes
#   0  every required check passed
#   1  a required check failed
#   2  a required check could not run
#
# This script must be run INSIDE the container.
set -uo pipefail

CANN_ROOT="${CANN_ROOT:-/data/pri/cann-9.2.0}"
VENV_PY="${VENV_PY:-/home/s00653124/workspace/tfpy312/bin/python}"
REPO="${REPO:-/home/s00653124/workspace/TileFoundry}"

failures=0
runs=0

hr() { printf -- '--------------------------------------------------------------\n'; }

# check NAME COMMAND... : run it, print both streams, record the exit code.
check() {
    local name="$1"; shift
    hr
    printf '[%s]\n  $ %s\n' "$name" "$*"
    local out rc
    out="$("$@" 2>&1)"; rc=$?
    printf '%s\n' "$out" | sed 's/^/  /'
    printf '  -> exit %d\n' "$rc"
    runs=$((runs + 1))
    if [ "$rc" -ne 0 ]; then
        printf '  !! FAILED\n'
        failures=$((failures + 1))
    fi
    return 0
}

# not_run NAME REASON : a check that could not be attempted is not a pass.
not_run() {
    hr
    printf '[%s]\n  NOT_RUN: %s\n' "$1" "$2"
    runs=$((runs + 1))
    failures=$((failures + 1))
}

printf 'TileFoundry Ascend950PR container verification\n'
printf 'host: %s\n' "$(hostname)"
printf 'date: %s\n' "$(date -Is)"

# --- identity -------------------------------------------------------------
hr; printf '[identity]\n'; id | sed 's/^/  /'; uname -a | sed 's/^/  /'

# --- toolchain versions ---------------------------------------------------
check gcc gcc --version
check g++ g++ --version
check cmake cmake --version
check ninja ninja --version
check python3 python3 --version
check make make --version

# --- CANN is visible (mounted, not installed by us) -----------------------
if [ -r "$CANN_ROOT/cann/set_env.sh" ]; then
    check cann_set_env_present test -r "$CANN_ROOT/cann/set_env.sh"
    hr; printf '[cann] set_env.sh head\n'
    head -5 "$CANN_ROOT/cann/set_env.sh" | sed 's/^/  /'
else
    not_run cann_set_env "no readable set_env.sh under $CANN_ROOT/cann"
fi

# Sourcing set_env.sh must not fail and must populate the toolchain paths.
if [ -r "$CANN_ROOT/cann/set_env.sh" ]; then
    hr; printf '[cann] sourcing set_env.sh\n'
    # shellcheck disable=SC1090
    ( set +u; . "$CANN_ROOT/cann/set_env.sh" >/dev/null 2>&1; printf '  ASCEND_TOOLKIT_HOME=%s\n' "${ASCEND_TOOLKIT_HOME:-<unset>}"; printf '  ASCEND_HOME_PATH=%s\n' "${ASCEND_HOME_PATH:-<unset>}"; printf '  PATH has cann: %s\n' "$(printf '%s' "$PATH" | grep -c cann)" )
    src_rc=$?
    printf '  -> exit %d\n' "$src_rc"
    [ "$src_rc" -eq 0 ] || failures=$((failures + 1))
fi

# --- the mounted venv -----------------------------------------------------
if [ -x "$VENV_PY" ]; then
    check venv_python "$VENV_PY" --version
    check venv_import_tilefoundry env PYTHONPATH="$REPO/src:$REPO/targets/ascend" "$VENV_PY" -c 'import tilefoundry; print("tilefoundry", tilefoundry.__version__ if hasattr(tilefoundry, "__version__") else "imported")'
else
    not_run venv_python "no interpreter at $VENV_PY (is the venv bind-mounted?)"
fi

# --- TileFoundry against copy_slice through the mounted venv --------------
if [ -x "$VENV_PY" ] && [ -d "$REPO/src" ]; then
    export PYTHONPATH="$REPO/src:$REPO/targets/ascend"
    check analyze "$VENV_PY" -m tilefoundry.cli analyze "$REPO/examples/rmsnorm_950pr/copy_slice.py:CopySlice"
    tir_out="$(mktemp -d)/copy_resolved_tir.py"
    check finalize "$VENV_PY" -m tilefoundry.cli schedule finalize "$REPO/examples/rmsnorm_950pr/copy_slice.py:CopySlice" "$tir_out"
    if [ -f "$tir_out" ]; then
        hr; printf '[finalize] emitted TIR (first 25 lines, %s bytes total)\n' "$(wc -c < "$tir_out")"
        head -25 "$tir_out" | sed 's/^/  /'
    else
        not_run finalize_output "finalize produced no file"
    fi
else
    not_run tilefoundry_flow "venv interpreter or repo tree missing"
fi

# --- device visibility (read-only) ----------------------------------------
hr; printf '[device] nodes visible in this container\n'
for node in /dev/davinci0 /dev/davinci1 /dev/davinci_manager /dev/hisi_hdc /dev/devmm_svm; do
    if [ -e "$node" ]; then printf '  %-24s present\n' "$node"; else printf '  %-24s absent\n' "$node"; fi
done
hr; printf '[device] ASCEND_RT_VISIBLE_DEVICES=%s\n' "${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"

hr
printf 'checks run: %d, failures: %d\n' "$runs" "$failures"
if [ "$failures" -ne 0 ]; then
    printf 'RESULT: FAIL\n'
    exit 1
fi
printf 'RESULT: PASS\n'
exit 0
