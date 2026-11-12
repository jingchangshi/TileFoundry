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
# -u is required, not optional: an unset variable must be an error here rather
# than an empty string. Several checks below ask whether something is SET, so a
# silently empty CANN root would make "set_env.sh missing" read as "CANN fine",
# which is the failure mode this script exists to prevent.
set -uo pipefail

CANN_ROOT="${CANN_ROOT:-${CANN_ROOT}}"
VENV_PY="${VENV_PY:-${HOME}/workspace/tfpy312/bin/python}"
REPO="${REPO:-${HOME}/workspace/TileFoundry}"

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

# entry.sh creates and verifies the container account but does NOT drop to it:
# the entrypoint keeps running as root and `docker exec` runs as root too unless
# told otherwise.
#
# Two modes, and the platform decides which one is correct:
#
#   default (CPU)   runs as the ordinary container account, via `docker exec -u`.
#                   This is the required mode for analyze / finalize / equality,
#                   which never touch the device.
#   --npu           runs as root. Running NPU cases MUST enter as root on this
#                   platform; a non-root NPU attempt is not representative, so it
#                   is not silently attempted either.
#
# Root WITHOUT --npu is an accident -- a forgotten `-u` -- and is refused: every
# check below would otherwise describe a root environment while looking identical
# to the intended one. --npu is never implied; it has to be asked for.
MODE=cpu
for arg in "$@"; do
    case "$arg" in
        --npu) MODE=npu ;;
        -h|--help)
            sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
            printf '\nUsage: %s [--npu]   (run inside the container)\n' "$0"
            exit 0 ;;
        *) printf 'verify-container.sh: unknown argument: %s\n' "$arg" >&2; exit 2 ;;
    esac
done

IS_ROOT=no
[ "$(id -u)" -eq 0 ] && IS_ROOT=yes

if [ "$MODE" = npu ]; then
    if [ "$IS_ROOT" != yes ]; then
        # Asked for device work but not entered as root. On this platform that
        # attempt would fail or, worse, half-succeed and look like a result.
        printf 'verify-container.sh: --npu requires running as root.\n' >&2
        printf '  NPU cases on this platform must enter as root.\n' >&2
        printf '  Re-run without -u:\n' >&2
        printf '    docker exec <container> bash %s --npu\n' "$0" >&2
        exit 2
    fi
    printf '================================================================\n'
    printf ' NPU MODE -- RUNNING AS ROOT\n'
    printf ' ================================================================\n'
    printf ' Reason: on this platform the NPU runtime requires root to enter.\n'
    printf ' A non-root NPU attempt is not representative, so this mode is\n'
    printf ' entered deliberately, not by omission.\n'
    printf ' NOTE: running as root is NOT device proof. Every check below is\n'
    printf ' the same CPU-side analyze/finalize/equality suite, executed as root.\n'
    printf ' No device kernel is executed by this script in either mode.\n'
    printf ' Scope: device access only. This does not license driver or CANN\n'
    printf ' changes, and it does not make the container a sandbox.\n'
    printf ' changes, and it does not make the container a sandbox.\n'
    printf ' Mode: npu; user: %s (uid %s)\n' "$(id -un)" "$(id -u)"
    printf '================================================================\n'
else
    if [ "$IS_ROOT" = yes ]; then
        printf 'verify-container.sh: refusing to run as root without --npu.\n' >&2
        printf '  entry.sh does not drop privileges, so this means `docker exec`\n' >&2
        printf '  was invoked without -u. CPU checks must run as the container\n' >&2
        printf '  account; root is reserved for device work and must be asked for.\n' >&2
        printf '  Re-run as the container account:\n' >&2
        printf '    docker exec -u <user> <container> bash %s\n' "$0" >&2
        printf '  Or, for an on-device case:\n' >&2
        printf '    docker exec <container> bash %s --npu\n' "$0" >&2
        exit 2
    fi
fi

printf 'TileFoundry Ascend950PR container verification\n'
printf 'host: %s\n' "$(hostname)"
printf 'date: %s\n' "$(date -Is)"
    printf 'mode: %s\n' "$MODE"
    printf 'device_kernel_executed: no\n'
printf 'user: %s (uid %s)\n' "$(id -un)" "$(id -u)"


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

# Sourcing set_env.sh must succeed AND must populate the toolchain paths.
#
# The status of `source` is captured inside the subshell, immediately after it
# and on its own statement. An earlier revision ran `source; printf ...` and
# then read $? outside, so the number reported was the exit status of the
# printf -- always 0. A broken set_env.sh would have been recorded as a pass.
# The assertions about what it exported are checked separately from the
# status, so neither can hide the other.
if [ -r "$CANN_ROOT/cann/set_env.sh" ]; then
    hr; printf '[cann] sourcing set_env.sh\n'
    cann_env="$(
        set +u
        # shellcheck disable=SC1090
        . "$CANN_ROOT/cann/set_env.sh" >/dev/null 2>&1
        printf 'source_rc=%s\n' "$?"
        printf 'ASCEND_TOOLKIT_HOME=%s\n' "${ASCEND_TOOLKIT_HOME:-<unset>}"
        printf 'ASCEND_HOME_PATH=%s\n' "${ASCEND_HOME_PATH:-<unset>}"
        printf 'PATH_has_cann=%s\n' "$(printf '%s' "$PATH" | grep -c cann)"
    )"
    printf '%s\n' "$cann_env" | sed 's/^/  /'
    src_rc="$(printf '%s\n' "$cann_env" | sed -n 's/^source_rc=//p')"
    if [ "${src_rc:-missing}" != "0" ]; then
        printf '  !! sourcing set_env.sh failed with status %s\n' "${src_rc:-missing}"
        failures=$((failures + 1))
    fi
    toolkit_home="$(printf '%s\n' "$cann_env" | sed -n 's/^ASCEND_TOOLKIT_HOME=//p')"
    if [ -z "${toolkit_home:-}" ] || [ "${toolkit_home:-}" = "<unset>" ]; then
        printf '  !! set_env.sh succeeded but did not export ASCEND_TOOLKIT_HOME\n'
        failures=$((failures + 1))
    fi
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
printf 'SCOPE: %s-mode checks; no device kernel executed\n' "$MODE"
exit 0