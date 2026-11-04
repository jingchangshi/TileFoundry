#!/usr/bin/env bash
# Record what this baseline actually ran on.
#
# A result is only as meaningful as the stack it was produced on, and this host
# is shared and mutable: the tree can carry uncommitted edits, the venv can be
# changed by another install, and CANN is a beta. A number without this file is
# a number nobody can reproduce or even interpret.
#
# Written ATOMICALLY: the content goes to a unique temporary file that is renamed
# into place, so a concurrent reader never sees a half-written manifest, and two
# concurrent manifest runs cannot corrupt each other's scratch state.
#
# Every fact that cannot be determined is recorded as UNKNOWN with the command
# that failed. "Clean" is only ever written when git actually said so -- an
# unreadable repository is not a clean one, and reporting it as clean is the
# exact false-clean this file exists to prevent.
#
# Exit codes
#   0  manifest written
#   2  the manifest could not be written at all
set -uo pipefail

HOST_REPO="${TF_REPO:-/home/s00653124/workspace/TileFoundry}"
VENV="${TF_VENV:-/home/s00653124/workspace/tfpy312}"
CANN_ROOT="${CANN_ROOT:-/data/pri/cann-9.2.0}"
CONTAINER="${TF_CONTAINER:-tf-ascend950pr}"
OUT="${1:-/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/artifacts/run02/p0/container/manifest.txt}"

log() { printf 'manifest.sh: %s\n' "$*" >&2; }
die() { log "$*"; exit 2; }

[ -d "$HOST_REPO/.git" ] || die "not a git repo: $HOST_REPO"
outdir="$(dirname -- "$OUT")"
mkdir -p "$outdir" || die "cannot create $outdir"

# Unique scratch, and unique enough that two runs started in the same second
# cannot collide. $$ alone is not unique across containers sharing a mount, so
# the PID is combined with the kernel's per-process start time when available.
scratch="$(mktemp "$outdir/.manifest.XXXXXX")" || die "cannot create scratch in $outdir"
trap 'rm -f "$scratch"' EXIT

emit() { printf '%s\n' "$@" >> "$scratch"; }

# Run a command, record its status, and return its output. The status is kept so
# a caller can distinguish "said nothing" from "could not be asked".
# Every git query goes through ONE helper that RETURNS the query's status as
# its own exit status, and is always told the repository with -C.
#
# Two ways the previous version could report a false clean, both fixed here:
#
#   * git_at set GIT_STATUS after the call. But the call was `$(git_at ...)`,
#     and a command substitution is a SUBSHELL: the assignment died with it,
#     so the parent's GIT_STATUS remained 0 and every failure looked like
#     success. A variable cannot carry a status out of a subshell; the exit
#     status can, so git_at now propagates it and the caller captures $? from
#     the assignment itself, immediately, in the parent.
#   * the status probe ran in the CALLER'S directory rather than HOST_REPO, so
#     it could describe a different repository entirely. Every query uses
#     git -C "$HOST_REPO".
#
# Statuses are captured into variables paired with their own exit code, and
# nothing is inferred from empty output: an empty answer to a failed query is
# not an empty answer to a good one.
git_at() {
    git -C "$HOST_REPO" "$@" 2>/dev/null
}

# run_git VAR: run a query, print its output, and set VAR plus VAR_rc.
# The $? is read from the assignment itself, in the parent, before anything
# else can overwrite it.
query() {
    local out
    out="$(git_at "$@")"
    local rc=$?
    printf '%s\n' "$out"
    return "$rc"
}

DIRTY_STATE="unknown"

# The probe is itself a checked query in HOST_REPO. If it fails, the state is
# UNKNOWN and no other claim is made about the tree.
probe_out="$(query status --porcelain)"
probe_rc=$?
if [ "$probe_rc" -ne 0 ]; then
    emit "dirty: UNKNOWN"
    emit "dirty_note: git status failed with status $probe_rc; the working tree"
    emit "dirty_note: state could NOT be determined. This is NOT a clean tree."
    emit "git_error: git -C $HOST_REPO status --porcelain -> exit $probe_rc"
    DIRTY_STATE="unknown"
else
    branch="$(query rev-parse --abbrev-ref HEAD)"; branch_rc=$?
    if [ "$branch_rc" -ne 0 ]; then
        emit "branch: UNKNOWN (git rev-parse --abbrev-ref HEAD -> exit $branch_rc)"
    else
        emit "branch: ${branch:-UNKNOWN}"
    fi

    commit="$(query rev-parse HEAD)"; commit_rc=$?
    if [ "$commit_rc" -ne 0 ]; then
        emit "commit: UNKNOWN (git rev-parse HEAD -> exit $commit_rc)"
    else
        emit "commit: ${commit:-UNKNOWN}"
    fi

    # probe_out was already captured above under its own status, so the tree
    # state is decided from THAT answer, not from a second uncontrolled query.
    if [ -n "$probe_out" ]; then
        emit "dirty: YES"
        emit "dirty_note: the commit above does NOT identify the source that ran;"
        emit "dirty_note: the working tree carried the changes below at capture time."
        emit "dirty_paths:"
        printf '%s\n' "$probe_out" | sed 's/^/  /' | while IFS= read -r line; do emit "$line"; done
        DIRTY_STATE="dirty"
    else
        emit "dirty: no"
        DIRTY_STATE="clean"
    fi

    tools_commit="$(query log -1 --format=%H -- examples/rmsnorm_950pr/tools)"; tools_rc=$?
    if [ "$tools_rc" -eq 0 ]; then
        emit "tool_commit_tools: ${tools_commit:-UNKNOWN}"
    else
        emit "tool_commit_tools: UNKNOWN (git log -> exit $tools_rc)"
    fi
    docker_commit="$(query log -1 --format=%H -- docker/ascend950pr)"; docker_rc=$?
    if [ "$docker_rc" -eq 0 ]; then
        emit "tool_commit_docker: ${docker_commit:-UNKNOWN}"
    else
        emit "tool_commit_docker: UNKNOWN (git log -> exit $docker_rc)"
    fi
fi
emit ""

emit "## Python environment"
emit "venv: $VENV"
if [ -x "$VENV/bin/python" ]; then
    emit "python: $("$VENV/bin/python" --version 2>&1)"
    emit "python_executable: $(readlink -f "$VENV/bin/python" 2>/dev/null || echo unknown)"
    if "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
        emit "deps_tool: pip freeze"
        emit "deps:"
        "$VENV/bin/python" -m pip freeze 2>/dev/null | sort | sed 's/^/  /' | while IFS= read -r line; do emit "$line"; done
    elif command -v uv >/dev/null 2>&1; then
        emit "deps_tool: uv pip list"
        emit "deps:"
        uv pip list --python "$VENV/bin/python" 2>/dev/null | sort | sed 's/^/  /' | while IFS= read -r line; do emit "$line"; done
    else
        emit "deps: UNKNOWN -- neither pip nor uv can inventory this venv"
    fi
    tf_ver="$(PYTHONPATH="$HOST_REPO/src:$HOST_REPO/targets/ascend" "$VENV/bin/python" -c 'import tilefoundry; print(getattr(tilefoundry, "__version__", "no-version"))' 2>&1)"
    emit "tilefoundry_import: $tf_ver"
    if "$VENV/bin/python" -c 'import torch_npu' >/dev/null 2>&1; then
        emit "torch_npu: importable"
    else
        emit "torch_npu: NOT INSTALLED -- no device execution is possible from this venv"
    fi
else
    emit "python: UNKNOWN at $VENV/bin/python"
fi
emit ""

emit "## CANN"
if [ -e "$CANN_ROOT" ]; then
    emit "cann_root: $CANN_ROOT"
    if [ -r "$CANN_ROOT/cann/set_env.sh" ]; then
        # The status of `source` is captured IMMEDIATELY after it, inside the
        # subshell. An earlier revision ran `source; printf ...` and read $?
        # outside, so the number recorded was the printf's -- always 0. A broken
        # set_env.sh would have been recorded as a working one.
        cann_env="$(
            set +u
            # shellcheck disable=SC1090
            . "$CANN_ROOT/cann/set_env.sh" >/dev/null 2>&1
            printf 'source_rc=%s\n' "$?"
            printf 'ASCEND_TOOLKIT_HOME=%s\n' "${ASCEND_TOOLKIT_HOME:-}"
        )"
        src_rc="$(printf '%s\n' "$cann_env" | sed -n 's/^source_rc=//p')"
        eff="$(printf '%s\n' "$cann_env" | sed -n 's/^ASCEND_TOOLKIT_HOME=//p')"
        if [ "${src_rc:-missing}" != "0" ]; then
            emit "cann_effective: UNKNOWN"
            emit "cann_error: sourcing set_env.sh -> exit ${src_rc:-missing}"
        elif [ -z "${eff:-}" ]; then
            emit "cann_effective: UNKNOWN"
            emit "cann_error: set_env.sh exited 0 but set no ASCEND_TOOLKIT_HOME"
        else
            emit "cann_effective_ascend_toolkit_home: $eff"
            if [ -e "$eff/version.cfg" ]; then
                emit "cann_version_cfg: $(head -3 "$eff/version.cfg" 2>/dev/null | tr '\n' ' ')"
            fi
        fi
    else
        emit "cann_effective: UNKNOWN"
        emit "cann_error: set_env.sh not readable under $CANN_ROOT/cann"
    fi
else
    emit "cann_root: UNKNOWN at $CANN_ROOT"
fi

driver="/usr/local/Ascend/driver"
if [ -e "$driver/version.info" ]; then
    emit "driver_version_info: $(head -3 "$driver/version.info" 2>/dev/null | tr '\n' ' ')"
elif [ -e "$driver" ]; then
    emit "driver: $driver present; version.info absent"
else
    emit "driver: UNKNOWN at $driver"
fi
emit ""

emit "## Kernel"
emit "kernel_release: $(uname -r)"
emit "kernel_build: $(uname -v)"
if [ -r /proc/version ]; then emit "proc_version: $(cat /proc/version)"; fi
emit ""

emit "## Container"
if command -v docker >/dev/null 2>&1; then
    cid="$(docker inspect "$CONTAINER" --format '{{.Id}}' 2>/dev/null || true)"
    if [ -n "$cid" ]; then
        emit "container: $CONTAINER"
        emit "container_id: $cid"
        emit "image: $(docker inspect "$CONTAINER" --format '{{.Config.Image}}' 2>/dev/null)"
        emit "image_id: $(docker inspect "$CONTAINER" --format '{{.Image}}' 2>/dev/null)"
        emit "privileged: $(docker inspect "$CONTAINER" --format '{{.HostConfig.Privileged}}' 2>/dev/null)"
        emit "devices:"
        docker inspect "$CONTAINER" --format '{{range .HostConfig.Devices}}  {{.PathOnHost}} -> {{.PathInContainer}}{{"\n"}}{{end}}' 2>/dev/null | sed '/^$/d' | while IFS= read -r line; do emit "$line"; done
    else
        emit "container: UNKNOWN -- $CONTAINER not found"
    fi
else
    emit "docker: UNKNOWN -- not available"
fi
emit ""

emit "## Isolation limits -- read this before quoting a result"
emit "devicenode_visibility: NOT security isolation."
emit "  Passing /dev/davinci0 narrows what docker exposes. It does not sandbox"
emit "  the NPU: the container shares the host driver, and the manager node"
emit "  /dev/davinci_manager is passed so the runtime can work. Any container"
emit "  holding the manager node can address more than device 0 at the driver"
emit "  level. Treat this as accident prevention, not containment."
emit "npu_mode: MUST enter as root on this platform."
emit "  CPU checks run as the ordinary account. Device work runs as root with"
emit "  --npu, deliberately and recorded -- not by omission."
emit "root_mode_is_not_device_proof: YES -- do not cite it as NPU evidence."
emit "  verify-container.sh runs the SAME CPU-side analyze/finalize/equality"
emit "  suite in both modes. --npu changes the identity it runs as, not what it"
emit "  runs. Its PASS is evidence about the root environment, not about the"
emit "  device. No device kernel has been executed by this container at all,"
emit "  and torch_npu is not installed in the venv."
emit "npu_runtime: NOT PROVEN."
emit "  torch_npu / the CANN runtime have not been exercised from inside this"
emit "  container. Everything verified so far is CPU-side analysis and finalize."
emit "shared_tenant: timings are PROVISIONAL."
emit "  Other tenants run on both NPUs. Numerics are unaffected; latency and"
emit "  throughput are not reproducible and must be labelled so."
emit ""

emit "## Verdict"
case "$DIRTY_STATE" in
    clean)
        emit "reproducible_from_commit: the recorded commit matches the tree" ;;
    dirty)
        emit "reproducible_from_commit: NO -- working tree was dirty at capture"
        emit "  The recorded commit does NOT identify the source that ran." ;;
    *)
        emit "reproducible_from_commit: UNKNOWN -- the tree state could not be read"
        emit "  This is NOT evidence of a clean tree. See Source above." ;;
esac

# Atomic publication: rename is the only step a reader can observe, so the file
# is either absent or complete.
if mv -f "$scratch" "$OUT" 2>/dev/null; then
    log "wrote $OUT ($(wc -l < "$OUT") lines)"
else
    die "could not publish $OUT"
fi
trap - EXIT
exit 0