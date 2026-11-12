#!/usr/bin/env bash
# Run one command against the Ascend950PR device, serialized correctly.
#
# Two things this exists to fix.
#
# 1. A lock inside the container does not serialize anything.
#    /tmp inside a container is a different inode from /tmp on the host, so a
#    flock taken there excludes nobody. The lock MUST live on a host path that
#    the container also sees through a bind mount. This script takes the lock on
#    the host, around the `docker exec`, and the container therefore cannot
#    acquire it concurrently -- the serialization happens at the only place both
#    sides agree on.
#
# 2. Device work must be attributable.
#    An npu-smi snapshot is written before and after every run, the raw output is
#    kept, and nothing is signalled, killed or reset. Other tenants on a shared
#    host are never disturbed by this script.
#
# Usage:
#     ./device-run.sh --name LABEL -- COMMAND [ARGS...]
#     ./device-run.sh --name baseline -- docker exec -u U CONTAINER bash ...
#
# Exit code is the command's own, so a caller can gate on it. If the lock
# cannot be acquired the script exits 75 (EX_TEMPFAIL) and runs nothing.
set -uo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROG="device-run.sh"

# Host-shared and stable. The evidence directory is bind-mounted into the
# container, so this path names the SAME inode from both sides -- which is the
# entire point. Override consciously; changing it silently disables the lock.
LOCK="${TF_DEVICE_LOCK:-/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/artifacts/run02/p0/container/.device.lock}"  # no-machine-path: allow
# Evidence goes next to the lock by default: the lock already lives on the shared
# mount, so its directory is the one path known to be identical on both sides.
OUTDIR="${TF_DEVICE_OUTDIR:-$(dirname -- "$LOCK")}"
LOCK_TIMEOUT="${TF_DEVICE_LOCK_TIMEOUT:-900}"

log() { printf '%s: %s\n' "$PROG" "$*" >&2; }
die() { log "$*"; exit 2; }

NAME=""
NPU_MODE=0
FORCE_NEW_ID=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --name) NAME="${2:-}"; [ -n "$NAME" ] || die '--name needs a label'; shift 2 ;;
        --npu) NPU_MODE=1; shift ;;
        --new-id) FORCE_NEW_ID=1; shift ;;
        --) shift; break ;;
        -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1 (try --help)" ;;
    esac
done

[ "$#" -gt 0 ] || die 'no command given; put it after --'

# The platform decides the correct identity: CPU checks run as the ordinary
# account, NPU work MUST enter as root. --npu records which one this run is, so
# the evidence says it rather than leaving a reader to infer it from the argv.
if [ "$NPU_MODE" = 1 ]; then
    MODE=npu
else
    MODE=cpu
fi
[ -n "$NAME" ] || die '--name is required (the label names the evidence files)'
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "label is not a plain name: $NAME"

mkdir -p "$OUTDIR" || die "cannot create evidence directory: $OUTDIR"

# Serialize. The lock path is on the shared mount, so a second invocation --
# from this host or from inside the container -- blocks here rather than
# racing onto the device.
exec 9>"$LOCK" || die "cannot open lock file: $LOCK"
if ! flock -w "$LOCK_TIMEOUT" 9; then
    log "could not acquire $LOCK within ${LOCK_TIMEOUT}s; running nothing"
    exit 75
fi
log "holding $LOCK"

# Reserve this run id ATOMICALLY, and never reuse one.
#
# Per-run labels only avoid collisions if the ids are unique, and a label alone
# does not make them so: a second `--name baseline` would silently truncate the
# first run's evidence. mkdir is the atomic primitive available here -- it either
# creates the directory or fails because it exists -- so the run directory is
# claimed by mkdir, and an existing id is refused rather than overwritten.
#
# The run id embeds a timestamp and the PID so two invocations that reuse a label
# are told apart rather than racing: the FIRST keeps the bare name, a later one is
# refused unless it asks for a fresh id with --new-id.
RUNDIR="$OUTDIR/runs/$NAME"
if [ "$FORCE_NEW_ID" = 1 ]; then
    RUNDIR="$OUTDIR/runs/$NAME.$(date -u +%Y%m%dT%H%M%SZ).$$"
fi
mkdir -p "$OUTDIR/runs" || die "cannot create $OUTDIR/runs"
if ! mkdir "$RUNDIR" 2>/dev/null; then
    log "run id already exists: $NAME"
    log "  evidence lives at: $RUNDIR"
    log "  refusing to overwrite a previous run's evidence."
    log "  Use a different --name, or --new-id to allocate a fresh one."
    exit 78
fi
trap 'rmdir "$RUNDIR" 2>/dev/null || true' EXIT

# Attribution: who holds the lock, and what the device looked like on arrival.
{
    printf 'run: %s\n' "$NAME"
    printf 'mode: %s\n' "$MODE"
    printf 'mode_reason: %s\n' "$([ "$MODE" = npu ] && echo 'NPU work must enter as root on this platform' || echo 'CPU-only work runs as the ordinary container account')"
    printf 'started: %s\n' "$(date -Is)"
    printf 'host: %s\n' "$(hostname)"
    printf 'user: %s\n' "$(id -un)"
    printf 'uid: %s\n' "$(id -u)"
    printf 'lock: %s\n' "$LOCK"
    printf 'run_dir: %s\n' "$RUNDIR"
    printf 'command:'
    printf ' %q' "$@"
    printf '\n'
} > "$RUNDIR/attribution.txt"

if [ "$MODE" = npu ]; then
    log '================================================================'
    log ' NPU MODE: device run'
    log ' Running device work. This is deliberate, recorded, and scoped to'
    log ' device access -- no driver or CANN changes, no other tenant touched.'
    log '================================================================'
    # Root does not authorize a device: a device id must still be named, and
    # it is never discovered or substituted. device-select.sh refuses without
    # one, reports occupancy rather than enforcing it, and yields BLOCKED_ENV
    # (75) if exclusive access was asked for and the card is held -- so a busy
    # card can never be mistaken for a passing run.
    # An unset device id is refused explicitly rather than through ${VAR:?},
    # which exits 1 and would be indistinguishable from the selection itself
    # failing for another reason.
    if [ -z "${TF_DEVICE_ID:-}" ]; then
        log 'BLOCKED: --npu requires an explicit device. Pass --device N or TF_DEVICE_ID=N.'
        log '  A device is never discovered or substituted: a run against a device'
        log '  the operator did not name is not evidence about the device they meant.'
        trap - EXIT
        flock -u 9
        exit 2
    fi
    sel="$HERE/device-select.sh"
    if [ ! -x "$sel" ]; then
        log 'device-select.sh missing; refusing to run device work unselected'
        trap - EXIT
        flock -u 9
        exit 2
    fi
    sel_args=(--device "$TF_DEVICE_ID")
    [ "${TF_REQUIRE_EXCLUSIVE:-0}" = 1 ] && sel_args+=(--exclusive)
    if "$sel" "${sel_args[@]}" > "$RUNDIR/device-select.txt" 2>&1; then
        log 'device selection ok; see device-select.txt'
    else
        sel_rc=$?
        log "device selection refused (exit $sel_rc); see device-select.txt"
        if [ "$sel_rc" -eq 75 ]; then
            log 'BLOCKED_ENV: the device run did not happen.'
        fi
        cat "$RUNDIR/device-select.txt" >&2 || true
        trap - EXIT
        flock -u 9
        exit "$sel_rc"
    fi

    # The selection must be RESOLVED HERE, in the caller, and then carried into
    # the payload explicitly.
    #
    # device-select.sh exports ASCEND_RT_VISIBLE_DEVICES into its OWN shell, and
    # that shell exits. Running it as a child therefore selected a device for a
    # process that then ended: the payload started with the variable UNSET and
    # inherited whatever the container was fixed at. A request for device 1 on a
    # container bound to device 0 silently ran on device 0 -- the substitution
    # this whole module exists to refuse.
    SELECTED_DEVICE="$(sed -n 's/^device_id=//p' "$RUNDIR/device-select.txt" | head -1)"
    if [ "${SELECTED_DEVICE:-}" != "$TF_DEVICE_ID" ]; then
        log "selection did not report the requested device:"
        log "  requested: $TF_DEVICE_ID"
        log "  selection reported: ${SELECTED_DEVICE:-<nothing>}"
        trap - EXIT
        flock -u 9
        exit 2
    fi
    log "device selection resolved in this shell: $SELECTED_DEVICE"

    # Export it so the payload inherits it, and record it as part of the run.
    export ASCEND_RT_VISIBLE_DEVICES="$SELECTED_DEVICE"
    printf 'selected_device: %s\n' "$SELECTED_DEVICE" >> "$RUNDIR/attribution.txt"

    # A docker exec is a separate process tree, so the export above reaches it
    # only if it is named on the command line. The flag is injected into the
    # caller's argv rather than left to the caller to remember.
    # A docker exec is a separate process tree: the export above reaches it only
    # if the variable is named on the docker command line. Rather than ask the
    # caller to remember `-e`, the flag is injected here, into the argv that will
    # actually run.
    if [ "$1" = docker ]; then
        shift
        # The word "docker" is restored below; dropping it would leave "exec" as
        # the command and the whole payload would fail to launch.
        #
        # A `-e ASCEND_RT_VISIBLE_DEVICES=...` already present in the caller's
        # argv is REMOVED first. Docker lets a later -e win, so merely appending
        # ours would leave the caller able to override the selection by adding
        # their own later. Removing every copy and inserting exactly one makes
        # the value unambiguous rather than order-dependent.
        new_args=(docker)
        injected=0
        dropped=0
        skip_next=0
        while [ "$#" -gt 0 ]; do
            if [ "$skip_next" = 1 ]; then
                skip_next=0; shift; continue
            fi
            if [ "$1" = "-e" ] || [ "$1" = "--env" ]; then
                case "${2:-}" in
                    ASCEND_RT_VISIBLE_DEVICES=*)
                        # A caller-supplied filter is REFUSED, not silently
                        # replaced. Docker applies the LAST -e for a variable, so
                        # appending ours after theirs would let them win; dropping
                        # theirs would mean running a device the caller explicitly
                        # named and being told nothing about it. Both are worse
                        # than stopping, so the conflict is named and the run does
                        # not start.
                        supplied="${2#ASCEND_RT_VISIBLE_DEVICES=}"
                        if [ "$supplied" != "$SELECTED_DEVICE" ]; then
                            log "refusing: the caller supplies a device filter that contradicts the selection"
                            log "  selection (TF_DEVICE_ID/--device): $SELECTED_DEVICE"
                            log "  caller $1 $2"
                            log "  These name different devices. One of them is wrong, and"
                            log "  guessing which would produce evidence about a device"
                            log "  nobody agreed on."
                            trap - EXIT
                            flock -u 9
                            exit 2
                        fi
                        log "caller supplies a matching filter ($1 $2); keeping a single copy"
                        dropped=$((dropped + 1))
                        shift 2
                        continue
                        ;;
                esac
                new_args+=("$1" "${2:-}")
                skip_next=1
                shift
                continue
            fi
            new_args+=("$1")
            if [ "$1" = exec ] && [ "$injected" = 0 ]; then
                new_args+=(-e "ASCEND_RT_VISIBLE_DEVICES=$SELECTED_DEVICE")
                injected=1
            fi
            shift
        done
        if [ "$injected" = 0 ]; then
            log 'refusing: could not find `exec` in the docker command to carry the selection'
            trap - EXIT
            flock -u 9
            exit 2
        fi
        [ "$dropped" -gt 0 ] && log "deduplicated $dropped caller-supplied ASCEND_RT_VISIBLE_DEVICES setting(s)"
        set -- "${new_args[@]}"
        # Find the container name: the first non-option word after `exec`.
        SEL_CONTAINER=""
        skip_next=0
        saw_exec=0
        for a in "${new_args[@]}"; do
            if [ "$skip_next" = 1 ]; then skip_next=0; continue; fi
            if [ "$a" = "-e" ] || [ "$a" = "-u" ] || [ "$a" = "--env" ]; then
                skip_next=1
                continue
            fi
            if [ "$a" = "exec" ]; then saw_exec=1; continue; fi
            case "$a" in
                -*|*=*|docker) continue ;;
            esac
            if [ "$saw_exec" = 1 ]; then SEL_CONTAINER="$a"; break; fi
        done
        if [ -n "${SEL_CONTAINER:-}" ]; then SEL_IN_CONTAINER=1; fi

        # The assertion belongs INSIDE the container, wrapping the real workload.
        # Checking the wrapper's own environment proves nothing about what the
        # workload sees: the wrapper never enters the container. This prefix runs
        # in the container, reads the environment the workload actually inherits,
        # and aborts before the workload starts if it disagrees.
        assertion="v=\${ASCEND_RT_VISIBLE_DEVICES:-<UNSET>}; wanted=$SELECTED_DEVICE; n=\$(printf %s \"\$v\" | tr \",\" \"\\n\" | grep -c .); if [ \"\$v\" != \"\$wanted\" ] || [ \"\$n\" -ne 1 ]; then echo \"DEVICE-ASSERT-FAIL: payload sees ASCEND_RT_VISIBLE_DEVICES=\$v (\$n device(s)), expected exactly 1 matching \$wanted\" >&2; exit 97; fi; echo \"DEVICE-ASSERT-OK: payload sees exactly one device (\$v)\"; exec \"\$@\""
        # Rebuild argv: keep everything up to and including the container name,
        # then insert sh -c <assertion> before the original command.
        final=()
        seen_container=0
        skip_next=0
        for a in "${new_args[@]}"; do
            if [ "$skip_next" = 1 ]; then skip_next=0; final+=("$a"); continue; fi
            if [ "$a" = "-e" ] || [ "$a" = "-u" ] || [ "$a" = "--env" ]; then
                skip_next=1; final+=("$a"); continue
            fi
            final+=("$a")
            if [ "$seen_container" = 0 ] && [ "$a" = "$SEL_CONTAINER" ]; then
                seen_container=1
                final+=(sh -c "$assertion" --)
            fi
        done
        set -- "${final[@]}"
        log "payload will run with ASCEND_RT_VISIBLE_DEVICES=$SELECTED_DEVICE (asserted inside the container)"
    fi
fi

# SCOPE: this check needs to know which container the payload targets, so it runs
# only when the payload is a `docker exec`. A payload that runs directly on the
# host is not confined by a container's device binds, so container reachability
# is not the relevant question for it -- the host presence check already done
# applies there. Stated explicitly because "not checked" must not be mistaken for
# "checked and passed".
# A device id that exists on the HOST but is not bound into the container would
# let the payload run against whatever the container does have -- the exact
# substitution this module refuses. Presence is therefore checked where the
# payload will actually run, not only on the host.
if [ "$MODE" = npu ] && [ "${SEL_IN_CONTAINER:-0}" = 1 ]; then
    if ! docker exec "$SEL_CONTAINER" test -e "/dev/davinci$SELECTED_DEVICE" 2>/dev/null; then
        log "BLOCKED: device $SELECTED_DEVICE is not bound into $SEL_CONTAINER."
        log "  the container exposes: $(docker exec "$SEL_CONTAINER" sh -c 'ls /dev/davinci* 2>/dev/null' 2>/dev/null | tr '\n' ' ')"
        log '  Refusing to run: the payload would execute against a different'
        log '  device than the one requested, which is not evidence about the'
        log '  device that was named.'
        trap - EXIT
        flock -u 9
        exit 2
    fi
    log "device $SELECTED_DEVICE is bound into $SEL_CONTAINER"
fi

# In-payload assertion. Whatever the payload is, it must see exactly the device
# that was requested -- checked from inside this script, where the environment
# the payload inherits is the one visible here.
if [ "$MODE" = npu ]; then
    if [ "${ASCEND_RT_VISIBLE_DEVICES:-}" != "${SELECTED_DEVICE:-}" ]; then
        log 'refusing to run: the payload would not see the selected device.'
        log "  selected: ${SELECTED_DEVICE:-<none>}"
        log "  ASCEND_RT_VISIBLE_DEVICES here: ${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
        trap - EXIT
        flock -u 9
        exit 2
    fi
    visible_count="$(printf '%s' "$ASCEND_RT_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)"
    if [ "$visible_count" -ne 1 ]; then
        log "refusing to run: expected exactly one visible device, got $visible_count"
        trap - EXIT
        flock -u 9
        exit 2
    fi
    log "payload assertion: exactly one device visible ($ASCEND_RT_VISIBLE_DEVICES)"
fi

npu-smi info > "$RUNDIR/npu-smi.before.txt" 2>&1
log "npu-smi before -> exit $?"

"$@"
rc=$?

npu-smi info > "$RUNDIR/npu-smi.after.txt" 2>&1
log "npu-smi after  -> exit $?"
printf 'exit: %s\nfinished: %s\n' "$rc" "$(date -Is)" >> "$RUNDIR/attribution.txt"

# The run directory is kept: the reservation is the evidence.
trap - EXIT
flock -u 9
log "released $LOCK; command exit $rc"
log "evidence: $RUNDIR"
# The command's own status is propagated, so this wrapper cannot turn a failure
# into a pass or a pass into a failure.
exit "$rc"
