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
LOCK="${TF_DEVICE_LOCK:-/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/artifacts/run02/p0/container/.device.lock}"
# Evidence goes next to the lock by default: the lock already lives on the shared
# mount, so its directory is the one path known to be identical on both sides.
OUTDIR="${TF_DEVICE_OUTDIR:-$(dirname -- "$LOCK")}"
LOCK_TIMEOUT="${TF_DEVICE_LOCK_TIMEOUT:-900}"

log() { printf '%s: %s\n' "$PROG" "$*" >&2; }
die() { log "$*"; exit 2; }

NAME=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --name) NAME="${2:-}"; [ -n "$NAME" ] || die '--name needs a label'; shift 2 ;;
        --) shift; break ;;
        -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1 (try --help)" ;;
    esac
done

[ "$#" -gt 0 ] || die 'no command given; put it after --'
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

# Attribution: who holds the lock, and what the device looked like on arrival.
{
    printf 'run: %s\n' "$NAME"
    printf 'started: %s\n' "$(date -Is)"
    printf 'host: %s\n' "$(hostname)"
    printf 'user: %s\n' "$(id -un)"
    printf 'lock: %s\n' "$LOCK"
    printf 'command:'
    printf ' %q' "$@"
    printf '\n'
} > "$OUTDIR/$NAME.attribution.txt"

npu-smi info > "$OUTDIR/$NAME.npu-smi.before.txt" 2>&1
log "npu-smi before -> exit $?"

"$@"
rc=$?

npu-smi info > "$OUTDIR/$NAME.npu-smi.after.txt" 2>&1
log "npu-smi after  -> exit $?"
printf 'exit: %s\nfinished: %s\n' "$rc" "$(date -Is)" >> "$OUTDIR/$NAME.attribution.txt"

flock -u 9
log "released $LOCK; command exit $rc"
# The command's own status is propagated, so this wrapper cannot turn a failure
# into a pass or a pass into a failure.
exit "$rc"