#!/usr/bin/env bash
# Report device occupancy. Report it -- do not enforce it.
#
# Adopted from AgentGates tla-npu (README "Device selection"): a device id is
# required and never discovered; occupancy is recorded rather than refused,
# because this is a shared machine and a busy card is not a fact about the code
# under test. Optional exclusivity, when asked for, yields BLOCKED_ENV -- the run
# is reported as not having happened, never as having passed.
#
# Exit codes
#   0          device is usable (occupied or not, unless exclusive was asked)
#   2          no device id given, or configuration is unusable
#   3          the named device is not present on this host
#   75         BLOCKED_ENV: exclusive access was requested and the device is held
set -uo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROG="device-select.sh"

log() { printf '%s: %s\n' "$PROG" "$*" >&2; }
die() { log "$*"; exit "${2:-2}"; }

# --- an explicit device id, never discovered ------------------------------
# There is no default and no search. Picking a card because it happens to be
# free would make the evidence describe a different device than the one the
# operator named, which is exactly the substitution this refuses to do.
DEVICE_ID="${TF_DEVICE_ID:-}"
REQUIRE_EXCLUSIVE="${TF_REQUIRE_EXCLUSIVE:-0}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --device) DEVICE_ID="${2:-}"; shift 2 ;;
        --exclusive) REQUIRE_EXCLUSIVE=1; shift ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

if [ -z "$DEVICE_ID" ]; then
    die "no device id given. Pass --device N or TF_DEVICE_ID=N."
    # Not a hint to guess: refusing is the point. A run against a device the
    # operator did not name is not evidence about the device they meant.
fi
case "$DEVICE_ID" in
    *[!0-9]*) die "device id must be a number, got: $DEVICE_ID" ;;
esac

command -v npu-smi >/dev/null 2>&1 || die "npu-smi not found; cannot report occupancy"

# --- what the host says ----------------------------------------------------
# npu-smi info is read-only. Nothing here resets, restarts or signals anything.
snapshot="$(npu-smi info 2>&1)"; rc=$?
if [ "$rc" -ne 0 ]; then
    die "npu-smi info failed with status $rc; occupancy is UNKNOWN, not empty"
fi

# Rows are drawn as a table, so the id column arrives as "| 0      |" and $1 is
# the pipe, not the id. Separators are stripped before the first field is read;
# matching $1 directly would find no device at all and report every card absent.
# Pipes separate every column, so they are replaced by spaces rather than merely
# trimmed from the edges: with the pipes left in place $2 is "|" and no row can
# be parsed, which is how a parsing bug reads as an idle device.
trim() { sed 's/|/ /g'; }

# A device counts as present when its id is the first field of some row. Anchoring
# on the field avoids matching a process id or a memory figure.
if ! printf '%s\n' "$snapshot" | trim | awk -v id="$DEVICE_ID" 'NF && $1 == id { found = 1 } END { exit !found }'; then
    die "device $DEVICE_ID is not present on this host" 3
fi
log "device $DEVICE_ID is present"

# --- occupancy -------------------------------------------------------------
# The Ascend tooling exposes no portable per-card "busy" flag, so occupancy is
# read from the process table: a row whose NPU ID is this card means something is
# running on it.
holders="$(printf '%s\n' "$snapshot" | trim | awk -v id="$DEVICE_ID" 'NF >= 3 && $1 == id && $2 ~ /^[0-9]+$/ { print }')"
if [ -n "$holders" ]; then
    log "device $DEVICE_ID is OCCUPIED by:"
    printf '%s\n' "$holders" | sed 's/^/  /' >&2
    OCCUPIED=yes
else
    log "device $DEVICE_ID reports no active process"
    OCCUPIED=no
fi

# Report, do not enforce. Another tenant on the card is recorded and the run
# proceeds; that is the policy for a shared server, and it is why no timing
# number from this host may be presented as reproducible.
if [ "$OCCUPIED" = yes ] && [ "$REQUIRE_EXCLUSIVE" = 1 ]; then
    log "BLOCKED_ENV: exclusive access was requested and device $DEVICE_ID is held."
    log "  The run did not happen. This is not a pass and not a failure: there"
    log "  is no result to report, and reporting one would be a fabrication."
    exit 75
fi

if [ "$OCCUPIED" = yes ]; then
    log "proceeding on a shared device (occupancy is reported, not enforced)"
    log "  any timing from this run is PROVISIONAL and must be labelled so;"
    log "  numerics are unaffected."
fi

# --- the single visible device --------------------------------------------
# Exported for the runtime, and asserted to be exactly one before anything
# imports it. A filter that silently expanded to every card would make a run
# that touched the wrong board indistinguishable from a correct one.
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE_ID"
visible="$(printf '%s' "$ASCEND_RT_VISIBLE_DEVICES" | tr ',' '\n' | grep -c . )"
if [ "${visible:-0}" -ne 1 ]; then
    die "ASCEND_RT_VISIBLE_DEVICES must name exactly one device, got: $ASCEND_RT_VISIBLE_DEVICES"
fi
log "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES (exactly one device)"

printf 'device_id=%s\noccupied=%s\nexclusive_requested=%s\nvisible_devices=%s\n' \
    "$DEVICE_ID" "$OCCUPIED" "$REQUIRE_EXCLUSIVE" "$ASCEND_RT_VISIBLE_DEVICES"
exit 0