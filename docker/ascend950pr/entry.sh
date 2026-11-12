#!/usr/bin/env bash
# Container entrypoint for the AscendNPU-IR development container.
#
# Runs as root. Its only job is to make the configured identity exist inside the
# container and to leave the container in a known state, then hand over to the
# configured command with `exec`:
#
#   1. create or verify the group named after CONTAINER_USER with CONTAINER_GID;
#   2. create or verify the account with the exact CONTAINER_UID/GID/HOME;
#   3. add the requested supplementary GIDs;
#   4. verify all of it with `id` and the passwd database;
#   5. write a readiness marker under /run;
#   6. `exec` the configured command (default: keep the container alive).
#
# It never installs Conda or CANN, never runs CMake, never builds AscendNPU-IR
# and never activates a Conda environment. The Adapter does that per Gate inside
# `docker exec`, because ENTRYPOINT exports are NOT inherited by a later
# `docker exec` (that starts a separate process from Docker's configured
# environment). The toolchain file named by ASCENDNPU_ENV_FILE is read by the
# Adapter on the Host; see ../README.md for manual use.
#
# Identity rules, deliberately strict:
#   * an existing account or group is reused only when its numeric IDs match;
#   * a conflicting name is an error, never silently renumbered;
#   * HOME is never created, chowned or populated here, because it is usually a
#     bind mount of the host home directory.
set -euo pipefail

log() { printf 'entry.sh: %s\n' "$*" >&2; }
die() { log "$*"; exit 1; }

READY_MARKER=/run/ascendnpu-ir-entry-ready

# `--check-ready` lets create-container.sh poll readiness without starting a
# second copy of the initialization.
if [ "${1:-}" = "--check-ready" ]; then
    [ -f "$READY_MARKER" ] || { log 'container is not initialized yet'; exit 1; }
    printf 'ready\n'
    exit 0
fi

# A fresh start always invalidates a previous readiness marker.
rm -f "$READY_MARKER"

CONTAINER_USER="${CONTAINER_USER:-}"
CONTAINER_UID="${CONTAINER_UID:-}"
CONTAINER_GID="${CONTAINER_GID:-}"
CONTAINER_HOME="${CONTAINER_HOME:-}"
SUPPLEMENTARY_GIDS="${SUPPLEMENTARY_GIDS:-}"

[ -n "$CONTAINER_USER" ] || die 'CONTAINER_USER is required'
[ -n "$CONTAINER_UID" ] || die 'CONTAINER_UID is required'
[ -n "$CONTAINER_GID" ] || die 'CONTAINER_GID is required'
[ -n "$CONTAINER_HOME" ] || die 'CONTAINER_HOME is required'
case "$CONTAINER_UID" in ''|*[!0-9]*) die "CONTAINER_UID must be numeric: $CONTAINER_UID" ;; esac
case "$CONTAINER_GID" in ''|*[!0-9]*) die "CONTAINER_GID must be numeric: $CONTAINER_GID" ;; esac
[ "$CONTAINER_UID" -gt 0 ] || die 'CONTAINER_UID must not be 0'
[ "$CONTAINER_HOME" != / ] || die 'CONTAINER_HOME must not be /'

# --- primary group ---------------------------------------------------------
# Reuse an existing group with this GID; otherwise create the named group.
existing_group="$(getent group "$CONTAINER_GID" || true)"
if [ -n "$existing_group" ]; then
    existing_name="${existing_group%%:*}"
    if [ "$existing_name" != "$CONTAINER_USER" ]; then
        # A different name already owns the GID. That is fine for the account's
        # primary group, but the requested name must not be repointed.
        if getent group "$CONTAINER_USER" >/dev/null 2>&1; then
            log "group $CONTAINER_USER exists with a different GID; using GID $CONTAINER_GID ($existing_name)"
        else
            log "GID $CONTAINER_GID is already used by group $existing_name; reusing it as the primary group"
        fi
    fi
else
    if getent group "$CONTAINER_USER" >/dev/null 2>&1; then
        die "group $CONTAINER_USER already exists with a different GID than $CONTAINER_GID"
    fi
    groupadd --gid "$CONTAINER_GID" "$CONTAINER_USER"
fi

# --- account ---------------------------------------------------------------
existing_passwd="$(getent passwd "$CONTAINER_USER" || true)"
if [ -n "$existing_passwd" ]; then
    existing_uid="$(cut -d: -f3 <<<"$existing_passwd")"
    existing_gid="$(cut -d: -f4 <<<"$existing_passwd")"
    existing_home="$(cut -d: -f6 <<<"$existing_passwd")"
    [ "$existing_uid" = "$CONTAINER_UID" ] \
        || die "user $CONTAINER_USER exists with UID $existing_uid, not $CONTAINER_UID; refusing to renumber"
    [ "$existing_gid" = "$CONTAINER_GID" ] \
        || die "user $CONTAINER_USER exists with GID $existing_gid, not $CONTAINER_GID; refusing to renumber"
    [ "$existing_home" = "$CONTAINER_HOME" ] \
        || die "user $CONTAINER_USER exists with HOME $existing_home, not $CONTAINER_HOME"
else
    if getent passwd "$CONTAINER_UID" >/dev/null 2>&1; then
        die "UID $CONTAINER_UID is already used by $(getent passwd "$CONTAINER_UID" | cut -d: -f1)"
    fi
    # --no-create-home: HOME is normally a bind mount; never populate it here.
    useradd --uid "$CONTAINER_UID" --gid "$CONTAINER_GID" \
            --home-dir "$CONTAINER_HOME" --no-create-home \
            --shell /bin/bash "$CONTAINER_USER"
fi

# --- supplementary groups --------------------------------------------------
# A requested GID may already be a named group (for example the host's
# HwHiAiUser). Otherwise create a deterministic name for it. Memberships are
# added, never replaced.
for gid in $SUPPLEMENTARY_GIDS; do
    case "$gid" in ''|*[!0-9]*) die "supplementary GID must be numeric: $gid" ;; esac
    [ "$gid" -gt 0 ] || die 'supplementary GID must not be 0'
    [ "$gid" != "$CONTAINER_GID" ] || continue
    if ! getent group "$gid" >/dev/null 2>&1; then
        groupadd --gid "$gid" "ascend-gid-$gid"
    fi
    group_name="$(getent group "$gid" | cut -d: -f1)"
    usermod --append --groups "$group_name" "$CONTAINER_USER"
done

# --- verification ----------------------------------------------------------
# Prove the identity before the readiness marker exists. create-container.sh
# additionally re-runs `id -u`/`id -g` through `docker exec` itself, so a bug
# here cannot fake a healthy container.
verify_failures=0
actual_uid="$(id -u "$CONTAINER_USER")"
actual_gid="$(id -g "$CONTAINER_USER")"
[ "$actual_uid" = "$CONTAINER_UID" ] || { log "verification: UID $actual_uid != $CONTAINER_UID"; verify_failures=1; }
[ "$actual_gid" = "$CONTAINER_GID" ] || { log "verification: GID $actual_gid != $CONTAINER_GID"; verify_failures=1; }
passwd_home="$(getent passwd "$CONTAINER_USER" | cut -d: -f6)"
[ "$passwd_home" = "$CONTAINER_HOME" ] || { log "verification: HOME $passwd_home != $CONTAINER_HOME"; verify_failures=1; }
for gid in $SUPPLEMENTARY_GIDS; do
    [ "$gid" = "$CONTAINER_GID" ] && continue
    if ! id -G "$CONTAINER_USER" | tr ' ' '\n' | grep -qx "$gid"; then
        log "verification: $CONTAINER_USER is not a member of GID $gid"
        verify_failures=1
    fi
done
[ "$verify_failures" -eq 0 ] || die 'user initialization verification failed'

install -m 0644 /dev/null "$READY_MARKER"
log "identity ready: $CONTAINER_USER uid=$CONTAINER_UID gid=$CONTAINER_GID home=$CONTAINER_HOME"

# --- hand over -------------------------------------------------------------
# `exec` so the configured command becomes PID 1's successor and receives
# signals directly. `--init` (recommended in create-container.sh) reaps orphans.
if [ "$#" -eq 0 ]; then
    set -- tail -f /dev/null
fi
exec "$@"
