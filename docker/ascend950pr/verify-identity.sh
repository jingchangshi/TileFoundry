#!/usr/bin/env bash
# Verify the container identity by ASKING the container, not by assuming it.
#
# entry.sh is told to create a user; whether it actually did is a separate fact.
# This walks the same three questions the reference does -- id -u, id -g, and
# printenv HOME -- through `docker exec -u`, and additionally checks that each
# supplementary GID in the configuration is one the account is really in.
#
# Adopted from AgentGates tla-npu/docker/create-container.sh:392-408, which
# verifies identity through docker exec rather than trusting the entrypoint.
#
# Config comes from env.sh only (see env.example.sh). Flags override.
#
# Exit codes
#   0  identity verified
#   2  configuration missing or unusable
#   3  identity does not match the configuration
#   4  container not reachable
set -uo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROG="verify-identity.sh"

log()  { printf '%s: %s\n' "$PROG" "$*" >&2; }
die()  { log "$*"; exit "${2:-2}"; }

# --- strict configuration file --------------------------------------------
# One NAME=value file, parsed as data, never executed. Unknown keys and
# malformed lines are errors: a typo that is silently ignored is a value the
# operator believes in and the script does not have.
ENV_FILE="${TF_ENV_FILE:-$HERE/env.sh}"

declare -A CONFIG=()
if [ -r "$ENV_FILE" ]; then
    lineno=0
    while IFS= read -r line || [ -n "$line" ]; do
        lineno=$((lineno + 1))
        case "$line" in
            ''|'#'*) continue ;;
        esac
        case "$line" in
            *=*) ;;
            *) die "$ENV_FILE:$lineno: not NAME=value: $line" ;;
        esac
        key="${line%%=*}"; value="${line#*=}"
        case "$key" in
            *[!A-Za-z0-9_]*) die "$ENV_FILE:$lineno: bad key name: $key" ;;
        esac
        if [ -n "${CONFIG[$key]+set}" ]; then
            die "$ENV_FILE:$lineno: duplicate key: $key"
        fi
        case "$key" in
            CONTAINER_USER|CONTAINER_UID|CONTAINER_GID|CONTAINER_HOME|SUPPLEMENTARY_GIDS|CONTAINER_NAME|IMAGE|HOST_REPO|HOST_CANN|HOST_VENV|NPU_PYTHON|NPU_ENV_DIR|MOUNT_NPU_ENV|TF_REQUIRE_EXCLUSIVE) ;;
            *) die "$ENV_FILE:$lineno: unknown key: $key" ;;
        esac
        CONFIG[$key]="$value"
    done < "$ENV_FILE"
elif [ -n "${TF_ENV_FILE:-}" ]; then
    # An explicitly named file that does not exist is an error; the default
    # path being absent only means "nothing configured".
    die "TF_ENV_FILE does not exist: $ENV_FILE"
fi

# Warn about a world-readable configuration: it holds no secret today, but the
# convention the reference sets is 600 and a loose file is worth noticing.
if [ -r "$ENV_FILE" ] && [ "$(stat -c '%a' "$ENV_FILE" 2>/dev/null)" != "600" ]; then
    log "note: $ENV_FILE is mode $(stat -c '%a' "$ENV_FILE" 2>/dev/null), expected 600"
fi

# The configuration file is the sole source of these values. An export of the
# same name is a contradiction, not an override: honouring it would mean the
# identity verified need not be the one written down, which defeats the purpose.
conflict=()
check_conflict() {
    local key="$1" exported="$2" recorded="${3:-}"
    if [ -n "$exported" ] && [ "$exported" != "$recorded" ]; then
        conflict+=("$key: exported as '$exported', but the file says '${recorded:-<absent>}'")
    fi
}
check_conflict CONTAINER_USER "${CONTAINER_USER:-}" "${CONFIG[CONTAINER_USER]:-}"
check_conflict CONTAINER_UID "${CONTAINER_UID:-}" "${CONFIG[CONTAINER_UID]:-}"
check_conflict CONTAINER_GID "${CONTAINER_GID:-}" "${CONFIG[CONTAINER_GID]:-}"
check_conflict CONTAINER_HOME "${CONTAINER_HOME:-}" "${CONFIG[CONTAINER_HOME]:-}"
check_conflict CONTAINER_NAME "${CONTAINER_NAME:-}" "${CONFIG[CONTAINER_NAME]:-}"
check_conflict SUPPLEMENTARY_GIDS "${SUPPLEMENTARY_GIDS:-}" "${CONFIG[SUPPLEMENTARY_GIDS]:-}"
if [ "${#conflict[@]}" -gt 0 ]; then
    log 'refusing: the environment contradicts the configuration file'
    for item in "${conflict[@]}"; do log "  $item"; done
    log '  The file is authoritative. Unset the export, or change the file.'
    exit 2
fi

CONTAINER_NAME="${CONFIG[CONTAINER_NAME]:-}"
CONTAINER_USER="${CONFIG[CONTAINER_USER]:-}"
EXPECT_UID="${CONFIG[CONTAINER_UID]:-}"
EXPECT_GID="${CONFIG[CONTAINER_GID]:-}"
EXPECT_HOME="${CONFIG[CONTAINER_HOME]:-}"
SUPP_GIDS="${CONFIG[SUPPLEMENTARY_GIDS]:-}"

# --- required values -------------------------------------------------------
# A missing identity value is refused rather than defaulted. Defaulting to the
# invoking user would "verify" whoever happens to run this.
missing=()
[ -n "$CONTAINER_NAME" ] || missing+=(CONTAINER_NAME)
[ -n "$CONTAINER_USER" ] || missing+=(CONTAINER_USER)
[ -n "$EXPECT_UID" ]    || missing+=(CONTAINER_UID)
[ -n "$EXPECT_GID" ]    || missing+=(CONTAINER_GID)
[ -n "$EXPECT_HOME" ]   || missing+=(CONTAINER_HOME)
if [ "${#missing[@]}" -gt 0 ]; then
    die "required configuration missing: ${missing[*]} (see env.example.sh)"
fi

command -v docker >/dev/null 2>&1 || die "docker not found"
if ! docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    die "container not reachable: $CONTAINER_NAME" 4
fi

# --- the container is asked, not assumed ----------------------------------
#
# HOME is READ, never injected. The previous version passed -e HOME=$EXPECT_HOME
# and then asked the container what HOME was, so a wrong expected HOME passed by
# construction: it compared the value it had just supplied against itself.
# Each answer is captured with its own exit status. A docker exec that fails
# returns empty, and empty must not be compared to empty and called a match.
x() { docker exec -u "$CONTAINER_USER" "$CONTAINER_NAME" "$@" 2>/dev/null; }

actual_uid="$(x id -u)"; uid_rc=$?
actual_gid="$(x id -g)"; gid_rc=$?
# docker exec -u supplies a HOME of its own, so asking printenv HOME directly
# would report the value docker chose, not the account's. The passwd entry is the
# container's own record of where the account lives, and nothing we pass to the
# exec can change it.
actual_home="$(x getent passwd "$CONTAINER_USER" | cut -d: -f6)"; home_rc=$?

fail=0
if [ "$uid_rc" -ne 0 ]; then
    log "could not read the uid inside the container (docker exec -> exit $uid_rc)"
    fail=1
elif [ "$actual_uid" != "$EXPECT_UID" ]; then
    log "uid mismatch: container says $actual_uid, configuration says $EXPECT_UID"
    fail=1
else
    log "uid verified: $actual_uid"
fi

if [ "$gid_rc" -ne 0 ]; then
    log "could not read the gid inside the container (docker exec -> exit $gid_rc)"
    fail=1
elif [ "$actual_gid" != "$EXPECT_GID" ]; then
    log "gid mismatch: container says $actual_gid, configuration says $EXPECT_GID"
    fail=1
else
    log "gid verified: $actual_gid"
fi

if [ "$home_rc" -ne 0 ]; then
    log "could not read HOME inside the container (docker exec -> exit $home_rc)"
    fail=1
elif [ "$actual_home" != "$EXPECT_HOME" ]; then
    log "HOME mismatch: container says '$actual_home', configuration says '$EXPECT_HOME'"
    fail=1
else
    log "HOME verified: $actual_home"
fi

# --- supplementary groups ------------------------------------------------
# Membership is checked, not assumed. The primary GID is not a supplementary
# one, so it is skipped rather than reported as missing.
if [ -n "$SUPP_GIDS" ]; then
    groups_out="$(x id -G)"; groups_rc=$?
    if [ "$groups_rc" -ne 0 ]; then
        log "could not read supplementary groups (docker exec -> exit $groups_rc)"
        fail=1
    else
        for gid in $SUPP_GIDS; do
            [ -n "$gid" ] || continue
            if [ "$gid" = "$EXPECT_GID" ]; then
                log "supplementary gid $gid is the primary gid; not checked separately"
                continue
            fi
            if printf '%s\n' "$groups_out" | tr ' ' '\n' | grep -qx "$gid"; then
                log "supplementary gid verified: $gid"
            else
                log "supplementary gid $gid missing: account is in $(printf '%s' "$groups_out")"
                fail=1
            fi
        done
    fi
fi

if [ "$fail" -ne 0 ]; then
    log "identity NOT verified for $CONTAINER_USER in $CONTAINER_NAME"
    exit 3
fi

log "identity verified for $CONTAINER_USER in $CONTAINER_NAME: uid=$actual_uid gid=$actual_gid home=$actual_home"
docker exec -u "$CONTAINER_USER" -e "HOME=$EXPECT_HOME" "$CONTAINER_NAME" id 2>/dev/null | sed 's/^/  /' >&2
exit 0