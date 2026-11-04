#!/usr/bin/env bash
# Create and start the TileFoundry Ascend950PR development container.
#
#     ./create-container.sh --config ./config.env
#     ./create-container.sh --config ./config.env --dry-run
#
# Adapted from the AgentGates tilelang-hivm reference
# (agent_gates/adapters/tilelang-hivm/docker/create-container.sh); see also
# adapters/ascendnpu-ir/docker. The conservative behaviour below is inherited
# unchanged -- it is reviewed, and it is the part that protects other users
# on a shared host.
#
# The script reads one trusted Bash configuration file (see config.example.env),
# turns it into a single `docker run` invocation, starts the container, waits for
# the entrypoint to finish creating the configured identity, and then re-verifies
# that identity with its own `docker exec`.
#
# It is intentionally conservative:
#
#   * a container with the configured name that already exists is a hard error —
#     it is never removed, stopped or renamed;
#   * no `docker rm -f`, no volume pruning, no recursive `chown` of /home or /data,
#     no host Driver change;
#   * a missing bind-mount source is an error (required) or an explicit reported
#     skip (optional), never a silently created directory;
#   * the container is only reported as created after the identity has been
#     verified twice, not merely after `docker run -d` returned.
#
# This script is deployment only. The toolchain (CANN) is not configured here
# either, and this script never reads CANN's configuration: CANN is bind-mounted
# and initialised inside the container by whoever runs a build. Mount guidance
# is in config.example.env and README.md.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROG='create-container.sh'

log()  { printf '%s: %s\n' "$PROG" "$*" >&2; }
die()  { log "$*"; exit 2; }

usage() {
    cat <<'EOF'
Usage: create-container.sh [--config PATH] [--dry-run] [--help]

  --config PATH   Configuration file (default: ./config.env next to this script)
  --dry-run       Validate configuration and host paths, print the final docker
                  run command, and create nothing
  --help          Show this help

See config.example.env for every option and its default.
EOF
}

CONFIG="$HERE/config.env"
DRY_RUN=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --config)  [ "$#" -ge 2 ] || die '--config needs a path'; CONFIG="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1 (try --help)" ;;
    esac
done

[ -r "$CONFIG" ] || die "configuration file is not readable: $CONFIG"
# shellcheck disable=SC1090  # config path is a run-time argument by design
. "$CONFIG"

# ---------------------------------------------------------------------------
# Defaults and validation
# ---------------------------------------------------------------------------
IMAGE="${IMAGE:-tilefoundry-ascend950pr-dev:24.04}"
CONTAINER_NAME="${CONTAINER_NAME:-}"
CONTAINER_USER="${CONTAINER_USER:-}"
CONTAINER_UID="${CONTAINER_UID:-}"
CONTAINER_GID="${CONTAINER_GID:-}"
CONTAINER_HOSTNAME="${CONTAINER_HOSTNAME:-}"
SUPPLEMENTARY_GIDS="${SUPPLEMENTARY_GIDS:-}"
CONTAINER_HOME="${CONTAINER_HOME:-}"
NETWORK="${NETWORK:-host}"
# Defaults to FALSE, deliberately departing from the reference (which defaulted
# true). Under --privileged the --device list is advisory: the container can
# open every host device regardless of what is listed. Unprivileged, the list
# is a hard limit, so NPU_DEVICES actually confines the container. On a shared
# host that difference is the whole point, so the safe value is the default and
# privilege has to be asked for explicitly.
PRIVILEGED="${PRIVILEGED:-false}"
CGROUPNS="${CGROUPNS:-host}"
SECURITY_OPTS="${SECURITY_OPTS:-label=disable}"
SHM_SIZE="${SHM_SIZE:-512g}"
MEMLOCK="${MEMLOCK:-150000000000:150000000000}"
# Omitted means no filtering, not "every card on this host". Defaulting to a
# literal device list would let a configuration file that simply leaves the
# line out expose cards nobody allocated to this container; an explicit value
# is the only way to name devices.
ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES-}"
NPU_DEVICES="${NPU_DEVICES:-}"
BIND_MOUNTS="${BIND_MOUNTS:-}"
OPTIONAL_BIND_MOUNTS="${OPTIONAL_BIND_MOUNTS:-}"
LONG_RUNNING="${LONG_RUNNING:-true}"
CONTAINER_COMMAND="${CONTAINER_COMMAND:-}"
INIT_TIMEOUT_SECONDS="${INIT_TIMEOUT_SECONDS:-30}"

[ -n "$CONTAINER_NAME" ] || die 'CONTAINER_NAME is required'
[ -n "$CONTAINER_USER" ] || die 'CONTAINER_USER is required'
[ -n "$CONTAINER_UID" ] || die 'CONTAINER_UID is required'
[ -n "$CONTAINER_GID" ] || die 'CONTAINER_GID is required'
[ -n "$CONTAINER_HOME" ] || die 'CONTAINER_HOME is required'

[[ "$CONTAINER_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || die "CONTAINER_NAME is not a plain Docker name: $CONTAINER_NAME"
[[ "$CONTAINER_USER" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ ]] \
    || die "CONTAINER_USER is not a plain account name: $CONTAINER_USER"
[[ "$CONTAINER_UID" =~ ^[0-9]+$ ]] || die "CONTAINER_UID must be numeric: $CONTAINER_UID"
[[ "$CONTAINER_GID" =~ ^[0-9]+$ ]] || die "CONTAINER_GID must be numeric: $CONTAINER_GID"
[ "$CONTAINER_UID" -gt 0 ] || die 'CONTAINER_UID must not be 0'
[ "$CONTAINER_GID" -gt 0 ] || die 'CONTAINER_GID must not be 0'
[ "${CONTAINER_HOME#/}" != "$CONTAINER_HOME" ] || die "CONTAINER_HOME must be absolute: $CONTAINER_HOME"
[[ "$INIT_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || die "INIT_TIMEOUT_SECONDS must be numeric: $INIT_TIMEOUT_SECONDS"

case "$LONG_RUNNING" in
    true|false) ;;
    *) die "LONG_RUNNING must be true or false: $LONG_RUNNING" ;;
esac
[ -n "$CONTAINER_HOSTNAME" ] || CONTAINER_HOSTNAME="${CONTAINER_NAME//_/-}"
[[ "$CONTAINER_HOSTNAME" =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ ]] \
    || die "CONTAINER_HOSTNAME is not a valid host name: $CONTAINER_HOSTNAME"

if [ "$LONG_RUNNING" = true ] && [ -n "$CONTAINER_COMMAND" ]; then
    die 'LONG_RUNNING=true keeps the container alive with a keepalive; CONTAINER_COMMAND must be empty'
fi
if [ "$LONG_RUNNING" = false ] && [ -z "$CONTAINER_COMMAND" ]; then
    die 'LONG_RUNNING=false requires CONTAINER_COMMAND'
fi

read -r -a SECURITY_OPTS_ARR <<<"$SECURITY_OPTS"
read -r -a SUPPLEMENTARY_GIDS_ARR <<<"$SUPPLEMENTARY_GIDS"
read -r -a NPU_DEVICES_ARR <<<"$NPU_DEVICES"

# ---------------------------------------------------------------------------
# Mount resolution.
#
# Each entry is `source:destination[:ro|rw]`. Required mounts must exist on the
# host; an optional mount whose source is missing is reported and skipped.
# A destination that differs from the source is a legitimate remapping.
# ---------------------------------------------------------------------------
MOUNT_ARGS=()

parse_mount() {
    local entry="$1" source destination mode
    IFS=':' read -r source destination mode <<<"$entry"
    [ -n "$source" ] && [ -n "$destination" ] || die "bad mount entry (want source:destination[:ro|rw]): $entry"
    [ "${source#/}" != "$source" ] || die "mount source must be absolute: $entry"
    [ "${destination#/}" != "$destination" ] || die "mount destination must be absolute: $entry"
    case "${mode:-rw}" in
        ro|rw) ;;
        *) die "mount mode must be ro or rw: $entry" ;;
    esac
    MOUNT_SOURCE="$source"
    MOUNT_DESTINATION="$destination"
    MOUNT_MODE="${mode:-rw}"
}

for entry in $BIND_MOUNTS; do
    parse_mount "$entry"
    if [ ! -e "$MOUNT_SOURCE" ]; then
        die "required bind-mount source does not exist on the host: $MOUNT_SOURCE (entry: $entry)"
    fi
    if [ "$MOUNT_MODE" = ro ]; then
        MOUNT_ARGS+=("type=bind,source=$MOUNT_SOURCE,destination=$MOUNT_DESTINATION,readonly")
    else
        MOUNT_ARGS+=("type=bind,source=$MOUNT_SOURCE,destination=$MOUNT_DESTINATION")
    fi
done

for entry in $OPTIONAL_BIND_MOUNTS; do
    parse_mount "$entry"
    if [ ! -e "$MOUNT_SOURCE" ]; then
        log "optional mount source is missing; skipping: $MOUNT_SOURCE (entry: $entry)"
        continue
    fi
    if [ "$MOUNT_MODE" = ro ]; then
        MOUNT_ARGS+=("type=bind,source=$MOUNT_SOURCE,destination=$MOUNT_DESTINATION,readonly")
    else
        MOUNT_ARGS+=("type=bind,source=$MOUNT_SOURCE,destination=$MOUNT_DESTINATION")
    fi
done

# ---------------------------------------------------------------------------
# Verify the configured identity can read its own HOME.
#
# `entry.sh` never creates or chowns HOME -- it is normally a bind mount of the
# host home directory, and the P0 contract forbids touching it. So if the
# container account's UID/GID differ from the owner of the mounted home, the
# mounted files stay owned by the host account and a login shell cannot read
# `~/.bashrc`, which fails with "Permission denied" only after the operator is
# already inside the container. Catch it here, before anything is created.
#
# This is a read/traverse check, not a write check: with the mode 750 that
# `useradd`-created homes commonly have, a non-owner cannot even enter the
# directory.
# ---------------------------------------------------------------------------
gid_matches() {
    local want="$1" gid
    [ "$want" = "$CONTAINER_GID" ] && return 0
    for gid in "${SUPPLEMENTARY_GIDS_ARR[@]:-}"; do
        [ -n "$gid" ] && [ "$gid" = "$want" ] && return 0
    done
    return 1
}

check_container_home_readable() {
    local entry source destination mode relative best_depth=-1 host_home='' depth
    local file_uid file_gid file_mode mode_digits perm_owner perm_group perm_other need=5

    for entry in $BIND_MOUNTS $OPTIONAL_BIND_MOUNTS; do
        IFS=':' read -r source destination mode <<<"$entry"
        if [ -z "$destination" ]; then continue; fi
        case "$CONTAINER_HOME" in
            "$destination")   relative='' ;;
            "$destination"/*) relative="${CONTAINER_HOME#"$destination"/}" ;;
            *)                continue ;;
        esac
        depth=$(printf '%s' "$destination" | tr -cd '/' | wc -c)
        if [ "$depth" -gt "$best_depth" ]; then
            best_depth="$depth"
            if [ -n "$relative" ]; then host_home="$source/$relative"; else host_home="$source"; fi
        fi
    done

    # HOME is not covered by a bind mount: nothing of the host can block it.
    [ -n "$host_home" ] || return 0

    if [ ! -e "$host_home" ]; then
        log "note: CONTAINER_HOME $CONTAINER_HOME ($host_home on the host) does not exist yet"
        return 0
    fi

    file_uid="$(stat -c '%u' "$host_home")"
    file_gid="$(stat -c '%g' "$host_home")"
    file_mode="$(stat -c '%a' "$host_home")"
    mode_digits="${file_mode: -3}"
    perm_owner=$(( ${mode_digits:0:1} ))
    perm_group=$(( ${mode_digits:1:1} ))
    perm_other=$(( ${mode_digits:2:1} ))

    local allowed=0
    if [ "$file_uid" = "$CONTAINER_UID" ]; then
        [ $(( perm_owner & need )) -eq $need ] && allowed=1
    elif gid_matches "$file_gid"; then
        [ $(( perm_group & need )) -eq $need ] && allowed=1
    else
        [ $(( perm_other & need )) -eq $need ] && allowed=1
    fi
    [ "$allowed" -eq 1 ] && return 0

    die "the container identity cannot read CONTAINER_HOME.
    host path     : $host_home
    owner / mode  : uid=$file_uid gid=$file_gid mode=$file_mode
    container     : CONTAINER_USER=$CONTAINER_USER UID=$CONTAINER_UID GID=$CONTAINER_GID
  The bind mount keeps the host ownership, so a login shell fails with
  'Permission denied' on $CONTAINER_HOME/.bashrc. Set CONTAINER_UID and
  CONTAINER_GID to the UID/GID that owns the host home directory:
      stat -c 'CONTAINER_UID=%u CONTAINER_GID=%g' $host_home
  Do not chown the host home directory to work around this."
}

check_container_home_readable

DEVICE_ARGS=()
for device in "${NPU_DEVICES_ARR[@]:-}"; do
    [ -n "$device" ] || continue
    [ -e "$device" ] || die "NPU device path does not exist on the host: $device"
    DEVICE_ARGS+=(--device "$device")
done

# ---------------------------------------------------------------------------
# Docker availability and image
# ---------------------------------------------------------------------------
need_docker() {
    command -v docker >/dev/null 2>&1 || die 'the docker CLI is not on PATH'
}

docker_run_argv() {
    DOCKER_RUN=(
        docker run -d
        --name "$CONTAINER_NAME"
        --hostname "$CONTAINER_HOSTNAME"
        --network "$NETWORK"
        --cgroupns "$CGROUPNS"
        --shm-size "$SHM_SIZE"
        --ulimit "memlock=$MEMLOCK"
        --init
        -e "CONTAINER_USER=$CONTAINER_USER"
        -e "CONTAINER_UID=$CONTAINER_UID"
        -e "CONTAINER_GID=$CONTAINER_GID"
        -e "CONTAINER_HOME=$CONTAINER_HOME"
        -e "SUPPLEMENTARY_GIDS=$SUPPLEMENTARY_GIDS"
        -e "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_VISIBLE_DEVICES"
    )
    if [ "$PRIVILEGED" = true ]; then
        DOCKER_RUN+=(--privileged)
        if [ -n "$NPU_DEVICES" ]; then
            # Say this out loud rather than letting the operator believe the
            # device list is confining the container. It is not.
            log "WARNING: PRIVILEGED=true makes NPU_DEVICES advisory;"
            log "         the container can open every host device."
        fi
    fi
    for opt in "${SECURITY_OPTS_ARR[@]:-}"; do
        [ -n "$opt" ] && DOCKER_RUN+=(--security-opt "$opt")
    done
    for device in "${DEVICE_ARGS[@]:-}"; do
        [ -n "$device" ] && DOCKER_RUN+=("$device")
    done
    for mount in "${MOUNT_ARGS[@]:-}"; do
        [ -n "$mount" ] && DOCKER_RUN+=(--mount "$mount")
    done
    DOCKER_RUN+=("$IMAGE")
    if [ "$LONG_RUNNING" = false ]; then
        # shellcheck disable=SC2206  # intentional word splitting of the command
        DOCKER_RUN+=($CONTAINER_COMMAND)
    fi
}

print_command() {
    printf '%q ' "${DOCKER_RUN[@]}"
    printf '\n'
}

# ---------------------------------------------------------------------------
# Dry run: validate what can be validated without mutating anything.
# ---------------------------------------------------------------------------
if [ "$DRY_RUN" = 1 ]; then
    docker_run_argv
    printf 'config: %s\n' "$CONFIG"
    printf 'image:  %s\n' "$IMAGE"
    printf 'dry run: the following command was NOT executed\n'
    print_command
    if command -v docker >/dev/null 2>&1; then
        if docker info >/dev/null 2>&1; then
            if docker image inspect "$IMAGE" >/dev/null 2>&1; then
                log "dry run: image $IMAGE exists locally"
            else
                log "dry run: image $IMAGE does not exist locally; a real run would fail"
            fi
            if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
                log "dry run: container $CONTAINER_NAME already exists; a real run would fail"
            fi
        else
            log 'dry run: the Docker daemon is not accessible; a real run would fail'
        fi
    else
        log 'dry run: the docker CLI is not on PATH; a real run would fail'
    fi
    exit 0
fi

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
need_docker
docker info >/dev/null 2>&1 || die 'cannot reach the Docker daemon (is it running and is this user allowed?)'

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    state="$(docker container inspect --format '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo unknown)"
    die "container $CONTAINER_NAME already exists (state: $state); refusing to remove, stop or reuse it. Choose another name, or remove it yourself after checking it is not in use."
fi

docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || die "image $IMAGE does not exist locally; build it first (see README.md). This script never pulls implicitly."

# ---------------------------------------------------------------------------
# Create and start
# ---------------------------------------------------------------------------
docker_run_argv
log "creating container $CONTAINER_NAME from $IMAGE"
if ! created="$("${DOCKER_RUN[@]}" 2>&1)"; then
    log "docker run failed: $created"
    die "container $CONTAINER_NAME was not created (any partially created container is left intact for diagnosis)"
fi

# ---------------------------------------------------------------------------
# Wait for the entrypoint to create and verify the identity.
# ---------------------------------------------------------------------------
deadline=$(( $(date +%s) + INIT_TIMEOUT_SECONDS ))
ready=0
while [ "$(date +%s)" -lt "$deadline" ]; do
    if ! docker container inspect --format '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -qx true; then
        log 'the container stopped before the identity was ready; see `docker logs`'
        break
    fi
    if docker exec "$CONTAINER_NAME" /usr/local/bin/entry.sh --check-ready >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 1
done

if [ "$ready" != 1 ]; then
    log "--- docker logs $CONTAINER_NAME (last 40 lines) ---"
    docker logs --tail 40 "$CONTAINER_NAME" >&2 2>&1 || true
    die "user initialization did not complete within ${INIT_TIMEOUT_SECONDS}s; container $CONTAINER_NAME is left intact for diagnosis"
fi

# ---------------------------------------------------------------------------
# Independent verification through docker exec.
# ---------------------------------------------------------------------------
actual_uid="$(docker exec -u "$CONTAINER_USER" -e "HOME=$CONTAINER_HOME" "$CONTAINER_NAME" id -u)"
actual_gid="$(docker exec -u "$CONTAINER_USER" -e "HOME=$CONTAINER_HOME" "$CONTAINER_NAME" id -g)"
actual_home="$(docker exec -u "$CONTAINER_USER" -e "HOME=$CONTAINER_HOME" "$CONTAINER_NAME" printenv HOME)"

[ "$actual_uid" = "$CONTAINER_UID" ] || die "verification failed: container UID is $actual_uid, expected $CONTAINER_UID"
[ "$actual_gid" = "$CONTAINER_GID" ] || die "verification failed: container GID is $actual_gid, expected $CONTAINER_GID"
[ "$actual_home" = "$CONTAINER_HOME" ] || die "verification failed: container HOME is $actual_home, expected $CONTAINER_HOME"

for gid in "${SUPPLEMENTARY_GIDS_ARR[@]:-}"; do
    [ -n "$gid" ] || continue
    [ "$gid" = "$CONTAINER_GID" ] && continue
    if ! docker exec -u "$CONTAINER_USER" "$CONTAINER_NAME" id -G | tr ' ' '\n' | grep -qx "$gid"; then
        die "verification failed: $CONTAINER_USER is not a member of supplementary GID $gid"
    fi
done

log "container $CONTAINER_NAME is ready: user=$CONTAINER_USER uid=$actual_uid gid=$actual_gid home=$actual_home"
docker exec -u "$CONTAINER_USER" -e "HOME=$CONTAINER_HOME" "$CONTAINER_NAME" id
