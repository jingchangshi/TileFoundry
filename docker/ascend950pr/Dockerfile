# TileFoundry Ascend950PR development container image.
#
# Adapted from the AgentGates tilelang-hivm reference at
#   agent_gates/adapters/tilelang-hivm/docker/Dockerfile
# (see also adapters/ascendnpu-ir/docker). The toolchain recipe is inherited
# deliberately: it is the one those adapters validate, and diverging from it
# would make a build here impossible to compare with a passing reference run.
#
# The image provides the *base toolchain* only. It deliberately does not
# contain CANN, the TileFoundry source tree, the Python venv, or any build
# products, and it never touches the host NPU driver:
#
#   * CANN is machine-specific and version-specific, so it is BIND-MOUNTED from
#     the host, not baked in. Baking a CANN version into an image silently pins
#     every future run to it, and the driver on the host is not ours to move.
#   * The TileFoundry source tree is bind-mounted, so the container imports and
#     compiles exactly the bytes the host sees, including uncommitted edits.
#   * The Python venv (tfpy312) is bind-mounted for the same reason: it is the
#     user environment, with torch already installed from a pinned mirror, and
#     copying it into an image would fork it from the one the CPU evidence was
#     produced in.
#
# Nothing here installs or upgrades a driver, CANN, or a kernel module. The
# container can only use an NPU that the host has already made available.
#
# Toolchain choices, inherited from the reference and kept for the same reasons:
#
#   * `gcc`/`g++` from Ubuntu noble, on the standard PATH. The reference build
#     tree records `CMAKE_C_COMPILER=/usr/bin/gcc`; forcing clang here would
#     diverge from the tree that tree was configured against.
#   * `cmake` >= 3.26 is a hard floor. Noble ships 3.28.3, which meets it, but
#     cmake is installed from pip and pinned so the recipe stays identical to
#     the reference image rather than tracking whatever noble updates to.
#   * `ninja` >= 1.12 is what the build documentation configures against
#     (`cmake .. -G Ninja`); noble's `ninja-build` is 1.11.1, below that, so
#     ninja is installed from pip too.
#   * `python3` from the distribution, on the standard PATH. The identity probe
#     is a bare `python3`/`stat` call in a plain `docker exec`, so an
#     interpreter that needs an activation first would make every check depend
#     on the venv being sourced. The venv is mounted for running TileFoundry,
#     not for making `python3` work.
#
# Only the `cmake` and `ninja` executables are exposed on the standard PATH via
# `/usr/local/bin`. The interpreter stays the distribution `python3`.
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
# Pinned because pip is the only source of a new enough cmake/ninja on noble.
ARG CMAKE_VERSION=3.31.6
ARG NINJA_VERSION=1.13.0
ARG LLVM_VERSION=18
# Ubuntu archive mirror. The Aliyun mirror is the default because
# archive.ubuntu.com is slow from the usual build hosts. Both the release and
# the security suites are served by the same mirror, and the distribution signing
# key still authenticates every package, so a wrong or hostile mirror cannot
# install one. Override it to use another mirror.
#
# Plain HTTP, not HTTPS, on purpose: the base image has no ca-certificates yet,
# and an HTTPS mirror therefore fails TLS verification while `apt-get update`
# still exits 0, which would let the build continue with an empty package index.
# Ubuntu's own default sources are HTTP for the same reason; upgrade to HTTPS
# after ca-certificates is installed if a deployment requires it.
# aliyun may be slow in some blue server
# ARG UBUNTU_MIRROR=http://mirrors.aliyun.com/ubuntu/
ARG UBUNTU_MIRROR=http://repo.huaweicloud.com/ubuntu/

# TORCH_DEVICE_BACKEND_AUTOLOAD=0 is a SAFETY setting, not a speed one.
#
# This image ships the torch_npu build, so `import torch` auto-loads the NPU
# backend extension: a device-path touch before any CPU test runs, and a hard
# RuntimeError when the device stack is unavailable. Setting it at the IMAGE
# level is the only place that reaches a bare `docker exec`, because docker
# exec does not inherit the entrypoint environment (see entry.sh). Device work
# opts back in per invocation instead of every CPU invocation opting out.
#
# NOTE: no inline comments inside the ENV continuation below. Docker would fold
# them into the value.
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0

# ---------------------------------------------------------------------------
# System packages.
#
# * `gcc`, `g++`, `make` and `build-essential` are the build toolchain;
#   the reference `CMakeCache.txt` records exactly those compilers.
# * `python3`, `python3-dev`, `python3-venv` and `python3-pip`: the interpreter
#   must be usable with no activation, and `python3-venv` is what the
#   cmake/ninja pip layer below needs.
# * `passwd` provides useradd/groupadd/usermod, which entry.sh needs at start.
# * `util-linux` provides `setpriv`/`runuser`, used to drop privileges.
# * `clang`/`lld`/LLVM and the dev libraries are not required by the Ascend
#   build (it uses gcc). They are kept so this file stays comparable to the
#   reference image and so LLVM-side work is possible in the same container;
#   the libraries are the ones LLVM/MLIR link against.
#
# Noble uses the deb822 `/etc/apt/sources.list.d/ubuntu.sources`, not the legacy
# `sources.list`, so the mirror is rewritten there. The rewrite is guarded: if a
# future base image drops that file the build fails here rather than silently
# installing from a different mirror. `apt-get update` is not trusted to fail on
# its own -- it warns and exits 0 when an index is unreachable -- so the rewrite
# and the resulting index are both verified explicitly.
# ---------------------------------------------------------------------------
RUN set -eux; \
    sources=/etc/apt/sources.list.d/ubuntu.sources; \
    [ -f "$sources" ] || { echo "expected $sources to exist" >&2; exit 1; }; \
    sed -i \
        -e "s|^URIs: http://archive.ubuntu.com/ubuntu/|URIs: ${UBUNTU_MIRROR}|" \
        -e "s|^URIs: http://security.ubuntu.com/ubuntu/|URIs: ${UBUNTU_MIRROR}|" \
        -e "s|^URIs: https://archive.ubuntu.com/ubuntu/|URIs: ${UBUNTU_MIRROR}|" \
        -e "s|^URIs: https://security.ubuntu.com/ubuntu/|URIs: ${UBUNTU_MIRROR}|" \
        "$sources"; \
    grep "^URIs:" "$sources"; \
    [ "$(grep -c "^URIs: ${UBUNTU_MIRROR}$" "$sources")" -eq 2 ] \
        || { echo "mirror rewrite did not cover both suites" >&2; exit 1; }; \
    apt-get -o Acquire::Retries=5 update; \
    apt-get install -y --no-install-recommends \
        bash coreutils ca-certificates curl wget gnupg git \
        make gcc g++ \
        build-essential pkg-config file ccache \
        clang-${LLVM_VERSION} lld-${LLVM_VERSION} libclang-${LLVM_VERSION}-dev \
        libclang-cpp${LLVM_VERSION}-dev llvm-${LLVM_VERSION}-dev \
        python3 python3-dev python3-venv python3-pip \
        passwd util-linux \
        locales \
        libgomp1 zlib1g-dev libzstd-dev libedit-dev libffi-dev \
        libssl-dev libxml2-dev libncurses-dev liblz4-dev; \
    # Prove the index actually came from the mirror rather than an empty cache:
    # apt names the list files after the mirror host, and the indexes are lz4
    # compressed, so the glob must allow a suffix. An unreachable mirror leaves
    # this directory empty but still exits 0 above.
    ls /var/lib/apt/lists/*_Packages* >/dev/null \
        || { echo "apt package index is empty after update" >&2; exit 1; }; \
    sed -i 's/^# *\(en_US.UTF-8\)/\1/' /etc/locale.gen; \
    locale-gen; \
    update-alternatives --install /usr/bin/clang clang "/usr/bin/clang-${LLVM_VERSION}" 100; \
    update-alternatives --install /usr/bin/clang++ clang++ "/usr/bin/clang++-${LLVM_VERSION}" 100; \
    update-alternatives --install /usr/bin/lld lld "/usr/bin/lld-${LLVM_VERSION}" 100; \
    update-alternatives --install /usr/bin/ld.lld ld.lld "/usr/bin/ld.lld-${LLVM_VERSION}" 100; \
    # `cc`/`c++` keep pointing at gcc/g++, because the reference build tree was
    # configured against /usr/bin/gcc and /usr/bin/g++. A build that wants
    # clang passes it explicitly; the image never silently switches.
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# cmake and ninja, pinned above the distribution versions.
#
# Installed into an image-private venv so the distribution Python stays
# untouched, then only the two executables are linked onto the standard PATH.
# `--break-system-packages` is therefore not needed.
#
# PYPI_MIRROR is declared here, immediately before its only use, so changing it
# invalidates this layer alone. Declaring it above the system-package layer would
# make a mirror change rebuild that entire layer.
# ---------------------------------------------------------------------------
# The Aliyun mirror is the default because the public PyPI index is slow or
# unreachable from the usual build hosts. The wheel contents are identical and
# pip still verifies them against the index. Override it to use another index.
ARG PYPI_MIRROR=https://mirrors.aliyun.com/pypi/simple/
RUN set -eux; \
    python3 -m venv /opt/build-tools; \
    /opt/build-tools/bin/pip install --no-cache-dir --disable-pip-version-check \
        -i "${PYPI_MIRROR}" "cmake==${CMAKE_VERSION}" "ninja==${NINJA_VERSION}"; \
    ln -sf /opt/build-tools/bin/cmake /usr/local/bin/cmake; \
    ln -sf /opt/build-tools/bin/ninja /usr/local/bin/ninja; \
    rm -rf /root/.cache; \
    command -v cmake; cmake --version | head -1; \
    command -v ninja; ninja --version; \
    command -v python3; python3 --version; \
    command -v gcc; gcc --version | head -1; \
    command -v clang; clang --version | head -1

# The entrypoint runs as root so it can create the configured UID/GID identity
# inside the container, then the Adapter runs its own `docker exec -u <user>`.
#
# `entry.sh` is copied verbatim from the reference adapter
# (adapters/tilelang-hivm/docker/entry.sh, byte-identical to
# adapters/ascendnpu-ir/docker/entry.sh). The file is project-neutral -- it
# creates and verifies the container account and writes a readiness marker, and
# knows nothing about any project -- so it is carried over unmodified rather
# than adapted. Keep it that way: an adapted copy would need its own review for
# behaviour that is already reviewed.
COPY entry.sh /usr/local/bin/entry.sh
RUN chmod +x /usr/local/bin/entry.sh

# `--init` is recommended by create-container.sh, so this container gets a
# reaping init without the image depending on tini.
ENTRYPOINT ["/usr/local/bin/entry.sh"]
CMD ["tail", "-f", "/dev/null"]