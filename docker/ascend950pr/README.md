# TileFoundry Ascend950PR development container

Adapted from the AgentGates reference:

- `agent_gates/adapters/tilelang-hivm/docker/` (Dockerfile, entry.sh,
  create-container.sh, config.example.env)
- `agent_gates/adapters/ascendnpu-ir/docker/` (same layout)

What changed from the reference, and why:

| file | change |
| --- | --- |
| `Dockerfile` | comments and justifications rewritten for TileFoundry; the
  toolchain recipe is unchanged |
| `entry.sh` | **byte-identical copy**, unmodified (SHA-256 in the commit) |
| `create-container.sh` | header rewritten; default image name is
  `tilefoundry-ascend950pr-dev:24.04`; behaviour unchanged |
| `config.example.env` | values and examples specific to TileFoundry + CANN
  9.2.0 + the `tfpy312` venv; every value still empty |

`entry.sh` is carried over unmodified on purpose. It creates and verifies the
container account and writes a readiness marker, and it knows nothing about any
project. An adapted copy would need its own review for behaviour that is
already reviewed.

## Build

```sh
cd docker/ascend950pr
docker build -t tilefoundry-ascend950pr-dev:24.04 .
```

The image contains the toolchain only: Ubuntu 24.04, distribution GCC/G++,
Clang/LLVM 18, CMake 3.31.6, Ninja 1.13.0, distribution Python 3.12. It
contains no CANN, no TileFoundry source, no venv, and no NPU driver, so the
image itself can be built on any host.

## Create

```sh
cp config.example.env config.env
$EDITOR config.env          # identity, mounts, device 0 only
./create-container.sh --config ./config.env --dry-run
./create-container.sh --config ./config.env
```

`config.env` is git-ignored. On this host the shape is:

```sh
CONTAINER_NAME=tf-ascend950pr
CONTAINER_USER=s00653124
CONTAINER_UID=1071
CONTAINER_GID=1071
CONTAINER_HOME=/home/s00653124
ASCEND_VISIBLE_DEVICES=0
NPU_DEVICES="/dev/davinci0 /dev/davinci_manager /dev/hisi_hdc"
BIND_MOUNTS="/data/pri/cann-9.2.0:/data/pri/cann-9.2.0:ro"
BIND_MOUNTS="$BIND_MOUNTS /home/s00653124/workspace/TileFoundry:/home/s00653124/workspace/TileFoundry"
BIND_MOUNTS="$BIND_MOUNTS /home/s00653124/workspace/tfpy312:/home/s00653124/workspace/tfpy312"
OPTIONAL_BIND_MOUNTS="/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro"
```

## CANN

CANN is bind-mounted, never baked in, and never installed from inside. To use
it in a shell inside the container:

```sh
source /data/pri/cann-9.2.0/cann/set_env.sh
```

Device 0 only. `ASCEND_VISIBLE_DEVICES=0` restricts what the runtime offers and
`NPU_DEVICES` restricts what docker passes through; both are set so neither is
the only barrier.

## Shared-host rules

This container is for running TileFoundry, not for changing the machine:

- no driver install or upgrade, no CANN install or upgrade, no kernel module;
- never kill or signal another user's process;
- serialize our own device runs with a `flock`;
- record an `npu-smi info` snapshot before and after each run;
- on a shared server, any timing number is provisional. Numerics are not
  affected by other tenants; latency is.
