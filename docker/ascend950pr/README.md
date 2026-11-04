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
## Scripts

| script | purpose |
| --- | --- |
| `create-container.sh` | create the container from the image |
| `verify-container.sh` | verify the environment from inside it |
| `verify-identity.sh` | verify the container account by asking the container |
| `device-select.sh` | require an explicit device, report occupancy |
| `device-run.sh` | serialized device run with per-run evidence |
| `check-receipt.sh` | judge a device receipt; missing data is false |
| `manifest.sh` | record the stack a baseline ran on |

## Configuration

`env.example.sh` is the single source of configuration. Copy it once:

```
cp env.example.sh env.sh
chmod 600 env.sh
```

`env.sh` is git-ignored and never committed. It is parsed as data, never
executed; unknown or duplicate keys fail rather than being ignored. Exporting
the same names in a shell has no effect, so a stale export cannot silently
override what the machine says it is.

A device id is deliberately **not** in the file: it authorizes one run, not a
machine. Pass `--device N` or set `TF_DEVICE_ID` per invocation.

## Two modes

On this platform NPU cases must enter as root, while CPU-only work must not.

```
# CPU-only analyze / finalize / equality: ordinary account, required
docker exec -u <user> <container> bash verify-container.sh

# On-device work: root, and only when asked for
docker exec <container> bash verify-container.sh --npu
```

Root without `--npu` is refused (exit 2) as an accident; `--npu` without root
is also refused (exit 2), because that attempt is not representative. Root does
not authorize a device: `--npu` still requires an explicit device id.

## Occupancy is reported, not enforced

This is a shared server and a busy card is not a fact about the code under
test. `device-select.sh` records who is on the card and proceeds. Setting
`TF_REQUIRE_EXCLUSIVE=1` turns a held card into **BLOCKED_ENV** (exit 75) --
the run is reported as not having happened, never as having passed.

## What the evidence does and does not prove

`verify-container.sh` runs the same CPU-side analyze/finalize/equality suite in
both modes. `--npu` changes the identity it runs as, not what it runs. Its PASS
is evidence about the root environment, **not** about the device, and it is
labelled that way in the log and the manifest. No device kernel has been
executed by this container; `torch_npu` is not installed in the venv.

A device PASS requires a receipt that passes `check-receipt.sh`: both markers
distinguished, tensors actually moved, peak allocation > 0, at least one
`.npubin`, synchronization successful with the original error text carried, and
exactly one visible device. Missing data is false.
