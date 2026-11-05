# Configuration for the TileFoundry Ascend950PR container.
#
# Create the real file once, on this machine:
#
#     cp env.example.sh env.sh
#     chmod 600 env.sh
#
# env.sh is git-ignored. It never leaves the host.
#
# This file is the ONLY source of these values. Exporting them in a shell has
# no effect -- the scripts read this file, so a stale export cannot silently
# override what the machine says it is. The file is parsed as DATA and never
# executed.
#
# A value left empty means "not configured", which is an error for the keys the
# scripts require and permitted for the optional ones. Nothing is guessed: a
# missing device id is a refusal, not a search.

# --- identity -------------------------------------------------------------
# Required. The container account. The scripts verify this by asking the
# container itself (id -u / id -g / printenv HOME) rather than assuming the
# entrypoint did what it was told.
CONTAINER_USER=
CONTAINER_UID=
CONTAINER_GID=
CONTAINER_HOME=
# Space-separated. Each is checked for actual membership with `id -G`.
SUPPLEMENTARY_GIDS=

# --- container ------------------------------------------------------------
CONTAINER_NAME=tf-ascend950pr
IMAGE=tilefoundry-ascend950pr-dev:24.04

# --- paths ----------------------------------------------------------------
HOST_REPO=${HOME}/workspace/TileFoundry
HOST_CANN=${CANN_ROOT}

# --- TWO interpreters, and they are not interchangeable --------------------
#
# HOST_VENV is the CPU environment. It has NO torch_npu and must not get one:
# TileFoundry analyze / finalize / equality are CPU work, and adding a device
# runtime to the interpreter that produces compiler evidence would change what
# that evidence was produced on. Verified: `import torch_npu` raises
# ModuleNotFoundError in tfpy312.
#
# NPU_PYTHON is the Runtime Twin interpreter. It is the only one that can run
# device work, and it is owned by another group -- mount it READ-ONLY. Nothing
# here writes to it, and nothing here installs into it.
#
# Requiring CANN to be sourced first is not incidental: without it torch_npu
# fails to import with "libhccl.so: cannot open shared object file", which reads
# like a broken environment rather than an unexported one. Verified both ways.
HOST_VENV=${HOME}/workspace/tfpy312
NPU_PYTHON=${CONDA_ROOT}/envs/l00849257/bin/python
NPU_ENV_DIR=${CONDA_ROOT}/envs/l00849257

# Whether NPU_PYTHON is mounted into the container. Device cases need it; CPU
# cases do not, and leaving it out is the smaller surface.
MOUNT_NPU_ENV=0

# --- device ---------------------------------------------------------------
# Deliberately NOT configured here: a device id authorizes ONE run and is
# passed per invocation (--device N / TF_DEVICE_ID). Baking it into the machine
# configuration would make an accident look like a default.
#
# Occupancy is REPORTED, not enforced: this is a shared server and a busy card
# is not a reason to refuse work. Set TF_REQUIRE_EXCLUSIVE=1 only when you need
# a card guaranteed free, in which case a busy card yields BLOCKED_ENV -- the
# run is reported as not having happened, never as having passed.
TF_REQUIRE_EXCLUSIVE=0

# --- CPU work must never touch the NPU, including at IMPORT time -------------
#
# TORCH_DEVICE_BACKEND_AUTOLOAD=0 is REQUIRED for every CPU-only invocation, and
# it is a safety setting rather than a speed knob.
#
# This image ships the torch_npu build. On import, torch auto-loads the NPU
# backend extension, which OPENS THE DEVICE PATH -- and with the device stack
# unavailable it raises
#
#     RuntimeError: Failed to load the backend extension: torch_npu.
#
# So a plain `import torch` in this image is already a device-path touch, before
# a single line of the test has run. That was found the hard way: two CPU
# equality tests reported NOT_RUN for a reason that had nothing to do with their
# module, and every one of them had walked onto the frozen device path merely by
# importing torch.
#
# With the variable set, measured in-container: torch 2.12.0+cpu, torch_npu NOT
# loaded, cuda False, exit 0. CPU arithmetic, no device, no ambiguity.
#
# MECHANISM WARNING: this file is parsed as DATA by the host scripts and passed
# to `docker run -e`. Docker exec does NOT inherit the entrypoint environment
# (see entry.sh), so a value set only here does NOT reach a later `docker exec`.
# Until the image carries it as an ENV (see Dockerfile), every `docker exec` that
# imports torch must pass `-e TORCH_DEVICE_BACKEND_AUTOLOAD=0` explicitly. The
# acceptance script exports it itself so it is correct either way.
TORCH_DEVICE_BACKEND_AUTOLOAD=0