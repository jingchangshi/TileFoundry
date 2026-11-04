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
HOST_REPO=/home/s00653124/workspace/TileFoundry
HOST_VENV=/home/s00653124/workspace/tfpy312
HOST_CANN=/data/pri/cann-9.2.0

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
