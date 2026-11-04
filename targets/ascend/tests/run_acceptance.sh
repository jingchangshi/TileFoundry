#!/bin/sh
# The acceptance run, in the env of record.
#
# Provenance rules this script is built around:
#
#   1. The interpreter identity is printed FIRST, so every line below is
#      attributable to a named interpreter rather than to "python".
#   2. Nothing uses /opt/tf-venv and nothing runs on the host. PY is the durable
#      conda env and the script is meant to be run INSIDE a container that has this
#      HOME mounted. Running it on the host proves nothing about M3.
#   3. The package versions used in the report were read from dist-info directory
#      names and from the installed file layout, WITHOUT importing anything on the
#      host. That is stated here so the report cannot be misread as a host run.
#   4. THE DEVICE STACK IS NEVER TOUCHED. No torch_npu import, no CANN sourcing, no
#      npu-smi, no /dev/davinci* open. A host-wide filesystem wedge was traced
#      (correlation only) to a davinci0 open, and nothing in this module needs an
#      NPU, so the rule is absolute rather than conditional. The script says in its
#      own output that a green run is CPU-only evidence and not device evidence in
#      either direction. The dist-info LISTING below is a directory read and imports
#      nothing; it is reported only so the reader knows which packages exist.
#   5. No line stops the script. Every check reports its own exit code and the next
#      one still runs, so one failure does not silently truncate the evidence.
set -u
# THE INTERPRETER IS RESOLVED, NOT ASSUMED FROM $HOME.
#
# An earlier version of this script wrote $HOME/.conda/envs/.../bin/python3.12. That
# is correct on the host and WRONG in the container, where docker exec reports
# HOME=/root and the env lives under /home/s00653124. The failure mode is nasty: the
# shell reports "not found" for a path that looks right, and it reads like the env is
# missing rather than like the script guessed. So the env root is resolved explicitly
# in priority order, and the path actually used is PRINTED before anything runs.
ENV_ROOT=${TF_ENV_ROOT:-}
if [ -z "$ENV_ROOT" ]; then
  for candidate in \
    /home/s00653124/.conda/envs/tilefoundry-npu \
    "$HOME/.conda/envs/tilefoundry-npu"
  do
    if [ -x "$candidate/bin/python3.12" ]; then
      ENV_ROOT=$candidate
      break
    fi
  done
fi
if [ -z "$ENV_ROOT" ] || [ ! -x "$ENV_ROOT/bin/python3.12" ]; then
  echo "FATAL: no interpreter of record found. HOME=$HOME" >&2
  echo "       looked for /home/s00653124/.conda/envs/tilefoundry-npu and" >&2
  echo "       $HOME/.conda/envs/tilefoundry-npu; set TF_ENV_ROOT to override." >&2
  exit 2
fi
PY=$ENV_ROOT/bin/python3.12
echo "env_root=$ENV_ROOT   (resolved, not assumed)"
REPO=${REPO:-/home/s00653124/workspace/TileFoundry}
export PYTHONPATH=$REPO/src:$REPO/targets/ascend:$REPO/targets/nvidia
export PYTHONDONTWRITEBYTECODE=1
# STOP TORCH AUTO-LOADING THE NPU BACKEND. This is a safety setting, not a
# convenience, and without it the CPU tests cannot even import torch.
#
# This container has the torch_npu build installed, so `import torch` tries to load
# the NPU backend extension automatically and raises RuntimeError when the device
# stack is unavailable. Two consequences, both bad: the CPU-only equality tests
# report NOT_RUN for a reason that has nothing to do with M3, and every one of them
# walks onto the device path we have frozen.
#
# TORCH_DEVICE_BACKEND_AUTOLOAD=0 keeps torch on CPU and never loads torch_npu,
# which is exactly the posture this run needs: CPU arithmetic, no device, and no
# ambiguity about whether a device was touched. Measured in-container:
# `torch 2.12.0+cpu / torch_npu loaded: False / cuda False`, exit 0.
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

echo "== interpreter (identity first) =="
"$PY" -c "import sys; print('python', sys.version.split()[0], sys.executable)"
echo "versions below are read from dist-info / installed layout, not from a host import"
SITE=$ENV_ROOT/lib/python3.12/site-packages
for pkg in torch z3_solver isl_python; do
  found=$(ls -d "$SITE"/${pkg}-*.dist-info 2>/dev/null | head -1)
  if [ -n "$found" ]; then
    printf '  %-12s %s\n' "$pkg" "$(basename "$found" .dist-info)"
  else
    printf '  %-12s %s\n' "$pkg" "ABSENT from site-packages"
  fi
done
echo

run() {
  name=$1
  shift
  echo "--- $name ---"
  "$@"
  echo "exit=$?"
  echo
}

# (1) The framework import alone. PYTHONPATH only: no device stack is touched, so a
# failure here is a TileFoundry-side problem and nothing else.
run framework_import   "$PY" -c "import tilefoundry, tilefoundry_ascend; print('tilefoundry import ok')"

# (2) NO DEVICE STACK. This line used to import torch_npu and probe CANN; it no
# longer does, and the reason is a safety rule rather than a simplification.
#
# A host-wide filesystem wedge was traced (correlation, not proof) to a davinci0
# open, and importing torch_npu or sourcing CANN puts this process on that path.
# Nothing in this module needs an NPU: every check is CPU arithmetic plus IR
# inspection. So the device stack is now ABSENT BY CONSTRUCTION rather than
# attempted-and-reported, and no line here may: import torch_npu, run npu-smi, or
# open /dev/davinci*.
#
# The consequence for the report has to be stated plainly: a green run of this
# script is CPU-only evidence and says NOTHING about the NPU. It was already true;
# now it is also impossible to misread, because there is no device line to mistake
# for a device result.
# If a future change ever adds a device line here, it must come through the gate
# first. The gate refuses unless authorised AND handed pre-checked occupancy
# evidence, because a probe expected to fail at aclInit still opens the node. This
# script has no device line today, so the gate is sourced and never called -- but
# it is sourced so the rule is in force the moment someone adds one.
. "$(dirname "$0")/device_precondition.sh"
echo "--- device_stack: NOT TOUCHED BY DESIGN ---"
echo "NPU EVIDENCE: NONE. This script does not import torch_npu, source CANN, or open"
echo "              any /dev/davinci* node. Every line below is CPU arithmetic and IR"
echo "              inspection. A green run here is not device evidence in either"
echo "              direction and must not be reported as the NPU agreeing with"
echo "              anything."
echo


# (3) The M3 checks. These run regardless of (2), and each carries its own code.
run ownership          "$PY" "$REPO/targets/ascend/tests/test_execution_ownership.py"
run vadd_equality      "$PY" "$REPO/targets/ascend/tests/test_vadd_equality.py"
run pair_negatives     "$PY" "$REPO/targets/ascend/tests/test_execution_pair_negatives.py"
run scope_verify       "$PY" "$REPO/targets/ascend/tests/test_execution_scope_verify.py"
run copy_slice_equality "$PY" "$REPO/targets/ascend/tests/test_copy_slice_equality.py"

# (4) The legacy paths that must not regress, so a green M3 cannot come from
# breaking CUDA or the copy lowering.
# The CUDA regression is an ANALYZE of a real in-repo fixture, not a test file.
#
# An earlier version of this line pointed at targets/cuda/tests/test_copy_equality.py,
# a path that has never existed in this repository -- there is no targets/cuda at all,
# the NVIDIA target lives at targets/nvidia. The line therefore failed with ENOENT and
# I had recorded that as a "legacy regression" without ever having run it. Naming a
# file is not running it; that is exactly the sort of line that makes a green report
# a lie, so the path is replaced with one that exists and the check is an analyze,
# which is what the fixture supports.
#
# What it proves: a legacy CUDA program using RMEM with a thread/CTA mesh and NO
# execution mode still analyzes. That is the case the M3 ownership rule could most
# easily have broken -- RMEM is private only in a program that has a mode -- so this
# is the regression that matters, not a generic smoke test.
run cuda_rmem_analyze  "$PY" -m tilefoundry.cli analyze "$REPO/tests/nvidia/cuda/fixtures/placed/square_cuda.py:Model.main"

echo "== done: read exit codes above, not this line =="
