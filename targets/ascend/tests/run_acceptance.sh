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
PY=$HOME/.conda/envs/tilefoundry-npu/bin/python3.12
REPO=${REPO:-/home/s00653124/workspace/TileFoundry}
export PYTHONPATH=$REPO/src:$REPO/targets/ascend
export PYTHONDONTWRITEBYTECODE=1

echo "== interpreter (identity first) =="
"$PY" -c "import sys; print('python', sys.version.split()[0], sys.executable)"
echo "versions below are read from dist-info / installed layout, not from a host import"
SITE=$HOME/.conda/envs/tilefoundry-npu/lib/python3.12/site-packages
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
run cuda_analyze       "$PY" "$REPO/targets/cuda/tests/test_copy_equality.py"

echo "== done: read exit codes above, not this line =="
