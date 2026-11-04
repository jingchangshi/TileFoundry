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
#   4. The TF import and the device-stack import are SEPARATE, INDEPENDENT lines.
#      A torch_npu/CANN problem must not be able to hide an M3 result, and an M3
#      result must not be able to look like an NPU-capable run.
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
for pkg in torch torch_npu z3_solver isl_python; do
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

# (2) The device stack. Independent of (1): this line is allowed to fail and that
# must not be read as an M3 failure, because nothing in this module issues a device
# kernel. This host keeps CANN under /data/pri rather than /usr/local, so the path is
# an ordered candidate list and NOT a hardcoded guess: whichever one is sourced is
# printed, and if NONE exists the line says so instead of claiming CANN is absent --
# those are different statements.
CANN_SET_ENV=""
for candidate in \
  "${CANN_HOME:-}/ascend-toolkit/set_env.sh" \
  /data/pri/cann-9.2.0/ascend-toolkit/set_env.sh \
  /usr/local/Ascend/ascend-toolkit/set_env.sh
do
  case "$candidate" in
    /ascend-toolkit/*) continue ;;     # CANN_HOME unset expands to "": skip the empty prefix
  esac
  if [ -f "$candidate" ]; then
    CANN_SET_ENV=$candidate
    break
  fi
done

if [ -n "$CANN_SET_ENV" ]; then
  echo "CANN: sourcing $CANN_SET_ENV"
  # shellcheck disable=SC1090
  . "$CANN_SET_ENV" >/dev/null 2>&1 || echo "note: $CANN_SET_ENV returned nonzero (continuing)"
else
  echo "CANN: NO set_env.sh found in the candidate list. This does NOT mean CANN is"
  echo "      absent from the machine; it means none of the candidates I know is"
  echo "      present at this path. The device-stack line below is therefore expected"
  echo "      to be unavailable for an environment reason, which is distinct from a"
  echo "      TileFoundry reason and must not be read as an M3 result."
fi
run device_stack       "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Whether torch_npu was imported AT ALL, as its own line. A CPU-only pass of this
# module must never be read as evidence about the NPU: nothing in M3 issues a device
# kernel, so this line exists to stop that inference being made by accident.
echo "--- device_stack_reached ---"
"$PY" -c "
import sys
print('torch_npu in sys.modules:', 'torch_npu' in sys.modules)
try:
    import torch_npu
except Exception as error:
    print('torch_npu import:', type(error).__name__, error)
    print('NPU EVIDENCE: NONE -- this run is CPU-only, so it says nothing about the NPU')
else:
    print('torch_npu import: ok')
    print('NPU EVIDENCE: NONE -- the library is importable, but M3 issues no device')
    print('              kernel, so this run is not device evidence in either direction.')
    print('              Nothing here should be read as the NPU agreeing with anything.')
"
echo "exit=$?"
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
