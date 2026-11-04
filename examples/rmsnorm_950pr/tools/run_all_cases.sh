#!/bin/bash
# Run all four Twin cases, one process each, into an evidence directory.
set -u
source /data/pri/cann-9.2.0/ascend-toolkit/set_env.sh >/dev/null 2>&1
export PYTHONPATH=/home/s00653124/workspace/tilelang
export LD_LIBRARY_PATH=/home/s00653124/workspace/tilelang/build/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export TILELANG_CACHE_DIR=/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/artifacts/run01/tl_cache_92
PY=/data/miniconda3/envs/l00849257/bin/python
CASES_DIR="$1"
cd /home/s00653124/workspace/TileFoundry/examples/rmsnorm_950pr
run() {
  local cid=$1; shift
  echo "=== $cid START $(date +%T) ==="
  timeout 900 $PY tools/run_case.py --case-id $cid --out "$CASES_DIR/$cid" "$@"
  echo "=== $cid EXIT $? $(date +%T) ==="
}
run prefill_dyn       --shape 8192 4096 --rows 4 --loop-mode dyn       --stages 4 --xversions 4 --yversions 2
run prefill_recompute --shape 8192 4096 --rows 4 --loop-mode recompute --stages 4 --xversions 4 --yversions 2
run decode_dyn        --shape 64 4096   --rows 1 --loop-mode dyn       --stages 2 --xversions 2 --yversions 1
run decode_recompute  --shape 64 4096   --rows 1 --loop-mode recompute --stages 2 --xversions 2 --yversions 1
echo "ALL_TWIN_CASES_DONE"