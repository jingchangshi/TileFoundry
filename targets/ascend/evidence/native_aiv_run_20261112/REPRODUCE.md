# Native AIV compile reproduction

Run from the TileFoundry integration checkout. The command below is an executable reproduction of the compile-only path; it was added after the recorded inline compile succeeded and has not been rerun in this round. It does not select an NPU context or launch the kernel.

```sh
source /data/pri/9.3.0-weekly/cann-9.3.0/set_env.sh
export PYTHONPATH=src:targets/ascend
flock -w 5 /home/.tf_device0.lock timeout --signal=TERM --kill-after=5s 120s ../envs/tfnpu-v2/bin/python targets/ascend/tests/native_aiv_device.py --compile-only targets/ascend/evidence/native_aiv_repro
```

For device verification, omit `--compile-only` and use a separate output directory **after diagnosing the recorded device initialization failure**. Lead requested no further launch attempts this round. Both `product` (x != z) and `square` require exact equality; a failure exits nonzero.

The retained compile and initialization logs are explicitly transcribed extracts, not raw complete logs. Full tool output was consumed from the recorded jobs; session history export subsequently reported unavailable. The generated TL and CCE are retained byte-for-byte with SHA256 in result.json. No device numerical or performance PASS is claimed.
