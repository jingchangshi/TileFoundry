# CANN 9.2 host validation commands

Executed from the integration TileFoundry checkout, as the current non-root user. These are environment tests, not formal AgentGates receipts. No source/design implementation resumed.

Adapter interpreter initialization/roundtrip (exit0):

```sh
source /data/pri/cann-9.2.0/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0
flock -w 5 /home/.tf_device0.lock timeout --signal=TERM --kill-after=3s 45s ~/.conda/envs/tilefoundry-npu/bin/python -X faulthandler - <<'PY' > targets/ascend/evidence/cann92_runtime_20261112/init_stdout.txt 2> targets/ascend/evidence/cann92_runtime_20261112/init_stderr.txt
import sys,os
print("PHASE before_import",os.getpid(),sys.executable,flush=True)
import torch
import torch_npu
print("PHASE before_set_device",flush=True)
torch.npu.set_device(0)
print("PHASE after_set_device",flush=True)
x=torch.tensor([0.,1.,-2.,3.5],dtype=torch.float32)
print("PHASE before_alloc",flush=True)
y=x.npu()
print("PHASE before_sync",flush=True)
torch.npu.synchronize()
print("PHASE before_readback",flush=True)
z=y.cpu()
assert torch.equal(x,z),(x,z)
print("ROUNDTRIP_PASS",z.tolist(),flush=True)
PY
```

After the first process exited and released the lock, existing native script with development interpreter (exit139):

```sh
source /data/pri/cann-9.2.0/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend
flock -w 5 /home/.tf_device0.lock timeout --signal=TERM --kill-after=3s 120s ../envs/tfnpu-v2/bin/python -X faulthandler targets/ascend/tests/native_aiv_device.py targets/ascend/evidence/cann92_runtime_20261112/native > targets/ascend/evidence/cann92_runtime_20261112/native_stdout.txt 2> targets/ascend/evidence/cann92_runtime_20261112/native_stderr.txt
```

Final stdout proves after_set_device, then before_finalize. The faulthandler stack identifies CPU ORTools allocation solve; compilation and kernel launch were never reached. Do not describe this as an NPU initialization failure, device hang, compiler failure or product/square numerical result. Raw logs are retained unchanged.

## Successful two-process validation

First process: CPU formal emit with torch_npu autoload disabled (exit0):

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend timeout --signal=TERM --kill-after=3s 40s ../envs/tfnpu-v2/bin/python -X faulthandler - <<'PY'
import sys,json
from pathlib import Path
from tilefoundry.cli.source import checked_program
from tilefoundry.analysis import analyze
from tilefoundry.schedule import finalize
from tilefoundry_ascend.codegen.native_aiv import NativeAIVBinding
from tilefoundry_ascend.codegen.x_runtime import emit_for_codegen
assert "torch_npu" not in sys.modules
p=checked_program("targets/ascend/tilefoundry_ascend/attnres/native_square.py",None,command="cpu-runtime-split")
m=finalize(analyze(p.module,p.entry,analysis=("memory",)),authored=p.authored)
a=emit_for_codegen(m.metadata["ascend.p3_plan"],NativeAIVBinding(m))
assert a.source_sha256=="022e44a121c84bcc864f99b7483f56508471506662420bf488f8210109f1a5e9"
assert a.tir_digest=="936c5b3ca6eb44f1a5d3466220446a429024a00c6acf7c98cd0f0de358913da3"
assert "torch_npu" not in sys.modules
out=Path("targets/ascend/evidence/cann92_runtime_20261112")
(out/"split_native_tl.py").write_text(a.source)
print("CPU_FORMAL_EMIT_PASS",a.tir_digest,a.source_sha256,flush=True)
PY
```

Actual CPU stdout/stderr: cpu_emit_stdout.txt/cpu_emit_stderr.txt; cpu_identity.json retains exact identities.

Second process: existing emitted TL only, no TF/ORTools imports (exit0, product/square exact):

```sh
source /data/pri/cann-9.2.0/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0
flock -w 5 /home/.tf_device0.lock timeout --signal=TERM --kill-after=3s 120s ../envs/tfnpu-v2/bin/python -X faulthandler targets/ascend/evidence/cann92_runtime_20261112/runtime_only.py > targets/ascend/evidence/cann92_runtime_20261112/split_stdout.txt 2> targets/ascend/evidence/cann92_runtime_20261112/split_stderr.txt
```

The second process may have reused the existing TileLang cache. A separate fresh compile was therefore run, not another device launch (exit0):

```sh
source /data/pri/cann-9.2.0/ascend-toolkit/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TILELANG_DISABLE_CACHE=1
flock -w 5 /home/.tf_device0.lock timeout --signal=TERM --kill-after=3s 120s ../envs/tfnpu-v2/bin/python -X faulthandler - <<'PY'
from pathlib import Path
import importlib.util,hashlib,sys
import tilelang
root=Path("targets/ascend/evidence/cann92_runtime_20261112")
p=root/"split_native_tl.py"
assert hashlib.sha256(p.read_bytes()).hexdigest()=="022e44a121c84bcc864f99b7483f56508471506662420bf488f8210109f1a5e9"
s=importlib.util.spec_from_file_location("native_fresh92",p)
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)
print("FRESH92 before_compile cache_disabled=1",flush=True)
k=tilelang.compile(m.main,target="ascend",out_idx=None,pass_configs={"tl.enable_auto_schedule":False})
c=k.get_kernel_source()
(root/"split_fresh92.cce").write_text(c)
print("FRESH92 COMPILE_PASS",len(c),hashlib.sha256(c.encode()).hexdigest(),flush=True)
assert not any(n=="tilefoundry" or n.startswith("tilefoundry.") or n=="ortools" or n.startswith("ortools.") for n in sys.modules)
PY
```

Raw fresh_compile_stdout.txt includes TileLang begins/completes compilation; fresh_compile_stderr.txt retained. Both CCE hashes equal97169f6e...b8fe5. All jobs completed and collected. No user process or driver intervention occurred. Only this new evidence directory was modified in this narrowly authorized environment validation; old review and CANN9.3 evidence remain historical. No implementation/design resumed, and the one-process formal wrapper crash remains unresolved.
