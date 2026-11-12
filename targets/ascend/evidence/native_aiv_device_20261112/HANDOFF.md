# Device environment handoff — 2026-11-12

Device numerical status remains NOT_RUN. One container run exited139 during/after runtime initialization; exact Python stage was not observable in that version. Do not describe this as a kernel hang or failure. No compiler/launch output or result.json was produced. No retries followed.

Host failure is concretely diagnosed: original own debug plog `~/ascend/log/debug/plog/plog-202413_20261112122236224.log` reports Permission denied opening CANN 9.3.0 `opp/built-in/op_impl/aicpu/kernel/aicpu_hccl.tar.gz` (root:root mode750). uid1071 cannot read; existing own container defaultroot can read. Original own host plog retained verbatim here as host_init_debug_plog.txt. No permissions were changed.

Existing container: own `s00653124-bska`; /dev/davinci0,/dev/davinci1,/dev/davinci_manager,/dev/hisi_hdc; /home,/data shared; driver read-only. Container own plog original path /root/ascend/log/run/plog/plog-835310_20261112043328551.log, retained as container_plog_835310.txt. Last lines show TSD successful, device opened, Engine/StarsEngine initialized and profiling callback registered. Container PID835310 corresponds to hostPID790807. Container UTC04:33 equals host12:33 +08.

Raw stdout.txt is empty; raw stderr.txt contains only torch_npu ownership warning; job bash-71c09b42b7a94098 exited139 in20s. Original command used host flock and outer timeout; the future command must put timeout **inside** container so Docker disconnection cannot leave the worker running. ps confirmed only container init/tail and diagnostic ps remain, no owned worker.

CPU-only import probe succeeded, no set_device or tensor allocation. Raw import_stdout.txt/import_stderr.txt retained. Versions: Python3.12.16; Torch2.12.0+cpu; torch_npu2.12.0; TileLang0.1.15+cpu.git0e687c17; CANN9.3.0 at /data/pri/9.3.0-weekly/cann-9.3.0; driver25.7.rc1 / Ascend950PR. Three devices healthy via npu-smi; other-user resident jobs untouched. Probe job bash-c0758e6dac854a47 exit0.

Runner now prints flushed import/set_device/finalize/compile/alloc/launch/sync phases, enables faulthandler and reports Python exception stage. --imports-only bypasses initialization and kernel. No phase-instrumented device rerun occurred.

Native code support refinement committed f61b2f6: reject non-GMEM/noncontiguous parameters, output_count !=1, GM reads outside two inputs, writes outside unique output and operation type/count census mismatch.13 native CPU tests passed4.10s, including distinct x*z via existing TF HIR evaluate. This does not satisfy requested future full fingerprint/Counter effect fidelity; that improvement is deferred.

All background jobs collected, none running. User will help diagnose environment; agents must not start new init/kernel probes pending feedback. No new environment, dependency install, driver change, reset or sudo.
