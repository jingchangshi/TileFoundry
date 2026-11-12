# AIBIN-PROBE（task-39）— 设 BISHENG_HOME 后的真编译可行性（不占设备）

explorer-g1，只读 + 只跑到编译层。基线 HEAD=f341e47。全程 **未 import torch_npu**（用 sys.meta_path 拦截器强制证明）、**未打开任何 /dev 节点**、**未 launch kernel**。探针脚本在 /tmp/tf_fac/probe_aibin_*.py。

## 1) 完整 tilelang.compile：env 设与不设各一次

- 不设（裸 shell，bisheng 不在 PATH、BISHENG_HOME 未设）：`tilelang.compile(vsum, target="ascend", pass_configs=契约两键)` → **exit≠0，0.5s**，异常原文：
  `RuntimeError: Cannot find the bisheng compiler.Please install it and make sure it is in PATH, or set the BISHENG_HOME environment variable.`（contrib/bisheng.py:44-46；此前 TileLang INFO 已打印 "begins to compile kernel vsum"，即失败点在 device compile，不在前端）。
- 设：`source /data/pri/cann-9.2.0/ascend-toolkit/set_env.sh`（exit 0；注意它 **不设 BISHENG_HOME**，而是把 bisheng 放进 PATH → tilelang 走 shutil.which 分支）→ **exit 0，wall≈2-3s**，`COMPILE OK 2.1s | kernel type: JITKernel`，TileLang INFO "completes to compile kernel vsum"。再跑一次 "compile 0.00s"＝命中缓存。
- **产物确实存在**（两处，均本次 02:37 新落盘）：
  - JIT 缓存 `~/.tilelang/cache/0.1.15_cpu_git0e687c17/linux-x86_64/kernels/<key>/`：device_kernel.asc(1424B, C 源)、**executable.so(20152B, x86-64 ELF 共享对象)**、host_kernel.c(11584B)、manifest.json(各文件 sha256+size)、params.json(ABI 节点表, tvm 0.25.dev0)。
  - Ascend 二进制缓存 `~/.tilelang/cache/0.1.15_cpu_git0e687c17/ascend-binaries/dfce341a…/{kernel.aibin, metadata.json}`：**kernel.aibin = 3128B ELF64 EXEC，Machine=0x1029（CCE 私有 arch），静态链接，未 strip**，sha256 bde7bde2f4b53cc1…69932；metadata.json `{"format":"tilelang.ascend-binary-cache.v1","sha256":…,"size":3128}`（cache/ascend_binary_cache.py）。

## 2) 编译是否隐式需要设备：不需要（有证据）

- 编译后进程内 `torch_npu in sys.modules: False`，且 meta_path 拦截器全程未触发（若后端真试 import torch_npu 会抛 BLOCKED-BY-PROBE）。
- 扫描 /proc/self/maps：加载 81 个 .so，其中 ascend 相关**只有 `libstub_ascendcl.so`**（wheel 自带 stub，tilelang/lib/），**没有**真实的 libascendcl/libdavincit 驱动库。
- 本机确有 /dev/davinci0-6 与 npu-smi，但本探针从未打开（仅 os.path.exists 级别）。bisheng 是纯交叉编译（--npu-arch=dav-3510）。→ **编译只需编译器，不需设备**；需要设备的是加载/执行（JITKernel.run_once / adapter，本次未触）。

## 3) 产物可检视信息（决定 read_actual_tl 之外是否还要 post-compile 观察）

- aibin 自带 section/symbol（readelf/strings 实测）：`.ascend.meta`、`.ascend.meta.vsum_kernel`、`.ascend.meta.elf`、`__CCE_KernelArgSize`、`__CCE_KernelMetaData`、`.ParamSummary_vsum_kernel`、`.ParamInfo_vsum_kernel_0`、`.Kernel_Type_vsum_kernel`、`.AIV_Kernel_Type_vsum_kernel`、`.CrossCoreSync_vsum_kernel`，以及编译时间戳与 clang 15.0.5 版本串。**这些是源码级 parse 拿不到的信息** → 后端 artifact 判据若要覆盖 ABI/arg-size/kernel-type，需要 post-compile 检视（可纯静态，不占设备）。
- JITKernel 侧可直接读：params=[KernelParam(float32,[64]),KernelParam(float32,[64])]、out_idx=[]、execution_backend='tvm_ffi'、target JSON（含 max_vector_bits=64 等 attrs）、kernel_source（Ascend C 文本）。**但 n_regs=None / n_spills=None / n_max_threads=None / resource_usage={} / latency=None** → 寄存器与占用统计在本 build 的 ascend 路径上不可得，别把它当"资源事实"来源。
- Ascend C 文本可数特征（契约那次）：buf_dyn_shmem 9、__ubuf__ 15、simd_inst:: 4、MODE_ZEROING 1，而 asc_create_flag/get_buf/rls_buf/set_flag/wait_flag **全 0**。

## 4) 两个 PassConfig 键在真编译路径上是否被消费

- `{auto_schedule=False, disable_shared_memory_reuse=True}` → Ascend C 1424B（sha 2674297f…）；反向 `{True, False}` → **2150B（sha 95297b70…），源不同 → 两键确实改变后端行为**，未被忽略。
- 只传 `auto_schedule=False`（省掉 reuse 键）与契约结果**字节相同**：说明对这个最小 VF kernel，reuse 键的差异不显现——**不能据此说它没被消费**，只能说该 kernel 观察不到（更大的多 buffer kernel 才可判）。
- **未知键是硬失败**：加 `tl.totally_bogus_key_xyz` → `AttributeError: Invalid config option 'tl.totally_bogus_key_xyz' candidates are: …`（tvm PassConfigManager::Legalize，含全部合法键清单）。→ 双键拼写错误不会被静默吞掉，可作回归断言点。
- 全程日志中无 "unknown/deprecated key" 告警（grep 零命中）。

## 5) 结论：**可行** —— 不占设备即可得到真实 kernel 二进制

设 `source …/cann-9.2.0/ascend-toolkit/set_env.sh`（或等价地把 bisheng 放入 PATH / 设 BISHENG_HOME）后，完整 tilelang.compile 在 2-3s 内产出真实 `kernel.aibin`（CCE ELF）+ host `executable.so` + Ascend C 源 + 带 sha256 的 manifest，全程零 torch_npu、零设备访问。未在任何一步因需要设备而停手。

**给后端 artifact 判据的确切缺项/条件（不是不可行项）**：
1. TF 侧**目前没有 compile 调用点**：codegen/__init__.py 自述 "Neither compiles a kernel nor touches a device"，emit.py 只到 parse + ascend_source_from（:855-890 源码级）。要产 aibin 需新增显式 compile 站点（属 MAIN-B，且必须与 PASS_CONFIGS 两键绑定）。
2. 必须把 `kernel.aibin sha256`、`executable.so sha256`、`toolchain(clang/CANN 版本串)` 纳入 artifact identity——源码级 parse 给不了（见 §3）。
3. 编译环境是**外部依赖**：bisheng/ld.lld 来自 CANN 9.2.0-beta.2（/data/pri），不在 wheel 内；换机/换 env 必须重新验证，且 tilelang 的缓存根随 `0.1.15_cpu_git0e687c17` 版本 tag 分目录。
4. 资源类事实（n_regs/n_spills/occupancy）在 ascend 路径为 None/{} → 别从 JITKernel 读，需另立来源或标 UNKNOWN。
5. UNKNOWN（留待更大 kernel / validator）：本最小 VF kernel 的 Ascend C 里事件槽位计数为 0，而同 kernel 的 aibin 里存在 `.CrossCoreSync_*` section —— 事件/flag 槽位数量应如何在 post-compile 层面对账，尚无可信判据；reuse 键的可见差异也需更大 kernel 验证。
