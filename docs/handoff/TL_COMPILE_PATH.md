# TL-COMPILE-PATH（task-35）— 本部署 TileLang 是否真有 Ascend 编译后端

基线 HEAD=f341e47。对象 wheel：site-packages/tilelang（Version 0.1.15+cpu.git0e687c17，dist-info METADATA）。以下 P=<...>/site-packages/tilelang。全部只读：未 compile kernel、未 import torch_npu、未触设备。

## 1) target="ascend" 编译路径是否真实存在

**存在，且已编进本 build**（实测，非文档推断）：
- Target kind 可用：tvm.target.Target({"kind":"ascend"}) 构造成功，attrs 含 max_vector_bits=64、supports_vector_predicate=True、launch_grid_only=True、supports_kernel_status_return=False。
- 四个 device codegen 全局函数**全部注册**：target.build.tilelang_ascend / ..._ascend_without_compile / ..._pto / ..._pto_without_compile（ffi.get_global_func 实测命中；实现于 ascend/codegen.py:8-18，字符串亦见于 lib/libtilelang.so）。
- 回调已注册：ascend/backend.py:55-66 BackendModule(name="ascend", pipelines=ascend_pipeline, device_codegens=ASCEND_CODEGEN, execution_backends=ASCEND_EXECUTION_BACKENDS, callbacks={"tilelang_callback_ascend_compile": ...})；ffi.get_global_func("tilelang_callback_ascend_compile") 实测成功。
- 分派方式：按 target.kind.name=="ascend" 选后端（jit/adapter/utils.py:103-104；backend/module.py:189 BackendContext；plain AscendC 与 PTO 用互斥谓词区分 target_is_plain_ascend/target_is_pto，ascend/target.py:28-41）。
- 编译是**外部工具链**而非纯 Python：codegen 产 AscendC 源文本，之后的 aibin 由 bisheng 子进程完成——ascend/backend.py:15-49 tilelang_callback_ascend_compile → bisheng.compile_ascend；jit/adapter/libgen.py:142-165 的 ascend 分支同样 shell 出 find_bisheng_path()；产物经 AscendBinaryCache（cache/ascend_binary_cache.py: binary_format="aibin"，cache_root="ascend-binaries"）。
- **本机工具链实际存在，但当前不可发现**：bisheng.find_bisheng_path()（contrib/bisheng.py:34-46）只查 BISHENG_HOME/bin/bisheng 与 PATH——实测 RAISES RuntimeError（两变量均未设，bisheng 不在 PATH）。但 /usr/local/Ascend/cann-9.1.0/bin/bisheng 是指向 tools/bisheng_compiler/bin/bisheng 的软链，实体为 150MB x86-64 ELF、-rwxr-xr-x（file 实测）；ccec 同目录存在；/usr/local/Ascend/ascend-toolkit/set_env.sh 存在；version.info=cann 25.7.rc1。→ **缺的是一行 env（BISHENG_HOME 或 source set_env.sh），不是缺编译器。**

## 2) 包名里 +cpu 的含义

- 该 local version tag 对应的是 **torch 侧运行环境**：site-packages 里是 torch-2.12.0+cpu.dist-info（同时装有 torch_npu-2.12.0）。tilelang 代码中**没有任何按 +cpu 分派的 capability 判定或 raise**（grep env.py/__init__.py/backend 无命中）；ascend codegen 与回调均已注册（见 §1 实测）。pipeline.py:22-34 的注释给出内部约定："TL_ENABLE_AUTO_SCHEDULE only exists when the Ascend backend is compiled in"——该键在本 build 生效，反证 **Ascend 后端确已编入**。
- 唯一显式可用性判定是 ascend/target.py:43-49 check_ascend_availability()：它只测 torch.npu.is_available()，是**设备**探针、不是编译能力探针；本环境实测 False（且调用后 torch_npu 未进入 sys.modules）。关键：显式传 target 时**不经过**该探测——backend/target.py:122-134，只有 target=="auto" 才走 auto_detect_target（:66-78，detector 用 _detect_ascend_target，ascend/target.py:52-55，无设备则返回 None）。→ **"ascend 编译被 +cpu 阉割"不成立；只有"auto 目标检测"和"运行时上机"受设备影响。**
- 真编译 CCE 的门槛是 §1 的 bisheng 发现，不是版本 tag。

## 3) 若 compile，最小可行调用与产物；能否不占设备出源码

- 签名：tilelang.compile(func, out_idx=None, execution_backend=None, target=None, target_host=None, verbose=None, pass_configs=None, compile_flags=None)（jit/__init__.py:94-103）；func 必须是 tvm.tirx.PrimFunc（:140 assert）。target 接受字符串 "ascend"/"asc"/"pto"（normalize_ascend_target/asc/pto，ascend/target.py:58-88）。pass_configs 为 dict（键见 transform/pass_config.py），函数级 attrs tilelang_out_idx/tilelang_pass_configs/tilelang_compile_flags 会被合并、外部 pass_configs 覆盖（jit/__init__.py:142-160）。out_idx 与 T.empty 冲突时 raise（:144-148）。arch：target.attrs[arch|mcpu] 优先，否则 ASCEND_NPU_ARCH，默认 **dav-3510**（bisheng.py:91-102）。
- **不占设备的源码级产物（现成 API，今天就能用）**：engine/lower.py:144-213 lower_with_context(func, context, enable_host_codegen=False, enable_device_compile=False) → :190 走 device_codegen_without_compile（:104-106）→ :191 kernel_source = codegen_mod.inspect_source()，返回 CompiledArtifact(host_mod, device_mod, params, kernel_source, target, target_host)。即**只跑 codegen 出 AscendC 文本，不 shell bisheng、不碰设备**。backend/device_codegen.py:37-46 的 lower(compile_device=False) 就是这条分支。
- 产物类型：源码阶段=AscendC 文本（.asc）；真编译阶段=aibin ELF 字节（AscendBinaryCache）；运行时对象=JITKernel（execution backends：tvm_ffi 优先 enable_host_codegen/enable_device_compile=True，cython 次之，ascend/execution_backend.py:8-25）。注意 cython adapter 会把 ascend 映射到 torch.device("npu")（jit/adapter/cython/adapter.py:415-416）→ **任何执行/绑定都必须有设备**；host 侧 ABI 由 supports_callee_allocated_outputs 单一门控（execution_backend.py:15-22 注释）。

## 4) auto_schedule=False 的真实 pass 顺序（本 build）与无设备 hook

- 开关：TL_ENABLE_AUTO_SCHEDULE，默认 True（ascend/pipeline.py:22-34；关闭后注释明确"kernel 保持 unscheduled"）。
- AscendPassPipelineBody（ascend/pipeline.py:36-187）主线：BindTarget :37 → MaterializeKernelLaunch :45-51（lower_grid_binding=True、lower_thread_binding=False、launch_dim_tags=["cthread"]）→ AddWrapperForSingleBufStore :58 → LegalizeNegativeIndex :59 → InjectAssumes :62 → RewriteDualCopy :68 → LegalizeSimdMerging :72 → Simplify/ReducerPlan/VerifyBufferInit :73-76 → UnrollLoopSkipVF :80 → AscendLayoutInference :83 → NormalizeAscendFractalStorage :86 → **InsertNd2Nz :91** → VFChecker :95 → AscendInsertOOBPadding :101 → [auto 时 :103-116 NormalizeControlFlow..RestoreWhileLoops，关闭则整段跳过] → Simplify :118 → NormalizeBufferVersion :120 → AscendSimdVFLowerParallel :121 → AscendLowerTileOp :122 → DecoupleTypeCast..FlattenBuffer :125-138 → RewriteAscendBufferVersionLayout :135 → VectorizeLoop :140 → RewriteFp4ToFp4x2 :146 → UnrollLoop :149 → VerifyMemory/InferFragment/LowerThreadAllreduce :157-160 → (global sync 可选 :162-163) → SplitHostDevice :166 → **not auto 时 InferBufferAliases :169-170** → **MergeUBAllocations(align_bytes=_MERGE_UB_ALIGNMENT, disable_reuse=TL_DISABLE_SHARED_MEMORY_REUSE) :171-174** → **RewriteFlagToBuf :179**（8-slot flag 命名空间，超限 knapsack 溢出到 get_buf/rls_buf mutex）→ AscendThreadSync("shared")/("shared.dyn") :181-182 → MakePackedAPI :184 → LowerDeviceKernelLaunch :186。与 TF 侧 tf_target.POST_SCHEDULE_PASSES（5 项）方向一致，但本 build 的实际顺序更长，且 InferBufferAliases 只在 auto 关闭时运行——与既有登记一致。
- 无设备 hook（都不需要跑内核）：tl.enable_dump_ir + tl.dump_ir_path（默认 ./dump_ir/，pass_config.py:338-341）；tl.pass_profile / tl.pass_profile_threshold_ms（:344-347）；tl.ast_print_enable（:249）；instrumentation.compile_pass_instrumentation（instrumentation/session.py:162-186）；**tools/lower_trace** 逐 pass before/after 文本 + codegen 输出（tools/lower_trace/core.py:77-88 把 target.build.tilelang_ascend 归入 _SOURCE_ONLY_CODEGEN_FFIS——官方就把 ascend codegen 输出当"可再编辑源码"处理，:495 消费）。

## 5) 要产出真实 TL artifact 仍缺什么（逐条判定）

1. **bisheng 发现（可修，不是缺失）**：现 env 下 find_bisheng_path() RAISE；工具链实体在 /usr/local/Ascend/cann-9.1.0/bin/bisheng（实测 ELF）。修法=设 BISHENG_HOME 或 source set_env.sh。判定依据：§1。
2. **设备运行/绑定不可用**：check_ascend_availability()=False、torch-2.12.0+cpu；aibin 产出后无法加载执行，bind_invoke_once 类验收必须留设备窗口。判定依据：§1/§3 探针。
3. **TF 侧没有 compile 调用者**：emit_supported_tl 仍 raise（codegen/__init__.py:31-32 TL_EMIT_NOT_IMPLEMENTED），全仓无 tilelang.compile 生产者（archived mla/kernel.py 不在本树）。→ 缺的是 MAIN-B 的 emitter+reader，不是 TL 能力。
4. **UNKNOWN：dav-3510 是否就是 950PR 的正确 arch 串**（默认值来自 bisheng.py:93，与 TF 本地 3510 target record 同名，但未对本硬件验证）——留给 validator，勿写成结论。
5. **UNKNOWN：ascend codegen 对 TF 降出的 IR 是否真能产出合法 AscendC**：本次只验证了 FFI 存在与注册，未运行编译（规则禁止）；源码级 compile 一跑即知，建议作为下一步最小实验（无设备）。

**结论（给后端计划）**：本 build **不是**"没有 Ascend 编译路径"——codegen/pipeline/callback 实测齐备，+cpu 无门禁；今天就能用 lower_with_context(enable_device_compile=False) 在无设备条件下拿真实 AscendC 源码 artifact；aibin 真编译只差一个 BISHENG_HOME/PATH env；上机执行仍不可用。