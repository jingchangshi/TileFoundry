# TILELANG-FACILITIES：部署 build 的解析/编译/PassConfig/IR 观察能力（task-12，只读）

- 对象：部署包 /home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang（实测 `tilelang.__version__ = 0.1.15+cpu.git0e687c17`）；参考源码树 /home/s00653124/workspace/tilelang（`git log -1` = 0e687c17「BugFix][Transform] Preserve explicit casts…」，与部署 build 同源）。
- 方法：纯导入/源码阅读 + /tmp 临时文件里构造一个 10 元素 vec_add PrimFunc（未 compile kernel、未碰设备、已删除临时文件）。所有 file:line 以部署包内路径为准（TL=…/site-packages/tilelang）。全程只读，未写仓库文件。

## 1. TL 源码 → PrimFunc 的真实 parse 路径（Q1，实测）

- 本 build **没有旧的 tvm-script 文本解析器路径**：bundled tvm（TL/3rdparty/tvm/python/tvm）**没有 tvm.tir 模块**，TIR 命名空间是 **tvm.tirx**（另有 tvm.s_tir、tvm.script 目录存在；`import tvm.script` 可行）。
- 真实路径：TL 源码文本 → `@T.prim_func` 装饰器（TL/language/eager/builder.py:1760，经 tilelang.language 与 tilelang.ascend.language 再导出）→ `mutate(func)`（TL/language/eager/ast.py:847）内部 `utils.get_ast`（TL/language/eager/utils.py:61，inspect.getsourcelines + AST）→ 生成 ir_gen → `Builder().prim_func(name)` 上下文内执行 ir_gen → `builder.get()` + `_patch_prim_func_attrs` → **tvm.tirx.function.PrimFunc**（builder.py:1034 类声明 `class PrimFunc(Generic[_P,_T], tvm.tirx.PrimFunc)`）。
- 实测（vec_add demo）：返回对象 `<class 'tvm.tirx.function.PrimFunc'>`；可读字段：`params`（[A_handle, B_handle] tvm.tirx.Var）、`buffer_map`（A_handle→Buffer 'A'…）、`attrs`（{"global_symbol": "vec_add"}）、`body`（SBlockRealize）、`orig_func`（源函数）；另有 ir_gen/span。**str()/repr() 直接输出可读的 tirx 方言文本**（首行 `# from tvm.script import tirx as T`，含 T.match_buffer/T.thread_binding/T.launch_thread_idx 等）。
- 关键约束：prim_func 依赖 `inspect.getsourcelines` → **必须来自真实磁盘源文件**；exec/stdin/heredoc 里的 @T.prim_func 会 OSError "could not get source code"（实测）。要检视字符串形式的 TL 源码，需先落盘临时 .py 再 import。
- `T.prim_func` 在 `tilelang.language`（cuda facade）与 `tilelang.ascend.language` 都存在（实测 hasattr=True）。

## 2. tilelang.compile 签名与 pass_configs 消费（Q2）

- 真实签名（TL/jit/__init__.py，`def compile`；经 TL/__init__.py:200 `from .jit import jit, JITKernel, compile, par_compile` 导出）：
  `compile(func: PrimFunc=None, out_idx=None, execution_backend: "auto"|"tvm_ffi"|"cython"|"nvrtc"|"torch"|"cutedsl"|"pto"=None, target=None, target_host=None, verbose=None, pass_configs: dict|None=None, compile_flags=None) -> JITKernel`。
- ascend target 声明：`target="ascend"` → `Target({"kind":"ascend"})`（TL/ascend/target.py `_make_ascend_target`）；PTO 变体 keys=["pto","ascend"]（_make_pto_target）；`target_is_ascend/target_is_pto/target_is_plain_ascend` 判定（TargetIsAscend FFI）。ascend/README.md：「backend defaults to **dav-3510** for Ascend 950」；ASCEND_NPU_ARCH 可选。示例用法（examples/ascend/test_vecadd.py:20）：`tilelang.compile(vector_add(n, mode), target="ascend", out_idx=-1)`。
- pass_configs 消费链（实测代码路径）：compile → JITKernel `self.pass_configs = normalize_pass_configs(pass_configs)`（TL/jit/kernel.py:109；normalize 在 TL/transform/pass_config.py:286-296，接受 PassConfigKey 或原始字符串键、应用 env 默认、发 deprecated 警告）→ 编译时 `tvm.transform.PassContext(opt_level=3, config=pass_configs, instruments=pass_instruments)`（TL/jit/kernel.py:310）→ 各 pass 经 `tilelang.transform.get_pass_context()`（TL/transform/__init__.py:14 = PassContext.current()）读取。
- **双键实测存在于本 build 且被 Ascend pipeline 消费**：
  - `tl.enable_auto_schedule`（PassConfigKey.TL_ENABLE_AUTO_SCHEDULE，TL/transform/pass_config.py「Enable Ascend auto scheduling. **Default: True**」）→ TL/ascend/pipeline.py:30 `allow_autoschedule` 读取（缺省 True），决定 AutoSchedule 大分支（:101-115）与 non-auto 的 InferBufferAliases（:182-184）。
  - `tl.disable_shared_memory_reuse`（TL_DISABLE_SHARED_MEMORY_REUSE，「Disable shared memory reuse planning in MergeSharedMemoryAllocations. Default: False」）→ TL/backend/pass_pipeline/pipeline_utils.py:71-74 `should_disable_shared_memory_reuse`（缺省 False）→ TL/ascend/pipeline.py:54 读取 → :185-187 `MergeUBAllocations(align_bytes=32, disable_reuse=…)`。即交接合同的 `{tl.enable_auto_schedule: False, tl.disable_shared_memory_reuse: True}` 两键本 build 都接受且都有真实效果点。
- 其它相关键（均在 pass_config.py）：tl.enable_dump_ir / tl.dump_ir_path、tl.ast_print_enable、tl.pass_profile(+threshold)、tl.layout_visualization_enable/formats、tl.Simplify 子配置、tl.disable_prelower_semantic_check 等。

## 3. Ascend pass pipeline 实际顺序（Q3，TL/ascend/pipeline.py，共 190 行）

- 注册：`ascend_pipeline = PassPipeline("ascend", AscendPassPipelineBody)`（pipeline.py:190）；由 TL/ascend/backend.py:60,73 装配（pipelines={"ascend": …}）。
- 主体顺序（真实名称）：BindTarget → MaterializeKernelLaunch（lower_grid_binding=True, launch_dim_tags=["cthread"]）→ [LetInline 仅当 force_let_inline] → AddWrapperForSingleBufStore → LegalizeNegativeIndex → [VerifyParallelLoop 仅当 race check] → InjectAssumes → **RewriteDualCopy** → **LegalizeSimdMerging** → Simplify → CanonicalizeLegacyReducer → VerifyReducerEpoch → VerifyBufferInit → UnrollLoopSkipVF → Simplify → AscendLayoutInference → ReducerPlanAndMaterialize → (LayoutVisual) → NormalizeAscendFractalStorage → **InsertNd2Nz** → VFChecker → AscendInsertOOBPadding →
- **AutoSchedule 分支（仅当 tl.enable_auto_schedule≠False）**：NormalizeControlFlowForSchedule → NormalizeConflictHints → MaterializeScheduleUnits → AnnotateMultiBufferEligible → EstimateLatency → AutoSchedule → AssignCore → PrepareMultiBuffer → ResolveCore → **InsertSync** → MaterializeMultiBuffer → LowerScheduledTIR → RestoreWhileLoops；关闭时改跑 InferBufferAliases（:182-184）。
- 后段：Simplify → NormalizeBufferVersion → AscendSimdVFLowerParallel → AscendLowerTileOp → VerifyReducerConsumed → DecoupleTypeCast → LegalizeVectorizedLoop → LegalizeSafeMemoryAccess → LowerAccessPtr → Simplify → HoistNonRestrictParams → HoistGlobalBufferAllocations → LowerOpaqueBlock → Simplify → RewriteAscendBufferVersionLayout → tirx NarrowDataType(32) → FlattenBuffer → ConfigIndexBitwidth → tirx.Simplify → VectorizeLoop(allow_vectorize) → RewriteFp4ToFp4x2 → LoopUnswitching → UnrollLoop → s_tir RenormalizeSplitPattern → tirx.Simplify → AscendRemoveNoOp → s_tir HoistIfThenElse → tirx.Simplify → AscendRemoveNoOp → tirx VerifyMemory → AnnotateEntryFunc → s_tir InferFragment → LowerThreadAllreduce → [AscendThreadSync("global") 仅当 allow_global_thread_synchronization] → AnnotateDeviceRegions → MarkScalarDcacheBypass → SplitHostDevice → AnnotateReadOnlyParams → [non-auto: InferBufferAliases] → **MergeUBAllocations(32, disable_reuse)** → **RewriteFlagToBuf**（hard_event→8-slot flag namespace，注释 :188-189）→ **AscendThreadSync("shared")** → **AscendThreadSync("shared.dyn")** → MergeIfStmt → MakePackedAPI → Simplify → LowerDeviceKernelLaunch。
- 注意：任务描述里的 "InsertSync" 在本 build 只存在于 AutoSchedule 分支内；AscendThreadSync 有三种 scope（"global"/"shared"/"shared.dyn"）。

## 4. 可用 tile ops 与 Ascend kernel 骨架（Q4，实测 dir() 枚举）

- Ascend 方言 `tilelang.ascend.language as T`（= language/common 全量 + Ascend 扩展，TL/ascend/language/__init__.py）：
  - 帧/骨架：`Kernel`（1-D NPU core grid，无 threads=）、`MixedKernel`、`Cube/CubeFrame`、`Vector/VectorFrame`、`SimdVF/SimdVFFrame`、`SimtVF/SimtVFFrame`、`get_thread_binding(s)/get_thread_extent(s)`。
  - 分配：`alloc_shared/alloc_l1/alloc_l0a(_sf)/alloc_l0b(_sf)/alloc_l0c`（shadow common 的 alloc_shared/alloc_fragment/alloc_global/alloc_local/alloc_buffer/alloc_reducer）。
  - 拷贝/DMA：`copy`、`dual_copy`、`async_copy`、`access_ptr`；T.dma.*（含 ascend_nd2nz_post_copy）；T.mode.*。
  - 同步（T.sync.*，sync.py `*` 导出）：ascend_set_flag/wait_flag、ascend_cross_core_set_flag/wait_flag、ascend_get_buf/ascend_rls_buf、ascend_pipe_barrier、ascend_threadfence、ascend_sync_inter_arrive/wait、ascend_set_copy_pad_value。
  - 计算：`gemm/gemm_blockscaled/gemm_sp`、`fill`、reduce 族（sum/max/min/abssum/absmax/bitand/bitor/bitxor）、math 全套（exp/sqrt/…）、atomic_add 族、cumsum/cummax、if_then_else/where。
  - 调度/注解：`PerCoreTask/Stage/Task`、`assume_conflict/assume_no_conflict`、`annotate_buffer_versions/annotate_manual_multi_buffer/annotate_unlimit_memory/annotate_pass_configs/annotate_compile_flags`、`unroll`（pragma_unroll_factor）、Ascend*TileScheduler 五个、`T.simd.*`（pset/vld/vadd…）、`T.Pipelined`、`T.Serial/T.Parallel/T.Unroll`。
  - 调试：`print`、`device_assert`、`rng_init/rng_rand/rng_rand_float`。CUDA 专属面（tma_*/wgmma*/tcgen05*/cluster*）**不在** ascend 方言。
- 真实 Ascend kernel 骨架示例（参考树，只读引用）：examples/ascend/example_vecadd.py（`@T.prim_func` + `T.Kernel(num_blocks)` + `T.alloc_shared` + `T.annotate_buffer_versions` + `T.Pipelined(iterations, num_stages)` + `T.copy` + `T.SimtVF(threads=…)`/`T.SimdVF()` + `T.simd.vld/vadd`）；同目录 test_gemm.py/test_rmsnorm.py/deepgemm/ 等可作更多骨架。测试式调用：`tilelang.compile(func, target="ascend", out_idx=-1)`。
- 与交接仓库的关系：本仓库 builders（targets/ascend/tilefoundry_ascend/attnres/build.py:13-25）不使用 TL 语言、不调 tilelang.compile，而是直接构造 tilefoundry IR；TL 面向的是后续 emit_supported_tl/read_actual_tl 读回 TL 源的场景（§1 的 tirx parse + §2 的 PassConfig 即该场景的工具）。

## 5. IR 观察手段（Q5）

- **PrimFunc 打印（已实测可用）**：`str()/repr()` 输出 tirx 方言全文（含 match_buffer/thread_binding/body 结构），无需编译。
- **逐 pass IR dump（需真实 compile 时生效）**：pass_configs `tl.enable_dump_ir=True` + `tl.dump_ir_path`（默认 ./dump_ir/）→ TL/jit/kernel.py:299-301 与 TL/autotuner/grouped_compile.py:49-51 注入 `tvm.ir.instrument.DumpIR(dump_dir=…)`。会落盘——只能在允许产物的 compile 会话用。
- **AST 打印**：`tl.ast_print_enable`（pass_config.py）→ TL/analysis/ast_printer.py。
- **per-pass 计时**：`tl.pass_profile`(+tl.pass_profile_threshold_ms) → jit/kernel.py:243 create_pass_timing_tool。
- **布局可视化**：`tl.layout_visualization_enable` + `tl.layout_visualization_formats`（pdf/png/svg）→ pipeline_utils.py:105 LayoutVisual → TL/analysis/layout_visual.py；ascend pipeline 在 ReducerPlanAndMaterialize 后调用（pipeline.py 内）。
- **静态检查器**：TL/ascend/analysis/vf_checker.py VFChecker（pipeline 内 ：66 调用）；TL/analysis/{fragment_loop_checker, nested_loop_checker, parallel_local_index_checker}.py。
- **jit 诊断日志**：TL/jit/diagnostics.py `jit_phase`（verbose 或 env 开关）。
- **边界/UNKNOWN**：未发现独立的“resource/footprint 读取”公开 API（UB/L1 资源在 MergeUBAllocations/LowerTileOp 内部处理，本任务未深入 C++/FFI 层）；PassContext instruments 之外的 IR 快照导出（不 compile 即逐 pass 观察）未找到；tvm.script 可 import 但 tirx 之外的老 tvm.tir 路径不存在，旧文档中的 tvm.tir API 不适用于本 build；设备侧观察（profiler/do_bench）本任务未验证（无设备授权）。NPU binary 缓存见 TL/cache/ascend_binary_cache.py（未实测）。

## 复核命令（只读）

PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 tfnpu-v2/bin/python -c "import tilelang; print(tilelang.__version__)"；grep 上述 file:line；真实源文件 + @T.prim_func 构造 PrimFunc 后 str() 查看。全部结论均可在不 compile、不触设备的情况下复核。
