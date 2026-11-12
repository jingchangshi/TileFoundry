# task-104 / MAIN-G：TileFoundry_native_mainline 正常入口 import 与部署 TileLang hooks 清单

> 只读产出，作者 explorer-g2，扫描快照 2026-11-12T20:05:01+08:00。唯一写入对象为 artifacts/native_v3/research/mainline_import_inventory.md；未编辑生产源码、未编译 kernel、未调用 NPU。
> 本清单是 MAIN-A/B/C 的当前活动文档，CLEAN-DOCS 不随旧 research 清单归档；最终文件名/路径固定为 artifacts/native_v3/research/mainline_import_inventory.md。
> 结论等级：REFERENCE_IMPLEMENTATION / CPU_METADATA_ONLY。DEVICE_VALIDATED = 0；kernel_compiled = false；device_called = false。

## 0. 锚点、口径与命令

- 主线工作树 [TF](../../../TileFoundry_native_mainline/)：/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/TileFoundry_native_mainline。分支 feat/ascend950pr-native-mainline-local，HEAD ef69b4f5fa8f3218bebbf4fb14fde90816723581，扫描时 git status --porcelain 为空、工作树 clean。姊妹 worktree /home/s00653124/workspace/TileFoundry_native_mainline 同 HEAD、同 clean，仅作一致性核对；未作为写入目标。MAIN-A/B 正在并发编辑生产文件，后续验收前须重跑本清单的动态探针。
- 正常入口扫描口径：pyproject.toml:34-35 的 tilefoundry = tilefoundry.cli:main；[schedule.py:30-40](../../../TileFoundry_native_mainline/src/tilefoundry/cli/schedule.py#L30-L40) run_finalize 走 checked_program → analyze → finalize。当前实现/验收依据是 [ACTIVE_CONTRACT.md](../ACTIVE_CONTRACT.md) 与 API 冻结规范；[test_symbolic_entry_v3.py:89-107](../../../TileFoundry_native_mainline/targets/ascend/tests/test_symbolic_entry_v3.py#L89-L107) 与 [minimal_aiv_pipeline.py](../../../TileFoundry_native_mainline/targets/ascend/tests/fixtures/authors/minimal_aiv_pipeline.py#L1-L43) 只是 2026-11-12 扫描快照的历史测试/fixture 证据，不得作为当前 normal-entry 合同。该测试 :90-91 当时受 TASK90_REQUIRE_NORMAL_ENTRY gate 且默认 skip，因此本快照不构成 normal-entry PASS。
- 部署 TileLang [SP](/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/)：/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang；解释器 /home/s00653124/workspace/envs/tfnpu-v2/bin/python（3.12.16）；importlib.metadata 版本 0.1.15+cpu.git0e687c17；参考源码树 /home/s00653124/workspace/tilelang HEAD 0e687c17bb3bd8a92bfeaa2bc8a5bdb37f6b3ca4。
- 当前实现/验收依据：[ACTIVE_CONTRACT.md](../ACTIVE_CONTRACT.md) 与 [INTERFACE_FREEZE_NATIVE_MAINLINE.md](../INTERFACE_FREEZE_NATIVE_MAINLINE.md)。策略溯源：[DECISION_NATIVE_MAINLINE_ONLY.md](../DECISION_NATIVE_MAINLINE_ONLY.md)。历史扫描输入（已归档，仅溯源，不得作为当前实现/验收依据）：[target_facts_inventory.md](../_archive_compat_defensive/research/target_facts_inventory.md)、[emitter_tl_reader_checklist.md](../_archive_compat_defensive/research/emitter_tl_reader_checklist.md)、[runtime_metadata.json](../_archive_compat_defensive/validation/capability/runtime_metadata.json)。
- 已执行命令及退出码：git status/rev-parse/date exit 0；正常入口 fresh-subprocess import 探针 exit 0；旧路径/direct-TL fresh-subprocess import 探针 exit 0；PassManager registry 探针 exit 0；部署 TL API introspection exit 0；PassContext + PrimFunc 字段探针 exit 0；sha256sum 与 git rev-parse HEAD exit 0。所有 Python 探针均设置 PYTHONDONTWRITEBYTECODE=1、TORCH_DEVICE_BACKEND_AUTOLOAD=0，且不调用 tilelang.compile / finalize / 设备。
- 一次初扫的 grep 工具对若干并发写入文件报 Resource temporarily unavailable (os error 11)；相关结论已用 Python AST 静态导入提取和逐文件 read 复核，不把该轮 grep 当作完整证据。动态 import 探针只导入模块，没有执行 checked_program/finalize；函数内 import 由 AST 标注为“运行时可达边”，不等同于 import-time sys.modules 快照。

## 1. 正常入口实际 import 图

### 1.1 动态快照（CPU，fresh subprocess）

导入根集合：tilefoundry、tilefoundry.dsl、tilefoundry.cli、tilefoundry.cli.source、tilefoundry.cli.schedule、tilefoundry.analysis、tilefoundry.schedule、tilefoundry.passes.pass_manager、tilefoundry_ascend、tilefoundry_ascend.passes.pipeline、tilefoundry_ascend.passes.p3_chain。exit 0；errors = {}；tilefoundry 核心模块 272 个；tilefoundry_ascend 模块 27 个；tilelang_loaded = false；torch_npu_loaded = false。按 sys.modules 插入顺序，27 个 Ascend 模块为：

~~~text
tilefoundry_ascend.ir.execution
tilefoundry_ascend.ir.isa
tilefoundry_ascend.ir.ops
tilefoundry_ascend.ir
tilefoundry_ascend.tf_target
tilefoundry_ascend.passes                       # namespace package, __file__ = None
tilefoundry_ascend.passes.pipeline
tilefoundry_ascend.completion.model
tilefoundry_ascend.completion.guards
tilefoundry_ascend.completion.obligations
tilefoundry_ascend.completion.events
tilefoundry_ascend.completion.protocol
tilefoundry_ascend.completion.schedule
tilefoundry_ascend.completion.emit
tilefoundry_ascend.completion.verify
tilefoundry_ascend.completion.plan
tilefoundry_ascend.completion
tilefoundry_ascend.storage.model
tilefoundry_ascend.storage.placement
tilefoundry_ascend.memory
tilefoundry_ascend.storage.plan_placement
tilefoundry_ascend.storage.intervals
tilefoundry_ascend.storage.verify
tilefoundry_ascend.storage.plan_storage
tilefoundry_ascend.storage
tilefoundry_ascend.passes.p3_chain
tilefoundry_ascend
~~~

结论：正常入口当前不 import 任何 tilelang/torch_npu，也不 import codegen、attnres、mixed、mla、text twin、旧 tilefoundry_ascend.cli/pipeline 聚合模块；但它确实在 import-time 拉入 legacy completion trace、storage placement 聚合和 D0/P3 旧 PassManager 注册模块。

### 1.2 关键静态 import 边

| 位置 | 边 | 判定 |
|---|---|---|
| [__init__.py:10](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/__init__.py#L10) | from tilefoundry_ascend import ir | 触发 ir/ops.py 的 op/typeinfer/verify/schedule_eval 注册；作为唯一 Ascend 指令注册表可保留，但不得再承载 v1/v2 兼容语义。 |
| [__init__.py:11](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/__init__.py#L11) | from tilefoundry_ascend.tf_target import Ascend950PRTarget | 触发 [tf_target.py:89](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/tf_target.py#L89) @register_target；registered_targets() 实测 cpu、ascend950pr。目标注册应保留且唯一。 |
| [__init__.py:18](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/__init__.py#L18) | from tilefoundry_ascend.passes import pipeline | 触发 [passes/pipeline.py:729,751,791,842,890,937,960](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/passes/pipeline.py#L729-L960) 共 7 个 D0 @register_pass；必须从正常链移除。 |
| [__init__.py:25](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/__init__.py#L25) | from tilefoundry_ascend.passes import p3_chain | 触发 [p3_chain.py:250](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L250) @register_pass 与 :42-43 的 completion/storage 聚合 import；必须替换为单事务 native 链。 |
| [p3_chain.py:42-43](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L42-L43) | from .. import completion as CP / storage as ST | 拉入 [completion/__init__.py:47-88](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/completion/__init__.py#L47-L88) 的 build_completion_trace / emit_trace_tir / read_trace_from_tir / reemit_into_unrolled，以及 [storage/__init__.py:26-79](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/storage/__init__.py#L26-L79) 的旧 placement/interval/verification。必须从正常链移除。 |
| [p3_chain.py:267,345,362-367,590,612,922-923,1410,1435](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L362-L367) | 函数内 import schedule.ledger、plan.pipeline/resolved_plan/materialize/scheduled_verify/semantic_correspondence、solver.bridge、ir.isa、memory | 运行 finalize 时的静态可达旧边：static materialize、metadata plan、旧 scheduled_verify、旧 ledger projection。动态探针未执行 pass，因此这些模块不在 import-time sys.modules，但仍是正常执行路径。 |
| [cli/source.py:41-44](../../../TileFoundry_native_mainline/src/tilefoundry/cli/source.py#L41-L44) | normalize_pipeline_references / rebind_cloned_pipeline_references | checked_program 的真实入口；当前 [ir/hir/pipeline.py:34-35,120-172,255-261](../../../TileFoundry_native_mainline/src/tilefoundry/ir/hir/pipeline.py#L34-L35) 同时定义 pipeline/1 与 pipeline/2 并保留 legacy_num_stages_mapping；[parser/pattern_nodes.py:5163-5256](../../../TileFoundry_native_mainline/src/tilefoundry/parser/pattern_nodes.py#L5163-L5256) 仍接受 num_stages (legacy) 或 max_stage (v2)。必须删除 v1 分支，而不是保留兼容 shim。 |
| [cli/target.py:95-99](../../../TileFoundry_native_mainline/src/tilefoundry/cli/target.py#L95-L99)、[target/base.py:340-446](../../../TileFoundry_native_mainline/src/tilefoundry/target/base.py#L340-L446) | registry_path / register_target / load_targets | 部署解释器的 share/tilefoundry/registry.toml 实测不存在；当前 ascend950pr 注册来自包 import 装饰器，不是持久化模块注册。load_targets() 仍可回放 tilefoundry.targets entry points，集成时应确认没有旧 provider module 被 replay。 |
| [pass_manager.py:39-62](../../../TileFoundry_native_mainline/src/tilefoundry/passes/pass_manager.py#L39-L62) | _TARGET_PASSES / register_pass / target_passes | 注册是 import 时的装饰副作用，决定是否进入 finalize 的唯一机制。 |

### 1.3 正常入口实际注册的 pass（fresh import 后读取 _TARGET_PASSES）

BEFORE_LOWER：ValidatePipelineDeclarations、NormalizeSchedulePredicates、NormalizePartSelection。
AFTER_LOWER：VerifyPredicateCoverage、VerifyPipelinePreservation、VerifyTirPartSelection、RecordPredicateProvenance、RunAscendAnalysisChain。
全部来自 tilefoundry_ascend.passes.pipeline 与 tilefoundry_ascend.passes.p3_chain，是 D0 manifest/predicate + 旧 P3 链；不是冻结文档的单事务 native 链。

[passes/completion.py:80](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/passes/completion.py#L80) 还有 InjectCompletionPass 的 @register_pass，但 __init__.py 未 import 它，正常入口快照不含该模块；单独显式 import 后动态探针确认它会加入 AFTER_LOWER。它是“未引用但可因任意 import 激活”的历史注册点，必须删除或保证永不 import，不能仅靠“当前没走到”当作安全。

## 2. 必须移出正常 import/注册图的旧路径

判定口径来自 DECISION_NATIVE_MAINLINE_ONLY.md：旧文件可作为历史材料保留，但不 import、不注册、不影响正常链；不为 pipeline/1、legacy completion trace、static materialize、text twin、旧 x_runtime 单键 PassConfig、旧 static micro-case、旧 g33/g40/g41 断言保留兼容分支。

| 类别 | 精确位置 | 当前事实 | 必须动作 |
|---|---|---|---|
| D0 旧 pass 注册 | [passes/pipeline.py:729,751,791,842,890,937,960](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/passes/pipeline.py#L729-L960)；[__init__.py:18](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/__init__.py#L18) | 7 个 @register_pass 在 BEFORE/AFTER_LOWER 生效，依赖 Module.metadata manifest，自述不求解、不生成 offset plan、不物化 buffer versions。 | 从正常 PassManager 注册移除；由单一 native native HIR→UnitTree→NormalizedProblem→solver→ResolvedPlan 事务 pass 取代。 |
| 旧 P3/export gate | [p3_chain.py:250-316,362-367](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L250-L367)、[p3_chain.py:825-855](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L825-L855) | RunAscendAnalysisChain 在 AFTER_LOWER 注册；函数内 import 旧 ledger/materialize/scheduled_verify/semantic_correspondence/solver.bridge；require_export_ready 仅检查 ChainRecord 的 stopped_at / completion_status / storage_status / scheduled_status 四个 metadata 字段，不读实际 entry、artifact identity、PassConfig。 | 删除旧注册与 metadata-only gate；冻结文档要求 freeze→analyze→solve→resolve→completion→storage→apply→verify→install→export 单事务，失败保持原 entry/拒导出。 |
| legacy completion trace / static storage | [completion/__init__.py:85-88](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/completion/__init__.py#L85-L88)、[storage/__init__.py:26-79](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/storage/__init__.py#L26-L79)、[p3_chain.py:42-43](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/passes/p3_chain.py#L42-L43) | 正常 import 快照实际加载 10 个 completion.* 与 7 个 storage.* 模块（另有 tilefoundry_ascend.memory）；导出 CompletionTrace / simulate_trace / emit_trace_tir / read_trace_from_tir / reemit_into_unrolled 和旧 placement/intervals/verify_storage_current。 | 正常入口不得再 import 这两个聚合包；由有限 SymbolicCompletionPlan / dense FREE / PhysicalEventNamespace 与版本后 StoragePlan 取代。 |
| pipeline/1 / v1-v2 兼容 | [ir/hir/pipeline.py:34-35,120-172,255-261](../../../TileFoundry_native_mainline/src/tilefoundry/ir/hir/pipeline.py#L34-L35)、[parser/pattern_nodes.py:5236-5256](../../../TileFoundry_native_mainline/src/tilefoundry/parser/pattern_nodes.py#L5236-L5256) | PIPELINE_SCHEMA_VERSION = "pipeline/1"，PIPELINE_SCHEMA_VERSION_V2 = "pipeline/2"，构造/校验按 schema_version 分支，parser 仍接受 num_stages (legacy)。 | 单一 typed PipelineSpec；删除 pipeline/1 与 legacy_num_stages_mapping；不重解释 num_stages，不保留兼容分支。 |
| 旧 static micro-case / 单键 PassConfig | [codegen/__init__.py:2](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/codegen/__init__.py#L2)、[native_aiv.py:46,54-106,283-284](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/codegen/native_aiv.py#L40-L106)、[x_runtime.py:414-435](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/codegen/x_runtime.py#L414-L435) | codegen 包 star-import x_runtime；NativeAIVArtifact.pass_configs 只有 (tl.enable_auto_schedule, False)；NativeAIVBinding 硬编码 stop=4、slot_count=1、256 f32 三参数并把 TL 文本写成文件/exec_module；compile_tilelang 对单键 dict 做完全相等判断后直接 tilelang.compile。 | 正常链不得 import codegen 旧包；删除单键 profile 与固定微例合同；新 NativeArtifact 固定两键 PassConfig，实际 compile 调用点记录完整配置并读 parsed PrimFunc。 |
| text twin / 诊断 conformance | [codegen/twin.py:325-338](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/codegen/twin.py#L325-L338)、[attnres/twin.py:288-298](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/attnres/twin.py#L288-L298)、[mixed/twin.py:457-467](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/mixed/twin.py#L457-L467)、[attnres/model.py:358-372](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/attnres/model.py#L358-L372) | 默认 fail-closed，显式 diagnostic_only=True 只给 DIAGNOSTIC_MATCHED；注释明确不验证 parsed TL、operands、guards、PassConfig。attnres/model.py 仍把 twin 报告作为历史诊断 findings 参与 evidence。 | 不得进入 read_actual_tl 或 export gate；DIAGNOSTIC_MATCHED 不得提升为 VERIFIED；保留也只能是未引用历史材料。 |
| direct tilelang.compile / 自建 HIR | [mla/kernel.py:39-42,400-409](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/mla/kernel.py#L39-L42)、[mixed/fa_core.py:22-24](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/mixed/fa_core.py#L22-L24)、[x_runtime.py:419-432](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/codegen/x_runtime.py#L419-L432)、[attnres/__init__.py:55,96](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/attnres/__init__.py#L55-L96) | 显式 direct 探针加载 tilelang 与 torch_npu（即使 TORCH_DEVICE_BACKEND_AUTOLOAD=0）；mla/kernel.py 直接 tilelang.compile(func, target="ascend", out_idx=None) 且无 pass_configs；mixed/fa_core.py 直接 import tilelang.ascend.language/layout；attnres 包导出 kernel/twin。 | AR/MLA 必须是 native builder 返回 native Module/For/ExecutionScope；禁止 direct compile 作为实现权威；CPU compile worker 必须保持 tilelang/torch_npu 不加载，runtime worker 才允许隔离加载。 |
| 旧 diagnostic CLI / aggregate pipeline | [__main__.py:1-2](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/__main__.py#L1-L2)、[cli.py:15-19](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/cli.py#L15-L19)、[pipeline.py:6-13](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/pipeline.py#L6-L13) | python -m tilefoundry_ascend.cli 走 compile_case / trip_count_tests / requested_num_stages / loop_mode 并 import schedule.program/problem/solver/sync、lowering.materialize/verify/layout_check；不是 pyproject 注册的 tilefoundry schedule finalize。 | 从正常生产入口移除/降级为未引用历史 CLI；不得让 case/microtests/trip-tests 定义当前合同。 |
| 未引用的 completion pass 注册 | [passes/completion.py:80](../../../TileFoundry_native_mainline/targets/ascend/tilefoundry_ascend/passes/completion.py#L80) | 正常 import 不加载；显式 import 会向 AFTER_LOWER 注册 InjectCompletionPass。 | 删除注册或保证永不 import；不能以“当前没加载”作为兼容层。 |

显式 direct/legacy 探针（与正常探针分离）import 了 tilefoundry_ascend.cli、pipeline、codegen.*、attnres.*、mixed.*、mla.kernel 等根；exit 0、errors = {}，共 61 个 Ascend 模块进入 sys.modules，且 tilelang_loaded = true、torch_npu_loaded = true。该探针没有执行任何 kernel/device 调用，但说明只要把这些包重新挂入正常入口，CPU 侧隔离立即失效。mla 目录没有 __init__.py，是 namespace package；不能依赖“包初始化缺失”来阻止 mla.kernel 被直接导入。

## 3. 部署 TileLang 0e687c17 的真实 hooks

### 3.1 版本与 PassConfig

PassConfigKey 定义在 [SP/transform/pass_config.py:10](/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/transform/pass_config.py#L10)，是 str Enum。

| 键 | 源码 | 默认 | 冻结 profile | 实测 |
|---|---|---|---|---|
| tl.enable_auto_schedule | [pass_config.py:76-77](/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/transform/pass_config.py#L76-L77) | True（Ascend 自动调度开） | False | 新 CPU PassContext(opt_level=3, config=...) 探针 exit 0，解析为 IntImm 0；既有 [archived runtime_metadata.json:3-37](../_archive_compat_defensive/validation/capability/runtime_metadata.json#L3-L37) 也记录 auto_schedule=false（历史证据，仅溯源）。 |
| tl.disable_shared_memory_reuse | [pass_config.py:210-214](/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/transform/pass_config.py#L210-L214) | False（允许 lifetime reuse） | True | 同探针解析为 IntImm 1；既有 capability probe 记录 disable_shared_memory_reuse=true。 |
| tl.disable_thread_storage_sync | [pass_config.py:239-244](/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/transform/pass_config.py#L239-L244) | False | 冻结 profile 未包含，不擅自加入 | 键确实存在；仅说明 AscendThreadSync 可被关闭。 |
| tl.enable_dump_ir / tl.dump_ir_path / tl.pass_profile / tl.pass_profile_threshold_ms | [pass_config.py:338-348](/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/transform/pass_config.py#L338-L348) | False / ./dump_ir/ / False / 0 | 可作为 instrumentation 附加键，但 compile 调用点必须记录完整配置 | 源码存在；[archived emitter_tl_reader_checklist.md:58-64](../_archive_compat_defensive/research/emitter_tl_reader_checklist.md#L58-L64) 已列（历史证据，仅溯源）。 |

既有 capability probe 状态 REFERENCE_IMPLEMENTATION，scope 为 CPU metadata/PassConfig，无 NPU/TF/ORTools；不得写成 DEVICE_VALIDATED。当前主线 native_aiv.py:46 与 x_runtime.py:422 的“只有 tl.enable_auto_schedule:false 且字典完全相等”违反冻结双键合同，必须移除，不应通过向后兼容分支修补。

### 3.2 PrimFunc reader / lowering hooks（真实存在，不发明 API）

| Hook | 签名 / 源码 | 可用事实 |
|---|---|---|
| tilelang.compile | [SP/jit/__init__.py:94-103](/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/jit/__init__.py#L94-L103)：compile(func: PrimFunc = None, out_idx=None, execution_backend=None, target=None, target_host=None, verbose=None, pass_configs: dict[str, Any] or None, compile_flags=None) -> JITKernel | :140 assert isinstance(func, PrimFunc)；:142-154 读 func.attrs 中 tilelang_out_idx / tilelang_pass_configs，外部 pass_configs 覆盖函数级配置。reader 应在 compile 前读真实 parsed PrimFunc 的 params / attrs / buffer_map / body，而不是只读发射文本注释。 |
| tvm.tirx.PrimFunc 对象字段 | 新 CPU 探针 exit 0 | 实例字段实测：params = Array，body = Evaluate，buffer_map = Map，attrs = DictAttrs；类还有 with_attr、script。部署 wheel Python 源码中 grep read_prim_func / parse_prim_func / read_actual_tl / PrimFuncReader 命中 0；不存在 TF 专用“read_actual_tl”API，必须由 TF reader 直接消费 PrimFunc 对象与 lowering 快照。 |
| JITKernel._compile_artifact（内部 hook） | [SP/jit/kernel.py:288-322](/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/jit/kernel.py#L288-L322) | :299-301 tl.enable_dump_ir 时加入 tvm.ir.instrument.DumpIR(dump_dir=...)；:304-307 create_pass_instruments(context=...)；:310 tvm.transform.PassContext(opt_level=3, config=pass_configs, instruments=pass_instruments)；:313-320 调 lower_with_context。 |
| compile_pass_instrumentation | [SP/instrumentation/session.py:162-186](/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/instrumentation/session.py#L162-L186)：(name=None, *, tools=(), include_default_tools=True, reuse_existing=True) | 自定义 PassInstrumentationTool 的官方上下文管理入口；同一 logical compilation 复用 session。create_pass_instruments(context=...) 也实测 callable，签名 (*, context: str or None = None) -> list[object]。 |
| lower_with_context | [SP/engine/lower.py:144-171](/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/engine/lower.py#L144-L171)：(func_or_mod: tirx.PrimFunc or tvm.IRModule, context: BackendContext, *, runtime_only=False, enable_host_codegen=False, enable_device_compile=False) -> CompiledArtifact | :184 调 lower_to_host_device_ir；:190-213 产生 CompiledArtifact(host_mod, device_mod, params, kernel_source, ...) ，可读 lowering 后 IR 与 CCE kernel_source。 |
| lower_to_host_device_ir | [SP/engine/lower.py:184-189](/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/engine/lower.py#L184-L189) | 返回 host_mod、device_mod、params、target、target_host，用于 refinement manifest，不是 parsed PrimFunc 的替代品。 |

建议 manifest 由同一 compile 调用点关联：parsed PrimFunc 快照 → DumpIR 各 pass 间 IR → CompiledArtifact 的 device_mod / kernel_source → 实际 PassConfig / 编译资源。annotation 文本、text twin 的 DIAGNOSTIC_MATCHED、TIR printer 字符串包含或指令数量相等均不能充当第 2 层 conformance。

### 3.3 Ascend refinement 真实 API 与 pipeline 顺序

所有下列 API 均通过部署 wheel import + inspect.signature 实测 callable（exit 0），并由 [SP/ascend/transform/__init__.py](/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/ascend/transform/__init__.py) 的 _ffi_api 返回 tvm pass；不要自行发明 pass 名或参数。

| API | 定义 / 签名 | 部署 pipeline 调用 |
|---|---|---|
| InsertNd2Nz() | [transform/__init__.py:9-11](/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/ascend/transform/__init__.py#L9-L11)，签名 () | [ascend/pipeline.py:88-91](/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/ascend/pipeline.py#L88-L91)，在 layout inference 后、AutoSchedule 前；UB→L1 ND→NZ scatter + post_copy。 |
| InferBufferAliases() | [transform/__init__.py:167-173](/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/ascend/transform/__init__.py#L167-L173)，签名 () | [ascend/pipeline.py:169-170](/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/ascend/pipeline.py#L169-L170)，仅 not auto_schedule_enabled 时执行；为 manual schedule 生成 MergeUBAllocations 所需的 pairwise reuse contract。 |
| MergeUBAllocations(align_bytes: int = 16, disable_reuse: bool = False) | [transform/__init__.py:176-189](/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/ascend/transform/__init__.py#L176-L189) | [ascend/pipeline.py:19,54,171-174](/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/ascend/pipeline.py#L19)，_MERGE_UB_ALIGNMENT=32，disable_reuse 由 should_disable_shared_memory_reuse(pass_ctx) 得出。即使 disable_reuse=True 仍合成 arena，TF 合法 alias 必须表达为同一 arena 的明确 views。 |
| RewriteFlagToBuf() | [transform/__init__.py:209-222](/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/ascend/transform/__init__.py#L209-L222)，签名 () | [ascend/pipeline.py:176-179](/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/ascend/pipeline.py#L176-L179)；每个 hard_event 8-slot flag namespace，超过 8 会 0/1 knapsack spill 到 32-slot get_buf/rls_buf mutex pool。首版 TF reader 必须把未登记 spill 判为拒绝/有限 replan。 |
| AscendThreadSync(storage_scope: str) | [transform/__init__.py:137-139](/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/ascend/transform/__init__.py#L137-L139) | [ascend/pipeline.py:162-163,181-182](/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/ascend/pipeline.py#L162-L163)：条件调用 "global"，随后固定 "shared"、"shared.dyn"；Ascend 上只处理 SIMT_VF block，global scope 原样返回。 |
| RewriteDualCopy() / AscendInsertOOBPadding() / RewriteAscendBufferVersionLayout() | [transform/__init__.py:112,122,154](/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/ascend/transform/__init__.py#L112) | [ascend/pipeline.py:68,101,135](/home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/ascend/pipeline.py#L68)；与上述四 pass 共同构成 refinement manifest 的相邻变换，不应在 TF 侧静默忽略。 |

[archived target_facts_inventory.md:121-123,152-160,175-179](../_archive_compat_defensive/research/target_facts_inventory.md#L121-L179) 与 [archived emitter_tl_reader_checklist.md:77-81](../_archive_compat_defensive/research/emitter_tl_reader_checklist.md#L77-L81) 记录三个必须登记的漂移（历史证据，仅溯源，不得作为当前实现依据）：InsertNd2Nz 新增 shared.dyn scratch (rows+1)*cols 且 AutoSchedule 关闭时不会自行插入 V→MTE3 事件；MergeUBAllocations 在 disable_reuse=True 下仍合成 arena；RewriteFlagToBuf 超限会替换为 mutex pool。设备同步等价性仍 UNRESOLVED，不能把源码事实写成 DEVICE_VALIDATED。

## 4. 给 MAIN-A/B/C 的最小验收清单

1. 正常入口 fresh import 探针必须保持 tilelang_loaded=false、torch_npu_loaded=false；sys.modules 中不得出现 tilefoundry_ascend.codegen、attnres、mixed、mla、cli、pipeline、completion.schedule/emit/plan、storage.* 旧聚合。
2. _TARGET_PASSES 中 Ascend950PRTarget 只剩单一 native 事务链；D0 pipeline 的 7 个 pass、旧 RunAscendAnalysisChain、InjectCompletionPass 均不得注册。@register_target 保留且唯一；registry.toml / entry-point replay 不得加载旧 provider。
3. src/tilefoundry/ir/hir/pipeline.py 与 parser/pattern_nodes.py 中不再有 pipeline/1、num_stages、legacy_num_stages_mapping 或 v1/v2 schema 分支；单一 PipelineSpec 字段与 INTERFACE_FREEZE_NATIVE_MAINLINE.md 一致。
4. require_export_ready 读当前 actual entry、同一 plan/completion/storage identity、实际 PassConfig；metadata-only APPLIED / VERIFIED 必须拒绝。
5. NativeArtifact.pass_configs 固定两项 {tl.enable_auto_schedule: False, tl.disable_shared_memory_reuse: True}；真实 tilelang.compile 调用点记录完整配置，不接受单键完全相等判断，不把注释或局部变量当证据。
6. read_actual_tl 使用 parsed PrimFunc 的 params/attrs/buffer_map/body + DumpIR/CompiledArtifact snapshot；禁止 text twin、annotation-only、IR 字符串包含或指令数量相等替代结构对应。
7. AR/MLA native builder 返回 native Module/For/ExecutionScope；mla/kernel.py:409 的 direct tilelang.compile 与 mixed/fa_core.py 的直接 TileLang DSL 不得成为实现权威。
8. CPU compile worker 禁 torch_npu，runtime worker 禁 TF/ORTools；直接 import 旧 direct-TL 模块会加载 torch_npu，这一隔离边界必须用进程级断言验证。
9. test_symbolic_entry_v3.py 的 TASK90_REQUIRE_NORMAL_ENTRY skip gate 必须撤销，normal-entry 测试走 checked_program→analyze→finalize 且不得 skip。
10. 本清单只是 import/API inventory，不关闭 CPU 数值门、原 40 例门或 950PR 设备门；任何 REFERENCE_IMPLEMENTATION 结论不得升格为 DEVICE_VALIDATED。

## 5. 证据哈希（扫描快照 20:05 +08，sha256sum exit 0）

<details>
<summary>主线源码 sha256</summary>

~~~text
4b0e3e9f33ad37b4aa891975c1b66459678d9116cb308f02be7ce855213236a1  targets/ascend/tilefoundry_ascend/__init__.py
d880d1b66ee59798dbaa4e321ce794d7cdb1ed85ab4733be955ce6d99c0963dc  targets/ascend/tilefoundry_ascend/tf_target.py
82ae0ebe06267b01d3b78293fe08ff8e954851caf6e8b8e161fe0779490832e6  targets/ascend/tilefoundry_ascend/ir/__init__.py
fbd58aa18d64a285da031eb496c2459640eacb4b97b8b4cec9dba03d33b3290c  targets/ascend/tilefoundry_ascend/ir/ops.py
3d2372a263821d57f105ede3b8ccec298467fd0fcd913e1ece77a996c5919723  targets/ascend/tilefoundry_ascend/passes/pipeline.py
20f4142f9c6200b2d2fccb9e86ad718deb3dcc9c3e7ddabea00be3f1d9c475bb  targets/ascend/tilefoundry_ascend/passes/p3_chain.py
0bf458553dc30a4ded1f9c279c92a32df28977c1b44e1c98429bd47e827b5912  targets/ascend/tilefoundry_ascend/passes/completion.py
6b4d5c937058bb14d73cfca330b0f9dfc1bdb0d0fc8bc55b4a8af9b6a9a956fc  targets/ascend/tilefoundry_ascend/completion/__init__.py
a684d1591957d174adb82a2e5c076c40b21f8a5305f1d3f72e9f42c074e64eff  targets/ascend/tilefoundry_ascend/storage/__init__.py
2bbe35551d8a13115cddd39aa1ee24617bee67a7f2ce23602c6d19240189f106  targets/ascend/tilefoundry_ascend/plan/__init__.py
c91111962cd441a81a6f5caa2cc53fe69d91d82840c6868ea180fd67fb526c66  targets/ascend/tilefoundry_ascend/pipeline.py
e3fc7f0b01a7e982f1b4fd39a682f4ecc6162ab3cbbc7659a6f624ba06d3e791  targets/ascend/tilefoundry_ascend/cli.py
e6007573092c9821f4b557945a5b36d7467d4ed4fe3fd9750e01b251bb6b94ba  targets/ascend/tilefoundry_ascend/codegen/__init__.py
c36d7d458defa1e70aa80e6428a5d0f695c3bd5c34ea5c0b1c30e6b44da082df  targets/ascend/tilefoundry_ascend/codegen/native_aiv.py
68700809634273dbd2b960a7ee68e1319b96fbde6207f9d96db3887347be7e00  targets/ascend/tilefoundry_ascend/codegen/x_runtime.py
9fd8cafb61e7bfce9fc5f98a6172f79c07d3f8b6306725a680018437f455bad8  targets/ascend/tilefoundry_ascend/codegen/twin.py
86eb23bae899c2ccb6336c4e1914dc150aa8b788072e78eb35873ea53f392b57  targets/ascend/tilefoundry_ascend/attnres/__init__.py
06f2371461419e80d14125f42ca8d2592cd0f898c04bea93d40131dd36a0f29a  targets/ascend/tilefoundry_ascend/attnres/model.py
391863e4a2b5337a849ab93d9f6f2de0e63d124b4fb686f9603268e6ffc12aea  targets/ascend/tilefoundry_ascend/attnres/twin.py
43e9710a3a53b420cd5b06319814b7ab86328f364f14a29cfb71728ff3410c6a  targets/ascend/tilefoundry_ascend/mixed/twin.py
4e364c341d2b1930cdfaae2c3c1c3babdd1aa37dfd09da42b2ae738611d57e83  targets/ascend/tilefoundry_ascend/mixed/fa_core.py
01e4fce7546f8047344295687d567c9e3832633f8709a826499f3636fcc6d5b2  targets/ascend/tilefoundry_ascend/mla/kernel.py
9527f10bb6a521d1994a5b20dba1907fd0d339df195283fe94e4010027157566  src/tilefoundry/cli/source.py
c022d4b8d39b197457ee195b51395a1d4b40543866d6312eac776ac18a65a8f4  src/tilefoundry/cli/schedule.py
31eb357a1e0d9a5ea49a80b2f1d95525166718f5ae446c03a9a2723a67e9325d  src/tilefoundry/cli/target.py
49e9505722923768f4575c625bf88a865f4f5f9ead324bb8c710b64d43441675  src/tilefoundry/passes/pass_manager.py
269299847a31e951b5025cd180d9197c037a6894b6cd60097a71aa18f8bc2cd8  src/tilefoundry/target/base.py
b6279b1d91334d8f3e3a30a18c16a9ecb3d4ef1a6cb3a5877f5a790a21c3a403  src/tilefoundry/ir/hir/pipeline.py
d9baf2afeb68b8726c560a0b13c0d97e7f3464552cfa1716b4c5be267b8481ae  src/tilefoundry/parser/pattern_nodes.py
847e9bf5dfc3546d03d02066f73cade3d827ff2f1e1a6150ce72a5531ebf7da1  src/tilefoundry/analysis/pipeline_reader.py
~~~

</details>

<details>
<summary>部署 TileLang 源码 sha256</summary>

~~~text
559b82c1ee7cbf694181903897c96c4c3af7c62c35c74026ae71cd3f5b6b38c7  /home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/transform/pass_config.py
a4b802595207a9248d114e7b0a200c97b8298387f8ca513e98310c7d95756845  /home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/jit/__init__.py
6514a175f416c36e36591b9c9cc794a10838e46bb9dddc2d2097981190a1aa67  /home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/jit/kernel.py
910652ea66dec54122c77569ff937a9dd6eadc1cb33768471c1e05c31786d4bc  /home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/instrumentation/session.py
9954e8854ee8ccb1d94a822bcb7657bf3ba401a09b2fb2a352274da5f0f11dd9  /home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/engine/lower.py
d099e7a08b792fc04f605c5ac88b47381a1426a42f8f55c578dc81e3bf37e1ab  /home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/ascend/pipeline.py
ffab6e686b6865abc1ac6ad904544d0513b3e9e5568d979a5bfecdfbe1055760  /home/s00653124/workspace/envs/tfnpu-v2/lib/python3.12/site-packages/tilelang/ascend/transform/__init__.py
~~~

</details>

## 6. 限制

- 动态探针只做模块 import，不执行 checked_program、analyze、finalize、tilelang.compile、bind/invoke；函数内 import 是 AST 静态可达边，不是运行时 trace。MAIN-A/B 改完后需用真实 normal-entry 测试重跑 sys.modules 与 PassManager registry 断言。
- 部署 TL hooks 是源码 + CPU metadata introspection，未编译 kernel、未触 NPU、未探测 CANN92 设备；同步等价性、容量、性能、950PR 完成证明仍 UNRESOLVED / NOT_RUN。
- 本清单不修改任何生产文件，也不把 CPU 结构检查、历史单键 PassConfig evidence 或 text twin DIAGNOSTIC_MATCHED 升格为当前验收 PASS。
