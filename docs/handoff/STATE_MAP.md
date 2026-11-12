# STATE-MAP — 交接树原生主链真实状态图（只读调查）

- 作者：explorer-g1（task-1 STATE-MAP）。产出只写本文件；未修改任何源码、未做 git 写操作、未调用 NPU/设备、未编译 kernel。
- **分析基线（报告所有 file:line 均以此为准）：`git rev-parse HEAD` = `f341e47529453bdbc8dd81f2c108d36628fb4b8b`（即 tag handoff-20261112 = `374de77` 之后 2 个并行提交：314e016 "bind AR/MLA outputs as trailing ABI parameters"、f341e47 "identity over actual Op attributes and real Var binders"）。**
- 动态探针（正常入口 import + PassManager registry）与 pytest 均在 HEAD=f341e47 上执行：`PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python`；pytest `-q -p no:cacheprovider targets/ascend/tests_native_mainline` → **24 passed in 0.32s**，git status 仅新增未跟踪 docs/（本报告与并行团队的 docs/mainline_validation/*）。
- 与交接快照（tag handoff-20261112 = 374de77）的差异见 §5；除 §5 标注的 6 个新增/3 个修改文件外，其余事实在 tag 与 HEAD 相同。
- 注：仓库内 docs/*.md（ACTIVE_CONTRACT/CURRENT_STATUS/mainline_import_inventory 等）描述的是**旧工作树** `TileFoundry_native_mainline` 的状态（其行号/模块清单与本树不一致，如旧清单的 completion.schedule/emit/plan、storage.plan_* 等模块在本树不存在），本报告只采信本树实测。

## 1. 正常入口链（实测，HEAD=f341e47）

入口：pyproject.toml:34-35 `tilefoundry = "tilefoundry.cli:main"` → `run_finalize`（src/tilefoundry/cli/schedule.py:30-40）= `checked_program → analyze → finalize`；`finalize`（src/tilefoundry/schedule/finalize.py:12-19）= `PassManager(BEFORE_LOWER passes) → ConvertHIRToTIR → PassManager(AFTER_LOWER passes)`。pass 注册机制：src/tilefoundry/passes/pass_manager.py:39-62（import 时装饰副作用）。

动态探针实测 `registered_targets() = ['ascend950pr','cpu']`，`Ascend950PRTarget`（targets/ascend/tilefoundry_ascend/tf_target.py:89-96 `@register_target`）名下注册：

- BEFORE_LOWER（3 个 D0 pass，全在 targets/ascend/tilefoundry_ascend/passes/pipeline.py）：ValidatePipelineDeclarations(:733)、NormalizeSchedulePredicates(:757)、NormalizePartSelection(:896)
- AFTER_LOWER（4 D0 + 1 主链）：VerifyPredicateCoverage(:797)、VerifyPipelinePreservation(:848)、VerifyTirPartSelection(:943)、RecordPredicateProvenance(:966)、RunAscendNativeMainline（passes/p3_chain.py:215-243）
- 正常入口 import 只拉入 13 个 ascend 模块（tilefoundry_ascend{,.ir*,.tf_target,.passes*,.codegen,.plan*}），`tilelang_loaded=False`、`torch_npu_loaded=False`、`torch 不在 sys.modules`（ir/ops.py:32 经 src/tilefoundry/utils/lazy.py LazyModule 惰性加载，求值时才真 import torch）。CPU import 隔离成立。

**结论：正常入口真实可执行的只有 D0 断言/记录 + 一个必然拒绝的主链事务记录器。冻结主链 API 没有一个被该链调用。**

## 2. 冻结职责 API 逐项判定（13 项，规范 docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md:19-32）

| # | API | 定义位置 | 状态 | 关键证据 | 从正常入口可达？ | 调用者 ← / 被调用 → |
|---|---|---|---|---|---|---|
| 1 | normalize_intent | **MISSING**（全仓库无定义；RegionBindingSummary 类型也不存在，grep 0 命中） | 缺失 | 无 | 否（无调用点） | 无 |
| 2 | build_native_unit_tree | **MISSING**（UnitTree 类型无定义） | 缺失 | 无 | 否 | 无 |
| 3 | analyze_native_problem | **MISSING**（NormalizedProblem 类型无定义） | 缺失 | 无 | 否 | 无 |
| 4 | solve_region | targets/ascend/tilefoundry_ascend/solver/__init__.py:26-27 | fail-closed stub：`raise MainlineNotImplemented("SOLVER_SHARED_REGION_NOT_IMPLEMENTED")`；:15-23 有 frozen 形状的 `Assignment` dataclass（status/starts/stages/phases/versions/issue_order/optimality，默认 UNKNOWN） | 否（无调用点） | 无 ← ；→ raise |
| 5 | resolve_schedule | **MISSING**（ResolvedPlan 类型无定义） | 缺失 | 无 | 否 | 无 |
| 6 | build_symbolic_completion | tilefoundry_ascend/completion/__init__.py:29-30 | fail-closed stub：`raise MainlineNotImplemented("COMPLETION_SYMBOLIC_NOT_IMPLEMENTED")`；:20-26 有占位 `CompletionPlan`（sites/channels/verified/identity 全默认空）；:33-38 verify_completion/require_completion_verified 同为 raise | 否 | 无 ← ；→ raise |
| 7 | place_resolved_versions | tilefoundry_ascend/storage/__init__.py:30-31 | fail-closed stub：`raise MainlineNotImplemented("STORAGE_PLACEMENT_NOT_IMPLEMENTED")`；:22-27 占位 `StoragePlan`；:34-39 verify_storage_current/require_storage_verified 同 raise；:15 `UNKNOWN_CAPACITY` 常量 | 否 | 无 ← ；→ raise |
| 8 | apply_symbolic_pipeline | tilefoundry_ascend/plan/mainline.py:78-79 | fail-closed stub：`raise MainlineNotImplemented("MAINLINE_APPLY_NOT_IMPLEMENTED")`（无 private candidate / source_map 实现） | 否 | 无 ← ；→ raise |
| 9 | verify_actual_program | plan/mainline.py:82-83 | fail-closed stub：`raise MainlineNotImplemented("MAINLINE_VERIFY_NOT_IMPLEMENTED")` | 否 | 无 ← ；→ raise |
| 10 | install_entry | plan/mainline.py:47-75 | **前置检查真实 + 最终 fail-closed**：:54-61 校验 module.entry/函数存在/名字匹配（PlanError PLAN_ENTRY_UNKNOWN/PLAN_ENTRY_MISMATCH），:62-72 拒绝 conformance=None 或 dict（INSTALL_CANDIDATE_UNVERIFIED），:73-75 `raise MainlineNotImplemented("INSTALL_VERIFIER_NOT_IMPLEMENTED")`；模块不可变承诺成立（拒绝路径不触碰 module.functions） | 否 | 无 ← ；→ PlanError/MainlineNotImplemented |
| 11 | emit_supported_tl | tilefoundry_ascend/codegen/__init__.py:31-32 | fail-closed stub：`raise MainlineNotImplemented("TL_EMIT_NOT_IMPLEMENTED")`；:13-14 `PASS_CONFIGS=(("tl.enable_auto_schedule",False),("tl.disable_shared_memory_reuse",True))` 双键合同存在；:17-28 `NativeArtifact` dataclass（source/abi/parsed_prim_func/六种 identity/pass_configs） | 否 | 无 ← ；→ raise |
| 12 | read_actual_tl | **MISSING**（全仓库无定义；conformance 对象无产生者，p3_chain 只能检查 metadata 里的 TL_CONFORMANCE_KEY 占位） | 缺失 | 无 | 否 | 无 |
| 13 | bind_invoke_once | **MISSING**（无定义；tf_ascend_runtime 只有纯 ABI 模型 abi.py，无 runtime loader/worker，无 invoke 路径） | 缺失 | 无 | 否 | 无 |

辅助事实（事务链内部函数，非冻结 13 API）：
- `compile_mainline` plan/mainline.py:86-87 raise MAINLINE_CHAIN_NOT_IMPLEMENTED；**唯一 prod 调用者** p3_chain.py:224（RunAscendNativeMainline.run），:225-226 捕获后 `stopped_at="freeze", reason=...` → :241-242 写 CHAIN_KEY/EXPORT_GATE_KEY。
- `require_export_ready` p3_chain.py:171-212：对 actual entry identity/artifact 身份/双键 PassConfig/parsed_prim_func 的检查逻辑是真实的（:122-168 `_native_artifact_gaps`），但 :199-201 **无条件** 追加 `("tl","NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED")` → `verdict.ready` 恒 False。
- 12 处 `raise MainlineNotImplemented`（plan 4 + solver 1 + completion 3 + storage 3 + codegen 1）在 tag 与 HEAD 上完全一致，未变。

## 3. targets/ascend 模块分类（HEAD=f341e47；tag=374de77 差异见 §5）

类别定义：**真实实现**=可执行逻辑；**类型与事实**=声明容量/契约/模型，无可执行主链逻辑；**fail-closed stub**=具名拒绝；**孤立**=正常入口不可达且无 prod 调用者（可与前述类别叠加）。

| 模块 | 类别 | 提供什么（主要类/函数） | 被谁引用 | active 测试覆盖 |
|---|---|---|---|---|
| tilefoundry_ascend/__init__.py | 真实接线 | import ir/tf_target/pipeline/p3_chain 触发注册 | pytest/用户 import | 间接（全部测试经它） |
| tf_target.py | 真实实现 | Ascend950PRTarget（facts: topology/memory/space_facts/completion_ops/torch_device_type:141-194）、nd2nz_scratch_extent:45-51、DUAL_COPY_PATHS/POST_SCHEDULE_PASSES/THREAD_SYNC_FACTS :56-86（REFERENCE_IMPLEMENTATION/UNRESOLVED 状态字） | package __init__:7、pipeline.py:37、p3_chain.py:20、attnres/mla build | 间接（测试构造 target） |
| ir/ops.py | 真实实现 | 17 个 AIV 指令 Op（GmToUb/UbToGm/Vld/Vsts/Pset/Pset16/Vcvt/Vmula/Vdup/Vadds/Vmuls/Vmul/Vsqrt/Vdiv/Vcadd/Vdupv/Vadd，:231-607）+ 约束类（_VldDist:96 等）+ infer/verify/eval 注册 + 目标候选注册 :1014-1017；torch 为 lazy :32 | ir/__init__、attnres/mla build:27、test_identity | test_identity.py:26（5 用例）；结构未直接测 |
| ir/isa.py | 类型与事实 | InstructionContract/Operand + ISA 指令契约表 :74-229（含 RETRACTED 累加器声明注释 :51-58、NATIVE_BUILD:211、PER_CASE_ACCUMULATOR_CONTRACT:231、reduction_order_for_contract:251） | 仅 ir/__init__.py re-export；**无其他消费者** | 无 |
| ir/execution.py | 真实实现 | SIMD_VF ExecutionKind 注册（:43-65，64 lane/f32/nestable=False） | ir/__init__、attnres/mla build:26 | 间接 |
| completion/__init__.py | fail-closed stub | build_symbolic_completion/verify_completion/require_completion_verified 全 raise；CompletionPlan 占位 | 无 prod 引用 | 无 |
| completion/model.py | 类型与事实 | EventPool/EventNamespace:265-313、HardEventPoolFacts:320、PhysicalEventNamespace:348、TargetEventFacts:381、ASCEND_EVENT_FACTS:487-535（declared bounds：AIV/AIC ready/free 16×2bank、state 8、exit 4、cross 16；hard-event intra 8/cross 16 mode4，bank UNRESOLVED）、event_facts_for:538 | tf_target.py:175（lazy） | 无直接 |
| storage/__init__.py | fail-closed stub | place_resolved_versions/verify_storage_current/require_storage_verified raise；StoragePlan 占位；UNKNOWN_CAPACITY | 无 prod 引用 | 无 |
| storage/model.py | 类型与事实 | align_up/dtype_bytes/bytes_for:57-235（NZ 分形 padding 计算）、StorageFacts/LAYOUT_FACTS:90-129、PhysicalLayout:263、Requirement:284、TargetLayoutFacts.resolve:313-328（L1 必须声明，拒绝默认 linear） | 仅 storage/placement.py:35-47 | **无**（孤立） |
| storage/placement.py | 真实实现（孤立） | Interval/PlacementEntry/SpaceUsage/PlacementReport/StoragePlacement:54-196：完成证据闭区间 + 输入 digest 失效语义 | 仅被 storage.model 间接依赖；**无任何导入者** | **无**（孤立） |
| passes/pipeline.py | 真实实现（含已声明缺口） | 7 个 D0 pass + 类型化规则函数（validate_pipeline_declarations:183、required_active_for_issue:283、normalize_tail_requirement:311、verify_predicate_coverage:451、verify_pipeline_preservation:569 等）；缺口：`_collect_actual_predicates` 恒 `return {}` :689-697、NormalizeSchedulePredicates 恒置 `requirements=()` :781-793（F/L source access window 缺失，注释自述 "stated gap"）；:963 `PREDICATE_PROVENANCE_KEY = "d0.predicate_provenance"` 为泄漏的模板占位符 | package __init__:8 | 无直接（规则函数无 active 测试） |
| passes/p3_chain.py | 真实实现（fail-closed gate） | RunAscendNativeMainline:215-243、require_export_ready:171-212（无条件 NOT_IMPLEMENTED 缺口 :199-201）、ChainRecord:88、ir_identity:61（→plan/identity） | package __init__:9、test_identity.py:28 | test_identity.py:97,104（经 ir_identity） |
| plan/mainline.py | fail-closed stub | install_entry（真前置检查+raise）:47-75、apply/verify/compile_mainline raise :78-87、SINGLE_TRANSACTION:24-27 | solver/completion/storage/codegen/p3_chain/plan __init__ | 无直接 |
| plan/identity.py | 真实实现 | node_identity_payload/function_identity/module_identity:164-193：Op 属性参数 + Var binder 标签的结构 sha256（_SKIP_FIELDS:40-51） | p3_chain.py:18、test_identity.py:27 | test_identity.py（5 用例） |
| codegen/__init__.py | fail-closed stub | emit_supported_tl raise；NativeArtifact+PASS_CONFIGS | p3_chain.py:17 | 无直接 |
| codegen/abi.py | 真实实现 | entry_function:22、abi_from_prim_function:35-72（trailing output_count 契约、设备 storage 校验）、abi_from_module:75 | 仅 tests（test_operator_abi_numeric.py:21、native_reference.py:183）；**无 prod 调用者** | backend_operators/test_operator_abi_numeric.py（7 用例） |
| memory.py | 类型与事实 | SPACE_FACTS 12 个空间（:77-110，含 ascend.ss UNRESOLVED/不可分配）+ TRANSFERS 传输合法性表 :119-141 + transfer_contract/assert_transfer_legal:142-176 | tf_target.py:169（lazy） | 无 |
| sync.py | 真实实现 | AivFlagSet/AivFlagWait/AivDrain :23-32（绑定 ("aiv",) mesh）+ typeinfer/verify/access 注册 :35-57 | tf_target.py:189（lazy，completion_ops） | 无 |
| bf16.py | 真实实现（孤立） | 纯 Python bf16 encode/decode/round/pack/lane_add/mul/scale/vcadd（:35-248，Fraction 精确舍入）；:32 自评 CREDIT="CPU-only software quantizer; full P2 native-ISA claim NOT established" | **无任何导入者** | **无**（孤立） |
| attnres/__init__.py + build.py | 真实实现 | AttnResSpec:66（families/score_modes/dtypes 校验 :80-164）+ build_attnres_module:175-269：native Module/PrimFunction/For/ExecutionScope/MeshScope 骨架 + typed contract metadata（device_validated=False:266）；无 tilelang.compile；post-314e016 ABI 顺序 (x,v,M,N,y) :247-253 | tests（test_native_builders.py:46 经文件加载；test_operator_abi_numeric.py:20 经包） | test_native_builders.py 3 用例 + ABI/数值 3 用例 |
| mla/__init__.py + build.py | 真实实现 | MLASpec:65（GQA/causal/shard/program_count :79-224）+ build_mla_module:233-330（同上骨架）；post-314e016 ABI 顺序 out 在尾 :307-313 | tests（同上两处） | test_native_builders.py 2 用例 + ABI/数值 2 用例 |
| mixed/__init__.py | 空占位 | 仅 1 行 docstring，无实现 | 无 | 无 |
| schedule/__init__.py | 空占位 | 空文件 | 无 | 无 |
| solver/__init__.py | fail-closed stub | solve_region raise + Assignment | 无 prod 引用 | 无 |
| search/__init__.py + process.py + rank.py | 真实实现（孤立） | Deadline/WorkerCommand/ProcessResult/run_process（process.py:14-137，subprocess+signal+uuid）+ CandidateMeasurement/Ranking/rank/top_k/require_device_source/confirmation_required（rank.py:186-251，device-only 排名边界，RANK_SIMULATED_NOT_COMPARABLE 拒绝模拟数据） | search/__init__ + tests；**正常入口不可达** | test_search_rank.py（4）+ test_native_builders.py:171-177 |
| tf_ascend_runtime/__init__.py | 真实接线 | re-export abi 全部符号 | codegen/abi.py:10、tests | 间接 |
| tf_ascend_runtime/abi.py | 真实实现 | ABI_SCHEMA "ascend.native-abi/1":22、ParameterSpec:75、OrderedABI:100-225（output_split/inputs/outputs/digest/from_dict 拒绝外schema/篡改digest）、validate_bindings:237-276（count/shape/dtype/contiguous 拒绝）；零依赖（无 tilefoundry/torch/tilelang） | codegen/abi.py:10、tests | backend_operators 7 用例内覆盖 |
| tests_native_mainline/（4 文件） | active 测试面 | test_identity.py（5）、test_native_builders.py（8）、test_search_rank.py（4）、backend_operators/{test_operator_abi_numeric.py（7）,native_reference.py（独立 numpy 模板求值器 :1-215）} | — | **24 passed**（实测） |

（任务描述所称"active 测试面只有两个文件"已过时：tag=374de77 时为 2 文件 12 用例；HEAD=f341e47 为 4 文件 24 用例。）

## 4. 结构性问题（按影响排序）

1. **主链前半 100% 缺失、后半 100% fail-closed**：13 个冻结 API 中 6 个连定义都没有（normalize_intent/build_native_unit_tree/analyze_native_problem/resolve_schedule/read_actual_tl/bind_invoke_once），5 个核心 schema 类型（RegionBindingSummary/UnitTree/NormalizedProblem/ResolvedPlan/SymbolicCompletionPlan）零定义；其余 7 个全部在第一行 raise（仅 install_entry 有真实前置检查）。事务链在 `RunAscendNativeMainline` 第一步 `compile_mainline` 即 stopped_at="freeze"（p3_chain.py:224-226），export gate 恒含 NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED（p3_chain.py:199-201）。当前树没有任何一条能把 native HIR 变成已安装 entry 的路径。
2. **正常入口 AFTER_LOWER 无调度承载点**：D0 只断言声明/守恒；`NormalizeSchedulePredicates` 恒产出空 requirements（pipeline.py:781-793），`_collect_actual_predicates` 恒空（:689-697），VerifyPredicateCoverage 因此永远空转；`RecordPredicateProvenance` 的 metadata 键是模板占位符字面量 `"d0.predicate_provenance"`（pipeline.py:963，其他键均为 ascend.* 命名空间）——正常入口既不求解也不物化任何调度事实。
3. **已实现的后半链资产与主链零接线（孤立）**：storage/model.py+storage/placement.py（完整的完成闭区间放置引擎，含 digest 失效语义）、bf16.py（纯 Python bf16 参考量化器）、search/*（设备-only 排名边界）、ir/isa.py（指令契约表，无任何消费者）都不在正常入口 import 图中、无 prod 调用者；completion/model.py/memory.py/sync.py 仅被 tf_target 的 lazy hook 引用。MAIN-B 需要的组件存在但没有一个接到 ResolvedPlan（其类型尚不存在）。
4. **文档/现实漂移**：ACTIVE_CONTRACT.md:5、CURRENT_STATUS.md:3、mainline_import_inventory.md 等仍指向已删除的旧工作树与旧模块清单；README_HANDOFF/HANDOFF_PLAN 说 gate "ends in NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED"（正确）但没记录 tag 之后 2 个并行提交已把 ABI/identity 从"纯拒绝"推进到"真实可测实现"；test_native_builders.py:3-5 docstring 声称 "archived attnres package surface is intentionally not restored"，而 attnres/__init__.py 实际存在并导出 .build（测试改经文件加载的理由已失效）。
5. **UNKNOWN 项**：(a) bind_invoke_once 的宿主进程/worker 协议在树内无任何原型（tf_ascend_runtime 只有 ABI 模型）；(b) ChainRecord.pipeline_status/applied 等"实现后"字段将来由谁写入——无实现，无法判定；(c) ir/isa.py 的契约表是否计划被 solver/emitter 消费——文档未说明；(d) 未跟踪的 docs/mainline_validation/run-20261112T210101+0800-task136/（并行团队 task-1xx 编号）内容未纳入本图。

## 5. tag handoff-20261112（=374de77）→ HEAD（f341e47）差异（事实分版本）

`git diff --name-status handoff-20261112..HEAD`（targets/ 下）：

| 文件 | 变化 | 事实差异 |
|---|---|---|
| tilefoundry_ascend/plan/identity.py | **新增（f341e47）** | 结构 identity（Op 属性+Var binder）仅存在于 HEAD；tag 上无此文件 |
| tilefoundry_ascend/codegen/abi.py | **新增（314e016）** | 同上，ABI 派生仅存在于 HEAD |
| tf_ascend_runtime/{__init__,abi}.py | **新增（314e016）** | 纯 ABI 模型仅存在于 HEAD |
| passes/p3_chain.py | 修改 ±38 | tag 上 `_function_identity` 是 name/kind dataclass 摘要（对 Vadds(value=1.0/7.0) 碰撞，test_identity.py:1-8 记录该缺陷）；HEAD 换用 plan/identity 的真实 identity。require_export_ready 的无条件 NOT_IMPLEMENTED 缺口两版都有 |
| attnres/build.py、mla/build.py | 修改 +6/+5 | tag 上 PrimFunction 参数序把输出放中间（output_count 语义错误）；HEAD（314e016）改为输出在尾。两版均为真实 native builder、无 tilelang.compile |
| tests_native_mainline/test_identity.py、backend_operators/* | **新增** | active 测试面由 2 文件 12 用例 → 4 文件 24 用例（实测 24 passed） |

pre/post **不变**的事实：12 处 MainlineNotImplemented、7 个 D0 pass + RunAscendNativeMainline 注册、pipeline.py:689-697/:781-793 声明缺口、`"d0.predicate_provenance"` 占位键（pipeline.py:963 在 tag 上同一内容，位置可能不同）、6 个 MISSING API 与 5 个 MISSING 类型、tf_target/memory/sync/completion.model/isa/storage.model/placement/bf16/search 全部内容。
