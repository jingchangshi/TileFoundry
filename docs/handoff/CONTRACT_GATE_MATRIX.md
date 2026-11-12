# CONTRACT-GATE-MATRIX — 冻结合同验收与实施切分

任务：task-3 / architect-g1。调查日期：2026-11-12（Asia/Shanghai）。只读源码；唯一写入是本报告。未运行 NPU、TL 编译或 Git 写操作，未读取历史 worktree。当前工具核实 cwd 为本交接树，branch main，HEAD f341e47529453bdbc8dd81f2c108d36628fb4b8b。本报告不是任何阶段的 PASS 证明。

## 0. 权威、快照与证据边界

唯一 API 权威是 [冻结合同:4](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md:4>)，签名 a6764633207d175148bcfdbdb6ff2e24eab006d31ac5b8d135f3f90d5752f790。[docs/ACTIVE_CONTRACT.md:5](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/ACTIVE_CONTRACT.md:5>) 中旧活动树路径已被本次用户指定交接路径覆盖；不得据此进入旧树。原工作包规格只读参考；其中 legacy parser/fallback/debug static materialize、Lead 写 schema、先 install 后 TL 核对的旧说明，均让位于 [冻结合同:3](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md:3>)、[冻结合同:13](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md:13>)、[冻结合同:34](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md:34>)。ResolvedPlan 仍由 MAIN-A 唯一写，不能另建 schema。

交接包缺 implementation/、evidence/、plan/：glob 三目录为空，manifest 仍列 evidence（[work_package/PACKAGE_MANIFEST.json:38](</home/s00653124/workspace/TileFoundry_950PR_handoff/work_package/PACKAGE_MANIFEST.json:38>)）。Lead 已报告交接包三个框架测试因 ModuleNotFoundError: implementation 无法收集，本代理未重跑该失败。按任务授权，仅读取原文包 plan 三文件：[原包 TASKS.json:4](</home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3/plan/TASKS.json:4>)、[原包 GATES.json:5](</home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3/plan/GATES.json:5>)、[原包 PHASES.md:7](</home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3/plan/PHASES.md:7>)；这是允许的原始文档副本，不是旧 worktree。算法参考实现的进一步读取/移植标为“原始工作包只读参考（待 Lead 裁定 D1）”，本次未把任何副本装入生产路径。

版本分界已由 Lead 与本代理 Git 只读命令共同核实：handoff-20261112 = 374de776f68d3e26eb80d9a1b7c8f67699708213（pre-314e016、pre-f341e47）；当前 HEAD = f341e47529453bdbc8dd81f2c108d36628fb4b8b（post-f341e47）。并行团队新增 314e016（trailing ABI outputs）与 f341e47（实际 Op attributes / Var binder identity）两个提交。本报告所有当前源码 file:line 均指 post-f341e47，不把它们倒记为交接 tag 已有能力。git diff handoff-20261112 HEAD 已核实差异；本代理未作任何提交。

| 能力/证据 | 交接 tag（pre-f341e47） | 当前 HEAD（post-f341e47） |
|---|---|---|
| 活动测试 | 两模块12项；12 passed 是 Lead 较早实测，本代理未复跑 | 四模块24项；24 passed 是 Lead 协调消息实测；本代理只collect-only，退出0、24 tests collected in 0.27s |
| AR/MLA ABI | y/out 位于 scalar 参数之前，与 trailing output_count 合同冲突 | 314e016将输出移到最后，新增 compiler ABI adapter、纯runtime ABI及7项ABI/template值测试；完整算子仍缺失 |
| entry identity | p3_chain内联 walk 对非dataclass Op只读kind/name，attributes/binder覆盖不足；已有output_count字段，不能写成该字段原先完全缺失 | f341e47新增统一identity walk与5项测试，读取实际attrs/局部binders；target完整配置/ABI/TL内容联接仍不完整 |
| 主链占位与拒绝 | apply/verify/compile/install/emitter/completion/storage/solver仍stub；导出最终兜底存在 | 未变；当前无条件拒绝位于p3_chain:201（tag旧行号不同，不混用） |

本代理 git ls-files 和 read 的绝对路径均证明当前 HEAD 跟踪下面四个测试模块与一个 evaluator helper；原包35 CPU框架测试属于不同参考实现，不并入交接树测试或阶段结果。

| 当前交接树测试 | 本代理核实内容 | 证明上限 |
|---|---|---|
| [tests_native_mainline/test_native_builders.py:84](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tests_native_mainline/test_native_builders.py:84>) | 8 项：native 形状/For/ExecutionScope、metadata、结构规模、部分合法性、无 direct compile 字符串、worker import | 不证明真实 Scheduled HIR→analyze/finalize，也不证明 AR/MLA 算式 |
| [tests_native_mainline/test_search_rank.py:43](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tests_native_mainline/test_search_rank.py:43>) | 4 项：SIMULATED 不能排名、样本校验、排序 | 使用构造的 DEVICE_TIMER 记录，不是设备计时证据 |
| [tests_native_mainline/test_identity.py:92](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tests_native_mainline/test_identity.py:92>) | 5 项：真实 Op attributes、同名不同 binder 及函数内容 hash | 不证明语义对应，也未覆盖本报告全部 identity 字段 |
| [backend_operators/test_operator_abi_numeric.py:42](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tests_native_mainline/backend_operators/test_operator_abi_numeric.py:42>) | 7 项：真实 trailing output ABI、独立 NumPy evaluator、descriptor/输入校验；:66 是 x*v，:114 是 q_nope*k_nope | 当前交接树中存在；不是原40，也不是完整 AR RMS/softmax 或 MLA QK/PV |

本轮唯一 Python 命令（只收集，不执行）：

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python -m pytest --collect-only -q -p no:cacheprovider targets/ascend/tests_native_mainline
```

“真实（局部）”表示源码实际执行该检查，不等于正常链已闭合；stub 表示具名拒绝骨架；MISSING 表示当前活动模块中无相应服务或证明。下面测试栏写“新增”者是验收要求，尚未实现/运行。所有设备/原40/P3动态与 MIX/P4–P7 均 NOT_RUN；[docs/HANDOFF_PLAN.md:8](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/HANDOFF_PLAN.md:8>) 已明确当前整体缺口。

## 1. 硬约束 → 实现 → 可机器验收矩阵

当前事务stub精确清单（post-f341e47）：plan/mainline.py 的 apply_symbolic_pipeline:79 = MAINLINE_APPLY_NOT_IMPLEMENTED；verify_actual_program:83 = MAINLINE_VERIFY_NOT_IMPLEMENTED；compile_mainline:87 = MAINLINE_CHAIN_NOT_IMPLEMENTED。install_entry 在:54–72已有entry存在/同名、缺conformance和dict conformance拒绝，:75是这些入参检查通过后的终止INSTALL_VERIFIER_NOT_IMPLEMENTED；不是已实现实际替换。下表相应链接均指同一当前文件。

| ID / 合同硬约束 | 责任模块/函数、当前真实状态及证据 | 必须提供的测试/gate 证据 | 阶段 |
|---|---|---|---|
| C01 单一生产链、唯一 API，无 wrapper/fallback/text twin | 真实局部：[tilefoundry_ascend/__init__.py:6](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/__init__.py:6>) 只导入 pipeline/p3_chain；[passes/p3_chain.py:215](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py:215>) 注册唯一 native mainline；compile 为 stub [plan/mainline.py:86](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/mainline.py:86>)。最终 export 消费路径完整性 MISSING | 新增从 checked_program→analyze→finalize→emit 的成功切片及旧入口拒绝；扫描/导入成功不能代替入口运行 | P0–P3 |
| C02 PipelineSpec 单 schema，stage/V 域独立 | 真实局部：[hir/pipeline.py:96](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/ir/hir/pipeline.py:96>) 独立 max_stage/max_buffer_versions/fixed_ii；[passes/pipeline.py:201](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/pipeline.py:201>) 显式 buffer domain 覆盖默认上界。完整 solver 消费 MISSING | 新增 max_stage=1 配 Fixed(3)/Auto(2,5)，默认 max_buffer_versions=1 不能限制显式域；改变 stage 不改变 V | P1/P2 |
| C03 Fixed 仅作者声明，Auto 按各 buffer 域求解，不 SAT 后重算 V | 真实 typed tuple 校验 [hir/pipeline.py:65](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/ir/hir/pipeline.py:65>)；[analysis/pipeline_reader.py:36](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/analysis/pipeline_reader.py:36>) 将 Auto 投影成上界，只能诊断，不能当 solved V；solver stub [solver/__init__.py:26](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/solver/__init__.py:26>) | 新增 Fixed2 保持2、Auto上下界、容量促使 solver 合法选择、SAT model V 原样到 allocation/slot/TL；不准默认 V1 | P1/P2 |
| C04 TaskIntent 真实 direct-child / buffer allocation identity | 真实局部 [hir/pipeline.py:162](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/ir/hir/pipeline.py:162>)，但最初由 anchor index/root name 定位，必须唯一并绑定；同名 allocation 收集/clone安全仍需验证 | 新增两个同名 binder、同 shape 不同 root/dtype、跨 module ref、anchor漂移、重复 root；定位字符串不能用于后续语义反查 | P1/P2 |
| C05 partial manual 默认0；绝对 stage；no-offset 全stage0；issue_after 非完成边 | 真实规范化 [hir/pipeline.py:211](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/ir/hir/pipeline.py:211>)；solver stub 无后续兑现 | 新增 stage2/3不归零、未标 sibling0、嵌套列表独立、offset冲突早拒；phase拓扑排序/同 owner+pipe源顺序；issue_after不冒充RAW | P1/P2 |
| C06 RegionBindingSummary captures/carries/init/yields/For/Var refs 不按名字 | 部分真实 [transforms/convert_hir_to_tir.py:2231](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/passes/transforms/convert_hir_to_tir.py:2231>) 从 lowering memo/loop identity取引用，但当前输出 dict；[transforms/convert_hir_to_tir.py:2295](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/passes/transforms/convert_hir_to_tir.py:2295>) rebind 用 zip顺序，typed schema/跨 rewrite身份核对未闭合 | 新增非零lower/step>1、nested/carries、同名Var、clone/reorder loop、初值/出口；不能仅靠相同loop数量重绑定 | P1/P2 |
| C07 UnitTree whole-task，bounds Expr、effects/owner/resources/predicate/captures/carries/yields | MISSING：[plan/__init__.py:2](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/__init__.py:2>) 只导出事务四函数；[plan/mainline.py:86](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/mainline.py:86>) compile stub；现有 ExecutionScope 不是 UnitTree 分析 | 新增 VF不拆为单指令任务；config/无buffer效果有归属；百万N的unit/variables数量不变 | P1 |
| C08 完整 NormalizedProblem，无 source-only sidecar；未知task/effect/access具名拒绝 | MISSING：同 compile stub [plan/mainline.py:87](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/mainline.py:87>)。不能拿 D0 manifest 或 builder metadata 当 problem | 新增故意删除 task/access/STATE/RAW/WAR/WAW/reuse/boundary边；source coverage 必须先于solve；未知view/control具名拒绝 | P1/P2 |
| C09 共享 solver 单一路径，SAT/UNSAT/UNKNOWN，独立整数+source postcheck | stub：[solver/__init__.py:16](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/solver/__init__.py:16>) 仅 Assignment词汇；[tilelang_z3/core.py:24](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/vendor/tilelang_z3/core.py:24>) 两旧入口均拒绝，当前没有真实 Z3解 | 新增固定II、recurrence/末读reuse、owner资源分域、环形跨II、issue偏序、容量；小域穷举对照；UNKNOWN不得安装、UNSAT仅有限域 | P2 |
| C10 deadline 为共享 monotonic绝对float，不重置 | 真实局部 [search/process.py:77](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/search/process.py:77>) Deadline.expires/remaining；run_process取剩余 [search/process.py:104](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/search/process.py:104>)。solve_region接受参数但stub，compile无deadline接线 [plan/mainline.py:86](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/mainline.py:86>) | 新增fake clock：多region/分析/求解/replan/发射只接同绝对截止；过期早停；None只显式无限；传expires不能误传remaining | P2/P3 |
| C11 唯一 ResolvedPlan/RegionPlan schema、同一 iteration版本字段被全链消费 | MISSING：[plan/__init__.py:13](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/__init__.py:13>) 无ResolvedPlan；plan主模块只stub。禁止第二套“冻结计划”或算子schema替代 | 新增一份schema/唯一writer；region Expr引用、II/origin/max_shift/tasks/buffers齐全；completion/storage/TIR/TL一致；版本/iteration篡改拒绝 | P2 |
| C12 immutable source snapshot、失败事务原entry不变、不发布 | 真实局部 install前检查 [plan/mainline.py:54](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/mainline.py:54>)，:75终止拒绝；完整事务/不可变快照/replan回滚 MISSING。metadata记录本身可写，不证明不可变 | 新增每阶段故障注入、原Module.functions/entry/源语义digest未变、无artifact发布；重复cut/预算耗尽终止 | P2/P3 |
| C13 finite SymbolicCompletionPlan 显式effects、dense FREE/READY/STATE/exit | stub [completion/__init__.py:29](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/completion/__init__.py:29>)；effects当前有默认()，调用点必须显式传真实effects；有限sites/channels仅空dataclass | 新增每代FREE、writer k>=V wait(k−V)、尾 min(V,N)、N0无通知/无drain、每个最大末读者/最终异步写drain、guard implication、delta generation | P2/P3 |
| C14 PhysicalEventNamespace 真实owner/core/pipe/bank，READY/FREE不独立池 | 真实局部数据/映射 [completion/model.py:349](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/completion/model.py:349>)、[completion/model.py:434](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/completion/model.py:434>)；同文件仍有logical kind pools :396，不能作为生产物理分配；分配/生命周期证明 MISSING | 新增owner分域、pipe别名归一、bank-mask交叠、多消费者、flag exhaustion、wrong endpoint/bank/generation；声明事实不称DEVICE_VALIDATED | P1/P2/P3 |
| C15 版本后物理StoragePlan、layout/stride/scratch一次计费、末读完成才复用 | stub [storage/__init__.py:30](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/storage/__init__.py:30>)；[storage/placement.py:76](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/storage/placement.py:76>) 有结构但无主链放置；[storage/model.py:159](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/storage/model.py:159>) 有独立bytes计算，默认layout/dtype不能填补未知native事实 | 新增ND/NZ/padding、owner分账、额外scratch单计、version_stride/offset篡改、completion变化失效、异步末读前不跨root复用 | P2/P3 |
| C16 UNKNOWN_CAPACITY拒绝，不用footprint回填 | 真实局部 [storage/placement.py:135](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/storage/placement.py:135>) unknown/None fits=False；[storage/__init__.py:15](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/storage/__init__.py:15>) 仅常量，主链 UNKNOWN_CAPACITY实际拒绝路径仍MISSING | 新增Target None/UNRESOLVED 容量使非空需求拒绝；已知超限独立码；不得把None改为需求、0或无限；GM ABI无片上ring强制预算 | P1/P2 |
| C17 当前Target facts、AIC/AIV兄弟域及合法transfer | 部分真实 [tilefoundry_ascend/tf_target.py:106](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/tf_target.py:106>) 拓扑与 :161 space/event hooks；正规MemoryExecutionFacts :145仍仅AIV，MIX模型未接通 | 新增AIC/AIV owner+sid、每空间value/source/status、UB→L1双端点、SS未知拒绝；当前硬件值和TL版本需实测核验，UNKNOWN | P1/P3 |
| C18 symbolic whole-task apply返回private module+source_map，O(tasks)不按N展开 | stub [plan/mainline.py:78](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/mainline.py:78>)；现builder For结构不等于apply已实现 | 新增N=0/1/V−1/V/V+1/17/1_000_003，真实IR节点数除常数不变；p/k/i公式、非零lower、step>1、stage gap>N；whole body保留 | P2 |
| C19 clone按binder/root identity、guard/effectful Let/type/layout/SSA不丢 | verify/apply都stub [plan/mainline.py:78](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/mainline.py:78>)；真实hash [plan/identity.py:127](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/identity.py:127>) 区分局部binder，不能替代转换证明 | 新增同名Var、错误operand/attrs、缺/多task、predicate/Let移guard外、carried init/yield损坏、寄存器跨VF逃逸 | P2 |
| C20 source→actual TIR独立核对，非metadata APPLIED | stub verify [plan/mainline.py:82](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/mainline.py:82>)；[passes/p3_chain.py:193](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py:193>) 只检查status/transaction字段，不能授予语义资格 | 新增逐source statement substitution/multiplicity/operand/type/layout/control核对；caller造verified对象/字段一律不能认证 | P2/P3 |
| C21 actual TL emitter，有限六模板，无重新求调度/延时猜测 | stub [codegen/__init__.py:31](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/codegen/__init__.py:31>)。完整kernel不能从case JSON/text twin重建 | 新增copy/view、wholeVF/reduce、Cube、FIX、PACK_NZ/UB→L1、sync/control，每类正例+未知op拒绝，检查实际expr/binder | P3 |
| C22 read_actual_tl(parsed PrimFunc,refinement)先于install/export/bind | MISSING reader；门当前只duck-type params/body与自报verified [passes/p3_chain.py:143](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py:143>)、:154；:201无条件拒绝兜底 | 新增真实解析类型、精确源绑定、独立模板匹配和后期快照；事件顺序trace必须apply→emit/read→VERIFIED→install→export→bind | P2跨P3 |
| C23 双键PassConfig完整identity，真实compile不改ResolvedPlan | 真实局部固定常量 [codegen/__init__.py:13](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/codegen/__init__.py:13>) 与门tuple完全比较 [passes/p3_chain.py:130](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py:130>)；真实compile调用/回执 MISSING | 新增实际compile spy/receipt同时捕获false/true、少键/值错/配置后改拒绝；后期精度/stage/V/owner/issue order漂移拒绝 | P3 |
| C24 registered TL refinement，不靠parsed PrimFunc或注释宣称一致 | MISSING；[tilefoundry_ascend/tf_target.py:68](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/tf_target.py:68>) 只是参考pass名单；具体要求 [MASTER_SPEC:733](</home/s00653124/workspace/TileFoundry_950PR_handoff/work_package/MASTER_SPEC.md:733>)–:763 | 新增dual-copy覆盖、ND/NZ footprint与PACK内部V→MTE3事件、OOB合法GM域、UBarena映射、flag双射/禁止mutex溢出、ThreadSync边界；实际IR/CCE/资源快照 | P3 |
| C25 entry/artifact identity覆盖 output_count/ABI/target/param types/actualTL | 真实局部函数identity [plan/identity.py:164](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/identity.py:164>) 含output_count、参数type；target :158只class/name，实例aiv/aic配置没入hash；ABI digest [tf_ascend_runtime/abi.py:188](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tf_ascend_runtime/abi.py:188>) 含ordered params，但门未完整比对；TL/source identity仅非空/局部字符串比较 [passes/p3_chain.py:146](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py:146>) | 新增逐字段篡改；重算source/TL bytes hash、ABI全字段digest、target facts/toolchain全配置；不得把caller提供digest当内容；每次export/bind重核 | P2/P3/P7 |
| C26 install实际替换Module.functions，仅完整VERIFIED可安装 | 真实入参检查：[plan/mainline.py:54](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/mainline.py:54>) entry存在/同名，:62无conformance拒，:68 dict拒；:75无条件 INSTALL_VERIFIER_NOT_IMPLEMENTED，真正replace MISSING | 新增full verifier正例实际entry对象改变、printer/emit同对象；fake verdict/foreign candidate/staleplan拒绝；拒绝后original不变 | P2跨P3 |
| C27 bind_invoke_once无host数学，一次logical callable，ordered ABI全检查 | MISSING bind/load/invoke；纯运行时 [tf_ascend_runtime/__init__.py:7](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tf_ascend_runtime/__init__.py:7>) 仅ABI；validate_bindings [tf_ascend_runtime/abi.py:237](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tf_ascend_runtime/abi.py:237>) 实查count/shape/dtype/contiguous/int scalar，未查device/alias/outputs；返回tensors再scalars会改变交错参数顺序 | 新增交错scalar/tensor ABI、stride/device/alias/output、floatalpha/epsilon按ABI类型、wrongstream；CPU callable计数+host Torch dispatch拒绝，最终设备logical launch trace | P3/P4 |
| C28 两worker隔离、fresh receipts、绝对脚本exec、共享预算 | 真实基础 [search/process.py:50](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/search/process.py:50>) 校验run_id/stage/identity，:113 start_new_session及owned timeout；compile_worker/runtime_worker入口 MISSING；compiler包eager导入 [tilefoundry_ascend/__init__.py:6](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/__init__.py:6>) | 新增CPU禁torch_npu/runtime禁tilefoundry*与ortools前缀，fake旧receipt/超时子树测试；调用现有WorkerCommand，不复制supervisor；runtime不从JSON重规划 | P3 |
| C29 AR native builder必须真实RMS/score/softmax/residual、全参数与dtype/tail | 当前stub数学/真实TIR壳：[attnres/build.py:175](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/attnres/build.py:175>) 创建64f32 tile与Vmul :212，参数 :249没有w/gamma/alpha/epsilon；[backend_operators/test_operator_abi_numeric.py:66](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tests_native_mainline/backend_operators/test_operator_abi_numeric.py:66>) 只证x*v | 新增真实Scheduled HIR builder，经normal entry两不同输入CPU；RMS定向反例、D2047/1000 tail、alpha/epsilon/gamma/w变化；原20不变 | P4/P5 |
| C30 MLA native MIX builder真实QK+rope/PV/online状态/layout/GQA/scale | 当前stub数学/真实AIV壳：[mla/build.py:233](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/mla/build.py:233>)，:273只有Vmul；:287只读q_nope/k_nope，无Cube/PACK/NZ/softmax；metadata不是效果 | 新增完整Cube→Vector→Cube、Qrope一次、GQA/layout/right causal、masked行、hi/lo FP32denom、列片grid、deferred epoch drain；原20+补充诊断 | P3/P4/P5 |
| C31 原40/key/seed/golden/comparator/threshold不改；device实测与模型分列 | 当前未执行，[docs/ACTIVE_CONTRACT.md:22](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/ACTIVE_CONTRACT.md:22>) 明确NOT_RUN；fixtures/runner输入保留，比较器/设备可用性 UNKNOWN | fixture hash/40唯一key与原main(operator_api)调用证据，缺比较器BLOCKED_ENV；原20+20、logical launch、容量、对应分列 | P0/P4/P7 |
| C32 搜索真实device计时、兑现knobs去重、完整预算与dispatch | 真实局部SIMULATED拒绝测试 [tests_native_mainline/test_search_rank.py:43](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tests_native_mainline/test_search_rank.py:43>)；完整builder/worker/search闭环MISSING | 新增实际artifact去重/unused knob不计候选、reject MODEL_ONLY、8代表×8实质候选/每case3竞争、邻域/原始event样本/噪声、dispatch冻结 | P5/P6 |
| C33 cold/hit一致、独立复现、没有skip/xfail/零测试假绿 | 当前完整路线MISSING；[冻结合同:53](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md:53>) 与 [原包 GATES.json:166](</home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3/plan/GATES.json:166>) 定义最终门 | 新增cold/hit同artifact身份、篡改缓存拒绝、另一验证者按固定环境重建/原40，测试收集>0与无正常入口skip/xfail，逐源state收据 | P7 |

## 2. 删除无条件 NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED 的必要且联合条件

裁定：现在不能删 [passes/p3_chain.py:201](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py:201>)。它是当前唯一防止伪造 ChainRecord/duck-typed artifact 获可运行资格的最终兜底，不是无用“防御”。下面 E01–E12 必须全部实现并经过正负例；只能将无条件缺口换成真实逐项检查。未支持域继续具名拒绝，不许整段删除 require_export_ready，不许特殊profile/测试开关放行。

现有拒绝码是已实现事实；标“拟新增”的名字是实施验收设计，不冒充当前代码已有异常类型。最终顶层保持 EXPORT_NOT_RUNNABLE，missing记录具体码与source定位。

| 检查 | 必须机器读取/比较的真实对象 | 保留或拟新增拒绝码 |
|---|---|---|
| E01 可信源与完整问题 | canonical immutable source snapshot + 当前source IR重算；UnitTree/effects→problem coverage双向完整，无missing/extra | 拟新增 SOURCE_SNAPSHOT_STALE、SOURCE_COVERAGE_INCOMPLETE、SOURCE_EFFECT_UNSUPPORTED |
| E02 Assignment/唯一计划 | 共享solver SAT、独立整数与source postcheck，所有region硬域、II/stage/start/phase/V/issue顺序与当前plan一致，UNKNOWN不得进入apply | 拟新增 SOLVER_UNKNOWN、SOLVER_UNSAT、ASSIGNMENT_POSTCHECK_FAILED、PLAN_DOMAIN_DRIFT |
| E03 完整private candidate | source_map逐原node substitution证明；wrongoperand/attr/dropped/extra/guard/binder拒绝，原entry尚未安装 | 拟新增 NATIVE_OPERAND_MISMATCH、NATIVE_SOURCE_MAP_INCOMPLETE、NATIVE_CONTROL_MISMATCH、NATIVE_TASK_MULTIPLICITY_MISMATCH |
| E04 completion/storage当前 | 同plan iteration/identity/effects/target，dense FREE及尾wait，namespace与物理bytes/stride/offset/scratch/alias重核 | 拟新增 COMPLETION_GENERATION_MISMATCH、COMPLETION_TAIL_WAIT_MISSING、COMPLETION_NAMESPACE_CONFLICT、STORAGE_PLAN_STALE、STORAGE_LAYOUT_MISMATCH、UNKNOWN_CAPACITY、STORAGE_CAPACITY_EXCEEDED |
| E05 artifact精确来源 | 实际TL source bytes重算digest与artifact.tl_identity一致；source_identity对source snapshot、tir_identity对private候选、plan/completion/storage均核对 | 现有 TL_SOURCE_MISSING、ARTIFACT_*_MISSING、ARTIFACT_PLAN_IDENTITY_MISMATCH；拟新增 TL_SOURCE_IDENTITY_MISMATCH、ARTIFACT_SOURCE_IDENTITY_MISMATCH、ARTIFACT_TIR_IDENTITY_MISMATCH、ARTIFACT_COMPLETION_IDENTITY_MISMATCH、ARTIFACT_STORAGE_IDENTITY_MISMATCH |
| E06 真正解析与reader | 钉住build实际parser从同一source得到实际PrimFunc；拒绝仅params/body对象；read_actual_tl逐支持模板识别，不读自报annotation作为事实 | 现有 PARSED_PRIMFUNC_MISSING、TL_CONFORMANCE_NOT_ACTUAL；拟新增 PARSED_PRIMFUNC_TYPE_INVALID、PARSED_PRIMFUNC_SOURCE_MISMATCH、TL_TEMPLATE_UNSUPPORTED、TL_OPERAND_MISMATCH、TL_CONTROL_MISMATCH、TL_SLOT_GENERATION_MISMATCH、TL_OWNER_MISMATCH、TL_SYNC_ENDPOINT_MISMATCH |
| E07 ABI全字段 | 从真实candidate params/output_count重建OrderedABI；精确比较entry/ordered input+scalar/output/shape/dtype/stride/layout/alias/parameter types，与actual PrimFunc buffer_map/params核对 | 现有 ABI_NOT_ACTUAL、ABI_INPUT_COUNT_MISMATCH、ABI_OUTPUT_COUNT_MISMATCH；拟新增 ABI_PARAMETER_ORDER_MISMATCH、ABI_PARAMETER_TYPE_MISMATCH、ABI_LAYOUT_MISMATCH、ABI_IDENTITY_MISMATCH。当前OrderedABI没有input_count属性而门读取它（p3_chain:136），必须由唯一ABI schema明确派生，不能放松为缺字段即通过 |
| E08 双配置+真实target/toolchain | artifact完整双键==profile实际compile调用捕获参数；target全部配置/空间与事件facts、TL/CANN build身份及source状态一致 | 现有 PASS_CONFIGS_MISMATCH；拟新增 COMPILE_PASS_CONFIGS_MISMATCH、ARTIFACT_TARGET_IDENTITY_MISMATCH、TOOLCHAIN_IDENTITY_MISMATCH。仅target class/name或非空字符串不足 |
| E09 后期refinement | manifest绑定actual前/后IR与final资源/CCE观察；注册dualcopy/VF/NDNZ/padding/arena/flag/ThreadSync，真实额外scratch/同步不可遗漏 | 拟新增 TL_REFINEMENT_MANIFEST_MISSING、TL_REFINEMENT_UNREGISTERED、TL_RESOURCE_REQUIREMENT_MISMATCH、TL_FLAG_MUTEX_UNSUPPORTED、TL_PRECISION_DRIFT、TL_SCHEDULE_DRIFT |
| E10 reader verdict不可伪造且绑定内容 | 由真实reader生成的typed verdict携带all checks及实际对象/digests；读取时重算，不因dataclass verified=True通过；source改变使旧verdict失效 | 保留 TL_CONFORMANCE_UNVERIFIED、TL_CONFORMANCE_*_MISMATCH；拟新增 TL_CONFORMANCE_STALE、TL_VERIFIER_EVIDENCE_INVALID |
| E11 install/export/bind三处重新检查 | private VERIFIED verdict先于install；install actual functions replacement且entry identity匹配；公开export和bind各自重核当前artifact/ABI/target/配置；失败无发布 | 保留 INSTALL_CANDIDATE_UNVERIFIED、PLAN_ENTRY_UNKNOWN、PLAN_ENTRY_MISMATCH、ENTRY_IDENTITY_MISMATCH、STALE_IR_IDENTITY；拟新增 INSTALL_CONFORMANCE_STALE、INSTALL_ENTRY_REPLACEMENT_MISMATCH、BIND_ARTIFACT_UNVERIFIED |
| E12 正常链和突变验收 | 至少一个从真实Scheduled HIR公开入口成功到export的CPU纵向例子，另一不同输入非平凡结果；各E项独立真实产物mutation均拒绝；失败事务原entry不变；无skip/xfail/零测试 | 保留 CHAIN_NOT_RUN、TRANSACTION_INCOMPLETE及EXPORT_NOT_RUNNABLE；拟新增 TRANSACTION_ORDER_VIOLATION、TRANSACTION_ROLLBACK_FAILED |

事件顺序验收必须观察实际调用与Module.functions对象身份，而非比较 SINGLE_TRANSACTION 字符串：当前 [plan/mainline.py:24](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/mainline.py:24>) 只有verify标签，没有明确emit/read节点，[passes/p3_chain.py:195](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py:195>) 仅tuple相等。MAIN-A在同一事务里保留可验证的emit/read→install依赖，标签不能成为第二种证明。

冻结合同先read后install，原包T12→T14却文字上似先安装再reader，不能制造循环依赖：把T12拆成 T12a私有candidate编排和T12b验证后原子install/export；T14消费T12a输出，T12b blocked_by T14实际reader。P2的CPU能力与P3的真实TL/设备证据分列；真实reader准备不足时P2“可公开运行”部分仍NOT_RUN，不能靠延后核对越过门。设备数值不是CPU reader实现资格的替代，完整P3设备门仍独立必须完成。

## 2b. 落地记录（task-49）：gate 现在实际检查什么、今天缺哪一层、什么条件才放行

状态：**已移除** [passes/p3_chain.py:201](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py:201>) 的那条无条件
`NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED`，换成逐层判定（新增 `passes/gate_layers.py`）。顶层仍是
`EXPORT_NOT_RUNNABLE`，`missing` 逐条记录具体码。每层**各有唯一 owner**：同一条缺口只在列表里出现一次
（曾实测 `(tl, TL_ARTIFACT_NOT_ACTUAL)` 在 minimal_aiv 的 gate.gaps 里出现两次；测试用 set() 掩盖了它，
故现在两个测试都断言"无重复行、无跨 label 重复码"，而不是靠去重兜底）。

| 层 | 真实读取/比较的对象 | 通过条件 | 今天缺失（实测 minimal_aiv 全链） |
|---|---|---|---|
| install | `plan/verify.py` 进程内 receipt 注册表，按**候选 entry 对象**匹配（不是按 module id：`install_entry` 返回新 Module，按 module 查永远 None——这是我第一版的错，已改） | 该 entry 对象有 receipt 且 `plan_identity` 非空 | `INSTALL_CANDIDATE_UNVERIFIED`（`verify_actual_program` 仍因 completion/storage/slot_generation 三项 GAP 停在 VERIFIED 之前） |
| plan | 真实 `ResolvedPlan`；用 `resolved.plan_identity_of()` **重算** bytes 与 digest 两个 stored 字段 | 三者一致、且与 record 一致 | `PLAN_ARTIFACT_MISSING`（当前无任何代码写 `PLAN_KEY`） |
| completion | 真实 `CompletionPlan` + `require_completion_verified()`（查内容注册表、逐字段比对、**重跑**结构判定与义务对账） | 重跑通过 | `COMPLETION_ARTIFACT_MISSING` |
| storage | `STORAGE_KEY` artifact 的 stored identity vs record | 一致 | `STORAGE_ARTIFACT_MISSING`（重算需 plan/target/completion，见"未闭合"） |
| tl | 真实 `NativeArtifact`；先判**类型**再判身份（否则 dict 会被报成 identity 不匹配，实测已修）；`tir_identity` 必须等于实际 entry 身份；再由 `read_actual_tl` **重算** verdict | `status==VERIFIED` 且 refusals/gaps 均空 | `TL_ARTIFACT_NOT_ACTUAL`（pass 未把 artifact 写进 metadata） |
| binary | `artifact.compiled` / `artifact.compile_blockage`，三态分开 | 有 `CompiledArtifactFacts` | `COMPILE_SITE_NOT_RUN`（未跑编译站点） |
| abi | `OrderedABI.output_count` + 全部参数名 vs 实际 entry；`OrderedABI` **没有** `input_count` 字段，故输入数按 `len(params)-output_count` 推导 | 计数与尾随输出参数名都一致 | 全链下实测通过（`abi_gaps == []`），不是缺项 |
| event | 遍历候选函数体**实际存在**的 FlagSet/FlagWait（`StmtVisitor`）vs completion 义务数 | 有义务⇒有语句 | `TL_EVENT_LOWERING_UNOBSERVED`（该程序不发事件） |
| chain/entry | 已有：实际 entry 身份、ir_identity、事务完整性、停止阶段 | 全部一致 | `chain:MAINLINE_CHAIN_NOT_IMPLEMENTED` / `TRANSACTION_INCOMPLETE` / `CHAIN_STOPPED_AT_FREEZE` / `entry:ENTRY_IDENTITY_MISMATCH` |

**放行条件**（全部满足才 `ready=True`）：`compile_mainline` 真的 APPLIED 并把 plan/completion/storage/TL artifact
写进 module metadata；`verify_actual_program` 走到 VERIFIED 从而写下 receipt；编译站点真实跑出
`CompiledArtifactFacts`；候选里出现与义务对应的 flag 语句。**今天没有任何真实程序满足任意两条以上**，
所以 gate 仍全拒——拒的是上表具体缺项，不是"尚未实现"。

实测（本机 CANN 9.2.0 可用时、进程内）：`compile_supported_artifact` 成功后 binary 层清空、`read_actual_tl`
返回 **VERIFIED**（gaps/refusals 均空），tl 与 abi 层同时清空；注入两条 flag 语句后 event 层清空。
也就是说 binary/tl/abi/event 四层**已具备真实通过路径**，缺口集中在 plan/completion/storage 的产物写入与
install receipt。

未闭合（不写成已做）：(a) storage 层仍是 stored-vs-record，`verify_storage_current` 是重算入口但需 plan/
target/completion，未接；(b) `plan_gaps` 的 `stored` bytes 为空时跳过比较（手搓 plan 无锚），此时只靠 digest
与 bounds 来源判定；(c) 本表只在 minimal_aiv/independent_ub 上实测。
## 3. 可按序实施的 MAIN-A / MAIN-B 切分

### 3.1 唯一写者和当前路径

MAIN-A 唯一写者负责 [hir/pipeline.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/ir/hir/pipeline.py>)、[transforms/convert_hir_to_tir.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/passes/transforms/convert_hir_to_tir.py>)、[plan/mainline.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/mainline.py>)、[plan/identity.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/identity.py>)、[plan/__init__.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/__init__.py>)、[solver/__init__.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/solver/__init__.py>)、[tilelang_z3/core.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/vendor/tilelang_z3/core.py>) 及其 adapters/provenance、[passes/p3_chain.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py>)、[passes/pipeline.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/pipeline.py>) 和 [tilefoundry_ascend/__init__.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/__init__.py>) 的注册切换。Lead裁定/审核接线，但不同时写这些文件。ResolvedPlan/RegionPlan与NormalizedProblem/UnitTree落现有plan主模块（可内部结构分离，但单一公开schema/authority），不要引用已删除 unit_tree/ledger/resolved_plan/bridge 等模块作为当前基础。

MAIN-B 唯一写者负责 [completion/__init__.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/completion/__init__.py>)、[completion/model.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/completion/model.py>)、[storage/__init__.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/storage/__init__.py>)、[storage/model.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/storage/model.py>)、[storage/placement.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/storage/placement.py>)、[codegen/__init__.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/codegen/__init__.py>)、[codegen/abi.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/codegen/abi.py>)、[tilefoundry_ascend/tf_target.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/tf_target.py>)、[tilefoundry_ascend/memory.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/memory.py>)、[tilefoundry_ascend/sync.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/sync.py>)、[ir/ops.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/ir/ops.py>)、[ir/execution.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/ir/execution.py>)、[attnres/build.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/attnres/build.py>)、[mla/build.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/mla/build.py>)、[tf_ascend_runtime/__init__.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tf_ascend_runtime/__init__.py>)、[tf_ascend_runtime/abi.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tf_ascend_runtime/abi.py>)、[search/process.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/search/process.py>)、[search/rank.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/search/rank.py>)、[search/__init__.py](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/search/__init__.py>)。新增 compile_worker.py/runtime_worker.py、runtime load/reader/bind/operator实现仅在目标现有目录内，标为NEW，不能恢复已删除的x_runtime/native_aiv等旧协议入口。纯runtime reader不能导入compiler，expected facts由编译侧导出typed descriptor；读者不能从case JSON重建调度。

MAIN-A/B均不得触碰对方上述范围；跨界变更提交接口需求由唯一写者落地。测试也单写者：MAIN-A NEW targets/ascend/tests_native_mainline/contract_main_a/，MAIN-B NEW .../contract_main_b/；validator NEW .../contract_acceptance/。现有四模块仅按其责任由对应单写者改，独立验证者不改期望来迎合错误实现。

全局禁写集：fixtures/golden/test/比较器/阈值、原文工作包/旧worktree、归档历史、冻结合同与本次其他报告；禁止Git写操作与NPU调用（本调查阶段）。后续实施需要设备验证时仅validator执行指定队列/锁，本报告中的设备命令未执行。

### 3.2 验收命令约定（后续实施交付必须可运行）

下列 shell 函数是完整解释器和环境前缀的简写，每次Python调用都会带三环境变量。未存在的contract_*测试目录是明确NEW交付，当前运行会失败，不能算PASS；每个切分须先交付对应目录测试，collect-only退出0且tests>0，再执行；不使用只靠-k可能选中零项的命令。这里不是发明生产CLI。

```sh
cpu() {
  PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend /home/s00653124/workspace/envs/tfnpu-v2/bin/python "$@"
}
# R0：现有活动测试；本报告只运行了前文collect-only形式。
cpu -m pytest -q -p no:cacheprovider targets/ascend/tests_native_mainline
# RA / RB / RV：NEW测试目录，Txx是任务专属子目录；必须交付实际测试。
# 例如 T03：
cpu -m pytest --collect-only -q -p no:cacheprovider targets/ascend/tests_native_mainline/contract_main_a/T03
cpu -m pytest -q -p no:cacheprovider targets/ascend/tests_native_mainline/contract_main_a/T03
# 对应下表 RA(Txx)/RB(Txx)/RV(Txx) 分别将目录改为contract_main_a/main_b/acceptance的Txx。
```

每个下表命令中的RA/RB/RV均严格指上面两条pytest命令顺序执行，带完整cpu环境；这使每个任务有明确可执行的验收命令与非零收集要求。P3以后RV含设备测试，必须显式队列运行并读取actual artifacts，CPU阶段不得偷跑。异常码均为“拟新增”除第1/2部分已列现有码。实际snapshot/CPU两输入/mutation/设备证据分栏，不能一个通过值覆盖所有结果。

| 子任务/唯一写者与文件范围（本节路径范围精确继承3.1） | 前置依赖（blocked_by） | 验收命令及可观察结果 | 失败拒绝码；不可触碰 |
|---|---|---|---|
| A1/T03 MAIN-A：HIR pipeline schema/parser-reader-printer消费者；D0规范化职责迁入现有pipeline模块 | T00,T02；D0切换须等A6替代检查 | RA(T03)：parse/print/evaluate真实动态AIV与MIX声明，引用绑定/FixedAuto/manual/nooffset | TASK_INTENT_REBIND、BUFFER_SCHEDULE_REBIND、PIPELINE_OFFSET_STAGE_CONFLICT；禁B/fixture/第二schema |
| B1/T04 MAIN-B：tf_target/memory/completion.model/ir/sync目标事实 | T00 | RB(T04)：Target正规查询所有owner/space/event/layout带provenance；未知拒绝；AIC纳正规执行域 | UNKNOWN_CAPACITY、COMPLETION_EVENT_NAMESPACE_MISSING、TARGET_CAPABILITY_UNKNOWN；禁A/猜测硬件值 |
| A2/T05 MAIN-A：ConvertHIRToTIR bindings、plan.mainline UnitTree | T03 | RA(T05)：真HIR wholeVF/captures/carries到Expr/ref、same-name binder/region映射；不按数量/顺序重绑 | REGION_BINDING_UNRESOLVED、SOURCE_UNIT_UNSUPPORTED；禁B/独立算子IR |
| A3/T06 MAIN-A：plan.mainline problem分析，analysis实际effects接口最小接线 | T04,T05 | RA(T06)：全部RAW/WAR/WAW/STATE/REUSE/边界与access coverage；未知具名拒绝 | SOURCE_COVERAGE_INCOMPLETE、SOURCE_EFFECT_UNSUPPORTED、DEPENDENCE_UNSUPPORTED；禁B/sidecar补账 |
| A4/T07 MAIN-A：vendor core/adapters/provenance、solver唯一adapter | T03,T06；D1参考输入裁定完成后才可移植 | RA(T07)：实际Z3固定II/FixedAuto/资源/三态、独立整数与小域穷举、shared绝对deadline | SOLVER_UNKNOWN、SOLVER_UNSAT、ASSIGNMENT_POSTCHECK_FAILED、DEADLINE_EXPIRED；禁B/第二solver/fallback |
| A5/T08 MAIN-A：plan.mainline唯一ResolvedPlan schema+resolver；identity | T05,T06,T07 | RA(T08)：多regionExpr、全部assignment消费、same iteration、V不重算、完整identity mutation | PLAN_DOMAIN_DRIFT、PLAN_IDENTITY_MISMATCH；禁B改schema/Lead并行编辑 |
| A6/T09 MAIN-A：plan.mainline private whole-task apply、source_map、native verifier | T05,T08；完整sync/storage模式另等T10,T11 | RA(T09)：p/k/i、百万N规模、two inputs、Let/guard/binder/operand/SSA mutation；只产private | NATIVE_OPERAND_MISMATCH、NATIVE_CONTROL_MISMATCH、NATIVE_SOURCE_MAP_INCOMPLETE；禁B/提前install |
| B2/T10 MAIN-B：completion init/model有限templates及verifier | T04,T06,T08 | RB(T10)：显式effects、denseFREE、小N独立token解释、N0/V边界、bank/owner/末读/exit | COMPLETION_GENERATION_MISMATCH、COMPLETION_TAIL_WAIT_MISSING、COMPLETION_NAMESPACE_CONFLICT；禁A/按N展开 |
| B3/T11 MAIN-B：storage init/model/placement、lowering requirements | T04,T08,T10 | RB(T11)：physical布局/stride/scratch单计、owner容量、stale offset拒绝、reuse HB | UNKNOWN_CAPACITY、STORAGE_CAPACITY_EXCEEDED、STORAGE_PLAN_STALE；禁A/FixedV静默降级 |
| A7/T12a MAIN-A：plan.mainline编排、p3_chain私有验证接口；D0切换 | T07,T08,T09,T10,T11；切换须全部替代谓词/part检查已就位 | RA(T12a)：normal finalize→private candidate、故障回滚/deadline/replan、没有public发布 | TRANSACTION_ROLLBACK_FAILED、REPLAN_BUDGET_EXHAUSTED、REPLAN_CUT_REPEATED；禁B/删:201 |
| B4/T13 MAIN-B：codegen init有限AIV emitter、scalar/expr注册及requirements | T04,T09；完整发射等T10,T11 | RB(T13)：真实TF纯expr/binder→parsed TL、copy/VF/control不重排、dtype/tail/guard | TL_TEMPLATE_UNSUPPORTED、TL_BINDER_UNRESOLVED；禁A/整算子字符串/猜延时 |
| B5/T14 MAIN-B：纯runtime actual TL reader/refinement、codegen桥与唯一ABI | T08,T12a,T13（原T12细分以满足冻结顺序） | RB(T14)：第2节E01–E10可机器检查，真实parser/后期快照，逐项mutations，双键compile capture | TL_OPERAND_MISMATCH、TL_REFINEMENT_UNREGISTERED、TL_RESOURCE_REQUIREMENT_MISMATCH、COMPILE_PASS_CONFIGS_MISMATCH；禁A/伪verified |
| A8/T12b MAIN-A：install_entry原子替换、p3_chain export gate；移除:201的唯一任务 | T12a,T14；E01–E12全绿 | RA(T12b)+RB(T14)：真实函数替换，export/bind前重核，拒绝保留源；成功normal entry+全部fault mutation | INSTALL_CONFORMANCE_STALE、INSTALL_ENTRY_REPLACEMENT_MISMATCH、EXPORT_NOT_RUNNABLE；禁B/绕reader |
| B6/T15 MAIN-B：纯runtime ABI/load/bind_invoke_once、NEW绝对worker脚本、search.process复用 | T00,T01,T04；可提前独立准备，不等solver全成 | RB(T15)：交错ABI、typed scalar/device/alias/stream、one call无hostmath、sys.modules前缀禁入、fresh receipts/timeout、cold/hit | ABI_PARAMETER_ORDER_MISMATCH、ABI_DEVICE_MISMATCH、ABI_ALIAS_UNSUPPORTED、WORKER_IMPORT_BOUNDARY、WORKER_RECEIPT_MISMATCH、BIND_ARTIFACT_UNVERIFIED；禁A/重复进程框架 |
| V1/T16 validator：NEW contract_acceptance/T16（测试only，生产由原writer修） | T12b,T13,T14,T15 | RV(T16)：动态AIV N矩阵/step/lower/V1–3，actual entry/TL/CCE常规模、两输入、设备结果/一次logical调用 | DEVICE_NUMERIC_FAILED、LOGICAL_LAUNCH_COUNT_MISMATCH；禁改生产/fixture/用CPU顶device |
| B7/T17 MAIN-B：ir/target/codegen MIX六模板，PACK_NZ/transfer/refinement | T04,T10,T11,T12b,T14,T15 | RB(T17)+RV(T17)：Cube→Vector→Cube全往返、dtype/sid/guard/exit、physical scratch与V→MTE3内部边 | TL_SYNC_ENDPOINT_MISMATCH、MIX_TRANSFER_OWNER_MISMATCH、TL_FLAG_MUTEX_UNSUPPORTED；禁A/遗漏返程 |
| B8/T18 MAIN-B：attnres/build完整native Scheduled HIR canonical | T03,T05,T06,T13 | RB(T18)：完整原ABI/数值所有输入，RMS/softmax/residual/native guard，两输入经公开链 | ATTNRES_NATIVE_BODY_INCOMPLETE、ABI_IDENTITY_MISMATCH；禁A/host数学/metadata补算式 |
| V2/T19 validator：AR原20/launch/资源/对应报告 | T01,T16,T18 | RV(T19)+下述原runner AR命令，20 exact labels且同artifact | ORIGINAL_CONTRACT_MISMATCH、DEVICE_NUMERIC_FAILED；禁fixture/threshold |
| B9/T20 MAIN-B：mla/build完整native canonical MIX | T03,T05,T06,T17 | RB(T20)：QK/rope/P/PV/FP32m-l-U，双layout/GQA/scale/causal/tail，实际owner与参数 | MLA_NATIVE_BODY_INCOMPLETE、MIX_TRANSFER_OWNER_MISMATCH；禁A/直接TLcompile |
| V3/T21 validator：MLA原20及unsharded补充 | T01,T17,T20 | RV(T21)+下述原runner MLA命令，20 exact标签+补充分列 | ORIGINAL_CONTRACT_MISMATCH、DEVICE_NUMERIC_FAILED；禁提前依赖列片P5 |
| B10/T22 MAIN-B：attnres/build三族+factor精度 | T19 | RB(T22)+RV(T22)：resident/block-online/full-online真实不同IR且RMS仍在，原comparator | CANDIDATE_KNOB_UNAPPLIED、DEVICE_NUMERIC_FAILED；禁A/改名充候选 |
| B11/T23 MAIN-B：attnres/build多行/流水/驻留/预取 | T19,T22 | RB(T23)+RV(T23)：token_rows/d_chunk/V改变实际计划layout与输出，不隐式重读伪驻留 | CANDIDATE_KNOB_UNAPPLIED、STORAGE_CAPACITY_EXCEEDED；禁A/弱阈值 |
| B12/T24 MAIN-B：mla/build列片/grid/precision | T21 | RB(T24)+RV(T24)：D448末列片coverage、actualjobs/grid变化、hi-lo precision/denom | MLA_OUTPUT_COVERAGE_MISMATCH、TL_PRECISION_DRIFT；禁A/host concat |
| A9+B13/T25 顺序单写者：MAIN-A先STATE/effects类型(仅A域)，MAIN-B后mla builder/completion(仅B域) | T08,T10,T21；B部分blocked_by A部分 | RA(T25)→RB(T25)：epoch/alpha/P/generation身份、所有列hi/lo旧PV完成后rescale/clear | STATE_EPOCH_MISMATCH、COMPLETION_GENERATION_MISMATCH；互禁对方文件 |
| B14/T26 MAIN-B：mla MIX deferred真实发射；validator设备 | T17,T24,T25 | RB(T26)+RV(T26)：迟到旧PV/不同sid分支负例、canonical/deferred原比较器对照 | STATE_EPOCH_MISMATCH、DEVICE_NUMERIC_FAILED；禁A/CPU模型顶替 |
| B15/T27 MAIN-B：现有search入口接actual builder/resource；仅必要NEW driver/harness职责 | T12b,T14,T15,T19,T21 | RB(T27)：真实artifact identity去重，数学硬拒/implementation missing/模型排序分列，actual storage硬过滤 | CANDIDATE_IDENTITY_DUPLICATE、CANDIDATE_KNOB_UNAPPLIED、IMPLEMENTATION_MISSING；禁A/引用已删search模块为现有基础 |
| B16/T28 MAIN-B：search/process/rank与runtimeworker actions；validator测量 | T15,T16,T17,T27 | RB(T28)+RV(T28)：真实numeric→launch→profile顺序/receipt，DEVICE_TIMER原始样本，预热/编译不入计时 | RANK_SIMULATED_NOT_COMPARABLE、WORKER_RECEIPT_MISMATCH、DEVICE_PROFILE_INVALID；禁A/模拟计时 |
| V4/T29 validator/executor：search运行报告和预算记录（生产B单写） | T22,T23,T24,T26,T27,T28 | RV(T29)：每算子8代表×8实质device正确候选、每原case3竞争及可行域证明，完整budget轨迹 | SEARCH_COVERAGE_INCOMPLETE、SEARCH_BUDGET_EXHAUSTED；禁fixture/预算耗尽冒充完成 |
| B17/T30 MAIN-B：NEW纯runtime dispatch与search winner冻结，验证者复查 | T29 | RB(T30)+RV(T30)：局部邻域/噪声重测、实际最快候选、全部40 specialization绑定artifact | DISPATCH_SPECIALIZATION_UNBUILT、DISPATCH_IDENTITY_STALE；禁A/JSON重规划 |
| V5/T31 validator：最终40/cold-hit结果；只报告 | T01,T30 | RV(T31)+两原runner命令cold/hit模式的真实环境复验（工具无通用cold开关，具体cache控制UNKNOWN待build能力） | ORIGINAL_CONTRACT_MISMATCH、CACHE_IDENTITY_MISMATCH；禁fixture/伪binaryhash |
| V6/T32 independent validator：独立复现/全CPU回归，只报告 | T31 | RV(T32)+R0；独立者从固定source/env/artifact重建，原40再现且所有active入口不skip | REPRODUCTION_INCOMPLETE、SOURCE_STATE_STALE；禁修改被验收源码 |
| L/T00,T01,T02,T33 Lead/validator：来源/合同/反例/最终索引，报告单写 | T00无；T01/T02等T00；T33等T32 | RV(T00/T01/T02/T33)需分目录真实测试；fixture哈希/原main入口/比较器身份、旧旁路拒绝、交付source/commands/hash齐全 | PACKAGE_INPUT_MISSING、ORIGINAL_CONTRACT_MISMATCH、HANDOFF_INCOMPLETE；禁生产schema/原fixtures |

T00/T01/T02/T33分别对应P0环境/原40/撤旁路与P7交付；上表完整覆盖V3-T00..T33，并只增加T12a/b及T25的串行细分。原DAG每条依赖保留（[MASTER_SPEC:1611](</home/s00653124/workspace/TileFoundry_950PR_handoff/work_package/MASTER_SPEC.md:1611>)–:1644），T12→T14改为私有准备/认证提交两节点是冻结合同强制顺序修正，不授权提前install。其他既有依赖不足之处仅加边，例如T09的完整同步结果等T10/T11。

后续授权设备阶段的原runner命令（NEW operators.py 是B6/T15+B17/T30交付；当前不存在，故状态NOT_RUN，禁止本次执行；唯有validator在外层持 /home/.tf_device0.lock 一次运行，不由worker再取同一锁）：

```sh
cpu work_package/tools/run_contract.py --case attention_residual --operator-file /home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tf_ascend_runtime/operators.py --callable attention_residual --device npu:0 --report /tmp/tf-ar-contract.json --log /tmp/tf-ar-contract.log
cpu work_package/tools/run_contract.py --case mla --operator-file /home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tf_ascend_runtime/operators.py --callable mla --device npu:0 --report /tmp/tf-mla-contract.json --log /tmp/tf-mla-contract.log
```

参数来自 [specs/06_validation_and_performance.md:29](</home/s00653124/workspace/TileFoundry_950PR_handoff/work_package/specs/06_validation_and_performance.md:29>)，而不是不存在的旧operator CLI。真实正式结果应使用fresh唯一报告路径，以上/tmp路径只是可执行示例，不用于最终身份复用。比较器路径/部署TL build/pass instrumentation的具体可用API仍UNKNOWN，T00/B5必须先核实；不要凭原包固定558参考推断部署。

### 3.3 八阶段门的结案口径

| 门 | 原34任务归属 | 必须读到的产物与当前结论 |
|---|---|---|
| G0/P0 | T00–02 | 当前source/env/比较器+40 hash/key+撤旧资格反例；仅本报告不是G0通过。NOT_RUN |
| G1/P1 | T03–06 | native策略→真实whole units→完整problem，Target事实统一且未知拒绝。NOT_RUN |
| G2/P2 | T07–12 | 真solver/唯一plan/completion/storage/private apply，CPU semantic mutation与实际安装（认证顺序按T12a/b）；大N常规模。NOT_RUN |
| G3/P3 | T13–17 | 真TL reader与后期refinement、双worker/bind一次调用、动态AIV及完整MIX设备数值/资源。NOT_RUN |
| G4/P4 | T18–21 | 原AR20+MLA20、unsharded补充，plan对应/资源/launch/数值分列。NOT_RUN |
| G5/P5 | T22–26 | AR三族/驻留流水、MLA列片/precision/deferred真实device候选，knob兑现。NOT_RUN |
| G6/P6 | T27–30 | 实际device样本、预算覆盖/竞争/邻域及完整dispatch。NOT_RUN |
| G7/P7 | T31–33 | 最终原40 cold/hit、独立复现、完整源码/产物/命令。NOT_RUN |

门定义来自 [原包 GATES.json:5](</home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3/plan/GATES.json:5>)、[原包 GATES.json:48](</home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3/plan/GATES.json:48>)、[原包 GATES.json:73](</home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3/plan/GATES.json:73>)、[原包 GATES.json:97](</home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3/plan/GATES.json:97>)、[原包 GATES.json:119](</home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3/plan/GATES.json:119>)、[原包 GATES.json:143](</home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3/plan/GATES.json:143>)、[原包 GATES.json:166](</home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/20261112/tf_ascend950pr_native_v3/plan/GATES.json:166>)；任何FINISHED、历史passed数、metadata VERIFIED、模型CPU结果都不证明这些门通过。当前无本轮deterministic gate run_id，本报告不提交设备门。

## 4. D0 七 pass 裁定：移除旧正常链注册，迁移必要检查

结论选择“移除”：移除的是七个D0注册pass与其metadata协议的正常链资格；有效的声明/谓词/part正确性检查必须由单一NormalizeIntent、actual program verifier、TL reader承接并保留负例。不是删除真实安全检查。[research/mainline_cleanup_manifest.md:48](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/research/mainline_cleanup_manifest.md:48>) 要求追完真实import/注册/调用，:50明确不能把正确性检查当兼容删除；当前七pass都是模块装饰器实际注册，[tilefoundry_ascend/__init__.py:8](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/__init__.py:8>) 导入触发，[schedule/finalize.py:17](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/schedule/finalize.py:17>) 运行，[passes/pass_manager.py:42](</home/s00653124/workspace/TileFoundry_950PR_handoff/src/tilefoundry/passes/pass_manager.py:42>) 按注册顺序append。

当前所读manifest并无逐字“D0链必须从正常链移除”的句子；该引述的精确出处UNKNOWN。它有上述清理原则，本裁定依据冻结单链/真实证明和当前源码，不伪造文档引文。

| 当前pass / 注册位置 | 最终主链仍正当的职责 | 处置与机器验收 |
|---|---|---|
| ValidatePipelineDeclarations / BEFORE_LOWER [passes/pipeline.py:733](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/pipeline.py:733>) | typed schema/FixedAuto/stage-V独立、作者输入合法性应保留；不能再依赖按loop路径+buffer名称的manifest | 移除D0注册，A1 NormalizeIntent接入同等/更严真实引用检查；RA(T03)显式域越默认上界可合法、未知/不合法拒绝 |
| NormalizeSchedulePredicates / BEFORE_LOWER [passes/pipeline.py:757](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/pipeline.py:757>) | 从真实access窗与原guard导出每issue需求应由normal analysis做 | 当前:792固定requirements=()，不保留“空需求即正常”的旁路。A3 coverage+plan guard承担；缺source窗必须具名拒绝 |
| VerifyPredicateCoverage / AFTER_LOWER [passes/pipeline.py:797](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/pipeline.py:797>) | 对每个真实issue/active lanes/精度的独立guard证明属于A6 verifier/B5 reader | 当前只遍历manifest.requirements，空时不查。迁后wrong/缺mask实际指令突变必须拒绝，非依据annotation |
| VerifyPipelinePreservation / AFTER_LOWER [passes/pipeline.py:848](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/pipeline.py:848>) | HIR intent到region plan的identity/source map保存必须保留 | 当前:871按位置loop key比较typed spec；符号rewrite改变循环结构后此比较不适用。A2/A5/A6逐region ref+source map/硬约束验证替换 |
| NormalizePartSelection / BEFORE_LOWER [passes/pipeline.py:896](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/pipeline.py:896>) | 合法transport分片/part属于lowering template requirements与typed intent | 当前无projection只NO_PROJECTION metadata :907；移除旧PART_* sidechannel，B4/B7按真实dtype/layout/transfer选择，未知拒绝 |
| VerifyTirPartSelection / AFTER_LOWER [passes/pipeline.py:943](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/pipeline.py:943>) | actual指令part/width/dtype合法性必须保留 | 迁B4/B5正规ISA/template核对，不能一刀切禁止有合法新模板的part；未支持实际part具名拒绝；VCVT属性mutation仍覆盖 |
| RecordPredicateProvenance / AFTER_LOWER [passes/pipeline.py:966](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/pipeline.py:966>) | per-issue实际mask provenance必要，证明应由实际reader提取 | 当前:1000只取首个Pset/Pset16并以dst shape全lanes构造active，不能证明所有动态mask。移除该metadata记录；A6/B5逐statement actual mask/guard/owner核对 |

影响面：正常包先import D0再import p3_chain（[tilefoundry_ascend/__init__.py:8](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/__init__.py:8>)），D0 AFTER_LOWER按定义顺序先于RunAscendNativeMainline执行；p3_chain导入和run体 [passes/p3_chain.py:12](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py:12>)、[passes/p3_chain.py:221](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/passes/p3_chain.py:221>) 不读取D0 manifest/PART/predicate provenance，只调用compile_mainline、identity及自己的artifacts。因此可按A1→A3/A6→A7顺序原子切换注册，p3_chain必须继续 fail-closed；删D0不会自动闭合主链、也不会允许删:201。不要修改通用PassManager排序规则迁就旧pass。

现有测试影响：[tests_native_mainline/test_native_builders.py:84](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tests_native_mainline/test_native_builders.py:84>) 直接builder+verify_prim_function，没有finalize；[tests_native_mainline/test_identity.py:28](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tests_native_mainline/test_identity.py:28>) 只导入ir_identity；[backend_operators/test_operator_abi_numeric.py:64](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tests_native_mainline/backend_operators/test_operator_abi_numeric.py:64>) 直接NumPy evaluator；[tests_native_mainline/test_search_rank.py:43](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tests_native_mainline/test_search_rank.py:43>) 只测rank。这些测试未断言七D0注册或其正常入口执行，可能全部继续绿却丢掉谓词/part检查。切换必须新增RA(T12a)正常公开入口测试，断言target_passes无d0.*、NormalizeIntent/actual verifier确实拒绝缺mask/错误part/丢pipeline/FixedV漂移，至少一个合法动态成功例；collect>0并运行R0。归档D0测试不恢复为生产资格；仍有效负例按当前单链重写。

## 5. 风险排序与下一步

1. 最高风险：把自报identity/duck PrimFunc/typed verified当真实conformance，提前删无条件拒绝或先install后reader。先做A1–A7+B1–B5；T12b只在E01–E12全部闭合后进行。
2. 最高风险：计划/效果根本不存在时，用builder metadata与64f32 transport测试冒充AR/MLA。完整problem/shared solver/唯一schema先闭合，再实现真实Scheduled HIR算式；完整原40仍独立设备门。
3. 最高风险：TL关闭AutoSchedule后仍改变layout/scratch/flag/内部完成，且ABI/target/source身份链不全。B6双worker可从P0并行准备，B5必须读实际后期IR/资源，MIX全往返通过后再推进MLA与deferred。

明确UNKNOWN：部署TL/CANN/比较器当前事实、可用pass instrumentation具体API、真实原40与设备结果。12→24测试面的变化已确认来自并行团队314e016/f341e47提交，不再列UNKNOWN。未实现与未运行保持分列，报告完成不等于生产任务或阶段完成。
