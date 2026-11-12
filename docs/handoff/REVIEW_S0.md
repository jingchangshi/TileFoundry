# REVIEW_S0 — task-15 / reviewer-g1

- 最终结论：**通过（已审受限S0：入口第一段、求解核心/公开数值接口、原生采集→按region求解→唯一计划、CPU执行器与类型设计）**；生产PassManager事务/完整S0仍待补。首尾HEAD=`f341e47529453bdbc8dd81f2c108d36628fb4b8b`；未提交增量按文末哈希绑定，不以HEAD代替内容身份。
- 范围：作者 Scheduled HIR→正常入口、固定 II 共用求解核心与类型方案；只写本报告，不改实现、不做 Git 写、不调用设备；不运行可能写 dump 的 legacy 套件。
- 状态：`待核对` 不等于 FAIL；`通过` 仅授予表中明确产物，不外推完整 S0、设备、原40或阶段门。

## 待核对项

下表返回值/事务次序来自唯一冻结合同。实现前后定位及动态证据在交付后补；未实现的后半链保留 MISSING，不能用记录字段代替产物。

| API / 合同 | 冻结签名→返回值；顺序 | 用什么证据判定 | 结果 |
|---|---|---|---|
| 1 / [INTERFACE_FREEZE_NATIVE_MAINLINE.md:20](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md#L20>) | normalize_intent(native HIR)→typed intent + RegionBindingSummary | 实际 LoopRegion 与 captures/carried/yields→lowered refs 的对象绑定，非字符串路径 | 本轮实现未验 / NOT_RUN |
| 2 / [INTERFACE_FREEZE_NATIVE_MAINLINE.md:21](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md#L21>) | build_native_unit_tree(checked TIR,region_bindings)→UnitTree | 实际 whole-task node/region/Expr/effects 引用与源节点一一对应 | S0 whole-unit采集通过；绑定近似见末节 |
| 3 / [INTERFACE_FREEZE_NATIVE_MAINLINE.md:22](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md#L22>) | analyze_native_problem(units,effects,alias,target)→NormalizedProblem + source coverage verdict | 从源 access/effect 重算 task/边/buffer 覆盖；未知域具名拒绝 | S0受限集合/归属覆盖通过；完整依赖充分性未验 |
| 4 / [INTERFACE_FREEZE_NATIVE_MAINLINE.md:23](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md#L23>) | solve_region(problem,deadline:float或None)→Assignment(status,starts,stages,phases,versions,issue_order,optimality) | 真实数值赋值、独立整数/source postcheck；SAT/UNSAT/UNKNOWN，绝对monotonic deadline | 数值接口通过；正式native投影NOT_RUN |
| 5 / [INTERFACE_FREEZE_NATIVE_MAINLINE.md:24](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md#L24>) | resolve_schedule(problem,assignment)→ResolvedPlan | 同一赋值消费到唯一 RegionPlan 的 II/stage/phase/start/rank/V，V不再推算 | S0按region SAT赋值→单一计划通过；完整字段待补 |
| 6 / [INTERFACE_FREEZE_NATIVE_MAINLINE.md:25](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md#L25>) | build_symbolic_completion(plan,target,effects)→SymbolicCompletionPlan | 有限真实 sites/channels、effects端点、dense FREE/每代SetWait/尾min(V,N) | 本轮实现未验 / NOT_RUN |
| 7 / [INTERFACE_FREEZE_NATIVE_MAINLINE.md:26](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md#L26>) | place_resolved_versions(plan,completion,target,lowering)→StoragePlan | 实际PhysicalLayout/Requirement/Key/LiveWindow与offset/stride/容量，版本后分配 | 本轮实现未验 / NOT_RUN |
| 8 / [INTERFACE_FREEZE_NATIVE_MAINLINE.md:27](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md#L27>) | apply_symbolic_pipeline(entry,plan,completion,storage)→private candidate module + source_map | 私有真实Module/PrimFunction和逐源节点operand/guard/binder对应；原entry尚未替换 | 本轮实现未验 / NOT_RUN |
| 9 / [INTERFACE_FREEZE_NATIVE_MAINLINE.md:28](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md#L28>) | verify_actual_program(original,rewritten,plan,completion,storage)→verdict | 读取两个真实程序与plan/completion/storage；错operand/漏task/错generation反例失败 | 本轮实现未验 / NOT_RUN |
| 10 / [INTERFACE_FREEZE_NATIVE_MAINLINE.md:29](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md#L29>) | install_entry(module,rewritten)→module | 仅完整VERIFIED后Module.functions实际替换；[INTERFACE_FREEZE_NATIVE_MAINLINE.md:39](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md#L39>)要求actual TL reader先行，拒绝保留源对象 | 本轮实现未验 / NOT_RUN |
| 11 / [INTERFACE_FREEZE_NATIVE_MAINLINE.md:30](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md#L30>) | emit_supported_tl(module,plan,completion,storage,target)→NativeArtifact | 私有candidate导出的实际TL source、ABI、parsed PrimFunc与双键PassConfig，不靠metadata自报 | 本轮实现未验 / NOT_RUN |
| 12 / [INTERFACE_FREEZE_NATIVE_MAINLINE.md:31](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md#L31>) | read_actual_tl(artifact,parsed_prim_func,refinement_manifest)→conformance verdict | 真实解析PrimFunc的params/buffer_map/body/attrs与源/candidate逐模板核对，早于install/export/bind | 本轮实现未验 / NOT_RUN |
| 13 / [INTERFACE_FREEZE_NATIVE_MAINLINE.md:32](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md#L32>) | bind_invoke_once(artifact,abi,inputs,device)→outputs | 有序ABI的真实一次logical调用/输出；host无数学；本轮禁设备，只评源码和CPU证据 | 本轮实现未验 / NOT_RUN |

事务固定：[INTERFACE_FREEZE_NATIVE_MAINLINE.md:35](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/INTERFACE_FREEZE_NATIVE_MAINLINE.md#L35>)冻结源→分析/求解/resolve→completion/storage→private apply→emit/read完整VERIFIED→install→export/bind；任何失败无发布并保留原entry。表的API枚举序号不代表可先install后emit/read。

| S0 硬断言 | 用什么真实证据判定 | 结果 |
|---|---|---|
| H1 正常入口真实执行 checked_program→analyze→finalize | 作者fixture由checked_program解析/检查，analyze真实调用compute-cost/memory，finalize真实进入ConvertHIRToTIR；不mock或绕过任一段；保留输入HIR与输出TIR对象 | 通过（入口段；证据见本轮评审） |
| H2 实际entry为PrimFunction，output_count对应尾部输出 | module.entry_function()真实类型、params末尾输出的storage/type及store目标，返回/输出绑定一致；不是仅断言output_count整数 | 通过（入口段；证据见本轮评审） |
| H3 链停在具名阶段而非异常逃逸 | H1实际返回后检查typed ChainRecord、stopped_at/reason与真实事务调用一致；当前拒绝是freeze/MAINLINE_CHAIN_NOT_IMPLEMENTED，不算已编译 | 通过（入口段；证据见本轮评审） |
| H4 gate真实ready=False且缺口具名 | 对同一实际Module直接重算require_export_ready，拒绝异常具有code/missing；metadata cached verdict仅旁证，不授予安装/导出资格 | 通过（入口段；证据见本轮评审） |
| H5 solve_region三态与共享deadline | 同一公开API真实SAT赋值、明确无解UNSAT、过期/未知UNKNOWN无可安装赋值；在调用链追踪同一绝对deadline，含None显式无界；不重置预算 | 通过局部三态/过期deadline/显式II域；全链共享预算/source覆盖NOT_RUN |
| H6 作者stage不被归一，no-offset不偷改约束 | 非零绝对stage实例(如2/3)、部分标记默认0、no-offset冲突；核对model与返回的每task stage/start/phase并独立验算 | 通过manual/未标0/no-offset反例；自动offset支持未交付 |
| H7 SAT后不重算V，stage/V域独立 | Fixed(V2)与Auto(min,max)、不同max_stage输入；读取实际solver model V和返回Assignment.versions/后续plan字段完全一致，禁止后处理latency猜V | 数值Fixed3保真及plan同V通过；自动域穷举待补 |
| H8 独立postcheck不复用Z3约束 | 纯整数重算t=II·stage+phase、d·II依赖、复用/跨II资源/容量/issue顺序；篡改真实SAT赋值必须失败，小域穷举与可行域对照；source覆盖另立非伪造接口 | 通过整数层C1–C8反例；真实source postcheck与三task/V穷举待补 |
| H9 无禁止项且测试看真实产物 | 读本轮diff与实际调用：无text twin/按N展开/direct tilelang.compile/metadata-only安装/兼容路径/skip/xfail/放宽旧断言；源码字符串搜索只作线索 | 入口无禁止项；adapter缺字段已拒；无skip/xfail，正式native产物链NOT_RUN |
| H10 唯一schema与唯一solver | TF/TL消费者同一物理core/函数；core纯数值视图与native refs有显式覆盖映射，不复制另一NormalizedProblem/ResolvedPlan；类型方案逐字段对合同:7–17比对并绑定现有分析记录 | 单core/单native schema；S0数值投影与真实单计划通过，通用覆盖未验 |

## 本轮落地评审与需改项

- S0-entry 第一段通过源码复审：作者源 [minimal_aiv.py:32](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tests_native_mainline/fixtures/authors/minimal_aiv.py#L32>) 实际解析为Scheduled HIR；[test_normal_entry.py:68](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tests_native_mainline/test_normal_entry.py#L68>)真实三步调用，:90–130检查PrimFunction/尾随ABI/实际For与VF及store输出对象，:149–186检查typed拒绝记录/重算gate/raising拒绝。focused初稿5 passed/exit0；加强版独立复跑-k 'not printer'为4 passed/1 deselected/exit0（不再触发写临时文件的roundtrip）；无生产代码修改。schema设计层已核对通过；稳定solver已独立复验，C1–C8已修复；本轮原生/计划增量结论见末节，仍不签完整S0。
- 完整S0的actual TL reader、真实安装与两组CPU数值仍需单独证据；入口拒绝测试通过只能证明S0第一段。

## 类型方案逐条合同核对（设计层，不是代码能力通过）

| 冻结字段/规则 | 可审稿证据与判断 |
|---|---|
| PipelineSpec/TaskIntent/BufferSchedule（合同:7–9） | [TYPED_SCHEMAS.md:52](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/handoff/TYPED_SCHEMAS.md#L52>)、:53、:55保留stage/V独立与显式Fixed/Auto、真实unit引用；未定义另一作者schema。实际代码消费者仍待求解落地。 |
| RegionBindingSummary/UnitTree（:10–11） | [TYPED_SCHEMAS.md:30](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/handoff/TYPED_SCHEMAS.md#L30>)–:41具备source loop、captures/carries/init/yields、actual For/IV/alloc refs及whole unit/Expr；正整常量step为明确支持域，未知具名拒绝，无名字重绑。 |
| NormalizedProblem（:12） | [TYPED_SCHEMAS.md:49](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/handoff/TYPED_SCHEMAS.md#L49>)–:64将tasks/access/effects/owner/resource/stage-order纳完整原生对象；core是显式数值投影，coverage从独立源遍历产生；不允许source-only sidecar。 |
| ResolvedPlan（:13） | [TYPED_SCHEMAS.md:70](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/handoff/TYPED_SCHEMAS.md#L70>)–:81只由SAT+整数postcheck+coverage构造，保留原stage/V与native region bounds、origin/max_shift；附加派生物不改变计划身份/版本。 |
| deadline/Assignment（:14/:23） | [TYPED_SCHEMAS.md:65](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/handoff/TYPED_SCHEMAS.md#L65>)–:66明确单一公开Assignment三态/赋值集、绝对monotonic参数共享与可行/最优分列；与core实际实现字段待协调。 |
| SymbolicCompletionPlan（:15/:25） | [TYPED_SCHEMAS.md:85](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/handoff/TYPED_SCHEMAS.md#L85>)–:95以同一CompletionPlan类型承载有限sites/channels、显式effects、dense FREE和尾min(V,N)；channel V来自ScheduledBuffer，READY/FREE不另拆物理池。 |
| StoragePlan（:16） | [TYPED_SCHEMAS.md:99](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/handoff/TYPED_SCHEMAS.md#L99>)–:110沿现有PhysicalLayout/Requirement演进，version后分配、completion寿命、未知容量不可回填；ScratchPlacementEntry区分copies与V，字节不重复收费。 |
| NativeArtifact/事务（:17/:35–42） | [TYPED_SCHEMAS.md:118](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/handoff/TYPED_SCHEMAS.md#L118>)–:130具备实际TL/parsed/有序ABI/完整身份/双键配置；NativeVerdict固定经plan→emit→artifact传入reader，先完整conformance再实际install；不以typed外壳/hash或状态字认证。此前证据传递歧义已按review修订。 |

- 类型方案最终可审稿132行，SHA256=`f98ecf6f4a0cf487f0c3232d09b511fd8c7676a03291c44d92ddc3a3b5f17257`；本表结论为**设计层通过**，未授予实现资格。

## 求解核心已定稿结论（上一轮保留）

- C1–C8全部关闭：独立复跑自重叠→UNSAT、manual未标0、跨stage硬phase序→UNSAT、缺字段投影拒绝、自边distance下界、foreign映射拒绝/上轮缺容量UNKNOWN（本轮模型策略已调整，见末节）、phase全序及篡改拒绝、裸II域拒绝/显式域II3 SAT；原证据定位core:203/:294/:320/:382/:448/:463、adapter:40、solver:85/:100。
- 上轮完整活动套件51 passed in1.05s/exit0；入口第一段、受限数值求解、类型设计通过，完整S0不授资格。仍待三task+V穷举、耗时阶段共享deadline预算、Assignment来源身份/深不可变；本轮新增受限native数值接线见末节。
- 上轮probe前后三源码SHA256稳定：core=`ea900772a0e35b7dfdf6db4c25382ce4cc4176de52df49bec8e5239cb6e363c9`；adapters=`00d3cf30ac4198b042a9ae04b1064b1399742831aa4ea2187e90088f5bbcf3ba`；solver=`9bc55de0d285ba5d3861ff6c9a5a0ebb9ee53545e69e46708007b7c73511f01c`。
- 上轮测试SHA256：modulo=`390029a425cd01b45373c8fd86e14e4a5ecf0b7d7fe1a26ea27f3efa0395fe63`；region=`23c9705abb83b977f512e26507b8121921d410093d8cfdcc0bce8eeb6dae2826`；这些通过只绑定对应内容。

## 原生分析 / ResolvedPlan / CPU executor 增量（task-24）

- 最终增量结论：**R1–R3及追加反例全部关闭；已审受限S0通过**。真实作者normal-entry输出上显式调用units/problem→共用solver→resolve，并非主链compile_mainline已闭合；来源分析七项近似与完整S0资格仍分开。
- R1关闭：[resolved.py:102](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/resolved.py#L102>)独立重算；坏phase/II越域/资源超II/Fixed3改V1/Auto越域/同phase逆issue均分别具名拒绝；[resolved.py:240](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/resolved.py#L240>)按region赋值、:300局部rank。真实root SAT128、loop SAT64→同一ResolvedPlan保留II[128,64]、rank各[0,1]，同输入identity相同；缺region→PLAN_ASSIGNMENT_MISSING、任一UNKNOWN→PLAN_NOT_SAT，不合并不同II冒充一个赋值。
- R2关闭：[problem.py:522](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/problem.py#L522>)按实际unit对象is/region直接子项/恰一次归属；原删tasks、重复Task、replace同ID外来unit、放错region四类真实对象变异均coverage=False，分别命中UNMAPPED/DUPLICATE/FOREIGN/REGION_MISMATCH（:568/:574/:580）。
- R3关闭：[problem.py:349](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/problem.py#L349>)保留带真实buffer_ref的Fixed3，真实solver赋值仍3且resolver保留同V；GM ring→PROBLEM_BUFFER_SCHEDULE_UNSUPPORTED、未匹配→PROBLEM_BUFFER_SCHEDULE_UNMATCHED独立实测。字符串root解析既有限制不当作吞约束；[test_numeric.py:44](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tests_native_mainline/test_numeric.py#L44>)已改SAT硬断言，原skip关闭。
- 新模型边界：[problem.py:615](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/problem.py#L615>)将直线region域设为串行跨度，修正原II1必UNSAT；[numeric.py:42](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/plan/numeric.py#L42>)只是唯一core的数值投影，不另求解/另造计划。缺容量事实的空间不建数值容量约束，与旧C6 UNKNOWN策略不同；模型SAT不表示无限容量，rmem/carrier的storage能力保持未验证/UNRESOLVED，真实storage门NOT_RUN。
- executor复用通过：[executor.py:380](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tilefoundry_ascend/verify/executor.py#L380>)lookup既有schedule_eval、:407调用handler，:450标量也读registry；[native_reference.py:37](</home/s00653124/workspace/TileFoundry_950PR_handoff/targets/ascend/tests_native_mainline/backend_operators/native_reference.py#L37>)只转调run_entry，原ABI数值断言未放宽。真实作者两输入与独立5*x参考最大误差0；为原始lowered TIR执行，不是安装后/设备数值。
- 能力上限：[S0_UNITS_REPORT.md:7](</home/s00653124/workspace/TileFoundry_950PR_handoff/docs/handoff/S0_UNITS_REPORT.md#L7>)–:15七项近似已披露；采集complete不替代真实delay/distance/代次/窗口/别名/跨region对应证明。executor内存仅(space,address)整块、MeshScope一次执行；S0外需显式限域或补真实语义。未审的completion/apply/verify/install半稿、TL conformance/安装后双输入/设备/原40均NOT_RUN。
- 最终独立回归：指定解释器/三环境变量，10个已审测试文件 -k 排除printer/trip_count/unsupported_instruction的写tmp源变体 → **70 passed / 3 deselected in2.66s / exit0**；Lead全量79通过为转述，不冒充本reviewer全量复跑。中途一次phase正例因逆序fixture失败，最终按合法order修正fixture且保留identity差异断言后已复跑绿。
- 最终probe前后SHA256不变：problem=`813188186f5580f3ced8edad56e96a5401b086060a524ba18387f845ab09cc34`；resolved=`3eb7b53af58847bf5d9dfa8a4610a0d1f89ec8a4865263b26942450981f22f6d`；numeric=`8b4cdf168f0816190463861ef7833f8ebe3f02bb9f61b4b941d18f092622165e`；core=`5ec3b1227834beb3293e92e48ef11cdd995ffd97fa67eb25dfca091c365709c9`。
- 其它绑定：units=`09af8f62630b21104425d81cabd70432fb5dce63f059013eb3d766c9627f4c78`；executor=`64a9c057a8b1061ef03f70ccc3c39c2b33168ca7571dca74b0d1a6562465d16a`；test_numeric=`7c825ce8c3b258ca58ef4520bc5b76a4b2b2f4af9f0abbf2c5e2a727fe1e891d`；test_resolved=`2992134859f50185e9f2cf871ef3dbd7812b60aa958ba6eea3991f88075d3a30`；仅写本报告，无源码/Git写/设备调用。
