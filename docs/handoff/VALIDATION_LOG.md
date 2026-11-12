# VALIDATION_LOG — 增量独立复核与 S0 CPU 数值检查（task-14）

- 执行者: teammate `flex-3-g1`（validator），只写本文件；不碰 src/targets/tests；无 git 写操作；不调用设备。
- 基准: `git rev-parse HEAD` = `f341e47529453bdbc8dd81f2c108d36628fb4b8b`（tag handoff-20261112=374de77+2）。当前 S0 增量以**未跟踪文件**落地（fixture 与 test_normal_entry.py），HEAD 未变。
- 边界: 本日志全程**未触碰设备**（npu-smi 亦未再查询）；数值 40 例 = **NOT_RUN**；任何 PASS 不外推为设备或阶段 PASS；evaluator 为 authored-HIR 语义 oracle，不代表 TIR/设备执行。
- 环境口径同 [VALIDATION_BASELINE.md](VALIDATION_BASELINE.md) §4（tfnpu-v2 venv、torch 2.12.0+cpu、no:cacheprovider）。

## 第 1 轮（2026-11-12 21:56–22:12）

### R1.1 基线与计数

| 时点 | 命令 | 结果 |
| --- | --- | --- |
| 21:58 前 | mainline pytest（不含 S0 入口测试） | 24 passed, 0.30s, RC=0 |
| 22:05 | mainline pytest（含新 `test_normal_entry.py`） | **29 passed**, 0.98s, RC=0 |
| 22:10 | 同上（`-rA` 复核） | **29 passed**, 0.92s, RC=0；-rA 摘要**无 skip/xfail** |

新增 5 例（全部 PASSED）: `test_normal_entry.py` 的 test_authored_pipeline_declaration_survives_parse / test_normal_entry_lowers_the_pipelined_aiv_kernel / test_printer_restores_the_authored_pipeline_surface / test_chain_stops_at_the_named_freeze_stage / test_export_gate_is_fail_closed_with_named_gaps。

### R1.2 独立产物复核（自写探针，只读；不经被测测试代码）

| 项 | 独立验证结果 | 证据 |
| --- | --- | --- |
| authored pipeline 是真实 IR 对象 | **是**: isinstance PipelineSpec，max_stage=2、max_buffer_versions=2、loop (0,4,1)、yield=1（loop-carried） | 探针读 `fixtures/authors/minimal_aiv.py`（`tf.pipelined(4, max_stage=2, max_buffer_versions=2)`），走 `checked_program` → 真实 LoopRegion.pipeline |
| finalize 产物是真实函数对象 | **是**: entry 为 `PrimFunction`，name=vsum，params=[x, out]（尾随输出） | 探针 `finalize(analyzed, authored=...)` 后 `module.entry_function()` |
| 输出数据流真实接线 | **是**: UbToGm store 的最后一参 **is** entry.params[-1]（out）——非仅 metadata 声明 | 探针遍历 TIR body，`stores[0].args[-1] is entry.params[-1`] = True |
| evaluator 数据流（两组+建模输入） | **是**: 3 组输入（arange64、rng seed 20261112、100−arange）经 normal-entry 语义 oracle，输出与**独立 numpy 参考**（out=((x+x)+x)+x)+x=5x）**位级一致**（max_abs_diff=0.0），输出非平凡（≠输入） | 探针 `tilefoundry.evaluator.evaluate(authored_entry, x)` vs 独立 numpy 参考；三组全过 |
| **已安装实际 TIR 上的两组输入数值** | **NOT_RUN**: `evaluate(PrimFunction,...)` → `AttributeError: 'PrimFunction' object has no attribute 'evaluation_target'`——TIR 层评估未接线 | 探针直测；与 export gate gap 一致 |
| stage0/V1 ResolvedPlan | **缺席**: 全库 `grep -rln ResolvedPlan src targets` 无命中 | 与 gate gap `(plan, PLAN_ARTIFACT_MISSING)` 断言一致（test_normal_entry.py:156-162） |
| TL 源码 / 真实解析 PrimFunc→TL | **缺席**: 无 TL lowering 产物 | gate gap `(tl, NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED)` |
| 安装后 entry 对象 / 入口同一性 | **缺席**: 未发现 install 流程 | gate gap `(entry, ENTRY_IDENTITY_MISMATCH)` |

### R1.3 异常扫描（任务第 5 条）

- 测试变绿但产物缺失: **未发现**（绿的 5 例断言的都是已存在的真实产物；缺失产物以 named gap fail-closed 拒绝导出）。
- 断言被放宽: **未发现**（test_normal_entry.py 仍断言 REFUSED/ready=False/完整 gap 集合；reviewer-g1 通知的加强点 `store.args[-1] is entry.params[-1]` 由本探针独立复核为真）。
- 出现 skip/xfail: **未发现**（-rA 全 PASSED）。

### R1.4 本轮结论

1. 入口段增量（S0 作者样例经唯一正常入口 checked_program→analyze→finalize）**独立复核通过**：真实 IR/函数对象/数据流接线 + evaluator 语义 oracle 三组输入与独立参考位级一致。
2. **完整 S0 尚未达成**：plan 工件、TL 导出验证、安装后 entry 同一性、以及“已安装实际 TIR 上的两组输入数值”四项均缺席且被 gate fail-closed 如实拒认——**不给阶段授信**。
3. FAIL-CLOSED 行为经独立复核成立（chain stopped_at=freeze、ready=False、gaps 完整）。

（后续轮次在此追加；本文件保持 ≤150 行。）

## 第 2 轮（2026-11-12 22:23–22:35）solver core + solve_region

### R2.0 增量与计数（防竞态协议）

- 增量文件: `M` src/tilefoundry/vendor/tilelang_z3/{core.py(469 行), adapters.py, __init__.py}、`M` targets/ascend/tilefoundry_ascend/solver/__init__.py、新增 targets/ascend/tests_native_mainline/{test_solver_modulo.py(14 例), test_solver_region.py(8 例)}；另观察到同窗新增 untracked `plan/problem.py`、`plan/units.py` 与 `test_unit_tree.py`(6 例)（非本轮指定清单，记录存在性）。HEAD 仍 f341e47，全部为未提交改动。
- 竞态备注: 22:24 首跑得 49 passed，但文件 mtime 显示 22:25:27–22:28:44 仍在编辑 → **作废**；以 /tmp marker 协议重跑: **57 passed in 1.82s, RC=0，运行期间仓库 .py 零改动**。
- 构成对账（--collect-only）: abi_numeric 7 + identity 5 + native_builders 8 + search_rank 4 = 24（基线，修正 R1.1 中 identity/builders 的口算误差）+ normal_entry 5 + solver_modulo 14 + solver_region 8 + unit_tree 6 = **57** ✓；无 skip/xfail。
- 供应商账目: core.py 当前 sha256 `ea900772a0e35b7dfdf6db4c25382ce4cc4176de52df49bec8e5239cb6e363c9`；`PROVENANCE.md` 绑定的 WBS 04-07/08 扩展值 `ec1c4343…` 已过期 → 文档一致性备注（按其规则 2 应在下一 provenance 事件刷新），非语义异常。

### R2.1 三条关键语义（全部用我自建问题独立验证，未抄测试文件）

| 语义 | 我的问题与结果 | 证据（file:line） |
| --- | --- | --- |
| fixed V 不被 SAT 后改写 | 问题 A: buf fixed=2（slot 8B, space ub/aiv, 容量 16B）+ reuse(R→W 环) → SAT II=2, **versions={buf:2} 精确**；我的独立 checker 无违例 | 等式约束 core.py:306-307；后检 FIXED_VERSIONS_CHANGED core.py:425-426；solve_region 先 verify 再返回 solver/__init__.py:101-103 |
| 作者 stage 绝对保留 | 问题 B: manual 模式, P authored stage=1, fixed_ii=2 → SAT **stages={P:1, Q:0}**, t_P=3=2·1+1, 边 Q→P(lat 2) 成立 | 等式 core.py:291-292（manual 未注任务钉 0 :294-296）；后检 AUTHORED_STAGE_CHANGED core.py:398-399；非 manual 拒 authored stage core.py:269-271 |
| 声明 II 域不被扩宽 | B: fixed_ii=2 → `candidates()==(2,)` 恰一元素；B1 负例: fixed_ii=1 → **UNSAT_WITHIN_DECLARED_II_BOUNDS**（ii=null, 不外搜）；C 负例: 双任务 q=2+2 同管道 → lower_bound=4 > 声明 ii_max=2 → `candidates()==()` → UNSAT_WITHIN_DECLARED_II_BOUNDS | solver/__init__.py:57-69（域只收窄、越界即 UNSAT）；solve 只迭代 candidates :93-98 |

### R2.2 我自己对 SAT 赋值的逐条重算（纯 python，不复用求解器代码；core.verify_assignment 仅作交叉确认）

- **t = II·stage + phase**: A: t={R:0,W:1}=导出 starts ✓；B: t={P:3,Q:0} ✓（B 的 t_P=3 由 stage=1 贡献 II=2）。
- **数据边**: B 的 Q→P(distance 0, latency 2): t_P=3 ≥ t_Q+2=2 ✓。
- **资源环形合取**（同 (owner,core,pipe) 每对任务: q₁≤Δ 且 q₂≤II−Δ, Δ=(t₂−t₁) mod II）: A: Δ=1, q=1/1, II=2 → 1≤1 且 1≤1 ✓（紧致）；C4: q=2+2 同管道, II=4, SAT 于 Δ=2 恰好（2≤2 且 2≤2）✓——合取语义正确，II=1 双任务重叠情形被拒（core.py:323-331 的合取而非析取）。
- **reuse**: A: t_W+(V+Δᵣ)·II = 1+(2+0)·2 = 5 ≥ t_R+1 = 1 ✓。
- **容量**: A: Σ V·slot = 2·8 = 16 ≤ 16 ✓。
- 三题 `core.verify_assignment` 与我的检查器结论一致（均空违例）；`core_source_sha256` 可复算账目。
- **UNKNOWN ≠ UNSAT**: 过期 deadline → status=UNKNOWN, detail=SOLVER_DEADLINE_EXPIRED（solver/__init__.py:95-97），不可安装语义成立。

### R2.3 边界与异常

- export gate 仍 **ready=False / pipeline=REFUSED**（本轮 gaps 9 条: chain×3、completion、entry、plan、storage、tl×2——较 R1 新增 completion/storage/TL_ARTIFACT_NOT_ACTUAL named gaps），fail-closed 保持。
- 未触碰设备 = 是；**数值 40 例 = NOT_RUN**；solver 为 CPU planning-only，任何 SAT/最优化结论不外推为设备或阶段 PASS。
- 异常: **未发现"测试绿但语义不对"**（三条语义与逐条重算全部一致）；唯一备注见 R2.0 供应商账目（PROVENANCE 扩展哈希待刷新）。

## 第 3 轮（2026-11-12 23:03 后）：真实 plan、storage 与 TIR CPU 执行

基准 HEAD f341e47529453bdbc8dd81f2c108d36628fb4b8b，含并行团队未提交改动。完整活动套件使用 R1 同一 pytest 命令：**79 passed，3.06s，退出码 0**；开始前创建时间标记，结束时 src/targets 下无更新的 Python 文件；无 skip。相较 R2 的 57 增加 numeric 4、resolved 10、storage 6、normal_entry 2。

| 独立检查 | 实测及证据 |
| --- | --- |
| 实际 UnitTree → problem → numeric → solver → 唯一 ResolvedPlan | 从作者 fixture 经 checked_program、analyze、finalize 构造：4 个 unit、2 个 region，coverage.complete=True。root（region:0）声明 II=[128,128]，SAT II=128；loop（region:1）声明 II=[1,80]，SAT II=64；按 region_id 分别传入 assignment，得到同一个 ResolvedPlan，两个 region 的 II 分别为 128、64。投影见 plan/numeric.py:42、plan/numeric.py:86；覆盖按 unit 对象身份及 region 成员检查，见 plan/problem.py:498、plan/problem.py:521。 |
| 越界与伪造 | 独立复制 SAT 赋值后修改 phase=II → PLAN_PHASE_OUT_OF_RANGE；II=上界+1 → PLAN_II_OUT_OF_DOMAIN；将 root 问题一个 buffer 的域设 Fixed(1)，求得 SAT 后将该 buffer 版本改为 2 → PLAN_FIXED_VERSION_CHANGED。把一个 assignment 直接用于双 region → PLAN_ASSIGNMENT_PER_REGION（plan/resolved.py:102、plan/resolved.py:181、plan/resolved.py:240）。 |
| coverage 对象身份 | 在真实问题副本上删去一个 task、重复同一个 unit、追加一个 dataclass 复制出的外来 unit：独立调用 coverage，分别得到 complete=False，含 SOURCE_COVERAGE_UNITS_UNMAPPED、SOURCE_COVERAGE_TASK_DUPLICATE、SOURCE_COVERAGE_TASK_FOREIGN（plan/problem.py:521、plan/problem.py:557）。这些是 coverage 判定的拒绝证据，非通过 resolve 实际安装。 |
| storage：stride、非重叠、scratch 一次 | 自建两个 smem buffer：A=96 个 f32×3 版本、B=32 个 f32×2 版本，加一个 32 个 f32×3 copies 的 scratch。实际 A offset=0、stride=384、bytes=1152；B offset=1152、stride=128、bytes=256；scratch offset=1408、stride=128、bytes=384。独立算峰值 3×96×4 + 2×32×4 + 3×32×4 = **1792 B**，与实际 peak=1792 一致；相邻区间不重叠，scratch 按一项计入（storage/__init__.py:641）。 |
| storage 拒绝及重算 | verify_storage_current 返回 verified=True；将首项 offset 篡改为 64 后重验 → STORAGE_PLACEMENT_MISMATCH；100000 个 f32 的 smem 单版本 → STORAGE_CAPACITY_EXCEEDED；rmem 缺容量事实 → UNKNOWN_CAPACITY（storage/__init__.py:376、storage/__init__.py:719、storage/__init__.py:766）。 |
| 实际 PrimFunction CPU 执行 | 在 finalize 返回的实际 entry 上调用 run_entry；自建两组 64 元素 float32：arange(64)−31 和 (arange(64)%11−5)²。独立 numpy 参考从 x 开始再逐次加 x 四遍：两组逐元素完全一致、max_abs_diff=0，输入均未变。第一组输出前四 [-155,-150,-145,-140]，第二组 [125,80,45,20]（verify/executor.py:186）。 |
| 未注册指令拒绝 | 自建含 MyUnknownInstruction 的 PrimFunction 调用 run_entry → EXEC_INSTRUCTION_UNSUPPORTED（verify/executor.py:377）。 |

导出 gate 在同一个真实模块仍为 **ready=False / pipeline=REFUSED**，包含 plan、entry、completion、storage、TL 与 chain 缺口。**TL、安装后入口、设备和原 40 数值例均 NOT_RUN**；本轮 CPU 执行是实际 PrimFunction 的解释执行，不能外推为设备或阶段 PASS。R1 所记“PrimFunction 不可由旧 HIR evaluator 执行”仍描述旧 API；本轮新增 executor 已可执行 PrimFunction，取代当时的 TIR 执行缺口。未发现本轮“测试绿但语义不对”。

## R3 更正：数值计划语义缺口（2026-11-12 23:13 后）

R2/R3 自建例未覆盖非零 resource.offset 或 access ID 与 task ID 不一致；此前“未发现本轮测试绿但语义不对”仅表示当时样例结果，**撤回其作为整体正确性结论**。两项缺陷均已独立复现并单独通知 lead，修复后必须重新验证。

| 缺陷 | 独立反例与源码证据 |
| --- | --- |
| resource.offset 投影丢失 | 自建 II=2，两个同资源任务 duration=1；solver 返回 SAT，starts={a:0,b:1}，a 实际 offset=1、b offset=0，实际周期槽均为 1，**发生重叠**。plan/numeric.py:44-50 仅投影 duration；R2 零 offset 资源例不能证明非零 offset 正确。另独立运行 reviewer 的真实 root 修改例：II=128、两项 duration=64、第二项 offset=64，solve 和 resolve 均 SAT，starts={unit:0:0,unit:3:64}，实际 Δ=(64+64−0)%128=0，两个占用完全重叠。 |
| reuse 端点身份混淆 | 自建 II=2、reader start=1、writer start=0、V=1、delay=2：真实约束 0+1×2 ≥ 1+2 为假，但 resolver 返回接受。plan/problem.py:475-484 将端点记为 access_id；plan/resolved.py:163-170 以它查询任务 starts 并默认 0。此例直接调用 _verify_region，证明 resolver 层缺口；不声称该构造已通过完整真实 problem→solver 路径。 |

修复状态：**待修复、待独立复测**。上述发现不改变 R3 实测 79 passed 这一历史结果，也不授予 plan、设备或阶段 PASS。TL、安装后入口、设备及原 40 例仍 NOT_RUN。

## offset 修复独立复测（2026-11-12，task-29）

- 全量活动套件 **90 passed，退出码 0**；运行前标记后无 src/targets Python 文件变更。单独 solver_modulo 16 passed；以下四文件测试前后 sha256 一致：core.py fa67f01e2779363844e5aa6e1fb441125cf4e2e09bb2c1251e838036973487f6；numeric.py e2b14601832464842f8768cf85b38cce0dedbd2ee31f22406d0fa447a5fdeb51；resolved.py 560d760d799ce208acc963d9c2bef479c38f6ca087bc504cbacbd57d3330ebaf；test_solver_modulo.py 3f8193e51c80ec79ea9c81553ab58d55473a134731f3e11b726ffc5a691bc1ec。
- 自写独立穷举：II=2..5，两任务各 offset 遍历 [0, II)、duration 遍历 [1, II]，共 **978** 组；用 Python 集合逐周期展开实际占用，核对可满足性及 SAT 返回赋值，**0 不符**。范围仅限这两任务、同一资源模型。
- 先前真实 root 反例重跑：II=128、两个 duration=64、offset=0/64。求解 SAT，starts=0/0，实际 Δ=64，不重叠，resolve SAT；将旧错误赋值 starts=0/64 强制送 resolver，得到 **PLAN_RESOURCE_OVERLAP**。实际投影保留 offset（plan/numeric.py:47-50），约束和后验均计算 start+offset（core.py:334-340、plan/resolved.py:210-219）。该 offset 反例已修复，限于上述证据范围。
- reuse 缺陷独立跟踪：当前 resolver 已增加 access_id→task_id 查询（plan/resolved.py:165-180），但 numeric.py:69-70 仍直接将 reuse 端点投影给 core；**端到端修复及复测仍待完成**。未触碰设备，TL、安装后入口和原 40 例仍 NOT_RUN，不授阶段 PASS。

## offset 补充与 reuse 修复复测

- 新 numeric.py 已更新；三个文件在本探针前后 sha256 不变：core fa67f01e2779363844e5aa6e1fb441125cf4e2e09bb2c1251e838036973487f6、numeric a8342f8f2cfc19d31a2d81c5a8301d3de3ffc2cbe5fc6546454c4d6194650205、resolved 560d760d799ce208acc963d9c2bef479c38f6ca087bc504cbacbd57d3330ebaf。前一轮 numeric e2b146… 仅代表当时版本。
- 自建 5 类多 reservation × II=1..5 共 25 组，逐槽独立穷举并检查 SAT 实际赋值，包括同 task 相邻/重叠、多个 task、offset 大于 II：**0 不符**。
- reuse 修复局部独立复测：含 access:r→reader、access:w→writer 的 region 数值投影得到 core Reuse(last_reader=reader,next_writer=writer)；此前真实不等式 2≥3 为假的伪造赋值现报 **PLAN_REUSE_VIOLATED**；删除 writer access 则投影报 **PROJECTION_REUSE_ACCESS_UNKNOWN**（plan/numeric.py:70-91、plan/resolved.py:165-181）。以上为构造的 region 局部复测；实际工作负载的端到端 reuse 行为仍待后续验证。

## task-31 reuse 端到端独立复测（2026-11-12 23:23）

- 从真实作者 fixture 经 checked_program→analyze→finalize→UnitTree→NormalizedProblem 得到 root region；该 fixture 原本无 reuse，因此基于其真实 access:0/unit:0 与 access:15/unit:3 及真实 buffer:0，**人工加入**正反两条 reuse（delay=1、distance_offset=0），并非声称原程序已有这些边。numeric 投影保留两条并映射 unit:0→unit:3 与 unit:3→unit:0；solve_region 得 SAT、II=128、两者 start=0、V=1。独立重算两约束均为 128 ≥ 1；resolve_schedule 对该 region 得 SAT。
- 在同一问题上追加第三条 delay=10000 的 reuse，solver 返回 UNSAT，表明强约束不能被弱约束覆盖。沿用先前错误时序模式，将真实 region 副本设 reader start=64、writer start=0、II=128、V=1、delay=65，128 < 129，resolver 具名拒绝 **PLAN_REUSE_VIOLATED**。
- 删除端点 access：投影 **PROJECTION_REUSE_ACCESS_UNKNOWN**，resolver **PLAN_REUSE_ACCESS_UNKNOWN**；重复 access：分别 **PROJECTION_ACCESS_DUPLICATE**、**PLAN_ACCESS_DUPLICATE**。测试期间 problem/numeric/resolved/core 的 SHA-256 前后完全一致，分别为 813188186f5580f3ced8edad56e96a5401b086060a524ba18387f845ab09cc34、a8342f8f2cfc19d31a2d81c5a8301d3de3ffc2cbe5fc6546454c4d6194650205、560d760d799ce208acc963d9c2bef479c38f6ca087bc504cbacbd57d3330ebaf、fa67f01e2779363844e5aa6e1fb441125cf4e2e09bb2c1251e838036973487f6。
- 结论：已验证给定真实 region 上人工加入的多条 reuse 的投影、求解与 resolve；不代表 source coverage 对 RAW 方向或 effect 语义的完整性（该问题由 task-28 跟踪）。未使用设备；TL、安装入口、原 40 例仍 NOT_RUN。





