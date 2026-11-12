# DELAY-FACTS（task-32）— 每指令写可见 / 读完成 / 状态完成事实清单

- 只读调查（explorer-g1）。基线 git rev-parse HEAD = **f341e47**；**本报告全部 file:line 指向工作树未提交文件**（plan/problem.py、plan/units.py、plan/source_audit.py、plan/numeric.py、plan/resolved.py、completion/plan.py 等由 implementer 在写），后续改动需复核行号。实测 active 测试面 108 passed（-p no:cacheprovider）。

## 1. 现有事实面（语义与单位）

- **InstructionContract.completion**（targets/ascend/tilefoundry_ascend/ir/isa.py:33）取值 issue / issue+visible / issue+write_visible / issue+data_ready / issue+state_complete；isa.py:1-7 自述「contract, not a performance claim; latency/II are optional cost-model inputs and never stand in for a completion protocol」。=> 只是**效果种类**，不含任何数值延迟。种类经 plan/units.py:353-359 _COMPLETIONS 落到 Effect.completion（units.py:96-99 四态常量）；source_audit.py:77-81 用同表校验，未知种类具名拒绝（SOURCE_COMPLETION_UNSUPPORTED）。
- **latency_model / ii_model**（isa.py:35-36，Optional[int]）：**单位从未声明**。ii_model 无任何消费者（只进 problem.py:236 的 identity digest）。latency_model 消费链：资源占用 duration = max(1, int(lm or 0))（units.py:641）；整 VF 按 pipe 求和（units.py:699）；task.latency（problem.py:740,751）**只**用于 ii_max 串行界（problem.py:616，core.py:198-209 的 lower_bound_ii 亦只用占用和 distance>0 的边）。=> 三者都是**模型 tick**，不可当设备完成延迟。
- **PIPE_FACTS**（completion/model.py:166 起）：每 pipe 记 {completion, visible_at_issue, async, same_pipe_order, source}——树内**唯一带完成语义的表**。vector/scalar: issue、visible_at_issue=True、same_pipe_order=True；mte2/mte3: issue+visible、False、False；fix/cube: issue+write_visible。core_for_pipe 对未知 pipe 具名拒绝（:201-213）；same_pipe_fifo_order 仅同核同 pipe 且 S/V/M 为 True（:242-263）。全部 status=REFERENCE。
- **memory.py**：SPACE_FACTS（:77-110）容量/对齐，无时序；TRANSFERS（:119-141）只声明合法路径与发出 pipe，无时序。
- **MemoryEffect**（src/tilefoundry/ir/core/param_def.py:44-49）：只有 READ/WRITE 方向；effect=None 表示未声明而非无副作用（:57-60）。
- **register_schedule_eval / register_verify**（src/tilefoundry/evaluator/registry.py:8-19；ops.py:856-861 等）：HIR 的 SSA 值语义与静态校验（如 _eval_gm_to_ub 直接透传 ctx.args[0].data）——**语义/校验，非时序**。

## 2. S0 / independent_ub 六条指令逐条清单

fixture: targets/ascend/tests_native_mainline/fixtures/authors/independent_ub.py（gm_to_ub -> 2×vld -> pset -> vadd -> vsts -> 2×ub_to_gm）

| 指令 | 契约 completion (isa.py 行) | Effect.completion | pipe | latency_model | 已声明完成语义 | 缺什么 | 现值性质 |
|---|---|---|---|---|---|---|---|
| GmToUb | issue+visible (:199) | WRITE_VISIBLE | mte2(async) | 64 | 写晚于 issue 才可见 | 可见延迟数值、事件绑定 | 种类=可审阅声明；64=tick |
| UbToGm | issue+visible (:204) | WRITE_VISIBLE | mte3(async) | 64 | 同上；终写走 completion drain（completion/plan.py:912-920：WRITE_VISIBLE 且 gmem 且无读者 -> drain-produce） | 同上 | 同上 |
| Vld | **issue+data_ready (:83)** | READ_COMPLETE | **vector**（但 category=dma） | 4 | 全表唯一 READ_COMPLETE 声明 | 数据就绪数值；**与 PIPE_FACTS(vector, visible_at_issue=True) 冲突**：按 pipe 判「issue 即可见」会错误豁免 vld | 种类=声明；4=tick |
| Vsts | issue+write_visible (:192) | WRITE_VISIBLE | vector | 4 | 写可见跟随 issue | 同上类冲突 | 同上 |
| Pset | issue (:78) | ISSUE | vector | **None** | issue 即完成且 pipe 可排序 | 占用时长未声明，却被 max(1,·) 伪造成 1 tick（units.py:641） | 缺失被默认值填补（可疑） |
| Vadd | issue (:103) | ISSUE | vector | 3 | issue 完成 + 同 pipe FIFO | contract.operands 只列 (a,b) 不含 mask/dst；审计按 op schema 对照（source_audit.py:62-76），contract.operands 仅描述性 | 种类=声明；3=tick |

全部 status=REFERENCE_IMPLEMENTATION，**无一条 DEVICE_VALIDATED**。isa.py:211-217 NATIVE_BUILD 自证 credit「CPU-only … NOT established」，且引用版本 0.1.15+cuda.git8e2d3c82 与本环境部署 **0.1.15+cpu.git0e687c17** 不一致（实测 importlib.metadata）=> 引用漂移，native 侧数值声明不可依赖。

## 3. 三件事判定

- **指令总 latency**：链只到搜索界级别（problem.py:740 -> 616）；单位未声明 => 只能当 tick，禁止换算成边 delay。
- **资源占用 duration/offset**：ResourceUse(ResourceKey(owner,pipe,0), duration, 0)（units.py:642、701-704，offset 恒 0）-> core per_resource 下界（core.py:198-209）。可当占用声明（tick 级）。
- **数据边 delay**：Dependence.delay 与 Reuse.delay 现**恒 0**（problem.py:459、485，自述见 problem.py:13-16）-> 投影 Edge(src,dst,distance, latency=item.delay)（numeric.py:58）与 Reuse(buffer,last_reader,next_writer, delta=distance_offset, read_latency=item.delay)（numeric.py:89-92）-> core 约束 t_dst + d*II >= t_src + latency（core.py:313）、t_next_writer + (V+delta)*II >= t_last_reader + read_latency（core.py:323-326）。
  **关键判定：0 不是「未知」，是把最小间隔清零。** core 的默认值 Edge.latency=1（core.py:74-81）、Reuse.read_latency=1（:113-120）被显式 0 **覆盖**：数据边允许同 tick，版本复用约束直接失效。目前只有 source_audit.py:301 SOURCE_COMPLETION_DELAY_UNPROVEN 与 resolved.py:327-328 PLAN_DEPENDENCE_PROOF_UNAVAILABLE（在 audit 之后无条件拒绝，即便 SAT）挡住认证。**这是正确的 fail-closed，解锁时绝不能删。**
- **可直接作为边 delay 来源的组合（唯一）**：completion=ISSUE 且 visible_at_issue=True 且 same_pipe_order=True 且同核同 pipe 且同 region 程序序（vector/scalar）=> 该类边可证 delay=0（PROVEN_ZERO），无需事件。mte2/mte3/data_ready/write_visible 类边**现无任何数值来源**。

## 4. 最小可信方案（作者/后端声明，非测量）

1. 新建 typed 事实表（建议与 PIPE_FACTS 同风格，位置 ir/isa.py 或 completion/model.py）：按 (instruction, 边角色) 记 write_visible / read_complete / state_complete 三个**可空**数值 + source + status，沿用 storage/model.py:243-245 的 REFERENCE_IMPLEMENTATION / DEVICE_VALIDATED / UNRESOLVED 词汇。
2. Dependence.delay / Reuse.delay 改三态：PROVEN_ZERO（由 §3 组合条件证明）/ DECLARED(n)（来自新表，带 source+status）/ UNSOURCED（构建期具名拒绝，如 DEPENDENCE_DELAY_UNSOURCED）。
3. 投影不得把 UNSOURCED 填 0：numeric.py:58/89-92 直传 int(item.delay) 是语义破坏点（覆盖 core 默认 1）。改为传声明值或拒绝。
4. 修默认值伪造：latency_model=None 不许被 max(1,·) 变成 1 tick（units.py:641,699）；缺声明具名拒绝（如 UNIT_TREE_COST_UNSOURCED）。
5. 定规则解 vld/vsts 的 pipe-vs-instruction 冲突：**效果可见性只认 instruction completion；pipe 的 visible_at_issue 只授权同 pipe FIFO 排序**，不得用它豁免 vld/vsts（PIPE_FACTS:160-165 现有注释留下该误用空间）。
6. 禁止把 CPU 串行执行（evaluator / _eval_* / native_reference）或 latency_model tick 冒充设备完成延迟；设备测量仍属 validator 独占，当前 NOT_RUN。

## 5. 当前事实下可证明的受限域

- **可证**：单 region 直线（distance 恒 0 已被单执行域校验，source_audit.py:291-292）+ 同 owner/AIV + 生产者 completion=ISSUE 且与消费者同 vector/scalar pipe（FIFO + issue 即可见）=> pset->vadd(mask)、vcvt/vadds/vmul 类 ALU 链，delay=PROVEN_ZERO，不依赖事件。
- **必须留给 completion 事件**：gm_to_ub->vld（mte2 async）、vld->vadd（data_ready 无数值源）、vsts->ub_to_gm（write_visible）、全部 reuse/版本回绕边（read_latency 无源）、跨核与跨 region 边（按 architect 澄清：需显式边界/composite 合同或拒绝，不得填 depth/1/0）。
- **仍不可证**：latency_model -> delay 的任何换算、ii_model（无消费者无单位）、fix/cube 边（S0 未用；fix 同样 visible_at_issue=False 且无数值）。

结论：SOURCE_COMPLETION_DELAY_UNPROVEN 在现有事实面下是**正确**的拒绝；解锁需要 §4 的登记表与三态 delay，而不是放松该门或补默认值。建议负例测试：(a) delay=0 的 mte2->vector 边不得被认证；(b) UNSOURCED delay 经投影不得变成 0 或 1；(c) pset 缺 latency_model 不得产生 1 tick 占用。