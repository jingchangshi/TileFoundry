# 当前状态交接（2026-11-13）

本文件是源码与测试状态交接，不是设备验证或阶段通过证明。用户要求整理并提交本任务工作区改动；lead 已核实当前位于本地 main，基线为 df8e777，正准备提交本文件和本任务源码改动。具体提交以 git log 为准；外来的 docs/mainline_validation 未跟踪材料不纳入提交。本交接基于整理时的源码与下述一次测试，其他成员并行修改可能改变结果。

## 架构与证据边界

正常作者入口是 checked_program → analyze(compute-cost, memory) → finalize。原生链将真实 TIR 转为 unit tree、problem、求解结果及 resolved plan；后续构建 completion、storage、apply、verify、TL emit/read、install，最终经 export gate。核心代码：

- 计划与身份：
  - `targets/ascend/tilefoundry_ascend/plan/units.py`、`problem.py`、`identity.py`、`numeric.py`、`resolved.py`、`source_audit.py`。
  - 身份编码当前版本为 2；problem/plan 对内容重算，不能以 dataclass 字段或两个已存 digest 相等作为权威。
- 事件和存储：`completion/` 构建和验证事件义务，`storage/` 处理放置与寄存器预算，`plan/apply.py` 将实际事件和存储窗口注入候选 TIR，`plan/verify.py` 对照源程序与候选。
- 产物与 gate：`codegen/` 发射、编译及读回 TL；`plan/mainline.py` 编排；`passes/p3_chain.py` 和 `passes/gate_layers.py` 判断导出资格。已知参考 drain 映射仍标注未证明，不能当设备语义证明。

## R41：历史发现与现在的源码

历史记录见 `docs/handoff/LEAD_PROBES_R5_R7.md` 的 R41：当时手工台架声称 ready=True，但 lead 的生产入口实测停在 storage；伪造满阶段 ChainRecord 而不跑流程时，chain 层也没有报缺项。那是历史时点，不能直接当作现在的源码状态。

当前 `plan/mainline.py:283` 的 `compile_mainline` 已写入 analyze、resolve、completion、storage、apply、verify、emit/compile 和 install 的调用，并将产物记录到 metadata；在错误时返回具名 REFUSED。代码在成功 install 后返回 record，注释称 export 由 gate 本身判断；函数并未在该处调用 export gate。它调用 `install_entry` 得到新 Module，但对外返回的是 record，随后也把产物挂到传入的原 module：下游实际使用哪个 module/entry 仍需端到端核实，不能仅凭代码或测试台架宣布生产 ready=True。

当前 `passes/gate_layers.py:75` 的 `chain_gaps` 已不读取 record.applied：检查 ResolvedPlan、CompletionPlan、StoragePlan、TL artifact 和候选 entry receipt，并做部分身份关联。此处部分关联仍比较对象上保存的 plan_identity；完整性及内容被篡改后是否被其他层重算拦住，需要真实链正例和针对性负例验证。`stage_report` 是报告而不是判定。R41 要求的“伪造满阶段但缺产物拒绝”和“产物齐但 record 少写不得制造绿灯”必须在同一真实链下复测。

`test_s0_gate_positive.py:554` 的 ready 正例手工组装链；若工具链不可发现，它断言命名阻塞后直接 return，因此裸环境中该测试通过不等于 ready=True 路径执行过。当前测试结果见下，未证明生产入口 ready。

**Reviewer 新发现，待高级模型审议（不作为生产 ready 证明）：**

- drain → `PIPE_ALL` 是尚未证明的参考映射；当前设计允许将未证明事项放入 `recorded_unproven`，而 reader/gate 仍可给 VERIFIED/ready。这样的资格是否足以放行，待审议，不代表设备语义已证明。
- `install_entry` 返回新的 Module；`compile_mainline` 仅返回 record，`passes/p3_chain.py:284-306` 返回原 module。候选是否最终成为下游执行的 entry 尚未闭合。
- storage gate 在 `passes/p3_chain.py:110-140` 仍将产物存储的 identity 与 record 存储的 identity 比较，尚未独立调用 storage 内容重算；不要将二者相等视为内容未变。
- native verify 的 receipt 证明其候选验证关系，不等同于“该候选已经 install 到实际下游 module”的 install receipt；当前 gate 从 verifier 登记表查候选 entry，是否足以证明实际安装待审。
- Reviewer 捕获并复现：`require_export_ready(Module(name="handoff_empty", functions=()), raising=False)` 抛出 `UnboundLocalError`；`p3_chain` 中 `entry` 未赋值。该路径待修复并复验，本交接未自行重跑此复现。
- 正常测试环境缺少 bisheng 时，`test_s0_gate_positive.py` 的 positive 测试在断言具名工具链阻塞后直接 `return`，不会执行 ready=True 的后续断言；测试显示通过不能作为正向链已经跑通的证据。

## 已有内容与未完成工作

- 真实静态事件正例与篡改检查已存在：apply 注入 FlagSet/FlagWait/Drain，verify 做对应核对。`event_key_of_site` 在 `plan/apply.py:642` 对非静态 generation 仍拒绝 `APPLY_EVENT_GENERATION_UNSOURCED`；循环事件需要可验证 control envelope，不能跳过拒绝或填 0。
- task-46 仍 in_progress。`problem.py` 已出现 carried self-edge 距离推导，既有 minimal_aiv 测试断言自环 d=1、非 PROVEN_ZERO；但这不等于完整跨迭代证明。**当前源码存在必须优先复核的矛盾**：`_carried_roots` 文档说循环外绑定的 root 才跨迭代存活，而 `problem.py:1184-1207` 实际将 `carried.get("inside")` 传给 `_dependences`。此前把该结果称为“循环外 root 已证明”并不可靠。需要真实内外绑定正反例与删除边的 SAT/UNSAT 差分证据；不得以当前单条自环断言替代。
- `source_audit.py:622` 对多 region 返回整体不完整证明，未逐 region 独立重读；跨 region carry 和循环域证明未完成。未接线的 `_multi_region_evidence` 半成品已删除，设计结论保留在共享 task-46 描述。
- 寄存器预算要同时有 target 声明与 lowering 占用报告；缺失时具名拒绝。bf16/fp16 作者程序和设备数值面仍需要独立证据，不能把结构支持写成硬件数值通过。
- 比较器登记物 `docs/handoff/COMPARATOR_THRESHOLD_REGISTRY.json` 是 v2 draft，44 个 case（40 权威加 4 补充）；`test_comparator_registry.py` 校验 source_key/source_file、阈值源表达式、hash 和否定声明。登记物不证明原例数值通过，运行时实际导入的 comparator 副本仍待设备运行记录核实。
- `__CCE_KernelArgSize` 在 `CompiledArtifactFacts.semantic_identity` 中，并与 binary hash 一起进入 payload；合成 facts 仅将 24 改为 32 时 semantic identity 改变。当前生产代码中未发现该 semantic identity 的消费方；方向性 ABI 检查只核最小尺寸及对齐，无法证明工具链应留多少隐藏参数空间。详见 `docs/handoff/CONTRACT_GATE_MATRIX.md` §2d；它不等于实测篡改二进制能绕过完整 gate。

## 本次验证与复现

2026-11-13 本轮在当前共享树运行一次全量命令，退出码 1：**300 passed，4 failed**，62.46 秒；验证者独立报告同样的退出码和数量，并对 372 个文件做内存编译，未发现语法错误（后者为验证者结果，本人未重跑）。失败：

1. `test_normal_entry.py::test_export_gate_names_every_missing_layer`：预期的 TRANSACTION_INCOMPLETE 与当前实际 chain gaps 不符。
2. `test_normal_entry.py::test_export_gate_top_level_refusal_shape_is_unchanged`：实际增加 PLAN/COMPLETION/STORAGE/APPLY/INSTALL_STAGE_UNEVIDENCED 等项，与旧断言不符。
3. `test_s0_gate_positive.py::test_every_real_fixture_reaches_a_named_stage_and_reports_it`：copy_only 实际 REFUSED，测试预期 APPLIED。
4. `test_s0_gate_positive.py::test_a_fabricated_record_cannot_claim_stages_the_artifacts_do_not_show`：实际 REFUSED，错误为 TL_EVENT_LOWERING_UNOBSERVED，测试预期 APPLIED。

这只是一次裸环境运行；不能据此把失败简单归因于环境、并行修改或测试过时，须分别复现。更早的 300 passed 属另一源码时点，不覆盖本次结果。可用以下命令重测，保留退出码与完整输出，不用管道吞掉 pytest 状态：

```sh
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend \
  /home/s00653124/workspace/envs/tfnpu-v2/bin/python -m pytest -q -p no:cacheprovider \
  targets/ascend/tests_native_mainline --tb=line
```

已有环境观察：source CANN 后 import tilelang 可能连带 import torch_npu；不要把未导入 torch_npu 当作普遍性质。是否访问计算设备及工具链是否可用，需要在各自运行中测量。

## 恢复时建议顺序

1. 先冻结共享树并核实 diff/各写域归属，重跑四个失败，区分实际回归、期望变化及环境阻塞；不得删除断言、skip/xfail 或伪造通过。
2. 核实 carried 内外绑定矛盾及 reuse 中同 task 情况，补真实正例、结构完整性与相同 II 的 SAT/UNSAT 差分。保留未能证明的具名缺口。
3. 在真实循环 region 上设计并证明 control envelope 下的事件 generation；再做每个 region 的独立重读及跨 region 关系，不新增第二套 event key 或身份 recipe。
4. 用生产 `compile_mainline` 实际入口和真实对象链重验 R41、install module/entry 归属及 gate；记录各阶段产物、拒绝码与 recorded_unproven。ABI 语义身份绑定须由相应写域负责。
5. 最后分别处理 fp16/bf16 结构与真实设备原例；未获取设备与比较器运行时证据前，不宣称原 40 数值通过。
