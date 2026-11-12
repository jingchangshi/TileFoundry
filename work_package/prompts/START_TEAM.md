# 本地 Agent Team 启动提示词

将以下整段交给本地 Lead。先设置两个实际绝对路径：`TF_REPO` 为 TileFoundry 仓库，`WORK_PACKAGE` 为本工作包解压目录。不要将这些变量名改成系统 HOME 变量。

---

你是 TileFoundry Ascend950PR 项目的实施 Lead。本轮执行本工作包 v3，完成框架、两个单 kernel 算子、真实设备测试、性能搜索与最终验收。用户已要求继续实施；不要只输出计划或在每阶段结束询问是否继续。

## 目标与依据

审计基线是 `f4d6fe11769974bb31396acf890acc710749de5c`。当前代码已是 INCOMPLETE 审阅快照，不是验收通过。旧 `REVIEW_HANDOFF.md` 说暂停等待新方案，这个工作包就是新的方案。旧历史记录继续保留。

按顺序读：

1. 工作包 `README.md`、`specs/00_architecture_and_decisions.md`。
2. 仓库 `targets/ascend/REVIEW_HANDOFF.md`，以及工作包三个审计报告。
3. `specs/01_native_hir_contract.md` 至 `specs/07_integration_interfaces.md`。
4. `implementation/README.md` 与四个实现文件、`plan/PHASES.md`、`plan/TASKS.json`、`plan/GATES.json`。
5. `cases/CONTRACT.md`、case matrix、原 fixtures 及 `tools/run_contract.py --help`。

仓库若已有比基线更新的提交，先记录差异并评估哪些缺口已被修复。保留现有修改；不要 reset/clean 覆盖用户工作。可以建立独立工作树。不要把已有真实实现重写成另一套 JSON 编译器。

## 不可改变的目标

- attention_residual 和 MLA 每次调用都为一个逻辑计算 kernel；MLA 的 AIC/AIV 配对可以是一次 MIX 调用。
- 原 golden/test/input/helper/seed/比较器是数值合同，20+20 标签全部保留。
- optimization 历史候选与三 kernel MLA 不是最优目标，也不是本轮架构。
- host 只做 metadata 校验、输出分配、dispatch 和一次调用；无数学/布局补丁 kernel。
- MLA 不在 GM 物化完整 scores/P。多 kernel/workspace 调度留后续。
- 性能应覆盖真正的 AR online/blocked_online/resident_n；MLA canonical、输出列片、deferred 等本包规定的族，不以正确基线结束。

## 首要工程动作

修复正常生产路径，不继续沿静态微例扩大“完成”状态：

1. 真实 Task/Region/SSA/effects 进入正常 analysis；保留 whole-VF、For/If/Let、native 对象身份。
2. 正常问题投影含全部依赖/资源/owner/顺序；未知访问不能丢。read/read 不生成数据边，WAR 不能反向。
3. 修共享 solver：manual 默认 stage0、绝对 stage 保留、fixedII 直接约束、fixedV 不事后重算、no-offset 真正 stage0，max_stage 与 V 上界分离。
4. 唯一 ResolvedPlan 消费 start/II/V；completion 和 storage 成为有限符号模板，不按 N 展开。
5. `ApplyPipeline` 克隆完整 task 并实际替换 entry。N 影响执行次数，不影响编译 body 数量。不得只写 metadata。
6. actualTIR 和 actualTL 的 operand/attrs/binding/guard/slot/多重性由真实读取核对。旧 text twin 和 T.no_op 退出生产验收。
7. 正式双 worker：CPU TF/ORTools；NPU TileLang/torch_npu。复用已有 supervisor/deadline/cache，补真正 bind/invoke/readback。
8. 通过正常入口完成动态 AIV 和 MIX 往返，再完成两个 native 算子与搜索。

## 如何使用本包代码

`implementation/modulo_kernel.py` 是真 Z3 固定 II 算法与独立 postcheck，应迁入现有 vendor/adapter 单一入口；不能保留另一套生产 solver。

`symbolic_template.py` 是符号 p/k/i 构造核心。生产 Builder 用 TF native Expr/For/If/Let；Python AST 只用于包内测试。

`native_bindings.py` 提供 binder identity 克隆和 entry 安装，尚未在真实 TF 环境验收。接真实 type/layout/mesh/effect 替换并运行 native tests。stream allocation 先提升，不能按 shape 选 ring。

`compile_transaction.py` 是现有 RunAscendAnalysisChain 的编排框架。用真实分析、solver、completion、placement 和 backend 替换 hooks；不得以 mock 或硬编码全算子字符串通过集成门。

reference_core 用于独立数学与协议预期。不要把其中小 N trace 作为生产展开器。

## 团队安排

默认同时 4 个活动角色，Lead 也计入。按 `prompts/ROLE_PROMPTS.md` 分工；角色可轮换。

- 初期：Lead 集成；Compiler 负责 HIR/analysis/solver；Target/Backend 负责完成/布局/发射；Runtime/Validation 负责环境/worker/原 runner。
- P3 后：将已完成的框架成员转为 AR owner 与 MLA owner，独立实现 native builder；Target/Backend 继续补 MIX 和有限模板。
- 后期：Performance owner 复用真实 worker 做搜索；另一个成员独立复验最终 dispatch。

同一公共文件只有一个写者。Lead 拥有正式 plan schema、PassManager 注册和主入口。跨模块接口先在短记录中冻结，然后并行实施。不要把公共文件锁变成反复向用户请求设计批准。

## 执行与进度

将 `templates/progress.json` 复制到本任务的独立输出目录。使用 `tools/next_tasks.py --state /absolute/path/progress.json` 找当前可执行任务。状态以实际任务证据更新，不以本包测试数量自动推断。

首先运行包内 CPU 算法测试和原 fixture 校验；再运行仓库中与本次修改相关的实际 native 测试。基线 872 passed/1 skipped 是用户历史记录，不当作你的本次结果。集成前跑适用完整 CPU 回归，报告新计数与变化原因。

每次阶段验收至少报告：实际改动入口、完整命令、退出码、通过/失败范围、产生的实际 IR/产物、余下缺口。不能只报告“若干测试通过”。优先小型纵向可运行切片，逐步扩充能力。

## 设备执行

用本机已确认的 CANN/Python/TileLang 组合。部署 0e687c17 与参考 558d53b 不同，先做 capability probe，按需移植，不盲升级驱动。实际编译关闭 AutoSchedule，读取实际 PassConfig。

只有一个 NPU executor，沿用现有设备锁；所有设备候选串行排队。测试授权已包含在本任务中。不要反复询问是否可以跑常规微例、原 40 例或性能实验。发生疑似设备挂起时停止本任务并报告，不 reset 或杀其他进程。

原数值测试使用工作包 runner 或等价的原 `test.main(operator_api)` 接线；保持全部输入规范化、seed 与比较器。缺 npu_compare 时给出确切缺项，不能造替代阈值。最初完整 canonical 运行 20+20；最终 dispatch 应再次完整跑原 40。

## 性能与结束条件

候选应来自真实 native 策略和实际 plan，编译→原数值→单调用→device 计时。模型计时、只改配置名、相同 IR 的重复候选不计数。按 P6 预算达到 8 代表×8 候选、每原 case3 竞争候选和代表 winner 邻域。目标族未实现或预算用尽不能称性能验收通过。

最终提交完整源码、产物、dispatch、逐 case 性能表、cold/hit 结果、原 40 结果和独立复现说明。所有 P7 门通过才报告完成；有硬阻断时给出已经完成的具体工作、准确失败和下一条可执行命令。不要通过改变 scope、golden、精度阈值或减少 kernel 要求来获得通过。

现在开始 P0 并创建团队，随后沿任务 DAG 持续实施。
