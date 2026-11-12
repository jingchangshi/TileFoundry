# 00 总体方案：让唯一计划成为实际执行程序

版本：v3，2026-11-12。审计基线：`jingchangshi/TileFoundry@f4d6fe11769974bb31396acf890acc710749de5c`。本包替代 v1/v2 的实施顺序和完成判定，保留原算子语义与单 kernel 目标。旧工作包不必另行取得；原始数值文件、必要数学模型、关键实现和启动提示均包含在本包。

## 0.1 结论

现有工程已经具备不少可复用组件，也打通了一个真实的静态 AIV 微例。下一轮的主要工作是修复生产编译链的表达和连接方式。当前正常入口没有完整消费作者策略、真实依赖和符号循环；独立的符号流水、协议模型和算子 TL 实现不能拼成已经验收的编译器。

本轮以一条主路径交付：**真实 Scheduled HIR → 原生分析与共享求解器 → 唯一 ResolvedPlan → 安装为实际入口的符号 TIR → 有限模板的显式 TileLang → 单逻辑 kernel → 原 20+20 例与设备性能搜索。**

v2 的三层设计方向保留，但应纠正实施组织上的不足：阶段门曾允许模块模型、文本检查和直接 TL 候选积累大量局部结果，而正常入口仍未消费这些能力。v3 把“实际入口改变了什么、实际执行了什么”写入每个集成门。框架代码和测试应直接迁入现有模块，不在项目旁再安装一个编译器。

## 0.2 已核实的进度

| 能力 | 当前真实状态 | v3 如何处理 |
|---|---|---|
| native TF Target、HIR/TIR、正常 analyze/finalize | 已存在；target 正规执行域主要为 AIV | 保留，增加 AIC/AIV 与目标存储事实 |
| PipelineSpec 从 HIR 传到 TIR | 声明能存活；不是完整策略兑现 | 增加版本化策略与 whole-task 来源映射 |
| shared vendor Z3、bridge、resolver | 真正调用；存在 fixed V、stage origin、partial stage、no-offset 不一致 | 原路径修复，不新增第二权威 solver |
| 符号 pipeline | 有公式和独立实现；正常入口未安装它，内部仍有 binder/shape/结构缺口 | 保留可用公式，重写 whole-task materializer |
| completion/storage | 有模型、反例和放置结构；仍按 N 展开，物理事实与发射不完全一致 | 迁为有限符号模板，版本确定后做正式分配 |
| native AIV emitter | 可输出实际 copy、算术与事件；受限于 4×64、f32、V1、串行微例 | 扩成有限、可组合的 native lowering registry |
| CANN 9.2 数值 | 用户仓库证据：product 与 square 两静态微例误差 0，禁缓存重编译复验通过 | 作为工具链起点，不外推完整算子/动态流水 |
| 完整 AR | 自建图和文本 TL 路线；存在 score/参数兑现缺口 | 从真实 native 构造器重接，保留正确指令经验 |
| 完整 MLA | 有直接 TL MixedKernel 候选；没有执行 TF 唯一计划 | 迁到同一正常入口，先 canonical，再高性能变体 |
| G61 | PARTIAL：CPU 数学/epoch 协议 | native state effects、MIX 真实实现仍需完成 |
| G69 | INCOMPLETE：生成/过滤/模拟计时，worker 边界已有价值 | 接实际编译、原比较器和设备计时 |
| G70–G72 | 未完成最终搜索、dispatch 和原 40 例验收 | v3 最终阶段明确交付 |

用户提供的 872 passed / 1 skipped 是集成前 CPU 回归记录。本次审阅没有复跑完整 TF 环境，也没有运行 NPU。本包另附实际执行的纯源码反例和框架算法测试，二者证明范围分别记录在 `evidence/VALIDATION_STATUS.md`。

来源：[当前交接](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/targets/ascend/REVIEW_HANDOFF.md)、[正常链](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/targets/ascend/tilefoundry_ascend/passes/p3_chain.py)、[CANN92 复验](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/targets/ascend/evidence/cann92_runtime_recheck_20261112/split_result.json)。

## 0.3 三层分别负责什么

| 层 | 给谁使用 | 应表达或完成 |
|---|---|---|
| Scheduled HIR | 写策略的 Agent | 算法、tile、task、stage、执行域、布局/版本约束、指令与精度；保持数学语义 |
| Compiler 分析与转换 | 所有候选共用 | SSA/依赖、phase/II/自由版本、资源冲突、真实完成、物理存储、符号流水、来源核对 |
| 命令式 TIR | backend 与参考实现 Agent | 实际 For/If/Let、GM/片上视图、slot/generation、核心、指令、Set/Wait/barrier |

TileLang 位于 TIR 之后。它负责目标指令的后续 lowering 与 codegen。TF 已确定的调度不能再次交给 TL AutoSchedule 改写后仍声称执行同一计划。

Agent 可阅读 TIR 手写专家 DSL，但正式提交的手写实现应属于支持的模板表达域，并通过实际 TL PrimFunc 读取核对。先提供可用的 canonical emitter，使“Agent 手写任意程序的一般等价证明”不成为整个项目的前置任务。

## 0.4 四个调度维度

| 维度 | 作者声明 | 编译器补齐 | TIR 中的结果 |
|---|---|---|---|
| 时间 | pipeline region、stage、issue_after、II 约束、预取与递推边界 | 真依赖、phase、可行 II、重叠与完成协议 | `p/k/i` 循环、发射顺序、guard、等待与退出 |
| 空间 | AIC/AIV、group/sid、program grid、tile 与行列分片 | ownership、参与域、transfer 端点、无重叠输出 | 实际 core scopes、program/sid 索引、写入域 |
| 存储 | GM/UB/L1/L0/SS、layout、fixed/auto versions、驻留/复用意图 | 最后消费者、版本需求、padding/scratch、地址与容量 | 实际分配、版本 stride、slot 视图、FREE |
| 指令 | DMA、SimdVF/SimtVF、Cube、cast、reduce、FP32 状态及 P 精度 | 指令合法性、资源与状态效果、模板展开 | 真实目标操作及属性、精度和布局转换 |

四个维度相互约束。例如把一个 V buffer 从 V1 改成 V3，会改变预取空间、FREE、片上容量和可能的最佳 II。不能只改 JSON 字段，而保持 allocation 和 kernel 原样。

## 0.5 软流水，用一个例子解释

设三个任务分别是 `load(k)`、`compute(k)`、`store(k)`，作者给 stage 0、1、2。某个物理步 p 中，可以同时推进 `load(p)`、`compute(p−1)`、`store(p−2)`。编译器仍须确定同一步中各 pipe 的发射位置和顺序，并防止旧槽被过早覆盖。

模型时间为：

\[
t_u=s_u II+\phi_u,\quad 0\le\phi_u<II.
\]

`stage` 是跨逻辑迭代的偏移，`phase` 是同一模型周期内的位置。`II` 是成本模型的间隔，不能把它当作实测纳秒，也不应在 kernel 中生成“等待 II 周期”的代码。硬件正确性依赖真实事件、FIFO 和 barrier。

实际循环保留符号 N：

```python
N = max(0, ceildiv(stop - lower, step))
for p in range(N + max_shift):
    for task in resolved_physical_issue_order:   # 编译时有限task列表
        k = p - task.shift
        if 0 <= k < N:
            i = lower + k * step
            execute_whole_task(task, i, generation=k)
```

编译器只构造一次各 task 的 body。N=1、N=17、N=一百万均使用这个结构。N 变化可以影响运行时间，不应按 N 复制编译 IR。原 guard 和 SSA/state 仍在有效域内，具体实现见 02/03。

## 0.6 架构决定

1. **一个 native 源程序。** 算子参数配置只选择策略，不能代替 HIR。生产中不再使用 `attnres/hir.py` 的自建任务图作为另一套算子 IR。
2. **一个正式计划。** 演进 `plan/resolved_plan.py`；旧诊断结构从它派生。
3. **一个共享 solver 路径。** 新固定 II 约束核迁入现有 vendor/adapter。保留 CPU 独立数学检查，不保留两个生产求解答案。
4. **一种正常流水物化。** stage0/V1 也经过符号路径；static materialize 只作有上限的 debug oracle。
5. **两个正式进程。** CPU compiler worker 处理 TF/ORTools；runtime worker 处理 TileLang/torch_npu。利用现有成功证据隔离库交互；单进程 ABI 段错根因不是两算子的前置门。
6. **有限模板后端。** 先实现两个算子需要的 copy/VF/Cube/transfer/control，不追求任意 TF 或 TL 程序通用翻译。
7. **真实数据驱动性能。** 手算资源模型可排优先级；只有真实 native plan、物理 layout 或明确硬件合同可作硬淘汰依据。

## 0.7 保留、改造、退出生产资格

| 动作 | 模块 |
|---|---|
| 保留并扩展 | TF 原生类型/layout/SSA/visitor、PassManager、analysis、StoragePlan、指令注册、runtime 边界；共享 Z3；现有 worker deadline；原始数值合同 |
| 重点改造 | `p3_chain.py`、`unit_tree.py`、`ledger.py`、`resolved_plan.py`、`plan/pipeline.py`、completion/storage、native emitter、正规 Target facts |
| 保留作数学/协议参考 | `mla/rescale.py`、现有小 N trace、包内 reference_core |
| 退出正式验收路径 | 缺依赖的 production projection、按真实 N 展开、T.no_op text twin、子串 conformance、直接 TL 自动调度旁路、模拟计时的设备排名 |

历史证据应保持原样。新增明确的当前状态索引即可，不应把过去 NOT_RUN 改成曾经 PASS，也不应删除旧反例以获得测试绿灯。

## 0.8 单 kernel 和性能范围

两个算子都须通过原 20 例。MLA 的 AIC/AIV 配对可以是一次逻辑调用；profiler 的多硬件条目不能机械算成多 kernel。host 不做数学或布局补丁。不在 GM 物化完整 scores/P。

AR 应实现并比较 online、blocked_online、真正 resident_n；两遍重读可另列 blocked_reread。MLA 应交付 canonical，继续实现并测试 deferred rescale 与独立输出列分片。BQ/BC、列片、精度和驻留适用域由目标与原数值实验决定；某模板没写完不等于硬件不可能。

多 kernel 调度、跨 kernel workspace、依靠全 grid barrier 的 split-KV 主路线留待后续。输出列分片重复部分 QK/softmax，但各 program 写不同输出列，可以保持一个 kernel。是否更快由真实形状测量决定。

“极致性能”在本包中落实为：完成有意义的算法族与参数邻域，处理资源/尾部/递推瓶颈，在同一环境选择最快已测合法单 kernel，公开未探索项和停止原因。没有测量依据时不承诺绝对最优或固定倍数。

## 0.9 版本和复现

TF 审计基线固定为 f4d6fe1。此次核对的 TL main 为 `558d53b491925364a9b95a26cb5af66a7e7462fe`；用户运行证据中安装的 TL 为 `0e687c17bb3bd8a92bfeaa2bc8a5bdb37f6b3ca4`。它们不是同一版本。最新参考不能代替本机能力验证，也不要求盲目升级驱动或工具链。

新 TL main 相对旧参考 d0a76166，在所核调度主流程文件中主要增加 Z3 的必要 II 下界估计；这不会自动修复 TF 的投影、binder 或生产路由问题。P0 应锁定可用本机组合，再决定最小必要移植。

来源：[TL 调度流程](https://github.com/tile-ai/tilelang/blob/558d53b491925364a9b95a26cb5af66a7e7462fe/tilelang/ascend/pipeline.py)、[Task/Stage 语义](https://github.com/tile-ai/tilelang/blob/558d53b491925364a9b95a26cb5af66a7e7462fe/tilelang/ascend/language/schedule_hint.py)、[Z3 参考](https://github.com/tile-ai/tilelang/blob/558d53b491925364a9b95a26cb5af66a7e7462fe/tilelang/ascend/transform/z3_scheduler.py)。

## 0.10 最终交付

完整交付包括：可复现环境、正常入口源码、唯一计划、正式 HIR/TIR、实际 TL/后端代码、正确性/对应/单调用/资源结果、真实性能候选与最终 dispatch、原 40 例重跑、独立复验命令及已知限制。详细阶段门与文件所有权在 `plan/PHASES.md`、`plan/TASKS.json`；本地团队启动入口是 `prompts/START_TEAM.md`。
