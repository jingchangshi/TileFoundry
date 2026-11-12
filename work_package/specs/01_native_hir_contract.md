# 01 原生 Scheduled HIR 与作者接口

本节是待实现接口合同。下文 `asc.*` 语法为 v3 目标写法，不表示 f4d6fe1 已有这些 API。P1 应把目标写法转成能被本项目 parser、evaluator、printer 和 lowering 接受的真实样例，并提交接口说明。

## 1.1 先保留现有基础

`LoopRegion` 已有 `induction_var/params/args/body/yield_values/start/extent/step`。`ExecutionRegion` 表达完整 VF。`tf.schedule` 选择已注册目标指令，其 `repeat/order` 表示指令重复维度遍历。本轮不改变这一含义。

源程序中的类型、shape、layout、storage、SSA 和 mesh 继续使用 TF 对象。不能在 JSON 中重新定义算子，再让 compiler 从 JSON 重建业务程序。

## 1.2 最小扩展

| 原生对象 | 扩展 | 说明 |
|---|---|---|
| LoopRegion / PipelineSpec | schema v2、enable_offset、max_stage、II 约束、auto 版本默认上界 | bounds 仍仅存原 LoopRegion，不重复一份 stop |
| 完整 ExecutionRegion 或单一 DMA/Cube Call | `TaskIntent` typed metadata | id、stage、issue_after、合法 cost override；不改执行域 |
| 多语句显式 Task 组 | 原生 `TaskRegion`，复用 region 的 params/args/body SSA 模式 | 仅当应把多个同 pipe 语句作为一个 task；透明数值语义，不伪造执行 mesh |
| buffer allocation | `BufferSchedule` typed metadata | fixed/auto 版本、min/max、owner/layout/驻留意图 |
| Target facts | AIC/AIV group、owner、空间、transfer、指令与 completion | 同一个正规 Target 查询路径 |

优先直接标注既有节点；只有多语句 task 分组需要新增 TaskRegion。新增原生节点应同时实现 parser、visitor、evaluator、printer、lowering 与验证。不能用一段字符串或只在 parser 之外可见的对象伪装原生节点。

### 策略数据的精确含义

```python
PipelineIntentV2(
    enable_offset=True,
    max_stage=8,              # 有限求解域的最大stage，非版本数
    ii=AutoII(lower=1, upper=explicit_budget),  # 或 FixedII(value)
    max_buffer_versions=4,   # 只作为未单独声明buffer的auto上界
)
TaskIntent(id="score", stage=2, issue_after=("load_next",))
BufferSchedule(versions=Fixed(2))
BufferSchedule(versions=Auto(minimum=1, maximum=4))
```

这些示例数值只是接口演示，不是 950PR 硬件默认能力。每个 target/cost profile 应给出实际编译搜索界。`max_stage`、`max_buffer_versions` 和 target flag 数相互独立。

### 手工 stage 规则

对一个 region 的 direct-child list，只要有一个 task/子 loop 带 stage，该列表进入 manual 模式；未标注 sibling 默认 stage0。嵌套列表独立选择 manual/auto。显式 stage 的绝对值不能被求解器减去最小值后丢失。物化时另计算 `shift=stage−min(stage)`。

`enable_offset=False` 表示所有 start 位于一个 II 窗口、所有 stage0。显式非零 stage 与它冲突，应在规范化阶段报错。不能让 solver 返回非零 stage，再在 resolver 才以另一合同拒绝。

相同 owner、core、pipe 的 manual tasks 保持源发射顺序。`issue_after` 是同物理步中有效实例的发射偏序，**不是** RAW、不是固定 distance，也不保证完成。真正 producer/consumer 完成关系由数据和状态效果分析产生。

### 旧接口迁移

现有 `PipelineSpec/1.num_stages` 不能静默改成新版的另一种含义。保留 legacy parser/roundtrip；Ascend normalizer 显式输出版本化的兼容映射和诊断。新生产样例全部使用 v2 字段。GPU 的既有 `tf.schedule(order=...)` 和已有 pipeline 行为应保持，通过小型目标回归核查。

## 1.3 时间之外的接口

空间策略应能指定 program grid、AIC/AIV 角色、sid、query/token/列片映射和 ownership。任务可能是动态 program_id 选择的一次实例，不要求作者指定物理核心编号。AIC 和 AIV 是同组的异构执行者，不应被描述成 AIV 下面嵌套 AIC 的线性层级。

存储策略应能表达 UB/L1/L0A/L0B/L0C/SS/GM、layout、alignment 要求、版本约束与驻留范围。GM ABI tensor 不应变成 UB ring。VREG/PREG 不能跨 VF 自由逃逸；合法跨 task 值要有显式 UB/SS 等传输表示及相应成本。

指令策略保留 `tf.schedule` 的具体目标操作选择。新增 RMS/reduce/exp/Cube/transfer 所需操作，应注册 memory effects、shape/layout 规则、execution scope、precision、状态效果与代价。精度不是只供报告的字符串：FP32 accumulation、P cast/hi-lo、HF32 模式、clear_accum 等应改变实际操作。

## 1.4 作者样例的要求

P1 应提交两份能正常 parse/print/evaluate 的原生样例：一个动态 AIV pipeline，一个带 AIC/AIV transfer 的有限 MIX region。样例不要求此时达到性能目标，但字段应能一路到 unit/plan，不得被投影为固定 f32/V1。

预期写法示意：

```python
# 提议的表面语法；P1产出真实可执行版本。
for k in tf.pipelined(stop_expr, policy=policy_v2):
    with asc.Stage(0), asc.Task("load", core="aiv", pipe="mte2"):
        tile = scheduled_copy(source_view(k), buffer_with_versions)
    with asc.Stage(1), T.SimdVF():
        result = complete_vector_body(tile, carried_state)
    with asc.Stage(2), asc.Task("store", core="aiv", pipe="mte3"):
        scheduled_store(result, destination_view(k))
```

Stage 应包住整个 task/VF；在 VF 内部给单条向量指令标 stage 应拒绝。用户仍可通过拆分 VF 并显式保存跨 VF 值表达不同策略。跨多个 pipe 的 composite 只允许具有已验证内部资源/effects 合同的模板，否则先展开为多个合法 tasks。

## 1.5 SSA 和来源信息

ConvertHIRToTIR 之前保留 `RegionBindingSummary`：原 region、captures、carried params、initial values、yields、原 guard 和值定义。summary 持真实对象引用。lowering 时补上它们对应的实际 TIR statement/Var/allocation。

原生符号身份使用对象/定义 identity。`Var.name` 仅供可读输出。两个叫 `tmp` 的 binder、两个同 shape buffer，不应因此被当作同一个值。序列化需要稳定编号时，以词法作用域和定义顺序建立编号；进程内不能从这个名字反查语义。

应区分三类跨 task 值：

| 值 | 处理 |
|---|---|
| 纯索引与不变 metadata | 允许在有效 task 域内重物化，按 binder identity 替换 |
| 含 load/随机/状态读取等效果的 Let | 保留唯一实际执行；不能当纯表达式复制或移到 guard 外 |
| 流水跨 task/跨代 tensor 或 scalar 状态 | 形成明确 value stream，版本/生成代次/初始化/最后 consumer 可分析 |

carried `S[k]→S[k+1]` 应保留 distance1。alpha、P、rescale predicate 等并非“最新值”全局变量；它们属于某个 KV generation 或 epoch。最初可将紧密相连的更新放在一个 whole VF 中，减少跨 task 状态；优化拆分时再增加真正的 stream。

## 1.6 支持域

本轮应支持：符号/动态正向 loop bounds、非零 lower、正整常量 step、外层 program/group loop、嵌套有限 region、原始 tail/causal guard、LoopRegion carries/captures、完整 VF 与合法 MIX transfers。

动态 step、一般 while、任意跨 region 非仿射依赖不是本轮硬性要求。遇到这些域，应保持可验证的原 serial 控制作为原生 composite，或明确拒绝该优化。不得填写 step1/distance1 或丢掉访问来假装支持。

来源：[LoopRegion](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/src/tilefoundry/ir/hir/loop_region.py)、[现有 PipelineSpec](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/src/tilefoundry/ir/hir/pipeline.py)、[tf.schedule](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/src/tilefoundry/ir/hir/schedule.py)。
