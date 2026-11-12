# 02 原生分析、依赖与符号流水

## 2.1 正常入口的次序

保持 `checked_program → analyze → finalize → emit` 及现有 PassManager。将 target passes 改为下面的职责顺序。名称是职责，不要求每项都新增一个文件或 Pass 类。

| 步骤 | 输入→输出 | 应检查 |
|---|---|---|
| NormalizeIntent / PreserveRegionBindings | Scheduled HIR→规范策略、原 SSA 摘要 | stage/II/V/精度/域合法；bounds 不复制 |
| 原 ConvertHIRToTIR 与目标合法化 | 原生 HIR、逻辑 memory 分析→目标合法 TIR 及 source map | whole-VF、carried/capture 来源不丢 |
| FormUnits / AnalyzeSchedule | native TIR→typed UnitTree、region problem | 每条效果有归属，真实依赖完整，未知不丢 |
| Solve / Resolve | 有限问题→assignment→唯一计划 | 原策略硬约束、coverage、II/V/start 全部消费 |
| Completion / PhysicalStorage | 计划→符号通道、owner/layout/offset | 真实最后读完成、namespace/容量合法 |
| ApplyPipeline | 原 native 入口+完整计划→新 native 入口 | 符号 loop、whole-task/binder、真实 slot 与 guard |
| Verify / Install / Emit | 实际新入口→冻结产物 | 源到 TIR、TIR 到 TL 结构对应、实际 PassConfig |

初始 memory analyze 负责逻辑对象、alias、可选指令与初始布局；它的旧物理偏移不能在 versions/packing 改变后直接复用。最终 StoragePlan 需更新到实际版本和 completion。

## 2.2 UnitTree 的必要字段

改造 `plan/unit_tree.py`，不要再在 production project 中按所有底层指令形成 task。

```python
SourceUnit(
    unit_id, node_ref, region_ref, source_order,
    whole_body, execution_owner, resources,
    original_guard, captured_values, carried_values,
    effects, intent,
)
Region(
    region_id, node_ref, direct_children,
    induction_var_ref, lower_expr, stop_expr, step_expr,
    binding_summary, activation_guard,
)
```

这些是 native 分析对象，`node_ref/Expr/Var` 使用项目现有类型。序列化视图可记录 path/IDs，不能反过来成为编译执行依据。

调度 unit 是完整 VF、DMA、Cube 或已注册内部合同的 composite。一个 VF 中的 vld/vmul/reduce/vst 仍供 effects/cost/ISA 合法性分析，但不被外层 Z3 拆成几十个独立任务。无 buffer 效果但有硬件配置、累加器状态、控制通知的操作也应归属 task。

嵌套 region 先独立分析。外层 serial program 分配不按 program 数复制 body。内层 D-panel 的小常量结构可以形成有内部合同的 Cube 模板；不把动态 KV loop 变成巨大 flatten graph。

## 2.3 依赖应从哪里来

每个 Access 至少持：实际逻辑 root、native view/offset、读写区域、owner、mode、原谓词、对应 value generation，以及 async 读/写完成语义。优先复用 TF `ParamDef.MemoryEffect`、access relation、buffer alias 与 SSA 设施。

| 来源关系 | 生成的方向 | 注意 |
|---|---|---|
| 写定义→读取同一值 | RAW，writer(k)→reader(k+d) | d 来自代次/地址关系 |
| 旧值读取→覆盖该物理对象 | WAR，reader→writer | 不能反转为 writer→reader |
| 前次写→后次覆盖 | WAW | 初值/clear_accum 也算写 |
| read/read | 无数据完成边 | 共用 pipe 的资源边仍可存在 |
| ring slot 重用 | last_reader(k)→next_writer(k+V) | 与 logical RAW 分开，V 可为求解变量 |
| scalar/config/accumulator | STATE | 包含 HF32、MMAD 方向、clear、epoch 等 |
| 外层初始化/输出 | region 边界关系 | 不能为适配 loop solver 随便填 distance1 |

地址关系先在相同 root/可能 alias 对象内判断。真实读取 `A[k]` 与 `A[k+1]` 不等于“同 root 就冲突”。未知区域可保守保序，必要时把它留在不重排 composite 中；若会影响优化安全且无法表达，应明确报不支持。当前生产投影“未知 offset 就省略”应删除。

依赖验证分两层：先检查所有 source task/effect/依赖均被问题覆盖或由明确 composite 合同承接；再检查求解赋值满足这些关系。单纯对已经漏掉依赖的字典做 postcheck 没有充分性。

## 2.4 Loop-carried 状态

原 `LoopRegion` carries 应在 lowering 前保留。对于状态 S，定义 `S(0)=init`，任务第 k 次产生 `S(k+1)`；consumer 的 value generation 不一定等于 task ordinal。

首次实现可以将 m/l/U 更新保持在同一个 VF task，并显式保留 distance1 及 state 读写效果。以后拆分时，对 alpha/P/U 等建立 stream：producer 的 generation、consumer 所读 generation、slot 表达式、初值和最后消费者应一致。不能将所有标量做成一个跨 step 共享的“当前值”。

寄存器只能在合法 VF 作用域内使用。跨 VF 状态使用目标允许的 UB/SS 等表示，新增 copy 和临时空间应进入 task/effects/cost/placement。不得通过隐式寄存器逃逸使 IR 看起来更快。

## 2.5 符号物化算法

对正整常量步长 h：

\[
N=\max(0,\lfloor(b-a+h-1)/h\rfloor),\quad
o=\min_u s_u,\quad shift_u=s_u-o,\quad H=\max_u shift_u.
\]

构造 `p∈[0,N+H)`。每个 task 仅创建一次：`k_u=p−shift_u`，`i_u=a+k_u*h`，外层 valid guard 为 `0≤k_u<N`。原 task 控制逻辑完整克隆到这个 guard 内。

`implementation/symbolic_template.py` 已给出这个构造算法，使用 Builder 协议直接构造原生节点，不定义另一种生产 IR。测试使用 Python 标准 AST 解释有限 trace；本地应接入 TF 原生表达式与 stmt 工厂。

不能复用当前 `plan/pipeline._bin` 的错误类型假设：索引使用本项目纯 dimension 算子；bool guard 需使用合法的纯 scalar 表达式。`tir.arith.Binary` 是带 destination 的 effect 指令，不应被冒充成两个参数的标量表达式。若现有 TIR 缺少 bool 纯表达式，应补最小注册算子及 typeinfer/evaluator/printer/backend 支持。

### whole-task 克隆

`implementation/native_bindings.py` 给出按 binder identity 克隆完整 Sequential/For/If/ExecutionScope/MeshScope/Evaluate 的实现框架。它保留词法作用域；不按 name 替换，不按 shape 找 ring。该文件在本次包内只做语法和源码 API 审查，真实 TF 集成测试是 P2 必做项。

`instantiate_whole_task` 应完成以下工作：

1. 按真实 allocation/root identity 绑定已分配的 ring 视图。
2. 将源 loop IV 替换为该 task 的 i，跨 taskstream 替换为该 generation 的 slot。
3. 在新作用域内 freshen 局部 binder，重写 type/layout 中的维度和视图。
4. 保留所有 arith/config/SSA 与嵌套控制，不从 access 列表重建 body。
5. 把 effectful Let 留在有效 guard 内；仅纯索引可在外构造。
6. 加入同一 plan 生成的 pre-wait/post-set 及 state 操作。

stream allocations 应先提升到其 owner region，再从 task 局部 body 移除。不能把 GM 参数、L0 寄存器或所有内存一律变为 SMEM ring。

## 2.6 安装和事务性编译

`implementation/compile_transaction.py` 给出现有链 Pass 的编排内核：正常分析→solve→resolve→completion→place→apply→verify→emit。它不产生第二计划。`implementation/native_bindings.install_entry` 实际替换 `module.functions` 中的入口，并核对 `module.entry_function()` 身份。

pass 不得修改原输入后再尝试 replan。每次失败保持原模块可用；只有完整验证的实际新入口才允许导出。缺资源时反馈具体约束，重复相同 cut 或超出有限次数应停止，不能把最后一次失败的部分计划导出。

analysis、solver、completion、allocation 与 TL 编译使用共同 deadline；每步只能取得剩余预算。原生 C++库无法合作中断时，由现有 fresh-process supervisor 执行硬超时。不要给每个子 region 重置一份完整超时。

## 2.7 独立检查与反例

必要测试包括：两个同名 Var、两个同 shape 不同 dtype/root、原谓词依赖合法输入、effectful Let、多个 nested loop、region init/exit、N=0/1/V−1/V/V+1、nonzero lower、step>1、符号 stop、stage 间距大于 N、buffer 固定 V、内存/配置无访问表项仍不丢。

将运行时 N 改到百万时，unit 数、solver 变量数、completion 模板数和 TIR 结构节点数应保持不变（除常数字面量外）；实际执行 trace 只对小 N 枚举。编译规模检查读取真实 IR 统计，不能只统计 metadata。

源码与详细位置见 `evidence/v3_schedule_audit.md`。关键依据：[UnitTree](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/targets/ascend/tilefoundry_ascend/plan/unit_tree.py)、[当前符号 pipeline](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/targets/ascend/tilefoundry_ascend/plan/pipeline.py)、[TF visitor](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/src/tilefoundry/ir/visitor.py)。
