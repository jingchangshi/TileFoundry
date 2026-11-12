# 03 共享求解器、phase 与唯一计划

## 3.1 先修已经复现的合同错误

| 反例 | f4d6fe1 实际行为 | v3 结果 |
|---|---|---|
| fixed V2 | core SAT 后重算为 V1，bridge 报 FIXED_VERSION_DRIFT | 返回 model 中的 V2 |
| stages 2/3 | core 减去最小 stage，返回 0/1 | 保留 2/3，lowering 单独计算 shift |
| 只标一个 stage | bridge 拒绝 partial，resolver 却定义缺省 0 | normalizer 统一 manual-default0 |
| no-offset、RAW 延迟 2 | bridge 可 SAT 于 start0/2、II1 | 求解时 stage 全 0，允许更大 II 或 UNSAT |
| fixed II | 正常接口没有直接固定该 II | 指定 II 只检查该值，不偷偷另搜 |
| production 投影遗漏 dependencies | SAT 仅证明漏项问题 | source 覆盖检查先拒绝遗漏 |

这些结果来自实际 CPU 纯源码探针，完整输入输出在 `evidence/v3_schedule_counterexamples.json`。它们不是 NPU 失败日志。

## 3.2 固定 II 的约束

本包 `implementation/modulo_kernel.py` 提供可执行 Z3 约束核与独立整数 postcheck，CPU 测试已运行。它应迁入现有 `src/tilefoundry/vendor/tilelang_z3` 的单一路径，由 TF/TL adapter 共用；不能让工程生产同时使用旧、新两个求解器。更新 vendor provenance 与共享入口测试。

对于一个有限 region，先选择候选整数 II，再求每 task 的 stage/phase 和 buffer 版本。固定 II 使主要时间关系成为整数线性表达式；显式有限边界和共同 deadline 限制搜索规模。

\[
t_u=s_u II+\phi_u,\ 0\le\phi_u<II,\ 0\le s_u\le max\_stage.
\]

作者 stage、fixed II、fixed V、owner、precision 和 issue_after 是硬约束。自动模式可固定 `min(stage)=0` 去掉平移自由度；manual 模式不能改写作者绝对值。

### 数据与状态依赖

\[
t_v+d\,II\ge t_u+L_{u\to v}.
\]

d 是同 region 的真实逻辑 distance；L 为相应写可见/读完成/状态完成的成本估计。实际硬件正确性仍由 completion 保证。循环 recurrence 与配置状态应参与这个模型。

### 版本复用

对 buffer b 的每个最大末读者 r、下一写者 w：

\[
t_w+(V_b+\delta)II\ge t_r+L_{read\_complete},\quad V_b+\delta\ge1.
\]

δ来自 task ordinal 与 value generation 的偏差。普通每迭代写一次的 ring 是δ=0；carried S[k+1]不能强行按 k 处理。fixed V 用等式；auto V 在每个 buffer 的合法[min,max]内求解。

### 资源占用

资源 key 应含 owner/core/pipe。两个不同 AIV 的 vector 任务不应因字符串都叫 vector 而互斥。同一资源上的周期占用使用环形区间约束，包含跨 II 边界的重叠。

对偏移 ou/ov、占用 qu/qv：

\[
\Delta=(t_v+o_v-t_u-o_u)\bmod II,\quad q_u\le\Delta,\ q_v\le II-\Delta.
\]

上述为 capacity1 模型。composite 可提供多个资源 reservation，但应有已验证内部时序合同。未知容量、多 lane 资源不能直接套单资源公式；先用真实资源映射或明确受限支持。

### 物理发射偏序

phase 排序后，对同 phase 的 hard issue edges 做拓扑线性扩展。manual 同 owner/pipe 源顺序也进入这个图。`issue_after` 只约束发射；不会产生读写完成关系。phase 相同且 stage 较大优先只是无硬边时的确定性选择，不能覆盖作者顺序。

### 容量

只对已明确 owner/space 的物理 slot 估算建约束：

\[
\sum_b V_b\,aligned\_slot\_bytes_b + reserved \le usable\_capacity.
\]

owner 不同的私有空间分开计费；未知容量不能填当前需求大小。正式 padding/layout/scratch 由后续 physical placement 复核。未经证明的跨 root 不相交不能据此取 max；同一底层内存也不能同时按两个别名重复计费。

## 3.3 搜索与状态

先计算必要下界，如资源发射需求与自递推边；它只是缩小搜索，不是性能预测。固定 II 直接求该 II。auto II 按有限候选递增检查；可用更好的已证明安全搜索算法，但不能依赖未经证明的单调性跳过手工 stage 的可行域。

首个可行 II 之后，可在同一约束内继续减少物理版本字节数；每次改善仍是一次求解。**禁止在 SAT 之后根据局部 reuse 边重算 V 并替换 model。** 保留已验证可行解，超时不丢掉它。

| 字段 | 含义 |
|---|---|
| feasibility=SAT | 存在通过全部约束与 source postcheck 的赋值 |
| feasibility=UNSAT | 声明的有限域内无解；不是硬件永远不可行 |
| feasibility=UNKNOWN | 超时或 solver 未决定；不能当 UNSAT |
| ii_optimal_within_bounds | 所有更小候选已被证明不可行且边界完整 |
| memory_optimal_for_ii | 在当前 II 内的版本目标已求到最优 |
| schedule_source | shared_solver / 经同约束验证的 fallback |

“模型内最优”不能写成设备最优。串行 fallback 也须保持作者 hard constraints；原 stage/II 不允许串行时，应明确无可用 fallback。初始 v3 可以不提供 fallback，只正确报告三态。

## 3.4 source postcheck

两种检查都要保留：一是 solver 输出的独立整数不等式检查；二是从真实 native unit/effects/intent 重核 coverage 和约束。前者不能发现投影漏边，后者不能替代求解值检查。

检查应发现：缺 task、缺 buffer、固定 V 变化、时间单位错误、stage/phase/start 不一致、错误 owner、原依赖方向变化、未知访问被丢、资源 pair 缺失、手工顺序被改、版本/容量/原精度不符。

## 3.5 ResolvedPlan 的增量结构

扩展现有 `plan/resolved_plan.py`，不要平行定义一个“v3 计划编译器”。正式字段如下：

```python
ResolvedPlan(
    source_identity, target_profile, normalized_intent,
    regions: tuple[RegionPlan, ...],
    task_to_native_source, logical_buffers, access_and_state_effects,
    completion_templates, physical_storage_plan,
    semantic_source_map,
    feasibility, optimality, model_version,
)
RegionPlan(
    native_region_ref, lower_expr, stop_expr, step_expr,
    ii, origin, max_shift, tasks,
    # task: authored_stage, resolved_stage, phase, start, issue_rank
    # buffer: fixed/auto domain, resolved_versions
)
```

进程内保留 native 引用；JSON 只导出可审阅视图。全 kernel 可有多个 region，各自 II/loop domain，不强迫所有 tasks 塞进一个全局循环模型。

完成/放置之前 plan 是未完成的中间结果。不要以 `VERIFIED` 字符串默认取得可运行资格。最终出口读取当前 entry 及其完整 plan 的语义 identity，确认使用了同一套 II/versions/completion/layout。

## 3.6 核心代码的范围

本包 modulo kernel 已实现：partial manual 默认 0、绝对 stage、no-offset、固定/自动 V、RAW/STATE/复用、owner 分域资源、环形冲突、issue 偏序、容量、共同 deadline、SAT/UNSAT/UNKNOWN 和独立整数校验。测试含与小域穷举的对照。

它没有实现：原生 effect/alias 抽取、目标成本校准、任意动态控制的关系推导、物理 event coloring、TL codegen 和 NPU 执行。这些明确由本包其余阶段完成。Task/Buffer dataclass 只是 solver 输入视图，不得用它们重新表示整个算子。

源码依据：[bridge](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/targets/ascend/tilefoundry_ascend/solver/bridge.py)、[共享 core](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/src/tilefoundry/vendor/tilelang_z3/core.py)、[resolver](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/targets/ascend/tilefoundry_ascend/plan/resolved_plan.py)。
