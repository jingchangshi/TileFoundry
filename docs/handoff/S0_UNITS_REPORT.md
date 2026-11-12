# S0-units 报告（task-16）与偏差收紧阶段

实现：targets/ascend/tilefoundry_ascend/plan/units.py + plan/problem.py（schema 对齐 docs/handoff/TYPED_SCHEMAS.md §1/§2）；测试 targets/ascend/tests_native_mainline/test_unit_tree.py（6 项）。实测：tree roots=1、regions=2（函数根 + 1 For loop Region）、units=4 = 3 DMA（root 初始 GmToUb、loop 内 GmToUb、UB→GM）+ 1 whole-VF（5 条域内指令同一 unit）；coverage 8/17/8 complete；trip 4→8 后集合不变。Lead 已验收 task-16。

下列偏差均为 S0 范围接受项；每条标注**到哪个阶段必须收紧**，不得带进 S2：

1. binding_map 只含 IV/carried/yields；captures、carried_initials 留 source-side（lowering record 无 actual Expr 对应物）。→ S1 引入真实跨区域 capture/出入值流时必须补全（否则 region 级来源绑定无法证明）。
2. SourceUnit.yields=()（TIR ExecutionScope 无 result slot）。→ S2 apply/verify 需要 region 输出对应时，必须由真实写入点与末读者建立 yield 关系，不能继续用空。
3. AccessFact.relation 用整 tensor identity；generation=所在 loop IV；guard=native true。→ S2 起必须有真实 access 窗口与 generation（否则依赖/复用/V 约束不可信）；S1 先把非整窗访问变成具名拒绝而不是 identity 近似。
4. LogicalBuffer.generation_delta=0；slot_bytes=dtype×shape；无 NZ/fractal padding；terminal_readers=区域内极大读者。→ storage placement（S2 版本后放置）必须给真实 layout/padding/scratch 与完整末读者集合。
5. RegionProblem.ii_max=串行 unit latency 和（**上界，不是 profile 值**）。→ S6 前换成目标 cost profile 的有限搜索界；S2 求解可先用该上界但不得宣称最优。
6. Dependence delay/distance=0。→ S1/S2 有真实迭代距离/完成延迟后必须重算，不能默认 0；S2 的 V/资源约束依赖它。
7. alias 用 (space,address,shape)+binder identity 归一：lowered 非 GM 窗口是地址 recipe，同一窗口的两种拼写归一到同一 root Var。→ S2 storage/apply 必须用完整 alias/version 关系替换地址归一，并保留跨 root 复用证明。

其余 S0 限制（已写进模块 docstring）：Capacity 仅 smem UB（target 常量）；rmem/carrier 无来源容量省略；issue_after/manual_pipe_order 为空（S0 无同 pipe 兄弟）；auto II 的 ii_min=1、fixed_ii 保持 min=max；alias 参数本切片不消费。
