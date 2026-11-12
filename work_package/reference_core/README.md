# 可执行核心参考：单 kernel 数学与流水协议

这组代码可直接在 CPU 上运行。公共模块与 MLA 模型只依赖 Python 3.10+ 标准库；attention 数组模型额外依赖 NumPy。代码不调用 Torch、TileLang 或 NPU。代码用于提供可移植的算法核心和独立小型检查，不替代附件 golden / 原比较器。

## 运行

```bash
python3 <本目录>/test_reference_core.py
python3 <本目录>/test_rescale_epoch.py
python3 <本目录>/mla_online_reference.py
python3 <本目录>/attention_reference_core.py
```

前两项分别为 14 个公共模块测试和 6 个尺度代次协议测试；后两项为 MLA 与 attention 的算法诊断。实际数量以各脚本输出为准。它们使用不同的数值模型，不应把检查总数写成正式 NPU 测试数量。

## 1. 在线 softmax 的实际实现

`online_softmax.py` 提供不可变的 `(maximum, denominator, numerator)` 状态，以及 `update`、`update_rows`、`merge`。更新式为：

```text
m' = max(m, block_max)
alpha = exp(m - m')，空状态取 0
p_j = exp(score_j - m')，无效位置取 0
l' = alpha*l + sum_j(p_j)
u' = alpha*u + sum_j(p_j*V_j)
Y = u/l，整行全 mask 时输出 0
```

完全无效的块直接保留旧状态，避免 `-inf - -inf`。`merge` 合并的是未归一化的状态，不能用来平均各 partition 已归一化的输出。

函数 `attention_residual_row` 展示一个 token 的完整算法，沿 N 块循环，并以原 V 更新分子。`mla_rows` 展示单 KV head、多 query 行的 tiled QK → online softmax → PV 数学计算，不分配完整 scores/P 矩阵。两者均使用 Python 浮点，通常为 binary64；生产实现的 FP32 累加和概率低精度转换应单独按照原比较器验证。

`reference_ops.py` 补齐完整嵌套 list 入口：`attention_residual(v_stack,x,w,gamma,alpha,epsilon)` 接收 `[N,B,T,D]` 等形状；`mla_dense(q_nope,q_rope,k_nope,k_rope,v,numKVHeads,scaleValue,inputLayout,is_causal)` 保留原参数名，完成 BSND/BNSD、`hq//G` 的 GQA 映射和右下 causal。V 与 K_nope 按值比较，允许独立分配。该 CPU 检查不表示生产 host wrapper 应额外发射 Tensor 比较 kernel；生产方案使用已声明的输入合同。

在单 kernel MLA 中，AIV 计算 `alpha` 和块概率，AIC 产生块 PV。已有输出分子应先乘 `alpha`，再加当前 PV。代码中的一行更新不能被直接替换为“连续累加 PV 而不缩放旧分子”。

## 2. 调度解析和有界结构模板

`pipeline_reference.py` 从 solver 已返回的 start/II 计算 stage/phase。代码检查固定 stage、manual 列表缺省 stage 0、同 pipe 源序和 offset 关闭时的单 II 合同。

这里的 pipe 标识一个完整资源域，例如 `AIV:VECTOR`，用于小型单域模型。完整 core/pipe 区分和 issue_after 见上级 implementation/modulo_kernel.py 及 v3 正式计划合同。同 phase 的 manual 源序是硬约束，stage 优先级仅用于已就绪任务的 tie-break；不能因默认排序而误拒合法解。

`PipelineTemplate` 只保存 loop、固定数量的 task 和版本数。`render()` 打印符号伪代码，不生成 fake TileLang API。`trace()` 仅供小型具体输入的检查，会枚举实例；生产编译不能调用 trace 来展开运行时循环。

每个 task 的逻辑 ordinal 为 `p-(stage-min_stage)`；源 induction 为 `start+ordinal*step`；slot 为 `ordinal % versions`；generation 保留完整 ordinal。N=0、1、短循环和 step>1 的处理已写入代码。

这不是另一套生产 IR。移植位置是既有 `ResolveSchedule`、plan 字段和 `ApplySoftwarePipeline`。生产实现仍需 TF 的原始节点引用、SSA/Let/guard、真实访问关系、作者额外顺序和 target resources；此小模块不冒充完整依赖检查或求解器。

## 3. 真实完成与 buffer 槽复用

`completion_reference.py` 以不可变 Event 描述每个逻辑 generation 的 write_begin、READY、read_begin、read_done、FREE。状态机只在所有声明的消费者都完成后允许 FREE。FREE 后保留已退休的 generation，防止旧代次重放。

对 attention_residual，V 槽的消费者至少包括 score 和 weighted accumulator。测试故意在 score 完成后提前 FREE，状态机会拒绝。对 MLA，score/P/output tile 的不同消费者需使用对应的实际完成点，不能把“发射完”标成 read_done。

这只检查**已声明的逻辑完成事件序列**。它不证明设备真的实现了这些事件。硬件 flag namespace、同步可见性、跨 AIC/AIV 路由、条件参与者和无死锁仍由目标 compiler 和设备测试验证。

## 4. 本组测试实际覆盖什么

- 极端有限 logits；多个 KV tile 宽度；partition 状态合并。
- 全 mask 行、全 mask 块、被 mask 的 NaN 不被读取。
- attention 的 N 轴聚合、原 V 与真实 D。
- MLA 右下 causal、rope 仅计一次、块尾部与不同 KV tile。
- 完整算子的 batch/token 映射、两种布局、`Nkv=2` 的 GQA 和显式/default scale。
- stage/phase 顺序；三 stage 与两槽；符号循环、短循环和非零 lower/step。
- 旧 generation、无 producer 的 read、提前复用与退出时未完成存储。

这些测试没有运行原 40 例，也没有产生 NPU 性能结果。正式验收继续读取原 golden、输入和原比较器。

## 5. 更接近高性能实现的两个模型

`attention_reference_core.py` 用 NumPy 实现完整 N/B/T/D 算法，比较 literal_fp32 与 factored_gw 评分，统一表示 CN=1、分块 CN、CN=N。它还包含 R/CN/versions/D chunk 候选枚举。所有硬件容量、对齐、raw load 宽度、reserve 都由调用者提供；估计用于初筛，最终 backend 实际分配仍需检查。

`mla_online_reference.py` 用 Python float 实现 canonical online 与 L0C pending/UB prefix 对应的延迟 rescale 代数模型，并检查按输出列分片后的结果。这里的 L0C/UB 是状态职责说明，CPU 代码没有模拟真实硬件存储和异步执行。`merge_partial_states` 仅展示数学恒等式，不授权本轮引入 split-KV 跨 program 归并。

v3 已改用上级 `implementation/` 的框架实现。此目录中的 pipeline_reference 保留为小型说明与独立预期，不作为第二套生产 adapter。

## 6. 延迟 rescale 的异步边界模型

`rescale_epoch_reference.py` 检查旧尺度全部 PV 完成后才能 drain/rescale/clear，再激活新尺度。producer 身份包含 KV 块、输出列和 hi/lo 分量；predicate 与 factor 在最后一个元数据消费者结束前不能覆盖。不换 anchor 的行使用 factor=1。

`test_rescale_epoch.py` 包含旧 PV 迟到、缺最后一个列/分量、提前切尺度、提前覆盖 factor 等反例。两块 score 为 0 和 10、V 均为 1 时，非法迟到可产生接近 2 的结果；合法协议输出 1。

这个小模型针对已初始化的有限 anchor，输入是完整 producer 集合和调用方声明的完成记录。实际空行初始化、硬件完成点、跨核心可见性与 Set/Wait 实现仍由生产编译器和设备检查负责。
