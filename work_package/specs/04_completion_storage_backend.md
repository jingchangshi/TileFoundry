# Ascend950PR v3：后端、完成、存储与运行规格

状态：实施规格，供本地 Agent Team 在真实环境实现。基线为 TileFoundry `f4d6fe11769974bb31396acf890acc710749de5c`，TileLang 参考固定到 `558d53b491925364a9b95a26cb5af66a7e7462fe`。本次已核对该版本的 pipeline、InsertSync、copy 路径相对前次 d0a76166 无改动；部署证据的 TL 0e687c17 仍是另一个版本。本文中的新函数签名是实现目标，不表示当前仓库已有可用 API。当前实际缺口见工作包审计文档。

## 1. 目标与生产边界

本阶段将同一份 native Scheduled HIR 的解析计划安装为真实 TF TIR，并用该 TIR 产生或核对有限表达域的 TileLang expert 实现。编译和运行应复用正常 TF 入口、唯一 `plan/resolved_plan.py`、现有 target/ISA 注册、`search/process.py` 和 runtime 边界。

每个目标算子一次调用应执行一个逻辑计算 kernel。AIC 与两路 AIV 可组成一次 MIX 调用。host 应只做元数据校验、输出分配、specialization 选择和该调用。完整 scores/P 不应在 GM 物化；不得通过 host Torch 预处理补全 concat、transpose、cast、softmax 或 residual。

本阶段不要求一般程序等价证明，也不要求支持任意 TileLang 文本。先支持完成 AR 和 fused MLA 所需的有限 lowering templates。未知 op/control/layout 应明确返回 unsupported。模板应能继续注册扩充，不能将“整个 AR/MLA 手写 kernel 字符串”作为 compiler 的黑盒。

## 2. 正常编译流程

```python
def finish_ascend(module, analysis, target):
    units = analysis.unit_tree(module)            # 真正 TF IR 引用，含词法作用域
    problem = analysis.schedule_problem(units)   # 有限任务，不按 N 展开
    assignment = solve_and_verify(problem)
    plan = resolve_assignment_into_plan(problem, assignment)

    for attempt in range(MAX_REPLANS):
        completion = build_symbolic_completion(plan, units, target)
        placement = place_resolved_versions(plan, completion, target)
        if placement.needs_replan:
            plan = replan_with_constraints(problem, placement.constraints)
            continue
        entry = apply_symbolic_pipeline(module.entry_function(), plan)
        entry = insert_symbolic_completion(entry, completion, plan)
        entry = materialize_resolved_storage(entry, placement, plan)
        verify_actual_program(entry, units, plan, completion, placement)
        module = replace_actual_entry(module, entry)
        return freeze_export_input(module, plan, completion, placement)
    raise CompileError("REPLAN_BUDGET_EXHAUSTED")
```

此处各 helper 应接入已存在 pass，而非新增并行 pipeline driver。actual entry 替换应让 `module.entry_function()`、正式 printer、后续 emit 和 cache 都读取同一对象。metadata 只保留分析结果与诊断，不可成为实际循环体的替身。MAX_REPLANS 是有限编译预算，超限应明确失败，不能在最后一次失败后仍导出。

## 3. 一个 target completion 与 layout 合同

### 3.1 区分依据和未知能力

新增或归并 `TargetCompletionModel`、`PhysicalLayoutModel` 进入现有 Target facts 接口。每项硬件事实应有 `value/source/status`，status 可取 `REFERENCE_IMPLEMENTATION`、`DEVICE_VALIDATED`、`UNRESOLVED`。只需小型 typed record，无需另建审计平台。

| 事实 | 已读依据 | 实施要求 |
|---|---|---|
| 同核 flag namespace 与上限 | 固定 TL 558d53b `insert_sync.cc` 按 core+hard-event 分域、limit8 | 核对本机固定 TL/CANN；actual emitter 和 allocator 应同源 |
| AIC/AIV cross-core flags | 同文件 limit16，Cube 端按 sid×16 寻址，mode4 | 需 MIX 往返小例；不能把 READY/FREE 误作独立物理池 |
| 同 pipe S/V/M 顺序 | 同文件 `HasImplicitSamePipeOrder` | 只在同 core、同 pipe、相应效果合同成立时消解同步 |
| UB/L1/L0 大小、alignment | TF 现有 local3510 record 为声明值 | 纳入本机目标配置并核编译资源；未核实值不得伪装测量值 |
| UB→L1 格式转换 | TL `copy_op.py` 支持，InsertNd2Nz 执行 | layout 和 scratch 在 placement 前公开 |
| L0C→UB dual copy | TL 558d53b 要求源/目标同 dtype | f32 搬运后在 VF 转换，不能假定 FIX 自带任意 cast |
| SS 容量、可见性、可用指令 | 需要锁定本机 TL/CANN 对应实现 | 未确认时不启用依赖它的候选，保留具体缺失 capability |

不能把本地 runtime 使用的 TL `0e687c17` 当成 558d53b。P0 应记录并选定一个兼容版本；支持项不一致时要明确移植差异。

### 3.2 Owner 语义

本地存储的 owner 表示逻辑程序实例，不要求预知硬件调度到哪个物理 core。

```python
OwnerKey = (program_group, core_kind, sid)  # AIC 的 sid 为 None
StorageKey = (space, OwnerKey)
```

GM 为 device-visible。每个 AIV 的 UB 独立；AIC 的 L1/L0 属于该 program group 的 Cube。UB→L1 transfer 具有两个明确端点 owner；允许 transfer 写到关联 AIC L1，不等于允许 AIV 任意直接读写该空间。SS 若使用，应明确为关联 AIC/AIV group 的 scalar 共享资源，不能当成全 grid barrier。

### 3.3 物理事件 namespace

```python
LocalEventKey = (owner, producer_pipe, consumer_pipe)
CrossEventKey = (program_group, physical_bank)
```

逻辑 obligation kind 不进入物理资源 key。跨核 channel 若占用两个 bank，应同时在两个底层 bank 预留同一 id 范围；不能用 bank-mask=3 和 bank-mask=1 的两个独立字典掩盖交叠。flag ID 可按 `base + (generation % versions)` 发射，边界应由静态 V 和 base 证明。

## 4. 符号 completion 合同

### 4.1 输入

使用 `ResolvedPlan` 的真实 task、buffer、access、stage、phase、II、versions 与分析层的精确依赖。不得再次从“同 root 的所有 writers×readers”猜依赖。

每个 access 应关联真实 view/alias、读写区域、owner、logical generation Expr 和任务执行 guard。alias analysis 对无法证明不交叠的区域应保守建立边，不可猜测互不影响。第一版支持 affine 常数 distance；无法分析的动态依赖应报出具体 region/access，而非按 distance0 或 1 兜底。

### 4.2 有限模板数据

以下对象保存在现有 CompletionPlan 中，它们是计划的派生数据，不是新 IR：

```python
@dataclass(frozen=True)
class CompletionSite:
    task_ref: object               # 实际 unit/IR 引用
    anchor: str                    # before_task / after_task / region_exit
    task_ordinal: Expr             # 实际任务的逻辑代次
    generation: Expr               # 该通道保护的数据完整代次
    guard: Expr
    owner: OwnerKey
    pipe: str

@dataclass(frozen=True)
class CompletionChannel:
    buffer_ref: object
    reason: str                    # READY/FREE/STATE，仅逻辑原因
    producer: CompletionSite
    consumers: tuple[CompletionSite, ...]
    versions: int
    physical_mechanism: object      # local event / cross event / FIFO / barrier
```

编译中的 channel/site 数由有限 tasks 和消费者数量决定，不随 trip N 增长。小 N `CompletionTrace` 由模板解释器生成，只用于测试与 debug。

### 4.3 READY 与 FREE

对于写第 k 代的任务，guard 为 `valid(k)`；producer set READY 应在正确 pipe 的写完成点之后，consumer wait 应在读之前。多个消费者不能默认重复消费一个普通 flag；为不同消费 pipe/domain 分别建立通道，或使用有明确 capability 的组通知机制。

覆盖同 slot 的下一代是 k+V。FREE 集合应覆盖所有最大末读者：同 pipe、同 domain 且已证明完成偏序的读者可合并；不同 pipe 或不同 AIV 的末读不能按源代码最后一个元素代替。

v3 统一采用“前 V 代免等旧 FREE、每代末读者都发布 FREE、尾部消费剩余 FREE”的协议，与 `implementation/symbolic_template.py` 的 `free_wait_generation`、`final_free_generations` 保持一致：

```python
# 每个域内 generation 均执行控制 envelope；原语义 predicate 只保护计算。
if in_domain(k):
    if k >= V:
        for terminal in maximal_readers:
            wait(terminal.free_channel, generation=k-V, slot=k%V)
    if original_write_guard(k):
        issue_write(k)
    publish_ready_after_completion(k)

# 第 j 代的末读者。
if in_domain(j):
    wait_ready(j)
    if original_read_guard(j):
        issue_read(j)
    publish_free_after_read_completion(j)

# 尾部：每个末读者通道都消费尚未被下一代写者消费的 FREE。
# 这是保留为原生符号 For 的有界循环，最多 V 代，不按 N 展开。
for g in symbolic_range(max(0, N - V), N):
    for terminal in maximal_readers:
        wait(terminal.free_channel, generation=g, slot=g%V)

if N > 0:
    drain_other_outstanding_work_at_exit()
```

`generation=k-V` 与 writer 的 `task_ordinal=k` 不应混为同一字段。`slot` 可相等，generation 不同。N=0 时本 loop 不执行任务、Set/Wait 或 drain；尾部区间也为空。N<V 时尾部恰好消费 N 代，N>=V 时恰好消费 V 代。最终输出写和未由 FREE 覆盖的异步效果仍要 exit drain。尾部 Wait 应落在该 FREE 通道原有的接收 owner/pipe，不能随意放到 host 或其他 core。

“末 V 代不发布 FREE”和“每代发布、尾部消费”在各自完整配对时都可以合法；v3 只实现后者，代码、spec、测试与 emitter 不得混用。这里 g 是 producer 域的归一化代次；值若写为 S[k+1]，记录应通过明确 generation delta 映射到真实值代次和 slot，不能漏掉初始化的 S[0] 或直接套 k%V。

上例采用保守的 balanced control envelope：即使某次原 If 不执行计算，各端仍对该域内 generation 完成一次空通知，并在末读位置发布 FREE。编译器先证明 `original_read_guard => original_write_guard`；不能用空 READY 掩盖真实缺写。Set/Wait 数不因不同分支失配；尾部统一消费最后的通知。后续只在 guard 等价及 token 平衡已证实时消去空通知。若整个 region 的激活条件不同，先将它作为有完整入口/退出合同的嵌套区域处理；不能直接套用上述同域模板。无法证明的跨域动态分支暂报 unsupported。

### 4.4 guard 与 SSA

Set 和对应 Wait 的执行域应匹配：读可能执行时，写及其发布应执行。loop pipeline 的计算 guard 是 `0 <= p-shift < N` 与原 guard 的 conjunction；上节 balanced control envelope 的通知 guard 仅取对应迭代域，计算 guard 不应错误地包住通知。因不同 task shift，两个同步端点不一定位于同一 p；它们的 generation Expr 应相同。

MLA 的 group predicate、alpha、rescale anchor、P 版本应按真实消费者保活。deferred 模板应表达旧 epoch 全部 PV（所有 output columns、hi/lo）完成→drain/rescale/clear→新 epoch PV。组内分支决定先统一发布；不能依靠“每核大概算出同样 predicate”。

### 4.5 消解与验证

`can_order_by_fifo` 只对精确 core+pipe+effect 判断；不能把 vector visible-at-issue 当成跨 pipe 豁免。没有合法 event 的同 pipe DMA/FIX 冲突使用 target 支持的 pipe barrier。

symbolic verifier 应检查有限 channel 的 guard implication、slot 公式、generation shift、producer/consumer core、physical namespace、ring overwrite 边与退出。可对 affine 关系使用现有符号工具；复杂 predicate 不在支持域时返回 unsupported。测试解释器枚举小 N 的实例执行独立 token/读写状态检查，不能只重跑生成器并比较自己的输出。

## 5. 版本之后的物理分配

### 5.1 物理布局对象

buffer 的内存需求应显式包含 logical shape、physical shape、dtype、ND/NZ/fractal/packed layout、alignment、owner、versions、version_stride、转换 scratch。`layout_of(space)` 只能用于明确默认，不可让 L1 强制 linear。

```python
def physical_requirement(buffer, lowering, target):
    layout = target.layouts.resolve(buffer.layout, buffer.dtype, buffer.shape)
    slot_bytes = align_up(layout.required_bytes, layout.alignment)
    return Requirement(owner=buffer.owner, space=buffer.space,
                       versions=buffer.resolved_versions,
                       slot_bytes=slot_bytes,
                       extra_scratch=lowering.required_scratch(layout))
```

Cube/dual-copy/ND→NZ 的临时需求应先由 lowering registry 报给此函数。不能在最后 codegen 中突然多分配一份 UB/L1。

### 5.2 复用策略

第一条正确主路可令不同 roots 的 slots 不重叠；它仍支持同 root V 版本的循环复用。跨 root 复用作为优化，需要 completion happens-before 的明确证明。首写和末读的“文本序号不交叠”不是异步完成证明。

版本数变化、布局变化或共享资源压力触发重新 completion/placement；最终 offsets 只能来自同一 plan iteration。空间不足时将具体 capacity/reuse 条件反馈给 solver 或候选选择，不能静默减少作者 fixed V、缩 tile 或覆盖策略。

### 5.3 缓存与验证

placement input key 应覆盖当前 plan、effects、completion templates、shape/dtype/layout、owner、target 容量、backend scratch/version。补全 `storage.verify` 对真实 bytes/layout 的比较；completion digest 应覆盖全部规范模板内容，不能只统计 records/counts。

验证时从当前输入重算所需 footprint，与 attached placement 比较；容量来自当前 target facts。重算需求不等于重新分配，也不会让错误 offsets 被 verifier 自动修好。

## 6. 有限模板 TileLang 发射与结构读取

### 6.1 模板注册

在 `codegen/native_aiv.py` 的真实发射基础上新增 registry，不继续扩展固定 256f32 条件：

```python
class LoweringTemplate(Protocol):
    def match(self, source_unit, target) -> Match: ...
    def requirements(self, match) -> LoweringRequirements: ...
    def emit(self, match, expr_env, tl_builder) -> EmittedNodes: ...
    def read_actual(self, tl_nodes, bindings) -> ActualTemplateFacts: ...
```

先实现这六类可组合模板：safe copy/view、whole VF element/reduction、Cube matmul、L0C/FIX copy、UB→L1 pack/copy、explicit synchronization/control。AR/MLA kernel 由这些模板组成；模板的计算 body 应在 native IR/registered intrinsic 定义中可展开审阅。

VF 的内部 vld/vadd/vmul/reduce 等仍可逐指令发射；solver 的 unit 是整个 VF。模板不允许自行增 stage、改变 V、插入独立 TileLang AutoSchedule 或编造全局 workspace。

### 6.2 Expr 与 binder

支持必要的 native Constant、Var、Call、TensorView/Slice、For、If、Let、carried/yield 与 execution/mesh scopes。Name 只用于输出可读代码，真实 binding key 是 Var/definition identity。作用域退出时恢复外层 binding。scalar index 使用 native Expr，不通过 `int()` 强制常量化。

输入 actual entry 已包含 `p` 循环、`k=p-shift`、guard、slot、全部 sync。发射器不得重新计算调度。对 layout/address 的 relocation 只能在已验证 placement 上执行。

solver 的 phase/II 决定可实现的发射偏序和重叠策略，不要求后端按估算成本精确等待硬件周期。Emitter 不应按 latency 插入猜测的延时。真实完成仍由事件/FIFO/barrier 保证；估算 II 与实测性能分别报告。

### 6.3 变换与发射两层核对

1. 在 source→scheduled TIR 阶段，逐 source statement 的语义 fingerprint 包含 operand/attrs/type/layout/control；按 solver 提供的 substitution 对实际 TIR 核对。这修复 Vmul 改线、常数/掩码变化和重复/遗漏。
2. 在 TF TIR→TL 阶段，读取实际解析后的 TL PrimFunc，对 registry 支持的模板执行结构识别。比较真实 operands、buffer views、guard、loop bound/step、slot、owner 与 sync endpoint。actual TL 不得仅凭自带 annotation 宣称对应。

可允许 alpha-renaming、等价 flag bijection、物理 base relocation、明确列举的模板展开。保持策略不同的两个数值等价 kernel 不能自动通过计划对应。超出 matcher 能力的手写 expert DSL 可回到 canonical emitter 或补一个有测试的模板；不把任意程序等价检查列为阶段门。

### 6.4 导出与 compile policy

`require_export_ready` 应读取 current actual entry 和对应 verified artifact identity，而非只检查 metadata 状态字符串。通过验证后冻结 entry/plan；导出前后检查修改。实际编译应从 artifact 读取 profile 的 PassConfig，并在真实 `tilelang.compile` 调用点记录全部配置。首个受支持 profile 使用 `{"tl.enable_auto_schedule": False, "tl.disable_shared_memory_reuse": True}`，具体剩余变换合同见下节。文本注释或局部变量不构成配置证据。当前微例 compile API 对单一字典做完全相等判断；接入新版 profile 时应改成校验该 profile 的完整预期配置，不能因此丢掉新增必需选项。

### 6.5 关闭 AutoSchedule 后的剩余 TL lowering

`auto_schedule=False` 只跳过 scheduling 分支，不能理解为后面的编译器不再改变程序。已核固定 TL 558d53b 的真实 pipeline：早期仍运行 `RewriteDualCopy`、`LegalizeSimdMerging`、layout inference、`NormalizeAscendFractalStorage`、`InsertNd2Nz`、`AscendInsertOOBPadding`；后期仍运行 version/layout rewrite、普通 VF/tile-op lowering、`InferBufferAliases`、`MergeUBAllocations`、`RewriteFlagToBuf` 和 shared ThreadSync。它们可能改变指令、地址、scratch、同步或布局，因此只有 parsed PrimFunc 对应还不够。

**首个可交付 profile 的具体选择：**

- 强制关闭 AutoSchedule。MIX emitter 明确生成 Cube/Vector scope 与 sid，而非依赖已经跳过的 AssignCore/ResolveCore。
- 使用实际存在的 `tl.disable_shared_memory_reuse=True`。固定版本中该选项仍允许把 UB allocations 合成一个 arena，但每个独立 allocation 使用专属区间，关闭按生命周期执行的二次复用。TF 已计划的合法 alias 应通过同一 arena 的明确 views 表达；不能期待 TL 从独立 allocations 重新猜出 TF 的 alias。
- 同核 flags 在已确认的 8 槽 hard-event domain 内预留，包含模板内部需求，避免 `RewriteFlagToBuf` 的溢出路径。首 profile 允许经过核对的同 namespace flag bijection；不接受未经登记的 get_buf/rls_buf mutex 替换。后期可增加具有真实资源与配对合同的 flag→mutex refinement，不将其误称为普通重编号。
- physical ND/NZ、padding、conversion scratch 与 dual-copy sid 分片均在 TF 的模板 requirements 中公开。后端若实际多出缓冲区、fill 或同步，应由登记的 refinement 解释；超出 requirements 返回具体 capacity/机制差异，触发有限 replan 或拒绝该候选。

这里的 `refinement` 指已登记的语义保持展开。它不允许更改作者 stage、phase 发射偏序、版本策略、core ownership 或精度政策。具体合同如下：

| 剩余变换 | 可支持的展开 | 核对内容 |
|---|---|---|
| dual copy | 同一逻辑 tile 按已声明 sid/axis 分片 | 端点 owner、每 sid 实际 view、guard、同 dtype 限制、合并覆盖 |
| SIMD merging / VF / tile-op lowering | 同一完整 VF 或已登记指令模板内的合法指令序列 | 算式输入输出、inactive lane 语义、precision、内部 effects；不要求降低前后 op 数量相等 |
| ND→NZ | 已登记 scatter/pack 与后续 copy | physical pitch/footprint、scratch、V 到 MTE3 内部完成、版本和原 guard |
| OOB padding | 对真实物理 padding 的 fill 与受界 copy | fill 写域与 producer 边、合法 GM 范围、真实数学 D 不变 |
| buffer/version layout 与 UB 合并 | 已知布局变换、地址重定位 | slot/generation、view 字节映射、alias 关系、最终峰值与目标容量 |
| flag rewrite | 首版仅合法 domain 内的双射重编号 | 所有 Set/Wait 端点、bank、generation、作用域和资源占用保持 |
| shared ThreadSync | 登记模板内部的必要线程同步 | 不越出 whole VF 改变跨 task 协议；实际插入与删除均可解释 |

**UB→L1 的关键实现例子。** 固定 TL `InsertNd2Nz::RewriteUBToL1` 可把一个 ND UB→L1 copy 展开成 SIMD scatter 与 MTE3 post-copy，并新增 `shared.dyn` scratch，采用该路径时元素数为 `(rows+1)*cols`。这一具体路径的 pass 与 codegen 不自行插入 V→MTE3 事件；AutoSchedule 关闭时 InsertSync 也被跳过。后期 AscendThreadSync 在 Ascend 目标只处理 SimtVF 内部线程同步，不能当作这条 SimdVF→MTE3 边的保证。

首版宜显式形成 `PACK_NZ` whole VF → TF 计划的 V→MTE3 事件 → 已 NZ UB→L1 post-copy。PACK_NZ 的 UB 物理 pitch、scratch、cast 精度和读写 effects 进入同一 plan；后续 copy 读取已 NZ 源，避免未经建模的隐式转换。若采用自动 ND→NZ 展开模板，应在展开后确认内部等待、scratch 与 guard 全部符合注册合同，缺失内部事件时不得只凭原 copy 两侧事件放行。

**只增加两个必要观察边界，不做通用等价证明。**

1. parser 产生的真实 TL PrimFunc：按支持的模板核对 native TIR 的算式、绑定、循环、owner、slot、guard 及显式事件。
2. 固定 TL pipeline 中相关 refinement 后的实际 IR，以及最终设备 IR/CCE/编译资源：读取真实 buffer footprint/alias 与事件/机制，核对已登记展开。普通 VF 降低为不同数量的机器指令不算漂移；未登记的跨 task 重排、flags 溢出、额外 scratch 或精度变化才需要拒绝。

本地实施应使用该固定 TL build 实际支持的 pass instrumentation/编译 hook 保留必要快照，不发明不存在的 API。审阅包只需一份当前 manifest 关联这些观察结果。此检查不试图证明任意手写 C++/汇编等价，目的在于关闭已知会改变调度和资源的有限变换边界。

## 7. 可用的分进程 runtime

### 7.1 当前选择

采用 CPU plan worker→NPU TL/runtime worker。它是正式运行路线，不是仅供 smoke 的例外。ORTools+torch_npu 单进程段错留给独立兼容性任务，不阻塞两算子实现。该选择依据仓库 CANN92 分进程成功及原 native wrapper 失败证据；未声称根因已经定位。

### 7.2 最小接口

```python
# CPU worker，加载 TF 与 ORTools。
def build_artifact(request_path, result_path):
    assert_no_torch_npu_import()
    program = checked_program(request.source, request.entry)
    finalized = finalize(analyze(program.module, program.entry), authored=program.authored)
    artifact = emit_unique_finalized_program(finalized)
    atomic_write_artifact(artifact, result_path)

# NPU worker，加载 torch_npu 与 TileLang；不调用 TF/ORTools。
def load_artifact(descriptor):
    check_source_hash_and_versions(descriptor)
    prim_func = import_and_parse_exact_source(descriptor.source_path)
    verify_actual_tl_templates(prim_func, descriptor.expected_facts)
    return tilelang.compile(prim_func, target="ascend", out_idx=None,
                            pass_configs=descriptor.pass_configs)
```

复用 `search/process.WorkerCommand` 的 fresh run_id/identity/stage、独立 result path、exec 新 session 和 deadline。不 fork 活跃 NPU runtime。CPU worker 设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`；runtime worker 检查未加载 ORTools。既有 `tilefoundry_ascend.__init__` 会 eager import compiler，运行时轻入口要调整为懒加载或隔离子包，不能仅口头说“不执行分析”。

跨进程不传 NPU tensor，不传完整 Python IR pickle，不让 runtime 从 schedule JSON 再规划。artifact descriptor 是已决结果的传输记录：源码/产物路径与 hash、ABI、specialization、固定 flags、target/编译版本。大型真实 IR 仅以正式 printer 产物供审阅。

### 7.3 bind 与 invoke 合同

`codegen/x_runtime.bind_kernel` 应验证 ordered ABI 中每个输入和 scalar：shape、dtype、strides、device、contiguous output 合同、input/output alias。输出按 ABI 分配，返回 `BoundKernel(callable, args, outputs, identity)`。

`invoke_kernel` 在正确 device/stream 下调用该 callable 一次，返回输出 Tensor。不得在 wrapper 中执行 cast、copy-to-device、transpose 或数学运算来修正 kernel。输入装载、oracle、比较、readback 在测试 harness 边界完成。

`synchronize_and_readback` 提供测试模式；性能模式用 device event/profiler 的时间区间，预热、分配和 compile 不进入 kernel 时间。原 runner 的对照精度阈值与 seed 不变。

### 7.4 cache 与身份

cache key 包含 source、ABI、precision、plan、target、TileLang、CANN、actual flags。冷编译和 cache hit 都返回相同 artifact identity。若 TL build 可导出 binary，则记录真实 binary；不能导出时明确能力与 `UNOBSERVED_BY_BUILD`，保留实际源码/CCE/load 身份。不得伪造 hash，也不得沿用当前“所有 OBSERVED binary 一律拒绝”的旧 build 特例。

### 7.5 现有 worker 框架的精确接法

现有 `search/process.py` 已足够承担监督进程、超时、日志和 fresh receipt；不复制它，不新写 `Popen/killpg` wrapper。新增工作仅为两个 worker 入口和轻运行时 adapter，并将它们接到现有 harness。

实施路径如下。这是本地 Agent 要实现的路径，当前基线尚无这些完整入口。

| 路径/接口 | 职责 | 依赖 |
|---|---|---|
| `targets/ascend/compile_worker.py` | parse request→正常 TF analyze/finalize→unique-plan artifact；写 descriptor 与正式 IR 审阅文件 | TF/ORTools，禁 torch_npu |
| `targets/ascend/runtime_worker.py` | `--action compile/numeric/launch/profile`；加载同一 descriptor，完成对应真实操作 | Torch NPU/TileLang，禁 TF/ORTools |
| `targets/ascend/tf_ascend_runtime/` | 纯运行时包，提供 artifact load、ABI、bind/invoke、实际 TL reader | 顶层无 TF import；可懒加载 Torch/TileLang |
| `codegen/x_runtime.py` | 将纯 ABI/运行时类型迁移后 re-export，保留原调用接口 | 编译侧桥，不复制两份 ABI 定义 |

runtime worker 应以**绝对脚本路径**启动，不能使用 `python -m tilefoundry_ascend.runtime_worker` 后触发现有 package `__init__` 的 compiler eager imports。纯运行时包不注册 Target，不导入分析层。最终 operator API 只加载已经构建的 specialization dispatch；正式 40 例不需要在 NPU 进程里临时分析 HIR。未知 specialization 应返回明确未构建信息，交由 CPU 构建入口处理。

接线顺序：

1. Lead 在载入任何 NPU runtime 之前构造 `Deadline`。对候选 HIR 调用一次 CPU `WorkerCommand.execute(stage="compile", ...)`，使用独立的 `plan.log/plan.result.json`。设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，禁止隐式加载 torch_npu。此次 settings 中的 `TF_SEARCH_IDENTITY_KEY` 使用构建请求的 source+strategy+target identity；该步骤产生真实 plan/TL source identity，因此它发生在构造最终 `CacheIdentity` 之前。不能在尚未产生计划时预填虚假的 plan digest。
2. Lead 根据返回的已决 artifact 构造完整 `CacheIdentity`。规划耗时单列；总 case deadline 不重置。
3. 复用 `search.harness.WorkerCandidateRun`，其 `compile/numeric/launch/profile` 四个 `WorkerCommand` 都指向 runtime worker，分别传对应 action。每阶段使用不同 log/result 路径，同一 artifact descriptor、同一 toolchain environment。
4. 现有 `run_candidate` 负责顺序执行与短路拒绝。compile action 做真实 TL compile/产物检查；numeric 使用原 runner；launch 做一次调用的观察；profile 做 warmup、device event/profiler 采样。
5. 四个阶段是不同进程，不能跨进程传 `compiled_kernel` Python 对象。使用 TL 已有 artifact/cache 接口；若该 TL build 只能从源码重新装载，各 worker 可用同一源码/flags 完成 compile 或 cache load，并在测量窗口前完成。每次校验身份，不能重求调度。所有编译/装载时间均不进入 kernel latency。

调用现有 harness 的接法如下；`commands` 是上述四个已构造的 `WorkerCommand`，不是 Python callback：

```python
from tilefoundry_ascend.search.harness import (
    WorkerCandidateRun, run_candidate, MEASUREMENT_SOURCE_DEVICE,
)

run = WorkerCandidateRun(
    compile=commands["compile"], numeric=commands["numeric"],
    launch=commands["launch"], profile=commands["profile"],
    measurement_source=MEASUREMENT_SOURCE_DEVICE,
)
result = run_candidate(run, identity, budget=budget, deadline=case_deadline)
```

此代码仅在 Lead 进程执行；runtime worker 不应导入 harness。NPU 单一执行者在设备阶段外持现有锁；worker 不应再次取得同一把排他锁导致自锁。CPU planning 可并行，设备阶段保持串行。

各 worker 开始、关键库加载后、退出前均检查 import 边界：CPU 拒绝 `torch_npu`；runtime 拒绝 `tilefoundry`、`tilefoundry_ascend` 与 `ortools`。检查 `sys.modules` 的完整包前缀，不只检查一个顶层别名。异常时返回真实失败；不能删除 `sys.modules` 条目来伪装隔离。

worker 结果应沿用当前 `WorkerCommand` 已有协议，原子写到环境给出的 `TF_SEARCH_RESULT_PATH`，回显环境给出的 run_id、stage、identity_key。compile/numeric/launch 返回 `failure: null|string`；profile 返回实际 `samples_us`、`discarded_warmups`、`measurement_source: DEVICE_TIMER` 和非空 `sync_note`。父进程当前已经校验这些字段，不再包一层新 receipt schema。原 40 例的逐例原始报告另存为已关联 artifact 的文件。

## 8. 原 runner 与单 launch 验收

### 8.1 数值

最终 dispatch 的公共函数签名保持原 golden/test 合同。runtime worker 应导入原 `test.py` 并调用其 `main(operator_api)`；不能手写一组“等价 20 例”替代。attention_residual 和 MLA 分别记录恰好 20 个权威标签、原 seed、输入 dtype/shape、比较器路径与阈值。缺比较器或缺 case 时状态是未完成。

优化目录候选不作为最优基线；3-kernel MLA 不纳入本阶段 dispatch。可以用其历史经验定位问题，不能继承其多 kernel 结构。

### 8.2 单逻辑 launch

在 operator wrapper 周围观测：一次 logical kernel callable 进入、无 host Torch 算术 dispatch、输入到输出期间无额外独立预处理/归约 kernel。NPU profiler 中 MIX AIC/AIV 配对条目归属同一次 runtime invocation。计数标准应预先在 MIX 微例上校准，不能只统计 profiler 行数。

### 8.3 门槛

| 阶段 | 应给出的实际结果 |
|---|---|
| B0 边界修复 | Vmul 改线/guard/slot/状态缺失等反例被真实出口拒绝；旧 text twin 无生产权限 |
| B1 runtime | 正式 CPU worker→runtime worker→binder→一次 invoke 跑静态 product；cold/hit 身份一致 |
| B2 symbolic AIV | N=0/1/2/V−1/V/V+1/17/大 N，nonzero lower/step>1；真实 entry、TL、CCE 保留循环，编译规模不随 N 线性增长 |
| B3 mixed | Cube→Vector→Cube 完整往返，sid 分片、dtype、guards、ring wrap 与退出正确；资源由当前 target 模型检查 |
| B4 两算子 | 正常入口单逻辑 kernel 完成原 20+20，plan 对应/资源/launch 分别通过 |
| B5 性能 | 真实设备候选搜索、正确性过滤、原 40case 最终 dispatch 重跑与独立复现 |

同阶段 CPU 和 NPU 门可以分开报告。缺设备结果不能伪装 PASS，但本次用户已要求本地继续真实环境实施，因此不要为常规测试反复请示。遵守现有设备唯一执行者与锁约定；疑似设备挂起应停止该任务，不自行 reset 或杀其他任务。

## 9. 任务所有权

- Completion owner：改现有 model/obligations/schedule/emit/verify，产出符号 channels、真实 namespace 和小 N 解释器；和 scheduler owner 共同冻结 generation/guard/anchor 接口。
- Storage owner：扩充正规 Target physical layout、versions 后分配、reuse completion 边、失效键；和 backend owner 共同列举 scratch。
- Backend owner：binder-aware visitor、有限 template registry、actual TL PrimFunc reader、freeze export 与 compile policy。
- Runtime owner：正式两个 worker、轻 runtime 入口、bind/invoke/readback、原 runner/launch/profiler。

每个 owner 应先把自己的功能接入一条可运行纵向切片，再扩大支持域。不得以“模块单测已绿”代替正常入口已消费，也不得为每个 gate 另写一个绕过框架的专用 kernel。

## 10. 固定源码依据

- [当前 TF handoff](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/targets/ascend/REVIEW_HANDOFF.md)
- [当前 native emitter](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/targets/ascend/tilefoundry_ascend/codegen/native_aiv.py)
- [当前 runtime 边界](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/targets/ascend/tilefoundry_ascend/codegen/x_runtime.py)
- [当前 completion 模型](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/targets/ascend/tilefoundry_ascend/completion/model.py)
- [当前 storage 验证](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/targets/ascend/tilefoundry_ascend/storage/verify.py)
- [当前 worker 执行基础](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/targets/ascend/tilefoundry_ascend/search/process.py)
- [CANN92 环境证据](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/targets/ascend/evidence/cann92_runtime_20261112/summary.json)
- [禁缓存运行复验](https://github.com/jingchangshi/TileFoundry/blob/f4d6fe11769974bb31396acf890acc710749de5c/targets/ascend/evidence/cann92_runtime_recheck_20261112/split_result.json)
- [TL 固定版本 InsertSync](https://github.com/tile-ai/tilelang/blob/558d53b491925364a9b95a26cb5af66a7e7462fe/src/ascend/transform/insert_sync.cc)
- [TL 固定版本 copy 路径](https://github.com/tile-ai/tilelang/blob/558d53b491925364a9b95a26cb5af66a7e7462fe/tilelang/ascend/language/copy_op.py)
- [TL 固定版本完整 pipeline](https://github.com/tile-ai/tilelang/blob/558d53b491925364a9b95a26cb5af66a7e7462fe/tilelang/ascend/pipeline.py)
- [TL 真实 PassConfig](https://github.com/tile-ai/tilelang/blob/558d53b491925364a9b95a26cb5af66a7e7462fe/tilelang/transform/pass_config.py)
- [TL ND→NZ 展开](https://github.com/tile-ai/tilelang/blob/558d53b491925364a9b95a26cb5af66a7e7462fe/src/ascend/transform/insert_nd2nz.cc)
- [TL 线程同步范围](https://github.com/tile-ai/tilelang/blob/558d53b491925364a9b95a26cb5af66a7e7462fe/src/ascend/transform/thread_storage_sync.cc)
